#include "azgo/coordinator.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <poll.h>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <utility>
#include "azgo/arena_slice.hpp"
#include "azgo/backend.hpp"
#include "azgo/build_info.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/cuda.hpp"
#include "azgo/game_queue.hpp"
#include "azgo/game.hpp"
#include "azgo/inference_hub.hpp"
#include "azgo/metal.hpp"
#include "azgo/parallel.hpp"
#include "azgo/run_state.hpp"
#include "azgo/selfplay_stream.hpp"
#include "azgo/serialize.hpp"

namespace azgo {

namespace {

const char* const kManifestHeader =
    "generation,first,count,planned,engine,fingerprint,build,seconds,positions,evaluations,note\n";
const char* const kGamesManifestHeader = "generation,game,engine\n";
const char* const kArenaManifestHeader =
    "generation,wave,first,count,planned,engine,fingerprint,build,seconds,positions,evaluations,note\n";

// A CSV field: quoted (with quotes doubled) when `always` or when it holds
// a comma, a quote or a line break.
std::string csvField(const std::string& v, bool always = false) {
    if (!always && v.find_first_of(",\"\n\r") == std::string::npos) return v;
    std::string out = "\"";
    for (char c : v) out += c == '"' ? std::string("\"\"") : std::string(1, c);
    return out + "\"";
}


constexpr double kCleanupTimeout = 10;    // cleanup after a cancel or a failure
constexpr double kTransferTimeout = 900;  // one scp of a network or a result

// [A-Za-z0-9_./~@:+-], not starting with '-': safe as one word of a remote
// shell command and never taken for an option.
bool safeWord(const std::string& w) {
    if (w.empty() || w[0] == '-') return false;
    for (char c : w)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
              c == '_' || c == '.' || c == '/' || c == '~' || c == '@' || c == ':' || c == '+' ||
              c == '-'))
            return false;
    return true;
}

// The same, but an option such as --job may start with '-'.
bool safeArg(const std::string& w) { return !w.empty() && safeWord(w[0] == '-' ? "x" + w : w); }

void requireSafe(bool ok, const std::string& what, const std::string& value) {
    if (!ok) throw std::logic_error("unsafe " + what + " for a remote command: '" + value + "'");
}

std::vector<std::string> sshBase(const MachineSettings& m) {
    requireSafe(safeWord(m.ssh), "ssh target", m.ssh);
    return {"ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=4", m.ssh};
}

// ssh <target> 'exec <words...>', every word checked.
std::vector<std::string> sshCommand(const MachineSettings& m,
                                    const std::vector<std::string>& words) {
    std::vector<std::string> argv = sshBase(m);
    std::string command = "exec";
    for (const std::string& w : words) {
        requireSafe(safeArg(w), "word", w);
        command += " " + w;
    }
    argv.push_back(command);
    return argv;
}

std::string parentOf(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? "." : path.substr(0, slash);
}

// The last line a failed child wrote to its log, for the error message.
std::string lastLine(const std::string& log) {
    std::ifstream in(log, std::ios::binary);
    std::string line, last;
    while (std::getline(in, line))
        if (!line.empty()) last = line;
    return last;
}

// What happened to a process, without paths or hosts: what a fallback's
// note in the manifest may say (the details stay in the slice's log).
std::string brief(const ProcessResult& r) {
    if (r.timedOut) return "timed out";
    if (r.cancelled) return "cancelled";
    if (!r.error.empty()) return "could not run";
    if (r.signal != 0) return "killed by signal " + std::to_string(r.signal);
    return "exit " + std::to_string(r.exitCode);
}

// A remote path's last part, for messages: a work directory can hold a
// user's home path.
std::string fileName(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

std::string describe(const ProcessResult& r) {
    if (!r.error.empty()) return r.error;
    if (r.timedOut) return "timed out";
    if (r.signal != 0) return "killed by signal " + std::to_string(r.signal);
    return "exit " + std::to_string(r.exitCode);
}

// A temporary name beside `path` for an upload in progress, unique to
// this upload: concurrent uploads never share one, even of the same file.
std::string partName(const std::string& path) {
    static std::atomic<unsigned long> counter{0};
    return path + ".part-" + std::to_string(getpid()) + "-" + std::to_string(++counter);
}

// "~/x" on this machine.
std::string expandHome(const std::string& path) {
    if (path.rfind("~/", 0) != 0) return path;
    const char* home = std::getenv("HOME");
    return (home ? std::string(home) : std::string()) + path.substr(1);
}


constexpr double kPreflightTimeout = 120;   // --version and --fingerprint on a machine

// The full build ID a `selfplay --version` wrote ("build <id>"), or "".
std::string buildLine(const std::string& log) {
    std::ifstream in(log, std::ios::binary);
    for (std::string line; std::getline(in, line);)
        if (line.rfind("build ", 0) == 0) return line.substr(6);
    return "";
}

// "09e008b" or "09e008b+1a2b3c4d" of a full build ID.
std::string shortOf(const std::string& id) {
    if (id.size() < 40) return id;
    return id.substr(0, 7) + id.substr(40);
}

// The engine a failed remote slice falls back to: the first local engine
// of its list, or the run's own backend.
std::string fallbackEngine(const std::vector<EngineSpec>& list, const std::string& backend) {
    for (const EngineSpec& e : list)
        if (e.machine == "local") return e.name();
    return "local/" + backend;
}

// A CSV row's fields; quoted fields may hold commas and doubled quotes.
std::vector<std::string> csvFields(const std::string& row) {
    std::vector<std::string> fields(1);
    bool quoted = false;
    for (size_t i = 0; i < row.size(); ++i) {
        const char c = row[i];
        if (quoted && c == '"' && i + 1 < row.size() && row[i + 1] == '"') {
            fields.back() += '"';
            ++i;
        } else if (c == '"') {
            quoted = !quoted;
        } else if (c == ',' && !quoted) {
            fields.emplace_back();
        } else {
            fields.back() += c;
        }
    }
    return fields;
}

// Each engine's fingerprint in a manifest's last row where it played: the
// self-play manifest's (11 fields, engine and fingerprint at 4 and 5) or the
// arena's (12 fields, at 5 and 6).
std::map<std::string, std::string> lastFingerprints(const std::string& path, size_t fields,
                                                    size_t engine) {
    std::map<std::string, std::string> out;
    std::ifstream in(path, std::ios::binary);
    std::string line;
    std::getline(in, line);   // the header
    while (std::getline(in, line)) {
        const std::vector<std::string> f = csvFields(line);
        if (f.size() == fields) out[f[engine]] = f[engine + 1];
    }
    return out;
}

// A network's file name in a remote work directory: by its hash, so a
// machine that has the bytes already is not sent them again.
std::string netFileName(uint64_t hash) {
    char name[48];
    std::snprintf(name, sizeof name, "net_%016llx.azn", static_cast<unsigned long long>(hash));
    return name;
}

// A phase's local work directory once its slices are merged: gone, or with
// failed attempts the logs they left (empty logs go).
void finishWorkDir(const std::string& workDir, bool failed) {
    std::error_code ec;
    if (!failed) {
        std::filesystem::remove_all(workDir, ec);
        return;
    }
    for (const auto& entry : std::filesystem::directory_iterator(workDir, ec))
        if (entry.is_regular_file() && entry.file_size() == 0) std::filesystem::remove(entry.path(), ec);
    if (std::filesystem::is_empty(workDir, ec)) std::filesystem::remove(workDir, ec);
}

// ---- probes (tests only) ----
std::mutex gProbeMutex;
std::string gRemoteBuild;
std::map<std::string, double> gDelays;
bool gInProcess = false;
struct ProbeFail {
    int times = 0;
    distributed_probe::Fail mode = distributed_probe::Fail::Exit;
};
std::map<std::string, ProbeFail> gFailures;
struct ProbeSession {
    int times = 0, afterGames = 0;
};
std::map<std::string, ProbeSession> gSessionFailures;
std::map<int, std::vector<std::string>> gForced;   // generation -> engine of each game

enum class ProbeFailure { None, Exit, Timeout, Corrupt };

// Takes one of the engine's injected failures, if any are left.
ProbeFailure probeFailure(const std::string& engine) {
    std::lock_guard<std::mutex> lock(gProbeMutex);
    auto it = gFailures.find(engine);
    if (it == gFailures.end() || it->second.times <= 0) return ProbeFailure::None;
    --it->second.times;
    switch (it->second.mode) {
    case distributed_probe::Fail::Exit: return ProbeFailure::Exit;
    case distributed_probe::Fail::Timeout: return ProbeFailure::Timeout;
    case distributed_probe::Fail::Corrupt: return ProbeFailure::Corrupt;
    }
    return ProbeFailure::None;
}

std::string probeRemoteBuild() {
    std::lock_guard<std::mutex> lock(gProbeMutex);
    return gRemoteBuild;
}
double probeDelay(const std::string& engine) {
    std::lock_guard<std::mutex> lock(gProbeMutex);
    auto it = gDelays.find(engine);
    return it == gDelays.end() ? 0.0 : it->second;
}
bool probeInProcess() {
    std::lock_guard<std::mutex> lock(gProbeMutex);
    return gInProcess;
}
// Whether this session of the engine is to fail once it has returned
// `games` games (each session asks once, when it starts: -1 = no).
int probeSessionFailure(const std::string& engine) {
    std::lock_guard<std::mutex> lock(gProbeMutex);
    auto it = gSessionFailures.find(engine);
    if (it == gSessionFailures.end() || it->second.times <= 0) return -1;
    --it->second.times;
    return it->second.afterGames;
}
std::vector<std::string> probeForced(int g) {
    std::lock_guard<std::mutex> lock(gProbeMutex);
    auto it = gForced.find(g);
    return it == gForced.end() ? std::vector<std::string>() : it->second;
}

} // namespace

Coordinator::Coordinator(const Config& cfg, const std::string& runDir, HostsFile hosts,
                         std::ostream* log)
    : cfg_(cfg), runDir_(runDir), runId_(runIdOf(runDir)), hosts_(std::move(hosts)), log_(log) {
    if (cfg_.selfplayEngines.empty() && cfg_.arenaEngines.empty())
        throw std::logic_error("Coordinator: selfplay_engines and arena_engines are empty (the default path)");
}

Coordinator::~Coordinator() {
    for (const auto& [alias, state] : machines_) {
        if (state->uploaded.empty()) continue;
        try {
            const MachineSettings m = hosts_.settings(alias);
            const std::unique_ptr<Transport> t = transportFor(m, "");
            t->setCommandTimeout(kCleanupTimeout);
            t->remove(std::vector<std::string>(state->uploaded.begin(), state->uploaded.end()));
            t->removeDirIfEmpty(remoteDir(m));   // the run's directory, now empty
        } catch (const std::exception&) {
        }
    }
}

std::unique_ptr<Transport> Coordinator::transportFor(const MachineSettings& m,
                                                     const std::string& log) const {
    if (m.ssh.empty() && probeInProcess()) return std::make_unique<InProcessTransport>(m);
    return makeTransport(m, log);
}

Coordinator::MachineState& Coordinator::machine(const std::string& alias) {
    return *machines_.at(alias);   // a lookup only: no slice thread inserts
}

std::string Coordinator::remoteDir(const MachineSettings& m) const {
    return m.workdir + "/" + runId_;
}

std::string Coordinator::keyFor(const std::string& alias) const {
    for (const EngineSpec& e : cfg_.selfplayEngines)
        if (e.machine == alias) return "selfplay_engines";
    return "arena_engines";
}

void Coordinator::unreachable(const std::string& alias, const std::string& why) {
    if (cfg_.selfplayFallback == "stop")
        throw ConfigError(keyFor(alias) + ": " + alias + " is unreachable (" + why +
                          "); with selfplay_fallback = stop the run cannot start");
    unreachable_.insert(alias);
    if (log_) {
        *log_ << "warning: " << alias << " unreachable (" << why << "); its slices fall back to ";
        // The engine of each list that lists the machine.
        std::string engines;
        for (const std::vector<EngineSpec>* list : {&cfg_.selfplayEngines, &cfg_.arenaEngines}) {
            bool listed = false;
            for (const EngineSpec& e : *list) listed = listed || e.machine == alias;
            const std::string fallback = fallbackEngine(*list, cfg_.backend);
            if (listed && engines.find(fallback) == std::string::npos)
                engines += (engines.empty() ? "" : " and ") + fallback;
        }
        *log_ << engines << "\n";
    }
}

void Coordinator::preflight() {
    // Every engine of both lists, self-play's first.
    std::vector<EngineSpec> all = cfg_.selfplayEngines;
    all.insert(all.end(), cfg_.arenaEngines.begin(), cfg_.arenaEngines.end());
    // Local engines: the backend runs here, and the GPU hubs fit.
    std::vector<std::string> remotes;   // aliases, in list order
    for (const EngineSpec& e : all) {
        if (e.machine != "local") {
            hosts_.settings(e.machine);   // ConfigError for an alias the hosts file lacks
            if (std::find(remotes.begin(), remotes.end(), e.machine) == remotes.end())
                remotes.push_back(e.machine);
            continue;
        }
        checkBackendAvailable(e.backend);
        if (!fingerprints_.count(e.name())) fingerprints_[e.name()] = engineFingerprint(e.backend);
    }
    // Self-play's engines run at once, each one hub with two slots; an
    // arena wave's engines too, each one one-slot hub per network of the
    // wave. The two phases never overlap.
    auto hubs = [](const std::vector<EngineSpec>& list, const std::string& backend) {
        int n = 0;
        for (const EngineSpec& e : list) n += e.machine == "local" && e.backend == backend ? 1 : 0;
        return n;
    };
    const int metalHubs = hubs(cfg_.selfplayEngines, "metal"), cudaHubs = hubs(cfg_.selfplayEngines, "cuda");
    const int metalArena = hubs(cfg_.arenaEngines, "metal") * arenaWaveNetworks(cfg_);
    const int cudaArena = hubs(cfg_.arenaEngines, "cuda") * arenaWaveNetworks(cfg_);
    const MachineSettings local = hosts_.settings("local");
    const int batch = local.inferenceBatch > 0 ? local.inferenceBatch : cfg_.inferenceBatch;
    if (batch < cfg_.mctsBatch)
        throw ConfigError("hosts file: local inference_batch (" + std::to_string(batch) +
                          ") is below mcts_batch (" + std::to_string(cfg_.mctsBatch) + ")");
#ifdef AZ_USE_METAL
    if (metalHubs > 0) checkMetalInferenceMemory(cfg_.netConfig(), batch, metalHubs, kInferenceSlots);
    if (metalArena > 0) checkMetalInferenceMemory(cfg_.netConfig(), batch, metalArena, 1);
#else
    (void)metalHubs;
    (void)metalArena;
#endif
#ifdef AZ_USE_CUDA
    if (cudaHubs > 0) checkCudaInferenceMemory(cfg_.netConfig(), batch, cudaHubs, kInferenceSlots);
    if (cudaArena > 0) checkCudaInferenceMemory(cfg_.netConfig(), batch, cudaArena, 1);
#else
    (void)cudaHubs;
    (void)cudaArena;
#endif

    // Remote machines: reachable, this build, and every engine's backend usable.
    if (!remotes.empty() && buildId() == "unknown")
        throw ConfigError("this binary has no build ID (built without git and without "
                          "AZGO_BUILD_ID), so remote engines cannot be checked; rebuild it");
    // Two aliases of one directory on one machine would clean each other's
    // files away (each clears the run's directory at its first contact).
    for (size_t a = 0; a < remotes.size(); ++a)
        for (size_t b = a + 1; b < remotes.size(); ++b) {
            const MachineSettings x = hosts_.settings(remotes[a]), y = hosts_.settings(remotes[b]);
            if (x.ssh == y.ssh && x.workdir == y.workdir)
                throw ConfigError(keyFor(remotes[a]) + ": " + remotes[a] + " and " + remotes[b] +
                                  " are the same machine and work directory in the hosts file; "
                                  "give each alias its own workdir");
        }
    for (const std::string& alias : remotes) {
        const MachineSettings m = hosts_.settings(alias);
        // Outside the run directory: preflight writes nothing there.
        const std::string logName =
            "azgo_preflight_" + std::to_string(getpid()) + "_" + alias + ".log";
        const std::string log = (std::filesystem::temp_directory_path() / logName).string();
        std::filesystem::remove(log);
        const std::unique_ptr<Transport> t = transportFor(m, log);
        const ProcessResult version = t->run({"--version"}, log, kPreflightTimeout);
        std::string id = buildLine(log);
        std::filesystem::remove(log);
        if (version.exitCode != 0 || id.empty()) {
            unreachable(alias, version.exitCode != 0 ? describe(version)
                                                     : "no build ID in its --version");
            continue;
        }
        if (!probeRemoteBuild().empty()) id = probeRemoteBuild();
        if (id != buildId())
            throw ConfigError(keyFor(alias) + ": " + alias + " runs build " + id +
                              " but this coordinator is build " + buildId() +
                              "; bring it to this commit with tools/cluster_sync.sh");
        remoteBuilds_[alias] = id;
        for (const EngineSpec& e : all) {
            if (e.machine != alias || fingerprints_.count(e.name())) continue;
            const ProcessResult r =
                t->run({"--fingerprint", "--backend", e.backend}, log, kPreflightTimeout);
            const std::string line = lastLine(log);
            std::filesystem::remove(log);
            if (r.exitCode == 1)   // selfplay ran and refused: the backend cannot run there
                throw ConfigError(keyFor(alias) + ": " + e.name() + ": " + alias +
                                  " cannot run backend " + e.backend + " (" + line + ")");
            if (r.exitCode != 0 || line.empty()) {
                unreachable(alias, describe(r));
                break;
            }
            bool printable = line.size() <= 1024;
            for (char c : line) printable = printable && static_cast<unsigned char>(c) >= 0x20;
            if (!printable) {
                unreachable(alias, "an unprintable fingerprint");
                break;
            }
            fingerprints_[e.name()] = line;
        }
    }

    // The engine lines, and a note for an engine whose arithmetic changed
    // since a manifest last recorded it.
    const std::map<std::string, std::string> recorded[2] = {
        lastFingerprints(manifestPath(runDir_), 11, 4),
        lastFingerprints(arenaManifestPath(runDir_), 12, 5)};
    std::vector<std::string> printed;
    for (const EngineSpec& e : all) {
        auto f = fingerprints_.find(e.name());
        if (f == fingerprints_.end()) continue;   // unreachable
        if (std::find(printed.begin(), printed.end(), e.name()) != printed.end()) continue;
        printed.push_back(e.name());
        const std::string build =
            e.machine == "local" ? shortBuildId() : shortOf(remoteBuilds_[e.machine]);
        // The fingerprint without its leading "<backend>; ", which the name says.
        const bool prefixed = f->second.rfind(e.backend + "; ", 0) == 0;
        const std::string shown = prefixed ? f->second.substr(e.backend.size() + 2) : f->second;
        if (log_) *log_ << "engine " << e.name() << ": " << shown << " [" << build << "]\n";
        for (const std::map<std::string, std::string>& manifest : recorded) {
            auto was = manifest.find(e.name());
            if (log_ && was != manifest.end() && was->second != f->second) {
                *log_ << "note: the fingerprint of " << e.name() << " changed from \"" << was->second
                      << "\"; from here on its games differ in the last bits from before\n";
                break;
            }
        }
    }
}

std::vector<std::string> Coordinator::prepareMachine(
    const std::string& alias, Transport& t, const std::vector<std::pair<std::string, uint64_t>>& networks) {
    // The networks, uploaded by this coordinator once per machine and hash
    // (a file already there may be a stranger's or cut short: it is
    // replaced, atomically); those the phase no longer needs go. At the
    // first contact with a machine, what an earlier coordinator of this run
    // left there (a crash: networks, jobs, results, parts of uploads) goes
    // first.
    const std::string dir = remoteDir(hosts_.settings(alias));
    std::vector<std::string> remoteNets;
    MachineState& state = machine(alias);
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.contacted) {
        std::vector<std::string> stale;
        for (const std::string& name : t.list(dir)) stale.push_back(dir + "/" + name);
        t.remove(stale);
        state.contacted = true;
    }
    for (const auto& [localPath, hash] : networks) {
        const std::string remoteNet = dir + "/" + netFileName(hash);
        remoteNets.push_back(remoteNet);
        if (!state.uploaded.count(remoteNet)) {
            t.upload(localPath, remoteNet);
            state.uploaded.insert(remoteNet);
        }
    }
    std::vector<std::string> unused;
    for (const std::string& net : state.uploaded)
        if (!state.keep.count(net)) unused.push_back(net);
    if (!unused.empty()) {
        t.remove(unused);
        for (const std::string& net : unused) state.uploaded.erase(net);
    }
    return remoteNets;
}

void Coordinator::attemptRemote(int attempt, const EngineSpec& e, const RemoteWork& work,
                                const std::string& workDir, const std::atomic<bool>& stop,
                                SliceReport& report) {
    auto checkStop = [&] {
        if (stop) throw SliceCancelled();
    };
    const MachineSettings m = hosts_.settings(e.machine);
    std::string engineFile = e.name();
    std::replace(engineFile.begin(), engineFile.end(), '/', '_');
    const std::string log = workDir + "/" + engineFile + ".log";
    const std::unique_ptr<Transport> t = transportFor(m, log);
    const std::string dir = remoteDir(m);

    checkStop();
    const std::vector<std::string> remoteNets = prepareMachine(e.machine, *t, work.networks);

    // The job: exactly what determines the slice's games. The file names
    // carry the attempt, so a straggler of attempt 1 cannot be taken for
    // attempt 2.
    const std::string base = work.base + "_a" + std::to_string(attempt);
    const std::string localJob = workDir + "/" + base + ".job";
    const std::string localResult = workDir + "/" + base + ".result";
    const std::string remoteJob = dir + "/" + base + ".job";
    const std::string remoteResult = dir + "/" + base + ".result";
    const std::string pattern = runId_ + "/" + base + ".job";
    try {
        writeFileAtomically(localJob, work.jobBytes);
        checkStop();
        t->upload(localJob, remoteJob);

        // --exit-on-stdin-eof: the remote stops if this process or the
        // connection goes away (the transport holds its stdin).
        std::vector<std::string> args = {"--job",     remoteJob, "--out",
                                         remoteResult, "--backend", e.backend,
                                         "--exit-on-stdin-eof"};
        // Worker counts the hosts file sets; otherwise the remote's own rule.
        const int workers = e.backend == "metal"  ? m.metalWorkers
                            : e.backend == "cuda" ? m.cudaWorkers
                                                  : m.cpuWorkers;
        if (workers > 0) args.insert(args.end(), {"--workers", std::to_string(workers)});
        if (m.inferenceBatch > 0)
            args.insert(args.end(), {"--inference-batch", std::to_string(m.inferenceBatch)});
        const double delay = probeDelay(e.name());
        if (delay > 0) std::this_thread::sleep_for(std::chrono::duration<double>(delay));
        checkStop();
        ProcessResult r;
        const ProbeFailure failure = probeFailure(e.name());
        switch (failure) {
        case ProbeFailure::Exit:   // the run fails as a dead connection would (ssh's 255)
            r.exitCode = 255;
            break;
        case ProbeFailure::Timeout:   // a remote that hangs, in place of the real run
            r = runProcess({"sleep", "600"}, log, m.timeout > 0 ? m.timeout : 1.0);
            break;
        default:
            r = t->run(args, log, m.timeout, &stop);
        }
        if (r.cancelled) throw SliceCancelled();
        if (r.exitCode != 0) {
            const std::string line = lastLine(log);
            // A network the binary refused (damaged on the machine since the
            // upload): remove and forget the slice's networks, so the next
            // attempt uploads them again and none is left behind.
            if (line.find("network") != std::string::npos) {
                MachineState& state = machine(e.machine);
                std::lock_guard<std::mutex> lock(state.mutex);
                try {
                    t->remove(remoteNets);
                } catch (const std::exception&) {   // best effort: a dead machine
                }
                for (const std::string& net : remoteNets) state.uploaded.erase(net);
            }
            // No log line in the message: selfplay's own words can hold the
            // machine's paths; the log keeps them.
            throw SliceError(e.machine + ": selfplay " + brief(r));
        }
        t->download(remoteResult, localResult);
        if (failure == ProbeFailure::Corrupt) {   // damage the copy as a bad transfer would
            std::fstream f(localResult, std::ios::in | std::ios::out | std::ios::binary);
            f.seekp(40);
            f.put('\x7f');
        }
        work.accept(localResult, report);
        t->remove({remoteJob, remoteResult});
        std::filesystem::remove(localJob);
        std::filesystem::remove(localResult);
    } catch (...) {
        // This attempt is over: its process (if it outlived its connection)
        // and its files go, best effort; the log stays.
        try {
            t->setCommandTimeout(kCleanupTimeout);   // a dead machine must not hold this up
            t->killJob(pattern);
            t->remove({remoteJob, remoteResult});
        } catch (const std::exception&) {
        }
        std::error_code ec;
        std::filesystem::remove(localJob, ec);
        std::filesystem::remove(localResult, ec);
        throw;
    }
}

void Coordinator::playRemote(int g, PhaseSlice& slice, const std::string& fallback,
                             const std::string& workDir, const std::atomic<bool>& stop,
                             const std::string& what, SliceReport& report) {
    // Two attempts on the planned engine: a transient failure costs time,
    // not reproducibility. A machine that does not answer at all costs
    // only a quick probe: one known to be down this phase (a generation's
    // self-play, or an arena wave), or found down after a failure, goes
    // straight to the fallback, and one that was down earlier is probed
    // before it is tried.
    const EngineSpec& e = slice.engine;
    std::string engineFile = e.name();
    std::replace(engineFile.begin(), engineFile.end(), '/', '_');
    const std::string log = workDir + "/" + engineFile + ".log";
    const MachineSettings m = hosts_.settings(e.machine);
    const std::unique_ptr<Transport> probe = transportFor(m, log);
    MachineState& state = machine(e.machine);
    auto isDown = [&] {
        std::lock_guard<std::mutex> lock(state.mutex);
        return state.down;
    };
    auto checkReachable = [&] {
        const bool up = probe->reachable();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.suspect = !up;
        if (!up) state.down = true;
        return up;
    };
    auto logLine = [&](const std::string& text) {
        std::ofstream out(log, std::ios::app | std::ios::binary);
        out << "azgo: " << text << "\n";
    };
    std::string reason;
    bool suspect;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        suspect = state.suspect;
    }
    if (isDown() || (suspect && !checkReachable())) {
        reason = e.machine + " unreachable";
        logLine("slice " + std::to_string(slice.slice.first) + ": " + reason + "; not tried");
        failedSlices_ = true;
    } else {
        for (int attempt = 1; attempt <= 2; ++attempt) {
            try {
                attemptRemote(attempt, e, slice.remote, workDir, stop, report);
                return;
            } catch (const SliceCancelled&) {
                throw;
            } catch (const std::exception& ex) {
                reason = ex.what();
                failedSlices_ = true;
                logLine("slice " + std::to_string(slice.slice.first) + ", attempt " +
                        std::to_string(attempt) + " failed: " + reason);
                if (stop) throw SliceCancelled();
                if (!checkReachable()) {
                    reason = e.machine + " unreachable";
                    break;
                }
            }
        }
    }
    if (cfg_.selfplayFallback == "stop")
        throw SliceError(e.name() + " failed twice (" + reason + "); selfplay_fallback = stop");

    // selfplay_fallback = local: the coordinator plays the slice itself, on
    // the fallback engine, and says so in the manifest and the log.
    const std::string backend = fallback.substr(fallback.find('/') + 1);
    const MachineSettings local = hosts_.settings("local");
    EngineRun run;
    run.backend = backend;
    run.workers = engineWorkers(local, backend, slice.slice.count, resolveWorkers(0));
    run.inferenceBatch = local.inferenceBatch > 0 ? local.inferenceBatch : cfg_.inferenceBatch;
    run.qos = backend == "metal" || backend == "cuda" ? CpuQos::Default : local.cpuQos;
    run.cancel = &stop;
    report.engine = fallback;
    report.fingerprint = engineFingerprint(backend);
    report.build = shortBuildId();
    report.note = "fallback: " + reason;
    if (log_) {
        std::lock_guard<std::mutex> lock(logMutex_);
        *log_ << "note: generation " << g << ": " << what << " " << slice.slice.first << "-"
              << slice.slice.first + slice.slice.count - 1 << " (" << e.name() << ") played by "
              << fallback << ": " << reason
              << "; from here on results differ in the last bits from a run without the failure\n";
    }
    slice.playLocal(run, &report.stats);
}

std::vector<SliceReport> Coordinator::runSlices(int g, std::vector<PhaseSlice>& slices,
                                                const std::vector<EngineSpec>& list,
                                                const std::string& workDir, const std::string& what) {
    const int cores = resolveWorkers(0);
    const size_t n = slices.size();
    std::vector<SliceReport> reports(n);
    std::vector<EngineRun> runs(n);
    std::map<std::string, std::set<std::string>> keep;   // alias -> the networks its slices need
    for (size_t k = 0; k < n; ++k) {
        const EngineSpec& e = slices[k].engine;
        const MachineSettings m = hosts_.settings(e.machine);
        reports[k].slice = slices[k].slice;
        reports[k].planned = reports[k].engine = e.name();
        if (!m.local()) {
            // Every remote machine's entry exists before the threads start.
            if (!machines_.count(e.machine)) {
                machines_[e.machine] = std::make_unique<MachineState>();
                machines_[e.machine]->suspect = unreachable_.count(e.machine) > 0;
            }
            machines_[e.machine]->down = false;   // tried afresh every phase
            std::set<std::string>& nets = keep[e.machine];
            for (const auto& net : slices[k].remote.networks)
                nets.insert(remoteDir(m) + "/" + netFileName(net.second));
            continue;
        }
        runs[k].backend = e.backend;
        runs[k].workers = engineWorkers(m, e.backend, slices[k].slice.count, cores);
        runs[k].inferenceBatch = m.inferenceBatch > 0 ? m.inferenceBatch : cfg_.inferenceBatch;
        // The GPU, which sets the pace, waits on a GPU engine's workers
        // (metal, cuda): they keep the default class. CPU engines take the
        // machine's.
        runs[k].qos = e.backend == "metal" || e.backend == "cuda" ? CpuQos::Default : m.cpuQos;
        if (!fingerprints_.count(e.name())) fingerprints_[e.name()] = engineFingerprint(e.backend);
        reports[k].fingerprint = fingerprints_[e.name()];
        reports[k].build = shortBuildId();
    }
    for (auto& [alias, nets] : keep) machines_.at(alias)->keep = nets;

    // The BLAS backend sets its environment once, with setenv, which is not
    // thread-safe: do that here, before any engine thread runs.
    const std::string fallback = fallbackEngine(list, cfg_.backend);
    for (const EngineRun& run : runs)
        if (run.backend == "blas" || (!keep.empty() && fallback == "local/blas")) {
            makeBackend("blas", 1);
            break;
        }
    failedSlices_ = false;

    // One thread per slice, all at once. Each local engine has its own
    // evaluators, so a slice's games are what it would give alone. The
    // first slice to fail sets `stop`, and the local ones stop at their
    // next network evaluation: the generation will not be committed anyway.
    std::atomic<bool> stop{false};
    for (EngineRun& run : runs) run.cancel = &stop;
    std::vector<std::exception_ptr> errors(n);
    std::vector<std::thread> threads;
    std::exception_ptr launchError;
    try {
        for (size_t k = 0; k < n; ++k)
            threads.emplace_back([&, k] {
                try {
                    if (slices[k].engine.machine == "local")
                        slices[k].playLocal(runs[k], &reports[k].stats);
                    else
                        playRemote(g, slices[k], fallback, workDir, stop, what, reports[k]);
                } catch (...) {
                    errors[k] = std::current_exception();
                    stop = true;
                }
            });
    } catch (...) {
        launchError = std::current_exception();   // a thread could not start: join the others first
        stop = true;
    }
    for (std::thread& t : threads) t.join();
    if (launchError) std::rethrow_exception(launchError);
    // The failure itself, by list order, rather than the cancellations it caused.
    std::exception_ptr first;
    for (const std::exception_ptr& e : errors) {
        if (!e) continue;
        try {
            std::rethrow_exception(e);
        } catch (const SliceCancelled&) {
            if (!first) first = e;
            continue;
        } catch (...) {
        }
        std::rethrow_exception(e);
    }
    if (first) std::rethrow_exception(first);
    return reports;
}

std::vector<GameRecord> Coordinator::play(int g, const std::string& networkPath, const Network& net,
                                          const SelfPlayConfig& sp, int games) {
    if (cfg_.selfplayEngines.empty())
        throw std::logic_error("Coordinator::play: selfplay_engines is empty (the default path)");
    if (cfg_.selfplayAllocation == "dynamic") return playDynamic(g, networkPath, net, sp, games);
    assignment_.clear();
    const std::vector<Slice> assigned = assignGames(cfg_.selfplayEngines, games);
    const size_t n = assigned.size();
    // Remote slices' work files: this generation's directory, made afresh.
    char genDir[32];
    std::snprintf(genDir, sizeof genDir, "gen_%04d", g);
    const std::string workDir =
        (std::filesystem::path(runDir_) / "selfplay-work" / genDir).string();
    bool anyRemote = false;
    for (const Slice& s : assigned)
        anyRemote = anyRemote ||
                    !hosts_.settings(cfg_.selfplayEngines[static_cast<size_t>(s.engine)].machine).local();
    uint64_t networkHash = 0;
    if (anyRemote) {
        std::filesystem::remove_all(workDir);
        std::filesystem::create_directories(workDir);
        networkHash = fileHash(networkPath);
    }

    std::vector<std::vector<GameRecord>> results(n);
    std::vector<PhaseSlice> slices(n);
    for (size_t k = 0; k < n; ++k) {
        PhaseSlice& ps = slices[k];
        ps.engine = cfg_.selfplayEngines[static_cast<size_t>(assigned[k].engine)];
        ps.slice = assigned[k];
        const int first = assigned[k].first, count = assigned[k].count;
        ps.playLocal = [&, k, first, count](const EngineRun& run, SliceStats* stats) {
            results[k] = playSelfPlayGames(net, sp, cfg_.seed, g, first, count, run, stats);
        };
        if (hosts_.settings(ps.engine.machine).local()) continue;
        // The job: exactly what determines the slice's games.
        SelfPlayJob job;
        job.seed = cfg_.seed;
        job.generation = g;
        job.first = first;
        job.count = count;
        job.boardSize = cfg_.boardSize;
        job.sp = sp;
        job.networkFile = netFileName(networkHash);
        job.networkHash = networkHash;
        job.buildId = buildId();
        job.inferenceBatch = cfg_.inferenceBatch;
        char base[48];
        std::snprintf(base, sizeof base, "g%04d_%d", g, first);
        ps.remote.base = base;
        ps.remote.jobBytes = serializeJob(job);
        ps.remote.networks = {{networkPath, networkHash}};
        const std::string jobBytes = ps.remote.jobBytes;
        ps.remote.accept = [&results, k, jobBytes](const std::string& path, SliceReport& report) {
            SliceResult result = readSliceResult(path, jobBytes, buildId());
            report.fingerprint = result.fingerprint;
            report.build = shortOf(result.buildId);
            report.stats = result.stats;
            results[k] = std::move(result.records);
        };
    }
    std::vector<SliceReport> reports = runSlices(g, slices, cfg_.selfplayEngines, workDir, "games");

    // Merge by game index: records[first + k] = slice's k-th, each index
    // filled exactly once, whatever order the slices finished in.
    std::vector<GameRecord> records(static_cast<size_t>(games));
    std::vector<char> filled(static_cast<size_t>(games), 0);
    for (size_t k = 0; k < n; ++k) {
        if (results[k].size() != static_cast<size_t>(assigned[k].count))
            throw std::logic_error("slice of " + reports[k].engine +
                                   " returned the wrong number of games");
        for (int j = 0; j < assigned[k].count; ++j) {
            const size_t i = static_cast<size_t>(assigned[k].first + j);
            if (i >= records.size() || filled[i])
                throw std::logic_error("game " + std::to_string(i) +
                                       " assigned twice or out of range");
            filled[i] = 1;
            records[i] = std::move(results[k][static_cast<size_t>(j)]);
        }
    }
    for (size_t i = 0; i < filled.size(); ++i)
        if (!filled[i]) throw std::logic_error("game " + std::to_string(i) + " was not played");
    if (anyRemote) finishWorkDir(workDir, failedSlices_);
    reports_ = std::move(reports);
    return records;
}

// ---- dynamic allocation -----------------------------------------------------------

// One generation's dynamic self-play: the queue, the engines and their
// threads, and the merged records. Everything shared is guarded by `mutex`;
// waiting threads sleep on `wake`, which every change to the queue
// notifies, and wake up at least every kDynamicPoll seconds, since the
// handout rule depends on time.
struct Coordinator::DynamicPhase {
    static constexpr double kDynamicPoll = 0.5;
    // How long a session may take to end after `end` (all its games are
    // back by then: only the end frame and the exit are left).
    static constexpr double kEndTimeout = 120;

    struct Engine {
        EngineSpec spec;
        MachineSettings machine;
        bool active = false;      // the fallback engine is inactive until it joins
        int concurrency = 1;
        SliceReport report;
        double first = -1, last = 0;   // first handout, last game back (phase seconds)
        int failedSessions = 0, requeued = 0;
        std::string why;               // the last failure
        bool retired = false;
        std::exception_ptr error;
    };

    Coordinator& c;
    const int g, games;
    const Network& net;
    const SelfPlayConfig& sp;
    std::string networkPath, workDir, jobBytes;
    uint64_t networkHash = 0;
    std::vector<Engine> engines;   // the queue's engines, in order; the last is the fallback
    std::mutex mutex;
    std::condition_variable wake;
    std::unique_ptr<GameQueue> queue;
    std::vector<GameRecord> records;
    std::atomic<bool> stop{false};
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    bool fallbackJoined = false;

    DynamicPhase(Coordinator& coordinator, int generation, int count, const Network& network,
                 const SelfPlayConfig& config)
        : c(coordinator), g(generation), games(count), net(network), sp(config),
          records(static_cast<size_t>(count)) {}

    double now() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    // Under the mutex: engine k was handed a game.
    void handedOut(size_t k) {
        if (engines[k].first < 0) engines[k].first = now();
    }

    // Under the mutex: game i of engine k came back, verified.
    void gameBack(size_t k, int i, GameRecord record) {
        records[static_cast<size_t>(i)] = std::move(record);
        queue->finished(static_cast<int>(k), i, now());
        engines[k].last = now();
        wake.notify_all();
    }

    void note(const std::string& text) {
        if (!c.log_) return;
        std::lock_guard<std::mutex> lock(c.logMutex_);
        *c.log_ << "note: generation " << g << ": " << text << "\n";
    }

    std::string logPath(const Engine& e) const {
        std::string engineFile = e.spec.name();
        std::replace(engineFile.begin(), engineFile.end(), '/', '_');
        return workDir + "/" + engineFile + ".log";
    }

    // ---- a local engine: a pool of workers taking games from the queue ----
    class QueueSource : public GameSource {
    public:
        QueueSource(DynamicPhase& phase, size_t k) : p_(phase), k_(k) {}
        int next() override {
            std::unique_lock<std::mutex> lock(p_.mutex);
            for (;;) {
                if (p_.stop) return -1;
                const int i = p_.queue->take(static_cast<int>(k_), p_.now());
                if (i >= 0) {
                    p_.handedOut(k_);
                    return i;
                }
                if (i == GameQueue::kNone) return -1;
                p_.wake.wait_for(lock, std::chrono::duration<double>(kDynamicPoll));
            }
        }
        void done(int index, GameRecord record, long long) override {
            std::lock_guard<std::mutex> lock(p_.mutex);
            p_.gameBack(k_, index, std::move(record));
        }
        void stop() override {
            // A worker failed: the generation fails with it.
            std::lock_guard<std::mutex> lock(p_.mutex);
            p_.stop = true;
            p_.wake.notify_all();
        }
    private:
        DynamicPhase& p_;
        size_t k_;
    };

    void runLocal(size_t k) {
        Engine& e = engines[k];
        EngineRun run;
        run.backend = e.spec.backend;
        run.workers = e.concurrency;
        run.inferenceBatch = e.machine.inferenceBatch > 0 ? e.machine.inferenceBatch : c.cfg_.inferenceBatch;
        // The GPU, which sets the pace, waits on a GPU engine's workers: they
        // keep the default class. CPU engines take the machine's.
        run.qos = run.backend == "metal" || run.backend == "cuda" ? CpuQos::Default : e.machine.cpuQos;
        run.cancel = &stop;
        const double delay = probeDelay(e.spec.name());
        if (delay > 0) std::this_thread::sleep_for(std::chrono::duration<double>(delay));
        QueueSource source(*this, k);
        SliceStats stats;
        playSelfPlayQueue(net, sp, c.cfg_.seed, g, run, source, &stats);
        std::lock_guard<std::mutex> lock(mutex);
        e.report.stats.positions = stats.positions;
        e.report.stats.evaluations = stats.evaluations;
    }

    // ---- a remote engine: sessions, retried once, then retired ----
    void runRemote(size_t k) {
        Engine& e = engines[k];
        const std::string log = logPath(e);
        const std::unique_ptr<Transport> probe = c.transportFor(e.machine, log);
        MachineState& state = c.machine(e.spec.machine);
        auto checkReachable = [&] {
            const bool up = probe->reachable();
            std::lock_guard<std::mutex> lock(state.mutex);
            state.suspect = !up;
            if (!up) state.down = true;
            return up;
        };
        auto logLine = [&](const std::string& text) {
            std::ofstream out(log, std::ios::app | std::ios::binary);
            out << "azgo: " << text << "\n";
        };
        bool down, suspect;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            down = state.down;
            suspect = state.suspect;
        }
        std::string reason;
        if (down || (suspect && !checkReachable())) {
            reason = e.spec.machine + " unreachable";
            logLine("stream: " + reason + "; not tried");
            c.failedSlices_ = true;
        } else {
            for (int attempt = 1; attempt <= 2; ++attempt) {
                try {
                    session(k, attempt);
                    return;
                } catch (const SliceCancelled&) {
                    throw;
                } catch (const std::exception& ex) {
                    reason = ex.what();
                    c.failedSlices_ = true;
                    logLine("stream, attempt " + std::to_string(attempt) + " failed: " + reason);
                    int requeued;
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        ++e.failedSessions;
                        e.why = reason;
                        requeued = e.requeued;
                    }
                    note(e.spec.name() + "'s stream failed (" + reason + "); its games in flight return to "
                         "the queue (" + std::to_string(requeued) + " so far)");
                    if (stop) throw SliceCancelled();
                    if (!checkReachable()) {
                        reason = e.spec.machine + " unreachable";
                        break;
                    }
                }
            }
        }
        if (c.cfg_.selfplayFallback == "stop")
            throw SliceError(e.spec.name() + " failed twice (" + reason + "); selfplay_fallback = stop");
        // selfplay_fallback = local: the engine sits out the rest of the
        // generation; the other engines play its games.
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue->retire(static_cast<int>(k));
            e.retired = true;
            e.why = reason;
            wake.notify_all();
        }
        note(e.spec.name() + " retired for this generation (" + reason +
             "); the other engines play its games; from here on results differ in the last bits from a "
             "run without the failure");
    }

    // One session of remote engine k: start to `end`, or a SliceError after
    // its games in flight went back to the queue and its files and process
    // are gone.
    void session(size_t k, int attempt) {
        Engine& e = engines[k];
        const MachineSettings& m = e.machine;
        const std::string log = logPath(e);
        const std::unique_ptr<Transport> t = c.transportFor(m, log);
        const std::string dir = c.remoteDir(m);
        char base[48];
        std::snprintf(base, sizeof base, "g%04d_s%zu_a%d", g, k, attempt);
        const std::string localJob = workDir + "/" + base + ".job";
        const std::string remoteJob = dir + "/" + base + ".job";
        const std::string pattern = c.runId_ + "/" + base + ".job";
        const int failAfter = probeSessionFailure(e.spec.name());

        std::mutex failMutex;
        std::string failure;   // the first failure of this session
        std::atomic<bool> failed{false};
        auto fail = [&](const std::string& why) {
            std::lock_guard<std::mutex> lock(failMutex);
            if (!failed) failure = why;
            failed = true;
            std::lock_guard<std::mutex> wakeLock(mutex);
            wake.notify_all();
        };
        std::unique_ptr<StreamChannel> channel;
        std::vector<std::string> remoteNets;
        try {
            if (stop) throw SliceCancelled();
            remoteNets = c.prepareMachine(e.spec.machine, *t, {{networkPath, networkHash}});
            writeFileAtomically(localJob, jobBytes);
            if (stop) throw SliceCancelled();
            t->upload(localJob, remoteJob);
            std::vector<std::string> args = {"--stream", "--job", remoteJob, "--backend", e.spec.backend,
                                             "--workers", std::to_string(e.concurrency)};
            if (m.inferenceBatch > 0) args.insert(args.end(), {"--inference-batch", std::to_string(m.inferenceBatch)});
            const double delay = probeDelay(e.spec.name());
            if (delay > 0) std::this_thread::sleep_for(std::chrono::duration<double>(delay));
            channel = t->openStream(args, log);
        } catch (const SliceCancelled&) {
            cleanupSession(*t, pattern, remoteJob, localJob);
            throw;
        } catch (const std::exception& ex) {
            cleanupSession(*t, pattern, remoteJob, localJob);
            throw SliceError(ex.what());
        }

        // The reader: frames in, each game verified and kept.
        std::atomic<bool> helloSeen{false}, endSeen{false}, ended{false};
        std::thread reader([&] {
            StreamFrames frames;
            StreamChecker check(jobBytes, buildId());
            double lastFrame = now(), endAt = 0;
            int returned = 0;
            char buf[1 << 16];
            while (!failed && !stop) {
                const long n = channel->receive(buf, sizeof buf, kDynamicPoll);
                if (n < 0) {
                    const double t0 = now();
                    if (ended) {
                        if (endAt == 0) endAt = t0;
                        if (t0 - endAt > kEndTimeout) fail("selfplay did not end after `end`");
                        continue;
                    }
                    if (m.timeout > 0 && t0 - lastFrame > m.timeout) {
                        bool busy;
                        {
                            std::lock_guard<std::mutex> lock(mutex);
                            busy = queue->inFlight(static_cast<int>(k)) > 0;
                        }
                        if (busy) fail("no game for " + std::to_string(static_cast<int>(m.timeout)) + " s");
                    }
                    continue;
                }
                if (n == 0) {
                    if (!endSeen) fail("selfplay ended before the generation did");
                    return;
                }
                lastFrame = now();
                frames.feed(buf, static_cast<size_t>(n));
                try {
                    char kind;
                    std::string payload;
                    while (frames.next(kind, payload)) {
                        if (kind == 'H') {
                            const StreamHello h = check.hello(payload);
                            std::lock_guard<std::mutex> lock(mutex);
                            e.report.fingerprint = h.fingerprint;
                            e.report.build = shortOf(h.buildId);
                            helloSeen = true;
                            wake.notify_all();
                        } else if (kind == 'G') {
                            StreamGame game = check.game(payload);
                            std::lock_guard<std::mutex> lock(mutex);
                            const std::vector<int> held = queue->held(static_cast<int>(k));
                            if (!std::binary_search(held.begin(), held.end(), game.index))
                                throw SliceError("stream: game " + std::to_string(game.index) +
                                                 " was not asked of this session");
                            e.report.stats.positions += game.record.positions();
                            e.report.stats.evaluations += game.evaluations;
                            gameBack(k, game.index, std::move(game.record));
                            ++returned;
                        } else {
                            check.end(payload);
                            endSeen = true;
                        }
                        if (failAfter >= 0 && returned >= failAfter && helloSeen)
                            throw SliceError("the connection dropped (test)");
                    }
                } catch (const std::exception& ex) {
                    fail(ex.what());
                    return;
                }
            }
        });

        // The feeder (this thread): keeps the session's games in flight at
        // its concurrency, and says `end` once every game is back.
        {
            std::unique_lock<std::mutex> lock(mutex);
            while (!stop && !failed) {
                if (queue->done()) {
                    lock.unlock();
                    if (!channel->send("end\n")) fail("cannot write to selfplay (the connection is gone)");
                    ended = true;
                    lock.lock();
                    break;
                }
                std::string commands;
                if (helloSeen)
                    for (int i; (i = queue->take(static_cast<int>(k), now())) >= 0;) {
                        handedOut(k);
                        commands += "play " + std::to_string(i) + "\n";
                    }
                if (!commands.empty()) {
                    lock.unlock();
                    if (!channel->send(commands)) fail("cannot write to selfplay (the connection is gone)");
                    lock.lock();
                    continue;
                }
                wake.wait_for(lock, std::chrono::duration<double>(kDynamicPoll));
            }
        }
        if (!ended || failed || stop) channel->stop();
        reader.join();
        const ProcessResult r = channel->wait();
        if (!failed && !stop && (!endSeen || r.exitCode != 0)) fail("selfplay " + brief(r));
        if (stop && !failed) {
            cleanupSession(*t, pattern, remoteJob, localJob);
            throw SliceCancelled();
        }
        if (failed) {
            // A network the binary refused (damaged on the machine since the
            // upload): forget it, so the next attempt uploads it again.
            if (lastLine(log).find("network") != std::string::npos) {
                MachineState& state = c.machine(e.spec.machine);
                std::lock_guard<std::mutex> lock(state.mutex);
                try {
                    t->remove(remoteNets);
                } catch (const std::exception&) {
                }
                for (const std::string& n : remoteNets) state.uploaded.erase(n);
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                for (int i : queue->held(static_cast<int>(k))) {
                    queue->requeue(static_cast<int>(k), i);
                    ++e.requeued;
                }
                wake.notify_all();
            }
            cleanupSession(*t, pattern, remoteJob, localJob);
            throw SliceError(e.spec.machine + ": " + failure);
        }
        t->remove({remoteJob});
        std::filesystem::remove(localJob);
    }

    void cleanupSession(Transport& t, const std::string& pattern, const std::string& remoteJob,
                        const std::string& localJob) {
        try {
            t.setCommandTimeout(kCleanupTimeout);   // a dead machine must not hold this up
            t.killJob(pattern);
            t.remove({remoteJob});
        } catch (const std::exception&) {
        }
        std::error_code ec;
        std::filesystem::remove(localJob, ec);
    }

    void runEngine(size_t k) {
        try {
            if (engines[k].machine.local()) runLocal(k);
            else runRemote(k);
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            engines[k].error = std::current_exception();
            stop = true;
            wake.notify_all();
        }
    }
};

std::vector<GameRecord> Coordinator::playDynamic(int g, const std::string& networkPath, const Network& net,
                                                 const SelfPlayConfig& sp, int games) {
    DynamicPhase p(*this, g, games, net, sp);
    p.networkPath = networkPath;
    const int cores = resolveWorkers(0);

    // The engines: every item with a share, in list order; their default
    // concurrency follows the share. Then the fallback engine, inactive.
    for (const Slice& s : assignGames(cfg_.selfplayEngines, games)) {
        DynamicPhase::Engine e;
        e.spec = cfg_.selfplayEngines[static_cast<size_t>(s.engine)];
        e.machine = hosts_.settings(e.spec.machine);
        e.active = true;
        e.concurrency = dynamicWorkers(e.machine, e.spec.backend, s.count, games, cores);
        p.engines.push_back(e);
    }
    const size_t listed = p.engines.size();
    {
        DynamicPhase::Engine f;
        f.spec = parseEngineList(fallbackEngine(cfg_.selfplayEngines, cfg_.backend) + ":1")[0];
        f.machine = hosts_.settings("local");
        f.concurrency = dynamicWorkers(f.machine, f.spec.backend, games, games, cores);
        p.engines.push_back(f);
    }
    std::vector<int> concurrency;
    for (size_t k = 0; k < listed; ++k) concurrency.push_back(p.engines[k].concurrency);
    const std::vector<std::string> forced = probeForced(g);
    if (!forced.empty()) {
        if (forced.size() != static_cast<size_t>(games))
            throw std::logic_error("forceAssignment: " + std::to_string(forced.size()) + " engines for " +
                                   std::to_string(games) + " games");
        std::vector<int> engineOf;
        for (const std::string& name : forced) {
            size_t k = 0;
            while (k < listed && p.engines[k].spec.name() != name) ++k;
            if (k == listed) throw std::logic_error("forceAssignment: no engine " + name);
            engineOf.push_back(static_cast<int>(k));
        }
        p.queue = std::make_unique<GameQueue>(engineOf, concurrency);
    } else {
        p.queue = std::make_unique<GameQueue>(games, concurrency);
    }

    // Remote machines: their state exists before any thread starts; their
    // work goes through this generation's work directory.
    char genDir[32];
    std::snprintf(genDir, sizeof genDir, "gen_%04d", g);
    p.workDir = (std::filesystem::path(runDir_) / "selfplay-work" / genDir).string();
    bool anyRemote = false;
    for (size_t k = 0; k < listed; ++k) anyRemote = anyRemote || !p.engines[k].machine.local();
    if (anyRemote) {
        std::filesystem::remove_all(p.workDir);
        std::filesystem::create_directories(p.workDir);
        p.networkHash = fileHash(networkPath);
        SelfPlayJob job;
        job.seed = cfg_.seed;
        job.generation = g;
        job.first = 0;
        job.count = games;
        job.boardSize = cfg_.boardSize;
        job.sp = sp;
        job.networkFile = netFileName(p.networkHash);
        job.networkHash = p.networkHash;
        job.buildId = buildId();
        job.inferenceBatch = cfg_.inferenceBatch;
        p.jobBytes = serializeJob(job);
    }
    std::map<std::string, std::set<std::string>> keep;
    for (size_t k = 0; k < listed; ++k) {
        DynamicPhase::Engine& e = p.engines[k];
        e.report.planned = e.report.engine = e.spec.name();
        e.report.dynamic = true;
        if (e.machine.local()) {
            if (!fingerprints_.count(e.spec.name())) fingerprints_[e.spec.name()] = engineFingerprint(e.spec.backend);
            e.report.fingerprint = fingerprints_[e.spec.name()];
            e.report.build = shortBuildId();
            continue;
        }
        if (!machines_.count(e.spec.machine)) {
            machines_[e.spec.machine] = std::make_unique<MachineState>();
            machines_[e.spec.machine]->suspect = unreachable_.count(e.spec.machine) > 0;
        }
        machines_[e.spec.machine]->down = false;   // tried afresh every phase
        keep[e.spec.machine].insert(remoteDir(e.machine) + "/" + netFileName(p.networkHash));
    }
    for (auto& [alias, nets] : keep) machines_.at(alias)->keep = nets;
    // The BLAS backend sets its environment once, with setenv, which is not
    // thread-safe: do that here, before any engine thread runs.
    for (const DynamicPhase::Engine& e : p.engines)
        if (e.machine.local() && e.spec.backend == "blas") {
            makeBackend("blas", 1);
            break;
        }
    failedSlices_ = false;

    std::vector<std::thread> threads;
    {
        std::unique_lock<std::mutex> lock(p.mutex);
        try {
            for (size_t k = 0; k < listed; ++k) threads.emplace_back([&p, k] { p.runEngine(k); });
            // Until every game is back or the generation failed; the
            // fallback engine joins once every listed engine has retired.
            while (!p.queue->done() && !p.stop) {
                bool allRetired = !p.fallbackJoined;
                for (size_t k = 0; k < listed && allRetired; ++k) allRetired = p.engines[k].retired;
                if (allRetired) {
                    DynamicPhase::Engine& f = p.engines[listed];
                    f.active = true;
                    f.report.planned = f.report.engine = f.spec.name();
                    f.report.dynamic = true;
                    f.report.fingerprint = engineFingerprint(f.spec.backend);
                    f.report.build = shortBuildId();
                    if (p.queue->addEngine(f.concurrency) != static_cast<int>(listed))
                        throw std::logic_error("playDynamic: the fallback engine's number");
                    p.fallbackJoined = true;
                    lock.unlock();
                    p.note("every engine retired; " + f.spec.name() + " plays the remaining games");
                    threads.emplace_back([&p, listed] { p.runEngine(listed); });
                    lock.lock();
                    continue;
                }
                p.wake.wait_for(lock, std::chrono::duration<double>(DynamicPhase::kDynamicPoll));
            }
        } catch (...) {
            p.stop = true;   // a thread could not start: stop the others, join them first
            p.engines[listed].error = std::current_exception();
        }
        p.wake.notify_all();
    }
    for (std::thread& t : threads) t.join();

    // The failure itself, by engine order, rather than the cancellations it caused.
    std::exception_ptr firstCancel;
    for (const DynamicPhase::Engine& e : p.engines) {
        if (!e.error) continue;
        try {
            std::rethrow_exception(e.error);
        } catch (const SliceCancelled&) {
            if (!firstCancel) firstCancel = e.error;
            continue;
        } catch (...) {
        }
        std::rethrow_exception(e.error);
    }
    if (firstCancel) std::rethrow_exception(firstCancel);
    if (!p.queue->done()) throw std::logic_error("playDynamic: games left unplayed");

    // Who played what.
    std::vector<SliceReport> reports;
    for (size_t k = 0; k < p.engines.size(); ++k) {
        DynamicPhase::Engine& e = p.engines[k];
        if (!e.active) continue;
        e.report.slice.engine = static_cast<int>(k);
        e.report.slice.count = p.queue->finishedBy(static_cast<int>(k));
        e.report.stats.seconds = e.first < 0 ? 0 : e.last - e.first;
        std::string note;
        if (e.failedSessions > 0)
            note = std::to_string(e.failedSessions) + " failed stream" + (e.failedSessions > 1 ? "s" : "") + ", " +
                   std::to_string(e.requeued) + " games requeued";
        if (e.retired) note += std::string(note.empty() ? "" : "; ") + "retired";
        if (!note.empty()) note += ": " + e.why;
        if (k == listed) note = "fallback: every engine retired";
        e.report.note = note;
        reports.push_back(e.report);
    }
    assignment_.assign(static_cast<size_t>(games), "");
    for (int i = 0; i < games; ++i)
        assignment_[static_cast<size_t>(i)] = p.engines[static_cast<size_t>(p.queue->playedBy(i))].spec.name();
    if (anyRemote) finishWorkDir(p.workDir, failedSlices_);
    reports_ = std::move(reports);
    return std::move(p.records);
}

void Coordinator::beginArena(int g) {
    arenaGeneration_ = g;
    arenaWave_ = 0;
    arenaReports_.clear();
}

std::vector<MatchResult> Coordinator::playArena(const std::vector<ArenaMatchSpec>& wave,
                                                const ArenaConfig& cfg) {
    if (cfg_.arenaEngines.empty())
        throw std::logic_error("Coordinator::playArena: arena_engines is empty (the default path)");
    const int g = arenaGeneration_, w = ++arenaWave_;
    int total = 0;
    for (const ArenaMatchSpec& m : wave) {
        if (m.games < 2 || m.games % 2 != 0)
            throw std::invalid_argument("playMatch: games must be even and >= 2, got " +
                                        std::to_string(m.games));
        total += m.games;
    }
    if (wave.empty()) return {};
    // Every match checked as playMatches checks it, before any slice starts
    // (ranges of no games play nothing but are checked).
    {
        std::vector<ArenaGameRange> unplayed;
        for (const ArenaMatchSpec& m : wave) unplayed.push_back({m.a, m.b, m.seed, 0, 0});
        EngineRun check;
        check.backend = cfg_.backend;
        check.inferenceBatch = cfg_.inferenceBatch;
        playArenaGames(unplayed, cfg, check);
    }
    const std::vector<Slice> assigned = assignGames(cfg_.arenaEngines, total);
    const size_t n = assigned.size();
    char waveDir[48];
    std::snprintf(waveDir, sizeof waveDir, "gen_%04d_w%d", g, w);
    const std::string workDir = (std::filesystem::path(runDir_) / "arena-work" / waveDir).string();
    const std::string netDir = workDir + "/networks";
    bool anyRemote = false;
    for (const Slice& s : assigned)
        anyRemote = anyRemote ||
                    !hosts_.settings(cfg_.arenaEngines[static_cast<size_t>(s.engine)].machine).local();
    // The wave's networks, in order of first appearance, each saved from
    // memory (a reference network in play is fitted to the run's shape, so
    // it is not its file) when a remote engine plays: path and fileHash.
    std::vector<const Network*> nets;
    for (const ArenaMatchSpec& m : wave)
        for (const ArenaPlayer* p : {&m.a, &m.b})
            if (p->net && std::find(nets.begin(), nets.end(), p->net) == nets.end()) nets.push_back(p->net);
    // The saved networks go whatever happens, a failed save included; the
    // logs stay as play()'s do.
    struct RemoveNetworks {
        std::string dir;
        ~RemoveNetworks() {
            std::error_code ec;
            if (!dir.empty()) std::filesystem::remove_all(dir, ec);
        }
    } removeNetworks{anyRemote ? netDir : ""};
    std::vector<std::pair<std::string, uint64_t>> files;
    if (anyRemote) {
        std::filesystem::remove_all(workDir);
        std::filesystem::create_directories(netDir);
        for (size_t i = 0; i < nets.size(); ++i) {
            const std::string path = netDir + "/wave_net_" + std::to_string(i) + ".azn";
            Network copy = *nets[i];
            saveNetwork(copy, path);
            files.push_back({path, fileHash(path)});
        }
    }

    ArenaConfig resolved = cfg;
    resolved.maxMoves = cfg.maxMoves > 0 ? cfg.maxMoves : defaultMaxMoves(cfg_.boardSize);
    std::vector<std::vector<ArenaGameResult>> results(n);
    std::vector<PhaseSlice> slices(n);
    for (size_t k = 0; k < n; ++k) {
        PhaseSlice& ps = slices[k];
        ps.engine = cfg_.arenaEngines[static_cast<size_t>(assigned[k].engine)];
        ps.slice = assigned[k];
        const std::vector<ArenaGameRange> ranges = waveSlice(wave, assigned[k].first, assigned[k].count);
        ps.playLocal = [&results, k, ranges, cfg](const EngineRun& run, SliceStats* stats) {
            results[k] = playArenaGames(ranges, cfg, run, stats);
        };
        if (hosts_.settings(ps.engine.machine).local()) continue;
        // The job: the slice's ranges, naming only the networks they use.
        ArenaJob job;
        job.boardSize = cfg_.boardSize;
        job.cfg = resolved;
        job.buildId = buildId();
        job.inferenceBatch = cfg_.inferenceBatch;
        std::vector<size_t> used;   // indices into nets, in the job's order
        auto seat = [&](const ArenaPlayer& p) {
            ArenaJobPlayer out;
            out.simulations = p.simulations;
            out.randomSymmetry = p.randomSymmetry;
            out.cPuct = p.cPuct;
            out.mctsBatch = p.mctsBatch;
            if (p.net) {
                const size_t i = static_cast<size_t>(std::find(nets.begin(), nets.end(), p.net) - nets.begin());
                auto at = std::find(used.begin(), used.end(), i);
                if (at == used.end()) {
                    used.push_back(i);
                    at = used.end() - 1;
                }
                out.network = static_cast<int>(at - used.begin());
            }
            return out;
        };
        for (const ArenaGameRange& r : ranges) job.ranges.push_back({seat(r.a), seat(r.b), r.seed, r.first, r.count});
        for (size_t i : used) {
            job.networks.push_back({netFileName(files[i].second), files[i].second});
            ps.remote.networks.push_back(files[i]);
        }
        char base[64];
        std::snprintf(base, sizeof base, "arena_g%04d_w%d_%d", g, w, assigned[k].first);
        ps.remote.base = base;
        ps.remote.jobBytes = serializeArenaJob(job);
        const std::string jobBytes = ps.remote.jobBytes;
        ps.remote.accept = [&results, k, jobBytes](const std::string& path, SliceReport& report) {
            ArenaSliceResult result = readArenaResult(path, jobBytes, buildId());
            report.fingerprint = result.fingerprint;
            report.build = shortOf(result.buildId);
            report.stats = result.stats;
            results[k] = std::move(result.games);
        };
    }
    std::vector<SliceReport> reports =
        runSlices(g, slices, cfg_.arenaEngines, workDir, "arena wave " + std::to_string(w) + " games");

    // Merge by wave index, each filled exactly once, then each match's
    // result from its games in index order.
    std::vector<ArenaGameResult> games(static_cast<size_t>(total));
    std::vector<char> filled(static_cast<size_t>(total), 0);
    for (size_t k = 0; k < n; ++k) {
        if (results[k].size() != static_cast<size_t>(assigned[k].count))
            throw std::logic_error("arena slice of " + reports[k].engine +
                                   " returned the wrong number of games");
        for (int j = 0; j < assigned[k].count; ++j) {
            const size_t i = static_cast<size_t>(assigned[k].first + j);
            if (i >= games.size() || filled[i])
                throw std::logic_error("arena game " + std::to_string(i) + " assigned twice or out of range");
            filled[i] = 1;
            games[i] = std::move(results[k][static_cast<size_t>(j)]);
        }
    }
    for (size_t i = 0; i < filled.size(); ++i)
        if (!filled[i]) throw std::logic_error("arena game " + std::to_string(i) + " was not played");
    std::vector<MatchResult> out;
    size_t at = 0;
    for (const ArenaMatchSpec& m : wave) {
        const auto from = games.begin() + static_cast<std::ptrdiff_t>(at);
        out.push_back(matchResult(std::vector<ArenaGameResult>(from, from + m.games)));
        at += static_cast<size_t>(m.games);
    }
    if (anyRemote) {
        std::error_code ec;
        std::filesystem::remove_all(netDir, ec);
        finishWorkDir(workDir, failedSlices_);
        // arena-work itself, once its last wave directory is gone.
        if (std::filesystem::is_empty(std::filesystem::path(workDir).parent_path(), ec))
            std::filesystem::remove(std::filesystem::path(workDir).parent_path(), ec);
    }
    for (SliceReport& r : reports) {
        r.wave = w;
        arenaReports_.push_back(r);
    }
    return out;
}

int arenaWaveNetworks(const Config& cfg) {
    const int calibration = 1 + static_cast<int>(cfg.referenceNetworks.size());
    const int round = 3 + cfg.arenaOpponents + (cfg.gating ? 1 : 0);
    return std::max(calibration, round);
}

std::string runIdOf(const std::string& runDir) {
    namespace fs = std::filesystem;
    const fs::path absolute = fs::absolute(runDir).lexically_normal();
    std::string name = absolute.filename().string();
    if (name.empty()) name = absolute.parent_path().filename().string();   // a trailing '/'
    for (char& c : name)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
              c == '_' || c == '.' || c == '-'))
            c = '_';
    if (name.empty() || name[0] == '-' || name[0] == '.') name = "run" + name;
    char hash[24];
    std::snprintf(hash, sizeof hash, "%016llx",
                  static_cast<unsigned long long>(fnv1a64(absolute.string())));
    return name + "-" + hash;
}

// ---- transports ----------------------------------------------------------------


std::vector<std::string> sshArgv(const MachineSettings& machine,
                                 const std::vector<std::string>& args) {
    requireSafe(safeWord(machine.exe), "exe", machine.exe);
    std::vector<std::string> words = {"nice", "-n", std::to_string(machine.nice), machine.exe};
    words.insert(words.end(), args.begin(), args.end());
    return sshCommand(machine, words);
}

std::vector<std::string> scpArgv(const MachineSettings& machine, const std::string& local,
                                 const std::string& remote, bool upload) {
    requireSafe(safeWord(machine.ssh), "ssh target", machine.ssh);
    requireSafe(safeWord(remote), "remote path", remote);
    // The local path never reaches a shell, so any characters are fine;
    // absolute, so neither a ':' (read as host:path) nor a '-' can mislead scp.
    const std::string here = std::filesystem::absolute(local).string();
    const std::string far = machine.ssh + ":" + remote;
    std::vector<std::string> argv = {"scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"};
    argv.push_back(upload ? here : far);
    argv.push_back(upload ? far : here);
    return argv;
}

SshTransport::SshTransport(MachineSettings machine, std::string log)
    : m_(std::move(machine)), log_(std::move(log)) {}

// The failure and the last line scp or ssh wrote about it.
// The failure as messages give it: no ssh or scp output, which can name
// hosts and addresses; that stays in the log.
std::string SshTransport::why(const ProcessResult& r) const { return brief(r); }

ProcessResult SshTransport::command(const std::vector<std::string>& words, double timeout) {
    return runProcess(sshCommand(m_, words), log_, timeout);
}

void SshTransport::upload(const std::string& local, const std::string& remote) {
    const ProcessResult dir = command({"mkdir", "-p", parentOf(remote)}, commandTimeout_);
    if (dir.exitCode != 0)
        throw SliceError(m_.alias + ": cannot create the work directory (ssh " + why(dir) + ")");
    // To a temporary name, then renamed: a cut upload never leaves a
    // partial file under the final name (only, at worst, a .part file).
    const std::string part = partName(remote);
    const ProcessResult copy = runProcess(scpArgv(m_, local, part, true), log_, kTransferTimeout);
    if (copy.exitCode != 0)
        throw SliceError(m_.alias + ": upload of " + fileName(remote) + " failed (scp " +
                         why(copy) + ")");
    const ProcessResult move = command({"mv", "-f", part, remote}, commandTimeout_);
    if (move.exitCode != 0)
        throw SliceError(m_.alias + ": cannot rename " + fileName(part) + " (ssh " + why(move) +
                         ")");
}

bool SshTransport::exists(const std::string& remote) {
    const ProcessResult r = command({"test", "-f", remote}, commandTimeout_);
    if (r.exitCode == 0) return true;
    if (r.exitCode == 1) return false;
    throw SliceError(m_.alias + ": cannot reach the machine (ssh " + why(r) + ")");
}

std::vector<std::string> SshTransport::list(const std::string& remoteDir) {
    static std::atomic<unsigned long> counter{0};
    const std::string name = "azgo_ls_" + std::to_string(getpid()) + "_" + std::to_string(++counter);
    const std::string out = (std::filesystem::temp_directory_path() / name).string();
    std::vector<std::string> names;
    const ProcessResult r =
        runProcess(sshCommand(m_, {"ls", "-1A", remoteDir}), out, commandTimeout_);
    if (r.exitCode == 0) {
        std::ifstream in(out, std::ios::binary);
        for (std::string line; std::getline(in, line);)
            if (!line.empty()) names.push_back(line);
    }
    std::filesystem::remove(out);
    if (r.exitCode == 255 || !r.error.empty())
        throw SliceError(m_.alias + ": cannot reach the machine (ssh " + brief(r) + ")");
    return names;   // another exit: no such directory yet, nothing in it
}

ProcessResult SshTransport::run(const std::vector<std::string>& args, const std::string& log,
                                double timeout, const std::atomic<bool>* cancel) {
    ProcessOptions options;
    options.log = log;
    options.timeout = timeout;
    options.cancel = cancel;
    options.holdStdin = true;   // ssh forwards its end of file to the remote selfplay
    return runProcess(sshArgv(m_, args), options);
}

void SshTransport::download(const std::string& remote, const std::string& local) {
    const ProcessResult copy =
        runProcess(scpArgv(m_, local, remote, false), log_, kTransferTimeout);
    if (copy.exitCode != 0)
        throw SliceError(m_.alias + ": download of " + fileName(remote) + " failed (scp " +
                         why(copy) + ")");
}

std::unique_ptr<StreamChannel> SshTransport::openStream(const std::vector<std::string>& args,
                                                      const std::string& log) {
    // ssh forwards stdin and stdout byte for byte, and its end of file;
    // its own messages and the remote's stderr go to the log.
    return std::make_unique<ChildProcess>(sshArgv(m_, args), log);
}

void SshTransport::remove(const std::vector<std::string>& remote) {
    if (remote.empty()) return;
    std::vector<std::string> words = {"rm", "-f"};
    words.insert(words.end(), remote.begin(), remote.end());
    const ProcessResult r = command(words, commandTimeout_);
    if (r.exitCode != 0)
        throw SliceError(m_.alias + ": cannot remove work files (ssh " + why(r) + ")");
}

bool SshTransport::reachable() {
    std::vector<std::string> argv = sshCommand(m_, {"true"});
    // Before the usual ConnectTimeout=10: ssh takes an option's first value.
    argv.insert(argv.begin() + 1, {"-o", "ConnectTimeout=3"});
    return runProcess(argv, log_, 15).exitCode == 0;
}

void SshTransport::removeDirIfEmpty(const std::string& remoteDir) {
    command({"rmdir", remoteDir}, commandTimeout_);   // fails, harmlessly, unless empty
}

void SshTransport::killJob(const std::string& pattern) {
    command({"pkill", "-f", pattern}, commandTimeout_);   // best effort; 1 = nothing matched
}

SubprocessTransport::SubprocessTransport(MachineSettings machine) : m_(std::move(machine)) {}

void SubprocessTransport::upload(const std::string& local, const std::string& remote) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path to = expandHome(remote);
    const fs::path part = partName(to.string());
    fs::create_directories(to.parent_path(), ec);
    if (!ec) fs::copy_file(local, part, fs::copy_options::overwrite_existing, ec);
    if (!ec) fs::rename(part, to, ec);   // atomic: never a partial file under the final name
    if (ec)
        throw SliceError(m_.alias + ": upload of " + fileName(remote) + " failed: " + ec.message());
}

bool SubprocessTransport::exists(const std::string& remote) {
    return std::filesystem::is_regular_file(expandHome(remote));
}

std::vector<std::string> SubprocessTransport::list(const std::string& remoteDir) {
    std::vector<std::string> names;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(expandHome(remoteDir), ec))
        if (entry.is_regular_file()) names.push_back(entry.path().filename().string());
    return names;
}

ProcessResult SubprocessTransport::run(const std::vector<std::string>& args, const std::string& log,
                                       double timeout, const std::atomic<bool>* cancel) {
    std::vector<std::string> argv;
    if (m_.nice > 0) argv = {"nice", "-n", std::to_string(m_.nice)};
    argv.push_back(expandHome(m_.exe));
    for (const std::string& a : args) argv.push_back(a[0] == '~' ? expandHome(a) : a);
    ProcessOptions options;
    options.log = log;
    options.timeout = timeout;
    options.cancel = cancel;
    options.holdStdin = true;
    return runProcess(argv, options);
}

void SubprocessTransport::download(const std::string& remote, const std::string& local) {
    std::error_code ec;
    std::filesystem::copy_file(expandHome(remote), local,
                               std::filesystem::copy_options::overwrite_existing, ec);
    if (ec)
        throw SliceError(m_.alias + ": download of " + fileName(remote) + " failed: " +
                         ec.message());
}

std::unique_ptr<StreamChannel> SubprocessTransport::openStream(const std::vector<std::string>& args,
                                                             const std::string& log) {
    std::vector<std::string> argv;
    if (m_.nice > 0) argv = {"nice", "-n", std::to_string(m_.nice)};
    argv.push_back(expandHome(m_.exe));
    for (const std::string& a : args) argv.push_back(a[0] == '~' ? expandHome(a) : a);
    return std::make_unique<ChildProcess>(argv, log);
}

void SubprocessTransport::remove(const std::vector<std::string>& remote) {
    for (const std::string& r : remote) {
        std::error_code ec;
        std::filesystem::remove(expandHome(r), ec);
    }
}

bool SubprocessTransport::reachable() {
    return access(expandHome(m_.exe).c_str(), X_OK) == 0;
}

void SubprocessTransport::removeDirIfEmpty(const std::string& remoteDir) {
    std::error_code ec;
    if (std::filesystem::is_empty(expandHome(remoteDir), ec) && !ec)
        std::filesystem::remove(expandHome(remoteDir), ec);
}

void SubprocessTransport::killJob(const std::string& pattern) {
    runProcess({"pkill", "-f", pattern}, "", commandTimeout_);   // best effort
}

ProcessResult InProcessTransport::run(const std::vector<std::string>& given, const std::string& log,
                                      double timeout, const std::atomic<bool>* cancel) {
    (void)timeout;
    (void)cancel;
    // The app's flag for a held stdin; a call has no stdin to watch.
    std::vector<std::string> args;
    for (const std::string& a : given)
        if (a != "--exit-on-stdin-eof") args.push_back(a);
    // What the binary answers to --version and --fingerprint, for preflight.
    auto append = [&](const std::string& text) {
        if (log.empty()) return;
        std::ofstream out(log, std::ios::app | std::ios::binary);
        out << text;
    };
    ProcessResult answer;
    if (args == std::vector<std::string>{"--version"}) {
        append("selfplay " + shortBuildId() + " (" + buildOptions() + ")\nbuild " + buildId() +
               "\n");
        answer.exitCode = 0;
        return answer;
    }
    if (args.size() == 3 && args[0] == "--fingerprint" && args[1] == "--backend") {
        try {
            append(engineFingerprint(args[2]) + "\n");
            answer.exitCode = 0;
        } catch (const std::exception& e) {
            append(std::string("selfplay: ") + e.what() + "\n");
            answer.exitCode = 1;
        }
        return answer;
    }
    std::map<std::string, std::string> options;
    for (size_t i = 0; i + 1 < args.size(); i += 2) options[args[i]] = expandHome(args[i + 1]);
    ProcessResult result;
    if (args.size() % 2 != 0 || !options.count("--job") || !options.count("--out") ||
        !options.count("--backend")) {
        result.exitCode = 2;
        return result;
    }
    std::ostringstream err;
    auto number = [&](const char* key) {
        return options.count(key) ? std::atoi(options[key].c_str()) : 0;
    };
    result.exitCode = runJob(options["--job"], options["--out"], options["--backend"],
                             number("--workers"), number("--inference-batch"), err);
    if (!log.empty()) {
        std::ofstream out(log, std::ios::app | std::ios::binary);
        out << err.str();
    }
    return result;
}

namespace {

// InProcessTransport's stream: runStreamJob on a thread, its stdin and
// stdout two pipes, its stderr appended to the log when it ends.
class InProcessChannel : public StreamChannel {
public:
    InProcessChannel(std::map<std::string, std::string> options, std::string log) : log_(std::move(log)) {
        if (pipe(toChild_) != 0 || pipe(fromChild_) != 0)
            throw std::runtime_error("InProcessChannel: pipe failed");
        auto number = [&](const char* key) {
            return options.count(key) ? std::atoi(options[key].c_str()) : 0;
        };
        const std::string job = options["--job"], backend = options["--backend"];
        const int workers = number("--workers"), batch = number("--inference-batch");
        thread_ = std::thread([this, job, backend, workers, batch] {
            std::ostringstream err;
            status_ = runStreamJob(job, backend, workers, batch, toChild_[0], fromChild_[1], err);
            if (!log_.empty()) {
                std::ofstream out(log_, std::ios::app | std::ios::binary);
                out << err.str();
            }
            close(fromChild_[1]);   // our reader's end of file
        });
    }
    ~InProcessChannel() override {
        stop();
        wait();
        close(fromChild_[0]);
        close(toChild_[0]);
    }
    bool send(const std::string& text) override {
        std::lock_guard<std::mutex> lock(mutex_);
        return toChild_[1] >= 0 && writeAll(toChild_[1], text);
    }
    long receive(char* buf, size_t size, double timeoutSeconds) override {
        pollfd p{fromChild_[0], POLLIN, 0};
        const int ready = poll(&p, 1, static_cast<int>(std::max(0.0, timeoutSeconds) * 1000));
        if (ready <= 0) return -1;
        const ssize_t got = read(fromChild_[0], buf, size);
        return got < 0 ? 0 : static_cast<long>(got);
    }
    void stop() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (toChild_[1] >= 0) close(toChild_[1]);   // end of file: it stops at once
        toChild_[1] = -1;
        stopped_ = true;
    }
    ProcessResult wait() override {
        if (thread_.joinable()) {
            // Whatever it still writes is read and dropped, so it can end.
            char buf[4096];
            while (read(fromChild_[0], buf, sizeof buf) > 0) {
            }
            thread_.join();
        }
        ProcessResult r;
        r.exitCode = stopped_ ? -1 : status_.load();
        r.cancelled = stopped_;
        return r;
    }
private:
    int toChild_[2] = {-1, -1}, fromChild_[2] = {-1, -1};
    std::string log_;
    std::thread thread_;
    std::mutex mutex_;
    std::atomic<int> status_{-1};
    std::atomic<bool> stopped_{false};
};

} // namespace

std::unique_ptr<StreamChannel> InProcessTransport::openStream(const std::vector<std::string>& args,
                                                            const std::string& log) {
    std::map<std::string, std::string> options;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--stream") continue;
        if (i + 1 < args.size()) {
            options[args[i]] = expandHome(args[i + 1]);
            ++i;
        }
    }
    return std::make_unique<InProcessChannel>(options, log);
}

std::unique_ptr<Transport> makeTransport(const MachineSettings& machine, const std::string& log) {
    if (machine.ssh.empty()) return std::make_unique<SubprocessTransport>(machine);
    return std::make_unique<SshTransport>(machine, log);
}

std::string manifestPath(const std::string& runDir) {
    return (std::filesystem::path(runDir) / "selfplay-engines.csv").string();
}

std::string arenaManifestPath(const std::string& runDir) {
    return (std::filesystem::path(runDir) / "arena-engines.csv").string();
}

namespace {

// A manifest's header and its rows of generations before g; *dropped
// counts the others.
std::string manifestBefore(const std::string& path, const char* header, int g, int* dropped = nullptr) {
    std::string out = header;
    std::ifstream in(path, std::ios::binary);
    if (in) {
        std::string line;
        std::getline(in, line);   // the header
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            // Earlier generations stay; g and later (a redone or abandoned
            // generation) go.
            if (std::atoi(line.c_str()) < g) out += line + "\n";
            else if (dropped) ++*dropped;
        }
    }
    return out;
}

void pruneManifestFile(const std::string& path, const char* header, int g) {
    if (!std::filesystem::exists(path)) return;
    int dropped = 0;
    const std::string kept = manifestBefore(path, header, g, &dropped);
    if (dropped > 0) writeFileAtomically(path, kept);
}

// A report's columns from planned on: planned, engine, fingerprint, build,
// seconds, positions, evaluations, note.
std::string reportColumns(const SliceReport& r) {
    char numbers[96];
    std::snprintf(numbers, sizeof numbers, "%.1f,%lld,%lld", r.stats.seconds, r.stats.positions,
                  r.stats.evaluations);
    return csvField(r.planned) + "," + csvField(r.engine) + "," + csvField(r.fingerprint, true) + "," +
           csvField(r.build) + "," + numbers + "," + csvField(r.note);
}

} // namespace

void pruneManifest(const std::string& runDir, int g) {
    pruneManifestFile(manifestPath(runDir), kManifestHeader, g);
}

void pruneArenaManifest(const std::string& runDir, int g) {
    pruneManifestFile(arenaManifestPath(runDir), kArenaManifestHeader, g);
}

void writeManifest(const std::string& runDir, int g, const std::vector<SliceReport>& reports) {
    std::string out = manifestBefore(manifestPath(runDir), kManifestHeader, g);
    for (const SliceReport& r : reports)
        out += std::to_string(g) + "," + (r.dynamic ? std::string() : std::to_string(r.slice.first)) + "," +
               std::to_string(r.slice.count) + "," + reportColumns(r) + "\n";
    writeFileAtomically(manifestPath(runDir), out);
}

std::string gamesManifestPath(const std::string& runDir) {
    return (std::filesystem::path(runDir) / "selfplay-games.csv").string();
}

void writeGamesManifest(const std::string& runDir, int g, const std::vector<std::string>& assignment) {
    std::string out = manifestBefore(gamesManifestPath(runDir), kGamesManifestHeader, g);
    for (size_t i = 0; i < assignment.size(); ++i)
        out += std::to_string(g) + "," + std::to_string(i) + "," + csvField(assignment[i]) + "\n";
    writeFileAtomically(gamesManifestPath(runDir), out);
}

void pruneGamesManifest(const std::string& runDir, int g) {
    pruneManifestFile(gamesManifestPath(runDir), kGamesManifestHeader, g);
}

void writeArenaManifest(const std::string& runDir, int g, const std::vector<SliceReport>& reports) {
    std::string out = manifestBefore(arenaManifestPath(runDir), kArenaManifestHeader, g);
    for (const SliceReport& r : reports)
        out += std::to_string(g) + "," + std::to_string(r.wave) + "," + std::to_string(r.slice.first) + "," +
               std::to_string(r.slice.count) + "," + reportColumns(r) + "\n";
    writeFileAtomically(arenaManifestPath(runDir), out);
}

std::string enginesLine(const std::vector<SliceReport>& reports, const std::string& label) {
    std::string out = label + ":";
    for (size_t k = 0; k < reports.size(); ++k) {
        const SliceReport& r = reports[k];
        const double rate =
            r.stats.seconds > 0 ? r.stats.evaluations / r.stats.seconds / 1000.0 : 0.0;
        char item[64];
        std::snprintf(item, sizeof item, " %d %.0fs %.1fk%s", r.slice.count, r.stats.seconds, rate,
                      k == 0 ? " ev/s" : "");
        out += (k == 0 ? " " : " | ") + r.engine + item;
    }
    return out;
}

namespace distributed_probe {
void setRemoteBuildId(const std::string& id) {
    std::lock_guard<std::mutex> lock(gProbeMutex);
    gRemoteBuild = id;
}
void delaySlice(const std::string& engine, double seconds) {
    std::lock_guard<std::mutex> lock(gProbeMutex);
    gDelays[engine] = seconds;
}
void useInProcessTransport(bool on) {
    std::lock_guard<std::mutex> lock(gProbeMutex);
    gInProcess = on;
}
void failSlice(const std::string& engine, int times, Fail mode) {
    std::lock_guard<std::mutex> lock(gProbeMutex);
    gFailures[engine] = {times, mode};
}
void failSession(const std::string& engine, int times, int afterGames) {
    std::lock_guard<std::mutex> lock(gProbeMutex);
    gSessionFailures[engine] = {times, afterGames};
}
void forceAssignment(int g, const std::vector<std::string>& engineOfGame) {
    std::lock_guard<std::mutex> lock(gProbeMutex);
    gForced[g] = engineOfGame;
}
void reset() {
    std::lock_guard<std::mutex> lock(gProbeMutex);
    gFailures.clear();
    gSessionFailures.clear();
    gForced.clear();
    gRemoteBuild.clear();
    gDelays.clear();
    gInProcess = false;
}
} // namespace distributed_probe

} // namespace azgo
