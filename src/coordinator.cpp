#include "azgo/coordinator.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <utility>
#include "azgo/backend.hpp"
#include "azgo/build_info.hpp"
#include "azgo/inference_hub.hpp"
#include "azgo/metal.hpp"
#include "azgo/parallel.hpp"
#include "azgo/run_state.hpp"
#include "azgo/serialize.hpp"

namespace azgo {

namespace {

const char* const kManifestHeader =
    "generation,first,count,planned,engine,fingerprint,build,seconds,positions,evaluations,note\n";

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
// listed, or the run's own backend.
std::string fallbackEngine(const Config& cfg) {
    for (const EngineSpec& e : cfg.selfplayEngines)
        if (e.machine == "local") return e.name();
    return "local/" + cfg.backend;
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

// Each engine's fingerprint in the manifest's last row where it played.
std::map<std::string, std::string> lastFingerprints(const std::string& runDir) {
    std::map<std::string, std::string> out;
    std::ifstream in(manifestPath(runDir), std::ios::binary);
    std::string line;
    std::getline(in, line);   // the header
    while (std::getline(in, line)) {
        const std::vector<std::string> f = csvFields(line);
        if (f.size() == 11) out[f[4]] = f[5];
    }
    return out;
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

} // namespace

SelfPlayCoordinator::SelfPlayCoordinator(const Config& cfg, const std::string& runDir,
                                         HostsFile hosts, std::ostream* log)
    : cfg_(cfg), runDir_(runDir), runId_(runIdOf(runDir)), hosts_(std::move(hosts)), log_(log) {
    if (cfg_.selfplayEngines.empty())
        throw std::logic_error("SelfPlayCoordinator: selfplay_engines is empty (the default path)");
}

SelfPlayCoordinator::~SelfPlayCoordinator() {
    for (const auto& [alias, state] : machines_) {
        if (state->uploadedNet.empty()) continue;
        try {
            const MachineSettings m = hosts_.settings(alias);
            const std::unique_ptr<Transport> t = transportFor(m, "");
            t->setCommandTimeout(kCleanupTimeout);
            t->remove({state->uploadedNet});
            t->removeDirIfEmpty(remoteDir(m));   // the run's directory, now empty
        } catch (const std::exception&) {
        }
    }
}

std::unique_ptr<Transport> SelfPlayCoordinator::transportFor(const MachineSettings& m,
                                                             const std::string& log) const {
    if (m.ssh.empty() && probeInProcess()) return std::make_unique<InProcessTransport>(m);
    return makeTransport(m, log);
}

SelfPlayCoordinator::MachineState& SelfPlayCoordinator::machine(const std::string& alias) {
    return *machines_.at(alias);   // a lookup only: no slice thread inserts
}

std::string SelfPlayCoordinator::remoteDir(const MachineSettings& m) const {
    return m.workdir + "/" + runId_;
}

void SelfPlayCoordinator::unreachable(const std::string& alias, const std::string& why) {
    if (cfg_.selfplayFallback == "stop")
        throw ConfigError("selfplay_engines: " + alias + " is unreachable (" + why +
                          "); with selfplay_fallback = stop the run cannot start");
    unreachable_.insert(alias);
    if (log_)
        *log_ << "warning: " << alias << " unreachable (" << why << "); its slices fall back to "
              << fallbackEngine(cfg_) << "\n";
}

void SelfPlayCoordinator::preflight() {
    // Local engines: the backend runs here, and the metal hubs fit.
    int metalHubs = 0;
    std::vector<std::string> remotes;   // aliases, in list order
    for (const EngineSpec& e : cfg_.selfplayEngines) {
        if (e.machine != "local") {
            hosts_.settings(e.machine);   // ConfigError for an alias the hosts file lacks
            if (std::find(remotes.begin(), remotes.end(), e.machine) == remotes.end())
                remotes.push_back(e.machine);
            continue;
        }
        checkBackendAvailable(e.backend);
        if (e.backend == "metal") ++metalHubs;
        if (!fingerprints_.count(e.name())) fingerprints_[e.name()] = engineFingerprint(e.backend);
    }
    const MachineSettings local = hosts_.settings("local");
    const int batch = local.inferenceBatch > 0 ? local.inferenceBatch : cfg_.inferenceBatch;
    if (batch < cfg_.mctsBatch)
        throw ConfigError("hosts file: local inference_batch (" + std::to_string(batch) +
                          ") is below mcts_batch (" + std::to_string(cfg_.mctsBatch) + ")");
#ifdef AZ_USE_METAL
    // Every metal item is its own hub with two slots, all at once.
    if (metalHubs > 0)
        checkMetalInferenceMemory(cfg_.netConfig(), batch, metalHubs, kInferenceSlots);
#else
    (void)metalHubs;
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
                throw ConfigError("selfplay_engines: " + remotes[a] + " and " + remotes[b] +
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
            throw ConfigError("selfplay_engines: " + alias + " runs build " + id +
                              " but this coordinator is build " + buildId() +
                              "; bring it to this commit with tools/cluster_sync.sh");
        remoteBuilds_[alias] = id;
        for (const EngineSpec& e : cfg_.selfplayEngines) {
            if (e.machine != alias || fingerprints_.count(e.name())) continue;
            const ProcessResult r =
                t->run({"--fingerprint", "--backend", e.backend}, log, kPreflightTimeout);
            const std::string line = lastLine(log);
            std::filesystem::remove(log);
            if (r.exitCode == 1)   // selfplay ran and refused: the backend cannot run there
                throw ConfigError("selfplay_engines: " + e.name() + ": " + alias +
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
    // since the manifest last recorded it.
    const std::map<std::string, std::string> recorded = lastFingerprints(runDir_);
    std::vector<std::string> printed;
    for (const EngineSpec& e : cfg_.selfplayEngines) {
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
        auto was = recorded.find(e.name());
        if (log_ && was != recorded.end() && was->second != f->second)
            *log_ << "note: the fingerprint of " << e.name() << " changed from \"" << was->second
                  << "\"; from here on its games differ in the last bits from before\n";
    }
}

std::vector<GameRecord> SelfPlayCoordinator::attemptRemote(int attempt, int g, const Slice& slice,
                                                           const EngineSpec& e,
                                                           const std::string& networkPath,
                                                           uint64_t networkHash,
                                                           const SelfPlayConfig& sp,
                                                           const std::string& workDir,
                                                           const std::atomic<bool>& stop,
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

    // The network, uploaded by this coordinator once per machine and hash
    // (a file already there may be a stranger's or cut short: it is
    // replaced, atomically); the one it replaces goes. At the first contact
    // with a machine, what an earlier coordinator of this run left there (a
    // crash: networks, jobs, results, parts of uploads) goes first.
    char netName[48];
    std::snprintf(netName, sizeof netName, "net_%016llx.azn",
                  static_cast<unsigned long long>(networkHash));
    const std::string remoteNet = dir + "/" + netName;
    checkStop();
    {
        MachineState& state = machine(e.machine);
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.contacted) {
            std::vector<std::string> stale;
            for (const std::string& name : t->list(dir)) stale.push_back(dir + "/" + name);
            t->remove(stale);
            state.contacted = true;
        }
        std::string& uploaded = state.uploadedNet;
        if (uploaded != remoteNet) {
            t->upload(networkPath, remoteNet);
            if (!uploaded.empty()) t->remove({uploaded});
            uploaded = remoteNet;
        }
    }

    // The job: exactly what determines the slice's games. The file names
    // carry the attempt, so a straggler of attempt 1 cannot be taken for
    // attempt 2.
    SelfPlayJob job;
    job.seed = cfg_.seed;
    job.generation = g;
    job.first = slice.first;
    job.count = slice.count;
    job.boardSize = cfg_.boardSize;
    job.sp = sp;
    job.networkFile = netName;
    job.networkHash = networkHash;
    job.buildId = buildId();
    job.inferenceBatch = cfg_.inferenceBatch;
    const std::string jobBytes = serializeJob(job);
    char base[64];
    std::snprintf(base, sizeof base, "g%04d_%d_a%d", g, slice.first, attempt);
    const std::string localJob = workDir + "/" + base + ".job";
    const std::string localResult = workDir + "/" + base + ".result";
    const std::string remoteJob = dir + "/" + base + ".job";
    const std::string remoteResult = dir + "/" + base + ".result";
    const std::string pattern = runId_ + "/" + base + ".job";
    try {
        writeFileAtomically(localJob, jobBytes);
        checkStop();
        t->upload(localJob, remoteJob);

        // --exit-on-stdin-eof: the remote stops if this process or the
        // connection goes away (the transport holds its stdin).
        std::vector<std::string> args = {"--job",     remoteJob, "--out",
                                         remoteResult, "--backend", e.backend,
                                         "--exit-on-stdin-eof"};
        // Worker counts the hosts file sets; otherwise the remote's own rule.
        const int workers = e.backend == "metal" ? m.metalWorkers : m.cpuWorkers;
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
            // upload): forget it, so the next attempt uploads it again.
            if (line.find("network") != std::string::npos) {
                MachineState& state = machine(e.machine);
                std::lock_guard<std::mutex> lock(state.mutex);
                if (state.uploadedNet == remoteNet) state.uploadedNet.clear();
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
        SliceResult result = readSliceResult(localResult, jobBytes, buildId());
        t->remove({remoteJob, remoteResult});
        std::filesystem::remove(localJob);
        std::filesystem::remove(localResult);
        report.fingerprint = result.fingerprint;
        report.build = shortOf(result.buildId);
        report.stats = result.stats;
        return std::move(result.records);
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

std::vector<GameRecord> SelfPlayCoordinator::playRemote(int g, const Slice& slice,
                                                        const EngineSpec& e,
                                                        const std::string& networkPath,
                                                        uint64_t networkHash, const Network& net,
                                                        const SelfPlayConfig& sp,
                                                        const std::string& workDir,
                                                        const std::atomic<bool>& stop,
                                                        SliceReport& report) {
    // Two attempts on the planned engine: a transient failure costs time,
    // not reproducibility. A machine that does not answer at all costs
    // only a quick probe: one known to be down this generation, or found
    // down after a failure, goes straight to the fallback, and one that was
    // down earlier is probed before it is tried.
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
        logLine("slice " + std::to_string(slice.first) + ": " + reason + "; not tried");
        failedSlices_ = true;
    } else {
        for (int attempt = 1; attempt <= 2; ++attempt) {
            try {
                return attemptRemote(attempt, g, slice, e, networkPath, networkHash, sp, workDir,
                                     stop, report);
            } catch (const SliceCancelled&) {
                throw;
            } catch (const std::exception& ex) {
                reason = ex.what();
                failedSlices_ = true;
                logLine("slice " + std::to_string(slice.first) + ", attempt " +
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
    const std::string fallback = fallbackEngine(cfg_);
    const std::string backend = fallback.substr(fallback.find('/') + 1);
    const MachineSettings local = hosts_.settings("local");
    EngineRun run;
    run.backend = backend;
    run.workers = engineWorkers(local, backend, slice.count, resolveWorkers(0));
    run.inferenceBatch = local.inferenceBatch > 0 ? local.inferenceBatch : cfg_.inferenceBatch;
    run.qos = backend == "metal" ? CpuQos::Default : local.cpuQos;
    run.cancel = &stop;
    report.engine = fallback;
    report.fingerprint = engineFingerprint(backend);
    report.build = shortBuildId();
    report.note = "fallback: " + reason;
    if (log_) {
        std::lock_guard<std::mutex> lock(logMutex_);
        *log_ << "note: generation " << g << ": games " << slice.first << "-"
              << slice.first + slice.count - 1 << " (" << e.name() << ") played by " << fallback
              << ": " << reason
              << "; from here on results differ in the last bits from a run without the failure\n";
    }
    return playSelfPlayGames(net, sp, cfg_.seed, g, slice.first, slice.count, run, &report.stats);
}

std::vector<GameRecord> SelfPlayCoordinator::play(int g, const std::string& networkPath,
                                                  const Network& net, const SelfPlayConfig& sp,
                                                  int games) {
    const std::vector<Slice> slices = assignGames(cfg_.selfplayEngines, games);
    const int cores = resolveWorkers(0);
    const size_t n = slices.size();
    std::vector<SliceReport> reports(n);
    std::vector<EngineRun> runs(n);
    bool anyRemote = false;
    for (size_t k = 0; k < n; ++k) {
        const EngineSpec& e = cfg_.selfplayEngines[static_cast<size_t>(slices[k].engine)];
        const MachineSettings m = hosts_.settings(e.machine);
        reports[k].slice = slices[k];
        reports[k].planned = reports[k].engine = e.name();
        if (!m.local()) {
            anyRemote = true;
            // Every remote machine's entry exists before the threads start.
            if (!machines_.count(e.machine)) {
                machines_[e.machine] = std::make_unique<MachineState>();
                machines_[e.machine]->suspect = unreachable_.count(e.machine) > 0;
            }
            machines_[e.machine]->down = false;   // tried afresh every generation
            continue;
        }
        runs[k].backend = e.backend;
        runs[k].workers = engineWorkers(m, e.backend, slices[k].count, cores);
        runs[k].inferenceBatch = m.inferenceBatch > 0 ? m.inferenceBatch : cfg_.inferenceBatch;
        // The GPU, which sets the pace, waits on the metal engine's workers:
        // they keep the default class. CPU engines take the machine's.
        runs[k].qos = e.backend == "metal" ? CpuQos::Default : m.cpuQos;
        if (!fingerprints_.count(e.name())) fingerprints_[e.name()] = engineFingerprint(e.backend);
        reports[k].fingerprint = fingerprints_[e.name()];
        reports[k].build = shortBuildId();
    }

    // The BLAS backend sets its environment once, with setenv, which is not
    // thread-safe: do that here, before any engine thread runs.
    for (const EngineRun& run : runs)
        if (run.backend == "blas") {
            makeBackend("blas", 1);
            break;
        }
    // Remote slices' work files: this generation's directory, made afresh.
    char genDir[32];
    std::snprintf(genDir, sizeof genDir, "gen_%04d", g);
    const std::string workDir =
        (std::filesystem::path(runDir_) / "selfplay-work" / genDir).string();
    uint64_t networkHash = 0;
    failedSlices_ = false;
    if (anyRemote) {
        std::filesystem::remove_all(workDir);
        std::filesystem::create_directories(workDir);
        networkHash = fileHash(networkPath);
    }

    // One thread per slice, all at once. Each local engine has its own
    // evaluators, so a slice's games are what playSelfPlayGames would give
    // alone. The first slice to fail sets `stop`, and the local ones stop
    // at their next network evaluation: the generation will not be
    // committed anyway.
    std::atomic<bool> stop{false};
    for (EngineRun& run : runs) run.cancel = &stop;
    std::vector<std::vector<GameRecord>> results(n);
    std::vector<std::exception_ptr> errors(n);
    std::vector<std::thread> threads;
    std::exception_ptr launchError;
    try {
        for (size_t k = 0; k < n; ++k)
            threads.emplace_back([&, k] {
                try {
                    const EngineSpec& e =
                        cfg_.selfplayEngines[static_cast<size_t>(slices[k].engine)];
                    if (e.machine == "local")
                        results[k] = playSelfPlayGames(net, sp, cfg_.seed, g, slices[k].first,
                                                       slices[k].count, runs[k], &reports[k].stats);
                    else
                        results[k] = playRemote(g, slices[k], e, networkPath, networkHash, net,
                                                sp, workDir, stop, reports[k]);
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

    // Merge by game index: records[first + k] = slice's k-th, each index
    // filled exactly once, whatever order the slices finished in.
    std::vector<GameRecord> records(static_cast<size_t>(games));
    std::vector<char> filled(static_cast<size_t>(games), 0);
    for (size_t k = 0; k < n; ++k) {
        if (results[k].size() != static_cast<size_t>(slices[k].count))
            throw std::logic_error("slice of " + reports[k].engine +
                                   " returned the wrong number of games");
        for (int j = 0; j < slices[k].count; ++j) {
            const size_t i = static_cast<size_t>(slices[k].first + j);
            if (i >= records.size() || filled[i])
                throw std::logic_error("game " + std::to_string(i) +
                                       " assigned twice or out of range");
            filled[i] = 1;
            records[i] = std::move(results[k][static_cast<size_t>(j)]);
        }
    }
    for (size_t i = 0; i < filled.size(); ++i)
        if (!filled[i]) throw std::logic_error("game " + std::to_string(i) + " was not played");
    // The remote files went slice by slice; the local ones too, but for the
    // logs of failed attempts, which stay (empty logs go).
    if (anyRemote && !failedSlices_) {
        std::filesystem::remove_all(workDir);
    } else if (anyRemote) {
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(workDir, ec))
            if (entry.is_regular_file() && entry.file_size() == 0)
                std::filesystem::remove(entry.path(), ec);
        if (std::filesystem::is_empty(workDir, ec)) std::filesystem::remove(workDir, ec);
    }
    reports_ = std::move(reports);
    return records;
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
    result.exitCode = runSliceJob(options["--job"], options["--out"], options["--backend"],
                                  number("--workers"), number("--inference-batch"), err);
    if (!log.empty()) {
        std::ofstream out(log, std::ios::app | std::ios::binary);
        out << err.str();
    }
    return result;
}

std::unique_ptr<Transport> makeTransport(const MachineSettings& machine, const std::string& log) {
    if (machine.ssh.empty()) return std::make_unique<SubprocessTransport>(machine);
    return std::make_unique<SshTransport>(machine, log);
}

std::string manifestPath(const std::string& runDir) {
    return (std::filesystem::path(runDir) / "selfplay-engines.csv").string();
}

namespace {

// The manifest's header and its rows of generations before g; *dropped
// counts the others.
std::string manifestBefore(const std::string& runDir, int g, int* dropped = nullptr) {
    std::string out = kManifestHeader;
    std::ifstream in(manifestPath(runDir), std::ios::binary);
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

} // namespace

void pruneManifest(const std::string& runDir, int g) {
    if (!std::filesystem::exists(manifestPath(runDir))) return;
    int dropped = 0;
    const std::string kept = manifestBefore(runDir, g, &dropped);
    if (dropped > 0) writeFileAtomically(manifestPath(runDir), kept);
}

void writeManifest(const std::string& runDir, int g, const std::vector<SliceReport>& reports) {
    std::string out = manifestBefore(runDir, g);
    for (const SliceReport& r : reports) {
        char numbers[96];
        std::snprintf(numbers, sizeof numbers, "%.1f,%lld,%lld", r.stats.seconds, r.stats.positions,
                      r.stats.evaluations);
        out += std::to_string(g) + "," + std::to_string(r.slice.first) + "," +
               std::to_string(r.slice.count) + "," + csvField(r.planned) + "," + csvField(r.engine) +
               "," + csvField(r.fingerprint, true) + "," + csvField(r.build) + "," + numbers + "," +
               csvField(r.note) + "\n";
    }
    writeFileAtomically(manifestPath(runDir), out);
}

std::string enginesLine(const std::vector<SliceReport>& reports) {
    std::string out = "selfplay engines:";
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
void reset() {
    std::lock_guard<std::mutex> lock(gProbeMutex);
    gFailures.clear();
    gRemoteBuild.clear();
    gDelays.clear();
    gInProcess = false;
}
} // namespace distributed_probe

} // namespace azgo
