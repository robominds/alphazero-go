#include "azgo/coordinator.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include "azgo/backend.hpp"
#include "azgo/build_info.hpp"
#include "azgo/inference_hub.hpp"
#include "azgo/metal.hpp"
#include "azgo/parallel.hpp"
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

} // namespace

SelfPlayCoordinator::SelfPlayCoordinator(const Config& cfg, const std::string& runDir,
                                         HostsFile hosts, std::ostream* log)
    : cfg_(cfg), runDir_(runDir), hosts_(std::move(hosts)), log_(log) {
    if (cfg_.selfplayEngines.empty())
        throw std::logic_error("SelfPlayCoordinator: selfplay_engines is empty (the default path)");
}

std::string SelfPlayCoordinator::fingerprint(const std::string& backend) {
    auto it = fingerprints_.find(backend);
    if (it == fingerprints_.end())
        it = fingerprints_.emplace(backend, engineFingerprint(backend)).first;
    return it->second;
}

void SelfPlayCoordinator::preflight() {
    int metalHubs = 0;
    for (const EngineSpec& e : cfg_.selfplayEngines) {
        if (e.machine != "local")
            throw ConfigError("selfplay_engines: " + e.name() +
                              ": remote engines are not supported yet (only local/...)");
        checkBackendAvailable(e.backend);
        if (e.backend == "metal") ++metalHubs;
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
    std::vector<std::string> printed;
    for (const EngineSpec& e : cfg_.selfplayEngines) {
        const std::string f = fingerprint(e.backend);
        if (std::find(printed.begin(), printed.end(), e.name()) != printed.end()) continue;
        printed.push_back(e.name());
        // The fingerprint without its leading "<backend>; ", which the name says.
        if (log_) *log_ << "engine " << e.name() << ": " << f.substr(e.backend.size() + 2) << " ["
                        << shortBuildId() << "]\n";
    }
}

std::vector<GameRecord> SelfPlayCoordinator::play(int g, const std::string& networkPath,
                                                  const Network& net, const SelfPlayConfig& sp,
                                                  int games) {
    (void)networkPath;   // remote engines will load it
    const std::vector<Slice> slices = assignGames(cfg_.selfplayEngines, games);
    const int cores = resolveWorkers(0);
    const size_t n = slices.size();
    std::vector<SliceReport> reports(n);
    std::vector<EngineRun> runs(n);
    for (size_t k = 0; k < n; ++k) {
        const EngineSpec& e = cfg_.selfplayEngines[static_cast<size_t>(slices[k].engine)];
        const MachineSettings m = hosts_.settings(e.machine);
        if (!m.local()) throw std::logic_error("remote engine " + e.name() + " reached play()");
        runs[k].backend = e.backend;
        runs[k].workers = engineWorkers(m, e.backend, slices[k].count, cores);
        runs[k].inferenceBatch = m.inferenceBatch > 0 ? m.inferenceBatch : cfg_.inferenceBatch;
        // The GPU, which sets the pace, waits on the metal engine's workers:
        // they keep the default class. CPU engines take the machine's.
        runs[k].qos = e.backend == "metal" ? CpuQos::Default : m.cpuQos;
        reports[k].slice = slices[k];
        reports[k].planned = reports[k].engine = e.name();
        reports[k].fingerprint = fingerprint(e.backend);
        reports[k].build = shortBuildId();
    }

    // The BLAS backend sets its environment once, with setenv, which is not
    // thread-safe: do that here, before any engine thread runs.
    for (const EngineRun& run : runs)
        if (run.backend == "blas") {
            makeBackend("blas", 1);
            break;
        }

    // One thread per slice, all at once. Each engine has its own evaluators,
    // so a slice's games are what playSelfPlayGames would give alone. The
    // first slice to fail sets `stop`, and the others stop at their next
    // network evaluation: the generation will not be committed anyway.
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
                    results[k] = playSelfPlayGames(net, sp, cfg_.seed, g, slices[k].first,
                                                   slices[k].count, runs[k], &reports[k].stats);
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
    reports_ = std::move(reports);
    return records;
}

// ---- transports ----------------------------------------------------------------

namespace {

constexpr double kCommandTimeout = 60;    // mkdir, test, rm, pkill over ssh
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

std::string describe(const ProcessResult& r) {
    if (!r.error.empty()) return r.error;
    if (r.timedOut) return "timed out";
    if (r.signal != 0) return "killed by signal " + std::to_string(r.signal);
    return "exit " + std::to_string(r.exitCode);
}

// "~/x" on this machine.
std::string expandHome(const std::string& path) {
    if (path.rfind("~/", 0) != 0) return path;
    const char* home = std::getenv("HOME");
    return (home ? std::string(home) : std::string()) + path.substr(1);
}

} // namespace

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
    requireSafe(safeWord(local), "local path", local);
    requireSafe(safeWord(remote), "remote path", remote);
    const std::string far = machine.ssh + ":" + remote;
    std::vector<std::string> argv = {"scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"};
    argv.push_back(upload ? local : far);
    argv.push_back(upload ? far : local);
    return argv;
}

SshTransport::SshTransport(MachineSettings machine, std::string log)
    : m_(std::move(machine)), log_(std::move(log)) {}

// The failure and the last line scp or ssh wrote about it.
std::string SshTransport::why(const ProcessResult& r) const {
    const std::string line = log_.empty() ? "" : lastLine(log_);
    return describe(r) + (line.empty() ? "" : ": " + line);
}

ProcessResult SshTransport::command(const std::vector<std::string>& words, double timeout) {
    return runProcess(sshCommand(m_, words), log_, timeout);
}

void SshTransport::upload(const std::string& local, const std::string& remote) {
    const ProcessResult dir = command({"mkdir", "-p", parentOf(remote)}, kCommandTimeout);
    if (dir.exitCode != 0)
        throw SliceError(m_.alias + ": cannot create " + parentOf(remote) + " (ssh " + why(dir) +
                         ")");
    const ProcessResult copy = runProcess(scpArgv(m_, local, remote, true), log_, kTransferTimeout);
    if (copy.exitCode != 0)
        throw SliceError(m_.alias + ": upload of " + remote + " failed (scp " + why(copy) + ")");
}

bool SshTransport::exists(const std::string& remote) {
    const ProcessResult r = command({"test", "-f", remote}, kCommandTimeout);
    if (r.exitCode == 0) return true;
    if (r.exitCode == 1) return false;
    throw SliceError(m_.alias + ": cannot reach the machine (ssh " + why(r) + ")");
}

ProcessResult SshTransport::run(const std::vector<std::string>& args, const std::string& log,
                                double timeout) {
    return runProcess(sshArgv(m_, args), log, timeout);
}

void SshTransport::download(const std::string& remote, const std::string& local) {
    const ProcessResult copy =
        runProcess(scpArgv(m_, local, remote, false), log_, kTransferTimeout);
    if (copy.exitCode != 0)
        throw SliceError(m_.alias + ": download of " + remote + " failed (scp " + why(copy) + ")");
}

void SshTransport::remove(const std::vector<std::string>& remote) {
    if (remote.empty()) return;
    std::vector<std::string> words = {"rm", "-f"};
    words.insert(words.end(), remote.begin(), remote.end());
    const ProcessResult r = command(words, kCommandTimeout);
    if (r.exitCode != 0)
        throw SliceError(m_.alias + ": cannot remove work files (ssh " + why(r) + ")");
}

void SshTransport::killStray(const std::string& runId) {
    command({"pkill", "-f", runId}, kCommandTimeout);   // best effort; 1 = nothing matched
}

SubprocessTransport::SubprocessTransport(MachineSettings machine) : m_(std::move(machine)) {}

void SubprocessTransport::upload(const std::string& local, const std::string& remote) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path to = expandHome(remote);
    fs::create_directories(to.parent_path(), ec);
    if (!ec) fs::copy_file(local, to, fs::copy_options::overwrite_existing, ec);
    if (ec) throw SliceError(m_.alias + ": upload of " + remote + " failed: " + ec.message());
}

bool SubprocessTransport::exists(const std::string& remote) {
    return std::filesystem::is_regular_file(expandHome(remote));
}

ProcessResult SubprocessTransport::run(const std::vector<std::string>& args, const std::string& log,
                                       double timeout) {
    std::vector<std::string> argv;
    if (m_.nice > 0) argv = {"nice", "-n", std::to_string(m_.nice)};
    argv.push_back(expandHome(m_.exe));
    for (const std::string& a : args) argv.push_back(a[0] == '~' ? expandHome(a) : a);
    return runProcess(argv, log, timeout);
}

void SubprocessTransport::download(const std::string& remote, const std::string& local) {
    std::error_code ec;
    std::filesystem::copy_file(expandHome(remote), local,
                               std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) throw SliceError(m_.alias + ": download of " + remote + " failed: " + ec.message());
}

void SubprocessTransport::remove(const std::vector<std::string>& remote) {
    for (const std::string& r : remote) {
        std::error_code ec;
        std::filesystem::remove(expandHome(r), ec);
    }
}

void SubprocessTransport::killStray(const std::string& runId) {
    runProcess({"pkill", "-f", runId}, "", kCommandTimeout);   // best effort
}

ProcessResult InProcessTransport::run(const std::vector<std::string>& args, const std::string& log,
                                      double timeout) {
    (void)timeout;
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

} // namespace azgo
