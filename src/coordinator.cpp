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
