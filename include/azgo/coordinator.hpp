// The self-play coordinator: plays one generation's games on the engines
// that selfplay_engines lists, and records who played what.
//
// assignGames cuts the generation's game indices into one contiguous slice
// per engine. Each local engine (local/metal, local/blas, ...) runs its
// slice through playSelfPlayGames on its own thread, with its own pool of
// workers and evaluators, at the same time as the others: a GPU engine's
// hub and a CPU engine's per-worker BLAS backends share nothing but the
// read-only network. The slices are then merged by game index, never by
// the order they finished in, so the generation's records depend only on
// the assignment and on what each engine computes.
//
// Every slice is written to the run's manifest, selfplay-engines.csv: the
// engine planned for it, the engine that played it, that engine's
// fingerprint and build, and what it cost. With the seed, the manifest is
// the exact recipe for every game of the run.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <map>
#include <ostream>
#include <string>
#include <vector>
#include "azgo/config.hpp"
#include "azgo/engines.hpp"
#include "azgo/network.hpp"
#include "azgo/selfplay.hpp"
#include "azgo/selfplay_slice.hpp"

namespace azgo {

// One slice of one generation: the games, the engine the assignment
// planned and the one that played it (equal unless a slice fell back),
// that engine's fingerprint and build (shortBuildId), the statistics, and
// a note ("" or why another engine played it).
struct SliceReport {
    Slice slice;
    std::string planned, engine, fingerprint, build, note;
    SliceStats stats;
};

class SelfPlayCoordinator {
public:
    // cfg.selfplayEngines must not be empty. `hosts` gives each machine's
    // settings (local's defaults when it has no line). Progress and notes go
    // to `log` (null: silent).
    SelfPlayCoordinator(const Config& cfg, const std::string& runDir, HostsFile hosts,
                        std::ostream* log);

    // Before a run writes anything: refuses (ConfigError) a remote engine,
    // which this version cannot reach yet; checks each local backend
    // (checkBackendAvailable, and for metal checkMetalInferenceMemory for
    // one two-slot hub per metal item); takes each local engine's
    // fingerprint and prints one line per engine:
    //   engine local/metal: Apple M5 Max; macOS 27.0.1 (25A123); Apple M5 Max, ... [09e008b]
    void preflight();

    // Generation g's `games` self-play games from `net` (the checkpoint at
    // networkPath), merged in index order. Throws the first failure of any
    // slice, by list order, after every slice has stopped.
    std::vector<GameRecord> play(int g, const std::string& networkPath, const Network& net,
                                 const SelfPlayConfig& sp, int games);
    // The last play()'s slices, in list order.
    const std::vector<SliceReport>& reports() const { return reports_; }
    void setLog(std::ostream* log) { log_ = log; }

private:
    std::string fingerprint(const std::string& backend);   // cached per backend
    Config cfg_;
    std::string runDir_;
    HostsFile hosts_;
    std::ostream* log_;
    std::map<std::string, std::string> fingerprints_;   // local backend -> fingerprint
    std::vector<SliceReport> reports_;
};

// runDir/selfplay-engines.csv
std::string manifestPath(const std::string& runDir);

// Rewrites the manifest atomically: its rows of generations before g, then
// one row per report for generation g:
//   generation,first,count,planned,engine,fingerprint,build,seconds,positions,evaluations,note
// The fingerprint is always quoted (a metal one holds commas); a note is
// quoted when it holds a comma or a quote. Rows of g and later are dropped
// first, so a redone generation leaves one set of rows. seconds is, like
// log.csv's timings, not part of a run's byte-identity.
void writeManifest(const std::string& runDir, int g, const std::vector<SliceReport>& reports);

// Drops the manifest's rows of generations g and later, if it has any (a
// crash between the manifest and state.bin leaves rows of an uncommitted
// generation). No manifest: nothing. A resume calls it with
// g = the committed generation + 1, with engines on or off.
void pruneManifest(const std::string& runDir, int g);

// "selfplay engines: local/metal 52 1512s 3.5k ev/s | local/blas 12 1490s 0.9k": each
// slice's engine, games, wall time and network evaluations per second.
std::string enginesLine(const std::vector<SliceReport>& reports);

} // namespace azgo
