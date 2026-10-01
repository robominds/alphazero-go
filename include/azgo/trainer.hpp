#pragma once
// The trainer is the outer AlphaZero loop, run as synchronous generations:
// self-play with the current (or, with gating, the best) network fills the
// replay buffer; SGD on samples from that buffer produces the next network;
// an arena measures it against earlier generations for Elo; then the
// checkpoint and the full training state are written. Every random stream is
// derived from (seed, generation, phase, index), so a run is reproducible for a
// seed, independent of the worker count, and a resumed run continues
// byte-identically to one that was never interrupted.
#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include "azgo/config.hpp"
#include "azgo/network.hpp"
#include "azgo/run_state.hpp"

namespace azgo {

struct GenerationStats {
    int generation = 0;
    int games = 0;
    double avgLength = 0;       // plies per game
    double passRate = 0;        // passes / plies
    double blackWinRate = 0;
    float policyLoss = 0, valueLoss = 0;   // mean over the generation's steps
    float lr = 0;
    double replayRatio = 0;     // samples drawn this generation / new positions
    double selfPlaySeconds = 0, trainSeconds = 0, arenaSeconds = 0;
    double elo = 0;             // after refit; previous value if no arena this generation
    double winRateVsPrev = -1;  // -1 when no match vs g-1
    int distinctVsPrev = -1;    // distinct games in that match; -1 when no match vs g-1
};

// The generation loop: self-play -> train -> arena/Elo -> checkpoint + state.
class Trainer {
public:
    // New run. Creates runDir (and checkpoints/, sgf/); throws ConfigError if
    // runDir already contains state.bin. Writes config.cfg, gen_0000.azn and
    // an initial state.bin (generation 0).
    static std::unique_ptr<Trainer> startNew(const Config& cfg, const std::string& runDir);
    // Loads config.cfg, applies overrides, refuses (ConfigError listing keys)
    // if an architecture key changed, loads state.bin and the latest checkpoint.
    // Rewrites config.cfg with the overridden values.
    static std::unique_ptr<Trainer> resume(const std::string& runDir,
                                           const std::vector<std::string>& overrides);

    int generation() const { return state_.generation; }
    const Config& config() const { return cfg_; }
    GenerationStats runGeneration();
    void runUntil(int lastGeneration);   // while generation() < lastGeneration
    void setLog(std::ostream* os) { log_ = os; }   // progress lines; nullptr = silent

private:
    Trainer(const Config& cfg, const std::string& runDir, Network net, TrainingState state);
    Config cfg_;
    std::string runDir_;
    Network net_;
    TrainingState state_;
    std::ostream* log_;
    // Further members: implementation-defined (Task 21).
};

} // namespace azgo
