// Resuming a run. A checkpoint holds only the network; continuing training
// exactly also needs the optimizer's momentum, the replay buffer, the
// generation counters, the arena results and the arena's reference set. They
// go into one binary file that is replaced atomically, so a crash mid-save
// leaves the old state.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "azgo/elo.hpp"
#include "azgo/replay_buffer.hpp"

namespace azgo {

// A frozen checkpoint the run plays as a fixed reference, as the run found
// it: the path from reference_networks and fileHash of its bytes, so a
// file replaced under the run is noticed instead of silently re-rated.
struct ReferenceNetwork {
    std::string path;
    uint64_t hash = 0;
};

// Everything needed to continue a run exactly, besides the checkpoints and
// config.cfg. The network for `generation` is checkpoints/gen_NNNN.azn.
struct TrainingState {
    int generation = 0;           // last completed generation
    int bestGeneration = 0;       // gating: network used for self-play
    long long samplesTrained = 0; // total training samples drawn
    ReplayBuffer buffer{1};
    std::vector<std::vector<float>> velocity;  // optimizer momentum, params() order
    ResultsTable results;
    // The frozen players new networks are measured against: the fixed
    // players (bots, then reference networks), generation 0 and every
    // promoted generation. Sorted ascending, always contains 0.
    std::vector<int> references{0};
    // The run's bot_simulations, in order (bot k is player -k). Recorded here
    // because config.cfg can be edited by hand, and resume must notice.
    std::vector<int> botSimulations;
    // The run's reference_networks, in order: reference network j (1-based)
    // is player -(botSimulations.size() + j). Recorded for the same reason.
    std::vector<ReferenceNetwork> referenceNetworks;
    // Resignation: the v_resign the next generation's self-play uses
    // (resign_threshold_initial's default until a trainer sets it), and the
    // winner minimums (GameRecord::winnerMinResignValue) of the most recent
    // resignation-disabled games, oldest first, at most resign_window.
    float resignThreshold = -0.9f;
    std::vector<float> resignWindow;
};

// The reference set of a new run: every fixed player (-numFixed, ..., -1:
// the bots, then the reference networks) and generation 0.
std::vector<int> initialReferences(int numFixed);

// Binary, magic "AZST" (u32 0x54535A41), version u32 = 5 (the replay buffer
// stores each game's final ownership; references, the bot list, the
// reference networks, then the resignation threshold and window come last).
// Atomic replace.
void saveTrainingState(const std::string& path, const TrainingState& state);
// Reads versions 5, 4, 3, 2 and 1. Versions 1 to 4 predate resignation and
// load with an empty window and the default threshold (the trainer
// recalibrates it on resume). Versions 1 to 3 predate reference networks
// and load with none. Versions 1 and 2 predate stored ownership: each
// game's is recomputed by replaying its moves (a game that does not replay is
// an error). Version 1 also predates the reference set and the bot list: it
// gets botSimulations = `bots` and references = initialReferences(bots.size()),
// so pass the bot list the run's config defines. Throws std::runtime_error
// naming the path.
TrainingState loadTrainingState(const std::string& path, const std::vector<int>& bots = {});

// 64-bit FNV-1a hash of the file's bytes: what state.bin records for each
// reference network. Throws std::runtime_error naming the path if it cannot
// be read.
uint64_t fileHash(const std::string& path);

// Seed phases for deriveSeed(cfg.seed, generation, phase, index). Part of a
// run's reproducibility contract: changing one changes every later game.
// kPhaseCalibration seeds fixed-player calibration (generation 0) and
// promotion games; kPhaseReference seeds fitting reference network j to the
// run's shape (generation 0, index j).
constexpr uint64_t kPhaseInit = 0, kPhaseSelfPlay = 1, kPhaseTrain = 2, kPhaseArena = 3,
                   kPhaseCalibration = 4, kPhaseReference = 5;

// Run-directory paths.
std::string checkpointPath(const std::string& runDir, int generation);  // runDir/checkpoints/gen_0007.azn
std::string statePath(const std::string& runDir);                        // runDir/state.bin
std::string configPath(const std::string& runDir);                       // runDir/config.cfg

} // namespace azgo
