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

// Everything needed to continue a run exactly, besides the checkpoints and
// config.cfg. The network for `generation` is checkpoints/gen_NNNN.azn.
struct TrainingState {
    int generation = 0;           // last completed generation
    int bestGeneration = 0;       // gating: network used for self-play
    long long samplesTrained = 0; // total training samples drawn
    ReplayBuffer buffer{1};
    std::vector<std::vector<float>> velocity;  // optimizer momentum, params() order
    ResultsTable results;
    // The frozen players new networks are measured against: bots, generation
    // 0 and every promoted generation. Sorted ascending, always contains 0.
    std::vector<int> references{0};
};

// The reference set of a new run: every bot (-numBots, ..., -1) and generation 0.
std::vector<int> initialReferences(int numBots);

// Binary, magic "AZST" (u32 0x54535A41), version u32 = 2. Atomic replace.
void saveTrainingState(const std::string& path, const TrainingState& state);
// Reads version 2 and version 1, which predates the reference set: a version-1
// file gets references = initialReferences(numBots), so pass the number of
// bots the run's config defines. Throws std::runtime_error naming the path.
TrainingState loadTrainingState(const std::string& path, int numBots = 0);

// Seed phases for deriveSeed(cfg.seed, generation, phase, index). Part of a
// run's reproducibility contract: changing one changes every later game.
// kPhaseCalibration seeds bot calibration (generation 0) and promotion games.
constexpr uint64_t kPhaseInit = 0, kPhaseSelfPlay = 1, kPhaseTrain = 2, kPhaseArena = 3,
                   kPhaseCalibration = 4;

// Run-directory paths.
std::string checkpointPath(const std::string& runDir, int generation);  // runDir/checkpoints/gen_0007.azn
std::string statePath(const std::string& runDir);                        // runDir/state.bin
std::string configPath(const std::string& runDir);                       // runDir/config.cfg

} // namespace azgo
