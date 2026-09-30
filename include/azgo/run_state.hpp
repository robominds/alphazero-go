// Resuming a run. A checkpoint holds only the network; continuing training
// exactly also needs the optimizer's momentum, the replay buffer, the
// generation counters and the arena results. They go into one binary file
// that is replaced atomically, so a crash mid-save leaves the old state.
#pragma once
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
};

// Binary, magic "AZST" (u32 0x54535A41), version u32 = 1. Atomic replace.
void saveTrainingState(const std::string& path, const TrainingState& state);
TrainingState loadTrainingState(const std::string& path);   // throws std::runtime_error

// Run-directory paths.
std::string checkpointPath(const std::string& runDir, int generation);  // runDir/checkpoints/gen_0007.azn
std::string statePath(const std::string& runDir);                        // runDir/state.bin
std::string configPath(const std::string& runDir);                       // runDir/config.cfg

} // namespace azgo
