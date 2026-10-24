// Resuming a run. A checkpoint holds only the network; continuing training
// exactly also needs the optimizer's momentum, the replay buffer, the
// generation counters, the arena results and the arena's reference set. They
// go into one binary file that is replaced atomically, so a crash mid-save
// leaves the old state.
#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include "azgo/elo.hpp"
#include "azgo/replay_buffer.hpp"
#include "azgo/selfplay.hpp"

namespace azgo {

// A frozen checkpoint the run plays as a fixed reference, as the run found
// it: the path from reference_networks and fileHash of its bytes, so a
// file replaced under the run is noticed instead of silently re-rated.
struct ReferenceNetwork {
    std::string path;
    uint64_t hash = 0;
};

// Pipelined generations (pipeline = on): the next generation's self-play
// games, played before the state was committed and not processed yet (their
// statistics, the resignation window, SGFs and the replay buffer come when
// that generation trains), with what processing them needs from the time
// they were played: a resume may have changed the keys since.
struct PendingSelfPlay {
    int network = 0;                  // the generation whose network played them
    bool resignation = false;         // resignation was on
    float resignThreshold = -0.9f;    // the v_resign they were played with
    int maxMoves = 0;                 // the resolved max_moves they were played with
    std::vector<GameRecord> records;  // in index order, resign flags and lows included
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
    // Resignation: the traces of recent resignation-disabled games, oldest
    // first (the trainer keeps at most resign_window, trimming as it
    // appends), and the v_resign calibrated from them after the last
    // generation, which the next one uses. The trainer recalibrates it on
    // resume from the window and the current keys, so the stored value only
    // records it; it is resign_threshold_initial's default until set.
    float resignThreshold = -0.9f;
    std::vector<ResignTrace> resignWindow;
    // Generation + 1's self-play games when the run is pipelined; empty
    // otherwise (and then the file is version 5, as before pipelining).
    std::optional<PendingSelfPlay> pending;
};

// The reference set of a new run: every fixed player (-numFixed, ..., -1:
// the bots, then the reference networks) and generation 0.
std::vector<int> initialReferences(int numFixed);

// Binary, magic "AZST" (u32 0x54535A41), version u32 = 5 (the replay buffer
// stores each game's final ownership and whether it ended by resignation;
// references, the bot list, the reference networks, then the resignation
// threshold and window come last), or 6 when the state has pending games:
// version 5's fields, then i32 network, u8 resignation, f32 v_resign, i32
// max_moves, u32 count and the games as a self-play result file stores them
// (the record, the resign-disabled flag, both sides' resign lows). A state
// without pending games is written as version 5. Atomic replace.
void saveTrainingState(const std::string& path, const TrainingState& state);
// Reads versions 6, 5, 4, 3, 2 and 1. Version 6's pending block is refused
// when it holds no games, names a network after the committed generation, a
// v_resign outside [-1, 0] or max_moves below 1, or a game that does not
// parse. Versions 1 to 4 predate resignation and
// load with no resigned games, an empty window and the default threshold.
// A version-5 threshold outside [-1, 0] or window entries that are not
// record lows (ply >= 1 rising, values in [-1, 1] falling) are refused. Versions 1 to 3 predate reference networks
// and load with none. Versions 1 and 2 predate stored ownership: each
// game's is recomputed by replaying its moves (a game that does not replay is
// an error). Version 1 also predates the reference set and the bot list: it
// gets botSimulations = `bots` and references = initialReferences(bots.size()),
// so pass the bot list the run's config defines. Throws std::runtime_error
// naming the path.
TrainingState loadTrainingState(const std::string& path, const std::vector<int>& bots = {});

// The generation a state.bin was committed at, read from its header alone
// (no buffer or optimizer state is loaded). Throws like loadTrainingState
// for a missing or foreign file.
int trainingStateGeneration(const std::string& path);

// 64-bit FNV-1a hash of the file's bytes: what state.bin records for each
// reference network. Throws std::runtime_error naming the path if it cannot
// be read.
uint64_t fileHash(const std::string& path);

// Seed phases for deriveSeed(cfg.seed, generation, phase, index). Part of a
// run's reproducibility contract: changing one changes every later game.
// kPhaseCalibration seeds fixed-player calibration (generation 0) and
// promotion games; kPhaseReference seeds fitting reference network j to the
// run's shape (generation 0, index j). kPhaseValidation seeds the buffer
// sample the validation loss compares with (a measurement: it feeds nothing
// back into the run). kPhaseBatchNorm seeds the positions generation 0's
// batch norm is calibrated on (generation 0, bn_calibrate; not to be confused
// with kPhaseCalibration, the fixed players' rating games).
constexpr uint64_t kPhaseInit = 0, kPhaseSelfPlay = 1, kPhaseTrain = 2, kPhaseArena = 3,
                   kPhaseCalibration = 4, kPhaseReference = 5, kPhaseValidation = 6,
                   kPhaseBatchNorm = 7;

namespace detail {
constexpr uint64_t kPhases[] = {kPhaseInit,        kPhaseSelfPlay,  kPhaseTrain,      kPhaseArena,
                                kPhaseCalibration, kPhaseReference, kPhaseValidation, kPhaseBatchNorm};
constexpr bool phasesDistinct() {
    for (size_t i = 0; i < sizeof kPhases / sizeof kPhases[0]; ++i)
        for (size_t j = i + 1; j < sizeof kPhases / sizeof kPhases[0]; ++j)
            if (kPhases[i] == kPhases[j]) return false;
    return true;
}
} // namespace detail
// Two phases with one value would give two streams the same seeds.
static_assert(detail::phasesDistinct(), "every kPhase* constant must be distinct");

// Run-directory paths.
std::string checkpointPath(const std::string& runDir, int generation);  // runDir/checkpoints/gen_0007.azn
std::string statePath(const std::string& runDir);                        // runDir/state.bin
std::string configPath(const std::string& runDir);                       // runDir/config.cfg

} // namespace azgo
