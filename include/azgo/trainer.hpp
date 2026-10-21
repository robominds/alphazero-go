#pragma once
// The trainer is the outer AlphaZero loop, run as synchronous generations:
// self-play with the current (or, with gating, the best) network fills the
// replay buffer; SGD on samples from that buffer produces the next network;
// an arena measures it against earlier generations for Elo; then the
// checkpoint and the full training state are written. Every random stream is
// derived from (seed, generation, phase, index), so a run is reproducible for a
// seed, independent of the worker count, and a resumed run continues
// byte-identically to one that was never interrupted. With backend = metal,
// self-play, the arena and the validation loss evaluate the network on the
// GPU through batching hubs, under the same guarantees on one machine. With
// selfplay_engines, a Coordinator plays each generation's games on
// several engines at once (azgo/coordinator.hpp), and with arena_engines
// each arena wave's games; without them, self-play and the arena each run
// on one engine, exactly as before.
#include <iostream>
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
    float ownershipLoss = -1;   // the same mean, unweighted; -1 when the network has no ownership head
    float lr = 0;
    double replayRatio = 0;     // samples drawn this generation / new positions
    double selfPlaySeconds = 0, trainSeconds = 0, arenaSeconds = 0;
    double elo = 0;             // after refit; previous value if no arena this generation
    double winRateVsPrev = -1;  // -1 when no match vs g-1
    int distinctVsPrev = -1;    // distinct games in that match; -1 when no match vs g-1
    // Resignation, when the config turns it on (the fields below are unset otherwise).
    bool resignation = false;
    float resignThreshold = 0;        // v_resign this generation's self-play used
    double resignRate = -1;           // resigned / games with resignation enabled; -1 when none were
    // False positives at resignThreshold over the window, this generation's
    // disabled games included; -1 while the window is empty.
    double resignFalsePositive = -1;
    // The validation loss, when validation_loss is on (-1 when not measured).
    // val*: the network this generation trains (generation g-1's), in
    // inference mode, on every position of this generation's self-play
    // games, before they join the replay buffer: data it has never trained
    // on. seen*: the same on as many positions drawn from its training
    // distribution, the buffer as generation g-1 left it (with --buffer-from,
    // at generation 1, a donor's games generation 0 never trained on); -1
    // while the buffer is empty. The ownership terms only with the head.
    float valPolicyLoss = -1, valValueLoss = -1, valOwnershipLoss = -1;
    float seenPolicyLoss = -1, seenValueLoss = -1, seenOwnershipLoss = -1;
};

class Coordinator;   // azgo/coordinator.hpp

// How a run reaches other machines; nothing here changes a result.
// hostsFile: the machine-local hosts file (azgo/engines.hpp); empty means
// $AZGO_HOSTS, else ~/.config/azgo/hosts. Read only when selfplay_engines
// or arena_engines is set.
struct TrainerOptions {
    std::string hostsFile;
};

// The generation loop: self-play -> (validation loss) -> train -> arena/Elo
// -> checkpoint + state.
class Trainer {
public:
    // Optional starting material for a new run. initFrom: a checkpoint that
    // becomes generation 0 instead of a fresh random network (its shape must
    // equal the config's, ownership head included); this is how a grown or
    // pre-trained network starts a run. bufferFrom: another run's directory whose replay buffer seeds the
    // new one, so the first generations train on real games rather than an
    // empty buffer (board size and history must match). Empty = not used.
    struct RunInit {
        std::string initFrom;
        std::string bufferFrom;
    };
    // New run. Creates runDir (and checkpoints/, sgf/); throws ConfigError if
    // runDir already contains state.bin, `init` does not fit cfg, or a
    // reference network is missing, unreadable or bigger than cfg's shape.
    // Writes config.cfg, gen_0000.azn and an initial state.bin (generation 0,
    // with each reference network's path and hash). A fresh generation 0 has
    // its batch norm calibrated (bn_calibrate, azgo/bn_calibration.hpp,
    // seeded by deriveSeed(seed, 0, kPhaseBatchNorm), on the CPU: on one
    // machine type the same bytes for any workers, train_threads and
    // backend; FMA contraction can differ across architectures); an initFrom
    // network is used as it is. With `init` also init.txt
    // recording where the start came from. Every fixed player (bot or
    // reference network) plays calibration_games against generation 0.
    // Optimizer momentum starts at zero either way. Refuses an unavailable
    // backend (checkBackendAvailable) before writing anything; with metal it
    // prints the device line to `log` (null: silent):
    //   gpu: Apple M5 Max, macOS 27.0.1, shaders 3f1c9a07 (inference: self-play, arena, validation)
    // With selfplay_engines or arena_engines, the coordinator's preflight
    // runs first, before anything is written: it reads the hosts file
    // (opts.hostsFile), checks every engine and prints one line per engine
    // (Coordinator::preflight). With arena_engines the calibration games are
    // split across its engines, recorded as generation 0's rows of
    // arena-engines.csv, with an "arena engines:" line to `log`.
    static std::unique_ptr<Trainer> startNew(const Config& cfg, const std::string& runDir,
                                             const RunInit& init = {},
                                             std::ostream* log = &std::cout,
                                             const TrainerOptions& opts = {});
    // Loads config.cfg (one without bn_calibrate, from before the key, as
    // bn_calibrate = false), applies overrides, refuses (ConfigError listing keys)
    // if an architecture key, bot_simulations or reference_networks changed,
    // or if a reference network's file no longer matches the hash in
    // state.bin; loads state.bin and the latest checkpoint.
    // Refuses inconsistent keys (checkConsistency) the same way, as does
    // startNew. Recalibrates v_resign from state.bin's window and the
    // (possibly overridden) resignation keys. Rewrites config.cfg with the
    // overridden values, and gives a log.csv written before the
    // ownership_loss, resignation or validation columns existed those
    // columns (empty). Refuses an unavailable backend first, like startNew,
    // and prints to `log` the device line with metal and, when backend
    // differs from the saved config.cfg, a note that the results' last bits
    // will differ from a run that kept the old one. The same preflight as
    // startNew with selfplay_engines or arena_engines, and the same kind of
    // note when either differs from the saved config.cfg's. Drops the
    // manifests' rows of uncommitted generations (pruneManifest,
    // pruneArenaManifest).
    static std::unique_ptr<Trainer> resume(const std::string& runDir,
                                           const std::vector<std::string>& overrides,
                                           std::ostream* log = &std::cout,
                                           const TrainerOptions& opts = {});
    ~Trainer();

    int generation() const { return state_.generation; }
    const Config& config() const { return cfg_; }
    GenerationStats runGeneration();
    void runUntil(int lastGeneration);   // while generation() < lastGeneration
    void setLog(std::ostream* os);   // progress lines; nullptr = silent

private:
    Trainer(const Config& cfg, const std::string& runDir, Network net, TrainingState state);
    Config cfg_;
    std::string runDir_;
    Network net_;
    TrainingState state_;
    std::ostream* log_;
    // The run's reference networks, read once and fitted to its shape: fixed
    // player -(bots + j + 1) is referenceNets_[j].
    std::vector<Network> referenceNets_;
    // Plays self-play when selfplay_engines is set and the arena when
    // arena_engines is; null on the default path.
    std::unique_ptr<Coordinator> coordinator_;
};

// Re-rates the run in runDir without training it: replays the arena phase of
// generations 1, 2, ... for as long as their checkpoints exist (and, with a
// readable state.bin, no further than the generation it committed), with today's
// opponent selection, starting from an empty results table, the initial
// reference set and the fixed players' calibration, and with the seeds of
// live training (every arena_every-th generation, as live). Gating is
// ignored. `overrides` change tunable keys for the replay only, including
// bot_simulations and reference_networks (so a finished run can be measured
// against other fixed players); architecture keys are refused (ConfigError).
// When the replay keeps the run's reference_networks (config.cfg's list),
// a file that no longer matches the hash in state.bin is refused too; only
// then is state.bin read. Writes
// runDir/rerate/{results.csv, elo.csv, references.csv, log.csv} (and
// reference-networks.csv with reference networks), rewritten after each
// generation; never touches state.bin or config.cfg. With gating off, a run's
// checkpoints do not depend on its arena, so this is exactly what a live run
// would have measured (with arena_engines, which the replay ignores, up to
// the last bits: every game is played here, with the run's backend). Progress lines go to `log` (nullptr = silent), the
// device line first with metal, then the fixed players' calibration. An
// unavailable backend is refused before anything is written. Returns the
// last generation replayed.
int rerateRun(const std::string& runDir, const std::vector<std::string>& overrides,
              std::ostream* log);

} // namespace azgo
