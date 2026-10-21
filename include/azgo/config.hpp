// Run configuration. Every knob of a training run lives in one plain-text
// "key = value" file so a run can be reproduced from its config.cfg alone.
// One table in config.cpp lists each key with its parser and printer, so the
// parser, the printer and the architecture check can never disagree.
// Architecture keys fix the network's tensor shapes and cannot change on
// resume. Neither can the fixed arena players, bot_simulations and
// reference_networks: bot k is arena player -k and reference network j comes
// after the bots, so a changed list would give old player IDs to new
// players. Every other key is tunable between generations.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/network.hpp"

namespace azgo {

// Upper limits of the `blocks` and `filters` keys; grow refuses to exceed them too.
constexpr int kMaxBlocks = 40;
constexpr int kMaxFilters = 512;

struct ConfigError : std::runtime_error { using std::runtime_error::runtime_error; };

// One item of selfplay_engines or arena_engines: an engine (a machine alias and a backend)
// and its share of a generation's games, either a weight (weight > 0,
// games = -1) or an exact count (games >= 0, weight = 0). Items are
// identified by their position in the list, so an engine may appear twice.
// Parsing, printing and the assignment of games live in azgo/engines.hpp.
struct EngineSpec {
    std::string machine, backend;
    int weight = 0;
    int games = -1;
    std::string name() const;   // "machine/backend"
};

struct Config {
    // run
    uint64_t seed = 1;
    int generations = 100;
    int workers = 0;                   // 0 = hardware concurrency
    // Threads of the training phase's network passes (training steps and
    // the validation loss); 0 = hardware concurrency, and a larger value is
    // capped at it. Changes speed only: the result is bitwise the same for
    // every value.
    int trainThreads = 0;
    // naive | blas | metal. metal: self-play, arena and validation inference
    // on the GPU through batching hubs; training and everything else on the
    // CPU with BLAS. Parsed in every build; a run checks it is available.
    std::string backend = "naive";
    // game
    int boardSize = 9;
    float komi = 7.5f;
    int maxMoves = 0;                  // 0 = 2 * n * n
    // network (architecture keys)
    int history = 8;
    int blocks = 6;
    int filters = 64;
    int valueHidden = 64;
    bool ownership = false;            // ownership head (KataGo's auxiliary target)
    // Batch norm calibration of a fresh generation 0 (azgo/bn_calibration.hpp):
    // its running statistics are set from one training-mode pass over fixed
    // positions, so that it does not saturate in inference mode. Read only
    // when startNew creates a random network: not with --init-from, not on
    // resume or re-rating. A run's config.cfg written before this key existed
    // lacks it, and resume and re-rating read that as false (what that run did).
    bool bnCalibrate = true;
    // self-play and search
    int gamesPerGeneration = 100;
    int simulations = 200;
    int mctsBatch = 8;
    // Most positions per GPU pass (backend = metal), at least mcts_batch.
    // Changes speed only: results do not depend on it.
    int inferenceBatch = 256;
    // Distributed self-play: which engine plays which of a generation's
    // games, e.g. "local/metal:50, local/blas=12, mini1/metal:10" (machine
    // aliases; remotes are defined in a machine-local hosts file, never
    // here). Empty: one engine, local/<backend>, through today's code.
    std::vector<EngineSpec> selfplayEngines;
    // When a remote slice fails twice: "local" plays it on the coordinator
    // and records that in the manifest; "stop" ends the generation uncommitted.
    std::string selfplayFallback = "local";
    float cPuct = 1.5f;
    float dirichletScale = 10.0f;
    float noiseFraction = 0.25f;
    int temperatureMoves = 20;
    bool mctsRandomSymmetry = false;   // self-play search: random symmetry per evaluation (AlphaGo Zero)
    // Resignation in self-play (AlphaGo Zero); the arena never resigns.
    bool resignation = false;
    float resignThresholdInitial = -0.9f;   // v_resign until resignMinGames disabled games exist
    float resignDisabledFraction = 0.1f;    // games played out with resignation off, to measure it
    float resignFalsePositive = 0.05f;      // most false positives v_resign may cause, as a fraction
    int resignWindow = 500;                 // most recent disabled games v_resign is calibrated on
    int resignMinGames = 20;                // disabled games needed before calibrating
    // training
    int bufferPositions = 200000;
    int trainSteps = 200;
    int batchSize = 256;
    std::string lrSchedule = "0:0.02";
    float momentum = 0.9f;
    float weightDecay = 1e-4f;
    float ownershipWeight = 1.5f;      // weight w of the ownership loss; 0 = the head does not train
    // Measure, each generation, the loss of the network on its fresh
    // self-play games before it trains on them (and on a sample of the
    // buffer it did train on). Measurement only: changes no result.
    bool validationLoss = true;
    // arena
    int arenaEvery = 1;                // 0 = never
    int arenaGames = 20;               // per opponent, even
    int arenaOpponents = 2;            // extra references chosen by rating (0-32), besides g-1 and the strongest
    std::vector<int> botSimulations = {100};   // uniform-prior MCTS bots; bot k is player -k; empty = none
    // Frozen checkpoints played as fixed references after the bots: reference
    // network j (1-based) is player -(bots + j). Empty = none.
    std::vector<std::string> referenceNetworks;
    int calibrationGames = 80;         // even: each fixed player vs generation 0 at the start; each promotion
    float promoteThreshold = 0.85f;    // score vs the strongest reference that adds g to the references
    int arenaSimulations = 100;
    int arenaTemperatureMoves = 4;
    bool arenaRandomSymmetry = false;  // arena search (networks): random symmetry per evaluation
    // Distributed arena: which engine plays which games of each arena wave,
    // e.g. "local/metal:16, pc/cuda:46", weights only (waves differ in size:
    // one opponent's games, several opponents', a promotion's, the
    // calibrations', so a fixed count would mean a different share in each). Empty: the whole arena on local/<backend>, through today's code.
    // selfplay_fallback governs its failed slices too.
    std::vector<EngineSpec> arenaEngines;
    bool gating = false;
    float gatingThreshold = 0.55f;
    // output
    int sgfPerGeneration = 2;

    NetConfig netConfig() const;
    int effectiveMaxMoves() const;     // maxMoves or 2 * n * n
    int effectiveWorkers() const;      // resolveWorkers(workers, backend == "metal")
    int effectiveTrainThreads() const; // resolveTrainThreads(trainThreads): capped at the core count
};

// key = value per line; '#' starts a comment (anywhere); blank lines ignored;
// surrounding whitespace and a trailing '\r' are stripped. Unknown key,
// malformed or out-of-range value, or a key repeated in one file ->
// ConfigError("<file>:<line>: <key>: <reason>").
Config loadConfigFile(const std::string& path, const Config& base = Config{});
Config parseConfigText(const std::string& text, const std::string& sourceName,
                       const Config& base = Config{});
// "key=value" (whitespace around '=' allowed). Same validation.
void applyOverride(Config& cfg, const std::string& keyEqualsValue);

// Every key, one "key = value" line each, in table order. Round-trips
// through parseConfigText exactly.
std::string configToString(const Config& cfg);

// Keys that fix the network's shape: board_size, history, blocks, filters,
// value_hidden, ownership.
std::vector<std::string> architectureDiff(const Config& a, const Config& b);

// Keys a resume must refuse to change: architectureDiff, then bot_simulations,
// then reference_networks.
std::vector<std::string> resumeLockedDiff(const Config& a, const Config& b);

// Rules between keys, which no single key's parser can check; a run checks
// them once every override is applied. ConfigError if resignation is on
// with ownership (a resigned game has no final position to take ownership
// targets from) or with resign_disabled_fraction = 0 (no games to measure
// false positives on), resign_min_games exceeds resign_window, or
// inference_batch is below mcts_batch (a request must fit in one GPU pass),
// or selfplay_engines' fixed counts add up to more than
// games_per_generation (or, with no weighted item, to anything else), or
// arena_engines holds a fixed count (it takes weights only).
void checkConsistency(const Config& cfg);

} // namespace azgo
