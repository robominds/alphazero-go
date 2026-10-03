// Run configuration. Every knob of a training run lives in one plain-text
// "key = value" file so a run can be reproduced from its config.cfg alone.
// One table in config.cpp lists each key with its parser and printer, so the
// parser, the printer and the architecture check can never disagree.
// Architecture keys fix the network's tensor shapes and cannot change on
// resume. Neither can bot_simulations: bot k is arena player -k, so a changed
// list would give old player IDs to new bots. Every other key is tunable
// between generations.
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

struct Config {
    // run
    uint64_t seed = 1;
    int generations = 100;
    int workers = 0;                   // 0 = hardware concurrency
    std::string backend = "naive";     // naive | blas
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
    // self-play and search
    int gamesPerGeneration = 100;
    int simulations = 200;
    int mctsBatch = 8;
    float cPuct = 1.5f;
    float dirichletScale = 10.0f;
    float noiseFraction = 0.25f;
    int temperatureMoves = 20;
    // training
    int bufferPositions = 200000;
    int trainSteps = 200;
    int batchSize = 256;
    std::string lrSchedule = "0:0.02";
    float momentum = 0.9f;
    float weightDecay = 1e-4f;
    float ownershipWeight = 1.5f;      // weight w of the ownership loss; 0 = the head does not train
    // arena
    int arenaEvery = 1;                // 0 = never
    int arenaGames = 20;               // per opponent, even
    int arenaOpponents = 2;            // extra references chosen by rating (0-32), besides g-1 and the strongest
    std::vector<int> botSimulations = {100};   // uniform-prior MCTS bots; bot k is player -k; empty = none
    int calibrationGames = 80;         // even: each bot vs generation 0 at the start; each promotion
    float promoteThreshold = 0.85f;    // score vs the strongest reference that adds g to the references
    int arenaSimulations = 100;
    int arenaTemperatureMoves = 4;
    bool gating = false;
    float gatingThreshold = 0.55f;
    // output
    int sgfPerGeneration = 2;

    NetConfig netConfig() const;
    int effectiveMaxMoves() const;     // maxMoves or 2 * n * n
    int effectiveWorkers() const;      // resolveWorkers(workers)
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

// Keys a resume must refuse to change: architectureDiff, then bot_simulations.
std::vector<std::string> resumeLockedDiff(const Config& a, const Config& b);

} // namespace azgo
