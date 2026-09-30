// The arena: checkpoints play each other with search but without the
// exploration noise of self-play, so the result measures playing strength.
// Colors alternate game by game (Black has the first-move advantage), a few
// opening moves are sampled at temperature 1 so the games are not all the
// same, and each new generation meets a sparse ladder of older ones.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "azgo/network.hpp"

namespace azgo {

struct ArenaConfig {
    int games = 20;              // even; A plays Black in even-indexed games
    int simulations = 100;
    int mctsBatch = 8;
    float cPuct = 1.5f;
    int temperatureMoves = 4;    // opening plies sampled at temperature 1
    float komi = 7.5f;
    int maxMoves = 0;            // 0 = defaultMaxMoves(n)
};

struct MatchResult {
    int games = 0;
    double scoreA = 0;           // wins + 0.5 * draws for A
    int winsA = 0, winsB = 0, draws = 0;
    int distinctGames = 0;       // distinct move sequences
};

// Plays cfg.games games between two networks of the same NetConfig (throws
// std::invalid_argument otherwise) on `workers` threads, no root noise, tree
// reuse on. Game i uses Rng(deriveSeed(seed, i)). Deterministic for a seed.
MatchResult playMatch(const Network& a, const Network& b, const ArenaConfig& cfg,
                      uint64_t seed, int workers, const std::string& backendName);

// Opponents for generation g: g-1, g-2, g-4, g-8, ... (>= 1), the first
// `maxOpponents` of those, then 0 if not already present. Empty for g == 0.
std::vector<int> ladderOpponents(int generation, int maxOpponents);

// score / games >= threshold.
bool passesGate(double score, int games, double threshold);

// Score of one finished game for player A: 1, 0.5 (resultForBlack == 0) or 0.
double matchScore(float resultForBlack, bool aIsBlack);

} // namespace azgo
