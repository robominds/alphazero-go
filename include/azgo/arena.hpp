// The arena: players meet with search but without the exploration noise of
// self-play, so the result measures playing strength. Colors alternate game
// by game (Black has the first-move advantage), and a few opening moves are
// sampled at temperature 1 so the games are not all the same. A player is a
// network or a uniform bot: MCTS that knows only the rules, a fixed yardstick
// that is the same in every run. Each new generation meets the opponents its
// results say the most about (selectOpponents): a game the new network is
// sure to win or sure to lose carries almost no information.
#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include "azgo/network.hpp"

namespace azgo {

struct ArenaConfig {
    int games = 20;              // even; A plays Black in even-indexed games
    int simulations = 100;       // a network's search budget per move
    int mctsBatch = 8;
    float cPuct = 1.5f;
    int temperatureMoves = 4;    // opening plies sampled at temperature 1
    float komi = 7.5f;
    int maxMoves = 0;            // 0 = defaultMaxMoves(n)
    bool randomSymmetry = false; // MctsConfig::randomSymmetry for both network players (a bot gains nothing)
};

struct MatchResult {
    int games = 0;
    double scoreA = 0;           // wins + 0.5 * draws for A
    int winsA = 0, winsB = 0, draws = 0;
    int distinctGames = 0;       // distinct move sequences
};

// One side of a match. With net, the player searches with the network, at
// `simulations` if that is positive and otherwise at ArenaConfig::simulations,
// so a frozen network can serve as a stronger fixed reference by searching
// more. With net == nullptr it is a uniform bot: MCTS whose priors are all
// equal and whose value is always 0, searching with `simulations`, which a bot
// must set (>= 1). A bot's strength depends only on its simulation count.
// randomSymmetry turns on MctsConfig::randomSymmetry for this network player
// alone (ArenaConfig::randomSymmetry turns it on for both), so a network can
// play itself with the search change on one side only. A bot ignores it.
// cPuct and mctsBatch replace ArenaConfig::cPuct and ::mctsBatch for this
// player alone (0 = the config's), so a network can play itself with one of
// AlphaGo Zero's search settings changed on one side. The batch size changes
// the search itself, not just its speed: the descents of a batch steer around
// each other by virtual loss. A bot searches with both, so they apply to it.
struct ArenaPlayer {
    const Network* net = nullptr;
    int simulations = 0;   // network: 0 = ArenaConfig::simulations; bot: required, >= 1
    bool randomSymmetry = false;
    float cPuct = 0;       // 0 = ArenaConfig::cPuct; otherwise (0, 100]
    int mctsBatch = 0;     // 0 = ArenaConfig::mctsBatch; otherwise 1..256
    static ArenaPlayer network(const Network& n, int simulations = 0, bool randomSymmetry = false) {
        return {&n, simulations, randomSymmetry};
    }
    static ArenaPlayer bot(int simulations) { return {nullptr, simulations}; }
};

// Plays cfg.games games between a and b on `workers` threads, no root noise,
// tree reuse on. Board size and history come from whichever player has a
// network. Throws std::invalid_argument for two bots, two networks with
// different NetConfigs, a bot with simulations < 1, a simulations override
// outside 0..100000 (bots: 1..100000), a c_puct outside (0, 100] or an MCTS
// batch outside 1..256 (the train config's ranges; for an override, 0 is
// also allowed), or an odd or < 2 game count. Bots are capped at 100000
// simulations here too. Game i uses Rng(deriveSeed(seed, i)). Deterministic
// for a seed.
MatchResult playMatch(const ArenaPlayer& a, const ArenaPlayer& b, const ArenaConfig& cfg,
                      uint64_t seed, int workers, const std::string& backendName);

// Two networks: the same as playMatch(ArenaPlayer::network(a), ArenaPlayer::network(b), ...).
MatchResult playMatch(const Network& a, const Network& b, const ArenaConfig& cfg,
                      uint64_t seed, int workers, const std::string& backendName);

// The rated reference with the highest rating, ties to the larger player ID.
// References missing from `ratings` are skipped; throws std::invalid_argument
// if none is rated.
int strongestReference(const std::map<int, double>& ratings, const std::vector<int>& references);

// Opponents for generation g >= 1 (std::invalid_argument otherwise), chosen
// from `ratings`, the Bradley-Terry refit of every result so far:
//   1. g-1, always;
//   2. the strongest reference, always;
//   3. up to maxExtra more rated references: those for which g's expected
//      score is closest to 0.7, ties to the smaller ID. g's rating is
//      estimated as that of g-1, or 0 if g-1 is unrated;
//   4. bestNetwork, if given (gating).
// No player appears twice; the order is the order above.
std::vector<int> selectOpponents(int generation, const std::map<int, double>& ratings,
                                 const std::vector<int>& references, int maxExtra,
                                 std::optional<int> bestNetwork = std::nullopt);

// score >= (threshold - 1e-6) * games: thresholds are float config keys (0.85f > 0.85).
bool passesGate(double score, int games, double threshold);

// Score of one finished game for player A: 1, 0.5 (resultForBlack == 0) or 0.
double matchScore(float resultForBlack, bool aIsBlack);

} // namespace azgo
