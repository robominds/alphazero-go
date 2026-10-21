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
#include "azgo/mcts.hpp"
#include "azgo/network.hpp"
#include "azgo/selfplay_slice.hpp"

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

// The search a player's seat runs: the player's overrides where set, else
// cfg's, and never root noise (the arena measures strength, not exploring).
// batchSize is also the seat's evaluator batch. A bot's simulations are its
// own and it never searches with random symmetry. No validation: playMatch
// checks the player and the config first.
MctsConfig arenaSearchConfig(const ArenaPlayer& player, const ArenaConfig& cfg);

// Plays cfg.games games between a and b on `workers` threads, no root noise,
// tree reuse on. Board size and history come from whichever player has a
// network. Throws std::invalid_argument for two bots, two networks with
// different NetConfigs, a bot with simulations < 1, a simulations override
// outside 0..100000 (bots: 1..100000), a c_puct outside (0, 100] or an MCTS
// batch outside 1..256 (the train config's ranges; for an override, 0 is
// also allowed), or an odd or < 2 game count. Bots are capped at 100000
// simulations here too. Game i uses Rng(deriveSeed(seed, i)). Deterministic
// for a seed.
// It is playMatches with this one match: one set of evaluators per distinct
// network (a network playing itself has one), per worker a backend and a
// workspace shared by its LocalEvaluators for naive and blas; with metal one
// batching hub per network, whose passes hold at most inferenceBatch
// positions (each side's MCTS batch must fit, else std::invalid_argument).
// workers <= 0 means arenaWorkers(0, backendName, cfg.games); never more
// than cfg.games threads. Results do not depend on workers or inferenceBatch.
MatchResult playMatch(const ArenaPlayer& a, const ArenaPlayer& b, const ArenaConfig& cfg,
                      uint64_t seed, int workers, const std::string& backendName,
                      int inferenceBatch = 256);

// One match of a set that playMatches plays at once: games >= 2, even.
struct ArenaMatchSpec {
    ArenaPlayer a, b;
    int games = 0;
    uint64_t seed = 0;
};

// Plays several matches at once: every game of every match is one task in a
// single pool of worker threads, and each network (by address) gets one set
// of evaluators for all its seats in all the matches. With metal that is one
// batching hub per network, so the leaves of every search a network runs in
// the whole set share its GPU passes; with naive and blas each worker has
// one backend and one workspace that all its evaluators share (it plays one
// game, and so evaluates for one network, at a time). A generation's arena
// is a few short matches in which the new network plays every time: played
// one after another, too few searches were in flight per network for the
// GPU to batch well.
// Result i is exactly playMatch(matches[i].a, matches[i].b, cfg with
// games = matches[i].games, matches[i].seed, ...): game j of match i uses
// Rng(deriveSeed(matches[i].seed, j)) and results are stored by match and
// game, never by completion order, so they do not depend on workers,
// inferenceBatch or which matches share the set. cfg.games is ignored.
// Every match is checked as playMatch checks it before any game starts, and
// all networks must share one NetConfig. workers <= 0 means
// arenaWorkers(0, backendName, total games); never more threads than games.
std::vector<MatchResult> playMatches(const std::vector<ArenaMatchSpec>& matches, const ArenaConfig& cfg,
                                     int workers, const std::string& backendName,
                                     int inferenceBatch = 256);

// ---- arena games as ranges ---------------------------------------------------
//
// A wave's games can be played on several engines at once (arena_engines,
// azgo/coordinator.hpp), each a slice of them. Game j of a match is
// played with Rng(deriveSeed(seed, j)), with A as Black when j is even, so
// it is the same game whichever engine plays it and whatever else that
// engine plays beside it.

// Games [first, first + count) of the match between a and b seeded `seed`.
struct ArenaGameRange {
    ArenaPlayer a, b;
    uint64_t seed = 0;
    int first = 0, count = 0;
};

// One finished arena game: A's score (1, 0.5 or 0) and its moves.
struct ArenaGameResult {
    double scoreA = 0;
    std::vector<int> moves;
};

// Plays every game of every range in one pool of run.workers threads (<= 0:
// arenaWorkers(0, run.backend, games)), with one set of evaluators per
// distinct network as playMatches has (with metal or cuda one hub per
// network), each worker thread in run.qos's class, and returns the results
// in range order, game order within each range. With run.cancel set, a
// network evaluation after the flag is raised throws SliceCancelled. Each
// range is checked as playMatch checks a match, except that its count may
// be odd (and 0); first >= 0, count >= 0. Results depend only on the
// ranges, cfg and the backend's arithmetic, never on workers,
// inferenceBatch, qos or which ranges share the call. stats, when given:
// wall seconds, plies played and network evaluations.
std::vector<ArenaGameResult> playArenaGames(const std::vector<ArenaGameRange>& ranges,
                                            const ArenaConfig& cfg, const EngineRun& run,
                                            SliceStats* stats = nullptr);

// A match's result from its games: A's score summed in game order, wins,
// losses, draws and distinct move sequences. playMatches reports exactly
// this of the games playArenaGames plays.
MatchResult matchResult(const std::vector<ArenaGameResult>& games);

// The games of a wave are numbered match-major: match 0's games 0, 1, ...,
// then match 1's, and so on. The wave indices [first, first + count) as
// one range per match they touch, in match order, without empty ranges.
// Throws std::invalid_argument for a negative first or count, or a range
// past the wave's last game.
std::vector<ArenaGameRange> waveSlice(const std::vector<ArenaMatchSpec>& wave, int first, int count);

// The worker threads for `games` arena games, never more than games: an
// explicit count (> 0) as given; otherwise, with metal, every game at once
// (up to kMaxArenaWorkers), and with naive and blas resolveWorkers(0), one
// per core. cuda, the other GPU backend, as metal. Results do not depend on it.
constexpr int kMaxArenaWorkers = 256;   // the most the workers key allows
int arenaWorkers(int workers, const std::string& backendName, int games);

namespace arena_probe {
// Tests only: when on, playMatches plays its matches one after another, each
// with its own evaluators (hubs), as arena rounds were played before matches
// ran at once. The program never turns it on.
void setOneMatchAtATime(bool on);
} // namespace arena_probe

// Two networks: the same as playMatch(ArenaPlayer::network(a), ArenaPlayer::network(b), ...).
MatchResult playMatch(const Network& a, const Network& b, const ArenaConfig& cfg,
                      uint64_t seed, int workers, const std::string& backendName,
                      int inferenceBatch = 256);

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
