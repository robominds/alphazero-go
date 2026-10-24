// One arena round, shared by live training and by re-rating a finished run,
// so both make the same choices from the same results. A round calibrates
// any fixed player (a bot or a reference network) that has never played
// generation 0, refits every rating, picks the
// opponents whose results will say the most about the new network, plays
// them, and promotes the new network into the reference set when it clearly
// beats the strongest reference. Every choice depends only on the results
// table, the reference set and the run's seed, so an uninterrupted run, a
// resumed run and a re-rated run play the same games.
#pragma once
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>
#include "azgo/arena.hpp"
#include "azgo/elo.hpp"

namespace azgo {

// Plays `games` games between players a and b with `seed` and returns the
// result from a's side. Players are generations (>= 0) and fixed players
// (-1, -2, ...): the bots, then the reference networks. The trainer plays
// real networks; tests can return made-up results.
using MatchFn = std::function<MatchResult(int a, int b, int games, uint64_t seed)>;

// One match of a wave: players a and b, games, seed.
struct MatchRequest {
    int a = 0, b = 0, games = 0;
    uint64_t seed = 0;
};

// Plays a wave of matches, which do not depend on each other's results, and
// returns each one's result from its a's side, in order. The trainer plays
// a wave's matches at once (playMatches, azgo/arena.hpp), so the GPU sees
// all their searches together; a match's result does not depend on which
// others share its wave. Never called with an empty wave.
using MatchesFn = std::function<std::vector<MatchResult>(const std::vector<MatchRequest>& wave)>;

// A MatchesFn that plays a wave's matches one after another with `play`.
MatchesFn oneAtATime(MatchFn play);

struct ArenaRoundSettings {
    uint64_t seed = 1;                // the run's seed
    int arenaGames = 20;              // games per opponent
    int maxExtra = 2;                 // arena_opponents
    int calibrationGames = 80;        // per fixed-player calibration and per promotion
    double promoteThreshold = 0.85;
    std::optional<int> bestNetwork;   // gating: the current best generation
};

struct ArenaMatch {
    int opponent;
    MatchResult result;               // from the new network's side
};

struct ArenaRoundResult {
    std::vector<ArenaMatch> matches;  // in selectOpponents order: g-1 first
    int strongest = 0;                // the strongest reference before the round
    bool promoted = false;            // the new network joined the references
};

// Plays settings.calibrationGames games of generation 0 against every fixed
// player (negative ID) in `references` that has never played generation 0,
// so each is rated before it can be chosen. Player p's seed is
// deriveSeed(seed, 0, kPhaseCalibration, p). The calibrations are one wave,
// added to `results` in `references` order. Returns the number calibrated.
int calibrateFixedPlayers(ResultsTable& results, const std::vector<int>& references,
                          const ArenaRoundSettings& settings, const MatchesFn& play);
// The same, one match at a time: calibrateFixedPlayers(..., oneAtATime(play)).
int calibrateFixedPlayers(ResultsTable& results, const std::vector<int>& references,
                          const ArenaRoundSettings& settings, const MatchFn& play);

// The arena round of generation g >= 1:
//   1. calibrateFixedPlayers;
//   2. refit: ratings = fitBradleyTerry(results, 0);
//   3. opponents = selectOpponents(g, ratings, references, maxExtra, bestNetwork);
//   4. arenaGames against each, seed deriveSeed(seed, g, kPhaseArena, opponent),
//      all in one wave;
//   5. promotion: if g scored >= promoteThreshold against the strongest
//      reference, calibrationGames more against it (seed deriveSeed(seed, g,
//      kPhaseCalibration, strongest)), a second wave since it depends on the
//      first, and g joins `references` (kept sorted).
// Every result is added to `results`, in the order above.
ArenaRoundResult runArenaRound(int generation, ResultsTable& results, std::vector<int>& references,
                               const ArenaRoundSettings& settings, const MatchesFn& play);
// The same, one match at a time: runArenaRound(..., oneAtATime(play)).
ArenaRoundResult runArenaRound(int generation, ResultsTable& results, std::vector<int>& references,
                               const ArenaRoundSettings& settings, const MatchFn& play);

// The gating match alone (arena_ladder = off), as AlphaGo Zero's evaluator
// played it: generation g >= 1 plays settings.arenaGames games against
// *settings.bestNetwork with seed deriveSeed(seed, g, kPhaseArena, best),
// g as player a: the request runArenaRound makes for that match, so the
// same games (on one arithmetic; arena engines split a wave by game
// index, so in a smaller wave a game may fall to an engine of another
// fingerprint). One wave of one match, added to `results`. Nothing
// is calibrated or promoted and the reference set is not involved; the
// result's `strongest` is unset (0). Throws std::invalid_argument without
// a best network or for g < 1.
ArenaRoundResult runGatingMatch(int generation, ResultsTable& results, const ArenaRoundSettings& settings,
                                const MatchesFn& play);

// The name of fixed player `player` < 0 among botSimulations.size() bots
// and `referenceNetworks` reference networks, the bots first: bot k
// (player -k) is named after its simulation count, botSimulations[k - 1],
// e.g. "bot100"; reference network j (1-based, player -(bots + j)) is "refj".
// Throws std::invalid_argument for any other player.
std::string fixedPlayerName(int player, const std::vector<int>& botSimulations, int referenceNetworks);

// "vs 149:0.55 120:0.70 bot100:1.00 ref1:0.35": each opponent with the new
// network's score, fixed players by fixedPlayerName.
std::string opponentSummary(const ArenaRoundResult& round, const std::vector<int>& botSimulations,
                            int referenceNetworks = 0);

// references.csv: header "player,kind,simulations,promoted_at,elo", one row
// per reference in ascending order. kind is bot, network (a reference
// network) or checkpoint; simulations is a bot's count (empty otherwise);
// promoted_at is the generation that joined (0 for the initial set); elo is
// from fitBradleyTerry(results, 0), empty while unrated.
std::string referencesCsv(const ResultsTable& results, const std::vector<int>& references,
                          const std::vector<int>& botSimulations, int referenceNetworks = 0);

// reference-networks.csv: header "player,name,path", one row per reference
// network in order, so a player ID can be traced to its file. Reference
// network j (1-based) is player -(numBots + j), named "refj".
std::string referenceNetworksCsv(const std::vector<std::string>& paths, int numBots);

} // namespace azgo
