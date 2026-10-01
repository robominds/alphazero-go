// One arena round, shared by live training and by re-rating a finished run,
// so both make the same choices from the same results. A round calibrates
// any bot that has never played generation 0, refits every rating, picks the
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
// result from a's side. Players are generations (>= 0) and bots (-1, -2, ...).
// The trainer plays real networks; tests can return made-up results.
using MatchFn = std::function<MatchResult(int a, int b, int games, uint64_t seed)>;

struct ArenaRoundSettings {
    uint64_t seed = 1;                // the run's seed
    int arenaGames = 20;              // games per opponent
    int maxExtra = 2;                 // arena_opponents
    int calibrationGames = 80;        // per bot calibration and per promotion
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

// Plays settings.calibrationGames games of generation 0 against every bot in
// `references` that has never played generation 0, so each bot is rated
// before it can be chosen. Bot b's seed is deriveSeed(seed, 0,
// kPhaseCalibration, b). Returns the number of bots calibrated.
int calibrateBots(ResultsTable& results, const std::vector<int>& references,
                  const ArenaRoundSettings& settings, const MatchFn& play);

// The arena round of generation g >= 1:
//   1. calibrateBots;
//   2. refit: ratings = fitBradleyTerry(results, 0);
//   3. opponents = selectOpponents(g, ratings, references, maxExtra, bestNetwork);
//   4. arenaGames against each, seed deriveSeed(seed, g, kPhaseArena, opponent);
//   5. promotion: if g scored >= promoteThreshold against the strongest
//      reference, calibrationGames more against it (seed deriveSeed(seed, g,
//      kPhaseCalibration, strongest)), and g joins `references` (kept sorted).
// Every result is added to `results`.
ArenaRoundResult runArenaRound(int generation, ResultsTable& results, std::vector<int>& references,
                               const ArenaRoundSettings& settings, const MatchFn& play);

// "vs 149:0.55 120:0.70 bot100:1.00": each opponent with the new network's
// score. Bot b is named after its simulation count, botSimulations[-b - 1].
std::string opponentSummary(const ArenaRoundResult& round, const std::vector<int>& botSimulations);

// references.csv: header "player,kind,simulations,promoted_at,elo", one row
// per reference in ascending order. kind is bot or checkpoint; simulations is
// a bot's count (empty for a checkpoint); promoted_at is the generation that
// joined (0 for the initial set); elo is from fitBradleyTerry(results, 0),
// empty while unrated.
std::string referencesCsv(const ResultsTable& results, const std::vector<int>& references,
                          const std::vector<int>& botSimulations);

} // namespace azgo
