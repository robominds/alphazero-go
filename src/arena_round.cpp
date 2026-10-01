#include "azgo/arena_round.hpp"
#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>
#include "azgo/rng.hpp"
#include "azgo/run_state.hpp"

namespace azgo {

namespace {

std::string botName(int player, const std::vector<int>& botSimulations) {
    if (player >= 0 || static_cast<size_t>(-player) > botSimulations.size())
        throw std::invalid_argument("no bot " + std::to_string(player) + " in bot_simulations");
    return "bot" + std::to_string(botSimulations[static_cast<size_t>(-player - 1)]);
}

} // namespace

int calibrateBots(ResultsTable& results, const std::vector<int>& references,
                  const ArenaRoundSettings& settings, const MatchFn& play) {
    std::set<int> calibrated;   // bots that already have results against generation 0
    for (const ResultsTable::Pair& p : results.pairs())
        if (p.a < 0 && p.b == 0) calibrated.insert(p.a);
    int count = 0;
    for (int bot : references) {
        if (bot >= 0 || calibrated.count(bot)) continue;
        const MatchResult r = play(0, bot, settings.calibrationGames,
                                   deriveSeed(settings.seed, 0, kPhaseCalibration,
                                              static_cast<uint64_t>(bot)));
        results.addMatch(0, bot, r.games, r.scoreA);
        ++count;
    }
    return count;
}

ArenaRoundResult runArenaRound(int generation, ResultsTable& results, std::vector<int>& references,
                               const ArenaRoundSettings& settings, const MatchFn& play) {
    // Normally a no-op: bots are calibrated when a run starts. A run resumed
    // from a state.bin written before bots existed calibrates here, once.
    calibrateBots(results, references, settings, play);

    // The refit of every result so far: identical on resume and on re-rating.
    const std::map<int, double> ratings = fitBradleyTerry(results, 0);
    ArenaRoundResult round;
    round.strongest = strongestReference(ratings, references);
    const std::vector<int> opponents =
        selectOpponents(generation, ratings, references, settings.maxExtra, settings.bestNetwork);

    double scoreVsStrongest = 0;
    for (int o : opponents) {
        const MatchResult r = play(generation, o, settings.arenaGames,
                                   deriveSeed(settings.seed, static_cast<uint64_t>(generation),
                                              kPhaseArena, static_cast<uint64_t>(o)));
        results.addMatch(generation, o, r.games, r.scoreA);
        if (o == round.strongest) scoreVsStrongest = r.scoreA;
        round.matches.push_back({o, r});
    }

    // Promotion: a clear win against the strongest reference makes the new
    // network the next yardstick. The extra games pin its rating down before
    // later generations are measured against it. promote_threshold is a float
    // config key (0.85f is 0.8500000238...), so passesGate's tolerance is what
    // accepts 17 of 20.
    if (passesGate(scoreVsStrongest, settings.arenaGames, settings.promoteThreshold)) {
        const MatchResult r = play(generation, round.strongest, settings.calibrationGames,
                                   deriveSeed(settings.seed, static_cast<uint64_t>(generation),
                                              kPhaseCalibration,
                                              static_cast<uint64_t>(round.strongest)));
        results.addMatch(generation, round.strongest, r.games, r.scoreA);
        references.insert(std::upper_bound(references.begin(), references.end(), generation),
                          generation);
        round.promoted = true;
    }
    return round;
}

std::string opponentSummary(const ArenaRoundResult& round, const std::vector<int>& botSimulations) {
    std::string out = "vs";
    for (const ArenaMatch& m : round.matches) {
        const std::string name =
            m.opponent < 0 ? botName(m.opponent, botSimulations) : std::to_string(m.opponent);
        char score[32];
        std::snprintf(score, sizeof score, ":%.2f", m.result.scoreA / m.result.games);
        out += " " + name + score;
    }
    return out;
}

std::string referencesCsv(const ResultsTable& results, const std::vector<int>& references,
                          const std::vector<int>& botSimulations) {
    const std::map<int, double> ratings = fitBradleyTerry(results, 0);
    std::string out = "player,kind,simulations,promoted_at,elo\n";
    for (int r : references) {
        std::string simulations;
        if (r < 0) simulations = botName(r, botSimulations).substr(3);   // "bot100" -> "100"
        char elo[32] = "";
        auto it = ratings.find(r);
        if (it != ratings.end()) std::snprintf(elo, sizeof elo, "%.1f", it->second + 0.0);   // -0.0 -> 0.0
        // A generation joins right after its own arena round; the initial set at 0.
        out += std::to_string(r) + (r < 0 ? ",bot," : ",checkpoint,") + simulations + "," +
               std::to_string(r > 0 ? r : 0) + "," + elo + "\n";
    }
    return out;
}

} // namespace azgo
