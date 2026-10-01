#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include "azgo/arena_round.hpp"
#include "azgo/rng.hpp"
#include "azgo/run_state.hpp"

using namespace azgo;

// A synthetic arena: players have fixed true Elo ratings and every match
// scores exactly its Bradley-Terry expectation (a fractional score is fine
// for ResultsTable). Each call is recorded so tests can check who played
// whom, how many games and with which seed.
struct FakeArena {
    std::map<int, double> truth;
    std::vector<std::tuple<int, int, int, uint64_t>> calls;   // (a, b, games, seed)
    MatchFn fn() {
        return [this](int a, int b, int games, uint64_t seed) {
            calls.emplace_back(a, b, games, seed);
            MatchResult r;
            r.games = games;
            r.scoreA = games / (1.0 + std::pow(10.0, (truth.at(b) - truth.at(a)) / 400.0));
            r.distinctGames = games;
            return r;
        };
    }
};

// A match whose score is fixed in advance, e.g. 17 of 20, whoever plays.
static MatchFn fixedScore(double fraction, std::vector<std::tuple<int, int, int>>* calls) {
    return [fraction, calls](int a, int b, int games, uint64_t) {
        calls->emplace_back(a, b, games);
        MatchResult r;
        r.games = games;
        r.scoreA = fraction * games;
        return r;
    };
}

static ArenaRoundSettings settings() {
    ArenaRoundSettings s;
    s.seed = 7;
    s.arenaGames = 20;
    s.maxExtra = 1;
    s.calibrationGames = 80;
    s.promoteThreshold = 0.85f;   // a float, as Config stores it
    return s;
}

static int gamesBetween(const ResultsTable& t, int a, int b) {
    for (const ResultsTable::Pair& p : t.pairs())
        if (p.a == std::min(a, b) && p.b == std::max(a, b)) return p.games;
    return 0;
}

static void test_calibration() {
    FakeArena arena;
    arena.truth = {{-2, -200.0}, {-1, 50.0}, {0, 0.0}};
    ResultsTable results;
    const std::vector<int> refs{-2, -1, 0};
    assert(calibrateBots(results, refs, settings(), arena.fn()) == 2);
    assert(gamesBetween(results, -2, 0) == 80 && gamesBetween(results, -1, 0) == 80);
    // Generation 0 is player a, ascending bot order, the calibration seed.
    assert(arena.calls.size() == 2);
    assert(arena.calls[0] == std::make_tuple(0, -2, 80, deriveSeed(7, 0, kPhaseCalibration,
                                                                   static_cast<uint64_t>(-2))));
    assert(std::get<1>(arena.calls[1]) == -1);
    // Already calibrated: nothing more to play.
    assert(calibrateBots(results, refs, settings(), arena.fn()) == 0);
    assert(arena.calls.size() == 2);
    // No bots, nothing to do.
    ResultsTable empty;
    assert(calibrateBots(empty, {0}, settings(), arena.fn()) == 0 && empty.empty());
}

static void test_round_uses_selection_and_live_seeds() {
    FakeArena arena;
    arena.truth = {{-1, 50.0}, {0, 0.0}, {1, 100.0}};
    ResultsTable results;
    std::vector<int> refs{-1, 0};
    // The bot is uncalibrated, so the round calibrates it first.
    ArenaRoundResult round = runArenaRound(1, results, refs, settings(), arena.fn());
    assert(arena.calls.size() == 3);
    assert(std::get<0>(arena.calls[0]) == 0 && std::get<1>(arena.calls[0]) == -1);
    // Then g-1 = 0, then the strongest reference: the bot (rated about +50).
    assert(round.strongest == -1 && !round.promoted);
    assert(round.matches.size() == 2 && round.matches[0].opponent == 0 && round.matches[1].opponent == -1);
    assert(arena.calls[1] == std::make_tuple(1, 0, 20, deriveSeed(7, 1, kPhaseArena, 0)));
    assert(arena.calls[2] == std::make_tuple(1, -1, 20, deriveSeed(7, 1, kPhaseArena,
                                                                   static_cast<uint64_t>(-1))));
    assert(gamesBetween(results, 0, 1) == 20 && gamesBetween(results, -1, 1) == 20);
    assert((refs == std::vector<int>{-1, 0}));
}

// Review Focus 3: a score exactly at the threshold.
static void test_promotion() {
    // 17 of 20 against the strongest reference reaches 0.85 exactly: promoted,
    // with exactly calibration_games more games against it.
    std::vector<std::tuple<int, int, int>> calls;
    ResultsTable results;
    std::vector<int> refs{0};
    ArenaRoundResult round = runArenaRound(1, results, refs, settings(), fixedScore(0.85, &calls));
    assert(round.promoted && round.strongest == 0);
    assert((refs == std::vector<int>{0, 1}));
    assert(calls.size() == 2 && calls[1] == std::make_tuple(1, 0, 80));
    assert(gamesBetween(results, 0, 1) == 100);

    // 16 of 20 is below the threshold: no extra games, no promotion.
    calls.clear();
    ResultsTable below;
    std::vector<int> refs2{0};
    round = runArenaRound(1, below, refs2, settings(), fixedScore(0.8, &calls));
    assert(!round.promoted && (refs2 == std::vector<int>{0}));
    assert(calls.size() == 1 && gamesBetween(below, 0, 1) == 20);

    // A promoted generation is inserted in order, after the bots.
    calls.clear();
    ResultsTable t;
    t.addMatch(0, -1, 80, 40.0);
    std::vector<int> refs3{-1, 0};
    round = runArenaRound(3, t, refs3, settings(), fixedScore(1.0, &calls));
    assert(round.promoted && (refs3 == std::vector<int>{-1, 0, 3}));
}

// Generation g is truly 100 g Elo strong, the bot 50. A generation 400 above
// the strongest reference scores E = 0.909 >= 0.85 and is promoted; 300 above
// gives E = 0.849, just short. So generations 4, 8, 12 and 16 are promoted,
// and once 4 is a reference, generation 0 (expected score > 0.9) loses its
// place as the one extra opponent to a closer match.
static void test_saturated_anchor_is_dropped() {
    FakeArena arena;
    arena.truth = {{-1, 50.0}};
    for (int g = 0; g <= 16; ++g) arena.truth[g] = 100.0 * g;
    ResultsTable results;
    std::vector<int> refs{-1, 0};
    calibrateBots(results, refs, settings(), arena.fn());
    std::vector<int> promoted;
    for (int g = 1; g <= 16; ++g) {
        ArenaRoundResult round = runArenaRound(g, results, refs, settings(), arena.fn());
        std::vector<int> opponents;
        for (const ArenaMatch& m : round.matches) opponents.push_back(m.opponent);
        if (round.promoted) promoted.push_back(g);
        if (g == 1) assert((opponents == std::vector<int>{0, -1}));
        if (g == 4) assert((opponents == std::vector<int>{3, -1, 0}));
        if (g == 5) assert((opponents == std::vector<int>{4, -1}));
        if (g == 9) assert((opponents == std::vector<int>{8, 4}));
        if (g >= 5) assert(std::find(opponents.begin(), opponents.end(), 0) == opponents.end());
    }
    assert((promoted == std::vector<int>{4, 8, 12, 16}));
    assert((refs == std::vector<int>{-1, 0, 4, 8, 12, 16}));
}

static void test_gating_appends_best() {
    FakeArena arena;
    arena.truth = {{0, 0.0}, {1, 100.0}, {2, 200.0}, {3, 300.0}};
    ResultsTable results;
    std::vector<int> refs{0};
    ArenaRoundSettings s = settings();
    s.maxExtra = 0;
    s.bestNetwork = 0;   // already g-1 and the strongest reference: played once
    ArenaRoundResult round = runArenaRound(1, results, refs, s, arena.fn());
    assert(round.matches.size() == 1 && gamesBetween(results, 0, 1) == 20);
    s.bestNetwork = 1;   // generation 3 plays 2 (g-1), 0 (strongest), then 1 (best)
    round = runArenaRound(3, results, refs, s, arena.fn());
    assert(round.matches.size() == 3 && round.matches[2].opponent == 1);
}

static void test_summary_and_references_csv() {
    ArenaRoundResult round;
    MatchResult r;
    r.games = 20;
    r.scoreA = 11;
    round.matches.push_back({149, r});
    r.scoreA = 20;
    round.matches.push_back({-1, r});
    assert(opponentSummary(round, {100}) == "vs 149:0.55 bot100:1.00");

    ResultsTable t;
    t.addMatch(0, -1, 10, 5.0);   // the bot draws level with generation 0: both 0 Elo
    const std::string csv = referencesCsv(t, {-1, 0, 5}, {100});
    assert(csv == "player,kind,simulations,promoted_at,elo\n"
                  "-1,bot,100,0,0.0\n"
                  "0,checkpoint,,0,0.0\n"
                  "5,checkpoint,,5,\n");   // 5 has no results yet: unrated
}

int main() {
    test_calibration();
    test_round_uses_selection_and_live_seeds();
    test_promotion();
    test_saturated_anchor_is_dropped();
    test_gating_appends_best();
    test_summary_and_references_csv();
    std::printf("test_arena_round: all tests passed\n");
    return 0;
}
