#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <stdexcept>
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
    assert(calibrateFixedPlayers(results, refs, settings(), arena.fn()) == 2);
    assert(gamesBetween(results, -2, 0) == 80 && gamesBetween(results, -1, 0) == 80);
    // Generation 0 is player a, ascending bot order, the calibration seed.
    assert(arena.calls.size() == 2);
    assert(arena.calls[0] == std::make_tuple(0, -2, 80, deriveSeed(7, 0, kPhaseCalibration,
                                                                   static_cast<uint64_t>(-2))));
    assert(std::get<1>(arena.calls[1]) == -1);
    // Already calibrated: nothing more to play.
    assert(calibrateFixedPlayers(results, refs, settings(), arena.fn()) == 0);
    assert(arena.calls.size() == 2);
    // No bots, nothing to do.
    ResultsTable empty;
    assert(calibrateFixedPlayers(empty, {0}, settings(), arena.fn()) == 0 && empty.empty());
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
    calibrateFixedPlayers(results, refs, settings(), arena.fn());
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

// The fixed players are the bots, then the reference networks: with one bot
// and two reference networks, -1 is the bot, -2 is ref1 and -3 is ref2.
static void test_fixed_player_names() {
    assert(fixedPlayerName(-1, {100}, 2) == "bot100");
    assert(fixedPlayerName(-2, {100}, 2) == "ref1");
    assert(fixedPlayerName(-3, {100}, 2) == "ref2");
    assert(fixedPlayerName(-1, {}, 1) == "ref1");   // no bots: reference networks start at -1
    assert(fixedPlayerName(-2, {50, 200}, 0) == "bot200");
    for (int bad : {-4, 0, 3}) {
        bool threw = false;
        try {
            fixedPlayerName(bad, {100}, 2);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
}

// A reference network is calibrated against generation 0 like a bot, with
// the same seed rule, and is then chosen by the same expected-score rule:
// ref1 (-2) sits where generation 1 is expected to score 0.70.
static void test_reference_networks_are_calibrated_and_chosen() {
    FakeArena arena;
    const double nearTarget = 400.0 * std::log10(1.0 / 0.7 - 1.0);   // -147: E(0 vs it) = 0.7
    arena.truth = {{-3, 900.0}, {-2, nearTarget}, {-1, -300.0}, {0, 0.0}, {1, 100.0}};
    ResultsTable results;
    std::vector<int> refs = initialReferences(3);   // bot -1, ref1 -2, ref2 -3, generation 0
    assert(calibrateFixedPlayers(results, refs, settings(), arena.fn()) == 3);
    for (int p : {-3, -2, -1}) assert(gamesBetween(results, p, 0) == 80);
    assert(arena.calls[0] == std::make_tuple(0, -3, 80, deriveSeed(7, 0, kPhaseCalibration,
                                                                   static_cast<uint64_t>(-3))));
    ArenaRoundResult round = runArenaRound(1, results, refs, settings(), arena.fn());
    // g-1, then the strongest reference (ref2), then the one extra: ref1,
    // whose expected score 0.70 beats the bot's 0.85.
    assert(round.strongest == -3);
    assert(round.matches.size() == 3 && round.matches[0].opponent == 0 &&
           round.matches[1].opponent == -3 && round.matches[2].opponent == -2);
    const std::string summary = opponentSummary(round, {100}, 2);
    assert(summary.rfind("vs 0:", 0) == 0 && summary.find(" ref2:") != std::string::npos &&
           summary.find(" ref1:") != std::string::npos);
    // Fixed players are never promoted or dropped.
    assert((refs == std::vector<int>{-3, -2, -1, 0}));
}

// references.csv names a reference network's kind "network", with no
// simulation count; reference-networks.csv maps its player ID to its file.
static void test_reference_network_csvs() {
    ResultsTable t;
    t.addMatch(-2, 0, 10, 5.0);
    t.addMatch(-1, 0, 10, 5.0);
    assert(referencesCsv(t, {-2, -1, 0}, {100}, 1) ==
           "player,kind,simulations,promoted_at,elo\n"
           "-2,network,,0,0.0\n"
           "-1,bot,100,0,0.0\n"
           "0,checkpoint,,0,0.0\n");
    assert(referenceNetworksCsv({"runs/a/gen_0200.azn", "/b.azn"}, 1) ==
           "player,name,path\n"
           "-2,ref1,runs/a/gen_0200.azn\n"
           "-3,ref2,/b.azn\n");
    assert(referenceNetworksCsv({"x.azn"}, 0) == "player,name,path\n-1,ref1,x.azn\n");
}


// The same synthetic arena seen through MatchesFn: each call is one wave of
// matches that the trainer plays at once. The wave's matches are recorded
// in order, and each scores as FakeArena would.
struct WaveArena {
    FakeArena arena;
    std::vector<std::vector<MatchRequest>> waves;
    MatchesFn fn() {
        return [this](const std::vector<MatchRequest>& wave) {
            assert(!wave.empty());   // never asked to play nothing
            waves.push_back(wave);
            std::vector<MatchResult> out;
            const MatchFn one = arena.fn();
            for (const MatchRequest& m : wave) out.push_back(one(m.a, m.b, m.games, m.seed));
            return out;
        };
    }
};

static bool sameRequest(const MatchRequest& m, const std::tuple<int, int, int, uint64_t>& call) {
    return std::make_tuple(m.a, m.b, m.games, m.seed) == call;
}

// A round played in waves: the fixed players' calibrations in one wave, then
// every opponent in one wave, then the promotion games. The matches, seeds,
// results table, opponents and references are exactly those of the same
// round played one match at a time (MatchFn), the order matches were added
// to the table included.
static void test_round_in_waves_equals_one_at_a_time() {
    const double nearTarget = 400.0 * std::log10(1.0 / 0.7 - 1.0);
    const std::map<int, double> truth = {{-3, 300.0}, {-2, nearTarget}, {-1, -300.0}, {0, 0.0},
                                         {1, 500.0}, {2, 900.0}, {3, 1300.0}};
    WaveArena waves;
    waves.arena.truth = truth;
    FakeArena single;
    single.truth = truth;
    ArenaRoundSettings s = settings();
    s.maxExtra = 2;
    ResultsTable wavesTable, singleTable;
    std::vector<int> wavesRefs = initialReferences(3), singleRefs = initialReferences(3);

    assert(calibrateFixedPlayers(wavesTable, wavesRefs, s, waves.fn()) == 3);
    assert(calibrateFixedPlayers(singleTable, singleRefs, s, single.fn()) == 3);
    assert(waves.waves.size() == 1 && waves.waves[0].size() == 3);   // every calibration at once
    assert(calibrateFixedPlayers(wavesTable, wavesRefs, s, waves.fn()) == 0);
    assert(waves.waves.size() == 1);   // nothing left: no wave at all

    bool promotedOnce = false;
    for (int g = 1; g <= 3; ++g) {
        s.bestNetwork = g == 3 ? std::optional<int>(1) : std::nullopt;   // gating in the last round
        const size_t before = waves.waves.size();
        const ArenaRoundResult w = runArenaRound(g, wavesTable, wavesRefs, s, waves.fn());
        const ArenaRoundResult o = runArenaRound(g, singleTable, singleRefs, s, single.fn());
        assert(w.strongest == o.strongest && w.promoted == o.promoted);
        assert(w.matches.size() == o.matches.size());
        for (size_t i = 0; i < w.matches.size(); ++i) {
            assert(w.matches[i].opponent == o.matches[i].opponent);
            assert(w.matches[i].result.scoreA == o.matches[i].result.scoreA);
        }
        // One wave of every opponent, then one of the promotion games.
        assert(waves.waves.size() == before + (w.promoted ? 2 : 1));
        assert(waves.waves[before].size() == w.matches.size());
        if (w.promoted) assert(waves.waves[before + 1].size() == 1);
        promotedOnce = promotedOnce || w.promoted;
        if (g == 3) assert(w.matches.size() >= 3 && w.matches.back().opponent == 1);   // gating: the best last
    }
    assert(promotedOnce);
    assert(wavesRefs == singleRefs);
    assert(wavesTable.toCsv() == singleTable.toCsv());
    // The same matches with the same seeds, in the same order.
    std::vector<MatchRequest> flat;
    for (const std::vector<MatchRequest>& wave : waves.waves) flat.insert(flat.end(), wave.begin(), wave.end());
    assert(flat.size() == single.calls.size());
    for (size_t i = 0; i < flat.size(); ++i) assert(sameRequest(flat[i], single.calls[i]));
}

// oneAtATime turns a single-match function into a MatchesFn that plays a
// wave's matches in order.
static void test_one_at_a_time_adapter() {
    FakeArena arena;
    arena.truth = {{0, 0.0}, {1, 100.0}, {2, 200.0}};
    const MatchesFn play = oneAtATime(arena.fn());
    const std::vector<MatchResult> r = play({{1, 0, 20, 5}, {2, 1, 10, 6}});
    assert(r.size() == 2 && r[0].games == 20 && r[1].games == 10);
    assert(arena.calls.size() == 2 && arena.calls[1] == std::make_tuple(2, 1, 10, uint64_t{6}));
}

// arena_ladder = off: the gating match alone, one wave of one match, with
// the seed and colours the full round gives that match, so its result is
// the round's gating result. No calibration (the bot below never played
// generation 0), no other opponent, no promotion; the reference set is
// not touched.
static void test_gating_match_alone() {
    const std::map<int, double> truth = {{-1, 50.0}, {0, 0.0}, {1, 100.0}, {2, 150.0}, {3, 400.0}};
    WaveArena waves;
    waves.arena.truth = truth;
    ArenaRoundSettings s = settings();
    s.bestNetwork = 1;
    ResultsTable results;
    results.addMatch(0, 1, 20, 6.0);
    results.addMatch(1, 2, 20, 9.0);
    const std::string before = results.toCsv();
    ArenaRoundResult round = runGatingMatch(3, results, s, waves.fn());
    assert(waves.waves.size() == 1 && waves.waves[0].size() == 1);
    const MatchRequest& m = waves.waves[0][0];
    assert(m.a == 3 && m.b == 1 && m.games == 20 && m.seed == deriveSeed(7, 3, kPhaseArena, 1));
    assert(round.matches.size() == 1 && round.matches[0].opponent == 1 && !round.promoted);
    assert(round.matches[0].result.games == 20);
    assert(gamesBetween(results, 1, 3) == 20 && gamesBetween(results, -1, 0) == 0);
    ResultsTable expected = ResultsTable::fromCsv(before);
    expected.addMatch(3, 1, 20, round.matches[0].result.scoreA);
    assert(results.toCsv() == expected.toCsv());

    // The full round, gating on, plays that same request and gets that same result.
    WaveArena full;
    full.arena.truth = truth;
    ResultsTable fullResults = ResultsTable::fromCsv(before);
    std::vector<int> refs{-1, 0};
    const ArenaRoundResult r = runArenaRound(3, fullResults, refs, s, full.fn());
    bool same = false;
    for (const std::vector<MatchRequest>& wave : full.waves)
        for (const MatchRequest& f : wave)
            same = same || (f.a == m.a && f.b == m.b && f.games == m.games && f.seed == m.seed);
    assert(same);
    for (const ArenaMatch& a : r.matches)
        if (a.opponent == 1) assert(a.result.scoreA == round.matches[0].result.scoreA);

    // The summary names the best network with the new network's score.
    assert(opponentSummary(round, {}).rfind("vs 1:", 0) == 0);

    // Without a best network (no gating) there is no gating match.
    s.bestNetwork.reset();
    bool threw = false;
    try {
        runGatingMatch(3, results, s, waves.fn());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw && waves.waves.size() == 1);
}

int main() {
    test_calibration();
    test_round_uses_selection_and_live_seeds();
    test_promotion();
    test_saturated_anchor_is_dropped();
    test_gating_appends_best();
    test_summary_and_references_csv();
    test_fixed_player_names();
    test_reference_networks_are_calibrated_and_chosen();
    test_reference_network_csvs();
    test_round_in_waves_equals_one_at_a_time();
    test_one_at_a_time_adapter();
    test_gating_match_alone();
    std::printf("test_arena_round: all tests passed\n");
    return 0;
}
