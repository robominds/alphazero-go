#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "azgo/arena.hpp"

using namespace azgo;

static NetConfig tinyNet() {
    NetConfig c;
    c.boardSize = 5;
    c.history = 2;
    c.blocks = 1;
    c.filters = 4;
    c.valueHidden = 8;
    return c;
}

static ArenaConfig tinyArena() {
    ArenaConfig c;
    c.games = 8;
    c.simulations = 8;
    c.mctsBatch = 4;
    c.temperatureMoves = 0;
    c.komi = 7.5f;
    return c;
}

static bool sameResult(const MatchResult& x, const MatchResult& y) {
    return x.games == y.games && x.scoreA == y.scoreA && x.winsA == y.winsA &&
           x.winsB == y.winsB && x.draws == y.draws && x.distinctGames == y.distinctGames;
}

// Every game is counted once and scoreA agrees with the win/draw counts.
static void checkBookkeeping(const MatchResult& r, int games) {
    assert(r.games == games);
    assert(r.winsA + r.winsB + r.draws == games);
    assert(r.scoreA == r.winsA + 0.5 * r.draws);
    double scoreB = r.winsB + 0.5 * r.draws;
    assert(r.scoreA + scoreB == games);
    assert(r.distinctGames >= 1 && r.distinctGames <= games);
}

static void test_ladder() {
    assert(ladderOpponents(0, 4).empty());
    assert((ladderOpponents(1, 4) == std::vector<int>{0}));
    assert((ladderOpponents(2, 4) == std::vector<int>{1, 0}));
    assert((ladderOpponents(3, 4) == std::vector<int>{2, 1, 0}));
    assert((ladderOpponents(10, 4) == std::vector<int>{9, 8, 6, 2, 0}));
    assert((ladderOpponents(10, 2) == std::vector<int>{9, 8, 0}));
    assert((ladderOpponents(9, 8) == std::vector<int>{8, 7, 5, 1, 0}));
}

static void test_gate() {
    assert(passesGate(11, 20, 0.55));
    assert(!passesGate(10.9, 20, 0.55));
    assert(passesGate(20, 20, 0.55));
    assert(!passesGate(0, 0, 0.55));
}

// Review Focus 2: an exact tie scores half a point for each side.
static void test_match_score() {
    assert(matchScore(0.0f, true) == 0.5 && matchScore(0.0f, false) == 0.5);
    assert(matchScore(+1.0f, true) == 1.0 && matchScore(-1.0f, true) == 0.0);
    assert(matchScore(+1.0f, false) == 0.0 && matchScore(-1.0f, false) == 1.0);
}

static void test_deterministic_across_workers_and_runs() {
    Network a(tinyNet(), 11), b(tinyNet(), 22);
    ArenaConfig cfg = tinyArena();
    MatchResult one = playMatch(a, b, cfg, 42, 1, "naive");
    MatchResult three = playMatch(a, b, cfg, 42, 3, "naive");
    MatchResult again = playMatch(a, b, cfg, 42, 1, "naive");
    checkBookkeeping(one, 8);
    assert(sameResult(one, three));
    assert(sameResult(one, again));
}

static void test_colors_balanced() {
    // With komi 100 on 5x5 (area score at most 25) White wins every game, so
    // A's score counts exactly the games in which A played White.
    Network a(tinyNet(), 11), b(tinyNet(), 22);
    ArenaConfig cfg = tinyArena();
    cfg.komi = 100.0f;
    MatchResult r = playMatch(a, b, cfg, 7, 2, "naive");
    checkBookkeeping(r, 8);
    assert(r.scoreA == 4.0 && r.winsA == 4 && r.winsB == 4 && r.draws == 0);
    cfg.komi = -100.0f;   // Black always wins: again 4 of 8 for A
    r = playMatch(a, b, cfg, 7, 2, "naive");
    assert(r.scoreA == 4.0 && r.winsA == 4 && r.winsB == 4);

    // A network against itself: the bookkeeping still adds up.
    cfg.komi = 7.5f;
    r = playMatch(a, a, cfg, 3, 2, "naive");
    checkBookkeeping(r, 8);
}

static void test_opening_temperature_diversifies() {
    Network a(tinyNet(), 11), b(tinyNet(), 22);
    ArenaConfig cfg = tinyArena();
    cfg.temperatureMoves = 4;
    MatchResult r = playMatch(a, b, cfg, 5, 2, "naive");
    checkBookkeeping(r, 8);
    assert(r.distinctGames > 1);
}

static bool throwsInvalid(const Network& a, const Network& b, const ArenaConfig& cfg) {
    try {
        playMatch(a, b, cfg, 1, 1, "naive");
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

static void test_rejects_bad_setup() {
    NetConfig other = tinyNet();
    other.filters = 8;
    Network a(tinyNet(), 1), wide(other, 2);
    ArenaConfig cfg = tinyArena();
    assert(throwsInvalid(a, wide, cfg));
    cfg.games = 7;
    assert(throwsInvalid(a, a, cfg));
    cfg.games = 0;
    assert(throwsInvalid(a, a, cfg));
}

int main() {
    test_ladder();
    test_gate();
    test_match_score();
    test_deterministic_across_workers_and_runs();
    test_colors_balanced();
    test_opening_temperature_diversifies();
    test_rejects_bad_setup();
    std::printf("test_arena: all tests passed\n");
    return 0;
}
