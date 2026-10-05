#include <cassert>
#include <cmath>
#include <cstdio>
#include <map>
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

static void test_gate() {
    assert(passesGate(11, 20, 0.55));
    assert(!passesGate(10.9, 20, 0.55));
    assert(passesGate(20, 20, 0.55));
    assert(!passesGate(0, 0, 0.55));
    // The trainer passes a float key: 0.55f is 0.5500000119, 0.85f is 0.8500000238.
    assert(passesGate(11, 20, 0.55f));
    assert(!passesGate(10.9, 20, 0.55f));
    assert(!passesGate(16.9, 20, 0.85f));
    assert(passesGate(17, 20, 0.85f));
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

// Random symmetry draws from each game's own Rng, so a match stays
// deterministic for its seed and independent of the worker count. An
// untrained network is not equivariant, so the draws change its games.
// A bot's uniform priors look the same under every transform; it gives
// nothing to average and draws nothing.
static void test_random_symmetry_matches() {
    Network a(tinyNet(), 11), b(tinyNet(), 22);
    ArenaConfig cfg = tinyArena();
    cfg.temperatureMoves = 2;
    const MatchResult plain = playMatch(a, b, cfg, 42, 1, "naive");
    cfg.randomSymmetry = true;
    const MatchResult one = playMatch(a, b, cfg, 42, 1, "naive");
    checkBookkeeping(one, 8);
    assert(sameResult(one, playMatch(a, b, cfg, 42, 3, "naive")));
    assert(sameResult(one, playMatch(a, b, cfg, 42, 1, "naive")));
    assert(!sameResult(one, plain));
    assert(sameResult(playMatch(ArenaPlayer::network(a), ArenaPlayer::bot(16), cfg, 5, 1, "naive"),
                      playMatch(ArenaPlayer::network(a), ArenaPlayer::bot(16), cfg, 5, 3, "naive")));
}

// Random symmetry for one side only (ArenaPlayer::randomSymmetry), so a
// network can be screened against itself with the search change on one side.
// A seat searches with it if either the config or its player asks.
static void test_one_sided_random_symmetry() {
    Network a(tinyNet(), 11), b(tinyNet(), 22);
    // Matches are deterministic, so these are fixed comparisons, not
    // statistical ones. 8 games give only a coarse summary, so the seed is one
    // whose summaries tell off, on and each one-sided match apart.
    ArenaConfig cfg = tinyArena();
    ArenaConfig cfgOn = cfg;
    cfgOn.randomSymmetry = true;
    const ArenaPlayer plainA = ArenaPlayer::network(a), symA = ArenaPlayer::network(a, 0, true);
    const ArenaPlayer plainB = ArenaPlayer::network(b), symB = ArenaPlayer::network(b, 0, true);
    assert(!plainA.randomSymmetry && symA.randomSymmetry);

    const MatchResult off = playMatch(plainA, plainB, cfg, 5, 1, "naive");
    const MatchResult on = playMatch(plainA, plainB, cfgOn, 5, 1, "naive");
    // Both players asking is the same as the config asking, and the config
    // still turns it on for a player that does not ask.
    assert(sameResult(on, playMatch(symA, symB, cfg, 5, 1, "naive")));
    assert(sameResult(on, playMatch(symA, plainB, cfgOn, 5, 1, "naive")));

    for (const MatchResult& oneSided : {playMatch(symA, plainB, cfg, 5, 1, "naive"),
                                        playMatch(plainA, symB, cfg, 5, 1, "naive")}) {
        checkBookkeeping(oneSided, 8);
        assert(!sameResult(oneSided, off));
        assert(!sameResult(oneSided, on));
    }
    // Deterministic for the seed and independent of the worker count.
    assert(sameResult(playMatch(symA, plainB, cfg, 5, 1, "naive"),
                      playMatch(symA, plainB, cfg, 5, 3, "naive")));
    assert(sameResult(playMatch(plainA, symB, cfg, 5, 1, "naive"),
                      playMatch(plainA, symB, cfg, 5, 3, "naive")));

    // The network against itself, symmetry on one side: the screening match.
    const MatchResult self = playMatch(symA, plainA, cfg, 5, 1, "naive");
    checkBookkeeping(self, 8);
    assert(sameResult(self, playMatch(symA, plainA, cfg, 5, 3, "naive")));
    assert(!sameResult(self, playMatch(plainA, plainA, cfg, 5, 1, "naive")));

    // Against a bot the flag reaches its own seat: a's flag alone equals the
    // config asking (a bot never searches with symmetry), from either side.
    const ArenaPlayer bot = ArenaPlayer::bot(16);
    assert(sameResult(playMatch(symA, bot, cfg, 5, 1, "naive"),
                      playMatch(plainA, bot, cfgOn, 5, 1, "naive")));
    assert(sameResult(playMatch(bot, symA, cfg, 5, 1, "naive"),
                      playMatch(bot, plainA, cfgOn, 5, 1, "naive")));
    // A bot ignores the flag: asking for it changes nothing.
    ArenaPlayer symBot = bot;
    symBot.randomSymmetry = true;
    assert(sameResult(playMatch(plainA, symBot, cfg, 5, 1, "naive"),
                      playMatch(plainA, bot, cfg, 5, 1, "naive")));
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

// A bot is a full arena player: deterministic for a seed, independent of the
// worker count, and playable from either side.
static void test_bot_matches() {
    Network a(tinyNet(), 11);
    ArenaConfig cfg = tinyArena();
    cfg.temperatureMoves = 2;
    const ArenaPlayer net = ArenaPlayer::network(a), bot = ArenaPlayer::bot(16);
    MatchResult one = playMatch(net, bot, cfg, 42, 1, "naive");
    MatchResult three = playMatch(net, bot, cfg, 42, 3, "naive");
    MatchResult again = playMatch(net, bot, cfg, 42, 1, "naive");
    checkBookkeeping(one, 8);
    assert(sameResult(one, three));
    assert(sameResult(one, again));
    checkBookkeeping(playMatch(bot, net, cfg, 42, 2, "naive"), 8);

    // With komi 100 White wins every game, so each side scores 4 of 8
    // whichever player is the bot.
    cfg.komi = 100.0f;
    MatchResult r = playMatch(bot, net, cfg, 7, 2, "naive");
    assert(r.scoreA == 4.0 && r.winsA == 4 && r.winsB == 4);
}

// The two-network signature is a thin wrapper: same result as two ArenaPlayers.
static void test_network_wrapper() {
    Network a(tinyNet(), 11), b(tinyNet(), 22);
    ArenaConfig cfg = tinyArena();
    assert(sameResult(playMatch(a, b, cfg, 9, 2, "naive"),
                      playMatch(ArenaPlayer::network(a), ArenaPlayer::network(b), cfg, 9, 2, "naive")));
}

// A network player's simulations override. 0 means ArenaConfig::simulations,
// so an explicit override equal to it must reproduce the default match
// exactly; that also proves the plumbing is byte-neutral for existing callers.
static void test_simulations_override() {
    Network a(tinyNet(), 11), b(tinyNet(), 22);
    ArenaConfig cfg = tinyArena();   // 8 simulations per move
    const MatchResult base = playMatch(a, b, cfg, 42, 1, "naive");
    assert(sameResult(base, playMatch(ArenaPlayer::network(a, 0), ArenaPlayer::network(b, 0),
                                      cfg, 42, 1, "naive")));
    assert(sameResult(base, playMatch(ArenaPlayer::network(a, 8), ArenaPlayer::network(b),
                                      cfg, 42, 1, "naive")));

    // The override replaces ArenaConfig::simulations for exactly its own side.
    ArenaConfig cfg64 = cfg;
    cfg64.simulations = 64;
    assert(sameResult(playMatch(ArenaPlayer::network(a, 64), ArenaPlayer::network(b, 64), cfg, 42, 1, "naive"),
                      playMatch(a, b, cfg64, 42, 1, "naive")));
    // A at 64, B at the config's 8, equals the config at 64 with B overridden to 8.
    assert(sameResult(playMatch(ArenaPlayer::network(a, 64), ArenaPlayer::network(b), cfg, 42, 1, "naive"),
                      playMatch(ArenaPlayer::network(a), ArenaPlayer::network(b, 8), cfg64, 42, 1, "naive")));

    // The override reaches the search: the same network at 1 simulation
    // against itself at 64 cannot repeat the 8-versus-8 games. The match is
    // deterministic for the seed, so this is a fixed comparison, not a
    // statistical one; an ignored override would give the symmetric result.
    const MatchResult sym = playMatch(a, a, cfg, 42, 1, "naive");
    const ArenaPlayer weak = ArenaPlayer::network(a, 1), strong = ArenaPlayer::network(a, 64);
    const MatchResult asym = playMatch(weak, strong, cfg, 42, 1, "naive");
    checkBookkeeping(asym, 8);
    assert(!sameResult(asym, sym));
    // Either side's override counts, and it is deterministic across workers.
    assert(sameResult(asym, playMatch(weak, strong, cfg, 42, 3, "naive")));
    assert(sameResult(asym, playMatch(weak, strong, cfg, 42, 1, "naive")));
    const MatchResult flipped = playMatch(strong, weak, cfg, 42, 1, "naive");
    checkBookkeeping(flipped, 8);
    assert(!sameResult(flipped, sym));
    // A bot keeps its own count whatever ArenaConfig says.
    assert(sameResult(playMatch(ArenaPlayer::network(a, 64), ArenaPlayer::bot(16), cfg, 5, 2, "naive"),
                      playMatch(ArenaPlayer::network(a, 64), ArenaPlayer::bot(16), cfg, 5, 1, "naive")));
}

static bool playersThrow(const ArenaPlayer& a, const ArenaPlayer& b) {
    try {
        playMatch(a, b, tinyArena(), 1, 1, "naive");
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

static void test_rejects_bad_players() {
    Network a(tinyNet(), 1);
    assert(playersThrow(ArenaPlayer::bot(8), ArenaPlayer::bot(8)));            // nobody has a network
    assert(playersThrow(ArenaPlayer::network(a), ArenaPlayer::bot(0)));        // a bot must search
    assert(playersThrow(ArenaPlayer::bot(-3), ArenaPlayer::network(a)));
    assert(playersThrow(ArenaPlayer::network(a, -1), ArenaPlayer::network(a)));   // override < 0
    assert(playersThrow(ArenaPlayer::network(a), ArenaPlayer::network(a, 100001)));
    assert(playersThrow(ArenaPlayer::network(a), ArenaPlayer::bot(100001)));
}

static void test_strongest_reference() {
    const std::map<int, double> ratings{{-1, 120.0}, {0, 0.0}, {5, 300.0}, {7, 300.0}};
    assert(strongestReference(ratings, {-1, 0, 5, 7}) == 7);   // tie: larger ID
    assert(strongestReference(ratings, {-1, 0}) == -1);       // a bot can be strongest
    assert(strongestReference(ratings, {0, 9}) == 0);          // 9 is unrated: skipped
    bool threw = false;
    try {
        strongestReference(ratings, {3, 4});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

// Expected scores for g, rated at R = ratings[g-1], against a reference rated r:
// E = 1 / (1 + 10^((r - R) / 400)). The comments give |E - 0.7|.
static void test_select_opponents() {
    using V = std::vector<int>;
    // Generation 1: g-1 is generation 0; the strongest reference is the bot.
    assert((selectOpponents(1, {{-1, 120.0}, {0, 0.0}}, {-1, 0}, 2) == V{0, -1}));

    // R = 800 (generation 9). Strongest: 6. Extras: 4 (E 0.882, 0.18),
    // then bot -1 (E 0.983, 0.28), then 0 (E 0.990, 0.29).
    const std::map<int, double> r{{-1, 100.0}, {0, 0.0}, {4, 450.0}, {6, 700.0}, {9, 800.0}};
    const V refs{-1, 0, 4, 6};
    assert((selectOpponents(10, r, refs, 0) == V{9, 6}));
    assert((selectOpponents(10, r, refs, 1) == V{9, 6, 4}));
    assert((selectOpponents(10, r, refs, 2) == V{9, 6, 4, -1}));
    assert((selectOpponents(10, r, refs, 5) == V{9, 6, 4, -1, 0}));

    // Gating appends the best network, once.
    assert((selectOpponents(10, r, refs, 1, 8) == V{9, 6, 4, 8}));
    assert((selectOpponents(10, r, refs, 1, 4) == V{9, 6, 4}));
    assert((selectOpponents(10, r, refs, 1, 9) == V{9, 6, 4}));

    // The saturated anchor: generation 0 is 900 below R. 4 (E 0.909, 0.21)
    // beats 0 (E 0.994, 0.29) for the one extra place.
    const std::map<int, double> far{{0, 0.0}, {4, 500.0}, {6, 950.0}, {9, 900.0}};
    assert((selectOpponents(10, far, {0, 4, 6}, 1) == V{9, 6, 4}));

    // Ties in |E - 0.7| go to the smaller ID: bots -2 and -1 are both rated 100.
    const std::map<int, double> tie{{-2, 100.0}, {-1, 100.0}, {0, 0.0}, {5, 400.0}, {6, 300.0}};
    assert((selectOpponents(7, tie, {-2, -1, 0, 5}, 1) == V{6, 5, -2}));
    assert((selectOpponents(7, tie, {-2, -1, 0, 5}, 2) == V{6, 5, -2, -1}));

    // Unrated references are never chosen; an unrated g-1 counts as R = 0, so
    // bot -1 at -150 (E 0.703) is the best extra.
    const std::map<int, double> u{{-1, -150.0}, {0, 0.0}, {4, 500.0}};
    assert((selectOpponents(12, u, {-1, 0, 3, 4}, 1) == V{11, 4, -1}));
    assert((selectOpponents(12, u, {-1, 0, 3, 4}, 3) == V{11, 4, -1, 0}));

    // g-1 can itself be the strongest reference: no duplicate.
    assert((selectOpponents(6, {{0, 0.0}, {5, 300.0}}, {0, 5}, 2) == V{5, 0}));

    bool threw = false;
    try {
        selectOpponents(0, r, refs, 2);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
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
    test_gate();
    test_match_score();
    test_deterministic_across_workers_and_runs();
    test_random_symmetry_matches();
    test_one_sided_random_symmetry();
    test_colors_balanced();
    test_opening_temperature_diversifies();
    test_rejects_bad_setup();
    test_bot_matches();
    test_network_wrapper();
    test_simulations_override();
    test_rejects_bad_players();
    test_strongest_reference();
    test_select_opponents();
    std::printf("test_arena: all tests passed\n");
    return 0;
}
