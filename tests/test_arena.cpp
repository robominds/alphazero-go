#include <cassert>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/arena.hpp"
#include "azgo/backend.hpp"
#include "azgo/evaluator.hpp"
#include "azgo/game.hpp"
#include "azgo/metal.hpp"
#include "azgo/parallel.hpp"
#include "azgo/rng.hpp"

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

static ArenaPlayer withSearch(ArenaPlayer p, float cPuct, int mctsBatch) {
    p.cPuct = cPuct;
    p.mctsBatch = mctsBatch;
    return p;
}

// The search each seat runs, player by player. A match's results cannot
// tell which seat got which settings (swapping every setting between the
// seats only relabels the players), so the mapping is checked here, on the
// function the seats are built from.
static void test_arena_search_config() {
    Network a(tinyNet(), 11), b(tinyNet(), 22);
    ArenaConfig cfg = tinyArena();   // 8 simulations, c_puct 1.5, batch 4
    cfg.cPuct = 2.5f;

    // Nothing overridden: everything comes from the config, never noise.
    const MctsConfig plain = arenaSearchConfig(ArenaPlayer::network(a), cfg);
    assert(plain.simulations == 8 && plain.cPuct == 2.5f && plain.batchSize == 4);
    assert(!plain.randomSymmetry && !plain.addNoise);

    // Two players with different overrides, as A and B of one match: each
    // gets its own, and an unset field still comes from the config.
    ArenaPlayer playerA = withSearch(ArenaPlayer::network(a, 64, true), 3.0f, 0);
    ArenaPlayer playerB = withSearch(ArenaPlayer::network(b), 0, 1);
    const MctsConfig sa = arenaSearchConfig(playerA, cfg), sb = arenaSearchConfig(playerB, cfg);
    assert(sa.simulations == 64 && sa.cPuct == 3.0f && sa.batchSize == 4 && sa.randomSymmetry);
    assert(sb.simulations == 8 && sb.cPuct == 2.5f && sb.batchSize == 1 && !sb.randomSymmetry);
    assert(!sa.addNoise && !sb.addNoise);
    playerB.cPuct = 0.5f;
    playerB.mctsBatch = 16;
    assert(arenaSearchConfig(playerB, cfg).cPuct == 0.5f && arenaSearchConfig(playerB, cfg).batchSize == 16);
    assert(arenaSearchConfig(playerA, cfg).cPuct == 3.0f && arenaSearchConfig(playerA, cfg).batchSize == 4);

    // The config's random symmetry reaches every network player.
    ArenaConfig sym = cfg;
    sym.randomSymmetry = true;
    assert(arenaSearchConfig(ArenaPlayer::network(b), sym).randomSymmetry);

    // A bot: its own count, the overrides, and never random symmetry.
    ArenaPlayer bot = withSearch(ArenaPlayer::bot(16), 4.0f, 2);
    bot.randomSymmetry = true;
    const MctsConfig sbot = arenaSearchConfig(bot, sym);
    assert(sbot.simulations == 16 && sbot.cPuct == 4.0f && sbot.batchSize == 2);
    assert(!sbot.randomSymmetry && !sbot.addNoise);
    const MctsConfig plainBot = arenaSearchConfig(ArenaPlayer::bot(16), cfg);
    assert(plainBot.cPuct == 2.5f && plainBot.batchSize == 4);
}

// Per-player c_puct and MCTS batch size (ArenaPlayer::cPuct and ::mctsBatch,
// 0 = ArenaConfig's), so a network can play itself with one search setting
// changed on one side. As with the other one-sided tests, these are fixed
// comparisons of deterministic matches, on a seed (29) whose 8-game
// summaries tell the matches apart.
static void test_search_overrides() {
    Network a(tinyNet(), 11), b(tinyNet(), 22);
    ArenaConfig cfg = tinyArena();   // c_puct 1.5, batch 4
    const ArenaPlayer plainA = ArenaPlayer::network(a), plainB = ArenaPlayer::network(b);
    assert(plainA.cPuct == 0 && plainA.mctsBatch == 0);   // inherit by default
    const MatchResult base = playMatch(plainA, plainB, cfg, 29, 1, "naive");

    // Overrides equal to the config's values are the default match exactly.
    assert(sameResult(base, playMatch(withSearch(plainA, 1.5f, 4), withSearch(plainB, 1.5f, 4),
                                      cfg, 29, 1, "naive")));
    assert(sameResult(base, playMatch(withSearch(plainA, 1.5f, 0), withSearch(plainB, 0, 4),
                                      cfg, 29, 1, "naive")));

    // Both sides overriding is the same as the config changing.
    ArenaConfig wide = cfg;
    wide.cPuct = 4.0f;
    wide.mctsBatch = 1;
    const MatchResult both = playMatch(withSearch(plainA, 4.0f, 1), withSearch(plainB, 4.0f, 1),
                                       cfg, 29, 1, "naive");
    assert(sameResult(both, playMatch(plainA, plainB, wide, 29, 1, "naive")));
    assert(!sameResult(both, base));
    // The override replaces the config for its own side only: A at c_puct 4
    // under the 1.5 config equals the 4 config with B overridden back to 1.5.
    ArenaConfig cfg4 = cfg;
    cfg4.cPuct = 4.0f;
    assert(sameResult(playMatch(withSearch(plainA, 4.0f, 0), plainB, cfg, 29, 1, "naive"),
                      playMatch(plainA, withSearch(plainB, 1.5f, 0), cfg4, 29, 1, "naive")));

    // The network against itself with one setting changed on one side: the
    // tuning match. It differs from the unchanged self-play match, it is
    // deterministic, and it does not depend on the worker count.
    const MatchResult self = playMatch(plainA, plainA, cfg, 29, 1, "naive");
    for (const ArenaPlayer& changed : {withSearch(plainA, 4.0f, 0), withSearch(plainA, 0.5f, 0),
                                       withSearch(plainA, 0, 1)}) {
        const MatchResult oneSided = playMatch(changed, plainA, cfg, 29, 1, "naive");
        checkBookkeeping(oneSided, 8);
        assert(!sameResult(oneSided, self));
        assert(sameResult(oneSided, playMatch(changed, plainA, cfg, 29, 1, "naive")));
        assert(sameResult(oneSided, playMatch(changed, plainA, cfg, 29, 3, "naive")));
        const MatchResult flipped = playMatch(plainA, changed, cfg, 29, 1, "naive");
        assert(!sameResult(flipped, self));
        assert(sameResult(flipped, playMatch(plainA, changed, cfg, 29, 3, "naive")));
    }

    // A bot searches with c_puct and a batch too, so its overrides apply:
    // a bot at c_puct 4 next to a network pinned to 1.5 is the 4 config with
    // the network pinned to 1.5.
    const ArenaPlayer bot = ArenaPlayer::bot(16);
    assert(sameResult(playMatch(withSearch(plainA, 1.5f, 0), withSearch(bot, 4.0f, 0), cfg, 29, 1, "naive"),
                      playMatch(withSearch(plainA, 1.5f, 0), bot, cfg4, 29, 1, "naive")));
    ArenaConfig cfg1 = cfg;
    cfg1.mctsBatch = 1;
    assert(sameResult(playMatch(withSearch(plainA, 0, 4), withSearch(bot, 0, 1), cfg, 29, 1, "naive"),
                      playMatch(withSearch(plainA, 0, 4), bot, cfg1, 29, 1, "naive")));
    assert(sameResult(playMatch(plainA, withSearch(bot, 4.0f, 1), cfg, 29, 1, "naive"),
                      playMatch(plainA, withSearch(bot, 4.0f, 1), cfg, 29, 3, "naive")));
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
    // c_puct and batch overrides: 0 inherits, otherwise the train config's
    // ranges, c_puct in (0, 100] and mcts_batch in [1, 256].
    const ArenaPlayer net = ArenaPlayer::network(a);
    assert(!playersThrow(withSearch(net, 100.0f, 256), withSearch(ArenaPlayer::bot(8), 0.01f, 1)));
    assert(playersThrow(withSearch(net, -1.0f, 0), net));
    assert(playersThrow(net, withSearch(net, 100.5f, 0)));
    assert(playersThrow(withSearch(net, std::nanf(""), 0), net));
    assert(playersThrow(withSearch(net, 0, -1), net));
    assert(playersThrow(net, withSearch(net, 0, 257)));
    assert(playersThrow(net, withSearch(ArenaPlayer::bot(8), 0, 257)));
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
    // The config's own c_puct and batch size: the train config's ranges.
    cfg = tinyArena();
    cfg.cPuct = 0.0f;
    assert(throwsInvalid(a, a, cfg));
    cfg.cPuct = 101.0f;
    assert(throwsInvalid(a, a, cfg));
    cfg = tinyArena();
    cfg.mctsBatch = 0;
    assert(throwsInvalid(a, a, cfg));
    cfg.mctsBatch = 257;
    assert(throwsInvalid(a, a, cfg));
}

// The match as playMatch played it before WorkerEvaluators, written out
// independently: every worker owns, per network player, its own
// single-threaded backend and LocalEvaluator, and per bot a UniformEvaluator.
static MatchResult perWorkerMatch(const ArenaPlayer& a, const ArenaPlayer& b, const ArenaConfig& cfg,
                                  uint64_t seed, int workers, const std::string& backend) {
    const NetConfig shape = (a.net ? a.net : b.net)->config();
    const int n = shape.boardSize;
    struct Side {
        std::unique_ptr<Backend> backend;
        std::unique_ptr<Evaluator> evaluator;
        MctsConfig mcts;
    };
    auto makeSide = [&](const ArenaPlayer& p) {
        Side s;
        s.mcts = arenaSearchConfig(p, cfg);
        if (p.net) {
            s.backend = makeBackend(backend, 1);
            s.evaluator = std::make_unique<LocalEvaluator>(*p.net, *s.backend, s.mcts.batchSize);
        } else {
            s.evaluator = std::make_unique<UniformEvaluator>(n, shape.history, 0.0f, s.mcts.batchSize);
        }
        return s;
    };
    std::vector<Side> sideA, sideB;
    for (int w = 0; w < workers; ++w) {
        sideA.push_back(makeSide(a));
        sideB.push_back(makeSide(b));
    }
    std::vector<double> scores(cfg.games);
    std::vector<std::vector<int>> moves(cfg.games);
    parallelFor(cfg.games, workers, [&](int index, int w) {
        Rng rng(deriveSeed(seed, static_cast<uint64_t>(index)));
        const bool aIsBlack = index % 2 == 0;
        Game game(n, cfg.komi, cfg.maxMoves > 0 ? cfg.maxMoves : defaultMaxMoves(n));
        Mcts searchA(sideA[w].mcts, *sideA[w].evaluator), searchB(sideB[w].mcts, *sideB[w].evaluator);
        while (!game.isTerminal()) {
            Mcts& mover = (game.board().toMove() == Color::Black) == aIsBlack ? searchA : searchB;
            mover.search(game, rng);
            const float temperature = game.moveCount() < cfg.temperatureMoves ? 1.0f : 0.0f;
            const int action = mover.selectAction(temperature, rng);
            game.play(action);
            if (searchA.nodeCount() > 0) searchA.advance(action);
            if (searchB.nodeCount() > 0) searchB.advance(action);
        }
        scores[index] = matchScore(game.resultForBlack(), aIsBlack);
        moves[index] = game.moves();
    });
    MatchResult r;
    r.games = cfg.games;
    for (double s : scores) {
        r.scoreA += s;
        if (s == 1.0) ++r.winsA;
        else if (s == 0.0) ++r.winsB;
        else ++r.draws;
    }
    r.distinctGames = static_cast<int>(std::set<std::vector<int>>(moves.begin(), moves.end()).size());
    return r;
}

// playMatch now builds its evaluators through WorkerEvaluators. With the CPU
// backends that must be exactly the old per-worker evaluators: the same
// results for network against network (random symmetry, per-side batches)
// and against a bot. With metal in a build without Metal, it refuses.
static void test_worker_evaluators_match_per_worker_loop() {
    Network a(tinyNet(), 11), b(tinyNet(), 22);
    ArenaConfig cfg = tinyArena();
    cfg.temperatureMoves = 2;
    std::vector<std::string> backends = {"naive"};
    if (blasAvailable()) backends.push_back("blas");
    for (const std::string& be : backends) {
        for (int workers : {1, 3}) {
            ArenaPlayer pa = ArenaPlayer::network(a, 0, true), pb = ArenaPlayer::network(b);
            pb.mctsBatch = 2;
            assert(sameResult(playMatch(pa, pb, cfg, 7, workers, be),
                              perWorkerMatch(pa, pb, cfg, 7, workers, be)));
            const ArenaPlayer bot = ArenaPlayer::bot(12);
            assert(sameResult(playMatch(pa, bot, cfg, 8, workers, be),
                              perWorkerMatch(pa, bot, cfg, 8, workers, be)));
        }
    }
    if (!metalBuilt()) {
        bool threw = false;
        try {
            playMatch(a, b, cfg, 1, 1, "metal");
        } catch (const std::invalid_argument& e) {
            threw = std::string(e.what()).find("-DAZ_USE_METAL=ON") != std::string::npos;
        }
        assert(threw);
    }
    // inference_batch must hold each network side's batch.
    bool threw = false;
    try {
        ArenaPlayer pa = ArenaPlayer::network(a);
        pa.mctsBatch = 16;
        playMatch(pa, ArenaPlayer::network(b), cfg, 1, 1, "naive", 8);
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()).find("inference batch") != std::string::npos;
    }
    assert(threw);
}

int main() {
    test_gate();
    test_worker_evaluators_match_per_worker_loop();
    test_match_score();
    test_deterministic_across_workers_and_runs();
    test_random_symmetry_matches();
    test_one_sided_random_symmetry();
    test_arena_search_config();
    test_search_overrides();
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
