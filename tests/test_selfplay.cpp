#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/encode.hpp"
#include "azgo/game.hpp"
#include "azgo/parallel.hpp"
#include "azgo/run_state.hpp"
#include "azgo/selfplay.hpp"
#include "azgo/selfplay_slice.hpp"

using namespace azgo;

// An "untrained network" that is certain it should pass.
class AlwaysPassEvaluator : public Evaluator {
public:
    int boardSize() const override { return 5; }
    int history() const override { return 2; }
    int maxBatch() const override { return 8; }
    void evaluate(const float*, int B, float* logits, float* values) override {
        for (int b = 0; b < B; ++b) {
            for (int a = 0; a < 25; ++a) logits[b * 26 + a] = -20.0f;
            logits[b * 26 + 25] = 20.0f;   // pass
            values[b] = 0.0f;
        }
    }
};

static void test_immediate_pass_pass_game() {
    // Review Focus 1. Why both sides pass even with root noise: with one
    // simulation the only descent from the root compares Q + U with Q = 0
    // for every edge and U proportional to the prior (sqrt(max(1, 0)) = 1).
    // Pass has raw prior ~1 (e^40 odds), so after mixing in 25% noise it
    // keeps >= 0.75 while every other move gets <= 0.25 * eta <= 0.25. The
    // single visit goes to pass, and temperature-1 sampling over the visit
    // counts {pass: 1} can only return pass. The same holds at ply 1.
    AlwaysPassEvaluator ev;
    SelfPlayConfig cfg;
    cfg.mcts.simulations = 1;
    cfg.mcts.batchSize = 1;
    cfg.komi = 7.5f;
    Rng rng(1);
    GameRecord rec = playSelfPlayGame(ev, cfg, rng);

    assert(rec.boardSize == 5 && rec.komi == 7.5f);
    assert(rec.positions() == 2);
    assert(rec.moves == std::vector<int>({25, 25}));
    assert(rec.stones.size() == 2u * 25);
    for (uint8_t s : rec.stones) assert(s == static_cast<uint8_t>(Color::Empty));
    for (int ply = 0; ply < 2; ++ply) {
        assert(rec.visits[ply].size() == 1);
        assert(rec.visits[ply][0].first == 25 && rec.visits[ply][0].second == 1);
    }
    // Empty board: area score 0, so komi decides it for White.
    assert(rec.resultForBlack == -1.0f);
    assert(rec.toMoveAt(0) == Color::Black && rec.toMoveAt(1) == Color::White);
    assert(rec.zAt(0) == -1.0f);   // Black's position: Black lost
    assert(rec.zAt(1) == +1.0f);   // White's position: White won
    // The final board is empty: nobody owns anything.
    assert(rec.finalOwnership == std::vector<int8_t>(25, 0));
}

static void test_z_for_a_black_win_and_a_tie() {
    GameRecord rec;
    rec.boardSize = 5;
    rec.moves = {12, 25, 25};
    rec.resultForBlack = 1.0f;
    assert(rec.zAt(0) == 1.0f && rec.zAt(1) == -1.0f && rec.zAt(2) == 1.0f);
    rec.resultForBlack = 0.0f;
    for (int ply = 0; ply < 3; ++ply) assert(rec.zAt(ply) == 0.0f);
}

static void test_record_is_consistent_with_a_replay() {
    UniformEvaluator ev(5, 2);
    SelfPlayConfig cfg;
    cfg.mcts.simulations = 16;
    cfg.mcts.batchSize = 4;
    cfg.temperatureMoves = 6;
    cfg.komi = 7.5f;
    Rng rng(2024);
    GameRecord rec = playSelfPlayGame(ev, cfg, rng);

    const int n = 5, area = 25;
    assert(rec.positions() >= 2);
    assert(rec.stones.size() == static_cast<size_t>(rec.positions()) * area);
    assert(rec.visits.size() == static_cast<size_t>(rec.positions()));
    for (int p = 0; p < area; ++p) assert(rec.stonesAt(0)[p] == static_cast<uint8_t>(Color::Empty));

    // Replay the same game with the same seed and a second search tree: every
    // recorded field must match what the search produced at that ply.
    Game game(n, cfg.komi, defaultMaxMoves(n));
    MctsConfig m = cfg.mcts;
    m.addNoise = true;
    UniformEvaluator ev2(5, 2);
    Mcts mcts(m, ev2);
    Rng replayRng(2024);
    std::vector<uint8_t> stones(area);
    bool reusedSomething = false;
    for (int ply = 0; ply < rec.positions(); ++ply) {
        assert(!game.isTerminal());
        assert(game.board().toMove() == rec.toMoveAt(ply));   // alternates, Black first
        game.board().exportStones(stones.data());
        assert(std::memcmp(stones.data(), rec.stonesAt(ply), area) == 0);

        mcts.search(game, replayRng);
        std::vector<int> visits = mcts.rootVisits();
        int recorded = 0;
        int prevAction = -1;
        for (const auto& av : rec.visits[ply]) {
            assert(av.first > prevAction);   // ascending action order
            prevAction = av.first;
            assert(av.second > 0 && visits[av.first] == av.second);
            recorded += av.second;
        }
        // Sum = this search's 16 simulations + whatever the reused subtree held.
        assert(recorded == mcts.rootTotalVisits());
        assert(recorded >= 16);
        if (recorded > 16) reusedSomething = true;

        const float T = ply < cfg.temperatureMoves ? 1.0f : 0.0f;
        const int a = mcts.selectAction(T, replayRng);
        assert(a == rec.moves[ply]);
        game.play(a);
        mcts.advance(a);
    }
    assert(game.isTerminal());
    assert(reusedSomething);   // tree reuse carried visits into later plies
    assert(rec.resultForBlack == game.resultForBlack());
    for (int ply = 0; ply < rec.positions(); ++ply)
        assert(rec.zAt(ply) == (ply % 2 == 0 ? rec.resultForBlack : -rec.resultForBlack));

    // finalOwnership is the final board's, and replaying the moves finds it too.
    std::vector<int8_t> own(area);
    game.board().ownership(own.data());
    assert(rec.finalOwnership == own);
    assert(replayFinalOwnership(rec) == own);
    int sum = 0;
    for (int8_t o : own) sum += o;
    assert(sum == game.board().areaScore());
}

// A record that does not replay is an error that says why. The record is
// built by hand: Black C3, White D4, Black C4, White D3 on 5x5.
static void test_replay_rejects_a_bad_record() {
    const int n = 5, area = 25;
    Game game(n, 7.5f, defaultMaxMoves(n));
    GameRecord good;
    good.boardSize = n;
    good.komi = 7.5f;
    for (int move : {12, 8, 7, 13}) {
        const size_t offset = good.stones.size();
        good.stones.resize(offset + area);
        game.board().exportStones(good.stones.data() + offset);
        good.visits.push_back({{static_cast<int16_t>(move), 1}});
        good.moves.push_back(move);
        game.play(move);
    }
    auto replayError = [](const GameRecord& r) {
        try {
            replayFinalOwnership(r);
        } catch (const std::runtime_error& e) {
            return std::string(e.what());
        }
        return std::string();
    };
    assert(replayError(good).empty());
    std::vector<int8_t> own(area);
    game.board().ownership(own.data());
    assert(replayFinalOwnership(good) == own);

    // Move 2 on the point Black took with move 1: occupied, so illegal.
    GameRecord occupied = good;
    occupied.moves[1] = 12;
    assert(replayError(occupied) == "move 2 (C3) is illegal");

    // A stored position that the moves do not produce: a stone on A5 at ply 2.
    GameRecord edited = good;
    edited.stones[2 * area + 0] = static_cast<uint8_t>(Color::Black);
    assert(replayError(edited) == "position 2 differs from the replayed one");
}

static bool sameRecord(const GameRecord& a, const GameRecord& b) {
    return a.boardSize == b.boardSize && a.komi == b.komi && a.moves == b.moves &&
           a.stones == b.stones && a.visits == b.visits && a.resultForBlack == b.resultForBlack &&
           a.finalOwnership == b.finalOwnership && a.resigned == b.resigned;
}

static std::vector<GameRecord> playGames(const Network& net, const Backend& be, int workers,
                                         bool randomSymmetry = false) {
    SelfPlayConfig cfg;
    cfg.mcts.randomSymmetry = randomSymmetry;
    cfg.mcts.simulations = 8;
    cfg.mcts.batchSize = 4;
    cfg.temperatureMoves = 4;
    std::vector<std::unique_ptr<LocalEvaluator>> evaluators;
    for (int w = 0; w < workers; ++w)
        evaluators.push_back(std::make_unique<LocalEvaluator>(net, be, cfg.mcts.batchSize));
    std::vector<GameRecord> records(6);
    parallelFor(6, workers, [&](int index, int worker) {
        Rng rng(deriveSeed(42, static_cast<uint64_t>(index)));
        records[index] = playSelfPlayGame(*evaluators[worker], cfg, rng);
    });
    return records;
}

static NetConfig tinyNet() {
    NetConfig c;
    c.boardSize = 5;
    c.history = 2;
    c.blocks = 1;
    c.filters = 4;
    c.valueHidden = 8;
    return c;
}

static void test_self_play_is_independent_of_worker_count() {
    Network net(tinyNet(), 7);
    NaiveBackend be;
    std::vector<GameRecord> one = playGames(net, be, 1);
    std::vector<GameRecord> three = playGames(net, be, 3);
    bool anyDifferent = false;
    for (int i = 0; i < 6; ++i) {
        assert(sameRecord(one[i], three[i]));
        assert(one[i].positions() >= 2);
        if (i > 0 && one[i].moves != one[0].moves) anyDifferent = true;
    }
    assert(anyDifferent);   // different seeds give different games
}

// Random symmetry draws from each game's own Rng, so it keeps the games
// reproducible and independent of the worker count. An untrained network is
// not equivariant, so the draws change what it says and the games change.
static void test_random_symmetry_self_play_is_independent_of_worker_count() {
    Network net(tinyNet(), 7);
    NaiveBackend be;
    std::vector<GameRecord> one = playGames(net, be, 1, true);
    std::vector<GameRecord> three = playGames(net, be, 3, true);
    std::vector<GameRecord> plain = playGames(net, be, 1, false);
    bool anyChanged = false;
    for (int i = 0; i < 6; ++i) {
        assert(sameRecord(one[i], three[i]));
        assert(one[i].positions() >= 2);
        if (!sameRecord(one[i], plain[i])) anyChanged = true;
    }
    assert(anyChanged);
}

// ---- resignation -------------------------------------------------------------

// A network sure that `hopeless` loses wherever it is to move (value -0.95
// for that side, +0.95 for the other), with equal priors on every point and
// none on pass, so no game ends early by passing.
class HopelessSideEvaluator : public Evaluator {
public:
    explicit HopelessSideEvaluator(Color hopeless) : hopeless_(hopeless) {}
    int boardSize() const override { return 5; }
    int history() const override { return 2; }
    int maxBatch() const override { return 8; }
    void evaluate(const float* input, int B, float* logits, float* values) override {
        const int area = 25;
        const size_t stride = static_cast<size_t>(numPlanes(2)) * area;
        for (int b = 0; b < B; ++b) {
            // Plane 2H is all ones when Black is to move (any symmetry keeps it so).
            const bool blackToMove = input[b * stride + 4 * area] == 1.0f;
            const Color toMove = blackToMove ? Color::Black : Color::White;
            for (int a = 0; a < area; ++a) logits[b * 26 + a] = 0.0f;
            logits[b * 26 + 25] = -20.0f;
            values[b] = toMove == hopeless_ ? -0.95f : 0.95f;
        }
    }
private:
    Color hopeless_;
};

static SelfPlayConfig resignConfig(float threshold, float disabledFraction) {
    SelfPlayConfig cfg;
    cfg.mcts.simulations = 8;
    cfg.mcts.batchSize = 4;
    cfg.temperatureMoves = 4;
    cfg.komi = 7.5f;
    cfg.maxMoves = 12;
    cfg.resignation = true;
    cfg.resignThreshold = threshold;
    cfg.resignDisabledFraction = disabledFraction;
    return cfg;
}

// White's every search says -0.95, below v_resign = -0.9: White resigns at
// its first move, Black wins, and z is +1 for Black's one position. The
// position White resigned in is not recorded (no move was played from it).
static void test_hopeless_side_resigns() {
    HopelessSideEvaluator ev(Color::White);
    Rng rng(3);
    const GameRecord rec = playSelfPlayGame(ev, resignConfig(-0.9f, 0.0f), rng);
    assert(rec.resigned && !rec.resignDisabled);
    assert(rec.positions() == 1 && rec.moves.size() == 1 && rec.visits.size() == 1);
    assert(rec.stones.size() == 25u && rec.finalOwnership.size() == 25u);
    assert(rec.resultForBlack == 1.0f);
    assert(rec.zAt(0) == 1.0f);
    assert(rec.resignLows[1].size() == 1 && rec.resignLows[1][0].first == 1);
    assert(rec.resignLows[1][0].second < -0.9f);
    assert(rec.resignLows[0].empty());   // Black's ply 0 is never considered
}

// Black is never offered resignation at the empty board (every record keeps
// at least one position), so a hopeless Black resigns at ply 2: White wins.
static void test_black_resigns_after_the_first_move() {
    HopelessSideEvaluator ev(Color::Black);
    Rng rng(5);
    const GameRecord rec = playSelfPlayGame(ev, resignConfig(-0.9f, 0.0f), rng);
    assert(rec.resigned);
    assert(rec.positions() == 2);
    assert(rec.resultForBlack == -1.0f);
    assert(rec.zAt(0) == -1.0f && rec.zAt(1) == 1.0f);
    assert(rec.resignLows[0].size() == 1 && rec.resignLows[0][0].first == 2);
    assert(rec.resignLows[1].size() == 1 && rec.resignLows[1][0].first == 1);
    assert(rec.resignLows[1][0].second > 0.9f);
}

// The resignation test is strict: a value exactly at v_resign plays on.
// HopelessSideEvaluator makes White's value exactly -0.95f (every child Q is
// -0.95f, and so is their mean).
static void test_resignation_needs_a_value_strictly_below_the_threshold() {
    HopelessSideEvaluator ev(Color::White);
    Rng a(3), b(3);
    const GameRecord at = playSelfPlayGame(ev, resignConfig(-0.95f, 0.0f), a);
    assert(at.resignLows[1].front().second == -0.95f);
    assert(at.positions() > 1);   // not at ply 1
    const GameRecord above = playSelfPlayGame(ev, resignConfig(std::nextafter(-0.95f, 0.0f), 0.0f), b);
    assert(above.resigned && above.positions() == 1);
}

// AlphaGo Zero resigns only when the root value AND the best child's value
// are below v_resign: the larger of the two decides.
static void test_resign_value_needs_both_below() {
    assert(resignValue(-0.95f, 0.5f) == 0.5f);    // root below, best child above: plays on
    assert(resignValue(0.5f, -0.95f) == 0.5f);    // and the reverse
    assert(resignValue(-0.95f, -0.92f) == -0.92f);
    assert(!(resignValue(-0.95f, 0.5f) < -0.9f) && !(resignValue(0.5f, -0.95f) < -0.9f));
    assert(resignValue(-0.95f, -0.92f) < -0.9f);

    // On a real search the two differ, and the larger one is used. Black
    // passed on an empty board, so White's pass ends the game and wins by
    // komi: the pass child's Q is +1, the other children's 0. It takes a
    // long search before the exploration bonus sends visits elsewhere.
    Game game(5, 7.5f, defaultMaxMoves(5));
    game.play(25);
    UniformEvaluator ev(5, 2);
    MctsConfig m;
    m.simulations = 1000;
    m.batchSize = 1;
    Mcts mcts(m, ev);
    Rng rng(16);
    mcts.search(game, rng);
    assert(mcts.bestChildValue() == 1.0f && mcts.rootValue() < 1.0f);
    assert(resignValue(mcts) == 1.0f);
}

// A resignation-disabled game is played to the end. Each side keeps its
// record lows, (ply, value) each time its resignation value reached a new
// minimum, so calibration can replay "who would have resigned first?".
static void test_disabled_game_plays_on_and_records_lows() {
    HopelessSideEvaluator ev(Color::White);
    Rng rng(3);
    const GameRecord rec = playSelfPlayGame(ev, resignConfig(-0.9f, 1.0f), rng);
    assert(!rec.resigned && rec.resignDisabled);
    assert(rec.positions() == 12);   // the move cap: nobody passes
    for (int side = 0; side < 2; ++side) {
        const ResignLows& lows = rec.resignLows[side];
        assert(!lows.empty());
        for (size_t i = 0; i < lows.size(); ++i) {
            assert(lows[i].first >= 1 && lows[i].first % 2 == side);   // its own plies, never ply 0
            if (i > 0) assert(lows[i].first > lows[i - 1].first && lows[i].second < lows[i - 1].second);
        }
    }
    // White saw -0.95 from its first move on.
    assert(rec.resignLows[1][0].first == 1 && rec.resignLows[1][0].second == -0.95f);
    Game replay(5, 7.5f, 12);
    for (int a : rec.moves) replay.play(a);
    assert(rec.resultForBlack == replay.resultForBlack());
    const ResignTrace t = resignTrace(rec);
    const int winner = rec.resultForBlack > 0 ? 0 : 1;
    assert(t.winnerLows == rec.resignLows[winner] && t.loserLows == rec.resignLows[1 - winner]);
}

// With resignation on, a game draws exactly once more from its Rng (whether
// resignation is disabled for it), first; with it off, nothing extra. A
// game that never resigns is otherwise the game resignation-off plays.
static void test_resignation_draws_once_per_game() {
    SelfPlayConfig off;
    off.mcts.simulations = 16;
    off.mcts.batchSize = 4;
    off.temperatureMoves = 6;
    for (float fraction : {0.0f, 1.0f}) {
        // v < -1 never holds: fraction 0 never resigns either.
        SelfPlayConfig on = off;
        on.resignation = true;
        on.resignThreshold = -1.0f;
        on.resignDisabledFraction = fraction;
        UniformEvaluator ev(5, 2);
        Rng a(77), b(77);
        b.uniform();
        const GameRecord withResign = playSelfPlayGame(ev, on, a);
        const GameRecord shifted = playSelfPlayGame(ev, off, b);
        assert(sameRecord(withResign, shifted));
        assert(withResign.resignDisabled == (fraction == 1.0f) && !withResign.resigned);
        assert(!shifted.resignDisabled && !shifted.resigned);
        assert(a.uniform() == b.uniform());   // the streams end in step
    }
}

static ResignTrace trace(ResignLows winner, ResignLows loser = {}) {
    ResignTrace t;
    t.winnerLows = std::move(winner);
    t.loserLows = std::move(loser);
    return t;
}

// A disabled game is a false positive at threshold v exactly when the
// winner's resignation value first drops below v before the loser's does:
// resigning at v would then have ended it as the winner's loss. The search
// draws nothing for the test, so the game is the same up to that ply.
static void test_false_positive_is_who_would_resign_first() {
    // The loser always crosses first (its lows lie below the winner's, earlier).
    const ResignTrace loserFirst = trace({{3, -0.9f}}, {{2, -0.95f}});
    for (float v : {-0.96f, -0.94f, -0.9f, -0.89f, -0.5f, 0.0f}) assert(!loserFirst.falsePositiveAt(v));
    // The winner dips to -0.3 at ply 1 and -0.8 at ply 5; the loser to -0.6 at ply 4.
    const ResignTrace t = trace({{1, -0.3f}, {5, -0.8f}}, {{4, -0.6f}});
    assert(t.falsePositiveAt(-0.2f));    // winner at ply 1, loser at ply 4
    assert(!t.falsePositiveAt(-0.5f));   // the loser at ply 4 comes before the winner's ply 5
    assert(t.falsePositiveAt(-0.7f));    // only the winner gets that low
    assert(!t.falsePositiveAt(-0.8f));   // strictly below: -0.8 is not
    assert(!t.falsePositiveAt(-0.85f));
    // A draw keeps no lows: never one.
    assert(!trace({}, {}).falsePositiveAt(0.0f));

    // resignTrace takes the winner's and the loser's lows from a record.
    GameRecord r;
    r.resignLows[0] = {{2, -0.4f}};
    r.resignLows[1] = {{1, -0.7f}, {3, -0.75f}};
    r.resultForBlack = 1.0f;
    assert(resignTrace(r).winnerLows == r.resignLows[0] && resignTrace(r).loserLows == r.resignLows[1]);
    r.resultForBlack = -1.0f;
    assert(resignTrace(r).winnerLows == r.resignLows[1] && resignTrace(r).loserLows == r.resignLows[0]);
    r.resultForBlack = 0.0f;
    assert(resignTrace(r).winnerLows.empty() && resignTrace(r).loserLows.empty());
}

// v_resign is the largest threshold on the 0.01 grid from -1 to 0 whose
// false-positive rate over the window stays within the limit; with fewer
// games than minGames, the initial one.
static void test_calibrate_resign_threshold() {
    ResignCalibration c;
    c.window = 100;
    c.minGames = 20;
    c.maxFalsePositive = 0.05f;
    c.initialThreshold = -0.9f;
    // 20 games: three winners dipped to -0.9, -0.7 and -0.3 (the oldest
    // three) while their losers never did; the other winners never dipped.
    std::vector<ResignTrace> games = {trace({{1, -0.9f}}), trace({{1, -0.7f}}), trace({{1, -0.3f}})};
    games.insert(games.end(), 17, trace({}));
    assert(games.size() == 20);
    // At -0.70 only the -0.9 game counts (1/20 = 5%); at -0.69 the -0.7 game joins (10%).
    assert(calibrateResignThreshold(games, c) == -0.7f);
    assert(resignFalsePositiveRate(games, c.window, -0.7f) == 0.05);
    assert(resignFalsePositiveRate(games, c.window, -0.69f) == 0.10);
    c.maxFalsePositive = 0.10f;
    assert(calibrateResignThreshold(games, c) == -0.3f);
    c.maxFalsePositive = 0.0f;
    assert(calibrateResignThreshold(games, c) == -0.9f);
    c.maxFalsePositive = 0.5f;
    assert(calibrateResignThreshold(games, c) == 0.0f);   // the top of the grid

    // A game whose loser would have resigned first is no false positive:
    // with it in place of the -0.3 game, 0 is within 10%.
    std::vector<ResignTrace> saved = games;
    saved[2] = trace({{3, -0.3f}}, {{2, -0.5f}});
    c.maxFalsePositive = 0.10f;
    assert(calibrateResignThreshold(saved, c) == 0.0f);

    // Fewer than minGames: the initial threshold, whatever the data says.
    c.maxFalsePositive = 0.05f;
    const std::vector<ResignTrace> few(games.begin(), games.begin() + 19);
    assert(calibrateResignThreshold(few, c) == -0.9f);
    assert(calibrateResignThreshold({}, c) == -0.9f);
    assert(resignFalsePositiveRate({}, c.window, -0.5f) == 0.0);

    // Only the last `window` games count: the three low ones fall out.
    c.window = 17;
    c.minGames = 17;
    assert(calibrateResignThreshold(games, c) == 0.0f);
    assert(resignFalsePositiveRate(games, 17, 0.0f) == 0.0);
    // Every winner sank to -1: only the bottom of the grid, -1, counts none
    // of them (a value is counted when it lies strictly below the threshold).
    c.window = 20;
    c.minGames = 1;
    assert(calibrateResignThreshold(std::vector<ResignTrace>(5, trace({{1, -1.0f}})), c) == -1.0f);
}


// ---- self-play as a slice of game indices --------------------------------------

static bool sameResignFields(const GameRecord& a, const GameRecord& b) {
    return a.resignDisabled == b.resignDisabled && a.resignLows[0] == b.resignLows[0] &&
           a.resignLows[1] == b.resignLows[1];
}

static SelfPlayConfig sliceConfig(bool resignation) {
    SelfPlayConfig sp;
    sp.mcts.simulations = 8;
    sp.mcts.batchSize = 4;
    sp.temperatureMoves = 4;
    sp.resignation = resignation;
    sp.resignThreshold = -0.05f;
    sp.resignDisabledFraction = 0.3f;
    return sp;
}

// Game i of generation g is seeded by deriveSeed(seed, g, kPhaseSelfPlay, i)
// whichever slice plays it, so a slice [a, b) is exactly games a..b-1 of the
// whole generation, for any worker counts.
static void test_slice_equals_part_of_whole() {
    const Network net(tinyNet(), 9);
    std::vector<std::string> backends = {"naive"};
    if (blasAvailable()) backends.push_back("blas");
    for (const std::string& backend : backends)
        for (bool resignation : {false, true}) {
            const SelfPlayConfig sp = sliceConfig(resignation);
            const std::vector<GameRecord> whole =
                playSelfPlayGames(net, sp, 77, 3, 0, 7, EngineRun{backend, 3});
            const std::vector<GameRecord> middle =
                playSelfPlayGames(net, sp, 77, 3, 2, 4, EngineRun{backend, 1});
            const std::vector<GameRecord> last =
                playSelfPlayGames(net, sp, 77, 3, 6, 1, EngineRun{backend, 2});
            assert(whole.size() == 7 && middle.size() == 4 && last.size() == 1);
            for (int k = 0; k < 4; ++k) {
                assert(sameRecord(middle[k], whole[2 + k]));
                assert(sameResignFields(middle[k], whole[2 + k]));
            }
            assert(sameRecord(last[0], whole[6]) && sameResignFields(last[0], whole[6]));
            bool lows = false, differ = false;
            for (int i = 0; i < 7; ++i) {
                lows = lows || !whole[i].resignLows[0].empty();
                differ = differ || whole[i].moves != whole[0].moves;
            }
            assert(lows == resignation && differ);
            // Another generation is another set of games.
            const std::vector<GameRecord> other =
                playSelfPlayGames(net, sp, 77, 4, 6, 1, EngineRun{backend, 1});
            assert(!sameRecord(other[0], whole[6]));
        }
}

// The statistics: every position of every game, the evaluator's count of
// evaluations (the same for any worker count: each search is deterministic),
// and the wall time.
static void test_slice_stats() {
    const Network net(tinyNet(), 9);
    const SelfPlayConfig sp = sliceConfig(false);
    SliceStats one, three;
    const std::vector<GameRecord> records =
        playSelfPlayGames(net, sp, 77, 3, 1, 5, EngineRun{"naive", 1}, &one);
    playSelfPlayGames(net, sp, 77, 3, 1, 5, EngineRun{"naive", 3}, &three);
    long long plies = 0;
    for (const GameRecord& r : records) plies += r.positions();
    assert(one.positions == plies && three.positions == plies);
    assert(one.seconds >= 0 && three.seconds >= 0);
    // Played by hand, one evaluator per game, counting what each evaluated.
    NaiveBackend be;
    long long evaluations = 0;
    for (int i = 1; i < 6; ++i) {
        LocalEvaluator ev(net, be, sp.mcts.batchSize);
        Rng rng(deriveSeed(77, 3, kPhaseSelfPlay, static_cast<uint64_t>(i)));
        playSelfPlayGame(ev, sp, rng);
        evaluations += ev.evaluations();
    }
    assert(evaluations > 0 && one.evaluations == evaluations && three.evaluations == evaluations);
}

// The QoS class only changes how macOS schedules the worker threads.
static void test_qos_changes_nothing() {
    const Network net(tinyNet(), 9);
    const SelfPlayConfig sp = sliceConfig(true);
    EngineRun utility{"naive", 2};
    utility.qos = CpuQos::Utility;
    const std::vector<GameRecord> a = playSelfPlayGames(net, sp, 5, 1, 0, 4, EngineRun{"naive", 2});
    const std::vector<GameRecord> b = playSelfPlayGames(net, sp, 5, 1, 0, 4, utility);
    for (int i = 0; i < 4; ++i) assert(sameRecord(a[i], b[i]) && sameResignFields(a[i], b[i]));
}

int main() {
    test_slice_equals_part_of_whole();
    test_slice_stats();
    test_qos_changes_nothing();
    test_immediate_pass_pass_game();
    test_z_for_a_black_win_and_a_tie();
    test_record_is_consistent_with_a_replay();
    test_self_play_is_independent_of_worker_count();
    test_random_symmetry_self_play_is_independent_of_worker_count();
    test_replay_rejects_a_bad_record();
    test_hopeless_side_resigns();
    test_black_resigns_after_the_first_move();
    test_resignation_needs_a_value_strictly_below_the_threshold();
    test_resign_value_needs_both_below();
    test_disabled_game_plays_on_and_records_lows();
    test_resignation_draws_once_per_game();
    test_false_positive_is_who_would_resign_first();
    test_calibrate_resign_threshold();
    std::printf("test_selfplay: all tests passed\n");
    return 0;
}
