#include <cassert>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#include "azgo/game.hpp"
#include "azgo/parallel.hpp"
#include "azgo/selfplay.hpp"

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
}

static bool sameRecord(const GameRecord& a, const GameRecord& b) {
    return a.boardSize == b.boardSize && a.komi == b.komi && a.moves == b.moves &&
           a.stones == b.stones && a.visits == b.visits && a.resultForBlack == b.resultForBlack;
}

static std::vector<GameRecord> playGames(const Network& net, const Backend& be, int workers) {
    SelfPlayConfig cfg;
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

static void test_self_play_is_independent_of_worker_count() {
    NetConfig c;
    c.boardSize = 5;
    c.history = 2;
    c.blocks = 1;
    c.filters = 4;
    c.valueHidden = 8;
    Network net(c, 7);
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

int main() {
    test_immediate_pass_pass_game();
    test_z_for_a_black_win_and_a_tie();
    test_record_is_consistent_with_a_replay();
    test_self_play_is_independent_of_worker_count();
    std::printf("test_selfplay: all tests passed\n");
    return 0;
}
