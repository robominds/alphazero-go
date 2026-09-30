#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>
#include "azgo/mcts.hpp"

using namespace azgo;

static bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

static MctsConfig serialConfig(int simulations) {
    MctsConfig c;
    c.simulations = simulations;
    c.batchSize = 1;
    return c;
}

static int sum(const std::vector<int>& v) { return std::accumulate(v.begin(), v.end(), 0); }

// Index of the root edge for `action`, or -1.
static int rootEdge(const Mcts& m, int action) {
    const MctsNode& root = m.node(0);
    for (int i = 0; i < root.numEdges; ++i)
        if (m.edge(root.firstEdge + i).action == action) return root.firstEdge + i;
    return -1;
}

// Black to move; Black's only winning move is D1 (row 4, col 3 = point 23),
// which captures the three white stones and leaves Black owning the whole
// board. With komi 20.5: capture -> 25 - 20.5 > 0; any other move or pass
// leaves Black 5 stones + 16 territory vs White 3 -> 18 - 20.5 < 0.
static Game winningCaptureGame(int maxMoves) {
    Board start = boardFromRows({".....",
                                 ".....",
                                 ".....",
                                 "XXXX.",
                                 "OOO.X"}, Color::Black);
    return Game(start, 20.5f, maxMoves);
}

static void test_root_visits_sum_to_simulations_and_tree_is_kept() {
    Game game(5, 7.5f, defaultMaxMoves(5));
    UniformEvaluator ev(5, 2);
    Mcts mcts(serialConfig(40), ev);
    Rng rng(1);
    mcts.search(game, rng);
    assert(sum(mcts.rootVisits()) == 40);
    assert(mcts.rootTotalVisits() == 40);
    assert(mcts.node(0).visits == 40);   // expanding the root is not a simulation
    const size_t nodesAfterFirst = mcts.nodeCount();

    mcts.search(game, rng);   // same position: the tree is kept and grows
    assert(sum(mcts.rootVisits()) == 80);
    assert(mcts.nodeCount() > nodesAfterFirst);
}

static void test_only_legal_actions_get_edges() {
    // White to move. A5 (point 0) is empty but surrounded by Black: suicide.
    Board start = boardFromRows({".X...",
                                 "X....",
                                 ".....",
                                 ".....",
                                 "....."}, Color::White);
    Game game(start, 7.5f, 50);
    UniformEvaluator ev(5, 2);
    Mcts mcts(serialConfig(10), ev);
    Rng rng(2);
    mcts.search(game, rng);

    const MctsNode& root = mcts.node(0);
    assert(root.numEdges == 25 - 2 - 1 + 1);   // minus stones, minus suicide, plus pass
    std::vector<int> visits = mcts.rootVisits();
    std::vector<float> priors = mcts.rootPriors();
    assert(visits.size() == 26 && priors.size() == 26);
    for (int a : {0, 1, 5}) {
        assert(rootEdge(mcts, a) == -1);
        assert(visits[a] == 0 && priors[a] == 0.0f);
    }
    float priorSum = 0.0f;
    for (int i = 0; i < root.numEdges; ++i) {
        const MctsEdge& e = mcts.edge(root.firstEdge + i);
        assert(game.isLegal(e.action));
        assert(near(e.rawPrior, 1.0 / 23));   // uniform over the legal actions only
        if (i > 0) assert(e.action > mcts.edge(root.firstEdge + i - 1).action);
        priorSum += e.prior;
    }
    assert(near(priorSum, 1.0));
}

static void test_superko_excludes_immediate_ko_retake_inside_tree() {
    // Black captures the ko at C4 (point 7), taking the white stone at B4
    // (point 6). White retaking at B4 would recreate the root position, so
    // the child node must have no edge for it -- even though Board alone
    // would allow it.
    Board start = boardFromRows({".XO..",
                                 "XO.O.",
                                 ".XO..",
                                 ".....",
                                 "....."}, Color::Black);
    Game game(start, 7.5f, 50);
    assert(game.isLegal(7));
    UniformEvaluator ev(5, 2);
    Mcts mcts(serialConfig(60), ev);   // > root edge count, so every root child gets expanded
    Rng rng(3);
    mcts.search(game, rng);

    const int e = rootEdge(mcts, 7);
    assert(e >= 0 && mcts.edge(e).child >= 0);
    const MctsNode& child = mcts.node(mcts.edge(e).child);
    assert(child.expanded);
    assert(child.board.at(6) == Color::Empty && child.board.toMove() == Color::White);
    assert(child.board.isLegal(6));   // locally legal ...
    for (int i = 0; i < child.numEdges; ++i)
        assert(mcts.edge(child.firstEdge + i).action != 6);   // ... but superko forbids it
    // A normal move for White is still there.
    bool hasE1 = false;
    for (int i = 0; i < child.numEdges; ++i)
        if (mcts.edge(child.firstEdge + i).action == 24) hasE1 = true;
    assert(hasE1);
}

// Same logits for every position: strongly prefers E1 (24), then C4 (7).
class PreferE1ThenC4 : public Evaluator {
public:
    int boardSize() const override { return 5; }
    int history() const override { return 2; }
    int maxBatch() const override { return 8; }
    void evaluate(const float*, int B, float* logits, float* values) override {
        for (int b = 0; b < B; ++b) {
            for (int a = 0; a < 26; ++a) logits[b * 26 + a] = 0.0f;
            logits[b * 26 + 24] = 10.0f;
            logits[b * 26 + 7] = 5.0f;
            values[b] = 0.0f;
        }
    }
};

static void test_superko_checks_positions_on_the_tree_path() {
    // Now it is White who can take the ko at C4 (7), capturing B4 (6).
    // Path: root -(Black E1)-> N1 -(White C4 takes)-> N2. Black retaking at
    // B4 from N2 would recreate N1, a position that exists only inside the
    // tree (the game has never seen it), so only the path check can catch it.
    Board start = boardFromRows({".OX..",
                                 "OX.X.",
                                 ".OX..",
                                 ".....",
                                 "....."}, Color::Black);
    Game game(start, 7.5f, 50);
    PreferE1ThenC4 ev;
    Mcts mcts(serialConfig(5), ev);
    Rng rng(8);
    mcts.search(game, rng);

    const int n1 = mcts.edge(rootEdge(mcts, 24)).child;
    assert(n1 >= 0 && mcts.node(n1).expanded);
    int n2 = -1;
    const MctsNode& a = mcts.node(n1);
    for (int i = 0; i < a.numEdges; ++i)
        if (mcts.edge(a.firstEdge + i).action == 7) n2 = mcts.edge(a.firstEdge + i).child;
    assert(n2 >= 0 && mcts.node(n2).expanded);
    const MctsNode& b = mcts.node(n2);
    assert(b.board.isLegal(6));
    assert(b.board.hashAfter(6) == a.board.hash());   // the retake recreates N1 ...
    assert(!game.seen(a.board.hash()));                // ... which the game never saw
    for (int i = 0; i < b.numEdges; ++i) assert(mcts.edge(b.firstEdge + i).action != 6);
}

static void test_finds_the_single_winning_capture() {
    Game game = winningCaptureGame(1);   // every child position is terminal
    UniformEvaluator ev(5, 2);
    Mcts mcts(serialConfig(50), ev);
    Rng rng(4);
    mcts.search(game, rng);

    assert(ev.calls() == 1);   // only the root: terminal leaves are scored by the rules
    assert(mcts.selectAction(0.0f, rng) == 23);
    assert(mcts.rootValue() > 0.0f);   // Black, to move, expects to win
    const MctsEdge& win = mcts.edge(rootEdge(mcts, 23));
    assert(win.visits > 25);
    assert(win.valueSum == static_cast<float>(win.visits));   // every visit was a +1 for Black
}

static void test_pass_edge_after_opponent_pass_wins_by_komi() {
    // Black passed on an empty board. White passing ends the game and White
    // wins by komi: every visit of the pass edge is a +1 for White, who chooses it.
    Game game(5, 7.5f, defaultMaxMoves(5));
    game.play(25);
    UniformEvaluator ev(5, 2, 0.0f);
    Mcts mcts(serialConfig(64), ev);
    Rng rng(16);
    mcts.search(game, rng);
    assert(mcts.node(0).passes == 1);
    const MctsEdge& pass = mcts.edge(rootEdge(mcts, 25));
    assert(pass.visits > 0);
    assert(pass.valueSum == static_cast<float>(pass.visits));
}

static void test_stale_pass_count_rebuilds_the_tree() {
    // Same stones, side to move and ply, but different consecutive passes.
    Board withStone = boardFromRows({".....", ".....", "..X..", ".....", "....."}, Color::Black);
    Board empty = boardFromRows({".....", ".....", ".....", ".....", "....."}, Color::Black);
    Game a(withStone, 7.5f, 50);
    a.play(25);                 // pass: ply 1, passes 1, White to move
    Game b(empty, 7.5f, 50);
    b.play(12);                 // stone at the centre: ply 1, passes 0, White to move
    assert(a.board() == b.board() && a.moveCount() == b.moveCount());
    assert(a.consecutivePasses() == 1 && b.consecutivePasses() == 0);
    UniformEvaluator ev(5, 2);
    Mcts mcts(serialConfig(10), ev);
    Rng rng(17);
    mcts.search(a, rng);
    assert(mcts.node(0).passes == 1);
    mcts.search(b, rng);
    assert(mcts.node(0).passes == 0);
    assert(mcts.rootTotalVisits() == 10);   // rebuilt, not accumulated
}

static void test_select_action_temperature_zero_picks_most_visited() {
    Game game(5, 7.5f, defaultMaxMoves(5));
    UniformEvaluator ev(5, 2);
    Mcts mcts(serialConfig(60), ev);
    Rng rng(5);
    mcts.search(game, rng);
    std::vector<int> visits = mcts.rootVisits();
    const int best = *std::max_element(visits.begin(), visits.end());
    for (int i = 0; i < 20; ++i) assert(visits[mcts.selectAction(0.0f, rng)] == best);
}

static void test_select_action_temperature_one_follows_visit_counts() {
    Game game(5, 7.5f, defaultMaxMoves(5));
    UniformEvaluator ev(5, 2);
    Mcts mcts(serialConfig(60), ev);
    Rng rng(6);
    mcts.search(game, rng);
    std::vector<int> visits = mcts.rootVisits();
    const int total = sum(visits);
    std::vector<int> counts(visits.size(), 0);
    const int draws = 2000;
    for (int i = 0; i < draws; ++i) counts[mcts.selectAction(1.0f, rng)]++;
    for (size_t a = 0; a < visits.size(); ++a) {
        if (visits[a] == 0) assert(counts[a] == 0);   // never an unvisited action
        assert(near(static_cast<double>(counts[a]) / draws, static_cast<double>(visits[a]) / total, 0.05));
    }
}

static void test_backup_flips_sign_one_ply() {
    // Every leaf is worth +0.5 to ITS side to move (White, one ply down), so
    // each root edge -- chosen by Black -- must average -0.5. With 20
    // simulations and 26 root edges every simulation stops one ply down:
    // an unvisited edge (Q = 0, full bonus) always beats a visited one.
    Game game(5, 7.5f, defaultMaxMoves(5));
    UniformEvaluator ev(5, 2, 0.5f);
    Mcts mcts(serialConfig(20), ev);
    Rng rng(7);
    mcts.search(game, rng);
    assert(mcts.node(0).netValue == 0.5f);
    const MctsNode& root = mcts.node(0);
    int visited = 0;
    for (int i = 0; i < root.numEdges; ++i) {
        const MctsEdge& e = mcts.edge(root.firstEdge + i);
        if (e.visits == 0) continue;
        ++visited;
        assert(e.valueSum / e.visits == -0.5f);
    }
    assert(visited == 20);
    assert(mcts.rootValue() == -0.5f);
}

// Records the batch size of every evaluate() call.
class RecordingEvaluator : public Evaluator {
public:
    explicit RecordingEvaluator(int n) : n_(n) {}
    int boardSize() const override { return n_; }
    int history() const override { return 2; }
    int maxBatch() const override { return 64; }
    void evaluate(const float*, int B, float* logits, float* values) override {
        batches.push_back(B);
        std::fill(logits, logits + static_cast<size_t>(B) * (n_ * n_ + 1), 0.0f);
        std::fill(values, values + B, 0.0f);
    }
    std::vector<int> batches;
private:
    int n_;
};

static Game earlyNineByNine() {
    Game g(9, 7.5f, defaultMaxMoves(9));
    g.play(40);   // E5
    g.play(30);   // D6
    return g;
}

static void test_batched_visits_exact_and_virtual_loss_removed() {
    for (int batch : {1, 3, 8, 16}) {
        for (int sims : {10, 50}) {
            Game game = earlyNineByNine();
            UniformEvaluator ev(9, 2);
            MctsConfig cfg;
            cfg.simulations = sims;
            cfg.batchSize = batch;   // 16 > 10 simulations: the round is cut short
            Mcts mcts(cfg, ev);
            Rng rng(11);
            mcts.search(game, rng);
            assert(mcts.rootTotalVisits() == sims);
            assert(mcts.node(0).visits == sims);
            assert(mcts.totalVirtualLoss() == 0);
            for (size_t i = 0; i < mcts.nodeCount(); ++i) assert(!mcts.node(static_cast<int>(i)).pending);
            mcts.search(game, rng);
            assert(mcts.rootTotalVisits() == 2 * sims);
            assert(mcts.totalVirtualLoss() == 0);
        }
    }
}

static void test_batch_one_calls_evaluator_once_per_leaf() {
    // 30 simulations over 80 root edges: every simulation stops one ply down,
    // so no leaf is terminal and each simulation evaluates exactly one leaf.
    Game game = earlyNineByNine();
    UniformEvaluator ev(9, 2);
    MctsConfig cfg;
    cfg.simulations = 30;
    cfg.batchSize = 1;
    Mcts mcts(cfg, ev);
    Rng rng(12);
    mcts.search(game, rng);
    assert(mcts.evaluatorBatches() == 30 + 1);   // + the root
    assert(ev.calls() == 31);
}

static void test_batching_reduces_evaluator_calls() {
    Game game = earlyNineByNine();
    UniformEvaluator ev1(9, 2), ev8(9, 2);
    MctsConfig cfg;
    cfg.simulations = 64;
    cfg.batchSize = 1;
    Mcts serial(cfg, ev1);
    cfg.batchSize = 8;
    Mcts batched(cfg, ev8);
    Rng r1(13), r8(13);
    serial.search(game, r1);
    batched.search(game, r8);
    assert(batched.rootTotalVisits() == 64);
    assert(batched.evaluatorBatches() < serial.evaluatorBatches());
    // Virtual loss spreads the 8 descents of a round over 8 different root
    // edges, so no round collides here: 1 root call + 64 / 8 rounds.
    assert(batched.evaluatorBatches() == 1 + 8);
}

static void test_evaluator_never_sees_more_than_batch_size() {
    // White owns the board with two one-point eyes (A5 = 0, E1 = 24). Black's
    // only legal action is pass, so every descent of a round funnels into the
    // same child: collisions are guaranteed. Pass-pass endings also give
    // terminal leaves inside the rounds.
    Board start = boardFromRows({".OOOO",
                                 "OOOOO",
                                 "OOOOO",
                                 "OOOOO",
                                 "OOOO."}, Color::Black);
    Game game(start, 7.5f, 50);
    RecordingEvaluator ev(5);
    MctsConfig cfg;
    cfg.simulations = 40;
    cfg.batchSize = 5;
    Mcts mcts(cfg, ev);
    Rng rng(14);
    mcts.search(game, rng);

    assert(mcts.node(0).numEdges == 1);                  // pass only
    assert(ev.batches.size() >= 2);
    assert(ev.batches[0] == 1);                          // the root, alone
    assert(ev.batches[1] == 1);                          // 2nd descent collided with the 1st
    int evaluated = 0;
    for (int b : ev.batches) {
        assert(b >= 1 && b <= 5);
        evaluated += b;
    }
    int expanded = 0, terminal = 0;
    for (size_t i = 0; i < mcts.nodeCount(); ++i) {
        const MctsNode& nd = mcts.node(static_cast<int>(i));
        expanded += nd.expanded ? 1 : 0;
        terminal += nd.passes >= 2 ? 1 : 0;
    }
    assert(terminal > 0);
    assert(evaluated == expanded);   // every evaluated leaf is expanded exactly once
    assert(static_cast<long long>(ev.batches.size()) == mcts.evaluatorBatches());
    assert(mcts.rootTotalVisits() == 40);
    assert(mcts.totalVirtualLoss() == 0);
}

static void test_batched_search_is_deterministic() {
    Game game = earlyNineByNine();
    MctsConfig cfg;
    cfg.simulations = 100;
    cfg.batchSize = 4;
    UniformEvaluator ev1(9, 2), ev2(9, 2);
    Mcts a(cfg, ev1), b(cfg, ev2);
    Rng ra(99), rb(99);
    a.search(game, ra);
    b.search(game, rb);
    assert(a.rootVisits() == b.rootVisits());
    assert(a.nodeCount() == b.nodeCount());
}

static void test_batched_search_still_finds_winning_capture() {
    Game game = winningCaptureGame(1);
    UniformEvaluator ev(5, 2);
    MctsConfig cfg;
    cfg.simulations = 50;
    cfg.batchSize = 8;
    Mcts mcts(cfg, ev);
    Rng rng(15);
    mcts.search(game, rng);
    assert(ev.calls() == 1);   // terminal leaves are backed up inside the round
    assert(mcts.rootTotalVisits() == 50);
    assert(mcts.totalVirtualLoss() == 0);
    assert(mcts.selectAction(0.0f, rng) == 23);
}

int main() {
    test_root_visits_sum_to_simulations_and_tree_is_kept();
    test_only_legal_actions_get_edges();
    test_superko_excludes_immediate_ko_retake_inside_tree();
    test_superko_checks_positions_on_the_tree_path();
    test_finds_the_single_winning_capture();
    test_select_action_temperature_zero_picks_most_visited();
    test_select_action_temperature_one_follows_visit_counts();
    test_backup_flips_sign_one_ply();
    test_pass_edge_after_opponent_pass_wins_by_komi();
    test_stale_pass_count_rebuilds_the_tree();
    test_batched_visits_exact_and_virtual_loss_removed();
    test_batch_one_calls_evaluator_once_per_leaf();
    test_batching_reduces_evaluator_calls();
    test_evaluator_never_sees_more_than_batch_size();
    test_batched_search_is_deterministic();
    test_batched_search_still_finds_winning_capture();
    std::printf("test_mcts: all tests passed\n");
    return 0;
}
