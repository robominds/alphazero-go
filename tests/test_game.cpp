#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include "azgo/game.hpp"
#include "azgo/rng.hpp"

using namespace azgo;

static bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

static int pt(int row, int col, int n = 5) { return row * n + col; }

static bool contains(const std::vector<int>& v, int x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

// A ko: White's (2,1) is in atari at (2,2). Black captures it at (2,2), after
// which Black's lone stone is itself in atari at (2,1).
static Board koStart() {
    return boardFromRows({".....",
                          ".XO..",
                          "XO.O.",
                          ".XO..",
                          "....."}, Color::Black);
}

static void test_simple_ko() {
    Game g(koStart(), 7.5f, defaultMaxMoves(5));
    const int take = pt(2, 2), retake = pt(2, 1);
    assert(g.isLegal(take));
    g.play(take);
    assert(g.board().at(retake) == Color::Empty);
    assert(g.board().libertiesAt(take) == 1);

    // Locally the retake is a legal capture, but it would recreate the
    // starting position, so positional superko forbids it.
    assert(g.board().isLegal(retake));
    assert(g.seen(g.board().hashAfter(retake)));
    assert(!g.isLegal(retake));
    assert(!contains(g.legalActions(), retake));
    assert(g.isLegal(g.board().passAction()));

    // Ko threat exchange: White plays elsewhere, Black answers elsewhere.
    // The board is now different, so the retake is allowed.
    g.play(pt(4, 4));
    g.play(pt(0, 4));
    assert(g.isLegal(retake));
    g.play(retake);
    assert(g.board().at(take) == Color::Empty);
    // And now Black's immediate retake is forbidden in turn.
    assert(g.board().isLegal(take));
    assert(!g.isLegal(take));
}

static void test_pass_never_blocked_by_superko() {
    Game g(5, 7.5f, defaultMaxMoves(5));
    // The current position is already in the history, and passing repeats it.
    assert(g.seen(g.board().hash()));
    assert(g.isLegal(g.board().passAction()));
    g.play(pt(2, 2));
    g.play(g.board().passAction());           // position repeats: still legal
    assert(g.board().hash() == g.boardAt(1).hash());
    assert(g.isLegal(g.board().passAction()));
}

static void test_two_passes_end_the_game() {
    Game g(5, 7.5f, defaultMaxMoves(5));
    const int pass = g.board().passAction();
    g.play(pass);
    assert(g.consecutivePasses() == 1);
    assert(!g.isTerminal());
    g.play(pass);
    assert(g.consecutivePasses() == 2);
    assert(g.isTerminal());
    assert(g.moveCount() == 2);
    assert(!g.isLegal(pass));
    assert(!g.isLegal(pt(0, 0)));
    assert(g.legalActions().empty());
}

static void test_move_after_pass_resets_count() {
    Game g(5, 7.5f, defaultMaxMoves(5));
    const int pass = g.board().passAction();
    g.play(pass);
    g.play(pt(1, 1));
    assert(g.consecutivePasses() == 0);
    g.play(pass);
    assert(g.consecutivePasses() == 1);
    assert(!g.isTerminal());
}

static void test_move_cap() {
    assert(defaultMaxMoves(5) == 50);
    assert(defaultMaxMoves(19) == 722);
    Game g(5, 7.5f, 3);
    assert(g.maxMoves() == 3);
    g.play(pt(0, 0));
    g.play(pt(4, 4));
    assert(!g.isTerminal());
    g.play(pt(2, 2));
    assert(g.isTerminal());
    assert(g.legalActions().empty());
    assert(!g.isLegal(g.board().passAction()));
}

static void test_score_and_result() {
    // Pass-pass on an empty board: White wins by komi.
    Game empty(5, 7.5f, defaultMaxMoves(5));
    empty.play(25);
    empty.play(25);
    assert(near(empty.scoreForBlack(), -7.5));
    assert(empty.resultForBlack() == -1.0f);

    // Review Focus 2: integer komi 0, pass-pass on an empty 5x5 board is an
    // exact tie, scored 0 rather than a win for either side.
    Game tie(5, 0.0f, defaultMaxMoves(5));
    tie.play(25);
    tie.play(25);
    assert(tie.isTerminal());
    assert(tie.scoreForBlack() == 0.0f);
    assert(tie.resultForBlack() == 0.0f);

    // One black stone owns the whole 5x5 board: 25 - 7.5.
    Board one(5);
    one.setStone(pt(2, 2), Color::Black);
    Game black(one, 7.5f, defaultMaxMoves(5));
    assert(near(black.scoreForBlack(), 17.5));
    assert(black.resultForBlack() == 1.0f);

    // Area score 10 (see test_board's dame position) with komi 10: exact tie.
    Board dame = boardFromRows({"..X.O",
                                "..X.O",
                                "..X.O",
                                "..X.O",
                                "..X.O"}, Color::Black);
    Game even(dame, 10.0f, defaultMaxMoves(5));
    assert(even.scoreForBlack() == 0.0f);
    assert(even.resultForBlack() == 0.0f);
    Game white(dame, 10.5f, defaultMaxMoves(5));
    assert(white.resultForBlack() == -1.0f);
}

static void test_history() {
    Game g(5, 7.5f, defaultMaxMoves(5));
    const std::vector<int> moves = {pt(0, 0), pt(4, 4), 25, pt(2, 2)};
    for (int m : moves) g.play(m);
    assert(g.moveCount() == 4);
    assert(g.moves() == moves);
    assert(g.boardAt(0) == Board(5));
    assert(g.boardAt(1).at(pt(0, 0)) == Color::Black);
    assert(g.boardAt(1).at(pt(4, 4)) == Color::Empty);
    assert(g.boardAt(1).toMove() == Color::White);
    assert(g.boardAt(2).at(pt(4, 4)) == Color::White);
    assert(g.boardAt(3).toMove() == Color::White);       // Black passed
    assert(g.boardAt(3).hash() == g.boardAt(2).hash());
    assert(g.boardAt(4) == g.board());
    assert(g.board().at(pt(2, 2)) == Color::White);
    assert(g.size() == 5);
    assert(g.komi() == 7.5f);
    for (int k = 0; k <= g.moveCount(); ++k) assert(g.seen(g.boardAt(k).hash()));
}

// Uniformly random legal play on every board size. Checks the invariants
// that the rest of the program relies on after every single move.
static void test_random_playouts() {
    int captureMoves = 0;
    for (int n = kMinBoardSize; n <= kMaxBoardSize; ++n) {
        for (int game = 0; game < 2; ++game) {
            Rng rng(deriveSeed(2026, static_cast<uint64_t>(n), static_cast<uint64_t>(game)));
            Game g(n, 7.5f, defaultMaxMoves(n));
            int plies = 0;
            while (!g.isTerminal()) {
                std::vector<int> legal = g.legalActions();
                assert(!legal.empty() && legal.back() == g.board().passAction());
                assert(std::is_sorted(legal.begin(), legal.end()));
                const int a = legal[static_cast<size_t>(rng.below(static_cast<int>(legal.size())))];
                const Color mover = g.board().toMove();
                const int stonesBefore = g.board().area() - g.board().stoneCount(Color::Empty);
                g.play(a);
                ++plies;
                const Board& b = g.board();
                assert(b.toMove() == opponent(mover));
                if (b.area() - b.stoneCount(Color::Empty) <= stonesBefore && a != b.passAction()) ++captureMoves;
                assert(b.hash() == b.computeHashFromScratch());
                for (int p = 0; p < b.area(); ++p) {
                    if (b.at(p) != Color::Empty) assert(b.libertiesAt(p) > 0);
                }
                assert(plies <= g.maxMoves());
            }
            assert(g.moveCount() == plies);
            assert(g.consecutivePasses() >= 2 || plies == g.maxMoves());
            const float r = g.resultForBlack();
            assert(r == 1.0f || r == -1.0f);                  // komi 7.5 cannot tie
        }
    }
    assert(captureMoves > 100);  // the playouts really exercised captures
}

int main() {
    test_simple_ko();
    test_pass_never_blocked_by_superko();
    test_two_passes_end_the_game();
    test_move_after_pass_resets_count();
    test_move_cap();
    test_score_and_result();
    test_history();
    test_random_playouts();
    std::printf("test_game: all tests passed\n");
    return 0;
}
