#include <cassert>
#include <cstdio>
#include <vector>
#include "azgo/encode.hpp"
#include "azgo/game.hpp"
#include "azgo/symmetry.hpp"

using namespace azgo;

static int pt(int row, int col, int n = 5) { return row * n + col; }

static std::vector<float> encodeB(const std::vector<const Board*>& history, int H, int sym) {
    const int n = history[0]->size();
    std::vector<float> out(static_cast<size_t>(numPlanes(H)) * n * n, -1.0f);  // -1: catch unwritten
    encodeBoards(history, H, sym, out.data());
    return out;
}

static std::vector<float> encodeG(const Game& g, int H, int sym) {
    const int n = g.size();
    std::vector<float> out(static_cast<size_t>(numPlanes(H)) * n * n, -1.0f);
    encodeGame(g, H, sym, out.data());
    return out;
}

static float at(const std::vector<float>& planes, int plane, int point, int n = 5) {
    return planes[static_cast<size_t>(plane) * n * n + point];
}

// The plane holds exactly `ones` set to 1 and every other point 0.
static bool planeIs(const std::vector<float>& planes, int plane, const std::vector<int>& ones, int n = 5) {
    for (int p = 0; p < n * n; ++p) {
        bool expectOne = false;
        for (int q : ones) expectOne = expectOne || (q == p);
        if (at(planes, plane, p, n) != (expectOne ? 1.0f : 0.0f)) return false;
    }
    return true;
}

static std::vector<int> all(int n = 5) {
    std::vector<int> v;
    for (int p = 0; p < n * n; ++p) v.push_back(p);
    return v;
}

static void test_plane_counts() {
    assert(numPlanes(1) == 4);
    assert(numPlanes(8) == 18);
    Board b(9);
    std::vector<float> out = encodeB({&b}, 8, 0);
    assert(out.size() == 18u * 81u);
}

static void test_own_and_opponent_swap_with_side_to_move() {
    Board b(5);
    b.setStone(pt(0, 0), Color::Black);
    b.setStone(pt(4, 4), Color::White);

    b.setToMove(Color::Black);
    std::vector<float> black = encodeB({&b}, 1, 0);
    assert(planeIs(black, 0, {pt(0, 0)}));          // own = Black
    assert(planeIs(black, 1, {pt(4, 4)}));          // opponent = White
    assert(planeIs(black, 2, all()));               // Black to move
    assert(planeIs(black, 3, {}));

    b.setToMove(Color::White);
    std::vector<float> white = encodeB({&b}, 1, 0);
    assert(planeIs(white, 0, {pt(4, 4)}));          // own = White
    assert(planeIs(white, 1, {pt(0, 0)}));
    assert(planeIs(white, 2, {}));
    assert(planeIs(white, 3, all()));               // White to move
}

static void test_missing_history_is_zero() {
    Board b(5);
    b.setStone(pt(2, 2), Color::Black);
    std::vector<float> out = encodeB({&b}, 4, 0);
    assert(planeIs(out, 0, {pt(2, 2)}));
    for (int plane = 1; plane < 8; ++plane) assert(planeIs(out, plane, {}));
    assert(planeIs(out, 8, all()));
    assert(planeIs(out, 9, {}));

    // A fresh game: no stones anywhere, Black to move.
    Game g(5, 7.5f, defaultMaxMoves(5));
    std::vector<float> start = encodeG(g, 8, 0);
    for (int plane = 0; plane < 16; ++plane) assert(planeIs(start, plane, {}));
    assert(planeIs(start, 16, all()));
    assert(planeIs(start, 17, {}));
}

static void test_encode_game_history() {
    // Black (0,0), White (4,4), Black (2,2): White to move after 3 moves.
    Game g(5, 7.5f, defaultMaxMoves(5));
    g.play(pt(0, 0));
    g.play(pt(4, 4));
    g.play(pt(2, 2));
    std::vector<float> out = encodeG(g, 4, 0);
    // Own = White throughout, because White is to move now.
    assert(planeIs(out, 0, {pt(4, 4)}));            // current position
    assert(planeIs(out, 1, {pt(0, 0), pt(2, 2)}));
    assert(planeIs(out, 2, {pt(4, 4)}));            // one ply ago
    assert(planeIs(out, 3, {pt(0, 0)}));
    assert(planeIs(out, 4, {}));                    // two plies ago
    assert(planeIs(out, 5, {pt(0, 0)}));
    assert(planeIs(out, 6, {}));                    // the empty start position
    assert(planeIs(out, 7, {}));
    assert(planeIs(out, 8, {}));                    // not Black to move
    assert(planeIs(out, 9, all()));                 // White to move

    // H = 2 keeps only the two most recent positions.
    std::vector<float> shortH = encodeG(g, 2, 0);
    assert(shortH.size() == 6u * 25u);
    assert(planeIs(shortH, 2, {pt(4, 4)}));
    assert(planeIs(shortH, 3, {pt(0, 0)}));
    assert(planeIs(shortH, 5, all()));

    // encodeGame is encodeBoards over the game's boards, newest first.
    std::vector<float> manual = encodeB({&g.boardAt(3), &g.boardAt(2), &g.boardAt(1), &g.boardAt(0)}, 4, 0);
    assert(manual == out);
}

static void test_symmetry_consistency() {
    Game g(5, 7.5f, defaultMaxMoves(5));
    for (int m : {pt(0, 1), pt(3, 3), pt(1, 0), pt(4, 2)}) g.play(m);
    const int H = 3, n = 5, P = numPlanes(H);
    const std::vector<float> base = encodeG(g, H, 0);
    for (int s = 0; s < kNumSymmetries; ++s) {
        const std::vector<float> t = encodeG(g, H, s);
        for (int plane = 0; plane < P; ++plane) {
            for (int p = 0; p < n * n; ++p) {
                assert(at(t, plane, transformPoint(s, p, n)) == at(base, plane, p));
            }
        }
    }
    // A non-identity transform really moves the stones.
    assert(encodeG(g, H, 1) != base);
}

int main() {
    test_plane_counts();
    test_own_and_opponent_swap_with_side_to_move();
    test_missing_history_is_zero();
    test_encode_game_history();
    test_symmetry_consistency();
    std::printf("test_encode: all tests passed\n");
    return 0;
}
