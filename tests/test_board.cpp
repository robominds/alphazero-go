#include <cassert>
#include <cstdio>
#include <string>
#include <vector>
#include "azgo/board.hpp"

using namespace azgo;

// Point index from (row, col) on an n x n board; row 0 is the top row.
static int pt(int row, int col, int n = 5) { return row * n + col; }

static void test_empty_board() {
    for (int n : {5, 9, 13, 19}) {
        Board b(n);
        assert(b.size() == n);
        assert(b.area() == n * n);
        assert(b.passAction() == n * n);
        assert(b.numActions() == n * n + 1);
        assert(b.toMove() == Color::Black);
        assert(b.stoneCount(Color::Black) == 0);
        assert(b.stoneCount(Color::White) == 0);
        assert(b.stoneCount(Color::Empty) == n * n);
        for (int a = 0; a < b.numActions(); ++a) assert(b.isLegal(a));
        assert(b.hash() == b.computeHashFromScratch());
        assert(b.areaScore() == 0);
    }
    assert(opponent(Color::Black) == Color::White);
    assert(opponent(Color::White) == Color::Black);
}

static void test_play_alternates_and_pass() {
    Board b(5);
    assert(b.play(pt(2, 2)) == 0);
    assert(b.at(pt(2, 2)) == Color::Black);
    assert(b.toMove() == Color::White);
    assert(!b.isLegal(pt(2, 2)));             // occupied
    uint64_t h = b.hash();
    assert(b.play(b.passAction()) == 0);
    assert(b.toMove() == Color::Black);
    assert(b.hash() == h);                    // pass changes no stones
}

static void test_capture_corner() {
    Board b = boardFromRows({"OX...",
                             ".....",
                             ".....",
                             ".....",
                             "....."}, Color::Black);
    assert(b.libertiesAt(pt(0, 0)) == 1);
    assert(b.play(pt(1, 0)) == 1);
    assert(b.at(pt(0, 0)) == Color::Empty);
    assert(b.stoneCount(Color::White) == 0);
    assert(b.stoneCount(Color::Black) == 2);
    assert(b.hash() == b.computeHashFromScratch());
}

static void test_capture_edge() {
    Board b = boardFromRows({".XOX.",
                             ".....",
                             ".....",
                             ".....",
                             "....."}, Color::Black);
    assert(b.play(pt(1, 2)) == 1);
    assert(b.at(pt(0, 2)) == Color::Empty);
    assert(b.hash() == b.computeHashFromScratch());
}

static void test_capture_center() {
    Board b = boardFromRows({".....",
                             "..X..",
                             ".XOX.",
                             ".....",
                             "....."}, Color::Black);
    assert(b.play(pt(3, 2)) == 1);
    assert(b.at(pt(2, 2)) == Color::Empty);
    assert(b.libertiesAt(pt(3, 2)) == 4);     // the captured point is a liberty again
    assert(b.hash() == b.computeHashFromScratch());
}

static void test_capture_multi_stone_group() {
    Board b = boardFromRows({".....",
                             ".XX..",
                             "XOO..",
                             ".XX..",
                             "....."}, Color::Black);
    assert(b.libertiesAt(pt(2, 1)) == 1);     // the whole white group shares one liberty
    assert(b.libertiesAt(pt(2, 2)) == 1);
    assert(b.play(pt(2, 3)) == 2);
    assert(b.at(pt(2, 1)) == Color::Empty && b.at(pt(2, 2)) == Color::Empty);
    assert(b.stoneCount(Color::White) == 0);
    assert(b.hash() == b.computeHashFromScratch());
}

static void test_capture_two_groups_at_once() {
    Board b = boardFromRows({"XO.OX",
                             ".X.X.",
                             ".....",
                             ".....",
                             "....."}, Color::Black);
    assert(b.play(pt(0, 2)) == 2);
    assert(b.at(pt(0, 1)) == Color::Empty && b.at(pt(0, 3)) == Color::Empty);
    assert(b.stoneCount(Color::White) == 0);
    assert(b.hash() == b.computeHashFromScratch());
}

static void test_suicide_illegal() {
    // Single-point eye: White may not play into Black's corner eye.
    Board eye = boardFromRows({".X...",
                               "X....",
                               ".....",
                               ".....",
                               "....."}, Color::White);
    assert(!eye.isLegal(pt(0, 0)));
    assert(eye.isLegal(eye.passAction()));
    // Black may fill its own eye (legal, just bad).
    eye.setToMove(Color::Black);
    assert(eye.isLegal(pt(0, 0)));

    // Multi-stone suicide: White (0,0) has one liberty at (0,1); playing
    // there joins the stones into a group with no liberties and captures nothing.
    Board multi = boardFromRows({"O.X..",
                                 "XX...",
                                 ".....",
                                 ".....",
                                 "....."}, Color::White);
    assert(!multi.isLegal(pt(0, 1)));
}

static void test_capture_that_looks_like_suicide_is_legal() {
    // Black's (0,0) is surrounded by white stones, but both are in atari
    // with (0,0) as their last liberty, so the move captures them.
    Board b = boardFromRows({".OX..",
                             "OX...",
                             "X....",
                             ".....",
                             "....."}, Color::Black);
    assert(b.isLegal(pt(0, 0)));
    // The same point is suicide for White: it would take its own groups'
    // last liberty without capturing anything.
    Board w = b;
    w.setToMove(Color::White);
    assert(!w.isLegal(pt(0, 0)));

    assert(b.play(pt(0, 0)) == 2);
    assert(b.at(pt(0, 1)) == Color::Empty && b.at(pt(1, 0)) == Color::Empty);
    assert(b.libertiesAt(pt(0, 0)) == 2);
    assert(b.hash() == b.computeHashFromScratch());
}

static void test_liberties() {
    Board b = boardFromRows({"X...X",
                             ".....",
                             "..X..",
                             ".....",
                             "..X.X"}, Color::Black);
    assert(b.libertiesAt(pt(0, 0)) == 2);     // corner
    assert(b.libertiesAt(pt(0, 4)) == 2);     // corner: the padded cells right of col 4 are Off
    assert(b.libertiesAt(pt(4, 4)) == 2);     // corner
    assert(b.libertiesAt(pt(4, 2)) == 3);     // edge
    assert(b.libertiesAt(pt(2, 2)) == 4);     // center
    assert(b.libertiesAt(pt(1, 1)) == 0);     // empty point

    // Shared liberties count once: in this L, (2,2) touches two stones.
    Board l = boardFromRows({".....",
                             ".XX..",
                             ".X...",
                             ".....",
                             "....."}, Color::Black);
    assert(l.libertiesAt(pt(1, 1)) == 7);
    assert(l.libertiesAt(pt(2, 1)) == 7);     // same group, same answer

    // An adjacent opponent stone takes a liberty away.
    l.setStone(pt(0, 1), Color::White);
    assert(l.libertiesAt(pt(1, 1)) == 6);
    assert(l.libertiesAt(pt(0, 1)) == 2);

    // Unused padded cells are Off on a 9x9 board too.
    Board nine(9);
    nine.setStone(pt(0, 8, 9), Color::White);
    nine.setStone(pt(8, 8, 9), Color::White);
    assert(nine.libertiesAt(pt(0, 8, 9)) == 2);
    assert(nine.libertiesAt(pt(8, 8, 9)) == 2);
}

static void test_hash_after_matches_play() {
    Board b = boardFromRows({"XO.OX",
                             ".X.X.",
                             ".....",
                             "..O..",
                             "....."}, Color::Black);
    for (int a = 0; a < b.numActions(); ++a) {
        if (!b.isLegal(a)) continue;
        Board next = b;
        next.play(a);
        assert(b.hashAfter(a) == next.hash());
        assert(next.hash() == next.computeHashFromScratch());
    }
    assert(b.hashAfter(b.passAction()) == b.hash());

    // Capturing move (two groups) and non-capturing move, explicitly.
    Board capture = b;
    capture.play(pt(0, 2));
    assert(b.hashAfter(pt(0, 2)) == capture.hash());
    // The same stones reached by setup have the same hash as by play.
    Board same = boardFromRows({"X.X.X",
                                ".X.X.",
                                ".....",
                                "..O..",
                                "....."}, Color::White);
    assert(capture.hash() == same.hash());
    assert(capture == same);
}

static void test_area_score() {
    assert(Board(5).areaScore() == 0);

    Board one(5);
    one.setStone(pt(2, 2), Color::Black);
    assert(one.areaScore() == 25);            // 1 stone + 24 empty points reaching only Black

    Board two = one;
    two.setStone(pt(0, 0), Color::White);
    assert(two.areaScore() == 0);             // the one empty region touches both: neutral

    // Dame: column 3 touches both walls and counts for nobody.
    // Black: 5 stones + 10 territory = 15. White: 5 stones. Score 10.
    Board dame = boardFromRows({"..X.O",
                                "..X.O",
                                "..X.O",
                                "..X.O",
                                "..X.O"}, Color::Black);
    assert(dame.areaScore() == 10);

    // Seki: the inner black pair (0,0),(1,0) and the white group share the
    // two liberties (0,1),(1,1); whoever fills one is captured, so neither
    // does. Those two points touch both colors and are neutral.
    // Black: 9 stones + 9 territory (right column and bottom row) = 18.
    // White: 5 stones. Score 13 (the 2 seki points count for nobody).
    Board seki = boardFromRows({"X.OX.",
                                "X.OX.",
                                "OOOX.",
                                "XXXX.",
                                "....."}, Color::Black);
    assert(seki.libertiesAt(pt(0, 0)) == 2);
    assert(seki.libertiesAt(pt(0, 2)) == 2);
    assert(seki.stoneCount(Color::Black) == 9);
    assert(seki.stoneCount(Color::White) == 5);
    assert(seki.areaScore() == 13);
    // Filling a shared liberty is legal but self-atari: the opponent captures.
    Board fill = seki;
    fill.play(pt(0, 1));
    assert(fill.libertiesAt(pt(0, 0)) == 1);
    assert(fill.play(pt(1, 1)) == 3);
}

static void test_gtp_coordinates() {
    for (int n : {9, 19}) {
        for (int a = 0; a <= n * n; ++a) assert(parseAction(actionToString(a, n), n) == a);
    }
    assert(actionToString(pt(8, 0, 9), 9) == "A1");        // bottom-left
    assert(actionToString(pt(0, 0, 9), 9) == "A9");        // top-left
    assert(actionToString(pt(0, 18, 19), 19) == "T19");    // top-right on 19x19
    assert(actionToString(pt(0, 8, 19), 19) == "J19");     // column 8 is J: no I
    assert(actionToString(pt(0, 7, 19), 19) == "H19");
    assert(actionToString(81, 9) == "pass");
    assert(parseAction("A1", 9) == pt(8, 0, 9));
    assert(parseAction("a1", 9) == pt(8, 0, 9));           // case-insensitive
    assert(parseAction("J9", 9) == pt(0, 8, 9));
    assert(parseAction("t19", 19) == 18);
    assert(parseAction("E3", 5) == pt(2, 4));
    assert(parseAction("pass", 9) == 81);
    assert(parseAction("PASS", 9) == 81);
    assert(parseAction("Pass", 5) == 25);
    assert(parseAction("I5", 9) == -1);                    // no I column
    assert(parseAction("Z99", 19) == -1);
    assert(parseAction("A0", 9) == -1);
    assert(parseAction("A10", 9) == -1);                   // off a 9x9 board
    assert(parseAction("K1", 9) == -1);                    // column 10 of 9
    assert(parseAction("", 9) == -1);
    assert(parseAction("A", 9) == -1);
    assert(parseAction("1A", 9) == -1);
    assert(parseAction("A1x", 9) == -1);
    assert(parseAction("A-1", 9) == -1);
}

static void test_board_from_rows_round_trip() {
    const std::vector<std::string> rows = {"X.O..",
                                           ".....",
                                           "..X..",
                                           "O...X",
                                           "....O"};
    Board b = boardFromRows(rows, Color::White);
    assert(b.toMove() == Color::White);
    for (int r = 0; r < 5; ++r) {
        for (int c = 0; c < 5; ++c) {
            char ch = rows[r][c];
            Color expect = ch == 'X' ? Color::Black : ch == 'O' ? Color::White : Color::Empty;
            assert(b.at(pt(r, c)) == expect);
        }
    }
    std::vector<uint8_t> stones(25);
    b.exportStones(stones.data());
    for (int p = 0; p < 25; ++p) assert(stones[p] == static_cast<uint8_t>(b.at(p)));
    assert(b.hash() == b.computeHashFromScratch());
}

static void test_set_stone_keeps_hash_valid() {
    Board b(5);
    b.setStone(pt(1, 1), Color::Black);
    b.setStone(pt(3, 3), Color::White);
    assert(b.hash() == b.computeHashFromScratch());
    b.setStone(pt(1, 1), Color::White);                    // overwrite a stone
    assert(b.hash() == b.computeHashFromScratch());
    b.setStone(pt(1, 1), Color::Empty);
    b.setStone(pt(3, 3), Color::Empty);
    assert(b.hash() == Board(5).hash());

    Board played(5);
    played.play(pt(1, 1));
    Board placed(5);
    placed.setStone(pt(1, 1), Color::Black);
    assert(played.hash() == placed.hash());
    assert(!(played == placed));                           // different side to move
    placed.setToMove(Color::White);
    assert(played == placed);
}

static void test_equality() {
    assert(Board(5) == Board(5));
    assert(!(Board(5) == Board(6)));
    Board a(9), b(9);
    a.setStone(40, Color::Black);
    assert(!(a == b));
    b.setStone(40, Color::Black);
    assert(a == b);
}

static void test_to_string() {
    Board b(5);
    b.setStone(pt(4, 0), Color::Black);
    b.setStone(pt(0, 4), Color::White);
    b.setToMove(Color::White);
    const std::string expected =
        "   A B C D E\n"
        " 5 . . . . O  5\n"
        " 4 . . . . .  4\n"
        " 3 . . . . .  3\n"
        " 2 . . . . .  2\n"
        " 1 X . . . .  1\n"
        "   A B C D E\n"
        "White to move\n";
    assert(b.toString() == expected);

    std::string big = Board(19).toString();
    assert(big.find("   A B C D E F G H J K L M N O P Q R S T\n") == 0);
    assert(big.find("\n19 . ") != std::string::npos);
    assert(big.find("Black to move\n") != std::string::npos);
}

int main() {
    test_empty_board();
    test_play_alternates_and_pass();
    test_capture_corner();
    test_capture_edge();
    test_capture_center();
    test_capture_multi_stone_group();
    test_capture_two_groups_at_once();
    test_suicide_illegal();
    test_capture_that_looks_like_suicide_is_legal();
    test_liberties();
    test_hash_after_matches_play();
    test_area_score();
    test_gtp_coordinates();
    test_board_from_rows_round_trip();
    test_set_stone_keeps_hash_valid();
    test_equality();
    test_to_string();
    std::printf("test_board: all tests passed\n");
    return 0;
}
