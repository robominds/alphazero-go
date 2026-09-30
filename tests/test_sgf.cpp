#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/board.hpp"
#include "azgo/game.hpp"
#include "azgo/rng.hpp"
#include "azgo/sgf.hpp"

using namespace azgo;

static void test_write_exact() {
    // C3 on 5x5 is column 2, row 5 - 3 = 2 from the top: point 12, "cc".
    assert(parseAction("C3", 5) == 12);
    std::string sgf = gameToSgf(5, 7.5f, {12, 25}, 3.5f);
    assert(sgf == "(;GM[1]FF[4]SZ[5]KM[7.5]RE[B+3.5];B[cc];W[])");

    // Top-right corner of 9x9 is column i (index 8), row a: "ia".
    assert(gameToSgf(9, 6.5f, {8}, 1.0f) == "(;GM[1]FF[4]SZ[9]KM[6.5]RE[B+1];B[ia])");
    assert(gameToSgf(5, 7.5f, {}, -7.5f) == "(;GM[1]FF[4]SZ[5]KM[7.5]RE[W+7.5])");
}

// Review Focus 2: integer komi can produce an exact tie.
static void test_tie_result() {
    std::string sgf = gameToSgf(5, 7.0f, {25, 25}, 0.0f);
    assert(sgf == "(;GM[1]FF[4]SZ[5]KM[7]RE[0];B[];W[])");
}

static void test_comment_escaping() {
    std::string sgf = gameToSgf(5, 7.5f, {0}, -7.5f, "gen 3] \\ note");
    assert(sgf == "(;GM[1]FF[4]SZ[5]KM[7.5]RE[W+7.5]C[gen 3\\] \\\\ note];B[aa])");
    SgfGame g = parseSgf(sgf);   // the escaped ']' does not end the value early
    assert(g.boardSize == 5 && g.komi == 7.5f && (g.moves == std::vector<int>{0}));
}

static void test_round_trip_random_game() {
    Game game(9, 7.5f, defaultMaxMoves(9));
    Rng rng(2024);
    for (int ply = 0; ply < 40; ++ply) {
        std::vector<int> legal = game.legalActions();
        std::vector<int> stones;   // avoid passing, except once, so the game continues
        for (int a : legal)
            if (a != game.board().passAction()) stones.push_back(a);
        int action = (ply == 17 || stones.empty()) ? game.board().passAction()
                                                   : stones[rng.below(static_cast<int>(stones.size()))];
        game.play(action);
    }
    assert(game.moveCount() == 40);
    std::string sgf = gameToSgf(9, game.komi(), game.moves(), game.scoreForBlack(), "round trip");
    SgfGame back = parseSgf(sgf);
    assert(back.boardSize == 9);
    assert(back.komi == 7.5f);
    assert(back.moves == game.moves());
}

static void test_parse_variants() {
    // Whitespace and newlines between tokens, "tt" as pass, unknown properties ignored.
    SgfGame g = parseSgf("  (;GM[1]FF[4]\n SZ[5] KM[6.5]PB[Black player]\r\n"
                         "  ;B[aa]\n  ;W[tt] ;B [ee]\n)\n");
    assert(g.boardSize == 5 && g.komi == 6.5f);
    assert((g.moves == std::vector<int>{0, 25, 24}));

    // Only the main line (first child at each branch) is read.
    g = parseSgf("(;SZ[5];B[aa](;W[bb];B[cc](;W[dd])(;W[ee]))(;W[ed];B[de]))");
    assert((g.moves == std::vector<int>{0, 6, 12, 18}));

    // Defaults: SZ 19, KM 0.
    g = parseSgf("(;B[pd];W[dp])");
    assert(g.boardSize == 19 && g.komi == 0.0f);
    assert((g.moves == std::vector<int>{3 * 19 + 15, 15 * 19 + 3}));

    // Setup and comment properties with several values or escapes are skipped.
    g = parseSgf("(;SZ[7]AB[aa][bb]C[a \\] tricky ( comment ;];B[];W[gg])");
    assert(g.boardSize == 7 && (g.moves == std::vector<int>{49, 48}));

    // A collection of several games: the first one is read.
    g = parseSgf("(;SZ[5];B[bb])\n(;SZ[9];B[aa])\n");
    assert(g.boardSize == 5 && (g.moves == std::vector<int>{6}));
}

static bool rejects(const std::string& text) {
    try {
        parseSgf(text);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

static void test_malformed_inputs_throw() {
    assert(rejects(""));
    assert(rejects("garbage"));
    assert(rejects("(;SZ[5];B[aa]"));         // missing ')'
    assert(rejects("(;SZ[5];B[aa]))"));        // extra ')'
    assert(rejects("(;SZ[5]C[unterminated"));
    assert(rejects("(SZ[5])"));                // no node
    assert(rejects("(;SZ[5];B)"));             // property without value
    assert(rejects("(;SZ[5];B[ff])"));         // off a 5x5 board
    assert(rejects("(;SZ[5];B[a])"));          // one letter
    assert(rejects("(;SZ[5];B[AA])"));         // uppercase
    assert(rejects("(;SZ[4])"));
    assert(rejects("(;SZ[20])"));
    assert(rejects("(;SZ[9x9])"));
    assert(rejects("(;SZ[9]KM[seven])"));
    assert(rejects("(;SZ[5];W[aa])"));         // White cannot move first
    assert(rejects("(;SZ[5];B[aa];B[bb])"));   // colors must alternate
}

int main() {
    test_write_exact();
    test_tie_result();
    test_comment_escaping();
    test_round_trip_random_game();
    test_parse_variants();
    test_malformed_inputs_throw();
    std::printf("test_sgf: all tests passed\n");
    return 0;
}
