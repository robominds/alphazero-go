#include <cassert>
#include <cstdio>
#include <string>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/board.hpp"
#include "azgo/evaluator.hpp"
#include "azgo/gtp.hpp"
#include "azgo/network.hpp"

using namespace azgo;

static GtpOptions fastOptions(int simulations = 16) {
    GtpOptions o;
    o.search.simulations = simulations;
    o.search.batchSize = 4;
    return o;
}

// The result text of a successful response, after "=[id] ", without the
// closing blank line. Asserts the response succeeded.
static std::string ok(GtpEngine& e, const std::string& line) {
    const std::string r = e.handle(line);
    assert(r.size() >= 3 && r[0] == '=' && r.compare(r.size() - 2, 2, "\n\n") == 0);
    std::string body = r.substr(1, r.size() - 3);
    if (!body.empty() && body[0] == ' ') body.erase(0, 1);
    return body;
}

// Framing: ids, '=' and '?', the blank-line terminator, comments, case.
static void test_framing() {
    UniformEvaluator eval(9, 2);
    GtpEngine e(eval, fastOptions());
    assert(e.handle("protocol_version") == "= 2\n\n");
    assert(e.handle("7 protocol_version") == "=7 2\n\n");
    assert(e.handle("name") == "= alphazero-go\n\n");
    assert(e.handle("version\n") == "= 1\n\n");
    assert(e.handle("12 name # a trailing comment") == "=12 alphazero-go\n\n");
    assert(e.handle("\tname\r") == "= alphazero-go\n\n");   // tab is a space, CR is dropped
    // No command, no response at all.
    assert(e.handle("") == "");
    assert(e.handle("   \t ") == "");
    assert(e.handle("# just a comment") == "");
    // Errors: '?' plus the message, with the id when one was given.
    assert(e.handle("frobnicate") == "? unknown command\n\n");
    assert(e.handle("3 frobnicate") == "?3 unknown command\n\n");
    assert(e.handle("NAME") == "? unknown command\n\n");    // command names are case-sensitive
    assert(e.handle("5") == "?5 syntax error\n\n");
    // An empty result is "=" (with the id) and the blank line.
    assert(e.handle("clear_board") == "=\n\n");
    assert(e.handle("4 clear_board") == "=4\n\n");

    assert(e.handle("known_command play") == "= true\n\n");
    assert(e.handle("known_command PLAY") == "= false\n\n");
    assert(e.handle("known_command gogui-analyze_commands") == "= false\n\n");
    assert(e.handle("known_command") == "? syntax error\n\n");
    std::string listed = ok(e, "list_commands");
    std::string expected;
    for (const std::string& c : GtpEngine::commands()) expected += (expected.empty() ? "" : "\n") + c;
    assert(listed == expected);
    for (const char* c : {"protocol_version", "name", "version", "known_command", "list_commands", "quit",
                          "boardsize", "clear_board", "komi", "play", "genmove", "undo", "showboard",
                          "final_score"})
        assert(e.handle(std::string("known_command ") + c) == "= true\n\n");

    assert(!e.quitRequested());
    assert(e.handle("9 quit") == "=9\n\n");
    assert(e.quitRequested());
}

// Only the network's own size is acceptable; accepting it clears the board.
static void test_boardsize() {
    UniformEvaluator eval(9, 2);
    GtpEngine e(eval, fastOptions());
    assert(e.handle("boardsize 19") == "? unacceptable size\n\n");
    assert(e.handle("boardsize 5") == "? unacceptable size\n\n");
    assert(e.handle("boardsize nine") == "? syntax error\n\n");
    assert(e.handle("boardsize") == "? syntax error\n\n");
    ok(e, "play b E5");
    assert(e.game().moveCount() == 1);
    assert(e.handle("boardsize 9") == "=\n\n");
    assert(e.game().moveCount() == 0 && e.game().size() == 9);
}

// GTP columns skip I: on 9x9 they are A-H and J, so J9 is the top-right
// corner and I5 is no point at all. Vertices and colors ignore case.
static void test_coordinates() {
    UniformEvaluator eval(9, 2);
    GtpEngine e(eval, fastOptions());
    assert(e.handle("play b J9") == "=\n\n");
    assert(e.game().board().at(8) == Color::Black);         // row 0 (top), column 8
    assert(e.handle("play white a1") == "=\n\n");
    assert(e.game().board().at(72) == Color::White);        // row 8 (bottom), column 0
    assert(e.handle("play BLACK h1") == "=\n\n");
    assert(e.game().board().at(79) == Color::Black);
    assert(e.handle("play w I5") == "? syntax error\n\n");   // no I column
    assert(e.handle("play w K5") == "? syntax error\n\n");   // beyond J on 9x9
    assert(e.handle("play w A10") == "? syntax error\n\n");
    assert(e.handle("play w A0") == "? syntax error\n\n");
    assert(e.handle("play x A2") == "? syntax error\n\n");   // not a color
    assert(e.handle("play w") == "? syntax error\n\n");
    assert(e.game().moveCount() == 3);
    const std::string board = ok(e, "showboard");
    assert(board[0] == '\n');
    assert(board.find("A B C D E F G H J") != std::string::npos);
    assert(board.find('I') == std::string::npos);
    assert(board.find("\n\n") == std::string::npos);          // a blank line would end the response early
}

// Occupied, suicide and positional superko are illegal; pass is a move.
static void test_play_illegal_and_pass() {
    UniformEvaluator eval(9, 2);
    GtpEngine e(eval, fastOptions());
    ok(e, "play b E5");
    assert(e.handle("play w E5") == "? illegal move\n\n");   // occupied
    assert(e.game().moveCount() == 1);                       // a refused move changes nothing

    // A ko. After Black F5 captures E5, White retaking at E5 would recreate
    // the position before F5.
    //     D E F G
    //  6  . X O .
    //  5  X O X O
    //  4  . X O .
    ok(e, "clear_board");
    for (const char* m : {"play b E6", "play w F6", "play b D5", "play w E5", "play b E4", "play w G5",
                          "play b A1", "play w F4", "play b F5"})
        ok(e, m);
    assert(e.game().board().at(parseAction("E5", 9)) == Color::Empty);   // captured
    assert(e.handle("play w E5") == "? illegal move\n\n");
    ok(e, "play w pass");
    assert(e.game().consecutivePasses() == 1);

    // Suicide: White A9 with Black on A8 and B9.
    ok(e, "clear_board");
    for (const char* m : {"play b A8", "play w J1", "play b B9"}) ok(e, m);
    assert(e.handle("play w A9") == "? illegal move\n\n");

    // Two passes end the game; after that no move is legal and genmove
    // answers pass without playing.
    ok(e, "clear_board");
    ok(e, "play b pass");
    ok(e, "play w PASS");
    assert(e.game().isTerminal());
    assert(e.handle("play b E5") == "? illegal move\n\n");
    assert(ok(e, "genmove b") == "pass");
    assert(e.game().moveCount() == 2);
}

// GTP allows two moves of one color in a row: the side to move is recorded
// as passing in between, and undo takes both back.
static void test_out_of_turn_and_undo() {
    UniformEvaluator eval(9, 2);
    GtpEngine e(eval, fastOptions());
    assert(e.handle("undo") == "? cannot undo\n\n");
    ok(e, "play b C3");
    ok(e, "play b D4");
    assert(e.game().moveCount() == 3);
    assert(e.game().moves()[1] == 81);                       // White's pass
    assert(e.game().board().toMove() == Color::White);
    ok(e, "undo");
    assert(e.game().moveCount() == 1);
    assert(e.game().board().toMove() == Color::White);
    const std::string w = ok(e, "genmove w");
    assert(e.game().moveCount() == 2);
    assert(parseAction(w, 9) == e.game().moves()[1]);
    ok(e, "undo");
    ok(e, "undo");
    assert(e.game().moveCount() == 0);
    assert(e.handle("undo") == "? cannot undo\n\n");
}

// genmove returns a legal vertex and plays it; a pass is spelled "pass".
static void test_genmove() {
    UniformEvaluator eval(9, 2);
    GtpEngine e(eval, fastOptions());
    for (int k = 0; k < 6; ++k) {
        assert(!e.game().isTerminal());                      // checked below: seed 1 never passes twice here
        const std::string color = k % 2 == 0 ? "b" : "w";
        const std::string v = ok(e, "genmove " + color);
        const int a = parseAction(v, 9);
        assert(a >= 0);
        assert(v == actionToString(a, 9));                   // upper case, "pass" for a pass
        assert(e.game().moveCount() == k + 1);
        assert(e.game().moves().back() == a);
    }
    assert(e.handle("genmove") == "? syntax error\n\n");
    assert(e.handle("genmove purple") == "? syntax error\n\n");
    // Out of turn: Black to move, asked for White.
    ok(e, "genmove w");
    assert(e.game().moveCount() == 8 && e.game().moves()[6] == 81);
}

// The same seed and commands give the same moves, with random symmetry
// too, so a match against a deterministic opponent can be replayed.
static void test_genmove_deterministic() {
    NetConfig c;
    c.boardSize = 5;
    c.history = 2;
    c.blocks = 1;
    c.filters = 4;
    c.valueHidden = 8;
    Network net(c, 7);
    NaiveBackend backend;
    for (bool symmetry : {false, true}) {
        auto play = [&](uint64_t seed) {
            LocalEvaluator eval(net, backend, 4);
            GtpOptions o = fastOptions(24);
            o.seed = seed;
            o.search.randomSymmetry = symmetry;
            GtpEngine e(eval, o);
            std::string moves;
            for (int k = 0; k < 8; ++k) moves += ok(e, k % 2 == 0 ? "genmove b" : "genmove w") + " ";
            ok(e, "play b pass");
            moves += ok(e, "genmove w");
            return moves;
        };
        const std::string first = play(3);
        assert(play(3) == first);
    }
}

// final_score is Tromp-Taylor area minus komi, from the board as it stands.
static void test_final_score() {
    UniformEvaluator eval(5, 2);
    GtpEngine e(eval, fastOptions());
    assert(ok(e, "final_score") == "W+7.5");                 // empty board: komi only
    // One Black stone owns the whole 5x5 board: 25 - 7.5.
    ok(e, "play b A1");
    assert(ok(e, "final_score") == "B+17.5");
    ok(e, "komi 25");
    assert(ok(e, "final_score") == "0");
    ok(e, "komi 30");
    assert(ok(e, "final_score") == "W+5");
    assert(e.game().moveCount() == 1);                       // komi keeps the game
    assert(e.handle("komi seven") == "? syntax error\n\n");

    // Walls on C and D: Black owns columns A-C (15), White D-E (10).
    ok(e, "clear_board");
    ok(e, "komi 7.5");
    for (int row = 1; row <= 5; ++row) {
        ok(e, "play b C" + std::to_string(row));
        ok(e, "play w D" + std::to_string(row));
    }
    assert(ok(e, "final_score") == "W+2.5");                  // 15 - 10 - 7.5
    ok(e, "komi 0");
    assert(ok(e, "final_score") == "B+5");
}

int main() {
    test_framing();
    test_boardsize();
    test_coordinates();
    test_play_illegal_and_pass();
    test_out_of_turn_and_undo();
    test_genmove();
    test_genmove_deterministic();
    test_final_score();
    std::printf("test_gtp: all tests passed\n");
    return 0;
}
