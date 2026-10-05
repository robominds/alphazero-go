// The Go Text Protocol (GTP, version 2) is how Go programs talk to each
// other and to graphical boards such as Sabaki: a controller writes one
// command per line ("play b D4", "genmove w") and the engine answers each
// with "= result" or "? error", terminated by a blank line. Speaking it lets
// a trained network play an outside reference program such as GNU Go, so
// its strength can be measured on a scale that is not its own ladder.
//
// GtpEngine is the whole protocol as a library class: handle() takes one
// input line and returns the complete response, so tests can drive it
// without pipes and apps/gtp.cpp is only a read-print loop. It is for
// evaluation only: genmove searches at temperature 0 without root noise,
// exactly like the arena after its opening moves, and nothing is trained.
//
// Rules are the project's own (Game): positional superko, suicide illegal,
// Tromp-Taylor area scoring for final_score. The engine never resigns.
#pragma once
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>
#include "azgo/evaluator.hpp"
#include "azgo/game.hpp"
#include "azgo/mcts.hpp"
#include "azgo/rng.hpp"

namespace azgo {

struct GtpOptions {
    MctsConfig search;              // addNoise is forced off: evaluation never adds root noise
    float komi = 7.5f;              // until the controller sends "komi"
    uint64_t seed = 1;              // one Rng for the whole session (ties, random symmetry)
    std::string name = "alphazero-go";
    std::string version = "1";
    std::ostream* log = nullptr;    // one diagnostic line per genmove; never stdout
};

class GtpEngine {
public:
    // The board size is the evaluator's (a network plays one size only).
    GtpEngine(Evaluator& evaluator, const GtpOptions& options);

    // One line of controller input (with or without its trailing newline)
    // in, the full response out: "=[id] result\n\n" or "?[id] message\n\n".
    // A line that holds no command (empty, blank, or only a # comment)
    // returns "" and must produce no output.
    std::string handle(const std::string& line);

    bool quitRequested() const { return quit_; }
    const Game& game() const { return game_; }

    // Every command handle() knows, in list_commands order.
    static const std::vector<std::string>& commands();

private:
    Evaluator& eval_;
    GtpOptions opts_;
    Game game_;
    Mcts mcts_;
    Rng rng_;
    // Game moves added by each accepted play/genmove: 1, or 2 when an
    // out-of-turn move first recorded a pass for the side to move. undo
    // removes the last entry's moves.
    std::vector<int> added_;
    bool quit_ = false;

    std::string dispatch(const std::string& command, const std::vector<std::string>& args,
                         bool& ok);
    void restart(float komi, std::vector<int> moves);   // new Game replaying moves; tree reset
    bool playAs(Color color, int action);                      // false (state unchanged) if illegal
};

} // namespace azgo
