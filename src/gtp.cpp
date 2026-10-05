#include "azgo/gtp.hpp"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <utility>
#include "azgo/board.hpp"
#include "azgo/sgf.hpp"

namespace azgo {

namespace {

// The controller decides when a game ends (two passes, or its own move
// limit), so the engine's Game has no move cap of its own: a cap would make
// it refuse a legal "play" late in a long game.
constexpr int kNoMoveCap = std::numeric_limits<int>::max();

const std::vector<std::string> kCommands = {
    "protocol_version", "name",    "version", "known_command", "list_commands",
    "quit",             "boardsize", "clear_board", "komi",    "play",
    "genmove",          "undo",    "showboard", "final_score",
};

// GTP 2, "Preprocessing": drop every control character except tab and
// newline (a CR from a Windows controller among them), cut the line at '#'
// (a comment), and turn tabs into spaces.
std::string preprocess(const std::string& line) {
    std::string out;
    for (char ch : line) {
        const unsigned char u = static_cast<unsigned char>(ch);
        if (ch == '#') break;
        if (ch == '\t') out += ' ';
        else if (u >= 32 && u != 127) out += ch;
    }
    return out;
}

std::vector<std::string> splitSpaces(const std::string& text) {
    std::istringstream in(text);
    std::vector<std::string> out;
    for (std::string tok; in >> tok;) out.push_back(tok);
    return out;
}

std::string lower(std::string s) {
    for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

bool allDigits(const std::string& s) {
    if (s.empty()) return false;
    for (char ch : s)
        if (!std::isdigit(static_cast<unsigned char>(ch))) return false;
    return true;
}

// Colors are case-insensitive: "b", "B", "black", "BLACK", "w", "white".
bool parseColor(const std::string& text, Color& out) {
    const std::string t = lower(text);
    if (t == "b" || t == "black") out = Color::Black;
    else if (t == "w" || t == "white") out = Color::White;
    else return false;
    return true;
}

bool parseFloat(const std::string& s, float& out) {
    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || end != s.c_str() + s.size() || errno == ERANGE || !std::isfinite(v)) return false;
    out = static_cast<float>(v);
    return true;
}

GtpOptions evaluationOnly(GtpOptions o) {
    o.search.addNoise = false;
    return o;
}

} // namespace

const std::vector<std::string>& GtpEngine::commands() { return kCommands; }

GtpEngine::GtpEngine(Evaluator& evaluator, const GtpOptions& options)
    : eval_(evaluator),
      opts_(evaluationOnly(options)),
      game_(evaluator.boardSize(), options.komi, kNoMoveCap),
      mcts_(opts_.search, evaluator),
      rng_(options.seed) {}

std::string GtpEngine::handle(const std::string& line) {
    const std::vector<std::string> tokens = splitSpaces(preprocess(line));
    if (tokens.empty()) return "";   // blank or comment-only: no response at all

    // An optional numeric id, echoed in the response so a controller can
    // match answers to commands.
    std::string id;
    size_t k = 0;
    if (allDigits(tokens[0])) {
        id = tokens[0];
        k = 1;
    }
    bool ok = false;
    std::string result = "syntax error";
    if (k < tokens.size()) {
        ok = true;
        const std::vector<std::string> args(tokens.begin() + static_cast<long>(k) + 1, tokens.end());
        result = dispatch(tokens[k], args, ok);
    }
    // A multi-line result (showboard) starts on the line after the "=".
    std::string out = (ok ? "=" : "?") + id;
    if (!result.empty() && result[0] != '\n') out += ' ';
    return out + result + "\n\n";
}

std::string GtpEngine::dispatch(const std::string& command, const std::vector<std::string>& args,
                                bool& ok) {
    const int n = eval_.boardSize();
    auto fail = [&](const std::string& message) {
        ok = false;
        return message;
    };
    // Command names are case-sensitive in GTP: "PLAY" is not "play".
    if (command == "protocol_version") return "2";
    if (command == "name") return opts_.name;
    if (command == "version") return opts_.version;
    if (command == "known_command") {
        if (args.size() != 1) return fail("syntax error");
        for (const std::string& c : kCommands)
            if (c == args[0]) return "true";
        return "false";
    }
    if (command == "list_commands") {
        std::string out;
        for (const std::string& c : kCommands) out += (out.empty() ? "" : "\n") + c;
        return out;
    }
    if (command == "quit") {
        quit_ = true;
        return "";
    }
    if (command == "boardsize") {
        if (args.size() != 1 || !allDigits(args[0]) || args[0].size() > 3) return fail("syntax error");
        // A network is trained for one board size and can play no other.
        if (std::atoi(args[0].c_str()) != n) return fail("unacceptable size");
        restart(game_.komi(), {});
        added_.clear();
        return "";
    }
    if (command == "clear_board") {
        restart(game_.komi(), {});
        added_.clear();
        return "";
    }
    if (command == "komi") {
        float komi = 0;
        if (args.size() != 1 || !parseFloat(args[0], komi)) return fail("syntax error");
        // Komi decides every terminal value already in the search tree, so
        // the game is replayed under the new komi and the tree dropped.
        restart(komi, game_.moves());
        return "";
    }
    if (command == "play") {
        Color color;
        if (args.size() != 2 || !parseColor(args[0], color)) return fail("syntax error");
        const int action = parseAction(args[1], n);   // case-insensitive; no 'I' column
        if (action < 0) return fail("syntax error");
        if (!playAs(color, action)) return fail("illegal move");
        return "";
    }
    if (command == "genmove") {
        Color color;
        if (args.size() != 1 || !parseColor(args[0], color)) return fail("syntax error");
        const int pass = n * n;
        // The other side's turn: GTP allows two moves of one color in a
        // row, which here means the side to move passed.
        int added = 0;
        if (color != game_.board().toMove() && !game_.isTerminal()) {
            game_.play(pass);
            mcts_.advance(pass);
            added = 1;
        }
        if (game_.isTerminal()) {
            // The game is over (both passed), so there is nothing to search;
            // genmove cannot fail, and passing changes nothing.
            if (added > 0) added_.push_back(added);
            if (opts_.log) *opts_.log << "genmove: the game is over, answering pass\n";
            return "pass";
        }
        mcts_.search(game_, rng_);
        const int action = mcts_.selectAction(0.0f, rng_);   // temperature 0: most visited
        if (opts_.log) {
            const std::vector<int> visits = mcts_.rootVisits();
            char buf[160];
            std::snprintf(buf, sizeof buf, "genmove %s: %s, %d of %d root visits, value %+.2f (side to move)\n",
                          color == Color::Black ? "B" : "W", actionToString(action, n).c_str(),
                          visits[static_cast<size_t>(action)], mcts_.rootTotalVisits(),
                          mcts_.rootValue());
            *opts_.log << buf << std::flush;
        }
        game_.play(action);
        mcts_.advance(action);   // keep the subtree for the next search
        added_.push_back(added + 1);
        return actionToString(action, n);
    }
    if (command == "undo") {
        if (added_.empty()) return fail("cannot undo");
        std::vector<int> moves = game_.moves();
        moves.resize(moves.size() - static_cast<size_t>(added_.back()));
        added_.pop_back();
        restart(game_.komi(), moves);
        return "";
    }
    if (command == "showboard") {
        std::string board = game_.board().toString();   // ends in '\n', which handle() supplies
        board.pop_back();
        return "\n" + board;
    }
    if (command == "final_score") {
        // Tromp-Taylor: every stone on the board counts, dead or not.
        return scoreToResult(game_.scoreForBlack());
    }
    return fail("unknown command");
}

// moves is taken by value: callers pass game_.moves(), which the assignment
// below destroys.
void GtpEngine::restart(float komi, std::vector<int> moves) {
    game_ = Game(eval_.boardSize(), komi, kNoMoveCap);
    for (int a : moves) game_.play(a);   // each was legal when first played
    mcts_.reset();
}

bool GtpEngine::playAs(Color color, int action) {
    // Tried on a copy, so a refused move leaves the game as it was.
    Game trial = game_;
    const int pass = trial.board().passAction();
    int added = 0;
    if (color != trial.board().toMove()) {
        if (!trial.isLegal(pass)) return false;
        trial.play(pass);
        ++added;
    }
    if (!trial.isLegal(action)) return false;
    trial.play(action);
    ++added;
    for (int k = trial.moveCount() - added; k < trial.moveCount(); ++k)
        mcts_.advance(trial.moves()[static_cast<size_t>(k)]);
    game_ = std::move(trial);
    added_.push_back(added);
    return true;
}

} // namespace azgo
