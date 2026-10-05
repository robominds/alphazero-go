#include "azgo/sgf.hpp"
#include <cassert>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace azgo {

namespace {

std::string formatG(double x) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%g", x);
    return buf;
}

// Inside a property value only ']' and '\' are special; a backslash escapes them.
std::string escapeValue(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == ']' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

std::string pointToSgf(int action, int n) {
    assert(action >= 0 && action <= n * n);
    if (action == n * n) return "";   // pass
    char xy[3] = {static_cast<char>('a' + action % n), static_cast<char>('a' + action / n), 0};
    return xy;
}

// ---- parser -------------------------------------------------------------

using Property = std::pair<std::string, std::vector<std::string>>;
using Node = std::vector<Property>;

// Recursive descent over  GameTree = "(" Node+ GameTree* ")",
// Node = ";" Property*,  Property = Ident ("[" value "]")+.
// Nodes on the main line (first child at every branch) are collected;
// variations are parsed for well-formedness and then dropped.
class Parser {
public:
    explicit Parser(const std::string& text) : s_(text) {}

    std::vector<Node> parse() {
        skipSpace();
        if (pos_ >= s_.size() || s_[pos_] != '(') fail("expected '(' at the start");
        std::vector<Node> mainLine;
        gameTree(true, mainLine);
        // A collection may hold further games; check them, but read only the first.
        while (peek() == '(') gameTree(false, mainLine);
        if (pos_ < s_.size()) fail("unexpected text after the game");
        return mainLine;
    }

private:
    const std::string& s_;
    size_t pos_ = 0;

    [[noreturn]] void fail(const std::string& what) const {
        throw std::runtime_error("sgf: " + what + " (at offset " + std::to_string(pos_) + ")");
    }
    void skipSpace() {
        while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
    }
    char peek() {
        skipSpace();
        return pos_ < s_.size() ? s_[pos_] : '\0';
    }

    void gameTree(bool onMainLine, std::vector<Node>& mainLine) {
        ++pos_;   // '('
        if (peek() != ';') fail("expected ';' to start a node");
        while (peek() == ';') {
            Node n = node();
            if (onMainLine) mainLine.push_back(std::move(n));
        }
        bool first = true;
        while (peek() == '(') {
            gameTree(onMainLine && first, mainLine);   // only the first child continues the main line
            first = false;
        }
        if (peek() != ')') fail("expected ')' (unbalanced parentheses)");
        ++pos_;
    }

    Node node() {
        ++pos_;   // ';'
        Node n;
        while (std::isalpha(static_cast<unsigned char>(peek()))) {
            std::string ident;
            while (pos_ < s_.size() && std::isalpha(static_cast<unsigned char>(s_[pos_]))) {
                // Old FF[3] files may spell identifiers with lowercase letters
                // ("AddBlack"); only the uppercase ones are significant.
                if (std::isupper(static_cast<unsigned char>(s_[pos_]))) ident += s_[pos_];
                ++pos_;
            }
            if (ident.empty()) fail("property identifier without uppercase letters");
            std::vector<std::string> values;
            while (peek() == '[') values.push_back(value());
            if (values.empty()) fail("property " + ident + " has no value");
            n.emplace_back(ident, std::move(values));
        }
        return n;
    }

    std::string value() {
        ++pos_;   // '['
        std::string v;
        while (pos_ < s_.size() && s_[pos_] != ']') {
            if (s_[pos_] == '\\') {
                ++pos_;   // the escaped character is taken literally
                if (pos_ >= s_.size()) break;
            }
            v += s_[pos_++];
        }
        if (pos_ >= s_.size()) fail("unterminated property value");
        ++pos_;   // ']'
        return v;
    }
};

int parseSgfInt(const std::string& v, const char* what) {
    errno = 0;
    char* end = nullptr;
    long x = std::strtol(v.c_str(), &end, 10);
    if (v.empty() || errno == ERANGE || *end != '\0')
        throw std::runtime_error(std::string("sgf: malformed ") + what + " '" + v + "'");
    return static_cast<int>(x);
}

int sgfToAction(const std::string& v, int n) {
    // "" is the FF[4] pass; "tt" is the traditional pass on boards up to 19x19.
    if (v.empty() || (v == "tt" && n <= 19)) return n * n;
    if (v.size() != 2 || v[0] < 'a' || v[0] >= 'a' + n || v[1] < 'a' || v[1] >= 'a' + n)
        throw std::runtime_error("sgf: bad coordinate '" + v + "' for a " + std::to_string(n) +
                                 "x" + std::to_string(n) + " board");
    int col = v[0] - 'a', row = v[1] - 'a';
    return row * n + col;
}

} // namespace

std::string gameToSgf(int n, float komi, const std::vector<int>& moves, float scoreForBlack,
                      const std::string& comment) {
    std::string result;
    if (scoreForBlack > 0) result = "B+" + formatG(scoreForBlack);
    else if (scoreForBlack < 0) result = "W+" + formatG(-scoreForBlack);
    else result = "0";   // exact tie, possible with integer komi
    return gameToSgf(n, komi, moves, result, comment);
}

std::string gameToSgf(int n, float komi, const std::vector<int>& moves, const std::string& result,
                      const std::string& comment) {
    std::string out = "(;GM[1]FF[4]SZ[" + std::to_string(n) + "]KM[" + formatG(komi) + "]RE[" +
                      result + "]";
    if (!comment.empty()) out += "C[" + escapeValue(comment) + "]";
    for (size_t ply = 0; ply < moves.size(); ++ply) {
        out += ply % 2 == 0 ? ";B[" : ";W[";   // Black moves first, no handicap
        out += pointToSgf(moves[ply], n) + "]";
    }
    out += ")";
    return out;
}

SgfGame parseSgf(const std::string& text) {
    std::vector<Node> nodes = Parser(text).parse();

    SgfGame game;
    bool haveSize = false, haveKomi = false;
    for (const Node& node : nodes) {
        for (const Property& p : node) {
            if (p.first == "SZ" && !haveSize) {
                game.boardSize = parseSgfInt(p.second[0], "SZ");
                haveSize = true;
            } else if (p.first == "KM" && !haveKomi) {
                const std::string& v = p.second[0];
                errno = 0;
                char* end = nullptr;
                double k = std::strtod(v.c_str(), &end);
                if (v.empty() || errno == ERANGE || *end != '\0' || !std::isfinite(k))
                    throw std::runtime_error("sgf: malformed KM '" + v + "'");
                game.komi = static_cast<float>(k);
                haveKomi = true;
            }
        }
    }
    if (game.boardSize < 5 || game.boardSize > 19)
        throw std::runtime_error("sgf: board size " + std::to_string(game.boardSize) +
                                 " out of range 5..19");

    for (const Node& node : nodes) {
        for (const Property& p : node) {
            if (p.first != "B" && p.first != "W") continue;
            // Our games alternate strictly, Black first; anything else
            // (handicap stones, two moves by one color) cannot be replayed.
            const char* expected = game.moves.size() % 2 == 0 ? "B" : "W";
            if (p.first != expected)
                throw std::runtime_error("sgf: move " + std::to_string(game.moves.size() + 1) +
                                         " should be played by " + expected);
            game.moves.push_back(sgfToAction(p.second[0], game.boardSize));
        }
    }
    return game;
}

} // namespace azgo
