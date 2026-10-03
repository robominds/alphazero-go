#include "azgo/board.hpp"
#include <cassert>
#include <cctype>
#include "azgo/rng.hpp"

namespace azgo {

namespace {

// Offsets to the four orthogonal neighbors in the padded array. Thanks to the
// Off border, p + d is always a valid index for any on-board p.
constexpr int kDirs[4] = {-kStride, -1, 1, kStride};

// GTP column letters: 'I' is skipped to avoid confusion with 'J' and '1'.
constexpr char kColumnLetters[] = "ABCDEFGHJKLMNOPQRST";

using ZobristTable = std::array<std::array<uint64_t, 2>, kCells>;

// One random key per (padded cell, stone color). Generated once, from a fixed
// seed, so hashes are identical across runs, threads and machines. A
// function-local static is initialized thread-safely on first use.
const ZobristTable& zobristTable() {
    static const ZobristTable table = [] {
        ZobristTable t{};
        uint64_t x = 0x5EEDB0A4D5EEDB0AULL;
        for (auto& cell : t) {
            cell[0] = splitmix64(x);
            cell[1] = splitmix64(x);
        }
        return t;
    }();
    return table;
}

uint64_t zobristKey(int padded, Color c) {
    assert(c == Color::Black || c == Color::White);
    return zobristTable()[padded][c == Color::Black ? 0 : 1];
}

} // namespace

Color opponent(Color c) {
    assert(c == Color::Black || c == Color::White);
    return c == Color::Black ? Color::White : Color::Black;
}

Board::Board(int n) : n_(n), toMove_(Color::Black), cells_(), hash_(0) {
    assert(n >= kMinBoardSize && n <= kMaxBoardSize);
    // Everything starts Off; only the n x n playing area is opened up. On a
    // 5x5 board the unused part of the 21x21 array stays Off as well, so it
    // behaves exactly like the sentinel border.
    cells_.fill(Color::Off);
    for (int p = 0; p < area(); ++p) cells_[pad(p)] = Color::Empty;
}

int Board::pad(int point) const {
    return (point / n_ + 1) * kStride + (point % n_ + 1);
}

Color Board::at(int point) const {
    assert(point >= 0 && point < area());
    return cells_[pad(point)];
}

int Board::stoneCount(Color c) const {
    int count = 0;
    for (int p = 0; p < area(); ++p) count += (cells_[pad(p)] == c);
    return count;
}

int Board::countLiberties(int p, int limit) const {
    const Color color = cells_[p];
    assert(color == Color::Black || color == Color::White);
    // Flood fill over the group. The scratch arrays live on this call's stack
    // (about 2 KB): no heap allocation, and safe to call from many threads.
    // `mark` flags both visited stones and already-counted liberties, so a
    // liberty shared by several stones of the group is counted once.
    std::array<int, kCells> stack;
    std::array<uint8_t, kCells> mark{};
    int top = 0, liberties = 0;
    stack[top++] = p;
    mark[p] = 1;
    while (top > 0) {
        const int q = stack[--top];
        for (int d : kDirs) {
            const int r = q + d;
            if (mark[r]) continue;
            if (cells_[r] == Color::Empty) {
                mark[r] = 1;
                if (++liberties >= limit) return liberties;
            } else if (cells_[r] == color) {
                mark[r] = 1;
                stack[top++] = r;
            }
        }
    }
    return liberties;
}

int Board::removeGroup(int p) {
    const Color color = cells_[p];
    assert(color == Color::Black || color == Color::White);
    // Stones are emptied as they are pushed, so nothing is visited twice and
    // no separate mark array is needed.
    std::array<int, kCells> stack;
    int top = 0, removed = 0;
    stack[top++] = p;
    cells_[p] = Color::Empty;
    hash_ ^= zobristKey(p, color);
    while (top > 0) {
        const int q = stack[--top];
        ++removed;
        for (int d : kDirs) {
            const int r = q + d;
            if (cells_[r] == color) {
                cells_[r] = Color::Empty;
                hash_ ^= zobristKey(r, color);
                stack[top++] = r;
            }
        }
    }
    return removed;
}

int Board::libertiesAt(int point) const {
    const int p = pad(point);
    if (cells_[p] != Color::Black && cells_[p] != Color::White) return 0;
    return countLiberties(p, kCells);
}

bool Board::isLegal(int action) const {
    if (action == passAction()) return true;
    if (action < 0 || action > passAction()) return false;
    const int p = pad(action);
    if (cells_[p] != Color::Empty) return false;
    const Color me = toMove_;
    const Color them = opponent(me);
    // The new stone survives if any neighbor gives it a liberty:
    //  - an empty neighbor is a liberty directly;
    //  - an opponent group whose only liberty is p gets captured, freeing a point;
    //  - a friendly group with another liberty besides p lends it one.
    for (int d : kDirs) {
        const int q = p + d;
        const Color c = cells_[q];
        if (c == Color::Empty) return true;
        if (c == them && countLiberties(q, 2) == 1) return true;
        if (c == me && countLiberties(q, 2) >= 2) return true;
    }
    return false;  // suicide
}

int Board::play(int action) {
    assert(isLegal(action));
    if (action == passAction()) {
        toMove_ = opponent(toMove_);
        return 0;
    }
    const int p = pad(action);
    const Color me = toMove_;
    const Color them = opponent(me);
    cells_[p] = me;
    hash_ ^= zobristKey(p, me);
    int captured = 0;
    for (int d : kDirs) {
        const int q = p + d;
        // A group already removed via another neighbor reads Empty here.
        if (cells_[q] == them && countLiberties(q, 1) == 0) captured += removeGroup(q);
    }
    assert(countLiberties(p, 1) > 0);  // isLegal ruled out suicide
    toMove_ = them;
    return captured;
}

uint64_t Board::hashAfter(int action) const {
    if (action == passAction()) return hash_;
    assert(isLegal(action));
    const int p = pad(action);
    const Color them = opponent(toMove_);
    for (int d : kDirs) {
        const int q = p + d;
        if (cells_[q] == them && countLiberties(q, 2) == 1) {
            // The move captures: let play() work out which stones go.
            Board next = *this;
            next.play(action);
            return next.hash();
        }
    }
    // No capture: the only change is one new stone.
    return hash_ ^ zobristKey(p, toMove_);
}

uint64_t Board::computeHashFromScratch() const {
    uint64_t h = 0;
    for (int point = 0; point < area(); ++point) {
        const int p = pad(point);
        if (cells_[p] == Color::Black || cells_[p] == Color::White) h ^= zobristKey(p, cells_[p]);
    }
    return h;
}

void Board::ownership(int8_t* out) const {
    // Every stone belongs to its color. Every maximal empty region is flood
    // filled once and belongs to a color only if all the stones it touches
    // have that color (seki liberties and dame touch both and belong to nobody).
    // region[0, size) lists the region's padded cells in the order they were
    // found; region[next, size) is the part still to be expanded.
    std::array<int, kCells> region;
    std::array<uint8_t, kCells> mark{};
    for (int point = 0; point < area(); ++point) {
        const int start = pad(point);
        const Color c = cells_[start];
        if (c == Color::Black) { out[point] = 1; continue; }
        if (c == Color::White) { out[point] = -1; continue; }
        if (mark[start]) continue;   // written with the region that reached it first
        int size = 0;
        bool reachesBlack = false, reachesWhite = false;
        region[size++] = start;
        mark[start] = 1;
        for (int next = 0; next < size; ++next) {
            for (int d : kDirs) {
                const int r = region[next] + d;
                if (cells_[r] == Color::Black) reachesBlack = true;
                else if (cells_[r] == Color::White) reachesWhite = true;
                else if (cells_[r] == Color::Empty && !mark[r]) {
                    mark[r] = 1;
                    region[size++] = r;
                }
            }
        }
        const int8_t owner = reachesBlack == reachesWhite ? 0 : (reachesBlack ? 1 : -1);
        for (int k = 0; k < size; ++k)   // padded index -> point (undoes pad())
            out[(region[k] / kStride - 1) * n_ + (region[k] % kStride - 1)] = owner;
    }
}

int Board::areaScore() const {
    std::array<int8_t, kMaxBoardSize * kMaxBoardSize> owner;
    ownership(owner.data());
    int score = 0;
    for (int point = 0; point < area(); ++point) score += owner[point];
    return score;
}

void Board::exportStones(uint8_t* out) const {
    for (int point = 0; point < area(); ++point) out[point] = static_cast<uint8_t>(cells_[pad(point)]);
}

std::string Board::toString() const {
    std::string letters = "  ";
    for (int col = 0; col < n_; ++col) {
        letters += ' ';
        letters += kColumnLetters[col];
    }
    letters += '\n';

    std::string s = letters;
    for (int row = 0; row < n_; ++row) {
        const int label = n_ - row;
        const std::string num = (label < 10 ? " " : "") + std::to_string(label);
        s += num;
        for (int col = 0; col < n_; ++col) {
            const Color c = cells_[pad(row * n_ + col)];
            s += ' ';
            s += c == Color::Black ? 'X' : c == Color::White ? 'O' : '.';
        }
        s += ' ' + num + '\n';
    }
    s += letters;
    s += toMove_ == Color::Black ? "Black to move\n" : "White to move\n";
    return s;
}

void Board::setStone(int point, Color c) {
    assert(point >= 0 && point < area());
    assert(c == Color::Empty || c == Color::Black || c == Color::White);
    const int p = pad(point);
    if (cells_[p] != Color::Empty) hash_ ^= zobristKey(p, cells_[p]);
    cells_[p] = c;
    if (c != Color::Empty) hash_ ^= zobristKey(p, c);
}

void Board::setToMove(Color c) {
    assert(c == Color::Black || c == Color::White);
    toMove_ = c;
}

bool Board::operator==(const Board& o) const {
    // Unused cells are Off on every board, so comparing whole arrays is exact.
    return n_ == o.n_ && toMove_ == o.toMove_ && cells_ == o.cells_;
}

Board boardFromRows(const std::vector<std::string>& rows, Color toMove) {
    const int n = static_cast<int>(rows.size());
    assert(n >= kMinBoardSize && n <= kMaxBoardSize);
    Board b(n);
    for (int row = 0; row < n; ++row) {
        assert(static_cast<int>(rows[row].size()) == n);
        for (int col = 0; col < n; ++col) {
            const char ch = rows[row][col];
            assert(ch == 'X' || ch == 'O' || ch == '.');
            if (ch == 'X') b.setStone(row * n + col, Color::Black);
            if (ch == 'O') b.setStone(row * n + col, Color::White);
        }
    }
    b.setToMove(toMove);
    return b;
}

std::string actionToString(int action, int n) {
    assert(n >= kMinBoardSize && n <= kMaxBoardSize);
    assert(action >= 0 && action <= n * n);
    if (action == n * n) return "pass";
    const int row = action / n, col = action % n;
    return std::string(1, kColumnLetters[col]) + std::to_string(n - row);
}

int parseAction(const std::string& text, int n) {
    assert(n >= kMinBoardSize && n <= kMaxBoardSize);
    std::string t;
    for (char ch : text) t += static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    if (t == "PASS") return n * n;
    // A letter followed by one or two digits.
    if (t.size() < 2 || t.size() > 3) return -1;
    int col = -1;
    for (int c = 0; c < n; ++c) {
        if (kColumnLetters[c] == t[0]) col = c;
    }
    if (col < 0) return -1;  // 'I', a letter beyond the board, or not a letter
    int label = 0;
    for (size_t i = 1; i < t.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(t[i]))) return -1;
        label = label * 10 + (t[i] - '0');
    }
    if (label < 1 || label > n) return -1;
    const int row = n - label;
    return row * n + col;
}

} // namespace azgo
