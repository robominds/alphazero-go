// The rules of Go for a single position.
//
// Stones of one color that touch horizontally or vertically form a group; the
// empty points next to a group are its liberties. A move that leaves opponent
// groups without liberties captures them; a move that leaves its own group
// without liberties (suicide) is illegal here. The board is stored padded
// with a ring of Off cells so neighbor lookups never need bounds checks, and
// every position carries a Zobrist hash (XOR of one random key per stone) that
// Game uses to detect repeated positions (superko). Scoring is Tromp-Taylor
// area scoring: your stones plus empty regions that touch only your stones.
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace azgo {

constexpr int kMinBoardSize = 5;
constexpr int kMaxBoardSize = 19;
constexpr int kStride = kMaxBoardSize + 2;   // padded row length: one sentinel column each side
constexpr int kCells = kStride * kStride;    // padded array size (441)

enum class Color : uint8_t { Empty = 0, Black = 1, White = 2, Off = 3 };

Color opponent(Color c);   // Black <-> White; asserts c is Black or White

class Board {
public:
    explicit Board(int n);  // empty board, Black to move; asserts kMinBoardSize <= n <= kMaxBoardSize

    int size() const { return n_; }
    int area() const { return n_ * n_; }
    int passAction() const { return n_ * n_; }
    int numActions() const { return n_ * n_ + 1; }
    Color toMove() const { return toMove_; }

    Color at(int point) const;          // asserts 0 <= point < area()
    int stoneCount(Color c) const;
    int libertiesAt(int point) const;   // liberties of the group containing point; 0 if empty

    // Local legality: pass is always legal; a point must be empty and the move
    // must not be suicide (after removing captured opponent groups, the new
    // stone's group must have a liberty). Superko is NOT checked here — it
    // needs history, see Game::isLegal and Mcts.
    bool isLegal(int action) const;
    // Plays a locally legal action (asserted), removes captured groups, flips
    // toMove. Pass only flips toMove. Returns the number of stones captured.
    int play(int action);
    // Hash of the position play(action) would produce. O(1) when the move
    // captures nothing; otherwise copies the board and plays.
    uint64_t hashAfter(int action) const;

    uint64_t hash() const { return hash_; }  // Zobrist over stones only (positional superko)
    uint64_t computeHashFromScratch() const; // test oracle for the incremental hash

    // Tromp-Taylor area score: (Black stones + empty regions reaching only Black)
    // minus (White stones + empty regions reaching only White). No komi.
    int areaScore() const;

    void exportStones(uint8_t* out) const;   // area() bytes of Color values, row-major
    std::string toString() const;            // ASCII diagram with GTP coordinates

    // Setup helpers for tests and tools: no capture logic; hash stays valid.
    void setStone(int point, Color c);       // c is Empty, Black or White
    void setToMove(Color c);

    bool operator==(const Board& o) const;   // size, stones and side to move

private:
    int pad(int point) const;                // point -> padded index
    int n_;
    Color toMove_;
    std::array<Color, kCells> cells_;
    uint64_t hash_;
    // Additional private helpers are implementation-defined (Task 2).
    // Liberties of the group containing the stone at padded index p. Counting
    // stops once `limit` liberties are found, which is all isLegal needs
    // ("none", "exactly one", or "at least two").
    int countLiberties(int p, int limit) const;
    // Removes the group containing the stone at padded index p (updating the
    // hash) and returns the number of stones removed.
    int removeGroup(int p);
};

// Rows like "X.O.." ('X' black, 'O' white, '.' empty), row 0 first (top).
// Asserts rows form an n x n square with n in range.
Board boardFromRows(const std::vector<std::string>& rows, Color toMove);

// GTP coordinates. parseAction is case-insensitive and returns -1 for
// malformed or off-board input.
std::string actionToString(int action, int n);
int parseAction(const std::string& text, int n);

} // namespace azgo
