// A game of Go in progress.
//
// A single Board knows the local rules, but two rules need the game's past.
// Positional superko: a move may not recreate any whole-board position that
// already occurred in this game; this is what stops ko fights from looping
// forever. It is enforced by remembering the Zobrist hash of every position.
// The end of the game: two consecutive passes (or a move cap, 2 * n * n by
// default, so self-play always finishes); the final position is then scored
// by Tromp-Taylor area minus komi, White's compensation for moving second.
#pragma once
#include <unordered_set>
#include <vector>
#include "azgo/board.hpp"

namespace azgo {

// A game in progress: the sequence of positions, for history planes and
// positional superko, plus the pass count and the move cap.
class Game {
public:
    Game(int n, float komi, int maxMoves);
    Game(const Board& start, float komi, int maxMoves);

    int size() const { return boards_.front().size(); }
    float komi() const { return komi_; }
    int maxMoves() const { return maxMoves_; }

    const Board& board() const { return boards_.back(); }
    const Board& boardAt(int ply) const;          // 0 = start; asserts ply <= moveCount()
    int moveCount() const { return static_cast<int>(moves_.size()); }
    const std::vector<int>& moves() const { return moves_; }
    int consecutivePasses() const { return passes_; }

    // Board::isLegal plus positional superko: a non-pass move may not
    // recreate any earlier position of this game. Always false once terminal.
    bool isLegal(int action) const;
    std::vector<int> legalActions() const;        // ascending; always contains pass unless terminal
    void play(int action);                        // asserts isLegal(action)
    bool seen(uint64_t positionHash) const;       // position occurred in this game

    // Two consecutive passes, or moveCount() >= maxMoves().
    bool isTerminal() const;
    float scoreForBlack() const;                  // board().areaScore() - komi
    float resultForBlack() const;                 // +1 Black wins, -1 White wins, 0 exact tie

private:
    std::vector<Board> boards_;                   // boards_[k] = position after k moves
    std::vector<int> moves_;
    std::unordered_set<uint64_t> seen_;
    int passes_ = 0;
    float komi_;
    int maxMoves_;
};

// 2 * n * n, the default move cap.
int defaultMaxMoves(int n);

} // namespace azgo
