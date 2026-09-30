#include "azgo/game.hpp"
#include <cassert>

namespace azgo {

Game::Game(int n, float komi, int maxMoves) : Game(Board(n), komi, maxMoves) {}

Game::Game(const Board& start, float komi, int maxMoves) : komi_(komi), maxMoves_(maxMoves) {
    assert(maxMoves > 0);
    boards_.push_back(start);
    seen_.insert(start.hash());
}

const Board& Game::boardAt(int ply) const {
    assert(ply >= 0 && ply <= moveCount());
    return boards_[static_cast<size_t>(ply)];
}

bool Game::isLegal(int action) const {
    if (isTerminal()) return false;
    const Board& b = board();
    // Passing never changes the stones, so it can never break superko.
    if (action == b.passAction()) return true;
    if (!b.isLegal(action)) return false;
    // Positional superko: compare the hash of the resulting stones against
    // every position of this game so far.
    return seen_.count(b.hashAfter(action)) == 0;
}

std::vector<int> Game::legalActions() const {
    std::vector<int> actions;
    if (isTerminal()) return actions;
    for (int a = 0; a < board().numActions(); ++a) {
        if (isLegal(a)) actions.push_back(a);
    }
    return actions;
}

void Game::play(int action) {
    assert(isLegal(action));
    Board next = board();
    next.play(action);
    passes_ = (action == next.passAction()) ? passes_ + 1 : 0;
    moves_.push_back(action);
    seen_.insert(next.hash());  // a pass re-inserts an existing hash: no-op
    boards_.push_back(next);
}

bool Game::seen(uint64_t positionHash) const {
    return seen_.count(positionHash) > 0;
}

bool Game::isTerminal() const {
    return passes_ >= 2 || moveCount() >= maxMoves_;
}

float Game::scoreForBlack() const {
    return static_cast<float>(board().areaScore()) - komi_;
}

float Game::resultForBlack() const {
    const float s = scoreForBlack();
    // An integer komi can produce an exact tie; it is scored as a draw.
    if (s > 0.0f) return 1.0f;
    if (s < 0.0f) return -1.0f;
    return 0.0f;
}

int defaultMaxMoves(int n) {
    return 2 * n * n;
}

} // namespace azgo
