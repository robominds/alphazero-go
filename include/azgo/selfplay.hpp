// Self-play: the network plays both sides against itself, choosing every
// move by MCTS. Each position is stored with the search's visit counts
// (the improved policy target pi); once the game ends, every position also
// gets the final result z seen from its own side to move. Root Dirichlet
// noise and temperature-1 sampling in the opening keep the games varied.
#pragma once
#include <cstdint>
#include <utility>
#include <vector>
#include "azgo/board.hpp"
#include "azgo/evaluator.hpp"
#include "azgo/mcts.hpp"
#include "azgo/rng.hpp"

namespace azgo {

struct SelfPlayConfig {
    MctsConfig mcts;             // addNoise is forced on by playSelfPlayGame
    int temperatureMoves = 20;   // plies sampled with temperature 1, then argmax
    float komi = 7.5f;
    int maxMoves = 0;            // 0 means defaultMaxMoves(n)
};

// A finished game in compact form. Position `ply` is the board BEFORE
// moves[ply] was played; Black moves at even plies (no handicap).
struct GameRecord {
    int boardSize = 0;
    float komi = 0;
    std::vector<int> moves;
    std::vector<uint8_t> stones;   // positions() * n*n Color bytes
    std::vector<std::vector<std::pair<int16_t, uint16_t>>> visits;  // per ply: (action, count), count > 0
    float resultForBlack = 0;      // +1, -1, or 0

    int positions() const { return static_cast<int>(moves.size()); }
    const uint8_t* stonesAt(int ply) const { return stones.data() + static_cast<size_t>(ply) * boardSize * boardSize; }
    Color toMoveAt(int ply) const { return ply % 2 == 0 ? Color::Black : Color::White; }
    float zAt(int ply) const;      // resultForBlack from the side to move at ply
};

// Plays one self-play game from the empty board with root noise and tree reuse.
GameRecord playSelfPlayGame(Evaluator& evaluator, const SelfPlayConfig& cfg, Rng& rng);

} // namespace azgo
