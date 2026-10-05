// Self-play: the network plays both sides against itself, choosing every
// move by MCTS. Each position is stored with the search's visit counts
// (the improved policy target pi); once the game ends, every position also
// gets the final result z seen from its own side to move, and the record
// keeps who owns each point of the final board (the ownership target). Root
// Dirichlet noise and temperature-1 sampling in the opening keep the games
// varied. Optionally a hopeless side resigns, as in AlphaGo Zero, with a
// threshold calibrated on games where resignation was switched off.
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
    // Resignation (AlphaGo Zero, Methods "Self-play"). On: each game first
    // draws once from its Rng whether resignation is disabled for it (with
    // probability resignDisabledFraction); otherwise, from ply 1 on, the side
    // to move resigns when resignValue < resignThreshold. Off: nothing extra
    // is drawn, so the games are exactly those played without the feature.
    bool resignation = false;
    float resignThreshold = -0.9f;        // v_resign
    float resignDisabledFraction = 0.1f;  // games played out to count false positives
};

// What resignation compares with v_resign after each search: the larger of
// the root value and the most visited child's Q (Mcts::bestChildValue), both
// from the side to move's view. AlphaGo Zero resigns when both are below.
float resignValue(const Mcts& mcts);

// A finished game in compact form. Position `ply` is the board BEFORE
// moves[ply] was played; Black moves at even plies (no handicap).
struct GameRecord {
    int boardSize = 0;
    float komi = 0;
    std::vector<int> moves;
    std::vector<uint8_t> stones;   // positions() * n*n Color bytes
    std::vector<std::vector<std::pair<int16_t, uint16_t>>> visits;  // per ply: (action, count), count > 0
    float resultForBlack = 0;      // +1, -1, or 0
    std::vector<int8_t> finalOwnership;   // n*n, Black's view: Board::ownership of the final board
    // Resignation bookkeeping, for this generation's statistics only (the
    // replay buffer does not store it). A resigned game ends before the
    // resigning side's position: its last position is the opponent's.
    bool resigned = false;         // the side to move after the last move resigned
    bool resignDisabled = false;   // resignation was on, but not for this game
    // Per side ([0] Black, [1] White): the lowest resignValue it saw from
    // ply 1 on; +1 (no value is higher) while it saw none. Kept whenever
    // resignation is on.
    float minResignValue[2] = {1.0f, 1.0f};

    int positions() const { return static_cast<int>(moves.size()); }
    const uint8_t* stonesAt(int ply) const { return stones.data() + static_cast<size_t>(ply) * boardSize * boardSize; }
    Color toMoveAt(int ply) const { return ply % 2 == 0 ? Color::Black : Color::White; }
    float zAt(int ply) const;      // resultForBlack from the side to move at ply
    // The winner's minResignValue: a resignation-disabled game is a false
    // positive at threshold v when this is below v (the winner would have
    // resigned). A draw has no winner and is never one: +1.
    float winnerMinResignValue() const;
};

// Plays one self-play game from the empty board with root noise and tree reuse.
GameRecord playSelfPlayGame(Evaluator& evaluator, const SelfPlayConfig& cfg, Rng& rng);

// The final board's ownership (Black's view), found by replaying the record's
// moves from the empty board with Game. Records saved before finalOwnership
// existed get it this way. Throws std::runtime_error if a stored position
// differs from the replayed one or a move is illegal.
std::vector<int8_t> replayFinalOwnership(const GameRecord& record);

// Choosing v_resign from resignation-disabled games, as AlphaGo Zero did:
// the largest threshold on the grid -1.00, -0.99, ..., 0.00 at which at most
// maxFalsePositive of the last `window` games are false positives (their
// winner's minimum lies strictly below it; draws never are). -1 always
// qualifies, since no value lies below it. With fewer than minGames games:
// initialThreshold.
struct ResignCalibration {
    int window = 500;
    int minGames = 20;
    float maxFalsePositive = 0.05f;
    float initialThreshold = -0.9f;
};
// winnerMinimums: GameRecord::winnerMinResignValue of each disabled game, oldest first.
float calibrateResignThreshold(const std::vector<float>& winnerMinimums, const ResignCalibration& c);
// The fraction of the last `window` games that are false positives at
// `threshold`; 0 when there are none.
double resignFalsePositiveRate(const std::vector<float>& winnerMinimums, int window, float threshold);

} // namespace azgo
