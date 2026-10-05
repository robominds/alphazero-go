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
// from the side to move's view. AlphaGo Zero resigns when both are below,
// which is when the larger one is.
float resignValue(float rootValue, float bestChildValue);
float resignValue(const Mcts& mcts);   // resignValue(mcts.rootValue(), mcts.bestChildValue())

// One side's record lows of resignValue over a game: (ply, value) each time
// the value reached a new minimum, so plies increase and values fall. The
// first entry below a threshold v is the ply where that side would have
// resigned at v.
using ResignLows = std::vector<std::pair<int, float>>;

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
    // The side to move after the last move resigned: the game ends before
    // its position, so the last position is the opponent's and the last
    // board is no final position. The replay buffer stores this flag.
    bool resigned = false;
    // For this generation's statistics only (not stored): resignation was
    // on, but not for this game; and per side ([0] Black, [1] White) its
    // record lows from ply 1 on, kept whenever resignation is on.
    bool resignDisabled = false;
    ResignLows resignLows[2];

    int positions() const { return static_cast<int>(moves.size()); }
    const uint8_t* stonesAt(int ply) const { return stones.data() + static_cast<size_t>(ply) * boardSize * boardSize; }
    Color toMoveAt(int ply) const { return ply % 2 == 0 ? Color::Black : Color::White; }
    float zAt(int ply) const;      // resultForBlack from the side to move at ply
};

// Plays one self-play game from the empty board with root noise and tree reuse.
GameRecord playSelfPlayGame(Evaluator& evaluator, const SelfPlayConfig& cfg, Rng& rng);

// The final board's ownership (Black's view), found by replaying the record's
// moves from the empty board with Game. Records saved before finalOwnership
// existed get it this way. Throws std::runtime_error if a stored position
// differs from the replayed one or a move is illegal.
std::vector<int8_t> replayFinalOwnership(const GameRecord& record);

// What calibration keeps of one resignation-disabled game: the record lows
// of its eventual winner and of the loser (both empty for a draw).
struct ResignTrace {
    ResignLows winnerLows, loserLows;
    // AlphaGo Zero's false positive, "a game that could have been won" had
    // resignation been on at `threshold`: the winner's value drops strictly
    // below it before the loser's does (or the loser's never does). Exact,
    // not an estimate: the resignation check draws nothing from the Rng, so
    // the game with resignation on is this game up to the first such drop.
    bool falsePositiveAt(float threshold) const;
};
ResignTrace resignTrace(const GameRecord& record);

// Choosing v_resign from resignation-disabled games, as AlphaGo Zero did:
// the largest threshold on the grid -1.00, -0.99, ..., 0.00 at which at most
// maxFalsePositive of the last `window` games are false positives. -1
// always qualifies, since no value lies below it. With fewer than minGames
// games: initialThreshold.
struct ResignCalibration {
    int window = 500;
    int minGames = 20;
    float maxFalsePositive = 0.05f;
    float initialThreshold = -0.9f;
};
// games: the disabled games' traces, oldest first.
float calibrateResignThreshold(const std::vector<ResignTrace>& games, const ResignCalibration& c);
// The fraction of the last `window` games that are false positives at
// `threshold`; 0 when there are none.
double resignFalsePositiveRate(const std::vector<ResignTrace>& games, int window, float threshold);

} // namespace azgo
