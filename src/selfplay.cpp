#include "azgo/selfplay.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include "azgo/game.hpp"

namespace azgo {

float GameRecord::zAt(int ply) const {
    // The value target is always from the side to move's point of view.
    return toMoveAt(ply) == Color::Black ? resultForBlack : -resultForBlack;
}

GameRecord playSelfPlayGame(Evaluator& evaluator, const SelfPlayConfig& cfg, Rng& rng) {
    const int n = evaluator.boardSize();
    const int area = n * n;
    Game game(n, cfg.komi, cfg.maxMoves > 0 ? cfg.maxMoves : defaultMaxMoves(n));
    MctsConfig mcfg = cfg.mcts;
    mcfg.addNoise = true;   // self-play always explores at the root
    Mcts mcts(mcfg, evaluator);

    GameRecord rec;
    rec.boardSize = n;
    rec.komi = cfg.komi;
    while (!game.isTerminal()) {
        mcts.search(game, rng);

        // Position before the move ...
        const size_t offset = rec.stones.size();
        rec.stones.resize(offset + area);
        game.board().exportStones(rec.stones.data() + offset);
        // ... and the search's visit counts, the policy target pi.
        const std::vector<int> visits = mcts.rootVisits();
        std::vector<std::pair<int16_t, uint16_t>> sparse;
        for (int a = 0; a < static_cast<int>(visits.size()); ++a) {
            if (visits[a] == 0) continue;
            // Counts beyond 65535 (only with huge simulation budgets) are
            // clamped; pi is a ratio, so this barely changes it.
            const int count = std::min(visits[a], static_cast<int>(std::numeric_limits<uint16_t>::max()));
            sparse.emplace_back(static_cast<int16_t>(a), static_cast<uint16_t>(count));
        }
        rec.visits.push_back(std::move(sparse));

        // Sample in the opening for variety, then play the best move.
        const float temperature = game.moveCount() < cfg.temperatureMoves ? 1.0f : 0.0f;
        const int action = mcts.selectAction(temperature, rng);
        rec.moves.push_back(action);
        game.play(action);
        mcts.advance(action);   // reuse the subtree under the move just played
    }
    rec.resultForBlack = game.resultForBlack();
    // Who owns each point at the end: the per-point target of the ownership head.
    rec.finalOwnership.resize(static_cast<size_t>(area));
    game.board().ownership(rec.finalOwnership.data());
    return rec;
}

std::vector<int8_t> replayFinalOwnership(const GameRecord& record) {
    const int n = record.boardSize, area = n * n;
    // The record's own length as the move cap: the cap cannot end the replay
    // early, while two passes before the last move (impossible in a record
    // self-play wrote) make the next move illegal.
    Game game(n, record.komi, std::max(1, record.positions()));
    std::vector<uint8_t> stones(static_cast<size_t>(area));
    for (int ply = 0; ply < record.positions(); ++ply) {
        game.board().exportStones(stones.data());
        if (std::memcmp(stones.data(), record.stonesAt(ply), static_cast<size_t>(area)) != 0)
            throw std::runtime_error("position " + std::to_string(ply) +
                                     " differs from the replayed one");
        const int move = record.moves[static_cast<size_t>(ply)];
        if (!game.isLegal(move))
            throw std::runtime_error("move " + std::to_string(ply + 1) + " (" +
                                     actionToString(move, n) + ") is illegal");
        game.play(move);
    }
    std::vector<int8_t> owner(static_cast<size_t>(area));
    game.board().ownership(owner.data());
    return owner;
}

} // namespace azgo
