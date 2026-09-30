#include "azgo/arena.hpp"
#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>
#include "azgo/backend.hpp"
#include "azgo/evaluator.hpp"
#include "azgo/game.hpp"
#include "azgo/mcts.hpp"
#include "azgo/parallel.hpp"
#include "azgo/rng.hpp"

namespace azgo {

namespace {

// One network's per-thread resources: its own backend and workspace.
struct Seat {
    std::unique_ptr<Backend> backend;
    std::unique_ptr<LocalEvaluator> evaluator;
    Seat(const Network& net, const std::string& backendName, int maxBatch)
        : backend(makeBackend(backendName, 1)),
          evaluator(new LocalEvaluator(net, *backend, maxBatch)) {}
};

struct WorkerContext {
    Seat a, b;
};

struct GameOutcome {
    double scoreA = 0;
    std::vector<int> moves;
};

GameOutcome playArenaGame(WorkerContext& ctx, const ArenaConfig& cfg, int n, uint64_t seed,
                          int index) {
    Rng rng(deriveSeed(seed, static_cast<uint64_t>(index)));
    bool aIsBlack = index % 2 == 0;   // colors alternate, so A is Black in half the games
    Game game(n, cfg.komi, cfg.maxMoves > 0 ? cfg.maxMoves : defaultMaxMoves(n));

    MctsConfig mc;
    mc.simulations = cfg.simulations;
    mc.batchSize = cfg.mctsBatch;
    mc.cPuct = cfg.cPuct;
    mc.addNoise = false;   // measuring strength, not exploring
    Mcts searchA(mc, *ctx.a.evaluator), searchB(mc, *ctx.b.evaluator);

    while (!game.isTerminal()) {
        bool blackToMove = game.board().toMove() == Color::Black;
        Mcts& mover = blackToMove == aIsBlack ? searchA : searchB;
        mover.search(game, rng);
        // A few sampled opening moves make the games differ from each other.
        float temperature = game.moveCount() < cfg.temperatureMoves ? 1.0f : 0.0f;
        int action = mover.selectAction(temperature, rng);
        game.play(action);
        // Both trees follow the game, so each reuses its subtree next turn.
        // A tree that has never searched has nothing to re-root.
        if (searchA.nodeCount() > 0) searchA.advance(action);
        if (searchB.nodeCount() > 0) searchB.advance(action);
    }
    return {matchScore(game.resultForBlack(), aIsBlack), game.moves()};
}

} // namespace

MatchResult playMatch(const Network& a, const Network& b, const ArenaConfig& cfg, uint64_t seed,
                      int workers, const std::string& backendName) {
    if (a.config() != b.config())
        throw std::invalid_argument("playMatch: networks have different architectures");
    if (cfg.games < 2 || cfg.games % 2 != 0)
        throw std::invalid_argument("playMatch: games must be even and >= 2, got " +
                                    std::to_string(cfg.games));

    int threads = std::min(resolveWorkers(workers), cfg.games);
    std::vector<std::unique_ptr<WorkerContext>> contexts;
    for (int w = 0; w < threads; ++w)
        contexts.emplace_back(new WorkerContext{Seat(a, backendName, cfg.mctsBatch),
                                                Seat(b, backendName, cfg.mctsBatch)});

    // Results are stored by game index, never by completion order, so the
    // match is identical for any number of workers.
    std::vector<GameOutcome> outcomes(cfg.games);
    int n = a.config().boardSize;
    parallelFor(cfg.games, threads, [&](int index, int worker) {
        outcomes[index] = playArenaGame(*contexts[worker], cfg, n, seed, index);
    });

    MatchResult result;
    result.games = cfg.games;
    std::set<std::vector<int>> distinct;
    for (const GameOutcome& g : outcomes) {
        result.scoreA += g.scoreA;
        if (g.scoreA == 1.0) ++result.winsA;
        else if (g.scoreA == 0.0) ++result.winsB;
        else ++result.draws;
        distinct.insert(g.moves);
    }
    result.distinctGames = static_cast<int>(distinct.size());
    return result;
}

std::vector<int> ladderOpponents(int generation, int maxOpponents) {
    std::vector<int> out;
    if (generation <= 0) return out;
    // g-1, g-2, g-4, g-8, ...: dense near the present, sparse into the past.
    for (int d = 1; generation - d >= 1 && static_cast<int>(out.size()) < maxOpponents; d *= 2)
        out.push_back(generation - d);
    out.push_back(0);   // the anchor: every rating is relative to generation 0
    return out;
}

bool passesGate(double score, int games, double threshold) {
    return games > 0 && score / games >= threshold;
}

double matchScore(float resultForBlack, bool aIsBlack) {
    if (resultForBlack == 0.0f) return 0.5;   // exact tie (integer komi)
    bool blackWon = resultForBlack > 0.0f;
    return blackWon == aIsBlack ? 1.0 : 0.0;
}

} // namespace azgo
