#include "azgo/arena.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include "azgo/backend.hpp"
#include "azgo/evaluator.hpp"
#include "azgo/game.hpp"
#include "azgo/mcts.hpp"
#include "azgo/parallel.hpp"
#include "azgo/rng.hpp"

namespace azgo {

namespace {

// One player's per-thread resources and search settings. A network gets its
// own backend and workspace; a bot needs only a UniformEvaluator. A player's
// c_puct and batch overrides (0 = the config's) apply to either kind.
struct Seat {
    std::unique_ptr<Backend> backend;
    std::unique_ptr<Evaluator> evaluator;
    MctsConfig mcts;
    Seat(const ArenaPlayer& player, const NetConfig& shape, const ArenaConfig& cfg,
         const std::string& backendName) {
        mcts.batchSize = player.mctsBatch > 0 ? player.mctsBatch : cfg.mctsBatch;
        mcts.cPuct = player.cPuct > 0 ? player.cPuct : cfg.cPuct;
        mcts.addNoise = false;   // measuring strength, not exploring
        if (player.net) {
            backend = makeBackend(backendName, 1);
            evaluator.reset(new LocalEvaluator(*player.net, *backend, mcts.batchSize));
            mcts.simulations = player.simulations > 0 ? player.simulations : cfg.simulations;
            mcts.randomSymmetry = cfg.randomSymmetry || player.randomSymmetry;   // a bot gains nothing
        } else {
            evaluator.reset(new UniformEvaluator(shape.boardSize, shape.history, 0.0f, mcts.batchSize));
            mcts.simulations = player.simulations;
        }
    }
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
    Mcts searchA(ctx.a.mcts, *ctx.a.evaluator), searchB(ctx.b.mcts, *ctx.b.evaluator);

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

// Information carried by one game whose expected score is p is proportional
// to p(1 - p): 0.25 at p = 0.5, 0.21 at 0.7, under 0.01 at 0.99. 0.7 keeps
// most of the information while still letting the new network show progress
// against an opponent it should beat.
constexpr double kTargetScore = 0.7;

// Bradley-Terry on the Elo scale: the expected score of `rating` against `opponent`.
double expectedScore(double rating, double opponent) {
    return 1.0 / (1.0 + std::pow(10.0, (opponent - rating) / 400.0));
}

constexpr int kMaxSimulations = 100000;   // the same bound as the arena app's --b-bot
// The train config's ranges for c_puct and mcts_batch (src/config.cpp).
constexpr float kMaxCPuct = 100.0f;
constexpr int kMaxMctsBatch = 256;

// "" if c is a valid c_puct, (0, 100], or 0 when `inherit` allows it;
// otherwise the reason. The negated test also refuses NaN.
std::string checkCPuct(float c, bool inherit) {
    if ((inherit && c == 0.0f) || (c > 0.0f && c <= kMaxCPuct)) return "";
    return std::string("c_puct must be in (0, 100]") + (inherit ? " or 0" : "") + ", got " +
           std::to_string(c);
}

std::string checkMctsBatch(int b, bool inherit) {
    if ((inherit && b == 0) || (b >= 1 && b <= kMaxMctsBatch)) return "";
    return std::string("mcts batch must be in [1, 256]") + (inherit ? " or 0" : "") + ", got " +
           std::to_string(b);
}

} // namespace

MatchResult playMatch(const ArenaPlayer& a, const ArenaPlayer& b, const ArenaConfig& cfg,
                      uint64_t seed, int workers, const std::string& backendName) {
    if (!a.net && !b.net) throw std::invalid_argument("playMatch: two bots (one player must be a network)");
    if (a.net && b.net && a.net->config() != b.net->config())
        throw std::invalid_argument("playMatch: networks have different architectures");
    const char* const names[] = {"A", "B"};
    int which = 0;
    for (const ArenaPlayer* p : {&a, &b}) {
        const std::string who = std::string("playMatch: player ") + names[which++];
        if (!p->net && p->simulations < 1)
            throw std::invalid_argument(who + " is a bot and needs at least 1 simulation, got " +
                                        std::to_string(p->simulations));
        if (p->simulations < 0 || p->simulations > kMaxSimulations)
            throw std::invalid_argument(who + " simulations must be in [0, " +
                                        std::to_string(kMaxSimulations) + "], got " +
                                        std::to_string(p->simulations));
        for (const std::string& reason : {checkCPuct(p->cPuct, true), checkMctsBatch(p->mctsBatch, true)})
            if (!reason.empty()) throw std::invalid_argument(who + " " + reason);
    }
    for (const std::string& reason : {checkCPuct(cfg.cPuct, false), checkMctsBatch(cfg.mctsBatch, false)})
        if (!reason.empty()) throw std::invalid_argument("playMatch: config " + reason);
    if (cfg.games < 2 || cfg.games % 2 != 0)
        throw std::invalid_argument("playMatch: games must be even and >= 2, got " +
                                    std::to_string(cfg.games));

    // A bot has no network, so the board size and history come from the
    // player that has one.
    const NetConfig shape = (a.net ? a.net : b.net)->config();
    int threads = std::min(resolveWorkers(workers), cfg.games);
    std::vector<std::unique_ptr<WorkerContext>> contexts;
    for (int w = 0; w < threads; ++w)
        contexts.emplace_back(new WorkerContext{Seat(a, shape, cfg, backendName),
                                                Seat(b, shape, cfg, backendName)});

    // Results are stored by game index, never by completion order, so the
    // match is identical for any number of workers.
    std::vector<GameOutcome> outcomes(cfg.games);
    parallelFor(cfg.games, threads, [&](int index, int worker) {
        outcomes[index] = playArenaGame(*contexts[worker], cfg, shape.boardSize, seed, index);
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

MatchResult playMatch(const Network& a, const Network& b, const ArenaConfig& cfg, uint64_t seed,
                      int workers, const std::string& backendName) {
    return playMatch(ArenaPlayer::network(a), ArenaPlayer::network(b), cfg, seed, workers,
                     backendName);
}

int strongestReference(const std::map<int, double>& ratings, const std::vector<int>& references) {
    int best = 0;
    double bestRating = 0;
    bool found = false;
    for (int r : references) {
        auto it = ratings.find(r);
        if (it == ratings.end()) continue;   // unrated: nothing known about it yet
        if (!found || it->second > bestRating || (it->second == bestRating && r > best)) {
            best = r;
            bestRating = it->second;
            found = true;
        }
    }
    if (!found) throw std::invalid_argument("strongestReference: no reference is rated");
    return best;
}

std::vector<int> selectOpponents(int generation, const std::map<int, double>& ratings,
                                 const std::vector<int>& references, int maxExtra,
                                 std::optional<int> bestNetwork) {
    if (generation < 1)
        throw std::invalid_argument("selectOpponents: generation must be >= 1, got " +
                                    std::to_string(generation));
    std::vector<int> out{generation - 1};   // the progress signal
    auto chosen = [&](int p) { return std::find(out.begin(), out.end(), p) != out.end(); };
    const int strongest = strongestReference(ratings, references);
    if (!chosen(strongest)) out.push_back(strongest);   // the promotion check

    // The new network has no rating yet; its predecessor's is the best guess.
    auto prev = ratings.find(generation - 1);
    const double estimate = prev == ratings.end() ? 0.0 : prev->second;
    std::vector<std::pair<double, int>> candidates;   // (|E - 0.7|, player)
    for (int r : references) {
        auto it = ratings.find(r);
        if (it == ratings.end() || chosen(r)) continue;
        candidates.push_back({std::fabs(expectedScore(estimate, it->second) - kTargetScore), r});
    }
    std::sort(candidates.begin(), candidates.end());   // by distance, then by smaller ID
    for (size_t i = 0; i < candidates.size() && static_cast<int>(i) < maxExtra; ++i)
        out.push_back(candidates[i].second);

    if (bestNetwork && !chosen(*bestNetwork)) out.push_back(*bestNetwork);   // gating
    return out;
}

bool passesGate(double score, int games, double threshold) {
    // gating_threshold is a float key: 0.55f is 0.550000011920929, so a plain
    // comparison would refuse 11 of 20. The 1e-6 margin, the same one promotion
    // uses, accepts exactly what the user wrote.
    return games > 0 && score >= (threshold - 1e-6) * games;
}

double matchScore(float resultForBlack, bool aIsBlack) {
    if (resultForBlack == 0.0f) return 0.5;   // exact tie (integer komi)
    bool blackWon = resultForBlack > 0.0f;
    return blackWon == aIsBlack ? 1.0 : 0.0;
}

} // namespace azgo
