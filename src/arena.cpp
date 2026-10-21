#include "azgo/arena.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include "azgo/backend.hpp"
#include "azgo/evaluator.hpp"
#include "azgo/game.hpp"
#include "azgo/inference_hub.hpp"
#include "azgo/mcts.hpp"
#include "azgo/parallel.hpp"
#include "azgo/rng.hpp"

namespace azgo {

namespace {

// One player's seat in one game: its search settings and the evaluator it
// searches with, a network player's from the round's evaluators (one per
// network per worker), a bot's its own UniformEvaluator sized for its batch.
struct Seat {
    MctsConfig mcts;
    Evaluator* evaluator = nullptr;
    std::unique_ptr<Evaluator> own;
};

ArenaGameResult playArenaGame(Seat& a, Seat& b, const ArenaConfig& cfg, int n, uint64_t seed, int index) {
    Rng rng(deriveSeed(seed, static_cast<uint64_t>(index)));
    bool aIsBlack = index % 2 == 0;   // colors alternate, so A is Black in half the games
    Game game(n, cfg.komi, cfg.maxMoves > 0 ? cfg.maxMoves : defaultMaxMoves(n));
    Mcts searchA(a.mcts, *a.evaluator), searchB(b.mcts, *b.evaluator);

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

// The evaluators of a set of matches: for each distinct network, one per
// worker. With metal, WorkerEvaluators gives each network its own hub, which
// every worker's searches with that network share. A hub alone on the GPU
// keeps two passes in flight, so one computes while the next fills; when
// several hubs share it, the others' passes keep it busy, and one pass in
// flight per hub lets each pass take everything that waits: fewer, larger
// passes (docs/results/bench.txt). Which positions share a pass changes no
// result. cuda, the other GPU backend, works the same way. With naive and blas, each
// worker has one single-threaded backend and one workspace, sized for the
// largest request, and a LocalEvaluator per network on that workspace: a
// worker plays one game at a time and only the side to move searches, so it
// never evaluates for two networks at once. That is less memory than a
// workspace per network per worker, and the outputs are the same, because a
// request is still evaluated alone, exactly as the search asked for it.
class RoundEvaluators {
public:
    RoundEvaluators(const std::vector<const Network*>& nets, const std::vector<int>& maxRequest,
                    const std::string& backend, int workers, int inferenceBatch) {
        if (nets.empty()) return;
        if (backend == "metal" || backend == "cuda") {
            const int slots = nets.size() > 1 ? 1 : kInferenceSlots;
            for (size_t k = 0; k < nets.size(); ++k)
                hubs_.push_back(std::make_unique<WorkerEvaluators>(*nets[k], backend, workers, maxRequest[k],
                                                                   inferenceBatch, slots));
            return;
        }
        const int largest = *std::max_element(maxRequest.begin(), maxRequest.end());
        local_.resize(nets.size());
        for (int w = 0; w < workers; ++w) {
            backends_.push_back(makeBackend(backend, 1));
            workspaces_.push_back(std::make_unique<Workspace>(nets[0]->config(), largest, *backends_.back()));
            for (size_t k = 0; k < nets.size(); ++k)
                local_[k].push_back(std::make_unique<LocalEvaluator>(*nets[k], *workspaces_.back()));
        }
    }
    ~RoundEvaluators() {
        local_.clear();   // before the workspaces and backends they use
    }
    Evaluator& get(size_t network, int worker) {
        if (!hubs_.empty()) return hubs_[network]->forWorker(worker);
        return *local_[network][static_cast<size_t>(worker)];
    }
    // Positions evaluated so far, every network and worker together. Read
    // it once the workers have stopped.
    long long evaluations() const {
        long long total = 0;
        for (const auto& hub : hubs_) total += hub->evaluations();
        for (const auto& perWorker : local_)
            for (const auto& e : perWorker) total += e->evaluations();
        return total;
    }

private:
    std::vector<std::unique_ptr<WorkerEvaluators>> hubs_;   // metal, cuda: one per network
    std::vector<std::unique_ptr<Backend>> backends_;        // naive, blas: one per worker
    std::vector<std::unique_ptr<Workspace>> workspaces_;    // one per worker
    std::vector<std::vector<std::unique_ptr<LocalEvaluator>>> local_;   // [network][worker]
};

std::atomic<bool> gOneMatchAtATime{false};   // arena_probe: tests only

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

MctsConfig arenaSearchConfig(const ArenaPlayer& player, const ArenaConfig& cfg) {
    MctsConfig mcts;
    mcts.batchSize = player.mctsBatch > 0 ? player.mctsBatch : cfg.mctsBatch;
    mcts.cPuct = player.cPuct > 0 ? player.cPuct : cfg.cPuct;
    mcts.addNoise = false;   // measuring strength, not exploring
    if (player.net) {
        mcts.simulations = player.simulations > 0 ? player.simulations : cfg.simulations;
        mcts.randomSymmetry = cfg.randomSymmetry || player.randomSymmetry;
    } else {
        mcts.simulations = player.simulations;   // a bot's own count
        mcts.randomSymmetry = false;             // equal priors gain nothing from it
    }
    return mcts;
}

namespace {

// The checks playMatch makes of one match's players, whatever its games.
void checkPlayers(const ArenaPlayer& a, const ArenaPlayer& b, const ArenaConfig& cfg,
                  int inferenceBatch) {
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
    for (const ArenaPlayer* p : {&a, &b}) {
        const int batch = arenaSearchConfig(*p, cfg).batchSize;
        if (batch > inferenceBatch)
            throw std::invalid_argument("playMatch: inference batch " + std::to_string(inferenceBatch) +
                                        " is below a side's mcts batch " + std::to_string(batch));
    }
}

// The checks playMatch makes of one match of `games` games.
void checkMatch(const ArenaPlayer& a, const ArenaPlayer& b, const ArenaConfig& cfg, int games,
                int inferenceBatch) {
    checkPlayers(a, b, cfg, inferenceBatch);
    if (games < 2 || games % 2 != 0)
        throw std::invalid_argument("playMatch: games must be even and >= 2, got " + std::to_string(games));
}

// One game of one range: the pool's unit of work.
struct GameTask {
    int range, game;   // game: its index in its match
};

} // namespace

int arenaWorkers(int workers, const std::string& backendName, int games) {
    // With metal, every game of the wave at once: a worker waiting for the
    // GPU costs no CPU, and an arena game has only one search in flight, so
    // the games in flight are what fills the passes.
    const bool gpu = backendName == "metal" || backendName == "cuda";
    if (workers <= 0 && gpu) workers = kMaxArenaWorkers;
    return std::max(1, std::min(resolveWorkers(workers, gpu), games));
}

void arena_probe::setOneMatchAtATime(bool on) { gOneMatchAtATime = on; }

std::vector<ArenaGameResult> playArenaGames(const std::vector<ArenaGameRange>& ranges,
                                            const ArenaConfig& cfg, const EngineRun& run,
                                            SliceStats* stats) {
    const auto start = std::chrono::steady_clock::now();
    for (const ArenaGameRange& r : ranges) {
        checkPlayers(r.a, r.b, cfg, run.inferenceBatch);
        if (r.first < 0 || r.count < 0)
            throw std::invalid_argument("playArenaGames: bad game range [" + std::to_string(r.first) +
                                        ", +" + std::to_string(r.count) + ")");
    }
    // Every network of the set: one shape, since a worker's workspace serves them all.
    const Network* first = nullptr;
    for (const ArenaGameRange& r : ranges)
        for (const ArenaPlayer* p : {&r.a, &r.b})
            if (p->net && !first) first = p->net;
    for (const ArenaGameRange& r : ranges)
        for (const ArenaPlayer* p : {&r.a, &r.b})
            if (p->net && p->net->config() != first->config())
                throw std::invalid_argument("playMatch: networks have different architectures");

    // The distinct networks, in order of first appearance, and the largest
    // request each one's seats make.
    std::vector<const Network*> nets;
    std::vector<int> maxRequest;
    std::vector<std::array<int, 2>> seatNet(ranges.size(), {-1, -1});   // index into nets; -1 = bot
    std::vector<std::array<MctsConfig, 2>> seatSearch(ranges.size());
    for (size_t i = 0; i < ranges.size(); ++i) {
        const ArenaPlayer* players[2] = {&ranges[i].a, &ranges[i].b};
        for (int p = 0; p < 2; ++p) {
            seatSearch[i][p] = arenaSearchConfig(*players[p], cfg);
            if (!players[p]->net) continue;
            auto it = std::find(nets.begin(), nets.end(), players[p]->net);
            if (it == nets.end()) {
                nets.push_back(players[p]->net);
                maxRequest.push_back(0);
                it = nets.end() - 1;
            }
            const size_t k = static_cast<size_t>(it - nets.begin());
            seatNet[i][p] = static_cast<int>(k);
            maxRequest[k] = std::max(maxRequest[k], seatSearch[i][p].batchSize);
        }
    }

    // The games, round-robin across ranges (the first game of every range,
    // then the second, ...), so every match is under way from the start.
    // The order only decides which games start first, never a result.
    // Results go to the game's place in the output: range by range.
    std::vector<GameTask> tasks;
    std::vector<size_t> offset(ranges.size(), 0);
    int longest = 0;
    size_t total = 0;
    for (size_t i = 0; i < ranges.size(); ++i) {
        offset[i] = total;
        total += static_cast<size_t>(ranges[i].count);
        longest = std::max(longest, ranges[i].count);
    }
    for (int j = 0; j < longest; ++j)
        for (size_t i = 0; i < ranges.size(); ++i)
            if (j < ranges[i].count) tasks.push_back({static_cast<int>(i), ranges[i].first + j});

    std::vector<ArenaGameResult> results(total);
    long long evaluations = 0;
    if (!tasks.empty()) {
        const int count = static_cast<int>(tasks.size());
        const int threads = arenaWorkers(run.workers, run.backend, count);
        // Declared before the games that point into them.
        RoundEvaluators evaluators(nets, maxRequest, run.backend, threads, run.inferenceBatch);
        const NetConfig shape = first->config();
        // Each worker is one thread for the whole pool: it sets its QoS
        // class once, before its first game.
        std::vector<char> qosSet(static_cast<size_t>(threads), 0);
        parallelFor(count, threads, [&](int t, int worker) {
            if (!qosSet[static_cast<size_t>(worker)]) {
                setThreadQos(run.qos);
                qosSet[static_cast<size_t>(worker)] = 1;
            }
            const GameTask& task = tasks[static_cast<size_t>(t)];
            const size_t i = static_cast<size_t>(task.range);
            Seat seats[2];
            for (int p = 0; p < 2; ++p) {
                Seat& seat = seats[p];
                seat.mcts = seatSearch[i][p];
                if (seatNet[i][p] >= 0) {
                    Evaluator& shared = evaluators.get(static_cast<size_t>(seatNet[i][p]), worker);
                    if (run.cancel) {
                        seat.own = std::make_unique<CancellableEvaluator>(shared, *run.cancel);
                        seat.evaluator = seat.own.get();
                    } else {
                        seat.evaluator = &shared;
                    }
                } else {
                    // A bot needs no network: the board size and history come
                    // from the round's networks.
                    seat.own = std::make_unique<UniformEvaluator>(shape.boardSize, shape.history, 0.0f,
                                                                  seat.mcts.batchSize);
                    seat.evaluator = seat.own.get();
                }
            }
            const size_t at = offset[i] + static_cast<size_t>(task.game - ranges[i].first);
            results[at] = playArenaGame(seats[0], seats[1], cfg, shape.boardSize, ranges[i].seed, task.game);
        });
        evaluations = evaluators.evaluations();
    }
    if (stats) {
        stats->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        stats->positions = 0;
        for (const ArenaGameResult& g : results) stats->positions += static_cast<long long>(g.moves.size());
        stats->evaluations = evaluations;
    }
    return results;
}

MatchResult matchResult(const std::vector<ArenaGameResult>& games) {
    MatchResult result;
    result.games = static_cast<int>(games.size());
    std::set<std::vector<int>> distinct;
    for (const ArenaGameResult& g : games) {
        result.scoreA += g.scoreA;
        if (g.scoreA == 1.0) ++result.winsA;
        else if (g.scoreA == 0.0) ++result.winsB;
        else ++result.draws;
        distinct.insert(g.moves);
    }
    result.distinctGames = static_cast<int>(distinct.size());
    return result;
}

std::vector<ArenaGameRange> waveSlice(const std::vector<ArenaMatchSpec>& wave, int first, int count) {
    long long total = 0;
    for (const ArenaMatchSpec& m : wave) total += m.games;
    if (first < 0 || count < 0 || first + static_cast<long long>(count) > total)
        throw std::invalid_argument("waveSlice: games [" + std::to_string(first) + ", +" +
                                    std::to_string(count) + ") are outside a wave of " +
                                    std::to_string(total) + " games");
    std::vector<ArenaGameRange> out;
    long long matchStart = 0;
    const long long end = first + static_cast<long long>(count);
    for (const ArenaMatchSpec& m : wave) {
        const long long matchEnd = matchStart + m.games;
        const long long from = std::max<long long>(first, matchStart), to = std::min(end, matchEnd);
        if (from < to)
            out.push_back({m.a, m.b, m.seed, static_cast<int>(from - matchStart), static_cast<int>(to - from)});
        matchStart = matchEnd;
    }
    return out;
}

std::vector<MatchResult> playMatches(const std::vector<ArenaMatchSpec>& matches, const ArenaConfig& cfg,
                                     int workers, const std::string& backendName, int inferenceBatch) {
    for (const ArenaMatchSpec& m : matches) checkMatch(m.a, m.b, cfg, m.games, inferenceBatch);
    if (matches.empty()) return {};

    if (gOneMatchAtATime && matches.size() > 1) {
        std::vector<MatchResult> out;
        for (const ArenaMatchSpec& m : matches) out.push_back(playMatches({m}, cfg, workers, backendName,
                                                                          inferenceBatch)[0]);
        return out;
    }

    // Every game of every match in one pool: each match is one whole range.
    std::vector<ArenaGameRange> ranges;
    for (const ArenaMatchSpec& m : matches) ranges.push_back({m.a, m.b, m.seed, 0, m.games});
    EngineRun run;
    run.backend = backendName;
    run.workers = workers;
    run.inferenceBatch = inferenceBatch;
    const std::vector<ArenaGameResult> games = playArenaGames(ranges, cfg, run);

    // Results are stored by match and game index, never by completion order,
    // so every match is identical for any number of workers.
    std::vector<MatchResult> results;
    size_t at = 0;
    for (const ArenaMatchSpec& m : matches) {
        const auto from = games.begin() + static_cast<std::ptrdiff_t>(at);
        results.push_back(matchResult(std::vector<ArenaGameResult>(from, from + m.games)));
        at += static_cast<size_t>(m.games);
    }
    return results;
}

MatchResult playMatch(const ArenaPlayer& a, const ArenaPlayer& b, const ArenaConfig& cfg,
                      uint64_t seed, int workers, const std::string& backendName, int inferenceBatch) {
    return playMatches({{a, b, cfg.games, seed}}, cfg, workers, backendName, inferenceBatch)[0];
}

MatchResult playMatch(const Network& a, const Network& b, const ArenaConfig& cfg, uint64_t seed,
                      int workers, const std::string& backendName, int inferenceBatch) {
    return playMatch(ArenaPlayer::network(a), ArenaPlayer::network(b), cfg, seed, workers,
                     backendName, inferenceBatch);
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
