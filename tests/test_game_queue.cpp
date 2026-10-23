// The queue of dynamic self-play allocation (azgo/game_queue.hpp): its
// bookkeeping (every index handed out once, lowest first, requeued games
// first, concurrency, retiring engines, forced assignment), the handout
// rule that keeps late games off slow engines, and a simulation of the
// 19x19 run's engines that measures what the rule is worth against static
// slices and against a plain first-come queue.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/config.hpp"
#include "azgo/engines.hpp"
#include "azgo/game_queue.hpp"
#include "azgo/rng.hpp"

using namespace azgo;

template <typename F>
static bool throwsLogic(F f) {
    try {
        f();
    } catch (const std::logic_error&) {
        return true;
    }
    return false;
}

// Lowest index first, at most the concurrency in flight, kNone only once
// every game is finished.
static void test_handout_order_and_concurrency() {
    GameQueue q(5, {2, 1});
    assert(q.games() == 5 && q.engines() == 2 && q.queued() == 5);
    assert(q.take(0, 0) == 0 && q.take(0, 0) == 1);
    assert(q.take(0, 0) == GameQueue::kWait);   // full
    assert(q.take(1, 0) == 2 && q.take(1, 0) == GameQueue::kWait);
    assert(q.inFlight(0) == 2 && q.held(0) == std::vector<int>({0, 1}) && q.queued() == 2);
    q.finished(0, 1, 1.0);
    assert(q.finishedBy(0) == 1 && q.playedBy(1) == 0 && q.playedBy(0) == -1);
    assert(q.take(0, 1.0) == 3);
    q.finished(1, 2, 1.0);
    assert(q.take(1, 1.0) == 4);
    assert(q.take(1, 1.0) == GameQueue::kWait);   // nothing queued, games in flight
    q.finished(0, 0, 2.0);
    q.finished(0, 3, 2.0);
    q.finished(1, 4, 2.0);
    assert(q.done() && q.finishedGames() == 5);
    assert(q.take(0, 3.0) == GameQueue::kNone && q.take(1, 3.0) == GameQueue::kNone);
    // A finished game cannot be finished or requeued again, nor a foreign one.
    assert(throwsLogic([&] { q.finished(0, 0, 3.0); }));
    assert(throwsLogic([&] { q.requeue(1, 0); }));
    assert(throwsLogic([&] { GameQueue bad(3, {0}); }));
    assert(throwsLogic([&] { GameQueue bad(3, {}); }));
    // No games: done at once.
    GameQueue empty(0, {1});
    assert(empty.done() && empty.take(0, 0) == GameQueue::kNone);
}

// A failed engine's games go back to the queue (lowest first); a retired
// engine gets no more; the fallback engine joins later.
static void test_requeue_and_retire() {
    GameQueue q(6, {3, 1});
    assert(q.take(0, 0) == 0 && q.take(0, 0) == 1 && q.take(0, 0) == 2);
    assert(q.take(1, 0) == 3);
    assert(throwsLogic([&] { q.requeue(1, 0); }));   // engine 1 does not hold game 0
    q.retire(0);
    assert(q.retired(0) && !q.retired(1));
    assert(q.take(0, 0) == GameQueue::kNone);
    assert(q.held(0) == std::vector<int>({0, 1, 2}));   // still held until requeued
    q.requeue(0, 2);
    q.requeue(0, 0);
    q.requeue(0, 1);
    assert(q.inFlight(0) == 0 && q.queued() == 5);
    q.finished(1, 3, 1.0);
    // Lowest first: the requeued games, then the rest.
    assert(q.take(1, 1.0) == 0);
    q.finished(1, 0, 2.0);
    const int fallback = q.addEngine(4);
    assert(fallback == 2 && q.engines() == 3);
    std::vector<int> got;
    for (int i; (i = q.take(fallback, 2.0)) >= 0;) got.push_back(i);
    assert(got == std::vector<int>({1, 2, 4, 5}));
    for (int i : got) q.finished(fallback, i, 3.0);
    assert(q.done() && q.playedBy(4) == fallback && q.playedBy(3) == 1);
}

// Forced assignment: each engine gets its own games, in index order, no
// matter who asks first or when.
static void test_forced_assignment() {
    const std::vector<int> engineOf = {1, 0, 0, 1, 1, 0};
    GameQueue q(engineOf, {2, 1});
    assert(q.take(1, 0) == 0 && q.take(1, 0) == GameQueue::kWait);
    assert(q.take(0, 0) == 1 && q.take(0, 0) == 2 && q.take(0, 0) == GameQueue::kWait);
    q.finished(1, 0, 5.0);
    q.finished(0, 1, 100.0);
    q.finished(0, 2, 100.0);
    // The rule would hold a slow engine back here; forced mode does not.
    assert(q.take(1, 100.0) == 3);
    assert(q.take(0, 100.0) == 5);
    q.finished(1, 3, 101.0);
    assert(q.take(1, 101.0) == 4);
    q.finished(1, 4, 102.0);
    q.finished(0, 5, 102.0);
    assert(q.done());
    for (int i = 0; i < 6; ++i) assert(q.playedBy(i) == engineOf[static_cast<size_t>(i)]);
    assert(throwsLogic([&] { GameQueue bad(std::vector<int>{0, 2}, {1, 1}); }));
    GameQueue r(engineOf, {2, 1});
    assert(throwsLogic([&] { r.retire(0); }));
    assert(throwsLogic([&] { r.addEngine(1); }));
}

// The handout rule. Engine 0 is fast (games of 10 s), engine 1 slow
// (100 s); both have finished a game, so their latencies are known.
static void test_handout_rule() {
    // Warm up: each engine finishes one game, so λ = 10 and 100.
    auto warm = [](GameQueue& q) {
        assert(q.take(0, 0) == 0);
        q.finished(0, 0, 10.0);
        assert(q.take(1, 0) == 1);
        q.finished(1, 1, 100.0);
        assert(std::abs(q.latency(0) - 10.0) < 1e-12 && std::abs(q.latency(1) - 100.0) < 1e-12);
    };
    {
        // Nothing in flight: bound = now, so nobody finishes "in time" and
        // nobody else can absorb the queue (C = 0): give.
        GameQueue q(4, {1, 1});
        warm(q);
        assert(q.take(0, 100.0) == 2);
        // Engine 0's game ends at 110 = bound; engine 1 would end at 200.
        // Engine 0's slot frees at 110 and could end one more at 120 > 110:
        // it does not count, C = 0 < 1 queued: give.
        assert(q.take(1, 100.0) == 3);
    }
    {
        // A slow game in flight sets the bound; a fast engine finishes in
        // time and is given games.
        GameQueue q(5, {2, 1});
        warm(q);
        assert(q.take(1, 100.0) == 2);   // C = 0 (engine 0's free slots end at 110 > bound 100)
        assert(q.take(0, 100.0) == 3);   // bound 200: 110 <= 200
        q.finished(1, 2, 150.0);         // λ_1 = (100 + 50) / 2 = 75
        // bound = max(now 150, game 3's 110) = 150; engine 1 would end at
        // 225. Engine 0: game 3's slot frees at 150 and ends one more at
        // 160 > 150; its free slot also ends at 160: C = 0. Give.
        assert(q.take(1, 150.0) == 4);
    }
    {
        // Predictions that are overdue stop counting.
        GameQueue q(8, {3, 1});
        warm(q);
        assert(q.take(1, 100.0) == 2);   // slow game: bound 200
        assert(q.take(0, 100.0) == 3 && q.take(0, 100.0) == 4 && q.take(0, 100.0) == 5);
        q.finished(1, 2, 199.0);         // λ_1 = 99.5
        // Engine 0's games were due at 110: bound = now = 199. Engine 1
        // would end at 298.5; engine 0's slots, if they freed now, would end
        // at 209 > 199: C = 0 < 2 queued. Give.
        assert(q.take(1, 199.0) == 6);
    }
    {
        // Holding: the fast engine's free slots end games before the slow
        // game in flight does, so the slow engine leaves the queue to it.
        GameQueue q(6, {2, 2});
        warm(q);
        assert(q.take(1, 100.0) == 2);   // C = 0: give
        assert(q.take(1, 100.0) == 3);   // bound 200 = now + 100: in time
        q.finished(1, 3, 101.0);         // a very short game: λ_1 = (100 + 1) / 2 = 50.5
        // bound = game 2's 100 + 50.5 = 150.5; engine 1 would end at 151.5.
        // Engine 0's two free slots end games at 111 <= 150.5: C = 2, and 2
        // games are queued: wait.
        assert(q.take(1, 101.0) == GameQueue::kWait);
        assert(q.take(0, 101.0) == 4);   // in time
        assert(q.take(0, 101.0) == 5);
        assert(q.take(1, 101.0) == GameQueue::kWait);   // nothing queued
        // Without the rule, the slow engine would have taken one.
        GameQueue plain(6, {2, 2});
        plain.setHandoutRule(false);
        warm(plain);
        assert(plain.take(1, 100.0) == 2 && plain.take(1, 100.0) == 3);
        plain.finished(1, 3, 101.0);
        assert(plain.take(1, 101.0) == 4);
    }
    {
        // Unknown latency: an engine that has not finished a game yet gets
        // games freely, and its games do not set the bound.
        GameQueue q(4, {1, 1});
        assert(q.take(0, 0) == 0);
        q.finished(0, 0, 10.0);
        assert(q.take(1, 10.0) == 1);    // engine 1: unknown, give
        assert(q.take(0, 10.0) == 2);    // bound 10, C = 0: give
        q.finished(0, 2, 1000.0);        // λ_0 = 505
        assert(q.take(0, 1000.0) == 3);  // bound 1000, C = 0: give
    }
}

// The rule never stalls: whatever the times, an engine that keeps asking
// with a free slot eventually gets every queued game.
static void test_rule_never_stalls() {
    Rng rng(7);
    for (int trial = 0; trial < 200; ++trial) {
        const int engines = 1 + rng.below(4), games = 1 + rng.below(30);
        std::vector<int> conc;
        for (int e = 0; e < engines; ++e) conc.push_back(1 + rng.below(4));
        GameQueue q(games, conc);
        double now = 0;
        std::vector<std::vector<std::pair<int, double>>> running(static_cast<size_t>(engines));
        int steps = 0;
        while (!q.done()) {
            assert(++steps < 100000);
            now += rng.uniform() * 5;
            const int e = rng.below(engines);
            auto& mine = running[static_cast<size_t>(e)];
            // Finish games whose (random) time has come.
            for (size_t k = 0; k < mine.size();) {
                if (mine[k].second <= now) {
                    q.finished(e, mine[k].first, now);
                    mine.erase(mine.begin() + static_cast<long>(k));
                } else {
                    ++k;
                }
            }
            for (int i; (i = q.take(e, now)) >= 0;) mine.push_back({i, now + 1 + rng.uniform() * 50 * (e + 1)});
        }
        for (int e = 0; e < engines; ++e) assert(q.take(e, now) == GameQueue::kNone);
    }
}

// Dynamic concurrency: the hosts file's worker keys when set, else a
// quarter of the share for a GPU engine and the CPU rule for naive and blas.
static void test_dynamic_workers() {
    MachineSettings local, remote;
    local.alias = "local";
    remote.alias = "mini1";
    assert(dynamicWorkers(local, "metal", 104, 400, 18) == 26);
    assert(dynamicWorkers(remote, "metal", 20, 400, 18) == 5);
    assert(dynamicWorkers(remote, "cuda", 256, 400, 18) == 64);
    assert(dynamicWorkers(remote, "cuda", 1023, 2000, 18) == 256);   // at most 256
    assert(dynamicWorkers(local, "metal", 1, 400, 18) == 1 && dynamicWorkers(local, "metal", 0, 400, 18) == 1);
    assert(dynamicWorkers(local, "metal", 400, 3, 18) == 3);         // at most the games
    // CPU engines: the rule, capped by the games, not the share.
    assert(dynamicWorkers(local, "blas", 2, 400, 18) == 4 && dynamicWorkers(remote, "naive", 1, 400, 18) == 2);
    local.metalWorkers = 36;
    remote.cudaWorkers = 128;
    remote.cpuWorkers = 6;
    assert(dynamicWorkers(local, "metal", 104, 400, 18) == 36);
    assert(dynamicWorkers(remote, "cuda", 256, 400, 18) == 128);
    assert(dynamicWorkers(remote, "blas", 1, 400, 18) == 6);
    assert(dynamicWorkers(local, "metal", 104, 20, 18) == 20);
}

// ---- the simulation ---------------------------------------------------------
//
// Engines evaluate R(n) = rmax * min(1, n / sat) positions per second with n
// games in flight, shared equally among them; a game costs a random number
// of evaluations. Static: assignGames' slices, every game of a slice at
// once. Dynamic: each engine keeps its concurrency in flight from the queue,
// asking whenever a game ends and every `tick` seconds while it waits.

struct SimEngine {
    std::string name;
    double rmax, sat;
    int concurrency;   // dynamic only
};

struct SimResult {
    double phase = 0;                  // seconds until the last game ends
    std::vector<double> engineEnd;     // when each engine's last game ended
};

// The phase with each engine e playing the games `given(e, now)` hands it.
static SimResult simulate(const std::vector<SimEngine>& engines, const std::vector<double>& cost,
                          const std::vector<std::vector<int>>* slices, GameQueue* queue) {
    const size_t E = engines.size();
    struct Running {
        int game;
        double left;
    };
    std::vector<std::vector<Running>> running(E);
    SimResult out;
    out.engineEnd.assign(E, 0);
    if (slices)
        for (size_t e = 0; e < E; ++e)
            for (int i : (*slices)[e]) running[e].push_back({i, cost[static_cast<size_t>(i)]});
    double now = 0;
    const double tick = 30;
    auto ask = [&] {
        if (!queue) return;
        for (size_t e = 0; e < E; ++e)
            for (int i; (i = queue->take(static_cast<int>(e), now)) >= 0;)
                running[e].push_back({i, cost[static_cast<size_t>(i)]});
    };
    ask();
    for (;;) {
        // The next game to end, at the current rates.
        double dt = 1e300;
        bool any = false;
        for (size_t e = 0; e < E; ++e) {
            const double n = static_cast<double>(running[e].size());
            if (n == 0) continue;
            any = true;
            const double perGame = engines[e].rmax * std::min(1.0, n / engines[e].sat) / n;
            for (const Running& r : running[e]) dt = std::min(dt, r.left / perGame);
        }
        if (!any && (!queue || queue->done())) break;
        if (queue && queue->queued() > 0) dt = std::min(dt, tick);
        for (size_t e = 0; e < E; ++e) {
            const double n = static_cast<double>(running[e].size());
            if (n == 0) continue;
            const double perGame = engines[e].rmax * std::min(1.0, n / engines[e].sat) / n;
            for (Running& r : running[e]) r.left -= perGame * dt;
        }
        now += dt;
        for (size_t e = 0; e < E; ++e)
            for (size_t k = 0; k < running[e].size();) {
                if (running[e][k].left <= 1e-6) {
                    if (queue) queue->finished(static_cast<int>(e), running[e][k].game, now);
                    running[e].erase(running[e].begin() + static_cast<long>(k));
                    out.engineEnd[e] = now;
                } else {
                    ++k;
                }
            }
        ask();
    }
    out.phase = now;
    return out;
}

// 19x19 10x128 games: 760 evaluations per ply, plies ~ N(490, 60) in
// [300, 722] (the live run's averages and range).
static std::vector<double> gameCosts(int games, uint64_t seed) {
    Rng rng(seed);
    std::vector<double> cost;
    for (int i = 0; i < games; ++i) cost.push_back(std::clamp(490 + 60 * rng.normal(), 300.0, 722.0) * 760);
    return cost;
}

struct Summary {
    double mean = 0, worst = 0, spread = 0;   // loss against the ideal; mean spread of engine ends
};

static void test_simulation() {
    const int G = 400;
    // Fitted to the evidence (spec, "What the weights mean"): Studio GPU,
    // two minis, the PC's GPU.
    std::vector<SimEngine> engines = {
        {"local/metal", 2780, 4, 0}, {"mini1/metal", 490, 2, 0}, {"mini2/metal", 490, 2, 0}, {"pc/cuda", 6770, 40, 0}};
    struct Weighting {
        const char* name;
        std::vector<int> weights;
    };
    const std::vector<Weighting> weightings = {
        {"live weights 26/5/5/64", {26, 5, 5, 64}},
        {"minis weighted x2", {20, 10, 10, 60}},
        {"PC weighted short", {31, 5, 5, 59}},
    };
    std::printf("simulation: 19x19 10x128, %d games, 30 generations each; loss = phase / ideal - 1 "
                "(mean, worst), spread = last engine end - first (mean)\n", G);
    std::printf("  %-26s %-34s %8s %8s %9s\n", "weights", "allocation", "mean", "worst", "spread");
    for (const Weighting& w : weightings) {
        std::vector<EngineSpec> list;
        for (size_t e = 0; e < engines.size(); ++e) {
            EngineSpec s = parseEngineList(engines[e].name + ":1")[0];
            s.weight = w.weights[e];
            list.push_back(s);
        }
        const std::vector<Slice> slices = assignGames(list, G);
        std::vector<int> share(engines.size(), 0);
        for (const Slice& s : slices) share[static_cast<size_t>(s.engine)] = s.count;
        // The default concurrency (a quarter of the share for a GPU engine)
        // and the spec's recommendation for the live run.
        std::vector<int> quarter, tuned = {36, 4, 4, 128};
        for (size_t e = 0; e < engines.size(); ++e) {
            MachineSettings m;
            m.alias = engines[e].name.substr(0, engines[e].name.find('/'));
            quarter.push_back(dynamicWorkers(m, list[e].backend, share[e], G, 18));
        }
        struct Allocation {
            std::string name;
            std::vector<int> concurrency;   // empty: static
            bool rule;
        };
        auto label = [](const char* kind, const std::vector<int>& c) {
            return std::string(kind) + " " + std::to_string(c[0]) + "/" + std::to_string(c[1]) + "/" +
                   std::to_string(c[3]);
        };
        const std::vector<Allocation> allocations = {
            {"static", {}, false},
            {label("plain queue", quarter), quarter, false},
            {label("handout rule", quarter), quarter, true},
            {label("plain queue", tuned), tuned, false},
            {label("handout rule", tuned), tuned, true},
        };
        std::vector<Summary> summary(allocations.size());
        const int seeds = 30;
        for (int seed = 0; seed < seeds; ++seed) {
            const std::vector<double> cost = gameCosts(G, 1000 + static_cast<uint64_t>(seed));
            double rmax = 0;
            for (const SimEngine& e : engines) rmax += e.rmax;
            const double ideal = std::accumulate(cost.begin(), cost.end(), 0.0) / rmax;
            for (size_t a = 0; a < allocations.size(); ++a) {
                SimResult r;
                if (allocations[a].concurrency.empty()) {
                    std::vector<std::vector<int>> given(engines.size());
                    for (const Slice& s : slices)
                        for (int k = 0; k < s.count; ++k) given[static_cast<size_t>(s.engine)].push_back(s.first + k);
                    r = simulate(engines, cost, &given, nullptr);
                } else {
                    // The plain queue: an engine never waits while games are
                    // queued (as if every latency were unknown).
                    GameQueue q(G, allocations[a].concurrency);
                    if (!allocations[a].rule) q.setHandoutRule(false);
                    r = simulate(engines, cost, nullptr, &q);
                    assert(q.done());
                }
                const double loss = r.phase / ideal - 1;
                summary[a].mean += loss / seeds;
                summary[a].worst = std::max(summary[a].worst, loss);
                const auto [lo, hi] = std::minmax_element(r.engineEnd.begin(), r.engineEnd.end());
                summary[a].spread += (*hi - *lo) / seeds;
            }
        }
        for (size_t a = 0; a < allocations.size(); ++a)
            std::printf("  %-26s %-34s %7.1f%% %7.1f%% %8.0fs\n", a == 0 ? w.name : "",
                        allocations[a].name.c_str(), 100 * summary[a].mean, 100 * summary[a].worst,
                        summary[a].spread);
        // Acceptance: with the rule, dynamic beats static on average under
        // every weighting, with the default concurrency and the tuned one.
        assert(summary[2].mean < summary[0].mean && summary[4].mean < summary[0].mean);
    }
}

int main() {
    test_handout_order_and_concurrency();
    test_requeue_and_retire();
    test_forced_assignment();
    test_handout_rule();
    test_rule_never_stalls();
    test_dynamic_workers();
    test_simulation();
    std::printf("test_game_queue: OK\n");
    return 0;
}
