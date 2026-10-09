// The inference hub, tested in every build with a CPU stand-in for the GPU.
//
// CpuTestRunner is a BatchRunner that runs Network::infer on the naive
// backend on one helper thread per slot and calls done from there, as the
// GPU's completion handler would. The naive backend is batch-independent by
// construction (every output element sums over k in one fixed order), so a
// position's outputs must be the same bits whichever batch the hub put it in:
// exactly the property the Metal runner provides on the GPU. In marker mode
// it skips the network and echoes each position's first input value, which
// makes lost or duplicated rows easy to see.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/inference_hub.hpp"
#include "azgo/network.hpp"
#include "azgo/parallel.hpp"
#include "azgo/rng.hpp"
#include "azgo/selfplay.hpp"

#ifdef __APPLE__
#include <mach/mach.h>
#endif

using namespace azgo;

namespace {

// ---------------------------------------------------------------- the CPU runner

class CpuTestRunner : public BatchRunner {
public:
    // net null: marker mode. failBatch > 0: that batch (1-based, in order of
    // begin calls) reports failure, without computing.
    CpuTestRunner(const Network* net, const NetConfig& cfg, int maxBatch, int slots = kInferenceSlots,
                  int failBatch = 0)
        : net_(net), cfg_(cfg), maxBatch_(maxBatch), failBatch_(failBatch) {
        for (int s = 0; s < slots; ++s) {
            auto st = std::make_unique<SlotState>();
            st->input.resize(static_cast<size_t>(maxBatch) * cfg.planes() * cfg.area());
            st->logits.resize(static_cast<size_t>(maxBatch) * cfg.actions());
            st->values.resize(static_cast<size_t>(maxBatch));
            if (net) st->ws = std::make_unique<Workspace>(cfg, maxBatch, st->backend);
            slots_.push_back(std::move(st));
        }
        for (int s = 0; s < slots; ++s) slots_[s]->thread = std::thread([this, s] { helper(s); });
    }
    ~CpuTestRunner() override {
        for (auto& st : slots_) {
            {
                std::lock_guard<std::mutex> lock(st->mutex);
                st->stop = true;
            }
            st->wake.notify_one();
            st->thread.join();
        }
    }
    int slots() const override { return static_cast<int>(slots_.size()); }
    int maxBatch() const override { return maxBatch_; }
    float* input(int slot) override { return slots_[slot]->input.data(); }
    const float* logits(int slot) const override { return slots_[slot]->logits.data(); }
    const float* values(int slot) const override { return slots_[slot]->values.data(); }
    void begin(int slot, int B, std::function<void(int, std::string)> done) override {
        SlotState& st = *slots_[slot];
        {
            std::lock_guard<std::mutex> lock(st.mutex);
            assert(!st.job);   // the hub never reuses a slot in flight
            st.job = true;
            st.B = B;
            st.batchNumber = ++begun_;
            st.done = std::move(done);
        }
        st.wake.notify_one();
    }

private:
    struct SlotState {
        std::vector<float> input, logits, values;
        NaiveBackend backend;
        std::unique_ptr<Workspace> ws;
        std::thread thread;
        std::mutex mutex;
        std::condition_variable wake;
        bool job = false, stop = false;
        int B = 0, batchNumber = 0;
        std::function<void(int, std::string)> done;
    };

    void helper(int s) {
        SlotState& st = *slots_[s];
        for (;;) {
            std::function<void(int, std::string)> done;
            int B, number;
            {
                std::unique_lock<std::mutex> lock(st.mutex);
                st.wake.wait(lock, [&] { return st.job || st.stop; });
                if (!st.job) return;
                B = st.B;
                number = st.batchNumber;
                done = std::move(st.done);
            }
            std::string error;
            if (number == failBatch_) {
                error = "test runner failed on batch " + std::to_string(number);
            } else if (net_) {
                net_->infer(*st.ws, st.input.data(), B, st.logits.data(), st.values.data());
            } else {
                const size_t in = static_cast<size_t>(cfg_.planes()) * cfg_.area();
                for (int b = 0; b < B; ++b) {
                    const float marker = st.input[b * in];
                    st.values[b] = marker;
                    for (int a = 0; a < cfg_.actions(); ++a) st.logits[b * cfg_.actions() + a] = marker + a;
                }
            }
            {
                std::lock_guard<std::mutex> lock(st.mutex);
                st.job = false;   // before done: the hub may begin this slot again at once
            }
            done(s, error);
        }
    }

    const Network* net_;
    NetConfig cfg_;
    int maxBatch_;
    int failBatch_;
    std::atomic<int> begun_{0};
    std::vector<std::unique_ptr<SlotState>> slots_;
};

// ---------------------------------------------------------------- helpers

NetConfig tinyConfig() {
    NetConfig c;
    c.boardSize = 5;
    c.history = 2;
    c.blocks = 1;
    c.filters = 8;
    c.valueHidden = 8;
    return c;
}

// Positions and their outputs evaluated one at a time (the reference).
struct Pool {
    NetConfig cfg;
    int count;
    std::vector<float> input, logits, values;
    size_t in() const { return static_cast<size_t>(cfg.planes()) * cfg.area(); }
    Pool(const Network& net, int count_, uint64_t seed) : cfg(net.config()), count(count_) {
        Rng rng(seed);
        input.resize(count * in());
        for (float& v : input) v = static_cast<float>(rng.below(2));
        logits.resize(static_cast<size_t>(count) * cfg.actions());
        values.resize(count);
        NaiveBackend be;
        Workspace ws(cfg, 1, be);
        for (int i = 0; i < count; ++i)
            net.infer(ws, input.data() + i * in(), 1, logits.data() + i * cfg.actions(), values.data() + i);
    }
};

// Runs fn(t) on n threads and joins them, or aborts the test if they have not
// all finished within `seconds` (a hung waiter is a failure, not a hang).
void runThreads(int n, double seconds, const std::function<void(int)>& fn) {
    std::mutex m;
    std::condition_variable cv;
    int finished = 0;
    std::vector<std::thread> threads;
    for (int t = 0; t < n; ++t) {
        threads.emplace_back([&, t] {
            fn(t);
            std::lock_guard<std::mutex> lock(m);
            ++finished;
            cv.notify_all();
        });
    }
    {
        std::unique_lock<std::mutex> lock(m);
        if (!cv.wait_for(lock, std::chrono::duration<double>(seconds), [&] { return finished == n; })) {
            std::printf("test_inference_hub: %d of %d threads still waiting after %.0f s\n",
                        n - finished, n, seconds);
            std::abort();
        }
    }
    for (auto& t : threads) t.join();
}

int threadCount() {
#ifdef __APPLE__
    thread_act_array_t list;
    mach_msg_type_number_t count = 0;
    if (task_threads(mach_task_self(), &list, &count) != KERN_SUCCESS) return -1;
    for (mach_msg_type_number_t i = 0; i < count; ++i) mach_port_deallocate(mach_task_self(), list[i]);
    vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(list), count * sizeof(thread_act_t));
    return static_cast<int>(count);
#else
    return -1;   // not checked elsewhere
#endif
}

std::unique_ptr<InferenceHub> makeHub(const Network* net, const NetConfig& cfg, int maxBatch,
                                      uint64_t chaos, int failBatch = 0) {
    InferenceHubOptions o;
    o.maxBatch = maxBatch;
    o.chaosSeed = chaos;
    return std::make_unique<InferenceHub>(
        std::make_unique<CpuTestRunner>(net, cfg, maxBatch, kInferenceSlots, failBatch), cfg, o);
}

// ---------------------------------------------------------------- tests

void test_hub_single_request() {
    const NetConfig cfg = tinyConfig();
    const Network net(cfg, 1);
    const Pool pool(net, 1, 2);
    auto hub = makeHub(&net, cfg, 8, 0);
    auto ev = hub->makeEvaluator(8);
    assert(ev->boardSize() == 5 && ev->history() == 2 && ev->maxBatch() == 8);
    std::vector<float> logits(cfg.actions()), values(1);
    ev->evaluate(pool.input.data(), 1, logits.data(), values.data());
    assert(std::memcmp(logits.data(), pool.logits.data(), logits.size() * sizeof(float)) == 0);
    assert(std::memcmp(values.data(), pool.values.data(), sizeof(float)) == 0);
    ev.reset();
    assert(hub->stats().batches == 1 && hub->stats().positions == 1);
}

// Every request's outputs equal its positions evaluated alone, bitwise, for
// 1, 4 and 32 client threads, batches capped at 8, 64 and 512, and chaos off
// and on with three seeds.
void test_hub_outputs_equal_alone_under_chaos() {
    const NetConfig cfg = tinyConfig();
    const Network net(cfg, 3);
    const Pool pool(net, 300, 4);
    const size_t in = pool.in(), L = static_cast<size_t>(cfg.actions());
    for (int clients : {1, 4, 32}) {
        for (int maxBatch : {8, 64, 512}) {
            for (uint64_t chaos : {0ull, 11ull, 12ull, 13ull}) {
                auto hub = makeHub(&net, cfg, maxBatch, chaos);
                std::atomic<long long> positions{0};
                runThreads(clients, 60, [&](int t) {
                    auto ev = hub->makeEvaluator(8);
                    Rng rng(deriveSeed(100, static_cast<uint64_t>(t), chaos,
                                       static_cast<uint64_t>(maxBatch)));
                    std::vector<float> x(8 * in), logits(8 * L), values(8);
                    for (int r = 0; r < 200; ++r) {
                        const int B = 1 + rng.below(8);
                        int picks[8];
                        for (int b = 0; b < B; ++b) {
                            picks[b] = rng.below(pool.count);
                            std::copy_n(pool.input.data() + picks[b] * in, in, x.data() + b * in);
                        }
                        ev->evaluate(x.data(), B, logits.data(), values.data());
                        for (int b = 0; b < B; ++b) {
                            assert(std::memcmp(logits.data() + b * L, pool.logits.data() + picks[b] * L,
                                               L * sizeof(float)) == 0);
                            assert(std::memcmp(&values[b], &pool.values[picks[b]], sizeof(float)) == 0);
                        }
                        positions += B;
                    }
                });
                const InferenceHub::Stats st = hub->stats();
                assert(st.positions == positions.load());
                assert(st.batches >= 1 && st.batches <= clients * 200);
            }
        }
    }
}

void test_hub_refuses_oversized_request() {
    const NetConfig cfg = tinyConfig();
    auto hub = makeHub(nullptr, cfg, 16, 0);
    bool threw = false;
    try {
        hub->makeEvaluator(17);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    auto ok = hub->makeEvaluator(16);   // the largest request that fits
    assert(ok->maxBatch() == 16);
}

// The runner fails the third batch: every request of that batch and every
// later one throws with the runner's text, and no thread is left waiting.
void test_hub_failure_reaches_every_waiter() {
    const NetConfig cfg = tinyConfig();
    for (uint64_t chaos : {0ull, 21ull}) {
        auto hub = makeHub(nullptr, cfg, 64, chaos, 3);
        std::atomic<int> failures{0};
        const size_t in = static_cast<size_t>(cfg.planes()) * cfg.area();
        runThreads(8, 30, [&](int) {
            auto ev = hub->makeEvaluator(4);
            std::vector<float> x(4 * in, 1.0f), logits(4 * cfg.actions()), values(4);
            bool failed = false;
            for (int r = 0; r < 100; ++r) {
                try {
                    ev->evaluate(x.data(), 4, logits.data(), values.data());
                    assert(!failed);   // once failed, the hub never succeeds again
                } catch (const std::runtime_error& e) {
                    const std::string what = e.what();
                    assert(what.find("test runner failed on batch 3") != std::string::npos);
                    failed = true;
                }
            }
            assert(failed);   // 8 x 100 requests cannot all fit in two batches
            ++failures;
        });
        assert(failures == 8);
    }
}

// The hub shuts down cleanly: after a burst with the evaluators gone, its
// destructor joins the dispatcher; 100 hubs leave no thread behind.
void test_hub_shutdown() {
    const NetConfig cfg = tinyConfig();
    const size_t in = static_cast<size_t>(cfg.planes()) * cfg.area();
    {
        auto hub = makeHub(nullptr, cfg, 32, 31);
        runThreads(16, 30, [&](int) {
            auto ev = hub->makeEvaluator(8);
            std::vector<float> x(8 * in, 2.0f), logits(8 * cfg.actions()), values(8);
            for (int r = 0; r < 50; ++r) ev->evaluate(x.data(), 8, logits.data(), values.data());
        });
    }
    const int before = threadCount();
    for (int i = 0; i < 100; ++i) {
        auto hub = makeHub(nullptr, cfg, 8, i % 2 ? 0 : 41 + i);
        if (i % 3 == 0) {
            auto ev = hub->makeEvaluator(2);
            std::vector<float> x(2 * in, 1.0f), logits(2 * cfg.actions()), values(2);
            ev->evaluate(x.data(), 2, logits.data(), values.data());
        }
    }
    const int after = threadCount();
    assert(before < 0 || after == before);
}

// 64 threads x 1,000 requests with chaos: every row comes back to its own
// request, none lost or duplicated (each position carries a unique marker,
// which the marker runner echoes).
void test_hub_stress() {
    const NetConfig cfg = tinyConfig();
    const size_t in = static_cast<size_t>(cfg.planes()) * cfg.area();
    const int L = cfg.actions();
    auto hub = makeHub(nullptr, cfg, 256, 51);
    std::atomic<long long> positions{0};
    runThreads(64, 120, [&](int t) {
        auto ev = hub->makeEvaluator(8);
        Rng rng(deriveSeed(200, static_cast<uint64_t>(t)));
        std::vector<float> x(8 * in, 0.0f), logits(8 * L), values(8);
        for (int r = 0; r < 1000; ++r) {
            const int B = 1 + rng.below(8);
            for (int b = 0; b < B; ++b) x[b * in] = static_cast<float>((t * 1000 + r) * 8 + b);   // < 2^24
            std::fill(values.begin(), values.end(), -1.0f);
            ev->evaluate(x.data(), B, logits.data(), values.data());
            for (int b = 0; b < B; ++b) {
                const float marker = static_cast<float>((t * 1000 + r) * 8 + b);
                assert(values[b] == marker);
                assert(logits[b * L] == marker && logits[b * L + L - 1] == marker + (L - 1));
            }
            for (int b = B; b < 8; ++b) assert(values[b] == -1.0f);   // nothing written past B
            positions += B;
        }
    });
    assert(hub->stats().positions == positions.load());
}

// End to end: self-play through the hub plays exactly the games of today's
// per-worker LocalEvaluators on the naive backend, for 1 and 6 workers, with
// chaos on. The hub changes nothing but throughput.
void test_selfplay_through_hub_equals_local() {
    const NetConfig cfg = tinyConfig();
    const Network net(cfg, 61);
    SelfPlayConfig sp;
    sp.mcts.simulations = 16;
    sp.mcts.batchSize = 8;
    sp.temperatureMoves = 6;
    const int games = 8;
    auto play = [&](bool hubbed, int workers) {
        std::vector<GameRecord> records(games);
        std::unique_ptr<InferenceHub> hub;
        std::vector<std::unique_ptr<NaiveBackend>> backends;
        std::vector<std::unique_ptr<Evaluator>> evaluators;
        if (hubbed) hub = makeHub(&net, cfg, 64, 71);
        for (int w = 0; w < workers; ++w) {
            if (hubbed) {
                evaluators.push_back(hub->makeEvaluator(sp.mcts.batchSize));
            } else {
                backends.push_back(std::make_unique<NaiveBackend>());
                evaluators.push_back(
                    std::make_unique<LocalEvaluator>(net, *backends.back(), sp.mcts.batchSize));
            }
        }
        parallelFor(games, workers, [&](int i, int w) {
            Rng rng(deriveSeed(81, static_cast<uint64_t>(i)));
            records[i] = playSelfPlayGame(*evaluators[w], sp, rng);
        });
        evaluators.clear();
        return records;
    };
    const std::vector<GameRecord> local = play(false, 1);
    for (int workers : {1, 6}) {
        const std::vector<GameRecord> hubbed = play(true, workers);
        for (int i = 0; i < games; ++i) {
            const GameRecord& a = local[i];
            const GameRecord& b = hubbed[i];
            assert(a.moves == b.moves && a.stones == b.stones && a.visits == b.visits);
            assert(a.resultForBlack == b.resultForBlack && a.finalOwnership == b.finalOwnership);
        }
    }
}

} // namespace

int main() {
    test_hub_single_request();
    test_hub_outputs_equal_alone_under_chaos();
    test_hub_refuses_oversized_request();
    test_hub_failure_reaches_every_waiter();
    test_hub_shutdown();
    test_hub_stress();
    test_selfplay_through_hub_equals_local();
    std::printf("test_inference_hub: all tests passed\n");
    return 0;
}
