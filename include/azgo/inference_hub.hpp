// Batching leaves across workers. Each self-play or arena worker runs its
// own search and asks its Evaluator for 8 positions at a time. On the CPU
// every worker evaluates its own small batches; a GPU only pays off on large
// ones. The InferenceHub sits between the two: every worker gets a
// BatchingEvaluator that queues its request and blocks, and one dispatcher
// thread packs whatever is waiting into one batch, runs it through a
// BatchRunner (the GPU network) and hands each worker its rows back.
//
// Flush policy: a batch is submitted as soon as a runner slot is free and
// anything waits. With two slots, one batch computes while the next one
// fills; under load batches grow by themselves, with light load nobody waits
// for company. No timer, no "wait for every worker", so nothing can deadlock.
//
// Determinism: which requests share a batch depends on timing. The results
// do not, provided the runner is batch-independent: a position's outputs a
// pure function of its own input (the Metal kernels are, by construction and
// by test; the naive CPU backend is too). Chaos mode, for tests, randomizes
// batch sizes, submission delays and the order inside a batch to prove it.
//
// Plain C++17: the hub knows nothing about Metal.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "azgo/evaluator.hpp"
#include "azgo/network.hpp"
#include "azgo/rng.hpp"

namespace azgo {

// Runs one forward pass on a batch, asynchronously, in one of a few slots.
class BatchRunner {
public:
    virtual ~BatchRunner() = default;
    virtual int slots() const = 0;               // batches that may be in flight at once
    virtual int maxBatch() const = 0;
    virtual float* input(int slot) = 0;          // [maxBatch][planes][area] staging for the slot
    // Starts the pass on the first B positions of input(slot); calls
    // done(slot, error) from any thread when the outputs can be read (error
    // empty on success). done must be called exactly once.
    virtual void begin(int slot, int B, std::function<void(int slot, std::string error)> done) = 0;
    virtual const float* logits(int slot) const = 0;   // [B][actions], valid after done
    virtual const float* values(int slot) const = 0;   // [B]
};

// The slots of the Metal runner: one batch on the GPU while the next fills.
constexpr int kInferenceSlots = 2;

struct InferenceHubOptions {
    int maxBatch = 256;           // inference_batch: most positions per pass
    uint64_t chaosSeed = 0;       // tests only: nonzero randomizes batching
};

class InferenceHub;

// A worker's handle on the hub: evaluate() queues the request and blocks
// until its rows are back. Copies happen in the dispatcher; the caller's
// arrays stay valid because the caller is blocked. Throws
// std::runtime_error("GPU inference failed: ...") if the runner failed this
// or any earlier batch.
class BatchingEvaluator : public Evaluator {
public:
    BatchingEvaluator(InferenceHub& hub, int maxRequest);
    int boardSize() const override;
    int history() const override;
    int maxBatch() const override { return maxRequest_; }
    void evaluate(const float* input, int B, float* logits, float* values) override;
private:
    InferenceHub& hub_;
    int maxRequest_;
};

// One per network per phase. Thread-safe; any number of BatchingEvaluators,
// which must all be destroyed before the hub.
class InferenceHub {
public:
    // options.maxBatch must be in [1, runner->maxBatch()].
    InferenceHub(std::unique_ptr<BatchRunner> runner, const NetConfig& cfg, InferenceHubOptions options);
    ~InferenceHub();   // stops the dispatcher after any batch in flight completes
    InferenceHub(const InferenceHub&) = delete;
    InferenceHub& operator=(const InferenceHub&) = delete;

    // One per worker. maxRequest (the positions per evaluate call, e.g.
    // mcts_batch) must be in [1, maxBatch], else std::invalid_argument.
    std::unique_ptr<Evaluator> makeEvaluator(int maxRequest);
    const NetConfig& config() const { return cfg_; }
    int maxBatch() const { return maxBatch_; }

    struct Stats {
        long long batches = 0, positions = 0;
        double gpuSeconds = 0;   // summed from begin to done over all batches
    };
    Stats stats() const;

private:
    friend class BatchingEvaluator;
    // One evaluate() call, on its caller's stack while it waits.
    struct Request {
        const float* input;
        int B;
        float* logits;
        float* values;
        int offset = 0;              // its first row in the slot's batch
        bool done = false;
        std::string error;
        std::condition_variable wake;
    };
    struct Slot {
        bool busy = false;           // a batch is in flight or being copied out
        bool completed = false;      // the runner called done
        std::string error;
        std::vector<Request*> requests;
        int positions = 0;
        double started = 0;          // seconds, steady clock
    };
    void submit(Request& r);         // from evaluate: queue and wait
    void dispatcherLoop();
    void finish(Slot& slot, int index, std::unique_lock<std::mutex>& lock);
    void fail(Request& r, const std::string& error);   // under the lock

    std::unique_ptr<BatchRunner> runner_;
    NetConfig cfg_;
    int maxBatch_;
    size_t inputFloats_, actions_;   // per position
    bool chaos_;
    Rng chaosRng_;                   // the hub's own: never touches a run's streams

    mutable std::mutex mutex_;
    std::condition_variable dispatcherWake_;
    std::deque<Request*> queue_;
    std::vector<Slot> slots_;
    std::string failed_;             // set once a batch fails: every later request fails
    bool stop_ = false;
    Stats stats_;
    std::thread dispatcher_;
};

} // namespace azgo
