// A tiny work-sharing loop. Self-play and arena games are independent, so
// W threads simply pull the next game index from a shared atomic counter
// until none are left. Which thread plays which game varies from run to
// run, but every game's randomness is seeded from its index and results
// are stored by index, so the output does not depend on the thread count.
//
// ThreadPool serves the training step, whose work comes as many short
// loops (one per layer) rather than whole games: its threads are started
// once and woken for each loop. Work is cut into fixed chunks that depend
// only on the data's shape, never on the number of threads, and each chunk
// writes its own part of the output; which thread runs a chunk then cannot
// change a single bit of the result.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace azgo {

// Calls fn(index, worker) for every index in [0, count) using `workers`
// std::threads (worker in [0, workers)). Indices are handed out by an atomic
// counter, so callers must store results by index, never by completion order.
// The first exception thrown by fn is rethrown after all threads join.
void parallelFor(int count, int workers, const std::function<void(int index, int worker)>& fn);

// workers <= 0 means std::thread::hardware_concurrency() (at least 1).
int resolveWorkers(int workers);

// resolveWorkers(threads), but never more than the hardware's thread count
// (when it is known): extra pool threads would only take turns on the same
// cores. The training step's result does not depend on the count.
int resolveTrainThreads(int threads);

// A fixed set of threads: the caller of run() plus threads - 1 helpers that
// live as long as the pool and sleep between runs.
class ThreadPool {
public:
    // threads >= 1. If a helper thread cannot be started, the ones already
    // running are stopped and joined and the error is rethrown.
    explicit ThreadPool(int threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    int threads() const { return threads_; }

    // Calls fn(chunk) once for every chunk in [0, chunks), on the calling
    // thread and the helpers, and returns when all are done. Chunks are
    // handed out by an atomic counter, so fn must write only its own chunk's
    // output. With one thread or one chunk it is a plain loop on the caller.
    // The first exception thrown by fn is rethrown after every thread has
    // finished. Each run waits for every helper to check in, even helpers
    // that found no chunk left: a run is a barrier, which is what lets fn
    // live on the caller's stack. One run at a time: a nested call (from fn)
    // or a concurrent one (from another thread) throws std::logic_error.
    void run(int chunks, const std::function<void(int chunk)>& fn);

private:
    void helperLoop();
    void stopHelpers();                  // stop and join every helper (destructor, failed constructor)
    void work();                         // claim and run chunks until none are left
    int threads_;
    std::vector<std::thread> helpers_;
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    const std::function<void(int)>* job_ = nullptr;
    int chunks_ = 0;
    std::atomic<int> next_{0};
    std::atomic<uint64_t> epoch_{0};     // bumped once per run; helpers wait for a change
    std::atomic<bool> stop_{false};
    std::atomic<bool> failed_{false};
    std::atomic<bool> running_{false};   // a run is in progress (guards misuse)
    int busy_ = 0;                       // helpers not yet finished with the current run
    std::exception_ptr error_;
};

// pool->run(chunks, fn), or a plain loop over the chunks in order when pool
// is null.
void parallelChunks(ThreadPool* pool, int chunks, const std::function<void(int chunk)>& fn);

// [0, count) cut into chunks of `chunk` elements (the last may be shorter),
// fn(begin, end) once per chunk, through parallelChunks. The cut depends only
// on count and chunk.
void parallelRanges(ThreadPool* pool, size_t count, size_t chunk,
                    const std::function<void(size_t begin, size_t end)>& fn);

} // namespace azgo
