#include "azgo/parallel.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <climits>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace azgo {

void parallelFor(int count, int workers, const std::function<void(int index, int worker)>& fn) {
    assert(workers >= 1);
    if (count <= 0) return;
    workers = std::min(workers, count);   // no idle threads

    std::atomic<int> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr firstError;
    std::mutex errorMutex;

    auto run = [&](int worker) {
        // Stop taking new indices once any worker has failed; indices already
        // running on other threads finish normally.
        while (!failed.load()) {
            const int index = next.fetch_add(1);
            if (index >= count) return;
            try {
                fn(index, worker);
            } catch (...) {
                std::lock_guard<std::mutex> lock(errorMutex);
                if (!firstError) firstError = std::current_exception();
                failed.store(true);
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (int w = 0; w < workers; ++w) threads.emplace_back(run, w);
    for (std::thread& t : threads) t.join();
    if (firstError) std::rethrow_exception(firstError);
}

int resolveWorkers(int workers) {
    if (workers > 0) return workers;
    const unsigned hw = std::thread::hardware_concurrency();   // 0 if unknown
    return hw > 0 ? static_cast<int>(hw) : 1;
}

int resolveWorkers(int workers, bool gpu) {
    if (workers > 0 || !gpu) return resolveWorkers(workers);
    return 2 * resolveWorkers(0);
}

int resolveTrainThreads(int threads) {
    const int resolved = resolveWorkers(threads);
    const unsigned hw = std::thread::hardware_concurrency();   // 0 if unknown
    return hw > 0 ? std::min(resolved, static_cast<int>(hw)) : resolved;
}

// ---------------------------------------------------------------- ThreadPool

namespace {

// A hint to the core that this is a spin-wait loop.
inline void cpuRelax() {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

// How long a helper keeps polling for the next run before it sleeps. A
// training step is ~100 short parallel loops separated by short serial
// stretches; waking a sleeping thread costs tens of microseconds, so a brief
// spin keeps the helpers ready between the loops of one step.
constexpr int kSpinIterations = 1 << 14;

} // namespace

ThreadPool::ThreadPool(int threads) : threads_(threads) {
    assert(threads >= 1);
    try {
        helpers_.reserve(static_cast<size_t>(threads - 1));
        for (int t = 1; t < threads; ++t) helpers_.emplace_back(&ThreadPool::helperLoop, this);
    } catch (...) {
        // The destructor will not run for a half-built pool: stop and join
        // the helpers already started, then report the failure.
        stopHelpers();
        throw;
    }
}

ThreadPool::~ThreadPool() { stopHelpers(); }

void ThreadPool::stopHelpers() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_.store(true);
    }
    wake_.notify_all();
    for (std::thread& t : helpers_) t.join();
    helpers_.clear();
}

void ThreadPool::work() {
    while (!failed_.load(std::memory_order_relaxed)) {
        const int chunk = next_.fetch_add(1);
        if (chunk >= chunks_) return;
        try {
            (*job_)(chunk);
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!error_) error_ = std::current_exception();
            failed_.store(true);
        }
    }
}

void ThreadPool::helperLoop() {
    uint64_t seen = 0;
    while (true) {
        for (int i = 0; i < kSpinIterations; ++i) {
            if (epoch_.load(std::memory_order_acquire) != seen || stop_.load(std::memory_order_relaxed)) break;
            cpuRelax();
        }
        {
            // job_ and chunks_ were written under the mutex before the epoch
            // changed, so taking it here makes them visible.
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return stop_.load() || epoch_.load() != seen; });
            if (stop_.load()) return;
            seen = epoch_.load();
        }
        work();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (--busy_ == 0) done_.notify_one();
        }
    }
}

void ThreadPool::run(int chunks, const std::function<void(int)>& fn) {
    if (chunks <= 0) return;
    // One run at a time. A nested run would deadlock (the caller's helpers
    // are busy with the outer one); a concurrent one would overwrite the job.
    if (running_.exchange(true))
        throw std::logic_error("ThreadPool::run called while another run of the same pool is in progress "
                               "(nested in a chunk, or from a second thread)");
    struct Clear {
        std::atomic<bool>& flag;
        ~Clear() { flag.store(false); }
    } clear{running_};
    if (threads_ == 1 || chunks == 1) {
        for (int c = 0; c < chunks; ++c) fn(c);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        job_ = &fn;
        chunks_ = chunks;
        next_.store(0);
        failed_.store(false);
        error_ = nullptr;
        busy_ = static_cast<int>(helpers_.size());
        epoch_.fetch_add(1, std::memory_order_release);
    }
    wake_.notify_all();
    work();
    std::exception_ptr error;
    {
        // Every helper must be done with this run (fn lives on the caller's
        // stack) before run returns.
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [&] { return busy_ == 0; });
        job_ = nullptr;
        error = error_;
        error_ = nullptr;
    }
    if (error) std::rethrow_exception(error);
}

void parallelChunks(ThreadPool* pool, int chunks, const std::function<void(int)>& fn) {
    if (pool) {
        pool->run(chunks, fn);
        return;
    }
    for (int c = 0; c < chunks; ++c) fn(c);
}

void parallelRanges(ThreadPool* pool, size_t count, size_t chunk,
                    const std::function<void(size_t, size_t)>& fn) {
    assert(chunk >= 1);
    const size_t chunks = (count + chunk - 1) / chunk;
    assert(chunks <= static_cast<size_t>(INT_MAX));
    parallelChunks(pool, static_cast<int>(chunks), [&](int c) {
        const size_t begin = static_cast<size_t>(c) * chunk;
        fn(begin, std::min(count, begin + chunk));
    });
}

} // namespace azgo
