#include <atomic>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "azgo/parallel.hpp"

using namespace azgo;

static void test_every_index_exactly_once() {
    const int cases[][2] = {{0, 4}, {1, 4}, {100, 1}, {100, 7}};
    for (const auto& c : cases) {
        const int count = c[0], workers = c[1];
        std::vector<std::atomic<int>> hits(count);
        for (auto& h : hits) h.store(0);
        std::atomic<int> badWorker{0};
        parallelFor(count, workers, [&](int index, int worker) {
            if (worker < 0 || worker >= workers) badWorker.fetch_add(1);
            hits[index].fetch_add(1);
        });
        assert(badWorker.load() == 0);
        for (int i = 0; i < count; ++i) assert(hits[i].load() == 1);
    }
}

static void test_results_stored_by_index() {
    std::vector<int> squares(50, -1);
    parallelFor(50, 5, [&](int index, int) { squares[index] = index * index; });
    for (int i = 0; i < 50; ++i) assert(squares[i] == i * i);
}

static void test_exception_propagates_after_join() {
    for (int workers : {1, 4}) {
        std::atomic<int> started{0};
        bool caught = false;
        try {
            parallelFor(100, workers, [&](int index, int) {
                started.fetch_add(1);
                if (index == 13) throw std::runtime_error("boom at 13");
            });
        } catch (const std::runtime_error& e) {
            caught = std::string(e.what()) == "boom at 13";
        }
        assert(caught);
        // Workers stop taking new indices after the failure, so not all 100
        // run; with one worker exactly indices 0..13 do.
        assert(started.load() >= 14 && started.load() <= 100);
        if (workers == 1) assert(started.load() == 14);
    }
}

static void test_resolve_workers() {
    assert(resolveWorkers(0) >= 1);
    assert(resolveWorkers(-2) >= 1);
    assert(resolveWorkers(3) == 3);
    assert(resolveWorkers(1) == 1);
}

// The pool's threads persist across runs; every chunk of every run is
// executed exactly once, whatever the thread and chunk counts.
static void test_pool_runs_every_chunk_once() {
    for (int threads : {1, 2, 3, 7}) {
        ThreadPool pool(threads);
        assert(pool.threads() == threads);
        for (int chunks : {0, 1, 2, 5, 64, 1000}) {
            for (int repeat = 0; repeat < 50; ++repeat) {
                std::vector<std::atomic<int>> hits(static_cast<size_t>(chunks));
                for (auto& h : hits) h.store(0);
                pool.run(chunks, [&](int c) { hits[static_cast<size_t>(c)].fetch_add(1); });
                for (auto& h : hits) assert(h.load() == 1);
            }
        }
    }
}

// One thread, or one chunk, runs inline on the calling thread.
static void test_pool_single_thread_is_inline() {
    const std::thread::id caller = std::this_thread::get_id();
    ThreadPool one(1);
    bool inline1 = true;
    one.run(10, [&](int) { inline1 = inline1 && std::this_thread::get_id() == caller; });
    assert(inline1);
    ThreadPool four(4);
    bool inline2 = false;
    four.run(1, [&](int) { inline2 = std::this_thread::get_id() == caller; });
    assert(inline2);
    // nullptr: no pool, a plain loop in chunk order.
    std::vector<int> order;
    parallelChunks(nullptr, 4, [&](int c) { order.push_back(c); });
    assert((order == std::vector<int>{0, 1, 2, 3}));
}

// With several threads the work really is spread: some chunk runs on a
// helper thread (each chunk waits until two threads have arrived, bounded).
static void test_pool_uses_helpers() {
    ThreadPool pool(4);
    std::atomic<int> arrived{0};
    std::atomic<bool> helperSeen{false};
    const std::thread::id caller = std::this_thread::get_id();
    pool.run(4, [&](int) {
        arrived.fetch_add(1);
        if (std::this_thread::get_id() != caller) helperSeen.store(true);
        for (int spin = 0; spin < 2000000 && arrived.load() < 2; ++spin) std::this_thread::yield();
    });
    assert(helperSeen.load());
}

static void test_pool_exception_propagates_and_pool_survives() {
    ThreadPool pool(3);
    bool caught = false;
    try {
        pool.run(100, [&](int c) {
            if (c == 13) throw std::runtime_error("chunk 13");
        });
    } catch (const std::runtime_error& e) {
        caught = std::string(e.what()) == "chunk 13";
    }
    assert(caught);
    std::atomic<int> sum{0};
    pool.run(10, [&](int c) { sum.fetch_add(c); });   // still usable afterwards
    assert(sum.load() == 45);
}

// Fixed chunks: the partition depends only on the count and the chunk size.
static void test_parallel_ranges() {
    for (int threads : {1, 3}) {
        ThreadPool pool(threads);
        for (size_t count : {size_t(0), size_t(1), size_t(9), size_t(10), size_t(1001)}) {
            std::vector<std::atomic<int>> hits(count);
            for (auto& h : hits) h.store(0);
            std::atomic<int> calls{0};
            parallelRanges(&pool, count, 10, [&](size_t begin, size_t end) {
                assert(begin % 10 == 0 && end > begin && end - begin <= 10 && end <= count);
                calls.fetch_add(1);
                for (size_t i = begin; i < end; ++i) hits[i].fetch_add(1);
            });
            for (auto& h : hits) assert(h.load() == 1);
            assert(calls.load() == static_cast<int>((count + 9) / 10));
        }
    }
}

// One run at a time: a run started from inside a chunk, or from a second
// thread while one is in progress, throws std::logic_error instead of
// deadlocking or corrupting the job; the pool stays usable.
static void test_pool_refuses_nested_and_concurrent_runs() {
    for (int threads : {1, 3}) {
        ThreadPool pool(threads);
        bool caught = false;
        try {
            pool.run(4, [&](int) { pool.run(2, [](int) {}); });
        } catch (const std::logic_error& e) {
            caught = std::string(e.what()).find("in progress") != std::string::npos;
        }
        assert(caught);
        std::atomic<int> sum{0};
        pool.run(10, [&](int c) { sum.fetch_add(c); });
        assert(sum.load() == 45);
    }
    ThreadPool pool(2);
    std::atomic<bool> started{false}, release{false};
    std::thread other([&] {
        pool.run(2, [&](int) {
            started.store(true);
            while (!release.load()) std::this_thread::yield();
        });
    });
    while (!started.load()) std::this_thread::yield();
    bool caught = false;
    try {
        pool.run(2, [](int) {});
    } catch (const std::logic_error&) {
        caught = true;
    }
    release.store(true);
    other.join();
    assert(caught);
    std::atomic<int> sum{0};
    pool.run(10, [&](int c) { sum.fetch_add(c); });
    assert(sum.load() == 45);
}

// The training pool never has more threads than the hardware.
static void test_resolve_train_threads() {
    const int cores = resolveWorkers(0);
    assert(resolveTrainThreads(0) == cores);
    assert(resolveTrainThreads(1) == 1);
    assert(resolveTrainThreads(256) == cores);
    assert(resolveTrainThreads(cores) == cores);
}

int main() {
    test_every_index_exactly_once();
    test_results_stored_by_index();
    test_exception_propagates_after_join();
    test_resolve_workers();
    test_pool_runs_every_chunk_once();
    test_pool_single_thread_is_inline();
    test_pool_uses_helpers();
    test_pool_exception_propagates_and_pool_survives();
    test_parallel_ranges();
    test_pool_refuses_nested_and_concurrent_runs();
    test_resolve_train_threads();
    std::printf("test_parallel: all tests passed\n");
    return 0;
}
