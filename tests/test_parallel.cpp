#include <atomic>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>
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

int main() {
    test_every_index_exactly_once();
    test_results_stored_by_index();
    test_exception_propagates_after_join();
    test_resolve_workers();
    std::printf("test_parallel: all tests passed\n");
    return 0;
}
