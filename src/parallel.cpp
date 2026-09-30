#include "azgo/parallel.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <exception>
#include <mutex>
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

} // namespace azgo
