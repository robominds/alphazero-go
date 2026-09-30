// A tiny work-sharing loop. Self-play and arena games are independent, so
// W threads simply pull the next game index from a shared atomic counter
// until none are left. Which thread plays which game varies from run to
// run, but every game's randomness is seeded from its index and results
// are stored by index, so the output does not depend on the thread count.
#pragma once
#include <functional>

namespace azgo {

// Calls fn(index, worker) for every index in [0, count) using `workers`
// std::threads (worker in [0, workers)). Indices are handed out by an atomic
// counter, so callers must store results by index, never by completion order.
// The first exception thrown by fn is rethrown after all threads join.
void parallelFor(int count, int workers, const std::function<void(int index, int worker)>& fn);

// workers <= 0 means std::thread::hardware_concurrency() (at least 1).
int resolveWorkers(int workers);

} // namespace azgo
