// The inference hub (azgo/inference_hub.hpp): a queue of requests, one
// dispatcher thread, and a runner with a few slots.
//
// The mutex only guards moving requests between the queue and the slots and
// marking them done. The planes are copied into a slot and the results out
// of it without the lock: requests in a slot belong to the dispatcher until
// it marks them done, and their callers are blocked until then.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include "azgo/inference_hub.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

#include "azgo/metal.hpp"

namespace azgo {

namespace {

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

// ---------------------------------------------------------------- BatchingEvaluator

BatchingEvaluator::BatchingEvaluator(InferenceHub& hub, int maxRequest)
    : hub_(hub), maxRequest_(maxRequest) {}

int BatchingEvaluator::boardSize() const { return hub_.config().boardSize; }
int BatchingEvaluator::history() const { return hub_.config().history; }

void BatchingEvaluator::evaluate(const float* input, int B, float* logits, float* values) {
    if (B < 1 || B > maxRequest_)
        throw std::invalid_argument("BatchingEvaluator: " + std::to_string(B) + " positions, at most " +
                                    std::to_string(maxRequest_));
    InferenceHub::Request r;
    r.input = input;
    r.B = B;
    r.logits = logits;
    r.values = values;
    hub_.submit(r);
}

// ---------------------------------------------------------------- InferenceHub

InferenceHub::InferenceHub(std::unique_ptr<BatchRunner> runner, const NetConfig& cfg,
                           InferenceHubOptions options)
    : runner_(std::move(runner)),
      cfg_(cfg),
      maxBatch_(options.maxBatch),
      inputFloats_(static_cast<size_t>(cfg.planes()) * cfg.area()),
      actions_(static_cast<size_t>(cfg.actions())),
      chaos_(options.chaosSeed != 0),
      chaosRng_(options.chaosSeed) {
    if (!runner_ || runner_->slots() < 1) throw std::invalid_argument("InferenceHub: a runner with slots");
    if (maxBatch_ < 1 || maxBatch_ > runner_->maxBatch())
        throw std::invalid_argument("InferenceHub: maxBatch " + std::to_string(maxBatch_) +
                                    " outside [1, " + std::to_string(runner_->maxBatch()) + "]");
    slots_.resize(static_cast<size_t>(runner_->slots()));
    dispatcher_ = std::thread([this] { dispatcherLoop(); });
}

InferenceHub::~InferenceHub() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        stop_ = true;
        dispatcherWake_.notify_all();
        // Threads still in evaluate (a broken contract, but not a crash):
        // the dispatcher fails their queued requests and finishes the ones
        // in flight; wait until the last has left submit, so none touches
        // the mutex after it is gone.
        drained_.wait(lock, [&] { return submitters_ == 0; });
    }
    dispatcher_.join();   // returns once no batch is in flight
}

std::unique_ptr<Evaluator> InferenceHub::makeEvaluator(int maxRequest) {
    if (maxRequest < 1 || maxRequest > maxBatch_)
        throw std::invalid_argument("InferenceHub: requests of " + std::to_string(maxRequest) +
                                    " positions do not fit batches of " + std::to_string(maxBatch_));
    return std::unique_ptr<Evaluator>(new BatchingEvaluator(*this, maxRequest));
}

InferenceHub::Stats InferenceHub::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void InferenceHub::submit(Request& r) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!failed_.empty()) throw std::runtime_error("GPU inference failed: " + failed_);
    if (stop_) throw std::logic_error("InferenceHub: evaluate after shutdown");
    ++submitters_;
    queue_.push_back(&r);
    dispatcherWake_.notify_one();
    r.wake.wait(lock, [&] { return r.done; });
    // The last thread out lets a waiting destructor go on (under the lock,
    // so the destructor cannot free the hub before we let go of it).
    if (--submitters_ == 0 && stop_) drained_.notify_all();
    if (!r.error.empty()) throw std::runtime_error("GPU inference failed: " + r.error);
}

void InferenceHub::fail(Request& r, const std::string& error) {
    r.error = error;
    r.done = true;
    r.wake.notify_one();   // under the lock: the waiter cannot leave (and free r) before we return
}

// A slot's batch is back: copy every request's rows out, wake its caller and
// free the slot. Called and returns with the lock held; drops it for the copies.
void InferenceHub::finish(Slot& slot, int index, std::unique_lock<std::mutex>& lock) {
    const std::string error = slot.error;
    if (error.empty()) {
        ++stats_.batches;
        stats_.positions += slot.positions;
        stats_.gpuSeconds += now() - slot.started;
        lock.unlock();
        const float* logits = runner_->logits(index);
        const float* values = runner_->values(index);
        for (Request* r : slot.requests) {
            std::memcpy(r->logits, logits + r->offset * actions_, r->B * actions_ * sizeof(float));
            std::memcpy(r->values, values + r->offset, r->B * sizeof(float));
        }
        lock.lock();
    } else if (failed_.empty()) {
        failed_ = error;   // from now on every request fails
    }
    for (Request* r : slot.requests) {
        if (error.empty()) {
            r->done = true;
            r->wake.notify_one();
        } else {
            fail(*r, error);
        }
    }
    slot.requests.clear();
    slot.busy = slot.completed = false;
    slot.error.clear();
}

void InferenceHub::dispatcherLoop() {
    std::unique_lock<std::mutex> lock(mutex_);
    auto anyBusy = [&] {
        for (const Slot& s : slots_)
            if (s.busy) return true;
        return false;
    };
    auto anyCompleted = [&] {
        for (const Slot& s : slots_)
            if (s.busy && s.completed) return true;
        return false;
    };
    auto anyFree = [&] {
        for (const Slot& s : slots_)
            if (!s.busy) return true;
        return false;
    };
    for (;;) {
        // 1. Wait until a slot has completed, or a slot is free and requests
        //    wait (or they must be failed: after a failure or once the hub
        //    stops), or the hub stops with nothing in flight.
        dispatcherWake_.wait(lock, [&] {
            return anyCompleted() || (!queue_.empty() && (anyFree() || !failed_.empty() || stop_)) ||
                   (stop_ && !anyBusy());
        });

        // 2. Hand back every completed batch.
        for (size_t i = 0; i < slots_.size(); ++i)
            if (slots_[i].busy && slots_[i].completed) finish(slots_[i], static_cast<int>(i), lock);

        // After a failure, nothing new runs: every waiting request fails too.
        // After shutdown began, neither: the queued requests fail.
        if (!failed_.empty() || stop_) {
            const std::string why = failed_.empty() ? std::string("inference hub shut down") : failed_;
            while (!queue_.empty()) {
                fail(*queue_.front(), why);
                queue_.pop_front();
            }
        }

        // 3. While a slot is free and requests wait: whole requests from the
        //    front of the queue, up to maxBatch positions, into one batch.
        for (size_t i = 0; i < slots_.size() && !queue_.empty() && failed_.empty(); ++i) {
            Slot& slot = slots_[i];
            if (slot.busy) continue;
            int cap = maxBatch_;
            // Chaos: a random cap between the next request and maxBatch.
            if (chaos_) cap = queue_.front()->B + chaosRng_.below(maxBatch_ - queue_.front()->B + 1);
            slot.positions = 0;
            while (!queue_.empty() && slot.positions + queue_.front()->B <= cap) {
                slot.requests.push_back(queue_.front());
                slot.positions += queue_.front()->B;
                queue_.pop_front();
            }
            bool pause = false;
            int pauseMicros = 0;
            if (chaos_) {
                // Chaos: shuffle the requests inside the batch, and sometimes
                // wait a moment before submitting it.
                for (size_t k = slot.requests.size(); k > 1; --k)
                    std::swap(slot.requests[k - 1], slot.requests[chaosRng_.below(static_cast<int>(k))]);
                pause = chaosRng_.below(2) == 0;
                pauseMicros = chaosRng_.below(201);
            }
            int offset = 0;
            for (Request* r : slot.requests) {
                r->offset = offset;
                offset += r->B;
            }
            slot.busy = true;
            slot.completed = false;
            slot.error.clear();
            const int index = static_cast<int>(i);
            lock.unlock();
            std::string error;
            try {
                float* staging = runner_->input(index);
                for (const Request* r : slot.requests)
                    std::memcpy(staging + r->offset * inputFloats_, r->input,
                                r->B * inputFloats_ * sizeof(float));
                if (pause) std::this_thread::sleep_for(std::chrono::microseconds(pauseMicros));
                slot.started = now();
                // The completion callback runs on the runner's thread: it only
                // marks the slot and wakes the dispatcher.
                runner_->begin(index, slot.positions, [this](int s, std::string err) {
                    std::lock_guard<std::mutex> guard(mutex_);
                    slots_[static_cast<size_t>(s)].completed = true;
                    slots_[static_cast<size_t>(s)].error = std::move(err);
                    dispatcherWake_.notify_one();
                });
            } catch (const std::exception& e) {
                error = std::string("runner: ") + e.what();
            } catch (...) {
                error = "runner: unknown exception";
            }
            lock.lock();
            if (!error.empty()) {   // begin never started: the batch fails as if the runner had said so
                slot.completed = true;
                slot.error = error;
            }
        }

        if (stop_ && !anyBusy()) return;   // the queue was emptied above
    }
}

// ---------------------------------------------------------------- WorkerEvaluators

WorkerEvaluators::WorkerEvaluators(const Network& net, const std::string& backend, int workers,
                                   int maxRequest, int inferenceBatch, int slots) {
    if (workers < 1) throw std::invalid_argument("WorkerEvaluators: workers must be >= 1");
    if (backend == "metal") {
#ifdef AZ_USE_METAL
        InferenceHubOptions options;
        options.maxBatch = inferenceBatch;
        options.chaosSeed = metal_probe::hubChaosSeed();   // a test hook; 0 in every run
        hub_ = std::make_unique<InferenceHub>(makeMetalRunner(net, inferenceBatch, slots), net.config(), options);
        for (int w = 0; w < workers; ++w) evaluators_.push_back(hub_->makeEvaluator(maxRequest));
#else
        (void)inferenceBatch;
        (void)slots;
        throw std::invalid_argument(
            "backend 'metal' not available: rebuild with -DAZ_USE_METAL=ON (macOS only)");
#endif
        return;
    }
    // naive and blas: exactly the per-worker evaluators self-play always had.
    for (int w = 0; w < workers; ++w) {
        backends_.push_back(makeBackend(backend, 1));
        auto local = std::make_unique<LocalEvaluator>(net, *backends_.back(), maxRequest);
        local_.push_back(local.get());
        evaluators_.push_back(std::move(local));
    }
}

long long WorkerEvaluators::evaluations() const {
    if (hub_) return hub_->stats().positions;
    long long total = 0;
    for (const LocalEvaluator* e : local_) total += e->evaluations();
    return total;
}

WorkerEvaluators::~WorkerEvaluators() {
    evaluators_.clear();   // before the hub they may point into
}

Evaluator& WorkerEvaluators::forWorker(int worker) {
    return *evaluators_.at(static_cast<size_t>(worker));
}

} // namespace azgo
