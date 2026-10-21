// CudaBatchRunner: the GPU network behind the hub's BatchRunner interface,
// as MetalBatchRunner is on a Mac. The weights are uploaded once, when the
// hub is built for a phase. Each slot owns a stream, a whole inference
// workspace and page-locked staging for its input and outputs, so one batch
// runs on the GPU while the dispatcher fills the other slot.
//
// begin enqueues a pass (copy in, the forward pass, copy out, an event) and
// returns at once. One completion thread of the runner's own waits for each
// pass's event, in order, polling it with short sleeps, and calls done. It is not a cudaLaunchHostFunc
// callback: those may not call CUDA, and they never run at all once the
// context has a sticky error, which would leave the hub's callers waiting
// forever. The destructor lets the thread finish every pass in flight, and
// only then frees memory those passes use.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "azgo/cuda.hpp"
#include "azgo/inference_hub.hpp"
#include "azgo/param_layout.hpp"
#include "cuda_context.cuh"
#include "cuda_network.cuh"
#include "kernels.cuh"

namespace azgo {

namespace {

// Test hooks (cuda_probe): the hub chaos seed WorkerEvaluators reads, failure
// injection, a stand-in memory limit and a count of finished passes.
std::atomic<uint64_t> gHubChaosSeed{0};
std::atomic<int> gFailCountdown{0};          // 0: off; else passes left before one fails
std::atomic<size_t> gMemoryLimit{0};         // 0: the GPU's free memory
std::atomic<long long> gPassesCompleted{0};
std::atomic<int> gCopyOutDelayUs{0};          // 0: none

// The completion thread's polling interval: first and longest (waitFor).
constexpr int kPollFirstUs = 50, kPollLastUs = 200;

// The memory a check leaves free on the GPU for everything else.
constexpr size_t kMemoryMargin = size_t(512) << 20;

class CudaBatchRunner : public BatchRunner {
public:
    CudaBatchRunner(const Network& net, int maxBatch, int slots)
        : network_(net.config()), weights_(ParamLayout::of(const_cast<Network&>(net))), maxBatch_(maxBatch) {
        weights_.upload(net);
        const NetConfig& cfg = net.config();
        for (int s = 0; s < slots; ++s) slots_.push_back(std::make_unique<Slot>(cfg, maxBatch));
        completion_ = std::thread([this] { completionLoop(); });
    }

    ~CudaBatchRunner() override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        completion_.join();   // after every pass in flight has been reported
    }

    int slots() const override { return static_cast<int>(slots_.size()); }
    int maxBatch() const override { return maxBatch_; }
    float* input(int slot) override { return slots_.at(slot)->input.data(); }
    const float* logits(int slot) const override { return slots_.at(slot)->logits.data(); }
    const float* values(int slot) const override { return slots_.at(slot)->values.data(); }

    void begin(int slot, int B, std::function<void(int slot, std::string error)> done) override {
        Slot& s = *slots_.at(slot);
        const NetConfig& cfg = network_.config();
        const cudaStream_t stream = s.stream.get();
        const size_t in = static_cast<size_t>(B) * cfg.planes() * cfg.area();
        std::string error;
        cudaError_t e = cudaSuccess;
        auto keep = [&](cudaError_t r) {
            if (e == cudaSuccess) e = r;
        };
        if (B >= 1 && B <= maxBatch_)   // else encodeForward throws
            keep(cudaMemcpyAsync(s.workspace.stage.data(), s.input.data(), in * sizeof(float), cudaMemcpyHostToDevice,
                                 stream));
        // An exception (a batch out of range, an unread tensor) leaves begin;
        // the hub turns it into a failed batch.
        network_.encodeForward(stream, s.workspace, weights_, B);
        keep(cudaGetLastError());
        keep(copyOut(s, B));
        // Recorded after the copies: done means the outputs are on the host.
        keep(cudaEventRecord(s.event, stream));
        if (e != cudaSuccess) error = std::string("cuda: ") + cudaGetErrorString(e);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(Pass{slot, std::move(done), std::move(error)});
        }
        wake_.notify_all();
    }

private:
    struct Slot;
    // The outputs to the slot's page-locked staging, after the test hook's
    // delay (cuda_probe::setCopyOutDelay) when one is set.
    cudaError_t copyOut(Slot& s, int B) {
        const NetConfig& cfg = network_.config();
        const cudaStream_t stream = s.stream.get();
        const int delayUs = gCopyOutDelayUs.load();
        if (delayUs > 0) cuda::launchSpin(1000ull * delayUs, stream);
        cudaError_t e = cudaMemcpyAsync(s.logits.data(), s.workspace.logits.data(),
                                        static_cast<size_t>(B) * cfg.actions() * sizeof(float),
                                        cudaMemcpyDeviceToHost, stream);
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(s.values.data(), s.workspace.values.data(), static_cast<size_t>(B) * sizeof(float),
                                cudaMemcpyDeviceToHost, stream);
        return e;
    }

    struct Slot {
        cuda::CudaWorkspace workspace;
        cuda::Stream stream;
        cuda::PinnedBuffer input, logits, values;
        cudaEvent_t event = nullptr;
        Slot(const NetConfig& cfg, int maxBatch)
            : workspace(cfg, maxBatch),
              input(static_cast<size_t>(maxBatch) * cfg.planes() * cfg.area()),
              logits(static_cast<size_t>(maxBatch) * cfg.actions()),
              values(static_cast<size_t>(maxBatch)) {
            cuda::check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreate");
        }
        ~Slot() {
            if (event) cudaEventDestroy(event);
        }
    };
    struct Pass {
        int slot;
        std::function<void(int, std::string)> done;
        std::string error;   // from enqueueing it
    };

    // Waits for the event by polling it with short sleeps: 50 us, doubling
    // to 200 us. cudaEventSynchronize spins a whole core while it waits (on
    // WSL2 the blocking-sync flags change nothing), and that core is one the
    // tree search needs. A pass of 256 positions takes ~8 ms, so the sleeps
    // add at most 200 us to it.
    static cudaError_t waitFor(cudaEvent_t event) {
        int sleepUs = kPollFirstUs;
        for (;;) {
            const cudaError_t e = cudaEventQuery(event);
            if (e != cudaErrorNotReady) return e;
            std::this_thread::sleep_for(std::chrono::microseconds(sleepUs));
            sleepUs = std::min(kPollLastUs, 2 * sleepUs);
        }
    }

    void completionLoop() {
        for (;;) {
            Pass pass;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) return;   // stop_, and nothing in flight
                pass = std::move(queue_.front());
                queue_.pop_front();
            }
            std::string error = pass.error;
            const cudaError_t e = waitFor(slots_[pass.slot]->event);
            if (error.empty() && e != cudaSuccess) error = std::string("cuda: ") + cudaGetErrorString(e);
            // Failure injection: after n passes, the next one fails (once).
            int left = gFailCountdown.load();
            while (left > 0 && !gFailCountdown.compare_exchange_weak(left, left - 1)) {
            }
            if (left == 1 && error.empty()) error = "cuda: injected failure (a test hook)";
            ++gPassesCompleted;
            pass.done(pass.slot, error);
        }
    }

    cuda::CudaNetwork network_;
    cuda::CudaWeights weights_;
    int maxBatch_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<Pass> queue_;
    bool stop_ = false;
    std::thread completion_;
};

// GPU memory of one hub: an inference workspace per slot and the weights
// (with each batch norm's folded scale and shift, a few KB).
double hubBytes(const NetConfig& cfg, size_t weightFloats, int maxBatch, int slots) {
    return static_cast<double>(slots) * cuda::CudaWorkspace::bytesFor(cfg, maxBatch) +
           static_cast<double>(weightFloats) * sizeof(float);
}

// Throws unless `need` more bytes fit in the GPU's free memory, less a
// margin for the runtime and the rest of the machine (or in the test hook's
// limit instead).
void checkFits(double need, int maxBatch, int hubs) {
    cuda::Context::instance();
    double have;
    const size_t limit = gMemoryLimit.load();
    if (limit > 0) {
        have = static_cast<double>(limit);
    } else {
        size_t freeBytes = 0, total = 0;
        cuda::check(cudaMemGetInfo(&freeBytes, &total), "cudaMemGetInfo");
        have = freeBytes > kMemoryMargin ? static_cast<double>(freeBytes - kMemoryMargin) : 0.0;
    }
    if (need <= have) return;
    throw std::runtime_error("cuda: inference needs " + std::to_string(static_cast<long long>(need / 1e6)) +
                             " MB of GPU memory at inference_batch " + std::to_string(maxBatch) +
                             (hubs > 1 ? " (" + std::to_string(hubs) + " hubs)" : std::string()) +
                             ", more than the " + std::to_string(static_cast<long long>(have / 1e6)) +
                             " MB the GPU has free; lower inference_batch");
}

} // namespace

std::unique_ptr<BatchRunner> makeCudaRunner(const Network& net, int maxBatch, int slots) {
    if (maxBatch < 1) throw std::invalid_argument("makeCudaRunner: maxBatch must be >= 1");
    if (slots < 1 || slots > kInferenceSlots)
        throw std::invalid_argument("makeCudaRunner: slots must be in [1, " + std::to_string(kInferenceSlots) + "]");
    const ParamLayout layout = ParamLayout::of(const_cast<Network&>(net));
    checkFits(hubBytes(net.config(), layout.paramFloats + layout.bufferFloats, maxBatch, slots), maxBatch, 1);
    return std::make_unique<CudaBatchRunner>(net, maxBatch, slots);
}

void checkCudaInferenceMemory(const NetConfig& cfg, int inferenceBatch, int hubs, int slotsPerHub) {
    if (inferenceBatch < 1 || hubs < 1 || slotsPerHub < 1)
        throw std::invalid_argument("checkCudaInferenceMemory: bad arguments");
    Network shape(cfg, 0);   // only for the tensor sizes
    const ParamLayout layout = ParamLayout::of(shape);
    checkFits(hubs * hubBytes(cfg, layout.paramFloats + layout.bufferFloats, inferenceBatch, slotsPerHub),
              inferenceBatch, hubs);
}

namespace cuda_probe {

void setHubChaosSeed(uint64_t seed) { gHubChaosSeed = seed; }
uint64_t hubChaosSeed() { return gHubChaosSeed.load(); }
void failPassesAfter(int n) { gFailCountdown = n > 0 ? n + 1 : 0; }
void setMemoryLimit(size_t bytes) { gMemoryLimit = bytes; }
long long passesCompleted() { return gPassesCompleted.load(); }
void setCopyOutDelay(int microseconds) { gCopyOutDelayUs = microseconds > 0 ? microseconds : 0; }

} // namespace cuda_probe

} // namespace azgo
