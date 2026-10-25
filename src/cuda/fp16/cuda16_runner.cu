// Cuda16BatchRunner: the 16-bit network behind the hub's BatchRunner
// interface. It is ../cuda_runner.cu's CudaBatchRunner with the network,
// weights and workspace swapped for the cuda16 ones, and the same reasons
// for every choice there: each slot owns a stream, a whole workspace and
// page-locked staging; begin enqueues and returns; one completion thread of
// the runner's own polls each pass's event with short sleeps (never a
// cudaLaunchHostFunc callback, which may not call CUDA and never runs after
// a sticky error) and calls done; the destructor lets that thread report
// every pass in flight before anything is freed.
//
// It is a copy, not a shared template, because the cuda backend's kernel
// hash covers ../cuda_runner.cu: changing it would change the cuda engine's
// fingerprint (docs/superpowers/specs/2026-10-24-cuda-fp16-design.md, "Why
// a subdirectory"). The tests of both runners are the same.
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
#include "azgo/cuda16.hpp"
#include "azgo/inference_hub.hpp"
#include "azgo/param_layout.hpp"
#include "../cuda_context.cuh"
#include "../kernels.cuh"
#include "cuda16_network.cuh"

namespace azgo {

namespace {

// Test hooks (cuda16_probe), as the cuda runner's.
std::atomic<int> gFailCountdown{0};          // 0: off; else passes left before one fails
std::atomic<size_t> gMemoryLimit{0};         // 0: the GPU's free memory
std::atomic<long long> gPassesCompleted{0};
std::atomic<int> gCopyOutDelayUs{0};         // 0: none

constexpr int kPollFirstUs = 50, kPollLastUs = 200;
constexpr size_t kMemoryMargin = size_t(512) << 20;

class Cuda16BatchRunner : public BatchRunner {
public:
    Cuda16BatchRunner(const Network& net, const ParamLayout& layout, int maxBatch, int slots)
        : network_(net.config()), weights_(layout, net.config(), cuda16::Half::Fp16), maxBatch_(maxBatch) {
        static_assert(kCuda16Type == Cuda16Type::Fp16, "the runner's type is the backend's");
        weights_.upload(net);
        const NetConfig& cfg = net.config();
        for (int s = 0; s < slots; ++s) slots_.push_back(std::make_unique<Slot>(cfg, maxBatch));
        completion_ = std::thread([this] { completionLoop(); });
    }

    ~Cuda16BatchRunner() override {
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
        network_.encodeForward(stream, s.workspace, weights_, B);
        keep(cudaGetLastError());
        keep(copyOut(s, B));
        keep(cudaEventRecord(s.event, stream));   // after the copies: done means the outputs are on the host
        if (e != cudaSuccess) error = std::string("cuda16: ") + cudaGetErrorString(e);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(Pass{slot, std::move(done), std::move(error)});
        }
        wake_.notify_all();
    }

private:
    struct Slot;
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
        cuda16::Cuda16Workspace workspace;
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
        std::string error;
    };

    // cudaEventSynchronize would spin a core the tree search needs.
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
            if (error.empty() && e != cudaSuccess) error = std::string("cuda16: ") + cudaGetErrorString(e);
            int left = gFailCountdown.load();
            while (left > 0 && !gFailCountdown.compare_exchange_weak(left, left - 1)) {
            }
            if (left == 1 && error.empty()) error = "cuda16: injected failure (a test hook)";
            ++gPassesCompleted;
            pass.done(pass.slot, error);
        }
    }

    cuda16::Cuda16Network network_;
    cuda16::Cuda16Weights weights_;
    int maxBatch_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<Pass> queue_;
    bool stop_ = false;
    std::thread completion_;
};

// GPU memory of one hub: a workspace per slot, the fp32 weights, and the
// 16-bit trunk weights (at most half the fp32 ones, plus padding: counted
// as the fp32 size again).
double hubBytes(const NetConfig& cfg, size_t weightFloats, int maxBatch, int slots) {
    return static_cast<double>(slots) * cuda16::Cuda16Workspace::bytesFor(cfg, maxBatch) +
           2.0 * static_cast<double>(weightFloats) * sizeof(float);
}

void checkFits(double need, int maxBatch, int hubs) {
    std::string why;
    if (!cuda16Available(&why)) throw std::runtime_error("cuda16: " + why);
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
    throw std::runtime_error("cuda16: inference needs " + std::to_string(static_cast<long long>(need / 1e6)) +
                             " MB of GPU memory at inference_batch " + std::to_string(maxBatch) +
                             (hubs > 1 ? " (" + std::to_string(hubs) + " hubs)" : std::string()) +
                             ", more than the " + std::to_string(static_cast<long long>(have / 1e6)) +
                             " MB the GPU has free; lower inference_batch");
}

} // namespace

std::unique_ptr<BatchRunner> makeCuda16Runner(const Network& net, int maxBatch, int slots) {
    if (maxBatch < 1) throw std::invalid_argument("makeCuda16Runner: maxBatch must be >= 1");
    if (slots < 1 || slots > kInferenceSlots)
        throw std::invalid_argument("makeCuda16Runner: slots must be in [1, " + std::to_string(kInferenceSlots) + "]");
    const ParamLayout layout = ParamLayout::of(const_cast<Network&>(net));
    checkFits(hubBytes(net.config(), layout.paramFloats + layout.bufferFloats, maxBatch, slots), maxBatch, 1);
    return std::make_unique<Cuda16BatchRunner>(net, layout, maxBatch, slots);
}

void checkCuda16InferenceMemory(const NetConfig& cfg, int inferenceBatch, int hubs, int slotsPerHub) {
    if (inferenceBatch < 1 || hubs < 1 || slotsPerHub < 1)
        throw std::invalid_argument("checkCuda16InferenceMemory: bad arguments");
    checkCuda16InferenceMemory(cfg, std::vector<GpuHubs>{{hubs, slotsPerHub, inferenceBatch}});
}

void checkCuda16InferenceMemory(const NetConfig& cfg, const std::vector<GpuHubs>& groups) {
    if (groups.empty()) throw std::invalid_argument("checkCuda16InferenceMemory: no hubs");
    Network shape(cfg, 0);   // only for the tensor sizes
    const ParamLayout layout = ParamLayout::of(shape);
    double need = 0;
    int hubs = 0, batch = 0;
    for (const GpuHubs& g : groups) {
        if (g.inferenceBatch < 1 || g.hubs < 1 || g.slotsPerHub < 1)
            throw std::invalid_argument("checkCuda16InferenceMemory: bad arguments");
        need += g.hubs * hubBytes(cfg, layout.paramFloats + layout.bufferFloats, g.inferenceBatch, g.slotsPerHub);
        hubs += g.hubs;
        batch = std::max(batch, g.inferenceBatch);
    }
    checkFits(need, batch, hubs);
}

namespace cuda16_probe {

void failPassesAfter(int n) { gFailCountdown = n > 0 ? n + 1 : 0; }
void setMemoryLimit(size_t bytes) { gMemoryLimit = bytes; }
long long passesCompleted() { return gPassesCompleted.load(); }
void setCopyOutDelay(int microseconds) { gCopyOutDelayUs = microseconds > 0 ? microseconds : 0; }

} // namespace cuda16_probe

} // namespace azgo
