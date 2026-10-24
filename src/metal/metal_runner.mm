// MetalBatchRunner: the GPU network behind the hub's BatchRunner interface.
// The weights are uploaded once, when the hub is built for a phase; each of
// the two slots owns a whole inference workspace, so one batch can run on the
// GPU while the dispatcher fills the other slot's staging buffer.
//
// begin encodes one forward pass into one command buffer and commits it; the
// completion handler, on a Metal thread, only passes the status on. The
// destructor waits for passes still in flight before freeing their buffers.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <algorithm>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "azgo/inference_hub.hpp"
#include "azgo/metal.hpp"
#include "azgo/param_layout.hpp"
#include "gpu_network.hpp"
#include "metal_context.hpp"

namespace azgo {

namespace {

class MetalBatchRunner : public BatchRunner {
public:
    MetalBatchRunner(const Network& net, int maxBatch, int slots)
        : network_(net.config()),
          weights_(ParamLayout::of(const_cast<Network&>(net))),
          maxBatch_(maxBatch) {
        weights_.upload(net);
        for (int s = 0; s < slots; ++s)
            workspaces_.push_back(std::make_unique<GpuWorkspace>(net.config(), maxBatch,
                                                                 GpuWorkspace::Purpose::Inference));
    }

    ~MetalBatchRunner() override {
        std::unique_lock<std::mutex> lock(mutex_);
        idle_.wait(lock, [&] { return inFlight_ == 0; });
    }

    int slots() const override { return static_cast<int>(workspaces_.size()); }
    int maxBatch() const override { return maxBatch_; }
    float* input(int slot) override { return workspaces_.at(slot)->stage.input.data(); }
    const float* logits(int slot) const override { return workspaces_.at(slot)->logits.data(); }
    const float* values(int slot) const override { return workspaces_.at(slot)->values.data(); }

    void begin(int slot, int B, std::function<void(int slot, std::string error)> done) override {
        GpuWorkspace& ws = *workspaces_.at(slot);
        // An exception (encoding refused) leaves the pool before it is
        // rethrown; the hub turns it into a failed batch.
        runInPool([&] {
            CommandBatch batch;
            network_.encodeForward(batch, ws, weights_, B);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++inFlight_;
            }
            try {
                batch.commit([this, slot, done](const std::string& err) {
                    done(slot, err);
                    std::lock_guard<std::mutex> lock(mutex_);
                    --inFlight_;
                    idle_.notify_all();   // under the lock: the destructor may free us right after
                });
            } catch (...) {
                // Not committed, so no handler will run: undo the count, or
                // the destructor would wait for it forever.
                std::lock_guard<std::mutex> lock(mutex_);
                --inFlight_;
                idle_.notify_all();
                throw;
            }
        });
    }

private:
    GpuNetwork network_;
    GpuWeights weights_;
    int maxBatch_;
    std::vector<std::unique_ptr<GpuWorkspace>> workspaces_;
    std::mutex mutex_;
    std::condition_variable idle_;
    int inFlight_ = 0;
};

} // namespace

namespace {

// GPU memory of one hub: an inference workspace per slot and the weights.
double hubBytes(const NetConfig& cfg, size_t weightFloats, int maxBatch, int slots) {
    return static_cast<double>(slots) *
               GpuWorkspace::bytesFor(cfg, maxBatch, GpuWorkspace::Purpose::Inference) +
           static_cast<double>(weightFloats) * sizeof(float);
}

// Throws unless `need` more bytes fit the GPU's recommended working set
// beside what the process already holds. Allocating past it would not fail,
// only page and crawl.
void checkFits(double need, int maxBatch, int hubs) {
    MetalContext& ctx = MetalContext::instance();
    const double used = static_cast<double>(ctx.currentAllocatedSize());
    const double have = static_cast<double>(ctx.recommendedWorkingSet());
    if (used + need <= have) return;
    throw std::runtime_error(
        "metal: inference needs " + std::to_string(static_cast<long long>(need / 1e6)) +
        " MB of GPU memory at inference_batch " + std::to_string(maxBatch) +
        (hubs > 1 ? " (" + std::to_string(hubs) + " hubs)" : std::string()) + ", which with the " +
        std::to_string(static_cast<long long>(used / 1e6)) + " MB already in use exceeds the GPU's " +
        "recommended working set of " + std::to_string(static_cast<long long>(have / 1e6)) +
        " MB; lower inference_batch");
}

} // namespace

std::unique_ptr<BatchRunner> makeMetalRunner(const Network& net, int maxBatch, int slots) {
    if (maxBatch < 1) throw std::invalid_argument("makeMetalRunner: maxBatch must be >= 1");
    if (slots < 1 || slots > kInferenceSlots)
        throw std::invalid_argument("makeMetalRunner: slots must be in [1, " + std::to_string(kInferenceSlots) + "]");
    const ParamLayout layout = ParamLayout::of(const_cast<Network&>(net));
    // What the process already holds counts too: an arena wave builds a hub
    // for each network.
    checkFits(hubBytes(net.config(), layout.paramFloats + layout.bufferFloats, maxBatch, slots), maxBatch, 1);
    return std::make_unique<MetalBatchRunner>(net, maxBatch, slots);
}

void checkMetalInferenceMemory(const NetConfig& cfg, int inferenceBatch, int hubs, int slotsPerHub) {
    if (inferenceBatch < 1 || hubs < 1 || slotsPerHub < 1)
        throw std::invalid_argument("checkMetalInferenceMemory: bad arguments");
    checkMetalInferenceMemory(cfg, std::vector<GpuHubs>{{hubs, slotsPerHub, inferenceBatch}});
}

void checkMetalInferenceMemory(const NetConfig& cfg, const std::vector<GpuHubs>& groups) {
    if (groups.empty()) throw std::invalid_argument("checkMetalInferenceMemory: no hubs");
    Network shape(cfg, 0);   // only for the tensor sizes
    const ParamLayout layout = ParamLayout::of(shape);
    double need = 0;
    int hubs = 0, batch = 0;
    for (const GpuHubs& g : groups) {
        if (g.inferenceBatch < 1 || g.hubs < 1 || g.slotsPerHub < 1)
            throw std::invalid_argument("checkMetalInferenceMemory: bad arguments");
        need += g.hubs * hubBytes(cfg, layout.paramFloats + layout.bufferFloats, g.inferenceBatch, g.slotsPerHub);
        hubs += g.hubs;
        batch = std::max(batch, g.inferenceBatch);
    }
    checkFits(need, batch, hubs);
}

} // namespace azgo
