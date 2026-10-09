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
    MetalBatchRunner(const Network& net, int maxBatch)
        : network_(net.config()),
          weights_(ParamLayout::of(const_cast<Network&>(net))),
          maxBatch_(maxBatch) {
        weights_.upload(net);
        for (int s = 0; s < kInferenceSlots; ++s)
            workspaces_.push_back(std::make_unique<GpuWorkspace>(net.config(), maxBatch,
                                                                 GpuWorkspace::Purpose::Inference));
    }

    ~MetalBatchRunner() override {
        std::unique_lock<std::mutex> lock(mutex_);
        idle_.wait(lock, [&] { return inFlight_ == 0; });
    }

    int slots() const override { return kInferenceSlots; }
    int maxBatch() const override { return maxBatch_; }
    float* input(int slot) override { return workspaces_.at(slot)->stage.input.data(); }
    const float* logits(int slot) const override { return workspaces_.at(slot)->logits.data(); }
    const float* values(int slot) const override { return workspaces_.at(slot)->values.data(); }

    void begin(int slot, int B, std::function<void(int slot, std::string error)> done) override {
        GpuWorkspace& ws = *workspaces_.at(slot);
        std::string error;
        @autoreleasepool {
            try {
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
                    std::lock_guard<std::mutex> lock(mutex_);
                    --inFlight_;   // never committed: no handler will run
                    throw;
                }
            } catch (const std::exception& e) {
                error = e.what();
            }
        }
        // Thrown after the pool: the hub turns it into a failed batch.
        if (!error.empty()) throw std::runtime_error(error);
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

std::unique_ptr<BatchRunner> makeMetalRunner(const Network& net, int maxBatch) {
    if (maxBatch < 1) throw std::invalid_argument("makeMetalRunner: maxBatch must be >= 1");
    MetalContext& ctx = MetalContext::instance();
    // Check the memory first: allocating past the working set would not
    // fail, only page and crawl.
    const NetConfig& cfg = net.config();
    const ParamLayout layout = ParamLayout::of(const_cast<Network&>(net));
    const double need = static_cast<double>(kInferenceSlots) *
                            GpuWorkspace::bytesFor(cfg, maxBatch, GpuWorkspace::Purpose::Inference) +
                        static_cast<double>(layout.paramFloats + layout.bufferFloats) * sizeof(float);
    const double have = static_cast<double>(ctx.recommendedWorkingSet());
    if (need > have)
        throw std::runtime_error("metal: inference needs " +
                                 std::to_string(static_cast<long long>(need / 1e6)) + " MB of GPU memory at inference_batch " + std::to_string(maxBatch) +
                                 ", more than the GPU's recommended working set of " +
                                 std::to_string(static_cast<long long>(have / 1e6)) +
                                 " MB; lower inference_batch");
    return std::make_unique<MetalBatchRunner>(net, maxBatch);
}

} // namespace azgo
