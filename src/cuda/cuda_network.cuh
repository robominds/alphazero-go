// The network on an NVIDIA GPU, mirroring src/network.cpp as the Metal port
// (src/metal/gpu_network.hpp) does: CudaWeights holds the parameters and
// running statistics (one flat arena each, ParamLayout) plus each batch
// norm's folded scale and shift; CudaWorkspace one batch's activations,
// channel-major [C][B][area] like the CPU Workspace; CudaNetwork enqueues
// Network::forward's layers, in its order, on a stream.
//
// Inference only: batch norm with running statistics, the policy head to its
// logits, the value head to its tanh and, when the network has one, the
// ownership head to its tanh. Every 3x3 convolution is the fused kernel
// (implicit im2col, batch norm, skip, ReLU), so the workspace holds no
// column matrix.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <map>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "azgo/network.hpp"
#include "azgo/param_layout.hpp"
#include "cuda_context.cuh"

namespace azgo {
namespace cuda {

// A network's tensors in GPU memory. The layout is fixed when it is created;
// upload copies a network of that shape in.
class CudaWeights {
public:
    explicit CudaWeights(const ParamLayout& layout);
    // Copies every parameter and running statistic of net, then folds every
    // batch norm's statistics into a scale and shift (bn_fold), and waits.
    // Throws std::logic_error naming any tensor whose name or size the layout
    // does not know (a layer added on the CPU without a GPU port), or that
    // net lacks.
    void upload(const Network& net);
    // A params() or buffers() tensor; throws std::logic_error if unknown.
    // Each call marks the tensor as read by the GPU network.
    const float* param(const std::string& name) const;
    const float* buffer(const std::string& name) const;
    // The folded scale and shift of the batch norm `name` (e.g. "stem.bn"),
    // [C] each; marks its four tensors as read.
    const float* scale(const std::string& name) const;
    const float* shift(const std::string& name) const;
    // Throws std::logic_error naming every tensor that no call above has
    // asked for. After a whole pass has been enqueued, an unread tensor is a
    // CPU layer the GPU does not evaluate.
    void checkEveryTensorRead() const;
    size_t bytes() const { return params_.bytes() + buffers_.bytes() + folded_.bytes(); }

private:
    ParamLayout layout_;
    DeviceBuffer params_, buffers_, folded_;
    std::map<std::string, size_t> foldedAt_;   // batch norm -> offset of its scale (shift follows)
    std::map<std::string, int> foldedChannels_;
    mutable std::vector<char> paramRead_, bufferRead_;
    void markRead(const std::string& name) const;
};

// One batch's activations for up to maxBatch positions, channel-major
// [C][B][area]. The trunk rotates through three buffers (block input,
// middle activation, block output).
class CudaWorkspace {
public:
    CudaWorkspace(const NetConfig& cfg, int maxBatch);
    const NetConfig& config() const { return cfg_; }
    int maxBatch() const { return maxBatch_; }
    size_t bytes() const;
    // What a workspace of this shape holds, without allocating it.
    static size_t bytesFor(const NetConfig& cfg, int maxBatch);

    DeviceBuffer stage;        // [B][planes][area]: the batch as the evaluator gives it
    DeviceBuffer input;        // [planes][B][area]: transposed
    DeviceBuffer trunk[3];     // [filters][B][area], rotated
    DeviceBuffer head;         // a head's 1x1 convolution before its batch norm, [2][B][area] at most
    DeviceBuffer policyAct;    // [2][B][area]
    DeviceBuffer policyFlat;   // [B][2 * area]
    DeviceBuffer logits;       // [B][actions]
    DeviceBuffer valueAct;     // [B][area]
    DeviceBuffer valueHidden;  // [B][valueHidden]
    DeviceBuffer values;       // [B], after tanh
    DeviceBuffer ownership;    // [B][area], after tanh (only with the head)

private:
    NetConfig cfg_;
    int maxBatch_;
};

class CudaNetwork {
public:
    explicit CudaNetwork(const NetConfig& cfg);
    const NetConfig& config() const { return cfg_; }
    // Enqueues the inference forward of the first B positions of ws.stage on
    // stream; the outputs land in ws.logits, ws.values and (with the head)
    // ws.ownership once the stream reaches them. Throws std::invalid_argument
    // for a batch out of range or another network's workspace, and
    // std::logic_error for an unread tensor (CudaWeights).
    void encodeForward(cudaStream_t stream, CudaWorkspace& ws, const CudaWeights& w, int B) const;

private:
    NetConfig cfg_;
};

} // namespace cuda
} // namespace azgo
