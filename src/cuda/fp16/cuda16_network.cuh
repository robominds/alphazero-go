// The network for backend = cuda16, beside ../cuda_network.cuh's: the same
// layers in the same order, with the residual trunk's 3x3 convolutions on the
// tensor cores (kernels16.cuh) and the heads on the cuda backend's fp32
// kernels.
//
// Cuda16Weights holds a cuda::CudaWeights (every fp32 tensor and folded
// batch norm, for the heads, and the record of which tensors the pass read)
// plus each trunk convolution in 16 bits with its folded scale and shift
// padded to its channel tile. Cuda16Workspace holds one batch's activations:
// NHWC 16-bit through the trunk, fp32 channel-major from the trunk's last
// layer on, as the cuda backend's. Cuda16Network enqueues a pass.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <map>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "azgo/network.hpp"
#include "azgo/param_layout.hpp"
#include "../cuda_context.cuh"
#include "../cuda_network.cuh"
#include "kernels16.cuh"

namespace azgo {
namespace cuda16 {

// Device memory for `count` 16-bit values, freed with the object.
class HalfBuffer {
public:
    HalfBuffer() = default;
    explicit HalfBuffer(size_t count) : raw_((count + 1) / 2), count_(count) {}
    void* data() const { return raw_.data(); }
    size_t count() const { return count_; }
    size_t bytes() const { return raw_.bytes(); }
private:
    cuda::DeviceBuffer raw_;
    size_t count_ = 0;
};

// One trunk convolution on the GPU: w16 [coutPadded][9 * cinPadded] and the
// folded scale and shift [coutPadded] (0 past cout).
struct Conv16Weights {
    int cin = 0, cout = 0, cinPadded = 0, coutPadded = 0;
    HalfBuffer w16;
    cuda::DeviceBuffer scaleShift;   // scale, then shift
    // Converts w [cout][cin][3][3] and the batch norm's statistics, on
    // stream s; the host arrays must stay alive until s has run (the caller
    // synchronizes). staging receives the fp32 copies the kernels read.
    void upload(Half type, const float* w, const float* gamma, const float* beta, const float* mean,
                const float* var, int cin_, int cout_, std::vector<cuda::DeviceBuffer>& staging, cudaStream_t s);
    const float* scale() const { return scaleShift.data(); }
    const float* shift() const { return scaleShift.data() + coutPadded; }
};

class Cuda16Weights {
public:
    Cuda16Weights(const ParamLayout& layout, const NetConfig& cfg, Half type);
    // Uploads every tensor as cuda::CudaWeights::upload does (and with its
    // refusals), then converts the stem's and every block's convolutions.
    void upload(const Network& net);
    Half type() const { return type_; }
    // The fp32 tensors and folded batch norms (the heads), as CudaWeights.
    const cuda::CudaWeights& fp32() const { return fp32_; }
    // The trunk convolution `conv` (e.g. "block3.conv2") with its batch norm
    // `bn`; marks the fp32 tensors it was made from as read.
    const Conv16Weights& conv(const std::string& conv, const std::string& bn) const;
    size_t bytes() const;

private:
    NetConfig cfg_;
    Half type_;
    cuda::CudaWeights fp32_;
    std::map<std::string, Conv16Weights> convs_;
};

// One batch's activations for up to maxBatch positions.
class Cuda16Workspace {
public:
    Cuda16Workspace(const NetConfig& cfg, int maxBatch);
    const NetConfig& config() const { return cfg_; }
    int maxBatch() const { return maxBatch_; }
    size_t bytes() const;
    static size_t bytesFor(const NetConfig& cfg, int maxBatch);

    cuda::DeviceBuffer stage;     // [B][planes][area] fp32: the batch as the evaluator gives it
    HalfBuffer input;             // [B * area][planes padded], 16-bit
    HalfBuffer trunk[3];          // [B * area][filters padded], 16-bit, rotated
    cuda::DeviceBuffer features;  // [filters][B * area] fp32: the trunk's output, the heads' input
    cuda::DeviceBuffer head;      // as cuda::CudaWorkspace's head buffers
    cuda::DeviceBuffer policyAct, policyFlat, logits, valueAct, valueHidden, values, ownership;

private:
    NetConfig cfg_;
    int maxBatch_;
};

class Cuda16Network {
public:
    explicit Cuda16Network(const NetConfig& cfg);
    const NetConfig& config() const { return cfg_; }
    // Enqueues the inference forward of the first B positions of ws.stage,
    // as cuda::CudaNetwork::encodeForward (same outputs, same refusals).
    void encodeForward(cudaStream_t stream, Cuda16Workspace& ws, const Cuda16Weights& w, int B) const;

private:
    NetConfig cfg_;
};

} // namespace cuda16
} // namespace azgo
