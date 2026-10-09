// The network on the GPU, mirroring src/network.cpp: GpuWeights holds the
// parameters and running statistics (one flat arena each, ParamLayout),
// GpuWorkspace one batch's activations under the CPU Workspace's names and
// layouts, and GpuNetwork encodes Network::forward's layers in its order.
//
// Phase A runs inference only: batch norm with running statistics, the
// policy head to its logits, the value head to its tanh and, when the
// network has one, the ownership head to its tanh.
//
// Plain C++; the Metal objects stay inside metal_context.mm.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <string>
#include "azgo/network.hpp"
#include "azgo/param_layout.hpp"
#include "metal_context.hpp"

namespace azgo {

// A network's tensors in GPU memory. The layout is fixed when it is created;
// upload copies a network of that shape in, one memcpy per tensor.
class GpuWeights {
public:
    explicit GpuWeights(const ParamLayout& layout);
    // Copies every parameter and running statistic of net. Throws
    // std::logic_error naming any tensor whose name or size the layout does
    // not know (a layer added on the CPU without a GPU port), or that net lacks.
    void upload(const Network& net);
    BufView param(const std::string& name) const;    // a params() tensor; throws if unknown
    BufView buffer(const std::string& name) const;   // a buffers() tensor; throws if unknown
    const ParamLayout& layout() const { return layout_; }
    size_t bytes() const { return params_.bytes() + buffers_.bytes(); }
private:
    ParamLayout layout_;
    GpuBuffer params_, buffers_;
};

// One batch's activations for up to maxBatch positions, channel-major
// [C][B][area] like the CPU Workspace. Inference keeps nothing for a backward
// pass: the trunk rotates through three buffers (block input, middle
// activation, block output).
class GpuWorkspace {
public:
    enum class Purpose { Inference };
    GpuWorkspace(const NetConfig& cfg, int maxBatch, Purpose purpose);
    const NetConfig& config() const { return cfg_; }
    int maxBatch() const { return maxBatch_; }
    size_t bytes() const;   // GPU memory held
    // What a workspace of this shape holds, without allocating it (for the
    // memory check before a hub is built).
    static size_t bytesFor(const NetConfig& cfg, int maxBatch, Purpose purpose);

    struct {
        GpuBuffer input;    // [B][planes][area]: the batch as the evaluator receives it
    } stage;
    GpuBuffer input;        // [planes][B][area]: transposed
    GpuBuffer col;          // im2col scratch shared by every 3x3 conv
    GpuBuffer convOut;      // a conv's output before its BN
    GpuBuffer trunk[3];     // [filters][B][area], rotated
    GpuBuffer policyAct;    // [2][B][area]
    GpuBuffer policyFlat;   // [B][2 * area]
    GpuBuffer logits;       // [B][actions]
    GpuBuffer valueAct;     // [1][B][area] = [B][area]
    GpuBuffer valueHidden;  // [B][valueHidden]
    GpuBuffer values;       // [B], after tanh
    GpuBuffer ownership;    // [B][area], after tanh (only with the head)

private:
    NetConfig cfg_;
    int maxBatch_;
};

class GpuNetwork {
public:
    explicit GpuNetwork(const NetConfig& cfg);
    const NetConfig& config() const { return cfg_; }
    // Encodes the inference forward of the first B positions of
    // ws.stage.input into batch; the outputs land in ws.logits, ws.values
    // and (with the head) ws.ownership once the batch has run.
    void encodeForward(CommandBatch& batch, GpuWorkspace& ws, const GpuWeights& w, int B) const;
    // Synchronous: copy input in, encode, commit, wait, copy the outputs out.
    // input [B][planes][area], logits [B][actions], values [B]; ownership
    // [B][area] if not null (needs the head).
    void infer(GpuWorkspace& ws, const GpuWeights& w, const float* input, int B, float* logits,
               float* values, float* ownership = nullptr) const;
private:
    NetConfig cfg_;
};

} // namespace azgo
