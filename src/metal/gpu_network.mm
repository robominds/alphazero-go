// The GPU network (gpu_network.hpp). encodeForward follows
// Network::forward (src/network.cpp) line by line, with its comments, in
// inference mode only: every layer appends its kernels to one command batch,
// so a whole pass is a single trip to the GPU.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include "gpu_network.hpp"

#include <cstring>
#include <stdexcept>

#include "gpu_layers.hpp"

namespace azgo {

// ---------------------------------------------------------------- GpuWeights

GpuWeights::GpuWeights(const ParamLayout& layout)
    : layout_(layout),
      params_(MetalContext::instance().newBuffer(layout.paramFloats * sizeof(float))),
      buffers_(MetalContext::instance().newBuffer(layout.bufferFloats * sizeof(float))) {}

namespace {

// Copies each view into its arena after checking it against the layout's
// entry at the same position.
template <class View>
void copyChecked(const std::vector<View>& views, const std::vector<ParamLayout::Entry>& entries,
                 const ParamLayout& layout, float* arena) {
    for (size_t i = 0; i < views.size(); ++i) {
        const View& v = views[i];
        if (i >= entries.size() || entries[i].name != v.name) {
            if (!layout.find(v.name))
                throw std::logic_error("GPU network: unknown tensor '" + v.name +
                                       "' (a layer without a GPU port?)");
            throw std::logic_error("GPU network: tensor '" + v.name + "' out of the layout's order");
        }
        if (entries[i].size != v.size)
            throw std::logic_error("GPU network: tensor '" + v.name + "' has " + std::to_string(v.size) +
                                   " floats, the layout " + std::to_string(entries[i].size));
        std::memcpy(arena + entries[i].offset, v.value, v.size * sizeof(float));
    }
    if (views.size() < entries.size())
        throw std::logic_error("GPU network: the network lacks tensor '" + entries[views.size()].name +
                               "'");
}

} // namespace

void GpuWeights::upload(const Network& net) {
    // params() and buffers() are non-const only because they hand out
    // writable views; this reads them.
    Network& n = const_cast<Network&>(net);
    copyChecked(n.params(), layout_.params, layout_, params_.data());
    copyChecked(n.buffers(), layout_.buffers, layout_, buffers_.data());
}

BufView GpuWeights::param(const std::string& name) const {
    for (const ParamLayout::Entry& e : layout_.params)
        if (e.name == name) return BufView(params_, e.offset * sizeof(float));
    throw std::logic_error("GPU network: no parameter '" + name + "'");
}

BufView GpuWeights::buffer(const std::string& name) const {
    for (const ParamLayout::Entry& e : layout_.buffers)
        if (e.name == name) return BufView(buffers_, e.offset * sizeof(float));
    throw std::logic_error("GPU network: no buffer '" + name + "'");
}

// ---------------------------------------------------------------- GpuWorkspace

GpuWorkspace::GpuWorkspace(const NetConfig& cfg, int maxBatch, Purpose purpose)
    : cfg_(cfg), maxBatch_(maxBatch) {
    (void)purpose;   // Phase A has inference workspaces only
    if (maxBatch < 1) throw std::invalid_argument("GpuWorkspace: maxBatch must be >= 1");
    MetalContext& ctx = MetalContext::instance();
    auto floats = [&ctx](size_t count) { return ctx.newBuffer(count * sizeof(float)); };
    const size_t M = static_cast<size_t>(maxBatch) * cfg.area();   // pixels per channel
    const size_t F = static_cast<size_t>(cfg.filters);
    stage.input = floats(static_cast<size_t>(cfg.planes()) * M);
    input = floats(static_cast<size_t>(cfg.planes()) * M);
    // One im2col buffer serves every 3x3 conv (stem: planes inputs, blocks: F inputs).
    col = floats(static_cast<size_t>(std::max(cfg.planes(), cfg.filters)) * 9 * M);
    convOut = floats(std::max<size_t>(F, 2) * M);   // the policy conv has 2 outputs even if F == 1
    for (GpuBuffer& t : trunk) t = floats(F * M);
    policyAct = floats(2 * M);
    policyFlat = floats(2 * M);
    logits = floats(static_cast<size_t>(maxBatch) * cfg.actions());
    valueAct = floats(M);
    valueHidden = floats(static_cast<size_t>(maxBatch) * cfg.valueHidden);
    values = floats(static_cast<size_t>(maxBatch));
    if (cfg.ownership) ownership = floats(M);
}

size_t GpuWorkspace::bytes() const {
    size_t total = stage.input.bytes() + input.bytes() + col.bytes() + convOut.bytes() + policyAct.bytes() +
                   policyFlat.bytes() + logits.bytes() + valueAct.bytes() + valueHidden.bytes() +
                   values.bytes() + ownership.bytes();
    for (const GpuBuffer& t : trunk) total += t.bytes();
    return total;
}

// ---------------------------------------------------------------- GpuNetwork

GpuNetwork::GpuNetwork(const NetConfig& cfg) : cfg_(cfg) {}

void GpuNetwork::encodeForward(CommandBatch& batch, GpuWorkspace& ws, const GpuWeights& w, int B) const {
    if (B < 1 || B > ws.maxBatch()) throw std::invalid_argument("GpuNetwork: batch out of range");
    if (ws.config() != cfg_) throw std::invalid_argument("GpuNetwork: workspace of another network");
    const int n = cfg_.boardSize, A = cfg_.area(), P = cfg_.planes(), F = cfg_.filters;
    const int M = B * A;
    const size_t trunk = static_cast<size_t>(F) * M;

    // [B][planes][A] -> [planes][B][A]. Channel-major lets a conv treat the
    // whole batch as one wide matrix: one gemm per layer, not one per sample.
    encodeTransposeInput(batch, ws.stage.input, B, P, A, ws.input);

    auto conv = [&](const std::string& name, int k, int cin, int cout, BufView x, BufView y) {
        encodeConvForward(batch, k, cin, cout, w.param(name + ".w"), x, B, n, ws.col, y);
    };
    // Inference batch norm: the running statistics, as BatchNorm2d::forwardInfer.
    auto norm = [&](const std::string& name, int C, BufView x, BufView y) {
        encodeBnInfer(batch, C, M, w.param(name + ".gamma"), w.param(name + ".beta"),
                      w.buffer(name + ".runMean"), w.buffer(name + ".runVar"), x, y);
    };

    // Stem: conv3x3 -> BN -> ReLU.
    conv("stem.conv", 3, P, F, ws.input, ws.convOut);
    norm("stem.bn", F, ws.convOut, ws.trunk[0]);
    encodeRelu(batch, ws.trunk[0], trunk);

    // Residual blocks: conv -> BN -> ReLU -> conv -> BN -> + input -> ReLU.
    // The trunk rotates through three buffers: the block's input x, its
    // middle activation and its output, which is the next block's input.
    int x = 0;
    for (int i = 0; i < cfg_.blocks; ++i) {
        const std::string prefix = "block" + std::to_string(i);
        const int mid = (x + 1) % 3, out = (x + 2) % 3;
        conv(prefix + ".conv1", 3, F, F, ws.trunk[x], ws.convOut);
        norm(prefix + ".bn1", F, ws.convOut, ws.trunk[mid]);
        encodeRelu(batch, ws.trunk[mid], trunk);
        conv(prefix + ".conv2", 3, F, F, ws.trunk[mid], ws.convOut);
        norm(prefix + ".bn2", F, ws.convOut, ws.trunk[out]);
        encodeAddInto(batch, ws.trunk[out], ws.trunk[x], trunk);   // the skip connection
        encodeRelu(batch, ws.trunk[out], trunk);
        x = out;
    }
    const BufView features = ws.trunk[x];

    // Policy head: conv1x1 (F -> 2) -> BN -> ReLU -> per-sample flatten -> Linear.
    conv("policy.conv", 1, F, 2, features, ws.convOut);
    norm("policy.bn", 2, ws.convOut, ws.policyAct);
    encodeRelu(batch, ws.policyAct, 2 * static_cast<size_t>(M));
    // [2][B][A] -> [B][2A], feature index c * A + p.
    encodePolicyFlatten(batch, ws.policyAct, B, A, ws.policyFlat);
    encodeLinearForward(batch, 2 * A, cfg_.actions(), w.param("policy.fc.w"), w.param("policy.fc.b"),
                        ws.policyFlat, B, ws.logits);

    // Value head: conv1x1 (F -> 1) -> BN -> ReLU -> Linear(A -> V) -> ReLU -> Linear(V -> 1) -> tanh.
    conv("value.conv", 1, F, 1, features, ws.convOut);
    norm("value.bn", 1, ws.convOut, ws.valueAct);
    encodeRelu(batch, ws.valueAct, static_cast<size_t>(M));
    // With a single channel, [1][B][A] already is [B][A]: one row per sample.
    encodeLinearForward(batch, A, cfg_.valueHidden, w.param("value.fc1.w"), w.param("value.fc1.b"),
                        ws.valueAct, B, ws.valueHidden);
    encodeRelu(batch, ws.valueHidden, static_cast<size_t>(B) * cfg_.valueHidden);
    encodeLinearForward(batch, cfg_.valueHidden, 1, w.param("value.fc2.w"), w.param("value.fc2.b"),
                        ws.valueHidden, B, ws.values);
    encodeTanh(batch, ws.values, static_cast<size_t>(B));

    // Ownership head: conv1x1 (F -> 1) -> BN -> tanh at every point, no
    // ReLU and no linear layer: each point's owner is read off the trunk's
    // features at that point. [1][B][A] already is [B][A].
    if (cfg_.ownership) {
        conv("ownership.conv", 1, F, 1, features, ws.convOut);
        norm("ownership.bn", 1, ws.convOut, ws.ownership);
        encodeTanh(batch, ws.ownership, static_cast<size_t>(M));
    }
}

void GpuNetwork::infer(GpuWorkspace& ws, const GpuWeights& w, const float* input, int B, float* logits,
                       float* values, float* ownership) const {
    if (ownership && !cfg_.ownership) throw std::invalid_argument("GpuNetwork: no ownership head");
    @autoreleasepool {
        const size_t inFloats = static_cast<size_t>(B) * cfg_.planes() * cfg_.area();
        if (B >= 1 && B <= ws.maxBatch())   // else encodeForward throws
            std::memcpy(ws.stage.input.data(), input, inFloats * sizeof(float));
        CommandBatch batch;
        encodeForward(batch, ws, w, B);
        batch.commitAndWait();
        std::memcpy(logits, ws.logits.data(), static_cast<size_t>(B) * cfg_.actions() * sizeof(float));
        std::memcpy(values, ws.values.data(), static_cast<size_t>(B) * sizeof(float));
        if (ownership)
            std::memcpy(ownership, ws.ownership.data(),
                        static_cast<size_t>(B) * cfg_.area() * sizeof(float));
    }
}

} // namespace azgo
