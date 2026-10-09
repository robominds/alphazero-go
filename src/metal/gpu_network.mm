// The GPU network (gpu_network.hpp). encodeForward follows
// Network::forward (src/network.cpp) line by line, with its comments, in
// inference mode only: every layer appends its kernels to one command batch,
// so a whole pass is a single trip to the GPU.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include "gpu_network.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "azgo/metal.hpp"
#include "azgo/validation.hpp"
#include "gpu_layers.hpp"

namespace azgo {

// ---------------------------------------------------------------- GpuWeights

GpuWeights::GpuWeights(const ParamLayout& layout)
    : layout_(layout),
      params_(MetalContext::instance().newBuffer(layout.paramFloats * sizeof(float))),
      buffers_(MetalContext::instance().newBuffer(layout.bufferFloats * sizeof(float))),
      paramRead_(layout.params.size(), 0),
      bufferRead_(layout.buffers.size(), 0) {}

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
    for (size_t i = 0; i < layout_.params.size(); ++i) {
        if (layout_.params[i].name != name) continue;
        paramRead_[i] = 1;
        return BufView(params_, layout_.params[i].offset * sizeof(float));
    }
    throw std::logic_error("GPU network: no parameter '" + name + "'");
}

BufView GpuWeights::buffer(const std::string& name) const {
    for (size_t i = 0; i < layout_.buffers.size(); ++i) {
        if (layout_.buffers[i].name != name) continue;
        bufferRead_[i] = 1;
        return BufView(buffers_, layout_.buffers[i].offset * sizeof(float));
    }
    throw std::logic_error("GPU network: no buffer '" + name + "'");
}

void GpuWeights::checkEveryTensorRead() const {
    std::string unread;
    for (size_t i = 0; i < layout_.params.size(); ++i)
        if (!paramRead_[i]) unread += (unread.empty() ? "" : ", ") + layout_.params[i].name;
    for (size_t i = 0; i < layout_.buffers.size(); ++i)
        if (!bufferRead_[i]) unread += (unread.empty() ? "" : ", ") + layout_.buffers[i].name;
    if (!unread.empty())
        throw std::logic_error("GPU network: the forward pass reads no tensor " + unread +
                               " (a layer without a GPU port?)");
}

// ---------------------------------------------------------------- GpuWorkspace

namespace {

// The floats of each inference buffer, for up to maxBatch positions.
struct WorkspaceSizes {
    size_t stageInput, input, col, convOut, trunk, policyAct, policyFlat, logits, valueAct, valueHidden,
        values, ownership;
    size_t total() const {
        return stageInput + input + col + convOut + 3 * trunk + policyAct + policyFlat + logits + valueAct +
               valueHidden + values + ownership;
    }
};

WorkspaceSizes workspaceSizes(const NetConfig& cfg, int maxBatch) {
    const size_t M = static_cast<size_t>(maxBatch) * cfg.area();   // pixels per channel
    const size_t F = static_cast<size_t>(cfg.filters);
    WorkspaceSizes z;
    z.stageInput = z.input = static_cast<size_t>(cfg.planes()) * M;
    // One im2col buffer serves every 3x3 conv (stem: planes inputs, blocks: F inputs).
    z.col = static_cast<size_t>(std::max(cfg.planes(), cfg.filters)) * 9 * M;
    z.convOut = std::max<size_t>(F, 2) * M;   // the policy conv has 2 outputs even if F == 1
    z.trunk = F * M;
    z.policyAct = z.policyFlat = 2 * M;
    z.logits = static_cast<size_t>(maxBatch) * cfg.actions();
    z.valueAct = M;
    z.valueHidden = static_cast<size_t>(maxBatch) * cfg.valueHidden;
    z.values = static_cast<size_t>(maxBatch);
    z.ownership = cfg.ownership ? M : 0;
    return z;
}

} // namespace

GpuWorkspace::GpuWorkspace(const NetConfig& cfg, int maxBatch, Purpose purpose)
    : cfg_(cfg), maxBatch_(maxBatch) {
    (void)purpose;   // Phase A has inference workspaces only
    if (maxBatch < 1) throw std::invalid_argument("GpuWorkspace: maxBatch must be >= 1");
    MetalContext& ctx = MetalContext::instance();
    auto floats = [&ctx](size_t count) { return ctx.newBuffer(count * sizeof(float)); };
    const WorkspaceSizes z = workspaceSizes(cfg, maxBatch);
    stage.input = floats(z.stageInput);
    input = floats(z.input);
    col = floats(z.col);
    convOut = floats(z.convOut);
    for (GpuBuffer& t : trunk) t = floats(z.trunk);
    policyAct = floats(z.policyAct);
    policyFlat = floats(z.policyFlat);
    logits = floats(z.logits);
    valueAct = floats(z.valueAct);
    valueHidden = floats(z.valueHidden);
    values = floats(z.values);
    if (cfg.ownership) ownership = floats(z.ownership);
}

size_t GpuWorkspace::bytesFor(const NetConfig& cfg, int maxBatch, Purpose purpose) {
    (void)purpose;
    return workspaceSizes(cfg, maxBatch).total() * sizeof(float);
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
    // Every tensor the network uploaded must have been read by some layer
    // above, or the GPU is evaluating a different network than the CPU's.
    w.checkEveryTensorRead();
}

void GpuNetwork::infer(GpuWorkspace& ws, const GpuWeights& w, const float* input, int B, float* logits,
                       float* values, float* ownership) const {
    if (ownership && !cfg_.ownership) throw std::invalid_argument("GpuNetwork: no ownership head");
    runInPool([&] {
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
    });
}

// ---------------------------------------------------------------- MetalInference

struct MetalInference::Impl {
    GpuNetwork network;
    GpuWeights weights;
    GpuWorkspace workspace;
    std::vector<float> logits, values, ownership;   // one batch's outputs on the CPU
    Impl(const Network& net, int maxBatch)
        : network(net.config()),
          weights(ParamLayout::of(const_cast<Network&>(net))),
          workspace(net.config(), maxBatch, GpuWorkspace::Purpose::Inference),
          logits(static_cast<size_t>(maxBatch) * net.config().actions()),
          values(static_cast<size_t>(maxBatch)),
          ownership(net.config().ownership ? static_cast<size_t>(maxBatch) * net.config().area() : 0) {
        weights.upload(net);
    }
};

MetalInference::MetalInference(const Network& net, int maxBatch) {
    MetalContext::instance();   // throws the reason when the GPU is unusable
    runInPool([&] { impl_ = std::make_unique<Impl>(net, maxBatch); });
}

MetalInference::~MetalInference() {
    runInPool([&] { impl_.reset(); });
}

void MetalInference::infer(const float* input, int B, float* logits, float* values, float* ownership) {
    try {
        impl_->network.infer(impl_->workspace, impl_->weights, input, B, logits, values, ownership);
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("GPU inference failed: ") + e.what());
    }
}

LossStats MetalInference::inferLoss(const float* input, const float* pi, const float* z, int B,
                                    const float* ownership) {
    Impl& m = *impl_;
    const NetConfig& cfg = m.network.config();
    float* owners = cfg.ownership ? m.ownership.data() : nullptr;
    infer(input, B, m.logits.data(), m.values.data(), owners);
    return lossFromOutputs(cfg, m.logits.data(), m.values.data(), owners, pi, z, ownership, B);
}

size_t MetalInference::bytes() const { return impl_->workspace.bytes(); }

} // namespace azgo
