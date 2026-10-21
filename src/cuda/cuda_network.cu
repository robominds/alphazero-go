// The network on the GPU (cuda_network.cuh). encodeForward follows
// Network::forward (src/network.cpp) layer by layer, as the Metal port's
// does, in inference mode only: every layer enqueues its kernels on one
// stream, so a whole pass is one trip to the GPU.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include "cuda_network.cuh"
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include "azgo/cuda.hpp"
#include "kernels.cuh"

namespace azgo {
namespace cuda {

// ---------------------------------------------------------------- CudaWeights

namespace {

// Copies each view into the host image of its arena after checking it
// against the layout's entry at the same position.
template <class View>
void copyChecked(const std::vector<View>& views, const std::vector<ParamLayout::Entry>& entries,
                 const ParamLayout& layout, float* arena) {
    for (size_t i = 0; i < views.size(); ++i) {
        const View& v = views[i];
        if (i >= entries.size() || entries[i].name != v.name) {
            if (!layout.find(v.name))
                throw std::logic_error("GPU network: unknown tensor '" + v.name + "' (a layer without a GPU port?)");
            throw std::logic_error("GPU network: tensor '" + v.name + "' out of the layout's order");
        }
        if (entries[i].size != v.size)
            throw std::logic_error("GPU network: tensor '" + v.name + "' has " + std::to_string(v.size) +
                                   " floats, the layout " + std::to_string(entries[i].size));
        std::memcpy(arena + entries[i].offset, v.value, v.size * sizeof(float));
    }
    if (views.size() < entries.size())
        throw std::logic_error("GPU network: the network lacks tensor '" + entries[views.size()].name + "'");
}

bool endsWith(const std::string& s, const std::string& tail) {
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

size_t aligned(size_t floats) {
    return (floats + ParamLayout::kAlignFloats - 1) / ParamLayout::kAlignFloats * ParamLayout::kAlignFloats;
}

} // namespace

CudaWeights::CudaWeights(const ParamLayout& layout)
    : layout_(layout),
      params_(layout.paramFloats),
      buffers_(layout.bufferFloats),
      paramRead_(layout.params.size(), 0),
      bufferRead_(layout.buffers.size(), 0) {
    // Every batch norm (a ".gamma" parameter) gets a folded scale and shift.
    size_t floats = 0;
    for (const ParamLayout::Entry& e : layout.params) {
        if (!endsWith(e.name, ".gamma")) continue;
        const std::string bn = e.name.substr(0, e.name.size() - 6);
        foldedAt_[bn] = floats;
        foldedChannels_[bn] = static_cast<int>(e.size);
        floats += aligned(2 * e.size);
    }
    folded_ = DeviceBuffer(floats);
}

void CudaWeights::upload(const Network& net) {
    Network& n = const_cast<Network&>(net);   // params() hands out writable views; this only reads
    std::vector<float> params(layout_.paramFloats, 0.0f), buffers(layout_.bufferFloats, 0.0f);
    copyChecked(n.params(), layout_.params, layout_, params.data());
    copyChecked(n.buffers(), layout_.buffers, layout_, buffers.data());
    check(cudaMemcpy(params_.data(), params.data(), params.size() * sizeof(float), cudaMemcpyHostToDevice),
          "upload parameters");
    check(cudaMemcpy(buffers_.data(), buffers.data(), buffers.size() * sizeof(float), cudaMemcpyHostToDevice),
          "upload statistics");
    Stream s;
    for (const auto& entry : foldedAt_) {
        const std::string& bn = entry.first;
        const int C = foldedChannels_.at(bn);
        float* scale = folded_.data() + entry.second;
        // Looked up without marking: the pass marks what it reads.
        auto offset = [&](const std::vector<ParamLayout::Entry>& list, const std::string& name) {
            for (const ParamLayout::Entry& e : list)
                if (e.name == name) return e.offset;
            throw std::logic_error("GPU network: batch norm '" + bn + "' lacks '" + name + "'");
        };
        launchBnFold(params_.data() + offset(layout_.params, bn + ".gamma"),
                     params_.data() + offset(layout_.params, bn + ".beta"),
                     buffers_.data() + offset(layout_.buffers, bn + ".runMean"),
                     buffers_.data() + offset(layout_.buffers, bn + ".runVar"), C, scale, scale + C, s.get());
    }
    check(cudaGetLastError(), "bn_fold");
    s.synchronize();
}

const float* CudaWeights::param(const std::string& name) const {
    for (size_t i = 0; i < layout_.params.size(); ++i) {
        if (layout_.params[i].name != name) continue;
        paramRead_[i] = 1;
        return params_.data() + layout_.params[i].offset;
    }
    throw std::logic_error("GPU network: no parameter '" + name + "'");
}

const float* CudaWeights::buffer(const std::string& name) const {
    for (size_t i = 0; i < layout_.buffers.size(); ++i) {
        if (layout_.buffers[i].name != name) continue;
        bufferRead_[i] = 1;
        return buffers_.data() + layout_.buffers[i].offset;
    }
    throw std::logic_error("GPU network: no buffer '" + name + "'");
}

void CudaWeights::markRead(const std::string& bn) const {
    param(bn + ".gamma");
    param(bn + ".beta");
    buffer(bn + ".runMean");
    buffer(bn + ".runVar");
}

const float* CudaWeights::scale(const std::string& name) const {
    auto it = foldedAt_.find(name);
    if (it == foldedAt_.end()) throw std::logic_error("GPU network: no batch norm '" + name + "'");
    markRead(name);
    return folded_.data() + it->second;
}

const float* CudaWeights::shift(const std::string& name) const {
    return scale(name) + foldedChannels_.at(name);
}

void CudaWeights::checkEveryTensorRead() const {
    std::string unread;
    for (size_t i = 0; i < layout_.params.size(); ++i)
        if (!paramRead_[i]) unread += (unread.empty() ? "" : ", ") + layout_.params[i].name;
    for (size_t i = 0; i < layout_.buffers.size(); ++i)
        if (!bufferRead_[i]) unread += (unread.empty() ? "" : ", ") + layout_.buffers[i].name;
    if (!unread.empty())
        throw std::logic_error("GPU network: the forward pass reads no tensor " + unread +
                               " (a layer without a GPU port?)");
}

// ---------------------------------------------------------------- CudaWorkspace

namespace {

struct WorkspaceSizes {
    size_t stage, input, trunk, head, policyAct, policyFlat, logits, valueAct, valueHidden, values, ownership;
    size_t total() const {
        return stage + input + 3 * trunk + head + policyAct + policyFlat + logits + valueAct + valueHidden + values +
               ownership;
    }
};

WorkspaceSizes workspaceSizes(const NetConfig& cfg, int maxBatch) {
    const size_t M = static_cast<size_t>(maxBatch) * cfg.area();   // pixels per channel
    WorkspaceSizes z;
    z.stage = z.input = static_cast<size_t>(cfg.planes()) * M;
    z.trunk = static_cast<size_t>(cfg.filters) * M;
    z.head = z.policyAct = z.policyFlat = 2 * M;
    z.logits = static_cast<size_t>(maxBatch) * cfg.actions();
    z.valueAct = M;
    z.valueHidden = static_cast<size_t>(maxBatch) * cfg.valueHidden;
    z.values = static_cast<size_t>(maxBatch);
    z.ownership = cfg.ownership ? M : 0;
    return z;
}

} // namespace

CudaWorkspace::CudaWorkspace(const NetConfig& cfg, int maxBatch) : cfg_(cfg), maxBatch_(maxBatch) {
    if (maxBatch < 1) throw std::invalid_argument("CudaWorkspace: maxBatch must be >= 1");
    const WorkspaceSizes z = workspaceSizes(cfg, maxBatch);
    stage = DeviceBuffer(z.stage);
    input = DeviceBuffer(z.input);
    for (DeviceBuffer& t : trunk) t = DeviceBuffer(z.trunk);
    head = DeviceBuffer(z.head);
    policyAct = DeviceBuffer(z.policyAct);
    policyFlat = DeviceBuffer(z.policyFlat);
    logits = DeviceBuffer(z.logits);
    valueAct = DeviceBuffer(z.valueAct);
    valueHidden = DeviceBuffer(z.valueHidden);
    values = DeviceBuffer(z.values);
    if (cfg.ownership) ownership = DeviceBuffer(z.ownership);
}

size_t CudaWorkspace::bytesFor(const NetConfig& cfg, int maxBatch) {
    return workspaceSizes(cfg, maxBatch).total() * sizeof(float);
}

size_t CudaWorkspace::bytes() const {
    size_t total = stage.bytes() + input.bytes() + head.bytes() + policyAct.bytes() + policyFlat.bytes() +
                   logits.bytes() + valueAct.bytes() + valueHidden.bytes() + values.bytes() + ownership.bytes();
    for (const DeviceBuffer& t : trunk) total += t.bytes();
    return total;
}

// ---------------------------------------------------------------- CudaNetwork

CudaNetwork::CudaNetwork(const NetConfig& cfg) : cfg_(cfg) {}

void CudaNetwork::encodeForward(cudaStream_t s, CudaWorkspace& ws, const CudaWeights& w, int B) const {
    if (B < 1 || B > ws.maxBatch()) throw std::invalid_argument("CudaNetwork: batch out of range");
    if (ws.config() != cfg_) throw std::invalid_argument("CudaNetwork: workspace of another network");
    const int n = cfg_.boardSize, A = cfg_.area(), P = cfg_.planes(), F = cfg_.filters;
    const int M = B * A;   // pixels per channel

    // [B][planes][A] -> [planes][B][A]. Channel-major lets a convolution
    // treat the whole batch as one wide matrix: one product per layer.
    launchTransposeInput(ws.stage.data(), B, P, A, ws.input.data(), s);

    // conv3x3 -> BN -> (+ skip) -> ReLU, in one fused kernel.
    auto conv = [&](const std::string& name, const std::string& bn, const float* x, int cin, float* y,
                    const float* skip) {
        ConvArgs cv;
        cv.in = x;
        cv.batch = B;
        cv.n = n;
        cv.scale = w.scale(bn);
        cv.shift = w.shift(bn);
        cv.skip = skip;
        launchConvBnRelu(w.param(name + ".w"), cin, F, cv, y, gemmTileFor(F), s);
    };
    // A head's 1x1 convolution (no unfolding: the trunk already is the
    // column matrix) and its inference batch norm.
    auto head = [&](const std::string& prefix, int C, const float* features, float* y) {
        launchGemm(false, C, M, F, w.param(prefix + ".conv.w"), features, ws.head.data(), gemmTileFor(C), s);
        const std::string bn = prefix + ".bn";
        launchBnInfer(ws.head.data(), y, C, static_cast<size_t>(M), w.param(bn + ".gamma"), w.param(bn + ".beta"),
                      w.buffer(bn + ".runMean"), w.buffer(bn + ".runVar"), s);
    };
    // Linear::forward: y [B][out] = x [B][in] . w^T + b.
    auto linear = [&](const std::string& name, int in, int out, const float* x, float* y) {
        launchGemm(true, B, out, in, x, w.param(name + ".w"), y, kLinearTile, s);
        launchBiasAddRows(y, w.param(name + ".b"), B, out, s);
    };

    // Stem: conv3x3 -> BN -> ReLU.
    conv("stem.conv", "stem.bn", ws.input.data(), P, ws.trunk[0].data(), nullptr);

    // Residual blocks: conv -> BN -> ReLU -> conv -> BN -> + input -> ReLU,
    // rotating through three buffers: input x, middle, output.
    int x = 0;
    for (int i = 0; i < cfg_.blocks; ++i) {
        const std::string prefix = "block" + std::to_string(i);
        const int mid = (x + 1) % 3, out = (x + 2) % 3;
        conv(prefix + ".conv1", prefix + ".bn1", ws.trunk[x].data(), F, ws.trunk[mid].data(), nullptr);
        conv(prefix + ".conv2", prefix + ".bn2", ws.trunk[mid].data(), F, ws.trunk[out].data(), ws.trunk[x].data());
        x = out;
    }
    const float* features = ws.trunk[x].data();

    // Policy head: conv1x1 (F -> 2) -> BN -> ReLU -> per-sample flatten -> Linear.
    head("policy", 2, features, ws.policyAct.data());
    launchRelu(ws.policyAct.data(), 2 * static_cast<size_t>(M), s);
    launchPolicyFlatten(ws.policyAct.data(), B, A, ws.policyFlat.data(), s);
    linear("policy.fc", 2 * A, cfg_.actions(), ws.policyFlat.data(), ws.logits.data());

    // Value head: conv1x1 (F -> 1) -> BN -> ReLU -> Linear(A -> V) -> ReLU
    // -> Linear(V -> 1) -> tanh. With one channel, [1][B][A] already is [B][A].
    head("value", 1, features, ws.valueAct.data());
    launchRelu(ws.valueAct.data(), static_cast<size_t>(M), s);
    linear("value.fc1", A, cfg_.valueHidden, ws.valueAct.data(), ws.valueHidden.data());
    launchRelu(ws.valueHidden.data(), static_cast<size_t>(B) * cfg_.valueHidden, s);
    linear("value.fc2", cfg_.valueHidden, 1, ws.valueHidden.data(), ws.values.data());
    launchTanh(ws.values.data(), static_cast<size_t>(B), s);

    // Ownership head: conv1x1 (F -> 1) -> BN -> tanh at every point.
    if (cfg_.ownership) {
        head("ownership", 1, features, ws.ownership.data());
        launchTanh(ws.ownership.data(), static_cast<size_t>(M), s);
    }
    // Every tensor the network uploaded must have been read by some layer
    // above, or the GPU is evaluating a different network than the CPU's.
    w.checkEveryTensorRead();
}

} // namespace cuda

// ---------------------------------------------------------------- CudaInference

struct CudaInference::Impl {
    cuda::CudaNetwork network;
    cuda::CudaWeights weights;
    cuda::CudaWorkspace workspace;
    cuda::Stream stream;
    Impl(const Network& net, int maxBatch)
        : network(net.config()),
          weights(ParamLayout::of(const_cast<Network&>(net))),
          workspace(net.config(), maxBatch) {
        weights.upload(net);
    }
};

CudaInference::CudaInference(const Network& net, int maxBatch) {
    cuda::Context::instance();   // throws the reason when the GPU is unusable
    impl_ = std::make_unique<Impl>(net, maxBatch);
}

CudaInference::~CudaInference() = default;

void CudaInference::infer(const float* input, int B, float* logits, float* values, float* ownership) {
    const NetConfig& cfg = impl_->network.config();
    if (ownership && !cfg.ownership) throw std::invalid_argument("CudaInference: no ownership head");
    if (B < 1 || B > impl_->workspace.maxBatch()) throw std::invalid_argument("CudaInference: batch out of range");
    cuda::CudaWorkspace& ws = impl_->workspace;
    const cudaStream_t s = impl_->stream.get();
    try {
        const size_t in = static_cast<size_t>(B) * cfg.planes() * cfg.area();
        cuda::check(cudaMemcpyAsync(ws.stage.data(), input, in * sizeof(float), cudaMemcpyHostToDevice, s), "input");
        impl_->network.encodeForward(s, ws, impl_->weights, B);
        cuda::check(cudaGetLastError(), "kernel launch");
        cuda::check(cudaMemcpyAsync(logits, ws.logits.data(), static_cast<size_t>(B) * cfg.actions() * sizeof(float),
                                    cudaMemcpyDeviceToHost, s),
                    "logits");
        cuda::check(cudaMemcpyAsync(values, ws.values.data(), static_cast<size_t>(B) * sizeof(float),
                                    cudaMemcpyDeviceToHost, s),
                    "values");
        if (ownership)
            cuda::check(cudaMemcpyAsync(ownership, ws.ownership.data(),
                                        static_cast<size_t>(B) * cfg.area() * sizeof(float), cudaMemcpyDeviceToHost, s),
                        "ownership");
        impl_->stream.synchronize();
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("GPU inference failed: ") + e.what());
    }
}

size_t CudaInference::bytes() const { return impl_->workspace.bytes(); }

namespace cuda_probe {

void checkUpload(const Network& layoutNet, const Network& net) {
    cuda::Context::instance();
    cuda::CudaWeights w(ParamLayout::of(const_cast<Network&>(layoutNet)));
    w.upload(net);
}

void checkForwardReadsEveryTensor(const Network& net, const NetConfig& encodeAs) {
    cuda::Context::instance();
    cuda::CudaWeights w(ParamLayout::of(const_cast<Network&>(net)));
    w.upload(net);
    cuda::CudaWorkspace ws(encodeAs, 1);
    cuda::Stream s;
    try {
        cuda::CudaNetwork(encodeAs).encodeForward(s.get(), ws, w, 1);
    } catch (...) {
        s.synchronize();
        throw;
    }
    s.synchronize();
}

} // namespace cuda_probe

} // namespace azgo
