// The cuda16 network (cuda16_network.cuh). encodeForward follows
// Network::forward (src/network.cpp) layer by layer, as the cuda backend's
// does; only the trunk's convolutions differ, and every head is the cuda
// backend's code on the same kernels.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include "cuda16_network.cuh"
#include <memory>
#include <stdexcept>
#include "azgo/cuda.hpp"
#include "azgo/cuda16.hpp"
#include "../kernels.cuh"

namespace azgo {
namespace cuda16 {

// ---------------------------------------------------------------- weights

void Conv16Weights::upload(Half type, const float* w, const float* gamma, const float* beta, const float* mean,
                           const float* var, int cin_, int cout_, std::vector<cuda::DeviceBuffer>& staging,
                           cudaStream_t s) {
    cin = cin_;
    cout = cout_;
    cinPadded = paddedChannels(cin);
    coutPadded = paddedChannels(cout);
    w16 = HalfBuffer(static_cast<size_t>(coutPadded) * 9 * cinPadded);
    scaleShift = cuda::DeviceBuffer(2 * static_cast<size_t>(coutPadded));
    const size_t weights = static_cast<size_t>(cout) * cin * 9;
    staging.emplace_back(weights);
    cuda::check(cudaMemcpyAsync(staging.back().data(), w, weights * sizeof(float), cudaMemcpyHostToDevice, s),
                "upload conv16 weights");
    launchWeightsTo16(type, staging.back().data(), cout, cin, coutPadded, cinPadded, w16.data(), s);
    // The statistics padded to coutPadded with gamma = beta = mean = 0 and
    // var = 1, which fold to scale = shift = 0: a pad channel's output is 0.
    // The fold is the cuda backend's bn_fold, so the real channels' scale and
    // shift are the cuda backend's bits.
    staging.emplace_back(4 * static_cast<size_t>(coutPadded));
    float* stats = staging.back().data();
    const size_t C = static_cast<size_t>(cout), P = static_cast<size_t>(coutPadded);
    const float* from[4] = {gamma, beta, mean, var};
    for (int i = 0; i < 4; ++i)
        cuda::check(cudaMemcpyAsync(stats + i * P, from[i], C * sizeof(float), cudaMemcpyHostToDevice, s),
                    "upload conv16 statistics");
    if (P > C) {
        cuda::check(cudaMemsetAsync(stats + C, 0, (P - C) * sizeof(float), s), "pad");
        cuda::check(cudaMemsetAsync(stats + P + C, 0, (P - C) * sizeof(float), s), "pad");
        cuda::check(cudaMemsetAsync(stats + 2 * P + C, 0, (P - C) * sizeof(float), s), "pad");
        // var = 1: 0x3f800000 is not a byte pattern, so a fill kernel.
        cuda::launchFill(stats + 3 * P + C, P - C, 1.0f, s);
    }
    cuda::launchBnFold(stats, stats + P, stats + 2 * P, stats + 3 * P, coutPadded, scaleShift.data(),
                       scaleShift.data() + P, s);
}

namespace {

// The trunk's 3x3 convolutions and their batch norms, in pass order.
std::vector<std::pair<std::string, std::string>> trunkConvs(const NetConfig& cfg) {
    std::vector<std::pair<std::string, std::string>> list = {{"stem.conv", "stem.bn"}};
    for (int i = 0; i < cfg.blocks; ++i) {
        const std::string prefix = "block" + std::to_string(i);
        list.push_back({prefix + ".conv1", prefix + ".bn1"});
        list.push_back({prefix + ".conv2", prefix + ".bn2"});
    }
    return list;
}

} // namespace

Cuda16Weights::Cuda16Weights(const ParamLayout& layout, const NetConfig& cfg, Half type)
    : cfg_(cfg), type_(type), fp32_(layout) {}

void Cuda16Weights::upload(const Network& net) {
    fp32_.upload(net);   // checks every tensor's name and size against the layout first
    Network& n = const_cast<Network&>(net);   // params() hands out writable views; this only reads
    std::map<std::string, const float*> host;
    std::map<std::string, size_t> sizes;
    for (const ParamView& v : n.params()) {
        host[v.name] = v.value;
        sizes[v.name] = v.size;
    }
    for (const BufferView& v : n.buffers()) {
        host[v.name] = v.value;
        sizes[v.name] = v.size;
    }
    auto get = [&](const std::string& name, size_t size) {
        auto it = host.find(name);
        if (it == host.end()) throw std::logic_error("GPU network: the network lacks tensor '" + name + "'");
        if (sizes[name] != size)
            throw std::logic_error("GPU network: tensor '" + name + "' has " + std::to_string(sizes[name]) +
                                   " floats, the 16-bit trunk expects " + std::to_string(size));
        return it->second;
    };
    cuda::Stream s;
    std::vector<cuda::DeviceBuffer> staging;
    for (const auto& entry : trunkConvs(cfg_)) {
        const std::string& conv = entry.first;
        const std::string& bn = entry.second;
        const int cin = conv == "stem.conv" ? cfg_.planes() : cfg_.filters, cout = cfg_.filters;
        const size_t C = static_cast<size_t>(cout);
        Conv16Weights& c = convs_[conv];
        c.upload(type_, get(conv + ".w", C * cin * 9), get(bn + ".gamma", C), get(bn + ".beta", C),
                 get(bn + ".runMean", C), get(bn + ".runVar", C), cin, cout, staging, s.get());
    }
    cuda::check(cudaGetLastError(), "conv16 weights");
    s.synchronize();   // before the host views and the staging buffers go
}

const Conv16Weights& Cuda16Weights::conv(const std::string& conv, const std::string& bn) const {
    auto it = convs_.find(conv);
    if (it == convs_.end()) throw std::logic_error("GPU network: no 16-bit convolution '" + conv + "'");
    fp32_.param(conv + ".w");   // the 16-bit copy is read: so is the tensor
    fp32_.scale(bn);            // marks the batch norm's four tensors
    return it->second;
}

size_t Cuda16Weights::bytes() const {
    size_t total = fp32_.bytes();
    for (const auto& c : convs_) total += c.second.w16.bytes() + c.second.scaleShift.bytes();
    return total;
}

// ---------------------------------------------------------------- workspace

namespace {

struct WorkspaceSizes {
    size_t stage, input16, trunk16, features, head, policyAct, policyFlat, logits, valueAct, valueHidden, values,
        ownership;
    size_t bytes() const {
        return (stage + features + head + policyAct + policyFlat + logits + valueAct + valueHidden + values +
                ownership) * sizeof(float) +
               ((input16 + 1) / 2 + 3 * ((trunk16 + 1) / 2)) * sizeof(float);
    }
};

WorkspaceSizes workspaceSizes(const NetConfig& cfg, int maxBatch) {
    const size_t M = static_cast<size_t>(maxBatch) * cfg.area();
    WorkspaceSizes z;
    z.stage = static_cast<size_t>(cfg.planes()) * M;
    z.input16 = static_cast<size_t>(paddedChannels(cfg.planes())) * M;
    z.trunk16 = static_cast<size_t>(paddedChannels(cfg.filters)) * M;
    z.features = static_cast<size_t>(cfg.filters) * M;
    z.head = z.policyAct = z.policyFlat = 2 * M;
    z.logits = static_cast<size_t>(maxBatch) * cfg.actions();
    z.valueAct = M;
    z.valueHidden = static_cast<size_t>(maxBatch) * cfg.valueHidden;
    z.values = static_cast<size_t>(maxBatch);
    z.ownership = cfg.ownership ? M : 0;
    return z;
}

} // namespace

Cuda16Workspace::Cuda16Workspace(const NetConfig& cfg, int maxBatch) : cfg_(cfg), maxBatch_(maxBatch) {
    if (maxBatch < 1) throw std::invalid_argument("Cuda16Workspace: maxBatch must be >= 1");
    const WorkspaceSizes z = workspaceSizes(cfg, maxBatch);
    stage = cuda::DeviceBuffer(z.stage);
    input = HalfBuffer(z.input16);
    for (HalfBuffer& t : trunk) t = HalfBuffer(z.trunk16);
    features = cuda::DeviceBuffer(z.features);
    head = cuda::DeviceBuffer(z.head);
    policyAct = cuda::DeviceBuffer(z.policyAct);
    policyFlat = cuda::DeviceBuffer(z.policyFlat);
    logits = cuda::DeviceBuffer(z.logits);
    valueAct = cuda::DeviceBuffer(z.valueAct);
    valueHidden = cuda::DeviceBuffer(z.valueHidden);
    values = cuda::DeviceBuffer(z.values);
    if (cfg.ownership) ownership = cuda::DeviceBuffer(z.ownership);
}

size_t Cuda16Workspace::bytesFor(const NetConfig& cfg, int maxBatch) {
    return workspaceSizes(cfg, maxBatch).bytes();
}

size_t Cuda16Workspace::bytes() const {
    size_t total = stage.bytes() + input.bytes() + features.bytes() + head.bytes() + policyAct.bytes() +
                   policyFlat.bytes() + logits.bytes() + valueAct.bytes() + valueHidden.bytes() + values.bytes() +
                   ownership.bytes();
    for (const HalfBuffer& t : trunk) total += t.bytes();
    return total;
}

// ---------------------------------------------------------------- the pass

Cuda16Network::Cuda16Network(const NetConfig& cfg) : cfg_(cfg) {}

void Cuda16Network::encodeForward(cudaStream_t s, Cuda16Workspace& ws, const Cuda16Weights& w, int B) const {
    if (B < 1 || B > ws.maxBatch()) throw std::invalid_argument("Cuda16Network: batch out of range");
    if (ws.config() != cfg_) throw std::invalid_argument("Cuda16Network: workspace of another network");
    const cuda::CudaWeights& w32 = w.fp32();
    const int n = cfg_.boardSize, A = cfg_.area(), P = cfg_.planes(), F = cfg_.filters;
    const int M = B * A;   // pixels per channel
    const Half type = w.type();

    // [B][planes][A] fp32 -> [B * A][planes padded] 16-bit (0/1 planes: exact).
    launchInputToNhwc(type, ws.stage.data(), B, P, A, paddedChannels(P), ws.input.data(), s);

    // conv3x3 -> BN -> (+ skip) -> ReLU on the tensor cores, into 16-bit
    // NHWC (out16) or, for the trunk's last layer, fp32 channel-major (out32).
    auto conv = [&](const std::string& name, const std::string& bn, const HalfBuffer& x, const HalfBuffer* skip,
                    HalfBuffer* out16, float* out32) {
        const Conv16Weights& c = w.conv(name, bn);
        Conv16Args a;
        a.in = x.data();
        a.w = c.w16.data();
        a.batch = B;
        a.n = n;
        a.cinPadded = c.cinPadded;
        a.coutPadded = c.coutPadded;
        a.cout = c.cout;
        a.scale = c.scale();
        a.shift = c.shift();
        a.skip = skip ? skip->data() : nullptr;
        a.out16 = out16 ? out16->data() : nullptr;
        a.out32 = out32;
        launchConv16(type, a, channelTileFor(c.coutPadded), s);
    };

    // Stem, then the residual blocks rotating through three buffers (input
    // x, middle, output), as the cuda backend's.
    const bool stemLast = cfg_.blocks == 0;
    conv("stem.conv", "stem.bn", ws.input, nullptr, stemLast ? nullptr : &ws.trunk[0],
         stemLast ? ws.features.data() : nullptr);
    int x = 0;
    for (int i = 0; i < cfg_.blocks; ++i) {
        const std::string prefix = "block" + std::to_string(i);
        const int mid = (x + 1) % 3, out = (x + 2) % 3;
        const bool last = i + 1 == cfg_.blocks;
        conv(prefix + ".conv1", prefix + ".bn1", ws.trunk[x], nullptr, &ws.trunk[mid], nullptr);
        conv(prefix + ".conv2", prefix + ".bn2", ws.trunk[mid], &ws.trunk[x], last ? nullptr : &ws.trunk[out],
             last ? ws.features.data() : nullptr);
        x = out;
    }
    const float* features = ws.features.data();

    // The heads: the cuda backend's encodeForward from here on, verbatim in
    // its kernels and their order (src/cuda/cuda_network.cu).
    auto head = [&](const std::string& prefix, int C, float* y) {
        cuda::launchGemm(false, C, M, F, w32.param(prefix + ".conv.w"), features, ws.head.data(),
                         cuda::gemmTileFor(C), s);
        const std::string bn = prefix + ".bn";
        cuda::launchBnInfer(ws.head.data(), y, C, static_cast<size_t>(M), w32.param(bn + ".gamma"),
                            w32.param(bn + ".beta"), w32.buffer(bn + ".runMean"), w32.buffer(bn + ".runVar"), s);
    };
    auto linear = [&](const std::string& name, int in, int out, const float* xin, float* y) {
        cuda::launchGemm(true, B, out, in, xin, w32.param(name + ".w"), y, cuda::kLinearTile, s);
        cuda::launchBiasAddRows(y, w32.param(name + ".b"), B, out, s);
    };

    // Policy head: conv1x1 (F -> 2) -> BN -> ReLU -> per-sample flatten -> Linear.
    head("policy", 2, ws.policyAct.data());
    cuda::launchRelu(ws.policyAct.data(), 2 * static_cast<size_t>(M), s);
    cuda::launchPolicyFlatten(ws.policyAct.data(), B, A, ws.policyFlat.data(), s);
    linear("policy.fc", 2 * A, cfg_.actions(), ws.policyFlat.data(), ws.logits.data());

    // Value head: conv1x1 (F -> 1) -> BN -> ReLU -> Linear(A -> V) -> ReLU
    // -> Linear(V -> 1) -> tanh.
    head("value", 1, ws.valueAct.data());
    cuda::launchRelu(ws.valueAct.data(), static_cast<size_t>(M), s);
    linear("value.fc1", A, cfg_.valueHidden, ws.valueAct.data(), ws.valueHidden.data());
    cuda::launchRelu(ws.valueHidden.data(), static_cast<size_t>(B) * cfg_.valueHidden, s);
    linear("value.fc2", cfg_.valueHidden, 1, ws.valueHidden.data(), ws.values.data());
    cuda::launchTanh(ws.values.data(), static_cast<size_t>(B), s);

    // Ownership head: conv1x1 (F -> 1) -> BN -> tanh at every point.
    if (cfg_.ownership) {
        head("ownership", 1, ws.ownership.data());
        cuda::launchTanh(ws.ownership.data(), static_cast<size_t>(M), s);
    }
    w32.checkEveryTensorRead();
}

} // namespace cuda16

// ---------------------------------------------------------------- Cuda16Inference

namespace {

cuda16::Half halfOf(Cuda16Type t) { return t == Cuda16Type::Bf16 ? cuda16::Half::Bf16 : cuda16::Half::Fp16; }

} // namespace

struct Cuda16Inference::Impl {
    cuda16::Cuda16Network network;
    cuda16::Cuda16Weights weights;
    cuda16::Cuda16Workspace workspace;
    cuda::Stream stream;
    Impl(const Network& net, int maxBatch, Cuda16Type type)
        : network(net.config()),
          weights(ParamLayout::of(const_cast<Network&>(net)), net.config(), halfOf(type)),
          workspace(net.config(), maxBatch) {
        weights.upload(net);
    }
};

Cuda16Inference::Cuda16Inference(const Network& net, int maxBatch, Cuda16Type type) {
    std::string why;
    if (!cuda16Available(&why)) throw std::runtime_error("cuda16: " + why);
    impl_ = std::make_unique<Impl>(net, maxBatch, type);
}

Cuda16Inference::~Cuda16Inference() = default;

void Cuda16Inference::infer(const float* input, int B, float* logits, float* values, float* ownership) {
    const NetConfig& cfg = impl_->network.config();
    if (ownership && !cfg.ownership) throw std::invalid_argument("Cuda16Inference: no ownership head");
    if (B < 1 || B > impl_->workspace.maxBatch()) throw std::invalid_argument("Cuda16Inference: batch out of range");
    cuda16::Cuda16Workspace& ws = impl_->workspace;
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

size_t Cuda16Inference::bytes() const { return impl_->workspace.bytes(); }

} // namespace azgo
