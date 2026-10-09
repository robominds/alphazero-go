// The test probes declared in azgo/metal.hpp. Each one copies host arrays into
// fresh GPU buffers, encodes one kernel (or a layer's few kernels) into one
// command batch, waits, and copies the results back: the slow, simple path a
// test needs to compare a kernel with its CPU counterpart. The program itself
// never calls them.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <stdexcept>

#include "azgo/layers.hpp"
#include "azgo/metal.hpp"
#include "azgo/network.hpp"
#include "azgo/param_layout.hpp"
#include "gpu_layers.hpp"
#include "gpu_network.hpp"
#include "metal_context.hpp"
#include "shader_types.h"

namespace azgo {
namespace metal_probe {

namespace {

// Every probe buffer is followed by a guard of NaNs with a recognizable
// payload. A kernel that writes past its output's end overwrites the guard,
// which download (or checkGuard) reports; one that reads past an operand's
// end picks up NaN, which then shows in the results.
constexpr size_t kGuardFloats = 1024;
constexpr uint32_t kGuardBits = 0x7fc0beefu;

// A new buffer holding a copy of count floats from src (zeros if src is
// null), then `guard` guard floats.
GpuBuffer upload(const float* src, size_t count, size_t guard = kGuardFloats) {
    GpuBuffer b = MetalContext::instance().newBuffer((count + guard) * sizeof(float));
    if (src && count > 0) std::memcpy(b.data(), src, count * sizeof(float));
    uint32_t* tail = reinterpret_cast<uint32_t*>(b.data() + count);
    for (size_t i = 0; i < guard; ++i) tail[i] = kGuardBits;
    return b;
}

// Throws std::logic_error if anything wrote into the guard after count floats.
void checkGuard(const GpuBuffer& b, size_t count) {
    const size_t total = b.bytes() / sizeof(float);
    const uint32_t* tail = reinterpret_cast<const uint32_t*>(b.data() + count);
    for (size_t i = 0; i + count < total; ++i)
        if (tail[i] != kGuardBits)
            throw std::logic_error("metal probe: a kernel wrote " + std::to_string(i) +
                                   " floats past the end of its buffer");
}

void download(const GpuBuffer& b, float* dst, size_t count) {
    checkGuard(b, count);
    if (count > 0) std::memcpy(dst, b.data(), count * sizeof(float));
}

} // namespace

std::string deviceName() { return MetalContext::instance().deviceName(); }   // pooled inside

void fill(float* data, size_t n, float value, size_t tail) {
    runInPool([&] {
        GpuBuffer b = upload(data, n + tail);
        CommandBatch batch;
        const FillParams p{static_cast<az_uint>(n), value};
        batch.dispatch("fill", {b}, p, {n, 1, 1});
        batch.commitAndWait();
        download(b, data, n + tail);
    });
}

void gemm(bool transA, bool transB, int M, int N, int K, float alpha, const float* A, const float* B,
          float beta, float* C) {
    runInPool([&] {
        const size_t mn = static_cast<size_t>(M) * N;
        GpuBuffer a = upload(A, static_cast<size_t>(M) * K);
        GpuBuffer b = upload(B, static_cast<size_t>(K) * N);
        // C goes up as given, NaNs and all, even with beta == 0 (when the
        // kernel must not read it). Its guard covers a whole 64-row tile past
        // the last row, so a write-back that ignores M lands in it.
        GpuBuffer c = upload(C, mn, static_cast<size_t>(kGemmTile) * std::max(N, 1) + kGuardFloats);
        CommandBatch batch;
        encodeGemm(batch, transA, transB, M, N, K, alpha, a, b, beta, c);
        batch.commitAndWait();
        checkGuard(a, static_cast<size_t>(M) * K);
        checkGuard(b, static_cast<size_t>(K) * N);
        download(c, C, mn);
    });
}

void transposeInput(const float* input, int B, int planes, int area, float* out) {
    runInPool([&] {
        const size_t count = static_cast<size_t>(B) * planes * area;
        GpuBuffer in = upload(input, count), o = upload(nullptr, count);
        CommandBatch batch;
        encodeTransposeInput(batch, in, B, planes, area, o);
        batch.commitAndWait();
        download(o, out, count);
    });
}

void im2col3(const float* in, int C, int B, int n, float* col) {
    runInPool([&] {
        const size_t count = static_cast<size_t>(C) * B * n * n;
        GpuBuffer x = upload(in, count), c = upload(nullptr, count * 9);
        CommandBatch batch;
        encodeIm2col3(batch, x, C, B, n, c);
        batch.commitAndWait();
        download(c, col, count * 9);
    });
}

void convForward(const Conv2d& conv, const float* x, int B, int n, float* y) {
    runInPool([&] {
        const size_t M = static_cast<size_t>(B) * n * n;
        GpuBuffer w = upload(conv.w.data(), conv.w.size());
        GpuBuffer in = upload(x, static_cast<size_t>(conv.cin) * M);
        GpuBuffer col = upload(nullptr, conv.colSize(B, n));
        GpuBuffer out = upload(nullptr, static_cast<size_t>(conv.cout) * M);
        CommandBatch batch;
        encodeConvForward(batch, conv.k, conv.cin, conv.cout, w, in, B, n, col, out);
        batch.commitAndWait();
        download(out, y, static_cast<size_t>(conv.cout) * M);
    });
}

void bnInfer(const BatchNorm2d& bn, const float* x, int M, float* y) {
    runInPool([&] {
        const size_t count = static_cast<size_t>(bn.c) * M;
        GpuBuffer gamma = upload(bn.gamma.data(), bn.c), beta = upload(bn.beta.data(), bn.c);
        GpuBuffer mean = upload(bn.runMean.data(), bn.c), var = upload(bn.runVar.data(), bn.c);
        GpuBuffer in = upload(x, count), out = upload(nullptr, count);
        CommandBatch batch;
        encodeBnInfer(batch, bn.c, M, gamma, beta, mean, var, in, out);
        batch.commitAndWait();
        download(out, y, count);
    });
}

void relu(float* x, size_t count) {
    runInPool([&] {
        GpuBuffer b = upload(x, count);
        CommandBatch batch;
        encodeRelu(batch, b, count);
        batch.commitAndWait();
        download(b, x, count);
    });
}

void addInto(float* dst, const float* src, size_t count) {
    runInPool([&] {
        GpuBuffer d = upload(dst, count), s = upload(src, count);
        CommandBatch batch;
        encodeAddInto(batch, d, s, count);
        batch.commitAndWait();
        download(d, dst, count);
    });
}

void policyFlatten(const float* act, int B, int A, float* flat) {
    runInPool([&] {
        const size_t count = static_cast<size_t>(2) * B * A;
        GpuBuffer in = upload(act, count), out = upload(nullptr, count);
        CommandBatch batch;
        encodePolicyFlatten(batch, in, B, A, out);
        batch.commitAndWait();
        download(out, flat, count);
    });
}

void linearForward(const Linear& fc, const float* x, int B, float* y) {
    runInPool([&] {
        GpuBuffer w = upload(fc.w.data(), fc.w.size()), b = upload(fc.b.data(), fc.b.size());
        GpuBuffer in = upload(x, static_cast<size_t>(B) * fc.in);
        GpuBuffer out = upload(nullptr, static_cast<size_t>(B) * fc.out);
        CommandBatch batch;
        encodeLinearForward(batch, fc.in, fc.out, w, b, in, B, out);
        batch.commitAndWait();
        download(out, y, static_cast<size_t>(B) * fc.out);
    });
}

void tanhInPlace(float* x, size_t count) {
    runInPool([&] {
        GpuBuffer b = upload(x, count);
        CommandBatch batch;
        encodeTanh(batch, b, count);
        batch.commitAndWait();
        download(b, x, count);
    });
}

// ---------------------------------------------------------------- hooks

namespace {
std::atomic<uint64_t> gHubChaosSeed{0};
} // namespace

void setHubChaosSeed(uint64_t seed) { gHubChaosSeed = seed; }
uint64_t hubChaosSeed() { return gHubChaosSeed; }

void failCommandBuffersAfter(int n) { setCommandBufferFailureAfter(n); }
void setWorkingSetLimit(size_t bytes) { setWorkingSetOverride(bytes); }
long long commandBuffersCommitted() { return azgo::commandBuffersCommitted(); }

void checkUpload(const Network& layoutNet, const Network& net) {
    runInPool([&] {
        GpuWeights weights(ParamLayout::of(const_cast<Network&>(layoutNet)));
        weights.upload(net);
    });
}

void checkForwardReadsEveryTensor(const Network& net, const NetConfig& encodeAs) {
    runInPool([&] {
        GpuWeights weights(ParamLayout::of(const_cast<Network&>(net)));
        weights.upload(net);
        GpuNetwork network(encodeAs);
        GpuWorkspace ws(encodeAs, 1, GpuWorkspace::Purpose::Inference);
        CommandBatch batch;
        network.encodeForward(batch, ws, weights, 1);
    });
}

} // namespace metal_probe
} // namespace azgo
