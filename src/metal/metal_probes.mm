// The test probes declared in azgo/metal.hpp. Each one copies host arrays into
// fresh GPU buffers, encodes one kernel (or a layer's few kernels) into one
// command batch, waits, and copies the results back: the slow, simple path a
// test needs to compare a kernel with its CPU counterpart. The program itself
// never calls them.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <cstring>
#include <stdexcept>

#include "azgo/layers.hpp"
#include "azgo/metal.hpp"
#include "gpu_layers.hpp"
#include "metal_context.hpp"
#include "shader_types.h"

namespace azgo {
namespace metal_probe {

namespace {

// A new buffer holding a copy of count floats from src (zeros if src is null).
GpuBuffer upload(const float* src, size_t count) {
    GpuBuffer b = MetalContext::instance().newBuffer(count * sizeof(float));
    if (src && count > 0) std::memcpy(b.data(), src, count * sizeof(float));
    return b;
}

void download(const GpuBuffer& b, float* dst, size_t count) {
    if (count > 0) std::memcpy(dst, b.data(), count * sizeof(float));
}

} // namespace

std::string deviceName() {
    @autoreleasepool {
        return MetalContext::instance().deviceName();
    }
}

void fill(float* data, size_t n, float value, size_t tail) {
    @autoreleasepool {
        GpuBuffer b = upload(data, n + tail);
        CommandBatch batch;
        const FillParams p{static_cast<az_uint>(n), value};
        batch.dispatch("fill", {b}, p, {n, 1, 1});
        batch.commitAndWait();
        download(b, data, n + tail);
    }
}

void gemm(bool transA, bool transB, int M, int N, int K, float alpha, const float* A, const float* B,
          float beta, float* C) {
    @autoreleasepool {
        const size_t mn = static_cast<size_t>(M) * N;
        GpuBuffer a = upload(A, static_cast<size_t>(M) * K);
        GpuBuffer b = upload(B, static_cast<size_t>(K) * N);
        // With beta == 0 the GPU's C starts as zeros, not the caller's values:
        // the kernel must not read it, and NaN on the host must not matter.
        GpuBuffer c = upload(beta == 0.0f ? nullptr : C, mn);
        CommandBatch batch;
        encodeGemm(batch, transA, transB, M, N, K, alpha, a, b, beta, c);
        batch.commitAndWait();
        download(c, C, mn);
    }
}

void transposeInput(const float* input, int B, int planes, int area, float* out) {
    @autoreleasepool {
        const size_t count = static_cast<size_t>(B) * planes * area;
        GpuBuffer in = upload(input, count), o = upload(nullptr, count);
        CommandBatch batch;
        encodeTransposeInput(batch, in, B, planes, area, o);
        batch.commitAndWait();
        download(o, out, count);
    }
}

void im2col3(const float* in, int C, int B, int n, float* col) {
    @autoreleasepool {
        const size_t count = static_cast<size_t>(C) * B * n * n;
        GpuBuffer x = upload(in, count), c = upload(nullptr, count * 9);
        CommandBatch batch;
        encodeIm2col3(batch, x, C, B, n, c);
        batch.commitAndWait();
        download(c, col, count * 9);
    }
}

void convForward(const Conv2d& conv, const float* x, int B, int n, float* y) {
    @autoreleasepool {
        const size_t M = static_cast<size_t>(B) * n * n;
        GpuBuffer w = upload(conv.w.data(), conv.w.size());
        GpuBuffer in = upload(x, static_cast<size_t>(conv.cin) * M);
        GpuBuffer col = upload(nullptr, conv.colSize(B, n));
        GpuBuffer out = upload(nullptr, static_cast<size_t>(conv.cout) * M);
        CommandBatch batch;
        encodeConvForward(batch, conv.k, conv.cin, conv.cout, w, in, B, n, col, out);
        batch.commitAndWait();
        download(out, y, static_cast<size_t>(conv.cout) * M);
    }
}

void bnInfer(const BatchNorm2d& bn, const float* x, int M, float* y) {
    @autoreleasepool {
        const size_t count = static_cast<size_t>(bn.c) * M;
        GpuBuffer gamma = upload(bn.gamma.data(), bn.c), beta = upload(bn.beta.data(), bn.c);
        GpuBuffer mean = upload(bn.runMean.data(), bn.c), var = upload(bn.runVar.data(), bn.c);
        GpuBuffer in = upload(x, count), out = upload(nullptr, count);
        CommandBatch batch;
        encodeBnInfer(batch, bn.c, M, gamma, beta, mean, var, in, out);
        batch.commitAndWait();
        download(out, y, count);
    }
}

void relu(float* x, size_t count) {
    @autoreleasepool {
        GpuBuffer b = upload(x, count);
        CommandBatch batch;
        encodeRelu(batch, b, count);
        batch.commitAndWait();
        download(b, x, count);
    }
}

void addInto(float* dst, const float* src, size_t count) {
    @autoreleasepool {
        GpuBuffer d = upload(dst, count), s = upload(src, count);
        CommandBatch batch;
        encodeAddInto(batch, d, s, count);
        batch.commitAndWait();
        download(d, dst, count);
    }
}

void policyFlatten(const float* act, int B, int A, float* flat) {
    @autoreleasepool {
        const size_t count = static_cast<size_t>(2) * B * A;
        GpuBuffer in = upload(act, count), out = upload(nullptr, count);
        CommandBatch batch;
        encodePolicyFlatten(batch, in, B, A, out);
        batch.commitAndWait();
        download(out, flat, count);
    }
}

void linearForward(const Linear& fc, const float* x, int B, float* y) {
    @autoreleasepool {
        GpuBuffer w = upload(fc.w.data(), fc.w.size()), b = upload(fc.b.data(), fc.b.size());
        GpuBuffer in = upload(x, static_cast<size_t>(B) * fc.in);
        GpuBuffer out = upload(nullptr, static_cast<size_t>(B) * fc.out);
        CommandBatch batch;
        encodeLinearForward(batch, fc.in, fc.out, w, b, in, B, out);
        batch.commitAndWait();
        download(out, y, static_cast<size_t>(B) * fc.out);
    }
}

void tanhInPlace(float* x, size_t count) {
    @autoreleasepool {
        GpuBuffer b = upload(x, count);
        CommandBatch batch;
        encodeTanh(batch, b, count);
        batch.commitAndWait();
        download(b, x, count);
    }
}

} // namespace metal_probe
} // namespace azgo
