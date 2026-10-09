// The GPU layers (gpu_layers.hpp): each function fills in a kernel's
// parameter struct (shader_types.h) and the shape of its grid, and appends the
// dispatch to the batch. Dispatch shapes depend only on the tensor shapes.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include "gpu_layers.hpp"

#include <stdexcept>
#include <string>

#include "shader_types.h"

namespace azgo {

namespace {

az_uint u32(long long v, const char* what) {
    if (v < 0 || v > 0xffffffffLL)
        throw std::invalid_argument(std::string(what) + " out of range for the GPU");
    return static_cast<az_uint>(v);
}

size_t tiles(int count) { return (static_cast<size_t>(count) + kGemmTile - 1) / kGemmTile; }

// Throws std::logic_error unless v holds `floats` floats from its offset on:
// the kernels index blindly, so an undersized view would let them touch
// whatever lies beyond it.
void need(BufView v, size_t floats, const char* what) {
    if (!v.buffer) throw std::logic_error(std::string(what) + ": no buffer");
    if (v.offset + floats * sizeof(float) > v.buffer->bytes())
        throw std::logic_error(std::string(what) + ": needs " + std::to_string(floats) +
                               " floats at byte " + std::to_string(v.offset) + " of a " +
                               std::to_string(v.buffer->bytes()) + "-byte buffer");
}

} // namespace

void encodeGemm(CommandBatch& batch, bool transA, bool transB, int M, int N, int K, float alpha,
                BufView A, BufView B, float beta, BufView C) {
    if (M <= 0 || N <= 0) return;
    // Every index into A, B and C must fit the kernel's 32-bit arithmetic.
    u32(static_cast<long long>(M) * K, "gemm: A");
    u32(static_cast<long long>(K) * N, "gemm: B");
    u32(static_cast<long long>(M) * N, "gemm: C");
    need(A, static_cast<size_t>(M) * K, "gemm: A");
    need(B, static_cast<size_t>(K) * N, "gemm: B");
    need(C, static_cast<size_t>(M) * N, "gemm: C");
    const GemmParams p{u32(M, "gemm: M"), u32(N, "gemm: N"), u32(K, "gemm: K"),
                       transA ? 1u : 0u, transB ? 1u : 0u, alpha, beta};
    // One threadgroup per 64 x 64 tile of C, x along N and y along M.
    batch.dispatchGroups("gemm_simd", {A, B, C}, p, {tiles(N), tiles(M), 1});
}

void encodeTransposeInput(CommandBatch& batch, BufView in, int B, int planes, int area, BufView out) {
    const az_uint count = u32(static_cast<long long>(B) * planes * area, "transpose: size");
    const TransposeParams p{u32(B, "B"), u32(planes, "planes"), u32(area, "area")};
    need(in, count, "transpose: in");
    need(out, count, "transpose: out");
    batch.dispatch("transpose_input", {in, out}, p, {count, 1, 1});
}

void encodeIm2col3(CommandBatch& batch, BufView in, int C, int B, int n, BufView col) {
    const long long columns = static_cast<long long>(B) * n * n;
    u32(columns * C * 9, "im2col: col");
    const Im2colParams p{u32(C, "C"), u32(B, "B"), u32(n, "n")};
    need(in, static_cast<size_t>(columns) * C, "im2col: in");
    need(col, static_cast<size_t>(columns) * C * 9, "im2col: col");
    // x: a pixel of a sample (a column of col); y: a row of col.
    batch.dispatch("im2col3", {in, col}, p, {static_cast<size_t>(columns), static_cast<size_t>(C) * 9, 1});
}

void encodeConvForward(CommandBatch& batch, int k, int cin, int cout, BufView w, BufView x, int B, int n,
                       BufView col, BufView y) {
    if (k != 1 && k != 3) throw std::invalid_argument("encodeConvForward: k must be 1 or 3");
    const int M = B * n * n;
    // A 1x1 convolution needs no unfolding: x [cin][M] already is the column matrix.
    BufView columns = x;
    if (k == 3) {
        encodeIm2col3(batch, x, cin, B, n, col);
        columns = col;
    }
    // y [cout][M] = w [cout][cin*k*k] . columns [cin*k*k][M]
    encodeGemm(batch, false, false, cout, M, cin * k * k, 1.0f, w, columns, 0.0f, y);
}

void encodeBnInfer(CommandBatch& batch, int C, int M, BufView gamma, BufView beta, BufView runMean,
                   BufView runVar, BufView x, BufView y) {
    const az_uint count = u32(static_cast<long long>(C) * M, "bn: size");
    const BnInferParams p{u32(C, "C"), u32(M, "M")};
    need(x, count, "bn: x");
    need(y, count, "bn: y");
    for (BufView v : {gamma, beta, runMean, runVar}) need(v, static_cast<size_t>(C), "bn: statistics");
    batch.dispatch("bn_infer", {x, y, gamma, beta, runMean, runVar}, p, {count, 1, 1});
}

void encodeRelu(CommandBatch& batch, BufView x, size_t count) {
    const CountParams p{u32(static_cast<long long>(count), "relu: count")};
    need(x, count, "relu: x");
    batch.dispatch("relu_forward", {x}, p, {count, 1, 1});
}

void encodeAddInto(CommandBatch& batch, BufView dst, BufView src, size_t count) {
    const CountParams p{u32(static_cast<long long>(count), "add: count")};
    need(dst, count, "add: dst");
    need(src, count, "add: src");
    batch.dispatch("add_into", {dst, src}, p, {count, 1, 1});
}

void encodePolicyFlatten(CommandBatch& batch, BufView act, int B, int A, BufView flat) {
    const az_uint count = u32(static_cast<long long>(B) * 2 * A, "flatten: size");
    const FlattenParams p{u32(B, "B"), u32(A, "area")};
    need(act, count, "flatten: act");
    need(flat, count, "flatten: flat");
    batch.dispatch("policy_flatten", {act, flat}, p, {count, 1, 1});
}

void encodeLinearForward(CommandBatch& batch, int in, int out, BufView w, BufView b, BufView x, int B,
                         BufView y) {
    // y [B][out] = x [B][in] . w^T (w is stored [out][in]), then + b per row.
    encodeGemm(batch, false, true, B, out, in, 1.0f, x, w, 0.0f, y);
    const az_uint count = u32(static_cast<long long>(B) * out, "linear: size");
    const BiasParams p{u32(B, "B"), u32(out, "out")};
    need(b, static_cast<size_t>(out), "linear: b");
    batch.dispatch("bias_add_rows", {y, b}, p, {count, 1, 1});
}

void encodeTanh(CommandBatch& batch, BufView x, size_t count) {
    const CountParams p{u32(static_cast<long long>(count), "tanh: count")};
    need(x, count, "tanh: x");
    batch.dispatch("tanh_inplace", {x}, p, {count, 1, 1});
}

} // namespace azgo
