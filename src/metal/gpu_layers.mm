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

} // namespace

void encodeGemm(CommandBatch& batch, bool transA, bool transB, int M, int N, int K, float alpha,
                BufView A, BufView B, float beta, BufView C) {
    if (M <= 0 || N <= 0) return;
    // Every index into A, B and C must fit the kernel's 32-bit arithmetic.
    u32(static_cast<long long>(M) * K, "gemm: A");
    u32(static_cast<long long>(K) * N, "gemm: B");
    u32(static_cast<long long>(M) * N, "gemm: C");
    const GemmParams p{u32(M, "gemm: M"), u32(N, "gemm: N"), u32(K, "gemm: K"),
                       transA ? 1u : 0u, transB ? 1u : 0u, alpha, beta};
    // One threadgroup per 64 x 64 tile of C, x along N and y along M.
    batch.dispatchGroups("gemm_simd", {A, B, C}, p, {tiles(N), tiles(M), 1});
}

} // namespace azgo
