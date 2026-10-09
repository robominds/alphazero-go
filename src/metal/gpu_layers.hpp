// The network's layers on the GPU: one encode function per CPU operation in
// src/layers.cpp and src/backend.cpp, so the two files can be read side by
// side. Each function appends its kernels to a CommandBatch; nothing runs
// until the batch is committed. Tensors keep the CPU's layouts
// (channel-major activations [C][B * n * n], row-major matrices).
//
// Plain C++: the buffers are BufViews into GpuBuffers (metal_context.hpp).
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include "metal_context.hpp"

namespace azgo {

// Backend::gemm: row-major C[M x N] = alpha * op(A) * op(B) + beta * C,
// tightly packed; C is not read when beta == 0. The whole of K is summed in
// order inside one threadgroup (never split), so each element's bits depend
// only on its row of op(A) and its column of op(B): see gemm_simd in
// kernels.metal.
void encodeGemm(CommandBatch& batch, bool transA, bool transB, int M, int N, int K, float alpha,
                BufView A, BufView B, float beta, BufView C);

} // namespace azgo
