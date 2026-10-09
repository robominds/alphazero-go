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

// Network::forward's input transpose: in [B][planes][area] -> out [planes][B][area].
void encodeTransposeInput(CommandBatch& batch, BufView in, int B, int planes, int area, BufView out);

// Backend::im2col with k = 3: in [C][B][n * n] -> col [C * 9][B * n * n].
void encodeIm2col3(CommandBatch& batch, BufView in, int C, int B, int n, BufView col);

// Conv2d::forward: x [cin][B * n * n] -> y [cout][B * n * n] with weights
// w [cout][cin * k * k], k = 1 or 3. For k = 3 the columns go through col
// (cin * 9 * B * n * n floats); a 1x1 convolution multiplies x directly.
void encodeConvForward(CommandBatch& batch, int k, int cin, int cout, BufView w, BufView x, int B, int n,
                       BufView col, BufView y);

// BatchNorm2d::forwardInfer: y = gamma * (x - runMean) / sqrt(runVar + eps) + beta
// per channel of x, y [C][M]. y may be x.
void encodeBnInfer(CommandBatch& batch, int C, int M, BufView gamma, BufView beta, BufView runMean,
                   BufView runVar, BufView x, BufView y);

void encodeRelu(CommandBatch& batch, BufView x, size_t count);                  // reluForward, in place
void encodeAddInto(CommandBatch& batch, BufView dst, BufView src, size_t count);  // dst += src

// The policy head's flatten: act [2][B][A] -> flat [B][2A].
void encodePolicyFlatten(CommandBatch& batch, BufView act, int B, int A, BufView flat);

// Linear::forward: y [B][out] = x [B][in] . w^T + b, w [out][in].
void encodeLinearForward(CommandBatch& batch, int in, int out, BufView w, BufView b, BufView x, int B,
                         BufView y);

void encodeTanh(CommandBatch& batch, BufView x, size_t count);   // x = tanh(x), in place

} // namespace azgo
