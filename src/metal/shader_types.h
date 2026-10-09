// Definitions shared by the C++ side and the Metal kernels: every kernel's
// parameter struct and the fixed threadgroup sizes. The .mm files include this
// header; CMake pastes it into the embedded shader source where kernels.metal
// includes it. So the CPU and the GPU cannot disagree on a struct's layout.
//
// It is written in the common subset of C++ and the Metal Shading Language.
// The block below gives the two languages the same names for a 32-bit
// unsigned integer and for a compile-time constant.
//
// Threadgroup sizes are constants, never derived from the device: a
// reduction's order follows from its threadgroup's shape, so adapting the size
// to a GPU would silently change the arithmetic. MetalContext refuses a
// device whose pipelines cannot run these sizes.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#ifndef AZGO_SHADER_TYPES_H
#define AZGO_SHADER_TYPES_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
typedef uint az_uint;
#define AZ_CONSTANT constant constexpr
#else
#include <cstdint>
typedef uint32_t az_uint;
#define AZ_CONSTANT constexpr
#endif

// Threads per threadgroup.
AZ_CONSTANT az_uint kGemmThreads = 128;      // gemm_simd: 4 simdgroups of 32, one 64x64 tile
AZ_CONSTANT az_uint kElementThreads = 256;   // one thread per element
AZ_CONSTANT az_uint kReduceThreads = 256;    // one threadgroup per reduction (Phase B)
AZ_CONSTANT az_uint kIm2colGroupX = 64;      // im2col3: 64 columns x 4 rows of col
AZ_CONSTANT az_uint kIm2colGroupY = 4;

// fill: data[i] = value for i < count.
struct FillParams {
    az_uint count;
    float value;
};

// gemm_simd: row-major C[M x N] = alpha * op(A) * op(B) + beta * C, as
// Backend::gemm. op(A) is M x K (A stored K x M when transA), op(B) is K x N
// (B stored N x K when transB). C is not read when beta == 0.
AZ_CONSTANT az_uint kGemmTile = 64;    // a threadgroup computes a 64 x 64 tile of C
AZ_CONSTANT az_uint kGemmStepK = 16;   // K advances 16 at a time through threadgroup memory
struct GemmParams {
    az_uint M, N, K;
    az_uint transA, transB;
    float alpha, beta;
};

// transpose_input: [B][planes][area] -> [planes][B][area].
struct TransposeParams {
    az_uint B, planes, area;
};

// im2col3: in [C][B][n * n] -> col [C * 9][B * n * n], as Backend::im2col with k = 3.
struct Im2colParams {
    az_uint C, B, n;
};

// bn_infer: x, y [C][M]; per channel gamma, beta, runMean, runVar.
AZ_CONSTANT float kBnEps = 1e-5f;   // BatchNorm2d::kEps
struct BnInferParams {
    az_uint C, M;
};

// relu_forward, add_into, tanh_inplace: count elements.
struct CountParams {
    az_uint count;
};

// policy_flatten: act [2][B][area] -> flat [B][2 * area].
struct FlattenParams {
    az_uint B, area;
};

// bias_add_rows: y [rows][cols] += b [cols] per row.
struct BiasParams {
    az_uint rows, cols;
};

#endif // AZGO_SHADER_TYPES_H
