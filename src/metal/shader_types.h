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

#endif // AZGO_SHADER_TYPES_H
