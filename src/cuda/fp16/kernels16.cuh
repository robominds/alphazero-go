// The 16-bit trunk's kernels (backend = cuda16): the tensor-core 3x3
// convolution and the two conversions that feed it. kernels16.cu holds them
// and the reasons behind their arithmetic; the design is
// docs/superpowers/specs/2026-10-24-cuda-fp16-design.md.
//
// They keep the rules of ../kernels.cuh: no atomics, exactly one thread
// writes each output, no reduction across threads, block sizes are the
// constants below and a grid depends only on tensor shapes, and nothing
// reads memory that was not written first. The one new fact they rest on is
// that a tensor-core mma computes each output from its own row and column of
// the operands and its own accumulator; tests/test_cuda16.cpp holds the
// whole pass to bitwise batch independence on the machine that runs it.
//
// This header and kernels16.cu depend on the CUDA runtime alone, so
// tools/cuda_bench can compile them on their own.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace azgo {
namespace cuda16 {

// The 16-bit operand type: IEEE half (10 mantissa bits) or bfloat16 (7).
enum class Half { Fp16 = 0, Bf16 = 1 };

constexpr int kConvThreads = 256;   // 8 warps: 4 along pixels x 2 along channels
constexpr int kTilePixels = 128;    // pixels (GEMM rows) per block
constexpr int kStepK = 32;          // k per shared-memory step: one tap, 32 channels
constexpr int kChannelAlign = 32;   // activations' channels are padded to a multiple of this
constexpr int kElementwiseThreads = 256;

// The channel count with its padding: a multiple of kChannelAlign.
inline int paddedChannels(int channels) { return (channels + kChannelAlign - 1) / kChannelAlign * kChannelAlign; }

// The channel tile: 16 * ni output channels per block, ni in {8, 6, 4, 2}
// (128, 96, 64 or 32 channels), the largest that divides the padded output
// channel count. Chosen from the layer's shape alone, never the batch; every
// choice gives the same bits (each output is the same chain of mma steps).
int channelTileFor(int coutPadded);

// A 3x3 convolution, 'same' padding, with its batch norm, optional skip and
// ReLU. Activations are NHWC: [pixel][channels padded], pixel = sample * n *
// n + y * n + x, 16-bit; pad channels hold 0.
struct Conv16Args {
    const void* in = nullptr;      // [batch * n * n][cinPadded], 16-bit
    const void* w = nullptr;       // [coutPadded][9 * cinPadded], 16-bit, column tap * cinPadded + channel
    int batch = 0, n = 0;
    int cinPadded = 0, coutPadded = 0;
    int cout = 0;                  // the real output channels (out32's rows)
    const float* scale = nullptr;  // [coutPadded], 0 past cout
    const float* shift = nullptr;  // [coutPadded], 0 past cout
    const void* skip = nullptr;    // [batch * n * n][coutPadded], 16-bit, or null
    void* out16 = nullptr;         // [batch * n * n][coutPadded], 16-bit, or null
    float* out32 = nullptr;        // [cout][batch * n * n] fp32 channel-major (the heads' input), or null
};

// out = ReLU(scale * (col . w^T) + shift (+ skip)) for every pixel and
// output channel, the dot product on the tensor cores with fp32
// accumulation, k from 0 to 9 * cinPadded in order. Writes out16, out32 or
// both. ni: channelTileFor(coutPadded), or a tile to force (tests).
void launchConv16(Half type, const Conv16Args& a, int ni, cudaStream_t stream);
// The dynamic shared memory one conv16 block uses (for tests and bench).
size_t conv16SharedBytes(int ni);

// The batch as the evaluator gives it, stage [B][planes][area] fp32, to NHWC
// 16-bit [B * area][planesPadded] with 0 in the pad channels.
void launchInputToNhwc(Half type, const float* stage, int B, int planes, int area, int planesPadded, void* out,
                       cudaStream_t stream);
// A convolution's weights w [cout][cin][3][3] fp32 to w16 [coutPadded][9 *
// cinPadded], column tap * cinPadded + channel, rounded to nearest even, 0 in
// every pad row and column.
void launchWeightsTo16(Half type, const float* w, int cout, int cin, int coutPadded, int cinPadded, void* out,
                       cudaStream_t stream);
// The 16-bit values of `count` elements widened to fp32 (tests and bench).
void launchWiden(Half type, const void* in, size_t count, float* out, cudaStream_t stream);

// Bytes of one 16-bit element.
constexpr size_t kHalfBytes = 2;

} // namespace cuda16
} // namespace azgo
