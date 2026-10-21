// The CUDA kernels' launchers, as the rest of src/cuda/ calls them. Each
// kernel mirrors one CPU function (or, for the fused convolution, a fixed
// sequence of them) so it can be tested against it; kernels.cu holds the
// kernels and the reasons behind their arithmetic.
//
// Rules every kernel follows, so that results are reproducible bit for bit
// and do not depend on how work is batched (the Metal kernels' rules):
//   - no atomics, and exactly one thread writes each output element;
//   - no reduction across threads: a dot product is one thread's fmaf chain
//     over k = 0 .. K-1, in that order, never split;
//   - block sizes are the constants below, and a launch's grid depends only
//     on the tensor shapes;
//   - nothing reads memory that was not written first.
// Every launcher enqueues on `stream` and returns at once; launch errors are
// left for the caller's next cudaGetLastError or synchronize.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace azgo {
namespace cuda {

constexpr int kElementwiseThreads = 256;   // threads per block of the one-thread-per-element kernels

constexpr int kGemmThreads = 256;   // threads per block of the gemm and the fused convolution
constexpr int kGemmStepK = 16;      // the k step through shared memory

// The gemm's tile shapes: BM x BN outputs per block of 256 threads. Every
// shape gives the same bits (each output is the same fmaf chain); the
// choice only changes speed.
enum class GemmTile { Rows32Cols256 = 0, Rows64Cols128 = 1, Rows128Cols128 = 2 };
// The tile for a product whose rows are a layer's output channels (M), by M
// alone, never by the batch: 128x128 for M a multiple of 128, 64x128 for a
// multiple of 64, else 32x256.
GemmTile gemmTileFor(int M);
// A Linear layer's rows are the samples, so its tile is fixed.
constexpr GemmTile kLinearTile = GemmTile::Rows32Cols256;

// C[M][N] = A[M][K] . op(B), all row-major and tightly packed; op(B) is B
// [K][N] (transB false: a convolution's columns) or B [N][K] read transposed
// (transB true: a Linear layer's weights w [out][in]). Each C[m][n] is
// acc = fmaf(A[m][k], op(B)[k][n], acc) for k = 0 .. K-1 from acc = 0: C is
// never read, and its bits depend only on row m of A and column n of op(B).
void launchGemm(bool transB, int M, int N, int K, const float* A, const float* B, float* C, GemmTile tile,
                cudaStream_t stream);

// data[0 .. n) = value.
void launchFill(float* data, size_t n, float value, cudaStream_t stream);
// The attributes of the fill kernel: its binaryVersion is the architecture of
// the machine code the runtime picked for this device.
cudaError_t fillAttributes(cudaFuncAttributes* attributes);

} // namespace cuda
} // namespace azgo
