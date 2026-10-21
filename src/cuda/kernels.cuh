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

// data[0 .. n) = value.
void launchFill(float* data, size_t n, float value, cudaStream_t stream);
// The attributes of the fill kernel: its binaryVersion is the architecture of
// the machine code the runtime picked for this device.
cudaError_t fillAttributes(cudaFuncAttributes* attributes);

} // namespace cuda
} // namespace azgo
