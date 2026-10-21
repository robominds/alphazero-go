// The CUDA kernels. Each one mirrors a CPU function so it can be tested
// against it on the same inputs (tests/test_cuda.cpp). The rules they all
// follow are in kernels.cuh.
//
// Floating point: the library compiles CUDA with --fmad=false, IEEE division
// and square root and denormals kept (cmake/azgo_cuda.cmake), so a * b + c is
// two roundings unless the code says fmaf. Where the CPU writes a * b + c the
// kernels write __fadd_rn(__fmul_rn(a, b), c), which states it twice.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include "kernels.cuh"

namespace azgo {
namespace cuda {

namespace {

// Blocks for one thread per element of `count`, capped: each kernel strides
// over the rest, so the grid depends only on the count.
unsigned blocksFor(size_t count) {
    const size_t blocks = (count + kElementwiseThreads - 1) / kElementwiseThreads;
    return static_cast<unsigned>(blocks < 65535 ? (blocks > 0 ? blocks : 1) : 65535);
}

} // namespace

// ---------------------------------------------------------------- fill

// The first kernel, a smoke test of the whole path: data[i] = value.
__global__ void fill_kernel(float* data, size_t n, float value) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        data[i] = value;
}

void launchFill(float* data, size_t n, float value, cudaStream_t stream) {
    if (n == 0) return;
    fill_kernel<<<blocksFor(n), kElementwiseThreads, 0, stream>>>(data, n, value);
}

cudaError_t fillAttributes(cudaFuncAttributes* attributes) {
    return cudaFuncGetAttributes(attributes, fill_kernel);
}

} // namespace cuda
} // namespace azgo
