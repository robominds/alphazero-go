// The Metal GPU backend, as the rest of the program sees it. backend = metal
// runs the network's forward pass on the Apple GPU with hand-written compute
// kernels (src/metal/kernels.metal), compiled by the OS when the program
// first asks for them. It exists only in a build configured with
// -DAZ_USE_METAL=ON (macOS); every other build still compiles this header,
// answers "not built" and so can refuse backend = metal with a clear message.
//
// Everything Metal-specific stays in src/metal/: no Objective-C type appears
// in this header, which is plain C++17.
#pragma once
#include <cstddef>
#include <string>

namespace azgo {

// True when this binary was built with -DAZ_USE_METAL=ON.
bool metalBuilt();

// True when the GPU can run the kernels: built with Metal, a Metal device
// of Apple GPU family 7 or later (simdgroup matrices; M1 or later), and every
// kernel compiled. The first call creates the process-wide Metal context,
// which compiles the shaders. When false and why is non-null, *why says what
// is missing, e.g. "not built: rebuild with -DAZ_USE_METAL=ON (macOS only)".
bool metalAvailable(std::string* why = nullptr);

// "Apple M5 Max, macOS 27.0.1, shaders 3f1c9a07": the device, the OS (which
// compiles the shaders) and a hash of the shader source. Two runs whose
// descriptions match ran the same GPU arithmetic. Without a usable GPU it is
// metalAvailable's reason.
std::string metalDescription();

#ifdef AZ_USE_METAL
// Test probes: each copies host arrays in, runs one command buffer and copies
// the results out. They exist so tests (and bench) can check a kernel against
// its CPU counterpart; the program itself never calls them. Each throws
// std::runtime_error when the GPU is not usable or a command buffer fails.
namespace metal_probe {

// The GPU's name as Metal reports it, e.g. "Apple M5 Max".
std::string deviceName();

// The first kernel: data[0..n) = value. data holds n + tail floats; all of
// them are copied to the GPU and back, so a test can check the tail is untouched.
void fill(float* data, size_t n, float value, size_t tail = 0);

// Backend::gemm on the GPU (gemm_simd): row-major
// C[M x N] = alpha * op(A) * op(B) + beta * C, tightly packed; C is read only
// when beta != 0.
void gemm(bool transA, bool transB, int M, int N, int K, float alpha, const float* A, const float* B,
          float beta, float* C);

} // namespace metal_probe
#endif

} // namespace azgo
