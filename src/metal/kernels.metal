// The GPU kernels, in the Metal Shading Language (a C++ dialect). Each one
// mirrors a CPU function so it can be tested against it on the same inputs.
//
// CMake embeds this file in the library (with shader_types.h pasted in where
// it is included below), and the OS compiles it when the program first asks
// for the GPU: no shader compiler runs at build time. It is compiled in
// MTLMathModeSafe, so the compiler does not reassociate floating-point
// arithmetic, and transcendentals are written precise:: explicitly.
//
// Rules every kernel here follows, so that a run's results are reproducible
// bit for bit and do not depend on how work is batched:
//   - no atomics, and exactly one thread writes each output element;
//   - a reduction runs inside one threadgroup, in a fixed tree, or in a
//     separate pass in a fixed order;
//   - threadgroup sizes are the constants of shader_types.h, and a dispatch's
//     shape depends only on the tensor shapes;
//   - nothing reads memory that was not written first.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include "shader_types.h"
using namespace metal;

// ---------------------------------------------------------------- fill

// The first kernel, a smoke test of the whole path: data[i] = value.
kernel void fill(device float* data [[buffer(0)]],
                 constant FillParams& p [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    data[i] = p.value;
}
