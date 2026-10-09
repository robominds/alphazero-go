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

// ---------------------------------------------------------------- gemm

// Backend::gemm: row-major C[M x N] = alpha * op(A) * op(B) + beta * C.
//
// A threadgroup of 128 threads (4 simdgroups of 32) computes one 64 x 64 tile
// of C. Each simdgroup owns a 32 x 32 quarter, held as 4 x 4 accumulators of
// 8 x 8 simdgroup matrices: the GPU's matrix instructions multiply and add a
// whole 8 x 8 block at once. K advances 16 at a time: the threadgroup first
// copies a 64 x 16 slice of op(A) and a 16 x 64 slice of op(B) into
// threadgroup memory (zeros past M, N or K), then each simdgroup multiplies
// its blocks from there, 8 values of k per instruction.
//
// Batch independence (what the batching evaluator relies on). The sum over k
// always runs over the whole of K, from 0 to K in steps of 16, in one
// threadgroup: K is never split across threadgroups. Every 8 x 8 output
// block, wherever it lies, runs the same instruction sequence over the same
// k order, and its operands are its own rows of op(A) and columns of op(B)
// (padding past M or N only adds rows and columns of zeros that are never
// stored). So an element's bits depend only on its row of op(A), its column
// of op(B) and K: not on M, N, its tile, its place in the tile, its
// neighbours or the run. In a convolution a sample's outputs read only its
// own columns of the im2col matrix; in a Linear layer only its own row of
// activations. Hence a position's outputs are the same bits alone or in any
// batch. tests/test_metal.cpp checks this bitwise; any change here that
// splits K or lets a tile's shape change the arithmetic must fail it.
kernel void gemm_simd(device const float* A [[buffer(0)]],
                      device const float* B [[buffer(1)]],
                      device float* C [[buffer(2)]],
                      constant GemmParams& p [[buffer(3)]],
                      uint2 tile [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float As[kGemmTile * kGemmStepK];   // [m][k]; reused to stage results at the end
    threadgroup float Bs[kGemmStepK * kGemmTile];   // [k][n]
    const uint sg = tid / 32;                        // this thread's simdgroup, 0..3
    const uint row0 = tile.y * kGemmTile, col0 = tile.x * kGemmTile;
    const uint sr = (sg / 2) * 32, sc = (sg % 2) * 32;   // the simdgroup's quarter of the tile
    simdgroup_float8x8 acc[4][4];
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 4; ++i)
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.K; k0 += kGemmStepK) {
        // Stage op(A)[row0 .. +64][k0 .. +16] and op(B)[k0 .. +16][col0 .. +64].
        for (uint e = tid; e < kGemmTile * kGemmStepK; e += kGemmThreads) {
            uint m, k;
            if (p.transA) { k = e / kGemmTile; m = e % kGemmTile; }   // read along A's stored rows
            else { m = e / kGemmStepK; k = e % kGemmStepK; }
            const uint gm = row0 + m, gk = k0 + k;
            float v = 0.0f;
            if (gm < p.M && gk < p.K) v = p.transA ? A[gk * p.M + gm] : A[gm * p.K + gk];
            As[m * kGemmStepK + k] = v;
        }
        for (uint e = tid; e < kGemmStepK * kGemmTile; e += kGemmThreads) {
            uint k, n;
            if (p.transB) { n = e / kGemmStepK; k = e % kGemmStepK; }
            else { k = e / kGemmTile; n = e % kGemmTile; }
            const uint gn = col0 + n, gk = k0 + k;
            float v = 0.0f;
            if (gn < p.N && gk < p.K) v = p.transB ? B[gn * p.K + gk] : B[gk * p.N + gn];
            Bs[k * kGemmTile + n] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("clang loop unroll(full)")
        for (uint kk = 0; kk < kGemmStepK; kk += 8) {
            simdgroup_float8x8 a[4], b[4];
            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i < 4; ++i)
                simdgroup_load(a[i], As + (sr + 8 * i) * kGemmStepK + kk, kGemmStepK);
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < 4; ++j)
                simdgroup_load(b[j], Bs + kk * kGemmTile + sc + 8 * j, kGemmTile);
            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i < 4; ++i)
                _Pragma("clang loop unroll(full)")
                for (uint j = 0; j < 4; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back one 8 x 8 block at a time through the simdgroup's own 64
    // floats of As (so threadgroup memory stays at 8 KB and several
    // threadgroups share a core); each lane writes two of the 64 elements.
    threadgroup float* stage = As + sg * 64;
    const uint lane = tid % 32;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 4; ++i)
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < 4; ++j) {
            simdgroup_store(acc[i][j], stage, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                const uint gm = row0 + sr + 8 * i + e / 8, gn = col0 + sc + 8 * j + e % 8;
                if (gm >= p.M || gn >= p.N) continue;
                float r = p.alpha * stage[e];
                if (p.beta != 0.0f) r += p.beta * C[gm * p.N + gn];   // beta == 0: C is never read
                C[gm * p.N + gn] = r;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
}
