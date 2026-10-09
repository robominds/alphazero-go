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

// ---------------------------------------------------------------- forward layers
//
// One thread per output element, each reading only the inputs that element
// is made from. None of them reads another sample's data, so in inference
// mode a sample's outputs are the same bits alone or in any batch.

// Network::forward's first step: [B][planes][area] -> [planes][B][area].
// Channel-major lets a convolution treat the whole batch as one wide matrix.
kernel void transpose_input(device const float* in [[buffer(0)]],
                            device float* out [[buffer(1)]],
                            constant TransposeParams& p [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= p.planes * p.B * p.area) return;
    const uint a = i % p.area, pb = i / p.area;          // out index (plane * B + b) * area + a
    const uint b = pb % p.B, plane = pb / p.B;
    out[i] = in[(b * p.planes + plane) * p.area + a];
}

// Backend::im2col for k = 3 ('same' padding): row (c * 3 + dy) * 3 + dx of
// col holds, for every pixel of every sample, the input value at offset
// (dy - 1, dx - 1) from it, or 0 off the board. One thread per element of
// col: x is the pixel (column), y the row. Padding reads 0, never a
// neighbouring sample's board.
kernel void im2col3(device const float* in [[buffer(0)]],
                    device float* col [[buffer(1)]],
                    constant Im2colParams& p [[buffer(2)]],
                    uint2 gid [[thread_position_in_grid]]) {
    const uint area = p.n * p.n, columns = p.B * area;
    if (gid.x >= columns || gid.y >= p.C * 9) return;
    const uint c = gid.y / 9, r = gid.y % 9;
    const uint b = gid.x / area, q = gid.x % area;
    const int y = int(q / p.n) + int(r / 3) - 1, x = int(q % p.n) + int(r % 3) - 1;
    float v = 0.0f;
    if (y >= 0 && y < int(p.n) && x >= 0 && x < int(p.n))
        v = in[(c * p.B + b) * area + uint(y) * p.n + uint(x)];
    col[gid.y * columns + gid.x] = v;
}

// BatchNorm2d::forwardInfer: the running statistics folded into one scale and
// shift per channel, y = scale * x + shift, written as the CPU writes it.
kernel void bn_infer(device const float* x [[buffer(0)]],
                     device float* y [[buffer(1)]],
                     device const float* gamma [[buffer(2)]],
                     device const float* beta [[buffer(3)]],
                     device const float* runMean [[buffer(4)]],
                     device const float* runVar [[buffer(5)]],
                     constant BnInferParams& p [[buffer(6)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= p.C * p.M) return;
    const uint c = i / p.M;
    const float scale = precise::divide(gamma[c], precise::sqrt(runVar[c] + kBnEps));
    const float shift = beta[c] - scale * runMean[c];
    y[i] = scale * x[i] + shift;
}

// reluForward, in place. x > 0 ? x : 0 maps -0 and NaN to +0, as on the CPU.
kernel void relu_forward(device float* x [[buffer(0)]],
                         constant CountParams& p [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    const float v = x[i];
    x[i] = v > 0.0f ? v : 0.0f;
}

// The skip connection: dst += src.
kernel void add_into(device float* dst [[buffer(0)]],
                     device const float* src [[buffer(1)]],
                     constant CountParams& p [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    dst[i] = dst[i] + src[i];
}

// The policy head's per-sample flatten: [2][B][area] -> [B][2 * area],
// feature index c * area + point.
kernel void policy_flatten(device const float* act [[buffer(0)]],
                           device float* flat [[buffer(1)]],
                           constant FlattenParams& p [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    const uint features = 2 * p.area;
    if (i >= p.B * features) return;
    const uint b = i / features, f = i % features;
    const uint c = f / p.area, q = f % p.area;
    flat[i] = act[(c * p.B + b) * p.area + q];
}

// The bias loop of Linear::forward: every row of y [rows][cols] += b.
kernel void bias_add_rows(device float* y [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          constant BiasParams& p [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.cols) return;
    y[i] = y[i] + b[i % p.cols];
}

// The value and ownership heads' last step, in place.
kernel void tanh_inplace(device float* x [[buffer(0)]],
                         constant CountParams& p [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    x[i] = precise::tanh(x[i]);
}
