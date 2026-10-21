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
#include "azgo/layers.hpp"

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

// ---------------------------------------------------------------- the gemm

// How a gemm reads op(B).
enum class Op { NN, NT };

// C[M][N] = A[M][K] . op(B): the CUDA counterpart of the Metal gemm_simd's
// contract. A block of 256 threads computes one BM x BN tile of C, each
// thread a TM x TN piece of it (TN in runs of 4 columns, read as float4).
// K is walked from 0 to K in steps of kGemmStepK through shared memory,
// always in order and never split, and each output is one fmaf chain:
//
//   acc = 0;  for k in 0 .. K-1:  acc = fmaf(A[m][k], op(B)[k][n], acc)
//
// Rows and columns beyond M, N or K load as 0. A zero past K adds
// fmaf(0, 0, acc) = acc exactly (acc, starting at +0, never becomes -0), so
// the padding changes no bit. An element's bits therefore depend only on its
// row of A and its column of op(B): not on M, N, the tile shape, its block,
// its neighbours or the run. In a convolution a column of op(B) is one pixel
// of one sample; in a Linear layer a row of A is one sample: either way a
// sample's outputs are the same bits alone or in any batch.
template <int BM, int BN, int TM, int TN, Op op>
__global__ void __launch_bounds__(kGemmThreads)
gemm_tiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    constexpr int BK = kGemmStepK;
    constexpr int kGroups = TN / 4;            // a thread's columns: kGroups runs of 4
    constexpr int kGroupStride = BN / kGroups;
    static_assert((BM / TM) * (BN / TN) == kGemmThreads, "one output piece per thread");
    static_assert(TN % 4 == 0 && TM % 4 == 0, "float4 fragments");
    __shared__ __align__(16) float As[BK][BM];   // transposed: As[k][m]
    __shared__ __align__(16) float Bs[BK][BN];
    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN), ty = tid / (BN / TN);
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Stage A[row0 .. +BM][k0 .. +BK] and op(B)[k0 .. +BK][col0 .. +BN].
        for (int e = tid; e < BM * BK; e += kGemmThreads) {
            const int m = e / BK, k = e % BK, gm = row0 + m, gk = k0 + k;
            As[k][m] = (gm < M && gk < K) ? A[static_cast<size_t>(gm) * K + gk] : 0.0f;
        }
        for (int e = tid; e < BK * BN; e += kGemmThreads) {
            if (op == Op::NN) {   // B [K][N]: read along its rows
                const int k = e / BN, n = e % BN, gn = col0 + n, gk = k0 + k;
                Bs[k][n] = (gn < N && gk < K) ? B[static_cast<size_t>(gk) * N + gn] : 0.0f;
            } else {              // B [N][K]: read along its rows, store transposed
                const int n = e / BK, k = e % BK, gn = col0 + n, gk = k0 + k;
                Bs[k][n] = (gn < N && gk < K) ? B[static_cast<size_t>(gn) * K + gk] : 0.0f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; i += 4) {
                const float4 v = *reinterpret_cast<const float4*>(&As[kk][ty * TM + i]);
                a[i] = v.x; a[i + 1] = v.y; a[i + 2] = v.z; a[i + 3] = v.w;
            }
#pragma unroll
            for (int g = 0; g < kGroups; ++g) {
                const float4 v = *reinterpret_cast<const float4*>(&Bs[kk][g * kGroupStride + tx * 4]);
                b[4 * g] = v.x; b[4 * g + 1] = v.y; b[4 * g + 2] = v.z; b[4 * g + 3] = v.w;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gm = row0 + ty * TM + i;
        if (gm >= M) continue;
#pragma unroll
        for (int g = 0; g < kGroups; ++g)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int gn = col0 + g * kGroupStride + tx * 4 + q;
                if (gn < N) C[static_cast<size_t>(gm) * N + gn] = acc[i][4 * g + q];
            }
    }
}

GemmTile gemmTileFor(int M) {
    if (M >= 128 && M % 128 == 0) return GemmTile::Rows128Cols128;
    if (M % 64 == 0) return GemmTile::Rows64Cols128;
    return GemmTile::Rows32Cols256;
}

namespace {

template <int BM, int BN, int TM, int TN>
void launchTile(bool transB, int M, int N, int K, const float* A, const float* B, float* C, cudaStream_t s) {
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    if (transB) gemm_tiled<BM, BN, TM, TN, Op::NT><<<grid, kGemmThreads, 0, s>>>(A, B, C, M, N, K);
    else gemm_tiled<BM, BN, TM, TN, Op::NN><<<grid, kGemmThreads, 0, s>>>(A, B, C, M, N, K);
}

} // namespace

void launchGemm(bool transB, int M, int N, int K, const float* A, const float* B, float* C, GemmTile tile,
                cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    switch (tile) {
        case GemmTile::Rows32Cols256: launchTile<32, 256, 4, 8>(transB, M, N, K, A, B, C, stream); break;
        case GemmTile::Rows64Cols128: launchTile<64, 128, 8, 4>(transB, M, N, K, A, B, C, stream); break;
        case GemmTile::Rows128Cols128: launchTile<128, 128, 8, 8>(transB, M, N, K, A, B, C, stream); break;
    }
}

// ---------------------------------------------------------------- the other layers

// Each of these is one thread per output element, striding over the rest
// (blocksFor), and each output is a copy or a fixed expression of its own
// inputs: per position, so batch-independent by construction.

#define AZGO_FOR_EACH(i, count) \
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < (count); i += size_t(gridDim.x) * blockDim.x)

// Network::forward's first step: [B][planes][A] -> [planes][B][A].
// Channel-major lets a convolution treat the whole batch as one wide matrix.
__global__ void transpose_input(const float* __restrict__ in, float* __restrict__ out, int B, int P, int A) {
    AZGO_FOR_EACH(i, size_t(P) * B * A) {
        const size_t a = i % A, pb = i / A;   // out index (plane * B + b) * A + a
        const size_t b = pb % B, plane = pb / B;
        out[i] = in[(b * P + plane) * A + a];
    }
}

// Backend::im2col for k = 3 ('same' padding): row (c * 3 + dy) * 3 + dx of
// col holds, for every pixel of every sample, the input value at offset
// (dy - 1, dx - 1) from it, or 0 off the board.
__global__ void im2col3(const float* __restrict__ in, float* __restrict__ col, int C, int B, int n) {
    const size_t area = size_t(n) * n, columns = B * area;
    AZGO_FOR_EACH(i, size_t(C) * 9 * columns) {
        const size_t row = i / columns, j = i % columns;
        const int c = static_cast<int>(row / 9), r = static_cast<int>(row % 9);
        const size_t b = j / area, q = j % area;
        const int y = static_cast<int>(q / n) + r / 3 - 1, x = static_cast<int>(q % n) + r % 3 - 1;
        float v = 0.0f;
        if (y >= 0 && y < n && x >= 0 && x < n) v = in[(size_t(c) * B + b) * area + size_t(y) * n + x];
        col[i] = v;
    }
}

// BatchNorm2d::kEps, for device code.
constexpr float kBnEps = 1e-5f;
static_assert(kBnEps == BatchNorm2d::kEps, "the GPU's batch norm epsilon is the CPU's");

// The running statistics folded into one scale and shift per channel, as
// BatchNorm2d::forwardInfer folds them: IEEE division and square root,
// each operation rounded on its own.
__device__ __forceinline__ void bnScaleShift(float gamma, float beta, float mean, float var, float& scale,
                                             float& shift) {
    scale = __fdiv_rn(gamma, __fsqrt_rn(__fadd_rn(var, kBnEps)));
    shift = __fsub_rn(beta, __fmul_rn(scale, mean));
}

// y = scale * x + shift, the multiply and the add rounded separately.
__device__ __forceinline__ float bnApply(float x, float scale, float shift) {
    return __fadd_rn(__fmul_rn(scale, x), shift);
}

__global__ void bn_infer(const float* x, float* y, int C, size_t M, const float* __restrict__ gamma,
                         const float* __restrict__ beta, const float* __restrict__ mean,
                         const float* __restrict__ var) {
    AZGO_FOR_EACH(i, size_t(C) * M) {
        const int c = static_cast<int>(i / M);
        float scale, shift;
        bnScaleShift(gamma[c], beta[c], mean[c], var[c], scale, shift);
        y[i] = bnApply(x[i], scale, shift);
    }
}

// reluForward. x > 0 ? x : 0 maps -0 and NaN to +0, as on the CPU.
__global__ void relu_forward(float* x, size_t count) {
    AZGO_FOR_EACH(i, count) {
        const float v = x[i];
        x[i] = v > 0.0f ? v : 0.0f;
    }
}

// The skip connection: dst += src.
__global__ void add_into(float* __restrict__ dst, const float* __restrict__ src, size_t count) {
    AZGO_FOR_EACH(i, count) dst[i] = __fadd_rn(dst[i], src[i]);
}

// The policy head's per-sample flatten: [2][B][A] -> [B][2A], feature index
// c * A + point.
__global__ void policy_flatten(const float* __restrict__ act, float* __restrict__ flat, int B, int A) {
    const size_t features = 2 * size_t(A);
    AZGO_FOR_EACH(i, B * features) {
        const size_t b = i / features, f = i % features;
        const size_t c = f / A, q = f % A;
        flat[i] = act[(c * B + b) * A + q];
    }
}

// The bias loop of Linear::forward: every row of y [rows][cols] += b.
__global__ void bias_add_rows(float* __restrict__ y, const float* __restrict__ b, int rows, int cols) {
    AZGO_FOR_EACH(i, size_t(rows) * cols) y[i] = __fadd_rn(y[i], b[i % cols]);
}

// The value head's last step, in place.
__global__ void tanh_inplace(float* x, size_t count) {
    AZGO_FOR_EACH(i, count) x[i] = tanhf(x[i]);
}

void launchTransposeInput(const float* in, int B, int P, int A, float* out, cudaStream_t s) {
    const size_t count = size_t(B) * P * A;
    if (count) transpose_input<<<blocksFor(count), kElementwiseThreads, 0, s>>>(in, out, B, P, A);
}

void launchIm2col3(const float* in, int C, int B, int n, float* col, cudaStream_t s) {
    const size_t count = size_t(C) * 9 * B * n * n;
    if (count) im2col3<<<blocksFor(count), kElementwiseThreads, 0, s>>>(in, col, C, B, n);
}

void launchBnInfer(const float* x, float* y, int C, size_t M, const float* gamma, const float* beta,
                   const float* mean, const float* var, cudaStream_t s) {
    const size_t count = size_t(C) * M;
    if (count) bn_infer<<<blocksFor(count), kElementwiseThreads, 0, s>>>(x, y, C, M, gamma, beta, mean, var);
}

void launchRelu(float* x, size_t count, cudaStream_t s) {
    if (count) relu_forward<<<blocksFor(count), kElementwiseThreads, 0, s>>>(x, count);
}

void launchAddInto(float* dst, const float* src, size_t count, cudaStream_t s) {
    if (count) add_into<<<blocksFor(count), kElementwiseThreads, 0, s>>>(dst, src, count);
}

void launchPolicyFlatten(const float* act, int B, int A, float* flat, cudaStream_t s) {
    const size_t count = size_t(2) * B * A;
    if (count) policy_flatten<<<blocksFor(count), kElementwiseThreads, 0, s>>>(act, flat, B, A);
}

void launchBiasAddRows(float* y, const float* b, int rows, int cols, cudaStream_t s) {
    const size_t count = size_t(rows) * cols;
    if (count) bias_add_rows<<<blocksFor(count), kElementwiseThreads, 0, s>>>(y, b, rows, cols);
}

void launchTanh(float* x, size_t count, cudaStream_t s) {
    if (count) tanh_inplace<<<blocksFor(count), kElementwiseThreads, 0, s>>>(x, count);
}

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
