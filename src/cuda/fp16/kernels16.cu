// The 16-bit trunk's kernels (kernels16.cuh). The convolution is an implicit
// GEMM on the tensor cores:
//
//   D[pixel][cout] = sum over k of col[pixel][k] * w16[cout][k],   k = tap * cinPadded + channel
//
// col[pixel][tap * cinPadded + c] is the input at the pixel moved by the tap
// (dy, dx) = (tap / 3 - 1, tap % 3 - 1), channel c, or 0 off the board: what
// im2col would write, read straight from the NHWC input and never stored.
// The pixels are the GEMM's rows, so both operands have k contiguous in
// memory: a step of 32 channels of one tap is 64 bytes per pixel and per
// output channel, four 16-byte cp.async copies, and every operand fragment
// is one ldmatrix.
//
// The order of the arithmetic, which is what makes the result independent
// of the batch:
//
//   acc = 0;  for s in 0 .. 9 * cinPadded / 16 - 1:
//       acc = mma(col[pixel][16 s .. 16 s + 15], w16[cout][16 s .. 16 s + 15], acc)
//
// one mma.sync.m16n8k16 step at a time, in k order, never split, for every
// output. A step's result for one output depends on that row's 16 values,
// that column's 16 values and that output's accumulator alone: the tensor
// core does not mix outputs. A pixel's row holds its own sample's
// neighbourhood (zeros off the board, and zeros for the rows past the last
// pixel), so a position's outputs are the same bits alone or in any batch,
// at any place in it, and for every tile shape. The tests hold the kernel to
// that on the GPU that runs it.
//
// Then, per output and in fp32, what the cuda backend's fused convolution
// does after its gemm: v = scale * acc + shift (two roundings), + skip, then
// v > 0 ? v : 0; converted to 16 bits (round to nearest even; fp16 clamped
// to its largest finite value, so an outlier saturates rather than becoming
// infinity) for the next layer, or stored fp32 channel-major for the heads.
//
// The library compiles CUDA with --fmad=false (cmake/azgo_cuda.cmake); the
// epilogue states its roundings with __fmul_rn and __fadd_rn anyway. The mma
// needs sm_80 or newer: built for an older architecture the kernel traps,
// and cuda16Available refuses that GPU before any pass.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include "kernels16.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace azgo {
namespace cuda16 {

namespace {

// The shared-memory ring of k steps. Two (40 KB a block with 128 channels)
// let two blocks share an SM; three or four were 20-25% slower on the RTX
// 4070 Ti SUPER (docs/results/cuda16.txt).
constexpr int kStages = 2;
constexpr int kRowHalves = kStepK + 8;   // a staged row: 32 values and 8 of padding (80 bytes, no bank conflicts)
constexpr float kFp16Max = 65504.0f;

unsigned blocksFor(size_t count) {
    const size_t blocks = (count + kElementwiseThreads - 1) / kElementwiseThreads;
    return static_cast<unsigned>(blocks < 65535 ? (blocks > 0 ? blocks : 1) : 65535);
}

// The two 16-bit types behind one interface.
template <class T> struct Ops;
template <> struct Ops<__half> {
    using Pair = __half2;
    static __device__ __forceinline__ __half fromFloat(float v) { return __float2half_rn(v); }
    static __device__ __forceinline__ float toFloat(__half v) { return __half2float(v); }
    static __device__ __forceinline__ Pair pairFromFloats(float a, float b) {
        return __floats2half2_rn(fminf(a, kFp16Max), fminf(b, kFp16Max));   // a, b >= 0 (after ReLU)
    }
    static __device__ __forceinline__ float2 pairToFloats(Pair p) { return __half22float2(p); }
};
template <> struct Ops<__nv_bfloat16> {
    using Pair = __nv_bfloat162;
    static __device__ __forceinline__ __nv_bfloat16 fromFloat(float v) { return __float2bfloat16_rn(v); }
    static __device__ __forceinline__ float toFloat(__nv_bfloat16 v) { return __bfloat162float(v); }
    static __device__ __forceinline__ Pair pairFromFloats(float a, float b) { return __floats2bfloat162_rn(a, b); }
    static __device__ __forceinline__ float2 pairToFloats(Pair p) { return __bfloat1622float2(p); }
};

__device__ __forceinline__ unsigned sharedAddress(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// 16 bytes from global to shared memory, asynchronously; with srcBytes = 0
// it writes 16 zero bytes and reads nothing.
__device__ __forceinline__ void copyAsync16(void* smem, const void* gmem, int srcBytes) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sharedAddress(smem)), "l"(gmem),
                 "r"(srcBytes));
#endif
}
__device__ __forceinline__ void copyAsyncCommit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N> __device__ __forceinline__ void copyAsyncWait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// Four 8x8 matrices of 16-bit values from shared memory: lane l gives the
// address of one row (lanes 8i .. 8i + 7: matrix i), and receives in r[i]
// two neighbouring values of row l / 4 of matrix i.
__device__ __forceinline__ void loadMatrices(unsigned (&r)[4], const void* smem) {
#if __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(sharedAddress(smem)));
#endif
}

// acc = A (16 x 16, row-major fragment) . B (16 x 8, column fragment) + acc,
// on the tensor cores with fp32 accumulation.
template <class T> __device__ __forceinline__ void mma(float (&acc)[4], const unsigned (&a)[4], unsigned b0, unsigned b1);
template <>
__device__ __forceinline__ void mma<__half>(float (&acc)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}
template <>
__device__ __forceinline__ void mma<__nv_bfloat16>(float (&acc)[4], const unsigned (&a)[4], unsigned b0,
                                                   unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

// The convolution. A block of 256 threads computes kTilePixels pixels x BN =
// 16 * NI output channels; warp w computes 32 pixels (rows (w % 4) * 32 ..)
// x 8 * NI channels (columns (w / 4) * 8 * NI ..), as 2 x NI mma tiles of
// 16 x 8.
template <class T, int NI>
__global__ void __launch_bounds__(kConvThreads) conv16(Conv16Args a) {
    constexpr int BM = kTilePixels, BN = 16 * NI, BK = kStepK, LD = kRowHalves;
    static_assert(NI % 2 == 0, "B fragments are loaded two tiles at a time");
    static_assert(BM * 4 == 2 * kConvThreads, "each thread stages two 16-byte pieces of A per step");
    extern __shared__ __align__(16) unsigned char smem[];
    T* As = reinterpret_cast<T*>(smem);    // [kStages][BM][LD]
    T* Bs = As + kStages * BM * LD;        // [kStages][BN][LD]
    const T* in = static_cast<const T*>(a.in);
    const T* w = static_cast<const T*>(a.w);

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int wm = warp % 4, wn = warp / 4;
    const int pix0 = blockIdx.x * BM, ch0 = blockIdx.y * BN;
    const int n = a.n, area = n * n, N = a.batch * area;
    const int cinP = a.cinPadded, K = 9 * cinP, steps = K / BK, cBlocks = cinP / BK;

    // The two A pieces this thread stages every step: rows tid / 4 and
    // tid / 4 + 64, bytes 16 * (tid % 4) of the row. Their pixels are found once.
    const int piece = tid % 4;
    int rowPixel[2], rowY[2], rowX[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int p = pix0 + tid / 4 + 64 * i;
        rowPixel[i] = p;
        const int q = p < N ? p % area : 0;
        rowY[i] = p < N ? q / n : -1000;   // a row past the last pixel: never on the board
        rowX[i] = q % n;
    }

    // Stage step `step` (tap step / cBlocks, channels 32 * (step % cBlocks) ..)
    // into ring slot `slot`.
    auto stage = [&](int slot, int step) {
        const int tap = step / cBlocks, c0 = (step % cBlocks) * BK;
        const int dy = tap / 3 - 1, dx = tap % 3 - 1;
        T* as = As + slot * BM * LD;
        T* bs = Bs + slot * BN * LD;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int y = rowY[i] + dy, x = rowX[i] + dx;
            const bool on = y >= 0 && y < n && x >= 0 && x < n;
            const T* src = on ? in + static_cast<size_t>(rowPixel[i] + dy * n + dx) * cinP + c0 + 8 * piece : in;
            copyAsync16(as + (tid / 4 + 64 * i) * LD + 8 * piece, src, on ? 16 : 0);
        }
        for (int e = tid; e < BN * 4; e += kConvThreads) {
            const int row = e / 4, part = e % 4;
            copyAsync16(bs + row * LD + 8 * part, w + static_cast<size_t>(ch0 + row) * K + step * BK + 8 * part, 16);
        }
    };

    float acc[2][NI][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.0f;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < steps) stage(s, s);
        copyAsyncCommit();
    }
    for (int step = 0; step < steps; ++step) {
        copyAsyncWait<kStages - 2>();   // this step's copies have landed (for this thread) ...
        __syncthreads();                // ... and for every thread; the slot refilled below is free
        const int next = step + kStages - 1;
        if (next < steps) stage(next % kStages, next);
        copyAsyncCommit();
        const T* as = As + (step % kStages) * BM * LD;
        const T* bs = Bs + (step % kStages) * BN * LD;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            unsigned fa[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i)
                loadMatrices(fa[i], as + (wm * 32 + i * 16 + lane % 16) * LD + kk + (lane / 16) * 8);
#pragma unroll
            for (int j = 0; j < NI; j += 2) {
                unsigned fb[4];   // tiles j (fb[0], fb[1]) and j + 1 (fb[2], fb[3])
                loadMatrices(fb, bs + (wn * 8 * NI + j * 8 + lane % 8 + (lane / 16) * 8) * LD + kk +
                                     ((lane / 8) % 2) * 8);
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    mma<T>(acc[i][j], fa[i], fb[0], fb[1]);
                    mma<T>(acc[i][j + 1], fa[i], fb[2], fb[3]);
                }
            }
        }
    }
    copyAsyncWait<0>();

    // The epilogue. acc[i][j][0..1] are row g, columns 2t and 2t + 1 of tile
    // (i, j); acc[i][j][2..3] the same columns of row g + 8.
    const int g = lane / 4, t = lane % 4;
    using Pair = typename Ops<T>::Pair;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int p = pix0 + wm * 32 + i * 16 + g + 8 * half;
            if (p >= N) continue;
#pragma unroll
            for (int j = 0; j < NI; ++j) {
                const int c = ch0 + wn * 8 * NI + j * 8 + 2 * t;
                float v[2];
#pragma unroll
                for (int q = 0; q < 2; ++q)
                    v[q] = __fadd_rn(__fmul_rn(a.scale[c + q], acc[i][j][2 * half + q]), a.shift[c + q]);
                const size_t at = static_cast<size_t>(p) * a.coutPadded + c;
                if (a.skip) {
                    const float2 s = Ops<T>::pairToFloats(*reinterpret_cast<const Pair*>(static_cast<const T*>(a.skip) + at));
                    v[0] = __fadd_rn(v[0], s.x);
                    v[1] = __fadd_rn(v[1], s.y);
                }
                v[0] = v[0] > 0.0f ? v[0] : 0.0f;
                v[1] = v[1] > 0.0f ? v[1] : 0.0f;
                if (a.out16) *reinterpret_cast<Pair*>(static_cast<T*>(a.out16) + at) = Ops<T>::pairFromFloats(v[0], v[1]);
                if (a.out32) {
#pragma unroll
                    for (int q = 0; q < 2; ++q)
                        if (c + q < a.cout) a.out32[static_cast<size_t>(c + q) * N + p] = v[q];
                }
            }
        }
}

template <class T, int NI> void launchTile(const Conv16Args& a, cudaStream_t s) {
    const size_t bytes = conv16SharedBytes(NI);
    // Once per instantiation, thread-safely (a function-local static): the
    // shared memory a block may ask for beyond the default 48 KB. A failure
    // shows as the launch's error.
    static const cudaError_t configured =
        cudaFuncSetAttribute(conv16<T, NI>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    (void)configured;
    const int N = a.batch * a.n * a.n;
    const dim3 grid((N + kTilePixels - 1) / kTilePixels, a.coutPadded / (16 * NI));
    conv16<T, NI><<<grid, kConvThreads, bytes, s>>>(a);
}

template <class T> void launchTyped(const Conv16Args& a, int ni, cudaStream_t s) {
    switch (ni) {
        case 8: launchTile<T, 8>(a, s); break;
        case 6: launchTile<T, 6>(a, s); break;
        case 4: launchTile<T, 4>(a, s); break;
        default: launchTile<T, 2>(a, s); break;
    }
}

template <class T>
__global__ void input_to_nhwc(const float* __restrict__ stage, int B, int P, int A, int Pp, T* __restrict__ out) {
    const size_t count = static_cast<size_t>(B) * A * Pp;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
        const int c = static_cast<int>(i % Pp);
        const size_t pixel = i / Pp, b = pixel / A, at = pixel % A;
        out[i] = Ops<T>::fromFloat(c < P ? stage[(b * P + c) * A + at] : 0.0f);
    }
}

template <class T>
__global__ void weights_to_16(const float* __restrict__ w, int cout, int cin, int coutP, int cinP, T* __restrict__ out) {
    const size_t K = 9 * static_cast<size_t>(cinP), count = static_cast<size_t>(coutP) * K;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
        const int m = static_cast<int>(i / K), k = static_cast<int>(i % K);
        const int tap = k / cinP, c = k % cinP;
        const float v = m < cout && c < cin ? w[(static_cast<size_t>(m) * cin + c) * 9 + tap] : 0.0f;
        out[i] = Ops<T>::fromFloat(v);
    }
}

template <class T> __global__ void widen(const T* __restrict__ in, size_t count, float* __restrict__ out) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
        out[i] = Ops<T>::toFloat(in[i]);
}

} // namespace

int channelTileFor(int coutPadded) {
    if (coutPadded % 128 == 0) return 8;
    if (coutPadded % 96 == 0) return 6;
    if (coutPadded % 64 == 0) return 4;
    return 2;
}

size_t conv16SharedBytes(int ni) {
    return static_cast<size_t>(kStages) * (kTilePixels + 16 * ni) * kRowHalves * kHalfBytes;
}

void launchConv16(Half type, const Conv16Args& a, int ni, cudaStream_t stream) {
    if (a.batch <= 0 || a.coutPadded <= 0) return;
    if (type == Half::Fp16) launchTyped<__half>(a, ni, stream);
    else launchTyped<__nv_bfloat16>(a, ni, stream);
}

void launchInputToNhwc(Half type, const float* stage, int B, int planes, int area, int planesPadded, void* out,
                       cudaStream_t s) {
    const size_t count = static_cast<size_t>(B) * area * planesPadded;
    if (!count) return;
    if (type == Half::Fp16)
        input_to_nhwc<__half><<<blocksFor(count), kElementwiseThreads, 0, s>>>(stage, B, planes, area, planesPadded,
                                                                             static_cast<__half*>(out));
    else
        input_to_nhwc<__nv_bfloat16><<<blocksFor(count), kElementwiseThreads, 0, s>>>(
            stage, B, planes, area, planesPadded, static_cast<__nv_bfloat16*>(out));
}

void launchWeightsTo16(Half type, const float* w, int cout, int cin, int coutPadded, int cinPadded, void* out,
                       cudaStream_t s) {
    const size_t count = static_cast<size_t>(coutPadded) * 9 * cinPadded;
    if (!count) return;
    if (type == Half::Fp16)
        weights_to_16<__half><<<blocksFor(count), kElementwiseThreads, 0, s>>>(w, cout, cin, coutPadded, cinPadded,
                                                                             static_cast<__half*>(out));
    else
        weights_to_16<__nv_bfloat16><<<blocksFor(count), kElementwiseThreads, 0, s>>>(
            w, cout, cin, coutPadded, cinPadded, static_cast<__nv_bfloat16*>(out));
}

void launchWiden(Half type, const void* in, size_t count, float* out, cudaStream_t s) {
    if (!count) return;
    if (type == Half::Fp16)
        widen<__half><<<blocksFor(count), kElementwiseThreads, 0, s>>>(static_cast<const __half*>(in), count, out);
    else
        widen<__nv_bfloat16><<<blocksFor(count), kElementwiseThreads, 0, s>>>(
            static_cast<const __nv_bfloat16*>(in), count, out);
}

} // namespace cuda16
} // namespace azgo
