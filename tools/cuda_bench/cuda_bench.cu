// cuda_bench: should azgo grow a CUDA BatchRunner? A measurement spike, not
// part of the library and not built by the main CMake project, in the spirit
// of tools/metal_bench. It needs no libazgo: the network's forward pass is
// rebuilt here from its shapes, with random weights, as the Metal backend's
// kernels do it (docs/superpowers/specs/2026-10-09-metal-gpu-backend-design.md,
// "Forward kernels").
//
//   gemm     a hand-written tiled fp32 SGEMM (three tile shapes) and cuBLAS
//            at the project's conv-gemm shapes: M = filters, K = filters * 9
//            (the stem: planes * 9), N = n * n * batch, for 9x9 6x64 and
//            6x96, 13x13 6x96 and 19x19 10x128 at batch 8, 144 and 256,
//            with the error against a double-precision reference.
//   det      determinism: a sample's output columns inside batches of 8, 144
//            and 256 against the sample alone, bitwise, for the hand kernel
//            (every tile shape) and for cuBLAS; repeated runs bitwise.
//   forward  a whole inference pass (transpose, im2col, gemm, batch norm,
//            ReLU, the skip connection, the policy and value heads, tanh)
//            per network and batch: one pass at a time with the copies in
//            and out (what one hub slot waits for) and two streams kept full
//            (the hub's two slots); evals/s; and whether a position's
//            outputs are bitwise the same alone and in a batch.
//
// The hand kernel is the CUDA counterpart of the Metal gemm_simd: a tile of C
// per block, K walked from 0 to K in steps of BK through shared memory, never
// split, each output one fmaf chain in k order. An element's bits therefore
// depend only on its row of A and its column of B: not on M, N, the tile
// shape, its tile, its neighbours or the run. All arithmetic is fp32 (no
// TF32, no tensor cores).
//
//   tools/cuda_bench/build.sh && tools/cuda_bench/build/cuda_bench all
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace {

#define CUDA_OK(x)                                                                          \
    do {                                                                                    \
        cudaError_t e_ = (x);                                                               \
        if (e_ != cudaSuccess) {                                                            \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)
#define CUBLAS_OK(x)                                                                        \
    do {                                                                                    \
        cublasStatus_t s_ = (x);                                                            \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                                  \
            std::fprintf(stderr, "%s:%d: %s: cublas status %d\n", __FILE__, __LINE__, #x, int(s_)); \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)

double gSeconds = 0.5;   // minimum time per measurement (--seconds)

// ---------------------------------------------------------------- the gemm

// A fused 3x3 convolution (kConv): B is not read; its elements come straight
// from the layer input [C][batch][n][n] (an implicit im2col), and the
// epilogue applies batch norm, the skip connection and ReLU.
struct ConvArgs {
    const float* in = nullptr;   // [C][batch][n][n]
    int batch = 0, n = 0;
    const float* scale = nullptr;   // per output channel (row of C)
    const float* shift = nullptr;
    const float* skip = nullptr;    // [M][N] or null
};

// C[M][N] = A[M][K] . B[K][N], all row-major (the convolution's weights times
// its column matrix; the column matrix's column j is pixel j % area of
// sample j / area). BM x BN tile per block, TM x TN outputs per thread.
template <int BM, int BN, int BK, int TM, int TN, bool kConv = false>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
gemm_tiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K,
           ConvArgs cv = ConvArgs()) {
    constexpr int kThreads = (BM / TM) * (BN / TN);
    constexpr int kGroups = TN / 4;           // a thread's columns: kGroups runs of 4
    constexpr int kGroupStride = BN / kGroups;
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
    // kConv: the one column of B this thread stages, found once.
    static_assert(!kConv || kThreads % BN == 0, "a thread stages one fixed column");
    const int myN = tid % BN, myGn = col0 + myN;
    const int area = cv.n * cv.n;
    int mySample = 0, myY = 0, myX = 0;
    if (kConv && myGn < N) {
        mySample = myGn / area;
        const int p = myGn % area;
        myY = p / cv.n;
        myX = p % cv.n;
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Stage A[row0 .. +BM][k0 .. +BK] and B[k0 .. +BK][col0 .. +BN]; zeros
        // outside the matrices.
        for (int e = tid; e < BM * BK; e += kThreads) {
            const int m = e / BK, k = e % BK, gm = row0 + m, gk = k0 + k;
            As[k][m] = (gm < M && gk < K) ? A[static_cast<size_t>(gm) * K + gk] : 0.0f;
        }
        if constexpr (kConv) {
            for (int k = tid / BN; k < BK; k += kThreads / BN) {
                const int gk = k0 + k;
                float v = 0.0f;
                if (myGn < N && gk < K) {
                    const int c = gk / 9, r = gk % 9, y = myY + r / 3 - 1, x = myX + r % 3 - 1;
                    if (y >= 0 && y < cv.n && x >= 0 && x < cv.n)
                        v = cv.in[(static_cast<size_t>(c) * cv.batch + mySample) * area + y * cv.n + x];
                }
                Bs[k][myN] = v;
            }
        } else {
            for (int e = tid; e < BK * BN; e += kThreads) {
                const int k = e / BN, n = e % BN, gn = col0 + n, gk = k0 + k;
                Bs[k][n] = (gn < N && gk < K) ? B[static_cast<size_t>(gk) * N + gn] : 0.0f;
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
                if (gn >= N) continue;
                const size_t at = static_cast<size_t>(gm) * N + gn;
                float v = acc[i][4 * g + q];
                if (kConv) {   // the same operations as bn_relu, in its order
                    v = fmaf(v, cv.scale[gm], cv.shift[gm]);
                    if (cv.skip) v += cv.skip[at];
                    v = fmaxf(v, 0.0f);
                }
                C[at] = v;
            }
    }
}

enum class Gemm { Tile32x256, Tile64x128, Tile128x128, Cublas, Auto, Fused };
const char* gemmName(Gemm g) {
    switch (g) {
        case Gemm::Tile32x256: return "hand 32x256";
        case Gemm::Tile64x128: return "hand 64x128";
        case Gemm::Tile128x128: return "hand 128x128";
        case Gemm::Cublas: return "cuBLAS";
        case Gemm::Auto: return "hand (auto)";
        case Gemm::Fused: return "hand fused";
    }
    return "?";
}

cublasHandle_t gCublas = nullptr;

// The tile shape for a product, by M alone (the weights' rows), never by N:
// one rule for every batch. (Any choice gives the same bits; see `det`.)
Gemm autoTile(int M) {
    if (M >= 128 && M % 128 == 0) return Gemm::Tile128x128;
    if (M % 64 == 0) return Gemm::Tile64x128;
    return Gemm::Tile32x256;
}

void gemm(Gemm g, const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
    if (g == Gemm::Auto || g == Gemm::Fused) g = autoTile(M);
    switch (g) {
        case Gemm::Tile32x256: {
            dim3 grid((N + 255) / 256, (M + 31) / 32);
            gemm_tiled<32, 256, 16, 4, 8><<<grid, 256, 0, s>>>(A, B, C, M, N, K);
            break;
        }
        case Gemm::Tile64x128: {
            dim3 grid((N + 127) / 128, (M + 63) / 64);
            gemm_tiled<64, 128, 16, 8, 4><<<grid, 256, 0, s>>>(A, B, C, M, N, K);
            break;
        }
        case Gemm::Tile128x128: {
            dim3 grid((N + 127) / 128, (M + 127) / 128);
            gemm_tiled<128, 128, 16, 8, 8><<<grid, 256, 0, s>>>(A, B, C, M, N, K);
            break;
        }
        case Gemm::Cublas: {
            // Row-major C = A . B is column-major C^T = B^T . A^T.
            const float one = 1.0f, zero = 0.0f;
            CUBLAS_OK(cublasSetStream(gCublas, s));
            CUBLAS_OK(cublasSgemm(gCublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N));
            break;
        }
        case Gemm::Auto:
        case Gemm::Fused: break;
    }
    CUDA_OK(cudaGetLastError());
}

// A 3x3 convolution with batch norm, skip and ReLU in one kernel: the same
// tile walk as gemm(), with B read through the implicit im2col. Its bits
// equal im2col3 + gemm + bn_relu (forward checks it).
void convFused(const float* W, const ConvArgs& cv, float* out, int M, int C, cudaStream_t s) {
    const int N = cv.batch * cv.n * cv.n, K = C * 9;
    switch (autoTile(M)) {
        case Gemm::Tile128x128: {
            dim3 grid((N + 127) / 128, (M + 127) / 128);
            gemm_tiled<128, 128, 16, 8, 8, true><<<grid, 256, 0, s>>>(W, nullptr, out, M, N, K, cv);
            break;
        }
        case Gemm::Tile64x128: {
            dim3 grid((N + 127) / 128, (M + 63) / 64);
            gemm_tiled<64, 128, 16, 8, 4, true><<<grid, 256, 0, s>>>(W, nullptr, out, M, N, K, cv);
            break;
        }
        default: {
            dim3 grid((N + 255) / 256, (M + 31) / 32);
            gemm_tiled<32, 256, 16, 4, 8, true><<<grid, 256, 0, s>>>(W, nullptr, out, M, N, K, cv);
            break;
        }
    }
    CUDA_OK(cudaGetLastError());
}

// ---------------------------------------------------------------- the other kernels

// [B][P][A] -> [P][B][A]
__global__ void transpose_input(const float* in, float* out, int B, int P, int A) {
    const size_t total = static_cast<size_t>(B) * P * A;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
        const int a = i % A, p = (i / A) % P, b = i / (size_t(A) * P);
        out[(size_t(p) * B + b) * A + a] = in[i];
    }
}

// 3x3, padding 1: in [C][B][n][n] -> col [C*9][B*n*n]; zeros off the board.
__global__ void im2col3(const float* in, float* col, int C, int B, int n) {
    const int area = n * n;
    const size_t N = size_t(B) * area, total = size_t(C) * 9 * N;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
        const size_t j = i % N, r = i / N;
        const int c = r / 9, ky = (r % 9) / 3, kx = r % 3;
        const int b = j / area, p = j % area, y = p / n + ky - 1, x = p % n + kx - 1;
        col[i] = (y >= 0 && y < n && x >= 0 && x < n) ? in[(size_t(c) * B + b) * area + y * n + x] : 0.0f;
    }
}

// x[c][j] = max(0, x * scale[c] + shift[c]) (+ the skip connection first).
__global__ void bn_relu(float* x, const float* scale, const float* shift, const float* skip, int C, size_t N,
                        int relu) {
    const size_t total = size_t(C) * N;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
        const int c = i / N;
        float v = fmaf(x[i], scale[c], shift[c]);
        if (skip) v += skip[i];
        x[i] = relu ? fmaxf(v, 0.0f) : v;
    }
}

// [2][B][A] -> [B][2A]
__global__ void policy_flatten(const float* in, float* out, int B, int A) {
    const size_t total = size_t(2) * B * A;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
        const int a = i % A, b = (i / A) % B, c = i / (size_t(A) * B);
        out[size_t(b) * 2 * A + c * A + a] = in[i];
    }
}

// x[b][o] += bias[o], then ReLU (1), tanh (2) or nothing (0).
__global__ void bias_act(float* x, const float* bias, int B, int O, int act) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < B * O; i += gridDim.x * blockDim.x) {
        float v = x[i] + bias[i % O];
        if (act == 1) v = fmaxf(v, 0.0f);
        if (act == 2) v = tanhf(v);
        x[i] = v;
    }
}

int gridFor(size_t total) { return static_cast<int>(std::min<size_t>((total + 255) / 256, 65535)); }

// ---------------------------------------------------------------- helpers

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }
    float uniform(float lo, float hi) { return lo + (hi - lo) * float(next() >> 40) / float(1 << 24); }
};

std::vector<float> randomVector(size_t n, uint64_t seed, float lo = -1.0f, float hi = 1.0f) {
    Rng r(seed);
    std::vector<float> v(n);
    for (float& x : v) x = r.uniform(lo, hi);
    return v;
}

struct DeviceBuffer {
    float* p = nullptr;
    size_t n = 0;
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) : n(count) { CUDA_OK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(float))); }
    DeviceBuffer(const std::vector<float>& h) : DeviceBuffer(h.size()) { upload(h); }
    ~DeviceBuffer() { if (p) cudaFree(p); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (p) cudaFree(p);
            p = o.p;
            n = o.n;
            o.p = nullptr;
        }
        return *this;
    }
    void upload(const std::vector<float>& h) { CUDA_OK(cudaMemcpy(p, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice)); }
    std::vector<float> download() const {
        std::vector<float> h(n);
        CUDA_OK(cudaMemcpy(h.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
        return h;
    }
};

// Seconds per call of fn, run until gSeconds have passed (after a warm-up).
double timeCalls(const std::function<void()>& fn) {
    cudaEvent_t a, b;
    CUDA_OK(cudaEventCreate(&a));
    CUDA_OK(cudaEventCreate(&b));
    fn();
    CUDA_OK(cudaDeviceSynchronize());
    int reps = 1;
    for (;;) {
        CUDA_OK(cudaEventRecord(a));
        for (int r = 0; r < reps; ++r) fn();
        CUDA_OK(cudaEventRecord(b));
        CUDA_OK(cudaEventSynchronize(b));
        float ms = 0;
        CUDA_OK(cudaEventElapsedTime(&ms, a, b));
        if (ms >= gSeconds * 1000.0 || reps >= (1 << 20)) {
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            return ms / 1000.0 / reps;
        }
        reps = std::max(reps * 2, static_cast<int>(reps * gSeconds * 1000.0 / std::max(ms, 0.01f) * 1.2));
    }
}

bool bitwiseEqual(const float* a, const float* b, size_t n, size_t* differing = nullptr) {
    size_t d = 0;
    for (size_t i = 0; i < n; ++i)
        if (std::memcmp(a + i, b + i, sizeof(float)) != 0) ++d;
    if (differing) *differing = d;
    return d == 0;
}

// ---------------------------------------------------------------- networks

struct Net {
    const char* name;
    int n, blocks, filters, valueHidden;
    int planes() const { return 18; }   // 2 * history + 2, history 8
    int area() const { return n * n; }
};
const Net kNets[] = {{"9x9 6x64", 9, 6, 64, 64},
                     {"9x9 6x96", 9, 6, 96, 64},
                     {"13x13 6x96", 13, 6, 96, 64},
                     {"19x19 10x128", 19, 10, 128, 256}};
const int kBatches[] = {8, 144, 256};
std::string gNetFilter;
bool wanted(const Net& net) { return gNetFilter.empty() || std::string(net.name).find(gNetFilter) != std::string::npos; }

// ---------------------------------------------------------------- gemm section

void sectionGemm() {
    std::printf("== gemm: fp32, C[M][N] = A[M][K] . B[K][N]; ms per call (GFLOP/s); max relative error vs double ==\n");
    std::printf("%-13s %5s %5s %6s %7s", "net", "conv", "batch", "M", "N");
    std::printf("  %5s", "K");
    const Gemm kinds[] = {Gemm::Tile32x256, Gemm::Tile64x128, Gemm::Tile128x128, Gemm::Cublas};
    for (Gemm g : kinds) std::printf("  %22s", gemmName(g));
    std::printf("  %9s %9s\n", "err hand", "err cuBLAS");
    for (const Net& net : kNets) {
        if (!wanted(net)) continue;
        for (int B : kBatches) {
            for (int stem = 1; stem >= 0; --stem) {
                const int M = net.filters, K = (stem ? net.planes() : net.filters) * 9, N = net.area() * B;
                const std::vector<float> hA = randomVector(size_t(M) * K, 1 + M + K, -0.1f, 0.1f);
                const std::vector<float> hB = randomVector(size_t(K) * N, 2 + N, -1.0f, 1.0f);
                DeviceBuffer A(hA), Bm(hB), C(size_t(M) * N);
                std::printf("%-13s %5s %5d %6d %7d  %5d", net.name, stem ? "stem" : "trunk", B, M, N, K);
                double errHand = 0, errCublas = 0;
                for (Gemm g : kinds) {
                    const double s = timeCalls([&] { gemm(g, A.p, Bm.p, C.p, M, N, K, 0); });
                    std::printf("  %8.3f ms (%7.0f)", s * 1e3, 2.0 * M * N * K / s / 1e9);
                    if (g == Gemm::Tile32x256 || g == Gemm::Cublas) {
                        const std::vector<float> hC = C.download();
                        Rng r(99);
                        double worst = 0;
                        for (int t = 0; t < 400; ++t) {
                            const int m = r.next() % M, j = r.next() % N;
                            double ref = 0, mag = 0;
                            for (int k = 0; k < K; ++k) {
                                ref += double(hA[size_t(m) * K + k]) * hB[size_t(k) * N + j];
                                mag += std::fabs(double(hA[size_t(m) * K + k]) * hB[size_t(k) * N + j]);
                            }
                            worst = std::max(worst, std::fabs(hC[size_t(m) * N + j] - ref) / std::max(mag, 1e-30));
                        }
                        (g == Gemm::Cublas ? errCublas : errHand) = worst;
                    }
                }
                std::printf("  %9.2e %9.2e\n", errHand, errCublas);
                std::fflush(stdout);
            }
        }
    }
}

// ---------------------------------------------------------------- det section

// A sample's columns alone (N = area) against the same columns at position
// `at` inside a batch of B random companions.
void sectionDet() {
    std::printf("== det: a sample's output columns alone vs inside a batch (bitwise) ==\n");
    const Gemm kinds[] = {Gemm::Tile32x256, Gemm::Tile64x128, Gemm::Tile128x128, Gemm::Cublas};
    for (const Net& net : kNets) {
        if (!wanted(net)) continue;
        for (int stem = 1; stem >= 0; --stem) {
            const int M = net.filters, K = (stem ? net.planes() : net.filters) * 9, A = net.area();
            const std::vector<float> hA = randomVector(size_t(M) * K, 5 + K, -0.1f, 0.1f);
            DeviceBuffer dA(hA);
            const std::vector<float> sample = randomVector(size_t(K) * A, 77, -1.0f, 1.0f);   // [K][A]
            for (Gemm g : kinds) {
                // Alone.
                DeviceBuffer dB1(sample), dC1(size_t(M) * A);
                gemm(g, dA.p, dB1.p, dC1.p, M, A, K, 0);
                const std::vector<float> alone = dC1.download();
                std::printf("%-13s %5s %-14s", net.name, stem ? "stem" : "trunk", gemmName(g));
                bool all = true;
                for (int B : kBatches) {
                    for (int at : {0, B / 2 + 1, B - 1}) {
                        std::vector<float> hB = randomVector(size_t(K) * A * B, 1000 + B + at, -1.0f, 1.0f);
                        for (int k = 0; k < K; ++k)
                            std::memcpy(&hB[size_t(k) * A * B + size_t(at) * A], &sample[size_t(k) * A], A * sizeof(float));
                        DeviceBuffer dB(hB), dC(size_t(M) * A * B);
                        gemm(g, dA.p, dB.p, dC.p, M, A * B, K, 0);
                        const std::vector<float> hC = dC.download();
                        size_t diff = 0, total = 0;
                        for (int m = 0; m < M; ++m) {
                            size_t d = 0;
                            bitwiseEqual(&hC[size_t(m) * A * B + size_t(at) * A], &alone[size_t(m) * A], A, &d);
                            diff += d;
                            total += A;
                        }
                        if (diff) all = false;
                        std::printf("  B=%d@%d %s", B, at, diff ? (std::to_string(diff) + "/" + std::to_string(total) + " differ").c_str() : "equal");
                    }
                }
                // Repeated run.
                gemm(g, dA.p, dB1.p, dC1.p, M, A, K, 0);
                const std::vector<float> again = dC1.download();
                const bool repeat = bitwiseEqual(again.data(), alone.data(), alone.size());
                std::printf("  repeat %s  => %s\n", repeat ? "equal" : "DIFFERS", all && repeat ? "batch-independent" : "NOT batch-independent");
                std::fflush(stdout);
            }
            // Every tile shape gives the same bits (one fmaf chain per element).
            std::vector<float> ref;
            bool same = true;
            for (Gemm g : {Gemm::Tile32x256, Gemm::Tile64x128, Gemm::Tile128x128}) {
                DeviceBuffer dB1(sample), dC1(size_t(M) * A);
                gemm(g, dA.p, dB1.p, dC1.p, M, A, K, 0);
                const std::vector<float> c = dC1.download();
                if (ref.empty()) ref = c;
                else same = same && bitwiseEqual(ref.data(), c.data(), c.size());
            }
            std::printf("%-13s %5s the three hand tile shapes: %s\n", net.name, stem ? "stem" : "trunk",
                        same ? "bitwise equal" : "DIFFER");
        }
    }
}

// ---------------------------------------------------------------- forward section

// One network's weights and one batch's workspace on the GPU.
struct Forward {
    Net net;
    int maxB;
    Gemm g;
    // weights
    std::vector<DeviceBuffer> convW, bnScale, bnShift;   // stem, then 2 per block
    DeviceBuffer polW, polScale, polShift, polFcW, polFcB;
    DeviceBuffer valW, valScale, valShift, valFc1W, valFc1B, valFc2W, valFc2B;
    // workspace
    DeviceBuffer stage, input, col, t0, t1, t2, pol, polFlat, logits, val, hidden, value;

    Forward(const Net& n, int maxBatch, Gemm gemmKind) : net(n), maxB(maxBatch), g(gemmKind) {
        const int F = net.filters, A = net.area();
        const size_t N = size_t(maxB) * A;
        uint64_t seed = 11;
        auto weights = [&](size_t count, float scale) { return DeviceBuffer(randomVector(count, ++seed, -scale, scale)); };
        auto scales = [&](size_t count) { return DeviceBuffer(randomVector(count, ++seed, 0.5f, 1.5f)); };
        convW.push_back(weights(size_t(F) * net.planes() * 9, 0.1f));
        bnScale.push_back(scales(F));
        bnShift.push_back(weights(F, 0.1f));
        for (int b = 0; b < 2 * net.blocks; ++b) {
            convW.push_back(weights(size_t(F) * F * 9, 0.05f));
            bnScale.push_back(scales(F));
            bnShift.push_back(weights(F, 0.1f));
        }
        polW = weights(2 * F, 0.1f); polScale = scales(2); polShift = weights(2, 0.1f);
        polFcW = weights(size_t(2 * A) * (A + 1), 0.05f); polFcB = weights(A + 1, 0.1f);
        valW = weights(F, 0.1f); valScale = scales(1); valShift = weights(1, 0.1f);
        valFc1W = weights(size_t(A) * net.valueHidden, 0.05f); valFc1B = weights(net.valueHidden, 0.1f);
        valFc2W = weights(net.valueHidden, 0.1f); valFc2B = weights(1, 0.1f);
        stage = DeviceBuffer(size_t(maxB) * net.planes() * A);
        input = DeviceBuffer(size_t(maxB) * net.planes() * A);
        col = DeviceBuffer(size_t(std::max(F, net.planes())) * 9 * N);
        t0 = DeviceBuffer(F * N); t1 = DeviceBuffer(F * N); t2 = DeviceBuffer(F * N);
        pol = DeviceBuffer(2 * N); polFlat = DeviceBuffer(2 * N);
        logits = DeviceBuffer(size_t(maxB) * (A + 1));
        val = DeviceBuffer(N); hidden = DeviceBuffer(size_t(maxB) * net.valueHidden); value = DeviceBuffer(maxB);
    }

    // The pass on the first B positions of `stage`, all on stream s.
    void encode(int B, cudaStream_t s) {
        const int F = net.filters, A = net.area(), P = net.planes();
        const size_t N = size_t(B) * A;
        // A Linear layer's rows are the samples (M = B), so its tile is fixed
        // rather than chosen by M.
        const Gemm fc = g == Gemm::Cublas ? g : Gemm::Tile32x256;
        transpose_input<<<gridFor(N * P), 256, 0, s>>>(stage.p, input.p, B, P, A);
        auto conv = [&](const float* in, int C, int layer, float* out, const float* skip) {
            if (g == Gemm::Fused) {
                ConvArgs cv;
                cv.in = in; cv.batch = B; cv.n = net.n;
                cv.scale = bnScale[layer].p; cv.shift = bnShift[layer].p; cv.skip = skip;
                convFused(convW[layer].p, cv, out, F, C, s);
                return;
            }
            im2col3<<<gridFor(size_t(C) * 9 * N), 256, 0, s>>>(in, col.p, C, B, net.n);
            gemm(g, convW[layer].p, col.p, out, F, int(N), C * 9, s);
            bn_relu<<<gridFor(F * N), 256, 0, s>>>(out, bnScale[layer].p, bnShift[layer].p, skip, F, N, 1);
        };
        conv(input.p, P, 0, t0.p, nullptr);
        float *x = t0.p, *mid = t1.p, *y = t2.p;
        for (int b = 0; b < net.blocks; ++b) {
            conv(x, F, 1 + 2 * b, mid, nullptr);
            conv(mid, F, 2 + 2 * b, y, x);   // bn, + x, ReLU
            std::swap(x, y);
        }
        // Policy: 1x1 conv to 2 planes, BN, ReLU, flatten, Linear to logits.
        gemm(g, polW.p, x, pol.p, 2, int(N), F, s);
        bn_relu<<<gridFor(2 * N), 256, 0, s>>>(pol.p, polScale.p, polShift.p, nullptr, 2, N, 1);
        policy_flatten<<<gridFor(2 * N), 256, 0, s>>>(pol.p, polFlat.p, B, A);
        gemm(fc, polFlat.p, polFcW.p, logits.p, B, A + 1, 2 * A, s);   // rows are samples
        bias_act<<<gridFor(size_t(B) * (A + 1)), 256, 0, s>>>(logits.p, polFcB.p, B, A + 1, 0);
        // Value: 1x1 conv to 1 plane, BN, ReLU, Linear, ReLU, Linear, tanh.
        gemm(g, valW.p, x, val.p, 1, int(N), F, s);
        bn_relu<<<gridFor(N), 256, 0, s>>>(val.p, valScale.p, valShift.p, nullptr, 1, N, 1);
        gemm(fc, val.p, valFc1W.p, hidden.p, B, net.valueHidden, A, s);
        bias_act<<<gridFor(size_t(B) * net.valueHidden), 256, 0, s>>>(hidden.p, valFc1B.p, B, net.valueHidden, 1);
        gemm(fc, hidden.p, valFc2W.p, value.p, B, 1, net.valueHidden, s);
        bias_act<<<gridFor(B), 256, 0, s>>>(value.p, valFc2B.p, B, 1, 2);
        CUDA_OK(cudaGetLastError());
    }
};

// Pinned host staging for one slot.
struct HostSlot {
    float *in = nullptr, *logits = nullptr, *values = nullptr;
    HostSlot(const Net& net, int B) {
        CUDA_OK(cudaMallocHost(&in, size_t(B) * net.planes() * net.area() * sizeof(float)));
        CUDA_OK(cudaMallocHost(&logits, size_t(B) * (net.area() + 1) * sizeof(float)));
        CUDA_OK(cudaMallocHost(&values, size_t(B) * sizeof(float)));
    }
    ~HostSlot() { cudaFreeHost(in); cudaFreeHost(logits); cudaFreeHost(values); }
};

// Copy in, encode, copy out, on stream s.
void pass(Forward& f, HostSlot& h, int B, cudaStream_t s) {
    const size_t in = size_t(B) * f.net.planes() * f.net.area();
    CUDA_OK(cudaMemcpyAsync(f.stage.p, h.in, in * sizeof(float), cudaMemcpyHostToDevice, s));
    f.encode(B, s);
    CUDA_OK(cudaMemcpyAsync(h.logits, f.logits.p, size_t(B) * (f.net.area() + 1) * sizeof(float), cudaMemcpyDeviceToHost, s));
    CUDA_OK(cudaMemcpyAsync(h.values, f.value.p, size_t(B) * sizeof(float), cudaMemcpyDeviceToHost, s));
}

using Clock = std::chrono::steady_clock;

void sectionForward() {
    std::printf("== forward: whole inference pass, fp32; copies in and out included ==\n");
    std::printf("%-13s %-12s %5s  %10s %12s  %12s  %10s  %s\n", "net", "gemm", "batch", "sync ms", "sync ev/s",
                "2-slot ev/s", "gemm share", "alone vs in batch");
    for (const Net& net : kNets) {
        if (!wanted(net)) continue;
        // The explicit hand pass's outputs per batch, for the other variants.
        std::vector<std::vector<float>> handLogits(3), handValues(3);
        for (Gemm g : {Gemm::Auto, Gemm::Fused, Gemm::Cublas}) {
            for (int bi = 0; bi < 3; ++bi) {
                const int B = kBatches[bi];
                Forward f0(net, B, g), f1(net, B, g);
                HostSlot h0(net, B), h1(net, B);
                const std::vector<float> x = randomVector(size_t(B) * net.planes() * net.area(), 3, 0.0f, 1.0f);
                for (size_t i = 0; i < x.size(); ++i) h0.in[i] = h1.in[i] = x[i] < 0.5f ? 0.0f : 1.0f;
                cudaStream_t s0, s1;
                CUDA_OK(cudaStreamCreate(&s0));
                CUDA_OK(cudaStreamCreate(&s1));
                // Synchronous: one pass, then wait (one slot's latency).
                pass(f0, h0, B, s0);
                CUDA_OK(cudaStreamSynchronize(s0));
                int reps = 0;
                auto t = Clock::now();
                double el = 0;
                do {
                    pass(f0, h0, B, s0);
                    CUDA_OK(cudaStreamSynchronize(s0));
                    ++reps;
                    el = std::chrono::duration<double>(Clock::now() - t).count();
                } while (el < gSeconds);
                const double sync = el / reps;
                // Two slots: each stream waits only for its own previous pass.
                reps = 0;
                t = Clock::now();
                do {
                    CUDA_OK(cudaStreamSynchronize(s0));
                    pass(f0, h0, B, s0);
                    CUDA_OK(cudaStreamSynchronize(s1));
                    pass(f1, h1, B, s1);
                    reps += 2;
                    el = std::chrono::duration<double>(Clock::now() - t).count();
                } while (el < gSeconds);
                CUDA_OK(cudaDeviceSynchronize());
                el = std::chrono::duration<double>(Clock::now() - t).count();
                const double twoSlot = reps * double(B) / el;
                // The gemms alone, for their share of the pass.
                const size_t N = size_t(B) * net.area();
                const double gemmS = timeCalls([&] {
                    if (g == Gemm::Fused) {   // the fused convolutions are the gemms
                        ConvArgs cv;
                        cv.batch = B; cv.n = net.n; cv.scale = f0.bnScale[0].p; cv.shift = f0.bnShift[0].p;
                        cv.in = f0.input.p;
                        convFused(f0.convW[0].p, cv, f0.t0.p, net.filters, net.planes(), 0);
                        cv.in = f0.t1.p;
                        for (int l = 1; l <= 2 * net.blocks; ++l)
                            convFused(f0.convW[l].p, cv, f0.t0.p, net.filters, net.filters, 0);
                        return;
                    }
                    gemm(g, f0.convW[0].p, f0.col.p, f0.t0.p, net.filters, int(N), net.planes() * 9, 0);
                    for (int l = 1; l <= 2 * net.blocks; ++l)
                        gemm(g, f0.convW[l].p, f0.col.p, f0.t0.p, net.filters, int(N), net.filters * 9, 0);
                });
                const double passGpu = timeCalls([&] { f0.encode(B, 0); });
                // Batch independence of the whole pass: position `at` alone.
                std::string det;
                {
                    std::vector<float> batchLogits(h0.logits, h0.logits + size_t(B) * (net.area() + 1));
                    std::vector<float> batchValues(h0.values, h0.values + B);
                    Forward fa(net, 1, g);
                    HostSlot ha(net, 1);
                    size_t diffs = 0;
                    for (int at : {0, B / 2, B - 1}) {
                        const size_t in = size_t(net.planes()) * net.area();
                        std::memcpy(ha.in, h0.in + at * in, in * sizeof(float));
                        pass(fa, ha, 1, 0);
                        CUDA_OK(cudaDeviceSynchronize());
                        size_t d = 0;
                        bitwiseEqual(ha.logits, &batchLogits[size_t(at) * (net.area() + 1)], net.area() + 1, &d);
                        diffs += d;
                        bitwiseEqual(ha.values, &batchValues[at], 1, &d);
                        diffs += d;
                    }
                    det = diffs ? std::to_string(diffs) + " outputs differ" : "bitwise equal";
                    // Against the explicit hand pass (im2col + gemm + bn_relu).
                    if (g == Gemm::Auto) {
                        handLogits[bi] = batchLogits;
                        handValues[bi] = batchValues;
                    } else {
                        size_t d1 = 0, d2 = 0;
                        bitwiseEqual(batchLogits.data(), handLogits[bi].data(), batchLogits.size(), &d1);
                        bitwiseEqual(batchValues.data(), handValues[bi].data(), batchValues.size(), &d2);
                        det += d1 + d2 ? "; " + std::to_string(d1 + d2) + "/" +
                                             std::to_string(batchLogits.size() + batchValues.size()) +
                                             " differ from hand"
                                       : "; = hand";
                    }
                }
                std::printf("%-13s %-12s %5d  %10.3f %12.0f  %12.0f  %9.0f%%  %s\n", net.name, gemmName(g), B,
                            sync * 1e3, B / sync, twoSlot, 100.0 * gemmS / passGpu, det.c_str());
                std::fflush(stdout);
                cudaStreamDestroy(s0);
                cudaStreamDestroy(s1);
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> sections;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--seconds" && i + 1 < argc) gSeconds = std::atof(argv[++i]);
        else if (a == "--net" && i + 1 < argc) gNetFilter = argv[++i];
        else if (a == "gemm" || a == "det" || a == "forward") sections.push_back(a);
        else if (a == "all") sections = {"gemm", "det", "forward"};
        else {
            std::fprintf(stderr, "usage: cuda_bench [gemm|det|forward|all] [--seconds S] [--net TEXT]\n");
            return 2;
        }
    }
    if (sections.empty()) sections = {"gemm", "det", "forward"};
    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    int runtime = 0, driver = 0;
    cudaRuntimeGetVersion(&runtime);
    cudaDriverGetVersion(&driver);
    std::printf("gpu: %s, compute %d.%d, %d SMs, %.1f GB; CUDA runtime %d, driver %d\n", prop.name, prop.major,
                prop.minor, prop.multiProcessorCount, prop.totalGlobalMem / 1e9, runtime, driver);
    CUBLAS_OK(cublasCreate(&gCublas));
    // The default math mode is plain fp32: TF32 tensor cores only with
    // CUBLAS_TF32_TENSOR_OP_MATH (or NVIDIA_TF32_OVERRIDE), which is not set.
    CUBLAS_OK(cublasSetMathMode(gCublas, CUBLAS_DEFAULT_MATH));
    for (const std::string& s : sections) {
        if (s == "gemm") sectionGemm();
        if (s == "det") sectionDet();
        if (s == "forward") sectionForward();
    }
    cublasDestroy(gCublas);
    return 0;
}
