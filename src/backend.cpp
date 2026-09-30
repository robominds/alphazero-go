#include "azgo/backend.hpp"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <vector>

namespace azgo {

#ifdef AZ_USE_BLAS
// Defined in src/backend_blas.cpp, which is compiled only with -DAZ_USE_BLAS=ON.
extern std::unique_ptr<Backend> makeBlasBackend();
#endif

// ---------------------------------------------------------------- im2col

void Backend::im2col(const float* in, int C, int B, int n, int k, float* col) const {
    const int r = k / 2;                          // 'same' padding: k = 3 reaches one point out
    const size_t area = static_cast<size_t>(n) * n;
    const size_t M = static_cast<size_t>(B) * area;
    for (int c = 0; c < C; ++c) {
        for (int dy = 0; dy < k; ++dy) {
            for (int dx = 0; dx < k; ++dx) {
                // One row of col holds, for every output pixel, the input value
                // at offset (dy - r, dx - r) from it; off-board reads are 0.
                float* row = col + static_cast<size_t>((c * k + dy) * k + dx) * M;
                const int oy = dy - r, ox = dx - r;
                for (int b = 0; b < B; ++b) {
                    const float* src = in + (static_cast<size_t>(c) * B + b) * area;
                    float* dst = row + static_cast<size_t>(b) * area;
                    for (int y = 0; y < n; ++y) {
                        const int sy = y + oy;
                        float* out = dst + static_cast<size_t>(y) * n;
                        if (sy < 0 || sy >= n) {
                            std::fill(out, out + n, 0.0f);
                            continue;
                        }
                        const float* srcRow = src + static_cast<size_t>(sy) * n;
                        for (int x = 0; x < n; ++x) {
                            const int sx = x + ox;
                            out[x] = (sx >= 0 && sx < n) ? srcRow[sx] : 0.0f;
                        }
                    }
                }
            }
        }
    }
}

void Backend::col2im(const float* col, int C, int B, int n, int k, float* in) const {
    const int r = k / 2;
    const size_t area = static_cast<size_t>(n) * n;
    const size_t M = static_cast<size_t>(B) * area;
    std::fill(in, in + static_cast<size_t>(C) * M, 0.0f);
    // Exactly the loops of im2col with the assignment reversed: every place
    // im2col copied in[p] into col, col2im adds that col entry back into in[p].
    // That makes col2im the transpose (adjoint) of im2col, which is what the
    // backward pass of a convolution needs.
    for (int c = 0; c < C; ++c) {
        for (int dy = 0; dy < k; ++dy) {
            for (int dx = 0; dx < k; ++dx) {
                const float* row = col + static_cast<size_t>((c * k + dy) * k + dx) * M;
                const int oy = dy - r, ox = dx - r;
                for (int b = 0; b < B; ++b) {
                    float* dst = in + (static_cast<size_t>(c) * B + b) * area;
                    const float* src = row + static_cast<size_t>(b) * area;
                    for (int y = 0; y < n; ++y) {
                        const int sy = y + oy;
                        if (sy < 0 || sy >= n) continue;
                        float* dstRow = dst + static_cast<size_t>(sy) * n;
                        const float* srcRow = src + static_cast<size_t>(y) * n;
                        for (int x = 0; x < n; ++x) {
                            const int sx = x + ox;
                            if (sx >= 0 && sx < n) dstRow[sx] += srcRow[x];
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- naive gemm

namespace {

// Blocking keeps a kBlockK x kBlockN tile of B (128 x 512 floats = 256 KB)
// hot in cache while every row of A in this thread's range sweeps over it.
constexpr int kBlockK = 128;
constexpr int kBlockN = 512;

// Below this many multiply-adds, starting threads costs more than it saves.
constexpr size_t kThreadingThreshold = size_t(1) << 18;

// dst (cols x rows) = transpose of src (rows x cols).
void transposeInto(const float* src, int rows, int cols, std::vector<float>& dst) {
    dst.resize(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            dst[static_cast<size_t>(c) * rows + r] = src[static_cast<size_t>(r) * cols + c];
}

// Rows [r0, r1) of C = alpha * A * B + beta * C with A (M x K) and B (K x N)
// both non-transposed. i-k-j order: row i of C accumulates alpha * A[i][p]
// times row p of B, so the innermost loop walks C and B contiguously and the
// compiler vectorizes it.
void gemmRows(int r0, int r1, int N, int K, float alpha, const float* A, const float* B,
              float beta, float* C) {
    for (int i = r0; i < r1; ++i) {
        float* c = C + static_cast<size_t>(i) * N;
        if (beta == 0.0f) {
            std::fill(c, c + N, 0.0f);            // never read C: it may hold NaN
        } else if (beta != 1.0f) {
            for (int j = 0; j < N; ++j) c[j] *= beta;
        }
    }
    for (int j0 = 0; j0 < N; j0 += kBlockN) {
        const int j1 = std::min(N, j0 + kBlockN);
        for (int p0 = 0; p0 < K; p0 += kBlockK) {
            const int p1 = std::min(K, p0 + kBlockK);
            for (int i = r0; i < r1; ++i) {
                float* c = C + static_cast<size_t>(i) * N;
                const float* a = A + static_cast<size_t>(i) * K;
                for (int p = p0; p < p1; ++p) {
                    const float s = alpha * a[p];
                    const float* b = B + static_cast<size_t>(p) * N;
                    for (int j = j0; j < j1; ++j) c[j] += s * b[j];
                }
            }
        }
    }
}

} // namespace

NaiveBackend::NaiveBackend(int threads) : threads_(std::max(1, threads)) {}

void NaiveBackend::gemm(bool transA, bool transB, int M, int N, int K, float alpha,
                        const float* A, const float* B, float beta, float* C) const {
    if (M <= 0 || N <= 0) return;
    // Pack transposed operands once into row-major scratch, so the kernel
    // only ever sees the contiguous layout. thread_local: several worker
    // threads may share one backend, and reusing the vectors avoids an
    // allocation per call.
    thread_local std::vector<float> packA, packB;
    if (transA) {
        transposeInto(A, K, M, packA);            // A is stored K x M
        A = packA.data();
    }
    if (transB) {
        transposeInto(B, N, K, packB);            // B is stored N x K
        B = packB.data();
    }

    const size_t work = static_cast<size_t>(M) * N * K;
    const int workers = (threads_ > 1 && work >= kThreadingThreshold) ? std::min(threads_, M) : 1;
    if (workers == 1) {
        gemmRows(0, M, N, K, alpha, A, B, beta, C);
        return;
    }
    // Split the rows of C into contiguous chunks, one per thread. Each element
    // of C is computed by exactly one thread with exactly the same sequence of
    // floating-point operations (the loop order over p does not depend on the
    // row range), so the result is bitwise identical for any thread count.
    const int chunk = (M + workers - 1) / workers;
    std::vector<std::thread> pool;
    for (int t = 1; t < workers; ++t) {
        const int r0 = t * chunk, r1 = std::min(M, r0 + chunk);
        if (r0 >= r1) break;
        pool.emplace_back(gemmRows, r0, r1, N, K, alpha, A, B, beta, C);
    }
    gemmRows(0, std::min(M, chunk), N, K, alpha, A, B, beta, C);   // chunk 0 on this thread
    for (std::thread& t : pool) t.join();
}

// ---------------------------------------------------------------- factory

bool blasAvailable() {
#ifdef AZ_USE_BLAS
    return true;
#else
    return false;
#endif
}

std::unique_ptr<Backend> makeBackend(const std::string& name, int threads) {
    if (name == "naive") return std::make_unique<NaiveBackend>(threads);
    if (name == "blas") {
#ifdef AZ_USE_BLAS
        return makeBlasBackend();
#else
        throw std::invalid_argument("backend 'blas' not available: rebuild with -DAZ_USE_BLAS=ON");
#endif
    }
    throw std::invalid_argument("unknown backend '" + name + "' (expected naive or blas)");
}

} // namespace azgo
