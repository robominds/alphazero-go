#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/parallel.hpp"
#include "azgo/rng.hpp"

using namespace azgo;

namespace {

struct Shape { int M, N, K; };
const Shape kShapes[] = {{7, 5, 3}, {33, 17, 29}, {1, 1, 1}, {64, 81, 576}};
struct Scale { float alpha, beta; };
const Scale kScales[] = {{1.0f, 0.0f}, {0.5f, 1.0f}, {-2.0f, 0.25f}};

std::vector<float> randomVector(Rng& rng, size_t count) {
    std::vector<float> v(count);
    for (float& x : v) x = static_cast<float>(rng.uniform() * 2.0 - 1.0);
    return v;
}

// Straight from the definition, accumulated in double.
void referenceGemm(bool transA, bool transB, int M, int N, int K, float alpha,
                   const float* A, const float* B, float beta, float* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int p = 0; p < K; ++p) {
                double a = transA ? A[p * M + i] : A[i * K + p];
                double b = transB ? B[j * K + p] : B[p * N + j];
                sum += a * b;
            }
            double old = beta == 0.0f ? 0.0 : static_cast<double>(beta) * C[i * N + j];
            C[i * N + j] = static_cast<float>(alpha * sum + old);
        }
    }
}

bool closeRelative(double actual, double expected, double tol) {
    return std::fabs(actual - expected) <= tol * std::max(1.0, std::fabs(expected));
}

// Runs one gemm on `be` and on the reference with identical inputs (C is
// prefilled with random values, or NaN when beta == 0) and compares.
void checkAgainstReference(const Backend& be, bool tA, bool tB, Shape s, Scale sc, Rng& rng) {
    std::vector<float> A = randomVector(rng, static_cast<size_t>(s.M) * s.K);
    std::vector<float> B = randomVector(rng, static_cast<size_t>(s.K) * s.N);
    std::vector<float> C0 = randomVector(rng, static_cast<size_t>(s.M) * s.N);
    if (sc.beta == 0.0f) std::fill(C0.begin(), C0.end(), std::numeric_limits<float>::quiet_NaN());
    std::vector<float> C = C0, expected = C0;
    be.gemm(tA, tB, s.M, s.N, s.K, sc.alpha, A.data(), B.data(), sc.beta, C.data());
    referenceGemm(tA, tB, s.M, s.N, s.K, sc.alpha, A.data(), B.data(), sc.beta, expected.data());
    for (size_t i = 0; i < C.size(); ++i) {
        assert(std::isfinite(C[i]));
        if (!closeRelative(C[i], expected[i], 1e-4)) {
            std::printf("%s gemm mismatch tA=%d tB=%d M=%d N=%d K=%d at %zu: %g vs %g\n",
                        be.name(), tA, tB, s.M, s.N, s.K, i, C[i], expected[i]);
            assert(false);
        }
    }
}

void checkAllCombinations(const Backend& be) {
    Rng rng(11);
    for (const Shape& s : kShapes)
        for (const Scale& sc : kScales)
            for (int t = 0; t < 4; ++t)
                checkAgainstReference(be, (t & 1) != 0, (t & 2) != 0, s, sc, rng);
}

} // namespace

static void test_naive_gemm_matches_reference() {
    NaiveBackend be;
    checkAllCombinations(be);
}

static void test_beta_zero_ignores_nan_in_c() {
    NaiveBackend be;
    const float A[] = {1, 2, 3, 4, 5, 6};          // 2 x 3
    const float B[] = {1, 0, 0, 1, 1, 1};          // 3 x 2
    float C[4];
    std::fill(C, C + 4, std::numeric_limits<float>::quiet_NaN());
    be.gemm(false, false, 2, 2, 3, 1.0f, A, B, 0.0f, C);
    // Row 0: [1*1 + 2*0 + 3*1, 1*0 + 2*1 + 3*1] = [4, 5]; row 1: [10, 11].
    assert(C[0] == 4.0f && C[1] == 5.0f && C[2] == 10.0f && C[3] == 11.0f);
}

static void test_threads_are_bitwise_identical() {
    // 64 x 300 x 576 is far above the threading threshold (2^18 multiply-adds).
    const int M = 64, N = 300, K = 576;
    Rng rng(5);
    std::vector<float> A = randomVector(rng, static_cast<size_t>(M) * K);
    std::vector<float> B = randomVector(rng, static_cast<size_t>(K) * N);
    std::vector<float> C0 = randomVector(rng, static_cast<size_t>(M) * N);
    NaiveBackend one(1), three(3), four(4);
    for (int t = 0; t < 4; ++t) {
        const bool tA = (t & 1) != 0, tB = (t & 2) != 0;
        std::vector<float> c1 = C0, c3 = C0, c4 = C0;
        one.gemm(tA, tB, M, N, K, 0.75f, A.data(), B.data(), 0.5f, c1.data());
        three.gemm(tA, tB, M, N, K, 0.75f, A.data(), B.data(), 0.5f, c3.data());
        four.gemm(tA, tB, M, N, K, 0.75f, A.data(), B.data(), 0.5f, c4.data());
        assert(std::memcmp(c1.data(), c3.data(), c1.size() * sizeof(float)) == 0);
        assert(std::memcmp(c1.data(), c4.data(), c1.size() * sizeof(float)) == 0);
    }
}

static void test_im2col_hand_example() {
    // One 3x3 image 1..9, 3x3 patches. col is 9 rows (patch offsets) x 9
    // columns (output pixels); column q holds the 3x3 neighborhood of pixel q.
    NaiveBackend be;
    const float in[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float col[81];
    be.im2col(in, 1, 1, 3, 3, col);
    for (int r = 0; r < 9; ++r) {
        assert(col[r * 9 + 4] == in[r]);          // center pixel: its neighborhood is the whole image
        assert(col[4 * 9 + r] == in[r]);          // center offset row: the image itself
    }
    const float topLeft[9] = {0, 0, 0, 0, 1, 2, 0, 4, 5};      // pixel (0,0), zero padded
    const float bottomRight[9] = {5, 6, 0, 8, 9, 0, 0, 0, 0};  // pixel (2,2)
    for (int r = 0; r < 9; ++r) {
        assert(col[r * 9 + 0] == topLeft[r]);
        assert(col[r * 9 + 8] == bottomRight[r]);
    }
}

static void test_im2col_k1_is_copy() {
    NaiveBackend be;
    Rng rng(3);
    const int C = 2, B = 3, n = 5;
    std::vector<float> in = randomVector(rng, static_cast<size_t>(C) * B * n * n);
    std::vector<float> col(in.size(), -1.0f);
    be.im2col(in.data(), C, B, n, 1, col.data());
    assert(col == in);
}

static void test_col2im_is_adjoint_of_im2col() {
    NaiveBackend be;
    Rng rng(4);
    const int C = 2, B = 3, n = 5, k = 3;
    const size_t inSize = static_cast<size_t>(C) * B * n * n;
    const size_t colSize = inSize * k * k;
    std::vector<float> x = randomVector(rng, inSize), y = randomVector(rng, colSize);
    std::vector<float> colX(colSize), backY(inSize);
    be.im2col(x.data(), C, B, n, k, colX.data());
    be.col2im(y.data(), C, B, n, k, backY.data());
    // <im2col(x), y> == <x, col2im(y)> is the definition of "adjoint".
    double lhs = 0.0, rhs = 0.0;
    for (size_t i = 0; i < colSize; ++i) lhs += static_cast<double>(colX[i]) * y[i];
    for (size_t i = 0; i < inSize; ++i) rhs += static_cast<double>(x[i]) * backY[i];
    assert(std::fabs(lhs - rhs) <= 1e-3);
}

static void test_col2im_overwrites_and_counts_overlaps() {
    // All-ones col: each pixel receives one contribution per patch that
    // covers it: 4 at corners, 6 on edges, 9 in the middle. The NaN prefill
    // proves col2im overwrites rather than accumulates into its output.
    NaiveBackend be;
    std::vector<float> col(81, 1.0f);
    float in[9];
    std::fill(in, in + 9, std::numeric_limits<float>::quiet_NaN());
    be.col2im(col.data(), 1, 1, 3, 3, in);
    const float expected[9] = {4, 6, 4, 6, 9, 6, 4, 6, 4};
    for (int p = 0; p < 9; ++p) assert(in[p] == expected[p]);
}

static void test_make_backend() {
    std::unique_ptr<Backend> naive = makeBackend("naive", 2);
    assert(std::string(naive->name()) == "naive");
    bool threw = false;
    try {
        makeBackend("bogus");
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()).find("bogus") != std::string::npos;
    }
    assert(threw);
    if (!blasAvailable()) {
        threw = false;
        try {
            makeBackend("blas");
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
}

static void test_blas_matches_naive() {
    if (!blasAvailable()) {
        std::printf("blas not built: skipping\n");
        return;
    }
    std::unique_ptr<Backend> blas = makeBackend("blas");
    assert(std::string(blas->name()) == "blas");
    checkAllCombinations(*blas);   // same reference, so blas and naive agree within 1e-4

    // Direct naive-vs-blas comparison on the largest shape, all four combinations.
    NaiveBackend naive;
    Rng rng(9);
    const int M = 64, N = 81, K = 576;
    std::vector<float> A = randomVector(rng, static_cast<size_t>(M) * K);
    std::vector<float> B = randomVector(rng, static_cast<size_t>(K) * N);
    for (int t = 0; t < 4; ++t) {
        std::vector<float> c1(static_cast<size_t>(M) * N), c2(c1.size());
        naive.gemm(t & 1, t & 2, M, N, K, 1.0f, A.data(), B.data(), 0.0f, c1.data());
        blas->gemm(t & 1, t & 2, M, N, K, 1.0f, A.data(), B.data(), 0.0f, c2.data());
        for (size_t i = 0; i < c1.size(); ++i) assert(closeRelative(c2[i], c1[i], 1e-4));
    }
}

// Leading dimensions: the gemm of a sub-matrix stored inside larger arrays
// equals the packed gemm of a copy of it, bit for bit (same backend).
static void checkStridedMatchesPacked(const Backend& be) {
    Rng rng(21);
    const int M = 13, N = 11, K = 9, pad = 5;
    for (int t = 0; t < 4; ++t) {
        const bool tA = (t & 1) != 0, tB = (t & 2) != 0;
        const int aRows = tA ? K : M, aCols = tA ? M : K, bRows = tB ? N : K, bCols = tB ? K : N;
        const int lda = aCols + pad, ldb = bCols + pad, ldc = N + pad;
        std::vector<float> A = randomVector(rng, static_cast<size_t>(aRows) * lda);
        std::vector<float> B = randomVector(rng, static_cast<size_t>(bRows) * ldb);
        std::vector<float> C = randomVector(rng, static_cast<size_t>(M) * ldc);
        std::vector<float> Ap, Bp, Cp;
        for (int r = 0; r < aRows; ++r) Ap.insert(Ap.end(), A.begin() + r * lda, A.begin() + r * lda + aCols);
        for (int r = 0; r < bRows; ++r) Bp.insert(Bp.end(), B.begin() + r * ldb, B.begin() + r * ldb + bCols);
        for (int r = 0; r < M; ++r) Cp.insert(Cp.end(), C.begin() + r * ldc, C.begin() + r * ldc + N);
        const std::vector<float> C0 = C;
        be.gemm(tA, tB, M, N, K, 0.5f, A.data(), lda, B.data(), ldb, 0.75f, C.data(), ldc);
        be.gemm(tA, tB, M, N, K, 0.5f, Ap.data(), Bp.data(), 0.75f, Cp.data());
        for (int r = 0; r < M; ++r) {
            assert(std::memcmp(C.data() + r * ldc, Cp.data() + r * N, N * sizeof(float)) == 0);
            // The padding between rows is never written.
            for (int j = N; j < ldc; ++j) assert(C[static_cast<size_t>(r) * ldc + j] == C0[static_cast<size_t>(r) * ldc + j]);
        }
    }
}

static void test_strided_gemm() {
    NaiveBackend naive;
    checkStridedMatchesPacked(naive);
    if (blasAvailable()) checkStridedMatchesPacked(*makeBackend("blas"));
}

// gemmColumns with any pool (1, 2, 3 or 7 threads) gives the bits of one
// plain gemm call. The shapes are the split gemms of a convolution (the
// forward product, the stem's, and the input gradient of 3x3 and 1x1
// convs) at several board sizes, widths and batches, including the
// presets' (5x5 3x32 B=128, 9x9 6x64 and 6x96 B=256, 19x19 10x128) and odd
// tails like a validation pass's last batch. For the naive backend this
// holds by construction; for Accelerate it is a measurement (an ~11,000-
// shape sweep in development found no exception for blocks of 128 to 1024
// columns with at least 8 rows). Also: how a shape is cut.
static void checkColumnBlocksMatchUnsplit(const Backend& be, bool large) {
    struct Conv { int n, F, B; };
    const Conv convs[] = {{5, 8, 16},  {5, 32, 128}, {5, 32, 7},  {7, 48, 33},  {9, 64, 256},
                          {9, 96, 256}, {9, 96, 100}, {9, 64, 3},  {9, 100, 64}, {13, 80, 16},
                          {19, 128, 16}, {19, 128, 1}};
    const int planes = 18;
    ThreadPool p1(1), p2(2), p3(3), p7(7);
    ThreadPool* pools[] = {&p1, &p2, &p3, &p7};
    Rng rng(31);
    for (const Conv& c : convs) {
        // The naive backend gets the small shapes only: there the equality
        // follows from the loop order, and the big ones are slow.
        if (!large && c.B * c.n * c.n * c.F > 64 * 1024) continue;
        const int Mpix = c.B * c.n * c.n;
        struct G { bool tA, tB; int M, N, K; float beta; };
        const G gemms[] = {
            {false, false, c.F, Mpix, 9 * c.F, 0.0f},     // forward
            {false, false, c.F, Mpix, 9 * planes, 0.0f},  // stem forward
            {false, false, 2, Mpix, c.F, 0.0f},           // policy head forward: one block
            {true, false, 9 * c.F, Mpix, c.F, 0.0f},      // dX into the columns
            {true, false, c.F, Mpix, 2, 0.0f},            // 1x1 policy conv dX
            {true, false, c.F, Mpix, 1, 0.0f},            // 1x1 value conv dX
            {true, false, 9 * c.F, Mpix, c.F, 0.5f},      // beta != 0 too
        };
        for (const G& g : gemms) {
            std::vector<float> A = randomVector(rng, static_cast<size_t>(g.M) * g.K);
            std::vector<float> B = randomVector(rng, static_cast<size_t>(g.K) * g.N);
            std::vector<float> C0 = randomVector(rng, static_cast<size_t>(g.M) * g.N);
            std::vector<float> expected = C0;
            be.gemm(g.tA, g.tB, g.M, g.N, g.K, 1.0f, A.data(), B.data(), g.beta, expected.data());
            for (ThreadPool* pool : pools) {
                std::vector<float> C = C0;
                gemmColumns(be, pool, g.tA, g.tB, g.M, g.N, g.K, 1.0f, A.data(), B.data(), g.beta, C.data());
                if (std::memcmp(C.data(), expected.data(), C.size() * sizeof(float)) != 0) {
                    std::printf("%s column blocks differ: n=%d F=%d B=%d M=%d N=%d K=%d threads=%d\n", be.name(),
                                c.n, c.F, c.B, g.M, g.N, g.K, pool->threads());
                    assert(false);
                }
            }
        }
    }
}

static void test_column_blocks_match_unsplit_gemm() {
    // The cut: N / 512 blocks, the remainder in the last; small or flat products stay whole.
    assert(gemmColumnBlockCount(64, 20736) == 40);
    assert(gemmColumnBlockCount(64, 1023) == 1 && gemmColumnBlockCount(64, 1024) == 2);
    assert(gemmColumnBlockCount(2, 20736) == 1 && gemmColumnBlockCount(8, 20736) == 40);
    NaiveBackend naive;
    checkColumnBlocksMatchUnsplit(naive, false);
    if (blasAvailable()) checkColumnBlocksMatchUnsplit(*makeBackend("blas"), true);
}

// Row bands on a pool, used for the weight gradient on a backend that has
// them (the naive one; BLAS declines): a dW-shaped gemm (beta = 1) on pools
// of 1, 2 and 5 threads gives the unsplit call's bits.
static void test_row_bands_match_unsplit_gemm() {
    NaiveBackend naive;
    ThreadPool p1(1), p2(2), p5(5);
    if (blasAvailable()) {
        float c = 7.0f;
        const float one = 1.0f;
        assert(!makeBackend("blas")->gemmRowsOnPool(p2, false, false, 1, 1, 1, 1.0f, &one, &one, 0.0f, &c));
        assert(c == 7.0f);   // declined: nothing computed
    }
    Rng rng(51);
    for (const Shape& s : {Shape{100, 288, 1200}, Shape{64, 162, 2025}, Shape{33, 27, 50}, Shape{8, 9, 10}}) {
        for (int t = 0; t < 4; ++t) {
            const bool tA = (t & 1) != 0, tB = (t & 2) != 0;
            std::vector<float> A = randomVector(rng, static_cast<size_t>(s.M) * s.K);
            std::vector<float> B = randomVector(rng, static_cast<size_t>(s.K) * s.N);
            std::vector<float> C0 = randomVector(rng, static_cast<size_t>(s.M) * s.N), expected = C0;
            naive.gemm(tA, tB, s.M, s.N, s.K, 1.0f, A.data(), B.data(), 1.0f, expected.data());
            for (ThreadPool* pool : {&p1, &p2, &p5}) {
                std::vector<float> C = C0;
                assert(naive.gemmRowsOnPool(*pool, tA, tB, s.M, s.N, s.K, 1.0f, A.data(), B.data(), 1.0f, C.data()));
                assert(std::memcmp(C.data(), expected.data(), C.size() * sizeof(float)) == 0);
            }
        }
    }
}

// The definitions of im2col and col2im as first written (plain loops). The
// backend's versions must give exactly these bits: im2col is a copy, and
// col2im adds each pixel's contributions in the same (dy, dx) order.
static void referenceIm2col(const float* in, int C, int B, int n, int k, float* col) {
    const int r = k / 2;
    const size_t area = static_cast<size_t>(n) * n, M = static_cast<size_t>(B) * area;
    for (int c = 0; c < C; ++c)
        for (int dy = 0; dy < k; ++dy)
            for (int dx = 0; dx < k; ++dx)
                for (int b = 0; b < B; ++b)
                    for (int y = 0; y < n; ++y)
                        for (int x = 0; x < n; ++x) {
                            const int sy = y + dy - r, sx = x + dx - r;
                            col[static_cast<size_t>((c * k + dy) * k + dx) * M + b * area + y * n + x] =
                                (sy >= 0 && sy < n && sx >= 0 && sx < n)
                                    ? in[(static_cast<size_t>(c) * B + b) * area + sy * n + sx]
                                    : 0.0f;
                        }
}

static void referenceCol2im(const float* col, int C, int B, int n, int k, float* in) {
    const int r = k / 2;
    const size_t area = static_cast<size_t>(n) * n, M = static_cast<size_t>(B) * area;
    std::fill(in, in + static_cast<size_t>(C) * M, 0.0f);
    for (int c = 0; c < C; ++c)
        for (int dy = 0; dy < k; ++dy)
            for (int dx = 0; dx < k; ++dx)
                for (int b = 0; b < B; ++b)
                    for (int y = 0; y < n; ++y)
                        for (int x = 0; x < n; ++x) {
                            const int sy = y + dy - r, sx = x + dx - r;
                            if (sy >= 0 && sy < n && sx >= 0 && sx < n)
                                in[(static_cast<size_t>(c) * B + b) * area + sy * n + sx] +=
                                    col[static_cast<size_t>((c * k + dy) * k + dx) * M + b * area + y * n + x];
                        }
}

static void test_im2col_col2im_match_definition_bitwise() {
    NaiveBackend be;
    Rng rng(41);
    for (int n : {1, 2, 3, 5, 9, 19})
        for (int k : {1, 3})
            for (int B : {1, 3}) {
                const int C = 4;
                const size_t inSize = static_cast<size_t>(C) * B * n * n, colSize = inSize * k * k;
                std::vector<float> x = randomVector(rng, inSize), y = randomVector(rng, colSize);
                std::vector<float> col(colSize, -7.0f), colRef(colSize, -9.0f);
                be.im2col(x.data(), C, B, n, k, col.data());
                referenceIm2col(x.data(), C, B, n, k, colRef.data());
                assert(std::memcmp(col.data(), colRef.data(), colSize * sizeof(float)) == 0);
                std::vector<float> back(inSize, std::numeric_limits<float>::quiet_NaN()), backRef(inSize);
                be.col2im(y.data(), C, B, n, k, back.data());
                referenceCol2im(y.data(), C, B, n, k, backRef.data());
                assert(std::memcmp(back.data(), backRef.data(), inSize * sizeof(float)) == 0);
            }
}

int main() {
    test_naive_gemm_matches_reference();
    test_beta_zero_ignores_nan_in_c();
    test_threads_are_bitwise_identical();
    test_im2col_hand_example();
    test_im2col_k1_is_copy();
    test_col2im_is_adjoint_of_im2col();
    test_col2im_overwrites_and_counts_overlaps();
    test_make_backend();
    test_blas_matches_naive();
    test_strided_gemm();
    test_column_blocks_match_unsplit_gemm();
    test_row_bands_match_unsplit_gemm();
    test_im2col_col2im_match_definition_bitwise();
    std::printf("test_backend: all tests passed\n");
    return 0;
}
