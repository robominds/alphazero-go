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
    std::printf("test_backend: all tests passed\n");
    return 0;
}
