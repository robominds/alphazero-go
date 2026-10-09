// GPU tests: the Metal context, every kernel against its CPU counterpart and
// the batch-independence properties the batching evaluator relies on.
//
// They run only in a build configured with -DAZ_USE_METAL=ON on a Mac with a
// usable GPU. Anywhere else main() returns 77, which CMake declares as this
// test's SKIP_RETURN_CODE, so CTest reports "Skipped" rather than a failure.
//
// Tolerances (see the spec's "Tolerance against the CPU"): maxErr is
// max|x - ref| / max|ref|; "bitwise" is memcmp. Each test prints its measured
// worst error once, next to its ceiling.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/metal.hpp"
#include "azgo/rng.hpp"

using namespace azgo;

#ifdef AZ_USE_METAL

namespace {

// ---------------------------------------------------------------- helpers

// How many seeds each tolerance test draws its inputs from: 1 by default;
// AZGO_METAL_SEEDS=20 re-measures the worst errors recorded in the comments.
int gSeeds = 1;

std::vector<float> randomVector(Rng& rng, size_t count, double scale = 1.0) {
    std::vector<float> v(count);
    for (float& x : v) x = static_cast<float>((rng.uniform() * 2.0 - 1.0) * scale);
    return v;
}

bool bitwiseEqual(const float* a, const float* b, size_t count) {
    return count == 0 || std::memcmp(a, b, count * sizeof(float)) == 0;
}

// The worst error a test saw against its ceiling: max|x - ref| / max|ref|,
// gathered over every comparison and printed once at the end.
struct MaxErr {
    const char* what;
    double ceiling;
    double worst = 0.0;
    MaxErr(const char* w, double c) : what(w), ceiling(c) {}
    double add(const float* x, const double* ref, size_t count) {
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < count; ++i) {
            assert(std::isfinite(x[i]));
            num = std::max(num, std::fabs(x[i] - ref[i]));
            den = std::max(den, std::fabs(ref[i]));
        }
        const double e = den > 0.0 ? num / den : num;
        if (!(e <= ceiling)) {
            std::printf("test_metal: %s: maxErr %.3g above the ceiling %.0e\n", what, e, ceiling);
            assert(false);
        }
        worst = std::max(worst, e);
        return e;
    }
    double add(const float* x, const float* ref, size_t count) {
        std::vector<double> r(ref, ref + count);
        return add(x, r.data(), count);
    }
    ~MaxErr() {
        std::printf("test_metal: %-34s maxErr %.2e (ceiling %.0e)\n", what, worst, ceiling);
    }
};

// ---------------------------------------------------------------- context

void test_context_comes_up() {
    const std::string d = metalDescription();
    assert(!d.empty());
    assert(d.find(metal_probe::deviceName()) != std::string::npos);
    assert(d.find("macOS ") != std::string::npos);
    assert(d.find("shaders ") != std::string::npos);
    for (int i = 0; i < 3; ++i) assert(metalDescription() == d);   // stable across calls
    std::printf("test_metal: %s\n", d.c_str());
}

// fill writes exactly n floats: a sentinel tail after them comes back untouched.
void test_fill_kernel() {
    const float kSentinel = -12345.0f;
    const size_t kTail = 37;
    for (size_t n : {size_t(0), size_t(1), size_t(255), size_t(256), size_t(257), size_t(100000)}) {
        std::vector<float> data(n + kTail, kSentinel);
        metal_probe::fill(data.data(), n, 2.5f, kTail);
        for (size_t i = 0; i < n; ++i) assert(data[i] == 2.5f);
        for (size_t i = n; i < n + kTail; ++i) assert(data[i] == kSentinel);
    }
}

void test_fill_repeat_bitwise() {
    const size_t n = 4097;
    std::vector<float> first(n);
    metal_probe::fill(first.data(), n, 0.1f);
    for (int r = 0; r < 20; ++r) {
        std::vector<float> again(n);
        metal_probe::fill(again.data(), n, 0.1f);
        assert(std::memcmp(first.data(), again.data(), n * sizeof(float)) == 0);
    }
}


// ---------------------------------------------------------------- gemm

struct GemmShape { int M, N, K; };

// One element of alpha * op(A) * op(B) + beta * C0, accumulated in double.
double referenceElement(bool tA, bool tB, int M, int N, int K, float alpha, const float* A,
                        const float* B, float beta, const float* C0, int i, int j) {
    double sum = 0.0;
    for (int k = 0; k < K; ++k) {
        const double a = tA ? A[static_cast<size_t>(k) * M + i] : A[static_cast<size_t>(i) * K + k];
        const double b = tB ? B[static_cast<size_t>(j) * K + k] : B[static_cast<size_t>(k) * N + j];
        sum += a * b;
    }
    const double old = beta == 0.0f ? 0.0 : static_cast<double>(beta) * C0[static_cast<size_t>(i) * N + j];
    return alpha * sum + old;
}

// Every trans combination, alpha 1 and 0.5, beta 0 and 1, against a double
// reference: on every element of the small products and on 3,000 random
// elements of the large ones (a full double reference of a 96 x 20736 x 864
// product would take seconds per combination).
// Measured on an M5 Max (macOS 27.0.1), 20 seeds: worst maxErr 1.5e-6
// (ceiling 1e-5; the spike measured 1.3e-6 on the forward shapes).
void test_gemm_against_double_reference() {
    const GemmShape shapes[] = {
        {1, 1, 1}, {7, 63, 65}, {64, 64, 64}, {65, 129, 17},
        {96, 81 * 256, 864},                      // 9x9 6x96 block conv forward, batch 256
        {96, 81 * 256, 162},                      // the stem (18 planes x 9)
        {2, 81 * 256, 96}, {1, 81 * 256, 96},     // the policy and value 1x1 convs
        {256, 82, 162},                           // policy fc
        {256, 64, 81}, {256, 1, 64},              // value fc1, fc2
    };
    MaxErr err("gemm vs double", 1e-5);
    for (int seed = 0; seed < gSeeds; ++seed) {
        Rng rng(201 + 1000 * seed);
        for (const GemmShape& s : shapes) {
            const size_t mn = static_cast<size_t>(s.M) * s.N;
            const std::vector<float> A = randomVector(rng, static_cast<size_t>(s.M) * s.K);
            const std::vector<float> B = randomVector(rng, static_cast<size_t>(s.K) * s.N);
            const std::vector<float> C0 = randomVector(rng, mn);
            std::vector<size_t> sample;
            if (mn <= 20000) {
                for (size_t e = 0; e < mn; ++e) sample.push_back(e);
            } else {
                for (int e = 0; e < 3000; ++e)
                    sample.push_back(static_cast<size_t>(rng.below(static_cast<int>(mn))));
            }
            for (int t = 0; t < 4; ++t) {
                const bool tA = (t & 1) != 0, tB = (t & 2) != 0;
                for (float alpha : {1.0f, 0.5f}) {
                    for (float beta : {0.0f, 1.0f}) {
                        std::vector<float> C = C0;
                        metal_probe::gemm(tA, tB, s.M, s.N, s.K, alpha, A.data(), B.data(), beta,
                                          C.data());
                        std::vector<float> got;
                        std::vector<double> ref;
                        for (size_t e : sample) {
                            const int i = static_cast<int>(e / s.N), j = static_cast<int>(e % s.N);
                            got.push_back(C[e]);
                            ref.push_back(referenceElement(tA, tB, s.M, s.N, s.K, alpha, A.data(),
                                                           B.data(), beta, C0.data(), i, j));
                        }
                        err.add(got.data(), ref.data(), got.size());
                    }
                }
            }
        }
    }
}

// beta = 0 means C is overwritten, never read: NaN in it must not leak through.
void test_gemm_beta_zero_ignores_nan() {
    Rng rng(202);
    for (const GemmShape s : {GemmShape{5, 7, 3}, GemmShape{70, 130, 33}}) {
        const std::vector<float> A = randomVector(rng, static_cast<size_t>(s.M) * s.K);
        const std::vector<float> B = randomVector(rng, static_cast<size_t>(s.K) * s.N);
        const size_t mn = static_cast<size_t>(s.M) * s.N;
        std::vector<float> zeros(mn, 0.0f), nans(mn, std::numeric_limits<float>::quiet_NaN());
        for (int t = 0; t < 4; ++t) {
            std::vector<float> a = zeros, b = nans;
            metal_probe::gemm(t & 1, t & 2, s.M, s.N, s.K, 0.5f, A.data(), B.data(), 0.0f, a.data());
            metal_probe::gemm(t & 1, t & 2, s.M, s.N, s.K, 0.5f, A.data(), B.data(), 0.0f, b.data());
            for (float v : b) assert(std::isfinite(v));
            assert(bitwiseEqual(a.data(), b.data(), mn));
        }
    }
}

// The forward convolution products of the three presets (stem, block, the two
// 1x1 head convs) at batches 1, 8 and 256, against the BLAS backend
// (Accelerate). The spike found these bitwise equal on its shapes; the
// fraction printed here is for information only.
// Measured on an M5 Max (macOS 27.0.1), 20 seeds: worst maxErr 6.7e-7
// (ceiling 2e-6); 99.995% of the elements bitwise equal to Accelerate. Every
// stem and block product was bitwise equal; the differences were all in the
// 1x1 head convolutions with one or two rows, where Accelerate sums in
// another order (as a matrix-vector product, presumably).
void test_gemm_vs_blas_forward_shapes() {
    struct Board { int n, F; };
    const Board boards[] = {{5, 32}, {9, 96}, {19, 128}};
    const int planes = 18;
    auto blas = makeBackend("blas");
    MaxErr err("gemm vs BLAS (conv forward)", 2e-6);
    size_t equal = 0, total = 0;
    for (int seed = 0; seed < gSeeds; ++seed) {
        Rng rng(203 + 1000 * seed);
        // One pool of random values large enough for the largest column matrix
        // (19x19, 128 filters, batch 256: 1152 x 92416), so it is drawn only once.
        const std::vector<float> pool = randomVector(rng, static_cast<size_t>(128) * 9 * 361 * 256);
        for (const Board& bd : boards) {
            for (int batch : {1, 8, 256}) {
                const int N = batch * bd.n * bd.n;
                const GemmShape shapes[] = {
                    {bd.F, N, planes * 9}, {bd.F, N, bd.F * 9}, {2, N, bd.F}, {1, N, bd.F}};
                for (const GemmShape& s : shapes) {
                    const std::vector<float> W = randomVector(rng, static_cast<size_t>(s.M) * s.K, 0.05);
                    const float* cols = pool.data();
                    const size_t mn = static_cast<size_t>(s.M) * s.N;
                    std::vector<float> gpu(mn), cpu(mn);
                    metal_probe::gemm(false, false, s.M, s.N, s.K, 1.0f, W.data(), cols, 0.0f,
                                      gpu.data());
                    blas->gemm(false, false, s.M, s.N, s.K, 1.0f, W.data(), cols, 0.0f, cpu.data());
                    err.add(gpu.data(), cpu.data(), mn);
                    for (size_t e = 0; e < mn; ++e)
                        equal += std::memcmp(&gpu[e], &cpu[e], sizeof(float)) == 0;
                    total += mn;
                }
            }
        }
    }
    std::printf("test_metal: gemm vs BLAS: %.4f%% of %zu elements bitwise equal (information only)\n",
                100.0 * static_cast<double>(equal) / static_cast<double>(total), total);
}

void test_gemm_repeat_bitwise() {
    Rng rng(204);
    const int M = 96, N = 81 * 64, K = 864;
    const std::vector<float> A = randomVector(rng, static_cast<size_t>(M) * K);
    const std::vector<float> B = randomVector(rng, static_cast<size_t>(K) * N);
    std::vector<float> first(static_cast<size_t>(M) * N);
    metal_probe::gemm(false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, first.data());
    for (int r = 0; r < 20; ++r) {
        std::vector<float> again(first.size());
        metal_probe::gemm(false, false, M, N, K, 1.0f, A.data(), B.data(), 0.0f, again.data());
        assert(bitwiseEqual(first.data(), again.data(), first.size()));
    }
}

// The property the batching evaluator rests on. A convolution's product reads
// a sample's outputs from that sample's columns of op(B) only, and a Linear
// layer's from its row of op(A) only. So a sample's outputs must be the same
// bits whether it is computed alone, in a batch of 8 at the first or a later
// slot after random companions, or in a batch of 256: whatever tile, and
// whatever place within a tile, its columns or rows land in.
void test_gemm_columns_and_rows_batch_independent() {
    Rng rng(205);
    // Columns: the 9x9 6x96 block convolution, K = 9 * 96.
    {
        const int F = 96, K = 9 * F, A = 81, full = 256;
        const std::vector<float> W = randomVector(rng, static_cast<size_t>(F) * K, 0.05);
        // col [K][full * A]: sample b's columns are b*A ... b*A + A - 1.
        const std::vector<float> cols = randomVector(rng, static_cast<size_t>(K) * full * A);
        // The columns of `samples` gathered, in that order, into [K][samples * A].
        auto gather = [&](const std::vector<int>& samples) {
            const size_t N = samples.size() * A;
            std::vector<float> out(static_cast<size_t>(K) * N);
            for (int k = 0; k < K; ++k)
                for (size_t s = 0; s < samples.size(); ++s)
                    std::copy_n(cols.data() + (static_cast<size_t>(k) * full + samples[s]) * A, A,
                                out.data() + static_cast<size_t>(k) * N + s * A);
            return out;
        };
        auto run = [&](const std::vector<int>& samples) {
            const std::vector<float> b = gather(samples);
            const int N = static_cast<int>(samples.size()) * A;
            std::vector<float> y(static_cast<size_t>(F) * N);
            metal_probe::gemm(false, false, F, N, K, 1.0f, W.data(), b.data(), 0.0f, y.data());
            return y;
        };
        // The outputs of the sample at slot `slot` of a run over `n` samples.
        auto columnsOf = [&](const std::vector<float>& y, int n, int slot) {
            std::vector<float> out(static_cast<size_t>(F) * A);
            for (int f = 0; f < F; ++f)
                std::copy_n(y.data() + (static_cast<size_t>(f) * n + slot) * A, A,
                            out.data() + static_cast<size_t>(f) * A);
            return out;
        };
        std::vector<int> all(full);
        for (int b = 0; b < full; ++b) all[b] = b;
        const std::vector<float> yFull = run(all);
        for (int s : {0, 1, 7, 100, 255}) {
            const std::vector<float> alone = columnsOf(run({s}), 1, 0);
            assert(bitwiseEqual(alone.data(), columnsOf(yFull, full, s).data(), alone.size()));
            for (int slot : {0, 5}) {
                std::vector<int> eight(8);
                for (int& c : eight) c = rng.below(full);   // random companions
                eight[slot] = s;
                const std::vector<float> inEight = columnsOf(run(eight), 8, slot);
                assert(bitwiseEqual(alone.data(), inEight.data(), alone.size()));
            }
        }
    }
    // Rows: the Linear head products y [B][out] = x [B][in] . w^T, with B rows
    // of x, including batches that are not a multiple of the 64-row tile.
    {
        const GemmShape heads[] = {{0, 82, 162}, {0, 64, 81}, {0, 1, 64}};   // {_, out, in}
        const int full = 256;
        for (const GemmShape& h : heads) {
            const int out = h.N, in = h.K;
            const std::vector<float> w = randomVector(rng, static_cast<size_t>(out) * in, 0.1);
            const std::vector<float> x = randomVector(rng, static_cast<size_t>(full) * in);
            auto run = [&](const std::vector<int>& samples) {
                const int B = static_cast<int>(samples.size());
                std::vector<float> rows(static_cast<size_t>(B) * in), y(static_cast<size_t>(B) * out);
                for (int r = 0; r < B; ++r)
                    std::copy_n(x.data() + static_cast<size_t>(samples[r]) * in, in,
                                rows.data() + static_cast<size_t>(r) * in);
                metal_probe::gemm(false, true, B, out, in, 1.0f, rows.data(), w.data(), 0.0f, y.data());
                return y;
            };
            std::vector<int> all(full);
            for (int b = 0; b < full; ++b) all[b] = b;
            const std::vector<float> yFull = run(all);
            std::vector<int> first100(all.begin(), all.begin() + 100);
            const std::vector<float> y100 = run(first100);
            for (int s : {0, 1, 7, 100, 255}) {
                const std::vector<float> alone = run({s});
                assert(bitwiseEqual(alone.data(), yFull.data() + static_cast<size_t>(s) * out, out));
                if (s < 100)
                    assert(bitwiseEqual(alone.data(), y100.data() + static_cast<size_t>(s) * out, out));
                for (int slot : {0, 5}) {
                    std::vector<int> eight(8);
                    for (int& c : eight) c = rng.below(full);
                    eight[slot] = s;
                    const std::vector<float> y8 = run(eight);
                    assert(bitwiseEqual(alone.data(), y8.data() + static_cast<size_t>(slot) * out, out));
                }
            }
        }
    }
}

} // namespace

#endif // AZ_USE_METAL

int main() {
    std::string why;
    if (!metalBuilt()) {
        // A build without Metal must say how to get it.
        assert(!metalAvailable(&why));
        assert(why.find("-DAZ_USE_METAL=ON") != std::string::npos);
        std::printf("test_metal: SKIPPED (%s)\n", why.c_str());
        return 77;
    }
    if (!metalAvailable(&why)) {
        std::printf("test_metal: SKIPPED (%s)\n", why.c_str());
        return 77;
    }
#ifdef AZ_USE_METAL
    if (const char* seeds = std::getenv("AZGO_METAL_SEEDS")) gSeeds = std::max(1, std::atoi(seeds));
    test_context_comes_up();
    test_fill_kernel();
    test_fill_repeat_bitwise();
    test_gemm_against_double_reference();
    test_gemm_beta_zero_ignores_nan();
    test_gemm_vs_blas_forward_shapes();
    test_gemm_repeat_bitwise();
    test_gemm_columns_and_rows_batch_independent();
#endif
    std::printf("test_metal: all tests passed\n");
    return 0;
}
