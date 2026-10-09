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
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/layers.hpp"
#include "azgo/metal.hpp"
#include "azgo/network.hpp"
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

// ---------------------------------------------------------------- forward layers

// The CPU loops of Network::forward that have no function of their own,
// written out: the input transpose [B][P][A] -> [P][B][A] and the policy
// flatten [2][B][A] -> [B][2A] (feature c * A + p).
std::vector<float> cpuTranspose(const std::vector<float>& in, int B, int P, int A) {
    std::vector<float> out(in.size());
    for (int p = 0; p < P; ++p)
        for (int b = 0; b < B; ++b)
            std::copy_n(in.data() + (static_cast<size_t>(b) * P + p) * A, A,
                        out.data() + (static_cast<size_t>(p) * B + b) * A);
    return out;
}

std::vector<float> cpuFlatten(const std::vector<float>& act, int B, int A) {
    std::vector<float> flat(act.size());
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < 2; ++c)
            std::copy_n(act.data() + (static_cast<size_t>(c) * B + b) * A, A,
                        flat.data() + static_cast<size_t>(b) * 2 * A + static_cast<size_t>(c) * A);
    return flat;
}

// Sample s of a channel-major tensor [C][B][A], as a batch of one [C][1][A].
std::vector<float> sampleOf(const std::vector<float>& t, int C, int B, int A, int s) {
    std::vector<float> out(static_cast<size_t>(C) * A);
    for (int c = 0; c < C; ++c)
        std::copy_n(t.data() + (static_cast<size_t>(c) * B + s) * A, A,
                    out.data() + static_cast<size_t>(c) * A);
    return out;
}

// Rows s * cols ... of a row-major [B][cols] matrix.
std::vector<float> rowOf(const std::vector<float>& t, int cols, int s) {
    return std::vector<float>(t.begin() + static_cast<size_t>(s) * cols,
                              t.begin() + static_cast<size_t>(s + 1) * cols);
}

bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && bitwiseEqual(a.data(), b.data(), a.size());
}

void test_copy_kernels_bitwise() {
    Rng rng(301);
    NaiveBackend cpu;
    for (int n : {5, 9, 19}) {
        const int A = n * n;
        for (int C : {1, 18, 96}) {
            for (int B : {1, 2, 7}) {
                const std::vector<float> x = randomVector(rng, static_cast<size_t>(B) * C * A);
                std::vector<float> t(x.size());
                metal_probe::transposeInput(x.data(), B, C, A, t.data());
                assert(same(t, cpuTranspose(x, B, C, A)));

                std::vector<float> col(static_cast<size_t>(C) * 9 * B * A), ref(col.size());
                metal_probe::im2col3(x.data(), C, B, n, col.data());
                cpu.im2col(x.data(), C, B, n, 3, ref.data());
                assert(same(col, ref));
            }
        }
        for (int B : {1, 2, 7}) {
            const std::vector<float> act = randomVector(rng, static_cast<size_t>(2) * B * A);
            std::vector<float> flat(act.size());
            metal_probe::policyFlatten(act.data(), B, A, flat.data());
            assert(same(flat, cpuFlatten(act, B, A)));
        }
    }
}

// Conv2d::forward with the BLAS backend against the GPU's im2col + gemm.
// Measured on an M5 Max (macOS 27.0.1), 20 seeds: worst maxErr 5.6e-7 (ceiling 2e-6).
void test_conv_forward_vs_cpu() {
    auto blas = makeBackend("blas");
    MaxErr err("conv forward vs CPU", 2e-6);
    for (int seed = 0; seed < gSeeds; ++seed) {
        Rng rng(302 + 1000 * seed);
        for (int F : {8, 32, 96}) {
            for (int n : {5, 9, 19}) {
                for (int B : {1, 16, 256}) {
                    const int M = B * n * n;
                    struct Shape { int cin, cout, k; };
                    const Shape shapes[] = {{18, F, 3}, {F, F, 3}, {F, 1, 1}, {F, 2, 1}};
                    for (const Shape sh : shapes) {
                        Conv2d conv(sh.cin, sh.cout, sh.k);
                        conv.init(rng);
                        const std::vector<float> x = randomVector(rng, static_cast<size_t>(sh.cin) * M);
                        std::vector<float> col(conv.colSize(B, n)), ref(static_cast<size_t>(sh.cout) * M);
                        conv.forward(*blas, x.data(), B, n, col.data(), ref.data());
                        std::vector<float> y(ref.size());
                        metal_probe::convForward(conv, x.data(), B, n, y.data());
                        err.add(y.data(), ref.data(), y.size());
                    }
                }
            }
        }
    }
}

// BatchNorm2d::forwardInfer with random gamma, beta and running statistics.
// Measured on an M5 Max (macOS 27.0.1), 20 seeds: maxErr 0 (ceiling 1e-6): the
// GPU's scale, shift and fma gave the CPU's bits on every element.
void test_bn_infer_vs_cpu() {
    MaxErr err("BN inference vs CPU", 1e-6);
    for (int seed = 0; seed < gSeeds; ++seed) {
        Rng rng(303 + 1000 * seed);
        for (int C : {1, 2, 32, 96}) {
            for (int M : {1, 25, 567, 20736}) {
                BatchNorm2d bn(C);
                for (int c = 0; c < C; ++c) {
                    bn.gamma[c] = static_cast<float>(0.5 + rng.uniform());
                    bn.beta[c] = static_cast<float>(rng.uniform() - 0.5);
                    bn.runMean[c] = static_cast<float>(2.0 * rng.uniform() - 1.0);
                    bn.runVar[c] = static_cast<float>(0.05 + 2.0 * rng.uniform());
                }
                const std::vector<float> x = randomVector(rng, static_cast<size_t>(C) * M, 3.0);
                std::vector<float> ref(x.size()), y(x.size());
                bn.forwardInfer(x.data(), M, ref.data());
                metal_probe::bnInfer(bn, x.data(), M, y.data());
                err.add(y.data(), ref.data(), y.size());
            }
        }
    }
}

void test_relu_add_bitwise() {
    Rng rng(304);
    for (size_t count : {size_t(1), size_t(255), size_t(257), size_t(96 * 20736)}) {
        std::vector<float> x = randomVector(rng, count);
        // The corner cases: signed zeros and NaN (the CPU's x > 0 ? x : 0 maps
        // all three to +0).
        x[0] = -0.0f;
        if (count > 2) {
            x[1] = 0.0f;
            x[2] = std::numeric_limits<float>::quiet_NaN();
        }
        std::vector<float> ref = x, y = x;
        reluForward(ref.data(), count);
        metal_probe::relu(y.data(), count);
        assert(same(y, ref));

        const std::vector<float> a = randomVector(rng, count), b = randomVector(rng, count);
        std::vector<float> sum = a, expect = a;
        for (size_t i = 0; i < count; ++i) expect[i] += b[i];
        metal_probe::addInto(sum.data(), b.data(), count);
        assert(same(sum, expect));
    }
}

// The value and ownership heads' tanh against std::tanh (float), over [-20, 20].
// Measured on an M5 Max (macOS 27.0.1), 20 seeds: worst maxErr 1.8e-7 (ceiling 1e-6).
void test_tanh_vs_cpu() {
    MaxErr err("tanh vs CPU", 1e-6);
    for (int seed = 0; seed < gSeeds; ++seed) {
        Rng rng(305 + 1000 * seed);
        std::vector<float> x = randomVector(rng, 100000, 20.0);
        for (float v : {0.0f, -0.0f, 1e-8f, -1e-8f, 0.5f, -0.5f, 9.0f, -9.0f, 20.0f, -20.0f})
            x.push_back(v);
        std::vector<float> ref(x.size()), y = x;
        for (size_t i = 0; i < x.size(); ++i) ref[i] = std::tanh(x[i]);
        metal_probe::tanhInPlace(y.data(), y.size());
        err.add(y.data(), ref.data(), y.size());
        // Element by element as well, so small values are held to their own scale.
        for (size_t i = 0; i < x.size(); ++i)
            assert(std::fabs(y[i] - ref[i]) <= 1e-6 * std::max(1e-3f, std::fabs(ref[i])));
    }
}

// Linear::forward (gemm against w^T, then the bias per row) with the BLAS
// backend, on the heads of the three presets. maxErr is taken per layer over
// its outputs at all three batches together: value fc2 at batch 1 is a single
// number, and when its bias cancels the product (0.0025 left from a bias of
// -0.10) max|ref| stops measuring the layer's scale. Taken per batch, that
// case reached 3.3e-5 with an absolute error of 8e-8, plain float rounding.
// Measured on an M5 Max (macOS 27.0.1), 20 seeds: worst maxErr 7.5e-7 (ceiling 2e-6).
void test_linear_forward_vs_cpu() {
    auto blas = makeBackend("blas");
    MaxErr err("Linear forward vs CPU", 2e-6);
    struct Preset { int n, V; };
    for (int seed = 0; seed < gSeeds; ++seed) {
        Rng rng(306 + 1000 * seed);
        for (const Preset pr : {Preset{5, 32}, Preset{9, 64}, Preset{19, 256}}) {
            const int A = pr.n * pr.n;
            struct Shape { int in, out; };
            for (const Shape sh : {Shape{2 * A, A + 1}, Shape{A, pr.V}, Shape{pr.V, 1}}) {
                Linear fc(sh.in, sh.out);
                fc.init(rng);
                for (float& b : fc.b) b = static_cast<float>(rng.uniform() - 0.5);
                std::vector<float> ys, refs;
                for (int B : {1, 16, 256}) {
                    const std::vector<float> x = randomVector(rng, static_cast<size_t>(B) * sh.in);
                    std::vector<float> ref(static_cast<size_t>(B) * sh.out), y(ref.size());
                    fc.forward(*blas, x.data(), B, ref.data());
                    metal_probe::linearForward(fc, x.data(), B, y.data());
                    ys.insert(ys.end(), y.begin(), y.end());
                    refs.insert(refs.end(), ref.begin(), ref.end());
                }
                err.add(ys.data(), refs.data(), ys.size());
            }
        }
    }
}

// Every forward layer kernel gives a sample the same bits alone (a batch of
// one) as at its place in a batch of 256 (9x9, 96 channels).
void test_layer_kernels_batch_independent() {
    Rng rng(307);
    const int n = 9, A = n * n, B = 256, F = 96, P = 18, V = 64;
    const std::vector<int> picks = {0, 1, 7, 100, 255};

    // transpose_input: [B][P][A] -> [P][B][A].
    {
        const std::vector<float> x = randomVector(rng, static_cast<size_t>(B) * P * A);
        std::vector<float> t(x.size());
        metal_probe::transposeInput(x.data(), B, P, A, t.data());
        for (int s : picks) {
            const std::vector<float> one = rowOf(x, P * A, s);
            std::vector<float> t1(one.size());
            metal_probe::transposeInput(one.data(), 1, P, A, t1.data());
            assert(same(t1, sampleOf(t, P, B, A, s)));
        }
    }
    // im2col3, the convolutions, BN, ReLU, add and tanh on [F][B][A].
    {
        const std::vector<float> x = randomVector(rng, static_cast<size_t>(F) * B * A);
        const std::vector<float> skip = randomVector(rng, x.size());
        std::vector<float> col(static_cast<size_t>(F) * 9 * B * A);
        metal_probe::im2col3(x.data(), F, B, n, col.data());
        Conv2d c3(F, F, 3), c1(F, 2, 1);
        c3.init(rng);
        c1.init(rng);
        std::vector<float> y3(x.size()), y1(static_cast<size_t>(2) * B * A);
        metal_probe::convForward(c3, x.data(), B, n, y3.data());
        metal_probe::convForward(c1, x.data(), B, n, y1.data());
        BatchNorm2d bn(F);
        for (int c = 0; c < F; ++c) {
            bn.gamma[c] = static_cast<float>(0.5 + rng.uniform());
            bn.beta[c] = static_cast<float>(rng.uniform() - 0.5);
            bn.runMean[c] = static_cast<float>(rng.uniform() - 0.5);
            bn.runVar[c] = static_cast<float>(0.1 + rng.uniform());
        }
        std::vector<float> yb(x.size());
        metal_probe::bnInfer(bn, x.data(), B * A, yb.data());
        std::vector<float> relu = x, added = x, th = x;
        metal_probe::relu(relu.data(), relu.size());
        metal_probe::addInto(added.data(), skip.data(), added.size());
        metal_probe::tanhInPlace(th.data(), th.size());
        for (int s : picks) {
            const std::vector<float> xs = sampleOf(x, F, B, A, s);
            const std::vector<float> ks = sampleOf(skip, F, B, A, s);
            std::vector<float> col1(static_cast<size_t>(F) * 9 * A);
            metal_probe::im2col3(xs.data(), F, 1, n, col1.data());
            assert(same(col1, sampleOf(col, F * 9, B, A, s)));
            std::vector<float> y31(xs.size()), y11(static_cast<size_t>(2) * A);
            metal_probe::convForward(c3, xs.data(), 1, n, y31.data());
            metal_probe::convForward(c1, xs.data(), 1, n, y11.data());
            assert(same(y31, sampleOf(y3, F, B, A, s)));
            assert(same(y11, sampleOf(y1, 2, B, A, s)));
            std::vector<float> yb1(xs.size());
            metal_probe::bnInfer(bn, xs.data(), A, yb1.data());
            assert(same(yb1, sampleOf(yb, F, B, A, s)));
            std::vector<float> r1 = xs, a1 = xs, t1 = xs;
            metal_probe::relu(r1.data(), r1.size());
            metal_probe::addInto(a1.data(), ks.data(), a1.size());
            metal_probe::tanhInPlace(t1.data(), t1.size());
            assert(same(r1, sampleOf(relu, F, B, A, s)));
            assert(same(a1, sampleOf(added, F, B, A, s)));
            assert(same(t1, sampleOf(th, F, B, A, s)));
        }
    }
    // policy_flatten [2][B][A] -> [B][2A] and the Linear heads [B][in] -> [B][out].
    {
        const std::vector<float> act = randomVector(rng, static_cast<size_t>(2) * B * A);
        std::vector<float> flat(act.size());
        metal_probe::policyFlatten(act.data(), B, A, flat.data());
        struct Shape { int in, out; };
        for (const Shape sh : {Shape{2 * A, A + 1}, Shape{A, V}, Shape{V, 1}}) {
            Linear fc(sh.in, sh.out);
            fc.init(rng);
            for (float& b : fc.b) b = static_cast<float>(rng.uniform() - 0.5);
            const std::vector<float> x = randomVector(rng, static_cast<size_t>(B) * sh.in);
            std::vector<float> y(static_cast<size_t>(B) * sh.out);
            metal_probe::linearForward(fc, x.data(), B, y.data());
            for (int s : picks) {
                const std::vector<float> xs = rowOf(x, sh.in, s);
                std::vector<float> ys(sh.out);
                metal_probe::linearForward(fc, xs.data(), 1, ys.data());
                assert(same(ys, rowOf(y, sh.out, s)));
            }
        }
        for (int s : picks) {
            const std::vector<float> as = sampleOf(act, 2, B, A, s);
            std::vector<float> f1(as.size());
            metal_probe::policyFlatten(as.data(), 1, A, f1.data());
            assert(same(f1, rowOf(flat, 2 * A, s)));
        }
    }
}

// ---------------------------------------------------------------- the network

NetConfig netConfig(int n, int blocks, int filters, bool ownership = false, int valueHidden = 16) {
    NetConfig c;
    c.boardSize = n;
    c.history = 8;
    c.blocks = blocks;
    c.filters = filters;
    c.valueHidden = valueHidden;
    c.ownership = ownership;
    return c;
}

// A network whose batch norms and biases are not the identity at
// initialization (gamma 1, beta 0, mean 0, variance 1, zero biases), so the
// GPU must apply every one of them.
Network randomNetwork(const NetConfig& cfg, uint64_t seed) {
    Network net(cfg, seed);
    Rng rng(seed + 17);
    for (ParamView& p : net.params()) {
        const std::string& n = p.name;
        if (n.size() > 6 && n.compare(n.size() - 6, 6, ".gamma") == 0)
            for (size_t i = 0; i < p.size; ++i) p.value[i] = static_cast<float>(0.5 + rng.uniform());
        else if (!p.decay)   // beta and Linear biases
            for (size_t i = 0; i < p.size; ++i) p.value[i] = static_cast<float>(0.4 * rng.uniform() - 0.2);
    }
    for (BufferView& b : net.buffers()) {
        const bool var = b.name.find("runVar") != std::string::npos;
        for (size_t i = 0; i < b.size; ++i)
            b.value[i] = static_cast<float>(var ? 0.5 + rng.uniform() : 0.4 * rng.uniform() - 0.2);
    }
    return net;
}

// Encoded positions: random 0/1 planes, [B][planes][area].
std::vector<float> randomPositions(Rng& rng, const NetConfig& cfg, int B) {
    std::vector<float> x(static_cast<size_t>(B) * cfg.planes() * cfg.area());
    for (float& v : x) v = static_cast<float>(rng.below(2));
    return x;
}

struct Outputs {
    std::vector<float> logits, values, ownership;
};

Outputs gpuInfer(metal_probe::InferNet& net, const NetConfig& cfg, const std::vector<float>& x, int B) {
    Outputs o;
    o.logits.resize(static_cast<size_t>(B) * cfg.actions());
    o.values.resize(B);
    if (cfg.ownership) o.ownership.resize(static_cast<size_t>(B) * cfg.area());
    net.infer(x.data(), B, o.logits.data(), o.values.data(), cfg.ownership ? o.ownership.data() : nullptr);
    return o;
}

// The whole inference forward against Network::infer with BLAS.
// Measured on an M5 Max (macOS 27.0.1), 20 seeds: worst maxErr 5.5e-6 (ceiling
// 1e-5), always in the values: a random network's value sits near 0 for
// every position (max |v| ~ 0.01 when fc2's bias cancels its product), so an
// absolute error of a few 1e-8 shows up large. Logits and ownership: <= 4.1e-7.
void test_infer_vs_cpu() {
    auto blas = makeBackend("blas");
    MaxErr err("network forward vs CPU", 1e-5);
    const NetConfig configs[] = {
        netConfig(5, 1, 8), netConfig(9, 2, 16), netConfig(9, 6, 32), netConfig(9, 0, 16),
        netConfig(9, 2, 16, true), netConfig(19, 2, 16),
    };
    for (int seed = 0; seed < gSeeds; ++seed) {
        Rng rng(401 + 1000 * seed);
        for (const NetConfig& cfg : configs) {
            const Network net = randomNetwork(cfg, 40 + seed);
            metal_probe::InferNet gpu(net, 64);
            for (int B : {1, 7, 64}) {
                const std::vector<float> x = randomPositions(rng, cfg, B);
                Workspace ws(cfg, B, *blas);
                Outputs ref;
                ref.logits.resize(static_cast<size_t>(B) * cfg.actions());
                ref.values.resize(B);
                if (cfg.ownership) ref.ownership.resize(static_cast<size_t>(B) * cfg.area());
                net.infer(ws, x.data(), B, ref.logits.data(), ref.values.data(),
                          cfg.ownership ? ref.ownership.data() : nullptr);
                const Outputs got = gpuInfer(gpu, cfg, x, B);
                err.add(got.logits.data(), ref.logits.data(), got.logits.size());
                err.add(got.values.data(), ref.values.data(), got.values.size());
                if (cfg.ownership)
                    err.add(got.ownership.data(), ref.ownership.data(), got.ownership.size());
            }
        }
    }
}

// What the batching evaluator rests on, for the whole network: a position's
// logits, value and ownership are the same bits alone, in batches of 8, 144
// and 256, at the first or the last slot, among random companions and with
// the batch permuted.
void test_infer_batch_independent() {
    const NetConfig cfg = netConfig(9, 6, 32, true);
    const Network net = randomNetwork(cfg, 402);
    metal_probe::InferNet gpu(net, 256);
    Rng rng(403);
    const int full = 256;
    const size_t planeFloats = static_cast<size_t>(cfg.planes()) * cfg.area();
    const std::vector<float> pool = randomPositions(rng, cfg, full);
    // The positions `picks`, in that order, as one batch.
    auto batchOf = [&](const std::vector<int>& picks) {
        std::vector<float> x(picks.size() * planeFloats);
        for (size_t i = 0; i < picks.size(); ++i)
            std::copy_n(pool.data() + picks[i] * planeFloats, planeFloats, x.data() + i * planeFloats);
        return x;
    };
    auto run = [&](const std::vector<int>& picks) {
        return gpuInfer(gpu, cfg, batchOf(picks), static_cast<int>(picks.size()));
    };
    // Slot `slot` of o equals the outputs alone a.
    auto sameAt = [&](const Outputs& o, int slot, const Outputs& a) {
        const size_t L = cfg.actions(), Ar = cfg.area();
        return bitwiseEqual(o.logits.data() + slot * L, a.logits.data(), L) &&
               bitwiseEqual(o.values.data() + slot, a.values.data(), 1) &&
               bitwiseEqual(o.ownership.data() + slot * Ar, a.ownership.data(), Ar);
    };
    std::vector<int> all(full);
    for (int i = 0; i < full; ++i) all[i] = i;
    const Outputs whole = run(all);
    std::vector<int> permuted = all;
    for (int i = full - 1; i > 0; --i) std::swap(permuted[i], permuted[rng.below(i + 1)]);
    const Outputs shuffled = run(permuted);
    for (int s : {0, 1, 7, 100, 255}) {
        const Outputs alone = run({s});
        assert(sameAt(whole, s, alone));
        const auto it = std::find(permuted.begin(), permuted.end(), s);
        const int where = static_cast<int>(it - permuted.begin());
        assert(sameAt(shuffled, where, alone));
        for (int size : {8, 144, 256}) {
            for (int slot : {0, size - 1}) {
                std::vector<int> picks(size);
                for (int& p : picks) p = rng.below(full);   // random companions
                picks[slot] = s;
                assert(sameAt(run(picks), slot, alone));
            }
        }
    }
}

void test_infer_repeat_bitwise() {
    const NetConfig cfg = netConfig(9, 2, 16, true);
    const Network net = randomNetwork(cfg, 404);
    metal_probe::InferNet gpu(net, 32);
    Rng rng(405);
    const std::vector<float> x = randomPositions(rng, cfg, 32);
    const Outputs first = gpuInfer(gpu, cfg, x, 32);
    for (int r = 0; r < 20; ++r) {
        const Outputs again = gpuInfer(gpu, cfg, x, 32);
        assert(same(again.logits, first.logits) && same(again.values, first.values) &&
               same(again.ownership, first.ownership));
    }
}

// Two separately uploaded networks (own weights and workspace) agree bitwise.
void test_two_networks_bitwise() {
    const NetConfig cfg = netConfig(9, 2, 16, true);
    const Network net = randomNetwork(cfg, 406);
    metal_probe::InferNet a(net, 16), b(net, 64);
    Rng rng(407);
    const std::vector<float> x = randomPositions(rng, cfg, 16);
    const Outputs oa = gpuInfer(a, cfg, x, 16), ob = gpuInfer(b, cfg, x, 16);
    assert(same(oa.logits, ob.logits) && same(oa.values, ob.values) && same(oa.ownership, ob.ownership));
}

// A layer added to the CPU network without a GPU port must be refused, not
// silently evaluated wrong: a layout without the ownership head refuses a
// network that has one, naming the tensor.
void test_unknown_tensor_refused() {
    const Network plain(netConfig(9, 1, 8), 1), withHead(netConfig(9, 1, 8, true), 1);
    metal_probe::checkUpload(plain, plain);   // the same shape uploads
    bool threw = false;
    try {
        metal_probe::checkUpload(plain, withHead);
    } catch (const std::logic_error& e) {
        threw = std::string(e.what()).find("ownership.conv.w") != std::string::npos;
    }
    assert(threw);
}

// The inference workspace of the 9x9 6x96 network at batch 256 is ~100 MB
// (the spec's estimate), most of it the shared im2col buffer.
void test_inference_workspace_size() {
    const NetConfig cfg = netConfig(9, 6, 96, false, 64);
    const Network net(cfg, 1);
    metal_probe::InferNet gpu(net, 256);
    const double mb = static_cast<double>(gpu.bytes()) / 1e6;
    std::printf("test_metal: 9x9 6x96 inference workspace at batch 256: %.1f MB\n", mb);
    assert(mb > 85.0 && mb < 115.0);
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
    test_copy_kernels_bitwise();
    test_conv_forward_vs_cpu();
    test_bn_infer_vs_cpu();
    test_relu_add_bitwise();
    test_tanh_vs_cpu();
    test_linear_forward_vs_cpu();
    test_layer_kernels_batch_independent();
    test_infer_vs_cpu();
    test_infer_batch_independent();
    test_infer_repeat_bitwise();
    test_two_networks_bitwise();
    test_unknown_tensor_refused();
    test_inference_workspace_size();
#endif
    std::printf("test_metal: all tests passed\n");
    return 0;
}
