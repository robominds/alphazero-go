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
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <system_error>
#include <unistd.h>
#include <atomic>
#include <thread>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/arena.hpp"
#include "azgo/backend.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/config.hpp"
#include "azgo/game.hpp"
#include "azgo/inference_hub.hpp"
#include "azgo/mcts.hpp"
#include "azgo/layers.hpp"
#include "azgo/metal.hpp"
#include "azgo/network.hpp"
#include "azgo/rng.hpp"
#include "azgo/run_state.hpp"
#include "azgo/trainer.hpp"

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

// The other half of the same guard: a tensor the layout knows (the network
// has it) but the GPU forward pass never reads, as when a CPU layer gains a
// tensor its GPU port does not use yet. Here the network has an ownership
// head and the GPU network is built without one: the first encode refuses,
// naming the unread tensors, instead of evaluating a different network.
void test_unread_tensor_refused() {
    const NetConfig withHead = netConfig(9, 1, 8, true), plain = netConfig(9, 1, 8);
    const Network net(withHead, 1);
    metal_probe::checkForwardReadsEveryTensor(net, withHead);   // the full port reads everything
    bool threw = false;
    try {
        metal_probe::checkForwardReadsEveryTensor(net, plain);
    } catch (const std::logic_error& e) {
        const std::string what = e.what();
        threw = what.find("ownership.conv.w") != std::string::npos &&
                what.find("ownership.bn.runVar") != std::string::npos;
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

// ---------------------------------------------------------------- the hub on the GPU

// Positions evaluated one at a time on the GPU: the reference for the hub.
struct GpuPool {
    NetConfig cfg;
    int count;
    std::vector<float> input, logits, values;
    size_t in() const { return static_cast<size_t>(cfg.planes()) * cfg.area(); }
    GpuPool(const Network& net, int count_, uint64_t seed) : cfg(net.config()), count(count_) {
        Rng rng(seed);
        input = randomPositions(rng, cfg, count);
        logits.resize(static_cast<size_t>(count) * cfg.actions());
        values.resize(count);
        metal_probe::InferNet alone(net, 1);
        for (int i = 0; i < count; ++i)
            alone.infer(input.data() + i * in(), 1, logits.data() + i * cfg.actions(), values.data() + i);
    }
};

// test_inference_hub's chaos test with the Metal runner: every request's
// outputs equal its positions evaluated alone on the GPU, bitwise, whatever
// batches the hub made of them.
void test_metal_hub_outputs_equal_alone_under_chaos() {
    const NetConfig cfg = netConfig(9, 2, 16);
    const Network net = randomNetwork(cfg, 501);
    const GpuPool pool(net, 300, 502);
    const size_t in = pool.in(), L = static_cast<size_t>(cfg.actions());
    for (int clients : {1, 4, 32}) {
        for (int maxBatch : {8, 64, 512}) {
            for (uint64_t chaos : {0ull, 11ull, 12ull, 13ull}) {
                InferenceHubOptions o;
                o.maxBatch = maxBatch;
                o.chaosSeed = chaos;
                InferenceHub hub(makeMetalRunner(net, maxBatch), cfg, o);
                std::atomic<long long> positions{0};
                std::vector<std::thread> threads;
                for (int t = 0; t < clients; ++t) {
                    threads.emplace_back([&, t] {
                        auto ev = hub.makeEvaluator(8);
                        Rng rng(deriveSeed(503, static_cast<uint64_t>(t), chaos,
                                           static_cast<uint64_t>(maxBatch)));
                        std::vector<float> x(8 * in), logits(8 * L), values(8);
                        for (int r = 0; r < 100; ++r) {
                            const int B = 1 + rng.below(8);
                            int picks[8];
                            for (int b = 0; b < B; ++b) {
                                picks[b] = rng.below(pool.count);
                                std::copy_n(pool.input.data() + picks[b] * in, in, x.data() + b * in);
                            }
                            ev->evaluate(x.data(), B, logits.data(), values.data());
                            for (int b = 0; b < B; ++b) {
                                assert(bitwiseEqual(logits.data() + b * L,
                                                    pool.logits.data() + picks[b] * L, L));
                                assert(bitwiseEqual(&values[b], &pool.values[picks[b]], 1));
                            }
                            positions += B;
                        }
                    });
                }
                for (auto& th : threads) th.join();
                assert(hub.stats().positions == positions.load());
            }
        }
    }
}

// The hub's outputs against Network::infer with BLAS.
// Measured on an M5 Max (macOS 27.0.1), 20 seeds: worst maxErr 6.5e-6 (ceiling
// 1e-5), in the values, for the reason test_infer_vs_cpu gives.
void test_metal_hub_vs_cpu() {
    auto blas = makeBackend("blas");
    MaxErr err("hub vs CPU", 1e-5);
    for (int seed = 0; seed < gSeeds; ++seed) {
        const NetConfig cfg = netConfig(9, 6, 32);
        const Network net = randomNetwork(cfg, 504 + seed);
        Rng rng(505 + 1000 * seed);
        const int B = 8, clients = 8;
        const size_t L = cfg.actions();
        std::vector<std::vector<float>> x(clients), logits(clients), values(clients);
        for (int t = 0; t < clients; ++t) {
            x[t] = randomPositions(rng, cfg, B);
            logits[t].resize(B * L);
            values[t].resize(B);
        }
        {
            InferenceHubOptions o;
            o.maxBatch = 64;
            InferenceHub hub(makeMetalRunner(net, 64), cfg, o);
            std::vector<std::thread> threads;
            for (int t = 0; t < clients; ++t)
                threads.emplace_back([&, t] {
                    auto ev = hub.makeEvaluator(B);
                    ev->evaluate(x[t].data(), B, logits[t].data(), values[t].data());
                });
            for (auto& th : threads) th.join();
        }
        Workspace ws(cfg, B, *blas);
        for (int t = 0; t < clients; ++t) {
            std::vector<float> refL(B * L), refV(B);
            net.infer(ws, x[t].data(), B, refL.data(), refV.data());
            err.add(logits[t].data(), refL.data(), refL.size());
            err.add(values[t].data(), refV.data(), refV.size());
        }
    }
}

// A runner whose two slots would not fit the GPU's recommended working set
// is refused before anything is allocated, with a message that names the
// size and the remedy.
void test_metal_runner_memory_check() {
    const NetConfig cfg = netConfig(19, 10, 128, false, 256);
    const Network net(cfg, 1);
    bool threw = false;
    try {
        makeMetalRunner(net, 1 << 20);
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        std::printf("test_metal: %s\n", what.c_str());
        threw = what.find("lower inference_batch") != std::string::npos &&
                what.find("MB") != std::string::npos;
    }
    assert(threw);
    auto ok = makeMetalRunner(net, 8);   // a sane batch fits
    assert(ok->slots() == kInferenceSlots && ok->maxBatch() == 8);
    // An arena wave's hubs keep one pass each in flight.
    assert(makeMetalRunner(net, 8, 1)->slots() == 1);
    for (int bad : {0, kInferenceSlots + 1}) {
        bool refused = false;
        try {
            makeMetalRunner(net, 8, bad);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        assert(refused);
    }
}

// ---------------------------------------------------------------- backend = metal in a run

namespace fs = std::filesystem;

// A 5x5 run small enough that a generation takes a fraction of a second,
// with the ownership head so the GPU validation pass covers it too.
Config metalRunConfig() {
    Config c;
    c.backend = "metal";
    c.boardSize = 5;
    c.history = 2;
    c.blocks = 1;
    c.filters = 8;
    c.valueHidden = 8;
    c.ownership = true;
    c.gamesPerGeneration = 4;
    c.simulations = 8;
    c.mctsBatch = 4;
    c.temperatureMoves = 4;
    c.bufferPositions = 1000;
    c.trainSteps = 4;
    c.batchSize = 16;
    c.arenaGames = 2;
    c.arenaSimulations = 4;
    c.arenaOpponents = 2;
    c.workers = 2;
    c.sgfPerGeneration = 1;
    c.botSimulations = {8};
    c.calibrationGames = 2;
    return c;
}

struct RunDir {
    fs::path path;
    explicit RunDir(const std::string& name)
        : path(fs::temp_directory_path() / ("azgo_test_metal_" + std::to_string(getpid()) + "_" + name)) {
        fs::remove_all(path);
    }
    ~RunDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string str() const { return path.string(); }
    std::string file(const std::string& rel) const { return (path / rel).string(); }
};

std::string readBytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    assert(in && "file should exist");
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::unique_ptr<Trainer> startRun(const Config& c, const RunDir& dir) {
    std::ostringstream startup;
    std::unique_ptr<Trainer> t = Trainer::startNew(c, dir.str(), {}, &startup);
    // The device line, as the spec gives it.
    assert(startup.str() == "gpu: " + metalDescription() + " (inference: self-play, arena, validation)\n");
    t->setLog(nullptr);
    return t;
}

// Everything a run produced that must not depend on workers, batching or
// timing: every checkpoint, state.bin, the rating files, the SGF and log.csv
// without its three timing columns.
std::string runFingerprint(const RunDir& dir, int generations) {
    std::string out;
    for (int g = 0; g <= generations; ++g) out += readBytes(checkpointPath(dir.str(), g));
    for (const char* f : {"state.bin", "results.csv", "references.csv", "elo.csv"})
        out += readBytes(dir.file(f));
    std::vector<std::string> sgf;
    for (const auto& e : fs::directory_iterator(dir.file("sgf"))) sgf.push_back(e.path().string());
    std::sort(sgf.begin(), sgf.end());
    assert(!sgf.empty());
    for (const std::string& f : sgf) out += readBytes(f);
    std::istringstream log(readBytes(dir.file("log.csv")));
    for (std::string row; std::getline(log, row);) {
        std::vector<std::string> fields(1);
        for (char ch : row) {
            if (ch == ',') fields.emplace_back();
            else fields.back() += ch;
        }
        for (size_t i = 0; i < fields.size(); ++i)
            if (i < 9 || i > 11) out += fields[i] + ",";   // drop selfplay_s, train_s, arena_s
        out += "\n";
    }
    return out;
}

// The determinism contract end to end: the same files for 1, 3 and 8
// workers, batches capped at 8 or 256, and the hub's chaos mode on.
void test_metal_run_independent_of_workers_and_batching() {
    struct Variant { int workers, inferenceBatch; uint64_t chaos; };
    const Variant variants[] = {{1, 256, 0}, {3, 8, 0}, {8, 256, 7}, {3, 8, 9}};
    std::string reference;
    for (const Variant& v : variants) {
        RunDir dir("independent");
        Config c = metalRunConfig();
        c.workers = v.workers;
        c.inferenceBatch = v.inferenceBatch;
        metal_probe::setHubChaosSeed(v.chaos);
        startRun(c, dir)->runUntil(2);
        metal_probe::setHubChaosSeed(0);
        const std::string fp = runFingerprint(dir, 2);
        if (reference.empty()) reference = fp;
        assert(fp == reference);
    }
}

void test_metal_resume_is_byte_identical() {
    RunDir a("resume_a"), b("resume_b");
    startRun(metalRunConfig(), a)->runUntil(4);
    startRun(metalRunConfig(), b)->runUntil(2);
    {
        std::ostringstream out;
        std::unique_ptr<Trainer> t = Trainer::resume(b.str(), {}, &out);
        assert(out.str() == "gpu: " + metalDescription() + " (inference: self-play, arena, validation)\n");
        t->setLog(nullptr);
        t->runUntil(4);
    }
    assert(runFingerprint(a, 4) == runFingerprint(b, 4));   // validation columns included
}

void test_metal_runs_are_reproducible() {
    RunDir a("repro_a"), b("repro_b");
    startRun(metalRunConfig(), a)->runUntil(2);
    startRun(metalRunConfig(), b)->runUntil(2);
    assert(runFingerprint(a, 2) == runFingerprint(b, 2));
}

void test_metal_rerate_reproduces_live() {
    RunDir dir("rerate");
    startRun(metalRunConfig(), dir)->runUntil(3);
    std::ostringstream log;
    assert(rerateRun(dir.str(), {}, &log) == 3);
    assert(log.str().rfind("gpu: " + metalDescription(), 0) == 0);
    for (const char* f : {"results.csv", "elo.csv", "references.csv"})
        assert(readBytes(dir.file(std::string("rerate/") + f)) == readBytes(dir.file(f)));
}

// A GPU failure during generation 2, in each phase that uses the GPU:
// self-play, the validation loss and the arena. The generation fails with
// the spec's message and state.bin still says generation 1. A failure in
// the arena comes after the checkpoint is written, which leaves a
// gen_0002.azn the run has not committed: re-rating must stop at
// state.bin's generation. A resume reproduces the uninterrupted run byte
// for byte. With one worker every pass holds one request, so the number of
// command buffers per phase is fixed, and the failing one can be placed:
// the 6th (self-play), the last before the arena (validation, measured on a
// run without an arena) and the last of the generation (the arena).
void test_gpu_failure_keeps_last_good_state() {
    Config c = metalRunConfig();
    c.workers = 1;
    RunDir clean("fail_clean"), noArena("fail_noarena");
    long long withArena = 0, withoutArena = 0;
    {
        std::unique_ptr<Trainer> t = startRun(c, clean);
        t->runUntil(1);
        const long long before = metal_probe::commandBuffersCommitted();
        t->runGeneration();
        withArena = metal_probe::commandBuffersCommitted() - before;
    }
    {
        Config a = c;
        a.arenaEvery = 0;
        std::unique_ptr<Trainer> t = startRun(a, noArena);
        t->runUntil(1);
        const long long before = metal_probe::commandBuffersCommitted();
        t->runGeneration();
        withoutArena = metal_probe::commandBuffersCommitted() - before;
    }
    assert(withoutArena > 6 && withArena > withoutArena);
    struct Case { const char* phase; long long n; };
    const Case cases[] = {{"self-play", 5}, {"validation", withoutArena - 1}, {"arena", withArena - 1}};
    for (const Case& k : cases) {
        RunDir broken(std::string("fail_") + k.phase);
        {
            std::unique_ptr<Trainer> t = startRun(c, broken);
            t->runUntil(1);
            metal_probe::failCommandBuffersAfter(static_cast<int>(k.n));
            bool threw = false;
            try {
                t->runGeneration();
            } catch (const std::runtime_error& e) {
                threw = std::string(e.what()).rfind("GPU inference failed: ", 0) == 0;
            }
            metal_probe::failCommandBuffersAfter(0);
            assert(threw);
        }
        assert(loadTrainingState(statePath(broken.str())).generation == 1);
        assert(fs::exists(checkpointPath(broken.str(), 2)) == (std::string(k.phase) == "arena"));
        assert(rerateRun(broken.str(), {}, nullptr) == 1);   // never the uncommitted generation 2
        {
            std::ostringstream out;
            std::unique_ptr<Trainer> t = Trainer::resume(broken.str(), {}, &out);
            t->setLog(nullptr);
            t->runUntil(2);
        }
        assert(runFingerprint(broken, 2) == runFingerprint(clean, 2));
    }
}

// The hubs' GPU memory is checked when a run starts or resumes, for
// inference_batch and two hubs at once (an arena match), before anything
// is written: a batch too large for the GPU fails up front, not in the
// first arena. The device's working set is replaced by a small limit here.
void test_metal_memory_checked_up_front() {
    RunDir fresh("mem_fresh"), existing("mem_existing");
    startRun(metalRunConfig(), existing)->runUntil(1);
    const std::string config = readBytes(configPath(existing.str()));
    metal_probe::setWorkingSetLimit(size_t(1) << 20);   // 1 MB
    auto refused = [](const std::function<void()>& f) {
        try {
            f();
        } catch (const std::runtime_error& e) {
            return std::string(e.what()).find("lower inference_batch") != std::string::npos;
        }
        return false;
    };
    std::ostringstream out;
    const bool startRefused = refused([&] { Trainer::startNew(metalRunConfig(), fresh.str(), {}, &out); });
    const bool resumeRefused =
        refused([&] { Trainer::resume(existing.str(), {"inference_batch=1024"}, &out); });
    metal_probe::setWorkingSetLimit(0);
    assert(startRefused && resumeRefused);
    assert(out.str().empty());   // no device line for a run that did not start
    assert(!fs::exists(fresh.path) || !fs::exists(statePath(fresh.str())));
    assert(!fs::exists(configPath(fresh.str())));
    assert(readBytes(configPath(existing.str())) == config);
}

// Arena matches on the GPU: network against network (random symmetry on one
// side, different per-side batches) and against a bot give the same result
// for 1 and 5 workers.
void test_metal_arena_matches() {
    NetConfig shape;
    shape.boardSize = 5;
    shape.history = 2;
    shape.blocks = 1;
    shape.filters = 8;
    shape.valueHidden = 8;
    const Network a = randomNetwork(shape, 601), b = randomNetwork(shape, 602);
    ArenaConfig cfg;
    cfg.games = 10;
    cfg.simulations = 12;
    cfg.mctsBatch = 4;
    cfg.temperatureMoves = 2;
    ArenaPlayer pa = ArenaPlayer::network(a, 0, true), pb = ArenaPlayer::network(b);
    pb.mctsBatch = 2;
    auto same = [](const MatchResult& x, const MatchResult& y) {
        return x.scoreA == y.scoreA && x.winsA == y.winsA && x.winsB == y.winsB && x.draws == y.draws &&
               x.distinctGames == y.distinctGames;
    };
    const MatchResult one = playMatch(pa, pb, cfg, 3, 1, "metal", 8);
    assert(same(one, playMatch(pa, pb, cfg, 3, 5, "metal", 8)));
    assert(same(one, playMatch(pa, pb, cfg, 3, 5, "metal", 256)));
    const ArenaPlayer bot = ArenaPlayer::bot(16);
    assert(same(playMatch(pa, bot, cfg, 4, 1, "metal", 8), playMatch(pa, bot, cfg, 4, 5, "metal", 8)));
    bool threw = false;   // the inference batch must hold each side's batch
    try {
        playMatch(pa, pb, cfg, 3, 1, "metal", 2);
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()).find("inference batch") != std::string::npos;
    }
    assert(threw);
}

// The GPU validation path (forward on the GPU, loss terms by
// lossFromOutputs) against Network::inferLoss with BLAS, on batches of 20
// and 32 like the validation pass's batches. Not on single positions: one
// whose saturated value matches its z has a value loss near 1e-8, which a
// float value cannot resolve (seed 5 of 20 gave 9.129e-9 against 9.140e-9).
// Measured on an M5 Max (macOS 27.0.1), 20 seeds: worst relative error 1.1e-7 (ceiling 1e-5).
void test_metal_validation_vs_cpu() {
    auto blas = makeBackend("blas");
    double worst = 0.0;
    for (int seed = 0; seed < gSeeds; ++seed) {
        for (bool ownership : {false, true}) {
            const NetConfig cfg = netConfig(9, 2, 16, ownership);
            const Network net = randomNetwork(cfg, 603 + seed);
            MetalInference gpu(net, 32);
            Workspace ws(cfg, 32, *blas);
            Rng rng(604 + 1000 * seed);
            for (int B : {20, 32}) {
                const std::vector<float> x = randomPositions(rng, cfg, B);
                std::vector<float> pi(static_cast<size_t>(B) * cfg.actions(), 0.0f), z(B), t(B * cfg.area());
                for (int i = 0; i < B; ++i) {
                    pi[i * cfg.actions() + rng.below(cfg.actions())] = 1.0f;
                    z[i] = rng.below(2) ? 1.0f : -1.0f;
                }
                for (float& v : t) v = static_cast<float>(rng.below(3)) - 1.0f;
                const float* own = ownership ? t.data() : nullptr;
                const LossStats ref = net.inferLoss(ws, x.data(), pi.data(), z.data(), B, own);
                const LossStats got = gpu.inferLoss(x.data(), pi.data(), z.data(), B, own);
                auto rel = [](float a, float r) {
                    return std::fabs(a - r) / std::max(std::fabs(r), 1e-30f);
                };
                worst = std::max({worst, (double)rel(got.policyLoss, ref.policyLoss),
                                  (double)rel(got.valueLoss, ref.valueLoss)});
                if (ownership) worst = std::max(worst, (double)rel(got.ownershipLoss, ref.ownershipLoss));
            }
        }
    }
    std::printf("test_metal: %-34s relErr %.2e (ceiling 1e-05)\n", "validation loss vs CPU", worst);
    assert(worst <= 1e-5);
}

// ---------------------------------------------------------------- analyze and gtp

// The GPU network called directly, one evaluate per pass: the reference for
// a search through a hub.
class DirectGpuEvaluator : public Evaluator {
public:
    DirectGpuEvaluator(const Network& net, int maxBatch)
        : cfg_(net.config()), gpu_(net, maxBatch), max_(maxBatch) {}
    int boardSize() const override { return cfg_.boardSize; }
    int history() const override { return cfg_.history; }
    int maxBatch() const override { return max_; }
    void evaluate(const float* input, int B, float* logits, float* values) override {
        gpu_.infer(input, B, logits, values);
    }
private:
    NetConfig cfg_;
    MetalInference gpu_;
    int max_;
};

// analyze and gtp search through a hub with one evaluator (WorkerEvaluators,
// workers = 1). That search must visit exactly what a search on the GPU
// network called directly visits, bitwise, with and without random symmetry.
void test_single_search_hub_matches_alone() {
    const NetConfig cfg = netConfig(9, 2, 16);
    const Network net = randomNetwork(cfg, 701);
    Game game(9, 7.5f, defaultMaxMoves(9));
    for (const char* m : {"E5", "C3", "G7", "D4"}) game.play(parseAction(m, 9));
    for (bool symmetry : {false, true}) {
        MctsConfig mc;
        mc.simulations = 64;
        mc.batchSize = 8;
        mc.randomSymmetry = symmetry;
        WorkerEvaluators hubbed(net, "metal", 1, mc.batchSize, mc.batchSize);
        DirectGpuEvaluator direct(net, mc.batchSize);
        Mcts a(mc, hubbed.forWorker(0)), b(mc, direct);
        Rng ra(5), rb(5);
        a.search(game, ra);
        b.search(game, rb);
        assert(a.rootVisits() == b.rootVisits());
        const float va = a.rootValue(), vb = b.rootValue();
        assert(std::memcmp(&va, &vb, sizeof va) == 0);
        assert(a.rootTotalVisits() >= 64);
    }
}

} // namespace

// ---------------------------------------------------------------- arena rounds at once

// playMatches on the GPU: every match of a set at once, one hub per
// network for all of them. Each result equals the match played alone, for
// any worker count, inference_batch and with the hubs' chaos mode on.
void test_metal_play_matches_equal_one_at_a_time() {
    NetConfig shape;
    shape.boardSize = 5;
    shape.history = 2;
    shape.blocks = 1;
    shape.filters = 8;
    shape.valueHidden = 8;
    const Network a = randomNetwork(shape, 611), b = randomNetwork(shape, 612), c = randomNetwork(shape, 613);
    ArenaConfig cfg;
    cfg.simulations = 12;
    cfg.mctsBatch = 4;
    cfg.temperatureMoves = 2;
    ArenaPlayer smallB = ArenaPlayer::network(b);
    smallB.mctsBatch = 2;
    const std::vector<ArenaMatchSpec> specs = {
        {ArenaPlayer::network(a), ArenaPlayer::network(b), 6, 201},
        {ArenaPlayer::network(a), ArenaPlayer::bot(12), 4, 202},
        {ArenaPlayer::network(a, 0, true), ArenaPlayer::network(c, 16), 4, 203},
        {ArenaPlayer::network(a), smallB, 2, 204},
        {ArenaPlayer::network(c), ArenaPlayer::network(c, 0, true), 4, 205},
        {ArenaPlayer::bot(4), ArenaPlayer::network(b), 2, 206},
    };
    auto same = [](const MatchResult& x, const MatchResult& y) {
        return x.games == y.games && x.scoreA == y.scoreA && x.winsA == y.winsA && x.winsB == y.winsB &&
               x.draws == y.draws && x.distinctGames == y.distinctGames;
    };
    std::vector<MatchResult> alone;
    for (const ArenaMatchSpec& m : specs) {
        ArenaConfig one = cfg;
        one.games = m.games;
        alone.push_back(playMatch(m.a, m.b, one, m.seed, 1, "metal", 8));
    }
    struct Variant { int workers, inferenceBatch; uint64_t chaos; };
    for (const Variant& v : {Variant{1, 8, 0}, Variant{3, 256, 0}, Variant{8, 16, 5}, Variant{0, 256, 0}}) {
        metal_probe::setHubChaosSeed(v.chaos);
        const std::vector<MatchResult> together = playMatches(specs, cfg, v.workers, "metal", v.inferenceBatch);
        metal_probe::setHubChaosSeed(0);
        for (size_t i = 0; i < specs.size(); ++i) assert(same(together[i], alone[i]));
    }
    arena_probe::setOneMatchAtATime(true);
    const std::vector<MatchResult> sequential = playMatches(specs, cfg, 5, "metal", 256);
    arena_probe::setOneMatchAtATime(false);
    for (size_t i = 0; i < specs.size(); ++i) assert(same(sequential[i], alone[i]));
}

// A metal run whose arena rounds have every kind of match: two bots, two
// reference networks, gating, and promotions (promote_threshold 0.5) with
// their calibration games. The trainer plays each wave at once; every file
// equals the run that plays its matches one after another, for workers 1,
// 3, 8 and 0, inference_batch 8 and 256, chaos on, on resume and on
// re-rating.
// A run whose arena rounds have every kind of match: two bots, two
// reference networks, gating and promotions (a second wave).
Config metalRoundsConfig(const RunDir& refs) {
    fs::create_directories(refs.path);
    for (int k = 0; k < 2; ++k) {
        if (fs::exists(refs.file("ref" + std::to_string(k) + ".azn"))) continue;
        RunDir donor("rounds_donor" + std::to_string(k));
        Config d = metalRunConfig();
        d.seed = 31 + k;
        startRun(d, donor)->runUntil(k);
        fs::copy_file(checkpointPath(donor.str(), k), refs.file("ref" + std::to_string(k) + ".azn"));
    }
    Config base = metalRunConfig();
    base.botSimulations = {8, 16};
    base.referenceNetworks = {refs.file("ref0.azn"), refs.file("ref1.azn")};
    base.arenaGames = 4;
    base.calibrationGames = 4;
    base.promoteThreshold = 0.5f;
    base.gating = true;
    base.gatingThreshold = 0.5f;
    // These settings were chosen so that a promotion happens within four
    // generations, which depends on generation 0's play. They were found
    // before generation 0's batch norm was calibrated (bn_calibrate), and a
    // calibrated generation 0 plays differently (here: no promotion by
    // generation 4), so the test keeps the uncalibrated one it was built on.
    base.bnCalibrate = false;
    return base;
}

void test_metal_rounds_at_once_equal_one_at_a_time() {
    RunDir refs("rounds_refs");
    const Config base = metalRoundsConfig(refs);

    RunDir sequential("rounds_seq");
    Config c = base;
    c.workers = 1;
    arena_probe::setOneMatchAtATime(true);
    startRun(c, sequential)->runUntil(4);
    arena_probe::setOneMatchAtATime(false);
    const std::string reference = runFingerprint(sequential, 4);
    assert(loadTrainingState(statePath(sequential.str())).references.back() > 0);   // a promotion

    struct Variant { int workers, inferenceBatch; uint64_t chaos; };
    for (const Variant& v : {Variant{1, 8, 0}, Variant{3, 256, 0}, Variant{8, 8, 7}, Variant{0, 256, 0}}) {
        RunDir dir("rounds_variant");
        c = base;
        c.workers = v.workers;
        c.inferenceBatch = v.inferenceBatch;
        metal_probe::setHubChaosSeed(v.chaos);
        startRun(c, dir)->runUntil(4);
        metal_probe::setHubChaosSeed(0);
        assert(runFingerprint(dir, 4) == reference);
    }

    RunDir resumed("rounds_resume");
    c = base;
    c.workers = 3;
    startRun(c, resumed)->runUntil(2);
    {
        std::ostringstream out;
        std::unique_ptr<Trainer> t = Trainer::resume(resumed.str(), {}, &out);
        t->setLog(nullptr);
        t->runUntil(4);
    }
    assert(runFingerprint(resumed, 4) == reference);

    arena_probe::setOneMatchAtATime(true);
    assert(rerateRun(sequential.str(), {}, nullptr) == 4);
    arena_probe::setOneMatchAtATime(false);
    assert(rerateRun(resumed.str(), {"workers=5"}, nullptr) == 4);
    for (const char* f : {"results.csv", "elo.csv", "references.csv", "reference-networks.csv", "log.csv"})
        assert(readBytes(resumed.file(std::string("rerate/") + f)) ==
               readBytes(sequential.file(std::string("rerate/") + f)));
}

// A metal engine plays every game of its slice at once by default
// (metal_workers = 0); with 8 or 64 workers, or as the default path with
// the workers key's rule, the files are the same.
void test_metal_engine_workers_change_nothing() {
    RunDir plain("engine_default");
    startRun(metalRunConfig(), plain)->runUntil(2);
    const std::string reference = runFingerprint(plain, 2);
    for (int workers : {0, 8, 64}) {
        RunDir dir("engine_workers");
        fs::create_directories(dir.path);
        const std::string hosts = dir.file("hosts");
        {
            std::ofstream out(hosts);
            out << "local metal_workers=" << workers << "\n";
        }
        Config c = metalRunConfig();
        applyOverride(c, "selfplay_engines=local/metal:1");
        TrainerOptions opts;
        opts.hostsFile = hosts;
        std::ostringstream startup;
        std::unique_ptr<Trainer> t = Trainer::startNew(c, dir.str(), {}, &startup, opts);
        assert(startup.str().find("engine local/metal: ") != std::string::npos);
        t->setLog(nullptr);
        t->runUntil(2);
        assert(runFingerprint(dir, 2) == reference);
        const std::string manifest = readBytes(dir.file("selfplay-engines.csv"));
        assert(manifest.find(metalDescription()) != std::string::npos);
    }
}

// GPU and CPU engines at once, all three backends: the files do not depend
// on the hosts file's worker counts or QoS classes.
void test_metal_mixed_engines_independent_of_hosts_settings() {
    const char* const settings[] = {"metal_workers=0 cpu_workers=0",
                                    "metal_workers=2 cpu_workers=1 cpu_qos=default",
                                    "metal_workers=64 cpu_workers=3 cpu_qos=utility"};
    std::string reference, referenceManifest;
    for (const char* line : settings) {
        RunDir dir("mixed_engines");
        fs::create_directories(dir.path);
        {
            std::ofstream out(dir.file("hosts"));
            out << "local " << line << "\n";
        }
        Config c = metalRunConfig();
        c.gamesPerGeneration = 12;
        applyOverride(c, "selfplay_engines=local/metal:3, local/blas=3, local/naive:1");
        TrainerOptions opts;
        opts.hostsFile = dir.file("hosts");
        std::ostringstream startup;
        std::unique_ptr<Trainer> t = Trainer::startNew(c, dir.str(), {}, &startup, opts);
        t->setLog(nullptr);
        t->runUntil(2);
        // The manifest without its seconds column (the 8th; the metal
        // fingerprint is quoted and holds commas, so split on the quotes).
        std::string manifest;
        std::istringstream rows(readBytes(dir.file("selfplay-engines.csv")));
        for (std::string row; std::getline(rows, row);) {
            std::vector<std::string> fields(1);
            bool quoted = false;
            for (char ch : row) {
                if (ch == '"') quoted = !quoted;
                else if (ch == ',' && !quoted) fields.emplace_back();
                else fields.back() += ch;
            }
            assert(fields.size() == 11);
            for (size_t i = 0; i < fields.size(); ++i)
                if (i != 7) manifest += fields[i] + "|";
            manifest += "\n";
        }
        // 12 games: blas 3 fixed, then 9 shared 3:1 as 7 (6 and the larger
        // remainder) and 2.
        assert(manifest.find("\n1|0|7|local/metal|") != std::string::npos);
        assert(manifest.find("\n1|7|3|local/blas|") != std::string::npos);
        assert(manifest.find("\n1|10|2|local/naive|") != std::string::npos);
        const std::string fp = runFingerprint(dir, 2);
        if (reference.empty()) {
            reference = fp;
            referenceManifest = manifest;
        }
        assert(fp == reference && manifest == referenceManifest);
    }
}


// The arena on its engines (arena_engines), all of them this GPU: the same
// files as the default metal run, gating decisions and promotions included,
// whatever the engines' workers and however the waves are split; with the
// hub's chaos mode on. A wave split between the GPU and a CPU engine plays
// other bits, but the same ones for any hosts settings.
void test_metal_arena_engines() {
    RunDir refs("arena_refs"), plain("arena_default");
    const Config base = metalRoundsConfig(refs);
    startRun(base, plain)->runUntil(4);
    const std::string reference = runFingerprint(plain, 4);
    struct Variant { const char* engines; const char* hosts; uint64_t chaos; };
    for (const Variant& v : {Variant{"local/metal:1", "metal_workers=0", 0},
                             Variant{"local/metal:1", "metal_workers=8 inference_batch=8", 5},
                             Variant{"local/metal:1, local/metal:2", "metal_workers=0", 0}}) {
        RunDir dir("arena_engines");
        fs::create_directories(dir.path);
        {
            std::ofstream out(dir.file("hosts"));
            out << "local " << v.hosts << "\n";
        }
        Config c = base;
        applyOverride(c, std::string("arena_engines=") + v.engines);
        TrainerOptions opts;
        opts.hostsFile = dir.file("hosts");
        std::ostringstream startup;
        metal_probe::setHubChaosSeed(v.chaos);
        std::unique_ptr<Trainer> t = Trainer::startNew(c, dir.str(), {}, &startup, opts);
        t->setLog(nullptr);
        t->runUntil(4);
        metal_probe::setHubChaosSeed(0);
        assert(startup.str().find("engine local/metal: ") != std::string::npos);
        assert(runFingerprint(dir, 4) == reference);
        assert(loadTrainingState(statePath(dir.str())).bestGeneration ==
               loadTrainingState(statePath(plain.str())).bestGeneration);
        const std::string manifest = readBytes(dir.file("arena-engines.csv"));
        assert(manifest.find(metalDescription()) != std::string::npos);
        assert(manifest.find("\n4,") != std::string::npos);
    }
    std::string mixed;
    for (const char* hosts : {"metal_workers=0 cpu_workers=1", "metal_workers=4 cpu_workers=3 cpu_qos=default"}) {
        RunDir dir("arena_mixed");
        fs::create_directories(dir.path);
        {
            std::ofstream out(dir.file("hosts"));
            out << "local " << hosts << "\n";
        }
        Config c = base;
        applyOverride(c, "arena_engines=local/metal:2, local/blas:1");
        TrainerOptions opts;
        opts.hostsFile = dir.file("hosts");
        std::ostringstream startup;
        std::unique_ptr<Trainer> t = Trainer::startNew(c, dir.str(), {}, &startup, opts);
        t->setLog(nullptr);
        t->runUntil(3);
        const std::string fp = runFingerprint(dir, 3);
        if (mixed.empty()) mixed = fp;
        assert(fp == mixed);
    }
}

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
    test_unread_tensor_refused();
    test_inference_workspace_size();
    test_metal_hub_outputs_equal_alone_under_chaos();
    test_metal_hub_vs_cpu();
    test_metal_runner_memory_check();
    test_metal_validation_vs_cpu();
    test_metal_arena_matches();
    test_metal_run_independent_of_workers_and_batching();
    test_metal_resume_is_byte_identical();
    test_metal_runs_are_reproducible();
    test_metal_rerate_reproduces_live();
    test_gpu_failure_keeps_last_good_state();
    test_metal_memory_checked_up_front();
    test_single_search_hub_matches_alone();
    test_metal_play_matches_equal_one_at_a_time();
    test_metal_rounds_at_once_equal_one_at_a_time();
    test_metal_engine_workers_change_nothing();
    test_metal_mixed_engines_independent_of_hosts_settings();
    test_metal_arena_engines();
#endif
    std::printf("test_metal: all tests passed\n");
    return 0;
}
