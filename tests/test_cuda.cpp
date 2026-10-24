// GPU tests for backend = cuda: the context and its architecture check, every
// kernel against an exact CPU reference, and the batch-independence
// properties the inference hub relies on.
//
// They run only in a build configured with -DAZ_USE_CUDA=ON on a machine with
// a usable NVIDIA GPU. Without CUDA built, or without a GPU, main() returns
// 77, which CMake declares as this test's SKIP_RETURN_CODE, so CTest reports
// "Skipped" rather than a failure. A GPU the build's machine code was not
// made for fails instead: that build cannot run where it is being tested.
//
// The references are exact, not tolerances: the gemm's is an std::fma chain
// in k order (what the kernel computes), batch norm's an explicit multiply
// then add. CMake compiles this file with -ffp-contract=off, so the compiler
// does not fuse the references' own a * b + c. Only tanh, which the GPU and
// the C library each round in their own way, is held to a tolerance (2 ulp);
// the whole network is held to the Metal spec's tolerances against the CPU
// Network ("Tolerance against the CPU").
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <algorithm>
#include <cassert>
#include <cmath>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <unistd.h>
#include <map>
#include <atomic>
#include <condition_variable>
#include <dirent.h>
#include <mutex>
#include <regex>
#include <thread>
#include <string>
#include <vector>
#include "azgo/arena.hpp"
#include "azgo/backend.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/config.hpp"
#include "azgo/cuda.hpp"
#include "azgo/inference_hub.hpp"
#include "azgo/layers.hpp"
#include "azgo/network.hpp"
#include "azgo/replay_buffer.hpp"
#include "azgo/rng.hpp"
#include "azgo/run_state.hpp"
#include "azgo/selfplay_slice.hpp"
#include "azgo/serialize.hpp"
#include "azgo/trainer.hpp"

using namespace azgo;

#ifdef AZ_USE_CUDA

namespace {

// ---------------------------------------------------------------- helpers

std::vector<float> randomVector(Rng& rng, size_t count, double scale = 1.0) {
    std::vector<float> v(count);
    for (float& x : v) x = static_cast<float>((rng.uniform() * 2.0 - 1.0) * scale);
    return v;
}

bool bitwiseEqual(const float* a, const float* b, size_t count) {
    return count == 0 || std::memcmp(a, b, count * sizeof(float)) == 0;
}

// ---------------------------------------------------------------- exact CPU references

// What the gemm computes for C[m][n]: one fma chain in k order from +0.
// std::fma rounds once, as the GPU's fmaf does, so the result is exact
// to the bit.
float fmaChain(const float* A, const float* B, bool transB, int N, int K, int m, int n) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float b = transB ? B[static_cast<size_t>(n) * K + k] : B[static_cast<size_t>(k) * N + n];
        acc = std::fma(A[static_cast<size_t>(m) * K + k], b, acc);
    }
    return acc;
}

// BatchNorm2d::forwardInfer as written, with no fused multiply-add: the
// scale and shift per channel, then y = scale * x + shift, each operation
// rounded on its own (this file is compiled with -ffp-contract=off).
std::vector<float> bnInferNoContract(const BatchNorm2d& bn, const std::vector<float>& x, int M) {
    std::vector<float> y(x.size());
    for (int c = 0; c < bn.c; ++c) {
        const float scale = bn.gamma[c] / std::sqrt(bn.runVar[c] + BatchNorm2d::kEps);
        const float shift = bn.beta[c] - scale * bn.runMean[c];
        for (int i = 0; i < M; ++i) {
            const size_t j = static_cast<size_t>(c) * M + i;
            y[j] = scale * x[j] + shift;
        }
    }
    return y;
}

// [B][P][A] -> [P][B][A], and the policy head's [2][B][A] -> [B][2A].
std::vector<float> cpuTranspose(const std::vector<float>& x, int B, int P, int A) {
    std::vector<float> t(x.size());
    for (int b = 0; b < B; ++b)
        for (int p = 0; p < P; ++p)
            for (int a = 0; a < A; ++a)
                t[(static_cast<size_t>(p) * B + b) * A + a] = x[(static_cast<size_t>(b) * P + p) * A + a];
    return t;
}

std::vector<float> cpuFlatten(const std::vector<float>& act, int B, int A) {
    std::vector<float> flat(act.size());
    for (int c = 0; c < 2; ++c)
        for (int b = 0; b < B; ++b)
            for (int a = 0; a < A; ++a)
                flat[static_cast<size_t>(b) * 2 * A + c * A + a] = act[(static_cast<size_t>(c) * B + b) * A + a];
    return flat;
}

// Distance in units in the last place between two finite floats.
long long ulpDistance(float a, float b) {
    auto key = [](float f) {
        int32_t i;
        std::memcpy(&i, &f, sizeof i);
        return i < 0 ? static_cast<long long>(INT32_MIN) - i : static_cast<long long>(i);
    };
    return std::llabs(key(a) - key(b));
}

// A batch-norm layer with running statistics like a trained network's.
BatchNorm2d randomBn(Rng& rng, int C) {
    BatchNorm2d bn(C);
    for (int c = 0; c < C; ++c) {
        bn.gamma[c] = static_cast<float>(0.5 + rng.uniform());
        bn.beta[c] = static_cast<float>(rng.uniform() - 0.5);
        bn.runMean[c] = static_cast<float>(2.0 * rng.uniform() - 1.0);
        bn.runVar[c] = static_cast<float>(0.05 + 2.0 * rng.uniform());
    }
    bn.runVar[0] = 0.0f;   // a dead channel: the variance at its floor, eps alone
    return bn;
}

// The worst error a test saw against its ceiling: max|x - ref| / max|ref|
// (the Metal spec's maxErr), gathered over every comparison and printed once.
struct MaxErr {
    const char* what;
    double ceiling;
    double worst = 0.0;
    MaxErr(const char* w, double c) : what(w), ceiling(c) {}
    void add(const float* x, const float* ref, size_t count) {
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < count; ++i) {
            assert(std::isfinite(x[i]));
            num = std::max(num, std::fabs(static_cast<double>(x[i]) - ref[i]));
            den = std::max(den, std::fabs(static_cast<double>(ref[i])));
        }
        const double e = den > 0.0 ? num / den : num;
        if (!(e <= ceiling)) {
            std::printf("test_cuda: %s: maxErr %.3g above the ceiling %.0e\n", what, e, ceiling);
            std::fflush(stdout);
            assert(false);
        }
        worst = std::max(worst, e);
    }
    ~MaxErr() { std::printf("test_cuda: %-28s maxErr %.2e (ceiling %.0e)\n", what, worst, ceiling); }
};

// ---------------------------------------------------------------- networks

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
// initialization, so the GPU must apply every one of them.
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

Outputs gpuInfer(cuda_probe::InferNet& net, const NetConfig& cfg, const std::vector<float>& x, int B) {
    Outputs o;
    o.logits.resize(static_cast<size_t>(B) * cfg.actions());
    o.values.resize(B);
    if (cfg.ownership) o.ownership.resize(static_cast<size_t>(B) * cfg.area());
    net.infer(x.data(), B, o.logits.data(), o.values.data(), cfg.ownership ? o.ownership.data() : nullptr);
    return o;
}

bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && bitwiseEqual(a.data(), b.data(), a.size());
}

// ---------------------------------------------------------------- the context

void test_available_and_description() {
    std::string why = "unchanged";
    assert(cudaAvailable(&why));
    assert(why == "unchanged");   // only written when unusable
    const std::string d = cudaDescription();
    std::printf("test_cuda: %s\n", d.c_str());
    static const std::regex shape("^.+ sm_[0-9]+, driver API [0-9]+\\.[0-9]+, runtime [0-9]+\\.[0-9]+, kernels [0-9a-f]{8}$");
    assert(std::regex_match(d, shape));
    assert(d == cudaDescription());
    assert(d.rfind(cuda_probe::deviceName() + " sm_" + std::to_string(cuda_probe::deviceArchitecture()), 0) == 0);
    assert(d.find(',') != std::string::npos && d.find(';') == std::string::npos);
}

// The machine code that runs was built for exactly this GPU's architecture:
// no PTX was compiled by the driver, and no older SASS runs on a newer GPU.
void test_architecture_matches() {
    assert(cuda_probe::binaryVersion() == cuda_probe::deviceArchitecture());
    assert(cuda_probe::deviceArchitecture() >= 50);
}

void test_fill() {
    for (size_t n : {size_t(0), size_t(1), size_t(1000), size_t(1) << 20}) {
        const size_t tail = 37;
        std::vector<float> data(n + tail, -3.0f);
        cuda_probe::fill(data.data(), n, 2.5f, tail);
        for (size_t i = 0; i < n; ++i) assert(data[i] == 2.5f);
        for (size_t i = n; i < n + tail; ++i) assert(data[i] == -3.0f);
    }
}

// ---------------------------------------------------------------- the gemm

struct GemmShape {
    bool transB;
    int M, N, K;
};

// Runs the GPU gemm (tile -1: the rule) and checks C against the fma chain,
// bitwise: every element of a small product, 4096 random ones of a large one.
void checkGemm(Rng& rng, const GemmShape& g, int tile) {
    const std::vector<float> A = randomVector(rng, static_cast<size_t>(g.M) * g.K);
    const std::vector<float> B = randomVector(rng, static_cast<size_t>(g.K) * g.N);
    std::vector<float> C(static_cast<size_t>(g.M) * g.N, std::nanf(""));
    cuda_probe::gemm(g.transB, g.M, g.N, g.K, A.data(), B.data(), C.data(), tile);
    const double work = double(g.M) * g.N * g.K;
    const size_t samples = work <= 2e7 ? C.size() : 4096;
    for (size_t s = 0; s < samples; ++s) {
        const size_t i = samples == C.size() ? s : static_cast<size_t>(rng.below(static_cast<int>(C.size())));
        const int m = static_cast<int>(i / g.N), n = static_cast<int>(i % g.N);
        const float ref = fmaChain(A.data(), B.data(), g.transB, g.N, g.K, m, n);
        if (!bitwiseEqual(&C[i], &ref, 1)) {
            std::printf("test_cuda: gemm %s M=%d N=%d K=%d tile %d: C[%d][%d] = %.9g, fma chain %.9g\n",
                        g.transB ? "NT" : "NN", g.M, g.N, g.K, tile, m, n, C[i], ref);
            std::fflush(stdout);
            assert(false);
        }
    }
}

// The project's networks' gemms: a convolution's (M = filters, K = inputs x
// 9, N = pixels x batch) and the heads' 1x1 convolutions and Linear layers
// (rows = samples, B transposed), plus edge sizes around the tiles.
void test_gemm_equals_fma_chain() {
    Rng rng(301);
    std::vector<GemmShape> shapes;
    for (int M : {1, 17, 96})
        for (int N : {1, 15, 257})
            for (int K : {1, 16, 17, 255})
                for (bool t : {false, true}) shapes.push_back({t, M, N, K});
    struct Net { int n, F, hidden; };
    for (const Net& net : {Net{5, 32, 64}, Net{9, 64, 64}, Net{9, 96, 64}, Net{13, 96, 64}, Net{19, 128, 256}}) {
        const int A = net.n * net.n;
        for (int B : {1, 8}) {
            shapes.push_back({false, net.F, A * B, 18 * 9});        // stem
            shapes.push_back({false, net.F, A * B, net.F * 9});     // a block's convolution
            shapes.push_back({false, 2, A * B, net.F});             // policy 1x1
            shapes.push_back({false, 1, A * B, net.F});             // value 1x1
            shapes.push_back({true, B, A + 1, 2 * A});              // policy Linear
            shapes.push_back({true, B, net.hidden, A});             // value fc1
            shapes.push_back({true, B, 1, net.hidden});             // value fc2
        }
    }
    shapes.push_back({false, 96, 169 * 144, 864});   // 13x13 6x96 at batch 144 (sampled)
    shapes.push_back({true, 256, 170, 338});         // its policy Linear at batch 256
    for (const GemmShape& g : shapes)
        for (int tile : {-1, 0, 1, 2}) checkGemm(rng, g, tile);
}

// The three tile shapes give the same bits on a whole product.
void test_gemm_tiles_agree() {
    Rng rng(302);
    for (const GemmShape& g : {GemmShape{false, 96, 169 * 8, 864}, GemmShape{false, 128, 361 * 3, 1152},
                               GemmShape{true, 37, 170, 338}}) {
        const std::vector<float> A = randomVector(rng, static_cast<size_t>(g.M) * g.K);
        const std::vector<float> B = randomVector(rng, static_cast<size_t>(g.K) * g.N);
        std::vector<float> first(static_cast<size_t>(g.M) * g.N), other(first.size());
        cuda_probe::gemm(g.transB, g.M, g.N, g.K, A.data(), B.data(), first.data(), 0);
        for (int tile : {1, 2}) {
            cuda_probe::gemm(g.transB, g.M, g.N, g.K, A.data(), B.data(), other.data(), tile);
            assert(bitwiseEqual(first.data(), other.data(), first.size()));
        }
    }
}

// What the inference hub rests on, for the gemm: a sample's outputs are the
// same bits alone and at any position in batches of 8, 144 and 256, among
// random companions. In a convolution a sample is `area` columns of op(B);
// in a Linear layer it is one row of A.
void test_gemm_batch_independent() {
    Rng rng(303);
    const int area = 169, F = 96, K = 9 * F;
    const std::vector<float> W = randomVector(rng, static_cast<size_t>(F) * K);
    const std::vector<float> sample = randomVector(rng, static_cast<size_t>(K) * area);   // [K][area]
    std::vector<float> alone(static_cast<size_t>(F) * area);
    cuda_probe::gemm(false, F, area, K, W.data(), sample.data(), alone.data());
    for (int B : {8, 144, 256}) {
        for (int at : {0, B / 2 + 1, B - 1}) {
            const int N = area * B;
            std::vector<float> cols = randomVector(rng, static_cast<size_t>(K) * N);
            for (int k = 0; k < K; ++k)
                std::copy_n(&sample[static_cast<size_t>(k) * area], area, &cols[static_cast<size_t>(k) * N + at * area]);
            std::vector<float> out(static_cast<size_t>(F) * N);
            cuda_probe::gemm(false, F, N, K, W.data(), cols.data(), out.data());
            for (int m = 0; m < F; ++m)
                assert(bitwiseEqual(&out[static_cast<size_t>(m) * N + at * area], &alone[static_cast<size_t>(m) * area], area));
        }
    }
    // Linear: y [B][out] = x [B][in] . w^T.
    const int in = 2 * area, outs = area + 1;
    const std::vector<float> w = randomVector(rng, static_cast<size_t>(outs) * in);
    const std::vector<float> row = randomVector(rng, in);
    std::vector<float> rowAlone(outs);
    cuda_probe::gemm(true, 1, outs, in, row.data(), w.data(), rowAlone.data());
    for (int B : {8, 144, 256}) {
        for (int at : {0, B / 2 + 1, B - 1}) {
            std::vector<float> x = randomVector(rng, static_cast<size_t>(B) * in);
            std::copy(row.begin(), row.end(), x.begin() + static_cast<size_t>(at) * in);
            std::vector<float> y(static_cast<size_t>(B) * outs);
            cuda_probe::gemm(true, B, outs, in, x.data(), w.data(), y.data());
            assert(bitwiseEqual(&y[static_cast<size_t>(at) * outs], rowAlone.data(), outs));
        }
    }
}

// A chain that ends at -0 (a product too small even for a denormal rounds
// to -0) stays -0 through the zero padding past K: K = 1 and K = 17 are not
// multiples of the 16-wide k step, so the padding runs after the last term.
void test_gemm_keeps_negative_zero() {
    for (int K : {1, 17}) {
        for (bool transB : {false, true}) {
            const int M = 3, N = 5;
            std::vector<float> A(static_cast<size_t>(M) * K, 0.0f), B(static_cast<size_t>(K) * N, -1.0f);
            for (int m = 0; m < M; ++m) A[static_cast<size_t>(m) * K] = -1e-30f;
            for (int n = 0; n < N; ++n) (transB ? B[static_cast<size_t>(n) * K] : B[n]) = 1e-30f;
            std::vector<float> C(static_cast<size_t>(M) * N);
            for (int tile : {0, 1, 2}) {
                cuda_probe::gemm(transB, M, N, K, A.data(), B.data(), C.data(), tile);
                for (int m = 0; m < M; ++m)
                    for (int n = 0; n < N; ++n) {
                        const float ref = fmaChain(A.data(), B.data(), transB, N, K, m, n);
                        assert(ref == 0.0f && std::signbit(ref));
                        assert(bitwiseEqual(&C[static_cast<size_t>(m) * N + n], &ref, 1));
                    }
            }
        }
    }
}

// C is only written: NaN in it beforehand leaves no trace.
void test_gemm_ignores_c() {
    Rng rng(304);
    const int M = 33, N = 300, K = 70;
    const std::vector<float> A = randomVector(rng, static_cast<size_t>(M) * K), B = randomVector(rng, static_cast<size_t>(K) * N);
    std::vector<float> C(static_cast<size_t>(M) * N, std::nanf(""));
    cuda_probe::gemm(false, M, N, K, A.data(), B.data(), C.data());
    for (float c : C) assert(std::isfinite(c));
}

// ---------------------------------------------------------------- the elementwise kernels

// The copies are bit for bit the CPU's: the input transpose, im2col (k = 3,
// zeros off the board, never a neighbouring sample's pixels) and the policy
// head's flatten.
void test_copy_kernels_bitwise() {
    Rng rng(401);
    NaiveBackend cpu;
    for (int n : {5, 9, 13, 19}) {
        const int A = n * n;
        for (int C : {1, 18, 96}) {
            for (int B : {1, 2, 8}) {
                const std::vector<float> x = randomVector(rng, static_cast<size_t>(B) * C * A);
                std::vector<float> t(x.size());
                cuda_probe::transposeInput(x.data(), B, C, A, t.data());
                const std::vector<float> tRef = cpuTranspose(x, B, C, A);
                assert(bitwiseEqual(t.data(), tRef.data(), t.size()));
                std::vector<float> col(static_cast<size_t>(C) * 9 * B * A), ref(col.size());
                cuda_probe::im2col3(x.data(), C, B, n, col.data());
                cpu.im2col(x.data(), C, B, n, 3, ref.data());
                assert(bitwiseEqual(col.data(), ref.data(), col.size()));
            }
        }
        for (int B : {1, 2, 8}) {
            const std::vector<float> act = randomVector(rng, static_cast<size_t>(2) * B * A);
            std::vector<float> flat(act.size());
            cuda_probe::policyFlatten(act.data(), B, A, flat.data());
            const std::vector<float> ref = cpuFlatten(act, B, A);
            assert(bitwiseEqual(flat.data(), ref.data(), flat.size()));
        }
    }
}

// Inference batch norm, bitwise against its formula evaluated without fused
// multiply-adds (what BatchNorm2d::forwardInfer computes where the compiler
// does not contract it, as g++ on x86-64 does not).
void test_bn_infer_bitwise() {
    Rng rng(402);
    for (int C : {1, 2, 96}) {
        for (int M : {1, 169, 169 * 8}) {
            const BatchNorm2d bn = randomBn(rng, C);
            const std::vector<float> x = randomVector(rng, static_cast<size_t>(C) * M, 5.0);
            std::vector<float> y(x.size());
            cuda_probe::bnInfer(bn, x.data(), M, y.data());
            const std::vector<float> ref = bnInferNoContract(bn, x, M);
            assert(bitwiseEqual(y.data(), ref.data(), y.size()));
#if defined(__x86_64__) && !defined(__FMA__)
            // Here the CPU layer itself is that formula, uncontracted.
            std::vector<float> cpu(x.size());
            bn.forwardInfer(x.data(), M, cpu.data());
            assert(bitwiseEqual(y.data(), cpu.data(), y.size()));
#endif
        }
    }
}

// ReLU (x > 0 ? x : 0, which maps -0 and NaN to +0), the skip connection's
// sum and the Linear layers' bias, bitwise.
void test_relu_add_bias_bitwise() {
    Rng rng(403);
    std::vector<float> x = randomVector(rng, 100000, 3.0);
    for (float v : {0.0f, -0.0f, std::nanf(""), INFINITY, -INFINITY, 1e-40f, -1e-40f}) x.push_back(v);
    std::vector<float> y = x, ref = x;
    reluForward(ref.data(), ref.size());
    cuda_probe::relu(y.data(), y.size());
    assert(bitwiseEqual(y.data(), ref.data(), y.size()));

    const std::vector<float> a = randomVector(rng, 50000), b = randomVector(rng, 50000);
    std::vector<float> sum = a, sumRef(a.size());
    for (size_t i = 0; i < a.size(); ++i) sumRef[i] = a[i] + b[i];
    cuda_probe::addInto(sum.data(), b.data(), sum.size());
    assert(bitwiseEqual(sum.data(), sumRef.data(), sum.size()));

    const int rows = 37, cols = 170;
    const std::vector<float> bias = randomVector(rng, cols);
    std::vector<float> m = randomVector(rng, static_cast<size_t>(rows) * cols), mRef = m;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) mRef[static_cast<size_t>(r) * cols + c] += bias[c];
    cuda_probe::biasAddRows(m.data(), bias.data(), rows, cols);
    assert(bitwiseEqual(m.data(), mRef.data(), m.size()));
}

// tanh: the GPU's tanhf and the C library's round differently, so it is the
// one kernel held to a tolerance: within 2 ulp of tanh in double, rounded.
void test_tanh_within_two_ulp() {
    Rng rng(404);
    std::vector<float> x = randomVector(rng, 1000000, 20.0);
    for (float v : {0.0f, -0.0f, 1e-8f, -1e-8f, 1e-40f, 0.5f, -0.5f, 9.0f, -9.0f, 20.0f, -20.0f, 100.0f})
        x.push_back(v);
    std::vector<float> y = x;
    cuda_probe::tanhInPlace(y.data(), y.size());
    long long worst = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        const float ref = static_cast<float>(std::tanh(static_cast<double>(x[i])));
        worst = std::max(worst, ulpDistance(y[i], ref));
    }
    std::printf("test_cuda: tanh worst %lld ulp (ceiling 2)\n", worst);
    assert(worst <= 2);
    float special[] = {INFINITY, -INFINITY, std::nanf(""), 0.0f, -0.0f};
    cuda_probe::tanhInPlace(special, 5);
    assert(special[0] == 1.0f && special[1] == -1.0f && std::isnan(special[2]));
    assert(special[3] == 0.0f && !std::signbit(special[3]) && special[4] == 0.0f && std::signbit(special[4]));
}

// Each kernel's output for a sample is the same alone and inside a batch:
// im2col (the one with neighbours: padding never reads another sample) and
// batch norm with the sample at the first, a middle and the last position.
void test_elementwise_batch_independent() {
    Rng rng(405);
    const int n = 13, A = n * n, C = 18;
    const std::vector<float> one = randomVector(rng, static_cast<size_t>(C) * A);   // [C][1][A]
    std::vector<float> colAlone(static_cast<size_t>(C) * 9 * A);
    cuda_probe::im2col3(one.data(), C, 1, n, colAlone.data());
    const BatchNorm2d bn = randomBn(rng, C);
    std::vector<float> bnAlone(one.size());
    cuda_probe::bnInfer(bn, one.data(), A, bnAlone.data());
    for (int B : {8, 144}) {
        for (int at : {0, B / 2, B - 1}) {
            std::vector<float> x = randomVector(rng, static_cast<size_t>(C) * B * A);   // [C][B][A]
            for (int c = 0; c < C; ++c)
                std::copy_n(&one[static_cast<size_t>(c) * A], A, &x[(static_cast<size_t>(c) * B + at) * A]);
            std::vector<float> col(static_cast<size_t>(C) * 9 * B * A);
            cuda_probe::im2col3(x.data(), C, B, n, col.data());
            for (int r = 0; r < C * 9; ++r)
                assert(bitwiseEqual(&col[static_cast<size_t>(r) * B * A + at * A], &colAlone[static_cast<size_t>(r) * A], A));
            std::vector<float> y(x.size());
            cuda_probe::bnInfer(bn, x.data(), B * A, y.data());
            for (int c = 0; c < C; ++c)
                assert(bitwiseEqual(&y[(static_cast<size_t>(c) * B + at) * A], &bnAlone[static_cast<size_t>(c) * A], A));
        }
    }
}

// ---------------------------------------------------------------- the fused convolution

Conv2d randomConv(Rng& rng, int cin, int cout) {
    Conv2d conv(cin, cout, 3);
    for (float& w : conv.w) w = static_cast<float>((rng.uniform() * 2.0 - 1.0) * std::sqrt(2.0 / (cin * 9)));
    return conv;
}

// The explicit sequence on the GPU, kernel by kernel: im2col, the gemm, batch
// norm, the skip connection (if any) and ReLU, as the Metal backend runs a
// residual block's convolution.
std::vector<float> explicitConv(const Conv2d& conv, const BatchNorm2d& bn, const std::vector<float>& x,
                                const std::vector<float>* skip, int B, int n) {
    const int N = B * n * n, K = conv.cin * 9;
    std::vector<float> col(static_cast<size_t>(K) * N), y(static_cast<size_t>(conv.cout) * N);
    cuda_probe::im2col3(x.data(), conv.cin, B, n, col.data());
    cuda_probe::gemm(false, conv.cout, N, K, conv.w.data(), col.data(), y.data());
    cuda_probe::bnInfer(bn, y.data(), N, y.data());
    if (skip) cuda_probe::addInto(y.data(), skip->data(), y.size());
    cuda_probe::relu(y.data(), y.size());
    return y;
}

// The fused convolution (the column matrix read straight from the input,
// batch norm, the skip and ReLU in the gemm's last step) gives exactly the
// explicit sequence's bits, for the stem's and the blocks' shapes on every
// board, with and without the skip, for every tile shape.
void test_conv_fused_equals_explicit() {
    Rng rng(501);
    struct Case { int cin, cout, n, B; };
    for (const Case& c : {Case{18, 32, 5, 1}, Case{18, 64, 9, 8}, Case{64, 64, 9, 144}, Case{96, 96, 13, 8},
                          Case{96, 96, 13, 256}, Case{18, 96, 13, 144}, Case{128, 128, 19, 1},
                          Case{128, 128, 19, 8}, Case{7, 3, 4, 3}}) {
        const Conv2d conv = randomConv(rng, c.cin, c.cout);
        const BatchNorm2d bn = randomBn(rng, c.cout);
        const size_t A = static_cast<size_t>(c.n) * c.n;
        const std::vector<float> x = randomVector(rng, static_cast<size_t>(c.cin) * c.B * A);
        const std::vector<float> skip = randomVector(rng, static_cast<size_t>(c.cout) * c.B * A);
        for (bool withSkip : {false, true}) {
            const std::vector<float> ref = explicitConv(conv, bn, x, withSkip ? &skip : nullptr, c.B, c.n);
            for (int tile : {-1, 0, 1, 2}) {
                std::vector<float> y(ref.size(), std::nanf(""));
                cuda_probe::convBnRelu(conv, bn, x.data(), withSkip ? skip.data() : nullptr, c.B, c.n, y.data(),
                                       tile);
                if (!bitwiseEqual(y.data(), ref.data(), y.size())) {
                    std::printf("test_cuda: fused conv %d->%d on %dx%d, batch %d, skip %d, tile %d differs\n",
                                c.cin, c.cout, c.n, c.n, c.B, withSkip, tile);
                    std::fflush(stdout);
                    assert(false);
                }
            }
        }
    }
}

// A sample's outputs from the fused convolution are the same bits alone and
// at the first, a middle and the last position of batches of 8, 144 and 256.
void test_conv_fused_batch_independent() {
    Rng rng(502);
    const int cin = 96, cout = 96, n = 13, A = n * n;
    const Conv2d conv = randomConv(rng, cin, cout);
    const BatchNorm2d bn = randomBn(rng, cout);
    const std::vector<float> one = randomVector(rng, static_cast<size_t>(cin) * A);
    const std::vector<float> oneSkip = randomVector(rng, static_cast<size_t>(cout) * A);
    std::vector<float> alone(static_cast<size_t>(cout) * A);
    cuda_probe::convBnRelu(conv, bn, one.data(), oneSkip.data(), 1, n, alone.data());
    for (int B : {8, 144, 256}) {
        for (int at : {0, B / 2 + 1, B - 1}) {
            std::vector<float> x = randomVector(rng, static_cast<size_t>(cin) * B * A);
            std::vector<float> skip = randomVector(rng, static_cast<size_t>(cout) * B * A);
            for (int c = 0; c < cin; ++c)
                std::copy_n(&one[static_cast<size_t>(c) * A], A, &x[(static_cast<size_t>(c) * B + at) * A]);
            for (int c = 0; c < cout; ++c)
                std::copy_n(&oneSkip[static_cast<size_t>(c) * A], A, &skip[(static_cast<size_t>(c) * B + at) * A]);
            std::vector<float> y(static_cast<size_t>(cout) * B * A);
            cuda_probe::convBnRelu(conv, bn, x.data(), skip.data(), B, n, y.data());
            for (int c = 0; c < cout; ++c)
                assert(bitwiseEqual(&y[(static_cast<size_t>(c) * B + at) * A], &alone[static_cast<size_t>(c) * A], A));
        }
    }
}

// ---------------------------------------------------------------- the whole network

// The worst of max|x - ref| / max|ref| over logits, values and ownership.
double outputsErr(const Outputs& x, const Outputs& ref) {
    double worst = 0.0;
    auto one = [&](const std::vector<float>& a, const std::vector<float>& r) {
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            assert(std::isfinite(a[i]));
            num = std::max(num, std::fabs(static_cast<double>(a[i]) - r[i]));
            den = std::max(den, std::fabs(static_cast<double>(r[i])));
        }
        worst = std::max(worst, den > 0.0 ? num / den : num);
    };
    one(x.logits, ref.logits);
    one(x.values, ref.values);
    one(x.ownership, ref.ownership);
    return worst;
}

Outputs cpuInfer(const Network& net, const Backend& backend, const std::vector<float>& x, int B) {
    const NetConfig& cfg = net.config();
    Workspace ws(cfg, B, backend);
    Outputs o;
    o.logits.resize(static_cast<size_t>(B) * cfg.actions());
    o.values.resize(B);
    if (cfg.ownership) o.ownership.resize(static_cast<size_t>(B) * cfg.area());
    net.infer(ws, x.data(), B, o.logits.data(), o.values.data(), cfg.ownership ? o.ownership.data() : nullptr);
    return o;
}

// The whole network in double precision, straight from its definition
// (Network::forward in inference mode): the reference both the GPU and the
// CPU backends are measured against when float noise is the question.
Outputs doubleInfer(const Network& network, const std::vector<float>& input, int B) {
    Network& net = const_cast<Network&>(network);   // params() hands out views; nothing is written
    const NetConfig& cfg = net.config();
    std::map<std::string, std::vector<double>> t;
    for (const ParamView& p : net.params()) t[p.name].assign(p.value, p.value + p.size);
    for (const BufferView& v : net.buffers()) t[v.name].assign(v.value, v.value + v.size);
    const int n = cfg.boardSize, A = cfg.area(), P = cfg.planes(), F = cfg.filters;
    const size_t M = static_cast<size_t>(B) * A;
    // x [C][B][A] -> y [cout][B][A], 'same' padding.
    auto conv = [&](const std::vector<double>& x, int cin, int cout, int k, const std::string& name) {
        const std::vector<double>& w = t.at(name + ".w");
        std::vector<double> y(static_cast<size_t>(cout) * M, 0.0);
        for (int o = 0; o < cout; ++o)
            for (int b = 0; b < B; ++b)
                for (int p = 0; p < A; ++p) {
                    double sum = 0.0;
                    for (int c = 0; c < cin; ++c)
                        for (int dy = 0; dy < k; ++dy)
                            for (int dx = 0; dx < k; ++dx) {
                                const int yy = p / n + dy - k / 2, xx = p % n + dx - k / 2;
                                if (yy < 0 || yy >= n || xx < 0 || xx >= n) continue;
                                sum += w[((static_cast<size_t>(o) * cin + c) * k + dy) * k + dx] *
                                       x[(static_cast<size_t>(c) * B + b) * A + yy * n + xx];
                            }
                    y[(static_cast<size_t>(o) * B + b) * A + p] = sum;
                }
        return y;
    };
    auto norm = [&](std::vector<double>& x, int C, const std::string& name) {
        for (int c = 0; c < C; ++c) {
            const double scale = t.at(name + ".gamma")[c] / std::sqrt(t.at(name + ".runVar")[c] + BatchNorm2d::kEps);
            for (size_t i = 0; i < M; ++i) {
                double& v = x[static_cast<size_t>(c) * M + i];
                v = (v - t.at(name + ".runMean")[c]) * scale + t.at(name + ".beta")[c];
            }
        }
    };
    auto relu = [](std::vector<double>& x) {
        for (double& v : x) v = v > 0.0 ? v : 0.0;
    };
    // y [B][out] = x [B][in] . w^T + b
    auto linear = [&](const std::vector<double>& x, int in, int out, const std::string& name) {
        const std::vector<double>&w = t.at(name + ".w"), &bias = t.at(name + ".b");
        std::vector<double> y(static_cast<size_t>(B) * out);
        for (int s = 0; s < B; ++s)
            for (int o = 0; o < out; ++o) {
                double sum = bias[o];
                for (int i = 0; i < in; ++i) sum += x[static_cast<size_t>(s) * in + i] * w[static_cast<size_t>(o) * in + i];
                y[static_cast<size_t>(s) * out + o] = sum;
            }
        return y;
    };
    std::vector<double> x(static_cast<size_t>(P) * M);
    for (int b = 0; b < B; ++b)
        for (int p = 0; p < P; ++p)
            for (int a = 0; a < A; ++a)
                x[(static_cast<size_t>(p) * B + b) * A + a] = input[(static_cast<size_t>(b) * P + p) * A + a];
    std::vector<double> h = conv(x, P, F, 3, "stem.conv");
    norm(h, F, "stem.bn");
    relu(h);
    for (int i = 0; i < cfg.blocks; ++i) {
        const std::string pre = "block" + std::to_string(i);
        std::vector<double> mid = conv(h, F, F, 3, pre + ".conv1");
        norm(mid, F, pre + ".bn1");
        relu(mid);
        std::vector<double> out = conv(mid, F, F, 3, pre + ".conv2");
        norm(out, F, pre + ".bn2");
        for (size_t j = 0; j < out.size(); ++j) out[j] += h[j];
        relu(out);
        h = out;
    }
    Outputs o;
    std::vector<double> pol = conv(h, F, 2, 1, "policy.conv");
    norm(pol, 2, "policy.bn");
    relu(pol);
    std::vector<double> flat(2 * M);
    for (int c = 0; c < 2; ++c)
        for (int b = 0; b < B; ++b)
            for (int a = 0; a < A; ++a)
                flat[static_cast<size_t>(b) * 2 * A + c * A + a] = pol[(static_cast<size_t>(c) * B + b) * A + a];
    for (double v : linear(flat, 2 * A, cfg.actions(), "policy.fc")) o.logits.push_back(static_cast<float>(v));
    std::vector<double> val = conv(h, F, 1, 1, "value.conv");
    norm(val, 1, "value.bn");
    relu(val);
    std::vector<double> hidden = linear(val, A, cfg.valueHidden, "value.fc1");
    relu(hidden);
    for (double v : linear(hidden, cfg.valueHidden, 1, "value.fc2")) o.values.push_back(static_cast<float>(std::tanh(v)));
    if (cfg.ownership) {
        std::vector<double> own = conv(h, F, 1, 1, "ownership.conv");
        norm(own, 1, "ownership.bn");
        for (double v : own) o.ownership.push_back(static_cast<float>(std::tanh(v)));
    }
    return o;
}

// The whole inference pass against Network::infer with BLAS (OpenBLAS here),
// within the Metal spec's ceiling of 1e-5 maxErr, on the Metal spec's networks.
void test_network_matches_cpu() {
    auto blas = makeBackend("blas");
    MaxErr err("network forward vs CPU", 1e-5);
    const NetConfig small[] = {
        netConfig(5, 1, 8), netConfig(9, 2, 16), netConfig(9, 6, 32), netConfig(9, 0, 16),
        netConfig(9, 2, 16, true), netConfig(19, 2, 16),
    };
    Rng rng(601);
    for (const NetConfig& cfg : small) {
        const Network net = randomNetwork(cfg, 40);
        cuda_probe::InferNet gpu(net, 64);
        for (int B : {1, 7, 64}) {
            const std::vector<float> x = randomPositions(rng, cfg, B);
            const Outputs ref = cpuInfer(net, *blas, x, B);
            const Outputs got = gpuInfer(gpu, cfg, x, B);
            err.add(got.logits.data(), ref.logits.data(), got.logits.size());
            err.add(got.values.data(), ref.values.data(), got.values.size());
            if (cfg.ownership) err.add(got.ownership.data(), ref.ownership.data(), got.ownership.size());
        }
    }
}

// How far one network's outputs are from a reference: the logits as
// maxErr (max|x - ref| / max|ref|), the value and ownership, which are tanh
// outputs in [-1, 1], as the largest absolute difference. (A value near 0,
// where fc2's bias nearly cancels its product, would make a relative
// measure meaningless.)
struct OutputsGap {
    double logits = 0.0, values = 0.0, ownership = 0.0;
    double worst() const { return std::max(logits, std::max(values, ownership)); }
};

OutputsGap outputsGap(const Outputs& x, const Outputs& ref) {
    OutputsGap g;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < x.logits.size(); ++i) {
        assert(std::isfinite(x.logits[i]));
        num = std::max(num, std::fabs(static_cast<double>(x.logits[i]) - ref.logits[i]));
        den = std::max(den, std::fabs(static_cast<double>(ref.logits[i])));
    }
    g.logits = den > 0.0 ? num / den : num;
    for (size_t i = 0; i < x.values.size(); ++i)
        g.values = std::max(g.values, std::fabs(static_cast<double>(x.values[i]) - ref.values[i]));
    for (size_t i = 0; i < x.ownership.size(); ++i)
        g.ownership = std::max(g.ownership, std::fabs(static_cast<double>(x.ownership[i]) - ref.ownership[i]));
    return g;
}

// The presets' shapes (5x5 3x32, 9x9 6x64 and 6x96, 13x13 6x96, 19x19 10x128
// with value_hidden 256) and one with the ownership head, with random
// weights. Through their depth these gather more float noise than the
// Metal spec's 1e-5 allows even between the CPU's own backends, so they are
// measured against the network in double precision (OutputsGap): for each
// output the GPU must be nearly as close to it as the CPU's backends are
// (within twice their worst gap, or 1e-5), and never further than 5e-5.
void test_network_presets_vs_double() {
    auto blas = makeBackend("blas");
    auto naive = makeBackend("naive");
    const NetConfig presets[] = {netConfig(5, 3, 32, false, 64),    netConfig(9, 6, 64, false, 64),
                                 netConfig(9, 6, 96, false, 64),    netConfig(13, 6, 96, false, 64),
                                 netConfig(19, 10, 128, false, 256), netConfig(19, 2, 128, true, 256)};
    Rng rng(611);
    for (const NetConfig& cfg : presets) {
        const Network net = randomNetwork(cfg, 40);
        cuda_probe::InferNet gpu(net, 16);
        OutputsGap gpuWorst, cpuWorst;
        for (int B : cfg.boardSize == 19 ? std::vector<int>{1, 4} : std::vector<int>{1, 7, 16}) {
            const std::vector<float> x = randomPositions(rng, cfg, B);
            const Outputs ref = doubleInfer(net, x, B);
            const OutputsGap a = outputsGap(cpuInfer(net, *blas, x, B), ref);
            const OutputsGap b = outputsGap(cpuInfer(net, *naive, x, B), ref);
            const OutputsGap g = outputsGap(gpuInfer(gpu, cfg, x, B), ref);
            const double pairs[3][2] = {{g.logits, std::max(a.logits, b.logits)},
                                        {g.values, std::max(a.values, b.values)},
                                        {g.ownership, std::max(a.ownership, b.ownership)}};
            for (const auto& p : pairs)
                if (!(p[0] <= std::min(5e-5, std::max(1e-5, 2.0 * p[1])))) {
                    std::printf("test_cuda: %dx%d %dx%d batch %d: GPU gap %.3g from double, the CPU's %.3g\n",
                                cfg.boardSize, cfg.boardSize, cfg.blocks, cfg.filters, B, p[0], p[1]);
                    std::fflush(stdout);
                    assert(false);
                }
            gpuWorst.logits = std::max(gpuWorst.logits, g.logits);
            gpuWorst.values = std::max(gpuWorst.values, g.values);
            gpuWorst.ownership = std::max(gpuWorst.ownership, g.ownership);
            cpuWorst.logits = std::max(cpuWorst.logits, std::max(a.logits, b.logits));
            cpuWorst.values = std::max(cpuWorst.values, std::max(a.values, b.values));
            cpuWorst.ownership = std::max(cpuWorst.ownership, std::max(a.ownership, b.ownership));
        }
        std::printf("test_cuda:   %2dx%-2d %2dx%-3d vs double: logits %.1e (CPU %.1e), values %.1e (CPU %.1e)%s\n",
                    cfg.boardSize, cfg.boardSize, cfg.blocks, cfg.filters, gpuWorst.logits, cpuWorst.logits,
                    gpuWorst.values, cpuWorst.values, cfg.ownership ? ", ownership checked" : "");
    }
}

// What the inference hub rests on, for the whole network: a position's
// logits, value and ownership are the same bits alone, in batches of 8, 144
// and 256, at the first or the last slot, among random companions and with
// the batch permuted.
void test_network_batch_independent() {
    const NetConfig cfg = netConfig(13, 6, 96, true, 64);
    const Network net = randomNetwork(cfg, 602);
    cuda_probe::InferNet gpu(net, 256);
    Rng rng(603);
    const int full = 256;
    const size_t planeFloats = static_cast<size_t>(cfg.planes()) * cfg.area();
    const std::vector<float> pool = randomPositions(rng, cfg, full);
    auto batchOf = [&](const std::vector<int>& picks) {
        std::vector<float> x(picks.size() * planeFloats);
        for (size_t i = 0; i < picks.size(); ++i)
            std::copy_n(pool.data() + picks[i] * planeFloats, planeFloats, x.data() + i * planeFloats);
        return x;
    };
    auto run = [&](const std::vector<int>& picks) {
        return gpuInfer(gpu, cfg, batchOf(picks), static_cast<int>(picks.size()));
    };
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
        const int where = static_cast<int>(std::find(permuted.begin(), permuted.end(), s) - permuted.begin());
        assert(sameAt(shuffled, where, alone));
        for (int size : {8, 144, 256}) {
            for (int slot : {0, size - 1}) {
                std::vector<int> picks(size);
                for (int& p : picks) p = rng.below(full);
                picks[slot] = s;
                assert(sameAt(run(picks), slot, alone));
            }
        }
    }
}

void test_network_repeatable() {
    const NetConfig cfg = netConfig(9, 2, 16, true);
    const Network net = randomNetwork(cfg, 604);
    cuda_probe::InferNet gpu(net, 32);
    Rng rng(605);
    const std::vector<float> x = randomPositions(rng, cfg, 32);
    const Outputs first = gpuInfer(gpu, cfg, x, 32);
    for (int r = 0; r < 20; ++r) {
        const Outputs again = gpuInfer(gpu, cfg, x, 32);
        assert(same(again.logits, first.logits) && same(again.values, first.values) &&
               same(again.ownership, first.ownership));
    }
    // A second copy of the network gives the same bits as the first.
    cuda_probe::InferNet other(net, 32);
    const Outputs second = gpuInfer(other, cfg, x, 32);
    assert(same(second.logits, first.logits) && same(second.values, first.values));
}

// A tensor the GPU layout does not know (a layer added on the CPU without a
// GPU port) is refused at upload, by name.
void test_unknown_tensor_refused() {
    const Network plain(netConfig(9, 1, 8), 1), withHead(netConfig(9, 1, 8, true), 1);
    cuda_probe::checkUpload(plain, plain);
    bool threw = false;
    try {
        cuda_probe::checkUpload(plain, withHead);
    } catch (const std::logic_error& e) {
        threw = std::string(e.what()).find("ownership.conv.w") != std::string::npos;
    }
    assert(threw);
}

// And one the layout knows but the GPU pass never reads (a GPU port that
// lags behind the CPU network) is refused when the pass is first encoded.
void test_unread_tensor_refused() {
    const NetConfig withHead = netConfig(9, 1, 8, true), plain = netConfig(9, 1, 8);
    const Network net(withHead, 1);
    cuda_probe::checkForwardReadsEveryTensor(net, withHead);
    bool threw = false;
    try {
        cuda_probe::checkForwardReadsEveryTensor(net, plain);
    } catch (const std::logic_error& e) {
        const std::string what = e.what();
        threw = what.find("ownership.conv.w") != std::string::npos &&
                what.find("ownership.bn.runVar") != std::string::npos;
    }
    assert(threw);
}

// The inference workspace holds no column matrix: 13x13 6x96 at batch 256 is
// the input twice, three trunk buffers and the heads, ~60 MB.
void test_workspace_size_and_speed() {
    const NetConfig cfg = netConfig(13, 6, 96, false, 64);
    const Network net = randomNetwork(cfg, 606);
    cuda_probe::InferNet gpu(net, 256);
    const double mb = static_cast<double>(gpu.bytes()) / 1e6;
    std::printf("test_cuda: 13x13 6x96 inference workspace at batch 256: %.1f MB\n", mb);
    assert(mb > 40.0 && mb < 90.0);
    Rng rng(607);
    const std::vector<float> x = randomPositions(rng, cfg, 256);
    gpuInfer(gpu, cfg, x, 256);
    const auto start = std::chrono::steady_clock::now();
    int passes = 0;
    while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(500)) {
        gpuInfer(gpu, cfg, x, 256);
        ++passes;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("test_cuda: 13x13 6x96 synchronous passes of 256: %.0f evals/s (not asserted)\n", passes * 256 / s);
}

// ---------------------------------------------------------------- the runner and the hub

// Positions evaluated one at a time on the GPU: the reference for the runner
// and the hub, which must give each position exactly these bits.
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
        cuda_probe::InferNet alone(net, 1);
        for (int i = 0; i < count; ++i)
            alone.infer(input.data() + i * in(), 1, logits.data() + i * cfg.actions(), values.data() + i);
    }
};

// One pass through a runner's slot, waited for.
std::string runSlot(BatchRunner& runner, int slot, int B) {
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    std::string error;
    runner.begin(slot, B, [&](int s, std::string e) {
        assert(s == slot);
        std::lock_guard<std::mutex> lock(m);
        error = e;
        done = true;
        cv.notify_one();
    });
    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [&] { return done; });
    return error;
}

// Threads of this process (the /proc/self/task entries).
int threadCount() {
    int count = 0;
    if (DIR* d = opendir("/proc/self/task")) {
        while (const dirent* e = readdir(d))
            if (e->d_name[0] != '.') ++count;
        closedir(d);
    }
    return count;
}

// The runner's slots, one or two, at every batch size up to its maximum:
// the outputs equal each position alone, bitwise; both slots in flight at once.
void test_runner_slots() {
    const NetConfig cfg = netConfig(9, 2, 16, true);
    const Network net = randomNetwork(cfg, 701);
    const GpuPool pool(net, 40, 702);
    const size_t in = pool.in(), L = cfg.actions();
    for (int slots : {1, 2}) {
        auto runner = makeCudaRunner(net, 40, slots);
        assert(runner->slots() == slots && runner->maxBatch() == 40);
        for (int B : {1, 7, 40}) {
            for (int slot = 0; slot < slots; ++slot) {
                std::copy_n(pool.input.data(), B * in, runner->input(slot));
                assert(runSlot(*runner, slot, B).empty());
                assert(bitwiseEqual(runner->logits(slot), pool.logits.data(), B * L));
                assert(bitwiseEqual(runner->values(slot), pool.values.data(), B));
            }
        }
    }
    // Both slots at once, different positions in each.
    auto runner = makeCudaRunner(net, 20, 2);
    std::copy_n(pool.input.data(), 20 * in, runner->input(0));
    std::copy_n(pool.input.data() + 20 * in, 20 * in, runner->input(1));
    std::mutex m;
    std::condition_variable cv;
    int finished = 0;
    for (int slot = 0; slot < 2; ++slot)
        runner->begin(slot, 20, [&](int, std::string e) {
            assert(e.empty());
            std::lock_guard<std::mutex> lock(m);
            ++finished;
            cv.notify_one();
        });
    {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&] { return finished == 2; });
    }
    assert(bitwiseEqual(runner->logits(0), pool.logits.data(), 20 * L));
    assert(bitwiseEqual(runner->logits(1), pool.logits.data() + 20 * L, 20 * L));
    assert(bitwiseEqual(runner->values(1), pool.values.data() + 20, 20));
    for (int bad : {0, kInferenceSlots + 1}) {
        bool refused = false;
        try {
            makeCudaRunner(net, 8, bad);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        assert(refused);
    }
}

// A pass is reported only once its outputs have reached the host. A test
// hook holds the GPU ~2 ms at the start of the copy-out, so a runner that
// recorded its completion event before the copies (or signalled from the
// pass alone) would report early, and the host would read the previous
// pass's outputs: each round uses other positions, so that shows.
void test_runner_reports_after_copy_out() {
    const NetConfig cfg = netConfig(9, 2, 16);
    const Network net = randomNetwork(cfg, 710);
    const GpuPool pool(net, 48, 711);
    const size_t in = pool.in(), L = cfg.actions();
    cuda_probe::setCopyOutDelay(2000);
    auto runner = makeCudaRunner(net, 8, 2);
    for (int round = 0; round < 6; ++round) {
        const int slot = round % 2, first = 8 * round;
        std::copy_n(pool.input.data() + first * in, 8 * in, runner->input(slot));
        assert(runSlot(*runner, slot, 8).empty());
        assert(bitwiseEqual(runner->logits(slot), pool.logits.data() + first * L, 8 * L));
        assert(bitwiseEqual(runner->values(slot), pool.values.data() + first, 8));
    }
    cuda_probe::setCopyOutDelay(0);
}

// test_inference_hub's chaos test with the CUDA runner: every request's
// outputs equal its positions evaluated alone on the GPU, bitwise, whatever
// batches the hub made of them.
void test_hub_outputs_equal_alone_under_chaos() {
    const NetConfig cfg = netConfig(9, 2, 16);
    const Network net = randomNetwork(cfg, 703);
    const GpuPool pool(net, 300, 704);
    const size_t in = pool.in(), L = static_cast<size_t>(cfg.actions());
    for (int clients : {1, 4, 64}) {
        for (int maxBatch : {8, 64, 512}) {
            for (uint64_t chaos : {0ull, 11ull, 12ull}) {
                for (int slots : {1, 2}) {
                    InferenceHubOptions o;
                    o.maxBatch = maxBatch;
                    o.chaosSeed = chaos;
                    InferenceHub hub(makeCudaRunner(net, maxBatch, slots), cfg, o);
                    std::atomic<long long> positions{0};
                    std::vector<std::thread> threads;
                    for (int t = 0; t < clients; ++t) {
                        threads.emplace_back([&, t] {
                            auto ev = hub.makeEvaluator(8);
                            Rng rng(deriveSeed(705, static_cast<uint64_t>(t), chaos, static_cast<uint64_t>(maxBatch)));
                            std::vector<float> x(8 * in), logits(8 * L), values(8);
                            for (int r = 0; r < 50; ++r) {
                                const int B = 1 + rng.below(8);
                                int picks[8];
                                for (int b = 0; b < B; ++b) {
                                    picks[b] = rng.below(pool.count);
                                    std::copy_n(pool.input.data() + picks[b] * in, in, x.data() + b * in);
                                }
                                ev->evaluate(x.data(), B, logits.data(), values.data());
                                for (int b = 0; b < B; ++b) {
                                    assert(bitwiseEqual(logits.data() + b * L, pool.logits.data() + picks[b] * L, L));
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
}

// A failed pass (as a GPU fault would report it) reaches its own requests
// and every later one as "GPU inference failed: cuda: ...", and the hub
// still shuts down cleanly.
void test_hub_failure_reaches_everyone() {
    const NetConfig cfg = netConfig(9, 1, 8);
    const Network net = randomNetwork(cfg, 706);
    const size_t in = static_cast<size_t>(cfg.planes()) * cfg.area(), L = cfg.actions();
    const long long before = cuda_probe::passesCompleted();
    cuda_probe::failPassesAfter(3);
    std::atomic<int> failures{0}, successes{0};
    {
        InferenceHubOptions o;
        o.maxBatch = 8;
        InferenceHub hub(makeCudaRunner(net, 8), cfg, o);
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t)
            threads.emplace_back([&] {
                auto ev = hub.makeEvaluator(1);
                std::vector<float> x(in, 1.0f), logits(L), values(1);
                for (int r = 0; r < 20; ++r) {
                    try {
                        ev->evaluate(x.data(), 1, logits.data(), values.data());
                        ++successes;
                    } catch (const std::runtime_error& e) {
                        assert(std::string(e.what()).rfind("GPU inference failed: cuda: ", 0) == 0);
                        ++failures;
                        return;   // the hub has failed for good
                    }
                }
            });
        for (auto& th : threads) th.join();
    }
    cuda_probe::failPassesAfter(0);
    assert(failures.load() == 8);   // every thread saw the failure
    assert(successes.load() >= 3 && cuda_probe::passesCompleted() > before);
    // The next runner is healthy again.
    auto runner = makeCudaRunner(net, 1);
    std::fill_n(runner->input(0), in, 1.0f);
    assert(runSlot(*runner, 0, 1).empty());
}

// A runner whose slots would not fit the GPU's free memory is refused before
// anything is allocated, with the size and the remedy; so is the same check
// made up front.
void test_memory_check() {
    const NetConfig cfg = netConfig(19, 10, 128, false, 256);
    const Network net(cfg, 1);
    bool threw = false;
    try {
        makeCudaRunner(net, 1 << 20);
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        std::printf("test_cuda: %s\n", what.c_str());
        threw = what.find("lower inference_batch") != std::string::npos && what.find("MB") != std::string::npos;
    }
    assert(threw);
    cuda_probe::setMemoryLimit(size_t(10) << 20);   // 10 MB
    threw = false;
    try {
        checkCudaInferenceMemory(cfg, 64, 1);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("lower inference_batch") != std::string::npos;
    }
    assert(threw);
    // Groups held at once (pipeline = on) are added up.
    threw = false;
    try {
        checkCudaInferenceMemory(cfg, {GpuHubs{1, 2, 64}, GpuHubs{4, 1, 64}});
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("lower inference_batch") != std::string::npos;
    }
    assert(threw);
    cuda_probe::setMemoryLimit(0);
    checkCudaInferenceMemory(cfg, 64, 1);   // fits the real GPU
    checkCudaInferenceMemory(cfg, {GpuHubs{1, 2, 64}, GpuHubs{4, 1, 64}});
    auto ok = makeCudaRunner(net, 8);
    assert(ok->slots() == kInferenceSlots && ok->maxBatch() == 8);
}

// No thread outlives a runner, including runners destroyed with passes in flight.
void test_runner_no_thread_leak() {
    const NetConfig cfg = netConfig(9, 1, 8);
    const Network net = randomNetwork(cfg, 707);
    { auto warm = makeCudaRunner(net, 8); }   // the CUDA runtime's own threads start once
    const int before = threadCount();
    for (int i = 0; i < 50; ++i) {
        auto runner = makeCudaRunner(net, 8);
        std::fill_n(runner->input(0), 8 * cfg.planes() * cfg.area(), 0.5f);
        if (i % 2) runner->begin(0, 8, [](int, std::string) {});   // destroyed in flight
    }
    int after = threadCount();
    for (int tries = 0; after > before && tries < 100; ++tries) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        after = threadCount();
    }
    if (after > before) std::printf("test_cuda: %d threads before, %d after\n", before, after);
    assert(after <= before);
}

// WorkerEvaluators with backend cuda: one hub on the GPU for all workers,
// each worker's rows its positions alone; evaluations() counts positions.
void test_worker_evaluators_cuda() {
    const NetConfig cfg = netConfig(9, 2, 16);
    const Network net = randomNetwork(cfg, 708);
    const GpuPool pool(net, 64, 709);
    const size_t in = pool.in(), L = cfg.actions();
    WorkerEvaluators evaluators(net, "cuda", 8, 8, 256);
    assert(evaluators.hub() != nullptr && evaluators.workers() == 8);
    std::vector<std::thread> threads;
    for (int w = 0; w < 8; ++w)
        threads.emplace_back([&, w] {
            Evaluator& ev = evaluators.forWorker(w);
            std::vector<float> logits(8 * L), values(8);
            ev.evaluate(pool.input.data() + w * 8 * in, 8, logits.data(), values.data());
            assert(bitwiseEqual(logits.data(), pool.logits.data() + w * 8 * L, 8 * L));
            assert(bitwiseEqual(values.data(), pool.values.data() + w * 8, 8));
        });
    for (auto& th : threads) th.join();
    assert(evaluators.evaluations() == 64);
}

// ---------------------------------------------------------------- self-play with backend = cuda

// A slice's games, serialized: what a slice's result file carries.
std::string recordBytes(const std::vector<GameRecord>& games) {
    std::ostringstream out;
    BinaryWriter w(out);
    for (const GameRecord& g : games) writeGameRecord(w, g);
    return out.str();
}

// The determinism contract for a cuda engine's slice: the same games, byte
// for byte, for 1, 3 and 8 workers, batches capped at 8 or 256, and the
// hub's chaos mode on.
void test_selfplay_slice_independent_of_workers_and_batching() {
    for (const NetConfig& cfg : {netConfig(5, 1, 8), netConfig(9, 2, 16)}) {
        const Network net = randomNetwork(cfg, 801);
        SelfPlayConfig sp;
        sp.mcts.simulations = 32;
        sp.mcts.batchSize = 8;
        sp.temperatureMoves = 6;
        std::string reference;
        for (int workers : {1, 3, 8})
            for (int batch : {8, 256})
                for (uint64_t chaos : {0ull, 21ull}) {
                    cuda_probe::setHubChaosSeed(chaos);
                    EngineRun run;
                    run.backend = "cuda";
                    run.workers = workers;
                    run.inferenceBatch = batch;
                    SliceStats stats;
                    const std::string bytes =
                        recordBytes(playSelfPlayGames(net, sp, 802, 3, 10, 6, run, &stats));
                    cuda_probe::setHubChaosSeed(0);
                    assert(stats.evaluations > 0);
                    if (reference.empty()) reference = bytes;
                    assert(bytes == reference);
                }
    }
}

// An arena wave on a cuda engine (arena_engines = pc/cuda:...): one hub
// per network, several networks sharing the GPU, every game at once. Its
// games are the same, move for move, for any workers, any batch cap and
// the hub's chaos mode, as on metal; a slice of the wave plays what the
// whole wave plays in those games.
void test_arena_wave_independent_of_workers_and_batching() {
    const NetConfig cfg = netConfig(5, 1, 8);
    const Network a = randomNetwork(cfg, 811), b = randomNetwork(cfg, 812), c = randomNetwork(cfg, 813);
    ArenaConfig ac;
    ac.simulations = 16;
    ac.mctsBatch = 4;
    ac.temperatureMoves = 2;
    const std::vector<ArenaMatchSpec> wave = {
        {ArenaPlayer::network(a), ArenaPlayer::network(b), 6, 901},
        {ArenaPlayer::network(a), ArenaPlayer::bot(8), 4, 902},
        {ArenaPlayer::network(a), ArenaPlayer::network(c), 4, 903},
    };
    auto bytes = [](const std::vector<ArenaGameResult>& games) {
        std::string out;
        for (const ArenaGameResult& g : games) {
            out += std::to_string(g.scoreA) + ":";
            for (int m : g.moves) out += std::to_string(m) + ",";
            out += ";";
        }
        return out;
    };
    std::string reference;
    for (int workers : {1, 3, 0})
        for (int batch : {8, 256})
            for (uint64_t chaos : {0ull, 23ull}) {
                cuda_probe::setHubChaosSeed(chaos);
                EngineRun run;
                run.backend = "cuda";
                run.workers = workers;
                run.inferenceBatch = batch;
                SliceStats stats;
                const std::string played = bytes(playArenaGames(waveSlice(wave, 0, 14), ac, run, &stats));
                cuda_probe::setHubChaosSeed(0);
                assert(stats.evaluations > 0);
                if (reference.empty()) reference = played;
                assert(played == reference);
            }
    EngineRun run;
    run.backend = "cuda";
    std::string joined = bytes(playArenaGames(waveSlice(wave, 0, 5), ac, run));
    joined += bytes(playArenaGames(waveSlice(wave, 5, 9), ac, run));
    assert(joined == reference);
}

namespace fs = std::filesystem;

struct RunDir {
    fs::path path;
    explicit RunDir(const std::string& name)
        : path(fs::temp_directory_path() / ("azgo_test_cuda_" + std::to_string(getpid()) + "_" + name)) {
        fs::remove_all(path);
        fs::create_directories(path);
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

// A 5x5 run small enough that a generation takes a fraction of a second.
// Its own backend is naive (a run's backend cannot be cuda); self-play goes
// to the cuda engine.
Config cudaRunConfig() {
    Config c;
    c.backend = "naive";
    c.boardSize = 5;
    c.history = 2;
    c.blocks = 1;
    c.filters = 8;
    c.valueHidden = 8;
    c.gamesPerGeneration = 6;
    c.simulations = 8;
    c.mctsBatch = 4;
    c.temperatureMoves = 4;
    c.bufferPositions = 1000;
    c.trainSteps = 4;
    c.batchSize = 16;
    c.arenaGames = 2;
    c.arenaSimulations = 4;
    c.workers = 2;
    c.sgfPerGeneration = 1;
    applyOverride(c, "selfplay_engines=local/cuda:1");
    return c;
}

// What a run produced that must not depend on workers, batching or timing:
// every checkpoint, state.bin and the rating files.
std::string runFiles(const fs::path& run, int generations) {
    std::string out;
    for (int g = 0; g <= generations; ++g) out += readBytes(checkpointPath(run.string(), g));
    for (const char* f : {"state.bin", "results.csv", "elo.csv"}) out += readBytes((run / f).string());
    return out;
}

// A run whose self-play goes to the local cuda engine: the same files for
// any cuda_workers and inference_batch in the hosts file, whether it ran
// two generations straight or one and a resume. The manifest names the
// engine's GPU.
void test_cuda_engine_run_independent_of_settings() {
    std::string reference;
    {
        RunDir dir("straight");
        const fs::path run = dir.path / "run";
        std::unique_ptr<Trainer> t = Trainer::startNew(cudaRunConfig(), run.string(), {}, nullptr);
        t->runUntil(2);
        reference = runFiles(run, 2);
    }
    for (const char* settings : {"cuda_workers=0", "cuda_workers=3 inference_batch=8",
                                 "cuda_workers=64 inference_batch=256"}) {
        RunDir dir("engine");
        const std::string hosts = dir.file("hosts");
        {
            std::ofstream out(hosts);
            out << "local " << settings << "\n";
        }
        TrainerOptions opts;
        opts.hostsFile = hosts;
        const fs::path run = dir.path / "run";
        std::ostringstream startup;
        {
            std::unique_ptr<Trainer> t = Trainer::startNew(cudaRunConfig(), run.string(), {}, &startup, opts);
            t->setLog(nullptr);
            t->runUntil(1);
        }
        assert(startup.str().find("engine local/cuda: ") != std::string::npos);
        {
            std::unique_ptr<Trainer> t = Trainer::resume(run.string(), {}, nullptr, opts);
            t->runUntil(2);
        }
        const std::string manifest = readBytes((run / "selfplay-engines.csv").string());
        assert(manifest.find(",local/cuda,local/cuda,\"cuda; ") != std::string::npos);
        assert(manifest.find(cudaDescription()) != std::string::npos);
        assert(runFiles(run, 2) == reference);
    }
}

// A run is refused before it writes anything when its cuda engine's hub
// would not fit the GPU's memory.
void test_cuda_engine_memory_checked_up_front() {
    RunDir dir("memory");
    const fs::path run = dir.path / "run";
    cuda_probe::setMemoryLimit(size_t(1) << 20);   // 1 MB
    bool refused = false;
    std::ostringstream out;
    try {
        Trainer::startNew(cudaRunConfig(), run.string(), {}, &out);
    } catch (const std::runtime_error& e) {
        refused = std::string(e.what()).find("lower inference_batch") != std::string::npos;
    }
    cuda_probe::setMemoryLimit(0);
    assert(refused);
    assert(!fs::exists(configPath(run.string())) && !fs::exists(statePath(run.string())));
}

} // namespace

#endif // AZ_USE_CUDA

int main() {
    std::string why;
    if (!cudaBuilt()) {
        // A build without CUDA must say how to get it, and refuse the backend.
        assert(!cudaAvailable(&why));
        assert(why.find("-DAZ_USE_CUDA=ON") != std::string::npos);
        bool refused = false;
        try {
            checkBackendAvailable("cuda");
        } catch (const std::invalid_argument& e) {
            refused = std::string(e.what()) ==
                      "backend 'cuda' not available: rebuild with -DAZ_USE_CUDA=ON (Linux, NVIDIA GPU)";
        }
        assert(refused);
        std::printf("test_cuda: SKIPPED (%s)\n", why.c_str());
        return 77;
    }
    if (!cudaAvailable(&why)) {
        // No GPU (or no driver) here: nothing to test. But a GPU the build's
        // machine code was not made for is a build that cannot run here:
        // that fails, so ctest does not report it as a harmless skip.
        if (why.rfind("no CUDA device", 0) == 0) {
            std::printf("test_cuda: SKIPPED (%s)\n", why.c_str());
            return 77;
        }
        std::printf("test_cuda: FAILED: a GPU is present but unusable: %s\n", why.c_str());
        return 1;
    }
#ifdef AZ_USE_CUDA
    checkBackendAvailable("cuda");   // usable here
    test_available_and_description();
    test_architecture_matches();
    test_fill();
    test_gemm_equals_fma_chain();
    test_gemm_tiles_agree();
    test_gemm_batch_independent();
    test_gemm_ignores_c();
    test_gemm_keeps_negative_zero();
    test_copy_kernels_bitwise();
    test_bn_infer_bitwise();
    test_relu_add_bias_bitwise();
    test_tanh_within_two_ulp();
    test_elementwise_batch_independent();
    test_conv_fused_equals_explicit();
    test_conv_fused_batch_independent();
    test_network_matches_cpu();
    test_network_presets_vs_double();
    test_network_batch_independent();
    test_network_repeatable();
    test_unknown_tensor_refused();
    test_unread_tensor_refused();
    test_workspace_size_and_speed();
    test_runner_slots();
    test_runner_reports_after_copy_out();
    test_hub_outputs_equal_alone_under_chaos();
    test_hub_failure_reaches_everyone();
    test_memory_check();
    test_runner_no_thread_leak();
    test_worker_evaluators_cuda();
    test_selfplay_slice_independent_of_workers_and_batching();
    test_arena_wave_independent_of_workers_and_batching();
    test_cuda_engine_run_independent_of_settings();
    test_cuda_engine_memory_checked_up_front();
#endif
    std::printf("test_cuda: all tests passed\n");
    return 0;
}
