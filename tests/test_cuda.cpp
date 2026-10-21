// GPU tests for backend = cuda: the context and its architecture check, every
// kernel against an exact CPU reference, and the batch-independence
// properties the inference hub relies on.
//
// They run only in a build configured with -DAZ_USE_CUDA=ON on a machine with
// a usable NVIDIA GPU. Anywhere else main() returns 77, which CMake declares
// as this test's SKIP_RETURN_CODE, so CTest reports "Skipped" rather than a
// failure.
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
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <string>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/cuda.hpp"
#include "azgo/layers.hpp"
#include "azgo/rng.hpp"

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

// ---------------------------------------------------------------- the context

void test_available_and_description() {
    std::string why = "unchanged";
    assert(cudaAvailable(&why));
    assert(why == "unchanged");   // only written when unusable
    const std::string d = cudaDescription();
    std::printf("test_cuda: %s\n", d.c_str());
    static const std::regex shape("^.+ sm_[0-9]+, driver [0-9]+\\.[0-9]+, runtime [0-9]+\\.[0-9]+, kernels [0-9a-f]{8}$");
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
        std::printf("test_cuda: SKIPPED (%s)\n", why.c_str());
        return 77;
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
    test_copy_kernels_bitwise();
    test_bn_infer_bitwise();
    test_relu_add_bias_bitwise();
    test_tanh_within_two_ulp();
    test_elementwise_batch_independent();
#endif
    std::printf("test_cuda: all tests passed\n");
    return 0;
}
