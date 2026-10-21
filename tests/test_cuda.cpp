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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <string>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/cuda.hpp"
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
#endif
    std::printf("test_cuda: all tests passed\n");
    return 0;
}
