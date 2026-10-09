// metal_bench: should azgo grow a Metal backend? A measurement spike, not
// part of the library: it links the unchanged libazgo (built with BLAS) and
// compares, for the 9x9 6x64 / 6x96 networks and the 19x19 10x128 one,
//
//   cpu1    the project's own BlasBackend (Accelerate, one thread, as every
//           worker and the trainer use it): gemm per shape, the gemm
//           sequence of a whole forward/training pass, Network::infer and
//           Network::trainStep, and 18 workers inferring at batch 8 at once
//   cpuall  the same with Accelerate left free to use all its threads
//   gpu     two hand-written tiled Metal gemms (scalar fma; 8x8 simdgroup
//           matrices) and MPSMatrixMultiplication:
//           per call (commit + wait each time) and a whole forward pass or
//           training step encoded into one command buffer (gemm, im2col,
//           col2im, BN/ReLU and BN-statistics kernels)
//   seam    a MetalBackend behind today's Backend seam (gemm only, copies in
//           and out, one commit + wait per gemm) driving the real Network
//   det     determinism and accuracy of the hand kernels (and of MPS and
//           Accelerate for comparison)
//
// One section per process (cpuall must not inherit the single-thread setting
// that makeBackend("blas") installs). See README.md next to this file.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "azgo/backend.hpp"
#include "azgo/network.hpp"
#include "azgo/rng.hpp"

namespace {

using Clock = std::chrono::steady_clock;
double secondsSince(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

double gSeconds = 0.5;   // minimum wall time per measurement (--seconds)
std::string gNetFilter;  // --net: only networks whose name contains this

// ---------------------------------------------------------------- networks

struct NetSpec { const char* name; int n, blocks, filters, valueHidden; };
const NetSpec kNets[] = {
    {"9x9 6x64", 9, 6, 64, 64},
    {"9x9 6x96", 9, 6, 96, 64},
    {"19x19 10x128", 19, 10, 128, 256},
};
const int kBatches[] = {8, 144, 256};
bool skipNet(const NetSpec& s) {
    return !gNetFilter.empty() && std::string(s.name).find(gNetFilter) == std::string::npos;
}
constexpr int kHistory = 8;
constexpr int kPlanes = 2 * kHistory + 2;   // 18 input planes
constexpr int kTrainBatch = 256;

azgo::NetConfig netConfig(const NetSpec& s) {
    azgo::NetConfig c;
    c.boardSize = s.n;
    c.history = kHistory;
    c.blocks = s.blocks;
    c.filters = s.filters;
    c.valueHidden = s.valueHidden;
    return c;
}

struct Gemm {
    std::string label;
    bool tA = false, tB = false;
    int M = 0, N = 0, K = 0;
    float beta = 0.0f;
    double flops() const { return 2.0 * M * N * K; }
};

// The conv gemms of Conv2d::forward/backward for the trunk (see src/layers.cpp).
std::vector<Gemm> convShapes(const NetSpec& s, int B) {
    const int F = s.filters, N = B * s.n * s.n;
    std::vector<Gemm> v;
    v.push_back({"fwd stem 3x3", false, false, F, N, kPlanes * 9, 0});
    v.push_back({"fwd block 3x3", false, false, F, N, F * 9, 0});
    if (B == kTrainBatch) {
        v.push_back({"bwd dW block", false, true, F, F * 9, N, 1});
        v.push_back({"bwd dX block", true, false, F * 9, N, F, 0});
    }
    return v;
}

// ---------------------------------------------------------------- pass plans

// A forward pass or training step as the sequence of kernels a whole-pass
// GPU design would run, mirroring Network::forward/backward op for op. The
// gemms are exact; BN, ReLU, the skip add, flatten and the loss are stand-in
// elementwise/reduction kernels of the right sizes.
enum Buf { bIn, bX, bY, bS, bCol, bDcol, bW, bGW, bH1, bH2, bScale, bShift, bStats, kBufs };

struct Op {
    enum Kind { kGemm, kIm2col, kCol2im, kBnRelu, kEw, kStats } kind = kGemm;
    Gemm g;
    Buf a = bX, b = bY, c = bS;
    int C = 0, B = 0, n = 0;            // im2col/col2im
    uint32_t count = 0, perChannel = 1;  // elementwise/stats
    bool skip = false, relu = true;
};

struct Plan {
    std::vector<Op> forward, backward;   // backward empty for inference
    size_t size[kBufs] = {};
    double gemmFlops() const {
        double f = 0;
        for (const auto* ops : {&forward, &backward})
            for (const Op& o : *ops) if (o.kind == Op::kGemm) f += o.g.flops();
        return f;
    }
    int count(Op::Kind k) const {
        int c = 0;
        for (const auto* ops : {&forward, &backward})
            for (const Op& o : *ops) c += o.kind == k;
        return c;
    }
};

Plan makePlan(const NetSpec& s, int B, bool train) {
    const int n = s.n, A = n * n, F = s.filters, V = s.valueHidden, P = kPlanes;
    const uint32_t M = static_cast<uint32_t>(B * A);
    const uint32_t trunk = static_cast<uint32_t>(F) * M;
    Plan p;
    auto& f = p.forward;
    auto gemm = [](std::vector<Op>& v, std::string label, bool tA, bool tB, int m, int nn, int k,
                   float beta, Buf a, Buf b, Buf c) {
        Op o; o.kind = Op::kGemm;
        o.g = {std::move(label), tA, tB, m, nn, k, beta};
        o.a = a; o.b = b; o.c = c;
        v.push_back(o);
    };
    auto im2col = [&](std::vector<Op>& v, int C, Buf in, Buf col) {
        Op o; o.kind = Op::kIm2col; o.C = C; o.B = B; o.n = n; o.a = in; o.c = col; v.push_back(o);
    };
    auto col2im = [&](std::vector<Op>& v, int C, Buf col, Buf out) {
        Op o; o.kind = Op::kCol2im; o.C = C; o.B = B; o.n = n; o.a = col; o.c = out; v.push_back(o);
    };
    auto bn = [&](std::vector<Op>& v, int C, Buf x, Buf y, bool skip, bool relu) {
        if (train) {   // batch statistics: one reduction per channel first
            Op st; st.kind = Op::kStats; st.count = C; st.perChannel = M; st.a = x; st.c = bStats; v.push_back(st);
        }
        Op o; o.kind = Op::kBnRelu; o.count = static_cast<uint32_t>(C) * M; o.perChannel = M;
        o.a = x; o.c = y; o.b = bS; o.skip = skip; o.relu = relu; v.push_back(o);
    };
    auto ew = [&](std::vector<Op>& v, uint32_t count, Buf x, Buf y, Buf out) {
        Op o; o.kind = Op::kEw; o.count = count; o.a = x; o.b = y; o.c = out; v.push_back(o);
    };
    auto stats = [&](std::vector<Op>& v, int C, Buf x) {
        Op o; o.kind = Op::kStats; o.count = C; o.perChannel = M; o.a = x; o.c = bStats; v.push_back(o);
    };

    // Forward (Network::forward).
    im2col(f, P, bIn, bCol);
    gemm(f, "stem", false, false, F, M, P * 9, 0, bW, bCol, bY);
    bn(f, F, bY, bX, false, true);
    for (int i = 0; i < s.blocks; ++i) {
        im2col(f, F, bX, bCol);
        gemm(f, "conv1", false, false, F, M, F * 9, 0, bW, bCol, bY);
        bn(f, F, bY, bS, false, true);
        im2col(f, F, bS, bCol);
        gemm(f, "conv2", false, false, F, M, F * 9, 0, bW, bCol, bY);
        bn(f, F, bY, bX, true, true);
    }
    gemm(f, "policy conv", false, false, 2, M, F, 0, bW, bX, bY);
    bn(f, 2, bY, bH1, false, true);
    ew(f, 2 * M, bH1, bH1, bH2);                                       // flatten
    gemm(f, "policy fc", false, true, B, A + 1, 2 * A, 0, bH2, bW, bH1);
    gemm(f, "value conv", false, false, 1, M, F, 0, bW, bX, bY);
    bn(f, 1, bY, bH1, false, true);
    gemm(f, "value fc1", false, true, B, V, A, 0, bH1, bW, bH2);
    ew(f, static_cast<uint32_t>(B * V), bH2, bH2, bH1);                // relu
    gemm(f, "value fc2", false, true, B, 1, V, 0, bH1, bW, bH2);
    ew(f, static_cast<uint32_t>(B), bH2, bH2, bH1);                    // tanh

    const uint32_t params = static_cast<uint32_t>(F * P * 9 + s.blocks * 2 * F * F * 9 + 2 * F +
                                                  2 * A * (A + 1) + F + A * V + V);
    if (train) {
        auto& b = p.backward;
        ew(b, static_cast<uint32_t>(B * (A + 1)), bH1, bH2, bH1);      // loss gradients
        // Value head.
        gemm(b, "value fc2 dW", true, false, 1, V, B, 1, bH1, bH2, bGW);
        gemm(b, "value fc2 dX", false, false, B, V, 1, 0, bH1, bW, bH2);
        ew(b, static_cast<uint32_t>(B * V), bH1, bH2, bH2);
        gemm(b, "value fc1 dW", true, false, V, A, B, 1, bH2, bH1, bGW);
        gemm(b, "value fc1 dX", false, false, B, A, V, 0, bH2, bW, bH1);
        ew(b, M, bH1, bH2, bH1);
        stats(b, 1, bH1);
        ew(b, M, bH1, bH2, bH1);
        gemm(b, "value conv dW", false, true, 1, F, M, 1, bH1, bX, bGW);
        gemm(b, "value conv dX", true, false, F, M, 1, 0, bW, bH1, bS);
        // Policy head.
        gemm(b, "policy fc dW", true, false, A + 1, 2 * A, B, 1, bH1, bH2, bGW);
        gemm(b, "policy fc dX", false, false, B, 2 * A, A + 1, 0, bH1, bW, bH2);
        ew(b, 2 * M, bH2, bH2, bH1);                                   // unflatten
        ew(b, 2 * M, bH1, bH2, bH1);                                   // relu backward
        stats(b, 2, bH1);
        ew(b, 2 * M, bH1, bH2, bH1);                                   // bn backward
        gemm(b, "policy conv dW", false, true, 2, F, M, 1, bH1, bX, bGW);
        gemm(b, "policy conv dX", true, false, F, M, 2, 0, bW, bH1, bY);
        ew(b, trunk, bS, bY, bS);                                      // dTrunk += dTmp
        // Residual blocks in reverse.
        for (int i = s.blocks - 1; i >= 0; --i) {
            ew(b, trunk, bX, bS, bS);                                  // relu backward
            stats(b, F, bS);
            ew(b, trunk, bS, bX, bY);                                  // bn2 backward
            im2col(b, F, bX, bCol);
            gemm(b, "conv2 dW", false, true, F, F * 9, M, 1, bY, bCol, bGW);
            gemm(b, "conv2 dX", true, false, F * 9, M, F, 0, bW, bY, bDcol);
            col2im(b, F, bDcol, bY);
            ew(b, trunk, bX, bY, bY);                                  // relu backward
            stats(b, F, bY);
            ew(b, trunk, bY, bX, bY);                                  // bn1 backward
            im2col(b, F, bX, bCol);
            gemm(b, "conv1 dW", false, true, F, F * 9, M, 1, bY, bCol, bGW);
            gemm(b, "conv1 dX", true, false, F * 9, M, F, 0, bW, bY, bDcol);
            col2im(b, F, bDcol, bY);
            ew(b, trunk, bY, bS, bS);                                  // dTrunk += dTmp
        }
        ew(b, trunk, bX, bS, bS);
        stats(b, F, bS);
        ew(b, trunk, bS, bX, bS);
        im2col(b, P, bIn, bCol);
        gemm(b, "stem dW", false, true, F, P * 9, M, 1, bS, bCol, bGW);
        // SGD with momentum over every parameter (a stand-in pass).
        ew(b, params, bW, bGW, bW);
    }

    // Buffer sizes in floats, generous enough for every op above.
    const size_t chans = static_cast<size_t>(std::max(F, P));
    const size_t big = chans * M;
    p.size[bIn] = static_cast<size_t>(P) * M;
    p.size[bX] = p.size[bY] = p.size[bS] = big;
    p.size[bCol] = p.size[bDcol] = chans * 9 * M;
    p.size[bW] = p.size[bGW] = std::max<size_t>(chans * 9 * F, params) + 64;
    p.size[bH1] = p.size[bH2] = std::max<size_t>(2 * M, static_cast<size_t>(B) * std::max(A + 1, V)) + 64;
    p.size[bScale] = p.size[bShift] = chans;
    p.size[bStats] = 2 * chans;
    return p;
}

// ---------------------------------------------------------------- timing

struct Timing { double median = 0, min = 0, gpuMedian = 0; int calls = 0; };

// Calls f repeatedly for at least gSeconds (and at least 5 times) after two
// warm-up calls. f returns GPU seconds for that call, or 0 if not applicable.
template <class F>
Timing timeCalls(F&& f, double minSeconds = -1) {
    if (minSeconds < 0) minSeconds = gSeconds;
    f();
    f();
    std::vector<double> wall, gpu;
    const auto start = Clock::now();
    do {
        const auto t0 = Clock::now();
        const double g = f();
        wall.push_back(secondsSince(t0));
        gpu.push_back(g);
    } while (secondsSince(start) < minSeconds || wall.size() < 5);
    Timing t;
    t.calls = static_cast<int>(wall.size());
    t.min = *std::min_element(wall.begin(), wall.end());
    std::nth_element(wall.begin(), wall.begin() + wall.size() / 2, wall.end());
    t.median = wall[wall.size() / 2];
    std::nth_element(gpu.begin(), gpu.begin() + gpu.size() / 2, gpu.end());
    t.gpuMedian = gpu[gpu.size() / 2];
    return t;
}

void fillRandom(float* p, size_t count, uint64_t seed, float scale) {
    azgo::Rng rng(seed);
    for (size_t i = 0; i < count; ++i) p[i] = static_cast<float>((rng.uniform() * 2.0 - 1.0) * scale);
}

uint64_t fnv(const float* p, size_t count) {
    uint64_t h = 1469598103934665603ull;
    const auto* bytes = reinterpret_cast<const unsigned char*>(p);
    for (size_t i = 0; i < count * sizeof(float); ++i) { h ^= bytes[i]; h *= 1099511628211ull; }
    return h;
}

// ---------------------------------------------------------------- CPU backends

// Accelerate with its own threading left alone (what the project avoids, so
// that 18 workers do not oversubscribe the cores).
class ThreadedBlas : public azgo::Backend {
public:
    const char* name() const override { return "blas-threaded"; }
    void gemm(bool transA, bool transB, int M, int N, int K, float alpha, const float* A,
              const float* B, float beta, float* C) const override {
        cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                    transB ? CblasTrans : CblasNoTrans, M, N, K, alpha, A, transA ? M : K, B,
                    transB ? K : N, beta, C, N);
    }
};

std::unique_ptr<azgo::Backend> cpuBackend(bool threaded) {
    if (threaded) return std::make_unique<ThreadedBlas>();
    return azgo::makeBackend("blas", 1);
}

struct CpuBuffers {
    std::vector<float> v[kBufs];
    explicit CpuBuffers(const Plan& p) {
        for (int i = 0; i < kBufs; ++i) {
            v[i].resize(p.size[i]);
            fillRandom(v[i].data(), v[i].size(), 100 + i, i == bW ? 0.05f : 1.0f);
        }
    }
    float* operator[](Buf b) { return v[b].data(); }
};

// The plan's gemms only, back to back: the CPU time a pass spends in gemm.
double cpuGemmSequence(const azgo::Backend& be, const Plan& p, CpuBuffers& buf) {
    for (const auto* ops : {&p.forward, &p.backward})
        for (const Op& o : *ops)
            if (o.kind == Op::kGemm)
                be.gemm(o.g.tA, o.g.tB, o.g.M, o.g.N, o.g.K, 1.0f, buf[o.a], buf[o.b], o.g.beta,
                        buf[o.c]);
    return 0;
}

struct NetInputs {
    std::vector<float> input, pi, z, logits, values;
    NetInputs(const azgo::NetConfig& nc, int B, uint64_t seed) {
        azgo::Rng rng(seed);
        input.resize(static_cast<size_t>(B) * nc.planes() * nc.area());
        for (float& x : input) x = static_cast<float>(rng.below(2));
        pi.assign(static_cast<size_t>(B) * nc.actions(), 0.0f);
        for (int b = 0; b < B; ++b) pi[static_cast<size_t>(b) * nc.actions() + rng.below(nc.actions())] = 1.0f;
        z.resize(B);
        for (float& v : z) v = rng.below(2) ? 1.0f : -1.0f;
        logits.resize(static_cast<size_t>(B) * nc.actions());
        values.resize(B);
    }
};

void sectionCpu(bool threaded) {
    auto be = cpuBackend(threaded);
    std::printf("# section %s: %s\n", threaded ? "cpuall" : "cpu1",
                threaded ? "Accelerate cblas_sgemm, default threading (VECLIB_MAXIMUM_THREADS unset)"
                         : "project BlasBackend = makeBackend(\"blas\"): Accelerate, VECLIB_MAXIMUM_THREADS=1");

    std::printf("\n## gemm per call (median of >= %.1f s)\n", gSeconds);
    std::printf("%-13s %4s  %-14s %2s %2s %6s %7s %6s %10s %9s  %s\n", "net", "B", "shape", "tA", "tB",
                "M", "N", "K", "ms", "GFLOP/s", "hash");
    for (const NetSpec& s : kNets) {
        if (skipNet(s)) continue;
        for (int B : kBatches) {
            for (const Gemm& g : convShapes(s, B)) {
                std::vector<float> A(static_cast<size_t>(g.M) * g.K), Bm(static_cast<size_t>(g.K) * g.N),
                    C(static_cast<size_t>(g.M) * g.N);
                fillRandom(A.data(), A.size(), 1, 0.05f);
                fillRandom(Bm.data(), Bm.size(), 2, 1.0f);
                const Timing t = timeCalls([&] {
                    std::fill(C.begin(), C.end(), 0.0f);   // beta = 1 shapes accumulate from 0 each call
                    be->gemm(g.tA, g.tB, g.M, g.N, g.K, 1.0f, A.data(), Bm.data(), g.beta, C.data());
                    return 0.0;
                });
                std::printf("%-13s %4d  %-14s %2d %2d %6d %7d %6d %10.3f %9.0f  %016llx\n", s.name, B,
                            g.label.c_str(), g.tA, g.tB, g.M, g.N, g.K, t.median * 1e3,
                            g.flops() / t.median * 1e-9, (unsigned long long)fnv(C.data(), C.size()));
                std::fflush(stdout);
            }
        }
    }

    std::printf("\n## whole passes: plan gemms only vs the real Network (median ms)\n");
    std::printf("%-13s %4s  %-9s %7s %10s %10s %10s %9s\n", "net", "B", "pass", "gemms", "GFLOP",
                "gemm ms", "network ms", "gemm %");
    for (const NetSpec& s : kNets) {
        if (skipNet(s)) continue;
        const azgo::NetConfig nc = netConfig(s);
        const azgo::Network net0(nc, 1);
        for (int B : kBatches) {
            const Plan p = makePlan(s, B, false);
            CpuBuffers buf(p);
            const Timing tg = timeCalls([&] { return cpuGemmSequence(*be, p, buf); });
            azgo::Workspace ws(nc, B, *be);
            NetInputs in(nc, B, 7);
            const Timing tn = timeCalls([&] {
                net0.infer(ws, in.input.data(), B, in.logits.data(), in.values.data());
                return 0.0;
            });
            std::printf("%-13s %4d  %-9s %7d %10.3f %10.3f %10.3f %8.0f%%   (%.0f evals/s)\n", s.name, B,
                        "infer", p.count(Op::kGemm), p.gemmFlops() * 1e-9, tg.median * 1e3,
                        tn.median * 1e3, 100 * tg.median / tn.median, B / tn.median);
            std::fflush(stdout);
        }
        {
            const int B = kTrainBatch;
            const Plan p = makePlan(s, B, true);
            CpuBuffers buf(p);
            const Timing tg = timeCalls([&] { return cpuGemmSequence(*be, p, buf); });
            azgo::Network net(nc, 1);
            azgo::Workspace ws(nc, B, *be);
            NetInputs in(nc, B, 7);
            const Timing tn = timeCalls([&] {
                net.trainStep(ws, in.input.data(), in.pi.data(), in.z.data(), B);
                return 0.0;
            }, std::max(gSeconds, 1.0));
            std::printf("%-13s %4d  %-9s %7d %10.3f %10.3f %10.3f %8.0f%%\n", s.name, B, "trainStep",
                        p.count(Op::kGemm), p.gemmFlops() * 1e-9, tg.median * 1e3, tn.median * 1e3,
                        100 * tg.median / tn.median);
            std::fflush(stdout);
        }
    }
    if (threaded) return;

    // Self-play's view: every core runs its own worker at batch 8.
    const int workers = static_cast<int>(std::thread::hardware_concurrency());
    std::printf("\n## %d workers, each Network::infer at batch 8 on its own thread (aggregate)\n", workers);
    for (const NetSpec& s : kNets) {
        if (skipNet(s)) continue;
        if (s.n != 9) continue;
        const azgo::NetConfig nc = netConfig(s);
        const azgo::Network net(nc, 1);
        std::atomic<long long> evals{0};
        std::atomic<bool> stop{false};
        std::vector<std::thread> pool;
        const auto start = Clock::now();
        for (int w = 0; w < workers; ++w) {
            pool.emplace_back([&, w] {
                auto wbe = azgo::makeBackend("blas", 1);
                azgo::Workspace ws(nc, 8, *wbe);
                NetInputs in(nc, 8, 11 + w);
                long long mine = 0;
                while (!stop.load(std::memory_order_relaxed)) {
                    net.infer(ws, in.input.data(), 8, in.logits.data(), in.values.data());
                    mine += 8;
                }
                evals += mine;
            });
        }
        std::this_thread::sleep_for(std::chrono::duration<double>(std::max(3.0, 4 * gSeconds)));
        stop = true;
        for (auto& t : pool) t.join();
        const double el = secondsSince(start);
        std::printf("%-13s  %8.0f evals/s total  %7.0f per worker  (%.2f ms per batch-8 call)\n", s.name,
                    evals / el, evals / el / workers, 8.0 / (evals / el / workers) * 1e3);
        std::fflush(stdout);
    }
}

// ---------------------------------------------------------------- Metal

const char* kShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

// Row-major C[M x N] = alpha * op(A) * op(B) + beta * C, as Backend::gemm.
// A 64 x 64 tile of C per threadgroup of 16 x 16 threads, 4 x 4 outputs per
// thread, K in steps of 16 through threadgroup memory. Every output element
// is one thread's fma chain over k = kBegin, kBegin+1, ... in order, so its
// bits depend only on its row of op(A) and its column of op(B): not on M, N,
// the tile it lands in, or the run. With partial = 1 (split-K) threadgroup z
// covers k in [z*kChunk, (z+1)*kChunk) and writes its partial sum to slice z.
struct GemmParams { uint M, N, K, transA, transB, kChunk, partial; float alpha, beta; };

constant constexpr uint BM = 64, BN = 64, BK = 16;

kernel void gemm_tiled(device const float* A [[buffer(0)]],
                       device const float* B [[buffer(1)]],
                       device float* C [[buffer(2)]],
                       constant GemmParams& p [[buffer(3)]],
                       uint3 tg [[threadgroup_position_in_grid]],
                       uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float As[BK][BM];
    threadgroup float Bs[BK][BN];
    const uint tx = lid.x, ty = lid.y, tid = ty * 16 + tx;
    const uint row0 = tg.y * BM, col0 = tg.x * BN;
    const uint kBegin = tg.z * p.kChunk;
    const uint kEnd = min(p.K, kBegin + p.kChunk);
    float acc[4][4];
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 4; ++i)
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    for (uint k0 = kBegin; k0 < kEnd; k0 += BK) {
        for (uint e = tid; e < BM * BK; e += 256) {
            uint m, k;
            if (p.transA) { k = e / BM; m = e % BM; } else { m = e / BK; k = e % BK; }
            const uint gm = row0 + m, gk = k0 + k;
            float v = 0.0f;
            if (gm < p.M && gk < kEnd) v = p.transA ? A[gk * p.M + gm] : A[gm * p.K + gk];
            As[k][m] = v;
        }
        for (uint e = tid; e < BK * BN; e += 256) {
            uint k, n;
            if (p.transB) { n = e / BK; k = e % BK; } else { k = e / BN; n = e % BN; }
            const uint gn = col0 + n, gk = k0 + k;
            float v = 0.0f;
            if (gn < p.N && gk < kEnd) v = p.transB ? B[gn * p.K + gk] : B[gk * p.N + gn];
            Bs[k][n] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("clang loop unroll(full)")
        for (uint k = 0; k < BK; ++k) {
            float a[4], b[4];
            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i < 4; ++i) a[i] = As[k][ty + 16 * i];
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < 4; ++j) b[j] = Bs[k][tx + 16 * j];
            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i < 4; ++i)
                _Pragma("clang loop unroll(full)")
                for (uint j = 0; j < 4; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 4; ++i) {
        const uint gm = row0 + ty + 16 * i;
        if (gm >= p.M) continue;
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < 4; ++j) {
            const uint gn = col0 + tx + 16 * j;
            if (gn >= p.N) continue;
            if (p.partial) {
                C[tg.z * p.M * p.N + gm * p.N + gn] = acc[i][j];
            } else {
                float r = p.alpha * acc[i][j];
                if (p.beta != 0.0f) r += p.beta * C[gm * p.N + gn];
                C[gm * p.N + gn] = r;
            }
        }
    }
}

// The same tiling and split-K on the GPU's 8x8 simdgroup matrix instructions:
// 4 simdgroups (128 threads) per 64 x 64 tile, each a 32 x 32 quarter held as
// 4 x 4 accumulators of 8 x 8, k in steps of 8 in order. Every 8 x 8 output
// block runs the same instruction sequence over the same k order wherever it
// lies, so it is as batch-independent as gemm_tiled.
kernel void gemm_simd(device const float* A [[buffer(0)]],
                      device const float* B [[buffer(1)]],
                      device float* C [[buffer(2)]],
                      constant GemmParams& p [[buffer(3)]],
                      uint3 tg [[threadgroup_position_in_grid]],
                      uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float As[BM * BK];   // [m][k]; reused to stage results at the end
    threadgroup float Bs[BK * BN];   // [k][n]
    const uint tid = lid.x, sg = tid / 32;
    const uint row0 = tg.y * BM, col0 = tg.x * BN;
    const uint kBegin = tg.z * p.kChunk;
    const uint kEnd = min(p.K, kBegin + p.kChunk);
    const uint sr = (sg / 2) * 32, sc = (sg % 2) * 32;
    simdgroup_float8x8 acc[4][4];
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 4; ++i)
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = kBegin; k0 < kEnd; k0 += BK) {
        for (uint e = tid; e < BM * BK; e += 128) {
            uint m, k;
            if (p.transA) { k = e / BM; m = e % BM; } else { m = e / BK; k = e % BK; }
            const uint gm = row0 + m, gk = k0 + k;
            float v = 0.0f;
            if (gm < p.M && gk < kEnd) v = p.transA ? A[gk * p.M + gm] : A[gm * p.K + gk];
            As[m * BK + k] = v;
        }
        for (uint e = tid; e < BK * BN; e += 128) {
            uint k, n;
            if (p.transB) { n = e / BK; k = e % BK; } else { k = e / BN; n = e % BN; }
            const uint gn = col0 + n, gk = k0 + k;
            float v = 0.0f;
            if (gn < p.N && gk < kEnd) v = p.transB ? B[gn * p.K + gk] : B[gk * p.N + gn];
            Bs[k * BN + n] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("clang loop unroll(full)")
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a[4], b[4];
            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i < 4; ++i) simdgroup_load(a[i], As + (sr + 8 * i) * BK + kk, BK);
            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < 4; ++j) simdgroup_load(b[j], Bs + kk * BN + sc + 8 * j, BN);
            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i < 4; ++i)
                _Pragma("clang loop unroll(full)")
                for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Each simdgroup stages one 8 x 8 block at a time in its own 64 floats
    // (threadgroup memory stays at 8 KB, so several threadgroups share a
    // core), and every lane writes two of its elements.
    threadgroup float* stage = As + sg * 64;
    const uint lane = tid % 32;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < 4; ++i)
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < 4; ++j) {
            simdgroup_store(acc[i][j], stage, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                const uint gm = row0 + sr + 8 * i + e / 8, gn = col0 + sc + 8 * j + e % 8;
                if (gm >= p.M || gn >= p.N) continue;
                if (p.partial) {
                    C[tg.z * p.M * p.N + gm * p.N + gn] = stage[e];
                } else {
                    float r = p.alpha * stage[e];
                    if (p.beta != 0.0f) r += p.beta * C[gm * p.N + gn];
                    C[gm * p.N + gn] = r;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
}

// Split-K: sums the S slices in order z = 0, 1, ..., S-1 (fixed for a shape).
struct ReduceParams { uint count, S; float alpha, beta; };
kernel void splitk_reduce(device const float* P [[buffer(0)]],
                          device float* C [[buffer(1)]],
                          constant ReduceParams& r [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= r.count) return;
    float s = 0.0f;
    for (uint z = 0; z < r.S; ++z) s += P[z * r.count + i];
    float v = r.alpha * s;
    if (r.beta != 0.0f) v += r.beta * C[i];
    C[i] = v;
}

// Backend::im2col for k = 3: in [C][B][n*n] -> col [C*9][B*n*n].
struct ConvParams { uint C, B, n; };
kernel void im2col3(device const float* in [[buffer(0)]],
                    device float* col [[buffer(1)]],
                    constant ConvParams& p [[buffer(2)]],
                    uint2 gid [[thread_position_in_grid]]) {
    const uint area = p.n * p.n, Mt = p.B * area;
    if (gid.x >= Mt || gid.y >= p.C * 9) return;
    const uint c = gid.y / 9, r = gid.y % 9;
    const uint b = gid.x / area, q = gid.x % area;
    const int y = int(q / p.n) + int(r / 3) - 1, x = int(q % p.n) + int(r % 3) - 1;
    float v = 0.0f;
    if (y >= 0 && y < int(p.n) && x >= 0 && x < int(p.n))
        v = in[(c * p.B + b) * area + uint(y) * p.n + uint(x)];
    col[gid.y * Mt + gid.x] = v;
}

// Backend::col2im as a gather: each input point sums its 9 contributions in
// the (dy, dx) order of the CPU loop, so the bits match it, with no atomics.
kernel void col2im3(device const float* col [[buffer(0)]],
                    device float* out [[buffer(1)]],
                    constant ConvParams& p [[buffer(2)]],
                    uint2 gid [[thread_position_in_grid]]) {
    const uint area = p.n * p.n, Mt = p.B * area;
    if (gid.x >= Mt || gid.y >= p.C) return;
    const uint c = gid.y, b = gid.x / area, q = gid.x % area;
    const int y = int(q / p.n), x = int(q % p.n);
    float s = 0.0f;
    for (int dy = 0; dy < 3; ++dy)
        for (int dx = 0; dx < 3; ++dx) {
            const int oy = y - dy + 1, ox = x - dx + 1;
            if (oy < 0 || oy >= int(p.n) || ox < 0 || ox >= int(p.n)) continue;
            s += col[((c * 3 + uint(dy)) * 3 + uint(dx)) * Mt + b * area + uint(oy) * p.n + uint(ox)];
        }
    out[c * Mt + gid.x] = s;
}

// Inference batch norm folded to scale/shift, optional skip add, optional ReLU.
struct EwParams { uint count, perChannel, hasSkip, relu; };
kernel void bn_relu(device const float* x [[buffer(0)]],
                    device float* y [[buffer(1)]],
                    device const float* scale [[buffer(2)]],
                    device const float* shift [[buffer(3)]],
                    device const float* skip [[buffer(4)]],
                    constant EwParams& p [[buffer(5)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    const uint c = i / p.perChannel;
    float v = scale[c] * x[i] + shift[c];
    if (p.hasSkip) v += skip[i];
    if (p.relu) v = max(v, 0.0f);
    y[i] = v;
}

// Two reads, one write: stands in for ReLU/BN backward, adds, flatten, SGD.
kernel void ew2(device const float* x [[buffer(0)]],
                device const float* y [[buffer(1)]],
                device float* out [[buffer(2)]],
                constant uint& count [[buffer(3)]],
                uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    out[i] = x[i] > 0.0f ? y[i] : 0.5f * y[i];
}

// Batch statistics of one channel per threadgroup (256 threads): mean, then
// variance around it, each a strided sum plus a fixed-shape tree.
kernel void channel_stats(device const float* x [[buffer(0)]],
                          device float* stats [[buffer(1)]],
                          constant uint& perChannel [[buffer(2)]],
                          uint c [[threadgroup_position_in_grid]],
                          uint t [[thread_position_in_threadgroup]]) {
    threadgroup float sh[256];
    device const float* xc = x + c * perChannel;
    float s = 0.0f;
    for (uint i = t; i < perChannel; i += 256) s += xc[i];
    sh[t] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = 128; w > 0; w >>= 1) {
        if (t < w) sh[t] += sh[t + w];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = sh[0] / float(perChannel);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q = 0.0f;
    for (uint i = t; i < perChannel; i += 256) { const float d = xc[i] - mean; q += d * d; }
    sh[t] = q;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = 128; w > 0; w >>= 1) {
        if (t < w) sh[t] += sh[t + w];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) { stats[2 * c] = mean; stats[2 * c + 1] = rsqrt(sh[0] / float(perChannel) + 1e-5f); }
}
)METAL";

struct GemmParams { uint32_t M, N, K, transA, transB, kChunk, partial; float alpha, beta; };

// Gemm implementations compared. kAccel is the CPU (Accelerate, one thread).
enum Impl { kHand = 0, kMps = 1, kAccel = 2, kSimd = 3 };
const char* kImpl[] = {"hand", "mps", "accelerate", "simd"};
const int kGpuImpls[] = {kHand, kSimd, kMps};
struct ReduceParams { uint32_t count, S; float alpha, beta; };
struct ConvParams { uint32_t C, B, n; };
struct EwParams { uint32_t count, perChannel, hasSkip, relu; };

// Split-K only where K is long and the output too small to fill the GPU: in
// practice the dW gemms (K = B * n * n). It depends on the shape alone, so a
// given shape always sums in the same order. Forward gemms (K <= 9 * filters)
// are never split, which keeps every output column independent of the batch.
bool gNoSplit = false;   // det: also try the dW gemms as one sequential k chain
int splitCount(const Gemm& g) {
    if (gNoSplit || g.K < 4096) return 1;
    const long tiles = static_cast<long>((g.M + 63) / 64) * ((g.N + 63) / 64);
    if (tiles >= 128) return 1;
    return std::min(64, g.K / 1024);
}

struct Gpu {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> gemm, gemmSimd, reduce, im2col, col2im, bnRelu, ew, stats;
    id<MTLBuffer> partial;   // split-K slices, grown on demand

    Gpu() {
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::fprintf(stderr, "no Metal device\n"); std::exit(1); }
        queue = [dev newCommandQueue];
        MTLCompileOptions* opt = [MTLCompileOptions new];
        opt.mathMode = MTLMathModeSafe;   // no fast-math reassociation
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(kShaderSource) options:opt error:&err];
        if (!lib) { std::fprintf(stderr, "shader compile failed: %s\n", err.localizedDescription.UTF8String); std::exit(1); }
        auto pipe = [&](const char* name) {
            NSError* e = nil;
            id<MTLFunction> fn = [lib newFunctionWithName:@(name)];
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&e];
            if (!ps) { std::fprintf(stderr, "pipeline %s: %s\n", name, e.localizedDescription.UTF8String); std::exit(1); }
            return ps;
        };
        gemm = pipe("gemm_tiled");
        gemmSimd = pipe("gemm_simd");
        reduce = pipe("splitk_reduce");
        im2col = pipe("im2col3");
        col2im = pipe("col2im3");
        bnRelu = pipe("bn_relu");
        ew = pipe("ew2");
        stats = pipe("channel_stats");
    }

    id<MTLBuffer> buffer(size_t floats) {
        return [dev newBufferWithLength:std::max<size_t>(floats, 1) * sizeof(float)
                                options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> ensurePartial(size_t floats) {
        if (!partial || partial.length < floats * sizeof(float)) partial = buffer(floats);
        return partial;
    }
};

// One command buffer being filled; compute encoders are opened lazily and
// closed before an MPS kernel encodes itself.
struct Ctx {
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc = nil;
    id<MTLComputeCommandEncoder> compute() {
        if (!enc) enc = [cb computeCommandEncoder];
        return enc;
    }
    void endCompute() {
        if (enc) { [enc endEncoding]; enc = nil; }
    }
};

MTLSize sz(NSUInteger x, NSUInteger y = 1, NSUInteger z = 1) { return MTLSizeMake(x, y, z); }

void encodeHandGemm(Gpu& gpu, Ctx& c, const Gemm& g, float alpha, id<MTLBuffer> A, id<MTLBuffer> B,
                    id<MTLBuffer> C, bool simd = false) {
    const int S0 = splitCount(g);
    uint32_t kChunk = static_cast<uint32_t>(g.K);
    int S = 1;
    if (S0 > 1) {
        kChunk = static_cast<uint32_t>(((g.K + S0 - 1) / S0 + 15) / 16 * 16);
        S = static_cast<int>((g.K + kChunk - 1) / kChunk);
    }
    id<MTLComputeCommandEncoder> e = c.compute();
    [e setComputePipelineState:simd ? gpu.gemmSimd : gpu.gemm];
    GemmParams p{static_cast<uint32_t>(g.M), static_cast<uint32_t>(g.N), static_cast<uint32_t>(g.K),
                 g.tA, g.tB, kChunk, S > 1, alpha, g.beta};
    [e setBuffer:A offset:0 atIndex:0];
    [e setBuffer:B offset:0 atIndex:1];
    id<MTLBuffer> P = S > 1 ? gpu.ensurePartial(static_cast<size_t>(S) * g.M * g.N) : C;
    [e setBuffer:P offset:0 atIndex:2];
    [e setBytes:&p length:sizeof p atIndex:3];
    [e dispatchThreadgroups:sz((g.N + 63) / 64, (g.M + 63) / 64, S)
      threadsPerThreadgroup:simd ? sz(128) : sz(16, 16)];
    if (S > 1) {
        ReduceParams r{static_cast<uint32_t>(g.M * g.N), static_cast<uint32_t>(S), alpha, g.beta};
        [e setComputePipelineState:gpu.reduce];
        [e setBuffer:P offset:0 atIndex:0];
        [e setBuffer:C offset:0 atIndex:1];
        [e setBytes:&r length:sizeof r atIndex:2];
        [e dispatchThreads:sz(r.count) threadsPerThreadgroup:sz(256)];
    }
}

// MPSMatrixMultiplication objects and matrix wrappers, made once per shape.
struct MpsGemm {
    MPSMatrixMultiplication* mm;
    MPSMatrix *a, *b, *c;
};

MPSMatrix* mpsMatrix(id<MTLBuffer> buf, int rows, int cols) {
    MPSMatrixDescriptor* d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols
                                                                 rowBytes:cols * sizeof(float)
                                                                 dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:buf descriptor:d];
}

MpsGemm makeMps(Gpu& gpu, const Gemm& g, float alpha, id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C) {
    MpsGemm m;
    m.mm = [[MPSMatrixMultiplication alloc] initWithDevice:gpu.dev transposeLeft:g.tA transposeRight:g.tB
                                                resultRows:g.M resultColumns:g.N interiorColumns:g.K
                                                     alpha:alpha beta:g.beta];
    m.a = g.tA ? mpsMatrix(A, g.K, g.M) : mpsMatrix(A, g.M, g.K);
    m.b = g.tB ? mpsMatrix(B, g.N, g.K) : mpsMatrix(B, g.K, g.N);
    m.c = mpsMatrix(C, g.M, g.N);
    return m;
}

void encodeMps(Ctx& c, const MpsGemm& m) {
    c.endCompute();
    [m.mm encodeToCommandBuffer:c.cb leftMatrix:m.a rightMatrix:m.b resultMatrix:m.c];
}

double gpuSeconds(id<MTLCommandBuffer> cb) { return cb.GPUEndTime - cb.GPUStartTime; }

// A plan bound to GPU buffers: one encode function per op.
struct GpuPlan {
    id<MTLBuffer> buf[kBufs];
    std::vector<std::function<void(Ctx&)>> forward, backward;
};

GpuPlan bindPlan(Gpu& gpu, const Plan& p, int impl) {
    GpuPlan gp;
    for (int i = 0; i < kBufs; ++i) {
        gp.buf[i] = gpu.buffer(p.size[i]);
        fillRandom(static_cast<float*>(gp.buf[i].contents), p.size[i], 100 + i, i == bW ? 0.05f : 1.0f);
    }
    float* scale = static_cast<float*>(gp.buf[bScale].contents);
    for (size_t i = 0; i < p.size[bScale]; ++i) scale[i] = 1.0f;
    auto bindOps = [&](const std::vector<Op>& ops, std::vector<std::function<void(Ctx&)>>& out) {
        for (const Op& o : ops) {
            id<MTLBuffer> a = gp.buf[o.a], b = gp.buf[o.b], cbuf = gp.buf[o.c];
            Gpu* g = &gpu;
            switch (o.kind) {
            case Op::kGemm:
                if (impl == kMps) {
                    MpsGemm m = makeMps(gpu, o.g, 1.0f, a, b, cbuf);
                    out.push_back([m](Ctx& c) { encodeMps(c, m); });
                } else {
                    Gemm gg = o.g;
                    const bool simd = impl == kSimd;
                    out.push_back([g, gg, a, b, cbuf, simd](Ctx& c) { encodeHandGemm(*g, c, gg, 1.0f, a, b, cbuf, simd); });
                }
                break;
            case Op::kIm2col:
            case Op::kCol2im: {
                ConvParams cp{static_cast<uint32_t>(o.C), static_cast<uint32_t>(o.B), static_cast<uint32_t>(o.n)};
                const bool fwd = o.kind == Op::kIm2col;
                const NSUInteger Mt = static_cast<NSUInteger>(o.B) * o.n * o.n;
                const NSUInteger rows = fwd ? o.C * 9 : o.C;
                out.push_back([g, cp, fwd, Mt, rows, a, cbuf](Ctx& c) {
                    id<MTLComputeCommandEncoder> e = c.compute();
                    [e setComputePipelineState:fwd ? g->im2col : g->col2im];
                    [e setBuffer:a offset:0 atIndex:0];
                    [e setBuffer:cbuf offset:0 atIndex:1];
                    [e setBytes:&cp length:sizeof cp atIndex:2];
                    [e dispatchThreads:sz(Mt, rows) threadsPerThreadgroup:sz(64, 4)];
                });
                break;
            }
            case Op::kBnRelu: {
                EwParams ep{o.count, o.perChannel, o.skip, o.relu};
                id<MTLBuffer> sc = gp.buf[bScale], sh = gp.buf[bShift], sk = gp.buf[bS];
                out.push_back([g, ep, a, cbuf, sc, sh, sk](Ctx& c) {
                    id<MTLComputeCommandEncoder> e = c.compute();
                    [e setComputePipelineState:g->bnRelu];
                    [e setBuffer:a offset:0 atIndex:0];
                    [e setBuffer:cbuf offset:0 atIndex:1];
                    [e setBuffer:sc offset:0 atIndex:2];
                    [e setBuffer:sh offset:0 atIndex:3];
                    [e setBuffer:sk offset:0 atIndex:4];
                    [e setBytes:&ep length:sizeof ep atIndex:5];
                    [e dispatchThreads:sz(ep.count) threadsPerThreadgroup:sz(256)];
                });
                break;
            }
            case Op::kEw: {
                const uint32_t count = o.count;
                out.push_back([g, count, a, b, cbuf](Ctx& c) {
                    id<MTLComputeCommandEncoder> e = c.compute();
                    [e setComputePipelineState:g->ew];
                    [e setBuffer:a offset:0 atIndex:0];
                    [e setBuffer:b offset:0 atIndex:1];
                    [e setBuffer:cbuf offset:0 atIndex:2];
                    [e setBytes:&count length:sizeof count atIndex:3];
                    [e dispatchThreads:sz(count) threadsPerThreadgroup:sz(256)];
                });
                break;
            }
            case Op::kStats: {
                const uint32_t per = o.perChannel, channels = o.count;
                out.push_back([g, per, channels, a, cbuf](Ctx& c) {
                    id<MTLComputeCommandEncoder> e = c.compute();
                    [e setComputePipelineState:g->stats];
                    [e setBuffer:a offset:0 atIndex:0];
                    [e setBuffer:cbuf offset:0 atIndex:1];
                    [e setBytes:&per length:sizeof per atIndex:2];
                    [e dispatchThreadgroups:sz(channels) threadsPerThreadgroup:sz(256)];
                });
                break;
            }
            }
        }
    };
    bindOps(p.forward, gp.forward);
    bindOps(p.backward, gp.backward);
    return gp;
}

// All ops of `ops` in one command buffer; returns GPU seconds.
double runOneBuffer(Gpu& gpu, const std::vector<std::function<void(Ctx&)>>& ops) {
    Ctx c{[gpu.queue commandBuffer]};
    for (const auto& op : ops) op(c);
    c.endCompute();
    [c.cb commit];
    [c.cb waitUntilCompleted];
    return gpuSeconds(c.cb);
}

// One command buffer, commit and wait per op: what a drop-in Backend does.
double runPerOp(Gpu& gpu, const std::vector<std::function<void(Ctx&)>>& ops) {
    double g = 0;
    for (const auto& op : ops) {
        Ctx c{[gpu.queue commandBuffer]};
        op(c);
        c.endCompute();
        [c.cb commit];
        [c.cb waitUntilCompleted];
        g += gpuSeconds(c.cb);
    }
    return g;
}

void sectionGpu() {
    Gpu gpu;
    std::printf("# section gpu: %s, hand kernel (64x64 tile, 4x4 per thread, MTLMathModeSafe) and MPSMatrixMultiplication\n",
                gpu.dev.name.UTF8String);
    std::printf("# per call = new command buffer, encode, commit, waitUntilCompleted (wall, median);\n"
                "# gpu = GPUEndTime - GPUStartTime of that buffer; batched = 20 copies in one buffer, GPU time / 20.\n");

    // Empty command buffer: the floor of any per-call design.
    {
        const Timing t = timeCalls([&] {
            id<MTLCommandBuffer> cb = [gpu.queue commandBuffer];
            [cb commit];
            [cb waitUntilCompleted];
            return 0.0;
        });
        id<MTLBuffer> one = gpu.buffer(256);
        const Timing t2 = timeCalls([&] {
            Ctx c{[gpu.queue commandBuffer]};
            id<MTLComputeCommandEncoder> e = c.compute();
            uint32_t count = 256;
            [e setComputePipelineState:gpu.ew];
            [e setBuffer:one offset:0 atIndex:0];
            [e setBuffer:one offset:0 atIndex:1];
            [e setBuffer:one offset:0 atIndex:2];
            [e setBytes:&count length:sizeof count atIndex:3];
            [e dispatchThreads:sz(256) threadsPerThreadgroup:sz(256)];
            c.endCompute();
            [c.cb commit];
            [c.cb waitUntilCompleted];
            return gpuSeconds(c.cb);
        });
        std::printf("\n## overhead: empty command buffer commit+wait %.1f us (min %.1f); one tiny kernel %.1f us wall, %.1f us GPU\n",
                    t.median * 1e6, t.min * 1e6, t2.median * 1e6, t2.gpuMedian * 1e6);
    }

    std::printf("\n## gemm per shape (ms; GFLOP/s from the batched time)\n");
    std::printf("%-13s %4s  %-14s %2s %2s %6s %7s %6s | %-4s %9s %9s %9s %8s\n", "net", "B", "shape", "tA",
                "tB", "M", "N", "K", "impl", "per-call", "gpu", "batched", "GFLOP/s");
    for (const NetSpec& s : kNets) {
        if (skipNet(s)) continue;
        for (int B : kBatches) {
            for (const Gemm& g : convShapes(s, B)) {
                id<MTLBuffer> A = gpu.buffer(static_cast<size_t>(g.M) * g.K);
                id<MTLBuffer> Bb = gpu.buffer(static_cast<size_t>(g.K) * g.N);
                id<MTLBuffer> C = gpu.buffer(static_cast<size_t>(g.M) * g.N);
                fillRandom(static_cast<float*>(A.contents), static_cast<size_t>(g.M) * g.K, 1, 0.05f);
                fillRandom(static_cast<float*>(Bb.contents), static_cast<size_t>(g.K) * g.N, 2, 1.0f);
                for (int impl : kGpuImpls) {
                    std::function<void(Ctx&)> op;
                    if (impl != kMps) {
                        const bool simd = impl == kSimd;
                        op = [&gpu, g, A, Bb, C, simd](Ctx& c) { encodeHandGemm(gpu, c, g, 1.0f, A, Bb, C, simd); };
                    } else {
                        MpsGemm m = makeMps(gpu, g, 1.0f, A, Bb, C);
                        op = [m](Ctx& c) { encodeMps(c, m); };
                    }
                    std::vector<std::function<void(Ctx&)>> one{op}, twenty(20, op);
                    const Timing t1 = timeCalls([&] { return runOneBuffer(gpu, one); });
                    const Timing t20 = timeCalls([&] { return runOneBuffer(gpu, twenty); });
                    const double per = t20.gpuMedian / 20;
                    std::printf("%-13s %4d  %-14s %2d %2d %6d %7d %6d | %-4s %9.3f %9.3f %9.3f %8.0f\n",
                                s.name, B, g.label.c_str(), g.tA, g.tB, g.M, g.N, g.K,
                                kImpl[impl], t1.median * 1e3, t1.gpuMedian * 1e3, per * 1e3,
                                g.flops() / per * 1e-9);
                    std::fflush(stdout);
                }
            }
        }
    }

    std::printf("\n## whole passes on the GPU (ms). one-cb = every op in one command buffer (training: forward\n"
                "## buffer, wait, backward buffer, wait, as the loss runs between); per-op = commit+wait per op.\n");
    std::printf("%-13s %4s  %-9s %-4s %4s %10s %10s %10s %12s\n", "net", "B", "pass", "impl", "ops", "one-cb",
                "gpu", "per-op", "evals/s(1cb)");
    for (const NetSpec& s : kNets) {
        if (skipNet(s)) continue;
        for (int B : kBatches) {
            const Plan p = makePlan(s, B, false);
            for (int impl : kGpuImpls) {
                GpuPlan gp = bindPlan(gpu, p, impl);
                const Timing t1 = timeCalls([&] { return runOneBuffer(gpu, gp.forward); });
                const Timing tp = timeCalls([&] { return runPerOp(gpu, gp.forward); });
                std::printf("%-13s %4d  %-9s %-4s %4zu %10.3f %10.3f %10.3f %12.0f\n", s.name, B, "infer",
                            kImpl[impl], gp.forward.size(), t1.median * 1e3, t1.gpuMedian * 1e3,
                            tp.median * 1e3, B / t1.median);
                std::fflush(stdout);
            }
        }
        const Plan p = makePlan(s, kTrainBatch, true);
        for (int impl : kGpuImpls) {
            GpuPlan gp = bindPlan(gpu, p, impl);
            const Timing t1 = timeCalls([&] { return runOneBuffer(gpu, gp.forward) + runOneBuffer(gpu, gp.backward); });
            const Timing tp = timeCalls([&] { return runPerOp(gpu, gp.forward) + runPerOp(gpu, gp.backward); });
            std::printf("%-13s %4d  %-9s %-4s %4zu %10.3f %10.3f %10.3f\n", s.name, kTrainBatch, "trainStep",
                        kImpl[impl], gp.forward.size() + gp.backward.size(), t1.median * 1e3,
                        t1.gpuMedian * 1e3, tp.median * 1e3);
            std::fflush(stdout);
        }
    }

    // A batching evaluator's view: many workers' leaves in one forward pass,
    // the whole pass in one command buffer, two in flight so the CPU encodes
    // the next while the GPU runs this one.
    std::printf("\n## pipelined inference: 2 command buffers in flight, one-cb forward passes\n");
    for (const NetSpec& s : kNets) {
        if (skipNet(s)) continue;
        for (int B : {144, 256})
        for (int impl : kGpuImpls) {
            const Plan p = makePlan(s, B, false);
            GpuPlan gp[2] = {bindPlan(gpu, p, impl), bindPlan(gpu, p, impl)};
            const int passes = 200;
            const auto start = Clock::now();
            id<MTLCommandBuffer> inflight[2] = {nil, nil};
            for (int i = 0; i < passes; ++i) {
                const int k = i & 1;
                if (inflight[k]) [inflight[k] waitUntilCompleted];
                Ctx c{[gpu.queue commandBuffer]};
                for (const auto& op : gp[k].forward) op(c);
                c.endCompute();
                [c.cb commit];
                inflight[k] = c.cb;
            }
            for (auto cb : inflight) if (cb) [cb waitUntilCompleted];
            const double el = secondsSince(start);
            std::printf("%-13s B=%3d %-4s %8.3f ms per pass  %9.0f evals/s\n", s.name, B, kImpl[impl], el / passes * 1e3,
                        passes * B / el);
            std::fflush(stdout);
        }
    }
}

// ---------------------------------------------------------------- seam backend

// A MetalBackend behind today's Backend interface: gemm only, operands copied
// into shared buffers (the Workspace's std::vectors are not page-aligned, so
// they cannot be wrapped without a copy), one commit + wait per call, im2col,
// col2im, BN and everything else left on the CPU.
class SeamBackend : public azgo::Backend {
public:
    SeamBackend(Gpu& gpu, int impl) : gpu_(gpu), impl_(impl) {}
    const char* name() const override {
        return impl_ == kMps ? "seam-mps" : impl_ == kSimd ? "seam-simd" : "seam-hand";
    }
    mutable double gemmSeconds = 0;
    mutable long calls = 0;
    void gemm(bool tA, bool tB, int M, int N, int K, float alpha, const float* A, const float* B,
              float beta, float* C) const override {
        const auto t0 = Clock::now();
        const size_t na = static_cast<size_t>(M) * K, nb = static_cast<size_t>(K) * N,
                     nc = static_cast<size_t>(M) * N;
        ensure(a_, na);
        ensure(b_, nb);
        ensure(c_, nc);
        std::memcpy(a_.contents, A, na * sizeof(float));
        std::memcpy(b_.contents, B, nb * sizeof(float));
        if (beta != 0.0f) std::memcpy(c_.contents, C, nc * sizeof(float));
        Gemm g{"", tA, tB, M, N, K, beta};
        Ctx c{[gpu_.queue commandBuffer]};
        if (impl_ == kMps) {
            // MPS objects are cheap to make relative to the call.
            encodeMps(c, makeMps(gpu_, g, alpha, a_, b_, c_));
        } else {
            encodeHandGemm(gpu_, c, g, alpha, a_, b_, c_, impl_ == kSimd);
        }
        c.endCompute();
        [c.cb commit];
        [c.cb waitUntilCompleted];
        std::memcpy(C, c_.contents, nc * sizeof(float));
        gemmSeconds += secondsSince(t0);
        ++calls;
    }
private:
    void ensure(__strong id<MTLBuffer>& b, size_t floats) const {
        if (!b || b.length < floats * sizeof(float)) b = gpu_.buffer(floats);
    }
    Gpu& gpu_;
    int impl_;
    mutable id<MTLBuffer> a_, b_, c_;
};

void sectionSeam() {
    Gpu gpu;
    std::printf("# section seam: the real Network driven through a Backend whose gemm runs on the GPU\n"
                "# (copy in, one command buffer, commit+wait, copy out); everything else on one CPU thread.\n");
    std::printf("%-13s %4s  %-9s %-9s %10s %10s %7s\n", "net", "B", "pass", "backend", "total ms", "gemm ms",
                "calls");
    for (const NetSpec& s : kNets) {
        if (skipNet(s)) continue;
        const azgo::NetConfig nc = netConfig(s);
        for (int impl : kGpuImpls) {
            SeamBackend be(gpu, impl);
            const azgo::Network net0(nc, 1);
            for (int B : kBatches) {
                azgo::Workspace ws(nc, B, be);
                NetInputs in(nc, B, 7);
                be.gemmSeconds = 0;
                be.calls = 0;
                const Timing t = timeCalls([&] {
                    net0.infer(ws, in.input.data(), B, in.logits.data(), in.values.data());
                    return 0.0;
                });
                const double perCallGemm = be.gemmSeconds / (t.calls + 2);
                std::printf("%-13s %4d  %-9s %-9s %10.3f %10.3f %7ld\n", s.name, B, "infer", be.name(),
                            t.median * 1e3, perCallGemm * 1e3, be.calls / (t.calls + 2));
                std::fflush(stdout);
            }
            azgo::Network net(nc, 1);
            azgo::Workspace ws(nc, kTrainBatch, be);
            NetInputs in(nc, kTrainBatch, 7);
            be.gemmSeconds = 0;
            be.calls = 0;
            const Timing t = timeCalls([&] {
                net.trainStep(ws, in.input.data(), in.pi.data(), in.z.data(), kTrainBatch);
                return 0.0;
            }, std::max(gSeconds, 1.0));
            std::printf("%-13s %4d  %-9s %-9s %10.3f %10.3f %7ld\n", s.name, kTrainBatch, "trainStep",
                        be.name(), t.median * 1e3, be.gemmSeconds / (t.calls + 2) * 1e3,
                        be.calls / (t.calls + 2));
            std::fflush(stdout);
        }
    }
}

// ---------------------------------------------------------------- determinism

struct Diff { double maxAbs = 0, maxRel = 0, refMax = 0; };
// maxRel is relative to max(|ref|, 1e-3 * max|ref|), so near-zero entries do not dominate.
Diff diff(const float* x, const double* ref, size_t count) {
    Diff d;
    for (size_t i = 0; i < count; ++i) d.refMax = std::max(d.refMax, std::fabs(ref[i]));
    const double floor = 1e-3 * d.refMax;
    for (size_t i = 0; i < count; ++i) {
        const double e = std::fabs(x[i] - ref[i]);
        d.maxAbs = std::max(d.maxAbs, e);
        d.maxRel = std::max(d.maxRel, e / std::max(std::fabs(ref[i]), floor));
    }
    return d;
}

// Double-precision reference C = op(A) op(B) (alpha 1, beta 0).
std::vector<double> refGemm(const Gemm& g, const float* A, const float* B) {
    std::vector<double> C(static_cast<size_t>(g.M) * g.N, 0.0);
    for (int i = 0; i < g.M; ++i)
        for (int p = 0; p < g.K; ++p) {
            const double a = g.tA ? A[static_cast<size_t>(p) * g.M + i] : A[static_cast<size_t>(i) * g.K + p];
            double* c = C.data() + static_cast<size_t>(i) * g.N;
            if (g.tB) {
                for (int j = 0; j < g.N; ++j) c[j] += a * B[static_cast<size_t>(j) * g.K + p];
            } else {
                const float* b = B + static_cast<size_t>(p) * g.N;
                for (int j = 0; j < g.N; ++j) c[j] += a * b[j];
            }
        }
    return C;
}

// Runs one gemm (beta 0) with the given implementation.
std::vector<float> runGemm(Gpu& gpu, const azgo::Backend& cpu, int impl, Gemm g, const std::vector<float>& A,
                           const std::vector<float>& B) {
    g.beta = 0;
    std::vector<float> C(static_cast<size_t>(g.M) * g.N);
    if (impl == kAccel) {
        cpu.gemm(g.tA, g.tB, g.M, g.N, g.K, 1.0f, A.data(), B.data(), 0.0f, C.data());
        return C;
    }
    id<MTLBuffer> a = gpu.buffer(A.size()), b = gpu.buffer(B.size()), c = gpu.buffer(C.size());
    std::memcpy(a.contents, A.data(), A.size() * sizeof(float));
    std::memcpy(b.contents, B.data(), B.size() * sizeof(float));
    std::memset(c.contents, 0xff, C.size() * sizeof(float));   // NaN: catches unwritten outputs
    Ctx ctx{[gpu.queue commandBuffer]};
    if (impl != kMps) encodeHandGemm(gpu, ctx, g, 1.0f, a, b, c, impl == kSimd);
    else encodeMps(ctx, makeMps(gpu, g, 1.0f, a, b, c));
    ctx.endCompute();
    [ctx.cb commit];
    [ctx.cb waitUntilCompleted];
    std::memcpy(C.data(), c.contents, C.size() * sizeof(float));
    return C;
}


void sectionDet() {
    Gpu gpu;
    auto cpu = azgo::makeBackend("blas", 1);
    std::printf("# section det: determinism and accuracy\n");
    const NetSpec& s = kNets[1];   // 9x9 6x96
    const int F = s.filters, A = s.n * s.n;

    // 1. Repeat runs, and accuracy against a double-precision reference.
    std::printf("\n## repeated runs (20) and error vs a double-precision reference, %s, B=256\n", s.name);
    std::printf("%-14s %-10s %-10s %12s %12s %12s  %s\n", "shape", "impl", "repeats", "max|ref|", "max abs err",
                "max rel err", "elements != accelerate");
    for (const Gemm& g : convShapes(s, kTrainBatch)) {
        std::vector<float> Am(static_cast<size_t>(g.M) * g.K), Bm(static_cast<size_t>(g.K) * g.N);
        fillRandom(Am.data(), Am.size(), 3, 0.05f);
        fillRandom(Bm.data(), Bm.size(), 4, 1.0f);
        const std::vector<double> ref = refGemm(g, Am.data(), Bm.data());
        const std::vector<float> accel = runGemm(gpu, *cpu, kAccel, g, Am, Bm);
        for (int impl : {kHand, kSimd, kMps, kAccel}) {
            const std::vector<float> first = runGemm(gpu, *cpu, impl, g, Am, Bm);
            size_t differ = 0;
            for (size_t i = 0; i < first.size(); ++i)
                differ += std::memcmp(&first[i], &accel[i], sizeof(float)) != 0;
            int same = 0;
            for (int r = 0; r < 20; ++r) {
                const std::vector<float> again = runGemm(gpu, *cpu, impl, g, Am, Bm);
                same += std::memcmp(first.data(), again.data(), first.size() * sizeof(float)) == 0;
            }
            const Diff d = diff(first.data(), ref.data(), ref.size());
            std::printf("%-14s %-10s %4d/20    %12.4g %12.3g %12.3g  %zu of %zu\n", g.label.c_str(), kImpl[impl],
                        same, d.refMax, d.maxAbs, d.maxRel, differ, first.size());
            std::fflush(stdout);
        }
        if (splitCount(g) > 1) {   // the same shape without split-K: one fma chain over all of k
            gNoSplit = true;
            for (int impl : {kHand, kSimd}) {
                const std::vector<float> first = runGemm(gpu, *cpu, impl, g, Am, Bm);
                size_t differ = 0;
                for (size_t i = 0; i < first.size(); ++i)
                    differ += std::memcmp(&first[i], &accel[i], sizeof(float)) != 0;
                const Diff d = diff(first.data(), ref.data(), ref.size());
                std::printf("%-14s %-10s %-10s %12.4g %12.3g %12.3g  %zu of %zu\n", g.label.c_str(),
                            impl == kHand ? "hand-nosplit" : "simd-nosplit", "", d.refMax, d.maxAbs, d.maxRel,
                            differ, first.size());
            }
            gNoSplit = false;
            // Their cost: 20 copies in one command buffer, GPU time per gemm.
            id<MTLBuffer> a = gpu.buffer(Am.size()), b = gpu.buffer(Bm.size()),
                          c = gpu.buffer(static_cast<size_t>(g.M) * g.N);
            for (int impl : {kHand, kSimd})
                for (int nosplit = 0; nosplit < 2; ++nosplit) {
                    gNoSplit = nosplit;
                    std::vector<std::function<void(Ctx&)>> ops(20, [&gpu, g, a, b, c, impl](Ctx& ctx) {
                        encodeHandGemm(gpu, ctx, g, 1.0f, a, b, c, impl == kSimd);
                    });
                    const Timing t = timeCalls([&] { return runOneBuffer(gpu, ops); });
                    std::printf("%-14s %-10s %s: %.3f ms per gemm (GPU)\n", g.label.c_str(), kImpl[impl],
                                nosplit ? "no split-K   " : "split-K (S=20)", t.gpuMedian / 20 * 1e3);
                }
            gNoSplit = false;
        }
    }

    // 2. Column independence: sample b's 81 columns computed inside a batch
    // of 256 vs alone (N = 81) vs inside a batch of 8.
    std::printf("\n## a sample's columns inside a large batch vs alone (forward block conv %s, K=%d)\n", s.name, 9 * F);
    {
        const int K = 9 * F, Bfull = 256;
        Gemm full{"", false, false, F, Bfull * A, K, 0};
        std::vector<float> W(static_cast<size_t>(F) * K), col(static_cast<size_t>(K) * full.N);
        fillRandom(W.data(), W.size(), 5, 0.05f);
        fillRandom(col.data(), col.size(), 6, 1.0f);
        auto columnsOf = [&](int b0, int count) {   // samples [b0, b0+count) as a K x (count*A) matrix
            std::vector<float> sub(static_cast<size_t>(K) * count * A);
            for (int k = 0; k < K; ++k)
                std::memcpy(sub.data() + static_cast<size_t>(k) * count * A,
                            col.data() + static_cast<size_t>(k) * full.N + static_cast<size_t>(b0) * A,
                            static_cast<size_t>(count) * A * sizeof(float));
            return sub;
        };
        std::printf("%-10s %-24s %s\n", "impl", "samples checked", "bitwise equal to the B=256 result");
        for (int impl : {kHand, kSimd, kMps, kAccel}) {
            const std::vector<float> Cfull = runGemm(gpu, *cpu, impl, full, W, col);
            int equalAlone = 0, equalIn8 = 0, checked = 0;
            double worst = 0;
            for (int b : {0, 1, 7, 100, 200, 255}) {
                ++checked;
                Gemm one{"", false, false, F, A, K, 0};
                const std::vector<float> C1 = runGemm(gpu, *cpu, impl, one, W, columnsOf(b, 1));
                bool eq = true;
                for (int i = 0; i < F; ++i)
                    for (int j = 0; j < A; ++j) {
                        const float x = Cfull[static_cast<size_t>(i) * full.N + static_cast<size_t>(b) * A + j];
                        const float y = C1[static_cast<size_t>(i) * A + j];
                        if (std::memcmp(&x, &y, sizeof x) != 0) eq = false;
                        worst = std::max(worst, static_cast<double>(std::fabs(x - y)));
                    }
                equalAlone += eq;
                const int b8 = std::min(b, Bfull - 8);   // sample b inside the batch of 8 starting at b8
                Gemm eight{"", false, false, F, 8 * A, K, 0};
                const std::vector<float> C8 = runGemm(gpu, *cpu, impl, eight, W, columnsOf(b8, 8));
                bool eq8 = true;
                for (int i = 0; i < F; ++i)
                    for (int j = 0; j < A; ++j) {
                        const float x = Cfull[static_cast<size_t>(i) * full.N + static_cast<size_t>(b) * A + j];
                        const float y = C8[static_cast<size_t>(i) * 8 * A + static_cast<size_t>(b - b8) * A + j];
                        if (std::memcmp(&x, &y, sizeof x) != 0) eq8 = false;
                    }
                equalIn8 += eq8;
            }
            std::printf("%-10s b in {0,1,7,100,200,255}  alone (N=81): %d/%d, in a batch of 8: %d/%d, max |diff| %.3g\n",
                        kImpl[impl], equalAlone, checked, equalIn8, checked, worst);
            std::fflush(stdout);
        }
    }

    // 3. End to end through the real Network: each position's logits and
    // value alone vs inside a batch, and a training step repeated.
    std::printf("\n## Network::infer, positions alone vs in a batch of 256 (%s)\n", s.name);
    const azgo::NetConfig nc = netConfig(s);
    SeamBackend seamHand(gpu, kHand), seamSimd(gpu, kSimd), seamMps(gpu, kMps);
    constexpr int kBackends = 4, kRef = 3;   // the last one, blas, is the reference
    const azgo::Backend* backends[kBackends] = {&seamHand, &seamSimd, &seamMps, cpu.get()};
    const char* names[kBackends] = {"seam-hand", "seam-simd", "seam-mps", "blas"};
    std::vector<float> logitsByBackend[kBackends];
    for (int k = 0; k < kBackends; ++k) {
        const azgo::Network net(nc, 1);
        NetInputs in(nc, 256, 9);
        azgo::Workspace ws(nc, 256, *backends[k]);
        net.infer(ws, in.input.data(), 256, in.logits.data(), in.values.data());
        logitsByBackend[k] = in.logits;
        // Position b alone (B = 1) and inside the batch of 8 starting at b8,
        // against the same position inside the batch of 256.
        int equal1 = 0, equal8 = 0, checked = 0;
        double worst = 0;
        azgo::Workspace ws1(nc, 1, *backends[k]), ws8(nc, 8, *backends[k]);
        const size_t stride = static_cast<size_t>(nc.planes()) * A;
        for (int b : {0, 3, 128, 255}) {
            const float* lb = in.logits.data() + static_cast<size_t>(b) * nc.actions();
            std::vector<float> l1(nc.actions()), v1(1);
            net.infer(ws1, in.input.data() + b * stride, 1, l1.data(), v1.data());
            equal1 += std::memcmp(l1.data(), lb, l1.size() * sizeof(float)) == 0 &&
                      std::memcmp(&v1[0], &in.values[b], sizeof(float)) == 0;
            for (int a = 0; a < nc.actions(); ++a) worst = std::max(worst, static_cast<double>(std::fabs(l1[a] - lb[a])));
            const int b8 = std::min(b, 256 - 8);
            std::vector<float> l8(static_cast<size_t>(8) * nc.actions()), v8(8);
            net.infer(ws8, in.input.data() + b8 * stride, 8, l8.data(), v8.data());
            equal8 += std::memcmp(l8.data() + static_cast<size_t>(b - b8) * nc.actions(), lb,
                                  nc.actions() * sizeof(float)) == 0 &&
                      std::memcmp(&v8[b - b8], &in.values[b], sizeof(float)) == 0;
            ++checked;
        }
        std::printf("%-10s logits+value bitwise equal to batch-256 result: alone %d/%d, in a batch of 8 %d/%d"
                    "  (max |logit diff| alone %.3g)\n", names[k], equal1, checked, equal8, checked, worst);
        std::fflush(stdout);
    }
    for (int k = 0; k < kRef; ++k) {
        double worst = 0, scale = 0;
        for (size_t i = 0; i < logitsByBackend[kRef].size(); ++i) {
            worst = std::max(worst, static_cast<double>(std::fabs(logitsByBackend[k][i] - logitsByBackend[kRef][i])));
            scale = std::max(scale, static_cast<double>(std::fabs(logitsByBackend[kRef][i])));
        }
        std::printf("%-10s vs blas: max |logit diff| %.3g (max |logit| %.3g)\n", names[k], worst, scale);
    }

    std::printf("\n## Network::trainStep from the same seed, 3 times per backend (%s, B=256): gradient hash\n", s.name);
    std::vector<std::vector<float>> grads(kBackends);
    for (int k = 0; k < kBackends; ++k) {
        std::vector<uint64_t> hashes;
        azgo::LossStats loss;
        for (int r = 0; r < 3; ++r) {
            azgo::Network net(nc, 1);
            NetInputs in(nc, 256, 9);
            azgo::Workspace ws(nc, 256, *backends[k]);
            loss = net.trainStep(ws, in.input.data(), in.pi.data(), in.z.data(), 256);
            std::vector<float> all;
            for (const azgo::ParamView& p : net.params()) all.insert(all.end(), p.grad, p.grad + p.size);
            hashes.push_back(fnv(all.data(), all.size()));
            grads[k] = all;
        }
        std::printf("%-10s loss p %.7f v %.7f  hashes %016llx %016llx %016llx  %s\n", names[k], loss.policyLoss,
                    loss.valueLoss, (unsigned long long)hashes[0], (unsigned long long)hashes[1],
                    (unsigned long long)hashes[2], hashes[0] == hashes[1] && hashes[1] == hashes[2] ? "identical" : "DIFFER");
        std::fflush(stdout);
    }
    for (int k = 0; k < kRef; ++k) {
        double worst = 0, scale = 0;
        for (size_t i = 0; i < grads[kRef].size(); ++i) {
            worst = std::max(worst, static_cast<double>(std::fabs(grads[k][i] - grads[kRef][i])));
            scale = std::max(scale, static_cast<double>(std::fabs(grads[kRef][i])));
        }
        std::printf("%-10s vs blas: max |grad diff| %.3g (max |grad| %.3g)\n", names[k], worst, scale);
    }
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        std::string section;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--seconds" && i + 1 < argc) gSeconds = std::atof(argv[++i]);
            else if (a == "--net" && i + 1 < argc) gNetFilter = argv[++i];
            else if (section.empty() && a[0] != '-') section = a;
            else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
        }
        if (section == "cpu1") sectionCpu(false);
        else if (section == "cpuall") sectionCpu(true);
        else if (section == "gpu") sectionGpu();
        else if (section == "seam") sectionSeam();
        else if (section == "det") sectionDet();
        else {
            std::fprintf(stderr, "usage: metal_bench cpu1|cpuall|gpu|seam|det [--seconds S] [--net 6x96]\n");
            return 2;
        }
    }
    return 0;
}
