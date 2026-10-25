// GPU tests for backend = cuda16: the 16-bit tensor-core convolution against
// a double-precision reference on the same 16-bit operands, the tolerance
// tier of the whole network against double precision, and every property of
// the cuda backend that the inference hub and the determinism contract rest
// on, bitwise: batch independence, tile shapes, repeats, the runner, the hub
// and end-to-end self-play, arena and run files.
//
// They run only in a build configured with -DAZ_USE_CUDA=ON on a machine with
// an NVIDIA GPU of sm_80 or newer. Without CUDA built, or without such a GPU,
// main() returns 77 (CTest: Skipped). Design:
// docs/superpowers/specs/2026-10-24-cuda-fp16-design.md.
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
#include "azgo/cuda16.hpp"
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
            std::printf("test_cuda16: %s: maxErr %.3g above the ceiling %.0e\n", what, e, ceiling);
            std::fflush(stdout);
            assert(false);
        }
        worst = std::max(worst, e);
    }
    ~MaxErr() { std::printf("test_cuda16: %-28s maxErr %.2e (ceiling %.0e)\n", what, worst, ceiling); }
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

Outputs gpuInfer(Cuda16Inference& net, const NetConfig& cfg, const std::vector<float>& x, int B) {
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


// The tolerance tier (the design's "Accuracy"; docs/results/cuda16.txt):
// against the network in double precision, per batch. On real positions of
// the 13x13 and 19x19 runs' checkpoints cuda16 measured logits 2.1e-3 and
// 2.6e-3, values 2.7e-3 and 3.0e-3.
constexpr double kTierLogits = 5e-3;     // max|x - ref| / max|ref|
constexpr double kTierValue = 1e-2;      // max |x - ref|
constexpr double kTierOwnership = 1e-2;  // max |x - ref|

// ---------------------------------------------------------------- the context

// The description is cuda's with the 16-bit kernels' hash in place of the
// fp32 kernels': the same GPU, driver and runtime, another hash.
void test_available_and_description() {
    std::string why = "unchanged";
    assert(cuda16Available(&why));
    assert(why == "unchanged");
    const std::string d = cuda16Description(), d32 = cudaDescription();
    std::printf("test_cuda16: %s\n", d.c_str());
    static const std::regex shape(
        "^.+ sm_[0-9]+, driver API [0-9]+\\.[0-9]+, runtime [0-9]+\\.[0-9]+, fp16 kernels [0-9a-f]{8}$");
    assert(std::regex_match(d, shape));
    const size_t at = d32.rfind(", kernels ");
    assert(at != std::string::npos && d.compare(0, at, d32, 0, at) == 0);
    assert(d.substr(d.size() - 8) != d32.substr(d32.size() - 8));
    assert(d.find(';') == std::string::npos);
    assert(cuda_probe::deviceArchitecture() >= 80);
}


// ---------------------------------------------------------------- the 16-bit convolution

Conv2d randomConv(Rng& rng, int cin, int cout) {
    Conv2d conv(cin, cout, 3);
    for (float& w : conv.w) w = static_cast<float>((rng.uniform() * 2.0 - 1.0) * 0.1);
    return conv;
}

std::vector<float> rounded(std::vector<float> v) {
    cuda16_probe::roundTo16(v.data(), v.size());
    return v;
}

// The reference for conv16: the same arithmetic in double on the operands
// the GPU sees (weights, input and skip rounded to 16 bits), the batch norm
// folded as BatchNorm2d::forwardInfer does in float. y [cout][B * A]. bound
// receives, per output, the scale of its rounding errors: |scale| times the
// sum of |products|, plus |shift| and |skip|.
std::vector<double> conv16Reference(const Conv2d& conv, const BatchNorm2d& bn, const std::vector<float>& x,
                                    const std::vector<float>* skip, int B, int n, std::vector<double>& bound) {
    const int A = n * n, cin = conv.cin, cout = conv.cout;
    const std::vector<float> w = rounded(conv.w), xr = rounded(x);
    const std::vector<float> sr = skip ? rounded(*skip) : std::vector<float>();
    std::vector<double> y(static_cast<size_t>(cout) * B * A);
    bound.assign(y.size(), 0.0);
    for (int m = 0; m < cout; ++m) {
        const float scale = bn.gamma[m] / std::sqrt(bn.runVar[m] + BatchNorm2d::kEps);
        const float shift = bn.beta[m] - scale * bn.runMean[m];
        for (int b = 0; b < B; ++b)
            for (int p = 0; p < A; ++p) {
                double sum = 0, mag = 0;
                for (int c = 0; c < cin; ++c)
                    for (int r = 0; r < 9; ++r) {
                        const int yy = p / n + r / 3 - 1, xx = p % n + r % 3 - 1;
                        if (yy < 0 || yy >= n || xx < 0 || xx >= n) continue;
                        const double prod = static_cast<double>(w[(static_cast<size_t>(m) * cin + c) * 9 + r]) *
                                            xr[(static_cast<size_t>(c) * B + b) * A + yy * n + xx];
                        sum += prod;
                        mag += std::fabs(prod);
                    }
                const size_t at = (static_cast<size_t>(m) * B + b) * A + p;
                double v = static_cast<double>(scale) * sum + shift;
                double s = 0;
                if (skip) s = sr[at];
                v += s;
                y[at] = v > 0 ? v : 0;
                bound[at] = std::fabs(scale) * mag + std::fabs(shift) + std::fabs(s);
            }
    }
    return y;
}

// conv16 within 2^-16 of the scale of its rounding (the tensor core's
// accumulation, the epilogue's two roundings and the skip's addition), on
// stem and block shapes, padded channel counts and every board size; and its
// 16-bit output is its fp32 output rounded to 16 bits, bitwise.
void test_conv16_matches_double() {
    struct Shape { int cin, cout, n, B; };
    const Shape shapes[] = {{18, 32, 5, 3}, {18, 96, 9, 2}, {96, 96, 13, 2}, {18, 128, 19, 1},
                            {128, 128, 19, 1}, {40, 48, 7, 3}, {8, 8, 5, 2}, {64, 64, 9, 2}};
    Rng rng(1601);
    double worst = 0;
    for (const Shape& s : shapes)
        for (bool withSkip : {false, true}) {
            const int A = s.n * s.n;
            const Conv2d conv = randomConv(rng, s.cin, s.cout);
            const BatchNorm2d bn = randomBn(rng, s.cout);
            const std::vector<float> x = randomVector(rng, static_cast<size_t>(s.cin) * s.B * A);
            const std::vector<float> skip = randomVector(rng, static_cast<size_t>(s.cout) * s.B * A);
            std::vector<float> y(static_cast<size_t>(s.cout) * s.B * A), y16(y.size());
            cuda16_probe::conv16(conv, bn, x.data(), withSkip ? skip.data() : nullptr, s.B, s.n, y.data(), y16.data());
            std::vector<double> bound;
            const std::vector<double> ref = conv16Reference(conv, bn, x, withSkip ? &skip : nullptr, s.B, s.n, bound);
            for (size_t i = 0; i < y.size(); ++i) {
                const double e = std::fabs(y[i] - ref[i]) / (bound[i] + 1e-30);
                worst = std::max(worst, e);
                if (!(e <= std::ldexp(1.0, -16))) {
                    std::printf("test_cuda16: conv16 %d->%d %dx%d: y %.9g, double %.9g (scale %.3g)\n", s.cin, s.cout,
                                s.n, s.n, y[i], ref[i], bound[i]);
                    std::fflush(stdout);
                    assert(false);
                }
            }
            assert(bitwiseEqual(y16.data(), rounded(y).data(), y.size()));
        }
    std::printf("test_cuda16: conv16 vs double on 16-bit operands: worst %.2e of the rounding scale (bound 2^-16)\n",
                worst);
}

// Every channel tile gives the same bits (each output is the same chain of
// mma steps in k order, whichever block and warp computes it).
void test_conv16_tiles_agree() {
    Rng rng(1602);
    for (int cout : {128, 96, 64}) {
        const int cin = 96, n = 9, B = 5, A = n * n;
        const Conv2d conv = randomConv(rng, cin, cout);
        const BatchNorm2d bn = randomBn(rng, cout);
        const std::vector<float> x = randomVector(rng, static_cast<size_t>(cin) * B * A);
        const std::vector<float> skip = randomVector(rng, static_cast<size_t>(cout) * B * A);
        std::vector<float> first;
        for (int tile : {2, 4, 6, 8}) {
            if (cout % (16 * tile) != 0) continue;
            std::vector<float> y(static_cast<size_t>(cout) * B * A), y16(y.size());
            cuda16_probe::conv16(conv, bn, x.data(), skip.data(), B, n, y.data(), y16.data(), tile);
            y.insert(y.end(), y16.begin(), y16.end());
            if (first.empty()) first = y;
            assert(same(y, first));
        }
    }
}

// A position's outputs, 16-bit and fp32, are the same bits alone and in
// batches of 8, 144 and 256 at the first, a middle and the last slot, among
// random companions, for the 13x13 and 19x19 trunk shapes and the stem.
void test_conv16_batch_independent() {
    Rng rng(1603);
    struct Shape { int cin, cout, n; };
    for (const Shape& s : {Shape{96, 96, 13}, Shape{128, 128, 19}, Shape{18, 128, 19}}) {
        const int A = s.n * s.n;
        const Conv2d conv = randomConv(rng, s.cin, s.cout);
        const BatchNorm2d bn = randomBn(rng, s.cout);
        const std::vector<float> one = randomVector(rng, static_cast<size_t>(s.cin) * A);
        const std::vector<float> oneSkip = randomVector(rng, static_cast<size_t>(s.cout) * A);
        std::vector<float> alone(static_cast<size_t>(s.cout) * A), alone16(alone.size());
        cuda16_probe::conv16(conv, bn, one.data(), oneSkip.data(), 1, s.n, alone.data(), alone16.data());
        for (int B : {8, 144, 256})
            for (int at : {0, B / 2 + 1, B - 1}) {
                std::vector<float> x = randomVector(rng, static_cast<size_t>(s.cin) * B * A);
                std::vector<float> skip = randomVector(rng, static_cast<size_t>(s.cout) * B * A);
                for (int c = 0; c < s.cin; ++c)
                    std::copy_n(&one[static_cast<size_t>(c) * A], A, &x[(static_cast<size_t>(c) * B + at) * A]);
                for (int c = 0; c < s.cout; ++c)
                    std::copy_n(&oneSkip[static_cast<size_t>(c) * A], A, &skip[(static_cast<size_t>(c) * B + at) * A]);
                std::vector<float> y(static_cast<size_t>(s.cout) * B * A), y16(y.size());
                cuda16_probe::conv16(conv, bn, x.data(), skip.data(), B, s.n, y.data(), y16.data());
                for (int c = 0; c < s.cout; ++c) {
                    assert(bitwiseEqual(&y[(static_cast<size_t>(c) * B + at) * A], &alone[static_cast<size_t>(c) * A], A));
                    assert(bitwiseEqual(&y16[(static_cast<size_t>(c) * B + at) * A], &alone16[static_cast<size_t>(c) * A],
                                        A));
                }
            }
    }
}

// fp16's range: a value past its largest finite one saturates there
// instead of becoming infinity.
void test_conv16_saturates() {
    Conv2d conv(32, 32, 3);
    std::fill(conv.w.begin(), conv.w.end(), 1.0f);
    BatchNorm2d bn(32);   // identity: gamma 1, beta 0, mean 0, var 1
    const int n = 5, A = n * n;
    const std::vector<float> x(static_cast<size_t>(32) * A, 1000.0f);   // a sum of up to 288 000
    std::vector<float> y(static_cast<size_t>(32) * A), y16(y.size());
    cuda16_probe::conv16(conv, bn, x.data(), nullptr, 1, n, y.data(), y16.data());
    for (size_t i = 0; i < y.size(); ++i) {
        assert(std::isfinite(y16[i]) && y16[i] <= 65504.0f);
        if (y[i] > 65504.0f) assert(y16[i] == 65504.0f);
    }
    assert(*std::max_element(y.begin(), y.end()) > 65504.0f);   // the test reached the clamp
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


// ---------------------------------------------------------------- the whole network

Outputs gpuInfer32(CudaInference& net, const NetConfig& cfg, const std::vector<float>& x, int B) {
    Outputs o;
    o.logits.resize(static_cast<size_t>(B) * cfg.actions());
    o.values.resize(B);
    if (cfg.ownership) o.ownership.resize(static_cast<size_t>(B) * cfg.area());
    net.infer(x.data(), B, o.logits.data(), o.values.data(), cfg.ownership ? o.ownership.data() : nullptr);
    return o;
}

// The 16-bit tier: the presets' shapes with random weights (whose
// activations are larger than a trained network's) against double
// precision, the logits as maxErr, value and ownership as absolute gaps
// (OutputsGap). The ceilings are the design's tier (docs/results/cuda16.txt
// has the measurements on real checkpoints they come from). The 32-bit cuda
// backend's gaps are printed beside them; cuda16 must not be bitwise cuda.
void test_network_presets_vs_double() {
    struct Tier { double logits, values, ownership; };
    const Tier tier = {kTierLogits, kTierValue, kTierOwnership};
    const NetConfig presets[] = {netConfig(5, 3, 32, false, 64),    netConfig(9, 6, 64, false, 64),
                                 netConfig(9, 6, 96, false, 64),    netConfig(13, 6, 96, false, 64),
                                 netConfig(19, 10, 128, false, 256), netConfig(19, 2, 128, true, 256)};
    Rng rng(1611);
    auto blas = makeBackend("blas");
    for (const NetConfig& cfg : presets) {
        // Random weights, batch norms calibrated on positions as a run's
        // generation 0 is (bn_calibrate): activations of a trained network's
        // scale. Without the calibration a random 19x19 10x128 network's
        // activations grow block by block, and its value is 3e-2 from double
        // in 16 bits; no trained network behaves so.
        Network net = randomNetwork(cfg, 40);
        {
            const std::vector<float> sample = randomPositions(rng, cfg, 64);
            Workspace ws(cfg, 64, *blas);
            net.calibrateBatchNorm(ws, sample.data(), 64);
        }
        Cuda16Inference gpu(net, 16);
        CudaInference gpu32(net, 16);
        OutputsGap worst, worst32;
        bool differs = false;
        for (int B : cfg.boardSize == 19 ? std::vector<int>{1, 4} : std::vector<int>{1, 7, 16}) {
            const std::vector<float> x = randomPositions(rng, cfg, B);
            const Outputs ref = doubleInfer(net, x, B);
            const Outputs got = gpuInfer(gpu, cfg, x, B), got32 = gpuInfer32(gpu32, cfg, x, B);
            differs = differs || !same(got.logits, got32.logits);
            const OutputsGap g = outputsGap(got, ref), g32 = outputsGap(got32, ref);
            if (!(g.logits <= tier.logits && g.values <= tier.values && g.ownership <= tier.ownership)) {
                std::printf("test_cuda16: %dx%d %dx%d batch %d: gaps logits %.3g, values %.3g, ownership %.3g; "
                            "tier %.0e, %.0e, %.0e\n",
                            cfg.boardSize, cfg.boardSize, cfg.blocks, cfg.filters, B, g.logits, g.values, g.ownership,
                            tier.logits, tier.values, tier.ownership);
                std::fflush(stdout);
                assert(false);
            }
            worst.logits = std::max(worst.logits, g.logits);
            worst.values = std::max(worst.values, g.values);
            worst.ownership = std::max(worst.ownership, g.ownership);
            worst32.logits = std::max(worst32.logits, g32.logits);
            worst32.values = std::max(worst32.values, g32.values);
        }
        assert(differs);   // 16-bit arithmetic, not the fp32 pass under another name
        std::printf("test_cuda16:   %2dx%-2d %2dx%-3d vs double: logits %.1e (cuda %.1e), values %.1e (cuda %.1e)%s\n",
                    cfg.boardSize, cfg.boardSize, cfg.blocks, cfg.filters, worst.logits, worst32.logits, worst.values,
                    worst32.values, cfg.ownership ? ", ownership checked" : "");
    }
}

// What the inference hub rests on, for the whole network: a position's
// logits, value and ownership are the same bits alone, in batches of 8, 144
// and 256, at the first or the last slot, among random companions and with
// the batch permuted.
void test_network_batch_independent() {
    const NetConfig cfg = netConfig(13, 6, 96, true, 64);
    const Network net = randomNetwork(cfg, 602);
    Cuda16Inference gpu(net, 256);
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
    Cuda16Inference gpu(net, 32);
    Rng rng(605);
    const std::vector<float> x = randomPositions(rng, cfg, 32);
    const Outputs first = gpuInfer(gpu, cfg, x, 32);
    for (int r = 0; r < 20; ++r) {
        const Outputs again = gpuInfer(gpu, cfg, x, 32);
        assert(same(again.logits, first.logits) && same(again.values, first.values) &&
               same(again.ownership, first.ownership));
    }
    // A second copy of the network gives the same bits as the first.
    Cuda16Inference other(net, 32);
    const Outputs second = gpuInfer(other, cfg, x, 32);
    assert(same(second.logits, first.logits) && same(second.values, first.values));
}

// A tensor the GPU layout does not know (a layer added on the CPU without a
// GPU port) is refused at upload, by name.
void test_unknown_tensor_refused() {
    const Network plain(netConfig(9, 1, 8), 1), withHead(netConfig(9, 1, 8, true), 1);
    cuda16_probe::checkUpload(plain, plain);
    bool threw = false;
    try {
        cuda16_probe::checkUpload(plain, withHead);
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
    cuda16_probe::checkForwardReadsEveryTensor(net, withHead);
    bool threw = false;
    try {
        cuda16_probe::checkForwardReadsEveryTensor(net, plain);
    } catch (const std::logic_error& e) {
        const std::string what = e.what();
        threw = what.find("ownership.conv.w") != std::string::npos &&
                what.find("ownership.bn.runVar") != std::string::npos;
    }
    assert(threw);
}


// The workspace is smaller than the cuda backend's (16-bit trunk buffers),
// and the pass's speed is printed.
void test_workspace_size_and_speed() {
    for (const NetConfig& cfg : {netConfig(13, 6, 96, false, 64), netConfig(19, 10, 128, false, 256)}) {
        const Network net = randomNetwork(cfg, 606);
        Cuda16Inference gpu(net, 256);
        CudaInference gpu32(net, 256);
        const double mb = static_cast<double>(gpu.bytes()) / 1e6, mb32 = static_cast<double>(gpu32.bytes()) / 1e6;
        assert(mb < mb32);
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
        std::printf("test_cuda16: %dx%d %dx%d at batch 256: workspace %.1f MB (cuda %.1f MB), %.0f evals/s "
                    "synchronous (not asserted)\n",
                    cfg.boardSize, cfg.boardSize, cfg.blocks, cfg.filters, mb, mb32, passes * 256 / s);
    }
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
        Cuda16Inference alone(net, 1);
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
        auto runner = makeCuda16Runner(net, 40, slots);
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
    auto runner = makeCuda16Runner(net, 20, 2);
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
            makeCuda16Runner(net, 8, bad);
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
    cuda16_probe::setCopyOutDelay(2000);
    auto runner = makeCuda16Runner(net, 8, 2);
    for (int round = 0; round < 6; ++round) {
        const int slot = round % 2, first = 8 * round;
        std::copy_n(pool.input.data() + first * in, 8 * in, runner->input(slot));
        assert(runSlot(*runner, slot, 8).empty());
        assert(bitwiseEqual(runner->logits(slot), pool.logits.data() + first * L, 8 * L));
        assert(bitwiseEqual(runner->values(slot), pool.values.data() + first, 8));
    }
    cuda16_probe::setCopyOutDelay(0);
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
                    InferenceHub hub(makeCuda16Runner(net, maxBatch, slots), cfg, o);
                    std::atomic<long long> positions{0};
                    std::vector<std::thread> threads;
                    for (int t = 0; t < clients; ++t) {
                        threads.emplace_back([&, t] {
                            auto ev = hub.makeEvaluator(8);
                            Rng rng(deriveSeed(705, static_cast<uint64_t>(t), chaos, static_cast<uint64_t>(maxBatch)));
                            std::vector<float> x(8 * in), logits(8 * L), values(8);
                            for (int r = 0; r < 20; ++r) {   // test_cuda runs 50; this keeps the GPU time short
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
    const long long before = cuda16_probe::passesCompleted();
    cuda16_probe::failPassesAfter(3);
    std::atomic<int> failures{0}, successes{0};
    {
        InferenceHubOptions o;
        o.maxBatch = 8;
        InferenceHub hub(makeCuda16Runner(net, 8), cfg, o);
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
                        assert(std::string(e.what()).rfind("GPU inference failed: cuda16: ", 0) == 0);
                        ++failures;
                        return;   // the hub has failed for good
                    }
                }
            });
        for (auto& th : threads) th.join();
    }
    cuda16_probe::failPassesAfter(0);
    assert(failures.load() == 8);   // every thread saw the failure
    assert(successes.load() >= 3 && cuda16_probe::passesCompleted() > before);
    // The next runner is healthy again.
    auto runner = makeCuda16Runner(net, 1);
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
        makeCuda16Runner(net, 1 << 20);
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        std::printf("test_cuda16: %s\n", what.c_str());
        threw = what.find("lower inference_batch") != std::string::npos && what.find("MB") != std::string::npos;
    }
    assert(threw);
    cuda16_probe::setMemoryLimit(size_t(10) << 20);   // 10 MB
    threw = false;
    try {
        checkCuda16InferenceMemory(cfg, 64, 1);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("lower inference_batch") != std::string::npos;
    }
    assert(threw);
    // Groups held at once (pipeline = on) are added up.
    threw = false;
    try {
        checkCuda16InferenceMemory(cfg, {GpuHubs{1, 2, 64}, GpuHubs{4, 1, 64}});
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("lower inference_batch") != std::string::npos;
    }
    assert(threw);
    cuda16_probe::setMemoryLimit(0);
    checkCuda16InferenceMemory(cfg, 64, 1);   // fits the real GPU
    checkCuda16InferenceMemory(cfg, {GpuHubs{1, 2, 64}, GpuHubs{4, 1, 64}});
    auto ok = makeCuda16Runner(net, 8);
    assert(ok->slots() == kInferenceSlots && ok->maxBatch() == 8);
}

// No thread outlives a runner, including runners destroyed with passes in flight.
void test_runner_no_thread_leak() {
    const NetConfig cfg = netConfig(9, 1, 8);
    const Network net = randomNetwork(cfg, 707);
    { auto warm = makeCuda16Runner(net, 8); }   // the CUDA runtime's own threads start once
    const int before = threadCount();
    for (int i = 0; i < 50; ++i) {
        auto runner = makeCuda16Runner(net, 8);
        std::fill_n(runner->input(0), 8 * cfg.planes() * cfg.area(), 0.5f);
        if (i % 2) runner->begin(0, 8, [](int, std::string) {});   // destroyed in flight
    }
    int after = threadCount();
    for (int tries = 0; after > before && tries < 100; ++tries) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        after = threadCount();
    }
    if (after > before) std::printf("test_cuda16: %d threads before, %d after\n", before, after);
    assert(after <= before);
}

// WorkerEvaluators with backend cuda: one hub on the GPU for all workers,
// each worker's rows its positions alone; evaluations() counts positions.
void test_worker_evaluators_cuda16() {
    const NetConfig cfg = netConfig(9, 2, 16);
    const Network net = randomNetwork(cfg, 708);
    const GpuPool pool(net, 64, 709);
    const size_t in = pool.in(), L = cfg.actions();
    WorkerEvaluators evaluators(net, "cuda16", 8, 8, 256);
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
        for (int workers : {1, 8})   // test_cuda adds 3; fewer slices keep the GPU time short
            for (int batch : {8, 256})
                for (uint64_t chaos : {0ull, 21ull}) {
                    cuda_probe::setHubChaosSeed(chaos);
                    EngineRun run;
                    run.backend = "cuda16";
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
    for (int workers : {1, 0})   // test_cuda adds 3
        for (int batch : {8, 256})
            for (uint64_t chaos : {0ull, 23ull}) {
                cuda_probe::setHubChaosSeed(chaos);
                EngineRun run;
                run.backend = "cuda16";
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
    run.backend = "cuda16";
    std::string joined = bytes(playArenaGames(waveSlice(wave, 0, 5), ac, run));
    joined += bytes(playArenaGames(waveSlice(wave, 5, 9), ac, run));
    assert(joined == reference);
}

namespace fs = std::filesystem;

struct RunDir {
    fs::path path;
    explicit RunDir(const std::string& name)
        : path(fs::temp_directory_path() / ("azgo_test_cuda16_" + std::to_string(getpid()) + "_" + name)) {
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
Config cuda16RunConfig() {
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
    applyOverride(c, "selfplay_engines=local/cuda16:1");
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
void test_cuda16_engine_run_independent_of_settings() {
    std::string reference;
    {
        RunDir dir("straight");
        const fs::path run = dir.path / "run";
        std::unique_ptr<Trainer> t = Trainer::startNew(cuda16RunConfig(), run.string(), {}, nullptr);
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
            std::unique_ptr<Trainer> t = Trainer::startNew(cuda16RunConfig(), run.string(), {}, &startup, opts);
            t->setLog(nullptr);
            t->runUntil(1);
        }
        assert(startup.str().find("engine local/cuda16: ") != std::string::npos);
        {
            std::unique_ptr<Trainer> t = Trainer::resume(run.string(), {}, nullptr, opts);
            t->runUntil(2);
        }
        const std::string manifest = readBytes((run / "selfplay-engines.csv").string());
        assert(manifest.find(",local/cuda16,local/cuda16,\"cuda16; ") != std::string::npos);
        assert(manifest.find(cuda16Description()) != std::string::npos);
        assert(runFiles(run, 2) == reference);
    }
}

// A run is refused before it writes anything when its cuda engine's hub
// would not fit the GPU's memory.
void test_cuda16_engine_memory_checked_up_front() {
    RunDir dir("memory");
    const fs::path run = dir.path / "run";
    cuda16_probe::setMemoryLimit(size_t(1) << 20);   // 1 MB
    bool refused = false;
    std::ostringstream out;
    try {
        Trainer::startNew(cuda16RunConfig(), run.string(), {}, &out);
    } catch (const std::runtime_error& e) {
        refused = std::string(e.what()).find("lower inference_batch") != std::string::npos;
    }
    cuda16_probe::setMemoryLimit(0);
    assert(refused);
    assert(!fs::exists(configPath(run.string())) && !fs::exists(statePath(run.string())));
}


} // namespace

#endif // AZ_USE_CUDA

int main() {
    std::string why;
    if (!cudaBuilt()) {
        // A build without CUDA must say how to get it, and refuse the backend.
        assert(!cuda16Available(&why));
        assert(why.find("-DAZ_USE_CUDA=ON") != std::string::npos);
        bool refused = false;
        try {
            checkBackendAvailable("cuda16");
        } catch (const std::invalid_argument& e) {
            refused = std::string(e.what()) ==
                      "backend 'cuda16' not available: rebuild with -DAZ_USE_CUDA=ON (Linux, NVIDIA GPU)";
        }
        assert(refused);
        assert(cpuBackendName("cuda16") == "blas");
        std::printf("test_cuda16: SKIPPED (%s)\n", why.c_str());
        return 77;
    }
    if (!cuda16Available(&why)) {
        // No GPU, or one without the tensor cores cuda16 needs: nothing to
        // test here (test_cuda fails a GPU the build cannot run).
        std::printf("test_cuda16: SKIPPED (%s)\n", why.c_str());
        return 77;
    }
#ifdef AZ_USE_CUDA
    checkBackendAvailable("cuda16");
    test_available_and_description();
    test_conv16_matches_double();
    test_conv16_tiles_agree();
    test_conv16_batch_independent();
    test_conv16_saturates();
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
    test_worker_evaluators_cuda16();
    test_selfplay_slice_independent_of_workers_and_batching();
    test_arena_wave_independent_of_workers_and_batching();
    test_cuda16_engine_run_independent_of_settings();
    test_cuda16_engine_memory_checked_up_front();
#endif
    std::printf("test_cuda16: all tests passed\n");
    return 0;
}
