#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/layers.hpp"
#include "azgo/rng.hpp"

using namespace azgo;

static bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

namespace {

constexpr float kH = 1e-2f;   // finite-difference step

std::vector<float> randomVector(Rng& rng, size_t count, double lo = -1.0, double hi = 1.0) {
    std::vector<float> v(count);
    for (float& x : v) x = static_cast<float>(lo + (hi - lo) * rng.uniform());
    return v;
}

// The scalar loss L = sum_i r_i * y_i: its gradient with respect to y is
// exactly r, so r is what we feed to backward as dy.
double weightedSum(const std::vector<float>& r, const std::vector<float>& y) {
    double s = 0.0;
    for (size_t i = 0; i < r.size(); ++i) s += static_cast<double>(r[i]) * y[i];
    return s;
}

// Compares analytic[i] with the central difference (L(v+h) - L(v-h)) / 2h
// for every entry of `value`. The step actually taken is measured after
// float rounding of v +- h.
template <typename LossFn>
void checkGradient(const char* what, float* value, const float* analytic, size_t count,
                   LossFn loss) {
    for (size_t i = 0; i < count; ++i) {
        const float saved = value[i];
        const float up = saved + kH, down = saved - kH;
        value[i] = up;
        const double lossUp = loss();
        value[i] = down;
        const double lossDown = loss();
        value[i] = saved;
        const double numeric = (lossUp - lossDown) / (static_cast<double>(up) - down);
        const double a = analytic[i];
        const double tol = 2e-2 * std::max(std::fabs(a), std::fabs(numeric)) + 2e-3;
        if (!(std::fabs(a - numeric) <= tol)) {
            std::printf("%s[%zu]: analytic %.6f numeric %.6f\n", what, i, a, numeric);
            assert(false);
        }
    }
}

void checkConvGradients(int k) {
    const int cin = 2, cout = 3, B = 2, n = 4, M = B * n * n;
    NaiveBackend be;
    Rng rng(100 + k);
    Conv2d conv(cin, cout, k);
    conv.init(rng);
    std::vector<float> x = randomVector(rng, static_cast<size_t>(cin) * M);
    std::vector<float> r = randomVector(rng, static_cast<size_t>(cout) * M);
    std::vector<float> col(conv.colSize(B, n)), dcol(col.size()), y(r.size()), dx(x.size());
    auto loss = [&] {
        conv.forward(be, x.data(), B, n, col.data(), y.data());
        return weightedSum(r, y);
    };
    loss();
    conv.backward(be, x.data(), col.data(), r.data(), B, n, dcol.data(), dx.data());
    checkGradient(k == 1 ? "conv1x1 dx" : "conv3x3 dx", x.data(), dx.data(), x.size(), loss);
    checkGradient(k == 1 ? "conv1x1 gw" : "conv3x3 gw", conv.w.data(), conv.gw.data(),
                  conv.w.size(), loss);
}

} // namespace

static void test_conv1x1_gradients() { checkConvGradients(1); }
static void test_conv3x3_gradients() { checkConvGradients(3); }

static void test_batchnorm_gradients() {
    const int c = 3, M = 10;
    Rng rng(7);
    BatchNorm2d bn(c);
    bn.gamma = randomVector(rng, c, 0.5, 1.5);
    bn.beta = randomVector(rng, c, -0.5, 0.5);
    std::vector<float> x = randomVector(rng, static_cast<size_t>(c) * M);
    std::vector<float> r = randomVector(rng, x.size());
    std::vector<float> y(x.size()), xhat(x.size()), invStd(c), dx(x.size());
    // updateRunning = false: probing the loss must not drift the running stats.
    auto loss = [&] {
        bn.forwardTrain(x.data(), M, y.data(), xhat.data(), invStd.data(), false);
        return weightedSum(r, y);
    };
    loss();
    bn.backward(r.data(), xhat.data(), invStd.data(), M, dx.data());
    checkGradient("bn dx", x.data(), dx.data(), x.size(), loss);
    checkGradient("bn ggamma", bn.gamma.data(), bn.ggamma.data(), c, loss);
    checkGradient("bn gbeta", bn.beta.data(), bn.gbeta.data(), c, loss);
    for (int ch = 0; ch < c; ++ch) {
        assert(bn.runMean[ch] == 0.0f && bn.runVar[ch] == 1.0f);
    }
}

static void test_linear_gradients() {
    const int in = 5, out = 4, B = 3;
    NaiveBackend be;
    Rng rng(8);
    Linear fc(in, out);
    fc.init(rng);
    fc.b = randomVector(rng, out);   // nonzero bias so forward's "+ b" is exercised
    std::vector<float> x = randomVector(rng, static_cast<size_t>(B) * in);
    std::vector<float> r = randomVector(rng, static_cast<size_t>(B) * out);
    std::vector<float> y(r.size()), dx(x.size());
    auto loss = [&] {
        fc.forward(be, x.data(), B, y.data());
        return weightedSum(r, y);
    };
    loss();
    fc.backward(be, x.data(), r.data(), B, dx.data());
    checkGradient("linear dx", x.data(), dx.data(), x.size(), loss);
    checkGradient("linear gw", fc.w.data(), fc.gw.data(), fc.w.size(), loss);
    checkGradient("linear gb", fc.b.data(), fc.gb.data(), fc.b.size(), loss);
}

static void test_relu_gradients() {
    Rng rng(9);
    // |x| >= 0.2 > h, so no finite-difference probe crosses the kink at 0.
    std::vector<float> x(40);
    for (float& v : x) v = static_cast<float>((rng.uniform() < 0.5 ? -1 : 1) * (0.2 + 0.8 * rng.uniform()));
    std::vector<float> r = randomVector(rng, x.size());
    std::vector<float> y(x.size());
    auto loss = [&] {
        y = x;
        reluForward(y.data(), y.size());
        return weightedSum(r, y);
    };
    loss();
    std::vector<float> dx = r;
    reluBackward(y.data(), dx.data(), dx.size());
    checkGradient("relu dx", x.data(), dx.data(), x.size(), loss);
    for (size_t i = 0; i < x.size(); ++i) assert(dx[i] == (x[i] > 0 ? r[i] : 0.0f));
}

static void test_conv_forward_hand_example() {
    // All-ones 3x3 kernel: each output is the sum of the input's 3x3
    // neighborhood, with zeros beyond the edge.
    NaiveBackend be;
    Conv2d conv(1, 1, 3);
    std::fill(conv.w.begin(), conv.w.end(), 1.0f);
    const float x[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> col(conv.colSize(1, 3));
    float y[9];
    conv.forward(be, x, 1, 3, col.data(), y);
    const float expected[9] = {12, 21, 16, 27, 45, 33, 24, 39, 28};
    for (int p = 0; p < 9; ++p) assert(y[p] == expected[p]);
}

static void test_batchnorm_inference_hand_values() {
    BatchNorm2d bn(2);
    bn.gamma = {2.0f, 0.5f};
    bn.beta = {1.0f, -1.0f};
    bn.runMean = {1.0f, -2.0f};
    bn.runVar = {4.0f - BatchNorm2d::kEps, 0.25f - BatchNorm2d::kEps};  // sqrt(var + eps) = 2, 0.5
    const float x[6] = {1, 3, -1, -2, -1.5f, 0};
    float y[6];
    bn.forwardInfer(x, 3, y);
    // Channel 0: 2 * (x - 1) / 2 + 1 = x.  Channel 1: 0.5 * (x + 2) / 0.5 - 1 = x + 1.
    const float expected[6] = {1, 3, -1, -1, -0.5f, 1};
    for (int i = 0; i < 6; ++i) assert(near(y[i], expected[i]));
}

static void test_batchnorm_running_stats_update() {
    BatchNorm2d bn(2);
    const int M = 4;
    const float x[8] = {1, 2, 3, 4, 0, 0, 0, 4};
    float y[8], xhat[8], invStd[2];

    bn.forwardTrain(x, M, y, xhat, invStd, false);
    assert(bn.runMean[0] == 0.0f && bn.runVar[0] == 1.0f);   // untouched without updateRunning

    bn.forwardTrain(x, M, y, xhat, invStd, true);
    // Channel 0: mean 2.5, biased var 1.25, unbiased 5/3.
    // Channel 1: mean 1,   biased var 3,    unbiased 4.
    assert(near(bn.runMean[0], 0.1 * 2.5));
    assert(near(bn.runVar[0], 0.9 * 1.0 + 0.1 * (5.0 / 3.0)));
    assert(near(bn.runMean[1], 0.1 * 1.0));
    assert(near(bn.runVar[1], 0.9 * 1.0 + 0.1 * 4.0));
    // Training output uses the batch statistics (biased variance), gamma = 1, beta = 0.
    assert(near(invStd[0], 1.0 / std::sqrt(1.25 + BatchNorm2d::kEps)));
    assert(near(y[0], (1.0 - 2.5) / std::sqrt(1.25 + BatchNorm2d::kEps)));
    assert(near(y[7], (4.0 - 1.0) / std::sqrt(3.0 + BatchNorm2d::kEps)));
}

static void test_he_init_scale() {
    Rng rng(1);
    Conv2d conv(64, 64, 3);                 // w is 64 x 576, fan-in 576
    conv.init(rng);
    double sum = 0.0, sq = 0.0;
    for (float v : conv.w) {
        sum += v;
        sq += static_cast<double>(v) * v;
    }
    const double count = static_cast<double>(conv.w.size());
    const double mean = sum / count;
    const double stddev = std::sqrt(sq / count - mean * mean);
    const double expected = std::sqrt(2.0 / 576.0);
    assert(std::fabs(stddev - expected) <= 0.1 * expected);
    assert(std::fabs(mean) <= 0.1 * expected);

    Linear fc(256, 32);
    fc.b.assign(32, 5.0f);
    fc.init(rng);
    for (float v : fc.b) assert(v == 0.0f);
    double fsq = 0.0;
    for (float v : fc.w) fsq += static_cast<double>(v) * v;
    const double fcStd = std::sqrt(fsq / fc.w.size());
    assert(std::fabs(fcStd - std::sqrt(2.0 / 256.0)) <= 0.1 * std::sqrt(2.0 / 256.0));
}

static void test_gradients_accumulate() {
    // Parameter gradients add up across backward calls (the network zeroes
    // them once per step); a second identical backward doubles them.
    NaiveBackend be;
    Rng rng(12);
    const int B = 2, n = 3, M = B * n * n;

    Conv2d conv(2, 3, 3);
    conv.init(rng);
    std::vector<float> x = randomVector(rng, 2 * M), dy = randomVector(rng, 3 * M);
    std::vector<float> col(conv.colSize(B, n)), dcol(col.size()), y(dy.size()), dx(x.size());
    conv.forward(be, x.data(), B, n, col.data(), y.data());
    conv.backward(be, x.data(), col.data(), dy.data(), B, n, dcol.data(), dx.data());
    std::vector<float> once = conv.gw;
    conv.backward(be, x.data(), col.data(), dy.data(), B, n, dcol.data(), nullptr);   // dx optional
    for (size_t i = 0; i < once.size(); ++i) assert(near(conv.gw[i], 2.0 * once[i], 1e-5 * (1 + std::fabs(once[i]))));

    BatchNorm2d bn(3);
    std::vector<float> xhat(dy.size()), invStd(3), bny(dy.size()), bndx(dy.size());
    bn.forwardTrain(y.data(), M, bny.data(), xhat.data(), invStd.data(), false);
    bn.backward(dy.data(), xhat.data(), invStd.data(), M, bndx.data());
    std::vector<float> g1 = bn.ggamma, b1 = bn.gbeta;
    bn.backward(dy.data(), xhat.data(), invStd.data(), M, bndx.data());
    for (int ch = 0; ch < 3; ++ch) {
        assert(near(bn.ggamma[ch], 2.0 * g1[ch], 1e-5 * (1 + std::fabs(g1[ch]))));
        assert(near(bn.gbeta[ch], 2.0 * b1[ch], 1e-5 * (1 + std::fabs(b1[ch]))));
    }

    Linear fc(4, 3);
    fc.init(rng);
    std::vector<float> fx = randomVector(rng, 8), fdy = randomVector(rng, 6);
    fc.backward(be, fx.data(), fdy.data(), 2, nullptr);
    std::vector<float> w1 = fc.gw, gb1 = fc.gb;
    fc.backward(be, fx.data(), fdy.data(), 2, nullptr);
    for (size_t i = 0; i < w1.size(); ++i) assert(near(fc.gw[i], 2.0 * w1[i], 1e-5 * (1 + std::fabs(w1[i]))));
    for (size_t i = 0; i < gb1.size(); ++i) assert(near(fc.gb[i], 2.0 * gb1[i], 1e-5 * (1 + std::fabs(gb1[i]))));
}

int main() {
    test_conv1x1_gradients();
    test_conv3x3_gradients();
    test_batchnorm_gradients();
    test_linear_gradients();
    test_relu_gradients();
    test_conv_forward_hand_example();
    test_batchnorm_inference_hand_values();
    test_batchnorm_running_stats_update();
    test_he_init_scale();
    test_gradients_accumulate();
    std::printf("test_layers: all tests passed\n");
    return 0;
}
