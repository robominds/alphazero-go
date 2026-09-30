#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/network.hpp"
#include "azgo/rng.hpp"

using namespace azgo;

static bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

namespace {

NetConfig makeConfig(int n, int history, int blocks, int filters, int valueHidden) {
    NetConfig cfg;
    cfg.boardSize = n;
    cfg.history = history;
    cfg.blocks = blocks;
    cfg.filters = filters;
    cfg.valueHidden = valueHidden;
    return cfg;
}

// A random batch: input planes in {0, 1}, a normalized random policy target
// (or a one-hot one), and outcomes z in {-1, +1}.
struct Batch {
    std::vector<float> input, pi, z;
};

Batch randomBatch(const NetConfig& cfg, int B, Rng& rng, bool oneHotPi = false) {
    Batch batch;
    batch.input.resize(static_cast<size_t>(B) * cfg.planes() * cfg.area());
    for (float& v : batch.input) v = rng.uniform() < 0.5 ? 0.0f : 1.0f;
    batch.pi.assign(static_cast<size_t>(B) * cfg.actions(), 0.0f);
    for (int b = 0; b < B; ++b) {
        float* row = batch.pi.data() + static_cast<size_t>(b) * cfg.actions();
        if (oneHotPi) {
            row[rng.below(cfg.actions())] = 1.0f;
            continue;
        }
        double sum = 0.0;
        for (int a = 0; a < cfg.actions(); ++a) sum += row[a] = static_cast<float>(rng.uniform());
        for (int a = 0; a < cfg.actions(); ++a) row[a] = static_cast<float>(row[a] / sum);
    }
    batch.z.resize(B);
    for (float& v : batch.z) v = rng.uniform() < 0.5 ? -1.0f : 1.0f;
    return batch;
}

double totalLoss(const LossStats& s) { return static_cast<double>(s.policyLoss) + s.valueLoss; }

enum class Probe { Match, Kink, Mismatch };

// Probes dL/dvalue with steps of +-h. Match: the central difference agrees
// with `analytic` (tolerance 2e-2 relative + 2e-3 absolute). Kink: it does
// not, but the forward and backward one-sided differences disagree with
// each other, the signature of a ReLU switching inside [v - h, v + h] (the
// central difference then averages two different slopes). Mismatch: a
// smooth region where the analytic gradient is simply wrong.
template <typename LossFn>
Probe probeGradient(float* value, double analytic, float h, LossFn loss) {
    auto tolerance = [](double a, double b) { return 2e-2 * std::max(std::fabs(a), std::fabs(b)) + 2e-3; };
    const float saved = *value;
    const float up = saved + h, down = saved - h;
    const double lossMid = loss();
    *value = up;
    const double lossUp = loss();
    *value = down;
    const double lossDown = loss();
    *value = saved;
    const double central = (lossUp - lossDown) / (static_cast<double>(up) - down);
    if (std::fabs(analytic - central) <= tolerance(analytic, central)) return Probe::Match;
    const double forward = (lossUp - lossMid) / (static_cast<double>(up) - saved);
    const double backward = (lossMid - lossDown) / (static_cast<double>(saved) - down);
    if (std::fabs(forward - backward) > tolerance(forward, backward)) return Probe::Kink;
    return Probe::Mismatch;
}

bool allFinite(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

} // namespace

static void test_whole_network_gradient_check() {
    const NetConfig cfg = makeConfig(5, 1, 2, 4, 8);
    const int B = 3;
    NaiveBackend be;
    Network net(cfg, 42);
    Workspace ws(cfg, B, be);
    Rng rng(1);
    // Random BN affine parameters: exercises gamma/beta != 1/0 and moves each
    // ReLU's kink away from its channel mean, where values are densest.
    for (ParamView& p : net.params()) {
        const bool isGamma = p.name.find(".gamma") != std::string::npos;
        const bool isBeta = p.name.find(".beta") != std::string::npos;
        for (size_t i = 0; i < p.size; ++i) {
            if (isGamma) p.value[i] = static_cast<float>(0.5 + rng.uniform());
            if (isBeta) p.value[i] = static_cast<float>(rng.uniform() - 0.5);
        }
    }
    Batch batch = randomBatch(cfg, B, rng);

    const LossStats trained = net.trainStep(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B);
    auto loss = [&] {
        return totalLoss(net.computeLoss(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B));
    };
    assert(near(totalLoss(trained), loss(), 1e-6));   // same forward, same batch statistics

    int checked = 0, kinks = 0;
    for (ParamView& p : net.params()) {
        const std::vector<float> analytic(p.grad, p.grad + p.size);   // snapshot after trainStep
        std::vector<size_t> indices;
        if (p.size <= 64) {
            for (size_t i = 0; i < p.size; ++i) indices.push_back(i);   // small tensors: every entry
        } else {
            for (int k = 0; k < 24; ++k) indices.push_back(static_cast<size_t>(rng.below(static_cast<int>(p.size))));
        }
        for (size_t i : indices) {
            ++checked;
            // Unlike a single layer, the whole network has ~1000 ReLUs right
            // behind batch norms, and one weight moves every unit of a
            // channel, so a probe of 1e-2 nearly always crosses some kink.
            // Smaller steps (2e-3, then 5e-4) still resolve an O(1) float
            // loss; a probe is only excused when it visibly straddles a kink.
            Probe first = probeGradient(p.value + i, analytic[i], 2e-3f, loss);
            if (first == Probe::Match) continue;
            Probe second = probeGradient(p.value + i, analytic[i], 5e-4f, loss);
            if (second == Probe::Match) continue;
            if (second == Probe::Kink) {
                ++kinks;
                continue;
            }
            std::printf("%s[%zu]: analytic %.6f does not match finite differences\n",
                        p.name.c_str(), i, analytic[i]);
            assert(false);
        }
    }
    assert(checked > 200);
    assert(kinks * 4 <= checked);   // kinks excuse only a minority of probes
}

static void test_param_names_order_and_decay() {
    const NetConfig cfg = makeConfig(5, 1, 1, 4, 8);
    Network a(cfg, 1), b(cfg, 2);
    std::vector<ParamView> pa = a.params(), pb = b.params();
    const std::vector<std::string> expected = {
        "stem.conv.w", "stem.bn.gamma", "stem.bn.beta",
        "block0.conv1.w", "block0.bn1.gamma", "block0.bn1.beta",
        "block0.conv2.w", "block0.bn2.gamma", "block0.bn2.beta",
        "policy.conv.w", "policy.bn.gamma", "policy.bn.beta", "policy.fc.w", "policy.fc.b",
        "value.conv.w", "value.bn.gamma", "value.bn.beta",
        "value.fc1.w", "value.fc1.b", "value.fc2.w", "value.fc2.b"};
    assert(pa.size() == expected.size() && pb.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(pa[i].name == expected[i] && pb[i].name == expected[i]);
        const bool isWeight = pa[i].name.size() > 2 && pa[i].name.compare(pa[i].name.size() - 2, 2, ".w") == 0;
        assert(pa[i].decay == isWeight);
    }
    const int A = cfg.area();
    assert(pa[0].size == static_cast<size_t>(4 * cfg.planes() * 9));      // stem.conv.w
    assert(pa[3].size == static_cast<size_t>(4 * 4 * 9));                // block0.conv1.w
    assert(pa[9].size == static_cast<size_t>(2 * 4));                    // policy.conv.w
    assert(pa[12].size == static_cast<size_t>(cfg.actions() * 2 * A));   // policy.fc.w
    assert(pa[17].size == static_cast<size_t>(8 * A));                   // value.fc1.w
    assert(pa[20].size == 1);                                            // value.fc2.b

    const NetConfig deep = makeConfig(5, 1, 3, 4, 8);
    Network d(deep, 1);
    std::set<std::string> names;
    for (const ParamView& p : d.params()) names.insert(p.name);
    assert(names.size() == d.params().size());                           // unique
    assert(names.count("block2.bn1.gamma") == 1 && names.count("block2.conv2.w") == 1);
}

static void test_buffers() {
    const NetConfig cfg = makeConfig(5, 1, 2, 4, 8);
    Network net(cfg, 1);
    std::vector<BufferView> bufs = net.buffers();
    const int bnCount = 1 + 2 * cfg.blocks + 2;
    assert(bufs.size() == static_cast<size_t>(2 * bnCount));
    assert(bufs[0].name == "stem.bn.runMean" && bufs[1].name == "stem.bn.runVar");
    assert(bufs[2].name == "block0.bn1.runMean" && bufs[5].name == "block0.bn2.runVar");
    assert(bufs[bufs.size() - 2].name == "value.bn.runMean" && bufs.back().name == "value.bn.runVar");
    for (const BufferView& b : bufs) {
        const bool isVar = b.name.find("runVar") != std::string::npos;
        for (size_t i = 0; i < b.size; ++i) assert(b.value[i] == (isVar ? 1.0f : 0.0f));
    }
}

static void test_infer_shapes_every_board_size() {
    NaiveBackend be;
    Rng rng(3);
    for (int n = 5; n <= 19; ++n) {
        const NetConfig cfg = makeConfig(n, 2, 1, 2, 4);
        const int B = 2;
        Network net(cfg, static_cast<uint64_t>(n));
        Workspace ws(cfg, B, be);
        Batch batch = randomBatch(cfg, B, rng);
        std::vector<float> logits(static_cast<size_t>(B) * cfg.actions(), std::numeric_limits<float>::quiet_NaN());
        std::vector<float> values(B, std::numeric_limits<float>::quiet_NaN());
        net.infer(ws, batch.input.data(), B, logits.data(), values.data());
        assert(allFinite(logits) && allFinite(values));
        for (float v : values) assert(v >= -1.0f && v <= 1.0f);
    }
}

static void test_zero_blocks() {
    const NetConfig cfg = makeConfig(5, 1, 0, 4, 8);
    NaiveBackend be;
    Network net(cfg, 5);
    assert(net.params().size() == 15);          // stem 3 + policy 5 + value 7
    Workspace ws(cfg, 4, be);
    Rng rng(4);
    Batch batch = randomBatch(cfg, 4, rng);
    std::vector<float> logits(4 * cfg.actions()), values(4);
    net.infer(ws, batch.input.data(), 4, logits.data(), values.data());
    assert(allFinite(logits) && allFinite(values));
    LossStats s = net.trainStep(ws, batch.input.data(), batch.pi.data(), batch.z.data(), 4);
    assert(std::isfinite(s.policyLoss) && std::isfinite(s.valueLoss));
}

static void test_seed_determinism() {
    const NetConfig cfg = makeConfig(5, 2, 1, 4, 8);
    NaiveBackend be;
    Network a(cfg, 7), b(cfg, 7), c(cfg, 8);
    Workspace ws(cfg, 2, be);
    Rng rng(5);
    Batch batch = randomBatch(cfg, 2, rng);
    std::vector<float> la(2 * cfg.actions()), lb(la.size()), lc(la.size()), va(2), vb(2), vc(2);
    a.infer(ws, batch.input.data(), 2, la.data(), va.data());
    b.infer(ws, batch.input.data(), 2, lb.data(), vb.data());
    c.infer(ws, batch.input.data(), 2, lc.data(), vc.data());
    assert(std::memcmp(la.data(), lb.data(), la.size() * sizeof(float)) == 0);
    assert(std::memcmp(va.data(), vb.data(), va.size() * sizeof(float)) == 0);
    assert(la != lc);
}

static void test_running_stats_only_change_in_train_step() {
    const NetConfig cfg = makeConfig(5, 1, 1, 4, 8);
    NaiveBackend be;
    Network net(cfg, 9);
    Workspace ws(cfg, 4, be);
    Rng rng(6);
    Batch batch = randomBatch(cfg, 4, rng);
    auto snapshot = [&net] {
        std::vector<float> all;
        for (const BufferView& b : net.buffers()) all.insert(all.end(), b.value, b.value + b.size);
        return all;
    };
    const std::vector<float> before = snapshot();
    net.computeLoss(ws, batch.input.data(), batch.pi.data(), batch.z.data(), 4);
    assert(snapshot() == before);
    net.trainStep(ws, batch.input.data(), batch.pi.data(), batch.z.data(), 4);
    const std::vector<float> after = snapshot();
    size_t changed = 0;
    for (size_t i = 0; i < before.size(); ++i) changed += after[i] != before[i];
    assert(changed > before.size() / 2);
}

static void test_sgd_reduces_loss_on_fixed_batch() {
    // Plain SGD through params() on one batch: the network can memorize 8
    // positions, so the loss must fall well below its starting value.
    const NetConfig cfg = makeConfig(5, 1, 1, 8, 16);
    const int B = 8;
    NaiveBackend be;
    Network net(cfg, 11);
    Workspace ws(cfg, B, be);
    Rng rng(7);
    Batch batch = randomBatch(cfg, B, rng, /*oneHotPi=*/true);
    double first = -1.0;
    for (int step = 0; step < 60; ++step) {
        LossStats s = net.trainStep(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B);
        if (step == 0) first = totalLoss(s);
        for (ParamView& p : net.params())
            for (size_t i = 0; i < p.size; ++i) p.value[i] -= 0.05f * p.grad[i];
    }
    const double last = totalLoss(net.computeLoss(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B));
    std::printf("sgd: loss %.4f -> %.4f\n", first, last);
    assert(last <= 0.5 * first);
}

static void test_inference_is_per_sample() {
    // Inference BN uses fixed running statistics, so a sample's output must
    // not depend on the rest of the batch.
    const NetConfig cfg = makeConfig(5, 2, 1, 4, 8);
    NaiveBackend be;
    Network net(cfg, 13);
    Workspace ws(cfg, 4, be);
    Rng rng(8);
    for (int i = 0; i < 3; ++i) {                // non-trivial running statistics
        Batch warm = randomBatch(cfg, 4, rng);
        net.trainStep(ws, warm.input.data(), warm.pi.data(), warm.z.data(), 4);
    }
    Batch batch = randomBatch(cfg, 4, rng);
    const int actions = cfg.actions();
    const size_t sampleInput = static_cast<size_t>(cfg.planes()) * cfg.area();
    std::vector<float> logits4(4 * actions), values4(4);
    net.infer(ws, batch.input.data(), 4, logits4.data(), values4.data());
    for (int b = 0; b < 4; ++b) {
        std::vector<float> logits1(actions), values1(1);
        net.infer(ws, batch.input.data() + b * sampleInput, 1, logits1.data(), values1.data());
        for (int a = 0; a < actions; ++a) {
            const float x = logits4[b * actions + a];
            assert(near(logits1[a], x, 1e-5 * (1 + std::fabs(x))));
        }
        assert(near(values1[0], values4[b], 1e-5));
    }
}

int main() {
    test_whole_network_gradient_check();
    test_param_names_order_and_decay();
    test_buffers();
    test_infer_shapes_every_board_size();
    test_zero_blocks();
    test_seed_determinism();
    test_running_stats_only_change_in_train_step();
    test_sgd_reduces_loss_on_fixed_batch();
    test_inference_is_per_sample();
    std::printf("test_network: all tests passed\n");
    return 0;
}
