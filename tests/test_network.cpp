#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/network.hpp"
#include "azgo/optimizer.hpp"
#include "azgo/parallel.hpp"
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

// Ownership targets in {-1, 0, +1}, [B][area].
std::vector<float> randomOwnership(const NetConfig& cfg, int B, Rng& rng) {
    std::vector<float> own(static_cast<size_t>(B) * cfg.area());
    for (float& v : own) v = static_cast<float>(rng.below(3) - 1);
    return own;
}

// The loss trainStep minimizes: policy + value + w * ownership.
double totalLoss(const LossStats& s, float ownershipWeight = 0.0f) {
    return static_cast<double>(s.policyLoss) + s.valueLoss +
           static_cast<double>(ownershipWeight) * s.ownershipLoss;
}

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

// With `ownership`, the network has the ownership head and the loss includes
// it with weight 1.5, so its gradients and the trunk's share of them are
// checked too.
static void checkWholeNetworkGradients(bool ownership) {
    NetConfig cfg = makeConfig(5, 1, 2, 4, 8);
    cfg.ownership = ownership;
    const float w = ownership ? 1.5f : 0.0f;
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
    std::vector<float> own;
    if (ownership) own = randomOwnership(cfg, B, rng);
    const float* target = ownership ? own.data() : nullptr;

    const LossStats trained =
        net.trainStep(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B, target, w);
    assert(ownership == (trained.ownershipLoss > 0.0f));
    auto loss = [&] {
        return totalLoss(net.computeLoss(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B,
                                         target, w),
                         w);
    };
    assert(near(totalLoss(trained, w), loss(), 1e-6));   // same forward, same batch statistics

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

static void test_whole_network_gradient_check() {
    checkWholeNetworkGradients(false);
    checkWholeNetworkGradients(true);
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

// The head's tensors come last, and it is initialized last, so a network with
// the head is the network without it plus three tensors: same names, same
// weights, same policy and value. Its output is n*n values in [-1, 1].
static void test_ownership_head_params_and_outputs() {
    NetConfig off = makeConfig(5, 2, 1, 4, 8), on = off;
    on.ownership = true;
    Network a(off, 21), b(on, 21);
    std::vector<ParamView> pa = a.params(), pb = b.params();
    assert(pb.size() == pa.size() + 3);
    for (size_t i = 0; i < pa.size(); ++i) {
        assert(pa[i].name == pb[i].name && pa[i].size == pb[i].size && pa[i].decay == pb[i].decay);
        assert(std::memcmp(pa[i].value, pb[i].value, pa[i].size * sizeof(float)) == 0);
    }
    const size_t k = pa.size();
    assert(pb[k].name == "ownership.conv.w" && pb[k].size == 4 && pb[k].decay);   // filters weights
    assert(pb[k + 1].name == "ownership.bn.gamma" && pb[k + 1].size == 1 && !pb[k + 1].decay);
    assert(pb[k + 2].name == "ownership.bn.beta" && pb[k + 2].size == 1 && !pb[k + 2].decay);
    std::vector<BufferView> ba = a.buffers(), bb = b.buffers();
    assert(bb.size() == ba.size() + 2);
    assert(bb[ba.size()].name == "ownership.bn.runMean" && bb.back().name == "ownership.bn.runVar");

    NaiveBackend be;
    Workspace wa(off, 3, be), wb(on, 3, be);
    Rng rng(9);
    Batch batch = randomBatch(off, 3, rng);
    const size_t A = static_cast<size_t>(off.area());
    std::vector<float> la(3 * off.actions()), lb(la.size()), lc(la.size()), va(3), vb(3), vc(3);
    std::vector<float> own(3 * A, std::numeric_limits<float>::quiet_NaN());
    a.infer(wa, batch.input.data(), 3, la.data(), va.data());
    b.infer(wb, batch.input.data(), 3, lb.data(), vb.data(), own.data());
    assert(std::memcmp(la.data(), lb.data(), la.size() * sizeof(float)) == 0);
    assert(std::memcmp(va.data(), vb.data(), va.size() * sizeof(float)) == 0);
    assert(allFinite(own));
    for (float o : own) assert(o >= -1.0f && o <= 1.0f);
    assert(*std::min_element(own.begin(), own.end()) < *std::max_element(own.begin(), own.end()));
    // Without an ownership output the head is simply not read.
    b.infer(wb, batch.input.data(), 3, lc.data(), vc.data());
    assert(lc == lb && vc == vb);

    // Every board size: [B][n*n] values, all in range.
    for (int n : {7, 19}) {
        NetConfig cfg = makeConfig(n, 2, 1, 2, 4);
        cfg.ownership = true;
        Network net(cfg, static_cast<uint64_t>(n));
        Workspace ws(cfg, 2, be);
        Batch nb = randomBatch(cfg, 2, rng);
        std::vector<float> logits(2 * cfg.actions()), values(2);
        std::vector<float> o(2 * static_cast<size_t>(cfg.area()), std::numeric_limits<float>::quiet_NaN());
        net.infer(ws, nb.input.data(), 2, logits.data(), values.data(), o.data());
        assert(allFinite(o));
        for (float x : o) assert(x >= -1.0f && x <= 1.0f);
    }
}

// w = 0: the ownership loss is measured, but the step is exactly the one the
// network without the head takes, and the head itself gets no gradient.
static void test_ownership_weight_zero_trains_nothing() {
    NetConfig off = makeConfig(5, 1, 1, 4, 8), on = off;
    on.ownership = true;
    Network a(off, 23), b(on, 23);
    NaiveBackend be;
    Workspace wa(off, 4, be), wb(on, 4, be);
    Rng rng(10);
    Batch batch = randomBatch(off, 4, rng);
    const std::vector<float> own = randomOwnership(on, 4, rng);
    const LossStats sa = a.trainStep(wa, batch.input.data(), batch.pi.data(), batch.z.data(), 4);
    const LossStats sb =
        b.trainStep(wb, batch.input.data(), batch.pi.data(), batch.z.data(), 4, own.data(), 0.0f);
    assert(sa.policyLoss == sb.policyLoss && sa.valueLoss == sb.valueLoss);
    assert(sa.ownershipLoss == 0.0f && sb.ownershipLoss > 0.0f);
    std::vector<ParamView> pa = a.params(), pb = b.params();
    for (size_t i = 0; i < pa.size(); ++i)
        assert(std::memcmp(pa[i].grad, pb[i].grad, pa[i].size * sizeof(float)) == 0);
    for (size_t i = pa.size(); i < pb.size(); ++i)
        for (size_t j = 0; j < pb[i].size; ++j) assert(pb[i].grad[j] == 0.0f);

    // w > 0: the head and, through it, the trunk receive gradient.
    b.trainStep(wb, batch.input.data(), batch.pi.data(), batch.z.data(), 4, own.data(), 1.5f);
    pb = b.params();
    bool headGrad = false;
    for (size_t j = 0; j < pb[pa.size()].size; ++j) headGrad = headGrad || pb[pa.size()].grad[j] != 0.0f;
    assert(headGrad);
    assert(std::memcmp(pa[0].grad, pb[0].grad, pa[0].size * sizeof(float)) != 0);   // stem.conv.w
}

// A few SGD steps on one batch reduce the ownership loss. The target is one a
// 1x1 head can learn from the trunk: +1 on the side to move's stones (input
// plane 0), -1 on the opponent's (plane 1), 0 where both or neither.
static void test_sgd_reduces_ownership_loss() {
    NetConfig cfg = makeConfig(5, 1, 1, 8, 16);
    cfg.ownership = true;
    const int B = 8, A = cfg.area();
    NaiveBackend be;
    Network net(cfg, 11);
    Workspace ws(cfg, B, be);
    Rng rng(7);
    Batch batch = randomBatch(cfg, B, rng, /*oneHotPi=*/true);
    std::vector<float> own(static_cast<size_t>(B) * A);
    for (int b = 0; b < B; ++b)
        for (int p = 0; p < A; ++p) {
            const float* planes = batch.input.data() + static_cast<size_t>(b) * cfg.planes() * A;
            own[static_cast<size_t>(b) * A + p] = planes[p] - planes[A + p];
        }
    double first = -1.0;
    for (int step = 0; step < 40; ++step) {
        const LossStats s =
            net.trainStep(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B, own.data(), 1.5f);
        if (step == 0) first = s.ownershipLoss;
        for (ParamView& p : net.params())
            for (size_t i = 0; i < p.size; ++i) p.value[i] -= 0.05f * p.grad[i];
    }
    const double last =
        net.computeLoss(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B, own.data(), 1.5f)
            .ownershipLoss;
    std::printf("sgd: ownership loss %.4f -> %.4f\n", first, last);
    assert(last <= 0.5 * first);
}

// inferLoss is the training loss terms with inference-mode batch norm:
// computed by hand from infer()'s outputs, it agrees, and it writes
// nothing to the network (parameters, gradients, running statistics).
static void test_infer_loss_matches_infer_outputs() {
    NetConfig cfg = makeConfig(5, 2, 1, 4, 8);
    cfg.ownership = true;
    const int B = 5, A = cfg.area(), actions = cfg.actions();
    NaiveBackend be;
    Network net(cfg, 21);
    Workspace ws(cfg, B, be);
    Rng rng(22);
    // A few training steps, so the running statistics are not their initial values.
    for (int step = 0; step < 3; ++step) {
        Batch warm = randomBatch(cfg, B, rng);
        const std::vector<float> own = randomOwnership(cfg, B, rng);
        net.trainStep(ws, warm.input.data(), warm.pi.data(), warm.z.data(), B, own.data(), 1.5f);
        for (ParamView& p : net.params())
            for (size_t i = 0; i < p.size; ++i) p.value[i] -= 0.05f * p.grad[i];
    }
    Batch batch = randomBatch(cfg, B, rng);
    const std::vector<float> own = randomOwnership(cfg, B, rng);

    auto snapshot = [&net] {
        std::vector<float> all;
        for (const ParamView& p : net.params()) {
            all.insert(all.end(), p.value, p.value + p.size);
            all.insert(all.end(), p.grad, p.grad + p.size);
        }
        for (const BufferView& b : net.buffers()) all.insert(all.end(), b.value, b.value + b.size);
        return all;
    };
    const std::vector<float> before = snapshot();
    const LossStats s = net.inferLoss(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B, own.data());
    const LossStats noOwnership = net.inferLoss(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B);
    assert(snapshot() == before);

    std::vector<float> logits(static_cast<size_t>(B) * actions), values(B), ownership(static_cast<size_t>(B) * A);
    net.infer(ws, batch.input.data(), B, logits.data(), values.data(), ownership.data());
    double policy = 0, value = 0, owner = 0;
    for (int b = 0; b < B; ++b) {
        const float* l = logits.data() + static_cast<size_t>(b) * actions;
        double sumExp = 0;
        for (int a = 0; a < actions; ++a) sumExp += std::exp(static_cast<double>(l[a]));
        for (int a = 0; a < actions; ++a)
            policy -= batch.pi[static_cast<size_t>(b) * actions + a] * (l[a] - std::log(sumExp));
        value += (batch.z[b] - values[b]) * (batch.z[b] - values[b]);
        for (int p = 0; p < A; ++p) {
            const double d = ownership[static_cast<size_t>(b) * A + p] - own[static_cast<size_t>(b) * A + p];
            owner += d * d;
        }
    }
    assert(near(s.policyLoss, policy / B, 1e-4));
    assert(near(s.valueLoss, value / B, 1e-5));
    assert(near(s.ownershipLoss, owner / (B * A), 1e-5));
    assert(s.policyLoss == noOwnership.policyLoss && s.valueLoss == noOwnership.valueLoss);
    assert(noOwnership.ownershipLoss == 0.0f);
    // Batch statistics (computeLoss) give a different number: the modes differ.
    const LossStats probe = net.computeLoss(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B);
    assert(probe.valueLoss != s.valueLoss);
}

// Everything a training step produces, as raw bytes: the losses, every
// gradient, the running statistics, the inference outputs and loss, and,
// after two SGD steps, the parameters and the momentum.
static std::vector<unsigned char> trainingFingerprint(const NetConfig& cfg, const Backend& be, int B,
                                                      ThreadPool* pool) {
    std::vector<unsigned char> out;
    auto append = [&out](const void* p, size_t bytes) {
        const unsigned char* c = static_cast<const unsigned char*>(p);
        out.insert(out.end(), c, c + bytes);
    };
    Network net(cfg, 17);
    Workspace ws(cfg, B, be, pool);
    SgdMomentum opt(net, 0.9f, 1e-4f);
    Rng rng(18);
    for (int step = 0; step < 2; ++step) {
        Batch batch = randomBatch(cfg, B, rng);
        const std::vector<float> own = randomOwnership(cfg, B, rng);
        const LossStats s = net.trainStep(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B,
                                          cfg.ownership ? own.data() : nullptr, 1.5f);
        append(&s, sizeof s);
        for (const ParamView& p : net.params()) append(p.grad, p.size * sizeof(float));
        opt.step(net, 0.05f, pool);
    }
    for (const ParamView& p : net.params()) append(p.value, p.size * sizeof(float));
    for (const std::vector<float>& v : opt.velocity()) append(v.data(), v.size() * sizeof(float));
    for (const BufferView& b : net.buffers()) append(b.value, b.size * sizeof(float));
    Batch batch = randomBatch(cfg, B, rng);
    const std::vector<float> own = randomOwnership(cfg, B, rng);
    std::vector<float> logits(static_cast<size_t>(B) * cfg.actions()), values(B), owners(static_cast<size_t>(B) * cfg.area());
    net.infer(ws, batch.input.data(), B, logits.data(), values.data(), cfg.ownership ? owners.data() : nullptr);
    append(logits.data(), logits.size() * sizeof(float));
    append(values.data(), values.size() * sizeof(float));
    append(owners.data(), owners.size() * sizeof(float));
    const LossStats held = net.inferLoss(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B,
                                         cfg.ownership ? own.data() : nullptr);
    append(&held, sizeof held);
    const LossStats probe = net.computeLoss(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B);
    append(&probe, sizeof probe);
    return out;
}

// A training step on a pool of 1, 2, 3 or 7 threads gives exactly the
// bits of one without a pool, on 5x5 and 9x9 networks wide enough (32
// filters) that the gemms are cut into row bands, and with batches that
// span several chunks of every kind; with the naive backend and, when
// built, BLAS.
static void test_thread_pool_gives_identical_bits() {
    struct Case { int n, blocks, filters, B; bool ownership; };
    const Case cases[] = {{5, 2, 32, 40, true}, {9, 1, 32, 24, false}, {9, 1, 48, 17, true}};
    std::vector<std::unique_ptr<Backend>> backends;
    backends.push_back(makeBackend("naive", 1));
    if (blasAvailable()) backends.push_back(makeBackend("blas", 1));
    ThreadPool p1(1), p2(2), p3(3), p7(7);
    for (const std::unique_ptr<Backend>& be : backends) {
        for (const Case& c : cases) {
            NetConfig cfg = makeConfig(c.n, 2, c.blocks, c.filters, 16);
            cfg.ownership = c.ownership;
            const std::vector<unsigned char> reference = trainingFingerprint(cfg, *be, c.B, nullptr);
            for (ThreadPool* pool : {&p1, &p2, &p3, &p7}) {
                if (trainingFingerprint(cfg, *be, c.B, pool) != reference) {
                    std::printf("%s %dx%d %dx%d B=%d: %d threads differ\n", be->name(), c.n, c.n, c.blocks,
                                c.filters, c.B, pool->threads());
                    assert(false);
                }
            }
        }
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
    test_ownership_head_params_and_outputs();
    test_ownership_weight_zero_trains_nothing();
    test_sgd_reduces_ownership_loss();
    test_infer_loss_matches_infer_outputs();
    test_thread_pool_gives_identical_bits();
    std::printf("test_network: all tests passed\n");
    return 0;
}
