#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/config.hpp"
#include "azgo/grow.hpp"
#include "azgo/network.hpp"
#include "azgo/rng.hpp"

using namespace azgo;

namespace {

NetConfig makeConfig(int n, int blocks) {
    NetConfig cfg;
    cfg.boardSize = n;
    cfg.history = 2;
    cfg.blocks = blocks;
    cfg.filters = 8;
    cfg.valueHidden = 8;
    return cfg;
}

struct Batch {
    std::vector<float> input, pi, z;
};

Batch randomBatch(const NetConfig& cfg, int B, Rng& rng) {
    Batch batch;
    batch.input.resize(static_cast<size_t>(B) * cfg.planes() * cfg.area());
    for (float& v : batch.input) v = rng.uniform() < 0.5 ? 0.0f : 1.0f;
    batch.pi.assign(static_cast<size_t>(B) * cfg.actions(), 0.0f);
    for (int b = 0; b < B; ++b) batch.pi[static_cast<size_t>(b) * cfg.actions() + rng.below(cfg.actions())] = 1.0f;
    batch.z.resize(B);
    for (float& v : batch.z) v = rng.uniform() < 0.5 ? -1.0f : 1.0f;
    return batch;
}

// Gives every BN of `net` non-trivial running statistics and affine
// parameters, as a trained network would have, by training a few steps. A
// network with an ownership head trains it too, on random targets.
void trainABit(Network& net, Rng& rng) {
    const NetConfig& cfg = net.config();
    std::unique_ptr<Backend> be = makeBackend("naive", 1);
    Workspace ws(cfg, 16, *be);
    for (int i = 0; i < 3; ++i) {
        Batch b = randomBatch(cfg, 16, rng);
        std::vector<float> own;
        if (cfg.ownership) {
            own.resize(static_cast<size_t>(16) * cfg.area());
            for (float& v : own) v = static_cast<float>(rng.below(3) - 1);
        }
        net.trainStep(ws, b.input.data(), b.pi.data(), b.z.data(), 16,
                      cfg.ownership ? own.data() : nullptr, 1.5f);
        for (ParamView& p : net.params())
            for (size_t k = 0; k < p.size; ++k) p.value[k] -= 0.05f * p.grad[k];
    }
}

void expectSameOutputs(const Network& a, const Network& b, Rng& rng) {
    const NetConfig& cfg = a.config();
    const int B = 8;
    std::unique_ptr<Backend> be = makeBackend("naive", 1);
    Workspace wa(a.config(), B, *be), wb(b.config(), B, *be);
    Batch batch = randomBatch(cfg, B, rng);
    std::vector<float> la(static_cast<size_t>(B) * cfg.actions()), lb(la.size());
    std::vector<float> va(B), vb(B);
    a.infer(wa, batch.input.data(), B, la.data(), va.data());
    b.infer(wb, batch.input.data(), B, lb.data(), vb.data());
    for (size_t i = 0; i < la.size(); ++i) assert(std::fabs(la[i] - lb[i]) <= 1e-5);
    for (int i = 0; i < B; ++i) assert(std::fabs(va[i] - vb[i]) <= 1e-5);
}

void test_growth_preserves_function(int n) {
    Rng rng(100 + n);
    Network net(makeConfig(n, 2), 7);
    trainABit(net, rng);
    Network grown = growBlocks(net, 4, 99);
    assert(grown.config().blocks == 4);
    assert(grown.config().boardSize == n && grown.config().filters == 8);
    expectSameOutputs(net, grown, rng);
}

void test_same_blocks_is_exact_copy() {
    Rng rng(5);
    Network net(makeConfig(5, 3), 7);
    trainABit(net, rng);
    Network same = growBlocks(net, 3, 1);
    assert(same.config() == net.config());
    expectSameOutputs(net, same, rng);
}

void test_shrinking_throws() {
    Network net(makeConfig(5, 3), 7);
    bool threw = false;
    try {
        growBlocks(net, 2, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_shared_tensors_are_bitwise_equal() {
    Rng rng(6);
    Network net(makeConfig(5, 2), 7);
    trainABit(net, rng);
    Network grown = growBlocks(net, 4, 3);
    std::map<std::string, std::vector<float>> source;
    for (ParamView& p : net.params()) source[p.name] = std::vector<float>(p.value, p.value + p.size);
    for (BufferView& b : net.buffers()) source[b.name] = std::vector<float>(b.value, b.value + b.size);
    size_t matched = 0;
    for (ParamView& p : grown.params()) {
        auto it = source.find(p.name);
        if (it == source.end()) continue;
        assert(it->second.size() == p.size);
        assert(std::memcmp(it->second.data(), p.value, p.size * sizeof(float)) == 0);
        ++matched;
    }
    for (BufferView& b : grown.buffers()) {
        auto it = source.find(b.name);
        if (it == source.end()) continue;
        assert(std::memcmp(it->second.data(), b.value, b.size * sizeof(float)) == 0);
        ++matched;
    }
    assert(matched == source.size());
}

void test_new_blocks_are_zero_gamma_and_learn() {
    Rng rng(8);
    Network net(makeConfig(5, 1), 7);
    trainABit(net, rng);
    Network grown = growBlocks(net, 3, 11);
    for (ParamView& p : grown.params()) {
        if (p.name == "block1.bn2.gamma" || p.name == "block2.bn2.gamma" ||
            p.name == "block1.bn2.beta" || p.name == "block2.bn2.beta")
            for (size_t k = 0; k < p.size; ++k) assert(p.value[k] == 0.0f);
    }
    // A new block's convolutions differ from a fresh zero tensor: He init.
    bool nonzeroConv = false;
    for (ParamView& p : grown.params())
        if (p.name == "block2.conv1.w")
            for (size_t k = 0; k < p.size; ++k) nonzeroConv = nonzeroConv || p.value[k] != 0.0f;
    assert(nonzeroConv);

    std::unique_ptr<Backend> be = makeBackend("naive", 1);
    Workspace ws(grown.config(), 16, *be);
    Batch b = randomBatch(grown.config(), 16, rng);
    grown.trainStep(ws, b.input.data(), b.pi.data(), b.z.data(), 16);
    bool gammaGrad = false;
    for (ParamView& p : grown.params())
        if (p.name == "block1.bn2.gamma" || p.name == "block2.bn2.gamma")
            for (size_t k = 0; k < p.size; ++k) gammaGrad = gammaGrad || p.grad[k] != 0.0f;
    assert(gammaGrad);

    // After one SGD step gamma is nonzero, so the gradient now reaches the
    // new block's convolutions (it was exactly zero on the first step).
    for (ParamView& p : grown.params())
        for (size_t k = 0; k < p.size; ++k) p.value[k] -= 0.1f * p.grad[k];
    grown.trainStep(ws, b.input.data(), b.pi.data(), b.z.data(), 16);
    bool conv1Grad = false, conv2Grad = false;
    for (ParamView& p : grown.params()) {
        if (p.name == "block2.conv1.w")
            for (size_t k = 0; k < p.size; ++k) conv1Grad = conv1Grad || p.grad[k] != 0.0f;
        if (p.name == "block2.conv2.w")
            for (size_t k = 0; k < p.size; ++k) conv2Grad = conv2Grad || p.grad[k] != 0.0f;
    }
    assert(conv1Grad && conv2Grad);
}

// ---------------------------------------------------------------- widening

const ParamView& findParam(const std::vector<ParamView>& params, const std::string& name) {
    for (const ParamView& p : params)
        if (p.name == name) return p;
    assert(false && "no such parameter");
    std::abort();
}

void expectSameLoss(Network& a, Network& b, Rng& rng) {
    const NetConfig& cfg = a.config();
    const int B = 8;
    std::unique_ptr<Backend> be = makeBackend("naive", 1);
    Workspace wa(a.config(), B, *be), wb(b.config(), B, *be);
    Batch batch = randomBatch(cfg, B, rng);
    LossStats la = a.computeLoss(wa, batch.input.data(), batch.pi.data(), batch.z.data(), B);
    LossStats lb = b.computeLoss(wb, batch.input.data(), batch.pi.data(), batch.z.data(), B);
    assert(std::fabs(la.policyLoss - lb.policyLoss) <= 1e-5);
    assert(std::fabs(la.valueLoss - lb.valueLoss) <= 1e-5);
}

void test_widening_preserves_function(int n) {
    Rng rng(200 + n);
    Network net(makeConfig(n, 4), 7);
    trainABit(net, rng);
    // 8 -> 20: some old channels get two copies, some three.
    Network wide = widenFilters(net, 20, 31);
    assert(wide.config().filters == 20);
    assert(wide.config().blocks == 4 && wide.config().boardSize == n);
    assert(wide.config().history == net.config().history);
    assert(wide.config().valueHidden == net.config().valueHidden);
    expectSameOutputs(net, wide, rng);   // inference: BN running statistics
    expectSameLoss(net, wide, rng);      // training mode: BN batch statistics
}

void test_same_filters_is_exact_copy() {
    Rng rng(12);
    Network net(makeConfig(5, 2), 7);
    trainABit(net, rng);
    Network same = widenFilters(net, 8, 1);
    assert(same.config() == net.config());
    std::vector<ParamView> a = net.params(), b = same.params();
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i)
        assert(std::memcmp(a[i].value, b[i].value, a[i].size * sizeof(float)) == 0);
    expectSameOutputs(net, same, rng);
}

void test_widening_refuses_shrinking_and_too_wide() {
    Network net(makeConfig(5, 1), 7);
    bool threw = false;
    try {
        widenFilters(net, 7, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    threw = false;
    try {
        widenFilters(net, kMaxFilters + 1, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

// Two copies of one old channel start out computing the same values. They
// must not stay identical: the unequal split of the weights that read them
// sends them different gradients.
void test_widened_copies_receive_different_gradients() {
    Rng rng(13);
    Network net(makeConfig(5, 2), 7);
    trainABit(net, rng);
    const int F = 8, wideF = 12;
    Network wide = widenFilters(net, wideF, 5);

    // The stem's input planes are not split, so a new stream channel's stem
    // row is a bitwise copy of its old channel's row: that finds the pair.
    const int rowSize = wide.config().planes() * 9;
    std::vector<ParamView> params = wide.params();
    const ParamView& stem = findParam(params, "stem.conv.w");
    const int copy = F;   // the first new channel
    int original = -1;
    for (int c = 0; c < F; ++c)
        if (std::memcmp(stem.value + c * rowSize, stem.value + copy * rowSize, rowSize * sizeof(float)) == 0)
            original = c;
    assert(original >= 0);

    // The weights that read the pair hold different shares of the old weight.
    const ParamView& conv1 = findParam(params, "block0.conv1.w");
    const int taps = 9, inSize = wideF * taps;
    bool sharesDiffer = false;
    for (int t = 0; t < taps; ++t)
        sharesDiffer = sharesDiffer || conv1.value[original * taps + t] != conv1.value[copy * taps + t];
    assert(sharesDiffer);

    std::unique_ptr<Backend> be = makeBackend("naive", 1);
    Workspace ws(wide.config(), 16, *be);
    Batch b = randomBatch(wide.config(), 16, rng);
    wide.trainStep(ws, b.input.data(), b.pi.data(), b.z.data(), 16);

    // First step: the pair's activations are still equal, so the gradients
    // of the weights reading them are (mathematically) equal too; the
    // gradients of the weights writing them differ, because the readers
    // pass back different shares of dL/dx.
    bool writerGradsDiffer = false;
    for (int k = 0; k < rowSize; ++k)
        writerGradsDiffer = writerGradsDiffer || stem.grad[original * rowSize + k] != stem.grad[copy * rowSize + k];
    assert(writerGradsDiffer);

    // After one SGD step the copies compute different values, and now the
    // weights reading them get different gradients as well.
    for (ParamView& p : wide.params())
        for (size_t k = 0; k < p.size; ++k) p.value[k] -= 0.05f * p.grad[k];
    wide.trainStep(ws, b.input.data(), b.pi.data(), b.z.data(), 16);
    bool readerGradsDiffer = false;
    for (int o = 0; o < wideF; ++o)
        for (int t = 0; t < taps; ++t)
            readerGradsDiffer = readerGradsDiffer ||
                std::fabs(conv1.grad[o * inSize + original * taps + t] - conv1.grad[o * inSize + copy * taps + t]) > 1e-7f;
    assert(readerGradsDiffer);
}

void test_widen_and_deepen_compose() {
    Rng rng(14);
    Network net(makeConfig(5, 2), 7);
    trainABit(net, rng);
    Network widenThenDeepen = growBlocks(widenFilters(net, 13, 3), 4, 4);
    Network deepenThenWiden = widenFilters(growBlocks(net, 4, 4), 13, 3);
    assert(widenThenDeepen.config().blocks == 4 && widenThenDeepen.config().filters == 13);
    assert(deepenThenWiden.config() == widenThenDeepen.config());
    expectSameOutputs(net, widenThenDeepen, rng);
    expectSameOutputs(net, deepenThenWiden, rng);
    expectSameLoss(net, widenThenDeepen, rng);
    expectSameLoss(net, deepenThenWiden, rng);
}

void test_widening_is_deterministic() {
    Rng rng(15);
    Network net(makeConfig(5, 2), 7);
    trainABit(net, rng);
    Network a = widenFilters(net, 16, 9), b = widenFilters(net, 16, 9), c = widenFilters(net, 16, 10);
    std::vector<ParamView> pa = a.params(), pb = b.params(), pc = c.params();
    bool seedMatters = false;
    for (size_t i = 0; i < pa.size(); ++i) {
        assert(std::memcmp(pa[i].value, pb[i].value, pa[i].size * sizeof(float)) == 0);
        seedMatters = seedMatters || std::memcmp(pa[i].value, pc[i].value, pa[i].size * sizeof(float)) != 0;
    }
    std::vector<BufferView> ba = a.buffers(), bb = b.buffers();
    for (size_t i = 0; i < ba.size(); ++i)
        assert(std::memcmp(ba[i].value, bb[i].value, ba[i].size * sizeof(float)) == 0);
    assert(seedMatters);
}

// ---------------------------------------------------------------- ownership

NetConfig withOwnership(NetConfig cfg) {
    cfg.ownership = true;
    return cfg;
}

// Inference outputs of all three heads, for exact or near comparisons.
struct Outputs {
    std::vector<float> logits, values, ownership;
};

Outputs outputsOf(const Network& net, const Batch& batch, int B) {
    const NetConfig& cfg = net.config();
    std::unique_ptr<Backend> be = makeBackend("naive", 1);
    Workspace ws(cfg, B, *be);
    Outputs out;
    out.logits.resize(static_cast<size_t>(B) * cfg.actions());
    out.values.resize(B);
    if (cfg.ownership) out.ownership.resize(static_cast<size_t>(B) * cfg.area());
    net.infer(ws, batch.input.data(), B, out.logits.data(), out.values.data(),
              cfg.ownership ? out.ownership.data() : nullptr);
    return out;
}

double maxDifference(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    double worst = 0;
    for (size_t i = 0; i < a.size(); ++i) worst = std::fmax(worst, std::fabs(a[i] - b[i]));
    return worst;
}

// A new head cannot change the policy or the value: bit for bit the same
// outputs, every old tensor copied, and a second head refused.
void test_add_ownership_head_keeps_policy_and_value() {
    Rng rng(16);
    Network net(makeConfig(5, 2), 7);
    trainABit(net, rng);
    Network owner = addOwnershipHead(net, 3);
    assert(owner.config() == withOwnership(net.config()));
    std::vector<ParamView> a = net.params(), b = owner.params();
    assert(b.size() == a.size() + 3 && b.back().name == "ownership.bn.beta");
    for (size_t i = 0; i < a.size(); ++i)
        assert(a[i].name == b[i].name && std::memcmp(a[i].value, b[i].value, a[i].size * sizeof(float)) == 0);
    std::vector<BufferView> ba = net.buffers(), bb = owner.buffers();
    for (size_t i = 0; i < ba.size(); ++i)
        assert(std::memcmp(ba[i].value, bb[i].value, ba[i].size * sizeof(float)) == 0);

    Batch batch = randomBatch(net.config(), 8, rng);
    const Outputs before = outputsOf(net, batch, 8), after = outputsOf(owner, batch, 8);
    assert(after.logits == before.logits && after.values == before.values);
    for (float o : after.ownership) assert(std::isfinite(o) && o >= -1.0f && o <= 1.0f);

    bool threw = false;
    try {
        addOwnershipHead(owner, 3);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

// Deepening copies an existing head by name: all three heads bit for bit.
void test_deepening_carries_the_ownership_head() {
    Rng rng(17);
    Network net(withOwnership(makeConfig(5, 2)), 7);
    trainABit(net, rng);
    Network deep = growBlocks(net, 4, 99);
    assert(deep.config().ownership && deep.config().blocks == 4);
    Batch batch = randomBatch(net.config(), 8, rng);
    const Outputs a = outputsOf(net, batch, 8), b = outputsOf(deep, batch, 8);
    assert(a.logits == b.logits && a.values == b.values && a.ownership == b.ownership);
}

// Widening splits the head's input weights like the other heads': all three
// heads agree within float rounding, in inference and in the training loss.
// A network without the head widens exactly as it did before the head existed.
void test_widening_carries_the_ownership_head() {
    Rng rng(18);
    Network net(withOwnership(makeConfig(5, 3)), 7);
    trainABit(net, rng);
    Network wide = widenFilters(net, 20, 31);
    assert(wide.config().ownership && wide.config().filters == 20);
    Batch batch = randomBatch(net.config(), 8, rng);
    const Outputs a = outputsOf(net, batch, 8), b = outputsOf(wide, batch, 8);
    assert(maxDifference(a.logits, b.logits) <= 1e-5 && maxDifference(a.values, b.values) <= 1e-5);
    assert(maxDifference(a.ownership, b.ownership) <= 1e-5);

    std::vector<float> own(static_cast<size_t>(8) * net.config().area());
    for (float& v : own) v = static_cast<float>(rng.below(3) - 1);
    std::unique_ptr<Backend> be = makeBackend("naive", 1);
    Workspace wa(net.config(), 8, *be), wb(wide.config(), 8, *be);
    const LossStats la = net.computeLoss(wa, batch.input.data(), batch.pi.data(), batch.z.data(), 8,
                                         own.data(), 1.5f);
    const LossStats lb = wide.computeLoss(wb, batch.input.data(), batch.pi.data(), batch.z.data(), 8,
                                          own.data(), 1.5f);
    assert(std::fabs(la.ownershipLoss - lb.ownershipLoss) <= 1e-5);
    assert(std::fabs(la.valueLoss - lb.valueLoss) <= 1e-5);

    // The same trunk without the head widens to the same tensors (the head's
    // random shares are drawn after every other layer's).
    Network plain(makeConfig(5, 3), 7);
    trainABit(plain, rng);
    Network plainWide = widenFilters(plain, 20, 31);
    Network ownerWide = widenFilters(addOwnershipHead(plain, 5), 20, 31);
    std::vector<ParamView> pp = plainWide.params(), po = ownerWide.params();
    for (size_t i = 0; i < pp.size(); ++i)
        assert(pp[i].name == po[i].name && std::memcmp(pp[i].value, po[i].value, pp[i].size * sizeof(float)) == 0);
}

// A reference network from a smaller run plays in a bigger one: fitToShape
// widens, deepens and adds the head, and the result computes what the
// original did (widening within float rounding, the rest bit for bit).
void test_fit_to_shape_grows_a_smaller_network() {
    Rng rng(19);
    Network small(makeConfig(5, 1), 7);
    trainABit(small, rng);
    NetConfig target = withOwnership(makeConfig(5, 3));
    target.filters = 12;
    Network fitted = fitToShape(small, target, 41);
    assert(fitted.config() == target);
    Batch batch = randomBatch(small.config(), 8, rng);
    const Outputs a = outputsOf(small, batch, 8), b = outputsOf(fitted, batch, 8);
    assert(maxDifference(a.logits, b.logits) <= 1e-5 && maxDifference(a.values, b.values) <= 1e-5);

    // Without widening it is exact: deepening and a new head change nothing.
    NetConfig deeper = withOwnership(makeConfig(5, 3));
    Network deep = fitToShape(small, deeper, 41);
    const Outputs c = outputsOf(deep, batch, 8);
    assert(c.logits == a.logits && c.values == a.values);

    // The same seed gives the same network; the same shape gives a copy.
    Network again = fitToShape(small, target, 41);
    std::vector<ParamView> p1 = fitted.params(), p2 = again.params();
    for (size_t i = 0; i < p1.size(); ++i)
        assert(std::memcmp(p1[i].value, p2[i].value, p1[i].size * sizeof(float)) == 0);
    const Outputs d = outputsOf(fitToShape(small, small.config(), 41), batch, 8);
    assert(d.logits == a.logits && d.values == a.values);
}

// A head the target lacks is dropped: nothing reads it, so policy and value
// are unchanged bit for bit.
void test_fit_to_shape_drops_an_extra_ownership_head() {
    Rng rng(20);
    Network owner(withOwnership(makeConfig(5, 2)), 7);
    trainABit(owner, rng);
    Network plain = fitToShape(owner, makeConfig(5, 2), 3);
    assert(plain.config() == makeConfig(5, 2));
    Batch batch = randomBatch(owner.config(), 8, rng);
    const Outputs a = outputsOf(owner, batch, 8), b = outputsOf(plain, batch, 8);
    assert(a.logits == b.logits && a.values == b.values);
}

// Growing only adds: a bigger network, or another board, history or value
// head width, is refused with the reason.
void test_fit_to_shape_refusals() {
    Network net(makeConfig(5, 2), 7);
    auto reason = [&](NetConfig target) -> std::string {
        try {
            fitToShape(net, target, 1);
        } catch (const std::invalid_argument& e) {
            return e.what();
        }
        return "";
    };
    NetConfig t = makeConfig(5, 1);
    assert(reason(t).find("2 blocks") != std::string::npos);
    t = makeConfig(5, 2);
    t.filters = 4;
    assert(reason(t).find("8 filters") != std::string::npos);
    t = makeConfig(9, 2);
    assert(reason(t).find("board size") != std::string::npos);
    t = makeConfig(5, 2);
    t.history = 3;
    assert(reason(t).find("history") != std::string::npos);
    t = makeConfig(5, 2);
    t.valueHidden = 16;
    assert(reason(t).find("value_hidden") != std::string::npos);
}

} // namespace

int main() {
    test_growth_preserves_function(5);
    test_growth_preserves_function(9);
    test_same_blocks_is_exact_copy();
    test_shrinking_throws();
    test_shared_tensors_are_bitwise_equal();
    test_new_blocks_are_zero_gamma_and_learn();
    test_widening_preserves_function(5);
    test_widening_preserves_function(9);
    test_same_filters_is_exact_copy();
    test_widening_refuses_shrinking_and_too_wide();
    test_widened_copies_receive_different_gradients();
    test_widen_and_deepen_compose();
    test_widening_is_deterministic();
    test_add_ownership_head_keeps_policy_and_value();
    test_deepening_carries_the_ownership_head();
    test_widening_carries_the_ownership_head();
    test_fit_to_shape_grows_a_smaller_network();
    test_fit_to_shape_drops_an_extra_ownership_head();
    test_fit_to_shape_refusals();
    std::printf("test_grow: all tests passed\n");
    return 0;
}
