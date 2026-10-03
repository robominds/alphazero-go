#include "azgo/network.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include "azgo/board.hpp"
#include "azgo/rng.hpp"

namespace azgo {

bool NetConfig::operator==(const NetConfig& o) const {
    return boardSize == o.boardSize && history == o.history && blocks == o.blocks &&
           filters == o.filters && valueHidden == o.valueHidden && ownership == o.ownership;
}

// ---------------------------------------------------------------- Workspace

Workspace::Workspace(const NetConfig& cfg, int maxBatch, const Backend& backend)
    : cfg_(cfg), maxBatch_(maxBatch), backend_(&backend) {
    assert(maxBatch >= 1);
    const size_t M = static_cast<size_t>(maxBatch) * cfg.area();    // pixels per channel
    const size_t F = static_cast<size_t>(cfg.filters);
    const size_t trunk = F * M;
    const size_t actions = static_cast<size_t>(maxBatch) * cfg.actions();
    const size_t hidden = static_cast<size_t>(maxBatch) * cfg.valueHidden;

    input_.resize(static_cast<size_t>(cfg.planes()) * M);
    // One im2col buffer serves every 3x3 conv (stem: planes inputs, blocks:
    // F inputs); backward rebuilds a conv's columns right before using them.
    col_.resize(static_cast<size_t>(std::max(cfg.planes(), cfg.filters)) * 9 * M);
    dcol_.resize(col_.size());
    convOut_.resize(std::max<size_t>(F, 2) * M);   // the policy conv has 2 outputs even if F == 1

    stemXhat_.resize(trunk);
    stemInvStd_.resize(F);
    stemOut_.resize(trunk);
    blocks_.resize(cfg.blocks);
    for (BlockBuffers& b : blocks_) {
        b.xhat1.resize(trunk);
        b.invStd1.resize(F);
        b.act1.resize(trunk);
        b.xhat2.resize(trunk);
        b.invStd2.resize(F);
        b.out.resize(trunk);
    }
    policyXhat_.resize(2 * M);
    policyInvStd_.resize(2);
    policyAct_.resize(2 * M);
    policyFlat_.resize(2 * M);
    logits_.resize(actions);
    valueXhat_.resize(M);
    valueInvStd_.resize(1);
    valueAct_.resize(M);
    valueHidden_.resize(hidden);
    values_.resize(maxBatch);

    dTrunk_.resize(trunk);
    dTmp_.resize(trunk);
    dTmp2_.resize(trunk);
    dLogits_.resize(actions);
    dPolicyFlat_.resize(2 * M);
    dPolicyAct_.resize(2 * M);
    dValues_.resize(maxBatch);
    dValueHidden_.resize(hidden);
    dValueAct_.resize(M);
}

// ---------------------------------------------------------------- Network

Network::Network(const NetConfig& cfg, uint64_t seed)
    : cfg_(cfg),
      stemConv_(cfg.planes(), cfg.filters, 3),
      stemBn_(cfg.filters),
      policyConv_(cfg.filters, 2, 1),
      policyBn_(2),
      policyFc_(2 * cfg.area(), cfg.actions()),
      valueConv_(cfg.filters, 1, 1),
      valueBn_(1),
      valueFc1_(cfg.area(), cfg.valueHidden),
      valueFc2_(cfg.valueHidden, 1) {
    assert(cfg.boardSize >= kMinBoardSize && cfg.boardSize <= kMaxBoardSize);
    assert(cfg.history >= 1 && cfg.blocks >= 0 && cfg.filters >= 1 && cfg.valueHidden >= 1);
    const int F = cfg.filters;
    for (int i = 0; i < cfg.blocks; ++i) {
        blocks_.push_back({Conv2d(F, F, 3), BatchNorm2d(F), Conv2d(F, F, 3), BatchNorm2d(F)});
    }
    // One generator, layers in params() order, so a seed fixes every weight.
    Rng rng(seed);
    stemConv_.init(rng);
    for (ResidualBlock& b : blocks_) {
        b.conv1.init(rng);
        b.conv2.init(rng);
    }
    policyConv_.init(rng);
    policyFc_.init(rng);
    valueConv_.init(rng);
    valueFc1_.init(rng);
    valueFc2_.init(rng);
}

std::vector<ParamView> Network::params() {
    std::vector<ParamView> out;
    auto conv = [&out](const std::string& prefix, Conv2d& c) {
        out.push_back({prefix + ".w", c.w.data(), c.gw.data(), c.w.size(), true});
    };
    auto bn = [&out](const std::string& prefix, BatchNorm2d& b) {
        out.push_back({prefix + ".gamma", b.gamma.data(), b.ggamma.data(), b.gamma.size(), false});
        out.push_back({prefix + ".beta", b.beta.data(), b.gbeta.data(), b.beta.size(), false});
    };
    auto linear = [&out](const std::string& prefix, Linear& l) {
        out.push_back({prefix + ".w", l.w.data(), l.gw.data(), l.w.size(), true});
        out.push_back({prefix + ".b", l.b.data(), l.gb.data(), l.b.size(), false});
    };
    conv("stem.conv", stemConv_);
    bn("stem.bn", stemBn_);
    for (size_t i = 0; i < blocks_.size(); ++i) {
        const std::string prefix = "block" + std::to_string(i);
        conv(prefix + ".conv1", blocks_[i].conv1);
        bn(prefix + ".bn1", blocks_[i].bn1);
        conv(prefix + ".conv2", blocks_[i].conv2);
        bn(prefix + ".bn2", blocks_[i].bn2);
    }
    conv("policy.conv", policyConv_);
    bn("policy.bn", policyBn_);
    linear("policy.fc", policyFc_);
    conv("value.conv", valueConv_);
    bn("value.bn", valueBn_);
    linear("value.fc1", valueFc1_);
    linear("value.fc2", valueFc2_);
    return out;
}

std::vector<BufferView> Network::buffers() {
    std::vector<BufferView> out;
    auto bn = [&out](const std::string& prefix, BatchNorm2d& b) {
        out.push_back({prefix + ".runMean", b.runMean.data(), b.runMean.size()});
        out.push_back({prefix + ".runVar", b.runVar.data(), b.runVar.size()});
    };
    bn("stem.bn", stemBn_);
    for (size_t i = 0; i < blocks_.size(); ++i) {
        const std::string prefix = "block" + std::to_string(i);
        bn(prefix + ".bn1", blocks_[i].bn1);
        bn(prefix + ".bn2", blocks_[i].bn2);
    }
    bn("policy.bn", policyBn_);
    bn("value.bn", valueBn_);
    return out;
}

void Network::zeroGrad() {
    for (ParamView& p : params()) std::fill(p.grad, p.grad + p.size, 0.0f);
}

void Network::infer(Workspace& ws, const float* input, int B, float* logits, float* values) const {
    // Mode::Infer only reads parameters (BN uses its running statistics via
    // the const forwardInfer), so this cast never writes to the shared
    // network; it lets a single forward() serve inference and training.
    const_cast<Network*>(this)->forward(ws, input, B, Mode::Infer);
    std::copy(ws.logits_.begin(), ws.logits_.begin() + static_cast<size_t>(B) * cfg_.actions(), logits);
    std::copy(ws.values_.begin(), ws.values_.begin() + B, values);
}

LossStats Network::trainStep(Workspace& ws, const float* input, const float* pi, const float* z,
                             int B) {
    zeroGrad();
    forward(ws, input, B, Mode::Train);
    LossStats stats = loss(ws, pi, z, B, true);
    backward(ws, B);
    return stats;
}

LossStats Network::computeLoss(Workspace& ws, const float* input, const float* pi, const float* z,
                               int B) {
    forward(ws, input, B, Mode::Probe);
    return loss(ws, pi, z, B, false);
}

void Network::forward(Workspace& ws, const float* input, int B, Mode mode) {
    assert(B >= 1 && B <= ws.maxBatch_);
    assert(ws.cfg_ == cfg_);
    const Backend& be = *ws.backend_;
    const int n = cfg_.boardSize, A = cfg_.area(), P = cfg_.planes();
    const int M = B * A;
    const size_t trunk = static_cast<size_t>(cfg_.filters) * M;
    float* col = ws.col_.data();
    float* convOut = ws.convOut_.data();

    // [B][planes][A] -> [planes][B][A]. Channel-major lets a conv treat the
    // whole batch as one wide matrix: one gemm per layer, not one per sample.
    for (int b = 0; b < B; ++b)
        for (int p = 0; p < P; ++p)
            std::copy(input + (static_cast<size_t>(b) * P + p) * A,
                      input + (static_cast<size_t>(b) * P + p + 1) * A,
                      ws.input_.data() + (static_cast<size_t>(p) * B + b) * A);

    auto norm = [&](BatchNorm2d& bn, const float* x, float* y, float* xhat, float* invStd) {
        if (mode == Mode::Infer) bn.forwardInfer(x, M, y);
        else bn.forwardTrain(x, M, y, xhat, invStd, mode == Mode::Train);
    };

    // Stem: conv3x3 -> BN -> ReLU.
    stemConv_.forward(be, ws.input_.data(), B, n, col, convOut);
    norm(stemBn_, convOut, ws.stemOut_.data(), ws.stemXhat_.data(), ws.stemInvStd_.data());
    reluForward(ws.stemOut_.data(), trunk);

    // Residual blocks: conv -> BN -> ReLU -> conv -> BN -> + input -> ReLU.
    const float* x = ws.stemOut_.data();
    for (size_t i = 0; i < blocks_.size(); ++i) {
        ResidualBlock& blk = blocks_[i];
        Workspace::BlockBuffers& buf = ws.blocks_[i];
        blk.conv1.forward(be, x, B, n, col, convOut);
        norm(blk.bn1, convOut, buf.act1.data(), buf.xhat1.data(), buf.invStd1.data());
        reluForward(buf.act1.data(), trunk);
        blk.conv2.forward(be, buf.act1.data(), B, n, col, convOut);
        norm(blk.bn2, convOut, buf.out.data(), buf.xhat2.data(), buf.invStd2.data());
        for (size_t j = 0; j < trunk; ++j) buf.out[j] += x[j];   // the skip connection
        reluForward(buf.out.data(), trunk);
        x = buf.out.data();
    }

    // Policy head: conv1x1 (F -> 2) -> BN -> ReLU -> per-sample flatten -> Linear.
    policyConv_.forward(be, x, B, n, col, convOut);
    norm(policyBn_, convOut, ws.policyAct_.data(), ws.policyXhat_.data(), ws.policyInvStd_.data());
    reluForward(ws.policyAct_.data(), 2 * static_cast<size_t>(M));
    // [2][B][A] -> [B][2A], feature index c * A + p.
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < 2; ++c)
            std::copy(ws.policyAct_.data() + (static_cast<size_t>(c) * B + b) * A,
                      ws.policyAct_.data() + (static_cast<size_t>(c) * B + b + 1) * A,
                      ws.policyFlat_.data() + static_cast<size_t>(b) * 2 * A + static_cast<size_t>(c) * A);
    policyFc_.forward(be, ws.policyFlat_.data(), B, ws.logits_.data());

    // Value head: conv1x1 (F -> 1) -> BN -> ReLU -> Linear(A -> V) -> ReLU -> Linear(V -> 1) -> tanh.
    valueConv_.forward(be, x, B, n, col, convOut);
    norm(valueBn_, convOut, ws.valueAct_.data(), ws.valueXhat_.data(), ws.valueInvStd_.data());
    reluForward(ws.valueAct_.data(), M);
    // With a single channel, [1][B][A] already is [B][A]: one row per sample.
    valueFc1_.forward(be, ws.valueAct_.data(), B, ws.valueHidden_.data());
    reluForward(ws.valueHidden_.data(), static_cast<size_t>(B) * cfg_.valueHidden);
    valueFc2_.forward(be, ws.valueHidden_.data(), B, ws.values_.data());
    for (int b = 0; b < B; ++b) ws.values_[b] = std::tanh(ws.values_[b]);
}

LossStats Network::loss(Workspace& ws, const float* pi, const float* z, int B,
                        bool withGradients) const {
    const int actions = cfg_.actions();
    double policySum = 0.0, valueSum = 0.0;
    for (int b = 0; b < B; ++b) {
        const float* logits = ws.logits_.data() + static_cast<size_t>(b) * actions;
        const float* target = pi + static_cast<size_t>(b) * actions;
        // log softmax(l)_a = l_a - logSumExp(l). Subtracting the largest
        // logit before exp keeps every exponent <= 0, so nothing overflows.
        const float maxLogit = *std::max_element(logits, logits + actions);
        double sumExp = 0.0;
        for (int a = 0; a < actions; ++a) sumExp += std::exp(static_cast<double>(logits[a]) - maxLogit);
        const double logZ = maxLogit + std::log(sumExp);
        for (int a = 0; a < actions; ++a) policySum -= target[a] * (logits[a] - logZ);

        const double v = ws.values_[b];
        valueSum += (z[b] - v) * (z[b] - v);

        if (withGradients) {
            // Cross-entropy through softmax: dL/dlogit = softmax - pi (pi sums to 1).
            float* d = ws.dLogits_.data() + static_cast<size_t>(b) * actions;
            for (int a = 0; a < actions; ++a) {
                d[a] = static_cast<float>((std::exp(logits[a] - logZ) - target[a]) / B);
            }
            // (z - v)^2 with v = tanh(u): dL/du = 2 (v - z) * (1 - v^2).
            ws.dValues_[b] = static_cast<float>(2.0 * (v - z[b]) * (1.0 - v * v) / B);
        }
    }
    LossStats stats;
    stats.policyLoss = static_cast<float>(policySum / B);
    stats.valueLoss = static_cast<float>(valueSum / B);
    return stats;
}

void Network::backward(Workspace& ws, int B) {
    const Backend& be = *ws.backend_;
    const int n = cfg_.boardSize, A = cfg_.area(), F = cfg_.filters;
    const int M = B * A;
    const size_t trunk = static_cast<size_t>(F) * M;
    float* col = ws.col_.data();
    float* dcol = ws.dcol_.data();
    float* dTrunk = ws.dTrunk_.data();
    float* dTmp = ws.dTmp_.data();
    float* dTmp2 = ws.dTmp2_.data();
    const float* trunkOut = blocks_.empty() ? ws.stemOut_.data() : ws.blocks_.back().out.data();

    // Value head, from the loss back to the trunk output. (1x1 convs ignore col.)
    valueFc2_.backward(be, ws.valueHidden_.data(), ws.dValues_.data(), B, ws.dValueHidden_.data());
    reluBackward(ws.valueHidden_.data(), ws.dValueHidden_.data(), static_cast<size_t>(B) * cfg_.valueHidden);
    valueFc1_.backward(be, ws.valueAct_.data(), ws.dValueHidden_.data(), B, ws.dValueAct_.data());
    reluBackward(ws.valueAct_.data(), ws.dValueAct_.data(), M);
    valueBn_.backward(ws.dValueAct_.data(), ws.valueXhat_.data(), ws.valueInvStd_.data(), M,
                      ws.dValueAct_.data());
    valueConv_.backward(be, trunkOut, col, ws.dValueAct_.data(), B, n, dcol, dTrunk);

    // Policy head.
    policyFc_.backward(be, ws.policyFlat_.data(), ws.dLogits_.data(), B, ws.dPolicyFlat_.data());
    for (int b = 0; b < B; ++b)                   // [B][2A] -> [2][B][A], undoing the flatten
        for (int c = 0; c < 2; ++c)
            std::copy(ws.dPolicyFlat_.data() + static_cast<size_t>(b) * 2 * A + static_cast<size_t>(c) * A,
                      ws.dPolicyFlat_.data() + static_cast<size_t>(b) * 2 * A + static_cast<size_t>(c + 1) * A,
                      ws.dPolicyAct_.data() + (static_cast<size_t>(c) * B + b) * A);
    reluBackward(ws.policyAct_.data(), ws.dPolicyAct_.data(), 2 * static_cast<size_t>(M));
    policyBn_.backward(ws.dPolicyAct_.data(), ws.policyXhat_.data(), ws.policyInvStd_.data(), M,
                       ws.dPolicyAct_.data());
    policyConv_.backward(be, trunkOut, col, ws.dPolicyAct_.data(), B, n, dcol, dTmp);
    for (size_t j = 0; j < trunk; ++j) dTrunk[j] += dTmp[j];   // the trunk feeds both heads

    // Residual blocks in reverse; dTrunk holds dL/d(block output). The
    // shared col buffer was overwritten by later convs, so each 3x3 conv's
    // columns are rebuilt from its saved input before its backward.
    for (size_t i = blocks_.size(); i-- > 0;) {
        ResidualBlock& blk = blocks_[i];
        Workspace::BlockBuffers& buf = ws.blocks_[i];
        const float* in = i == 0 ? ws.stemOut_.data() : ws.blocks_[i - 1].out.data();
        reluBackward(buf.out.data(), dTrunk, trunk);           // now dL/d(bn2 output + skip)
        blk.bn2.backward(dTrunk, buf.xhat2.data(), buf.invStd2.data(), M, dTmp);
        be.im2col(buf.act1.data(), F, B, n, 3, col);
        blk.conv2.backward(be, buf.act1.data(), col, dTmp, B, n, dcol, dTmp2);
        reluBackward(buf.act1.data(), dTmp2, trunk);
        blk.bn1.backward(dTmp2, buf.xhat1.data(), buf.invStd1.data(), M, dTmp2);
        be.im2col(in, F, B, n, 3, col);
        blk.conv1.backward(be, in, col, dTmp2, B, n, dcol, dTmp);
        // The skip path passes dTrunk to the block input unchanged; the conv
        // path adds its own contribution.
        for (size_t j = 0; j < trunk; ++j) dTrunk[j] += dTmp[j];
    }

    // Stem. No gradient is needed for the input planes themselves.
    reluBackward(ws.stemOut_.data(), dTrunk, trunk);
    stemBn_.backward(dTrunk, ws.stemXhat_.data(), ws.stemInvStd_.data(), M, dTrunk);
    be.im2col(ws.input_.data(), cfg_.planes(), B, n, 3, col);
    stemConv_.backward(be, ws.input_.data(), col, dTrunk, B, n, dcol, nullptr);
}

} // namespace azgo
