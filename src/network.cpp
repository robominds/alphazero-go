#include "azgo/network.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include "azgo/board.hpp"
#include "azgo/parallel.hpp"
#include "azgo/rng.hpp"

namespace azgo {

bool NetConfig::operator==(const NetConfig& o) const {
    return boardSize == o.boardSize && history == o.history && blocks == o.blocks &&
           filters == o.filters && valueHidden == o.valueHidden && ownership == o.ownership;
}

// ---------------------------------------------------------------- Workspace

Workspace::Workspace(const NetConfig& cfg, int maxBatch, const Backend& backend, ThreadPool* pool)
    : cfg_(cfg), maxBatch_(maxBatch), backend_(&backend), pool_(pool) {
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
    logZ_.resize(maxBatch);

    dTrunk_.resize(trunk);
    dTmp_.resize(trunk);
    dTmp2_.resize(trunk);
    dLogits_.resize(actions);
    dPolicyFlat_.resize(2 * M);
    dPolicyAct_.resize(2 * M);
    dValues_.resize(maxBatch);
    dValueHidden_.resize(hidden);
    dValueAct_.resize(M);
    if (cfg.ownership) {
        ownershipXhat_.resize(M);
        ownershipInvStd_.resize(1);
        ownership_.resize(M);
        dOwnership_.resize(M);
    }
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
      valueFc2_(cfg.valueHidden, 1),
      ownershipConv_(cfg.filters, 1, 1),
      ownershipBn_(1) {
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
    // Last, so a network with the head has the same other weights as one
    // without it for the same seed.
    if (cfg.ownership) ownershipConv_.init(rng);
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
    if (cfg_.ownership) {
        conv("ownership.conv", ownershipConv_);
        bn("ownership.bn", ownershipBn_);
    }
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
    if (cfg_.ownership) bn("ownership.bn", ownershipBn_);
    return out;
}

void Network::zeroGrad() {
    for (ParamView& p : params()) std::fill(p.grad, p.grad + p.size, 0.0f);
}

void Network::infer(Workspace& ws, const float* input, int B, float* logits, float* values,
                    float* ownership) const {
    assert(ownership == nullptr || cfg_.ownership);
    // Mode::Infer only reads parameters (BN uses its running statistics via
    // the const forwardInfer), so this cast never writes to the shared
    // network; it lets a single forward() serve inference and training.
    const_cast<Network*>(this)->forward(ws, input, B, Mode::Infer);
    std::copy(ws.logits_.begin(), ws.logits_.begin() + static_cast<size_t>(B) * cfg_.actions(), logits);
    std::copy(ws.values_.begin(), ws.values_.begin() + B, values);
    if (ownership && cfg_.ownership)
        std::copy(ws.ownership_.begin(), ws.ownership_.begin() + static_cast<size_t>(B) * cfg_.area(),
                  ownership);
}

LossStats Network::trainStep(Workspace& ws, const float* input, const float* pi, const float* z,
                             int B, const float* ownership, float ownershipWeight) {
    assert(ownership == nullptr || cfg_.ownership);
    zeroGrad();
    forward(ws, input, B, Mode::Train);
    LossStats stats = loss(ws, pi, z, ownership, ownershipWeight, B, true);
    backward(ws, B, cfg_.ownership && ownership != nullptr && ownershipWeight != 0.0f);
    return stats;
}

LossStats Network::computeLoss(Workspace& ws, const float* input, const float* pi, const float* z,
                               int B, const float* ownership, float ownershipWeight) {
    assert(ownership == nullptr || cfg_.ownership);
    forward(ws, input, B, Mode::Probe);
    return loss(ws, pi, z, ownership, ownershipWeight, B, false);
}

LossStats Network::inferLoss(Workspace& ws, const float* input, const float* pi, const float* z,
                             int B, const float* ownership) const {
    assert(ownership == nullptr || cfg_.ownership);
    // As in infer: Mode::Infer only reads the parameters.
    const_cast<Network*>(this)->forward(ws, input, B, Mode::Infer);
    return loss(ws, pi, z, ownership, 0.0f, B, false);
}

namespace {

// Samples per chunk of a per-sample loop on the pool.
constexpr size_t kSampleChunk = 16;

// dst[j] += src[j] for j < count, in fixed chunks on the pool.
void addInto(float* dst, const float* src, size_t count, ThreadPool* pool) {
    parallelRanges(pool, count, kElementChunk, [&](size_t begin, size_t end) {
        for (size_t j = begin; j < end; ++j) dst[j] += src[j];
    });
}

} // namespace

void Network::forward(Workspace& ws, const float* input, int B, Mode mode) {
    assert(B >= 1 && B <= ws.maxBatch_);
    assert(ws.cfg_ == cfg_);
    const Backend& be = *ws.backend_;
    ThreadPool* pool = ws.pool_;
    const int n = cfg_.boardSize, A = cfg_.area(), P = cfg_.planes();
    const int M = B * A;
    const size_t trunk = static_cast<size_t>(cfg_.filters) * M;
    float* col = ws.col_.data();
    float* convOut = ws.convOut_.data();

    // [B][planes][A] -> [planes][B][A]. Channel-major lets a conv treat the
    // whole batch as one wide matrix: one gemm per layer, not one per sample.
    parallelChunks(pool, P, [&](int p) {
        for (int b = 0; b < B; ++b)
            std::copy(input + (static_cast<size_t>(b) * P + p) * A,
                      input + (static_cast<size_t>(b) * P + p + 1) * A,
                      ws.input_.data() + (static_cast<size_t>(p) * B + b) * A);
    });

    auto norm = [&](BatchNorm2d& bn, const float* x, float* y, float* xhat, float* invStd) {
        if (mode == Mode::Infer) bn.forwardInfer(x, M, y, pool);
        else bn.forwardTrain(x, M, y, xhat, invStd, mode == Mode::Train, pool);
    };

    // Stem: conv3x3 -> BN -> ReLU.
    stemConv_.forward(be, ws.input_.data(), B, n, col, convOut, pool);
    norm(stemBn_, convOut, ws.stemOut_.data(), ws.stemXhat_.data(), ws.stemInvStd_.data());
    reluForward(ws.stemOut_.data(), trunk, pool);

    // Residual blocks: conv -> BN -> ReLU -> conv -> BN -> + input -> ReLU.
    const float* x = ws.stemOut_.data();
    for (size_t i = 0; i < blocks_.size(); ++i) {
        ResidualBlock& blk = blocks_[i];
        Workspace::BlockBuffers& buf = ws.blocks_[i];
        blk.conv1.forward(be, x, B, n, col, convOut, pool);
        norm(blk.bn1, convOut, buf.act1.data(), buf.xhat1.data(), buf.invStd1.data());
        reluForward(buf.act1.data(), trunk, pool);
        blk.conv2.forward(be, buf.act1.data(), B, n, col, convOut, pool);
        norm(blk.bn2, convOut, buf.out.data(), buf.xhat2.data(), buf.invStd2.data());
        addInto(buf.out.data(), x, trunk, pool);                 // the skip connection
        reluForward(buf.out.data(), trunk, pool);
        x = buf.out.data();
    }

    // Policy head: conv1x1 (F -> 2) -> BN -> ReLU -> per-sample flatten -> Linear.
    policyConv_.forward(be, x, B, n, col, convOut, pool);
    norm(policyBn_, convOut, ws.policyAct_.data(), ws.policyXhat_.data(), ws.policyInvStd_.data());
    reluForward(ws.policyAct_.data(), 2 * static_cast<size_t>(M), pool);
    // [2][B][A] -> [B][2A], feature index c * A + p.
    parallelRanges(pool, static_cast<size_t>(B), kSampleChunk, [&](size_t b0, size_t b1) {
        for (size_t b = b0; b < b1; ++b)
            for (int c = 0; c < 2; ++c)
                std::copy(ws.policyAct_.data() + (static_cast<size_t>(c) * B + b) * A,
                          ws.policyAct_.data() + (static_cast<size_t>(c) * B + b + 1) * A,
                          ws.policyFlat_.data() + b * 2 * A + static_cast<size_t>(c) * A);
    });
    policyFc_.forward(be, ws.policyFlat_.data(), B, ws.logits_.data());

    // Value head: conv1x1 (F -> 1) -> BN -> ReLU -> Linear(A -> V) -> ReLU -> Linear(V -> 1) -> tanh.
    valueConv_.forward(be, x, B, n, col, convOut, pool);
    norm(valueBn_, convOut, ws.valueAct_.data(), ws.valueXhat_.data(), ws.valueInvStd_.data());
    reluForward(ws.valueAct_.data(), M, pool);
    // With a single channel, [1][B][A] already is [B][A]: one row per sample.
    valueFc1_.forward(be, ws.valueAct_.data(), B, ws.valueHidden_.data());
    reluForward(ws.valueHidden_.data(), static_cast<size_t>(B) * cfg_.valueHidden, pool);
    valueFc2_.forward(be, ws.valueHidden_.data(), B, ws.values_.data());
    for (int b = 0; b < B; ++b) ws.values_[b] = std::tanh(ws.values_[b]);

    // Ownership head: conv1x1 (F -> 1) -> BN -> tanh at every point, no
    // ReLU and no linear layer: each point's owner is read off the trunk's
    // features at that point. [1][B][A] already is [B][A].
    if (cfg_.ownership) {
        ownershipConv_.forward(be, x, B, n, col, convOut, pool);
        norm(ownershipBn_, convOut, ws.ownership_.data(), ws.ownershipXhat_.data(),
             ws.ownershipInvStd_.data());
        parallelRanges(pool, static_cast<size_t>(M), kElementChunk, [&](size_t begin, size_t end) {
            for (size_t j = begin; j < end; ++j) ws.ownership_[j] = std::tanh(ws.ownership_[j]);
        });
    }
}

LossStats Network::loss(Workspace& ws, const float* pi, const float* z, const float* ownership,
                        float ownershipWeight, int B, bool withGradients) const {
    const int actions = cfg_.actions();
    // Per sample, on the pool: the softmax normalizer and, for training, the
    // gradients, which belong to their sample alone.
    parallelRanges(ws.pool_, static_cast<size_t>(B), kSampleChunk, [&](size_t b0, size_t b1) {
        for (size_t b = b0; b < b1; ++b) {
            const float* logits = ws.logits_.data() + b * actions;
            const float* target = pi + b * actions;
            // log softmax(l)_a = l_a - logSumExp(l). Subtracting the largest
            // logit before exp keeps every exponent <= 0, so nothing overflows.
            const float maxLogit = *std::max_element(logits, logits + actions);
            double sumExp = 0.0;
            for (int a = 0; a < actions; ++a) sumExp += std::exp(static_cast<double>(logits[a]) - maxLogit);
            const double logZ = maxLogit + std::log(sumExp);
            ws.logZ_[b] = logZ;

            if (withGradients) {
                // Cross-entropy through softmax: dL/dlogit = softmax - pi (pi sums to 1).
                float* d = ws.dLogits_.data() + b * actions;
                for (int a = 0; a < actions; ++a) {
                    d[a] = static_cast<float>((std::exp(logits[a] - logZ) - target[a]) / B);
                }
                // (z - v)^2 with v = tanh(u): dL/du = 2 (v - z) * (1 - v^2).
                const double v = ws.values_[b];
                ws.dValues_[b] = static_cast<float>(2.0 * (v - z[b]) * (1.0 - v * v) / B);
            }
        }
    });
    // The batch sums run serially, sample by sample: one fixed order of
    // additions, so one rounding, however many threads there are.
    double policySum = 0.0, valueSum = 0.0;
    for (int b = 0; b < B; ++b) {
        const float* logits = ws.logits_.data() + static_cast<size_t>(b) * actions;
        const float* target = pi + static_cast<size_t>(b) * actions;
        const double logZ = ws.logZ_[b];
        for (int a = 0; a < actions; ++a) policySum -= target[a] * (logits[a] - logZ);

        const double v = ws.values_[b];
        valueSum += (z[b] - v) * (z[b] - v);
    }
    LossStats stats;
    stats.policyLoss = static_cast<float>(policySum / B);
    stats.valueLoss = static_cast<float>(valueSum / B);

    // Ownership: w * (1/A) * sum_p (o_p - t_p)^2, averaged over the batch, so
    // the mean over all B * A points. With o = tanh(u):
    // dL/du = 2 w (o - t)(1 - o^2) / (B * A).
    if (cfg_.ownership && ownership) {
        const size_t points = static_cast<size_t>(B) * cfg_.area();
        double ownershipSum = 0.0;
        for (size_t j = 0; j < points; ++j) {
            const double o = ws.ownership_[j];
            const double d = o - ownership[j];
            ownershipSum += d * d;
            if (withGradients)
                ws.dOwnership_[j] = static_cast<float>(2.0 * ownershipWeight * d * (1.0 - o * o) / points);
        }
        stats.ownershipLoss = static_cast<float>(ownershipSum / points);
    }
    return stats;
}

void Network::backward(Workspace& ws, int B, bool withOwnership) {
    const Backend& be = *ws.backend_;
    ThreadPool* pool = ws.pool_;
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
    reluBackward(ws.valueHidden_.data(), ws.dValueHidden_.data(), static_cast<size_t>(B) * cfg_.valueHidden, pool);
    valueFc1_.backward(be, ws.valueAct_.data(), ws.dValueHidden_.data(), B, ws.dValueAct_.data());
    reluBackward(ws.valueAct_.data(), ws.dValueAct_.data(), M, pool);
    valueBn_.backward(ws.dValueAct_.data(), ws.valueXhat_.data(), ws.valueInvStd_.data(), M,
                      ws.dValueAct_.data(), pool);
    valueConv_.backward(be, trunkOut, col, ws.dValueAct_.data(), B, n, dcol, dTrunk, pool);

    // Policy head.
    policyFc_.backward(be, ws.policyFlat_.data(), ws.dLogits_.data(), B, ws.dPolicyFlat_.data());
    parallelRanges(pool, static_cast<size_t>(B), kSampleChunk, [&](size_t b0, size_t b1) {
        for (size_t b = b0; b < b1; ++b)          // [B][2A] -> [2][B][A], undoing the flatten
            for (int c = 0; c < 2; ++c)
                std::copy(ws.dPolicyFlat_.data() + b * 2 * A + static_cast<size_t>(c) * A,
                          ws.dPolicyFlat_.data() + b * 2 * A + static_cast<size_t>(c + 1) * A,
                          ws.dPolicyAct_.data() + (static_cast<size_t>(c) * B + b) * A);
    });
    reluBackward(ws.policyAct_.data(), ws.dPolicyAct_.data(), 2 * static_cast<size_t>(M), pool);
    policyBn_.backward(ws.dPolicyAct_.data(), ws.policyXhat_.data(), ws.policyInvStd_.data(), M,
                       ws.dPolicyAct_.data(), pool);
    policyConv_.backward(be, trunkOut, col, ws.dPolicyAct_.data(), B, n, dcol, dTmp, pool);
    addInto(dTrunk, dTmp, trunk, pool);           // the trunk feeds both heads

    // Ownership head: through tanh (already in dOwnership_), BN and the 1x1 conv.
    if (withOwnership) {
        ownershipBn_.backward(ws.dOwnership_.data(), ws.ownershipXhat_.data(),
                              ws.ownershipInvStd_.data(), M, ws.dOwnership_.data(), pool);
        ownershipConv_.backward(be, trunkOut, col, ws.dOwnership_.data(), B, n, dcol, dTmp, pool);
        addInto(dTrunk, dTmp, trunk, pool);
    }

    // Residual blocks in reverse; dTrunk holds dL/d(block output). The
    // shared col buffer was overwritten by later convs, so each 3x3 conv's
    // columns are rebuilt from its saved input before its backward.
    for (size_t i = blocks_.size(); i-- > 0;) {
        ResidualBlock& blk = blocks_[i];
        Workspace::BlockBuffers& buf = ws.blocks_[i];
        const float* in = i == 0 ? ws.stemOut_.data() : ws.blocks_[i - 1].out.data();
        reluBackward(buf.out.data(), dTrunk, trunk, pool);     // now dL/d(bn2 output + skip)
        blk.bn2.backward(dTrunk, buf.xhat2.data(), buf.invStd2.data(), M, dTmp, pool);
        im2colChannels(be, pool, buf.act1.data(), F, B, n, 3, col);
        blk.conv2.backward(be, buf.act1.data(), col, dTmp, B, n, dcol, dTmp2, pool);
        reluBackward(buf.act1.data(), dTmp2, trunk, pool);
        blk.bn1.backward(dTmp2, buf.xhat1.data(), buf.invStd1.data(), M, dTmp2, pool);
        im2colChannels(be, pool, in, F, B, n, 3, col);
        blk.conv1.backward(be, in, col, dTmp2, B, n, dcol, dTmp, pool);
        // The skip path passes dTrunk to the block input unchanged; the conv
        // path adds its own contribution.
        addInto(dTrunk, dTmp, trunk, pool);
    }

    // Stem. No gradient is needed for the input planes themselves.
    reluBackward(ws.stemOut_.data(), dTrunk, trunk, pool);
    stemBn_.backward(dTrunk, ws.stemXhat_.data(), ws.stemInvStd_.data(), M, dTrunk, pool);
    im2colChannels(be, pool, ws.input_.data(), cfg_.planes(), B, n, 3, col);
    stemConv_.backward(be, ws.input_.data(), col, dTrunk, B, n, dcol, nullptr, pool);
}

} // namespace azgo
