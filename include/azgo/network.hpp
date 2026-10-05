// The two-headed residual network of AlphaGo Zero. A shared convolutional
// trunk (a stem plus residual blocks, each adding its output to its input so
// gradients have a short path back through deep stacks) reads the input
// planes; a policy head turns the trunk's features into one logit per action
// and a value head into one number in [-1, 1]: the expected result for the
// side to move. Parameters (Network) are shared read-only by all threads;
// activations (Workspace) are per thread, so many searches can run at once.
//
// Optionally (NetConfig::ownership) a third head predicts, for every point,
// who will own it when the game ends: a 1x1 conv to one channel, batch norm
// and a tanh per point, in [-1, 1] from the side to move's view. It is
// KataGo's auxiliary target: n*n training signals per position instead of
// one game result, and every point of a dead group is labelled with the
// opponent's color. Search never reads it; it only shapes the shared trunk.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/layers.hpp"

namespace azgo {

struct NetConfig {
    int boardSize = 9;
    int history = 8;       // H
    int blocks = 6;        // residual blocks, >= 0
    int filters = 64;
    int valueHidden = 64;
    bool ownership = false;   // an ownership head: who owns each point at the end
    int planes() const { return 2 * history + 2; }
    int area() const { return boardSize * boardSize; }
    int actions() const { return area() + 1; }
    bool operator==(const NetConfig& o) const;
    bool operator!=(const NetConfig& o) const { return !(*this == o); }
};

struct ParamView  { std::string name; float* value; float* grad; size_t size; bool decay; };
struct BufferView { std::string name; float* value; size_t size; };

// Batch means. ownershipLoss is the unweighted mean squared error per point,
// 0 when no ownership target was given.
struct LossStats { float policyLoss = 0; float valueLoss = 0; float ownershipLoss = 0; };

// Per-thread activation and scratch storage for up to maxBatch samples.
class Workspace {
public:
    Workspace(const NetConfig& cfg, int maxBatch, const Backend& backend);
    int maxBatch() const { return maxBatch_; }
    const Backend& backend() const { return *backend_; }
private:
    friend class Network;
    NetConfig cfg_;
    int maxBatch_;
    const Backend* backend_;
    // Activation buffers (Task 7). All are channel-major [C][B][area] unless
    // noted, sized for maxBatch, allocated once here and never resized.
    struct BlockBuffers {
        std::vector<float> xhat1, invStd1, act1;   // bn1 cache; act1 = relu(bn1) = conv2 input
        std::vector<float> xhat2, invStd2, out;    // bn2 cache; out = relu(bn2 + skip)
    };
    std::vector<float> input_;                    // [planes][B][area] (transposed network input)
    std::vector<float> col_, dcol_;               // im2col scratch shared by every conv
    std::vector<float> convOut_;                  // conv output before its BN
    std::vector<float> stemXhat_, stemInvStd_, stemOut_;
    std::vector<BlockBuffers> blocks_;
    std::vector<float> policyXhat_, policyInvStd_, policyAct_;   // policyAct_ [2][B][area]
    std::vector<float> policyFlat_;               // [B][2 * area]
    std::vector<float> logits_;                   // [B][actions]
    std::vector<float> valueXhat_, valueInvStd_, valueAct_;      // valueAct_ [1][B][area] = [B][area]
    std::vector<float> valueHidden_;              // [B][valueHidden]
    std::vector<float> values_;                   // [B] (after tanh)
    // Ownership head, allocated only when cfg.ownership. ownership_ is
    // [1][B][area] = [B][area], after tanh.
    std::vector<float> ownershipXhat_, ownershipInvStd_, ownership_;
    // Backward scratch.
    std::vector<float> dTrunk_, dTmp_, dTmp2_;    // [filters][B][area]
    std::vector<float> dLogits_, dPolicyFlat_, dPolicyAct_;
    std::vector<float> dValues_, dValueHidden_, dValueAct_;
    std::vector<float> dOwnership_;               // [B][area]
};

class Network {
public:
    Network(const NetConfig& cfg, uint64_t seed);   // He-initialized from Rng(seed)
    const NetConfig& config() const { return cfg_; }

    // Fixed order, stable names ("stem.conv.w", "block2.bn1.gamma", "policy.fc.b", ...),
    // the ownership head last ("ownership.conv.w", "ownership.bn.gamma", ".beta").
    // decay is true for conv and linear weights only (not biases, not BN).
    std::vector<ParamView> params();
    std::vector<BufferView> buffers();              // BN runMean/runVar, fixed order
    void zeroGrad();

    // Inference (BN running statistics). input [B][planes][area],
    // logits [B][actions], values [B] (tanh). ownership, if not null,
    // receives [B][area] in [-1, 1] from the side to move's view; it needs a
    // network with the head (asserted). Asserts B <= ws.maxBatch().
    void infer(Workspace& ws, const float* input, int B, float* logits, float* values,
               float* ownership = nullptr) const;

    // Training step: zeroGrad, forward with batch statistics (updates BN
    // running stats), loss = mean((z - v)^2) + mean(-sum pi log softmax(logits))
    // + w * mean((1/area) sum_p (o_p - t_p)^2), backward into grads.
    // pi [B][actions], z [B]. ownership: targets t [B][area] in [-1, 1] from
    // the side to move's view, or null for no ownership loss; it needs a
    // network with the head (asserted). w = ownershipWeight; with w = 0 the
    // ownership loss is reported but the head receives no gradient.
    LossStats trainStep(Workspace& ws, const float* input, const float* pi, const float* z, int B,
                        const float* ownership = nullptr, float ownershipWeight = 0.0f);

    // Same loss as trainStep (batch statistics) without backward and without
    // touching running stats. Used by gradient checks.
    LossStats computeLoss(Workspace& ws, const float* input, const float* pi, const float* z, int B,
                          const float* ownership = nullptr, float ownershipWeight = 0.0f);

    // The same loss terms with inference-mode batch norm (running
    // statistics, as infer and search use), without backward: how the
    // network does on data it is not training on. Writes only to ws; the
    // network is shared read-only like for infer. ownershipLoss is 0 without
    // a target (which needs the head, asserted).
    LossStats inferLoss(Workspace& ws, const float* input, const float* pi, const float* z, int B,
                        const float* ownership = nullptr) const;

private:
    NetConfig cfg_;
    // Layers (Task 7).
    struct ResidualBlock {
        Conv2d conv1;
        BatchNorm2d bn1;
        Conv2d conv2;
        BatchNorm2d bn2;
    };
    Conv2d stemConv_;
    BatchNorm2d stemBn_;
    std::vector<ResidualBlock> blocks_;
    Conv2d policyConv_;
    BatchNorm2d policyBn_;
    Linear policyFc_;
    Conv2d valueConv_;
    BatchNorm2d valueBn_;
    Linear valueFc1_, valueFc2_;
    Conv2d ownershipConv_;            // in params() only when cfg_.ownership
    BatchNorm2d ownershipBn_;

    // Infer: BN running statistics. Train: batch statistics, running stats
    // updated. Probe: batch statistics, running stats untouched (computeLoss).
    enum class Mode { Infer, Train, Probe };
    void forward(Workspace& ws, const float* input, int B, Mode mode);
    LossStats loss(Workspace& ws, const float* pi, const float* z, const float* ownership,
                   float ownershipWeight, int B, bool withGradients) const;
    // withOwnership: backpropagate the ownership head too (a target and w != 0).
    void backward(Workspace& ws, int B, bool withOwnership);
};

} // namespace azgo
