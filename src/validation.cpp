#include "azgo/validation.hpp"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include "azgo/encode.hpp"
#include "azgo/parallel.hpp"

namespace azgo {

std::vector<PositionRef> allPositions(const std::vector<GameRecord>& games) {
    std::vector<PositionRef> out;
    for (const GameRecord& g : games)
        for (int ply = 0; ply < g.positions(); ++ply) out.push_back({&g, ply});
    return out;
}

std::vector<PositionRef> samplePositions(const ReplayBuffer& buffer, int count, Rng& rng) {
    std::vector<PositionRef> out;
    if (count <= 0) return out;
    assert(buffer.positions() > 0 && buffer.positions() <= static_cast<size_t>(INT_MAX));
    out.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        const auto [g, ply] = buffer.locate(static_cast<size_t>(rng.below(static_cast<int>(buffer.positions()))));
        out.push_back({&buffer.game(g), ply});
    }
    return out;
}

HeldOutLoss heldOutLoss(const Network& net, Workspace& ws, const std::vector<PositionRef>& positions,
                        int history) {
    const BatchLossFn lossOf = [&](const float* input, const float* pi, const float* z, int B,
                                   const float* ownership) {
        return net.inferLoss(ws, input, pi, z, B, ownership);
    };
    return heldOutLoss(net.config(), ws.maxBatch(), ws.pool(), lossOf, positions, history);
}

HeldOutLoss heldOutLoss(const NetConfig& cfg, int maxBatch, ThreadPool* pool, const BatchLossFn& lossOf,
                        const std::vector<PositionRef>& positions, int history) {
    HeldOutLoss out;
    if (positions.empty()) return out;
    assert(cfg.history == history);
    const bool ownership = cfg.ownership;
    const size_t area = static_cast<size_t>(cfg.area());
    const size_t inputStride = static_cast<size_t>(numPlanes(history)) * area;
    const size_t piStride = area + 1;
    std::vector<float> input(static_cast<size_t>(maxBatch) * inputStride);
    std::vector<float> pi(static_cast<size_t>(maxBatch) * piStride);
    std::vector<float> z(static_cast<size_t>(maxBatch));
    std::vector<float> owners(static_cast<size_t>(maxBatch) * area);

    // inferLoss returns batch means; weighting each by its batch size gives
    // the mean over every position whatever the batching.
    double policySum = 0, valueSum = 0, ownershipSum = 0;
    for (size_t start = 0; start < positions.size(); start += static_cast<size_t>(maxBatch)) {
        const int B = static_cast<int>(std::min(positions.size() - start, static_cast<size_t>(maxBatch)));
        // Encoding writes each position's own rows, so it can use the pool;
        // a CPU loss function does too.
        parallelRanges(pool, static_cast<size_t>(B), 16, [&](size_t b0, size_t b1) {
            for (size_t b = b0; b < b1; ++b) {
                const PositionRef& p = positions[start + b];
                assert(p.game && p.game->boardSize == cfg.boardSize);
                encodeTrainingSample(*p.game, p.ply, history, 0, input.data() + b * inputStride,
                                     pi.data() + b * piStride, z.data() + b,
                                     ownership ? owners.data() + b * area : nullptr);
            }
        });
        const LossStats loss =
            lossOf(input.data(), pi.data(), z.data(), B, ownership ? owners.data() : nullptr);
        policySum += static_cast<double>(loss.policyLoss) * B;
        valueSum += static_cast<double>(loss.valueLoss) * B;
        ownershipSum += static_cast<double>(loss.ownershipLoss) * B;
    }
    const double n = static_cast<double>(positions.size());
    out.positions = positions.size();
    out.policyLoss = static_cast<float>(policySum / n);
    out.valueLoss = static_cast<float>(valueSum / n);
    if (ownership) out.ownershipLoss = static_cast<float>(ownershipSum / n);
    return out;
}

LossStats lossFromOutputs(const NetConfig& cfg, const float* logits, const float* values,
                          const float* ownership, const float* pi, const float* z, const float* targets,
                          int B) {
    // Network::loss without gradients, line for line (src/network.cpp).
    const int actions = cfg.actions();
    double policySum = 0.0, valueSum = 0.0;
    for (int b = 0; b < B; ++b) {
        const float* l = logits + static_cast<size_t>(b) * actions;
        const float* target = pi + static_cast<size_t>(b) * actions;
        // log softmax(l)_a = l_a - logSumExp(l), the largest logit subtracted first.
        const float maxLogit = *std::max_element(l, l + actions);
        double sumExp = 0.0;
        for (int a = 0; a < actions; ++a) sumExp += std::exp(static_cast<double>(l[a]) - maxLogit);
        const double logZ = maxLogit + std::log(sumExp);
        for (int a = 0; a < actions; ++a) policySum -= target[a] * (l[a] - logZ);
        const double v = values[b];
        valueSum += (z[b] - v) * (z[b] - v);
    }
    LossStats stats;
    stats.policyLoss = static_cast<float>(policySum / B);
    stats.valueLoss = static_cast<float>(valueSum / B);
    if (cfg.ownership && ownership && targets) {
        const size_t points = static_cast<size_t>(B) * cfg.area();
        double ownershipSum = 0.0;
        for (size_t j = 0; j < points; ++j) {
            const double d = static_cast<double>(ownership[j]) - targets[j];
            ownershipSum += d * d;
        }
        stats.ownershipLoss = static_cast<float>(ownershipSum / points);
    }
    return stats;
}

} // namespace azgo
