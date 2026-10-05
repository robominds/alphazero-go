#include "azgo/validation.hpp"
#include <algorithm>
#include <cassert>
#include <climits>
#include "azgo/encode.hpp"

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
    HeldOutLoss out;
    if (positions.empty()) return out;
    const NetConfig& cfg = net.config();
    assert(cfg.history == history);
    const bool ownership = cfg.ownership;
    const size_t area = static_cast<size_t>(cfg.area());
    const size_t inputStride = static_cast<size_t>(numPlanes(history)) * area;
    const size_t piStride = area + 1;
    const int maxBatch = ws.maxBatch();
    std::vector<float> input(static_cast<size_t>(maxBatch) * inputStride);
    std::vector<float> pi(static_cast<size_t>(maxBatch) * piStride);
    std::vector<float> z(static_cast<size_t>(maxBatch));
    std::vector<float> owners(static_cast<size_t>(maxBatch) * area);

    // inferLoss returns batch means; weighting each by its batch size gives
    // the mean over every position whatever the batching.
    double policySum = 0, valueSum = 0, ownershipSum = 0;
    for (size_t start = 0; start < positions.size(); start += static_cast<size_t>(maxBatch)) {
        const int B = static_cast<int>(std::min(positions.size() - start, static_cast<size_t>(maxBatch)));
        for (int b = 0; b < B; ++b) {
            const PositionRef& p = positions[start + static_cast<size_t>(b)];
            assert(p.game && p.game->boardSize == cfg.boardSize);
            encodeTrainingSample(*p.game, p.ply, history, 0, input.data() + b * inputStride,
                                 pi.data() + b * piStride, z.data() + b,
                                 ownership ? owners.data() + b * area : nullptr);
        }
        const LossStats loss =
            net.inferLoss(ws, input.data(), pi.data(), z.data(), B, ownership ? owners.data() : nullptr);
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

} // namespace azgo
