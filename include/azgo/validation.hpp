// The validation loss: the training loss measured where the network has not
// been fitted. The training loss is taken on batches from the replay
// buffer, the very data SGD is fitting, so a falling training loss cannot
// tell learning (judging new positions better) from memorizing the buffer.
// Each generation's self-play games are positions the network has never
// trained on; scoring the network on them, before they join the buffer,
// answers that. The same measurement on a sample of the buffer gives the
// fitted side of the comparison with everything else equal: the same
// network, inference-mode batch norm, no symmetry augmentation.
//
// A measurement only: it reads the network and the games, draws from no
// generator the run uses, and changes nothing the run produces.
#pragma once
#include <cstddef>
#include <vector>
#include "azgo/network.hpp"
#include "azgo/replay_buffer.hpp"
#include "azgo/rng.hpp"
#include "azgo/selfplay.hpp"

namespace azgo {

// One position of a game record (ply as in GameRecord). The record must
// outlive the reference.
struct PositionRef {
    const GameRecord* game = nullptr;
    int ply = 0;
};

// Means over a set of positions of the training loss terms (LossStats):
// policy cross-entropy against the visit distribution pi, squared error of
// the value against z, and the unweighted per-point squared error of the
// ownership head against the final owners. -1 means not measured: every
// loss when there were no positions, ownershipLoss for a network without the head.
struct HeldOutLoss {
    size_t positions = 0;
    float policyLoss = -1, valueLoss = -1;
    float ownershipLoss = -1;
};

// Every position of `games`, game by game, plies in order.
std::vector<PositionRef> allPositions(const std::vector<GameRecord>& games);

// `count` positions drawn uniformly over the buffer's positions, with
// replacement, as ReplayBuffer::sample draws them (one rng.below per
// position, no symmetry drawn). Asserts the buffer is not empty when count > 0.
std::vector<PositionRef> samplePositions(const ReplayBuffer& buffer, int count, Rng& rng);

// The loss terms of Network::trainStep over `positions`, each encoded at
// the identity symmetry (encodeTrainingSample) and evaluated in inference
// mode (Network::inferLoss) in batches of ws.maxBatch(), in order. The
// ownership term is measured when the network has the head. Only ws is
// written; the network is read like in search.
HeldOutLoss heldOutLoss(const Network& net, Workspace& ws, const std::vector<PositionRef>& positions,
                        int history);

} // namespace azgo
