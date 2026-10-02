// Growing a network: add residual blocks without changing what it computes.
//
// A residual block returns relu(x + F(x)), where x is the block's input and
// F is conv -> BN -> ReLU -> conv -> BN. The input x is itself the output of
// a ReLU (the stem's, or the previous block's), so x >= 0 and relu(x) = x.
// If F(x) is exactly zero, the block is therefore an exact identity.
//
// F ends in batch norm, y = gamma * xhat + beta, so setting gamma = 0 and
// beta = 0 in the block's second BN forces F(x) = 0 whatever the convolutions
// hold. That is "zero-init residual": new blocks start as identities, so the
// grown network has exactly the old network's policy and value, and a trained
// 6-block net becomes a 10-block net that plays the same moves.
//
// The new blocks are not stuck at zero. The gradient of the loss with respect
// to gamma is sum(dy * xhat), which does not depend on gamma, so bn2.gamma and
// bn2.beta receive gradient on the first step. Once gamma moves off zero the
// gradient also flows into the block's convolutions, and the block learns.
//
// Existing blocks keep their indices ("block0" ... "block5"); new blocks are
// appended after them, so the trunk's last layers are the new ones.
#pragma once
#include <cstdint>
#include "azgo/network.hpp"

namespace azgo {

// A network with the same board size, history, filters and value head width
// as `net` and `newBlocks` residual blocks. Every parameter and batch norm
// running statistic of `net` is copied by name; the extra blocks are
// He-initialized from Rng(seed) except for a zeroed bn2, which makes them
// identities. newBlocks == net's blocks returns an exact copy. Throws
// std::invalid_argument if newBlocks is smaller than net's block count.
Network growBlocks(const Network& net, int newBlocks, uint64_t seed);

} // namespace azgo
