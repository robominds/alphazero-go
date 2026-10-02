// Growing a network: add residual blocks, or filters, without changing what
// it computes.
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
//
// Widening: more filters per layer without changing what the network computes
// (Net2WiderNet, Chen, Goodfellow and Shlens 2015, here with batch norm).
// A new channel j starts as a copy of an old channel g(j): the layer that
// writes it gets a copy of g(j)'s conv weights and of its BN's gamma, beta
// and running statistics, so j carries exactly g(j)'s values, in inference
// (running statistics) and in training (identical channels have identical
// batch statistics). A layer that reads the old channel c now sees it k
// times, so each of its weights on c is split among the k copies in shares
// that sum to the old weight: the sum over the copies is the old term.
//
// The residual stream is written by the stem and by every block's second
// conv + BN, and the skip connections add those writes together, so all
// stream writers share one mapping g; then a channel's copies stay equal
// after every addition. Each block's inner channels (between its two convs)
// get their own mapping. The heads' 1x1 convs read the stream and are split;
// their outputs, the head BNs and the linear layers are unchanged.
//
// Why random shares instead of halving each weight and adding noise: two
// copies with equal shares get equal gradients forever and stay redundant,
// and noise breaks the exactness. Any shares that sum to the old weight
// keep the function exact, so unequal random ones break the symmetry for
// free: the readers send the copies different gradients, the writers'
// weights move apart, and from then on the copies compute different
// features. Shares are drawn per (output channel, old input channel, kernel
// tap) from [0.5, 1.5) and normalized; the last copy takes the remainder, so
// the float shares add up to the old weight as closely as rounding allows.
//
// Width is not free: a 3x3 conv costs 9 * F^2 * n^2 multiply-adds per
// position, so 96 filters cost about (96/64)^2 = 2.25x as much as 64.
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

// A network with the same board size, history, blocks and value head width
// as `net` and `newFilters` filters, computing the same policy and value.
// Channels below net's filter count keep their indices; the rest are copies
// of old channels chosen, with the weight shares, from Rng(seed).
// newFilters == net's filters returns an exact copy. Throws
// std::invalid_argument if newFilters is below net's filters or above
// kMaxFilters.
Network widenFilters(const Network& net, int newFilters, uint64_t seed);

} // namespace azgo
