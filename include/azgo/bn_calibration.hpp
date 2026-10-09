// Batch norm calibration of a fresh network.
//
// Batch norm normalizes with batch statistics in training but with running
// statistics in inference (search, the arena, the validation loss), and a
// freshly initialized network's running statistics are mean 0, variance 1:
// in inference mode it does not normalize at all. Through a deep trunk the
// activations then grow block after block, so generation 0 of a 10x128
// network saturates its value head (+1.00 at the empty board) and its
// policy, its self-play games end after a few moves, and training never
// recovers. Calibration sets every layer's running statistics from one
// training-mode pass over a fixed batch of positions, so generation 0 in
// inference mode is the normalized random network that training starts from.
#pragma once
#include <cstdint>
#include <vector>
#include "azgo/network.hpp"

namespace azgo {

// Positions in one calibration batch: as many as a training batch of the
// presets (batch_size = 256), so each layer's statistics are those of a
// training batch's size (256 * 81 = 20736 values per channel on 9x9; the
// standard error of a channel's mean is under 1% of its deviation), for
// the cost of one forward pass.
constexpr int kBnCalibrationPositions = 256;

// `count` positions encoded as the network sees them ([count][planes][area],
// history planes included, identity symmetry), each from its own random
// game: uniformly random legal moves (a pass only when no point is legal),
// position i after i * n * n / count moves, so lengths are spread evenly
// from the empty board to n * n moves (the middle of a 9x9 self-play game),
// with Black and White to move alike. Deterministic in seed.
std::vector<float> bnCalibrationPositions(int boardSize, int history, uint64_t seed,
                                          int count = kBnCalibrationPositions);

// Network::calibrateBatchNorm on bnCalibrationPositions(seed), in one batch,
// on the naive backend with a pool of `threads` threads: the result is the
// same bits for any thread count and whatever backend the run uses.
void calibrateBatchNorm(Network& net, uint64_t seed, int threads);

} // namespace azgo
