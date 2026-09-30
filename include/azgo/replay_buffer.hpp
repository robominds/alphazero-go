// The replay buffer is the training set: the most recent self-play games,
// kept whole in a sliding window of about `capacity` positions. A training
// sample is one (game, ply); its history planes are rebuilt from the same
// game's earlier positions, so storing history costs nothing extra. Each
// sample is shown to the network under a random one of the board's 8
// symmetries, which multiplies the data for free (Go has no preferred side).
#pragma once
#include <cstddef>
#include <deque>
#include <vector>
#include "azgo/rng.hpp"
#include "azgo/selfplay.hpp"
#include "azgo/serialize.hpp"

namespace azgo {

struct TrainingBatch {
    int size = 0;
    std::vector<float> input;   // [size][planes][area]
    std::vector<float> pi;      // [size][area + 1]
    std::vector<float> z;       // [size]
};

// Sliding window of whole games. A sample is (game, ply); its history planes
// come from the same game's earlier positions, so history costs no storage.
class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t capacityPositions);
    void add(GameRecord game);            // then evicts oldest games while positions() > capacity,
                                          // never evicting the newest game
    void setCapacity(size_t capacityPositions);   // evicts as add() does
    size_t capacity() const { return capacity_; }
    size_t positions() const { return positions_; }
    size_t games() const { return games_.size(); }
    long long totalPositionsAdded() const { return totalAdded_; }
    const GameRecord& game(size_t i) const { return games_[i]; }   // 0 = oldest

    // B samples uniformly over positions (with replacement), each with a
    // uniformly random symmetry. Asserts positions() > 0.
    void sample(int B, int history, Rng& rng, TrainingBatch& out) const;
    // One sample into caller buffers: input numPlanes(H)*area, pi area+1, *z.
    void encodeSample(size_t gameIndex, int ply, int history, int sym,
                      float* input, float* pi, float* z) const;

    void save(BinaryWriter& w) const;
    static ReplayBuffer load(BinaryReader& r);

private:
    size_t capacity_;
    size_t positions_ = 0;
    long long totalAdded_ = 0;
    std::deque<GameRecord> games_;
    // Sampling index helpers: implementation-defined (Task 15).
    std::vector<size_t> firstPosition_;   // firstPosition_[g] = positions in games 0..g-1; back() = positions_
    void evictOldest();                   // drop oldest games while over capacity (keeps the newest)
    void rebuildIndex();                  // recompute firstPosition_ after add/evict/load
};

} // namespace azgo
