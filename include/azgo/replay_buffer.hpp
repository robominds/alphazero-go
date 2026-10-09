// The replay buffer is the training set: the most recent self-play games,
// kept whole in a sliding window of about `capacity` positions. A training
// sample is one (game, ply); its history planes are rebuilt from the same
// game's earlier positions, so storing history costs nothing extra. Each
// sample is shown to the network under a random one of the board's 8
// symmetries, which multiplies the data for free (Go has no preferred side).
// The ownership target travels with the position: the final board's owners,
// turned to the side to move's view and moved by the same symmetry.
#pragma once
#include <cstddef>
#include <deque>
#include <utility>
#include <vector>
#include "azgo/rng.hpp"
#include "azgo/selfplay.hpp"
#include "azgo/serialize.hpp"

namespace azgo {

class ThreadPool;   // azgo/parallel.hpp

struct TrainingBatch {
    int size = 0;
    std::vector<float> input;   // [size][planes][area]
    std::vector<float> pi;      // [size][area + 1]
    std::vector<float> z;       // [size]
    std::vector<float> ownership;   // [size][area]: final owner of each point, side to move's view
};

// One training sample of `game`, buffered or not, into caller buffers:
// input numPlanes(history)*area (history from the same game's earlier
// positions), pi area+1 (the visit counts normalised), *z (the result from
// the side to move's view) and, unless null, ownership area
// (finalOwnership, negated when White is to move), all moved by symmetry
// `sym`.
void encodeTrainingSample(const GameRecord& game, int ply, int history, int sym,
                          float* input, float* pi, float* z, float* ownership = nullptr);

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
    // uniformly random symmetry. Asserts positions() > 0. Every draw is
    // made first, in sample order; then the samples are encoded, on the
    // pool if one is given (each writes only its own rows).
    void sample(int B, int history, Rng& rng, TrainingBatch& out, ThreadPool* pool = nullptr) const;
    // encodeTrainingSample of game(gameIndex).
    void encodeSample(size_t gameIndex, int ply, int history, int sym,
                      float* input, float* pi, float* z, float* ownership = nullptr) const;
    // (game index, ply) of position number `position`, counting from the
    // oldest game's first position: what sample() draws uniformly.
    // Asserts position < positions().
    std::pair<size_t, int> locate(size_t position) const;

    // withOwnership = false writes the layout of state.bin versions 1 and 2,
    // which stored no ownership; withResigned = false that of versions 1 to
    // 4, which stored no resigned flag. Tests use them to make old files.
    void save(BinaryWriter& w, bool withOwnership = true, bool withResigned = true) const;
    // withOwnership = false reads that older layout and recomputes each
    // game's finalOwnership with replayFinalOwnership (a std::runtime_error
    // naming the game if one does not replay); withResigned = false loads
    // every game as not resigned (resignation did not exist yet).
    static ReplayBuffer load(BinaryReader& r, bool withOwnership = true, bool withResigned = true);

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
