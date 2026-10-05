#include "azgo/replay_buffer.hpp"
#include <algorithm>
#include <cassert>
#include <climits>
#include <stdexcept>
#include <string>
#include <utility>
#include "azgo/encode.hpp"
#include "azgo/symmetry.hpp"

namespace azgo {

ReplayBuffer::ReplayBuffer(size_t capacityPositions) : capacity_(capacityPositions) {
    assert(capacity_ >= 1);
    rebuildIndex();
}

void ReplayBuffer::add(GameRecord game) {
    const size_t area = static_cast<size_t>(game.boardSize) * game.boardSize;
    assert(game.positions() > 0);
    assert(game.stones.size() == static_cast<size_t>(game.positions()) * area);
    assert(game.visits.size() == static_cast<size_t>(game.positions()));
    assert(game.finalOwnership.size() == area);
    assert(games_.empty() || games_.front().boardSize == game.boardSize);
    (void)area;
    positions_ += game.positions();
    totalAdded_ += game.positions();
    games_.push_back(std::move(game));
    evictOldest();
    rebuildIndex();
}

void ReplayBuffer::setCapacity(size_t capacityPositions) {
    assert(capacityPositions >= 1);
    capacity_ = capacityPositions;
    evictOldest();
    rebuildIndex();
}

void ReplayBuffer::evictOldest() {
    // Whole games leave the window, oldest first. The newest game always
    // stays, even if it alone is longer than the capacity.
    while (positions_ > capacity_ && games_.size() > 1) {
        positions_ -= games_.front().positions();
        games_.pop_front();
    }
}

void ReplayBuffer::rebuildIndex() {
    firstPosition_.assign(1, 0);
    for (const GameRecord& g : games_) firstPosition_.push_back(firstPosition_.back() + g.positions());
}

void ReplayBuffer::encodeSample(size_t gameIndex, int ply, int history, int sym,
                                float* input, float* pi, float* z, float* ownership) const {
    const GameRecord& g = games_[gameIndex];
    assert(ply >= 0 && ply < g.positions());
    const int n = g.boardSize;

    // History comes straight from the same game's earlier positions.
    std::vector<const uint8_t*> stones;
    for (int k = 0; k < history && ply - k >= 0; ++k) stones.push_back(g.stonesAt(ply - k));
    encodePlanes(stones, n, g.toMoveAt(ply), history, sym, input);

    // pi = visit counts normalised, moved to where the symmetry puts each point.
    std::fill(pi, pi + n * n + 1, 0.0f);
    int total = 0;
    for (const auto& av : g.visits[ply]) total += av.second;
    assert(total > 0);
    for (const auto& av : g.visits[ply])
        pi[transformAction(sym, av.first, n)] = static_cast<float>(av.second) / static_cast<float>(total);

    *z = g.zAt(ply);

    // Ownership, like z, from the side to move's view, and moved like the
    // input planes, so every point's target stays on its point.
    if (ownership) {
        assert(g.finalOwnership.size() == static_cast<size_t>(n) * n);
        const float sign = g.toMoveAt(ply) == Color::Black ? 1.0f : -1.0f;
        for (int p = 0; p < n * n; ++p)
            ownership[transformPoint(sym, p, n)] = sign * g.finalOwnership[static_cast<size_t>(p)];
    }
}

void ReplayBuffer::sample(int B, int history, Rng& rng, TrainingBatch& out) const {
    assert(positions_ > 0 && B >= 1);
    assert(positions_ <= static_cast<size_t>(INT_MAX));
    const int n = games_.front().boardSize;
    const size_t inputStride = static_cast<size_t>(numPlanes(history)) * n * n;
    const size_t piStride = static_cast<size_t>(n) * n + 1;
    const size_t area = static_cast<size_t>(n) * n;
    out.size = B;
    out.input.resize(B * inputStride);
    out.pi.resize(B * piStride);
    out.z.resize(B);
    out.ownership.resize(B * area);
    for (int b = 0; b < B; ++b) {
        // Uniform over positions, not games: long games are sampled more.
        const size_t global = static_cast<size_t>(rng.below(static_cast<int>(positions_)));
        const int sym = rng.below(kNumSymmetries);
        // Last game whose first position is <= global.
        const size_t g = static_cast<size_t>(
            std::upper_bound(firstPosition_.begin(), firstPosition_.end(), global) - firstPosition_.begin() - 1);
        const int ply = static_cast<int>(global - firstPosition_[g]);
        encodeSample(g, ply, history, sym, out.input.data() + b * inputStride,
                     out.pi.data() + b * piStride, out.z.data() + b, out.ownership.data() + b * area);
    }
}

// Layout: u64 capacity, u64 totalAdded (two's complement of the i64), u32 games,
// then per game: i32 boardSize, f32 komi, u32 moves + i32 each, the stones as
// raw bytes (positions * n * n), per ply u32 count + u32 (action << 16 | count)
// each, f32 resultForBlack, then (withOwnership) finalOwnership as n * n
// signed bytes, then (withResigned) one byte, 1 if the game ended by
// resignation.
void ReplayBuffer::save(BinaryWriter& w, bool withOwnership, bool withResigned) const {
    w.u64(capacity_);
    w.u64(static_cast<uint64_t>(totalAdded_));
    w.u32(static_cast<uint32_t>(games_.size()));
    for (const GameRecord& g : games_) {
        w.i32(g.boardSize);
        w.f32(g.komi);
        w.u32(static_cast<uint32_t>(g.moves.size()));
        for (int m : g.moves) w.i32(m);
        w.bytes(g.stones.data(), g.stones.size());
        for (const auto& ply : g.visits) {
            w.u32(static_cast<uint32_t>(ply.size()));
            for (const auto& av : ply)
                w.u32((static_cast<uint32_t>(static_cast<uint16_t>(av.first)) << 16) | av.second);
        }
        w.f32(g.resultForBlack);
        if (withOwnership) {
            assert(g.finalOwnership.size() == static_cast<size_t>(g.boardSize) * g.boardSize);
            w.bytes(g.finalOwnership.data(), g.finalOwnership.size());
        }
        if (withResigned) {
            const uint8_t resigned = g.resigned ? 1 : 0;
            w.bytes(&resigned, 1);
        }
    }
}

ReplayBuffer ReplayBuffer::load(BinaryReader& r, bool withOwnership, bool withResigned) {
    const uint64_t capacity = r.u64();
    if (capacity == 0) throw std::runtime_error("replay buffer: capacity is 0");
    ReplayBuffer buf(static_cast<size_t>(capacity));
    const long long totalAdded = static_cast<long long>(r.u64());
    const uint32_t games = r.u32();
    for (uint32_t i = 0; i < games; ++i) {
        GameRecord g;
        g.boardSize = r.i32();
        if (g.boardSize < kMinBoardSize || g.boardSize > kMaxBoardSize)
            throw std::runtime_error("replay buffer: bad board size " + std::to_string(g.boardSize));
        const int area = g.boardSize * g.boardSize;
        g.komi = r.f32();
        // Sanity bounds keep a corrupt file from asking for gigabytes.
        const uint32_t moves = r.u32();
        if (moves == 0 || moves > 100000) throw std::runtime_error("replay buffer: bad move count");
        g.moves.resize(moves);
        for (int& m : g.moves) {
            m = r.i32();
            if (m < 0 || m > area) throw std::runtime_error("replay buffer: bad move");
        }
        g.stones.resize(static_cast<size_t>(moves) * area);
        r.bytes(g.stones.data(), g.stones.size());
        g.visits.resize(moves);
        for (auto& ply : g.visits) {
            const uint32_t k = r.u32();
            if (k == 0 || k > static_cast<uint32_t>(area + 1))
                throw std::runtime_error("replay buffer: bad visit list");
            ply.resize(k);
            for (auto& av : ply) {
                const uint32_t packed = r.u32();
                av.first = static_cast<int16_t>(packed >> 16);
                av.second = static_cast<uint16_t>(packed & 0xFFFFu);
                if (av.first < 0 || av.first > area || av.second == 0)
                    throw std::runtime_error("replay buffer: bad visit entry");
            }
        }
        g.resultForBlack = r.f32();
        if (withOwnership) {
            g.finalOwnership.resize(static_cast<size_t>(area));
            r.bytes(g.finalOwnership.data(), g.finalOwnership.size());
            for (int8_t o : g.finalOwnership)
                if (o < -1 || o > 1) throw std::runtime_error("replay buffer: bad ownership value");
        } else {
            // An older file: recompute what self-play would have recorded.
            try {
                g.finalOwnership = replayFinalOwnership(g);
            } catch (const std::runtime_error& e) {
                throw std::runtime_error("replay buffer: game " + std::to_string(i) +
                                         " does not replay: " + e.what());
            }
        }
        if (withResigned) {
            uint8_t resigned = 0;
            r.bytes(&resigned, 1);
            if (resigned > 1) throw std::runtime_error("replay buffer: bad resigned flag");
            g.resigned = resigned == 1;
        }
        // Straight into the deque: the saved window already respected the
        // capacity, and totalAdded is restored below rather than recounted.
        buf.positions_ += g.positions();
        buf.games_.push_back(std::move(g));
    }
    buf.totalAdded_ = totalAdded;
    buf.rebuildIndex();
    return buf;
}

} // namespace azgo
