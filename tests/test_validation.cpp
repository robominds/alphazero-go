#include <cassert>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/encode.hpp"
#include "azgo/game.hpp"
#include "azgo/network.hpp"
#include "azgo/replay_buffer.hpp"
#include "azgo/rng.hpp"
#include "azgo/validation.hpp"

using namespace azgo;

static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

// A 5x5 game as self-play records it (see test_replay_buffer): the board
// before each move, visits {move: 3, pass: 1} ({pass: 4} for a pass), the
// result if the game ended (else `result`), and the final board's owners.
static GameRecord makeRecord(const std::vector<int>& moves, float result = 1.0f) {
    const int n = 5, area = 25;
    Game game(n, 7.5f, defaultMaxMoves(n));
    GameRecord rec;
    rec.boardSize = n;
    rec.komi = 7.5f;
    for (int m : moves) {
        const size_t off = rec.stones.size();
        rec.stones.resize(off + area);
        game.board().exportStones(rec.stones.data() + off);
        if (m == area) rec.visits.push_back({{static_cast<int16_t>(area), 4}});
        else rec.visits.push_back({{static_cast<int16_t>(m), 3}, {static_cast<int16_t>(area), 1}});
        rec.moves.push_back(m);
        game.play(m);
    }
    rec.resultForBlack = game.isTerminal() ? game.resultForBlack() : result;
    rec.finalOwnership.resize(area);
    game.board().ownership(rec.finalOwnership.data());
    return rec;
}

static std::vector<GameRecord> someGames() {
    return {makeRecord({6, 18, 12, 25, 25}), makeRecord({12, 7, 13}, -1.0f), makeRecord({0, 24, 1, 23}, 1.0f)};
}

static NetConfig tinyNet(bool ownership) {
    NetConfig cfg;
    cfg.boardSize = 5;
    cfg.history = 2;
    cfg.blocks = 1;
    cfg.filters = 4;
    cfg.valueHidden = 8;
    cfg.ownership = ownership;
    return cfg;
}

// Every value the network holds: parameters, gradients, running statistics.
static std::vector<float> snapshot(Network& net) {
    std::vector<float> all;
    for (const ParamView& p : net.params()) {
        all.insert(all.end(), p.value, p.value + p.size);
        all.insert(all.end(), p.grad, p.grad + p.size);
    }
    for (const BufferView& b : net.buffers()) all.insert(all.end(), b.value, b.value + b.size);
    return all;
}

static void test_all_positions_in_game_then_ply_order() {
    const std::vector<GameRecord> games = someGames();
    const std::vector<PositionRef> all = allPositions(games);
    assert(all.size() == 12u);
    size_t i = 0;
    for (const GameRecord& g : games)
        for (int ply = 0; ply < g.positions(); ++ply, ++i) assert(all[i].game == &g && all[i].ply == ply);
}

// The held-out loss by hand: each position encoded at the identity
// symmetry, one network evaluation per position in inference mode, and the
// three training loss terms written out (cross-entropy against the visit
// distribution, squared error against z, per-point squared error against
// the final owners), averaged over every position. heldOutLoss agrees for
// any batch size, and leaves the network exactly as it was.
static void test_held_out_loss_matches_hand_computation() {
    const std::vector<GameRecord> games = someGames();
    const std::vector<PositionRef> positions = allPositions(games);
    for (bool withHead : {true, false}) {
        const NetConfig cfg = tinyNet(withHead);
        const int A = cfg.area(), actions = cfg.actions();
        NaiveBackend be;
        Network net(cfg, 31);
        // A few SGD steps on unrelated data, so the running statistics and
        // weights are not their initial values.
        {
            Workspace ws(cfg, 4, be);
            ReplayBuffer buf(100);
            for (const GameRecord& g : games) buf.add(g);
            Rng rng(32);
            TrainingBatch batch;
            for (int step = 0; step < 3; ++step) {
                buf.sample(4, cfg.history, rng, batch);
                net.trainStep(ws, batch.input.data(), batch.pi.data(), batch.z.data(), 4,
                              withHead ? batch.ownership.data() : nullptr, 1.5f);
                for (ParamView& p : net.params())
                    for (size_t k = 0; k < p.size; ++k) p.value[k] -= 0.1f * p.grad[k];
            }
        }

        double policy = 0, value = 0, owner = 0;
        Workspace one(cfg, 1, be);
        std::vector<float> input(static_cast<size_t>(cfg.planes()) * A), pi(actions), target(A);
        std::vector<float> logits(actions), own(A);
        for (const PositionRef& p : positions) {
            float z = 0, v = 0;
            encodeTrainingSample(*p.game, p.ply, cfg.history, 0, input.data(), pi.data(), &z, target.data());
            net.infer(one, input.data(), 1, logits.data(), &v, withHead ? own.data() : nullptr);
            double sumExp = 0;
            for (int a = 0; a < actions; ++a) sumExp += std::exp(static_cast<double>(logits[a]));
            for (int a = 0; a < actions; ++a) policy -= pi[a] * (logits[a] - std::log(sumExp));
            value += (z - v) * (z - v);
            if (withHead)
                for (int q = 0; q < A; ++q) owner += (own[q] - target[q]) * (own[q] - target[q]);
        }
        const double count = static_cast<double>(positions.size());

        const std::vector<float> before = snapshot(net);
        for (int batchSize : {1, 5, 64}) {
            Workspace ws(cfg, batchSize, be);
            const HeldOutLoss loss = heldOutLoss(net, ws, positions, cfg.history);
            assert(loss.positions == positions.size());
            assert(near(loss.policyLoss, policy / count, 1e-4));
            assert(near(loss.valueLoss, value / count, 1e-5));
            if (withHead) assert(near(loss.ownershipLoss, owner / (count * A), 1e-5));
            else assert(loss.ownershipLoss == -1.0f);
        }
        assert(snapshot(net) == before);
    }
}

// No positions, no measurement: every loss is -1.
static void test_held_out_loss_of_nothing() {
    const NetConfig cfg = tinyNet(true);
    NaiveBackend be;
    Network net(cfg, 1);
    Workspace ws(cfg, 4, be);
    const HeldOutLoss loss = heldOutLoss(net, ws, {}, cfg.history);
    assert(loss.positions == 0 && loss.policyLoss == -1.0f && loss.valueLoss == -1.0f &&
           loss.ownershipLoss == -1.0f);
}

// samplePositions draws uniformly over the buffer's positions with only the
// generator it is given, one draw per position: the same seed gives the
// same sample, and it is exactly ReplayBuffer::locate of rng.below(positions).
static void test_sample_positions_uses_only_its_rng() {
    ReplayBuffer buf(100);
    for (const GameRecord& g : someGames()) buf.add(g);
    Rng a(77), b(77), manual(77);
    const std::vector<PositionRef> s1 = samplePositions(buf, 50, a);
    const std::vector<PositionRef> s2 = samplePositions(buf, 50, b);
    assert(s1.size() == 50u && a.state() == b.state());
    for (size_t i = 0; i < s1.size(); ++i) {
        assert(s1[i].game == s2[i].game && s1[i].ply == s2[i].ply);
        const std::pair<size_t, int> where = buf.locate(static_cast<size_t>(manual.below(12)));
        assert(s1[i].game == &buf.game(where.first) && s1[i].ply == where.second);
    }
    assert(manual.state() == a.state());
    assert(samplePositions(buf, 0, a).empty());
}

int main() {
    test_all_positions_in_game_then_ply_order();
    test_held_out_loss_matches_hand_computation();
    test_held_out_loss_of_nothing();
    test_sample_positions_uses_only_its_rng();
    std::printf("test_validation: all tests passed\n");
    return 0;
}
