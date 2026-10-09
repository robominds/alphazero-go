#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/bn_calibration.hpp"
#include "azgo/network.hpp"
#include "azgo/rng.hpp"
#include "azgo/run_state.hpp"

using namespace azgo;

static std::vector<float> buffersOf(Network& net) {
    std::vector<float> all;
    for (const BufferView& b : net.buffers()) all.insert(all.end(), b.value, b.value + b.size);
    return all;
}

static std::vector<float> paramsOf(Network& net) {
    std::vector<float> all;
    for (const ParamView& p : net.params()) all.insert(all.end(), p.value, p.value + p.size);
    return all;
}

// The positions: deterministic in the seed, encoded with history, lengths
// spread from the empty board to n * n moves, both colors to move.
static void test_positions() {
    const int n = 9, H = 8, A = n * n, P = 2 * H + 2, count = 256;
    const std::vector<float> a = bnCalibrationPositions(n, H, 5, count);
    assert(a.size() == static_cast<size_t>(count) * P * A);
    assert(bnCalibrationPositions(n, H, 5, count) == a);
    assert(bnCalibrationPositions(n, H, 6, count) != a);
    int maxStones = 0, blackToMove = 0;
    for (int i = 0; i < count; ++i) {
        const float* pos = a.data() + static_cast<size_t>(i) * P * A;
        const int length = i * A / count;
        int stones = 0;
        for (int p = 0; p < 2 * A; ++p) stones += pos[p] != 0.0f;   // own and opponent, now
        if (length == 0) assert(stones == 0);
        assert(stones <= length);
        // History: the planes of one move ago differ from the current ones
        // once a move has been played (it put a stone on the board).
        if (length >= 1) assert(!std::equal(pos, pos + 2 * A, pos + 2 * A));
        maxStones = std::max(maxStones, stones);
        // Exactly one of the two side-to-move planes is all ones: Black after
        // an even number of moves (no pass is played before n * n moves here).
        const bool black = pos[2 * H * A] == 1.0f;
        for (int p = 0; p < A; ++p) {
            assert(pos[2 * H * A + p] == (black ? 1.0f : 0.0f));
            assert(pos[(2 * H + 1) * A + p] == (black ? 0.0f : 1.0f));
        }
        assert(black == (length % 2 == 0));
        blackToMove += black;
    }
    assert(maxStones >= A / 2);                                  // a mid-game board
    assert(blackToMove > count / 3 && blackToMove < 2 * count / 3);
}

// Value and mean policy entropy (nats) of `net` in inference mode on `B`
// positions.
struct Outputs {
    float emptyValue = 0;
    std::vector<float> absValues;   // sorted
    double meanEntropy = 0;
    float minAbs() const { return absValues.front(); }
    float medianAbs() const { return absValues[absValues.size() / 2]; }
};

static Outputs inferOutputs(const Network& net, const std::vector<float>& positions, int B) {
    const NetConfig& cfg = net.config();
    NaiveBackend be;
    Workspace ws(cfg, B, be);
    std::vector<float> logits(static_cast<size_t>(B) * cfg.actions()), values(B);
    net.infer(ws, positions.data(), B, logits.data(), values.data());
    Outputs out;
    out.emptyValue = values[0];
    for (int b = 0; b < B; ++b) {
        out.absValues.push_back(std::fabs(values[b]));
        const float* l = logits.data() + static_cast<size_t>(b) * cfg.actions();
        const double maxLogit = *std::max_element(l, l + cfg.actions());
        double z = 0;
        for (int a = 0; a < cfg.actions(); ++a) z += std::exp(l[a] - maxLogit);
        double entropy = 0;
        for (int a = 0; a < cfg.actions(); ++a) {
            const double p = std::exp(l[a] - maxLogit) / z;
            if (p > 0) entropy -= p * std::log(p);
        }
        out.meanEntropy += entropy / B;
    }
    std::sort(out.absValues.begin(), out.absValues.end());
    return out;
}

// The bug this fixes, and the fix: the 9x9 preset's generation 0 with a
// 10x128 network (seed 1) saturates in inference mode without calibration
// (|value| 1.00 everywhere, the empty board included, and a near one-hot
// policy) and does not with it. A calibrated random network is still a
// random function: on some positions its value is large (as in training
// mode, whose batch statistics it now matches), but not everywhere, and its
// policy is close to uniform.
static void test_deep_network_is_not_saturated() {
    NetConfig cfg;
    cfg.boardSize = 9;
    cfg.history = 8;
    cfg.blocks = 10;
    cfg.filters = 128;
    cfg.valueHidden = 64;
    const uint64_t runSeed = 1;
    // Positions the calibration did not see (position 0 is the empty board),
    // from the empty board to n * n moves.
    const int B = 16;
    std::vector<float> probe = bnCalibrationPositions(9, 8, 12345, B);

    Network net(cfg, deriveSeed(runSeed, 0, kPhaseInit));
    const Outputs before = inferOutputs(net, probe, B);
    const auto start = std::chrono::steady_clock::now();
    calibrateBatchNorm(net, deriveSeed(runSeed, 0, kPhaseBatchNorm), 4);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const Outputs after = inferOutputs(net, probe, B);
    std::printf("10x128 9x9 gen 0: empty-board value %.4f -> %.4f, min |v| %.4f -> %.4f, median |v| "
                "%.4f -> %.4f, policy entropy %.3f -> %.3f nats (uniform %.3f); calibration %.2f s\n",
                before.emptyValue, after.emptyValue, before.minAbs(), after.minAbs(), before.medianAbs(),
                after.medianAbs(), before.meanEntropy, after.meanEntropy, std::log(82.0), seconds);
    assert(before.minAbs() > 0.99f && before.meanEntropy < 1.0);    // the bug
    assert(std::fabs(after.emptyValue) < 0.9f && after.medianAbs() < 0.9f && after.minAbs() < 0.5f);
    assert(after.meanEntropy > 3.0);
}

// The free function: the same bits for any thread count; only the running
// statistics change.
static void test_calibration_is_thread_independent() {
    NetConfig cfg;
    cfg.boardSize = 7;
    cfg.history = 2;
    cfg.blocks = 2;
    cfg.filters = 24;
    cfg.valueHidden = 8;
    cfg.ownership = true;
    Network fresh(cfg, 3);
    Network one(cfg, 3);
    calibrateBatchNorm(one, 99, 1);
    assert(paramsOf(one) == paramsOf(fresh));
    assert(buffersOf(one) != buffersOf(fresh));
    for (int threads : {2, 5}) {
        Network other(cfg, 3);
        calibrateBatchNorm(other, 99, threads);
        assert(buffersOf(other) == buffersOf(one));
    }
    Network otherSeed(cfg, 3);
    calibrateBatchNorm(otherSeed, 100, 1);
    assert(buffersOf(otherSeed) != buffersOf(one));
}

int main() {
    test_positions();
    test_calibration_is_thread_independent();
    test_deep_network_is_not_saturated();
    std::printf("test_bn_calibration: all tests passed\n");
    return 0;
}
