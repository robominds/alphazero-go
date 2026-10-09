#include "azgo/bn_calibration.hpp"
#include <cassert>
#include <memory>
#include "azgo/backend.hpp"
#include "azgo/encode.hpp"
#include "azgo/game.hpp"
#include "azgo/parallel.hpp"
#include "azgo/rng.hpp"

namespace azgo {

std::vector<float> bnCalibrationPositions(int boardSize, int history, uint64_t seed, int count) {
    assert(count >= 1);
    const int area = boardSize * boardSize;
    const size_t floats = static_cast<size_t>(numPlanes(history)) * area;
    std::vector<float> out(static_cast<size_t>(count) * floats);
    Rng rng(seed);
    for (int i = 0; i < count; ++i) {
        const int length = static_cast<int>(static_cast<long long>(i) * area / count);
        // Komi is not an input plane, and the default move cap is 2 * n * n.
        Game game(boardSize, 0.0f, defaultMaxMoves(boardSize));
        while (game.moveCount() < length && !game.isTerminal()) {
            // legalActions is ascending, so the pass is last.
            const std::vector<int> legal = game.legalActions();
            const int points = static_cast<int>(legal.size()) - 1;
            game.play(points > 0 ? legal[rng.below(points)] : legal.back());
        }
        encodeGame(game, history, 0, out.data() + static_cast<size_t>(i) * floats);
    }
    return out;
}

void calibrateBatchNorm(Network& net, uint64_t seed, int threads) {
    const NetConfig& cfg = net.config();
    const std::vector<float> input =
        bnCalibrationPositions(cfg.boardSize, cfg.history, seed, kBnCalibrationPositions);
    // Naive, not the run's backend: BLAS (and the GPU) round differently,
    // and generation 0 must not depend on the backend. Every value is
    // computed by one chunk of the pool in a fixed order (layers.hpp), so
    // it does not depend on the thread count either.
    std::unique_ptr<Backend> backend = makeBackend("naive", 1);
    ThreadPool pool(threads);
    Workspace ws(cfg, kBnCalibrationPositions, *backend, &pool);
    net.calibrateBatchNorm(ws, input.data(), kBnCalibrationPositions);
}

} // namespace azgo
