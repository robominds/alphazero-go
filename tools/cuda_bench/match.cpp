// cuda16_match: 16-bit against 32-bit. One network plays itself, side A's
// searches evaluated by backend = cuda16 and side B's by a 32-bit engine
// (cuda by default, or blas on the CPU), colours alternating game by game,
// with the arena's search (no root noise, tree reuse, greedy after the
// opening plies). If 16-bit evaluation costs no strength, A scores 50%
// within the interval the game count allows.
//
// Game i is seeded deriveSeed(seed, i) and A plays Black in the even games,
// as in playArenaGame (src/arena.cpp), which this repeats with an evaluator
// per side instead of per network.
//
//   cmake --build build-cuda --target cuda16_match
//   build-cuda/cuda16_match --checkpoint gen_0100.azn [--games 400] [--simulations 800] [--workers 64]
//                           [--b-backend cuda|blas] [--max-seconds 3600] [--temperature-moves 0]
//                           [--random-symmetry 1] [--seed 1]
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>
#include "azgo/arena.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/cuda16.hpp"
#include "azgo/game.hpp"
#include "azgo/inference_hub.hpp"
#include "azgo/mcts.hpp"
#include "azgo/parallel.hpp"
#include "azgo/rng.hpp"

using namespace azgo;

int main(int argc, char** argv) {
    std::string checkpoint, bBackend = "cuda";
    int games = 400, simulations = 800, workers = 64, mctsBatch = 8, temperatureMoves = 0, randomSymmetry = 1;
    double maxSeconds = 3600, komi = 7.5, cPuct = 1.5;
    unsigned long long seed = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--checkpoint") checkpoint = v;
        else if (a == "--b-backend") bBackend = v;
        else if (a == "--games") games = std::atoi(v.c_str());
        else if (a == "--simulations") simulations = std::atoi(v.c_str());
        else if (a == "--workers") workers = std::atoi(v.c_str());
        else if (a == "--mcts-batch") mctsBatch = std::atoi(v.c_str());
        else if (a == "--temperature-moves") temperatureMoves = std::atoi(v.c_str());
        else if (a == "--random-symmetry") randomSymmetry = std::atoi(v.c_str());
        else if (a == "--max-seconds") maxSeconds = std::atof(v.c_str());
        else if (a == "--komi") komi = std::atof(v.c_str());
        else if (a == "--seed") seed = std::strtoull(v.c_str(), nullptr, 10);
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (checkpoint.empty() || games < 2 || workers < 1) {
        std::fprintf(stderr, "usage: cuda16_match --checkpoint CKPT [options]\n");
        return 2;
    }
    try {
        const Network net = loadNetwork(checkpoint);
        const NetConfig cfg = net.config();
        const int sideWorkers = std::min(workers, games);
        WorkerEvaluators sideA(net, "cuda16", sideWorkers, mctsBatch, 256);
        WorkerEvaluators sideB(net, bBackend, sideWorkers, mctsBatch, 256);
        MctsConfig mcts;
        mcts.simulations = simulations;
        mcts.batchSize = mctsBatch;
        mcts.cPuct = static_cast<float>(cPuct);
        mcts.addNoise = false;
        mcts.randomSymmetry = randomSymmetry != 0;
        std::printf("cuda16_match: %dx%d %dx%d, A = cuda16 (%s), B = %s; %d games at %d simulations, %d workers\n",
                    cfg.boardSize, cfg.boardSize, cfg.blocks, cfg.filters, cuda16Description().c_str(),
                    bBackend.c_str(), games, simulations, sideWorkers);
        std::fflush(stdout);

        const auto start = std::chrono::steady_clock::now();
        auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };
        std::mutex m;
        std::vector<double> score(games, -1.0);   // A's score per game, -1: not played
        int done = 0, aWinsBlack = 0, aWinsWhite = 0;
        long long plies = 0;
        parallelFor(games, sideWorkers, [&](int i, int worker) {
            if (elapsed() > maxSeconds) return;   // out of time: the game is not played
            Rng rng(deriveSeed(seed, static_cast<uint64_t>(i)));
            const bool aIsBlack = i % 2 == 0;
            Game game(cfg.boardSize, static_cast<float>(komi), defaultMaxMoves(cfg.boardSize));
            Mcts searchA(mcts, sideA.forWorker(worker)), searchB(mcts, sideB.forWorker(worker));
            while (!game.isTerminal()) {
                const bool blackToMove = game.board().toMove() == Color::Black;
                Mcts& mover = blackToMove == aIsBlack ? searchA : searchB;
                mover.search(game, rng);
                const float temperature = game.moveCount() < temperatureMoves ? 1.0f : 0.0f;
                const int action = mover.selectAction(temperature, rng);
                game.play(action);
                if (searchA.nodeCount() > 0) searchA.advance(action);
                if (searchB.nodeCount() > 0) searchB.advance(action);
            }
            const double s = matchScore(game.resultForBlack(), aIsBlack);
            std::lock_guard<std::mutex> lock(m);
            score[i] = s;
            ++done;
            plies += game.moveCount();
            if (s == 1.0) ++(aIsBlack ? aWinsBlack : aWinsWhite);
            double total = 0;
            for (double v : score)
                if (v >= 0) total += v;
            const double p = total / done, se = std::sqrt(std::max(p * (1 - p), 1e-12) / done);
            std::printf("game %4d (A %s) A %.1f | %d games: A %.1f = %.1f%% +- %.1f%% (95%%) | %.0f s\n", i,
                        aIsBlack ? "black" : "white", s, done, total, 100 * p, 196 * se, elapsed());
            std::fflush(stdout);
        });
        double total = 0;
        int black = 0, white = 0;
        for (int i = 0; i < games; ++i)
            if (score[i] >= 0) {
                total += score[i];
                ++(i % 2 == 0 ? black : white);
            }
        const double p = done ? total / done : 0, se = done ? std::sqrt(std::max(p * (1 - p), 1e-12) / done) : 0;
        std::printf("final: %d games (A black %d, white %d): A scores %.1f = %.2f%% +- %.2f%% (95%%); A won %d as "
                    "black, %d as white; %.1f plies a game; %.0f s; evaluations A %lld, B %lld\n",
                    done, black, white, total, 100 * p, 196 * se, aWinsBlack, aWinsWhite,
                    done ? static_cast<double>(plies) / done : 0.0, elapsed(), sideA.evaluations(), sideB.evaluations());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
