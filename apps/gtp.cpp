// gtp: a checkpoint as a Go Text Protocol engine, for evaluation only. A GTP
// controller (tools/gtp_match.py, gogui-twogtp) or a board program such as
// Sabaki starts it and talks to it on stdin/stdout, so the network can play
// an outside reference such as GNU Go. Every move is an MCTS search at
// temperature 0 without root noise, reusing the subtree between moves.
// stdout carries only GTP; the startup line, one line per genmove and
// errors go to stderr.
//
//   gtp --checkpoint runs/nine/checkpoints/gen_0040.azn
//   gtp --checkpoint gen_0040.azn --simulations 800 --backend blas --random-symmetry
//   echo "genmove b" | gtp --checkpoint gen_0040.azn
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "azgo/backend.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/evaluator.hpp"
#include "azgo/gtp.hpp"
#include "azgo/network.hpp"
#include "cli.hpp"

namespace {

const char* const kUsage =
    "usage: gtp --checkpoint CKPT [--simulations 200] [--mcts-batch 8] [--c-puct 1.5]\n"
    "           [--backend naive] [--seed 1] [--random-symmetry] [--komi 7.5]\n"
    "\n"
    "Speaks GTP version 2 on stdin/stdout: protocol_version, name, version,\n"
    "known_command, list_commands, quit, boardsize (the checkpoint's size only),\n"
    "clear_board, komi, play, genmove, undo, showboard and final_score.\n"
    "genmove searches --simulations (1-100000) simulations at temperature 0 with no\n"
    "root noise and never resigns; final_score is the Tromp-Taylor area score\n"
    "minus komi, every stone on the board counting. Rules are the project's:\n"
    "positional superko, no suicide. After two passes in a row the game is over:\n"
    "play answers \"? illegal move\" and genmove \"? game is over\" until clear_board.\n"
    "--komi applies until the controller sends komi. --random-symmetry evaluates\n"
    "each position under a random one of the 8 board symmetries, as AlphaGo Zero\n"
    "did; --seed (>= 0) seeds it and the tie-breaks. --c-puct is in (0, 100],\n"
    "--mcts-batch in [1, 256].\n";

int usageError(const std::string& message) {
    std::fprintf(stderr, "error: %s\n\n%s", message.c_str(), kUsage);
    return 1;
}

std::string baseName(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

} // namespace

int main(int argc, char** argv) {
    std::unique_ptr<ArgParser> args;
    try {
        args = std::make_unique<ArgParser>(
            argc, argv,
            std::vector<std::string>{"checkpoint", "simulations", "mcts-batch", "c-puct", "backend",
                                     "seed", "komi"},
            std::vector<std::string>{"help", "random-symmetry"});
    } catch (const std::invalid_argument& e) {
        return usageError(e.what());
    }
    if (args->flag("help")) {
        std::fputs(kUsage, stdout);
        return 0;
    }
    if (!args->has("checkpoint")) return usageError("--checkpoint CKPT is required");

    try {
        azgo::GtpOptions opts;
        opts.search.simulations = args->getInt("simulations", 200);
        opts.search.batchSize = args->getInt("mcts-batch", 8);
        opts.search.cPuct = static_cast<float>(args->getDouble("c-puct", 1.5));
        opts.search.randomSymmetry = args->flag("random-symmetry");
        opts.komi = static_cast<float>(args->getDouble("komi", 7.5));
        const int seed = args->getInt("seed", 1);
        if (opts.search.simulations < 1 || opts.search.simulations > 100000)
            throw std::invalid_argument("--simulations must be in [1, 100000]");
        if (opts.search.batchSize < 1 || opts.search.batchSize > 256)
            throw std::invalid_argument("--mcts-batch must be in [1, 256]");
        if (!(opts.search.cPuct > 0.0f && opts.search.cPuct <= 100.0f))
            throw std::invalid_argument("--c-puct must be in (0, 100]");
        if (seed < 0) throw std::invalid_argument("--seed must be >= 0");
        opts.seed = static_cast<uint64_t>(seed);

        const std::string ckpt = args->get("checkpoint");
        const azgo::Network net = azgo::loadNetwork(ckpt);   // errors name the file
        const int n = net.config().boardSize;
        std::unique_ptr<azgo::Backend> backend = azgo::makeBackend(args->get("backend", "naive"), 1);
        azgo::LocalEvaluator evaluator(net, *backend, opts.search.batchSize);
        // The version names what is playing, so a controller's records
        // (SGF PB/PW) say which network and budget a game was played with.
        opts.version = baseName(ckpt) + ", " + std::to_string(opts.search.simulations) + " simulations" +
                       (opts.search.randomSymmetry ? ", random symmetry" : "");
        opts.log = &std::cerr;
        azgo::GtpEngine engine(evaluator, opts);
        std::fprintf(stderr, "gtp: %s (%dx%d), %d simulations per move, komi %g; reading GTP on stdin\n",
                     ckpt.c_str(), n, n, opts.search.simulations, opts.komi);

        std::string line;
        while (!engine.quitRequested() && std::getline(std::cin, line)) {
            const std::string response = engine.handle(line);
            if (response.empty()) continue;
            std::fputs(response.c_str(), stdout);
            std::fflush(stdout);   // the controller waits for each answer
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
