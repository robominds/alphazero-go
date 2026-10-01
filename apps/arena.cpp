// arena: plays a checkpoint against another checkpoint, or against a uniform
// bot, and prints A's score with a 95% Wilson confidence interval. For manual
// experiments; the trainer runs its own arena every generation.
//
//   arena --a runs/nine/checkpoints/gen_0040.azn --b runs/nine/checkpoints/gen_0020.azn
//   arena --a runs/nine/checkpoints/gen_0040.azn --b-bot 100
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "azgo/arena.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/elo.hpp"
#include "azgo/network.hpp"
#include "azgo/parallel.hpp"
#include "cli.hpp"

namespace {

const char* const kUsage =
    "usage: arena --a CKPT (--b CKPT | --b-bot SIMS) [--games 40] [--simulations 200]\n"
    "             [--komi 7.5] [--temperature-moves 4] [--workers 0] [--backend naive] [--seed 1]\n"
    "\n"
    "Colors alternate: A plays Black in even-numbered games. --workers 0 uses every core.\n"
    "--b-bot SIMS makes B a uniform bot: MCTS with equal priors and value 0, searching\n"
    "SIMS simulations per move (1-100000) on A's board size.\n";

int usageError(const std::string& message) {
    std::fprintf(stderr, "error: %s\n\n%s", message.c_str(), kUsage);
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    std::unique_ptr<ArgParser> args;
    try {
        args = std::make_unique<ArgParser>(
            argc, argv,
            std::vector<std::string>{"a", "b", "b-bot", "games", "simulations", "komi",
                                     "temperature-moves", "workers", "backend", "seed"},
            std::vector<std::string>{"help"});
    } catch (const std::invalid_argument& e) {
        return usageError(e.what());
    }
    if (args->flag("help")) {
        std::fputs(kUsage, stdout);
        return 0;
    }
    if (!args->has("a")) return usageError("--a is required");
    if (args->has("b") == args->has("b-bot"))
        return usageError("give exactly one of --b CKPT and --b-bot SIMS");

    try {
        azgo::ArenaConfig cfg;
        cfg.games = args->getInt("games", 40);
        cfg.simulations = args->getInt("simulations", 200);
        cfg.komi = static_cast<float>(args->getDouble("komi", 7.5));
        cfg.temperatureMoves = args->getInt("temperature-moves", 4);
        const int workers = args->getInt("workers", 0);
        const int seed = args->getInt("seed", 1);
        const std::string backend = args->get("backend", "naive");
        const int botSims = args->getInt("b-bot", 0);
        if (cfg.games < 2 || cfg.games % 2 != 0)
            throw std::invalid_argument("--games must be even and at least 2 (colors are balanced)");
        if (cfg.simulations < 1) throw std::invalid_argument("--simulations must be at least 1");
        if (cfg.temperatureMoves < 0) throw std::invalid_argument("--temperature-moves must be >= 0");
        if (workers < 0) throw std::invalid_argument("--workers must be >= 0");
        if (seed < 0) throw std::invalid_argument("--seed must be >= 0");
        if (args->has("b-bot") && (botSims < 1 || botSims > 100000))
            throw std::invalid_argument("--b-bot must be in [1, 100000]");

        const std::string pathA = args->get("a");
        const azgo::Network a = azgo::loadNetwork(pathA);   // errors name the file
        std::unique_ptr<azgo::Network> b;
        azgo::ArenaPlayer playerB = azgo::ArenaPlayer::bot(botSims);
        std::string nameB = "bot(" + std::to_string(botSims) + " simulations)";
        if (args->has("b")) {
            nameB = args->get("b");
            b = std::make_unique<azgo::Network>(azgo::loadNetwork(nameB));
            playerB = azgo::ArenaPlayer::network(*b);
        }

        const azgo::MatchResult r =
            azgo::playMatch(azgo::ArenaPlayer::network(a), playerB, cfg, static_cast<uint64_t>(seed),
                            azgo::resolveWorkers(workers), backend);
        const std::pair<double, double> ci = azgo::wilsonInterval(r.scoreA, r.games);
        std::printf("A %s vs B %s: A scored %g/%d = %.3f (95%% CI %.3f-%.3f), distinct games %d\n",
                    pathA.c_str(), nameB.c_str(), r.scoreA, r.games, r.scoreA / r.games,
                    ci.first, ci.second, r.distinctGames);
        std::printf("A wins %d, B wins %d, draws %d\n", r.winsA, r.winsB, r.draws);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
