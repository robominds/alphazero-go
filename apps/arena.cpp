// arena: plays a checkpoint against another checkpoint, or against a uniform
// bot, and prints A's score with a 95% Wilson confidence interval. For manual
// experiments; the trainer runs its own arena every generation.
//
//   arena --a runs/nine/checkpoints/gen_0040.azn --b runs/nine/checkpoints/gen_0020.azn
//   arena --a runs/nine/checkpoints/gen_0040.azn --b-bot 100
//   arena --a runs/nine/checkpoints/gen_0040.azn --b runs/nine/checkpoints/gen_0020.azn --b-simulations 3200
//   arena --a runs/nine/checkpoints/gen_0040.azn --b runs/nine/checkpoints/gen_0020.azn --random-symmetry
//   arena --a runs/nine/checkpoints/gen_0040.azn --b runs/nine/checkpoints/gen_0040.azn --a-random-symmetry
//   arena --a runs/nine/checkpoints/gen_0040.azn --b runs/nine/checkpoints/gen_0040.azn --a-c-puct 3
//   arena --a runs/nine/checkpoints/gen_0040.azn --b runs/nine/checkpoints/gen_0040.azn --a-mcts-batch 1
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
    "             [--a-simulations N] [--b-simulations N] [--komi 7.5] [--temperature-moves 4]\n"
    "             [--workers 0] [--backend naive] [--seed 1] [--random-symmetry]\n"
    "             [--a-random-symmetry] [--b-random-symmetry] [--c-puct 1.5] [--mcts-batch 8]\n"
    "             [--a-c-puct X] [--b-c-puct X] [--a-mcts-batch N] [--b-mcts-batch N]\n"
    "\n"
    "Colors alternate: A plays Black in even-numbered games. --workers 0 uses every core.\n"
    "--b-bot SIMS makes B a uniform bot: MCTS with equal priors and value 0, searching\n"
    "SIMS simulations per move (1-100000) on A's board size.\n"
    "--a-simulations and --b-simulations (1-100000) override --simulations for one side, so a\n"
    "frozen checkpoint can serve as a stronger reference by searching more. A bot already has its\n"
    "own count, so --b-bot and --b-simulations cannot be combined.\n"
    "--random-symmetry evaluates every position in both networks' searches under a random\n"
    "one of the 8 board symmetries, as AlphaGo Zero did (a bot is unaffected).\n"
    "--a-random-symmetry and --b-random-symmetry do so for one side only, so a checkpoint can\n"
    "play itself with symmetry on one side to measure what it is worth. A bot gains nothing\n"
    "from it, so --b-bot and --b-random-symmetry cannot be combined.\n"
    "--c-puct (0, 100] and --mcts-batch ([1, 256], the descents per network call, kept apart by\n"
    "virtual loss) set both sides' search; --a-c-puct, --b-c-puct, --a-mcts-batch and\n"
    "--b-mcts-batch set one side's, so a checkpoint can play itself with one search setting\n"
    "changed on one side. A bot searches with both, so B's apply to --b-bot too.\n";

int usageError(const std::string& message) {
    std::fprintf(stderr, "error: %s\n\n%s", message.c_str(), kUsage);
    return 1;
}

std::string formatNumber(double x) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%g", x);
    return buf;
}

// --NAME as a c_puct, in the train config's range (0, 100]; `def` if absent.
// The range is checked on the float the search will use: 1e-50 rounds to 0,
// which would otherwise slip through as "inherit".
float cPuctOption(const ArgParser& args, const std::string& name, float def) {
    if (!args.has(name)) return def;
    const float c = static_cast<float>(args.getDouble(name, def));
    if (!(c > 0.0f && c <= 100.0f)) throw std::invalid_argument("--" + name + " must be in (0, 100]");
    return c;
}

// --NAME as an MCTS batch size, in the train config's range [1, 256]; `def` if absent.
int mctsBatchOption(const ArgParser& args, const std::string& name, int def) {
    if (!args.has(name)) return def;
    const int b = args.getInt(name, def);
    if (b < 1 || b > 256) throw std::invalid_argument("--" + name + " must be in [1, 256]");
    return b;
}

// The one "Search:" line, from the searches the two seats will actually run,
// or "" when every setting it covers is at its default:
//   Search: random symmetry per network evaluation, A only; c_puct 2; mcts batch A 8, B 1
// Random symmetry is named when either side has it; c_puct and the batch,
// each once if the sides share it and per side otherwise, when either side
// is off the defaults.
std::string searchLine(const azgo::MctsConfig& a, const azgo::MctsConfig& b) {
    std::string line;
    auto add = [&](const std::string& part) { line += (line.empty() ? "Search: " : "; ") + part; };
    if (a.randomSymmetry || b.randomSymmetry)
        add(std::string("random symmetry per network evaluation") +
            (a.randomSymmetry && b.randomSymmetry ? "" : a.randomSymmetry ? ", A only" : ", B only"));
    const azgo::ArenaConfig defaults;
    if (a.cPuct != defaults.cPuct || b.cPuct != defaults.cPuct || a.batchSize != defaults.mctsBatch ||
        b.batchSize != defaults.mctsBatch) {
        auto setting = [](const std::string& name, const std::string& x, const std::string& y) {
            return x == y ? name + " " + x : name + " A " + x + ", B " + y;
        };
        add(setting("c_puct", formatNumber(a.cPuct), formatNumber(b.cPuct)));
        add(setting("mcts batch", std::to_string(a.batchSize), std::to_string(b.batchSize)));
    }
    return line;
}

} // namespace

int main(int argc, char** argv) {
    std::unique_ptr<ArgParser> args;
    try {
        args = std::make_unique<ArgParser>(
            argc, argv,
            std::vector<std::string>{"a", "b", "b-bot", "games", "simulations", "a-simulations",
                                     "b-simulations", "komi", "c-puct", "mcts-batch",
                                     "a-c-puct", "b-c-puct", "a-mcts-batch", "b-mcts-batch",
                                     "temperature-moves", "workers", "backend", "seed"},
            std::vector<std::string>{"help", "random-symmetry", "a-random-symmetry",
                                     "b-random-symmetry"});
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
        cfg.randomSymmetry = args->flag("random-symmetry");
        cfg.cPuct = cPuctOption(*args, "c-puct", cfg.cPuct);
        cfg.mctsBatch = mctsBatchOption(*args, "mcts-batch", cfg.mctsBatch);
        // Each side's c_puct and batch: its own option, else the shared one.
        const float cPuctA = cPuctOption(*args, "a-c-puct", cfg.cPuct);
        const float cPuctB = cPuctOption(*args, "b-c-puct", cfg.cPuct);
        const int batchA = mctsBatchOption(*args, "a-mcts-batch", cfg.mctsBatch);
        const int batchB = mctsBatchOption(*args, "b-mcts-batch", cfg.mctsBatch);
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
        const int simsA = args->getInt("a-simulations", 0);
        const int simsB = args->getInt("b-simulations", 0);
        if (args->has("a-simulations") && (simsA < 1 || simsA > 100000))
            throw std::invalid_argument("--a-simulations must be in [1, 100000]");
        if (args->has("b-bot") && args->has("b-simulations"))
            throw std::invalid_argument("--b-simulations applies to a checkpoint B; a bot's count is --b-bot");
        if (args->has("b-simulations") && (simsB < 1 || simsB > 100000))
            throw std::invalid_argument("--b-simulations must be in [1, 100000]");
        if (args->has("b-bot") && (botSims < 1 || botSims > 100000))
            throw std::invalid_argument("--b-bot must be in [1, 100000]");
        if (args->has("b-bot") && args->flag("b-random-symmetry"))
            throw std::invalid_argument("--b-random-symmetry applies to a checkpoint B; a bot's equal priors gain nothing from it");
        const std::string pathA = args->get("a");
        const azgo::Network a = azgo::loadNetwork(pathA);   // errors name the file
        std::unique_ptr<azgo::Network> b;
        azgo::ArenaPlayer playerB = azgo::ArenaPlayer::bot(botSims);
        std::string nameB = "bot(" + std::to_string(botSims) + " simulations)";
        if (args->has("b")) {
            nameB = args->get("b");
            b = std::make_unique<azgo::Network>(azgo::loadNetwork(nameB));
            playerB = azgo::ArenaPlayer::network(*b, simsB, args->flag("b-random-symmetry"));
        }
        azgo::ArenaPlayer playerA = azgo::ArenaPlayer::network(a, simsA, args->flag("a-random-symmetry"));
        playerA.cPuct = cPuctA;
        playerA.mctsBatch = batchA;
        playerB.cPuct = cPuctB;
        playerB.mctsBatch = batchB;
        // What each seat will search with, read back from the players
        // themselves, so the labels and the Search: line record the match
        // that is actually played.
        const azgo::MctsConfig searchA = azgo::arenaSearchConfig(playerA, cfg);
        const azgo::MctsConfig searchB = azgo::arenaSearchConfig(playerB, cfg);
        // Name each side's settings where they differ from the shared ones
        // (--simulations, --c-puct, --mcts-batch). A bot's name has its count.
        auto settings = [&](const azgo::MctsConfig& m, bool bot) {
            std::string parts;
            auto add = [&](const std::string& part) { parts += (parts.empty() ? "" : ", ") + part; };
            if (!bot && m.simulations != cfg.simulations) add(std::to_string(m.simulations) + " simulations");
            if (m.cPuct != cfg.cPuct) add("c_puct " + formatNumber(m.cPuct));
            if (m.batchSize != cfg.mctsBatch) add("mcts batch " + std::to_string(m.batchSize));
            return parts.empty() ? parts : " (" + parts + ")";
        };
        const std::string labelA = pathA + settings(searchA, false);
        nameB += settings(searchB, !playerB.net);
        // Printed before the match starts, so a long run's output says what
        // it is measuring from the first line.
        const std::string search = searchLine(searchA, searchB);
        if (!search.empty()) {
            std::printf("%s\n", search.c_str());
            std::fflush(stdout);
        }

        const azgo::MatchResult r =
            azgo::playMatch(playerA, playerB, cfg, static_cast<uint64_t>(seed),
                            azgo::resolveWorkers(workers), backend);
        const std::pair<double, double> ci = azgo::wilsonInterval(r.scoreA, r.games);
        std::printf("A %s vs B %s: A scored %g/%d = %.3f (95%% CI %.3f-%.3f), distinct games %d\n",
                    labelA.c_str(), nameB.c_str(), r.scoreA, r.games, r.scoreA / r.games,
                    ci.first, ci.second, r.distinctGames);
        std::printf("A wins %d, B wins %d, draws %d\n", r.winsA, r.winsB, r.draws);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
