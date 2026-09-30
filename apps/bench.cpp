// bench: measures what the presets cost, so their sizes are chosen from
// numbers rather than guesses. For each board size's preset network and each
// backend: inference throughput at the MCTS batch size on one thread (what a
// self-play worker sees) and the time of one training step at batch 128 on
// all cores (what the training phase sees). Then raw rules-engine speed.
//
//   bench [--sizes 5,9,19] [--backend naive|blas|all] [--seconds 2] [--configs configs]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "azgo/backend.hpp"
#include "azgo/config.hpp"
#include "azgo/game.hpp"
#include "azgo/network.hpp"
#include "azgo/parallel.hpp"
#include "azgo/rng.hpp"
#include "cli.hpp"

namespace {

const char* const kUsage =
    "usage: bench [--sizes 5,9,19] [--backend naive|blas|all] [--seconds 2] [--configs DIR]\n"
    "\n"
    "Network shapes come from DIR/<n>x<n>.cfg (default DIR: configs). --seconds is\n"
    "the minimum duration of each measurement.\n";

constexpr int kInferBatch = 8;     // the default MCTS batch (mcts_batch)
constexpr int kTrainBatch = 128;

using Clock = std::chrono::steady_clock;

int usageError(const std::string& message) {
    std::fprintf(stderr, "error: %s\n\n%s", message.c_str(), kUsage);
    return 1;
}

double secondsSince(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::vector<int> parseSizes(const std::string& text) {
    std::vector<int> sizes;
    std::stringstream in(text);
    for (std::string item; std::getline(in, item, ',');) {
        char* end = nullptr;
        const long n = std::strtol(item.c_str(), &end, 10);
        if (item.empty() || *end != '\0' || n < azgo::kMinBoardSize || n > azgo::kMaxBoardSize)
            throw std::invalid_argument("--sizes: '" + item + "' is not a board size in 5..19");
        sizes.push_back(static_cast<int>(n));
    }
    if (sizes.empty()) throw std::invalid_argument("--sizes is empty");
    return sizes;
}

// Random {0,1} planes: the network's cost does not depend on the values.
std::vector<float> randomInput(const azgo::NetConfig& nc, int batch, azgo::Rng& rng) {
    std::vector<float> input(static_cast<size_t>(batch) * nc.planes() * nc.area());
    for (float& x : input) x = static_cast<float>(rng.below(2));
    return input;
}

// Positions evaluated per second, batch kInferBatch, one thread.
double measureInference(const azgo::NetConfig& nc, const std::string& backendName, double seconds) {
    std::unique_ptr<azgo::Backend> backend = azgo::makeBackend(backendName, 1);
    const azgo::Network net(nc, 1);
    azgo::Workspace ws(nc, kInferBatch, *backend);
    azgo::Rng rng(7);
    const std::vector<float> input = randomInput(nc, kInferBatch, rng);
    std::vector<float> logits(static_cast<size_t>(kInferBatch) * nc.actions());
    std::vector<float> values(kInferBatch);

    net.infer(ws, input.data(), kInferBatch, logits.data(), values.data());   // warm-up
    long long calls = 0;
    const auto start = Clock::now();
    double elapsed = 0;
    do {
        net.infer(ws, input.data(), kInferBatch, logits.data(), values.data());
        ++calls;
        elapsed = secondsSince(start);
    } while (elapsed < seconds);
    return static_cast<double>(calls) * kInferBatch / elapsed;
}

// Milliseconds per training step (forward + backward), batch kTrainBatch,
// backend given every hardware thread as in the trainer.
double measureTrainStep(const azgo::NetConfig& nc, const std::string& backendName, double seconds) {
    std::unique_ptr<azgo::Backend> backend = azgo::makeBackend(backendName, azgo::resolveWorkers(0));
    azgo::Network net(nc, 1);
    azgo::Workspace ws(nc, kTrainBatch, *backend);
    azgo::Rng rng(11);
    const std::vector<float> input = randomInput(nc, kTrainBatch, rng);
    std::vector<float> pi(static_cast<size_t>(kTrainBatch) * nc.actions(), 0.0f);
    std::vector<float> z(kTrainBatch);
    for (int b = 0; b < kTrainBatch; ++b) {
        pi[static_cast<size_t>(b) * nc.actions() + rng.below(nc.actions())] = 1.0f;   // one-hot target
        z[static_cast<size_t>(b)] = rng.below(2) ? 1.0f : -1.0f;
    }

    net.trainStep(ws, input.data(), pi.data(), z.data(), kTrainBatch);   // warm-up
    long long steps = 0;
    const auto start = Clock::now();
    double elapsed = 0;
    do {
        net.trainStep(ws, input.data(), pi.data(), z.data(), kTrainBatch);
        ++steps;
        elapsed = secondsSince(start);
    } while (elapsed < seconds);
    return 1000.0 * elapsed / static_cast<double>(steps);
}

// Moves per second of uniformly random legal play through Game (legality
// including superko, captures, hashing), games restarted when they end.
double measurePlayouts(int n, double seconds) {
    azgo::Rng rng(static_cast<uint64_t>(n));
    long long moves = 0;
    const auto start = Clock::now();
    double elapsed = 0;
    do {
        azgo::Game game(n, 7.5f, azgo::defaultMaxMoves(n));
        while (!game.isTerminal()) {
            const std::vector<int> legal = game.legalActions();
            game.play(legal[static_cast<size_t>(rng.below(static_cast<int>(legal.size())))]);
            ++moves;
        }
        elapsed = secondsSince(start);
    } while (elapsed < seconds);
    return static_cast<double>(moves) / elapsed;
}

} // namespace

int main(int argc, char** argv) {
    std::unique_ptr<ArgParser> args;
    try {
        args = std::make_unique<ArgParser>(argc, argv,
                                           std::vector<std::string>{"sizes", "backend", "seconds", "configs"},
                                           std::vector<std::string>{"help"});
    } catch (const std::invalid_argument& e) {
        return usageError(e.what());
    }
    if (args->flag("help")) {
        std::fputs(kUsage, stdout);
        return 0;
    }

    try {
        const std::vector<int> sizes = parseSizes(args->get("sizes", "5,9,19"));
        const double seconds = args->getDouble("seconds", 2.0);
        if (seconds <= 0) throw std::invalid_argument("--seconds must be positive");
        const std::string configDir = args->get("configs", "configs");
        const std::string which = args->get("backend", azgo::blasAvailable() ? "all" : "naive");
        std::vector<std::string> backends;
        if (which == "all") {
            backends.push_back("naive");
            if (azgo::blasAvailable()) backends.push_back("blas");
        } else {
            azgo::makeBackend(which, 1);   // throws for an unknown or unavailable backend
            backends.push_back(which);
        }

        std::printf("%-6s %-17s %-8s %12s %14s\n", "size", "blocks x filters", "backend", "evals/s",
                    "train-step ms");
        for (int n : sizes) {
            const std::string preset =
                (std::filesystem::path(configDir) / (std::to_string(n) + "x" + std::to_string(n) + ".cfg")).string();
            if (!std::filesystem::exists(preset))
                throw std::runtime_error("no preset " + preset + " (use --configs DIR)");
            const azgo::NetConfig nc = azgo::loadConfigFile(preset).netConfig();
            if (nc.boardSize != n)
                throw std::runtime_error(preset + " has board_size " + std::to_string(nc.boardSize));
            for (const std::string& b : backends) {
                const double evals = measureInference(nc, b, seconds);
                const double stepMs = measureTrainStep(nc, b, seconds);
                char shape[32], size[16];
                std::snprintf(shape, sizeof shape, "%d x %d", nc.blocks, nc.filters);
                std::snprintf(size, sizeof size, "%dx%d", n, n);
                std::printf("%-6s %-17s %-8s %12.0f %14.1f\n", size, shape, b.c_str(), evals, stepMs);
                std::fflush(stdout);
            }
        }

        std::printf("\n%-6s %22s\n", "size", "random-playout moves/s");
        for (int n : sizes) {
            char size[16];
            std::snprintf(size, sizeof size, "%dx%d", n, n);
            std::printf("%-6s %22.0f\n", size, measurePlayouts(n, seconds));
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
