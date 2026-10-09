// bench: measures what the presets cost, so their sizes are chosen from
// numbers rather than guesses. For each board size's preset network and each
// backend: inference throughput at the MCTS batch size on one thread (what a
// self-play worker sees) and the time of one training step at batch 128 on
// all cores (what the training phase sees with train_threads = 0). With
// --train-threads, a second table times the training step at each of the
// given thread counts (batch --train-batch). With the metal backend (a
// build with -DAZ_USE_METAL=ON), three more tables: the GPU's raw forward
// pass at batches 8, 144 and 256, one at a time and with two passes in
// flight; the batching hub under N threads sending batch-8 requests, as
// self-play does; and, for reference, every core running its own CPU worker.
// Then raw rules-engine speed.
//
//   bench [--sizes 5,9,19] [--backend naive|blas|metal|all] [--seconds 2] [--configs configs]
//         [--train-threads 1,2,4,8] [--train-batch 256] [--hub-workers 18,36,72]
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
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
#include "azgo/inference_hub.hpp"
#include "azgo/metal.hpp"
#include "azgo/network.hpp"
#include "azgo/optimizer.hpp"
#include "azgo/parallel.hpp"
#include "azgo/rng.hpp"
#include "cli.hpp"

namespace {

const char* const kUsage =
    "usage: bench [--sizes 5,9,19] [--backend naive|blas|metal|all] [--seconds 2] [--configs DIR]\n"
    "             [--train-threads 1,2,4,8] [--train-batch 256] [--hub-workers 18,36,72]\n"
    "\n"
    "Network shapes come from DIR/<n>x<n>.cfg (default DIR: configs). --seconds is\n"
    "the minimum duration of each measurement. --train-threads adds a table of the\n"
    "training step's time at each thread count (0 = all cores), at batch\n"
    "--train-batch (default 256). all includes metal in a build with it.\n"
    "With metal, the main table's train-step column is '-' (training stays on the\n"
    "CPU), and three tables follow: raw GPU forward passes (one at a time, and two\n"
    "in flight), the batching hub under each --hub-workers thread count (each\n"
    "thread sending batch-8 requests: evals/s, mean pass size, slot occupancy),\n"
    "and every core running its own BLAS worker at batch 8 (the CPU reference).\n";

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

// "1,2,4": positive integers (thread counts; 0 = all cores).
std::vector<int> parseThreadList(const std::string& text) {
    std::vector<int> list;
    std::stringstream in(text);
    for (std::string item; std::getline(in, item, ',');) {
        char* end = nullptr;
        const long t = std::strtol(item.c_str(), &end, 10);
        if (item.empty() || *end != '\0' || t < 0 || t > 256)
            throw std::invalid_argument("--train-threads: '" + item + "' is not a thread count in 0..256");
        list.push_back(static_cast<int>(t));
    }
    if (list.empty()) throw std::invalid_argument("--train-threads is empty");
    return list;
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

// Calls fn until `seconds` have passed (after one warm-up call); returns
// calls per second.
template <class Fn>
double callsPerSecond(Fn fn, double seconds) {
    fn();   // warm-up
    long long calls = 0;
    const auto start = Clock::now();
    double elapsed = 0;
    do {
        fn();
        ++calls;
        elapsed = secondsSince(start);
    } while (elapsed < seconds);
    return static_cast<double>(calls) / elapsed;
}

#ifdef AZ_USE_METAL
// Positions per second through the GPU network, one synchronous pass of
// `batch` at a time (copy in, one command buffer, wait, copy out).
double measureGpuSync(const azgo::NetConfig& nc, int batch, double seconds) {
    const azgo::Network net(nc, 1);
    azgo::MetalInference gpu(net, batch);
    azgo::Rng rng(7);
    const std::vector<float> input = randomInput(nc, batch, rng);
    std::vector<float> logits(static_cast<size_t>(batch) * nc.actions()), values(static_cast<size_t>(batch));
    return batch * callsPerSecond([&] { gpu.infer(input.data(), batch, logits.data(), values.data()); },
                                  seconds);
}

// Positions per second with the hub's runner kept full: both slots always
// have a pass of `batch` in flight, each resubmitted as soon as it is done.
double measureGpuPipelined(const azgo::NetConfig& nc, int batch, double seconds) {
    const azgo::Network net(nc, 1);
    std::unique_ptr<azgo::BatchRunner> runner = azgo::makeMetalRunner(net, batch);
    azgo::Rng rng(7);
    const std::vector<float> input = randomInput(nc, batch, rng);
    for (int s = 0; s < runner->slots(); ++s)
        std::copy(input.begin(), input.end(), runner->input(s));
    std::mutex m;
    std::condition_variable cv;
    std::vector<int> ready;   // slots whose pass is done
    std::string error;
    auto done = [&](int slot, std::string err) {
        std::lock_guard<std::mutex> lock(m);
        if (!err.empty()) error = err;
        ready.push_back(slot);
        cv.notify_one();
    };
    for (int s = 0; s < runner->slots(); ++s) runner->begin(s, batch, done);
    long long passes = 0;
    const auto start = Clock::now();
    int inFlight = runner->slots();
    while (inFlight > 0) {
        int slot;
        {
            std::unique_lock<std::mutex> lock(m);
            cv.wait(lock, [&] { return !ready.empty(); });
            slot = ready.back();
            ready.pop_back();
            if (!error.empty()) throw std::runtime_error(error);
        }
        --inFlight;
        ++passes;
        if (secondsSince(start) < seconds) {
            runner->begin(slot, batch, done);
            ++inFlight;
        }
    }
    return static_cast<double>(passes) * batch / secondsSince(start);
}

struct HubLoad {
    double evalsPerSecond, meanBatch, occupancy;
};

// `workers` threads each sending batch-8 requests of random positions in a
// loop through one hub (inference_batch 256), as self-play's workers do.
HubLoad measureHub(const azgo::NetConfig& nc, int workers, double seconds) {
    const azgo::Network net(nc, 1);
    azgo::InferenceHubOptions options;
    options.maxBatch = 256;
    azgo::InferenceHub hub(azgo::makeMetalRunner(net, options.maxBatch), nc, options);
    std::atomic<bool> stop{false};
    std::atomic<long long> evals{0};
    std::vector<std::thread> threads;
    const auto start = Clock::now();
    for (int w = 0; w < workers; ++w) {
        threads.emplace_back([&, w] {
            std::unique_ptr<azgo::Evaluator> ev = hub.makeEvaluator(kInferBatch);
            azgo::Rng rng(static_cast<uint64_t>(100 + w));
            const std::vector<float> input = randomInput(nc, kInferBatch, rng);
            std::vector<float> logits(static_cast<size_t>(kInferBatch) * nc.actions()), values(kInferBatch);
            long long mine = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                ev->evaluate(input.data(), kInferBatch, logits.data(), values.data());
                mine += kInferBatch;
            }
            evals += mine;
        });
    }
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop = true;
    for (std::thread& t : threads) t.join();
    const double elapsed = secondsSince(start);
    const azgo::InferenceHub::Stats st = hub.stats();
    return {static_cast<double>(evals) / elapsed,
            st.batches > 0 ? static_cast<double>(st.positions) / static_cast<double>(st.batches) : 0.0,
            st.gpuSeconds / (elapsed * azgo::kInferenceSlots)};
}
#endif

// Every core running its own worker at batch kInferBatch with BLAS (or the
// naive backend without it), as self-play does on the CPU: positions per
// second, all workers together.
double measureCpuWorkers(const azgo::NetConfig& nc, double seconds) {
    const azgo::Network net(nc, 1);
    const int workers = azgo::resolveWorkers(0);
    const std::string backendName = azgo::blasAvailable() ? "blas" : "naive";
    std::atomic<bool> stop{false};
    std::atomic<long long> evals{0};
    std::vector<std::thread> threads;
    const auto start = Clock::now();
    for (int w = 0; w < workers; ++w) {
        threads.emplace_back([&, w] {
            std::unique_ptr<azgo::Backend> backend = azgo::makeBackend(backendName, 1);
            azgo::Workspace ws(nc, kInferBatch, *backend);
            azgo::Rng rng(static_cast<uint64_t>(200 + w));
            const std::vector<float> input = randomInput(nc, kInferBatch, rng);
            std::vector<float> logits(static_cast<size_t>(kInferBatch) * nc.actions()), values(kInferBatch);
            long long mine = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                net.infer(ws, input.data(), kInferBatch, logits.data(), values.data());
                mine += kInferBatch;
            }
            evals += mine;
        });
    }
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop = true;
    for (std::thread& t : threads) t.join();
    return static_cast<double>(evals) / secondsSince(start);
}

// Positions evaluated per second, batch kInferBatch, one thread.
double measureInference(const azgo::NetConfig& nc, const std::string& backendName, double seconds) {
#ifdef AZ_USE_METAL
    if (backendName == "metal") return measureGpuSync(nc, kInferBatch, seconds);
#endif
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

// Milliseconds per training step (forward, loss, backward and the SGD
// update) at the given batch, the network passes spread over `threads`
// threads (resolveTrainThreads: 0 = all cores, capped at them) as the
// trainer's train_threads does.
double measureTrainStep(const azgo::NetConfig& nc, const std::string& backendName, int batch, int threads,
                        double seconds) {
    std::unique_ptr<azgo::Backend> backend = azgo::makeBackend(backendName, 1);
    azgo::ThreadPool pool(azgo::resolveTrainThreads(threads));
    azgo::Network net(nc, 1);
    azgo::Workspace ws(nc, batch, *backend, &pool);
    azgo::SgdMomentum opt(net, 0.9f, 1e-4f);
    azgo::Rng rng(11);
    const std::vector<float> input = randomInput(nc, batch, rng);
    std::vector<float> pi(static_cast<size_t>(batch) * nc.actions(), 0.0f);
    std::vector<float> z(static_cast<size_t>(batch));
    for (int b = 0; b < batch; ++b) {
        pi[static_cast<size_t>(b) * nc.actions() + rng.below(nc.actions())] = 1.0f;   // one-hot target
        z[static_cast<size_t>(b)] = rng.below(2) ? 1.0f : -1.0f;
    }
    auto step = [&] {
        net.trainStep(ws, input.data(), pi.data(), z.data(), batch);
        opt.step(net, 0.001f, &pool);
    };

    step();   // warm-up
    long long steps = 0;
    const auto start = Clock::now();
    double elapsed = 0;
    do {
        step();
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
                                           std::vector<std::string>{"sizes", "backend", "seconds", "configs",
                                                                    "train-threads", "train-batch",
                                                                    "hub-workers"},
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
        const bool threadTable = args->has("train-threads");
        const std::vector<int> threadCounts = parseThreadList(args->get("train-threads", "0"));
        const int trainBatch = args->getInt("train-batch", 256);
        if (trainBatch < 1) throw std::invalid_argument("--train-batch must be positive");
        const std::string which = args->get("backend", azgo::blasAvailable() ? "all" : "naive");
        std::vector<std::string> backends;
        const std::vector<int> hubWorkers = parseThreadList(args->get("hub-workers", "18,36,72"));
        if (which == "all") {
            backends.push_back("naive");
            if (azgo::blasAvailable()) backends.push_back("blas");
            if (azgo::metalBuilt() && azgo::metalAvailable()) backends.push_back("metal");
        } else {
            azgo::checkBackendAvailable(which);   // throws for an unknown or unavailable backend
            backends.push_back(which);
        }
        bool metal = false;
        for (const std::string& b : backends) metal = metal || b == "metal";
        if (metal) std::printf("gpu: %s\n\n", azgo::metalDescription().c_str());

        std::printf("%-6s %-17s %-8s %12s %14s\n", "size", "blocks x filters", "backend", "evals/s",
                    "train-step ms");
        std::vector<azgo::NetConfig> shapes;
        for (int n : sizes) {
            const std::string preset =
                (std::filesystem::path(configDir) / (std::to_string(n) + "x" + std::to_string(n) + ".cfg")).string();
            if (!std::filesystem::exists(preset))
                throw std::runtime_error("no preset " + preset + " (use --configs DIR)");
            const azgo::NetConfig nc = azgo::loadConfigFile(preset).netConfig();
            if (nc.boardSize != n)
                throw std::runtime_error(preset + " has board_size " + std::to_string(nc.boardSize));
            shapes.push_back(nc);
            for (const std::string& b : backends) {
                const double evals = measureInference(nc, b, seconds);
                char shape[32], size[16], step[32];
                std::snprintf(shape, sizeof shape, "%d x %d", nc.blocks, nc.filters);
                std::snprintf(size, sizeof size, "%dx%d", n, n);
                // Training stays on the CPU with metal (BLAS): no GPU step to time.
                if (b == "metal") std::snprintf(step, sizeof step, "-");
                else std::snprintf(step, sizeof step, "%.1f", measureTrainStep(nc, b, kTrainBatch, 0, seconds));
                std::printf("%-6s %-17s %-8s %12.0f %14s\n", size, shape, b.c_str(), evals, step);
                std::fflush(stdout);
            }
        }

#ifdef AZ_USE_METAL
        if (metal) {
            std::printf("\n%-6s %-17s %6s %14s %16s %10s\n", "size", "blocks x filters", "batch",
                        "sync evals/s", "2-slot evals/s", "ms/pass");
            for (const azgo::NetConfig& nc : shapes) {
                for (int batch : {8, 144, 256}) {
                    const double sync = measureGpuSync(nc, batch, seconds);
                    const double piped = measureGpuPipelined(nc, batch, seconds);
                    char shape[32], size[16];
                    std::snprintf(shape, sizeof shape, "%d x %d", nc.blocks, nc.filters);
                    std::snprintf(size, sizeof size, "%dx%d", nc.boardSize, nc.boardSize);
                    std::printf("%-6s %-17s %6d %14.0f %16.0f %10.2f\n", size, shape, batch, sync, piped,
                                1000.0 * batch / sync);
                    std::fflush(stdout);
                }
            }
            std::printf("\n%-6s %-17s %-14s %8s %12s %10s %10s\n", "size", "blocks x filters", "evaluator",
                        "threads", "evals/s", "mean pass", "slots busy");
            for (const azgo::NetConfig& nc : shapes) {
                char shape[32], size[16];
                std::snprintf(shape, sizeof shape, "%d x %d", nc.blocks, nc.filters);
                std::snprintf(size, sizeof size, "%dx%d", nc.boardSize, nc.boardSize);
                const double cpu = measureCpuWorkers(nc, seconds);
                std::printf("%-6s %-17s %-14s %8d %12.0f %10s %10s\n", size, shape,
                            azgo::blasAvailable() ? "cpu (blas)" : "cpu (naive)", azgo::resolveWorkers(0), cpu,
                            "8", "-");
                std::fflush(stdout);
                for (int w : hubWorkers) {
                    const HubLoad h = measureHub(nc, azgo::resolveWorkers(w, true), seconds);
                    std::printf("%-6s %-17s %-14s %8d %12.0f %10.1f %9.0f%%\n", size, shape, "metal hub",
                                azgo::resolveWorkers(w, true), h.evalsPerSecond, h.meanBatch, 100.0 * h.occupancy);
                    std::fflush(stdout);
                }
            }
        }
#endif

        if (threadTable) {
            // Speedup against the first thread count listed.
            std::printf("\n%-6s %-17s %-8s %6s %8s %14s %8s\n", "size", "blocks x filters", "backend", "batch",
                        "threads", "train-step ms", "speedup");
            for (const azgo::NetConfig& nc : shapes) {
                for (const std::string& b : backends) {
                    if (b == "metal") continue;   // training stays on the CPU
                    double first = 0;
                    for (int t : threadCounts) {
                        const double stepMs = measureTrainStep(nc, b, trainBatch, t, seconds);
                        if (first == 0) first = stepMs;
                        char shape[32], size[16];
                        std::snprintf(shape, sizeof shape, "%d x %d", nc.blocks, nc.filters);
                        std::snprintf(size, sizeof size, "%dx%d", nc.boardSize, nc.boardSize);
                        std::printf("%-6s %-17s %-8s %6d %8d %14.1f %7.2fx\n", size, shape, b.c_str(), trainBatch,
                                    azgo::resolveTrainThreads(t), stepMs, first / stepMs);
                        std::fflush(stdout);
                    }
                }
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
