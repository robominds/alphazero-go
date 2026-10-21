// bench: measures what the presets cost, so their sizes are chosen from
// numbers rather than guesses. For each board size's preset network and each
// backend: inference throughput at the MCTS batch size on one thread (what a
// self-play worker sees) and the time of one training step at batch 128 on
// all cores (what the training phase sees with train_threads = 0). With
// --train-threads, a second table times the training step at each of the
// given thread counts (batch --train-batch). With a GPU backend (metal in a
// build with -DAZ_USE_METAL=ON, cuda with -DAZ_USE_CUDA=ON), three more
// tables: the GPU's raw forward
// pass at batches 8, 144 and 256, one at a time and with two passes in
// flight; the batching hub under N threads sending batch-8 requests, as
// self-play does; and, for reference, every core running its own CPU worker.
// Then raw rules-engine speed.
//
// --mixed measures what distributed self-play's engines on one machine
// share: the batching hub with N threads (a GPU engine) and k BLAS workers
// (a CPU engine) running at the same time, for each k given, and the CPU
// workers alone. That is the number engine weights need, since a CPU engine
// beside a busy hub is not the CPU engine alone.
//
//   bench [--sizes 5,9,19] [--backend naive|blas|metal|cuda|all] [--seconds 2] [--configs configs]
//         [--train-threads 1,2,4,8] [--train-batch 256] [--hub-workers 18,36,72]
//   bench --mixed [--sizes 19] [--hub-backend metal|cuda] [--hub-workers N] [--cpu-workers 0,2,4,6,8,12]
//         [--qos default|utility] [--seconds 10] [--net CKPT]
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <cstdlib>
#include <exception>
#include <functional>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "azgo/backend.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/config.hpp"
#include "azgo/cuda.hpp"
#include "azgo/game.hpp"
#include "azgo/inference_hub.hpp"
#include "azgo/metal.hpp"
#include "azgo/network.hpp"
#include "azgo/optimizer.hpp"
#include "azgo/parallel.hpp"
#include "azgo/rng.hpp"
#include "azgo/selfplay_slice.hpp"
#include "cli.hpp"

namespace {

const char* const kUsage =
    "usage: bench [--sizes 5,9,19] [--backend naive|blas|metal|cuda|all] [--seconds 2] [--configs DIR]\n"
    "             [--train-threads 1,2,4,8] [--train-batch 256] [--hub-workers 18,36,72]\n"
    "\n"
    "Network shapes come from DIR/<n>x<n>.cfg (default DIR: configs). --seconds is\n"
    "the minimum duration of each measurement. --train-threads adds a table of the\n"
    "training step's time at each thread count (0 = all cores), at batch\n"
    "--train-batch (default 256). all includes metal or cuda in a build with it.\n"
    "With metal or cuda, the main table's train-step column is '-' (training stays on the\n"
    "CPU), and three tables follow: raw GPU forward passes (one at a time, and two\n"
    "in flight), the batching hub under each --hub-workers thread count (each\n"
    "thread sending batch-8 requests: evals/s, mean pass size, slot occupancy),\n"
    "and every core running its own BLAS worker at batch 8 (the CPU reference).\n"
    "\n"
    "       bench --mixed [--sizes 19] [--hub-backend metal|cuda] [--hub-workers N]\n"
    "                     [--cpu-workers 0,2,4,6,8,12]\n"
    "                     [--qos default|utility] [--seconds 10] [--net CKPT]\n"
    "\n"
    "--mixed (a build with Metal or CUDA): for each k of --cpu-workers, the GPU hub\n"
    "(--hub-backend, default the one built: metal on a Mac, cuda on Linux) with N\n"
    "threads (default 0: twice the cores, the workers key's rule for backend = metal;\n"
    "a metal engine of selfplay_engines plays every game of its slice at once\n"
    "instead, so pass its slice's games to measure that) and k BLAS workers run\n"
    "at the same time, every thread sending batch-8 requests as self-play does; it\n"
    "prints both rates, their total, the hub's rate against k = 0 (the hub alone), the\n"
    "mean pass and slot occupancy. A second table gives the k BLAS workers alone.\n"
    "--qos sets the CPU workers' QoS class (macOS), as the hosts file's cpu_qos does.\n"
    "--net uses that checkpoint's network instead of a random one of the preset's\n"
    "shape (one board size only).\n";

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

// "1,2,4": thread counts in 0..256 (0 = all cores, or none for --cpu-workers),
// for option --<name>.
std::vector<int> parseThreadList(const std::string& text, const std::string& name) {
    std::vector<int> list;
    std::stringstream in(text);
    for (std::string item; std::getline(in, item, ',');) {
        char* end = nullptr;
        const long t = std::strtol(item.c_str(), &end, 10);
        if (item.empty() || *end != '\0' || t < 0 || t > 256)
            throw std::invalid_argument("--" + name + ": '" + item + "' is not a thread count in 0..256");
        list.push_back(static_cast<int>(t));
    }
    if (list.empty()) throw std::invalid_argument("--" + name + " is empty");
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

// Runs body(w) on `workers` threads until stop is set after `seconds`, or
// until one of them throws; rethrows the first exception after joining
// (an exception escaping a std::thread would end the program).
template <class Body>
double runWorkers(int workers, double seconds, std::atomic<bool>& stop, Body body) {
    std::mutex m;
    std::exception_ptr first;
    std::vector<std::thread> threads;
    const auto start = Clock::now();
    for (int w = 0; w < workers; ++w) {
        threads.emplace_back([&, w] {
            try {
                body(w);
            } catch (...) {
                std::lock_guard<std::mutex> lock(m);
                if (!first) first = std::current_exception();
                stop = true;
            }
        });
    }
    while (!stop.load() && secondsSince(start) < seconds)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    stop = true;
    for (std::thread& t : threads) t.join();
    if (first) std::rethrow_exception(first);
    return secondsSince(start);
}

#if defined(AZ_USE_METAL) || defined(AZ_USE_CUDA)
#define AZGO_BENCH_GPU 1

// The GPU backends this binary has: metal (macOS) or cuda (Linux).
bool gpuBackend(const std::string& b) { return b == "metal" || b == "cuda"; }

std::string gpuDescription(const std::string& gpu) {
    return gpu == "cuda" ? azgo::cudaDescription() : azgo::metalDescription();
}

// The hub's runner for `gpu`.
std::unique_ptr<azgo::BatchRunner> makeGpuRunner(const std::string& gpu, const azgo::Network& net, int maxBatch) {
#ifdef AZ_USE_CUDA
    if (gpu == "cuda") return azgo::makeCudaRunner(net, maxBatch);
#endif
#ifdef AZ_USE_METAL
    if (gpu == "metal") return azgo::makeMetalRunner(net, maxBatch);
#endif
    throw std::invalid_argument("backend '" + gpu + "' is not built in");
}

// Positions per second through the GPU network, one synchronous pass of
// `batch` at a time (copy in, the pass, wait, copy out).
double measureGpuSync(const std::string& gpu, const azgo::NetConfig& nc, int batch, double seconds) {
    const azgo::Network net(nc, 1);
    azgo::Rng rng(7);
    const std::vector<float> input = randomInput(nc, batch, rng);
    std::vector<float> logits(static_cast<size_t>(batch) * nc.actions()), values(static_cast<size_t>(batch));
#ifdef AZ_USE_CUDA
    if (gpu == "cuda") {
        azgo::CudaInference cuda(net, batch);
        return batch * callsPerSecond([&] { cuda.infer(input.data(), batch, logits.data(), values.data()); },
                                      seconds);
    }
#endif
#ifdef AZ_USE_METAL
    if (gpu == "metal") {
        azgo::MetalInference metal(net, batch);
        return batch * callsPerSecond([&] { metal.infer(input.data(), batch, logits.data(), values.data()); },
                                      seconds);
    }
#endif
    throw std::invalid_argument("backend '" + gpu + "' is not built in");
}

// Positions per second with the hub's runner kept full: both slots always
// have a pass of `batch` in flight, each resubmitted as soon as it is done.
double measureGpuPipelined(const std::string& gpu, const azgo::NetConfig& nc, int batch, double seconds) {
    const azgo::Network net(nc, 1);
    // The completion handler's state is declared before the runner, so it
    // outlives it: the runner's destructor waits for passes in flight, whose
    // handlers still use it.
    std::mutex m;
    std::condition_variable cv;
    std::vector<int> ready;   // slots whose pass is done
    std::string error;
    auto done = [&](int slot, std::string err) {
        std::lock_guard<std::mutex> lock(m);
        if (!err.empty() && error.empty()) error = err;
        ready.push_back(slot);
        cv.notify_one();
    };
    std::unique_ptr<azgo::BatchRunner> runner = makeGpuRunner(gpu, net, batch);
    azgo::Rng rng(7);
    const std::vector<float> input = randomInput(nc, batch, rng);
    for (int s = 0; s < runner->slots(); ++s)
        std::copy(input.begin(), input.end(), runner->input(s));
    int inFlight = 0;
    for (int s = 0; s < runner->slots(); ++s) {
        runner->begin(s, batch, done);
        ++inFlight;
    }
    long long passes = 0;
    const auto start = Clock::now();
    // After an error nothing is resubmitted; the passes in flight are
    // drained before it is reported.
    while (inFlight > 0) {
        int slot;
        bool failed;
        {
            std::unique_lock<std::mutex> lock(m);
            cv.wait(lock, [&] { return !ready.empty(); });
            slot = ready.back();
            ready.pop_back();
            failed = !error.empty();
        }
        --inFlight;
        ++passes;
        if (!failed && secondsSince(start) < seconds) {
            runner->begin(slot, batch, done);
            ++inFlight;
        }
    }
    if (!error.empty()) throw std::runtime_error(error);
    return static_cast<double>(passes) * batch / secondsSince(start);
}

struct HubLoad {
    double evalsPerSecond, meanBatch, occupancy;
};

// `workers` threads each sending batch-8 requests of random positions in a
// loop through one hub (inference_batch 256), as self-play's workers do.
HubLoad measureHub(const std::string& gpu, const azgo::NetConfig& nc, int workers, double seconds) {
    const azgo::Network net(nc, 1);
    azgo::InferenceHubOptions options;
    options.maxBatch = 256;
    azgo::InferenceHub hub(makeGpuRunner(gpu, net, options.maxBatch), nc, options);
    std::atomic<bool> stop{false};
    std::atomic<long long> evals{0};
    const double elapsed = runWorkers(workers, seconds, stop, [&](int w) {
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
    const azgo::InferenceHub::Stats st = hub.stats();
    return {static_cast<double>(evals) / elapsed,
            st.batches > 0 ? static_cast<double>(st.positions) / static_cast<double>(st.batches) : 0.0,
            st.gpuSeconds / (elapsed * azgo::kInferenceSlots)};
}
struct MixedLoad {
    double hubEvals, cpuEvals, meanBatch, occupancy;
};

// hubWorkers threads sending batch-8 requests through one hub
// (inference_batch 256) and cpuWorkers BLAS workers, each evaluating its own
// batches of 8, all at the same time: a GPU engine and a CPU engine of
// distributed self-play on one machine. The CPU workers take QoS class `qos`.
MixedLoad measureMixed(const std::string& gpu, const azgo::Network& net, int hubWorkers, int cpuWorkers,
                       azgo::CpuQos qos, double seconds) {
    const azgo::NetConfig& nc = net.config();
    azgo::InferenceHubOptions options;
    options.maxBatch = 256;
    azgo::InferenceHub hub(makeGpuRunner(gpu, net, options.maxBatch), nc, options);
    std::atomic<bool> stop{false};
    std::atomic<long long> hubEvals{0}, cpuEvals{0};
    const double elapsed = runWorkers(hubWorkers + cpuWorkers, seconds, stop, [&](int w) {
        azgo::Rng rng(static_cast<uint64_t>(300 + w));
        const std::vector<float> input = randomInput(nc, kInferBatch, rng);
        std::vector<float> logits(static_cast<size_t>(kInferBatch) * nc.actions()), values(kInferBatch);
        long long mine = 0;
        if (w < hubWorkers) {
            std::unique_ptr<azgo::Evaluator> ev = hub.makeEvaluator(kInferBatch);
            while (!stop.load(std::memory_order_relaxed)) {
                ev->evaluate(input.data(), kInferBatch, logits.data(), values.data());
                mine += kInferBatch;
            }
            hubEvals += mine;
        } else {
            azgo::setThreadQos(qos);
            std::unique_ptr<azgo::Backend> backend = azgo::makeBackend("blas", 1);
            azgo::Workspace ws(nc, kInferBatch, *backend);
            while (!stop.load(std::memory_order_relaxed)) {
                net.infer(ws, input.data(), kInferBatch, logits.data(), values.data());
                mine += kInferBatch;
            }
            cpuEvals += mine;
        }
    });
    const azgo::InferenceHub::Stats st = hub.stats();
    return {static_cast<double>(hubEvals) / elapsed, static_cast<double>(cpuEvals) / elapsed,
            st.batches > 0 ? static_cast<double>(st.positions) / static_cast<double>(st.batches) : 0.0,
            st.gpuSeconds / (elapsed * azgo::kInferenceSlots)};
}

// `workers` BLAS workers alone, each evaluating its own batches of 8, with
// QoS class `qos`: positions per second, all together.
double measureCpuWorkersAlone(const azgo::Network& net, int workers, azgo::CpuQos qos, double seconds) {
    const azgo::NetConfig& nc = net.config();
    std::atomic<bool> stop{false};
    std::atomic<long long> evals{0};
    const double elapsed = runWorkers(workers, seconds, stop, [&](int w) {
        azgo::setThreadQos(qos);
        std::unique_ptr<azgo::Backend> backend = azgo::makeBackend("blas", 1);
        azgo::Workspace ws(nc, kInferBatch, *backend);
        azgo::Rng rng(static_cast<uint64_t>(400 + w));
        const std::vector<float> input = randomInput(nc, kInferBatch, rng);
        std::vector<float> logits(static_cast<size_t>(kInferBatch) * nc.actions()), values(kInferBatch);
        long long mine = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            net.infer(ws, input.data(), kInferBatch, logits.data(), values.data());
            mine += kInferBatch;
        }
        evals += mine;
    });
    return static_cast<double>(evals) / elapsed;
}
#endif

// bench --mixed: the two tables for each board size (see the usage text).
int runMixed(const ArgParser& args) {
    // The GPU engine's backend: --hub-backend, else the one this binary has
    // (metal on a Mac, cuda on Linux; metal in a build with neither, which
    // the check below refuses with the not-built message).
    const std::string gpu = args.get("hub-backend", azgo::cudaBuilt() ? "cuda" : "metal");
    if (gpu != "metal" && gpu != "cuda")
        throw std::invalid_argument("--hub-backend: expected metal|cuda, got '" + gpu + "'");
    azgo::checkBackendAvailable(gpu);   // the not-built message without it
#ifdef AZGO_BENCH_GPU
    const std::vector<int> sizes = parseSizes(args.get("sizes", "19"));
    const double seconds = args.getDouble("seconds", 10.0);
    if (seconds <= 0) throw std::invalid_argument("--seconds must be positive");
    const std::string configDir = args.get("configs", "configs");
    const std::vector<int> hubList = parseThreadList(args.get("hub-workers", "0"), "hub-workers");
    if (hubList.size() != 1) throw std::invalid_argument("--mixed takes one --hub-workers count");
    const int hubWorkers = azgo::resolveWorkers(hubList[0], true);
    const std::vector<int> cpuList = parseThreadList(args.get("cpu-workers", "0,2,4,6,8,12"), "cpu-workers");
    const std::string qosName = args.get("qos", "default");
    if (qosName != "default" && qosName != "utility")
        throw std::invalid_argument("--qos: expected default|utility, got '" + qosName + "'");
    const azgo::CpuQos qos = qosName == "utility" ? azgo::CpuQos::Utility : azgo::CpuQos::Default;
    if (args.has("net") && sizes.size() != 1)
        throw std::invalid_argument("--net measures one network: give one --sizes");

    // The BLAS backend sets its environment once, with setenv, which is not
    // thread-safe: do that here, before any worker thread starts.
    azgo::makeBackend("blas", 1);
    std::printf("gpu: %s\n", gpuDescription(gpu).c_str());
    for (int n : sizes) {
        std::unique_ptr<azgo::Network> net;
        if (args.has("net")) {
            net = std::make_unique<azgo::Network>(azgo::loadNetwork(args.get("net")));
            if (net->config().boardSize != n)
                throw std::runtime_error(args.get("net") + " is not a " + std::to_string(n) + "x" +
                                         std::to_string(n) + " network");
        } else {
            const std::string preset = (std::filesystem::path(configDir) /
                                        (std::to_string(n) + "x" + std::to_string(n) + ".cfg")).string();
            if (!std::filesystem::exists(preset))
                throw std::runtime_error("no preset " + preset + " (use --configs DIR)");
            net = std::make_unique<azgo::Network>(azgo::loadConfigFile(preset).netConfig(), 1);
        }
        const azgo::NetConfig& nc = net->config();
        std::printf("\nmixed: %dx%d %d x %d, %s hub %d threads (inference_batch 256) + k blas "
                    "workers, batch 8, qos %s, %g s each\n",
                    n, n, nc.blocks, nc.filters, gpu.c_str(), hubWorkers, qosName.c_str(), seconds);
        std::printf("%-11s %12s %12s %14s %13s %10s %10s\n", "cpu workers", "hub evals/s", "cpu evals/s",
                    "total evals/s", "hub vs alone", "mean pass", "slots busy");
        double alone = 0;
        for (int k : cpuList) {
            const MixedLoad m = measureMixed(gpu, *net, hubWorkers, k, qos, seconds);
            if (k == 0) alone = m.hubEvals;   // the hub alone: the baseline
            char versus[16] = "-";
            if (alone > 0) std::snprintf(versus, sizeof versus, "%.1f%%", 100.0 * m.hubEvals / alone);
            std::printf("%11d %12.0f %12.0f %14.0f %13s %10.1f %9.0f%%\n", k, m.hubEvals, m.cpuEvals,
                        m.hubEvals + m.cpuEvals, versus, m.meanBatch, 100.0 * m.occupancy);
            std::fflush(stdout);
        }
        std::printf("\n%-17s %12s %11s\n", "cpu workers alone", "cpu evals/s", "per worker");
        for (int k : cpuList) {
            if (k == 0) continue;
            const double cpu = measureCpuWorkersAlone(*net, k, qos, seconds);
            std::printf("%17d %12.0f %11.0f\n", k, cpu, cpu / k);
            std::fflush(stdout);
        }
    }
    return 0;
#else
    (void)args;
    return 1;   // checkBackendAvailable threw
#endif
}

// Every core running its own worker at batch kInferBatch with BLAS (or the
// naive backend without it), as self-play does on the CPU: positions per
// second, all workers together.
double measureCpuWorkers(const azgo::NetConfig& nc, double seconds) {
    const azgo::Network net(nc, 1);
    const int workers = azgo::resolveWorkers(0);
    const std::string backendName = azgo::blasAvailable() ? "blas" : "naive";
    azgo::makeBackend(backendName, 1);   // BLAS's one-time setenv, before the threads
    std::atomic<bool> stop{false};
    std::atomic<long long> evals{0};
    const double elapsed = runWorkers(workers, seconds, stop, [&](int w) {
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
    return static_cast<double>(evals) / elapsed;
}

// Positions evaluated per second, batch kInferBatch, one thread.
double measureInference(const azgo::NetConfig& nc, const std::string& backendName, double seconds) {
#ifdef AZGO_BENCH_GPU
    if (gpuBackend(backendName)) return measureGpuSync(backendName, nc, kInferBatch, seconds);
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
                                                                    "hub-workers", "cpu-workers", "qos",
                                                                    "net", "hub-backend"},
                                           std::vector<std::string>{"help", "mixed"});
    } catch (const std::invalid_argument& e) {
        return usageError(e.what());
    }
    if (args->flag("help")) {
        std::fputs(kUsage, stdout);
        return 0;
    }

    try {
        if (args->flag("mixed")) return runMixed(*args);
        if (args->has("cpu-workers") || args->has("qos") || args->has("net") || args->has("hub-backend"))
            throw std::invalid_argument("--cpu-workers, --qos, --net and --hub-backend belong to --mixed");
        const std::vector<int> sizes = parseSizes(args->get("sizes", "5,9,19"));
        const double seconds = args->getDouble("seconds", 2.0);
        if (seconds <= 0) throw std::invalid_argument("--seconds must be positive");
        const std::string configDir = args->get("configs", "configs");
        const bool threadTable = args->has("train-threads");
        const std::vector<int> threadCounts = parseThreadList(args->get("train-threads", "0"), "train-threads");
        const int trainBatch = args->getInt("train-batch", 256);
        if (trainBatch < 1) throw std::invalid_argument("--train-batch must be positive");
        const std::string which = args->get("backend", azgo::blasAvailable() ? "all" : "naive");
        std::vector<std::string> backends;
        const std::vector<int> hubWorkers =
            parseThreadList(args->get("hub-workers", "18,36,72"), "hub-workers");
        if (which == "all") {
            backends.push_back("naive");
            if (azgo::blasAvailable()) backends.push_back("blas");
            if (azgo::metalBuilt() && azgo::metalAvailable()) backends.push_back("metal");
            if (azgo::cudaBuilt() && azgo::cudaAvailable()) backends.push_back("cuda");
        } else {
            azgo::checkBackendAvailable(which);   // throws for an unknown or unavailable backend
            backends.push_back(which);
        }
        std::vector<std::string> gpus;   // the GPU backends measured: their extra tables follow
        for (const std::string& b : backends)
            if (b == "metal" || b == "cuda") gpus.push_back(b);
        for (const std::string& g : gpus)
            std::printf("gpu: %s\n\n", g == "cuda" ? azgo::cudaDescription().c_str() : azgo::metalDescription().c_str());

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
                // Training stays on the CPU with metal and cuda: no GPU step to time.
                if (b == "metal" || b == "cuda") std::snprintf(step, sizeof step, "-");
                else std::snprintf(step, sizeof step, "%.1f", measureTrainStep(nc, b, kTrainBatch, 0, seconds));
                std::printf("%-6s %-17s %-8s %12.0f %14s\n", size, shape, b.c_str(), evals, step);
                std::fflush(stdout);
            }
        }

#ifdef AZGO_BENCH_GPU
        for (const std::string& gpu : gpus) {
            std::printf("\n%-6s %-17s %6s %14s %16s %10s   (%s)\n", "size", "blocks x filters", "batch",
                        "sync evals/s", "2-slot evals/s", "ms/pass", gpu.c_str());
            for (const azgo::NetConfig& nc : shapes) {
                for (int batch : {8, 144, 256}) {
                    const double sync = measureGpuSync(gpu, nc, batch, seconds);
                    const double piped = measureGpuPipelined(gpu, nc, batch, seconds);
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
                const std::string label = gpu + " hub";
                for (int w : hubWorkers) {
                    const HubLoad h = measureHub(gpu, nc, azgo::resolveWorkers(w, true), seconds);
                    std::printf("%-6s %-17s %-14s %8d %12.0f %10.1f %9.0f%%\n", size, shape, label.c_str(),
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
                    if (b == "metal" || b == "cuda") continue;   // training stays on the CPU
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
