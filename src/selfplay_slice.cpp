#include "azgo/selfplay_slice.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include "azgo/inference_hub.hpp"
#include "azgo/parallel.hpp"
#include "azgo/rng.hpp"
#include "azgo/run_state.hpp"
#ifdef __APPLE__
#include <pthread.h>
#include <sys/qos.h>
#endif

namespace azgo {

namespace {

// Lowers the calling thread's QoS class (macOS); elsewhere nothing.
void applyQos(CpuQos qos) {
#ifdef __APPLE__
    if (qos == CpuQos::Utility) pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#else
    (void)qos;
#endif
}

} // namespace

std::vector<GameRecord> playSelfPlayGames(const Network& net, const SelfPlayConfig& sp, uint64_t seed,
                                          int generation, int first, int count, const EngineRun& run,
                                          SliceStats* stats) {
    if (first < 0 || count < 0) throw std::invalid_argument("playSelfPlayGames: negative index range");
    const auto start = std::chrono::steady_clock::now();
    std::vector<GameRecord> records(static_cast<size_t>(count));
    long long evaluations = 0;
    if (count > 0) {
        // One evaluator per worker; the network itself is shared read-only.
        // naive, blas: a single-threaded backend and workspace per worker.
        // metal: one batching hub that gathers every worker's leaves into
        // GPU passes; a position's outputs do not depend on its batch, so
        // the games are the same either way (azgo/inference_hub.hpp).
        const int workers = std::max(1, std::min(run.workers, count));
        WorkerEvaluators evaluators(net, run.backend, workers, sp.mcts.batchSize, run.inferenceBatch);
        // Each worker is one thread for the whole loop: it sets its QoS
        // class once, before its first game.
        std::vector<char> qosSet(static_cast<size_t>(workers), 0);
        // Game i's randomness depends only on i, and its record is stored at
        // index i, so the result does not depend on which thread played it.
        parallelFor(count, workers, [&](int k, int w) {
            if (!qosSet[static_cast<size_t>(w)]) {
                applyQos(run.qos);
                qosSet[static_cast<size_t>(w)] = 1;
            }
            const int i = first + k;
            Rng rng(deriveSeed(seed, static_cast<uint64_t>(generation), kPhaseSelfPlay,
                               static_cast<uint64_t>(i)));
            records[static_cast<size_t>(k)] = playSelfPlayGame(evaluators.forWorker(w), sp, rng);
        });
        evaluations = evaluators.evaluations();
    }
    if (stats) {
        stats->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        stats->positions = 0;
        for (const GameRecord& r : records) stats->positions += r.positions();
        stats->evaluations = evaluations;
    }
    return records;
}

} // namespace azgo
