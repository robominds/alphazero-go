// Self-play as a slice of game indices. A generation's self-play is G
// independent games: game i is seeded by deriveSeed(seed, g, kPhaseSelfPlay,
// i) and stored at index i, so which thread, which process or which machine
// plays it changes nothing but who computed it. playSelfPlayGames plays the
// games [first, first + count) of a generation with one engine (a backend
// and a pool of workers); the trainer's default path calls it once for
// [0, G), and distributed self-play calls it once per engine, each slice
// then merged back by index.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "azgo/network.hpp"
#include "azgo/selfplay.hpp"

namespace azgo {

// The scheduling class of a CPU engine's worker threads. Utility asks macOS
// to serve interactive work (the window server, the terminal) first, which
// keeps the desktop usable while CPU self-play runs beside a GPU engine. It
// changes when a thread runs, never what it computes. Other systems ignore it.
enum class CpuQos { Default, Utility };
// Gives the calling thread that class (macOS; elsewhere nothing). Default
// leaves the thread as it is.
void setThreadQos(CpuQos qos);

// How one engine plays its slice: the backend (naive, blas or metal), its
// worker threads, the most positions per GPU pass (metal), and the worker
// threads' QoS class. None of these change a single bit of the games.
struct EngineRun {
    std::string backend;
    int workers = 1;
    int inferenceBatch = 256;
    CpuQos qos = CpuQos::Default;
};

// What a slice cost: wall seconds, positions recorded, and network
// evaluations (positions the evaluators evaluated, all workers).
struct SliceStats {
    double seconds = 0;
    long long positions = 0, evaluations = 0;
};

// Games [first, first + count) of generation `generation`: game i is played
// with Rng(deriveSeed(seed, generation, kPhaseSelfPlay, i)) by one of
// run.workers threads (capped at count), each with its own evaluator from a
// WorkerEvaluators of run.backend (azgo/inference_hub.hpp), and stored at
// [i - first]. The result depends only on the network, sp, seed,
// generation and the indices, and on the arithmetic of the backend on this
// machine; never on workers, inferenceBatch or qos. Throws what
// WorkerEvaluators or a game throws (the first, after every worker stopped).
std::vector<GameRecord> playSelfPlayGames(const Network& net, const SelfPlayConfig& sp,
                                          uint64_t seed, int generation, int first, int count,
                                          const EngineRun& run, SliceStats* stats = nullptr);

} // namespace azgo
