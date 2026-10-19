// Self-play as a slice of game indices. A generation's self-play is G
// independent games: game i is seeded by deriveSeed(seed, g, kPhaseSelfPlay,
// i) and stored at index i, so which thread, which process or which machine
// plays it changes nothing but who computed it. playSelfPlayGames plays the
// games [first, first + count) of a generation with one engine (a backend
// and a pool of workers); the trainer's default path calls it once for
// [0, G), and distributed self-play calls it once per engine, each slice
// then merged back by index.
//
// A slice played in another process (the selfplay app, possibly on another
// machine) is described by a job file and answered by a result file, both
// binary and exact to the bit. readSliceResult verifies an answer before
// any of it can reach the replay buffer: checksum, the job it answers, the
// build that played it, the game indices, and a replay of every game.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <atomic>
#include <cstdint>
#include <ostream>
#include <stdexcept>
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
// cancel, when not null, is watched at every network evaluation: once it is
// set (another engine of the generation failed, so the generation will not
// be committed) the slice stops with SliceCancelled.
struct EngineRun {
    std::string backend;
    int workers = 1;
    int inferenceBatch = 256;
    CpuQos qos = CpuQos::Default;
    const std::atomic<bool>* cancel = nullptr;
};

// What a slice throws when its EngineRun's cancel flag was set.
struct SliceCancelled : std::runtime_error {
    SliceCancelled() : std::runtime_error("self-play slice cancelled") {}
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
struct SliceError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Everything that determines a slice's games, as exact bits: games
// [first, first + count) of `generation` of the run seeded `seed`, on a
// boardSize board, with `sp` (maxMoves already resolved, resignThreshold
// this generation's v_resign), from the network in networkFile (a file name
// in the remote work directory) whose fileHash is networkHash, by a binary
// built from buildId. inferenceBatch is a speed hint.
struct SelfPlayJob {
    uint64_t seed = 0;
    int generation = 0, first = 0, count = 0, boardSize = 0;
    SelfPlayConfig sp;
    std::string networkFile;
    uint64_t networkHash = 0;
    std::string buildId;
    int inferenceBatch = 256;
};

// Little-endian, magic "AZSJ", version 1: u64 seed; i32 generation, first,
// count, boardSize; MctsConfig (i32 simulations, i32 batchSize, f32 cPuct,
// u8 addNoise, f32 dirichletScale, f32 noiseFraction, u8 randomSymmetry);
// i32 temperatureMoves, f32 komi, i32 maxMoves, u8 resignation, f32
// resignThreshold, f32 resignDisabledFraction; str networkFile, u64
// networkHash, str buildId, i32 inferenceBatch; then a u64 FNV-1a checksum
// of every byte before it.
std::string serializeJob(const SelfPlayJob& job);
// Refuses (SliceError, naming the defect) a wrong magic or version, a bad
// checksum, a truncated or overlong file, and values no job can have.
SelfPlayJob parseJob(const std::string& bytes);
// 64-bit FNV-1a of the bytes: fileHash's function, in memory. The job hash
// is fnv1a64 of a job file's bytes.
uint64_t fnv1a64(const std::string& bytes);

// A slice's answer: the job it answers (its hash), the build and engine
// fingerprint that played it, the games in index order, and the statistics.
struct SliceResult {
    uint64_t jobHash = 0;
    std::string buildId, fingerprint;
    std::vector<GameRecord> records;
    SliceStats stats;
};

// Magic "AZSR", version 1, u64 the file's length; u64 jobHash, str buildId,
// str fingerprint, u32 count; per record i32 index (first, first + 1, ...),
// writeGameRecord's layout, u8 resignDisabled and per side u32 n and n x
// (i32 ply, f32 value) of resign lows; f64 seconds, u64 positions, u64
// evaluations; a u64 FNV-1a checksum of every byte before it.
std::string serializeSliceResult(const SliceResult& result, int first);
// Reads and verifies the result at `path` for the job whose file bytes are
// jobBytes, refusing with SliceError (naming the defect): a missing file; a
// wrong magic, version, length or checksum; an answer to another job (job
// hash) or from another build than expectedBuild; other than the job's
// count of games, or indices out of order; and any game that does not
// replay: another board size or komi, a position that differs from the
// replayed one, a move that is illegal or was not among the searched
// moves, a visit to an illegal move, a game that ends before its last move
// or does not end, a resign-disabled flag that is not the game's first draw
// from its seed, a played-out
// game whose result is not the replayed board's, a resigned game that is not
// a loss for the side that resigned, resignation flags or lows with
// resignation off, resign lows that are not record lows, a game that went
// below v_resign without resigning (unless resignation was disabled for it)
// or resigned without doing so, and statistics that do not count the
// positions.
SliceResult readSliceResult(const std::string& path, const std::string& jobBytes,
                            const std::string& expectedBuild);

// What the selfplay app does with a job: reads and checks the job at
// jobPath (its build ID must be this binary's), loads the network
// job.networkFile from the job's directory after checking its fileHash, plays
// the slice with `backend` on sliceWorkers(backend, workers, ...) threads
// (workers <= 0: a remote's default rule, azgo/engines.hpp) and
// inferenceBatch (<= 0: the
// job's), and writes the result to outPath atomically. Returns the exit
// status: 0, or 1 with the reason on `err` and no file at outPath.
int runSliceJob(const std::string& jobPath, const std::string& outPath, const std::string& backend,
                int workers, int inferenceBatch, std::ostream& err);

std::vector<GameRecord> playSelfPlayGames(const Network& net, const SelfPlayConfig& sp,
                                          uint64_t seed, int generation, int first, int count,
                                          const EngineRun& run, SliceStats* stats = nullptr);

} // namespace azgo
