// A slice of an arena wave played in another process: the selfplay app, on
// this machine or a remote one (arena_engines, azgo/coordinator.hpp). As
// with self-play slices, a job file says exactly what to play and a result
// file answers it, both binary and exact to the bit, and readArenaResult
// verifies an answer before any of it can reach the ratings or the gate:
// checksum, the job it answers, the build that played it, the number of
// games, and a replay of every game that recomputes its score.
//
// An arena job is a list of game ranges (azgo/arena.hpp): games [first,
// first + count) of a match, with the match's players and seed. Game j of a
// match is seeded by the match's seed and j, and A is Black when j is even,
// so the remote plays exactly the games the coordinator would have.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>
#include "azgo/arena.hpp"
#include "azgo/selfplay_slice.hpp"

namespace azgo {

// One seat of a range in a job: a network, by its index in the job's
// networks, or a bot (network = -1), with ArenaPlayer's overrides.
struct ArenaJobPlayer {
    int network = -1;
    int simulations = 0;
    bool randomSymmetry = false;
    float cPuct = 0;
    int mctsBatch = 0;
};

struct ArenaJobRange {
    ArenaJobPlayer a, b;
    uint64_t seed = 0;
    int first = 0, count = 0;
};

// A network of a job: a file name in the job's directory and its fileHash.
struct ArenaJobNetwork {
    std::string file;
    uint64_t hash = 0;
};

// Everything that determines a slice's arena games: the board, the arena
// settings (maxMoves resolved; games is not used), the networks the ranges
// name, the ranges, and the build that wrote the job. inferenceBatch is a
// speed hint.
struct ArenaJob {
    int boardSize = 0;
    ArenaConfig cfg;
    std::vector<ArenaJobNetwork> networks;
    std::vector<ArenaJobRange> ranges;
    std::string buildId;
    int inferenceBatch = 256;
    int games() const;   // the ranges' counts summed
};

// Little-endian, magic "AZAJ", version 1: i32 boardSize; i32 simulations,
// i32 mctsBatch, f32 cPuct, i32 temperatureMoves, f32 komi, i32 maxMoves,
// u8 randomSymmetry; u32 networks, each str file and u64 hash; u32 ranges,
// each two players (i32 network, i32 simulations, u8 randomSymmetry, f32
// cPuct, i32 mctsBatch), u64 seed, i32 first, i32 count; str buildId, i32
// inferenceBatch; then a u64 FNV-1a checksum of every byte before it.
std::string serializeArenaJob(const ArenaJob& job);
// Refuses (SliceError, naming the defect) a wrong magic or version, a bad
// checksum, a truncated or overlong file, and values no job can have: a
// board size, search setting or move cap out of range, a network file name
// with a path, a range naming a network the job does not have or no network
// at all, a negative or too large range, no build ID.
ArenaJob parseArenaJob(const std::string& bytes);
// Whether a job file's bytes are an arena job (its magic), not self-play's.
bool isArenaJob(const std::string& bytes);

// A slice's answer: the job it answers (its hash), the build and engine
// fingerprint that played it, its games in the job's order (range by range,
// game by game), and the statistics (positions: the plies played).
struct ArenaSliceResult {
    uint64_t jobHash = 0;
    std::string buildId, fingerprint;
    std::vector<ArenaGameResult> games;
    SliceStats stats;
};

// Magic "AZAR", version 1, u64 the file's length; u64 jobHash, str buildId,
// str fingerprint, u32 games; per game u32 n, n x i32 moves, f64 scoreA;
// f64 seconds, u64 positions, u64 evaluations; a u64 FNV-1a checksum of
// every byte before it.
std::string serializeArenaResult(const ArenaSliceResult& result);
// Reads and verifies the result at `path` for the job whose file bytes are
// jobBytes, refusing with SliceError (naming the defect): a missing file; a
// wrong magic, version, length or checksum; an answer to another job (job
// hash) or from another build than expectedBuild; a bad fingerprint; other
// than the job's number of games; and any game that does not replay on the
// job's board, komi and move cap: a move off the board or illegal where it
// was played, a move after the game had ended, a game that does not end at
// its last move, or a score that is not the replayed board's result for A
// (Black in the even games of its match); statistics that do not count the
// moves; trailing bytes.
ArenaSliceResult readArenaResult(const std::string& path, const std::string& jobBytes,
                                 const std::string& expectedBuild);

// What the selfplay app does with an arena job: reads and checks the job at
// jobPath (its build ID must be this binary's), loads each network from the
// job's directory after checking its fileHash, plays the ranges with
// playArenaGames on `backend` with sliceWorkers(backend, workers, ...)
// threads (workers <= 0: a remote's default rule, azgo/engines.hpp) and
// inferenceBatch (<= 0: the job's), and writes the result to outPath
// atomically. Returns the exit status: 0, or 1 with the reason on `err` and
// no file at outPath.
int runArenaJob(const std::string& jobPath, const std::string& outPath, const std::string& backend,
                int workers, int inferenceBatch, std::ostream& err);

// The selfplay app's work: runArenaJob for an arena job (its magic), else
// runSliceJob. The same arguments and exit status.
int runJob(const std::string& jobPath, const std::string& outPath, const std::string& backend,
           int workers, int inferenceBatch, std::ostream& err);

} // namespace azgo
