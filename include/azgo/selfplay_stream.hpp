// A self-play stream: how a remote engine plays its part of a generation
// under dynamic allocation (selfplay_allocation = dynamic). Instead of one
// process per fixed slice, the coordinator starts one `selfplay --stream`
// process per engine and generation and keeps it busy: it writes the index
// of each game to start on the process's stdin, and the process writes
// each finished game back on its stdout as soon as it ends. One hub keeps
// its games in flight, refilled as they end, so a GPU's batches stay full
// until the generation's queue is empty; a broken connection loses only
// the games in flight.
//
// The job is a self-play job file (azgo/selfplay_slice.hpp) whose range
// [first, first + count) is the set of indices the process may be asked for.
//
// Commands (coordinator -> stdin), one per line:
//   play <index>   start game <index> (in the job's range, never twice)
//   end            no more games: finish the ones in flight, then exit 0
// End of file before `end` means the coordinator or the connection is gone:
// the process stops at once and exits 1.
//
// Answers (stdout -> coordinator), frames:
//   magic[4] u32 length payload[length] u64 FNV-1a of every byte before it
//   AZSH hello (first): u64 job hash, str build ID, str engine fingerprint
//   AZSG game: i32 index, the result file's record layout, f64 seconds,
//        u64 evaluations (the game's own)
//   AZSE end (after `end`): f64 seconds, u64 positions, u64 evaluations
// The coordinator verifies every game as a result file's (the same replay).
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <cstdint>
#include <ostream>
#include <set>
#include <string>
#include "azgo/selfplay.hpp"
#include "azgo/selfplay_slice.hpp"

namespace azgo {

// The longest frame payload a reader accepts (a 19x19 game is under 1 MB).
constexpr uint32_t kMaxStreamFrame = 64u << 20;

struct StreamHello {
    uint64_t jobHash = 0;
    std::string buildId, fingerprint;
};

struct StreamGame {
    int index = 0;
    GameRecord record;
    double seconds = 0;
    long long evaluations = 0;
};

// Whole frames, checksum included.
std::string streamHelloFrame(const StreamHello& hello);
std::string streamGameFrame(int index, const GameRecord& record, double seconds, long long evaluations);
std::string streamEndFrame(const SliceStats& stats);

// Cuts a byte stream into frames. feed() appends bytes; next() returns the
// next whole frame's kind ('H', 'G' or 'E') and payload, or false when the
// bytes so far end inside a frame. SliceError for an unknown magic, a
// length above kMaxStreamFrame or a checksum that does not match.
class StreamFrames {
public:
    void feed(const char* data, size_t size) { buffer_.append(data, size); }
    bool next(char& kind, std::string& payload);
    size_t buffered() const { return buffer_.size(); }
private:
    std::string buffer_;
};

// The coordinator's side of one session: checks each frame against the
// job, in order. SliceError naming the defect:
//   hello: not first, or twice; another job (hash) or build; a fingerprint
//          that is empty, too long or not printable;
//   game:  before the hello; an index outside the job's range or already
//          returned; a record that does not verify (verifySelfPlayRecord);
//          trailing bytes;
//   end:   before the hello; positions other than the games' total.
class StreamChecker {
public:
    StreamChecker(const std::string& jobBytes, const std::string& expectedBuild);
    StreamHello hello(const std::string& payload);
    StreamGame game(const std::string& payload);
    SliceStats end(const std::string& payload);
    const SelfPlayJob& job() const { return job_; }
private:
    std::string jobBytes_, expectedBuild_;
    SelfPlayJob job_;
    bool helloSeen_ = false;
    std::set<int> returned_;
    long long positions_ = 0;
};

// What `selfplay --stream` does: loads and checks the job and its network
// as runSliceJob does, writes the hello to outFd, then plays the games the
// commands on inFd ask for on `workers` threads (at least 1) of `backend`
// (inferenceBatch <= 0: the job's), writing each game's frame as it ends.
// After `end` and the last game, the end frame; returns 0. Returns 1 with
// the reason on `err`, promptly, on a bad job, a bad command, end of file
// before `end`, or a failed write (the coordinator is gone). Never raises
// SIGPIPE.
int runStreamJob(const std::string& jobPath, const std::string& backend, int workers, int inferenceBatch,
                 int inFd, int outFd, std::ostream& err);

} // namespace azgo
