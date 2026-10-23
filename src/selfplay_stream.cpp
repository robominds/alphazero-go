#include "azgo/selfplay_stream.hpp"
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <functional>
#include <set>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <map>
#include <mutex>
#include <poll.h>
#include <sstream>
#include <thread>
#include <unistd.h>
#include "azgo/backend.hpp"
#include "azgo/build_info.hpp"
#include "azgo/engines.hpp"
#include "azgo/parallel.hpp"
#include "azgo/serialize.hpp"
#include "azgo/subprocess.hpp"
#include "slice_files.hpp"

namespace azgo {

namespace {

const char kHello[4] = {'A', 'Z', 'S', 'H'};
const char kGame[4] = {'A', 'Z', 'S', 'G'};
const char kEnd[4] = {'A', 'Z', 'S', 'E'};

// magic, u32 length, payload, u64 FNV-1a of all of it.
std::string frame(const char magic[4], const std::string& payload) {
    std::ostringstream os(std::ios::binary);
    BinaryWriter w(os);
    w.bytes(magic, 4);
    w.u32(static_cast<uint32_t>(payload.size()));
    std::string bytes = os.str() + payload;
    slice_files::putChecksum(bytes);
    return bytes;
}

std::string payloadOf(const std::function<void(BinaryWriter&)>& write) {
    std::ostringstream os(std::ios::binary);
    BinaryWriter w(os);
    write(w);
    return os.str();
}

} // namespace

std::string streamHelloFrame(const StreamHello& hello) {
    return frame(kHello, payloadOf([&](BinaryWriter& w) {
                     w.u64(hello.jobHash);
                     w.str(hello.buildId);
                     w.str(hello.fingerprint);
                 }));
}

std::string streamGameFrame(int index, const GameRecord& record, double seconds, long long evaluations) {
    return frame(kGame, payloadOf([&](BinaryWriter& w) {
                     w.i32(index);
                     slice_files::writeSliceRecord(w, record);
                     w.f64(seconds);
                     w.u64(static_cast<uint64_t>(evaluations));
                 }));
}

std::string streamEndFrame(const SliceStats& stats) {
    return frame(kEnd, payloadOf([&](BinaryWriter& w) {
                     w.f64(stats.seconds);
                     w.u64(static_cast<uint64_t>(stats.positions));
                     w.u64(static_cast<uint64_t>(stats.evaluations));
                 }));
}

bool StreamFrames::next(char& kind, std::string& payload) {
    if (buffer_.size() < 8) return false;
    if (buffer_.compare(0, 4, kHello, 4) == 0) kind = 'H';
    else if (buffer_.compare(0, 4, kGame, 4) == 0) kind = 'G';
    else if (buffer_.compare(0, 4, kEnd, 4) == 0) kind = 'E';
    else throw SliceError("stream: not a frame (wrong magic)");
    const uint64_t length = slice_files::storedUint(buffer_, 4, 4);
    if (length > kMaxStreamFrame)
        throw SliceError("stream: a frame of " + std::to_string(length) + " bytes (at most " +
                         std::to_string(kMaxStreamFrame) + ")");
    const size_t whole = 8 + static_cast<size_t>(length) + 8;
    if (buffer_.size() < whole) return false;
    const std::string bytes = buffer_.substr(0, whole);
    if (slice_files::storedChecksum(bytes) != fnv1a64(bytes.substr(0, whole - 8)))
        throw SliceError("stream: checksum mismatch (damaged in transit)");
    payload = bytes.substr(8, static_cast<size_t>(length));
    buffer_.erase(0, whole);
    return true;
}

StreamChecker::StreamChecker(const std::string& jobBytes, const std::string& expectedBuild)
    : jobBytes_(jobBytes), expectedBuild_(expectedBuild), job_(parseJob(jobBytes)) {}

namespace {

// Reads a payload, turning truncation into a SliceError and refusing
// trailing bytes.
template <typename F>
auto readPayload(const std::string& payload, const std::string& what, F read) {
    std::istringstream is(payload, std::ios::binary);
    BinaryReader r(is);
    try {
        auto value = read(r);
        if (is.peek() != std::char_traits<char>::eof()) throw SliceError("trailing bytes");
        return value;
    } catch (const std::runtime_error& e) {   // SliceError included
        throw SliceError("stream " + what + ": " + e.what());
    }
}

} // namespace

StreamHello StreamChecker::hello(const std::string& payload) {
    if (helloSeen_) throw SliceError("stream: a second hello");
    StreamHello h = readPayload(payload, "hello", [](BinaryReader& r) {
        StreamHello out;
        out.jobHash = r.u64();
        out.buildId = r.str();
        out.fingerprint = r.str();
        return out;
    });
    if (h.jobHash != fnv1a64(jobBytes_))
        throw SliceError("stream hello: answers another job (job hash differs)");
    if (h.buildId != expectedBuild_)
        throw SliceError("stream hello: built from another commit (build ID " + h.buildId + ", expected " +
                         expectedBuild_ + ")");
    if (!slice_files::printableFingerprint(h.fingerprint))
        throw SliceError("stream hello: bad engine fingerprint (empty, too long or not printable)");
    helloSeen_ = true;
    return h;
}

StreamGame StreamChecker::game(const std::string& payload) {
    if (!helloSeen_) throw SliceError("stream: a game before the hello");
    StreamGame g = readPayload(payload, "game", [&](BinaryReader& r) {
        StreamGame out;
        out.index = r.i32();
        if (out.index < job_.first || out.index >= job_.first + job_.count)
            throw SliceError("index " + std::to_string(out.index) + " is outside the job's games [" +
                             std::to_string(job_.first) + ", " + std::to_string(job_.first + job_.count) +
                             ")");
        if (returned_.count(out.index))
            throw SliceError("game " + std::to_string(out.index) + " returned twice");
        out.record = slice_files::readSliceRecord(r, "game " + std::to_string(out.index));
        out.seconds = r.f64();
        out.evaluations = static_cast<long long>(r.u64());
        return out;
    });
    try {
        verifySelfPlayRecord(job_, g.record, g.index);
    } catch (const SliceError& e) {
        throw SliceError(std::string("stream game: ") + e.what());
    }
    returned_.insert(g.index);
    positions_ += g.record.positions();
    return g;
}

SliceStats StreamChecker::end(const std::string& payload) {
    if (!helloSeen_) throw SliceError("stream: the end before the hello");
    SliceStats stats = readPayload(payload, "end", [](BinaryReader& r) {
        SliceStats out;
        out.seconds = r.f64();
        out.positions = static_cast<long long>(r.u64());
        out.evaluations = static_cast<long long>(r.u64());
        return out;
    });
    if (stats.positions != positions_)
        throw SliceError("stream end: the statistics count " + std::to_string(stats.positions) +
                         " positions, the games held " + std::to_string(positions_));
    return stats;
}

// ---- the selfplay app's side --------------------------------------------------

namespace {

// The games the commands on inFd ask for, for playSelfPlayQueue; each
// finished game written to outFd as a frame.
class CommandSource : public GameSource {
public:
    CommandSource(const SelfPlayJob& job, int inFd, int outFd) : job_(job), in_(inFd), out_(outFd) {}

    // The reader thread: commands until `end`, end of file or a bad line.
    void readCommands() {
        std::string line;
        char buf[4096];
        while (!quit_) {
            pollfd p{in_, POLLIN, 0};
            const int ready = poll(&p, 1, 100);
            if (ready < 0 && errno == EINTR) continue;
            if (ready == 0) continue;
            const ssize_t got = ready < 0 ? -1 : read(in_, buf, sizeof buf);
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) {
                fail("stdin closed before `end` (the coordinator is gone); stopping");
                return;
            }
            for (ssize_t k = 0; k < got; ++k) {
                if (buf[k] != '\n') {
                    line += buf[k];
                    if (line.size() > 64) return fail("a command line longer than 64 characters");
                    continue;
                }
                if (!command(line)) return;
                line.clear();
            }
        }
    }

    int next() override {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [&] { return !ready_.empty() || ended_ || broken_ || stopped_; });
        if (broken_ || stopped_ || ready_.empty()) return -1;
        const int index = ready_.front();
        ready_.pop_front();
        started_[index] = std::chrono::steady_clock::now();
        return index;
    }

    void done(int index, GameRecord record, long long evaluations) override {
        double seconds = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_[index]).count();
            started_.erase(index);
        }
        const std::string bytes = streamGameFrame(index, record, seconds, evaluations);
        std::lock_guard<std::mutex> lock(writeMutex_);
        if (!writeAll(out_, bytes)) throw SliceError("cannot write a game to stdout (the coordinator is gone)");
    }

    void stop() override {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        wake_.notify_all();
    }

    void quit() { quit_ = true; }
    const std::atomic<bool>& abort() const { return abort_; }
    bool broken(std::string& why) {
        std::lock_guard<std::mutex> lock(mutex_);
        why = why_;
        return broken_;
    }

private:
    // One command line; false when reading should end.
    bool command(const std::string& line) {
        if (line == "end") {
            std::lock_guard<std::mutex> lock(mutex_);
            ended_ = true;
            wake_.notify_all();
            return false;
        }
        int index = -1;
        char extra = 0;
        if (std::sscanf(line.c_str(), "play %d%c", &index, &extra) != 1 || line.rfind("play ", 0) != 0) {
            fail("a bad command: '" + line + "'");
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < job_.first || index >= job_.first + job_.count || given_.count(index)) {
            failLocked("game " + std::to_string(index) + " is outside the job or was asked for twice");
            return false;
        }
        given_.insert(index);
        ready_.push_back(index);
        wake_.notify_all();
        return true;
    }
    void fail(const std::string& why) {
        std::lock_guard<std::mutex> lock(mutex_);
        failLocked(why);
    }
    void failLocked(const std::string& why) {
        if (!broken_) why_ = why;
        broken_ = true;
        abort_ = true;   // games in flight stop at their next evaluation
        wake_.notify_all();
    }

    const SelfPlayJob& job_;
    const int in_, out_;
    std::mutex mutex_, writeMutex_;
    std::condition_variable wake_;
    std::deque<int> ready_;
    std::set<int> given_;
    std::map<int, std::chrono::steady_clock::time_point> started_;
    bool ended_ = false, broken_ = false, stopped_ = false;
    std::string why_;
    std::atomic<bool> quit_{false}, abort_{false};
};

} // namespace

int runStreamJob(const std::string& jobPath, const std::string& backend, int workers, int inferenceBatch,
                 int inFd, int outFd, std::ostream& err) {
    try {
        const slice_files::LoadedSliceJob loaded = slice_files::loadSliceJob(jobPath);
        checkBackendAvailable(backend);
        const SelfPlayJob& job = loaded.job;
        StreamHello hello;
        hello.jobHash = fnv1a64(loaded.bytes);
        hello.buildId = buildId();
        hello.fingerprint = engineFingerprint(backend);
        if (!writeAll(outFd, streamHelloFrame(hello)))
            throw SliceError("cannot write to stdout (the coordinator is gone)");

        CommandSource source(job, inFd, outFd);
        std::thread reader([&] { source.readCommands(); });
        EngineRun run;
        run.backend = backend;
        run.workers = sliceWorkers(backend, workers, std::max(1, job.count), resolveWorkers(0));
        run.inferenceBatch = inferenceBatch > 0 ? inferenceBatch : job.inferenceBatch;
        run.cancel = &source.abort();
        SliceStats stats;
        std::exception_ptr error;
        try {
            playSelfPlayQueue(*loaded.net, job.sp, job.seed, job.generation, run, source, &stats);
        } catch (...) {
            error = std::current_exception();
        }
        source.quit();
        reader.join();
        std::string why;
        if (source.broken(why)) throw SliceError(why);
        if (error) std::rethrow_exception(error);
        if (!writeAll(outFd, streamEndFrame(stats)))
            throw SliceError("cannot write to stdout (the coordinator is gone)");
        return 0;
    } catch (const std::exception& e) {
        err << "selfplay: " << e.what() << "\n";
        return 1;
    }
}

} // namespace azgo
