// Self-play streams (azgo/selfplay_stream.hpp): the frames a `selfplay
// --stream` process answers with, the checks the coordinator makes of
// them, and the process's side played in this process on pipes: commands
// in, a game per frame out, each game the one playSelfPlayGames gives;
// end of file or a bad command stops it.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include "azgo/build_info.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/coordinator.hpp"
#include "azgo/engines.hpp"
#include "azgo/network.hpp"
#include "azgo/run_state.hpp"
#include "azgo/selfplay_slice.hpp"
#include "azgo/selfplay_stream.hpp"
#include "azgo/serialize.hpp"
#include "azgo/subprocess.hpp"

using namespace azgo;

static bool sameRecord(const GameRecord& x, const GameRecord& y) {
    return x.boardSize == y.boardSize && std::memcmp(&x.komi, &y.komi, 4) == 0 && x.moves == y.moves &&
           x.stones == y.stones && x.visits == y.visits &&
           std::memcmp(&x.resultForBlack, &y.resultForBlack, 4) == 0 && x.finalOwnership == y.finalOwnership &&
           x.resigned == y.resigned && x.resignDisabled == y.resignDisabled &&
           x.resignLows[0] == y.resignLows[0] && x.resignLows[1] == y.resignLows[1];
}

static NetConfig streamNet() {
    NetConfig c;
    c.boardSize = 5;
    c.history = 2;
    c.blocks = 1;
    c.filters = 4;
    c.valueHidden = 8;
    return c;
}

// A stream job for games [0, 8) of generation 2, resignation on with a
// threshold the tiny network crosses, in its own directory with its network.
struct StreamWork {
    std::filesystem::path dir;
    std::string jobPath, jobBytes;
    SelfPlayJob job;
    explicit StreamWork(const std::string& name, int simulations = 8)
        : dir(std::filesystem::temp_directory_path() /
              ("azgo_test_stream_" + std::to_string(getpid()) + "_" + name)) {
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        const std::string netPath = (dir / "net_test.azn").string();
        Network net(streamNet(), 33);
        saveNetwork(net, netPath);
        job.seed = 11;
        job.generation = 2;
        job.first = 0;
        job.count = 8;
        job.boardSize = 5;
        job.sp.mcts.simulations = simulations;
        job.sp.mcts.batchSize = 4;
        job.sp.temperatureMoves = 4;
        job.sp.komi = 7.5f;
        job.sp.maxMoves = 50;
        job.sp.resignation = true;
        job.sp.resignThreshold = -0.0f;
        job.sp.resignDisabledFraction = 0.5f;
        job.networkFile = "net_test.azn";
        job.networkHash = fileHash(netPath);
        job.buildId = buildId();
        job.inferenceBatch = 64;
        jobBytes = serializeJob(job);
        jobPath = (dir / "g0002_stream_a1.job").string();
        std::ofstream(jobPath, std::ios::binary) << jobBytes;
    }
    Network net() const { return loadNetwork((dir / "net_test.azn").string()); }
    ~StreamWork() {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
};

template <typename F>
static std::string sliceError(F f) {
    try {
        f();
    } catch (const SliceError& e) {
        return e.what();
    }
    return "";
}

static bool contains(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

// Frames round trip, fed whole or a byte at a time.
static void test_frames_round_trip() {
    StreamWork work("frames");
    const std::vector<GameRecord> games =
        playSelfPlayGames(work.net(), work.job.sp, work.job.seed, work.job.generation, 0, 3, EngineRun{"naive", 1});
    StreamHello hello{fnv1a64(work.jobBytes), buildId(), "naive; test cpu"};
    SliceStats stats;
    stats.seconds = 1.5;
    for (const GameRecord& g : games) stats.positions += g.positions();
    stats.evaluations = 1234;
    std::string bytes = streamHelloFrame(hello);
    for (int i = 0; i < 3; ++i) bytes += streamGameFrame(i, games[static_cast<size_t>(i)], 0.25 * i, 100 + i);
    bytes += streamEndFrame(stats);
    assert(bytes.compare(0, 4, "AZSH") == 0);
    for (bool byteByByte : {false, true}) {
        StreamFrames frames;
        StreamChecker check(work.jobBytes, buildId());
        std::string kinds;
        auto drain = [&] {
            char kind;
            std::string payload;
            while (frames.next(kind, payload)) {
                kinds += kind;
                if (kind == 'H') {
                    const StreamHello h = check.hello(payload);
                    assert(h.jobHash == hello.jobHash && h.buildId == hello.buildId &&
                           h.fingerprint == hello.fingerprint);
                } else if (kind == 'G') {
                    const StreamGame g = check.game(payload);
                    assert(sameRecord(g.record, games[static_cast<size_t>(g.index)]));
                    assert(g.seconds == 0.25 * g.index && g.evaluations == 100 + g.index);
                } else {
                    const SliceStats s = check.end(payload);
                    assert(s.seconds == 1.5 && s.positions == stats.positions && s.evaluations == 1234);
                }
            }
        };
        if (byteByByte) {
            for (char c : bytes) {
                frames.feed(&c, 1);
                drain();
            }
        } else {
            frames.feed(bytes.data(), bytes.size());
            drain();
        }
        assert(kinds == "HGGGE" && frames.buffered() == 0);
    }
}

// Damage anywhere in a frame is caught: it throws or never completes.
static void test_frame_refusals() {
    StreamWork work("frame_refusals");
    const std::vector<GameRecord> games =
        playSelfPlayGames(work.net(), work.job.sp, work.job.seed, work.job.generation, 0, 1, EngineRun{"naive", 1});
    const std::string frame = streamGameFrame(0, games[0], 0.5, 10);
    int caught = 0;
    for (size_t at = 0; at < frame.size(); ++at) {
        std::string bad = frame;
        bad[at] = static_cast<char>(bad[at] ^ 0x5a);
        StreamFrames frames;
        frames.feed(bad.data(), bad.size());
        char kind;
        std::string payload;
        try {
            assert(!frames.next(kind, payload));   // a longer length: still incomplete
        } catch (const SliceError&) {
            ++caught;
        }
    }
    assert(caught > static_cast<int>(frame.size()) - 8);
    // An oversized length and an unknown magic.
    std::string big = frame;
    big[7] = static_cast<char>(0x7f);
    StreamFrames f1;
    f1.feed(big.data(), big.size());
    char kind;
    std::string payload;
    assert(contains(sliceError([&] { f1.next(kind, payload); }), "bytes (at most"));
    StreamFrames f2;
    f2.feed("AZXX\0\0\0\0", 8);
    assert(contains(sliceError([&] { f2.next(kind, payload); }), "wrong magic"));
}

// The coordinator's checks of a session's frames.
static void test_checker_refusals() {
    StreamWork work("checker");
    const std::vector<GameRecord> games =
        playSelfPlayGames(work.net(), work.job.sp, work.job.seed, work.job.generation, 0, 2, EngineRun{"naive", 1});
    auto payloadOf = [](const std::string& frame) { return frame.substr(8, frame.size() - 16); };
    const StreamHello good{fnv1a64(work.jobBytes), buildId(), "naive; test cpu"};
    const std::string hello = payloadOf(streamHelloFrame(good));
    const std::string game0 = payloadOf(streamGameFrame(0, games[0], 0, 1));
    {
        StreamChecker c(work.jobBytes, buildId());
        assert(contains(sliceError([&] { c.game(game0); }), "before the hello"));
        assert(contains(sliceError([&] { c.end(payloadOf(streamEndFrame(SliceStats{}))); }), "before the hello"));
        StreamHello other = good;
        other.jobHash ^= 1;
        assert(contains(sliceError([&] { c.hello(payloadOf(streamHelloFrame(other))); }), "another job"));
        other = good;
        other.buildId = "0000000000000000000000000000000000000000";
        assert(contains(sliceError([&] { c.hello(payloadOf(streamHelloFrame(other))); }), "another commit"));
        other = good;
        other.fingerprint = "";
        assert(contains(sliceError([&] { c.hello(payloadOf(streamHelloFrame(other))); }), "fingerprint"));
        assert(contains(sliceError([&] { c.hello(hello + "x"); }), "trailing bytes"));
        c.hello(hello);
        assert(contains(sliceError([&] { c.hello(hello); }), "second hello"));
        // Outside the job's range, twice, damaged, trailing bytes.
        assert(contains(sliceError([&] { c.game(payloadOf(streamGameFrame(8, games[0], 0, 1))); }), "outside"));
        assert(contains(sliceError([&] { c.game(payloadOf(streamGameFrame(-1, games[0], 0, 1))); }), "outside"));
        // A record that does not replay.
        GameRecord bad = games[1];
        bad.moves[0] = (bad.moves[0] + 1) % 25;
        assert(contains(sliceError([&] { c.game(payloadOf(streamGameFrame(1, bad, 0, 1))); }), "stream game: game 1"));
        assert(contains(sliceError([&] { c.game(game0 + "x"); }), "trailing bytes"));
        c.game(game0);
        assert(contains(sliceError([&] { c.game(game0); }), "returned twice"));
        SliceStats wrong;
        wrong.positions = games[0].positions() + 1;
        assert(contains(sliceError([&] { c.end(payloadOf(streamEndFrame(wrong))); }), "positions"));
        wrong.positions = games[0].positions();
        c.end(payloadOf(streamEndFrame(wrong)));
    }
}

// The process's side, on pipes in this process: commands go in, frames
// come out; each game is the one playSelfPlayGames gives.
struct InProcessStream {
    int toChild[2] = {-1, -1}, fromChild[2] = {-1, -1};
    std::thread thread;
    std::atomic<int> status{-100};
    std::ostringstream err;
    StreamFrames frames;
    InProcessStream(const std::string& jobPath, int workers) {
        assert(pipe(toChild) == 0 && pipe(fromChild) == 0);
        thread = std::thread([this, jobPath, workers] {
            status = runStreamJob(jobPath, "naive", workers, 0, toChild[0], fromChild[1], err);
            close(fromChild[1]);
            fromChild[1] = -1;
        });
    }
    bool send(const std::string& text) { return writeAll(toChild[1], text); }
    void closeInput() {
        close(toChild[1]);
        toChild[1] = -1;
    }
    // The next frame, or false at end of file.
    bool frame(char& kind, std::string& payload) {
        for (;;) {
            if (frames.next(kind, payload)) return true;
            char buf[4096];
            const ssize_t got = read(fromChild[0], buf, sizeof buf);
            if (got <= 0) return false;
            frames.feed(buf, static_cast<size_t>(got));
        }
    }
    int join() {
        thread.join();
        return status;
    }
    ~InProcessStream() {
        if (thread.joinable()) {
            closeInput();
            thread.join();
        }
        for (int fd : {toChild[0], toChild[1], fromChild[0], fromChild[1]})
            if (fd >= 0) close(fd);
    }
};

static void test_run_stream_in_process() {
    StreamWork work("inproc");
    const std::vector<GameRecord> whole = playSelfPlayGames(work.net(), work.job.sp, work.job.seed,
                                                            work.job.generation, 0, 8, EngineRun{"naive", 1});
    for (int workers : {1, 3}) {
        InProcessStream s(work.jobPath, workers);
        StreamChecker check(work.jobBytes, buildId());
        char kind;
        std::string payload;
        assert(s.frame(kind, payload) && kind == 'H');
        assert(check.hello(payload).fingerprint == engineFingerprint("naive"));
        // Ask for a few, out of order; then more as games come back.
        std::vector<int> ask = {6, 1, 3, 0, 7};
        for (int i : {6, 1}) assert(s.send("play " + std::to_string(i) + "\n"));
        size_t asked = 2;
        std::map<int, GameRecord> got;
        while (got.size() < ask.size()) {
            assert(s.frame(kind, payload) && kind == 'G');
            StreamGame g = check.game(payload);
            assert(g.evaluations > 0 && g.seconds >= 0);
            got[g.index] = std::move(g.record);
            if (asked < ask.size()) assert(s.send("play " + std::to_string(ask[asked++]) + "\n"));
        }
        assert(s.send("end\n"));
        assert(s.frame(kind, payload) && kind == 'E');
        const SliceStats stats = check.end(payload);
        assert(stats.evaluations > 0);
        assert(!s.frame(kind, payload));   // the process's end of file
        assert(s.join() == 0);
        for (const auto& [i, r] : got) assert(sameRecord(r, whole[static_cast<size_t>(i)]));
    }
    // `end` at once: no game, an end frame.
    InProcessStream s(work.jobPath, 2);
    assert(s.send("end\n"));
    char kind;
    std::string payload;
    assert(s.frame(kind, payload) && kind == 'H' && s.frame(kind, payload) && kind == 'E');
    assert(s.join() == 0);
}

// End of file before `end` (the coordinator is gone) stops games in flight
// at once; a bad command, an index asked twice or outside the job stop it
// too; a bad job never says hello.
static void test_run_stream_refusals() {
    StreamWork work("refusals", 400);
    {
        InProcessStream s(work.jobPath, 2);
        assert(s.send("play 0\nplay 1\n"));
        const auto start = std::chrono::steady_clock::now();
        s.closeInput();
        assert(s.join() == 1);
        assert(std::chrono::steady_clock::now() - start < std::chrono::seconds(5));
        assert(contains(s.err.str(), "stdin closed before `end`"));
    }
    for (const std::string& bad : {std::string("play 8\n"), std::string("play 2\nplay 2\n"),
                                   std::string("stop\n"), std::string("play 1x\n"), std::string(80, 'p')}) {
        InProcessStream s(work.jobPath, 2);
        assert(s.send(bad));
        assert(s.join() == 1);
        assert(contains(s.err.str(), "selfplay: "));
    }
    {
        StreamWork wrong("wrong_build");
        wrong.job.buildId = "0000000000000000000000000000000000000000";
        std::ofstream(wrong.jobPath, std::ios::binary | std::ios::trunc) << serializeJob(wrong.job);
        InProcessStream s(wrong.jobPath, 1);
        char kind;
        std::string payload;
        assert(!s.frame(kind, payload));   // no hello
        assert(s.join() == 1 && contains(s.err.str(), "build ID mismatch"));
    }
}

// A child process with pipes: what it writes comes back, its stdin's end
// ends it, stop() ends one that would not, a write to one that is gone
// fails quietly, and a program that cannot start reads as end of file.
static void test_child_process() {
    {
        ChildProcess cat({"cat"}, "");
        assert(cat.pid() > 0 && cat.send("hello\n"));
        std::string got;
        char buf[64];
        while (got.size() < 6) {
            const long n = cat.receive(buf, sizeof buf, 5.0);
            assert(n > 0);
            got.append(buf, static_cast<size_t>(n));
        }
        assert(got == "hello\n");
        assert(cat.receive(buf, sizeof buf, 0.05) == -1);   // nothing more for now
        cat.stop();
        const ProcessResult r = cat.wait();
        assert(r.cancelled);
    }
    {
        ChildProcess sleeper({"sleep", "600"}, "");
        const auto start = std::chrono::steady_clock::now();
        sleeper.stop();
        sleeper.wait();
        assert(std::chrono::steady_clock::now() - start < std::chrono::seconds(10));
    }
    {
        ChildProcess quick({"true"}, "");
        char buf[8];
        while (quick.receive(buf, sizeof buf, 5.0) != 0) {
        }
        assert(quick.wait().exitCode == 0);
        assert(!quick.send(std::string(1 << 20, 'x')));   // gone: false, no SIGPIPE
    }
    {
        ChildProcess missing({"/nonexistent/azgo-program"}, "");
        char buf[8];
        assert(missing.receive(buf, sizeof buf, 1.0) == 0);
        const ProcessResult r = missing.wait();
        assert(r.exitCode != 0);
    }
}

// The real selfplay binary through the subprocess transport, and
// runStreamJob through the in-process transport, answer a stream exactly
// as playSelfPlayGames plays its games.
static void test_transports_stream() {
    StreamWork work("transport");
    const std::vector<GameRecord> whole = playSelfPlayGames(work.net(), work.job.sp, work.job.seed,
                                                            work.job.generation, 0, 8, EngineRun{"naive", 1});
    MachineSettings m;
    m.alias = "fake";
    m.exe = AZGO_SELFPLAY_EXE;
    m.workdir = work.dir.string();
    for (bool inProcess : {false, true}) {
        std::unique_ptr<Transport> t;
        if (inProcess) t = std::make_unique<InProcessTransport>(m);
        else t = makeTransport(m);
        const std::string log = (work.dir / "stream.log").string();
        std::unique_ptr<StreamChannel> c =
            t->openStream({"--stream", "--job", work.jobPath, "--backend", "naive", "--workers", "2"}, log);
        StreamFrames frames;
        StreamChecker check(work.jobBytes, buildId());
        auto frame = [&](char& kind, std::string& payload) {
            for (;;) {
                if (frames.next(kind, payload)) return true;
                char buf[4096];
                const long n = c->receive(buf, sizeof buf, 30.0);
                assert(n != -1);
                if (n == 0) return false;
                frames.feed(buf, static_cast<size_t>(n));
            }
        };
        char kind;
        std::string payload;
        assert(frame(kind, payload) && kind == 'H');
        check.hello(payload);
        assert(c->send("play 4\nplay 2\nplay 7\n"));
        std::map<int, GameRecord> got;
        while (got.size() < 3) {
            assert(frame(kind, payload) && kind == 'G');
            StreamGame g = check.game(payload);
            got[g.index] = std::move(g.record);
        }
        assert(c->send("end\n"));
        assert(frame(kind, payload) && kind == 'E');
        check.end(payload);
        assert(!frame(kind, payload));
        assert(c->wait().exitCode == 0);
        for (const auto& [i, r] : got) assert(sameRecord(r, whole[static_cast<size_t>(i)]));
        // Stopped in the middle: it ends promptly, not with 0.
        StreamWork slow("transport_slow", 400);
        std::unique_ptr<StreamChannel> d =
            t->openStream({"--stream", "--job", slow.jobPath, "--backend", "naive", "--workers", "1"}, log);
        assert(d->send("play 0\n"));
        const auto start = std::chrono::steady_clock::now();
        d->stop();
        assert(d->wait().exitCode != 0);
        assert(std::chrono::steady_clock::now() - start < std::chrono::seconds(10));
    }
}

// writeAll to a pipe nobody reads fails without killing the process.
static void test_write_all_without_sigpipe() {
    int fds[2];
    assert(pipe(fds) == 0);
    close(fds[0]);
    assert(!writeAll(fds[1], "hello"));
    close(fds[1]);
}

int main() {
    test_frames_round_trip();
    test_frame_refusals();
    test_checker_refusals();
    test_run_stream_in_process();
    test_run_stream_refusals();
    test_write_all_without_sigpipe();
    test_child_process();
    test_transports_stream();
    std::printf("test_selfplay_stream: OK\n");
    return 0;
}
