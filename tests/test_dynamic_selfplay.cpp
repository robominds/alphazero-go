// Dynamic allocation in the coordinator (selfplay_allocation = dynamic):
// local engines and remote streams (a fake remote: the real selfplay binary
// as a separate process, or runStreamJob on a thread) share one queue of a
// generation's games; every index is played exactly once and merged in
// index order, so with engines of one arithmetic the records are those of
// one engine alone. A stream that fails mid-generation gives its games in
// flight back, is retried, then retires (or stops the generation); the
// fallback engine joins when every engine has retired; a forced assignment
// replays a recorded one.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include "azgo/build_info.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/config.hpp"
#include "azgo/coordinator.hpp"
#include "azgo/engines.hpp"
#include "azgo/network.hpp"
#include "azgo/selfplay_slice.hpp"
#include "azgo/selfplay_stream.hpp"

using namespace azgo;

static bool sameRecords(const std::vector<GameRecord>& a, const std::vector<GameRecord>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const GameRecord& x = a[i];
        const GameRecord& y = b[i];
        if (!(x.boardSize == y.boardSize && std::memcmp(&x.komi, &y.komi, 4) == 0 && x.moves == y.moves &&
              x.stones == y.stones && x.visits == y.visits &&
              std::memcmp(&x.resultForBlack, &y.resultForBlack, 4) == 0 &&
              x.finalOwnership == y.finalOwnership && x.resigned == y.resigned &&
              x.resignDisabled == y.resignDisabled && x.resignLows[0] == y.resignLows[0] &&
              x.resignLows[1] == y.resignLows[1]))
            return false;
    }
    return true;
}

static bool contains(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

// A run directory, a network and a self-play configuration for 5x5 games,
// resignation on (resigned, disabled and played-out games).
struct Fixture {
    std::filesystem::path dir;
    std::string netPath;
    SelfPlayConfig sp;
    Config cfg;
    explicit Fixture(const std::string& name)
        : dir(std::filesystem::temp_directory_path() /
              ("azgo_test_dynamic_" + std::to_string(getpid()) + "_" + name)) {
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir / "run");
        cfg.boardSize = 5;
        cfg.history = 2;
        cfg.blocks = 1;
        cfg.filters = 4;
        cfg.valueHidden = 8;
        cfg.seed = 11;
        cfg.gamesPerGeneration = 12;
        cfg.selfplayAllocation = "dynamic";
        netPath = (dir / "net.azn").string();
        Network net(cfg.netConfig(), 33);
        saveNetwork(net, netPath);
        sp.mcts.simulations = 8;
        sp.mcts.batchSize = 4;
        sp.temperatureMoves = 4;
        sp.komi = 7.5f;
        sp.maxMoves = 50;
        sp.resignation = true;
        sp.resignThreshold = -0.0f;
        sp.resignDisabledFraction = 0.5f;
    }
    std::string runDir() const { return (dir / "run").string(); }
    // A hosts file line for a fake remote: no ssh, so the subprocess
    // transport runs the real selfplay binary, through a link of the
    // alias's own: without the link the machine is unreachable (drop), and
    // with it again it is back (restore).
    std::string fake(const std::string& alias, const std::string& extra = "") const {
        restore(alias);
        return alias + " exe=" + exe(alias) + " workdir=" + (dir / ("remote_" + alias)).string() + " " + extra +
               "\n";
    }
    std::string exe(const std::string& alias) const { return (dir / ("selfplay_" + alias)).string(); }
    void drop(const std::string& alias) const { std::filesystem::remove(exe(alias)); }
    void restore(const std::string& alias) const {
        if (!std::filesystem::is_symlink(exe(alias))) std::filesystem::create_symlink(AZGO_SELFPLAY_EXE, exe(alias));
    }
    std::vector<GameRecord> whole(int g) const {
        return playSelfPlayGames(loadNetwork(netPath), sp, cfg.seed, g, 0, cfg.gamesPerGeneration,
                                 EngineRun{"naive", 1});
    }
    std::vector<GameRecord> play(Coordinator& c, int g) const {
        return c.play(g, netPath, loadNetwork(netPath), sp, cfg.gamesPerGeneration);
    }
    ~Fixture() {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
};

// Reports count every game once, and the assignment names an engine of the
// list for each game.
static void checkBookkeeping(const Coordinator& c, int games) {
    int counted = 0;
    for (const SliceReport& r : c.reports()) {
        assert(r.dynamic && r.planned == r.engine && !r.fingerprint.empty() && !r.build.empty());
        counted += r.slice.count;
    }
    assert(counted == games && static_cast<int>(c.assignment().size()) == games);
    std::map<std::string, int> byEngine;
    for (const std::string& e : c.assignment()) ++byEngine[e];
    for (const SliceReport& r : c.reports()) assert(byEngine[r.engine] >= r.slice.count);
}

// Local engines only: two pools of one arithmetic share the queue; the
// records are one engine's.
static void test_local_engines() {
    Fixture f("local");
    applyOverride(f.cfg, "selfplay_engines=local/naive:1, local/naive:2");
    Coordinator c(f.cfg, f.runDir(), HostsFile{}, nullptr);
    c.preflight();
    for (int g : {1, 2}) assert(sameRecords(f.play(c, g), f.whole(g)));
    checkBookkeeping(c, 12);
    assert(c.reports().size() == 2 && c.reports()[0].engine == "local/naive");
    // The static path is untouched and says nothing about an assignment.
    Config fixed = f.cfg;
    fixed.selfplayAllocation = "static";
    Coordinator s(fixed, f.runDir(), HostsFile{}, nullptr);
    assert(sameRecords(f.play(s, 2), f.whole(2)) && s.assignment().empty() && !s.reports()[0].dynamic);
}

// A fake remote beside a local engine, as a separate process and in
// process: the same records, the remote's own fingerprint and build, and
// nothing left behind but the network.
static void test_fake_remote() {
    for (bool inProcess : {false, true}) {
        Fixture f(inProcess ? "remote_inproc" : "remote");
        applyOverride(f.cfg, "selfplay_engines=local/naive:1, fake/naive:2");
        distributed_probe::useInProcessTransport(inProcess);
        Coordinator c(f.cfg, f.runDir(), parseHostsText(f.fake("fake"), "h"), nullptr);
        c.preflight();
        for (int g : {1, 2}) assert(sameRecords(f.play(c, g), f.whole(g)));
        checkBookkeeping(c, 12);
        const SliceReport& remote = c.reports()[1];
        assert(remote.engine == "fake/naive" && remote.fingerprint == engineFingerprint("naive") &&
               remote.build == shortBuildId());
        assert(remote.note.empty() && c.reports()[0].note.empty());
        assert(std::filesystem::is_empty(f.dir / "run" / "selfplay-work"));   // as static slices leave it
        int files = 0;
        for (const auto& e : std::filesystem::recursive_directory_iterator(f.dir / "remote_fake"))
            files += e.is_regular_file() ? 1 : 0;
        assert(files == 1);   // the network
        distributed_probe::reset();
    }
}

// A remote engine only, slow to start: the queue waits for it.
static void test_remote_only() {
    Fixture f("remote_only");
    applyOverride(f.cfg, "selfplay_engines=fa/naive:1, fb/naive:1");
    distributed_probe::delaySlice("fb/naive", 0.3);
    Coordinator c(f.cfg, f.runDir(), parseHostsText(f.fake("fa") + f.fake("fb"), "h"), nullptr);
    c.preflight();
    assert(sameRecords(f.play(c, 3), f.whole(3)));
    checkBookkeeping(c, 12);
    distributed_probe::reset();
}

// A stream that fails mid-generation gives back its games in flight; the
// retry plays on; two failures retire the engine and the others play its
// games; with every engine retired the fallback engine joins; with
// selfplay_fallback = stop the generation throws. No game is lost or
// played twice, and the records are the failure-free ones.
// The local engine starts late, so the remote has games when it fails.
static void test_failures() {
    {
        Fixture f("retry");
        applyOverride(f.cfg, "selfplay_engines=local/naive:1, fake/naive:2");
        distributed_probe::failSession("fake/naive", 1, 2);
        distributed_probe::delaySlice("local/naive", 0.5);
        std::ostringstream log;
        Coordinator c(f.cfg, f.runDir(), parseHostsText(f.fake("fake"), "h"), &log);
        c.preflight();
        assert(sameRecords(f.play(c, 1), f.whole(1)));
        checkBookkeeping(c, 12);
        assert(contains(c.reports()[1].note, "1 failed stream") && !contains(c.reports()[1].note, "retired"));
        assert(contains(log.str(), "fake/naive's stream failed"));
        // Its log is kept (a failure), with the reason.
        assert(std::filesystem::exists(f.dir / "run" / "selfplay-work" / "gen_0001" / "fake_naive.log"));
        distributed_probe::reset();
    }
    {
        Fixture f("retire");
        applyOverride(f.cfg, "selfplay_engines=local/naive:1, fake/naive:2");
        distributed_probe::failSession("fake/naive", 2, 1);
        distributed_probe::delaySlice("local/naive", 0.5);
        std::ostringstream log;
        Coordinator c(f.cfg, f.runDir(), parseHostsText(f.fake("fake"), "h"), &log);
        c.preflight();
        assert(sameRecords(f.play(c, 1), f.whole(1)));
        checkBookkeeping(c, 12);
        assert(contains(c.reports()[1].note, "2 failed streams") && contains(c.reports()[1].note, "retired"));
        assert(contains(log.str(), "fake/naive retired for this generation"));
        // The next generation tries it again.
        distributed_probe::reset();
        assert(sameRecords(f.play(c, 2), f.whole(2)));
        assert(c.reports()[1].note.empty());
        distributed_probe::reset();
    }
    {
        Fixture f("fallback");
        applyOverride(f.cfg, "selfplay_engines=fa/naive:1, fb/naive:1");
        distributed_probe::failSession("fa/naive", 2, 1);
        distributed_probe::failSession("fb/naive", 2, 0);
        std::ostringstream log;
        Coordinator c(f.cfg, f.runDir(), parseHostsText(f.fake("fa") + f.fake("fb"), "h"), &log);
        c.preflight();
        assert(sameRecords(f.play(c, 1), f.whole(1)));
        checkBookkeeping(c, 12);
        assert(c.reports().size() == 3 && c.reports()[2].engine == "local/naive" &&
               c.reports()[2].note == "fallback: every engine retired");
        assert(contains(log.str(), "every engine retired; local/naive plays the remaining games"));
        distributed_probe::reset();
    }
    {
        Fixture f("stop");
        applyOverride(f.cfg, "selfplay_engines=local/naive:1, fake/naive:2");
        f.cfg.selfplayFallback = "stop";
        distributed_probe::failSession("fake/naive", 2, 1);
        distributed_probe::delaySlice("local/naive", 0.5);
        Coordinator c(f.cfg, f.runDir(), parseHostsText(f.fake("fake"), "h"), nullptr);
        c.preflight();
        std::string error;
        try {
            f.play(c, 1);
        } catch (const SliceError& e) {
            error = e.what();
        }
        assert(contains(error, "fake/naive failed twice") && contains(error, "selfplay_fallback = stop"));
        distributed_probe::reset();
    }
    {
        // A machine that cannot be reached: retired at once, not tried.
        Fixture f("unreachable");
        applyOverride(f.cfg, "selfplay_engines=local/naive:1, gone/naive:1");
        std::ostringstream log;
        Coordinator c(f.cfg, f.runDir(),
                      parseHostsText("gone exe=/nonexistent/selfplay workdir=" + (f.dir / "gone").string() + "\n", "h"),
                      &log);
        c.preflight();
        assert(sameRecords(f.play(c, 1), f.whole(1)));
        assert(c.reports()[1].slice.count == 0 && contains(c.reports()[1].note, "retired: gone unreachable"));
        distributed_probe::reset();
    }
}

// A remote whose stream never answers (stuck before its hello, as a GPU
// that hangs while it starts) and ignores its stdin: the other engines
// play every game, and once the session has had its time to end after
// `end`, it is stopped rather than waited for.
static void test_silent_stream() {
    Fixture f("silent");
    const std::string script = (f.dir / "silent.sh").string();
    {
        std::ofstream out(script);
        out << "#!/bin/sh\n"
               "for a in \"$@\"; do [ \"$a\" = --stream ] && exec sleep 60; done\n"
               "exec '"
            << AZGO_SELFPLAY_EXE << "' \"$@\"\n";
    }
    std::filesystem::permissions(script, std::filesystem::perms::owner_all);
    applyOverride(f.cfg, "selfplay_engines=local/naive:1, silent/naive:1");
    distributed_probe::endTimeout(1);
    Coordinator c(f.cfg, f.runDir(),
                  parseHostsText("silent exe=" + script + " workdir=" + (f.dir / "remote_silent").string() + "\n",
                                 "h"),
                  nullptr);
    c.preflight();
    const auto start = std::chrono::steady_clock::now();
    assert(sameRecords(f.play(c, 1), f.whole(1)));
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    assert(seconds < 30);   // not the 60 s the stuck process would take to end by itself
    assert(c.reports()[0].slice.count == 12 && c.reports()[1].slice.count == 0);
    distributed_probe::reset();
}

// A remote whose stream answers a self-play job with an arena game frame
// (a well-formed frame of the wrong kind): each session fails on it, before
// any checker sees it, and the engine retires; the local engine, starting
// late so the sessions have games, plays every game.
static void test_unexpected_frame() {
    Fixture f("unexpected");
    const std::string frame = (f.dir / "arena.frame").string();
    {
        std::ofstream out(frame, std::ios::binary);
        ArenaGameResult game;
        game.scoreA = 1;
        game.moves = {12, 25, 25};
        out << streamArenaGameFrame(0, game, 0.5, 10);
    }
    const std::string script = (f.dir / "arena.sh").string();
    {
        std::ofstream out(script);
        out << "#!/bin/sh\n"
               "for a in \"$@\"; do [ \"$a\" = --stream ] && { cat '"
            << frame << "'; exec sleep 60; }; done\n"
               "exec '"
            << AZGO_SELFPLAY_EXE << "' \"$@\"\n";
    }
    std::filesystem::permissions(script, std::filesystem::perms::owner_all);
    applyOverride(f.cfg, "selfplay_engines=local/naive:1, odd/naive:1");
    distributed_probe::delaySlice("local/naive", 0.5);
    std::ostringstream log;
    Coordinator c(f.cfg, f.runDir(),
                  parseHostsText("odd exe=" + script + " workdir=" + (f.dir / "remote_odd").string() + "\n", "h"),
                  &log);
    c.preflight();
    const auto start = std::chrono::steady_clock::now();
    assert(sameRecords(f.play(c, 1), f.whole(1)));
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    assert(seconds < 30);   // each session stopped, not the 60 s its process would sleep
    assert(c.reports()[0].slice.count == 12 && c.reports()[1].slice.count == 0);
    assert(contains(c.reports()[1].note, "2 failed streams") &&
           contains(c.reports()[1].note, "retired: odd: stream: an unexpected 'A' frame"));
    assert(contains(log.str(), "odd/naive's stream failed (odd: stream: an unexpected 'A' frame)"));
    distributed_probe::reset();
}

// A forced assignment is played as given, whoever is faster.
static void test_forced_assignment() {
    Fixture f("forced");
    applyOverride(f.cfg, "selfplay_engines=local/naive:1, fake/naive:1");
    std::vector<std::string> forced;
    for (int i = 0; i < 12; ++i) forced.push_back(i % 3 == 0 ? "fake/naive" : "local/naive");
    distributed_probe::forceAssignment(4, forced);
    Coordinator c(f.cfg, f.runDir(), parseHostsText(f.fake("fake"), "h"), nullptr);
    c.preflight();
    assert(sameRecords(f.play(c, 4), f.whole(4)));
    assert(c.assignment() == forced);
    assert(c.reports()[0].slice.count == 8 && c.reports()[1].slice.count == 4);
    distributed_probe::reset();
}

// The games manifest: one row per game, rows of later generations dropped.
static void test_games_manifest() {
    Fixture f("manifest");
    writeGamesManifest(f.runDir(), 1, {"local/metal", "pc/cuda"});
    writeGamesManifest(f.runDir(), 2, {"pc/cuda"});
    writeGamesManifest(f.runDir(), 2, {"local/metal", "local/metal", "pc/cuda"});   // redone
    std::ifstream in(gamesManifestPath(f.runDir()));
    std::stringstream text;
    text << in.rdbuf();
    assert(text.str() == "generation,game,engine\n1,0,local/metal\n1,1,pc/cuda\n2,0,local/metal\n"
                         "2,1,local/metal\n2,2,pc/cuda\n");
    pruneGamesManifest(f.runDir(), 2);
    std::ifstream in2(gamesManifestPath(f.runDir()));
    std::stringstream pruned;
    pruned << in2.rdbuf();
    assert(pruned.str() == "generation,game,engine\n1,0,local/metal\n1,1,pc/cuda\n");
    // Dynamic rows of the engines manifest leave `first` empty.
    SliceReport r;
    r.dynamic = true;
    r.slice.count = 7;
    r.planned = r.engine = "pc/cuda";
    r.fingerprint = "cuda; x";
    writeManifest(f.runDir(), 1, {r});
    std::ifstream in3(manifestPath(f.runDir()));
    std::stringstream rows;
    rows << in3.rdbuf();
    assert(contains(rows.str(), "\n1,,7,pc/cuda,pc/cuda,\"cuda; x\","));
}

// ---- readmission ----------------------------------------------------------------

// A log the test may read while the coordinator writes to it.
class SharedLog : public std::streambuf {
public:
    std::string str() {
        std::lock_guard<std::mutex> lock(mutex_);
        return text_;
    }
    // Waits (at most a minute) until the log holds `part`.
    void await(const std::string& part) {
        for (int i = 0; i < 6000 && !contains(str(), part); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        assert(contains(str(), part));
    }
protected:
    int overflow(int c) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (c != EOF) text_ += static_cast<char>(c);
        return c;
    }
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        std::lock_guard<std::mutex> lock(mutex_);
        text_.append(s, static_cast<size_t>(n));
        return n;
    }
private:
    std::mutex mutex_;
    std::string text_;
};

// Every file name seen under some directories while it runs, as
// "<parent directory>/<file>": a session's files live only while it does.
struct FileWatcher {
    std::set<std::string> seen;
    std::atomic<bool> done{false};
    std::thread thread;
    explicit FileWatcher(std::vector<std::string> dirs) {
        thread = std::thread([this, dirs] {
            while (!done) {
                for (const std::string& dir : dirs) {
                    std::error_code ec;
                    for (std::filesystem::recursive_directory_iterator it(dir, ec), end; !ec && it != end;
                         it.increment(ec))
                        seen.insert(it->path().parent_path().filename().string() + "/" +
                                    it->path().filename().string());
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        });
    }
    void stop() {
        done = true;
        thread.join();
    }
    bool any(const std::string& part) const {
        for (const std::string& name : seen)
            if (contains(name, part)) return true;
        return false;
    }
};

static int count(const std::string& s, const std::string& part) {
    int n = 0;
    for (size_t at = s.find(part); at != std::string::npos; at = s.find(part, at + 1)) ++n;
    return n;
}

// A process that runs for 20 s with `word` in its command line, as a
// session's selfplay holds its job's path there; in a process group of
// its own (with its sleep), its output nowhere.
static pid_t holding(const std::string& word) {
    const pid_t pid = fork();
    if (pid == 0) {
        setpgid(0, 0);
        const int null = open("/dev/null", O_WRONLY);
        dup2(null, 1);
        dup2(null, 2);
        execl("/bin/sh", "sh", "-c", "sleep 20; :", word.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    assert(pid > 0);
    return pid;
}

// Whether `pid` (a child of holding) was killed by a signal; waits for it,
// then stops what is left of its group.
static bool killed(pid_t pid) {
    int status = 0;
    const bool signalled = waitpid(pid, &status, 0) == pid && WIFSIGNALED(status);
    kill(-pid, SIGKILL);
    return signalled;
}

// Whether `pid` (a child of holding) still runs; then its group is stopped
// and it is waited for.
static bool alive(pid_t pid) {
    int status = 0;
    const bool running = waitpid(pid, &status, WNOHANG) == 0;
    kill(-pid, SIGKILL);
    waitpid(pid, &status, 0);
    return running;
}

// A remote whose session drops after a game, its machine gone (the session
// plays in process, so it needs no binary; the probe after the failure
// finds none): the engine retires and waits; once the machine is back it is
// readmitted, plays games again in a session of the next attempt, and every
// game is played once, as one engine plays them. The local engine starts
// late, so the remote has games when it fails and when it comes back.
// What a session that failed with its machine gone may leave there (its
// selfplay, its job file; a link that went and came back) is removed
// before the new session, and nothing of another engine's, generation's
// or phase's session.
static void test_readmit_selfplay() {
    Fixture f("readmit");
    applyOverride(f.cfg, "selfplay_engines=local/naive:1, fake/naive:2");
    distributed_probe::useInProcessTransport(true);
    SharedLog buf;
    std::ostream log(&buf);
    Coordinator c(f.cfg, f.runDir(), parseHostsText(f.fake("fake"), "h"), &log);
    c.preflight();
    distributed_probe::failSession("fake/naive", 1, 1);
    distributed_probe::delaySlice("local/naive", 3);
    distributed_probe::readmitSeconds(0.05);
    f.drop("fake");
    FileWatcher watch({f.runDir(), (f.dir / "remote_fake").string()});
    std::string remote;   // the run's directory on the machine
    const std::vector<std::string> others = {"g0001_s11_a1.job", "g0002_s1_a1.job", "arena_g0001_w1_s1_a1.job"};
    pid_t stale = -1;
    std::vector<pid_t> decoys;
    std::thread back([&] {
        buf.await("fake/naive retired for this generation (fake unreachable)");
        for (const auto& entry : std::filesystem::directory_iterator(f.dir / "remote_fake"))
            if (entry.is_directory()) remote = entry.path().string();
        stale = holding(remote + "/g0001_s1_a1.job");
        std::ofstream(remote + "/g0001_s1_a1.job").put('x');
        for (const std::string& other : others) {
            decoys.push_back(holding(remote + "/" + other));
            std::ofstream(remote + "/" + other).put('x');
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        f.restore("fake");
    });
    const std::vector<GameRecord> records = f.play(c, 1);
    back.join();
    watch.stop();
    assert(!remote.empty() && killed(stale) && !std::filesystem::exists(remote + "/g0001_s1_a1.job"));
    for (size_t i = 0; i < others.size(); ++i)
        assert(alive(decoys[i]) && std::filesystem::exists(remote + "/" + others[i]));
    assert(sameRecords(records, f.whole(1)));
    checkBookkeeping(c, 12);
    const SliceReport& r = c.reports()[1];
    assert(r.engine == "fake/naive" && r.slice.count > 1);
    assert(contains(buf.str(),
                    "note: generation 1: fake/naive readmitted (machine answered again); it plays games again"));
    // The retirement's reason went with it.
    assert(contains(r.note, "1 failed stream") && contains(r.note, "readmitted 1 time") &&
           !contains(r.note, "retired") && !contains(r.note, "unreachable"));
    assert(!contains(buf.str(), "fingerprint changed"));   // the same machine, the same arithmetic
    assert(watch.any("_s1_a2.job") || watch.any("_s1_a3.job"));
    distributed_probe::reset();
}

// A machine down at preflight and at the phase's start (not tried): it is
// probed, checked (its build and fingerprint recorded then), readmitted,
// and plays, here as a separate process through the link. Its fingerprint
// then is compared with the manifest's, as preflight would have.
static void test_readmit_not_tried() {
    Fixture f("readmit_not_tried");
    applyOverride(f.cfg, "selfplay_engines=local/naive:1, fake/naive:2");
    SliceReport before;
    before.dynamic = true;
    before.planned = before.engine = "fake/naive";
    before.slice.count = 5;
    before.fingerprint = "naive; before";
    writeManifest(f.runDir(), 1, {before});
    SharedLog buf;
    std::ostream log(&buf);
    const HostsFile hosts = parseHostsText(f.fake("fake"), "h");
    f.drop("fake");
    Coordinator c(f.cfg, f.runDir(), hosts, &log);
    c.preflight();
    assert(contains(buf.str(), "warning: fake unreachable ("));
    distributed_probe::delaySlice("local/naive", 3);
    distributed_probe::readmitSeconds(0.05);
    FileWatcher watch({f.runDir(), (f.dir / "remote_fake").string()});
    std::thread back([&] {
        buf.await("fake/naive retired for this generation (fake unreachable)");
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        f.restore("fake");
    });
    const std::vector<GameRecord> records = f.play(c, 1);
    back.join();
    watch.stop();
    assert(sameRecords(records, f.whole(1)));
    checkBookkeeping(c, 12);
    const SliceReport& r = c.reports()[1];
    assert(r.engine == "fake/naive" && r.slice.count > 0 && r.fingerprint == engineFingerprint("naive"));
    assert(contains(buf.str(), "fake/naive readmitted (machine answered again)"));
    assert(!contains(buf.str(), "note: the fingerprint of"));   // not measured at preflight
    assert(contains(buf.str(), "note: generation 1: fake/naive's fingerprint changed: naive; before -> " +
                                   engineFingerprint("naive") + " after 0 games\n"));
    assert(contains(r.note, "readmitted 1 time"));
    assert(watch.any("_s1_a1.job"));
    distributed_probe::reset();
}

// Two failed sessions while the machine still answers: retired for the
// generation, never probed, never readmitted.
static void test_retired_while_reachable_stays_retired() {
    Fixture f("readmit_reachable");
    applyOverride(f.cfg, "selfplay_engines=local/naive:1, fake/naive:2");
    distributed_probe::useInProcessTransport(true);
    SharedLog buf;
    std::ostream log(&buf);
    Coordinator c(f.cfg, f.runDir(), parseHostsText(f.fake("fake"), "h"), &log);
    c.preflight();
    distributed_probe::failSession("fake/naive", 2, 1);
    distributed_probe::delaySlice("local/naive", 1.5);
    distributed_probe::readmitSeconds(0.05);
    assert(sameRecords(f.play(c, 1), f.whole(1)));
    checkBookkeeping(c, 12);
    assert(contains(c.reports()[1].note, "2 failed streams") && contains(c.reports()[1].note, "retired"));
    assert(contains(buf.str(), "fake/naive retired for this generation") && !contains(buf.str(), "readmitted"));
    distributed_probe::reset();
}

// A machine back with another build: one note, however many probes find
// it so, no readmission, and the generation completes.
static void test_readmit_wrong_build() {
    Fixture f("readmit_build");
    applyOverride(f.cfg, "selfplay_engines=local/naive:1, fake/naive:2");
    distributed_probe::useInProcessTransport(true);
    SharedLog buf;
    std::ostream log(&buf);
    Coordinator c(f.cfg, f.runDir(), parseHostsText(f.fake("fake"), "h"), &log);
    c.preflight();
    distributed_probe::failSession("fake/naive", 1, 1);
    distributed_probe::delaySlice("local/naive", 3);
    distributed_probe::readmitSeconds(0.05);
    f.drop("fake");
    std::thread back([&] {
        buf.await("fake/naive retired for this generation (fake unreachable)");
        distributed_probe::setRemoteBuildId("other");
        f.restore("fake");
    });
    assert(sameRecords(f.play(c, 1), f.whole(1)));
    back.join();
    checkBookkeeping(c, 12);
    assert(count(buf.str(), "not readmitted") == 1);
    assert(contains(buf.str(), "note: generation 1: fake/naive not readmitted: its machine runs build other, "
                               "this run " + buildId() + "\n"));
    assert(!contains(buf.str(), "readmitted (machine answered again)"));
    assert(contains(c.reports()[1].note, "retired"));
    distributed_probe::reset();
}

// A probe that takes long (an ssh to a machine that hangs) does not hold
// the phase up: the generation ends with its last game, the probe is left
// to finish by itself, and no session starts after every game is back,
// though the machine is back by the time the probe answers.
static void test_phase_end_not_delayed() {
    Fixture f("readmit_slow_probe");
    applyOverride(f.cfg, "selfplay_engines=local/naive:1, fake/naive:2");
    distributed_probe::useInProcessTransport(true);
    SharedLog buf;
    std::ostream log(&buf);
    Coordinator c(f.cfg, f.runDir(), parseHostsText(f.fake("fake"), "h"), &log);
    c.preflight();
    distributed_probe::failSession("fake/naive", 1, 1);
    distributed_probe::delaySlice("local/naive", 1);
    distributed_probe::readmitSeconds(0.05);
    distributed_probe::slowProbe("fake/naive", 5);
    f.drop("fake");
    FileWatcher watch({f.runDir(), (f.dir / "remote_fake").string()});
    std::thread back([&] {
        buf.await("fake/naive retired for this generation (fake unreachable)");
        f.restore("fake");
    });
    const auto start = std::chrono::steady_clock::now();
    assert(sameRecords(f.play(c, 1), f.whole(1)));
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    back.join();
    assert(seconds < 4);   // not the 5 s of the probe
    // Past the probe's answer: still no session of a later attempt.
    std::this_thread::sleep_for(std::chrono::duration<double>(std::max(0.0, 7 - seconds)));
    watch.stop();
    assert(watch.any("_s1_a1.job") && !watch.any("_s1_a2.job"));
    assert(!contains(buf.str(), "readmitted"));
    checkBookkeeping(c, 12);
    distributed_probe::reset();
}

// readmit_seconds = 0: a dropped engine stays retired, its machine back or not.
static void test_readmit_off() {
    Fixture f("readmit_off");
    applyOverride(f.cfg, "selfplay_engines=local/naive:1, fake/naive:2");
    f.cfg.readmitSeconds = 0;
    distributed_probe::useInProcessTransport(true);
    SharedLog buf;
    std::ostream log(&buf);
    Coordinator c(f.cfg, f.runDir(), parseHostsText(f.fake("fake"), "h"), &log);
    c.preflight();
    distributed_probe::failSession("fake/naive", 1, 1);
    distributed_probe::delaySlice("local/naive", 1.5);
    distributed_probe::readmitSeconds(0.05);   // the spacing only: 0 still turns it off
    f.drop("fake");
    std::thread back([&] {
        buf.await("fake/naive retired for this generation (fake unreachable)");
        f.restore("fake");
    });
    assert(sameRecords(f.play(c, 1), f.whole(1)));
    back.join();
    checkBookkeeping(c, 12);
    assert(contains(c.reports()[1].note, "retired") && !contains(buf.str(), "readmitted"));
    distributed_probe::reset();
}

// A machine back with another fingerprint (an OS update across a reboot):
// readmitted, with a note of the old and the new one and of the games
// played before the change. The fake remote's binary is a script that, as
// it starts a stream, removes itself (the machine goes, its process stays),
// and that once `updated` exists answers --fingerprint otherwise.
static void test_readmit_fingerprint_change() {
    Fixture f("readmit_fingerprint");
    const std::string flag = (f.dir / "updated").string();
    const std::string script = (f.dir / "selfplay_fake").string();
    auto writeScript = [&] {
        std::ofstream out(script);
        out << "#!/bin/sh\n"
               "[ \"$1\" = --fingerprint ] && [ -e '"
            << flag << "' ] && { echo 'naive; updated'; exit 0; }\n"
               "for a in \"$@\"; do [ \"$a\" = --stream ] && rm -f '"
            << script << "'; done\n"
               "exec '"
            << AZGO_SELFPLAY_EXE << "' \"$@\"\n";
        out.close();
        std::filesystem::permissions(script, std::filesystem::perms::owner_all);
    };
    writeScript();
    applyOverride(f.cfg, "selfplay_engines=local/naive:1, fake/naive:2");
    SharedLog buf;
    std::ostream log(&buf);
    Coordinator c(f.cfg, f.runDir(),
                  parseHostsText("fake exe=" + script + " workdir=" + (f.dir / "remote_fake").string() + "\n", "h"),
                  &log);
    c.preflight();
    distributed_probe::failSession("fake/naive", 1, 1);
    distributed_probe::delaySlice("local/naive", 3);
    distributed_probe::readmitSeconds(0.05);
    std::thread back([&] {
        buf.await("fake/naive retired for this generation (fake unreachable)");
        std::ofstream(flag).put('x');
        writeScript();
    });
    const std::vector<GameRecord> records = f.play(c, 1);
    back.join();
    assert(sameRecords(records, f.whole(1)));
    assert(contains(buf.str(), "note: generation 1: fake/naive's fingerprint changed: " + engineFingerprint("naive") +
                                   " -> naive; updated after 1 game\n"));
    assert(contains(buf.str(), "fake/naive readmitted (machine answered again)"));
    assert(contains(c.reports()[1].note, "readmitted 1 time"));
    distributed_probe::reset();
}

// Preflight's fingerprint-change note compares with an engine's last row
// in a manifest where it played: a row of no games (an engine not tried,
// or retired before its hello) has no fingerprint and is no change.
static void test_fingerprint_note_skips_unplayed_rows() {
    Fixture f("fingerprint_rows");
    applyOverride(f.cfg, "selfplay_engines=local/naive:1");
    SliceReport played, unplayed;
    played.dynamic = unplayed.dynamic = true;
    played.planned = played.engine = unplayed.planned = unplayed.engine = "local/naive";
    played.slice.count = 12;
    played.fingerprint = engineFingerprint("naive");
    writeManifest(f.runDir(), 1, {played});
    writeManifest(f.runDir(), 2, {unplayed});   // count 0, fingerprint ""
    std::ostringstream same;
    Coordinator c(f.cfg, f.runDir(), HostsFile{}, &same);
    c.preflight();
    assert(!contains(same.str(), "the fingerprint of"));
    // A row of games with another fingerprint is a change.
    played.fingerprint = "naive; another machine";
    writeManifest(f.runDir(), 2, {played, unplayed});
    std::ostringstream changed;
    Coordinator d(f.cfg, f.runDir(), HostsFile{}, &changed);
    d.preflight();
    assert(contains(changed.str(), "note: the fingerprint of local/naive changed from \"naive; another machine\""));
}

int main() {
    test_local_engines();
    test_fake_remote();
    test_remote_only();
    test_failures();
    test_silent_stream();
    test_unexpected_frame();
    test_forced_assignment();
    test_games_manifest();
    test_fingerprint_note_skips_unplayed_rows();
    test_readmit_selfplay();
    test_readmit_not_tried();
    test_retired_while_reachable_stays_retired();
    test_readmit_wrong_build();
    test_phase_end_not_delayed();
    test_readmit_off();
    test_readmit_fingerprint_change();
    std::printf("test_dynamic_selfplay: OK\n");
    return 0;
}
