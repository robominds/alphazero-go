// Pipelined generations (pipeline = on): a generation's training and arena
// run beside the next generation's self-play.
//
// First the coordinator with two phases at once: a self-play phase (static
// or dynamic) on one thread and arena waves on another, sharing a fake
// remote, give exactly what each gives alone; neither phase removes the
// other's network from the machine; cancel() stops either.
//
// Then the trainer: a pipelined run's commits hold the next generation's
// games; self-play plays the best network of the last commit (the lag is
// checked by replaying committed games from the checkpoint the rule names);
// a run stopped at any commit, or crashed at any point of an iteration, and
// resumed is the straight run byte for byte; switching on and off at
// commits; failures in either half stop the other and commit nothing.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include "azgo/arena.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/config.hpp"
#include "azgo/coordinator.hpp"
#include "azgo/engines.hpp"
#include "azgo/network.hpp"
#include "azgo/backend.hpp"
#include "azgo/metal.hpp"
#include "azgo/run_state.hpp"
#include "azgo/selfplay_slice.hpp"
#include "azgo/trainer.hpp"

using namespace azgo;
namespace fs = std::filesystem;

static bool contains(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

// A scratch directory under the system temp dir, removed before and after.
struct Dir {
    fs::path path;
    explicit Dir(const std::string& name)
        : path(fs::temp_directory_path() / ("azgo_test_pipeline_" + std::to_string(getpid()) + "_" + name)) {
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~Dir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string str() const { return path.string(); }
    std::string file(const std::string& rel) const { return (path / rel).string(); }
};

static bool sameRecords(const std::vector<GameRecord>& a, const std::vector<GameRecord>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const GameRecord& x = a[i];
        const GameRecord& y = b[i];
        if (!(x.boardSize == y.boardSize && std::memcmp(&x.komi, &y.komi, 4) == 0 && x.moves == y.moves &&
              x.stones == y.stones && x.visits == y.visits &&
              std::memcmp(&x.resultForBlack, &y.resultForBlack, 4) == 0 && x.finalOwnership == y.finalOwnership &&
              x.resigned == y.resigned && x.resignDisabled == y.resignDisabled &&
              x.resignLows[0] == y.resignLows[0] && x.resignLows[1] == y.resignLows[1]))
            return false;
    }
    return true;
}

static bool sameResult(const MatchResult& x, const MatchResult& y) {
    return std::memcmp(&x.scoreA, &y.scoreA, sizeof x.scoreA) == 0 && x.games == y.games && x.winsA == y.winsA &&
           x.winsB == y.winsB && x.draws == y.draws && x.distinctGames == y.distinctGames;
}

// ---- the coordinator with two phases at once ------------------------------------

static NetConfig tinyNet() {
    NetConfig c;
    c.boardSize = 5;
    c.history = 2;
    c.blocks = 1;
    c.filters = 4;
    c.valueHidden = 8;
    return c;
}

// Both lists name this machine and a fake remote (the real selfplay binary
// as a child process): every engine naive here, one arithmetic.
static Config twoPhaseConfig(const std::string& allocation) {
    Config c;
    c.boardSize = 5;
    c.history = 2;
    c.blocks = 1;
    c.filters = 4;
    c.valueHidden = 8;
    c.seed = 17;
    c.mctsBatch = 4;
    c.inferenceBatch = 64;
    c.gamesPerGeneration = 8;
    c.pipeline = true;
    c.selfplayAllocation = allocation;
    applyOverride(c, "selfplay_engines=local/naive:1, fake/naive:1");
    applyOverride(c, "arena_engines=local/naive:1, fake/naive:1");
    return c;
}

static SelfPlayConfig tinySelfPlay() {
    SelfPlayConfig sp;
    sp.mcts.simulations = 8;
    sp.mcts.batchSize = 4;
    sp.temperatureMoves = 4;
    sp.komi = 7.5f;
    sp.maxMoves = 50;
    sp.resignation = true;
    sp.resignThreshold = -0.0f;
    sp.resignDisabledFraction = 0.5f;
    return sp;
}

static ArenaConfig tinyArena() {
    ArenaConfig c;
    c.simulations = 8;
    c.mctsBatch = 4;
    c.temperatureMoves = 2;
    c.komi = 7.5f;
    return c;
}

static HostsFile fakeHosts(const Dir& dir) {
    return parseHostsText("fake exe=" + std::string(AZGO_SELFPLAY_EXE) + " workdir=" + dir.file("remote") + "\n",
                          "hosts");
}

// The name a network has in a remote work directory (by its file's hash).
static std::string remoteNetName(const std::string& path) {
    char name[48];
    std::snprintf(name, sizeof name, "net_%016llx.azn", static_cast<unsigned long long>(fileHash(path)));
    return name;
}

static bool waitFor(const std::function<bool()>& ready, double seconds) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() < end) {
        if (ready()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return ready();
}

// Self-play (static, then dynamic) on one thread and two arena waves on
// another, both using the fake remote: the same records and results as
// each alone, and no attempt failed. The arena's waves start once the
// self-play network is on the remote and before self-play runs there (its
// slices wait a second), so a wave that removed that network would make
// self-play's attempt fail and be retried (its logs kept, its stream
// noted); none is.
static void test_two_phases_at_once() {
    const Network a(tinyNet(), 61), b(tinyNet(), 62), c(tinyNet(), 63);
    const std::vector<ArenaMatchSpec> wave = {
        {ArenaPlayer::network(a), ArenaPlayer::network(b), 6, 701},
        {ArenaPlayer::network(a), ArenaPlayer::bot(6), 4, 702},
        {ArenaPlayer::network(a), ArenaPlayer::network(c), 4, 703},
    };
    ArenaConfig acfg = tinyArena();
    const std::vector<MatchResult> expectedArena = playMatches(wave, acfg, 3, "naive", 64);
    const std::vector<MatchResult> expectedSecond = playMatches({wave[2]}, acfg, 3, "naive", 64);
    const SelfPlayConfig sp = tinySelfPlay();
    for (const std::string allocation : {"static", "dynamic"}) {
        Dir dir("two_" + allocation);
        const Config cfg = twoPhaseConfig(allocation);
        const std::string netPath = dir.file("selfplay.azn");
        Network net(tinyNet(), 33);
        saveNetwork(net, netPath);
        const std::vector<GameRecord> expected =
            playSelfPlayGames(net, sp, cfg.seed, 4, 0, 8, EngineRun{"naive", 1});

        std::ostringstream log;
        Coordinator coordinator(cfg, dir.file("run"), fakeHosts(dir), &log);
        coordinator.preflight();
        distributed_probe::delaySlice("fake/naive", 1.0);
        std::vector<GameRecord> records;
        std::exception_ptr error;
        std::thread selfplay([&] {
            try {
                records = coordinator.play(4, netPath, net, sp, 8);
            } catch (...) {
                error = std::current_exception();
            }
        });
        assert(waitFor([&] {
            std::error_code ec;
            for (const auto& e : fs::recursive_directory_iterator(dir.file("remote"), ec))
                if (e.path().filename() == remoteNetName(netPath)) return true;
            return false;
        }, 20));
        coordinator.beginArena(3);
        const std::vector<MatchResult> first = coordinator.playArena(wave, acfg);
        const std::vector<MatchResult> second = coordinator.playArena({wave[2]}, acfg);
        selfplay.join();
        distributed_probe::reset();
        if (error) std::rethrow_exception(error);

        assert(sameRecords(records, expected));
        for (size_t i = 0; i < wave.size(); ++i) assert(sameResult(first[i], expectedArena[i]));
        assert(sameResult(second[0], expectedSecond[0]));
        // No attempt failed: the work directories are gone, no note.
        assert(!fs::exists(dir.file("run/selfplay-work/gen_0004")));
        assert(!fs::exists(dir.file("run/arena-work")));
        for (const SliceReport& r : coordinator.reports()) assert(r.note.empty() && r.planned == r.engine);
        for (const SliceReport& r : coordinator.arenaReports()) assert(r.note.empty() && r.planned == r.engine);
        assert(!contains(log.str(), "note:"));
    }
}

// cancel() stops a phase in flight (self-play static and dynamic, an arena
// wave) with SliceCancelled, and a phase that starts while it is set at
// once; resetCancel() lets the next one play normally.
static void test_cancel() {
    const Network a(tinyNet(), 61), b(tinyNet(), 62);
    const std::vector<ArenaMatchSpec> wave = {{ArenaPlayer::network(a), ArenaPlayer::network(b), 6, 701}};
    const ArenaConfig acfg = tinyArena();
    const SelfPlayConfig sp = tinySelfPlay();
    auto cancelled = [](const std::function<void()>& f) {
        try {
            f();
        } catch (const SliceCancelled&) {
            return true;
        }
        return false;
    };
    for (const std::string allocation : {"static", "dynamic"}) {
        Dir dir("cancel_" + allocation);
        const Config cfg = twoPhaseConfig(allocation);
        const std::string netPath = dir.file("selfplay.azn");
        Network net(tinyNet(), 33);
        saveNetwork(net, netPath);
        Coordinator coordinator(cfg, dir.file("run"), fakeHosts(dir), nullptr);
        coordinator.preflight();

        // In flight: the remote waits two seconds before it plays (and,
        // dynamic, the local engine before its first game).
        distributed_probe::delaySlice("fake/naive", 2.0);
        distributed_probe::delaySlice("local/naive", 2.0);
        bool threw = false;
        std::thread selfplay([&] { threw = cancelled([&] { coordinator.play(1, netPath, net, sp, 8); }); });
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        coordinator.cancel();
        selfplay.join();
        assert(threw);
        coordinator.beginArena(1);
        threw = false;
        std::thread arena([&] { threw = cancelled([&] { coordinator.playArena(wave, acfg); }); });
        arena.join();   // started while cancelled: at once
        assert(threw);
        distributed_probe::reset();

        // Cleared: both kinds play normally again.
        coordinator.resetCancel();
        assert(sameRecords(coordinator.play(2, netPath, net, sp, 8),
                           playSelfPlayGames(net, sp, cfg.seed, 2, 0, 8, EngineRun{"naive", 1})));
        coordinator.beginArena(2);
        assert(sameResult(coordinator.playArena(wave, acfg)[0], playMatches(wave, acfg, 2, "naive", 64)[0]));
    }
    // An arena wave in flight.
    Dir dir("cancel_arena");
    Coordinator coordinator(twoPhaseConfig("static"), dir.file("run"), fakeHosts(dir), nullptr);
    coordinator.preflight();
    distributed_probe::delaySlice("fake/naive", 2.0);
    coordinator.beginArena(1);
    bool threw = false;
    const auto start = std::chrono::steady_clock::now();
    std::thread arena([&] { threw = cancelled([&] { coordinator.playArena(wave, acfg); }); });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    coordinator.cancel();
    arena.join();
    distributed_probe::reset();
    assert(threw && std::chrono::steady_clock::now() - start < std::chrono::seconds(10));
}

// ---- the trainer ----------------------------------------------------------------

static std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    assert(in && "file should exist");
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void writeFile(const std::string& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << bytes;
    assert(out);
}

static std::vector<std::string> splitLines(const std::string& s) {
    std::vector<std::string> lines;
    std::istringstream in(s);
    for (std::string line; std::getline(in, line);) lines.push_back(line);
    return lines;
}

static std::vector<std::string> splitFields(const std::string& row) {
    std::vector<std::string> fields(1);
    for (char ch : row) {
        if (ch == ',') fields.emplace_back();
        else fields.back() += ch;
    }
    return fields;
}

// A 5x5 run of a fraction of a second per generation, resignation on
// (resigned, disabled and played-out games), gating with a gate a
// candidate passes about half the time, a bot.
static Config pipeConfig() {
    Config c;
    c.boardSize = 5;
    c.history = 2;
    c.blocks = 1;
    c.filters = 8;
    c.valueHidden = 8;
    c.gamesPerGeneration = 8;
    c.simulations = 8;
    c.mctsBatch = 4;
    c.temperatureMoves = 4;
    c.bufferPositions = 1000;
    c.trainSteps = 4;
    c.batchSize = 16;
    c.arenaGames = 4;
    c.arenaSimulations = 4;
    c.arenaOpponents = 1;
    c.workers = 2;
    c.sgfPerGeneration = 2;
    c.lrSchedule = "0:0.02";
    c.botSimulations = {8};
    c.calibrationGames = 2;
    c.resignation = true;
    c.resignThresholdInitial = 0.0f;
    c.resignDisabledFraction = 0.5f;
    c.resignWindow = 6;
    c.resignMinGames = 2;
    c.gating = true;
    c.gatingThreshold = 0.5f;
    c.pipeline = true;
    return c;
}

static std::unique_ptr<Trainer> start(const Config& c, const Dir& dir, const std::string& hosts = "",
                                      std::ostream* log = nullptr) {
    TrainerOptions opts;
    opts.hostsFile = hosts.empty() ? dir.file("no-hosts") : hosts;
    std::ostringstream quiet;
    std::unique_ptr<Trainer> t = Trainer::startNew(c, dir.file("run"), {}, &quiet, opts);
    t->setLog(log);
    return t;
}

static std::unique_ptr<Trainer> resume(const Dir& dir, const std::vector<std::string>& overrides,
                                       const std::string& hosts = "", std::ostream* out = nullptr) {
    TrainerOptions opts;
    opts.hostsFile = hosts.empty() ? dir.file("no-hosts") : hosts;
    std::ostringstream quiet;
    std::unique_ptr<Trainer> t = Trainer::resume(dir.file("run"), overrides, out ? out : &quiet, opts);
    t->setLog(nullptr);
    return t;
}

// A hosts file defining `fake`: the real selfplay binary as a child process.
static std::string fakeHostsFile(const Dir& dir) {
    writeFile(dir.file("hosts"),
              "fake exe=" + std::string(AZGO_SELFPLAY_EXE) + " workdir=" + dir.file("remote") + "\n");
    return dir.file("hosts");
}

// Everything a run wrote that must not depend on timing or interruptions:
// every checkpoint, state.bin, the rating files, the SGFs, log.csv without
// its timing columns (selfplay_s, train_s, arena_s), and the manifests
// without their seconds. With dynamic allocation, which engine played
// which game (and so each engine's count, positions and evaluations)
// depends on timing too: of its self-play manifests only the generations,
// game indices and planned engines count.
static std::string runFingerprint(const Dir& dir, int generations, bool withLog = true) {
    const std::string run = dir.file("run");
    std::string out;
    for (int g = 0; g <= generations; ++g) out += readFile(checkpointPath(run, g));
    for (const char* f : {"state.bin", "results.csv", "references.csv", "elo.csv"})
        out += readFile((fs::path(run) / f).string());
    std::vector<std::string> sgf;
    for (const auto& e : fs::directory_iterator(fs::path(run) / "sgf")) sgf.push_back(e.path().string());
    std::sort(sgf.begin(), sgf.end());
    assert(!sgf.empty());
    for (const std::string& f : sgf) out += f.substr(f.rfind('/')) + readFile(f);
    if (withLog)
        for (const std::string& row : splitLines(readFile((fs::path(run) / "log.csv").string()))) {
            const std::vector<std::string> fields = splitFields(row);
            for (size_t i = 0; i < fields.size(); ++i)
                if (i < 9 || i > 11) out += fields[i] + ",";
            out += "\n";
        }
    const bool dynamic = contains(readFile(configPath(run)), "\nselfplay_allocation = dynamic\n");
    for (const std::string f : {"selfplay-engines.csv", "selfplay-games.csv", "arena-engines.csv"}) {
        const fs::path path = fs::path(run) / f;
        if (!fs::exists(path)) continue;
        for (const std::string& row : splitLines(readFile(path.string()))) {
            // The fields that count: not the seconds (the 8th of self-play's
            // rows, the 9th of the arena's), nor with dynamic allocation who
            // played what.
            auto counts = [&](size_t field) {
                if (f == "arena-engines.csv") return field != 8;
                if (f == "selfplay-games.csv") return !dynamic || field != 2;
                return dynamic ? field == 0 || field == 3 : field != 7;
            };
            std::string kept;
            bool quoted = false;
            size_t field = 0;
            for (char ch : row) {
                if (ch == '"') quoted = !quoted;
                if (ch == ',' && !quoted) {
                    ++field;
                    kept += '|';
                    continue;
                }
                if (counts(field)) kept += ch;
            }
            out += kept + "\n";
        }
    }
    return out;
}

static TrainingState stateOf(const Dir& dir) { return loadTrainingState(statePath(dir.file("run"))); }

// The trainer's self-play settings for a config and a v_resign.
static SelfPlayConfig selfPlayOf(const Config& c, float threshold) {
    SelfPlayConfig sp;
    sp.mcts.simulations = c.simulations;
    sp.mcts.batchSize = c.mctsBatch;
    sp.mcts.cPuct = c.cPuct;
    sp.mcts.dirichletScale = c.dirichletScale;
    sp.mcts.noiseFraction = c.noiseFraction;
    sp.mcts.randomSymmetry = c.mctsRandomSymmetry;
    sp.temperatureMoves = c.temperatureMoves;
    sp.komi = c.komi;
    sp.maxMoves = c.effectiveMaxMoves();
    sp.resignation = c.resignation;
    sp.resignThreshold = threshold;
    sp.resignDisabledFraction = c.resignDisabledFraction;
    return sp;
}

// Generation g's games as `network`'s checkpoint plays them.
static std::vector<GameRecord> replay(const Config& c, const Dir& dir, int g, int network, float threshold) {
    return playSelfPlayGames(loadNetwork(checkpointPath(dir.file("run"), network)), selfPlayOf(c, threshold), c.seed,
                             g, 0, c.gamesPerGeneration, EngineRun{"naive", 1});
}

// pipeline = off writes version-5 state and no pending-selfplay.txt; on,
// every commit holds the next generation's games (version 6), the
// prologue's included, and log.csv has a row per trained generation.
static void test_pipelined_run_files() {
    Dir off("off"), on("on");
    Config c = pipeConfig();
    c.pipeline = false;
    start(c, off)->runUntil(2);
    assert(readFile(statePath(off.file("run")))[4] == 5);
    assert(!fs::exists(off.file("run/pending-selfplay.txt")));
    assert(!stateOf(off).pending);

    std::ostringstream log;
    std::unique_ptr<Trainer> t = start(pipeConfig(), on, "", &log);
    t->runUntil(3);
    assert(readFile(statePath(on.file("run")))[4] == 6);
    const TrainingState s = stateOf(on);
    assert(s.generation == 3 && s.pending && s.pending->records.size() == 8);
    assert(s.pending->resignation && s.pending->maxMoves == pipeConfig().effectiveMaxMoves());
    assert(splitLines(readFile(on.file("run/log.csv"))).size() == 4);
    assert(contains(readFile(on.file("run/pending-selfplay.txt")), "generation = 4\nseconds = "));
    assert(contains(readFile(configPath(on.file("run"))), "\npipeline = on\n"));
    // SGFs of the trained generations only: generation 4's games are not processed yet.
    assert(fs::exists(on.file("run/sgf/gen_0003_game_01.sgf")) && !fs::exists(on.file("run/sgf/gen_0004_game_00.sgf")));
    // The prologue, then one line per iteration after the generation's own.
    const std::string text = log.str();
    assert(contains(text, "pipeline: gen 1 self-play by network 0, 8 games "));
    assert(contains(text, ", committed with generation 0\n"));
    assert(contains(text, "gen 1 | selfplay 8 games"));
    assert(contains(text, "pipeline: gen 2 self-play by network 0, 8 games "));
    assert(contains(text, "beside gen 1's training and arena"));
    assert(contains(text, "; generation 3 took "));
    assert(text.find("gen 2 | selfplay") < text.find("pipeline: gen 3 self-play"));
    // A run that is already there does nothing more.
    t->runUntil(3);
    assert(stateOf(on).generation == 3);
}

// The rule: self-play plays the best network of the last commit before it
// starts. After commit g (pending: generation g+1's games), that is the
// best after the arena of g-1 (with gating), or network g-1 (without). The
// committed games are exactly what that network's checkpoint plays, and
// not what network g's plays (which the sequential loop would use after a
// promotion of g). The prologue plays the committed best (no lag).
static void test_lag_rule() {
    for (bool gating : {true, false}) {
        Dir dir(gating ? "lag_gating" : "lag_nogate");
        Config c = pipeConfig();
        c.gating = gating;
        std::unique_ptr<Trainer> t = start(c, dir);
        int bestBefore = 0, promotions = 0;   // best after commit g-1
        for (int g = 1; g <= 5; ++g) {
            t->runGeneration();
            const TrainingState s = stateOf(dir);
            assert(s.generation == g && s.pending);
            const int expected = gating ? bestBefore : g - 1;
            assert(s.pending->network == expected);
            const float threshold = s.pending->resignThreshold;
            assert(threshold == s.resignThreshold);   // calibrated from generation g's games
            assert(sameRecords(s.pending->records, replay(c, dir, g + 1, expected, threshold)));
            assert(!sameRecords(s.pending->records, replay(c, dir, g + 1, g, threshold)));
            promotions += s.bestGeneration == g ? 1 : 0;
            bestBefore = s.bestGeneration;
        }
        if (gating) assert(promotions > 0 && promotions < 5);   // the rule was exercised both ways
    }
    // The prologue: turned on at a sequential run's commit 2, generation 3
    // is played by the best of commit 2 (no lag), generation 4 by it as well
    // (the last commit before it is still 2's).
    Dir dir("lag_prologue");
    Config c = pipeConfig();
    c.pipeline = false;
    start(c, dir)->runUntil(2);
    const int best = stateOf(dir).bestGeneration;
    std::ostringstream log;
    std::unique_ptr<Trainer> t = resume(dir, {"pipeline=on"});
    t->setLog(&log);
    t->runGeneration();
    const TrainingState s = stateOf(dir);
    assert(s.generation == 3 && s.pending->network == best);
    assert(contains(log.str(), "pipeline: gen 3 self-play by network " + std::to_string(best) + ", 8 games"));
    assert(contains(log.str(), "pipeline: gen 4 self-play by network " + std::to_string(best) + ", 8 games"));
    assert(sameRecords(s.pending->records, replay(c, dir, 4, best, s.pending->resignThreshold)));
}

// A run stopped at any commit and resumed is the straight run, byte for
// byte; so is one crashed at any point of an iteration (or of the
// prologue) and resumed: the redo is the iteration that never crashed.
// A crash after state.bin and before log.csv loses that generation's row,
// as in a sequential run.
static void test_stops_and_crashes_resume_exactly(const std::string& name, const Config& c, bool fake,
                                                  bool every = true) {
    const int last = 4;
    Dir straight(name + "_straight");
    start(c, straight, fake ? fakeHostsFile(straight) : "")->runUntil(last);
    const std::string expected = runFingerprint(straight, last);
    const std::string expectedNoLog = runFingerprint(straight, last, false);
    const std::vector<std::string> expectedLog = splitLines(readFile(straight.file("run/log.csv")));

    for (int stop = every ? 0 : 2; stop < (every ? last : 3); ++stop) {
        Dir dir(name + "_stop" + std::to_string(stop));
        const std::string hosts = fake ? fakeHostsFile(dir) : "";
        start(c, dir, hosts)->runUntil(stop);
        resume(dir, {}, hosts)->runUntil(last);
        assert(runFingerprint(dir, last) == expected);
    }
    using P = pipeline_probe::Point;
    struct Crash {
        P point;
        int generation;
    };
    const Crash crashes[] = {{P::PrologueSelfPlay, 1}, {P::Processed, 1}, {P::SelfPlay, 2}, {P::Validation, 2},
                             {P::Training, 1},         {P::Checkpoint, 2}, {P::Arena, 1},   {P::Manifests, 2},
                             {P::State, 1},            {P::SelfPlay, 4}, {P::Arena, 3}};
    for (const Crash& crash : crashes) {
        if (!every && crash.point != P::SelfPlay && crash.point != P::Validation && crash.point != P::Arena) continue;
        Dir dir(name + "_crash" + std::to_string(static_cast<int>(crash.point)) + "_" +
                std::to_string(crash.generation));
        const std::string hosts = fake ? fakeHostsFile(dir) : "";
        std::unique_ptr<Trainer> t = start(c, dir, hosts);
        pipeline_probe::crashAt(crash.point, crash.generation);
        bool crashed = false;
        try {
            t->runUntil(last);
        } catch (const std::runtime_error& e) {
            crashed = contains(e.what(), "pipeline_probe: crash");
        }
        pipeline_probe::reset();
        assert(crashed);
        t.reset();
        resume(dir, {}, hosts)->runUntil(last);
        if (crash.point == P::State) {
            assert(runFingerprint(dir, last, false) == expectedNoLog);
            std::vector<std::string> rows = splitLines(readFile(dir.file("run/log.csv")));
            std::vector<std::string> want = expectedLog;
            want.erase(want.begin() + crash.generation);   // the lost row
            assert(rows.size() == want.size());
            for (size_t i = 0; i < rows.size(); ++i) {
                const std::vector<std::string> a = splitFields(rows[i]), b = splitFields(want[i]);
                for (size_t f = 0; f < a.size(); ++f)
                    if (f < 9 || f > 11) assert(a[f] == b[f]);
            }
        } else {
            assert(runFingerprint(dir, last) == expected);
        }
    }
}

static void test_resume_exactly() {
    test_stops_and_crashes_resume_exactly("naive", pipeConfig(), false);
    // Engines: this machine and a fake remote (one arithmetic), static and
    // dynamic self-play, and the arena on both.
    for (const char* allocation : {"static", "dynamic"}) {
        Config c = pipeConfig();
        applyOverride(c, "selfplay_engines=local/naive:1, fake/naive:1");
        applyOverride(c, "arena_engines=local/naive:1, fake/naive:1");
        c.selfplayAllocation = allocation;
        test_stops_and_crashes_resume_exactly(std::string("engines_") + allocation, c, true);
    }
    if (blasAvailable()) {
        Config c = pipeConfig();
        c.backend = "blas";
        test_stops_and_crashes_resume_exactly("blas", c, false);
    }
    // metal: self-play's hub, the validation loss's runner and the arena's
    // hubs share the GPU; a few stops and crashes.
    if (metalAvailable()) {
        Config c = pipeConfig();
        c.backend = "metal";
        c.inferenceBatch = 64;
        test_stops_and_crashes_resume_exactly("metal", c, false, false);
    }
}

// Switching on at a commit plays the prologue and notes it; switching off
// at a commit with pending games trains on them (their manifest rows
// stay), then plays as the sequential loop does. The same switches twice
// give the same run.
static void test_switching() {
    auto schedule = [](const Dir& dir, std::ostringstream& on, std::ostringstream& off) {
        Config c = pipeConfig();
        c.pipeline = false;
        applyOverride(c, "selfplay_engines=local/naive:1, local/naive:1");
        start(c, dir)->runUntil(2);
        resume(dir, {"pipeline=on"}, "", &on)->runUntil(4);
        const TrainingState s = stateOf(dir);
        assert(s.pending && s.generation == 4);
        resume(dir, {"pipeline=off"}, "", &off)->runUntil(6);
        assert(!stateOf(dir).pending && readFile(statePath(dir.file("run")))[4] == 5);
    };
    Dir a("switch_a"), b("switch_b");
    std::ostringstream onA, offA, onB, offB;
    schedule(a, onA, offA);
    schedule(b, onB, offB);
    assert(runFingerprint(a, 6) == runFingerprint(b, 6));
    assert(contains(onA.str(), "note: pipeline changed from off to on; from here on self-play plays the best "
                               "network of the last commit, one generation\nbehind training\n"));
    assert(contains(offA.str(), "note: pipeline changed from on to off"));
    // Every generation's self-play rows once, generation 5's from the
    // pipelined commit that played its games.
    const std::vector<std::string> rows = splitLines(readFile(a.file("run/selfplay-engines.csv")));
    assert(rows.size() == 1 + 2 * 6);
    for (int g = 1; g <= 6; ++g) assert(rows[static_cast<size_t>(2 * g - 1)].rfind(std::to_string(g) + ",", 0) == 0);
    assert(splitLines(readFile(a.file("run/log.csv"))).size() == 7);
}

// A pipelined commit's pending games are settled: a self-play key changed
// on resume applies a generation later, and the resume says so. The
// statistics of those games keep the v_resign they were played with.
static void test_changed_keys_on_resume() {
    Dir dir("keys");
    start(pipeConfig(), dir)->runUntil(2);
    const float used = stateOf(dir).pending->resignThreshold;
    std::ostringstream same, changed;
    resume(dir, {}, "", &same);
    assert(!contains(same.str(), "note:"));
    std::unique_ptr<Trainer> t = resume(dir, {"simulations=6", "resign_false_positive=0.5"}, "", &changed);
    assert(contains(changed.str(), "note: generation 3's self-play games were committed before this resume "
                                   "(pipeline); changed self-play keys (simulations, resign_false_positive) "
                                   "apply from generation 4\n"));
    t->runUntil(3);
    char want[32];
    std::snprintf(want, sizeof want, "%.2f", used);
    const std::vector<std::string> row = splitFields(splitLines(readFile(dir.file("run/log.csv"))).back());
    assert(row[0] == "3" && row[16] == want);
}

// Either half failing stops the other and commits nothing: training's
// non-finite weights with self-play in flight (the state stays the last
// commit), and self-play failing under selfplay_fallback = stop with the
// arena in flight; resumed after the failure, the run is the failure-free one.
static void test_failures() {
    {
        Dir dir("nonfinite");
        Config c = pipeConfig();
        std::unique_ptr<Trainer> t = start(c, dir);
        t->runUntil(1);
        const std::string committed = readFile(statePath(dir.file("run")));
        t.reset();
        std::unique_ptr<Trainer> broken = resume(dir, {"lr_schedule=0:3e38", "train_steps=1"});
        bool threw = false;
        try {
            broken->runGeneration();
        } catch (const std::runtime_error& e) {
            threw = contains(e.what(), "last good state kept");
        }
        assert(threw);
        assert(readFile(statePath(dir.file("run"))) == committed);
        assert(!fs::exists(checkpointPath(dir.file("run"), 2)));
    }
    Config c = pipeConfig();
    applyOverride(c, "selfplay_engines=local/naive:1, fake/naive:1");
    applyOverride(c, "arena_engines=local/naive:1, fake/naive:1");
    c.selfplayFallback = "stop";
    Dir plain("stop_plain"), dir("stop");
    start(c, plain, fakeHostsFile(plain))->runUntil(3);
    const std::string hosts = fakeHostsFile(dir);
    std::unique_ptr<Trainer> t = start(c, dir, hosts);
    t->runUntil(1);
    const std::string committed = readFile(statePath(dir.file("run")));
    // The fake remote fails every attempt of iteration 2: generation 3's
    // self-play slice and generation 2's arena slice there, each twice;
    // whichever fails first stops the other.
    distributed_probe::failSlice("fake/naive", 4, distributed_probe::Fail::Exit);
    bool threw = false;
    try {
        t->runGeneration();
    } catch (const std::runtime_error& e) {
        threw = contains(e.what(), "failed twice") && contains(e.what(), "selfplay_fallback = stop");
        if (!threw) std::printf("test_pipeline: unexpected: %s\n", e.what());
    }
    distributed_probe::reset();
    assert(threw);
    assert(readFile(statePath(dir.file("run"))) == committed);
    t.reset();
    resume(dir, {}, hosts)->runUntil(3);
    assert(runFingerprint(dir, 3) == runFingerprint(plain, 3));
}

// Two arithmetics (naive and blas) in dynamic allocation: replaying a
// pipelined run's recorded assignment reproduces it byte for byte, and
// another assignment does not.
static void test_dynamic_replay_of_assignment() {
    if (!blasAvailable()) return;
    Config c = pipeConfig();
    applyOverride(c, "selfplay_engines=local/naive:1, local/blas:1");
    c.selfplayAllocation = "dynamic";
    Dir live("replay_live"), again("replay_again"), other("replay_other");
    start(c, live)->runUntil(3);
    std::vector<std::vector<std::string>> recorded(5);
    for (const std::string& row : splitLines(readFile(live.file("run/selfplay-games.csv")))) {
        const std::vector<std::string> f = splitFields(row);
        if (f[0] == "generation") continue;
        recorded[static_cast<size_t>(std::stoi(f[0]))].push_back(f[2]);
    }
    for (int g = 1; g <= 4; ++g) {
        assert(recorded[static_cast<size_t>(g)].size() == 8);
        distributed_probe::forceAssignment(g, recorded[static_cast<size_t>(g)]);
    }
    start(c, again)->runUntil(3);
    distributed_probe::reset();
    assert(runFingerprint(live, 3) == runFingerprint(again, 3));
    for (int g = 1; g <= 4; ++g) {
        std::vector<std::string> swapped = recorded[static_cast<size_t>(g)];
        for (std::string& e : swapped) e = e == "local/naive" ? "local/blas" : "local/naive";
        distributed_probe::forceAssignment(g, swapped);
    }
    start(c, other)->runUntil(3);
    distributed_probe::reset();
    assert(runFingerprint(live, 3) != runFingerprint(other, 3));
}

// The manifests: self-play rows of the pending generation are written with
// the commit that holds its games, arena rows of the committed one; a
// resume drops what a crash left of an uncommitted iteration.
static void test_manifests_and_pruning() {
    Dir dir("manifests");
    Config c = pipeConfig();
    applyOverride(c, "selfplay_engines=local/naive:1, fake/naive:1");
    applyOverride(c, "arena_engines=local/naive:1, fake/naive:1");
    c.selfplayAllocation = "dynamic";
    const std::string hosts = fakeHostsFile(dir);
    start(c, dir, hosts)->runUntil(2);
    auto generations = [&](const char* file) {
        std::vector<int> out;
        for (const std::string& row : splitLines(readFile(dir.file(std::string("run/") + file))))
            if (row.rfind("generation", 0) != 0) out.push_back(std::stoi(row));
        return out;
    };
    const std::vector<int> games = generations("selfplay-games.csv");
    assert(games.size() == 3 * 8 && games.front() == 1 && games.back() == 3);
    assert(generations("selfplay-engines.csv").back() == 3);
    assert(generations("arena-engines.csv").front() == 0 && generations("arena-engines.csv").back() == 2);
    const std::string selfplay = readFile(dir.file("run/selfplay-games.csv"));
    const std::string arena = readFile(dir.file("run/arena-engines.csv"));
    // What a crash of iteration 3 after its manifests leaves.
    writeFile(dir.file("run/selfplay-games.csv"), selfplay + "4,0,local/naive\n");
    writeFile(dir.file("run/arena-engines.csv"), arena + "3,1,0,2,local/naive,local/naive,\"x\",b,1.0,10,20,\n");
    resume(dir, {}, hosts);
    assert(readFile(dir.file("run/selfplay-games.csv")) == selfplay);
    assert(readFile(dir.file("run/arena-engines.csv")) == arena);
}

// GPU memory: a pipelined run holds self-play's hub, the validation
// runner and the arena's hubs at once, so its up-front check adds them up.
// At the smallest working set where a sequential run's checks pass (each
// phase alone), a pipelined run is refused before writing anything, on the
// default path (backend = metal) and with local/metal engines (the
// coordinator's preflight).
static void test_gpu_memory_checked_together() {
#ifdef AZ_USE_METAL
    if (!metalAvailable()) return;
    // The smallest limit at which `fits` holds, by bisection.
    auto smallest = [](const std::function<void()>& check) {
        auto fits = [&] {
            try {
                check();
            } catch (const std::runtime_error&) {
                return false;
            }
            return true;
        };
        size_t lo = 1, hi = size_t(1) << 40;
        while (hi - lo > 1) {
            const size_t mid = lo + (hi - lo) / 2;
            metal_probe::setWorkingSetLimit(mid);
            (fits() ? hi : lo) = mid;
        }
        metal_probe::setWorkingSetLimit(0);
        return hi;
    };
    auto refused = [](const std::function<void()>& f) {
        try {
            f();
        } catch (const std::runtime_error& e) {
            return contains(e.what(), "lower inference_batch");
        }
        return false;
    };
    for (bool engines : {false, true}) {
        Config c = pipeConfig();
        c.pipeline = false;
        c.backend = engines ? "naive" : "metal";
        if (engines) {
            applyOverride(c, "selfplay_engines=local/metal:1");
            applyOverride(c, "arena_engines=local/metal:1");
        }
        const NetConfig shape = c.netConfig();
        const int b = c.inferenceBatch, waves = arenaWaveNetworks(c);
        const size_t sequential = std::max(smallest([&] { checkMetalInferenceMemory(shape, b, 1, 2); }),
                                           smallest([&] { checkMetalInferenceMemory(shape, b, waves, 1); }));
        Dir on(engines ? "mem_on_engines" : "mem_on"), off(engines ? "mem_off_engines" : "mem_off");
        metal_probe::setWorkingSetLimit(sequential);
        Config pipelined = c;
        pipelined.pipeline = true;
        const bool onRefused = refused([&] { start(pipelined, on); });
        bool offStarted = false;
        try {
            start(c, off);
            offStarted = true;
        } catch (const std::exception& e) {
            std::printf("test_pipeline: unexpected: %s\n", e.what());
        }
        metal_probe::setWorkingSetLimit(0);
        assert(onRefused && offStarted);
        assert(!fs::exists(on.file("run/config.cfg")));
    }
#endif
}

int main() {
    test_gpu_memory_checked_together();
    test_two_phases_at_once();
    test_cancel();
    test_pipelined_run_files();
    test_lag_rule();
    test_switching();
    test_changed_keys_on_resume();
    test_manifests_and_pruning();
    test_failures();
    test_dynamic_replay_of_assignment();
    test_resume_exactly();
    std::printf("test_pipeline: all tests passed\n");
    return 0;
}
