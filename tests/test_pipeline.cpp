// Pipelined generations (pipeline = on): a generation's training and arena
// run beside the next generation's self-play.
//
// First the coordinator with two phases at once: a self-play phase (static
// or dynamic) on one thread and arena waves on another, sharing a fake
// remote, give exactly what each gives alone; neither phase removes the
// other's network from the machine; cancel() stops either.
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
#include "azgo/run_state.hpp"
#include "azgo/selfplay_slice.hpp"

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

int main() {
    test_two_phases_at_once();
    test_cancel();
    std::printf("test_pipeline: all tests passed\n");
    return 0;
}
