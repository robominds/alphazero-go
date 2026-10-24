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
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>
#include "azgo/build_info.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/config.hpp"
#include "azgo/coordinator.hpp"
#include "azgo/engines.hpp"
#include "azgo/network.hpp"
#include "azgo/selfplay_slice.hpp"

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
    // transport runs the real selfplay binary.
    std::string fake(const std::string& alias, const std::string& extra = "") const {
        return alias + " exe=" + AZGO_SELFPLAY_EXE + " workdir=" + (dir / ("remote_" + alias)).string() + " " +
               extra + "\n";
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

int main() {
    test_local_engines();
    test_fake_remote();
    test_remote_only();
    test_failures();
    test_silent_stream();
    test_forced_assignment();
    test_games_manifest();
    std::printf("test_dynamic_selfplay: OK\n");
    return 0;
}
