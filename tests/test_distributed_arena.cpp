// The distributed arena, tested on one machine: the arena job and result
// files and how a result is verified by replaying its games, the selfplay
// app playing an arena job (in process and as the real binary), and the
// coordinator playing an arena wave's slices on several engines, merged
// back by game index into exactly the results playMatches gives.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>
#include "azgo/arena.hpp"
#include "azgo/arena_slice.hpp"
#include "azgo/backend.hpp"
#include "azgo/build_info.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/config.hpp"
#include "azgo/coordinator.hpp"
#include "azgo/engines.hpp"
#include "azgo/game.hpp"
#include "azgo/network.hpp"
#include "azgo/run_state.hpp"
#include "azgo/selfplay_slice.hpp"
#include "azgo/subprocess.hpp"

using namespace azgo;
namespace fs = std::filesystem;

static bool contains(const std::string& s, const std::string& part) {
    return s.find(part) != std::string::npos;
}

static std::string readText(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void writeText(const std::string& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << bytes;
    assert(out);
}

static NetConfig tinyNet() {
    NetConfig c;
    c.boardSize = 5;
    c.history = 2;
    c.blocks = 1;
    c.filters = 4;
    c.valueHidden = 8;
    return c;
}

static ArenaConfig tinyArena() {
    ArenaConfig c;
    c.simulations = 8;
    c.mctsBatch = 4;
    c.cPuct = 1.25f;
    c.temperatureMoves = 2;
    c.komi = 7.5f;
    c.maxMoves = 50;
    return c;
}

static bool sameGames(const std::vector<ArenaGameResult>& x, const std::vector<ArenaGameResult>& y) {
    if (x.size() != y.size()) return false;
    for (size_t k = 0; k < x.size(); ++k)
        if (std::memcmp(&x[k].scoreA, &y[k].scoreA, sizeof(double)) != 0 || x[k].moves != y[k].moves)
            return false;
    return true;
}

static bool sameResult(const MatchResult& x, const MatchResult& y) {
    return x.games == y.games && x.scoreA == y.scoreA && x.winsA == y.winsA && x.winsB == y.winsB &&
           x.draws == y.draws && x.distinctGames == y.distinctGames;
}

// A directory with two networks saved as a remote would hold them, and an
// arena job over them: games 3-5 of a match a vs b, a vs a bot (a with
// overrides), and b vs a with random symmetry on one side.
struct ArenaWork {
    fs::path dir;
    std::string jobPath, outPath, jobBytes;
    ArenaJob job;
    std::unique_ptr<Network> a, b;
    explicit ArenaWork(const std::string& name)
        : dir(fs::temp_directory_path() / ("azgo_test_arena_work_" + std::to_string(getpid()) + "_" + name)) {
        fs::remove_all(dir);
        fs::create_directories(dir);
        a = std::make_unique<Network>(tinyNet(), 41);
        b = std::make_unique<Network>(tinyNet(), 42);
        saveNetwork(*a, (dir / "net_a.azn").string());
        saveNetwork(*b, (dir / "net_b.azn").string());
        job.boardSize = 5;
        job.cfg = tinyArena();
        job.networks = {{"net_a.azn", fileHash((dir / "net_a.azn").string())},
                        {"net_b.azn", fileHash((dir / "net_b.azn").string())}};
        ArenaJobPlayer pa{0}, pb{1}, bot{-1, 6}, special{0, 12, false, 2.0f, 2}, symmetric{1, 0, true};
        job.ranges = {{pa, pb, 501, 3, 3}, {special, bot, 502, 0, 2}, {symmetric, pa, 503, 1, 1}};
        job.buildId = buildId();
        job.inferenceBatch = 64;
        writeJob();
        outPath = (dir / "arena.result").string();
    }
    void writeJob() {
        jobBytes = serializeArenaJob(job);
        jobPath = (dir / "arena_g0002_w1_0_a1.job").string();
        writeText(jobPath, jobBytes);
    }
    // The job's ranges with the in-memory networks.
    std::vector<ArenaGameRange> ranges() const {
        auto player = [&](const ArenaJobPlayer& p) {
            ArenaPlayer out;
            out.net = p.network == 0 ? a.get() : p.network == 1 ? b.get() : nullptr;
            out.simulations = p.simulations;
            out.randomSymmetry = p.randomSymmetry;
            out.cPuct = p.cPuct;
            out.mctsBatch = p.mctsBatch;
            return out;
        };
        std::vector<ArenaGameRange> out;
        for (const ArenaJobRange& r : job.ranges) out.push_back({player(r.a), player(r.b), r.seed, r.first, r.count});
        return out;
    }
    ~ArenaWork() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

// ---- job and result files --------------------------------------------------------

static std::string jobError(const std::string& bytes) {
    try {
        parseArenaJob(bytes);
    } catch (const SliceError& e) {
        return e.what();
    }
    return "accepted";
}

static void test_arena_job_round_trip() {
    ArenaWork work("job");
    ArenaJob job = work.job;
    job.cfg.randomSymmetry = true;
    job.cfg.komi = -0.5f;
    const std::string bytes = serializeArenaJob(job);
    assert(bytes.compare(0, 4, "AZAJ") == 0 && isArenaJob(bytes) && !isArenaJob("AZSJ" + bytes.substr(4)));
    const ArenaJob back = parseArenaJob(bytes);
    assert(back.boardSize == job.boardSize && back.cfg.simulations == job.cfg.simulations &&
           back.cfg.mctsBatch == job.cfg.mctsBatch && back.cfg.temperatureMoves == job.cfg.temperatureMoves &&
           back.cfg.maxMoves == job.cfg.maxMoves && back.cfg.randomSymmetry);
    assert(std::memcmp(&back.cfg.cPuct, &job.cfg.cPuct, 4) == 0 && std::memcmp(&back.cfg.komi, &job.cfg.komi, 4) == 0);
    assert(back.networks.size() == 2 && back.networks[1].file == "net_b.azn" &&
           back.networks[1].hash == job.networks[1].hash);
    assert(back.ranges.size() == 3 && back.games() == 6);
    const ArenaJobRange& r = back.ranges[1];
    assert(r.a.network == 0 && r.a.simulations == 12 && r.a.mctsBatch == 2 && r.a.cPuct == 2.0f);
    assert(r.b.network == -1 && r.b.simulations == 6 && r.seed == 502 && r.first == 0 && r.count == 2);
    assert(back.ranges[2].a.randomSymmetry && !back.ranges[2].b.randomSymmetry);
    assert(back.buildId == job.buildId && back.inferenceBatch == 64);
    assert(serializeArenaJob(back) == bytes);

    // A damaged or impossible job is refused, naming the defect.
    assert(contains(jobError("AZSJ" + bytes.substr(4)), "magic"));
    assert(contains(jobError(bytes.substr(0, 10)), "truncated"));
    std::string flipped = bytes;
    flipped[30] ^= 1;
    assert(contains(jobError(flipped), "checksum"));
    auto variant = [&](const std::function<void(ArenaJob&)>& change) {
        ArenaJob j = work.job;
        change(j);
        return jobError(serializeArenaJob(j));
    };
    assert(contains(variant([](ArenaJob& j) { j.cfg.maxMoves = 0; }), "max_moves"));
    assert(contains(variant([](ArenaJob& j) { j.cfg.simulations = 0; }), "search settings"));
    assert(contains(variant([](ArenaJob& j) { j.cfg.cPuct = 0; }), "search settings"));
    assert(contains(variant([](ArenaJob& j) { j.boardSize = 1; }), "board size"));
    assert(contains(variant([](ArenaJob& j) { j.networks[0].file = "../net.azn"; }), "file name"));
    assert(contains(variant([](ArenaJob& j) { j.ranges[0].b.network = 2; }), "not one of the job's 2"));
    assert(contains(variant([](ArenaJob& j) { j.ranges[1].a.network = -1; }), "two bots"));
    assert(contains(variant([](ArenaJob& j) { j.ranges[0].first = -1; }), "game range"));
    assert(contains(variant([](ArenaJob& j) { j.ranges[0].count = 2147483647; }), "game range"));
    assert(contains(variant([](ArenaJob& j) { j.buildId.clear(); }), "build"));
    assert(contains(variant([](ArenaJob& j) { j.inferenceBatch = 0; }), "inference_batch"));
}

// The games of the work's job, played here.
static ArenaSliceResult playedResult(const ArenaWork& work) {
    ArenaSliceResult r;
    r.jobHash = fnv1a64(work.jobBytes);
    r.buildId = work.job.buildId;
    r.fingerprint = "naive; test cpu; test os";
    r.games = playArenaGames(work.ranges(), work.job.cfg, EngineRun{"naive", 2, 64}, &r.stats);
    return r;
}

static void test_arena_result_round_trip() {
    ArenaWork work("result");
    const ArenaSliceResult result = playedResult(work);
    const std::string bytes = serializeArenaResult(result);
    assert(bytes.compare(0, 4, "AZAR") == 0);
    writeText(work.outPath, bytes);
    const ArenaSliceResult back = readArenaResult(work.outPath, work.jobBytes, work.job.buildId);
    assert(back.jobHash == result.jobHash && back.buildId == result.buildId &&
           back.fingerprint == result.fingerprint);
    assert(sameGames(back.games, result.games) && back.games.size() == 6);
    assert(back.stats.positions == result.stats.positions && back.stats.evaluations == result.stats.evaluations &&
           std::memcmp(&back.stats.seconds, &result.stats.seconds, 8) == 0);
}

// Every way an arena result can be wrong, stale, truncated or foreign is
// refused with a SliceError naming the defect; the games are replayed.
static void test_arena_result_refusals() {
    ArenaWork work("refusals");
    const ArenaSliceResult good = playedResult(work);
    const std::string goodBytes = serializeArenaResult(good);
    auto errorOf = [&](const std::string& bytes, const std::string& jobBytes = "", const std::string& build = "") {
        writeText(work.outPath, bytes);
        try {
            readArenaResult(work.outPath, jobBytes.empty() ? work.jobBytes : jobBytes,
                            build.empty() ? work.job.buildId : build);
        } catch (const SliceError& e) {
            const std::string what = e.what();
            assert(contains(what, "arena result arena.result: ") || contains(what, "cannot read"));
            return what;
        }
        return std::string("accepted");
    };
    auto changed = [&](const std::function<void(ArenaSliceResult&)>& change) {
        ArenaSliceResult r = good;
        change(r);
        return errorOf(serializeArenaResult(r));
    };
    assert(errorOf(goodBytes) == "accepted");
    fs::remove(work.outPath);
    try {
        readArenaResult(work.outPath, work.jobBytes, work.job.buildId);
        assert(false);
    } catch (const SliceError& e) {
        assert(contains(e.what(), "cannot read"));
    }
    assert(contains(errorOf("AZSR" + goodBytes.substr(4)), "magic"));
    assert(contains(errorOf(goodBytes.substr(0, 10)), "truncated"));
    assert(contains(errorOf(goodBytes.substr(0, goodBytes.size() - 1)), "truncated"));
    assert(contains(errorOf(goodBytes + "x"), "length"));
    std::string version = goodBytes;
    version[4] = 2;
    assert(contains(errorOf(version), "version 2"));
    // One flipped byte anywhere past the header is caught by the checksum.
    for (size_t at = 16; at < goodBytes.size(); at += 7) {
        std::string flipped = goodBytes;
        flipped[at] ^= 0x20;
        assert(contains(errorOf(flipped), "checksum"));
    }
    ArenaJob other = work.job;
    other.ranges[0].seed = 999;
    assert(contains(errorOf(goodBytes, serializeArenaJob(other)), "another job"));
    assert(contains(errorOf(goodBytes, "", "0000000000000000000000000000000000000000"), "another commit"));
    assert(contains(changed([](ArenaSliceResult& r) { r.fingerprint = "bad\nfingerprint"; }), "fingerprint"));
    assert(contains(changed([](ArenaSliceResult& r) { r.games.pop_back(); }), "holds 5 games"));
    // Games that do not replay.
    assert(contains(changed([](ArenaSliceResult& r) { r.games[0].moves[0] = 26; }), "off the board"));
    assert(contains(changed([](ArenaSliceResult& r) { r.games[0].moves[0] = -1; }), "off the board"));
    assert(contains(changed([](ArenaSliceResult& r) {
                        r.games[0].moves[1] = r.games[0].moves[0] == 25 ? 0 : r.games[0].moves[0];
                        if (r.games[0].moves[0] == 25) r.games[0].moves[0] = 0;
                    }),
                    "illegal"));
    // A game that ended by two passes, not at the move cap: one more move
    // comes after its end.
    size_t ended = 0;
    while (good.games[ended].moves.size() >= static_cast<size_t>(work.job.cfg.maxMoves)) ++ended;
    assert(contains(changed([&](ArenaSliceResult& r) {
                        r.games[ended].moves.push_back(25);
                        ++r.stats.positions;
                    }),
                    "had ended before move"));
    assert(contains(changed([](ArenaSliceResult& r) {
                        r.games[2].moves.pop_back();
                        --r.stats.positions;
                    }),
                    "does not end"));
    assert(contains(changed([](ArenaSliceResult& r) { r.games[3].scoreA = 1.0 - r.games[3].scoreA; }), "score"));
    assert(contains(changed([](ArenaSliceResult& r) { r.games[4].scoreA = 0.5; }), "score"));
    assert(contains(changed([](ArenaSliceResult& r) { ++r.stats.positions; }), "positions"));
}

// ---- the selfplay app's arena jobs ------------------------------------------------------

static void test_run_arena_job_in_process() {
    ArenaWork work("inproc");
    std::ostringstream err;
    assert(runJob(work.jobPath, work.outPath, "naive", 2, 0, err) == 0);
    const ArenaSliceResult result = readArenaResult(work.outPath, work.jobBytes, buildId());
    assert(result.fingerprint == engineFingerprint("naive") && result.stats.evaluations > 0);
    const std::vector<ArenaGameResult> direct = playArenaGames(work.ranges(), work.job.cfg, EngineRun{"naive", 1, 8});
    assert(sameGames(result.games, direct));
    assert(runArenaJob(work.jobPath, work.outPath, "naive", 1, 8, err) == 0);
    assert(sameGames(readArenaResult(work.outPath, work.jobBytes, buildId()).games, direct));
}

static void test_run_arena_job_refusals() {
    auto refusal = [](ArenaWork& work, const std::string& backend) {
        std::ostringstream err;
        fs::remove(work.outPath);
        const int status = runJob(work.jobPath, work.outPath, backend, 1, 0, err);
        assert(status == 1 && !fs::exists(work.outPath));
        return err.str();
    };
    {
        ArenaWork work("hash");
        work.job.networks[1].hash ^= 1;
        work.writeJob();
        assert(contains(refusal(work, "naive"), "hash"));
    }
    {
        ArenaWork work("build");
        work.job.buildId = "0000000000000000000000000000000000000000";
        work.writeJob();
        const std::string msg = refusal(work, "naive");
        assert(contains(msg, "build") && contains(msg, buildId()));
    }
    {
        ArenaWork work("missing");
        fs::remove(work.dir / "net_b.azn");
        assert(contains(refusal(work, "naive"), "net_b.azn"));
    }
    {
        ArenaWork work("badjob");
        writeText(work.jobPath, work.jobBytes.substr(0, 20));
        assert(contains(refusal(work, "naive"), "arena job"));
    }
    {
        // A player playMatch would refuse: an mcts batch above the inference batch.
        ArenaWork work("batch");
        work.job.ranges[1].a.mctsBatch = 128;
        work.writeJob();
        assert(contains(refusal(work, "naive"), "inference batch"));
    }
    for (const char* backend : {"blas", "metal", "cuda"}) {
        try {
            checkBackendAvailable(backend);
            continue;   // available here: nothing to refuse
        } catch (const std::invalid_argument&) {
        }
        ArenaWork work(std::string("unavailable_") + backend);
        assert(contains(refusal(work, backend), "not available"));
    }
}

// The real binary answers an arena job as the in-process call does,
// bitwise; it tells the kinds of job apart by their magic.
static void test_selfplay_exe_plays_arena_jobs() {
    ArenaWork work("exe");
    const std::string log = (work.dir / "selfplay.log").string();
    assert(runProcess({AZGO_SELFPLAY_EXE, "--job", work.jobPath, "--out", work.outPath, "--backend", "naive",
                       "--workers", "3", "--inference-batch", "16"},
                      log, 300)
               .exitCode == 0);
    const ArenaSliceResult viaExe = readArenaResult(work.outPath, work.jobBytes, buildId());
    const std::string inProcessOut = (work.dir / "inproc.result").string();
    std::ostringstream err;
    assert(runJob(work.jobPath, inProcessOut, "naive", 1, 0, err) == 0);
    const ArenaSliceResult inProcess = readArenaResult(inProcessOut, work.jobBytes, buildId());
    assert(sameGames(viaExe.games, inProcess.games) && viaExe.fingerprint == inProcess.fingerprint);
    assert(viaExe.stats.positions == inProcess.stats.positions &&
           viaExe.stats.evaluations == inProcess.stats.evaluations);
}

// ---- the coordinator plays arena waves ----------------------------------------------

static Config arenaRunConfig(const std::string& engines) {
    Config c;
    c.boardSize = 5;
    c.history = 2;
    c.blocks = 1;
    c.filters = 4;
    c.valueHidden = 8;
    c.mctsBatch = 4;
    c.inferenceBatch = 64;
    applyOverride(c, "arena_engines=" + engines);
    return c;
}

// A wave as a round plays one: the new network a against b, a bot, and c
// with a simulations override and one-sided random symmetry.
static std::vector<ArenaMatchSpec> roundWave(const Network& a, const Network& b, const Network& c) {
    return {
        {ArenaPlayer::network(a), ArenaPlayer::network(b), 6, 701},
        {ArenaPlayer::network(a), ArenaPlayer::bot(6), 4, 702},
        {ArenaPlayer::network(a, 0, true), ArenaPlayer::network(c, 12), 4, 703},
    };
}

struct Dir {
    fs::path path;
    explicit Dir(const std::string& name)
        : path(fs::temp_directory_path() / ("azgo_test_arena_coord_" + std::to_string(getpid()) + "_" + name)) {
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~Dir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string file(const std::string& rel) const { return (path / rel).string(); }
};

// "fake" has no ssh: its slices run the selfplay binary here (subprocess
// transport), or on the calling thread with the in-process probe.
static HostsFile fakeHosts(const Dir& dir) {
    return parseHostsText("fake exe=" + std::string(AZGO_SELFPLAY_EXE) + " workdir=" + dir.file("remote") + "\n",
                          "hosts");
}

static std::vector<std::string> filesIn(const std::string& dir) {
    std::vector<std::string> names;
    std::error_code ec;
    for (const auto& entry : fs::recursive_directory_iterator(dir, ec))
        if (entry.is_regular_file()) names.push_back(entry.path().filename().string());
    std::sort(names.begin(), names.end());
    return names;
}

// Every engine here is naive on this machine, the same arithmetic: a wave
// split between local engines, or between this process and a fake remote,
// gives exactly playMatches' results. The reports name the slices and the
// wave; a remote's work files are gone after the wave, and only the
// networks its slices needed were sent to it.
static void test_play_arena_equals_play_matches() {
    const Network a(tinyNet(), 61), b(tinyNet(), 62), c(tinyNet(), 63);
    ArenaConfig cfg = tinyArena();
    cfg.maxMoves = 0;   // as a run's: resolved by each engine
    const std::vector<ArenaMatchSpec> wave = roundWave(a, b, c);
    const std::vector<MatchResult> expected = playMatches(wave, cfg, 3, "naive", 64);
    for (bool inProcess : {false, true})
        for (const std::string& engines : {std::string("local/naive:1"), std::string("local/naive:2, local/naive:1"),
                                           std::string("local/naive:1, fake/naive:1"),
                                           std::string("fake/naive:3, local/naive:1, fake/naive:1")}) {
            if (inProcess && engines.find("fake") == std::string::npos) continue;
            distributed_probe::useInProcessTransport(inProcess);
            Dir dir("equal");
            Coordinator coordinator(arenaRunConfig(engines), dir.file("run"), fakeHosts(dir), nullptr);
            coordinator.preflight();
            coordinator.beginArena(7);
            const std::vector<MatchResult> got = coordinator.playArena(wave, cfg);
            assert(got.size() == wave.size());
            for (size_t i = 0; i < wave.size(); ++i) assert(sameResult(got[i], expected[i]));
            // One report per slice, wave 1, covering the wave's 14 games in order.
            const std::vector<SliceReport>& reports = coordinator.arenaReports();
            const std::vector<EngineSpec> list = parseEngineList(engines);
            assert(reports.size() == list.size());
            int next = 0;
            for (size_t k = 0; k < reports.size(); ++k) {
                assert(reports[k].wave == 1 && reports[k].slice.first == next);
                assert(reports[k].planned == list[k].name() && reports[k].engine == list[k].name());
                assert(reports[k].fingerprint == engineFingerprint("naive") && reports[k].build == shortBuildId());
                assert(reports[k].note.empty() && reports[k].stats.positions > 0 && reports[k].stats.evaluations > 0);
                next += reports[k].slice.count;
            }
            assert(next == 14);
            assert(!fs::exists(dir.file("run/arena-work")));
            if (engines.find("fake") != std::string::npos) {
                // Left on the remote: the networks of its slices, until the
                // coordinator goes. In "local/naive:1, fake/naive:1" the fake
                // plays wave games 7-13: the end of match 1 (a against a bot)
                // and match 2 (a, c), so two networks. In "fake/naive:3,
                // local/naive:1, fake/naive:1" its slices are 0-7 (a, b)
                // and 11-13 (a, c): all three.
                const std::vector<std::string> left = filesIn(dir.file("remote"));
                for (const std::string& f : left) assert(f.rfind("net_", 0) == 0);
                assert(left.size() == (list.size() == 2 ? 2u : 3u));
            }
            // A second wave of the generation is wave 2; a new generation starts over.
            coordinator.playArena({wave[1]}, cfg);
            assert(coordinator.arenaReports().size() == 2 * list.size() &&
                   coordinator.arenaReports().back().wave == 2);
            coordinator.beginArena(8);
            assert(coordinator.arenaReports().empty());
        }
    distributed_probe::reset();
}

// A remote slice that fails twice is played here by the fallback engine,
// with the same bits (the same arithmetic here), and the report and the
// log say so; with selfplay_fallback = stop the wave throws.
static void test_play_arena_failures() {
    const Network a(tinyNet(), 61), b(tinyNet(), 62), c(tinyNet(), 63);
    const ArenaConfig cfg = tinyArena();
    const std::vector<ArenaMatchSpec> wave = roundWave(a, b, c);
    const std::vector<MatchResult> expected = playMatches(wave, cfg, 3, "naive", 64);
    distributed_probe::useInProcessTransport(true);
    {
        Dir dir("fallback");
        std::ostringstream log;
        Coordinator coordinator(arenaRunConfig("local/naive:1, fake/naive:1"), dir.file("run"), fakeHosts(dir), &log);
        coordinator.preflight();
        coordinator.beginArena(3);
        distributed_probe::failSlice("fake/naive", 2, distributed_probe::Fail::Exit);
        const std::vector<MatchResult> got = coordinator.playArena(wave, cfg);
        for (size_t i = 0; i < wave.size(); ++i) assert(sameResult(got[i], expected[i]));
        const SliceReport& r = coordinator.arenaReports()[1];
        assert(r.planned == "fake/naive" && r.engine == "local/naive" && r.note == "fallback: fake: selfplay exit 255");
        assert(contains(log.str(), "note: generation 3: arena wave 1 games 7-13 (fake/naive) played by local/naive: "
                                   "fake: selfplay exit 255; from here on results differ"));
        // The failed attempts' log stays.
        assert(fs::exists(dir.file("run/arena-work/gen_0003_w1/fake_naive.log")));
        assert(!fs::exists(dir.file("run/arena-work/gen_0003_w1/networks")));
    }
    {
        // One failure is retried on the same engine: nothing to note.
        Dir dir("retry");
        Coordinator coordinator(arenaRunConfig("fake/naive:1"), dir.file("run"), fakeHosts(dir), nullptr);
        coordinator.preflight();
        coordinator.beginArena(3);
        distributed_probe::failSlice("fake/naive", 1, distributed_probe::Fail::Corrupt);
        const std::vector<MatchResult> got = coordinator.playArena(wave, cfg);
        for (size_t i = 0; i < wave.size(); ++i) assert(sameResult(got[i], expected[i]));
        assert(coordinator.arenaReports()[0].engine == "fake/naive" && coordinator.arenaReports()[0].note.empty());
    }
    {
        Dir dir("stop");
        Config stop = arenaRunConfig("local/naive:1, fake/naive:1");
        stop.selfplayFallback = "stop";
        Coordinator coordinator(stop, dir.file("run"), fakeHosts(dir), nullptr);
        coordinator.preflight();
        coordinator.beginArena(3);
        distributed_probe::failSlice("fake/naive", 2, distributed_probe::Fail::Exit);
        bool threw = false;
        try {
            coordinator.playArena(wave, cfg);
        } catch (const SliceError& e) {
            threw = contains(e.what(), "fake/naive failed twice") && contains(e.what(), "stop");
        }
        assert(threw && coordinator.arenaReports().empty());
    }
    distributed_probe::reset();
}

// The arena manifest: one row per slice, its wave and wave indices; rows of
// generations g and later go when g is written again or pruned. The summary
// line is self-play's with its own label.
static void test_arena_manifest() {
    Dir dir("manifest");
    SliceReport r;
    r.slice = {0, 0, 22};
    r.planned = "pc/cuda";
    r.engine = "local/metal";
    r.fingerprint = "metal; chip, os";
    r.build = "1a2b3c4";
    r.note = "fallback: pc: selfplay exit 255";
    r.stats = {301.25, 3652, 2913024};
    r.wave = 2;
    writeArenaManifest(dir.path.string(), 4, {r});
    r.wave = 1;
    writeArenaManifest(dir.path.string(), 5, {r, r});
    const std::string header =
        "generation,wave,first,count,planned,engine,fingerprint,build,seconds,positions,evaluations,note\n";
    const std::string row4 = "4,2,0,22,pc/cuda,local/metal,\"metal; chip, os\",1a2b3c4,301.2,3652,2913024,"
                             "fallback: pc: selfplay exit 255\n";
    const std::string row5 = "5,1,0,22,pc/cuda,local/metal,\"metal; chip, os\",1a2b3c4,301.2,3652,2913024,"
                             "fallback: pc: selfplay exit 255\n";
    assert(readText(arenaManifestPath(dir.path.string())) == header + row4 + row5 + row5);
    writeArenaManifest(dir.path.string(), 5, {});
    assert(readText(arenaManifestPath(dir.path.string())) == header + row4);
    pruneArenaManifest(dir.path.string(), 4);
    assert(readText(arenaManifestPath(dir.path.string())) == header);
    assert(contains(arenaManifestPath("x"), "arena-engines.csv"));
    r.stats = {10.0, 100, 20000};
    assert(enginesLine({r}, "arena engines") == "arena engines: local/metal 22 10s 2.0k ev/s");
    assert(enginesLine({r}) == "selfplay engines: local/metal 22 10s 2.0k ev/s");
}

int main() {
    test_arena_job_round_trip();
    test_arena_result_round_trip();
    test_arena_result_refusals();
    test_run_arena_job_in_process();
    test_run_arena_job_refusals();
    test_selfplay_exe_plays_arena_jobs();
    test_play_arena_equals_play_matches();
    test_play_arena_failures();
    test_arena_manifest();
    std::printf("test_distributed_arena: OK\n");
    return 0;
}
