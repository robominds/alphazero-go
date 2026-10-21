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

int main() {
    test_arena_job_round_trip();
    test_arena_result_round_trip();
    test_arena_result_refusals();
    test_run_arena_job_in_process();
    test_run_arena_job_refusals();
    test_selfplay_exe_plays_arena_jobs();
    std::printf("test_distributed_arena: OK\n");
    return 0;
}
