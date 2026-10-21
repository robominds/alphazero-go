#include "azgo/selfplay_slice.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include "azgo/backend.hpp"
#include "azgo/board.hpp"
#include "azgo/build_info.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/engines.hpp"
#include "azgo/game.hpp"
#include "azgo/inference_hub.hpp"
#include "azgo/parallel.hpp"
#include "azgo/replay_buffer.hpp"
#include "azgo/rng.hpp"
#include "azgo/run_state.hpp"
#include "azgo/serialize.hpp"
#ifdef __APPLE__
#include <pthread.h>
#include <sys/qos.h>
#endif

namespace azgo {

void setThreadQos(CpuQos qos) {
#ifdef __APPLE__
    if (qos == CpuQos::Utility) pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#else
    (void)qos;
#endif
}

std::vector<GameRecord> playSelfPlayGames(const Network& net, const SelfPlayConfig& sp,
                                          uint64_t seed, int generation, int first, int count,
                                          const EngineRun& run, SliceStats* stats) {
    if (first < 0 || count < 0)
        throw std::invalid_argument("playSelfPlayGames: negative index range");
    const auto start = std::chrono::steady_clock::now();
    std::vector<GameRecord> records(static_cast<size_t>(count));
    long long evaluations = 0;
    if (count > 0) {
        // One evaluator per worker; the network itself is shared read-only.
        // naive, blas: a single-threaded backend and workspace per worker.
        // metal: one batching hub that gathers every worker's leaves into
        // GPU passes; a position's outputs do not depend on its batch, so
        // the games are the same either way (azgo/inference_hub.hpp).
        const int workers = std::max(1, std::min(run.workers, count));
        WorkerEvaluators evaluators(net, run.backend, workers, sp.mcts.batchSize,
                                    run.inferenceBatch);
        // Each worker is one thread for the whole loop: it sets its QoS
        // class once, before its first game.
        std::vector<char> qosSet(static_cast<size_t>(workers), 0);
        // Game i's randomness depends only on i, and its record is stored at
        // index i, so the result does not depend on which thread played it.
        parallelFor(count, workers, [&](int k, int w) {
            if (!qosSet[static_cast<size_t>(w)]) {
                setThreadQos(run.qos);
                qosSet[static_cast<size_t>(w)] = 1;
            }
            const int i = first + k;
            Rng rng(deriveSeed(seed, static_cast<uint64_t>(generation), kPhaseSelfPlay,
                               static_cast<uint64_t>(i)));
            if (run.cancel) {
                CancellableEvaluator ev(evaluators.forWorker(w), *run.cancel);
                records[static_cast<size_t>(k)] = playSelfPlayGame(ev, sp, rng);
            } else {
                Evaluator& ev = evaluators.forWorker(w);
                records[static_cast<size_t>(k)] = playSelfPlayGame(ev, sp, rng);
            }
        });
        evaluations = evaluators.evaluations();
    }
    if (stats) {
        const auto elapsed = std::chrono::steady_clock::now() - start;
        stats->seconds = std::chrono::duration<double>(elapsed).count();
        stats->positions = 0;
        for (const GameRecord& r : records) stats->positions += r.positions();
        stats->evaluations = evaluations;
    }
    return records;
}

} // namespace azgo

// ---- job and result files ---------------------------------------------------

namespace azgo {

namespace {

constexpr uint32_t kJobVersion = 1, kResultVersion = 1;
const char kJobMagic[4] = {'A', 'Z', 'S', 'J'};
const char kResultMagic[4] = {'A', 'Z', 'S', 'R'};

void putChecksum(std::string& bytes) {
    const uint64_t h = fnv1a64(bytes);
    for (int i = 0; i < 8; ++i) bytes += static_cast<char>((h >> (8 * i)) & 0xff);
}

// A little-endian unsigned integer of `size` bytes at `at`.
uint64_t storedUint(const std::string& bytes, size_t at, int size) {
    uint64_t v = 0;
    for (int i = 0; i < size; ++i)
        v |= static_cast<uint64_t>(static_cast<unsigned char>(bytes[at + i])) << (8 * i);
    return v;
}

uint64_t storedChecksum(const std::string& bytes) { return storedUint(bytes, bytes.size() - 8, 8); }

bool readFlag(BinaryReader& r, const char* what) {
    const uint8_t v = r.u8();
    if (v > 1) throw SliceError(std::string("job: bad ") + what + " flag");
    return v == 1;
}

// A network file name in the work directory: no path, nothing to quote.
bool plainFileName(const std::string& name) {
    if (name.empty() || name.size() > 128 || name[0] == '.') return false;
    for (char c : name)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
              c == '_' || c == '.' || c == '-'))
            return false;
    return true;
}

std::string ownerName(int side) { return side == 0 ? "Black" : "White"; }

// The checks of readSliceResult that need only the record and the job.
void verifyRecord(const SelfPlayJob& job, const GameRecord& g, int index) {
    const std::string game = "game " + std::to_string(index) + ": ";
    const int n = g.boardSize, area = n * n;
    if (n != job.boardSize)
        throw SliceError(game + "board size " + std::to_string(n) + ", the job's is " +
                         std::to_string(job.boardSize));
    if (std::memcmp(&g.komi, &job.sp.komi, sizeof g.komi) != 0)
        throw SliceError(game + "komi " + std::to_string(g.komi) + ", the job's is " +
                         std::to_string(job.sp.komi));
    // Replay with the job's move cap: every position, move and the end.
    Game replay(n, g.komi, job.sp.maxMoves);
    std::vector<uint8_t> stones(static_cast<size_t>(area));
    for (int ply = 0; ply < g.positions(); ++ply) {
        if (replay.isTerminal())
            throw SliceError(game + "the game had ended before move " + std::to_string(ply + 1));
        replay.board().exportStones(stones.data());
        if (std::memcmp(stones.data(), g.stonesAt(ply), static_cast<size_t>(area)) != 0)
            throw SliceError(game + "position " + std::to_string(ply) +
                             " differs from the replayed one");
        const int move = g.moves[static_cast<size_t>(ply)];
        bool visited = false;
        for (const auto& av : g.visits[static_cast<size_t>(ply)]) {
            visited = visited || av.first == move;
            // The search only ever visits legal moves.
            if (!replay.isLegal(av.first))
                throw SliceError(game + "a visit to an illegal move (" +
                                 actionToString(av.first, n) + ") at position " +
                                 std::to_string(ply));
        }
        const std::string what =
            game + "move " + std::to_string(ply + 1) + " (" + actionToString(move, n) + ")";
        if (!visited) throw SliceError(what + " is not among the visited moves");
        if (!replay.isLegal(move)) throw SliceError(what + " is illegal");
        replay.play(move);
    }
    std::vector<int8_t> owner(static_cast<size_t>(area));
    replay.board().ownership(owner.data());
    if (owner != g.finalOwnership)
        throw SliceError(game + "final ownership differs from the replayed board's");

    const int toMove = g.positions() % 2;   // 0 Black, 1 White: who would move next
    if (g.resigned) {
        if (replay.isTerminal()) throw SliceError(game + "resigned a game that had already ended");
        const float loss = toMove == 0 ? -1.0f : 1.0f;
        if (g.resultForBlack != loss)
            throw SliceError(game + "resigned, but the result is not a loss for " +
                             ownerName(toMove) + ", who resigned");
    } else {
        if (!replay.isTerminal()) throw SliceError(game + "neither finished nor resigned");
        const float result = replay.resultForBlack();
        if (std::memcmp(&result, &g.resultForBlack, sizeof result) != 0)
            throw SliceError(game + "result " + std::to_string(g.resultForBlack) +
                             ", the replayed board gives " + std::to_string(result));
    }

    if (!job.sp.resignation) {
        if (g.resigned || g.resignDisabled || !g.resignLows[0].empty() || !g.resignLows[1].empty())
            throw SliceError(game + "resignation is off, but the game has resignation flags "
                                    "or resign lows");
        return;
    }
    if (g.resigned && g.resignDisabled)
        throw SliceError(game + "resigned with resignation disabled");
    // Whether resignation was off for this game is its first draw from its
    // own seed (playSelfPlayGame): recompute it.
    Rng rng(deriveSeed(job.seed, static_cast<uint64_t>(job.generation), kPhaseSelfPlay,
                       static_cast<uint64_t>(index)));
    if (g.resignDisabled != (rng.uniform() < job.sp.resignDisabledFraction))
        throw SliceError(game + "resign-disabled flag does not match the game's seed");
    for (int side = 0; side < 2; ++side) {
        const ResignLows& lows = g.resignLows[side];
        for (size_t k = 0; k < lows.size(); ++k) {
            const auto [ply, value] = lows[k];
            const bool falling =
                k == 0 || (ply > lows[k - 1].first && value < lows[k - 1].second);
            // Searched positions only: the last position too only when the
            // side to move resigned there (a game that played on ended at
            // its last move, with no search after it).
            if (ply == g.positions() && !g.resigned)
                throw SliceError(game + "a resign low after the last move (" + ownerName(side) +
                                 ", ply " + std::to_string(ply) + ")");
            const bool ok = ply >= 1 && ply <= g.positions() && ply % 2 == side &&
                            value >= -1.0f && value <= 1.0f && falling;
            if (!ok)
                throw SliceError(game + "resign lows of " + ownerName(side) +
                                 " are not record lows (entry " + std::to_string(k) + ")");
        }
    }
    if (g.resignDisabled) return;
    // Resignation was on for this game: it resigned at the first value
    // below v_resign, which is then the resigning side's last low, at the
    // position it resigned in; no other low may be below v_resign.
    const float threshold = job.sp.resignThreshold;
    if (g.resigned) {
        const ResignLows& lows = g.resignLows[toMove];
        if (lows.empty() || lows.back().first != g.positions() || !(lows.back().second < threshold))
            throw SliceError(game + "resigned without a value below v_resign");
    }
    for (int side = 0; side < 2; ++side) {
        const ResignLows& lows = g.resignLows[side];
        const size_t checked = g.resigned && side == toMove ? lows.size() - 1 : lows.size();
        for (size_t k = 0; k < checked; ++k)
            if (lows[k].second < threshold)
                throw SliceError(game + ownerName(side) + " went below v_resign at ply " +
                                 std::to_string(lows[k].first) + " without resigning");
    }
}

} // namespace

uint64_t fnv1a64(const std::string& bytes) {
    uint64_t hash = 0xcbf29ce484222325ull;   // as fileHash: offset basis, then xor and multiply
    for (char c : bytes) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 0x100000001b3ull;
    }
    return hash;
}

std::string serializeJob(const SelfPlayJob& job) {
    std::ostringstream os(std::ios::binary);
    BinaryWriter w(os);
    w.bytes(kJobMagic, 4);
    w.u32(kJobVersion);
    w.u64(job.seed);
    w.i32(job.generation);
    w.i32(job.first);
    w.i32(job.count);
    w.i32(job.boardSize);
    const MctsConfig& m = job.sp.mcts;
    w.i32(m.simulations);
    w.i32(m.batchSize);
    w.f32(m.cPuct);
    w.u8(m.addNoise ? 1 : 0);
    w.f32(m.dirichletScale);
    w.f32(m.noiseFraction);
    w.u8(m.randomSymmetry ? 1 : 0);
    w.i32(job.sp.temperatureMoves);
    w.f32(job.sp.komi);
    w.i32(job.sp.maxMoves);
    w.u8(job.sp.resignation ? 1 : 0);
    w.f32(job.sp.resignThreshold);
    w.f32(job.sp.resignDisabledFraction);
    w.str(job.networkFile);
    w.u64(job.networkHash);
    w.str(job.buildId);
    w.i32(job.inferenceBatch);
    std::string bytes = os.str();
    putChecksum(bytes);
    return bytes;
}

SelfPlayJob parseJob(const std::string& bytes) {
    if (bytes.size() < 16)
        throw SliceError("job truncated (" + std::to_string(bytes.size()) + " bytes)");
    if (bytes.compare(0, 4, kJobMagic, 4) != 0)
        throw SliceError("job: wrong magic (not a self-play job)");
    if (storedChecksum(bytes) != fnv1a64(bytes.substr(0, bytes.size() - 8)))
        throw SliceError("job: checksum mismatch (damaged or truncated)");
    const uint32_t version = static_cast<uint32_t>(storedUint(bytes, 4, 4));
    if (version != kJobVersion)
        throw SliceError("job: version " + std::to_string(version) + " (expected " +
                         std::to_string(kJobVersion) + ")");
    std::istringstream is(bytes.substr(8, bytes.size() - 16), std::ios::binary);
    BinaryReader r(is);
    SelfPlayJob job;
    try {
        job.seed = r.u64();
        job.generation = r.i32();
        job.first = r.i32();
        job.count = r.i32();
        job.boardSize = r.i32();
        MctsConfig& m = job.sp.mcts;
        m.simulations = r.i32();
        m.batchSize = r.i32();
        m.cPuct = r.f32();
        m.addNoise = readFlag(r, "add_noise");
        m.dirichletScale = r.f32();
        m.noiseFraction = r.f32();
        m.randomSymmetry = readFlag(r, "random_symmetry");
        job.sp.temperatureMoves = r.i32();
        job.sp.komi = r.f32();
        job.sp.maxMoves = r.i32();
        job.sp.resignation = readFlag(r, "resignation");
        job.sp.resignThreshold = r.f32();
        job.sp.resignDisabledFraction = r.f32();
        job.networkFile = r.str();
        job.networkHash = r.u64();
        job.buildId = r.str();
        job.inferenceBatch = r.i32();
    } catch (const SliceError&) {
        throw;
    } catch (const std::runtime_error& e) {
        throw SliceError(std::string("job truncated: ") + e.what());
    }
    if (is.peek() != std::char_traits<char>::eof()) throw SliceError("job: trailing bytes");
    if (job.generation < 0 || job.first < 0 || job.count < 0 || job.count > 1000000 ||
        static_cast<long long>(job.first) + job.count > 1000000)
        throw SliceError("job: bad generation or game range");
    if (job.boardSize < kMinBoardSize || job.boardSize > kMaxBoardSize)
        throw SliceError("job: bad board size " + std::to_string(job.boardSize));
    if (job.sp.mcts.simulations < 1 || job.sp.mcts.batchSize < 1 || job.sp.temperatureMoves < 0)
        throw SliceError("job: bad search settings");
    if (job.sp.maxMoves < 1)
        throw SliceError("job: max_moves not resolved (" + std::to_string(job.sp.maxMoves) + ")");
    if (job.inferenceBatch < 1) throw SliceError("job: bad inference_batch");
    if (!plainFileName(job.networkFile))
        throw SliceError("job: bad network file name '" + job.networkFile + "'");
    if (job.buildId.empty()) throw SliceError("job: no build ID");
    return job;
}

std::string serializeSliceResult(const SliceResult& result, int first) {
    std::ostringstream os(std::ios::binary);
    BinaryWriter w(os);
    w.u64(result.jobHash);
    w.str(result.buildId);
    w.str(result.fingerprint);
    w.u32(static_cast<uint32_t>(result.records.size()));
    for (size_t k = 0; k < result.records.size(); ++k) {
        const GameRecord& g = result.records[k];
        w.i32(first + static_cast<int>(k));
        writeGameRecord(w, g);
        w.u8(g.resignDisabled ? 1 : 0);
        for (const ResignLows& lows : g.resignLows) {
            w.u32(static_cast<uint32_t>(lows.size()));
            for (const auto& [ply, value] : lows) {
                w.i32(ply);
                w.f32(value);
            }
        }
    }
    w.f64(result.stats.seconds);
    w.u64(static_cast<uint64_t>(result.stats.positions));
    w.u64(static_cast<uint64_t>(result.stats.evaluations));
    const std::string body = os.str();
    // Header: magic, version and the whole file's length (body + 16 + 8).
    std::ostringstream hs(std::ios::binary);
    BinaryWriter h(hs);
    h.bytes(kResultMagic, 4);
    h.u32(kResultVersion);
    h.u64(static_cast<uint64_t>(body.size() + 16 + 8));
    std::string bytes = hs.str() + body;
    putChecksum(bytes);
    return bytes;
}

SliceResult readSliceResult(const std::string& path, const std::string& jobBytes,
                            const std::string& expectedBuild) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw SliceError(path + ": cannot read the slice result");
    const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    // The file's name only: a refusal can end up in the manifest (a
    // fallback's note), which must not depend on where the run lives.
    const std::string where =
        "slice result " + std::filesystem::path(path).filename().string() + ": ";
    const std::string size = std::to_string(bytes.size());
    if (bytes.size() < 24) throw SliceError(where + "truncated (" + size + " bytes)");
    if (bytes.compare(0, 4, kResultMagic, 4) != 0)
        throw SliceError(where + "wrong magic (not a slice result)");
    const uint32_t version = static_cast<uint32_t>(storedUint(bytes, 4, 4));
    if (version != kResultVersion)
        throw SliceError(where + "version " + std::to_string(version) + " (expected " +
                         std::to_string(kResultVersion) + ")");
    const uint64_t length = storedUint(bytes, 8, 8);
    if (bytes.size() < length)
        throw SliceError(where + "truncated (" + size + " of " + std::to_string(length) +
                         " bytes)");
    if (bytes.size() > length)
        throw SliceError(where + "length: " + size + " bytes on disk, " + std::to_string(length) +
                         " declared");
    if (storedChecksum(bytes) != fnv1a64(bytes.substr(0, bytes.size() - 8)))
        throw SliceError(where + "checksum mismatch (damaged in transit or on disk)");

    // Every refusal below names the file; readGameRecord's and BinaryReader's
    // errors (std::runtime_error) become SliceErrors.
    SliceResult result;
    std::istringstream is(bytes.substr(16, bytes.size() - 24), std::ios::binary);
    BinaryReader r(is);
    try {
        result.jobHash = r.u64();
        if (result.jobHash != fnv1a64(jobBytes))
            throw SliceError("answers another job (job hash differs): a stale or foreign result");
        result.buildId = r.str();
        if (result.buildId != expectedBuild)
            throw SliceError("built from another commit (build ID " + result.buildId +
                             ", expected " + expectedBuild + ")");
        result.fingerprint = r.str();
        bool printable = !result.fingerprint.empty() && result.fingerprint.size() <= 1024;
        for (char c : result.fingerprint)
            printable = printable && static_cast<unsigned char>(c) >= 0x20;
        if (!printable)
            throw SliceError("bad engine fingerprint (empty, too long or not printable)");
        const SelfPlayJob job = parseJob(jobBytes);
        const uint32_t count = r.u32();
        if (count != static_cast<uint32_t>(job.count))
            throw SliceError("holds " + std::to_string(count) + " games, the job's slice has " +
                             std::to_string(job.count) + " games");
        long long positions = 0;
        for (uint32_t k = 0; k < count; ++k) {
            const int expected = job.first + static_cast<int>(k);
            const int index = r.i32();
            if (index != expected)
                throw SliceError("record " + std::to_string(k) + " has index " +
                                 std::to_string(index) + ", expected " + std::to_string(expected));
            const std::string game = "game " + std::to_string(index);
            GameRecord g = readGameRecord(r, game);
            const uint8_t disabled = r.u8();
            if (disabled > 1) throw SliceError(game + ": bad resign-disabled flag");
            g.resignDisabled = disabled == 1;
            for (ResignLows& lows : g.resignLows) {
                const uint32_t m = r.u32();
                if (m > static_cast<uint32_t>(g.positions()) + 1)
                    throw SliceError(game + ": more resign lows than positions");
                lows.resize(m);
                for (auto& [ply, value] : lows) {
                    ply = r.i32();
                    value = r.f32();
                }
            }
            verifyRecord(job, g, index);
            positions += g.positions();
            result.records.push_back(std::move(g));
        }
        result.stats.seconds = r.f64();
        result.stats.positions = static_cast<long long>(r.u64());
        result.stats.evaluations = static_cast<long long>(r.u64());
        if (result.stats.positions != positions)
            throw SliceError("positions: the statistics count " +
                             std::to_string(result.stats.positions) + ", the games hold " +
                             std::to_string(positions));
        if (is.peek() != std::char_traits<char>::eof()) throw SliceError("trailing bytes");
    } catch (const std::runtime_error& e) {   // SliceError included
        throw SliceError(where + e.what());
    }
    return result;
}

int runSliceJob(const std::string& jobPath, const std::string& outPath, const std::string& backend,
                int workers, int inferenceBatch, std::ostream& err) {
    try {
        std::ifstream in(jobPath, std::ios::binary);
        if (!in) throw SliceError(jobPath + ": cannot read the job");
        const std::string jobBytes((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
        const SelfPlayJob job = parseJob(jobBytes);
        if (job.buildId != buildId())
            throw SliceError("build ID mismatch: the job comes from " + job.buildId +
                             ", this binary is " + buildId() + " (sync the machines)");
        const std::string netPath =
            (std::filesystem::path(jobPath).parent_path() / job.networkFile).string();
        if (!std::filesystem::exists(netPath)) throw SliceError(netPath + ": network missing");
        if (fileHash(netPath) != job.networkHash)
            throw SliceError(netPath + ": its hash differs from the job's network hash");
        checkBackendAvailable(backend);
        const Network net = loadNetwork(netPath);
        if (net.config().boardSize != job.boardSize)
            throw SliceError(netPath + " is not a " + std::to_string(job.boardSize) + "x" +
                             std::to_string(job.boardSize) + " network");
        EngineRun run;
        run.backend = backend;
        run.workers = sliceWorkers(backend, workers, job.count, resolveWorkers(0));
        run.inferenceBatch = inferenceBatch > 0 ? inferenceBatch : job.inferenceBatch;
        SliceResult result;
        result.jobHash = fnv1a64(jobBytes);
        result.buildId = buildId();
        result.fingerprint = engineFingerprint(backend);
        result.records = playSelfPlayGames(net, job.sp, job.seed, job.generation, job.first,
                                           job.count, run, &result.stats);
        writeFileAtomically(outPath, serializeSliceResult(result, job.first));
        return 0;
    } catch (const std::exception& e) {
        err << "selfplay: " << e.what() << "\n";
        return 1;
    }
}

} // namespace azgo
