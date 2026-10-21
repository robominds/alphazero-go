#include "azgo/arena_slice.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include "azgo/backend.hpp"
#include "azgo/board.hpp"
#include "azgo/build_info.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/engines.hpp"
#include "azgo/game.hpp"
#include "azgo/parallel.hpp"
#include "azgo/run_state.hpp"
#include "azgo/serialize.hpp"
#include "slice_files.hpp"

namespace azgo {

namespace {

using slice_files::plainFileName;
using slice_files::putChecksum;
using slice_files::storedChecksum;
using slice_files::storedUint;

constexpr uint32_t kArenaJobVersion = 1, kArenaResultVersion = 1;
const char kArenaJobMagic[4] = {'A', 'Z', 'A', 'J'};
const char kArenaResultMagic[4] = {'A', 'Z', 'A', 'R'};
constexpr int kMaxJobNetworks = 64;        // a wave holds a few; far more is a damaged file
constexpr int kMaxSimulations = 100000;   // the arena's own bound (playMatch)
constexpr int kMaxGames = 1000000;

bool readFlag(BinaryReader& r, const char* what) {
    const uint8_t v = r.u8();
    if (v > 1) throw SliceError(std::string("arena job: bad ") + what + " flag");
    return v == 1;
}

void writePlayer(BinaryWriter& w, const ArenaJobPlayer& p) {
    w.i32(p.network);
    w.i32(p.simulations);
    w.u8(p.randomSymmetry ? 1 : 0);
    w.f32(p.cPuct);
    w.i32(p.mctsBatch);
}

ArenaJobPlayer readPlayer(BinaryReader& r) {
    ArenaJobPlayer p;
    p.network = r.i32();
    p.simulations = r.i32();
    p.randomSymmetry = readFlag(r, "random_symmetry");
    p.cPuct = r.f32();
    p.mctsBatch = r.i32();
    return p;
}

// The checks of readArenaResult that replay one game: game `index` of its
// match (A Black when even), on the job's board, komi and move cap.
void verifyArenaGame(const ArenaJob& job, int index, const ArenaGameResult& g, const std::string& game) {
    const int n = job.boardSize;
    Game replay(n, job.cfg.komi, job.cfg.maxMoves);
    for (size_t k = 0; k < g.moves.size(); ++k) {
        const int move = g.moves[k];
        const std::string what = game + "move " + std::to_string(k + 1);
        if (replay.isTerminal()) throw SliceError(game + "the game had ended before move " + std::to_string(k + 1));
        if (move < 0 || move > n * n) throw SliceError(what + " (" + std::to_string(move) + ") is off the board");
        if (!replay.isLegal(move)) throw SliceError(what + " (" + actionToString(move, n) + ") is illegal");
        replay.play(move);
    }
    if (!replay.isTerminal()) throw SliceError(game + "the game does not end at its last move");
    const double score = matchScore(replay.resultForBlack(), index % 2 == 0);
    if (std::memcmp(&score, &g.scoreA, sizeof score) != 0)
        throw SliceError(game + "score " + std::to_string(g.scoreA) + " for A, the replayed board gives " +
                         std::to_string(score));
}

} // namespace

int ArenaJob::games() const {
    int total = 0;
    for (const ArenaJobRange& r : ranges) total += r.count;
    return total;
}

bool isArenaJob(const std::string& bytes) {
    return bytes.size() >= 4 && bytes.compare(0, 4, kArenaJobMagic, 4) == 0;
}

std::string serializeArenaJob(const ArenaJob& job) {
    std::ostringstream os(std::ios::binary);
    BinaryWriter w(os);
    w.bytes(kArenaJobMagic, 4);
    w.u32(kArenaJobVersion);
    w.i32(job.boardSize);
    w.i32(job.cfg.simulations);
    w.i32(job.cfg.mctsBatch);
    w.f32(job.cfg.cPuct);
    w.i32(job.cfg.temperatureMoves);
    w.f32(job.cfg.komi);
    w.i32(job.cfg.maxMoves);
    w.u8(job.cfg.randomSymmetry ? 1 : 0);
    w.u32(static_cast<uint32_t>(job.networks.size()));
    for (const ArenaJobNetwork& net : job.networks) {
        w.str(net.file);
        w.u64(net.hash);
    }
    w.u32(static_cast<uint32_t>(job.ranges.size()));
    for (const ArenaJobRange& r : job.ranges) {
        writePlayer(w, r.a);
        writePlayer(w, r.b);
        w.u64(r.seed);
        w.i32(r.first);
        w.i32(r.count);
    }
    w.str(job.buildId);
    w.i32(job.inferenceBatch);
    std::string bytes = os.str();
    putChecksum(bytes);
    return bytes;
}

ArenaJob parseArenaJob(const std::string& bytes) {
    if (bytes.size() < 16)
        throw SliceError("arena job truncated (" + std::to_string(bytes.size()) + " bytes)");
    if (!isArenaJob(bytes)) throw SliceError("arena job: wrong magic (not an arena job)");
    if (storedChecksum(bytes) != fnv1a64(bytes.substr(0, bytes.size() - 8)))
        throw SliceError("arena job: checksum mismatch (damaged or truncated)");
    const uint32_t version = static_cast<uint32_t>(storedUint(bytes, 4, 4));
    if (version != kArenaJobVersion)
        throw SliceError("arena job: version " + std::to_string(version) + " (expected " +
                         std::to_string(kArenaJobVersion) + ")");
    std::istringstream is(bytes.substr(8, bytes.size() - 16), std::ios::binary);
    BinaryReader r(is);
    ArenaJob job;
    try {
        job.boardSize = r.i32();
        job.cfg.simulations = r.i32();
        job.cfg.mctsBatch = r.i32();
        job.cfg.cPuct = r.f32();
        job.cfg.temperatureMoves = r.i32();
        job.cfg.komi = r.f32();
        job.cfg.maxMoves = r.i32();
        job.cfg.randomSymmetry = readFlag(r, "random_symmetry");
        const uint32_t networks = r.u32();
        if (networks > static_cast<uint32_t>(kMaxJobNetworks))
            throw SliceError("arena job: " + std::to_string(networks) + " networks");
        for (uint32_t k = 0; k < networks; ++k) {
            ArenaJobNetwork net;
            net.file = r.str();
            net.hash = r.u64();
            job.networks.push_back(net);
        }
        const uint32_t ranges = r.u32();
        if (ranges > static_cast<uint32_t>(kMaxGames)) throw SliceError("arena job: bad range count");
        for (uint32_t k = 0; k < ranges; ++k) {
            ArenaJobRange range;
            range.a = readPlayer(r);
            range.b = readPlayer(r);
            range.seed = r.u64();
            range.first = r.i32();
            range.count = r.i32();
            job.ranges.push_back(range);
        }
        job.buildId = r.str();
        job.inferenceBatch = r.i32();
    } catch (const SliceError&) {
        throw;
    } catch (const std::runtime_error& e) {
        throw SliceError(std::string("arena job truncated: ") + e.what());
    }
    if (is.peek() != std::char_traits<char>::eof()) throw SliceError("arena job: trailing bytes");
    if (job.boardSize < kMinBoardSize || job.boardSize > kMaxBoardSize)
        throw SliceError("arena job: bad board size " + std::to_string(job.boardSize));
    const ArenaConfig& c = job.cfg;
    if (c.simulations < 1 || c.simulations > kMaxSimulations || c.mctsBatch < 1 || c.mctsBatch > 256 ||
        !(c.cPuct > 0.0f && c.cPuct <= 100.0f) || c.temperatureMoves < 0)
        throw SliceError("arena job: bad search settings");
    if (c.maxMoves < 1) throw SliceError("arena job: max_moves not resolved (" + std::to_string(c.maxMoves) + ")");
    for (const ArenaJobNetwork& net : job.networks)
        if (!plainFileName(net.file)) throw SliceError("arena job: bad network file name '" + net.file + "'");
    long long games = 0;
    for (size_t k = 0; k < job.ranges.size(); ++k) {
        const ArenaJobRange& range = job.ranges[k];
        const std::string which = "arena job: range " + std::to_string(k) + ": ";
        for (const ArenaJobPlayer* p : {&range.a, &range.b}) {
            if (p->network < -1 || p->network >= static_cast<int>(job.networks.size()))
                throw SliceError(which + "network " + std::to_string(p->network) + " is not one of the job's " +
                                 std::to_string(job.networks.size()));
            if (p->simulations < 0 || p->simulations > kMaxSimulations)
                throw SliceError(which + "bad simulations " + std::to_string(p->simulations));
        }
        if (range.a.network < 0 && range.b.network < 0) throw SliceError(which + "two bots");
        if (range.first < 0 || range.count < 0 || static_cast<long long>(range.first) + range.count > kMaxGames)
            throw SliceError(which + "bad game range");
        games += range.count;
    }
    if (games > kMaxGames) throw SliceError("arena job: too many games");
    if (job.inferenceBatch < 1) throw SliceError("arena job: bad inference_batch");
    if (job.buildId.empty()) throw SliceError("arena job: no build ID");
    return job;
}

std::string serializeArenaResult(const ArenaSliceResult& result) {
    std::ostringstream os(std::ios::binary);
    BinaryWriter w(os);
    w.u64(result.jobHash);
    w.str(result.buildId);
    w.str(result.fingerprint);
    w.u32(static_cast<uint32_t>(result.games.size()));
    for (const ArenaGameResult& g : result.games) {
        w.u32(static_cast<uint32_t>(g.moves.size()));
        for (int m : g.moves) w.i32(m);
        w.f64(g.scoreA);
    }
    w.f64(result.stats.seconds);
    w.u64(static_cast<uint64_t>(result.stats.positions));
    w.u64(static_cast<uint64_t>(result.stats.evaluations));
    const std::string body = os.str();
    // Header: magic, version and the whole file's length (body + 16 + 8).
    std::ostringstream hs(std::ios::binary);
    BinaryWriter h(hs);
    h.bytes(kArenaResultMagic, 4);
    h.u32(kArenaResultVersion);
    h.u64(static_cast<uint64_t>(body.size() + 16 + 8));
    std::string bytes = hs.str() + body;
    putChecksum(bytes);
    return bytes;
}

ArenaSliceResult readArenaResult(const std::string& path, const std::string& jobBytes,
                                 const std::string& expectedBuild) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw SliceError(path + ": cannot read the arena result");
    const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    // The file's name only: a refusal can end up in the manifest.
    const std::string where = "arena result " + std::filesystem::path(path).filename().string() + ": ";
    const std::string size = std::to_string(bytes.size());
    if (bytes.size() < 24) throw SliceError(where + "truncated (" + size + " bytes)");
    if (bytes.compare(0, 4, kArenaResultMagic, 4) != 0)
        throw SliceError(where + "wrong magic (not an arena result)");
    const uint32_t version = static_cast<uint32_t>(storedUint(bytes, 4, 4));
    if (version != kArenaResultVersion)
        throw SliceError(where + "version " + std::to_string(version) + " (expected " +
                         std::to_string(kArenaResultVersion) + ")");
    const uint64_t length = storedUint(bytes, 8, 8);
    if (bytes.size() < length)
        throw SliceError(where + "truncated (" + size + " of " + std::to_string(length) + " bytes)");
    if (bytes.size() > length)
        throw SliceError(where + "length: " + size + " bytes on disk, " + std::to_string(length) + " declared");
    if (storedChecksum(bytes) != fnv1a64(bytes.substr(0, bytes.size() - 8)))
        throw SliceError(where + "checksum mismatch (damaged in transit or on disk)");

    ArenaSliceResult result;
    std::istringstream is(bytes.substr(16, bytes.size() - 24), std::ios::binary);
    BinaryReader r(is);
    try {
        result.jobHash = r.u64();
        if (result.jobHash != fnv1a64(jobBytes))
            throw SliceError("answers another job (job hash differs): a stale or foreign result");
        result.buildId = r.str();
        if (result.buildId != expectedBuild)
            throw SliceError("built from another commit (build ID " + result.buildId + ", expected " +
                             expectedBuild + ")");
        result.fingerprint = r.str();
        if (!slice_files::printableFingerprint(result.fingerprint))
            throw SliceError("bad engine fingerprint (empty, too long or not printable)");
        const ArenaJob job = parseArenaJob(jobBytes);
        const uint32_t count = r.u32();
        if (count != static_cast<uint32_t>(job.games()))
            throw SliceError("holds " + std::to_string(count) + " games, the job's slice has " +
                             std::to_string(job.games()));
        long long positions = 0;
        for (size_t k = 0; k < job.ranges.size(); ++k) {
            const ArenaJobRange& range = job.ranges[k];
            for (int j = range.first; j < range.first + range.count; ++j) {
                const std::string game = "range " + std::to_string(k) + " game " + std::to_string(j) + ": ";
                const uint32_t n = r.u32();
                if (n > static_cast<uint32_t>(job.cfg.maxMoves))
                    throw SliceError(game + std::to_string(n) + " moves, more than max_moves");
                ArenaGameResult g;
                g.moves.resize(n);
                for (int& m : g.moves) m = r.i32();
                g.scoreA = r.f64();
                verifyArenaGame(job, j, g, game);
                positions += n;
                result.games.push_back(std::move(g));
            }
        }
        result.stats.seconds = r.f64();
        result.stats.positions = static_cast<long long>(r.u64());
        result.stats.evaluations = static_cast<long long>(r.u64());
        if (result.stats.positions != positions)
            throw SliceError("positions: the statistics count " + std::to_string(result.stats.positions) +
                             ", the games hold " + std::to_string(positions));
        if (is.peek() != std::char_traits<char>::eof()) throw SliceError("trailing bytes");
    } catch (const std::runtime_error& e) {   // SliceError included
        throw SliceError(where + e.what());
    }
    return result;
}

int runArenaJob(const std::string& jobPath, const std::string& outPath, const std::string& backend,
                int workers, int inferenceBatch, std::ostream& err) {
    try {
        std::ifstream in(jobPath, std::ios::binary);
        if (!in) throw SliceError(jobPath + ": cannot read the job");
        const std::string jobBytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const ArenaJob job = parseArenaJob(jobBytes);
        if (job.buildId != buildId())
            throw SliceError("build ID mismatch: the job comes from " + job.buildId + ", this binary is " +
                             buildId() + " (sync the machines)");
        checkBackendAvailable(backend);
        // Every network the ranges name, each checked against its hash.
        std::vector<std::unique_ptr<Network>> nets;
        for (const ArenaJobNetwork& file : job.networks) {
            const std::string netPath = (std::filesystem::path(jobPath).parent_path() / file.file).string();
            if (!std::filesystem::exists(netPath)) throw SliceError(netPath + ": network missing");
            if (fileHash(netPath) != file.hash)
                throw SliceError(netPath + ": its hash differs from the job's network hash");
            nets.push_back(std::make_unique<Network>(loadNetwork(netPath)));
            if (nets.back()->config().boardSize != job.boardSize)
                throw SliceError(netPath + " is not a " + std::to_string(job.boardSize) + "x" +
                                 std::to_string(job.boardSize) + " network");
        }
        auto player = [&](const ArenaJobPlayer& p) {
            ArenaPlayer out;
            out.net = p.network >= 0 ? nets[static_cast<size_t>(p.network)].get() : nullptr;
            out.simulations = p.simulations;
            out.randomSymmetry = p.randomSymmetry;
            out.cPuct = p.cPuct;
            out.mctsBatch = p.mctsBatch;
            return out;
        };
        std::vector<ArenaGameRange> ranges;
        for (const ArenaJobRange& r : job.ranges) ranges.push_back({player(r.a), player(r.b), r.seed, r.first, r.count});
        EngineRun run;
        run.backend = backend;
        run.workers = sliceWorkers(backend, workers, std::max(1, job.games()), resolveWorkers(0));
        run.inferenceBatch = inferenceBatch > 0 ? inferenceBatch : job.inferenceBatch;
        ArenaSliceResult result;
        result.jobHash = fnv1a64(jobBytes);
        result.buildId = buildId();
        result.fingerprint = engineFingerprint(backend);
        // playArenaGames refuses what playMatch refuses (players, shapes, batches).
        try {
            result.games = playArenaGames(ranges, job.cfg, run, &result.stats);
        } catch (const std::invalid_argument& e) {
            throw SliceError(std::string("arena job: ") + e.what());
        }
        writeFileAtomically(outPath, serializeArenaResult(result));
        return 0;
    } catch (const std::exception& e) {
        err << "selfplay: " << e.what() << "\n";
        return 1;
    }
}

int runJob(const std::string& jobPath, const std::string& outPath, const std::string& backend, int workers,
           int inferenceBatch, std::ostream& err) {
    char magic[4] = {0, 0, 0, 0};
    {
        std::ifstream in(jobPath, std::ios::binary);
        in.read(magic, 4);
    }
    if (isArenaJob(std::string(magic, 4)))
        return runArenaJob(jobPath, outPath, backend, workers, inferenceBatch, err);
    return runSliceJob(jobPath, outPath, backend, workers, inferenceBatch, err);
}

} // namespace azgo
