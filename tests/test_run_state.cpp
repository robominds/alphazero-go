#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/evaluator.hpp"
#include "azgo/rng.hpp"
#include "azgo/run_state.hpp"
#include "azgo/selfplay.hpp"
#include "azgo/serialize.hpp"

using namespace azgo;
namespace fs = std::filesystem;

static fs::path freshDir(const std::string& name) {
    fs::path dir = fs::temp_directory_path() / ("azgo_test_run_state_" + name);
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

static std::string readBytes(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static void writeBytes(const fs::path& p, const std::string& bytes) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// Bytes of the resignation state at the end of a version-5 file: the f32
// threshold, the u32 game count, and per game the winner's and the loser's
// lows (u32 count, then u32 ply + f32 value each).
static size_t resignTailSize(const TrainingState& state) {
    size_t size = 4 + 4;
    for (const ResignTrace& t : state.resignWindow) size += 4 + 8 * t.winnerLows.size() + 4 + 8 * t.loserLows.size();
    return size;
}

// The bytes an older version wrote for `state`, given the current (version
// 5) bytes: without the resignation threshold and window at the end and
// the buffer's resigned flags (version 4); for a state without reference networks also without their
// list (version 3, an empty list is one u32), also the replay buffer without
// ownership (version 2), and for version 1 also without the reference set
// and the bot list.
static std::string olderStateBytes(const std::string& v5, const TrainingState& state, int version) {
    assert(version >= 4 || state.referenceNetworks.empty());
    std::ostringstream with, without;
    BinaryWriter ww(with), wo(without);
    state.buffer.save(ww, true);
    state.buffer.save(wo, version >= 3, version >= 5);
    const size_t header = 4 + 4 + 4 + 4 + 8;   // magic, version, generation, best, samples
    assert(v5.compare(header, with.str().size(), with.str()) == 0);
    std::string tail = v5.substr(header + with.str().size());
    tail.resize(tail.size() - resignTailSize(state));
    if (version <= 3) tail.resize(tail.size() - 4);
    if (version == 1)
        tail.resize(tail.size() - 8 - 4 * (state.references.size() + state.botSimulations.size()));
    std::string out = v5.substr(0, header) + without.str() + tail;
    out[4] = static_cast<char>(version);
    return out;
}

// A state with real content: two short self-play games, momentum buffers,
// arena results for three pairs (one with a bot) and a reference set with
// two bots and a promoted generation.
static TrainingState makeState() {
    UniformEvaluator eval(5, 2, 0.0f, 8);
    SelfPlayConfig sp;
    sp.mcts.simulations = 8;
    sp.mcts.batchSize = 4;
    sp.temperatureMoves = 4;
    sp.maxMoves = 12;
    Rng rng(99);
    TrainingState s;
    s.generation = 7;
    s.bestGeneration = 5;
    s.samplesTrained = (1LL << 40) + 3;   // needs all 64 bits
    s.buffer = ReplayBuffer(1000);
    s.buffer.add(playSelfPlayGame(eval, sp, rng));
    s.buffer.add(playSelfPlayGame(eval, sp, rng));
    s.velocity = {{0.5f, -1.25f, 3.0f}, {}, {1e-7f}};
    s.results.addMatch(0, 7, 20, 12.5);
    s.results.addMatch(6, 7, 20, 9.0);
    s.results.addMatch(-1, 0, 8, 3.0);
    s.references = {-2, -1, 0, 5};
    s.botSimulations = {100, 400};
    return s;
}

static bool sameGame(const GameRecord& x, const GameRecord& y) {
    return x.boardSize == y.boardSize && x.komi == y.komi && x.moves == y.moves &&
           x.stones == y.stones && x.visits == y.visits && x.resultForBlack == y.resultForBlack &&
           x.finalOwnership == y.finalOwnership && x.resigned == y.resigned;
}

static void assertSameState(const TrainingState& x, const TrainingState& y) {
    assert(x.generation == y.generation);
    assert(x.bestGeneration == y.bestGeneration);
    assert(x.samplesTrained == y.samplesTrained);
    assert(x.buffer.capacity() == y.buffer.capacity());
    assert(x.buffer.positions() == y.buffer.positions());
    assert(x.buffer.games() == y.buffer.games());
    assert(x.buffer.totalPositionsAdded() == y.buffer.totalPositionsAdded());
    for (size_t i = 0; i < x.buffer.games(); ++i) assert(sameGame(x.buffer.game(i), y.buffer.game(i)));
    assert(x.velocity == y.velocity);
    assert(x.results.toCsv() == y.results.toCsv());
    assert(x.results.pairs().size() == y.results.pairs().size());
    assert(x.references == y.references);
    assert(x.botSimulations == y.botSimulations);
    assert(x.referenceNetworks.size() == y.referenceNetworks.size());
    for (size_t i = 0; i < x.referenceNetworks.size(); ++i)
        assert(x.referenceNetworks[i].path == y.referenceNetworks[i].path &&
               x.referenceNetworks[i].hash == y.referenceNetworks[i].hash);
    assert(x.resignThreshold == y.resignThreshold);
    assert(x.resignWindow.size() == y.resignWindow.size());
    for (size_t i = 0; i < x.resignWindow.size(); ++i)
        assert(x.resignWindow[i].winnerLows == y.resignWindow[i].winnerLows &&
               x.resignWindow[i].loserLows == y.resignWindow[i].loserLows);
}

static void test_round_trip() {
    fs::path dir = freshDir("roundtrip");
    std::string path = (dir / "state.bin").string();
    TrainingState original = makeState();
    assert(original.buffer.games() == 2 && original.buffer.positions() > 0);
    saveTrainingState(path, original);
    TrainingState loaded = loadTrainingState(path);
    assertSameState(original, loaded);

    // The restored buffer samples exactly like the original one.
    TrainingBatch a, b;
    Rng ra(5), rb(5);
    original.buffer.sample(16, 2, ra, a);
    loaded.buffer.sample(16, 2, rb, b);
    assert(a.size == 16 && b.size == 16);
    assert(a.input == b.input && a.pi == b.pi && a.z == b.z && a.ownership == b.ownership);

    // Saving the loaded state reproduces the same bytes.
    std::string path2 = (dir / "again.bin").string();
    saveTrainingState(path2, loaded);
    assert(readBytes(path) == readBytes(path2));
    fs::remove_all(dir);
}

static std::string loadError(const std::string& path) {
    try {
        loadTrainingState(path);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    assert(false && "expected std::runtime_error");
    return "";
}

// Review Focus 3: leftovers from a crash.
static void test_crash_leftovers() {
    fs::path dir = freshDir("crash");
    fs::path state = dir / "state.bin";
    fs::path tmp = dir / "state.bin.tmp";
    TrainingState original = makeState();
    saveTrainingState(state.string(), original);

    // A half-written temporary file from an interrupted save is ignored...
    Rng rng(1);
    std::string garbage;
    for (int i = 0; i < 1000; ++i) garbage += static_cast<char>(rng.below(256));
    writeBytes(tmp, garbage);
    assertSameState(original, loadTrainingState(state.string()));

    // ...and the next save replaces it and leaves no .tmp behind.
    original.generation = 8;
    saveTrainingState(state.string(), original);
    assert(!fs::exists(tmp));
    assert(loadTrainingState(state.string()).generation == 8);

    // A truncated state file fails with a message naming the file.
    std::string bytes = readBytes(state);
    fs::path truncated = dir / "truncated.bin";
    writeBytes(truncated, bytes.substr(0, bytes.size() / 2));
    std::string msg = loadError(truncated.string());
    assert(msg.find(truncated.string()) != std::string::npos);

    // Missing file.
    fs::path missing = dir / "missing.bin";
    msg = loadError(missing.string());
    assert(msg.find(missing.string()) != std::string::npos);

    // Wrong magic (e.g. a checkpoint passed by mistake).
    std::string badMagic = bytes;
    badMagic[0] = 'X';
    fs::path wrong = dir / "wrong.bin";
    writeBytes(wrong, badMagic);
    msg = loadError(wrong.string());
    assert(msg.find(wrong.string()) != std::string::npos && msg.find("magic") != std::string::npos);

    // A newer format version.
    std::string newer = bytes;
    newer[4] = 7;   // version is the u32 after the magic, little-endian
    writeBytes(wrong, newer);
    msg = loadError(wrong.string());
    assert(msg.find("version") != std::string::npos);

    // Trailing bytes mean the file is not what we wrote.
    writeBytes(wrong, bytes + "x");
    msg = loadError(wrong.string());
    assert(msg.find(wrong.string()) != std::string::npos);
    fs::remove_all(dir);
}

static void test_initial_references() {
    assert((initialReferences(0) == std::vector<int>{0}));
    assert((initialReferences(2) == std::vector<int>{-2, -1, 0}));
    assert((TrainingState{}.references == std::vector<int>{0}));
}

// A version-1 file has no ownership in the buffer and no reference set or
// bot list at the end (a u32 count and the i32 entries of each).
static void test_version1_gets_default_references() {
    fs::path dir = freshDir("version1");
    TrainingState original = makeState();
    saveTrainingState((dir / "current.bin").string(), original);
    std::string v1 = olderStateBytes(readBytes(dir / "current.bin"), original, 1);
    fs::path path = dir / "v1.bin";
    writeBytes(path, v1);

    TrainingState loaded = loadTrainingState(path.string(), {100, 400});
    assert((loaded.references == std::vector<int>{-2, -1, 0}));
    assert((loaded.botSimulations == std::vector<int>{100, 400}));   // adopts the config's list
    TrainingState expected = original;
    expected.references = {-2, -1, 0};
    assertSameState(expected, loaded);
    assert((loadTrainingState(path.string()).references == std::vector<int>{0}));   // no bots
    assert(loadTrainingState(path.string()).botSimulations.empty());

    // Saving it again writes version 5.
    saveTrainingState(path.string(), loaded);
    assert(readBytes(path)[4] == 5);
    assert((loadTrainingState(path.string()).references == std::vector<int>{-2, -1, 0}));
    fs::remove_all(dir);
}

// A version-2 file has everything but the buffer's ownership, which loading
// recomputes by replaying each game: the result equals what self-play
// recorded, and saving writes version 5 again, byte for byte.
static void test_version2_recomputes_ownership() {
    fs::path dir = freshDir("version2");
    TrainingState original = makeState();
    saveTrainingState((dir / "current.bin").string(), original);
    const std::string current = readBytes(dir / "current.bin");
    fs::path path = dir / "v2.bin";
    writeBytes(path, olderStateBytes(current, original, 2));
    TrainingState loaded = loadTrainingState(path.string());
    assertSameState(original, loaded);   // finalOwnership included
    for (size_t i = 0; i < loaded.buffer.games(); ++i)
        assert(loaded.buffer.game(i).finalOwnership.size() == 25);
    saveTrainingState(path.string(), loaded);
    assert(readBytes(path) == current);

    // A game that does not replay makes the file unreadable, naming it.
    TrainingState broken = original;
    broken.buffer = ReplayBuffer(1000);
    GameRecord bad = original.buffer.game(0);
    bad.stones[25] ^= 1;   // ply 1, point A5: a position the moves do not produce
    broken.buffer.add(bad);
    saveTrainingState(path.string(), broken);
    writeBytes(path, olderStateBytes(readBytes(path), broken, 2));
    const std::string msg = loadError(path.string());
    assert(msg.find(path.string()) != std::string::npos);
    assert(msg.find("game 0 does not replay: position 1 differs") != std::string::npos);
    fs::remove_all(dir);
}

// Version 4 records the run's reference networks after the bot list: each
// path and the hash of its bytes. Their players come after the bots, so
// with two bots and two reference networks the set may reach down to -4.
static void test_reference_networks_round_trip() {
    fs::path dir = freshDir("refnets");
    std::string path = (dir / "state.bin").string();
    TrainingState original = makeState();
    original.referenceNetworks = {{"runs/a/checkpoints/gen_0200.azn", 0x0123456789abcdefull},
                                  {"/abs/b.azn", ~0ull}};
    original.references = {-4, -3, -2, -1, 0, 5};
    saveTrainingState(path, original);
    assert(readBytes(path)[4] == 5);
    TrainingState loaded = loadTrainingState(path);
    assertSameState(original, loaded);
    saveTrainingState((dir / "again.bin").string(), loaded);
    assert(readBytes(path) == readBytes(dir / "again.bin"));

    // A version-3 file (no list) loads with no reference networks, and
    // saving it writes version 5.
    TrainingState plain = makeState();
    saveTrainingState(path, plain);
    const std::string current = readBytes(path);
    writeBytes(path, olderStateBytes(current, plain, 3));
    TrainingState v3 = loadTrainingState(path, {7});   // the bot list is stored, so {7} is unused
    assertSameState(plain, v3);
    assert(v3.referenceNetworks.empty());
    saveTrainingState(path, v3);
    assert(readBytes(path) == current);

    // Player -5 is neither a bot nor a reference network.
    original.references = {-5, -4, -3, -2, -1, 0, 5};
    saveTrainingState(path, original);
    const std::string msg = loadError(path);
    assert(msg.find(path) != std::string::npos && msg.find("reference") != std::string::npos);
    fs::remove_all(dir);
}

// Version 5 adds resignation: each buffered game's resigned flag, and at
// the end the threshold calibrated after the last generation and the window
// of recent resignation-disabled games (each its winner's and loser's
// lows). A version-4 file loads with no flags, an empty window and the
// default threshold (resume recalibrates from the window and the config).
static void test_resignation_round_trip() {
    fs::path dir = freshDir("resign");
    std::string path = (dir / "state.bin").string();
    TrainingState original = makeState();
    assert(original.resignThreshold == -0.9f && original.resignWindow.empty());   // the defaults
    original.resignThreshold = -0.83f;
    ResignTrace a, b;
    a.winnerLows = {{1, -0.5f}, {7, -0.95f}};
    a.loserLows = {{2, 0.25f}};
    b.loserLows = {{4, -1.0f}};   // the winner never dipped
    original.resignWindow = {a, b, ResignTrace{}};   // the last one a draw
    original.referenceNetworks = {{"/abs/ref.azn", 42}};
    original.references = {-3, -2, -1, 0, 5};
    GameRecord resigned = original.buffer.game(1);
    resigned.resigned = true;
    original.buffer.add(resigned);
    saveTrainingState(path, original);
    const std::string v5 = readBytes(path);
    assert(v5[4] == 5);
    TrainingState loaded = loadTrainingState(path);
    assertSameState(original, loaded);
    assert(loaded.buffer.game(2).resigned && !loaded.buffer.game(0).resigned);
    saveTrainingState((dir / "again.bin").string(), loaded);
    assert(readBytes(dir / "again.bin") == v5);

    writeBytes(path, olderStateBytes(v5, original, 4));
    TrainingState v4 = loadTrainingState(path);
    assert(v4.resignWindow.empty() && v4.resignThreshold == -0.9f);
    TrainingState expected = original;
    expected.resignThreshold = -0.9f;
    expected.resignWindow.clear();
    expected.buffer = ReplayBuffer(original.buffer.capacity());
    for (size_t i = 0; i < original.buffer.games(); ++i) {
        GameRecord g = original.buffer.game(i);
        g.resigned = false;
        expected.buffer.add(g);
    }
    assertSameState(expected, v4);

    // Damaged resignation state is refused, naming the file: a game count
    // larger than the file, a threshold out of [-1, 0] or not finite, and
    // lows that are out of [-1, 1], not falling, or not at increasing plies.
    const size_t tail = v5.size() - resignTailSize(original);
    auto refused = [&](TrainingState s) {
        saveTrainingState(path, s);
        const std::string msg = loadError(path);
        return msg.find(path) != std::string::npos && msg.find("resign") != std::string::npos;
    };
    std::string bad = v5;
    bad[tail + 4] = static_cast<char>(0xff);   // low byte of the u32 game count
    bad[tail + 7] = static_cast<char>(0x7f);
    writeBytes(path, bad);
    assert(loadError(path).find(path) != std::string::npos);
    for (float threshold : {0.5f, -1.5f, std::nanf("")}) {
        TrainingState s = original;
        s.resignThreshold = threshold;
        assert(refused(s));
    }
    for (const ResignLows& lows : {ResignLows{{1, -1.5f}}, ResignLows{{1, std::nanf("")}},
                                   ResignLows{{1, -0.5f}, {3, -0.4f}}, ResignLows{{3, -0.5f}, {3, -0.6f}},
                                   ResignLows{{0, -0.5f}}}) {
        TrainingState s = original;
        s.resignWindow[0].loserLows = lows;
        assert(refused(s));
    }
    fs::remove_all(dir);
}

// Pipelined generations: a commit holds the next generation's games,
// unprocessed. A state with them is version 6 (version 5 and the pending
// block); a state without them is still version 5, byte for byte, so a
// sequential run writes what it always wrote.
static bool sameRecord(const GameRecord& x, const GameRecord& y) {
    return sameGame(x, y) && x.resignDisabled == y.resignDisabled && x.resignLows[0] == y.resignLows[0] &&
           x.resignLows[1] == y.resignLows[1];
}

static void test_pending_selfplay_round_trip() {
    fs::path dir = freshDir("pending");
    const std::string path = (dir / "state.bin").string();
    TrainingState original = makeState();
    saveTrainingState(path, original);
    const std::string v5 = readBytes(path);
    assert(v5[4] == 5 && !loadTrainingState(path).pending);

    PendingSelfPlay p;
    p.network = 5;
    p.resignation = true;
    p.resignThreshold = -0.83f;
    p.maxMoves = 12;
    p.records = {original.buffer.game(0), original.buffer.game(1), original.buffer.game(0)};
    p.records[0].resignDisabled = true;
    p.records[0].resignLows[0] = {{1, -0.25f}, {3, -0.5f}};
    p.records[0].resignLows[1] = {{2, 0.5f}};
    p.records[2].resigned = true;
    p.records[2].resignLows[1] = {{2, -0.9f}};
    original.pending = p;
    saveTrainingState(path, original);
    const std::string v6 = readBytes(path);
    assert(v6[4] == 6);
    // Version 5's bytes, then the block.
    assert(v6.compare(8, v5.size() - 8, v5, 8, v5.size() - 8) == 0);
    assert(trainingStateGeneration(path) == 7);
    const TrainingState loaded = loadTrainingState(path);
    assertSameState(original, loaded);
    assert(loaded.pending && loaded.pending->network == 5 && loaded.pending->resignation);
    assert(loaded.pending->resignThreshold == -0.83f && loaded.pending->maxMoves == 12);
    assert(loaded.pending->records.size() == 3);
    for (size_t i = 0; i < 3; ++i) assert(sameRecord(loaded.pending->records[i], p.records[i]));
    saveTrainingState((dir / "again.bin").string(), loaded);
    assert(readBytes(dir / "again.bin") == v6);

    // Without the block again: version 5, the original bytes.
    TrainingState cleared = loaded;
    cleared.pending.reset();
    saveTrainingState(path, cleared);
    assert(readBytes(path) == v5);

    // A damaged block is refused, naming the file: no games, a network
    // after the committed generation, v_resign outside [-1, 0], max_moves
    // below 1, a cut or extended file.
    auto refused = [&](const TrainingState& s) {
        saveTrainingState(path, s);
        const std::string msg = loadError(path);
        return msg.find(path) != std::string::npos && msg.find("pending") != std::string::npos;
    };
    for (int v = 0; v < 5; ++v) {
        TrainingState s = original;
        if (v == 0) s.pending->records.clear();
        if (v == 1) s.pending->network = 8;
        if (v == 2) s.pending->network = -1;
        if (v == 3) s.pending->resignThreshold = 0.5f;
        if (v == 4) s.pending->maxMoves = 0;
        assert(refused(s));
    }
    writeBytes(path, v6.substr(0, v6.size() - 3));
    assert(loadError(path).find(path) != std::string::npos);
    writeBytes(path, v6 + "x");
    assert(loadError(path).find("trailing") != std::string::npos);
    // Version 5 with a block appended is not a version-6 file.
    std::string mislabeled = v6;
    mislabeled[4] = 5;
    writeBytes(path, mislabeled);
    assert(loadError(path).find("trailing") != std::string::npos);
    fs::remove_all(dir);
}

// FNV-1a over a file's bytes: equal bytes hash equally, one changed byte
// changes the hash, and a missing file is an error naming it.
static void test_file_hash() {
    fs::path dir = freshDir("hash");
    writeBytes(dir / "empty", "");
    assert(fileHash((dir / "empty").string()) == 0xcbf29ce484222325ull);   // FNV-1a offset basis
    writeBytes(dir / "a", "a");
    assert(fileHash((dir / "a").string()) == 0xaf63dc4c8601ec8cull);       // FNV-1a("a")
    std::string bytes(4096, '\0');
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<char>(i * 7);
    writeBytes(dir / "x", bytes);
    writeBytes(dir / "y", bytes);
    assert(fileHash((dir / "x").string()) == fileHash((dir / "y").string()));
    bytes[2000] ^= 1;
    writeBytes(dir / "y", bytes);
    assert(fileHash((dir / "x").string()) != fileHash((dir / "y").string()));
    bool threw = false;
    try {
        fileHash((dir / "missing").string());
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find((dir / "missing").string()) != std::string::npos;
    }
    assert(threw);
    fs::remove_all(dir);
}

static void test_rejects_bad_references() {
    fs::path dir = freshDir("badrefs");
    fs::path path = dir / "state.bin";
    // The last one names bot -3, but the file stores two bots.
    for (const std::vector<int>& bad : {std::vector<int>{0, -1}, std::vector<int>{-1, 5},
                                        std::vector<int>{0, 0}, std::vector<int>{},
                                        std::vector<int>{-3, 0}}) {
        TrainingState s = makeState();
        s.references = bad;
        saveTrainingState(path.string(), s);
        std::string msg = loadError(path.string());
        assert(msg.find(path.string()) != std::string::npos);
        assert(msg.find("reference") != std::string::npos);
    }
    fs::remove_all(dir);
}

static void test_paths() {
    assert(checkpointPath("run", 7) == "run/checkpoints/gen_0007.azn");
    assert(checkpointPath("run", 0) == "run/checkpoints/gen_0000.azn");
    assert(checkpointPath("run", 12345) == "run/checkpoints/gen_12345.azn");
    assert(checkpointPath("runs/a/", 42) == "runs/a/checkpoints/gen_0042.azn");
    assert(statePath("run") == "run/state.bin");
    assert(configPath("run") == "run/config.cfg");
}

int main() {
    test_round_trip();
    test_crash_leftovers();
    test_initial_references();
    test_version1_gets_default_references();
    test_version2_recomputes_ownership();
    test_reference_networks_round_trip();
    test_resignation_round_trip();
    test_pending_selfplay_round_trip();
    test_file_hash();
    test_rejects_bad_references();
    test_paths();
    std::printf("test_run_state: all tests passed\n");
    return 0;
}
