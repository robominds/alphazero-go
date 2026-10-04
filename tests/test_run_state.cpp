#include <cassert>
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

// The bytes an older version wrote for `state`, which has no reference
// networks, given the current (version 4) bytes: without the reference
// network list at the end (version 3, an empty list is one u32), also the
// replay buffer without ownership (version 2), and for version 1 also
// without the reference set and the bot list.
static std::string olderStateBytes(const std::string& v4, const TrainingState& state, int version) {
    assert(state.referenceNetworks.empty());
    std::ostringstream with, without;
    BinaryWriter ww(with), wo(without);
    state.buffer.save(ww, true);
    state.buffer.save(wo, version >= 3);
    const size_t header = 4 + 4 + 4 + 4 + 8;   // magic, version, generation, best, samples
    assert(v4.compare(header, with.str().size(), with.str()) == 0);
    std::string tail = v4.substr(header + with.str().size());
    tail.resize(tail.size() - 4);
    if (version == 1)
        tail.resize(tail.size() - 8 - 4 * (state.references.size() + state.botSimulations.size()));
    std::string out = v4.substr(0, header) + without.str() + tail;
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
           x.finalOwnership == y.finalOwnership;
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
    newer[4] = 5;   // version is the u32 after the magic, little-endian
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
    saveTrainingState((dir / "v4.bin").string(), original);
    std::string v1 = olderStateBytes(readBytes(dir / "v4.bin"), original, 1);
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

    // Saving it again writes version 4.
    saveTrainingState(path.string(), loaded);
    assert(readBytes(path)[4] == 4);
    assert((loadTrainingState(path.string()).references == std::vector<int>{-2, -1, 0}));
    fs::remove_all(dir);
}

// A version-2 file has everything but the buffer's ownership, which loading
// recomputes by replaying each game: the result equals what self-play
// recorded, and saving writes version 4 again, byte for byte.
static void test_version2_recomputes_ownership() {
    fs::path dir = freshDir("version2");
    TrainingState original = makeState();
    saveTrainingState((dir / "v4.bin").string(), original);
    const std::string v4 = readBytes(dir / "v4.bin");
    fs::path path = dir / "v2.bin";
    writeBytes(path, olderStateBytes(v4, original, 2));
    TrainingState loaded = loadTrainingState(path.string());
    assertSameState(original, loaded);   // finalOwnership included
    for (size_t i = 0; i < loaded.buffer.games(); ++i)
        assert(loaded.buffer.game(i).finalOwnership.size() == 25);
    saveTrainingState(path.string(), loaded);
    assert(readBytes(path) == v4);

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
    assert(readBytes(path)[4] == 4);
    TrainingState loaded = loadTrainingState(path);
    assertSameState(original, loaded);
    saveTrainingState((dir / "again.bin").string(), loaded);
    assert(readBytes(path) == readBytes(dir / "again.bin"));

    // A version-3 file (no list) loads with no reference networks, and
    // saving it writes version 4.
    TrainingState plain = makeState();
    saveTrainingState(path, plain);
    const std::string v4 = readBytes(path);
    writeBytes(path, olderStateBytes(v4, plain, 3));
    TrainingState v3 = loadTrainingState(path, {7});   // the bot list is stored, so {7} is unused
    assertSameState(plain, v3);
    assert(v3.referenceNetworks.empty());
    saveTrainingState(path, v3);
    assert(readBytes(path) == v4);

    // Player -5 is neither a bot nor a reference network.
    original.references = {-5, -4, -3, -2, -1, 0, 5};
    saveTrainingState(path, original);
    const std::string msg = loadError(path);
    assert(msg.find(path) != std::string::npos && msg.find("reference") != std::string::npos);
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
    test_file_hash();
    test_rejects_bad_references();
    test_paths();
    std::printf("test_run_state: all tests passed\n");
    return 0;
}
