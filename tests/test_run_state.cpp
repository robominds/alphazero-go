#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/evaluator.hpp"
#include "azgo/rng.hpp"
#include "azgo/run_state.hpp"
#include "azgo/selfplay.hpp"

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

// A state with real content: two short self-play games, momentum buffers,
// and arena results for two pairs.
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
    return s;
}

static bool sameGame(const GameRecord& x, const GameRecord& y) {
    return x.boardSize == y.boardSize && x.komi == y.komi && x.moves == y.moves &&
           x.stones == y.stones && x.visits == y.visits && x.resultForBlack == y.resultForBlack;
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
    assert(a.input == b.input && a.pi == b.pi && a.z == b.z);

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
    newer[4] = 2;   // version is the u32 after the magic, little-endian
    writeBytes(wrong, newer);
    msg = loadError(wrong.string());
    assert(msg.find("version") != std::string::npos);

    // Trailing bytes mean the file is not what we wrote.
    writeBytes(wrong, bytes + "x");
    msg = loadError(wrong.string());
    assert(msg.find(wrong.string()) != std::string::npos);
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
    test_paths();
    std::printf("test_run_state: all tests passed\n");
    return 0;
}
