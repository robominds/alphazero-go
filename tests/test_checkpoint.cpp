#include <unistd.h>   // getpid
#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/network.hpp"
#include "azgo/rng.hpp"

using namespace azgo;
namespace fs = std::filesystem;

namespace {

fs::path gDir;

std::string pathFor(const std::string& name) { return (gDir / name).string(); }

std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void writeFile(const std::string& path, const std::string& data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

NetConfig smallConfig() {
    NetConfig cfg;
    cfg.boardSize = 5;
    cfg.history = 2;
    cfg.blocks = 1;
    cfg.filters = 4;
    cfg.valueHidden = 8;
    return cfg;
}

// A network whose weights AND BN running statistics differ from a fresh one.
Network trainedNetwork(uint64_t seed) {
    const NetConfig cfg = smallConfig();
    Network net(cfg, seed);
    NaiveBackend be;
    Workspace ws(cfg, 4, be);
    Rng rng(seed);
    std::vector<float> input(4 * cfg.planes() * cfg.area()), pi(4 * cfg.actions(), 0.0f), z(4);
    for (float& v : input) v = rng.uniform() < 0.5 ? 0.0f : 1.0f;
    for (int b = 0; b < 4; ++b) pi[b * cfg.actions() + rng.below(cfg.actions())] = 1.0f;
    for (float& v : z) v = rng.uniform() < 0.5 ? -1.0f : 1.0f;
    net.trainStep(ws, input.data(), pi.data(), z.data(), 4);
    for (ParamView& p : net.params())
        for (size_t i = 0; i < p.size; ++i) p.value[i] -= 0.01f * p.grad[i];
    return net;
}

bool sameTensors(Network& a, Network& b) {
    std::vector<ParamView> pa = a.params(), pb = b.params();
    std::vector<BufferView> ba = a.buffers(), bb = b.buffers();
    if (pa.size() != pb.size() || ba.size() != bb.size()) return false;
    for (size_t i = 0; i < pa.size(); ++i)
        if (pa[i].name != pb[i].name || pa[i].size != pb[i].size ||
            std::memcmp(pa[i].value, pb[i].value, pa[i].size * sizeof(float)) != 0)
            return false;
    for (size_t i = 0; i < ba.size(); ++i)
        if (ba[i].name != bb[i].name || ba[i].size != bb[i].size ||
            std::memcmp(ba[i].value, bb[i].value, ba[i].size * sizeof(float)) != 0)
            return false;
    return true;
}

// Loads `path` expecting a runtime_error whose message names the file and
// contains `fragment`.
void expectLoadError(const std::string& path, const std::string& fragment) {
    try {
        loadNetwork(path);
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        if (what.find(path) == std::string::npos || what.find(fragment) == std::string::npos) {
            std::printf("unexpected message: %s\n", what.c_str());
            assert(false);
        }
        return;
    }
    assert(false);   // loading must fail
}

} // namespace

static void test_round_trip() {
    Network net = trainedNetwork(3);
    const std::string path = pathFor("round_trip.azn");
    saveNetwork(net, path);
    assert(!fs::exists(path + ".tmp"));

    Network loaded = loadNetwork(path);
    assert(loaded.config() == net.config());
    assert(sameTensors(net, loaded));
    assert(readCheckpointConfig(path) == smallConfig());

    // BN running statistics were restored too, so inference is identical.
    NaiveBackend be;
    Workspace ws(smallConfig(), 2, be);
    Rng rng(9);
    std::vector<float> input(2 * smallConfig().planes() * smallConfig().area());
    for (float& v : input) v = rng.uniform() < 0.5 ? 0.0f : 1.0f;
    std::vector<float> la(2 * smallConfig().actions()), lb(la.size()), va(2), vb(2);
    net.infer(ws, input.data(), 2, la.data(), va.data());
    loaded.infer(ws, input.data(), 2, lb.data(), vb.data());
    assert(std::memcmp(la.data(), lb.data(), la.size() * sizeof(float)) == 0);
    assert(std::memcmp(va.data(), vb.data(), va.size() * sizeof(float)) == 0);
}

static void test_missing_file() {
    const std::string path = pathFor("does_not_exist.azn");
    bool threw = false;
    try {
        readCheckpointConfig(path);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()) == "cannot open checkpoint " + path;
    }
    assert(threw);
    expectLoadError(path, "cannot open checkpoint");
}

static void test_bad_magic_and_version() {
    const std::string path = pathFor("bad_magic.azn");
    writeFile(path, "NOT A CHECKPOINT AT ALL, JUST TEXT");
    expectLoadError(path, "not an AZGO checkpoint");

    Network net(smallConfig(), 1);
    const std::string good = pathFor("good.azn");
    saveNetwork(net, good);
    std::string data = readFile(good);
    data[4] = 7;                                       // version field, little-endian u32 at offset 4
    const std::string versioned = pathFor("version7.azn");
    writeFile(versioned, data);
    expectLoadError(versioned, "unsupported version 7");
    bool threw = false;
    try {
        readCheckpointConfig(versioned);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()) == versioned + ": unsupported version 7";
    }
    assert(threw);
}

static void test_header_does_not_match_tensors() {
    // A header claiming 5 filters in front of a 4-filter network's tensors:
    // the first tensor's size gives it away.
    Network net(smallConfig(), 1);
    const std::string good = pathFor("four_filters.azn");
    saveNetwork(net, good);
    std::string data = readFile(good);
    data[8 + 3 * 4] = 5;                               // filters: 4th i32 of the NetConfig
    const std::string path = pathFor("five_filters.azn");
    writeFile(path, data);
    NetConfig claimed = readCheckpointConfig(path);
    assert(claimed.filters == 5 && claimed != smallConfig());
    expectLoadError(path, "tensor 'stem.conv.w' has");

    // A checkpoint of a different architecture loads as that architecture;
    // callers compare readCheckpointConfig() with what they expected.
    NetConfig other = smallConfig();
    other.blocks = 2;
    Network deeper(other, 1);
    const std::string deeperPath = pathFor("deeper.azn");
    saveNetwork(deeper, deeperPath);
    assert(readCheckpointConfig(deeperPath) != smallConfig());
    assert(loadNetwork(deeperPath).config() == other);

    // Trailing garbage is rejected too.
    writeFile(path, readFile(good) + "x");
    expectLoadError(path, "unexpected data after the last tensor");
}

static void test_truncated_checkpoint_names_file() {
    // Review Focus 3: a crash can leave a cut-off file; loading must fail
    // with a message naming it, never crash or return a half-loaded network.
    Network net = trainedNetwork(4);
    const std::string good = pathFor("full.azn");
    saveNetwork(net, good);
    const std::string data = readFile(good);
    const std::string path = pathFor("truncated.azn");
    writeFile(path, data.substr(0, data.size() / 2));
    expectLoadError(path, "truncated");
    writeFile(path, data.substr(0, 10));                // inside the header
    bool threw = false;
    try {
        readCheckpointConfig(path);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()) == path + ": truncated";
    }
    assert(threw);
    writeFile(path, "");
    expectLoadError(path, "truncated");
}

static void test_leftover_tmp_is_ignored_and_overwritten() {
    // Review Focus 3: garbage "<path>.tmp" from an interrupted save must not
    // affect loading, and the next save must replace it.
    Network first = trainedNetwork(5);
    const std::string path = pathFor("leftover.azn");
    saveNetwork(first, path);
    writeFile(path + ".tmp", "half-written garbage from a crash");
    Network loaded = loadNetwork(path);
    assert(sameTensors(first, loaded));

    Network second = trainedNetwork(6);
    saveNetwork(second, path);
    assert(!fs::exists(path + ".tmp"));
    Network reloaded = loadNetwork(path);
    assert(sameTensors(second, reloaded));
    assert(!sameTensors(first, reloaded));
}

int main() {
    gDir = fs::temp_directory_path() / ("azgo_test_checkpoint_" + std::to_string(getpid()));
    fs::remove_all(gDir);
    assert(fs::create_directory(gDir));
    test_round_trip();
    test_missing_file();
    test_bad_magic_and_version();
    test_header_does_not_match_tensors();
    test_truncated_checkpoint_names_file();
    test_leftover_tmp_is_ignored_and_overwritten();
    fs::remove_all(gDir);
    std::printf("test_checkpoint: all tests passed\n");
    return 0;
}
