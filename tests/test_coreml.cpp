// Tests for the Core ML engines (azgo/coreml.hpp).
//
// In every build: the float16 conversion, the chunking of a pass, and the
// model writer, by reading the package it writes back (the protobuf, the
// weight blob and the folded weights in it). In a Core ML build, on each
// device the test runs on: where Core ML places the model (every operation
// on the Neural Engine for ane, on the GPU for coreml_gpu), the outputs
// against a double-precision reference within the float16 tier, batch
// independence, repeatability, the hub, failures, memory checks and whole
// runs. Returns 77 (CTest Skipped) after the portable part when Core ML is
// not built or neither engine can run here.
//
// AZGO_COREML_DATA, when set, names a directory of real checkpoints
// (*.azn) and games (*.sgf): the accuracy and batch-independence checks
// then run on those networks and positions too (the evidence of the design
// spec used runs/13x13-agz and runs/19x19-agz).
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/config.hpp"
#include "azgo/coreml.hpp"
#include "azgo/encode.hpp"
#include "azgo/engines.hpp"
#include "azgo/game.hpp"
#include "azgo/inference_hub.hpp"
#include "azgo/metal.hpp"
#include "azgo/network.hpp"
#include "azgo/rng.hpp"
#include "azgo/run_state.hpp"
#include "azgo/sgf.hpp"
#include "azgo/trainer.hpp"
#ifdef AZ_USE_COREML
#include <Accelerate/Accelerate.h>
#endif

using namespace azgo;
namespace fs = std::filesystem;

namespace {

bool contains(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

NetConfig netConfig(int n, int history, int blocks, int filters, int valueHidden = 16) {
    NetConfig c;
    c.boardSize = n;
    c.history = history;
    c.blocks = blocks;
    c.filters = filters;
    c.valueHidden = valueHidden;
    return c;
}

// A network whose batch norms and biases are not the identity, so the fold
// is exercised: gamma in [0.5, 1.5], beta and biases in [-0.2, 0.2], means
// in [-0.2, 0.2], variances in [0.5, 1.5].
Network randomNetwork(const NetConfig& cfg, uint64_t seed) {
    Network net(cfg, seed);
    Rng rng(seed + 17);
    for (ParamView& p : net.params()) {
        const std::string& n = p.name;
        if (n.size() > 6 && n.compare(n.size() - 6, 6, ".gamma") == 0)
            for (size_t i = 0; i < p.size; ++i) p.value[i] = static_cast<float>(0.5 + rng.uniform());
        else if (!p.decay)
            for (size_t i = 0; i < p.size; ++i) p.value[i] = static_cast<float>(0.4 * rng.uniform() - 0.2);
    }
    for (BufferView& b : net.buffers()) {
        const bool var = b.name.find("runVar") != std::string::npos;
        for (size_t i = 0; i < b.size; ++i)
            b.value[i] = static_cast<float>(var ? 0.5 + rng.uniform() : 0.4 * rng.uniform() - 0.2);
    }
    return net;
}

// Positions from games of uniformly random legal moves, encoded as search
// encodes them: [count][planes][area].
std::vector<float> randomGamePositions(const NetConfig& cfg, int count, uint64_t seed) {
    Rng rng(seed);
    const size_t per = static_cast<size_t>(cfg.planes()) * cfg.area();
    std::vector<float> out(static_cast<size_t>(count) * per);
    int made = 0;
    while (made < count) {
        Game game(cfg.boardSize, 7.5f, defaultMaxMoves(cfg.boardSize));
        const int length = rng.below(static_cast<uint32_t>(cfg.area()));
        for (int m = 0; m < length && !game.isTerminal() && made < count; ++m) {
            if (m % 3 == 0) encodeGame(game, cfg.history, 0, out.data() + static_cast<size_t>(made++) * per);
            const std::vector<int> legal = game.legalActions();
            game.play(legal[rng.below(static_cast<uint32_t>(legal.size()))]);
        }
    }
    return out;
}

bool bitwiseEqual(const float* a, const float* b, size_t n) { return std::memcmp(a, b, n * sizeof(float)) == 0; }

// ---- float16 ------------------------------------------------------------------

void test_half_conversion() {
    // Every half survives the round trip (NaNs stay NaNs).
    for (uint32_t h = 0; h < 65536; ++h) {
        const float f = halfToFloat(static_cast<uint16_t>(h));
        if (std::isnan(f)) {
            assert(std::isnan(halfToFloat(floatToHalf(f))));
            continue;
        }
        assert(floatToHalf(f) == h);
    }
    // Between two neighbouring finite halves: below the midpoint to the
    // lower, above to the upper, on it to the even one.
    for (uint32_t h = 0; h < 0x7bff; ++h) {
        const double lo = halfToFloat(static_cast<uint16_t>(h)), hi = halfToFloat(static_cast<uint16_t>(h + 1));
        const float mid = static_cast<float>((lo + hi) / 2);
        assert(static_cast<double>(mid) == (lo + hi) / 2);   // exact in float
        assert(floatToHalf(mid) == ((h & 1) ? h + 1 : h));
        assert(floatToHalf(std::nextafter(mid, -1.0f)) == h);
        assert(floatToHalf(std::nextafter(mid, 2.0f * mid)) == h + 1);
        assert(floatToHalf(-mid) == (((h & 1) ? h + 1 : h) | 0x8000u));
    }
    assert(floatToHalf(65504.0f) == 0x7bff && floatToHalf(65520.0f) == 0x7c00 && floatToHalf(1e30f) == 0x7c00);
    assert(floatToHalf(-1e30f) == 0xfc00 && floatToHalf(1e-30f) == 0 && floatToHalf(-0.0f) == 0x8000);
    assert(floatToHalf(std::ldexp(1.0f, -24)) == 1 && floatToHalf(std::ldexp(1.0f, -25)) == 0);   // tie: to even (0)
    assert(floatToHalf(std::ldexp(1.5f, -25)) == 1);
#if defined(__FLT16_MANT_DIG__)
    // Against the compiler's own float16, on random floats of every magnitude.
    Rng rng(9);
    for (int i = 0; i < 1000000; ++i) {
        uint32_t bits = static_cast<uint32_t>(rng.next());
        float f;
        std::memcpy(&f, &bits, 4);
        if (std::isnan(f)) continue;
        const _Float16 ref = static_cast<_Float16>(f);
        uint16_t rb;
        std::memcpy(&rb, &ref, 2);
        assert(floatToHalf(f) == rb);
    }
#endif
    std::printf("float16 conversion: ok\n");
}

void test_chunks() {
    const std::vector<int> ane = {8, 32, 128};
    auto sum = [](const std::vector<int>& v) {
        int s = 0;
        for (int x : v) s += x;
        return s;
    };
    assert((coremlChunks(ane, 1) == std::vector<int>{8}));
    assert((coremlChunks(ane, 8) == std::vector<int>{8}));
    assert((coremlChunks(ane, 9) == std::vector<int>{8, 8}));
    assert((coremlChunks(ane, 40) == std::vector<int>{32, 8}));
    assert((coremlChunks(ane, 256) == std::vector<int>{128, 128}));
    assert((coremlChunks(ane, 100) == std::vector<int>{32, 32, 32, 8}));
    assert((coremlChunks({64}, 1) == std::vector<int>{64}));
    assert((coremlChunks({64}, 200) == std::vector<int>{64, 64, 64, 64}));
    for (int B = 1; B <= 600; ++B)
        for (const std::vector<int>& sizes : {ane, std::vector<int>{64}, std::vector<int>{1, 7}}) {
            const int s = sum(coremlChunks(sizes, B));
            assert(s >= B && s < B + sizes[0]);
        }
    bool threw = false;
    try {
        coremlChunks({8, 8}, 3);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    assert(coremlBatchSizes("ane") == ane && coremlBatchSizes("coreml_gpu") == std::vector<int>{64});
    assert(coremlBackend("ane") && coremlBackend("coreml_gpu") && !coremlBackend("metal") && !coremlBackend("gpu"));
    std::printf("chunks: ok\n");
}

// ---- reading the package back -------------------------------------------------
//
// A minimal protobuf reader: a message is a list of (field, wire type,
// value) where value is a varint or a byte string.

struct Field {
    int number = 0, wire = 0;
    uint64_t varint = 0;
    std::string bytes;
};

std::vector<Field> parseMessage(const std::string& s) {
    std::vector<Field> out;
    size_t i = 0;
    auto varint = [&]() {
        uint64_t v = 0;
        int shift = 0;
        for (;;) {
            assert(i < s.size());
            const uint8_t b = static_cast<uint8_t>(s[i++]);
            v |= static_cast<uint64_t>(b & 0x7f) << shift;
            if (!(b & 0x80)) break;
            shift += 7;
        }
        return v;
    };
    while (i < s.size()) {
        Field f;
        const uint64_t tag = varint();
        f.number = static_cast<int>(tag >> 3);
        f.wire = static_cast<int>(tag & 7);
        if (f.wire == 0) {
            f.varint = varint();
        } else {
            assert(f.wire == 2);   // the writer uses no other wire type
            const size_t len = static_cast<size_t>(varint());
            assert(i + len <= s.size());
            f.bytes = s.substr(i, len);
            i += len;
        }
        out.push_back(f);
    }
    return out;
}

std::vector<Field> all(const std::string& msg, int number) {
    std::vector<Field> out;
    for (const Field& f : parseMessage(msg))
        if (f.number == number) out.push_back(f);
    return out;
}

Field one(const std::string& msg, int number) {
    const std::vector<Field> f = all(msg, number);
    assert(f.size() == 1);
    return f[0];
}

std::vector<int64_t> packed(const std::string& bytes) {
    std::vector<int64_t> out;
    size_t i = 0;
    while (i < bytes.size()) {
        uint64_t v = 0;
        int shift = 0;
        for (;;) {
            const uint8_t b = static_cast<uint8_t>(bytes[i++]);
            v |= static_cast<uint64_t>(b & 0x7f) << shift;
            if (!(b & 0x80)) break;
            shift += 7;
        }
        out.push_back(static_cast<int64_t>(v));
    }
    return out;
}

// map<string, M> entries {1: key, 2: value} as a map.
std::map<std::string, std::string> mapOf(const std::string& msg, int number) {
    std::map<std::string, std::string> out;
    for (const Field& e : all(msg, number)) out[one(e.bytes, 1).bytes] = one(e.bytes, 2).bytes;
    return out;
}

// A TensorType's (data type, shape).
std::pair<int, std::vector<int64_t>> tensorTypeOf(const std::string& valueType) {
    const std::string tt = one(valueType, 1).bytes;
    std::vector<int64_t> shape;
    for (const Field& d : all(tt, 3)) shape.push_back(static_cast<int64_t>(one(one(d.bytes, 1).bytes, 1).varint));
    const int rank = static_cast<int>(one(tt, 2).varint);
    assert(rank == static_cast<int>(shape.size()));
    return {static_cast<int>(one(tt, 1).varint), shape};
}

struct ReadOp {
    std::string type, name;
    std::map<std::string, std::string> inputs;   // argument -> bound name
    int dtype = 0;
    std::vector<int64_t> shape;
    std::string value;                           // a const's Value message
};

std::vector<ReadOp> readOps(const std::string& spec) {
    const std::string program = one(spec, 502).bytes;
    assert(one(program, 1).varint == 1);
    const std::map<std::string, std::string> functions = mapOf(program, 2);
    assert(functions.size() == 1 && functions.count("main"));
    const std::string& main = functions.at("main");
    assert(one(main, 2).bytes == "CoreML8");
    const std::map<std::string, std::string> blocks = mapOf(main, 3);
    assert(blocks.size() == 1 && blocks.count("CoreML8"));
    const std::string& block = blocks.at("CoreML8");
    const std::vector<Field> outs = all(block, 2);
    assert(outs.size() == 2 && outs[0].bytes == "logits" && outs[1].bytes == "value");
    std::vector<ReadOp> ops;
    for (const Field& f : all(block, 3)) {
        ReadOp op;
        op.type = one(f.bytes, 1).bytes;
        for (const auto& [key, arg] : mapOf(f.bytes, 2)) op.inputs[key] = one(one(arg, 1).bytes, 1).bytes;
        const Field out = one(f.bytes, 3);
        op.name = one(out.bytes, 1).bytes;
        std::tie(op.dtype, op.shape) = tensorTypeOf(one(out.bytes, 2).bytes);
        const std::map<std::string, std::string> attrs = mapOf(f.bytes, 5);
        if (op.type == "const") {
            assert(attrs.size() == 2 && attrs.count("val"));
            op.value = attrs.at("val");
        } else {
            assert(attrs.size() == 1);
        }
        ops.push_back(op);
    }
    return ops;
}

// The float16 values of a const: inline bytes or from the blob.
std::vector<uint16_t> halvesOf(const ReadOp& op, const std::string& blob) {
    size_t count = 1;
    for (int64_t d : op.shape) count *= static_cast<size_t>(d);
    std::vector<uint16_t> out(count);
    const std::vector<Field> fields = parseMessage(op.value);
    for (const Field& f : fields) {
        if (f.number == 3) {   // immediate: tensor -> bytes
            const std::string bytes = one(one(one(f.bytes, 1).bytes, 7).bytes, 1).bytes;
            assert(bytes.size() == count * 2 && count < 10);
            std::memcpy(out.data(), bytes.data(), bytes.size());
            return out;
        }
        if (f.number == 5) {   // blob file
            assert(one(f.bytes, 1).bytes == "@model_path/weights/weight.bin");
            const size_t at = static_cast<size_t>(one(f.bytes, 2).varint);
            assert(at % 64 == 0 && at + 64 <= blob.size() && count >= 10);
            uint32_t sentinel, dtype;
            uint64_t size, data;
            std::memcpy(&sentinel, blob.data() + at, 4);
            std::memcpy(&dtype, blob.data() + at + 4, 4);
            std::memcpy(&size, blob.data() + at + 8, 8);
            std::memcpy(&data, blob.data() + at + 16, 8);
            assert(sentinel == 0xDEADBEEFu && dtype == 1 && size == count * 2 && data == at + 64);
            assert(data + size <= blob.size());
            std::memcpy(out.data(), blob.data() + data, size);
            return out;
        }
    }
    assert(false && "const without a value");
    return out;
}

// The writer's package read back: the interface, the operation graph in
// the order of Network::forward, every name bound to an earlier value,
// and every weight equal to the fold computed here independently.
void test_writer_structure() {
    const NetConfig cfg = netConfig(5, 2, 2, 8, 6);
    Network net = randomNetwork(cfg, 31);
    const int B = 3;
    const CoreMLModelFiles f = buildCoreMLModel(net, B);
    const int64_t P = cfg.planes(), n = 5, A = 25, ACT = 26;

    // The model: version 9, the interface of float32 multi-arrays.
    assert(one(f.spec, 1).varint == 9);
    const std::string desc = one(f.spec, 2).bytes;
    auto feature = [](const std::string& fd) {
        const std::string array = one(one(fd, 3).bytes, 5).bytes;
        assert(one(array, 2).varint == 65568);   // FLOAT32
        return std::make_pair(one(fd, 1).bytes, packed(one(array, 1).bytes));
    };
    assert((feature(one(desc, 1).bytes) == std::make_pair(std::string("planes"), std::vector<int64_t>{B, P, n, n})));
    const std::vector<Field> outs = all(desc, 10);
    assert(outs.size() == 2);
    assert((feature(outs[0].bytes) == std::make_pair(std::string("logits"), std::vector<int64_t>{B, ACT})));
    assert((feature(outs[1].bytes) == std::make_pair(std::string("value"), std::vector<int64_t>{B})));

    // The graph: names bound only to the input or earlier outputs.
    const std::vector<ReadOp> ops = readOps(f.spec);
    std::map<std::string, const ReadOp*> byName;
    std::vector<std::string> kinds;
    for (const ReadOp& op : ops) {
        for (const auto& [key, bound] : op.inputs) {
            (void)key;
            assert(bound == "planes" || byName.count(bound));
        }
        assert(!byName.count(op.name));
        byName[op.name] = &op;
        if (op.type != "const") kinds.push_back(op.type);
    }
    const std::vector<std::string> expected = {
        "cast", "conv", "relu",                                     // stem
        "conv", "relu", "conv", "add", "relu",                      // block 0
        "conv", "relu", "conv", "add", "relu",                      // block 1
        "conv", "relu", "reshape", "linear", "cast",                // policy
        "conv", "relu", "reshape", "linear", "relu", "linear", "tanh", "reshape", "cast"};   // value
    assert(kinds == expected);
    assert(byName.at("logits")->dtype == 11 && byName.at("logits")->shape == (std::vector<int64_t>{B, ACT}));
    assert(byName.at("value")->dtype == 11 && byName.at("value")->shape == (std::vector<int64_t>{B}));
    assert(byName.at("block1_add")->inputs.at("y") == "block0_relu2");   // the skip
    assert(byName.at("stem_conv")->shape == (std::vector<int64_t>{B, 8, n, n}));

    // Every folded weight, recomputed: s = gamma / sqrt(var + eps) in double.
    std::map<std::string, const float*> t;
    for (const ParamView& p : net.params()) t[p.name] = p.value;
    for (const BufferView& b : net.buffers()) t[b.name] = b.value;
    auto checkConv = [&](const std::string& op, const std::string& prefix, const std::string& bn, int cout, int per) {
        const ReadOp& conv = *byName.at(op);
        const std::vector<uint16_t> w = halvesOf(*byName.at(conv.inputs.at("weight")), f.weights);
        const std::vector<uint16_t> b = halvesOf(*byName.at(conv.inputs.at("bias")), f.weights);
        assert(w.size() == static_cast<size_t>(cout) * per && b.size() == static_cast<size_t>(cout));
        for (int c = 0; c < cout; ++c) {
            const double s = static_cast<double>(t[bn + ".gamma"][c]) /
                             std::sqrt(static_cast<double>(t[bn + ".runVar"][c]) + 1e-5f);
            for (int i = 0; i < per; ++i)
                assert(w[c * per + i] == floatToHalf(static_cast<float>(t[prefix + ".w"][c * per + i] * s)));
            assert(b[c] == floatToHalf(static_cast<float>(t[bn + ".beta"][c] - s * t[bn + ".runMean"][c])));
        }
    };
    checkConv("stem_conv", "stem.conv", "stem.bn", 8, static_cast<int>(P) * 9);
    checkConv("block1_conv2", "block1.conv2", "block1.bn2", 8, 8 * 9);
    checkConv("policy_conv", "policy.conv", "policy.bn", 2, 8);
    checkConv("value_conv", "value.conv", "value.bn", 1, 8);
    auto checkLinear = [&](const std::string& op, const std::string& prefix, size_t out, size_t in) {
        const ReadOp& lin = *byName.at(op);
        const std::vector<uint16_t> w = halvesOf(*byName.at(lin.inputs.at("weight")), f.weights);
        const std::vector<uint16_t> b = halvesOf(*byName.at(lin.inputs.at("bias")), f.weights);
        assert(w.size() == out * in && b.size() == out);
        for (size_t i = 0; i < out * in; ++i) assert(w[i] == floatToHalf(t[prefix + ".w"][i]));
        for (size_t i = 0; i < out; ++i) assert(b[i] == floatToHalf(t[prefix + ".b"][i]));
    };
    checkLinear("policy_fc", "policy.fc", ACT, 2 * A);

    checkLinear("value_fc1", "value.fc1", 6, A);
    checkLinear("value_fc2", "value.fc2", 1, 6);   // the bias: one element, inline

    // The blob: its header counts the records, each 64-aligned.
    uint32_t count, version;
    std::memcpy(&count, f.weights.data(), 4);
    std::memcpy(&version, f.weights.data() + 4, 4);
    assert(version == 2 && f.weights.size() % 64 == 0);
    int blobConsts = 0;
    for (const ReadOp& op : ops)
        if (op.type == "const" && op.dtype == 10 && all(op.value, 5).size() == 1) ++blobConsts;
    assert(static_cast<int>(count) == blobConsts);

    // The manifest names both items; the root is the specification.
    assert(contains(f.manifest, "\"path\": \"com.apple.CoreML/model.mlmodel\""));
    assert(contains(f.manifest, "\"path\": \"com.apple.CoreML/weights\""));
    assert(contains(f.manifest, "\"rootModelIdentifier\""));
    std::printf("writer structure: ok (%zu ops, %zu-byte spec, %zu-byte blob)\n", ops.size(), f.spec.size(),
                f.weights.size());
}

// The same network gives the same bytes; any changed tensor (a batch norm
// statistic too) changes them; the ownership head is left out.
void test_writer_deterministic() {
    NetConfig cfg = netConfig(5, 1, 1, 8);
    const Network a = randomNetwork(cfg, 5), b = randomNetwork(cfg, 5);
    const CoreMLModelFiles fa = buildCoreMLModel(a, 4), fb = buildCoreMLModel(b, 4);
    assert(fa.spec == fb.spec && fa.weights == fb.weights && fa.manifest == fb.manifest);
    assert(coremlNetworkHash(a) == coremlNetworkHash(b));
    Network c = randomNetwork(cfg, 5);
    for (BufferView& v : c.buffers())
        if (v.name == "block0.bn2.runVar") v.value[3] *= 1.5f;
    const CoreMLModelFiles fc = buildCoreMLModel(c, 4);
    assert(fc.weights != fa.weights && coremlNetworkHash(c) != coremlNetworkHash(a));
    assert(buildCoreMLModel(a, 5).spec != fa.spec);   // the batch is part of the model
    cfg.ownership = true;
    const Network o = randomNetwork(cfg, 5);
    const std::vector<ReadOp> ops = readOps(buildCoreMLModel(o, 4).spec);
    for (const ReadOp& op : ops) assert(!contains(op.name, "ownership"));
    const std::string format = coremlModelFormatHash();
    assert(format.size() == 8 && format.find_first_not_of("0123456789abcdef") == std::string::npos);
    assert(format == coremlModelFormatHash());

    // The package on disk.
    const fs::path dir = fs::temp_directory_path() / ("azgo_test_coreml_" + std::to_string(getpid()) + ".mlpackage");
    fs::remove_all(dir);
    writeCoreMLPackage(a, 4, dir.string());
    auto read = [](const fs::path& p) {
        std::ifstream in(p, std::ios::binary);
        std::ostringstream s;
        s << in.rdbuf();
        return s.str();
    };
    assert(read(dir / "Manifest.json") == fa.manifest);
    assert(read(dir / "Data/com.apple.CoreML/model.mlmodel") == fa.spec);
    assert(read(dir / "Data/com.apple.CoreML/weights/weight.bin") == fa.weights);
    bool threw = false;
    try {
        writeCoreMLPackage(a, 4, dir.string());   // exists already
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    fs::remove_all(dir);
    std::printf("writer determinism: ok (model format %s)\n", format.c_str());
}

void test_not_built() {
    std::string why;
    for (const char* backend : {"ane", "coreml_gpu"}) {
        assert(!coremlAvailable(backend, &why) && contains(why, "-DAZ_USE_COREML=ON"));
        bool threw = false;
        try {
            checkBackendAvailable(backend);
        } catch (const std::invalid_argument& e) {
            threw = contains(e.what(), "rebuild with -DAZ_USE_COREML=ON");
        }
        assert(threw);
        threw = false;
        try {
            const Network net(netConfig(5, 1, 1, 8), 1);
            WorkerEvaluators w(net, backend, 1, 8, 8);
        } catch (const std::invalid_argument& e) {
            threw = contains(e.what(), "-DAZ_USE_COREML=ON");
        }
        assert(threw);
    }
    std::printf("not built: refused with the rebuild hint\n");
}

#ifdef AZ_USE_COREML

// ---- the double-precision reference --------------------------------------------
//
// Network::forward in inference mode, every operation in double: each
// convolution as a dgemm over unfolded patches, batch norm with its
// running statistics, the heads as written.
struct Reference {
    std::vector<double> logits, values;
};

Reference forward64(Network& net, const float* input, int B) {
    const NetConfig cfg = net.config();
    const int n = cfg.boardSize, A = cfg.area(), F = cfg.filters, P = cfg.planes(), ACT = cfg.actions(),
              V = cfg.valueHidden;
    std::map<std::string, std::vector<double>> t;
    for (const ParamView& p : net.params()) t[p.name].assign(p.value, p.value + p.size);
    for (const BufferView& b : net.buffers()) t[b.name].assign(b.value, b.value + b.size);
    const size_t M = static_cast<size_t>(B) * A;
    // Activations channel-major [C][B][A].
    auto conv = [&](const std::vector<double>& x, int cin, int cout, int k, const std::string& w,
                    const std::string& bn) {
        std::vector<double> col(static_cast<size_t>(cin) * k * k * M, 0.0);
        const int r = k / 2;
        for (int c = 0; c < cin; ++c)
            for (int dy = 0; dy < k; ++dy)
                for (int dx = 0; dx < k; ++dx) {
                    double* row = col.data() + (static_cast<size_t>(c) * k * k + dy * k + dx) * M;
                    for (int b = 0; b < B; ++b)
                        for (int y = 0; y < n; ++y)
                            for (int xx = 0; xx < n; ++xx) {
                                const int sy = y + dy - r, sx = xx + dx - r;
                                if (sy < 0 || sy >= n || sx < 0 || sx >= n) continue;
                                row[static_cast<size_t>(b) * A + y * n + xx] =
                                    x[static_cast<size_t>(c) * M + static_cast<size_t>(b) * A + sy * n + sx];
                            }
                }
        std::vector<double> y(static_cast<size_t>(cout) * M);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, cout, static_cast<int>(M), cin * k * k, 1.0,
                    t[w].data(), cin * k * k, col.data(), static_cast<int>(M), 0.0, y.data(), static_cast<int>(M));
        for (int c = 0; c < cout; ++c) {
            const double s = t[bn + ".gamma"][c] / std::sqrt(t[bn + ".runVar"][c] + 1e-5f);
            for (size_t i = 0; i < M; ++i)
                y[c * M + i] = s * (y[c * M + i] - t[bn + ".runMean"][c]) + t[bn + ".beta"][c];
        }
        return y;
    };
    auto relu = [](std::vector<double>& v) {
        for (double& x : v) x = x > 0 ? x : 0;
    };
    std::vector<double> x(static_cast<size_t>(P) * M);
    for (int b = 0; b < B; ++b)
        for (int p = 0; p < P; ++p)
            for (int a = 0; a < A; ++a)
                x[static_cast<size_t>(p) * M + static_cast<size_t>(b) * A + a] =
                    input[(static_cast<size_t>(b) * P + p) * A + a];
    std::vector<double> h = conv(x, P, F, 3, "stem.conv.w", "stem.bn");
    relu(h);
    for (int i = 0; i < cfg.blocks; ++i) {
        const std::string blk = "block" + std::to_string(i);
        std::vector<double> a = conv(h, F, F, 3, blk + ".conv1.w", blk + ".bn1");
        relu(a);
        std::vector<double> c = conv(a, F, F, 3, blk + ".conv2.w", blk + ".bn2");
        for (size_t j = 0; j < c.size(); ++j) c[j] += h[j];
        relu(c);
        h = c;
    }
    Reference out;
    out.logits.assign(static_cast<size_t>(B) * ACT, 0.0);
    out.values.assign(B, 0.0);
    std::vector<double> p = conv(h, F, 2, 1, "policy.conv.w", "policy.bn");
    relu(p);
    std::vector<double> v = conv(h, F, 1, 1, "value.conv.w", "value.bn");
    relu(v);
    for (int b = 0; b < B; ++b) {
        for (int o = 0; o < ACT; ++o) {
            double s = t["policy.fc.b"][o];
            for (int c = 0; c < 2; ++c)
                for (int a = 0; a < A; ++a)
                    s += t["policy.fc.w"][static_cast<size_t>(o) * 2 * A + c * A + a] *
                         p[static_cast<size_t>(c) * M + static_cast<size_t>(b) * A + a];
            out.logits[static_cast<size_t>(b) * ACT + o] = s;
        }
        double z = t["value.fc2.b"][0];
        for (int j = 0; j < V; ++j) {
            double s = t["value.fc1.b"][j];
            for (int a = 0; a < A; ++a) s += t["value.fc1.w"][static_cast<size_t>(j) * A + a] * v[static_cast<size_t>(b) * A + a];
            z += t["value.fc2.w"][j] * (s > 0 ? s : 0);
        }
        out.values[b] = std::tanh(z);
    }
    return out;
}

// The networks the device checks run on: preset-shaped random networks
// (batch norm calibrated on positions, as generation 0 is) and, with
// AZGO_COREML_DATA, real checkpoints with positions from real games.
struct Case {
    std::string name;
    std::shared_ptr<Network> net;
    std::vector<float> positions;
    int count = 0;
};

std::vector<Case> cases(int perCase) {
    std::vector<Case> out;
    for (const NetConfig& cfg : {netConfig(9, 8, 6, 64, 64), netConfig(13, 8, 6, 96, 64), netConfig(19, 8, 10, 128, 256)}) {
        Case c;
        c.name = "random " + std::to_string(cfg.boardSize) + "x" + std::to_string(cfg.boardSize) + " " +
                 std::to_string(cfg.blocks) + "x" + std::to_string(cfg.filters);
        c.net = std::make_shared<Network>(randomNetwork(cfg, 1000 + cfg.boardSize));
        {
            const std::vector<float> cal = randomGamePositions(cfg, 64, 77);
            auto blas = makeBackend("blas");
            Workspace ws(cfg, 64, *blas);
            c.net->calibrateBatchNorm(ws, cal.data(), 64);
        }
        c.count = perCase;
        c.positions = randomGamePositions(cfg, perCase, 99 + cfg.boardSize);
        out.push_back(c);
    }
    const char* data = std::getenv("AZGO_COREML_DATA");
    if (!data || !*data) return out;
    std::vector<fs::path> nets, games;
    for (const auto& e : fs::directory_iterator(data)) {
        if (e.path().extension() == ".azn") nets.push_back(e.path());
        if (e.path().extension() == ".sgf") games.push_back(e.path());
    }
    std::sort(nets.begin(), nets.end());
    std::sort(games.begin(), games.end());
    for (const fs::path& p : nets) {
        Case c;
        c.name = p.filename().string();
        c.net = std::make_shared<Network>(loadNetwork(p.string()));
        const NetConfig cfg = c.net->config();
        const size_t per = static_cast<size_t>(cfg.planes()) * cfg.area();
        std::vector<float> all;
        for (const fs::path& g : games) {
            std::ifstream in(g);
            std::stringstream text;
            text << in.rdbuf();
            const SgfGame sgf = parseSgf(text.str());
            if (sgf.boardSize != cfg.boardSize) continue;
            Game game(sgf.boardSize, sgf.komi, 100000);
            std::vector<float> buf(per);
            for (size_t m = 0; m <= sgf.moves.size(); ++m) {
                encodeGame(game, cfg.history, 0, buf.data());
                all.insert(all.end(), buf.begin(), buf.end());
                if (m < sgf.moves.size()) game.play(sgf.moves[m]);
            }
        }
        const int available = static_cast<int>(all.size() / per);
        if (available == 0) continue;
        c.count = std::min(perCase, available);
        for (int i = 0; i < c.count; ++i) {   // evenly spread over the games' positions
            const size_t src = static_cast<size_t>(i) * available / c.count;
            c.positions.insert(c.positions.end(), all.begin() + src * per, all.begin() + (src + 1) * per);
        }
        out.push_back(c);
    }
    return out;
}

// Core ML's compiler cache: named for this program, and its entries older
// than the age removed (with OS build directories left empty), newer ones
// kept.
void test_compiler_cache() {
    const std::string dir = coreml_probe::compilerCacheDirectory();
    const std::string tail = "/Library/Caches/test_coreml/com.apple.e5rt.e5bundlecache";
    assert(dir.size() > tail.size() && dir.compare(dir.size() - tail.size(), tail.size(), tail) == 0);
    const fs::path root = fs::temp_directory_path() / ("azgo_test_coreml_cache_" + std::to_string(getpid()));
    fs::remove_all(root);
    for (const char* d : {"26A434/old", "26A434/new/inner", "25B100/old"}) fs::create_directories(root / d);
    std::ofstream(root / "26A434/new/inner/file") << "x";
    const auto twoDays = std::chrono::duration_cast<fs::file_time_type::duration>(std::chrono::hours(48));
    for (const char* d : {"26A434/old", "25B100/old"})
        fs::last_write_time(root / d, fs::file_time_type::clock::now() - twoDays);
    assert(coreml_probe::pruneCompilerCache(root.string(), 24 * 3600) == 2);
    assert(!fs::exists(root / "26A434/old") && !fs::exists(root / "25B100"));
    assert(fs::exists(root / "26A434/new/inner/file"));
    assert(coreml_probe::pruneCompilerCache(root.string(), 24 * 3600) == 0);
    assert(coreml_probe::pruneCompilerCache((root / "absent").string(), 0) == 0);
    fs::remove_all(root);
    std::printf("compiler cache pruning: ok\n");
}

bool usable(const std::string& backend) {
    std::string why;
    if (coremlAvailable(backend, &why)) return true;
    std::printf("%s: not usable here (%s): its device checks are skipped\n", backend.c_str(), why.c_str());
    return false;
}

struct Outputs {
    std::vector<float> logits, values;
};

Outputs run(CoreMLInference& ci, const NetConfig& cfg, const float* x, int B, int size = 0) {
    Outputs o;
    o.logits.resize(static_cast<size_t>(B) * cfg.actions());
    o.values.resize(B);
    if (size > 0) ci.inferAtSize(size, x, B, o.logits.data(), o.values.data());
    else ci.infer(x, B, o.logits.data(), o.values.data());
    return o;
}

// Every operation on the backend's device, for every compiled size, on
// the preset shapes; a network too small for Core ML to place it there is
// refused by the runner, naming the placement.
void test_placement(const std::string& backend, const std::vector<Case>& cs) {
    const std::string want = backend == "ane" ? "ANE" : "GPU";
    for (const Case& c : cs) {
        for (int size : coremlBatchSizes(backend)) {
            const coreml_probe::Placement p = coreml_probe::placement(*c.net, backend, size);
            assert(p.devices.size() == 1 && p.devices[0].first == want && p.devices[0].second == p.operations);
            assert(p.operations == 17 + 5 * c.net->config().blocks);   // input cast, stem 2, 5 per block, heads 14
        }
        std::printf("%s placement %s: every operation on the %s at batches", backend.c_str(), c.name.c_str(),
                    want.c_str());
        for (int s : coremlBatchSizes(backend)) std::printf(" %d", s);
        std::printf("\n");
    }
    // 5x5, history 1, 1 block of 8: Core ML's cost model keeps it on the CPU.
    const Network tiny(netConfig(5, 1, 1, 8), 3);
    // The runner checks every compiled size: refused if any is off the device.
    bool elsewhere = false;
    coreml_probe::Placement p;
    for (int size : coremlBatchSizes(backend)) {
        p = coreml_probe::placement(tiny, backend, size);
        for (const auto& [device, count] : p.devices) elsewhere = elsewhere || (device != want && count > 0);
    }
    bool threw = false;
    try {
        makeCoreMLRunner(tiny, backend, 8);
    } catch (const std::runtime_error& e) {
        threw = contains(e.what(), "operations off the " + want);
    }
    assert(threw == elsewhere);
    std::printf("%s placement 5x5 1x8: %s (%s)\n", backend.c_str(), elsewhere ? "refused" : "accepted",
                p.summary.c_str());
}

// The float16 tier against the double reference, measured per network:
//   logits: max |diff| <= kLogitTier * max |reference logit|
//   values: max |diff| <= kValueTier
//   policy: total variation of the softmax <= kPolicyTier per position,
//           and the same best move wherever the reference's two best
//           logits are kArgmaxGap or more apart.
// Measured (docs/superpowers/specs/2026-10-24-coreml-engines-design.md,
// "Tolerance"): ANE worst 2.2e-3 / 1.6e-2 / 3.1e-3 on real 19x19 and 13x13
// networks, Core ML GPU 6.5e-4 / 3.1e-3 / 2.1e-3.
constexpr double kLogitTier = 5e-3, kValueTier = 3e-2, kPolicyTier = 1e-2, kArgmaxGap = 0.05;

void test_accuracy(const std::string& backend, const std::vector<Case>& cs) {
    for (const Case& c : cs) {
        const NetConfig cfg = c.net->config();
        const int ACT = cfg.actions();
        CoreMLInference ci(*c.net, backend, 256);
        double maxRef = 0, maxL = 0, maxV = 0, maxTv = 0, meanL = 0;
        int agree = 0, decided = 0;
        const size_t per = static_cast<size_t>(cfg.planes()) * cfg.area();
        for (int s = 0; s < c.count; s += 64) {
            const int B = std::min(64, c.count - s);
            const float* x = c.positions.data() + static_cast<size_t>(s) * per;
            const Reference ref = forward64(*c.net, x, B);
            const Outputs got = run(ci, cfg, x, B);
            for (int b = 0; b < B; ++b) {
                const double* rl = ref.logits.data() + static_cast<size_t>(b) * ACT;
                const float* gl = got.logits.data() + static_cast<size_t>(b) * ACT;
                double rmax = -1e300, gmax = -1e300;
                int rbest = 0, gbest = 0;
                double second = -1e300;
                for (int a = 0; a < ACT; ++a) {
                    maxRef = std::max(maxRef, std::fabs(rl[a]));
                    maxL = std::max(maxL, std::fabs(rl[a] - gl[a]));
                    meanL += std::fabs(rl[a] - gl[a]);
                    if (rl[a] > rmax) {
                        second = rmax;
                        rmax = rl[a];
                        rbest = a;
                    } else {
                        second = std::max(second, rl[a]);
                    }
                    if (gl[a] > gmax) {
                        gmax = gl[a];
                        gbest = a;
                    }
                }
                double zr = 0, zg = 0;
                for (int a = 0; a < ACT; ++a) {
                    zr += std::exp(rl[a] - rmax);
                    zg += std::exp(gl[a] - gmax);
                }
                double tv = 0;
                for (int a = 0; a < ACT; ++a) tv += std::fabs(std::exp(rl[a] - rmax) / zr - std::exp(gl[a] - gmax) / zg);
                maxTv = std::max(maxTv, tv / 2);
                if (rmax - second >= kArgmaxGap) {
                    ++decided;
                    agree += rbest == gbest;
                }
                maxV = std::max(maxV, std::fabs(ref.values[b] - got.values[b]));
            }
        }
        meanL /= static_cast<double>(c.count) * ACT;
        const double rel = maxL / maxRef;
        std::printf("%s accuracy %-24s %4d positions: logits maxErr %.2e (max |diff| %.3g of %.3g, mean %.2e), "
                    "value max |diff| %.2e, policy TV max %.2e, best move %d/%d\n",
                    backend.c_str(), c.name.c_str(), c.count, rel, maxL, maxRef, meanL, maxV, maxTv, agree, decided);
        assert(rel <= kLogitTier && maxV <= kValueTier && maxTv <= kPolicyTier && agree == decided);
    }
}

// A position's outputs are the same bits alone (its chunk padded), in
// passes of 8, 64 and 256 with random companions, at the first, a middle
// or the last row, and with the pass permuted; for ane also through each
// compiled size on its own.
void test_batch_independent(const std::string& backend, const std::vector<Case>& cs) {
    for (const Case& c : cs) {
        const NetConfig cfg = c.net->config();
        const size_t per = static_cast<size_t>(cfg.planes()) * cfg.area(), L = static_cast<size_t>(cfg.actions());
        CoreMLInference ci(*c.net, backend, 256);
        const int probes = std::min(c.count, 24);
        // Alone: each probe position evaluated by itself.
        std::vector<Outputs> alone;
        for (int i = 0; i < probes; ++i) alone.push_back(run(ci, cfg, c.positions.data() + i * per, 1));
        Rng rng(4242);
        int checked = 0;
        for (int B : {8, 64, 256}) {
            for (int trial = 0; trial < 3; ++trial) {
                // A pass: random companions from the case, the probes at chosen rows.
                std::vector<int> pick(B);
                for (int& p : pick) p = rng.below(static_cast<uint32_t>(c.count));
                const int rows[3] = {0, B / 2, B - 1};
                const int probe = rng.below(static_cast<uint32_t>(probes));
                pick[rows[trial]] = probe;
                if (trial == 2) {   // permuted: a shuffle of the whole pass
                    for (int i = B - 1; i > 0; --i) std::swap(pick[i], pick[rng.below(static_cast<uint32_t>(i + 1))]);
                }
                std::vector<float> x(static_cast<size_t>(B) * per);
                for (int b = 0; b < B; ++b)
                    std::copy_n(c.positions.data() + pick[b] * per, per, x.data() + b * per);
                const Outputs got = run(ci, cfg, x.data(), B);
                for (int b = 0; b < B; ++b) {
                    if (pick[b] >= probes) continue;
                    assert(bitwiseEqual(got.logits.data() + b * L, alone[pick[b]].logits.data(), L));
                    assert(bitwiseEqual(&got.values[b], alone[pick[b]].values.data(), 1));
                    ++checked;
                }
            }
        }
        // Each compiled size on its own (ane's sizes; coreml_gpu has one).
        for (int size : coremlBatchSizes(backend)) {
            const int B = std::min(probes, size);
            const Outputs got = run(ci, cfg, c.positions.data(), B, size);
            for (int b = 0; b < B; ++b) {
                assert(bitwiseEqual(got.logits.data() + b * L, alone[b].logits.data(), L));
                assert(bitwiseEqual(&got.values[b], alone[b].values.data(), 1));
                ++checked;
            }
        }
        std::printf("%s batch independence %s: %d rows bitwise equal to alone\n", backend.c_str(), c.name.c_str(),
                    checked);
    }
}

// Two loads of the same network (the cache emptied in between by a
// network that pushes it out) give the same bits; the cache serves a
// network still in use without compiling again.
void test_repeatable(const std::string& backend) {
    const NetConfig cfg = netConfig(9, 2, 2, 64);
    const Network net = randomNetwork(cfg, 61);
    const std::vector<float> x = randomGamePositions(cfg, 40, 62);
    Outputs first;
    {
        CoreMLInference ci(net, backend, 64);
        first = run(ci, cfg, x.data(), 40);
        const long long loaded = coreml_probe::modelsLoaded();
        CoreMLInference again(net, backend, 64);   // in use: no new models
        assert(coreml_probe::modelsLoaded() == loaded);
        const Outputs second = run(again, cfg, x.data(), 40);
        assert(second.logits == first.logits && second.values == first.values);
    }
    for (int k = 0; k < 5; ++k) {   // five other networks push it out of the recent four
        const Network other = randomNetwork(cfg, 70 + k);
        CoreMLInference ci(other, backend, 8);
    }
    const long long loaded = coreml_probe::modelsLoaded();
    CoreMLInference fresh(net, backend, 64);
    assert(coreml_probe::modelsLoaded() == loaded + static_cast<long long>(coremlBatchSizes(backend).size()));
    const Outputs third = run(fresh, cfg, x.data(), 40);
    assert(bitwiseEqual(third.logits.data(), first.logits.data(), first.logits.size()));
    assert(bitwiseEqual(third.values.data(), first.values.data(), first.values.size()));
    std::printf("%s repeatable: a reloaded network gives the same bits\n", backend.c_str());
}

// The hub's chaos test with the Core ML runner: every request's rows equal
// its positions evaluated alone, bitwise, whatever passes the hub made.
void test_hub_equal_alone_under_chaos(const std::string& backend) {
    const NetConfig cfg = netConfig(9, 2, 2, 64);
    const Network net = randomNetwork(cfg, 81);
    const int poolSize = 200;
    const std::vector<float> pool = randomGamePositions(cfg, poolSize, 82);
    const size_t in = static_cast<size_t>(cfg.planes()) * cfg.area(), L = static_cast<size_t>(cfg.actions());
    std::vector<float> refL(poolSize * L), refV(poolSize);
    {
        CoreMLInference ci(net, backend, 1);
        for (int i = 0; i < poolSize; ++i) ci.infer(pool.data() + i * in, 1, refL.data() + i * L, refV.data() + i);
    }
    for (int clients : {1, 4, 32}) {
        for (int maxBatch : {8, 64, 256}) {
            for (uint64_t chaos : {0ull, 11ull, 12ull}) {
                for (int slots : {1, 2}) {
                    InferenceHubOptions o;
                    o.maxBatch = maxBatch;
                    o.chaosSeed = chaos;
                    InferenceHub hub(makeCoreMLRunner(net, backend, maxBatch, slots), cfg, o);
                    std::atomic<long long> positions{0};
                    std::vector<std::thread> threads;
                    for (int t = 0; t < clients; ++t) {
                        threads.emplace_back([&, t] {
                            auto ev = hub.makeEvaluator(8);
                            Rng rng(deriveSeed(83, static_cast<uint64_t>(t), chaos, static_cast<uint64_t>(maxBatch)));
                            std::vector<float> x(8 * in), logits(8 * L), values(8);
                            for (int r = 0; r < 20; ++r) {
                                const int B = 1 + rng.below(8);
                                int picks[8];
                                for (int b = 0; b < B; ++b) {
                                    picks[b] = rng.below(poolSize);
                                    std::copy_n(pool.data() + picks[b] * in, in, x.data() + b * in);
                                }
                                ev->evaluate(x.data(), B, logits.data(), values.data());
                                for (int b = 0; b < B; ++b) {
                                    assert(bitwiseEqual(logits.data() + b * L, refL.data() + picks[b] * L, L));
                                    assert(bitwiseEqual(&values[b], &refV[picks[b]], 1));
                                }
                                positions += B;
                            }
                        });
                    }
                    for (auto& th : threads) th.join();
                    assert(hub.stats().positions == positions.load());
                }
            }
        }
    }
    std::printf("%s hub: every request equal to its positions alone, under chaos, 1 and 2 slots\n", backend.c_str());
}

// A failed prediction reaches every waiting request as "GPU inference
// failed", and later hubs work again; the memory check refuses up front.
void test_failures_and_memory(const std::string& backend) {
    const NetConfig cfg = netConfig(9, 2, 2, 64);
    const Network net = randomNetwork(cfg, 91);
    const std::vector<float> x = randomGamePositions(cfg, 8, 92);
    {
        InferenceHubOptions o;
        o.maxBatch = 64;
        InferenceHub hub(makeCoreMLRunner(net, backend, 64), cfg, o);
        auto ev = hub.makeEvaluator(8);
        std::vector<float> logits(8 * cfg.actions()), values(8);
        ev->evaluate(x.data(), 8, logits.data(), values.data());
        coreml_probe::failPassesAfter(1);
        ev->evaluate(x.data(), 8, logits.data(), values.data());   // the one that may still succeed
        bool failed = false;
        for (int i = 0; i < 4 && !failed; ++i) {
            try {
                ev->evaluate(x.data(), 8, logits.data(), values.data());
            } catch (const std::runtime_error& e) {
                failed = contains(e.what(), "GPU inference failed") && contains(e.what(), "injected failure");
            }
        }
        assert(failed);
        coreml_probe::failPassesAfter(0);
    }
    {
        InferenceHubOptions o;
        o.maxBatch = 8;
        InferenceHub hub(makeCoreMLRunner(net, backend, 8), cfg, o);
        auto ev = hub.makeEvaluator(8);
        std::vector<float> logits(8 * cfg.actions()), values(8);
        ev->evaluate(x.data(), 8, logits.data(), values.data());   // a new hub is fine again
    }
    coreml_probe::setMemoryLimit(1000000);
    bool threw = false;
    try {
        makeCoreMLRunner(net, backend, 256);
    } catch (const std::runtime_error& e) {
        threw = contains(e.what(), backend + ": inference needs") && contains(e.what(), "lower inference_batch");
    }
    assert(threw);
    threw = false;
    try {
        checkCoreMLInferenceMemory(cfg, backend, {{4, 2, 256}});
    } catch (const std::runtime_error& e) {
        threw = contains(e.what(), "(4 hubs)");
    }
    assert(threw);
    coreml_probe::setMemoryLimit(0);
    checkCoreMLInferenceMemory(netConfig(19, 8, 10, 128, 256), backend, {{2, 2, 256}, {2, 1, 256}});
    std::printf("%s failures and memory checks: ok\n", backend.c_str());
}

// ---- whole runs ------------------------------------------------------------------

struct RunDir {
    fs::path path;
    explicit RunDir(const std::string& name)
        : path(fs::temp_directory_path() / ("azgo_test_coreml_" + std::to_string(getpid()) + "_" + name)) {
        fs::remove_all(path);
    }
    ~RunDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string str() const { return path.string(); }
    std::string file(const std::string& rel) const { return (path / rel).string(); }
};

std::string readBytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    assert(in && "file should exist");
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// A small run whose network Core ML places on the device (9x9, history 2,
// 2 blocks of 64): CPU training, self-play and arena on the engine.
Config engineRunConfig(const std::string& backend) {
    Config c;
    c.backend = "blas";
    c.boardSize = 9;
    c.history = 2;
    c.blocks = 2;
    c.filters = 64;
    c.valueHidden = 16;
    c.gamesPerGeneration = 4;
    c.simulations = 16;
    c.mctsBatch = 4;
    c.temperatureMoves = 4;
    c.maxMoves = 60;
    c.bufferPositions = 2000;
    c.trainSteps = 2;
    c.batchSize = 16;
    c.arenaGames = 2;
    c.arenaSimulations = 8;
    c.arenaOpponents = 1;
    c.workers = 2;
    c.sgfPerGeneration = 1;
    c.botSimulations = {8};
    c.calibrationGames = 2;
    applyOverride(c, "selfplay_engines=local/" + backend + ":1");
    applyOverride(c, "arena_engines=local/" + backend + ":1");
    return c;
}

std::string runFingerprint(const RunDir& dir, int generations) {
    std::string out;
    for (int g = 0; g <= generations; ++g) out += readBytes(checkpointPath(dir.str(), g));
    for (const char* f : {"state.bin", "results.csv", "references.csv", "elo.csv"}) out += readBytes(dir.file(f));
    std::vector<std::string> sgf;
    for (const auto& e : fs::directory_iterator(dir.file("sgf"))) sgf.push_back(e.path().string());
    std::sort(sgf.begin(), sgf.end());
    for (const std::string& f : sgf) out += readBytes(f);
    return out;
}

void runOnce(const Config& c, const RunDir& dir, const std::string& hostsLine, int generations) {
    fs::create_directories(dir.path);
    {
        std::ofstream out(dir.file("hosts"));
        out << "local " << hostsLine << "\n";
    }
    TrainerOptions opts;
    opts.hostsFile = dir.file("hosts");
    std::ostringstream startup;
    std::unique_ptr<Trainer> t = Trainer::startNew(c, dir.str(), {}, &startup, opts);
    t->setLog(nullptr);
    t->runUntil(generations);
}

// The contract end to end: the same files whatever the engine's workers,
// inference_batch or the hub's chaos; resumed byte for byte; the manifest
// records the engine's fingerprint.
void test_runs(const std::string& backend) {
    std::string reference;
    const std::string key = backend == "ane" ? "ane_workers" : "coreml_gpu_workers";
    struct Variant { int workers, batch; uint64_t chaos; };
    for (const Variant& v : {Variant{0, 0, 0}, Variant{1, 8, 0}, Variant{3, 64, 5}, Variant{8, 256, 6}}) {
        RunDir dir("run_" + backend);
        coreml_probe::setHubChaosSeed(v.chaos);
        runOnce(engineRunConfig(backend), dir,
                key + "=" + std::to_string(v.workers) + (v.batch ? " inference_batch=" + std::to_string(v.batch) : ""), 2);
        coreml_probe::setHubChaosSeed(0);
        const std::string fp = runFingerprint(dir, 2);
        if (reference.empty()) {
            reference = fp;
            const std::string manifest = readBytes(dir.file("selfplay-engines.csv"));
            assert(contains(manifest, "local/" + backend) && contains(manifest, coremlDescription(backend)));
            assert(contains(readBytes(dir.file("arena-engines.csv")), coremlDescription(backend)));
        }
        assert(fp == reference);
    }
    RunDir a("resume_a_" + backend), b("resume_b_" + backend);
    runOnce(engineRunConfig(backend), a, "", 3);
    runOnce(engineRunConfig(backend), b, "", 1);
    {
        TrainerOptions opts;
        opts.hostsFile = b.file("hosts");
        std::unique_ptr<Trainer> t = Trainer::resume(b.str(), {}, nullptr, opts);
        t->setLog(nullptr);
        t->runUntil(3);
    }
    assert(runFingerprint(a, 3) == runFingerprint(b, 3));
    // The 32-bit engines play other bits: the comparison above is not vacuous.
    RunDir m("blas_" + backend);
    Config c = engineRunConfig(backend);
    applyOverride(c, "selfplay_engines=local/blas:1");
    applyOverride(c, "arena_engines=local/blas:1");
    runOnce(c, m, "", 2);
    assert(runFingerprint(m, 2) != reference);
    // metal beside coreml_gpu on one Mac: a note at start (Metal starves Core ML's GPU).
    if (backend == "coreml_gpu" && metalBuilt() && metalAvailable()) {
        RunDir d("note_" + backend);
        Config both = engineRunConfig(backend);
        applyOverride(both, "selfplay_engines=local/metal:1, local/coreml_gpu:1");
        fs::create_directories(d.path);
        {
            std::ofstream out(d.file("hosts"));   // its own hosts file, never the machine's
            out << "local\n";
        }
        TrainerOptions opts;
        opts.hostsFile = d.file("hosts");
        std::ostringstream startup;
        Trainer::startNew(both, d.str(), {}, &startup, opts);
        assert(contains(startup.str(), "note: local/metal and local/coreml_gpu share one GPU"));
    }
    std::printf("%s runs: independent of workers, batching and chaos; resumed byte for byte\n", backend.c_str());
}

#endif // AZ_USE_COREML

} // namespace

int main() {
    test_half_conversion();
    test_chunks();
    test_writer_structure();
    test_writer_deterministic();
    if (!coremlBuilt()) {
        test_not_built();
        std::printf("test_coreml: portable part passed; the device checks SKIPPED (not built)\n");
        return 77;
    }
#ifdef AZ_USE_COREML
    int ran = 0;
    const int perCase = std::getenv("AZGO_COREML_POSITIONS") ? std::max(8, std::atoi(std::getenv("AZGO_COREML_POSITIONS"))) : 128;
    const std::vector<Case> cs = cases(perCase);
    test_compiler_cache();
    for (const std::string backend : {"ane", "coreml_gpu"}) {
        if (!usable(backend)) continue;
        ++ran;
        std::printf("%s: %s\n", backend.c_str(), coremlDescription(backend).c_str());
        test_placement(backend, cs);
        test_accuracy(backend, cs);
        test_batch_independent(backend, cs);
        test_repeatable(backend);
        test_hub_equal_alone_under_chaos(backend);
        test_failures_and_memory(backend);
        test_runs(backend);
    }
    if (ran == 0) {
        std::printf("test_coreml: portable part passed; the device checks SKIPPED (no usable engine)\n");
        return 77;
    }
#endif
    std::printf("test_coreml: all tests passed\n");
    return 0;
}
