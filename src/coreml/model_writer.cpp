// The Core ML model writer: a Network becomes an ML Program package with no
// converter in between.
//
// A Core ML model is a protobuf message (CoreML.Specification.Model,
// coremltools/mlmodel/format/*.proto). An ML Program's body is a typed
// operation graph (MIL: Model Intermediate Language): one function, "main",
// whose block is a list of operations, each naming its operator ("conv",
// "relu", ...), binding its inputs by name to earlier values and declaring
// the type and shape of its outputs. Constants are operations too
// ("const"); large ones keep their bytes in a separate blob file,
// weights/weight.bin, and point into it. An .mlpackage is a directory with
// the specification, the blob and a Manifest.json naming both.
//
// The protobuf wire format is simple enough to write by hand: every field
// is a tag (field number << 3 | wire type) followed by a varint (wire type
// 0) or a length-prefixed byte string (wire type 2: strings, bytes, nested
// messages, packed repeated numbers). That is all this file needs, so the
// project takes on no protobuf library. The field numbers below are those
// of Model.proto, FeatureTypes.proto and MIL.proto.
//
// The network, in MIL (n x n board, F filters, A = n * n points):
//   planes fp32 [B, P, n, n] -> cast fp16
//   stem:    conv 3x3 (BN folded) -> relu
//   block i: conv 3x3 -> relu -> conv 3x3 -> add skip -> relu
//   policy:  conv 1x1 to 2 -> relu -> reshape [B, 2A] -> linear to A + 1 -> cast fp32: "logits"
//   value:   conv 1x1 to 1 -> relu -> reshape [B, A] -> linear -> relu -> linear to 1
//            -> tanh -> reshape [B] -> cast fp32: "value"
// which is Network::forward in inference mode with batch norm folded:
// gamma * (conv(x) - mean) / sqrt(var + eps) + beta = conv'(x) + bias' with
// the weights of output channel c scaled by s_c = gamma_c / sqrt(var_c + eps)
// and bias'_c = beta_c - s_c * mean_c.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>
#include "azgo/coreml.hpp"
#include "azgo/layers.hpp"

namespace azgo {

// ---- float16 ------------------------------------------------------------------

uint16_t floatToHalf(float x) {
    uint32_t f;
    std::memcpy(&f, &x, 4);
    const uint32_t sign = (f >> 16) & 0x8000u;
    const uint32_t absf = f & 0x7fffffffu;
    if (absf >= 0x7f800000u)   // infinity or NaN (a NaN keeps a payload bit)
        return static_cast<uint16_t>(sign | 0x7c00u | (absf > 0x7f800000u ? 0x200u | ((absf >> 13) & 0x3ffu) : 0));
    if (absf >= 0x477ff000u)   // rounds to above 65504: infinity
        return static_cast<uint16_t>(sign | 0x7c00u);
    if (absf < 0x38800000u) {   // below 2^-14: a subnormal half, or zero
        // The value in units of 2^-24, rounded to nearest even.
        const uint32_t exp = absf >> 23;
        if (exp < 102) return static_cast<uint16_t>(sign);   // under half of 2^-24: zero
        const uint32_t mant = (absf & 0x7fffffu) | 0x800000u;
        const uint32_t shift = 126 - exp;                  // mant * 2^(exp-150) / 2^-24 = mant >> (126 - exp)
        uint32_t h = mant >> shift;
        const uint32_t rest = mant & ((1u << shift) - 1u), half = 1u << (shift - 1);
        if (rest > half || (rest == half && (h & 1u))) ++h;
        return static_cast<uint16_t>(sign | h);
    }
    // Normal: rebias the exponent (127 -> 15) and round 13 mantissa bits away.
    uint32_t h = ((absf >> 13) - (112u << 10));
    const uint32_t rest = absf & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;   // a carry may step into the exponent: still right
    return static_cast<uint16_t>(sign | h);
}

float halfToFloat(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {   // subnormal: mant * 2^-24
            const float v = static_cast<float>(mant) * (1.0f / 16777216.0f);
            std::memcpy(&f, &v, 4);
            f |= sign;
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

namespace {

// ---- protobuf wire format ---------------------------------------------------

class Pb {
public:
    const std::string& bytes() const { return b_; }
    Pb& varint(int field, uint64_t v) {
        tag(field, 0);
        raw(v);
        return *this;
    }
    Pb& str(int field, const std::string& s) {
        tag(field, 2);
        raw(s.size());
        b_ += s;
        return *this;
    }
    Pb& msg(int field, const Pb& m) { return str(field, m.bytes()); }
    // Packed repeated varints (proto3's default for repeated numbers).
    // int32 and int64 values are sign-extended to 64 bits, as protobuf does.
    Pb& packed(int field, const std::vector<int64_t>& values) {
        Pb inner;
        for (int64_t v : values) inner.raw(static_cast<uint64_t>(v));
        return str(field, inner.bytes());
    }
    // map<string, V> entries are messages {1: key, 2: value}.
    Pb& mapEntry(int field, const std::string& key, const Pb& value) {
        Pb e;
        e.str(1, key).msg(2, value);
        return msg(field, e);
    }

private:
    void tag(int field, int wire) { raw((static_cast<uint64_t>(field) << 3) | static_cast<uint64_t>(wire)); }
    void raw(uint64_t v) {
        while (v >= 0x80) {
            b_ += static_cast<char>((v & 0x7f) | 0x80);
            v >>= 7;
        }
        b_ += static_cast<char>(v);
    }
    std::string b_;
};

// MIL.proto's DataType, FeatureTypes.proto's ArrayDataType.
enum MilType { kString = 2, kFloat16 = 10, kFloat32 = 11, kInt32 = 23 };
constexpr uint64_t kArrayFloat32 = 65568;

// ---- the weight blob ---------------------------------------------------------
//
// MIL blob storage, version 2: a 64-byte header (u32 blob count, u32
// version, zeros), then per blob a 64-byte record (u32 0xDEADBEEF, u32 data
// type: 1 = float16, u64 size in bytes, u64 offset of the data in the file,
// zeros) and the data, each starting on a multiple of 64 bytes. A constant
// points at its record.
class Blob {
public:
    Blob() : bytes_(64, '\0') {}
    uint64_t add(const std::vector<uint16_t>& halves) {
        const uint64_t record = bytes_.size();
        const uint64_t size = halves.size() * 2, data = record + 64;
        std::string meta(64, '\0');
        put32(meta, 0, 0xDEADBEEFu);
        put32(meta, 4, 1u);   // float16
        put64(meta, 8, size);
        put64(meta, 16, data);
        bytes_ += meta;
        bytes_.append(reinterpret_cast<const char*>(halves.data()), size);   // little-endian, as the file is
        bytes_.resize((bytes_.size() + 63) / 64 * 64, '\0');
        ++count_;
        return record;
    }
    std::string finish() {
        put32(bytes_, 0, count_);
        put32(bytes_, 4, 2u);
        return bytes_;
    }

private:
    static void put32(std::string& s, size_t at, uint32_t v) {
        for (int i = 0; i < 4; ++i) s[at + i] = static_cast<char>((v >> (8 * i)) & 0xff);
    }
    static void put64(std::string& s, size_t at, uint64_t v) {
        for (int i = 0; i < 8; ++i) s[at + i] = static_cast<char>((v >> (8 * i)) & 0xff);
    }
    std::string bytes_;
    uint32_t count_ = 0;
};

// ---- MIL values and operations ---------------------------------------------

Pb tensorType(int dtype, const std::vector<int64_t>& shape) {
    Pb t;
    t.varint(1, static_cast<uint64_t>(dtype)).varint(2, shape.size());
    for (int64_t d : shape) {
        Pb constant, dim;
        constant.varint(1, static_cast<uint64_t>(d));
        dim.msg(1, constant);
        t.msg(3, dim);
    }
    return t;
}

Pb valueType(int dtype, const std::vector<int64_t>& shape) {
    Pb v;
    v.msg(1, tensorType(dtype, shape));
    return v;
}

Pb namedValue(const std::string& name, int dtype, const std::vector<int64_t>& shape) {
    Pb n;
    n.str(1, name).msg(2, valueType(dtype, shape));
    return n;
}

// A Value holding a tensor given inline: TensorValue field 2 (ints), 4
// (strings) or 7 (bytes).
Pb immediate(int dtype, const std::vector<int64_t>& shape, int tensorField, const Pb& values) {
    Pb tensor, imm, v;
    tensor.msg(tensorField, values);
    imm.msg(1, tensor);
    v.msg(2, valueType(dtype, shape)).msg(3, imm);
    return v;
}

Pb stringValue(const std::string& s) {
    Pb values;
    values.str(1, s);
    return immediate(kString, {}, 4, values);
}

class Program {
public:
    explicit Program(Blob& blob) : blob_(blob) {}

    // The const ops: a string, int32 tensors, and float16 tensors (in the
    // blob from 10 elements on, as coremltools does, else inline bytes).
    std::string constString(const std::string& name, const std::string& value) {
        return constOp(name, kString, {}, stringValue(value));
    }
    std::string constInts(const std::string& name, const std::vector<int64_t>& values, bool scalar = false) {
        Pb ints;
        ints.packed(1, values);
        const std::vector<int64_t> shape = scalar ? std::vector<int64_t>{} : std::vector<int64_t>{static_cast<int64_t>(values.size())};
        return constOp(name, kInt32, shape, immediate(kInt32, shape, 2, ints));
    }
    std::string constHalf(const std::string& name, const std::vector<int64_t>& shape, const std::vector<uint16_t>& h) {
        if (h.size() < 10) {
            Pb bytes;
            bytes.str(1, std::string(reinterpret_cast<const char*>(h.data()), h.size() * 2));
            return constOp(name, kFloat16, shape, immediate(kFloat16, shape, 7, bytes));
        }
        Pb file, v;
        file.str(1, "@model_path/weights/weight.bin").varint(2, blob_.add(h));
        v.msg(2, valueType(kFloat16, shape)).msg(5, file);
        return constOp(name, kFloat16, shape, v);
    }

    // An operation: its type, named inputs, one output.
    std::string op(const std::string& type, const std::string& name,
                   const std::vector<std::pair<std::string, std::string>>& inputs, int dtype,
                   const std::vector<int64_t>& shape) {
        Pb o;
        o.str(1, type);
        for (const auto& [key, value] : inputs) {
            Pb binding, argument;
            binding.str(1, value);
            argument.msg(1, binding);
            o.mapEntry(2, key, argument);
        }
        o.msg(3, namedValue(name, dtype, shape));
        o.mapEntry(5, "name", stringValue(name));
        ops_.push_back(o);
        return name;
    }

    // conv (bias folded in) with "same" (3x3) or "valid" (1x1) padding.
    std::string conv(const std::string& name, const std::string& x, const std::vector<int64_t>& wShape,
                     const std::vector<uint16_t>& w, const std::vector<uint16_t>& b,
                     const std::vector<int64_t>& outShape) {
        const bool k3 = wShape[2] == 3;
        const std::string padType = constString(name + "_pad_type", k3 ? "same" : "valid");
        const std::string strides = constInts(name + "_strides", {1, 1});
        const std::string dilations = constInts(name + "_dilations", {1, 1});
        const std::string pad = constInts(name + "_pad", {0, 0, 0, 0});
        const std::string groups = constInts(name + "_groups", {1}, true);
        const std::string weight = constHalf(name + "_weight", wShape, w);
        const std::string bias = constHalf(name + "_bias", {wShape[0]}, b);
        return op("conv", name,
                  {{"x", x}, {"weight", weight}, {"bias", bias}, {"strides", strides}, {"pad_type", padType},
                   {"pad", pad}, {"dilations", dilations}, {"groups", groups}},
                  kFloat16, outShape);
    }

    std::string serializeOps(Pb& block) const {
        for (const Pb& o : ops_) block.msg(3, o);
        return "";
    }

private:
    std::string constOp(const std::string& name, int dtype, const std::vector<int64_t>& shape, const Pb& val) {
        Pb o;
        o.str(1, "const");
        o.msg(3, namedValue(name, dtype, shape));
        o.mapEntry(5, "name", stringValue(name));
        o.mapEntry(5, "val", val);
        ops_.push_back(o);
        return name;
    }
    Blob& blob_;
    std::vector<Pb> ops_;
};

std::vector<uint16_t> toHalves(const std::vector<double>& v) {
    std::vector<uint16_t> h(v.size());
    // Through float first, as a float32 checkpoint converted by coremltools is.
    for (size_t i = 0; i < v.size(); ++i) h[i] = floatToHalf(static_cast<float>(v[i]));
    return h;
}

std::vector<uint16_t> toHalves(const std::vector<float>& v) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = floatToHalf(v[i]);
    return h;
}

// A deterministic UUID-shaped identifier from the package's bytes, so the
// same network always gives the same package.
std::string uuidFrom(uint64_t a, uint64_t b) {
    char s[40];
    std::snprintf(s, sizeof s, "%08X-%04X-4%03X-8%03X-%012llX", static_cast<unsigned>(a >> 32),
                  static_cast<unsigned>((a >> 16) & 0xffff), static_cast<unsigned>(a & 0xfff),
                  static_cast<unsigned>((b >> 48) & 0xfff), static_cast<unsigned long long>(b & 0xffffffffffffull));
    return s;
}

uint64_t fnv(const std::string& bytes, uint64_t hash = 0xcbf29ce484222325ull) {
    for (char c : bytes) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 0x100000001b3ull;
    }
    return hash;
}

} // namespace

// The layers the writer reads, by their names in Network::params() and
// buffers(): the Network keeps its layers private, and the named views are
// the stable interface (checkpoints use them too).
namespace {

struct NamedTensors {
    std::vector<ParamView> params;
    std::vector<BufferView> buffers;
    const float* get(const std::string& name, size_t size) const {
        for (const ParamView& p : params)
            if (p.name == name) {
                if (p.size != size) throw std::logic_error("coreml writer: " + name + " has an unexpected size");
                return p.value;
            }
        for (const BufferView& b : buffers)
            if (b.name == name) {
                if (b.size != size) throw std::logic_error("coreml writer: " + name + " has an unexpected size");
                return b.value;
            }
        throw std::logic_error("coreml writer: the network has no tensor " + name);
    }
};

// A convolution's weights [cout][cin * k * k] with its batch norm folded
// in, read by name.
void foldNamed(const NamedTensors& t, const std::string& conv, const std::string& bn, int cout, int cin, int k,
               std::vector<uint16_t>& w, std::vector<uint16_t>& b) {
    const size_t per = static_cast<size_t>(cin) * k * k, channels = static_cast<size_t>(cout);
    const float* cw = t.get(conv + ".w", channels * per);
    const float* gamma = t.get(bn + ".gamma", channels);
    const float* beta = t.get(bn + ".beta", channels);
    const float* mean = t.get(bn + ".runMean", channels);
    const float* var = t.get(bn + ".runVar", channels);
    std::vector<double> wf(channels * per), bf(channels);
    for (size_t c = 0; c < channels; ++c) {
        const double s = static_cast<double>(gamma[c]) /
                         std::sqrt(static_cast<double>(var[c]) + static_cast<double>(BatchNorm2d::kEps));
        for (size_t i = 0; i < per; ++i) wf[c * per + i] = static_cast<double>(cw[c * per + i]) * s;
        bf[c] = static_cast<double>(beta[c]) - s * static_cast<double>(mean[c]);
    }
    w = toHalves(wf);
    b = toHalves(bf);
}

} // namespace

CoreMLModelFiles buildCoreMLModel(const Network& net, int batch) {
    if (batch < 1) throw std::invalid_argument("buildCoreMLModel: batch must be >= 1");
    const NetConfig& cfg = net.config();
    const int64_t B = batch, P = cfg.planes(), n = cfg.boardSize, F = cfg.filters, A = cfg.area(),
                  ACT = cfg.actions(), V = cfg.valueHidden;
    NamedTensors t{const_cast<Network&>(net).params(), const_cast<Network&>(net).buffers()};

    Blob blob;
    Program prog(blob);
    std::vector<uint16_t> w, b;
    const std::vector<int64_t> trunk = {B, F, n, n};

    const std::string fp16 = prog.constString("planes_dtype", "fp16");
    std::string x = prog.op("cast", "planes_fp16", {{"x", "planes"}, {"dtype", fp16}}, kFloat16, {B, P, n, n});

    foldNamed(t, "stem.conv", "stem.bn", static_cast<int>(F), static_cast<int>(P), 3, w, b);
    x = prog.conv("stem_conv", x, {F, P, 3, 3}, w, b, trunk);
    x = prog.op("relu", "stem_relu", {{"x", x}}, kFloat16, trunk);
    for (int i = 0; i < cfg.blocks; ++i) {
        const std::string blk = "block" + std::to_string(i);
        foldNamed(t, blk + ".conv1", blk + ".bn1", static_cast<int>(F), static_cast<int>(F), 3, w, b);
        std::string y = prog.conv(blk + "_conv1", x, {F, F, 3, 3}, w, b, trunk);
        y = prog.op("relu", blk + "_relu1", {{"x", y}}, kFloat16, trunk);
        foldNamed(t, blk + ".conv2", blk + ".bn2", static_cast<int>(F), static_cast<int>(F), 3, w, b);
        y = prog.conv(blk + "_conv2", y, {F, F, 3, 3}, w, b, trunk);
        y = prog.op("add", blk + "_add", {{"x", y}, {"y", x}}, kFloat16, trunk);
        x = prog.op("relu", blk + "_relu2", {{"x", y}}, kFloat16, trunk);
    }

    // Policy head: [B, 2, n, n] flattened channel-major is the CPU's
    // policyFlatten ([2][B][A] -> [B][2A]).
    foldNamed(t, "policy.conv", "policy.bn", 2, static_cast<int>(F), 1, w, b);
    std::string p = prog.conv("policy_conv", x, {2, F, 1, 1}, w, b, {B, 2, n, n});
    p = prog.op("relu", "policy_relu", {{"x", p}}, kFloat16, {B, 2, n, n});
    const std::string pshape = prog.constInts("policy_flat_shape", {B, 2 * A});
    p = prog.op("reshape", "policy_flat", {{"x", p}, {"shape", pshape}}, kFloat16, {B, 2 * A});
    {
        const float* fw = t.get("policy.fc.w", static_cast<size_t>(ACT * 2 * A));
        const float* fb = t.get("policy.fc.b", static_cast<size_t>(ACT));
        const std::string weight = prog.constHalf("policy_fc_weight", {ACT, 2 * A},
                                                  toHalves(std::vector<float>(fw, fw + ACT * 2 * A)));
        const std::string bias = prog.constHalf("policy_fc_bias", {ACT}, toHalves(std::vector<float>(fb, fb + ACT)));
        p = prog.op("linear", "policy_fc", {{"x", p}, {"weight", weight}, {"bias", bias}}, kFloat16, {B, ACT});
    }
    const std::string fp32 = prog.constString("outputs_dtype", "fp32");
    prog.op("cast", "logits", {{"x", p}, {"dtype", fp32}}, kFloat32, {B, ACT});

    // Value head.
    foldNamed(t, "value.conv", "value.bn", 1, static_cast<int>(F), 1, w, b);
    std::string v = prog.conv("value_conv", x, {1, F, 1, 1}, w, b, {B, 1, n, n});
    v = prog.op("relu", "value_relu", {{"x", v}}, kFloat16, {B, 1, n, n});
    const std::string vshape = prog.constInts("value_flat_shape", {B, A});
    v = prog.op("reshape", "value_flat", {{"x", v}, {"shape", vshape}}, kFloat16, {B, A});
    {
        const float* w1 = t.get("value.fc1.w", static_cast<size_t>(V * A));
        const float* b1 = t.get("value.fc1.b", static_cast<size_t>(V));
        const float* w2 = t.get("value.fc2.w", static_cast<size_t>(V));
        const float* b2 = t.get("value.fc2.b", 1);
        const std::string weight1 = prog.constHalf("value_fc1_weight", {V, A}, toHalves(std::vector<float>(w1, w1 + V * A)));
        const std::string bias1 = prog.constHalf("value_fc1_bias", {V}, toHalves(std::vector<float>(b1, b1 + V)));
        v = prog.op("linear", "value_fc1", {{"x", v}, {"weight", weight1}, {"bias", bias1}}, kFloat16, {B, V});
        v = prog.op("relu", "value_fc1_relu", {{"x", v}}, kFloat16, {B, V});
        const std::string weight2 = prog.constHalf("value_fc2_weight", {1, V}, toHalves(std::vector<float>(w2, w2 + V)));
        const std::string bias2 = prog.constHalf("value_fc2_bias", {1}, toHalves(std::vector<float>(b2, b2 + 1)));
        v = prog.op("linear", "value_fc2", {{"x", v}, {"weight", weight2}, {"bias", bias2}}, kFloat16, {B, 1});
    }
    v = prog.op("tanh", "value_tanh", {{"x", v}}, kFloat16, {B, 1});
    const std::string flat = prog.constInts("value_out_shape", {B});
    v = prog.op("reshape", "value_out", {{"x", v}, {"shape", flat}}, kFloat16, {B});
    prog.op("cast", "value", {{"x", v}, {"dtype", fp32}}, kFloat32, {B});

    // Function main: the input, opset CoreML8 (macOS 15), the block.
    Pb block;
    block.str(2, "logits").str(2, "value");
    prog.serializeOps(block);
    Pb function;
    function.msg(1, namedValue("planes", kFloat32, {B, P, n, n})).str(2, "CoreML8").mapEntry(3, "CoreML8", block);
    Pb program;
    program.varint(1, 1).mapEntry(2, "main", function);

    // The model's interface: multi-arrays of float32.
    auto feature = [](const std::string& name, const std::vector<int64_t>& shape) {
        Pb array, type, fd;
        array.packed(1, shape).varint(2, kArrayFloat32);
        type.msg(5, array);
        fd.str(1, name).msg(3, type);
        return fd;
    };
    Pb description;
    description.msg(1, feature("planes", {B, P, n, n}));
    description.msg(10, feature("logits", {B, ACT}));
    description.msg(10, feature("value", {B}));
    Pb model;
    model.varint(1, 9).msg(2, description).msg(502, program);

    CoreMLModelFiles files;
    files.spec = model.bytes();
    files.weights = blob.finish();
    const uint64_t h1 = fnv(files.spec), h2 = fnv(files.weights, h1);
    const std::string specId = uuidFrom(h1, h2), weightsId = uuidFrom(h2, h1 ^ 0x5bd1e995ull);
    files.manifest =
        "{\n"
        "    \"fileFormatVersion\": \"1.0.0\",\n"
        "    \"itemInfoEntries\": {\n"
        "        \"" + specId + "\": {\n"
        "            \"author\": \"com.apple.CoreML\",\n"
        "            \"description\": \"CoreML Model Specification\",\n"
        "            \"name\": \"model.mlmodel\",\n"
        "            \"path\": \"com.apple.CoreML/model.mlmodel\"\n"
        "        },\n"
        "        \"" + weightsId + "\": {\n"
        "            \"author\": \"com.apple.CoreML\",\n"
        "            \"description\": \"CoreML Model Weights\",\n"
        "            \"name\": \"weights\",\n"
        "            \"path\": \"com.apple.CoreML/weights\"\n"
        "        }\n"
        "    },\n"
        "    \"rootModelIdentifier\": \"" + specId + "\"\n"
        "}\n";
    return files;
}

namespace {

void makeDir(const std::string& path) {
    if (::mkdir(path.c_str(), 0755) != 0) throw std::runtime_error("coreml: cannot create " + path);
}

void writeFile(const std::string& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    if (!out) throw std::runtime_error("coreml: cannot write " + path);
}

} // namespace

void writeCoreMLPackage(const Network& net, int batch, const std::string& dir) {
    const std::string suffix = ".mlpackage";
    if (dir.size() <= suffix.size() || dir.compare(dir.size() - suffix.size(), suffix.size(), suffix) != 0)
        throw std::invalid_argument("writeCoreMLPackage: " + dir + " does not end in .mlpackage");
    const CoreMLModelFiles files = buildCoreMLModel(net, batch);
    makeDir(dir);
    makeDir(dir + "/Data");
    makeDir(dir + "/Data/com.apple.CoreML");
    makeDir(dir + "/Data/com.apple.CoreML/weights");
    writeFile(dir + "/Manifest.json", files.manifest);
    writeFile(dir + "/Data/com.apple.CoreML/model.mlmodel", files.spec);
    writeFile(dir + "/Data/com.apple.CoreML/weights/weight.bin", files.weights);
}

uint64_t coremlNetworkHash(const Network& net) {
    const NetConfig& c = net.config();
    std::string shape;
    for (int v : {c.boardSize, c.history, c.blocks, c.filters, c.valueHidden, c.ownership ? 1 : 0})
        shape.append(reinterpret_cast<const char*>(&v), sizeof v);
    uint64_t h = fnv(shape);
    Network& n = const_cast<Network&>(net);
    for (const ParamView& p : n.params()) h = fnv(std::string(reinterpret_cast<const char*>(p.value), p.size * 4), fnv(p.name, h));
    for (const BufferView& b : n.buffers()) h = fnv(std::string(reinterpret_cast<const char*>(b.value), b.size * 4), fnv(b.name, h));
    return h;
}

std::string coremlModelFormatHash() {
    NetConfig cfg;
    cfg.boardSize = 5;
    cfg.history = 1;
    cfg.blocks = 1;
    cfg.filters = 8;
    cfg.valueHidden = 8;
    const Network net(cfg, 20261024);
    const CoreMLModelFiles f = buildCoreMLModel(net, 2);
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx",
                  static_cast<unsigned long long>(fnv(f.manifest, fnv(f.weights, fnv(f.spec)))));
    return std::string(hex, 8);
}

bool coremlBackend(const std::string& name) { return name == "ane" || name == "coreml_gpu"; }

std::vector<int> coremlBatchSizes(const std::string& backend) {
    if (backend == "ane") return {8, 32, 128};
    if (backend == "coreml_gpu") return {64};
    throw std::invalid_argument("coremlBatchSizes: not a Core ML backend: '" + backend + "'");
}

std::vector<int> coremlChunks(const std::vector<int>& sizes, int B) {
    if (B < 1) throw std::invalid_argument("coremlChunks: B must be >= 1");
    if (sizes.empty()) throw std::invalid_argument("coremlChunks: no sizes");
    for (size_t i = 0; i < sizes.size(); ++i)
        if (sizes[i] < 1 || (i > 0 && sizes[i] <= sizes[i - 1]))
            throw std::invalid_argument("coremlChunks: sizes must be ascending, distinct and >= 1");
    std::vector<int> chunks;
    int left = B;
    while (left > 0) {
        int pick = sizes[0];   // nothing fits what is left: the smallest, padded
        for (int s : sizes)
            if (s <= left) pick = s;
        if (pick > left) {     // the last chunk: the smallest size that holds the rest
            for (int s : sizes)
                if (s >= left) {
                    pick = s;
                    break;
                }
        }
        chunks.push_back(pick);
        left -= pick;
    }
    return chunks;
}

} // namespace azgo
