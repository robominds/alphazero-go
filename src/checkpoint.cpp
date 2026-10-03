#include "azgo/checkpoint.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "azgo/config.hpp"
#include "azgo/serialize.hpp"

namespace azgo {

namespace {

constexpr uint32_t kMagic = 0x4F475A41;   // bytes 'A' 'Z' 'G' 'O' when written little-endian
// Version 2 added the ownership flag to the header.
constexpr uint32_t kVersionPlain = 1, kVersionOwnership = 2;

// A problem found in the file's contents (as opposed to running out of data).
struct FormatError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// One tensor as the checkpoint stores it.
struct Tensor {
    std::string name;
    float* data;
    size_t size;
};

std::vector<Tensor> tensorsOf(Network& net) {
    std::vector<Tensor> out;
    for (const ParamView& p : net.params()) out.push_back({p.name, p.value, p.size});
    for (const BufferView& b : net.buffers()) out.push_back({b.name, b.value, b.size});
    return out;
}

NetConfig readHeader(BinaryReader& r) {
    if (r.u32() != kMagic) throw FormatError("not an AZGO checkpoint");
    const uint32_t version = r.u32();
    if (version != kVersionPlain && version != kVersionOwnership)
        throw FormatError("unsupported version " + std::to_string(version));
    NetConfig cfg;
    cfg.boardSize = r.i32();
    cfg.history = r.i32();
    cfg.blocks = r.i32();
    cfg.filters = r.i32();
    cfg.valueHidden = r.i32();
    if (version == kVersionOwnership) {
        const int32_t flag = r.i32();
        if (flag != 0 && flag != 1) throw FormatError("invalid ownership flag in header");
        cfg.ownership = flag == 1;
    }
    // Same ranges as the config file accepts; also keeps a corrupt header
    // from asking for an absurd allocation.
    if (cfg.boardSize < 5 || cfg.boardSize > 19 || cfg.history < 1 || cfg.history > 16 ||
        cfg.blocks < 0 || cfg.blocks > kMaxBlocks || cfg.filters < 1 || cfg.filters > kMaxFilters ||
        cfg.valueHidden < 1 || cfg.valueHidden > 1024) {
        throw FormatError("invalid network shape in header");
    }
    return cfg;
}

// Opens path and runs body(reader, stream). Every failure becomes a
// runtime_error that names the file: the reader's own "unexpected end of
// data" means the file was cut short.
template <typename Body>
auto withCheckpoint(const std::string& path, Body body) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open checkpoint " + path);
    BinaryReader reader(in);
    try {
        return body(reader, in);
    } catch (const FormatError& e) {
        throw std::runtime_error(path + ": " + e.what());
    } catch (const std::runtime_error&) {
        throw std::runtime_error(path + ": truncated");
    }
}

} // namespace

void saveNetwork(Network& net, const std::string& path) {
    std::ostringstream buffer(std::ios::binary);
    BinaryWriter w(buffer);
    const NetConfig& cfg = net.config();
    w.u32(kMagic);
    w.u32(cfg.ownership ? kVersionOwnership : kVersionPlain);
    w.i32(cfg.boardSize);
    w.i32(cfg.history);
    w.i32(cfg.blocks);
    w.i32(cfg.filters);
    w.i32(cfg.valueHidden);
    if (cfg.ownership) w.i32(1);
    const std::vector<Tensor> tensors = tensorsOf(net);
    w.u32(static_cast<uint32_t>(tensors.size()));
    for (const Tensor& t : tensors) {
        w.str(t.name);
        w.u64(t.size);
        w.floats(t.data, t.size);
    }
    writeFileAtomically(path, buffer.str());
}

Network loadNetwork(const std::string& path) {
    return withCheckpoint(path, [](BinaryReader& r, std::istream& in) {
        Network net(readHeader(r), 0);   // the seed is irrelevant: every tensor is overwritten
        const std::vector<Tensor> tensors = tensorsOf(net);
        const uint32_t count = r.u32();
        if (count != tensors.size()) {
            throw FormatError("has " + std::to_string(count) + " tensors but the network needs " +
                              std::to_string(tensors.size()));
        }
        for (size_t i = 0; i < tensors.size(); ++i) {
            const std::string name = r.str();
            if (name != tensors[i].name) {
                throw FormatError("tensor " + std::to_string(i) + " is '" + name + "' but expected '" +
                                  tensors[i].name + "'");
            }
            const uint64_t size = r.u64();
            if (size != tensors[i].size) {
                throw FormatError("tensor '" + name + "' has " + std::to_string(size) +
                                  " values but expected " + std::to_string(tensors[i].size));
            }
            r.floats(tensors[i].data, tensors[i].size);
        }
        if (in.peek() != std::char_traits<char>::eof()) throw FormatError("unexpected data after the last tensor");
        return net;
    });
}

NetConfig readCheckpointConfig(const std::string& path) {
    return withCheckpoint(path, [](BinaryReader& r, std::istream&) { return readHeader(r); });
}

} // namespace azgo
