// Where each of a network's tensors lives in one flat array. The GPU keeps
// the parameters (params() order) in one buffer and the batch-norm running
// statistics (buffers() order) in another, each tensor at its own offset, so
// uploading the network is one copy per tensor and a kernel finds a tensor
// by its offset. Offsets are multiples of 64 floats (256 bytes), the
// alignment a GPU buffer view needs. Plain C++, compiled in every build.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace azgo {

class Network;

struct ParamLayout {
    static constexpr size_t kAlignFloats = 64;
    struct Entry {
        std::string name;   // as in Network::params() / buffers()
        size_t size;        // floats
        size_t offset;      // floats from the start of its arena, a multiple of kAlignFloats
        bool decay;         // weight decay applies (params only; false for buffers)
    };
    std::vector<Entry> params, buffers;
    size_t paramFloats = 0, bufferFloats = 0;   // arena sizes, multiples of kAlignFloats

    // The layout of net's tensors. Network::params() is non-const (it hands
    // out writable views), hence the non-const reference; nothing is changed.
    static ParamLayout of(Network& net);
    // The entry called name in either list, or null.
    const Entry* find(const std::string& name) const;
};

} // namespace azgo
