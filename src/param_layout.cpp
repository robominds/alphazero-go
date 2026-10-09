// ParamLayout (azgo/param_layout.hpp): params() and buffers() packed into two
// arenas, each tensor starting on a 64-float boundary.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include "azgo/param_layout.hpp"
#include "azgo/network.hpp"

namespace azgo {

namespace {

size_t alignUp(size_t floats) {
    return (floats + ParamLayout::kAlignFloats - 1) / ParamLayout::kAlignFloats * ParamLayout::kAlignFloats;
}

} // namespace

ParamLayout ParamLayout::of(Network& net) {
    ParamLayout layout;
    for (const ParamView& p : net.params()) {
        layout.params.push_back({p.name, p.size, layout.paramFloats, p.decay});
        layout.paramFloats = alignUp(layout.paramFloats + p.size);
    }
    for (const BufferView& b : net.buffers()) {
        layout.buffers.push_back({b.name, b.size, layout.bufferFloats, false});
        layout.bufferFloats = alignUp(layout.bufferFloats + b.size);
    }
    return layout;
}

const ParamLayout::Entry* ParamLayout::find(const std::string& name) const {
    for (const auto* list : {&params, &buffers})
        for (const Entry& e : *list)
            if (e.name == name) return &e;
    return nullptr;
}

} // namespace azgo
