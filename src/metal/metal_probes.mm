// The test probes declared in azgo/metal.hpp. Each one copies host arrays into
// fresh GPU buffers, encodes one kernel (or a layer's few kernels) into one
// command batch, waits, and copies the results back: the slow, simple path a
// test needs to compare a kernel with its CPU counterpart. The program itself
// never calls them.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <cstring>
#include <stdexcept>

#include "azgo/metal.hpp"
#include "metal_context.hpp"
#include "shader_types.h"

namespace azgo {
namespace metal_probe {

namespace {

// A new buffer holding a copy of count floats from src (zeros if src is null).
GpuBuffer upload(const float* src, size_t count) {
    GpuBuffer b = MetalContext::instance().newBuffer(count * sizeof(float));
    if (src && count > 0) std::memcpy(b.data(), src, count * sizeof(float));
    return b;
}

void download(const GpuBuffer& b, float* dst, size_t count) {
    if (count > 0) std::memcpy(dst, b.data(), count * sizeof(float));
}

} // namespace

std::string deviceName() {
    @autoreleasepool {
        return MetalContext::instance().deviceName();
    }
}

void fill(float* data, size_t n, float value, size_t tail) {
    @autoreleasepool {
        GpuBuffer b = upload(data, n + tail);
        CommandBatch batch;
        const FillParams p{static_cast<az_uint>(n), value};
        batch.dispatch("fill", {b}, p, {n, 1, 1});
        batch.commitAndWait();
        download(b, data, n + tail);
    }
}

} // namespace metal_probe
} // namespace azgo
