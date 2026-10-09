// The Metal context, buffers and command batches (metal_context.hpp): the one
// file that talks to Metal's Objective-C API. Everything above it is C++.
//
// Errors never cross into C++ as Objective-C exceptions: each function checks
// Metal's NSError results and statuses and throws a C++ exception itself.
// Every function that calls Metal does so inside @autoreleasepool, because
// Metal hands out autoreleased objects and no thread here has a run loop to
// drain them. No C++ exception is thrown from inside a pool (unwinding would
// skip popping it): errors are noted inside and thrown after the pool closes.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_context.hpp"

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "azgo/metal.hpp"
#include "shader_types.h"

namespace azgo {

namespace {

// The embedded kernel source: shader_types.h + kernels.metal, written by CMake.
const char kShaderSource[] =
#include "azgo_metal_kernels.inc"
    ;

// Every kernel and its fixed threadgroup shape. All pipelines are created
// with the context, so a kernel that does not compile stops the program at
// its first GPU use, before a run writes anything.
struct KernelSpec {
    const char* name;
    Size3 threads;
};
const KernelSpec kKernels[] = {
    {"fill", {kElementThreads, 1, 1}},
    {"gemm_simd", {kGemmThreads, 1, 1}},
    {"transpose_input", {kElementThreads, 1, 1}},
    {"im2col3", {kIm2colGroupX, kIm2colGroupY, 1}},
    {"bn_infer", {kElementThreads, 1, 1}},
    {"relu_forward", {kElementThreads, 1, 1}},
    {"add_into", {kElementThreads, 1, 1}},
    {"policy_flatten", {kElementThreads, 1, 1}},
    {"bias_add_rows", {kElementThreads, 1, 1}},
    {"tanh_inplace", {kElementThreads, 1, 1}},
};

// 32-bit FNV-1a: a short fingerprint of the shader source for the description.
uint32_t fnv1a(const char* s, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; ++i) {
        h ^= static_cast<unsigned char>(s[i]);
        h *= 16777619u;
    }
    return h;
}

std::string text(NSError* e) {
    return e ? std::string(e.localizedDescription.UTF8String) : std::string("unknown error");
}

MTLSize mtl(Size3 s) { return MTLSizeMake(s.x, s.y, s.z); }

} // namespace

// ---------------------------------------------------------------- Pipeline

struct Pipeline::Impl {
    std::string name;
    Size3 threads;
    id<MTLComputePipelineState> state;
};

Pipeline::Pipeline(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Pipeline::~Pipeline() = default;
const std::string& Pipeline::name() const { return impl_->name; }
Size3 Pipeline::threads() const { return impl_->threads; }

// ---------------------------------------------------------------- GpuBuffer

struct GpuBuffer::Impl {
    id<MTLBuffer> buffer;
};

GpuBuffer::GpuBuffer() = default;
GpuBuffer::GpuBuffer(std::unique_ptr<Impl> impl, size_t bytes)
    : impl_(std::move(impl)), bytes_(bytes) {}
GpuBuffer::GpuBuffer(GpuBuffer&&) noexcept = default;
GpuBuffer& GpuBuffer::operator=(GpuBuffer&&) noexcept = default;
GpuBuffer::~GpuBuffer() {
    @autoreleasepool {
        impl_.reset();
    }
}
float* GpuBuffer::data() const { return impl_ ? static_cast<float*>(impl_->buffer.contents) : nullptr; }

BufView::BufView(const GpuBuffer& b, size_t offsetBytes) : buffer(&b), offset(offsetBytes) {
    if (offsetBytes % kViewAlignment != 0 || offsetBytes > b.bytes())
        throw std::logic_error("BufView: offset " + std::to_string(offsetBytes) +
                               " is not a multiple of 256 inside the buffer");
}

float* BufView::data() const {
    return reinterpret_cast<float*>(reinterpret_cast<char*>(buffer->data()) + offset);
}

// ---------------------------------------------------------------- MetalContext

struct MetalContext::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    std::map<std::string, std::unique_ptr<Pipeline>> pipelines;
    std::string description;
};

namespace {

std::once_flag gOnce;
MetalContext* gContext = nullptr;   // never destroyed: lives until exit
std::string gFailure;               // why the context could not be created

std::string osVersion() {
    const NSOperatingSystemVersion v = [[NSProcessInfo processInfo] operatingSystemVersion];
    return std::to_string(v.majorVersion) + "." + std::to_string(v.minorVersion) + "." +
           std::to_string(v.patchVersion);
}

} // namespace

namespace {

// Creates the device, queue, library and pipelines into m. Returns the reason
// the GPU is unusable, or "" on success. Runs inside an autorelease pool, so
// it reports errors instead of throwing.
std::string createContext(MetalContext::Impl& m) {
    m.device = MTLCreateSystemDefaultDevice();
    if (!m.device) return "no Metal device";
    const std::string name = m.device.name.UTF8String;
    // The gemm is built on simdgroup matrices, which arrived with Apple 7.
    if (![m.device supportsFamily:MTLGPUFamilyApple7])
        return name + " lacks Apple GPU family 7 (simdgroup matrices; M1 or later)";
    m.queue = [m.device newCommandQueue];
    if (!m.queue) return "could not create a Metal command queue";

    MTLCompileOptions* options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeSafe;   // no fast-math reassociation
    NSError* error = nil;
    id<MTLLibrary> library = [m.device newLibraryWithSource:@(kShaderSource) options:options error:&error];
    if (!library) return "shader compile failed: " + text(error);

    for (const KernelSpec& k : kKernels) {
        id<MTLFunction> fn = [library newFunctionWithName:@(k.name)];
        if (!fn) return std::string("kernel ") + k.name + " not found in the shaders";
        NSError* e = nil;
        id<MTLComputePipelineState> state = [m.device newComputePipelineStateWithFunction:fn error:&e];
        if (!state) return std::string("pipeline ") + k.name + ": " + text(e);
        const size_t need = k.threads.x * k.threads.y * k.threads.z;
        // Adapting a threadgroup to the device would silently change a
        // reduction's order, so a device that cannot run the fixed size is refused.
        if (state.maxTotalThreadsPerThreadgroup < need)
            return std::string("kernel ") + k.name + " needs " + std::to_string(need) +
                   " threads per threadgroup; " + name + " allows " +
                   std::to_string(state.maxTotalThreadsPerThreadgroup);
        // The gemm divides its threadgroup into simdgroups of 32 lanes.
        if (state.threadExecutionWidth != 32)
            return std::string("kernel ") + k.name + ": SIMD width " +
                   std::to_string(state.threadExecutionWidth) + ", expected 32";
        auto impl = std::make_unique<Pipeline::Impl>();
        impl->name = k.name;
        impl->threads = k.threads;
        impl->state = state;
        m.pipelines[k.name] = std::make_unique<Pipeline>(std::move(impl));
    }

    char hash[16];
    std::snprintf(hash, sizeof hash, "%08x", fnv1a(kShaderSource, std::strlen(kShaderSource)));
    m.description = name + ", macOS " + osVersion() + ", shaders " + hash;
    return "";
}

} // namespace

MetalContext::MetalContext() : impl_(std::make_unique<Impl>()) {
    std::string error;
    @autoreleasepool {
        error = createContext(*impl_);
    }
    if (!error.empty()) throw std::runtime_error(error);
}

MetalContext::~MetalContext() = default;

MetalContext& MetalContext::instance() {
    std::call_once(gOnce, [] {
        try {
            gContext = new MetalContext();
        } catch (const std::exception& e) {
            gFailure = e.what();
        }
    });
    if (!gContext) throw std::runtime_error(gFailure);
    return *gContext;
}

bool MetalContext::usable(std::string* why) {
    try {
        instance();
        return true;
    } catch (const std::exception& e) {
        if (why) *why = e.what();
        return false;
    }
}

const Pipeline& MetalContext::pipeline(const std::string& name) const {
    auto it = impl_->pipelines.find(name);
    if (it == impl_->pipelines.end()) throw std::logic_error("unknown Metal kernel '" + name + "'");
    return *it->second;
}

GpuBuffer MetalContext::newBuffer(size_t bytes) {
    auto impl = std::make_unique<GpuBuffer::Impl>();
    // Metal refuses a zero-length buffer; one float keeps data() valid.
    const size_t length = bytes == 0 ? sizeof(float) : bytes;
    @autoreleasepool {
        impl->buffer = [impl_->device newBufferWithLength:length options:MTLResourceStorageModeShared];
    }
    if (!impl->buffer)
        throw std::runtime_error("metal: could not allocate a buffer of " + std::to_string(bytes) +
                                 " bytes");
    std::memset(impl->buffer.contents, 0, length);   // never hand out uninitialized memory
    return GpuBuffer(std::move(impl), bytes);
}

std::string MetalContext::description() const { return impl_->description; }
void runInPool(const std::function<void()>& fn) {
    std::exception_ptr error;
    @autoreleasepool {
        try {
            fn();
        } catch (...) {
            error = std::current_exception();
        }
    }
    if (error) std::rethrow_exception(error);
}

size_t MetalContext::currentAllocatedSize() const {
    return static_cast<size_t>(impl_->device.currentAllocatedSize);
}

size_t MetalContext::recommendedWorkingSet() const {
    return static_cast<size_t>(impl_->device.recommendedMaxWorkingSetSize);
}

std::string MetalContext::deviceName() const {
    @autoreleasepool {
        return impl_->device.name.UTF8String;
    }
}

// ---------------------------------------------------------------- CommandBatch

namespace {

std::mutex gInjectMutex;
int gInjectAfter = -1;   // -1: off; otherwise command buffers left before the failing one

// Whether this command buffer is the one to fail (called once per commit).
bool injectFailure() {
    std::lock_guard<std::mutex> lock(gInjectMutex);
    if (gInjectAfter < 0) return false;
    if (gInjectAfter > 0) {
        --gInjectAfter;
        return false;
    }
    gInjectAfter = -1;   // one failure, then normal again
    return true;
}

const char* const kInjected = "injected failure (test)";

} // namespace

void setCommandBufferFailureAfter(int n) {
    std::lock_guard<std::mutex> lock(gInjectMutex);
    gInjectAfter = n > 0 ? n : -1;
}

struct CommandBatch::Impl {
    id<MTLCommandBuffer> commands;
    id<MTLComputeCommandEncoder> encoder;
    bool committed = false;
};

CommandBatch::CommandBatch() : impl_(std::make_unique<Impl>()) {
    MetalContext& ctx = MetalContext::instance();
    @autoreleasepool {
        impl_->commands = [ctx.impl().queue commandBuffer];
        if (impl_->commands)
            impl_->encoder = [impl_->commands computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    }
    if (!impl_->commands) throw std::runtime_error("metal: could not create a command buffer");
    if (!impl_->encoder) throw std::runtime_error("metal: could not create a compute encoder");
}

CommandBatch::~CommandBatch() {
    @autoreleasepool {
        // An abandoned batch (an exception while encoding) is ended, never committed.
        if (impl_ && !impl_->committed && impl_->encoder) [impl_->encoder endEncoding];
        impl_.reset();
    }
}

namespace {

// Throws std::logic_error for an argument that has no buffer behind it.
// Called before the autorelease pool opens.
void checkBuffers(const Pipeline& p, std::initializer_list<BufView> buffers) {
    int index = 0;
    for (const BufView& v : buffers) {
        if (!v.buffer || !v.buffer->data())
            throw std::logic_error("kernel " + p.name() + ": buffer " + std::to_string(index) +
                                   " is empty");
        ++index;
    }
}

void bind(id<MTLComputeCommandEncoder> enc, const Pipeline& p, std::initializer_list<BufView> buffers,
          const void* params, size_t paramBytes) {
    [enc setComputePipelineState:p.impl().state];
    NSUInteger index = 0;
    for (const BufView& v : buffers)
        [enc setBuffer:v.buffer->impl().buffer offset:v.offset atIndex:index++];
    if (paramBytes > 0) [enc setBytes:params length:paramBytes atIndex:index];
}

} // namespace

void CommandBatch::dispatch(const std::string& kernel, std::initializer_list<BufView> buffers,
                            const void* params, size_t paramBytes, Size3 threads) {
    if (impl_->committed) throw std::logic_error("CommandBatch: dispatch after commit");
    if (threads.x == 0 || threads.y == 0 || threads.z == 0) return;
    const Pipeline& p = MetalContext::instance().pipeline(kernel);
    checkBuffers(p, buffers);
    @autoreleasepool {
        bind(impl_->encoder, p, buffers, params, paramBytes);
        [impl_->encoder dispatchThreads:mtl(threads) threadsPerThreadgroup:mtl(p.threads())];
    }
}

void CommandBatch::dispatchGroups(const std::string& kernel, std::initializer_list<BufView> buffers,
                                  const void* params, size_t paramBytes, Size3 groups) {
    if (impl_->committed) throw std::logic_error("CommandBatch: dispatch after commit");
    if (groups.x == 0 || groups.y == 0 || groups.z == 0) return;
    const Pipeline& p = MetalContext::instance().pipeline(kernel);
    checkBuffers(p, buffers);
    @autoreleasepool {
        bind(impl_->encoder, p, buffers, params, paramBytes);
        [impl_->encoder dispatchThreadgroups:mtl(groups) threadsPerThreadgroup:mtl(p.threads())];
    }
}

void CommandBatch::commitAndWait() {
    if (impl_->committed) throw std::logic_error("CommandBatch: committed twice");
    std::string error;
    const bool inject = injectFailure();
    @autoreleasepool {
        impl_->committed = true;
        [impl_->encoder endEncoding];
        [impl_->commands commit];
        [impl_->commands waitUntilCompleted];
        if (impl_->commands.status != MTLCommandBufferStatusCompleted)
            error = "metal: command buffer failed: " + text(impl_->commands.error);
        else if (inject)
            error = std::string("metal: command buffer failed: ") + kInjected;
    }
    if (!error.empty()) throw std::runtime_error(error);
}

void CommandBatch::commit(std::function<void(const std::string& error)> onDone) {
    if (impl_->committed) throw std::logic_error("CommandBatch: committed twice");
    @autoreleasepool {
        impl_->committed = true;
        [impl_->encoder endEncoding];
        // The handler only reports the status: no C++ exception may unwind
        // through Metal's thread.
        const bool inject = injectFailure();
        [impl_->commands addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            std::string error;
            if (cb.status != MTLCommandBufferStatusCompleted)
                error = "command buffer failed: " + text(cb.error);
            else if (inject)
                error = std::string("command buffer failed: ") + kInjected;
            onDone(error);
        }];
        [impl_->commands commit];
    }
}

// ---------------------------------------------------------------- azgo/metal.hpp

bool metalBuilt() { return true; }

bool metalAvailable(std::string* why) { return MetalContext::usable(why); }

std::string metalDescription() {
    std::string why;
    if (!MetalContext::usable(&why)) return why;
    return MetalContext::instance().description();
}

} // namespace azgo
