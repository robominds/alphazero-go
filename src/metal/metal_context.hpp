// The Metal plumbing every GPU layer is built on: one process-wide context
// (device, command queue, compiled kernels), buffers in unified memory and
// command batches that encode kernels and run them.
//
// Plain C++: each Metal object sits behind a pointer to an implementation
// struct defined in metal_context.mm, the only file that talks to Metal
// directly. The layer code (gpu_layers.mm) encodes kernels through
// CommandBatch::dispatch, naming a kernel and its buffers and parameters.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>

namespace azgo {

// A grid or threadgroup shape, in threads or in threadgroups.
struct Size3 {
    size_t x = 1, y = 1, z = 1;
};

// A compiled kernel and its fixed threadgroup shape (shader_types.h).
class Pipeline {
public:
    struct Impl;
    explicit Pipeline(std::unique_ptr<Impl> impl);
    ~Pipeline();
    const std::string& name() const;
    Size3 threads() const;          // threads per threadgroup, fixed per kernel
    Impl& impl() const { return *impl_; }
private:
    std::unique_ptr<Impl> impl_;
};

// A buffer in unified memory (MTLResourceStorageModeShared): the CPU reads
// and writes it through data(), with no copy. Zero-filled when created.
// Move-only; the memory is released with the last owner.
class GpuBuffer {
public:
    struct Impl;
    GpuBuffer();
    explicit GpuBuffer(std::unique_ptr<Impl> impl, size_t bytes);
    GpuBuffer(GpuBuffer&&) noexcept;
    GpuBuffer& operator=(GpuBuffer&&) noexcept;
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
    ~GpuBuffer();
    float* data() const;
    size_t bytes() const { return bytes_; }
    Impl& impl() const { return *impl_; }
private:
    std::unique_ptr<Impl> impl_;
    size_t bytes_ = 0;
};

// A kernel argument: a buffer and a byte offset into it, a multiple of 256
// (kViewAlignment) so any tensor can start any view.
constexpr size_t kViewAlignment = 256;
struct BufView {
    const GpuBuffer* buffer = nullptr;
    size_t offset = 0;
    BufView() = default;
    BufView(const GpuBuffer& b, size_t offsetBytes = 0);   // NOLINT: a buffer is a view at 0
    float* data() const;                                     // the CPU's view of the same memory
};

// The process-wide Metal context, created on first use and alive until exit.
// It holds no run state: hubs, networks and engines own their buffers.
class MetalContext {
public:
    // The context; throws std::runtime_error with the reason when the GPU is
    // not usable (no device, family below Apple 7, a kernel failed to compile).
    static MetalContext& instance();
    // Creates the context if needed; false and the reason when it is unusable.
    static bool usable(std::string* why);

    const Pipeline& pipeline(const std::string& name) const;   // throws std::logic_error if unknown
    GpuBuffer newBuffer(size_t bytes);   // throws std::runtime_error naming the bytes on failure
    std::string description() const;     // device, macOS version, shader hash
    std::string deviceName() const;
    // Bytes the device has allocated so far, all buffers of the process.
    size_t currentAllocatedSize() const;
    // Metal's estimate of how much memory the GPU can use without hurting
    // performance (device.recommendedMaxWorkingSetSize), in bytes.
    size_t recommendedWorkingSet() const;

    struct Impl;
    Impl& impl() const { return *impl_; }
    ~MetalContext();
private:
    MetalContext();
    std::unique_ptr<Impl> impl_;
};

// Runs fn inside an autorelease pool, so the Metal objects it creates are
// released when it returns. An exception from fn is caught inside the pool
// and rethrown after the pool is closed (unwinding through @autoreleasepool
// would skip popping it). Every entry point that calls Metal goes through it
// or follows the same pattern.
void runInPool(const std::function<void()>& fn);

// Failure injection for tests (metal_probe::failCommandBuffersAfter): after n
// more command buffers complete normally, the next one reports a failure as
// if the GPU had; n = 0 disables it.
void setCommandBufferFailureAfter(int n);

// Test hooks: a stand-in for the device's recommended working set (0: none),
// and the number of command buffers committed so far.
void setWorkingSetOverride(size_t bytes);
long long commandBuffersCommitted();

// One command buffer with one compute encoder whose dispatches run in order,
// each seeing the previous ones' results (serial dispatch, default hazard
// tracking). Encode kernels with dispatch, then commit once.
class CommandBatch {
public:
    CommandBatch();
    ~CommandBatch();
    CommandBatch(const CommandBatch&) = delete;
    CommandBatch& operator=(const CommandBatch&) = delete;

    // Runs `kernel` over a grid of `threads` threads, in threadgroups of the
    // kernel's fixed shape (the last ones may be partial). Buffers are bound
    // at indices 0, 1, ... in order and the parameter struct after them. An
    // empty grid encodes nothing.
    void dispatch(const std::string& kernel, std::initializer_list<BufView> buffers,
                  const void* params, size_t paramBytes, Size3 threads);
    // The same over a grid of whole threadgroups (the gemm's tiles).
    void dispatchGroups(const std::string& kernel, std::initializer_list<BufView> buffers,
                        const void* params, size_t paramBytes, Size3 groups);
    template <class Params>
    void dispatch(const std::string& kernel, std::initializer_list<BufView> buffers,
                  const Params& params, Size3 threads) {
        dispatch(kernel, buffers, &params, sizeof params, threads);
    }
    template <class Params>
    void dispatchGroups(const std::string& kernel, std::initializer_list<BufView> buffers,
                        const Params& params, Size3 groups) {
        dispatchGroups(kernel, buffers, &params, sizeof params, groups);
    }

    // Ends encoding, commits and waits; throws std::runtime_error with the
    // GPU's error text if the command buffer failed.
    void commitAndWait();
    // Ends encoding and commits; onDone(error) runs on a Metal thread when the
    // GPU has finished, error empty on success. It must not throw.
    void commit(std::function<void(const std::string& error)> onDone);

    struct Impl;
private:
    std::unique_ptr<Impl> impl_;
};

} // namespace azgo
