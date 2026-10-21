// The process-wide CUDA context: device 0, checked once. The check is what
// keeps the bits fixed across driver updates: the library carries machine
// code (SASS) for named architectures only, never PTX that a driver would
// compile, and the context refuses a GPU whose architecture is not exactly
// one of them. It also owns the description that goes into an engine's
// fingerprint, and small RAII wrappers for device memory and streams.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <cstddef>
#include <string>
#include <cuda_runtime.h>

namespace azgo {
namespace cuda {

// Throws std::runtime_error("cuda: <what>: <error string>") unless e is cudaSuccess.
void check(cudaError_t e, const char* what);

class Context {
public:
    // The context, set up on the first call (thread-safe). Throws
    // std::runtime_error with the reason when the GPU is not usable.
    static const Context& instance();
    // Whether instance() would succeed; *why gets the reason when not.
    static bool usable(std::string* why);

    const std::string& deviceName() const { return name_; }
    int architecture() const { return architecture_; }     // 10 * major + minor
    int binaryVersion() const { return binaryVersion_; }   // of the machine code that runs
    const std::string& description() const { return description_; }
    size_t totalMemory() const { return totalMemory_; }

private:
    Context() = default;
    std::string name_, description_;
    int architecture_ = 0, binaryVersion_ = 0;
    size_t totalMemory_ = 0;
    friend struct ContextSetup;
};

// Device memory for `count` floats, freed with the object.
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count);
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept;
    float* data() const { return data_; }
    size_t count() const { return count_; }
    size_t bytes() const { return count_ * sizeof(float); }
private:
    float* data_ = nullptr;
    size_t count_ = 0;
};

// Page-locked host memory for `count` floats (fast, asynchronous copies).
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t count);
    ~PinnedBuffer();
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    float* data() const { return data_; }
    size_t count() const { return count_; }
private:
    float* data_ = nullptr;
    size_t count_ = 0;
};

// A stream of its own (not the legacy default stream), destroyed with the object.
class Stream {
public:
    Stream();
    ~Stream();
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return stream_; }
    // Waits for everything enqueued so far; throws on an error.
    void synchronize() const;
private:
    cudaStream_t stream_ = nullptr;
};

} // namespace cuda
} // namespace azgo
