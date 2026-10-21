// The CUDA context (cuda_context.cuh) and azgo/cuda.hpp's answers in a build
// with -DAZ_USE_CUDA=ON.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include "cuda_context.cuh"
#include <memory>
#include <stdexcept>
#include <vector>
#include "azgo/cuda.hpp"
#include "kernels.cuh"
// Generated at configure time (cmake/azgo_cuda.cmake): AZGO_CUDA_KERNEL_HASH
// (8 hex: the kernel sources, nvcc's version, the architectures and the
// flags) and AZGO_CUDA_ARCHITECTURES ("89-real").
#include "azgo_cuda_info.inc"

namespace azgo {
namespace cuda {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("cuda: ") + what + ": " + cudaGetErrorString(e));
}

namespace {

// "13.3" from 13030.
std::string versionText(int v) { return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10); }

} // namespace

// Sets the context up once; holds either the context or the reason it is
// unusable.
struct ContextSetup {
    std::unique_ptr<Context> context;
    std::string why;

    ContextSetup() {
        try {
            context.reset(new Context());
            setUp(*context);
        } catch (const std::exception& e) {
            context.reset();
            why = e.what();
        }
    }

    static void setUp(Context& c) {
        int devices = 0;
        const cudaError_t e = cudaGetDeviceCount(&devices);
        if (e != cudaSuccess) throw std::runtime_error(std::string("no CUDA device: ") + cudaGetErrorString(e));
        if (devices < 1) throw std::runtime_error("no CUDA device");
        check(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp prop;
        check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
        c.name_ = prop.name;
        c.architecture_ = 10 * prop.major + prop.minor;
        c.totalMemory_ = prop.totalGlobalMem;
        const std::string sm = "sm_" + std::to_string(c.architecture_);
        // The machine code the runtime picked for this device. With SASS-only
        // binaries there is none for an architecture that was not built, and a
        // newer minor version of the same major would run older SASS: both
        // are refused, so the GPU in a fingerprint always ran code built for it.
        cudaFuncAttributes attributes;
        const cudaError_t found = fillAttributes(&attributes);
        if (found != cudaSuccess) {
            cudaGetLastError();   // clear it: it is not sticky
            throw std::runtime_error(std::string("no kernels for this GPU (") + sm + "; built for " +
                                     AZGO_CUDA_ARCHITECTURES + "): reconfigure with -DAZ_CUDA_ARCHITECTURES=" +
                                     std::to_string(c.architecture_) + "-real");
        }
        c.binaryVersion_ = attributes.binaryVersion;
        if (c.binaryVersion_ != c.architecture_)
            throw std::runtime_error("kernels built for sm_" + std::to_string(c.binaryVersion_) + "; this GPU is " +
                                     sm + ": reconfigure with -DAZ_CUDA_ARCHITECTURES=" +
                                     std::to_string(c.architecture_) + "-real");
        // A first kernel, through memory, a stream and back.
        {
            DeviceBuffer probe(16);
            Stream stream;
            launchFill(probe.data(), 16, 1.5f, stream.get());
            check(cudaGetLastError(), "fill");
            float host[16] = {};
            check(cudaMemcpyAsync(host, probe.data(), sizeof host, cudaMemcpyDeviceToHost, stream.get()),
                  "cudaMemcpyAsync");
            stream.synchronize();
            for (float v : host)
                if (v != 1.5f) throw std::runtime_error("the first kernel wrote wrong values");
        }
        int driver = 0, runtime = 0;
        check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
        check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
        // "driver API": the CUDA API version the installed driver supports
        // (cudaDriverGetVersion), not the NVIDIA driver's build number, which
        // only NVML reports; this binary links no NVML.
        c.description_ = c.name_ + " " + sm + ", driver API " + versionText(driver) + ", runtime " +
                         versionText(runtime) + ", kernels " + AZGO_CUDA_KERNEL_HASH;
    }
};

namespace {
const ContextSetup& setup() {
    static const ContextSetup s;
    return s;
}
} // namespace

const Context& Context::instance() {
    const ContextSetup& s = setup();
    if (!s.context) throw std::runtime_error("cuda: " + s.why);
    return *s.context;
}

bool Context::usable(std::string* why) {
    const ContextSetup& s = setup();
    if (!s.context && why) *why = s.why;
    return s.context != nullptr;
}

// ---------------------------------------------------------------- memory and streams

DeviceBuffer::DeviceBuffer(size_t count) : count_(count) {
    check(cudaMalloc(reinterpret_cast<void**>(&data_), (count > 0 ? count : 1) * sizeof(float)), "cudaMalloc");
}

DeviceBuffer::~DeviceBuffer() {
    if (data_) cudaFree(data_);
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& o) noexcept : data_(o.data_), count_(o.count_) {
    o.data_ = nullptr;
    o.count_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
        if (data_) cudaFree(data_);
        data_ = o.data_;
        count_ = o.count_;
        o.data_ = nullptr;
        o.count_ = 0;
    }
    return *this;
}

PinnedBuffer::PinnedBuffer(size_t count) : count_(count) {
    check(cudaHostAlloc(reinterpret_cast<void**>(&data_), (count > 0 ? count : 1) * sizeof(float),
                        cudaHostAllocDefault),
          "cudaHostAlloc");
}

PinnedBuffer::~PinnedBuffer() {
    if (data_) cudaFreeHost(data_);
}

Stream::Stream() { check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate"); }

Stream::~Stream() {
    if (stream_) cudaStreamDestroy(stream_);
}

void Stream::synchronize() const { check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize"); }

} // namespace cuda

// ---------------------------------------------------------------- azgo/cuda.hpp

bool cudaBuilt() { return true; }

bool cudaAvailable(std::string* why) { return cuda::Context::usable(why); }

std::string cudaDescription() {
    std::string why;
    if (!cuda::Context::usable(&why)) return why;
    return cuda::Context::instance().description();
}

} // namespace azgo
