// azgo/cuda16.hpp's availability and description in a build with
// -DAZ_USE_CUDA=ON. The device, its architecture check and the driver and
// runtime versions are the cuda backend's context (../cuda_context.cuh); this
// adds the tensor-core requirement and the 16-bit kernels' own hash.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <string>
#include "azgo/cuda.hpp"
#include "azgo/cuda16.hpp"
#include "../cuda_context.cuh"
// Generated at configure time (cmake/azgo_cuda.cmake): AZGO_CUDA16_KERNEL_HASH,
// 8 hex over src/cuda/*.cu, *.cuh, src/cuda/fp16/*.cu, *.cuh, nvcc's version,
// the architectures and the flags.
#include "azgo_cuda16_info.inc"

namespace azgo {

namespace {

// mma.sync.m16n8k16 with fp32 accumulation and cp.async: sm_80 (Ampere) on.
constexpr int kMinArchitecture = 80;

std::string versionText(int v) { return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10); }

} // namespace

bool cuda16Available(std::string* why) {
    if (!cuda::Context::usable(why)) return false;
    const int arch = cuda::Context::instance().architecture();
    if (arch < kMinArchitecture) {
        if (why)
            *why = "cuda16 needs tensor cores of sm_80 or newer; this GPU is sm_" + std::to_string(arch) +
                   " (use cuda)";
        return false;
    }
    return true;
}

std::string cuda16Description() {
    std::string why;
    if (!cuda16Available(&why)) return why;
    const cuda::Context& c = cuda::Context::instance();
    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    const char* type = kCuda16Type == Cuda16Type::Bf16 ? "bf16" : "fp16";
    return c.deviceName() + " sm_" + std::to_string(c.architecture()) + ", driver API " + versionText(driver) +
           ", runtime " + versionText(runtime) + ", " + type + " kernels " + AZGO_CUDA16_KERNEL_HASH;
}

} // namespace azgo
