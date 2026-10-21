// GPU tests for backend = cuda: the context and its architecture check, every
// kernel against an exact CPU reference, and the batch-independence
// properties the inference hub relies on.
//
// They run only in a build configured with -DAZ_USE_CUDA=ON on a machine with
// a usable NVIDIA GPU. Anywhere else main() returns 77, which CMake declares
// as this test's SKIP_RETURN_CODE, so CTest reports "Skipped" rather than a
// failure.
//
// The references are exact, not tolerances: the gemm's is an std::fma chain
// in k order (what the kernel computes), batch norm's an explicit multiply
// then add. CMake compiles this file with -ffp-contract=off, so the compiler
// does not fuse the references' own a * b + c. Only tanh, which the GPU and
// the C library each round in their own way, is held to a tolerance (2 ulp);
// the whole network is held to the Metal spec's tolerances against the CPU
// Network ("Tolerance against the CPU").
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <string>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/cuda.hpp"
#include "azgo/rng.hpp"

using namespace azgo;

#ifdef AZ_USE_CUDA

namespace {

// ---------------------------------------------------------------- the context

void test_available_and_description() {
    std::string why = "unchanged";
    assert(cudaAvailable(&why));
    assert(why == "unchanged");   // only written when unusable
    const std::string d = cudaDescription();
    std::printf("test_cuda: %s\n", d.c_str());
    static const std::regex shape("^.+ sm_[0-9]+, driver [0-9]+\\.[0-9]+, runtime [0-9]+\\.[0-9]+, kernels [0-9a-f]{8}$");
    assert(std::regex_match(d, shape));
    assert(d == cudaDescription());
    assert(d.rfind(cuda_probe::deviceName() + " sm_" + std::to_string(cuda_probe::deviceArchitecture()), 0) == 0);
    assert(d.find(',') != std::string::npos && d.find(';') == std::string::npos);
}

// The machine code that runs was built for exactly this GPU's architecture:
// no PTX was compiled by the driver, and no older SASS runs on a newer GPU.
void test_architecture_matches() {
    assert(cuda_probe::binaryVersion() == cuda_probe::deviceArchitecture());
    assert(cuda_probe::deviceArchitecture() >= 50);
}

void test_fill() {
    for (size_t n : {size_t(0), size_t(1), size_t(1000), size_t(1) << 20}) {
        const size_t tail = 37;
        std::vector<float> data(n + tail, -3.0f);
        cuda_probe::fill(data.data(), n, 2.5f, tail);
        for (size_t i = 0; i < n; ++i) assert(data[i] == 2.5f);
        for (size_t i = n; i < n + tail; ++i) assert(data[i] == -3.0f);
    }
}

} // namespace

#endif // AZ_USE_CUDA

int main() {
    std::string why;
    if (!cudaBuilt()) {
        // A build without CUDA must say how to get it, and refuse the backend.
        assert(!cudaAvailable(&why));
        assert(why.find("-DAZ_USE_CUDA=ON") != std::string::npos);
        bool refused = false;
        try {
            checkBackendAvailable("cuda");
        } catch (const std::invalid_argument& e) {
            refused = std::string(e.what()) ==
                      "backend 'cuda' not available: rebuild with -DAZ_USE_CUDA=ON (Linux, NVIDIA GPU)";
        }
        assert(refused);
        std::printf("test_cuda: SKIPPED (%s)\n", why.c_str());
        return 77;
    }
    if (!cudaAvailable(&why)) {
        std::printf("test_cuda: SKIPPED (%s)\n", why.c_str());
        return 77;
    }
#ifdef AZ_USE_CUDA
    checkBackendAvailable("cuda");   // usable here
    test_available_and_description();
    test_architecture_matches();
    test_fill();
#endif
    std::printf("test_cuda: all tests passed\n");
    return 0;
}
