// What a build without CUDA answers about it (azgo/cuda.hpp). It is compiled
// in every build so that config parsing, the engines list and the tools can
// name backend = cuda and refuse it with a message that says how to get it.
// A build with -DAZ_USE_CUDA=ON defines these functions in cuda_context.cu
// instead.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include "azgo/cuda.hpp"

#ifndef AZ_USE_CUDA

namespace azgo {

namespace {
const char* const kNotBuilt = "not built: rebuild with -DAZ_USE_CUDA=ON (Linux, NVIDIA GPU)";
} // namespace

bool cudaBuilt() { return false; }

bool cudaAvailable(std::string* why) {
    if (why) *why = kNotBuilt;
    return false;
}

std::string cudaDescription() { return kNotBuilt; }

} // namespace azgo

#endif // !AZ_USE_CUDA
