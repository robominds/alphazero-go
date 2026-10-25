// What a build without CUDA answers about backend = cuda16 (azgo/cuda16.hpp),
// as ../cuda_info.cpp does for cuda: compiled in every build, so config
// parsing, the engines list and the tools can name cuda16 and refuse it with
// a message that says how to get it. A build with -DAZ_USE_CUDA=ON defines
// these in cuda16_context.cu instead.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include "azgo/cuda16.hpp"

#ifndef AZ_USE_CUDA

namespace azgo {

namespace {
const char* const kNotBuilt = "not built: rebuild with -DAZ_USE_CUDA=ON (Linux, NVIDIA GPU)";
} // namespace

bool cuda16Available(std::string* why) {
    if (why) *why = kNotBuilt;
    return false;
}

std::string cuda16Description() { return kNotBuilt; }

} // namespace azgo

#endif // !AZ_USE_CUDA
