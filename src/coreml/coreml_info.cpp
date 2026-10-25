// What a build without Core ML answers about it (azgo/coreml.hpp). It is
// compiled in every build so that the engine list, the hosts file and the
// tools can name ane and coreml_gpu and refuse them with a message that
// says how to get them. A build with -DAZ_USE_COREML=ON defines these
// functions in coreml_runner.mm instead.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include "azgo/coreml.hpp"

#ifndef AZ_USE_COREML

namespace azgo {

namespace {
const char* const kNotBuilt = "not built: rebuild with -DAZ_USE_COREML=ON (macOS only)";
} // namespace

bool coremlBuilt() { return false; }

bool coremlAvailable(const std::string& /*backend*/, std::string* why) {
    if (why) *why = kNotBuilt;
    return false;
}

std::string coremlDescription(const std::string& /*backend*/) { return kNotBuilt; }

} // namespace azgo

#endif // !AZ_USE_COREML
