// What a build without Metal answers about it (azgo/metal.hpp). It is
// compiled in every build so that config parsing, the trainer and the tools
// can name backend = metal and refuse it with a message that says how to get
// it. A build with -DAZ_USE_METAL=ON defines these functions in
// metal_context.mm instead.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include "azgo/metal.hpp"

#ifndef AZ_USE_METAL

namespace azgo {

namespace {
const char* const kNotBuilt = "not built: rebuild with -DAZ_USE_METAL=ON (macOS only)";
} // namespace

bool metalBuilt() { return false; }

bool metalAvailable(std::string* why) {
    if (why) *why = kNotBuilt;
    return false;
}

std::string metalDescription() { return kNotBuilt; }

} // namespace azgo

#endif // !AZ_USE_METAL
