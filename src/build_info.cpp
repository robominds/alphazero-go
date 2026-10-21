#include "azgo/build_info.hpp"
#include "azgo/backend.hpp"
#include "azgo/cuda.hpp"
#include "azgo/metal.hpp"
// Generated at build time into the build directory: #define AZGO_BUILD_ID "...".
// Only this file includes it, so a new ID recompiles one file.
#include "azgo_build_info.inc"

namespace azgo {

std::string buildId() { return AZGO_BUILD_ID; }

std::string shortBuildId() {
    const std::string id = buildId();
    if (id == "unknown") return id;
    return id.substr(0, 7) + id.substr(40);   // the suffix, "+xxxxxxxx", or nothing
}

std::string buildOptions() {
    if (metalBuilt()) return "metal";
    if (cudaBuilt()) return "cuda";
    if (blasAvailable()) return "blas";
    return "naive";
}

} // namespace azgo
