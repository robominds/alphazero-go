// Distributed self-play, tested on one machine: the build identity every
// binary carries, so the coordinator can tell whether two binaries run the
// same code.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <cassert>
#include <cstdio>
#include <string>
#include "azgo/backend.hpp"
#include "azgo/build_info.hpp"
#include "azgo/metal.hpp"

using namespace azgo;

static bool isLowerHex(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}

// buildId() is "unknown", or the 40-hex commit, optionally followed by "+"
// and 8 hex of the uncommitted changes; shortBuildId() keeps 7 hex and the
// suffix. Both are compiled in, so they never change while a program runs.
static void test_build_id_shape() {
    const std::string id = buildId(), shortId = shortBuildId();
    assert(id == buildId() && shortId == shortBuildId());
    if (id == "unknown") {
        assert(shortId == "unknown");
        return;
    }
    const std::string commit = id.substr(0, 40);
    assert(commit.size() == 40 && isLowerHex(commit));
    const std::string suffix = id.substr(40);
    assert(suffix.empty() || (suffix.size() == 9 && suffix[0] == '+' && isLowerHex(suffix.substr(1))));
    assert(shortId == id.substr(0, 7) + suffix);
}

// The strongest backend the binary was built with.
static void test_build_options() {
    const std::string options = buildOptions();
    if (metalBuilt()) assert(options == "metal");
    else if (blasAvailable()) assert(options == "blas");
    else assert(options == "naive");
}

int main() {
    test_build_id_shape();
    test_build_options();
    std::printf("test_distributed: OK\n");
    return 0;
}
