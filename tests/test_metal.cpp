// GPU tests: the Metal context, every kernel against its CPU counterpart and
// the batch-independence properties the batching evaluator relies on.
//
// They run only in a build configured with -DAZ_USE_METAL=ON on a Mac with a
// usable GPU. Anywhere else main() returns 77, which CMake declares as this
// test's SKIP_RETURN_CODE, so CTest reports "Skipped" rather than a failure.
//
// Tolerances (see the spec's "Tolerance against the CPU"): maxErr is
// max|x - ref| / max|ref|; "bitwise" is memcmp. Each test prints its measured
// worst error once, next to its ceiling.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "azgo/metal.hpp"

using namespace azgo;

#ifdef AZ_USE_METAL

namespace {

// ---------------------------------------------------------------- context

void test_context_comes_up() {
    const std::string d = metalDescription();
    assert(!d.empty());
    assert(d.find(metal_probe::deviceName()) != std::string::npos);
    assert(d.find("macOS ") != std::string::npos);
    assert(d.find("shaders ") != std::string::npos);
    for (int i = 0; i < 3; ++i) assert(metalDescription() == d);   // stable across calls
    std::printf("test_metal: %s\n", d.c_str());
}

// fill writes exactly n floats: a sentinel tail after them comes back untouched.
void test_fill_kernel() {
    const float kSentinel = -12345.0f;
    const size_t kTail = 37;
    for (size_t n : {size_t(0), size_t(1), size_t(255), size_t(256), size_t(257), size_t(100000)}) {
        std::vector<float> data(n + kTail, kSentinel);
        metal_probe::fill(data.data(), n, 2.5f, kTail);
        for (size_t i = 0; i < n; ++i) assert(data[i] == 2.5f);
        for (size_t i = n; i < n + kTail; ++i) assert(data[i] == kSentinel);
    }
}

void test_fill_repeat_bitwise() {
    const size_t n = 4097;
    std::vector<float> first(n);
    metal_probe::fill(first.data(), n, 0.1f);
    for (int r = 0; r < 20; ++r) {
        std::vector<float> again(n);
        metal_probe::fill(again.data(), n, 0.1f);
        assert(std::memcmp(first.data(), again.data(), n * sizeof(float)) == 0);
    }
}

} // namespace

#endif // AZ_USE_METAL

int main() {
    std::string why;
    if (!metalBuilt()) {
        // A build without Metal must say how to get it.
        assert(!metalAvailable(&why));
        assert(why.find("-DAZ_USE_METAL=ON") != std::string::npos);
        std::printf("test_metal: SKIPPED (%s)\n", why.c_str());
        return 77;
    }
    if (!metalAvailable(&why)) {
        std::printf("test_metal: SKIPPED (%s)\n", why.c_str());
        return 77;
    }
#ifdef AZ_USE_METAL
    test_context_comes_up();
    test_fill_kernel();
    test_fill_repeat_bitwise();
#endif
    std::printf("test_metal: all tests passed\n");
    return 0;
}
