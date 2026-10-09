#!/usr/bin/env bash
# Builds tools/metal_bench/build/metal_bench: libazgo (unchanged, with BLAS)
# in its own CMake build directory, then the Objective-C++ benchmark linked
# against it. The main build is not touched. Needs macOS with Xcode (Metal).
#
# Mark Castelluccio, with assistance from Claude (Anthropic).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
build="$here/build"

cmake -S "$root" -B "$build/azgo" -DAZ_USE_BLAS=ON -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$build/azgo" --target azgo -j 4
clang++ -std=c++17 -O3 -DNDEBUG -fobjc-arc -Wall -Wextra \
    -DAZ_USE_BLAS -DACCELERATE_NEW_LAPACK -I"$root/include" \
    "$here/metal_bench.mm" "$build/azgo/libazgo.a" \
    -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework Accelerate \
    -o "$build/metal_bench"
echo "built $build/metal_bench"
