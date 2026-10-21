#!/usr/bin/env bash
# Builds tools/cuda_bench/build/cuda_bench with nvcc: one standalone CUDA
# file plus cuBLAS, no libazgo and no CMake. Needs the CUDA toolkit
# (CUDA_HOME, default /usr/local/cuda) and an NVIDIA GPU to run. The
# architecture defaults to sm_89 (RTX 40xx); override with CUDA_ARCH=sm_86.
#
# Mark Castelluccio, with assistance from Claude (Anthropic).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cuda="${CUDA_HOME:-/usr/local/cuda}"
arch="${CUDA_ARCH:-sm_89}"
mkdir -p "$here/build"
"$cuda/bin/nvcc" -std=c++17 -O3 -arch="$arch" -Xcompiler -Wall,-Wextra \
    "$here/cuda_bench.cu" -lcublas -o "$here/build/cuda_bench"
echo "built $here/build/cuda_bench"
