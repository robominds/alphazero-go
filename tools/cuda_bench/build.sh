#!/usr/bin/env bash
# Builds tools/cuda_bench/build/cuda_bench with nvcc: one standalone CUDA
# file plus cuBLAS, no libazgo and no CMake; and build/half_bench, the 16-bit
# convolution of backend = cuda16 on its own. Needs the CUDA toolkit
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
# half_bench: the cuda16 convolution from src/cuda/fp16, with the library's
# CUDA flags (cmake/azgo_cuda.cmake), beside the cuda backend's fp32 kernel.
"$cuda/bin/nvcc" -std=c++17 -O3 -arch="$arch" --fmad=false --prec-div=true --prec-sqrt=true --ftz=false \
    -Xptxas=-O3 -Xcompiler -Wall,-Wextra -I "$here/../../include" \
    "$here/half_bench.cu" -o "$here/build/half_bench"
echo "built $here/build/half_bench"
