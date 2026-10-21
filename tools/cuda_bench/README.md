# cuda_bench

A measurement spike for the question "should azgo get a CUDA `BatchRunner`,
so an NVIDIA GPU can join distributed self-play?" It is the CUDA
counterpart of `tools/metal_bench`. It is not part of the library and not
built by the main CMake project. It does not link `libazgo` either: it
rebuilds the network's inference pass from its shapes, with random
weights, kernel for kernel as the Metal backend does it
(`docs/superpowers/specs/2026-10-09-metal-gpu-backend-design.md`,
"Forward kernels").

| section   | what it measures |
|-----------|------------------|
| `gemm`    | A hand-written tiled fp32 SGEMM in three tile shapes (32x256, 64x128 and 128x128 outputs per block of 256 threads) and cuBLAS, at the convolution gemm shapes of 9x9 6x64 and 6x96, 13x13 6x96 and 19x19 10x128 at batch 8, 144 and 256 (M = filters, K = filters · 9, or 18 · 9 for the stem, N = n² · batch). It also gives the largest error against a double-precision reference. |
| `det`     | Determinism. A sample's output columns alone are compared bitwise with the same columns inside batches of 8, 144 and 256 at three positions, for each hand tile shape and for cuBLAS. It also checks that a repeated run and the three tile shapes give the same bits. |
| `forward` | A whole inference pass: input transpose, im2col, gemm, batch norm, ReLU, the skip connection, both heads, tanh. It runs with the explicit im2col (`hand (auto)`), with each convolution fused into one implicit-GEMM kernel (`hand fused`), and with cuBLAS. It times one pass at a time with the copies in and out (one hub slot's latency) and two streams kept full (the hub's two slots), gives the gemm share of the GPU time, checks that a position's outputs are bitwise the same alone and in a batch, and compares the fused and cuBLAS outputs with the explicit hand pass. |

**The hand kernel** is the CUDA version of the Metal `gemm_simd`'s
contract. Each block computes one tile of C. K is walked from 0 to K in
steps of 16 through shared memory and never split. Each output is a single
`fmaf` chain in k order, and rows and columns past M, N or K load as zero.
An element's bits therefore depend only on its row of A and its column of
B. They do not depend on M, N, the tile shape, its tile, its neighbours or
the run, so a position's outputs cannot depend on its batch. The fused
kernel reads B through an implicit im2col and applies batch norm, the skip
connection and ReLU in its epilogue, in `bn_relu`'s order, so its bits
equal the explicit pass. Everything is fp32: no TF32, no tensor cores,
and cuBLAS stays in its default (fp32) math mode.

## Build and run

Needs the CUDA toolkit (`nvcc`, cuBLAS) and an NVIDIA GPU. It was written
for and measured on an RTX 4070 Ti SUPER (sm_89) under WSL2.

    tools/cuda_bench/build.sh                     # -> tools/cuda_bench/build/cuda_bench
    tools/cuda_bench/build/cuda_bench all         # gemm, det, forward
    tools/cuda_bench/build/cuda_bench forward --net 13x13 --seconds 1

Options: `--seconds S` is the minimum time per measurement (default 0.5).
`--net TEXT` keeps only the networks whose name contains the text.
`CUDA_ARCH=sm_86 tools/cuda_bench/build.sh` builds for another GPU.

The results and their analysis are in `docs/results/cuda-bench.txt`.

Mark Castelluccio, with assistance from Claude (Anthropic).
