# metal_bench

A measurement spike for the question "should azgo get a Metal GPU backend?"
It is not part of the library and not built by the main CMake project. It
links the unchanged `libazgo` (built with BLAS) and times, for the 9x9
6x64 and 6x96 networks and the 19x19 10x128 one:

| section  | what it measures |
|----------|------------------|
| `cpu1`   | The project's own `BlasBackend` (Accelerate, one thread, as each worker and the trainer use it): every conv gemm shape at batch 8 / 144 / 256 (and the backward dW / dX shapes at 256), a whole pass's gemms back to back against the real `Network::infer` / `Network::trainStep`, and 18 workers inferring at batch 8 at once (self-play's view). |
| `cpuall` | The same with Accelerate allowed to use all its threads. |
| `gpu`    | A hand-written tiled Metal gemm (`gemm_tiled`: scalar fma, 64x64 tile, 4x4 per thread), the same tiling on 8x8 simdgroup matrices (`gemm_simd`), and `MPSMatrixMultiplication`: one gemm per command buffer (commit + wait each time) and 20 per buffer; then a whole forward pass or training step (gemms plus im2col, col2im, BN/ReLU and BN-statistics kernels) in one command buffer, per op, and pipelined. |
| `seam`   | A Metal backend behind today's `Backend` seam: only gemm on the GPU, operands copied in and out, one commit + wait per call, everything else on the CPU, driving the real `Network`. |
| `det`    | Determinism: repeated runs bitwise, a sample's output columns alone vs inside a batch, the real `Network` through each backend (alone vs in a batch; `trainStep` gradients hashed over repeats) and the error against a double-precision reference. |

The shader source is embedded in `metal_bench.mm` and compiled at run time
(`newLibraryWithSource`, `MTLMathModeSafe`). All buffers use shared storage
(unified memory, no copies) except in `seam`, where the network's
`std::vector`s are not page-aligned and must be copied.

## Build and run

Needs macOS with Xcode (the Metal and MetalPerformanceShaders frameworks).

    tools/metal_bench/build.sh                    # -> tools/metal_bench/build/metal_bench
    tools/metal_bench/build/metal_bench cpu1      # one section per process
    tools/metal_bench/build/metal_bench cpuall
    tools/metal_bench/build/metal_bench gpu
    tools/metal_bench/build/metal_bench seam
    tools/metal_bench/build/metal_bench det

Options: `--seconds S` is the minimum time per measurement (default 0.5);
`--net 6x96` keeps only the networks whose name contains the text.

Run each section in its own process: `cpu1` calls `makeBackend("blas")`,
which pins Accelerate to one thread for the rest of the process, and
`cpuall` must not inherit that. CPU numbers need an otherwise idle machine
(`cpu1` takes about 1.5 minutes, `seam` 2.5, `gpu` several).

Results and the analysis from an M5 Max: `docs/results/metal-bench.txt`.

Mark Castelluccio, with assistance from Claude (Anthropic).
