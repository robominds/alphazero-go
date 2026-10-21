# CUDA Inference Backend — Design

Author: Mark Castelluccio, with design assistance from Claude (Anthropic).

This design builds on three others. From
`docs/superpowers/specs/2026-10-09-metal-gpu-backend-design.md` it takes
the BatchRunner and InferenceHub, the forward kernels, the determinism
contract and the tolerances against the CPU, and repeats them for NVIDIA
GPUs. From `docs/superpowers/specs/2026-10-18-distributed-selfplay-design.md`
it takes the engines, the hosts file, cluster_sync and fingerprints, and
adds one backend to them. The numbers come from the CUDA spike in
`tools/cuda_bench` and `docs/results/cuda-bench.txt`.

The algorithm does not change. A game played by `pc/cuda` is the same
AlphaGo Zero self-play game, seeded the same way, from the same network.
Only the hardware that evaluates the network is new, so the last bits of
its evaluations differ, as they do between `blas` and `metal`.

## Problem

Self-play is ~70% of a 13×13 6×96 generation: 2,967 s of 4,188 s at
generation 18, with arenas taking most of the rest. The three Macs
evaluate about 19,000 positions per second together. The Studio's
local/metal engine does 13,300–14,300 evals/s in real self-play, and each
M4 mini about 2,400 (that run's `selfplay-engines.csv`).

A PC with an RTX 4070 Ti SUPER (16 GB, sm_89) is now on the network,
running Ubuntu 24.04 under WSL2. Today it can only join as a CPU engine
(`pc/blas`, ~1,900 evals/s with 8 workers). The spike measured its GPU on
the same inference pass:

| 13×13 6×96, batch 256, two slots kept full | evals/s |
|---|---|
| hand kernels, explicit im2col (the Metal structure) | 19,100 |
| hand kernels, each convolution fused (implicit GEMM + BN/skip/ReLU) | **30,800** |
| cuBLAS + im2col (not batch-independent) | 24,500 |
| Studio M5 Max, Metal hub (real self-play) | 14,200 |

The fused pass is batch-independent and bitwise equal to the explicit one.
A CUDA engine would be 2.1× the Studio's GPU and ~1.6× the current
cluster, cutting 13×13 self-play from ~2,970 s to ~1,130 s per generation.

## Goals

1. `backend = cuda`: inference on an NVIDIA GPU. It is a `BatchRunner`
   behind the existing `InferenceHub` and `WorkerEvaluators`, as Metal's
   runner is. The hub, MCTS, self-play slices, job and result files and
   the coordinator are unchanged apart from accepting the name.
2. The distributed engine `pc/cuda`. The Studio coordinator plans slices
   for it, the PC's `selfplay` plays them, and results are verified and
   merged as for every remote.
3. The Metal determinism contract, unchanged in wording, on one machine
   with one GPU model and one build:
   - reproducible;
   - independent of workers, `inference_batch`, slot timing and batch
     composition;
   - byte-identical resume.
4. Every kernel is checked against an exact CPU oracle (bitwise), except
   tanh, which is checked within a tolerance. The whole network is
   checked against the CPU `Network` within the Metal spec's tolerances.
5. Throughput on the PC, 13×13 6×96:
   - ≥ 28,000 evals/s for raw two-slot passes at batch 256;
   - ≥ 25,000 evals/s for the hub under batch-8 requests;
   - ≥ 2× the Studio's local/metal rate in a real remote slice.
6. Every other build and every Mac stays byte-identical. Without
   `-DAZ_USE_CUDA=ON`, nothing CUDA is compiled, and `backend = cuda` is
   refused with a clear message.

## Non-Goals

- **Training on the GPU.** Training stays on the Studio's CPU. Mirroring
  Metal's phase B is a later design.
- **`backend = cuda` for `train`, `arena`, `analyze` and `gtp`** (the
  config's `backend` key). The PC is a remote self-play engine only. The
  config refuses `backend = cuda`, naming `selfplay_engines` as its
  place. A Linux coordinator with a CUDA arena could come later on the same
  runner.
- **cuBLAS, cuDNN, TF32, fp16, tensor cores.** cuBLAS is not
  batch-independent (spike: 75–95% of a sample's elements change with
  the batch). TF32 and fp16 change the arithmetic, and the Metal path is
  fp32 everywhere.
- **PTX or JIT.** Only machine code for named architectures is shipped
  (see "Architectures").
- **Native Windows, several GPUs per process, AMD GPUs.**
- **Returning the ownership output to search.** The hub returns logits
  and values, as `Evaluator` does. The pass computes the ownership head
  when the network has one, as the Metal pass does (a 1×1 convolution, its
  batch norm and tanh: under 1% of a pass), so every uploaded tensor is
  read and `CudaInference` can return it to tests.

## Overview

```
selfplay (on the PC) --backend cuda
  └─ playSelfPlayGames → WorkerEvaluators("cuda")
       └─ InferenceHub ── BatchRunner = CudaBatchRunner (2 slots: 2 streams,
            pinned staging, one completion thread)
                └─ CudaNetwork::encodeForward(stream, workspace, weights, B)
                     transpose_input → conv_fused (stem) →
                     [conv_fused, conv_fused+skip] × blocks →
                     policy: gemm 1×1 → bn → relu → flatten → gemm_nt → bias
                     value:  gemm 1×1 → bn → relu → gemm_nt → bias → relu →
                             gemm_nt → bias → tanh
```

Everything CUDA-specific lives in `src/cuda/` (`.cu` and `.cuh`) plus one
portable file, `src/cuda/cuda_info.cpp`. That file always compiles and
answers "not built" without CUDA, as `src/metal/metal_info.cpp` does for
Metal. The public header `include/azgo/cuda.hpp` is plain C++17, with no
CUDA types.

## Build

- **The option.** `option(AZ_USE_CUDA "GPU inference with CUDA (Linux;
  implies AZ_USE_BLAS)" OFF)`. With it on:
  - `APPLE` is a fatal error.
  - It needs CMake 3.18 or later for CUDA architectures; the project's
    minimum stays 3.16 without CUDA.
  - `enable_language(CUDA)`, `find_package(CUDAToolkit REQUIRED)`.
  - `CMAKE_CUDA_STANDARD 17`.
  - `AZ_USE_BLAS` is shadowed on, exactly as Metal does, so one PC binary
    serves both `pc/cuda` and `pc/blas`. A machine has one `exe` in the
    hosts file.
- **Sources and linking.** `src/cuda/*.cu` join `azgo`. It links
  `CUDA::cudart_static`, so the runtime version is fixed in the binary,
  and defines `AZ_USE_CUDA` publicly.
- **Compiler flags.** CUDA sources compile with:
  - `--fmad=false`: no a·b+c is contracted unless the code says `fmaf`.
  - `--prec-div=true --prec-sqrt=true --ftz=false`: IEEE division and
    square root, denormals kept. These are the defaults, stated
    explicitly so a global `--use_fast_math` cannot slip in.
  - `-Xptxas -O3`.
  - No `--use_fast_math`, ever.
- **cluster_sync.** A build directory with the word `cuda` (e.g.
  `build-cuda`) is configured with `-DAZ_USE_CUDA=ON`. cuda wins over
  blas, since a CUDA build has BLAS too. `nvcc` must be on the remote's
  `PATH`, or `CUDACXX` set, through its non-interactive shell profile; the
  sync's configure step fails with CMake's message otherwise.

### Architectures

The bits a kernel produces are fixed by the SASS (machine code) that runs
it. If the binary carried PTX, a driver update could JIT different SASS
and change results with no rebuild. So:

- `AZ_CUDA_ARCHITECTURES` (cache string, default `89-real`) becomes
  `CMAKE_CUDA_ARCHITECTURES`. Every entry must end in `-real` (SASS only,
  no PTX). Configure fails otherwise: "AZ_CUDA_ARCHITECTURES entries must
  be NN-real (machine code only; PTX would let a driver change results)".
- **At startup**, `cudaAvailable()` takes device 0's compute capability
  `M.m` and checks that a probe kernel's `cudaFuncGetAttributes`
  `binaryVersion` equals `10·M + m` exactly.
  - A newer minor GPU can run older SASS of the same major, but this
    design refuses it, so a fingerprint's GPU and its SASS always agree.
  - The refusal reads: "kernels built for sm_89; this GPU is sm_86:
    reconfigure with -DAZ_CUDA_ARCHITECTURES=86-real".
- **Other GPUs** (an RTX 30xx is sm_86, an RTX 50xx sm_120, an A100
  sm_80): add the architecture, rebuild, and run `test_cuda` on that
  machine before using it in a run. The plan measures sm_89 only.

## Kernels

All arithmetic is fp32. Each kernel mirrors one CPU function, so it can
be tested against it. Threads per block and dispatch shapes are constants
of `src/cuda/kernels.cuh`; a grid depends only on tensor shapes. There are
no atomics, every output element has exactly one writer, there are no
reductions across threads, and nothing reads unwritten memory.

| kernel | mirrors | arithmetic as written |
|---|---|---|
| `transpose_input` | `[B][P][A] → [P][B][A]` copy in `Network::forward` | copy |
| `gemm` (NN, NT) | `Backend::gemm` for convolutions (NN) and `Linear::forward` (NT) | one `fmaf` chain per output, k = 0…K−1 |
| `conv_fused` | `Conv2d::forward` (k = 3) + `BatchNorm2d::forwardInfer` + optional skip + `reluForward` | as `gemm`, then the epilogue below |
| `im2col3` | `Backend::im2col`, k = 3 | copy, 0 off the board (tests and the reference path only) |
| `bn_infer` | `BatchNorm2d::forwardInfer` | `scale = gamma / sqrtf(var + eps)` (IEEE), `shift = beta − scale·mean`, `y = scale·x + shift`: separate multiply and add, as written |
| `relu_forward` | `reluForward` | `v > 0 ? v : 0` (maps −0 and NaN to +0, as the CPU does) |
| `add_into` | the skip connection | `dst + src` |
| `policy_flatten` | `[2][B][A] → [B][2A]` | copy |
| `bias_add_rows` | the bias loop of `Linear::forward` | `y + b` |
| `tanh_inplace` | the value head's tanh | CUDA `tanhf` (≤ 2 ulp; the one kernel checked within a tolerance) |

**The gemm.** This is the spike's tiled kernel. Each block computes a
BM×BN tile of C with 256 threads, and each thread a TM×TN micro-tile:

- K advances in steps of 16 through shared memory, always from 0 to K in
  order, never split.
- Operands beyond M, N or K load as 0.
- Each output is one `fmaf(a, b, acc)` chain in k order.

So an element's bits depend only on its row of op(A) and its column of
op(B): not on M, N, the tile shape, the block, its neighbours or the run.
The spike verified this for the three tile shapes (32×256, 64×128,
128×128), at batches 1, 8, 144 and 256, at three positions each. The tile
shape is chosen from M (the weights' rows) by a fixed rule. For Linear
layers M is the batch, so their tile is fixed (32×256). Since every shape
gives the same bits, the rule only affects speed.

NT (B transposed: `Linear`'s weights stored `[out][in]`) changes only the
staging index, not the k order. With β = 0, C is never read. There is no
α or β beyond 1 and 0: inference needs only those.

**The fused convolution.** This is the same tile walk, but B's tile is
staged straight from the layer input `[C][B][n][n]`. Element `(c·9 + r,
column j)` is the input at sample `j / area`, pixel `j % area` offset by
`(r/3 − 1, r%3 − 1)`, or 0 off the board. Each thread stages one fixed
column, so the division by `area` happens once per thread. The epilogue
does, in this order:

1. `v = scale[m]·acc + shift[m]`: two roundings, `__fmul_rn` and
   `__fadd_rn`, so `bn_infer` gives identical bits;
2. optionally `v = v + skip[m][j]`;
3. `v = v > 0 ? v : 0`.

`scale` and `shift` are folded once per upload by a small `bn_fold`
kernel using `bn_infer`'s formula. Its output is bitwise equal to
`im2col3` + `gemm` + `bn_infer` (+ `add_into`) + `relu_forward`, and a test
holds it to that. The explicit kernels stay for that test and for the
probes.

The forward pass uses the fused kernel for every 3×3 convolution (stem,
both convolutions of every block) and the plain `gemm` for 1×1 head
convolutions and Linear layers. Heads are under 3% of a pass.

### Relation to the CPU (tolerance)

The CPU's `scale * x + shift` and its gemm accumulation are contracted to
FMA or not depending on compiler and target: clang on arm64 contracts
them, g++ on x86-64 without `-mfma` does not. So "CUDA equals the CPU" is
not a meaningful bitwise claim, and the tests use exact oracles written in
the test itself:

- an `std::fma` chain for the gemm;
- explicit, non-contracted multiply-then-add for batch norm, in a test
  translation unit compiled with `-ffp-contract=off`.

Against those, every kernel except tanh must be bitwise equal. Against the
real `Network` on the CPU (naive or blas), the whole pass must meet the
Metal spec's tolerance table ("Tolerance against the CPU").

## The runner

`CudaBatchRunner : BatchRunner`, `slots` = 1 or 2
(`kInferenceSlots`). It owns:

- **Weights.** One device arena laid out by `ParamLayout` (64-float
  aligned) plus the buffers arena, uploaded once with `cudaMemcpy`.
  `bn_fold` runs once after upload. The upload checks every tensor's name
  and size against the layout and throws `std::logic_error` naming an
  unknown tensor, as `GpuWeights::upload` does. The probe
  `checkForwardReadsEveryTensor` ensures a layer added to the CPU network
  is not silently skipped.
- **Per slot.**
  - A `cudaStream_t`.
  - Pinned host staging: `input`, `logits`, `values`, via
    `cudaHostAlloc`.
  - A device workspace: `stage`, `input`, the trunk's three rotating
    buffers, head buffers and outputs. There is no `col`: the fused
    convolution needs none.
  - A completion `cudaEvent_t`.
- **`begin(slot, B, done)`** enqueues on the slot's stream: H2D copy of B
  positions, `encodeForward`, D2H copies, record the event. It then
  pushes `(slot, done)` to the completion thread's queue. It never
  blocks on the GPU.
- **The completion thread** pops in order, calls
  `cudaEventSynchronize(event)` and calls `done(slot, error)` with `""` or
  `"cuda: <cudaGetErrorString>"`. It takes the first error from the
  synchronize or from `cudaGetLastError` after the enqueue.
  - It is a thread of ours, not a `cudaLaunchHostFunc` callback: host
    functions may not call CUDA, and they never run once the context has a
    sticky error.
  - A sticky error (an illegal address, for example) fails that pass and
    every later one. The hub turns that into "GPU inference failed: …",
    the slice fails, and the coordinator retries or falls back as for any
    remote failure.
- **Destructor.** It waits for the queue to drain, synchronizes both
  streams, then frees everything.
- **Memory check.** `checkCudaInferenceMemory(cfg, inferenceBatch, hubs,
  slots)` compares the workspace bytes with `cudaMemGetInfo`'s free
  memory minus a 512 MB margin. It throws "cuda: inference needs N MB
  …; lower inference_batch" before allocating. 13×13 6×96 at batch 256
  needs about 3 × 96 × 43,264 × 4 B ≈ 50 MB per slot, so this only matters
  for large boards and batches.
- **The context.** One process-wide `CudaContext` (device 0) is created
  on first use, as Metal's is, with the architecture check. It holds the
  description string. `cudaAvailable(&why)` is false with a reason when:
  - no driver or device ("no CUDA device: <error>");
  - the architecture does not match (above);
  - the probe `fill` kernel fails.

### Throughput levers kept from the spike

- Two slots: one pass computes while the next fills.
- The fused convolution: no 150 MB column matrix per layer.
- The hub's batches grow with load. A remote cuda slice uses every game at
  once by default (the `cuda_workers` hosts key; 0 = all), as metal does.
  The spike measured batch 8 at 3,500 evals/s and batch 256 at 30,800,
  so passes must be large.
- No tuning of the gemm (double buffering, vector loads) in this design.
  The spike's kernel reaches 0.65–0.69× cuBLAS, and targets are set from
  it. A faster kernel with the same k order is a later, contained change:
  the tests define its bits.

## Who uses it

- **`WorkerEvaluators(net, "cuda", …)`** builds an `InferenceHub` over
  `makeCudaRunner(net, inferenceBatch, slots)`, with the chaos seed hook
  `cuda_probe::hubChaosSeed()`. It is the same code path as metal.
- **The engines list** accepts `cuda` (`EngineSpec`, `assignGames`).
  - A local `cuda` engine needs a CUDA build and passes
    `checkBackendAvailable`.
  - A remote one is checked by the remote's `selfplay --fingerprint
    --backend cuda` at preflight, as today.
  - `engineWorkers`: `cuda_workers`, or every game of the slice. The
    remote default rule for naive and blas is unchanged.
- **The hosts file** gains `cuda_workers` (0–`kMaxArenaWorkers`, like
  `metal_workers`). `inference_batch` applies to cuda as to metal.
- **`selfplay --backend cuda`** (the remote's whole interface) plays a job
  with a cuda engine. Its `--version` options read `cuda`, from
  `buildOptions()`: `metal`, `cuda`, `blas` or `naive`, the strongest
  built.
- **The config's `backend` key** refuses `cuda`: "backend = cuda is a
  self-play engine only: use selfplay_engines (e.g. pc/cuda:60)".
  `cpuBackendName("cuda")` is `blas`, so the coordinator's CPU work is
  unaffected.
- **bench.** `bench --backend cuda` gives the raw-pass table (batch 8,
  144 and 256, synchronous and two-slot) and the hub table, as metal does.
  `--mixed` accepts a cuda hub too, to measure `pc/cuda` beside `pc/blas`
  on the same PC.

## Fingerprint

`engineFingerprint("cuda")` = `cuda; <cpu>; <os>; <cudaDescription()>`:

```
cuda; AMD Ryzen 7 7700X 8-Core Processor; Ubuntu 24.04.4 LTS (Linux 6.6.114.1-microsoft-standard-WSL2); NVIDIA GeForce RTX 4070 Ti SUPER sm_89, driver API 13.3, runtime 12.8, kernels 5d0c2e91
```

- **GPU and architecture** (`cudaGetDeviceProperties`). The architecture
  is the SASS that runs, by the check above.
- **Driver API** (`driver API 13.3`), from `cudaDriverGetVersion`: the
  CUDA API version the installed driver supports. It is not the NVIDIA
  driver's build number (e.g. 581.57), which only NVML reports; the binary
  links no NVML, so a driver update within one CUDA version does not show.
  With SASS-only binaries the driver should not change the bits. The field
  is recorded anyway, so a driver update that raises the CUDA version shows
  in the manifest's fingerprint-changed note. That is conservative and
  costs nothing.
- **Runtime** (`cudaRuntimeGetVersion`, static, i.e. the toolkit that
  built it).
- **`kernels`**: the first 8 hex of a SHA-256, computed at configure time
  over `src/cuda/*.cu` and `*.cuh`, the nvcc version, the architecture
  list and the CUDA flags. Editing a kernel re-runs configure
  (`CMAKE_CONFIGURE_DEPENDS`), as for Metal's shader hash.

The CPU and OS parts stay because the hub's companions (encoding, MCTS)
run on the CPU. The host is not named.

## Determinism contract

The Metal contract applies with "macOS version and GPU family" read as
"GPU model, SASS architecture, toolkit and kernel hash", which is the
fingerprint. On one machine, with one build:

1. **Reproducible.** Two runs of a slice give byte-identical result
   files.
2. **Independent of** `cuda_workers`, `inference_batch`, slots, the
   order requests arrive in and the hub's chaos mode. This follows from
   the batch-independence argument: every kernel computes a sample's
   outputs from its own inputs and the weights alone, in a fixed order.
3. **Byte-identical resume.** The runner uploads from the CPU `Network` at
   every slice, and no GPU state outlives it.

**Not guaranteed:**

- across GPU models, toolkits, kernel edits or `AZ_CUDA_ARCHITECTURES`;
- equality with `metal`, `blas` or `naive` (tolerance only);
- across a change of the static runtime.

Switching an engine between backends mid-run is valid, as switching
`backend` is today. The manifest records which engine played each slice
and its fingerprint.

## Error handling

| failure | effect |
|---|---|
| `backend = cuda` in a build without CUDA | `checkBackendAvailable`: "backend 'cuda' not available: rebuild with -DAZ_USE_CUDA=ON (Linux, NVIDIA GPU)" |
| no device or driver; architecture mismatch | `cudaAvailable` false with the reason; `selfplay --fingerprint --backend cuda` exits 1 with it; the coordinator's preflight reports the machine's engine as unusable |
| a pass fails (sticky error, launch failure) | that and every later pass fail; `selfplay` exits non-zero; the coordinator retries, then falls back |
| not enough GPU memory | the memory check throws before allocating |
| `backend = cuda` in a config | `ConfigError` naming `selfplay_engines` |

## Testing

`tests/test_cuda.cpp` mirrors `test_metal.cpp`. It returns 77 (CTest
Skipped) when CUDA is not built or `cudaAvailable()` is false. All
randomness comes from `azgo::Rng`.

1. **Context and probes.**
   - `cudaAvailable`, `cudaDescription` shape.
   - The architecture check: binary version equals the device's.
   - `fill` with an untouched tail.
2. **Kernels against exact oracles.**
   - `gemm` NN and NT against an `std::fma` chain, bitwise, on the
     convolution and Linear shapes of all four presets, edge sizes
     (M, N, K = 1, 15, 17, 255), and every tile shape.
   - `bn_infer`, `relu`, `add_into`, `policy_flatten`, `bias_add_rows`,
     `transpose_input` and `im2col3`, bitwise against their CPU
     counterparts or explicit non-contracted references.
   - `tanh` within 2 ulp of `std::tanh` in double, rounded.
3. **The fused convolution** equals `im2col3` + `gemm` + `bn_infer`
   (+ `add_into`) + `relu`, bitwise, for stem and block shapes, with and
   without skip, on 5×5, 9×9, 13×13 and 19×19.
4. **Batch independence.**
   - Each kernel and the whole network: positions alone, in batches of 8,
     144 and 256, at different offsets, with random companions and in
     permuted order, are bitwise identical.
   - The tile shapes give equal bits.
5. **Whole network against the CPU** `Network` with BLAS, within the
   Metal spec's 1e-5 for its networks. The preset-sized random networks
   (5×5 3×32 to 19×19 10×128) gather more float noise through their
   depth: the CPU's naive and BLAS backends already differ by up to 1.7e-5
   there. So they are measured against the network in double precision.
   The logits use maxErr; the value and ownership, which are tanh outputs
   in [−1, 1], use absolute difference. Per output, the GPU must be within
   the larger of 1e-5 and twice the CPU backends' worst gap, and never
   above 5e-5. The upload refusals:
   `checkUpload`, `checkForwardReadsEveryTensor`.
6. **Runner and hub.**
   - One and two slots, many threads, chaos seeds: every request's rows
     equal its positions evaluated alone, bitwise.
   - Failure injection (`cuda_probe::failPassesAfter(n)`) reaches every
     waiting request as "GPU inference failed".
   - The memory check with a lowered limit.
   - No thread outlives a runner (the `/proc/self/task` count).
7. **End to end.**
   - `playSelfPlayGames` with backend cuda gives byte-identical records for
     workers 1, 3 and 8, `inference_batch` 8 and 256, and chaos on.
   - A `selfplay` job with `--backend cuda` gives byte-identical result
     files twice.
   - The smoke test, in a CUDA build, runs a tiny run with
     `selfplay_engines = local/naive:1, local/cuda=2`, resumes it and
     compares. In other builds it checks the refusal.
8. **Every existing test** passes unchanged in every build, on the Mac
   and on the PC.

## Acceptance

On the PC (RTX 4070 Ti SUPER, sm_89), with the Studio's training job
undisturbed:

| measure | target | spike |
|---|---|---|
| raw two-slot pass, 13×13 6×96, batch 256 | ≥ 28,000 evals/s | 30,800 |
| raw two-slot pass, 9×9 6×96 / 19×19 10×128, batch 256 | ≥ 45,000 / ≥ 6,000 | 48,400 / 6,780 |
| hub, 13×13 6×96, N threads × batch 8 (N ≥ 256) | ≥ 25,000 evals/s | — |
| a real `pc/cuda` slice of the 13×13 run (800 simulations) | ≥ 28,000 evals/s, i.e. ≥ 2× the Studio's local/metal | — |
| determinism tests (above) | all pass | hand kernels batch-independent |
| three-machine-plus-PC run | a generation with `pc/cuda` verifies, merges and resumes byte for byte | — |

If the hub or the real slice misses its target while the raw pass meets
its own, the CPU side (MCTS on 16 threads) is the limit. In that case the
results report it, and the engine weights follow the measured rate. The
targets for the kernels stay.

## Risks

| risk | effect | mitigation |
|---|---|---|
| MCTS on 8 cores / 16 threads cannot feed 30k evals/s | the engine runs below its GPU rate | measured in the plan (hub and real-slice targets); weights follow measurements; no algorithmic change |
| WSL2 adds latency or pins fewer resources | lower rate at small batches | many games in flight; the spike's numbers already ran under WSL2 |
| a driver update changes behaviour | fingerprint note; bits should not change with SASS only | driver in the fingerprint; `test_cuda` after updates |
| a CUDA sticky error mid-slice | slice fails | the existing retry and fallback |
| toolkit upgrade changes SASS | bits change; fingerprint `kernels` and `runtime` change | recorded; same as an OS update for Metal |
| `nvcc` not on the remote's non-interactive PATH | cluster_sync's configure fails | documented in the README's hosts section; `CUDACXX` |

## Teaching material

The README's "Extending to the GPU" section gains a "CUDA" paragraph with
the spike's table, the fused convolution as an implicit GEMM, and why
cuBLAS is not used. The explainer's GPU chapter gets a short note:
"the same contract on another vendor's GPU". The CUDA source opens with
teaching comments, as `kernels.metal` does.
