# 16-bit CUDA Inference (cuda16) — Design

Author: Mark Castelluccio, with design assistance from Claude (Anthropic).

This design adds a second CUDA backend beside the 32-bit one of
`docs/superpowers/specs/2026-10-20-cuda-inference-backend-design.md`. It
keeps that design's runner, hub, engines, fingerprints and tests, and
changes only the arithmetic of the residual trunk. The numbers come from
`docs/results/cuda-bench.txt` and `docs/results/cuda16.txt`.

The algorithm does not change. A game played by `pc/cuda16` is the same
AlphaGo Zero self-play (or arena) game, seeded the same way, from the same
network. Only the last bits of the network's evaluations change, as they do
between `blas`, `metal` and `cuda`, though by more.

## Problem

The 19×19 10×128 run's self-play is bound by the PC's GPU. In generation 7,
`pc/cuda` evaluated 103 M of the run's 138 M positions (75%), at 6,620
evals/s, with the GPU at 100% and the process using about a quarter of one
CPU core. The fp32 fused kernel reaches ~15 TFLOPS, a third of the RTX 4070
Ti SUPER's 44 TFLOPS fp32 peak. Its tensor cores do ~88 TFLOPS dense with
16-bit operands and fp32 accumulation, so a tensor-core convolution could
make the engine 3–5× faster, if it keeps the determinism contract.

## Decision (Mark, 2026-10-24)

- 16-bit inference on the tensor cores for self-play and arena engines:
  16-bit operands, fp32 accumulation, hand-written kernels (no cuBLAS, cuDNN
  or TensorRT: not batch-independent, `docs/results/cuda-bench.txt`).
- A new backend `cuda16` (engine `pc/cuda16`) beside `cuda`, which stays
  byte-identical, with the same fingerprint.
- Training, validation and ratings stay 32-bit.
- The 16-bit contract: outputs reproducible given the engine fingerprint; a
  looser tolerance tier against the 32-bit CPU (defined below from
  measurements); batch independence by construction, as for `cuda`.

## Goals

1. `backend = cuda16` for `selfplay_engines` and `arena_engines`: a
   `BatchRunner` behind the existing `InferenceHub`, exactly where the `cuda`
   runner sits. The hosts key `cuda_workers` and `inference_batch` apply to
   it as to `cuda` (one GPU, one setting).
2. The trunk's 3×3 convolutions on tensor cores: fp16 operands (weights and
   activations), fp32 accumulation, batch norm, skip and ReLU in fp32 in the
   epilogue. The heads stay fp32, on the `cuda` backend's kernels.
3. The determinism contract of `cuda`, unchanged in wording, with the
   `cuda16` fingerprint in place of the `cuda` one: reproducible;
   independent of workers, `inference_batch`, slots, request order and
   chaos mode; byte-identical resume.
4. Accuracy measured on real positions from real checkpoints (13×13 6×96
   and 19×19 10×128) against the network in double precision, fp16 against
   bf16, and a tolerance tier set from those measurements.
5. Throughput: at least 2× `cuda` on 19×19 10×128 at batch 256 (raw two-slot
   pass), target 3×.
6. Strength: the same network with `cuda16` on one side and `cuda` on the
   other scores 50% within the confidence interval of a few hundred games.
7. Nothing else changes: `cuda` keeps its kernels, bits and fingerprint
   (`kernels 176be8be` in the live run); builds without CUDA are unchanged.

## Non-Goals

- **fp16 accumulation.** It doubles the GeForce tensor rate but adds a
  rounding per 16-product step to every dot product of 1,152 terms.
- **16-bit heads.** The heads are a few percent of a pass; they stay fp32.
- **Training, validation, `train`/`arena`/`analyze`/`gtp` on cuda16.** As for
  `cuda`, the config's `backend` key refuses it.
- **Replacing `cuda`.** It stays, so a run can switch back, and as the
  reference the strength check plays against.
- **GPUs before sm_80.** The kernel uses `mma.sync.m16n8k16` with fp32
  accumulation (sm_80+; the PC is sm_89). On an older GPU `cuda16` is
  unavailable with a reason; `cuda` still works.
- **CUDA graphs, split-K, cp.async pipelines deeper than needed.** Later,
  contained changes if the measurements ask for them.

## Overview

```
selfplay --backend cuda16
  └─ WorkerEvaluators("cuda16") → InferenceHub ── Cuda16BatchRunner (2 slots)
       └─ Cuda16Network::encodeForward(stream, workspace, weights, B)
            input → fp16 NHWC (channels padded to 32)
            stem:   conv16 (fp16 out)
            blocks: conv16, conv16 + skip   (fp16 out; the last block fp32 NCHW out)
            heads:  the cuda backend's fp32 kernels, unchanged
```

All new CUDA code lives in `src/cuda/fp16/`. It includes `src/cuda`'s
headers (context, buffers, streams, the fp32 head kernels) but edits no file
there. The public header is `include/azgo/cuda16.hpp`, plain C++17.

### Why a subdirectory

The `cuda` fingerprint's kernel hash covers `src/cuda/*.cu` and `*.cuh`
(non-recursive). A new file there, or an edit to the runner, would change
the hash, and the live run would record a fingerprint change for an engine
whose bits did not change. So the 16-bit code is beside it, not in it, and
the `cuda16` runner is a copy of the `cuda` runner's structure rather than
a shared template (sharing it would mean editing `cuda_runner.cu`). The copy
is ~200 lines; folding both onto one runner is a later change made at a run
boundary.

## Numbers and layouts

### The 16-bit type

fp16 (1 sign, 5 exponent, 10 mantissa bits; unit roundoff 2⁻¹¹ ≈ 4.9e-4)
or bf16 (8 exponent, 7 mantissa; 2⁻⁸ ≈ 3.9e-3). Both run at the same tensor
rate on Ada with fp32 accumulation. The kernels are templates on the type;
the bench and the accuracy tool measure both. The backend ships fp16 if its
range suffices (below), since its rounding is 8× finer.

Range: fp16's largest finite value is 65,504. The weights are well inside it.
The activations after batch norm and ReLU are what could exceed it; the
accuracy tool records their largest magnitude on real positions, and the
epilogue clamps to ±65,504 before converting, so an outlier saturates
instead of becoming infinity and NaN logits. Rounding is
round-to-nearest-even everywhere (`__float2half_rn`, `__float2bfloat16_rn`).

### Activations: NHWC, 16-bit, channels padded

Between trunk layers an activation is `[pixel][channel]` with
pixel = sample · area + y · n + x and the channel count padded to a
multiple of 32 (`Cp`), the pad channels 0. The input planes (18 for
history 8) become 32 channels of 0/1, exact in 16 bits.

### Weights: converted once per upload

A 3×3 convolution's `w [cout][cin][3][3]` (fp32) becomes `w16 [Mp][9 · Cp]`
with column `k' = r · Cp + c` (tap r = dy · 3 + dx, then channel), rows
padded to a multiple of the channel tile and every pad 0. A conversion
kernel does it on the GPU at upload, after the fp32 copy, so the fp32
weights stay for the heads and the batch norm fold; the folded scale and
shift (`bn_fold`, the `cuda` kernel) are padded to `Mp` with 0.

## The tensor-core convolution

One kernel, `conv16`, computes the implicit GEMM

    D[pixel][cout] = Σ_k col[pixel][k] · w16[cout][k],   k = r · Cp + c

with `col[pixel][r · Cp + c]` the input at the pixel offset by tap r in
channel c, 0 off the board or past the last pixel. The pixels are the GEMM's
rows (mma's A operand, row-major) and the output channels its columns (mma's
B operand, "col" layout = `w16`'s rows): both operands then have k
contiguous in memory, so every load is a 16-byte `cp.async` and every
fragment one `ldmatrix`.

- **Tiles.** A block of 256 threads (8 warps, 4 along pixels × 2 along
  channels) computes 128 pixels × BN channels, with two k steps of shared
  memory (40 KB at 128 channels, so two blocks share an SM; three or four
  steps were 20–25% slower). BN = 16 · NI with NI ∈ {8, 6,
  4, 2} (128, 96, 64, 32 channels), chosen from `Cp` alone by a fixed rule
  (the largest that divides it). A warp computes 32 × 8·NI with
  `mma.sync.aligned.m16n8k16.row.col.f32.{f16,bf16}`.
- **K.** Walked from 0 to 9 · Cp in steps of 32 (one tap, 32 channels per
  step), two mma k-steps each, always in order, never split, through a
  shared-memory ring of two steps filled by `cp.async`. An off-board
  tap or a pixel past the end loads zeros (`cp.async` with source size 0).
- **Epilogue, in fp32**, per output: `v = scale[c] · acc + shift[c]`
  (`__fmul_rn`, `__fadd_rn`), `v = v + skip` (skip read as 16-bit and
  widened), `v = v > 0 ? v : 0`, then either clamp and convert to 16-bit NHWC
  (the next layer's input) or store fp32 channel-major `[cout][pixel]` (the
  last block's output, the heads' input).

### Batch independence by construction

Each output `D[p][c]` is

    acc = 0;  for s in 0 .. 9·Cp/16 − 1:  acc = mma16(A[p][16s .. 16s+15], B[16s .. 16s+15][c], acc)

where `mma16` is the tensor core's dot product of 16 products added to the
accumulator. Its result is a function of that row's 16 values, that
column's 16 values and the accumulator alone: the hardware never mixes
outputs. Row p of A is pixel p's 3×3 neighbourhood in its own sample (zeros
off the board), and the k order is fixed by `Cp`. So a position's outputs
depend only on its own planes and the weights: not on the batch, its place
in it, its companions, the tile shape or the run. The epilogue is per
output. The heads are the `cuda` backend's kernels, batch-independent by
that design.

The one thing hardware could break is position invariance: that an mma
gives the same bits for a row whether it sits in lane group 0 or 7 of a
fragment, in warp 0 or 3. NVIDIA documents the fragment layout but not the
dot product's internal order, so this is held by test, at the same strength
as the fp32 kernel's: a position alone and inside batches of 8, 144 and 256,
at the first, middle and last slots, with random companions, permuted, and
repeated, bitwise equal; and every channel tile shape equal.

The kernel follows `kernels.cuh`'s rules otherwise: no atomics, one writer
per output, no reduction across threads, a grid from tensor shapes only,
nothing read unwritten (pad channels are written as zeros by the producer).

## Accuracy

Measured, not assumed: `tools/cuda_bench/accuracy` (a small program linked
against the library) loads checkpoints, replays real games from the run's
SGF files, encodes their positions (every ply, every eighth position kept)
and compares `cuda16` (fp16 and bf16) and `cuda` against the network in
double precision:

- logits: max and mean |Δ|, and max |Δ| / max |ref| per position;
- value: max and mean |Δ|; sign agreement (|ref| > 0.05);
- policy: argmax agreement over legal moves; KL(ref ‖ x) of the softmax;
- the largest activation magnitude (fp16 range).

### The tolerance tier

Measured on real positions (`docs/results/cuda16.txt`): 1,024 positions of
the 13×13 run's latest games with its generation 100, and 600 of the 19×19
run's with its generation 7. fp16 was 2.1e-3 and 2.6e-3 from double in the
logits (max |Δ| / max |ref| per position), 2.7e-3 and 3.0e-3 in the value,
with argmax agreement 100% and mean KL 9e-7 and 3e-8. bf16 was 7–10× further
(1.4e-2 and 1.8e-2 in the logits, argmax 99.3% and 99.7%), so the backend
ships fp16. Its largest trunk activation was 28.8, far inside fp16's range.
The 32-bit CPU is within 1e-5 of double, so the tier holds against it too.

The tier, with about a factor of two over those measurements:

| output | tier (cuda16) | cuda (fp32) |
|---|---|---|
| logits, max |Δ| / max |ref| per batch | ≤ 5e-3 | ≤ 5e-5 |
| value, max |Δ| | ≤ 1e-2 | ≤ 5e-5 |
| ownership, max |Δ| | ≤ 1e-2 | ≤ 5e-5 |

`test_cuda16` holds the presets' shapes to it with random weights whose
batch norms are calibrated on positions, as a run's generation 0 is
(`bn_calibrate`); those measured 1.3e-3 to 2.4e-3 in the logits. Without
calibration a random 19×19 10×128 network's activations grow block by
block and its value is 3e-2 from double in 16 bits; no trained network
behaves so.

## The runner and the public interface

`include/azgo/cuda16.hpp` mirrors `azgo/cuda.hpp`:

- `cuda16Available(&why)`: CUDA usable (`cudaAvailable`) and the device is
  sm_80 or newer.
- `cuda16Description()`: `"<GPU> sm_89, driver API 13.3, runtime 12.8, fp16
  kernels <hash16>"`. `hash16` is the first 8 hex of a SHA-256 at configure
  time over `src/cuda/*.cu`, `*.cuh`, `src/cuda/fp16/*.cu`, `*.cuh`, nvcc's
  version, the architectures and the CUDA flags: every source a `cuda16`
  pass runs. The `cuda` hash's inputs are unchanged.
- `makeCuda16Runner(net, maxBatch, slots)`, `checkCuda16InferenceMemory`,
  `Cuda16Inference` (synchronous, for tests, bench and the accuracy tool;
  it takes the 16-bit type, fp16 by default, so the tool can measure bf16).
- `cuda16_probe`: the test hooks of `cuda_probe` for this runner (failure
  injection, memory limit, passes completed, copy-out delay) and a
  `conv16` probe. The hub chaos seed is `cuda_probe`'s: one hook for both
  CUDA backends' hubs.

`Cuda16BatchRunner` has the `cuda` runner's structure: two slots, each a
stream, a workspace and pinned staging; one completion thread that polls
each pass's event with 50–200 µs sleeps; the same destructor order and
error reporting ("cuda16: ...", which the hub turns into "GPU inference
failed: cuda16: ...").

Workspace per slot: the fp32 stage and transposed input, the 16-bit input
and three 16-bit trunk buffers (`Cp · B · area` each), one fp32 trunk
buffer for the heads, and the `cuda` head buffers. At 19×19 10×128 and
batch 256 that is ~140 MB per slot against ~160 MB for `cuda`.

## Plumbing (shared files)

Each is a small, separate addition next to the `cuda` line:

| place | change |
|---|---|
| `src/engines.cpp` | `cuda16` in the engine list; `cuda_workers` for it; its fingerprint `cuda16; <cpu>; <os>; <cuda16Description()>` |
| `src/backend.cpp` | `makeBackend` refuses it as a GPU backend; `cpuBackendName` → `blas`; `checkBackendAvailable` |
| `src/config.cpp` | the `backend` key refuses it, naming the engine lists |
| `src/inference_hub.cpp` | `WorkerEvaluators("cuda16")` builds the hub over `makeCuda16Runner` |
| `src/arena.cpp`, `src/coordinator.cpp` | treated as a GPU backend like `cuda` (one hub per network, every game at once, default QoS, `cuda_workers`); local hubs memory-checked with `checkCuda16InferenceMemory` |
| `apps/selfplay.cpp`, `apps/bench.cpp` | accept `cuda16`; bench's raw-pass and hub tables, `--batches` for the batch list |
| `CMakeLists.txt`, `cmake/azgo_cuda.cmake` | compile `src/cuda/fp16/*`, the second hash, `test_cuda16` skip code |
| `README.md` | a "16-bit CUDA" paragraph and the engine name |

`local/cuda` and `local/cuda16` in one run are memory-checked separately
(the case does not arise on the PC, where engines are remote).

## Determinism contract

`cuda`'s, with "GPU model, SASS architecture, toolkit and fp16 kernel hash"
(the `cuda16` fingerprint). Not guaranteed: equality with `cuda`, `metal`,
`blas` or `naive` (the tolerance tier only); across GPU models, toolkits or
kernel edits. Switching an engine between `cuda` and `cuda16` mid-run is
valid, as switching backends is; the manifest records it.

## Testing

`tests/test_cuda16.cpp` (skips with 77 like `test_cuda`):

1. Availability and description shape; the architecture check.
2. The conversion to fp16 is round to nearest even (known cases). `conv16`
   against a double reference computed from the same 16-bit operands:
   |Δ| ≤ 2⁻¹⁸ · (|scale| · Σ|products| + |shift| + |skip|) (measured 2⁻²¹),
   on stem and block shapes, padded channel counts, 5×5 to 19×19, with and
   without skip; the 16-bit output is the fp32 output rounded; every channel
   tile shape gives equal bits; fp16 saturates at 65,504.
3. Batch independence: `conv16` and the whole network, B = 1, 8, 144, 256,
   first/middle/last slot, random companions, permuted batches, repeats, a
   second upload: bitwise.
4. The whole network against double precision on the presets' shapes within
   the tier; the upload refusals (unknown and unread tensors).
5. Runner and hub: slots, copy-out ordering, chaos, failure injection,
   memory check, no thread leak, `WorkerEvaluators("cuda16")`.
6. End to end: a self-play slice and an arena wave byte-identical for any
   workers, batch cap and chaos; a run with `selfplay_engines =
   local/cuda16:1` byte-identical for any hosts settings and across resume,
   its manifest naming `cuda16Description()`.
7. Every other test unchanged; in builds without CUDA, `cuda16` is refused
   with "rebuild with -DAZ_USE_CUDA=ON".

## Acceptance

| measure | target |
|---|---|
| raw two-slot pass, 19×19 10×128, batch 256 | ≥ 2× `cuda` (13,400 evals/s), aim 3× |
| raw two-slot pass, 13×13 6×96, batch 256 | ≥ 1.5× `cuda` |
| tolerance tier on real checkpoints | met with a 2× margin |
| batch-independence tests | bitwise, all |
| 16-bit vs 32-bit, gen 100 13×13 at 800 simulations | 50% inside the 95% interval |
| `cuda` fingerprint and default paths | unchanged (identity runs, fingerprint compare) |

## Risks

| risk | effect | mitigation |
|---|---|---|
| fp16 overflow on an unusual position | saturated activation, a worse evaluation | clamp; max activation measured on real positions; bf16 as fallback type |
| mma not position-invariant on some GPU/driver | batch dependence | tested bitwise on the machine (`test_cuda16`) before use, as for `cuda` |
| the CPU's tree search cannot feed a 3× faster GPU | engine below its GPU rate | measured in a real slice on the PC; `cuda_workers` up to 256 |
| 16-bit search plays weaker | worse self-play data | the strength check; switch back to `cuda` is one config edit |
| the runner copy drifts from `cuda`'s | two runners to maintain | the tests are mirrored; folding them is a later change |
