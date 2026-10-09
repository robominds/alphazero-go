# Metal Training Backend — Design

Author: Mark Castelluccio, with design assistance from Claude (Anthropic).

Amends the training and "Extending to the GPU" sections of
`docs/superpowers/specs/2026-09-29-alphazero-go-design.md`. Implements
option 2 of the plan that followed the Metal benchmark
(`docs/results/metal-bench.txt`): the whole network, forward and
backward, on the GPU for the training phase. Option 1, every CPU core for
the training step
(`docs/superpowers/specs/2026-10-08-multithreaded-training-design.md`), is
merged. Option 3, a cross-worker batching evaluator that runs self-play
and arena inference on the GPU, is not part of this design, but the
design leaves a place for it (see "Toward option 3").

The algorithm does not change: the same network, loss, batch norm and
SGD with momentum and weight decay as AlphaGo Zero. Only where the
arithmetic runs changes.

## Problem

A generation spends its time in three places. On the 9×9 6×96 run
(gen 1 of `9x9-scratch96`, M5 Max): self-play 235.6 s, training 174 s
before option 1, arena 23 s. Option 1 took a training step from 425 ms
to 93 ms on 18 threads, so the phase (400 steps plus the validation
loss) now takes about 40 s.

The spike measured what the GPU can add, and where it must be added:

- **Not behind the `Backend` seam.** A `MetalBackend` that runs only
  gemm on the GPU gains nothing. A 6×96 step took 409–426 ms against
  424.5 ms on the CPU. Gemm is only ~30% of a step. im2col, col2im,
  batch norm and the copies make up the rest and stay on the CPU, and
  copying each gemm's operands in and out (the `Workspace`'s
  `std::vector`s are not page-aligned, so Metal cannot wrap them) costs
  about what the GPU saves. Each per-op command buffer also pays
  ~130 µs of wake-up for a few µs of work.
- **Only the whole pass pays.** With every layer as a kernel on
  resident shared buffers and one command buffer per pass, a 6×96 step
  at batch 256 took 28.7 ms with the hand-written simdgroup-matrix gemm
  (36.3 ms with the scalar-tiled one). Those runs used stand-in BN, ReLU
  and loss kernels of the right sizes, so a real port will be somewhat
  slower.
- **Hand-written kernels can be deterministic; MPS cannot serve us.**
  The hand kernels gave the same bits in 20 of 20 repeats, and a
  sample's outputs were bitwise the same alone, in a batch of 8 or in a
  batch of 256. MPS was reproducible run to run, but a position's output
  depended on its batch, so it cannot back a batching evaluator under
  the project's determinism guarantee.

Against today's 93 ms CPU step the GPU step is ~3× faster, which takes a
6×96 training phase from ~37 s (400 steps) to an expected 12–16 s. That
is about 1.1× on a whole 9×9 generation, because self-play dominates.
Option 2 is still worth building first. It moves every layer, forward
and backward, onto the GPU, with the determinism machinery, in the
simplest setting: one thread, a fixed batch, no concurrency. Option 3
then reuses the forward half for self-play, where most of the time is.
Larger networks gain more on their own. The 19×19 10×128 step took
4.23 s on one CPU thread and 280 ms on the GPU.

## Goals

- `backend = metal` runs the training phase on the GPU: the
  validation-loss pass and every SGD step (forward with batch
  statistics, loss, backward, the momentum and weight-decay update, the
  BN running-statistics update).
- Hand-written Metal compute kernels. Their source is embedded in the
  library and compiled at run time, so the build needs only the
  Command Line Tools (the Mac minis have no Xcode).
- **Determinism with `backend = metal`.** A run is reproducible run to
  run, independent of `workers` and `train_threads`, and resumes byte
  for byte, on the same machine and OS (see "Determinism contract").
- A precise, tested tolerance against the CPU backends: per layer, for a
  whole training step, and over a few steps.
- Weights, momentum and activations stay on the GPU for the whole
  training phase. They cross to the CPU at two defined points.
- Portable code stays portable C++17. Objective-C++ lives only in
  `src/metal/`.
- An interface that a later batching evaluator (option 3) can call: GPU
  inference on a batch, from weights uploaded once.
- Teaching quality: kernels that mirror the CPU layers one for one, a
  readable `.metal` file, and an explainer section.

## Non-Goals

- Self-play, arena, `analyze` and `gtp` on the GPU. They stay on the CPU
  (BLAS) in this option; option 3 adds the batching evaluator.
- Bit-identity with the CPU backends. The GPU has no `double`, and the
  weight gradient sums in a different order (split-K). The tolerance is
  specified instead.
- Bit-identity across machines, GPU families or macOS versions. Shaders
  are compiled by the OS's Metal compiler at run time.
- MPS, MPSGraph, MLX, Core ML, Metal 4 tensor operations, fp16 or bf16.
- Implicit-GEMM convolution and kernel fusion in the first version. Both
  are optimizations for later, measured with `bench` (see "Kernels").
- Overlapping batch sampling on the CPU with the GPU step.
- Intel Macs and non-Apple GPUs. The gemm needs simdgroup matrices
  (Apple GPU family 7, M1 and later).
- Changing the learning algorithm in any way.

## Design

### Where the seam is

The CPU training path today is `Network::trainStep` (forward, loss,
backward into the `Network`'s gradient arrays) followed by
`SgdMomentum::step`, with a `Workspace` holding the activations and a
`ThreadPool` sharing the work. `Backend` sits under the layers and
offers gemm, im2col and col2im.

The spike rules out `Backend` as the GPU seam. Its operations are too
fine-grained: every call would cross from CPU memory to GPU memory and
back, and 70% of the work is not behind it. The seam must sit where a
whole step crosses, so it moves up one level. The new seam is a
**training engine**, the training phase's view of "the network plus its
optimizer":

```cpp
// include/azgo/training_engine.hpp (portable C++17)
class TrainingEngine {
public:
    virtual ~TrainingEngine() = default;
    virtual std::string description() const = 0;   // "cpu (blas, 18 threads)", "metal (Apple M5 Max, ...)"
    // Start of a training phase: take the network's parameters, BN running
    // statistics and the optimizer's momentum (params() order). Until
    // store(), the engine owns the training state; the Network must not
    // be read or written by anyone else.
    virtual void load(Network& net, const std::vector<std::vector<float>>& velocity) = 0;
    // Network::inferLoss of the loaded network: inference-mode batch norm,
    // no gradients, nothing changed. B <= maxBatch.
    virtual LossStats inferLoss(const float* input, const float* pi, const float* z, int B,
                                const float* ownership) = 0;
    // Network::trainStep followed by SgdMomentum::step(lr), as one step.
    virtual LossStats trainStep(const float* input, const float* pi, const float* z, int B,
                                const float* ownership, float lr) = 0;
    // End of the phase: write parameters, BN running statistics and
    // momentum back, and give the Network back to its other users.
    virtual void store(Network& net, std::vector<std::vector<float>>& velocity) = 0;
};

struct TrainingEngineOptions {
    std::string backend;          // naive | blas | metal
    int maxBatch;                 // batch_size
    float momentum, weightDecay, ownershipWeight;
    ThreadPool* pool;             // the training phase's pool (CPU engine; batch assembly for both)
};
std::unique_ptr<TrainingEngine> makeTrainingEngine(const TrainingEngineOptions&, const NetConfig&);
```

Two implementations:

- **`CpuTrainingEngine`** wraps exactly today's code. `load` keeps a
  reference to the trainer's `Network` and copies the momentum into an
  `SgdMomentum`. `trainStep` calls `Network::trainStep` and then
  `SgdMomentum::step` on the pool, `inferLoss` calls
  `Network::inferLoss` through a pool-backed `Workspace`, and `store`
  copies the momentum back. The parameters were updated in place. The
  same calls in the same order give the same bits as before, so CPU
  runs are byte-identical to the code before this change.
- **`MetalTrainingEngine`** copies everything into GPU buffers at `load`
  and back at `store`. Every step in between runs on the GPU.

The trainer calls only the engine, for every backend. One training
loop serves both engines, and a test can drive them side by side
through one interface. `heldOutLoss` (the validation pass) gets a
variant that takes a batch-loss function in place of the network and
workspace, and the old signature becomes a thin wrapper over it.

`Backend` stays as it is: the per-operation interface the CPU layers use.
Its header comment and the README's "Extending to the GPU" no longer say
a `MetalBackend` would implement it. They point to the engine and to the
spike that showed why.

### Code structure

```
include/azgo/training_engine.hpp   TrainingEngine, options, factory, cpuBackendName()
include/azgo/param_layout.hpp      ParamLayout: offsets of params()/buffers() in flat arenas (portable)
include/azgo/metal.hpp             metalBuilt(), metalAvailable(), metalDescription(); test probes (AZ_USE_METAL only)
src/training_engine.cpp            CpuTrainingEngine, makeTrainingEngine (portable)
src/param_layout.cpp               ParamLayout (portable; tested in every build)
src/metal/metal_info.cpp           metalBuilt()/metalAvailable() for builds without Metal (portable, empty otherwise)
src/metal/shader_types.h           parameter structs and fixed threadgroup sizes, shared by C++ and the shaders
src/metal/kernels.metal            every kernel, Metal Shading Language, embedded at build time
src/metal/metal_context.hpp/.mm    device, queue, library, pipelines, buffers, command batches
src/metal/gpu_layers.hpp/.mm       encode functions per layer, mirroring src/layers.cpp
src/metal/gpu_network.hpp/.mm      GpuWeights, GpuWorkspace, GpuNetwork, mirroring src/network.cpp
src/metal/metal_engine.mm          MetalTrainingEngine
src/metal/metal_probes.mm          the test probes declared in azgo/metal.hpp
```

The private headers in `src/metal/` are plain C++. Each Metal object
sits behind a pointer to an implementation struct defined in the `.mm`
file, so no Objective-C type appears in a header. Only the `.mm` files
include `<Metal/Metal.h>`. The file pairs `layers.cpp` /
`gpu_layers.mm` and `network.cpp` / `gpu_network.mm` mirror each other
function for function, so a reader can put them side by side.

**Embedding the shaders.** `src/metal/kernels.metal` is an ordinary
Metal source file, readable on its own and citable from the explainer.
CMake reads it at configure time (`file(READ)`, with the file added to
`CMAKE_CONFIGURE_DEPENDS` so an edit re-runs the step) and writes
`${CMAKE_BINARY_DIR}/generated/azgo_metal_kernels.inc`, one C++ raw
string literal (delimiter `AZGO_METAL`). `metal_context.mm` includes it
and passes it to `newLibraryWithSource:options:error:`. No `metal` or
`metallib` tool runs at build time. `shader_types.h` holds every
kernel's parameter struct and fixed threadgroup size, written in the
common subset of C++ and the Metal Shading Language. CMake prepends it
to the embedded source, and the `.mm` files include it, so the CPU and
the GPU cannot disagree on a struct's layout.

### Device, library and pipelines

`MetalContext` is created once per process, on first use (thread-safe
static), and lives until exit:

- `MTLCreateSystemDefaultDevice()`. It must exist and support
  `MTLGPUFamilyApple7` (simdgroup matrices), or the context reports why
  it is unusable.
- One `MTLCommandQueue`.
- The library, compiled from the embedded source with
  `MTLCompileOptions.mathMode = MTLMathModeSafe`, so the compiler does
  not reassociate floating-point arithmetic. Transcendentals use the
  `precise::` functions explicitly. Compilation takes a fraction of a
  second, once per process.
- One compute pipeline per kernel. Every kernel has a fixed threadgroup
  size, written in the source and the encoder, never derived from the
  device. Creating a pipeline checks
  `maxTotalThreadsPerThreadgroup` against that size and fails if the
  kernel cannot run with it. Adapting the size would silently change a
  reduction's order.
- `description()`: device name, macOS version and an FNV-1a hash of the
  shader source, for example `Apple M5 Max, macOS 27.0.1, shaders
  3f1c9a07`. Two runs whose lines match ran the same arithmetic.

Every buffer uses `MTLResourceStorageModeShared`: one allocation in
unified memory, which the CPU writes and reads through `contents` with
no copy and no blit. Buffers keep the default hazard tracking. Buffers
are zero-filled when allocated, so no result can depend on what the
memory held before.

### Memory layout

**Parameter arenas.** `ParamLayout` lists, in `Network::params()`
order, each tensor's name, size, decay flag and offset in a flat arena,
with every offset aligned to 64 floats (256 bytes). Three arenas share
that layout: `weights`, `grads` and `velocity`. A fourth, `bnStats`,
holds `Network::buffers()` (running mean and variance) the same way.
Copying a tensor in or out is one `memcpy` at its offset.
`ParamLayout` is built from a `Network`, so it always matches
`params()`. `load` checks every name and size against it and throws on
any difference. A layer added to the CPU network without a GPU port is
refused, not silently trained wrong.

`GpuWeights` (`weights` + `bnStats`) holds what inference needs. The
engine adds `grads` and `velocity`. Option 3 will need only
`GpuWeights`.

**Workspace.** `GpuWorkspace` mirrors `Workspace` field for field: the
same names, the same channel-major `[C][B][area]` layouts and the same
sizes for `maxBatch`. All fields live in one arena buffer at named,
256-byte-aligned offsets. Keeping the CPU's layout lets each kernel be
checked against its CPU counterpart on identical data, and lets a
reader of `network.cpp` find every GPU buffer by name. Like the CPU,
the workspace has one `col`/`dcol` pair shared by all 3×3 convolutions.
Backward rebuilds a convolution's columns from its saved input right
before using them. Keeping a column matrix per convolution would cost
13 × 72 MB on 6×96 and ~9 GB on 19×19.

Workspace extras with no CPU counterpart:

| buffer | size | use |
|---|---|---|
| `stage.input`, `stage.pi`, `stage.z`, `stage.ownership` | one batch, CPU layouts | the batch, `memcpy`'d in before a command buffer |
| `lossTerms` | `[maxBatch][3]` | per-sample policy, value and ownership loss, read back |
| `logZ` | `[maxBatch]` float | the softmax normalizer (`double` on the CPU) |
| `bnScratch` | `[max channels][2]` | per-channel mean/invStd or the backward sums |
| `splitK` | the largest S·M·N of the dW gemms | split-K partial sums |

**Size.** For 6×96 on 9×9 at batch 256, the workspace is ~390 MB
(the six blocks' saved activations 191 MB, `col` and `dcol` 72 MB
each) and the parameter arenas ~12 MB. For 19×19 10×128 at batch 256
it is ~3.0 GB. At creation the engine adds up its buffers and refuses
a total above `device.recommendedMaxWorkingSetSize`. The message names
the size and suggests a smaller `batch_size`.

### Kernels

All arithmetic is fp32. Each kernel mirrors one CPU function, so it can
be tested against it on the same inputs. The first version deliberately
fuses nothing. The spike's estimate puts all elementwise passes of a
6×96 step at about 4 ms of ~30, and one kernel per CPU operation keeps
both the tests and the explainer simple. Fusion (BN apply + skip + ReLU,
ReLU mask into BN backward) and implicit GEMM come later if `bench`
shows the need. They keep GPU determinism, but they change a metal
run's bits, like any kernel change.

| kernel | mirrors | grid | notes |
|---|---|---|---|
| `transpose_input` | the `[B][P][A] → [P][B][A]` copy in `Network::forward` | one thread per element | bitwise |
| `im2col3` | `Backend::im2col`, k = 3 | (column, row of col) | bitwise (copies and zeros) |
| `col2im3` | `Backend::col2im`, k = 3 | (pixel, channel) | a gather: each input point adds its ≤ 9 contributions in the CPU's (dy, dx) order, starting from 0; no atomics; bitwise |
| `gemm_simd` | `Backend::gemm` (packed, transA/transB, alpha, beta) | 64×64 tile per threadgroup of 128 threads | below |
| `splitk_reduce` | — | one thread per element of C | adds the S partial slices in order z = 0…S−1, then alpha/beta |
| `bn_train_stats` | the statistics half of `BatchNorm2d::forwardTrain` | one threadgroup of 256 per channel | mean, then variance around it (two passes, like the CPU); invStd; running mean/var updated (unbiased M/(M−1)) unless probing |
| `bn_train_apply` | the normalize half of `forwardTrain` | one thread per element | xhat and y = gamma·xhat + beta |
| `bn_infer` | `BatchNorm2d::forwardInfer` | one thread per element | scale = gamma/sqrt(runVar + eps), shift = beta − scale·runMean, as the CPU folds them |
| `bn_backward_sums` | the sums of `BatchNorm2d::backward` | one threadgroup of 256 per channel | sum(dy), sum(dy·xhat); writes ggamma, gbeta |
| `bn_backward_apply` | the dx line of `BatchNorm2d::backward` | one thread per element | dx may alias dy, as on the CPU |
| `relu_forward`, `relu_backward` | `reluForward`, `reluBackward` | one thread per element | bitwise |
| `add_into` | `addInto` (skip connection, head gradients) | one thread per element | bitwise |
| `tanh_inplace` | the value and ownership tanh | one thread per element | `precise::tanh` |
| `policy_flatten`, `policy_unflatten` | the `[2][B][A] ↔ [B][2A]` copies | one thread per element | bitwise |
| `bias_add_rows` | the bias loop of `Linear::forward` | one thread per element | |
| `bias_grad_rows` | the bias loop of `Linear::backward` | one thread per output | serial over the batch in sample order, from 0, as the CPU: bitwise |
| `policy_value_loss` | the per-sample part of `Network::loss` | one thread per sample | serial over actions in the CPU's order: max, sum of exp, logZ, dLogits = (softmax − π)/B, dValue = 2(v − z)(1 − v²)/B, the sample's policy and value loss terms |
| `ownership_loss` | the ownership part of `Network::loss` | one thread per sample | serial over the sample's points: d = o − t, dOwnership = 2w·d·(1 − o²)/(B·A), the sample's squared-error sum |
| `fill_zero` | `zeroGrad` for the ownership head | one thread per element | when the head gets no gradient |
| `sgd_update` | `SgdMomentum::step` for one tensor | one thread per element | g = grad + decay·w; v = m·v + g; w −= lr·v |

**The gemm.** `gemm_simd` is the spike's simdgroup-matrix kernel. A
64×64 tile of C per threadgroup is split into four 32×32 quarters, one
per simdgroup, held as 4×4 accumulators of 8×8. K advances in steps of
16 through threadgroup memory, in order. Out-of-range rows and columns
load as 0 and are not stored. With `beta == 0`, C is never read, so a
NaN left in it cannot leak. Each 8×8 output block runs the same
instruction sequence over the same k order wherever it lies. An
element's bits therefore depend only on its row of op(A) and its column
of op(B), never on M, N, its tile or the run. That makes the forward
pass batch-independent. The spike measured it bitwise equal to
Accelerate on every forward and dX shape. The scalar-tiled kernel
(`gemm_tiled`, 36.3 ms per step against 28.7 ms) is not shipped. The
explainer uses it to introduce tiling before the simdgroup version.

**Split-K, for the weight gradients only.** dW = dy · columnsᵀ sums over
every pixel (K = B·n², 20,736 for batch 256 on 9×9), into a small output
(96 × 864). One chain over K would leave most of the GPU idle: 4.7 ms
against 0.65 ms per gemm for the simdgroup kernel in the spike. The
rule, a function of the shape alone:

- S = 1 when K < 4096 or the output has ≥ 128 tiles of 64×64;
- otherwise S = min(64, K / 1024), kChunk = ⌈K/S⌉ rounded up to a
  multiple of 16, and S recomputed as ⌈K/kChunk⌉.

Threadgroup z sums k in [z·kChunk, (z+1)·kChunk) into slice z of
`splitK`, and `splitk_reduce` adds the slices in order. Each sum has a
fixed order for a given shape, and the batch size is fixed for a run,
so the order is fixed for a run. The 1×1 head convolutions' dW
(K = B·n²) split the same way. The heads' Linear layers (K = B) do not
split. Forward and dX products never split.

**Reductions without `double`.** Metal has no fp64. The CPU sums batch
norm's statistics and the loss's normalizer in `double`. The GPU uses
float in a fixed shape. Each of the 256 threads of a channel's
threadgroup sums a strided slice of the channel serially (t, t + 256,
t + 512, …), then a fixed binary tree in threadgroup memory combines the
256 partial sums (128, 64, …, 1). The shape depends only on M = B·n², so
it is deterministic. Its error grows with M/256 + log₂ 256 additions
(89 at batch 256 on 9×9), against the CPU's ~M additions in double.
The difference stays within the tolerance below. If a test shows batch
norm's backward sums to be the largest error, the fallback is Kahan
summation in the strided phase. Safe math mode keeps the compiler from
optimizing it away.

**The loss.** The per-sample work runs on the GPU. One thread per
sample loops over that sample's actions in the CPU's order: the max,
then the sum of exp in float where the CPU uses double, then the
gradients. The batch sums run on the CPU. After the command buffer
completes, the engine reads `lossTerms` (B × 3 floats) and adds them in
sample order, in `double`, into `LossStats`, as `Network::loss` adds its
per-sample terms. The read is a few kilobytes.

### One step, one command buffer

A training step is one command buffer with one compute encoder (serial
dispatch: each dispatch sees the previous one's writes):

1. `transpose_input` from `stage.input`.
2. Forward with batch statistics: stem, residual blocks, policy head,
   value head, ownership head when present. The order and buffers are
   those of `Network::forward`. The BN running statistics are updated in
   `bnStats` here, as the CPU updates them.
3. Loss: `policy_value_loss`, and `ownership_loss` when the batch
   carries targets.
4. `fill_zero` on the ownership head's gradients when the head gets no
   gradient (no targets or `ownership_weight = 0`). The CPU's `zeroGrad`
   leaves them zero, and the optimizer still applies momentum and decay
   to the head's weights.
5. Backward in the order of `Network::backward`, gradients written
   (beta 0) to `grads`. Every parameter receives its gradient from
   exactly one place, so writing equals the CPU's "zero, then
   accumulate".
6. `sgd_update`, one dispatch per tensor in `params()` order, with that
   tensor's decay (`weight_decay` or 0).

Then `commit`, `waitUntilCompleted`, a status check, and the loss
terms are read back. A 6×96 step is about 300 dispatches. At a few µs
each they cost ~1–1.5 ms of GPU time, plus one ~130 µs commit-and-wait
round trip per step.

A validation batch is one command buffer: `transpose_input`, the forward
pass with `bn_infer`, and the loss kernels with gradients off (the
`lossTerms` only).

The engine is synchronous. The next command buffer is encoded only
after the previous one completes, and sampling the next batch (on the
CPU pool, ~0.4 ms) does not overlap the GPU. Overlap would save ~1% and
needs double-buffered staging. It is left for option 3, which needs
that machinery anyway.

Each engine call runs inside `@autoreleasepool`. Command buffers and
encoders are autoreleased objects, and the trainer's thread has no run
loop to drain them. Without the pool, 400 steps would accumulate them
until the phase ends.

### Synchronization points

The CPU touches GPU memory only while no command buffer is in flight.
The synchronous engine makes that structural. These are the only
crossings:

| when | direction | what |
|---|---|---|
| `load` (start of the phase, before the validation loss) | CPU → GPU | every parameter (`params()`), every BN running mean and variance (`buffers()`), the momentum from `state.bin`'s `velocity`; grads zeroed |
| before each command buffer | CPU → GPU | the batch: input planes, π, z, ownership targets |
| after each command buffer | GPU → CPU | `lossTerms` (and, in tests and `bench`, chosen outputs) |
| `store` (after the last step, before the finite scan, the checkpoint and `state.bin`) | GPU → CPU | every parameter, every BN running statistic, the momentum |

Not copied back: the gradients. `Network`'s gradient arrays keep
whatever they held before `load`. No checkpoint, `state.bin` or CPU path
reads them. The activations stay on the GPU too.

After `store`, the CPU `Network` and momentum are the GPU's bits
exactly (`memcpy`). The trainer then works as today: it scans
parameters and momentum for non-finite values, writes the checkpoint,
plays the arena and self-play on the CPU, and commits `state.bin`.

**No GPU state outlives a phase.** The engine is created when the
training phase starts and destroyed when it ends, so every phase starts
from the CPU's state. Only the `MetalContext` (device, queue, compiled
pipelines) lives longer, and it holds no run state. This invariant
makes resume byte-identical. A resumed generation starts from the same
checkpoint, `state.bin` and momentum as an uninterrupted one, and the
GPU arithmetic is deterministic.

### Determinism contract

With `backend = metal`, on one machine, GPU family and macOS version:

1. **Reproducible.** The same run gives the same checkpoints,
   `state.bin`, `results.csv`, `elo.csv` and `log.csv` (timings aside)
   every time.
2. **Independent of `workers` and `train_threads`.** Self-play and the
   arena are unchanged CPU code (BLAS). The pool assembles batches with
   the existing fixed chunks, and the GPU runs one fixed sequence of
   kernels whatever the counts.
3. **Byte-identical resume**, by the invariant above.
4. **Batch-independent inference.** In inference mode, a position's
   logits, value and ownership are the same bits alone, in a batch of 8
   or in a batch of 256. The validation loss does not need this, but a
   batching evaluator (option 3) does. It holds because no forward or
   Linear product splits K, every elementwise kernel is per element, and
   inference BN uses running statistics. Training-mode BN depends on the
   batch by definition.

How the kernels guarantee 1–3: no atomics, and no result that depends
on scheduling. Each output is written by exactly one thread. Every
reduction runs inside one threadgroup in a fixed tree, or in a separate
pass in a fixed order (split-K). Threadgroup sizes and dispatch
geometry are fixed per kernel and shape. Uninitialized memory is never
read.

**Not guaranteed:**

- **Across machines, GPU families or OS updates.** The OS compiles the
  shaders at run time. Another compiler version may contract a·b + c
  into an fma differently or implement `precise::exp` differently. A run
  resumed on a different Mac or after a macOS update is valid, but not
  byte-identical to the uninterrupted run. The same already holds for
  Accelerate. The trainer prints the device line (device, OS, shader
  hash) at the start, so a reader can tell.
- **Across code versions.** Any change to a kernel changes a metal
  run's bits.
- **Equality with the CPU backends.** See the tolerance below.

**Switching backends.** `backend` stays a tunable key: a run may resume
with a different one. The run stays valid, since the algorithm is the
same, but from the switch on its bits differ from a run that never
switched. `blas` ↔ `metal` changes only the training phase. `naive` ↔
anything changes self-play too. On resume, when the backend in
`config.cfg` differs from the effective one, the trainer prints:

```
note: backend changed from blas to metal; from here on results differ in the
last bits from a run that kept blas (training arithmetic differs)
```

The note is printed for every switch, including naive ↔ blas. Nothing
is recorded in `state.bin`. `config.cfg` is rewritten with the new
value, as for any tunable key.

### Tolerance against the CPU

The GPU differs from the CPU in three places:

- The weight gradients' split-K order. The spike measured 3.6e-7 on a
  gradient of 0.6, closer to an exact result than Accelerate's.
- Float where the CPU uses double: BN statistics and backward sums,
  logsumexp, the ownership sum.
- Compiler choices of fma contraction and the GPU's transcendentals.

Everything else is bitwise equal by construction or measured so:
im2col, col2im, copies, ReLU, adds, bias gradients, and the forward
and dX gemms against Accelerate.

Tests compare with two metrics. `maxErr(x, ref)` is the largest
|x − ref| divided by the largest |ref|. `normErr(x, ref)` is
‖x − ref‖₂ / ‖ref‖₂, per tensor. Elementwise relative error is not
used: a gradient element near 0, or an activation that crosses ReLU's 0
in one backend and not the other, would make it meaningless. The
reference is the CPU backend: BLAS in a Metal build, since `AZ_USE_METAL`
implies `AZ_USE_BLAS`. A few kernel tests use a double-precision
reference computed in the test instead.

Ceilings, which the tests assert:

| quantity | metric | ceiling |
|---|---|---|
| im2col, col2im, transpose, flatten/unflatten, ReLU fwd/bwd, add, bias gradient | bitwise | 0 |
| gemm, split or not (any trans, alpha, beta) | maxErr vs a double reference | 1e-5 (spike: 1.3e-6 forward, 6.4e-7 split-K dW, 5.1e-6 for one unsplit chain of 20,736) |
| conv forward, dX (vs `Conv2d`) | maxErr | 2e-6 |
| conv dW (vs `Conv2d`) | maxErr | 1e-5 |
| BN train forward: y, xhat, invStd | maxErr | 1e-5 |
| BN running mean / var after one update | maxErr | 1e-6 |
| BN inference | maxErr | 1e-6 |
| tanh | maxErr | 1e-6 |
| BN backward: dx, ggamma, gbeta | maxErr | 1e-4 |
| Linear forward / backward | maxErr | 2e-6 / 1e-5 |
| per-sample loss terms, `LossStats` | relative | 1e-6 |
| whole forward, inference mode (logits, value, ownership) | maxErr | 1e-5 |
| whole forward, batch statistics | maxErr | 1e-4 |
| one `trainStep`: `LossStats` | relative | 1e-5 |
| one `trainStep`: each parameter's gradient | normErr / maxErr | 1e-4 / 1e-3 |
| five steps (lr 0.02, momentum 0.9, decay 1e-4): each step's `LossStats` | relative | 1e-4 |
| five steps: each weight tensor / momentum tensor | normErr | 1e-5 / 1e-3 |

These are ceilings from float32 error analysis and the spike, not
measurements of the port. The task that writes each test first measures
the actual error over 20 seeds and records it in the test's comment.
If the measured error exceeds a ceiling, that is a bug to find, not a
tolerance to raise. The exception is a documented cause, such as
float-versus-double in batch norm's backward. Over a whole generation
the two backends' trajectories drift apart, as two CPU builds with
different BLAS libraries would. No test compares them step for step
beyond five steps. The benchmark task compares their loss curves
statistically (see "Testing").

### Configuration, CLI and build

**Build.** `option(AZ_USE_METAL "Train on the GPU with Metal (macOS; implies AZ_USE_BLAS)" OFF)`:

- On a non-Apple system, `FATAL_ERROR`.
- Turns `AZ_USE_BLAS` on (with a status message), because `metal` runs
  self-play and the arena with BLAS (Accelerate ships with macOS).
- `enable_language(OBJCXX)` (CMake ≥ 3.16), adds `src/metal/*.mm` to
  the library with `-fobjc-arc` and C++17, links `-framework Metal
  -framework Foundation`, and defines `AZ_USE_METAL` publicly, as
  `AZ_USE_BLAS` is. `src/metal/*.cpp` is portable C++ and is compiled in
  every build (`metal_info.cpp` answers "not built" without the option).
- Generates the embedded shader include (above).

Only the macOS SDK headers are needed. The Command Line Tools' SDK has
`Metal.framework`. The `metal` shader compiler, which only Xcode
ships, is never invoked.

**Config.** `backend = naive | blas | metal`. The value parses in every
build, so a config file is portable. Availability is checked when a run
starts or resumes, before anything is written, as for `blas`. Re-rating
plays only the arena, so `--rerate` of a metal run needs only BLAS. The
messages:

- not built: `backend 'metal' not available: rebuild with -DAZ_USE_METAL=ON (macOS only)`
- built, but no usable GPU: `backend 'metal': no usable Metal GPU (<reason>)`,
  with reason `no Metal device`, or `<name> lacks Apple GPU family 7
  (simdgroup matrices; M1 or later)`, or the shader compiler's message.

The check creates the `MetalContext`, which compiles the shaders, so a
compile failure also stops the run before it writes anything.

`cpuBackendName(name)` maps `metal` to `blas` and returns the others
unchanged. Self-play, the arena phase, `--rerate` and the reference
networks' arena games use it. The training phase calls
`makeTrainingEngine`. The trainer prints the engine's description once,
when a run starts or resumes with `metal`:

```
training engine: metal (Apple M5 Max, macOS 27.0.1, shaders 3f1c9a07)
```

`train_threads` keeps its meaning: the pool assembles batches and
encodes validation positions. With `metal` the network passes do not
use it. `batch_size` sizes the GPU workspace.

**Apps.**

- `bench --backend metal` (and `--backend all` in a Metal build) adds
  the GPU's numbers (see "Performance").
- `arena`, `analyze` and `gtp` refuse `--backend metal` with
  `backend 'metal' runs only the training phase for now; use --backend blas`.
  `makeBackend("metal")` throws that message, so every caller of the
  per-op factory refuses in one place, and the trainer asks for
  `makeBackend(cpuBackendName(...))`. Accepting `metal` as a silent
  alias of `blas` would let option 3 change what those commands compute
  without anyone asking for it.
- `grow` uses the naive backend internally and is unchanged.

**Presets.** The presets keep `backend = blas`. `metal` is opt-in per run
(`--set backend=metal`), at least until it has trained a full reference
run.

### Error handling

| situation | behavior |
|---|---|
| `metal` requested, not built | error at start or resume, nothing written (above) |
| no device, wrong GPU family, shader compile or pipeline failure | error at start or resume with the reason, nothing written |
| buffers exceed `recommendedMaxWorkingSetSize`, or an allocation returns nil | `std::runtime_error` naming the bytes and suggesting a smaller `batch_size`, when the engine is created, before the first step |
| a command buffer ends with an error (status `MTLCommandBufferStatusError`: a GPU fault, a timeout) | `std::runtime_error("GPU command buffer failed at generation g step s: <error>; last good state kept")`. `state.bin` is the commit point, so resuming replays generation g |
| a non-finite loss | as on the CPU: the trainer's per-step check throws, last good state kept |
| non-finite weights or momentum after the phase | as on the CPU: the scan after `store` throws before anything is saved |
| `load` finds a tensor the GPU network does not know | `std::logic_error` naming the tensor (a CPU layer without a GPU port) |

No Objective-C exception or callback crosses into C++. Each `.mm` entry
point checks `NSError`s and statuses and throws C++ exceptions itself.
The engine's destructor waits for any command buffer still in flight,
so an exception never leaves the GPU writing into freed buffers.

### Toward option 3

What a batching evaluator will need, and where it already exists:

- **GPU inference on a batch:** `GpuNetwork::infer(GpuWeights&,
  GpuWorkspace&, input, B, logits, values, ownership)`. It is synchronous
  and built from `encodeForward(…, Mode::Infer)`. Option 2's validation
  pass is its first caller.
- **Weights without an optimizer:** `GpuWeights` uploads from a `const
  Network&`, once per generation, for the self-play network and the
  arena's opponents.
- **Several batches in flight:** each `GpuWorkspace` owns its staging
  buffers, so two workspaces can pipeline. `encodeForward` only encodes
  and leaves commit and wait to the caller.
- **The guarantee a shared batch needs:** batch-independent inference,
  tested in this option.

Option 3 adds the queue of leaves from all workers, the dispatcher
thread and asynchronous self-play. It does not change a kernel.

## Testing

Every GPU test is in one executable, `tests/test_metal.cpp`. It returns
77 when the build has no Metal or the machine has no usable GPU, and
CMake sets `SKIP_RETURN_CODE 77` on it, so CTest reports it as
"Skipped", not passed or failed. The project has no CI, so these tests
run locally on an Apple-silicon Mac. GPU tests may run while a training
job uses the CPU. Timing measurements may not (see "Performance").

- **Context.** Device and library come up, every pipeline is created
  with its fixed threadgroup size, and a trivial kernel round-trips a
  buffer. `metalDescription()` is stable within a process.
- **Kernels against the CPU (per layer, forward and backward).** Each
  kernel group runs through a test probe (`azgo/metal.hpp`: host arrays
  in, one command buffer, host arrays out) and is compared with the CPU
  function it mirrors on the same random data, at the ceilings above.
  Shapes are 5×5, 9×9 and 19×19; 1, 2, 7 and 256 samples; 1, 2, 18, 32
  and 96 channels; gemms with every transA/transB combination, M, N
  and K that are not multiples of the tile, alpha ≠ 1, beta 0 and 1, a
  NaN-filled C with beta 0, and every split-K branch.
- **Bitwise kernels** (im2col, col2im, copies, ReLU, add, bias
  gradient) are compared with `memcmp`.
- **Whole network against the CPU.** `GpuNetwork` and the CPU
  `Network` start from the same seed: inference forward; a forward with
  batch statistics, including the updated running statistics; one
  `trainStep` (losses and every parameter's gradient through a probe
  that skips the update); five steps with SGD (losses each step,
  weights, momentum and BN statistics at the end). Configurations: 5×5
  1×8, 9×9 2×16, 9×9 6×32, a network with 0 blocks, the ownership head
  with w = 1.5, w = 0 (head gradients zero, head weights still decayed)
  and no targets.
- **Run-to-run reproducibility.** One step repeated 20 times from the
  same state, and three steps in each of two separately created
  engines, give bitwise identical losses, weights, momentum and BN
  statistics.
- **Batch independence.** Inference outputs (logits, value, ownership)
  of positions 0, 1, 7, 100 and 255 are bitwise equal alone, in a batch
  of 8 and in a batch of 256.
- **Split-K order.** The same dW twice gives identical bits, and the
  rule gives the documented S for the 6×96 and 19×19 shapes.
- **Engine.** `MetalTrainingEngine` against `CpuTrainingEngine`:
  `load`/`store` round-trip a network and momentum bit for bit with no
  step in between; `inferLoss` and `trainStep` are within tolerance.
  The memory check refuses an oversized batch with the documented
  message.
- **Trainer with `backend = metal`** (tiny 5×5 config):
  - a resumed run is byte-identical to an uninterrupted one
    (checkpoints, `state.bin`, results, ratings, validation columns);
  - `workers` 1 against 3 and `train_threads` 1 against 4 give
    identical files;
  - a `blas` run resumed with `metal` completes and prints the note;
  - a GPU failure injected through a test hook leaves `state.bin` at
    the previous generation.

Outside `test_metal`:

- `test_training_engine` (every build): `CpuTrainingEngine` gives
  exactly `Network::trainStep` + `SgdMomentum::step` (bitwise, with and
  without the ownership head, naive and BLAS). `heldOutLoss` through
  its loss-function variant equals the old signature bitwise.
  `makeTrainingEngine("metal")` in a build without Metal throws the
  documented message.
- `test_config`: `backend = metal` parses, prints and round-trips;
  `cpuBackendName`.
- `test_trainer`: the existing byte-identity tests now run through
  `CpuTrainingEngine` and must pass unchanged. In a build without Metal,
  starting a run with `backend = metal` fails with the message and
  writes nothing.
- `integration_smoke.sh` takes a third argument, `metal=ON|OFF`. With ON
  it trains the tiny run for two generations with `--set backend=metal`,
  resumes it for a third, and checks that `arena --backend metal` is
  refused. With OFF it checks that `train --set backend=metal` fails
  with the message.
- **Old against new binary**, as in option 1: short 5×5 (naive and
  BLAS) and 9×9 6×64 BLAS runs of main's `train` and this branch's
  write byte-identical checkpoints, `state.bin` and `results.csv`. This
  shows the engine refactor changed no CPU bits.

## Performance

**Measurement.** `bench --backend metal --sizes 9 --configs DIR
--train-batch 256` prints, per network shape:

- the training step: wall time per `MetalTrainingEngine::trainStep`
  (copying the batch in, one command buffer, commit, wait, reading the
  loss), median over ≥ 2 s after a warm-up, plus the GPU's own time
  (`GPUEndTime − GPUStartTime`);
- inference throughput at the training batch (the validation pass's use)
  and at 8 and 144 (for option 3's planning);
- `load` and `store` times.

In the main table the metal row shows the step at batch 128, like the
other rows, and `-` for search-batch-8 evals/s, which search does not
use. In the `--train-threads` table the metal row appears once, with
threads `gpu`. CPU and GPU timings need an otherwise idle machine:
pause the running jobs (SIGSTOP/SIGCONT) as the spike did.

**Expectations (M5 Max, 9×9, batch 256).**

| network | CPU now (18 threads) | spike, stand-in kernels | expected | acceptance |
|---|---|---|---|---|
| 6×64 step | 52 ms | 16.8 ms | ~20 ms | ≤ 30 ms |
| 6×96 step | 93 ms | 28.7 ms | 30–40 ms | ≤ 45 ms |
| 6×96 training phase (400 steps + validation) | ~40 s | — | 12–16 s | ≤ 20 s |
| 19×19 10×128 step | (4.23 s on 1 thread) | 280 ms | 300–400 ms | ≤ 500 ms |

The training phase is also measured in a real run:
`train --generations 1` on the 6×96 9×9 preset, once with `blas` and
once with `metal`, comparing `train_s` in `log.csv`. That run's loss
columns must agree within noise: `policy_loss` and `value_loss` within
2% relative, and the validation losses likewise. That is the
whole-phase sanity check that no test can make bitwise.

If the step misses its acceptance, the levers are, in order: fusing BN
apply with skip and ReLU, fusing the ReLU mask into BN backward,
tuning the split-K rule, an implicit-GEMM convolution (no `col` round
trip, 72 MB written and read per 3×3 conv on 6×96), and overlapping
batch sampling with the step. Each one is measured with `bench` before
it is kept.

Results go to `docs/results/bench.txt` (the bench output) and
`docs/results/README.md` (the phase times), with the machine and
conditions stated.

## Teaching material

- **README.** Build gains the `AZ_USE_METAL` command and states what it
  needs (an Apple-silicon Mac, the Command Line Tools). The Usage
  section's **Threads** paragraph gains a "Backends" paragraph: what
  `metal` moves to the GPU, what stays on the CPU, the determinism
  contract and the switching note. "Extending to the GPU" is rewritten:
  option 1 done, option 2 done with its measured numbers, option 3
  next. The test count is updated: `test_metal` and
  `test_training_engine` bring the CTest total to 30, with `test_metal`
  reported as Skipped in builds without Metal.
- **Explainer** (`docs/algorithm-explained.md` and `.html`). Chapter 03's
  paragraph "This is where a GPU would plug in" predates the spike and is
  corrected. It describes the measured dead end of the per-op seam and
  points to the new section. A new section in chapter 08, after the
  passage on the training step's threads, titled "Training on the GPU":
  - why the seam moved: copies, wake-ups and the 70% that is not gemm,
    with the spike's numbers;
  - residency: weights, momentum and activations in shared buffers,
    two crossings per phase, one command buffer per step;
  - a gemm tile: the scalar version first, then the simdgroup 8×8 form,
    quoting `kernels.metal`;
  - determinism without atomics: the gather col2im, a fixed split-K
    order, fixed reduction trees, fixed threadgroup sizes, and why MPS
    was rejected;
  - no `double` on the GPU, and what "within tolerance of the CPU"
    means, with the measured errors;
  - the measured results.
  The HTML version gets one diagram: the training phase's timeline
  (load → validation batches → 400 step command buffers → store), with
  the CPU/GPU crossings marked.
- **This spec** and its plan are listed under "Learn how it works".
- `tools/check_citations.py` must keep passing. It is extended to `.mm`
  and `.metal` files if it does not already handle them.

## Risks

| risk | consequence | mitigation |
|---|---|---|
| Two implementations of every layer drift apart | a GPU run trains a slightly different network than the CPU code describes | the whole-network GPU-vs-CPU tests cover every configuration; `load` refuses unknown tensors; the plan's global constraints require any network change to update both |
| A wrong factor in a backward kernel still trains, just badly | silently worse runs | per-layer tests against the CPU backward, whole-step gradient tests, and the real-run loss comparison |
| Shader compilation varies with the OS | resume across an OS update is not byte-identical | documented; the device line in the log shows it |
| The Command Line Tools' SDK lacks something Metal needs at build time | no build on the Mac minis | Task 1's acceptance is a clean build and `test_metal` pass on a CLT-only machine |
| Float-only reductions exceed the tolerance | tests fail on BN backward | Kahan summation in the strided phase; ceilings set from analysis, measured before the tests are fixed |
| The step misses its target | less gain than expected | the levers above, each measured with `bench` |
| GPU memory on a 16 GB Mac mini | 19×19 at batch 256 needs ~3 GB plus the CPU's share | the check at engine creation with a clear message |
| Autoreleased Metal objects accumulate in the training loop | memory grows during a phase | `@autoreleasepool` around every engine call; the bench watches resident size over 1,000 steps |
| A GPU fault mid-phase | lost generation | `state.bin` is the commit point; resume replays the generation |
