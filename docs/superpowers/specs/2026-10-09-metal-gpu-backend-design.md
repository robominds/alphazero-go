# Metal GPU Backend — Design

Author: Mark Castelluccio, with design assistance from Claude (Anthropic).

Amends the self-play, arena, training and "Extending to the GPU" sections
of `docs/superpowers/specs/2026-09-29-alphazero-go-design.md`. Follows the
Metal benchmark (`docs/results/metal-bench.txt`) and option 1, every CPU
core for the training step
(`docs/superpowers/specs/2026-10-08-multithreaded-training-design.md`,
merged). The spike's remaining two options become two phases, built in
this order:

- **Phase A: inference.** The network's forward pass on the GPU, and a
  batching evaluator that gathers the leaves of every worker's search
  into one GPU call. Self-play, the arena and the validation loss use
  it. This was option 3. It comes first because it is where a
  generation's time goes.
- **Phase B: training.** Backward, losses and the optimizer on the GPU,
  behind a training-engine seam, reusing Phase A's context, kernels and
  weights. This was option 2.

The algorithm does not change in either phase: the same network,
search, loss, batch norm and SGD with momentum and weight decay as
AlphaGo Zero. Only where the arithmetic runs changes.

## Problem

A 9×9 generation of the 6×96 network on the 18-core M5 Max now takes
roughly 285–325 s:

| phase | time | where it goes |
|---|---|---|
| self-play | ~200–240 s | ~90% network evaluation, batches of 8 per worker |
| arena | ~45 s | the same, two networks |
| training (400 steps + validation loss) | ~40 s | 93 ms per step on 18 threads |

The spike measured three things that decide the design:

- **CPU inference does not scale with cores.** One worker alone
  evaluates 2,491 positions/s on 6×96 at batch 8. Eighteen together
  reach only 14,554/s, 809 each, because the cores share their matrix
  units.
- **The GPU wins at large batches.** A whole forward pass in one
  command buffer, with two command buffers in flight, reaches 21.5k
  evals/s (scalar-tiled gemm) or 30.2k (simdgroup gemm) at batch 144.
  That is 1.5× or 2.1× the 18 CPU workers. The break-even batch against
  all 18 cores is about 47–54 positions. At batch 8, a single worker's
  call, the GPU is slower than all cores together.
- **Only whole passes pay, and only hand-written kernels are safe.** A
  gemm-only backend behind the existing `Backend` seam gains nothing:
  the copies and the ~130 µs per-call wake-up eat the gain, and 70% of
  the work is not gemm. MPS was faster, but a position's output depended
  on the other positions in its batch. That breaks the project's
  determinism guarantee as soon as batches mix leaves from different
  games. The hand kernels were bitwise identical alone, in a batch of 8
  and in a batch of 256.

So the big win is self-play and arena inference, and it needs both
halves: kernels whose per-position output does not depend on the batch,
and an evaluator that builds large batches out of many workers' small
ones. Training on the GPU (Phase B) saves a further ~25 s per generation.

## Goals

Both phases:

- **Runtime-selectable.** `backend = metal` in the config, and
  `--backend metal` on the command-line tools. Built only on macOS
  behind `AZ_USE_METAL` (OFF by default). Asking for `metal` without it
  is a clear error. The presets are unchanged (`backend = blas`).
- **Hand-written Metal compute kernels.** No MPS, MPSGraph or MLX. The
  source is embedded in the library and compiled at run time, so the
  build needs only the Command Line Tools (the Mac minis have no
  Xcode). Objective-C++ lives only in `src/metal/`, and everything else
  stays portable C++17.
- **Determinism with `backend = metal`**, on one machine and macOS
  version: runs are reproducible, independent of `workers`,
  `train_threads` and every batching setting, and they resume byte for
  byte (see "Determinism contract").
- Tested tolerance against the CPU. Tests skip cleanly where Metal is
  not built or not usable.
- Teaching quality: kernels that mirror the CPU layers one for one, a
  readable `.metal` file, and an explainer section.

Phase A: GPU inference for self-play (including gating's best network),
the arena (both networks, and reference networks), the validation-loss
pass, and optionally `analyze` and `gtp`.

Phase B: the training step (forward with batch statistics, loss,
backward, SGD with momentum and weight decay, BN running statistics) on
the GPU, opt-in with `gpu_training = true`.

## Non-Goals

- Bit-identity with the CPU backends. The GPU has no `double`, and
  Phase B's weight gradients sum in a different order. The tolerance is
  specified instead.
- Bit-identity across machines, GPU families or macOS versions. The OS
  compiles the shaders at run time.
- Changing the search. In particular, no larger `mcts_batch` or deeper
  virtual loss to make bigger batches. That would change the algorithm
  and every result.
- MPS, MPSGraph, MLX, Core ML, Metal 4 tensor operations, fp16/bf16,
  implicit-GEMM convolution and kernel fusion. The last two are later
  optimizations, measured with `bench` first.
- Intel Macs and non-Apple GPUs. The gemm needs simdgroup matrices,
  Apple GPU family 7 (M1) or later.
- A CPU batching evaluator for users. BLAS is not batch-independent.
  The batching machinery runs on the CPU only in tests (see "Testing").

---

## Phase A — GPU inference and the batching evaluator

### Overview

```
worker 1 ─ Mcts ─ BatchingEvaluator ─┐
worker 2 ─ Mcts ─ BatchingEvaluator ─┤   requests      ┌──────────────────┐  slot 0 ┌───────────┐
   ...                                ├──────────────► │  InferenceHub     │ ──────► │ GPU: one  │
worker W ─ Mcts ─ BatchingEvaluator ─┘   (8 leaves     │  queue +          │  slot 1 │ forward   │
                                          each)        │  dispatcher thread│ ──────► │ pass per  │
                                     ◄────────────────  └──────────────────┘ ◄────── │ batch     │
                                      logits, values     copies results back          └───────────┘
```

One `InferenceHub` exists per network for the length of a phase:
self-play's network, or each player network of an arena match. Each
worker's search keeps calling `Evaluator::evaluate` exactly as today.
Its evaluator is a `BatchingEvaluator`, a thin handle that enqueues the
request in the hub and blocks until the results are back. A dispatcher
thread packs pending requests into a batch, copies their planes into a
GPU staging buffer and commits one forward pass. With two such slots,
one batch runs on the GPU while the next fills. When a pass completes,
the dispatcher copies each request's rows back and wakes its worker.

MCTS, self-play and the arena's game loop do not change. Only the
evaluator a worker is given changes.

### Code structure

```
include/azgo/inference_hub.hpp     BatchRunner, InferenceHub, BatchingEvaluator (portable C++)
src/inference_hub.cpp              queue, dispatcher, flush policy (portable; tested in every build)
include/azgo/param_layout.hpp      ParamLayout: params()/buffers() in flat arenas (portable)
src/param_layout.cpp
include/azgo/metal.hpp             metalBuilt(), metalAvailable(), metalDescription(),
                                   makeMetalRunner(...); test probes (AZ_USE_METAL only)
src/metal/metal_info.cpp           the three functions for builds without Metal (portable)
src/metal/shader_types.h           parameter structs and fixed threadgroup sizes, shared by C++ and the shaders
src/metal/kernels.metal            every kernel, Metal Shading Language, embedded at build time
src/metal/metal_context.hpp/.mm    device, queue, library, pipelines, buffers, command batches
src/metal/gpu_layers.hpp/.mm       encode functions per layer, mirroring src/layers.cpp
src/metal/gpu_network.hpp/.mm      GpuWeights, GpuWorkspace, GpuNetwork, mirroring src/network.cpp
src/metal/metal_runner.mm          MetalBatchRunner: GpuNetwork behind the BatchRunner interface
src/metal/metal_probes.mm          the test probes declared in azgo/metal.hpp
```

The private headers in `src/metal/` are plain C++. Each Metal object
sits behind a pointer to an implementation struct defined in a `.mm`
file, so no Objective-C type appears in any header, and only `.mm` files
include `<Metal/Metal.h>`. The pairs `layers.cpp` / `gpu_layers.mm` and
`network.cpp` / `gpu_network.mm` mirror each other function for
function, so a reader can put them side by side.

The batching machinery (`inference_hub.cpp`) knows nothing about Metal.
It drives an abstract `BatchRunner`. Metal provides one runner, and the
tests provide a CPU runner that is batch-independent by construction
(below). The queue, the flush policy and the determinism argument can
therefore be tested in every build.

### Metal context, build and embedded shaders

`MetalContext` is created once per process, on first use (thread-safe
static), and lives until exit. It holds no run state:

- `MTLCreateSystemDefaultDevice()`. The device must exist and support
  `MTLGPUFamilyApple7` (simdgroup matrices); otherwise the context
  reports why it is unusable.
- One `MTLCommandQueue`, shared by every hub. Metal queues are
  thread-safe.
- The library, compiled from the embedded source with
  `MTLCompileOptions.mathMode = MTLMathModeSafe`, so the compiler does
  not reassociate floating-point arithmetic. Transcendentals use
  `precise::` explicitly. This happens once per process and takes a
  fraction of a second.
- One compute pipeline per kernel, created from a table of kernel names
  and fixed threadgroup sizes. Creation fails, naming the kernel, if
  `maxTotalThreadsPerThreadgroup` is below that size. Adapting the size
  to the device would silently change a reduction's order.
- `description()`: device name, macOS version and an FNV-1a hash of the
  shader source, for example `Apple M5 Max, macOS 27.0.1, shaders
  3f1c9a07`. Two runs whose lines match ran the same arithmetic.

Every buffer uses `MTLResourceStorageModeShared`: unified memory that
the CPU reads and writes through `contents`, with no copy or blit.
Buffers are zero-filled at allocation and keep the default hazard
tracking.

**Embedding.** `src/metal/kernels.metal` is an ordinary Metal source
file, readable on its own and citable from the explainer.
`src/metal/shader_types.h` holds every kernel's parameter struct and
fixed threadgroup size, in the common subset of C++ and the Metal
Shading Language. At configure time CMake reads both files (`file(READ)`,
added to `CMAKE_CONFIGURE_DEPENDS`) and writes
`${CMAKE_BINARY_DIR}/generated/azgo_metal_kernels.inc`: one C++ raw
string literal with `shader_types.h` first. The `.mm` files include the
same header, so the CPU and the GPU cannot disagree on a struct's
layout. No `metal` or `metallib` tool runs at build time.

**Build.** `option(AZ_USE_METAL "GPU inference and training with Metal (macOS; implies AZ_USE_BLAS)" OFF)`:

- `FATAL_ERROR` on a non-Apple system.
- Turns `AZ_USE_BLAS` on, with a status message: the network work that
  stays on the CPU with `metal` (training, in Phase A) uses BLAS, which
  on macOS is Accelerate. (The bots need no network, and `grow` keeps the
  naive backend.)
- `enable_language(OBJCXX)` (CMake ≥ 3.16). Adds `src/metal/*.mm` with
  `-fobjc-arc` and C++17, links `-framework Metal -framework
  Foundation`, and defines `AZ_USE_METAL` publicly, as `AZ_USE_BLAS` is
  defined. `src/metal/*.cpp` is portable and is compiled in every build.
- Generates the embedded shader include.

Only the macOS SDK's headers are needed. The Command Line Tools' SDK
includes `Metal.framework`. The `metal` shader compiler, which only
Xcode ships, is never invoked.

### Forward kernels

All arithmetic is fp32. Each kernel mirrors one CPU function, so it can
be tested against it on the same inputs. Phase A needs only the forward
pass in inference mode. The policy head ends at its logits (MCTS takes
logits; no softmax), the value head at its tanh, and the ownership head,
when the network has one, at its tanh. `Evaluator` returns only logits
and values. Ownership is computed only for the validation loss and
`analyze`.

| kernel | mirrors | grid | per-position? |
|---|---|---|---|
| `transpose_input` | the `[B][P][A] → [P][B][A]` copy in `Network::forward` | one thread per element | yes: a copy |
| `im2col3` | `Backend::im2col`, k = 3 | (column, row of col) | yes: reads only its own sample's board, zero outside it |
| `gemm_simd` | `Backend::gemm` | 64×64 tile per threadgroup of 128 threads | yes, see below |
| `bn_infer` | `BatchNorm2d::forwardInfer` | one thread per element | yes: scale and shift from running statistics |
| `relu_forward` | `reluForward` | one thread per element | yes |
| `add_into` | the skip connection | one thread per element | yes |
| `policy_flatten` | the `[2][B][A] → [B][2A]` copy | one thread per element | yes |
| `bias_add_rows` | the bias loop of `Linear::forward` | one thread per element | yes |
| `tanh_inplace` | the value and ownership tanh | one thread per element | yes: `precise::tanh` |
| `fill` | none (the first test kernel) | one thread per element | — |

**The gemm.** `gemm_simd` is the spike's simdgroup-matrix kernel. A
64×64 tile of C per threadgroup is split into four 32×32 quarters, one
per simdgroup, held as 4×4 accumulators of 8×8. K advances in steps of
16 through threadgroup memory, always from 0 to K in order and never
split. Rows and columns beyond M or N load as 0 and are not stored.
With beta = 0, C is never read. Every 8×8 output block runs the same
instruction sequence over the same k order wherever it lies, so an
element's bits depend only on its row of op(A) and its column of op(B):
not on M, N, its tile, its neighbours or the run. The spike measured
the forward and dX products bitwise equal to Accelerate on every shape it
tried; task A2 found the same for every stem and block convolution of the
presets, but not for the 1×1 head convolutions with one or two output rows
(0.0054% of all elements differ, within tolerance), where Accelerate sums
in another order.
The scalar-tiled kernel (`gemm_tiled`) is not shipped. The explainer
uses it, from `tools/metal_bench`, to introduce tiling.

In a convolution, op(B) is the column matrix: column j is one pixel of
one sample, so a sample's outputs read only that sample's columns. In a
Linear layer, op(A) is the activations with one row per sample, so a
sample's outputs read only its own row. Either way, the weights are the
other operand, the same for every sample.

### The GPU network

**Weights.** `ParamLayout` lists, in `Network::params()` order, each
tensor's name, size, decay flag and offset in a flat arena, with every
offset aligned to 64 floats (256 bytes). `Network::buffers()` (BN
running mean and variance) gets a second arena. `GpuWeights` holds the
two arenas. `upload(const Network&)` is one `memcpy` per tensor. It
checks every name and size against the layout and throws
`std::logic_error` naming any tensor it does not know, so a layer added
to the CPU network without a GPU port is refused, not silently
evaluated wrong.

**When the weights move.** A hub is created at the start of a phase
(self-play, each arena match, the validation pass) and uploads the
network it serves. It is destroyed at the end. The weights never change
during inference phases, so nothing is copied back. No GPU state
outlives a phase.

**Workspace.** `GpuWorkspace(cfg, maxBatch, Purpose::Inference)` holds
one batch's activations, under the CPU `Workspace`'s names and layouts
(channel-major `[C][B][area]`) so each kernel can be checked against its
CPU counterpart and found by name. Inference keeps no activation for a
backward pass, so it needs only `stage.input` (the batch as the
evaluator receives it), the transposed input, `col`, `convOut`, three
trunk buffers (block input, middle activation, block output, rotated),
the heads' buffers, and the outputs `logits`, `values` and
`ownership`. That is about 100 MB for 6×96 on 9×9 at batch 256, against
~390 MB for the training workspace of Phase B (`Purpose::Training`,
which adds every saved activation). A hub owns one workspace per
pipeline slot.

**Forward.** `GpuNetwork::encodeForward(CommandBatch&, GpuWorkspace&,
const GpuWeights&, B, Mode::Infer)` encodes, into one compute encoder
with serial dispatch, the layers of `Network::forward` in its order. A
6×96 pass is about 50 dispatches. `infer(…)` is the synchronous
convenience: copy in, encode, commit, wait, copy out. The validation
pass and the tests use it.

### The batching evaluator

```cpp
// include/azgo/inference_hub.hpp (portable C++17)

// Runs one forward pass on a batch, asynchronously, in one of a few slots.
class BatchRunner {
public:
    virtual ~BatchRunner() = default;
    virtual int slots() const = 0;               // batches that may be in flight at once (2)
    virtual int maxBatch() const = 0;
    virtual float* input(int slot) = 0;          // [maxBatch][planes][area] staging for the slot
    // Starts the pass on the first B positions of input(slot); calls done(slot, error)
    // from any thread when the outputs can be read (error empty on success).
    virtual void begin(int slot, int B, std::function<void(int slot, std::string error)> done) = 0;
    virtual const float* logits(int slot) const = 0;   // [B][actions], valid after done
    virtual const float* values(int slot) const = 0;   // [B]
};

struct InferenceHubOptions {
    int maxBatch = 256;           // inference_batch: most positions per pass
    uint64_t chaosSeed = 0;       // tests only: nonzero randomizes batching (below)
};

// One per network per phase. Thread-safe; any number of BatchingEvaluators.
class InferenceHub {
public:
    InferenceHub(std::unique_ptr<BatchRunner> runner, const NetConfig& cfg, InferenceHubOptions);
    ~InferenceHub();             // after every evaluator is gone; stops the dispatcher
    std::unique_ptr<Evaluator> makeEvaluator(int maxRequest);   // one per worker; maxRequest <= maxBatch
    struct Stats { long long batches, positions; double gpuSeconds; };
    Stats stats() const;
};
```

A small factory hides the difference between backends from self-play
and the arena:

```cpp
// One evaluator per worker for one network: LocalEvaluators (own Workspace
// and single-threaded backend each) for naive and blas; for metal, one
// InferenceHub with a MetalBatchRunner and a BatchingEvaluator per worker.
class WorkerEvaluators {
public:
    WorkerEvaluators(const Network& net, const std::string& backend, int workers,
                     int maxRequest, int inferenceBatch);
    Evaluator& forWorker(int worker);
};
```

**A request** is one `evaluate(input, B, logits, values)` call from one
worker: B ≤ `mcts_batch` positions, at most 8 by default. The worker's
`BatchingEvaluator` puts a record in the hub's queue under one mutex:
pointers to the caller's input and output arrays, B, and a completion
flag. It then waits on its own condition variable. The caller's arrays
stay valid because the caller is blocked. The worker copies nothing:
the dispatcher copies the planes in and the results out.

**The dispatcher** is one thread per hub, running this loop:

1. Wait until (a slot is free and the queue is not empty) or (a slot
   has completed) or (stop).
2. For each completed slot, copy every request's rows from the slot's
   `logits` and `values` into that request's output arrays, mark the
   requests done and notify their workers. If the pass failed, mark
   them failed with the error. Free the slot.
3. While a slot is free and the queue is not empty, take requests from
   the front of the queue, whole requests only, until the next would
   exceed `maxBatch`. Copy their planes into the slot's staging buffer
   back to back and call `begin`. The completion callback, which runs on
   a Metal thread, only marks the slot completed and notifies the
   dispatcher.

**Flush policy: as soon as the GPU can take it.** A batch is submitted
whenever a slot is free and anything is waiting. There is no timer and
no "wait until every worker has submitted":

- *It self-balances.* While both slots are busy, requests pile up in the
  queue. When a slot frees, the next batch takes everything waiting, up
  to `maxBatch`. Under load, batches grow toward
  workers × `mcts_batch` / slots. When the load is light, no position
  waits for company it may never get.
- *It cannot deadlock.* A worker waits only for its own request, and a
  request waits only for a free slot. Slots free themselves when the GPU
  finishes. No worker ever waits on another worker. "Wait for all
  workers" would have to track workers that are between moves, between
  games or finished, and a wrong count stalls every game.
- *Timing decides batch composition, and that is allowed.* Which
  requests share a batch, in what order, and how large it is depend on
  thread scheduling, and they will differ from run to run. Results do
  not, because of the next section.

`maxBatch` (`inference_batch`, default 256) caps a pass. It must be at
least the largest request, `mcts_batch`, or the arena's per-side
`--a-mcts-batch` / `--b-mcts-batch`. A smaller value is refused with a
message. Two slots match the spike's pipelined measurement. The count
is a constant, `kInferenceSlots = 2`.

### Why the results cannot depend on batching (the determinism argument)

**Claim.** With the Metal runner, the logits, value and ownership the
hub returns for a position are a pure function of the network's weights
and that position's input planes. They do not depend on which requests
share its batch, its index in the batch, the batch size, the slot, the
time, the number of workers, or `inference_batch`.

**Why, kernel by kernel.** In inference mode, every output element for
sample b is computed from sample b's data and the weights alone, by the
same sequence of operations wherever b sits in the batch:

- `transpose_input`, `policy_flatten`: copies of sample b's own
  elements.
- `im2col3`: column (b, pixel) holds sample b's 3×3 neighbourhood, with
  zeros off its own board. Padding never reads a neighbouring sample.
- Convolution gemm: element (out channel, column of b) is one fma chain
  over the fixed K of the layer. It reads the weights' row and b's
  column, in the same order for every tile position (see "The gemm").
  Linear gemm: row b of the output reads row b of the input and the
  weights.
- `bn_infer`, `relu_forward`, `add_into`, `bias_add_rows`,
  `tanh_inplace`: one element each, with per-channel constants (running
  statistics, biases) that do not depend on the batch.
- No reduction runs across samples in inference mode. Batch statistics
  exist only in training (Phase B).
- No atomics. Each output element is written by exactly one thread.
  Threadgroup sizes and dispatch shapes are fixed per kernel, so a
  kernel's instruction sequence for one element does not change with
  the grid.

**From positions to games.** A search uses the evaluator's outputs and
the game's own `Rng`, which is seeded from (seed, generation, phase,
game index) and is touched only by that game's worker. MCTS's
random-symmetry draws happen in the worker, in batch order within the
worker's own request. So every game is a pure function of (seed,
generation, game index, weights), and every phase's records, stored by
game index, are too. That is the existing argument for worker-count
independence, now extended over the GPU.

**What the CPU backends cannot do.** The naive backend is also
batch-independent by construction: each gemm element sums over k in
one fixed order whatever the matrix size. BLAS is not: the spike found
0 of 4 positions bitwise equal alone and in a batch of 256 on
Accelerate. That is harmless today, because each worker evaluates only
its own batches. A BLAS batching evaluator would make results depend on
timing, which is why the project offers none.

**How the tests enforce it** (detailed under "Testing"):

1. Per kernel and for the whole network: positions alone, in batches of
   8, 144 and 256, at different offsets, with random companions and in
   permuted order, give bitwise identical outputs.
2. The hub with a *chaos* option. A nonzero `chaosSeed` makes the
   dispatcher cap each batch at a random size, submit with random short
   delays and shuffle the requests inside a batch. Each request's
   outputs must equal its position evaluated alone, bitwise. The chaos
   generator is the hub's own and touches nothing of the run. This runs
   in every build with a CPU test runner built on the naive backend,
   which is batch-independent, and in Metal builds with the Metal
   runner.
3. End to end. In every build, self-play through the hub with the naive
   runner gives exactly the game records of today's `LocalEvaluator`
   with the naive backend. With Metal, a tiny run gives byte-identical
   files for `workers` 1, 3 and 8, for `inference_batch` 8 and 256, and
   with chaos on.

### Keeping the GPU busy

A worker blocked in `evaluate` uses no CPU. With one worker per core,
18 workers each holding 8 leaves make one batch of 144. The GPU then
idles while they expand those leaves and select the next ones, and
they idle while the GPU computes. To overlap the two, more searches
must be in flight than one batch holds. Three ways were weighed:

| option | how | verdict |
|---|---|---|
| more worker threads than cores | `workers` = 2 × cores: two groups of 18 alternate between CPU and GPU | **chosen.** No change to self-play, MCTS or the arena. Blocked threads cost a stack and nothing else. Results are already independent of `workers` |
| each worker interleaves several games | turn `playSelfPlayGame` and the arena loop into resumable state machines that yield at `evaluate` | rejected for now: a large rewrite of teaching code for the same effect |
| bigger `mcts_batch`, more virtual loss | larger requests per worker | rejected: it changes the search and every result |

So with `backend = metal`, `workers = 0` resolves to twice the
hardware's thread count, capped as now by the number of games. An
explicit `workers` is used as given. Results do not depend on it. The
spike's estimate: 36 searches × 8 leaves keep two batches of ~144 in
flight, about 30k evals/s with the simdgroup kernel, against 14.5k on
the CPU.

**CPU side per batch.** Copying 144 positions' planes in (18 planes × 81
floats each, 0.8 MB) and their logits and values out is microseconds.
The dispatcher's critical sections hold the mutex only to move queue
entries.

### Who uses it

- **Self-play.** `WorkerEvaluators` for the self-play network (the
  best network with gating) replaces the per-worker backends and
  `LocalEvaluator`s. With `metal`, that is one hub, living for the
  self-play loop.
- **Arena.** `playMatch` creates `WorkerEvaluators` per network player
  (two hubs for network against network, one against a bot). Each
  worker's `Seat` takes its evaluator from its player's set. Bots keep the
  `UniformEvaluator`. Random symmetry and per-side `c_puct` and batch
  are untouched: they act inside the worker's search. Matches run one
  after another, so at most two hubs exist at once. Arena matches have
  20–80 games, so fewer searches are in flight than in self-play, split
  between two networks. The arena gains less (see "Performance").
- **Validation loss.** The pass already evaluates batches of 256 in
  order on one thread, so it needs no hub. A `GpuNetwork` with one
  workspace runs `infer` with ownership, and a new CPU function
  computes the loss terms from the outputs:
  `LossStats lossFromOutputs(const NetConfig&, logits, values,
  ownership, pi, z, targets, B)`. It uses `Network::inferLoss`'s
  formulas in double and adds samples in order. A test feeds it the CPU
  network's outputs and checks it equals `Network::inferLoss` bit for
  bit, so the two cannot drift. The CPU path keeps calling
  `Network::inferLoss` unchanged.
- **`analyze` and `gtp`** (optional, last task of Phase A). With
  `--backend metal`, the single search uses a hub with one evaluator.
  The spike puts a batch-8 GPU pass at ~1.8 ms against ~3.2 ms on one
  CPU core. `analyze`'s raw network lines (value, policy, ownership map)
  go through `GpuNetwork::infer`.
- **Training** stays on the CPU pool with BLAS until Phase B.

### Configuration and CLI (Phase A)

| key | values | default | meaning with `metal` | tunable on resume |
|---|---|---|---|---|
| `backend` | `naive`, `blas`, `metal` | `naive` (presets: `blas`) | self-play, arena and validation inference on the GPU through hubs; training, bots and everything else on the CPU with BLAS | yes (see "Switching backends") |
| `workers` | 0–256 | 0 | 0 = 2 × hardware threads (CPU backends: 1 ×) | yes; results independent |
| `inference_batch` | 1–1024, ≥ `mcts_batch` | 256 | most positions per GPU pass | yes; results independent |

The `backend` value parses in every build, so config files are
portable. Availability is checked when a run starts, resumes or
re-rates, before anything is written:

- not built: `backend 'metal' not available: rebuild with -DAZ_USE_METAL=ON (macOS only)`
- built, but no usable GPU: `backend 'metal': no usable Metal GPU (<reason>)`.
  The reason is `no Metal device`, `<name> lacks Apple GPU family 7
  (simdgroup matrices; M1 or later)`, or the shader compiler's message.

The check creates the `MetalContext`, which compiles the shaders, so a
compile failure also stops the run before it writes anything.
Re-rating plays the arena, so a metal run is re-rated with `metal` to
reproduce its live results. `--rerate --set backend=blas` is allowed
and rates with the CPU's slightly different bits.

`cpuBackendName(name)` maps `metal` to `blas` and returns the others
unchanged. Everything that still runs on the CPU uses it. The
per-operation factory `makeBackend("metal")` throws `"backend 'metal' is
not a per-operation backend; use WorkerEvaluators"`, a programming
error that tests check. The trainer prints, when a run starts or
resumes with `metal`:

```
gpu: Apple M5 Max, macOS 27.0.1, shaders 3f1c9a07 (inference: self-play, arena, validation)
```

**Command-line tools.** `arena --backend metal` (with new
`--inference-batch N`, default 256), `analyze --backend metal`, `gtp
--backend metal` and `bench --backend metal`. `--workers 0` in `arena`
follows the same rule as the config. `grow` stays on the naive backend.

**Presets** keep `backend = blas`. `metal` is opt-in per run
(`--set backend=metal`) until it has trained a full reference run.

### Error handling (Phase A)

| situation | behavior |
|---|---|
| `metal` requested, not built; no device; wrong GPU family; shader compile or pipeline failure | error at start, resume or rerate, nothing written; the CLI tools exit 1 with the message |
| buffers exceed `device.recommendedMaxWorkingSetSize`, or an allocation returns nil | `std::runtime_error` naming the bytes and suggesting a smaller `inference_batch`, when the hub is created |
| a command buffer ends with an error (a GPU fault or timeout) | every request in that batch, and every later request to the hub, fails. Each waiting `evaluate` throws `std::runtime_error("GPU inference failed: <error>")`, `parallelFor` rethrows the first, and the generation fails. `state.bin` is the commit point, so resuming replays the generation exactly |
| `inference_batch` below the largest request | `ConfigError` (config) or usage error (CLI), naming both values |
| `upload` meets a tensor it does not know | `std::logic_error` naming it |

No Objective-C exception or callback crosses into C++ code that could
throw. The completion handler only stores the status and notifies. Each
`.mm` entry point checks `NSError`s and statuses and throws C++
exceptions itself. A hub's destructor waits for its in-flight passes
before freeing their buffers. Every `.mm` entry point and the
dispatcher's loop body run inside `@autoreleasepool`, because command
buffers are autoreleased and no thread here has a run loop to drain
them.

### Performance (Phase A)

**Measurement.** `bench --backend metal` adds two tables per network
shape:

- **Raw GPU throughput**: `GpuNetwork` forward passes at batches 8, 144
  and 256, synchronous and pipelined with two slots.
- **The hub under load**: N worker threads (`--hub-workers 18,36,72`)
  each submit batch-8 requests of random positions in a loop through
  one hub, as self-play does. The table gives total evals/s, mean batch
  size and GPU busy fraction, next to the CPU's 18-worker figure.

Real-run measurement: one generation of the 6×96 9×9 preset variant
with `blas` and with `metal`, the same seed, comparing `selfplay_s`,
`arena_s` and the loss columns. CPU timings need an otherwise idle
machine: pause the training jobs (SIGSTOP/SIGCONT), as the spike did,
with Mark's go-ahead.

**Expectations (M5 Max, 9×9 6×96).** The spike's rule: self-play and
arena time scales by 0.1 + 0.9/r, where r is GPU over CPU network
throughput. r = 2.07 with the simdgroup kernel at batch 144, 1.48 with
the scalar kernel.

| | CPU now | expected with metal | acceptance |
|---|---|---|---|
| hub throughput, 36 workers × batch 8 | 14.5k evals/s (18 CPU workers) | ~28–30k | ≥ 25k |
| self-play phase | ~200–240 s | ~110–130 s (× 0.535) | ≥ 1.6× faster |
| arena phase | ~45 s | ~25–30 s | ≥ 1.3× faster (fewer searches in flight, two networks) |
| validation loss | ~3 s | ~1 s | — |
| whole generation | ~285–325 s | ~175–200 s | ≥ 1.4× faster |

If the hub misses its target, the levers are, in order: more workers
(3–4 × cores), a larger `inference_batch`, three slots, kernel fusion
(BN + skip + ReLU), and implicit GEMM. Each is measured with `bench`
before it is kept.

---

## Phase B — GPU training

Phase B moves the training step onto the GPU, reusing everything from
Phase A. Since Phase A, the step at 93 ms on 18 CPU threads is ~40 s of
a generation that is now ~190 s. The spike's whole-step GPU time was
28.7 ms, with stand-in kernels, so the phase should drop to ~12–16 s.

### What Phase B reuses and adds

| from Phase A, unchanged | added in Phase B |
|---|---|
| `MetalContext`, the build, embedding, `shader_types.h` | `col2im3` (a gather in the CPU's order, no atomics) |
| `gemm_simd` | split-K for weight gradients + `splitk_reduce` |
| `transpose_input`, `im2col3`, `relu_forward`, `add_into`, `policy_flatten`, `bias_add_rows`, `tanh_inplace` | `bn_train_stats`, `bn_train_apply`, `bn_backward_sums`, `bn_backward_apply`, `relu_backward`, `policy_unflatten`, `bias_grad_rows`, `fill_zero` |
| `ParamLayout`, `GpuWeights` (+ `download`) | `GpuOptimizerState`: `grads` and `velocity` arenas in the same layout; `sgd_update` |
| `GpuWorkspace` (`Purpose::Inference`) | `Purpose::Training`: every saved activation, `col`/`dcol`, `lossTerms`, `logZ`, `bnScratch`, `splitK` |
| `GpuNetwork::encodeForward(Mode::Infer)` | `Mode::Train` and `Mode::Probe`; `encodeLoss`, `encodeBackward`, `encodeSgd` |
| `lossFromOutputs` (validation) | `policy_value_loss` and `ownership_loss` kernels (with gradients) |
| the test harness, probes and skip logic | the per-layer backward tests, step tests |

### The training-engine seam

The CPU training path is `Network::trainStep` followed by
`SgdMomentum::step`, with a `Workspace` and the training `ThreadPool`.
The new seam sits above `Network`, for the reason the spike gave about
`Backend`: a whole step must cross at once.

```cpp
// include/azgo/training_engine.hpp (portable C++17)
class TrainingEngine {
public:
    virtual ~TrainingEngine() = default;
    virtual std::string description() const = 0;
    // Start of the training phase: take the parameters, BN running statistics
    // and momentum (params() order). Until store(), the engine owns them.
    virtual void load(Network& net, const std::vector<std::vector<float>>& velocity) = 0;
    // Network::inferLoss of the loaded network (the validation pass).
    virtual LossStats inferLoss(const float* input, const float* pi, const float* z, int B,
                                const float* ownership) = 0;
    // Network::trainStep followed by SgdMomentum::step(lr).
    virtual LossStats trainStep(const float* input, const float* pi, const float* z, int B,
                                const float* ownership, float lr) = 0;
    // End of the phase: parameters, BN running statistics and momentum back.
    virtual void store(Network& net, std::vector<std::vector<float>>& velocity) = 0;
};
std::unique_ptr<TrainingEngine> makeTrainingEngine(const TrainingEngineOptions&, const NetConfig&);
```

- `CpuTrainingEngine` wraps exactly today's calls in the same order, so
  CPU runs (and Phase A metal runs with `gpu_training = false`) are
  byte-identical to before. `inferLoss` is `Network::inferLoss` for
  `naive`/`blas`. With `backend = metal` it is Phase A's GPU forward
  plus `lossFromOutputs`, so the validation pass keeps its Phase A bits.
- `MetalTrainingEngine` copies everything into GPU arenas at `load` and
  back at `store`. Each step runs on the GPU. Its `inferLoss` is the
  same Phase A path on its resident weights.

`heldOutLoss` gets a variant that takes a batch-loss function, and the
old signature wraps it. The trainer calls only the engine. The engine
lives for one training phase, so, as in Phase A, no GPU state outlives
a phase.

### Training kernels

| kernel | mirrors | notes |
|---|---|---|
| `col2im3` | `Backend::col2im` | each input point adds its ≤ 9 contributions in the CPU's (dy, dx) order from 0; bitwise |
| split-K + `splitk_reduce` | the dW gemm | S = 1 when K < 4096 or the output has ≥ 128 tiles; otherwise S = min(64, K/1024), kChunk = ⌈K/S⌉ rounded up to 16, S = ⌈K/kChunk⌉. Slices added in order z = 0…S−1. Weight gradients only (K = B·n²); forward, dX and Linear products never split |
| `bn_train_stats` | the statistics of `BatchNorm2d::forwardTrain` | one threadgroup of 256 per channel: strided float sums, then a fixed tree; mean, then variance around it (two passes, like the CPU); invStd; running update (unbiased M/(M−1)) unless probing |
| `bn_train_apply` | its normalize step | xhat and y |
| `bn_backward_sums`, `bn_backward_apply` | `BatchNorm2d::backward` | sums in a fixed tree; dx may alias dy |
| `relu_backward`, `policy_unflatten`, `fill_zero` | their CPU loops | bitwise |
| `bias_grad_rows` | the bias loop of `Linear::backward` | one thread per output, serial over samples from 0: bitwise |
| `policy_value_loss`, `ownership_loss` | the per-sample part of `Network::loss` | one thread per sample, serial in the CPU's order; float where the CPU uses double; per-sample terms summed on the CPU in double |
| `sgd_update` | `SgdMomentum::step` per tensor | g = grad + decay·w; v = m·v + g; w −= lr·v |

**Reductions without `double`.** Metal has no fp64. Each of a channel's
256 threads sums a strided slice serially, and a fixed binary tree
combines the partials. The shape depends only on M = B·n², so it is
deterministic. The error grows with M/256 + 8 additions (89 at batch
256 on 9×9). If batch norm's backward sums turn out to be the largest
error, the fallback is Kahan summation in the strided phase; safe math
mode keeps it from being optimized away.

### One step, one command buffer

One command buffer with one serial compute encoder: `transpose_input`;
the forward pass with batch statistics in `Network::forward`'s order,
updating the BN running statistics; the loss kernels; `fill_zero` on
the ownership head's gradients when it gets none (no targets or
`ownership_weight = 0`), matching the CPU's `zeroGrad`; backward in
`Network::backward`'s order, writing gradients with beta 0 (every
parameter gets its gradient from one place); `sgd_update` per tensor
with its decay. Then commit, wait, check the status, and read back the
per-sample loss terms (B × 3 floats). A 6×96 step is about 300
dispatches, ~1–1.5 ms of GPU time, plus one ~130 µs round trip.

Sampling the next batch on the CPU pool (~0.4 ms) does not overlap the
step. Overlap would save ~1%.

### Synchronization points (Phase B)

| when | direction | what |
|---|---|---|
| `load` (start of the training phase, before the validation loss) | CPU → GPU | every parameter, BN running statistic and momentum tensor; grads zeroed |
| before each step | CPU → GPU | the batch: planes, π, z, ownership targets |
| after each step | GPU → CPU | the per-sample loss terms |
| `store` (after the last step, before the finite scan, the checkpoint and `state.bin`) | GPU → CPU | every parameter, BN running statistic and momentum tensor |

Gradients and activations are never copied back. After `store`, the
CPU `Network` and momentum are exactly the GPU's bits, and the trainer
continues as today.

### Memory (Phase B)

The training workspace mirrors `Workspace` field for field, with one
shared `col`/`dcol` pair. Backward rebuilds each convolution's columns
from its saved input, as the CPU does: a column matrix per convolution
would cost 13 × 72 MB on 6×96. Total: ~390 MB for 6×96 on 9×9 at batch
256, ~3.0 GB for 19×19 10×128, checked against the device's
recommended working set when the engine is created.

### Configuration (Phase B)

`gpu_training = false | true` (default false, tunable on resume, only
meaningful with `backend = metal`; `ConfigError` with another backend).
Opting in keeps a Phase A run's bits unchanged when the binary is
upgraded: the training engine changes only when someone asks for it.
The startup line then reads `(inference: …, training)`.

### Performance (Phase B)

| | CPU (18 threads) | spike | expected | acceptance |
|---|---|---|---|---|
| 6×64 step, batch 256 | 52 ms | 16.8 ms | ~20 ms | ≤ 30 ms |
| 6×96 step | 93 ms | 28.7 ms | 30–40 ms | ≤ 45 ms |
| 6×96 training phase | ~40 s | — | 12–16 s | ≤ 20 s |
| 19×19 10×128 step | (4.23 s on 1 thread) | 280 ms | 300–400 ms | ≤ 500 ms |

`bench --backend metal --gpu-training` times the step (wall and GPU),
`load` and `store`. A real one-generation run with `gpu_training` off
and on must agree on `policy_loss` and `value_loss` within 2%.

---

## Determinism contract (both phases)

With `backend = metal`, on one machine, GPU family and macOS version:

1. **Reproducible.** The same run gives the same checkpoints,
   `state.bin`, `results.csv`, `elo.csv`, SGF files and `log.csv`
   (timings aside).
2. **Independent of `workers`, `train_threads`, `inference_batch` and
   thread or GPU timing.** Phase A by the batch-independence argument.
   Phase B because the GPU runs one fixed sequence of kernels with
   fixed shapes, and the pool only assembles batches in its existing
   fixed chunks.
3. **Byte-identical resume.** Every phase starts by uploading from the
   CPU `Network` (and, in Phase B, momentum from `state.bin`), and no
   GPU state outlives a phase. A resumed generation therefore starts
   from exactly the inputs an uninterrupted one had, and the GPU
   arithmetic is deterministic.

The kernels guarantee this with no atomics, one writer per output,
reductions inside one threadgroup in a fixed tree or in a separate
fixed-order pass, threadgroup sizes and dispatch shapes fixed per kernel
and shape, and no reads of uninitialized memory.

**Not guaranteed:**

- *Across machines, GPU families or OS updates.* The OS compiles the
  shaders at run time, and another compiler may contract a·b + c
  differently or implement `precise::exp` differently. A run resumed on
  another Mac or after a macOS update is valid, but not byte-identical
  to the uninterrupted run. The same holds for Accelerate. The startup
  line shows device, OS and shader hash.
- *Across code versions.* Changing a kernel changes a metal run's bits.
- *Equality with the CPU backends.* See "Tolerance".

**Switching backends.** `backend` and `gpu_training` are tunable on
resume. A run that switches stays valid, since the algorithm is the
same, but from the switch on its bits differ from a run that did not
switch. `blas` ↔ `metal` changes the network's last bits in self-play
and the arena (and in training with `gpu_training`). `naive` ↔ anything
changes them everywhere. On resume, when either key differs from the
saved `config.cfg`, the trainer prints:

```
note: backend changed from blas to metal; from here on results differ in the
last bits from a run that kept blas
```

Nothing is recorded in `state.bin`. `config.cfg` is rewritten with the
new values, as for any tunable key.

## Tolerance against the CPU

`maxErr(x, ref)` = max|x − ref| / max|ref|. `normErr(x, ref)` =
‖x − ref‖₂ / ‖ref‖₂, per tensor. Elementwise relative error is not
used: values near 0, or an activation that crosses ReLU's 0 in one
backend and not the other, would make it meaningless. The reference is
the BLAS backend (a Metal build always has it), or a double-precision
reference computed in the test where noted. The ceilings below come
from float32 error analysis and the spike. Each test first measures the
actual error over 20 seeds and records it in a comment. A measured
error above a ceiling is a bug to find, not a ceiling to raise, unless
the cause is documented (for example float against double in batch
norm's backward).

| phase | quantity | metric | ceiling |
|---|---|---|---|
| A | transpose, im2col, flatten, ReLU, add | bitwise vs CPU | 0 |
| A | gemm (any trans, alpha, beta) | maxErr vs double | 1e-5 (spike: 1.3e-6 forward; measured 1.5e-6) |
| A | conv forward, Linear forward | maxErr vs CPU | 2e-6 (measured 5.6e-7, 7.5e-7†) |
| A | BN inference, tanh | maxErr vs CPU | 1e-6 (measured 0, 1.8e-7) |
| A | whole inference forward: logits, value, ownership | maxErr vs `Network::infer` | 1e-5 |
| A | `lossFromOutputs` on CPU outputs | bitwise vs `Network::inferLoss` | 0 |
| A | validation `LossStats` (GPU outputs) | relative | 1e-5 |
| A | **batch independence** (any kernel, whole network, hub) | bitwise | 0 |
| B | col2im, ReLU backward, unflatten, bias gradient | bitwise | 0 |
| B | gemm with split-K | maxErr vs double | 1e-5 (spike: 6.4e-7) |
| B | conv dX / dW | maxErr vs CPU | 2e-6 / 1e-5 |
| B | BN train forward: y, xhat, invStd / running stats | maxErr | 1e-5 / 1e-6 |
| B | BN backward: dx, ggamma, gbeta | maxErr | 1e-4 |
| B | Linear backward | maxErr | 1e-5 |
| B | per-sample loss terms, `LossStats` | relative | 1e-6 |
| B | one `trainStep`: `LossStats` / each gradient | relative / normErr, maxErr | 1e-5 / 1e-4, 1e-3 |
| B | five steps (lr 0.02, momentum 0.9, decay 1e-4): `LossStats` / weights / momentum | relative / normErr / normErr | 1e-4 / 1e-5 / 1e-3 |

Measured values (tasks A2–A3) are the worst over 20 seeds on an M5 Max
with macOS 27.0.1 (`AZGO_METAL_SEEDS=20 build-metal/test_metal`). Every
batch-independence check was bitwise equal. BN inference matched the CPU
bit for bit. † Linear forward is measured per layer over its outputs at
batches 1, 16 and 256 together: value fc2 at batch 1 is a single number,
and when its bias cancels the product, max|ref| no longer measures the
layer's scale (3.3e-5 relative for an absolute error of 8e-8).

Over a generation, metal and BLAS runs drift apart in their last bits,
as two CPU builds with different BLAS libraries would. No test compares
them game for game. The real-run comparisons compare losses and
ratings statistically.

## Testing

All GPU tests live in `tests/test_metal.cpp`. It returns 77 when the
build has no Metal or the machine has no usable GPU, and CMake sets
`SKIP_RETURN_CODE 77` on it, so CTest reports Skipped. The project has
no CI, so these tests run locally on an Apple-silicon Mac, and they may
run while training jobs use the CPU.

### Phase A

- **Context**: device, library and every pipeline come up with their
  fixed threadgroup sizes; a trivial kernel round-trips a buffer; the
  description is stable.
- **Kernels against the CPU**, through probes (`azgo/metal.hpp`: host
  arrays in, one command buffer, host arrays out), at the ceilings
  above. Boards 5×5, 9×9 and 19×19; batches 1, 2, 7, 144 and 256; 1, 2,
  18, 32 and 96 channels; gemms with every trans combination, sizes
  that are not multiples of the tile, alpha ≠ 1, beta 0 and 1, and a
  NaN-filled C with beta 0.
- **Whole forward against `Network::infer`** for 5×5 1×8, 9×9 2×16,
  9×9 6×32, 0 blocks, and with the ownership head.
- **Batch independence, bitwise**: positions {0, 1, 7, 100, 255} alone,
  in batches of 8, 144 and 256, at different offsets, among random
  companions and in permuted order: per kernel (gemm columns and rows,
  im2col) and for the whole network.
- **Run-to-run reproducibility**: 20 identical passes give identical
  bytes; two separately created `GpuNetwork`s agree bitwise.
- **The hub, in every build** (`tests/test_inference_hub.cpp`, with a
  `CpuTestRunner` that runs `Network::infer` on the naive backend
  asynchronously on a helper thread):
  - each request's outputs equal its positions evaluated alone, with 1,
    4 and 32 client threads, `maxBatch` from the request size to 512,
    and chaos on;
  - requests larger than `maxBatch` are refused;
  - a failing runner makes every waiting and later `evaluate` throw the
    error, and the hub shuts down cleanly;
  - the hub shuts down cleanly with idle and with busy clients, and
    stats add up (positions = sum of requests).
- **Self-play through the hub (every build)**: with the naive test
  runner, game records are identical to today's per-worker
  `LocalEvaluator` on the naive backend, for workers 1 and 6, chaos on.
  This is the end-to-end proof that the hub changes nothing but
  throughput.
- **The hub with the Metal runner**: the same request-level tests;
  chaos on, outputs identical.
- **Trainer with `backend = metal`** (tiny 5×5 config):
  - byte-identical files for `workers` 1, 3 and 8, for
    `inference_batch` 8 and 256, and with chaos on (through a test hook);
  - a resumed run is byte-identical to an uninterrupted one
    (checkpoints, `state.bin`, results, ratings, SGF, validation
    columns);
  - two fresh runs are identical;
  - `--rerate` reproduces the live results;
  - a GPU failure injected through a test hook leaves `state.bin` at the
    previous generation, and a resume reproduces the uninterrupted run.
- **Arena** with `--backend metal`: network against network, against a
  bot and with random symmetry, independent of `--workers`.
- `test_config`: `backend = metal`, `inference_batch` (range, the
  ≥ `mcts_batch` rule), the `workers = 0` resolution.
- `test_validation`: `lossFromOutputs` on the CPU network's outputs
  equals `Network::inferLoss` bitwise, including a partial batch and
  ownership.
- `test_trainer`: in a build without Metal, starting a run with
  `backend = metal` fails with the message and writes nothing.
- `integration_smoke.sh` takes a third argument, `metal=ON|OFF`, from
  CMake. ON: the tiny run trains two generations with `--set
  backend=metal`, resumes to a third and re-rates; `arena`, `analyze`
  and `gtp` run with `--backend metal`. OFF: the same commands fail
  with the not-built message, and no `state.bin` is written.

### Phase B

- Per-layer backward tests against `Conv2d`, `BatchNorm2d` and `Linear`
  backward, and loss kernels against a double reference, at Phase B's
  ceilings.
- Whole network: one `trainStep` (losses, every gradient, running
  statistics) and five steps (losses, weights, momentum), for the same
  configurations plus w = 0 (head gradients zero, head weights still
  decayed) and no targets.
- Reproducibility: one step 20 times, and two engines three steps each,
  give identical bytes.
- `test_training_engine` (every build): `CpuTrainingEngine` equals
  `Network::trainStep` + `SgdMomentum::step` bitwise, naive and BLAS,
  with and without the head. The `heldOutLoss` variant equals the old
  signature.
- `MetalTrainingEngine` against `CpuTrainingEngine` within tolerance;
  `load`/`store` round-trip bitwise.
- Trainer with `backend = metal`, `gpu_training = true`: resume
  byte-identity, `workers` and `train_threads` independence, failure
  injection, and the switching note.
- Old against new binary: short naive and BLAS runs of main's `train`
  and the branch's write byte-identical files. The seam refactor must
  change no CPU bit.

## Toward later work

- **Interleaved games per worker.** If more threads stop scaling, each
  worker can run several games as state machines. Results stay the
  same, by the same argument.
- **Fusion and implicit GEMM**, measured with `bench`. They keep
  determinism, but they change a metal run's bits, like any kernel
  change.
- **Overlapping training-batch sampling with the GPU step**: double
  staging, the same pattern as the hub's slots.

## Teaching material

- **README.** Build: the `AZ_USE_METAL` command and its requirements
  (an Apple-silicon Mac; the Command Line Tools suffice). Usage: a
  **Backends** paragraph after **Threads**: what `metal` moves to the
  GPU in each phase, the determinism contract and its "same machine and
  macOS" limit, `workers = 0` with metal, `inference_batch`,
  `gpu_training`, the switching note. "Extending to the GPU": rewritten
  with the measured numbers. The test count: `test_metal`,
  `test_inference_hub` and (Phase B) `test_training_engine`, with
  `test_metal` reported as Skipped elsewhere.
- **Explainer.** Chapter 03's paragraph "This is where a GPU would plug
  in" predates the spike. It is rewritten to describe the hub as built:
  workers block in `evaluate`, a dispatcher packs leaves from every game
  into one pass, and the batch-independence argument proves that
  timing cannot change a game. It quotes the dispatcher loop and the
  gemm tile (scalar first, then simdgroup). An HTML diagram shows
  workers, queue, two slots and the GPU. Phase B adds a section to
  chapter 08 after the training-threads passage, "Training on the
  GPU": why the seam moved above `Network`, residency and the two
  crossings, the gather col2im and the fixed split-K, no `double` on
  the GPU, and the measured tolerance.
- This spec and its plan are listed under "Learn how it works", and
  `tools/check_citations.py` keeps passing, extended to `.mm` and
  `.metal` files if needed.

## Risks

| risk | consequence | mitigation |
|---|---|---|
| A kernel that is not batch-independent (e.g. a future "optimization" that splits K or reduces across samples) | results depend on timing: the worst kind of bug, intermittent | the bitwise batch-independence tests per kernel and for the network; chaos tests through the hub; worker-count tests end to end |
| The GPU is under-fed in self-play | less gain | `workers` = 2 × cores by default; the hub benchmark measures batch size and busy fraction; the levers above |
| The arena gains little | small matches, two networks | measured; acceptable, since self-play dominates |
| Shader compilation varies with the OS | resume across an OS update is not byte-identical | documented; device line in the log |
| The Command Line Tools' SDK lacks something the build needs | no build on the Mac minis | Task A1's acceptance: build and pass `test_metal` on a CLT-only machine |
| Two implementations of every layer drift apart | metal runs evaluate or train a slightly different network | whole-network GPU-vs-CPU tests in every configuration; `upload` refuses unknown tensors |
| A wrong factor in a backward kernel (Phase B) still trains | silently worse runs | per-layer backward tests, whole-step gradients, real-run loss comparison |
| Many blocked threads, one dispatcher | lock contention | the mutex is held only to move queue entries; the bench reports throughput against worker count |
| GPU memory on a 16 GB Mac mini | 19×19 hubs (~0.6 GB per slot at batch 256) and Phase B training (~3 GB) | the size check with a clear message; `inference_batch` lowers it |
| Autoreleased Metal objects accumulate | memory grows during a phase | `@autoreleasepool` around every entry point and dispatcher iteration; the bench watches resident size |
