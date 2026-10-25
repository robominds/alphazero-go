# Core ML Engines (16-bit, Neural Engine and GPU) — Design

Author: Mark Castelluccio, with design assistance from Claude (Anthropic).

Builds on `2026-10-09-metal-gpu-backend-design.md` (the BatchRunner, the
inference hub and the 32-bit determinism contract),
`2026-10-18-distributed-selfplay-design.md` and
`2026-10-20-distributed-arena-design.md` (engines, the hosts file,
cluster_sync, fingerprints, manifests), `2026-10-20-cuda-inference-backend-design.md`
(an engine-only backend) and `2026-10-23-dynamic-selfplay-design.md`
(dynamic allocation). Mark's decision: "implement the 16-bit solution for
the Macs across the CPUs and GPUs" — Core ML engines on the Apple Neural
Engine and on the GPU, while the Macs' CPUs keep the 32-bit BLAS engines.

The algorithm does not change. A game played by `mini1/ane` is the same
AlphaGo Zero self-play game, seeded the same way, from the same network.
Only the arithmetic of its network evaluations is different: float16
instead of float32, so its outputs agree with the CPU's within a stated
tolerance, not bit for bit, and a run that uses these engines is
reproducible given their fingerprints, as one that uses `metal` is.

## Problem

Self-play of the live 19×19 10×128 run (800 simulations) evaluates about
8,700 positions per second across the cluster (generation 7 of
`selfplay-engines.csv`):

| engine | evals/s |
|---|---|
| local/metal (M5 Max Studio, 36 workers) | 1,550 |
| mini1/metal, mini2/metal (M4, 4 workers each) | 260 each |
| pc/cuda (RTX 4070 Ti SUPER) | 6,620 |

The project's Metal kernels are fp32 and hand-written for batch
independence; at 19×19 they reach 1,400–1,450 evals/s on the Studio even
alone (measured below). A feasibility study (Core ML models built with
coremltools, `MLComputePlan` placement, benchmarks; raw data kept with the
study) found:

- With `cpuAndNeuralEngine`, every operation of the network runs on the
  Neural Engine for fixed batch sizes 1–256. About 7,500 evals/s on the
  Studio's M5 Max and 4,800 on an M4 mini, flat from batch 8.
  Enumerated or flexible shapes fall back to the CPU.
- On the Neural Engine the outputs were bitwise independent of the batch
  size, of a position's companions and of its row, and repeatable; they
  differ across chips.
- Core ML's float16 GPU path (`cpuAndGPU`) reaches 13,000–15,000 evals/s on
  the Studio and about 1,060 on a mini. Its bits differ between batch 8
  and 64, but not with a pass's companions or rows at one size.
- Core ML's CPU float16 path was slower than BLAS and less accurate.

## Goals

1. Two engine backends, `ane` and `coreml_gpu`, usable wherever an engine
   is: `selfplay_engines`, `arena_engines`, static and dynamic allocation,
   local and remote (`selfplay --backend ane`).
2. No Python and no converter at run time. The program writes the Core
   ML model itself from a `Network`, so every Mac converts the `.azn` it
   receives.
3. A stated contract: reproducible given the engine fingerprint; a
   position's outputs independent of its pass (checked per device); the
   float16 tolerance tier against the 32-bit CPU, defined from
   measurements.
4. The default path unchanged: a build without the option, and a build
   with it whose run uses no Core ML engine, write byte-identical files to
   the code before; existing fingerprints do not change.
5. Strength: a network evaluated by `ane` plays itself evaluated by the
   32-bit Metal engine at about 50%.

## Non-Goals

- Training, the validation loss, the outside ratings (GNU Go, Pachi) and
  the config's `backend` stay 32-bit. `backend = ane` is refused, as
  `backend = cuda` is: the Core ML engines are engines only.
- The Macs' CPUs: Core ML's CPU float16 path is slower and less accurate
  than BLAS. CPU engines stay `blas`.
- float16 on the PC's CUDA GPU (tensor cores): a later, separate design.
- The ownership head on the engines: search never reads it, so the model
  leaves it out.
- Choosing the batch size per model at run time, flexible shapes, or Core
  ML's `.all` compute units (which pick the GPU at batch 64 and up).

## Overview

```
WorkerEvaluators(net, "ane" | "coreml_gpu")
  └─ InferenceHub ── BatchRunner = CoreMLBatchRunner (1–2 slots, a thread each)
        └─ the network's models, one per compiled batch size (ModelCache)
              buildCoreMLModel (model_writer.cpp): Network → ML Program
              → .mlpackage → MLModel compileModelAtURL → load (compute units)
              → MLComputePlan placement check (first load of each shape)
```

Files:

- `include/azgo/coreml.hpp` — the public interface, plain C++17.
- `src/coreml/model_writer.cpp` — the writer, float16 conversion, chunking,
  hashes. Portable, compiled in every build.
- `src/coreml/coreml_info.cpp` — the "not built" answers.
- `src/coreml/coreml_runner.mm` — Objective-C++: compile, load, cache,
  placement, the runner, the probe, fingerprint, memory checks.
  Only with `-DAZ_USE_COREML=ON`.

## The model writer

`buildCoreMLModel(net, batch)` returns the three files of an `.mlpackage`:
the protobuf specification (`Data/com.apple.CoreML/model.mlmodel`), the
weight blob (`Data/com.apple.CoreML/weights/weight.bin`) and
`Manifest.json`. The protobuf is written by hand: a tag (field number and
wire type), then a varint or a length-prefixed string. The writer needs
only varints, strings, nested messages, packed repeated integers and maps
(repeated `{1: key, 2: value}` messages). The project takes on no protobuf
library. The field numbers are those of coremltools' `Model.proto`,
`FeatureTypes.proto` and `MIL.proto`.

- **The model.** Specification version 9 (macOS 15), an ML Program with
  one function, `main`, in opset `CoreML8`. Input `planes`, float32
  `[B][planes][n][n]`: the Evaluator's layout, so the hub's staging rows
  are passed without a copy. Outputs `logits` float32 `[B][n·n+1]` and
  `value` float32 `[B]`.
- **The graph**, all float16 inside: a cast of the input; the stem (3×3
  conv, ReLU); per block conv, ReLU, conv, add the skip, ReLU; the policy
  head (1×1 conv to 2, ReLU, reshape `[B][2A]`, linear, cast to float32);
  the value head (1×1 conv to 1, ReLU, reshape, linear, ReLU, linear,
  tanh, reshape `[B]`, cast). That is `Network::forward` in inference mode.
  The reshape of `[B, 2, n, n]` to `[B, 2A]` is the CPU's policy flatten.
- **Batch norm folded.** For output channel c,
  `s = gamma / sqrt(var + 1e-5)` in double; the weights become `w · s`
  and the bias `beta − s · mean`, rounded to float, then to float16 to
  nearest even. Linear layers' weights are rounded the same way. The
  float → float16 conversion (`floatToHalf`) is tested against every
  half, every midpoint and the compiler's `_Float16` on a million random
  floats.
- **Constants.** Every operation parameter is a `const` operation, as
  coremltools writes them. float16 tensors of 10 or more elements live in
  the blob (MIL blob storage version 2: a 64-byte header, then a 64-byte
  record `0xDEADBEEF`, data type, size, offset, and the data, each on a
  multiple of 64 bytes); smaller ones are inline bytes.
- **Deterministic.** The manifest's identifiers are derived from the
  content's hash, so the same network at the same batch always gives the
  same bytes. `coremlModelFormatHash()` hashes the writer's output for a
  fixed small network: it stands for the writer in the fingerprint and
  changes exactly when the writer's output changes.

The writer's output was checked against coremltools 9.0, in the
feasibility study's scratch environment only (the project does not depend
on it): on the Studio's Neural Engine and GPU, the model built by this
writer and the model built by coremltools from the same checkpoint (19×19
generation 7, 512 positions) gave bitwise identical outputs.

## The runtime

- **Loading.** `ModelCache::get(net, backend)` writes the package for
  each compiled size into a private temporary directory, has Core ML
  compile it (`compileModelAtURL:completionHandler:`; the synchronous call
  is deprecated), moves the `.mlmodelc` into that directory and loads it
  with the backend's compute units. The probe and the first 19×19
  network's three models together take about 1.9 s in a fresh process.
- **Placement check.** The first load of each (backend, network shape,
  batch size) asks `MLComputePlan` where each operation runs. If any
  non-const operation's preferred device is not the backend's (the Neural
  Engine for `ane`, the GPU for `coreml_gpu`), the load fails: "ane: Core
  ML places 22 of 22 operations off the ANE at batch 8 (...)". Core ML's
  cost model keeps small networks on the CPU: 5×5 with 1×8, 9×9 with
  history 2 and 32 filters, 19×19 with history 2 and 16 filters. Every
  preset from 9×9 6×64 up and both live networks run entirely on the
  device (`test_coreml`, on the Studio and a mini).
- **The batch-independence check.** The same first load of a shape then
  runs eight fixed positions each alone and all together, through every
  compiled size, and refuses the network unless every output is bitwise
  equal ("ane: a position's outputs differ alone and among others at batch
  32: ane is not batch-independent on this machine"). The contract below
  rests on Core ML's behaviour, so every process checks it on its own
  device rather than trusting a test run elsewhere.
- **Scratch files.** A process keeps its packages and compiled models
  under `$TMPDIR/azgo-coreml-<pid>`, removed when it exits; the first use
  also removes the directories of processes that no longer exist (killed
  ones never ran their exit handler).
- **The cache.** Models are kept per (network hash, backend) while a hub
  uses them; the four most recently used stay loaded after that, so an
  arena that builds a hub for the same network wave after wave compiles it
  once. Their directory is removed when the last user goes.
- **Batch sizes and chunks.** `ane` compiles 8, 32 and 128; `coreml_gpu`
  only 64 (see the contract). A pass of B positions runs as chunks:
  repeatedly the largest size that fits what is left, then the smallest
  size that holds the rest, padded with zero rows (`coremlChunks`). The
  hub's passes are multiples of `mcts_batch`, so with mcts_batch = 8
  `ane` never pads.
- **The runner.** `CoreMLBatchRunner : BatchRunner`, 1 or 2 slots, each
  with its own thread. `begin` hands the pass to the slot's thread, which
  runs one synchronous `predictionFromFeatures` per chunk. The input
  multi-array points into the slot's staging rows. The outputs are copied
  out row by row with their strides, then `done(slot, error)` is called.
  Predictions on one model may run concurrently, so two slots put two in
  flight: on the Studio two threads give 7,590 against 7,260 evals/s on
  the Neural Engine at batch 128 and 12,140 against 9,160 on the GPU.
- **Errors.** A failed prediction is tried twice more, after 20 and 40
  ms (the same input gives the same bits). One failed once, transiently,
  in hours of tests: on the Neural Engine, while another process loaded
  models. A third failure fails the pass ("coreml: prediction failed:
  ..."); the hub reports "GPU inference failed" to every waiting
  request and every later one. A remote slice then fails, and the
  coordinator retries or falls back as for any engine.
- **Availability.** `coremlAvailable(backend)` needs a Core ML build,
  macOS 15, the device in `MLAllComputeDevices()`, and a probe network
  (9×9, history 8, 2×64) compiled, loaded and run. It runs once per
  process and backend. `checkBackendAvailable` calls it, so a local
  engine is refused before a run writes anything, and a remote one at
  preflight (`selfplay --fingerprint --backend ane`).
- **Memory.** `checkCoreMLInferenceMemory` adds up, per hub, every
  compiled model (float16 weights and three activation tensors of its
  size) and the slots' float32 staging. The total must fit the GPU's
  recommended working set for `coreml_gpu`, or a quarter of the machine's
  memory for `ane`. It is a coarse and conservative bound: hubs of one
  network share its models but are each charged in full. The coordinator checks its local engines' hubs up
  front, self-play's beside the arena's with `pipeline = on`. A 19×19
  hub at inference_batch 256 needs about 90 MB on `ane` (three models)
  and 40 MB on `coreml_gpu`, so this matters only for much larger
  networks.

## Integration

- **Build.** `option(AZ_USE_COREML ...)`, macOS only, turns BLAS on (as
  Metal does) and needs only the Command Line Tools. It links CoreML and
  Foundation and defines `AZ_USE_COREML`. It is independent of
  `AZ_USE_METAL`; the Macs use both. `buildOptions()` reads
  `metal+coreml`, or `coreml` without Metal.
- **cluster_sync.** The build directory's word `coreml` adds
  `-DAZ_USE_COREML=ON` to whatever the other words ask for:
  `build-metal-coreml` has Metal and Core ML, `build-coreml` Core ML (with
  BLAS). Every other name configures `-DAZ_USE_COREML=OFF` explicitly.
  So `build-metal` keeps its meaning: the minis' existing builds do not
  change, and switching a mini to Core ML means pointing its `exe` at
  `build-metal-coreml`.
- **Names.** `hubBackend(name)` is true for metal, cuda, ane and
  coreml_gpu. Every place that treated metal and cuda as "a GPU behind a
  hub" uses it: worker defaults, QoS (hub engines keep the default class),
  `cpuBackendName` (blas), the arena's evaluators.
- **The hosts file** gains `ane_workers` and `coreml_gpu_workers`. Like
  `metal_workers`, 0 means every game of the slice, or in dynamic mode a
  quarter of the share. `hostsWorkers(machine, backend)` names the key.
- **A note.** When one machine's engines (either list) include both
  `metal` and `coreml_gpu`, `train` prints "note: local/metal and
  local/coreml_gpu share one GPU, where Metal's passes starve Core ML's;
  use coreml_gpu in place of metal" (see "Measured").
- **The config.** `backend = ane` and `backend = coreml_gpu` are refused:
  "ane is an engine only, not a run's backend: use selfplay_engines or
  arena_engines". `engine_weights.py` treats the Core ML engines as GPU
  engines.
- **bench** `--backend ane|coreml_gpu` adds the engine's rows (synchronous
  passes of 8) to the inference table. A preset Core ML places off the
  device reads 0, with the reason on stderr.

## Fingerprint

`engineFingerprint("ane")` = `ane; <cpu>; <os>; <coremlDescription>`:

```
ane; Apple M5 Max; macOS 27.0.1 (26A434); Core ML 3600.25.2, Neural Engine 16 cores, cpuAndNeuralEngine, float16, batches 8/32/128, model ecc217a1
coreml_gpu; Apple M5 Max; macOS 27.0.1 (26A434); Core ML 3600.25.2, GPU Apple M5 Max, cpuAndGPU, float16, batch 64, model ecc217a1
ane; Apple M4; macOS 27.0.1 (26A434); Core ML 3600.25.2, Neural Engine 16 cores, cpuAndNeuralEngine, float16, batches 8/32/128, model ecc217a1
```

- The chip is the CPU model (the Neural Engine and GPU are on the same
  chip); the OS build pins the Core ML framework and the Neural Engine's
  compiler.
- Core ML's framework build (`CFBundleVersion`) is recorded too.
- The compute units, the precision, the compiled batch sizes (the batch
  layout a position can be evaluated in) and the model format hash.
- The network itself is not in the fingerprint: it changes every
  generation, and the manifests already name each generation's network.

The manifests (`selfplay-engines.csv`, `arena-engines.csv`) record each
slice's engine and this fingerprint, and a fingerprint change prints a
note, as for every engine.

## The contract

On one machine (one fingerprint), with one build:

1. **Reproducible.** Two runs of a slice give byte-identical result
   files; a resume continues byte for byte.
2. **Independent of the pass.** A position's outputs depend on its own
   input and the network only, not on the positions it shares a pass
   with, its row, the chunking, the workers, inference_batch, the slots
   or the hub's chaos mode. For `ane` this holds across the compiled
   sizes too. Unlike Metal's, this is not by construction: it is Core ML's
   behaviour, observed in the study and checked by `test_coreml` on every
   device it runs on. Each position is checked alone, in passes of 8, 64
   and 256 among random companions, first, middle and last, in permuted
   passes, through each compiled size, and through the hub under chaos.
   It held on the Studio (M5 Max) and on a mini (M4), on random preset
   networks and on real 13×13 and 19×19 checkpoints.
3. **The GPU engine compiles one batch size.** Core ML's GPU gives
   different bits at batch 8 and batch 64 (differences of about 2e-6),
   but the same bits for any companions and rows at one size. So
   `coreml_gpu` evaluates every position through its batch-64 model,
   padding the last chunk, and is batch-independent in the sense above.
   The cost is padding at low load: a pass of 8 costs a pass of 64.
   Batch 64 reaches about 90% of batch 256's throughput (13,200 against
   15,000 evals/s in the study). The fingerprint's "batch 64" records the
   layout.
4. **Not guaranteed:** equality across chips (Studio and mini Neural
   Engines differ by up to 0.011 on a logit), OS builds, Core ML versions
   or model formats, all of which are in the fingerprint; equality with
   `metal`, `blas`, `cuda` or each other (tolerance only).

Should an OS update break batch independence, `test_coreml` fails on that
machine: run it there after an update, as `test_metal` and `test_cuda` are
run after driver or OS updates.

### Tolerance (the float16 tier)

Against the network in double precision (`test_coreml`'s reference, every
layer in double with the running statistics), per network over its
positions:

| measure | tier | worst measured, `ane` | worst measured, `coreml_gpu` |
|---|---|---|---|
| logits: max \|diff\| / max \|reference logit\| | 5e-3 | 3.97e-3 (random 19×19), 2.46e-3 (real) | 2.39e-3 (random 9×9), 1.76e-3 (real) |
| value: max \|diff\| | 3e-2 | 1.56e-2 (19×19 gen 7) | 1.01e-2 (random 9×9) |
| policy: max total variation of the softmax | 1e-2 | 3.14e-3 | 2.22e-3 |
| best move where the reference's top two logits differ by ≥ 0.05 | all | all (817 positions on the Studio, 557 on the mini) | all |

Measured on the Studio and a mini, 128 positions per network:

- random networks in the 9×9 6×64, 13×13 6×96 and 19×19 10×128 shapes,
  batch norm calibrated as generation 0 is, on positions from random
  games;
- real checkpoints (13×13 generations 50 and 100, 19×19 generations 3 and
  7) on positions from their own self-play games.

For comparison, the 32-bit CPU and Metal engines are within 1e-6 of the
double reference. The same position under another of the board's
symmetries differs from itself by 0.75 relative on the logits: that is
the variation search already averages over with `mcts_random_symmetry`.
The study found the same agreement: 99.8–100% of best moves, ~2e-3
relative logit error on the Neural Engine and ~6.5e-4 on the GPU for the
real 19×19 network.

The 32-bit tier (1e-5) stays the contract of metal and cuda; this is a
second tier, for the Core ML engines only.

**A bias on the Neural Engine.** The Neural Engine's errors on the value
are not centred. On 13×13 generation 100's 512 study positions the value
is +1.58e-3 too high on average, against a mean absolute error of
1.70e-3, at every |v|; before the tanh the shift is a uniform +2.1e-3.
Core ML's GPU, with the same float16 weights, shows +6.7e-5. It is
already in the value convolution's output: the head's features are
+2.2e-4 too high where they are active (+1.9e-5 on the GPU, measured with
coremltools on this writer's package). Running the head's two linear
layers and tanh on the CPU in float32 (tried) changed the shift only
from +1.58e-3 to +1.54e-3, so the model keeps them, and the writer's
output stays bitwise equal to coremltools'. A shift of every value by the
same small amount cancels in most of search, which compares the Q values
of siblings; the strength check below measures what is left.

## Measured

### Throughput in real self-play

`playSelfPlayQueue` (the engine's own loop: noise, temperature, the hub)
on the 19×19 generation 7 network at 800 simulations, 40 s per row after
the models load, the machine otherwise as noted:

| Studio (M5 Max, 18 cores), live run idle | evals/s | CPU cores |
|---|---|---|
| ane, 4 workers | 6,890 | 0.54 |
| ane, 16 workers | 7,130 | 0.49 |
| ane, 64 workers | 7,260 | 0.37 |
| coreml_gpu, 16 / 32 / 64 / 128 workers | 8,530 / 9,840 / 11,700 / 13,100 | 0.8–0.9 |
| metal, 36 / 64 workers | 1,380 / 1,450 | 0.05 |
| ane 16 + coreml_gpu 64 | 7,135 + 10,750 = 17,885 | 1.1 |
| ane 16 + metal 36 | 7,180 + 1,410 = 8,590 | 0.55 |
| coreml_gpu 64 + metal 36 | 252 + 1,335 | 0.05 |

| mini (M4, 10 cores), its live metal slice running | evals/s | CPU cores |
|---|---|---|
| ane, 4 / 8 / 16 workers | 4,550 / 4,410 / 4,430 | 0.2 |
| coreml_gpu, 16 / 32 workers (beside live Metal) | 490 / 590 | 0.03 |
| ane 8 + coreml_gpu 32 (beside live Metal) | 4,490 + 520 | 0.7 |

What this shows:

- **The CPU is not the limit.** Search, encoding and the hub cost about
  70 µs of one core per evaluation. The Neural Engine is full at 4
  workers on either machine, and a whole Studio engine pair uses about
  one core.
- **The Neural Engine and the GPU work side by side.** ane beside
  coreml_gpu loses nothing on the Neural Engine, and about 8% on the GPU.
  ane beside metal loses nothing either.
- **Core ML's GPU path must replace Metal, not join it.** With both on
  one GPU, Metal's long command buffers starve Core ML's: 252 evals/s
  instead of 11,700. On the Studio the best pair is local/ane +
  local/coreml_gpu, about 18,000 evals/s, 12× today's local/metal. On a
  mini, mini/ane alone gives 17× its metal engine. mini/coreml_gpu
  measured 1,060 alone in the study, so ane + coreml_gpu in place of
  metal gives about 5,500.

### Strength: 16-bit against 32-bit

The same network playing itself, one side evaluated by a Core ML engine
and the other by the 32-bit Metal engine:

- 13×13 generation 100 (6×96), 800 simulations, mcts_batch 8, c_puct 1.5;
- no root noise, the first eight plies sampled at temperature 1, random
  symmetry on both sides;
- colours alternating, A Black in even games, game i seeded by i;
- 16 games in flight on the Studio at nice 10, beside the live run.

A scratch harness (copying the arena's game loop with one
`WorkerEvaluators` per side) played it; the project's arena gives both
players one backend.

| side A vs side B | games | A's score | 95% interval (Wilson) | Elo |
|---|---|---|---|---|
| ane vs metal | 636 | 49.4% | 45.5–53.3% | −4 (−31 to +23) |
| coreml_gpu vs metal | 300 | 55.0% | 49.3–60.5% | +35 (−5 to +74) |
| ane vs ane (control) | 249 | 53.0% | 46.8–59.1% | +21 (−22 to +64) |

Neither shows a loss: both intervals contain 50%. A 3-hour slice of
`ane` against `metal` (636 games) puts any difference within about ±30
Elo; `coreml_gpu`'s 55% over 300 games leans the other way, inside its
interval. The `coreml_gpu` match ran slower and played fewer
games, because Metal's passes starve Core ML's on one GPU (above). The
`ane` match's first 150 games stood at 42.8% before drifting back to 50%:
the control match shows the same spread (53% for one side of identical
engines), so early streaks are noise and longer matches are needed.

## Switching the live run

See the README ("The Core ML engines"). In short: build and sync
`build-metal-coreml`, then at a generation boundary resume with the
Studio's local/metal engines replaced by local/ane and local/coreml_gpu,
and the minis' by mini/ane (and mini/coreml_gpu).

## Testing

`tests/test_coreml.cpp`:

- **Every build.** float16 conversion (every half, every midpoint,
  `_Float16`); chunking; the writer read back by a small protobuf reader
  (the interface, the graph in `Network::forward`'s order, every name
  bound to an earlier value, every weight equal to an independently
  computed fold, the blob's records); determinism and sensitivity to every
  tensor; the package on disk. Without Core ML: the refusals, then 77
  (Skipped).
- **Core ML build, per usable device:** placement at every compiled size
  (and the refusal of a network Core ML keeps on the CPU); accuracy
  against the double reference within the tier; batch independence;
  repeatability across reloads and the cache; the hub under chaos with 1
  and 2 slots; failure injection; the memory checks; whole runs
  independent of workers, inference_batch and chaos, a byte-identical
  resume, and the manifests' fingerprints. `AZGO_COREML_DATA=DIR` adds
  real checkpoints and games.
- `test_distributed`, `test_config`, `test_game_queue`: the names, the
  hosts keys, worker rules, fingerprints, the config refusal, build
  options.
- The smoke test: cluster_sync's `coreml` word.
- Identity: a build without the option, and a Metal+Core ML build whose
  runs use no Core ML engine, against main on the default, distributed,
  dynamic, pipelined and arena-engine paths: every file identical (log.csv
  without its timings, the manifests without seconds and build), except,
  in dynamic mode, which engine played which game (`selfplay-games.csv`
  and the per-engine counts), which depend on timing and differ between
  two runs of main itself too.

## Risks

| risk | effect | mitigation |
|---|---|---|
| an OS update changes Core ML's placement or arithmetic | a fingerprint note; bits change; an operation off the device is refused at load | OS build and Core ML version in the fingerprint; the placement check; run `test_coreml` after updates |
| an OS update breaks batch independence | games would depend on timing | `test_coreml` per device; the contract states it is observed, not constructed |
| Metal and Core ML GPU on one GPU | Core ML's GPU starves | replace metal with coreml_gpu on a Mac's self-play and arena engines; the validation loss's Metal pass runs between phases |
| float16 costs strength | weaker self-play data | the strength check above; the tier; AlphaGo Zero's search averages symmetries that differ far more |
| many games in flight on one fast engine | long games, a long tail at a generation's end | moderate `coreml_gpu_workers` (64); dynamic allocation moves games |
| Core ML compiles per process | ~1–2 s per slice and network | the per-process cache; slices take minutes |
