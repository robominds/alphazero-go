# Metal GPU Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `backend = metal` runs the network on the Apple GPU with hand-written, deterministic Metal kernels. **Phase A** (tasks A1–A10) does inference: self-play, the arena and the validation loss go through a batching evaluator that gathers the leaves of every worker's search into one GPU pass. **Phase B** (tasks B1–B7) does training: backward, losses, SGD and BN statistics, opt-in with `gpu_training = true`. Results are reproducible bit for bit, independent of `workers`, batching and timing, and byte-identical on resume. They stay within a specified tolerance of the CPU.
**Architecture:** Phase A: `GpuNetwork` (forward kernels mirroring the CPU layers) behind a `BatchRunner`; a portable `InferenceHub` (queue, dispatcher thread, two pipeline slots, greedy flush) hands out `BatchingEvaluator`s to workers; `WorkerEvaluators` hides the choice between it and today's `LocalEvaluator`s. Determinism rests on the kernels' batch independence and is enforced at kernel, network, hub and run level. Phase B: a `TrainingEngine` seam above `Network` (`CpuTrainingEngine` = today's code; `MetalTrainingEngine` reuses Phase A's context, kernels, weights and workspace).
**Tech Stack:** C++17, CMake ≥ 3.16, Objective-C++ (ARC) only in `src/metal/*.mm`, the Metal and Foundation frameworks (no MPS), Accelerate (BLAS), assert-based test executables run by CTest, bash for the smoke test.
**Spec:** docs/superpowers/specs/2026-10-09-metal-gpu-backend-design.md
**Author:** Mark Castelluccio, with planning assistance from Claude (Anthropic).

## Global Constraints

- C++17, CMake ≥ 3.16, default build type Release. `AZ_USE_METAL` defaults to OFF. It is macOS-only (`FATAL_ERROR` elsewhere) and turns `AZ_USE_BLAS` on. A build without it compiles no Objective-C and links no Metal.
- Namespace `azgo`. Public headers in `include/azgo/` are portable C++. Objective-C++ appears only in `src/metal/*.mm`, and no header contains an Objective-C type: Metal objects sit behind pimpl structs.
- **No MPS, MPSGraph, MLX or Core ML.** Kernels live in `src/metal/kernels.metal`. They are embedded by CMake and compiled at run time (`newLibraryWithSource`, `MTLMathModeSafe`). No `metal`/`metallib` tool at build time: the Mac minis have only the Command Line Tools.
- **GPU determinism rules** (every kernel): no atomics; one writer per output; reductions within one threadgroup in a fixed tree or in a separate fixed-order pass; threadgroup sizes are constants in `shader_types.h`, never derived from the device; dispatch shapes depend only on tensor shapes; never read uninitialized memory; transcendentals are `precise::`.
- **Batch independence (Phase A's foundation):** in inference mode, no kernel may let one sample's output depend on another sample, on its index in the batch, or on the batch size. No K-splitting in forward products, ever. Any change that breaks this fails the bitwise batch-independence tests and must not be merged.
- **CPU runs stay byte-identical.** Every change to CPU paths leaves `naive` and `blas` runs bit for bit as before. The existing byte-identity tests must pass unchanged.
- **The algorithm does not change.** Search (including `mcts_batch` and virtual loss), loss, BN and SGD stay AlphaGo Zero's.
- Tests: assert-based executables compiled with `-UNDEBUG`. GPU tests live in `tests/test_metal.cpp`, which returns **77** when Metal is not built or not usable; CMake sets `SKIP_RETURN_CODE 77`. The hub's tests (`tests/test_inference_hub.cpp`) run in every build. Tolerances are the spec's ceilings; never raise one without a documented cause in the test.
- All randomness in tests through `azgo::Rng` with fixed seeds. The hub's chaos generator is its own and is never seeded from a run's streams.
- The machine runs training jobs in `runs/`. Never touch that directory. GPU tests may run beside the jobs. Timings (tasks A9, B7) need the jobs paused with SIGSTOP/SIGCONT, and only with Mark's go-ahead. Build with `-j 4` while jobs run.
- New files open with a teaching comment and `Mark Castelluccio, with assistance from Claude (Anthropic).` Changed headers keep their opening comment accurate.
- Code style: camelCase functions, PascalCase types, `trailing_` members, `kConstants`, 4-space indent, ≤ 100 columns. Metal kernels in snake_case.
- Commits: author Mark Castelluccio <markacastelluccio@gmail.com> (repo-local config); imperative subject; body explains why; message ends with `Co-Authored-By: Claude <model> <noreply@anthropic.com>`, naming the model that writes it.

## Review Focus

1. **Batch independence and the hub's determinism.** Without it, results depend on timing, and the bug would be intermittent. Tests: A2 `test_gemm_columns_and_rows_batch_independent`; A4 `test_infer_batch_independent`; A5 `test_hub_outputs_equal_alone_under_chaos`, `test_selfplay_through_hub_equals_local`; A7 `test_metal_run_independent_of_workers_and_batching`.
2. **The hub's concurrency.** Shutdown, failure propagation, no lost or duplicated request, no deadlock with idle workers. Tests: A5 `test_hub_failure_reaches_every_waiter`, `test_hub_shutdown`, `test_hub_stress`.
3. **Resume and phase boundaries.** Weights are uploaded at each phase start, and nothing outlives a phase. Tests: A7 `test_metal_resume_is_byte_identical`, `test_gpu_failure_keeps_last_good_state`; B6 the same with `gpu_training`.
4. **The CPU path.** Self-play and the arena through `WorkerEvaluators` with CPU backends (A7), and training through `CpuTrainingEngine` (B5), must be today's code. Tests: the existing byte-identity tests, B5 `test_cpu_engine_equals_direct_calls` and the old-vs-new binary comparisons in A7 and B5.
5. **Builds without Metal.** No Objective-C compiled, `test_metal` Skipped, `metal` refused before anything is written. Tests: A1 acceptance; A7 `test_metal_unavailable_writes_nothing`; the smoke test's OFF branch.
6. **Phase B's backward kernels.** A wrong factor still trains. Tests: B1–B3 per-layer, B4 whole-step gradients, B7 real-run loss comparison.

---

## Notes Before You Start

**Read first:** the spec; `docs/results/metal-bench.txt`; `tools/metal_bench/metal_bench.mm`, whose `gemm_simd`, `im2col3`, `bn_relu` (inference BN), `Gpu`/`Ctx` structs and pipelined inference section are the starting point; `src/layers.cpp`, `src/network.cpp`, `src/evaluator.cpp`, `src/selfplay.cpp`, `src/arena.cpp` and `src/trainer.cpp`.

**Decisions binding on every task:**

- *`MetalContext`* is a process-wide lazily created singleton (`std::call_once`) with one command queue, and it holds no run state. Hubs, `GpuNetwork`s and Phase B engines live for one phase.
- *Shared definitions.* `src/metal/shader_types.h` holds every kernel parameter struct and the constants `kGemmThreads = 128`, `kElementThreads = 256`, `kReduceThreads = 256`, `kIm2colGroup = {64, 4}`, behind a tiny `#ifdef __METAL_VERSION__` typedef block. CMake prepends it to the embedded source, and the `.mm` files include it.
- *Buffers and encoding.* `GpuBuffer` is a move-only RAII wrapper: shared storage, zero-filled, with `data()` and `bytes()`. A `BufView` is (buffer, 256-byte-aligned offset). `CommandBatch` wraps one command buffer with one serial compute encoder. It has `commitAndWait()`, which throws with the error text, and `commit(onDone)` for the hub, whose completion handler only stores the status and calls `onDone`. Every `.mm` entry point and every dispatcher iteration runs in `@autoreleasepool`.
- *Probes.* `include/azgo/metal.hpp` declares, under `#ifdef AZ_USE_METAL`, `namespace azgo::metal_probe`: host arrays in, one command buffer, host arrays out. They are for tests and `bench` only.
- *Tolerance metrics* in tests: `maxErr` = max|x − ref| / max|ref|; `normErr` = ‖x − ref‖₂/‖ref‖₂; `bitwise` = `memcmp`. Each test prints its measured worst error once with its ceiling.
- *Reference backend* for GPU-vs-CPU: `makeBackend("blas")`. For batch independence, the reference is the GPU itself (alone against batched).
- *Test runtime:* keep `test_metal` under ~30 s and `test_inference_hub` under ~10 s.

**Spec ambiguities resolved here:**

- *`WorkerEvaluators`* lives in `include/azgo/inference_hub.hpp` / `src/inference_hub.cpp`. Its metal branch calls `makeMetalRunner(const Network&, int maxBatch)`, declared in `azgo/metal.hpp` and defined in `src/metal/metal_runner.mm`. In builds without Metal, `WorkerEvaluators` with `metal` throws the not-built message.
- *`workers = 0` with metal* resolves through a new `resolveWorkers(int workers, bool gpu)` overload: 2 × hardware threads. The old one-argument function is unchanged. Self-play caps at `games_per_generation` and the arena at `games`, as now.
- *`inference_batch`* is printed after `mcts_batch` in `configToString`. `checkConsistency` refuses a value below `mcts_batch`; the arena checks the per-side batches.
- *`lossFromOutputs`* is declared in `include/azgo/validation.hpp`.
- *The device line* is printed by `Trainer::startNew`, `Trainer::resume` and `rerateRun` when the backend is `metal`. CPU runs' output is unchanged. *The resume note* is printed when `backend` or `gpu_training` differs from the saved `config.cfg`.
- *Chaos mode* is reachable from the trainer only through a test hook: `metal_probe::setHubChaosSeed(uint64_t)` for Metal, and `InferenceHubOptions::chaosSeed` directly in the portable tests. No config key.
- *Failure injection:* `metal_probe::failCommandBuffersAfter(int n)`. After n more successful command buffers, the next one reports a failure through the normal error path. 0 disables it.

## File Structure

```
CMakeLists.txt                               A1 option, OBJCXX, embedding, SKIP_RETURN_CODE; A7 smoke arg
cmake/azgo_metal_kernels.inc.in              A1 raw-string template
include/azgo/metal.hpp                       A1 metalBuilt/Available/Description; probes A1-A7; makeMetalRunner A6
src/metal/metal_info.cpp                     A1 the three functions without Metal
src/metal/shader_types.h                     A1, grows each task
src/metal/kernels.metal                      A1 fill; A2 gemm_simd; A3 forward layer kernels; B1-B4 training kernels
src/metal/metal_context.hpp, .mm             A1 MetalContext, GpuBuffer, CommandBatch
src/metal/gpu_layers.hpp, .mm                A2 gemm; A3 forward layers; B1-B3 backward layers, losses
src/metal/gpu_network.hpp, .mm               A4 GpuWeights, GpuWorkspace(Inference), GpuNetwork forward; B4 training
src/metal/metal_runner.mm                    A6 MetalBatchRunner
src/metal/metal_probes.mm                    A1-A7, B1-B6
include/azgo/param_layout.hpp, src/param_layout.cpp      A4
include/azgo/inference_hub.hpp, src/inference_hub.cpp    A5 BatchRunner, InferenceHub, BatchingEvaluator; A6 WorkerEvaluators
include/azgo/validation.hpp, src/validation.cpp          A7 lossFromOutputs, GPU validation path; B5 batch-loss variant
include/azgo/parallel.hpp, src/parallel.cpp  A7 resolveWorkers(workers, gpu)
src/trainer.cpp, src/arena.cpp, include/azgo/arena.hpp   A7 WorkerEvaluators, cpuBackendName, inference batch
src/config.cpp, include/azgo/config.hpp      A7 backend = metal, inference_batch; B6 gpu_training
src/backend.cpp, include/azgo/backend.hpp    A7 messages, header comment
apps/arena.cpp                               A7 --backend metal, --inference-batch
apps/analyze.cpp, apps/gtp.cpp               A8
apps/bench.cpp                               A9 hub and raw GPU tables; B7 training
include/azgo/training_engine.hpp, src/training_engine.cpp   B5 seam, CpuTrainingEngine; B6 metal branch
src/metal/metal_engine.mm                    B6 MetalTrainingEngine
tests/test_metal.cpp                         A1-A8, B1-B6
tests/test_inference_hub.cpp                 A5, A6
tests/test_training_engine.cpp               B5, B6
tests/test_network.cpp                       A4 ParamLayout
tests/test_config.cpp, test_validation.cpp, test_trainer.cpp, test_arena.cpp, test_parallel.cpp   A7; B6
tests/integration_smoke.sh                   A7, A8; B6
README.md, docs/algorithm-explained.md, .html, docs/results/*   A9, A10; B7
```

## Conventions for Every Task

**Builds** (from the worktree root):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake -S . -B build-blas -DCMAKE_BUILD_TYPE=Release -DAZ_USE_BLAS=ON
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DAZ_USE_METAL=ON
cmake --build build-metal -j 4 && ctest --test-dir build-metal --output-on-failure -R test_metal
```

At the end of every code task, all three builds compile and pass `ctest`. `test_metal` is Skipped in `build` and `build-blas`. `ctest` does not build: run `cmake --build` first. A test that uses something not written yet fails at compile time, which is the expected "fails" state.

**TDD cycle:** write the listed tests; build and watch them fail; implement; pass; run all three full suites; commit.

**Commit command shape:**

```bash
git add <files>
git commit -F - <<'EOF'
<Imperative subject line>

<Why this change, what it enables, anything surprising.>

Co-Authored-By: Claude <model> <noreply@anthropic.com>
EOF
```

---

# Phase A — GPU inference and the batching evaluator

### Task A1: Build plumbing, the Metal context and a first kernel

**Effort:** ~0.5 day.

**Files:** Modify `CMakeLists.txt`. Create `cmake/azgo_metal_kernels.inc.in`, `include/azgo/metal.hpp`, `src/metal/metal_info.cpp`, `src/metal/shader_types.h`, `src/metal/kernels.metal`, `src/metal/metal_context.hpp`, `src/metal/metal_context.mm`, `src/metal/metal_probes.mm`, `tests/test_metal.cpp`.

**Produces:** `bool metalBuilt()`; `bool metalAvailable(std::string* why = nullptr)`, which creates the context on first call and, when false, gives a reason: not built (`not built: rebuild with -DAZ_USE_METAL=ON (macOS only)`), no device, GPU family below Apple 7, or a shader or pipeline compile failure. `std::string metalDescription()`. `metal_probe::fill(float*, size_t, float)`. Internal: `MetalContext::instance()`, `pipeline(name)`, `newBuffer(bytes)`, `GpuBuffer`, `CommandBatch`.

- [ ] **Step 1: CMake.** Add `option(AZ_USE_METAL …)` before the BLAS block. With it: `FATAL_ERROR` unless `APPLE`; force `AZ_USE_BLAS` on with a status message; `enable_language(OBJCXX)`; `CMAKE_OBJCXX_STANDARD 17`; add `src/metal/*.mm` with `-fobjc-arc`; link `Metal` and `Foundation`; define `AZ_USE_METAL` publicly. Always compile `src/metal/*.cpp`. Embedding: `file(READ)` `shader_types.h` + `kernels.metal` → `configure_file(… @ONLY)` with the template `R"AZGO_METAL(@AZGO_METAL_SOURCE@)AZGO_METAL"` into `${CMAKE_BINARY_DIR}/generated/`; add both sources to `CMAKE_CONFIGURE_DEPENDS`. After the test loop: `set_tests_properties(test_metal PROPERTIES SKIP_RETURN_CODE 77)`.
- [ ] **Step 2: Failing tests** (`tests/test_metal.cpp`). `main`: if `!metalBuilt()`, assert that `metalAvailable(&why)` is false and `why` contains `-DAZ_USE_METAL=ON`, print `test_metal: SKIPPED (<why>)` and return 77. If it is built but unavailable, skip the same way. Tests: `test_context_comes_up` (the description is non-empty, contains the device name, and is stable across calls); `test_fill_kernel` (n ∈ {0, 1, 255, 256, 257, 100000}, a sentinel past the end untouched); `test_fill_repeat_bitwise` (20 repeats).
- [ ] **Step 3: Implement**, starting from the spike's `Gpu`/`Ctx`. The pipeline table holds the fixed sizes, and creation checks `maxTotalThreadsPerThreadgroup`.
- [ ] **Step 4: Run** the three builds.

**Acceptance:** all builds pass; `build` reports `test_metal` Skipped and 29 tests in total; the default build compiles no `.mm` file; `grep -n "id<\|@interface\|#import" src/metal/*.hpp include/azgo/*.hpp` is empty. **CLT-only check:** on a Mac mini with only the Command Line Tools, `build-metal` builds and `test_metal` passes. If none is free, record it as an open item; it must pass before A7 merges.

**Commit:** `Add the Metal build option, context and a first GPU kernel`.

---

### Task A2: The forward gemm and its batch independence

**Effort:** ~1 day.

**Files:** Modify `shader_types.h`, `kernels.metal`, `metal_probes.mm`, `include/azgo/metal.hpp`. Create `src/metal/gpu_layers.hpp`, `src/metal/gpu_layers.mm`. Test `tests/test_metal.cpp`.

**Produces:** the kernel `gemm_simd`, the spike's 64×64 simdgroup tile with 128 threads and BK = 16: transA/transB, alpha, beta (C not read when beta = 0), the full K always in order, no split. `encodeGemm(CommandBatch&, tA, tB, M, N, K, alpha, BufView A, BufView B, beta, BufView C)`. Probe `metal_probe::gemm(...)` with `Backend::gemm` semantics.

- [ ] **Step 1: Failing tests.**
  - `test_gemm_against_double_reference`: all four trans combinations × shapes {(1,1,1), (7,63,65), (64,64,64), (65,129,17), (96, 81·256, 864) forward 6×96, (96, 81·256, 162) stem, (2, 81·256, 96) and (1, 81·256, 96) heads, (256, 82, 162) policy fc, (256, 64, 81) and (256, 1, 64) value fc} × alpha {1, 0.5} × beta {0, 1}. maxErr ≤ 1e-5.
  - `test_gemm_beta_zero_ignores_nan`.
  - `test_gemm_vs_blas_forward_shapes`: 5×5, 9×9 and 19×19 forward conv shapes at batches 1, 8 and 256 against BLAS, maxErr ≤ 2e-6. Print the bitwise-equal fraction (information only).
  - `test_gemm_repeat_bitwise`: 20 repeats.
  - `test_gemm_columns_and_rows_batch_independent`: for the 9×9 6×96 conv forward, the columns of samples {0, 1, 7, 100, 255} with N = 81·256 equal those computed with that sample alone (N = 81) and in N = 81·8 at offsets 0 and 5, bitwise. The same holds for rows of the Linear shapes with M = B, including M not a multiple of 64. A sample's columns placed after random companions give the same bits.
- [ ] **Step 2: Implement** (port the spike's kernel without the `partial` path).
- [ ] **Step 3: Run** the three builds.

**Acceptance:** all pass; the measured worst errors are recorded in the test comments; a comment in `kernels.metal` states the batch-independence property and why it holds.

**Commit:** `Add the deterministic, batch-independent simdgroup gemm`.

---

### Task A3: The forward layer kernels

**Effort:** ~1 day.

**Files:** Modify `shader_types.h`, `kernels.metal`, `gpu_layers.*`, `metal_probes.mm`, `include/azgo/metal.hpp`. Test `tests/test_metal.cpp`.

**Produces:** kernels `transpose_input`, `im2col3`, `bn_infer`, `relu_forward`, `add_into`, `policy_flatten`, `bias_add_rows`, `tanh_inplace`. Encoders `encodeTransposeInput`, `encodeIm2col3`, `encodeConvForward(k, cin, cout, w, x, B, n, col, y)` (im2col for k = 3, then the gemm), `encodeBnInfer`, `encodeRelu`, `encodeAddInto`, `encodePolicyFlatten`, `encodeLinearForward(in, out, w, b, x, B, y)`, `encodeTanh`. Probes for each, taking the CPU layer objects (`const Conv2d&`, `const BatchNorm2d&`, `const Linear&`).

- [ ] **Step 1: Failing tests.**
  - `test_copy_kernels_bitwise`: transpose, im2col (n ∈ {5, 9, 19}, C ∈ {1, 18, 96}, B ∈ {1, 2, 7}) and flatten against the CPU loops (`Backend::im2col`, and the loops of `Network::forward` written out in the test).
  - `test_conv_forward_vs_cpu`: 3×3 (18 → F, F → F) and 1×1 (F → 1, F → 2), F ∈ {8, 32, 96}, n ∈ {5, 9, 19}, B ∈ {1, 16, 256}; maxErr ≤ 2e-6.
  - `test_bn_infer_vs_cpu`: random gamma, beta and running statistics; maxErr ≤ 1e-6.
  - `test_relu_add_bitwise`; `test_tanh_vs_cpu` (maxErr ≤ 1e-6 over [−20, 20]).
  - `test_linear_forward_vs_cpu`: the head shapes for 5×5, 9×9 and 19×19 at B ∈ {1, 16, 256}; maxErr ≤ 2e-6.
  - `test_layer_kernels_batch_independent`: for each kernel, a sample alone against inside a batch of 256 gives bitwise equal output.
- [ ] **Step 2: Implement.**
- [ ] **Step 3: Run** the three builds.

**Acceptance:** all pass; measured errors recorded.

**Commit:** `Port the forward layers to Metal`.

---

### Task A4: The GPU inference network

**Effort:** ~1–1.5 days.

**Files:** Create `include/azgo/param_layout.hpp`, `src/param_layout.cpp`, `src/metal/gpu_network.hpp`, `src/metal/gpu_network.mm`. Modify `metal_probes.mm`, `include/azgo/metal.hpp`. Tests: `tests/test_network.cpp` (ParamLayout), `tests/test_metal.cpp`.

**Produces:**
- `struct ParamLayout { struct Entry { std::string name; size_t size, offset; bool decay; }; std::vector<Entry> params, buffers; size_t paramFloats, bufferFloats; static ParamLayout of(Network&); }`, with offsets aligned to 64 floats. Portable.
- `GpuWeights(const ParamLayout&)` with `upload(const Network&)`, which throws `std::logic_error` naming any unknown tensor.
- `GpuWorkspace(const NetConfig&, int maxBatch, Purpose::Inference)`: `stage.input`, transposed input, `col`, `convOut`, three rotating trunk buffers, head buffers, and the outputs `logits`, `values`, `ownership`; `bytes()`.
- `GpuNetwork(const NetConfig&)`: `encodeForward(CommandBatch&, GpuWorkspace&, const GpuWeights&, int B)` (inference mode, `Network::forward`'s order, the ownership head when present); `infer(GpuWorkspace&, const GpuWeights&, const float* input, int B, float* logits, float* values, float* ownership)`, synchronous.
- Probe class `metal_probe::InferNet(const Network&, int maxBatch)` with `infer(...)` and `bytes()`.

- [ ] **Step 1: Failing tests.**
  - `test_network.cpp` `test_param_layout_matches_params` (every build): names, sizes and decay flags in `params()`/`buffers()` order, with and without the ownership head and with 0 blocks; offsets aligned, no overlap.
  - `test_infer_vs_cpu`: 5×5 1×8, 9×9 2×16, 9×9 6×32, 9×9 0 blocks × 16, 9×9 2×16 + ownership, 19×19 2×16. Logits, values and ownership against `Network::infer` (BLAS); maxErr ≤ 1e-5.
  - `test_infer_batch_independent`: 9×9 6×32 + ownership. Positions {0, 1, 7, 100, 255} alone, in batches of 8, 144 and 256, at offsets 0 and the last slot, among random companions and permuted: bitwise identical logits, value and ownership.
  - `test_infer_repeat_bitwise` (20 passes) and `test_two_networks_bitwise` (two separately uploaded `InferNet`s).
  - `test_unknown_tensor_refused`.
  - `test_inference_workspace_size`: 9×9 6×96 at batch 256 is ~100 MB (within 15%).
- [ ] **Step 2: Implement** `encodeForward` by following `network.cpp`'s forward line by line, keeping its comments, with the inference-mode BN path only.
- [ ] **Step 3: Run** the three builds; run `test_metal` 3 times in a row (no flakiness).

**Acceptance:** all pass; a 10,000-pass loop in a scratch program (not committed) shows flat resident memory, which checks the autorelease pools.

**Commit:** `Run the network's forward pass on the GPU`.

---

### Task A5: The inference hub (portable)

**Effort:** ~1.5 days. Independent of A2–A4 (it needs only `Evaluator`), so it can run in parallel with them.

**Files:** Create `include/azgo/inference_hub.hpp`, `src/inference_hub.cpp`, `tests/test_inference_hub.cpp`. Modify `README.md` (test count).

**Produces:** `BatchRunner`, `InferenceHubOptions` (`maxBatch`, `chaosSeed`), `InferenceHub` (`makeEvaluator(maxRequest)`, `stats()`) and `BatchingEvaluator`, as in the spec. The dispatcher loop follows the spec's three steps: complete slots; submit while a slot is free and requests wait; whole requests only, FIFO, up to `maxBatch`. `kInferenceSlots = 2`. Chaos mode (nonzero seed, the hub's own `Rng`): it caps each batch at a random size between the next request and `maxBatch`, sleeps 0–200 µs before a random fraction of submissions, and shuffles request order within a batch.

The test file defines `CpuTestRunner : BatchRunner`. It runs `Network::infer` on a **naive** backend (batch-independent by construction) on a helper thread per slot and calls `done` from there, mimicking the GPU's completion thread.

- [ ] **Step 1: Failing tests** (`tests/test_inference_hub.cpp`, every build):
  - `test_hub_single_request`: one evaluator, one request, outputs equal `Network::infer` alone, bitwise.
  - `test_hub_outputs_equal_alone_under_chaos`: 1, 4 and 32 client threads, each issuing 200 requests of 1–8 random positions; `maxBatch` ∈ {8, 64, 512}; chaos off and on with three seeds. Every request's outputs equal its positions evaluated alone, bitwise. `stats().positions` equals the sum of the requests.
  - `test_hub_refuses_oversized_request`: `makeEvaluator(maxRequest > maxBatch)` throws `std::invalid_argument`.
  - `test_hub_failure_reaches_every_waiter`: the runner fails the third batch. Every request of that batch and every later request throws `std::runtime_error` containing the runner's text; no thread hangs (each test thread joins within a timeout).
  - `test_hub_shutdown`: destroying the hub with idle evaluators gone, after a burst, joins the dispatcher; creating and destroying 100 hubs leaks no thread.
  - `test_hub_stress`: 64 threads × 1,000 requests with chaos: no lost or duplicated results (each output row carries a marker derived from its position).
  - `test_selfplay_through_hub_equals_local`: 5×5 1×8, 8 games, `workers` 1 and 6, chaos on. `playSelfPlayGame` with hub evaluators (CpuTestRunner) gives `GameRecord`s identical to today's per-worker `LocalEvaluator` on the naive backend.
- [ ] **Step 2: Implement.** Keep the mutex for queue moves only. Copies in and out run outside the lock (the requests in flight belong to the dispatcher).
- [ ] **Step 3: Run** the three builds (`test_inference_hub` runs in all of them).

**Acceptance:** all pass under `-fsanitize=thread` too: a one-off `build-tsan` configuration with `-DCMAKE_CXX_FLAGS=-fsanitize=thread` running `test_inference_hub`, not committed. The CTest total is 30.

**Commit:** `Add a batching evaluator that gathers leaves from every worker`. The body gives the flush policy and the determinism argument in brief.

---

### Task A6: The Metal batch runner and `WorkerEvaluators`

**Effort:** ~0.5–1 day.

**Files:** Create `src/metal/metal_runner.mm`. Modify `include/azgo/metal.hpp` (`makeMetalRunner`), `include/azgo/inference_hub.hpp`, `src/inference_hub.cpp` (`WorkerEvaluators`), `metal_probes.mm`. Tests: `tests/test_metal.cpp`, `tests/test_inference_hub.cpp`.

**Produces:**
- `std::unique_ptr<BatchRunner> makeMetalRunner(const Network&, int maxBatch)`: uploads `GpuWeights` once; holds two `GpuWorkspace`s (one per slot); `begin` encodes the forward pass and commits with a completion handler. It checks GPU memory against `recommendedMaxWorkingSetSize` (`metal: inference needs N MB …; lower inference_batch`). The destructor waits for in-flight slots.
- `WorkerEvaluators(net, backend, workers, maxRequest, inferenceBatch)`: for naive/blas, per-worker backends and `LocalEvaluator`s (today's code); for metal, one hub with the Metal runner. `forWorker(w)`.

- [ ] **Step 1: Failing tests.**
  - `test_metal.cpp` `test_metal_hub_outputs_equal_alone_under_chaos`: the A5 test with the Metal runner (reference: the same GPU network alone).
  - `test_metal.cpp` `test_metal_hub_vs_cpu`: hub outputs against `Network::infer` (BLAS); maxErr ≤ 1e-5.
  - `test_metal.cpp` `test_metal_runner_memory_check`: an oversized `maxBatch` for 19×19 10×128 throws the message.
  - `test_inference_hub.cpp` `test_worker_evaluators_cpu_is_local`: with naive and blas, the evaluators are `LocalEvaluator`s with the same outputs as before (bitwise). With metal in a build without Metal, it throws the not-built message.
- [ ] **Step 2: Implement.**
- [ ] **Step 3: Run** the three builds.

**Acceptance:** all pass. A scratch measurement (not committed): 36 threads × batch-8 requests through the Metal hub on 9×9 6×96 exceed 20k evals/s, a first sign that the A9 target is reachable. Report the number in the commit body.

**Commit:** `Serve batched inference from the GPU through the hub`.

---

### Task A7: `backend = metal` for self-play, the arena and the validation loss

**Effort:** ~1.5–2 days.

**Files:** Modify `src/config.cpp`, `include/azgo/config.hpp`, `include/azgo/parallel.hpp`, `src/parallel.cpp`, `src/backend.cpp`, `include/azgo/backend.hpp`, `include/azgo/arena.hpp`, `src/arena.cpp`, `src/trainer.cpp`, `include/azgo/validation.hpp`, `src/validation.cpp`, `apps/arena.cpp`, `CMakeLists.txt`, `tests/integration_smoke.sh`. Tests: `tests/test_config.cpp`, `test_parallel.cpp`, `test_validation.cpp`, `test_arena.cpp`, `test_trainer.cpp`, `test_metal.cpp`.

**Produces:**
- Config: `backend` accepts `metal` (`expected naive|blas|metal`); `inference_batch` (1–1024, default 256, tunable; `checkConsistency`: ≥ `mcts_batch`); `Config::effectiveWorkers()` uses `resolveWorkers(workers, backend == "metal")`.
- `std::string cpuBackendName(const std::string&)` and `void checkBackendAvailable(const std::string&)` (the spec's messages); `makeBackend("metal")` throws the per-operation message; the unknown-backend message names all three.
- Self-play: `WorkerEvaluators` in place of the per-worker backends and `LocalEvaluator`s. `playMatch` gains an `inferenceBatch` parameter (default 256) and builds `WorkerEvaluators` per network player. `Seat` takes an `Evaluator&`.
- Validation: `LossStats lossFromOutputs(const NetConfig&, logits, values, ownership, pi, z, targets, B)`. With `metal`, `heldOutLoss` runs on a `GpuNetwork` (one workspace at `batch_size`) plus `lossFromOutputs`. The CPU path is unchanged.
- Trainer: `checkBackendAvailable` in `startNew`, `resume` and `rerateRun` before anything is written; the device line; the resume note; training with `makeBackend(cpuBackendName(...))`; GPU failures surfacing as the spec's error.
- `arena` CLI: `--backend metal`, `--inference-batch N`, `--workers 0` as the config.
- Smoke: a third argument `metal=ON|OFF`.

- [ ] **Step 1: Failing tests.**
  - `test_config`: `metal` parses, prints and round-trips; `gpu` fails; `inference_batch` range and the `mcts_batch` rule; `effectiveWorkers()` is 2 × cores with metal and 1 × otherwise.
  - `test_parallel`: `resolveWorkers(0, true)` is 2 × hardware threads; explicit values unchanged.
  - `test_validation`: `lossFromOutputs` on `Network::infer`'s outputs equals `Network::inferLoss` bitwise (5×5 and 9×9, with ownership, a partial batch).
  - `test_arena` (every build): `playMatch` through `WorkerEvaluators` with naive and blas gives the same results as before. Compare with values recorded from main's code in the test, or with an independent per-worker `LocalEvaluator` loop.
  - `test_trainer` (every build): `test_metal_unavailable_writes_nothing` (no Metal build: `startNew` with `metal` throws; no `state.bin`, `config.cfg` or checkpoint); `test_backend_switch_prints_note` (naive → blas on resume prints the note).
  - `test_metal.cpp`, tiny 5×5 config with `backend = metal`:
    - `test_metal_run_independent_of_workers_and_batching`: 2 generations with `workers` 1, 3 and 8, `inference_batch` 8 and 256, and chaos on (`setHubChaosSeed`). The checkpoints, `state.bin`, `results.csv`, `references.csv`, SGF and `log.csv` loss columns are identical.
    - `test_metal_resume_is_byte_identical`: 4 generations against 2 + resume to 4, including the validation columns.
    - `test_metal_runs_are_reproducible`: two fresh runs.
    - `test_metal_rerate_reproduces_live`: `rerateRun` matches the live `results.csv` and ratings.
    - `test_gpu_failure_keeps_last_good_state`: `failCommandBuffersAfter(n)` during generation 2's self-play. `runGeneration` throws `GPU inference failed: …`; `state.bin` still says generation 1; a resume reproduces the uninterrupted generation 2 byte for byte.
    - `test_metal_arena_matches`: network against network, against a bot, with random symmetry and different per-side batches. Identical results for `--workers` 1 and 5.
    - `test_metal_validation_vs_cpu`: GPU validation `LossStats` within 1e-5 relative of the CPU's.
  - Smoke ON: the `TINY` run with `--set backend=metal` (2 generations, the device line printed, resume to 3, `--rerate`), and `arena --backend metal` between two checkpoints and against a bot. OFF: `train … --set backend=metal` and `arena --backend metal` fail with the not-built message, and no `state.bin` is written.
- [ ] **Step 2: Implement.** Self-play and the arena change only in how their evaluators are made.
- [ ] **Step 3: Run** all three builds' full suites, including `integration_smoke`.
- [ ] **Step 4: Old against new binary.** Main's `train` against this branch's, both BLAS, on the shrunk 5×5 preset (naive and BLAS) and a 9×9 6×64 BLAS run with `train_steps = 20` and `games_per_generation = 8`, 3 generations each, plus `arena` on two checkpoints. `cmp` the checkpoints, `state.bin`, `results.csv` and the arena output. Use `-j 4` and small runs.

**Acceptance:** all pass; the old-vs-new comparison is byte-identical (record it in the commit body); a manual 9×9 run with `--set backend=metal --set games_per_generation=16 --generations 2` prints the device line and completes.

**Commit:** `Run self-play, the arena and the validation loss on the GPU with backend = metal`.

---

### Task A8: `analyze` and `gtp` on the GPU (optional)

**Effort:** ~0.5 day.

**Files:** Modify `apps/analyze.cpp`, `apps/gtp.cpp`, `tests/integration_smoke.sh`. Test `tests/test_metal.cpp`.

- [ ] **Step 1: Failing tests.** `test_metal.cpp` `test_single_search_hub_matches_alone`: one search through a one-evaluator hub gives the same visit counts as through the GPU network called directly (bitwise). Smoke ON: `analyze --backend metal` on a 6-move line prints a value and a policy, and the ownership map for a checkpoint with the head; a scripted `gtp --backend metal` session plays a move. OFF: both fail with the not-built message.
- [ ] **Step 2: Implement.** Searches go through `WorkerEvaluators(…, workers = 1, …)`. `analyze`'s raw network lines go through `GpuNetwork::infer`.

**Acceptance:** all pass. `analyze --backend metal` output matches `--backend blas` in every printed digit for the README's example. If a last digit differs, document it in the README.

**Commit:** `Let analyze and gtp search on the GPU`.

---

### Task A9: Benchmark and measurements (Phase A)

**Effort:** ~0.5 day, plus about an hour of machine time with the jobs paused (Mark's go-ahead).

**Files:** Modify `apps/bench.cpp`. Results in `docs/results/bench.txt` and `docs/results/README.md`.

- [ ] **Step 1: Bench.** `--backend metal` (and `all` in a Metal build). The main table's metal row gives batch-8 GPU evals/s on one thread and `-` for the train step, which runs on the CPU in Phase A. New tables: raw GPU forward at batches 8, 144 and 256, synchronous and two-slot pipelined; the hub under load with `--hub-workers 18,36,72` threads of batch-8 requests: evals/s, mean batch, GPU busy fraction. The 18-worker CPU figure is printed for reference. Usage text updated. Smoke ON: `bench --sizes 5 --seconds 0.1 --backend metal`.
- [ ] **Step 2: Measure** (jobs paused). `bench --sizes 9 --backend all` with 6×64 and 6×96, and `--sizes 19`. Append to `docs/results/bench.txt` with the machine, date and conditions.
- [ ] **Step 3: Real run.** One generation of the 6×96 9×9 preset variant with `blas` and with `metal`, same seed, into the scratchpad. Compare `selfplay_s`, `arena_s`, `train_s`, the loss columns (within 2% relative) and the arena scores (within noise). Record them in `docs/results/README.md`.
- [ ] **Step 4: Acceptance against the spec**: hub ≥ 25k evals/s at 36 workers on 6×96; self-play ≥ 1.6× faster; arena ≥ 1.3×; generation ≥ 1.4×. If one misses, stop and report the numbers to Mark with the spec's levers; do not optimize inside this task.

**Commit:** `Benchmark GPU inference and the batching evaluator`.

---

### Task A10: Documentation and the explainer (Phase A)

**Effort:** ~1 day.

**Files:** Modify `README.md`, `docs/algorithm-explained.md`, `docs/algorithm-explained.html`, `docs/results/README.md`, `include/azgo/evaluator.hpp` (its "A future GPU evaluator…" comment), `include/azgo/backend.hpp` (its GPU sentence), and possibly `tools/check_citations.py`.

- [ ] **Step 1: README.** Build: `AZ_USE_METAL`, its requirements, the test count. Usage: a **Backends** paragraph after **Threads** (what `metal` does in Phase A, the determinism contract and its same-machine limit, `workers = 0` with metal, `inference_batch`, the switching note, the device line). "Extending to the GPU": rewritten with A9's numbers and Phase B as next. "Learn how it works": this spec and plan.
- [ ] **Step 2: Explainer.** Rewrite chapter 03's "This is where a GPU would plug in": workers block in `evaluate`, the dispatcher packs leaves from every game, the greedy flush, the two slots, and the batch-independence argument for why timing cannot change a game. Quote the dispatcher loop and the gemm tile (the scalar `gemm_tiled` from `tools/metal_bench`, then `gemm_simd`), with `file:line` citations. Give the measured results.
- [ ] **Step 3: HTML.** The same text, plus one diagram (workers → queue → two slots → GPU → back) in the existing style, following the light/dark theme.
- [ ] **Step 4: Check.** `python3 tools/check_citations.py` passes; extend it to `.mm`/`.metal` if needed. Re-read the opening comments of `evaluator.hpp`, `backend.hpp`, `mcts.hpp` and `trainer.hpp`.

**Acceptance:** citations pass; the README's numbers match `docs/results/`; nothing claims CPU–GPU bit-identity.

**Commit:** `Explain GPU inference and the batching evaluator`.

---

# Phase B — GPU training

Phase B reuses from Phase A, unchanged: `MetalContext`, the build and embedding, `shader_types.h`, `gemm_simd`, `transpose_input`, `im2col3`, `relu_forward`, `add_into`, `policy_flatten`, `bias_add_rows`, `tanh_inplace`, `ParamLayout`, `GpuWeights`, `GpuWorkspace` (extended with `Purpose::Training`), `GpuNetwork::encodeForward` (extended with `Mode::Train`/`Probe`), `lossFromOutputs` and the GPU validation path, the probes, the test skip logic, failure injection, the device line and the resume note. Phase B adds the backward kernels, the training workspace, the optimizer state, the engine seam and the `gpu_training` key.

### Task B1: Split-K, col2im and convolution backward

**Effort:** ~1 day.

**Files:** `shader_types.h`, `kernels.metal`, `gpu_layers.*`, `metal_probes.mm`, `include/azgo/metal.hpp`; test `tests/test_metal.cpp`.

**Produces:** `gemm_simd`'s `partial` path and `splitk_reduce`; `int gemmSplitCount(M, N, K)` (the spec's rule, a pure function); `encodeGemm` gains a split-K scratch argument used for dW only; `col2im3` (gather, CPU order); `encodeConvBackward(k, cin, cout, w, x, col, dy, B, n, dcol, gw, dx /*optional*/, scratch)`, which writes gw with beta 0. Probes `gemmSplitCount`, `col2im3`, `convBackward(const Conv2d&, …)`.

- [ ] **Tests first:** `test_split_rule` (`(96, 864, 20736) → 20`, `(64, 576, 20736) → 20`, `(1, 96, 20736) → 20`, `(128, 1152, 92416) → 64`, `(96, 864, 4095) → 1`, `(1024, 1024, 20736) → 1`, and kChunk/S as documented); `test_splitk_gemm_vs_double` (maxErr ≤ 1e-5); `test_splitk_repeat_bitwise`; `test_col2im_bitwise` (against `Backend::col2im`); `test_conv_backward_vs_cpu` (3×3 and 1×1, gw ≤ 1e-5, dx ≤ 2e-6, null dx leaves the output untouched); `test_conv_backward_repeat_bitwise`. Also re-run A2's forward batch-independence tests: forward products must not take the split path.
- [ ] Implement; run the three builds.

**Acceptance:** all pass; measured errors recorded. **Commit:** `Add split-K weight gradients and convolution backward on Metal`.

### Task B2: Batch norm training and backward, ReLU backward

**Effort:** ~1 day.

**Produces:** `bn_train_stats` (two passes, fixed tree, running update), `bn_train_apply`, `bn_backward_sums`, `bn_backward_apply`, `relu_backward`, `fill_zero`; encoders and probes taking `BatchNorm2d&`.

- [ ] **Tests first:** `test_bn_forward_train_vs_cpu` (C ∈ {1, 2, 32, 96}, M ∈ {2, 25, 567, 20736, 5776}; y, xhat, invStd ≤ 1e-5; running statistics ≤ 1e-6; no update leaves them bitwise unchanged); `test_bn_two_pass_accuracy` (mean 1000, standard deviation 0.01); `test_bn_backward_vs_cpu` (≤ 1e-4, including dx aliasing dy); `test_relu_backward_bitwise`; `test_reductions_repeat_bitwise`.
- [ ] Implement. If BN backward exceeds 1e-4, add Kahan summation to the strided phase and document it. Run the three builds.

**Acceptance:** all pass; the tree shape documented in `kernels.metal`. **Commit:** `Port batch norm training and backward to Metal`.

### Task B3: Heads backward and the loss kernels

**Effort:** ~1 day.

**Produces:** `policy_unflatten`, `bias_grad_rows`, `encodeLinearBackward`, `policy_value_loss`, `ownership_loss`, `encodeLoss(…, withGradients, …)`, `LossStats sumLossTerms(terms, B, hasOwnership, area)`; probes `linearBackward`, `loss`.

- [ ] **Tests first:** `test_linear_backward_vs_cpu` (gw ≤ 1e-5, **gb bitwise**, dx ≤ 1e-5); `test_unflatten_bitwise`; `test_policy_value_loss_vs_reference` (a double reference in the test; ±80 logits stay finite; terms, dLogits and dValues within 1e-6); `test_ownership_loss_vs_reference` (w ∈ {0, 1.5}); `test_loss_without_gradients_writes_none`.
- [ ] Implement; run the three builds.

**Acceptance:** all pass. **Commit:** `Port the heads' backward and the losses to Metal`.

### Task B4: The whole training step on the GPU

**Effort:** ~2 days.

**Produces:** `GpuWorkspace(…, Purpose::Training)` (every `Workspace` field plus `lossTerms`, `logZ`, `bnScratch`, `splitK`); `GpuOptimizerState` (`grads` and `velocity` arenas, `uploadVelocity`/`downloadVelocity`); `GpuWeights::download`; `sgd_update` and `encodeSgd` per tensor; `GpuNetwork` `Mode::Train`/`Probe`, `encodeLoss`, `encodeBackward`, `trainStep` (one command buffer: the spec's sequence), `gradients` (no SGD, for tests); probe class `metal_probe::TrainNet`.

- [ ] **Tests first** (configurations as A4, plus ownership with w = 0 and without targets): `test_probe_loss_vs_cpu` (`Network::computeLoss`, 1e-5); `test_train_forward_vs_cpu` (running statistics ≤ 1e-6; outputs ≤ 1e-4); `test_train_step_gradients_vs_cpu` (`LossStats` 1e-5; each gradient normErr 1e-4, maxErr 1e-3; with w = 0 the head's grads are exactly 0); `test_five_steps_vs_cpu` (losses 1e-4; weights normErr 1e-5; momentum 1e-3; BN statistics 1e-5; with w = 0 the head's weights still decay); `test_step_repeat_bitwise`; `test_two_trainers_bitwise`; `test_load_store_round_trip`; `test_training_workspace_size` (~390 MB for 9×9 6×96 at batch 256).
- [ ] Implement by following `network.cpp`'s forward, loss and backward and `optimizer.cpp` line by line. Run the three builds; run `test_metal` 3 times.

**Acceptance:** all pass within the ceilings; a 1,000-step scratch loop shows flat resident memory. **Commit:** `Run the whole training step on the GPU`.

### Task B5: The training-engine seam (CPU only)

**Effort:** ~1 day. It needs only A7, and can run in parallel with B1–B4.

**Files:** create `include/azgo/training_engine.hpp`, `src/training_engine.cpp`, `tests/test_training_engine.cpp`; modify `include/azgo/validation.hpp`, `src/validation.cpp` (`heldOutLoss` with a `BatchLossFn`, the old signature wrapping it), `src/trainer.cpp`, `README.md` (test count).

**Produces:** `TrainingEngine`, `TrainingEngineOptions`, `makeTrainingEngine` (naive/blas, and metal with `gpu_training = false`: CPU training with Phase A's GPU validation path), `CpuTrainingEngine`. The trainer's validation and training block goes through the engine.

- [ ] **Tests first:** `test_cpu_engine_equals_direct_calls` (naive and BLAS, 5×5 and 9×9, with and without the head, pools of 1 and 3, three steps, everything bitwise); `test_cpu_engine_infer_loss_equals_network`; `test_held_out_loss_variant_equals_old` (bitwise, including a partial batch).
- [ ] Implement; run the three builds. All existing trainer byte-identity tests pass unchanged, including A7's metal tests (Phase A bits unchanged).
- [ ] **Old against new binary** (as in A7, Step 4): byte-identical.

**Acceptance:** all pass; the comparison recorded in the commit body; the CTest total is 31. **Commit:** `Train through a TrainingEngine seam, the CPU path unchanged`.

### Task B6: `gpu_training = true`

**Effort:** ~1–1.5 days.

**Files:** create `src/metal/metal_engine.mm`; modify `include/azgo/training_engine.hpp`, `src/training_engine.cpp`, `src/config.cpp`, `include/azgo/config.hpp`, `src/trainer.cpp`, `metal_probes.mm`, `tests/integration_smoke.sh`; tests `test_config.cpp`, `test_training_engine.cpp`, `test_metal.cpp`.

**Produces:** `MetalTrainingEngine` (built on `TrainNet`'s internals: `load` uploads weights, BN statistics and momentum and zeroes grads; `inferLoss` is Phase A's path on resident weights; `trainStep` is one command buffer; `store` downloads; memory check with `lower batch_size`); config `gpu_training` (bool, default false, tunable; `ConfigError` unless `backend = metal`); the startup line `(inference: …, training)`; the resume note covering `gpu_training`; the training failure message `GPU command buffer failed at generation g step s: …; last good state kept`.

- [ ] **Tests first:** `test_config` (`gpu_training` parses, round-trips, is refused without metal); `test_metal_engine_vs_cpu_engine` (within B4's ceilings, round trip bitwise); with `backend = metal`, `gpu_training = true` on the tiny config: `test_gpu_training_resume_is_byte_identical`, `test_gpu_training_independent_of_workers_and_threads` (`workers` 1/3, `train_threads` 1/4, chaos on), `test_gpu_training_reproducible`, `test_gpu_training_failure_keeps_last_good_state`, `test_gpu_training_switch_prints_note`. Smoke ON: the `TINY` metal run resumed with `--set gpu_training=true`.
- [ ] Implement; run all three builds' full suites.

**Acceptance:** all pass; a manual 9×9 run with `gpu_training=true` completes, and its checkpoints load in `arena --backend blas`. **Commit:** `Train on the GPU with gpu_training = true`.

### Task B7: Benchmark and documentation (Phase B)

**Effort:** ~1 day, plus machine time with the jobs paused.

- [ ] `bench --backend metal --gpu-training`: step wall and GPU ms, `load`/`store` ms, at `--train-batch`. Measure 6×64, 6×96 and 19×19 10×128. Real run: one 6×96 generation with `gpu_training` off and on; `train_s` and the losses within 2%. Acceptance: the spec's Phase B table (6×96 step ≤ 45 ms, phase ≤ 20 s). On a miss, report to Mark with the levers.
- [ ] README: `gpu_training`, Phase B numbers, test count. Explainer: "Training on the GPU" in chapter 08 after the training-threads passage (why the seam moved above `Network`, residency and the crossings, gather col2im, fixed split-K, no `double`, measured tolerance), with an HTML timeline diagram (load → validation → step command buffers → store). Citations pass.

**Commit:** `Benchmark and explain GPU training`.

---

## Effort Summary

| task | what | effort |
|---|---|---|
| A1 | build, context, embedded shaders, first kernel, skip | 0.5 d |
| A2 | forward gemm, batch independence | 1 d |
| A3 | forward layer kernels | 1 d |
| A4 | GPU inference network | 1–1.5 d |
| A5 | inference hub (portable), chaos, self-play equality | 1.5 d |
| A6 | Metal runner, `WorkerEvaluators` | 0.5–1 d |
| A7 | `backend = metal` in self-play, arena, validation, config, CLI | 1.5–2 d |
| A8 | analyze and gtp (optional) | 0.5 d |
| A9 | bench and measurements | 0.5 d + machine time |
| A10 | docs and explainer | 1 d |
| | **Phase A** | **~9–10.5 days** |
| B1 | split-K, col2im, conv backward | 1 d |
| B2 | BN training and backward | 1 d |
| B3 | heads backward, losses | 1 d |
| B4 | whole training step | 2 d |
| B5 | engine seam (CPU) | 1 d |
| B6 | `gpu_training` | 1–1.5 d |
| B7 | bench and docs | 1 d |
| | **Phase B** | **~8–8.5 days** |

Order: A1 → A2 → A3 → A4. A5 can run in parallel with A2–A4. A6 needs A4 and A5; A7 needs A6; then A8, A9 and A10. Phase B: B1 → B2 → B3 → B4. B5 can run in parallel with B1–B4 once A7 is in. B6 needs B4 and B5; B7 is last. Phase A is complete and shippable on its own.
