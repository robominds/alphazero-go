# Metal Training Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `backend = metal` runs the training phase (the validation loss and every SGD step: forward, loss, backward, momentum and weight-decay update, BN running statistics) on the Apple GPU with hand-written, deterministic Metal kernels, while self-play and the arena stay on the CPU (BLAS). The arithmetic is within a specified tolerance of the CPU, reproducible bit for bit run to run, and byte-identical on resume.
**Architecture:** A new seam above `Network`: `TrainingEngine` (`load` → `inferLoss`/`trainStep` → `store`), with a `CpuTrainingEngine` that wraps today's code unchanged and a `MetalTrainingEngine` that keeps weights, momentum and activations in shared GPU buffers for the phase. Kernels mirror the CPU layers one for one (`layers.cpp` ↔ `gpu_layers.mm`, `network.cpp` ↔ `gpu_network.mm`). A step is one command buffer. The Metal source is embedded at build time and compiled at run time.
**Tech Stack:** C++17, CMake ≥ 3.16, Objective-C++ (ARC) confined to `src/metal/*.mm`, the Metal and Foundation frameworks (no MPS), Accelerate (BLAS), assert-based test executables run by CTest, bash for the smoke test.
**Spec:** docs/superpowers/specs/2026-10-09-metal-training-backend-design.md
**Author:** Mark Castelluccio, with planning assistance from Claude (Anthropic).

## Global Constraints

- C++17, CMake ≥ 3.16, default build type Release. `AZ_USE_METAL` defaults to OFF. It is macOS-only (`FATAL_ERROR` elsewhere) and turns `AZ_USE_BLAS` on. A build without it compiles no Objective-C and links no Metal.
- Namespace `azgo`. Public headers in `include/azgo/` are portable C++. Objective-C++ appears only in `src/metal/*.mm`, and no header contains an Objective-C type: Metal objects sit behind pimpl structs.
- **No MPS, MPSGraph, MLX or Core ML.** Kernels are hand-written in `src/metal/kernels.metal` and compiled at run time with `newLibraryWithSource` and `MTLMathModeSafe`. No `metal`/`metallib` tool at build time: the Mac minis have only the Command Line Tools.
- **GPU determinism rules** (every kernel, every task): no atomics. Each output is written by exactly one thread. Reductions run within one threadgroup in a fixed tree, or in a separate pass in a fixed order. Threadgroup sizes are constants in `shader_types.h`, never derived from the device. Dispatch geometry depends only on the shapes. Never read uninitialized memory (gemm with beta 0 does not read C, and buffers are zero-filled at allocation). Transcendentals are `precise::`.
- **CPU runs stay byte-identical.** Every change to CPU code paths must leave `naive` and `blas` runs bit for bit what they were. The existing byte-identity tests must pass unchanged, and Task 7 adds the old-vs-new binary comparison.
- **The algorithm does not change** (AlphaGo Zero's loss, BN, SGD). No new hyperparameter.
- Tests: assert-based executables, compiled with `-UNDEBUG`. All GPU tests live in `tests/test_metal.cpp`, which returns **77** when Metal is not built or not usable. CMake sets `SKIP_RETURN_CODE 77` on it, so CTest reports Skipped. Tolerances are the spec's ceilings: never raise one to make a test pass without a documented cause in the test's comment.
- All randomness in tests through `azgo::Rng` with fixed seeds.
- The machine runs training jobs in `runs/`. Never touch that directory. GPU tests may run beside the jobs. CPU or GPU **timings** (Task 9) need the jobs paused with SIGSTOP/SIGCONT, as the spike did, and only with Mark's go-ahead. Use `cmake --build … -j 4` rather than an unbounded `-j` while jobs run.
- Each changed header keeps its opening teaching comment accurate. New files open with a teaching comment and the author line `Mark Castelluccio, with assistance from Claude (Anthropic).`
- Code style: camelCase functions, PascalCase types, `trailing_` members, `kConstants`, 4-space indent, ≤ 100 columns as in the surrounding code. Metal kernels: snake_case names, as in the spike.
- Commits: author Mark Castelluccio <markacastelluccio@gmail.com> (repo-local config, already set); imperative subject without a conventional-commit prefix; body explains why; message ends with `Co-Authored-By: Claude <model> <noreply@anthropic.com>`, naming the model that writes the commit.

## Review Focus

1. **Determinism machinery.** Fixed threadgroup sizes checked at pipeline creation, the split-K rule as a pure function of the shape, no atomics, a gather `col2im`. A slip here gives runs that are almost reproducible, and no single test catches that. Tests: Task 2 `test_gemm_repeat_bitwise`, `test_split_rule`; Task 4 `test_reductions_repeat_bitwise`; Task 6 `test_step_repeat_bitwise`, `test_two_engines_bitwise`; Task 8 `test_metal_resume_is_byte_identical`.
2. **Backward kernels.** A wrong factor still trains, just badly. Tests: the per-layer backward tests of Tasks 3–5, and Task 6 `test_train_step_gradients_vs_cpu` across all configurations (ownership on/off, w = 0, 0 blocks). Task 9 adds the real-run loss comparison.
3. **What crosses between CPU and GPU, and when.** The engine must copy every parameter, BN statistic and momentum tensor in at `load` and out at `store`, and nothing else may outlive a phase. Tests: Task 6 `test_load_store_round_trip`; Task 8 `test_metal_resume_is_byte_identical`, `test_failed_command_buffer_keeps_last_good_state`.
4. **The CPU path after the seam refactor.** `CpuTrainingEngine` must be today's code, same calls, same order. Tests: Task 7 `test_cpu_engine_equals_direct_calls` and the old-vs-new binary comparison.
5. **Builds without Metal.** They must compile no Objective-C, report `test_metal` as Skipped, and fail `backend = metal` with the documented message before writing anything. Tests: Task 1 acceptance (the default build); Task 8 `test_metal_unavailable_writes_nothing` and the smoke test's OFF branch.
6. **The tolerance ceilings.** Each kernel test records the measured error over 20 seeds in its comment, and the ceiling stays the spec's.

---

## Notes Before You Start

**Read first:** the spec; `docs/results/metal-bench.txt`; `tools/metal_bench/metal_bench.mm`, whose `gemm_simd`, `splitk_reduce`, `im2col3`, `col2im3` and `channel_stats` kernels, `Gpu` struct and `encodeHandGemm` are the starting point for Tasks 1–4; `src/layers.cpp`, `src/network.cpp` and `src/optimizer.cpp`, which the GPU code mirrors.

**Decisions binding on every task:**

- *Engine lifetime.* An engine lives for one training phase: created after the pool, before the validation loss, destroyed after `store`. `MetalContext` (device, queue, library, pipelines) is a process-wide lazily created singleton (`std::call_once`), and it holds no run state.
- *`makeBackend("metal")`* throws `std::invalid_argument("backend 'metal' runs only the training phase for now; use --backend blas")`. The unknown-backend message becomes `unknown backend 'X' (expected naive, blas or metal)`. The trainer always calls `makeBackend(cpuBackendName(cfg.backend), …)`.
- *Shared definitions.* `src/metal/shader_types.h` holds every kernel parameter struct (`GemmParams`, `ConvParams`, `BnParams`, `LossParams`, `SgdParams`, …) and the threadgroup-size constants (`kGemmThreads = 128`, `kReduceThreads = 256`, `kElementThreads = 256`), using `uint32_t`/`float` through a tiny `#ifdef __METAL_VERSION__` typedef block. CMake prepends it to the embedded shader string, and the `.mm` files `#include` it.
- *Pipelines.* `MetalContext` creates every pipeline at start from one table `{kernel name, threadgroup size}`. A pipeline whose `maxTotalThreadsPerThreadgroup` is below its size is an error naming the kernel.
- *Buffers.* `GpuBuffer` is a move-only RAII wrapper (pimpl around `id<MTLBuffer>`), always `MTLResourceStorageModeShared`, zero-filled at allocation, exposing `float* data()` and `size_t bytes()`. Sub-allocations in an arena are `(GpuBuffer*, byte offset)` pairs, with offsets aligned to 256 bytes.
- *Encoding.* `CommandBatch` wraps one command buffer and one serial compute encoder. `commitAndWait()` throws `std::runtime_error` with the command buffer's error text when the status is not `Completed`. Every public entry point of a `.mm` file is wrapped in `@autoreleasepool`.
- *Test probes.* `include/azgo/metal.hpp` declares, under `#ifdef AZ_USE_METAL`, a namespace `azgo::metal_probe` of host-array functions: copy in, one `CommandBatch`, copy out. Tests and `bench` use them. They are part of the library (`src/metal/metal_probes.mm`), but no production path calls them.
- *Tolerance metrics* (in `tests/test_metal.cpp`): `maxErr(x, ref)` = max|x − ref| / max|ref|; `normErr` = ‖x − ref‖₂ / ‖ref‖₂; `bitwise` = `memcmp`. Each test prints its measured worst error once (`test_metal: conv dW 9x9 6x96 B=256 maxErr 4.1e-7 (ceiling 1e-5)`), so a run of the test is also a measurement.
- *Test sizes.* Keep `test_metal` under ~30 s on an M-series Mac. Use 6×96 at batch 256 only for single-layer tests, and smaller networks (5×5 1×8, 9×9 2×16, 9×9 6×32) for whole-network tests.
- *The reference backend* for GPU-vs-CPU tests is `makeBackend("blas")`. A Metal build always has BLAS.

**Spec ambiguities resolved here:**

- *Layer encode functions* live in `src/metal/gpu_layers.mm` (`encodeConvForward`, `encodeConvBackward`, `encodeBnTrain`, `encodeBnInfer`, `encodeBnBackward`, `encodeLinearForward`, `encodeLinearBackward`, `encodeRelu…`, `encodeAddInto`, `encodeTanh`), each taking a `CommandBatch&`, buffer views and shapes. `gpu_network.mm` composes them in `network.cpp`'s order.
- *Probing mode.* `GpuNetwork` supports `Mode::{Infer, Train, Probe}` like `Network::forward`. `Probe` (batch statistics, running statistics untouched) exists so tests can compare with `Network::computeLoss`.
- *Gradients in a training step* are written with beta 0. The ownership head's grads are zero-filled when the head gets no gradient. `GpuNetwork` never accumulates across steps.
- *Loss sums.* `sumLossTerms(terms, B, hasOwnership, area)` adds the per-sample terms on the CPU in `double`, in sample order, and divides as `Network::loss` does: policy and value by B, ownership by B·A, where each sample's ownership term is its sum over points.
- *Validation batches* use the engine's `maxBatch` (= `batch_size`), as today.
- *The device line* is printed by `Trainer::startNew` and `Trainer::resume` only when the backend is `metal`. CPU runs' output is unchanged. `rerateRun` plays only the arena: it maps the backend with `cpuBackendName` and needs no GPU.
- *The resume note* is printed whenever the saved `config.cfg`'s `backend` differs from the effective one (any pair).
- *`bench` metal rows* follow the spec's Performance section.

## File Structure

```
CMakeLists.txt                                   T1 option, OBJCXX, shader embedding, SKIP_RETURN_CODE; T8 smoke arg
cmake/azgo_metal_kernels.inc.in                  T1 raw-string template for the embedded shaders
include/azgo/metal.hpp                           T1 metalBuilt/metalAvailable/metalDescription; probes T1-T6, T8
src/metal/metal_info.cpp                         T1 the three functions for builds without Metal
src/metal/shader_types.h                         T1 (grows each task) shared structs and constants
src/metal/kernels.metal                          T1 fill; T2 gemm_simd, splitk_reduce; T3 transpose_input, im2col3, col2im3;
                                                 T4 bn_*, relu_*, add_into, tanh_inplace, fill_zero; T5 bias_*, policy_*, *_loss;
                                                 T6 sgd_update
src/metal/metal_context.hpp, .mm                 T1 MetalContext, GpuBuffer, CommandBatch
src/metal/gpu_layers.hpp, .mm                    T2 gemm encode; T3 conv; T4 BN, elementwise; T5 Linear, loss
src/metal/gpu_network.hpp, .mm                   T6 GpuWeights, GpuWorkspace, GpuNetwork
src/metal/metal_probes.mm                        T1-T6, T8 probe implementations
include/azgo/param_layout.hpp, src/param_layout.cpp   T6 ParamLayout (portable)
include/azgo/training_engine.hpp, src/training_engine.cpp   T7 TrainingEngine, CpuTrainingEngine; T8 metal branch, cpuBackendName
src/metal/metal_engine.mm                        T8 MetalTrainingEngine
include/azgo/validation.hpp, src/validation.cpp  T7 heldOutLoss with a batch-loss function
src/trainer.cpp                                  T7 the engine; T8 cpuBackendName, device line, resume note, availability check
src/backend.cpp, include/azgo/backend.hpp        T8 makeBackend messages; header comment on the GPU seam
src/config.cpp, include/azgo/config.hpp          T8 backend = metal
apps/bench.cpp                                   T9 --backend metal
tests/test_metal.cpp                             T1-T6, T8
tests/test_training_engine.cpp                   T7, T8
tests/test_network.cpp                           T6 ParamLayout
tests/test_config.cpp, tests/test_trainer.cpp    T8
tests/integration_smoke.sh                       T8
README.md, docs/algorithm-explained.md, .html, docs/results/README.md, docs/results/bench.txt   T9, T10
```

## Conventions for Every Task

**Builds** (from the worktree root):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release                       # default: no BLAS, no Metal
cmake -S . -B build-blas -DCMAKE_BUILD_TYPE=Release -DAZ_USE_BLAS=ON
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DAZ_USE_METAL=ON
cmake --build build-metal -j 4 && ctest --test-dir build-metal --output-on-failure -R test_metal
```

At the end of every code task, all three builds compile and pass `ctest`. In `build` and `build-blas`, `test_metal` shows as Skipped. `ctest` does not build: run `cmake --build` first. A test that uses something not written yet fails at compile time, which is the expected "fails" state of a TDD cycle.

**TDD cycle per task:** write the listed tests first; build and watch them fail (compile error or assertion); implement; build and pass; run the three full suites; commit.

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

### Task 1: Build plumbing, the Metal context and a first kernel

**Effort:** ~0.5 day.

**Files:**
- Modify: `CMakeLists.txt`
- Create: `cmake/azgo_metal_kernels.inc.in`, `include/azgo/metal.hpp`, `src/metal/metal_info.cpp`, `src/metal/shader_types.h`, `src/metal/kernels.metal`, `src/metal/metal_context.hpp`, `src/metal/metal_context.mm`, `src/metal/metal_probes.mm`
- Test: `tests/test_metal.cpp`

**Interfaces:**
- Produces:
  - `bool metalBuilt()`: whether `AZ_USE_METAL` was compiled in.
  - `bool metalAvailable(std::string* why = nullptr)`: creates the context on first call; false with a reason when it is not built (`not built: rebuild with -DAZ_USE_METAL=ON (macOS only)`), when there is no device, when the GPU family is below Apple 7, or when the shaders or a pipeline fail to compile.
  - `std::string metalDescription()`: `"<device>, macOS <version>, shaders <fnv1a-8-hex>"`; empty when unavailable.
  - `metal_probe::fill(float* out, size_t n, float value)`.
  - Internal: `MetalContext::instance()`, `pipeline(name)`, `newBuffer(bytes)`; `GpuBuffer`; `CommandBatch` (`encoder()`, `dispatch1d/2d/threadgroups`, `commitAndWait()`).

- [ ] **Step 1: CMake.** Add `option(AZ_USE_METAL …)` before the BLAS block. With it on: `FATAL_ERROR` unless `APPLE`; `set(AZ_USE_BLAS ON CACHE BOOL "" FORCE)` plus a status message; `enable_language(OBJCXX)`; `set(CMAKE_OBJCXX_STANDARD 17)`. Always add `src/metal/*.cpp` to the library, which is portable C++. With Metal, also add `src/metal/*.mm` with `-fobjc-arc`, link `-framework Metal -framework Foundation`, and define `AZ_USE_METAL` publicly. Shader embedding: `file(READ)` `shader_types.h` and `kernels.metal`, concatenate, `configure_file(cmake/azgo_metal_kernels.inc.in ${CMAKE_BINARY_DIR}/generated/azgo_metal_kernels.inc @ONLY)` (template: `R"AZGO_METAL(@AZGO_METAL_SOURCE@)AZGO_METAL"`), add both sources to `CMAKE_CONFIGURE_DEPENDS`, and give the library that include directory privately. After the test loop: `set_tests_properties(test_metal PROPERTIES SKIP_RETURN_CODE 77)`.
- [ ] **Step 2: Write the failing tests** in `tests/test_metal.cpp`. `main` first checks `metalBuilt()`. If it is false, the test asserts that `metalAvailable(&why)` is false and that `why` contains `-DAZ_USE_METAL=ON`, prints `test_metal: SKIPPED (<why>)` and returns 77. Then it checks `metalAvailable`, with the same skip on false. Then the tests (inside `#ifdef AZ_USE_METAL`):
  - `test_context_comes_up`: the description is non-empty, contains the device name and is identical across two calls.
  - `test_fill_kernel`: `fill` of n ∈ {0, 1, 255, 256, 257, 100000} with 3.5 writes exactly n values of 3.5, and a sentinel past the end is untouched (the probe allocates n + 1 and checks).
  - `test_fill_repeat_bitwise`: 20 fills give identical bytes (the pattern every later reproducibility test follows).
- [ ] **Step 3: Implement** `metal_info.cpp` (stubs under `#ifndef AZ_USE_METAL`), `shader_types.h` (`FillParams`, threadgroup constants), `kernels.metal` (`fill`), `metal_context.mm` (singleton, family check, library compile with safe math, pipeline table with size checks, description with FNV-1a of the embedded string, `GpuBuffer`, `CommandBatch`), `metal_probes.mm` (`fill`). Start from the spike's `Gpu` and `Ctx` structs.
- [ ] **Step 4: Run all three builds.** `test_metal` passes in `build-metal` and is reported Skipped in `build` and `build-blas`. The default build compiles no `.mm` file (check the build log).

**Acceptance:**
- All three builds pass. `ctest` in `build` prints `test_metal …***Skipped` and a total of 29 tests.
- **CLT-only check:** on a Mac mini with only the Command Line Tools (`xcode-select -p` prints `/Library/Developer/CommandLineTools`), `build-metal` configures, builds, and `test_metal` passes. If no such machine is free during the task, record it as an open item, and it must pass before Task 8 merges.
- No Objective-C type in any header (`grep -n "id<\|@interface\|#import" src/metal/*.hpp include/azgo/*.hpp` is empty).

**Commit:** `Add the Metal build option, context and a first GPU kernel`. The body explains: why runtime-compiled embedded shaders (CLT-only Macs), why `AZ_USE_METAL` implies BLAS, the skip code, and the shared `shader_types.h`.

---

### Task 2: The gemm kernel and split-K

**Effort:** ~1 day.

**Files:**
- Modify: `src/metal/shader_types.h`, `src/metal/kernels.metal`, `src/metal/metal_probes.mm`, `include/azgo/metal.hpp`
- Create: `src/metal/gpu_layers.hpp`, `src/metal/gpu_layers.mm`
- Test: `tests/test_metal.cpp`

**Interfaces:**
- Produces:
  - Kernels `gemm_simd` (from the spike: 64×64 tile, 128 threads, BK = 16, transA/transB, alpha/beta epilogue that does not read C when beta = 0, `partial` mode) and `splitk_reduce`.
  - `int gemmSplitCount(int M, int N, int K)`: the spec's rule, as a pure function in `gpu_layers.hpp` with no Metal types.
  - `void encodeGemm(CommandBatch&, bool tA, bool tB, int M, int N, int K, float alpha, BufView A, BufView B, float beta, BufView C, GpuBuffer& splitScratch)`.
  - Probes: `metal_probe::gemm(tA, tB, M, N, K, alpha, A, B, beta, C)` (host arrays, `Backend::gemm` semantics) and `metal_probe::gemmSplitCount(M, N, K)`.

- [ ] **Step 1: Write the failing tests.**
  - `test_split_rule`: `gemmSplitCount(96, 864, 20736) == 20`; `(64, 576, 20736) == 20`; `(1, 96, 20736) == 20`; `(96, 864, 4095) == 1`; `(1024, 1024, 20736) == 1` (≥ 128 tiles); `(128, 1152, 92416) == 64`. Assert the documented formula for kChunk and the recomputed S on these shapes.
  - `test_gemm_against_double_reference`: all four trans combinations × shapes {(1,1,1), (7,63,65), (64,64,64), (65,129,17), (96,20736,864) forward 6×96, (864,20736,96) dX, (96,864,20736) dW, (256,82,162) policy fc, (82,162,256) its gW, (256,1,64) value fc2} × alpha {1, 0.5} × beta {0, 1}, against a `double` triple loop. Ceiling maxErr 1e-5. Print the worst.
  - `test_gemm_beta_zero_ignores_nan`: C prefilled with NaN, beta 0, gives finite results equal to C prefilled with 0.
  - `test_gemm_vs_blas_conv_shapes`: forward and dX conv shapes of 5×5, 9×9 (6×64, 6×96) and 19×19 at batches 1, 8 and 256, against `makeBackend("blas")`. Ceiling 2e-6. Also print the fraction of bitwise-equal elements (the spike saw 100%). It is information only, not asserted.
  - `test_gemm_repeat_bitwise`: 20 repeats of the dW (split) and forward shapes give identical bytes.
  - `test_gemm_columns_batch_independent`: for the 9×9 6×96 forward shape, the columns of sample b ∈ {0, 1, 7, 100, 255} computed with N = 81·256 equal those computed with N = 81 (that sample alone) and N = 81·8, bitwise. The same for rows of the Linear shape (M = B).
- [ ] **Step 2: Implement** `gemm_simd` and `splitk_reduce` in `kernels.metal` (port the spike's kernels; params from `shader_types.h`). `encodeGemm` dispatches `(⌈N/64⌉, ⌈M/64⌉, S)` threadgroups of 128 and, when S > 1, the reduce. Grow `splitScratch` on demand only outside a command buffer: the probe sizes it before encoding, and Task 6 sizes it once for the largest dW.
- [ ] **Step 3: Run** all builds and the full suites.

**Acceptance:** every test passes. The measured worst errors are written into the test comments. `gemm_tiled` is not added to `kernels.metal`: the explainer quotes it from `tools/metal_bench`.

**Commit:** `Add the deterministic simdgroup gemm with fixed split-K`. The body explains why split-K is only for dW and why its rule depends on the shape alone.

---

### Task 3: Data movement and convolutions

**Effort:** ~0.5–1 day.

**Files:**
- Modify: `src/metal/shader_types.h`, `src/metal/kernels.metal`, `src/metal/gpu_layers.hpp`, `src/metal/gpu_layers.mm`, `src/metal/metal_probes.mm`, `include/azgo/metal.hpp`
- Test: `tests/test_metal.cpp`

**Interfaces:**
- Produces:
  - Kernels `transpose_input`, `im2col3`, `col2im3`. The spike's `col2im3` gather already adds in the CPU's (dy, dx) order from 0.
  - `encodeTransposeInput`, `encodeIm2col3`, `encodeCol2im3`.
  - `encodeConvForward(cb, k, cin, cout, w, x, B, n, col, y, scratch)`: im2col when k = 3, then the gemm. `encodeConvBackward(cb, k, cin, cout, w, x, col, dy, B, n, dcol, gw, dx /*optional*/, scratch)`: dW with split-K, written with beta 0; dX into `dcol` then col2im for k = 3, or directly into `dx` for k = 1; nothing when `dx` is absent (the stem). The caller rebuilds `col` before backward, as `Network::backward` does.
  - Probes `metal_probe::transposeInput`, `im2col3`, `col2im3`, `convForward(const Conv2d&, x, B, n, y)`, `convBackward(const Conv2d&, x, dy, B, n, gw, dx /*nullable*/)`.

- [ ] **Step 1: Write the failing tests.**
  - `test_im2col_col2im_bitwise`: n ∈ {5, 9, 19}, C ∈ {1, 18, 96}, B ∈ {1, 2, 7}, random data. `memcmp` against `Backend::im2col` and `Backend::col2im` (a `NaiveBackend`'s base implementations).
  - `test_transpose_input_bitwise`: against the copy loop of `Network::forward`, written out in the test.
  - `test_conv_forward_vs_cpu`: 3×3 (stem 18 → F, block F → F) and 1×1 (F → 1, F → 2), F ∈ {8, 32, 96}, n ∈ {5, 9, 19}, B ∈ {1, 16, 256} (19×19 only up to 16), weights from `Conv2d::init`. Against `Conv2d::forward` with BLAS, no pool. Ceiling maxErr 2e-6.
  - `test_conv_backward_vs_cpu`: the same shapes. gw against `Conv2d::backward`'s gw (zeroed first): maxErr 1e-5. dx: maxErr 2e-6. With `dx` null the probe's dx output buffer, prefilled with a sentinel, is untouched.
  - `test_conv_repeat_bitwise`: 20 repeats of a 6×96 B = 256 backward give identical gw and dx.
- [ ] **Step 2: Implement** the kernels and encode functions. Dispatch grids from the spike (`(Mt, rows)` with threadgroups of 64×4 for im2col/col2im, a constant in `shader_types.h`).
- [ ] **Step 3: Run** all builds and the full suites.

**Acceptance:** all pass; the copy kernels are bitwise; the measured conv errors are recorded.

**Commit:** `Port im2col, col2im and the convolutions to Metal`. The body covers why col2im is a gather (no atomics, CPU order) and why backward rebuilds `col` (memory).

---

### Task 4: Batch norm and the elementwise kernels

**Effort:** ~1 day.

**Files:**
- Modify: `src/metal/shader_types.h`, `src/metal/kernels.metal`, `src/metal/gpu_layers.hpp`, `src/metal/gpu_layers.mm`, `src/metal/metal_probes.mm`, `include/azgo/metal.hpp`
- Test: `tests/test_metal.cpp`

**Interfaces:**
- Produces:
  - Kernels `bn_train_stats` (one threadgroup of `kReduceThreads` = 256 per channel: strided float sums, fixed tree; mean, then the variance around it; invStd; optional running update with momentum 0.1 and unbiased M/(M−1)), `bn_train_apply`, `bn_infer`, `bn_backward_sums` (writes ggamma = Σdy·xhat and gbeta = Σdy), `bn_backward_apply` (dx may alias dy), `relu_forward`, `relu_backward`, `add_into`, `tanh_inplace`, `fill_zero`.
  - `encodeBnTrain(cb, C, M, x, y, xhat, invStd, gamma, beta, runMean, runVar, updateRunning, scratch)`, `encodeBnInfer(...)`, `encodeBnBackward(cb, C, M, dy, xhat, invStd, gamma, dx, ggamma, gbeta, scratch)`, `encodeReluForward`, `encodeReluBackward`, `encodeAddInto`, `encodeTanh`, `encodeFillZero`.
  - Probes `metal_probe::bnForwardTrain(const BatchNorm2d&, x, M, updateRunning, y, xhat, invStd, runMean, runVar)`, `bnForwardInfer`, `bnBackward(const BatchNorm2d&, dy, xhat, invStd, M, dx, ggamma, gbeta)`, `reluForward`, `reluBackward`, `addInto`, `tanhInPlace`.

- [ ] **Step 1: Write the failing tests.**
  - `test_bn_forward_train_vs_cpu`: C ∈ {1, 2, 32, 96}, M ∈ {2, 25, 81·7, 81·256, 361·16} (not multiples of 256 too), gamma/beta random, runMean/runVar random. Compared with `BatchNorm2d::forwardTrain`: y, xhat, invStd maxErr 1e-5; runMean and runVar after one update maxErr 1e-6; with `updateRunning = false` they are bitwise unchanged.
  - `test_bn_two_pass_accuracy`: a channel with mean 1000 and standard deviation 0.01 normalizes to unit variance within 1e-3. A one-pass E[x²] − E[x]² would fail this, which guards the two-pass rule.
  - `test_bn_forward_infer_vs_cpu`: maxErr 1e-6.
  - `test_bn_backward_vs_cpu`: dx, ggamma and gbeta against `BatchNorm2d::backward` (ggamma/gbeta zeroed first). maxErr 1e-4. Also with dx aliasing dy.
  - `test_elementwise_bitwise`: ReLU forward/backward (including −0.0, +0.0, NaN-free random data with exact zeros), `add_into` against the CPU loops, `memcmp`.
  - `test_tanh_vs_cpu`: against `std::tanh` over [−20, 20]. maxErr 1e-6.
  - `test_reductions_repeat_bitwise`: 20 repeats of BN train and backward on 96 × 20736 give identical bytes.
- [ ] **Step 2: Implement.** Start the statistics kernel from the spike's `channel_stats`, adding the two passes, the running update, xhat and y (split into stats + apply), and the backward pair. If `test_bn_backward_vs_cpu` exceeds 1e-4, add Kahan summation to the strided phase of `bn_backward_sums` (and `bn_train_stats`), re-measure, and say so in the comment. Do not raise the ceiling.
- [ ] **Step 3: Run** all builds and the full suites.

**Acceptance:** all pass; measured errors recorded; the fixed tree shape is documented in a comment in `kernels.metal` (strided over 256 threads, then 128, 64, …, 1).

**Commit:** `Port batch norm and the elementwise layers to Metal`. The body explains float-in-a-fixed-tree as the GPU's answer to the CPU's double sums.

---

### Task 5: Heads and losses

**Effort:** ~1 day.

**Files:**
- Modify: `src/metal/shader_types.h`, `src/metal/kernels.metal`, `src/metal/gpu_layers.hpp`, `src/metal/gpu_layers.mm`, `src/metal/metal_probes.mm`, `include/azgo/metal.hpp`
- Test: `tests/test_metal.cpp`

**Interfaces:**
- Produces:
  - Kernels `bias_add_rows`, `bias_grad_rows` (one thread per output, serial over samples from 0), `policy_flatten`, `policy_unflatten`, `policy_value_loss` (one thread per sample: max, sum of exp, logZ; with gradients dLogits = (exp(l − logZ) − π)/B and dValue = 2(v − z)(1 − v²)/B; the terms −Σπ(l − logZ) and (z − v)²), `ownership_loss` (one thread per sample: dOwnership = 2w·d·(1 − o²)/(B·A); the term Σd²).
  - `encodeLinearForward(cb, in, out, w, b, x, B, y)`, `encodeLinearBackward(cb, in, out, w, x, dy, B, gw, gb, dx /*optional*/)`, `encodePolicyFlatten/Unflatten`, `encodeLoss(cb, cfg, B, logits, values, ownership, pi, z, targets, w, withGradients, logZ, dLogits, dValues, dOwnership, lossTerms)`.
  - `LossStats sumLossTerms(const float* terms, int B, bool hasOwnership, int area)` (CPU, in double, in sample order).
  - Probes `metal_probe::linearForward(const Linear&, x, B, y)`, `linearBackward(const Linear&, x, dy, B, gw, gb, dx)`, `policyFlatten`, `policyUnflatten`, and `metal_probe::LossResult loss(const NetConfig&, logits, values, ownershipOut, pi, z, targets, w, B, withGradients)`, which returns the terms, dLogits, dValues, dOwnership and the `LossStats`.

- [ ] **Step 1: Write the failing tests.**
  - `test_linear_vs_cpu`: the three head layers' shapes for 5×5, 9×9 and 19×19 at B ∈ {1, 16, 256}, against `Linear::forward`/`backward` (BLAS): y maxErr 2e-6; gw 1e-5; **gb bitwise**; dx 1e-5.
  - `test_policy_flatten_bitwise`: both directions against the loops of `Network::forward`/`backward`, written out in the test.
  - `test_policy_value_loss_vs_reference`: random logits (also ±80 extremes; no overflow, finite results), π one-hot and spread, v in (−1, 1), z ∈ {−1, 0, 1}. Against a double reference written in the test from `Network::loss`'s formulas: per-sample terms relative 1e-6, dLogits and dValues maxErr 1e-6, `LossStats` relative 1e-6.
  - `test_ownership_loss_vs_reference`: w ∈ {0, 1.5}, targets in {−1, 0, 1}. dOwnership is all zero for w = 0, and the reported term is unchanged by w.
  - `test_loss_without_gradients_writes_no_gradients`: gradient buffers prefilled with a sentinel stay untouched when `withGradients` is false.
- [ ] **Step 2: Implement.**
- [ ] **Step 3: Run** all builds and the full suites.

**Acceptance:** all pass; the bias gradient is bitwise; measured errors recorded.

**Commit:** `Port the policy, value and ownership heads and their losses to Metal`. The body explains why the per-sample work is on the GPU and the batch sums on the CPU in double.

---

### Task 6: The whole network on the GPU

**Effort:** ~2 days. The largest task; review it with care.

**Files:**
- Create: `include/azgo/param_layout.hpp`, `src/param_layout.cpp`, `src/metal/gpu_network.hpp`, `src/metal/gpu_network.mm`
- Modify: `src/metal/shader_types.h`, `src/metal/kernels.metal` (`sgd_update`), `src/metal/gpu_layers.*` (`encodeSgd`), `src/metal/metal_probes.mm`, `include/azgo/metal.hpp`
- Test: `tests/test_network.cpp` (ParamLayout), `tests/test_metal.cpp`

**Interfaces:**
- Produces:
  - `struct ParamLayout { struct Entry { std::string name; size_t size, offset; bool decay; }; std::vector<Entry> params, buffers; size_t paramFloats, bufferFloats; static ParamLayout of(Network&); }`. Offsets are aligned to 64 floats. Portable.
  - `GpuWeights(MetalContext&, const ParamLayout&)` with `upload(const Network&)` (params + BN statistics) and `download(Network&)`. `GpuOptimizerState` adds the `grads` and `velocity` arenas, with `uploadVelocity`/`downloadVelocity`. `GpuWorkspace(MetalContext&, const NetConfig&, int maxBatch)`: the `Workspace` fields plus staging, `lossTerms`, `logZ`, `bnScratch`, `splitK`; `bytes()` for the memory check.
  - `GpuNetwork(const NetConfig&)` with `encodeForward(cb, ws, weights, B, Mode)`, `encodeLoss(…)`, `encodeBackward(cb, ws, weights, grads, B, withOwnership)`, `encodeSgd(cb, weights, opt, layout, lr, momentum, weightDecay)`. Synchronous conveniences: `infer(...)`, `inferLoss(...) → LossStats`, `trainStep(..., lr, momentum, weightDecay, ownershipWeight) → LossStats` (one `CommandBatch`), and `gradients(...) → LossStats` (no SGD, for tests).
  - Probe class `metal_probe::Net` (pimpl), built from a `Network&`, `maxBatch`, momentum, weight decay and ownership weight. It offers `infer`, `computeLoss` (Probe mode), `trainForward` (Train mode, outputs and running statistics), `gradients(…, Network& into)` (writes into `params()[i].grad` and running statistics), `step(…, lr)`, `read(Network&, velocity&)`, and `bytes()`.

- [ ] **Step 1: Write the failing tests.**
  - `tests/test_network.cpp` `test_param_layout_matches_params` (every build): the names, sizes and decay flags equal `params()`/`buffers()` in order, with and without the ownership head and with 0 blocks; offsets are aligned to 64 and do not overlap.
  - `test_metal.cpp`, configurations C = {5×5 1×8, 9×9 2×16, 9×9 6×32, 9×9 0 blocks × 16, 9×9 2×16 + ownership}:
    - `test_infer_vs_cpu`: logits, values and ownership against `Network::infer`, maxErr 1e-5.
    - `test_probe_loss_vs_cpu`: against `Network::computeLoss`, `LossStats` relative 1e-5.
    - `test_train_forward_vs_cpu`: one Train-mode forward against `trainStep`'s forward (compare the BN running statistics after it): maxErr 1e-6 on statistics; logits and values 1e-4.
    - `test_train_step_gradients_vs_cpu`: against `Network::trainStep`: `LossStats` relative 1e-5, and every parameter's gradient normErr 1e-4 and maxErr 1e-3. Ownership configuration with w = 1.5, with w = 0 (the head's grads exactly 0) and with no targets.
    - `test_five_steps_vs_cpu`: lr 0.02, momentum 0.9, decay 1e-4, five different random batches: each step's `LossStats` relative 1e-4; after step 5 every weight tensor normErr 1e-5, every momentum tensor normErr 1e-3, BN statistics maxErr 1e-5. With w = 0 the head's weights still change (decay and momentum), as on the CPU.
    - `test_step_repeat_bitwise`: from one uploaded state, step 1 run 20 times (re-uploading the state each time) gives identical bytes in every arena.
    - `test_two_engines_bitwise`: two `Net` probes from the same network, three steps each, give identical bytes.
    - `test_infer_batch_independent`: positions {0, 1, 7, 100, 255} alone, in a batch of 8 and in a batch of 256 give bitwise identical logits, value and ownership (9×9 6×32 + ownership).
    - `test_load_store_round_trip`: upload then download with no step gives the network and momentum back byte for byte.
    - `test_unknown_tensor_refused`: a `ParamLayout` whose names differ from the network's makes upload throw `std::logic_error` naming the tensor.
    - `test_memory_estimate`: `bytes()` of 9×9 6×96 at batch 256 is within 10% of the spec's ~390 MB plus arenas.
- [ ] **Step 2: Implement** `ParamLayout`, the arenas and the workspace (field names as in `Workspace`). Write `encodeForward`/`encodeBackward` by following `network.cpp` line by line, and keep the comments that explain each step. Write `sgd_update` and the per-tensor `encodeSgd`. Implement `trainStep` as in the spec's "One step, one command buffer", items 1–6. Implement `inferLoss`: transpose, Infer forward, loss terms without gradients. Wrap every public entry point in `@autoreleasepool`.
- [ ] **Step 3: Run** all builds and the full suites. Also run `test_metal` alone 3 times in a row: no flaky results.

**Acceptance:** all pass within the spec's ceilings. A 1,000-step loop in a scratch program (not committed) shows flat resident memory, which checks the autorelease pools. Measured errors recorded.

**Commit:** `Run the whole network on the GPU: forward, loss, backward and SGD`. The body covers residency, one command buffer per step, and what the tests guarantee.

---

### Task 7: The training-engine seam (CPU only)

**Effort:** ~1 day. Independent of Tasks 1–6; it may be done first or in parallel.

**Files:**
- Create: `include/azgo/training_engine.hpp`, `src/training_engine.cpp`, `tests/test_training_engine.cpp`
- Modify: `include/azgo/validation.hpp`, `src/validation.cpp`, `src/trainer.cpp`, `README.md` (test count only)

**Interfaces:**
- Produces: `TrainingEngine`, `TrainingEngineOptions`, `makeTrainingEngine` as in the spec (this task supports `naive` and `blas`; `metal` throws `backend 'metal' not available: rebuild with -DAZ_USE_METAL=ON (macOS only)` until Task 8). `CpuTrainingEngine`. `using BatchLossFn = std::function<LossStats(const float* input, const float* pi, const float* z, int B, const float* ownership)>;` and `HeldOutLoss heldOutLoss(const BatchLossFn&, int maxBatch, ThreadPool* pool, const NetConfig&, const std::vector<PositionRef>&, int history)`. The old `heldOutLoss(net, ws, …)` becomes a wrapper.
- Consumes: `Network`, `Workspace`, `SgdMomentum`, `ThreadPool`.

- [ ] **Step 1: Write the failing tests** in `tests/test_training_engine.cpp`:
  - `test_cpu_engine_equals_direct_calls`: for naive (and BLAS when available), 5×5 and 9×9, with and without the ownership head, pool sizes 1 and 3. Three steps through the engine against `Network::trainStep` + `SgdMomentum::step` on a copy. `LossStats`, every parameter, BN statistic and momentum value are **bitwise** equal.
  - `test_cpu_engine_infer_loss_equals_network`: `inferLoss` equals `Network::inferLoss` bitwise.
  - `test_held_out_loss_variant_equals_old`: the new signature with a `BatchLossFn` over `Network::inferLoss` equals the old one bitwise, including a final partial batch.
  - `test_metal_engine_unavailable_message`: in a build without Metal, `makeTrainingEngine` with `metal` throws `std::invalid_argument` whose message contains `-DAZ_USE_METAL=ON`. In a Metal build this test is skipped inside the executable (until Task 8 adds the engine).
- [ ] **Step 2: Implement** the engine and the validation variant. Rewrite the trainer's validation-and-train block: create the engine after the pool; `load(net_, state_.velocity)`; both `heldOutLoss` calls through `engine->inferLoss`; the step loop calls `engine->trainStep(…, lr)` and keeps its per-step finite check and loss sums unchanged; `store(net_, velocity)` into a local; the existing finite scan over `net_.params()` and that local; then `state_.velocity = velocity`. Destroy the engine before `trainPoolOwner.reset()`.
- [ ] **Step 3: Run** the full suites in `build` and `build-blas`. All existing `test_trainer` byte-identity tests must pass unchanged.
- [ ] **Step 4: Old against new binary.** Build main's `train` (a separate worktree or `git stash`-free checkout into the scratchpad) and this branch's, both with BLAS. Run each for 3 generations on: the 5×5 preset shrunk (`TINY` settings from the smoke test) with naive and with BLAS, and a 9×9 6×64 BLAS run with `train_steps = 20`, `games_per_generation = 8`. `cmp` the checkpoints, `state.bin` and `results.csv`. Use `-j 4` and small runs: the machine is busy.

**Acceptance:** all suites pass; the old-vs-new comparison is byte-identical for every file. Record the commands and outcome in the commit body. The CTest total is 30 (29 + `test_training_engine`) and the README count is updated.

**Commit:** `Train through a TrainingEngine seam, the CPU path unchanged`. The body explains why the seam moved above `Network` (the spike) and how byte-identity was verified.

---

### Task 8: The Metal engine in the trainer, config and CLI

**Effort:** ~1.5 days.

**Files:**
- Create: `src/metal/metal_engine.mm`
- Modify: `include/azgo/training_engine.hpp`, `src/training_engine.cpp` (the metal branch, `cpuBackendName`, `checkTrainingBackend`), `src/backend.cpp`, `include/azgo/backend.hpp` (messages, and the header comment on where the GPU plugs in), `src/config.cpp`, `include/azgo/config.hpp` (comment), `src/trainer.cpp`, `include/azgo/metal.hpp` (failure-injection probe), `CMakeLists.txt` (smoke-test argument), `tests/integration_smoke.sh`
- Test: `tests/test_metal.cpp`, `tests/test_training_engine.cpp`, `tests/test_config.cpp`, `tests/test_trainer.cpp`

**Interfaces:**
- Produces:
  - `MetalTrainingEngine : TrainingEngine`. Creating it builds `ParamLayout`, `GpuWeights`, `GpuOptimizerState` and `GpuWorkspace(maxBatch)`, and checks the total against `recommendedMaxWorkingSetSize` (`std::runtime_error`: `metal: training needs N MB of GPU memory, more than the device's recommended M MB; lower batch_size`). `load` uploads and zeroes the grads; `inferLoss` and `trainStep` are synchronous, one command buffer each; `store` downloads. The destructor waits for any command buffer in flight.
  - `std::string cpuBackendName(const std::string&)`; `void checkTrainingBackend(const std::string& name)`, which throws the spec's messages (not built; no usable GPU with the reason).
  - `metal_probe::failNextCommandBuffers(int n)`: the next n command buffers report a failure through the normal error path.
  - Config: `backend` accepts `metal` (message `expected naive|blas|metal`).
  - Trainer: the startNew/resume availability check via `checkTrainingBackend` before anything is written; self-play, the arena phase and `rerateRun` via `cpuBackendName` (re-rating a metal run needs only BLAS); the device line when the backend is `metal`; the resume note when the backend changed; command-buffer failure message `GPU command buffer failed at generation g step s: <error>; last good state kept` (the engine throws, and the trainer adds g and s).
  - `makeBackend("metal")` throws the training-only message. The unknown-backend message names all three.

- [ ] **Step 1: Write the failing tests.**
  - `test_config`: `backend = metal` parses, prints and round-trips; `backend = gpu` fails with `expected naive|blas|metal`; `cpuBackendName` maps metal → blas and leaves the others.
  - `test_training_engine`: `makeBackend("metal")` throws the training-only message; in a build without Metal, `checkTrainingBackend("metal")` throws the not-built message.
  - `test_trainer` (every build): `test_metal_unavailable_writes_nothing`. In a build without Metal, `startNew` with `backend = metal` throws, and the run directory has no `state.bin`, `config.cfg` or checkpoint. In a Metal build it is skipped inside the test.
  - `test_metal.cpp`, tiny 5×5 config (`tinyConfig` copied from `test_trainer.cpp`, with `backend = metal`):
    - `test_metal_engine_vs_cpu_engine`: `load` → `inferLoss` → three `trainStep`s → `store` against `CpuTrainingEngine` (BLAS). Tolerances as Task 6; `load`/`store` with no step round-trips bitwise.
    - `test_metal_resume_is_byte_identical`: a 4-generation run against 2 generations + resume to 4. The checkpoint, `state.bin`, `elo.csv`, `results.csv`, `references.csv`, the SGF and every validation column are identical.
    - `test_metal_worker_and_thread_independence`: `workers` 1 vs 3 and `train_threads` 1 vs 4 give identical files.
    - `test_metal_runs_are_reproducible`: two fresh runs of 2 generations give identical files.
    - `test_switch_backend_prints_note`: 2 generations with `blas`, then resume with `--set backend=metal`. The log contains the note, and generation 3 completes.
    - `test_failed_command_buffer_keeps_last_good_state`: `failNextCommandBuffers(1)` during generation 2's training. `runGeneration` throws with the message; `state.bin` still says generation 1; a resume then reproduces the uninterrupted run's generation 2 byte for byte.
    - `test_memory_check_refuses_oversized_batch`: an engine for 19×19 10×128 with `maxBatch` large enough to exceed the device's recommended working set throws the memory message. Compute the batch from `recommendedMaxWorkingSetSize`; the probe exposes it.
  - `integration_smoke.sh`: a third argument `metal=ON|OFF`, passed by CMake from `AZ_USE_METAL`. ON: two generations of the `TINY` run with `--set backend=metal`, the device line printed, resume to 3, `--rerate` of that run succeeds, `arena --backend metal` fails with the training-only message. OFF: `train … --set backend=metal` fails with the not-built message and leaves no `state.bin`.
- [ ] **Step 2: Implement.** Keep `metal_engine.mm` thin: it adapts `GpuNetwork` to `TrainingEngine`.
- [ ] **Step 3: Run** all three builds' full suites, including `integration_smoke`.

**Acceptance:** all pass. A manual run, `train --config configs/9x9.cfg --run <scratchpad>/m --set backend=metal --set games_per_generation=8 --set train_steps=20 --generations 2`, prints the device line, trains, and its checkpoints load in `arena --backend blas`.

**Commit:** `Train on the GPU with backend = metal`. The body covers the determinism contract, the sync points, the switching note, and why the apps refuse `metal` for now.

---

### Task 9: Benchmark and measurements

**Effort:** ~0.5 day of work, plus about an hour of machine time with the jobs paused (only with Mark's go-ahead).

**Files:**
- Modify: `apps/bench.cpp`
- Results: `docs/results/bench.txt`, `docs/results/README.md`

- [ ] **Step 1: Bench.** `--backend metal`, and `all` including metal when `metalAvailable()`. Main table: the metal row shows the train step at batch 128 and `-` for evals/s. `--train-threads` table: the metal row once, threads `gpu`. A metal-only table per shape: train-step wall ms, GPU ms (`GPUEndTime − GPUStartTime`, exposed by a probe), inference evals/s at batches 8, 144 and `--train-batch`, and `load`/`store` ms. The usage text is updated. Smoke test: `bench --sizes 5 --seconds 0.1 --backend metal` in the ON branch.
- [ ] **Step 2: Measure** (jobs paused, machine otherwise idle). `bench --sizes 9 --backend all --configs DIR --train-threads 18 --train-batch 256` with 6×64 and 6×96, and `--sizes 19` with the preset. Append the output to `docs/results/bench.txt` with the machine, date and conditions.
- [ ] **Step 3: Real-run comparison.** On the 6×96 9×9 preset variant, run `train --generations 1` twice into the scratchpad (`backend=blas` and `backend=metal`, same seed) and compare `train_s`, `policy_loss`, `value_loss` and the validation columns. The losses must agree within 2% relative. Record the numbers in `docs/results/README.md`.
- [ ] **Step 4: Acceptance against the spec.** 6×64 step ≤ 30 ms; 6×96 step ≤ 45 ms; 6×96 phase ≤ 20 s; 19×19 10×128 step ≤ 500 ms. If one misses, stop and report the numbers to Mark with the spec's list of levers. Do not optimize inside this task.

**Commit:** `Benchmark the Metal training engine`. The body gives the headline numbers against the CPU's 18 threads.

---

### Task 10: Documentation and the explainer

**Effort:** ~1 day.

**Files:**
- Modify: `README.md`, `docs/algorithm-explained.md`, `docs/algorithm-explained.html`, `docs/results/README.md`, possibly `tools/check_citations.py`

- [ ] **Step 1: README.** Build: the `AZ_USE_METAL` command, its requirements (an Apple-silicon Mac, Command Line Tools suffice), the test count (30, `test_metal` Skipped elsewhere). Usage: a **Backends** paragraph after **Threads** (what `metal` moves, what stays on the CPU, the determinism contract including "same machine and macOS", the switching note, the device line). "Extending to the GPU": rewrite with options 1–3 and the measured numbers from Task 9. "Learn how it works": list this spec and plan.
- [ ] **Step 2: Explainer.** Fix chapter 03's "This is where a GPU would plug in" paragraph: the per-op seam was measured and rejected; point to the new section. Add "Training on the GPU" to chapter 08 after the training-threads passage, with the spec's outline: why the seam moved, residency and the two crossings, a gemm tile (the scalar `gemm_tiled` from `tools/metal_bench/metal_bench.mm`, then the simdgroup form from `src/metal/kernels.metal`), determinism without atomics, no double on the GPU and the measured tolerance, and the results. Quote real source with `file:line` citations, as the rest of the explainer does.
- [ ] **Step 3: HTML.** The same text, plus one diagram of the training phase's timeline (load → validation batches → step command buffers → store) with the crossings marked, in the existing diagram style. It must follow the light/dark theme as the others do.
- [ ] **Step 4: Check.** `python3 tools/check_citations.py` passes. Extend it to `.mm`/`.metal` if it rejects them. Re-read every changed header's opening comment for accuracy (`backend.hpp`, `evaluator.hpp` "A future GPU evaluator…", `trainer.hpp`).

**Acceptance:** citations pass; the README's numbers match `docs/results/`; no claim of CPU–GPU bit-identity anywhere.

**Commit:** `Explain the Metal training engine in the README and the explainer`.

---

## Effort Summary

| task | what | effort |
|---|---|---|
| 1 | build plumbing, context, first kernel | 0.5 d |
| 2 | gemm, split-K | 1 d |
| 3 | im2col, col2im, convolutions | 0.5–1 d |
| 4 | batch norm, elementwise | 1 d |
| 5 | heads, losses | 1 d |
| 6 | whole network, SGD, one command buffer per step | 2 d |
| 7 | engine seam, CPU byte-identity | 1 d |
| 8 | Metal engine, trainer, config, CLI, smoke | 1.5 d |
| 9 | bench and measurements | 0.5 d + machine time |
| 10 | docs and explainer | 1 d |
| | **total** | **~10–11 days** |

Tasks 1–6 are sequential (each builds on the previous kernels). Task 7 is independent and can go first or in parallel. Task 8 needs 6 and 7. Tasks 9 and 10 come last.
