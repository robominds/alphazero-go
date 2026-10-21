# CUDA Inference Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `backend = cuda`, an inference-only NVIDIA GPU backend for Linux, joins distributed self-play as the engine `pc/cuda`. It is a `BatchRunner` behind the existing `InferenceHub`. Its hand-written fp32 kernels are batch-independent with a fixed accumulation order, and the PC plays its slice of every generation at about twice the Studio GPU's rate.

**Architecture:** `src/cuda/` holds a one-to-one port of the Metal forward kernels (`.cu`). Every 3×3 convolution goes through the spike's fused implicit-GEMM kernel (BN, skip and ReLU in the epilogue). `CudaBatchRunner` has two slots (streams, pinned staging) and one completion thread. `include/azgo/cuda.hpp` is plain C++ and answers "not built" elsewhere. `WorkerEvaluators`, the engines list, the hosts file (`cuda_workers`), `selfplay`, `bench` and `cluster_sync.sh` learn the name `cuda`. Only SASS is built (`AZ_CUDA_ARCHITECTURES`, default `89-real`, no PTX), so a driver never changes the bits.

**Tech Stack:** C++17, CUDA 12.8 (`nvcc`, CUDA runtime, static), CMake ≥ 3.18 for the CUDA language (≥ 3.16 stays the minimum without CUDA), Ubuntu 24.04 under WSL2, assert-based CTest executables.

**Spec:** docs/superpowers/specs/2026-10-20-cuda-inference-backend-design.md

**Spike:** tools/cuda_bench, docs/results/cuda-bench.txt

**Author:** Mark Castelluccio, with planning assistance from Claude (Anthropic).

## Global Constraints

- **Every existing build is byte-identical.** Without `-DAZ_USE_CUDA=ON` nothing CUDA is compiled; the Mac builds (`build`, `build-blas`, `build-metal`) and the PC's `build` and `build-blas` pass every test unchanged, and a run without a cuda engine writes exactly what `main` writes. Task 10 checks old against new binaries on the default path.
- **The algorithm does not change.** No new randomness, no change to search, self-play, resignation, training or the arena. A `pc/cuda` game is the same game with differently rounded evaluations.
- **Determinism by construction.** No atomics, one writer per output, no cross-thread reductions, k always from 0 to K in order, fixed block sizes, grids that depend only on shapes, no reads of unwritten memory. `--fmad=false`; `fmaf` only where the design says so; no `--use_fast_math`; IEEE div/sqrt; denormals kept.
- **No cuBLAS, cuDNN, TF32, half precision, PTX or JIT.** Only `NN-real` architectures.
- **Machines.** Build and test CUDA on the PC (`ssh pc`, work in `~/azgo-dev`; it is idle). On the Studio: `nice -n 10`, `-j6`, never touch `runs/`, no GPU timing while it trains (a 13×13 run), never use the minis. The cuda code must still configure-and-skip cleanly on the Mac (`AZ_USE_CUDA=ON` there is a fatal error by design; test that once).
- **Privacy.** No host names, user names, IP addresses or home paths in any committed file. Before every commit: `git diff --cached | grep -nE '([0-9]{1,3}\.){3}[0-9]{1,3}|/Users/|/home/'` plus the remote user's name; it must print nothing except version strings checked by eye.
- **Tests.** Assert-based, `-UNDEBUG`, fixed `Rng` seeds. `test_cuda` returns 77 when CUDA is not built or not usable. A test translation unit that holds exact CPU references is compiled with `-ffp-contract=off`.
- **Files.** New files open with a teaching comment and `Mark Castelluccio, with assistance from Claude (Anthropic).` Style: camelCase functions, PascalCase types, `trailing_` members, `kConstants`, 4-space indent, ≤ 100 columns.
- **Commits.** Imperative subject, a body that says why, ending `Co-Authored-By: Claude <model> <noreply@anthropic.com>` naming the model that writes it. No merge, no push.

## Review Focus

1. **Bits.** The gemm equals an `std::fma` chain; the fused convolution equals the explicit sequence; every kernel is batch-independent (Tasks 3–5, 7).
2. **The runner's lifetime.** No CUDA call from a host callback, errors reach every waiting request, the destructor never frees memory a pass still uses, no thread outlives the runner (Task 7).
3. **No PTX.** The architecture check at startup and at configure (Task 1).
4. **Default path identity** (Task 10) and privacy (every task).

---

## Notes Before You Start

**Read first:** the spec; the Metal design's "Forward kernels", "Why the results cannot depend on batching", "Determinism contract" and "Tolerance against the CPU"; `include/azgo/metal.hpp`; `src/metal/metal_info.cpp`, `metal_runner.mm`, `gpu_network.mm`, `kernels.metal`; `include/azgo/inference_hub.hpp` and `src/inference_hub.cpp` (`WorkerEvaluators`); `include/azgo/param_layout.hpp`; `src/layers.cpp` (`Conv2d`, `BatchNorm2d::forwardInfer`, `Linear::forward`); `src/network.cpp` (`forward`); `src/engines.cpp`; `src/coordinator.cpp` (preflight); `apps/selfplay.cpp`; `apps/bench.cpp`; `tests/test_metal.cpp`; `tools/cuda_bench/cuda_bench.cu` (the kernels to port).

**The PC.** `nvcc` is `/usr/local/cuda/bin/nvcc`; put `/usr/local/cuda/bin` on the non-interactive `PATH` (`~/.bashrc` is read by bash only for interactive shells: use `~/.profile` plus `BASH_ENV`, or pass `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`). For development, rsync the worktree to `~/azgo-dev/src` and commit it into the scratch git repository there, so build IDs stay meaningful; the real distributed test uses `tools/cluster_sync.sh`.

## File Structure

| file | role |
|---|---|
| `include/azgo/cuda.hpp` | public, CUDA-free: `cudaBuilt`, `cudaAvailable`, `cudaDescription`, `makeCudaRunner`, `checkCudaInferenceMemory`, `namespace cuda_probe` |
| `src/cuda/cuda_info.cpp` | always compiled: the not-built answers |
| `src/cuda/kernels.cuh` | block sizes, tile shapes, parameter structs, kernel declarations |
| `src/cuda/kernels.cu` | every kernel (`fill`, `transpose_input`, `gemm` NN/NT, `conv_fused`, `im2col3`, `bn_fold`, `bn_infer`, `relu_forward`, `add_into`, `policy_flatten`, `bias_add_rows`, `tanh_inplace`) and their launchers |
| `src/cuda/cuda_context.cu/.cuh` | the process-wide device context, the architecture check, the description, error strings |
| `src/cuda/cuda_network.cu/.cuh` | weights (ParamLayout arenas, upload, `bn_fold`), workspace, `encodeForward` |
| `src/cuda/cuda_runner.cu` | `CudaBatchRunner`, the memory check |
| `src/cuda/cuda_probes.cu` | `cuda_probe::*` for tests and bench |
| `cmake/azgo_cuda.cmake` | option handling, architectures check, flags, the kernel hash |
| `tests/test_cuda.cpp`, `tests/cuda_reference.cpp` | the tests; exact CPU references (compiled `-ffp-contract=off`) |

## Conventions for Every Task

- Steps: **failing tests first** (show them fail: compile error or assertion), then the implementation, then the run.
- "Run" means, unless the task says otherwise: on the PC `build-cuda` (`-DAZ_USE_CUDA=ON`), `build` and `build-blas`, full `ctest`; on the Studio `build`, `build-blas`, `build-metal` full `ctest` (nice, -j6) whenever a portable file changed.
- Each task ends with one commit (or the few named), after the privacy grep.

---

### Task 1: The build option, the context and the first kernel

**Effort:** ~0.5 day.

**Files:** Create `cmake/azgo_cuda.cmake`, `include/azgo/cuda.hpp`, `src/cuda/cuda_info.cpp`, `src/cuda/cuda_context.cu/.cuh`, `src/cuda/kernels.cuh`, `src/cuda/kernels.cu` (`fill` only), `src/cuda/cuda_probes.cu`, `tests/test_cuda.cpp`. Modify `CMakeLists.txt`, `src/backend.cpp` (`checkBackendAvailable`), `src/build_info.cpp` (`buildOptions`), `tests/test_distributed.cpp` (`test_build_options`).

**Produces:**

```cpp
bool cudaBuilt();
bool cudaAvailable(std::string* why = nullptr);   // device 0, architecture check, fill probe
std::string cudaDescription();   // "NVIDIA GeForce RTX 4070 Ti SUPER sm_89, driver 13.3, runtime 12.8, kernels <8 hex>"
namespace cuda_probe { std::string deviceName(); int binaryVersion(); void fill(float*, size_t n, float v, size_t tail = 0); }
```

- [ ] **Step 1: Failing tests.**
  - `test_cuda` (new; returns 77 unless `cudaBuilt() && cudaAvailable()`): `test_available_and_description` (the shape `^.+ sm_[0-9]+, driver [0-9.]+, runtime [0-9.]+, kernels [0-9a-f]{8}$`, stable across calls, no host name); `test_architecture_matches` (`binaryVersion()` = 10·major + minor of the device); `test_fill` (n = 0, 1, 1000, 1 << 20; tail untouched).
  - In every build: `test_distributed`'s `test_build_options` accepts `cuda`; `checkBackendAvailable("cuda")` without CUDA throws "backend 'cuda' not available: rebuild with -DAZ_USE_CUDA=ON (Linux, NVIDIA GPU)".
- [ ] **Step 2: Implement.**
  - `option(AZ_USE_CUDA …)`: `APPLE` → `FATAL_ERROR`; `cmake_minimum_required(VERSION 3.18)` inside the branch; `enable_language(CUDA)`; `find_package(CUDAToolkit REQUIRED)`; shadow `AZ_USE_BLAS` on (as Metal does); `AZ_CUDA_ARCHITECTURES` (default `89-real`), each entry must match `^[0-9]+-real$` else `FATAL_ERROR` with the spec's message; `CMAKE_CUDA_ARCHITECTURES`; flags `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false -Xptxas -O3`; a check that `CMAKE_CUDA_FLAGS` holds no `use_fast_math`; glob `src/cuda/*.cu` into `azgo`; `CUDA::cudart_static`; `AZ_USE_CUDA` public define.
  - The kernel hash: SHA-256 at configure over the sorted contents of `src/cuda/*.cu`/`*.cuh`, `CMAKE_CUDA_COMPILER_VERSION`, the architectures and the CUDA flags; first 8 hex into `generated/azgo_cuda_info.inc`; the sources added to `CMAKE_CONFIGURE_DEPENDS`.
  - `CudaContext::get()` (once, thread-safe): `cudaGetDeviceCount`, device 0 properties, `cudaFuncGetAttributes(fill)` → `binaryVersion` check, then a `fill` probe; its failure becomes `why`.
  - `test_cuda` registered with `SKIP_RETURN_CODE 77` in every build (it compiles everywhere: the CUDA-only parts sit under `#ifdef AZ_USE_CUDA`).
- [ ] **Step 3: Run.** PC three builds; Studio three builds; on the Studio also `cmake -S . -B /tmp-scratch -DAZ_USE_CUDA=ON` fails with the macOS message (in the scratchpad, then deleted). On the PC configure once with `-DAZ_CUDA_ARCHITECTURES=89` and see the refusal, and once with `86-real` and see `cudaAvailable` false with the reconfigure hint.

**Acceptance:** `ctest -N | tail -1` reads `Total Tests: 34` in every build; `test_cuda` passes on the PC's `build-cuda` and is Skipped elsewhere. `cuobjdump --list-ptx build-cuda/libazgo.a` lists nothing.

**Commit:** `Add the AZ_USE_CUDA build option and the CUDA context`.

---

### Task 2: `cuda` as a backend name everywhere it is checked

**Effort:** ~0.25 day.

**Files:** Modify `src/backend.cpp` (`cpuBackendName`), `src/config.cpp` (`backend` key refusal), `src/engines.cpp` (`EngineSpec` validation, `cuda_workers`, `engineWorkers`, `engineFingerprint`), `include/azgo/engines.hpp`, `src/coordinator.cpp` (preflight: a local cuda engine is checked like metal), `tests/test_distributed.cpp`, `tests/test_config.cpp`.

- [ ] **Step 1: Failing tests.**
  - `test_engine_list_parse_print`: `pc/cuda:60, local/metal:40` parses and prints back.
  - `test_hosts_parse`: `cuda_workers=0..kMaxArenaWorkers` accepted, `-1` and too large refused naming the key.
  - `test_engine_workers`: cuda with `cuda_workers = 0` → every game of the slice; with 12 → 12; naive/blas/metal unchanged.
  - `test_fingerprint`: with CUDA usable, `cuda; <cpu>; <os>; ` + `cudaDescription()`; without, `engineFingerprint("cuda")` throws `checkBackendAvailable`'s message.
  - `test_config`: `backend = cuda` → `ConfigError` containing `selfplay_engines`.
  - `cpuBackendName("cuda") == "blas"`.
- [ ] **Step 2: Implement** exactly that.
- [ ] **Step 3: Run** (Studio and PC, all builds).

**Acceptance:** all pass; a coordinator on the Mac accepts `selfplay_engines = local/metal:40, pc/cuda:60` with a hosts file that defines `pc` (dry preflight in a test with the in-process transport reporting `--fingerprint` failure as unreachable, not as a config error).

**Commit:** `Accept cuda as a self-play engine backend`.

---

### Task 3: The gemm (NN and NT), bitwise against an fma chain

**Effort:** ~0.75 day.

**Files:** Modify `src/cuda/kernels.cu/.cuh`, `src/cuda/cuda_probes.cu`, `include/azgo/cuda.hpp`. Create `tests/cuda_reference.cpp` (+ header), compiled into `test_cuda` with `-ffp-contract=off`.

**Produces:** `cuda_probe::gemm(bool transB, int M, int N, int K, const float* A, const float* B, float* C, int tile = -1)` (row-major, β = 0, α = 1; `tile` −1 = the rule, else 0/1/2 forces 32×256 / 64×128 / 128×128); `reference::gemmFmaChain(...)`: `acc = std::fma(a, b, acc)` for k = 0…K−1 from `acc = 0.0f`.

- [ ] **Step 1: Failing tests** (`test_cuda`).
  - `test_gemm_equals_fma_chain`: NN and NT, bitwise, on (M, N, K) = every preset's stem, block and head shapes at batches 1, 8, 144, 256, plus edge sizes M, N, K ∈ {1, 15, 16, 17, 255, 257}, for every tile.
  - `test_gemm_tiles_agree` and `test_gemm_batch_independent`: a sample's columns (NN) or rows (NT, Linear) alone vs at offsets 0, mid, last in 8/144/256 with random companions: bitwise.
  - `test_gemm_ignores_c`: C pre-filled with NaN; output has no NaN.
- [ ] **Step 2: Implement** the spike's `gemm_tiled` with an NT staging variant (`B[n][k]` read as op(B)[k][n]) and the M-based tile rule (Linear: fixed 32×256). One kernel template; launchers in `kernels.cu`.
- [ ] **Step 3: Run** on the PC.

**Acceptance:** all bitwise tests pass; the Task 3 tests run in < 20 s.

**Commit:** `Add the CUDA gemm with a fixed fma order`.

---

### Task 4: The elementwise kernels

**Effort:** ~0.5 day.

**Files:** `src/cuda/kernels.cu/.cuh`, `src/cuda/cuda_probes.cu`, `include/azgo/cuda.hpp`, `tests/test_cuda.cpp`, `tests/cuda_reference.cpp`.

**Produces:** probes `transposeInput`, `im2col3`, `bnInfer(const BatchNorm2d&, x, M, y)`, `relu`, `addInto`, `policyFlatten`, `biasAddRows`, `tanhInPlace` with the Metal probes' signatures.

- [ ] **Step 1: Failing tests.**
  - Bitwise against the CPU: `transposeInput` and `policyFlatten` (copies), `im2col3` vs `Backend::im2col` (k = 3; boards 5, 9, 13, 19; batches 1, 8, 144), `relu` (including −0, NaN, ±inf, denormals) vs `reluForward`, `addInto`, `biasAddRows`.
  - `bnInfer` bitwise vs `reference::bnInferNoContract` (`scale = gamma / std::sqrt(var + eps)`, `shift = beta - scale * mean`, `y = scale * x + shift`, in the `-ffp-contract=off` unit), with random statistics and a near-zero variance.
  - `tanhInPlace` within 2 ulp of `(float)std::tanh((double)x)` over 10⁶ values in [−20, 20] and the special values (±0, ±inf, NaN).
  - Batch independence of each kernel (alone vs in a batch).
- [ ] **Step 2: Implement** (`__fmul_rn`/`__fadd_rn` where the CPU writes `a * b + c`; `__fdiv_rn`, `__fsqrt_rn` in `bn_fold`/`bn_infer`).
- [ ] **Step 3: Run** on the PC.

**Acceptance:** all pass.

**Commit:** `Add the CUDA forward layers' elementwise kernels`.

---

### Task 5: The fused convolution

**Effort:** ~0.5 day.

**Files:** `src/cuda/kernels.cu/.cuh`, `src/cuda/cuda_probes.cu`, `tests/test_cuda.cpp`.

**Produces:** `conv_fused` (template on the tile shape; the B-tile staged from `[C][B][n][n]`, one fixed column per thread; epilogue `__fadd_rn(__fmul_rn(scale[m], acc), shift[m])`, `+ skip`, `v > 0 ? v : 0`), `bn_fold` (gamma, beta, mean, var → scale, shift, as `bn_infer`); probe `convBnRelu(const Conv2d&, const BatchNorm2d&, x, skip, B, n, y)`.

- [ ] **Step 1: Failing tests.**
  - `test_conv_fused_equals_explicit`: bitwise against `im2col3` → `gemm` → `bnInfer` → (`addInto`) → `relu` on the GPU, for C_in ∈ {18, 64, 96, 128}, C_out ∈ {64, 96, 128}, boards 5/9/13/19, batches 1/8/144/256, with and without skip, every tile.
  - `test_conv_fused_batch_independent`: alone vs at three offsets in 8/144/256, permuted companions.
- [ ] **Step 2: Implement** from `tools/cuda_bench/cuda_bench.cu` (`gemm_tiled<…, true>`), with `bn_fold` replacing the spike's `fmaf` epilogue.
- [ ] **Step 3: Run** on the PC.

**Acceptance:** all pass.

**Commit:** `Add the fused implicit-GEMM convolution`.

---

### Task 6: Weights, workspace and the forward pass

**Effort:** ~1 day.

**Files:** Create `src/cuda/cuda_network.cu/.cuh`. Modify `include/azgo/cuda.hpp`, `src/cuda/cuda_probes.cu`, `tests/test_cuda.cpp`.

**Produces:** `CudaWeights(ParamLayout)` with `upload(const Network&)` (name/size checks; `bn_fold` per BN after upload), `CudaWorkspace(cfg, maxBatch)` (stage, input, three trunk buffers, head buffers, outputs; no `col`), `CudaNetwork::encodeForward(cudaStream_t, CudaWorkspace&, const CudaWeights&, int B)` in `Network::forward`'s order (spec "Overview"); probes `InferNet` (synchronous `infer(input, B, logits, values)`), `checkUpload`, `checkForwardReadsEveryTensor` (ownership tensors exempt, by name).

- [ ] **Step 1: Failing tests** (mirroring `test_metal`'s network tests).
  - `test_network_matches_cpu`: every preset shape (5×5 3×32, 9×9 6×64, 9×9 6×96, 13×13 6×96, 19×19 10×128, and one with `ownership = true`), random weights with trained-like BN statistics, logits and values against `Network::infer` on naive (and blas when built) within the Metal spec's tolerances.
  - `test_network_batch_independent`: positions alone vs in 8/144/256, offsets, random companions, permutation: bitwise.
  - `test_network_repeatable`: two passes bitwise.
  - `test_upload_refusals`: a network with an unknown tensor → `std::logic_error` naming it; `checkForwardReadsEveryTensor` names a tensor a lagging pass never reads.
- [ ] **Step 2: Implement.**
- [ ] **Step 3: Run** on the PC.

**Acceptance:** all pass; 13×13 6×96 at B = 256 through `InferNet` already ≥ 25,000 evals/s synchronous (print it; not asserted).

**Commit:** `Run the whole inference pass on the CUDA GPU`.

---

### Task 7: The batch runner and the hub

**Effort:** ~1 day.

**Files:** Create `src/cuda/cuda_runner.cu`. Modify `include/azgo/cuda.hpp`, `src/inference_hub.cpp` (`WorkerEvaluators` cuda branch), `src/cuda/cuda_probes.cu`, `tests/test_cuda.cpp`.

**Produces:** `std::unique_ptr<BatchRunner> makeCudaRunner(const Network&, int maxBatch, int slots = 2)`; `void checkCudaInferenceMemory(const NetConfig&, int inferenceBatch, int hubs, int slotsPerHub = 2)`; probes `setHubChaosSeed/hubChaosSeed`, `failPassesAfter(int)`, `setMemoryLimit(size_t)`, `passesCompleted()`.

- [ ] **Step 1: Failing tests.**
  - `test_runner_slots`: one and two slots, B = 1…maxBatch; outputs equal `InferNet` bitwise.
  - `test_hub_chaos`: 64 threads × 200 batch-8 requests, chaos seeds 1–5, one and two slots: each request's rows equal its positions alone, bitwise; no request lost.
  - `test_hub_failure_reaches_everyone`: `failPassesAfter(3)`: every waiting and later request throws "GPU inference failed: cuda: …"; the hub's destructor returns.
  - `test_memory_check`: a lowered limit makes `checkCudaInferenceMemory` and `makeCudaRunner` throw "lower inference_batch" before allocating.
  - `test_runner_no_thread_leak`: 50 runners made and destroyed (some mid-flight); the `/proc/self/task` count returns to its start.
  - `test_worker_evaluators_cuda`: `WorkerEvaluators(net, "cuda", 8, 8, 256)` evaluates as `InferNet` bitwise; `evaluations()` counts positions.
- [ ] **Step 2: Implement** per the spec's "The runner": per-slot stream, pinned staging (`cudaHostAlloc`), device workspace, event; `begin` enqueues H2D, forward, D2H, event record and hands `(slot, done)` to the completion thread; the completion thread synchronizes the event, gathers the first error, calls `done`; destructor drains and synchronizes before freeing. No CUDA call inside any callback.
- [ ] **Step 3: Run** on the PC; then `test_cuda` 20 times in a loop (`for i in $(seq 20)`), and 4 copies at once, to shake out races.

**Acceptance:** all pass in all loops.

**Commit:** `Add the CUDA batch runner behind the inference hub`.

---

### Task 8: Self-play with `backend = cuda`, end to end

**Effort:** ~0.75 day.

**Files:** Modify `apps/selfplay.cpp` (accepts `--backend cuda`; `--fingerprint --backend cuda`), `src/coordinator.cpp` (local cuda engine: memory check like metal's), `tests/test_cuda.cpp`, `tests/integration_smoke.sh` (a `cuda=ON|OFF` argument from CMake, like `metal=`), `CMakeLists.txt`.

- [ ] **Step 1: Failing tests.**
  - `test_selfplay_slice_independent_of_workers`: `playSelfPlayGames` (5×5 tiny net and 9×9 6×64, 8 games, 32 simulations) with `EngineRun{"cuda", w, batch}` for w ∈ {1, 3, 8}, batch ∈ {8, 256}, chaos on/off: records byte-identical (serialized with `writeGameRecord`).
  - `test_selfplay_job_twice`: a job file through `runSliceJob` with `--backend cuda` twice → identical result files.
  - Smoke, CUDA build: a tiny run with `selfplay_engines = local/naive:1, local/cuda=2`, 2 generations, resumed to 3; equal to the same run done straight; the manifest's cuda rows carry `cuda; `-fingerprints. Other builds: `selfplay --fingerprint --backend cuda` exits 1 with the not-built message, and a run naming `local/cuda` is refused before writing anything.
- [ ] **Step 2: Implement.**
- [ ] **Step 3: Run** PC three builds; Studio three builds.

**Acceptance:** all pass.

**Commit:** `Play self-play slices on the CUDA GPU`.

---

### Task 9: Measuring it: `bench --backend cuda` and `--mixed`

**Effort:** ~0.5 day.

**Files:** Modify `apps/bench.cpp`, `tests/integration_smoke.sh` (bench rows in a CUDA build).

- [ ] **Step 1: Failing smoke checks:** `bench --sizes 5 --backend cuda --seconds 0.1` prints a `cuda` row, a raw-pass table and a hub table; `bench --mixed --sizes 5 --hub-backend cuda --cpu-workers 0,1 --seconds 0.1` prints both tables. Without CUDA, both refuse with the not-built message.
- [ ] **Step 2: Implement** by generalizing the metal tables over the hub backend (`metal` or `cuda`); `--hub-backend` defaults to the one built (`metal` on a Mac).
- [ ] **Step 3: Measure on the PC** (idle), 13×13 6×96 (`configs/9x9.cfg` with `board_size = 13`, `filters = 96`), 9×9 6×96, 19×19 10×128: raw passes at 8/144/256, hub at 64/128/256/512 threads; `--mixed` with 0, 2, 4, 8 blas workers beside the hub. Append to `docs/results/cuda-bench.txt` with the conditions.

**Acceptance:** the spec's raw-pass and hub targets are met (13×13 6×96: ≥ 28,000 raw, ≥ 25,000 hub), or the shortfall is explained in the results file with its cause measured (GPU vs CPU side), and Mark decides whether to continue.

**Commit:** `Measure the CUDA engine with bench`.

---

### Task 10: Default path identity, old against new

**Effort:** ~0.25 day.

- [ ] On the PC: build `main` (the commit this branch started from) and this branch's `build` and `build-blas` from `git archive`; run the 5×5 preset (3 generations, resume) and 9×9 (2 generations) with naive and blas, compare checkpoints, `state.bin`, results, SGF and `log.csv` without timings: identical.
- [ ] On the Studio: confirm `build-metal` of this branch reports the same metal fingerprint as `main` and its `test_metal` passes (no runs: the Studio is training).
- [ ] Record the procedure and the result in `docs/results/cuda-bench.txt`.

**Commit:** `Check that the CUDA branch leaves the default path unchanged`.

---

### Task 11: The engine in the cluster

**Effort:** ~0.5 day (+ waiting for a generation).

**Files:** Modify `tools/cluster_sync.sh` (word `cuda` → `-DAZ_USE_CUDA=ON`; cuda wins over blas), `tests/integration_smoke.sh` (dry-run line for `build-cuda`), `README.md` (Build: the CUDA build; Distributed self-play: `pc/cuda`, `cuda_workers`, nvcc on the remote PATH; Extending to the GPU: the CUDA paragraph), the explainer's GPU chapter note, `tools/check_citations.py` run.

- [ ] **Step 1: Failing smoke check:** the cluster_sync dry run gives `-DAZ_USE_CUDA=ON` for `build-cuda` and `build_cuda_2`, not for `build-nocuda`.
- [ ] **Step 2: Implement** and document.
- [ ] **Step 3: Manual, with Mark's go-ahead** (the 13×13 run is live): sync the PC (`exe=~/azgo-cluster/build-cuda/selfplay` in the real hosts file, which stays out of the repo); `selfplay --fingerprint --backend cuda` over ssh; a scratch 13×13 run on a copy of the latest checkpoint with `selfplay_engines = local/metal:W1, mini1/metal:W2, mini2/metal:W3, pc/cuda:W4` (weights from `tools/engine_weights.py` and the Task 9 rates), one generation, resumed once: it verifies, merges and resumes byte for byte; the `pc/cuda` slice's evals/s ≥ 28,000 (≥ 2× local/metal's in the same generation).
- [ ] **Step 4:** Mark decides when the live run switches (`tools/stop_after_generation.sh`, then `--resume --set selfplay_engines=…`); not part of this plan's acceptance.

**Acceptance:** the scratch generation passes; the measured rates are added to `docs/results/cuda-bench.txt`.

**Commit:** `Sync and document the CUDA engine for distributed self-play`.

---

## Task Summary

| # | task | effort | proves |
|---|---|---|---|
| 1 | build option, context, `fill` | 0.5 d | no PTX; skip with 77; not-built messages |
| 2 | `cuda` as a backend name | 0.25 d | config, engines, hosts, fingerprint |
| 3 | gemm NN/NT | 0.75 d | bitwise = fma chain; batch-independent |
| 4 | elementwise kernels | 0.5 d | bitwise = CPU; tanh ≤ 2 ulp |
| 5 | fused convolution | 0.5 d | bitwise = explicit sequence |
| 6 | weights and forward | 1 d | whole net within tolerance; batch-independent |
| 7 | runner and hub | 1 d | chaos, failures, lifetime |
| 8 | self-play end to end | 0.75 d | byte-identical across workers/batches |
| 9 | bench | 0.5 d | throughput targets |
| 10 | default path identity | 0.25 d | nothing else changed |
| 11 | cluster | 0.5 d | a real generation with `pc/cuda` |

About 6.5 days of work. Tasks 3–5 are independent once Task 1 is in and can run in parallel; 6 needs 3–5; 7 needs 6; 8–9 need 7; 10 can run any time after 2; 11 last.
