# Pipelined Generations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** With `pipeline = on`, each generation's training and arena run beside the next generation's self-play, self-play always playing the best network of the last commit (a one-generation lag, AlphaGo Zero's evaluator lag fixed to a boundary); one commit per generation holds the next generation's games, so a crash anywhere resumes exactly. With `off` (the default) runs are byte-identical to `main`.
**Architecture:** `TrainingState` gains an optional pending self-play block (state.bin version 6, written only when present). The trainer's generation splits into processSelfPlay / B(g) (validation, buffer, training, checkpoint, arena) / commit, shared by the sequential and the pipelined loop; the pipelined loop runs SP(g+1) on a thread beside B(g). The coordinator becomes safe for a self-play phase and an arena phase at once (machine map under a mutex, per-kind kept networks, per-phase failure flags, a cancel).
**Tech Stack:** C++17, CMake, assert-based CTest executables, bash (`integration_smoke.sh`), Python 3 standard library.
**Spec:** docs/superpowers/specs/2026-10-23-pipelined-generations-design.md
**Author:** Mark Castelluccio, with planning assistance from Claude (Anthropic).

## Global Constraints

- **The default is byte-identical to `main`.** With `pipeline` absent or `off`, every file a run writes (but `config.cfg`'s new line and timings) is `main`'s, with and without engines, for naive, blas and metal; `state.bin` stays version 5.
- **Scheduling only.** No change to games, training, arena, seeds, resignation calibration, buffer or lr.
- **No host names, user names, IP addresses or absolute home paths** in any commit. Before each commit: `git diff --cached | grep -nE '([0-9]{1,3}\.){3}[0-9]{1,3}|/Users/|/home/'` and a grep for the remote service user's name; both must print nothing.
- **A live run trains on this machine.** Never touch `runs/` (read only), its processes, the minis, the PC's cluster/work/dev directories, or the real hosts file. Build with `nice -n 10`, `-j6`, in scratch directories. The PC only in `~/pipe-scratch`, CPU or tiny GPU checks, removed afterwards.
- New files open with a teaching comment and `Mark Castelluccio, with assistance from Claude (Anthropic).`
- Commits: author Mark; message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Off-path identity (Task 5's refactor of `runGeneration`; Task 8's old-vs-new binaries).
2. Crash windows and resume exactness (Tasks 2, 6, 7).
3. Thread safety of two coordinator phases at once (Task 4) and of the trainer's two threads (Task 6): no shared mutable state outside the listed mutexes; exceptions never leave a joinable thread.
4. The lag rule (Task 6's replay test).

---

## Task 1: the `pipeline` key

**Files:** `include/azgo/config.hpp`, `src/config.cpp`, `tests/test_config.cpp`.

- [ ] Test: default `off`, prints `pipeline = off` after `backend`; `on` round-trips; other values refused ("expected off|on"); tunable on resume.
- [ ] Implement `Config::pipeline` (bool), the table entry.
- [ ] Run `test_config`, commit.

## Task 2: pending self-play in `state.bin` (version 6)

**Files:** `include/azgo/run_state.hpp`, `src/run_state.cpp`, `tests/test_run_state.cpp`, `tools/buf.py`, `tools/pass.py`, `tools/side.py`.

- [ ] Tests: a state without pending games saves version 5 with today's bytes; with pending games version 6, round trip (records equal field by field, resign lows, resign-disabled flags, network, resignation, v_resign, max_moves); refusals (count 0, a v_resign outside [−1, 0], max_moves < 1, a damaged record, trailing bytes); `trainingStateGeneration` reads version 6.
- [ ] Implement `PendingSelfPlay` and `std::optional<PendingSelfPlay> pending` in `TrainingState`; save/load with `writeSliceRecord` / `readSliceRecord`.
- [ ] Python tools accept version 6 (the buffer's layout is version 5's).
- [ ] Run, commit.

## Task 3: GPU memory of phases at once

**Files:** `include/azgo/metal.hpp`, `src/metal/metal_runner.mm`, `src/metal/*.cpp` (not-built answer), `include/azgo/cuda.hpp`, `src/cuda/cuda_runner.cu`, `src/cuda/*.cpp`, `tests/test_metal.cpp`.

- [ ] Test (metal build): the group overload with one group equals the old call's verdict; a sum that cannot fit throws the usual message.
- [ ] Implement `checkMetalInferenceMemory(const NetConfig&, const std::vector<GpuHubs>&)` and the CUDA twin.
- [ ] Run, commit.

## Task 4: the coordinator with a self-play and an arena phase at once

**Files:** `include/azgo/coordinator.hpp`, `src/coordinator.cpp`, `tests/test_distributed_arena.cpp` (or `test_dynamic_selfplay.cpp`).

- [ ] Tests: with a fake remote in both lists, `play` (static and dynamic) on one thread and `playArena` waves on another give exactly the results each gives alone; the self-play network stays on the remote while arena waves come and go (and the other way round); `cancel()` stops a phase in flight with `SliceCancelled` (both kinds), and `resetCancel()` lets the next one run.
- [ ] Implement: `machines_` under a mutex (lookup and insertion); `down` and `keep` set under the machine's mutex; `keep` per phase kind; per-phase failure flag; `fingerprints_` under a mutex; `cancel()` / `resetCancel()` watched by `runSlices` and `playDynamic`; preflight's combined memory check when `cfg.pipeline`.
- [ ] Run `test_distributed*`, `test_dynamic_selfplay`, `test_trainer`, commit.

## Task 5: the sequential generation in parts (no behaviour change)

**Files:** `include/azgo/trainer.hpp`, `src/trainer.cpp`.

- [ ] Refactor `runGeneration` into `playSelfPlayPhase`, `processSelfPlay`, `trainAndRate` (validation, buffer, training, checkpoint, arena) and `commitGeneration`; the sequential loop uses a pending block when the state has one (pipeline switched off) and then writes no self-play manifest rows for it.
- [ ] Run `test_trainer` (every existing byte-identity test), commit.

## Task 6: pipelined iterations

**Files:** `include/azgo/trainer.hpp`, `src/trainer.cpp`, new `tests/test_pipeline.cpp`.

- [ ] Tests (see the spec's Testing section): straight vs resumed at every commit; crash at every point (a test hook `pipeline_probe::crashAt(point, generation)` throwing a `std::runtime_error`), resumed equals straight; the lag (replay of pending games from the expected checkpoint, and not from the next one); prologue; switching on and off; changed self-play keys note; manifests, pruning, `log.csv`, progress lines, `pending-selfplay.txt`; failures and cancellation; dynamic + fake remote; arena engines; blas.
- [ ] Implement: prologue, iteration (self-play thread, B on the trainer's thread, barrier, commit), cancellation both ways, the BLAS once-call, `pending-selfplay.txt`, resume (pruning, note), memory check in `checkBackend`.
- [ ] Run all tests, commit.

## Task 7: end to end

**Files:** `tests/integration_smoke.sh`.

- [ ] A pipelined run with a fake remote killed with SIGKILL at several moments, resumed each time, equals the straight run (checkpoints, `state.bin`, rating files, SGFs).
- [ ] Run ctest, commit.

## Task 8: evidence

- [ ] Old vs new binaries (pipeline off): 5×5 and 9×9; naive, blas, metal; default path, local engines, fake remote (static, dynamic); straight and resumed; identical files but `config.cfg`'s line.
- [ ] TSan build: `test_pipeline`, `test_dynamic_selfplay`, `test_distributed_arena`.
- [ ] Wall time: small 9×9 run, two CPU engines of different speeds, off vs on; training QoS/threads measurement repeated; record in `docs/results/bench.txt`.

## Task 9: documentation

**Files:** `README.md`, `docs/algorithm-explained.md`, `docs/algorithm-explained.html`, spec.

- [ ] README: a "Pipelined generations" subsection (the key, the schedule, the lag and AlphaGo Zero, what a commit holds, log semantics, switching, training threads, measured gain); usage line; explainer passage; `python3 tools/check_citations.py` clean.
- [ ] Commit.

## Task 10: CUDA build on the PC, review

- [ ] `git archive` to the PC's `~/pipe-scratch`, build with `-DAZ_USE_CUDA=ON`, full ctest (CPU, tiny GPU), remove the directory.
- [ ] A fresh-context review subagent on the branch (concurrency, deadlocks, crash windows, determinism); fix its findings.
