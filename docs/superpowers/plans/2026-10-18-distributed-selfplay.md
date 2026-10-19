# Distributed Self-Play Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Self-play uses every engine Mark has: the coordinator's GPU and CPU at once, and the GPUs and CPUs of remote Macs over ssh. Each game of a generation is assigned up front, by index, to a fixed engine (machine, backend) from a new `selfplay_engines` key. Remote slices are played by a new `selfplay` app, verified, merged in index order, and the generation continues as today. Without the key, runs are byte-identical to `main`.
**Architecture:** Today's self-play loop becomes `playSelfPlayGames(net, sp, seed, g, first, count, EngineRun)`. The default path calls it once for [0, G). `assignGames` turns the key into contiguous slices with integer largest-remainder apportionment. A `SelfPlayCoordinator` runs each local engine's slice on its own thread with its own `WorkerEvaluators`. It runs remote slices through a `Transport` (ssh/scp, a local subprocess for tests, or in-process for unit tests), with binary job and result files, checksums, replay verification, a retry, then a fallback or stop. Every slice is recorded in `selfplay-engines.csv`. A machine-local hosts file says how to reach and drive machines. The build ID and engine fingerprints make every game's provenance exact.
**Tech Stack:** C++17, CMake ≥ 3.16, POSIX (`posix_spawn`, `waitpid`, `kill`), `pthread_set_qos_class_self_np` on macOS, OpenSSH `ssh`/`scp`, bash, Python 3 (standard library) for the weights tool, assert-based test executables under CTest.
**Spec:** docs/superpowers/specs/2026-10-18-distributed-selfplay-design.md
**Author:** Mark Castelluccio, with planning assistance from Claude (Anthropic).

## Global Constraints

- **The default path is byte-identical to `main`.** With `selfplay_engines` empty, every file a run writes, apart from `config.cfg`'s two new lines and timings, is bit for bit what `main` writes, for naive, blas and metal. The existing determinism tests in `test_trainer`, `test_selfplay`, `test_replay_buffer`, `test_run_state` and `test_metal` must pass unchanged. Task 8 checks old against new binaries.
- **Assignment never depends on timing.** No code may choose an engine for a game from timings, load or completion order. Only `assignGames(config, G)` and the failure policy decide. Merging is by game index only.
- **The algorithm does not change.** Self-play, search, resignation, training and the arena stay AlphaGo Zero's. No new randomness is drawn from any run stream. The only randomness added (attempt nonces in file names) never touches `deriveSeed` streams.
- **No host names, user names, IP addresses or absolute home paths** in any committed file: code, tests, configs, docs, results. Use aliases (`mini1`) and placeholders (`<user>@<address>`). Before each commit, run `git diff --cached | grep -nE '([0-9]{1,3}\.){3}[0-9]{1,3}|/Users/|\.local\b'`, and grep the same diff for the remote service user's name, which this plan deliberately does not spell out. It must print nothing, except IP-like version strings in `bench.txt`, which must be checked by eye.
- **Never touch `runs/`** on the Studio. A 19×19 run trains there. Build with `-j 4`. No timing measurement on the Studio without Mark's go-ahead and the job paused (SIGSTOP/SIGCONT). The minis are idle and may be used for builds, tests and measurements.
- **No shell on the coordinator.** Processes start through `posix_spawn` with argument vectors. The single remote command string passed to `ssh` is built only from hosts-file values validated against `[A-Za-z0-9_./~@:+-]` and from paths the coordinator generated.
- Tests: assert-based executables compiled with `-UNDEBUG`. `test_distributed` runs in every build, in under ~20 s. Tests needing BLAS or Metal skip their part cleanly when not built. All randomness in tests comes through `azgo::Rng` with fixed seeds.
- New files open with a teaching comment and `Mark Castelluccio, with assistance from Claude (Anthropic).` Changed headers keep their opening comment accurate.
- Code style: camelCase functions, PascalCase types, `trailing_` members, `kConstants`, 4-space indent, ≤ 100 columns.
- Commits: author Mark Castelluccio <markacastelluccio@gmail.com> (repo-local config); imperative subject; the body explains why; the message ends with `Co-Authored-By: Claude <model> <noreply@anthropic.com>`, naming the model that writes it.

## Review Focus

1. **Default-path identity.** The refactor of the self-play loop (Task 3) and of game serialization (Task 2) must not change a bit. Tests: Task 2 `test_state_bytes_unchanged_by_record_refactor`; Task 3 `test_slice_equals_part_of_whole`, the existing `test_trainer` suite; Task 7 `test_single_local_engine_equals_default`; Task 8 old-vs-new binaries.
2. **Assignment and merge.** Integer apportionment, contiguity, no timing dependence, every index exactly once. Tests: Task 4 `test_assign_*`; Task 14 `test_merge_order_independent`, `test_fake_remote_equals_local`.
3. **Verification.** A wrong, stale, truncated or foreign result must never reach the buffer. Tests: Task 10 `test_result_refusals` (every check), Task 15 `test_corrupt_result_is_retried`.
4. **Failure policy and commit point.** A failing slice leaves `state.bin` at g − 1 under `stop`. Under `local`, the manifest records exactly who played what. Tests: Task 15.
5. **Process handling.** No zombie, no leaked descriptor, timeouts kill the child, a missing binary fails cleanly. Tests: Task 12.
6. **Privacy.** No host data in the repo (the grep above in every task's acceptance).

---

## Notes Before You Start

**Read first:** the spec; `src/trainer.cpp` (`runGeneration`, steps 1 and "Validation loss"); `src/selfplay.cpp`; `include/azgo/inference_hub.hpp` (`WorkerEvaluators`); `src/replay_buffer.cpp` (`save`/`load`); `src/run_state.cpp` (window checks, `fileHash`); `src/config.cpp` (the key table, `checkConsistency`); `apps/train.cpp`, `apps/cli.hpp`; `tests/test_trainer.cpp` (`tinyConfig`, `TempDir`, the byte-identity tests); `tests/integration_smoke.sh`; `docs/results/bench.txt` (19×19 hub and CPU rows; the old-vs-new procedure).

**Decisions binding on every task:**

- *`EngineSpec`* is declared in `include/azgo/config.hpp` (it is config data): `struct EngineSpec { std::string machine, backend; int weight = 0; int games = -1; std::string name() const; }`. Exactly one of `weight > 0` and `games >= 0` holds. `name()` is `machine + "/" + backend`. The same engine may appear more than once; items are identified by list position (the manifest's `first` column tells them apart). `Config::selfplayEngines` is a `std::vector<EngineSpec>` (empty = default path). `Config::selfplayFallback` is a `std::string`, `"local"` or `"stop"`.
- *Canonical printing* of the key: items joined by `", "`, each `machine/backend:weight` or `machine/backend=games`. Parsing accepts any whitespace around separators. Print-parse round-trips exactly.
- *`playSelfPlayGames`* lives in `include/azgo/selfplay_slice.hpp` / `src/selfplay_slice.cpp`, together with `EngineRun`, `SliceStats`, the job and result formats and `runSliceJob`. Engine parsing, `assignGames`, the hosts file, worker rules and fingerprints live in `include/azgo/engines.hpp` / `src/engines.cpp`. Processes live in `include/azgo/subprocess.hpp` / `src/subprocess.cpp`. Transports, the coordinator, the manifest and the probes live in `include/azgo/coordinator.hpp` / `src/coordinator.cpp`.
- *Build ID* is `azgo::buildId()` (full 40-hex commit, plus `+xxxxxxxx` when dirty, or `unknown`). Short form for printing: `shortBuildId()`, 7 hex plus the dirty suffix. `buildOptions()` is `"metal"`, `"blas"` or `"naive"`, the strongest backend built.
- *Fingerprint* is `engineFingerprint(backend)`, built from `backend`, `sysctlbyname("machdep.cpu.brand_string")` (or `/proc/cpuinfo`'s model name, else `uname -m`), `kern.osproductversion` + `kern.osversion` (or `uname -sr`), and with metal `metalDescription()`. Example: `metal; Apple M5 Max; macOS 27.0.1 (27A123); Apple M5 Max, macOS 27.0.1, shaders b7b31a09`. It contains no host name.
- *Workers per engine*: `metal`: `metal_workers`, where 0 means every game of the slice (≤ `kMaxArenaWorkers`). CPU: `cpu_workers`, where 0 means `max(1, min(4, cores / 3))` on `local` and `max(1, min(2, cores − 2))` on remotes (task 9 set both from `bench --mixed` on an M4 mini and on the Studio, replacing the first `max(1, cores / 3)` and `max(1, cores − 2)`). Task 9 may change these two defaults from measurements, and must update the spec when it does. Always capped at the slice's games.
- *Trainer options*: `Trainer::startNew(cfg, runDir, init, log, opts)` and `Trainer::resume(runDir, overrides, log, opts)` take a new trailing `const TrainerOptions& opts = {}` with `std::string hostsFile` (empty = the search order). Existing callers compile unchanged.
- *Remote paths*: `<workdir>/<run-id>/`, where `<run-id>` = run directory basename (characters outside `[A-Za-z0-9_.-]` replaced by `_`) + `-` + 16 hex of FNV-1a of its absolute path. Files: `net_<16 hex hash>.azn`, `g<NNNN>_<first>_a<attempt>.job` and `.result`. Locally: `runDir/selfplay-work/gen_NNNN/`.
- *Probes* (tests only): `namespace azgo::distributed_probe { enum class Fail { Exit, Timeout, Corrupt }; void failSlice(const std::string& engine, int times, Fail); void setRemoteBuildId(const std::string&); void reset(); }`. They act in the coordinator, before or after a transport call. They are never reachable from config.

**Spec ambiguities resolved here:**

- The fallback engine when several `local/...` engines exist is the *first listed*. The fallback slice runs with that machine's worker rule for that backend.
- `selfplay_fallback = stop` with an unreachable remote at preflight is a `ConfigError` at start or resume. With `local`, it is a warning line `warning: mini1 unreachable (<reason>); its slices fall back to local/metal`, and every generation still tries it first.
- The note on a changed engine list is printed when `selfplay_engines` differs from the saved `config.cfg` (canonical form compared), in the same style as the backend note.
- Slices with 0 games are skipped: no job, no manifest row.
- `timeout` counts from the start of the `ssh` run, not including uploads.

## File Structure

```
CMakeLists.txt                                   T1 build-info generation; T11 selfplay app; T11 test deps
cmake/azgo_build_info.cmake, cmake/build_info.hpp.in   T1
include/azgo/build_info.hpp, src/build_info.cpp  T1 buildId(), shortBuildId(), buildOptions()
include/azgo/replay_buffer.hpp, src/replay_buffer.cpp  T2 writeGameRecord / readGameRecord
include/azgo/selfplay_slice.hpp, src/selfplay_slice.cpp  T3 playSelfPlayGames, EngineRun, SliceStats; T10 job/result; T11 runSliceJob
include/azgo/inference_hub.hpp, src/inference_hub.cpp   T3 WorkerEvaluators::evaluations()
include/azgo/config.hpp, src/config.cpp          T4 EngineSpec, selfplay_engines, selfplay_fallback
include/azgo/engines.hpp, src/engines.cpp        T4 parse/print/assignGames; T5 hosts file, worker rules; T6 fingerprint
include/azgo/subprocess.hpp, src/subprocess.cpp  T12 runProcess
include/azgo/coordinator.hpp, src/coordinator.cpp  T7 local engines, manifest; T13 transports; T14 remote slices; T15 failures, probes
include/azgo/trainer.hpp, src/trainer.cpp        T3 use playSelfPlayGames; T7 coordinator, note, TrainerOptions; T14 preflight
apps/train.cpp                                   T1 --version; T14 --hosts
apps/selfplay.cpp                                T11
apps/bench.cpp                                   T9 --mixed
tests/test_distributed.cpp                       T1, T4-T6, T10-T15 (new)
tests/test_replay_buffer.cpp, test_run_state.cpp T2
tests/test_selfplay.cpp                          T3
tests/test_config.cpp                            T4
tests/test_trainer.cpp                           T3, T7, T14, T15
tests/test_metal.cpp                             T7
tests/integration_smoke.sh                       T16
tools/cluster_sync.sh, tools/stop_after_generation.sh, tools/engine_weights.py   T17
docs/results/bench.txt                           T8, T9, T18, T19
README.md, docs/algorithm-explained.md, .html    T20
```

## Conventions for Every Task

**Builds** (from the worktree root; `-j 4` on the Studio while the run trains):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake -S . -B build-blas -DCMAKE_BUILD_TYPE=Release -DAZ_USE_BLAS=ON
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DAZ_USE_METAL=ON
cmake --build build-metal -j 4 && ctest --test-dir build-metal --output-on-failure
```

At the end of every code task, all three builds compile and pass `ctest`. A test that uses something not written yet fails at compile time, which is the expected "fails" state.

**TDD cycle:** write the listed tests; build and watch them fail; implement; pass; run all three full suites; run the privacy grep; commit.

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

# Phase 1 — Several engines on one machine

### Task 1: Build identity

**Effort:** ~0.25 day.

**Files:** Create `cmake/azgo_build_info.cmake`, `cmake/build_info.hpp.in`, `include/azgo/build_info.hpp`, `src/build_info.cpp`, `tests/test_distributed.cpp`. Modify `CMakeLists.txt`, `apps/train.cpp`.

**Produces:** `std::string buildId()`, `shortBuildId()`, `buildOptions()`; `train --version` prints `train <short id> (<options>)`.

- [ ] **Step 1: Failing tests** (`tests/test_distributed.cpp`, new; `main` runs each test and prints `test_distributed: OK`).
  - `test_build_id_shape`: `buildId()` is `unknown`, or 40 lowercase hex, optionally followed by `+` and 8 hex. `shortBuildId()` is its first 7 characters plus the same suffix. Both are stable across calls.
  - `test_build_options`: one of `metal`, `blas`, `naive`, matching `metalBuilt()` and the BLAS define.
- [ ] **Step 2: Implement.** `cmake/azgo_build_info.cmake` is run by `add_custom_target(azgo_build_info ALL COMMAND ${CMAKE_COMMAND} -DSRC=... -DOUT=... -DOVERRIDE=${AZGO_BUILD_ID} -P ...)`. It runs `git rev-parse HEAD`, `git status --porcelain` and, if dirty, `git diff HEAD | cmake -E sha256sum` (first 8 hex). It configures `build_info.hpp.in` into `${CMAKE_BINARY_DIR}/generated/azgo_build_info.inc`, rewriting only on change (`configure_file` semantics via a temp file plus `copy_if_different`). `azgo` depends on the target. `AZGO_BUILD_ID` (cache, default empty) overrides git. With no git and no override: `unknown`. `train --version` exits 0.
- [ ] **Step 3: Run** the three builds. Check that a second `cmake --build` with no change relinks nothing (`ninja -n`/`make -q`, or timestamps).

**Acceptance:** all pass. `./build/train --version` prints the commit. Configuring with `-DAZGO_BUILD_ID=abc...` (40 hex) prints that. An untouched rebuild does not recompile `build_info.cpp`.

**Commit:** `Embed the git commit as a build ID`.

---

### Task 2: One serializer for a game record

**Effort:** ~0.5 day.

**Files:** Modify `include/azgo/replay_buffer.hpp`, `src/replay_buffer.cpp`. Tests `tests/test_replay_buffer.cpp`, `tests/test_run_state.cpp`.

**Produces:** `void writeGameRecord(BinaryWriter&, const GameRecord&, bool withOwnership = true, bool withResigned = true)` and `GameRecord readGameRecord(BinaryReader&, const std::string& what, bool withOwnership = true, bool withResigned = true)`. These are the per-game body of `ReplayBuffer::save/load`, moved verbatim, with the error text prefix `what` (`"replay buffer: game 7"`). `ReplayBuffer::save/load` call them.

- [ ] **Step 1: Failing tests.**
  - `test_game_record_round_trip`: records from real self-play (with and without resignation, a resigned game, ownership) written and read back are equal field by field, for all four flag combinations (old layouts included).
  - `test_state_bytes_unchanged_by_record_refactor`: a golden check. Build a buffer from fixed games, `save` it, and compare against bytes produced by an in-test copy of today's loop. (Copy the current `save` body into the test as `legacySave` before refactoring.)
  - `test_read_game_record_refusals`: the existing load refusals (bad board size, move, visit entry, ownership value, resigned flag) through `readGameRecord`, messages prefixed with `what`.
- [ ] **Step 2: Implement** (move the code; no behavioural change).
- [ ] **Step 3: Run** the three builds; every existing `test_run_state` and `test_trainer` byte test passes.

**Acceptance:** all pass. `git diff` of `replay_buffer.cpp` shows only the move and the calls.

**Commit:** `Factor the per-game serializer out of the replay buffer`.

---

### Task 3: Self-play as a slice of game indices

**Effort:** ~0.5 day.

**Files:** Create `include/azgo/selfplay_slice.hpp`, `src/selfplay_slice.cpp`. Modify `include/azgo/inference_hub.hpp`, `src/inference_hub.cpp`, `src/trainer.cpp`. Tests `tests/test_selfplay.cpp`, `tests/test_inference_hub.cpp`.

**Produces:**

```cpp
enum class CpuQos { Default, Utility };
struct EngineRun { std::string backend; int workers = 1; int inferenceBatch = 256; CpuQos qos = CpuQos::Default; };
struct SliceStats { double seconds = 0; long long positions = 0, evaluations = 0; };
std::vector<GameRecord> playSelfPlayGames(const Network& net, const SelfPlayConfig& sp, uint64_t seed,
                                          int generation, int first, int count, const EngineRun& run,
                                          SliceStats* stats = nullptr);
long long WorkerEvaluators::evaluations() const;   // positions evaluated so far, all workers
```

The trainer's step 1 becomes one call with `first = 0`, `count = numGames`, `{cfg_.backend, min(workers, numGames), cfg_.inferenceBatch}`. The loop body is today's: `Rng(deriveSeed(seed, g, kPhaseSelfPlay, i))`, stored at `i − first`. With `qos == Utility`, each worker thread calls `pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0)` once, before its first game (`#ifdef __APPLE__`; elsewhere nothing).

- [ ] **Step 1: Failing tests.**
  - `test_slice_equals_part_of_whole` (`test_selfplay`): naive 5×5 tiny network, G = 7. `playSelfPlayGames(…, 0, 7, workers 3)` against `(…, 2, 4, workers 1)` and `(…, 6, 1, workers 2)`: records equal field by field, resign lows included, with resignation on and off. With BLAS built, the same with blas.
  - `test_slice_stats`: positions = Σ plies, evaluations > 0 and equal to the evaluators' count, seconds ≥ 0.
  - `test_qos_changes_nothing`: `CpuQos::Utility` gives the same records.
  - `test_worker_evaluators_count` (`test_inference_hub`): `evaluations()` sums the LocalEvaluators, and is the hub's `positions` with metal.
- [ ] **Step 2: Implement**; replace the trainer's loop with the call (keep the comment text).
- [ ] **Step 3: Run** the three builds; the whole `test_trainer` passes unchanged.

**Acceptance:** all pass. The trainer's self-play block shrinks to building `sp` and one call.

**Commit:** `Lift self-play's game loop into a function over a slice of indices`.

---

### Task 4: The `selfplay_engines` and `selfplay_fallback` keys, and `assignGames`

**Effort:** ~0.5 day.

**Files:** Create `include/azgo/engines.hpp`, `src/engines.cpp`. Modify `include/azgo/config.hpp`, `src/config.cpp`. Tests `tests/test_config.cpp`, `tests/test_distributed.cpp`.

**Produces:** `EngineSpec` (config.hpp). In `engines.hpp`: `std::vector<EngineSpec> parseEngineList(const std::string&)` (throws `std::invalid_argument` with the reason), `std::string printEngineList(const std::vector<EngineSpec>&)`, `struct Slice { int engine, first, count; }`, `std::vector<Slice> assignGames(const std::vector<EngineSpec>&, int games)`. Config keys `selfplay_engines` (after `inference_batch`) and `selfplay_fallback`, both tunable. In `checkConsistency`: Σ fixed > `games_per_generation` is refused, and Σ fixed ≠ G with no weighted item. Duplicate engines are allowed.

- [ ] **Step 1: Failing tests.**
  - `test_engine_list_parse_print` (`test_distributed`): the spec's example round-trips; whitespace variants parse; an empty value gives an empty list. Refusals with reasons: an unknown backend, a bad alias (`Mini1`, `1mini`, 33 characters), weight 0, a negative or huge count, a missing `:`/`=`, a trailing comma. A repeated engine parses into two items.
  - `test_assign_largest_remainder`: the spec's worked example gives 52, 12, 10, 8, 10, 8. Weights 1:1:1 with G = 10 give 4, 3, 3 (tie to the earlier). Weights 50:10:10 with R = 72 give 52/10/10.
  - `test_assign_properties`: over 2,000 random lists (fixed `Rng` seed, 1–8 engines, weights 1–1000, counts 0–20, G 1–500, valid only): counts sum to G, slices are contiguous from 0 in list order, zero-game engines are absent, and the result equals a second call. Each weighted count is within 1 of its quota.
  - `test_assign_fixed_only`: fixed counts that sum to G are exact.
  - `test_config_selfplay_keys` (`test_config`): defaults (empty, `local`); `configToString` prints both; a run's old config text without them parses to the defaults; `checkConsistency` refusals; `selfplay_fallback` accepts only `local`/`stop`.
- [ ] **Step 2: Implement** (64-bit integer arithmetic for quotas).
- [ ] **Step 3: Run** the three builds.

**Acceptance:** all pass; `configToString` of a default config has exactly two more lines than `main`'s.

**Commit:** `Add the selfplay_engines key and integer game assignment`.

---

### Task 5: The machine-local hosts file

**Effort:** ~0.5 day.

**Files:** Modify `include/azgo/engines.hpp`, `src/engines.cpp`. Test `tests/test_distributed.cpp`.

**Produces:** `struct MachineSettings { std::string alias, ssh, exe, workdir; int metalWorkers = 0, cpuWorkers = 0, inferenceBatch = 0, nice = 0; CpuQos cpuQos; double timeout = 0; bool local() const; }`; `struct HostsFile { std::map<std::string, MachineSettings> machines; MachineSettings settings(const std::string& alias) const; }`. `settings` returns the defaults for `local` when it is absent, and throws `ConfigError` for an unknown remote. Also `HostsFile parseHostsText(const std::string& text, const std::string& source)`, `HostsFile loadHostsFile(const std::string& path)` (a missing file gives an empty `HostsFile`), `std::string resolveHostsPath(const std::string& cli)` (CLI, then `$AZGO_HOSTS`, then `$HOME/.config/azgo/hosts`), and `int engineWorkers(const MachineSettings&, const std::string& backend, int sliceGames, int cores)`.

- [ ] **Step 1: Failing tests.**
  - `test_hosts_parse`: the spec's example (with placeholders replaced by `u@h`-style test values) parses. Comments and blank lines are ignored. Defaults: `cpu_qos` is `utility` on `local` and `default` elsewhere.
  - `test_hosts_refusals`, each with `source:line: reason`: an unknown key, a repeated alias, `ssh` on `local`, a remote without `exe` or `workdir`, a bad integer, a value with a space, `;`, `$`, a backtick, `'` or `|`, and a bad alias.
  - `test_hosts_search_order`: CLI, then environment (`setenv` in the test), then `HOME`.
  - `test_engine_workers`: metal 0 gives slice games (capped at 256); CPU 0 gives `max(1, min(4, cores/3))` locally and `max(1, min(2, cores − 2))` remotely (since task 9); explicit values are capped at the slice's games.
- [ ] **Step 2: Implement.**
- [ ] **Step 3: Run** the three builds.

**Acceptance:** all pass; the privacy grep is clean.

**Commit:** `Read machine settings from a machine-local hosts file`.

---

### Task 6: Engine fingerprints

**Effort:** ~0.25 day.

**Files:** Modify `include/azgo/engines.hpp`, `src/engines.cpp`. Test `tests/test_distributed.cpp`.

**Produces:** `std::string engineFingerprint(const std::string& backend)`, which throws `std::invalid_argument` for an unavailable backend (`checkBackendAvailable`'s message).

- [ ] **Step 1: Failing tests.** `test_fingerprint`: for each available backend, it starts with `<backend>; `, contains the CPU model (non-empty) and the OS, has no newline or comma outside the metal description, and is stable across calls. For metal, when available, it contains `metalDescription()`. It does not contain the host name (`gethostname`). Unavailable backends throw.
- [ ] **Step 2: Implement** (`sysctlbyname` on Apple, else `/proc/cpuinfo` and `uname`).
- [ ] **Step 3: Run** the three builds.

**Acceptance:** all pass; printed once by the test for the record.

**Commit:** `Describe what determines an engine's arithmetic`.

---

### Task 7: Local engines at once, the manifest and the switch note

**Effort:** ~1 day.

**Files:** Create `include/azgo/coordinator.hpp`, `src/coordinator.cpp`. Modify `include/azgo/trainer.hpp`, `src/trainer.cpp`. Tests `tests/test_trainer.cpp`, `tests/test_metal.cpp`, `tests/test_distributed.cpp`.

**Produces:** `struct SliceReport { Slice slice; std::string planned, engine, fingerprint, build, note; SliceStats stats; }`; `class SelfPlayCoordinator` with constructor `(const Config&, const std::string& runDir, HostsFile, std::ostream* log)`, `void preflight()` (local engines only in this task: availability, metal memory, the engine lines), `std::vector<GameRecord> play(int g, const std::string& networkPath, const Network& net, const SelfPlayConfig& sp, int games)`, `const std::vector<SliceReport>& reports() const`. Also `void writeManifest(const std::string& runDir, int g, const std::vector<SliceReport>&)`, which rewrites `selfplay-engines.csv` atomically after dropping rows ≥ g. `TrainerOptions { std::string hostsFile; }`. The trainer uses the coordinator when `selfplay_engines` is non-empty. It writes the manifest just before `saveTrainingState`, prints the `selfplay engines:` line, and on resume prints the switch note. In this task a remote alias in the list throws `ConfigError("remote engines are not supported yet")`; Task 14 lifts that.

- [ ] **Step 1: Failing tests.**
  - `test_single_local_engine_equals_default` (`test_trainer`): `tinyConfig()` with `selfplay_engines = local/naive:1`, two generations, against the default. Every checkpoint, `state.bin`, `results.csv`, `elo.csv`, `references.csv`, the SGFs and `log.csv` without timing columns are identical. `selfplay-engines.csv` exists only in the engine run, with 2 rows.
  - `test_split_local_engines_equal_default`: `local/naive:2, local/naive=1` (so one engine is weighted and one fixed). Same comparison, with resignation on.
  - `test_engines_resume_is_byte_identical`: engines on, 4 generations straight against 2 + resume 2. Files and manifests (without timing columns) are identical.
  - `test_switch_on_at_resume`: a default run for 2 generations, then resume with `--set selfplay_engines=local/naive:1,local/naive:1` to 4. Generations 1–2 files equal the default run's. The note is printed once (captured `log`). `config.cfg` carries the new value. The manifest has rows for 3–4 only. Switching back prints the note again.
  - `test_manifest_rows` (`test_distributed`): `writeManifest` drops rows ≥ g, quotes the fingerprint, and keeps the header.
  - `test_metal_engine_workers_change_nothing` (`test_metal`, skipped without Metal): `local/metal:1` with hosts `metal_workers` 0, 8 and 64, plus the default metal run: identical files.
  - With BLAS built: `test_mixed_local_engines_deterministic`: `local/naive:1, local/blas=3`, run twice, gives identical files.
- [ ] **Step 2: Implement.** One `std::thread` per local engine, each calling `playSelfPlayGames` with its own `EngineRun` (workers from `engineWorkers`, `qos` from settings, `inferenceBatch` from settings or the run). The first exception is rethrown after joining all. Merge by index, checking each index is filled exactly once (a `std::vector<char>` filled-map).
- [ ] **Step 3: Run** the three builds.

**Acceptance:** all pass. The default path calls no coordinator code (checked by reading `runGeneration`).

**Commit:** `Play a generation's games on several local engines at once`.

---

### Task 8: Old against new binaries on the default path

**Effort:** ~0.25 day (mostly waiting; run on a mini if the Studio is busy).

**Files:** Modify `docs/results/bench.txt`.

- [ ] **Step 1.** Build `main` (the commit this branch started from) from `git archive` in the scratchpad: `build-metal`, Release, `-j 4`.
- [ ] **Step 2.** Run, old and new, as in the arena-GPU comparison: the 5×5 preset for 3 generations (40 games, 50 steps, 64 simulations; arena 10 games at 48 simulations; calibration 10; one bot) with naive, blas and metal, resumed at generation 2 with workers 3. Then the 9×9 preset with filters 96 for 2 generations (8 games, 50 simulations, 20 steps) with blas and metal.
- [ ] **Step 3.** Compare every checkpoint, `state.bin`, the rating files, the SGFs, `log.csv` without timings, and `config.cfg` with the two new lines removed (`grep -v '^selfplay_'`).

**Acceptance:** all identical; a section "Distributed self-play: default path, old against new" in `bench.txt` lists what was compared.

**Commit:** `Record that the default path is byte-identical to main`.

---

### Task 9: Measuring engines side by side (`bench --mixed`)

**Effort:** ~0.5 day, plus measurement windows.

**Files:** Modify `apps/bench.cpp`, `docs/results/bench.txt`; possibly `src/engines.cpp` and the spec (default worker rules).

**Produces:** `bench --mixed --sizes 19 [--hub-workers 36] [--cpu-workers 0,2,4,6,8,12] [--qos default|utility] [--seconds 10]`. For each k it runs the hub with N threads and k CPU BLAS workers at the same time, for `--seconds`, and prints `hub evals/s`, `cpu evals/s`, `total`, mean pass and slots busy. k = 0 is the hub alone. A second table gives the CPU workers alone, for k in the list.

- [ ] **Step 1: Failing test.** `integration_smoke.sh` gains `bench --mixed --sizes 5 --seconds 0.2 --hub-workers 2 --cpu-workers 0,1` on the metal build, checking that the table header and rows appear (skipped without Metal).
- [ ] **Step 2: Implement** (reuse the hub and CPU measurement helpers).
- [ ] **Step 3: Measure** on mini1 (idle, any time): 19×19 10×128, `--hub-workers 20`, `--cpu-workers 0,2,4,6,8`, with `--qos` default and utility. On the Studio only with Mark's go-ahead and the 19×19 job paused: `--hub-workers 36`, `--cpu-workers 0,2,4,6,8,12,18`, with both QoS settings.
- [ ] **Step 4.** If the measurements show a better default CPU worker rule (the knee of the CPU curve, while the hub keeps ≥ 90%), change `engineWorkers` and the spec's numbers, with the measurement cited.

**Acceptance:** the tables are in `bench.txt`. The estimate rows of the spec's Problem table are replaced by measurements, or annotated with them. The Studio hub's rate with the chosen local CPU workers is recorded against the ≥ 90% criterion.

**Commit:** `Measure GPU and CPU engines running side by side`.

---

# Phase 2 — Remote slices

### Task 10: Job and result files, and their verification

**Effort:** ~1 day.

**Files:** Modify `include/azgo/selfplay_slice.hpp`, `src/selfplay_slice.cpp`. Test `tests/test_distributed.cpp`.

**Produces:**

```cpp
struct SelfPlayJob {
    uint64_t seed = 0; int generation = 0, first = 0, count = 0;
    SelfPlayConfig sp;                 // maxMoves resolved
    std::string networkFile; uint64_t networkHash = 0;
    std::string buildId; int inferenceBatch = 256;
};
std::string serializeJob(const SelfPlayJob&);            // "AZSJ", version 1
SelfPlayJob parseJob(const std::string& bytes);          // SliceError on any defect
uint64_t fnv1a64(const std::string& bytes);
struct SliceResult { uint64_t jobHash = 0; std::string buildId, fingerprint; std::vector<GameRecord> records; SliceStats stats; };
std::string serializeSliceResult(const SliceResult&, int first);       // "AZSR", version 1, trailing checksum
SliceResult readSliceResult(const std::string& path, const std::string& jobBytes, const std::string& expectedBuild);
struct SliceError : std::runtime_error { using std::runtime_error::runtime_error; };
```

Each record is written as `i32 index`, `writeGameRecord(w, g)`, `u8 resignDisabled`, and per side `u32 n` then `(i32 ply, f32 value)` × n.

- [ ] **Step 1: Failing tests.**
  - `test_job_round_trip`: every field bitwise, including floats such as `0.1f` and a v_resign of `-0.87f`.
  - `test_result_round_trip`: records from real self-play with resignation, all fields equal.
  - `test_result_refusals`, each a `SliceError` whose message names the defect: wrong magic; version 2; one flipped byte at 20 evenly spaced offsets (checksum); truncation at 10 points; another job's hash; another build ID; 3 records for count 4; indices out of order; a stone flipped in one position (recomputed checksum, so replay catches it); an illegal move; a wrong `resultForBlack` on a played-out game; a resigned game with the wrong winner; `resigned` with resignation off; lows rising; lows at ply 0.
  - `test_fnv1a64_matches_fileHash`: equal for the same bytes.
- [ ] **Step 2: Implement.** The replay check uses `replayFinalOwnership` and a `Game` replay with `sp.maxMoves` for the result.
- [ ] **Step 3: Run** the three builds.

**Acceptance:** all pass, with ≥ 30 refusal cases.

**Commit:** `Define verified job and result files for a self-play slice`.

---

### Task 11: The `selfplay` app

**Effort:** ~0.5 day.

**Files:** Create `apps/selfplay.cpp`. Modify `include/azgo/selfplay_slice.hpp`, `src/selfplay_slice.cpp` (`runSliceJob`), `CMakeLists.txt`. Test `tests/test_distributed.cpp`.

**Produces:** `int runSliceJob(const std::string& jobPath, const std::string& outPath, const std::string& backend, int workers, int inferenceBatch, std::ostream& err)`. It returns an exit code. It checks the network hash and the build ID against its own, loads the network, plays, and writes the result atomically. The app: `selfplay --job --out --backend [--workers] [--inference-batch]`, `--version`, `--fingerprint --backend B`, and `--help`, with exit codes 0, 1 and 2. In CMake, `selfplay` is added to the apps, and `test_distributed` gets `add_dependencies(test_distributed selfplay)` and `AZGO_SELFPLAY_EXE="$<TARGET_FILE:selfplay>"`.

- [ ] **Step 1: Failing tests.**
  - `test_run_slice_job_in_process`: write a job for a 5×5 tiny network saved to a temp dir. `runSliceJob` returns 0, and the result read back equals `playSelfPlayGames` for that slice.
  - `test_run_slice_job_refusals`: a network hash mismatch, a build ID mismatch, a missing network, and an unavailable backend each return 1 with the reason on `err`, and leave no output file.
  - `test_selfplay_exe_end_to_end`: run the real binary (`AZGO_SELFPLAY_EXE`) through a minimal `posix_spawn` + `waitpid` helper local to the test; Task 12 replaces it with `runProcess`. Its result is bitwise equal to the in-process one. `--version` prints `shortBuildId()`.
- [ ] **Step 2: Implement.**
- [ ] **Step 3: Run** the three builds.

**Acceptance:** all pass; `selfplay --help` documents every option.

**Commit:** `Add selfplay, which plays one slice of a generation from a job file`.

---

### Task 12: Running a child process

**Effort:** ~0.5 day.

**Files:** Create `include/azgo/subprocess.hpp`, `src/subprocess.cpp`. Test `tests/test_distributed.cpp`.

**Produces:** `struct ProcessResult { int exitCode = -1; int signal = 0; bool timedOut = false; std::string error; }`; `ProcessResult runProcess(const std::vector<std::string>& argv, const std::string& logPath, double timeoutSeconds, const std::string& stdinPath = "")`. It uses `posix_spawnp`, with stdout and stderr appended to `logPath` (or `/dev/null`) and stdin from `stdinPath` or `/dev/null`. It polls `waitpid(WNOHANG)` every 50 ms. On timeout it sends SIGTERM, then SIGKILL after 5 s, and always reaps. `POSIX_SPAWN_SETSIGMASK` gives the child a clean mask. File actions close every other descriptor (`posix_spawn_file_actions_addclosefrom_np` on Apple, otherwise close-on-exec is set on everything the coordinator opens).

- [ ] **Step 1: Failing tests.** `test_run_process`: `true` gives 0; `false` gives 1; `sh -c 'echo hi; echo err >&2; exit 3'` gives 3 with both lines in the log (the test may use `sh`; the coordinator does not); `sleep 10` with timeout 0.3 gives `timedOut` in under 6 s; a missing executable gives `error` set and exit −1; stdin redirection works (`cat`); 200 sequential runs leave no zombie (`waitpid(-1, WNOHANG)` returns −1/ECHILD) and no growth in open descriptors.
- [ ] **Step 2: Implement.**
- [ ] **Step 3: Run** the three builds. Task 11's end-to-end test switches to `runProcess`.

**Acceptance:** all pass.

**Commit:** `Add a small posix_spawn runner with timeouts`.

---

### Task 13: Transports

**Effort:** ~0.5 day.

**Files:** Modify `include/azgo/coordinator.hpp`, `src/coordinator.cpp`. Test `tests/test_distributed.cpp`.

**Produces:** `class Transport { virtual void upload(local, remote); virtual bool exists(remote); virtual ProcessResult run(const std::vector<std::string>& args, const std::string& log, double timeout); virtual void download(remote, local); virtual void remove(const std::vector<std::string>&); virtual void killStray(const std::string& runId); }` (errors as `SliceError`). Implementations:

- `SshTransport(MachineSettings)`: `scp -q -o BatchMode=yes …`; `ssh` with the spec's options and the command `exec nice -n N <exe> <args>`; `exists` via `ssh … test -f`; `killStray` via `pkill -f '<run-id>'`, best effort.
- `SubprocessTransport(MachineSettings)`: filesystem copies, and `runProcess({exe, args…})`.
- `InProcessTransport` (tests): `run` calls `runSliceJob` on the calling thread.

Also `std::unique_ptr<Transport> makeTransport(const MachineSettings&)`, and `std::vector<std::string> sshArgv(const MachineSettings&, const std::vector<std::string>& args)`, exposed for tests.

- [ ] **Step 1: Failing tests.**
  - `test_ssh_argv`: the exact vectors for a settings object with `nice=5` and `timeout`; the command string contains only validated pieces. A setting smuggled past validation (constructed directly in the test) makes `sshArgv` throw `std::logic_error`.
  - `test_subprocess_transport_round_trip`: upload, exists, run `selfplay --version`, download, remove.
  - `test_in_process_transport`: `run` with job args gives the same result file as the subprocess transport.
- [ ] **Step 2: Implement.**
- [ ] **Step 3: Run** the three builds.

**Acceptance:** all pass; no test opens a network connection.

**Commit:** `Add ssh, subprocess and in-process transports for slices`.

---

### Task 14: Remote slices in the coordinator

**Effort:** ~1–1.5 days.

**Files:** Modify `include/azgo/coordinator.hpp`, `src/coordinator.cpp`, `include/azgo/trainer.hpp`, `src/trainer.cpp`, `apps/train.cpp`. Tests `tests/test_distributed.cpp`, `tests/test_trainer.cpp`.

**Produces:** `preflight()` for remote machines: `--version` equal to `buildId()`, and `--fingerprint` per engine, with a note when a fingerprint differs from the manifest's last row for that engine. `play()` runs remote slices: upload the network if absent, write and upload the job, run, download, `readSliceResult`, and fill the report, while local engines run. It cleans up remote and local work files after merging. `train --hosts FILE`. The "not supported yet" refusal is gone.

- [ ] **Step 1: Failing tests.**
  - `test_fake_remote_equals_local` (`test_trainer`): a temporary hosts file defines `fake` (no `ssh`; `exe` = `AZGO_SELFPLAY_EXE`; `workdir` = a temp dir). `local/naive:1, fake/naive:1` for 3 generations with resignation on is byte-identical to the default naive run. The manifest names `fake/naive` with the same fingerprint as `local/naive`.
  - `test_fake_remote_resume`: 2 + resume 2 equals 4 straight (files and manifest without timings).
  - `test_merge_order_independent` (`test_distributed`, in-process transport): three slices, with `distributed_probe` delays making them finish in each of the 6 orders. Merged records are identical.
  - `test_preflight_refusals` (`test_trainer`): `setRemoteBuildId("0000…")` gives a `ConfigError` naming `fake` and both IDs, and nothing is written (the start case) or `config.cfg` is unchanged (the resume case). An alias missing from the hosts file is refused, as is `fake/metal` in a build without Metal.
  - `test_remote_cleanup`: after a generation, the fake workdir holds only the network, and `selfplay-work/` is empty.
- [ ] **Step 2: Implement.**
- [ ] **Step 3: Run** the three builds.

**Acceptance:** all pass; `test_trainer` stays under its previous runtime + 15 s.

**Commit:** `Play self-play slices on remote machines and merge them by index`.

---

### Task 15: Failures: retry, fallback, stop

**Effort:** ~1 day.

**Files:** Modify `include/azgo/coordinator.hpp`, `src/coordinator.cpp`. Tests `tests/test_trainer.cpp`, `tests/test_distributed.cpp`.

**Produces:** the spec's policy. A failed attempt triggers `killStray`, then one retry on the same engine. Under `local`, the fallback engine plays the slice at once, in-process, and its report has `engine`/`fingerprint` set to the fallback's and `note = "fallback: <reason>"`. The log prints the note line. Under `stop`, `SliceError` propagates out of `runGeneration` before anything of g is committed. The `distributed_probe` functions are implemented.

- [ ] **Step 1: Failing tests** (fake remote `fake/naive`, `local/naive`).
  - `test_one_failure_is_retried`: for each mode (Exit, Timeout with the host `timeout=0.5`, Corrupt), `failSlice("fake/naive", 1, mode)` gives files byte-identical to the failure-free run, a manifest without a note, and the attempt-1 file name used.
  - `test_two_failures_fall_back`: under `local` the run completes. The manifest row has `planned = fake/naive`, `engine = local/naive` and a `fallback:` note, and the log has the note line. With equal fingerprints (naive, same machine) the files equal the failure-free run. That is the expected special case, asserted to document it.
  - `test_fallback_to_different_backend` (BLAS builds): engines `local/blas:1, fake/naive:1`, two failures. The fallback runs on `local/blas`, so the record differs from the failure-free run, and a second run with the same failures reproduces it exactly.
  - `test_stop_keeps_last_good_state`: under `stop`, two failures make `runUntil(2)` throw at generation 2. `state.bin` is still generation 1, and `selfplay-engines.csv` has no row for 2. Resuming with the probe reset gives files byte-identical to an uninterrupted failure-free run.
  - `test_unreachable_at_preflight`: `exe` pointing at a missing file; under `stop` a `ConfigError`, under `local` a warning and fallback rows.
- [ ] **Step 2: Implement.**
- [ ] **Step 3: Run** the three builds.

**Acceptance:** all pass. Review focus 4 holds.

**Commit:** `Retry a failed slice, then fall back or stop, and record who played it`.

---

### Task 16: The smoke test

**Effort:** ~0.5 day.

**Files:** Modify `tests/integration_smoke.sh`.

- [ ] **Step 1: Write the new section** (it fails until the tools are built): `selfplay --version` equals `train --version`'s ID. `selfplay --fingerprint --backend naive` succeeds, and `--backend metal` fails with the not-built message in non-Metal builds. A temporary hosts file defines `fake` (subprocess transport, `exe=$BUILD/selfplay`, `workdir=$W/remote`). A tiny run with `--hosts` and `--set selfplay_engines=local/naive:1,fake/naive=2` trains 2 generations and resumes to 3. `selfplay-engines.csv` has 2 rows per generation, and its fingerprint columns match. The same run with `selfplay_engines=` gives an identical `state.bin` (naive, same machine). `AZGO_HOSTS` pointing to the file works without `--hosts`. An unknown alias stops with the message and writes nothing.
- [ ] **Step 2: Run** the three builds' ctest.

**Acceptance:** `integration_smoke` passes in all three builds.

**Commit:** `Smoke-test distributed self-play with a subprocess remote`.

---

### Task 17: Operations tools

**Effort:** ~0.5 day.

**Files:** Create `tools/cluster_sync.sh`, `tools/stop_after_generation.sh`, `tools/engine_weights.py`. Modify `tests/integration_smoke.sh` (tool checks).

**Produces:**

- `tools/cluster_sync.sh [--hosts FILE] [--build build-metal] [--dry-run]`: refuses a dirty tree. For each remote alias (ssh set) in the hosts file, it derives the repo as `exe`'s grandparent. It pipes `git archive HEAD` into `ssh … 'tar -x -C <repo>.new'`, swaps directories while keeping `<repo>/<build>`, configures with `-DAZ_USE_METAL=ON -DAZGO_BUILD_ID=$(git rev-parse HEAD)`, builds with `-j $(sysctl -n hw.ncpu)`, and compares `selfplay --version` with the local `train --version`. `--dry-run` prints the commands.
- `tools/stop_after_generation.sh RUN_DIR PID`: waits for `log.csv` to gain a row, then `kill -INT PID`, and prints the generation of that row (committed: `log.csv` is appended after `state.bin`).
- `tools/engine_weights.py RUN_DIR [--last N] [--games G] [--hosts FILE]`: the spec's suggestion and predicted finish times. It warns when an engine has fewer games than workers. It uses only the standard library and is run with `python3 -I`.

- [ ] **Step 1: Failing checks.** The smoke test runs `cluster_sync.sh --dry-run` against the temporary hosts file (no ssh alias: it prints "no remote machines") and checks `bash -n` on all scripts. It runs `engine_weights.py` on the manifest written by Task 16's run and checks it prints a `selfplay_engines =` line that `train` accepts (`--resume --set` on a copy). `stop_after_generation.sh` is run against a `sleep` PID while a fake `log.csv` gains a row, and the sleep must be killed.
- [ ] **Step 2: Implement.**
- [ ] **Step 3: Run** the three builds' ctest.

**Acceptance:** all pass; the scripts have usage text; the privacy grep is clean.

**Commit:** `Add tools to sync the minis, stop at a generation boundary and suggest weights`.

---

### Task 18: The real three-machine smoke (manual)

**Effort:** ~0.5 day.

**Files:** Modify `docs/results/bench.txt`.

- [ ] **Step 1.** On each mini, configure `build-metal` (the CLT-only build was already proven by Metal task A1). Write the hosts file outside the repo. Run `tools/cluster_sync.sh` and check that the version lines match.
- [ ] **Step 2.** Run a tiny 5×5 run (the smoke's tiny settings, but 24 games and blas/metal engines on all three machines: `local/metal:2, local/blas=4, mini1/metal:1, mini1/blas=4, mini2/metal:1, mini2/blas=4`) for 3 generations. Run it again into a second directory. Every file must be identical, manifests without timings.
- [ ] **Step 3.** Run 2 generations, then resume to 3. The files must equal the straight run's.
- [ ] **Step 4.** Failure: during generation 2, `ssh <mini1> pkill -f 'selfplay --job'` twice (once per attempt). The run continues, and the manifest and log show the fallback. Under `stop`, the run stops with `state.bin` at 1, and the resume equals the failure-free run.
- [ ] **Step 5.** Record the per-engine fingerprints (aliases only), seconds and evaluations. Note that the two minis' fingerprints are or are not equal.

**Acceptance:** all checks hold and are written up in `bench.txt` without host data.

**Commit:** `Record the three-machine smoke of distributed self-play`.

---

### Task 19: The 19×19 run: weights, the switch and acceptance

**Effort:** ~0.5 day of work plus generations of wall time. **Needs Mark's go-ahead** for the switch and for any Studio measurement.

**Files:** Modify `docs/results/bench.txt`, `docs/results/README.md` (the 19×19 section), the spec's Performance section (measured values).

- [ ] **Step 1.** From Task 9's measurements and `tools/engine_weights.py`, choose `selfplay_engines` for the run's `games_per_generation`. Write down the predicted per-engine finish times.
- [ ] **Step 2.** With Mark: build `build-metal` at the branch head on the Studio (`-j 4`), sync the minis, and use `tools/stop_after_generation.sh` to stop the run just after a commit. Resume it with `--set selfplay_engines=...` and check the note. Keep a copy of `state.bin` and `config.cfg` from before the switch, outside `runs/`, with Mark's agreement on where.
- [ ] **Step 3.** After 3 generations, compare `selfplay_s` with the last 3 generations before the switch, scaled by evaluations per generation (from `avg_length`; the manifest's evaluations after the switch). Check the acceptance criteria: ≥ 1.6× self-play; hub ≥ 90% (from Task 9); desktop usable (Mark's judgement); transfers ≤ 2% (manifest seconds against slice wall times). Adjust weights once with the tool if one engine's finish time is more than 10% behind the others.

**Acceptance:** the measured speedup and the tail per engine are written up. If the speedup is below 1.6×, say why (which engine, which tail) rather than tuning silently.

**Commit:** `Measure distributed self-play on the 19x19 run`.

---

### Task 20: Teaching material

**Effort:** ~0.5 day.

**Files:** Modify `README.md`, `docs/algorithm-explained.md`, `docs/algorithm-explained.html`; run `tools/check_citations.py`.

- [ ] **Step 1.** README: a "Distributed self-play" section after **Backends**, as the spec's "Teaching material" lists. Add `selfplay` to the tools, update the test count (`test_distributed`), and link the spec and plan under "Learn how it works".
- [ ] **Step 2.** Explainer: "More than one machine" in the self-play chapter, quoting `assignGames` and the merge, with `file:line` citations. HTML: one diagram (coordinator, engines, slices, merge), following the page's theme tokens.
- [ ] **Step 3.** `python3 tools/check_citations.py` passes, and the privacy grep is clean.

**Acceptance:** docs build and read correctly in light and dark mode; citations pass.

**Commit:** `Explain distributed self-play in the README and the explainer`.

---

# Phase 3 (not planned here) — the arena

The spec's "Later: the arena" outlines it. Plan it after Task 19 shows how much of a 19×19 generation remains outside self-play.

## Task Summary

| # | task | effort |
|---|---|---|
| 1 | Build identity | 0.25 d |
| 2 | One serializer for a game record | 0.5 d |
| 3 | Self-play as a slice of game indices | 0.5 d |
| 4 | `selfplay_engines`, `selfplay_fallback`, `assignGames` | 0.5 d |
| 5 | The hosts file | 0.5 d |
| 6 | Engine fingerprints | 0.25 d |
| 7 | Local engines at once, manifest, switch note | 1 d |
| 8 | Old against new binaries (default path) | 0.25 d |
| 9 | `bench --mixed` and measurements | 0.5 d |
| 10 | Job and result files, verification | 1 d |
| 11 | The `selfplay` app | 0.5 d |
| 12 | Running a child process | 0.5 d |
| 13 | Transports | 0.5 d |
| 14 | Remote slices in the coordinator | 1–1.5 d |
| 15 | Retry, fallback, stop | 1 d |
| 16 | Smoke test | 0.5 d |
| 17 | Operations tools | 0.5 d |
| 18 | Three-machine smoke (manual) | 0.5 d |
| 19 | The 19×19 run: weights, switch, acceptance | 0.5 d + wall time |
| 20 | Teaching material | 0.5 d |
| | **total** | **~11–11.5 days** |

Phase 1 (Tasks 1–9) is useful on its own: the Studio's CPU joins its GPU, which the bench estimates at up to ~1.3× self-play, before any remote work.
