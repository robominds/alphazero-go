# Distributed Arena Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A generation's arena waves are played on several engines at once, as self-play's games are: a new `arena_engines` key assigns each wave's games by index to engines (local GPU or CPU, remote Macs, the PC's CUDA GPU); remote slices go through the existing transports and `selfplay` binary, are verified by replay, merged by index, and recorded in `arena-engines.csv`. Without the key, runs are byte-identical to `main`.
**Architecture:** `playMatches`' game pool becomes `playArenaGames(ranges, cfg, EngineRun, stats)` over game ranges; `playMatches` is that over whole matches plus `matchResult`. A wave's games are numbered match-major; `assignGames` cuts them into slices and `waveSlice` turns a slice into ranges. The coordinator (renamed `Coordinator`) generalises its remote slice code (upload networks, job, run, download, verify, retry, fallback) from "a self-play slice" to "a slice", and gains `playArena`. The `selfplay` binary dispatches on the job's magic (`AZSJ` or `AZAJ`).
**Tech Stack:** C++17, CMake, assert-based CTest executables, bash (`integration_smoke.sh`), Python 3 standard library (`engine_weights.py`).
**Spec:** docs/superpowers/specs/2026-10-20-distributed-arena-design.md
**Author:** Mark Castelluccio, with planning assistance from Claude (Anthropic).

## Global Constraints

- **The default path is byte-identical to `main`.** With `arena_engines` empty, every file a run writes, apart from `config.cfg`'s new line and timings, is bit for bit `main`'s, for naive, blas and metal. Existing tests pass unchanged (except for the class rename).
- **Self-play is unchanged** with `selfplay_engines` set: same files, names, manifest and bits.
- **Assignment never depends on timing.** Only `assignGames(arena_engines, T)` and the failure policy decide; merging is by wave index.
- **The algorithm does not change.** Same games, opponents, thresholds, simulations, seeds.
- **No host names, user names, IP addresses or absolute home paths** in any commit. Before each commit: `git diff --cached | grep -nE '([0-9]{1,3}\.){3}[0-9]{1,3}|/Users/|/home/'` and a grep for the remote service user's name; both must print nothing.
- **A live run trains on this machine.** Never touch `runs/`, its processes, the remotes' sync or work directories, or the real hosts file. Build with `nice -n 10`, `-j6`, in scratch directories. Use the PC only in a scratch directory, with small runs.
- New files open with a teaching comment and `Mark Castelluccio, with assistance from Claude (Anthropic).`
- Commits: author Mark; message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Default-path identity (Task 2's `playMatches` refactor; Task 9's old-vs-new binaries).
2. Self-play unchanged through the coordinator refactor (Task 4; the existing `test_distributed`, `test_trainer` suites).
3. Verification of arena results (Task 3 refusals).
4. Merge by wave index, failure policy, commit point (Tasks 4–6).

---

## Task 1: the `arena_engines` key

**Files:** `include/azgo/config.hpp`, `src/config.cpp`, `tests/test_config.cpp`.

- [ ] Test: `arena_engines = local/metal:2, pc/cuda:3` parses, prints canonically, round-trips through `configToString`; the default is empty and prints `arena_engines = `; a bad item is refused with the parser's message; `checkConsistency` refuses an `=N` item ("arena waves differ in size; arena_engines takes weights (:W) only").
- [ ] Implement: `Config::arenaEngines` (`std::vector<EngineSpec>`), the table entry after `arena_random_symmetry`, the consistency rule.
- [ ] Run `test_config`, commit.

## Task 2: arena games as ranges

**Files:** `include/azgo/arena.hpp`, `src/arena.cpp`, `include/azgo/selfplay_slice.hpp`, `src/selfplay_slice.cpp`, `tests/test_arena.cpp`, `tests/test_cuda.cpp`.

- [ ] Tests: `waveSlice` for every (first, count) of a 3-match wave (ranges in match order, each range's first/count, totals, nothing for count 0, out-of-range refused); `playArenaGames` over every split of a wave into 1–3 slices equals `playMatches` (per-game scores and moves via `matchResult`, and the `MatchResult`s), naive and blas, workers 1, 3, 0; a cancelled flag stops a slice with `SliceCancelled`; `SliceStats` counts the plies; ranges are checked as `playMatch` checks a match (except an odd count).
- [ ] Implement: `ArenaGameRange`, `ArenaGameResult`, `playArenaGames(ranges, cfg, EngineRun, SliceStats*)` (the pool of `playMatches`, results by range and game), `matchResult(games)`, `waveSlice(wave, first, count)`; `playMatches` built on them; `CancellableEvaluator` moved to `selfplay_slice.hpp` (public) and used for network seats; `cuda` in `RoundEvaluators` and `arenaWorkers` as `metal`; `RoundEvaluators::evaluations()`.
- [ ] `test_cuda`: an arena wave with `cuda` equals itself at other worker counts and batches (skipped without a GPU).
- [ ] Run `test_arena`, `test_arena_round`, `test_trainer`, commit.

## Task 3: arena job and result files

**Files:** new `include/azgo/arena_slice.hpp`, `src/arena_slice.cpp`; `apps/selfplay.cpp`; `src/coordinator.cpp` (`InProcessTransport`); `tests/test_distributed.cpp`.

- [ ] Tests: `ArenaJob` round trip, every field bitwise; `parseArenaJob` refusals (magic, version, checksum, truncation, trailing bytes, bad network index, bad range); result round trip; `readArenaResult` refusals: missing file, magic, version, length, checksum (a flipped byte anywhere), job hash, build, fingerprint, game count, an off-board or illegal move, a game that had ended before a move, one that never ends, a wrong score, positions, trailing bytes; `runJob` plays an arena job in process and its result equals `playArenaGames`; an arena job's network with a wrong hash is refused; the real `selfplay` binary answers an arena job, bitwise as in process.
- [ ] Implement: `ArenaJob` (board size, `ArenaConfig`, networks, ranges with player descriptions, build, inference batch), `serializeArenaJob`/`parseArenaJob` (`AZAJ` v1), `ArenaSliceResult`, `serializeArenaResult`/`readArenaResult` (`AZAR` v1, replay verification), `runArenaJob`, `runJob` (dispatch by magic); the app and `InProcessTransport` call `runJob`; usage text.
- [ ] Run `test_distributed`, commit.

## Task 4: the coordinator plays arena waves

**Files:** `include/azgo/coordinator.hpp`, `src/coordinator.cpp`, `include/azgo/trainer.hpp`, `src/trainer.cpp` (rename only), `tests/test_distributed.cpp`.

- [ ] Rename `SelfPlayCoordinator` to `Coordinator` (it serves both lists); constructor refuses only when both lists are empty.
- [ ] Generalise the remote slice: a `RemoteWork` (file base name, job bytes, networks to have on the machine, a function that reads and keeps the result) and a local fallback function; `attemptRemote`/`playRemote` take them; a machine keeps the set of networks its slices of the current phase need; `runSlices` runs a phase's slices at once (threads, cancel, first error by list order, work-directory cleanup). `play()` builds self-play's works and is otherwise unchanged (file names `g%04d_%d_a%d`, logs, manifest).
- [ ] `playArena(g, wave, cfg)`: assignment, networks saved and hashed into `arena-work/gen_NNNN_wW/` when a remote plays, local slices with `playArenaGames`, remote slices as `arena_g%04d_w%d_%d_a%d`, merge by wave index, `matchResult` per match; `beginArena(g)`, `arenaReports()` (each report with its wave); fallback engine from `arena_engines`.
- [ ] Preflight over both lists (messages name the key), local memory checks for arena items (`arenaWaveNetworks(cfg)` one-slot hubs per local GPU item), fingerprint notes from both manifests.
- [ ] `arenaManifestPath`, `writeArenaManifest`, `pruneArenaManifest`, `enginesLine(reports, label)`.
- [ ] Tests: `playArena` with `local/naive:1`, `local/naive:2, local/naive:1` and `local/naive:1, fake/naive:1` (in-process and subprocess) equals `playMatches` on each wave; reports and manifest rows; a remote's work files are gone after a wave and only the wave's networks were uploaded; existing self-play tests unchanged.
- [ ] Run `test_distributed`, `test_trainer`, commit.

## Task 5: the trainer uses it

**Files:** `include/azgo/trainer.hpp`, `src/trainer.cpp`, `tests/test_trainer.cpp`.

- [ ] Tests: `arena_engines = local/naive:1`, `local/naive:2, local/naive:1` and `fake/naive:1` equal the default run byte for byte with gating, bots, reference networks and promotions (`roundsConfig`); the arena manifest's rows (generation 0's calibrations included); resume is byte-identical with the manifest; switching on and off at resume prints the note once and records `config.cfg`; a resume prunes uncommitted arena rows; the progress output's `arena engines:` line; preflight refuses an unknown arena-only alias and a remote of another build before writing anything; `rerate` ignores `arena_engines`.
- [ ] Implement: `makeCoordinator` for either list; self-play through the coordinator only with `selfplay_engines`; `runMatches(..., Coordinator*, generation)`; `beginArena` before each generation's arena and the start's calibrations; `writeArenaManifest` before `state.bin`; the resume note and pruning; the log lines.
- [ ] Run `test_trainer`, commit.

## Task 6: failures in arena slices

**Files:** `tests/test_trainer.cpp` (and `src/coordinator.cpp` if anything is missing).

- [ ] Tests (fake remote, `arena_engines` only): one failure (exit, timeout, corrupt) is retried with the same bits; two fall back to the first local arena engine with the note and a manifest row naming both engines; `selfplay_fallback = stop` throws, keeps `state.bin` at g − 1, and a resume equals the failure-free run.
- [ ] Run, commit.

## Task 7: Metal, CUDA and the smoke test

**Files:** `tests/test_metal.cpp`, `tests/integration_smoke.sh`.

- [ ] `test_metal`: `arena_engines = local/metal:1` and `local/metal:1, local/metal:2` equal the default metal run (gating and promotions decided the same), with chaos batching.
- [ ] `integration_smoke.sh`: a fake remote playing arena slices (`arena_engines=local/naive:1,fake/naive:1`), resumed, the manifest and summary line, equal to the default run's `state.bin`; on a CUDA build, `fake/cuda` and `local/cuda` arena slices.
- [ ] Run, commit.

## Task 8: tools and documentation

**Files:** `tools/engine_weights.py`, `README.md`, `docs/results/bench.txt` (or a note in `docs/results`).

- [ ] `engine_weights.py --arena`: reads `arena-engines.csv` and prints an `arena_engines` line of weights (GPU and CPU alike).
- [ ] README: "Distributed arena" in the distributed self-play section: the key, weights only, the manifest, the summary line, fallback, rerate, switching on.
- [ ] Commit.

## Task 9: verification

- [ ] Old against new `train` on 5×5 and 9×9 (naive, blas, metal): byte-identical files but `config.cfg`'s new line.
- [ ] A split-engine arena run resumed equals a straight one; all games on one engine vs the default: same gating.
- [ ] Full ctest: Metal and BLAS builds on the Studio; the CUDA build on the PC in a scratch directory.
- [ ] Results noted in `docs/results/bench.txt`.
