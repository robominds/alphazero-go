# Dynamic Self-Play Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** With `selfplay_allocation = dynamic`, a generation's self-play games form one queue: each engine keeps its concurrency of games in flight and takes more as they end, a handout rule gives the last games to the engines that finish them soonest, a failed remote's unfinished games flow to every other engine, and each game's engine is recorded. With `static` (the default) runs are byte-identical to `main`.
**Architecture:** `GameQueue` (pure bookkeeping, time passed in) decides who gets which index. Local engines are worker pools of `playSelfPlayQueue` pulling from it. Each remote engine runs one `selfplay --stream` session per generation through a new `Transport::openStream`: `play <i>` lines in, checksummed `AZSH`/`AZSG`/`AZSE` frames out, each game verified as a slice result's records are. The coordinator merges by index and writes `selfplay-engines.csv` rows per engine and `selfplay-games.csv` rows per game.
**Tech Stack:** C++17, CMake, assert-based CTest executables, bash (`integration_smoke.sh`), Python 3 standard library.
**Spec:** docs/superpowers/specs/2026-10-23-dynamic-selfplay-design.md
**Author:** Mark Castelluccio, with planning assistance from Claude (Anthropic).

## Global Constraints

- **The default is byte-identical to `main`.** With `selfplay_allocation` absent or `static`, every file a run writes (but `config.cfg`'s new line and timings) is `main`'s, with and without `selfplay_engines`, for naive, blas and metal. The static path's code (`assignGames`, `runSlices`, `attemptRemote`, `playRemote`) is not changed in behaviour; `playSelfPlayGames` keeps its output bit for bit.
- **Every index exactly once**, verified per game as today, merged in index order.
- **The algorithm does not change.** Scheduling only.
- **No host names, user names, IP addresses or absolute home paths** in any commit. Before each commit: `git diff --cached | grep -nE '([0-9]{1,3}\.){3}[0-9]{1,3}|/Users/|/home/'` and a grep for the remote service user's name; both must print nothing.
- **A live run trains on this machine.** Never touch `runs/` (read only), its processes, the minis, the PC's cluster/work/dev directories, or the real hosts file. Build with `nice -n 10`, `-j6`, in scratch directories. Use the PC only in `~/dyn-scratch`, CPU or tiny GPU checks.
- New files open with a teaching comment and `Mark Castelluccio, with assistance from Claude (Anthropic).`
- Commits: author Mark; message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Static path identity (Task 3's `playSelfPlayGames` refactor; Task 9's old-vs-new binaries).
2. No lost or doubled index under failures (Tasks 2, 6, 7).
3. Stream verification (Task 4 refusals).
4. Thread safety of the coordinator's dynamic phase (Task 6).

---

## Task 1: the `selfplay_allocation` key

**Files:** `include/azgo/config.hpp`, `src/config.cpp`, `tests/test_config.cpp`.

- [ ] Test: default `static`, prints `selfplay_allocation = static` after `selfplay_fallback`; `dynamic` round-trips; anything else refused ("expected static|dynamic"); tunable on resume (`applyOverride` across a saved config).
- [ ] Implement: `Config::selfplayAllocation` (string), the table entry.
- [ ] Run `test_config`, commit.

## Task 2: `GameQueue` and the simulation

**Files:** new `include/azgo/game_queue.hpp`, `src/game_queue.cpp`, `tests/test_game_queue.cpp`; `CMakeLists.txt` if the source list is explicit.

- [ ] Tests: lowest index first; concurrency respected; `finished` frees a slot; `kNone` only once every index is finished; `requeue` to the front, only an index the engine holds; `retire` (no more games; its held indices must be requeued by the caller); double `finished` or a foreign index is a `logic_error`; the handout rule (unknown latency gives; in time gives; C counts other engines' slots; holds when others can absorb; gives once their predictions are overdue; never stalls when nothing is in flight); forced assignment.
- [ ] The simulation: discrete events over engines with `R(n) = rmax · min(1, n/sat)`, shared equally, random game costs; static (`assignGames`, all at once) against dynamic (plain and with the rule) for the live run's engines under three weightings, 30 seeds each; print the table; assert the rule beats static's mean in every scenario and the plain queue on the live weights.
- [ ] Implement `GameQueue`.
- [ ] Run, commit.

## Task 3: `playSelfPlayQueue`

**Files:** `include/azgo/selfplay_slice.hpp`, `src/selfplay_slice.cpp`, `tests/test_distributed.cpp` (or `test_selfplay.cpp`).

- [ ] Tests: `playSelfPlayGames` output unchanged (compare against a direct `playSelfPlayGame` loop for workers 1, 3); `playSelfPlayQueue` with a `next` over a permuted index list gives the same records per index; `next` returning −1 at once; an exception in `done` or a game stops the workers and is rethrown; cancel.
- [ ] Implement `playSelfPlayQueue(net, sp, seed, generation, run, next, done, stats)` and `playSelfPlayGames` on it.
- [ ] Run `test_distributed`, `test_selfplay`, commit.

## Task 4: the stream protocol and `selfplay --stream`

**Files:** `include/azgo/selfplay_slice.hpp` (or new `include/azgo/selfplay_stream.hpp`), `src/selfplay_slice.cpp` (or new `src/selfplay_stream.cpp`), `apps/selfplay.cpp`, `tests/test_distributed.cpp`.

- [ ] Tests: frames round trip (hello, game, end); `StreamReader` refusals: wrong magic, checksum (each byte flipped), oversized length, truncated at end of file, job hash, build ID, game index not in flight or twice, a record that does not verify, end frame positions; `runStreamJob` on pipes in process (play a permuted set, `end`, all frames back, records equal `playSelfPlayGames`'); end of file without `end` stops with exit 1; a bad command line exits 1; the real binary `selfplay --stream` through a spawned child equals in process.
- [ ] Implement: frame writer/reader with FNV trailer, `runStreamJob(jobPath, backend, workers, inferenceBatch, inFd, outFd, err)`, the app's `--stream` (SIGPIPE ignored), usage text.
- [ ] Run `test_distributed`, commit.

## Task 5: child processes with pipes, `Transport::openStream`

**Files:** `include/azgo/subprocess.hpp`, `src/subprocess.cpp`, `include/azgo/coordinator.hpp`, `src/coordinator.cpp`, `tests/test_distributed.cpp`.

- [ ] Tests: `ChildProcess` (spawn with stdin and stdout pipes, stderr to a log; write, read with timeout, close input, kill the group, wait; registered for signal forwarding); writing to a dead child returns false without SIGPIPE; `sshStreamArgv` vectors; `SubprocessTransport::openStream` and `InProcessTransport::openStream` play a stream end to end.
- [ ] Implement.
- [ ] Run, commit.

## Task 6: the coordinator's dynamic phase

**Files:** `include/azgo/coordinator.hpp`, `src/coordinator.cpp`, `tests/test_distributed.cpp`.

- [ ] Tests: `play()` with `dynamic` and `local/naive:1, local/naive:2` equals the static records; with a fake remote (in-process and subprocess) too; reports (one per engine item) count every game once; `assignment()` gives each game's engine; concurrency rules (hosts, quarter share, CPU rule, fallback); work files removed.
- [ ] Implement: `Coordinator::play` dispatches on `selfplayAllocation`; `playDynamic`: shared `GameQueue` behind a mutex and condition variable, a thread per local engine running `playSelfPlayQueue`, a session per remote (upload network and job, open stream, reader and feeder threads, verification, timeout), failure handling (requeue, retry once, retire or stop, fallback engine when all retired), merge, reports and assignment; `distributed_probe::failSession(engine, times, afterGames)`, `forceAssignment(g, vector)`.
- [ ] Run, commit.

## Task 7: the trainer, manifests and notes

**Files:** `src/coordinator.cpp` (`writeGamesManifest`, `pruneGamesManifest`), `src/trainer.cpp`, `tests/test_trainer.cpp`.

- [ ] Tests: dynamic with `local/naive:1, fake/naive:1` equals the default run byte for byte; manifests (per-engine rows, per-game rows each index once per generation); resume equals uninterrupted; switching static → dynamic → static prints notes and records `config.cfg`; a remote failing mid-generation (exit after k games) requeues, retries, retires, and the run equals the failure-free one (one fingerprint) with notes in the manifest; `stop` throws, `state.bin` stays, resume equals; the per-game manifest pruned on resume; in BLAS builds the forced-assignment replay of a naive+blas dynamic run is byte-identical and a different assignment is not.
- [ ] Implement: write both manifests before `state.bin`, prune on resume, the allocation note, the progress line.
- [ ] Run, commit.

## Task 8: smoke test, tools, documentation

**Files:** `tests/integration_smoke.sh`, `tools/engine_weights.py`, `README.md`, `docs/algorithm-explained.md`, `docs/algorithm-explained.html`.

- [ ] Smoke: a tiny dynamic run with a fake remote, resumed; `selfplay-games.csv` holds each index once per generation; `selfplay --stream` usage.
- [ ] `engine_weights.py`: dynamic manifest rows (empty `first`) read; a note that in dynamic mode weights set concurrency only.
- [ ] README: "Dynamic allocation" in the distributed self-play section; the explainer's "More than one machine" passage gains the queue; citations checked with `tools/check_citations.py`.
- [ ] Commit.

## Task 9: verification

- [ ] Old against new `train` (5×5 and 9×9; naive, blas, metal; default path, engines with a fake remote; straight and resumed): byte-identical but `config.cfg`'s line.
- [ ] A real local balance check: static against dynamic with engines of different speeds; finish-time spread from the manifest.
- [ ] Full ctest: Metal and BLAS builds on the Studio; the CUDA build on the PC in `~/dyn-scratch` (deleted afterwards).
- [ ] Results in `docs/results/bench.txt`; the spec's simulation table filled in.
- [ ] Fresh-context code review; fix what it finds.
