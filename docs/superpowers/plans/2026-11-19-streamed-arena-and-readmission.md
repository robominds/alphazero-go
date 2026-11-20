# Streamed Arena Waves and Readmission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Mark Castelluccio, 2026-11-19. Written with Claude Code (Claude Opus 5.5).

**Goal:** Arena waves can be played from a shared, streamed per-game queue
(`arena_allocation = dynamic`). In both dynamic phases, a remote engine
whose machine went unreachable is readmitted when the machine comes back.

**Architecture:** The arena gets a pull-based game pool and a stream job
type. The streaming protocol and checker learn arena jobs. The
coordinator's `DynamicPhase` gains a phase kind so `playArena` can run a
wave through it. Readmission is a waiting state in `DynamicPhase::runRemote`,
with a non-throwing per-machine check factored out of preflight.

**Tech Stack:** C++17, CMake/ctest, the repo's assert-based test programs
(`tests/test_*.cpp`, one `main` per file).

**Spec:** `docs/superpowers/specs/2026-11-19-streamed-arena-and-readmission-design.md`.
Read it first; this plan argues from it.

## Global Constraints

- **Defaults:** `arena_allocation = static` and `readmit_seconds = 300` (0
  = off). Both keys can change on resume (not architecture keys).
- **Identity:** with defaults, results are byte-identical to `main` on
  every path. Every existing test passes unchanged; don't edit existing
  assertions except where a task says so.
- **No game index lost or doubled.** Merges refuse a missing or doubled
  index.
- **Readmission only after the machine was observed unreachable.** An
  engine retired after two failed sessions while its machine answers stays
  retired for the phase.
- **Same build everywhere:** the hello's build check stays, and readmission
  requires `--version == buildId()`.
- **Repo hygiene:** no host names, user names, IPs, `/Users/` or
  `/home/<name>` paths in the repo; grep the staged diff before each commit.
  Match the surrounding code's comment density and naming.
- **The live run on this Mac Studio is running** (`runs/19x19-agz256`,
  binaries in `runs/bin-r39`):
  - Never touch `runs/` or `~/.config/azgo/hosts`, and never ssh to the
    cluster machines.
  - Build with `nice -n 10` and `-j6`.
  - Work in a git worktree on branch `streamed-arena`.
- **Every commit message** ends with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. The git author
  is Mark Castelluccio, as configured.
- **Build and test commands** (default build, no BLAS):
  `nice -n 10 cmake --build build -j6 && ctest --test-dir build --output-on-failure -R <test>`.
  Configure once with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`.

## Review Focus

1. **A machine that comes back with another build** (someone synced or
   rebuilt it mid-phase) must not be readmitted. It stays waiting, with one
   note, and is not probed into a loop of failed sessions. Task 7's test.
2. **The phase ends while a readmission probe or check is in progress.**
   The phase must end promptly, and no session may start after `done()`.
   Task 7's test, using a slow-probe hook.
3. **Waves that aren't a plain gating match:**
   - a ladder round with several matches;
   - a bot opponent (`network = -1`, no network to upload);
   - calibration waves (generation 0 against references).
   
   A dynamic wave must merge all of them per match. Task 6's test.
4. **A promotion wave right after the first wave:** a second dynamic phase
   in the same generation needs fresh sessions, distinct file names
   (`arena_g%04d_w%d_...`), and `down[kArenaPhase]` reset. Task 6's test.
5. **Pipelining with both phases dynamic on the same fake machine:** a
   readmission in one phase must not clear the other phase's `down` flag or
   start a session for it. Task 8's test.

---

### Task 1: `GameQueue::readmit`

**Files:**
- Modify: `include/azgo/game_queue.hpp`, `src/game_queue.cpp`
- Test: `tests/test_game_queue.cpp`

**Interfaces:**
- Produces: `void GameQueue::readmit(int engine);`
  - The inverse of `retire`: the engine may `take` games again.
  - `std::logic_error` in forced mode, or if the engine is not retired.

- [ ] **Step 1: Write `test_readmit` in `tests/test_game_queue.cpp`** and
  call it from `main`. Assert:
  - after `retire(1)`, `take(1, t) == kNone`;
  - after `readmit(1)`, `retired(1) == false` and `take(1, t)` returns the
    lowest queued index (requeued first);
  - `readmit` on a non-retired engine throws `std::logic_error`;
  - in forced mode, `readmit` throws `std::logic_error`;
  - `done()` and `finishedBy` counts are unaffected.
- [ ] **Step 2: Run it and see it fail to compile.**
  `ctest --test-dir build -R test_game_queue` → build error: no `readmit`.
- [ ] **Step 3: Implement `readmit`.** It clears `Engine::retired`. Check
  that `inTime` and `take` need nothing else.
- [ ] **Step 4: Run it and see it pass.** `test_game_queue` passes.
- [ ] **Step 5: Commit**, "Let a retired engine take games again
  (GameQueue::readmit)".

### Task 2: Config keys `arena_allocation` and `readmit_seconds`

**Files:**
- Modify: `include/azgo/config.hpp`, `src/config.cpp`, `README.md` (the
  config key table, next to `selfplay_allocation`), and the config presets
  under `configs/` only if they list every key
- Test: `tests/test_config.cpp`

**Interfaces:**
- Produces:
  - `std::string Config::arenaAllocation = "static";`, values
    `static|dynamic`, error text like `selfplay_allocation`'s;
  - `double Config::readmitSeconds = 300;`, range [0, 86400], 0 = off.
  
  Neither is an architecture key nor locked on resume.

- [ ] **Step 1: Write the failing tests** (`test_config.cpp`):
  - defaults `"static"` and `300`;
  - `arena_allocation=dynamic` parses and `=foo` is refused naming
    `static|dynamic`;
  - `readmit_seconds=0` and `=60` parse, and `=-1` is refused;
  - printing round-trips;
  - `architectureDiff` and the resume-locked list don't contain either key.
- [ ] **Step 2: Run them and see them fail.** `test_config` fails to
  compile or fails its asserts.
- [ ] **Step 3: Add the keys** to the key table in `src/config.cpp`, next
  to `selfplay_allocation`.
- [ ] **Step 4: Run them and see them pass.** `test_config` passes.
- [ ] **Step 5: Commit**, "Add arena_allocation and readmit_seconds".

### Task 3: A pull-based arena game pool

**Files:**
- Modify: `include/azgo/arena.hpp`, `src/arena.cpp`
- Test: `tests/test_arena.cpp`

**Interfaces:**
- Produces, in `arena.hpp`:
  ```cpp
  // Where a queue-driven arena engine gets its games (wave indices) and leaves
  // its results. Called from the worker threads.
  class ArenaGameSource {
  public:
      virtual ~ArenaGameSource() = default;
      virtual int next() = 0;   // a wave index, blocking, or -1 when done
      virtual void done(int index, ArenaGameResult result, long long evaluations) = 0;
      virtual void stop() = 0;
  };
  // playArenaGames with its games from a source. `wave` is the whole wave as
  // ranges in match-major order, each a full match (first = 0); wave index i
  // is game j of the range it falls in, by cumulative counts. One set of
  // evaluators per distinct network for the whole call, as playArenaGames.
  // A game's result depends only on (range, j) and the backend's arithmetic.
  // Exceptions: as playSelfPlayQueue (first one stops the others, source.stop(),
  // rethrown). stats: wall, plies of the games done, evaluations.
  void playArenaQueue(const std::vector<ArenaGameRange>& wave, const ArenaConfig& cfg,
                      const EngineRun& run, ArenaGameSource& source, SliceStats* stats = nullptr);
  ```
  It uses `run.workers` threads (<= 0: `arenaWorkers(0, run.backend,
  totalGames)`).

- [ ] **Step 1: Write `test_arena_queue` in `tests/test_arena.cpp`.**
  - Build a small 5×5 wave with two matches: one network against
    network, one network against a bot.
  - Use a test `ArenaGameSource` that hands indices out of order, for
    example a reversed list, and stores the results.
  - Assert that each result equals `playArenaGames(waveSlice-style ranges,
    ...)` at the same index, for 1 and 3 workers.
  - Assert that the `stats` plies equal the sum of the moves.
  - Assert that a source whose `done` throws makes the call rethrow, after
    `stop()` was called.
- [ ] **Step 2: Run it and see it fail.** `test_arena` fails to compile.
- [ ] **Step 3: Implement `playArenaQueue`** in `src/arena.cpp`, reusing
  `RoundEvaluators` and the per-game play `playArenaGames` uses. The
  existing fixed-list `playArenaGames` must stay byte-identical; leaving it
  untouched is fine.
- [ ] **Step 4: Run it and see it pass.** `test_arena` passes, and so do
  `test_arena_round` and `test_distributed_arena`.
- [ ] **Step 5: Commit**, "Add playArenaQueue, an arena pool fed by a
  game source".

### Task 4: An arena stream on the remote side

**Files:**
- Modify: `include/azgo/selfplay_stream.hpp`, `src/selfplay_stream.cpp`,
  `include/azgo/arena_slice.hpp`, `src/arena_slice.cpp` (make
  `verifyArenaGame` public), `apps/selfplay.cpp` (only if the dispatch
  needs it)
- Test: `tests/test_selfplay_stream.cpp`

**Interfaces:**
- Consumes: `playArenaQueue`, `ArenaGameSource` (Task 3).
- Produces:
  - `void verifyArenaGame(const ArenaJob& job, int index, const ArenaGameResult& g, const std::string& game);`
    moved to `arena_slice.hpp`, with its arguments exactly as today.
  - A new frame, documented in the header's protocol comment:
    `AZSA arena game: i32 wave index, u32 n, n x i32 moves, f64 scoreA, f64 seconds, u64 evaluations`.
  - `struct StreamArenaGame { int index = 0; ArenaGameResult result; double seconds = 0; long long evaluations = 0; };`
  - `std::string streamArenaGameFrame(int index, const ArenaGameResult& result, double seconds, long long evaluations);`
  - `StreamFrames::next` returns kind `'A'` for `AZSA`.
  - `StreamChecker`:
    - its constructor detects an arena job (`isArenaJob`), and
      `bool StreamChecker::arena() const`;
    - `StreamArenaGame StreamChecker::arenaGame(const std::string& payload)`
      refuses: before the hello, an index outside `[0, job.games())`, an
      index returned twice, a record that fails `verifyArenaGame`, and
      trailing bytes;
    - `game()` on an arena job and `arenaGame()` on a self-play job are
      refused;
    - `end()` checks positions against the arena games' plies;
    - `const ArenaJob& StreamChecker::arenaJob() const`.
  - `runStreamJob` dispatches on the job file's magic. An arena job's
    stream plays `playArenaQueue` over the job's ranges (each range a full
    match, first = 0, as the coordinator writes them in Task 6). The
    command source accepts indices in `[0, job.games())`. Hello, commands
    and the end frame are as for self-play.

- [ ] **Step 1: Write the failing tests:**
  - `test_arena_frames_round_trip`;
  - `test_arena_checker_refusals`: each refusal above, plus a self-play
    checker given an `'A'` frame;
  - `test_run_arena_stream_in_process`: an arena job over the in-process
    pipes, out-of-order `play` lines, then `end`. The games equal
    `playArenaGames`, and the exit is 0;
  - `test_run_arena_stream_child`: the real `selfplay --stream` binary
    with an arena job, as `test_child_process` does for self-play.
- [ ] **Step 2: Run them and see them fail.** `test_selfplay_stream`
  fails to compile.
- [ ] **Step 3: Implement** the frame, the checker kind, the public
  `verifyArenaGame`, and the dispatch. Generalise `CommandSource`'s
  range check to the job's game count.
- [ ] **Step 4: Run them and see them pass.** `test_selfplay_stream` and
  `test_distributed_arena` pass.
- [ ] **Step 5: Commit**, "Stream arena jobs: an arena game frame,
  checker and remote loop".

### Task 5: Give `DynamicPhase` a phase kind (refactor, no behavior change)

**Files:**
- Modify: `src/coordinator.cpp` (`DynamicPhase`, `playDynamic`),
  `include/azgo/coordinator.hpp` if declarations move
- Test: the existing tests are the test

**Interfaces:**
- Produces: `DynamicPhase` parameterised by `PhaseKind`. The builder
  picks the mechanism (spec, "Left to the builder"). It must provide:
  - **per-kind wording:** the note prefix ("note: generation g:" for
    self-play, "note: generation g arena wave w:" for arena), "retired for
    this generation" vs "retired for this wave";
  - **per-kind locations:** the work directory and log path
    (`selfplay-work/gen_NNNN/`, `arena-work/gen_NNNN_wW/`);
  - **session file bases:** `g%04d_s%zu_a%d` for self-play, unchanged, and
    `arena_g%04d_w%d_s%zu_a%d` for arena;
  - **per-kind work:** the job bytes; the local engine's play loop
    (`playSelfPlayQueue` or `playArenaQueue`); the stored record type
    (`GameRecord` or `ArenaGameResult`); the frame-checking call
    (`game` or `arenaGame`);
  - **`down[kind]`** and `startPhaseOn(alias, kind, ...)` per kind.

- [ ] **Step 1: Refactor** so self-play goes through the kind-generic
  code. Add no arena path yet.
- [ ] **Step 2: Run the identity tests.** `test_dynamic_selfplay`,
  `test_pipeline`, `test_distributed`, `test_game_queue` and
  `test_selfplay_stream` all pass unchanged.
- [ ] **Step 3: Commit**, "Generalise DynamicPhase over the phase kind".

### Task 6: Dynamic arena waves in the coordinator, with manifests

**Files:**
- Modify: `src/coordinator.cpp` (`playArena`, `writeArenaManifest`, the
  new games manifest), `include/azgo/coordinator.hpp`, `src/trainer.cpp`
  (`printArenaEngines`, prune on resume), `README.md` (arena sections)
- Test: `tests/test_distributed_arena.cpp`

**Interfaces:**
- Consumes: Tasks 3–5.
- Produces:
  - `playArena` with `cfg.arenaAllocation == "dynamic"`:
    - builds the job (all of the wave's matches as full ranges, all of
      the wave's networks), saves and hashes the networks as today;
    - runs the wave as an arena `DynamicPhase`, with the fallback engine
      and rules as self-play's;
    - merges by wave index, then `matchResult` per match;
    - reports one `SliceReport` per engine that played, with
      `dynamic = true` and `wave` set.
  - `writeArenaManifest`: for `report.dynamic` rows, `first` is empty,
    as `writeManifest` does.
  - `std::string arenaGamesManifestPath(const std::string& runDir);`
    (`runDir/arena-games.csv`), with
    `void writeArenaGamesManifest(const std::string& runDir, int g, int wave, const std::vector<std::string>& engineOfGame);`
    (rows `generation,wave,game,engine`; replaces this generation and
    wave's rows) and `void pruneArenaGamesManifest(const std::string& runDir, int g);`.
    The trainer prunes it where it prunes `arena-engines.csv`.
  - `printArenaEngines` prints a dynamic wave's rows like the self-play
    engines line, with the label "arena engines (gen g, wave w)".

- [ ] **Step 1: Write the failing tests** (`test_distributed_arena.cpp`):
  - `test_dynamic_arena_equals_static`: one fingerprint (`local/naive`
    plus a fake remote `fake/naive`, subprocess and in-process). Assert
    that the match results (`scoreA`, wins, `distinctGames`) and every
    game's moves equal `playMatches`, for a gating wave and for a ladder
    wave with three matches including a bot. Assert that the reports
    count every game once and are dynamic.
  - `test_dynamic_arena_two_waves`: a first wave and then a promotion
    wave in the same generation. Assert distinct `arena_g..._w1_` and
    `_w2_` session files in the work dirs, that both waves complete, and
    the reports' `wave` fields.
  - `test_dynamic_arena_failures`:
    - `failSession("fake/naive", 1, 1)` → "1 failed stream", results
      unchanged;
    - `failSession(..., 2, 1)` with the machine reachable → "retired for
      this wave", and the next wave tries it again;
    - every listed engine failing → the fallback joins with the note
      "every engine retired";
    - `selfplay_fallback = stop` → `SliceError`.
  - `test_dynamic_arena_manifests`: `arena-engines.csv` dynamic rows have
    an empty `first`, and `arena-games.csv` has one row per game. Prune
    on resume drops rows of g and later.
  - `test_calibration_wave_dynamic`: a generation-0 calibration wave
    against a reference network, dynamic, merges correctly.
- [ ] **Step 2: Run them and see them fail.** `test_distributed_arena`
  fails its asserts.
- [ ] **Step 3: Implement** the dynamic path, the manifests and the
  printing. Static waves keep their code path. Confirm that preflight's local hub
  sizing (`hubs(arena_engines) * arenaWaveNetworks`) already covers
  dynamic waves, and leave it unchanged.
- [ ] **Step 4: Run them and see them pass.** `test_distributed_arena`
  passes, and `test_arena_round`, `test_trainer` and `test_pipeline` pass
  unchanged.
- [ ] **Step 5: Commit**, "Play arena waves from a shared streamed queue
  (arena_allocation = dynamic)".

### Task 7: Readmission of returning machines

**Files:**
- Modify: `src/coordinator.cpp` (`DynamicPhase::runRemote` waiting state;
  a factored per-machine check; fingerprint bookkeeping),
  `include/azgo/coordinator.hpp` (new hooks)
- Test: `tests/test_dynamic_selfplay.cpp`, `tests/test_distributed_arena.cpp`

**Interfaces:**
- Consumes: `GameQueue::readmit` (Task 1), `cfg.readmitSeconds` (Task 2),
  the kind-generic `DynamicPhase` (Task 5).
- Produces:
  - `struct Coordinator::MachineCheck { bool ok = false; std::string why, build, fingerprint; };`
  - `MachineCheck Coordinator::checkMachine(const std::string& alias, const EngineSpec& e);`
    - It is the remote part of preflight (`--version`, then
      `--fingerprint` for e's backend), factored out.
    - It never throws `ConfigError`.
    - It writes `remoteBuilds_` and `fingerprints_` under
      `fingerprintsMutex_`.
    - Preflight calls it, and its behavior and messages are unchanged.
  - Waiting rules in `runRemote`:
    - **Enters waiting** only when the machine was observed unreachable:
      "not tried" at the phase's start, or `checkReachable()` false
      after a failed session.
    - **While waiting:** wake every `readmitSeconds`, or at once on
      `wake`, on `queue->done()`, or on cancel. On waking, probe
      `reachable()`, then `checkMachine`.
    - **On ok:** clear `down[kind]` and `suspect` for that machine, call
      `queue->readmit(k)` (if it had been retired), start a session with
      the next attempt number, and give it two attempts as at phase
      start.
    - **On a build mismatch:** one note, and keep waiting.
    - **Off:** with `readmitSeconds == 0`, or in forced mode.
    - **The fallback engine**, once it has joined, stays active after a
      readmission, and competes for games.
  - Notes:
    - "X readmitted (machine answered again); it plays games again";
    - "X not readmitted: its machine runs build B, this run C";
    - "X's fingerprint changed: F1 -> F2 after N games".
    
    The engine's report note gains "readmitted N times".
  - Test hooks in `distributed_probe`:
    - `void readmitSeconds(double seconds);`, which overrides the
      config's spacing; tests use 0.05;
    - `void slowProbe(const std::string& engine, double seconds);`,
      which makes that engine's reachability probe take this long.
    
    `reset()` clears both.

- [ ] **Step 1: Write the failing tests.**
  - **Fixture change:** in both test files, the fake remote's `exe` is a
    symlink in the fixture dir to `AZGO_SELFPLAY_EXE`. Removing the
    symlink makes the machine unreachable (`reachable` is `access(exe)`),
    and recreating it brings the machine back.
  - `test_readmit_selfplay`:
    - `failSession("fake/naive", 1, 1)`;
    - the session's failure is observed with the symlink removed;
    - a thread recreates it after about 0.3 s, with
      `readmitSeconds(0.05)` and a slow local engine (`delaySlice`).
    
    Assert:
    - the records equal `f.whole(1)`;
    - `checkBookkeeping`;
    - the fake engine finished more than 1 game;
    - the log has "readmitted";
    - the report note has "readmitted 1 time";
    - a session file with `_a2` or `_a3` exists.
  - `test_readmit_not_tried`: the symlink is absent at the phase's start
    and appears later. The engine is readmitted and plays.
  - `test_retired_while_reachable_stays_retired`: `failSession(..., 2, 1)`
    with the symlink present. The engine stays "retired", and the log
    has no "readmitted".
  - `test_readmit_wrong_build`: after the drop, set
    `setRemoteBuildId("other")` before the symlink returns. The engine
    is not readmitted, the log has exactly one "not readmitted" note,
    and the phase completes.
  - `test_phase_end_not_delayed`: `slowProbe("fake/naive", 5)` with the
    machine dropped. The phase finishes in well under 5 s after its last
    game, and no session file is created after `done()`.
  - `test_readmit_arena` (in `test_distributed_arena.cpp`): the same drop
    and return during a dynamic arena wave. The match results are
    unchanged, and the note wording uses "wave".
  - `test_readmit_off`: with `readmit_seconds=0`, a dropped engine stays
    retired.
- [ ] **Step 2: Run them and see them fail.** The new asserts fail.
- [ ] **Step 3: Implement** `checkMachine` (refactor preflight onto it
  first and run `test_distributed` to confirm preflight is unchanged), then
  the waiting state, the notes and the hooks.
- [ ] **Step 4: Run them and see them pass.** `test_dynamic_selfplay`,
  `test_distributed_arena`, `test_distributed` and `test_pipeline` pass.
- [ ] **Step 5: Commit**, "Readmit remote engines whose machines come
  back mid-phase".

### Task 8: Pipelined dynamic phases, identity, docs

**Files:**
- Test: `tests/test_pipeline.cpp`
- Modify: `README.md`; `docs/superpowers/specs/2026-10-23-dynamic-selfplay-design.md`
  ("The arena stays static" and the non-goal), `2026-10-20-distributed-arena-design.md`
  (the static goal and non-goal), and `2026-10-18-distributed-selfplay-design.md`
  ("down for the rest of the phase"). Each gets a short dated note
  pointing to the new spec.

- [ ] **Step 1: Write `test_two_dynamic_phases`** (`test_pipeline.cpp`):
  `pipeline = on`, both allocations dynamic, and the same fake machine in
  `selfplay_engines` and `arena_engines`. Drop the machine (symlink)
  during both phases, and bring it back. Assert:
  - both phases complete with results equal to a run without the drop
    (one fingerprint);
  - the arena's readmission does not clear self-play's `down` flag
    before self-play's own probe succeeds (check the order of notes);
  - the session files of the two phases are distinct.
- [ ] **Step 2: Run it, fix anything it finds, and see it pass.**
  `test_pipeline` passes.
- [ ] **Step 3: Run the full suite twice:**
  `ctest --test-dir build --output-on-failure` (all pass), and the same in
  `build-metal` if it is configured (all pass or Skipped as before).
- [ ] **Step 4: Update the docs.** README: the new keys, "Dynamic arena
  waves", "Readmission", the revised determinism note, `arena-games.csv`.
  Spec notes as listed above.
- [ ] **Step 5: Commit**, "Test pipelined dynamic phases; document
  dynamic arena waves and readmission".

## After the plan (not tasks; each needs Mark's go-ahead)

The steps for switching the live run are in the spec's "Switching the
live run":
1. merge;
2. a frozen `runs/bin-r44`;
3. remote builds in new directories;
4. stop at a commit;
5. resume with `--set arena_allocation=dynamic`.
