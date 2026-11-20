# Streamed arena waves and readmitted engines

Mark Castelluccio, 2026-11-19. Written with Claude Code (Claude Opus 5.5).

## Why

The 19×19 20×256 run (1,600 simulations) changed the arithmetic behind two
earlier decisions:

- **A gating wave now takes about 28 h**, about as long as a generation's
  self-play, not "minutes, about 10% of a generation"
  (`2026-10-23-dynamic-selfplay-design.md`, "The arena stays static"). Its
  static slices, weighted from 10×128 rates, came out badly unbalanced: in
  generation 1 the Studio needed 27.7 h for its 46 games, mini1 14.4 h and
  mini2 19.5 h for 27 each, and the wave waited on the Studio.
- **Machines drop out mid-phase**: mini2's Thunderbolt link (05:41,
  2026-11-17), and the PC's WSL twice (10:09 2026-11-17, 14:30
  2026-11-19). Today a remote whose session fails and whose machine then
  does not answer is retired until the phase ends, so a machine that comes back an hour later
  sits idle for the rest of a ~30 h phase. In an arena wave the whole
  remaining static slice falls back to the Studio, which could add a day.

Mark chose (2026-11-19): readmit returning machines in **both** self-play
and the arena, and move arena waves to a **shared, streamed queue** like
dynamic self-play's. Exact reproducibility of the gating match does
**not** matter to him.

## Goals

1. **Dynamic arena waves**: each wave's games are handed out one at a time
   from a shared queue to every engine of `arena_engines`, local and
   remote, through long-lived stream sessions, with the same retry,
   requeue, retire and fallback rules as dynamic self-play.
2. **Readmission**: a remote engine that was retired, or "not tried"
   because its machine was down at the phase's start, is probed
   periodically for the rest of the phase. When it answers with this
   build, it gets a new session and pulls games again. This applies to
   dynamic self-play and to dynamic arena waves.
3. The live run (`runs/19x19-agz256`) can switch to both at a generation
   boundary, with a stop and a resume.

## Non-goals

- Saving games in flight. A game whose session breaks still restarts from
  scratch when an engine takes it again.
- Readmission for static allocation (`selfplay_allocation = static` or
  `arena_allocation = static`). Static phases keep today's behavior.
- Any change to what the arena computes: the matches, gating, promotion,
  ladder and Elo, and the per-game rules (colors by parity, temperature,
  symmetry). AlphaGo Zero's evaluator is unchanged; only where games run
  changes.
- Adding the PC to `arena_engines`. After this change that's a config
  choice, made separately.
- A real scheduler or cross-project coordination (the `cluster` leases
  stay bookkeeping).

## Anti-goals (a failure even if it works)

- A game index lost or played twice into the results. The merge must still
  refuse a missing or doubled index.
- Changing results on the default paths. A run with static allocation
  (the default) must stay byte-identical to main, and existing static tests
  must pass unchanged.
- A readmission loop that hammers a dead machine or fills the logs. Probes
  are spaced out, and a note is written once per state change, not per
  probe.
- Touching the live run's binaries or the scripts it runs.

## Constraints

- C++17 and the repo's style (see the surrounding code). bash 3.2-safe
  tools.
- Every machine must run the same build: the stream hello's build check
  stays.
- The per-machine 16-bit reproducibility contract stays
  (`2026-10-24-coreml-engines-design.md`): an engine's fingerprint is
  recorded per phase.
- No host names, user names, IPs or home paths in the repo.
- Stay faithful to AlphaGo Zero: nothing here changes the algorithm.

## Approaches considered

- **A. Stream arena games like self-play, and readmit in the shared
  session code (chosen).** One session per engine and wave holds all of the
  wave's networks and pulls games one at a time. The merge by wave index
  already ignores finish order. Readmission is written once, in
  `DynamicPhase`, for both phases. It's the most work: an arena stream job,
  frame and checker, and a pull-based arena pool. But it fixes drops and
  imbalance together and keeps the Neural Engine's cross-game batching.
- **B. Many small static slices pulled from a queue.** This keeps today's
  slice jobs, but each slice is its own process with its own model and
  batching. Small slices waste the Neural Engine's batching, and
  worker-sized slices idle until their slowest game (hours at this size).
  Readmission would still need its own code. Rejected.
- **C. Keep static slices and re-split on failure.** This fixes drops but
  not the imbalance seen in generation 1, and the bookkeeping is fiddly.
  Rejected.

## Design

### Config

- **New key `arena_allocation = static | dynamic`, default `static`.**
  This is our call: the default paths and every existing static test stay
  byte-identical, and the live run sets `dynamic` on resume. Like
  `selfplay_allocation`, it is not an architecture key, so it can change on
  resume.
- **New key `readmit_seconds`, default 300; 0 turns readmission off.** It
  is the spacing of reachability probes for a retired or not-tried remote
  engine in a dynamic phase. It is a run key, not a hosts key, because it
  is a run policy, and it can change on resume.
- With `arena_allocation = dynamic`, `arena_engines` weights keep their
  dynamic-self-play meaning: they set each engine's default concurrency
  through `dynamicWorkers` when the hosts file gives no `*_workers`.
  `checkConsistency` still refuses fixed counts in `arena_engines`.

### The arena stream (remote side)

- **Job.** An arena stream job is an `ArenaJob` covering the whole wave
  (every match's full game range, match-major as today) plus every network
  of the wave. `runStreamJob` dispatches on the job's magic: self-play jobs
  go to today's path, arena jobs to a new arena stream loop. `selfplay
  --stream --job` takes either.
- **Commands** are unchanged: `play <wave index>` and `end`. `CommandSource`
  refuses an index outside the wave or asked twice, as now.
- **A pull-based arena pool.** This is a new variant of `playArenaGames`
  that takes games from a source (like `playSelfPlayQueue` and
  `GameSource`) instead of a fixed task list. It keeps one evaluator hub per
  network for the whole session, so games against the same network batch
  together on the device.
- **Frames.** The hello (AZSH) is unchanged. A new arena game frame carries
  the wave index and the game's result record: whatever `ArenaGameResult`
  holds today (moves, score from A's side, positions, evaluations). The end
  frame (AZSE) is unchanged in meaning: positions equal the sum of the
  games.
- **Checking.** `StreamChecker` learns the job kind. For an arena job it
  checks each game frame with `verifyArenaGame`, moved out of the anonymous
  namespace in `arena_slice.cpp` so the checker can use it. It refuses a
  game outside the wave, a game returned twice, or one not handed to that
  session (`queue->held(k)`), as for self-play.

### The arena stream (coordinator side)

- **One dynamic phase per wave.** `playArena` with `arena_allocation =
  dynamic` builds the wave as today (wave index, networks, matches), then
  runs it through the same `DynamicPhase` machinery self-play uses:
  - the queue holds the wave's game indices;
  - local engines run the pull-based arena pool on their own threads;
  - remote engines run sessions;
  - the fallback engine joins when every listed engine has retired.
- **Generalising `DynamicPhase`.** It gains a phase kind (self-play or arena
  wave). Its kind-specific parts become per-kind:
  - `note()` wording and prefix, such as "note: generation g arena wave w:";
  - `logPath` and `workDir` (`arena-work/gen_NNNN_wW/`);
  - session file names (`arena_g%04d_w%d_s%zu_a%d`), so kill patterns never
    match a self-play session's files;
  - the job built;
  - the record type stored by `gameBack`.
- **Merging** is today's: results by wave index, each filled exactly once,
  then `matchResult` per match.
- **Networks.** Every engine of a dynamic wave needs every network of the
  wave. Uploads go through `prepareMachine` as now: once per machine and
  hash, two networks for a gating wave. Local hub sizing in preflight
  already assumes all of a wave's networks per arena engine
  (`arenaWaveNetworks`), so the memory checks need no change. The builder
  should confirm that.
- **Waves stay sequential.** The promotion wave depends on the first
  wave's result, so a session lasts one wave, and `down[kArenaPhase]` resets
  per wave, as today.

### Reports and manifests

- **`arena-engines.csv`** gets one row per engine that played in a dynamic
  wave, as `writeManifest` does for dynamic self-play:
  - `first` is empty;
  - `count` is the number of games that engine finished;
  - `seconds` runs from first handout to last game back;
  - the note carries failed sessions, requeued games and retirement, as
    for self-play.
  
  `writeArenaManifest` honours `report.dynamic` the same way `writeManifest`
  does.
- **New `arena-games.csv`** records `generation, wave, index, engine` for
  each game of a dynamic wave, like `selfplay-games.csv`, for diagnostics.
  A resume prunes it like the other manifests.
- **`printArenaEngines`** prints dynamic rows like self-play's engine line,
  for example "arena engines (gen g, wave w): local/ane 40 ... | mini1/ane
  30 ...".

### Readmission (both dynamic phases)

- **When it applies.** Readmission is for machines that went away and came
  back, not for engines that keep failing on a machine that answers. A
  remote engine in a dynamic phase enters a **waiting** state, instead of
  ending its thread, when its machine was **observed unreachable**: down or
  unreachable at the phase's start ("not tried"), or found unreachable by
  the probe after a failed session. **(Our call, from Mark's "if it is
  restarted".)**
- An engine retired after two failed sessions **while its machine still
  answered** (a crash loop, silence timeouts, a refused network) stays
  retired for the rest of the phase, as today. Probing such an engine would
  readmit it at once and repeat the failure every `readmit_seconds`.
- **Probing.** While waiting, and while the phase's queue is not done, it
  probes the machine's reachability (`Transport::reachable`) every
  `readmit_seconds`. Probes stop as soon as the queue is done or the run is
  cancelled. The wait wakes on the phase's condition variable, so ending the
  phase is never delayed by a sleeping probe.
- **Checking a machine that answers.** Before readmitting, run a non-throwing
  per-machine check, factored out of the remote part of preflight (no GPU
  memory checks, no `ConfigError`, no unlocked writes to shared maps):
  - `--version` must equal `buildId()`; if not, stay waiting, with one note;
  - `--fingerprint` for the engine's backend must succeed.
  
  A fingerprint that differs from the one recorded at preflight (for
  example, an OS update after a reboot) is accepted, with the same note
  wording preflight uses for a fingerprint change. A machine that was
  unreachable at preflight is checked the same way, and its build and
  fingerprint are recorded under the coordinator's existing locks.
- **Readmitting.**
  - `down[kind]` and `suspect` are cleared for that machine.
  - `GameQueue` gets a new operation, `readmit(k)`, the inverse of
    `retire`: engine k may take games again. It is refused in forced mode,
    as `retire` is.
  - Any process or job file of the engine's earlier sessions in the phase
    is removed from the machine first, best effort: a session that failed
    with the machine unreachable could not remove them itself.
  - A new session starts with the next attempt number (for example a2
    after one failed session; file names carry it).
  - A note is written: "X readmitted (machine answered again); it plays
    games again".
  - Each readmission gets the same two attempts as a phase start, then goes
    back to waiting. There is no cap on readmissions within a phase; the
    probe spacing bounds the cost.
- **The fallback engine**, once it has joined because every listed engine
  retired, stays active after a readmission. It just competes for games,
  which is harmless. **(Our call.)**
- **Reports.** The engine's report counts every session's games and
  failures. The note adds "readmitted N times".
- **A fingerprint that changes mid-phase.** The engine's row keeps one
  fingerprint, the last session's, so the manifest formats (which
  `lastFingerprints` and `tools/engine_weights.py` parse) don't change. The
  change is recorded in a note that gives the old and new fingerprints and
  how many games the engine had finished when it changed. The shared
  `fingerprints_` entry takes the new value, so later phases and later
  generations report it. This leaves a gap in the per-game record, which
  Mark accepted along with non-reproducible gating. Such a change needs an
  OS update across a reboot, so it is rare. **(Our call.)**
- **Static phases** are untouched: no waiting, no probes.

### Determinism contract (revised)

- **Dynamic arena waves**, like dynamic self-play: given the per-game
  assignment (`arena-games.csv`), engines with the same fingerprint give the
  same results. A rerun, a resume, or a readmission may assign games
  differently, so with mixed arithmetic the last bits of a wave, and rarely
  a gating decision near the threshold, may differ from an uninterrupted
  run. Mark accepted this (2026-11-19).
- **Static arena waves** keep their current contract unchanged.
- The specs and README text that say "the arena stays static" or "dynamic
  load balancing rejected" are revised to point here. That covers
  `2026-10-23-dynamic-selfplay-design.md` ("The arena stays static",
  non-goal list), `2026-10-20-distributed-arena-design.md` (goal and
  non-goal), and README "dynamic" sections. "A down machine falls back for
  the rest of the phase" (`2026-10-18-distributed-selfplay-design.md`) is
  amended for dynamic phases.

### Testing

- **Existing tests stay as they are** on static allocation: the
  static-shape assertions in `test_distributed_arena`,
  `test_metal`'s mixed-arithmetic arena equality, and `test_pipeline`'s
  arena-row fingerprint.
- **New tests, using the in-process and subprocess transports and the
  existing hooks:**
  - arena stream frames and checker refusals: out-of-wave index, a game
    returned twice, a game not held by the session, a bad record (in
    `test_selfplay_stream`);
  - a dynamic arena wave whose merged match results equal a static run's
    when all engines share a fingerprint, local and fake-remote;
  - session failures in a dynamic wave (`failSession`): retry, requeue,
    retire, fallback join, and `stop`;
  - readmission in both phases. Make the fake remote's exe disappear and
    reappear (`reachable` is `access(exe)`). A new test hook sets the probe
    spacing to milliseconds (like `endTimeout`). Check that games are played
    after readmission, that no index is lost or doubled, and that notes and
    reports are right;
  - an engine retired after two failed sessions while its machine still
    answers stays retired (no probing, no readmission);
  - readmission refused when the build differs (`setRemoteBuildId` is read at each
    call, so it works mid-run);
  - pipeline with both phases dynamic, sessions on the same fake engine in
    both phases at once, with distinct file names;
  - `GameQueue::readmit`, including refusal in forced mode;
  - config: the new keys, their defaults, and that they can change on
    resume.
- **Identity:** a build with this change, run with defaults, matches main
  byte for byte on the default, distributed, dynamic, pipelined and
  arena-engine paths, as earlier specs checked.

### Switching the live run

The steps below are for after the code is merged and Mark approves.

1. Build the new commit into a frozen local copy (`runs/bin-r44`).
2. Sync and build it on the remotes in **new** build directories, so the
   running engines' binaries are never overwritten. Add a second hosts
   file, or edit the `exe=` lines only at the switch.
3. Wait for a generation commit with `tools/stop_after_generation.sh`,
   which loses minutes.
4. Resume with the new binary and `--set arena_allocation=dynamic`. Keep
   `readmit_seconds` at its default.
5. Check the first dynamic wave's `arena-engines.csv` and notes.

Rolling back to an older binary means deleting the `arena_allocation` and
`readmit_seconds` lines from the run's `config.cfg` first: once the run
has resumed with the new binary, `config.cfg` carries both, and an older
binary refuses them as unknown keys.

## Left to the builder

- How to factor `DynamicPhase` into per-kind parts: templates, a small
  strategy interface, or tagged branches.
- The arena game frame's exact byte layout. It should follow the existing
  frames' conventions: magic, length, little-endian.
- Whether the pull-based arena pool replaces the fixed-list
  `playArenaGames` internally, with the fixed list as a source. That's
  acceptable if static results stay byte-identical.
- The exact names of the test hooks and of new notes, beyond the wording
  given above.
