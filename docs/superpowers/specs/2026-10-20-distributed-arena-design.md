# Distributed Arena — Design

Author: Mark Castelluccio, with design assistance from Claude (Anthropic).

Amends the arena step of the generation loop
(`docs/superpowers/specs/2026-09-29-alphazero-go-design.md`), carries out
"Later: the arena" of `2026-10-18-distributed-selfplay-design.md`, and
builds on the waves of `2026-10-10-arena-gpu-design.md`.

The algorithm does not change. Every arena game is the same game, seeded
the same way, between the same players with the same search; the gating
threshold, the game counts, the simulations, opponent selection and
promotion are untouched. What changes is *where* each game of a wave is
played: on the coordinator's GPU or CPU, or on a remote engine (a mini's
GPU, the PC's CUDA GPU), exactly as distributed self-play places its games.

Decisions the coordinator of this work made without asking Mark (he opted
out of questions) are marked **Call**.

## Problem

The 13×13 6×96 run (`runs/13x13-agz`, gating on, 800 arena simulations,
40 games per opponent) spends its arena on the Studio's GPU alone, while
the minis and the PC, which played 48% of its self-play, sit idle:

| generation | wave | games | arena_s |
|---|---|---|---|
| 20 | g−1 (= best = strongest reference) | 40 | 573 |
| 21 | g−1, best | 80 | 1,075 |
| 22 | g−1, best | 80 | 1,021 |

That is ~10.4k evaluations per second (80 games × ~166 plies × ~800
simulations in 1,020 s): the Studio is throughput-bound in the arena as in
self-play. With self-play at ~49 min across four engines, the arena is a
quarter of a generation. A promotion adds a second wave of 80 more games.

## Goals

- **Arena waves on several engines**, with the self-play machinery: the
  hosts file, the transports, the `selfplay` binary, build-ID checks,
  retries, the fallback, a manifest and a per-engine summary line.
- **Each arena game is what it was.** Game j of a match is seeded by the
  match's seed and j, its colours by j's parity, whoever plays it.
- **Static assignment by game index**, so given the assignment (and each
  engine's fingerprint and build) a run is reproducible and resumes byte
  for byte. Mark accepted that a game's moves may differ in the last bits
  depending on its engine, as self-play's do; the manifest records which
  engine played which games.
- **Default unchanged.** With the new key empty a run plays its arena
  through today's code: byte-identical results, `state.bin`, rating files
  and logs (`config.cfg` gains one line, as with every key before).
- **Switchable at a generation boundary** with `--resume --set
  arena_engines=...`, so the live run can turn it on.
- **Every backend:** `metal`, `cuda`, `blas` and `naive` engines can play
  arena slices, with the arena's GPU batching (one hub per network).
- **The decisions stay on the coordinator.** Gating, opponent selection,
  promotion, Bradley–Terry and the rating files run on the merged results,
  in game-index order, never in completion order.

## Non-Goals

- Changing what the arena plays (games, opponents, thresholds,
  simulations) or adding evaluation rules (Stay faithful to AlphaGo Zero).
- Distributing `train --rerate` or the `arena` app (see "Who uses it").
- Equality of a game's bits across engines with different fingerprints.
- Dynamic load balancing (rejected for self-play; the same reasons hold).

## Terms

As in the self-play spec (machine, engine, slice, coordinator,
fingerprint, build ID), plus:

- **Wave**: the matches `runArenaRound` or `calibrateFixedPlayers` hands
  to a `MatchesFn` at once (calibrations, the opponents, a promotion).
  A generation has one or two waves (three when a resumed old run first
  calibrates its fixed players), numbered 1, 2, ... in the order played.
- **Wave index**: the games of a wave in *match-major* order. A wave of
  matches m₀, m₁, ... with g₀, g₁, ... games numbers game j of match i as
  `g₀ + ... + g_{i−1} + j`. A slice is a range of wave indices.
- **Game range**: games [first, first + count) of one match. A slice is a
  list of game ranges, one per match it touches.

## The `arena_engines` key

A tunable config key, the same syntax as `selfplay_engines`, printed
after `arena_random_symmetry`:

```
arena_engines = local/metal:133, mini1/metal:24, mini2/metal:24, pc/cuda:300
```

- **Empty (the default)**: today's arena, `playMatches` with the run's
  `backend` and `workers`.
- **Weights only.** **Call:** fixed counts (`=N`) are refused by
  `checkConsistency` ("arena waves differ in size"). Self-play's games per
  generation are fixed, but an arena wave holds 40, 80 or 160 games
  (one opponent, two, a promotion's 80 calibration games, the start's
  calibrations), so a count cannot mean the same share in each.
- **No "same as selfplay_engines" value.** **Call:** the live run's
  self-play list is fixed counts summing to 400, which do not carry over
  to a wave of 40 or 80, and arena rates differ from self-play's (fewer
  games in flight). A second explicit list is simpler to read and to
  reason about.
- **Not locked on resume**, like `selfplay_engines`. A change prints
  `note: arena engines changed from local/metal to ...; from here on
  arena results differ in the last bits from a run that kept local/metal`.
- **`selfplay_fallback` governs arena slices too.** **Call:** one failure
  policy for the run; a second key would only allow combinations nobody
  needs ("retry the arena locally but stop on self-play").

`selfplay_engines` and `arena_engines` are independent: either can be set
without the other. The hosts file is read when either names a remote.

## Assignment

For each wave, `assignGames(arena_engines, T)` with T the wave's game
count: the same largest-remainder function self-play uses, integers only,
contiguous slices in list order. Slice [first, first + count) of the wave
is cut into game ranges by `waveSlice(wave, first, count)`.

Why match-major. A contiguous slice then touches as few matches as
possible, so a remote needs as few networks as possible (the new network
plus one or two opponents), and its hubs are as few. Colours stay
balanced, since they alternate with j inside each match. Game index
carries no meaning (every game is its own seed stream), so contiguity
biases nothing.

Waves are assigned independently. A promotion wave, decided by wave 1's
result, is assigned when it is played; that result is itself a function of
the assignment of wave 1, so the whole generation is still determined by
the key, the run state and the engines' fingerprints.

## Playing a slice

### `playArenaGames`

The arena's game pool, lifted out of `playMatches`:

```
playArenaGames(ranges, ArenaConfig, EngineRun, SliceStats*) -> one ArenaGameResult per game
```

plays every game of every range in one pool of workers, one set of
evaluators per distinct network (one hub per network on a GPU, a shared
workspace per worker on the CPU; `RoundEvaluators`, unchanged), round-robin
across ranges so every match is under way at once. Game j of a range is
played with `Rng(deriveSeed(seed, j))`, A as Black when j is even, and its
result `{scoreA, moves}` is stored at its position. The results come back
in range order, game order within each range: the slice's wave order.

`playMatches(matches, ...)` becomes "every match as one whole range,
then `matchResult` per match": the same games, the same evaluators,
the same workers (`arenaWorkers`), so the default path is byte-identical
by construction. `matchResult(games)` sums scores, counts wins, losses,
draws and distinct move sequences exactly as before.

`EngineRun` (from self-play) carries the backend, workers, inference
batch, QoS class and the cancel flag. `workers ≤ 0` keeps the arena rule
(`arenaWorkers`); the coordinator passes `engineWorkers` for its local
engines (every game at once on a GPU, `min(4, cores / 3)` CPU workers
locally), as self-play's engines do. Workers never change a result.

**CUDA in the arena.** Until now `backend = cuda` was a self-play engine
only, and `RoundEvaluators` knew metal and the CPU. It now treats `cuda`
as `metal`: one `WorkerEvaluators` hub per network, one slot each when
several networks share the GPU, and `arenaWorkers` gives every game at
once for both GPU backends. The run's `backend` still cannot be `cuda`;
a CUDA engine plays arena slices only through `arena_engines`.

### The arena job and result files

Remote slices reuse the transport, the `selfplay` binary and its command
line. **Call:** the binary tells the two job kinds apart by their magic
(`AZSJ` self-play, `AZAJ` arena), so `selfplay --job JOB --out OUT
--backend B` runs either, and nothing in the transports, `InProcessTransport`
or `cluster_sync.sh` changes. A new app was rejected: the remote surface
would double for one more dispatch on four bytes.

`AZAJ` version 1, little-endian, exact bits, FNV-1a trailer:

- i32 boardSize;
- the `ArenaConfig`: i32 simulations, i32 mctsBatch, f32 cPuct,
  i32 temperatureMoves, f32 komi, i32 maxMoves (resolved), u8 randomSymmetry;
- u32 networks; per network str file (`net_<hash>.azn` in the job's
  directory), u64 hash;
- u32 ranges; per range two players (i32 network index or −1 for a bot,
  i32 simulations, u8 randomSymmetry, f32 cPuct, i32 mctsBatch), u64
  seed, i32 first, i32 count;
- str buildId, i32 inferenceBatch (a speed hint).

`AZAR` version 1: magic, version, length; u64 job hash, str build ID,
str fingerprint, u32 games; per game u32 n and n × i32 moves, f64 scoreA;
f64 seconds, u64 positions (plies), u64 evaluations; FNV-1a trailer.

A job names only the networks its ranges use. Networks are shipped as
self-play's are: `net_<hash>.azn` in the run's remote directory, uploaded
once per machine and hash. The coordinator writes each network of a wave
from memory (`saveNetwork`) to `arena-work/`, because a reference network
in play is fitted to the run's shape and is not its file on disk;
checkpoints written by `saveNetwork` load back bit for bit.

### Verification

`readArenaResult(path, jobBytes, build)` refuses, naming the defect, the
file defects self-play's reader refuses (magic, version, length,
checksum, truncation, another job's hash, another build, an unprintable
fingerprint, trailing bytes) and, per game, a record that does not
replay. It replays every game on `Game(n, komi, maxMoves)` from the job:

- every move is on the board or a pass, and legal where it is played;
- the game had not ended before any of its moves, and has ended after the
  last (two passes or the move cap);
- the stored score is `matchScore` of the replayed board's Tromp–Taylor
  result for A, with A Black in even games;
- the statistics' positions equal the moves.

Arena games record no visit counts, so unlike self-play the reader cannot
check that each move was searched. It can and does check everything the
coordinator uses: the score, which feeds the ratings and the gate, and the
moves, which give `distinct`.

## The coordinator

`SelfPlayCoordinator` becomes `Coordinator`: one object, one preflight,
one set of machine states, for both lists. Its remote machinery is
generalised from "a self-play slice" to "a slice": a job's bytes, the
networks it needs, and a function that reads and verifies the result. On
top of it:

- `play(g, ...)`, self-play, unchanged in behaviour, file names and
  manifest;
- `playArena(g, wave, cfg)`, one wave: assigns its games, saves and
  hashes its networks (only when a remote engine plays), runs every slice
  at once (local slices with `playArenaGames` on their own threads,
  remote ones through the transport), merges the results by wave index
  (each index filled exactly once, whatever finished first), and returns
  one `MatchResult` per match with `matchResult`.

Remote file names carry the kind: `g0012_64_a1.job` (self-play, as
before) and `arena_g0012_w1_40_a1.job` (generation 12, wave 1, first
wave index 40, attempt 1). Work files go to `runDir/arena-work/gen_NNNN_wW/`
(networks, jobs, results, one log per engine), removed after the merge
but for logs of failed attempts, as `selfplay-work/` is.

A machine now holds the *set* of networks the current phase needs,
computed before the slice threads start (the union over its slices). Each
slice uploads what is missing, under the machine's mutex, and removes what
the phase no longer needs. Self-play needs one network, so it behaves as
before ("the network kept only for the current hash"); an arena wave needs
two to four.

**Retry and fallback** are self-play's, unchanged: two attempts on the
planned engine (with the reachability probe, `pkill` of a stray attempt,
per-attempt file names), then `selfplay_fallback`. With `local`, the
fallback engine is the first `local/...` item of `arena_engines`, else
`local/<backend>`; it plays the slice with `playArenaGames`, and the
manifest and the log say so:

```
note: generation 12: arena wave 1 games 40-79 (pc/cuda) played by local/metal: pc: selfplay exit 255; from here on results differ in the last bits from a run without the failure
```

With `stop` the generation throws before `state.bin`, and a resume plays
it again as planned. A failure cancels the wave's other slices at their
next network evaluation (`CancellableEvaluator`, now shared by both kinds).

**Preflight** covers the union of both lists: every remote alias defined,
its `selfplay --version` equal to this build, `--fingerprint` for each of
its backends, one line per engine. Messages name the key that brought the
engine in. Local GPU memory is checked for what each list can run at
once: self-play's two-slot hub per local metal (cuda) item, and the
arena's one-slot hubs, `arenaWaveNetworks(cfg)` per local metal (cuda)
arena item (the formula `checkBackend` already uses for the run's
backend). A remote's GPU memory is checked as for self-play, by its
runner when each hub is created: a refusal fails the slice, which retries
and falls back. Self-play and the arena never run at once, so neither
check adds the other's hubs.

**Fingerprint notes** compare each engine's fingerprint with the last row
of either manifest that names it.

### The arena manifest: `arena-engines.csv`

One row per slice per wave per generation, in the run directory:

```
generation,wave,first,count,planned,engine,fingerprint,build,seconds,positions,evaluations,note
22,1,0,22,local/metal,local/metal,"metal; Apple M5 Max; ...",1a2b3c4,301.2,3652,2913024,
22,1,30,50,pc/cuda,pc/cuda,"cuda; AMD Ryzen ...; NVIDIA ...",1a2b3c4,288.0,8301,6630112,
```

`first` and `count` are wave indices. With the seed and the run state,
the rows are the exact recipe for every arena game. Like the self-play
manifest it is rewritten atomically before `state.bin`, rows of
generations ≥ g are dropped first, a resume prunes uncommitted rows, the
timing column is not part of byte-identity, and the default path writes
none. A run's generation-0 calibrations (fixed players against generation
0, at the start) have rows of generation 0 when `arena_engines` is set
then.

The progress output gains one line per wave, before the generation line:

```
arena engines: local/metal 22 301s 9.7k ev/s | mini1/metal 4 290s 1.8k | mini2/metal 4 295s 1.8k | pc/cuda 50 288s 23.0k
```

### Who uses it

- Live arena rounds and the start's calibrations: yes.
- `train --rerate`: **Call:** no; it plays every wave locally, as today.
  Re-rating is an offline replay that writes only `rerate/`; adding
  remotes, a manifest and a preflight there buys little. With
  `arena_engines` on, a re-rated run's games can therefore differ in the
  last bits from the live run's (as they already do after a backend
  change); the README says so.
- The `arena` app: unchanged.

## Determinism contract

An arena game is identified by (match seed, game index j). Its moves and
score are a pure function of:

1. `Rng(deriveSeed(match seed, j))` and j's parity (colours);
2. both players: network bytes or bot simulations, and their overrides;
3. the `ArenaConfig`;
4. the **fingerprint and build of the engine that played it**.

Not of workers, `inference_batch`, which games share its slice or pass,
the number of engines, the transport, retries or timing (the existing
per-worker and batch-independence arguments, unchanged inside a slice).
A match's result is `matchResult` of its games in index order. The
matches of a wave, their seeds and the waves themselves come from the
results table, so by induction every generation's arena is determined.

Hence, with `arena_engines` set:

- **Reproducible**: two runs with the same config whose arena slices were
  played by their planned engines, with the same fingerprints and builds,
  write byte-identical checkpoints, `state.bin`, rating files, SGFs,
  `log.csv` (timings aside) and manifests (timings aside).
- **Resume is byte-identical** under the same conditions.
- **Splitting is free within a fingerprint**: `local/naive:1` equals
  `local/naive:2, local/naive:1` equals `fake/naive:1` (a fake remote of
  the same build on the same machine) equals the default path with
  `backend = naive`. Likewise `local/metal:1` equals the default with
  `backend = metal`.
- **After a fallback**, the run is reproducible from its config and its
  manifests.

## Switching the live run on

At a generation boundary, after this branch is merged and built:

1. `tools/stop_after_generation.sh runs/13x13-agz PID` (stop the wrapper
   first, as the README says);
2. `tools/cluster_sync.sh` (the remotes must run the new build: the
   coordinator refuses another build ID);
3. `train --run runs/13x13-agz --resume --set "arena_engines=..."`.

Nothing in `state.bin`, `selfplay-engines.csv` or the checkpoints changes
format. `config.cfg` gains `arena_engines = ...`. Self-play is the same
code, so its games stay as they would have been with the old binary;
from the switch on, arena games are the new assignment's. An older
binary refuses the rewritten `config.cfg` (`arena_engines` is an unknown
key to it), so going back to one means deleting that line by hand;
switching the arena back to the Studio alone is `--set arena_engines=`
with the new binary.

## Testing

- `test_config`: the key's parse, print, round trip; `=N` refused.
- `test_arena`: `waveSlice` for every cut of a wave (ranges in order,
  counts, empty slices); `playArenaGames` over any slices of a wave equals
  `playMatches` game for game (scores, moves, `matchResult`), naive and
  blas, workers 1, 3 and 0; refusals; cancellation; CUDA arena evaluators
  in `test_cuda` (skipped without a GPU).
- `test_distributed`: arena job and result round trip; every refusal
  (magic, version, checksum, truncation, job hash, build, game count, an
  illegal move, a game that ends early or never ends, a wrong score,
  positions); `runJob` dispatch; the real `selfplay` binary answers an
  arena job end to end, bitwise as in process; `Coordinator::playArena`
  with split local engines and a fake remote equals `playMatches`.
- `test_trainer`: `arena_engines = local/naive:1` and a split and a fake
  remote equal the default run byte for byte (gating, bots, reference
  networks, promotions); resume; switching on and off at resume; the
  manifest's rows and pruning; failure injection (retry, fallback with the
  note, `stop` keeps the last state and a resume equals the failure-free
  run); preflight refusals for an arena-only remote.
- `test_metal`: `arena_engines = local/metal:1` and `local/metal:1,
  local/metal:2` equal the default metal run (gating decided the same).
- `integration_smoke.sh`: a fake remote playing arena slices, resumed;
  `arena-engines.csv`; the summary line; on a CUDA build `fake/cuda`
  arena slices.
- Old against new `train` (default path): `main`'s binary and this
  branch's write byte-identical files on 5×5 and 9×9 runs, naive, blas and
  metal; `config.cfg` differs only by the new line.

## Sizing (recommendation for the live run)

Arena rates are lower than self-play's: a wave has 40–80 games, so each
engine has fewer searches in flight than in self-play, split across two to
four hubs. Measured on the Studio: 40 games ran at ~9.2k evaluations per
second, 80 at ~10.4k. CUDA passes cost ~2.3 ms at batch 8 and ~8.3 ms at
256 (`docs/results/cuda-bench.txt`), so a CUDA slice of 25–50 games
(passes of ~100–200 positions) should run at roughly 15–25k. Weights
proportional to the self-play rates are a sound start:

```
arena_engines = local/metal:133, mini1/metal:24, mini2/metal:24, pc/cuda:300
```

A wave of 80 games becomes 22 / 4 / 4 / 50, one of 40 games 11 / 2 / 2 /
25. Expected: ~4–6 minutes per 80-game wave instead of 17. After a few
generations `tools/engine_weights.py --arena` suggests weights from the
measured arena rates.

## Risks

| risk | consequence | mitigation |
|---|---|---|
| small slices fill a GPU poorly | less than the summed self-play rates | weights from the arena manifest (`engine_weights.py --arena`); the Studio already keeps ~9k with 40 games |
| a slow engine is the wave's tail | a smaller gain | static weights, as for self-play; arena waves are short, so a tail costs minutes |
| a mini's or the PC's GPU fails mid-wave | the wave waits for two attempts | retry, then the local fallback, as for self-play |
| remotes on another build | refused at resume | `cluster_sync.sh` before resuming |
| `--rerate` of a run with `arena_engines` | re-rated games played locally can differ in the last bits | documented; the live `results.csv` stays the record |
