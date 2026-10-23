# Dynamic Self-Play Allocation — Design

Author: Mark Castelluccio, with design assistance from Claude (Anthropic).

Amends "Engines and the assignment", "Failure handling" and "Sizing the
engines" of `docs/superpowers/specs/2026-10-18-distributed-selfplay-design.md`,
which rejected dynamic load balancing as a default. This design keeps that
default and adds dynamic allocation as an opt-in mode, with a determinism
contract that says exactly what it gives up. The arena
(`2026-10-20-distributed-arena-design.md`) is unchanged.

The algorithm does not change. Every game is the same AlphaGo Zero
self-play game: seeded by its index, from the same network, with the same
v_resign, merged in index order. Only *which engine* plays game i is
decided while the generation runs instead of before it.

## Problem

With `selfplay_engines`, `assignGames` cuts a generation into one
contiguous slice per engine, and each GPU engine plays every game of its
slice at once. The 19×19 10×128 run (400 games, `local/metal:26,
mini1/metal:5, mini2/metal:5, pc/cuda:64`, so slices of 104, 20, 20 and
256 games) records each slice's wall time in `selfplay-engines.csv`:

| generation | local/metal (104) | mini1/metal (20) | mini2/metal (20) | pc/cuda (256) | self-play phase |
|---|---|---|---|---|---|
| 1 | 16,598 s | 16,949 s | 17,585 s | 16,275 s | 17,587 s |
| 2 | 14,972 s | 18,063 s | 16,687 s | 16,009 s | 18,064 s |
| 3 | 14,119 s | 15,149 s | 15,360 s | 13,721 s | 15,361 s |

The phase lasts as long as its slowest slice. In generation 3 the engines
evaluated 144.6 M positions at a combined 10,350 evaluations/s, which is
13,970 s of work if every engine had finished together: 9% of the phase
was spent waiting (12% in generation 2). Three causes:

1. **Small slices are dominated by a single long game.** A mini's 20
   games (19×19 with resignation off: 450–700 plies) vary in total length
   by several percent from generation to generation (mini1 and mini2,
   identical machines, differ by 8% in generation 2). Weights can only
   fix the average.
2. **The weights are a person's estimate.** They are set from earlier
   generations' rates (`tools/engine_weights.py`); when the estimate is
   off, one engine waits for another.
3. **A failure moves a whole slice.** When the PC dropped out in the
   middle of a generation, its whole 256-game slice was replayed on the
   Studio, about 2.5× slower, while the minis sat idle.

## Goals

- **Opt-in, switchable at a generation boundary.** A new tunable key,
  `selfplay_allocation = static | dynamic`, default `static`. With
  `static`, a run is byte-identical to `main` (its `config.cfg` gains one
  line). The live run switches with `--resume --set
  selfplay_allocation=dynamic` and can switch back the same way.
- **Engines finish together.** A generation's games form one queue; each
  engine keeps a number of games in flight and takes another when one
  ends, so fast engines take more, and a generation's last games go to the
  engines that will finish them soonest.
- **Failures flow.** A failed remote's unfinished games go back to the
  queue and are played by every engine still running; its finished games
  are kept.
- **Exact bookkeeping.** Every game index is played exactly once, verified
  as today (replay, legality, end, score, resignation), and merged in
  index order. The engine that played each game is recorded.
- **A precise determinism contract**: given the per-game assignment (the
  new `selfplay-games.csv`), a run is byte-identical; a test replays a
  recorded assignment to prove it.
- **No algorithmic change.** Scheduling only.
- **Keep the coordinator ready for the next feature** (gating and training
  running beside the next generation's self-play, AlphaGo Zero's
  asynchronous evaluator): engines must be able to run more than one kind
  of job.

## Non-Goals

- **Bit-identity of dynamic runs from the config alone.** The assignment
  depends on timing, so a rerun or a resumed generation may assign games
  differently and, where engines' arithmetic differs, produce different
  last bits. The manifest makes the run reproducible; the config alone
  does not (see "Determinism contract").
- **Dynamic arena waves** (see "The arena stays static").
- **Moving a game that has started.** A game runs to its end on the
  engine that started it, unless that engine fails.
- **Automatic tuning of concurrency.** How many games an engine keeps in
  flight is a machine setting (hosts file) with a default.

---

## Terms

- **Queue**: the generation's game indices not yet given to an engine.
- **In flight**: a game an engine has taken and not yet returned.
- **Concurrency** of an engine, W_e: the most games it has in flight. For
  a GPU engine it is also its number of worker threads, so it sets how
  full its batches are.
- **Session**: one `selfplay --stream` process on a remote machine,
  playing the games the coordinator sends it for one generation.
- **Latency** of an engine, λ_e: the mean wall time of the games it
  finished this generation, from handout to verified record.

## The `selfplay_allocation` key

| value | meaning |
|---|---|
| `static` (default) | today: `assignGames` slices, each engine plays its slice; nothing below applies |
| `dynamic` | one queue per generation; the rules below |

Tunable on resume (not an architecture key), printed by `configToString`
after `selfplay_fallback`. It applies only when `selfplay_engines` is set;
with the default single engine there is nothing to allocate.
`checkConsistency` keeps the `selfplay_engines` rules unchanged (fixed
counts may not exceed G, and must equal G without a weight), since the
weights and counts still mean something (below).

A resume that changes the value prints a note, as an engine change does:

```
note: self-play allocation changed from static to dynamic; from here on which engine plays
which game depends on timing (recorded in selfplay-games.csv)
```

## What the weights mean in dynamic mode

Weights and fixed counts still go through `assignGames`, giving each
engine item its **share** s_e. In dynamic mode the share no longer fixes
which games the engine plays; it sets only the engine's **default
concurrency**. An engine's concurrency is, in this order:

1. the hosts file's `metal_workers` / `cuda_workers` / `cpu_workers` for
   its machine and backend, when positive (the same keys and meaning as
   today: worker threads, one game each);
2. for a GPU engine (metal, cuda): ⌈s_e / 4⌉, at most 256, at least 1;
3. for a CPU engine (naive, blas): today's worker rule (`max(1, min(4,
   cores / 3))` on `local`, `max(1, min(2, cores − 2))` on a remote),
   capped at G rather than at its share.

Why a quarter of the share: a GPU engine needs enough games in flight to
keep its batches full, and few enough that it plays several "rolling
waves", so that the queue can still move games between engines. The
evidence on both sides:

- **Apple GPUs fill with few games.** In `runs/19x19-agz-failed3`
  (1600 simulations, slices of 19, 3 and 3 games), each mini evaluated
  477–492 positions/s with 3 games in flight, the same as with 20
  (479–482 in the live run); the Studio evaluated 2,551–2,630/s with 19,
  93–96% of its 2,752/s with 104. One worker alone gets 1,010/s on the
  Studio (`bench`, batch 8 synchronous).
- **The CUDA hub needs more.** `docs/results/cuda-bench.txt`, 19×19
  10×128: 5,448 evaluations/s with 32 threads, 6,630 with 64, 6,678 with
  128, 6,942 with 256. The PC's real self-play at 256 games ran at 6,636/s,
  the rate of 64–128 threads (its 8-core CPU feeds the searches).

A quarter gives the live run 26, 5, 5 and 64: every Apple engine stays
full, the PC within ~5% of full, and every engine plays about four waves.
The hosts file overrides it per machine; the recommendation for the live
run is in "Switching the live run".

A **fallback engine** that is not listed (below) gets the concurrency of a
GPU engine with the whole generation as its share.

## The queue and the handout rule

`GameQueue` (new, `include/azgo/game_queue.hpp`) is pure bookkeeping
with no threads and no clock: every call takes `now`, the seconds since
the phase began. That makes it testable and lets a simulation drive the
very code the coordinator runs.

```
GameQueue(int games, std::vector<int> concurrency);     // one queue per generation
int  take(int engine, double now);       // a game index, kWait or kNone
void finished(int engine, int index, double now);
void requeue(int engine, int index);     // its engine failed: back to the queue
void retire(int engine);                 // takes no more games
```

The queue hands out the lowest index first; requeued games go to the
front. `take` returns kNone once every game is finished (the engine's
work is over) and kWait when the engine should ask again later (full,
nothing queued now, or the rule below says so).

### The end of a generation

A plain "whoever asks first" queue balances the middle of a generation
but not its end: a game handed out late to an engine whose games take
long finishes long after everything else. The PC with 64 games in flight
takes ~3,500 s per 19×19 game; a mini with 5, ~3,800 s; the Studio with
26, ~3,500 s. A simulation of the live run (below) shows that the plain
queue, with these concurrencies, is barely better than well-tuned static
weights: what it gains in the middle it loses at the end.

So `take(e, now)` gives engine e a game only if it will finish it in
time, or if no other engine can. With λ_f each engine's mean latency so
far this generation (unknown until it finishes a game):

1. If λ_e is unknown, give (nothing is known yet; this is the start).
2. **bound** = the latest predicted end of a game in flight anywhere:
   max over engines f with known λ_f and their in-flight games g of
   (start_g + λ_f), and at least `now`. The generation will not end
   before it, whatever e does.
3. If now + λ_e ≤ bound, give: e finishes the game before the others
   finish theirs.
4. Otherwise count **C**, the games other engines can still start and
   finish by `bound`: for each other engine f with known λ_f that is not
   retired, each in-flight game g with max(now, start_g + λ_f) + λ_f ≤
   bound (its slot frees in time to play one more), plus its free slots
   when now + λ_f ≤ bound.
5. If the queue holds more than C games, give (the others cannot take
   them all in time); otherwise wait.

The rule never stalls: as time passes, a slot whose game runs late stops
counting in C (its predicted end moves with `now`), and once nothing is
in flight, bound = now and C = 0, so the next engine to ask is given the
game. Waiting engines ask again whenever a game finishes, is requeued or
an engine retires, and at least every 0.5 s.

The rule decides only *who* plays a game, never what the game is.

### Forced assignment (tests)

`GameQueue(std::vector<int> engineOfGame, concurrency)` hands engine e
only the games the vector gives it, in index order, and ignores the rule.
It exists to replay a recorded assignment (`distributed_probe::
forceAssignment`), which is how the determinism contract is tested.

## Where the games are played

### Local engines

`playSelfPlayQueue(net, sp, seed, generation, run, next, done, stats)`
generalises `playSelfPlayGames`: W worker threads, each with its evaluator
from one `WorkerEvaluators` (one hub per GPU engine, as today), each loop
`i = next()` (blocking; −1 ends the worker), play game i with
`Rng(deriveSeed(seed, g, kPhaseSelfPlay, i))`, `done(i, record)`.
`playSelfPlayGames` becomes this with `next` counting through
[first, first + count): its output is bit for bit today's. A dynamic local
engine's `next` is `GameQueue::take` behind a condition variable.

### Remote engines: one streaming session per generation

A remote engine gets one `selfplay` process per generation that stays up
while the queue lasts:

```
selfplay --stream --job JOB --backend B --workers W [--inference-batch N]
```

**Why a stream rather than chunks of games** (a job of N games per
process, several at once): a GPU engine is efficient only with one hub
and many games in flight. The CUDA hub gives 5,448 evaluations/s with 32
threads and 6,630–6,942 with 64–256, and a pass of 8 positions costs
nearly what a pass of 144 does (4.5 ms against 23 ms) — several small
processes would each run small, separate batches on one GPU, every chunk
would reload the network and recompile or reinitialise the GPU code, and
each chunk's last games would drain its own hub. A stream keeps one hub
with W games in flight, refilled as games end, so its batches stay full
until the queue is empty. It also returns each game as soon as it ends,
so a failure loses only the games in flight.

**The job** is today's `AZSJ` version 1 self-play job with `first = 0`
and `count = G`: the indices the session may be asked for. Its hash names
the session in every answer.

**The commands** (coordinator → the process's stdin), text lines:

```
play <index>      start game <index> (in [first, first + count), never twice)
end               no more games: finish those in flight, then exit 0
```

End of file without `end` means the coordinator or the connection is gone:
the process stops at once (exit 1), as `--exit-on-stdin-eof` does today.

**The answers** (process's stdout → coordinator), binary frames:

```
frame  = magic[4] u32 length payload[length] u64 fnv1a64(magic, length, payload)
AZSH   hello: u64 job hash, str build ID, str engine fingerprint   (first frame)
AZSG   game:  i32 index, the result file's per-record layout (writeGameRecord,
              resignDisabled, both sides' resign lows), f64 seconds, u64 evaluations
AZSE   end:   f64 seconds, u64 positions, u64 evaluations           (after `end`)
```

stderr goes to the session's log, as today. The coordinator verifies:
the hello's job hash and build ID (as a result file's); every game frame
with the same record checks as `readSliceResult` (`verifyRecord`: board
size, komi, every position replayed, legal and visited moves, the end,
the Tromp–Taylor result, resignation flags and lows against v_resign and
the game's seed), plus that the index is one this session was given and
has not returned; the end frame's positions against the games received.
Anything else is a session failure.

**Transports** gain one call, `openStream(args, log)`, returning a child
process with a pipe to its stdin and one from its stdout (stderr to the
log): `ssh … 'exec nice -n N exe --stream …'` for `SshTransport`, `exe`
for `SubprocessTransport`, and the app's `runStreamJob` on a thread with
two pipes for `InProcessTransport`. Uploads, the network cache per machine,
`killJob` and the work directory are today's. The session's file names
carry the generation and the attempt (`g0012_stream_a1.job`).

A session is driven by two threads on the coordinator: a reader that
parses and verifies frames, and a feeder that keeps the session's in-flight
count at W by taking games from the queue and writing `play` lines. When
every game of the generation is finished, the feeder writes `end`, the
reader takes the end frame and the process exits 0.

## Failure handling

A session fails when ssh or the process exits early, a frame does not
parse or verify, or no frame arrives for the hosts file's `timeout`
seconds while it has games in flight (in dynamic mode `timeout` limits the
silence between two games, not a slice's total time). Then:

1. Its in-flight games go back to the front of the queue at once; its
   finished, verified games are kept. Other engines start them as soon as
   they have a free slot.
2. **Retry once**: a new session, attempt 2 (after `killJob` and removing
   the attempt's files), if the machine answers the reachability probe.
3. If the retry fails, or the machine is unreachable, act on
   `selfplay_fallback`:
   - `local`: the engine is retired for the generation. Its games are
     played by the other engines. If every listed engine is retired and
     games remain, the **fallback engine** (the first `local/…` item, or
     `local/<backend>` when there is none) joins the queue. The next
     generation tries the machine again.
   - `stop`: the generation throws, nothing is committed, and a resume
     replays it.

A local engine that throws (a GPU failure) fails the generation, as today.
No game index is ever lost or doubled: the queue hands an index to one
engine at a time, takes it back only from a failed session, and the merge
refuses a missing or doubled index.

## The manifest

`selfplay-engines.csv` keeps its header. In dynamic mode it has one row
per engine item (and per fallback engine that played) per generation:
`first` is empty, `count` the games it played, `planned` and `engine` the
item's engine, `seconds` its busy time (first handout to last game back),
`positions` and `evaluations` its totals, and `note` what went wrong
(`2 failed sessions; 13 games requeued: mini1: selfplay exit 255`) or
empty. `tools/engine_weights.py` reads it unchanged (it ignores `first`
and skips rows with a note).

A new sibling, **`selfplay-games.csv`**, records the assignment, one row
per game, written only in dynamic mode:

```
generation,game,engine
12,0,local/metal
12,1,local/metal
12,2,pc/cuda
...
```

Both are rewritten atomically before `state.bin`, and rows of
generations ≥ g are dropped first, so a redone generation leaves one set
of rows; a resume prunes both to the committed generation. With the
fingerprints in `selfplay-engines.csv` (and the preflight lines), the
pair is the exact recipe of every game.

The progress line is today's `selfplay engines:` line, one item per
engine with the games it played.

## Determinism contract

Game i's record is a pure function of its seed stream, the network, the
`SelfPlayConfig` and the fingerprint and build of the engine that played
it (the existing argument, unchanged). So with `selfplay_allocation =
dynamic`:

- **Given the assignment, the run is byte-identical.** Two runs (or a run
  and a replay) in which every game index was played by an engine of the
  same fingerprint and build write identical checkpoints, `state.bin`,
  rating files, SGFs and `log.csv` (timings aside). The test replays a
  dynamic run's `selfplay-games.csv` with `forceAssignment` on engines of
  two different arithmetics (naive and blas) and compares bytes.
- **Merged in index order.** The order in which games finish plays no
  part.
- **Engines of one fingerprint split freely.** When every engine has the
  same fingerprint (the fake remote and `local/naive` in the tests; two
  identical minis), the assignment cannot change a bit: such a run is
  byte-identical to the static run and to the default path.
- **A resumed generation may differ.** A generation interrupted before
  its commit is redone from the start; its games may be assigned
  differently, so with engines of different arithmetic its last bits may
  differ from the interrupted attempt's (and from an uninterrupted run).
  This is the dynamic counterpart of the arena's documented last-bit
  differences. Committed generations are never touched.

**Not guaranteed**: that a rerun from the config assigns the same games
to the same engines.

## The arena stays static

Arena waves keep `assignGames` slices. A wave is 40–80 games played in
minutes rather than hours, about 10% of a generation; its matches must be
merged per match, its jobs carry several networks, and its static slices
already fall back as a whole. A dynamic arena would need its own stream
job type for a small share of the time. It is not built; the queue and
session code do not preclude it.

## The next feature

The asynchronous evaluator (gating and training beside the next
generation's self-play) needs engines that run more than one kind of job
at once. Nothing here assumes an engine does only self-play: a session is
one process per engine and phase, local engines are worker pools created
per phase, and the arena's slices still run through `runSlices` as
separate processes. `Coordinator::play` keeps its signature (records in
index order), so a later trainer can run it on its own thread beside the
arena.

## Measuring the gain: the simulation

`tests/test_game_queue.cpp` drives `GameQueue` with a discrete-event
simulation of engines: each engine e evaluates at `R_e(n) = rmax_e ·
min(1, n / sat_e)` positions/s with n games in flight, shared equally
among them, and games cost a random number of evaluations (19×19: 760
per ply, plies ~ N(490, 60) clipped to [300, 722]). Static allocation is
`assignGames` with every slice at once. The figures below are filled in
from the C++ simulation (Task 2) and compared against the fluid ideal,
Σ evaluations / Σ rmax.

Fitted to the evidence above: the Studio `rmax` 2,780 and `sat` 4, each
mini 490 and 2, the PC 6,770 and 40.

| allocation (W: Studio/mini/PC) | live weights | minis weighted ×2 | PC weighted short |
|---|---|---|---|
| static | (Task 2) | | |
| dynamic, plain queue, 26/5/64 | | | |
| dynamic, handout rule, 26/5/64 | | | |
| dynamic, handout rule, 24/3/128 | | | |

A real check on one machine (Task 9) runs a short 9×9 run with engines of
different speeds (`local/naive` with 1 worker and a fake remote with 3)
both ways and compares the engines' finish times from the manifest.

## Switching the live run

1. Build, sync the minis and the PC (`tools/cluster_sync.sh`): the build
   ID changes, and the preflight refuses a mismatch.
2. Optionally set the concurrency in the coordinator's hosts file (see
   "Sizing" in the README for the measured basis): `local metal_workers=24`,
   `mini1`/`mini2` `metal_workers=3`, `pc cuda_workers=128`. These keys
   also cap arena slices on those machines (worker threads), which play
   fewer games than that except on the minis (2–4 arena games each, which
   then play in waves of 3).
3. Stop after a commit (`tools/stop_after_generation.sh`) and resume with
   `--set selfplay_allocation=dynamic`.

The resume prints the allocation note. Generations before the switch are
untouched; the redone or next generation is the first dynamic one.

## Configuration summary

| setting | where | default | affects results? |
|---|---|---|---|
| `selfplay_allocation` | config key, tunable | `static` | dynamic: which engine plays which game (recorded) |
| `selfplay_engines` | config key, tunable | empty | static: the slices; dynamic: the engines and their default concurrency |
| `selfplay_fallback` | config key, tunable | `local` | dynamic: whether a twice-failed engine retires or stops the generation |
| hosts `*_workers` | hosts file | 0 | dynamic: concurrency (which engine plays which game) |
| hosts `timeout` | hosts file | 0 | dynamic: longest silence of a session with games in flight |
| `selfplay --stream` | CLI | — | no |

## Testing

- **`test_game_queue`** (new): every index handed out once; lowest first;
  requeue to the front; concurrency respected; retire; kNone only when all
  are finished; the handout rule's cases (unknown latency, in time, C
  counting, never stalling as time passes); forced assignment; the
  simulation's table and its acceptance (dynamic with the rule beats
  static's mean in every scenario).
- **`test_distributed`**: the stream frames round trip; every refusal
  (magic, checksum, length, job hash, build ID, an index not given or
  given twice, a record that does not verify, a bad end frame);
  `runStreamJob` in process; the real `selfplay --stream` through the
  subprocess transport equals in-process slices bit for bit; end of file
  without `end` stops it; `playSelfPlayGames` unchanged against its
  previous output.
- **`test_trainer`**: dynamic with `local/naive` and a fake remote
  (`fake/naive`) is byte-identical to the default path (one fingerprint);
  every index once; manifests written; a remote killed mid-generation
  (`distributed_probe::failSession`) requeues its games, retries, falls
  back, no loss or duplication, and the run continues; with `stop` it
  throws and the resume equals a failure-free run; resume of a dynamic run
  equals uninterrupted (one fingerprint); switching static → dynamic → static
  at boundaries; in BLAS builds, `local/naive` + `local/blas` dynamic, then
  a replay of its `selfplay-games.csv` with `forceAssignment` is
  byte-identical, and a different forced assignment is not (the test has
  teeth).
- **Old against new binaries**: with `selfplay_allocation` absent or
  `static`, main's and the branch's `train` write identical files (but for
  `config.cfg`'s new line) for 5×5 and 9×9, naive, blas and metal, with
  and without engines and a fake remote, straight and resumed.
- **`integration_smoke.sh`**: a tiny dynamic run with a fake remote,
  resumed; `selfplay-games.csv` holds each index once per generation.
- **CUDA build on the PC**: the full ctest in a scratch directory.

## Risks

| risk | consequence | mitigation |
|---|---|---|
| concurrency too low on a GPU engine | lower throughput than static | defaults from the evidence; hosts override; the simulation; `engine_weights.py` rates still apply |
| the handout rule mispredicts (latency varies with load and game length) | a late game extends the phase | the rule only holds when others are predicted to absorb the queue in time; mispredicted slots stop counting as time passes; never stalls |
| a stream breaks mid-generation | lost in-flight work | finished games are kept; in-flight games requeued at once; retry once |
| a protocol bug loses or doubles a game | corrupt generation | the queue hands each index once; the merge refuses gaps and doubles; per-game verification |
| results differ on a resumed generation | not byte-reproducible from the config | documented; the assignment is recorded; `static` remains for exact experiments |
| SIGPIPE when a session's process dies while the coordinator writes | the trainer dies | writes block SIGPIPE on the writing thread and treat EPIPE as a failed session |
| build ID change | remotes refuse | `cluster_sync.sh` before resuming, as for every build |
