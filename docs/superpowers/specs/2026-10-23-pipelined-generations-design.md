# Pipelined Generations — Design

Author: Mark Castelluccio, with design assistance from Claude (Anthropic).

Builds on `2026-10-18-distributed-selfplay-design.md`,
`2026-10-20-distributed-arena-design.md` and
`2026-10-23-dynamic-selfplay-design.md`, whose "The next feature" section
this is. Only scheduling changes: every game is played, every network
trained and every match rated exactly as today.

## Problem

`train` runs a generation as a strict sequence:

```
self-play(g) on every engine -> train candidate g on the Studio -> arena(g) waves on every engine -> commit -> self-play(g+1)
```

The live 19×19 run (`runs/19x19-agz`, 10×128, 400 games of 800
simulations, engines `local/metal`, `mini1/metal`, `mini2/metal`,
`pc/cuda`) shows what that costs. Generation 3: self-play 15,361 s,
training 1,018 s, arena 3,826 s. While the Studio trains, the three remote
GPUs and the Studio's GPU idle; during the arena the GPUs are fed from
waves of 40–120 games.

The manifests give the arena's work: 33.8 M network evaluations in
generation 3 (33.2 M in generation 2), against 144.6 M for self-play at a
combined 10,350 evaluations/s. So the arena is a real GPU workload (about
3,270 s at self-play's rate) played at 85% of that rate. What overlapping
can recover per generation is training (≈1,000 s, all GPUs idle) plus the
arena's under-fill (≈550 s): ≈1,550 s of ≈20,200 s, **≈7–8%** (see
"Expected gain" for the arithmetic and its limits).

## AlphaGo Zero

AlphaGo Zero's pipeline had three components running **asynchronously in
parallel**: self-play generated games with the current best player αθ*;
optimisation trained on the most recent 500,000 games and wrote a
checkpoint every 1,000 steps; the evaluator played each checkpoint
against αθ* (400 games) and, at a 55% win rate, made it the new αθ*,
"subsequently used for self-play generation". So a promoted network
reached self-play only after its evaluation finished, while self-play
went on with the previous best: a lag of one evaluation.

This design runs the same three components side by side, with the one
change that makes it reproducible: the switch of self-play network
happens at a generation boundary chosen by rule, never when an
evaluation happens to finish.

## Goals

- **Opt-in, switchable at a commit.** A tunable key `pipeline = off | on`,
  default `off`. With `off`, a run is byte-identical to `main` (its
  `config.cfg` gains one line, as with `selfplay_allocation`); a run
  switches with `--resume --set pipeline=on` and back the same way.
- **No idle phases.** Self-play of generation g+1 starts as soon as
  self-play of g ends; training and the arena of g run beside it, on the
  same engines.
- **Determinism by rule.** Which network plays a generation's self-play
  is fixed by the schedule, never by timing. A pipelined run is
  reproducible (and resumes byte for byte) under exactly the conditions
  a sequential one is: one arithmetic, or the recorded dynamic
  assignment.
- **Exact crash recovery.** One commit point per generation, as today;
  a crash anywhere redoes only uncommitted work, and the redo is the run
  that never crashed.
- **Nothing else changes**: the replay buffer window, the learning-rate
  schedule (by generation), resignation calibration, the validation loss,
  the arena's opponents and gating rule.

## Non-goals

- Overlapping training with its own generation's self-play (training
  would see a partial buffer, or the buffer would depend on timing).
- Continuous (per-game) switching of the self-play network, as AlphaGo
  Zero's self-play workers did. It would make games depend on timing.
- Training on the GPU (the CPU training phase is off the critical path
  once it runs beside self-play).
- Recovering the self-play tail (the last games of a generation, when
  some engines are idle): the arena of g finishes long before it.

---

## The `pipeline` key

| value | meaning |
|---|---|
| `off` (default) | today: self-play, training, arena, commit, in sequence |
| `on` | each generation's training and arena run beside the next generation's self-play |

Tunable (not an architecture key), printed in the run section after
`backend`. `checkConsistency` refuses nothing new: the pipeline works with
and without engines, gating, resignation and arenas.

## The schedule

Generation g's work is three parts:

- **SP(g)**: self-play of generation g (on the self-play engines);
- **B(g)**: validation loss of generation g−1's network on SP(g)'s
  games, the buffer gains them, training of candidate g, its checkpoint,
  and the arena of g (on the arena engines), which may promote g.
- **the commit** of generation g.

With `pipeline = on` the trainer runs **iterations**. Iteration g
starts with the committed state of generation g−1 *and SP(g)'s games*
(see "What a commit holds"), and:

1. **processes SP(g)'s games** as today's self-play phase ends: the
   statistics, the resignation window and v_resign for g+1, the SGFs;
2. starts **SP(g+1)** on a thread of its own, with v_resign from step 1
   and the network of the rule below;
3. runs **B(g)** on the trainer's thread at the same time;
4. **barrier**: waits for both;
5. **commits generation g** together with SP(g+1)'s games.

```
          iteration g-1                 iteration g                    iteration g+1
SP:   ... SP(g) ----------------------| SP(g+1) ----------------------| SP(g+2) ------ ...
B:    ... B(g-1) --------|            | B(g): val, train, arena |     | B(g+1) ----|
                         idle (waits)                           idle   ^
                                      ^ commit g-1                     ^ commit g: B(g) done, SP(g+1) done
```

The **barrier** is exact: SP(g+2) starts only after commit g, so
after the arena of g has finished and its promotion has been committed.
When B(g) takes longer than SP(g+1) (a very long arena, or no remote
engines), the self-play engines wait for it; the run stays correct, it
only gains less.

### The self-play network rule

> Self-play uses the best network of the **last commit before it starts**
> (`best_generation` with gating, the latest generation without).

Self-play starts only right after a commit, so this is a rule on
generation numbers, not on timing:

- **steady state**: SP(g+1) starts after commit g−1, so it is played by
  the best network after the arena of **g−1** (without gating: network
  g−1). Today SP(g+1) is played by the best after the arena of g (network
  g). **The cost is a one-generation lag**: a promoted network reaches
  self-play one generation later than today. This is AlphaGo Zero's
  lag (a promotion reaches self-play when its evaluation ends, while
  self-play goes on with the previous best), fixed to a generation
  boundary.
- **the prologue**: an iteration needs its generation's games committed.
  When the state has none (a new run, or a resume that turns the pipeline
  on), the trainer first plays SP(g) alone with the best network of the
  last commit (no lag: today's choice) and commits it (below). The
  network that played SP(g) then also plays SP(g+1): around a switch,
  one network plays two generations.

So the training data of generation g+1 comes from a network one
promotion older than today's; candidates are still trained from the
previous candidate, on the buffer as of SP(g), exactly as today.

## What a commit holds

`state.bin` stays the single commit point. A pipelined commit of
generation g holds exactly what today's commit of g holds (network g's
optimizer state, the buffer with SP(1..g), results, references, the best
generation after the arena of g, the resignation window through SP(g)
and v_resign for g+1) **plus the next generation's games, unprocessed**:

```
pending: generation g+1's self-play games, in index order,
         the network that played them (its generation),
         the settings their statistics and SGFs need:
         resignation (on/off), the v_resign they were played with, max_moves
```

`state.bin` version 6 is version 5 plus that block; it is written **only
when there are pending games**, so `pipeline = off` (which never has any)
keeps writing version 5, byte for byte. The pending games are stored as a
self-play result file stores them (`writeSliceRecord`: the record, the
resign-disabled flag and both sides' resign lows), since processing them
needs everything a slice's result carries. The trainer reads versions 1–6.

| moment | `state.bin` | a crash here resumes into |
|---|---|---|
| a new run (or `pipeline` switched on) | generation k, no pending | the prologue |
| prologue committed | generation k, pending SP(k+1) | iteration k+1 |
| during iteration g (SP(g+1) and B(g) in flight, either done) | generation g−1, pending SP(g) | iteration g, redone whole |
| commit of g | generation g, pending SP(g+1) | iteration g+1 |

A crash anywhere inside an iteration loses that iteration's work, both
halves: at most one self-play generation, as today (today a crash in the
arena of g loses SP(g) and the training too). Everything an iteration
writes before its commit (the SGFs of g in step 1, checkpoint g, the
rating files, the manifests) is rewritten with identical bytes by the
redo, as today.

Because every input of an iteration is in `state.bin`, the redo is the
iteration that never crashed: the same games (by index and seed), the
same network for SP(g+1) (the committed best), the same v_resign
(recomputed from the committed window and the pending games), the same
training (buffer, optimizer, seeds), the same arena. With engines of one
arithmetic, or given the recorded dynamic assignment, a run interrupted
and resumed at any moment is byte-identical to the straight run.

### What is processed when

SP(g)'s games are processed at the start of iteration g, not when SP(g)
ends. That is what lets the commit hold them raw: their statistics and
the resignation calibration are a function of the committed state, so a
resumed run recomputes them identically, and v_resign for SP(g+1) is
calibrated from SP(g)'s games before SP(g+1) starts, as today.

## Order inside an iteration

| step | thread | writes |
|---|---|---|
| process SP(g): stats, window, v_resign(g+1), SGFs | trainer | `sgf/gen_g_*` |
| load or copy SP(g+1)'s network; start SP(g+1) | trainer | — |
| SP(g+1) | self-play thread | `selfplay-work/gen_(g+1)` (engines) |
| validation, buffer, training, checkpoint g | trainer | `checkpoints/gen_g` |
| arena of g | trainer | rating files, `arena-work/gen_g_w*` |
| barrier | trainer | — |
| manifests: self-play rows of g+1, arena rows of g | trainer | `selfplay-engines.csv`, `selfplay-games.csv`, `arena-engines.csv` |
| `pending-selfplay.txt` (SP(g+1)'s wall time) | trainer | |
| `state.bin` (generation g, pending SP(g+1)) | trainer | the commit |
| `log.csv` row g | trainer | |

The self-play thread writes nothing in the run directory but its work
directory; everything else is written by the trainer's thread, after the
barrier or before the thread starts. SP(g+1) has its own copy of its
network (copied from memory, or loaded from its checkpoint with gating),
since B(g) trains the trainer's network in place.

## Manifests and `log.csv`

**Manifests.** Self-play rows of generation g+1 are written with commit g
(they describe the pending games); arena rows of g with commit g. A
resume prunes `selfplay-engines.csv` and `selfplay-games.csv` to the
committed generation, plus the pending one when `state.bin` has pending
games, and `arena-engines.csv` to the committed generation. A crash
between the manifests and `state.bin` leaves rows of an uncommitted
generation, which the resume drops, as today.

**`log.csv`.** One row per generation, appended right after its commit,
as today (so `tools/stop_after_generation.sh` works unchanged). Row g
holds SP(g)'s statistics and timing, training of g and the arena of g.
With `pipeline = on` the three timing columns overlap: `selfplay_s` of g
ran beside the training and arena of g−1, and `train_s` and `arena_s`
of g beside the self-play of g+1, so they no longer add up to a
generation's wall time. The progress output adds a line per iteration
with that wall time:

```
gen 12 | selfplay 400 games avg 474.9 plies 15361.4s | train loss p 5.83 v 0.52 lr 0.02 2032.0s | arena ... 6013.7s | best 11
pipeline: gen 13 self-play by network 11, 15402.6s beside gen 12's training and arena 8045.7s; generation 12 took 15403.1s
```

`selfplay_s` of the pending generation is measured by the process that
played it. `state.bin` must not hold timings (a resumed run's would differ
from the straight run's), so it is kept beside it in
`pending-selfplay.txt` (`generation = 13`, `seconds = 15402.583`),
written before `state.bin`; a resume uses it when its generation is the
pending one, and writes `0.000` when it is not there.

## Switching

- **On**, at any commit: the resume's first `runGeneration` plays the
  prologue (SP(k+1) with today's network, committed with generation k),
  then iterates.
- **Off**, at any commit: a pending generation is committed data; the
  sequential loop uses it instead of playing SP(k+1) (its manifests rows
  are already written), then goes on as today.
- **Other keys changed on resume** take effect, as today, from the first
  work not yet committed. With pending games, the self-play of k+1 is
  committed, so a change of a self-play key (`simulations`, `komi`,
  `resignation`, `selfplay_engines`, ...) applies from k+2; the resume
  says so:

  ```
  note: generation 11's self-play games were committed before this resume (pipeline); changed self-play keys (resignation) apply from generation 12
  ```

  A switch that must take effect at k+1 (the 19×19 plan turns
  resignation on at generation 10) is made at a commit without pending
  games: before turning the pipeline on, or together with it.

A run stopped by `generations = N` commits N with SP(N+1) pending: the
last iteration plays the next generation's self-play as every iteration
does, so a run continued later is the run that never stopped (stopping
at N without it would make SP(N+1)'s network depend on where the run was
stopped). The cost is one self-play generation of extra work at the very
end of a run; `tools/stop_after_generation.sh` stops a run sooner.

## The coordinator with two phases at once

With engines, SP(g+1) (`Coordinator::play`) and the arena of g
(`Coordinator::playArena`) run on two threads against one coordinator.
Each kind still runs one phase at a time; what they share changes:

- **Machine states** (`machines_`): created by the phase that first
  meets a machine; now every lookup and insertion holds a mutex. The
  per-machine fields stay under the machine's own mutex, which now also
  covers `down` and `keep` when a phase starts (today set while no thread
  runs).
- **Networks kept on a machine**: `keep` becomes one set per phase kind
  (self-play, arena). A phase removes only networks neither kind needs,
  so the arena never deletes self-play's network on a machine both use,
  nor the other way round.
- **Failed attempts** (`failedSlices_`, which keeps a phase's work
  directory and logs): one flag per phase, not per coordinator.
- **Fingerprints** cache: under a mutex.
- **Cancellation**: `Coordinator::cancel()` stops every phase in flight
  (each phase's slices and sessions see their `stop` flag within 0.1 s);
  the trainer calls it when the other half of the iteration fails, so a
  failed arena does not wait hours for self-play to finish, nor the other
  way round. The flag is cleared at the start of an iteration.
- **The log**: the coordinator's notes already go through one mutex; the
  trainer prints only after the barrier.

Work files never collide: self-play uses `selfplay-work/gen_NNNN` and
`gNNNN_*` names on remotes, the arena `arena-work/gen_NNNN_wW` and
`arena_gNNNN_*`. A remote machine listed in both runs a stream (or a
slice) and an arena slice at once: two `selfplay` processes, two hubs
on its GPU, each verified as today.

The BLAS backend sets its environment once with `setenv`, which is not
thread-safe; the pipelined trainer does that first call before any
thread starts.

## GPU memory

A local GPU now holds self-play's hubs and the arena's at once (and, with
`backend = metal`, the validation loss's runner before the arena). With
`pipeline = on` the up-front checks add them up instead of checking each
phase alone:

- the trainer (`backend = metal`): self-play's hub (two slots) plus the
  larger of the arena's hubs (`arenaWaveNetworks`, one slot each) and the
  validation runner (one slot at `batch_size`);
- the coordinator's preflight, per local GPU backend: self-play engines'
  hubs plus arena engines' hubs.

`checkMetalInferenceMemory` (and its CUDA twin) gain an overload taking a
list of (hubs, slots, batch) groups. Each hub's allocation is still
checked against what the process already holds when it is made, as
today. Remote machines are not checked up front (as today); a 10×128
19×19 hub at batch 256 needs a few hundred MB.

## Training beside self-play on one machine

Training runs on the CPU (BLAS, `train_threads`) while the Studio's GPU
engine plays self-play from worker threads that mostly wait for the GPU.
Measured on the Studio (M5 Max, 6+12 cores) with the live run's
self-play on its GPU, 19×19 10×128, batch 256 (`bench`):

| training | step alone | step beside a 36-thread GPU hub | hub rate |
|---|---|---|---|
| none | — | — | 1,443/s |
| 18 threads, default QoS | 1.30 s | 2.40 s | 1,413/s |
| 6 threads, default QoS | 1.31 s | 1.74 s | 1,348/s |
| 18 threads, utility QoS | 2.84 s | — | 1,368/s |

(The hub's rate is shared with the live run on the same GPU, so the
column is noisy at ±5%; the implementation repeats the measurement
before it is recorded in `docs/results/bench.txt`.)

- Training saturates at about 6 threads (memory and the matrix units,
  not cores), so more threads only compete with self-play.
- The utility class makes training 2.2× slower and does not measurably
  help the hub.
- Training is not on the critical path: 800 steps at 2.4 s are 1,900 s,
  far below self-play's ≈15,000 s.

So the design adds no new QoS mechanism; training keeps `train_threads`
and the default class, and the recommendation for the live run is
`train_threads = 6` (a tunable key; results do not depend on it).

## What stays exactly as today

- **Resignation**: v_resign for g+1 is calibrated from the window
  through SP(g), before SP(g+1) starts; the statistics of g use the
  v_resign SP(g) was played with (stored with the pending games, so a
  resume that changed the resignation keys still reports what was used).
- **Replay buffer**: SP(g) joins the buffer before candidate g trains;
  SP(g+1)'s games are not in it until iteration g+1 (training never
  sees a partly played generation). Window and capacity unchanged.
- **Learning rate**: `lr_schedule` at generation g, as today.
- **Validation loss**: network g−1 on SP(g)'s games before they join the
  buffer, and on a buffer sample, as today. (Without gating today
  network g−1 also played SP(g); in the pipeline network g−2 did.)
- **Arena and gating**: the same opponents, seeds and rule.
- **Seeds**: every stream is `deriveSeed(seed, generation, phase,
  index)`; no new phase.

## Failures

- **Self-play fails** (a local engine throws, or `selfplay_fallback =
  stop`): the trainer cancels B(g) (the coordinator's arena phase, and
  training at its next step), waits for it, and throws the self-play
  error. Nothing of the iteration is committed; a resume redoes it.
- **B(g) fails** (non-finite loss, an arena slice with `stop`): the
  trainer cancels SP(g+1), waits for it, throws B(g)'s error ("last good
  state kept"). The default path's arena (`playMatches`) cannot be
  cancelled; it is short and runs to its end.
- **Fallbacks** (`selfplay_fallback = local`) work as today in each
  phase.
- **A kill** (SIGTERM, SIGKILL, power): `state.bin` is the last commit;
  see "What a commit holds".

## Determinism contract

With `pipeline = on`:

- **The schedule is a rule.** Which network plays each generation, which
  games join the buffer before each training, which arena decides each
  promotion: all fixed by generation numbers and the commit in which the
  pipeline was switched on.
- **Byte-identical where today's runs are.** On one arithmetic (or
  with the recorded `selfplay-games.csv` of dynamic allocation replayed),
  two pipelined runs, and a pipelined run against the same run killed at
  any moment and resumed, write identical checkpoints, `state.bin`,
  rating files, SGFs and `log.csv` (timings aside).
- **Not the sequential run.** `pipeline = on` is a different (equally
  reproducible) run: its self-play networks lag by a generation.

## Expected gain

Per generation of the live run (generation 3, dynamic allocation's
efficiency not yet included):

| | sequential | pipelined |
|---|---|---|
| self-play work | 144.6 M evaluations | 144.6 M |
| arena work | 33.8 M evaluations, its own 3,826 s | 33.8 M, sharing the engines with self-play |
| training | 1,018 s, GPUs idle | beside self-play, ≈2,000 s on the Studio's CPU |
| wall time | 15,361 + 1,018 + 3,826 = 20,205 s | ≈ (144.6 + 33.8) M / 9,413 per s = 18,950 s |

9,413/s is generation 3's achieved self-play rate (144.6 M in 15,361 s);
the pipelined iteration plays both workloads at it, and its length is set
by self-play's end, with the arena's evaluations folded in. The gain is
training plus the arena's under-fill, ≈1,250 s or **≈6%**; with dynamic
allocation's better balance (rate ≈9,870/s) ≈1,450 s, ≈7%. It is less
than the 10–15% guessed at first because the arena is mostly real GPU work
that the pipeline moves but cannot remove. The ceiling of any scheduling
change is training plus the arena's under-fill plus self-play's own tail.

A real check (implementation, recorded in `docs/results/bench.txt`): a
small 9×9 run on the Studio with two CPU engines of different speeds
(`local/naive` with 2 workers and a fake remote with 1), dynamic
allocation, gating, 5 generations, pipeline off against on, wall time per
generation from the progress lines.

## Switching the live run

1. Build at this branch's commit in a new `runs/bin-r28/`; sync the minis
   and the PC with `tools/cluster_sync.sh` (the build ID changes; the
   preflight refuses a mismatch).
2. Stop the r27 job script first (it would resume train on its own),
   then train after a commit (`tools/stop_after_generation.sh`).
3. Resume with `--set pipeline=on` (plus, at the generation-10 boundary,
   `resignation=true`, so that SP(11) is the prologue's and plays with
   resignation), optionally `--set train_threads=6`.

The resume's first output is the prologue: generation k+1's self-play,
alone, committed with generation k. Switching back is `--set
pipeline=off` at any commit.

## Configuration summary

| setting | where | default | affects results? |
|---|---|---|---|
| `pipeline` | config key, tunable | `off` | yes: self-play networks lag one generation |
| `train_threads` | config key, tunable | 0 | no |

## Testing

- **`test_config`**: the key's default, printing position, round trip,
  refusal of other values, tunable on resume.
- **`test_run_state`**: version 6 round trip with pending games (records
  with resign lows, the settings); a state without pending games is
  version 5, byte-identical to before; refusals of a damaged block;
  `trainingStateGeneration` reads 6; `loadTrainingState` of 1–5 as before.
- **`test_pipeline`** (new, the trainer):
  - pipeline off is the sequential run (it runs the same code); a run
    switched off at a commit with pending games uses them;
  - straight vs. stopped at every generation and resumed: byte-identical
    (naive, blas; default path, `local/naive` engines, a fake remote with
    static and dynamic allocation, arena engines);
  - a crash at every point of an iteration (inside SP(g+1), validation,
    training, checkpoint, arena, after the manifests, after `state.bin`,
    in the prologue), resumed: byte-identical to the straight run;
  - the lag: with gating and a gate every candidate passes, the pending
    games of commit g replay exactly from network g−1's checkpoint (the
    best after the arena of g−1) and differ from network g's; without
    gating from network g−1; the prologue's from the committed best;
  - failures: a non-finite loss with self-play in flight (cancelled, state
    kept); a self-play `stop` failure with the arena in flight; resume
    equals the failure-free run;
  - resignation statistics and the window equal what a sequential run
    computes from the same games; a resume that changes resignation keys
    reports the v_resign actually used;
  - manifests and pruning; `log.csv` rows; the progress lines;
  - switching on and off at commits; the note on changed self-play keys.
- **Old against new binaries**: with `pipeline` absent or `off`, main's and
  the branch's `train` write identical files (but for `config.cfg`'s new
  line) for 5×5 and 9×9, naive, blas and metal, with and without engines
  and a fake remote, straight and resumed.
- **`integration_smoke.sh`**: a tiny pipelined run with a fake remote,
  killed with SIGKILL at several moments and resumed, equal to the
  straight run.
- **Thread sanitizer**: `test_pipeline`, `test_dynamic_selfplay`,
  `test_trainer` in a TSan build.
- **CUDA build on the PC**: the full ctest in a scratch directory.

## Risks

| risk | consequence | mitigation |
|---|---|---|
| a data race between the two phases in the coordinator | wrong network removed, crash | shared state enumerated above, each under a mutex; TSan; review |
| a remote GPU holds two hubs | memory pressure on the minis | small networks; the remote processes check their own allocations |
| B(g) longer than SP(g+1) | engines wait at the barrier | correct, only less gain; the progress line shows it |
| the lag slows learning | fewer effective generations | one generation, AlphaGo Zero's own lag; switch off at any commit |
| a self-play key changed on resume applies a generation late | surprise | the resume prints a note; switch at a commit without pending games |
| an old binary meets a version-6 `state.bin` | it refuses ("unsupported version 6") | switch off and commit once (version 5 again) before going back |
| training slows the Studio's self-play | longer iterations | measured (above); `train_threads = 6` |
