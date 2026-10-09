# Arena on the GPU — Design

Author: Mark Castelluccio, with design assistance from Claude (Anthropic).

Amends "Who uses it" (Arena) and "Keeping the GPU busy" in
`docs/superpowers/specs/2026-10-09-metal-gpu-backend-design.md` (Phase A,
merged), and the arena round of
`docs/superpowers/specs/2026-10-01-rating-aware-arena-design.md`. The
algorithm does not change: the same opponents, games, seeds, searches and
results. Only the order in which the arena's games are played changes, and
which evaluators they share.

## Problem

Phase A put arena inference on the GPU through one `InferenceHub` per
network player and match. The arena gained almost nothing. At generation 1
of a 9×9 6×96 run, a single 20-game match took 12.0 s with `blas` and
11.8 s with `metal` (1.02×; the target was 1.3×, `docs/results/bench.txt`).

A generation's arena is a few short matches played one after another.
Later in a run it plays g−1, the strongest reference and two extra
references, 20 games each, plus 80 calibration games when g is
promoted. Each match had its own two hubs. In a match:

- at most 20 games run at once, so at most 20 searches are in flight;
- in each game only the side to move searches, so those 20 searches are
  split between the two networks' hubs, about 10 each;
- a GPU pass of 6×96 costs 2.8 ms at batch 8 and 5.9 ms at batch 144
  (`bench`), so passes of ~80 positions spend most of their time on fixed
  costs, and the GPU waits while the searches expand their leaves;
- the match ends with its longest game, so the last seconds of each match
  run with a few games, so its passes are presumably smaller still.

The new network plays in every match. Played one after another, its
searches never share a pass with those of another match.

## Design

**Play the independent matches of a round at once.** A round has three
waves, and the matches within a wave do not depend on each other:

1. *Calibration.* Fixed players (bots, reference networks) that have
   never played generation 0 play it. When a run starts, that is every
   fixed player. Otherwise it is normally none.
2. *Opponents.* Every opponent `selectOpponents` chose. All of them were
   chosen from the results before the round.
3. *Promotion.* If g scored at least `promote_threshold` against the
   strongest reference, `calibration_games` more games against it. This
   depends on wave 2's result, so it is its own wave.

`runArenaRound` and `calibrateFixedPlayers` hand each wave to a
`MatchesFn`, which returns one result per match in order. The results
are added to the results table in the same order as before, so
`results.csv`, `elo.csv`, `references.csv` and the logs are unchanged.
The old single-match `MatchFn` overloads remain (`oneAtATime` adapts
one), and the arena-round tests use them as a reference.

**One pool of games per wave.** `playMatches(matches, cfg, workers,
backend, inferenceBatch)` turns every game of every match into one task
of one `parallelFor`. Tasks are ordered round-robin across matches (game
0 of each match, then game 1, ...), so every match is under way from the
start. Game j of match i still uses `Rng(deriveSeed(seed_i, j))`, and its
outcome is stored at (i, j). Each match's result is therefore exactly
what `playMatch` gives for that match alone. `playMatch` is now
`playMatches` with one match.

**One set of evaluators per network for the whole wave.** Networks are
identified by address. The trainer reads each checkpoint once per wave,
and the live network and the reference networks are already in memory.

- `metal`: one `InferenceHub` per network. The new network plays in
  every match, so its hub gathers about half of all the searches in
  flight in the wave, and each opponent's hub gathers the rest of its
  own match.
- `naive`, `blas`: each worker has one single-threaded backend and one
  `Workspace` sized for the largest request. It also has one
  `LocalEvaluator` per network, and all of them evaluate in that
  workspace (new constructor `LocalEvaluator(net, Workspace&)`). A
  worker plays one game at a time, and in it only the side to move
  searches, so it never evaluates for two networks at once. A workspace
  is scratch memory, so the outputs do not change. Memory is lower than
  before (one workspace per worker instead of two). That matters
  because a wave can hold five or six networks.
- Bots keep a `UniformEvaluator` per seat, on the CPU, in the same pool.

All networks in a wave must share one `NetConfig` (they always do in a
run). `playMatches` checks every match as `playMatch` does before any
game starts, and it refuses a set whose networks differ in shape.

**Worker count: with metal, every game of the wave at once.**
`arenaWorkers(workers, backend, games)` takes an explicit count as
given. For 0 it gives:
- naive and blas: one worker per core (`resolveWorkers(0)`);
- metal: every game of the wave, up to `kMaxArenaWorkers` (256, the
  `workers` key's limit). Never more than the wave's games.

The trainer passes `workers` unresolved. An arena game has only one
search in flight, the side to move's, so the games in flight are what
fills the passes. A worker waiting for the GPU costs no CPU. This was
measured, not assumed. With the 36 workers of self-play's rule, the wave
was *slower* than matches one after another (59.6 s against 50.0 s): five
hubs split 36 searches, so passes shrank to ~20 positions. With all 80
games it took 38.1 s. CPU backends keep one worker per core. 18 workers
over 80 games leave one tail per wave instead of one per match: 47.0 and
44.5 s against 51.5 s.

**One pass in flight per hub when hubs share the GPU.** A hub alone keeps
two passes in flight (`kInferenceSlots`), so one computes while the next
fills. A wave's hubs share one GPU, so the other hubs' passes keep it
busy, and a second slot per hub only splits what waits into smaller
passes. So with several networks each hub's runner has one slot
(`makeMetalRunner(net, maxBatch, slots)`,
`WorkerEvaluators(..., slots)`). A wave of one network (against bots
only) keeps two. Measured on the four-opponent wave with 80 workers:
- 2 slots: 20,700 passes, the new network's averaging 77 positions and
  the opponents' ~21; 37–40 s.
- 1 slot: 14,300 passes, averaging 115 and ~30; 31.2 s.

Which positions share a pass changes no result.
`inference_batch` is unchanged: 80 searches × 8 positions rarely fill
256 in one hub. Two other ideas were tried and not kept:
- a command queue per hub, so that passes of different hubs could run on
  the GPU at the same time: no change;
- more than 80 workers (capped at the games anyway): the same.

**Who uses it.** Live arena rounds, the calibration when a run starts
(every fixed player at once) and `train --rerate`, which replays the
same rounds. `rerate` cannot run several generations' rounds at once,
because each round's opponents depend on the previous rounds' results.
The `arena` app plays one long match (`playMatch`) and needs nothing
new. It keeps its own worker rule: `--workers 0` resolves to every core
(twice the hardware threads with `--backend metal`), as its usage text
says. The every-game rule was measured only for a round's waves.

**GPU memory.** A wave holds one hub per network at once. When a run
starts, resumes or re-rates with `metal`, the memory check covers two
cases. One is self-play's hub with two slots. The other is the most hubs
a wave can need, at one slot each: `max(1 + reference networks, 3 +
arena_opponents + gating)`, so 5 for the presets and 6 with gating.
Before, it checked two two-slot hubs. At `inference_batch` 256, a
two-slot hub needs 218 MB for 9×9 6×96 and 1,274 MB for 19×19 10×128
(measured with the check itself). A one-slot hub needs a little more
than half that, so a 19×19 wave of five hubs needs ~3.3 GB. That is fine on the 128 GB Studio. On a 16 GB Mac
mini the check may refuse it with its usual message (lower
`inference_batch`) before the run writes anything. `makeMetalRunner` still checks each hub as it is
created.

## Determinism

The argument is Phase A's, applied at a larger scale:

- A game's moves depend only on its own `Rng` (seeded by match seed and
  game index) and on the network outputs for its own positions.
- With `naive` and `blas`, each evaluate call is still evaluated alone,
  at the batch the search asked for, by a single-threaded backend, as
  before. Only the workspace's address and capacity change, and they
  hold no state between calls.
- With `metal`, a position's outputs are a pure function of its own
  input planes and the network's weights, whatever else shares its pass
  (the batch-independence tests). The hub's chaos mode randomizes
  batching to prove it.
- Results are stored by (match, game) and added to the results table in
  wave order, which is `selectOpponents` order, as before.

So a run is byte-identical to what the sequential code wrote, for every
backend, worker count and `inference_batch`, on resume and on re-rating.

## Alternatives considered

| option | verdict |
|---|---|
| one hub for all networks | impossible without multi-network kernels: a pass applies one set of weights |
| play the promotion games speculatively in wave 2 | rejected: 80 games wasted whenever g is not promoted (most generations) |
| overlap the arena with the next generation's self-play | rejected: it changes the commit order of a generation (`state.bin` after the arena) for a few seconds |
| interleave several games per worker (resumable searches) | rejected again, as in Phase A: a large rewrite of teaching code for what more threads give |
| more workers than 2 × cores | **chosen** for metal: every game of the wave (see "Worker count") |
| a command queue per hub | measured: no gain; not kept |
| three slots per hub | moot: one slot per hub is what helped once several hubs share the GPU |

## Testing

- `test_arena`: `playMatches` over a set mixing network against
  network, a network in several matches with different batches, bots in
  either seat, one-sided random symmetry, a simulations override and a
  network against itself. Each result equals `playMatch` alone and the
  per-worker loop from before `WorkerEvaluators`, for workers 1, 3, 8
  and 0, with naive and blas, and through the one-match-at-a-time hook.
  There are refusal tests too.
- `test_arena_round`: a round played in waves gives the same matches,
  seeds, order, results table, opponents and references as one match at
  a time, with calibrations, gating and promotions. Each wave has the
  expected size, and no wave is empty.
- `test_trainer` (naive, blas) and `test_metal` (metal, skipped without
  Metal): a run with two bots, two reference networks, gating and
  promotions (`promote_threshold` 0.5). It is compared, by every
  checkpoint, `state.bin`, the rating files, the SGFs and `log.csv`
  without its timing columns, against the same run played one match at
  a time (`arena_probe::setOneMatchAtATime`). This holds for workers 1,
  3, 8 and 0, `inference_batch` 8 and 256, chaos (metal), resume after
  generation 2, and `--rerate` played either way.
- Old-against-new `train` binaries: short 5×5 and 9×9 runs, byte for
  byte (`docs/results/bench.txt`).

## Performance

The setup: generation 2 of a 9×9 6×96 run started from a copy of
9x9-recipe generation 250, with copies of generations 150, 200 and 230
as reference networks. Its arena plays four opponents (g−1, generation
0, ref1, ref2), 80 games. The training job was paused for every
measurement; details are in `docs/results/bench.txt`.

| | arena_s |
|---|---|
| blas, before | 51.5 |
| metal, before (matches one after another) | 50.0 |
| blas, after | 47.0 / 44.5 |
| metal, after (80 workers, one slot per hub) | **31.2** |

**Acceptance (≥ 1.3× arena phase with metal): met.** Metal is 1.43–1.51×
faster than blas with the same code, 1.65× faster than blas before the
change, and 1.60× faster than metal before it. Old and new binaries wrote
byte-identical files for blas and for metal.

Each hub's slots stay busy ~95% of the wave, so the GPU's speed on
passes of 30–115 positions now sets the pace. The levers left are the
Phase A spec's kernel fusion and implicit GEMM. A single 20-game match
(generation 1, or the arena app) still has too few searches to gain
much.
