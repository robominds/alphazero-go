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
  run with a few games and passes of 8–16 positions.

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
  own match. With 36 workers and 80 games, ~18 searches share the new
  network's passes, against ~10 before.
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

**Worker count.** `arenaWorkers(workers, backend, games)`: an explicit
count as given; 0 gives `resolveWorkers(0, metal)` (twice the hardware
threads with metal, the cores otherwise), never more than the wave's
games. The trainer passes `workers` unresolved. A wave of 80 games keeps
all 36 metal workers busy until its last games. Before, a 20-game match
used at most 20 of them. CPU backends gain too: 18 workers over 80 games
leave one tail per wave instead of one per match. Running more workers
than twice the cores is a lever for later. It is not on by default,
because it raises the CPU load of a machine that may also be running
another job (see "Performance").

**Pipeline slots and `inference_batch`** stay as they are. With 36
workers × `mcts_batch` 8, at most 288 positions are in flight across
all of a wave's hubs, and one hub's pass rarely approaches 256. A third
slot is a lever for later, kept only if `bench` shows it helps.

**Who uses it.** Live arena rounds, the calibration when a run starts
(every fixed player at once) and `train --rerate`, which replays the
same rounds. `rerate` cannot run several generations' rounds at once,
because each round's opponents depend on the previous rounds' results.
The `arena` app plays one long match (`playMatch`) and needs nothing
new. It now passes `--workers` unresolved, which resolves the same way.

**GPU memory.** A wave holds one hub per network at once. When a run
starts, resumes or re-rates with `metal`, the memory check counts the
most a wave can need, `max(1 + reference networks, 3 + arena_opponents
+ gating)`, instead of two: 5 hubs for the presets, 6 with gating. At
`inference_batch` 256 a hub needs 218 MB for 9×9 6×96 and 1,274 MB for
19×19 10×128 (measured with the check itself), so a 19×19 wave of five
hubs needs ~6.4 GB. That is fine on the 128 GB Studio. On a 16 GB Mac
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
| more workers than 2 × cores | kept as a lever: no change to results, measured before it becomes a default |

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

Measured on one 9×9 6×96 generation whose arena has four opponents,
`blas` against `metal`, old against new code. The numbers and
conditions are in `docs/results/bench.txt`. Acceptance: arena phase
≥ 1.3× faster with `metal` than with `blas`.
