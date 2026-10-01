# Measured results

This directory holds the results of the reference trainings, made with
the final code (after the pass fix, commit 5da9b90). Every number below
names the file it comes from. Runs were made 2026-09-30 to 2026-10-01;
the follow-up matches and analyses were made on 2026-10-01.

## Machine

From `bench.txt`: Apple M4, 10 cores, 24 GiB. Single-thread inference
at batch 8, BLAS (Accelerate) backend:

| board | network | evaluations/s |
|---|---|---|
| 5×5 | 3×32 | 30,488 |
| 9×9 | 6×64 | 3,696 |
| 19×19 | 10×128 | 229 |

At batch 1 the rate is about 1.3× lower: 5×5 23.0k–23.3k, 9×9 2.72k–2.77k
evaluations/s (`bench.txt`, last table).

## Presets

The values are from `configs/*.cfg`. The comments in those files give
the reasons. All three presets use:

- `backend = blas`, `workers = 0` (every core);
- `history = 8`, `mcts_batch = 8`, `c_puct = 1.5`;
- `dirichlet_scale = 10`, `noise_fraction = 0.25`;
- momentum 0.9, `weight_decay = 1e-4`;
- `arena_simulations = 100`, gating off.

| key | 5×5 | 9×9 | 19×19 |
|---|---|---|---|
| komi | 24.5 | 7.5 | 7.5 |
| blocks × filters (value_hidden) | 3 × 32 (32) | 6 × 64 (64) | 10 × 128 (256) |
| generations | 60 | 150 | 20 (run for 2) |
| games_per_generation × simulations | 200 × 200 | 100 × 200 | 20 × 400 |
| temperature_moves | 8 | 20 | 30 |
| buffer_positions | 100,000 | 200,000 | 500,000 |
| train_steps × batch_size | 200 × 128 | 200 × 256 | 80 × 256 |
| lr_schedule | 0:0.02,40:0.002 | 0:0.02,90:0.002 | 0:0.02 |
| arena_games × arena_opponents (+ gen 0) | 20 × 4 | 20 × 4 | 10 × 2 |
| arena_temperature_moves | 2 | 4 | 4 |

## Results by board size

Notes on the columns:

- "Wall time" comes from `runs/reference-runs.log`, which is not in the repository.
- "Final Elo" is the last row of `elo.csv`: a Bradley-Terry refit anchored at generation 0 = 0. The 19×19 check has no `elo.csv` copied here; its figure is the last `elo` in `19x19/log.csv`.
- "vs gen 0" is a 100-game `arena` match of the final generation against generation 0:
  - 200 simulations, colors alternating;
  - 95% Wilson interval as `arena` prints it;
  - the commands are in the `final-arena.txt` files.
- Game lengths are `avg_length` in `log.csv` for generation 1 and the last generation.

| run | generations | wall time | final Elo | vs gen 0 | avg plies, first → last |
|---|---|---|---|---|---|
| 5×5 seed 1 (`5x5/`) | 60 | 11.4 min | 1117.5 | 100/100 (0.963–1.000) | 43.1 → 43.3 |
| 5×5 seed 2 (`5x5-seed2/`) | 60 | 11.8 min | 852.5 | 100/100 (0.963–1.000) | 39.9 → 43.7 |
| 9×9 (`9x9/`) | 150 | 10 h 10 min | 674.5 | 100/100 (0.963–1.000) | 88.6 → 93.0 |
| 19×19 (`19x19/`) | 2 | 3 h 08 min | 206.2 | not measured (training arena: 9/10) | 145.0 → 364.5 |

### 5×5: criterion met by both runs (2 of 2)

The criterion has two parts:

- each final network scores at least 90% against generation 0;
- seed 1's Elo rises clearly: the mean of the last 10 generations is at
  least 200 above the mean of the first 10.

Both seeds meet both parts:

| run | vs gen 0 | `log.csv` Elo, last 10 − first 10 | `elo.csv` Elo, last 10 − first 10 |
|---|---|---|---|
| seed 1 | 100/100 | 1137.6 − 668.1 = +469.5 | 1089.1 − 349.7 = +739.4 |
| seed 2 | 100/100 | 910.5 − 549.2 = +361.3 | 875.1 − 311.9 = +563.3 |

Both curves rise and then go flat. The plateau starts around generation
34 for seed 1 and generation 31 for seed 2.

- The refit's decade means:
  - seed 1: 349.7, 630.1, 910.0, 1092.2, 1086.8, 1089.1;
  - seed 2: 311.9, 638.7, 809.7, 927.1, 894.5, 875.1.
- From then on, Black wins at least 85% of self-play games (`black_win_rate`).
- The network plays 5×5's known perfect line. On the empty board, the
  final prior puts 96.8% on the center point, C3, in both seeds.
- A color-balanced arena then scores about 50% whatever the strength.
  Seed 1's matches against g − 1 average 0.49 over the last 20 generations.

The final networks remain unbeaten by generation 0 (`final-arena.txt`).
`sample.sgf` is a generation-60 self-play game, won B+0.5.

### 9×9: criterion not met (the eye check fails)

The criterion has three parts:

- Elo still rises over the last third (a linear fit over those generations has a positive slope);
- the atari check passes;
- the eye check passes.

**Elo slope.** A least-squares fit of `elo.csv` over generations 101–150
gives **+0.41 Elo per generation** (generations 100–150: +0.44). This is
positive, so this part is met on its sign. But it amounts to +20 Elo over
50 generations, far inside the noise of 20-game matches, and the curve
looks flat:

- the refit rises from 0 to 784.8 at generation 54;
- it lies between 601 and 735 from generation 80 on;
- it ends at 674.5.

`log.csv`'s rating at the time falls over the same generations, with a
fitted slope of −2.98 Elo per generation. It peaked at 1211.0 at
generation 54, was 1050.6 at generation 85, and ends at 674.5.

**Is the plateau real? No: the late networks are clearly stronger.**
Five 100-game head-to-heads settle it. Each uses 200 simulations, komi
7.5 and alternating colors; the files are `9x9/h2h-*.txt`.

| match | score | 95% CI | as an Elo gap |
|---|---|---|---|
| gen 150 vs gen 50 | 98/100 | 0.930–0.994 | +676 |
| gen 150 vs gen 90 | 86/100 | 0.779–0.915 | +315 |
| gen 150 vs gen 100 | 76/100 | 0.668–0.833 | +200 |
| gen 150 vs gen 120 | 67/100 | 0.573–0.754 | +123 |
| gen 90 vs gen 50 | 89/100 | 0.814–0.937\* | +363 |

\* `arena` printed the gen 50 side: 11/100, CI 0.063–0.186.

The results are mutually consistent:

- gen 50 → 90 (+363) plus gen 90 → 150 (+315) is +678, against +676 measured directly;
- the improvement rate does slow:
  - about 9 Elo per generation from 50 to 90;
  - about 4–5 per generation from 90 or 100 to 150.

So the late networks are not weaker than generation 50. Every later
checkpoint tested beats every earlier one, by a margin that grows with
the generation gap. The flat refit is an artifact of how it is anchored.

1. **Generation 0 caps the ratings.** From generation 14 on, every
   generation beats generation 0 by 18–2 or better; 110 of the 150
   matches are 20–0 (`9x9/results.csv`).
   - With the virtual draw that `fitBradleyTerry` adds to each pair, a
     20–0 pair says "400·log₁₀(20.5/0.5) = 645 Elo above generation 0".
   - The 137 pairs scoring 18–2 or better (generations 14–150; 110 of them 20–0)
     pull every late rating toward 645. The late
     refit values, 601–735, sit right on that cap.
2. **The ladder cannot see 4 Elo per generation.** In generations
   101–150, newer networks score 9.6, 10.6, 10.3 and 10.7 of 20 against
   g − 1, g − 2, g − 4 and g − 8.
   - Those (47.8%, 52.8%, 51.6%, 53.7%) are consistent, within noise,
     with the 50.6–54.6% that +4 Elo per generation predicts.
   - But each 20-game result is ±150 Elo, so the ladder's upward pull is
     too weak to beat the anchor's.
3. **Evidence for both points.** Refit the same table without the 150
   pairs against generation 0 (`9x9/elo-without-gen0.txt`, a Python port
   of `fitBradleyTerry` that first reproduces `elo.csv` to within 0.05):
   - the curve then rises steadily: generations 60/90/120/150 are +75,
     +242, +367 and +449 above generation 50;
   - the slope over generations 101–150 is +2.78 Elo per generation.

Possible reasons the progress slowed (not reversed) after generation 90.
These are plausible, not tested:

- **The learning-rate drop at generation 90** (0.02 → 0.002) shrinks
  every update tenfold. The head-to-head rate roughly halves around then.
- **The replay window.** 200,000 positions is about 21 generations of
  games, so late training mostly refits the network's own recent play.
- **Policy sharpening.** Policy loss falls from 3.14 at generation 50 to
  2.47 at generation 150, and narrower priors explore less. Self-play
  `black_win_rate` also drifts from decade means of 0.45–0.51
  (generations 11–50) to 0.23–0.29 (generations 61–150): White wins most
  games at komi 7.5.

**analyze checks** (`9x9/analyze-*.txt`): final checkpoint gen_0150,
200 simulations, komi 7.5. Each move list was first checked for
acceptance on gen_0000.

| check | position (moves from the empty board) | target | result |
|---|---|---|---|
| `atari` | `D5 E5 F5 C3 E6 G3`: White E5 is surrounded on three sides, its last liberty is E4, Black to move | E4 top-visited | **pass** as defined: E4 is rank 1 by visits (63/200, 31.5%) and rank 1 by prior (51.4%); but its visits are prior-driven, since its search Q of −0.02 is the lowest of the five most visited moves |
| `eye` | `A2 A3 B2 B3 C2 C3 D2 D3 D1 E3 F7 E2 G5 E1`: the Black group A2–D2, D1 encloses the straight three A1–B1–C1 against a solid White wall | B1 in the top 3 by visits | **fail**: B1 has a prior below 1% and is not among the 10 most visited moves (the 10th has 4 visits); the top 3 are C6 (39), D6 (18) and E5 (18) |
| `opening` | empty board | (record only) | prior on E5 (tengen) 11.2%, then E6 9.6%, E4 and D5 8.6%, F5 8.1%; search visits E5 39/200; value −0.54 for Black at komi 7.5 |

In the `eye` position, B1 makes two eyes (A1 and C1). If White plays B1
instead, the group is dead.

`9x9/analyze-eye-probe.txt` shows the network does not understand this.

- The values it gives, from Black's side:
  - −0.76 after Black's saving B1;
  - −0.65 after White's killing B1;
  - −0.26 after Black tenukis to C6 and White kills with B1.
- In other words, it rates the group's death as no worse for Black than
  its life.
- This is a real gap in the 150-generation 9×9 network's life-and-death
  knowledge, not a search artifact. At 200 simulations, search never
  looks at B1 long enough to find out.

`9x9/sample.sgf` is a generation-150 self-play game, won W+6.5.

### 19×19: criterion met (it runs)

The criterion is that both generations complete, the losses are finite
and the checkpoints load.

| generation | self-play | train | arena | total | avg plies | losses p / v |
|---|---|---|---|---|---|---|
| 1 | 2,809.5 s | 378.9 s | 302.7 s | 58.2 min | 145.0 | 5.82 / 0.24 |
| 2 | 6,630.8 s | 380.7 s | 771.6 s | 129.7 min | 364.5 | 5.68 / 0.26 |

Sources are `19x19/timing.txt` and `19x19/log.csv`. Both checkpoints
load in `analyze`. `19x19/analyze-gen2.txt` shows generation 2 (exit
status 0); generation 1 was loaded the same way.

Generation 2 took more than twice as long as generation 1. Its self-play
games were 2.5 times longer (364.5 plies on average; the cap is 722).
Two generations say nothing about strength, and none is claimed.

## What did not work

1. **The 9×9 eye check failed.** The data is above. The final 9×9
   network misjudges a basic two-eye shape, so the 9×9 criterion as a
   whole is not met.
2. **The 9×9 Elo curve does not show the progress that happened.** The
   slope criterion is met on its sign only (+0.41 Elo per generation).
   - The refit is flat from about generation 50 on.
   - The rating at the time falls.
   - The 100-game head-to-heads show steady but slowing improvement
     throughout: gen 150 beats gen 50 98/100.
   - The rating method saturates once every generation sweeps the
     generation-0 anchor. A fixed panel of stronger reference checkpoints
     would fix that; the code was not changed for this report.
3. **The first set of reference runs was invalid: the pass bug.** The
   runs made before commits 87b37ea and 5da9b90 had a search blind spot.
   - At low simulation counts, a game-ending pass reply with a tiny prior
     was never visited.
   - So trained networks passed into Tromp-Taylor losses with dead stones
     still on the board.
   - Generation 0, with flat priors, exploited it: gen 150 beat gen 0 only
     67/100, and the 9×9 refit ended at 91.8.
   - `passbug/README.md` has the symptom, the diagnosis, the fix and a
     before/after table. All four runs above were redone after the fix.
4. **5×5 Elo goes flat once the game is solved.** This is not a missed
   criterion, but the curve is "rise, then plateau", not "rising to the
   end". After about generation 31–34, Black wins almost every self-play
   game and the arena loses resolution.

## Files

- `bench.txt`: throughput, from `bench` plus a one-off batch 1 vs batch 8 timing.
- `pilots/`: the short pilot runs used to size the presets. These were
  made with the pre-fix code; they were used for timing and komi
  selection only.
- `5x5/`: seed 1's `log.csv`, `elo.csv`, `results.csv`, `sample.sgf`, and
  `final-arena.txt` (both seeds).
- `5x5-seed2/`: seed 2's `log.csv`, `elo.csv`, `results.csv`.
- `9x9/`:
  - `log.csv`, `elo.csv`, `results.csv`, `sample.sgf`;
  - `final-arena.txt`, the `h2h-*.txt` head-to-heads;
  - `elo-without-gen0.txt`, and `elo-without-gen0.csv`, every generation of its second fit;
  - `analyze-{atari,eye,opening}.txt` and `analyze-eye-probe.txt`.
- `19x19/`: `log.csv`, `timing.txt`, `analyze-gen2.txt`.
- `passbug/`: the pre-fix runs and the case study.
- `elo.svg`, `elo-5x5.svg`, `elo-9x9.svg`, `elo-passbug.svg`: Elo charts
  drawn from the CSV files above by `tools/plot_elo.py` (the commands are in
  the main `README.md`).
