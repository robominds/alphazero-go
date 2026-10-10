# Measured results

This directory holds the results of the reference trainings, made with
the final code (after the pass fix, commit 15db72a). Every number below
names the file it comes from. Runs were made 2026-09-30 to 2026-10-01;
the follow-up matches and analyses were made on 2026-10-01. The 9×9
experiments that follow them ran until 2026-10-08.

## Headline: the 9×9 preset now trains a much stronger network

The 9×9 preset's four changes (`games_per_generation` 100 → 200,
`train_steps` 200 → 400, `temperature_moves` 20 → 7 and resignation;
the experiments' "recipe") were first tested by continuing from a
trained network, where they beat every other setting measured (+170 Elo
over the control, pooled over generations 46–50) and were still
improving at generation 250 (253/400 against their own generation 200).
Run from scratch for 250 generations, with the same 6×64 network, seed
and random generation 0 as `9x9-main`, the earlier preset's run, the new
preset's network beats the earlier one:

- at equal generations, 358/400 at 50 and 100, 363/400 at 150 and
  366/400 at 250 (about 0.9, +370 to +415 Elo);
- at equal SGD steps, 238/400 (+67) with generation 75 against 150 and
  318/400 (+235) with generation 125 against 250, having made about 70%
  of the self-play positions;
- with its generation 150 against the earlier run's last, 250: 341/400
  (0.853, +305).

The two versions of the recipe, both at generation 250, were also
played against each other: the continued run's 6×96 network, which is
twice the size and had 300 generations of earlier training behind its
start, beat the from-scratch 6×64 network 244/400 (0.610, +78). That
compares two training paths, not network size or history alone.

One seed and one board size, 400-game matches at 200 simulations. See
[The 9×9 preset from scratch](#the-99-preset-from-scratch-250-generations),
[The recipe](#the-recipe-four-changes-together-250-generations) and
[The recipe against the preset from scratch](#the-recipe-against-the-preset-from-scratch-generation-250).

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

- `backend = blas`, `workers = 0` (every core), `train_threads` unset
  (0: the training phase on every core too; added 2026-10-08, it does
  not change results);
- `bn_calibrate` unset (true: a new run's generation 0 has its batch
  norm calibrated; added 2026-10-10, after every run on this page, so
  their generation 0 was uncalibrated and a new run with the same seed
  starts elsewhere; `--set bn_calibrate=false` reproduces them);
- `history = 8`, `mcts_batch = 8`, `c_puct = 1.5`;
- `dirichlet_scale = 10`, `noise_fraction = 0.25`;
- momentum 0.9, `weight_decay = 1e-4`;
- `arena_simulations = 100`, gating off.

| key | 5×5 | 9×9 | 19×19 |
|---|---|---|---|
| komi | 24.5 | 7.5 | 7.5 |
| blocks × filters (value_hidden) | 3 × 32 (32) | 6 × 64 (64) | 10 × 128 (256) |
| generations | 60 | 150 | 20 (run for 2) |
| games_per_generation × simulations | 200 × 200 | 200 × 200 | 20 × 400 |
| temperature_moves | 8 | 7 | 30 |
| resignation | off | on | off |
| buffer_positions | 100,000 | 200,000 | 500,000 |
| train_steps × batch_size | 200 × 128 | 400 × 256 | 80 × 256 |
| lr_schedule | 0:0.02,40:0.002 | 0:0.02,90:0.002 | 0:0.02 |
| arena_games × arena_opponents (+ gen 0) | 20 × 4 | 20 × 4 | 10 × 2 |
| arena_temperature_moves | 2 | 4 | 4 |

The reference runs used the ladder (`arena_opponents = 4` plus generation
0; 2 plus generation 0 on 19×19). The presets now choose opponents by
rating (`arena_opponents = 2`, `bot_simulations = 100`,
`calibration_games = 80`, `promote_threshold = 0.85`); the re-rated
results below use those values. (The presets have since dropped the bot,
`bot_simulations =`, in favour of optional reference networks; see the
main README. The results here were measured with it.)

The 9×9 preset has since adopted the experiments' "recipe":
`games_per_generation` 100 → 200, `train_steps` 200 → 400,
`temperature_moves` 20 → 7 and `resignation = true`, validated first by
continuing from a trained 6×96 network and then from scratch (see the
main README's Presets section,
[The recipe](#the-recipe-four-changes-together-250-generations) and
[The 9×9 preset from scratch](#the-99-preset-from-scratch-250-generations)).
Every other 9×9 run in this directory
started from the earlier values (100 games, 200 steps,
`temperature_moves = 20`, no resignation) and changed only the settings
it tested (the recipe arm, `9x9-recipe/`, changed all four; the
from-scratch run, `9x9-scratch/`, is the preset itself); each run's `config.cfg` records
what it used, and `--resume` keeps it.

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

| run | vs gen 0 | `log.csv` Elo, last 10 − first 10 | `elo.csv` Elo, last 10 − first 10 | re-rated `rerate/elo.csv`, last 10 − first 10 |
|---|---|---|---|---|
| seed 1 | 100/100 | 1137.6 − 668.1 = +469.5 | 1089.1 − 349.7 = +739.4 | 1709.9 − 716.2 = +993.7 |
| seed 2 | 100/100 | 910.5 − 549.2 = +361.3 | 875.1 − 311.9 = +563.3 | 1511.4 − 551.2 = +960.2 |

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

The table's last column comes from re-rating both runs with
rating-aware opponents (`5x5/rerate/`, `5x5-seed2/rerate/`; the method
and its caveats are in "9×9 re-rated with rating-aware opponents"
below). It used
`train --run runs/5x5-seed1 --rerate --set arena_opponents=2`, and the
same for `runs/5x5-seed2`.

- Seed 1: generation 60 is rated 1671.5 (peak 1728.0 at generation 57).
  The 100-simulation bot is rated 281.5; it scored 70/80 against
  generation 0. Generations 3, 9, 17, 20, 24, 29, 33 and 43 were
  promoted (`5x5/rerate/references.csv`, `results.csv`, `elo.csv`).
- Seed 2: generation 60 is rated 1579.3. The bot is rated −65.9; it
  scored 32/80 against generation 0. Generations 3, 6, 11, 18, 24, 31
  and 43 were promoted (`5x5-seed2/rerate/references.csv`, `results.csv`,
  `elo.csv`).
- No generation from 30 on played generation 0 in either run
  (`rerate/log.csv`).

The re-rated curves keep the same shape, a rise and then a plateau from
about generation 40. Their decade means are 716.2, 1099.4, 1423.3,
1626.4, 1697.9, 1709.9 (seed 1) and 551.2, 970.4, 1225.2, 1411.0,
1499.7, 1511.4 (seed 2).

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
   - The 137 pairs scoring 18–2 or better (generations 14–150; 109 of them 20–0)
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

*Later note.* These probes do not compare like with like: after Black's
B1 White has the next move, and in the C6 B1 line Black has played an
extra move. A later comparison that gives each side one move found four
9×9 networks all preferring C6 to saving the group, and the ownership
maps of one of them show that it reads the group's fate correctly. The
"misjudgment" may be a whole-board choice rather than a misread shape;
see [An ownership head](#an-ownership-head-auxiliary-target). The
paragraph above is kept as it was written.

`9x9/sample.sgf` is a generation-150 self-play game, won W+6.5.

### 9×9 re-rated with rating-aware opponents

The code now chooses arena opponents by rating instead of by ladder
distance: each generation plays g − 1, the strongest reference and
`arena_opponents` more references, those against which its expected
score is closest to 0.7. The references are a fixed uniform-prior bot,
generation 0, and every generation that scored at least
`promote_threshold` against the strongest reference (`docs/algorithm-explained.md`, chapter 07). With
gating off, the arena never feeds back into training, so the run's
checkpoints can be re-measured without retraining:

```bash
./build-blas/train --run runs/9x9-main --rerate --set arena_opponents=2
```

This replays the arena over generations 1–150 into
`runs/9x9-main/rerate/`, copied to `9x9/rerate/`. It took 2 h 06 min
(`runs/rerate-runs.log`, not in the repository).

- **Checks.** Every generation played g − 1 first, and generation 1's
  opponents are `0 -1`, as in a fresh run (`9x9/rerate/log.csv`).
- **The bot** (100 simulations) scored 75/80 against generation 0 in
  calibration and is rated 328.9 (`9x9/rerate/results.csv`,
  `references.csv`).
- **Promoted generations:** 23, 30, 43, 53, 78, 95 and 121, seven in all
  (`9x9/rerate/references.csv`).
- **Generation 0:** one generation from 30 on played it, generation 30
  itself, when generation 0 and the bot were the only references left
  to fill the two extra slots (`9x9/rerate/log.csv`).

The re-rated gaps against the 100-game head-to-heads (`9x9/h2h-*.txt`):

| match | h2h | ladder `elo.csv` | re-rated |
|---|---|---|---|
| gen 150 vs gen 50 | +676 | −106 | +634 |
| gen 150 vs gen 90 | +315 | +17 | +249 |
| gen 150 vs gen 100 | +200 | +45 | +234 |
| gen 150 vs gen 120 | +123 | +36 | +133 |
| gen 90 vs gen 50 | +363 | −122 | +385 |

Every re-rated gap has the head-to-head's sign and lies within 66 Elo of
it. The re-rated ratings (`9x9/rerate/elo.csv`) are 1132.5 at generation
50, 1517.4 at 90, 1532.1 at 100, 1633.6 at 120 and 1766.6 at 150.

A least-squares fit of `9x9/rerate/elo.csv` over generations 101–150
gives **+2.83 Elo per generation**, so the Elo-slope part of the
criterion is met on the re-rated curve. On the ladder `elo.csv` it was
+0.41 (met on its sign only). +2.83 is about +140 Elo over 50
generations; the refit without generation-0 pairs gave +2.78, and the
head-to-heads about 4 per generation after generation 90.

**The 9×9 criterion as a whole is still not met.** The eye check is a
property of the generation-150 network, and re-rating does not change
the network.

Three caveats about reading these files:

- **The scale depends on `arena_opponents`.** A re-rate of 5×5 seed 1
  with four extra references (`--set arena_opponents=4`, on a copy of the
  run) compressed the scale: generation 60 at 1306.9
  (`5x5/rerate-opp4/elo.csv`), against 1671.5 with two
  (`5x5/rerate/elo.csv`), with the same promoted generations. The ratings
  and the gaps between them hold for `arena_opponents = 2`; numbers from
  different settings do not compare.
- **Promotion can admit a generation that lost to g − 1.** It only has
  to reach the threshold against the strongest reference. It happened in
  both 5×5 runs. In seed 1, generation 24 scored 3/20 against generation
  23 and 18/20 against generation 20, and was promoted; generation 33
  scored 4/20 against generation 32 and was promoted too
  (`5x5/rerate/log.csv`). In seed 2, generation 11 was promoted with
  9/20 against generation 10 (`5x5-seed2/rerate/log.csv`). No promoted
  9×9 generation scored below 0.50 against g − 1.
- **`winrate_vs_prev` can count more games than `score_vs_prev`.** When
  g is promoted against g − 1 (g − 1 is the strongest reference), the
  80 calibration games are added to that pair, so `elo.csv`'s
  `winrate_vs_prev` covers 100 games while `log.csv`'s `score_vs_prev`
  covers the 20 arena games. In these three runs the two columns agree
  for every generation (`rerate/elo.csv`, `rerate/log.csv`).

### 9×9 continued to 250 and grown to 10 blocks

The 9×9 run was continued from generation 150 to 250 with the same
6-block network, re-rated with `arena_opponents = 2`, and generation 250
was then grown into a 10-block network and trained for 50 more
generations. Nothing here changes the verdict on the criterion: the
network still fails the eye check.

**The 6-block continuation, 150 to 250.** The re-rate replays the arena
over generations 1–250 (`9x9/rerate-250/`: `elo.csv`, `references.csv`,
`log.csv`, `results.csv`, `progress.txt`). It is a different file from
`9x9/rerate/`, which covers 1–150 only.

- Re-rated Elo: 1532.3 at generation 100, 1728.3 at 150, 1758.1 at 200,
  a peak of 1870.0 at generation 223 (1865.3 at 225), and 1799.5 at 250
  (`9x9/rerate-250/elo.csv`).
- Least-squares slope: +2.47 Elo per generation over generations
  101–150 and +1.10 over 151–250 (computed from `elo.csv`). Progress
  slows but does not stop.
- Promoted references: generations 170 and 228 (`references.csv`, rated
  1723.3 and 1791.5). Generation 250 scored 0.60 against 228 in its
  20-game re-rate match (`log.csv`).
- The eye check on the 6-block generation 250 also fails: the group's
  one two-eye move B1 is not among the ten most visited moves.

**The grown run** (`9x9-grow10/`).

- *Setup.* `grow` copied generation 250 into a 10-block, 64-filter
  network (473,277 → 769,213 parameters) with the four new blocks
  initialised so the function is unchanged; on 8 random positions the
  maximum output difference was 0. Training started from it, with the
  6-block run's replay buffer (199,966 positions from 2,262 games,
  `init.txt`):

  ```bash
  train --init-from runs/grown/gen250-10blocks.azn --buffer-from runs/9x9-main \
      --set blocks=10 --set lr_schedule=0:0.002 --generations 50
  ```

  Everything else is the 9×9 preset of the time (100 games of 200
  simulations per generation, `arena_opponents = 2`, gating off). `init.txt` is the
  run's file with the absolute paths made repo-relative. Generation 0 of
  this run is exactly the old generation 250, since growing preserves
  the function.
- *Time.* Mean 322.2 s per generation: 176.0 s self-play, 97.1 s
  training and 49.1 s arena (`9x9-grow10/log.csv`); 4.47 h for the 50
  generations.
- *No promotions.* No generation scored 0.85 against generation 0, so
  `references.csv` holds only the bot and generation 0. `elo.csv` is
  anchored at this run's own generation 0 (peak 141.5, 28.4 at
  generation 50), a noisy 20-game ladder; it is not comparable to
  `9x9/`'s or `9x9/rerate-250/`'s scale.
- *Losses stay flat.* Policy loss 2.406 at generation 1 and 2.415 at 50
  (range 2.395–2.415); value loss 0.381 and 0.364 (range 0.354–0.393);
  average game length 88.8 and 87.8 plies (`log.csv`).

**Three 100-game matches** (200 simulations, colors alternating; 95%
Wilson intervals as `arena` prints them; Elo is 400·log10(p/(1−p)) and
approximate; files in `9x9-grow10/`):

| match | score | 95% interval | Elo (interval) |
|---|---|---|---|
| gen 50 vs old gen 250 (`g50-vs-old250.txt`) | 59/100 | 0.492–0.681 | +63 (−6 to +132) |
| gen 25 vs old gen 250 (`g25-vs-old250.txt`) | 61/100 | 0.512–0.700 | +78 (+8 to +147) |
| gen 50 vs old gen 228 (`g50-vs-old228.txt`) | 70/100 | 0.604–0.781 | +147 (+73 to +221) |

The arena needs equal shapes, so generation 228 was compared through a
grown copy of it, `runs/grown/gen228-10blocks.azn` (not in the
repository).

**Verdict.** The 10-block network is modestly stronger than the 6-block
generation 250: roughly +60 to +80 Elo, with the interval's lower end at
or near 0.5 (0.492 for generation 50, 0.512 for generation 25). That is
about what the 6-block run gained over its last 50–100 generations
(1728.3 at 150 to 1799.5 at 250 is +71; the 1.10 Elo per generation
slope gives +55 over 50). The generation 50 eye check
(`analyze-eye.txt`) still fails: B1 is not in the top ten priors (the
top is C7 at 23.5%) or the top ten visited moves, and the network gives
the position −0.54 (value) and −0.72 (after 200 simulations). So
capacity alone is not the bottleneck at this compute budget.

**Next levers, all untested hypotheses when this was written.** The first
three have since been tested; each says where.

- More simulations per move, for self-play and training targets.
  Tested (400 vs 200, 50 generations): no measurable gain per
  generation and worse per unit of compute; see
  [the section](#more-simulations-in-self-play-400-vs-200).
- Learning-rate changes (the grown run held 0.002 constant). Tested (a
  restart at 0.02 stepped back to 0.002, 50 generations): behind the
  control at generation 25, level at 50; see
  [the section](#a-learning-rate-restart-002-to-0002).
- More self-play games per generation (100 at the time, with a replay
  ratio of about 5.8). Tested (200 vs 100, 50 generations): at best a small
  edge, not clearly real, at 1.6× the time; see
  [the section](#more-self-play-games-per-generation-200-vs-100). That run
  kept 200 training steps, halving the replay ratio to 2.9, so it tested
  fresher data, not more learning. 200 games with 400 steps, which keeps
  the ratio, beat the control at generations 25 and 50 for twice the
  compute per generation; see [the section](#the-learning-budget-200-games-and-400-steps-per-generation).
- A targeted analysis of why the eye shape is rare in self-play: if the
  network never reaches such positions, no amount of capacity adds the
  training signal.

### Grown to 6×96 (wider) and width vs depth

The same generation 250 was also widened instead of deepened, to ask
whether width or depth is the better way to add capacity. Nothing here
changes the verdict: the network still fails the eye check, and the data
does not separate width from depth.

- *Setup.* `grow --filters 96` copied generation 250 into a 6-block,
  96-filter network (473,277 → 1,032,349 parameters; the 10-block,
  64-filter network has 769,213). The maximum output difference on the
  check positions was 7.2e-06 (not exactly 0, unlike the depth growth).
  Training started from it with the same replay buffer (199,966
  positions from 2,262 games), for 50 generations (`9x9-wide96/`):

  ```bash
  train --config configs/9x9.cfg --run runs/9x9-wide96 \
      --init-from runs/grown/gen250-6x96.azn --buffer-from runs/9x9-main \
      --set filters=96 --set lr_schedule=0:0.002 --generations 50
  ```

  Everything else is as in the 10-block run (100 games of 200
  simulations, `arena_opponents = 2`, gating off). The files are the
  run's own with absolute paths in `init.txt` made repo-relative;
  `progress.txt` is the `gen`, `calibrat` and `done` lines of the
  run's console output.
- *Time.* Mean 383.0 s per generation (self-play + training + arena):
  201.9 s, 95.0 s and 86.1 s (`9x9-wide96/log.csv`), 5.32 h for the 50
  generations, against 322.2 s for the 10-block run. Part of the arena
  figure is calibration games after promotions.
- *Promotions.* Generations 1 and 14 were promoted (`references.csv`).
  Each scored at least 0.85 against generation 0 in a 20-game test
  (19/20 at generation 1, 17/20 at 14, `progress.txt`). A 20-game test
  at that threshold can pass by chance for a network of equal strength,
  and these promotions are not jumps: in the final `elo.csv` they sit at
  -1.1 and 8.6, while the ladder as a whole is noisy (-41.8 to 130.4,
  118.0 at generation 50, a 20-game ladder anchored at this run's
  generation 0; not comparable to other runs' scales). The 100-game
  matches below are the evidence for strength.
- *Losses stay flat.* Policy loss 2.406 at generation 1 and 2.404 at 50
  (range 2.388-2.415); value loss 0.382 and 0.361 (range 0.358-0.393);
  average game length 88.8 and 88.7 plies (`log.csv`).

**Matches** (100 games, 200 simulations, colors alternating; 95% Wilson
intervals as `arena` prints them; Elo is 400·log10(p/(1−p)) and
approximate; files in `9x9-wide96/`; generation 0 of this run is the old
generation 250):

| match | score | 95% interval | Elo (interval) |
|---|---|---|---|
| gen 50 vs old gen 250 (`w50-vs-old250.txt`) | 70/100 | 0.604-0.781 | +147 (+73 to +221) |
| gen 25 vs old gen 250 (`w25-vs-old250.txt`) | 60/100 | 0.502-0.691 | +70 (+1 to +140) |
| wide gen 50 vs deep gen 50, both as 10×96 (`wide50-vs-deep50.txt`) | 52/100 | 0.423-0.615 | +14 (−54 to +81) |

The arena needs equal shapes, so for the head-to-head both generation 50
networks were converted to 10 blocks of 96 filters: the wide one by
deepening (maximum difference 0) and the 10×64 one by widening (8.1e-06;
`runs/grown/*.azn`, not in the repository).

**Width vs depth**, each against the old generation 250:

| | 10×64 (depth) | 6×96 (width) |
|---|---|---|
| parameters | 769,213 | 1,032,349 |
| seconds per generation | 322.2 | 383.0 |
| gen 25 | 61/100 (+78) | 60/100 (+70) |
| gen 50 | 59/100 (+63) | 70/100 (+147) |
| gen 50 eye check | fails | fails |

**Verdict.** Width gave +147 Elo at generation 50 (interval about +73
to +221) against depth's +63 (−6 to +132), and the intervals overlap.
The direct head-to-head, the cleaner test, is 52/100 (0.423-0.615): it
cannot tell the two apart. The data therefore does not separate width
from depth; at generation 25 they are indistinguishable too (60 and 61).
The wide network costs about 19% more time per generation and has 34%
more parameters. The generation 50 eye check (`analyze-eye.txt`) still
fails: B1 is not among the top ten moves by prior (the top is C6 at
14.8%) or by visits (the top is C6, 35 visits), and the value is −0.72
(−0.73 after 200 simulations). Neither shape fixes the eye, and capacity
is not the main bottleneck at this budget. The "next levers" above
remain untested.

### More simulations in self-play (400 vs 200)

Hypothesis, from the "next levers" above: more simulations per move give
better search targets for the policy and value heads, so a run with
400 simulations in self-play should learn more per generation than one
with 200. The result: at equal generations, doubling simulations gave no
measurable gain at generation 50 and a weak one at generation 25, and the
eye check still fails.

- *Setup.* Two arms from the same start, the 6×96 generation 50 of
  `9x9-wide96` and that run's replay buffer (199,972 positions from
  2,272 games), 50 generations each. The configuration is identical
  except `simulations`: 400 (`9x9-sims/sims400/`) and 200
  (`9x9-sims/sims200/`). Both arms keep `arena_opponents = 2` and gating
  off, as in the earlier runs. The commands are in
  `9x9-sims/sims-experiment.sh`; the 200 arm was stopped at generation 9
  when the session ended and finished with `train --resume`
  (`sims-experiment-resume.sh`). Resume is byte-identical by design, so
  the 200 arm is the same run it would have been without the stop. Files
  are each run's own, with absolute paths in `init.txt` made
  repo-relative and `progress.txt` the `gen`, `calibrat` and `done`
  lines of the console output (the 200 arm's contains the resume).
- *Cost.* Mean seconds per generation (self-play + training + arena,
  `log.csv`): 581.5 for 400 simulations (426.1 + 97.7 + 57.6) and 351.3
  for 200 (201.4 + 94.4 + 55.4), a factor of 1.66. Self-play alone costs
  2.12 times as much; training and arena time are unchanged. The 50
  generations took 8.08 h against 4.88 h of generation time.
- *Training curves.* Mean policy loss 2.367 (400) and 2.390 (200), value
  loss 0.375 and 0.362, average game length 90.9 and 88.3 plies. These
  are on different data and are not a strength measure.

**Matches** (100 games, 200 simulations for both sides, komi 7.5, colors
alternating; 95% Wilson intervals as `arena` prints them; Elo is
400·log10(p/(1−p)), approximate; files in `9x9-sims/`; "start" is
`9x9-wide96` generation 50):

| match | score | 95% interval | Elo (interval) |
|---|---|---|---|
| 400 vs 200, gen 50 (`s400g50-vs-s200g50.txt`) | 52/100 | 0.423-0.615 | +14 (−54 to +81) |
| 400 vs 200, gen 25 (`s400g25-vs-s200g25.txt`) | 61/100 | 0.512-0.700 | +78 (+8 to +147) |
| 400 gen 50 vs start (`s400g50-vs-start.txt`) | 61/100 | 0.512-0.700 | +78 (+8 to +147) |
| 200 gen 50 vs start (`s200g50-vs-start.txt`) | 64/100 | 0.542-0.727 | +100 (+29 to +170) |

A later 400-game match of the 200 arm's generation 50 against the start
scored 232/400 = 0.580 (0.531-0.627, +56 Elo;
`s200g50-vs-start-400.txt`), which replaces the 64/100 as the control's
gain over the start; see
[AlphaGo Zero training arms](#alphago-zero-training-arms-setup-and-the-control-at-400-games).

**Eye check** (the 14-move position of the earlier eye checks, generation
50 of each arm, `eye-*.txt`). Still failing in all four analyses: B1 is
not among the top ten moves by visits at 200 or 800 simulations, for
either arm. The value for Black to move is −0.68 (400 arm) and −0.65
(200 arm) before search:

| arm | simulations | MCTS value | top visits |
|---|---|---|---|
| 400 | 200 | −0.65 | D6 46 (23.0%), C6 38 (19.0%) |
| 400 | 800 | −0.64 | D6 354 (44.2%), C6 132 (16.5%) |
| 200 | 200 | −0.58 | C6 42 (21.0%), D6 40 (20.0%) |
| 200 | 800 | −0.57 | C6 412 (51.5%), D6 85 (10.6%) |

**Verdict.** At equal generations, doubling simulations gave no
measurable gain at generation 50 (52/100, interval 0.423-0.615) and a
weak edge at generation 25 (61/100, lower bound 0.512, one match). Both
arms improved over the start by similar amounts, +78 and +100 Elo, with
overlapping intervals; if anything the 200-simulation arm is nominally
ahead. Per unit of compute the 200-simulation arm is better: it reached
about the same strength in 60% of the time per generation. The eye shape
is untouched: B1 never appears among the top visits, and the value stays
near −0.6 at every search budget tried (200 and 800 simulations). This
is consistent with a value-head blind spot about life and death that
deeper search at analysis time does not fix, though these runs do not
test that explanation directly. (The ownership experiment below points
another way: a fair comparison suggests the networks choose C6 over B1
as the bigger move, not that they misread the group.)

Not tested here, and no result is claimed for any of them: playout cap
randomization (full search on only some moves), larger simulation gaps
than 2×, runs longer than 50 generations, changes to the value targets,
and more than one seed per arm (each number above is one run).

### An ownership head (auxiliary target)

Hypothesis, from the design
(`docs/superpowers/specs/2026-10-03-ownership-head-design.md`): every
9×9 network has failed the eye check, and the weakness looked like the
value head's life-and-death judgment. That head learns from one ±1 game
result per position, a signal that is noisy (it also depends on every
later move of both sides) and slow (under Tromp-Taylor scoring a dead
group costs points only once it has been captured). An ownership head
trained on the final board gets n² = 81 labels per position, each
saying directly whose a point ended up, so the shared trunk should learn
life and death faster. The result: no measurable change in strength
against the identical run without the head, and the eye check still
fails by its original criterion. But the ownership maps show that the
network reads the bottom group's life and death correctly, and a fairer
version of the eye comparison suggests that the check does not test
what it was meant to (see "A fairer eye comparison" below).

- *Setup.* `grow --ownership` gave a fresh ownership head (98
  parameters, 1,032,349 → 1,032,447) to the 6×96 generation 50 of
  `9x9-wide96`, the start of both arms of the simulations experiment;
  on 8 random positions the maximum output difference was 0
  (`9x9-ownership/ownership.log`). Training started from it with
  `9x9-wide96`'s replay buffer, 199,972 positions from 2,272 games
  (`init.txt`). That buffer was saved as `state.bin` version 2, before
  ownership existed, so each game's final ownership was recomputed by
  replaying its moves. The settings are those of `9x9-sims/sims200/`
  (100 games of 200 simulations per generation, learning rate 0.002,
  seed 1, `arena_opponents = 2`, gating off) plus `ownership = true`
  and `ownership_weight = 1.5`; the two runs' `config.cfg` files differ
  in those two lines only. Because the head changes neither policy nor
  value, generation 1 had to replay the 200 arm's self-play game for
  game, and it did: the same bot calibration (`results.csv` starts with
  `-1,0,80,0` in both runs), the same self-play columns in `log.csv`
  (100 games, 89.24 plies, pass rate 0.1075, Black win rate 0.2400), and
  byte-identical copies of the two game records each run saves per
  generation (in `runs/`, not in the repository).
  The commands are in `9x9-ownership/ownership.sh`. The files are the
  run's own, with absolute paths in `init.txt` made repo-relative and
  `progress.txt` the `gen` and `done` lines of the console output. This
  is one run with one seed.
- *Cost.* Mean seconds per generation (self-play + training + arena,
  `log.csv`): 380.2 (206.5 + 95.3 + 78.4), against the 200 arm's 351.3
  (201.4 + 94.4 + 55.4); 5.28 h of generation time against 4.88 h.
  Self-play and training took 2.5% and 1.0% longer. Most of the
  difference is arena time, and it is not a cost of the head:
  generation 6 scored 0.85 against generation 0 and was promoted to a
  reference (`references.csv`, `progress.txt`), which cost calibration
  games in that generation (140.9 s of arena) and added an opponent to
  every later generation's arena (80.4 s per generation over generations
  7–50, against the 200 arm's 56.3; the 200 arm promoted none).
- *Training curves.* The ownership loss is 0.590 at generation 1, 0.401
  at 25 and 0.408 at 50 (`log.csv`). It falls fastest at the start
  (0.471 at generation 2, 0.423 at 5) and stays between 0.401 and 0.409
  from generation 10 on. Mean policy loss 2.398 and value loss 0.363,
  against the 200 arm's 2.390 and 0.362; average game length 87.6
  plies against 88.3. These are on different data and are not a
  strength measure.

**Matches** (100 games, 200 simulations for both sides, komi 7.5, colors
alternating; 95% Wilson intervals as `arena` prints them; Elo is
400·log10(p/(1−p)), approximate; files in `9x9-ownership/`; "start" is
this run's generation 0, the 6×96 generation 50 of `9x9-wide96` with the
untrained head). The 200 arm's checkpoints were given a head with
`grow --ownership` so that `arena` accepts the pair; it changes no move
(maximum difference 0 for both, `ownership.log`).

| match | score | 95% interval | Elo (interval) |
|---|---|---|---|
| ownership vs 200 arm, gen 25 (`own25-vs-s200g25.txt`) | 48/100 | 0.385-0.577 | −14 (−81 to +54) |
| ownership vs 200 arm, gen 50 (`own50-vs-s200g50.txt`) | 55/100 | 0.452-0.644 | +35 (−33 to +103) |
| ownership gen 25 vs start (`own25-vs-start.txt`) | 53/100 | 0.433-0.625 | +21 (−47 to +89) |
| ownership gen 50 vs start (`own50-vs-start.txt`) | 59/100 | 0.492-0.681 | +63 (−6 to +132) |

For comparison, the 200 arm's generation 50 scored 64/100 (0.542-0.727)
against the same start (`9x9-sims/s200g50-vs-start.txt`).

**Eye check** (the 14-move position of the earlier eye checks, Black to
move, generation 50 of this run, `eye-own50-s{200,800}.txt`). The value
for Black before search is −0.61, and B1's prior is below 1%:

| simulations | MCTS value | top visits | B1's rank by visits |
|---|---|---|---|
| 200 | −0.66 | D7 40 (20.0%), C7 22 (11.0%), C6 22 (11.0%) | not in the top 10 |
| 800 | −0.67 | D7 258 (32.2%), C6 95 (11.9%), C7 76 (9.5%) | not in the top 10 |

By the original criterion (B1 in the top 3 by visits at 200
simulations) the check fails, as for every earlier 9×9 network. The 200
arm's generation 50 put C6 42 and D6 40 on top at 200 simulations, with
B1 absent too (`9x9-sims/eye-sims200-s200.txt`).

**Ownership maps.** `analyze` prints the head's prediction in Black's
terms, +9 surely Black's to −9 surely White's, `.` below 0.1 in absolute
value. The eye position (`eye-own50-s200.txt`):

```
     A  B  C  D  E  F  G  H  J
 9  +2 +3 +3 +3 +3 +4 +3 +4 +4
 8  +2 +2 +2 +2 +4 +4 +4 +3 +3
 7  +2 +2 +3 +3 +5 +8 +5 +4 +4
 6   . +2 +2 +1 +3 +5 +5 +4 +3
 5  -4 -2  .  . +2 +2 +7 +4 +3
 4  -7 -7 -7 -5 -3  . +2 +2 +2
 3  -9 -9 -9 -9 -9 -4  . +1  .
 2   . -2  . +1 -8 -6 -3  .  .
 1  -2 -3 -4 -5 -9 -7 -4 -2  .
```

The White wall is −9 (A3–E3, E1) and −8 (E2). The group is the Black
stones A2, B2, C2, D2 and D1 and the points it encloses, A1, B1 and C1:

| position | A2 | B2 | C2 | D2 | D1 | A1 | B1 | C1 |
|---|---|---|---|---|---|---|---|---|
| eye position, Black to move (`eye-own50-s200.txt`) | . | −2 | . | +1 | −5 | −2 | −3 | −4 |
| Black C6, White kills with B1 (`eye-own50-killed.txt`) | −2 | −5 | −6 | −4 | −6 | −6 | −8 | −8 |
| Black saves with B1 (`eye-own50-saved.txt`) | +3 | +4 | +4 | +4 | +4 | +4 | +2 | +4 |
| Black B1, then White C6 (`fair-own50-save.txt`) | +4 | +5 | +5 | +6 | −2 | +2 | +2 | −1 |

After White's B1 every point of the group is White's (−2 to −8): the
head knows the group is dead. After Black's B1 every point is Black's
(+2 to +4): it knows the group lives. In the eye position itself the
group leans White (D1 and A1–C1 at −2 to −5), which fits the network's
own preference for playing elsewhere (below). After Black's B1 and
White's C6 the head is less sure of the corner (D1 −2, C1 −1) although
the group has two eyes; its other six points are +2 to +6.

**A fairer eye comparison.** The earlier probes
(`9x9/analyze-eye-probe.txt`) and the `killed` and `saved` positions
above do not compare like with like. In `killed` Black has also played
C6 and has the next move, while in `saved` White has it: the killing
line gives Black two moves to White's one. A fair
comparison gives each side one move and leaves Black to move in both:
`$EYE C6 B1` (Black plays elsewhere, White kills) against `$EYE B1 C6`
(Black saves, White takes C6), where `$EYE` is the 14-move position.
MCTS value for Black after 800 simulations, komi 7.5, with the value
before search in brackets (files `fair-*-tenuki.txt` and
`fair-*-save.txt`, commands in `fair-comparison.sh`):

| network | Black C6, White B1 | Black B1, White C6 | difference |
|---|---|---|---|
| this run, gen 50 (`fair-own50-*`) | +0.04 (−0.26) | −0.87 (−0.78) | 0.91 |
| this run's start, gen 0 (`fair-own0-*`) | −0.33 (−0.48) | −0.85 (−0.85) | 0.52 |
| 200 arm, gen 50 (`fair-sims200g50-*`) | +0.04 (−0.21) | −0.84 (−0.80) | 0.88 |
| 6×64 main run, gen 250 (`fair-main250-*`) | −0.18 (−0.36) | −0.84 (−0.87) | 0.66 |

The main run's generation 250 (`runs/9x9-main`) and the 200 arm have no
ownership head, so for them there are values only, no map. Every
network tested values giving up the group for C6 well above saving it.
The map after Black's B1 and White's C6 (`fair-own50-save.txt`) shows
why, in this network's terms: White's C6 claims the left side, −4 to
−8 on A4–C6 and −2 to −3 on A7–C7. After Black's C6 instead, A6–C7 is
Black's (+3 to +8) and A5–C5 undecided (`fair-own50-tenuki.txt`).

What this shows, and what it does not:

- *Evidence.* For this run's generation 50, the values and maps are
  consistent with a whole-board judgment that another move is bigger
  than B1, rather than a failure to read the group: its maps get the
  group's fate right in the `killed` and `saved` lines. Two limits: in
  each fair line White's reply (B1 or C6) was fixed by hand, not chosen
  by search, and the network's own top move in the eye position is D7,
  not C6 (`eye-own50-s200.txt`, `eye-own50-s800.txt`). The other three
  networks prefer the C6 line by value too; without a map it is
  consistent with, not shown, that they read the group.
- *Not settled.* Whether the judgment is correct cannot be decided
  here. That needs an independent strong engine (for example KataGo)
  to evaluate both lines, and none was run.
- *Recommendation, not a result.* The eye check of the original plan is
  not a clean life-and-death test, because the rest of the board is
  unsettled and the vital point may not be the largest move. It should
  be replaced by a position in which the vital point is unambiguously
  the largest move, or its intended answer should first be confirmed
  with a strong engine.

**Verdict.** Against the 200 arm, the ownership run is **not measurably
different** at generation 25 (48/100, 0.385-0.577, −14 Elo) or at
generation 50 (55/100, 0.452-0.644, +35 Elo, interval −33 to +103):
ownership training gave no measurable strength change over 50
generations. Against the start it is not measurably different either
(53/100 at 25; 59/100 at 50, lower bound 0.492), where the 200 arm alone
reached 64/100; the intervals overlap, and the direct match is the
cleaner comparison. The eye check **fails** by its original criterion at
200 and 800 simulations. The maps **do** show that the head knows the
group dies after White's B1 and lives after Black's B1. The head trains
(loss 0.590 → 0.408) and is a useful diagnostic: it is what suggests
that the eye-check failure is a choice of move rather than a misread
group.

Not tested here, and no result is claimed for any of them: more than one
seed, other values of `ownership_weight` than 1.5, runs longer than 50
generations, using the head in search (score estimation or
ownership-weighted values) and KataGo's other auxiliary targets (score,
opponent policy), which the design lists as non-goals.

### All post-250 runs against one reference (gen 200)

Why: each post-250 run's `elo.csv` is anchored at its own generation 0, and
those starting points are different networks, so the runs' Elo values
could not be compared with each other. The 100-simulation bot
that earlier re-rates used as a common yardstick saturates (the networks
beat it nearly every game), so it could not serve as one either. The
presets no longer include it (`bot_simulations =` is empty by default),
and a run can name frozen networks as reference opponents instead
(`reference_networks`).

Method: each of the five runs was re-rated with
`train --rerate --set bot_simulations= --set reference_networks=runs/9x9-main/checkpoints/gen_0200.azn --set arena_opponents=2`,
that is, one shared frozen reference (6-block gen 200 of `9x9-main`,
converted exactly to each run's network shape, called `ref1`) and no bot.
Every checkpoint is rated against its two neighbours in rating plus
`ref1`, in 20-game, 100-simulation matches. Below, Elo is the run's Elo
minus `ref1`'s Elo from that run's `references.csv`, so all rows sit on
the same scale (0 = gen 200). Sources: `docs/results/9x9-ref200/<run>/`
(`elo.csv`, `references.csv`, `results.csv`, `log.csv`,
`reference-networks.csv`, `progress.txt`), `rerate-ref200.sh` and its
log. Paths in the copied script, log and CSVs were made repo-relative.

| run | start (gen 0) | gen 25 | gen 50 | calibration: `ref1` vs gen 0 |
|---|---|---|---|---|
| wide96 (6×96, from gen 250) | +61 | +128 | +131 | 0.36 of 80 games |
| grow10 (10 blocks, from gen 250) | +49 | +73 | +85 | 0.36 of 80 games |
| sims200 | +104 | +152 | +132 | 0.26 of 80 games |
| sims400 | +105 | +147 | +91 | 0.26 of 80 games |
| ownership | +85 | +105 | +140 | 0.26 of 80 games |
| games200 (200 games per generation; added later) | +101 | +49 | +88 | 0.26 of 80 games |
| lr-restart (learning rate 0.02 → 0.006 → 0.002; added later) | +105 | +182 | +126 | 0.26 of 80 games |
| resign (`resignation = true`; added later) | +144 | +146 | +201 | 0.26 of 80 games |
| temp7 (`temperature_moves = 7`; added later) | +105 | +151 | +175 | 0.26 of 80 games |
| sym (`mcts_random_symmetry = true`; added later) | +123 | +132 | +125 | 0.26 of 80 games |
| budget (200 games and 400 steps per generation; added later) | +94 | +164 | +141 | 0.26 of 80 games |
| steps400 (100 games and 400 steps per generation; added later) | +119 | +151 | +55 | 0.26 of 80 games |
| recipe (`temperature_moves = 7`, resignation, 200 games, 400 steps; 200 generations; added later) | +104 | +233 | +303 (gen 100: +356; gen 150: +400; gen 200: +469) | 0.26 of 80 games |

(`ref1` scored 29/80 against gen 0 of wide96 and grow10 and 21/80 against
the other runs, which share a start; the table is `elo.csv` minus
`references.csv` in each run's folder: `9x9-ref200/<run>/` for the first
five rows, `9x9-games200/rerate/`, `9x9-lr-restart/rerate/`,
`9x9-resign/rerate/`, `9x9-temp7/rerate/`, `9x9-sym/rerate/`,
`9x9-budget-g200-s400/rerate/`, `9x9-steps400/rerate/` and
`9x9-recipe/rerate/` for the rows added later,
which were re-rated the same way after their runs. The recipe's row is
the re-rate over all 200 generations; the one made at generation 100,
`9x9-recipe/rerate-to100/`, gave +102, +234, +303 and +360 at 0, 25,
50 and 100, which shows the re-rate's noise from one replay to the
next.)

The ranking of these runs that the later sections rely on is not this
table but a pooled evaluation with less noise: each run's generations
46–50 against the start, 1,000 games per run; see
[Final strength of the post-250 runs](#final-strength-of-the-post-250-runs-pooled-generations-46-to-50).

Reading it, honestly:

- **The noise is about ±40 Elo.** The same network (wide96 gen 50) is
  +131 as the last generation of the wide96 run but +85 to +144 as the
  start of the runs from it (+144 in the resign run, +104 in sims200). Differences between
  runs under about 50 Elo are not meaningful.
- **The scale is compressed.** Gen 250, the start of wide96 and grow10,
  shows only about +50 to +60 over gen 200 here, but it beat gen 200
  69/100 head to head (95% interval 0.594-0.772, about +139 Elo;
  `docs/results/9x9-ref200/g250-vs-g200.txt`, 200 simulations, komi 7.5,
  96 distinct games). These re-rates use 100-simulation, 20-game matches
  and the fit squeezes the gaps, so read the table for ranking, not for
  exact gaps.
- **Conclusion.** Every post-250 experiment gained a similar, modest
  amount in 50 generations, and none stands clear of the noise. That
  agrees with the direct 100-game matches reported in the sections above.
  Two of the AlphaGo Zero arms added later (resign, temp7) are the
  highest at generation 50, +201 and +175 against the control's +132,
  and their 400-game matches against the control exclude 0.5 (about +70
  Elo each; see their sections). The third (sym) is level with the
  control, +125, as its match at generation 50 was (0.495). The fourth
  (budget, twice the games and twice the steps) is +141 at generation
  50 but +187 over generations 26–50 and +93 over its own start there,
  the largest gain from a start in the table; its matches against the
  control exclude 0.5 at generations 25 and 50. Even so,
  every post-250 run's gain is modest next to this table's noise, and
  resign's lead is partly the noise in its start (+144), as the
  budget run's gain from its start is partly the noise in its (+94).
  The exceptions came last. The steps-only arm (steps400) is level with
  the control (+55 at generation 50, +146 over generations 26–50 against
  the control's +152). The recipe arm, which combines the three settings
  that beat the control, stands clear of the noise: +303 at generation
  50, against the control's +132 and at most +201 for any other run,
  +356 at generation 100 and +469 at 200.
- **All but one had the same learning budget.** Every other run in the
  table took 200 SGD steps of 256 positions per generation, 10,000 steps in
  50 generations, at a learning rate of 0.002 (the lr-restart run took
  the same number of steps, most of them at higher rates). That is
  consistent with every lever gaining about the same, though it does not
  show that the budget is the limit. The budget run doubled both games
  and steps (20,000 steps) and gained the most over its start, at twice
  the compute per generation; see
  [The learning budget](#the-learning-budget-200-games-and-400-steps-per-generation).

### Final strength of the post-250 runs, pooled (generations 46 to 50)

Why: the per-run verdicts below rest on single checkpoints, and those
did not always agree. The random-symmetry arm's generation 50 was level
with the control's (0.495) yet gained +104 over the start against the
control's +56; the steps-only arm's generation 50 beat the control's
(0.573) yet gained only +58 over the start. A network's strength moves
from one generation to the next by more than a 400-game interval
allows, about 50 Elo in those examples, and the re-rate against gen 200
has about ±40 of noise. Ranking the runs needed a measurement averaged over several
checkpoints and with more games.

Method: ten runs that share a start (the `sims200` control, `sims400`,
`games200`, `lr-restart`, the five AlphaGo Zero arms of 50 generations
and the recipe arm; the ownership run was not included) each played
generations 46, 47, 48, 49 and 50 against that start (generation 50 of
`9x9-wide96`), 200 games each: 1,000 games per run over five
checkpoints. 200 simulations for both sides, komi 7.5, colors
alternating, `arena`'s default opening temperature (4 plies), binary
`runs/bin-next`. `final-eval.sh` ran the other nine on the second Mac mini,
6 October 18:06 to 23:25 (`final-eval.log`); `final-eval-recipe.sh` ran
the recipe arm's generations 46–50 and 96–100 the same way on 7
October (`final-eval-recipe.log`, which does not record the machine).
Below, the score is pooled over the 1,000 games, with its 95% Wilson
interval, Elo is 400·log10(p/(1−p)) against the start, and "vs control"
is the difference between a run's Elo and the control's. Sources:
`docs/results/9x9-final-eval/` (the two scripts, their logs and one
`arena` output per checkpoint, `<run>-g<generation>.txt`).

| run | pooled score | 95% interval | Elo vs start (interval) | vs control | per checkpoint, gens 46, 47, 48, 49, 50 |
|---|---|---|---|---|---|
| recipe, gens 96–100 | 849/1000 = 0.849 | 0.825-0.870 | +300 (+270 to +330) | +232 (at twice the generations) | 0.840, 0.855, 0.845, 0.870, 0.835 (gens 96–100) |
| recipe | 798/1000 = 0.798 | 0.772-0.822 | +239 (+212 to +265) | **+170** | 0.835, 0.765, 0.805, 0.775, 0.810 |
| budget (200 games, 400 steps) | 656/1000 = 0.656 | 0.626-0.685 | +112 (+89 to +135) | **+44** | 0.665, 0.655, 0.675, 0.640, 0.645 |
| temp7 | 647/1000 = 0.647 | 0.617-0.676 | +105 (+83 to +128) | **+37** | 0.725, 0.635, 0.635, 0.595, 0.645 |
| resign | 627/1000 = 0.627 | 0.597-0.656 | +90 (+68 to +112) | +22 | 0.660, 0.595, 0.650, 0.630, 0.600 |
| sym | 627/1000 = 0.627 | 0.597-0.656 | +90 (+68 to +112) | +22 | 0.605, 0.580, 0.625, 0.655, 0.670 |
| games200 | 615/1000 = 0.615 | 0.584-0.645 | +81 (+59 to +103) | +13 | 0.650, 0.635, 0.685, 0.550, 0.555 |
| lr-restart | 615/1000 = 0.615 | 0.584-0.645 | +81 (+59 to +103) | +13 | 0.600, 0.610, 0.605, 0.675, 0.585 |
| steps400 | 604/1000 = 0.604 | 0.573-0.634 | +73 (+51 to +95) | +5 | 0.575, 0.615, 0.565, 0.670, 0.595 |
| control (`sims200`) | 597/1000 = 0.597 | 0.566-0.627 | +68 (+46 to +90) | 0 | 0.595, 0.645, 0.620, 0.510, 0.615 |
| sims400 | 572/1000 = 0.572 | 0.541-0.602 | +50 (+29 to +72) | −18 | 0.545, 0.570, 0.575, 0.620, 0.550 |

How precise it is:

- **About ±30 to ±35 Elo for a difference between two runs.** Each
  run's Elo has a 95% interval of about ±22; a difference of two runs,
  about ±31 from the binomial error alone. The five checkpoints of a run
  spread a little more than 200-game matches do by chance (a χ² of 55 on
  44 degrees of freedom over the eleven sets of five), so allowing for
  checkpoint variance widens that to about ±35. Differences in bold are
  outside it.
- **Not all new games.** `arena` seeds each game by its index
  (`--seed 1` by default), so the 200 generation-50 games of each run
  should repeat the first games of that run's generation-50 match
  against the start in the sections below (the first 200 of 400, or all
  100 of a 100-game match), as long as the search code is unchanged; the
  games of generations 46–49 are new. Distinct games were 867 (games200)
  to 983 (temp7) of 1,000.
- **One opponent compresses large gaps.** Every run is measured against
  the same, weaker start, and scores far from 0.5 resolve gaps poorly:
  the recipe's generations 96–100 are +61 above its generations 46–50
  by this table, but its generation 100 beat its generation 50 directly
  0.670 (+123). Read the recipe's margins as understated, not exact.
- **The control agrees with its earlier match.** Its +68 here against
  +56 in the 400-game generation-50 match against the start (below).

Reconciled with the per-run verdicts (each section's own verdict is
kept as written, with a later note):

- **The recipe** is far ahead, +170 over the control at equal
  generations, and still gaining at 96–100. See
  [The recipe](#the-recipe-four-changes-together-250-generations).
- **Budget (+44) and temp7 (+37) are confirmed**: both outside the
  ±35, though smaller than their generation-50 matches against the
  control suggested (+65 and +74).
- **Resign and sym (+22 each) are borderline**, inside the noise.
  Resignation's +67 at generation 50 was at the high end of its
  checkpoints; random symmetry's 0.495 at generation 50 was at the low
  end of its. Neither is shown to help or hurt.
- **Steps400 (+5), games200 (+13), lr-restart (+13) and sims400 (−18)
  gained nothing measurable**, as their sections concluded (steps400's
  0.573 against the control at generation 50 notwithstanding).

Read together: the control's settings with any one change gained at
most about 45 Elo at generation 50; more steps alone, more games alone,
a learning-rate restart and more simulations gained nothing; and the
three changes that did gain, combined, gained about four times as much
as the best of them. One seed per run, one start, 50 generations (100
for the recipe).

### More self-play games per generation (200 vs 100)

Hypothesis: every post-250 lever (depth, width, simulations, an ownership
head) gained about the same, which would fit a run limited by the amount
of data rather than by the model or the search. If so, doubling the games
per generation should help where those did not. The result: a small edge
at best, not clearly real, for 1.6 times the time.

**Correction: this did not test the amount of learning.** The run kept
`train_steps = 200`, so it took the same 10,000 SGD steps as the control
while generating twice the positions, and the replay ratio (`train_steps
× batch_size / (games × average plies)`, the `replay_ratio` column of
`log.csv`) fell by half: a mean of 2.90 against the control's 5.80
(200 × 256 / (100 × 88.3) = 5.80). Each position was drawn about half as
often. What the run tested is fresher data at half the replay ratio, not
more data with more learning, so it does **not** show that data quantity
is not the bottleneck; an earlier version of this section said it did.
The test that keeps the replay ratio constant, 200 games and 400
training steps per generation (`9x9-budget-g200-s400`), was run
afterwards and beat the control at generations 25 and 50; its
configuration differs from this run's only in `train_steps`. See
[The learning budget](#the-learning-budget-200-games-and-400-steps-per-generation).

- *Setup.* The control is `9x9-sims/sims200/` (generation 50 of
  `9x9-wide96` as the start, its buffer, same seed, `simulations = 200`,
  100 games per generation, 50 generations). The new run,
  `9x9-games200/`, has the same start, buffer, seed and settings, and
  the configurations were checked identical except
  `games_per_generation = 200`. Sources: `docs/results/9x9-games200/`
  (`log.csv`, `elo.csv`, `results.csv`, `references.csv`, `init.txt`,
  `progress.txt`, the three eval files, `games200.sh`, `games200.log`,
  and `rerate/` with the re-rate CSVs and progress lines). Absolute paths
  in `init.txt` and the script are made repo-relative. The matches below
  were run after training, in the script.
- *Cost.* Mean seconds per generation (self-play + training + arena,
  `log.csv`): 559.9 (401.3 + 95.4 + 63.2) against 351.3 for the control,
  a factor of 1.59. Self-play time is 1.99 times the control's (401.3 against
  201.4); training time is about the same (95.4 against 94.4) and arena
  time rose from 55.4 to 63.2 s. The 50 generations
  took 7.78 h of generation time.
- *Training curves.* Mean policy loss 2.388, value loss 0.361, average
  game length 88.3 plies (control 2.390, 0.362, 88.3). These are on
  different data and are not a strength measure.

**Matches** (100 games, 200 simulations for both sides, komi 7.5, colors
alternating; 95% Wilson intervals as `arena` prints them; Elo is
400·log10(p/(1−p)), approximate; "start" is `9x9-wide96` generation 50;
files in `9x9-games200/`):

| match | score | 95% interval | Elo (interval) |
|---|---|---|---|
| 200 games vs 100, gen 25 (`g25-vs-s200g25.txt`) | 61/100 | 0.512-0.700 | +78 (+8 to +147) |
| 200 games vs 100, gen 50 (`g50-vs-s200g50.txt`) | 56/100 | 0.462-0.653 | +42 (−26 to +110) |
| 200 games gen 50 vs start (`g50-vs-start.txt`) | 54/100 | 0.443-0.634 | +28 (−40 to +95) |

**Against gen 200.** Re-rated like the other runs (`rerate/`), Elo minus
`ref1`'s Elo (−100.8 in `rerate/references.csv`) is +101 at the start,
+49 at generation 25 and +88 at generation 50, on the scale of the
all-runs table above (the sims200 control: +104, +152, +132). The
same-scale noise there is about ±40 Elo, and this run's generation 25 and
50 sit below the control's.

**Verdict.** At equal generations the run shows at best a small edge over
the control: +78 at generation 25 (lower bound 0.512, one match) and +42
at generation 50 (interval includes 0.5). Against its own start it is
+28 at generation 50, with an interval that includes 0.5. The re-rate
does not show an edge either. Together that is not clearly a real
difference, and it cost about 1.6 times as much time per generation.
What this shows is narrow: with the number of SGD steps fixed, twice the
self-play games at half the replay ratio neither clearly helped nor
clearly hurt. It says nothing about whether more data with
proportionally more training would help (see the correction above).
One seed per arm; no claim beyond that. The next test was the learning
rate; see [the section below](#a-learning-rate-restart-002-to-0002).

### A learning-rate restart (0.02 to 0.002)

Hypothesis, from the "next levers" above: every post-250 run trained at
a flat learning rate of 0.002, the 9×9 preset's late rate. A restart at a
higher rate, stepped back down, might move the network further from where
it started and end stronger than the flat-rate control. The result: the
restart fell behind while the rate was high, caught up once it was back
at 0.002, and at generation 50 is level with the control (49/100). No
measurable gain.

- *Setup.* The control is `9x9-sims/sims200/`, as for the games run. The
  new run, `9x9-lr-restart/`, has the same start (generation 50 of
  `9x9-wide96`, 6×96), buffer, seed and settings: 200 simulations, 100
  games and 200 training steps of 256 positions per generation, 50
  generations. Its `config.cfg` differs from the control's only in
  `lr_schedule = 0:0.02,20:0.006,35:0.002`, apart from three keys added
  to the code since the control ran, at their defaults (`ownership`,
  `ownership_weight`, `reference_networks`). The learning rate was 0.02
  for generations 1–19, 0.006 for 20–34 and 0.002 for 35–50 (`log.csv`).
  0.02 is the presets' from-scratch rate. The run took the same 10,000
  SGD steps as the control, with a mean replay ratio of 5.81 against
  5.80. Sources: `docs/results/9x9-lr-restart/` (`log.csv`, `elo.csv`,
  `results.csv`, `references.csv`, `init.txt`, `progress.txt`, the three
  eval files, `lr-restart.sh`, `lr-restart.log`, and `rerate/` with the
  re-rate CSVs and progress lines). Absolute paths in `init.txt` and the
  script are made repo-relative. The matches below were run after
  training, in the script.
- *Cost.* Mean seconds per generation (self-play + training + arena,
  `log.csv`): 358.0 (205.5 + 95.5 + 57.0) against 351.3 for the control;
  4.97 h of generation time. The learning rate does not change the cost.
- *Training curves* (`log.csv`, mean per phase; these are losses on the
  training batches and not a strength measure):

  | generations (rate) | policy loss | control | value loss | control |
  |---|---|---|---|---|
  | 1–19 (0.02) | 2.444 | 2.401 | 0.352 | 0.367 |
  | 20–34 (0.006) | 2.415 | 2.388 | 0.302 | 0.362 |
  | 35–50 (0.002) | 2.385 | 2.379 | 0.330 | 0.356 |

  At 0.02 the policy loss rose at once (2.460 at generation 1, against
  the control's 2.411 on the same self-play games) and stayed about 0.04
  above the control; back at 0.002 it ended within 0.01 of it (2.369 at
  generation 50, control 2.376). The value loss went the other way: below
  the control at the higher rates, lowest at 0.006 (0.293), and back up
  to 0.345 at generation 50 (control 0.359). The batches are drawn from a
  buffer in which each position is sampled about 5.8 times, so a lower
  value loss there may mean a closer fit to the buffer rather than better
  judgment; without a loss on held-out games (see
  [Next experiments](#next-experiments)) this cannot be told apart.
  The run predates `validation_loss`; a rerun would log one.

**Matches** (100 games, 200 simulations for both sides, komi 7.5, colors
alternating; 95% Wilson intervals as `arena` prints them; Elo is
400·log10(p/(1−p)), approximate; "start" is `9x9-wide96` generation 50;
files in `9x9-lr-restart/`):

| match | score | 95% interval | Elo (interval) |
|---|---|---|---|
| restart vs control, gen 25 (`g25-vs-s200g25.txt`) | 36/100 | 0.273-0.458 | −100 (−170 to −29) |
| restart vs control, gen 50 (`g50-vs-s200g50.txt`) | 49/100 | 0.394-0.587 | −7 (−75 to +61) |
| restart gen 50 vs start (`g50-vs-start.txt`) | 62/100 | 0.522-0.709 | +85 (+15 to +155) |

For comparison, the control's generation 50 scored 64/100 (0.542-0.727)
against the same start (`9x9-sims/s200g50-vs-start.txt`).

**Against gen 200.** Re-rated like the other runs (`rerate/`), Elo minus
`ref1`'s Elo (−105.4 in `rerate/references.csv`) is +105 at the start,
+182 at generation 25 and +126 at generation 50, on the scale of the
all-runs table above (the control: +104, +152, +132). Single generations
carry the table's ±40 noise: generation 25's +182 is above the control's
+152, yet it lost the direct match 36/100. Averaged over stretches of the
run, the same numbers show the high-rate dip:

| generations | restart | control |
|---|---|---|
| 1–10 (0.02) | +25 | +128 |
| 11–19 (0.02) | +85 | +125 |
| 20–34 (0.006) | +118 | +152 |
| 35–50 (0.002) | +153 | +150 |

For its first ten generations the restart sat below its own start (+105;
lowest −35 at generation 9), and generation 1 scored 0.40 against
generation 0 in the training arena (`progress.txt`). These are means of
neighbouring ratings, which are not independent: a description of the
curve, not a test.

**Verdict.** The higher rate cost ground while it was high and the run
made it back once the rate returned to 0.002. At generation 25 the
restart was behind the control, 36/100 (interval 0.273-0.458, about −100
Elo), the one clear difference in these matches; at generation 50 it was
level, 49/100 (0.394-0.587), and its gain over the start (62/100) matches
the control's (64/100). So this restart gave **no measurable gain** at
100-game resolution. It fits the warning in the main README that 0.02
would wreck what a trained network learned: it set the network back,
though not beyond recovery in 50 generations.

What it does not show: that the learning rate does not matter. It tests
one schedule (a peak of 0.02 with steps at generations 20 and 35), one
seed and 50 generations; a smaller restart or a longer tail at 0.002
might differ, and a gain under about 70 Elo would not show in a 100-game
match. Nor is it a test of more learning: it took the same 10,000 SGD
steps as the control (see the correction in the games section above).

### Arena screens of AlphaGo Zero search settings

Two cheap screens from the [Next experiments](#next-experiments) list
(items 2 and 6), made with arena games only and no training: AlphaGo
Zero's random symmetry in search, and its search settings `c_puct` and
`mcts_batch`. The result: random symmetry makes the search clearly
stronger at play time (about +40 Elo at 200 simulations, +60 at 800);
the preset `c_puct = 1.5` and `mcts_batch = 8` are as good as anything
tried, so they stay.

- *Method.* Generation 250 of `9x9-main` (6×64) plays itself, 400 games,
  komi 7.5, colors alternating, with one setting changed on side A only
  (arena's one-sided flags; see "Random symmetry in search" and "Tuning
  search settings in the arena" in the main README). Both sides have the
  same network and budget, so A's score away from 0.5 is the effect of
  that setting alone, at this network and budget. Elo is
  400·log10(p/(1−p)), approximate; intervals are the 95% Wilson
  intervals `arena` prints. At 400 games an even match's interval is
  about ±35 Elo.
- *Symmetry screen* (`sym-screen.sh`): `--a-random-symmetry` at 200 and
  at 800 simulations, with arena's default opening temperature
  (`--temperature-moves 4`). Random symmetry changes the evaluations, not
  PUCT's exploration rule, so the opening sampling should penalise it far
  less than it does the settings below (main README).
- *Tuning screen* (`tune-screen.sh`, then `tune-screen2.sh` for the
  points around 1.5): `--a-c-puct` or `--a-mcts-batch 1` at 200
  simulations, with `--temperature-moves 0` and `--random-symmetry` on
  both sides. Opening temperature samples the first plies from each
  side's visit counts, and a setting that spreads those counts would be
  penalised for sampling worse moves rather than for searching worse; at
  temperature 0 each side plays its most-visited move and the symmetries
  keep the games apart. Every match had 400 distinct games. Side B keeps
  the presets' `c_puct = 1.5` and `mcts_batch = 8`.
- *Sources.* `docs/results/9x9-arena-screens/`: the three scripts and
  their logs, `sym-screen/*.txt` and `tune-screen/*.txt` (each `arena`'s
  output, with its `Search:` line). The scripts ran binaries copied from
  builds of main (`runs/bin-sym`, `runs/bin-tune`; the latter at commit
  28f0ee0); the `cd` to the absolute checkout path is made `cd .`.

| side A (B: 1.5, batch 8, same network) | file | A's score | 95% interval | Elo (interval) |
|---|---|---|---|---|
| random symmetry, 200 sims | `sym-screen/g250-symA-vs-g250.txt` | 223/400 = 0.557 | 0.509-0.605 | +40 (+6 to +74) |
| random symmetry, 800 sims | `sym-screen/g250-symA-vs-g250-s800.txt` | 234/400 = 0.585 | 0.536-0.632 | +60 (+25 to +94) |
| `c_puct` 0.75 | `tune-screen/cpuct0.75.txt` | 193/400 = 0.482 | 0.434-0.531 | −12 (−46 to +22) |
| `c_puct` 1.0 | `tune-screen/cpuct1.txt` | 212/400 = 0.530 | 0.481-0.578 | +21 (−13 to +55) |
| `c_puct` 1.25 | `tune-screen/cpuct1.25.txt` | 198/400 = 0.495 | 0.446-0.544 | −3 (−38 to +31) |
| `c_puct` 1.5 (the preset) | — | 0.5 by construction | | 0 |
| `c_puct` 2.0 | `tune-screen/cpuct2.txt` | 174/400 = 0.435 | 0.387-0.484 | −45 (−80 to −11) |
| `c_puct` 3.0 | `tune-screen/cpuct3.txt` | 143/400 = 0.357 | 0.312-0.406 | −102 (−137 to −66) |
| `mcts_batch` 1 | `tune-screen/batch1.txt` | 217/400 = 0.542 | 0.494-0.591 | +30 (−4 to +64) |

(No draws in any match. The symmetry rows used opening temperature 4;
the rest temperature 0 with symmetry on both sides.)

Reading it, honestly:

- **Random symmetry: a real gain in play strength.** Both intervals
  exclude 0.5. Each evaluation sees the position in a random one of its
  8 orientations, so the network's orientation-specific errors average
  out over the tree instead of repeating at every visit. The gain at 800
  simulations (+60) is larger than at 200 (+40), but the intervals
  overlap, so this does not show that it grows with the budget. This is
  search strength at play time; it is **not yet evidence that it helps
  training**. That was the self-play arm `9x9-sym`: the `sims200`
  control's start, seed and settings plus `mcts_random_symmetry = true`,
  evaluated with 400-game matches. It trained no measurably stronger
  network than the control; see
  [Random symmetry in self-play](#random-symmetry-in-self-play).
- **`c_puct`: flat from 0.75 to 1.5, worse above.** 0.75, 1.0 and 1.25
  are all within about 20 Elo of the preset, and every interval includes
  0.5; 2.0 (−45) and 3.0 (−102) are clearly worse. The preset 1.5 is at
  or near the best for this network at 200 simulations, and no change is
  recommended.
- **`mcts_batch` 1 against 8: no significant difference.** A sequential
  search scored +30 Elo, with an interval that includes 0.5. Virtual-loss
  batches of 8, AlphaGo Zero's mini-batch size, cost little or nothing
  measurable at 400 games while making 8 times fewer network calls (on
  this CPU backend a batch of 8 evaluates about 1.3 times as many
  positions per second as a batch of 1; `bench.txt`).

What this does not show: one network (generation 250), one simulation
count for the tuning, and matches against itself, which measure how
well a setting plays, not what it does to training. Settings that
search better at play time need not train a better network, and a
smaller effect than about ±35 Elo would not show at 400 games. Six
tuning matches were run, so one interval excluding 0.5 by chance would
not be surprising; the two that do (2.0 and 3.0) agree with each other
in direction and size.

### AlphaGo Zero training arms: setup and the control at 400 games

Four training arms from the [Next experiments](#next-experiments) list
(items 1, 3, 2 and 4). Three are the `sims200` control plus one AlphaGo
Zero setting: resignation, fewer opening moves at temperature, and a
random symmetry for each network evaluation in self-play. The fourth
doubles the learning budget: twice the games and twice the SGD steps per
generation, at the control's replay ratio. Resignation and temperature
beat the control at generation 50 by about +70 Elo, with intervals that
exclude 0.5. The resignation result is confounded by how much less data
it makes; the temperature result is the cleaner one. Random symmetry was
level with the control at generation 50. The learning-budget arm was
ahead at generations 25 and 50 (about +100 and +65 Elo) and gained the
most over its start, for twice the compute per generation. Details in
the four sections below.

Two more arms followed, launched together on 6 October: the steps only
(the control plus `train_steps = 400`), which gained nothing, and the
recipe (the control plus `temperature_moves = 7`, resignation, 200 games
and 400 steps, for 100 generations), which gained by far the most. They
have sections of their own after the first four. The pooled evaluation
in [Final strength of the post-250 runs](#final-strength-of-the-post-250-runs-pooled-generations-46-to-50)
ranks all six against the control with less noise than these matches.

- *Setup, common to the arms.* The control is `9x9-sims/sims200/`
  (start: the 6×96 generation 50 of `9x9-wide96` and its buffer of
  199,972 positions from 2,272 games; 200 simulations, 100 games and 200
  training steps of 256 positions per generation, learning rate 0.002,
  seed 1, the 100-simulation bot in the training arena, 50 generations).
  Each arm is the same command plus one `--set`, two for the budget
  arm (`agz-experiment.sh`, `agz-experiment2.sh`; the second differs only in
  taking the binary directory from `B`). Each arm's `config.cfg` differs
  from the control's only in that setting (for resignation, with its
  other `resign_*` keys at their defaults; for the budget arm,
  `games_per_generation` and `train_steps`), apart from keys added to the
  code since the control ran, at their defaults (`ownership`,
  `ownership_weight`, `reference_networks`; for the resignation arm
  `mcts_random_symmetry` and `arena_random_symmetry`; for the symmetry
  arm `arena_random_symmetry` and the `resignation` and `resign_*`
  keys). Like the control, none of the first four arms' binaries had
  `validation_loss` yet. The last two were run by `next-arm.sh` (the same
  command plus their `--set`s, then the matches and the re-rate) with
  `runs/bin-next`, built from commit be9e708, which logs `val_*` and
  `seen_*`; their `config.cfg` also lists `validation_loss = true`.
- *Matches.* 400 games per match (the earlier sections used 100), 200
  simulations for both sides, komi 7.5, colors alternating, arena's
  default opening temperature (4 plies), made by the script after
  training: generation 25 against the control's generation 25,
  generation 50 against the control's generation 50, and generation 50
  against the start. At 400 games an even match's interval is about ±35
  Elo. Opening temperature keeps the games apart, but not entirely: 347
  to 388 of each 400 were distinct (334 in the recipe's generation 100
  against its generation 50). `next-arm.sh` added generation 50 against
  the budget arm's generation 50 and, for the recipe, generation 100
  against its own generation 50 and against the start.
- *Re-rate.* Each arm was then re-rated against generation 200 of
  `9x9-main` exactly like the other post-250 runs (`rerate-arms.sh` and
  its log; `rerate/` in each folder). The symmetry arm's re-rate is the
  script's `rr` command run by hand afterwards with `runs/bin-tune`, the
  binary it trained with; that command was not saved, and its exit is
  the log's fourth line. The budget arm's re-rate was also run by hand,
  the same way, after its matches; that command was not saved either,
  and its exit is the log's last line. `next-arm.sh` re-rated the last
  two arms itself, with `runs/bin-next`. See the rows added to
  [the all-runs table](#all-post-250-runs-against-one-reference-gen-200).
- *Cost.* Wall times in these runs' `log.csv` are not comparable with the
  control's or with each other: the first three arms ran at the same
  time, with the arena screens above, so three or four jobs shared the
  CPU (the control's training took 94.4 s per generation; the
  resignation arm's 274.3 s, the temperature arm's 244.8 s and the
  symmetry arm's 275.6 s, for the same 200 steps). The budget arm
  followed the temperature arm in the same queue, alongside the
  symmetry arm and the re-rates. Plies per game, and for the budget arm
  its work per generation, are used below instead. The last two arms
  each ran alone on one of two Mac minis, so their times compare with
  the control's: GNU Parallel ran `next-arm.sh` once per line of
  `next-jobs.txt`, the steps-only arm on the second Mac mini and the
  recipe on this Mac (`next-jobs.log`, `next-jobs.joblog`).
- *Sources.* `docs/results/9x9-resign/`, `docs/results/9x9-sym/`,
  `docs/results/9x9-temp7/`, `docs/results/9x9-budget-g200-s400/`,
  `docs/results/9x9-steps400/` and `docs/results/9x9-recipe/`:
  `log.csv`, `elo.csv`, `results.csv`, `references.csv`, `config.cfg`,
  `init.txt`, `progress.txt`, the match files, the scripts and
  their logs, `rerate/`, and `buf.txt`, `pass.txt` and `side.txt`, the
  output of `tools/buf.py`, `tools/pass.py` and `tools/side.py` on each
  finished run's `state.bin` (the buffer at generation 50; the same for
  the control in `9x9-sims/sims200/`; the recipe's is at generation 200,
  see its section). Absolute paths are made
  repo-relative. One seed per arm.

**The control against its start, at 400 games.** The control's
generation 50 against the start scored **232/400 = 0.580** (95% interval
0.531-0.627, +56 Elo, +22 to +90; `9x9-sims/s200g50-vs-start-400.txt`,
made after the first three arms finished; the command was not saved, and the
file is `arena`'s output). This replaces the 64/100 (0.542-0.727, +100)
quoted against the start in the sections above; the two agree within
their intervals. It puts the seven runs on one scale against the start:

| generation 50 against the start | score | 95% interval | Elo (interval) |
|---|---|---|---|
| control (`9x9-sims/s200g50-vs-start-400.txt`) | 232/400 = 0.580 | 0.531-0.627 | +56 (+22 to +90) |
| resignation (`9x9-resign/g50-vs-start.txt`) | 256/400 = 0.640 | 0.592-0.686 | +100 (+65 to +136) |
| random symmetry (`9x9-sym/g50-vs-start.txt`) | 258/400 = 0.645 | 0.597-0.690 | +104 (+68 to +139) |
| `temperature_moves = 7` (`9x9-temp7/g50-vs-start.txt`) | 253/400 = 0.632 | 0.584-0.678 | +94 (+59 to +129) |
| learning budget (`9x9-budget-g200-s400/g50-vs-start.txt`) | 264/400 = 0.660 | 0.612-0.705 | +115 (+79 to +151) |
| steps only (`9x9-steps400/g50-vs-start.txt`) | 233/400 = 0.583 | 0.534-0.630 | +58 (+23 to +92) |
| recipe (`9x9-recipe/g50-vs-start.txt`) | 318/400 = 0.795 | 0.753-0.832 | +235 (+193 to +278) |

The scale is not exactly transitive: the control's +56 plus each arm's
margin over the control at generation 50 (+67, +74, −3; below) predicts
+123, +130 and +53 against the start, where the direct matches give
+100, +94 and +104; the budget arm's +65 predicts +121, where the
direct match gives +115. The first two gaps, about 25 to 35 Elo, and
the budget arm's 6 Elo are within the intervals. The third, about 50
Elo, is the largest: the
random-symmetry arm's generation 50 scored 0.645 against the start but
0.495 against the control's generation 50 (see
[Random symmetry in self-play](#random-symmetry-in-self-play)). The
steps-only arm misses by as much the other way: +56 plus its +51 over
the control predicts +107, where the direct match gives +58. The
recipe's +56 plus +228 predicts +284 against +235 measured, a gap
expected for scores this far from 0.5. A
single checkpoint also carries variance from one generation to the next
that a binomial interval does not include. The re-rate's curve over
many generations is one check on that; the pooled evaluation over
generations 46–50
([Final strength of the post-250 runs](#final-strength-of-the-post-250-runs-pooled-generations-46-to-50))
is the other, and the one the rankings rely on.

### Resignation in self-play

Hypothesis, from finding 2 of [Next experiments](#next-experiments):
35% of buffer positions come at or after the first pass, mostly the
loser playing on in a lost game. AlphaGo Zero resigned such games (Silver
et al. 2017, Methods, "Self-play"; see "Resignation" in the main
README): v_resign is calibrated so that at most 5% of the games played
out with resignation disabled (10% of games) would have been resigned by
their eventual winner. That saves self-play work and drops positions
whose targets say little. The result: the arm beat the control
(59.5% at generation 50) and roughly halved the plies per game, but it
also changed how much the network trains on each position, so the gain
cannot be put down to resignation alone.

- *Setup.* `9x9-resign/`: the control plus `resignation = true`, with
  the defaults `resign_threshold_initial = -0.9`,
  `resign_disabled_fraction = 0.1`, `resign_false_positive = 0.05`,
  `resign_window = 500`, `resign_min_games = 20` (`resign-arm.sh`,
  binary `runs/bin-resign`).
- *Resignation behaviour* (`log.csv`, generations 1–50). v_resign was
  the initial −0.90 for generations 1–2 and −0.88 from then on (once
  −0.87). The false-positive rate over the window was 0.041 on average
  (at most 0.050, within the 5% target). Of the games with resignation
  enabled, 98.0% were resigned (93.9% to 100% per generation): almost
  every game the loser could resign, it did. Black resigned 3,498 of the
  4,267 resigned games in the final buffer (`pass.txt`).
- *Game length.* A mean of **41.4 plies** per game (32.6 to 49.7 per
  generation) against the control's 88.3, counting the 10% played out
  to the end: 47% of the control's positions, and so of its network
  evaluations, per game. In the final buffer the resigned games average
  36.5 plies (`buf.txt`: 4,267 games, 155,611 positions). The review's
  estimate before the run was a saving of about 30%; the measured saving
  in plies is 53%.
- *What the buffer holds* (`buf.txt` and `pass.txt` against the
  control's, both at generation 50):

  | | control | resignation |
  |---|---|---|
  | games in the 200,000-position buffer | 2,261 | 4,843 |
  | positions at or after the first pass | 36.3% | 12.2% |
  | the same, played-out games only | 36.3% | 35.9% |
  | opening positions (plies 0–19) | 22% | 43% |
  | middle game (ply 20 to the first pass) | 42% | 45% |
  | mean target entropy (nats) | 1.948 | 1.900 |

  Resignation removed most of the after-pass positions of finding 2,
  as intended, and in their place the buffer holds about twice as many
  openings.

**The confound: the replay ratio and the buffer's span.** The arm kept
`train_steps = 200`, as the control did, but each generation added only
about 4,141 positions instead of 8,833. So the replay ratio
(`train_steps × batch_size / (games × average plies)`, the
`replay_ratio` column of `log.csv`, as in the games section) was a mean
of **12.49** against the control's **5.80**: each position was drawn
about 2.2 times as often. And the 200,000-position buffer spans about
48 generations of this run's games instead of about 23. The start's
buffer was flushed only at generation 49 (the control's by generation
23): at generation 25 about 45% of what the arm trained on was still
the start's full-length games. So this arm changed three things at once
compared with the control: resignation itself, the absence of most
after-pass positions, and 2.2 times the replay ratio over a buffer twice
as old. The games run halved the replay ratio and showed no clear
effect, which does not settle what doubling it does. A run that isolates
resignation would keep the replay ratio constant, for example
`train_steps` scaled to the positions per generation (about 94 steps
here); none was run.

- *Training curves.* Mean policy loss 2.372 against 2.390; value loss
  0.444 against 0.362, rising through the run (0.363 at generation 1,
  0.509 at 50) as openings made up more of the buffer. Black's
  self-play win rate was 0.184 against 0.265 (`log.csv` means). These are on different
  data and are not a strength measure.

**Matches** (400 games; see the common setup above; files in
`9x9-resign/`):

| match | score | 95% interval | Elo (interval) |
|---|---|---|---|
| resignation vs control, gen 25 (`g25-vs-control-g25.txt`) | 227/400 = 0.568 | 0.519-0.615 | +47 (+13 to +81) |
| resignation vs control, gen 50 (`g50-vs-control-g50.txt`) | 238/400 = 0.595 | 0.546-0.642 | +67 (+32 to +101) |
| resignation gen 50 vs start (`g50-vs-start.txt`) | 256/400 = 0.640 | 0.592-0.686 | +100 (+65 to +136) |

**Against gen 200.** Re-rated like the other runs (`rerate/`), Elo minus
`ref1`'s Elo (−143.6 in `rerate/references.csv`) is +144 at the start,
+146 at generation 25 and +201 at generation 50 (the control: +104,
+152, +132). The start is the same network as the control's, so its
+144 against +104 is the re-rate's noise, about ±40 Elo as in the
all-runs table. Means over stretches of the run (neighbouring ratings,
not independent; a description of the curve, not a test):

| generations | resignation | temp7 | symmetry | budget | steps400 | recipe | control |
|---|---|---|---|---|---|---|---|
| start (gen 0) | +144 | +105 | +123 | +94 | +119 | +104 | +104 |
| 1–10 | +126 | +145 | +127 | +114 | +104 | +112 | +128 |
| 11–25 | +178 | +150 | +136 | +132 | +148 | +187 | +133 |
| 26–50 | +191 | +166 | +155 | +187 | +146 | +240 | +152 |
| 51–75 | | | | | | +297 | |
| 76–100 | | | | | | +331 | |
| 101–125 | | | | | | +384 | |
| 126–150 | | | | | | +412 | |
| 151–175 | | | | | | +431 | |
| 176–200 | | | | | | +457 | |

On the gen-200 scale the arm is ahead of the control from generation 11
on, by about 40 to 45 Elo; measured from each run's own start it is
level with the control over generations 26–50 (+47 and +48). The
re-rate agrees with the matches in direction and cannot tell more.

**Verdict.** At equal generations the resignation arm **beat the
control**: 0.568 at generation 25 and 0.595 at generation 50 (about +67
Elo, interval +32 to +101), and it gained +100 over the start against
the control's +56. Its self-play games were 53% shorter, which is the
saving AlphaGo Zero used it for, at a false-positive rate within the 5%
target. What the run does **not** show is that resignation caused the
gain: the arm also trained at 2.2 times the control's replay ratio over
a buffer spanning twice as many generations, and its data lost most of
the after-pass positions. Any of those could account for some or all
of the +67. Either way it is AlphaGo Zero's method, and on this
evidence it costs no strength while saving about half the self-play
positions per game. One seed; no claim beyond that.

*Later note.* Pooled over generations 46–50 (1,000 games against the
start), the arm is +22 Elo over the control, inside that comparison's
±35 noise; the +67 at generation 50 came from one of its stronger
checkpoints. So the gain is borderline, not established. Combined with
the other changes in the recipe arm, resignation was part of by far
the largest gain; see [the pooled evaluation](#final-strength-of-the-post-250-runs-pooled-generations-46-to-50).

### Seven opening moves at temperature (`temperature_moves = 7`)

Hypothesis: AlphaGo Zero sampled its self-play moves in proportion to
visit counts (temperature 1) for the first 30 moves and played the
most-visited move after that (Silver et al. 2017, Methods,
"Self-play"). 30 moves of 361 points is about 8%, about 7 moves on 9×9;
the 9×9 preset then sampled for 20. Sampling fewer moves keeps more of each
game at the search's best move, which should give the value head
outcomes that follow from better play. The result: the arm beat the
control at generation 50 by about the same margin as resignation, with
nothing else changed.

- *Setup.* `9x9-temp7/`: the control plus `temperature_moves = 7`
  (`agz-queue.sh`, which then started the `budget-g200-s400` run;
  binary `runs/bin-agz`). Root Dirichlet noise is unchanged. The
  setting changes only which moves self-play plays, not how much data
  it makes: a mean of 91.1 plies per game against 88.3, a replay ratio
  of 5.62 against 5.80, and the same 10,000 SGD steps.
- *What the buffer holds* (`buf.txt`, `pass.txt`, `side.txt` against
  the control's, at generation 50). Black's self-play win rate rose
  from 0.255 to 0.363 (games in the buffer). Fewer positions come at or
  after the first pass (30.2% against 36.3%; the loser plays on for 27.5
  plies after the first pass against 32.1), and more are middle game
  (48% against 42%). The mean policy-target entropy fell from 1.948 to
  1.873 nats, and in the middle game from 1.927 to 1.787 (finding 1's
  table, plies 20 to the first pass, `side.txt`):

  | side to move | entropy, temp7 | control | moves visited, temp7 | control |
  |---|---|---|---|---|
  | eventual loser, Black | 2.375 | 2.700 | 40.4 | 42.4 |
  | eventual loser, White | 1.969 | 2.092 | 28.9 | 29.7 |
  | eventual winner, Black | 1.502 | 1.511 | 26.1 | 22.8 |
  | eventual winner, White | 1.249 | 1.242 | 14.7 | 10.4 |

  The losing side's targets, Black's above all, are less spread; the
  winners' are unchanged. The opening positions (plies 0–19) have
  slightly more spread targets (2.137 against 2.002 nats).
- *Training curves.* Mean policy loss 2.348 against 2.390 (the targets'
  entropy, the floor for that loss, also fell); value loss 0.493 against
  0.362, rising from 0.37 at generation 1 to 0.51–0.52 from generation
  11 on. More balanced games (Black winning 36% rather than 26%) make
  the result harder to predict, which fits a higher value loss, but that
  was not tested. These are on different data and are not a strength
  measure.

**Matches** (400 games; see the common setup above; files in
`9x9-temp7/`):

| match | score | 95% interval | Elo (interval) |
|---|---|---|---|
| temp7 vs control, gen 25 (`g25-vs-control-g25.txt`) | 220/400 = 0.550 | 0.501-0.598 | +35 (+1 to +69) |
| temp7 vs control, gen 50 (`g50-vs-control-g50.txt`) | 242/400 = 0.605 | 0.556-0.652 | +74 (+39 to +109) |
| temp7 gen 50 vs start (`g50-vs-start.txt`) | 253/400 = 0.632 | 0.584-0.678 | +94 (+59 to +129) |

**Against gen 200.** Re-rated like the other runs (`rerate/`), Elo minus
`ref1`'s Elo (−104.6 in `rerate/references.csv`) is +105 at the start,
+151 at generation 25 and +175 at generation 50 (the control: +104,
+152, +132). The stretch means are in the resignation section's table:
about 15 Elo above the control over generations 1–50 (+61 over its own
start in 26–50, the control +48), within the ±40 noise but in the same
direction as the matches.

**Verdict.** `temperature_moves = 7` **beat the control** at generation
50, 0.605 (about +74 Elo, interval +39 to +109), was barely ahead at
generation 25 (0.550, lower bound 0.501), and gained +94 over the start
against the control's +56. It is the cleanest single-change result of
the post-250 runs: one config key, the same data volume, the same
replay ratio and the same number of SGD steps. It also brings the
setting closer to AlphaGo Zero's proportion of 30 of 361 points.
`temperature_moves` around 7 is therefore a candidate for the 9×9
presets; the presets were unchanged here (that decision was Mark's; the
9×9 preset later adopted it as part of the recipe). What
it does not show: one seed, 50 generations from one start, one value
(7 against 20, nothing between). Earlier 100-game matches showed gains
of this size at generation 25 that were gone by generation 50 (sims400,
games200); here the margin is larger at 50 than at 25, and the 400-game
interval excludes 0.5.

*Later note.* Pooled over generations 46–50, the arm is +37 Elo over the
control, outside that comparison's ±35 noise: the gain is confirmed,
though smaller than the +74 of the generation-50 match; see
[the pooled evaluation](#final-strength-of-the-post-250-runs-pooled-generations-46-to-50).

### Random symmetry in self-play

Hypothesis: AlphaGo Zero evaluated each position in its search under a
random one of the board's 8 rotations and reflections (Silver et al.
2017, Methods, "Expand and evaluate"; see "Random symmetry in search"
in the main README). In the [arena screens](#arena-screens-of-alphago-zero-search-settings)
that made the search stronger at play time, about +40 Elo at 200
simulations. If self-play searches better, its visit counts and game
results should be better targets, and the network trained on them
stronger. The result: it was not measurably stronger. The arm was level
with the control at generation 50 and on the re-rate, so the play-time
gain did not turn into a training gain here.

- *Setup.* `9x9-sym/`: the control plus `mcts_random_symmetry = true`
  (`sym-arm.sh`, which waited for the tuning screen and then ran
  `agz-experiment2.sh` with `B=runs/bin-tune`). Only self-play's search
  used the symmetries: `arena_random_symmetry` stayed off, so the
  trainer's arena rounds, the re-rate and the 400-game matches searched
  on the untransformed board, as for the control and the other arms.
  The setting changes only the evaluations, not how much data self-play
  makes: a mean of 87.8 plies per game against 88.3, a replay ratio of
  5.84 against 5.80, and the same 10,000 SGD steps.
- *Cost.* Nothing measurable. The input planes are always encoded
  through a transform, the identity when the setting is off; with it on,
  a random one is drawn and the 82 policy logits are mapped back
  (`src/mcts.cpp`). The network call is the same. The wall times cannot
  show it (see the common setup).
- *What the buffer holds* (`buf.txt`, `pass.txt`, `side.txt` against
  the control's, at generation 50). Almost the same as the control's.
  The mean policy-target entropy was 1.969 nats against 1.948, with 25.3
  moves visited against 25.4 and a top-move share of 0.413 against
  0.422; 35.1% of positions came at or after the first pass against
  36.3%; Black won 0.236 of the buffer's games against 0.255. By phase,
  the openings (plies 0–19) were 2.102 nats against 2.002, the middle
  game 1.908 against 1.927, after the first pass 1.958 against 1.939. In
  the middle game by side to move (finding 1's table, `side.txt`):

  | side to move | entropy, symmetry | control | moves visited, symmetry | control |
  |---|---|---|---|---|
  | eventual loser, Black | 2.669 | 2.700 | 42.0 | 42.4 |
  | eventual loser, White | 1.968 | 2.092 | 27.0 | 29.7 |
  | eventual winner, Black | 1.568 | 1.511 | 25.2 | 22.8 |
  | eventual winner, White | 1.237 | 1.242 | 10.3 | 10.4 |

  Finding 1's spread of the losing side's search is unchanged, as
  expected: it comes from PUCT's Q = 0 for unvisited moves, which the
  symmetries do not touch. The openings' targets are about 0.1 nats
  more spread, which would fit visits shared among symmetrically
  equivalent points instead of following one orientation's preference,
  but that was not checked.
- *Training curves.* Mean policy loss 2.389 against 2.390, value loss
  0.362 against 0.362; Black's self-play win rate 0.224 against 0.265
  (`log.csv` means). These are on different data and are not a strength
  measure, but here they are almost the same data.

**Matches** (400 games; see the common setup above; files in
`9x9-sym/`):

| match | score | 95% interval | Elo (interval) |
|---|---|---|---|
| symmetry vs control, gen 25 (`g25-vs-control-g25.txt`) | 222/400 = 0.555 | 0.506-0.603 | +38 (+4 to +73) |
| symmetry vs control, gen 50 (`g50-vs-control-g50.txt`) | 198/400 = 0.495 | 0.446-0.544 | −3 (−38 to +31) |
| symmetry gen 50 vs start (`g50-vs-start.txt`) | 258/400 = 0.645 | 0.597-0.690 | +104 (+68 to +139) |

The three matches do not tell one story. Generation 25 was barely ahead
of the control's (lower bound 0.506), generation 50 level with it; and
generation 50 gained +104 over the start against the control's +56,
which the level match at generation 50 does not fit (about 50 Elo off;
see the common setup). Each match measures a single checkpoint, and a
network's strength moves from one generation to the next by more than
a 400-game interval includes. Read alone, the 0.555 or the 0.645 would
suggest a gain; the 0.495 would not. The re-rate's curve over all 50
checkpoints is the better guide here.

**Against gen 200.** Re-rated like the other runs (`rerate/`), Elo minus
`ref1`'s Elo (−122.8 in `rerate/references.csv`) is +123 at the start,
+132 at generation 25 and +125 at generation 50 (the control: +104,
+152, +132). The stretch means are in the resignation section's table:
within 3 Elo of the control over each stretch (+143 over generations
1–50, the control +142). Measured from its own start it gained +32 in
26–50 against the control's +48 (resignation +47, temp7 +61); the start
is the same network in every arm, so the difference is the re-rate's
±40 noise, not a loss.

**Verdict.** Random symmetry in self-play **trained no measurably
stronger network** at this scale: level with the control at generation
50 (0.495, −3 Elo, interval −38 to +31) and on the re-rate over all 50
generations. The play-time gain of the arena screens (+40 Elo at 200
simulations, +60 at 800) did not turn into a stronger network over 50
generations here, and the data barely changed: game length, the
after-pass share and the targets' entropy are within a few percent of
the control's. The single-checkpoint matches (0.555 at generation 25,
0.495 at 50, 0.645 against the start) are an example of checkpoint
variance: any one of them alone could have been read as a gain or as
none. The setting costs nothing, so leaving it on would be harmless and
closer to AlphaGo Zero, but this run gives no reason to. What it does not
show: one seed, 50 generations from one start, 200 simulations; an
effect under about ±35 Elo would not show in the matches, and the
re-rate's noise is about ±40. The follow-up, not run: if play strength
is the goal, use the symmetries at evaluation, where the screens
measured them, with `arena_random_symmetry` in the trainer's arena
rounds and re-rates and `--random-symmetry` in `arena`, `analyze` and
`gtp`, for about +40 to +60 Elo at play time. (A re-rate with a different
`arena_random_symmetry` from the live run plays other games, so ratings
made with and without it should not be mixed.)

*Later note.* Pooled over generations 46–50, the arm is +22 Elo over the
control, inside that comparison's ±35 noise, the same as the
resignation arm: consistent with no measurable gain, and the
generation-50 match (0.495) was one of its weaker checkpoints; see
[the pooled evaluation](#final-strength-of-the-post-250-runs-pooled-generations-46-to-50).

### The learning budget: 200 games and 400 steps per generation

Hypothesis, from finding 3 of [Next experiments](#next-experiments):
every post-250 run took the same 10,000 SGD steps in 50 generations,
and the games run doubled the self-play games without the steps,
halving the replay ratio, and showed no clear gain. If the amount of
learning is the limit, doubling the steps together with the games, so
that each position is drawn as often as in the control, should help
where the games run did not. The result: the largest and most
consistent gain of the post-250 arms, for twice the compute per
generation.

- *Setup.* `9x9-budget-g200-s400/`: the control plus
  `games_per_generation = 200` and `train_steps = 400` (`agz-queue.sh`,
  after the temperature arm; binary `runs/bin-agz`). Its `config.cfg`
  differs from the games run's only in `train_steps` (400 against
  200), so the two runs together separate the steps from the games at
  200 games per generation. In 50 generations it took 20,000 SGD steps
  on 10,000 games, against the control's 10,000 on 5,000. The replay
  ratio (the `replay_ratio` column of `log.csv`) was a mean of **5.81**
  (5.58 to 6.22 per generation; 400 × 256 / (200 × 88.2)) against the
  control's 5.80 and the games run's 2.90, with 88.2 plies per game
  against 88.3. With about 17,600 new positions per generation, the
  200,000-position buffer spans about 11 generations of games instead
  of about 23, and the start's buffer was flushed at generation 12, as
  in the games run.
- *Cost.* Twice the self-play and twice the training per generation,
  by construction; the arena rounds are unchanged. The wall times are
  not comparable with the control's (see the common setup): training
  took 300 to 880 s per generation in generations 1–10 and 18–21,
  most likely while other jobs shared the CPU, and 189 to 198 s in the
  others. In generations 30–50, self-play took 391 s and training 189 s per
  generation, 1.9 and 2.0 times the control's means of 201 and 94 s, as
  the doubled work predicts.
- *What the buffer holds* (`buf.txt`, `pass.txt`, `side.txt` against
  the control's, at generation 50). Almost the same as the control's:
  mean target entropy 1.953 nats against 1.948 (openings 2.061 against
  2.002, middle game 1.897 against 1.927), 25.2 moves visited against
  25.4, 35.9% of positions at or after the first pass against 36.3%,
  and Black winning 0.229 of the buffer's games against 0.255. Finding
  1's spread of the losing side's search is unchanged (the eventual
  loser as Black in the middle game: 2.679 nats and 42.5 moves visited,
  against 2.700 and 42.4). The buffer holds games from fewer, more
  recent generations, but by these statistics they are the same kind of
  data.
- *Training curves.* Mean policy loss 2.381 against 2.390 (2.366
  against 2.376 at generation 50), value loss 0.359 against 0.362;
  Black's self-play win rate 0.248 against 0.265 (`log.csv` means).
  These are on different data and are not a strength measure.

**Matches** (400 games; see the common setup above; files in
`9x9-budget-g200-s400/`):

| match | score | 95% interval | Elo (interval) |
|---|---|---|---|
| budget vs control, gen 25 (`g25-vs-control-g25.txt`) | 256/400 = 0.640 | 0.592-0.686 | +100 (+65 to +136) |
| budget vs control, gen 50 (`g50-vs-control-g50.txt`) | 237/400 = 0.593 | 0.544-0.640 | +65 (+31 to +100) |
| budget gen 50 vs start (`g50-vs-start.txt`) | 264/400 = 0.660 | 0.612-0.705 | +115 (+79 to +151) |

**Against gen 200.** Re-rated like the other runs (`rerate/`, run by
hand; see the common setup), Elo minus `ref1`'s Elo (−94.0 in
`rerate/references.csv`) is +94 at the start, +164 at generation 25 and
+141 at generation 50 (the control: +104, +152, +132). The stretch
means are in the resignation section's table: +187 over generations
26–50, and measured from its own start +93 there, the largest gain of
the arms (the control +48, resignation +47, temp7 +61, symmetry +32;
the games run +34). Two cautions. Its start, the same network as in
every arm, rated +94, the lowest of the five, so part of that +93 is
the start's noise; on the absolute scale its 26–50 mean, +187, is
level with resignation's +191 and above temp7's +166 and the control's
+152. And per unit of compute it is not ahead: its generation g has
played as many self-play games and taken as many SGD steps as the
control's generation 2g, and at those matched points (its generations
1–25 against the control's even generations 2–50) it averages 19 Elo
below the control, within the noise. Its generation 25, at the
control's generation-50 compute, is +164 against +132 as single
checkpoints; no match between those two was played.

**Verdict.** Doubling the games and the SGD steps together **beat the
control** at both checkpoints: 0.640 at generation 25 (+100 Elo,
interval +65 to +136), the largest margin of the arms there (the others
0.550 to 0.568), and 0.593 at generation 50 (+65, interval +31 to
+100), like resignation's and temperature's. It gained +115 over the
start against the control's +56, the most of the five runs, and the
most over its own start on the re-rate. Read with the games run, which
differs from it only in `train_steps`: at 200 games per generation,
200 steps (a replay ratio of 2.9) gave no clear gain over the control,
and 400 steps (5.8) the largest gain here. That points at the learning
budget, the SGD steps, as the binding constraint of these runs, as
finding 3 suspected. It is not a free win: it costs about twice the
compute per generation, and at equal compute the re-rate puts it level
with the control, which would reach the same steps and games in twice
the generations. What it does not separate is whether more steps help
on their own or only together with more fresh data at the same replay
ratio. One seed per arm; no claim beyond that.

*Later note.* Two runs started on 6 October answered that question and
the next. Steps alone, 400 per generation on the control's 100 games
(a replay ratio of 11.7), gained nothing (+5 Elo over the control
pooled over generations 46–50; see
[Steps only](#steps-only-100-games-and-400-steps-per-generation)), so
it is the steps together with fresh data that helped here, not the
steps alone, and "the SGD steps as the binding constraint" above is too
strong. Pooled, this arm is +44 over the control, outside that
comparison's ±35 noise: the gain is confirmed, smaller than the +65 to
+100 of its single matches; see [the pooled evaluation](#final-strength-of-the-post-250-runs-pooled-generations-46-to-50). Combined with
`temperature_moves = 7` and resignation it gave by far the largest gain;
see [The recipe](#the-recipe-four-changes-together-250-generations).

### Steps only: 100 games and 400 steps per generation

Hypothesis: the learning-budget arm doubled the games and the SGD steps
together and gained; the games run doubled only the games and did not.
If the steps are what was missing, doubling them alone, on the
control's 100 games per generation, should gain too, at twice the
control's replay ratio. The result: no gain. Its single matches
disagreed with one another, and pooled over generations 46–50 it is +5
Elo over the control.

- *Setup.* `9x9-steps400/`: the control plus `train_steps = 400`
  (`next-arm.sh steps400 50 --set train_steps=400`, from
  `next-jobs.txt`; binary `runs/bin-next`). Its `config.cfg` differs
  from the control's only in `train_steps`, apart from keys added since
  the control ran, at their defaults. In 50 generations it took 20,000
  SGD steps on the control's 5,000 games. The data are the control's
  kind and amount: a mean of 87.3 plies per game against 88.3, so the
  200,000-position buffer spans about 23 generations, as in the
  control. The replay ratio (`replay_ratio` in `log.csv`) was a mean of
  **11.74** (11.13 to 12.38) against the control's 5.80: each position
  was drawn about twice as often.
- *Cost.* The run had its Mac mini to itself, and its self-play took
  201.7 s per generation against the control's 201.4, so the times
  compare. Training took 194.3 s against 94.4 (2.06 times; it includes
  the validation pass) and the arena 55.7 s against 55.4: 1.29 times the
  control's time per generation. The whole job, with the four matches
  and the re-rate, took 8.2 h (29,446 s, `next-jobs.joblog`).
- *What the buffer holds* (`buf.txt`, `pass.txt`, `side.txt` against
  the control's, at generation 50). The same as the control's: mean
  target entropy 1.975 nats against 1.948, 25.4 moves visited against
  25.4, 38.1% of positions at or after the first pass against 36.3%,
  Black winning 0.245 of the buffer's games against 0.255. The eventual
  loser as Black in the middle game: 2.720 nats and 42.6 moves visited,
  against 2.700 and 42.4.
- *Training curves.* Mean policy loss 2.400 against 2.390; value loss
  0.327 against 0.362, lower because the same positions were fitted
  twice as often; Black's self-play win rate 0.235 against 0.265
  (`log.csv` means). These are on the same kind of data, but the loss
  is measured on the buffer it fits, so it is not a strength measure.
- *The validation loss* (`val_*` and `seen_*` in `log.csv`; this and
  the recipe are the first runs that log them; see "The validation loss"
  in the main README). The policy terms are level with each other
  throughout: 2.360 fresh against 2.369 on the buffer over generations
  1–25, 2.354 against 2.359 over 26–50. The value terms part: 0.377
  fresh against 0.343 on the buffer over 1–25, 0.358 against 0.308 over
  26–50. The buffer's value loss fell and the fresh games' barely did,
  which by the README's reading is a mild sign of the value head fitting
  the buffer it draws from twice as often. The fresh value loss swings
  from 0.24 to 0.49 between generations, so the trend is not certain,
  and the control has no such columns to compare (its binary predates
  them).

**Matches** (400 games; see the common setup above; files in
`9x9-steps400/`):

| match | score | 95% interval | Elo (interval) |
|---|---|---|---|
| steps400 vs control, gen 25 (`g25-vs-control-g25.txt`) | 211/400 = 0.527 | 0.479-0.576 | +19 (−15 to +53) |
| steps400 vs control, gen 50 (`g50-vs-control-g50.txt`) | 229/400 = 0.573 | 0.524-0.620 | +51 (+16 to +85) |
| steps400 vs budget, gen 50 (`g50-vs-budget-g50.txt`) | 201/400 = 0.502 | 0.454-0.551 | +2 (−32 to +36) |
| steps400 gen 50 vs start (`g50-vs-start.txt`) | 233/400 = 0.583 | 0.534-0.630 | +58 (+23 to +92) |

The matches do not tell one story. Generation 50 beat the control's
(lower bound 0.524) and was level with the budget arm's, which had
itself beaten the control's 0.593; yet against the start it gained +58,
the control's +56, and generation 25 was level with the control's. The
pooled evaluation settles it: +73 against the start over generations
46–50 (604/1000), against the control's +68.

**Against gen 200.** Re-rated like the other runs (`rerate/`), Elo minus
`ref1`'s Elo (−119.0 in `rerate/references.csv`) is +119 at the start,
+151 at generation 25 and +55 at generation 50 (the control: +104,
+152, +132). The stretch means are in the resignation section's table:
+148 over generations 11–25 and +146 over 26–50, against the control's
+133 and +152; from its own start, +27 over 26–50 against the control's
+48. Single generations swing widely here (+252 at generation 37, +55 at
50), another example of single-checkpoint noise. No generation was
promoted in the re-rate.

**Verdict.** Doubling the SGD steps on the control's data **trained no
stronger network**: +5 Elo over the control pooled over generations
46–50 (well inside the ±35 noise of that comparison), level with it on
the re-rate, for 1.29 times the time per generation. Its generation-50
match against the control (0.573, interval excluding 0.5) would on its
own have been read as a gain of about +50; it is the clearest example
in these runs of why one checkpoint is not enough. Read with the games
run (more games at the same steps: +13 pooled) and the budget arm (both
together: +44 pooled): neither more steps on the same data nor more
data at the same steps helped, and both together did, modestly. So the
budget arm's gain is not the steps alone; the budget section's reading
of the steps as the binding constraint is corrected in its later note.
A replay ratio of 11.7 did not hurt measurably either, though the value
head's fresh-game loss stopped following its buffer loss. One seed; no
claim beyond that.

### The recipe: four changes together (250 generations)

Hypothesis: the three single changes that beat the control at
generation 50 act on different things: `temperature_moves = 7` on which
moves self-play plays, resignation on which positions it keeps and how
long its games are, and the learning budget (200 games and 400 steps)
on how much the network learns per generation. Combined, they might
add. With resignation, 200 games should cost about the self-play time
of the control's 100, so the learning budget comes cheaper. The run
was given 100 generations, then continued to 200 and to 250, to see
whether the gain keeps growing. The result: by far the largest gain of
the 9×9 experiments, still growing at generation 250, by about as much
per generation as over generations 100–200. Mark has
since made these four values the 9×9 preset (`configs/9x9.cfg`, commit
63b3979), and the preset has since been run from scratch; see
[The 9×9 preset from scratch](#the-99-preset-from-scratch-250-generations).

- *Setup.* `9x9-recipe/`: the control plus `temperature_moves = 7`,
  `resignation = true` (the other `resign_*` keys at their defaults, as
  in the resignation arm), `games_per_generation = 200` and
  `train_steps = 400`, for 100 generations (`next-arm.sh recipe 100`
  with those four `--set`s, from `next-jobs.txt`; binary
  `runs/bin-next`). Its `config.cfg` differs from the control's only in
  those four keys and `generations`, apart from keys added since the
  control ran, at their defaults. It ran alone on this Mac, launched by
  GNU Parallel at the same time as the steps-only arm on the second Mac
  mini; training took 15 h 15 min (09:41 on 6 October to 00:56 on 7
  October, `next-jobs.log`), the whole job with its six matches and the
  re-rate 19.0 h (68,533 s, `next-jobs.joblog`).
- *Continued to generation 200.* `recipe-continue.sh` resumed the run on
  the same Mac with the same binary (`train --resume --generations 200`;
  the resumed `config.cfg` differs from the one the run started with
  only in `generations`, 200 against 100), from 06:32 to 20:31 on 7
  October (13 h 59 min, `recipe-continue.log`, with the host name
  redacted). It then played three 400-game matches and re-rated
  generations 1–200 against gen 200, finishing at 01:59 on 8 October.
  `rerate/` is that re-rate over generations 1–200. The re-rate made at
  generation 100, which this section first used, is kept as
  `rerate-to100/`. The buffer statistics (`buf.txt`, `pass.txt`,
  `side.txt`) are from `state.bin` at generation 200 (`buf.txt`, first
  line); it holds the games of about generations 178–200. (The section's
  first version used a copy at generation 107; the figures were close,
  and are given below where they are compared.)
- *Continued to generation 250.* `recipe-continue2.sh` resumed it again
  on this Mac with the same binary (`train --resume --generations 250`;
  `config.cfg` changes only in `generations`), from 08:44 to 16:38 on 8
  October (`recipe-continue2.log`). This Mac's session was interrupted
  after generation 237, at about 13:46, and the run was resumed from
  there at 14:53 with `train --resume` (the fourth start in
  `progress.txt`); resume is byte-identical by design, so the result is
  the same run it would have been without the stop. The 50 generations
  took 6.8 h of generation time. It then played two 400-game matches,
  generation 250 against 200 and against the start. No re-rate was made
  past generation 200, so the re-rate figures below stop there, and the
  buffer statistics are still those of generation 200. The other files
  cover all 250 generations: `log.csv`, `progress.txt` (the lines of
  all four starts), `elo.csv`, `references.csv`, `results.csv` and
  `config.cfg` are the run's own.
- *Resignation behaviour* (`log.csv`). v_resign was the initial −0.90 at
  generation 1, then −0.76 to −0.84 through generation 50 (−0.76 to
  −0.78 for most of generations 7–45), −0.82 to −0.87 over generations
  51–100, −0.82 to −0.88 over 101–200 and −0.86 to −0.91 over 201–250;
  the resignation arm alone held −0.87 to −0.90. Of the games with
  resignation enabled, 98.6% were resigned in both halves of the first
  200 generations (96.2% to 100% per generation over generations 1–100,
  92.1% to 100% over 101–200) and 98.7% over 201–250 (95.5% to 100%).
  The false-positive rate over the window was a mean of 0.048 over
  generations 7–100 (at most 0.056; above 0.050 in 20 of those 94
  generations, since v_resign moves in steps of 0.01), 0.046 over
  101–200 (at most 0.058; above 0.050 in 12) and 0.048 over 201–250 (at
  most 0.056; above 0.050 in 7); it was 0.086 at generation 3, when the
  window held only a few dozen played-out games. The resignation arm
  averaged 0.041. In the generation-200 buffer, 4,101 of 4,569 games
  were resigned, 2,934 of them by Black (`pass.txt`).
- *Game length.* A mean of **50.8 plies** per game over generations
  1–100 (39.5 to 66.4 per generation; 66.4 at generation 1, 47.4 at
  100), **45.3** over 101–200 (34.4 to 53.9; 49.7 at 200) and **42.4**
  over 201–250 (34.9 to 51.0; 44.2 at 250), counting
  the 10% played out, against the control's 88.3 and the resignation
  arm's 41.4. The resigned games in the generation-200 buffer average
  39.4 plies (`buf.txt`: 161,687 positions; 44.9 at generation 107).
  Over generations 1–100, 200 games made about 10,100 positions per
  generation, 1.14 times the control's 8,830: the doubled games cost
  about the self-play of 115 played-out ones. Over 101–200 they made
  about 9,060, and over 201–250 about 8,470.
- *Replay ratio.* A mean of **10.15** over generations 1–100 (7.71 to
  12.97 per generation), **11.40** over 101–200 (9.49 to 14.89) and
  **12.18** over 201–250 (10.05 to 14.69), as the games shortened,
  against the control's 5.80, the budget arm's 5.81, the resignation
  arm's 12.49 and the steps-only arm's 11.74. The 200,000-position
  buffer spans about 20 to 23 generations; the start's buffer was
  flushed at generation 19.
- *Cost.* Generations 1–4 were slower (self-play 431 to 710 s, training
  308 to 474 s; the cause was not recorded). Over generations 5–100,
  self-play took 247.7 s per generation (1.23 times the control's
  201.4), training 199.8 s (2.12 times 94.4; it includes the validation
  pass) and the arena 81.4 s (55.4). Self-play and training together,
  447.5 s against 295.8, are 1.51 times the control's: a little more
  than the control's self-play and twice its training. Over generations
  101–200, with shorter games, self-play took 218.6 s, training 197.8 s
  and the arena 87.0 s, and over 201–250 206.6 s, 196.7 s and 84.5 s
  (8.1 minutes per generation).
- *What the buffer holds* (`buf.txt`, `pass.txt`, `side.txt`; the
  recipe's at generation 200, the others' at generation 50):

  | | control | resignation | recipe |
  |---|---|---|---|
  | games in the 200,000-position buffer | 2,261 | 4,843 | 4,569 |
  | positions at or after the first pass | 36.3% | 12.2% | 11.0% |
  | the same, played-out games only | 36.3% | 35.9% | 32.7% |
  | opening positions (plies 0–19) | 22% | 43% | 39% |
  | middle game (ply 20 to the first pass) | 42% | 45% | 50% |
  | mean target entropy (nats) | 1.948 | 1.900 | 1.617 |
  | Black's share of the games' wins | 0.255 | 0.184 | 0.290 |

  At generation 107 the recipe's column read 4,030 games, 10.5%, 30.2%,
  36%, 53%, 1.688 nats and 0.316. Resignation removed most of the
  after-pass positions, as in the resignation arm, and with seven moves
  at temperature fewer of the remaining positions are openings: half
  are middle game. The targets are sharper for every side to move. In
  the middle game (finding 1's table, `side.txt`) the eventual loser as
  Black has 2.007 nats and 42.8 moves visited (control 2.700 and 42.4),
  as White 1.712 (2.092); the eventual winner as Black 1.210 (1.511),
  as White 1.016 (1.242). Part of that is a stronger network's search:
  this buffer was made by generations 178–200, the control's by
  generations 27–50.
- *Training curves.* Policy loss 2.404 at generation 1, 2.290 at 50,
  2.228 at 100, 2.162 at 150, 2.136 at 200 and 2.140 at 250 (the
  control's mean 2.390); its targets' lower entropy lowers that loss's
  floor too. Value loss 0.38 at generation 1, 0.67 to 0.75 from about
  generation 15 to 100, and drifting down after that (a mean of 0.651
  over 101–200, 0.621 at 200, a mean of 0.607 over 201–250 and 0.580 at
  250; the control 0.362, the temperature arm 0.493, the resignation arm
  0.444), as resigned and more balanced games filled the buffer. Black's
  self-play win rate was 0.331 over generations 1–100 (0.363 over 1–25,
  0.311 over 76–100), 0.298 over 101–200 and 0.270 over 201–250, against
  the control's 0.265, the temperature arm's 0.363 and the resignation
  arm's 0.184. These are on different data and are not a strength
  measure.
- *The validation loss* (`val_*` and `seen_*`, means by quarter of each
  hundred generations, and over 201–250 by halves):

  | generations | value, fresh games | value, buffer | policy, fresh games | policy, buffer |
  |---|---|---|---|---|
  | 1–25 | 0.834 | 0.601 | 2.276 | 2.308 |
  | 26–50 | 0.815 | 0.720 | 2.268 | 2.272 |
  | 51–75 | 0.822 | 0.707 | 2.232 | 2.244 |
  | 76–100 | 0.796 | 0.681 | 2.198 | 2.215 |
  | 101–125 | 0.799 | 0.675 | 2.137 | 2.165 |
  | 126–150 | 0.809 | 0.664 | 2.128 | 2.134 |
  | 151–175 | 0.795 | 0.650 | 2.119 | 2.120 |
  | 176–200 | 0.797 | 0.634 | 2.100 | 2.124 |
  | 201–225 | 0.784 | 0.620 | 2.107 | 2.098 |
  | 226–250 | 0.768 | 0.604 | 2.100 | 2.113 |

  The policy terms fall together to generation 200, fresh games at or
  slightly below the buffer: learning, by the main README's reading.
  Over 201–250 they barely moved (fresh 2.107 and 2.100, buffer 2.098
  and 2.113), with the training loss flat at about 2.14. The value
  terms do not. Until generation 19 the buffer still held the start's
  played-out games, whose outcomes are easier to predict, so that gap
  says little. From generation 26 on, the fresh games' value loss
  stays at about 0.80 (0.815 over generations 26–50, 0.797 over
  176–200; 0.70 to 0.94 per generation over 101–200) while the
  buffer's keeps falling (0.720 to 0.634), so the gap widens slowly:
  0.095, 0.114, 0.115, 0.124, 0.145, 0.146 and 0.163 by quarter. Over
  generations 201–250 it held at 0.164 in both halves, while the fresh
  games' loss fell a little (0.784 and 0.768): the widening stopped. A gap
  that keeps widening while the fresh loss stops falling is the main
  README's description of overfitting the buffer, here in the value
  head only, mildly, at a replay ratio of about 11. It has not stopped
  the network getting stronger (the matches below), and the gap also
  reflects how fresh games differ from older ones, so it is a sign to
  watch, not a demonstrated limit. The steps-only arm, at a replay
  ratio of 11.7, showed a gap of 0.03 to 0.05 on value losses half as
  large; the from-scratch run's is in
  [its section](#the-99-preset-from-scratch-250-generations).

**Matches** (400 games; see the common setup above; files in
`9x9-recipe/`):

| match | score | 95% interval | Elo (interval) |
|---|---|---|---|
| recipe vs control, gen 25 (`g25-vs-control-g25.txt`) | 265/400 = 0.662 | 0.615-0.707 | +117 (+81 to +153) |
| recipe vs control, gen 50 (`g50-vs-control-g50.txt`) | 315/400 = 0.787 | 0.745-0.825 | +228 (+186 to +269) |
| recipe vs budget, gen 50 (`g50-vs-budget-g50.txt`) | 290/400 = 0.725 | 0.679-0.766 | +168 (+130 to +206) |
| recipe gen 50 vs start (`g50-vs-start.txt`) | 318/400 = 0.795 | 0.753-0.832 | +235 (+193 to +278) |
| recipe gen 100 vs gen 50 (`g100-vs-g50.txt`) | 268/400 = 0.670 | 0.622-0.714 | +123 (+87 to +159) |
| recipe gen 100 vs start (`g100-vs-start.txt`) | 339/400 = 0.848 | 0.809-0.879 | +298 (+251 to +345) |
| recipe gen 150 vs gen 100 (`g150-vs-g100.txt`) | 258/400 = 0.645 | 0.597-0.690 | +104 (+68 to +139) |
| recipe gen 200 vs gen 100 (`g200-vs-g100.txt`) | 299/400 = 0.748 | 0.703-0.788 | +189 (+150 to +228) |
| recipe gen 200 vs start (`g200-vs-start.txt`) | 361/400 = 0.902 | 0.869-0.928 | +387 (+329 to +444) |
| recipe gen 250 vs gen 200 (`g250-vs-g200.txt`) | 253/400 = 0.632 | 0.584-0.678 | +94 (+59 to +129) |
| recipe gen 250 vs start (`g250-vs-start.txt`) | 370/400 = 0.925 | 0.895-0.947 | +436 (+372 to +501) |

Every interval excludes 0.5 by a wide margin. The gaps add up only
roughly: +235 to generation 50 and +123 from 50 to 100 make +358, where
generation 100 against the start measures +298, and +298 plus +189 from
100 to 200 make +487, where generation 200 against the start measures
+387, and +387 plus +94 from 200 to 250 make +481, where generation 250
against the start measures +436, as expected for scores this far from
0.5. The gain per 50 generations fell in these matches, then held: +235
over the first 50 (against the start), +123 from 50 to 100, +104 from
100 to 150, about +85 from 150 to 200 (+189 from 100 to 200, less the
+104) and +94 from 200 to 250, about +19 Elo per 10 generations over
generations 100–250. Generation 100 against generation 50 had only 334
distinct games of 400; the two matches against generation 100 had 320
and 319, generation 250 against 200 had 330, and the matches against
the start 380 (generation 200) and 376 (250).

**Against gen 200.** Re-rated like the other runs (`rerate/`,
generations 1–200; no re-rate was made past generation 200, so for
generations 201–250 the matches above are the only strength measure),
Elo minus `ref1`'s Elo (−103.5 in `rerate/references.csv`) is +104 at
the start, +233 at generation 25, +303 at 50, +331 at 75, +356 at 100,
+400 at 150 and +469 at 200 (the control: +104, +152, +132 at 0, 25 and
50). The stretch means are in the resignation section's table: +240 over
generations 26–50 (from its own start +136; the control +48, the budget
arm +93), +297 over 51–75, +331 over 76–100, +384, +412 and +431 over
the next three quarters and +457 over 176–200 (from its own start +353).
The least-squares slope is +3.4 Elo per generation over generations
1–50, +1.75 over 51–100, +1.04 over 101–150 and +0.83 over 151–200:
still rising, more slowly. That is about half the gain the matches show
(+113 from generation 100 to 200 against +189), as with the other
re-rates, whose 100-simulation 20-game matches compress the scale (see
[the all-runs table](#all-post-250-runs-against-one-reference-gen-200)).
Single generations are noisy: generation 190 is +550 and 198 is +387.
Generations 23, 59, 111 and 187 were promoted to references
(`rerate/references.csv`; the other arms' re-rates promoted one
generation at most). The re-rate made at generation 100
(`rerate-to100/`, `ref1` at −101.5) gave +102 at the start, +334 at 75
and +360 at 100, and +334 over generations 76–100 against +331 here: the
same checkpoints replayed again move by a few Elo.

**At equal compute.** Per generation the recipe takes 1.51 times the
control's self-play and training time, so its generation 33 has cost
about what the control's generation 50 did. Its generation 25 is
already rated +233 against the control's generation 50 at +132, and
its generations 11–25 average +187 against the control's 26–50 at
+152. Unlike the budget arm, which was level with the control at equal
compute, the recipe is ahead at equal compute too.

**Verdict.** The recipe **beat everything else measured on 9×9 from
this start, by a wide margin**: 0.787 against the control at generation
50 (+228 Elo), 0.725 against the budget arm, and +170 over the control
pooled over generations 46–50 (1,000 games each; see
[the pooled evaluation](#final-strength-of-the-post-250-runs-pooled-generations-46-to-50)).
It was still improving at generation 250: 0.645 for generation 150 and
0.748 for generation 200 against generation 100, 0.632 for generation
250 against 200 (+94; about +19 Elo per 10 generations over generations
100–250), 0.925 for generation 250 against the start (+436), and the
re-rate still rising to generation 200, where it stops. Its parts alone gained +37
(`temperature_moves = 7`), +44 (200 games and 400 steps) and +22
(resignation) on the same pooled measure; together they gained more
than those three sum to (+103), though each part's figure is good to
about ±35 and no run was designed to test how they interact. A
hypothesis, not tested: resignation removes most of the after-pass
positions, whose targets say little, and shortens the games, so the
200 games cost about the self-play of 115 played-out ones (1.23 times
the control's self-play time), and the doubled steps then train on
positions that matter. A high replay ratio
alone is not the explanation: the steps-only arm (11.7) and the
resignation arm (12.5) gained little or nothing pooled. It costs about
1.5 times the control's compute per generation and is ahead at equal
compute as well. What it does not show: one seed, one start (a trained
6×96 network and its buffer), 250 generations. Over generations
100–200 the value head's fresh-game loss stayed flat while its buffer
loss kept falling, a slowly widening gap that has not stopped the gains
and held at 0.164 over 201–250. From scratch, with the 6×64 network,
the preset also trained a far stronger network than the earlier preset;
see [The 9×9 preset from scratch](#the-99-preset-from-scratch-250-generations).
At generation 250 this run's 6×96 network beat that one 0.610 (+78);
see [The recipe against the preset from scratch](#the-recipe-against-the-preset-from-scratch-generation-250).

### The 9×9 preset from scratch (250 generations)

Hypothesis: the recipe was measured only by continuing from a trained
6×96 network and its full buffer. From a random network the picture
could differ: resignation has to calibrate v_resign on a network whose
values mean nothing yet, early games are long whatever the settings,
and doubling the steps per generation might overfit while the buffer
is still small. So the new preset (`configs/9x9.cfg` as merged in
63b3979) was run from scratch for its full 150 generations and compared
with `9x9-main`, the earlier preset's run on the same network and seed
(the 9×9 sections near the top), then continued to generation 250, as
`9x9-main` had been, to compare the two at equal generations to the end.
The result: the new preset trains a far stronger network, at equal
generations (250 included), at equal SGD steps and with its generation
150 against `9x9-main`'s 250, and it was still improving at 250.

- *Setup.* `9x9-scratch/`: `train --config configs/9x9.cfg --run
  runs/9x9-scratch --set bot_simulations=100` (`scratch-recipe.sh`),
  with binaries (`runs/bin-scratch`, not in the repository) built from
  main at 63b3979: 6×64, 150 generations of 200 games at 200
  simulations, `temperature_moves = 7`, resignation at its defaults,
  400 SGD steps of 256 positions per generation, learning rate 0.02 and
  0.002 from generation 90, a 200,000-position buffer, seed 1, gating
  off. The 100-simulation bot was kept only so that its `elo.csv` is
  rated like `9x9-main`'s; with gating off the arena never affects
  training. It ran alone on the second Mac mini (an Apple M4 with 16
  GB), from 07:39 to 21:35 on 7 October (13 h 55 min,
  `scratch-recipe.log`, host name redacted).
- *Continued to generation 250.* `scratch-continue.sh` resumed it on the
  same machine with the same binaries (`train --resume --generations
  250`; `config.cfg` changes only in `generations`), from 08:39 to 16:38
  on 8 October (`scratch-continue.log`), at learning rate 0.002
  throughout, and then played three 400-game matches. These 100
  generations go beyond the preset's 150, as `9x9-main`'s continuation
  did. The files cover generations 1–250 (`log.csv`, `progress.txt`,
  `elo.csv`, `references.csv`, `results.csv`, `config.cfg`); the buffer
  statistics at generation 250 are `{buf,pass,side}-250.txt`, beside
  generation 150's.
- *The baseline, `9x9-main`.* The reference run of the 9×9 sections
  above (generations 1–150, with the ladder arena, `arena_opponents =
  4`) and its continuation to generation 250 (with the rating-aware
  arena, `arena_opponents = 2`), both on this Mac. Its `config.cfg` as
  resumed to 250 (`9x9/config-250.cfg`) differs from the scratch run's
  only in `generations` and the four recipe keys (100 games, 200 steps,
  `temperature_moves = 20`, no resignation), apart from keys added to
  the code since, at their defaults. Both runs have the same network
  shape, seed and learning-rate schedule, and their generation-0
  checkpoints are byte-identical: they start from the same random
  network. `9x9/log-250.csv` is its `log.csv` over generations 1–250
  (the first 150 rows are `9x9/log.csv`).
- *Equal SGD steps.* With 400 steps per generation against 200, the
  scratch run's generation g has taken as many SGD steps as
  `9x9-main`'s generation 2g: 30,000 at generation 75 against 150, and
  50,000 at 125 against 250. The learning-rate histories differ at
  equal steps, since both runs drop the rate at generation 90: at 30,000
  steps the scratch run has taken all of them at 0.02, `9x9-main` 17,800
  at 0.02 and 12,200 at 0.002; at 50,000, 35,600 and 14,400 against
  17,800 and 32,200.

**Matches** (`scratch-eval.sh` and its log, after training, and the
three at generation 200 and 250 from `scratch-continue.sh` and its log,
with the same binaries; 400 games, 200 simulations for both
sides, komi 7.5, colors alternating, `arena`'s default opening
temperature; files in `9x9-scratch/`):

| match | score | 95% interval | Elo (interval) |
|---|---|---|---|
| equal generations: gen 50 vs `9x9-main` gen 50 (`g50-vs-main-g50.txt`) | 358/400 = 0.895 | 0.861-0.921 | +372 (+317 to +427) |
| equal generations: gen 100 vs gen 100 (`g100-vs-main-g100.txt`) | 358/400 = 0.895 | 0.861-0.921 | +372 (+317 to +427) |
| equal generations: gen 150 vs gen 150 (`g150-vs-main-g150.txt`) | 363/400 = 0.907 | 0.875-0.932 | +397 (+338 to +455) |
| equal SGD steps: gen 75 vs gen 150 (`g75-vs-main-g150-equal-steps.txt`) | 238/400 = 0.595 | 0.546-0.642 | +67 (+32 to +101) |
| equal SGD steps: gen 125 vs gen 250 (`g125-vs-main-g250-equal-steps.txt`) | 318/400 = 0.795 | 0.753-0.832 | +235 (+194 to +278) |
| against `9x9-main`'s last: gen 150 vs gen 250 (`g150-vs-main-g250.txt`) | 341/400 = 0.853 | 0.814-0.884 | +305 (+256 to +353) |
| equal generations: gen 250 vs gen 250 (`g250-vs-main-g250.txt`) | 366/400 = 0.915 | 0.884-0.939 | +413 (+353 to +475) |
| own progress: gen 200 vs gen 150 (`g200-vs-g150.txt`) | 263/400 = 0.657 | 0.610-0.702 | +113 (+78 to +149) |
| own progress: gen 250 vs gen 150 (`g250-vs-g150.txt`) | 291/400 = 0.728 | 0.682-0.769 | +171 (+133 to +209) |

Every interval excludes 0.5. 371 to 398 of each 400 games were distinct.
At equal generations the new preset wins about nine games in ten at
every point measured, 250 included; scores this far from 0.5 resolve
gaps poorly, so read the +370 to +415 as "very large", not as exact. At
equal steps the gap grows with training, from +67 at 30,000 steps to
+235 at 50,000, and the scratch run's generation 150 beats `9x9-main`'s
last, which trained 100 generations longer than it (though 10,000 fewer
SGD steps: 50,000 against 60,000), 0.853. Past generation 150 the
scratch run kept improving, more slowly: +113 by generation 200 and +171
by 250 against its own generation 150, so about +58 from 200 to 250 by
difference. The two routes from generation 150 to 250 disagree in size,
as expected near 0.9: +171 directly, but only +108 by the matches
against `9x9-main`'s generation 250 (+305 for generation 150, +413 for
250).

**The Elo curves.** Both `elo.csv` files come from the rating-aware
arena with `arena_opponents = 2`, the 100-simulation bot (75/80 against
generation 0 in both, the same network) and generation 0 = 0: the
scratch run's live one (`9x9-scratch/elo.csv`, the refit at generation
250, which moved the values up to 150 by at most 3) and `9x9-main`'s
re-rate over generations 1–250 (`9x9/rerate-250/elo.csv`). Their
references differ (the scratch run promoted generations 11, 14, 21, 25,
32, 43, 67, 102, 129, 161, 221 and 243, `9x9-scratch/references.csv`;
`9x9-main`'s re-rate 23, 30,
43, 53, 78, 95, 121, 170 and 228), and one is live while the other is a
replay, so the two scales are only roughly comparable:

| generation | scratch (new preset) | `9x9-main` (earlier preset) | difference | 400-game match |
|---|---|---|---|---|
| 25 | 1200.3 | 641.7 | +559 | |
| 50 | 1599.0 | 1132.5 | +467 | +372 |
| 75 | 1865.6 | 1393.0 | +473 | |
| 100 | 1898.0 | 1532.3 | +366 | +372 |
| 125 | 2032.6 | 1716.4 | +316 | |
| 150 | 2127.0 | 1728.3 | +399 | +397 |
| 175 | 2111.8 | 1678.7 | +433 | |
| 200 | 2148.4 | 1758.1 | +390 | |
| 225 | 2224.8 | 1865.3 | +360 | |
| 250 | 2216.5 | 1799.5 | +417 | +413 |
| scratch 75 vs main 150 (equal steps) | 1865.6 | 1728.3 | +137 | +67 |
| scratch 125 vs main 250 (equal steps) | 2032.6 | 1799.5 | +233 | +235 |

The curves are level for the first ten generations (344.6 against
436.2 at generation 10) and the scratch run is ahead from generation 11
to the end, by +236 to +522 over generations 151–250 (a mean of +391);
its peak is 2261.4 at generation 246. Both curves rise slowly after
150: +90 for the scratch run from 150 to 250, about half the +171 its
own match shows, as with the re-rates above, and +71 for `9x9-main`. The
differences agree with the matches in sign and roughly in size.

- *Game length* (`log.csv`). A mean of **63.9 plies** per game over
  generations 1–150, counting the 10% played out: 99.4 over generations
  1–10 (76.9 at generation 1, 108.0 at 5), then 90.2 over 1–25, 65.4
  over 26–50, 59.0, 63.1 and 56.9 over the next three quarters and 49.1
  over 126–150 (59.8 at 150), and **46.4** over 151–250 (34.9 to 57.6
  per generation; 44.0 at 250). `9x9-main` averaged 97.8 over
  generations 1–10, 98.5 over 1–150 (93.0 at 150) and 90.4 over 151–250
  (85.9 at 250). The early games are as long as `9x9-main`'s although
  from generation 6 on almost all of them were resigned (below): the
  young networks become sure of the result only near the end. The recipe
  arm, from a trained start, averaged 50.8 over its first 100
  generations.
- *Self-play positions.* 200 games of 63.9 plies are about 12,800
  positions per generation, against `9x9-main`'s 9,850 (100 games of
  98.5): 1.30 times as many per generation, and 1.92 million against
  1.48 million over 150 generations. At equal SGD steps the scratch run
  made fewer: 1.07 million by generation 75 against 1.48 million by
  `9x9-main`'s 150 (0.73), and 1.67 million by 125 against 2.38 million
  by 250 (0.70). At equal generations 250 it made 2.85 million against
  2.38 million (1.20 times; about 9,280 per generation over 151–250
  against `9x9-main`'s 9,040). Each position costs one 200-simulation
  search in both runs, so at equal steps the new preset used about 70%
  of the old one's self-play.
- *Replay ratio.* A mean of **8.36** over generations 1–150 (4.74 to
  13.02 per generation): 5.2 over generations 1–10, while the games were
  long, then 7.9, 8.8, 8.2 and 9.0 over the quarters from 26 to 125 and
  10.5 over 126–150; then **11.13** over 151–250 (8.89 to 14.65).
  `9x9-main`'s was 5.22 over generations 1–150 (4.84 to 5.50 by
  quarter) and 5.67 over 151–250.
- *Resignation from a random network.* Generation 1's games were
  played by the random generation 0 at the initial v_resign of −0.90.
  Even so, 30% of the games with resignation enabled were resigned, and
  over the first 20 played-out games 10% of the winners would have
  resigned at that threshold (`resign_false_positive` 0.100, the run's
  highest). Calibration moved v_resign to −0.91 at generation 2 (false
  positives 0.023), and the share of resigned games rose with the
  network: 0.52, 0.64, 0.77 and 0.87 in generations 2–5, and 0.92 to
  1.00 from generation 6 on (96.8% over generations 1–150, 98.8% over
  151–250). v_resign stayed at
  −0.86 to −0.87 over generations 6–25, fell to −0.92 to −0.95 over
  64–89 (the last generations at learning rate 0.02), and was −0.80 to
  −0.89 after the drop, and −0.82 to −0.90 over 151–250. The
  false-positive rate averaged 0.047 over generations 1–150 (0.048 from
  generation 7 on), at most 0.058 after generation 1, and above 0.050 in
  29 later generations, since v_resign moves in steps of 0.01; over
  151–250 it averaged 0.048 (at most 0.058, above 0.050 in 20). In the
  generation-150 buffer, 3,649 of 4,112 games were resigned, 2,618 of
  them by Black (`pass.txt`); in the generation-250 buffer 3,655 of
  4,128, 2,681 by Black (`pass-250.txt`).
- *What the buffer holds* (`buf.txt`, `pass.txt`, `side.txt` from
  `state.bin`; the scratch run's at generation 150, and at 250 the
  `-250.txt` files; `9x9-main`'s at generation 250 is
  `9x9/{buf,pass,side}-250.txt`, the buffer of findings 1 and 2 of
  [Next experiments](#next-experiments); the recipe's at generation 200):

  | | `9x9-main`, gen 250 | scratch, gen 150 | scratch, gen 250 | recipe, gen 200 |
  |---|---|---|---|---|
  | games in the 200,000-position buffer | 2,262 | 4,112 | 4,128 | 4,569 |
  | positions at or after the first pass | 35.4% | 7.8% | 8.2% | 11.0% |
  | the same, played-out games only | 35.4% | 26.6% | 27.2% | 32.7% |
  | opening positions (plies 0–19) | 22% | 36% | 38% | 39% |
  | middle game (ply 20 to the first pass) | 43% | 56% | 54% | 50% |
  | mean target entropy (nats) | 1.990 | 1.768 | 1.689 | 1.617 |
  | Black's share of the games' wins | 0.230 | 0.284 | 0.274 | 0.290 |
  | middle game, eventual loser as Black: entropy, moves visited | 2.684, 41.9 | 2.119, 41.0 | 2.112, 42.8 | 2.007, 42.8 |

  The new preset's buffer is the recipe's in kind: few after-pass
  positions, over half middle game, sharper targets. Its losing side
  still spreads its search over about 41 to 43 moves (finding 1), with
  lower entropy. From generation 150 to 250 the buffer changed little;
  its targets became a little sharper (1.768 to 1.689 nats).
- *Training curves.* Policy loss 4.178 at generation 1, 2.902 at 50,
  2.454 at 100, 2.284 at 150, 2.225 at 200 and 2.191 at 250
  (`9x9-main`: 4.188, 3.135, 2.677, 2.471, 2.420 and 2.403); the
  targets' lower entropy lowers that loss's floor too. Value loss 0.611
  at generation 1, 0.575 at 50, 0.621 at 100, 0.695 at 150, 0.649 at
  200 and 0.644 at 250 (`9x9-main` 0.582, 0.556, 0.392, 0.417, 0.411
  and 0.373): it was 0.52 to 0.55 over generations 84–91, rose after
  the learning-rate drop (see the validation loss below) and drifted
  down after 150. Black's self-play win rate was 0.361 over generations
  1–150 and 0.265 over 151–250, against `9x9-main`'s 0.355 and 0.251.
  These are on different data and are not a strength measure.
- *The validation loss* (`val_*` and `seen_*`; generation 1 is left
  out: its `val` row measures the random generation 0 and its `seen`
  row is empty, the buffer being empty):

  | generations | value, fresh games | value, buffer | policy, fresh games | policy, buffer |
  |---|---|---|---|---|
  | 2–10 | 1.150 | 0.520 | 3.957 | 4.008 |
  | 11–25 | 0.932 | 0.667 | 3.381 | 3.441 |
  | 26–50 | 0.869 | 0.602 | 2.994 | 3.146 |
  | 51–75 | 0.851 | 0.581 | 2.573 | 2.693 |
  | 76–89 | 0.823 | 0.563 | 2.493 | 2.538 |
  | 90–100 | 0.821 | 0.550 | 2.400 | 2.466 |
  | 101–125 | 0.781 | 0.670 | 2.309 | 2.343 |
  | 126–150 | 0.786 | 0.685 | 2.259 | 2.270 |
  | 151–175 | 0.749 | 0.673 | 2.221 | 2.228 |
  | 176–200 | 0.755 | 0.654 | 2.184 | 2.203 |
  | 201–225 | 0.746 | 0.649 | 2.167 | 2.170 |
  | 226–250 | 0.746 | 0.646 | 2.147 | 2.152 |

  The policy terms fall together throughout, the fresh games below the
  buffer: learning. At learning rate 0.02 the value head fits the
  buffer much better than new games, a gap of about 0.27 from
  generation 11 to 100 that does not widen while the fresh games' value
  loss keeps falling (0.932 to 0.821): a large gap, but not the widening
  one that would mark overfitting. After the drop to 0.002 the buffer's
  value loss rose, from about 0.50 at generations 91–93 to 0.68 by
  generation 112, the training loss with it, and the gap narrowed to
  about 0.10 (0.111 and 0.102 over the last two quarters) while the
  fresh games' loss went on falling slightly. Why the buffer's value
  loss rose after the drop was not investigated. Over generations
  151–250 the gap stayed at 0.08 to 0.10 (0.076, 0.101, 0.098 and 0.100
  by quarter), with the fresh games' value loss at about 0.75 and the
  buffer's falling slowly (0.673 to 0.646), and the policy terms went on
  falling together (fresh 2.221 to 2.147). Unlike the recipe's
  continuation, whose gap widened slowly over its generations 101–200
  at a similar replay ratio, this run's last 150 generations show no
  widening.
- *Cost* (`log.csv`; the machines differ, so this is indicative only).
  A mean of 5.6 minutes per generation over generations 1–150 (4.5 to
  7.2): self-play 156.4 s, training 127.3 s (with the validation pass)
  and the arena 50.2 s. Over 151–250, with shorter games, 4.8 minutes
  (4.3 to 5.7): 116.1 s, 124.2 s and 47.0 s, 8.0 h for the 100
  generations.
  `9x9-main`, on this Mac, took 4.1 minutes over generations 1–150:
  120.9 s, 61.9 s and 61.3 s (its arena then played four ladder
  opponents and generation 0). Per unit of work the two machines ran at
  about the same speed: 12.2 ms of self-play per position against 12.3,
  and 318 ms of training per SGD step (with the validation pass)
  against 309. By the logs' times, the scratch run reached generation
  75 in 7.4 h and 125 in 11.9 h, where `9x9-main` reached 150 in 10.2 h
  and 250 in 16.3 h: at equal steps it took about 73% of the time, and
  its generation 150 (13.9 h) beat `9x9-main`'s generation 250 (16.3 h).
  `9x9-main` took 3.7 minutes per generation over 151–250 (115.1 s,
  61.0 s and 44.3 s); the scratch run's 250 generations took 21.9 h of
  generation time against `9x9-main`'s 16.3 h.

**Verdict.** The new 9×9 preset **trains a far stronger network from
scratch than the earlier preset**: about 0.9 against `9x9-main` at
equal generations (+372 to +413 Elo, 0.915 at generation 250), 0.595
(+67) and 0.795 (+235) at equal SGD steps with about 70% of the
self-play positions, and 0.853 (+305) for its generation 150 against
`9x9-main`'s generation 250, which had trained 100 generations longer.
Continued past the preset's 150 generations, it was still improving at
250 (0.728 against its generation 150, +171), more slowly than before.
Resignation calibrated itself from a random network within a few
generations, and the gap between the value head's fresh-game and
buffer losses did not widen late in the run, through generation 250.
Caveats: one seed and one board size; 400-game matches at 200
simulations, whose scores near 0.9 resolve the equal-generation gaps
poorly; the equal-steps gap grows with training (+67 at generation 75
against 150, +235 at 125 against 250), and at equal steps the two runs
had different learning-rate histories (the earlier preset drops the
rate at 17,800 steps, the new one at 35,600), so the steps comparison
is not a clean one; the wall times come from different machines; and
the eye check was not run on this network.

### The recipe against the preset from scratch (generation 250)

Question: the same four settings were run two ways to generation 250,
which network ends stronger? The recipe arm (`9x9-recipe/`) is a 6×96
network that started from generation 50 of `9x9-wide96`, itself
`9x9-main`'s generation 250 widened, so it had 300 generations of
training on the earlier settings before its 250 on the recipe's. The
preset from scratch (`9x9-scratch/`) is a 6×64 network with 250
generations, all on the new settings. The result: the recipe's network
is modestly stronger. This compares two training paths; it is not a
controlled test of network size, history or settings.

- *Setup* (`9x9-versus/`). `versus.sh` ran on this Mac with
  `runs/bin-next` once both runs had finished, copying the scratch run
  from the second Mac mini (host names redacted in the copy here),
  from 17:07 to 17:48 on 8 October (`versus.log`). The arena plays only
  networks of the same shape, so `grow --filters 96` widened the
  scratch run's generations 150 and 250 to 6×96 (473,277 → 1,032,349
  parameters; `value_hidden` stays 64, as in the recipe's network).
  Widening preserves the function (each new channel copies an old one
  and the weights reading it are split among the copies; see the main
  README's `grow`), and `grow` checked the outputs on 8 random
  positions: a maximum difference of 1.5e-05 and 1.2e-05, float
  rounding (`grow-g150.txt`, `grow-g250.txt`). The widened networks play
  as the 6×64 ones do. The
  matches are as above: 400 games, 200 simulations for both sides, komi
  7.5, colors alternating, `arena`'s default opening temperature.

| match | score | 95% interval | Elo (interval) |
|---|---|---|---|
| recipe gen 250 vs scratch gen 250 (`recipe-g250-vs-scratch-g250.txt`) | 244/400 = 0.610 | 0.561-0.657 | +78 (+43 to +113) |
| recipe gen 200 vs scratch gen 250 (`recipe-g200-vs-scratch-g250.txt`) | 246/400 = 0.615 | 0.566-0.661 | +81 (+46 to +116) |
| recipe gen 250 vs scratch gen 150 (`recipe-g250-vs-scratch-g150.txt`) | 311/400 = 0.777 | 0.734-0.816 | +217 (+176 to +259) |

Every interval excludes 0.5. 360, 333 and 366 of the 400 games were
distinct.

- *At generation 250 the recipe's network is ahead*, 0.610 (+78). It is
  about twice the size (1.03 million parameters against 0.47 million)
  and had the 300 earlier generations, so the margin is modest given
  those advantages; which of them it comes from, these matches cannot
  say.
- *Recipe generations 200 and 250 score the same* against the scratch
  run's generation 250 (0.615 and 0.610, a difference well inside the
  noise), although generation 250 beat 200 directly 0.632 (+94). The
  three results are not transitive: either the noise of three 400-game
  matches (about ±35 Elo each), or generation 250's gain over 200 does
  not carry over to this opponent; the matches cannot tell which. Only
  333 of the generation-200 match's games were distinct.
- *Both runs were still improving.* The scratch run gained +171 from
  generation 150 to 250 against its own generation 150, and against the
  recipe's generation 250 it went from −217 at 150 to −78 at 250, about
  +139. The recipe gained +94 from 200 to 250 and +189 from 100 to 200.
  Per generation that is about the same, +1.7 Elo for the scratch run
  over its last 100 generations and +1.9 for the recipe over its last 50,
  though the scratch run's own gain slowed (+113 by 200, about +58 more
  by 250).
- *Cost.* Over their last generations the recipe took 8.1 minutes per
  generation on this Mac and the scratch run 4.8 on the second Mac mini
  (indicative only: different machines), and the recipe's network also
  carries its 300 earlier generations.

**Verdict.** At generation 250 the recipe's 6×96 network, larger and
with a longer history, is **modestly stronger** than the 6×64 network
the preset trains from scratch (0.610, +78), and both were still
improving at about the same rate per generation. Caveats: different
network sizes and histories, so this is a comparison of two training
paths, not a test of either factor; one seed each; 400-game matches at
200 simulations; and the recipe's generations 200 and 250 scored the
same against the scratch network, so the gap is known only to within
the matches' noise.

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

### GPU inference with `backend = metal`

Task A9 of the [Metal plan](../superpowers/plans/2026-10-09-metal-gpu-backend.md),
on the M5 Max under macOS 27.0.1, with the training job paused
(`bench.txt`, last section).

Throughput: batch-8 requests from N threads through one hub, against 18
CPU workers each evaluating its own batches of 8 with BLAS.

| network | 18 CPU workers | hub, 18 threads | hub, 36 | hub, 72 |
|---|---|---|---|---|
| 9×9 6×64 | 32,369 | 34,793 | 46,897 (1.45×) | 57,417 |
| 9×9 6×96 | 14,958 | 23,085 | 27,220 (1.82×) | 29,392 |
| 19×19 10×128 | 863 | 2,742 | 2,743 (3.2×) | 2,782 |

The hub's passes average 50, 93 and 187 positions at 18, 36 and 72
threads on 6×96. Its slots are busy 98–100% of the time. The GPU alone,
both slots kept full with passes of 256, reaches 30,400 positions/s on
6×96. The hub at 36 threads gets within 11% of that.

One real generation of the 9×9 preset at 6×96, started from a copy of
9x9-recipe generation 250, with the same seed:

| backend | selfplay_s | train_s | arena_s | wall |
|---|---|---|---|---|
| blas (18 workers) | 142.7 | 39.3 | 12.0 | 194.2 s |
| metal (36 workers) | 89.2 | 37.2 | 11.8 | 138.3 s |
| speedup | 1.60× | 1.06× | 1.02× | 1.40× |

All other `log.csv` columns came out identical (policy loss 2.23369,
value loss 0.39413, validation losses, the arena score). So did the new
checkpoint, `results.csv` and the SGF: the GPU's last-bit differences
changed no move in this generation. They were not expected to be
identical, and over more generations they will not be.

Against the design's acceptance:
- hub ≥ 25k evals/s at 36 workers: met.
- self-play ≥ 1.6×: met, just (1.60×).
- whole generation ≥ 1.4×: met, just (1.40×).
- arena ≥ 1.3×: **not met** here (1.02×). At generation 1 the arena is
  one 20-game match: 20 searches at once, split between two networks,
  too few to fill the passes. **Met after the arena change below**
  (1.43–1.51×).

**The arena, its matches played at once**
([design](../superpowers/specs/2026-10-10-arena-gpu-design.md), branch
arena-gpu, `bench.txt`, last section).
- Setup: generation 2 of a 9×9 6×96 run started from a copy of
  9x9-recipe generation 250, with copies of generations 150, 200 and 230
  as reference networks. Its arena plays four opponents (generation 1,
  generation 0, ref1, ref2), 80 games. The training job was paused for
  every measurement.
- How it works: the round's matches share one pool of games, with one
  hub per network. With metal every game runs at once (80 workers), and
  each of the five hubs keeps one pass in flight.

| code | backend | arena_s | metal's speedup |
|---|---|---|---|
| before | blas (18 workers) | 51.5 | |
| before | metal (36 workers) | 50.0 | 1.03× |
| after | blas (18 workers) | 47.0, 44.5 | |
| after | metal, 36 workers, 2 slots per hub | 59.6 | 0.75–0.79× |
| after | metal, 80 workers, 2 slots per hub | 38.1 | 1.17–1.23× |
| after | **metal, 80 workers, 1 slot per hub (the default)** | **31.2** | **1.43–1.51×** (1.65× against blas before) |

- The intermediate rows show the two levers that mattered.
  - *Every game in flight.* An arena game has only one search waiting at
    a time, so the games in flight are what fills the passes.
  - *One pass in flight per hub.* With five hubs sharing the GPU, the
    others' passes keep it busy. Each hub then takes everything waiting
    into one pass. An arena-only harness on the same wave, with the
    training job running, counted 14,300 passes instead of 20,700,
    averaging 115
    positions for the new network and ~30 for each opponent (77 and ~21
    with two slots).
- Before and after are byte-identical: checkpoint, `state.bin`,
  `results.csv`, `elo.csv`, `references.csv`, the SGFs and `log.csv`
  without its timing columns, for blas and for metal.
- The spec's remaining levers (a larger `inference_batch`, kernel fusion,
  implicit GEMM) are untried. The GPU is busy ~95% of the arena with
  passes of ~30–115 positions, so faster small passes are what would help
  next.

Correctness, all on the same machine:

- *Correctness, measured.* Positions get bitwise the same outputs alone
  and in any batch, through every kernel, the whole network and the hub.
  Tiny metal runs are byte-identical for 1, 3 and 8 workers and for
  `inference_batch` 8 and 256. Worst error against the BLAS CPU path over
  20 seeds: the network's outputs within 5.5e-6 through the GPU network
  called directly and 6.5e-6 through the hub (`maxErr`, spec ceiling
  1e-5), the validation losses within 1.1e-7 relative.
  `analyze --backend metal` prints the same digits as `--backend blas` on
  the README's example.

## Next experiments

A living to-do list. A review of the 9×9 runs (Claude, at Mark's
request) found no correctness bugs in the core (gradients, batch norm,
the symmetries of the policy and ownership targets, input encoding,
superko in the tree, terminal pass scoring, backup signs) and made four
findings. Findings 1 and 2 are from the replay buffer of `9x9-main` at
generation 250 (`runs/9x9-main/state.bin`, 199,966 positions from 2,262
games; not in the repository) and can be reproduced with the read-only
scripts `tools/side.py`, `tools/pass.py` and `tools/buf.py`, for example
`python3 tools/side.py runs/9x9-main/state.bin`.

1. **The losing side's search spreads over almost every legal move.**
   An unvisited move has Q = 0, so when the side to move is behind,
   every untried move looks better than every tried one. Middle-game
   positions (ply 20 up to the first pass), by the side to move
   (`tools/side.py`):

   | side to move | positions | moves visited | target entropy (nats) | top move's share |
   |---|---|---|---|---|
   | eventual loser, Black | 30,583 | 41.9 | 2.68 | 0.30 |
   | eventual loser, White | 12,747 | 26.3 | 1.99 | 0.42 |
   | eventual winner, Black | 12,886 | 25.9 | 1.66 | 0.52 |
   | eventual winner, White | 29,081 | 10.4 | 1.30 | 0.54 |

   Black loses 77% of these games (`tools/pass.py`), so most of Black's
   policy targets are close to uniform. This is how AlphaGo Zero's
   search works (unvisited moves at Q = 0), so it is a known cost to
   track in new runs, not a bug.
2. **35% of buffer positions come at or after the first pass**
   (`tools/buf.py`). The eventual winner passes first in 84% of games;
   of the passes made before the final two moves, the winner made 16,861
   and the loser 570 (`tools/pass.py`). The loser plays on for a mean
   31.3 plies rather than pass into a counted loss, and those positions'
   targets still have 1.97 nats of entropy. `sims200`, `games200` and
   `lr-restart` show the same pattern (36% after the first pass). With
   resignation on, 12% of the buffer is (`9x9-resign/buf.txt`); see
   [Resignation in self-play](#resignation-in-self-play).
3. **The games200 experiment is confounded.** It halved the replay ratio
   (5.8 to 2.9), so it tested fresher data, not more learning; see the
   correction in
   [More self-play games per generation](#more-self-play-games-per-generation-200-vs-100).
   All post-250 runs had the same learning budget (10,000 SGD steps,
   at 0.002 except in the lr-restart run), which fits "every lever gained
   about the same". Doubling games and steps together (replay ratio 5.8)
   gained, at twice the compute per generation; see [The learning budget](#the-learning-budget-200-games-and-400-steps-per-generation).
   Doubling the steps alone on the same games (replay ratio 11.7) did
   not; see [Steps only](#steps-only-100-games-and-400-steps-per-generation).
   Both together with `temperature_moves = 7` and resignation gained by
   far the most; see [The recipe](#the-recipe-four-changes-together-250-generations).
   As the 9×9 preset, from scratch, they beat the earlier preset by
   +235 Elo at equal SGD steps; see
   [The 9×9 preset from scratch](#the-99-preset-from-scratch-250-generations).
4. **The measurements are coarse for the effects tested.** A 100-game
   match resolves about ±70 Elo and the re-rate's noise is about ±40,
   while the effects tested so far are 30–80 Elo. The AlphaGo Zero
   arms are evaluated with 400-game matches (about ±35 Elo), and the
   post-250 runs are ranked by a pooled evaluation, generations 46–50
   against the start, 1,000 games per run (about ±31 to ±35 Elo for a
   difference between two runs; see
   [Final strength of the post-250 runs](#final-strength-of-the-post-250-runs-pooled-generations-46-to-50)). And the policy loss
   is about 2.40 (2.403 at generation 250 of `9x9-main`) against a mean
   target entropy of 1.99 nats (`tools/buf.py`), the floor for that
   cross-entropy, so only about 0.4 nats can still be learned.

**Principle.** The project stays faithful to the original AlphaGo Zero
(Silver et al., Nature 2017): only its methods are on the list. KataGo
and Leela Zero refinements are out of scope: first-play-urgency tuning,
Gumbel search, playout-cap randomization, policy-target pruning, score
and ownership heads, and Q-blended value targets. (The ownership head
above was an earlier experiment; it is off in every preset.)

The experiments, in priority order. Training arms use the same start,
seed and control as the other post-250 runs (generation 50 of
`9x9-wide96`, against `9x9-sims200`) and are evaluated with 400-game
matches (`runs/agz-experiment.sh`, not in the repository).

| # | experiment (AlphaGo Zero) | why | status |
|---|---|---|---|
| 1 | Resignation: `v_resign` set so that wrong resignations stay under 5%, with 10% of games played out with resignation off | finding 2; the review estimates it saves about 30% of self-play time | done (`resignation`; state.bin version 5): beat the control 0.595 at generation 50 (+67 Elo, interval excludes 0.5) with games 53% shorter (41.4 plies against 88.3) at a false-positive rate of 0.041; confounded by a replay ratio of 12.5 against 5.8 and a buffer spanning about 48 generations against 23; pooled over generations 46–50 only +22 Elo over the control, inside the noise; see [Resignation in self-play](#resignation-in-self-play) |
| 2 | A random symmetry for each network evaluation in search | AGZ's search; screen with arena games only | merged (`mcts_random_symmetry`, `arena_random_symmetry`, and arena's one-sided `--a-random-symmetry` / `--b-random-symmetry`; see "Random symmetry in search" in the main README); screened: generation 250 against itself with symmetry on side A only, 400 games, scored 0.557 at 200 simulations and 0.585 at 800 (about +40 and +60 Elo, both intervals exclude 0.5) at play time; see [Arena screens](#arena-screens-of-alphago-zero-search-settings). Self-play arm done (`9x9-sym`, the `sims200` control plus `mcts_random_symmetry = true`): no measurable training gain, level with the control at generation 50 (0.495, interval 0.446–0.544) and on the re-rate against gen 200; it costs nothing, but the play-time gain did not become a stronger network (pooled over generations 46–50: +22 Elo, inside the noise); see [Random symmetry in self-play](#random-symmetry-in-self-play) |
| 3 | `temperature_moves = 7`: AGZ's 30 moves of 361 points is about 8%, about 7 moves on 9×9 (the preset then used 20) | config only | done: beat the control 0.605 at generation 50 (+74 Elo, interval excludes 0.5) with the same data volume and replay ratio, confirmed at +37 pooled over generations 46–50; since adopted in the 9×9 preset as part of the recipe; see [Seven opening moves at temperature](#seven-opening-moves-at-temperature-temperature_moves--7) |
| 4 | Learning budget: more `train_steps`, or steps scaled with games so the replay ratio stays constant | finding 3 | done. Steps scaled with games (`9x9-budget-g200-s400`: 200 games and 400 steps per generation, replay ratio 5.8 as in the control): beat the control 0.640 at generation 25 and 0.593 at 50, and +44 Elo over it pooled over generations 46–50 (outside the ±35 noise), at twice the compute per generation; see [The learning budget](#the-learning-budget-200-games-and-400-steps-per-generation). Steps only (`9x9-steps400`: 100 games and 400 steps, replay ratio 11.7): no gain, +5 Elo pooled, although its generation 50 beat the control's 0.573; see [Steps only](#steps-only-100-games-and-400-steps-per-generation). With the games run (games only: +13 pooled), neither more steps on the same data nor more data at the same steps helped; both together did. Both together with items 1 and 3: see row 9 |
| 5 | More simulations (AGZ used 1,600) | only after the evaluation additions below, since the gain will be small for its cost | not started at 1,600. The evaluation it waited for now exists: the pooled evaluation (1,000 games over five checkpoints, about ±31 to ±35 Elo for a difference). On it, the earlier 400-simulation run (`9x9-sims/sims400`) is −18 Elo against the 200-simulation control, no gain for 1.66 times the time; see [Final strength of the post-250 runs](#final-strength-of-the-post-250-runs-pooled-generations-46-to-50) |
| 6 | Arena-only tuning of `c_puct`, and `mcts_batch` 1 against 8 (AGZ evaluated leaves in batches of 8): generation 250 against itself, 400 games per setting | cheap: arena games only, no training | done, no change: `c_puct` 0.75 to 1.25 within noise of the preset 1.5, 2.0 and 3.0 clearly worse (−45, −102 Elo); batch 1 +30 Elo, not significant; see [Arena screens](#arena-screens-of-alphago-zero-search-settings) |
| 7 | AGZ gating (`gating = true`, 55% over 400 games) | expensive; only if regressions appear | not started |
| 8 | `value_hidden = 256`, as in AGZ | architecture change, low priority | not started |
| 9 | The recipe: items 1, 3 and 4 together (`temperature_moves = 7`, resignation, 200 games and 400 steps) | the three single changes that beat the control | done, 250 generations from the same start (`9x9-recipe`): beat the control 0.787 at generation 50 (+228 Elo) and the budget arm 0.725, +170 over the control pooled over generations 46–50, and still gaining at 250: 0.670 for generation 100 against 50, 0.645 for 150 and 0.748 for 200 against 100, 0.632 for 250 against 200 (about +19 Elo per 10 generations over 100–250), 0.925 for 250 against the start; self-play time about the control's, training twice; the value head's fresh-game loss stays flat while its buffer loss falls, a gap widening slowly (0.12 to 0.16 over generations 101–200), then holding at 0.164 over 201–250; no re-rate past 200; see [The recipe](#the-recipe-four-changes-together-250-generations). Adopted as the 9×9 preset (`configs/9x9.cfg`, commit 63b3979); see row 10. At generation 250 its 6×96 network beat the 6×64 one from scratch 0.610 (+78); see [The recipe against the preset from scratch](#the-recipe-against-the-preset-from-scratch-generation-250) |
| 10 | The new 9×9 preset from scratch | the recipe was validated only from a trained start | done (`9x9-scratch`, the preset's 150 generations continued to 250, 6×64, seed 1, against `9x9-main`, the earlier preset's run with the same network, seed and generation 0): 0.895, 0.895, 0.907 and 0.915 at equal generations 50, 100, 150 and 250 (+372 to +413 Elo); 0.595 (+67) and 0.795 (+235) at equal SGD steps (generation 75 against 150, 125 against 250) with about 70% of the self-play positions; 0.853 (+305) for generation 150 against `9x9-main`'s 250; still improving at 250 (0.657 for 200 and 0.728 for 250 against 150); resignation calibrated itself from a random network within about five generations; one seed; see [The 9×9 preset from scratch](#the-99-preset-from-scratch-250-generations). Against the recipe's 6×96 network at generation 250 (widened to 6×96 for the arena, which preserves its function) it scored 0.390 (−78); see [The recipe against the preset from scratch](#the-recipe-against-the-preset-from-scratch-generation-250) |

Evaluation only (the algorithm unchanged): a validation loss on
held-out fresh games each generation is merged (`validation_loss`, on by
default: `val_*` and `seen_*` columns in `log.csv`; see "The validation
loss" in the main README), first measured in the steps-only and recipe
arms (see their sections; in the recipe the fresh games' value loss
stays flat while the buffer's falls, a gap that widened from about 0.1
to 0.16 by generation 200 and held there to 250, and from scratch it
narrowed to about 0.1 after the learning-rate drop and stayed there to
250). An outside
reference opponent over GTP has its tools merged (the `gtp` engine and
`tools/gtp_match.py`; see "Playing other programs via GTP" in the main
README), with no match against GNU Go played yet. Not started: 400-game
matches at temperature 0 (AGZ's evaluation size).

## What did not work

1. **The 9×9 eye check failed.** The data is above. The final 9×9
   network misjudges a basic two-eye shape, so the 9×9 criterion as a
   whole is not met. (Later evidence suggests the network reads the
   shape but judges another move bigger, and that the check is not a
   clean life-and-death test; see
   [An ownership head](#an-ownership-head-auxiliary-target).)
2. **The 9×9 Elo curve does not show the progress that happened.** The
   slope criterion is met on its sign only (+0.41 Elo per generation).
   - The refit is flat from about generation 50 on.
   - The rating at the time falls.
   - The 100-game head-to-heads show steady but slowing improvement
     throughout: gen 150 beats gen 50 98/100.
   - The rating method saturates once every generation sweeps the
     generation-0 anchor. A fixed panel of stronger reference checkpoints
     would fix that; the code was not changed for this report.
   - The code has since been changed to choose arena opponents by
     rating (a fixed bot plus promoted reference checkpoints), and
     `train --rerate` re-measured this run with it: the re-rated curve
     rises to the end, +2.83 Elo per generation over generations
     101–150 and +634 from generation 50 to 150 against the head-to-head
     +676 (`9x9/rerate/elo.csv`).
3. **The first set of reference runs was invalid: the pass bug.** The
   runs made before commits 5329b10 and 15db72a had a search blind spot.
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
- `9x9/rerate/`, `5x5/rerate/`, `5x5-seed2/rerate/`: the three runs
  re-rated with rating-aware opponents (`train --rerate --set
  arena_opponents=2`):
  - `results.csv`, every pairwise result of the replay, starting with the
    bot's calibration against generation 0 (bot k is player −k);
  - `elo.csv`, the final refit, one row per generation, in the format of
    a run's `elo.csv`;
  - `references.csv`, the bot and the promoted generations with their
    final ratings;
  - `log.csv`, one row per generation: its opponents, its scores against
    g − 1 and the strongest reference, whether it was promoted, and its
    rating at the time;
  - `progress.txt`, the per-generation lines `train --rerate` printed.
- `9x9/rerate-250/`: the same files for the continued 6-block run re-rated over
  generations 1–250 (`elo.csv`, `references.csv`, `log.csv`, `results.csv`,
  `progress.txt`).
- `9x9-grow10/`: the grown 10-block run: `log.csv`, `elo.csv`, `results.csv`,
  `references.csv`, `init.txt` (paths made repo-relative), `progress.txt`, and
  the evaluations `g50-vs-old250.txt`, `g25-vs-old250.txt`,
  `g50-vs-old228.txt`, `analyze-eye.txt`.
- `9x9-wide96/`: the 6×96 widening run: `log.csv`, `elo.csv`,
  `results.csv`, `references.csv`, `init.txt`, `progress.txt`, the
  matches `w25-vs-old250.txt`, `w50-vs-old250.txt`,
  `wide50-vs-deep50.txt`, and `analyze-eye.txt`.
- `9x9-sims/`: the simulations experiment: `sims400/` and `sims200/`
  (each with `log.csv`, `elo.csv`, `results.csv`, `references.csv`,
  `init.txt`, `progress.txt`), the four matches `s*-vs-*.txt`, the eye
  analyses `eye-sims{200,400}-s{200,800}.txt`, and the commands
  `sims-experiment.sh` and `sims-experiment-resume.sh`; added later, the
  400-game control match `s200g50-vs-start-400.txt` and the 200 arm's
  buffer statistics `sims200/{buf,pass,side}.txt`.
- `9x9-ownership/`: the ownership-head run: `log.csv` (with the
  `ownership_loss` column), `elo.csv`, `results.csv`, `references.csv`,
  `init.txt` (paths made repo-relative), `progress.txt`; the commands
  `ownership.sh` and their output `ownership.log` (with the `grow`
  checks and the match lines); the matches `own25-vs-start.txt`,
  `own50-vs-start.txt`, `own25-vs-s200g25.txt`, `own50-vs-s200g50.txt`;
  the eye analyses `eye-own50-s200.txt`, `eye-own50-s800.txt`,
  `eye-own50-killed.txt`, `eye-own50-saved.txt`; and the fair eye
  comparison, `fair-{own50,own0,sims200g50,main250}-{tenuki,save}.txt`
  with its commands `fair-comparison.sh`.
- `9x9-ref200/`: the post-250 runs re-rated against generation 200 of
  `9x9-main`: one folder per run (`wide96/`, `grow10/`, `sims200/`,
  `sims400/`, `ownership/`, each with `elo.csv`, `references.csv`,
  `results.csv`, `log.csv`, `reference-networks.csv`, `progress.txt`),
  the commands `rerate-ref200.sh` and their log, and the head-to-head
  `g250-vs-g200.txt`.
- `9x9-games200/`: the 200-games-per-generation run: `log.csv`,
  `elo.csv`, `results.csv`, `references.csv`, `init.txt`, `progress.txt`,
  the matches `g25-vs-s200g25.txt`, `g50-vs-s200g50.txt`,
  `g50-vs-start.txt`, the commands `games200.sh` and their output
  `games200.log`, and `rerate/` (the re-rate against generation 200).
- `9x9-lr-restart/`: the learning-rate restart run, with the same files
  as `9x9-games200/` (commands `lr-restart.sh`, output `lr-restart.log`).
- `9x9-resign/` and `9x9-temp7/`: the two AlphaGo Zero training arms:
  `log.csv` (the resignation arm's with the `resign_*` columns),
  `elo.csv`, `results.csv`, `references.csv`, `config.cfg`, `init.txt`,
  `progress.txt`, the 400-game matches `g25-vs-control-g25.txt`,
  `g50-vs-control-g50.txt`, `g50-vs-start.txt`, the buffer statistics
  `buf.txt`, `pass.txt` and `side.txt` (from `tools/`), `rerate/` (the
  re-rate against generation 200) and the commands with their logs:
  `resign-arm.sh`, `agz-experiment2.sh` and `resign-arm.log`, or
  `agz-queue.sh`, `agz-experiment.sh` and `agz-queue.log`, and in both
  `rerate-arms.sh` and `rerate-arms.log`.
- `9x9-sym/`: the random-symmetry self-play arm, with the same files as
  `9x9-temp7/` (commands `sym-arm.sh` and `agz-experiment2.sh`, output
  `sym-arm.log`; `rerate-arms.log` ends with this arm's re-rate).
- `9x9-budget-g200-s400/`: the learning-budget arm (200 games and 400
  steps per generation), with the same files as `9x9-temp7/` (commands
  `agz-queue.sh` and `agz-experiment.sh`, whose `agz-queue.log` here
  runs to the end of the queue; `rerate-arms.log` ends with this arm's
  re-rate).
- `9x9-steps400/`: the steps-only arm (100 games and 400 steps per
  generation): `log.csv` (with the `val_*` and `seen_*` columns),
  `elo.csv`, `results.csv`, `references.csv`, `config.cfg`, `init.txt`,
  `progress.txt`, the 400-game matches `g25-vs-control-g25.txt`,
  `g50-vs-control-g50.txt`, `g50-vs-budget-g50.txt` and
  `g50-vs-start.txt`, the buffer statistics `buf.txt`, `pass.txt` and
  `side.txt`, `rerate/` (the re-rate against generation 200, with its
  `progress.txt`), and the GNU Parallel launch shared with the recipe
  arm: the script `next-arm.sh`, the job list `next-jobs.txt` and the
  commands it expanded to, `next-jobs.cmds`, the controller's output
  `next-jobs.log` and its job log `next-jobs.joblog`.
- `9x9-recipe/`: the recipe arm (`temperature_moves = 7`, resignation,
  200 games and 400 steps per generation), generations 1–250, with the
  same files as `9x9-steps400/` (the `log.csv` also with the `resign_*`
  columns; `progress.txt` with the lines of the first run and of all
  three resumes; the buffer statistics at generation 200) and the matches
  `g25-vs-control-g25.txt`, `g50-vs-control-g50.txt`,
  `g50-vs-budget-g50.txt`, `g50-vs-start.txt`, `g100-vs-g50.txt`,
  `g100-vs-start.txt`, `g150-vs-g100.txt`, `g200-vs-g100.txt`,
  `g200-vs-start.txt`, `g250-vs-g200.txt` and `g250-vs-start.txt`; the
  continuations' scripts `recipe-continue.sh` (to 200) and
  `recipe-continue2.sh` (to 250) with their logs `recipe-continue.log`
  and `recipe-continue2.log`; `rerate/`, the re-rate against
  generation 200 over generations 1–200 (none was made past 200), and
  `rerate-to100/`, the one made at generation 100. See
  [its section](#the-recipe-four-changes-together-250-generations).
- `9x9-scratch/`: the 9×9 preset from scratch, generations 1–250:
  `log.csv` (with the `resign_*`, `val_*` and `seen_*` columns),
  `elo.csv`, `results.csv`, `references.csv`, `config.cfg`,
  `progress.txt`, the command `scratch-recipe.sh` and its log
  `scratch-recipe.log`, the six 400-game matches against `9x9-main` up
  to generation 150 (`g50-vs-main-g50.txt`, `g100-vs-main-g100.txt`,
  `g150-vs-main-g150.txt`, `g75-vs-main-g150-equal-steps.txt`,
  `g125-vs-main-g250-equal-steps.txt`, `g150-vs-main-g250.txt`) with
  their script `scratch-eval.sh` and log `scratch-eval.log`; the
  continuation to 250, `scratch-continue.sh` and its log
  `scratch-continue.log`, with its matches `g200-vs-g150.txt`,
  `g250-vs-g150.txt` and `g250-vs-main-g250.txt`; and the buffer
  statistics `buf.txt`, `pass.txt` and `side.txt` at generation 150 and
  `buf-250.txt`, `pass-250.txt` and `side-250.txt` at 250.
- `9x9-versus/`: the recipe against the preset from scratch at
  generation 250: the script `versus.sh` (host names redacted) and its
  log `versus.log`, the `grow` outputs `grow-g150.txt` and
  `grow-g250.txt` (the scratch networks widened to 6×96; the widened
  checkpoints are not in the repository), and the three 400-game
  matches `recipe-g250-vs-scratch-g250.txt`,
  `recipe-g200-vs-scratch-g250.txt` and
  `recipe-g250-vs-scratch-g150.txt`. See
  [its section](#the-recipe-against-the-preset-from-scratch-generation-250).
- `9x9/log-250.csv`, `9x9/config-250.cfg` and `9x9/{buf,pass,side}-250.txt`:
  `9x9-main` continued to generation 250: its `log.csv` over generations
  1–250 (the first 150 rows are `9x9/log.csv`), its `config.cfg` as
  resumed, and the buffer statistics of its generation-250 `state.bin`
  (the buffer of findings 1 and 2), for the from-scratch comparison.
- `9x9-final-eval/`: the pooled evaluation, generations 46–50 of ten
  runs (and 96–100 of the recipe) against the shared start, 200 games
  each: one `arena` output per checkpoint, `<run>-g<generation>.txt`,
  and the scripts `final-eval.sh` and `final-eval-recipe.sh` with their
  logs.
- `5x5/rerate-opp4/`: `elo.csv`, `references.csv` and `log.csv` of a
  re-rate of 5×5 seed 1 with `--set arena_opponents=4`, made on a copy of
  the run, for the scale caveat above.
- `19x19/`: `log.csv`, `timing.txt`, `analyze-gen2.txt`.
- `passbug/`: the pre-fix runs and the case study.
- `elo.svg`, `elo-5x5.svg`, `elo-9x9.svg`, `elo-passbug.svg`: Elo charts
  drawn from the CSV files above by `tools/plot_elo.py` (the commands are in
  the main `README.md`).
