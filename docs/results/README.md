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

The reference runs used the ladder (`arena_opponents = 4` plus generation
0; 2 plus generation 0 on 19×19). The presets now choose opponents by
rating (`arena_opponents = 2`, `bot_simulations = 100`,
`calibration_games = 80`, `promote_threshold = 0.85`); the re-rated
results below use those values.

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

  Everything else is the 9×9 preset (100 games of 200 simulations per
  generation, `arena_opponents = 2`, gating off). `init.txt` is the
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

**Next levers, all untested hypotheses.** None of these was run.

- More simulations per move, for self-play and training targets.
  Tested (400 vs 200, 50 generations): no measurable gain per
  generation and worse per unit of compute; see
  [the section](#more-simulations-in-self-play-400-vs-200).
- Learning-rate changes (the grown run held 0.002 constant).
- More self-play games per generation (100 now, with a replay ratio of
  about 5.8).
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
test that explanation directly.

Not tested here, and no result is claimed for any of them: playout cap
randomization (full search on only some moves), larger simulation gaps
than 2×, runs longer than 50 generations, changes to the value targets,
and more than one seed per arm (each number above is one run).

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
   - The code has since been changed to choose arena opponents by
     rating (a fixed bot plus promoted reference checkpoints), and
     `train --rerate` re-measured this run with it: the re-rated curve
     rises to the end, +2.83 Elo per generation over generations
     101–150 and +634 from generation 50 to 150 against the head-to-head
     +676 (`9x9/rerate/elo.csv`).
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
- `5x5/rerate-opp4/`: `elo.csv`, `references.csv` and `log.csv` of a
  re-rate of 5×5 seed 1 with `--set arena_opponents=4`, made on a copy of
  the run, for the scale caveat above.
- `19x19/`: `log.csv`, `timing.txt`, `analyze-gen2.txt`.
- `passbug/`: the pre-fix runs and the case study.
- `elo.svg`, `elo-5x5.svg`, `elo-9x9.svg`, `elo-passbug.svg`: Elo charts
  drawn from the CSV files above by `tools/plot_elo.py` (the commands are in
  the main `README.md`).
