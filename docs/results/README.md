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
results below use those values. (The presets have since dropped the bot,
`bot_simulations =`, in favour of optional reference networks; see the
main README. The results here were measured with it.)

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
  fresher data, not more learning; 200 games with 400 steps is queued.
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

(`ref1` scored 29/80 against gen 0 of wide96 and grow10 and 21/80 against
the other runs, which share a start; the table is `elo.csv` minus
`references.csv` in each run's folder: `9x9-ref200/<run>/` for the first
five rows, `9x9-games200/rerate/` and `9x9-lr-restart/rerate/` for the
rows added later, which were re-rated the same way after their runs.)

Reading it, honestly:

- **The noise is about ±40 Elo.** The same network (wide96 gen 50) is
  +131 as the last generation of the wide96 run but +85 to +105 as the
  start of the sims200, sims400 and ownership runs. Differences between
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
- **They all had the same learning budget.** Every run in the table
  took 200 SGD steps of 256 positions per generation, 10,000 steps in
  50 generations, at a learning rate of 0.002 (the lr-restart run took
  the same number of steps, most of them at higher rates). That is
  consistent with every lever gaining about the same, though it does not
  show that the budget is the limit. A run that doubles both games and steps is in progress (see
  the games section below).

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
training steps per generation (`runs/9x9-budget-g200-s400`), is queued
and has no result yet (item 4 in [Next experiments](#next-experiments)).

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
   `lr-restart` show the same pattern (36% after the first pass).
3. **The games200 experiment is confounded.** It halved the replay ratio
   (5.8 to 2.9), so it tested fresher data, not more learning; see the
   correction in
   [More self-play games per generation](#more-self-play-games-per-generation-200-vs-100).
   All post-250 runs had the same learning budget (10,000 SGD steps,
   at 0.002 except in the lr-restart run), which fits "every lever gained
   about the same".
4. **The measurements are coarse for the effects tested.** A 100-game
   match resolves about ±70 Elo and the re-rate's noise is about ±40,
   while the effects tested so far are 30–80 Elo. And the policy loss
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
| 1 | Resignation: `v_resign` set so that wrong resignations stay under 5%, with 10% of games played out with resignation off | finding 2; the review estimates it saves about 30% of self-play time | merged (`resignation`; state.bin version 5); running as `runs/9x9-resign`: the `sims200` control's start and settings plus `resignation = true`, 400-game evaluations |
| 2 | A random symmetry for each network evaluation in search | AGZ's search; screen with arena games only | merged (`mcts_random_symmetry`, `arena_random_symmetry`, and arena's one-sided `--a-random-symmetry` / `--b-random-symmetry`; see "Random symmetry in search" in the main README); screening: generation 250 against itself with symmetry on side A only, 400 games at 200 and at 800 simulations (`runs/sym-screen.sh`) |
| 3 | `temperature_moves = 7`: AGZ's 30 moves of 361 points is about 8%, about 7 moves on 9×9 (the preset uses 20) | config only | running (`runs/9x9-temp7`) |
| 4 | Learning budget: more `train_steps`, or steps scaled with games so the replay ratio stays constant | finding 3 | 200 games and 400 steps per generation (`runs/9x9-budget-g200-s400`) queued after `temp7` |
| 5 | More simulations (AGZ used 1,600) | only after the evaluation additions below, since the gain will be small for its cost | not started |
| 6 | Arena-only tuning of `c_puct`, and `mcts_batch` 1 against 8 (AGZ evaluated leaves in batches of 8): generation 250 against itself, 400 games per setting | cheap: arena games only, no training | not started |
| 7 | AGZ gating (`gating = true`, 55% over 400 games) | expensive; only if regressions appear | not started |
| 8 | `value_hidden = 256`, as in AGZ | architecture change, low priority | not started |

Evaluation only (the algorithm unchanged), all not started: 400-game
matches at temperature 0 (AGZ's evaluation size); a validation loss on
held-out fresh games each generation; and an outside reference opponent
over GTP, for example GNU Go.

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
- `9x9-wide96/`: the 6×96 widening run: `log.csv`, `elo.csv`,
  `results.csv`, `references.csv`, `init.txt`, `progress.txt`, the
  matches `w25-vs-old250.txt`, `w50-vs-old250.txt`,
  `wide50-vs-deep50.txt`, and `analyze-eye.txt`.
- `9x9-sims/`: the simulations experiment: `sims400/` and `sims200/`
  (each with `log.csv`, `elo.csv`, `results.csv`, `references.csv`,
  `init.txt`, `progress.txt`), the four matches `s*-vs-*.txt`, the eye
  analyses `eye-sims{200,400}-s{200,800}.txt`, and the commands
  `sims-experiment.sh` and `sims-experiment-resume.sh`.
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
- `5x5/rerate-opp4/`: `elo.csv`, `references.csv` and `log.csv` of a
  re-rate of 5×5 seed 1 with `--set arena_opponents=4`, made on a copy of
  the run, for the scale caveat above.
- `19x19/`: `log.csv`, `timing.txt`, `analyze-gen2.txt`.
- `passbug/`: the pre-fix runs and the case study.
- `elo.svg`, `elo-5x5.svg`, `elo-9x9.svg`, `elo-passbug.svg`: Elo charts
  drawn from the CSV files above by `tools/plot_elo.py` (the commands are in
  the main `README.md`).
