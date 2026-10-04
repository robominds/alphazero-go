# AlphaZero Go

A from-scratch C++17 implementation of AlphaZero — self-play plus PUCT
Monte Carlo tree search guided by a residual neural network — for the game
of Go on boards from 5×5 to 19×19. It is a **teaching example**, the
successor to `alphazero-tictactoe`: every moving part (rules, convolutions,
backpropagation, search, self-play, training, evaluation) is written
directly, with no machine-learning library, and explained in
`docs/algorithm-explained.md`. It is not meant to be a strong Go engine.

Every number below comes from `docs/results/`, measured on one Apple M4
with 10 cores.

## Build

Requires CMake ≥ 3.16 and a C++17 compiler. The default build uses only the
C++ standard library:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Optionally, matrix multiplication can use BLAS (Apple Accelerate on macOS,
OpenBLAS elsewhere):

```bash
cmake -S . -B build-blas -DCMAKE_BUILD_TYPE=Release -DAZ_USE_BLAS=ON
cmake --build build-blas -j
ctest --test-dir build-blas --output-on-failure
```

Each build has 25 component tests plus an integration smoke test (26 in
ctest; `ctest -N | tail -1` prints `Total Tests: 26`).

**The presets in `configs/` say `backend = blas`**, because BLAS is 2.9×
faster on 5×5 and 18× faster on 19×19 (`docs/results/bench.txt`). Use them
with the `build-blas` binaries. On the default build, add
`--set backend=naive` to `train` (and pass `--backend naive` to `arena` and
`analyze`, which is their default); otherwise `train` stops with
`error: backend 'blas' not available: rebuild with -DAZ_USE_BLAS=ON`.

## Usage

Five programs are built next to the tests. Each prints its options with
`--help`.

**`train`** runs the generation loop (self-play, training, checkpoint,
arena, commit). A new run copies the config into the run directory:

```
$ ./build-blas/train --config configs/5x5.cfg --run runs/5x5-seed1
starting runs/5x5-seed1 at generation 0; effective config (saved to runs/5x5-seed1/config.cfg):
...
gen 1 | selfplay 200 games avg 43.1 plies 6.5s | train loss p 2.85 v 0.49 lr 0.02 2.3s | arena elo 445.6 vs-prev 0.95 (distinct 19/20) 0.4s
...
gen 60 | selfplay 200 games avg 43.3 plies 7.7s | train loss p 1.73 v 0.09 lr 0.002 2.5s | arena elo 1117.5 vs-prev 0.50 (distinct 13/20) 2.0s
done: generation 60 in runs/5x5-seed1
```

These lines are from the reference runs, made before the arena chose its
opponents by rating. The arena part of the line now lists every opponent
with the new network's score, in the form `vs 149:0.55 120:0.70
ref1:0.40` (fixed players by name: `bot100` for a bot, `ref1` for a
reference network, see below), and adds `promoted` when the generation is
promoted into the reference set. `references.csv` in the run directory
lists the reference set with ratings, fixed players included.

`--set key=value` overrides any config key and `--generations N` trains
until generation N; the reference runs used, for example,
`--set seed=2` and `--generations 2`:

```
$ ./build-blas/train --config configs/5x5.cfg --run runs/5x5-seed2 --set seed=2
$ ./build-blas/train --config configs/19x19.cfg --run runs/19x19-check --generations 2
...
gen 2 | selfplay 20 games avg 364.5 plies 6630.8s | train loss p 5.68 v 0.26 lr 0.02 380.7s | arena elo 206.2 vs-prev 0.50 (distinct 10/10) 771.6s
done: generation 2 in runs/19x19-check
```

`--resume` continues a run from its `state.bin`, the commit point written
at the end of every generation, so a run interrupted at any moment resumes
exactly. Tunable keys may be changed on resume; architecture keys may not:

```
$ ./build-blas/train --run runs/5x5-seed1 --resume
resuming runs/5x5-seed1 at generation 60; effective config (saved to runs/5x5-seed1/config.cfg):
...
done: generation 60 in runs/5x5-seed1
$ ./build-blas/train --run runs/5x5-seed1 --resume --set blocks=4
error: cannot change architecture keys on resume: blocks
```

`bot_simulations` and `reference_networks` cannot change on resume
either: bot k is arena player −k and the reference networks come after
the bots, so both lists are fixed for the life of a run. `state.bin`
records them, so a hand-edited `config.cfg` is refused too, as is a
reference network file whose bytes changed.

`--rerate` trains nothing. It replays only the arena over a run's
existing checkpoints with the current opponent selection, and writes
`DIR/rerate/` (`results.csv`, `elo.csv`, `references.csv` and a `log.csv`
with one row per generation). `state.bin` and `config.cfg` are left
untouched, and `--set` applies to the replay only. A run made before
`arena_opponents` changed meaning should be re-rated with the presets'
new value:

```
$ ./build-blas/train --run runs/9x9-main --rerate --set arena_opponents=2
```

`--set` may also change the fixed players for the replay, for example
measuring a run against a strong frozen network instead of its bot:

```
$ ./build-blas/train --run runs/9x9-sims200 --rerate --set bot_simulations= \
      --set reference_networks=runs/9x9-main/checkpoints/gen_0200.azn
```

**`arena`** plays two checkpoints against each other (or, with
`--b-bot SIMS` instead of `--b`, a checkpoint against a uniform bot
searching SIMS simulations per move), colors alternating, and prints the
score with a 95% Wilson interval
(`docs/results/9x9/final-arena.txt`). `--a-simulations N` and
`--b-simulations N` give one side its own search budget instead of
`--simulations`, so a frozen checkpoint can be a stronger reference by
searching more (`--b ckpt --b-simulations 3200`):

```
$ ./build-blas/arena --a runs/9x9-main/checkpoints/gen_0150.azn --b runs/9x9-main/checkpoints/gen_0000.azn --games 100 --simulations 200 --komi 7.5 --backend blas
A runs/9x9-main/checkpoints/gen_0150.azn vs B runs/9x9-main/checkpoints/gen_0000.azn: A scored 100/100 = 1.000 (95% CI 0.963-1.000), distinct games 97
A wins 100, B wins 0, draws 0
```

**`analyze`** plays a move list (GTP coordinates) or an SGF file, then
prints the board, the network's value and policy, and an MCTS search
(`docs/results/9x9/analyze-atari.txt`). For a checkpoint with an
ownership head (below) it also prints, after the value line, an
ownership map: who the network expects to own each point at the end, in
Black's terms, from +9 (surely Black's) to −9 (surely White's), with `.`
where it is unsure (|ownership| < 0.1):

```
$ ./build-blas/analyze --checkpoint runs/9x9-main/checkpoints/gen_0150.azn --moves "D5 E5 F5 C3 E6 G3" --simulations 200 --komi 7.5 --backend blas
runs/9x9-main/checkpoints/gen_0150.azn: 6 moves played, Black to move, komi 7.5
...
Value (side to move): -0.01

Policy prior, % of legal moves ("." below 1%):
     A  B  C  D  E  F  G  H  J
...
 5   .  .  .  X  O  X  .  .  .
 4   .  .  3  3 51  8  5  .  .
 3   .  .  O  .  1  2  O  .  .
...
```

**`bench`** measures network throughput (inference at the search batch of
8 on one thread; a training step at batch 128) and the raw rules engine
(`docs/results/bench.txt`):

```
$ ./build-blas/bench
size   blocks x filters  backend       evals/s  train-step ms
5x5    3 x 32            naive           10544           28.0
5x5    3 x 32            blas            30488           12.1
...
19x19  10 x 128          blas              229         2393.5

size   random-playout moves/s
5x5                   1306178
...
```

### Growing a network

A trained network can be made deeper or wider without losing what it
learned. `grow` appends residual blocks that start as exact identities (a
zero-initialized second batch norm in each, see
`include/azgo/grow.hpp`), checks on 8 random positions that the grown
network's outputs match the original's within 1e-4, and saves it. Then a
new run starts from it, optionally seeded with the old run's replay
buffer. Growing 6 blocks to 10 on 9x9:

```
$ ./build-blas/grow --from runs/9x9-main/checkpoints/gen_0150.azn --blocks 10 --out runs/9x9-ten/start.azn
$ ./build-blas/train --config configs/9x9.cfg --run runs/9x9-ten --set blocks=10 \
      --set lr_schedule=0:0.002 --init-from runs/9x9-ten/start.azn --buffer-from runs/9x9-main
```

`grow --filters 96` widens instead (given `--blocks` too, it widens
first). Each new channel starts as a copy of an old one and every weight
that reads the old channel is split among the copies in random shares
that sum to it, so the outputs are unchanged while the copies receive
different gradients and grow apart. Width costs more than depth: compute
grows linearly with blocks but quadratically with filters, so 96 filters
cost about (96/64)² ≈ 2.25× as much as 64. Widening changes how many terms
each convolution adds up, so in float32 its outputs agree to about 1e-5
rather than bit for bit, hence the 1e-4 limit.

```
$ ./build-blas/grow --from runs/9x9-main/checkpoints/gen_0250.azn --filters 96 --out runs/9x9-wide/start.azn
```

`--init-from` makes the checkpoint generation 0 of the new run (its shape
must equal the config's, so pass `--set blocks=10` or `--set filters=96`);
`--buffer-from` loads the donor run's replay buffer, which must have the
same board size and history, and trims it to `buffer_positions`. The
optimizer's momentum starts at zero. `DIR/init.txt` records where the start came from and how
many positions and games the buffer held. Both options are for new runs
only: `--resume` refuses architecture changes, so growth always starts a
new run rather than continuing the old one. The new run's Elo is anchored
at its own generation 0 (the grown network), so its `elo.csv` is not on
the donor run's scale; compare the two runs with `arena` or by rating
them together.

The presets' learning-rate schedule restarts at 0.02, which is the rate
for a network that knows nothing; for a trained one it would wreck what
the network learned. Override it, for example `--set lr_schedule=0:0.002`
(the 9x9 preset's late-training rate).

### The ownership head

An optional third head predicts, for every point, who owns it when the
game ends: KataGo's auxiliary ownership target, explained in chapter 06
of the explainer. It is off in every preset. Two config keys control it:

- `ownership = true` adds the head. It is an architecture key, so it
  cannot change on resume.
- `ownership_weight` (0–10, default 1.5) weights the head's loss; 0 keeps
  the head but does not train it.

A run with the head shows its loss in the progress line
(`train loss p 2.42 v 0.36 o 0.590 lr 0.002`) and in the last column of
`log.csv`, `ownership_loss`, which stays empty for a run without the
head. A run started before that column existed gets it, empty, on resume.

`grow --ownership` adds a randomly initialized head to a trained
checkpoint. The head changes neither the policy nor the value, so the
usual check of 8 random positions still applies, and it combines with
`--blocks` and `--filters`. A new run starts from the result like from
any grown network. Its donor buffer may come from a run made before
ownership existed: each game's final ownership is then recomputed by
replaying its moves.

```
$ ./build-blas/grow --from runs/9x9-wide96/checkpoints/gen_0050.azn --ownership --out runs/grown/wide96-g50-own.azn
old: 9x9, history 8, 6 blocks, 96 filters, value_hidden 64, 1032349 parameters
new: 9x9, history 8, 6 blocks, 96 filters, value_hidden 64, ownership head, 1032447 parameters
verified on 8 random positions (max difference 0); wrote runs/grown/wide96-g50-own.azn
$ ./build-blas/train --config configs/9x9.cfg --run runs/9x9-ownership --set filters=96 \
      --set ownership=true --set lr_schedule=0:0.002 --init-from runs/grown/wide96-g50-own.azn \
      --buffer-from runs/9x9-wide96
```

`--init-from` refuses a checkpoint whose head does not match the config,
and says how to fix it (`grow --ownership`, or `--set ownership=true`).

### Reference networks

The arena's fixed players are the ones a run never promotes or drops:
uniform bots (`bot_simulations`) and **reference networks**
(`reference_networks`), frozen checkpoints such as the best network of an
earlier run. A bot stops measuring anything early: in the 9×9 re-rate the
100-simulation bot scored at most 1 of 20 against every generation from 30
on and was never chosen after 43 (`docs/results/9x9/rerate/results.csv`),
so the presets set `bot_simulations =` (none) and carry a commented-out
`reference_networks` line instead. Leaving the key out of a config keeps
one 100-simulation bot, so a `config.cfg` written before the key existed
resumes as it did. For a 9×9 run with 96 filters, for example:

```
reference_networks = runs/9x9-main/checkpoints/gen_0200.azn, runs/9x9-sims400/checkpoints/gen_0050.azn
```

The first is a 6×64 network, widened to 96 filters to play in this run;
the second already has the run's shape.

- Paths are comma-separated and resolved from the directory `train` runs
  in (absolute paths are safest). Each file is read once per run.
- A reference network smaller than the run's shape is fitted to it with
  `grow`'s operations: widened, deepened, and given an ownership head (or
  its head dropped), so its policy and value are unchanged (widening up to
  float rounding, about 1e-5). Board size, history and `value_hidden` must
  match, and a network with more blocks or filters than the run is
  refused; so is a missing or corrupt file. Each error names the file.
- Fixed players come bots first: with B bots, reference network j is
  player −(B + j), named `ref1`, `ref2`, … in the progress lines. Like a
  bot it plays `calibration_games` against generation 0 when the run
  starts, and is then chosen by the same expected-score rule. It searches
  with `arena_simulations`.
- `references.csv` lists it with kind `network` (no simulation count), and
  `reference-networks.csv` maps each player ID to its file
  (`player,name,path`).
- `state.bin` (version 4) records each path with a 64-bit hash of the
  file's bytes. `--resume` refuses a changed list or a changed file, and
  `--rerate` refuses a changed file when it keeps the run's list.

## Presets

`configs/5x5.cfg`, `configs/9x9.cfg` and `configs/19x19.cfg`; the comments
in each file give the measurements behind the values. Time per generation
is self-play + training + arena from the runs' `log.csv`
(`docs/results/README.md` has the full table of keys).

| board | komi | blocks × filters | simulations | games per generation | measured time per generation |
|---|---|---|---|---|---|
| 5×5 | 24.5 | 3 × 32 | 200 | 200 | 11.3 s on average; 60 generations in 11.4 min wall time |
| 9×9 | 7.5 | 6 × 64 | 200 | 100 | 4.1 min on average (2.7 to 4.8); 150 generations in 10 h 10 min |
| 19×19 | 7.5 | 10 × 128 | 400 | 20 | 58.2 min for generation 1, 129.7 min for generation 2 |

No preset has a bot or a reference network (see Reference networks
above); the reference runs in `docs/results/` were made with one
100-simulation bot or none.

5×5 uses komi 24.5 because 5×5 Go is solved as a Black win by the whole
board; at komi 7.5 Black wins almost every game and the arena stops
telling generations apart (`docs/results/pilots/`). Checkpoints do not store
komi, and `arena` and `analyze` default to 7.5, so evaluating 5×5 checkpoints
needs `--komi 24.5` (and `--backend blas` with the BLAS build).

## Results

From `docs/results/README.md`, which has the details and the sources.

- **5×5: criterion met in 2 of 2 runs.** Both final networks (seeds 1 and
  2) beat generation 0 in **100 of 100** games (95% interval
  0.963–1.000), and Elo rises clearly: the mean `log.csv` rating of the
  last 10 generations is +469.5 above the first 10 for seed 1 (+361.3 for
  seed 2). Both curves rise until generation 31–34 and then go flat, once
  the network plays 5×5's known perfect line (96.8% of the empty-board
  prior on the center point) and Black wins nearly every self-play game.
- **9×9: criterion not met — the eye check fails.** The final network
  beats generation 0 100/100 (0.963–1.000), and the atari check passes
  (the capture is the top-visited move, 63 of 200 visits). But in the eye
  check the one move that gives a Black group two eyes has a prior below
  1% and is not among the ten most visited moves, and the network rates
  the group's death as no worse than its life (later evidence from a fair
  comparison qualifies this; see the ownership-head bullet below). The `elo.csv` curve rises
  to 784.8 at generation 54 and then goes flat (674.5 at generation 150; a
  slope of only +0.41 Elo per generation over the last third), but that
  is an artifact of anchoring on generation 0: 100-game head-to-heads show
  steady improvement — generation 150 beats generation 50 **98/100**,
  generation 90 86/100, generation 100 76/100 and generation 120 67/100,
  and generation 90 beats generation 50 89/100. Refit without the games
  against generation 0, the curve rises to the end (+449 Elo from
  generation 50 to 150). Re-rated with rating-aware opponents (with
  `arena_opponents = 2`; `docs/results/9x9/rerate/elo.csv`), it rises
  with a slope of +2.83 Elo per generation over the last third, and
  generation 150 is +634 above generation 50, against the head-to-head's
  +676.
- **9×9 continued to 250 and grown to 10 blocks.** The 6-block run was
  continued to generation 250 (re-rated Elo 1728.3 at 150, 1799.5 at 250),
  then grown to 10 blocks and trained 50 more generations. The grown
  network beats the old generation 250 59/100 (0.492–0.681) and 61/100
  at generation 25, about +60 to +80 Elo, and still fails the eye check:
  capacity alone is not the bottleneck at this budget. See
  [the section](docs/results/README.md#9x9-continued-to-250-and-grown-to-10-blocks).
  A 6×96 widening of the same network scored 70/100 against the old 250
  (+147 Elo) but only 52/100 against the 10-block net, so width and depth
  are not separated (see
  [the section](docs/results/README.md#grown-to-696-wider-and-width-vs-depth)).
- **9×9, more simulations.** Doubling self-play simulations (400 vs 200)
  gave no measurable gain at generation 50 (52/100) for 1.66× the time, and
  the eye check still fails (see
  [the section](docs/results/README.md#more-simulations-in-self-play-400-vs-200)).
- **9×9, ownership head.** Training an ownership head alongside the 200-simulation
  run made no measurable difference against the same run without it (48/100 at
  generation 25, 55/100 at 50, interval 0.452–0.644) and the eye check still fails,
  but the head's maps show the network reads the group's life and death correctly,
  and a fair comparison suggests it skips B1 because it judges another move bigger
  (see [the section](docs/results/README.md#an-ownership-head-auxiliary-target)).
- **19×19: criterion met (it runs).** Two generations complete with finite
  losses in 58.2 and 129.7 minutes, and both checkpoints load in
  `analyze`. Two generations say nothing about strength, and none is
  claimed.
- **The pass bug.** The first set of reference runs exposed a search
  blind spot: a game-ending pass reply with a tiny prior was never
  visited, so trained 9×9 networks passed into Tromp-Taylor losses and
  generation 0 won 938 of 3,000 arena games against them. After the fix
  it won 72 of 3,000. `docs/results/passbug/README.md` is the case study.

<img src="docs/results/elo.svg" width="640" alt="Elo by generation, elo.csv refits: 5x5 seed 1 rises to about 1100 by generation 34 and stays there; 9x9 rises to 785 at generation 54 and then stays flat, between 601 and 735 from generation 80 on">

<img src="docs/results/elo-9x9.svg" width="640" alt="Three fits of the 9x9 run: the ladder elo.csv refit, anchored on generation 0, goes flat after generation 50 and ends at 674.5; the ladder refit without generation-0 pairs, anchored at generation 50, keeps rising to +449 at generation 150; the dashed re-rated curve, the same checkpoints replayed against rating-aware opponents and anchored on generation 0, rises to the end at 1766.6, 634 above generation 50">

The first chart is the `elo.csv` refit of the 5×5 seed-1 and 9×9 runs,
both anchored on generation 0 = 0. The second shows why the 9×9 curve
flattens: refit without the pairs against generation 0 (anchored so
generation 50 = 0), the same table rises to the end; so does the dashed
curve, the run re-rated with rating-aware opponents. The charts are drawn
by `tools/plot_elo.py` (standard-library Python):

```bash
python3 tools/plot_elo.py docs/results/5x5/elo.csv docs/results/9x9/elo.csv > docs/results/elo.svg
python3 tools/plot_elo.py "9×9 elo.csv, ladder (gen 0 = 0)=docs/results/9x9/elo.csv" \
    "9×9 ladder without gen-0 pairs (gen 50 = 0)=docs/results/9x9/elo-without-gen0.csv" \
    "9×9 re-rated (gen 0 = 0)=docs/results/9x9/rerate/elo.csv" > docs/results/elo-9x9.svg
python3 tools/plot_elo.py "5×5 seed 1=docs/results/5x5/elo.csv" "5×5 seed 2=docs/results/5x5-seed2/elo.csv" > docs/results/elo-5x5.svg
python3 tools/plot_elo.py "9×9 before the pass fix=docs/results/passbug/9x9-main/elo.csv" \
    "9×9 after the fix=docs/results/9x9/elo.csv" > docs/results/elo-passbug.svg
```

## Learn how it works

- [`docs/algorithm-explained.md`](docs/algorithm-explained.md) — the
  explainer: every idea, chapter by chapter in the order the code runs,
  quoted against the real source, with the measured results.
- [`docs/algorithm-explained.html`](docs/algorithm-explained.html) — the
  same text, illustrated: board, ko, im2col, network, search and
  training-loop diagrams, and the Elo charts. Open it in a browser; it
  follows the system's light or dark theme.
- [`docs/results/README.md`](docs/results/README.md) — every measured
  number and the verdict for each board size.
- [`docs/results/passbug/README.md`](docs/results/passbug/README.md) — the
  pass blind spot: symptom, diagnosis, fix, before and after.
- [`docs/superpowers/specs/2026-09-29-alphazero-go-design.md`](docs/superpowers/specs/2026-09-29-alphazero-go-design.md)
  — the design spec.
- [`docs/superpowers/plans/2026-09-29-alphazero-go.md`](docs/superpowers/plans/2026-09-29-alphazero-go.md)
  — the implementation plan.
- [`docs/superpowers/specs/2026-10-01-rating-aware-arena-design.md`](docs/superpowers/specs/2026-10-01-rating-aware-arena-design.md)
  and [`docs/superpowers/plans/2026-10-01-rating-aware-arena.md`](docs/superpowers/plans/2026-10-01-rating-aware-arena.md)
  — the rating-aware arena: opponents chosen by rating, the uniform bot and `--rerate`.
- [`docs/superpowers/specs/2026-10-03-ownership-head-design.md`](docs/superpowers/specs/2026-10-03-ownership-head-design.md)
  and [`docs/superpowers/plans/2026-10-03-ownership-head.md`](docs/superpowers/plans/2026-10-03-ownership-head.md)
  — the ownership head: an auxiliary per-point target, `grow --ownership` and the analyze map.

`python3 tools/check_citations.py` checks that every `file:line` citation
in both explainers still points at the quoted definition.

## Extending to the GPU

A `MetalBackend` would implement the `Backend` interface (gemm, im2col,
col2im) with Metal compute kernels, so training steps benefit first. A
batching `Evaluator` would then gather leaves from all worker threads into
batches of 64–256 for the GPU, leaving MCTS unchanged. On Apple Silicon the
BLAS backend already runs on the CPU's matrix units, and the spec estimates
the M4 GPU's additional gain at roughly 2× and only at large batch sizes,
which is why v1 stops at the CPU.

## Authorship

Written by Mark Castelluccio. Designed and implemented with the assistance of
Claude (Anthropic), an AI assistant; commits carry a Co-Authored-By trailer.
