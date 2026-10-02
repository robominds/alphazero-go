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

Each build has 24 component tests plus an integration smoke test (25 in
ctest; `ctest -N | tail -1` prints `Total Tests: 25`).

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
bot100:1.00`, and adds `promoted` when the generation is promoted into the
reference set. `references.csv` in the run directory lists the reference
set with ratings, bots included.

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

`bot_simulations` cannot change on resume either: bot k is arena player
−k, so the list is fixed for the life of a run.

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

**`arena`** plays two checkpoints against each other (or, with
`--b-bot SIMS` instead of `--b`, a checkpoint against a uniform bot
searching SIMS simulations per move), colors alternating, and prints the
score with a 95% Wilson interval
(`docs/results/9x9/final-arena.txt`):

```
$ ./build-blas/arena --a runs/9x9-main/checkpoints/gen_0150.azn --b runs/9x9-main/checkpoints/gen_0000.azn --games 100 --simulations 200 --komi 7.5 --backend blas
A runs/9x9-main/checkpoints/gen_0150.azn vs B runs/9x9-main/checkpoints/gen_0000.azn: A scored 100/100 = 1.000 (95% CI 0.963-1.000), distinct games 97
A wins 100, B wins 0, draws 0
```

**`analyze`** plays a move list (GTP coordinates) or an SGF file, then
prints the board, the network's value and policy, and an MCTS search
(`docs/results/9x9/analyze-atari.txt`):

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

A trained network can be made deeper without losing what it learned.
`grow` appends residual blocks that start as exact identities (a
zero-initialized second batch norm in each, see
`include/azgo/grow.hpp`), checks on 8 random positions that the grown
network's outputs match the original's within 1e-5, and saves it. Then a
new run starts from it, optionally seeded with the old run's replay
buffer. Growing 6 blocks to 10 on 9x9:

```
$ ./build-blas/grow --from runs/9x9-main/checkpoints/gen_0150.azn --blocks 10 --out runs/9x9-ten/start.azn
$ ./build-blas/train --config configs/9x9.cfg --run runs/9x9-ten --set blocks=10 \
      --set lr_schedule=0:0.002 --init-from runs/9x9-ten/start.azn --buffer-from runs/9x9-main
```

`--init-from` makes the checkpoint generation 0 of the new run (its shape
must equal the config's, so pass `--set blocks=10`); `--buffer-from` loads
the donor run's replay buffer, which must have the same board size and
history, and trims it to `buffer_positions`. The optimizer's momentum
starts at zero. `DIR/init.txt` records where the start came from and how
many positions and games the buffer held. Both options are for new runs
only: `--resume` refuses architecture changes, so growth always starts a
new run rather than continuing the old one.

The presets' learning-rate schedule restarts at 0.02, which is the rate
for a network that knows nothing; for a trained one it would wreck what
the network learned. Override it, for example `--set lr_schedule=0:0.002`
(the 9x9 preset's late-training rate).

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
  the group's death as no worse than its life. The `elo.csv` curve rises
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
