# Batch Norm Calibration of Generation 0 — Design

Author: Mark Castelluccio, with design assistance from Claude (Anthropic).

Amends "`layers` / `network` — the ResNet" (batch norm's two modes) in
`docs/superpowers/specs/2026-09-29-alphazero-go-design.md`, and how
`Trainer::startNew` makes generation 0. The algorithm
does not change: self-play, training, the arena and every later
generation work as before. Only generation 0's batch norm running
statistics change, and only in a new run started from a random network.

## Problem

A 9×9 run with a 10×128 network (`configs/9x9.cfg` with `blocks = 10`,
`filters = 128`) collapsed at once, twice (`9x9-scratch128-collapsed`,
resignation on, and `-collapsed2`, resignation off, on the Mac mini):

| run | gen | avg plies | value loss | val policy loss | val value loss |
|---|---|---|---|---|---|
| collapsed (resign on) | 1 | 17.6 | 0.121 | 19.03 | 2.02 |
| collapsed (resign on) | 2 | 7.4 | 0.110 | 3.92 | 0.85 |
| collapsed2 (resign off) | 1 | 17.8 | 0.006 | 19.05 | 2.05 |
| collapsed2 (resign off) | 2 | 19.5 | 0.0001 | 3.22 | 0.003 |

Generation 0 is a freshly initialized network. Self-play, the arena and
the validation loss evaluate it in inference mode, where batch norm
normalizes with its running statistics, and a network that has never
trained has the initial ones: mean 0, variance 1. With γ = 1 and β = 0
such a batch norm is the identity. He initialization keeps each
convolution near its input's scale, but each residual block adds its
output to its input, so through 21 convolutions the trunk's activations
grow. The value head saturated: +1.00 at the empty board and on every
position tried (CPU and GPU agree), and the policy's entropy was 0.009
nats (uniform over 82 actions: 4.41). The search then rested on a value
that never moved, the games ended after about 18 plies in early
pass-pass endings whose exact Tromp-Taylor scores dominated it, the
value loss went to 0.00 on those games, and training never recovered.
6×64 and 6×96 networks survived by luck of scale.

Training mode never sees this: it normalizes with batch statistics. The
fault is only that generation 0 is used in inference mode with statistics
nobody collected.

## Decision

When `Trainer::startNew` creates a random generation 0 (no `--init-from`),
and `bn_calibrate` is true (the default), it calibrates the network's
batch norm before saving `gen_0000.azn`: one training-mode forward pass
over a fixed batch of positions, in which every batch norm layer's
running mean and variance are set to the batch statistics of its own
input in that pass.

### Exact, in one pass

`BatchNorm2d::forwardTrain` gains a `momentum` argument, default
`kMomentum` (0.1, so training is unchanged bit for bit), and its update
`running = (1 − momentum) · running + momentum · batch` with momentum 1
sets the running statistics to the batch's exactly. `Network` gains a
fourth mode, `Calibrate` (batch statistics, running statistics set), and
`Network::calibrateBatchNorm(ws, input, B)`. In the one pass, each layer
normalizes with the batch statistics of its own input, which the layers
below produced with batch statistics, so every layer's statistics are
those training mode sees, and afterwards inference mode on that batch
computes what training mode does. The variance is the unbiased one
(M/(M − 1)), as the training-mode update stores. Parameters and gradients
are untouched; calibrating twice gives the same bits.

### The positions

`bnCalibrationPositions(n, history, seed, count)`: `count` = 256
positions, each from its own random game of uniformly random legal moves
(a pass only when no point is legal), position i after i·n²/count moves,
so lengths run evenly from the empty board to n² moves (81 on 9×9, the
middle of a self-play game), Black and White to move alike. Each is
encoded like a self-play position, history planes included, at the
identity symmetry. The generator is `Rng(deriveSeed(seed, 0,
kPhaseBatchNorm))` with the new phase constant `kPhaseBatchNorm = 7`
(`kPhaseInit`, which seeds the weights, and `kPhaseCalibration`, the fixed
players' rating games, are taken).

Why 256: it is the training batch size of the 9×9 and 19×19 presets
(5×5 trains on 128), so each layer's statistics come from at least a
training batch's worth of values (256 · 81 = 20,736 per channel on 9×9;
the standard error of a channel's mean is under 1% of its
standard deviation), and one pass costs about one forward of a training
step: 1.4 to 1.9 s for 10×128 on 9×9 with the naive backend on 4 threads.
The workspace is the training step's size, which the run allocates for
training anyway. Random games rather than self-play ones: no network is
needed to make them, and generation 0's own games would be a circular
source. Lengths up to n² cover the positions generation 0 meets early in
its games, the part of a game that shapes its first training data.

### Determinism

The pass runs on the CPU with the naive backend and a `ThreadPool` of
`effectiveTrainThreads()` threads, whatever `backend` says. The layers
cut their work into chunks fixed by the shapes, so the bits do not depend
on the thread count, and BLAS and the GPU, which round differently, are
not used. `gen_0000.azn` is therefore the same bytes for any `workers`,
`train_threads` and `backend` (naive, blas, metal): on one machine type,
a function of the seed and the network keys. Across architectures the
compiler may contract multiply-adds into FMAs differently, so the last
bits can differ (an M5 Max and an M4 gave the same bytes for 10×128).

### The key

`bn_calibrate` (bool, default true), listed with the network keys after
`ownership` and recorded in `config.cfg`. It is not an architecture key
and is tunable; it is read only when a run starts. `--init-from` never
calibrates: a given network has statistics of its own (grown and trained
networks alike). Resume and re-rating never calibrate either. A
`config.cfg` written before the key existed lacks it, and that run's
generation 0 was not calibrated, so `Trainer::resume`, `rerateRun` and
`--buffer-from` read a run's `config.cfg` with false as the missing
key's value (`loadRunConfig`), and a resume rewrites `config.cfg` with
`bn_calibrate = false`. A new config's missing key is true.

`config.cfg`'s `bn_calibrate` records the setting, not necessarily what
generation 0 got: a run started with `--init-from` records true but
used the given network as it was, and a value changed on resume is
recorded although generation 0 is long made.

The presets leave the key at its default, as they do `train_threads`.

## What changes for users

- A new run from a preset gets a calibrated generation 0. A run made
  before this change with the same seed and keys had a different
  generation 0, and so a different run. `--set bn_calibrate=false`
  reproduces the old run byte for byte (checked: a 2-generation 5×5 run
  of the previous commit and of this one with the key off have identical
  checkpoints, `state.bin`, `results.csv` and `elo.csv`).
- Old runs, resumes, `--init-from` and `--rerate` are unaffected.
- `tests/test_metal.cpp`'s run whose arena rounds have every kind of
  match chose its settings so that a promotion happens within four
  generations; a calibrated generation 0 plays differently and no
  promotion happens, so that test sets `bn_calibrate = false`. No other
  test pinned a fresh run's generation-0 bytes.

## Measured

On the M5 Max (`tests/test_bn_calibration.cpp`), the 9×9 preset's
generation 0 with a 10×128 network (seed 1), 16 random positions from the
empty board to 81 moves:

| | uncalibrated | calibrated |
|---|---|---|
| value at the empty board | 1.0000 | 0.4701 |
| smallest \|value\| | 1.0000 | 0.1736 |
| median \|value\| | 1.0000 | 0.6524 |
| mean policy entropy (nats; uniform 4.407) | 0.009 | 3.856 |

A calibrated random network is still a random function: on some
positions its value is large, as it is in training mode, whose batch
statistics it now matches. But it is no longer saturated everywhere, and
its policy is close to uniform.

A real run on the Mac mini (M4, `backend = metal`), the 9×9 preset with
`blocks = 10`, `filters = 128`, seed 1:

| run | gen | avg plies (median) | value loss | val policy loss | val value loss | resign: v_resign, rate, fp | self-play |
|---|---|---|---|---|---|---|---|
| resignation on | 1 | 125.6 (124) | 0.560 | 4.65 | 1.54 | −0.90, 0.89, 0.000 | 2154 s |
| resignation on | 2 | 76.0 (83) | 0.40 | 4.28 | 1.14 | −0.77, 0.98, 0.093 | 1441 s |
| resignation off | 1 | 124.1 (120) | 0.588 | 4.65 | 1.56 | — | 1999 s |

Training policy loss 4.21–4.22 in all three rows (`seen` at generation 2:
policy 4.21, value 0.33). Against the collapsed runs above: games 7 to 17
times longer, a validation policy loss of 4.65 instead of 19.0 at
generation 1, and a value loss that stays well above 0. The 6×64 run of
the same preset from scratch (`docs/results/9x9-scratch/log.csv`) had
76.9 and 89.2 plies, value loss 0.61 and 0.53, and validation policy loss
5.52 and 4.18 at generations 1 and 2.

Two things to watch, neither a collapse:

- Without resignation the games are longer than the 6×64 run's: 24 of
  200 reached the 162-move cap, and the first pass came at ply 49 on
  average. Generation 0's policy is close to uniform, so passes are rare
  and games run on.
- With resignation, generation 2 resigned 98% of the games that could
  resign, and the false-positive rate at the threshold the first
  generation's 20 disabled games gave (−0.77) was 0.093, above the 5%
  target while the calibration window is small. 25 of its 200 games
  ended within 20 plies (21 of them resigned), against 2 at generation 1.
  The collapsed run's generation 2 had threshold −0.37, false positives
  0.14 and 7.4 plies per game. The runs were stopped there (two
  generations with resignation, one without), so whether the threshold
  settles is not measured.

## Tests

- `tests/test_layers.cpp`: momentum 1 sets the running statistics to the
  batch's, whatever they were.
- `tests/test_network.cpp`: on a network without blocks (stem, policy,
  value and ownership batch norms), the running statistics after
  calibration equal the mean and unbiased variance of each layer's input,
  computed by hand in double from the weights; with blocks, inference
  mode on the calibration batch then matches training mode's loss to
  2e-3, calibrating again changes no bit, and pools of 1, 3 and 7
  threads give the same bits.
- `tests/test_bn_calibration.cpp`: the positions are deterministic in the
  seed, have history and both colors to move, and reach mid-game boards;
  calibration depends on the seed and not on the thread count, and
  changes only the running statistics; the 10×128 table above (the bug
  without calibration, none with it).
- `tests/test_trainer.cpp`: `gen_0000.azn` equals calibrating
  `Network(cfg, deriveSeed(seed, 0, kPhaseInit))` with
  `deriveSeed(seed, 0, kPhaseBatchNorm)`, for 1 and 3 workers, 1 and 4
  training threads and every available backend; with the key off it has
  the same weights and the initial statistics; `--init-from` keeps the
  given file's bytes; a resume never touches a checkpoint, writes
  `bn_calibrate = false` for a `config.cfg` without the key and keeps
  `true` for one with it.
- `tests/test_config.cpp`: the key's default, parsing, printing,
  position, that it is tunable, the presets, and a missing key taking the
  base's value.
