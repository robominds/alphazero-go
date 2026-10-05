# Validation Loss on Fresh Self-Play Games — Design

Author: Mark Castelluccio, with design assistance from Claude (Anthropic).

Amends the training and output sections of
`docs/superpowers/specs/2026-09-29-alphazero-go-design.md`. Implements the
evaluation-only item "a validation loss on held-out fresh games each
generation" from the Next experiments list in `docs/results/README.md`.

## Problem

`policy_loss` and `value_loss` in `log.csv` are means over the
generation's training batches, drawn from the replay buffer the network
is being fitted to; on the late 9×9 runs each buffered position is drawn
about 5.8 times. A falling training loss therefore cannot separate
better judgment from a closer fit to the window. The learning-rate
restart run had a lower value loss than its control at the higher rates
and lost the generation-25 match 36/100. That lower loss may have been
overfitting, and the logs cannot say.

## Goal

Measure, each generation, the network's loss on data it has never
trained on, and on its training distribution, in the same way, so the
two can be compared. Measurement only: the algorithm stays AlphaGo Zero's, and a run
produces exactly the same checkpoints, buffer and ratings with the
measurement on or off.

## Non-Goals

- Using the measurement for anything: early stopping, learning-rate
  changes, gating. Nothing feeds back.
- A separate validation set kept across generations, or games played
  only for validation: every self-play game is still used for training.
- Symmetry-averaged evaluation (all 8 symmetries per position). It would
  cost 8 times as much and is not needed for a comparison.

## Design

### What is measured

Generation g, right after self-play (statistics, resignation window, SGF)
and **before** its games are added to the buffer:

- **val**: `net_`, the network about to train, which is generation g − 1's
  checkpoint, scored on every position of generation g's games. Without
  gating it is the network that played them; with gating the games come
  from the best network, but the question is still about the network
  being trained, so `net_` is the one measured. It has never trained on
  these positions.
- **seen**: the same network scored on positions drawn uniformly, with
  replacement, from the buffer as it stands: its training distribution,
  the window generation g − 1 sampled its batches from. Training drew
  from it at random symmetries, so some of these positions were seen
  several times and some rarely or never. The sample has as many
  positions as the fresh games, so both means are equally noisy. With an
  empty buffer (generation 1 of a run started without `--buffer-from`)
  nothing is measured. With `--buffer-from`, generation 1's sample comes
  from the donor's games, which generation 0 never trained on, so that
  one row is held out too.

Both use the training loss's terms (`Network::loss`): policy cross-entropy
against the normalised visit counts π, (z − v)², and with the ownership
head the unweighted per-point squared error (as `ownership_loss`
reports it). Each position is encoded once at the identity symmetry
(`encodeTrainingSample`, the encoder the buffer uses) and evaluated in
inference mode: batch norm with its running statistics, as search uses
the network.

Seen against val in the same generation is a like-for-like comparison:
the same network, mode and encoding. The training loss is not: it uses
batch statistics and random symmetries, and its mean runs over a network
that changes during the generation's steps. The gap mixes fitting with
the drift between older and fresh games. Val below seen is possible and
not a fault: without gating the fresh games' π targets come from this
network's own search, the buffer's mostly from older networks'.

In a run started from scratch, generation 1's val row scores the random
generation-0 network with batch norm's initial running statistics, so it
is not comparable with later rows; with `--init-from`, generation 0 is the
given network.

### Why it cannot change a run

- No generator of self-play, training or the arena is touched. The buffer
  sample uses its own: `Rng(deriveSeed(seed, g, kPhaseValidation))`, with
  the new phase constant `kPhaseValidation = 6`.
- `Network::inferLoss` runs the inference forward pass (`Mode::Infer`,
  which only reads parameters and running statistics) and `loss()` without
  gradients; it is `const`, like `infer`. It writes only to its own
  `Workspace`.
- The buffer is only read; the new games are added afterwards in the same
  order as before.
- The workspace has `batch_size` rows and the backend `workers` threads,
  like training. The naive backend computes every output element on one
  thread and BLAS runs single-threaded, so the result does not depend on
  the worker count.

### Components

- `encodeTrainingSample(game, ply, history, sym, ...)`, a free function
  in `replay_buffer`, holds the encoding; `ReplayBuffer::encodeSample`
  calls it. `ReplayBuffer::locate(position)` returns the (game, ply) that
  `sample()` already computed inline.
- `Network::inferLoss(ws, input, pi, z, B, ownership)`: the loss terms in
  inference mode.
- `validation.hpp`: `PositionRef`, `allPositions(games)`,
  `samplePositions(buffer, count, rng)` (one `rng.below(positions)` per
  position), and `heldOutLoss(net, ws, positions, history)`, which batches
  positions by `ws.maxBatch()` and weights each batch mean by its size.
  It returns -1 for every loss when there were no positions, and for the
  ownership loss when the network has no head.

### Config

`validation_loss` (bool, default `true`, tunable, not an architecture
key). It is on by default because it costs little and changes no
result; a `config.cfg` written before the key existed measures it too.
On resume it can be turned off or on; the next row follows.

### Output

- `log.csv` gains six trailing columns after `resign_false_positive`:
  `val_policy_loss, val_value_loss, val_ownership_loss, seen_policy_loss,
  seen_value_loss, seen_ownership_loss`, printed `%.5f`, empty when not
  measured, `nan` when a measurement is not a number (so a broken
  network shows instead of looking unmeasured). `Trainer::resume` upgrades every older header (before
  ownership, before resignation, before validation) to the new one,
  adding empty fields to old rows.
- The progress line adds `val p X v Y` after the learning rate, and
  `seen p X v Y` once there is a buffer.
- `GenerationStats` gains the six values (-1 when not measured).
- The pass's time is added to `train_s` and taken out of `selfplay_s`.
- `train --rerate` replays only the arena and is unaffected.

### Cost

One forward pass per fresh position and one per sampled one: on 9×9 with
100 games of about 88 plies, about 17,600 positions, 69 batches of 256.
A micro-benchmark (6×96 network, BLAS, one thread) put that at 8% of the
time of the generation's 200 training steps of 256.

## Testing

- **Network:** `inferLoss` equals a hand computation from `infer()`'s
  logits, values and ownership after some training steps, and leaves
  parameters, gradients and running statistics bit-identical; batch
  statistics (`computeLoss`) give another value.
- **Buffer:** `locate` numbers positions from the oldest game;
  `encodeTrainingSample` equals `encodeSample` for every position and two
  symmetries.
- **Validation:** `allPositions` order; `heldOutLoss` equals a
  per-position hand computation (encode, `infer` at batch 1,
  cross-entropy, squared errors) for batch sizes 1, 5 and 64, with and
  without the head, and leaves the network untouched; no positions give
  -1; `samplePositions` is `locate(rng.below(n))` per draw and consumes
  nothing else.
- **Config:** default, order (after `ownership_weight`), round trip,
  parse errors, tunability, old configs and presets on.
- **Trainer (tiny 5×5 runs):**
  - the logged values for generations 1 and 2 equal `heldOutLoss`
    recomputed from the saved checkpoints and buffer (generation 2's seen
    sample with `deriveSeed(seed, 2, kPhaseValidation)`), and the
    progress line shows them;
  - generation 1 has no seen values; no ownership columns without the
    head, both with it;
  - on against off, with and without the head: every checkpoint,
    `state.bin`, the rating files, the SGF and the other non-timing
    `log.csv` fields are byte-identical;
  - the values do not depend on the worker count;
  - a `log.csv` without the six columns is upgraded on resume, and
    `validation_loss` can be switched on resume;
  - a resumed run logs the same six values as an uninterrupted one;
  - a generation-0 network whose ownership running means are NaN (never
    read by search or by training) logs `nan` for the ownership columns.
  The existing log tests read `log.csv` by column name now, not by
  trailing commas.
- **Against the previous binary:** short 5×5 runs (40 games, 50
  simulations, a 1,500-position buffer so games are evicted) built from
  `main` and from this branch, plain, with the ownership head, with
  resignation, and resumed halfway: every checkpoint, SGF, `state.bin`,
  `results.csv`, `elo.csv`, `references.csv` and the non-timing columns
  of `log.csv` are byte-identical.
- **Smoke test:** the resumed run's progress line and `log.csv` carry
  the values; the other log checks read columns by name.
