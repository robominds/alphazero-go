# Ownership Head — Design

Author: Mark Castelluccio, with design assistance from Claude (Anthropic).

Amends the network, self-play and training sections of
`docs/superpowers/specs/2026-09-29-alphazero-go-design.md`.

## Problem

Every 9×9 experiment so far has failed the eye check:

- the 6-block run to generation 250;
- the network grown to 10×64;
- the network widened to 6×96;
- self-play at 400 versus 200 simulations.

In the test position Black must play B1 to give a bottom-edge group two
eyes. The networks rate White's B1 (killing) about the same as Black's B1
(saving), and search does not fix that: up to 3200 analysis simulations
never visit B1 among the top moves.

The weakness is the value head's life-and-death judgment. Its only
training signal is one ±1 game result per position. That signal is noisy,
and under Tromp-Taylor scoring a group's death is only settled after a
long capture sequence.

## Goal

Add KataGo's auxiliary **ownership** target: for every point, predict who
owns it at the end of the game. This gives n² dense training signals per
position. In particular, every point of a dead group is labelled with the
opponent's colour, which teaches life and death directly. `analyze` also
gains an ownership map, so the network's judgment of each group can be
seen on the board.

## Non-Goals

- Using ownership in search, for example score estimation or
  ownership-weighted values. Search is unchanged.
- KataGo's other auxiliary targets (score, opponent policy, etc.).
- Changing presets. Ownership is opt-in. The existing runs and checkpoints
  keep working unchanged.

## Design

### Ownership of a final position

- `Board::ownership(int8_t* out)` writes n² values from Black's
  perspective, using Tromp-Taylor area rules:
  - +1 for a Black stone or an empty region reaching only Black;
  - −1 for the White equivalents;
  - 0 for neutral points (empty regions touching both colours, i.e. dame).
- `Board::areaScore()` becomes the sum of `ownership`, so both share one
  implementation.

### Network

- `NetConfig` gains `bool ownership` (default false). It is an
  architecture key: it changes the parameter set, so resume refuses to
  change it.
- When enabled, an **ownership head** reads the trunk output:
  - 1×1 conv, `filters` → 1;
  - BatchNorm;
  - then, separately from the value head's path, a per-point tanh.

  The output is n² values in [−1, 1], from the **side to move's**
  perspective, matching the value head's convention. The head has
  `filters` + 2 + 2 parameters (conv weights plus BN gamma/beta) and adds
  negligible compute.
- Its parameter names are `ownership.conv.w`, `ownership.bn.gamma` and
  `ownership.bn.beta`, plus the BN running statistics.
- `infer` gains an optional ownership output. Callers that pass none
  (MCTS, the arena) are unaffected.
- **Loss:**

  `L = L_policy + L_value + w · (1/n²) · Σ_p (ô_p − o_p)²`

  averaged over the batch, where `o` is the target ownership in the side
  to move's perspective. The weight `w` comes from config key
  `ownership_weight` (tunable, 0–10, default 1.5). With `w = 0` the head
  still exists but does not train.

  `LossStats` gains `ownershipLoss`. The trainer logs it as a new trailing
  `log.csv` column `ownership_loss`, left empty when ownership is off.

### Data

- `GameRecord` gains `std::vector<int8_t> finalOwnership`: n² values from
  Black's perspective, computed from the game's final board when the game
  ends.
- **Replay buffer:**
  - Each sample gains an ownership target: `finalOwnership` negated when
    White is to move, then mapped through the same symmetry as the input
    planes and π (`transformPoint`).
  - `TrainingBatch` gains `ownership [size][area]`.
  - Buffer serialization stores `finalOwnership`.
  - **Loading older buffers** (state.bin v2) recomputes ownership by
    replaying each game's moves with `Game`. Replay is deterministic and
    cheap relative to a training run. A donor buffer from a pre-ownership
    run (`--buffer-from`) therefore works.
- **state.bin** moves to version 3 (buffer records include ownership).
  Versions 1 and 2 still load.

### Checkpoints and growth

- Checkpoint format version 2 adds the `ownership` flag to the
  architecture header. Version-1 checkpoints load as `ownership = false`.
- `grow --ownership` adds an ownership head to a checkpoint that lacks one.
  - The new head is randomly initialized.
  - It cannot change the policy or value, so `grow`'s equivalence check
    (which compares policy and value) still applies.
  - It can be combined with `--blocks` / `--filters`.
- `growBlocks` and `widenFilters` carry an existing ownership head along.
  Widening splits its conv input weights like the other stream readers.

### Analyze

- When the checkpoint has an ownership head, `analyze` prints an
  **ownership map** after the value line.
- The map shows each point's predicted ownership from **Black's**
  perspective as an integer from −9 to +9 (rounded 9·ô), with `.` when
  |ô| < 0.1.
- It uses the same coordinates as the board diagram.

### Config

- New keys:
  - `ownership`: bool, default false, architecture key;
  - `ownership_weight`: tunable, 0–10, default 1.5.
- Presets are unchanged (ownership off).

## Error handling

- Resume refuses a changed `ownership` (architecture-key rule).
- `--init-from` with a checkpoint whose ownership flag differs from the
  config gives a `ConfigError` that suggests `grow --ownership`.
- Loading an older buffer whose moves fail to replay is a
  `runtime_error` that names the state file. This should be impossible
  for files the program wrote.

## Testing

- **Ownership of a position:**
  - `Board::ownership` matches hand-counted examples: an empty board
    (all 0), a single stone, a seki and dame.
  - Σ ownership equals `areaScore` on random final positions.
- **Network:**
  - A finite-difference gradient check of the whole network includes the
    ownership head (small 5×5 net, ownership on).
  - Output shapes are correct.
  - With ownership off, parameters, outputs and checkpoints are identical
    to before (the existing tests still pass).
- **Perspective and symmetry:** for a sample where White is to move, the
  ownership target is negated. Under every symmetry it maps exactly like
  the input planes. These are hand-checked on a small game.
- **Loss:**
  - With w = 0, the ownership loss gradient is 0 and policy/value
    training is unchanged.
  - With w > 0, a few SGD steps on a fixed batch reduce the ownership
    loss.
- **Data persistence:**
  - state.bin version 3 round-trips.
  - Version-2 state loads with recomputed ownership equal to direct
    computation.
  - Checkpoint version 1 loads as ownership off.
- **`grow --ownership`:** preserves policy and value exactly. Widen and
  deepen carry an ownership head along exactly, comparing all three heads.
- **Determinism:** resume byte-identity and worker-count independence hold
  with ownership on.
- **`analyze`:** prints the map only for ownership checkpoints.

## Teaching material

- Explainer chapter 06: a section "An auxiliary target: ownership"
  explaining:
  - why a single game result is a weak signal;
  - how per-point ownership densifies it;
  - the perspective and symmetry handling;
  - the KataGo reference.

  It shows the ownership map from the experiment, with the HTML mirrored
  and citations checked.
- README usage: the `ownership` keys, `grow --ownership`, and the
  analyze map.

## Experiment

1. `grow --ownership` the 6×96 generation-50 network.
2. Train 50 generations in `runs/9x9-ownership`:
   - `--init-from` that network, `--buffer-from runs/9x9-wide96`;
   - the same settings as the existing `runs/9x9-sims200` arm (200
     simulations, `lr_schedule = 0:0.002`, `filters = 96`), plus
     `ownership = true`.
3. Compare with `runs/9x9-sims200`, which has the same start, settings and
   seed apart from ownership:
   - 100-game matches at generations 25 and 50, each against the start
     and against sims200 at the same generation;
   - the eye check at 200 and 800 simulations;
   - the ownership map of the eye position, both before and after White
     B1, so it is visible whether the network knows the group dies.
4. Report in `docs/results/9x9-ownership/` with an honest verdict.
