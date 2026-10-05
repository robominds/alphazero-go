# Resignation in Self-Play — Design

Author: Mark Castelluccio, with design assistance from Claude (Anthropic).

Amends the self-play, config and training-state sections of
`docs/superpowers/specs/2026-09-29-alphazero-go-design.md`, which listed
resignation as a non-goal.

## Problem

Every self-play game is played to two passes or the move cap. Once one
side is clearly lost, the remaining plies cost the same search as the
opening and confirm a result the value head already predicts. On 9×9,
games average 87 to 116 plies, so the tail of lopsided games is a large
share of self-play time.

## Goal

Add AlphaGo Zero's resignation rule, faithfully (Silver et al., "Mastering
the game of Go without human knowledge", Nature 2017, Methods,
"Self-play"):

> To save computation, clearly lost games are resigned. The resignation
> threshold v_resign is selected automatically to keep the fraction of
> false positives (games that could have been won if AlphaGo had not
> resigned) below 5%. To measure false positives, we disable resignation
> in 10% of self-play games and play until termination.

A player resigns when its root value and its best child's value are both
below v_resign.

## Non-Goals

- Refinements from later engines (KataGo, Leela Zero), such as resigning
  only after several consecutive low values, or playing out resigned games
  with fewer playouts.
- Resignation in the arena, `--rerate`, `arena` or `analyze`: those
  measure strength and always play games out.
- Changing presets. Resignation is opt-in, and runs without it are
  byte-identical to before.

## Design

### The resignation value

- After each self-play search, the side to move computes
  `resignValue = max(Mcts::rootValue(), Mcts::bestChildValue())`, both
  from its own view. "Root value and best child value both below v" is
  the same as this maximum being below v.
- `Mcts::rootValue()` already exists: the mean backed-up value over the
  root's edges (the network's value at the root while nothing is
  visited).
- `Mcts::bestChildValue()` is new: the Q of the most visited root edge.
  Among equally visited edges it takes the best Q, the conservative choice.
- Resignation is never offered at ply 0 (Black at the empty board). This
  keeps every `GameRecord` non-empty, which the replay buffer requires.

### Self-play

- `SelfPlayConfig` gains `resignation`, `resignThreshold` and
  `resignDisabledFraction`.
- With resignation on, each game first draws `rng.uniform()` once from its
  own `Rng`: below `resignDisabledFraction`, resignation is disabled for
  the game. With resignation off, nothing is drawn, so existing runs keep
  their exact random streams.
- In an enabled game, a side whose `resignValue < resignThreshold`
  resigns. The game ends before that position is recorded (no move was
  played from it). `resultForBlack` is a loss for the resigning side, and
  z follows from it as for any game.
- Every game, enabled or not, records per side the minimum
  `resignValue` over its moves from ply 1 on (+1 when there were none).
  `GameRecord` gains `resigned`, `resignDisabled` and
  `minResignValue[2]`. These fields are for the generation's statistics
  only; the replay buffer does not store them.
- `GameRecord::winnerMinResignValue()` is the winner's minimum. A
  disabled game is a false positive at threshold v when this is below v:
  the winner would have resigned. A draw (possible only with integer
  komi) has no winner and counts as +1, never a false positive.

### Calibration

- `calibrateResignThreshold(winnerMinimums, {window, minGames,
  maxFalsePositive, initialThreshold})`:
  - It considers the last `window` entries.
  - With fewer than `minGames` entries it returns `initialThreshold`.
  - Otherwise it returns the largest v on the grid −1.00, −0.99, …,
    0.00 (v = k / 100) whose false-positive rate
    (count(min < v) / games) is at most `maxFalsePositive`, compared as
    floats so a rate exactly at the limit qualifies. v = −1 always
    qualifies, since no value is below −1.
- `resignFalsePositiveRate(winnerMinimums, window, v)` gives that rate,
  or 0 for an empty window.
- The trainer appends each generation's disabled games, in game-index
  order, to `TrainingState::resignWindow` and trims it to `resign_window`.
  It then calibrates the threshold for the **next** generation. So
  generation g's threshold depends only on games of generations before g:
  every game of a generation shares it, and the result does not depend on
  the worker count.

### Config

Six tunable keys, after `mcts_random_symmetry`:

| key | range | default | meaning |
|---|---|---|---|
| `resignation` | bool | false | self-play resigns |
| `resign_threshold_initial` | −1 to 0 | −0.9 | v_resign before calibration |
| `resign_disabled_fraction` | 0 to 1 | 0.1 | games played out to measure |
| `resign_false_positive` | 0 to 0.5 | 0.05 | false-positive limit |
| `resign_window` | 1 to 10⁶ | 500 | recent disabled games used |
| `resign_min_games` | 1 to 10⁶ | 20 | disabled games before calibrating |

- `resign_window` (500) is five generations of disabled games at 9×9's
  100 games per generation and 10%, so the threshold follows the network
  without jumping on one generation's handful of games.
- `resign_min_games` (20) is the smallest window where 5% is a whole game.
- A new cross-key check, `checkConsistency(cfg)`, runs in
  `Trainer::startNew` and `Trainer::resume` once overrides are applied.
  It refuses (`ConfigError`):
  - `resignation = true` with `ownership = true`: a resigned game has no
    final position to take ownership targets from;
  - `resign_min_games > resign_window`: the threshold could never be
    calibrated.
- The presets leave resignation off; `configs/9x9.cfg` lists the keys
  commented out.

### Training state

- `state.bin` version 5 appends an f32 `resignThreshold` (the v_resign
  the next generation uses) and the window: a u32 count, then the f32
  winner minimums.
- Versions 1–4 load with an empty window and the default threshold.
- `Trainer::resume` trims the window to `resign_window` and recalibrates
  the threshold from the window and the current keys. With unchanged keys
  this gives back the stored value, so resume stays byte-identical. With
  changed keys the new values take effect at once, and an old run gets
  `resign_threshold_initial`.

### Output

- `log.csv` gains three trailing columns:
  - `resign_threshold`: the v_resign this generation used (`%.2f`);
  - `resign_rate`: resigned games / games with resignation enabled;
    empty if every game was disabled;
  - `resign_false_positive`: the false-positive rate at that threshold
    over the window, this generation's disabled games included; empty
    while the window is empty.
  All three are empty without resignation. Resume gives an older
  `log.csv` (with or without `ownership_loss`) the missing columns, empty.
- The progress line adds `resign -0.87 rate 0.62 fp 0.031` after the
  self-play plies, with `-` for a rate that does not exist yet.
- Resigned games are written to SGF with `RE[B+R]` or `RE[W+R]`, through a
  new `gameToSgf` overload that takes RE as written. Disabled games say
  so in the SGF comment.

## Error handling

- Inconsistent keys are a `ConfigError` before anything is written, for a
  new run and for a resume. A refused resume leaves `config.cfg`
  untouched.
- A version-5 `state.bin` whose window count exceeds the file is an error
  that names the file.

## Testing

- **MCTS:** `bestChildValue` is the most visited edge's Q, with the best
  Q among ties, and 0 before any search.
- **Self-play:**
  - with an evaluator that thinks White is lost, White resigns at ply 1:
    one position, Black wins, z = +1, and Black's ply 0 is never
    considered;
  - a hopeless Black resigns at ply 2, never at the empty board;
  - a disabled game plays to the end and records the minimums;
  - resignation on draws exactly one extra number per game, first, so a
    game that never resigns equals the resignation-off game with one
    `uniform()` drawn beforehand. The existing determinism tests pass
    unchanged with resignation off.
- **Calibration** on hand-made data covers the grid answer at 5% and
  10%, the strict "below", the top and bottom of the grid, the
  `minGames` fallback and window trimming.
- **SGF:** `RE[B+R]` and `RE[W+R]`.
- **Config:**
  - the keys' defaults, order, round trip, bounds and tunability;
  - the ownership refusal and the `resign_min_games` rule;
  - presets off, with the 9×9 commented lines present.
- **State:** a version-5 round trip including the window; a version-4
  file loads with an empty window; an oversized window count is refused.
  Older-version tests are updated to the new trailing bytes.
- **Trainer (tiny 5×5 run):**
  - generation g uses the threshold calibrated after g−1;
  - the log columns match the statistics;
  - some SGF records a resignation and the threshold moves off its
    initial value;
  - resume is byte-identical and the worker count changes nothing with
    resignation on;
  - ownership with resignation is refused by `startNew` and `resume`;
  - resignation can be switched on at resume;
  - older logs gain the columns.
- **Integration smoke:**
  - a run with resignation shows it in the progress line, `log.csv` and
    SGF;
  - the ownership combination is refused.

## Teaching material

- Explainer chapter 04 gets a short section, "Resignation (optional)",
  citing AlphaGo Zero. It quotes the resignation block of
  `playSelfPlayGame` and `calibrateResignThreshold`, and explains why
  false positives poison the value targets. It adds a concept-map row.
  The HTML is mirrored and the citations are checked.
- README: the keys, how it works, and the log columns.
