# Gating-Only Arena — Design

Author: Mark Castelluccio, with design assistance from Claude (Anthropic).

Builds on `2026-10-01-rating-aware-arena-design.md` (the arena round and
its Elo ladder), `2026-10-10-arena-gpu-design.md` (waves) and
`2026-10-20-distributed-arena-design.md` (arena engines). A new tunable
key, `arena_ladder`, lets a gated run's arena play only what AlphaGo
Zero's evaluator played.

## Problem

With `gating = true` a generation's arena round plays more than the
gating match that decides promotion. The round (`runArenaRound`,
`src/arena_round.cpp`) plays, in one wave, g−1, the strongest reference,
`arena_opponents` more references and the best network; and when g scores
at least `promote_threshold` against the strongest reference, a second
wave of `calibration_games` against it. Only the match against the best
network decides anything for training: the rest feeds the internal Elo
ladder (`elo.csv`, `references.csv`, log.csv's `elo`).

The live 19×19 run (`arena_games = 40`, `arena_opponents = 0`, no fixed
players) shows the cost. Generation 4 (rejected, best 3) played
`vs 3:0.47 0:0.60`, 80 games, where the gating match is the 40 against 3.
Generation 5 (promoted, best 3) played `vs 4:0.42 0:0.68 3:0.55`, 120
games, where the gating match is again the 40 against 3. At about an hour
per 40-game slice of work on all four machines, the ladder is half to two
thirds of the arena: roughly 30 minutes or more per generation, more after
a promotion into the reference set (80 more games).

## AlphaGo Zero

AlphaGo Zero's evaluator (Silver et al. 2017, Methods, "Evaluator")
played each new checkpoint against the current best network αθ*, 400
games, and made it the new αθ* when it won by a margin of more than 55%.
Nothing else was played for training: its Elo curves came from separate
evaluation tournaments, not from the evaluator. This project's arena
round is the evaluator plus a running ladder; `arena_ladder = off` keeps
the evaluator only.

## The key

`arena_ladder = on | off`, default `on`, tunable on resume (not locked,
not architecture), printed after `gating_threshold` in `config.cfg`.

- `on`: today's round, unchanged. Every file of a run is byte-identical
  to the code before the key, except `config.cfg`, which gains the line
  `arena_ladder = on`.
- `off`: each arena generation plays **one wave, one match**: generation g
  against the best network, `arena_games` games, seed
  `deriveSeed(seed, g, kPhaseArena, best)`, g as player a. These are
  exactly the seed and colours today's round gives its gating match, so
  game k of the match is the same game it would have been with the ladder
  on (a match's result never depends on which others share its wave).
  Promotion is `passesGate(score, arena_games, gating_threshold)`, as
  today. No calibration, no g−1 / strongest / extra opponents, no
  promotion into the reference set.

`arena_ladder = off` with `gating = false` is refused by
`checkConsistency` (at start and on resume): without gating there is no
gating match, so the arena would play nothing. A run that wants no arena
sets `arena_every = 0`, which already means that.

`arena_every` still applies: with `arena_every = k`, the gating match is
played at every k-th generation and the best network can only change
there, as today.

## What a ladder-off generation writes

- **state.bin**: the gating match is added to the results table (the
  table is the record of every arena game); `bestGeneration` as today;
  `references` unchanged.
- **results.csv**: rewritten from the table, so it gains the gating games
  as the pair (best, g).
- **elo.csv, references.csv**: *not rewritten*. They keep the last ladder
  refit. A Bradley-Terry fit of gating games alone would chain each
  generation to the best network with 40 games per link: an Elo curve,
  but not one worth printing. When the ladder is switched back on, the
  next refit includes the gating games (they are real results), so the
  generations rated meanwhile appear there, rated through those games.
- **log.csv**: `elo` is **empty** for every generation run with the
  ladder off (arena or not). `winrate_vs_prev` and `distinct_vs_prev`
  keep their meaning, the match against g−1: filled when the best network
  is g−1 (after a promotion), empty otherwise. log.csv has no column for
  the score against the best network in either mode; it is in
  results.csv and on the progress line.
- **progress line**: `arena gating vs 3:0.55 (distinct 40/40) 812.3s |
  best 5`, the distinct count being the gating match's; a generation
  without an arena prints `no arena | best 3`.
- **arena-engines.csv** (with `arena_engines`): one wave per generation,
  split by weight as any wave; generation 0 has no rows when the run
  started with the ladder off (no calibrations, below).
- **GenerationStats::elo** is NaN for a ladder-off generation.

A run started with `arena_ladder = off` skips the calibrations of the
fixed players against generation 0 at start: they only rate the ladder.
If the ladder is switched on later, the first round calibrates them (the
existing `calibrateFixedPlayers` call at the top of every round), with the
same seeds, so they are the same games a run started with the ladder on
played.

## Switching and resume

The key is read each generation, so a resume may switch it either way at
any commit. A switch prints a note. Self-play and training never read the
results table or the reference set: the arena influences them only
through `bestGeneration`, and the gating decision is the same with the
ladder on or off. So **checkpoints, self-play games, SGFs and every
promotion decision are identical** whether a run kept the ladder on,
kept it off, or switched in between; only the arena's own files differ.

- off → on: the round runs as today on a results table that also holds
  the gating-only generations' games; fixed players not yet calibrated
  are calibrated first.
- on → off: references and elo.csv stay as the last ladder round left
  them.
- Pipeline on or off: both loops rate through `trainAndRate`, unchanged.
- `selfplay_allocation = dynamic`: unaffected (the arena is not part of
  it).
- `arena_engines`: the single wave is split across the engines as any
  wave. `arenaWaveNetworks` (the most networks a wave holds, used to
  budget GPU memory) is 2 with the ladder off: the new network and the
  best.

## Re-rating

`train --rerate` rebuilds the ladder from the checkpoints alone: it never
reads the run's results table, ignores gating, and plays the full round
for every generation. It ignores `arena_ladder` too (forced on for the
replay, including the GPU memory check), so a run trained with the ladder
off gets a complete `rerate/` ladder, identical to the one the same run
with the ladder on would get, since their checkpoints are identical.

## Tests

- Config: default `on`, parse/print, round trip, tunable on resume,
  `off` without gating refused.
- `runGatingMatch` (`arena_round`): one wave of one match with the live
  round's seed, added to the table, references untouched, no calibration,
  refused without a best network.
- Trainer, 5×5: a ladder-off run plays only the gating pairs (results.csv),
  leaves elo.csv/references.csv as at start, empty `elo` in log.csv; its
  checkpoints and best network equal a ladder-on run's and each gating
  score equals the ladder-on run's score against the same best; resume is
  byte-identical; on→off→on switching keeps checkpoints and decisions and
  calibrates late with the same games; pipeline on; arena engines (local
  and fake remote) equal the plain run; `--rerate` of a ladder-off run
  equals that of the ladder-on run.
- Default byte-identity: the existing suite, plus main's and this
  branch's `train` on the same configs (5×5 and 9×9; naive, blas, metal;
  with and without a fake-remote arena engine; straight and resumed),
  every file compared except config.cfg's new line, timings and build IDs.
