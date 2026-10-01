# Rating-Aware Arena Opponents — Design

Author: Mark Castelluccio, with design assistance from Claude (Anthropic).

Amends the arena section of
`docs/superpowers/specs/2026-09-29-alphazero-go-design.md`.

## Problem

The arena plays each new network against g−1, a sparse ladder (g−2, g−4,
g−8, …) and generation 0. Once a run is strong, generation 0 loses almost
every game. A 20–0 match carries almost no information: the information
in a game with win probability p is proportional to p(1−p). The
Bradley-Terry fit's virtual draw also caps a 20–0 pair at
400·log₁₀(41) ≈ 645 Elo. In the 9×9 reference run, generation 0 scored
≤ 2/20 in 137 matches (0/20 in 110). `elo.csv` therefore sat at about
600–735 from generation 80 to 150. Meanwhile 100-game head-to-heads
showed steady progress: gen 150 beat gen 50 98/100 and gen 120 67/100.
The rating scale saturated; the networks did not.

## Goal

Pick opponents whose results are informative: an expected score of about
0.6–0.85 for the new network. Keep one fixed zero point (generation 0 =
0 Elo) and add a fixed external yardstick that does not change across
runs.

## Non-Goals

- Changing training. With gating off, the arena never affects self-play
  or SGD, so checkpoints are unchanged by this work.
- Other rating systems (Glicko, TrueSkill). Bradley-Terry with virtual
  draws stays.
- External engines (GNU Go). The uniform-prior MCTS bot below is the only
  fixed yardstick.

## Design

### Players

- **Checkpoints** are identified by their generation number (≥ 0), as
  now.
- **Bots** are MCTS players with uniform priors (all logits 0) and value
  0. They search with the arena's settings (no noise, the same opening
  temperature and tree reuse), but at a fixed simulation count.
  - New config key `bot_simulations`: a comma-separated list of positive
    integers, or empty for no bots. Default `100` in every preset.
  - Bot k (1-based position in the list) has player ID −k.
  - Bot results go in the same `ResultsTable` as everything else, so
    `ResultsTable` and its CSV accept negative IDs (a ≠ b still holds).
  - Generation 0 stays pinned at 0 Elo.

### Reference set

- The reference set is the frozen set of players that new networks are
  measured against.
- It is stored in `state.bin`:
  - `TrainingState` gains `std::vector<int> references`, kept sorted
    ascending.
  - `TrainingState` also gains `std::vector<int> botSimulations`, the
    bot list the run rated. It is stored right after the references, so
    resume can refuse a `config.cfg` whose list was edited by hand.
  - The state format moves to version 2.
  - A version-1 file loads with references = {0} ∪ {all configured bots}
    and adopts the config's bot list.
- **Initial set:** {0} ∪ {every bot}.
- **Calibration at run start:** `Trainer::startNew` plays each bot
  `calibration_games` games against generation 0 and records the results,
  so every bot's rating is known before it can be chosen.
  - New config key `calibration_games`: even, 2–10000, default 80.
  - A resumed version-1 run does the same calibration on its first
    arena generation.
- **Promotion:** after generation g's arena, if g scored ≥
  `promote_threshold` against the strongest reference, then:
  - g plays `calibration_games` extra games against that same reference,
    and those are added to the results;
  - g is added to the reference set.

  New config key `promote_threshold`: 0.5–1, default 0.85.
- **Strongest reference:** the reference with the highest rating in the
  pre-arena refit, ties broken toward the larger player ID.
- Gen 0 and bots are never removed. Saturated references simply stop
  being selected (see below), and their past results stay in the fit.

### Opponent selection

```
selectOpponents(g, ratings, references, maxExtra) -> std::vector<int>
```

Deterministic. `ratings` is the Bradley-Terry refit of all results so
far, so it is identical on resume. The new network's estimated rating R̂
is the rating of g−1, or 0 if g−1 is unrated.

1. **g−1, always.** It is the progress signal, and `distinct_vs_prev`
   still uses it.
2. **The strongest reference, always.** This is the promotion check.
3. **Up to `maxExtra` (= `arena_opponents`) more references,** excluding
   those already chosen.
   - For each candidate c, the expected score is
     E(c) = 1 / (1 + 10^((R_c − R̂) / 400)).
   - Candidates are sorted by |E(c) − 0.7| ascending, ties going to the
     smaller player ID, and the first `maxExtra` are taken.
   - Unrated references are skipped.
4. **The best network,** if gating is on and it is not already chosen
   (unchanged behaviour).

The result has no duplicates. Order is g−1, then the strongest reference,
then the extras in selection order, then the best network.

`arena_opponents` keeps its name but now counts the extra references,
with default 2 and range 0–32. All three presets set it to 2.
`ladderOpponents` is removed.

### Arena players

- `playMatch` becomes
  `playMatch(const ArenaPlayer& a, const ArenaPlayer& b, …)`, where
  `struct ArenaPlayer { const Network* net; int botSimulations; }`.
  - With `net`, the player searches with the network at
    `ArenaConfig::simulations`.
  - With `net == nullptr`, the player is a uniform bot searching with
    `botSimulations`.
  - Board size and history are read from whichever player has a network.
    Two bots, or two networks with different `NetConfig`s, throw
    `std::invalid_argument`.
- The existing `playMatch(const Network&, const Network&, …)` stays as a
  thin wrapper.
- The `arena` app gains `--b-bot <sims>` as an alternative to `--b`.

### Re-rating existing runs

- `train --run DIR --rerate` replays only the arena phase over the
  checkpoints that already exist (gen 1 … the last checkpoint) with the
  current selection rules. It does not train and does not touch
  `state.bin`.
- It starts from an empty results table and the initial reference set,
  runs bot calibration, then for each generation in order does selection,
  matches, refit and promotion, with the same seeds as live training
  (`deriveSeed(seed, g, kPhaseArena, opponent)`).
- It writes `DIR/rerate/{results.csv, elo.csv, references.csv, log.csv}`.
  That `log.csv` has one row per generation:
  `generation,opponents,score_vs_prev,score_vs_strongest,promoted,elo`.
- Gating is ignored: no best-network opponent is used.
- It is valid because, with gating off, the arena never affects training,
  so the checkpoints are exactly what a live run would produce.

### Outputs

- `elo.csv` keeps its format, generations only, so `plot_elo.py` and the
  docs keep working.
- New `references.csv` with header `player,kind,simulations,promoted_at,elo`:
  - `kind` is `checkpoint` or `bot`;
  - `simulations` is the bot's count, or empty for a checkpoint;
  - `promoted_at` is the generation at which it joined, or 0 for the
    initial set.

  It is rewritten after each refit.
- The progress line lists the opponents played, e.g.
  `vs 149:0.55 120:0.70 bot100:1.00`.
- The bot ratings appear only in `references.csv`.

## Error handling

- `bot_simulations` entries must be positive integers ≤ 100000.
  Malformed lists are a `ConfigError`.
- `bot_simulations` cannot change on resume: bots are identified by
  position, so a changed list would make the same IDs mean different
  players. Resume refuses the change with a `ConfigError` that names the
  key, the same way it treats architecture keys. The check compares the
  list recorded in `state.bin` with the effective config, so it also
  catches a hand edit of `config.cfg` made without `--set`.
- A `state.bin` version other than 1 or 2 is rejected with the path in
  the message, as now.

## Testing

- **selectOpponents:** g−1 and the strongest reference are always
  present. The extras are chosen by |E − 0.7|, with deterministic
  tie-breaks. There are no duplicates. Unrated references are skipped.
  With gating, the best network is appended. A synthetic rating set where
  gen 0 is 900 Elo below R̂ does not select gen 0 as an extra.
- **Promotion:** a score ≥ threshold against the strongest reference
  adds g and plays exactly `calibration_games` extra games. A score below
  threshold does neither.
- **Bots:** a bot-vs-network match is deterministic for a seed and
  independent of the worker count. Bot-vs-bot and mismatched-config
  networks throw.
- **Elo and CSV:** negative IDs round-trip through CSV and fit correctly.
- **State:** version-2 round trip. A version-1 file loads with the
  default reference set.
- **Trainer:** on a tiny run whose network quickly beats gen 0, gen 0
  stops appearing among the opponents after it saturates. The
  resume-byte-identity and worker-count-independence tests still pass.
  Bot calibration rows exist after `startNew`.
- **Re-rate:** on a tiny completed run, `--rerate` produces the same
  `results.csv` as a fresh live run with the same config. It is also
  deterministic across worker counts.
- **Config:** new keys parse, validate and round-trip. Resume refuses a
  changed `bot_simulations`.

## Teaching material and results

- **Explainer chapter 07:** replace the ladder section with "choosing
  informative opponents". Tie it to the saturation evidence (the p(1−p)
  argument and the 645-Elo cap), show `selectOpponents`, and update the
  concept map.
- **Spec:** update the arena section of the original spec to point here.
  Update the README and the preset comments.
- **Results:** run `train --run runs/9x9-main --rerate` and publish
  `docs/results/9x9/rerate/`. Add a chart overlaying the old `elo.csv`,
  the refit without gen 0, and the re-rated curve. The 9×9 success
  criterion is judged on the re-rated `elo.csv` and reported as
  measured, with the old verdict kept for comparison. The two 5×5 runs
  may be re-rated the same way, at about 20 minutes each.
