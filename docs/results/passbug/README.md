# Case study: the pass blind spot

The four reference runs were first made with the code as it stood at
commit 57e7333. That code had a blind spot in its tree search around
passing. These files keep the logs of those runs, so the symptom, the
diagnosis and the fix can be checked against data:

| directory | run (pre-fix code, same presets and seeds as the final runs) |
|---|---|
| `5x5-seed1/`, `5x5-seed2/` | 60 generations, komi 24.5: `log.csv`, `elo.csv`, `results.csv` |
| `9x9-main/` | 150 generations, komi 7.5: `log.csv`, `elo.csv`, `results.csv` |
| `19x19-check/` | 2 generations: `log.csv` |
| `9x9-gen150-vs-gen0-loss.sgf` | a game lost by pre-fix generation 150 to generation 0 (the diagnosis' `g002_Aloss.sgf`; each move's comment gives the mover's search value `v` and the Tromp-Taylor margin `tt` from the mover's side) |

The final runs, made after the fix, are in `../5x5/`, `../5x5-seed2/`,
`../9x9/` and `../19x19/`.

## Symptom

The 9×9 Elo curve barely left the ground and then fell
(`9x9-main/elo.csv`). The refit peaked at 288.4 at generation 35 and
ended at 91.8. The untrained generation 0 kept winning: over the run it
scored 938 of the 3,000 arena games it played against later generations
(`9x9-main/results.csv`). That averaged 7.1 of 20 per match from
generation 50 on, and as many as 14 of 20 in a single match. A trained 9×9
network should beat a randomly initialized one nearly every time.

## Diagnosis

The investigation (2026-09-30) used 100-game matches and per-move
instrumentation on the pre-fix checkpoints. Its diagnosis report and the
later pass-fix report are working notes that are **not in this
repository**, and neither are the instrumented tools or their outputs
(only the one SGF above was kept). The numbers below are quoted from
those reports; the key tables follow at the end of this section, so the
case study stands on its own.

- **The losses have one shape.** gen 150 against gen 0 scored only 67/100
  (gen 30: 63/100, gen 92: 48/100). In every one of those losses:
  - the trained network passed while the Tromp-Taylor count was against it;
  - generation 0 passed back and won on the count.

  In this file's example game, gen 150 (Black) owns most of the board at
  ply 24. But ten of White's hopeless stones are still standing, so the
  margin is −4.5 for Black. Its search value is +0.93, it passes, and
  White passes back.
- **Why search missed it.** A probe of that position at 100 simulations:
  - the network puts 0.256 of its prior on pass;
  - after the pass, it gives the game-ending reply (White's pass) only
    0.0074 of the prior, among 59 legal replies;
  - with first-play urgency 0 (an unvisited edge counts as Q = 0) and
    edges tried in prior order, that reply was **never visited** (N = 0);
  - so the pass kept the network's optimistic Q of +0.88 and was played.

  At 800 simulations the reply got 4 visits, scored an exact +1 for White,
  and search chose E7 instead.
- **Why generation 0 exploits it.** It sees the same position at its own
  root. With near-flat priors and more simulations than legal moves, it
  tries pass, and pass scores an exact win. A uniform-prior evaluator did
  even better: gen 150 scored only 47/100 against it.
- **The network was not getting weaker.** Head-to-head, gen 150 beat
  gen 30 92/100 and gen 60 89/100.

  The Elo fall was an anchor effect. Every generation plays generation 0,
  so the exploit's 20-game results against generation 0 pulled every
  late rating down.

  It also grew with training. In the probe position:

  | gen | P(pass) | P(opponent's pass reply) |
  |---|---|---|
  | 60 | 0.019 | 0.018 |
  | 92 | 0.139 | 0.014 |
  | 150 | 0.256 | 0.007 |

- **Why 5×5 was immune.** A 5×5 node has at most 26 actions, so the pass
  reply always gets visited. Komi 24.5 also makes "won on the board, lost
  on the count" almost impossible.

### Key tables from the diagnosis

All matches 100 games, colors alternating, 100 simulations per move,
komi 7.5, no noise, 4 sampled opening plies, pre-fix checkpoints.

Trained generation (A) against generation 0, pre-fix search. In every
row, every A loss was a game where A passed while losing on the
Tromp-Taylor count and generation 0 made the final pass:

| A vs gen 0 | A score | A as Black | A as White |
|---|---|---|---|
| gen 10 | 77/100 | 48/50 | 29/50 |
| gen 30 | 63/100 | 50/50 | 13/50 |
| gen 60 | 76/100 | 31/50 | 45/50 |
| gen 92 | 48/100 | 7/50 | 41/50 |
| gen 100 | 60/100 | 15/50 | 45/50 |
| gen 150 | 67/100 | 18/50 | 49/50 |

Trained against trained, pre-fix search:

| match | score |
|---|---|
| gen 150 vs gen 30 | 92/100 |
| gen 150 vs gen 60 | 89/100 |
| gen 150 vs gen 100 | 68/100 |

The same checkpoints with the exact-pass rule patched into the search
(both sides), and other opponents:

| match | score |
|---|---|
| gen 30 / 60 / 100 / 150 vs gen 0, patched | 99/100 each |
| gen 150 vs gen 60, patched | 91/100 |
| gen 150 vs a uniform-prior evaluator, unpatched | 47/100 |
| gen 150 vs a uniform-prior evaluator, patched | 99/100 |
| gen 150 at 800 simulations vs gen 0 at 100, unpatched | 99/100 |
| gen 150 and gen 60 vs gen 0 with the committed fix (pass-fix report) | 100/100 each |

Search probe of the example position (ply 24 of the SGF above, gen 150
to move as Black):

| simulations | pass edge | opponent's pass reply | move chosen |
|---|---|---|---|
| 100 | N = 17, Q = +0.88 | N = 0 (prior 0.0074, 59 edges) | pass |
| 800 | N = 50, Q = +0.74 | N = 4, exact win for White | E7 |
| 3200 | N = 56, Q = +0.55 | N = 10 | E2 |
| 100, patched | N = 3, Q = −0.35 | N = 2, exact win for White | E7 |

## Fix

Commits 87b37ea and 5da9b90 make the fix: "score the game-ending pass edge
exactly while unvisited".

- When a node is expanded right after a pass, its own pass edge ends the
  game, so `Mcts::expand` stores that edge's exact value: the game
  result, +1, −1 or 0 from the chooser's side, given by the sign of the
  Tromp-Taylor score minus komi.
- `Mcts::selectEdge` uses that value instead of Q = 0 while the edge is
  unvisited. This is the `MctsEdge::unvisitedQ` field.
- It is not general first-play-urgency tuning. Every other unvisited edge
  still counts as Q = 0.
- `tests/test_mcts.cpp` covers it with two tests:
  - `test_known_terminal_pass_edge_has_exact_unvisited_q`;
  - `test_search_does_not_pass_into_a_tromp_taylor_loss`, a 7×7 position
    that fails on the old code with pass the most visited move.

With the fix, the **pre-fix** checkpoints beat generation 0 100/100 (gen
150 and gen 60, 100 simulations; the pass-fix report). The bad play was
in the search, not in the weights.

## Before and after

All four runs were then repeated from scratch with the fixed code. Same
presets, same seeds.

| | pre-fix (this directory) | final (after the fix) |
|---|---|---|
| 9×9 generation-0 score, all arena games | 938 / 3,000 | 72 / 3,000 (`../9x9/results.csv`) |
| 9×9 generation-0 score per match, gens 50–150 | mean 7.1 / 20, max 14 | mean 0.2 / 20, max 2 |
| 9×9 `elo.csv` peak, final | 288.4 (gen 35), 91.8 | 784.8 (gen 54), 674.5 |
| 9×9 gen 150 vs gen 0, 100 games | 67/100 (100 sims, diagnosis) | 100/100, 95% CI 0.963–1.000 (200 sims, `../9x9/final-arena.txt`) |
| 9×9 hours for 150 generations | 9.58 | 10.17 |
| 5×5 seed 1 `elo.csv` final | 1024.4 | 1117.5 (`../5x5/elo.csv`) |
| 5×5 seed 2 `elo.csv` final | 945.5 | 852.5 (`../5x5-seed2/elo.csv`) |
| 19×19 generation 1 / 2 time | 59.6 / 105.1 min (avg 150.8 / 282.4 plies) | 58.2 / 129.7 min (avg 145.0 / 364.5 plies) |

The 5×5 runs barely changed, as the diagnosis predicted. Both before and
after, generation 0 scores 0 or 1 out of 20 in almost every match, and
the curves have the same rise-then-plateau shape.

On 9×9 the fix turned the curve from noise around 100–300 into a clear
rise. Even so, the final `elo.csv` stops rising after about generation
50. `../README.md` shows with 100-game head-to-heads that this later
plateau is again an artifact of the rating method, not lost strength.

The longer 19×19 generation-2 games after the fix fit a player that no
longer passes into a lost count. With two generations of one run, that is
only a plausible reading.

## Lessons

1. A rating anchored on one fixed opponent measures "how well do you beat
   *that* opponent". A quirk that opponent can exploit drags every rating.
2. 20-game matches hide trends that 100-game head-to-heads show plainly.
3. Reading the actual games (SGF with per-move values) found the cause
   faster than any aggregate statistic. Every loss ended the same way.
