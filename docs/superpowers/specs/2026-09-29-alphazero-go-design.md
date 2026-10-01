# AlphaZero-Style Go Player — Design

Author: Mark Castelluccio, with design assistance from Claude (Anthropic).

## Purpose

A from-scratch C++17 implementation of AlphaZero (self-play + PUCT/MCTS
guided by a residual neural network) applied to Go, built as a
**teaching example**. It is the successor to `alphazero-tictactoe` and
follows the same philosophy: every moving part — rules, convolutions,
backprop, search, self-play, training, evaluation — is implemented
directly and explained in accompanying teaching material. It is not
intended to be a strong Go engine.

The board size is a runtime choice from 5×5 to 19×19, and the network's
depth (residual blocks) and width (filters) are configurable.

## Success Criteria

- All unit tests and the integration smoke test pass with both the
  naive and the BLAS compute backend.
- **5×5:** a full training run shows a clearly rising Elo curve, and the
  final network wins ≥ 90% against the generation-0 (untrained) network.
- **9×9:** an overnight run (~8–12 h on an Apple M4, 10 cores) shows a
  still-rising Elo curve, and `analyze` shows sensible shape (capturing
  stones in atari, making eyes).
- **19×19:** training runs correctly (smoke test plus a few
  generations). No strength claim is made.
- Results reported in the README and explainer are measured from real
  runs and reported honestly.
- `docs/algorithm-explained.{md,html}` exists, covers the chapters listed
  under Teaching Material, and its code citations pass
  `tools/check_citations.py`.
- No dependencies beyond the C++17 standard library, except an optional
  BLAS library behind a CMake switch that is off by default.

## Non-Goals

- A play interface (terminal play, GTP, GUI integration). Games can be
  watched as SGF files; positions can be inspected with `analyze`.
- Baseline bots (random/heuristic players) or external engines as
  yardsticks. Progress is measured by the checkpoint arena and Elo.
- GPU acceleration in v1. The design leaves two seams (compute
  `Backend`, MCTS `Evaluator`) so a Metal backend and a cross-thread
  batching evaluator can be added later without restructuring.
- Resignation, playout-cap randomization, first-play-urgency tuning (the exact scoring of a known
  terminal pass edge in Select is not FPU tuning),
  "don't fill own eye" heuristics, and other KataGo/Leela-era
  refinements. The explainer mentions them as possible extensions.
- Board-size-agnostic networks (global pooling heads). A checkpoint is
  tied to the board size it was trained for.
- Asynchronous actor/learner training or multi-process pipelines.

## Architecture

A single library `azgo` (`include/azgo/*.hpp`, `src/*.cpp`) plus four
executables in `apps/`. Training runs as **synchronous generations in
one process**:

```
for generation g = 1, 2, ...:
    1. self-play   W worker threads play G games with the current network
    2. train       S SGD steps on minibatches sampled from the replay buffer
    3. arena       every E generations: new network vs. earlier ones -> Elo
    4. checkpoint  weights + full training state written atomically
```

This is the AlphaGo Zero loop with gating optional (off by default,
which is the AlphaZero variant). It was chosen over an asynchronous
actor/learner design because it reads as a sequence of plain steps, is
reproducible from a seed, and resumes cleanly at generation boundaries;
worker threads still use all cores during self-play and arena, where
most time is spent.

### Components

**`board` — Go rules engine.**
- `Board` is a value type (cheap to copy for MCTS). Board size N is a
  runtime value in [5, 19]. Storage is a fixed 21×21 array with a
  one-point off-board sentinel border, so neighbor lookups need no
  bounds checks and one type serves every size.
- Actions are integers: `0 .. N²−1` are points (row-major), `N²` is
  **pass**. Pass is always legal.
- Captures: a move removes opponent groups left without liberties;
  then, if the moved stone's own group has no liberties, the move is
  illegal (**suicide is illegal** — a deliberate deviation from strict
  Tromp-Taylor, matching AlphaGo Zero).
- **Positional superko**: a move is illegal if it recreates any earlier
  whole-board position. Positions are tracked by incremental Zobrist
  hashes (one random 64-bit key per point × color, seeded
  deterministically). The history of hashes lives with the game.
- Game end: two consecutive passes, or a move cap (default `2·N²`),
  after which the position is scored as-is.
- **Tromp-Taylor area scoring**: stones plus empty regions that reach
  only one color. Komi from config (default 7.5). Outcome is Black win
  or White win (half-point komi prevents draws; an integer komi that
  produces a tie is scored as a draw with z = 0).
- Groups/liberties are computed by flood fill with reusable scratch
  buffers, not incrementally maintained chains — simpler to read, and
  board operations (µs) are never the bottleneck next to network
  evaluation (ms).
- API sketch: `Board(int n)`, `size()`, `toMove()`, `at(point)`,
  `isLegal(action)`, `legalActions()`, `play(action) -> Board`,
  `isTerminal()`, `score() -> float` (Black minus White minus komi),
  `hash()`. `Board::isLegal` checks occupancy and suicide only;
  superko needs position history, so it is checked by `Game` (and by
  MCTS, see below).

**`game` — a game in progress.** Holds the sequence of `Board`s (for
history planes and superko), the move list, and consecutive-pass
count, and a set of past position hashes. `Game::isLegal` adds the
superko check to `Board::isLegal`. `encode(H)` produces the network input for the current
position. Used by self-play, arena, `analyze`, and SGF export.

**`encode` / `symmetry` — network input and augmentation.**
- Input planes (from the side-to-move's perspective): for each of the
  last H positions (default H = 8), one plane of own stones and one of
  opponent stones; plus a "black to move" plane and a "white to move"
  plane. Total `2H + 2` planes of N×N. Exactly one of the two
  color planes is all ones, which gives zero-padded 3×3 convolutions a
  way to tell the board edge from empty points. Missing history before
  move 0 is zero planes.
- The 8 dihedral symmetries (rotations/reflections) as index maps over
  points; pass maps to pass. Applied consistently to planes and to the
  policy target.

**`tensor` / `backend` — numeric core.**
- `Tensor`: a contiguous float32 NCHW buffer with a shape.
- `Backend` interface (seam #1): `gemm` (with transpose flags),
  `im2col`, `col2im`. Implementations:
  - `NaiveBackend` (default): blocked loops. During the training phase
    it may split output rows across threads; each output element is
    computed by exactly one thread, so results are thread-count
    independent.
  - `BlasBackend` (CMake `-DAZ_USE_BLAS=ON`): `cblas_sgemm` from Apple
    Accelerate on macOS, OpenBLAS elsewhere. Forced single-threaded
    inside self-play/arena workers (our threads already occupy the
    cores); may use internal threading during the training phase.
- A future `MetalBackend` would be a third implementation; unified
  memory on Apple Silicon means buffers need not be copied.

**`layers` / `network` — the ResNet.**
- Layers: `Conv2d` (3×3 and 1×1, same padding, via im2col + gemm),
  `BatchNorm2d`, `ReLU`, `Linear`, residual add. Each has
  `forward(train)` and a hand-written `backward` using activations
  cached in the forward pass.
- Architecture (AlphaGo Zero shape):
  - stem: 3×3 conv, F filters → BN → ReLU
  - body: B residual blocks: conv → BN → ReLU → conv → BN → + skip → ReLU
  - policy head: 1×1 conv, 2 filters → BN → ReLU → Linear → N²+1 logits
  - value head: 1×1 conv, 1 filter → BN → ReLU → Linear(V) → ReLU →
    Linear(1) → tanh
- Configurable: `blocks` (B ≥ 0), `filters` (F), `value_hidden` (V),
  `history` (H). Suggested presets: 5×5 3×32, 9×9 6×64, 19×19 10×128.
- **Parameters vs. activations are separate**: `Network` owns weights and
  BN running statistics (shared read-only by worker threads);
  `Workspace` owns per-thread activation/scratch buffers.
- BN uses batch statistics in training mode and running statistics
  (momentum-updated) in inference mode.
- Loss: `(z − v)² + cross_entropy(π, softmax(logits))`; L2 weight decay
  (default 1e-4) is applied by the optimizer.

**`optimizer`** — SGD with momentum (default 0.9) and a stepped
learning-rate schedule keyed on generation. A separate class from the
network, as in tic-tac-toe. Its momentum buffers are part of the saved
training state.

**`evaluator`** — seam #2. Interface:
`evaluate(encoded input planes [B][planes][n*n]) -> {policy_logits[], values[]}`;
MCTS does the encoding, so an evaluator needs to know nothing about Go.
v1 ships
`LocalEvaluator`, owned by one worker thread, wrapping a shared
`Network` and a private `Workspace`. A future batching evaluator could
gather leaves from all threads for a GPU without touching MCTS.

**`mcts` — PUCT search.**
- Node storage: one contiguous `vector<Node>` arena; a node refers to
  its children by index range. Each child edge stores action, prior P,
  visit count N, value sum W, and virtual-loss count. Iterative, not
  recursive.
- Select: argmax over children of `Q + c_puct · P · √(ΣN) / (1 + N)`,
  `c_puct` default 1.5. Unvisited children use Q = 0, with one exception: the game-ending
  pass edge after a pass uses its exact Tromp-Taylor result (+1/−1/0 from
  the chooser's side) while unvisited; see `docs/results/passbug/README.md`
  for why.
- Each expanded node stores its `Board` and hash. Superko legality
  during search checks the game's past hashes plus the hashes on the
  current root-to-node path; history planes for a leaf are built by
  walking parent nodes and then the game's earlier boards.
- Expand: children only for legal actions; priors are a softmax over
  the legal logits only.
- Evaluate: terminal positions are scored exactly by the rules; others
  go to the `Evaluator`. Values are from the side-to-move's
  perspective and flip sign at each ply during backup.
- **Virtual-loss batching**: each round performs K descents (default
  K = 8), applying a virtual loss along each path so later descents
  diverge; the K leaves are evaluated in one `Evaluator` call; values
  are backed up and virtual losses removed. A leaf reached by more
  than one descent in a round is evaluated once. K = 1 is exactly the
  serial algorithm.
- Root Dirichlet noise (self-play only): `ε = 0.25`,
  `α = dirichlet_scale / N²` with `dirichlet_scale` default 10
  (≈0.03 at 19×19, 0.12 at 9×9, 0.4 at 5×5).
- Move choice: sample proportional to visit counts for the first T
  moves (`temperature_moves`, per preset, e.g. 8 / 20 / 30), then play
  the most-visited move (random tie-break). The policy target π is the
  normalized root visit counts.
- **Tree reuse**: after a move, the chosen child's subtree becomes the
  new root. Only the reachable subtree is copied into a second arena,
  then the arenas swap, bounding memory to one move's search. Fresh
  noise is applied at each new root in self-play.
- `simulations` per move from config (presets: 200 / 400 / 800).

**`selfplay`** — plays one game with a given `Evaluator`, RNG, and
config; returns a `GameRecord`: per position the board (one byte per
point), the side to move, sparse π (visited actions and counts), and,
after the game ends, z from that position's side-to-move perspective.

**`replay_buffer`** — stores whole `GameRecord`s in play order in a
sliding window capped at `buffer_positions` positions (oldest games
evicted first). A sample is a `(game, move)` pair; its history planes
are rebuilt from the same game's earlier positions, so history costs
no extra storage. Uniform sampling over positions. Each sample gets a
uniformly random symmetry. Serializable as part of training state.

**`arena` / `elo`**
- Every `arena_every` generations (default 1), the new network plays
  `arena_games` games per opponent (default 20, colors balanced)
  against: the previous generation; a sparse ladder of older
  generations (g−2, g−4, g−8, …, the most recent `arena_opponents`
  of them, default 4); and generation 0, the Elo anchor fixed at 0.
- Arena search: `arena_simulations` (config), no Dirichlet noise, tree
  reuse on. Opening diversity: temperature-1 sampling for the first
  `arena_temperature_moves` moves (default 2 on 5×5, 4 on larger
  boards) and random tie-breaks. The number of distinct games is
  logged as a diversity check (tic-tac-toe's arena once produced
  2 distinct games out of 40 with deterministic play).
- Elo: all pairwise results are kept in a results table (persisted).
  After each arena round all ratings are refit by maximum-likelihood
  Bradley-Terry (iterative MM algorithm) with generation 0 pinned at 0.
  Order-independent, so resume reproduces identical ratings. Also
  logged: win rate vs. the previous generation with a 95% Wilson
  interval.
- Optional gating (`gating = on`): self-play uses the "best" network;
  a new network replaces it only if it scores ≥ 55% against it.
  Default `off`.

**`config`** — a dependency-free `key = value` file format with `#`
comments. One table maps each key to its field, default, and
validator, so all settings are documented in one place. Unknown keys,
malformed values, and out-of-range values are errors naming the key.
Presets: `configs/5x5.cfg`, `configs/9x9.cfg`, `configs/19x19.cfg`.
Command-line `--set key=value` overrides. The effective config is
printed at startup and saved into the run directory. Keys are
classified as **architecture keys** (board size, history, blocks,
filters, value_hidden) or tunable keys.

**`checkpoint` / `run_state`**
- Checkpoint `gen_NNNN.azn`: magic `AZGO`, format version,
  architecture header (N, H, B, F, V), then tensors in a fixed order,
  each with its shape; float32 little-endian. Includes BN running
  statistics. Load validates everything and fails with a message
  naming the mismatch.
- Training state (`state.bin`, one file): optimizer momentum, replay
  buffer, generation counter, current "best" network pointer, arena
  results table. RNG state needs no saving because every random stream
  is seeded from (run seed, generation, phase, index). Written to a
  temporary file and renamed into place so a crash never leaves a
  partial state.
- Resume refuses if any architecture key differs from the saved
  config; tunable keys may be overridden.

**`sgf`** — writes a few self-play games per generation
(`sgf_per_generation`, default 2) as SGF files (board size, komi,
result, moves), viewable in any SGF viewer such as Sabaki. Also reads
simple SGF move sequences for `analyze`.

### Determinism and Threading

- Self-play and arena games are dispensed by index from an atomic
  counter to W worker threads (default: hardware concurrency). Each
  game's RNG is seeded from `(run_seed, generation, phase, game_index)`
  and results are stored by index, so output is identical for a given
  seed regardless of thread count.
- The training step runs on the main thread; backend-level parallelism
  never changes results (one thread per output element).
- A resumed run produces byte-identical results to an uninterrupted
  run.

### Executables

- **`train`** — `train --config configs/9x9.cfg --run runs/<name>
  [--set k=v ...] [--resume]`. Runs the generation loop.
- **`arena`** — plays two checkpoints against each other for M games
  and prints the score with a confidence interval. For manual
  experiments.
- **`analyze`** — given a checkpoint and a position (move list or SGF),
  prints an ASCII board with the network's policy as a heat map, its
  value estimate, and the MCTS visit distribution. The successor to
  tic-tac-toe's `diag_eval`; used heavily by the explainer.
- **`bench`** — measures network evaluations/second (naive vs. BLAS,
  per board size and preset) and board operations/second. Used to
  choose presets and to back README performance numbers with
  measurements.

## Data Flow

```
config ──► train
            │
            ├─ self-play (W threads) ── LocalEvaluator ─► Network (shared, read-only)
            │        │  MCTS ◄─► Game/Board
            │        ▼
            │   GameRecords ──► ReplayBuffer ──► sgf/
            │
            ├─ training (main thread)
            │   ReplayBuffer ─► sample + symmetry ─► encode ─► Network.forward/backward
            │                                                      │
            │                                          SgdMomentum.step ◄┘
            │
            ├─ arena (W threads) ─► results table ─► Bradley-Terry ─► elo.csv
            │
            └─ checkpoint ─► checkpoints/gen_NNNN.azn, state.bin (atomic), log.csv
```

Run directory `runs/<name>/`: `config.cfg`, `checkpoints/`, `state.bin`,
`log.csv` (per generation: games, average game length, pass rate, Black
win rate, policy loss, value loss, learning rate, replay ratio, phase
timings, Elo), `elo.csv` (generation, Elo, interval), `results.csv`
(raw pairwise results), `sgf/`.

## Error Handling

- Config: strict parsing; errors name the key and the accepted range;
  the program exits non-zero before doing any work.
- Checkpoint and state loading: magic, version, and architecture
  header are validated; mismatches (e.g. a 9×9 network in a 5×5 run)
  produce a clear message and non-zero exit.
- Resume with a changed architecture key is refused with a message
  listing the differing keys.
- Illegal actions passed to `Board::play` are a programming error
  (assert). External input (move lists, SGF in `analyze`) is validated
  and reported as a user error.
- Writes of training state are atomic (temp directory + rename).
- Non-finite loss during training aborts the run with the generation
  and step number, leaving the last good state intact.

## Testing

One assert-based test executable per component in `tests/test_*.cpp`,
registered with CTest. Test targets are compiled with `-UNDEBUG` so
asserts fire in Release builds (carried over from tic-tac-toe).

- **board**: single- and multi-stone captures; suicide illegal;
  capture-that-looks-like-suicide legal; simple ko; superko; edge and
  corner liberties; Tromp-Taylor scoring incl. seki and dame;
  random-playout invariants on every size 5–19 (incremental hash ==
  recomputed hash; no zero-liberty group on board; games terminate).
- **encode/symmetry**: plane contents and color planes; history at game
  start; every symmetry and its inverse compose to identity; planes and
  π transform consistently.
- **layers/network**: finite-difference gradient check of every layer
  and of a whole tiny network (5×5, B=1, F=4) in training mode; output
  shapes for every size 5–19; naive vs. BLAS backend agreement within
  tolerance (when BLAS is built).
- **optimizer**: momentum and weight-decay updates match hand-computed
  values.
- **mcts**: visits sum to simulations; virtual losses fully removed
  after each batch; K = 1 batching equals serial search; tree reuse
  preserves subtree statistics; noise only at root; with a
  uniform-prior evaluator, finds a single winning capture on a 5×5
  position.
- **selfplay**: z sign correct for both colors incl. komi-decided
  games; identical output for 1 and 4 threads with the same seed.
- **replay_buffer**: window eviction; history reconstruction at game
  start; serialization round-trip.
- **arena/elo**: Bradley-Terry recovers known ratings from synthetic
  results; 50/50 results give equal ratings; color balance; gating
  threshold; seeded arena games are distinct and reproducible.
- **config**: defaults, overrides, unknown keys, out-of-range values.
- **checkpoint/resume**: checkpoint round-trip; header mismatch
  rejected; 2 generations + resume + 2 generations is byte-identical to
  4 uninterrupted generations.
- **integration** (`tests/integration_smoke.sh`): 2 generations on 5×5
  with a tiny network, then `arena`, `analyze`, and `bench` run
  successfully.

## Build

CMake ≥ 3.16, C++17, default build type Release. Option `AZ_USE_BLAS`
(default OFF) links Accelerate on macOS or OpenBLAS elsewhere.
Layout:

```
include/azgo/   library headers
src/            library sources
apps/           train, arena, analyze, bench
tests/          test_*.cpp, integration_smoke.sh
configs/        5x5.cfg, 9x9.cfg, 19x19.cfg
tools/          check_citations.py
docs/           algorithm-explained.{md,html}, superpowers/{specs,plans}
```

`.gitignore`: `build/`, `runs/`, `.worktrees/`, `.DS_Store`.

## Teaching Material

`docs/algorithm-explained.md` and an HTML rendering, in the style of the
tic-tac-toe explainer: numbered chapters quoting real code with
`file:line — Symbol` captions, verified by `tools/check_citations.py`
(adapted from tic-tac-toe; standard library only; `--fix` rewrites line
numbers).

Chapters:

- **00 — The state**: Go rules for programmers (liberties, capture,
  suicide, ko/superko, Tromp-Taylor, komi); input planes and why the
  constant color plane matters.
- **01 — The two-headed guess**: from the tic-tac-toe MLP to a ResNet;
  convolution as im2col + gemm; batch norm train vs. inference;
  residual connections; heads; what depth and width buy; backprop
  through conv/BN; gradient checking.
- **02 — Search corrects the guess**: PUCT, legal masking, backup sign
  flip.
- **03 — Batching the search**: virtual loss, collisions, why batch size
  matters to gemm/BLAS, the `Evaluator` seam and where a GPU would go.
- **04 — Self-play & exploration**: Dirichlet α scaling with board size,
  temperature, tree reuse.
- **05 — Assigning credit**: z, komi, passes, why Tromp-Taylor requires
  dead stones to actually be captured.
- **06 — Learning from itself**: loss, SGD, the game-based replay
  buffer, symmetry augmentation, replay ratio.
- **07 — Measuring progress without an oracle**: arena, Bradley-Terry
  Elo, anchors, drift, intransitivity, confidence intervals, gating vs.
  no gating.
- **08 — The full loop at scale**: deterministic threading, compute
  budgets per board size, an honest comparison with AlphaGo Zero's
  19×19 compute.
- **09 — What changed from tic-tac-toe**: a side-by-side table.
- **10 — Concept map.**

Source headers open with teaching-oriented comment blocks. Results
quoted in the explainer and README (Elo curves, win rates, timings) come
from actual runs; the HTML includes the Elo plots.

The README covers build, usage, presets, measured results, and an
**Authorship** section crediting Mark Castelluccio as author and
acknowledging Claude (Anthropic) as an AI assistant used in design and
implementation. Commits carry a `Co-Authored-By` trailer for Claude.

## Extension Path: GPU

Not part of v1; recorded so the seams are designed for it.

1. `MetalBackend` implementing the `Backend` interface with Metal compute
   kernels (via `metal-cpp`) for gemm/im2col, then conv/BN/loss.
   Training steps (batch ~256) benefit first.
2. A batching `Evaluator` that gathers leaves from all worker threads
   into batches of 64–256 for the GPU. MCTS is unchanged.

On Apple Silicon, `BlasBackend` via Accelerate already uses the CPU's
matrix units (SME/AMX); the M4 GPU's additional gain is roughly 2× and
only at large batch sizes, which is why the GPU is deferred.
