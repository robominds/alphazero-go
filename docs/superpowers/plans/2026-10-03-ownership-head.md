# Ownership Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add KataGo's auxiliary ownership target as an opt-in third head (n² dense life-and-death labels per position), carry it through checkpoints, self-play records, the replay buffer, training, `grow` and `analyze`, teach it in the explainer, and measure it against the matching 200-simulation 9×9 run.
**Architecture:** `Board::ownership` scores every point of the final board (Tromp-Taylor, Black's view) and `areaScore` becomes its sum. Self-play stores it per game (`GameRecord::finalOwnership`); the replay buffer turns it to the side to move's view and moves it with the same symmetry as the input planes. A 1×1 conv → BN → tanh head, present only when `NetConfig::ownership` is set, adds `w · mean((ô − o)²)` to the loss. Search never reads it. Old checkpoints load as "no head", old buffers recompute ownership by replaying their moves, and `grow --ownership` adds a head without changing policy or value.
**Tech Stack:** C++17, CMake ≥ 3.16, the C++ standard library and `std::thread`; optional BLAS behind `-DAZ_USE_BLAS=ON`; assert-based test executables run by CTest; bash for the integration smoke test; Python 3 standard library for the docs tooling.
**Spec:** docs/superpowers/specs/2026-10-03-ownership-head-design.md
**Author:** Mark Castelluccio, with planning assistance from Claude (Anthropic).

## Global Constraints

- C++17, CMake ≥ 3.16, default build type Release; no dependencies beyond the C++ standard library, except optional BLAS behind `AZ_USE_BLAS` (default OFF).
- Namespace `azgo`; public headers in `include/azgo/`, sources in `src/`, apps in `apps/`, tests in `tests/`. Tasks 1–8 add no file and need no CMake edit: every change goes into an existing file, and the suite stays at 26 CTest tests (25 component tests plus `integration_smoke`).
- Tests: one assert-based executable per component, `tests/test_<component>.cpp`, compiled with `-UNDEBUG` so asserts fire in Release. No test framework. The library itself is compiled with `NDEBUG`, so an `assert` in `src/` documents a precondition but checks nothing in Release.
- All randomness goes through `azgo::Rng`, with streams seeded by `deriveSeed(...)`. Never `std::random_device`, `std::mt19937` or `std::*_distribution`.
- Determinism: self-play, training, arena and rating output must be identical for a given seed and backend, whatever the worker count, with or without the head. A resumed run continues byte-identically (`test_resume_is_byte_identical` keeps passing, and Task 5 adds the same check with the head).
- **A network without the head is unchanged, bit for bit**: the same parameters for a seed, the same outputs, the same gradients, the same checkpoint bytes, and the same training run (the head's random numbers are drawn after every other layer's, and sampling the ownership target draws none).
- **Two perspectives, never mixed:** stored ownership (`Board::ownership`, `GameRecord::finalOwnership`, the `analyze` map) is from **Black's** view as `int8_t` −1/0/+1; everything the network reads or predicts (targets, `ô`) is from the **side to move's** view, like `z` and `v`.
- Formats: checkpoint version 1 (no head) or 2 (head; i32 flag after the shape); `state.bin` version 3 (versions 1 and 2 still load); `log.csv` gains a trailing `ownership_loss` column, empty without the head.
- Config: `ownership` (bool, default false, architecture key, printed after `value_hidden`); `ownership_weight` (float 0–10, default 1.5, tunable, printed after `weight_decay`). Presets are unchanged.
- Each changed header keeps its opening teaching comment accurate.
- Code style: camelCase functions, PascalCase types, `trailing_` members, `kConstants`, 4-space indent, ≤ 100 columns as in the surrounding code.
- Commits: author Mark Castelluccio <markacastelluccio@gmail.com> (repo-local config, already set); imperative subject without a conventional-commit prefix; body explains why; message ends with `Co-Authored-By: Claude <model> <noreply@anthropic.com>`, naming the model that writes the commit (the blocks below show `Claude Opus 5.5`; substitute your own model name).

## Review Focus

1. **The head's gradient.** BN and tanh per point, the `w / (B · n²)` scaling and the head's share of the trunk gradient are all hand-written backward code. One wrong factor still trains, just badly, and nothing else would notice. Test: Task 3 `test_whole_network_gradient_check`, which now runs the whole-network finite-difference check a second time with the head on and `w = 1.5` (writing the tanh derivative `1 − ô²` as `1 + ô²` fails it at `stem.conv.w`).
2. **The target's perspective and symmetry.** A target left in Black's view for White's positions says every living group is dead; one not moved with the planes labels the wrong points. Both train without any error. Tests: Task 4 `test_ownership_target_perspective_and_symmetry` (checks the target against `encodePlanes` for all 8 symmetries and both colors) and `test_ownership_target_with_territory` (an asymmetric board, checked row by row by hand).
3. **Data from before this change.** The experiment's donor buffer is a `state.bin` version 2, and an old run's `log.csv` has 15 columns. Recomputing must give exactly what self-play would have recorded, a game that does not replay must be an error that names the file, and a resumed old run must keep `log.csv` rectangular. Tests: Task 4 `test_version2_recomputes_ownership` and `test_old_layout_that_does_not_replay`; Task 5 `test_resume_run_from_before_ownership`.
4. **A network without the head really is unchanged.** It must keep the same weights for a seed, outputs, gradients, checkpoint bytes and widening, and `w = 0` must leave training exactly as it was. Tests: Task 2 `test_ownership_flag_and_versions`; Task 3 `test_ownership_head_params_and_outputs` and `test_ownership_weight_zero_trains_nothing`; Task 6 `test_widening_carries_the_ownership_head` (its last block).
5. **`--init-from` with and without a head.** A checkpoint whose head matches the config must be accepted (a first draft of this plan refused it, and only the smoke test caught that), and a mismatch must say how to fix it. Tests: Task 5 `test_ownership_mismatches_are_refused`, and the `grow --ownership` → `train --init-from` block of the smoke test in Task 6.

---

## Notes Before You Start

**The task breakdown is the requested one.** Only one thing moves: the spec asks the explainer section to "show the ownership map from the experiment", and that map exists only after Task 9's run. Task 8 therefore writes the section without measured numbers, and Task 9 adds the map and the verdict to it.

**Spec ambiguities resolved here (each is binding for the tasks below):**
- *Parameter count.* The spec's "`filters` + 2 + 2" counts the running statistics. The head has `filters` + 2 trainable parameters (the 1×1 conv weights, BN gamma and beta) plus 2 BN running statistics (buffers): 98 + 2 on the 6×96 network.
- *The head's layers.* 1×1 conv (`filters` → 1, no bias) → BN → tanh at every point. There is no ReLU and no linear layer: each point's owner is read off the trunk's features at that point.
- *Checkpoint versions.* `saveNetwork` writes version 1 for a network without the head: those files stay byte-identical to before, and older builds can still read them. A network with the head is written as version 2, whose header is version 1's followed by an i32 flag (1). A version-2 file with flag 0 is read as "no head", and any other flag is a corrupt header.
- *`LossStats::ownershipLoss`* is the **unweighted** per-point mean squared error, so logs of runs with different weights compare. The loss that is trained is `w` times it. It is 0 when no target is given.
- *`w = 0`.* The backward pass skips the head entirely, so the head gets no gradient and the trunk gets no contribution from it. The head's forward pass still runs, and in a training step its BN running statistics update. The loss is still reported.
- *Initialization order.* `Network(cfg, seed)` initializes the head after every other layer, and `widenFilters` draws the head's weight shares last. So adding the head changes no other weight, for a seed, a widening seed or a growth.
- *The target.* `finalOwnership` is the ownership of the board after the last move: the board the game was scored on, including a game ended by the move cap. `TrainingBatch::ownership` is always filled, because encoding it draws no random numbers; only a network with the head is given it.
- *Old buffers.* `replayFinalOwnership` replays each game with `Game`. It checks every stored position against the replayed one as well as each move's legality, and either failure is a `std::runtime_error` (`replay buffer: game K does not replay: move 2 (B4) is illegal`), to which `loadTrainingState` prefixes the path.
- *`log.csv` of an older run.* `Trainer::resume` adds the column: the old header gains `,ownership_loss` and every old row a trailing empty field. Nothing else in the file changes.
- *`--init-from` messages.* When only the head differs, the messages are `<ckpt> has no ownership head but the config sets ownership = true; add one with: grow --from <ckpt> --ownership --out NEW.azn` and `<ckpt> has an ownership head but the config sets ownership = false; train it with --set ownership=true`. Any other difference keeps the shape message, which now ends in `, ownership head` for a network with the head.
- *`grow --ownership`* refuses a checkpoint that already has a head. Combined with `--filters` and `--blocks`, it widens, then deepens, then adds the head. The equivalence check also compares the ownership maps when both networks have a head.
- *The `analyze` map.* It is printed right after the value line, under the heading `Ownership, Black + / White - (9 = certain, "." below 0.1):`. Each cell is `%+3ld` of `lround(9 · ô)` in Black's terms, or `  .` when |ô| < 0.1. Stones are not drawn, because whether a stone survives is what the map shows. `analyze` reads the head through `Network::infer`; the `Evaluator` interface (MCTS, arena) is unchanged.
- *The experiment's comparisons.* `playMatch` requires equal `NetConfig`s, so the 9x9-sims200 checkpoints are compared through copies with a head added by `grow --ownership`. The head cannot change what a network plays.

## File Structure

```
include/azgo/board.hpp, src/board.cpp               Task 1   Board::ownership; areaScore = its sum
include/azgo/network.hpp, src/network.cpp           Task 2   NetConfig::ownership (in operator==)
                                                    Task 3   the head: Workspace buffers, params/buffers, infer/trainStep/computeLoss
include/azgo/config.hpp, src/config.cpp             Task 2   keys ownership, ownership_weight
include/azgo/checkpoint.hpp, src/checkpoint.cpp     Task 2   format version 2 (flag), version 1 = no head
include/azgo/selfplay.hpp, src/selfplay.cpp         Task 4   GameRecord::finalOwnership, replayFinalOwnership
include/azgo/replay_buffer.hpp, src/replay_buffer.cpp  Task 4   TrainingBatch::ownership, target encoding, save/load with or without it
include/azgo/run_state.hpp, src/run_state.cpp       Task 4   state.bin version 3
include/azgo/trainer.hpp, src/trainer.cpp           Task 5   training with targets, log column, progress line, --init-from messages, log upgrade
include/azgo/grow.hpp, src/grow.cpp, apps/grow.cpp  Task 6   addOwnershipHead, widenFilters carries the head, grow --ownership
apps/analyze.cpp                                    Task 7   the ownership map
tests/test_board.cpp                                Task 1
tests/test_config.cpp, tests/test_checkpoint.cpp    Task 2
tests/test_network.cpp                              Task 3
tests/test_selfplay.cpp, test_replay_buffer.cpp, test_run_state.cpp   Task 4
tests/test_trainer.cpp                              Tasks 4, 5
tests/test_grow.cpp                                 Task 6
tests/integration_smoke.sh                          Tasks 5, 6, 7
docs/algorithm-explained.md, .html, README.md       Task 8; Task 9 adds the measured results
runs/ownership.sh (not in git), docs/results/9x9-ownership/, docs/results/README.md   Task 9 (operational)
```

## Conventions for Every Task

**Build and test commands** (from the repo root):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure -R test_board    # one component
ctest --test-dir build --output-on-failure                  # everything
```

BLAS variant (run it at the end of every code task):

```bash
cmake -S . -B build-blas -DCMAKE_BUILD_TYPE=Release -DAZ_USE_BLAS=ON
cmake --build build-blas -j && ctest --test-dir build-blas --output-on-failure
```

`ctest` does not build: always run `cmake --build build -j` first. A test whose code refers to something that does not exist yet fails at **compile** time. That is the expected "fails" state in step 2 of a TDD cycle (`cmake --build` prints the error and the target is not built). The error texts quoted below are Apple clang's; GCC words them differently, but names the same symbol.

**Edits.** "Replace" steps quote the exact current text. It must occur exactly once in the file. Replace it verbatim and keep the surrounding code. Several replacements in one file are listed in file order, and each one quotes the text as it is after the previous ones. A replaced block often keeps a few unchanged lines around the change so that it is unique.

**Commit command shape:**

```bash
git add <files>
git commit -F - <<'EOF'
<Imperative subject line>

<Why this change, what it enables, anything surprising.>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 1: Who owns each point of a final position

**Files:**
- Modify: `include/azgo/board.hpp`, `src/board.cpp`
- Test: `tests/test_board.cpp`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `void Board::ownership(int8_t* out) const`: writes `area()` values, row-major, from Black's view. +1 is a Black stone or an empty point whose region reaches only Black, −1 the White equivalents, 0 an empty point whose region reaches both colors or none.
  - `int Board::areaScore() const` keeps its signature and is now the sum of `ownership`.

`areaScore` already flood-fills every empty region; `ownership` is the same fill writing each region's owner to its points instead of adding up its size.

- [ ] **Step 1: Write the failing tests**

The hand examples are the spec's: the empty board, a single stone, dame and seki, plus a dead stone, which Tromp-Taylor still counts. The random-game test compares every point with an independent oracle that uses only `Board`'s public interface. It also checks Σ ownership = `areaScore`, which becomes true by construction in Step 3, but which states the contract.

In `tests/test_board.cpp`, replace:

```cpp
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>
#include "azgo/board.hpp"

using namespace azgo;

```

with:

```cpp
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "azgo/board.hpp"
#include "azgo/game.hpp"
#include "azgo/rng.hpp"

using namespace azgo;

```

In `tests/test_board.cpp`, replace:

```cpp
    assert(fill.play(pt(1, 1)) == 3);
}

static void test_gtp_coordinates() {
    for (int n : {9, 19}) {
        for (int a = 0; a <= n * n; ++a) assert(parseAction(actionToString(a, n), n) == a);
```

with:

```cpp
    assert(fill.play(pt(1, 1)) == 3);
}

// ownership() as rows of '+' (Black), '-' (White) and '0' (nobody), row 0 first.
static std::vector<std::string> ownershipRows(const Board& b) {
    std::vector<int8_t> own(b.area(), 99);
    b.ownership(own.data());
    std::vector<std::string> rows;
    for (int r = 0; r < b.size(); ++r) {
        std::string row;
        for (int c = 0; c < b.size(); ++c) {
            const int8_t o = own[r * b.size() + c];
            assert(o == -1 || o == 0 || o == 1);
            row += o > 0 ? '+' : o < 0 ? '-' : '0';
        }
        rows.push_back(row);
    }
    return rows;
}

static void test_ownership_hand_examples() {
    // The empty board: one region that touches no stone belongs to nobody.
    assert((ownershipRows(Board(5)) == std::vector<std::string>(5, "00000")));

    // One Black stone: the single region reaches only Black.
    Board one(5);
    one.setStone(pt(2, 2), Color::Black);
    assert((ownershipRows(one) == std::vector<std::string>(5, "+++++")));

    // Add a White stone: the region now touches both, so only stones count.
    Board two = one;
    two.setStone(pt(0, 0), Color::White);
    assert((ownershipRows(two) == std::vector<std::string>{"-0000", "00000", "00+00", "00000",
                                                           "00000"}));

    // Dame: column D touches both walls and belongs to nobody.
    Board dame = boardFromRows({"..X.O",
                                "..X.O",
                                "..X.O",
                                "..X.O",
                                "..X.O"}, Color::Black);
    assert((ownershipRows(dame) == std::vector<std::string>(5, "+++0-")));

    // Seki: the shared liberties B5 and B4 touch both colors; Black's right
    // column and bottom row reach only Black. Sum: 18 - 5 = 13 = areaScore.
    Board seki = boardFromRows({"X.OX.",
                                "X.OX.",
                                "OOOX.",
                                "XXXX.",
                                "....."}, Color::Black);
    assert((ownershipRows(seki) == std::vector<std::string>{"+0-++", "+0-++", "---++", "+++++",
                                                            "+++++"}));

    // A dead stone still counts: the lone White stone is surrounded by Black's
    // area but uncaptured, so it is White's and the region it touches is
    // nobody's. Below the wall everything is Black's. Score -1 + 15 = 14.
    Board dead = boardFromRows({".O...",
                                ".....",
                                "XXXXX",
                                ".....",
                                "....."}, Color::Black);
    assert((ownershipRows(dead) == std::vector<std::string>{"0-000", "00000", "+++++", "+++++",
                                                            "+++++"}));
    assert(dead.areaScore() == 14);
}

// An independent oracle: for each empty point, a breadth-first search over
// empty neighbors through the public interface only, noting which colors
// the region touches.
static int oracleOwner(const Board& b, int point) {
    if (b.at(point) == Color::Black) return 1;
    if (b.at(point) == Color::White) return -1;
    const int n = b.size();
    std::vector<int> queue{point};
    std::vector<bool> seen(b.area(), false);
    seen[point] = true;
    bool black = false, white = false;
    for (size_t i = 0; i < queue.size(); ++i) {
        const int r = queue[i] / n, c = queue[i] % n;
        const int nbrs[4][2] = {{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}};
        for (const auto& rc : nbrs) {
            if (rc[0] < 0 || rc[0] >= n || rc[1] < 0 || rc[1] >= n) continue;
            const int q = rc[0] * n + rc[1];
            if (b.at(q) == Color::Black) black = true;
            else if (b.at(q) == Color::White) white = true;
            else if (!seen[q]) {
                seen[q] = true;
                queue.push_back(q);
            }
        }
    }
    return black == white ? 0 : (black ? 1 : -1);
}

static void test_ownership_sums_to_area_score_on_random_games() {
    Rng rng(2026);
    int games = 0, withNeutral = 0;
    for (int n : {5, 7, 9}) {
        for (int g = 0; g < 40; ++g) {
            // Random legal play, passing 5% of the time, until two passes or the move cap.
            Game game(n, 7.5f, defaultMaxMoves(n));
            while (!game.isTerminal()) {
                const std::vector<int> legal = game.legalActions();
                const bool pass = legal.size() == 1 || rng.uniform() < 0.05;
                game.play(pass ? n * n : legal[static_cast<size_t>(rng.below(static_cast<int>(legal.size()) - 1))]);
            }
            const Board& b = game.board();
            std::vector<int8_t> own(b.area());
            b.ownership(own.data());
            int sum = 0;
            bool neutral = false;
            for (int p = 0; p < b.area(); ++p) {
                assert(own[p] == oracleOwner(b, p));
                sum += own[p];
                neutral = neutral || own[p] == 0;
            }
            assert(sum == b.areaScore());
            ++games;
            withNeutral += neutral ? 1 : 0;
        }
    }
    assert(games == 120 && withNeutral > 0);   // the oracle saw neutral points too
}

static void test_gtp_coordinates() {
    for (int n : {9, 19}) {
        for (int a = 0; a <= n * n; ++a) assert(parseAction(actionToString(a, n), n) == a);
```

In `tests/test_board.cpp`, replace:

```cpp
    test_liberties();
    test_hash_after_matches_play();
    test_area_score();
    test_gtp_coordinates();
    test_board_from_rows_round_trip();
    test_set_stone_keeps_hash_valid();
```

with:

```cpp
    test_liberties();
    test_hash_after_matches_play();
    test_area_score();
    test_ownership_hand_examples();
    test_ownership_sums_to_area_score_on_random_games();
    test_gtp_coordinates();
    test_board_from_rows_round_trip();
    test_set_stone_keeps_hash_valid();
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build build -j`
Expected: the build fails in `tests/test_board.cpp` with:

```
tests/test_board.cpp:262:7: error: no member named 'ownership' in 'azgo::Board'
```

- [ ] **Step 3: Declare `ownership`**

In `include/azgo/board.hpp`, replace:

```cpp
    uint64_t hash() const { return hash_; }  // Zobrist over stones only (positional superko)
    uint64_t computeHashFromScratch() const; // test oracle for the incremental hash

    // Tromp-Taylor area score: (Black stones + empty regions reaching only Black)
    // minus (White stones + empty regions reaching only White). No komi.
    int areaScore() const;

    void exportStones(uint8_t* out) const;   // area() bytes of Color values, row-major
```

with:

```cpp
    uint64_t hash() const { return hash_; }  // Zobrist over stones only (positional superko)
    uint64_t computeHashFromScratch() const; // test oracle for the incremental hash

    // Who owns each point under Tromp-Taylor area rules, from Black's point of
    // view: +1 for a Black stone or an empty point whose region reaches only
    // Black, -1 for the White equivalents, 0 for an empty point whose region
    // reaches both colors (dame, seki liberties) or none (the empty board).
    // Writes area() values, row-major. A stone counts for its color even if
    // it is hopelessly dead: Tromp-Taylor only counts what is on the board.
    void ownership(int8_t* out) const;

    // Tromp-Taylor area score, the sum of ownership(): (Black stones + empty
    // regions reaching only Black) minus (White stones + empty regions
    // reaching only White). No komi.
    int areaScore() const;

    void exportStones(uint8_t* out) const;   // area() bytes of Color values, row-major
```

- [ ] **Step 4: Implement it, and `areaScore` as its sum**

The fill keeps the region's cells in `region[0, size)` (rather than popping them off a stack, as `areaScore` did) so that it can write the owner to each of them afterwards.

In `src/board.cpp`, replace:

```cpp
    return h;
}

int Board::areaScore() const {
    // Every stone counts for its color. Every maximal empty region is flood
    // filled once; it counts for a color only if all stones it touches have
    // that color (seki liberties and dame touch both and count for nobody).
    std::array<int, kCells> stack;
    std::array<uint8_t, kCells> mark{};
    int black = 0, white = 0;
    for (int point = 0; point < area(); ++point) {
        const int start = pad(point);
        const Color c = cells_[start];
        if (c == Color::Black) { ++black; continue; }
        if (c == Color::White) { ++white; continue; }
        if (mark[start]) continue;
        int top = 0, regionSize = 0;
        bool reachesBlack = false, reachesWhite = false;
        stack[top++] = start;
        mark[start] = 1;
        while (top > 0) {
            const int q = stack[--top];
            ++regionSize;
            for (int d : kDirs) {
                const int r = q + d;
                if (cells_[r] == Color::Black) reachesBlack = true;
                else if (cells_[r] == Color::White) reachesWhite = true;
                else if (cells_[r] == Color::Empty && !mark[r]) {
                    mark[r] = 1;
                    stack[top++] = r;
                }
            }
        }
        if (reachesBlack && !reachesWhite) black += regionSize;
        if (reachesWhite && !reachesBlack) white += regionSize;
    }
    return black - white;
}

void Board::exportStones(uint8_t* out) const {
```

with:

```cpp
    return h;
}

void Board::ownership(int8_t* out) const {
    // Every stone belongs to its color. Every maximal empty region is flood
    // filled once and belongs to a color only if all the stones it touches
    // have that color (seki liberties and dame touch both and belong to nobody).
    // region[0, size) lists the region's padded cells in the order they were
    // found; region[next, size) is the part still to be expanded.
    std::array<int, kCells> region;
    std::array<uint8_t, kCells> mark{};
    for (int point = 0; point < area(); ++point) {
        const int start = pad(point);
        const Color c = cells_[start];
        if (c == Color::Black) { out[point] = 1; continue; }
        if (c == Color::White) { out[point] = -1; continue; }
        if (mark[start]) continue;   // written with the region that reached it first
        int size = 0;
        bool reachesBlack = false, reachesWhite = false;
        region[size++] = start;
        mark[start] = 1;
        for (int next = 0; next < size; ++next) {
            for (int d : kDirs) {
                const int r = region[next] + d;
                if (cells_[r] == Color::Black) reachesBlack = true;
                else if (cells_[r] == Color::White) reachesWhite = true;
                else if (cells_[r] == Color::Empty && !mark[r]) {
                    mark[r] = 1;
                    region[size++] = r;
                }
            }
        }
        const int8_t owner = reachesBlack == reachesWhite ? 0 : (reachesBlack ? 1 : -1);
        for (int k = 0; k < size; ++k)   // padded index -> point (undoes pad())
            out[(region[k] / kStride - 1) * n_ + (region[k] % kStride - 1)] = owner;
    }
}

int Board::areaScore() const {
    std::array<int8_t, kMaxBoardSize * kMaxBoardSize> owner;
    ownership(owner.data());
    int score = 0;
    for (int point = 0; point < area(); ++point) score += owner[point];
    return score;
}

void Board::exportStones(uint8_t* out) const {
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_board`
Expected: `100% tests passed out of 1`. Then the whole suite and the BLAS variant (Conventions). Expected: `100% tests passed out of 26` in both builds. `areaScore` is used by MCTS terminal values and by every game result, so this run confirms that the refactor changed no score.

- [ ] **Step 6: Commit**

```bash
git add include/azgo/board.hpp src/board.cpp tests/test_board.cpp
git commit -F - <<'EOF'
Score every point of a position, and make areaScore its sum

The ownership target needs, for every point of the final board, who owns
it under Tromp-Taylor rules. areaScore already flood-filled each empty
region to decide that and then only kept the total. Board::ownership
writes the per-point answer (Black's view: +1, -1, 0 for dame and seki
liberties), and areaScore sums it, so the score and the target can never
disagree.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 2: The `ownership` and `ownership_weight` keys, and checkpoint version 2

**Files:**
- Modify: `include/azgo/network.hpp`, `src/network.cpp` (only `NetConfig`), `include/azgo/config.hpp`, `src/config.cpp`, `include/azgo/checkpoint.hpp`, `src/checkpoint.cpp`
- Test: `tests/test_config.cpp`, `tests/test_checkpoint.cpp`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `bool NetConfig::ownership = false`, compared by `NetConfig::operator==`. Task 3 gives it a meaning; until then a network with the flag has no extra tensors.
  - `bool Config::ownership = false` (key `ownership`, an architecture key, parsed like `gating`), `float Config::ownershipWeight = 1.5f` (key `ownership_weight`, 0–10, tunable). `Config::netConfig()` copies the flag.
  - Checkpoints: version 1 (no head, exactly as before) or version 2 (shape, then an i32 flag). `readCheckpointConfig` and `loadNetwork` read both; version 1 is `ownership = false`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_config.cpp`, replace:

```cpp
    assert((resumeLockedDiff(a, b) == std::vector<std::string>{"bot_simulations"}));
}

static void test_errors_name_key_and_line() {
    std::string msg = errorOf("komi = 7.5\nfoo = 1\n");
    assert(contains(msg, "t.cfg:2") && contains(msg, "foo") && contains(msg, "unknown key"));
```

with:

```cpp
    assert((resumeLockedDiff(a, b) == std::vector<std::string>{"bot_simulations"}));
}

static void test_ownership_keys() {
    Config c;
    assert(!c.ownership && c.ownershipWeight == 1.5f && !c.netConfig().ownership);
    // ownership sits with the network keys, ownership_weight with the training keys.
    assert(contains(configToString(c), "value_hidden = 64\nownership = false\n"));
    assert(contains(configToString(c), "weight_decay = 0.0001\nownership_weight = 1.5\n"));

    c = parseConfigText("ownership = on\nownership_weight = 0\n", "t.cfg");
    assert(c.ownership && c.ownershipWeight == 0.0f && c.netConfig().ownership);
    assert(contains(configToString(c), "ownership = true\n"));
    assert(contains(configToString(c), "ownership_weight = 0\n"));
    assert(configToString(parseConfigText(configToString(c), "rt")) == configToString(c));
    assert(parseConfigText("ownership_weight = 10", "t.cfg").ownershipWeight == 10.0f);
    assert(contains(errorOf("ownership = maybe"), "t.cfg:1: ownership: "));
    assert(contains(errorOf("ownership_weight = -0.5"), "t.cfg:1: ownership_weight: "));
    assert(contains(errorOf("ownership_weight = 10.5"), "t.cfg:1: ownership_weight: "));

    // The head changes the parameter set, so ownership is an architecture key
    // (and NetConfig tells the two shapes apart); its weight is tunable.
    Config a, b;
    b.ownership = true;
    assert((architectureDiff(a, b) == std::vector<std::string>{"ownership"}));
    assert((resumeLockedDiff(a, b) == std::vector<std::string>{"ownership"}));
    assert(a.netConfig() != b.netConfig());
    b = a;
    b.ownershipWeight = 3.0f;
    assert(architectureDiff(a, b).empty() && resumeLockedDiff(a, b).empty());

    // A config.cfg written before these keys existed leaves ownership off,
    // and the presets do not turn it on.
    assert(!parseConfigText("filters = 96\n", "old.cfg").ownership);
    for (const char* name : {"5x5.cfg", "9x9.cfg", "19x19.cfg"}) {
        const Config p = loadConfigFile(preset(name));
        assert(!p.ownership && p.ownershipWeight == 1.5f);
    }
}

static void test_errors_name_key_and_line() {
    std::string msg = errorOf("komi = 7.5\nfoo = 1\n");
    assert(contains(msg, "t.cfg:2") && contains(msg, "foo") && contains(msg, "unknown key"));
```

In `tests/test_config.cpp`, replace:

```cpp
    test_arena_keys();
    test_bot_simulations_rejects_bad_lists();
    test_resume_locked_diff();
    test_errors_name_key_and_line();
    test_hand_edited_files();
    std::printf("test_config: all tests passed\n");
```

with:

```cpp
    test_arena_keys();
    test_bot_simulations_rejects_bad_lists();
    test_resume_locked_diff();
    test_ownership_keys();
    test_errors_name_key_and_line();
    test_hand_edited_files();
    std::printf("test_config: all tests passed\n");
```

In `tests/test_checkpoint.cpp`, replace:

```cpp
    assert(!sameTensors(first, reloaded));
}

int main() {
    gDir = fs::temp_directory_path() / ("azgo_test_checkpoint_" + std::to_string(getpid()));
    fs::remove_all(gDir);
```

with:

```cpp
    assert(!sameTensors(first, reloaded));
}

// A network without an ownership head is saved exactly as before (version 1,
// no flag); one with the head as version 2 with the flag after the shape.
static void test_ownership_flag_and_versions() {
    NetConfig withHead = smallConfig();
    withHead.ownership = true;
    Network plain(smallConfig(), 1), owner(withHead, 1);
    const std::string plainPath = pathFor("plain.azn"), ownerPath = pathFor("owner.azn");
    saveNetwork(plain, plainPath);
    saveNetwork(owner, ownerPath);
    const std::string p = readFile(plainPath), o = readFile(ownerPath);
    auto u32At = [](const std::string& bytes, size_t at) {   // little-endian, as written
        uint32_t v = 0;
        for (int k = 3; k >= 0; --k) v = (v << 8) | static_cast<uint8_t>(bytes[at + k]);
        return v;
    };

    // Version 1: the tensor count follows the 5 shape fields directly.
    const size_t header = 4 + 4 + 5 * 4;
    assert(u32At(p, 4) == 1);
    assert(u32At(p, header) == plain.params().size() + plain.buffers().size());
    assert(!readCheckpointConfig(plainPath).ownership);

    // Version 2: an i32 flag of 1, then the tensor count.
    assert(u32At(o, 4) == 2 && u32At(o, header) == 1);
    assert(u32At(o, header + 4) == owner.params().size() + owner.buffers().size());
    assert(readCheckpointConfig(ownerPath) == withHead);
    Network loaded = loadNetwork(ownerPath);
    assert(loaded.config() == withHead && sameTensors(owner, loaded));

    // A version-2 file whose flag is 0 is a plain network.
    std::string v2plain = p;
    v2plain[4] = 2;
    v2plain.insert(header, std::string(4, '\0'));
    const std::string v2Path = pathFor("v2plain.azn");
    writeFile(v2Path, v2plain);
    assert(readCheckpointConfig(v2Path) == smallConfig());
    Network fromV2 = loadNetwork(v2Path);
    assert(sameTensors(plain, fromV2));

    // Any other flag is a corrupt header.
    std::string badFlag = o;
    badFlag[header] = 2;
    const std::string badPath = pathFor("bad_flag.azn");
    writeFile(badPath, badFlag);
    expectLoadError(badPath, "invalid ownership flag");
}

int main() {
    gDir = fs::temp_directory_path() / ("azgo_test_checkpoint_" + std::to_string(getpid()));
    fs::remove_all(gDir);
```

In `tests/test_checkpoint.cpp`, replace:

```cpp
    test_header_does_not_match_tensors();
    test_truncated_checkpoint_names_file();
    test_leftover_tmp_is_ignored_and_overwritten();
    fs::remove_all(gDir);
    std::printf("test_checkpoint: all tests passed\n");
    return 0;
```

with:

```cpp
    test_header_does_not_match_tensors();
    test_truncated_checkpoint_names_file();
    test_leftover_tmp_is_ignored_and_overwritten();
    test_ownership_flag_and_versions();
    fs::remove_all(gDir);
    std::printf("test_checkpoint: all tests passed\n");
    return 0;
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cmake --build build -j`
Expected: the build fails in both test files, for example with:

```
tests/test_config.cpp:239:15: error: no member named 'ownership' in 'azgo::Config'
tests/test_config.cpp:239:30: error: no member named 'ownershipWeight' in 'azgo::Config'
```

and, in `tests/test_checkpoint.cpp`, `error: no member named 'ownership' in 'azgo::NetConfig'`.

- [ ] **Step 3: Add the flag to `NetConfig`**

In `include/azgo/network.hpp`, replace:

```cpp
    int blocks = 6;        // residual blocks, >= 0
    int filters = 64;
    int valueHidden = 64;
    int planes() const { return 2 * history + 2; }
    int area() const { return boardSize * boardSize; }
    int actions() const { return area() + 1; }
```

with:

```cpp
    int blocks = 6;        // residual blocks, >= 0
    int filters = 64;
    int valueHidden = 64;
    bool ownership = false;   // an ownership head: who owns each point at the end
    int planes() const { return 2 * history + 2; }
    int area() const { return boardSize * boardSize; }
    int actions() const { return area() + 1; }
```

In `src/network.cpp`, replace:

```cpp

bool NetConfig::operator==(const NetConfig& o) const {
    return boardSize == o.boardSize && history == o.history && blocks == o.blocks &&
           filters == o.filters && valueHidden == o.valueHidden;
}

// ---------------------------------------------------------------- Workspace
```

with:

```cpp

bool NetConfig::operator==(const NetConfig& o) const {
    return boardSize == o.boardSize && history == o.history && blocks == o.blocks &&
           filters == o.filters && valueHidden == o.valueHidden && ownership == o.ownership;
}

// ---------------------------------------------------------------- Workspace
```

- [ ] **Step 4: Add the config keys**

In `include/azgo/config.hpp`, replace:

```cpp
    int blocks = 6;
    int filters = 64;
    int valueHidden = 64;
    // self-play and search
    int gamesPerGeneration = 100;
    int simulations = 200;
```

with:

```cpp
    int blocks = 6;
    int filters = 64;
    int valueHidden = 64;
    bool ownership = false;            // ownership head (KataGo's auxiliary target)
    // self-play and search
    int gamesPerGeneration = 100;
    int simulations = 200;
```

In `include/azgo/config.hpp`, replace:

```cpp
    std::string lrSchedule = "0:0.02";
    float momentum = 0.9f;
    float weightDecay = 1e-4f;
    // arena
    int arenaEvery = 1;                // 0 = never
    int arenaGames = 20;               // per opponent, even
```

with:

```cpp
    std::string lrSchedule = "0:0.02";
    float momentum = 0.9f;
    float weightDecay = 1e-4f;
    float ownershipWeight = 1.5f;      // weight w of the ownership loss; 0 = the head does not train
    // arena
    int arenaEvery = 1;                // 0 = never
    int arenaGames = 20;               // per opponent, even
```

In `include/azgo/config.hpp`, replace:

```cpp
// through parseConfigText exactly.
std::string configToString(const Config& cfg);

// Keys that fix the network's shape: board_size, history, blocks, filters, value_hidden.
std::vector<std::string> architectureDiff(const Config& a, const Config& b);

// Keys a resume must refuse to change: architectureDiff, then bot_simulations.
```

with:

```cpp
// through parseConfigText exactly.
std::string configToString(const Config& cfg);

// Keys that fix the network's shape: board_size, history, blocks, filters,
// value_hidden, ownership.
std::vector<std::string> architectureDiff(const Config& a, const Config& b);

// Keys a resume must refuse to change: architectureDiff, then bot_simulations.
```

In `src/config.cpp`, replace:

```cpp
    {"blocks", true, intKey<&Config::blocks, 0, kMaxBlocks>, printInt<&Config::blocks>},
    {"filters", true, intKey<&Config::filters, 1, kMaxFilters>, printInt<&Config::filters>},
    {"value_hidden", true, intKey<&Config::valueHidden, 1, 1024>, printInt<&Config::valueHidden>},
    // self-play and search
    {"games_per_generation", false, intKey<&Config::gamesPerGeneration, 1, 1000000>,
     printInt<&Config::gamesPerGeneration>},
```

with:

```cpp
    {"blocks", true, intKey<&Config::blocks, 0, kMaxBlocks>, printInt<&Config::blocks>},
    {"filters", true, intKey<&Config::filters, 1, kMaxFilters>, printInt<&Config::filters>},
    {"value_hidden", true, intKey<&Config::valueHidden, 1, 1024>, printInt<&Config::valueHidden>},
    {"ownership", true, [](Config& c, const std::string& v) { return parseBool(v, c.ownership); },
     [](const Config& c) { return std::string(c.ownership ? "true" : "false"); }},
    // self-play and search
    {"games_per_generation", false, intKey<&Config::gamesPerGeneration, 1, 1000000>,
     printInt<&Config::gamesPerGeneration>},
```

In `src/config.cpp`, replace:

```cpp
    {"weight_decay", false,
     [](Config& c, const std::string& v) { return parseFloat(v, 0, 1, c.weightDecay); },
     printFloat<&Config::weightDecay>},
    // arena
    {"arena_every", false, intKey<&Config::arenaEvery, 0, 1000000>, printInt<&Config::arenaEvery>},
    {"arena_games", false, [](Config& c, const std::string& v) { return parseEvenGames(v, c.arenaGames); },
```

with:

```cpp
    {"weight_decay", false,
     [](Config& c, const std::string& v) { return parseFloat(v, 0, 1, c.weightDecay); },
     printFloat<&Config::weightDecay>},
    {"ownership_weight", false,
     [](Config& c, const std::string& v) { return parseFloat(v, 0, 10, c.ownershipWeight); },
     printFloat<&Config::ownershipWeight>},
    // arena
    {"arena_every", false, intKey<&Config::arenaEvery, 0, 1000000>, printInt<&Config::arenaEvery>},
    {"arena_games", false, [](Config& c, const std::string& v) { return parseEvenGames(v, c.arenaGames); },
```

In `src/config.cpp`, replace:

```cpp
    net.blocks = blocks;
    net.filters = filters;
    net.valueHidden = valueHidden;
    return net;
}

```

with:

```cpp
    net.blocks = blocks;
    net.filters = filters;
    net.valueHidden = valueHidden;
    net.ownership = ownership;
    return net;
}

```

- [ ] **Step 5: Write and read checkpoint version 2**

In `include/azgo/checkpoint.hpp`, replace:

```cpp

namespace azgo {

// Format: magic "AZGO" (u32 0x4F475A41), version u32 = 1, NetConfig as 5 x i32
// (boardSize, history, blocks, filters, valueHidden), u32 tensor count, then
// per tensor: name (str), u64 size, floats. Params in params() order, then
// buffers in buffers() order. Saved atomically.
void saveNetwork(Network& net, const std::string& path);

// Throws std::runtime_error naming the problem: missing file, bad magic,
```

with:

```cpp

namespace azgo {

// Format: magic "AZGO" (u32 0x4F475A41), version u32, NetConfig as 5 x i32
// (boardSize, history, blocks, filters, valueHidden), in version 2 followed by
// the ownership flag (i32, 0 or 1), then u32 tensor count, then per tensor:
// name (str), u64 size, floats. Params in params() order, then buffers in
// buffers() order. Saved atomically. A network without an ownership head is
// written as version 1, byte for byte what older builds wrote (and can still
// read); one with the head as version 2. Version 1 loads as ownership = false.
void saveNetwork(Network& net, const std::string& path);

// Throws std::runtime_error naming the problem: missing file, bad magic,
```

In `src/checkpoint.cpp`, replace:

```cpp
namespace {

constexpr uint32_t kMagic = 0x4F475A41;   // bytes 'A' 'Z' 'G' 'O' when written little-endian
constexpr uint32_t kVersion = 1;

// A problem found in the file's contents (as opposed to running out of data).
struct FormatError : std::runtime_error {
```

with:

```cpp
namespace {

constexpr uint32_t kMagic = 0x4F475A41;   // bytes 'A' 'Z' 'G' 'O' when written little-endian
// Version 2 added the ownership flag to the header.
constexpr uint32_t kVersionPlain = 1, kVersionOwnership = 2;

// A problem found in the file's contents (as opposed to running out of data).
struct FormatError : std::runtime_error {
```

In `src/checkpoint.cpp`, replace:

```cpp
NetConfig readHeader(BinaryReader& r) {
    if (r.u32() != kMagic) throw FormatError("not an AZGO checkpoint");
    const uint32_t version = r.u32();
    if (version != kVersion) throw FormatError("unsupported version " + std::to_string(version));
    NetConfig cfg;
    cfg.boardSize = r.i32();
    cfg.history = r.i32();
    cfg.blocks = r.i32();
    cfg.filters = r.i32();
    cfg.valueHidden = r.i32();
    // Same ranges as the config file accepts; also keeps a corrupt header
    // from asking for an absurd allocation.
    if (cfg.boardSize < 5 || cfg.boardSize > 19 || cfg.history < 1 || cfg.history > 16 ||
```

with:

```cpp
NetConfig readHeader(BinaryReader& r) {
    if (r.u32() != kMagic) throw FormatError("not an AZGO checkpoint");
    const uint32_t version = r.u32();
    if (version != kVersionPlain && version != kVersionOwnership)
        throw FormatError("unsupported version " + std::to_string(version));
    NetConfig cfg;
    cfg.boardSize = r.i32();
    cfg.history = r.i32();
    cfg.blocks = r.i32();
    cfg.filters = r.i32();
    cfg.valueHidden = r.i32();
    if (version == kVersionOwnership) {
        const int32_t flag = r.i32();
        if (flag != 0 && flag != 1) throw FormatError("invalid ownership flag in header");
        cfg.ownership = flag == 1;
    }
    // Same ranges as the config file accepts; also keeps a corrupt header
    // from asking for an absurd allocation.
    if (cfg.boardSize < 5 || cfg.boardSize > 19 || cfg.history < 1 || cfg.history > 16 ||
```

In `src/checkpoint.cpp`, replace:

```cpp
    BinaryWriter w(buffer);
    const NetConfig& cfg = net.config();
    w.u32(kMagic);
    w.u32(kVersion);
    w.i32(cfg.boardSize);
    w.i32(cfg.history);
    w.i32(cfg.blocks);
    w.i32(cfg.filters);
    w.i32(cfg.valueHidden);
    const std::vector<Tensor> tensors = tensorsOf(net);
    w.u32(static_cast<uint32_t>(tensors.size()));
    for (const Tensor& t : tensors) {
```

with:

```cpp
    BinaryWriter w(buffer);
    const NetConfig& cfg = net.config();
    w.u32(kMagic);
    w.u32(cfg.ownership ? kVersionOwnership : kVersionPlain);
    w.i32(cfg.boardSize);
    w.i32(cfg.history);
    w.i32(cfg.blocks);
    w.i32(cfg.filters);
    w.i32(cfg.valueHidden);
    if (cfg.ownership) w.i32(1);
    const std::vector<Tensor> tensors = tensorsOf(net);
    w.u32(static_cast<uint32_t>(tensors.size()));
    for (const Tensor& t : tensors) {
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R 'test_config|test_checkpoint'`
Expected: `100% tests passed out of 2`. Then the whole suite and the BLAS variant. Expected: `100% tests passed out of 26` in both builds. The trainer tests write `config.cfg` with the two new lines and resume from it, so they cover the round trip through a run directory.

- [ ] **Step 7: Commit**

```bash
git add include/azgo/network.hpp src/network.cpp include/azgo/config.hpp src/config.cpp \
        include/azgo/checkpoint.hpp src/checkpoint.cpp tests/test_config.cpp tests/test_checkpoint.cpp
git commit -F - <<'EOF'
Add the ownership config keys and checkpoint format version 2

The ownership head changes a network's parameter set, so its switch is
an architecture key that resume refuses to change, and checkpoints
must record it. Version 2 adds the flag after the shape. A network
without the head is still written as version 1, so those checkpoints
stay byte-identical and older builds can read them. ownership_weight
(default 1.5) is tunable.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 3: The ownership head

**Files:**
- Modify: `include/azgo/network.hpp`, `src/network.cpp`
- Test: `tests/test_network.cpp`

**Interfaces:**
- Consumes: `NetConfig::ownership` (Task 2).
- Produces:
  - Parameters `ownership.conv.w` (`filters` values, decayed), `ownership.bn.gamma`, `ownership.bn.beta` after every other parameter, and buffers `ownership.bn.runMean`, `ownership.bn.runVar` after every other buffer, only when `cfg.ownership`.
  - `void Network::infer(Workspace&, const float* input, int B, float* logits, float* values, float* ownership = nullptr) const`. With a non-null pointer it writes `[B][area]` values in [−1, 1] from the side to move's view; that needs the head (asserted).
  - `LossStats Network::trainStep(Workspace&, const float* input, const float* pi, const float* z, int B, const float* ownership = nullptr, float ownershipWeight = 0.0f)` and `computeLoss` with the same parameters. The targets are `[B][area]` in [−1, 1], side to move's view. The loss adds `w · mean over B·area of (ô − t)²`; with `w = 0` the head gets no gradient.
  - `float LossStats::ownershipLoss`: the unweighted mean squared error per point, 0 without a target.
  - Every existing caller compiles unchanged, thanks to the defaults; MCTS and the arena never pass an ownership pointer.

- [ ] **Step 1: Write the failing tests**

The whole-network gradient check becomes a helper that runs twice, without and with the head. The run without it draws exactly the random numbers it drew before, so it is the old test. Three new tests pin the rest. The head must leave every other parameter, output and gradient of the network untouched. `w = 0` must reproduce the step of the network without the head bit for bit. And plain SGD must fit a target that a 1×1 head can learn: +1 on the side to move's stones and −1 on the opponent's, read from input planes 0 and 1.

In `tests/test_network.cpp`, replace:

```cpp
    return batch;
}

double totalLoss(const LossStats& s) { return static_cast<double>(s.policyLoss) + s.valueLoss; }

enum class Probe { Match, Kink, Mismatch };

```

with:

```cpp
    return batch;
}

// Ownership targets in {-1, 0, +1}, [B][area].
std::vector<float> randomOwnership(const NetConfig& cfg, int B, Rng& rng) {
    std::vector<float> own(static_cast<size_t>(B) * cfg.area());
    for (float& v : own) v = static_cast<float>(rng.below(3) - 1);
    return own;
}

// The loss trainStep minimizes: policy + value + w * ownership.
double totalLoss(const LossStats& s, float ownershipWeight = 0.0f) {
    return static_cast<double>(s.policyLoss) + s.valueLoss +
           static_cast<double>(ownershipWeight) * s.ownershipLoss;
}

enum class Probe { Match, Kink, Mismatch };

```

In `tests/test_network.cpp`, replace:

```cpp

} // namespace

static void test_whole_network_gradient_check() {
    const NetConfig cfg = makeConfig(5, 1, 2, 4, 8);
    const int B = 3;
    NaiveBackend be;
    Network net(cfg, 42);
```

with:

```cpp

} // namespace

// With `ownership`, the network has the ownership head and the loss includes
// it with weight 1.5, so its gradients and the trunk's share of them are
// checked too.
static void checkWholeNetworkGradients(bool ownership) {
    NetConfig cfg = makeConfig(5, 1, 2, 4, 8);
    cfg.ownership = ownership;
    const float w = ownership ? 1.5f : 0.0f;
    const int B = 3;
    NaiveBackend be;
    Network net(cfg, 42);
```

In `tests/test_network.cpp`, replace:

```cpp
        }
    }
    Batch batch = randomBatch(cfg, B, rng);

    const LossStats trained = net.trainStep(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B);
    auto loss = [&] {
        return totalLoss(net.computeLoss(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B));
    };
    assert(near(totalLoss(trained), loss(), 1e-6));   // same forward, same batch statistics

    int checked = 0, kinks = 0;
    for (ParamView& p : net.params()) {
```

with:

```cpp
        }
    }
    Batch batch = randomBatch(cfg, B, rng);
    std::vector<float> own;
    if (ownership) own = randomOwnership(cfg, B, rng);
    const float* target = ownership ? own.data() : nullptr;

    const LossStats trained =
        net.trainStep(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B, target, w);
    assert(ownership == (trained.ownershipLoss > 0.0f));
    auto loss = [&] {
        return totalLoss(net.computeLoss(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B,
                                         target, w),
                         w);
    };
    assert(near(totalLoss(trained, w), loss(), 1e-6));   // same forward, same batch statistics

    int checked = 0, kinks = 0;
    for (ParamView& p : net.params()) {
```

In `tests/test_network.cpp`, replace:

```cpp
    assert(kinks * 4 <= checked);   // kinks excuse only a minority of probes
}

static void test_param_names_order_and_decay() {
    const NetConfig cfg = makeConfig(5, 1, 1, 4, 8);
    Network a(cfg, 1), b(cfg, 2);
```

with:

```cpp
    assert(kinks * 4 <= checked);   // kinks excuse only a minority of probes
}

static void test_whole_network_gradient_check() {
    checkWholeNetworkGradients(false);
    checkWholeNetworkGradients(true);
}

static void test_param_names_order_and_decay() {
    const NetConfig cfg = makeConfig(5, 1, 1, 4, 8);
    Network a(cfg, 1), b(cfg, 2);
```

In `tests/test_network.cpp`, replace:

```cpp
    }
}

int main() {
    test_whole_network_gradient_check();
    test_param_names_order_and_decay();
```

with:

```cpp
    }
}

// The head's tensors come last, and it is initialized last, so a network with
// the head is the network without it plus three tensors: same names, same
// weights, same policy and value. Its output is n*n values in [-1, 1].
static void test_ownership_head_params_and_outputs() {
    NetConfig off = makeConfig(5, 2, 1, 4, 8), on = off;
    on.ownership = true;
    Network a(off, 21), b(on, 21);
    std::vector<ParamView> pa = a.params(), pb = b.params();
    assert(pb.size() == pa.size() + 3);
    for (size_t i = 0; i < pa.size(); ++i) {
        assert(pa[i].name == pb[i].name && pa[i].size == pb[i].size && pa[i].decay == pb[i].decay);
        assert(std::memcmp(pa[i].value, pb[i].value, pa[i].size * sizeof(float)) == 0);
    }
    const size_t k = pa.size();
    assert(pb[k].name == "ownership.conv.w" && pb[k].size == 4 && pb[k].decay);   // filters weights
    assert(pb[k + 1].name == "ownership.bn.gamma" && pb[k + 1].size == 1 && !pb[k + 1].decay);
    assert(pb[k + 2].name == "ownership.bn.beta" && pb[k + 2].size == 1 && !pb[k + 2].decay);
    std::vector<BufferView> ba = a.buffers(), bb = b.buffers();
    assert(bb.size() == ba.size() + 2);
    assert(bb[ba.size()].name == "ownership.bn.runMean" && bb.back().name == "ownership.bn.runVar");

    NaiveBackend be;
    Workspace wa(off, 3, be), wb(on, 3, be);
    Rng rng(9);
    Batch batch = randomBatch(off, 3, rng);
    const size_t A = static_cast<size_t>(off.area());
    std::vector<float> la(3 * off.actions()), lb(la.size()), lc(la.size()), va(3), vb(3), vc(3);
    std::vector<float> own(3 * A, std::numeric_limits<float>::quiet_NaN());
    a.infer(wa, batch.input.data(), 3, la.data(), va.data());
    b.infer(wb, batch.input.data(), 3, lb.data(), vb.data(), own.data());
    assert(std::memcmp(la.data(), lb.data(), la.size() * sizeof(float)) == 0);
    assert(std::memcmp(va.data(), vb.data(), va.size() * sizeof(float)) == 0);
    assert(allFinite(own));
    for (float o : own) assert(o >= -1.0f && o <= 1.0f);
    assert(*std::min_element(own.begin(), own.end()) < *std::max_element(own.begin(), own.end()));
    // Without an ownership output the head is simply not read.
    b.infer(wb, batch.input.data(), 3, lc.data(), vc.data());
    assert(lc == lb && vc == vb);

    // Every board size: [B][n*n] values, all in range.
    for (int n : {7, 19}) {
        NetConfig cfg = makeConfig(n, 2, 1, 2, 4);
        cfg.ownership = true;
        Network net(cfg, static_cast<uint64_t>(n));
        Workspace ws(cfg, 2, be);
        Batch nb = randomBatch(cfg, 2, rng);
        std::vector<float> logits(2 * cfg.actions()), values(2);
        std::vector<float> o(2 * static_cast<size_t>(cfg.area()), std::numeric_limits<float>::quiet_NaN());
        net.infer(ws, nb.input.data(), 2, logits.data(), values.data(), o.data());
        assert(allFinite(o));
        for (float x : o) assert(x >= -1.0f && x <= 1.0f);
    }
}

// w = 0: the ownership loss is measured, but the step is exactly the one the
// network without the head takes, and the head itself gets no gradient.
static void test_ownership_weight_zero_trains_nothing() {
    NetConfig off = makeConfig(5, 1, 1, 4, 8), on = off;
    on.ownership = true;
    Network a(off, 23), b(on, 23);
    NaiveBackend be;
    Workspace wa(off, 4, be), wb(on, 4, be);
    Rng rng(10);
    Batch batch = randomBatch(off, 4, rng);
    const std::vector<float> own = randomOwnership(on, 4, rng);
    const LossStats sa = a.trainStep(wa, batch.input.data(), batch.pi.data(), batch.z.data(), 4);
    const LossStats sb =
        b.trainStep(wb, batch.input.data(), batch.pi.data(), batch.z.data(), 4, own.data(), 0.0f);
    assert(sa.policyLoss == sb.policyLoss && sa.valueLoss == sb.valueLoss);
    assert(sa.ownershipLoss == 0.0f && sb.ownershipLoss > 0.0f);
    std::vector<ParamView> pa = a.params(), pb = b.params();
    for (size_t i = 0; i < pa.size(); ++i)
        assert(std::memcmp(pa[i].grad, pb[i].grad, pa[i].size * sizeof(float)) == 0);
    for (size_t i = pa.size(); i < pb.size(); ++i)
        for (size_t j = 0; j < pb[i].size; ++j) assert(pb[i].grad[j] == 0.0f);

    // w > 0: the head and, through it, the trunk receive gradient.
    b.trainStep(wb, batch.input.data(), batch.pi.data(), batch.z.data(), 4, own.data(), 1.5f);
    pb = b.params();
    bool headGrad = false;
    for (size_t j = 0; j < pb[pa.size()].size; ++j) headGrad = headGrad || pb[pa.size()].grad[j] != 0.0f;
    assert(headGrad);
    assert(std::memcmp(pa[0].grad, pb[0].grad, pa[0].size * sizeof(float)) != 0);   // stem.conv.w
}

// A few SGD steps on one batch reduce the ownership loss. The target is one a
// 1x1 head can learn from the trunk: +1 on the side to move's stones (input
// plane 0), -1 on the opponent's (plane 1), 0 where both or neither.
static void test_sgd_reduces_ownership_loss() {
    NetConfig cfg = makeConfig(5, 1, 1, 8, 16);
    cfg.ownership = true;
    const int B = 8, A = cfg.area();
    NaiveBackend be;
    Network net(cfg, 11);
    Workspace ws(cfg, B, be);
    Rng rng(7);
    Batch batch = randomBatch(cfg, B, rng, /*oneHotPi=*/true);
    std::vector<float> own(static_cast<size_t>(B) * A);
    for (int b = 0; b < B; ++b)
        for (int p = 0; p < A; ++p) {
            const float* planes = batch.input.data() + static_cast<size_t>(b) * cfg.planes() * A;
            own[static_cast<size_t>(b) * A + p] = planes[p] - planes[A + p];
        }
    double first = -1.0;
    for (int step = 0; step < 40; ++step) {
        const LossStats s =
            net.trainStep(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B, own.data(), 1.5f);
        if (step == 0) first = s.ownershipLoss;
        for (ParamView& p : net.params())
            for (size_t i = 0; i < p.size; ++i) p.value[i] -= 0.05f * p.grad[i];
    }
    const double last =
        net.computeLoss(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B, own.data(), 1.5f)
            .ownershipLoss;
    std::printf("sgd: ownership loss %.4f -> %.4f\n", first, last);
    assert(last <= 0.5 * first);
}

int main() {
    test_whole_network_gradient_check();
    test_param_names_order_and_decay();
```

In `tests/test_network.cpp`, replace:

```cpp
    test_running_stats_only_change_in_train_step();
    test_sgd_reduces_loss_on_fixed_batch();
    test_inference_is_per_sample();
    std::printf("test_network: all tests passed\n");
    return 0;
}
```

with:

```cpp
    test_running_stats_only_change_in_train_step();
    test_sgd_reduces_loss_on_fixed_batch();
    test_inference_is_per_sample();
    test_ownership_head_params_and_outputs();
    test_ownership_weight_zero_trains_nothing();
    test_sgd_reduces_ownership_loss();
    std::printf("test_network: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build build -j`
Expected: the build fails in `tests/test_network.cpp` with:

```
tests/test_network.cpp:66:53: error: no member named 'ownershipLoss' in 'azgo::LossStats'
tests/test_network.cpp:131:83: error: too many arguments to function call, expected 5, have 7
```

- [ ] **Step 3: Declare the head**

In `include/azgo/network.hpp`, replace:

```cpp
// and a value head into one number in [-1, 1]: the expected result for the
// side to move. Parameters (Network) are shared read-only by all threads;
// activations (Workspace) are per thread, so many searches can run at once.
#pragma once
#include <cstdint>
#include <string>
```

with:

```cpp
// and a value head into one number in [-1, 1]: the expected result for the
// side to move. Parameters (Network) are shared read-only by all threads;
// activations (Workspace) are per thread, so many searches can run at once.
//
// Optionally (NetConfig::ownership) a third head predicts, for every point,
// who will own it when the game ends: a 1x1 conv to one channel, batch norm
// and a tanh per point, in [-1, 1] from the side to move's view. It is
// KataGo's auxiliary target: n*n training signals per position instead of
// one game result, and every point of a dead group is labelled with the
// opponent's color. Search never reads it; it only shapes the shared trunk.
#pragma once
#include <cstdint>
#include <string>
```

In `include/azgo/network.hpp`, replace:

```cpp
struct ParamView  { std::string name; float* value; float* grad; size_t size; bool decay; };
struct BufferView { std::string name; float* value; size_t size; };

struct LossStats { float policyLoss = 0; float valueLoss = 0; };  // batch means

// Per-thread activation and scratch storage for up to maxBatch samples.
class Workspace {
```

with:

```cpp
struct ParamView  { std::string name; float* value; float* grad; size_t size; bool decay; };
struct BufferView { std::string name; float* value; size_t size; };

// Batch means. ownershipLoss is the unweighted mean squared error per point,
// 0 when no ownership target was given.
struct LossStats { float policyLoss = 0; float valueLoss = 0; float ownershipLoss = 0; };

// Per-thread activation and scratch storage for up to maxBatch samples.
class Workspace {
```

In `include/azgo/network.hpp`, replace:

```cpp
    std::vector<float> valueXhat_, valueInvStd_, valueAct_;      // valueAct_ [1][B][area] = [B][area]
    std::vector<float> valueHidden_;              // [B][valueHidden]
    std::vector<float> values_;                   // [B] (after tanh)
    // Backward scratch.
    std::vector<float> dTrunk_, dTmp_, dTmp2_;    // [filters][B][area]
    std::vector<float> dLogits_, dPolicyFlat_, dPolicyAct_;
    std::vector<float> dValues_, dValueHidden_, dValueAct_;
};

class Network {
```

with:

```cpp
    std::vector<float> valueXhat_, valueInvStd_, valueAct_;      // valueAct_ [1][B][area] = [B][area]
    std::vector<float> valueHidden_;              // [B][valueHidden]
    std::vector<float> values_;                   // [B] (after tanh)
    // Ownership head, allocated only when cfg.ownership. ownership_ is
    // [1][B][area] = [B][area], after tanh.
    std::vector<float> ownershipXhat_, ownershipInvStd_, ownership_;
    // Backward scratch.
    std::vector<float> dTrunk_, dTmp_, dTmp2_;    // [filters][B][area]
    std::vector<float> dLogits_, dPolicyFlat_, dPolicyAct_;
    std::vector<float> dValues_, dValueHidden_, dValueAct_;
    std::vector<float> dOwnership_;               // [B][area]
};

class Network {
```

In `include/azgo/network.hpp`, replace:

```cpp
    Network(const NetConfig& cfg, uint64_t seed);   // He-initialized from Rng(seed)
    const NetConfig& config() const { return cfg_; }

    // Fixed order, stable names ("stem.conv.w", "block2.bn1.gamma", "policy.fc.b", ...).
    // decay is true for conv and linear weights only (not biases, not BN).
    std::vector<ParamView> params();
    std::vector<BufferView> buffers();              // BN runMean/runVar, fixed order
    void zeroGrad();

    // Inference (BN running statistics). input [B][planes][area],
    // logits [B][actions], values [B] (tanh). Asserts B <= ws.maxBatch().
    void infer(Workspace& ws, const float* input, int B, float* logits, float* values) const;

    // Training step: zeroGrad, forward with batch statistics (updates BN
    // running stats), loss = mean((z - v)^2) + mean(-sum pi log softmax(logits)),
    // backward into grads. pi [B][actions], z [B].
    LossStats trainStep(Workspace& ws, const float* input, const float* pi, const float* z, int B);

    // Same loss as trainStep (batch statistics) without backward and without
    // touching running stats. Used by gradient checks.
    LossStats computeLoss(Workspace& ws, const float* input, const float* pi, const float* z, int B);

private:
    NetConfig cfg_;
```

with:

```cpp
    Network(const NetConfig& cfg, uint64_t seed);   // He-initialized from Rng(seed)
    const NetConfig& config() const { return cfg_; }

    // Fixed order, stable names ("stem.conv.w", "block2.bn1.gamma", "policy.fc.b", ...),
    // the ownership head last ("ownership.conv.w", "ownership.bn.gamma", ".beta").
    // decay is true for conv and linear weights only (not biases, not BN).
    std::vector<ParamView> params();
    std::vector<BufferView> buffers();              // BN runMean/runVar, fixed order
    void zeroGrad();

    // Inference (BN running statistics). input [B][planes][area],
    // logits [B][actions], values [B] (tanh). ownership, if not null,
    // receives [B][area] in [-1, 1] from the side to move's view; it needs a
    // network with the head (asserted). Asserts B <= ws.maxBatch().
    void infer(Workspace& ws, const float* input, int B, float* logits, float* values,
               float* ownership = nullptr) const;

    // Training step: zeroGrad, forward with batch statistics (updates BN
    // running stats), loss = mean((z - v)^2) + mean(-sum pi log softmax(logits))
    // + w * mean((1/area) sum_p (o_p - t_p)^2), backward into grads.
    // pi [B][actions], z [B]. ownership: targets t [B][area] in [-1, 1] from
    // the side to move's view, or null for no ownership loss; it needs a
    // network with the head (asserted). w = ownershipWeight; with w = 0 the
    // ownership loss is reported but the head receives no gradient.
    LossStats trainStep(Workspace& ws, const float* input, const float* pi, const float* z, int B,
                        const float* ownership = nullptr, float ownershipWeight = 0.0f);

    // Same loss as trainStep (batch statistics) without backward and without
    // touching running stats. Used by gradient checks.
    LossStats computeLoss(Workspace& ws, const float* input, const float* pi, const float* z, int B,
                          const float* ownership = nullptr, float ownershipWeight = 0.0f);

private:
    NetConfig cfg_;
```

In `include/azgo/network.hpp`, replace:

```cpp
    Conv2d valueConv_;
    BatchNorm2d valueBn_;
    Linear valueFc1_, valueFc2_;

    // Infer: BN running statistics. Train: batch statistics, running stats
    // updated. Probe: batch statistics, running stats untouched (computeLoss).
    enum class Mode { Infer, Train, Probe };
    void forward(Workspace& ws, const float* input, int B, Mode mode);
    LossStats loss(Workspace& ws, const float* pi, const float* z, int B, bool withGradients) const;
    void backward(Workspace& ws, int B);
};

} // namespace azgo
```

with:

```cpp
    Conv2d valueConv_;
    BatchNorm2d valueBn_;
    Linear valueFc1_, valueFc2_;
    Conv2d ownershipConv_;            // in params() only when cfg_.ownership
    BatchNorm2d ownershipBn_;

    // Infer: BN running statistics. Train: batch statistics, running stats
    // updated. Probe: batch statistics, running stats untouched (computeLoss).
    enum class Mode { Infer, Train, Probe };
    void forward(Workspace& ws, const float* input, int B, Mode mode);
    LossStats loss(Workspace& ws, const float* pi, const float* z, const float* ownership,
                   float ownershipWeight, int B, bool withGradients) const;
    // withOwnership: backpropagate the ownership head too (a target and w != 0).
    void backward(Workspace& ws, int B, bool withOwnership);
};

} // namespace azgo
```

- [ ] **Step 4: Implement it**

The order of these replacements follows the file: the workspace buffers, the layers and their initialization (the head's conv after `valueFc2_`), `params()`, `buffers()`, `infer`, `trainStep` and `computeLoss`, the forward pass, the loss and the backward pass. The backward pass runs the head only when it has something to learn (a target and `w ≠ 0`). It adds the head's contribution to `dTrunk` right after the policy head's.

In `src/network.cpp`, replace:

```cpp
    dValues_.resize(maxBatch);
    dValueHidden_.resize(hidden);
    dValueAct_.resize(M);
}

// ---------------------------------------------------------------- Network
```

with:

```cpp
    dValues_.resize(maxBatch);
    dValueHidden_.resize(hidden);
    dValueAct_.resize(M);
    if (cfg.ownership) {
        ownershipXhat_.resize(M);
        ownershipInvStd_.resize(1);
        ownership_.resize(M);
        dOwnership_.resize(M);
    }
}

// ---------------------------------------------------------------- Network
```

In `src/network.cpp`, replace:

```cpp
      valueConv_(cfg.filters, 1, 1),
      valueBn_(1),
      valueFc1_(cfg.area(), cfg.valueHidden),
      valueFc2_(cfg.valueHidden, 1) {
    assert(cfg.boardSize >= kMinBoardSize && cfg.boardSize <= kMaxBoardSize);
    assert(cfg.history >= 1 && cfg.blocks >= 0 && cfg.filters >= 1 && cfg.valueHidden >= 1);
    const int F = cfg.filters;
```

with:

```cpp
      valueConv_(cfg.filters, 1, 1),
      valueBn_(1),
      valueFc1_(cfg.area(), cfg.valueHidden),
      valueFc2_(cfg.valueHidden, 1),
      ownershipConv_(cfg.filters, 1, 1),
      ownershipBn_(1) {
    assert(cfg.boardSize >= kMinBoardSize && cfg.boardSize <= kMaxBoardSize);
    assert(cfg.history >= 1 && cfg.blocks >= 0 && cfg.filters >= 1 && cfg.valueHidden >= 1);
    const int F = cfg.filters;
```

In `src/network.cpp`, replace:

```cpp
    valueConv_.init(rng);
    valueFc1_.init(rng);
    valueFc2_.init(rng);
}

std::vector<ParamView> Network::params() {
```

with:

```cpp
    valueConv_.init(rng);
    valueFc1_.init(rng);
    valueFc2_.init(rng);
    // Last, so a network with the head has the same other weights as one
    // without it for the same seed.
    if (cfg.ownership) ownershipConv_.init(rng);
}

std::vector<ParamView> Network::params() {
```

In `src/network.cpp`, replace:

```cpp
    bn("value.bn", valueBn_);
    linear("value.fc1", valueFc1_);
    linear("value.fc2", valueFc2_);
    return out;
}

```

with:

```cpp
    bn("value.bn", valueBn_);
    linear("value.fc1", valueFc1_);
    linear("value.fc2", valueFc2_);
    if (cfg_.ownership) {
        conv("ownership.conv", ownershipConv_);
        bn("ownership.bn", ownershipBn_);
    }
    return out;
}

```

In `src/network.cpp`, replace:

```cpp
    }
    bn("policy.bn", policyBn_);
    bn("value.bn", valueBn_);
    return out;
}

```

with:

```cpp
    }
    bn("policy.bn", policyBn_);
    bn("value.bn", valueBn_);
    if (cfg_.ownership) bn("ownership.bn", ownershipBn_);
    return out;
}

```

In `src/network.cpp`, replace:

```cpp
    for (ParamView& p : params()) std::fill(p.grad, p.grad + p.size, 0.0f);
}

void Network::infer(Workspace& ws, const float* input, int B, float* logits, float* values) const {
    // Mode::Infer only reads parameters (BN uses its running statistics via
    // the const forwardInfer), so this cast never writes to the shared
    // network; it lets a single forward() serve inference and training.
    const_cast<Network*>(this)->forward(ws, input, B, Mode::Infer);
    std::copy(ws.logits_.begin(), ws.logits_.begin() + static_cast<size_t>(B) * cfg_.actions(), logits);
    std::copy(ws.values_.begin(), ws.values_.begin() + B, values);
}

LossStats Network::trainStep(Workspace& ws, const float* input, const float* pi, const float* z,
                             int B) {
    zeroGrad();
    forward(ws, input, B, Mode::Train);
    LossStats stats = loss(ws, pi, z, B, true);
    backward(ws, B);
    return stats;
}

LossStats Network::computeLoss(Workspace& ws, const float* input, const float* pi, const float* z,
                               int B) {
    forward(ws, input, B, Mode::Probe);
    return loss(ws, pi, z, B, false);
}

void Network::forward(Workspace& ws, const float* input, int B, Mode mode) {
```

with:

```cpp
    for (ParamView& p : params()) std::fill(p.grad, p.grad + p.size, 0.0f);
}

void Network::infer(Workspace& ws, const float* input, int B, float* logits, float* values,
                    float* ownership) const {
    assert(ownership == nullptr || cfg_.ownership);
    // Mode::Infer only reads parameters (BN uses its running statistics via
    // the const forwardInfer), so this cast never writes to the shared
    // network; it lets a single forward() serve inference and training.
    const_cast<Network*>(this)->forward(ws, input, B, Mode::Infer);
    std::copy(ws.logits_.begin(), ws.logits_.begin() + static_cast<size_t>(B) * cfg_.actions(), logits);
    std::copy(ws.values_.begin(), ws.values_.begin() + B, values);
    if (ownership && cfg_.ownership)
        std::copy(ws.ownership_.begin(), ws.ownership_.begin() + static_cast<size_t>(B) * cfg_.area(),
                  ownership);
}

LossStats Network::trainStep(Workspace& ws, const float* input, const float* pi, const float* z,
                             int B, const float* ownership, float ownershipWeight) {
    assert(ownership == nullptr || cfg_.ownership);
    zeroGrad();
    forward(ws, input, B, Mode::Train);
    LossStats stats = loss(ws, pi, z, ownership, ownershipWeight, B, true);
    backward(ws, B, cfg_.ownership && ownership != nullptr && ownershipWeight != 0.0f);
    return stats;
}

LossStats Network::computeLoss(Workspace& ws, const float* input, const float* pi, const float* z,
                               int B, const float* ownership, float ownershipWeight) {
    assert(ownership == nullptr || cfg_.ownership);
    forward(ws, input, B, Mode::Probe);
    return loss(ws, pi, z, ownership, ownershipWeight, B, false);
}

void Network::forward(Workspace& ws, const float* input, int B, Mode mode) {
```

In `src/network.cpp`, replace:

```cpp
    reluForward(ws.valueHidden_.data(), static_cast<size_t>(B) * cfg_.valueHidden);
    valueFc2_.forward(be, ws.valueHidden_.data(), B, ws.values_.data());
    for (int b = 0; b < B; ++b) ws.values_[b] = std::tanh(ws.values_[b]);
}

LossStats Network::loss(Workspace& ws, const float* pi, const float* z, int B,
                        bool withGradients) const {
    const int actions = cfg_.actions();
    double policySum = 0.0, valueSum = 0.0;
    for (int b = 0; b < B; ++b) {
```

with:

```cpp
    reluForward(ws.valueHidden_.data(), static_cast<size_t>(B) * cfg_.valueHidden);
    valueFc2_.forward(be, ws.valueHidden_.data(), B, ws.values_.data());
    for (int b = 0; b < B; ++b) ws.values_[b] = std::tanh(ws.values_[b]);

    // Ownership head: conv1x1 (F -> 1) -> BN -> tanh at every point, no
    // ReLU and no linear layer: each point's owner is read off the trunk's
    // features at that point. [1][B][A] already is [B][A].
    if (cfg_.ownership) {
        ownershipConv_.forward(be, x, B, n, col, convOut);
        norm(ownershipBn_, convOut, ws.ownership_.data(), ws.ownershipXhat_.data(),
             ws.ownershipInvStd_.data());
        for (int j = 0; j < M; ++j) ws.ownership_[j] = std::tanh(ws.ownership_[j]);
    }
}

LossStats Network::loss(Workspace& ws, const float* pi, const float* z, const float* ownership,
                        float ownershipWeight, int B, bool withGradients) const {
    const int actions = cfg_.actions();
    double policySum = 0.0, valueSum = 0.0;
    for (int b = 0; b < B; ++b) {
```

In `src/network.cpp`, replace:

```cpp
    LossStats stats;
    stats.policyLoss = static_cast<float>(policySum / B);
    stats.valueLoss = static_cast<float>(valueSum / B);
    return stats;
}

void Network::backward(Workspace& ws, int B) {
    const Backend& be = *ws.backend_;
    const int n = cfg_.boardSize, A = cfg_.area(), F = cfg_.filters;
    const int M = B * A;
```

with:

```cpp
    LossStats stats;
    stats.policyLoss = static_cast<float>(policySum / B);
    stats.valueLoss = static_cast<float>(valueSum / B);

    // Ownership: w * (1/A) * sum_p (o_p - t_p)^2, averaged over the batch, so
    // the mean over all B * A points. With o = tanh(u):
    // dL/du = 2 w (o - t)(1 - o^2) / (B * A).
    if (cfg_.ownership && ownership) {
        const size_t points = static_cast<size_t>(B) * cfg_.area();
        double ownershipSum = 0.0;
        for (size_t j = 0; j < points; ++j) {
            const double o = ws.ownership_[j];
            const double d = o - ownership[j];
            ownershipSum += d * d;
            if (withGradients)
                ws.dOwnership_[j] = static_cast<float>(2.0 * ownershipWeight * d * (1.0 - o * o) / points);
        }
        stats.ownershipLoss = static_cast<float>(ownershipSum / points);
    }
    return stats;
}

void Network::backward(Workspace& ws, int B, bool withOwnership) {
    const Backend& be = *ws.backend_;
    const int n = cfg_.boardSize, A = cfg_.area(), F = cfg_.filters;
    const int M = B * A;
```

In `src/network.cpp`, replace:

```cpp
    policyConv_.backward(be, trunkOut, col, ws.dPolicyAct_.data(), B, n, dcol, dTmp);
    for (size_t j = 0; j < trunk; ++j) dTrunk[j] += dTmp[j];   // the trunk feeds both heads

    // Residual blocks in reverse; dTrunk holds dL/d(block output). The
    // shared col buffer was overwritten by later convs, so each 3x3 conv's
    // columns are rebuilt from its saved input before its backward.
```

with:

```cpp
    policyConv_.backward(be, trunkOut, col, ws.dPolicyAct_.data(), B, n, dcol, dTmp);
    for (size_t j = 0; j < trunk; ++j) dTrunk[j] += dTmp[j];   // the trunk feeds both heads

    // Ownership head: through tanh (already in dOwnership_), BN and the 1x1 conv.
    if (withOwnership) {
        ownershipBn_.backward(ws.dOwnership_.data(), ws.ownershipXhat_.data(),
                              ws.ownershipInvStd_.data(), M, ws.dOwnership_.data());
        ownershipConv_.backward(be, trunkOut, col, ws.dOwnership_.data(), B, n, dcol, dTmp);
        for (size_t j = 0; j < trunk; ++j) dTrunk[j] += dTmp[j];
    }

    // Residual blocks in reverse; dTrunk holds dL/d(block output). The
    // shared col buffer was overwritten by later convs, so each 3x3 conv's
    // columns are rebuilt from its saved input before its backward.
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cmake --build build -j && ./build/test_network`
Expected: the output ends with

```
sgd: loss 5.2080 -> 0.0155
sgd: ownership loss 0.8638 -> 0.1422
test_network: all tests passed
```

(The two `sgd:` numbers may differ in the last digits on another compiler or BLAS build; the assertions are `last <= 0.5 * first`.) Then the whole suite and the BLAS variant. Expected: `100% tests passed out of 26` in both builds. Every other test is unchanged and passes, which is the spec's check that networks without the head are unaffected.

- [ ] **Step 6: Commit**

```bash
git add include/azgo/network.hpp src/network.cpp tests/test_network.cpp
git commit -F - <<'EOF'
Add an optional ownership head to the network

A 1x1 conv, batch norm and a tanh at every point predict who owns each
point at the end, from the side to move's view: n*n training signals
per position next to the value head's single game result. Its loss is
w * mean (o - t)^2, with gradient 2w(o - t)(1 - o^2) / (B n^2), and the
whole-network finite-difference check now runs with the head too. It
is initialized after every other layer and skipped entirely without
the flag, so networks without it are bit-for-bit what they were, and
with w = 0 a step is exactly the step without the head.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 4: Ownership targets in self-play records, the replay buffer and `state.bin` version 3

**Files:**
- Modify: `include/azgo/selfplay.hpp`, `src/selfplay.cpp`, `include/azgo/replay_buffer.hpp`, `src/replay_buffer.cpp`, `include/azgo/run_state.hpp`, `src/run_state.cpp`
- Test: `tests/test_selfplay.cpp`, `tests/test_replay_buffer.cpp`, `tests/test_run_state.cpp`, `tests/test_trainer.cpp`

**Interfaces:**
- Consumes: `Board::ownership` (Task 1).
- Produces:
  - `std::vector<int8_t> GameRecord::finalOwnership`: `n*n` values from Black's view, filled by `playSelfPlayGame` from the final board.
  - `std::vector<int8_t> replayFinalOwnership(const GameRecord&)`: replays the moves with `Game`. It throws `std::runtime_error("position P differs from the replayed one")` or `("move K (C3) is illegal")`.
  - `std::vector<float> TrainingBatch::ownership` (`[size][area]`), always filled by `sample`.
  - `void ReplayBuffer::encodeSample(size_t, int ply, int history, int sym, float* input, float* pi, float* z, float* ownership = nullptr) const`: the target is `finalOwnership` negated when White is to move, with point `p` written at `transformPoint(sym, p, n)`.
  - `void ReplayBuffer::save(BinaryWriter&, bool withOwnership = true) const` and `static ReplayBuffer load(BinaryReader&, bool withOwnership = true)`. `false` is the layout of `state.bin` versions 1 and 2; loading it recomputes every game's `finalOwnership`, and a failure is `replay buffer: game K does not replay: <reason>`.
  - `state.bin` version 3. `loadTrainingState` reads versions 1–3 and prefixes every error with the path.
  - `ReplayBuffer::add` now expects `finalOwnership` (asserted), so hand-made records in tests must fill it, as `makeRecord` in `tests/test_replay_buffer.cpp` now does.

- [ ] **Step 1: Write the failing tests**

`tests/test_selfplay.cpp`: the record's `finalOwnership` is the final board's, replaying finds the same, and a hand-built record whose moves do not replay produces the exact messages.

In `tests/test_selfplay.cpp`, replace:

```cpp
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#include "azgo/game.hpp"
#include "azgo/parallel.hpp"
```

with:

```cpp
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/game.hpp"
#include "azgo/parallel.hpp"
```

In `tests/test_selfplay.cpp`, replace:

```cpp
    assert(rec.toMoveAt(0) == Color::Black && rec.toMoveAt(1) == Color::White);
    assert(rec.zAt(0) == -1.0f);   // Black's position: Black lost
    assert(rec.zAt(1) == +1.0f);   // White's position: White won
}

static void test_z_for_a_black_win_and_a_tie() {
```

with:

```cpp
    assert(rec.toMoveAt(0) == Color::Black && rec.toMoveAt(1) == Color::White);
    assert(rec.zAt(0) == -1.0f);   // Black's position: Black lost
    assert(rec.zAt(1) == +1.0f);   // White's position: White won
    // The final board is empty: nobody owns anything.
    assert(rec.finalOwnership == std::vector<int8_t>(25, 0));
}

static void test_z_for_a_black_win_and_a_tie() {
```

In `tests/test_selfplay.cpp`, replace:

```cpp
    assert(rec.resultForBlack == game.resultForBlack());
    for (int ply = 0; ply < rec.positions(); ++ply)
        assert(rec.zAt(ply) == (ply % 2 == 0 ? rec.resultForBlack : -rec.resultForBlack));
}

static bool sameRecord(const GameRecord& a, const GameRecord& b) {
    return a.boardSize == b.boardSize && a.komi == b.komi && a.moves == b.moves &&
           a.stones == b.stones && a.visits == b.visits && a.resultForBlack == b.resultForBlack;
}

static std::vector<GameRecord> playGames(const Network& net, const Backend& be, int workers) {
```

with:

```cpp
    assert(rec.resultForBlack == game.resultForBlack());
    for (int ply = 0; ply < rec.positions(); ++ply)
        assert(rec.zAt(ply) == (ply % 2 == 0 ? rec.resultForBlack : -rec.resultForBlack));

    // finalOwnership is the final board's, and replaying the moves finds it too.
    std::vector<int8_t> own(area);
    game.board().ownership(own.data());
    assert(rec.finalOwnership == own);
    assert(replayFinalOwnership(rec) == own);
    int sum = 0;
    for (int8_t o : own) sum += o;
    assert(sum == game.board().areaScore());
}

// A record that does not replay is an error that says why. The record is
// built by hand: Black C3, White D4, Black C4, White D3 on 5x5.
static void test_replay_rejects_a_bad_record() {
    const int n = 5, area = 25;
    Game game(n, 7.5f, defaultMaxMoves(n));
    GameRecord good;
    good.boardSize = n;
    good.komi = 7.5f;
    for (int move : {12, 8, 7, 13}) {
        const size_t offset = good.stones.size();
        good.stones.resize(offset + area);
        game.board().exportStones(good.stones.data() + offset);
        good.visits.push_back({{static_cast<int16_t>(move), 1}});
        good.moves.push_back(move);
        game.play(move);
    }
    auto replayError = [](const GameRecord& r) {
        try {
            replayFinalOwnership(r);
        } catch (const std::runtime_error& e) {
            return std::string(e.what());
        }
        return std::string();
    };
    assert(replayError(good).empty());
    std::vector<int8_t> own(area);
    game.board().ownership(own.data());
    assert(replayFinalOwnership(good) == own);

    // Move 2 on the point Black took with move 1: occupied, so illegal.
    GameRecord occupied = good;
    occupied.moves[1] = 12;
    assert(replayError(occupied) == "move 2 (C3) is illegal");

    // A stored position that the moves do not produce: a stone on A5 at ply 2.
    GameRecord edited = good;
    edited.stones[2 * area + 0] = static_cast<uint8_t>(Color::Black);
    assert(replayError(edited) == "position 2 differs from the replayed one");
}

static bool sameRecord(const GameRecord& a, const GameRecord& b) {
    return a.boardSize == b.boardSize && a.komi == b.komi && a.moves == b.moves &&
           a.stones == b.stones && a.visits == b.visits && a.resultForBlack == b.resultForBlack &&
           a.finalOwnership == b.finalOwnership;
}

static std::vector<GameRecord> playGames(const Network& net, const Backend& be, int workers) {
```

In `tests/test_selfplay.cpp`, replace:

```cpp
    test_z_for_a_black_win_and_a_tie();
    test_record_is_consistent_with_a_replay();
    test_self_play_is_independent_of_worker_count();
    std::printf("test_selfplay: all tests passed\n");
    return 0;
}
```

with:

```cpp
    test_z_for_a_black_win_and_a_tie();
    test_record_is_consistent_with_a_replay();
    test_self_play_is_independent_of_worker_count();
    test_replay_rejects_a_bad_record();
    std::printf("test_selfplay: all tests passed\n");
    return 0;
}
```

`tests/test_replay_buffer.cpp`: `makeRecord` fills `finalOwnership` as self-play does. Two hand-checked games pin perspective and symmetry. The round trip covers the target, and the old layout recomputes it. A broken old record is an error naming the game.

In `tests/test_replay_buffer.cpp`, replace:

```cpp
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "azgo/encode.hpp"
#include "azgo/game.hpp"
```

with:

```cpp
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/encode.hpp"
#include "azgo/game.hpp"
```

In `tests/test_replay_buffer.cpp`, replace:

```cpp
// Plays `moves` on a 5x5 Game and records it as self-play would: the board
// before each move, and visit counts {move: 3, pass: 1} (just {pass: 4} when
// the move is a pass). The result comes from the game unless it is still
// running, in which case `result` is used.
static GameRecord makeRecord(const std::vector<int>& moves, float komi = 7.5f, float result = 1.0f) {
    const int n = 5, area = 25;
    Game game(n, komi, defaultMaxMoves(n));
```

with:

```cpp
// Plays `moves` on a 5x5 Game and records it as self-play would: the board
// before each move, and visit counts {move: 3, pass: 1} (just {pass: 4} when
// the move is a pass). The result comes from the game unless it is still
// running, in which case `result` is used. finalOwnership is the board after
// the last move, as self-play records it.
static GameRecord makeRecord(const std::vector<int>& moves, float komi = 7.5f, float result = 1.0f) {
    const int n = 5, area = 25;
    Game game(n, komi, defaultMaxMoves(n));
```

In `tests/test_replay_buffer.cpp`, replace:

```cpp
        game.play(m);
    }
    rec.resultForBlack = game.isTerminal() ? game.resultForBlack() : result;
    return rec;
}

```

with:

```cpp
        game.play(m);
    }
    rec.resultForBlack = game.isTerminal() ? game.resultForBlack() : result;
    rec.finalOwnership.resize(area);
    game.board().ownership(rec.finalOwnership.data());
    return rec;
}

```

In `tests/test_replay_buffer.cpp`, replace:

```cpp
    assert(batch.input.size() == static_cast<size_t>(B) * numPlanes(H) * 25);
    assert(batch.pi.size() == static_cast<size_t>(B) * 26);
    assert(batch.z.size() == static_cast<size_t>(B));
    for (int b = 0; b < B; ++b) {
        double total = 0.0;
        for (int a = 0; a < 26; ++a) total += batch.pi[b * 26 + a];
```

with:

```cpp
    assert(batch.input.size() == static_cast<size_t>(B) * numPlanes(H) * 25);
    assert(batch.pi.size() == static_cast<size_t>(B) * 26);
    assert(batch.z.size() == static_cast<size_t>(B));
    assert(batch.ownership.size() == static_cast<size_t>(B) * 25);
    for (float o : batch.ownership) assert(o == -1.0f || o == 0.0f || o == 1.0f);
    for (int b = 0; b < B; ++b) {
        double total = 0.0;
        for (int a = 0; a < 26; ++a) total += batch.pi[b * 26 + a];
```

In `tests/test_replay_buffer.cpp`, replace:

```cpp
    assert(near(static_cast<double>(fromGame1) / draws, 0.75, 0.03));
}

static void test_save_load_round_trip() {
    ReplayBuffer buf(9);
    buf.add(lengthGame(4, 0));
```

with:

```cpp
    assert(near(static_cast<double>(fromGame1) / draws, 0.75, 0.03));
}

// Black C4 (6), White D2 (18), Black C3 (12): the one empty region touches
// both colors, so only the three stones are owned: +1 on 6 and 12, -1 on 18.
// The target is then exactly "own stones minus opponent stones" of the final
// board, which encodePlanes computes independently, for every symmetry and
// both sides to move.
static void test_ownership_target_perspective_and_symmetry() {
    ReplayBuffer buf(100);
    buf.add(makeRecord({6, 18, 12}));
    const GameRecord& g = buf.game(0);
    std::vector<int8_t> expected(25, 0);
    expected[6] = expected[12] = 1;
    expected[18] = -1;
    assert(g.finalOwnership == expected);

    // The final board (after move 3) as stone bytes, for the oracle.
    std::vector<uint8_t> finalStones(25, static_cast<uint8_t>(Color::Empty));
    finalStones[6] = finalStones[12] = static_cast<uint8_t>(Color::Black);
    finalStones[18] = static_cast<uint8_t>(Color::White);
    const int H = 1, area = 25;
    std::vector<float> input(numPlanes(H) * area), pi(26), own(area), ref(numPlanes(H) * area);
    float z;
    for (int ply = 0; ply < 3; ++ply)
        for (int sym = 0; sym < kNumSymmetries; ++sym) {
            buf.encodeSample(0, ply, H, sym, input.data(), pi.data(), &z, own.data());
            encodePlanes({finalStones.data()}, 5, g.toMoveAt(ply), H, sym, ref.data());
            for (int p = 0; p < area; ++p) assert(own[p] == ref[p] - ref[area + p]);
            // By hand: White to move (ply 1) sees every owner negated.
            const float sign = ply == 1 ? -1.0f : 1.0f;
            assert(own[transformPoint(sym, 6, 5)] == sign && own[transformPoint(sym, 18, 5)] == -sign);
        }
}

// Black walls off columns A-C, White columns D-E:
//   C5 D5 C4 D4 C3 D3 C2 D2 C1 D1. Every row of the final board is "+++--"
// (Black's view). The target is not symmetric, so a wrong mapping shows.
static void test_ownership_target_with_territory() {
    ReplayBuffer buf(100);
    buf.add(makeRecord({2, 3, 7, 8, 12, 13, 17, 18, 22, 23}));
    const GameRecord& g = buf.game(0);
    for (int p = 0; p < 25; ++p) assert(g.finalOwnership[p] == (p % 5 < 3 ? 1 : -1));
    std::vector<float> input(numPlanes(2) * 25), pi(26), own(25);
    float z;
    auto row = [&](int r) {
        std::string s;
        for (int c = 0; c < 5; ++c) s += own[r * 5 + c] > 0 ? '+' : '-';
        return s;
    };
    // Ply 3, White to move, no symmetry: White's view of the same board.
    buf.encodeSample(0, 3, 2, 0, input.data(), pi.data(), &z, own.data());
    for (int r = 0; r < 5; ++r) assert(row(r) == "---++");
    // Symmetry 4 mirrors left-right: White's two columns move to the left.
    buf.encodeSample(0, 3, 2, 4, input.data(), pi.data(), &z, own.data());
    for (int r = 0; r < 5; ++r) assert(row(r) == "++---");
    // Symmetry 1 turns the board a quarter clockwise: columns become rows,
    // so the top three rows are White's area (column A became row 5).
    buf.encodeSample(0, 3, 2, 1, input.data(), pi.data(), &z, own.data());
    for (int r = 0; r < 5; ++r) assert(row(r) == (r < 3 ? "-----" : "+++++"));
    // Ply 4, Black to move: Black's view.
    buf.encodeSample(0, 4, 2, 0, input.data(), pi.data(), &z, own.data());
    for (int r = 0; r < 5; ++r) assert(row(r) == "+++--");
}

static void test_save_load_round_trip() {
    ReplayBuffer buf(9);
    buf.add(lengthGame(4, 0));
```

In `tests/test_replay_buffer.cpp`, replace:

```cpp
        const GameRecord& b = back.game(g);
        assert(a.boardSize == b.boardSize && a.komi == b.komi && a.moves == b.moves);
        assert(a.stones == b.stones && a.visits == b.visits && a.resultForBlack == b.resultForBlack);
    }
    TrainingBatch x, y;
    Rng r1(77), r2(77);
```

with:

```cpp
        const GameRecord& b = back.game(g);
        assert(a.boardSize == b.boardSize && a.komi == b.komi && a.moves == b.moves);
        assert(a.stones == b.stones && a.visits == b.visits && a.resultForBlack == b.resultForBlack);
        assert(a.finalOwnership == b.finalOwnership);
    }
    TrainingBatch x, y;
    Rng r1(77), r2(77);
```

In `tests/test_replay_buffer.cpp`, replace:

```cpp
    assert(std::memcmp(x.input.data(), y.input.data(), x.input.size() * sizeof(float)) == 0);
    assert(std::memcmp(x.pi.data(), y.pi.data(), x.pi.size() * sizeof(float)) == 0);
    assert(std::memcmp(x.z.data(), y.z.data(), x.z.size() * sizeof(float)) == 0);

    // A truncated stream is an error, not a crash.
    std::istringstream cut(bytes.substr(0, bytes.size() / 2));
```

with:

```cpp
    assert(std::memcmp(x.input.data(), y.input.data(), x.input.size() * sizeof(float)) == 0);
    assert(std::memcmp(x.pi.data(), y.pi.data(), x.pi.size() * sizeof(float)) == 0);
    assert(std::memcmp(x.z.data(), y.z.data(), x.z.size() * sizeof(float)) == 0);
    assert(x.ownership == y.ownership);

    // The older layout (state.bin versions 1 and 2) has no ownership; loading
    // it recomputes each game's by replaying the moves, and gets the same.
    std::ostringstream oldOs;
    BinaryWriter oldW(oldOs);
    buf.save(oldW, false);
    assert(oldOs.str().size() == bytes.size() - 25 * buf.games());
    std::istringstream oldIs(oldOs.str());
    BinaryReader oldR(oldIs);
    ReplayBuffer recomputed = ReplayBuffer::load(oldR, false);
    for (size_t g = 0; g < buf.games(); ++g)
        assert(recomputed.game(g).finalOwnership == buf.game(g).finalOwnership);

    // A truncated stream is an error, not a crash.
    std::istringstream cut(bytes.substr(0, bytes.size() / 2));
```

In `tests/test_replay_buffer.cpp`, replace:

```cpp
    assert(threw);
}

int main() {
    test_add_evicts_oldest_games();
    test_oversized_game_is_kept_alone();
```

with:

```cpp
    assert(threw);
}

// An older buffer whose moves do not replay cannot get its ownership: an
// error naming the game, not a wrong target.
static void test_old_layout_that_does_not_replay() {
    ReplayBuffer buf(100);
    buf.add(makeRecord({6, 18, 12}));
    GameRecord bad = makeRecord({6, 18});
    bad.moves[1] = 6;   // White on Black's stone
    buf.add(bad);
    std::ostringstream os;
    BinaryWriter w(os);
    buf.save(w, false);
    std::istringstream is(os.str());
    BinaryReader r(is);
    std::string msg;
    try {
        ReplayBuffer::load(r, false);
    } catch (const std::runtime_error& e) {
        msg = e.what();
    }
    assert(msg == "replay buffer: game 1 does not replay: move 2 (B4) is illegal");
}

int main() {
    test_add_evicts_oldest_games();
    test_oversized_game_is_kept_alone();
```

In `tests/test_replay_buffer.cpp`, replace:

```cpp
    test_exact_tie_gives_zero_z_everywhere();
    test_sample_shapes_and_pi_sums();
    test_sampling_is_uniform_over_positions();
    test_save_load_round_trip();
    std::printf("test_replay_buffer: all tests passed\n");
    return 0;
}
```

with:

```cpp
    test_exact_tie_gives_zero_z_everywhere();
    test_sample_shapes_and_pi_sums();
    test_sampling_is_uniform_over_positions();
    test_ownership_target_perspective_and_symmetry();
    test_ownership_target_with_territory();
    test_save_load_round_trip();
    test_old_layout_that_does_not_replay();
    std::printf("test_replay_buffer: all tests passed\n");
    return 0;
}
```

`tests/test_run_state.cpp`: old files are now made from version-3 bytes by `olderStateBytes`. The buffer section (it starts at byte 24) is swapped for its old layout, and for version 1 the references and bot list are dropped from the end. The "newer version" probe becomes 4.

In `tests/test_run_state.cpp`, replace:

```cpp
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
```

with:

```cpp
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
```

In `tests/test_run_state.cpp`, replace:

```cpp
#include "azgo/rng.hpp"
#include "azgo/run_state.hpp"
#include "azgo/selfplay.hpp"

using namespace azgo;
namespace fs = std::filesystem;
```

with:

```cpp
#include "azgo/rng.hpp"
#include "azgo/run_state.hpp"
#include "azgo/selfplay.hpp"
#include "azgo/serialize.hpp"

using namespace azgo;
namespace fs = std::filesystem;
```

In `tests/test_run_state.cpp`, replace:

```cpp
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// A state with real content: two short self-play games, momentum buffers,
// arena results for three pairs (one with a bot) and a reference set with
// two bots and a promoted generation.
```

with:

```cpp
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// The bytes an older version wrote for `state`, given the current (version 3)
// bytes: the replay buffer without ownership (version 2), and for version 1
// also without the reference set and the bot list at the end.
static std::string olderStateBytes(const std::string& v3, const TrainingState& state, int version) {
    std::ostringstream with, without;
    BinaryWriter ww(with), wo(without);
    state.buffer.save(ww, true);
    state.buffer.save(wo, false);
    const size_t header = 4 + 4 + 4 + 4 + 8;   // magic, version, generation, best, samples
    assert(v3.compare(header, with.str().size(), with.str()) == 0);
    std::string tail = v3.substr(header + with.str().size());
    if (version == 1)
        tail.resize(tail.size() - 8 - 4 * (state.references.size() + state.botSimulations.size()));
    std::string out = v3.substr(0, header) + without.str() + tail;
    out[4] = static_cast<char>(version);
    return out;
}

// A state with real content: two short self-play games, momentum buffers,
// arena results for three pairs (one with a bot) and a reference set with
// two bots and a promoted generation.
```

In `tests/test_run_state.cpp`, replace:

```cpp

static bool sameGame(const GameRecord& x, const GameRecord& y) {
    return x.boardSize == y.boardSize && x.komi == y.komi && x.moves == y.moves &&
           x.stones == y.stones && x.visits == y.visits && x.resultForBlack == y.resultForBlack;
}

static void assertSameState(const TrainingState& x, const TrainingState& y) {
```

with:

```cpp

static bool sameGame(const GameRecord& x, const GameRecord& y) {
    return x.boardSize == y.boardSize && x.komi == y.komi && x.moves == y.moves &&
           x.stones == y.stones && x.visits == y.visits && x.resultForBlack == y.resultForBlack &&
           x.finalOwnership == y.finalOwnership;
}

static void assertSameState(const TrainingState& x, const TrainingState& y) {
```

In `tests/test_run_state.cpp`, replace:

```cpp
    original.buffer.sample(16, 2, ra, a);
    loaded.buffer.sample(16, 2, rb, b);
    assert(a.size == 16 && b.size == 16);
    assert(a.input == b.input && a.pi == b.pi && a.z == b.z);

    // Saving the loaded state reproduces the same bytes.
    std::string path2 = (dir / "again.bin").string();
```

with:

```cpp
    original.buffer.sample(16, 2, ra, a);
    loaded.buffer.sample(16, 2, rb, b);
    assert(a.size == 16 && b.size == 16);
    assert(a.input == b.input && a.pi == b.pi && a.z == b.z && a.ownership == b.ownership);

    // Saving the loaded state reproduces the same bytes.
    std::string path2 = (dir / "again.bin").string();
```

In `tests/test_run_state.cpp`, replace:

```cpp

    // A newer format version.
    std::string newer = bytes;
    newer[4] = 3;   // version is the u32 after the magic, little-endian
    writeBytes(wrong, newer);
    msg = loadError(wrong.string());
    assert(msg.find("version") != std::string::npos);
```

with:

```cpp

    // A newer format version.
    std::string newer = bytes;
    newer[4] = 4;   // version is the u32 after the magic, little-endian
    writeBytes(wrong, newer);
    msg = loadError(wrong.string());
    assert(msg.find("version") != std::string::npos);
```

In `tests/test_run_state.cpp`, replace:

```cpp
    assert((TrainingState{}.references == std::vector<int>{0}));
}

// A version-1 file is a version-2 file without the reference set and the bot
// list at the end: a u32 count and the i32 entries of each.
static void test_version1_gets_default_references() {
    fs::path dir = freshDir("version1");
    TrainingState original = makeState();
    saveTrainingState((dir / "v2.bin").string(), original);
    std::string bytes = readBytes(dir / "v2.bin");
    std::string v1 = bytes.substr(
        0, bytes.size() - 8 - 4 * (original.references.size() + original.botSimulations.size()));
    v1[4] = 1;
    fs::path path = dir / "v1.bin";
    writeBytes(path, v1);

```

with:

```cpp
    assert((TrainingState{}.references == std::vector<int>{0}));
}

// A version-1 file has no ownership in the buffer and no reference set or
// bot list at the end (a u32 count and the i32 entries of each).
static void test_version1_gets_default_references() {
    fs::path dir = freshDir("version1");
    TrainingState original = makeState();
    saveTrainingState((dir / "v3.bin").string(), original);
    std::string v1 = olderStateBytes(readBytes(dir / "v3.bin"), original, 1);
    fs::path path = dir / "v1.bin";
    writeBytes(path, v1);

```

In `tests/test_run_state.cpp`, replace:

```cpp
    assert((loadTrainingState(path.string()).references == std::vector<int>{0}));   // no bots
    assert(loadTrainingState(path.string()).botSimulations.empty());

    // Saving it again writes version 2.
    saveTrainingState(path.string(), loaded);
    assert(readBytes(path)[4] == 2);
    assert((loadTrainingState(path.string()).references == std::vector<int>{-2, -1, 0}));
    fs::remove_all(dir);
}

static void test_rejects_bad_references() {
    fs::path dir = freshDir("badrefs");
    fs::path path = dir / "state.bin";
```

with:

```cpp
    assert((loadTrainingState(path.string()).references == std::vector<int>{0}));   // no bots
    assert(loadTrainingState(path.string()).botSimulations.empty());

    // Saving it again writes version 3.
    saveTrainingState(path.string(), loaded);
    assert(readBytes(path)[4] == 3);
    assert((loadTrainingState(path.string()).references == std::vector<int>{-2, -1, 0}));
    fs::remove_all(dir);
}

// A version-2 file has everything but the buffer's ownership, which loading
// recomputes by replaying each game: the result equals what self-play
// recorded, and saving writes version 3 again, byte for byte.
static void test_version2_recomputes_ownership() {
    fs::path dir = freshDir("version2");
    TrainingState original = makeState();
    saveTrainingState((dir / "v3.bin").string(), original);
    const std::string v3 = readBytes(dir / "v3.bin");
    fs::path path = dir / "v2.bin";
    writeBytes(path, olderStateBytes(v3, original, 2));
    TrainingState loaded = loadTrainingState(path.string());
    assertSameState(original, loaded);   // finalOwnership included
    for (size_t i = 0; i < loaded.buffer.games(); ++i)
        assert(loaded.buffer.game(i).finalOwnership.size() == 25);
    saveTrainingState(path.string(), loaded);
    assert(readBytes(path) == v3);

    // A game that does not replay makes the file unreadable, naming it.
    TrainingState broken = original;
    broken.buffer = ReplayBuffer(1000);
    GameRecord bad = original.buffer.game(0);
    bad.stones[25] ^= 1;   // ply 1, point A5: a position the moves do not produce
    broken.buffer.add(bad);
    saveTrainingState(path.string(), broken);
    writeBytes(path, olderStateBytes(readBytes(path), broken, 2));
    const std::string msg = loadError(path.string());
    assert(msg.find(path.string()) != std::string::npos);
    assert(msg.find("game 0 does not replay: position 1 differs") != std::string::npos);
    fs::remove_all(dir);
}

static void test_rejects_bad_references() {
    fs::path dir = freshDir("badrefs");
    fs::path path = dir / "state.bin";
```

In `tests/test_run_state.cpp`, replace:

```cpp
    test_crash_leftovers();
    test_initial_references();
    test_version1_gets_default_references();
    test_rejects_bad_references();
    test_paths();
    std::printf("test_run_state: all tests passed\n");
```

with:

```cpp
    test_crash_leftovers();
    test_initial_references();
    test_version1_gets_default_references();
    test_version2_recomputes_ownership();
    test_rejects_bad_references();
    test_paths();
    std::printf("test_run_state: all tests passed\n");
```

`tests/test_trainer.cpp`: the resume-from-version-1 test builds its version-1 file the same way.

In `tests/test_trainer.cpp`, replace:

```cpp
#include "azgo/elo.hpp"
#include "azgo/network.hpp"
#include "azgo/run_state.hpp"
#include "azgo/trainer.hpp"

using namespace azgo;
```

with:

```cpp
#include "azgo/elo.hpp"
#include "azgo/network.hpp"
#include "azgo/run_state.hpp"
#include "azgo/serialize.hpp"
#include "azgo/trainer.hpp"

using namespace azgo;
```

In `tests/test_trainer.cpp`, replace:

```cpp
    c.botSimulations = {};   // the old run had no bots
    startQuiet(c, dir)->runUntil(1);

    // state.bin as version 1: without the reference set {0} (u32 count + one i32)
    // and the empty bot list (u32 count).
    const std::string bytes = readFile(statePath(dir.str()));
    std::string v1 = bytes.substr(0, bytes.size() - 12);
    v1[4] = 1;
    writeFile(statePath(dir.str()), v1);
    // config.cfg without the keys the old code did not know.
```

with:

```cpp
    c.botSimulations = {};   // the old run had no bots
    startQuiet(c, dir)->runUntil(1);

    // state.bin as version 1: the buffer without ownership, and without the
    // reference set {0} (u32 count + one i32) and the empty bot list (u32 count).
    const std::string bytes = readFile(statePath(dir.str()));
    const TrainingState current = loadTrainingState(statePath(dir.str()));
    std::ostringstream with, without;
    BinaryWriter ww(with), wo(without);
    current.buffer.save(ww, true);
    current.buffer.save(wo, false);
    const size_t header = 24;   // magic, version, generation, best generation, samples trained
    assert(bytes.compare(header, with.str().size(), with.str()) == 0);
    std::string v1 = bytes.substr(0, header) + without.str() +
                     bytes.substr(header + with.str().size(), bytes.size() - header - with.str().size() - 12);
    v1[4] = 1;
    writeFile(statePath(dir.str()), v1);
    // config.cfg without the keys the old code did not know.
```

In `tests/test_trainer.cpp`, replace:

```cpp
    assert(!pairs.count({-1, 1}));           // generation 1 ran before the bot existed
    TrainingState s = loadTrainingState(statePath(dir.str()));
    assert(s.references.front() == -1);
    assert(readFile(statePath(dir.str()))[4] == 2);   // saved as version 2
    assert(readFile(dir.file("references.csv")).find("\n-1,bot,100,0,") != std::string::npos);
}

```

with:

```cpp
    assert(!pairs.count({-1, 1}));           // generation 1 ran before the bot existed
    TrainingState s = loadTrainingState(statePath(dir.str()));
    assert(s.references.front() == -1);
    assert(readFile(statePath(dir.str()))[4] == 3);   // saved as version 3
    assert(readFile(dir.file("references.csv")).find("\n-1,bot,100,0,") != std::string::npos);
}

```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cmake --build build -j`
Expected: the build fails in the four test files, for example with:

```
tests/test_selfplay.cpp:60:16: error: no member named 'finalOwnership' in 'azgo::GameRecord'
tests/test_selfplay.cpp:136:12: error: use of undeclared identifier 'replayFinalOwnership'
```

- [ ] **Step 3: Record the final ownership in self-play, and replay it for old records**

In `include/azgo/selfplay.hpp`, replace:

```cpp
// Self-play: the network plays both sides against itself, choosing every
// move by MCTS. Each position is stored with the search's visit counts
// (the improved policy target pi); once the game ends, every position also
// gets the final result z seen from its own side to move. Root Dirichlet
// noise and temperature-1 sampling in the opening keep the games varied.
#pragma once
#include <cstdint>
#include <utility>
```

with:

```cpp
// Self-play: the network plays both sides against itself, choosing every
// move by MCTS. Each position is stored with the search's visit counts
// (the improved policy target pi); once the game ends, every position also
// gets the final result z seen from its own side to move, and the record
// keeps who owns each point of the final board (the ownership target). Root
// Dirichlet noise and temperature-1 sampling in the opening keep the games
// varied.
#pragma once
#include <cstdint>
#include <utility>
```

In `include/azgo/selfplay.hpp`, replace:

```cpp
    std::vector<uint8_t> stones;   // positions() * n*n Color bytes
    std::vector<std::vector<std::pair<int16_t, uint16_t>>> visits;  // per ply: (action, count), count > 0
    float resultForBlack = 0;      // +1, -1, or 0

    int positions() const { return static_cast<int>(moves.size()); }
    const uint8_t* stonesAt(int ply) const { return stones.data() + static_cast<size_t>(ply) * boardSize * boardSize; }
```

with:

```cpp
    std::vector<uint8_t> stones;   // positions() * n*n Color bytes
    std::vector<std::vector<std::pair<int16_t, uint16_t>>> visits;  // per ply: (action, count), count > 0
    float resultForBlack = 0;      // +1, -1, or 0
    std::vector<int8_t> finalOwnership;   // n*n, Black's view: Board::ownership of the final board

    int positions() const { return static_cast<int>(moves.size()); }
    const uint8_t* stonesAt(int ply) const { return stones.data() + static_cast<size_t>(ply) * boardSize * boardSize; }
```

In `include/azgo/selfplay.hpp`, replace:

```cpp
// Plays one self-play game from the empty board with root noise and tree reuse.
GameRecord playSelfPlayGame(Evaluator& evaluator, const SelfPlayConfig& cfg, Rng& rng);

} // namespace azgo
```

with:

```cpp
// Plays one self-play game from the empty board with root noise and tree reuse.
GameRecord playSelfPlayGame(Evaluator& evaluator, const SelfPlayConfig& cfg, Rng& rng);

// The final board's ownership (Black's view), found by replaying the record's
// moves from the empty board with Game. Records saved before finalOwnership
// existed get it this way. Throws std::runtime_error if a stored position
// differs from the replayed one or a move is illegal.
std::vector<int8_t> replayFinalOwnership(const GameRecord& record);

} // namespace azgo
```

In `src/selfplay.cpp`, replace:

```cpp
#include "azgo/selfplay.hpp"
#include <algorithm>
#include <cassert>
#include <limits>
#include "azgo/game.hpp"

namespace azgo {
```

with:

```cpp
#include "azgo/selfplay.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include "azgo/game.hpp"

namespace azgo {
```

In `src/selfplay.cpp`, replace:

```cpp
        mcts.advance(action);   // reuse the subtree under the move just played
    }
    rec.resultForBlack = game.resultForBlack();
    return rec;
}

} // namespace azgo
```

with:

```cpp
        mcts.advance(action);   // reuse the subtree under the move just played
    }
    rec.resultForBlack = game.resultForBlack();
    // Who owns each point at the end: the per-point target of the ownership head.
    rec.finalOwnership.resize(static_cast<size_t>(area));
    game.board().ownership(rec.finalOwnership.data());
    return rec;
}

std::vector<int8_t> replayFinalOwnership(const GameRecord& record) {
    const int n = record.boardSize, area = n * n;
    // The record's own length as the move cap: the cap cannot end the replay
    // early, while two passes before the last move (impossible in a record
    // self-play wrote) make the next move illegal.
    Game game(n, record.komi, std::max(1, record.positions()));
    std::vector<uint8_t> stones(static_cast<size_t>(area));
    for (int ply = 0; ply < record.positions(); ++ply) {
        game.board().exportStones(stones.data());
        if (std::memcmp(stones.data(), record.stonesAt(ply), static_cast<size_t>(area)) != 0)
            throw std::runtime_error("position " + std::to_string(ply) +
                                     " differs from the replayed one");
        const int move = record.moves[static_cast<size_t>(ply)];
        if (!game.isLegal(move))
            throw std::runtime_error("move " + std::to_string(ply + 1) + " (" +
                                     actionToString(move, n) + ") is illegal");
        game.play(move);
    }
    std::vector<int8_t> owner(static_cast<size_t>(area));
    game.board().ownership(owner.data());
    return owner;
}

} // namespace azgo
```

- [ ] **Step 4: Encode, save and load the target in the replay buffer**

In `include/azgo/replay_buffer.hpp`, replace:

```cpp
// game's earlier positions, so storing history costs nothing extra. Each
// sample is shown to the network under a random one of the board's 8
// symmetries, which multiplies the data for free (Go has no preferred side).
#pragma once
#include <cstddef>
#include <deque>
```

with:

```cpp
// game's earlier positions, so storing history costs nothing extra. Each
// sample is shown to the network under a random one of the board's 8
// symmetries, which multiplies the data for free (Go has no preferred side).
// The ownership target travels with the position: the final board's owners,
// turned to the side to move's view and moved by the same symmetry.
#pragma once
#include <cstddef>
#include <deque>
```

In `include/azgo/replay_buffer.hpp`, replace:

```cpp
    std::vector<float> input;   // [size][planes][area]
    std::vector<float> pi;      // [size][area + 1]
    std::vector<float> z;       // [size]
};

// Sliding window of whole games. A sample is (game, ply); its history planes
```

with:

```cpp
    std::vector<float> input;   // [size][planes][area]
    std::vector<float> pi;      // [size][area + 1]
    std::vector<float> z;       // [size]
    std::vector<float> ownership;   // [size][area]: final owner of each point, side to move's view
};

// Sliding window of whole games. A sample is (game, ply); its history planes
```

In `include/azgo/replay_buffer.hpp`, replace:

```cpp
    // B samples uniformly over positions (with replacement), each with a
    // uniformly random symmetry. Asserts positions() > 0.
    void sample(int B, int history, Rng& rng, TrainingBatch& out) const;
    // One sample into caller buffers: input numPlanes(H)*area, pi area+1, *z.
    void encodeSample(size_t gameIndex, int ply, int history, int sym,
                      float* input, float* pi, float* z) const;

    void save(BinaryWriter& w) const;
    static ReplayBuffer load(BinaryReader& r);

private:
    size_t capacity_;
```

with:

```cpp
    // B samples uniformly over positions (with replacement), each with a
    // uniformly random symmetry. Asserts positions() > 0.
    void sample(int B, int history, Rng& rng, TrainingBatch& out) const;
    // One sample into caller buffers: input numPlanes(H)*area, pi area+1, *z,
    // and, unless null, ownership area (finalOwnership, negated when White is
    // to move, each point moved where the symmetry moves it).
    void encodeSample(size_t gameIndex, int ply, int history, int sym,
                      float* input, float* pi, float* z, float* ownership = nullptr) const;

    // withOwnership = false writes the layout of state.bin versions 1 and 2,
    // which stored no ownership; tests use it to make old files.
    void save(BinaryWriter& w, bool withOwnership = true) const;
    // withOwnership = false reads that older layout and recomputes each
    // game's finalOwnership with replayFinalOwnership (a std::runtime_error
    // naming the game if one does not replay).
    static ReplayBuffer load(BinaryReader& r, bool withOwnership = true);

private:
    size_t capacity_;
```

In `src/replay_buffer.cpp`, replace:

```cpp
    assert(game.positions() > 0);
    assert(game.stones.size() == static_cast<size_t>(game.positions()) * area);
    assert(game.visits.size() == static_cast<size_t>(game.positions()));
    assert(games_.empty() || games_.front().boardSize == game.boardSize);
    (void)area;
    positions_ += game.positions();
```

with:

```cpp
    assert(game.positions() > 0);
    assert(game.stones.size() == static_cast<size_t>(game.positions()) * area);
    assert(game.visits.size() == static_cast<size_t>(game.positions()));
    assert(game.finalOwnership.size() == area);
    assert(games_.empty() || games_.front().boardSize == game.boardSize);
    (void)area;
    positions_ += game.positions();
```

In `src/replay_buffer.cpp`, replace:

```cpp
}

void ReplayBuffer::encodeSample(size_t gameIndex, int ply, int history, int sym,
                                float* input, float* pi, float* z) const {
    const GameRecord& g = games_[gameIndex];
    assert(ply >= 0 && ply < g.positions());
    const int n = g.boardSize;
```

with:

```cpp
}

void ReplayBuffer::encodeSample(size_t gameIndex, int ply, int history, int sym,
                                float* input, float* pi, float* z, float* ownership) const {
    const GameRecord& g = games_[gameIndex];
    assert(ply >= 0 && ply < g.positions());
    const int n = g.boardSize;
```

In `src/replay_buffer.cpp`, replace:

```cpp
        pi[transformAction(sym, av.first, n)] = static_cast<float>(av.second) / static_cast<float>(total);

    *z = g.zAt(ply);
}

void ReplayBuffer::sample(int B, int history, Rng& rng, TrainingBatch& out) const {
```

with:

```cpp
        pi[transformAction(sym, av.first, n)] = static_cast<float>(av.second) / static_cast<float>(total);

    *z = g.zAt(ply);

    // Ownership, like z, from the side to move's view, and moved like the
    // input planes, so every point's target stays on its point.
    if (ownership) {
        assert(g.finalOwnership.size() == static_cast<size_t>(n) * n);
        const float sign = g.toMoveAt(ply) == Color::Black ? 1.0f : -1.0f;
        for (int p = 0; p < n * n; ++p)
            ownership[transformPoint(sym, p, n)] = sign * g.finalOwnership[static_cast<size_t>(p)];
    }
}

void ReplayBuffer::sample(int B, int history, Rng& rng, TrainingBatch& out) const {
```

In `src/replay_buffer.cpp`, replace:

```cpp
    const int n = games_.front().boardSize;
    const size_t inputStride = static_cast<size_t>(numPlanes(history)) * n * n;
    const size_t piStride = static_cast<size_t>(n) * n + 1;
    out.size = B;
    out.input.resize(B * inputStride);
    out.pi.resize(B * piStride);
    out.z.resize(B);
    for (int b = 0; b < B; ++b) {
        // Uniform over positions, not games: long games are sampled more.
        const size_t global = static_cast<size_t>(rng.below(static_cast<int>(positions_)));
```

with:

```cpp
    const int n = games_.front().boardSize;
    const size_t inputStride = static_cast<size_t>(numPlanes(history)) * n * n;
    const size_t piStride = static_cast<size_t>(n) * n + 1;
    const size_t area = static_cast<size_t>(n) * n;
    out.size = B;
    out.input.resize(B * inputStride);
    out.pi.resize(B * piStride);
    out.z.resize(B);
    out.ownership.resize(B * area);
    for (int b = 0; b < B; ++b) {
        // Uniform over positions, not games: long games are sampled more.
        const size_t global = static_cast<size_t>(rng.below(static_cast<int>(positions_)));
```

In `src/replay_buffer.cpp`, replace:

```cpp
            std::upper_bound(firstPosition_.begin(), firstPosition_.end(), global) - firstPosition_.begin() - 1);
        const int ply = static_cast<int>(global - firstPosition_[g]);
        encodeSample(g, ply, history, sym, out.input.data() + b * inputStride,
                     out.pi.data() + b * piStride, out.z.data() + b);
    }
}

// Layout: u64 capacity, u64 totalAdded (two's complement of the i64), u32 games,
// then per game: i32 boardSize, f32 komi, u32 moves + i32 each, the stones as
// raw bytes (positions * n * n), per ply u32 count + u32 (action << 16 | count)
// each, f32 resultForBlack.
void ReplayBuffer::save(BinaryWriter& w) const {
    w.u64(capacity_);
    w.u64(static_cast<uint64_t>(totalAdded_));
    w.u32(static_cast<uint32_t>(games_.size()));
```

with:

```cpp
            std::upper_bound(firstPosition_.begin(), firstPosition_.end(), global) - firstPosition_.begin() - 1);
        const int ply = static_cast<int>(global - firstPosition_[g]);
        encodeSample(g, ply, history, sym, out.input.data() + b * inputStride,
                     out.pi.data() + b * piStride, out.z.data() + b, out.ownership.data() + b * area);
    }
}

// Layout: u64 capacity, u64 totalAdded (two's complement of the i64), u32 games,
// then per game: i32 boardSize, f32 komi, u32 moves + i32 each, the stones as
// raw bytes (positions * n * n), per ply u32 count + u32 (action << 16 | count)
// each, f32 resultForBlack, then (withOwnership) finalOwnership as n * n
// signed bytes.
void ReplayBuffer::save(BinaryWriter& w, bool withOwnership) const {
    w.u64(capacity_);
    w.u64(static_cast<uint64_t>(totalAdded_));
    w.u32(static_cast<uint32_t>(games_.size()));
```

In `src/replay_buffer.cpp`, replace:

```cpp
                w.u32((static_cast<uint32_t>(static_cast<uint16_t>(av.first)) << 16) | av.second);
        }
        w.f32(g.resultForBlack);
    }
}

ReplayBuffer ReplayBuffer::load(BinaryReader& r) {
    const uint64_t capacity = r.u64();
    if (capacity == 0) throw std::runtime_error("replay buffer: capacity is 0");
    ReplayBuffer buf(static_cast<size_t>(capacity));
```

with:

```cpp
                w.u32((static_cast<uint32_t>(static_cast<uint16_t>(av.first)) << 16) | av.second);
        }
        w.f32(g.resultForBlack);
        if (withOwnership) {
            assert(g.finalOwnership.size() == static_cast<size_t>(g.boardSize) * g.boardSize);
            w.bytes(g.finalOwnership.data(), g.finalOwnership.size());
        }
    }
}

ReplayBuffer ReplayBuffer::load(BinaryReader& r, bool withOwnership) {
    const uint64_t capacity = r.u64();
    if (capacity == 0) throw std::runtime_error("replay buffer: capacity is 0");
    ReplayBuffer buf(static_cast<size_t>(capacity));
```

In `src/replay_buffer.cpp`, replace:

```cpp
            }
        }
        g.resultForBlack = r.f32();
        // Straight into the deque: the saved window already respected the
        // capacity, and totalAdded is restored below rather than recounted.
        buf.positions_ += g.positions();
```

with:

```cpp
            }
        }
        g.resultForBlack = r.f32();
        if (withOwnership) {
            g.finalOwnership.resize(static_cast<size_t>(area));
            r.bytes(g.finalOwnership.data(), g.finalOwnership.size());
            for (int8_t o : g.finalOwnership)
                if (o < -1 || o > 1) throw std::runtime_error("replay buffer: bad ownership value");
        } else {
            // An older file: recompute what self-play would have recorded.
            try {
                g.finalOwnership = replayFinalOwnership(g);
            } catch (const std::runtime_error& e) {
                throw std::runtime_error("replay buffer: game " + std::to_string(i) +
                                         " does not replay: " + e.what());
            }
        }
        // Straight into the deque: the saved window already respected the
        // capacity, and totalAdded is restored below rather than recounted.
        buf.positions_ += g.positions();
```

- [ ] **Step 5: Move `state.bin` to version 3**

In `include/azgo/run_state.hpp`, replace:

```cpp
// The reference set of a new run: every bot (-numBots, ..., -1) and generation 0.
std::vector<int> initialReferences(int numBots);

// Binary, magic "AZST" (u32 0x54535A41), version u32 = 2 (references, then the bot list, come last). Atomic replace.
void saveTrainingState(const std::string& path, const TrainingState& state);
// Reads version 2 and version 1, which predates the reference set and the bot
// list: a version-1 file gets botSimulations = `bots` and references =
// initialReferences(bots.size()), so pass the bot list the run's config
// defines. Throws std::runtime_error naming the path.
TrainingState loadTrainingState(const std::string& path, const std::vector<int>& bots = {});

// Seed phases for deriveSeed(cfg.seed, generation, phase, index). Part of a
```

with:

```cpp
// The reference set of a new run: every bot (-numBots, ..., -1) and generation 0.
std::vector<int> initialReferences(int numBots);

// Binary, magic "AZST" (u32 0x54535A41), version u32 = 3 (the replay buffer
// stores each game's final ownership; references, then the bot list, come
// last). Atomic replace.
void saveTrainingState(const std::string& path, const TrainingState& state);
// Reads versions 3, 2 and 1. Versions 1 and 2 predate stored ownership: each
// game's is recomputed by replaying its moves (a game that does not replay is
// an error). Version 1 also predates the reference set and the bot list: it
// gets botSimulations = `bots` and references = initialReferences(bots.size()),
// so pass the bot list the run's config defines. Throws std::runtime_error
// naming the path.
TrainingState loadTrainingState(const std::string& path, const std::vector<int>& bots = {});

// Seed phases for deriveSeed(cfg.seed, generation, phase, index). Part of a
```

In `src/run_state.cpp`, replace:

```cpp

namespace {
constexpr uint32_t kStateMagic = 0x54535A41;   // "AZST" read as little-endian bytes
constexpr uint32_t kStateVersion = 2;   // 2 added the reference set and bot list
} // namespace

std::vector<int> initialReferences(int numBots) {
```

with:

```cpp

namespace {
constexpr uint32_t kStateMagic = 0x54535A41;   // "AZST" read as little-endian bytes
// 2 added the reference set and bot list, 3 the buffer's ownership targets.
constexpr uint32_t kStateVersion = 3;
} // namespace

std::vector<int> initialReferences(int numBots) {
```

In `src/run_state.cpp`, replace:

```cpp
        BinaryReader r(is);
        if (r.u32() != kStateMagic) throw std::runtime_error("bad magic (not a training state file)");
        uint32_t version = r.u32();
        if (version != 1 && version != kStateVersion)
            throw std::runtime_error("unsupported version " + std::to_string(version));
        TrainingState state;
        state.generation = r.i32();
        state.bestGeneration = r.i32();
        state.samplesTrained = static_cast<long long>(r.u64());
        state.buffer = ReplayBuffer::load(r);
        uint32_t tensors = r.u32();
        state.velocity.resize(tensors);
        for (std::vector<float>& v : state.velocity) {
```

with:

```cpp
        BinaryReader r(is);
        if (r.u32() != kStateMagic) throw std::runtime_error("bad magic (not a training state file)");
        uint32_t version = r.u32();
        if (version < 1 || version > kStateVersion)
            throw std::runtime_error("unsupported version " + std::to_string(version));
        TrainingState state;
        state.generation = r.i32();
        state.bestGeneration = r.i32();
        state.samplesTrained = static_cast<long long>(r.u64());
        state.buffer = ReplayBuffer::load(r, version >= 3);
        uint32_t tensors = r.u32();
        state.velocity.resize(tensors);
        for (std::vector<float>& v : state.velocity) {
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R 'test_selfplay|test_replay_buffer|test_run_state|test_trainer'`
Expected: `100% tests passed out of 4`. Then the whole suite and the BLAS variant. Expected: `100% tests passed out of 26` in both builds. The trainer does not use the target yet; Task 5 does.

- [ ] **Step 7: Commit**

```bash
git add include/azgo/selfplay.hpp src/selfplay.cpp include/azgo/replay_buffer.hpp src/replay_buffer.cpp \
        include/azgo/run_state.hpp src/run_state.cpp tests/test_selfplay.cpp tests/test_replay_buffer.cpp \
        tests/test_run_state.cpp tests/test_trainer.cpp
git commit -F - <<'EOF'
Store each game's final ownership and sample it as a training target

Self-play records who owns every point of the final board (Black's
view). A sample turns it to the side to move's view, as z is turned,
and moves each point with the same symmetry as the input planes;
hand-checked games pin both. state.bin version 3 stores it. Versions 1
and 2 recompute it by replaying each game's moves, checking every stored
position, so a donor buffer from a run made before the head exists
still works, and a buffer that does not replay is an error naming the
file.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 5: Train the head, log its loss, and handle old runs and mismatched starts

**Files:**
- Modify: `include/azgo/trainer.hpp`, `src/trainer.cpp`
- Test: `tests/test_trainer.cpp`, `tests/integration_smoke.sh`

**Interfaces:**
- Consumes: `Network::trainStep` with targets (Task 3), `TrainingBatch::ownership` (Task 4), `Config::ownership`/`ownershipWeight` (Task 2).
- Produces:
  - `float GenerationStats::ownershipLoss`: the mean unweighted ownership loss over the generation's steps, or −1 without the head.
  - `log.csv` header `...,winrate_vs_prev,distinct_vs_prev,ownership_loss`, each row ending in `%.5f` or an empty field.
  - Progress line `train loss p 2.42 v 0.36 o 0.590 lr 0.002 112.5s`: the ` o …` part appears only with the head.
  - `Trainer::startNew` with `initFrom` refuses a head mismatch with the two messages in Notes, and accepts a matching head.
  - `Trainer::resume` adds the `ownership_loss` column to a `log.csv` written before it existed.

- [ ] **Step 1: Write the failing tests**

Two existing assertions change with the new column: the exact header, and "the row's last field is the diversity count", which now reads the second-to-last field. Four new tests cover the rest. A run with the head trains it. Determinism holds with the head. Mismatches are refused, and a matching head is accepted. A run made before this change resumes: version 2, old config, 15-column `log.csv`.

In `tests/test_trainer.cpp`, replace:

```cpp

    assert(readFile(dir.file("log.csv")) ==
           "generation,games,avg_length,pass_rate,black_win_rate,policy_loss,value_loss,"
           "lr,replay_ratio,selfplay_s,train_s,arena_s,elo,winrate_vs_prev,distinct_vs_prev\n");
    assert(readFile(dir.file("elo.csv")) ==
           "generation,elo,winrate_vs_prev,wr_low,wr_high\n0,0.0,,,\n");
}
```

with:

```cpp

    assert(readFile(dir.file("log.csv")) ==
           "generation,games,avg_length,pass_rate,black_win_rate,policy_loss,value_loss,"
           "lr,replay_ratio,selfplay_s,train_s,arena_s,elo,winrate_vs_prev,distinct_vs_prev,"
           "ownership_loss\n");
    assert(readFile(dir.file("elo.csv")) ==
           "generation,elo,winrate_vs_prev,wr_low,wr_high\n0,0.0,,,\n");
}
```

In `tests/test_trainer.cpp`, replace:

```cpp
    assert(std::isfinite(st.elo));
    assert(st.distinctVsPrev >= 1 && st.distinctVsPrev <= 2);   // arena_games = 2

    {
        // The log row's last field is the same diversity count.
        const std::string log = readFile(dir.file("log.csv"));
        const size_t end = log.find_last_not_of('\n');
        const std::string row = log.substr(log.find_last_of('\n', end) + 1, end);
        const std::string last = row.substr(row.find_last_of(',') + 1);
        assert(!last.empty() && std::stoi(last) == st.distinctVsPrev);
    }

    t->runUntil(2);
```

with:

```cpp
    assert(std::isfinite(st.elo));
    assert(st.distinctVsPrev >= 1 && st.distinctVsPrev <= 2);   // arena_games = 2

    assert(st.ownershipLoss == -1.0f);   // no ownership head
    {
        // The log row ends with the same diversity count and an empty
        // ownership_loss field.
        const std::vector<std::string> rows = splitLines(readFile(dir.file("log.csv")));
        const std::string& row = rows.back();
        assert(std::count(row.begin(), row.end(), ',') == 15 && row.back() == ',');
        const std::string fields = row.substr(0, row.size() - 1);
        const std::string distinct = fields.substr(fields.find_last_of(',') + 1);
        assert(!distinct.empty() && std::stoi(distinct) == st.distinctVsPrev);
    }

    t->runUntil(2);
```

In `tests/test_trainer.cpp`, replace:

```cpp
    assert(!fs::exists(statePath(target.str())));
}

int main() {
    test_start_new_writes_run_files();
    test_two_generations();
```

with:

```cpp
    assert(!fs::exists(statePath(target.str())));
}

// ---- the ownership head ------------------------------------------------------

static Config ownershipConfig() {
    Config c = tinyConfig();
    c.ownership = true;
    return c;
}

// A run with the head trains it: the loss is logged, the progress line shows
// it, and the head's weights move.
static void test_ownership_run_trains_the_head() {
    TempDir dir("ownership");
    std::unique_ptr<Trainer> t = startQuiet(ownershipConfig(), dir);
    assert(readCheckpointConfig(checkpointPath(dir.str(), 0)).ownership);
    std::ostringstream out;
    t->setLog(&out);
    const GenerationStats st = t->runGeneration();
    // A mean squared error of values in [-1, 1].
    assert(std::isfinite(st.ownershipLoss) && st.ownershipLoss > 0 && st.ownershipLoss < 4);
    const std::string row = splitLines(readFile(dir.file("log.csv"))).back();
    const std::string last = row.substr(row.find_last_of(',') + 1);
    char expected[32];
    std::snprintf(expected, sizeof expected, "%.5f", st.ownershipLoss);
    assert(last == expected);
    assert(out.str().find(" v ") != std::string::npos && out.str().find(" o ") != std::string::npos);

    Network gen0 = loadNetwork(checkpointPath(dir.str(), 0));
    Network gen1 = loadNetwork(checkpointPath(dir.str(), 1));
    std::vector<ParamView> p0 = gen0.params(), p1 = gen1.params();
    assert(p1.back().name == "ownership.bn.beta" && p1[p1.size() - 3].name == "ownership.conv.w");
    const ParamView& w0 = p0[p0.size() - 3];
    const ParamView& w1 = p1[p1.size() - 3];
    assert(std::memcmp(w0.value, w1.value, w0.size * sizeof(float)) != 0);
}

// Determinism holds with the head: resume continues byte-identically and the
// worker count changes nothing.
static void test_ownership_determinism() {
    TempDir a("own_resume_a"), b("own_resume_b");
    const Config c = ownershipConfig();
    startQuiet(c, a)->runUntil(3);
    startQuiet(c, b)->runUntil(1);
    resumeQuiet(b, {})->runUntil(3);
    for (const char* f : {"checkpoints/gen_0003.azn", "state.bin", "elo.csv", "results.csv",
                          "references.csv", "sgf/gen_0003_game_00.sgf"})
        assert(readFile(a.file(f)) == readFile(b.file(f)));

    TempDir one("own_workers1"), three("own_workers3");
    Config w = c;
    w.workers = 1;
    startQuiet(w, one)->runUntil(2);
    w.workers = 3;
    startQuiet(w, three)->runUntil(2);
    for (const char* f : {"checkpoints/gen_0002.azn", "state.bin", "results.csv", "references.csv"})
        assert(readFile(one.file(f)) == readFile(three.file(f)));
}

// --init-from with a checkpoint whose head does not match the config says how
// to fix it; resume refuses to switch the head on or off.
static void test_ownership_mismatches_are_refused() {
    TempDir plain("own_plain"), owner("own_owner"), target("own_target");
    startQuiet(tinyConfig(), plain);
    startQuiet(ownershipConfig(), owner);

    Trainer::RunInit init;
    init.initFrom = checkpointPath(plain.str(), 0);
    std::string msg;
    try {
        Trainer::startNew(ownershipConfig(), target.str(), init);
    } catch (const ConfigError& e) {
        msg = e.what();
    }
    assert(msg.find("has no ownership head") != std::string::npos);
    assert(msg.find("grow --from " + init.initFrom + " --ownership") != std::string::npos);
    assert(!fs::exists(statePath(target.str())));

    init.initFrom = checkpointPath(owner.str(), 0);
    msg.clear();
    try {
        Trainer::startNew(tinyConfig(), target.str(), init);
    } catch (const ConfigError& e) {
        msg = e.what();
    }
    assert(msg.find("has an ownership head") != std::string::npos);
    assert(msg.find("--set ownership=true") != std::string::npos);
    assert(!fs::exists(statePath(target.str())));

    // Another shape difference as well: the shape message, naming the head.
    Config wider = ownershipConfig();
    wider.filters = 16;
    msg.clear();
    try {
        Trainer::startNew(wider, target.str(), init);
    } catch (const ConfigError& e) {
        msg = e.what();
    }
    assert(msg.find("8 filters, value_hidden 8, ownership head but the config asks for") !=
           std::string::npos);

    // A matching head is accepted: generation 0 is that checkpoint.
    {
        std::unique_ptr<Trainer> t = Trainer::startNew(ownershipConfig(), target.str(), init);
        t->setLog(nullptr);
    }
    assert(readFile(checkpointPath(target.str(), 0)) == readFile(init.initFrom));

    for (const char* change : {"ownership=true", "ownership=false"}) {
        const TempDir& dir = std::string(change) == "ownership=true" ? plain : owner;
        msg.clear();
        try {
            Trainer::resume(dir.str(), {change});
        } catch (const ConfigError& e) {
            msg = e.what();
        }
        assert(msg == "cannot change architecture keys on resume: ownership");
    }
    // ownership_weight is tunable.
    assert(resumeQuiet(owner, {"ownership_weight=0.5"})->config().ownershipWeight == 0.5f);
}

// A run made before this change: state.bin version 2 (no ownership in the
// buffer), a config.cfg without the ownership keys and a log.csv without the
// ownership_loss column. Resuming it recomputes the buffer's ownership, adds
// the column to every old row, and continues as a run without the head.
static void test_resume_run_from_before_ownership() {
    TempDir dir("pre_ownership");
    startQuiet(tinyConfig(), dir)->runUntil(1);

    // state.bin version 2: the buffer without ownership.
    const std::string bytes = readFile(statePath(dir.str()));
    const TrainingState current = loadTrainingState(statePath(dir.str()));
    std::ostringstream with, without;
    BinaryWriter ww(with), wo(without);
    current.buffer.save(ww, true);
    current.buffer.save(wo, false);
    const size_t header = 24;
    std::string v2 = bytes.substr(0, header) + without.str() + bytes.substr(header + with.str().size());
    v2[4] = 2;
    writeFile(statePath(dir.str()), v2);
    // config.cfg and log.csv as the old code wrote them.
    std::string oldConfig;
    for (const std::string& line : splitLines(readFile(configPath(dir.str()))))
        if (line.rfind("ownership", 0) != 0) oldConfig += line + "\n";
    writeFile(configPath(dir.str()), oldConfig);
    std::vector<std::string> rows = splitLines(readFile(dir.file("log.csv")));
    assert(rows.size() == 2 && rows[1].back() == ',');
    std::string oldLog = rows[0].substr(0, rows[0].size() - std::string(",ownership_loss").size()) +
                         "\n" + rows[1].substr(0, rows[1].size() - 1) + "\n";
    writeFile(dir.file("log.csv"), oldLog);

    std::unique_ptr<Trainer> t = resumeQuiet(dir, {});
    assert(!t->config().ownership);
    const TrainingState resumed = loadTrainingState(statePath(dir.str()));
    for (size_t i = 0; i < current.buffer.games(); ++i)
        assert(resumed.buffer.game(i).finalOwnership == current.buffer.game(i).finalOwnership);
    t->runUntil(2);
    rows = splitLines(readFile(dir.file("log.csv")));
    assert(rows.size() == 3 && rows[0].find(",distinct_vs_prev,ownership_loss") != std::string::npos);
    for (const std::string& row : rows) assert(std::count(row.begin(), row.end(), ',') == 15);
    assert(readFile(statePath(dir.str()))[4] == 3);
}

int main() {
    test_start_new_writes_run_files();
    test_two_generations();
```

In `tests/test_trainer.cpp`, replace:

```cpp
    test_rerate_refusals_and_gaps();
    test_start_from_network_and_buffer();
    test_start_from_refusals();
    std::printf("test_trainer: all tests passed\n");
    return 0;
}
```

with:

```cpp
    test_rerate_refusals_and_gaps();
    test_start_from_network_and_buffer();
    test_start_from_refusals();
    test_ownership_run_trains_the_head();
    test_ownership_determinism();
    test_ownership_mismatches_are_refused();
    test_resume_run_from_before_ownership();
    std::printf("test_trainer: all tests passed\n");
    return 0;
}
```

The smoke test runs one generation with the head and checks both `log.csv` forms, then the refused `--init-from`:

In `tests/integration_smoke.sh`, replace:

```bash
echo "$out" | grep -q "architecture" || fail "resume refusal does not mention architecture: $out"
test ! -e "$CK/gen_0004.azn" || fail "a refused resume still trained"

echo "integration_smoke: all checks passed"
```

with:

```bash
echo "$out" | grep -q "architecture" || fail "resume refusal does not mention architecture: $out"
test ! -e "$CK/gen_0004.azn" || fail "a refused resume still trained"

echo "== train: a run with the ownership head =="
"$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/own_run" "${TINY[@]}" --set ownership=true \
    --generations 1
head -1 "$W/own_run/log.csv" | grep -q ',distinct_vs_prev,ownership_loss$' || fail "log.csv has no ownership_loss column"
tail -1 "$W/own_run/log.csv" | grep -Eq ',[0-9]+\.[0-9]{5}$' || fail "log.csv row has no ownership loss"
tail -1 "$RUN/log.csv" | grep -q ',$' || fail "a run without the head should leave ownership_loss empty"
status=0
out="$("$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/own_bad" "${TINY[@]}" --set ownership=true \
    --init-from "$CK/gen_0003.azn" --generations 1 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "--init-from a checkpoint without the head exited $status, expected 1"
echo "$out" | grep -q -- "--ownership --out" || fail "the refusal does not suggest grow --ownership: $out"
test ! -e "$W/own_bad/state.bin" || fail "a refused start left a run behind"

echo "integration_smoke: all checks passed"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cmake --build build -j`
Expected: the build fails in `tests/test_trainer.cpp` with:

```
tests/test_trainer.cpp:175:15: error: no member named 'ownershipLoss' in 'azgo::GenerationStats'
```

- [ ] **Step 3: Declare the new statistic and behavior**

In `include/azgo/trainer.hpp`, replace:

```cpp
    double passRate = 0;        // passes / plies
    double blackWinRate = 0;
    float policyLoss = 0, valueLoss = 0;   // mean over the generation's steps
    float lr = 0;
    double replayRatio = 0;     // samples drawn this generation / new positions
    double selfPlaySeconds = 0, trainSeconds = 0, arenaSeconds = 0;
```

with:

```cpp
    double passRate = 0;        // passes / plies
    double blackWinRate = 0;
    float policyLoss = 0, valueLoss = 0;   // mean over the generation's steps
    float ownershipLoss = -1;   // the same mean, unweighted; -1 when the network has no ownership head
    float lr = 0;
    double replayRatio = 0;     // samples drawn this generation / new positions
    double selfPlaySeconds = 0, trainSeconds = 0, arenaSeconds = 0;
```

In `include/azgo/trainer.hpp`, replace:

```cpp
public:
    // Optional starting material for a new run. initFrom: a checkpoint that
    // becomes generation 0 instead of a fresh random network (its shape must
    // equal the config's); this is how a grown or pre-trained network starts a
    // run. bufferFrom: another run's directory whose replay buffer seeds the
    // new one, so the first generations train on real games rather than an
    // empty buffer (board size and history must match). Empty = not used.
    struct RunInit {
```

with:

```cpp
public:
    // Optional starting material for a new run. initFrom: a checkpoint that
    // becomes generation 0 instead of a fresh random network (its shape must
    // equal the config's, ownership head included); this is how a grown or
    // pre-trained network starts a run. bufferFrom: another run's directory whose replay buffer seeds the
    // new one, so the first generations train on real games rather than an
    // empty buffer (board size and history must match). Empty = not used.
    struct RunInit {
```

In `include/azgo/trainer.hpp`, replace:

```cpp
                                             const RunInit& init = {});
    // Loads config.cfg, applies overrides, refuses (ConfigError listing keys)
    // if an architecture key changed, loads state.bin and the latest checkpoint.
    // Rewrites config.cfg with the overridden values.
    static std::unique_ptr<Trainer> resume(const std::string& runDir,
                                           const std::vector<std::string>& overrides);

```

with:

```cpp
                                             const RunInit& init = {});
    // Loads config.cfg, applies overrides, refuses (ConfigError listing keys)
    // if an architecture key changed, loads state.bin and the latest checkpoint.
    // Rewrites config.cfg with the overridden values, and gives a log.csv
    // written before the ownership_loss column existed that column (empty).
    static std::unique_ptr<Trainer> resume(const std::string& runDir,
                                           const std::vector<std::string>& overrides);

```

- [ ] **Step 4: Implement them**

In order: the includes, the two log headers, the row's new field, the shape string and the `log.csv` upgrade, the `--init-from` check, the upgrade call in `resume`, the training loop and the progress line. The training loop passes the batch's targets only to a network with the head. A network without it trains exactly as before: `sample` draws no extra random numbers, and `trainStep` ignores the null target.

In `src/trainer.cpp`, replace:

```cpp
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>

```

with:

```cpp
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

```

In `src/trainer.cpp`, replace:

```cpp
namespace {

const char* const kLogHeader =
    "generation,games,avg_length,pass_rate,black_win_rate,policy_loss,value_loss,"
    "lr,replay_ratio,selfplay_s,train_s,arena_s,elo,winrate_vs_prev,distinct_vs_prev\n";
const char* const kEloHeader = "generation,elo,winrate_vs_prev,wr_low,wr_high\n";
```

with:

```cpp
namespace {

const char* const kLogHeader =
    "generation,games,avg_length,pass_rate,black_win_rate,policy_loss,value_loss,"
    "lr,replay_ratio,selfplay_s,train_s,arena_s,elo,winrate_vs_prev,distinct_vs_prev,"
    "ownership_loss\n";
// The header before ownership_loss was added; Trainer::resume upgrades it.
const char* const kLogHeaderBeforeOwnership =
    "generation,games,avg_length,pass_rate,black_win_rate,policy_loss,value_loss,"
    "lr,replay_ratio,selfplay_s,train_s,arena_s,elo,winrate_vs_prev,distinct_vs_prev\n";
const char* const kEloHeader = "generation,elo,winrate_vs_prev,wr_low,wr_high\n";
```

In `src/trainer.cpp`, replace:

```cpp
    if (s.winRateVsPrev >= 0) std::snprintf(winRate, sizeof winRate, "%.4f", s.winRateVsPrev);
    char distinct[32] = "";
    if (s.distinctVsPrev >= 0) std::snprintf(distinct, sizeof distinct, "%d", s.distinctVsPrev);
    char line[512];
    std::snprintf(line, sizeof line,
                  "%d,%d,%.2f,%.4f,%.4f,%.5f,%.5f,%g,%.3f,%.3f,%.3f,%.3f,%.1f,%s,%s\n",
                  s.generation, s.games, s.avgLength, s.passRate, s.blackWinRate,
                  s.policyLoss, s.valueLoss, s.lr, s.replayRatio, s.selfPlaySeconds,
                  s.trainSeconds, s.arenaSeconds, s.elo, winRate, distinct);
    std::ofstream out(logPath(runDir), std::ios::app | std::ios::binary);
    out << line;
    if (!out) throw std::runtime_error("cannot append to " + logPath(runDir));
}

// "9x9, history 8, 6 blocks, 64 filters, value_hidden 64": the keys a
// checkpoint's shape depends on, for messages about a mismatch.
std::string shapeString(const NetConfig& c) {
    return std::to_string(c.boardSize) + "x" + std::to_string(c.boardSize) + ", history " +
           std::to_string(c.history) + ", " + std::to_string(c.blocks) + " blocks, " +
           std::to_string(c.filters) + " filters, value_hidden " + std::to_string(c.valueHidden);
}

std::string joinInts(const std::vector<int>& v) {
```

with:

```cpp
    if (s.winRateVsPrev >= 0) std::snprintf(winRate, sizeof winRate, "%.4f", s.winRateVsPrev);
    char distinct[32] = "";
    if (s.distinctVsPrev >= 0) std::snprintf(distinct, sizeof distinct, "%d", s.distinctVsPrev);
    char ownership[32] = "";
    if (s.ownershipLoss >= 0) std::snprintf(ownership, sizeof ownership, "%.5f", s.ownershipLoss);
    char line[512];
    std::snprintf(line, sizeof line,
                  "%d,%d,%.2f,%.4f,%.4f,%.5f,%.5f,%g,%.3f,%.3f,%.3f,%.3f,%.1f,%s,%s,%s\n",
                  s.generation, s.games, s.avgLength, s.passRate, s.blackWinRate,
                  s.policyLoss, s.valueLoss, s.lr, s.replayRatio, s.selfPlaySeconds,
                  s.trainSeconds, s.arenaSeconds, s.elo, winRate, distinct, ownership);
    std::ofstream out(logPath(runDir), std::ios::app | std::ios::binary);
    out << line;
    if (!out) throw std::runtime_error("cannot append to " + logPath(runDir));
}

// "9x9, history 8, 6 blocks, 64 filters, value_hidden 64[, ownership head]":
// the keys a checkpoint's shape depends on, for messages about a mismatch.
std::string shapeString(const NetConfig& c) {
    return std::to_string(c.boardSize) + "x" + std::to_string(c.boardSize) + ", history " +
           std::to_string(c.history) + ", " + std::to_string(c.blocks) + " blocks, " +
           std::to_string(c.filters) + " filters, value_hidden " + std::to_string(c.valueHidden) +
           (c.ownership ? ", ownership head" : "");
}

// A run started before log.csv had the ownership_loss column gets it, empty
// in every old row (those generations had no ownership head), so every row
// has as many fields as the header. Anything else is left alone.
void addOwnershipColumn(const std::string& runDir) {
    std::ifstream in(logPath(runDir), std::ios::binary);
    if (!in) return;
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (text.compare(0, std::strlen(kLogHeaderBeforeOwnership), kLogHeaderBeforeOwnership) != 0) return;
    std::string out = kLogHeader;
    std::istringstream rows(text.substr(std::strlen(kLogHeaderBeforeOwnership)));
    for (std::string row; std::getline(rows, row);)
        if (!row.empty()) out += row + ",\n";
    writeFileAtomically(logPath(runDir), out);
}

std::string joinInts(const std::vector<int>& v) {
```

In `src/trainer.cpp`, replace:

```cpp
    std::optional<Network> given;
    if (!init.initFrom.empty()) {
        given.emplace(loadNetwork(init.initFrom));
        if (given->config() != cfg.netConfig())
            throw ConfigError(init.initFrom + " has shape " + shapeString(given->config()) +
                              " but the config asks for " + shapeString(cfg.netConfig()));
    }
    std::optional<ReplayBuffer> donorBuffer;
    if (!init.bufferFrom.empty()) {
```

with:

```cpp
    std::optional<Network> given;
    if (!init.initFrom.empty()) {
        given.emplace(loadNetwork(init.initFrom));
        if (given->config() != cfg.netConfig()) {
            NetConfig sameHead = given->config();
            sameHead.ownership = cfg.ownership;
            if (sameHead == cfg.netConfig() && cfg.ownership)   // only the head is missing
                throw ConfigError(init.initFrom + " has no ownership head but the config sets "
                                  "ownership = true; add one with: grow --from " + init.initFrom +
                                  " --ownership --out NEW.azn");
            if (sameHead == cfg.netConfig())                    // only the head is extra
                throw ConfigError(init.initFrom + " has an ownership head but the config sets "
                                  "ownership = false; train it with --set ownership=true");
            throw ConfigError(init.initFrom + " has shape " + shapeString(given->config()) +
                              " but the config asks for " + shapeString(cfg.netConfig()));
        }
    }
    std::optional<ReplayBuffer> donorBuffer;
    if (!init.bufferFrom.empty()) {
```

In `src/trainer.cpp`, replace:

```cpp
    // Rewritten only after every check passed, so a refused resume leaves the
    // run directory untouched.
    writeFileAtomically(configPath(runDir), configToString(cfg));

    return std::unique_ptr<Trainer>(new Trainer(cfg, runDir, std::move(net), std::move(state)));
}
```

with:

```cpp
    // Rewritten only after every check passed, so a refused resume leaves the
    // run directory untouched.
    writeFileAtomically(configPath(runDir), configToString(cfg));
    addOwnershipColumn(runDir);

    return std::unique_ptr<Trainer>(new Trainer(cfg, runDir, std::move(net), std::move(state)));
}
```

In `src/trainer.cpp`, replace:

```cpp
        opt.velocity() = state_.velocity;
        Rng rng(deriveSeed(cfg_.seed, static_cast<uint64_t>(g), kPhaseTrain));
        TrainingBatch batch;
        double policySum = 0, valueSum = 0;
        for (int s = 0; s < cfg_.trainSteps; ++s) {
            state_.buffer.sample(cfg_.batchSize, cfg_.history, rng, batch);
            const LossStats loss = net_.trainStep(ws, batch.input.data(), batch.pi.data(),
                                                  batch.z.data(), batch.size);
            // Nothing of generation g is committed yet (state.bin is the
            // commit point), so the last good state (generation g-1) stays.
            if (!std::isfinite(loss.policyLoss) || !std::isfinite(loss.valueLoss))
                throw std::runtime_error("non-finite loss at generation " + std::to_string(g) +
                                         " step " + std::to_string(s) +
                                         "; last good state kept");
            opt.step(net_, lr);
            policySum += loss.policyLoss;
            valueSum += loss.valueLoss;
        }
        if (cfg_.trainSteps > 0) {
            stats.policyLoss = static_cast<float>(policySum / cfg_.trainSteps);
            stats.valueLoss = static_cast<float>(valueSum / cfg_.trainSteps);
        }
        // The per-step loss check cannot see the weights written by the last
        // optimizer step, so scan the parameters and momentum before anything
```

with:

```cpp
        opt.velocity() = state_.velocity;
        Rng rng(deriveSeed(cfg_.seed, static_cast<uint64_t>(g), kPhaseTrain));
        TrainingBatch batch;
        double policySum = 0, valueSum = 0, ownershipSum = 0;
        for (int s = 0; s < cfg_.trainSteps; ++s) {
            state_.buffer.sample(cfg_.batchSize, cfg_.history, rng, batch);
            // The batch always carries ownership targets; only a network
            // with the head learns from them, with weight ownership_weight.
            const LossStats loss = net_.trainStep(
                ws, batch.input.data(), batch.pi.data(), batch.z.data(), batch.size,
                cfg_.ownership ? batch.ownership.data() : nullptr, cfg_.ownershipWeight);
            // Nothing of generation g is committed yet (state.bin is the
            // commit point), so the last good state (generation g-1) stays.
            if (!std::isfinite(loss.policyLoss) || !std::isfinite(loss.valueLoss) ||
                !std::isfinite(loss.ownershipLoss))
                throw std::runtime_error("non-finite loss at generation " + std::to_string(g) +
                                         " step " + std::to_string(s) +
                                         "; last good state kept");
            opt.step(net_, lr);
            policySum += loss.policyLoss;
            valueSum += loss.valueLoss;
            ownershipSum += loss.ownershipLoss;
        }
        if (cfg_.ownership) stats.ownershipLoss = 0;
        if (cfg_.trainSteps > 0) {
            stats.policyLoss = static_cast<float>(policySum / cfg_.trainSteps);
            stats.valueLoss = static_cast<float>(valueSum / cfg_.trainSteps);
            if (cfg_.ownership) stats.ownershipLoss = static_cast<float>(ownershipSum / cfg_.trainSteps);
        }
        // The per-step loss check cannot see the weights written by the last
        // optimizer step, so scan the parameters and momentum before anything
```

In `src/trainer.cpp`, replace:

```cpp
    appendLogRow(runDir_, stats);

    if (log_) {
        char line[512];
        std::snprintf(line, sizeof line,
                      "gen %d | selfplay %d games avg %.1f plies %.1fs | "
                      "train loss p %.2f v %.2f lr %g %.1fs | ",
                      g, stats.games, stats.avgLength, stats.selfPlaySeconds, stats.policyLoss,
                      stats.valueLoss, stats.lr, stats.trainSeconds);
        *log_ << line;
        if (arenaThisGeneration) {
            // e.g. "arena elo 512.3 vs 149:0.55 120:0.70 bot100:1.00 (distinct 19/20) promoted 3.2s"
```

with:

```cpp
    appendLogRow(runDir_, stats);

    if (log_) {
        // e.g. "train loss p 1.73 v 0.09 lr 0.002 2.5s", with "o 0.412" after
        // the value loss when the network has an ownership head.
        char ownership[32] = "";
        if (stats.ownershipLoss >= 0)
            std::snprintf(ownership, sizeof ownership, " o %.3f", stats.ownershipLoss);
        char line[512];
        std::snprintf(line, sizeof line,
                      "gen %d | selfplay %d games avg %.1f plies %.1fs | "
                      "train loss p %.2f v %.2f%s lr %g %.1fs | ",
                      g, stats.games, stats.avgLength, stats.selfPlaySeconds, stats.policyLoss,
                      stats.valueLoss, ownership, stats.lr, stats.trainSeconds);
        *log_ << line;
        if (arenaThisGeneration) {
            // e.g. "arena elo 512.3 vs 149:0.55 120:0.70 bot100:1.00 (distinct 19/20) promoted 3.2s"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R 'test_trainer|integration_smoke'`
Expected: `100% tests passed out of 2`. Then the whole suite and the BLAS variant. Expected: `100% tests passed out of 26` in both builds. `test_resume_is_byte_identical` and `test_thread_count_independence` still pass unchanged, which confirms that runs without the head train exactly as before.

- [ ] **Step 6: Commit**

```bash
git add include/azgo/trainer.hpp src/trainer.cpp tests/test_trainer.cpp tests/integration_smoke.sh
git commit -F - <<'EOF'
Train the ownership head and log its loss

With ownership on, every training step passes the batch's ownership
targets and ownership_weight. The mean unweighted loss goes to the
progress line and a new trailing log.csv column, ownership_loss, which
stays empty without the head; resuming an older run adds the column to
its log. --init-from refuses a checkpoint whose head does not match the
config and names the fix (grow --ownership, or --set ownership=true).
Resume and worker-count determinism are tested with the head on.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 6: `grow --ownership`, and growth that carries a head

**Files:**
- Modify: `include/azgo/grow.hpp`, `src/grow.cpp`, `apps/grow.cpp`
- Test: `tests/test_grow.cpp`, `tests/integration_smoke.sh`

**Interfaces:**
- Consumes: the head (Task 3), checkpoint version 2 (Task 2), `--init-from` with a head (Task 5).
- Produces:
  - `Network addOwnershipHead(const Network& net, uint64_t seed)`: every tensor of `net` copied by name, and the head as `Network(cfg, seed)` initializes it. Throws `std::invalid_argument` if `net` already has a head.
  - `growBlocks` carries a head unchanged (it already copies by name; the test makes that a guarantee).
  - `widenFilters` splits `ownership.conv.w`'s input weights with the stream mapping `g`, after every other layer's, and copies the head's BN.
  - `grow --ownership` (a flag; it combines with `--blocks`/`--filters` and refuses a checkpoint that has a head). The equivalence check also compares ownership maps when both networks have a head, and the shape lines end in `, ownership head`.

- [ ] **Step 1: Write the failing tests**

`trainABit` now trains a network's head too, on random targets, so that a carried head is a trained one. It draws extra random numbers only for networks with the head, so every existing test is unchanged.

In `tests/test_grow.cpp`, replace:

```cpp
}

// Gives every BN of `net` non-trivial running statistics and affine
// parameters, as a trained network would have, by training a few steps.
void trainABit(Network& net, Rng& rng) {
    const NetConfig& cfg = net.config();
    std::unique_ptr<Backend> be = makeBackend("naive", 1);
    Workspace ws(cfg, 16, *be);
    for (int i = 0; i < 3; ++i) {
        Batch b = randomBatch(cfg, 16, rng);
        net.trainStep(ws, b.input.data(), b.pi.data(), b.z.data(), 16);
        for (ParamView& p : net.params())
            for (size_t k = 0; k < p.size; ++k) p.value[k] -= 0.05f * p.grad[k];
    }
```

with:

```cpp
}

// Gives every BN of `net` non-trivial running statistics and affine
// parameters, as a trained network would have, by training a few steps. A
// network with an ownership head trains it too, on random targets.
void trainABit(Network& net, Rng& rng) {
    const NetConfig& cfg = net.config();
    std::unique_ptr<Backend> be = makeBackend("naive", 1);
    Workspace ws(cfg, 16, *be);
    for (int i = 0; i < 3; ++i) {
        Batch b = randomBatch(cfg, 16, rng);
        std::vector<float> own;
        if (cfg.ownership) {
            own.resize(static_cast<size_t>(16) * cfg.area());
            for (float& v : own) v = static_cast<float>(rng.below(3) - 1);
        }
        net.trainStep(ws, b.input.data(), b.pi.data(), b.z.data(), 16,
                      cfg.ownership ? own.data() : nullptr, 1.5f);
        for (ParamView& p : net.params())
            for (size_t k = 0; k < p.size; ++k) p.value[k] -= 0.05f * p.grad[k];
    }
```

In `tests/test_grow.cpp`, replace:

```cpp
    assert(seedMatters);
}

} // namespace

int main() {
```

with:

```cpp
    assert(seedMatters);
}

// ---------------------------------------------------------------- ownership

NetConfig withOwnership(NetConfig cfg) {
    cfg.ownership = true;
    return cfg;
}

// Inference outputs of all three heads, for exact or near comparisons.
struct Outputs {
    std::vector<float> logits, values, ownership;
};

Outputs outputsOf(const Network& net, const Batch& batch, int B) {
    const NetConfig& cfg = net.config();
    std::unique_ptr<Backend> be = makeBackend("naive", 1);
    Workspace ws(cfg, B, *be);
    Outputs out;
    out.logits.resize(static_cast<size_t>(B) * cfg.actions());
    out.values.resize(B);
    if (cfg.ownership) out.ownership.resize(static_cast<size_t>(B) * cfg.area());
    net.infer(ws, batch.input.data(), B, out.logits.data(), out.values.data(),
              cfg.ownership ? out.ownership.data() : nullptr);
    return out;
}

double maxDifference(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    double worst = 0;
    for (size_t i = 0; i < a.size(); ++i) worst = std::fmax(worst, std::fabs(a[i] - b[i]));
    return worst;
}

// A new head cannot change the policy or the value: bit for bit the same
// outputs, every old tensor copied, and a second head refused.
void test_add_ownership_head_keeps_policy_and_value() {
    Rng rng(16);
    Network net(makeConfig(5, 2), 7);
    trainABit(net, rng);
    Network owner = addOwnershipHead(net, 3);
    assert(owner.config() == withOwnership(net.config()));
    std::vector<ParamView> a = net.params(), b = owner.params();
    assert(b.size() == a.size() + 3 && b.back().name == "ownership.bn.beta");
    for (size_t i = 0; i < a.size(); ++i)
        assert(a[i].name == b[i].name && std::memcmp(a[i].value, b[i].value, a[i].size * sizeof(float)) == 0);
    std::vector<BufferView> ba = net.buffers(), bb = owner.buffers();
    for (size_t i = 0; i < ba.size(); ++i)
        assert(std::memcmp(ba[i].value, bb[i].value, ba[i].size * sizeof(float)) == 0);

    Batch batch = randomBatch(net.config(), 8, rng);
    const Outputs before = outputsOf(net, batch, 8), after = outputsOf(owner, batch, 8);
    assert(after.logits == before.logits && after.values == before.values);
    for (float o : after.ownership) assert(std::isfinite(o) && o >= -1.0f && o <= 1.0f);

    bool threw = false;
    try {
        addOwnershipHead(owner, 3);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

// Deepening copies an existing head by name: all three heads bit for bit.
void test_deepening_carries_the_ownership_head() {
    Rng rng(17);
    Network net(withOwnership(makeConfig(5, 2)), 7);
    trainABit(net, rng);
    Network deep = growBlocks(net, 4, 99);
    assert(deep.config().ownership && deep.config().blocks == 4);
    Batch batch = randomBatch(net.config(), 8, rng);
    const Outputs a = outputsOf(net, batch, 8), b = outputsOf(deep, batch, 8);
    assert(a.logits == b.logits && a.values == b.values && a.ownership == b.ownership);
}

// Widening splits the head's input weights like the other heads': all three
// heads agree within float rounding, in inference and in the training loss.
// A network without the head widens exactly as it did before the head existed.
void test_widening_carries_the_ownership_head() {
    Rng rng(18);
    Network net(withOwnership(makeConfig(5, 3)), 7);
    trainABit(net, rng);
    Network wide = widenFilters(net, 20, 31);
    assert(wide.config().ownership && wide.config().filters == 20);
    Batch batch = randomBatch(net.config(), 8, rng);
    const Outputs a = outputsOf(net, batch, 8), b = outputsOf(wide, batch, 8);
    assert(maxDifference(a.logits, b.logits) <= 1e-5 && maxDifference(a.values, b.values) <= 1e-5);
    assert(maxDifference(a.ownership, b.ownership) <= 1e-5);

    std::vector<float> own(static_cast<size_t>(8) * net.config().area());
    for (float& v : own) v = static_cast<float>(rng.below(3) - 1);
    std::unique_ptr<Backend> be = makeBackend("naive", 1);
    Workspace wa(net.config(), 8, *be), wb(wide.config(), 8, *be);
    const LossStats la = net.computeLoss(wa, batch.input.data(), batch.pi.data(), batch.z.data(), 8,
                                         own.data(), 1.5f);
    const LossStats lb = wide.computeLoss(wb, batch.input.data(), batch.pi.data(), batch.z.data(), 8,
                                          own.data(), 1.5f);
    assert(std::fabs(la.ownershipLoss - lb.ownershipLoss) <= 1e-5);
    assert(std::fabs(la.valueLoss - lb.valueLoss) <= 1e-5);

    // The same trunk without the head widens to the same tensors (the head's
    // random shares are drawn after every other layer's).
    Network plain(makeConfig(5, 3), 7);
    trainABit(plain, rng);
    Network plainWide = widenFilters(plain, 20, 31);
    Network ownerWide = widenFilters(addOwnershipHead(plain, 5), 20, 31);
    std::vector<ParamView> pp = plainWide.params(), po = ownerWide.params();
    for (size_t i = 0; i < pp.size(); ++i)
        assert(pp[i].name == po[i].name && std::memcmp(pp[i].value, po[i].value, pp[i].size * sizeof(float)) == 0);
}

} // namespace

int main() {
```

In `tests/test_grow.cpp`, replace:

```cpp
    test_widened_copies_receive_different_gradients();
    test_widen_and_deepen_compose();
    test_widening_is_deterministic();
    std::printf("test_grow: all tests passed\n");
    return 0;
}
```

with:

```cpp
    test_widened_copies_receive_different_gradients();
    test_widen_and_deepen_compose();
    test_widening_is_deterministic();
    test_add_ownership_head_keeps_policy_and_value();
    test_deepening_carries_the_ownership_head();
    test_widening_carries_the_ownership_head();
    std::printf("test_grow: all tests passed\n");
    return 0;
}
```

In `tests/integration_smoke.sh`, replace:

```bash
echo "$out" | grep -q -- "--ownership --out" || fail "the refusal does not suggest grow --ownership: $out"
test ! -e "$W/own_bad/state.bin" || fail "a refused start left a run behind"

echo "integration_smoke: all checks passed"
```

with:

```bash
echo "$out" | grep -q -- "--ownership --out" || fail "the refusal does not suggest grow --ownership: $out"
test ! -e "$W/own_bad/state.bin" || fail "a refused start left a run behind"

echo "== grow: add an ownership head, then train from it =="
out="$("$BUILD/grow" --from "$CK/gen_0003.azn" --ownership --out "$W/owner.azn")"
echo "$out"
echo "$out" | grep -q "new: 5x5, history 2, 1 blocks, 8 filters, value_hidden 8, ownership head" \
    || fail "grow --ownership did not report the head: $out"
out="$("$BUILD/grow" --from "$W/owner.azn" --filters 12 --blocks 2 --out "$W/owner_big.azn")"
echo "$out" | grep -q "new: 5x5, history 2, 2 blocks, 12 filters, value_hidden 8, ownership head" \
    || fail "widening and deepening did not keep the head: $out"
status=0
out="$("$BUILD/grow" --from "$W/owner.azn" --ownership --out "$W/owner2.azn" 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "a second --ownership exited $status, expected 1"
echo "$out" | grep -q "already has an ownership head" || fail "unexpected refusal: $out"
test ! -e "$W/owner2.azn" || fail "a refused grow still wrote a checkpoint"
"$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/owner_run" "${TINY[@]}" --set ownership=true \
    --init-from "$W/owner.azn" --buffer-from "$RUN" --generations 1
cmp "$W/owner.azn" "$W/owner_run/checkpoints/gen_0000.azn" || fail "gen_0000 is not the checkpoint with the head"
test -s "$W/owner_run/checkpoints/gen_0001.azn" || fail "the run from the grown head did not train"

echo "integration_smoke: all checks passed"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build build -j`
Expected: the build fails in `tests/test_grow.cpp` with:

```
tests/test_grow.cpp:369:21: error: use of undeclared identifier 'addOwnershipHead'
```

- [ ] **Step 3: Declare `addOwnershipHead` and document the head in the growth comment**

In `include/azgo/grow.hpp`, replace:

```cpp
//
// Width is not free: a 3x3 conv costs 9 * F^2 * n^2 multiply-adds per
// position, so 96 filters cost about (96/64)^2 = 2.25x as much as 64.
#pragma once
#include <cstdint>
#include "azgo/network.hpp"
```

with:

```cpp
//
// Width is not free: a 3x3 conv costs 9 * F^2 * n^2 multiply-adds per
// position, so 96 filters cost about (96/64)^2 = 2.25x as much as 64.
//
// Adding an ownership head: the head reads the trunk's output and nothing
// reads the head, so a network with a new, randomly initialized head computes
// exactly the old policy and value; only training changes them, through the
// trunk. An existing head is carried along by both growths: deepening copies
// it by name like every other tensor, and widening splits its 1x1 conv's
// input weights like the other heads' (it reads the stream too).
#pragma once
#include <cstdint>
#include "azgo/network.hpp"
```

In `include/azgo/grow.hpp`, replace:

```cpp
// kMaxFilters.
Network widenFilters(const Network& net, int newFilters, uint64_t seed);

} // namespace azgo
```

with:

```cpp
// kMaxFilters.
Network widenFilters(const Network& net, int newFilters, uint64_t seed);

// `net` plus an ownership head. Every tensor of `net` is copied by name; the
// head is initialized as Network(cfg, seed) initializes it (He-initialized
// 1x1 conv; BN gamma 1, beta 0, running mean 0, variance 1). Throws
// std::invalid_argument if net already has an ownership head.
Network addOwnershipHead(const Network& net, uint64_t seed);

} // namespace azgo
```

- [ ] **Step 4: Implement it, and widen the head**

In `src/grow.cpp`, replace:

```cpp
    // after those convs keeps its shape and is copied as is.
    widenConv(from.param("policy.conv.w"), F, 1, {0, 1}, g, rng, to.writeParam("policy.conv.w"));
    widenConv(from.param("value.conv.w"), F, 1, {0}, g, rng, to.writeParam("value.conv.w"));
    for (const char* name : {"policy.bn.gamma", "policy.bn.beta", "policy.fc.w", "policy.fc.b",
                             "value.bn.gamma", "value.bn.beta", "value.fc1.w", "value.fc1.b",
                             "value.fc2.w", "value.fc2.b"})
```

with:

```cpp
    // after those convs keeps its shape and is copied as is.
    widenConv(from.param("policy.conv.w"), F, 1, {0, 1}, g, rng, to.writeParam("policy.conv.w"));
    widenConv(from.param("value.conv.w"), F, 1, {0}, g, rng, to.writeParam("value.conv.w"));
    // The ownership head last, so a network without one draws exactly the
    // random numbers it did before the head existed.
    if (old.ownership) {
        widenConv(from.param("ownership.conv.w"), F, 1, {0}, g, rng, to.writeParam("ownership.conv.w"));
        for (const char* name : {"ownership.bn.gamma", "ownership.bn.beta"}) copyTensor(from, to, name, true);
        for (const char* name : {"ownership.bn.runMean", "ownership.bn.runVar"})
            copyTensor(from, to, name, false);
    }
    for (const char* name : {"policy.bn.gamma", "policy.bn.beta", "policy.fc.w", "policy.fc.b",
                             "value.bn.gamma", "value.bn.beta", "value.fc1.w", "value.fc1.b",
                             "value.fc2.w", "value.fc2.b"})
```

In `src/grow.cpp`, replace:

```cpp
    return wide;
}

} // namespace azgo
```

with:

```cpp
    return wide;
}

Network addOwnershipHead(const Network& net, uint64_t seed) {
    if (net.config().ownership) throw std::invalid_argument("the network already has an ownership head");
    NetConfig cfg = net.config();
    cfg.ownership = true;
    Network withHead(cfg, seed);   // the head keeps this initialization
    Network source = net;          // params() is non-const; see growBlocks
    const Tensors from(source);
    Tensors to(withHead);
    for (const auto& entry : from.params) copyTensor(from, to, entry.first, true);
    for (const auto& entry : from.buffers) copyTensor(from, to, entry.first, false);
    return withHead;
}

} // namespace azgo
```

- [ ] **Step 5: Add `--ownership` to the app**

In `apps/grow.cpp`, replace:

```cpp
// grow: deepens a checkpoint by appending residual blocks that start as exact
// identities, and/or widens it by copying channels and splitting the weights
// that read them (see include/azgo/grow.hpp), so the new network plays
// exactly as the old one did and training can continue from there.
//
//   grow --from runs/nine/checkpoints/gen_0100.azn --blocks 10 --out runs/ten/start.azn
//   grow --from runs/nine/checkpoints/gen_0100.azn --filters 96 --out runs/wide/start.azn
//
// With both options it widens first, then deepens, so the new blocks are
// built at the new width; the other order gives an equally exact network.
// Before saving, the tool checks the claim on 8 random positions: the grown
// network's policy and value must match the original's within kTolerance.
// Deepening reproduces them bit for bit. Widening is exact in real arithmetic,
```

with:

```cpp
// grow: deepens a checkpoint by appending residual blocks that start as exact
// identities, and/or widens it by copying channels and splitting the weights
// that read them, and/or adds an ownership head (see include/azgo/grow.hpp),
// so the new network plays exactly as the old one did and training can
// continue from there.
//
//   grow --from runs/nine/checkpoints/gen_0100.azn --blocks 10 --out runs/ten/start.azn
//   grow --from runs/nine/checkpoints/gen_0100.azn --filters 96 --out runs/wide/start.azn
//   grow --from runs/nine/checkpoints/gen_0100.azn --ownership --out runs/own/start.azn
//
// With both options it widens first, then deepens, so the new blocks are
// built at the new width; the other order gives an equally exact network.
// --ownership adds the head last; it changes neither policy nor value.
// Before saving, the tool checks the claim on 8 random positions: the grown
// network's policy and value must match the original's within kTolerance.
// Deepening reproduces them bit for bit. Widening is exact in real arithmetic,
```

In `apps/grow.cpp`, replace:

```cpp
namespace {

const char* const kUsage =
    "usage: grow --from CKPT [--blocks N] [--filters N] --out PATH [--seed 1]\n"
    "\n"
    "Writes PATH: CKPT widened to --filters filters, then with residual blocks\n"
    "appended up to --blocks. New channels are copies of old ones and new blocks\n"
    "are exact identities, so PATH computes what CKPT computed. Each defaults to\n"
    "CKPT's own value and may not be below it. --seed seeds the new blocks'\n"
    "convolution weights and the widening's channel copies and weight shares.\n";

constexpr double kTolerance = 1e-4;

```

with:

```cpp
namespace {

const char* const kUsage =
    "usage: grow --from CKPT [--blocks N] [--filters N] [--ownership] --out PATH [--seed 1]\n"
    "\n"
    "Writes PATH: CKPT widened to --filters filters, then with residual blocks\n"
    "appended up to --blocks. New channels are copies of old ones and new blocks\n"
    "are exact identities, so PATH computes what CKPT computed. Each defaults to\n"
    "CKPT's own value and may not be below it. --ownership adds a randomly\n"
    "initialized ownership head, which changes neither policy nor value (CKPT\n"
    "must not have one). --seed seeds the new blocks' convolution weights, the\n"
    "widening's channel copies and weight shares, and the ownership head.\n";

constexpr double kTolerance = 1e-4;

```

In `apps/grow.cpp`, replace:

```cpp
    return std::to_string(c.boardSize) + "x" + std::to_string(c.boardSize) +
           ", history " + std::to_string(c.history) + ", " + std::to_string(c.blocks) +
           " blocks, " + std::to_string(c.filters) + " filters, value_hidden " +
           std::to_string(c.valueHidden);
}

// Largest difference between the two networks' logits and values over `count`
// random {0, 1} input positions.
double maxOutputDifference(const azgo::Network& a, const azgo::Network& b, int count, azgo::Rng& rng) {
    const azgo::NetConfig& cfg = a.config();
    std::unique_ptr<azgo::Backend> backend = azgo::makeBackend("naive", 1);
```

with:

```cpp
    return std::to_string(c.boardSize) + "x" + std::to_string(c.boardSize) +
           ", history " + std::to_string(c.history) + ", " + std::to_string(c.blocks) +
           " blocks, " + std::to_string(c.filters) + " filters, value_hidden " +
           std::to_string(c.valueHidden) + (c.ownership ? ", ownership head" : "");
}

// Largest difference between the two networks' logits and values over `count`
// random {0, 1} input positions, and between their ownership maps when both
// have the head.
double maxOutputDifference(const azgo::Network& a, const azgo::Network& b, int count, azgo::Rng& rng) {
    const azgo::NetConfig& cfg = a.config();
    std::unique_ptr<azgo::Backend> backend = azgo::makeBackend("naive", 1);
```

In `apps/grow.cpp`, replace:

```cpp
    for (float& v : input) v = rng.uniform() < 0.5 ? 0.0f : 1.0f;
    std::vector<float> la(static_cast<size_t>(count) * cfg.actions()), lb(la.size());
    std::vector<float> va(count), vb(count);
    a.infer(wa, input.data(), count, la.data(), va.data());
    b.infer(wb, input.data(), count, lb.data(), vb.data());
    double worst = 0;
    for (size_t i = 0; i < la.size(); ++i) worst = std::fmax(worst, std::fabs(la[i] - lb[i]));
    for (int i = 0; i < count; ++i) worst = std::fmax(worst, std::fabs(va[i] - vb[i]));
    return worst;
}

```

with:

```cpp
    for (float& v : input) v = rng.uniform() < 0.5 ? 0.0f : 1.0f;
    std::vector<float> la(static_cast<size_t>(count) * cfg.actions()), lb(la.size());
    std::vector<float> va(count), vb(count);
    const bool ownership = a.config().ownership && b.config().ownership;
    std::vector<float> oa(ownership ? static_cast<size_t>(count) * cfg.area() : 0), ob(oa.size());
    a.infer(wa, input.data(), count, la.data(), va.data(), ownership ? oa.data() : nullptr);
    b.infer(wb, input.data(), count, lb.data(), vb.data(), ownership ? ob.data() : nullptr);
    double worst = 0;
    for (size_t i = 0; i < la.size(); ++i) worst = std::fmax(worst, std::fabs(la[i] - lb[i]));
    for (int i = 0; i < count; ++i) worst = std::fmax(worst, std::fabs(va[i] - vb[i]));
    for (size_t i = 0; i < oa.size(); ++i) worst = std::fmax(worst, std::fabs(oa[i] - ob[i]));
    return worst;
}

```

In `apps/grow.cpp`, replace:

```cpp
    try {
        args = std::make_unique<ArgParser>(argc, argv,
                                           std::vector<std::string>{"from", "blocks", "filters", "out", "seed"},
                                           std::vector<std::string>{"help"});
    } catch (const std::invalid_argument& e) {
        return usageError(e.what());
    }
```

with:

```cpp
    try {
        args = std::make_unique<ArgParser>(argc, argv,
                                           std::vector<std::string>{"from", "blocks", "filters", "out", "seed"},
                                           std::vector<std::string>{"help", "ownership"});
    } catch (const std::invalid_argument& e) {
        return usageError(e.what());
    }
```

In `apps/grow.cpp`, replace:

```cpp
        if (filters < net.config().filters)
            throw std::runtime_error("--filters " + std::to_string(filters) + " is fewer than the checkpoint's " +
                                     std::to_string(net.config().filters) + "; growing only adds filters");
        azgo::Network grown = azgo::growBlocks(azgo::widenFilters(net, filters, static_cast<uint64_t>(seed)),
                                               blocks, static_cast<uint64_t>(seed));

        azgo::Rng rng(static_cast<uint64_t>(seed));
        const double diff = maxOutputDifference(net, grown, 8, rng);
```

with:

```cpp
        if (filters < net.config().filters)
            throw std::runtime_error("--filters " + std::to_string(filters) + " is fewer than the checkpoint's " +
                                     std::to_string(net.config().filters) + "; growing only adds filters");
        if (args->flag("ownership") && net.config().ownership)
            throw std::runtime_error("--ownership: " + args->get("from") + " already has an ownership head");
        azgo::Network grown = azgo::growBlocks(azgo::widenFilters(net, filters, static_cast<uint64_t>(seed)),
                                               blocks, static_cast<uint64_t>(seed));
        if (args->flag("ownership")) grown = azgo::addOwnershipHead(grown, static_cast<uint64_t>(seed));

        azgo::Rng rng(static_cast<uint64_t>(seed));
        const double diff = maxOutputDifference(net, grown, 8, rng);
```

In `apps/grow.cpp`, replace:

```cpp
        std::printf("old: %s, %zu parameters\n", shape(net.config()).c_str(), parameterCount(net));
        std::printf("new: %s, %zu parameters\n", shape(grown.config()).c_str(), parameterCount(grown));
        if (grown.config() == net.config())
            std::printf("note: neither --blocks nor --filters changes the shape; PATH is a plain copy\n");
        std::printf("verified on 8 random positions (max difference %.2g); wrote %s\n", diff,
                    out.string().c_str());
        return 0;
```

with:

```cpp
        std::printf("old: %s, %zu parameters\n", shape(net.config()).c_str(), parameterCount(net));
        std::printf("new: %s, %zu parameters\n", shape(grown.config()).c_str(), parameterCount(grown));
        if (grown.config() == net.config())
            std::printf("note: neither --blocks, --filters nor --ownership changes the shape; "
                        "PATH is a plain copy\n");
        std::printf("verified on 8 random positions (max difference %.2g); wrote %s\n", diff,
                    out.string().c_str());
        return 0;
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R 'test_grow|integration_smoke'`
Expected: `100% tests passed out of 2`. Then the whole suite and the BLAS variant. Expected: `100% tests passed out of 26` in both builds.

- [ ] **Step 7: Commit**

```bash
git add include/azgo/grow.hpp src/grow.cpp apps/grow.cpp tests/test_grow.cpp tests/integration_smoke.sh
git commit -F - <<'EOF'
Add an ownership head with grow --ownership, and carry it through growth

The head reads the trunk and nothing reads the head, so a fresh,
randomly initialized one changes neither policy nor value: a trained
checkpoint can gain one and keep playing exactly as before, which the
grow check confirms. Deepening copies an existing head by name;
widening splits its 1x1 conv's input weights like the other heads',
drawing its shares last so networks without a head widen exactly as
before. Tests compare all three heads.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 7: The ownership map in `analyze`

**Files:**
- Modify: `apps/analyze.cpp`
- Test: `tests/integration_smoke.sh`

**Interfaces:**
- Consumes: `Network::infer` with an ownership output (Task 3), a checkpoint with a head from the smoke test (Task 6).
- Produces: for a checkpoint with a head, right after `Value (side to move): …` and a blank line, the heading `Ownership, Black + / White - (9 = certain, "." below 0.1):`, the column letters, n rows such as ` 5  +9 +9 -3 -7 -8`, and a blank line. A checkpoint without a head prints exactly what it printed before.

- [ ] **Step 1: Write the failing test**

In `tests/integration_smoke.sh`, replace:

```bash
#!/usr/bin/env bash
# End-to-end smoke test of the executables on a tiny 5x5 run:
# train -> resume -> re-rate -> arena (checkpoint and bot) -> analyze (good and
# bad input) -> bench -> grow (deeper, wider, and a run started from the grown network), plus the refusal to change the architecture on resume.
#
# usage: tests/integration_smoke.sh BUILD_DIR SOURCE_DIR
set -euo pipefail
```

with:

```bash
#!/usr/bin/env bash
# End-to-end smoke test of the executables on a tiny 5x5 run:
# train -> resume -> re-rate -> arena (checkpoint and bot) -> analyze (good and
# bad input) -> bench -> grow (deeper, wider, and a run started from the grown network), plus the refusal to change the architecture on resume,
# then the ownership head: a run with it, grow --ownership, and analyze's ownership map.
#
# usage: tests/integration_smoke.sh BUILD_DIR SOURCE_DIR
set -euo pipefail
```

In `tests/integration_smoke.sh`, replace:

```bash
cmp "$W/owner.azn" "$W/owner_run/checkpoints/gen_0000.azn" || fail "gen_0000 is not the checkpoint with the head"
test -s "$W/owner_run/checkpoints/gen_0001.azn" || fail "the run from the grown head did not train"

echo "integration_smoke: all checks passed"
```

with:

```bash
cmp "$W/owner.azn" "$W/owner_run/checkpoints/gen_0000.azn" || fail "gen_0000 is not the checkpoint with the head"
test -s "$W/owner_run/checkpoints/gen_0001.azn" || fail "the run from the grown head did not train"

echo "== analyze: the ownership map, only for a checkpoint with the head =="
out="$("$BUILD/analyze" --checkpoint "$W/owner_run/checkpoints/gen_0001.azn" --moves "C3 D4" --simulations 8)"
echo "$out"
echo "$out" | grep -q '^Ownership, Black + / White - ' || fail "analyze printed no ownership map"
# Five rows of five cells after the column letters, each " +9" .. " -9" or "  .".
rows="$(echo "$out" | sed -n '/^Ownership/,/^$/p' | grep -Ec '^ [1-5] ( [+-][1-9]|  \.){5}$')"
[ "$rows" = "5" ] || fail "the ownership map should have 5 rows of 5 cells, found $rows"
[ "$(echo "$out" | grep -n '^Value' | cut -d: -f1)" -lt "$(echo "$out" | grep -n '^Ownership' | cut -d: -f1)" ] \
    || fail "the ownership map should follow the value line"
"$BUILD/analyze" --checkpoint "$CK/gen_0003.azn" --moves "C3 D4" --simulations 8 | grep -q '^Ownership' \
    && fail "analyze printed an ownership map for a checkpoint without the head"

echo "integration_smoke: all checks passed"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build build -j && bash tests/integration_smoke.sh build . 2>&1 | tail -1`
Expected: `integration_smoke: FAILED: analyze printed no ownership map`

- [ ] **Step 3: Print the map**

`analyze` asks the network directly, because the `Evaluator` interface that search uses carries no ownership. The value it gets back is the same one the evaluator just printed.

In `apps/analyze.cpp`, replace:

```cpp
// (value and policy heat map) next to what search makes of it (MCTS visits).
// The gap between the two is the whole point of AlphaZero: search corrects
// the network's guess, and training teaches the network to guess the result.
//
//   analyze --checkpoint runs/nine/checkpoints/gen_0040.azn --moves "E5 C3 pass"
//   analyze --checkpoint gen_0040.azn --sgf game.sgf --simulations 800
```

with:

```cpp
// (value and policy heat map) next to what search makes of it (MCTS visits).
// The gap between the two is the whole point of AlphaZero: search corrects
// the network's guess, and training teaches the network to guess the result.
// A checkpoint with an ownership head also shows who it thinks will own each
// point at the end, which shows how it judges every group on the board.
//
//   analyze --checkpoint runs/nine/checkpoints/gen_0040.azn --moves "E5 C3 pass"
//   analyze --checkpoint gen_0040.azn --sgf game.sgf --simulations 800
```

In `apps/analyze.cpp`, replace:

```cpp
    "               [--simulations 400] [--komi 7.5] [--backend naive]\n"
    "\n"
    "Plays the moves (GTP coordinates, or the main line of an SGF file) from the\n"
    "empty board, then prints the network's value and policy for the position\n"
    "and the result of an MCTS search. With --sgf the file's komi is used unless\n"
    "--komi is given.\n";

constexpr char kColumns[] = "ABCDEFGHJKLMNOPQRST";   // GTP skips I
constexpr int kMctsBatch = 8;
```

with:

```cpp
    "               [--simulations 400] [--komi 7.5] [--backend naive]\n"
    "\n"
    "Plays the moves (GTP coordinates, or the main line of an SGF file) from the\n"
    "empty board, then prints the network's value, its ownership map (only for a\n"
    "checkpoint with an ownership head) and policy for the position, and the\n"
    "result of an MCTS search. With --sgf the file's komi is used unless --komi\n"
    "is given.\n";

constexpr char kColumns[] = "ABCDEFGHJKLMNOPQRST";   // GTP skips I
constexpr int kMctsBatch = 8;
```

In `apps/analyze.cpp`, replace:

```cpp
    }
}

} // namespace

int main(int argc, char** argv) {
```

with:

```cpp
    }
}

// Who the network expects to own each point when the game ends, in Black's
// terms: 9 * ownership rounded, from +9 (surely Black's) to -9 (surely
// White's), "." when |ownership| < 0.1. Stones are not drawn: whether a stone
// survives is exactly what the map answers. Same frame as printHeatMap.
void printOwnershipMap(int n, const std::vector<float>& owner) {
    std::printf("   ");
    for (int c = 0; c < n; ++c) std::printf("  %c", kColumns[c]);
    std::printf("\n");
    for (int r = 0; r < n; ++r) {
        std::printf("%2d ", n - r);
        for (int c = 0; c < n; ++c) {
            const float o = owner[static_cast<size_t>(r * n + c)];
            if (std::fabs(o) < 0.1f) std::printf("  .");
            else std::printf("%+3ld", std::lround(9.0 * o));
        }
        std::printf("\n");
    }
}

} // namespace

int main(int argc, char** argv) {
```

In `apps/analyze.cpp`, replace:

```cpp
        evaluator.evaluate(input.data(), 1, logits.data(), &value);
        std::printf("Value (side to move): %+.2f\n\n", value);

        // Priors as MCTS sees them: softmax over the legal actions only.
        const std::vector<int> legal = game.legalActions();
        std::vector<double> prior(static_cast<size_t>(actions), 0.0);
```

with:

```cpp
        evaluator.evaluate(input.data(), 1, logits.data(), &value);
        std::printf("Value (side to move): %+.2f\n\n", value);

        if (nc.ownership) {
            // The evaluator interface carries no ownership, so ask the network
            // directly; it answers for the side to move, the map is Black's.
            azgo::Workspace ws(nc, 1, *backend);
            std::vector<float> headLogits(static_cast<size_t>(actions)), owner(static_cast<size_t>(n) * n);
            float headValue = 0;
            net.infer(ws, input.data(), 1, headLogits.data(), &headValue, owner.data());
            if (board.toMove() == azgo::Color::White)
                for (float& o : owner) o = -o;
            std::printf("Ownership, Black + / White - (9 = certain, \".\" below 0.1):\n");
            printOwnershipMap(n, owner);
            std::printf("\n");
        }

        // Priors as MCTS sees them: softmax over the legal actions only.
        const std::vector<int> legal = game.legalActions();
        std::vector<double> prior(static_cast<size_t>(actions), 0.0);
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R integration_smoke`
Expected: `100% tests passed out of 1`. The test prints the map, for example:

```
Ownership, Black + / White - (9 = certain, "." below 0.1):
     A  B  C  D  E
 5  +9 +9 -3 -7 -8
 4  +9 +4 -8 -7 -8
 3  +9 +4 -1 -7 -9
 2  +9 +2 +2 -8 -6
 1  +8 +6 +3 -4 -1
```

(The numbers are a one-generation 5×5 network's, so they mean nothing; they differ between builds.) Then the whole suite and the BLAS variant. Expected: `100% tests passed out of 26` in both builds.

- [ ] **Step 5: Commit**

```bash
git add apps/analyze.cpp tests/integration_smoke.sh
git commit -F - <<'EOF'
Show the ownership map in analyze

For a checkpoint with an ownership head, analyze prints after the value
line who the network expects to own each point at the end, in Black's
terms from +9 to -9, with "." where it is unsure. That is how its
judgment of every group can be seen, which is the eye check's question.
Checkpoints without a head print exactly what they did.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 8: Teaching material: the explainer and the README

**Files:**
- Modify: `docs/algorithm-explained.md`, `docs/algorithm-explained.html`, `README.md`
- Check: `tools/check_citations.py` (unchanged)

**Interfaces:**
- Consumes: the code of Tasks 1–7. The new captions cite `Board::ownership` at `src/board.cpp:201`, `Network::forward` at `src/network.cpp:200`, `Network::trainStep` at `src/network.cpp:183` and `ReplayBuffer::encodeSample` at `src/replay_buffer.cpp:54`. The concept map cites `Board::areaScore` at 236, `replayFinalOwnership` at `src/selfplay.cpp:61`, `addOwnershipHead` at `src/grow.cpp:218` and `printOwnershipMap` at `apps/analyze.cpp:96`. These line numbers hold when Tasks 1–7 were applied exactly as written; Step 4 corrects them otherwise.
- Produces: chapter 00 shows `Board::ownership`, chapter 01 the new `trainStep`, and chapter 06 ends with "An auxiliary target: ownership". The concept map has an ownership row, the README documents the keys, `grow --ownership` and the map, and `check_citations.py` passes. Task 9 adds the measured results to the same section.

The HTML is a chapter-for-chapter rendering of the Markdown with identical captions. Every Markdown change below has its HTML twin in Step 2. Text in the HTML is one `<p>` per paragraph, with `<code>` for backticks and HTML escapes inside `<pre><code>`.

- [ ] **Step 1: Update the Markdown explainer**

Chapter 00 now quotes `Board::ownership`, the code that scores a position, and says that `areaScore` is its sum. Chapter 01's `trainStep` gains the new parameters, and `forward`'s abridged tail names the third head. Chapter 06's `encodeSample` excerpt gets the new signature, and the chapter ends with the new section.

In `docs/algorithm-explained.md`, replace:

````markdown
no judgment of which stones are "dead" — chapter 05 shows what that
does to the games.

```cpp
int Board::areaScore() const {
    // Every stone counts for its color. Every maximal empty region is flood
    // filled once; it counts for a color only if all stones it touches have
    // that color (seki liberties and dame touch both and count for nobody).
    /* ... stack, mark, black = 0, white = 0 ... */
    for (int point = 0; point < area(); ++point) {
        const int start = pad(point);
        const Color c = cells_[start];
        if (c == Color::Black) { ++black; continue; }
        if (c == Color::White) { ++white; continue; }
        if (mark[start]) continue;
        /* ... flood fill the empty region from start, counting regionSize
               and noting whether it touches Black and/or White ... */
        if (reachesBlack && !reachesWhite) black += regionSize;
        if (reachesWhite && !reachesBlack) white += regionSize;
    }
    return black - white;
}
```
*src/board.cpp:201 — `Board::areaScore` (abridged)*
````

with:

````markdown
no judgment of which stones are "dead" — chapter 05 shows what that
does to the games. The code first decides who owns each point (chapter 06
reuses that as a training target), and the score is the sum:

```cpp
void Board::ownership(int8_t* out) const {
    // Every stone belongs to its color. Every maximal empty region is flood
    // filled once and belongs to a color only if all the stones it touches
    // have that color (seki liberties and dame touch both and belong to nobody).
    /* ... region, mark ... */
    for (int point = 0; point < area(); ++point) {
        const int start = pad(point);
        const Color c = cells_[start];
        if (c == Color::Black) { out[point] = 1; continue; }
        if (c == Color::White) { out[point] = -1; continue; }
        if (mark[start]) continue;   // written with the region that reached it first
        /* ... flood fill the empty region from start, noting whether it
               touches Black and/or White ... */
        const int8_t owner = reachesBlack == reachesWhite ? 0 : (reachesBlack ? 1 : -1);
        /* ... write owner at every point of the region ... */
    }
}
```
*src/board.cpp:201 — `Board::ownership` (abridged)*

`Board::areaScore()` adds up those n² values: +1 for every point that is
Black's, −1 for every point that is White's.
````

In `docs/algorithm-explained.md`, replace:

````markdown
        reluForward(buf.out.data(), trunk);
    /* ... policy head and value head ... */
}
```
*src/network.cpp:177 — `Network::forward` (abridged)*
````

with:

````markdown
        reluForward(buf.out.data(), trunk);
    /* ... policy head, value head and (chapter 06) the ownership head ... */
}
```
*src/network.cpp:200 — `Network::forward` (abridged)*
````

In `docs/algorithm-explained.md`, replace:

````markdown
```cpp
LossStats Network::trainStep(Workspace& ws, const float* input, const float* pi, const float* z,
                             int B) {
    zeroGrad();
    forward(ws, input, B, Mode::Train);
    LossStats stats = loss(ws, pi, z, B, true);
    backward(ws, B);
    return stats;
}
```
*src/network.cpp:162 — `Network::trainStep`*
````

with:

````markdown
```cpp
LossStats Network::trainStep(Workspace& ws, const float* input, const float* pi, const float* z,
                             int B, const float* ownership, float ownershipWeight) {
    assert(ownership == nullptr || cfg_.ownership);
    zeroGrad();
    forward(ws, input, B, Mode::Train);
    LossStats stats = loss(ws, pi, z, ownership, ownershipWeight, B, true);
    backward(ws, B, cfg_.ownership && ownership != nullptr && ownershipWeight != 0.0f);
    return stats;
}
```
*src/network.cpp:183 — `Network::trainStep`*

(The `ownership` arguments are chapter 06's auxiliary target. Without
them the loss is the two terms above.)
````

In `docs/algorithm-explained.md`, replace:

````markdown
```cpp
void ReplayBuffer::encodeSample(size_t gameIndex, int ply, int history, int sym,
                                float* input, float* pi, float* z) const {
````

with:

````markdown
```cpp
void ReplayBuffer::encodeSample(size_t gameIndex, int ply, int history, int sym,
                                float* input, float* pi, float* z, float* ownership) const {
````

In `docs/algorithm-explained.md`, replace:

````markdown
    *z = g.zAt(ply);
}
```
*src/replay_buffer.cpp:53 — `ReplayBuffer::encodeSample` (abridged)*
````

with:

````markdown
    *z = g.zAt(ply);
    /* ... the ownership target (end of this chapter) ... */
}
```
*src/replay_buffer.cpp:54 — `ReplayBuffer::encodeSample` (abridged)*
````

In `docs/algorithm-explained.md`, replace:

```markdown
generation 60, while the policy loss stayed flat at 1.73–1.75
(`docs/results/5x5/log.csv`).

---
```

with:

````markdown
generation 60, while the policy loss stayed flat at 1.73–1.75
(`docs/results/5x5/log.csv`).

**An auxiliary target: ownership.** Every 9×9 network so far fails the
eye check (`docs/results/README.md`): it rates White's B1, which kills a
Black group on the bottom edge, about the same as Black's B1, which gives
the group two eyes. That is a life-and-death judgment, the value head's
job, and the value head learns it from one number per position, the
game's result `z = ±1`. That signal is weak in two ways. It is noisy: a
position's `z` also depends on every later move of both sides, so a
position that was won gets −1 whenever the game was thrown away
afterwards. And it is slow: under Tromp-Taylor rules a dead group only
costs points once it has actually been captured (chapter 05), so "this
group has one eye" and "this game was lost" can be dozens of moves apart.

KataGo (David J. Wu, *Accelerating Self-Play Learning in Go*, 2019,
arXiv:1902.10565) trains its network on auxiliary targets next to `π`
and `z`. The simplest is **ownership**: for every point, who owns it when
the game ends. The final board already holds the answer, and chapter
00's `Board::ownership` computes it, so each game now carries n² labels
(`GameRecord::finalOwnership`, from Black's view) instead of one. Every
point of a group that died is labelled with the opponent's color, which
is the life-and-death lesson in its most direct form.

The **ownership head** is the smallest head that can say this: a 1×1
convolution from the trunk's filters to one channel, batch norm, and a
`tanh` at every point, n² numbers in [−1, 1] from the side to move's view,
like the value. It has `filters` + 2 parameters (96 + 2 on the 6×96
network, next to its 1,032,349) and is optional (`ownership = true`):

```cpp
    if (cfg_.ownership) {
        ownershipConv_.forward(be, x, B, n, col, convOut);
        norm(ownershipBn_, convOut, ws.ownership_.data(), ws.ownershipXhat_.data(),
             ws.ownershipInvStd_.data());
        for (int j = 0; j < M; ++j) ws.ownership_[j] = std::tanh(ws.ownership_[j]);
    }
```
*src/network.cpp:200 — `Network::forward` (abridged)*

The loss gains a third term, the squared error averaged over the n²
points and weighted by `w = ownership_weight` (1.5 by default):

```
L = (z − v)²  −  Σₐ π(a) · log softmax(logits)ₐ  +  w · (1/n²) · Σₚ (ôₚ − oₚ)²
```

Through the `tanh` its gradient is the value head's formula once per
point, `2w(ô − o)(1 − ô²)`, divided by the batch size and n². With
`w = 0` the head exists but nothing trains it. KataGo's own ownership
loss differs in its details; the idea is the same.

**Perspective and symmetry.** A sample turns the stored ownership to the
side to move's view, as `z` is turned, and moves every point to where the
symmetry moves the input planes:

```cpp
    if (ownership) {
        assert(g.finalOwnership.size() == static_cast<size_t>(n) * n);
        const float sign = g.toMoveAt(ply) == Color::Black ? 1.0f : -1.0f;
        for (int p = 0; p < n * n; ++p)
            ownership[transformPoint(sym, p, n)] = sign * g.finalOwnership[static_cast<size_t>(p)];
    }
```
*src/replay_buffer.cpp:54 — `ReplayBuffer::encodeSample` (abridged)*

Either mistake would teach nonsense without any error: a target from the
wrong side's view says that every living group is dead, and one that is
not moved with the planes puts each point's label on another point.
`tests/test_replay_buffer.cpp` checks both by hand on small games, for
all 8 symmetries.

Search does not read the head; MCTS still asks only for the policy and
the value. The head's effect is on the shared trunk: features that
predict which groups live are the features the value head was missing.
Two tools make it usable on existing runs. `grow --ownership` adds a
fresh head to a trained checkpoint without changing what it plays, and a
replay buffer saved before ownership existed gets its targets by
replaying each game's moves (`replayFinalOwnership`), so an old run can
still donate its games. Whether the extra signal fixes the eye check is
an experiment; `docs/results/README.md` reports it.

---
````

- [ ] **Step 2: Mirror it in the HTML explainer**

In `docs/algorithm-explained.html`, replace:

```html
<pre><code>int Board::areaScore() const {
    // Every stone counts for its color. Every maximal empty region is flood
    // filled once; it counts for a color only if all stones it touches have
    // that color (seki liberties and dame touch both and count for nobody).
    /* ... stack, mark, black = 0, white = 0 ... */
    for (int point = 0; point &lt; area(); ++point) {
        const int start = pad(point);
        const Color c = cells_[start];
        if (c == Color::Black) { ++black; continue; }
        if (c == Color::White) { ++white; continue; }
        if (mark[start]) continue;
        /* ... flood fill the empty region from start, counting regionSize
               and noting whether it touches Black and/or White ... */
        if (reachesBlack &amp;&amp; !reachesWhite) black += regionSize;
        if (reachesWhite &amp;&amp; !reachesBlack) white += regionSize;
    }
    return black - white;
}</code></pre>
<span class="src">src/board.cpp:201 — <code>Board::areaScore</code> (abridged)</span>
```

with:

```html
<pre><code>void Board::ownership(int8_t* out) const {
    // Every stone belongs to its color. Every maximal empty region is flood
    // filled once and belongs to a color only if all the stones it touches
    // have that color (seki liberties and dame touch both and belong to nobody).
    /* ... region, mark ... */
    for (int point = 0; point &lt; area(); ++point) {
        const int start = pad(point);
        const Color c = cells_[start];
        if (c == Color::Black) { out[point] = 1; continue; }
        if (c == Color::White) { out[point] = -1; continue; }
        if (mark[start]) continue;   // written with the region that reached it first
        /* ... flood fill the empty region from start, noting whether it
               touches Black and/or White ... */
        const int8_t owner = reachesBlack == reachesWhite ? 0 : (reachesBlack ? 1 : -1);
        /* ... write owner at every point of the region ... */
    }
}</code></pre>
<span class="src">src/board.cpp:201 — <code>Board::ownership</code> (abridged)</span>
<p><code>Board::areaScore()</code> adds up those n² values: +1 for every point that is Black's, −1 for every point that is White's.</p>
```

In `docs/algorithm-explained.html`, replace:

```html
There is no judgment of which stones are "dead" — chapter 05 shows what that does to the games.</p>
```

with:

```html
There is no judgment of which stones are "dead" — chapter 05 shows what that does to the games. The code first decides who owns each point (chapter 06 reuses that as a training target), and the score is the sum:</p>
```

In `docs/algorithm-explained.html`, replace:

```html
    /* ... policy head and value head ... */
}</code></pre>
<span class="src">src/network.cpp:177 — <code>Network::forward</code> (abridged)</span>
```

with:

```html
    /* ... policy head, value head and (chapter 06) the ownership head ... */
}</code></pre>
<span class="src">src/network.cpp:200 — <code>Network::forward</code> (abridged)</span>
```

In `docs/algorithm-explained.html`, replace:

```html
<pre><code>LossStats Network::trainStep(Workspace&amp; ws, const float* input, const float* pi, const float* z,
                             int B) {
    zeroGrad();
    forward(ws, input, B, Mode::Train);
    LossStats stats = loss(ws, pi, z, B, true);
    backward(ws, B);
    return stats;
}</code></pre>
<span class="src">src/network.cpp:162 — <code>Network::trainStep</code></span>
```

with:

```html
<pre><code>LossStats Network::trainStep(Workspace&amp; ws, const float* input, const float* pi, const float* z,
                             int B, const float* ownership, float ownershipWeight) {
    assert(ownership == nullptr || cfg_.ownership);
    zeroGrad();
    forward(ws, input, B, Mode::Train);
    LossStats stats = loss(ws, pi, z, ownership, ownershipWeight, B, true);
    backward(ws, B, cfg_.ownership &amp;&amp; ownership != nullptr &amp;&amp; ownershipWeight != 0.0f);
    return stats;
}</code></pre>
<span class="src">src/network.cpp:183 — <code>Network::trainStep</code></span>
<p>(The <code>ownership</code> arguments are chapter 06's auxiliary target. Without them the loss is the two terms above.)</p>
```

In `docs/algorithm-explained.html`, replace:

```html
<pre><code>void ReplayBuffer::encodeSample(size_t gameIndex, int ply, int history, int sym,
                                float* input, float* pi, float* z) const {
```

with:

```html
<pre><code>void ReplayBuffer::encodeSample(size_t gameIndex, int ply, int history, int sym,
                                float* input, float* pi, float* z, float* ownership) const {
```

In `docs/algorithm-explained.html`, replace:

```html
    *z = g.zAt(ply);
}</code></pre>
<span class="src">src/replay_buffer.cpp:53 — <code>ReplayBuffer::encodeSample</code> (abridged)</span>
```

with:

```html
    *z = g.zAt(ply);
    /* ... the ownership target (end of this chapter) ... */
}</code></pre>
<span class="src">src/replay_buffer.cpp:54 — <code>ReplayBuffer::encodeSample</code> (abridged)</span>
```

In `docs/algorithm-explained.html`, replace:

```html
while the policy loss stayed flat at 1.73–1.75 (<code>docs/results/5x5/log.csv</code>).</p>
</section>
```

with:

```html
while the policy loss stayed flat at 1.73–1.75 (<code>docs/results/5x5/log.csv</code>).</p>
<p><strong>An auxiliary target: ownership.</strong> Every 9×9 network so far fails the eye check (<code>docs/results/README.md</code>): it rates White's B1, which kills a Black group on the bottom edge, about the same as Black's B1, which gives the group two eyes. That is a life-and-death judgment, the value head's job, and the value head learns it from one number per position, the game's result <code>z = ±1</code>. That signal is weak in two ways. It is noisy: a position's <code>z</code> also depends on every later move of both sides, so a position that was won gets −1 whenever the game was thrown away afterwards. And it is slow: under Tromp-Taylor rules a dead group only costs points once it has actually been captured (chapter 05), so "this group has one eye" and "this game was lost" can be dozens of moves apart.</p>
<p>KataGo (David J. Wu, <em>Accelerating Self-Play Learning in Go</em>, 2019, arXiv:1902.10565) trains its network on auxiliary targets next to <code>π</code> and <code>z</code>. The simplest is <strong>ownership</strong>: for every point, who owns it when the game ends. The final board already holds the answer, and chapter 00's <code>Board::ownership</code> computes it, so each game now carries n² labels (<code>GameRecord::finalOwnership</code>, from Black's view) instead of one. Every point of a group that died is labelled with the opponent's color, which is the life-and-death lesson in its most direct form.</p>
<p>The <strong>ownership head</strong> is the smallest head that can say this: a 1×1 convolution from the trunk's filters to one channel, batch norm, and a <code>tanh</code> at every point, n² numbers in [−1, 1] from the side to move's view, like the value. It has <code>filters</code> + 2 parameters (96 + 2 on the 6×96 network, next to its 1,032,349) and is optional (<code>ownership = true</code>):</p>
<pre><code>    if (cfg_.ownership) {
        ownershipConv_.forward(be, x, B, n, col, convOut);
        norm(ownershipBn_, convOut, ws.ownership_.data(), ws.ownershipXhat_.data(),
             ws.ownershipInvStd_.data());
        for (int j = 0; j &lt; M; ++j) ws.ownership_[j] = std::tanh(ws.ownership_[j]);
    }</code></pre>
<span class="src">src/network.cpp:200 — <code>Network::forward</code> (abridged)</span>
<p>The loss gains a third term, the squared error averaged over the n² points and weighted by <code>w = ownership_weight</code> (1.5 by default):</p>
<pre><code>L = (z − v)²  −  Σₐ π(a) · log softmax(logits)ₐ  +  w · (1/n²) · Σₚ (ôₚ − oₚ)²</code></pre>
<p>Through the <code>tanh</code> its gradient is the value head's formula once per point, <code>2w(ô − o)(1 − ô²)</code>, divided by the batch size and n². With <code>w = 0</code> the head exists but nothing trains it. KataGo's own ownership loss differs in its details; the idea is the same.</p>
<p><strong>Perspective and symmetry.</strong> A sample turns the stored ownership to the side to move's view, as <code>z</code> is turned, and moves every point to where the symmetry moves the input planes:</p>
<pre><code>    if (ownership) {
        assert(g.finalOwnership.size() == static_cast&lt;size_t&gt;(n) * n);
        const float sign = g.toMoveAt(ply) == Color::Black ? 1.0f : -1.0f;
        for (int p = 0; p &lt; n * n; ++p)
            ownership[transformPoint(sym, p, n)] = sign * g.finalOwnership[static_cast&lt;size_t&gt;(p)];
    }</code></pre>
<span class="src">src/replay_buffer.cpp:54 — <code>ReplayBuffer::encodeSample</code> (abridged)</span>
<p>Either mistake would teach nonsense without any error: a target from the wrong side's view says that every living group is dead, and one that is not moved with the planes puts each point's label on another point. <code>tests/test_replay_buffer.cpp</code> checks both by hand on small games, for all 8 symmetries.</p>
<p>Search does not read the head; MCTS still asks only for the policy and the value. The head's effect is on the shared trunk: features that predict which groups live are the features the value head was missing. Two tools make it usable on existing runs. <code>grow --ownership</code> adds a fresh head to a trained checkpoint without changing what it plays, and a replay buffer saved before ownership existed gets its targets by replaying each game's moves (<code>replayFinalOwnership</code>), so an old run can still donate its games. Whether the extra signal fixes the eye check is an experiment; <code>docs/results/README.md</code> reports it.</p>
</section>
```

- [ ] **Step 3: Update both concept maps**

The checker verifies the HTML concept map but not the Markdown table, so the Markdown numbers are set by hand here. They include the entries that Tasks 1–7 moved.

In `docs/algorithm-explained.md`, replace:

```markdown
| Tromp-Taylor area scoring, komi, the draw | `Board::areaScore` — `src/board.cpp:201`; `Game::resultForBlack` — `src/game.cpp:61` |
```

with:

```markdown
| Tromp-Taylor area scoring, komi, the draw | `Board::ownership`, `Board::areaScore` — `src/board.cpp:201, 236`; `Game::resultForBlack` — `src/game.cpp:61` |
```

In `docs/algorithm-explained.md`, replace:

```markdown
| Residual ResNet, policy and value heads | `Network::forward`, `Network::backward` — `src/network.cpp:177, 277` |
```

with:

```markdown
| Residual ResNet, policy and value heads | `Network::forward`, `Network::backward` — `src/network.cpp:200, 326` |
```

In `docs/algorithm-explained.md`, replace:

```markdown
| The loss | `Network::loss`, `Network::trainStep` — `src/network.cpp:243, 162` |
```

with:

```markdown
| The loss | `Network::loss`, `Network::trainStep` — `src/network.cpp:276, 183` |
```

In `docs/algorithm-explained.md`, replace:

```markdown
| Temperature | `Mcts::selectAction` — `src/mcts.cpp:282`; `playSelfPlayGame` — `src/selfplay.cpp:14` |
```

with:

```markdown
| Temperature | `Mcts::selectAction` — `src/mcts.cpp:282`; `playSelfPlayGame` — `src/selfplay.cpp:17` |
```

In `docs/algorithm-explained.md`, replace:

```markdown
| The value target z | `GameRecord::zAt` — `src/selfplay.cpp:9` |
| Replay buffer of whole games | `ReplayBuffer::encodeSample`, `ReplayBuffer::sample` — `src/replay_buffer.cpp:53, 75` |
| Symmetry augmentation | `transformPoint`, `transformAction` — `src/symmetry.cpp:6, 21` |
```

with:

```markdown
| The value target z | `GameRecord::zAt` — `src/selfplay.cpp:12` |
| Replay buffer of whole games | `ReplayBuffer::encodeSample`, `ReplayBuffer::sample` — `src/replay_buffer.cpp:54, 85` |
| Symmetry augmentation | `transformPoint`, `transformAction` — `src/symmetry.cpp:6, 21` |
| Ownership, an auxiliary target | `replayFinalOwnership` — `src/selfplay.cpp:61`; `addOwnershipHead` — `src/grow.cpp:218`; `printOwnershipMap` — `apps/analyze.cpp:96` |
```

In `docs/algorithm-explained.md`, replace:

```markdown
| `log.csv` rating vs. the `elo.csv` refit | `ratingAtOrBefore`, `eloCsv` — `src/trainer.cpp:91, 64` |
| One generation | `Trainer::runGeneration` — `src/trainer.cpp:313` |
```

with:

```markdown
| `log.csv` rating vs. the `elo.csv` refit | `ratingAtOrBefore`, `eloCsv` — `src/trainer.cpp:99, 72` |
| One generation | `Trainer::runGeneration` — `src/trainer.cpp:350` |
```

In `docs/algorithm-explained.md`, replace:

```markdown
| Training state and resume | `saveTrainingState` — `src/run_state.cpp:27`; `Trainer::resume` — `src/trainer.cpp:261` |
| Inspecting a position | `printHeatMap` and `main` — `apps/analyze.cpp:70, 91` |
```

with:

```markdown
| Training state and resume | `saveTrainingState` — `src/run_state.cpp:28`; `Trainer::resume` — `src/trainer.cpp:297` |
| Inspecting a position | `printHeatMap` and `main` — `apps/analyze.cpp:73, 113` |
```

In `docs/algorithm-explained.html`, replace:

```html
  <dd><code>Board::areaScore</code> <span class="loc">— src/board.cpp:201</span>; <code>Game::resultForBlack</code> <span class="loc">— src/game.cpp:61</span></dd>
```

with:

```html
  <dd><code>Board::ownership</code>, <code>Board::areaScore</code> <span class="loc">— src/board.cpp:201, 236</span>; <code>Game::resultForBlack</code> <span class="loc">— src/game.cpp:61</span></dd>
```

In `docs/algorithm-explained.html`, replace:

```html
  <dd><code>transformPoint</code>, <code>transformAction</code> <span class="loc">— src/symmetry.cpp:6, 21</span></dd>
```

with:

```html
  <dd><code>transformPoint</code>, <code>transformAction</code> <span class="loc">— src/symmetry.cpp:6, 21</span></dd>
  <dt>Ownership, an auxiliary target</dt>
  <dd><code>replayFinalOwnership</code> <span class="loc">— src/selfplay.cpp:61</span>; <code>addOwnershipHead</code> <span class="loc">— src/grow.cpp:218</span>; <code>printOwnershipMap</code> <span class="loc">— apps/analyze.cpp:96</span></dd>
```

- [ ] **Step 4: Repair and check the citations**

Tasks 1–7 moved code, so several old captions now point at the wrong lines. Run: `python3 tools/check_citations.py`
Expected: exit 1, with exactly these problems:

```
algorithm-explained.md:744: `Network::backward` cites src/network.cpp:277, belongs at 326
algorithm-explained.md:1320: `playSelfPlayGame` cites src/selfplay.cpp:14, belongs at 17
algorithm-explained.md:2229: `Trainer::runGeneration` cites src/trainer.cpp:313, belongs at 350
algorithm-explained.html: `Network::backward` cites src/network.cpp:277, markdown says 326
algorithm-explained.html: `playSelfPlayGame` cites src/selfplay.cpp:14, markdown says 17
algorithm-explained.html: `Trainer::runGeneration` cites src/trainer.cpp:313, markdown says 350
algorithm-explained.html: concept map cites src/network.cpp:177 for `Network::forward`, defined at 200
algorithm-explained.html: concept map cites src/network.cpp:277 for `Network::backward`, defined at 326
algorithm-explained.html: concept map cites src/network.cpp:243 for `Network::loss`, defined at 276
algorithm-explained.html: concept map cites src/network.cpp:162 for `Network::trainStep`, defined at 183
algorithm-explained.html: concept map cites src/selfplay.cpp:14 for `playSelfPlayGame`, defined at 17
algorithm-explained.html: concept map cites src/selfplay.cpp:9 for `GameRecord::zAt`, defined at 12
algorithm-explained.html: concept map cites src/replay_buffer.cpp:53 for `ReplayBuffer::encodeSample`, defined at 54
algorithm-explained.html: concept map cites src/replay_buffer.cpp:75 for `ReplayBuffer::sample`, defined at 85
algorithm-explained.html: concept map cites src/trainer.cpp:91 for `ratingAtOrBefore`, defined at 99
algorithm-explained.html: concept map cites src/trainer.cpp:64 for `eloCsv`, defined at 72
algorithm-explained.html: concept map cites src/trainer.cpp:313 for `Trainer::runGeneration`, defined at 350
algorithm-explained.html: concept map cites src/run_state.cpp:27 for `saveTrainingState`, defined at 28
algorithm-explained.html: concept map cites src/trainer.cpp:261 for `Trainer::resume`, defined at 297
algorithm-explained.html: concept map cites apps/analyze.cpp:70 for `printHeatMap`, defined at 73
algorithm-explained.html: concept map cites apps/analyze.cpp:91 for `main`, defined at 113

21 citation problem(s)
```

No line may say `cannot locate`; if one does, a caption names a symbol that no longer exists and must be fixed by hand. Run: `python3 tools/check_citations.py --fix && python3 tools/check_citations.py`
Expected: `repaired 3 markdown and 18 html citations`, then `all 38 markdown citations resolve, and the html agrees`. (The count is of distinct symbols: the new captions cite `Network::forward` and `ReplayBuffer::encodeSample` a second time.)

- [ ] **Step 5: Update the README**

`analyze` gains a sentence about the map. A new subsection, "The ownership head", follows "Growing a network", and "Learn how it works" links the spec and this plan. The `grow` output quoted there is real: it is the 6×96 generation 50 of `runs/9x9-wide96`, which Task 9 grows the same way.

In `README.md`, replace:

```markdown
**`analyze`** plays a move list (GTP coordinates) or an SGF file, then
prints the board, the network's value and policy, and an MCTS search
(`docs/results/9x9/analyze-atari.txt`):
```

with:

```markdown
**`analyze`** plays a move list (GTP coordinates) or an SGF file, then
prints the board, the network's value and policy, and an MCTS search
(`docs/results/9x9/analyze-atari.txt`). For a checkpoint with an
ownership head (below) it also prints, after the value line, an
ownership map: who the network expects to own each point at the end, in
Black's terms, from +9 (surely Black's) to −9 (surely White's), with `.`
where it is unsure (|ownership| < 0.1):
```

In `README.md`, replace:

```markdown
The presets' learning-rate schedule restarts at 0.02, which is the rate
for a network that knows nothing; for a trained one it would wreck what
the network learned. Override it, for example `--set lr_schedule=0:0.002`
(the 9x9 preset's late-training rate).
```

with:

````markdown
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
````

In `README.md`, replace:

```markdown
  — the rating-aware arena: opponents chosen by rating, the uniform bot and `--rerate`.
```

with:

```markdown
  — the rating-aware arena: opponents chosen by rating, the uniform bot and `--rerate`.
- [`docs/superpowers/specs/2026-10-03-ownership-head-design.md`](docs/superpowers/specs/2026-10-03-ownership-head-design.md)
  and [`docs/superpowers/plans/2026-10-03-ownership-head.md`](docs/superpowers/plans/2026-10-03-ownership-head.md)
  — the ownership head: an auxiliary per-point target, `grow --ownership` and the analyze map.
```

- [ ] **Step 6: Verify**

Run: `python3 tools/check_citations.py && cmake --build build -j && ctest --test-dir build -N | tail -1 && grep -c 'ownership' docs/algorithm-explained.md docs/algorithm-explained.html README.md`
Expected: `all 38 markdown citations resolve, and the html agrees`, `Total Tests: 26`, then three nonzero counts. Open `docs/algorithm-explained.html` in a browser and check chapter 06: the new section renders with three code blocks (the head, the loss line and the target encoding) and no stray Markdown. Check chapter 00 too: the `Board::ownership` excerpt and the sentence after it.

- [ ] **Step 7: Commit**

```bash
git add docs/algorithm-explained.md docs/algorithm-explained.html README.md
git commit -F - <<'EOF'
Teach the ownership head in the explainer and README

Chapter 06 ends with "An auxiliary target: ownership": why one game
result is a weak and slow signal for life and death, how per-point
ownership densifies it (KataGo), the head and its loss, and the
perspective and symmetry handling that would otherwise fail silently.
Chapter 00 now quotes Board::ownership, of which areaScore is the sum,
and chapter 01 the new trainStep. The README documents the two keys,
grow --ownership and the analyze map, and every citation is re-anchored.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 9: The experiment: 50 generations with the head, against the 200-simulation run

This task is operational, not TDD. Its "tests" are the spec's comparisons and a set of sanity checks, run with the project's own tools. **Report results honestly**, whatever they are. The settings below are the spec's and are fixed: do not change `ownership_weight`, the learning rate, the simulations, the seed or anything else to improve a number, and do not rerun an arm because its result is disappointing. One run is one seed; say so. If a result looks wrong, use superpowers:systematic-debugging before changing anything. That covers a sanity check in Step 4 that fails, an ownership loss that is not finite or does not fall at all, a map that is all `.` or all one sign on a mixed board, or a run that crashes. Find the cause first, then fix it in a new commit with a test, and only then rerun.

**Files:**
- Create (not in git): `runs/bin-ownership/`, `runs/ownership.sh`, `runs/grown/wide96-g50-own.azn`, `runs/grown/sims200-g25-own.azn`, `runs/grown/sims200-g50-own.azn`, `runs/9x9-ownership/`, `runs/ownership-eval/`
- Create: `docs/results/9x9-ownership/` (the run's `log.csv`, `elo.csv`, `results.csv`, `references.csv`, `init.txt`, `progress.txt`, the four match files, the four analyses and `ownership.sh`)
- Modify: `docs/results/README.md`, `README.md`, `docs/algorithm-explained.md`, `docs/algorithm-explained.html`

**Interfaces:**
- Consumes: Tasks 1–8; the run directories `runs/9x9-wide96` (the start and donor buffer, `state.bin` version 2) and `runs/9x9-sims200` (the comparison arm), which hold checkpoints that are not in git.
- Produces: the published run, the comparisons the spec lists, and the verdict.

The comparison is fair by construction. `runs/9x9-sims200` started from the same network (`runs/9x9-wide96` generation 50) and the same buffer, with the same seed and settings (`runs/sims-experiment.sh`). Only `ownership = true` differs. The grown start plays exactly like that network, because the head cannot change policy or value. So generation 1's self-play must reproduce the 200 arm's game for game (Step 4 checks this).

- [ ] **Step 1: Build both variants and run the full test suite**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure
cmake -S . -B build-blas -DCMAKE_BUILD_TYPE=Release -DAZ_USE_BLAS=ON && cmake --build build-blas -j && ctest --test-dir build-blas --output-on-failure
```

Expected: `100% tests passed out of 26` in both builds.

- [ ] **Step 2: Check the inputs**

```bash
ls runs/9x9-wide96/checkpoints/gen_0050.azn runs/9x9-sims200/checkpoints/gen_0025.azn runs/9x9-sims200/checkpoints/gen_0050.azn
od -An -tu4 -j4 -N4 runs/9x9-wide96/state.bin
cat runs/9x9-sims200/init.txt
grep -E '^(seed|backend|filters|simulations|lr_schedule|ownership)' runs/9x9-sims200/config.cfg
wc -l < runs/9x9-sims200/log.csv
```

Expected: the three checkpoints are listed; `od` prints `2` (a version-2 state, so the run will recompute the buffer's ownership); `init.txt` names `runs/9x9-wide96/checkpoints/gen_0050.azn` and `runs/9x9-wide96`, with `buffer_positions = 199972` and `buffer_games = 2272`; the config prints `seed = 1`, `backend = blas`, `filters = 96`, `simulations = 200`, `lr_schedule = 0:0.002` and no `ownership` line; `log.csv` has 51 lines.

- [ ] **Step 3: Write the script and start it**

The binaries are copied first, so that rebuilding during the run cannot change it (the earlier experiments did the same, with `runs/bin-sims` and `runs/bin-widen`).

```bash
mkdir -p runs/bin-ownership && cp build-blas/{train,arena,analyze,grow} runs/bin-ownership/
```

Create `runs/ownership.sh`:

```bash
#!/bin/bash
# Ownership head: give the 6x96 generation 50 an ownership head, train 50 generations
# with the settings of runs/9x9-sims200 plus ownership = true, then evaluate.
set -uo pipefail
cd "$(dirname "$0")/.."
B=runs/bin-ownership; E=runs/ownership-eval; mkdir -p $E runs/grown
START=runs/grown/wide96-g50-own.azn
EYE="A2 A3 B2 B3 C2 C3 D2 D3 D1 E3 F7 E2 G5 E1"
echo "start $(date)"
$B/grow --from runs/9x9-wide96/checkpoints/gen_0050.azn --ownership --out $START || exit 1
$B/train --config configs/9x9.cfg --run runs/9x9-ownership --init-from $START \
  --buffer-from runs/9x9-wide96 --set filters=96 --set lr_schedule=0:0.002 \
  --set simulations=200 --set ownership=true --generations 50 > runs/9x9-ownership.out 2>&1
echo "train exit $? $(date)"
O=runs/9x9-ownership/checkpoints; L=runs/9x9-sims200/checkpoints
# The 200 arm's checkpoints with a head added (it changes no move), so that arena accepts the pair.
for g in 25 50; do
  $B/grow --from $L/gen_00$g.azn --ownership --out runs/grown/sims200-g$g-own.azn | tail -1
done
run(){ $B/arena --a "$1" --b "$2" --games 100 --simulations 200 --komi 7.5 --backend blas > $E/$3.txt 2>&1; echo "$3: $(head -2 $E/$3.txt | tr '\n' ' ')"; }
run $O/gen_0025.azn $O/gen_0000.azn own25-vs-start
run $O/gen_0050.azn $O/gen_0000.azn own50-vs-start
run $O/gen_0025.azn runs/grown/sims200-g25-own.azn own25-vs-s200g25
run $O/gen_0050.azn runs/grown/sims200-g50-own.azn own50-vs-s200g50
for s in 200 800; do
  $B/analyze --checkpoint $O/gen_0050.azn --moves "$EYE" --simulations $s --komi 7.5 --backend blas > $E/eye-own50-s$s.txt 2>&1
done
# The same group after White's killing B1 (Black first plays elsewhere, C6), and after Black's saving B1.
$B/analyze --checkpoint $O/gen_0050.azn --moves "$EYE C6 B1" --simulations 200 --komi 7.5 --backend blas > $E/eye-own50-killed.txt 2>&1
$B/analyze --checkpoint $O/gen_0050.azn --moves "$EYE B1" --simulations 200 --komi 7.5 --backend blas > $E/eye-own50-saved.txt 2>&1
echo "all done $(date)"
```

Then start it:

```bash
chmod +x runs/ownership.sh && nohup runs/ownership.sh > runs/ownership.log 2>&1 &
```

The 200-simulation arm took 351 s per generation, 4.9 h of generation time for 50 generations (its `log.csv`). The head adds one 1×1 conv per position, which is negligible, so expect about 5 hours, plus a few minutes for the start (it loads the 42 MB donor buffer, replays its 2,272 games and calibrates the bot) and about 15 minutes of evaluation. Follow it with `tail -f runs/9x9-ownership.out`. Each line looks like `gen 7 | selfplay 100 games avg … | train loss p … v … o … lr 0.002 …s | arena elo …`.

- [ ] **Step 4: Sanity checks before reading any number**

Run these as soon as generation 1 is done (they need only `init.txt`, `config.cfg`, `results.csv` and generation 1):

```bash
cat runs/9x9-ownership/init.txt
diff <(grep -v '^ownership' runs/9x9-ownership/config.cfg) runs/9x9-sims200/config.cfg
head -2 runs/9x9-ownership/results.csv; head -2 runs/9x9-sims200/results.csv
cmp <(sed -n 2p runs/9x9-ownership/log.csv | cut -d, -f2-5) <(sed -n 2p runs/9x9-sims200/log.csv | cut -d, -f2-5)
cmp runs/9x9-ownership/sgf/gen_0001_game_00.sgf runs/9x9-sims200/sgf/gen_0001_game_00.sgf
```

Expected:
- `init.txt` shows `buffer_positions = 199972` and `buffer_games = 2272`, the same as the 200 arm's: the version-2 buffer loaded and replayed completely.
- `diff` prints nothing: apart from the two `ownership` lines, the configs are identical.
- Both `results.csv` files start with `-1,0,80,0`: the bot's calibration against generation 0 is identical.
- Neither `cmp` prints anything: generation 1's 100 self-play games are the 200 arm's (`100,89.24,0.1075,0.2400`), because they were played by the same network with the same seeds. The check run mentioned below reproduced exactly this, game record included.

Any difference means the start is not what the comparison assumes. Stop the run (`pkill -f runs/ownership.sh; pkill -f bin-ownership/train`) and debug before going on.

When the run has finished (`all done` in `runs/ownership.log`):

```bash
grep -c '^gen ' runs/9x9-ownership.out                                    # expected: 50
awk -F, 'NR>1 && $16 == "" { n++ } END { print n + 0 }' runs/9x9-ownership/log.csv   # expected: 0
awk -F, 'NR==2 || NR==51 { print $1, $16 }' runs/9x9-ownership/log.csv   # first and last ownership loss
cat runs/ownership.log
```

Generation 1's ownership loss should be about 0.6: a check run of exactly this generation, made while writing this plan, logged 0.59046. The loss is the mean over the generation's 200 steps. The untrained head starts near 1, because it outputs values near 0 while nearly every target is ±1 (a finished 9×9 board has few dame), and it falls within the first generation. Record the first and last values. A last value that is not below the first is a "looks wrong" result (see the top of this task). `runs/ownership.log` lists the four match lines (`own25-vs-start: A … scored …`) and the grow lines; each grow line must end in `max difference 0`, which confirms that the head left play unchanged.

- [ ] **Step 5: Publish the files**

```bash
D=docs/results/9x9-ownership; mkdir -p $D
cp runs/9x9-ownership/{log,elo,results,references}.csv $D/
sed "s|$PWD/||" runs/9x9-ownership/init.txt > $D/init.txt
grep -E '^(gen |done)' runs/9x9-ownership.out > $D/progress.txt
cp runs/ownership-eval/*.txt runs/ownership.sh $D/
ls $D
```

Expected: `elo.csv eye-own50-killed.txt eye-own50-s200.txt eye-own50-s800.txt eye-own50-saved.txt init.txt log.csv own25-vs-s200g25.txt own25-vs-start.txt own50-vs-s200g50.txt own50-vs-start.txt ownership.sh progress.txt references.csv results.csv` (one per line), and `init.txt` with repo-relative paths.

- [ ] **Step 6: Read the results**

Match scores (each file's first line is `A … scored S/100 = p (95% CI lo-hi), distinct games d`):

```bash
head -1 docs/results/9x9-ownership/own*.txt
```

The rule for each match is fixed in advance. With A = the ownership run, A is **better** if the interval's lower bound is above 0.5, **worse** if its upper bound is below 0.5, and **not measurably different** otherwise. Convert p to Elo as the earlier sections do, 400·log10(p/(1−p)), with the interval's ends converted the same way. For the context, the 200 arm alone scored 64/100 (0.542–0.727) against the same start at generation 50 (`docs/results/9x9-sims/s200g50-vs-start.txt`).

The eye check is the original criterion: B1 in the top 3 by visits, judged at 200 simulations, with 800 reported too.

```bash
for s in 200 800; do f=docs/results/9x9-ownership/eye-own50-s$s.txt
  echo "== $s"; grep -E '^(Value|MCTS value)' $f; sed -n '/^Top 10 moves by visits/,$p' $f | head -11
done
```

Write down the value before search, the MCTS value, the top three moves with their visits, and B1's rank by visits ("not in the top 10" if it is absent). The 200 arm's numbers for the same position are in `docs/results/9x9-sims/eye-sims200-s{200,800}.txt` (C6 42 and D6 40 at 200 simulations; B1 absent).

The ownership maps answer whether the head knows the group's fate:

```bash
for f in s200 killed saved; do echo "== $f"
  sed -n '/^Ownership/,/^$/p' docs/results/9x9-ownership/eye-own50-$f.txt; done
```

The group is the Black stones A2, B2, C2, D2 and D1, and the points it encloses are A1, B1 and C1. From the maps, record these eight points' values in each of the three positions, and the White wall's (A3–E3, E2, E1) in the eye position. A head that understands the shape shows clearly negative values on the group after White's B1, the `killed` map: the group is dead there, so in Black's terms its points will be White's. After Black's B1, the `saved` map, it shows clearly positive values. Before either move it can be uncertain. Report what the maps show, whether or not it matches this.

- [ ] **Step 7: Write up the results**

`docs/results/README.md`: after the section "More simulations in self-play (400 vs 200)", add a section `### An ownership head (auxiliary target)` in the same form:
- the hypothesis, in one paragraph (the spec's Problem: the value head's single, noisy, slow signal; the ownership target gives n² dense labels);
- *Setup.* The grown start, the donor buffer (199,972 positions from 2,272 games, recomputed from a version-2 `state.bin`), and the settings equal to `9x9-sims/sims200` except `ownership = true` and `ownership_weight = 1.5`. The Step 4 cross-check (generation 1 reproduced the 200 arm game for game), the commands (`9x9-ownership/ownership.sh`), and the file list;
- *Cost.* Mean seconds per generation (self-play + training + arena) from `log.csv`, next to the 200 arm's 351.3 (201.4 + 94.4 + 55.4);
- *Training curves.* The ownership loss at generations 1, 25 and 50, and the mean policy and value losses next to the 200 arm's 2.390 and 0.362 (different data, not a strength measure);
- the **Matches** table (match, score, 95% interval, Elo with interval), with the four rows of Step 6;
- the **Eye check** table (simulations, value before search, MCTS value, top visits, B1's rank), and a sentence on whether it passes by the original criterion;
- the **Ownership maps**: the eye position's map, with the eight group points' values in the three positions as a small table;
- the **Verdict**, applying the Step 6 rules without softening. It states whether the ownership run is better, worse or not measurably different from the 200 arm at generations 25 and 50, whether the eye check passes, and whether the maps show that the head knows the group dies after White's B1. It ends with what the run does not test: one seed, one weight (1.5), 50 generations, the head used only as a training signal (not in search), and any lever named in the spec's non-goals.

In "Files", add `9x9-ownership/` and what each file is (as for `9x9-sims/`).

`README.md`, Results section: after the "9×9, more simulations" bullet, add a bullet "**9×9, ownership head.**" with the two match results against the 200 arm, the eye-check result, and a link to the new section (`docs/results/README.md#an-ownership-head-auxiliary-target`).

`docs/algorithm-explained.md` and `.html` (identical text): replace the last sentence of the ownership section, "Whether the extra signal fixes the eye check is an experiment; `docs/results/README.md` reports it." (in the HTML, the same sentence with `<code>` tags), with a paragraph headed `**The experiment.**`. It gives the setup in one sentence, the generation-50 result against the 200 arm and the eye-check result (each naming its file in `docs/results/9x9-ownership/`), and then shows the `killed` map. In the Markdown it goes in a fenced block below the position's diagram from the same file; in the HTML it goes in `<pre><code>`. Add one sentence saying what the map shows about the group. Every number must name its file.

Run: `python3 tools/check_citations.py`
Expected: `all 38 markdown citations resolve, and the html agrees`.

- [ ] **Step 8: Commit**

```bash
git add docs/results README.md docs/algorithm-explained.md docs/algorithm-explained.html
git commit -F - <<'EOF'
Document the ownership head experiment

<One paragraph: the ownership run against the 200-simulation arm at
generations 25 and 50 with intervals and the verdict for each; the eye
check result at 200 and 800 simulations; what the ownership map shows
about the bottom group before and after White's B1. One sentence on cost
per generation. State plainly what the single run does not show.>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```
