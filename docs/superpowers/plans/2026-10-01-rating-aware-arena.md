# Rating-Aware Arena Opponents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the arena's fixed ladder with opponents chosen by rating, add a uniform-prior MCTS bot as a fixed yardstick and a promoted reference set, and add `train --rerate` so the existing reference runs can be re-measured.
**Architecture:** `selectOpponents` and `strongestReference` (pure functions in `arena`) pick g−1, the strongest reference and the references with an expected score closest to 0.7. A new `arena_round` component runs one round (bot calibration → refit → selection → matches → promotion) behind a `MatchFn` callback, so the live trainer, `--rerate` and synthetic tests all run the same code. The reference set is persisted in `state.bin` version 2, and `references.csv` reports it next to `results.csv` and `elo.csv`.
**Tech Stack:** C++17, CMake ≥ 3.16, the C++ standard library and `std::thread`; optional BLAS behind `-DAZ_USE_BLAS=ON`; assert-based test executables run by CTest; bash for the integration smoke test; Python 3 standard library for the docs tooling.
**Spec:** docs/superpowers/specs/2026-10-01-rating-aware-arena-design.md

## Global Constraints

- C++17, CMake ≥ 3.16, default build type Release; no dependencies beyond the C++ standard library, except optional BLAS behind `AZ_USE_BLAS` (default OFF).
- Namespace `azgo`; public headers in `include/azgo/`, sources in `src/`, apps in `apps/`, tests in `tests/`. A new `src/*.cpp` or `tests/test_*.cpp` is picked up by the CMake glob; no CMake edit is needed.
- Tests: one assert-based executable per component, `tests/test_<component>.cpp`, compiled with `-UNDEBUG` so asserts fire in Release. No test framework.
- All randomness goes through `azgo::Rng`, with streams seeded by `deriveSeed(...)`. Never `std::random_device`, `std::mt19937` or `std::*_distribution`.
- Determinism: self-play, arena, calibration, promotion and re-rating output must be identical for a given seed and backend, whatever the worker count.
- A resumed run must continue byte-identically to an uninterrupted one (`test_resume_is_byte_identical` keeps passing).
- Each new header opens with a short teaching comment block explaining the concept it implements; changed headers keep theirs accurate.
- Code style: camelCase functions, PascalCase types, `trailing_` members, `kConstants`, 4-space indent, ≤ 100 columns as in the surrounding code.
- Player IDs: generations are ≥ 0; bot k (1-based position in `bot_simulations`) is player −k. Generation 0 is pinned at 0 Elo.
- `bot_simulations`: comma-separated positive integers ≤ 100000, distinct, or empty. Default `{100}`. Locked on resume.
- `calibration_games`: even, 2–10000, default 80. `promote_threshold`: 0.5–1, default 0.85. `arena_opponents`: 0–32, default 2.
- `elo.csv` keeps its format and lists generations only. `references.csv` has the header `player,kind,simulations,promoted_at,elo`.
- Commits: author Mark Castelluccio <markacastelluccio@gmail.com> (repo-local config, already set); imperative subject without a conventional-commit prefix; body explains why; message ends with `Co-Authored-By: Claude <model> <noreply@anthropic.com>`, naming the model that writes the commit (the blocks below show `Claude Opus 5.5`; substitute your own model name).

## Review Focus

1. **Resuming a run made before this change** (`state.bin` version 1, a `config.cfg` without the new keys, `arena_opponents = 4` from the old preset). The resume must work: references become {bot −1, 0}, the bot plays 80 calibration games at the first arena, and `state.bin` is saved as version 2. Test: Task 5 `test_resume_from_version1_state`.
2. **Hand-edited `bot_simulations`.** `50, 200` is accepted (printed `50,200`) and an empty value means no bots. `100,`, `,100`, `100,,200`, `100 200`, `0`, `1e3`, `100001` and `100,100` are rejected with `file:line: bot_simulations:`. Test: Task 2 `test_bot_simulations_rejects_bad_lists`.
3. **A score exactly at the promotion threshold** (17/20 with the float key 0.85): the network is promoted and plays exactly `calibration_games` extra games. Test: Task 5 `test_promotion`.
4. **No bots and `arena_opponents = 0`.** Every generation plays only g−1 and the strongest reference, no negative ID appears in any file, and `references.csv` starts at generation 0. Test: Task 5 `test_no_bots_no_extras`.
5. **`--rerate` on something that is not a complete run.** A directory without `config.cfg` gives a `ConfigError` naming it. An architecture override is refused before anything is written. A missing checkpoint ends the replay at the gap. `state.bin` and `config.cfg` are never touched. Tests: Task 6 `test_rerate_refusals_and_gaps` and `test_rerate_reproduces_live_results`.

---

## Notes Before You Start

**Changes to the requested task breakdown, and why:**
- `ladderOpponents` is removed in **Task 5**, not Task 3. The trainer calls it until Task 5 rewrites the arena phase, so removing it in Task 3 would leave the build broken for two tasks.
- The spec's trainer test "on a tiny run whose network quickly beats gen 0, gen 0 stops appearing" is written against the shared round function with a synthetic arena (`test_saturated_anchor_is_dropped`, Task 5). Real tiny networks win or lose depending on floating-point details that differ between compilers, so a test that needs a particular outcome would be brittle. The trainer tests check what holds for any outcome: calibration rows, promotion bookkeeping, byte identity on resume and independence from the worker count.

**Spec ambiguities resolved here (each is binding for the tasks below):**
- `bot_simulations` entries must be distinct. Bots are named after their simulation count (`bot100`), so `100,100` would be two players with one name. Spaces around entries are allowed. The printed form is `50,200`.
- Promotion compares `score ≥ (threshold − 1e-6) · games`. `promote_threshold` is a float, 0.85f is 0.8500000238…, and a plain comparison would refuse 17 of 20.
- Seeds: arena matches keep `deriveSeed(seed, g, kPhaseArena, opponent)`. Bot calibration uses `deriveSeed(seed, 0, kPhaseCalibration, bot)` and promotion games use `deriveSeed(seed, g, kPhaseCalibration, strongest)`, with `kPhaseCalibration = 4`. The phase constants move from `trainer.hpp` to `run_state.hpp`, so that `arena_round` can use them without including the trainer.
- `promoted_at` is the generation itself for a promoted checkpoint (it joins right after its own arena round) and 0 for the initial set. It is derived, not stored.
- `loadTrainingState(path, numBots = 0)` takes the configured bot count. A version-1 file gets `references = initialReferences(numBots)`, and `Trainer::resume` passes `cfg.botSimulations.size()`.
- A bot "needs calibration" when it has no result against generation 0. `runArenaRound` calibrates such bots first. This covers `startNew`, the first arena after a version-1 resume, and `--rerate` with a single rule.
- `--rerate` follows `arena_every` as live training does, ends at the first missing checkpoint, and accepts `--set` for tunable keys (in memory only). It refuses architecture keys. Its `log.csv` lists opponents as player IDs separated by spaces.
- Resume refuses a changed `bot_simulations` with `cannot change keys on resume: bot_simulations (bot k is arena player -k, so the bot list is fixed for a run)`. A change to architecture keys alone keeps the existing message.
- `startNew` writes `elo.csv` through the same function as every later refit, so its first row becomes `0,0.0,,,` (it used to be `0,0,,,`).
- When g−1 is unrated (only possible with `arena_every > 1`), R̂ = 0, exactly as the spec says.

## File Structure

```
include/azgo/elo.hpp          Task 1   comment: players may be bots (negative IDs)
src/elo.cpp                   Task 1   fromCsv accepts negative IDs
include/azgo/config.hpp       Task 2   botSimulations, calibrationGames, promoteThreshold; resumeLockedDiff
src/config.cpp                Task 2   parsers/printers for the new keys; arena_opponents 0-32
configs/{5x5,9x9,19x19}.cfg   Task 2   arena_opponents 2, bot_simulations 100, calibration_games 80, promote_threshold 0.85
include/azgo/arena.hpp        Task 3   ArenaPlayer, generalized playMatch, strongestReference, selectOpponents (Task 5 removes ladderOpponents)
src/arena.cpp                 Task 3   uniform-bot seats, selection
apps/arena.cpp                Task 3   --b-bot SIMS
include/azgo/run_state.hpp    Task 4   TrainingState::references, initialReferences, version 2; Task 5: seed phases
src/run_state.cpp             Task 4   format version 2, version-1 loading
include/azgo/arena_round.hpp  Task 5   NEW: MatchFn, calibrateBots, runArenaRound, opponentSummary, referencesCsv
src/arena_round.cpp           Task 5   NEW
include/azgo/trainer.hpp      Task 5   seed phases moved out; Task 6: rerateRun
src/trainer.cpp               Tasks 2, 4, 5, 6   resume lock, references, the shared round, references.csv, rerate
apps/train.cpp                Task 6   --rerate
tests/test_elo.cpp            Task 1
tests/test_config.cpp         Task 2
tests/test_arena.cpp          Tasks 3, 5
tests/test_run_state.cpp      Task 4
tests/test_arena_round.cpp    Task 5   NEW: synthetic-arena tests of the round
tests/test_trainer.cpp        Tasks 2, 4, 5, 6
tests/integration_smoke.sh    Tasks 3, 5, 6
docs/algorithm-explained.md, .html, README.md, docs/superpowers/specs/2026-09-29-alphazero-go-design.md   Task 7
docs/results/** , README.md, explainer numbers   Task 8 (operational)
```

## Conventions for Every Task

**Build and test commands** (from the repo root):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure -R test_arena    # one component
ctest --test-dir build --output-on-failure                  # everything
```

BLAS variant (run it at the end of every code task):

```bash
cmake -S . -B build-blas -DCMAKE_BUILD_TYPE=Release -DAZ_USE_BLAS=ON
cmake --build build-blas -j && ctest --test-dir build-blas --output-on-failure
```

`ctest` does not build: always run `cmake --build build -j` first. A test whose code refers to something that does not exist yet fails at **compile** time. That is the expected "fails" state in step 2 of a TDD cycle (`cmake --build` prints the error and the target is not built).

**Edits.** "Replace" steps quote the exact current text. It must occur exactly once in the file. Replace it verbatim and keep the surrounding code. "Full file" steps replace the whole file.

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

### Task 1: Results table and Elo accept bots (negative player IDs)

**Files:**
- Modify: `include/azgo/elo.hpp`, `src/elo.cpp`
- Test: `tests/test_elo.cpp`

**Interfaces:**
- Consumes: nothing new.
- Produces: `ResultsTable` (including `fromCsv`/`toCsv`) and `fitBradleyTerry` accept any `int` player IDs, negative ones included, with `a != b`. Negative IDs sort before generation 0 in `pairs()`, `players()` and the CSV. Signatures are unchanged.

`addMatch` already asserts only `a != b`, and `fitBradleyTerry` keys everything by `std::map<int, …>`. Only the CSV reader rejects negative IDs today.

- [ ] **Step 1: Write the failing test**

In `tests/test_elo.cpp`, replace:

```cpp
#include <cstdio>
#include <map>
```

with:

```cpp
#include <cstdio>
#include <iterator>
#include <map>
```

Replace:

```cpp
static bool csvThrows(const std::string& csv, const std::string& expectInMessage) {
```

with:

```cpp
// Bots are players -1, -2, ...: they sort before generation 0, round-trip
// through the CSV and are rated like anyone else.
static void test_negative_ids() {
    ResultsTable t;
    t.addMatch(0, -1, 10, 8.0);   // generation 0 scored 8 of 10 against bot 1
    t.addMatch(-2, 3, 4, 1.0);
    t.addMatch(-1, -2, 6, 3.0);
    assert((t.players() == std::vector<int>{-2, -1, 0, 3}));
    std::string csv = t.toCsv();
    assert(csv == "a,b,games,score_a\n-2,-1,6,3\n-2,3,4,1\n-1,0,10,2\n");
    assert(ResultsTable::fromCsv(csv).toCsv() == csv);

    // The same known-ladder check as above, with two of the players being bots.
    std::map<int, double> truth{{-2, -300.0}, {-1, 150.0}, {0, 0.0}, {4, 500.0}};
    ResultsTable r;
    const int games = 100000;
    for (auto i = truth.begin(); i != truth.end(); ++i)
        for (auto j = std::next(i); j != truth.end(); ++j)
            r.addMatch(i->first, j->first, games, games * expectedScore(i->second, j->second));
    auto elo = fitBradleyTerry(r, 0);
    assert(elo.size() == 4 && elo.at(0) == 0.0);
    for (const auto& [player, rating] : truth) assert(near(elo.at(player), rating, 2.0));
}

static bool csvThrows(const std::string& csv, const std::string& expectInMessage) {
```

Replace:

```cpp
    assert(csvThrows("a,b,games,score_a\n1,1,2,1\n", "line 2"));    // a == b
```

with:

```cpp
    assert(csvThrows("a,b,games,score_a\n1,1,2,1\n", "line 2"));    // a == b
    assert(csvThrows("a,b,games,score_a\n-1,-1,2,1\n", "line 2"));  // a == b, bots too
```

Replace:

```cpp
    test_csv_round_trip();
```

with:

```cpp
    test_csv_round_trip();
    test_negative_ids();
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build build -j && ./build/test_elo`
Expected: the build succeeds and the test aborts with
`terminating due to uncaught exception of type std::runtime_error: results csv line 2: players must be distinct generations >= 0`.

- [ ] **Step 3: Implement**

In `src/elo.cpp`, replace:

```cpp
        if (a == b || a < 0 || b < 0) fail("players must be distinct generations >= 0");
```

with:

```cpp
        // Generations are >= 0 and bots < 0, so any int is a player.
        if (a == b) fail("players must be distinct");
```

In `include/azgo/elo.hpp`, replace:

```cpp
// All arena results so far. Players are generation numbers.
```

with:

```cpp
// All arena results so far. Players are generation numbers (>= 0) and
// fixed-strength bots (-1, -2, ...: bot k is player -k).
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_elo`
Expected: `100% tests passed out of 1`. Then run the whole suite (`ctest --test-dir build --output-on-failure`). Expected: `100% tests passed out of 24`.

- [ ] **Step 5: Commit**

```bash
git add include/azgo/elo.hpp src/elo.cpp tests/test_elo.cpp
git commit -F - <<'EOF'
Accept negative player IDs in the results table

Uniform bots join the arena as players -1, -2, ... and their results live
in the same table as the checkpoints'. addMatch and the Bradley-Terry fit
already handled any int; only the CSV reader insisted on generations >= 0.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 2: Config keys for bots, calibration and promotion

**Files:**
- Modify: `include/azgo/config.hpp`, `src/config.cpp`, `src/trainer.cpp` (resume check), `configs/5x5.cfg`, `configs/9x9.cfg`, `configs/19x19.cfg`
- Test: `tests/test_config.cpp`, `tests/test_trainer.cpp`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `Config` fields: `int arenaOpponents = 2` (range 0–32), `std::vector<int> botSimulations = {100}`, `int calibrationGames = 80`, `float promoteThreshold = 0.85f`.
  - Keys `bot_simulations`, `calibration_games` and `promote_threshold`, printed in this order right after `arena_opponents`.
  - `std::vector<std::string> azgo::resumeLockedDiff(const Config& a, const Config& b)`: `architectureDiff(a, b)` followed by `"bot_simulations"` if the lists differ.
  - `Trainer::resume` refuses any key in `resumeLockedDiff`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_config.cpp`, replace:

```cpp
    assert(c.arenaEvery == 1 && c.arenaGames == 20 && c.arenaOpponents == 4);
```

with:

```cpp
    assert(c.arenaEvery == 1 && c.arenaGames == 20 && c.arenaOpponents == 2);
    assert((c.botSimulations == std::vector<int>{100}));
    assert(c.calibrationGames == 80 && c.promoteThreshold == 0.85f);
```

Replace:

```cpp
    assert(c19.valueHidden == 256 && c19.simulations == 400 && c19.arenaGames == 10);
```

with:

```cpp
    assert(c19.valueHidden == 256 && c19.simulations == 400 && c19.arenaGames == 10);

    // Every preset uses the same rating-aware arena settings.
    for (const Config& c : {c5, c9, c19}) {
        assert(c.arenaOpponents == 2);
        assert((c.botSimulations == std::vector<int>{100}));
        assert(c.calibrationGames == 80 && c.promoteThreshold == 0.85f);
    }
```

Replace:

```cpp
static void test_errors_name_key_and_line() {
```

with:

```cpp
static void test_arena_keys() {
    // A list, printed without spaces whatever the input spacing.
    Config c = parseConfigText("bot_simulations = 50, 200\n", "t.cfg");
    assert((c.botSimulations == std::vector<int>{50, 200}));
    assert(contains(configToString(c), "bot_simulations = 50,200\n"));
    assert(configToString(parseConfigText(configToString(c), "rt")) == configToString(c));

    // Empty means no bots, in a file and as an override, and round-trips.
    Config none = parseConfigText("bot_simulations =\n", "t.cfg");
    assert(none.botSimulations.empty());
    assert(configToString(parseConfigText(configToString(none), "rt")) == configToString(none));
    Config o;
    applyOverride(o, "bot_simulations=");
    assert(o.botSimulations.empty());
    applyOverride(o, "bot_simulations=100000");
    assert((o.botSimulations == std::vector<int>{100000}));

    c = parseConfigText("calibration_games = 2\npromote_threshold = 1\narena_opponents = 0\n", "t.cfg");
    assert(c.calibrationGames == 2 && c.promoteThreshold == 1.0f && c.arenaOpponents == 0);
    c = parseConfigText("calibration_games = 10000\npromote_threshold = 0.5\narena_opponents = 32\n", "t.cfg");
    assert(c.calibrationGames == 10000 && c.promoteThreshold == 0.5f && c.arenaOpponents == 32);

    std::string msg = errorOf("calibration_games = 7");
    assert(contains(msg, "t.cfg:1: calibration_games: ") && contains(msg, "even"));
    assert(contains(errorOf("calibration_games = 0"), "t.cfg:1: calibration_games: "));
    assert(contains(errorOf("calibration_games = 10002"), "t.cfg:1: calibration_games: "));
    assert(contains(errorOf("promote_threshold = 0.4"), "t.cfg:1: promote_threshold: "));
    assert(contains(errorOf("promote_threshold = 1.01"), "t.cfg:1: promote_threshold: "));
    assert(contains(errorOf("arena_opponents = 33"), "t.cfg:1: arena_opponents: "));
    assert(contains(errorOf("arena_opponents = -1"), "t.cfg:1: arena_opponents: "));
}

// Review Focus 2: hand-edited bot lists.
static void test_bot_simulations_rejects_bad_lists() {
    for (const char* bad : {"0", "-5", "100001", "1e3", "abc", "100,", ",100", "100,,200",
                            "100 200", "100,100"}) {
        std::string msg = errorOf(std::string("bot_simulations = ") + bad);
        assert(contains(msg, "t.cfg:1: bot_simulations: "));
    }
    assert(contains(errorOf("bot_simulations = 100,100"), "duplicate"));
    assert(contains(errorOf("bot_simulations = 50,0"), "bot 2"));
}

static void test_resume_locked_diff() {
    Config a, b;
    assert(resumeLockedDiff(a, b).empty());
    // Tunable arena keys are not locked.
    b.arenaOpponents = 5;
    b.calibrationGames = 20;
    b.promoteThreshold = 0.9f;
    assert(resumeLockedDiff(a, b).empty());
    // bot_simulations is locked but is not an architecture key.
    b.botSimulations = {50};
    assert((resumeLockedDiff(a, b) == std::vector<std::string>{"bot_simulations"}));
    assert(architectureDiff(a, b).empty());
    b.filters = 32;
    assert((resumeLockedDiff(a, b) == std::vector<std::string>{"filters", "bot_simulations"}));
    b = a;
    b.botSimulations = {};
    assert((resumeLockedDiff(a, b) == std::vector<std::string>{"bot_simulations"}));
}

static void test_errors_name_key_and_line() {
```

Replace:

```cpp
    test_architecture_diff();
```

with:

```cpp
    test_architecture_diff();
    test_arena_keys();
    test_bot_simulations_rejects_bad_lists();
    test_resume_locked_diff();
```

In `tests/test_trainer.cpp`, inside `test_resume_overrides`, replace:

```cpp
    assert(threw);
    assert(readFile(configPath(dir.str())) == before);

    // Tunable keys are accepted and persisted.
```

with:

```cpp
    assert(threw);
    assert(readFile(configPath(dir.str())) == before);

    // So is bot_simulations: bots are numbered by their position in it.
    threw = false;
    try {
        Trainer::resume(dir.str(), {"bot_simulations=50"});
    } catch (const ConfigError& e) {
        const std::string msg = e.what();
        threw = msg.find("cannot change") != std::string::npos &&
                msg.find("bot_simulations") != std::string::npos;
    }
    assert(threw);
    assert(readFile(configPath(dir.str())) == before);

    // Tunable keys are accepted and persisted.
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cmake --build build -j`
Expected: the build fails in `tests/test_config.cpp` with `error: no member named 'botSimulations' in 'azgo::Config'` (and similar errors for `calibrationGames`, `promoteThreshold` and `resumeLockedDiff`).

Run: `cmake --build build -j --target test_trainer && ./build/test_trainer`
Expected: `Assertion failed: (threw), function test_resume_overrides`. The resume fails with `--set: bot_simulations: unknown key`, which is not a refusal.

- [ ] **Step 3: Add the fields and the declaration**

In `include/azgo/config.hpp`, replace:

```cpp
// Architecture keys fix the network's tensor shapes and cannot change on
// resume; every other key is tunable between generations.
```

with:

```cpp
// Architecture keys fix the network's tensor shapes and cannot change on
// resume. Neither can bot_simulations: bot k is arena player -k, so a changed
// list would give old player IDs to new bots. Every other key is tunable
// between generations.
```

Replace:

```cpp
    int arenaOpponents = 4;            // most recent ladder entries (generation 0 always added)
```

with:

```cpp
    int arenaOpponents = 2;            // extra references chosen by rating (0-32), besides g-1 and the strongest
    std::vector<int> botSimulations = {100};   // uniform-prior MCTS bots; bot k is player -k; empty = none
    int calibrationGames = 80;         // even: each bot vs generation 0 at the start; each promotion
    float promoteThreshold = 0.85f;    // score vs the strongest reference that adds g to the references
```

Replace:

```cpp
// Keys that fix the network's shape: board_size, history, blocks, filters, value_hidden.
std::vector<std::string> architectureDiff(const Config& a, const Config& b);
```

with:

```cpp
// Keys that fix the network's shape: board_size, history, blocks, filters, value_hidden.
std::vector<std::string> architectureDiff(const Config& a, const Config& b);

// Keys a resume must refuse to change: architectureDiff, then bot_simulations.
std::vector<std::string> resumeLockedDiff(const Config& a, const Config& b);
```

- [ ] **Step 4: Parse, print and lock the new keys**

In `src/config.cpp`, replace:

```cpp
#include "azgo/config.hpp"
#include <cerrno>
```

with:

```cpp
#include "azgo/config.hpp"
#include <algorithm>
#include <cerrno>
```

Replace:

```cpp
// The shortest "%.Ng" (N = 6..9) that reads back as exactly the same float:
```

with:

```cpp
// Colors alternate game by game, so an odd count would favor one side.
std::string parseEvenGames(const std::string& v, int& out) {
    int games = 0;
    std::string reason = parseInt(v, 2, 10000, games);
    if (!reason.empty()) return reason;
    if (games % 2 != 0) return "expected an even number (colors alternate), got " + quoted(v);
    out = games;
    return "";
}

// "100" or "50, 200": the bots' simulation counts, each in [1, 100000] and
// all different (bots are named by their count, e.g. "bot100"). An empty
// value means no bots. Splitting by hand, rather than with getline, keeps the
// empty item of "100," so that it is rejected.
std::string parseBotList(const std::string& v, std::vector<int>& out) {
    std::vector<int> list;
    for (size_t start = 0; !v.empty();) {
        size_t comma = v.find(',', start);
        // const: a non-const std::string would make quoted() below pick std::quoted by ADL.
        const std::string item =
            trim(v.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        int sims = 0;
        std::string reason = parseInt(item, 1, 100000, sims);
        if (!reason.empty()) return "bot " + std::to_string(list.size() + 1) + ": " + reason;
        if (std::find(list.begin(), list.end(), sims) != list.end())
            return "duplicate entry " + quoted(item) + " (each bot needs its own simulation count)";
        list.push_back(sims);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    out = list;
    return "";
}

std::string printBotList(const std::vector<int>& list) {
    std::string out;
    for (size_t i = 0; i < list.size(); ++i) out += (i ? "," : "") + std::to_string(list[i]);
    return out;
}

// The shortest "%.Ng" (N = 6..9) that reads back as exactly the same float:
```

(That line is the first line of the comment above `formatFloat`; the rest of that comment stays.) Then, in the `kKeys` table, replace:

```cpp
    {"arena_games", false,
     [](Config& c, const std::string& v) -> std::string {
         int games = 0;
         std::string reason = parseInt(v, 2, 10000, games);
         if (!reason.empty()) return reason;
         // Colors alternate game by game, so an odd count would favor one side.
         if (games % 2 != 0) return "expected an even number (colors alternate), got " + quoted(v);
         c.arenaGames = games;
         return "";
     },
     printInt<&Config::arenaGames>},
    {"arena_opponents", false, intKey<&Config::arenaOpponents, 1, 32>, printInt<&Config::arenaOpponents>},
```

with:

```cpp
    {"arena_games", false, [](Config& c, const std::string& v) { return parseEvenGames(v, c.arenaGames); },
     printInt<&Config::arenaGames>},
    {"arena_opponents", false, intKey<&Config::arenaOpponents, 0, 32>, printInt<&Config::arenaOpponents>},
    {"bot_simulations", false,
     [](Config& c, const std::string& v) { return parseBotList(v, c.botSimulations); },
     [](const Config& c) { return printBotList(c.botSimulations); }},
    {"calibration_games", false,
     [](Config& c, const std::string& v) { return parseEvenGames(v, c.calibrationGames); },
     printInt<&Config::calibrationGames>},
    {"promote_threshold", false,
     [](Config& c, const std::string& v) { return parseFloat(v, 0.5, 1, c.promoteThreshold); },
     printFloat<&Config::promoteThreshold>},
```

At the end of the file, replace:

```cpp
        if (e.architecture && e.print(a) != e.print(b)) keys.push_back(e.key);
    return keys;
}
```

with:

```cpp
        if (e.architecture && e.print(a) != e.print(b)) keys.push_back(e.key);
    return keys;
}

std::vector<std::string> resumeLockedDiff(const Config& a, const Config& b) {
    std::vector<std::string> keys = architectureDiff(a, b);
    if (a.botSimulations != b.botSimulations) keys.push_back("bot_simulations");
    return keys;
}
```

In `src/trainer.cpp` (`Trainer::resume`), replace:

```cpp
    const std::vector<std::string> diff = architectureDiff(saved, cfg);
    if (!diff.empty()) {
        std::string keys;
        for (const std::string& k : diff) keys += (keys.empty() ? "" : ", ") + k;
        throw ConfigError("cannot change architecture keys on resume: " + keys);
    }
```

with:

```cpp
    const std::vector<std::string> diff = resumeLockedDiff(saved, cfg);
    if (!diff.empty()) {
        std::string keys;
        for (const std::string& k : diff) keys += (keys.empty() ? "" : ", ") + k;
        // resumeLockedDiff lists bot_simulations last, after any architecture keys.
        if (diff.back() == "bot_simulations")
            throw ConfigError("cannot change keys on resume: " + keys +
                              " (bot k is arena player -k, so the bot list is fixed for a run)");
        throw ConfigError("cannot change architecture keys on resume: " + keys);
    }
```

- [ ] **Step 5: Run the tests and watch the presets fail**

Run: `cmake --build build -j && ./build/test_config`
Expected: `Assertion failed: (c.arenaOpponents == 2), function test_presets_load`. The presets still say 4 and lack the new keys.

- [ ] **Step 6: Update the three presets**

In each of `configs/5x5.cfg`, `configs/9x9.cfg` and `configs/19x19.cfg`, replace the arena comment line:

```
# arena: how often, how many games per opponent (even), ladder size, search budget, gating
```

with:

```
# arena: how often, games per opponent (even), opponents chosen by rating, the
# fixed bot, calibration and promotion, search budget, gating
```

In `configs/5x5.cfg` and `configs/9x9.cfg`, replace:

```
arena_opponents = 4
```

with:

```
# arena_opponents 4 -> 2, and it now counts extra references chosen by rating
# (docs/superpowers/specs/2026-10-01-rating-aware-arena-design.md). Each
# generation plays g-1, the strongest reference, and the arena_opponents
# references whose expected score is closest to 0.7. The old ladder always
# ended at generation 0, which the 9x9 reference run beat 20-0 in 110 of 150
# matches: games that carry almost no information and capped Elo near 645.
arena_opponents = 2
# One uniform-prior MCTS bot (all priors equal, value 0) at 100 simulations:
# a yardstick that is the same in every run. It cannot change on resume.
bot_simulations = 100
# Games each bot plays generation 0 when a run starts, and extra games a
# newly promoted reference plays against the strongest one.
calibration_games = 80
# A generation that scores at least this against the strongest reference
# joins the reference set.
promote_threshold = 0.85
```

In `configs/19x19.cfg`, replace:

```
arena_opponents = 2
```

with:

```
# arena_opponents 2 kept, but it now counts extra references chosen by rating
# (docs/superpowers/specs/2026-10-01-rating-aware-arena-design.md). Each
# generation plays g-1, the strongest reference, and the arena_opponents
# references whose expected score is closest to 0.7. The old ladder always
# ended at generation 0, which the 9x9 reference run beat 20-0 in 110 of 150
# matches: games that carry almost no information and capped Elo near 645.
arena_opponents = 2
# One uniform-prior MCTS bot (all priors equal, value 0) at 100 simulations:
# a yardstick that is the same in every run. It cannot change on resume.
bot_simulations = 100
# Games each bot plays generation 0 when a run starts, and extra games a
# newly promoted reference plays against the strongest one.
# On 19x19 the 80 calibration games add a long first step to a run; a short
# check can pass --set calibration_games=10.
calibration_games = 80
# A generation that scores at least this against the strongest reference
# joins the reference set.
promote_threshold = 0.85
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: `100% tests passed out of 24`. The trainer still uses the ladder. `arena_opponents = 2` only shortens it, and `bot_simulations` is not read by the trainer yet.

- [ ] **Step 8: Commit**

```bash
git add include/azgo/config.hpp src/config.cpp src/trainer.cpp configs tests/test_config.cpp tests/test_trainer.cpp
git commit -F - <<'EOF'
Add bot, calibration and promotion config keys

bot_simulations lists the uniform-prior bots that become the arena's
fixed yardstick; calibration_games and promote_threshold drive the
reference set. Bots are numbered by their position in the list, so
resume refuses to change it, like an architecture key. arena_opponents
now counts extra references chosen by rating; 0 is valid and the
presets use 2.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 3: Arena players, uniform bots and opponent selection

**Files:**
- Modify (full file): `include/azgo/arena.hpp`, `src/arena.cpp`, `apps/arena.cpp`
- Modify: `tests/integration_smoke.sh`
- Test: `tests/test_arena.cpp`

**Interfaces:**
- Consumes: nothing from Tasks 1–2.
- Produces:
  - `struct azgo::ArenaPlayer { const Network* net = nullptr; int botSimulations = 0; static ArenaPlayer network(const Network&); static ArenaPlayer bot(int simulations); }`.
  - `MatchResult azgo::playMatch(const ArenaPlayer& a, const ArenaPlayer& b, const ArenaConfig& cfg, uint64_t seed, int workers, const std::string& backendName)`, plus the unchanged two-`Network` overload as a thin wrapper.
  - `int azgo::strongestReference(const std::map<int, double>& ratings, const std::vector<int>& references)`.
  - `std::vector<int> azgo::selectOpponents(int generation, const std::map<int, double>& ratings, const std::vector<int>& references, int maxExtra, std::optional<int> bestNetwork = std::nullopt)`.
  - `arena --b-bot SIMS`.
  - `ladderOpponents` stays until Task 5, because the trainer still calls it.

A bot searches with a `UniformEvaluator` (all logits 0, value 0) built for the board size and history of the network it plays. Everything else (no noise, the opening temperature, tree reuse, colors, seeds) is the network arena's.

- [ ] **Step 1: Write the failing tests**

In `tests/test_arena.cpp`, replace:

```cpp
#include <cstdio>
#include <stdexcept>
```

with:

```cpp
#include <cstdio>
#include <map>
#include <stdexcept>
```

Replace:

```cpp
static bool throwsInvalid(const Network& a, const Network& b, const ArenaConfig& cfg) {
```

with:

```cpp
// A bot is a full arena player: deterministic for a seed, independent of the
// worker count, and playable from either side.
static void test_bot_matches() {
    Network a(tinyNet(), 11);
    ArenaConfig cfg = tinyArena();
    cfg.temperatureMoves = 2;
    const ArenaPlayer net = ArenaPlayer::network(a), bot = ArenaPlayer::bot(16);
    MatchResult one = playMatch(net, bot, cfg, 42, 1, "naive");
    MatchResult three = playMatch(net, bot, cfg, 42, 3, "naive");
    MatchResult again = playMatch(net, bot, cfg, 42, 1, "naive");
    checkBookkeeping(one, 8);
    assert(sameResult(one, three));
    assert(sameResult(one, again));
    checkBookkeeping(playMatch(bot, net, cfg, 42, 2, "naive"), 8);

    // With komi 100 White wins every game, so each side scores 4 of 8
    // whichever player is the bot.
    cfg.komi = 100.0f;
    MatchResult r = playMatch(bot, net, cfg, 7, 2, "naive");
    assert(r.scoreA == 4.0 && r.winsA == 4 && r.winsB == 4);
}

// The two-network signature is a thin wrapper: same result as two ArenaPlayers.
static void test_network_wrapper() {
    Network a(tinyNet(), 11), b(tinyNet(), 22);
    ArenaConfig cfg = tinyArena();
    assert(sameResult(playMatch(a, b, cfg, 9, 2, "naive"),
                      playMatch(ArenaPlayer::network(a), ArenaPlayer::network(b), cfg, 9, 2, "naive")));
}

static bool playersThrow(const ArenaPlayer& a, const ArenaPlayer& b) {
    try {
        playMatch(a, b, tinyArena(), 1, 1, "naive");
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

static void test_rejects_bad_players() {
    Network a(tinyNet(), 1);
    assert(playersThrow(ArenaPlayer::bot(8), ArenaPlayer::bot(8)));            // nobody has a network
    assert(playersThrow(ArenaPlayer::network(a), ArenaPlayer::bot(0)));        // a bot must search
    assert(playersThrow(ArenaPlayer::bot(-3), ArenaPlayer::network(a)));
}

static void test_strongest_reference() {
    const std::map<int, double> ratings{{-1, 120.0}, {0, 0.0}, {5, 300.0}, {7, 300.0}};
    assert(strongestReference(ratings, {-1, 0, 5, 7}) == 7);   // tie: larger ID
    assert(strongestReference(ratings, {-1, 0}) == -1);       // a bot can be strongest
    assert(strongestReference(ratings, {0, 9}) == 0);          // 9 is unrated: skipped
    bool threw = false;
    try {
        strongestReference(ratings, {3, 4});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

// Expected scores for g, rated at R = ratings[g-1], against a reference rated r:
// E = 1 / (1 + 10^((r - R) / 400)). The comments give |E - 0.7|.
static void test_select_opponents() {
    using V = std::vector<int>;
    // Generation 1: g-1 is generation 0; the strongest reference is the bot.
    assert((selectOpponents(1, {{-1, 120.0}, {0, 0.0}}, {-1, 0}, 2) == V{0, -1}));

    // R = 800 (generation 9). Strongest: 6. Extras: 4 (E 0.882, 0.18),
    // then bot -1 (E 0.983, 0.28), then 0 (E 0.990, 0.29).
    const std::map<int, double> r{{-1, 100.0}, {0, 0.0}, {4, 450.0}, {6, 700.0}, {9, 800.0}};
    const V refs{-1, 0, 4, 6};
    assert((selectOpponents(10, r, refs, 0) == V{9, 6}));
    assert((selectOpponents(10, r, refs, 1) == V{9, 6, 4}));
    assert((selectOpponents(10, r, refs, 2) == V{9, 6, 4, -1}));
    assert((selectOpponents(10, r, refs, 5) == V{9, 6, 4, -1, 0}));

    // Gating appends the best network, once.
    assert((selectOpponents(10, r, refs, 1, 8) == V{9, 6, 4, 8}));
    assert((selectOpponents(10, r, refs, 1, 4) == V{9, 6, 4}));
    assert((selectOpponents(10, r, refs, 1, 9) == V{9, 6, 4}));

    // The saturated anchor: generation 0 is 900 below R. 4 (E 0.909, 0.21)
    // beats 0 (E 0.994, 0.29) for the one extra place.
    const std::map<int, double> far{{0, 0.0}, {4, 500.0}, {6, 950.0}, {9, 900.0}};
    assert((selectOpponents(10, far, {0, 4, 6}, 1) == V{9, 6, 4}));

    // Ties in |E - 0.7| go to the smaller ID: bots -2 and -1 are both rated 100.
    const std::map<int, double> tie{{-2, 100.0}, {-1, 100.0}, {0, 0.0}, {5, 400.0}, {6, 300.0}};
    assert((selectOpponents(7, tie, {-2, -1, 0, 5}, 1) == V{6, 5, -2}));
    assert((selectOpponents(7, tie, {-2, -1, 0, 5}, 2) == V{6, 5, -2, -1}));

    // Unrated references are never chosen; an unrated g-1 counts as R = 0, so
    // bot -1 at -150 (E 0.703) is the best extra.
    const std::map<int, double> u{{-1, -150.0}, {0, 0.0}, {4, 500.0}};
    assert((selectOpponents(12, u, {-1, 0, 3, 4}, 1) == V{11, 4, -1}));
    assert((selectOpponents(12, u, {-1, 0, 3, 4}, 3) == V{11, 4, -1, 0}));

    // g-1 can itself be the strongest reference: no duplicate.
    assert((selectOpponents(6, {{0, 0.0}, {5, 300.0}}, {0, 5}, 2) == V{5, 0}));

    bool threw = false;
    try {
        selectOpponents(0, r, refs, 2);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

static bool throwsInvalid(const Network& a, const Network& b, const ArenaConfig& cfg) {
```

Replace:

```cpp
    test_rejects_bad_setup();
```

with:

```cpp
    test_rejects_bad_setup();
    test_bot_matches();
    test_network_wrapper();
    test_rejects_bad_players();
    test_strongest_reference();
    test_select_opponents();
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build build -j`
Expected: the build fails in `tests/test_arena.cpp` with errors such as `unknown type name 'ArenaPlayer'` and `use of undeclared identifier 'strongestReference'` (GCC: `'ArenaPlayer' does not name a type`).

- [ ] **Step 3: Rewrite the arena header**

Full file `include/azgo/arena.hpp`:

```cpp
// The arena: players meet with search but without the exploration noise of
// self-play, so the result measures playing strength. Colors alternate game
// by game (Black has the first-move advantage), and a few opening moves are
// sampled at temperature 1 so the games are not all the same. A player is a
// network or a uniform bot: MCTS that knows only the rules, a fixed yardstick
// that is the same in every run. Each new generation meets the opponents its
// results say the most about (selectOpponents): a game the new network is
// sure to win or sure to lose carries almost no information.
#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include "azgo/network.hpp"

namespace azgo {

struct ArenaConfig {
    int games = 20;              // even; A plays Black in even-indexed games
    int simulations = 100;       // a network's search budget per move
    int mctsBatch = 8;
    float cPuct = 1.5f;
    int temperatureMoves = 4;    // opening plies sampled at temperature 1
    float komi = 7.5f;
    int maxMoves = 0;            // 0 = defaultMaxMoves(n)
};

struct MatchResult {
    int games = 0;
    double scoreA = 0;           // wins + 0.5 * draws for A
    int winsA = 0, winsB = 0, draws = 0;
    int distinctGames = 0;       // distinct move sequences
};

// One side of a match. With net, the player searches with the network at
// ArenaConfig::simulations. With net == nullptr it is a uniform bot: MCTS
// whose priors are all equal and whose value is always 0, searching with
// botSimulations. A bot's strength depends only on its simulation count.
struct ArenaPlayer {
    const Network* net = nullptr;
    int botSimulations = 0;
    static ArenaPlayer network(const Network& n) { return {&n, 0}; }
    static ArenaPlayer bot(int simulations) { return {nullptr, simulations}; }
};

// Plays cfg.games games between a and b on `workers` threads, no root noise,
// tree reuse on. Board size and history come from whichever player has a
// network. Throws std::invalid_argument for two bots, two networks with
// different NetConfigs, a bot with botSimulations < 1, or an odd or < 2 game
// count. Game i uses Rng(deriveSeed(seed, i)). Deterministic for a seed.
MatchResult playMatch(const ArenaPlayer& a, const ArenaPlayer& b, const ArenaConfig& cfg,
                      uint64_t seed, int workers, const std::string& backendName);

// Two networks: the same as playMatch(ArenaPlayer::network(a), ArenaPlayer::network(b), ...).
MatchResult playMatch(const Network& a, const Network& b, const ArenaConfig& cfg,
                      uint64_t seed, int workers, const std::string& backendName);

// Opponents for generation g: g-1, g-2, g-4, g-8, ... (>= 1), the first
// `maxOpponents` of those, then 0 if not already present. Empty for g == 0.
std::vector<int> ladderOpponents(int generation, int maxOpponents);

// The rated reference with the highest rating, ties to the larger player ID.
// References missing from `ratings` are skipped; throws std::invalid_argument
// if none is rated.
int strongestReference(const std::map<int, double>& ratings, const std::vector<int>& references);

// Opponents for generation g >= 1 (std::invalid_argument otherwise), chosen
// from `ratings`, the Bradley-Terry refit of every result so far:
//   1. g-1, always;
//   2. the strongest reference, always;
//   3. up to maxExtra more rated references: those for which g's expected
//      score is closest to 0.7, ties to the smaller ID. g's rating is
//      estimated as that of g-1, or 0 if g-1 is unrated;
//   4. bestNetwork, if given (gating).
// No player appears twice; the order is the order above.
std::vector<int> selectOpponents(int generation, const std::map<int, double>& ratings,
                                 const std::vector<int>& references, int maxExtra,
                                 std::optional<int> bestNetwork = std::nullopt);

// score / games >= threshold.
bool passesGate(double score, int games, double threshold);

// Score of one finished game for player A: 1, 0.5 (resultForBlack == 0) or 0.
double matchScore(float resultForBlack, bool aIsBlack);

} // namespace azgo
```

- [ ] **Step 4: Rewrite the arena implementation**

Full file `src/arena.cpp`:

```cpp
#include "azgo/arena.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include "azgo/backend.hpp"
#include "azgo/evaluator.hpp"
#include "azgo/game.hpp"
#include "azgo/mcts.hpp"
#include "azgo/parallel.hpp"
#include "azgo/rng.hpp"

namespace azgo {

namespace {

// One player's per-thread resources and search settings. A network gets its
// own backend and workspace; a bot needs only a UniformEvaluator.
struct Seat {
    std::unique_ptr<Backend> backend;
    std::unique_ptr<Evaluator> evaluator;
    MctsConfig mcts;
    Seat(const ArenaPlayer& player, const NetConfig& shape, const ArenaConfig& cfg,
         const std::string& backendName) {
        mcts.batchSize = cfg.mctsBatch;
        mcts.cPuct = cfg.cPuct;
        mcts.addNoise = false;   // measuring strength, not exploring
        if (player.net) {
            backend = makeBackend(backendName, 1);
            evaluator.reset(new LocalEvaluator(*player.net, *backend, cfg.mctsBatch));
            mcts.simulations = cfg.simulations;
        } else {
            evaluator.reset(new UniformEvaluator(shape.boardSize, shape.history, 0.0f, cfg.mctsBatch));
            mcts.simulations = player.botSimulations;
        }
    }
};

struct WorkerContext {
    Seat a, b;
};

struct GameOutcome {
    double scoreA = 0;
    std::vector<int> moves;
};

GameOutcome playArenaGame(WorkerContext& ctx, const ArenaConfig& cfg, int n, uint64_t seed,
                          int index) {
    Rng rng(deriveSeed(seed, static_cast<uint64_t>(index)));
    bool aIsBlack = index % 2 == 0;   // colors alternate, so A is Black in half the games
    Game game(n, cfg.komi, cfg.maxMoves > 0 ? cfg.maxMoves : defaultMaxMoves(n));
    Mcts searchA(ctx.a.mcts, *ctx.a.evaluator), searchB(ctx.b.mcts, *ctx.b.evaluator);

    while (!game.isTerminal()) {
        bool blackToMove = game.board().toMove() == Color::Black;
        Mcts& mover = blackToMove == aIsBlack ? searchA : searchB;
        mover.search(game, rng);
        // A few sampled opening moves make the games differ from each other.
        float temperature = game.moveCount() < cfg.temperatureMoves ? 1.0f : 0.0f;
        int action = mover.selectAction(temperature, rng);
        game.play(action);
        // Both trees follow the game, so each reuses its subtree next turn.
        // A tree that has never searched has nothing to re-root.
        if (searchA.nodeCount() > 0) searchA.advance(action);
        if (searchB.nodeCount() > 0) searchB.advance(action);
    }
    return {matchScore(game.resultForBlack(), aIsBlack), game.moves()};
}

// Information carried by one game whose expected score is p is proportional
// to p(1 - p): 0.25 at p = 0.5, 0.21 at 0.7, under 0.01 at 0.99. 0.7 keeps
// most of the information while still letting the new network show progress
// against an opponent it should beat.
constexpr double kTargetScore = 0.7;

// Bradley-Terry on the Elo scale: the expected score of `rating` against `opponent`.
double expectedScore(double rating, double opponent) {
    return 1.0 / (1.0 + std::pow(10.0, (opponent - rating) / 400.0));
}

} // namespace

MatchResult playMatch(const ArenaPlayer& a, const ArenaPlayer& b, const ArenaConfig& cfg,
                      uint64_t seed, int workers, const std::string& backendName) {
    if (!a.net && !b.net) throw std::invalid_argument("playMatch: two bots (one player must be a network)");
    if (a.net && b.net && a.net->config() != b.net->config())
        throw std::invalid_argument("playMatch: networks have different architectures");
    for (const ArenaPlayer* p : {&a, &b})
        if (!p->net && p->botSimulations < 1)
            throw std::invalid_argument("playMatch: a bot needs at least 1 simulation, got " +
                                        std::to_string(p->botSimulations));
    if (cfg.games < 2 || cfg.games % 2 != 0)
        throw std::invalid_argument("playMatch: games must be even and >= 2, got " +
                                    std::to_string(cfg.games));

    // A bot has no network, so the board size and history come from the
    // player that has one.
    const NetConfig shape = (a.net ? a.net : b.net)->config();
    int threads = std::min(resolveWorkers(workers), cfg.games);
    std::vector<std::unique_ptr<WorkerContext>> contexts;
    for (int w = 0; w < threads; ++w)
        contexts.emplace_back(new WorkerContext{Seat(a, shape, cfg, backendName),
                                                Seat(b, shape, cfg, backendName)});

    // Results are stored by game index, never by completion order, so the
    // match is identical for any number of workers.
    std::vector<GameOutcome> outcomes(cfg.games);
    parallelFor(cfg.games, threads, [&](int index, int worker) {
        outcomes[index] = playArenaGame(*contexts[worker], cfg, shape.boardSize, seed, index);
    });

    MatchResult result;
    result.games = cfg.games;
    std::set<std::vector<int>> distinct;
    for (const GameOutcome& g : outcomes) {
        result.scoreA += g.scoreA;
        if (g.scoreA == 1.0) ++result.winsA;
        else if (g.scoreA == 0.0) ++result.winsB;
        else ++result.draws;
        distinct.insert(g.moves);
    }
    result.distinctGames = static_cast<int>(distinct.size());
    return result;
}

MatchResult playMatch(const Network& a, const Network& b, const ArenaConfig& cfg, uint64_t seed,
                      int workers, const std::string& backendName) {
    return playMatch(ArenaPlayer::network(a), ArenaPlayer::network(b), cfg, seed, workers,
                     backendName);
}

std::vector<int> ladderOpponents(int generation, int maxOpponents) {
    std::vector<int> out;
    if (generation <= 0) return out;
    // g-1, g-2, g-4, g-8, ...: dense near the present, sparse into the past.
    for (int d = 1; generation - d >= 1 && static_cast<int>(out.size()) < maxOpponents; d *= 2)
        out.push_back(generation - d);
    out.push_back(0);   // the anchor: every rating is relative to generation 0
    return out;
}

int strongestReference(const std::map<int, double>& ratings, const std::vector<int>& references) {
    int best = 0;
    double bestRating = 0;
    bool found = false;
    for (int r : references) {
        auto it = ratings.find(r);
        if (it == ratings.end()) continue;   // unrated: nothing known about it yet
        if (!found || it->second > bestRating || (it->second == bestRating && r > best)) {
            best = r;
            bestRating = it->second;
            found = true;
        }
    }
    if (!found) throw std::invalid_argument("strongestReference: no reference is rated");
    return best;
}

std::vector<int> selectOpponents(int generation, const std::map<int, double>& ratings,
                                 const std::vector<int>& references, int maxExtra,
                                 std::optional<int> bestNetwork) {
    if (generation < 1)
        throw std::invalid_argument("selectOpponents: generation must be >= 1, got " +
                                    std::to_string(generation));
    std::vector<int> out{generation - 1};   // the progress signal
    auto chosen = [&](int p) { return std::find(out.begin(), out.end(), p) != out.end(); };
    const int strongest = strongestReference(ratings, references);
    if (!chosen(strongest)) out.push_back(strongest);   // the promotion check

    // The new network has no rating yet; its predecessor's is the best guess.
    auto prev = ratings.find(generation - 1);
    const double estimate = prev == ratings.end() ? 0.0 : prev->second;
    std::vector<std::pair<double, int>> candidates;   // (|E - 0.7|, player)
    for (int r : references) {
        auto it = ratings.find(r);
        if (it == ratings.end() || chosen(r)) continue;
        candidates.push_back({std::fabs(expectedScore(estimate, it->second) - kTargetScore), r});
    }
    std::sort(candidates.begin(), candidates.end());   // by distance, then by smaller ID
    for (size_t i = 0; i < candidates.size() && static_cast<int>(i) < maxExtra; ++i)
        out.push_back(candidates[i].second);

    if (bestNetwork && !chosen(*bestNetwork)) out.push_back(*bestNetwork);   // gating
    return out;
}

bool passesGate(double score, int games, double threshold) {
    return games > 0 && score / games >= threshold;
}

double matchScore(float resultForBlack, bool aIsBlack) {
    if (resultForBlack == 0.0f) return 0.5;   // exact tie (integer komi)
    bool blackWon = resultForBlack > 0.0f;
    return blackWon == aIsBlack ? 1.0 : 0.0;
}

} // namespace azgo
```

The two lines worth a second look:
- `Seat` builds a `UniformEvaluator` for a bot from `shape`, the `NetConfig` of whichever player has a network, and gives it `botSimulations` instead of `cfg.simulations`.
- `selectOpponents` sorts `(distance, player)` pairs, so equal distances fall back to the smaller ID, as the spec requires.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_arena`
Expected: `100% tests passed out of 1`. All expected scores in `test_select_opponents` can be checked by hand from E = 1 / (1 + 10^((r − R)/400)). For example, R = 800 against 450 gives 1 / (1 + 10^−0.875) = 0.882.

- [ ] **Step 6: Add `--b-bot` to the arena app**

Full file `apps/arena.cpp`:

```cpp
// arena: plays a checkpoint against another checkpoint, or against a uniform
// bot, and prints A's score with a 95% Wilson confidence interval. For manual
// experiments; the trainer runs its own arena every generation.
//
//   arena --a runs/nine/checkpoints/gen_0040.azn --b runs/nine/checkpoints/gen_0020.azn
//   arena --a runs/nine/checkpoints/gen_0040.azn --b-bot 100
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "azgo/arena.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/elo.hpp"
#include "azgo/network.hpp"
#include "azgo/parallel.hpp"
#include "cli.hpp"

namespace {

const char* const kUsage =
    "usage: arena --a CKPT (--b CKPT | --b-bot SIMS) [--games 40] [--simulations 200]\n"
    "             [--komi 7.5] [--temperature-moves 4] [--workers 0] [--backend naive] [--seed 1]\n"
    "\n"
    "Colors alternate: A plays Black in even-numbered games. --workers 0 uses every core.\n"
    "--b-bot SIMS makes B a uniform bot: MCTS with equal priors and value 0, searching\n"
    "SIMS simulations per move (1-100000) on A's board size.\n";

int usageError(const std::string& message) {
    std::fprintf(stderr, "error: %s\n\n%s", message.c_str(), kUsage);
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    std::unique_ptr<ArgParser> args;
    try {
        args = std::make_unique<ArgParser>(
            argc, argv,
            std::vector<std::string>{"a", "b", "b-bot", "games", "simulations", "komi",
                                     "temperature-moves", "workers", "backend", "seed"},
            std::vector<std::string>{"help"});
    } catch (const std::invalid_argument& e) {
        return usageError(e.what());
    }
    if (args->flag("help")) {
        std::fputs(kUsage, stdout);
        return 0;
    }
    if (!args->has("a")) return usageError("--a is required");
    if (args->has("b") == args->has("b-bot"))
        return usageError("give exactly one of --b CKPT and --b-bot SIMS");

    try {
        azgo::ArenaConfig cfg;
        cfg.games = args->getInt("games", 40);
        cfg.simulations = args->getInt("simulations", 200);
        cfg.komi = static_cast<float>(args->getDouble("komi", 7.5));
        cfg.temperatureMoves = args->getInt("temperature-moves", 4);
        const int workers = args->getInt("workers", 0);
        const int seed = args->getInt("seed", 1);
        const std::string backend = args->get("backend", "naive");
        const int botSims = args->getInt("b-bot", 0);
        if (cfg.games < 2 || cfg.games % 2 != 0)
            throw std::invalid_argument("--games must be even and at least 2 (colors are balanced)");
        if (cfg.simulations < 1) throw std::invalid_argument("--simulations must be at least 1");
        if (cfg.temperatureMoves < 0) throw std::invalid_argument("--temperature-moves must be >= 0");
        if (workers < 0) throw std::invalid_argument("--workers must be >= 0");
        if (seed < 0) throw std::invalid_argument("--seed must be >= 0");
        if (args->has("b-bot") && (botSims < 1 || botSims > 100000))
            throw std::invalid_argument("--b-bot must be in [1, 100000]");

        const std::string pathA = args->get("a");
        const azgo::Network a = azgo::loadNetwork(pathA);   // errors name the file
        std::unique_ptr<azgo::Network> b;
        azgo::ArenaPlayer playerB = azgo::ArenaPlayer::bot(botSims);
        std::string nameB = "bot(" + std::to_string(botSims) + " simulations)";
        if (args->has("b")) {
            nameB = args->get("b");
            b = std::make_unique<azgo::Network>(azgo::loadNetwork(nameB));
            playerB = azgo::ArenaPlayer::network(*b);
        }

        const azgo::MatchResult r =
            azgo::playMatch(azgo::ArenaPlayer::network(a), playerB, cfg, static_cast<uint64_t>(seed),
                            azgo::resolveWorkers(workers), backend);
        const std::pair<double, double> ci = azgo::wilsonInterval(r.scoreA, r.games);
        std::printf("A %s vs B %s: A scored %g/%d = %.3f (95%% CI %.3f-%.3f), distinct games %d\n",
                    pathA.c_str(), nameB.c_str(), r.scoreA, r.games, r.scoreA / r.games,
                    ci.first, ci.second, r.distinctGames);
        std::printf("A wins %d, B wins %d, draws %d\n", r.winsA, r.winsB, r.draws);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
```

In `tests/integration_smoke.sh`, replace:

```bash
# train -> resume -> arena -> analyze (good and bad input) -> bench, plus the
# refusal to change the architecture on resume.
```

with:

```bash
# train -> resume -> arena (checkpoint and bot) -> analyze (good and bad input)
# -> bench, plus the refusal to change the architecture on resume.
```

Replace:

```bash
echo "$out" | grep -Eq '^A .* scored' || fail "arena did not print a score line"
```

with:

```bash
echo "$out" | grep -Eq '^A .* scored' || fail "arena did not print a score line"

echo "== arena: against the uniform bot =="
out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b-bot 8 --games 4 --simulations 4 --workers 2)"
echo "$out"
echo "$out" | grep -q 'vs B bot(8 simulations): A scored' || fail "arena --b-bot printed no score line"

# B is exactly one of a checkpoint and a bot, and a bot must search.
expect_arena_error() {
    local want="$1"; shift
    local status=0 out
    out="$("$BUILD/arena" --a "$CK/gen_0003.azn" "$@" 2>&1)" || status=$?
    [ "$status" -eq 1 ] || fail "arena $* exited $status, expected 1; output: $out"
    echo "$out" | grep -qF -- "$want" || fail "arena $* printed '$out', expected '$want'"
    echo "  ok: $want"
}
expect_arena_error "give exactly one of --b CKPT and --b-bot SIMS" --b "$CK/gen_0000.azn" --b-bot 8
expect_arena_error "give exactly one of --b CKPT and --b-bot SIMS"
expect_arena_error "--b-bot must be in [1, 100000]" --b-bot 0
```

- [ ] **Step 7: Run everything, both builds**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: `100% tests passed out of 24`. `bash tests/integration_smoke.sh build .` prints, among the rest:

```
== arena: against the uniform bot ==
A …/run/checkpoints/gen_0003.azn vs B bot(8 simulations): A scored …
  ok: give exactly one of --b CKPT and --b-bot SIMS
  ok: give exactly one of --b CKPT and --b-bot SIMS
  ok: --b-bot must be in [1, 100000]
```

Run the BLAS variant from the Conventions. Expected: `100% tests passed out of 24`.

- [ ] **Step 8: Commit**

```bash
git add include/azgo/arena.hpp src/arena.cpp apps/arena.cpp tests/test_arena.cpp tests/integration_smoke.sh
git commit -F - <<'EOF'
Add uniform bots and rating-based opponent selection to the arena

A bot is MCTS with equal priors and value 0: it knows only the rules, so
its strength is fixed by its simulation count and identical in every
run. playMatch now takes two ArenaPlayers (the two-network form stays as
a wrapper). selectOpponents picks g-1, the strongest reference and the
references whose expected score is closest to 0.7, because a game's
information is proportional to p(1-p) and 20-0 sweeps of generation 0
say almost nothing. arena --b-bot plays a checkpoint against a bot.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 4: Training state version 2 with the reference set

**Files:**
- Modify: `include/azgo/run_state.hpp`, `src/run_state.cpp`, `src/trainer.cpp` (startNew and resume)
- Test: `tests/test_run_state.cpp`, `tests/test_trainer.cpp`

**Interfaces:**
- Consumes: `Config::botSimulations` (Task 2).
- Produces:
  - `TrainingState::references`, a `std::vector<int>` (default `{0}`, sorted ascending, always contains 0).
  - `std::vector<int> azgo::initialReferences(int numBots)` returns `{-numBots, …, -1, 0}` and throws `std::invalid_argument` if `numBots < 0`.
  - `TrainingState azgo::loadTrainingState(const std::string& path, int numBots = 0)`. A version-1 file gets `initialReferences(numBots)`.
  - `saveTrainingState` writes version 2: the version-1 layout, then a `u32` count and one `i32` per reference.
  - `Trainer::startNew` stores `initialReferences(cfg.botSimulations.size())`, and `Trainer::resume` passes that count to the loader.

- [ ] **Step 1: Write the failing tests**

In `tests/test_run_state.cpp`, replace:

```cpp
// A state with real content: two short self-play games, momentum buffers,
// and arena results for two pairs.
```

with:

```cpp
// A state with real content: two short self-play games, momentum buffers,
// arena results for three pairs (one with a bot) and a reference set with
// two bots and a promoted generation.
```

Replace:

```cpp
    s.results.addMatch(6, 7, 20, 9.0);
    return s;
```

with:

```cpp
    s.results.addMatch(6, 7, 20, 9.0);
    s.results.addMatch(-1, 0, 8, 3.0);
    s.references = {-2, -1, 0, 5};
    return s;
```

Replace:

```cpp
    assert(x.results.pairs().size() == y.results.pairs().size());
}
```

with:

```cpp
    assert(x.results.pairs().size() == y.results.pairs().size());
    assert(x.references == y.references);
}
```

Version 2 is now the current version, so the "newer version" probe must use 3. Replace:

```cpp
    newer[4] = 2;   // version is the u32 after the magic, little-endian
```

with:

```cpp
    newer[4] = 3;   // version is the u32 after the magic, little-endian
```

Replace:

```cpp
static void test_paths() {
```

with:

```cpp
static void test_initial_references() {
    assert((initialReferences(0) == std::vector<int>{0}));
    assert((initialReferences(2) == std::vector<int>{-2, -1, 0}));
    assert((TrainingState{}.references == std::vector<int>{0}));
}

// A version-1 file is a version-2 file without the reference set at the end:
// the u32 count and one i32 per reference.
static void test_version1_gets_default_references() {
    fs::path dir = freshDir("version1");
    TrainingState original = makeState();
    saveTrainingState((dir / "v2.bin").string(), original);
    std::string bytes = readBytes(dir / "v2.bin");
    std::string v1 = bytes.substr(0, bytes.size() - 4 - 4 * original.references.size());
    v1[4] = 1;
    fs::path path = dir / "v1.bin";
    writeBytes(path, v1);

    TrainingState loaded = loadTrainingState(path.string(), 2);
    assert((loaded.references == std::vector<int>{-2, -1, 0}));
    TrainingState expected = original;
    expected.references = {-2, -1, 0};
    assertSameState(expected, loaded);
    assert((loadTrainingState(path.string()).references == std::vector<int>{0}));   // no bots

    // Saving it again writes version 2.
    saveTrainingState(path.string(), loaded);
    assert(readBytes(path)[4] == 2);
    assert((loadTrainingState(path.string()).references == std::vector<int>{-2, -1, 0}));
    fs::remove_all(dir);
}

static void test_rejects_bad_references() {
    fs::path dir = freshDir("badrefs");
    fs::path path = dir / "state.bin";
    for (const std::vector<int>& bad : {std::vector<int>{0, -1}, std::vector<int>{-1, 5},
                                        std::vector<int>{0, 0}, std::vector<int>{}}) {
        TrainingState s = makeState();
        s.references = bad;
        saveTrainingState(path.string(), s);
        std::string msg = loadError(path.string());
        assert(msg.find(path.string()) != std::string::npos);
        assert(msg.find("reference") != std::string::npos);
    }
    fs::remove_all(dir);
}

static void test_paths() {
```

Replace:

```cpp
    test_crash_leftovers();
```

with:

```cpp
    test_crash_leftovers();
    test_initial_references();
    test_version1_gets_default_references();
    test_rejects_bad_references();
```

In `tests/test_trainer.cpp`, give the tiny run one cheap bot. Replace:

```cpp
    c.lrSchedule = "0:0.02";
    return c;
```

with:

```cpp
    c.lrSchedule = "0:0.02";
    c.botSimulations = {8};   // one cheap bot: player -1
    c.calibrationGames = 2;
    return c;
```

In `test_start_new_writes_run_files`, replace:

```cpp
    assert(s.buffer.positions() == 0 && s.buffer.capacity() == 1000);
```

with:

```cpp
    assert(s.buffer.positions() == 0 && s.buffer.capacity() == 1000);
    assert((s.references == std::vector<int>{-1, 0}));   // the bot and generation 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cmake --build build -j`
Expected: the build fails in `tests/test_run_state.cpp` with errors such as `no member named 'references' in 'azgo::TrainingState'` and `use of undeclared identifier 'initialReferences'`.

- [ ] **Step 3: Extend the state header**

In `include/azgo/run_state.hpp`, replace:

```cpp
// generation counters and the arena results. They go into one binary file
// that is replaced atomically, so a crash mid-save leaves the old state.
```

with:

```cpp
// generation counters, the arena results and the arena's reference set. They
// go into one binary file that is replaced atomically, so a crash mid-save
// leaves the old state.
```

Replace:

```cpp
    ResultsTable results;
};

// Binary, magic "AZST" (u32 0x54535A41), version u32 = 1. Atomic replace.
void saveTrainingState(const std::string& path, const TrainingState& state);
TrainingState loadTrainingState(const std::string& path);   // throws std::runtime_error
```

with:

```cpp
    ResultsTable results;
    // The frozen players new networks are measured against: bots, generation
    // 0 and every promoted generation. Sorted ascending, always contains 0.
    std::vector<int> references{0};
};

// The reference set of a new run: every bot (-numBots, ..., -1) and generation 0.
std::vector<int> initialReferences(int numBots);

// Binary, magic "AZST" (u32 0x54535A41), version u32 = 2. Atomic replace.
void saveTrainingState(const std::string& path, const TrainingState& state);
// Reads version 2 and version 1, which predates the reference set: a version-1
// file gets references = initialReferences(numBots), so pass the number of
// bots the run's config defines. Throws std::runtime_error naming the path.
TrainingState loadTrainingState(const std::string& path, int numBots = 0);
```

- [ ] **Step 4: Write and read version 2**

In `src/run_state.cpp`, replace:

```cpp
#include "azgo/run_state.hpp"
#include <cstdio>
```

with:

```cpp
#include "azgo/run_state.hpp"
#include <algorithm>
#include <cstdio>
```

Replace:

```cpp
#include <sstream>
#include <stdexcept>
```

with:

```cpp
#include <sstream>
#include <stdexcept>
#include <vector>
```

Replace:

```cpp
constexpr uint32_t kStateVersion = 1;
} // namespace
```

with:

```cpp
constexpr uint32_t kStateVersion = 2;   // 2 added the reference set
} // namespace

std::vector<int> initialReferences(int numBots) {
    if (numBots < 0) throw std::invalid_argument("initialReferences: numBots must be >= 0");
    std::vector<int> refs;
    for (int k = numBots; k >= 1; --k) refs.push_back(-k);   // ascending: -numBots first
    refs.push_back(0);
    return refs;
}
```

Replace:

```cpp
    w.str(state.results.toCsv());
    writeFileAtomically(path, os.str());
```

with:

```cpp
    w.str(state.results.toCsv());
    w.u32(static_cast<uint32_t>(state.references.size()));
    for (int r : state.references) w.i32(r);
    writeFileAtomically(path, os.str());
```

Replace:

```cpp
TrainingState loadTrainingState(const std::string& path) {
```

with:

```cpp
TrainingState loadTrainingState(const std::string& path, int numBots) {
```

Replace:

```cpp
        if (version != kStateVersion)
            throw std::runtime_error("unsupported version " + std::to_string(version));
```

with:

```cpp
        if (version != 1 && version != kStateVersion)
            throw std::runtime_error("unsupported version " + std::to_string(version));
```

Replace:

```cpp
        state.results = ResultsTable::fromCsv(r.str());
        if (is.peek() != std::char_traits<char>::eof())
```

with:

```cpp
        state.results = ResultsTable::fromCsv(r.str());
        if (version == 1) {
            state.references = initialReferences(numBots);
        } else {
            uint32_t count = r.u32();
            if (count > bytes.size() / sizeof(int32_t)) throw std::runtime_error("truncated reference set");
            state.references.resize(count);
            for (int& ref : state.references) ref = r.i32();
            // Sorted, unique and anchored, or the file is not what we wrote.
            bool valid = std::find(state.references.begin(), state.references.end(), 0) !=
                         state.references.end();
            for (size_t i = 1; valid && i < state.references.size(); ++i)
                valid = state.references[i - 1] < state.references[i];
            if (!valid) throw std::runtime_error("invalid reference set (must be ascending and contain 0)");
        }
        if (is.peek() != std::char_traits<char>::eof())
```

- [ ] **Step 5: Give new and resumed runs their references**

In `src/trainer.cpp` (`Trainer::startNew`), replace:

```cpp
    state.velocity = SgdMomentum(net, cfg.momentum, cfg.weightDecay).velocity();   // all zeros
```

with:

```cpp
    state.velocity = SgdMomentum(net, cfg.momentum, cfg.weightDecay).velocity();   // all zeros
    state.references = initialReferences(static_cast<int>(cfg.botSimulations.size()));
```

In `Trainer::resume`, replace:

```cpp
    TrainingState state = loadTrainingState(statePath(runDir));
```

with:

```cpp
    // A version-1 state.bin (written before bots existed) gets every
    // configured bot as a reference.
    TrainingState state =
        loadTrainingState(statePath(runDir), static_cast<int>(cfg.botSimulations.size()));
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: `100% tests passed out of 24`. The references are stored but not used yet: the arena still plays the ladder.

- [ ] **Step 7: Commit**

```bash
git add include/azgo/run_state.hpp src/run_state.cpp src/trainer.cpp tests/test_run_state.cpp tests/test_trainer.cpp
git commit -F - <<'EOF'
Store the arena's reference set in state.bin (format version 2)

The reference set grows by promotion and must survive a resume exactly,
so it lives in the training state. Version-1 files from runs made before
this change still load: their reference set is generation 0 plus every
configured bot, which the first arena round then calibrates.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 5: The shared arena round, calibration, promotion and references.csv

**Files:**
- Create: `include/azgo/arena_round.hpp`, `src/arena_round.cpp`, `tests/test_arena_round.cpp`
- Modify: `include/azgo/run_state.hpp`, `include/azgo/trainer.hpp` (seed phases move), `src/trainer.cpp`, `include/azgo/arena.hpp`, `src/arena.cpp`, `tests/test_arena.cpp` (remove `ladderOpponents`), `tests/test_trainer.cpp`, `tests/integration_smoke.sh`

**Interfaces:**
- Consumes:
  - from Task 3: `ArenaPlayer`, `playMatch(const ArenaPlayer&, …)`, `strongestReference` and `selectOpponents`;
  - from Task 4: `TrainingState::references` and `initialReferences`;
  - from Task 2: `Config::botSimulations`, `calibrationGames`, `promoteThreshold` and `arenaOpponents`.
- Produces:
  - `using azgo::MatchFn = std::function<MatchResult(int a, int b, int games, uint64_t seed)>`.
  - `struct ArenaRoundSettings { uint64_t seed; int arenaGames; int maxExtra; int calibrationGames; double promoteThreshold; std::optional<int> bestNetwork; }`.
  - `struct ArenaMatch { int opponent; MatchResult result; }` and `struct ArenaRoundResult { std::vector<ArenaMatch> matches; int strongest; bool promoted; }`.
  - `int calibrateBots(ResultsTable&, const std::vector<int>& references, const ArenaRoundSettings&, const MatchFn&)`.
  - `ArenaRoundResult runArenaRound(int generation, ResultsTable&, std::vector<int>& references, const ArenaRoundSettings&, const MatchFn&)`.
  - `std::string opponentSummary(const ArenaRoundResult&, const std::vector<int>& botSimulations)` and `std::string referencesCsv(const ResultsTable&, const std::vector<int>& references, const std::vector<int>& botSimulations)`.
  - Seed phase constants, now in `run_state.hpp`: `kPhaseInit = 0, kPhaseSelfPlay = 1, kPhaseTrain = 2, kPhaseArena = 3, kPhaseCalibration = 4`.
  - Private helpers in `src/trainer.cpp` that Task 6 reuses: `writeRatingFiles(dir, results, references, botSimulations)`, `arenaConfig(cfg)`, `roundSettings(cfg, bestNetwork)` and `runMatches(cfg, runDir, liveGeneration, liveNet)`.
  - Run files: `references.csv` next to `results.csv`. `elo.csv` lists generations only.

The round function takes the matches as a callback. The trainer passes real networks; the tests pass a synthetic arena in which every match scores exactly its Bradley-Terry expectation, so selection, calibration and promotion can be checked by hand.

- [ ] **Step 1: Write the failing round tests**

Create `tests/test_arena_round.cpp`:

```cpp
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include "azgo/arena_round.hpp"
#include "azgo/rng.hpp"
#include "azgo/run_state.hpp"

using namespace azgo;

// A synthetic arena: players have fixed true Elo ratings and every match
// scores exactly its Bradley-Terry expectation (a fractional score is fine
// for ResultsTable). Each call is recorded so tests can check who played
// whom, how many games and with which seed.
struct FakeArena {
    std::map<int, double> truth;
    std::vector<std::tuple<int, int, int, uint64_t>> calls;   // (a, b, games, seed)
    MatchFn fn() {
        return [this](int a, int b, int games, uint64_t seed) {
            calls.emplace_back(a, b, games, seed);
            MatchResult r;
            r.games = games;
            r.scoreA = games / (1.0 + std::pow(10.0, (truth.at(b) - truth.at(a)) / 400.0));
            r.distinctGames = games;
            return r;
        };
    }
};

// A match whose score is fixed in advance, e.g. 17 of 20, whoever plays.
static MatchFn fixedScore(double fraction, std::vector<std::tuple<int, int, int>>* calls) {
    return [fraction, calls](int a, int b, int games, uint64_t) {
        calls->emplace_back(a, b, games);
        MatchResult r;
        r.games = games;
        r.scoreA = fraction * games;
        return r;
    };
}

static ArenaRoundSettings settings() {
    ArenaRoundSettings s;
    s.seed = 7;
    s.arenaGames = 20;
    s.maxExtra = 1;
    s.calibrationGames = 80;
    s.promoteThreshold = 0.85f;   // a float, as Config stores it
    return s;
}

static int gamesBetween(const ResultsTable& t, int a, int b) {
    for (const ResultsTable::Pair& p : t.pairs())
        if (p.a == std::min(a, b) && p.b == std::max(a, b)) return p.games;
    return 0;
}

static void test_calibration() {
    FakeArena arena;
    arena.truth = {{-2, -200.0}, {-1, 50.0}, {0, 0.0}};
    ResultsTable results;
    const std::vector<int> refs{-2, -1, 0};
    assert(calibrateBots(results, refs, settings(), arena.fn()) == 2);
    assert(gamesBetween(results, -2, 0) == 80 && gamesBetween(results, -1, 0) == 80);
    // Generation 0 is player a, ascending bot order, the calibration seed.
    assert(arena.calls.size() == 2);
    assert(arena.calls[0] == std::make_tuple(0, -2, 80, deriveSeed(7, 0, kPhaseCalibration,
                                                                   static_cast<uint64_t>(-2))));
    assert(std::get<1>(arena.calls[1]) == -1);
    // Already calibrated: nothing more to play.
    assert(calibrateBots(results, refs, settings(), arena.fn()) == 0);
    assert(arena.calls.size() == 2);
    // No bots, nothing to do.
    ResultsTable empty;
    assert(calibrateBots(empty, {0}, settings(), arena.fn()) == 0 && empty.empty());
}

static void test_round_uses_selection_and_live_seeds() {
    FakeArena arena;
    arena.truth = {{-1, 50.0}, {0, 0.0}, {1, 100.0}};
    ResultsTable results;
    std::vector<int> refs{-1, 0};
    // The bot is uncalibrated, so the round calibrates it first.
    ArenaRoundResult round = runArenaRound(1, results, refs, settings(), arena.fn());
    assert(arena.calls.size() == 3);
    assert(std::get<0>(arena.calls[0]) == 0 && std::get<1>(arena.calls[0]) == -1);
    // Then g-1 = 0, then the strongest reference: the bot (rated about +50).
    assert(round.strongest == -1 && !round.promoted);
    assert(round.matches.size() == 2 && round.matches[0].opponent == 0 && round.matches[1].opponent == -1);
    assert(arena.calls[1] == std::make_tuple(1, 0, 20, deriveSeed(7, 1, kPhaseArena, 0)));
    assert(arena.calls[2] == std::make_tuple(1, -1, 20, deriveSeed(7, 1, kPhaseArena,
                                                                   static_cast<uint64_t>(-1))));
    assert(gamesBetween(results, 0, 1) == 20 && gamesBetween(results, -1, 1) == 20);
    assert((refs == std::vector<int>{-1, 0}));
}

// Review Focus 3: a score exactly at the threshold.
static void test_promotion() {
    // 17 of 20 against the strongest reference reaches 0.85 exactly: promoted,
    // with exactly calibration_games more games against it.
    std::vector<std::tuple<int, int, int>> calls;
    ResultsTable results;
    std::vector<int> refs{0};
    ArenaRoundResult round = runArenaRound(1, results, refs, settings(), fixedScore(0.85, &calls));
    assert(round.promoted && round.strongest == 0);
    assert((refs == std::vector<int>{0, 1}));
    assert(calls.size() == 2 && calls[1] == std::make_tuple(1, 0, 80));
    assert(gamesBetween(results, 0, 1) == 100);

    // 16 of 20 is below the threshold: no extra games, no promotion.
    calls.clear();
    ResultsTable below;
    std::vector<int> refs2{0};
    round = runArenaRound(1, below, refs2, settings(), fixedScore(0.8, &calls));
    assert(!round.promoted && (refs2 == std::vector<int>{0}));
    assert(calls.size() == 1 && gamesBetween(below, 0, 1) == 20);

    // A promoted generation is inserted in order, after the bots.
    calls.clear();
    ResultsTable t;
    t.addMatch(0, -1, 80, 40.0);
    std::vector<int> refs3{-1, 0};
    round = runArenaRound(3, t, refs3, settings(), fixedScore(1.0, &calls));
    assert(round.promoted && (refs3 == std::vector<int>{-1, 0, 3}));
}

// Generation g is truly 100 g Elo strong, the bot 50. A generation 400 above
// the strongest reference scores E = 0.909 >= 0.85 and is promoted; 300 above
// gives E = 0.849, just short. So generations 4, 8, 12 and 16 are promoted,
// and once 4 is a reference, generation 0 (expected score > 0.9) loses its
// place as the one extra opponent to a closer match.
static void test_saturated_anchor_is_dropped() {
    FakeArena arena;
    arena.truth = {{-1, 50.0}};
    for (int g = 0; g <= 16; ++g) arena.truth[g] = 100.0 * g;
    ResultsTable results;
    std::vector<int> refs{-1, 0};
    calibrateBots(results, refs, settings(), arena.fn());
    std::vector<int> promoted;
    for (int g = 1; g <= 16; ++g) {
        ArenaRoundResult round = runArenaRound(g, results, refs, settings(), arena.fn());
        std::vector<int> opponents;
        for (const ArenaMatch& m : round.matches) opponents.push_back(m.opponent);
        if (round.promoted) promoted.push_back(g);
        if (g == 1) assert((opponents == std::vector<int>{0, -1}));
        if (g == 4) assert((opponents == std::vector<int>{3, -1, 0}));
        if (g == 5) assert((opponents == std::vector<int>{4, -1}));
        if (g == 9) assert((opponents == std::vector<int>{8, 4}));
        if (g >= 5) assert(std::find(opponents.begin(), opponents.end(), 0) == opponents.end());
    }
    assert((promoted == std::vector<int>{4, 8, 12, 16}));
    assert((refs == std::vector<int>{-1, 0, 4, 8, 12, 16}));
}

static void test_gating_appends_best() {
    FakeArena arena;
    arena.truth = {{0, 0.0}, {1, 100.0}, {2, 200.0}, {3, 300.0}};
    ResultsTable results;
    std::vector<int> refs{0};
    ArenaRoundSettings s = settings();
    s.maxExtra = 0;
    s.bestNetwork = 0;   // already g-1 and the strongest reference: played once
    ArenaRoundResult round = runArenaRound(1, results, refs, s, arena.fn());
    assert(round.matches.size() == 1 && gamesBetween(results, 0, 1) == 20);
    s.bestNetwork = 1;   // generation 3 plays 2 (g-1), 0 (strongest), then 1 (best)
    round = runArenaRound(3, results, refs, s, arena.fn());
    assert(round.matches.size() == 3 && round.matches[2].opponent == 1);
}

static void test_summary_and_references_csv() {
    ArenaRoundResult round;
    MatchResult r;
    r.games = 20;
    r.scoreA = 11;
    round.matches.push_back({149, r});
    r.scoreA = 20;
    round.matches.push_back({-1, r});
    assert(opponentSummary(round, {100}) == "vs 149:0.55 bot100:1.00");

    ResultsTable t;
    t.addMatch(0, -1, 10, 5.0);   // the bot draws level with generation 0: both 0 Elo
    const std::string csv = referencesCsv(t, {-1, 0, 5}, {100});
    assert(csv == "player,kind,simulations,promoted_at,elo\n"
                  "-1,bot,100,0,0.0\n"
                  "0,checkpoint,,0,0.0\n"
                  "5,checkpoint,,5,\n");   // 5 has no results yet: unrated
}

int main() {
    test_calibration();
    test_round_uses_selection_and_live_seeds();
    test_promotion();
    test_saturated_anchor_is_dropped();
    test_gating_appends_best();
    test_summary_and_references_csv();
    std::printf("test_arena_round: all tests passed\n");
    return 0;
}
```

The numbers in `test_saturated_anchor_is_dropped` follow from the truth table. A gap of 400 Elo gives E = 1/(1 + 10^−1) = 0.909 ≥ 0.85, and a gap of 300 gives E = 1/(1 + 10^−0.75) = 0.849 < 0.85. So the strongest reference moves to 4, 8, 12 and 16. From generation 5 on, the one extra place goes to a reference whose E is closer to 0.7 than generation 0's.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build build -j`
Expected: the build fails with `fatal error: 'azgo/arena_round.hpp' file not found`.

- [ ] **Step 3: Move the seed phases next to the run state**

`arena_round` needs `kPhaseArena` and the new `kPhaseCalibration`, but must not include the trainer. In `include/azgo/trainer.hpp`, delete:

```cpp

// Seed phases for deriveSeed(cfg.seed, generation, phase, index).
constexpr uint64_t kPhaseInit = 0, kPhaseSelfPlay = 1, kPhaseTrain = 2, kPhaseArena = 3;
```

In `include/azgo/run_state.hpp`, replace:

```cpp
#pragma once
#include <string>
```

with:

```cpp
#pragma once
#include <cstdint>
#include <string>
```

Replace:

```cpp
// Run-directory paths.
```

with:

```cpp
// Seed phases for deriveSeed(cfg.seed, generation, phase, index). Part of a
// run's reproducibility contract: changing one changes every later game.
// kPhaseCalibration seeds bot calibration (generation 0) and promotion games.
constexpr uint64_t kPhaseInit = 0, kPhaseSelfPlay = 1, kPhaseTrain = 2, kPhaseArena = 3,
                   kPhaseCalibration = 4;

// Run-directory paths.
```

(`trainer.hpp` includes `run_state.hpp`, so the trainer still sees all four old constants.)

- [ ] **Step 4: Write the round header**

Create `include/azgo/arena_round.hpp`:

```cpp
// One arena round, shared by live training and by re-rating a finished run,
// so both make the same choices from the same results. A round calibrates
// any bot that has never played generation 0, refits every rating, picks the
// opponents whose results will say the most about the new network, plays
// them, and promotes the new network into the reference set when it clearly
// beats the strongest reference. Every choice depends only on the results
// table, the reference set and the run's seed, so an uninterrupted run, a
// resumed run and a re-rated run play the same games.
#pragma once
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>
#include "azgo/arena.hpp"
#include "azgo/elo.hpp"

namespace azgo {

// Plays `games` games between players a and b with `seed` and returns the
// result from a's side. Players are generations (>= 0) and bots (-1, -2, ...).
// The trainer plays real networks; tests can return made-up results.
using MatchFn = std::function<MatchResult(int a, int b, int games, uint64_t seed)>;

struct ArenaRoundSettings {
    uint64_t seed = 1;                // the run's seed
    int arenaGames = 20;              // games per opponent
    int maxExtra = 2;                 // arena_opponents
    int calibrationGames = 80;        // per bot calibration and per promotion
    double promoteThreshold = 0.85;
    std::optional<int> bestNetwork;   // gating: the current best generation
};

struct ArenaMatch {
    int opponent;
    MatchResult result;               // from the new network's side
};

struct ArenaRoundResult {
    std::vector<ArenaMatch> matches;  // in selectOpponents order: g-1 first
    int strongest = 0;                // the strongest reference before the round
    bool promoted = false;            // the new network joined the references
};

// Plays settings.calibrationGames games of generation 0 against every bot in
// `references` that has never played generation 0, so each bot is rated
// before it can be chosen. Bot b's seed is deriveSeed(seed, 0,
// kPhaseCalibration, b). Returns the number of bots calibrated.
int calibrateBots(ResultsTable& results, const std::vector<int>& references,
                  const ArenaRoundSettings& settings, const MatchFn& play);

// The arena round of generation g >= 1:
//   1. calibrateBots;
//   2. refit: ratings = fitBradleyTerry(results, 0);
//   3. opponents = selectOpponents(g, ratings, references, maxExtra, bestNetwork);
//   4. arenaGames against each, seed deriveSeed(seed, g, kPhaseArena, opponent);
//   5. promotion: if g scored >= promoteThreshold against the strongest
//      reference, calibrationGames more against it (seed deriveSeed(seed, g,
//      kPhaseCalibration, strongest)), and g joins `references` (kept sorted).
// Every result is added to `results`.
ArenaRoundResult runArenaRound(int generation, ResultsTable& results, std::vector<int>& references,
                               const ArenaRoundSettings& settings, const MatchFn& play);

// "vs 149:0.55 120:0.70 bot100:1.00": each opponent with the new network's
// score. Bot b is named after its simulation count, botSimulations[-b - 1].
std::string opponentSummary(const ArenaRoundResult& round, const std::vector<int>& botSimulations);

// references.csv: header "player,kind,simulations,promoted_at,elo", one row
// per reference in ascending order. kind is bot or checkpoint; simulations is
// a bot's count (empty for a checkpoint); promoted_at is the generation that
// joined (0 for the initial set); elo is from fitBradleyTerry(results, 0),
// empty while unrated.
std::string referencesCsv(const ResultsTable& results, const std::vector<int>& references,
                          const std::vector<int>& botSimulations);

} // namespace azgo
```

- [ ] **Step 5: Write the round implementation**

Create `src/arena_round.cpp`:

```cpp
#include "azgo/arena_round.hpp"
#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>
#include "azgo/rng.hpp"
#include "azgo/run_state.hpp"

namespace azgo {

namespace {

// promote_threshold is a float config key, and 0.85f is 0.8500000238...: a
// plain comparison would refuse 17 of 20. Config values carry about seven
// significant digits, so a 1e-6 margin accepts exactly what the user wrote.
bool reaches(double score, int games, double threshold) {
    return score >= (threshold - 1e-6) * games;
}

std::string botName(int player, const std::vector<int>& botSimulations) {
    if (player >= 0 || static_cast<size_t>(-player) > botSimulations.size())
        throw std::invalid_argument("no bot " + std::to_string(player) + " in bot_simulations");
    return "bot" + std::to_string(botSimulations[static_cast<size_t>(-player - 1)]);
}

} // namespace

int calibrateBots(ResultsTable& results, const std::vector<int>& references,
                  const ArenaRoundSettings& settings, const MatchFn& play) {
    std::set<int> calibrated;   // bots that already have results against generation 0
    for (const ResultsTable::Pair& p : results.pairs())
        if (p.a < 0 && p.b == 0) calibrated.insert(p.a);
    int count = 0;
    for (int bot : references) {
        if (bot >= 0 || calibrated.count(bot)) continue;
        const MatchResult r = play(0, bot, settings.calibrationGames,
                                   deriveSeed(settings.seed, 0, kPhaseCalibration,
                                              static_cast<uint64_t>(bot)));
        results.addMatch(0, bot, r.games, r.scoreA);
        ++count;
    }
    return count;
}

ArenaRoundResult runArenaRound(int generation, ResultsTable& results, std::vector<int>& references,
                               const ArenaRoundSettings& settings, const MatchFn& play) {
    // Normally a no-op: bots are calibrated when a run starts. A run resumed
    // from a state.bin written before bots existed calibrates here, once.
    calibrateBots(results, references, settings, play);

    // The refit of every result so far: identical on resume and on re-rating.
    const std::map<int, double> ratings = fitBradleyTerry(results, 0);
    ArenaRoundResult round;
    round.strongest = strongestReference(ratings, references);
    const std::vector<int> opponents =
        selectOpponents(generation, ratings, references, settings.maxExtra, settings.bestNetwork);

    double scoreVsStrongest = 0;
    for (int o : opponents) {
        const MatchResult r = play(generation, o, settings.arenaGames,
                                   deriveSeed(settings.seed, static_cast<uint64_t>(generation),
                                              kPhaseArena, static_cast<uint64_t>(o)));
        results.addMatch(generation, o, r.games, r.scoreA);
        if (o == round.strongest) scoreVsStrongest = r.scoreA;
        round.matches.push_back({o, r});
    }

    // Promotion: a clear win against the strongest reference makes the new
    // network the next yardstick. The extra games pin its rating down before
    // later generations are measured against it.
    if (reaches(scoreVsStrongest, settings.arenaGames, settings.promoteThreshold)) {
        const MatchResult r = play(generation, round.strongest, settings.calibrationGames,
                                   deriveSeed(settings.seed, static_cast<uint64_t>(generation),
                                              kPhaseCalibration,
                                              static_cast<uint64_t>(round.strongest)));
        results.addMatch(generation, round.strongest, r.games, r.scoreA);
        references.insert(std::upper_bound(references.begin(), references.end(), generation),
                          generation);
        round.promoted = true;
    }
    return round;
}

std::string opponentSummary(const ArenaRoundResult& round, const std::vector<int>& botSimulations) {
    std::string out = "vs";
    for (const ArenaMatch& m : round.matches) {
        const std::string name =
            m.opponent < 0 ? botName(m.opponent, botSimulations) : std::to_string(m.opponent);
        char score[32];
        std::snprintf(score, sizeof score, ":%.2f", m.result.scoreA / m.result.games);
        out += " " + name + score;
    }
    return out;
}

std::string referencesCsv(const ResultsTable& results, const std::vector<int>& references,
                          const std::vector<int>& botSimulations) {
    const std::map<int, double> ratings = fitBradleyTerry(results, 0);
    std::string out = "player,kind,simulations,promoted_at,elo\n";
    for (int r : references) {
        std::string simulations;
        if (r < 0) simulations = botName(r, botSimulations).substr(3);   // "bot100" -> "100"
        char elo[32] = "";
        auto it = ratings.find(r);
        if (it != ratings.end()) std::snprintf(elo, sizeof elo, "%.1f", it->second + 0.0);   // -0.0 -> 0.0
        // A generation joins right after its own arena round; the initial set at 0.
        out += std::to_string(r) + (r < 0 ? ",bot," : ",checkpoint,") + simulations + "," +
               std::to_string(r > 0 ? r : 0) + "," + elo + "\n";
    }
    return out;
}

} // namespace azgo
```

- [ ] **Step 6: Run the round tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_arena_round`
Expected: `100% tests passed out of 1`.

- [ ] **Step 7: Write the failing trainer tests**

In `tests/test_trainer.cpp`, add a file-writing helper. Replace:

```cpp
static int countLines(const std::string& s) {
```

with:

```cpp
static void writeFile(const std::string& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << bytes;
    assert(out && "file should be writable");
}

static int countLines(const std::string& s) {
```

In `test_start_new_writes_run_files`, replace:

```cpp
    assert((s.references == std::vector<int>{-1, 0}));   // the bot and generation 0
```

with:

```cpp
    assert((s.references == std::vector<int>{-1, 0}));   // the bot and generation 0
    // The bot played calibration_games = 2 games against generation 0.
    const std::vector<ResultsTable::Pair> calibration = s.results.pairs();
    assert(calibration.size() == 1);
    assert(calibration[0].a == -1 && calibration[0].b == 0 && calibration[0].games == 2);
    assert(readFile(dir.file("results.csv")) == s.results.toCsv());
    const std::vector<std::string> refs = splitLines(readFile(dir.file("references.csv")));
    assert(refs.size() == 3 && refs[0] == "player,kind,simulations,promoted_at,elo");
    assert(refs[1].rfind("-1,bot,8,0,", 0) == 0 && refs[1].size() > 11);   // rated
    assert(refs[2] == "0,checkpoint,,0,0.0");
```

and replace:

```cpp
           "generation,elo,winrate_vs_prev,wr_low,wr_high\n0,0,,,\n");
```

with:

```cpp
           "generation,elo,winrate_vs_prev,wr_low,wr_high\n0,0.0,,,\n");
```

In `test_two_generations`, replace:

```cpp
    // Generation 1 plays 0; generation 2 plays its ladder {1} plus the anchor 0.
    std::map<std::pair<int, int>, ResultsTable::Pair> pairs = readResults(dir);
    assert(pairs.size() == 3);
    for (std::pair<int, int> key : {std::make_pair(0, 1), std::make_pair(0, 2), std::make_pair(1, 2)}) {
        assert(pairs.count(key));
        assert(pairs[key].games == 2);
    }

    TrainingState s = loadTrainingState(statePath(dir.str()));
```

with:

```cpp
    // References start as {bot -1, 0}, and arena_opponents = 2 leaves room for
    // all of them: generation 1 plays 0 and the bot, generation 2 plays 1, 0
    // and the bot. A promoted generation played calibration_games = 2 more
    // games against the strongest reference, so exactly one of its pairs has 4.
    TrainingState s = loadTrainingState(statePath(dir.str()));
    std::map<std::pair<int, int>, ResultsTable::Pair> pairs = readResults(dir);
    assert(pairs.size() == 6);
    assert(pairs.at({-1, 0}).games == 2);   // calibration
    for (std::pair<int, int> key : {std::make_pair(0, 1), std::make_pair(-1, 1), std::make_pair(0, 2),
                                    std::make_pair(1, 2), std::make_pair(-1, 2)}) {
        assert(pairs.count(key));
        assert(pairs[key].games == 2 || pairs[key].games == 4);
    }
    for (int g = 1; g <= 2; ++g) {
        int extended = 0;
        for (const auto& [key, p] : pairs)
            if (key.second == g && p.games == 4) ++extended;
        const bool promoted = std::find(s.references.begin(), s.references.end(), g) != s.references.end();
        assert(extended == (promoted ? 1 : 0));
    }
```

In `test_resume_is_byte_identical`, replace:

```cpp
    for (const char* f : {"checkpoints/gen_0004.azn", "state.bin", "elo.csv", "results.csv",
                          "sgf/gen_0004_game_00.sgf"})
```

with:

```cpp
    for (const char* f : {"checkpoints/gen_0004.azn", "state.bin", "elo.csv", "results.csv",
                          "references.csv", "sgf/gen_0004_game_00.sgf"})
```

In `test_thread_count_independence`, replace:

```cpp
    for (const char* f : {"checkpoints/gen_0002.azn", "state.bin", "results.csv"})
```

with:

```cpp
    for (const char* f : {"checkpoints/gen_0002.azn", "state.bin", "results.csv", "references.csv"})
```

In `test_gating_state_is_consistent`, promotion games now share the `(best, g)` pair with the gate match. Replace:

```cpp
    c.gating = true;
    c.gatingThreshold = 1.0f;
    startQuiet(c, dir)->runUntil(3);

    std::map<std::pair<int, int>, ResultsTable::Pair> pairs = readResults(dir);
    int best = 0;
    for (int g = 1; g <= 3; ++g) {
        // The current best is always among g's opponents.
        auto it = pairs.find({best, g});
        assert(it != pairs.end());
        // scoreA is the older network's score: 0 means g won every game.
        if (it->second.scoreA == 0.0) best = g;
    }
```

with:

```cpp
    c.gating = true;
    c.gatingThreshold = 1.0f;
    c.promoteThreshold = 1.0f;
    startQuiet(c, dir)->runUntil(3);

    std::map<std::pair<int, int>, ResultsTable::Pair> pairs = readResults(dir);
    int best = 0;
    for (int g = 1; g <= 3; ++g) {
        // The current best is always among g's opponents.
        auto it = pairs.find({best, g});
        assert(it != pairs.end());
        // scoreA is the older network's score: 0 means g won every game. More
        // than arena_games games means promotion games were added, which with
        // promote_threshold 1 needs a clean sweep of the arena games, so g
        // passed the gate too.
        if (it->second.scoreA == 0.0 || it->second.games > c.arenaGames) best = g;
    }
```

Replace:

```cpp
// A blown-up learning rate must abort
```

with:

```cpp
// The progress line names every opponent with the new network's score.
static void test_progress_line_lists_opponents() {
    TempDir dir("progress");
    std::unique_ptr<Trainer> t = startQuiet(tinyConfig(), dir);
    std::ostringstream out;
    t->setLog(&out);
    t->runGeneration();
    const std::string line = out.str();
    assert(line.rfind("gen 1 | ", 0) == 0);
    assert(line.find("| arena elo ") != std::string::npos);
    assert(line.find(" vs 0:") != std::string::npos);      // g-1 first
    assert(line.find(" bot8:") != std::string::npos);      // bots by simulation count
    assert(line.find(" (distinct ") != std::string::npos);
}

// Review Focus 1: a run started before rating-aware opponents. Its state.bin
// is version 1 (no reference set) and its config.cfg has none of the new
// keys, so the defaults apply on resume: one bot at 100 simulations, which
// is calibrated with 80 games at the first arena and then played like any
// reference.
static void test_resume_from_version1_state() {
    TempDir dir("version1");
    Config c = tinyConfig();
    c.botSimulations = {};   // the old run had no bots
    startQuiet(c, dir)->runUntil(1);

    // state.bin as version 1: without the reference set {0} (u32 count + one i32).
    const std::string bytes = readFile(statePath(dir.str()));
    std::string v1 = bytes.substr(0, bytes.size() - 8);
    v1[4] = 1;
    writeFile(statePath(dir.str()), v1);
    // config.cfg without the keys the old code did not know.
    std::string oldConfig;
    for (const std::string& line : splitLines(readFile(configPath(dir.str()))))
        if (line.rfind("bot_simulations", 0) != 0 && line.rfind("calibration_games", 0) != 0 &&
            line.rfind("promote_threshold", 0) != 0)
            oldConfig += line + "\n";
    writeFile(configPath(dir.str()), oldConfig);

    std::unique_ptr<Trainer> t = resumeQuiet(dir, {});
    assert((t->config().botSimulations == std::vector<int>{100}));
    t->runUntil(2);
    std::map<std::pair<int, int>, ResultsTable::Pair> pairs = readResults(dir);
    assert(pairs.at({-1, 0}).games == 80);   // calibrated once, at generation 2
    assert(pairs.count({-1, 2}));            // and then chosen
    assert(!pairs.count({-1, 1}));           // generation 1 ran before the bot existed
    TrainingState s = loadTrainingState(statePath(dir.str()));
    assert(s.references.front() == -1);
    assert(readFile(statePath(dir.str()))[4] == 2);   // saved as version 2
    assert(readFile(dir.file("references.csv")).find("\n-1,bot,100,0,") != std::string::npos);
}

// Review Focus 4: no bots and no extra opponents. Each generation plays only
// g-1 and the strongest reference (the same player while generation 0 is the
// whole reference set), and nothing negative appears anywhere.
static void test_no_bots_no_extras() {
    TempDir dir("nobots");
    Config c = tinyConfig();
    c.botSimulations = {};
    c.arenaOpponents = 0;
    startQuiet(c, dir)->runUntil(3);
    std::map<std::pair<int, int>, ResultsTable::Pair> pairs = readResults(dir);
    for (const auto& [key, p] : pairs) assert(key.first >= 0);
    for (int g = 1; g <= 3; ++g) {
        assert(pairs.count({g - 1, g}));
        int opponents = 0;
        for (const auto& [key, p] : pairs)
            if (key.second == g) ++opponents;
        assert(opponents >= 1 && opponents <= 2);
    }
    assert(pairs.size() >= 3 && pairs.at({0, 1}).games >= 2);
    const std::vector<std::string> refs = splitLines(readFile(dir.file("references.csv")));
    assert(refs[1] == "0,checkpoint,,0,0.0");
}

// A blown-up learning rate must abort
```

Replace:

```cpp
    test_non_finite_weights_keep_last_good_state();
```

with:

```cpp
    test_non_finite_weights_keep_last_good_state();
    test_progress_line_lists_opponents();
    test_resume_from_version1_state();
    test_no_bots_no_extras();
```

- [ ] **Step 8: Run the trainer tests to verify they fail**

Run: `cmake --build build -j && ./build/test_trainer`
Expected: `Assertion failed: (calibration.size() == 1), function test_start_new_writes_run_files`. Nothing calibrates the bot yet.

- [ ] **Step 9: Use the round in the trainer**

In `src/trainer.cpp`, replace:

```cpp
#include <map>
#include <stdexcept>
#include <utility>

#include "azgo/arena.hpp"
```

with:

```cpp
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>

#include "azgo/arena.hpp"
#include "azgo/arena_round.hpp"
```

Replace:

```cpp
std::string resultsPath(const std::string& runDir) { return (fs::path(runDir) / "results.csv").string(); }
```

with:

```cpp
std::string resultsPath(const std::string& runDir) { return (fs::path(runDir) / "results.csv").string(); }
std::string referencesPath(const std::string& runDir) {
    return (fs::path(runDir) / "references.csv").string();
}
```

`elo.csv` lists generations only. Generation 0 must not read the bot's calibration pair (−1, 0) as a "g−1 vs g" match. Replace:

```cpp
// elo.csv: one row per rated generation, ascending. The win rate columns come
```

with:

```cpp
// elo.csv: one row per rated generation, ascending; bots (negative IDs) are
// listed in references.csv instead. The win rate columns come
```

Replace:

```cpp
    for (const auto& [generation, rating] : fitBradleyTerry(results, 0)) {
        const double elo = rating + 0.0;   // turns a -0.0 into 0.0, so the anchor prints "0.0"
        char line[160];
        auto it = byPair.find({generation - 1, generation});
```

with:

```cpp
    for (const auto& [generation, rating] : fitBradleyTerry(results, 0)) {
        if (generation < 0) continue;
        const double elo = rating + 0.0;   // turns a -0.0 into 0.0, so the anchor prints "0.0"
        char line[160];
        // Generation 0 has no predecessor; (-1, 0) is bot 1's calibration pair.
        auto it = generation > 0 ? byPair.find({generation - 1, generation}) : byPair.end();
```

Replace:

```cpp
void appendLogRow(const std::string& runDir, const GenerationStats& s) {
```

with:

```cpp
// The rating files, rewritten after every refit: results.csv, elo.csv and
// references.csv in `dir`.
void writeRatingFiles(const std::string& dir, const ResultsTable& results,
                      const std::vector<int>& references, const std::vector<int>& botSimulations) {
    writeFileAtomically(resultsPath(dir), results.toCsv());
    writeFileAtomically(eloPath(dir), eloCsv(results));
    writeFileAtomically(referencesPath(dir), referencesCsv(results, references, botSimulations));
}

ArenaConfig arenaConfig(const Config& cfg) {
    ArenaConfig ac;
    ac.games = cfg.arenaGames;
    ac.simulations = cfg.arenaSimulations;
    ac.mctsBatch = cfg.mctsBatch;
    ac.cPuct = cfg.cPuct;
    ac.temperatureMoves = cfg.arenaTemperatureMoves;
    ac.komi = cfg.komi;
    ac.maxMoves = cfg.effectiveMaxMoves();
    return ac;
}

ArenaRoundSettings roundSettings(const Config& cfg, std::optional<int> bestNetwork) {
    ArenaRoundSettings s;
    s.seed = cfg.seed;
    s.arenaGames = cfg.arenaGames;
    s.maxExtra = cfg.arenaOpponents;
    s.calibrationGames = cfg.calibrationGames;
    s.promoteThreshold = cfg.promoteThreshold;
    s.bestNetwork = bestNetwork;
    return s;
}

// Real arena matches between the players of the run in runDir: bots come from
// bot_simulations, generations from their checkpoints, except `liveNet`
// (generation `liveGeneration`), which is already in memory.
MatchFn runMatches(const Config& cfg, const std::string& runDir, int liveGeneration,
                   const Network* liveNet) {
    return [cfg, runDir, liveGeneration, liveNet](int a, int b, int games, uint64_t seed) {
        ArenaConfig ac = arenaConfig(cfg);
        ac.games = games;
        std::unique_ptr<Network> loaded[2];
        ArenaPlayer players[2];
        const int ids[2] = {a, b};
        for (int i = 0; i < 2; ++i) {
            if (ids[i] < 0) {
                players[i] = ArenaPlayer::bot(cfg.botSimulations.at(static_cast<size_t>(-ids[i] - 1)));
            } else if (liveNet && ids[i] == liveGeneration) {
                players[i] = ArenaPlayer::network(*liveNet);
            } else {
                loaded[i] = std::make_unique<Network>(loadNetwork(checkpointPath(runDir, ids[i])));
                players[i] = ArenaPlayer::network(*loaded[i]);
            }
        }
        return playMatch(players[0], players[1], ac, seed, cfg.effectiveWorkers(), cfg.backend);
    };
}

void appendLogRow(const std::string& runDir, const GenerationStats& s) {
```

In `Trainer::startNew`, replace:

```cpp
    state.references = initialReferences(static_cast<int>(cfg.botSimulations.size()));

    writeFileAtomically(logPath(runDir), kLogHeader);
    writeFileAtomically(eloPath(runDir), std::string(kEloHeader) + "0,0,,,\n");
```

with:

```cpp
    state.references = initialReferences(static_cast<int>(cfg.botSimulations.size()));
    // Each bot plays generation 0 now, so it has a rating before it can be chosen.
    calibrateBots(state.results, state.references, roundSettings(cfg, std::nullopt),
                  runMatches(cfg, runDir, 0, &net));

    writeFileAtomically(logPath(runDir), kLogHeader);
    writeRatingFiles(runDir, state.results, state.references, cfg.botSimulations);
```

In `Trainer::runGeneration`, replace the arena block:

```cpp
    const bool arenaThisGeneration = cfg_.arenaEvery > 0 && g % cfg_.arenaEvery == 0;
    if (arenaThisGeneration) {
        std::vector<int> opponents = ladderOpponents(g, cfg_.arenaOpponents);
        if (cfg_.gating &&
            std::find(opponents.begin(), opponents.end(), state_.bestGeneration) == opponents.end())
            opponents.push_back(state_.bestGeneration);

        ArenaConfig ac;
        ac.games = cfg_.arenaGames;
        ac.simulations = cfg_.arenaSimulations;
        ac.mctsBatch = cfg_.mctsBatch;
        ac.cPuct = cfg_.cPuct;
        ac.temperatureMoves = cfg_.arenaTemperatureMoves;
        ac.komi = cfg_.komi;
        ac.maxMoves = cfg_.effectiveMaxMoves();

        for (int o : opponents) {
            const Network opponent = loadNetwork(checkpointPath(runDir_, o));
            const MatchResult r = playMatch(net_, opponent, ac,
                                            deriveSeed(cfg_.seed, static_cast<uint64_t>(g),
                                                       kPhaseArena, static_cast<uint64_t>(o)),
                                            workers, cfg_.backend);
            state_.results.addMatch(g, o, r.games, r.scoreA);
            if (o == g - 1) {
                stats.winRateVsPrev = r.scoreA / r.games;
                stats.distinctVsPrev = r.distinctGames;
            }
            if (cfg_.gating && o == state_.bestGeneration &&
                passesGate(r.scoreA, r.games, cfg_.gatingThreshold))
                newBest = g;
        }
        // Refit every rating from all results so far: Bradley-Terry does not
        // depend on the order results arrived in, so a resumed run agrees.
        writeFileAtomically(resultsPath(runDir_), state_.results.toCsv());
        writeFileAtomically(eloPath(runDir_), eloCsv(state_.results));
    }
```

with:

```cpp
    const bool arenaThisGeneration = cfg_.arenaEvery > 0 && g % cfg_.arenaEvery == 0;
    ArenaRoundResult round;
    if (arenaThisGeneration) {
        const std::optional<int> best =
            cfg_.gating ? std::optional<int>(state_.bestGeneration) : std::nullopt;
        round = runArenaRound(g, state_.results, state_.references, roundSettings(cfg_, best),
                              runMatches(cfg_, runDir_, g, &net_));
        for (const ArenaMatch& m : round.matches) {
            if (m.opponent == g - 1) {
                stats.winRateVsPrev = m.result.scoreA / m.result.games;
                stats.distinctVsPrev = m.result.distinctGames;
            }
            if (best && m.opponent == *best &&
                passesGate(m.result.scoreA, m.result.games, cfg_.gatingThreshold))
                newBest = g;
        }
        // Refit every rating from all results so far: Bradley-Terry does not
        // depend on the order results arrived in, so a resumed run agrees.
        writeRatingFiles(runDir_, state_.results, state_.references, cfg_.botSimulations);
    }
```

The progress line can now be long (up to 35 opponents), so it is streamed instead of formatted into one fixed buffer. Replace:

```cpp
    if (log_) {
        char line[512];
        int len = std::snprintf(line, sizeof line,
                                "gen %d | selfplay %d games avg %.1f plies %.1fs | "
                                "train loss p %.2f v %.2f lr %g %.1fs | ",
                                g, stats.games, stats.avgLength, stats.selfPlaySeconds,
                                stats.policyLoss, stats.valueLoss, stats.lr, stats.trainSeconds);
        if (arenaThisGeneration) {
            char vsPrev[32] = "n/a";
            if (stats.winRateVsPrev >= 0)
                std::snprintf(vsPrev, sizeof vsPrev, "%.2f", stats.winRateVsPrev);
            char distinct[48] = "";
            if (stats.distinctVsPrev >= 0)
                std::snprintf(distinct, sizeof distinct, " (distinct %d/%d)",
                              stats.distinctVsPrev, cfg_.arenaGames);
            std::snprintf(line + len, sizeof line - static_cast<size_t>(len),
                          "arena elo %.1f vs-prev %s%s %.1fs", stats.elo, vsPrev, distinct,
                          stats.arenaSeconds);
        } else {
            std::snprintf(line + len, sizeof line - static_cast<size_t>(len),
                          "no arena, elo %.1f", stats.elo);
        }
        *log_ << line;
```

with:

```cpp
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
            char elo[48], tail[64];
            std::snprintf(elo, sizeof elo, "arena elo %.1f ", stats.elo);
            std::snprintf(tail, sizeof tail, " (distinct %d/%d)%s %.1fs", stats.distinctVsPrev,
                          cfg_.arenaGames, round.promoted ? " promoted" : "", stats.arenaSeconds);
            *log_ << elo << opponentSummary(round, cfg_.botSimulations) << tail;
        } else {
            char noArena[48];
            std::snprintf(noArena, sizeof noArena, "no arena, elo %.1f", stats.elo);
            *log_ << noArena;
        }
```

(The two lines after it, `if (cfg_.gating) *log_ << " | best " …` and `*log_ << std::endl;`, stay.)

- [ ] **Step 10: Remove the ladder**

Nothing calls `ladderOpponents` any more. In `include/azgo/arena.hpp`, delete:

```cpp
// Opponents for generation g: g-1, g-2, g-4, g-8, ... (>= 1), the first
// `maxOpponents` of those, then 0 if not already present. Empty for g == 0.
std::vector<int> ladderOpponents(int generation, int maxOpponents);

```

In `src/arena.cpp`, delete:

```cpp
std::vector<int> ladderOpponents(int generation, int maxOpponents) {
    std::vector<int> out;
    if (generation <= 0) return out;
    // g-1, g-2, g-4, g-8, ...: dense near the present, sparse into the past.
    for (int d = 1; generation - d >= 1 && static_cast<int>(out.size()) < maxOpponents; d *= 2)
        out.push_back(generation - d);
    out.push_back(0);   // the anchor: every rating is relative to generation 0
    return out;
}

```

In `tests/test_arena.cpp`, delete the whole `test_ladder` function:

```cpp
static void test_ladder() {
    assert(ladderOpponents(0, 4).empty());
    assert((ladderOpponents(1, 4) == std::vector<int>{0}));
    assert((ladderOpponents(2, 4) == std::vector<int>{1, 0}));
    assert((ladderOpponents(3, 4) == std::vector<int>{2, 1, 0}));
    assert((ladderOpponents(10, 4) == std::vector<int>{9, 8, 6, 2, 0}));
    assert((ladderOpponents(10, 2) == std::vector<int>{9, 8, 0}));
    assert((ladderOpponents(9, 8) == std::vector<int>{8, 7, 5, 1, 0}));
}

```

and its call in `main`:

```cpp
    test_ladder();
```

Check: `grep -rn ladderOpponents src include tests apps` prints nothing.

- [ ] **Step 11: Check calibration end to end in the smoke test**

The smoke test trains from `configs/5x5.cfg`, so it gets the preset bot (100 simulations, 80 calibration games). In `tests/integration_smoke.sh`, replace:

```bash
test -s "$RUN/sgf/gen_0001_game_00.sgf" || fail "no SGF written for generation 1"
```

with:

```bash
test -s "$RUN/sgf/gen_0001_game_00.sgf" || fail "no SGF written for generation 1"
grep -q '^-1,bot,100,0,' "$RUN/references.csv" || fail "references.csv does not list the calibrated bot"
grep -q '^-1,0,80,' "$RUN/results.csv" || fail "results.csv has no bot calibration games"
```

- [ ] **Step 12: Run everything, both builds**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: `100% tests passed out of 25` (`test_arena_round` is new). `test_resume_is_byte_identical` and `test_thread_count_independence` pass unchanged apart from the extra `references.csv` comparison: calibration, selection and promotion depend only on the results table, the reference set and the seed. Run the BLAS variant. Expected: `100% tests passed out of 25`.

- [ ] **Step 13: Commit**

```bash
git add include/azgo/arena_round.hpp src/arena_round.cpp tests/test_arena_round.cpp \
        include/azgo/run_state.hpp include/azgo/trainer.hpp src/trainer.cpp \
        include/azgo/arena.hpp src/arena.cpp tests/test_arena.cpp tests/test_trainer.cpp \
        tests/integration_smoke.sh
git commit -F - <<'EOF'
Choose arena opponents by rating, with bot calibration and promotion

Each arena round now calibrates new bots against generation 0, refits,
plays g-1, the strongest reference and the references with an expected
score closest to 0.7, and promotes a generation that scores at least
promote_threshold against the strongest reference. The round is one
function behind a match callback, so the trainer and the coming
--rerate share it, and tests can drive it with a synthetic arena: there
the saturated generation 0 stops being chosen once better yardsticks
exist. references.csv reports the reference set; elo.csv stays
generations only. The ladder is gone.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 6: `train --run DIR --rerate`

**Files:**
- Modify: `include/azgo/trainer.hpp`, `src/trainer.cpp`, `apps/train.cpp`, `tests/integration_smoke.sh`
- Test: `tests/test_trainer.cpp`

**Interfaces:**
- Consumes:
  - from Task 5: `calibrateBots`, `runArenaRound`, `opponentSummary`, and the private helpers `writeRatingFiles`, `roundSettings` and `runMatches` in `src/trainer.cpp`;
  - from Task 4: `initialReferences`.
- Produces:
  - `int azgo::rerateRun(const std::string& runDir, const std::vector<std::string>& overrides, std::ostream* log)`, which returns the last generation replayed.
  - The files `DIR/rerate/{results.csv, elo.csv, references.csv, log.csv}`. This `log.csv` has the header `generation,opponents,score_vs_prev,score_vs_strongest,promoted,elo`, opponents as space-separated player IDs, scores `%.4f`, promoted `0`/`1` and elo `%.1f`.
  - The CLI form `train --run DIR --rerate [--set key=value]...`.

The replay calls the same `calibrateBots` and `runArenaRound` as live training, with the same seeds, and loads every network from its checkpoint (`runMatches(cfg, runDir, -1, nullptr)`). With gating off, a re-rate of a finished run therefore reproduces its `results.csv` byte for byte. The tests check exactly that.

- [ ] **Step 1: Write the failing tests**

In `tests/test_trainer.cpp`, replace:

```cpp
// A blown-up learning rate must abort
```

with:

```cpp
// --rerate replays the arena over the checkpoints. With gating off the arena
// never feeds back into training, so a re-rate of a finished run must play
// exactly the games the live run played: the same results.csv, byte for byte.
static void test_rerate_reproduces_live_results() {
    TempDir dir("rerate");
    startQuiet(tinyConfig(), dir)->runUntil(3);
    const std::string state = readFile(statePath(dir.str()));
    const std::string config = readFile(configPath(dir.str()));

    assert(rerateRun(dir.str(), {}, nullptr) == 3);
    for (const char* f : {"results.csv", "elo.csv", "references.csv"})
        assert(readFile(dir.file(std::string("rerate/") + f)) == readFile(dir.file(f)));
    // It trains nothing and changes nothing outside rerate/.
    assert(readFile(statePath(dir.str())) == state);
    assert(readFile(configPath(dir.str())) == config);
    assert(!fs::exists(checkpointPath(dir.str(), 4)));

    // log.csv: one row per generation; the opponents start with g-1.
    const std::vector<std::string> rows = splitLines(readFile(dir.file("rerate/log.csv")));
    assert(rows.size() == 4);
    assert(rows[0] == "generation,opponents,score_vs_prev,score_vs_strongest,promoted,elo");
    for (int g = 1; g <= 3; ++g) {
        const std::string& row = rows[static_cast<size_t>(g)];
        assert(row.rfind(std::to_string(g) + "," + std::to_string(g - 1), 0) == 0);
        assert(std::count(row.begin(), row.end(), ',') == 5);
    }

    // Independent of the worker count, log.csv included (it has no timings).
    std::map<std::string, std::string> one;
    rerateRun(dir.str(), {"workers=1"}, nullptr);
    for (const char* f : {"results.csv", "elo.csv", "references.csv", "log.csv"})
        one[f] = readFile(dir.file(std::string("rerate/") + f));
    rerateRun(dir.str(), {"workers=3"}, nullptr);
    for (const char* f : {"results.csv", "elo.csv", "references.csv", "log.csv"})
        assert(readFile(dir.file(std::string("rerate/") + f)) == one[f]);
}

// Review Focus 5: re-rating something that is not a complete run.
static void test_rerate_refusals_and_gaps() {
    TempDir empty("rerate_empty");
    fs::create_directories(empty.path);
    bool threw = false;
    try {
        rerateRun(empty.str(), {}, nullptr);
    } catch (const ConfigError& e) {
        threw = std::string(e.what()).find(empty.str()) != std::string::npos;
    }
    assert(threw);

    TempDir dir("rerate_gap");
    startQuiet(tinyConfig(), dir)->runUntil(3);
    threw = false;
    try {
        rerateRun(dir.str(), {"filters=16"}, nullptr);   // would not match the checkpoints
    } catch (const ConfigError& e) {
        threw = std::string(e.what()).find("filters") != std::string::npos;
    }
    assert(threw);
    assert(!fs::exists(dir.path / "rerate"));

    // Every generation is measured against g-1, so a missing checkpoint ends
    // the replay: with gen_0002 gone only generation 1 is re-rated.
    fs::remove(checkpointPath(dir.str(), 2));
    assert(rerateRun(dir.str(), {}, nullptr) == 1);
    assert(countLines(readFile(dir.file("rerate/log.csv"))) == 2);
}

// A blown-up learning rate must abort
```

Replace:

```cpp
    test_no_bots_no_extras();
```

with:

```cpp
    test_no_bots_no_extras();
    test_rerate_reproduces_live_results();
    test_rerate_refusals_and_gaps();
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build build -j`
Expected: the build fails in `tests/test_trainer.cpp` with `use of undeclared identifier 'rerateRun'`.

- [ ] **Step 3: Declare `rerateRun`**

In `include/azgo/trainer.hpp`, replace:

```cpp
    // Further members: implementation-defined (Task 21).
};
```

with:

```cpp
    // Further members: implementation-defined (Task 21).
};

// Re-rates the run in runDir without training it: replays the arena phase of
// generations 1, 2, ... for as long as their checkpoints exist, with today's
// opponent selection, starting from an empty results table, the initial
// reference set and bot calibration, and with the seeds of live training
// (every arena_every-th generation, as live). Gating is ignored. `overrides`
// change tunable keys for the replay only; architecture keys are refused
// (ConfigError). Writes runDir/rerate/{results.csv, elo.csv, references.csv,
// log.csv}, rewritten after each generation; never touches state.bin or
// config.cfg. With gating off, a run's checkpoints do not depend on its
// arena, so this is exactly what a live run would have measured. Progress
// lines go to `log` (nullptr = silent). Returns the last generation replayed.
int rerateRun(const std::string& runDir, const std::vector<std::string>& overrides,
              std::ostream* log);
```

- [ ] **Step 4: Implement it**

In `src/trainer.cpp`, replace:

```cpp
const char* const kEloHeader = "generation,elo,winrate_vs_prev,wr_low,wr_high\n";
```

with:

```cpp
const char* const kEloHeader = "generation,elo,winrate_vs_prev,wr_low,wr_high\n";
const char* const kRerateLogHeader =
    "generation,opponents,score_vs_prev,score_vs_strongest,promoted,elo\n";
```

Replace:

```cpp
void Trainer::runUntil(int lastGeneration) {
    while (generation() < lastGeneration) runGeneration();
}
```

with:

```cpp
void Trainer::runUntil(int lastGeneration) {
    while (generation() < lastGeneration) runGeneration();
}

int rerateRun(const std::string& runDir, const std::vector<std::string>& overrides,
              std::ostream* log) {
    if (!fs::exists(configPath(runDir)))
        throw ConfigError(runDir + " does not contain a run (no config.cfg)");
    const Config saved = loadConfigFile(configPath(runDir));
    Config cfg = saved;
    for (const std::string& o : overrides) applyOverride(cfg, o);
    const std::vector<std::string> diff = architectureDiff(saved, cfg);
    if (!diff.empty()) {
        std::string keys;
        for (const std::string& k : diff) keys += (keys.empty() ? "" : ", ") + k;
        throw ConfigError("cannot change architecture keys when re-rating: " + keys);
    }
    if (!fs::exists(checkpointPath(runDir, 0)))
        throw std::runtime_error(checkpointPath(runDir, 0) + ": missing");

    // Every generation is measured against g-1, so the replay ends at the
    // first missing checkpoint.
    int last = 0;
    while (fs::exists(checkpointPath(runDir, last + 1))) ++last;

    const std::string outDir = (fs::path(runDir) / "rerate").string();
    fs::create_directories(outDir);
    ResultsTable results;
    std::vector<int> references = initialReferences(static_cast<int>(cfg.botSimulations.size()));
    const ArenaRoundSettings settings = roundSettings(cfg, std::nullopt);   // gating ignored
    const MatchFn play = runMatches(cfg, runDir, -1, nullptr);   // every network from disk
    calibrateBots(results, references, settings, play);
    writeRatingFiles(outDir, results, references, cfg.botSimulations);
    std::string logCsv = kRerateLogHeader;
    writeFileAtomically(logPath(outDir), logCsv);

    for (int g = 1; g <= last; ++g) {
        if (cfg.arenaEvery == 0 || g % cfg.arenaEvery != 0) continue;
        const auto start = Clock::now();
        const ArenaRoundResult round = runArenaRound(g, results, references, settings, play);
        writeRatingFiles(outDir, results, references, cfg.botSimulations);

        // generation,opponents,score_vs_prev,score_vs_strongest,promoted,elo
        // opponents are player IDs separated by spaces; matches[0] is g-1.
        std::string ids;
        double vsStrongest = 0;
        for (const ArenaMatch& m : round.matches) {
            ids += (ids.empty() ? "" : " ") + std::to_string(m.opponent);
            if (m.opponent == round.strongest) vsStrongest = m.result.scoreA / m.result.games;
        }
        const MatchResult& prev = round.matches.front().result;
        const double elo = fitBradleyTerry(results, 0).at(g) + 0.0;
        char row[128];
        std::snprintf(row, sizeof row, ",%.4f,%.4f,%d,%.1f\n", prev.scoreA / prev.games, vsStrongest,
                      round.promoted ? 1 : 0, elo);
        logCsv += std::to_string(g) + "," + ids + row;
        writeFileAtomically(logPath(outDir), logCsv);

        if (log) {
            char head[64], tail[48];
            std::snprintf(head, sizeof head, "rerate gen %d | elo %.1f ", g, elo);
            std::snprintf(tail, sizeof tail, "%s %.1fs", round.promoted ? " promoted" : "",
                          secondsSince(start));
            *log << head << opponentSummary(round, cfg.botSimulations) << tail << std::endl;
        }
    }
    return last;
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_trainer`
Expected: `100% tests passed out of 1`.

- [ ] **Step 6: Add `--rerate` to the train app**

Full file `apps/train.cpp`:

```cpp
// train: runs the AlphaZero generation loop for a new or resumed run.
//
//   train --config configs/9x9.cfg --run runs/nine [--set key=value]... [--generations N]
//   train --run runs/nine --resume [--set key=value]... [--generations N]
//   train --run runs/nine --rerate [--set key=value]...
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "azgo/config.hpp"
#include "azgo/run_state.hpp"
#include "azgo/trainer.hpp"
#include "cli.hpp"

namespace {

const char* const kUsage =
    "usage:\n"
    "  train --config FILE --run DIR [--set key=value]... [--generations N]\n"
    "  train --run DIR --resume [--set key=value]... [--generations N]\n"
    "  train --run DIR --rerate [--set key=value]...\n"
    "\n"
    "A new run copies FILE (plus overrides) to DIR/config.cfg. --resume continues\n"
    "from DIR/state.bin; tunable keys may be overridden, architecture keys may not.\n"
    "--generations N is shorthand for --set generations=N (train until generation N).\n"
    "--rerate trains nothing: it replays the arena over DIR's checkpoints with the\n"
    "current opponent selection and writes DIR/rerate/ (results.csv, elo.csv,\n"
    "references.csv, log.csv). state.bin and config.cfg are left as they are;\n"
    "--set applies to the replay only.\n";

int usageError(const std::string& message) {
    std::fprintf(stderr, "error: %s\n\n%s", message.c_str(), kUsage);
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    std::unique_ptr<ArgParser> args;
    try {
        args = std::make_unique<ArgParser>(argc, argv,
                                           std::vector<std::string>{"config", "run", "set", "generations"},
                                           std::vector<std::string>{"resume", "rerate", "help"});
    } catch (const std::invalid_argument& e) {
        return usageError(e.what());
    }
    if (args->flag("help")) {
        std::fputs(kUsage, stdout);
        return 0;
    }
    if (!args->has("run")) return usageError("--run DIR is required");
    const bool resume = args->flag("resume");
    if (args->flag("rerate")) {
        if (resume || args->has("config") || args->has("generations"))
            return usageError("--rerate cannot be combined with --resume, --config or --generations");
        try {
            const std::string runDir = args->get("run");
            std::cout << "re-rating " << runDir << " (results in " << runDir << "/rerate)" << std::endl;
            const int last = azgo::rerateRun(runDir, args->sets(), &std::cout);
            std::cout << "done: re-rated generations 1-" << last << " of " << runDir << std::endl;
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "error: %s\n", e.what());
            return 1;
        }
    }
    if (resume && args->has("config"))
        return usageError("--config cannot be combined with --resume (the run's config.cfg is used)");
    if (!resume && !args->has("config"))
        return usageError("a new run needs --config FILE (or pass --resume to continue one)");

    try {
        const std::string runDir = args->get("run");
        std::vector<std::string> overrides = args->sets();
        if (args->has("generations"))
            overrides.push_back("generations=" + std::to_string(args->getInt("generations", 0)));

        std::unique_ptr<azgo::Trainer> trainer;
        if (resume) {
            trainer = azgo::Trainer::resume(runDir, overrides);
        } else {
            azgo::Config cfg = azgo::loadConfigFile(args->get("config"));
            for (const std::string& o : overrides) azgo::applyOverride(cfg, o);
            trainer = azgo::Trainer::startNew(cfg, runDir);
        }

        std::cout << (resume ? "resuming " : "starting ") << runDir << " at generation "
                  << trainer->generation() << "; effective config (saved to "
                  << azgo::configPath(runDir) << "):\n"
                  << azgo::configToString(trainer->config()) << std::endl;

        trainer->setLog(&std::cout);
        trainer->runUntil(trainer->config().generations);
        std::cout << "done: generation " << trainer->generation() << " in " << runDir << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
```

In `tests/integration_smoke.sh`, replace:

```bash
# train -> resume -> arena (checkpoint and bot) -> analyze (good and bad input)
# -> bench, plus the refusal to change the architecture on resume.
```

with:

```bash
# train -> resume -> re-rate -> arena (checkpoint and bot) -> analyze (good and
# bad input) -> bench, plus the refusal to change the architecture on resume.
```

Replace:

```bash
[ "$(wc -l < "$RUN/log.csv" | tr -d ' ')" = "4" ] || fail "log.csv should have a header and 3 rows"
```

with:

```bash
[ "$(wc -l < "$RUN/log.csv" | tr -d ' ')" = "4" ] || fail "log.csv should have a header and 3 rows"

echo "== train: re-rate the finished run =="
state_before="$(cksum < "$RUN/state.bin")"
"$BUILD/train" --run "$RUN" --rerate
# Gating is off, so the replay plays exactly the live run's arena games.
cmp "$RUN/results.csv" "$RUN/rerate/results.csv" || fail "rerate/results.csv differs from the live run's"
cmp "$RUN/references.csv" "$RUN/rerate/references.csv" || fail "rerate/references.csv differs"
[ "$(wc -l < "$RUN/rerate/log.csv" | tr -d ' ')" = "4" ] || fail "rerate/log.csv should have a header and 3 rows"
[ "$(cksum < "$RUN/state.bin")" = "$state_before" ] || fail "--rerate changed state.bin"
status=0
out="$("$BUILD/train" --run "$RUN" --rerate --resume 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "--rerate --resume exited $status, expected 1"
echo "$out" | grep -q -- "--rerate cannot be combined" || fail "unexpected refusal message: $out"
```

- [ ] **Step 7: Run everything, both builds**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: `100% tests passed out of 25`. `bash tests/integration_smoke.sh build .` prints a block like this one (the numbers depend on the run):

```
== train: re-rate the finished run ==
re-rating …/run (results in …/run/rerate)
rerate gen 1 | elo … vs 0:… bot100:… …s
rerate gen 2 | elo … vs 1:… … …s
rerate gen 3 | elo … vs 2:… … …s
done: re-rated generations 1-3 of …/run
```

Run the BLAS variant. Expected: `100% tests passed out of 25`.

- [ ] **Step 8: Commit**

```bash
git add include/azgo/trainer.hpp src/trainer.cpp apps/train.cpp tests/test_trainer.cpp tests/integration_smoke.sh
git commit -F - <<'EOF'
Add train --rerate to re-measure a run with the new arena

With gating off the arena never feeds back into self-play or SGD, so a
finished run's checkpoints are exactly what a live run with the new
opponent rules would produce. --rerate replays only the arena over them,
through the same round function and seeds as training, and writes
DIR/rerate/ without touching state.bin or config.cfg. On a tiny run it
reproduces the live results.csv byte for byte, for any worker count.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 7: Teaching material: explainer, spec pointer, README

**Files:**
- Modify: `docs/superpowers/specs/2026-09-29-alphazero-go-design.md`, `docs/algorithm-explained.md`, `docs/algorithm-explained.html`, `README.md`
- Check: `tools/check_citations.py` (unchanged)

**Interfaces:**
- Consumes: the code of Tasks 1–6. The new caption cites `src/arena.cpp:152 — selectOpponents`, and the concept map cites `selectOpponents`/`passesGate` at `src/arena.cpp:152, 180` and `calibrateBots`/`runArenaRound` at `src/arena_round.cpp:29, 46`. These line numbers hold when Tasks 1–6 were applied exactly as written; Step 4 corrects them otherwise.
- Produces: an explainer whose chapter 07 teaches the rating-aware arena; `check_citations.py` passes. Task 8 adds measured numbers to the same chapter.

The HTML is a chapter-for-chapter rendering of the Markdown with identical captions. Every Markdown change below has its HTML twin in Step 3. Text in the HTML is one `<p>` per paragraph, with `<code>` for backticks and HTML escapes inside `<pre><code>`.

- [ ] **Step 1: Point the original spec at the amendment**

In `docs/superpowers/specs/2026-09-29-alphazero-go-design.md`:

Replace:

```markdown
**`arena` / `elo`**
```

with:

```markdown
**`arena` / `elo`**

> Amended 2026-10-01 by `2026-10-01-rating-aware-arena-design.md`: the
> ladder in the first bullet is replaced by opponents chosen by rating (g−1,
> the strongest reference and the references with an expected score closest
> to 0.7), plus a uniform-prior MCTS bot as a fixed yardstick. Generation 0
> stays the anchor at 0 Elo.
```

Then insert one blank line after the quote: the original has none between the heading and the bullet list that follows it.

- [ ] **Step 2: Rewrite the ladder section of the Markdown explainer**

In `docs/algorithm-explained.md`, make these seven replacements. In order, they cover: the abridged `playArenaGame` comment, the ladder section (which becomes "Choosing informative opponents" plus "A yardstick from outside the lineage"), the drift paragraph, the outside-yardstick sentence, the chapter 08 step list, the chapter 09 table row, and the concept-map row.

Replace:

```markdown
    /* ... game; MctsConfig mc from cfg with addNoise = false; searchA, searchB ... */
```

with:

```markdown
    /* ... game; searchA, searchB from each Seat's MctsConfig, addNoise = false ... */
```

Replace:

~~~~markdown
**The ladder of opponents.** Playing only g − 1 measures one step at a
time, and small errors add up. Playing everyone is too expensive. The
compromise is a sparse ladder, dense near the present and sparse into
the past, plus generation 0:

```cpp
std::vector<int> ladderOpponents(int generation, int maxOpponents) {
    std::vector<int> out;
    if (generation <= 0) return out;
    // g-1, g-2, g-4, g-8, ...: dense near the present, sparse into the past.
    for (int d = 1; generation - d >= 1 && static_cast<int>(out.size()) < maxOpponents; d *= 2)
        out.push_back(generation - d);
    out.push_back(0);   // the anchor: every rating is relative to generation 0
    return out;
}
```
*src/arena.cpp:102 — `ladderOpponents`*

With `arena_opponents = 4`, generation 33 plays 32, 31, 29, 25 and 0.
~~~~

with:

~~~~markdown
**Choosing informative opponents.** Playing only g − 1 measures one
step at a time, and small errors add up. Playing everyone is too
expensive. So which earlier players are worth twenty games? A game the
new network wins with probability p has a result with variance
p(1 − p), and that is how much it can tell you: 0.25 at p = 0.5, 0.21
at 0.7, 0.09 at 0.9, 0.01 at 0.99. A game against an opponent it beats
99% of the time is worth a twenty-fifth of an even game. The rating fit
adds a second limit (next section): a 20–0 sweep can never say more
than 400·log₁₀(41) ≈ 645 Elo. The reference runs in `docs/results/`
were made with a fixed ladder: g − 1, g − 2, g − 4, g − 8, and always
generation 0. On 9×9, generation 0 scored at most 2 of 20 in 137 of its
150 matches, and 0 of 20 in 110 (`docs/results/9x9/results.csv`). The
end of this chapter shows what that did to the curve. The arena now
picks its opponents from the current ratings instead:

```cpp
std::vector<int> selectOpponents(int generation, const std::map<int, double>& ratings,
                                 const std::vector<int>& references, int maxExtra,
                                 std::optional<int> bestNetwork) {
    /* ... generation must be >= 1 ... */
    std::vector<int> out{generation - 1};   // the progress signal
    auto chosen = [&](int p) { return std::find(out.begin(), out.end(), p) != out.end(); };
    const int strongest = strongestReference(ratings, references);
    if (!chosen(strongest)) out.push_back(strongest);   // the promotion check

    // The new network has no rating yet; its predecessor's is the best guess.
    auto prev = ratings.find(generation - 1);
    const double estimate = prev == ratings.end() ? 0.0 : prev->second;
    std::vector<std::pair<double, int>> candidates;   // (|E - 0.7|, player)
    for (int r : references) {
        auto it = ratings.find(r);
        if (it == ratings.end() || chosen(r)) continue;
        candidates.push_back({std::fabs(expectedScore(estimate, it->second) - kTargetScore), r});
    }
    std::sort(candidates.begin(), candidates.end());   // by distance, then by smaller ID
    for (size_t i = 0; i < candidates.size() && static_cast<int>(i) < maxExtra; ++i)
        out.push_back(candidates[i].second);

    if (bestNetwork && !chosen(*bestNetwork)) out.push_back(*bestNetwork);   // gating
    return out;
}
```
*src/arena.cpp:152 — `selectOpponents` (abridged)*

The candidates are the **references**: a frozen set of players that
starts as generation 0 plus the bots, and only grows.

- **g − 1** is always played. It is the progress signal, and its match
  is the one `log.csv` reports.
- **The strongest reference** is always played. It is the promotion
  check: a generation that scores at least `promote_threshold` against
  it (0.85, about 300 Elo) plays it `calibration_games` (80) more games
  and joins the references, as the next yardstick.
- **`arena_opponents` more references** (2 in the presets) are the ones
  for which the new network's expected score,
  E = 1 / (1 + 10^((R_c − R̂)/400)), is closest to 0.7. R̂, the new
  network's rating, is not known yet, so g − 1's rating stands in.

Generation 0 stays pinned at 0 Elo and is never removed. Once every new
network beats it, it simply stops being chosen, and its old results stay
in the fit. For example, take generation 10, with g − 1 rated 800 and
references at 100 (a bot), 0 (generation 0), 450 (generation 4) and 700
(generation 6). It plays 9, then 6 (the strongest), then 4 (E = 0.88)
and the bot (E = 0.98). Generation 0, at E = 0.99, is left out
(`test_select_opponents` in `tests/test_arena.cpp`).

**A yardstick from outside the lineage.** The references include
**uniform bots** (`bot_simulations = 100` in the presets): MCTS players
whose priors are all equal and whose value is always 0. A bot knows the
rules and nothing else, so its strength depends only on its simulation
count, and it is the same in every run. Each bot plays generation 0
`calibration_games` games when a run starts, so it is rated before it
can be chosen. `references.csv` lists every reference with its rating.
Selection, matches, refit and promotion are one function,
`runArenaRound` (`src/arena_round.cpp`). Training uses it, and so does
`train --run DIR --rerate`, which replays the arena over a finished
run's checkpoints into `DIR/rerate/`. With gating off, the arena never
feeds back into training, so a re-rated run measures exactly the
networks a live run would have.
~~~~

Replace:

```markdown
**Drift and non-transitivity.** Above the anchor's 645-point cap, every
rating is held up by a chain of ladder matches, each only 20 games.
```

with:

```markdown
**Drift and non-transitivity.** In the ladder-era runs of
`docs/results/`, every rating above the anchor's 645-point cap is held
up by a chain of ladder matches, each only 20 games.
```

Replace:

```markdown
opponents. This project has no outside yardstick by design (the spec
rules out baseline bots and external engines), so its Elo claims are
claims about this lineage only.
```

with:

```markdown
opponents. The uniform bot is the only outside yardstick here, and a
weak one; the reference runs were made before it existed, so their Elo
claims are claims about their own lineage only. External engines stay
out of scope.
```

Replace:

```markdown
4. arena       new network vs the ladder; results.csv and elo.csv refit
```

with:

```markdown
4. arena       new network vs g-1 and chosen references; rating files refit
```

Replace:

```markdown
| Evaluation | vs. exhaustive minimax | arena vs. own ladder, Bradley-Terry Elo, Wilson intervals, optional gating | no oracle for Go |
```

with:

```markdown
| Evaluation | vs. exhaustive minimax | arena vs. own references and a uniform bot, Bradley-Terry Elo, Wilson intervals, optional gating | no oracle for Go |
```

Replace:

```markdown
| Ladder of opponents, gating | `ladderOpponents`, `passesGate` — `src/arena.cpp:102, 112` |
```

with:

```markdown
| Choosing opponents, gating | `selectOpponents`, `passesGate` — `src/arena.cpp:152, 180` |
| Arena rounds: calibration, promotion | `calibrateBots`, `runArenaRound` — `src/arena_round.cpp:29, 46` |
```

- [ ] **Step 3: Mirror it in the HTML explainer**

In `docs/algorithm-explained.html`, make the seven matching replacements (the fifth is the arena box of the generation-loop diagram):

Replace:

```html
    /* ... game; MctsConfig mc from cfg with addNoise = false; searchA, searchB ... */
```

with:

```html
    /* ... game; searchA, searchB from each Seat's MctsConfig, addNoise = false ... */
```

Replace:

```html
<p><strong>The ladder of opponents.</strong> Playing only g − 1 measures one step at a time, and small errors add up. Playing everyone is too expensive. The compromise is a sparse ladder, dense near the present and sparse into the past, plus generation 0:</p>
<pre><code>std::vector&lt;int&gt; ladderOpponents(int generation, int maxOpponents) {
    std::vector&lt;int&gt; out;
    if (generation &lt;= 0) return out;
    // g-1, g-2, g-4, g-8, ...: dense near the present, sparse into the past.
    for (int d = 1; generation - d &gt;= 1 &amp;&amp; static_cast&lt;int&gt;(out.size()) &lt; maxOpponents; d *= 2)
        out.push_back(generation - d);
    out.push_back(0);   // the anchor: every rating is relative to generation 0
    return out;
}</code></pre>
<span class="src">src/arena.cpp:102 — <code>ladderOpponents</code></span>
<p>With <code>arena_opponents = 4</code>, generation 33 plays 32, 31, 29, 25 and 0.</p>
```

with:

```html
<p><strong>Choosing informative opponents.</strong> Playing only g − 1 measures one step at a time, and small errors add up. Playing everyone is too expensive. So which earlier players are worth twenty games? A game the new network wins with probability p has a result with variance p(1 − p), and that is how much it can tell you: 0.25 at p = 0.5, 0.21 at 0.7, 0.09 at 0.9, 0.01 at 0.99. A game against an opponent it beats 99% of the time is worth a twenty-fifth of an even game. The rating fit adds a second limit (next section): a 20–0 sweep can never say more than 400·log₁₀(41) ≈ 645 Elo. The reference runs in <code>docs/results/</code> were made with a fixed ladder: g − 1, g − 2, g − 4, g − 8, and always generation 0. On 9×9, generation 0 scored at most 2 of 20 in 137 of its 150 matches, and 0 of 20 in 110 (<code>docs/results/9x9/results.csv</code>). The end of this chapter shows what that did to the curve. The arena now picks its opponents from the current ratings instead:</p>
<pre><code>std::vector&lt;int&gt; selectOpponents(int generation, const std::map&lt;int, double&gt;&amp; ratings,
                                 const std::vector&lt;int&gt;&amp; references, int maxExtra,
                                 std::optional&lt;int&gt; bestNetwork) {
    /* ... generation must be &gt;= 1 ... */
    std::vector&lt;int&gt; out{generation - 1};   // the progress signal
    auto chosen = [&amp;](int p) { return std::find(out.begin(), out.end(), p) != out.end(); };
    const int strongest = strongestReference(ratings, references);
    if (!chosen(strongest)) out.push_back(strongest);   // the promotion check

    // The new network has no rating yet; its predecessor's is the best guess.
    auto prev = ratings.find(generation - 1);
    const double estimate = prev == ratings.end() ? 0.0 : prev-&gt;second;
    std::vector&lt;std::pair&lt;double, int&gt;&gt; candidates;   // (|E - 0.7|, player)
    for (int r : references) {
        auto it = ratings.find(r);
        if (it == ratings.end() || chosen(r)) continue;
        candidates.push_back({std::fabs(expectedScore(estimate, it-&gt;second) - kTargetScore), r});
    }
    std::sort(candidates.begin(), candidates.end());   // by distance, then by smaller ID
    for (size_t i = 0; i &lt; candidates.size() &amp;&amp; static_cast&lt;int&gt;(i) &lt; maxExtra; ++i)
        out.push_back(candidates[i].second);

    if (bestNetwork &amp;&amp; !chosen(*bestNetwork)) out.push_back(*bestNetwork);   // gating
    return out;
}</code></pre>
<span class="src">src/arena.cpp:152 — <code>selectOpponents</code> (abridged)</span>
<p>The candidates are the <strong>references</strong>: a frozen set of players that starts as generation 0 plus the bots, and only grows.</p>
<ul>
<li><strong>g − 1</strong> is always played. It is the progress signal, and its match is the one <code>log.csv</code> reports.</li>
<li><strong>The strongest reference</strong> is always played. It is the promotion check: a generation that scores at least <code>promote_threshold</code> against it (0.85, about 300 Elo) plays it <code>calibration_games</code> (80) more games and joins the references, as the next yardstick.</li>
<li><strong><code>arena_opponents</code> more references</strong> (2 in the presets) are the ones for which the new network's expected score, E = 1 / (1 + 10^((R_c − R̂)/400)), is closest to 0.7. R̂, the new network's rating, is not known yet, so g − 1's rating stands in.</li>
</ul>
<p>Generation 0 stays pinned at 0 Elo and is never removed. Once every new network beats it, it simply stops being chosen, and its old results stay in the fit. For example, take generation 10, with g − 1 rated 800 and references at 100 (a bot), 0 (generation 0), 450 (generation 4) and 700 (generation 6). It plays 9, then 6 (the strongest), then 4 (E = 0.88) and the bot (E = 0.98). Generation 0, at E = 0.99, is left out (<code>test_select_opponents</code> in <code>tests/test_arena.cpp</code>).</p>
<p><strong>A yardstick from outside the lineage.</strong> The references include <strong>uniform bots</strong> (<code>bot_simulations = 100</code> in the presets): MCTS players whose priors are all equal and whose value is always 0. A bot knows the rules and nothing else, so its strength depends only on its simulation count, and it is the same in every run. Each bot plays generation 0 <code>calibration_games</code> games when a run starts, so it is rated before it can be chosen. <code>references.csv</code> lists every reference with its rating. Selection, matches, refit and promotion are one function, <code>runArenaRound</code> (<code>src/arena_round.cpp</code>). Training uses it, and so does <code>train --run DIR --rerate</code>, which replays the arena over a finished run's checkpoints into <code>DIR/rerate/</code>. With gating off, the arena never feeds back into training, so a re-rated run measures exactly the networks a live run would have.</p>
```

Replace:

```html
<p><strong>Drift and non-transitivity.</strong> Above the anchor's 645-point cap, every rating is held up by a chain of ladder matches, each only 20 games.
```

with:

```html
<p><strong>Drift and non-transitivity.</strong> In the ladder-era runs of <code>docs/results/</code>, every rating above the anchor's 645-point cap is held up by a chain of ladder matches, each only 20 games.
```

Replace:

```html
opponents. This project has no outside yardstick by design (the spec rules out baseline bots and external engines), so its Elo claims are claims about this lineage only.</p>
```

with:

```html
opponents. The uniform bot is the only outside yardstick here, and a weak one; the reference runs were made before it existed, so their Elo claims are claims about their own lineage only. External engines stay out of scope.</p>
```

Replace:

```html
>vs the ladder; results.csv, elo.csv refit</text>
```

with:

```html
>vs g−1 and references; rating files refit</text>
```

Replace:

```html
<tr><td>Evaluation</td><td>vs. exhaustive minimax</td><td>arena vs. own ladder, Bradley-Terry Elo, Wilson intervals, optional gating</td><td>no oracle for Go</td></tr>
```

with:

```html
<tr><td>Evaluation</td><td>vs. exhaustive minimax</td><td>arena vs. own references and a uniform bot, Bradley-Terry Elo, Wilson intervals, optional gating</td><td>no oracle for Go</td></tr>
```

Replace:

```html
  <dt>Ladder of opponents, gating</dt>
  <dd><code>ladderOpponents</code>, <code>passesGate</code> <span class="loc">— src/arena.cpp:102, 112</span></dd>
```

with:

```html
  <dt>Choosing opponents, gating</dt>
  <dd><code>selectOpponents</code>, <code>passesGate</code> <span class="loc">— src/arena.cpp:152, 180</span></dd>
  <dt>Arena rounds: calibration, promotion</dt>
  <dd><code>calibrateBots</code>, <code>runArenaRound</code> <span class="loc">— src/arena_round.cpp:29, 46</span></dd>
```

- [ ] **Step 4: Repair and check the citations**

Tasks 1–6 moved code, so several old captions now point at the wrong lines. Run: `python3 tools/check_citations.py`
Expected: exit 1, with problems such as:

```
algorithm-explained.md:…: `playArenaGame` cites src/arena.cpp:35, belongs at 50
algorithm-explained.md:…: `fitBradleyTerry` cites src/elo.cpp:112, belongs at 113
algorithm-explained.md:…: `Trainer::runGeneration` cites src/trainer.cpp:180, belongs at 258
algorithm-explained.html: concept map cites src/run_state.cpp:17 for `saveTrainingState`, defined at 27
…
```

No line may say `cannot locate`; if one does, a caption names a symbol that no longer exists and must be fixed by hand. Run: `python3 tools/check_citations.py --fix && python3 tools/check_citations.py`
Expected: `repaired 4 markdown and 13 html citations`, then `all 36 markdown citations resolve, and the html agrees`.

- [ ] **Step 5: Update the Markdown concept map to match**

The checker verifies the HTML concept map but not the Markdown table, so update the Markdown numbers by hand. In `docs/algorithm-explained.md`:

Replace:

```markdown
`src/arena.cpp:35, 66`
```

with:

```markdown
`src/arena.cpp:50, 86`
```

Replace:

```markdown
`fitBradleyTerry` — `src/elo.cpp:112`
```

with:

```markdown
`fitBradleyTerry` — `src/elo.cpp:113`
```

Replace:

```markdown
`wilsonInterval` — `src/elo.cpp:183`
```

with:

```markdown
`wilsonInterval` — `src/elo.cpp:184`
```

Replace:

```markdown
`src/trainer.cpp:81, 56`
```

with:

```markdown
`src/trainer.cpp:91, 64`
```

Replace:

```markdown
`Trainer::runGeneration` — `src/trainer.cpp:180`
```

with:

```markdown
`Trainer::runGeneration` — `src/trainer.cpp:258`
```

Replace:

```markdown
`src/run_state.cpp:17`; `Trainer::resume` — `src/trainer.cpp:140`
```

with:

```markdown
`src/run_state.cpp:27`; `Trainer::resume` — `src/trainer.cpp:211`
```

Then confirm that the two concept maps agree:

```bash
python3 - <<'PY'
import re
md = open("docs/algorithm-explained.md").read()
html = open("docs/algorithm-explained.html").read()
m = re.findall(r"`([\w./]+\.(?:cpp|hpp)):([\d, ]+)`", md[md.index("## 10"):])
h = re.findall(r'<span class="loc">— ([\w./]+\.(?:cpp|hpp)):([\d, ]+)</span>', html)
print(len(m), len(h), [x for x in zip(m, h) if x[0] != x[1]])
PY
```

Expected: `46 46 []`. If a pair differs, copy the HTML number (the checked one) into the Markdown.

- [ ] **Step 6: Update the README**

In `README.md`:

Replace:

```markdown
Each build has 23 component tests plus an integration smoke test (24 in
ctest; `ctest -N | tail -1` prints `Total Tests: 24`).
```

with:

```markdown
Each build has 24 component tests plus an integration smoke test (25 in
ctest; `ctest -N | tail -1` prints `Total Tests: 25`).
```

Replace:

```markdown
`--set key=value` overrides any config key and `--generations N` trains
```

with:

```markdown
These lines are from the reference runs, made before the arena chose its
opponents by rating. The arena part of the line now lists every opponent
with the new network's score, in the form `vs 149:0.55 120:0.70
bot100:1.00`, and ends with `promoted` when the generation joins the
reference set. `references.csv` in the run directory lists the reference
set with ratings, bots included.

`--set key=value` overrides any config key and `--generations N` trains
```

Replace:

~~~~markdown
$ ./build-blas/train --run runs/5x5-seed1 --resume --set blocks=4
error: cannot change architecture keys on resume: blocks
```
~~~~

with:

~~~~markdown
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
~~~~

Replace:

```markdown
**`arena`** plays two checkpoints against each other, colors alternating,
and prints the score with a 95% Wilson interval
```

with:

```markdown
**`arena`** plays two checkpoints against each other (or, with
`--b-bot SIMS` instead of `--b`, a checkpoint against a uniform bot
searching SIMS simulations per move), colors alternating, and prints the
score with a 95% Wilson interval
```

Replace:

```markdown
- [`docs/superpowers/plans/2026-09-29-alphazero-go.md`](docs/superpowers/plans/2026-09-29-alphazero-go.md)
  — the implementation plan.
```

with:

```markdown
- [`docs/superpowers/plans/2026-09-29-alphazero-go.md`](docs/superpowers/plans/2026-09-29-alphazero-go.md)
  — the implementation plan.
- [`docs/superpowers/specs/2026-10-01-rating-aware-arena-design.md`](docs/superpowers/specs/2026-10-01-rating-aware-arena-design.md)
  and [`docs/superpowers/plans/2026-10-01-rating-aware-arena.md`](docs/superpowers/plans/2026-10-01-rating-aware-arena.md)
  — the rating-aware arena: opponents chosen by rating, the uniform bot and `--rerate`.
```

- [ ] **Step 7: Verify**

Run: `python3 tools/check_citations.py && cmake --build build -j && ctest --test-dir build -N | tail -1 && grep -rn "ladderOpponents\|vs. own ladder" docs/algorithm-explained.* README.md`
Expected: `all 36 markdown citations resolve, and the html agrees`, then `Total Tests: 25`, and no grep output. (The remaining mentions of "ladder" describe how the reference runs were made, and stay.) Open `docs/algorithm-explained.html` in a browser and check that chapter 07 renders: a code block, a bulleted list and the two new paragraphs.

- [ ] **Step 8: Commit**

```bash
git add docs/superpowers/specs/2026-09-29-alphazero-go-design.md docs/algorithm-explained.md docs/algorithm-explained.html README.md
git commit -F - <<'EOF'
Teach rating-aware opponents in the explainer and README

Chapter 07 replaces the ladder with "choosing informative opponents":
why a game's information is p(1-p), why 20-0 sweeps of generation 0
capped the 9x9 curve near 645 Elo, how selectOpponents picks g-1, the
strongest reference and the expected-0.7 references, and what the
uniform bot adds. The original spec now points at the amendment, the
README documents --rerate, --b-bot and the 25 tests, and every citation
is re-anchored.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 8: Re-rate the reference runs and report the results

This task is operational, not TDD. Its "tests" are the spec's success criterion and a set of sanity checks, run with the project's own tools. **Report results honestly.** Judge the 9×9 criterion exactly as the original plan defines it: "Elo is still rising over the last third of the run (a linear fit over those generations has a positive slope)". Apply it to the re-rated `elo.csv`, and keep the old verdict next to the new one. Do not change the criterion, the presets or the selection rules to improve the numbers. If a result looks wrong, use superpowers:systematic-debugging before changing anything. That covers a re-rated curve that falls while the head-to-heads rise, a bot rated above the late networks, a generation with no g−1 match, or a run that crashes.

**Files:**
- Create: `docs/results/9x9/rerate/`, `docs/results/5x5/rerate/` and `docs/results/5x5-seed2/rerate/`, each holding `results.csv`, `elo.csv`, `references.csv`, `log.csv` and `progress.txt`
- Modify: `docs/results/elo-9x9.svg`, `docs/results/README.md`, `README.md`, `docs/algorithm-explained.md`, `docs/algorithm-explained.html`

**Interfaces:**
- Consumes:
  - `train --rerate` (Task 6), `tools/plot_elo.py` and `tools/check_citations.py`;
  - the run directories `runs/9x9-main`, `runs/5x5-seed1` and `runs/5x5-seed2`, which hold the checkpoints of the published reference runs (not in git).
- Produces: the published re-rated results and the 9×9 verdict on them.

- [ ] **Step 1: Build both variants and run the full test suite**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure
cmake -S . -B build-blas -DCMAKE_BUILD_TYPE=Release -DAZ_USE_BLAS=ON && cmake --build build-blas -j && ctest --test-dir build-blas --output-on-failure
```

Expected: `100% tests passed out of 25` in both builds.

- [ ] **Step 2: Check the inputs**

```bash
ls runs/9x9-main/checkpoints | wc -l; ls runs/5x5-seed1/checkpoints | wc -l; ls runs/5x5-seed2/checkpoints | wc -l
grep -E '^(arena_opponents|arena_games|komi|backend|bot_simulations) ' runs/9x9-main/config.cfg
```

Expected: `151`, `61` and `61` (generations 0–150 and 0–60). The config prints `backend = blas`, `komi = 7.5`, `arena_games = 20` and `arena_opponents = 4`, and no `bot_simulations` line: these configs predate the key, so the default `100` applies. `arena_opponents = 4` meant "ladder size" when the run was made. The current presets use 2, so every re-rate below passes `--set arena_opponents=2`.

- [ ] **Step 3: Re-rate the three runs**

```bash
./build-blas/train --run runs/9x9-main --rerate --set arena_opponents=2 > runs/9x9-main-rerate.out 2>&1 &
```

The original 9×9 arena phase took 9,200 s over 150 generations: 61 s per generation for five 20-game matches (the `arena_s` column of `docs/results/9x9/log.csv`). A re-rated round plays four matches plus 80 promotion games now and then, so expect about 2–3 hours. Follow it with `tail -f runs/9x9-main-rerate.out`. Each line looks like `rerate gen 37 | elo … vs 36:… …`. When it ends with `done: re-rated generations 1-150 of runs/9x9-main`, re-rate the 5×5 runs. The spec estimates about 20 minutes each; the original 5×5 arenas took 1.8 s per generation. Run them one after the other, so that they do not share the cores:

```bash
./build-blas/train --run runs/5x5-seed1 --rerate --set arena_opponents=2 > runs/5x5-seed1-rerate.out 2>&1
./build-blas/train --run runs/5x5-seed2 --rerate --set arena_opponents=2 > runs/5x5-seed2-rerate.out 2>&1
```

- [ ] **Step 4: Sanity checks before reading any number**

For each run directory `R` in `runs/9x9-main`, `runs/5x5-seed1` and `runs/5x5-seed2`:

```bash
R=runs/9x9-main
head -2 $R/rerate/results.csv                          # expected: header, then -1,0,80,<score>
wc -l < $R/rerate/log.csv                              # expected: 151 (5x5: 61)
awk -F, 'NR>1 { split($2, o, " "); if (o[1] != $1 - 1) bad++ } END { print bad + 0 }' $R/rerate/log.csv
                                                       # expected: 0 (every generation played g-1 first)
cat $R/rerate/references.csv                           # the bot, generation 0, then the promoted generations
awk -F, 'NR>1 && $1 >= 30 && (" " $2 " ") ~ / 0 / { n++ } END { print n + 0 }' $R/rerate/log.csv
                                                       # generations from 30 on that still played generation 0
```

None of these has a fixed expected value except the first three. Write down the bot's rating, the number of promoted generations and the generation-0 count; Step 7 reports them. Then confirm that the replay matches the live run where it should. The calibration and the generation-1 match use the same seeds as a fresh run, so this works as a cross-check:

```bash
cmp <(sed -n 2p $R/rerate/log.csv | cut -d, -f1-2) <(echo "1,0 -1")
```

Expected: no output. Generation 1's opponents are always `0 -1`: g−1 is generation 0, and the bot is either the strongest reference or the only extra left. Investigate any other output.

- [ ] **Step 5: Publish the files**

```bash
for pair in 9x9-main:9x9 5x5-seed1:5x5 5x5-seed2:5x5-seed2; do
  run=${pair%%:*}; dest=docs/results/${pair##*:}/rerate
  mkdir -p $dest
  cp runs/$run/rerate/{results,elo,references,log}.csv $dest/
  grep -E '^(rerate gen|done)' runs/$run-rerate.out > $dest/progress.txt
done
ls docs/results/9x9/rerate docs/results/5x5/rerate docs/results/5x5-seed2/rerate
```

Expected: each directory lists `elo.csv log.csv progress.txt references.csv results.csv`.

- [ ] **Step 6: Judge the 9×9 criterion on the re-rated curve**

The slope is the least-squares slope of `elo` against `generation` over the last third of the run, generations 101–150:

```bash
for f in docs/results/9x9/elo.csv docs/results/9x9/rerate/elo.csv; do
python3 - "$f" <<'PY'
import csv, sys
rows = [(int(r["generation"]), float(r["elo"])) for r in csv.DictReader(open(sys.argv[1]))]
last = max(g for g, _ in rows)
first = last - last // 3 + 1                      # 150 -> 101
pts = [(g, e) for g, e in rows if g >= first]
mg = sum(g for g, _ in pts) / len(pts)
me = sum(e for _, e in pts) / len(pts)
slope = sum((g - mg) * (e - me) for g, e in pts) / sum((g - mg) ** 2 for g, _ in pts)
print(f"{sys.argv[1]}: generations {first}-{last}: slope {slope:+.2f} Elo/generation")
PY
done
```

Expected first line, which checks the script against the published number: `docs/results/9x9/elo.csv: generations 101-150: slope +0.41 Elo/generation`. The second line is the new measurement. The Elo-slope part of the criterion is met if and only if it is positive. State its size as well: +0.4 Elo per generation is "positive" but carries no information.

Compare the re-rated ratings with the 100-game head-to-heads in `docs/results/9x9/h2h-*.txt`. Those are the independent check, and the reason the spec exists:

```bash
python3 - <<'PY'
import csv
def load(p): return {int(r["generation"]): float(r["elo"]) for r in csv.DictReader(open(p))}
old, new = load("docs/results/9x9/elo.csv"), load("docs/results/9x9/rerate/elo.csv")
h2h = [(150, 50, 676), (150, 90, 315), (150, 100, 200), (150, 120, 123), (90, 50, 363)]
print("match        h2h   ladder elo.csv   re-rated")
for a, b, gap in h2h:
    print(f"{a:3d} vs {b:3d}  {gap:+5d}   {old[a] - old[b]:+8.0f}   {new[a] - new[b]:+8.0f}")
PY
```

The re-rated gaps should be much closer to the h2h column than the ladder's. Two things count as "results look wrong" and send you to systematic debugging (the result is still reported, never hidden): a re-rated gap with the opposite sign to the h2h gap, or the re-rated `elo.csv` falling over generations 50–150 while the h2h column rises.

For the 5×5 runs, record the same quantities the original 5×5 criterion used, from the re-rated `elo.csv`: the mean Elo of the last 10 generations minus the mean of the first 10.

```bash
for f in docs/results/5x5/rerate/elo.csv docs/results/5x5-seed2/rerate/elo.csv; do
  python3 -c "
import csv, sys
e = {int(r['generation']): float(r['elo']) for r in csv.DictReader(open('$f'))}
g = sorted(k for k in e if k > 0)
first, last = g[:10], g[-10:]
print('$f', round(sum(e[k] for k in last) / 10 - sum(e[k] for k in first) / 10, 1))"
done
```

- [ ] **Step 7: Extend the 9×9 Elo chart**

```bash
python3 tools/plot_elo.py "9×9 elo.csv, ladder (gen 0 = 0)=docs/results/9x9/elo.csv" \
    "9×9 ladder without gen-0 pairs (gen 50 = 0)=docs/results/9x9/elo-without-gen0.csv" \
    "9×9 re-rated (gen 0 = 0)=docs/results/9x9/rerate/elo.csv" > docs/results/elo-9x9.svg
```

The third series is drawn in the first series' color, dashed. In `README.md`, make two changes. Replace the `elo-9x9.svg` command in the chart-commands block with the command above. Rewrite the `alt` text of the `elo-9x9.svg` image so it names all three curves and what each does, using the measured numbers.

In `docs/algorithm-explained.html`, the chart is inlined. Inside the `<figure>` that follows the paragraph starting `<p>Refitting the same table without the matches against generation 0`, replace everything from `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 360" role="img" aria-label="Elo by generation: 9×9 elo.csv` through the matching `</svg>` with the full contents of the new `docs/results/elo-9x9.svg`. (Today that inlined block is byte-identical to the old file, so `cmp` on the extracted block is a quick check.) Then rewrite the `<figcaption>`: "Three fits of the 9×9 run.", one sentence per curve with its final value at generation 150, and the existing sentence about the head-to-heads.

- [ ] **Step 8: Write up the results**

`docs/results/README.md`:
- Under the **Presets** table, add: "The reference runs used the ladder (`arena_opponents = 4` plus generation 0). The presets now choose opponents by rating (`arena_opponents = 2`, `bot_simulations = 100`, `calibration_games = 80`, `promote_threshold = 0.85`); the re-rated results below use those values."
- After the 9×9 head-to-head discussion, add a subsection `### 9×9 re-rated with rating-aware opponents` containing:
  - the command;
  - the bot's rating (from `references.csv`), the promoted generations, and how many generations from 30 on still played generation 0 (Step 4);
  - the Step 6 comparison table (match, h2h, ladder `elo.csv`, re-rated);
  - the re-rated rating at generations 50, 90, 100, 120 and 150;
  - the slope sentence: "A least-squares fit of `9x9/rerate/elo.csv` over generations 101–150 gives **<slope> Elo per generation**, so the Elo-slope part of the criterion is <met / not met> on the re-rated curve. On the ladder `elo.csv` it was +0.41 (met on its sign only)."
  - The heading "### 9×9: criterion not met (the eye check fails)" stays as it is. The eye check is unchanged, so the 9×9 criterion as a whole is still not met, and the subsection says so.
- In the 5×5 section, add the re-rated last-10 minus first-10 values next to the existing ones.
- In "What did not work", item 2: keep the original text, and add that the code has since been changed to choose opponents by rating, with one sentence giving the re-rated outcome.
- In "Files", add the three `rerate/` directories and what each file is.

`README.md`, Results section: in the 9×9 bullet, add one sentence with the re-rated slope and the generation 150 − generation 50 gap next to the h2h +676. Leave the other bullets alone.

`docs/algorithm-explained.md` and `.html` (identical text): after the paragraph ending "…and read some of the games.", add a paragraph headed `**Re-rated with informative opponents.**`. It gives the command, the re-rated rating of generation 150 and the 150 − 50 gap next to the h2h +676, the last-third slope next to the old +0.41, and the bot's rating. End it with one sentence on what the curve shows now. Every number must name its file.

Run: `python3 tools/check_citations.py`
Expected: `all 36 markdown citations resolve, and the html agrees`.

- [ ] **Step 9: Commit**

```bash
git add docs/results README.md docs/algorithm-explained.md docs/algorithm-explained.html
git commit -F - <<'EOF'
Re-rate the 9x9 and 5x5 reference runs with rating-aware opponents

<One paragraph: the 9x9 re-rated slope over generations 101-150 and the
verdict on it, next to the old +0.41; the 150-50 gap next to the h2h
+676; the bot's rating and the promoted generations. One sentence each
for the 5x5 runs. State plainly that the 9x9 criterion as a whole is
still not met because of the eye check.>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```
