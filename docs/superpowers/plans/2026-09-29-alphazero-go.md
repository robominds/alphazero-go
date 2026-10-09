# AlphaZero Go Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a from-scratch C++17 AlphaZero for Go (board 5×5–19×19, configurable ResNet depth/width) trained by self-play, with a checkpoint arena + Elo and a teaching explainer.

**Architecture:** One library `azgo` (rules, numeric backend, hand-written ResNet with backprop, batched PUCT MCTS, self-play, replay buffer, arena/Elo, config, persistence, trainer) and four executables (`train`, `arena`, `analyze`, `bench`). Training runs synchronous generations in one process — threaded self-play → SGD → arena/Elo → atomic checkpoint — deterministic for a seed regardless of thread count.

**Tech Stack:** C++17, CMake ≥ 3.16, C++ standard library + `std::thread`; optional BLAS (Apple Accelerate / OpenBLAS) behind `-DAZ_USE_BLAS=ON`; plain `assert`-based test executables run by CTest; Python 3 standard library for the docs tooling only.

**Spec:** `docs/superpowers/specs/2026-09-29-alphazero-go-design.md`

## Global Constraints

- C++17, CMake ≥ 3.16, default build type Release. No dependencies beyond the C++ standard library, except optional BLAS behind CMake option `AZ_USE_BLAS` (default OFF): Accelerate on macOS, OpenBLAS elsewhere.
- Namespace `azgo`; public headers in `include/azgo/`, sources in `src/`, apps in `apps/`, tests in `tests/`.
- Every public header matches the **Interface Contract** at the end of this plan exactly (names, signatures, semantics). Private members marked implementation-defined are the owning task's choice.
- Board size is a runtime value in [5, 19]; nothing may be sized by a compile-time board size except the 21×21 padded board array.
- Tests: one assert-based executable per component, `tests/test_<component>.cpp`, compiled with `-UNDEBUG` so asserts fire in Release. No test framework.
- All randomness through `azgo::Rng`; never `std::random_device`, `std::mt19937`, or `std::*_distribution`. Independent streams are seeded with `deriveSeed(...)`.
- Determinism: self-play, arena and training output must be identical for a given seed and backend regardless of worker count.
- Values are from the side to move's perspective, in [-1, 1]. Rules: no suicide, positional superko, Tromp-Taylor area scoring, komi from config (default 7.5), game ends on two consecutive passes or at `2·n²` moves.
- Each header opens with a short teaching comment block explaining the concept it implements.
- Code style follows `alphazero-tictactoe`: camelCase functions, PascalCase types, `trailing_` members, `kConstants`, 4-space indent.
- Commits: author Mark Castelluccio <markacastelluccio@gmail.com> (set as repo-local git config in Task 1); imperative-sentence subjects without conventional-commit prefixes; body explains why; every message ends with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Immediate pass-pass game** (an untrained network may pass at move 1): the game must end after 2 plies, be recorded with 2 positions, scored as a White win by komi, with z = −1 at Black's position and +1 at White's. Test: Task 14.
2. **Integer komi producing an exact tie**: `resultForBlack() == 0`, z = 0 at every position, SGF `RE[0]`, arena scores it 0.5. Tests: Tasks 3, 15, 18, 19.
3. **Leftovers from a crash**: a garbage `state.bin.tmp` next to a good `state.bin` must be ignored and overwritten; a truncated `state.bin` or checkpoint must fail with a message naming the file, not crash. Tests: Tasks 9 and 20.
4. **Bad user moves in `analyze`**: occupied point, superko retake, suicide, `Z99`, `I5` → `error: move K (<text>): <reason>` and exit 1, never an assert. Test: Task 22.
5. **Hand-edited config files**: CRLF line endings, trailing `# comments`, blank lines, spaces around `=` all accepted; a duplicated key and `arena_games = 7` rejected with file:line. Test: Task 16.

## File Structure

```
CMakeLists.txt                 Task 1 (globs src/*.cpp and tests/test_*.cpp); apps added in Tasks 21-22
.gitignore                     exists (build/, runs/, .worktrees/, .DS_Store)
README.md                      Task 1 skeleton (build + Authorship); completed in Task 25
include/azgo/rng.hpp           Task 1   deterministic RNG, seed derivation, Dirichlet
include/azgo/board.hpp         Task 2   single-position Go rules, Zobrist hash, scoring, GTP coords
include/azgo/game.hpp          Task 3   move sequence, superko, passes, termination, result
include/azgo/symmetry.hpp      Task 4   8 dihedral transforms
include/azgo/encode.hpp        Task 4   input planes
include/azgo/backend.hpp       Task 5   gemm/im2col interface, naive backend, makeBackend
include/azgo/layers.hpp        Task 6   Conv2d, BatchNorm2d, Linear, ReLU (forward + backward)
include/azgo/network.hpp       Task 7   NetConfig, Workspace, Network (ResNet, loss, trainStep)
include/azgo/optimizer.hpp     Task 8   SgdMomentum, LrSchedule
include/azgo/serialize.hpp     Task 9   little-endian binary I/O, atomic file write
include/azgo/checkpoint.hpp    Task 9   network save/load
include/azgo/evaluator.hpp     Task 10  Evaluator seam, LocalEvaluator, UniformEvaluator
include/azgo/mcts.hpp          Task 11  PUCT tree (Task 12 batching, Task 13 reuse/noise/selection)
include/azgo/parallel.hpp      Task 14  parallelFor
include/azgo/selfplay.hpp      Task 14  GameRecord, playSelfPlayGame
include/azgo/replay_buffer.hpp Task 15  game-window buffer, sampling, augmentation
include/azgo/config.hpp        Task 16  Config, parser, overrides
include/azgo/elo.hpp           Task 17  ResultsTable, Bradley-Terry, Wilson interval
include/azgo/arena.hpp         Task 18  playMatch, ladder, gating
include/azgo/sgf.hpp           Task 19  SGF write/read
include/azgo/run_state.hpp     Task 20  TrainingState save/load, run paths
include/azgo/trainer.hpp       Task 21  generation loop
src/<same names>.cpp           matching implementations; src/backend_blas.cpp (Task 5)
apps/cli.hpp                   Task 21  ArgParser
apps/train.cpp                 Task 21
apps/arena.cpp, analyze.cpp, bench.cpp   Task 22
configs/5x5.cfg, 9x9.cfg, 19x19.cfg      Task 16 (retuned in Task 23)
tests/test_*.cpp               one per component, in the task that owns it
tests/integration_smoke.sh     Task 22
tools/check_citations.py       Task 24
tools/plot_elo.py              Task 25  elo.csv -> inline SVG (standard library only)
docs/algorithm-explained.md    Task 24
docs/algorithm-explained.html  Task 25
docs/results/                  Task 23  copied CSVs and notes from measured runs
```

## Conventions for Every Task

**Build and test commands** (from the repo root):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure -R test_board   # one component
ctest --test-dir build --output-on-failure                 # everything
```

BLAS variant (Tasks 5+ verify both when relevant):

```bash
cmake -S . -B build-blas -DCMAKE_BUILD_TYPE=Release -DAZ_USE_BLAS=ON
cmake --build build-blas -j && ctest --test-dir build-blas --output-on-failure
```

Because `CMakeLists.txt` globs with `CONFIGURE_DEPENDS`, a new `src/*.cpp` or `tests/test_*.cpp` is picked up by re-running `cmake --build build` — no CMake edit needed. A test whose source exists but whose implementation is missing fails at **link** time; that is the expected "fails" state in step 2 of a TDD cycle.

**Test file shape:**

```cpp
#include <cassert>
#include <cmath>
#include <cstdio>
#include "azgo/<component>.hpp"

using namespace azgo;

static bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

static void test_something() {
    // arrange, act, assert
}

int main() {
    test_something();
    std::printf("test_<component>: all tests passed\n");
    return 0;
}
```

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

### Task 1: Project scaffold + deterministic RNG

**Files:**
- Create: `CMakeLists.txt`, `README.md`, `include/azgo/rng.hpp`, `src/rng.cpp`
- Test: `tests/test_rng.cpp`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `uint64_t azgo::splitmix64(uint64_t& x)`; `uint64_t azgo::deriveSeed(uint64_t a, uint64_t b = 0, uint64_t c = 0, uint64_t d = 0)`; `class azgo::Rng` with `explicit Rng(uint64_t seed = 0)`, `uint64_t next()`, `double uniform()`, `int below(int n)`, `double normal()`, `double gamma(double shape)`, `std::array<uint64_t, 4> state() const`, `void setState(const std::array<uint64_t, 4>&)`; `std::vector<double> azgo::dirichlet(Rng& rng, double alpha, int n)`. Build: static library target `azgo` (globs `src/*.cpp`, drops `src/backend_blas.cpp` unless `AZ_USE_BLAS`), CMake option `AZ_USE_BLAS` (defines `AZ_USE_BLAS` publicly when ON), one CTest executable per `tests/test_*.cpp` compiled with `-UNDEBUG` and `AZGO_SOURCE_DIR="<repo root>"`.

- [ ] **Step 1: Set the repo-local commit author**

The repository already exists (spec commit, `.gitignore`). From the repo root:

```bash
git config user.name "Mark Castelluccio"
git config user.email "markacastelluccio@gmail.com"
git config user.name && git config user.email
```

Expected output:
```
Mark Castelluccio
markacastelluccio@gmail.com
```

- [ ] **Step 2: Create the CMake project and the README skeleton**

Create `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(alphazero_go CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(AZ_USE_BLAS "Use BLAS for gemm (Accelerate on macOS, OpenBLAS elsewhere)" OFF)

find_package(Threads REQUIRED)

# ---- Library ---------------------------------------------------------------
# Every src/*.cpp belongs to the library, so a task that adds a component adds
# no CMake lines. CONFIGURE_DEPENDS makes `cmake --build` re-glob when files
# appear. The BLAS backend is the one exception: it only compiles against a
# BLAS header, so it is dropped unless AZ_USE_BLAS is on. REMOVE_ITEM is a
# no-op while that file does not exist yet.
file(GLOB AZGO_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
if(NOT AZ_USE_BLAS)
  list(REMOVE_ITEM AZGO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/backend_blas.cpp)
endif()

add_library(azgo STATIC ${AZGO_SOURCES})
target_include_directories(azgo PUBLIC include)
target_compile_options(azgo PRIVATE -Wall -Wextra)
target_link_libraries(azgo PUBLIC Threads::Threads)

if(AZ_USE_BLAS)
  if(APPLE)
    # Accelerate ships with macOS. ACCELERATE_NEW_LAPACK selects the current
    # cblas headers instead of the deprecated ones.
    target_link_libraries(azgo PUBLIC "-framework Accelerate")
    target_compile_definitions(azgo PUBLIC AZ_USE_BLAS ACCELERATE_NEW_LAPACK)
  else()
    find_package(BLAS REQUIRED)
    find_path(CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas)
    if(NOT CBLAS_INCLUDE_DIR)
      message(FATAL_ERROR "AZ_USE_BLAS=ON but cblas.h was not found (install OpenBLAS headers)")
    endif()
    target_include_directories(azgo PRIVATE ${CBLAS_INCLUDE_DIR})
    target_link_libraries(azgo PUBLIC ${BLAS_LIBRARIES})
    target_compile_definitions(azgo PUBLIC AZ_USE_BLAS)
  endif()
endif()

# ---- Tests -----------------------------------------------------------------
# One executable per tests/test_<component>.cpp, named after the file.
# -UNDEBUG: the default build type is Release, which defines NDEBUG and would
# compile every assert() in a test to nothing. Stripping it here keeps the
# tests meaningful without slowing the library down.
# AZGO_SOURCE_DIR lets tests find repository files such as configs/.
enable_testing()

file(GLOB AZGO_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp)
foreach(test_src ${AZGO_TESTS})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE azgo)
  target_compile_options(${test_name} PRIVATE -UNDEBUG)
  target_compile_definitions(${test_name} PRIVATE AZGO_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# ---- Apps ------------------------------------------------------------------
# train is added in Task 21; arena, analyze, bench and the integration smoke
# test (tests/integration_smoke.sh) in Task 22.
```

Create `README.md`:

````markdown
# AlphaZero Go

A from-scratch C++17 implementation of AlphaZero — self-play plus PUCT
Monte Carlo tree search guided by a residual neural network — for the game
of Go on boards from 5×5 to 19×19. It is a **teaching example**, the
successor to `alphazero-tictactoe`: every moving part (rules, convolutions,
backpropagation, search, self-play, training, evaluation) is written
directly, with no machine-learning library, and explained in
`docs/algorithm-explained.md`. It is not meant to be a strong Go engine.

Status: under construction — the sections below grow as components land.

## Build

Requires CMake ≥ 3.16 and a C++17 compiler. The default build uses only the
C++ standard library:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Optionally, matrix multiplication can use BLAS (Apple Accelerate on macOS,
OpenBLAS elsewhere):

```bash
cmake -S . -B build-blas -DCMAKE_BUILD_TYPE=Release -DAZ_USE_BLAS=ON
cmake --build build-blas -j
ctest --test-dir build-blas --output-on-failure
```

## Authorship

Written by Mark Castelluccio. Designed and implemented with the assistance of
Claude (Anthropic), an AI assistant; commits carry a Co-Authored-By trailer.
````

- [ ] **Step 3: Write the failing test**

Create `tests/test_rng.cpp`:

```cpp
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include "azgo/rng.hpp"

using namespace azgo;

static bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

static void test_splitmix64_known_outputs() {
    // Reference values of the published splitmix64 algorithm from x = 0.
    uint64_t x = 0;
    assert(splitmix64(x) == 0xE220A8397B1DCDAFULL);
    assert(splitmix64(x) == 0x6E789E6AA1B965F4ULL);
    assert(splitmix64(x) == 0x06C45D188009454FULL);
    assert(x == 3 * 0x9E3779B97F4A7C15ULL);  // advanced by the Weyl increment each call
}

static void test_same_seed_same_sequence() {
    Rng a(42), b(42);
    for (int i = 0; i < 1000; ++i) assert(a.next() == b.next());
}

static void test_different_seeds_differ() {
    Rng a(1), b(2);
    int equal = 0;
    for (int i = 0; i < 100; ++i) equal += (a.next() == b.next());
    assert(equal == 0);
}

static void test_state_round_trip_resumes_sequence() {
    Rng a(7);
    for (int i = 0; i < 10; ++i) a.next();
    std::array<uint64_t, 4> saved = a.state();
    std::vector<uint64_t> expected;
    for (int i = 0; i < 50; ++i) expected.push_back(a.next());

    Rng b(12345);  // different seed, then overwritten
    b.setState(saved);
    for (int i = 0; i < 50; ++i) assert(b.next() == expected[i]);
    assert(b.state() == a.state());
}

static void test_uniform_range_and_mean() {
    Rng rng(3);
    const int draws = 100000;
    double sum = 0.0;
    for (int i = 0; i < draws; ++i) {
        double u = rng.uniform();
        assert(u >= 0.0 && u < 1.0);
        sum += u;
    }
    // Standard error of the mean is sqrt(1/12 / 1e5) ~ 0.0009.
    assert(near(sum / draws, 0.5, 0.005));
}

static void test_below_is_uniform() {
    Rng rng(4);
    const int n = 7, draws = 70000;
    std::vector<int> counts(n, 0);
    for (int i = 0; i < draws; ++i) {
        int k = rng.below(n);
        assert(k >= 0 && k < n);
        ++counts[k];
    }
    const double expected = static_cast<double>(draws) / n;  // 10000
    for (int c : counts) assert(std::fabs(c - expected) <= 0.05 * expected);

    Rng one(5);
    for (int i = 0; i < 100; ++i) assert(one.below(1) == 0);
}

static void test_normal_moments() {
    Rng rng(5);
    const int draws = 200000;
    double sum = 0.0, sumSq = 0.0;
    for (int i = 0; i < draws; ++i) {
        double x = rng.normal();
        assert(std::isfinite(x));
        sum += x;
        sumSq += x * x;
    }
    double mean = sum / draws;
    double var = sumSq / draws - mean * mean;
    assert(near(mean, 0.0, 0.01));
    assert(near(var, 1.0, 0.02));
}

static void test_gamma_moments() {
    // Gamma(k, 1) has mean k and variance k. Small shapes (the Dirichlet
    // alpha on 19x19 is ~0.03) go through the shape < 1 boost path.
    const double shapes[] = {0.03, 0.5, 2.0, 7.5};
    Rng rng(6);
    for (double k : shapes) {
        const int draws = 200000;
        double sum = 0.0, sumSq = 0.0;
        for (int i = 0; i < draws; ++i) {
            double x = rng.gamma(k);
            assert(x >= 0.0 && std::isfinite(x));
            sum += x;
            sumSq += x * x;
        }
        double mean = sum / draws;
        double var = sumSq / draws - mean * mean;
        // Allow 5 standard errors. SE(mean) = sqrt(k / draws); the sample
        // variance of Gamma(k) has SE sqrt((mu4 - k^2) / draws) with fourth
        // central moment mu4 = 3k^2 + 6k. Tiny shapes are very skewed, so
        // their variance estimate is noisy (~3% SE at k = 0.03).
        double seMean = std::sqrt(k / draws);
        double seVar = std::sqrt((2.0 * k * k + 6.0 * k) / draws);
        assert(std::fabs(mean - k) <= 5.0 * seMean);
        assert(std::fabs(var - k) <= 5.0 * seVar);
    }
}

static void test_dirichlet() {
    Rng rng(8);
    const int n = 5, samples = 20000;
    std::vector<double> mean(n, 0.0);
    for (int s = 0; s < samples; ++s) {
        std::vector<double> d = dirichlet(rng, 0.3, n);
        assert(static_cast<int>(d.size()) == n);
        double total = 0.0;
        for (int i = 0; i < n; ++i) {
            assert(d[i] >= 0.0);
            total += d[i];
            mean[i] += d[i] / samples;
        }
        assert(near(total, 1.0, 1e-9));
    }
    for (double m : mean) assert(near(m, 1.0 / n, 0.01));

    // 19x19-sized noise with the small alpha used there.
    std::vector<double> big = dirichlet(rng, 0.03, 362);
    double total = 0.0;
    for (double v : big) { assert(v >= 0.0); total += v; }
    assert(near(total, 1.0, 1e-9));

    // With alpha = 1e-4 every Gamma draw usually underflows to 0, which
    // exercises the one-hot fallback; the result must still be a distribution.
    for (int s = 0; s < 100; ++s) {
        std::vector<double> d = dirichlet(rng, 1e-4, 5);
        double t = 0.0;
        for (double v : d) { assert(v >= 0.0); t += v; }
        assert(near(t, 1.0, 1e-9));
    }
}

static void test_derive_seed() {
    assert(deriveSeed(1, 2, 3, 4) == deriveSeed(1, 2, 3, 4));
    assert(deriveSeed(1, 2, 3, 4) != deriveSeed(4, 3, 2, 1));
    assert(deriveSeed(1, 2, 3, 4) != deriveSeed(2, 1, 3, 4));
    assert(deriveSeed(1, 2) != deriveSeed(2, 1));
    assert(deriveSeed(0) != deriveSeed(1));
    assert(deriveSeed(7) == deriveSeed(7, 0, 0, 0));  // defaults are zeros
    // Neighbouring game indices must give unrelated streams.
    Rng a(deriveSeed(99, 1, 0, 0)), b(deriveSeed(99, 1, 0, 1));
    assert(a.next() != b.next());
}

int main() {
    test_splitmix64_known_outputs();
    test_same_seed_same_sequence();
    test_different_seeds_differ();
    test_state_round_trip_resumes_sequence();
    test_uniform_range_and_mean();
    test_below_is_uniform();
    test_normal_moments();
    test_gamma_moments();
    test_dirichlet();
    test_derive_seed();
    std::printf("test_rng: all tests passed\n");
    return 0;
}
```

- [ ] **Step 4: Run the build to verify it fails**

Run: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
Expected: FAIL at generate time, because the library has no sources yet:
```
CMake Error at CMakeLists.txt:27 (add_library):
  No SOURCES given to target: azgo
```

- [ ] **Step 5: Write the header**

Create `include/azgo/rng.hpp`:

```cpp
// Deterministic randomness.
//
// AlphaZero is full of randomness: weight initialization, Dirichlet noise at
// the search root, sampling moves by visit count, replay-buffer sampling and
// symmetry augmentation. To make a training run reproducible from one seed,
// every random number in this project comes from this small, fully specified
// generator (xoshiro256**) rather than std::mt19937 + std::*_distribution,
// whose outputs differ between standard libraries. Independent streams (one
// per game, per phase, per generation) get their own seed via deriveSeed.
#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace azgo {

// Advances x and returns the next splitmix64 output.
uint64_t splitmix64(uint64_t& x);

// Mixes up to four integers into one well-distributed 64-bit seed.
uint64_t deriveSeed(uint64_t a, uint64_t b = 0, uint64_t c = 0, uint64_t d = 0);

// xoshiro256** generator, seeded by running splitmix64 on `seed`.
class Rng {
public:
    explicit Rng(uint64_t seed = 0);
    uint64_t next();
    double uniform();              // [0, 1), 53 random bits
    int below(int n);              // uniform in [0, n); asserts n > 0; unbiased (rejection)
    double normal();               // standard normal via Box-Muller (no cached second value)
    double gamma(double shape);    // Gamma(shape, 1); Marsaglia-Tsang, shape < 1 via boost
    std::array<uint64_t, 4> state() const;
    void setState(const std::array<uint64_t, 4>& s);
private:
    std::array<uint64_t, 4> s_;
};

// Symmetric Dirichlet sample of length n with concentration alpha > 0.
std::vector<double> dirichlet(Rng& rng, double alpha, int n);

} // namespace azgo
```

- [ ] **Step 6: Write the implementation**

Create `src/rng.cpp`:

```cpp
#include "azgo/rng.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <initializer_list>

namespace azgo {

namespace {

uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

constexpr double kTwoPi = 6.283185307179586476925286766559;

} // namespace

uint64_t splitmix64(uint64_t& x) {
    // Weyl sequence step followed by an invertible bit mixer (Steele et al.).
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

uint64_t deriveSeed(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    // Fold the inputs in one at a time: h = mix(h ^ v). Because each step
    // mixes before the next input arrives, the result depends on the order
    // of the inputs, so (1, 2, 3, 4) and (4, 3, 2, 1) name different streams.
    uint64_t h = 0x243F6A8885A308D3ULL;  // arbitrary nonzero start (digits of pi)
    for (uint64_t v : {a, b, c, d}) {
        uint64_t x = h ^ v;
        h = splitmix64(x);
    }
    return h;
}

Rng::Rng(uint64_t seed) {
    // xoshiro's state must not be all zero; four consecutive splitmix64
    // outputs never are, and they spread even small seeds (0, 1, 2) apart.
    uint64_t x = seed;
    for (auto& w : s_) w = splitmix64(x);
}

uint64_t Rng::next() {
    // xoshiro256** (Blackman & Vigna, 2018).
    const uint64_t result = rotl(s_[1] * 5, 7) * 9;
    const uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
}

double Rng::uniform() {
    // The top 53 bits fill a double's mantissa exactly: k / 2^53, k < 2^53.
    return static_cast<double>(next() >> 11) * 0x1.0p-53;
}

int Rng::below(int n) {
    assert(n > 0);
    // Lemire's multiply-shift: the high 32 bits of x * n are uniform in
    // [0, n) except for a small bias, which we remove by rejecting the
    // (2^32 mod n) lowest products.
    const uint32_t range = static_cast<uint32_t>(n);
    uint64_t m = (next() >> 32) * range;
    uint32_t low = static_cast<uint32_t>(m);
    if (low < range) {
        const uint32_t threshold = (0u - range) % range;  // 2^32 mod n
        while (low < threshold) {
            m = (next() >> 32) * range;
            low = static_cast<uint32_t>(m);
        }
    }
    return static_cast<int>(m >> 32);
}

double Rng::normal() {
    const double u1 = 1.0 - uniform();  // (0, 1], so log(u1) is finite
    const double u2 = uniform();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(kTwoPi * u2);
}

double Rng::gamma(double shape) {
    assert(shape > 0.0);
    if (shape < 1.0) {
        // Boost: if X ~ Gamma(shape + 1) and U ~ Uniform(0, 1], then
        // X * U^(1/shape) ~ Gamma(shape). Marsaglia-Tsang needs shape >= 1.
        const double u = 1.0 - uniform();
        return gamma(shape + 1.0) * std::pow(u, 1.0 / shape);
    }
    // Marsaglia & Tsang (2000): transform a normal draw x into d * (1 + c x)^3
    // and accept with a squeeze test (cheap) or the exact log test.
    const double d = shape - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
        double x, v;
        do {
            x = normal();
            v = 1.0 + c * x;
        } while (v <= 0.0);
        v = v * v * v;
        const double u = uniform();
        const double x2 = x * x;
        if (u < 1.0 - 0.0331 * x2 * x2) return d * v;
        if (u > 0.0 && std::log(u) < 0.5 * x2 + d * (1.0 - v + std::log(v))) return d * v;
    }
}

std::array<uint64_t, 4> Rng::state() const {
    return s_;
}

void Rng::setState(const std::array<uint64_t, 4>& s) {
    s_ = s;
}

std::vector<double> dirichlet(Rng& rng, double alpha, int n) {
    assert(alpha > 0.0 && n > 0);
    // A Dirichlet sample is a vector of independent Gamma(alpha) draws
    // divided by their sum.
    std::vector<double> x(static_cast<size_t>(n));
    double sum = 0.0;
    for (double& v : x) {
        v = rng.gamma(alpha);
        sum += v;
    }
    if (!(sum > 0.0)) {
        // With a tiny alpha every Gamma draw can underflow to 0. The limit of
        // Dirichlet(alpha) as alpha -> 0 puts all mass on one component, so
        // return a one-hot vector at a uniformly random index.
        std::fill(x.begin(), x.end(), 0.0);
        x[static_cast<size_t>(rng.below(n))] = 1.0;
        return x;
    }
    for (double& v : x) v /= sum;
    return x;
}

} // namespace azgo
```

- [ ] **Step 7: Run tests to verify they pass (both build variants)**

Run:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure -R test_rng
```
Expected (older CTest versions word the summary `100% tests passed, 0 tests failed out of N`): the library compiles with no warnings under `-Wall -Wextra`, and
```
1/1 Test #1: test_rng .........................   Passed
100% tests passed out of 1
```
(`./build/test_rng` prints `test_rng: all tests passed`.)

Then check that the BLAS switch configures even though `src/backend_blas.cpp` does not exist yet (the `REMOVE_ITEM`/glob must tolerate that):
```bash
cmake -S . -B build-blas -DCMAKE_BUILD_TYPE=Release -DAZ_USE_BLAS=ON
cmake --build build-blas -j && ctest --test-dir build-blas --output-on-failure
```
Expected: configures (on macOS it links Accelerate; elsewhere it needs OpenBLAS with `cblas.h`) and `100% tests passed out of 1`.

- [ ] **Step 8: Commit**

```bash
git add CMakeLists.txt README.md include/azgo/rng.hpp src/rng.cpp tests/test_rng.cpp
git commit -F - <<'EOF'
Scaffold the CMake project and add a deterministic RNG

Training must be reproducible from one seed regardless of platform or
thread count, and std::mt19937 with std::*_distribution gives different
numbers on different standard libraries. Rng is xoshiro256** seeded by
splitmix64, with our own uniform, unbiased below(), Box-Muller normal,
Marsaglia-Tsang gamma and Dirichlet, so every later component draws
randomness from one fully specified source. deriveSeed gives each game
and phase its own independent stream.

CMake globs src/ and tests/ so later tasks only add files; test targets
are built with -UNDEBUG so asserts fire in the default Release build.
The README skeleton records authorship from the start.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 2: Board

**Files:**
- Create: `include/azgo/board.hpp`, `src/board.cpp`
- Test: `tests/test_board.cpp`

**Interfaces:**
- Consumes: `uint64_t splitmix64(uint64_t& x)` (Task 1) to generate the Zobrist keys.
- Produces: `kMinBoardSize`, `kMaxBoardSize`, `kStride`, `kCells`; `enum class Color : uint8_t { Empty, Black, White, Off }`; `Color opponent(Color)`; `class Board` with `explicit Board(int n)`, `size()`, `area()`, `passAction()`, `numActions()`, `toMove()`, `Color at(int) const`, `int stoneCount(Color) const`, `int libertiesAt(int) const`, `bool isLegal(int) const`, `int play(int)`, `uint64_t hashAfter(int) const`, `uint64_t hash() const`, `uint64_t computeHashFromScratch() const`, `int areaScore() const`, `void exportStones(uint8_t*) const`, `std::string toString() const`, `void setStone(int, Color)`, `void setToMove(Color)`, `bool operator==(const Board&) const`; free functions `Board boardFromRows(const std::vector<std::string>&, Color)`, `std::string actionToString(int action, int n)`, `int parseAction(const std::string& text, int n)`. Private helpers (implementation-defined): `int countLiberties(int p, int limit) const`, `int removeGroup(int p)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_board.cpp`. Positions are drawn with `boardFromRows` (row 0 on top, `X` Black, `O` White); `pt(row, col)` converts to a point index. Every expected number below was counted by hand from the diagram next to it.

```cpp
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>
#include "azgo/board.hpp"

using namespace azgo;

// Point index from (row, col) on an n x n board; row 0 is the top row.
static int pt(int row, int col, int n = 5) { return row * n + col; }

static void test_empty_board() {
    for (int n : {5, 9, 13, 19}) {
        Board b(n);
        assert(b.size() == n);
        assert(b.area() == n * n);
        assert(b.passAction() == n * n);
        assert(b.numActions() == n * n + 1);
        assert(b.toMove() == Color::Black);
        assert(b.stoneCount(Color::Black) == 0);
        assert(b.stoneCount(Color::White) == 0);
        assert(b.stoneCount(Color::Empty) == n * n);
        for (int a = 0; a < b.numActions(); ++a) assert(b.isLegal(a));
        assert(b.hash() == b.computeHashFromScratch());
        assert(b.areaScore() == 0);
    }
    assert(opponent(Color::Black) == Color::White);
    assert(opponent(Color::White) == Color::Black);
}

static void test_play_alternates_and_pass() {
    Board b(5);
    assert(b.play(pt(2, 2)) == 0);
    assert(b.at(pt(2, 2)) == Color::Black);
    assert(b.toMove() == Color::White);
    assert(!b.isLegal(pt(2, 2)));             // occupied
    uint64_t h = b.hash();
    assert(b.play(b.passAction()) == 0);
    assert(b.toMove() == Color::Black);
    assert(b.hash() == h);                    // pass changes no stones
}

static void test_capture_corner() {
    Board b = boardFromRows({"OX...",
                             ".....",
                             ".....",
                             ".....",
                             "....."}, Color::Black);
    assert(b.libertiesAt(pt(0, 0)) == 1);
    assert(b.play(pt(1, 0)) == 1);
    assert(b.at(pt(0, 0)) == Color::Empty);
    assert(b.stoneCount(Color::White) == 0);
    assert(b.stoneCount(Color::Black) == 2);
    assert(b.hash() == b.computeHashFromScratch());
}

static void test_capture_edge() {
    Board b = boardFromRows({".XOX.",
                             ".....",
                             ".....",
                             ".....",
                             "....."}, Color::Black);
    assert(b.play(pt(1, 2)) == 1);
    assert(b.at(pt(0, 2)) == Color::Empty);
    assert(b.hash() == b.computeHashFromScratch());
}

static void test_capture_center() {
    Board b = boardFromRows({".....",
                             "..X..",
                             ".XOX.",
                             ".....",
                             "....."}, Color::Black);
    assert(b.play(pt(3, 2)) == 1);
    assert(b.at(pt(2, 2)) == Color::Empty);
    assert(b.libertiesAt(pt(3, 2)) == 4);     // the captured point is a liberty again
    assert(b.hash() == b.computeHashFromScratch());
}

static void test_capture_multi_stone_group() {
    Board b = boardFromRows({".....",
                             ".XX..",
                             "XOO..",
                             ".XX..",
                             "....."}, Color::Black);
    assert(b.libertiesAt(pt(2, 1)) == 1);     // the whole white group shares one liberty
    assert(b.libertiesAt(pt(2, 2)) == 1);
    assert(b.play(pt(2, 3)) == 2);
    assert(b.at(pt(2, 1)) == Color::Empty && b.at(pt(2, 2)) == Color::Empty);
    assert(b.stoneCount(Color::White) == 0);
    assert(b.hash() == b.computeHashFromScratch());
}

static void test_capture_two_groups_at_once() {
    Board b = boardFromRows({"XO.OX",
                             ".X.X.",
                             ".....",
                             ".....",
                             "....."}, Color::Black);
    assert(b.play(pt(0, 2)) == 2);
    assert(b.at(pt(0, 1)) == Color::Empty && b.at(pt(0, 3)) == Color::Empty);
    assert(b.stoneCount(Color::White) == 0);
    assert(b.hash() == b.computeHashFromScratch());
}

static void test_suicide_illegal() {
    // Single-point eye: White may not play into Black's corner eye.
    Board eye = boardFromRows({".X...",
                               "X....",
                               ".....",
                               ".....",
                               "....."}, Color::White);
    assert(!eye.isLegal(pt(0, 0)));
    assert(eye.isLegal(eye.passAction()));
    // Black may fill its own eye (legal, just bad).
    eye.setToMove(Color::Black);
    assert(eye.isLegal(pt(0, 0)));

    // Multi-stone suicide: White (0,0) has one liberty at (0,1); playing
    // there joins the stones into a group with no liberties and captures nothing.
    Board multi = boardFromRows({"O.X..",
                                 "XX...",
                                 ".....",
                                 ".....",
                                 "....."}, Color::White);
    assert(!multi.isLegal(pt(0, 1)));
}

static void test_capture_that_looks_like_suicide_is_legal() {
    // Black's (0,0) is surrounded by white stones, but both are in atari
    // with (0,0) as their last liberty, so the move captures them.
    Board b = boardFromRows({".OX..",
                             "OX...",
                             "X....",
                             ".....",
                             "....."}, Color::Black);
    assert(b.isLegal(pt(0, 0)));
    // The same point is suicide for White: it would take its own groups'
    // last liberty without capturing anything.
    Board w = b;
    w.setToMove(Color::White);
    assert(!w.isLegal(pt(0, 0)));

    assert(b.play(pt(0, 0)) == 2);
    assert(b.at(pt(0, 1)) == Color::Empty && b.at(pt(1, 0)) == Color::Empty);
    assert(b.libertiesAt(pt(0, 0)) == 2);
    assert(b.hash() == b.computeHashFromScratch());
}

static void test_liberties() {
    Board b = boardFromRows({"X...X",
                             ".....",
                             "..X..",
                             ".....",
                             "..X.X"}, Color::Black);
    assert(b.libertiesAt(pt(0, 0)) == 2);     // corner
    assert(b.libertiesAt(pt(0, 4)) == 2);     // corner: the padded cells right of col 4 are Off
    assert(b.libertiesAt(pt(4, 4)) == 2);     // corner
    assert(b.libertiesAt(pt(4, 2)) == 3);     // edge
    assert(b.libertiesAt(pt(2, 2)) == 4);     // center
    assert(b.libertiesAt(pt(1, 1)) == 0);     // empty point

    // Shared liberties count once: in this L, (2,2) touches two stones.
    Board l = boardFromRows({".....",
                             ".XX..",
                             ".X...",
                             ".....",
                             "....."}, Color::Black);
    assert(l.libertiesAt(pt(1, 1)) == 7);
    assert(l.libertiesAt(pt(2, 1)) == 7);     // same group, same answer

    // An adjacent opponent stone takes a liberty away.
    l.setStone(pt(0, 1), Color::White);
    assert(l.libertiesAt(pt(1, 1)) == 6);
    assert(l.libertiesAt(pt(0, 1)) == 2);

    // Unused padded cells are Off on a 9x9 board too.
    Board nine(9);
    nine.setStone(pt(0, 8, 9), Color::White);
    nine.setStone(pt(8, 8, 9), Color::White);
    assert(nine.libertiesAt(pt(0, 8, 9)) == 2);
    assert(nine.libertiesAt(pt(8, 8, 9)) == 2);
}

static void test_hash_after_matches_play() {
    Board b = boardFromRows({"XO.OX",
                             ".X.X.",
                             ".....",
                             "..O..",
                             "....."}, Color::Black);
    for (int a = 0; a < b.numActions(); ++a) {
        if (!b.isLegal(a)) continue;
        Board next = b;
        next.play(a);
        assert(b.hashAfter(a) == next.hash());
        assert(next.hash() == next.computeHashFromScratch());
    }
    assert(b.hashAfter(b.passAction()) == b.hash());

    // Capturing move (two groups) and non-capturing move, explicitly.
    Board capture = b;
    capture.play(pt(0, 2));
    assert(b.hashAfter(pt(0, 2)) == capture.hash());
    // The same stones reached by setup have the same hash as by play.
    Board same = boardFromRows({"X.X.X",
                                ".X.X.",
                                ".....",
                                "..O..",
                                "....."}, Color::White);
    assert(capture.hash() == same.hash());
    assert(capture == same);
}

static void test_area_score() {
    assert(Board(5).areaScore() == 0);

    Board one(5);
    one.setStone(pt(2, 2), Color::Black);
    assert(one.areaScore() == 25);            // 1 stone + 24 empty points reaching only Black

    Board two = one;
    two.setStone(pt(0, 0), Color::White);
    assert(two.areaScore() == 0);             // the one empty region touches both: neutral

    // Dame: column 3 touches both walls and counts for nobody.
    // Black: 5 stones + 10 territory = 15. White: 5 stones. Score 10.
    Board dame = boardFromRows({"..X.O",
                                "..X.O",
                                "..X.O",
                                "..X.O",
                                "..X.O"}, Color::Black);
    assert(dame.areaScore() == 10);

    // Seki: the inner black pair (0,0),(1,0) and the white group share the
    // two liberties (0,1),(1,1); whoever fills one is captured, so neither
    // does. Those two points touch both colors and are neutral.
    // Black: 9 stones + 9 territory (right column and bottom row) = 18.
    // White: 5 stones. Score 13 (the 2 seki points count for nobody).
    Board seki = boardFromRows({"X.OX.",
                                "X.OX.",
                                "OOOX.",
                                "XXXX.",
                                "....."}, Color::Black);
    assert(seki.libertiesAt(pt(0, 0)) == 2);
    assert(seki.libertiesAt(pt(0, 2)) == 2);
    assert(seki.stoneCount(Color::Black) == 9);
    assert(seki.stoneCount(Color::White) == 5);
    assert(seki.areaScore() == 13);
    // Filling a shared liberty is legal but self-atari: the opponent captures.
    Board fill = seki;
    fill.play(pt(0, 1));
    assert(fill.libertiesAt(pt(0, 0)) == 1);
    assert(fill.play(pt(1, 1)) == 3);
}

static void test_gtp_coordinates() {
    for (int n : {9, 19}) {
        for (int a = 0; a <= n * n; ++a) assert(parseAction(actionToString(a, n), n) == a);
    }
    assert(actionToString(pt(8, 0, 9), 9) == "A1");        // bottom-left
    assert(actionToString(pt(0, 0, 9), 9) == "A9");        // top-left
    assert(actionToString(pt(0, 18, 19), 19) == "T19");    // top-right on 19x19
    assert(actionToString(pt(0, 8, 19), 19) == "J19");     // column 8 is J: no I
    assert(actionToString(pt(0, 7, 19), 19) == "H19");
    assert(actionToString(81, 9) == "pass");
    assert(parseAction("A1", 9) == pt(8, 0, 9));
    assert(parseAction("a1", 9) == pt(8, 0, 9));           // case-insensitive
    assert(parseAction("J9", 9) == pt(0, 8, 9));
    assert(parseAction("t19", 19) == 18);
    assert(parseAction("E3", 5) == pt(2, 4));
    assert(parseAction("pass", 9) == 81);
    assert(parseAction("PASS", 9) == 81);
    assert(parseAction("Pass", 5) == 25);
    assert(parseAction("I5", 9) == -1);                    // no I column
    assert(parseAction("Z99", 19) == -1);
    assert(parseAction("A0", 9) == -1);
    assert(parseAction("A10", 9) == -1);                   // off a 9x9 board
    assert(parseAction("K1", 9) == -1);                    // column 10 of 9
    assert(parseAction("", 9) == -1);
    assert(parseAction("A", 9) == -1);
    assert(parseAction("1A", 9) == -1);
    assert(parseAction("A1x", 9) == -1);
    assert(parseAction("A-1", 9) == -1);
}

static void test_board_from_rows_round_trip() {
    const std::vector<std::string> rows = {"X.O..",
                                           ".....",
                                           "..X..",
                                           "O...X",
                                           "....O"};
    Board b = boardFromRows(rows, Color::White);
    assert(b.toMove() == Color::White);
    for (int r = 0; r < 5; ++r) {
        for (int c = 0; c < 5; ++c) {
            char ch = rows[r][c];
            Color expect = ch == 'X' ? Color::Black : ch == 'O' ? Color::White : Color::Empty;
            assert(b.at(pt(r, c)) == expect);
        }
    }
    std::vector<uint8_t> stones(25);
    b.exportStones(stones.data());
    for (int p = 0; p < 25; ++p) assert(stones[p] == static_cast<uint8_t>(b.at(p)));
    assert(b.hash() == b.computeHashFromScratch());
}

static void test_set_stone_keeps_hash_valid() {
    Board b(5);
    b.setStone(pt(1, 1), Color::Black);
    b.setStone(pt(3, 3), Color::White);
    assert(b.hash() == b.computeHashFromScratch());
    b.setStone(pt(1, 1), Color::White);                    // overwrite a stone
    assert(b.hash() == b.computeHashFromScratch());
    b.setStone(pt(1, 1), Color::Empty);
    b.setStone(pt(3, 3), Color::Empty);
    assert(b.hash() == Board(5).hash());

    Board played(5);
    played.play(pt(1, 1));
    Board placed(5);
    placed.setStone(pt(1, 1), Color::Black);
    assert(played.hash() == placed.hash());
    assert(!(played == placed));                           // different side to move
    placed.setToMove(Color::White);
    assert(played == placed);
}

static void test_equality() {
    assert(Board(5) == Board(5));
    assert(!(Board(5) == Board(6)));
    Board a(9), b(9);
    a.setStone(40, Color::Black);
    assert(!(a == b));
    b.setStone(40, Color::Black);
    assert(a == b);
}

static void test_to_string() {
    Board b(5);
    b.setStone(pt(4, 0), Color::Black);
    b.setStone(pt(0, 4), Color::White);
    b.setToMove(Color::White);
    const std::string expected =
        "   A B C D E\n"
        " 5 . . . . O  5\n"
        " 4 . . . . .  4\n"
        " 3 . . . . .  3\n"
        " 2 . . . . .  2\n"
        " 1 X . . . .  1\n"
        "   A B C D E\n"
        "White to move\n";
    assert(b.toString() == expected);

    std::string big = Board(19).toString();
    assert(big.find("   A B C D E F G H J K L M N O P Q R S T\n") == 0);
    assert(big.find("\n19 . ") != std::string::npos);
    assert(big.find("Black to move\n") != std::string::npos);
}

int main() {
    test_empty_board();
    test_play_alternates_and_pass();
    test_capture_corner();
    test_capture_edge();
    test_capture_center();
    test_capture_multi_stone_group();
    test_capture_two_groups_at_once();
    test_suicide_illegal();
    test_capture_that_looks_like_suicide_is_legal();
    test_liberties();
    test_hash_after_matches_play();
    test_area_score();
    test_gtp_coordinates();
    test_board_from_rows_round_trip();
    test_set_stone_keeps_hash_valid();
    test_equality();
    test_to_string();
    std::printf("test_board: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL compiling `tests/test_board.cpp` with `fatal error: 'azgo/board.hpp' file not found` (GCC: `azgo/board.hpp: No such file or directory`). The glob picks up the new test without any CMake edit.

- [ ] **Step 3: Write the header**

Create `include/azgo/board.hpp` (contract header; the two private helpers after the "implementation-defined" comment are this task's choice):

```cpp
// The rules of Go for a single position.
//
// Stones of one color that touch horizontally or vertically form a group; the
// empty points next to a group are its liberties. A move that leaves opponent
// groups without liberties captures them; a move that leaves its own group
// without liberties (suicide) is illegal here. The board is stored padded
// with a ring of Off cells so neighbor lookups never need bounds checks, and
// every position carries a Zobrist hash (XOR of one random key per stone) that
// Game uses to detect repeated positions (superko). Scoring is Tromp-Taylor
// area scoring: your stones plus empty regions that touch only your stones.
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace azgo {

constexpr int kMinBoardSize = 5;
constexpr int kMaxBoardSize = 19;
constexpr int kStride = kMaxBoardSize + 2;   // padded row length: one sentinel column each side
constexpr int kCells = kStride * kStride;    // padded array size (441)

enum class Color : uint8_t { Empty = 0, Black = 1, White = 2, Off = 3 };

Color opponent(Color c);   // Black <-> White; asserts c is Black or White

class Board {
public:
    explicit Board(int n);  // empty board, Black to move; asserts kMinBoardSize <= n <= kMaxBoardSize

    int size() const { return n_; }
    int area() const { return n_ * n_; }
    int passAction() const { return n_ * n_; }
    int numActions() const { return n_ * n_ + 1; }
    Color toMove() const { return toMove_; }

    Color at(int point) const;          // asserts 0 <= point < area()
    int stoneCount(Color c) const;
    int libertiesAt(int point) const;   // liberties of the group containing point; 0 if empty

    // Local legality: pass is always legal; a point must be empty and the move
    // must not be suicide (after removing captured opponent groups, the new
    // stone's group must have a liberty). Superko is NOT checked here — it
    // needs history, see Game::isLegal and Mcts.
    bool isLegal(int action) const;
    // Plays a locally legal action (asserted), removes captured groups, flips
    // toMove. Pass only flips toMove. Returns the number of stones captured.
    int play(int action);
    // Hash of the position play(action) would produce. O(1) when the move
    // captures nothing; otherwise copies the board and plays.
    uint64_t hashAfter(int action) const;

    uint64_t hash() const { return hash_; }  // Zobrist over stones only (positional superko)
    uint64_t computeHashFromScratch() const; // test oracle for the incremental hash

    // Tromp-Taylor area score: (Black stones + empty regions reaching only Black)
    // minus (White stones + empty regions reaching only White). No komi.
    int areaScore() const;

    void exportStones(uint8_t* out) const;   // area() bytes of Color values, row-major
    std::string toString() const;            // ASCII diagram with GTP coordinates

    // Setup helpers for tests and tools: no capture logic; hash stays valid.
    void setStone(int point, Color c);       // c is Empty, Black or White
    void setToMove(Color c);

    bool operator==(const Board& o) const;   // size, stones and side to move

private:
    int pad(int point) const;                // point -> padded index
    int n_;
    Color toMove_;
    std::array<Color, kCells> cells_;
    uint64_t hash_;
    // Additional private helpers are implementation-defined (Task 2).
    // Liberties of the group containing the stone at padded index p. Counting
    // stops once `limit` liberties are found, which is all isLegal needs
    // ("none", "exactly one", or "at least two").
    int countLiberties(int p, int limit) const;
    // Removes the group containing the stone at padded index p (updating the
    // hash) and returns the number of stones removed.
    int removeGroup(int p);
};

// Rows like "X.O.." ('X' black, 'O' white, '.' empty), row 0 first (top).
// Asserts rows form an n x n square with n in range.
Board boardFromRows(const std::vector<std::string>& rows, Color toMove);

// GTP coordinates. parseAction is case-insensitive and returns -1 for
// malformed or off-board input.
std::string actionToString(int action, int n);
int parseAction(const std::string& text, int n);

} // namespace azgo
```

- [ ] **Step 4: Write the implementation**

Create `src/board.cpp`:

```cpp
#include "azgo/board.hpp"
#include <cassert>
#include <cctype>
#include "azgo/rng.hpp"

namespace azgo {

namespace {

// Offsets to the four orthogonal neighbors in the padded array. Thanks to the
// Off border, p + d is always a valid index for any on-board p.
constexpr int kDirs[4] = {-kStride, -1, 1, kStride};

// GTP column letters: 'I' is skipped to avoid confusion with 'J' and '1'.
constexpr char kColumnLetters[] = "ABCDEFGHJKLMNOPQRST";

using ZobristTable = std::array<std::array<uint64_t, 2>, kCells>;

// One random key per (padded cell, stone color). Generated once, from a fixed
// seed, so hashes are identical across runs, threads and machines. A
// function-local static is initialized thread-safely on first use.
const ZobristTable& zobristTable() {
    static const ZobristTable table = [] {
        ZobristTable t{};
        uint64_t x = 0x5EEDB0A4D5EEDB0AULL;
        for (auto& cell : t) {
            cell[0] = splitmix64(x);
            cell[1] = splitmix64(x);
        }
        return t;
    }();
    return table;
}

uint64_t zobristKey(int padded, Color c) {
    assert(c == Color::Black || c == Color::White);
    return zobristTable()[padded][c == Color::Black ? 0 : 1];
}

} // namespace

Color opponent(Color c) {
    assert(c == Color::Black || c == Color::White);
    return c == Color::Black ? Color::White : Color::Black;
}

Board::Board(int n) : n_(n), toMove_(Color::Black), cells_(), hash_(0) {
    assert(n >= kMinBoardSize && n <= kMaxBoardSize);
    // Everything starts Off; only the n x n playing area is opened up. On a
    // 5x5 board the unused part of the 21x21 array stays Off as well, so it
    // behaves exactly like the sentinel border.
    cells_.fill(Color::Off);
    for (int p = 0; p < area(); ++p) cells_[pad(p)] = Color::Empty;
}

int Board::pad(int point) const {
    return (point / n_ + 1) * kStride + (point % n_ + 1);
}

Color Board::at(int point) const {
    assert(point >= 0 && point < area());
    return cells_[pad(point)];
}

int Board::stoneCount(Color c) const {
    int count = 0;
    for (int p = 0; p < area(); ++p) count += (cells_[pad(p)] == c);
    return count;
}

int Board::countLiberties(int p, int limit) const {
    const Color color = cells_[p];
    assert(color == Color::Black || color == Color::White);
    // Flood fill over the group. The scratch arrays live on this call's stack
    // (about 2 KB): no heap allocation, and safe to call from many threads.
    // `mark` flags both visited stones and already-counted liberties, so a
    // liberty shared by several stones of the group is counted once.
    std::array<int, kCells> stack;
    std::array<uint8_t, kCells> mark{};
    int top = 0, liberties = 0;
    stack[top++] = p;
    mark[p] = 1;
    while (top > 0) {
        const int q = stack[--top];
        for (int d : kDirs) {
            const int r = q + d;
            if (mark[r]) continue;
            if (cells_[r] == Color::Empty) {
                mark[r] = 1;
                if (++liberties >= limit) return liberties;
            } else if (cells_[r] == color) {
                mark[r] = 1;
                stack[top++] = r;
            }
        }
    }
    return liberties;
}

int Board::removeGroup(int p) {
    const Color color = cells_[p];
    assert(color == Color::Black || color == Color::White);
    // Stones are emptied as they are pushed, so nothing is visited twice and
    // no separate mark array is needed.
    std::array<int, kCells> stack;
    int top = 0, removed = 0;
    stack[top++] = p;
    cells_[p] = Color::Empty;
    hash_ ^= zobristKey(p, color);
    while (top > 0) {
        const int q = stack[--top];
        ++removed;
        for (int d : kDirs) {
            const int r = q + d;
            if (cells_[r] == color) {
                cells_[r] = Color::Empty;
                hash_ ^= zobristKey(r, color);
                stack[top++] = r;
            }
        }
    }
    return removed;
}

int Board::libertiesAt(int point) const {
    const int p = pad(point);
    if (cells_[p] != Color::Black && cells_[p] != Color::White) return 0;
    return countLiberties(p, kCells);
}

bool Board::isLegal(int action) const {
    if (action == passAction()) return true;
    if (action < 0 || action > passAction()) return false;
    const int p = pad(action);
    if (cells_[p] != Color::Empty) return false;
    const Color me = toMove_;
    const Color them = opponent(me);
    // The new stone survives if any neighbor gives it a liberty:
    //  - an empty neighbor is a liberty directly;
    //  - an opponent group whose only liberty is p gets captured, freeing a point;
    //  - a friendly group with another liberty besides p lends it one.
    for (int d : kDirs) {
        const int q = p + d;
        const Color c = cells_[q];
        if (c == Color::Empty) return true;
        if (c == them && countLiberties(q, 2) == 1) return true;
        if (c == me && countLiberties(q, 2) >= 2) return true;
    }
    return false;  // suicide
}

int Board::play(int action) {
    assert(isLegal(action));
    if (action == passAction()) {
        toMove_ = opponent(toMove_);
        return 0;
    }
    const int p = pad(action);
    const Color me = toMove_;
    const Color them = opponent(me);
    cells_[p] = me;
    hash_ ^= zobristKey(p, me);
    int captured = 0;
    for (int d : kDirs) {
        const int q = p + d;
        // A group already removed via another neighbor reads Empty here.
        if (cells_[q] == them && countLiberties(q, 1) == 0) captured += removeGroup(q);
    }
    assert(countLiberties(p, 1) > 0);  // isLegal ruled out suicide
    toMove_ = them;
    return captured;
}

uint64_t Board::hashAfter(int action) const {
    if (action == passAction()) return hash_;
    assert(isLegal(action));
    const int p = pad(action);
    const Color them = opponent(toMove_);
    for (int d : kDirs) {
        const int q = p + d;
        if (cells_[q] == them && countLiberties(q, 2) == 1) {
            // The move captures: let play() work out which stones go.
            Board next = *this;
            next.play(action);
            return next.hash();
        }
    }
    // No capture: the only change is one new stone.
    return hash_ ^ zobristKey(p, toMove_);
}

uint64_t Board::computeHashFromScratch() const {
    uint64_t h = 0;
    for (int point = 0; point < area(); ++point) {
        const int p = pad(point);
        if (cells_[p] == Color::Black || cells_[p] == Color::White) h ^= zobristKey(p, cells_[p]);
    }
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
    for (int point = 0; point < area(); ++point) out[point] = static_cast<uint8_t>(cells_[pad(point)]);
}

std::string Board::toString() const {
    std::string letters = "  ";
    for (int col = 0; col < n_; ++col) {
        letters += ' ';
        letters += kColumnLetters[col];
    }
    letters += '\n';

    std::string s = letters;
    for (int row = 0; row < n_; ++row) {
        const int label = n_ - row;
        const std::string num = (label < 10 ? " " : "") + std::to_string(label);
        s += num;
        for (int col = 0; col < n_; ++col) {
            const Color c = cells_[pad(row * n_ + col)];
            s += ' ';
            s += c == Color::Black ? 'X' : c == Color::White ? 'O' : '.';
        }
        s += ' ' + num + '\n';
    }
    s += letters;
    s += toMove_ == Color::Black ? "Black to move\n" : "White to move\n";
    return s;
}

void Board::setStone(int point, Color c) {
    assert(point >= 0 && point < area());
    assert(c == Color::Empty || c == Color::Black || c == Color::White);
    const int p = pad(point);
    if (cells_[p] != Color::Empty) hash_ ^= zobristKey(p, cells_[p]);
    cells_[p] = c;
    if (c != Color::Empty) hash_ ^= zobristKey(p, c);
}

void Board::setToMove(Color c) {
    assert(c == Color::Black || c == Color::White);
    toMove_ = c;
}

bool Board::operator==(const Board& o) const {
    // Unused cells are Off on every board, so comparing whole arrays is exact.
    return n_ == o.n_ && toMove_ == o.toMove_ && cells_ == o.cells_;
}

Board boardFromRows(const std::vector<std::string>& rows, Color toMove) {
    const int n = static_cast<int>(rows.size());
    assert(n >= kMinBoardSize && n <= kMaxBoardSize);
    Board b(n);
    for (int row = 0; row < n; ++row) {
        assert(static_cast<int>(rows[row].size()) == n);
        for (int col = 0; col < n; ++col) {
            const char ch = rows[row][col];
            assert(ch == 'X' || ch == 'O' || ch == '.');
            if (ch == 'X') b.setStone(row * n + col, Color::Black);
            if (ch == 'O') b.setStone(row * n + col, Color::White);
        }
    }
    b.setToMove(toMove);
    return b;
}

std::string actionToString(int action, int n) {
    assert(n >= kMinBoardSize && n <= kMaxBoardSize);
    assert(action >= 0 && action <= n * n);
    if (action == n * n) return "pass";
    const int row = action / n, col = action % n;
    return std::string(1, kColumnLetters[col]) + std::to_string(n - row);
}

int parseAction(const std::string& text, int n) {
    assert(n >= kMinBoardSize && n <= kMaxBoardSize);
    std::string t;
    for (char ch : text) t += static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    if (t == "PASS") return n * n;
    // A letter followed by one or two digits.
    if (t.size() < 2 || t.size() > 3) return -1;
    int col = -1;
    for (int c = 0; c < n; ++c) {
        if (kColumnLetters[c] == t[0]) col = c;
    }
    if (col < 0) return -1;  // 'I', a letter beyond the board, or not a letter
    int label = 0;
    for (size_t i = 1; i < t.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(t[i]))) return -1;
        label = label * 10 + (t[i] - '0');
    }
    if (label < 1 || label > n) return -1;
    const int row = n - label;
    return row * n + col;
}

} // namespace azgo
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
cmake --build build -j
ctest --test-dir build --output-on-failure -R test_board
```
Expected: no compiler warnings from `src/board.cpp`, and
```
1/1 Test #1: test_board .......................   Passed
100% tests passed out of 1
```
(`./build/test_board` prints `test_board: all tests passed`.) Then run the whole suite, `ctest --test-dir build --output-on-failure`: `100% tests passed out of 2`.

- [ ] **Step 6: Commit**

```bash
git add include/azgo/board.hpp src/board.cpp tests/test_board.cpp
git commit -F - <<'EOF'
Add the Go board: captures, suicide, Zobrist hash and area scoring

Board is the single-position rules engine every later component builds
on: MCTS copies it at each node, self-play and the arena play it, and
the network sees it through the input planes. It uses a 21x21 padded
array so neighbor lookups never bounds-check and one type serves every
size from 5 to 19. Groups and liberties come from flood fills with
stack-local scratch, which keeps the code short, allocation-free and
thread-safe; board work is microseconds next to network evaluation.

The incremental Zobrist hash (stones only) is what Game and MCTS use for
positional superko, and hashAfter answers "which position would this
move make?" without copying the board unless the move captures. Tests
cover captures in the corner, on the edge and in the center, suicide,
captures that look like suicide, seki and dame under Tromp-Taylor
scoring, and GTP coordinates.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 3: Game

**Files:**
- Create: `include/azgo/game.hpp`, `src/game.cpp`
- Test: `tests/test_game.cpp`

**Interfaces:**
- Consumes: from Task 2 — `Board(int n)`, `Board::isLegal(int)`, `Board::play(int)`, `Board::hashAfter(int)`, `Board::hash()`, `Board::areaScore()`, `Board::passAction()`, `Board::numActions()`, `Board::size()`; in tests also `boardFromRows`, `Board::setStone`, `Board::libertiesAt`, `Board::computeHashFromScratch`, `Board::stoneCount`, `opponent`, `kMinBoardSize`, `kMaxBoardSize`. From Task 1 (tests only) — `Rng`, `Rng::below`, `deriveSeed`.
- Produces: `class Game` with `Game(int n, float komi, int maxMoves)`, `Game(const Board& start, float komi, int maxMoves)`, `size()`, `komi()`, `maxMoves()`, `const Board& board() const`, `const Board& boardAt(int ply) const`, `int moveCount() const`, `const std::vector<int>& moves() const`, `int consecutivePasses() const`, `bool isLegal(int) const`, `std::vector<int> legalActions() const`, `void play(int)`, `bool seen(uint64_t) const`, `bool isTerminal() const`, `float scoreForBlack() const`, `float resultForBlack() const`; `int defaultMaxMoves(int n)` (= `2 * n * n`).

- [ ] **Step 1: Write the failing test**

The ko position used below (5×5, Black to move):

```
. . . . .
. X O . .
X O . O .      Black captures at (2,2); White's retake at (2,1)
. X O . .      would recreate the start position.
. . . . .
```

Create `tests/test_game.cpp`:

```cpp
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include "azgo/game.hpp"
#include "azgo/rng.hpp"

using namespace azgo;

static bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

static int pt(int row, int col, int n = 5) { return row * n + col; }

static bool contains(const std::vector<int>& v, int x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

// A ko: White's (2,1) is in atari at (2,2). Black captures it at (2,2), after
// which Black's lone stone is itself in atari at (2,1).
static Board koStart() {
    return boardFromRows({".....",
                          ".XO..",
                          "XO.O.",
                          ".XO..",
                          "....."}, Color::Black);
}

static void test_simple_ko() {
    Game g(koStart(), 7.5f, defaultMaxMoves(5));
    const int take = pt(2, 2), retake = pt(2, 1);
    assert(g.isLegal(take));
    g.play(take);
    assert(g.board().at(retake) == Color::Empty);
    assert(g.board().libertiesAt(take) == 1);

    // Locally the retake is a legal capture, but it would recreate the
    // starting position, so positional superko forbids it.
    assert(g.board().isLegal(retake));
    assert(g.seen(g.board().hashAfter(retake)));
    assert(!g.isLegal(retake));
    assert(!contains(g.legalActions(), retake));
    assert(g.isLegal(g.board().passAction()));

    // Ko threat exchange: White plays elsewhere, Black answers elsewhere.
    // The board is now different, so the retake is allowed.
    g.play(pt(4, 4));
    g.play(pt(0, 4));
    assert(g.isLegal(retake));
    g.play(retake);
    assert(g.board().at(take) == Color::Empty);
    // And now Black's immediate retake is forbidden in turn.
    assert(g.board().isLegal(take));
    assert(!g.isLegal(take));
}

static void test_pass_never_blocked_by_superko() {
    Game g(5, 7.5f, defaultMaxMoves(5));
    // The current position is already in the history, and passing repeats it.
    assert(g.seen(g.board().hash()));
    assert(g.isLegal(g.board().passAction()));
    g.play(pt(2, 2));
    g.play(g.board().passAction());           // position repeats: still legal
    assert(g.board().hash() == g.boardAt(1).hash());
    assert(g.isLegal(g.board().passAction()));
}

static void test_two_passes_end_the_game() {
    Game g(5, 7.5f, defaultMaxMoves(5));
    const int pass = g.board().passAction();
    g.play(pass);
    assert(g.consecutivePasses() == 1);
    assert(!g.isTerminal());
    g.play(pass);
    assert(g.consecutivePasses() == 2);
    assert(g.isTerminal());
    assert(g.moveCount() == 2);
    assert(!g.isLegal(pass));
    assert(!g.isLegal(pt(0, 0)));
    assert(g.legalActions().empty());
}

static void test_move_after_pass_resets_count() {
    Game g(5, 7.5f, defaultMaxMoves(5));
    const int pass = g.board().passAction();
    g.play(pass);
    g.play(pt(1, 1));
    assert(g.consecutivePasses() == 0);
    g.play(pass);
    assert(g.consecutivePasses() == 1);
    assert(!g.isTerminal());
}

static void test_move_cap() {
    assert(defaultMaxMoves(5) == 50);
    assert(defaultMaxMoves(19) == 722);
    Game g(5, 7.5f, 3);
    assert(g.maxMoves() == 3);
    g.play(pt(0, 0));
    g.play(pt(4, 4));
    assert(!g.isTerminal());
    g.play(pt(2, 2));
    assert(g.isTerminal());
    assert(g.legalActions().empty());
    assert(!g.isLegal(g.board().passAction()));
}

static void test_score_and_result() {
    // Pass-pass on an empty board: White wins by komi.
    Game empty(5, 7.5f, defaultMaxMoves(5));
    empty.play(25);
    empty.play(25);
    assert(near(empty.scoreForBlack(), -7.5));
    assert(empty.resultForBlack() == -1.0f);

    // Review Focus 2: integer komi 0, pass-pass on an empty 5x5 board is an
    // exact tie, scored 0 rather than a win for either side.
    Game tie(5, 0.0f, defaultMaxMoves(5));
    tie.play(25);
    tie.play(25);
    assert(tie.isTerminal());
    assert(tie.scoreForBlack() == 0.0f);
    assert(tie.resultForBlack() == 0.0f);

    // One black stone owns the whole 5x5 board: 25 - 7.5.
    Board one(5);
    one.setStone(pt(2, 2), Color::Black);
    Game black(one, 7.5f, defaultMaxMoves(5));
    assert(near(black.scoreForBlack(), 17.5));
    assert(black.resultForBlack() == 1.0f);

    // Area score 10 (see test_board's dame position) with komi 10: exact tie.
    Board dame = boardFromRows({"..X.O",
                                "..X.O",
                                "..X.O",
                                "..X.O",
                                "..X.O"}, Color::Black);
    Game even(dame, 10.0f, defaultMaxMoves(5));
    assert(even.scoreForBlack() == 0.0f);
    assert(even.resultForBlack() == 0.0f);
    Game white(dame, 10.5f, defaultMaxMoves(5));
    assert(white.resultForBlack() == -1.0f);
}

static void test_history() {
    Game g(5, 7.5f, defaultMaxMoves(5));
    const std::vector<int> moves = {pt(0, 0), pt(4, 4), 25, pt(2, 2)};
    for (int m : moves) g.play(m);
    assert(g.moveCount() == 4);
    assert(g.moves() == moves);
    assert(g.boardAt(0) == Board(5));
    assert(g.boardAt(1).at(pt(0, 0)) == Color::Black);
    assert(g.boardAt(1).at(pt(4, 4)) == Color::Empty);
    assert(g.boardAt(1).toMove() == Color::White);
    assert(g.boardAt(2).at(pt(4, 4)) == Color::White);
    assert(g.boardAt(3).toMove() == Color::White);       // Black passed
    assert(g.boardAt(3).hash() == g.boardAt(2).hash());
    assert(g.boardAt(4) == g.board());
    assert(g.board().at(pt(2, 2)) == Color::White);
    assert(g.size() == 5);
    assert(g.komi() == 7.5f);
    for (int k = 0; k <= g.moveCount(); ++k) assert(g.seen(g.boardAt(k).hash()));
}

// Uniformly random legal play on every board size. Checks the invariants
// that the rest of the program relies on after every single move.
static void test_random_playouts() {
    int captureMoves = 0;
    for (int n = kMinBoardSize; n <= kMaxBoardSize; ++n) {
        for (int game = 0; game < 2; ++game) {
            Rng rng(deriveSeed(2026, static_cast<uint64_t>(n), static_cast<uint64_t>(game)));
            Game g(n, 7.5f, defaultMaxMoves(n));
            int plies = 0;
            while (!g.isTerminal()) {
                std::vector<int> legal = g.legalActions();
                assert(!legal.empty() && legal.back() == g.board().passAction());
                assert(std::is_sorted(legal.begin(), legal.end()));
                const int a = legal[static_cast<size_t>(rng.below(static_cast<int>(legal.size())))];
                const Color mover = g.board().toMove();
                const int stonesBefore = g.board().area() - g.board().stoneCount(Color::Empty);
                g.play(a);
                ++plies;
                const Board& b = g.board();
                assert(b.toMove() == opponent(mover));
                if (b.area() - b.stoneCount(Color::Empty) <= stonesBefore && a != b.passAction()) ++captureMoves;
                assert(b.hash() == b.computeHashFromScratch());
                for (int p = 0; p < b.area(); ++p) {
                    if (b.at(p) != Color::Empty) assert(b.libertiesAt(p) > 0);
                }
                assert(plies <= g.maxMoves());
            }
            assert(g.moveCount() == plies);
            assert(g.consecutivePasses() >= 2 || plies == g.maxMoves());
            const float r = g.resultForBlack();
            assert(r == 1.0f || r == -1.0f);                  // komi 7.5 cannot tie
        }
    }
    assert(captureMoves > 100);  // the playouts really exercised captures
}

int main() {
    test_simple_ko();
    test_pass_never_blocked_by_superko();
    test_two_passes_end_the_game();
    test_move_after_pass_resets_count();
    test_move_cap();
    test_score_and_result();
    test_history();
    test_random_playouts();
    std::printf("test_game: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL compiling `tests/test_game.cpp` with `fatal error: 'azgo/game.hpp' file not found`.

- [ ] **Step 3: Write the header**

Create `include/azgo/game.hpp`:

```cpp
// A game of Go in progress.
//
// A single Board knows the local rules, but two rules need the game's past.
// Positional superko: a move may not recreate any whole-board position that
// already occurred in this game; this is what stops ko fights from looping
// forever. It is enforced by remembering the Zobrist hash of every position.
// The end of the game: two consecutive passes (or a move cap, 2 * n * n by
// default, so self-play always finishes); the final position is then scored
// by Tromp-Taylor area minus komi, White's compensation for moving second.
#pragma once
#include <unordered_set>
#include <vector>
#include "azgo/board.hpp"

namespace azgo {

// A game in progress: the sequence of positions, for history planes and
// positional superko, plus the pass count and the move cap.
class Game {
public:
    Game(int n, float komi, int maxMoves);
    Game(const Board& start, float komi, int maxMoves);

    int size() const { return boards_.front().size(); }
    float komi() const { return komi_; }
    int maxMoves() const { return maxMoves_; }

    const Board& board() const { return boards_.back(); }
    const Board& boardAt(int ply) const;          // 0 = start; asserts ply <= moveCount()
    int moveCount() const { return static_cast<int>(moves_.size()); }
    const std::vector<int>& moves() const { return moves_; }
    int consecutivePasses() const { return passes_; }

    // Board::isLegal plus positional superko: a non-pass move may not
    // recreate any earlier position of this game. Always false once terminal.
    bool isLegal(int action) const;
    std::vector<int> legalActions() const;        // ascending; always contains pass unless terminal
    void play(int action);                        // asserts isLegal(action)
    bool seen(uint64_t positionHash) const;       // position occurred in this game

    // Two consecutive passes, or moveCount() >= maxMoves().
    bool isTerminal() const;
    float scoreForBlack() const;                  // board().areaScore() - komi
    float resultForBlack() const;                 // +1 Black wins, -1 White wins, 0 exact tie

private:
    std::vector<Board> boards_;                   // boards_[k] = position after k moves
    std::vector<int> moves_;
    std::unordered_set<uint64_t> seen_;
    int passes_ = 0;
    float komi_;
    int maxMoves_;
};

// 2 * n * n, the default move cap.
int defaultMaxMoves(int n);

} // namespace azgo
```

- [ ] **Step 4: Write the implementation**

Create `src/game.cpp`:

```cpp
#include "azgo/game.hpp"
#include <cassert>

namespace azgo {

Game::Game(int n, float komi, int maxMoves) : Game(Board(n), komi, maxMoves) {}

Game::Game(const Board& start, float komi, int maxMoves) : komi_(komi), maxMoves_(maxMoves) {
    assert(maxMoves > 0);
    boards_.push_back(start);
    seen_.insert(start.hash());
}

const Board& Game::boardAt(int ply) const {
    assert(ply >= 0 && ply <= moveCount());
    return boards_[static_cast<size_t>(ply)];
}

bool Game::isLegal(int action) const {
    if (isTerminal()) return false;
    const Board& b = board();
    // Passing never changes the stones, so it can never break superko.
    if (action == b.passAction()) return true;
    if (!b.isLegal(action)) return false;
    // Positional superko: compare the hash of the resulting stones against
    // every position of this game so far.
    return seen_.count(b.hashAfter(action)) == 0;
}

std::vector<int> Game::legalActions() const {
    std::vector<int> actions;
    if (isTerminal()) return actions;
    for (int a = 0; a < board().numActions(); ++a) {
        if (isLegal(a)) actions.push_back(a);
    }
    return actions;
}

void Game::play(int action) {
    assert(isLegal(action));
    Board next = board();
    next.play(action);
    passes_ = (action == next.passAction()) ? passes_ + 1 : 0;
    moves_.push_back(action);
    seen_.insert(next.hash());  // a pass re-inserts an existing hash: no-op
    boards_.push_back(next);
}

bool Game::seen(uint64_t positionHash) const {
    return seen_.count(positionHash) > 0;
}

bool Game::isTerminal() const {
    return passes_ >= 2 || moveCount() >= maxMoves_;
}

float Game::scoreForBlack() const {
    return static_cast<float>(board().areaScore()) - komi_;
}

float Game::resultForBlack() const {
    const float s = scoreForBlack();
    // An integer komi can produce an exact tie; it is scored as a draw.
    if (s > 0.0f) return 1.0f;
    if (s < 0.0f) return -1.0f;
    return 0.0f;
}

int defaultMaxMoves(int n) {
    return 2 * n * n;
}

} // namespace azgo
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
cmake --build build -j
ctest --test-dir build --output-on-failure -R test_game
```
Expected:
```
1/1 Test #2: test_game ........................   Passed
100% tests passed out of 1
```
(`./build/test_game` prints `test_game: all tests passed`; the 30 random playouts on 5×5–19×19 take well under a second in Release — if this test takes more than ~5 s, something in the flood fills is quadratic.) Then `ctest --test-dir build --output-on-failure`: `100% tests passed out of 3`.

- [ ] **Step 6: Commit**

```bash
git add include/azgo/game.hpp src/game.cpp tests/test_game.cpp
git commit -F - <<'EOF'
Add Game with positional superko, pass handling and scoring

Superko and the end of the game need history that a single Board does
not have. Game keeps every position (later used for the history input
planes and SGF export) plus a set of their Zobrist hashes, so rejecting
a move that recreates an earlier position is one hash lookup. Passing
is always legal, two consecutive passes or the 2*n*n move cap end the
game, and the result is Tromp-Taylor area minus komi, with an exact tie
(possible with integer komi) reported as 0 rather than forced to a win.

Random playouts on every size from 5 to 19 check after every move that
the incremental hash matches a recomputation and that no group without
liberties is left on the board.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 4: Symmetry and input encoding

**Files:**
- Create: `include/azgo/symmetry.hpp`, `src/symmetry.cpp`, `include/azgo/encode.hpp`, `src/encode.cpp`
- Test: `tests/test_symmetry.cpp`, `tests/test_encode.cpp`

**Interfaces:**
- Consumes: from Task 2 — `Color`, `opponent(Color)`, `kMinBoardSize`, `kMaxBoardSize`, `Board::size()`, `Board::toMove()`, `Board::exportStones(uint8_t*)`; in tests `Board::setStone`, `Board::setToMove`. From Task 3 — `Game::boardAt(int)`, `Game::moveCount()`, `Game::size()`; in tests `Game(int, float, int)`, `Game::play`, `defaultMaxMoves`.
- Produces: `constexpr int kNumSymmetries = 8`; `int transformPoint(int sym, int point, int n)`; `int transformAction(int sym, int action, int n)`; `int inverseSymmetry(int sym)`; `int numPlanes(int history)`; `void encodePlanes(const std::vector<const uint8_t*>& stones, int n, Color toMove, int H, int sym, float* out)`; `void encodeBoards(const std::vector<const Board*>& history, int H, int sym, float* out)`; `void encodeGame(const Game& game, int H, int sym, float* out)`.

- [ ] **Step 1: Write the failing symmetry test**

Create `tests/test_symmetry.cpp`:

```cpp
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>
#include "azgo/symmetry.hpp"

using namespace azgo;

static int pt(int row, int col, int n) { return row * n + col; }

static void test_identity() {
    for (int n : {5, 9}) {
        for (int p = 0; p < n * n; ++p) assert(transformPoint(0, p, n) == p);
    }
    assert(inverseSymmetry(0) == 0);
}

static void test_hand_computed_images() {
    const int n = 5;
    // s = 1: one clockwise quarter turn. Top-left goes to top-right,
    // top-right to bottom-right.
    assert(transformPoint(1, pt(0, 0, n), n) == pt(0, 4, n));
    assert(transformPoint(1, pt(0, 4, n), n) == pt(4, 4, n));
    assert(transformPoint(1, pt(0, 1, n), n) == pt(1, 4, n));
    // s = 2: half turn. s = 3: counter-clockwise quarter turn.
    assert(transformPoint(2, pt(0, 1, n), n) == pt(4, 3, n));
    assert(transformPoint(3, pt(0, 1, n), n) == pt(3, 0, n));
    // s = 4: mirror left-right.
    assert(transformPoint(4, pt(0, 1, n), n) == pt(0, 3, n));
    // s = 5: mirror, then a clockwise turn = reflection in the anti-diagonal:
    // (r, c) -> (n-1-c, n-1-r).
    assert(transformPoint(5, pt(0, 1, n), n) == pt(3, 4, n));
    // s = 6: mirror, then a half turn = flip top-bottom.
    assert(transformPoint(6, pt(0, 1, n), n) == pt(4, 1, n));
    // s = 7: mirror, then three turns = transpose (r, c) -> (c, r).
    assert(transformPoint(7, pt(0, 1, n), n) == pt(1, 0, n));
    // The center of an odd board is fixed by everything.
    for (int s = 0; s < kNumSymmetries; ++s) assert(transformPoint(s, pt(2, 2, n), n) == pt(2, 2, n));
}

static void test_inverse_composes_to_identity() {
    for (int n : {5, 6, 9, 19}) {
        for (int s = 0; s < kNumSymmetries; ++s) {
            const int inv = inverseSymmetry(s);
            assert(inv >= 0 && inv < kNumSymmetries);
            for (int p = 0; p < n * n; ++p) {
                assert(transformPoint(inv, transformPoint(s, p, n), n) == p);
                assert(transformPoint(s, transformPoint(inv, p, n), n) == p);
            }
        }
    }
    assert(inverseSymmetry(1) == 3 && inverseSymmetry(3) == 1 && inverseSymmetry(2) == 2);
}

static void test_transform_is_bijection() {
    for (int n : {5, 6, 19}) {
        for (int s = 0; s < kNumSymmetries; ++s) {
            std::vector<int> hits(static_cast<size_t>(n * n), 0);
            for (int p = 0; p < n * n; ++p) {
                const int q = transformPoint(s, p, n);
                assert(q >= 0 && q < n * n);
                ++hits[static_cast<size_t>(q)];
            }
            for (int h : hits) assert(h == 1);
        }
    }
}

static void test_eight_distinct_images() {
    // An L of three and two stones in the corner has no symmetry of its own,
    // so its 8 images are pairwise different.
    const int n = 5;
    const std::vector<int> shape = {pt(0, 0, n), pt(0, 1, n), pt(0, 2, n), pt(1, 0, n)};
    std::vector<std::vector<int>> images;
    for (int s = 0; s < kNumSymmetries; ++s) {
        std::vector<int> img;
        for (int p : shape) img.push_back(transformPoint(s, p, n));
        std::sort(img.begin(), img.end());
        images.push_back(img);
    }
    for (int a = 0; a < kNumSymmetries; ++a) {
        for (int b = a + 1; b < kNumSymmetries; ++b) assert(images[a] != images[b]);
    }
}

static void test_transform_action_keeps_pass() {
    for (int n : {5, 9, 19}) {
        for (int s = 0; s < kNumSymmetries; ++s) {
            assert(transformAction(s, n * n, n) == n * n);
            assert(transformAction(s, 0, n) == transformPoint(s, 0, n));
        }
    }
}

int main() {
    test_identity();
    test_hand_computed_images();
    test_inverse_composes_to_identity();
    test_transform_is_bijection();
    test_eight_distinct_images();
    test_transform_action_keeps_pass();
    std::printf("test_symmetry: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL compiling `tests/test_symmetry.cpp` with `fatal error: 'azgo/symmetry.hpp' file not found`.

- [ ] **Step 3: Write the symmetry header**

Create `include/azgo/symmetry.hpp`:

```cpp
// The eight symmetries of the square board.
//
// Go's rules do not change if the board is rotated or mirrored, so a
// position and its 7 transformed copies have the same value, and their best
// moves correspond. Training exploits this by showing the network a random
// transform of each sample (8x more data for free). A transform is a
// permutation of points; pass is not a point and maps to itself.
#pragma once

namespace azgo {

constexpr int kNumSymmetries = 8;

// Symmetry s in [0, 8): s & 3 = number of clockwise quarter turns, s & 4 = mirror
// left-right first. 0 is the identity.
int transformPoint(int sym, int point, int n);
int transformAction(int sym, int action, int n);  // pass (n*n) maps to itself
int inverseSymmetry(int sym);                      // transformPoint(inverse(s), transformPoint(s, p)) == p

} // namespace azgo
```

- [ ] **Step 4: Write the symmetry implementation**

Create `src/symmetry.cpp`:

```cpp
#include "azgo/symmetry.hpp"
#include <cassert>

namespace azgo {

int transformPoint(int sym, int point, int n) {
    assert(sym >= 0 && sym < kNumSymmetries);
    assert(point >= 0 && point < n * n);
    int row = point / n, col = point % n;
    if (sym & 4) col = n - 1 - col;               // mirror left-right first
    for (int turn = 0; turn < (sym & 3); ++turn) {
        // One clockwise quarter turn: the top row becomes the right column.
        const int r = col;
        const int c = n - 1 - row;
        row = r;
        col = c;
    }
    return row * n + col;
}

int transformAction(int sym, int action, int n) {
    if (action == n * n) return action;           // pass has no location
    return transformPoint(sym, action, n);
}

int inverseSymmetry(int sym) {
    assert(sym >= 0 && sym < kNumSymmetries);
    // Pure rotations: undo k clockwise turns with 4 - k more.
    // Mirrored ones: T = R^k M, and mirroring reverses the direction of a
    // rotation (M R^k = R^-k M), so T T = R^k (M R^k) M = R^k R^-k M M = I.
    // Every mirrored transform is its own inverse.
    if (sym & 4) return sym;
    return (4 - sym) % 4;
}

} // namespace azgo
```

- [ ] **Step 5: Run the symmetry test to verify it passes**

Run:
```bash
cmake --build build -j
ctest --test-dir build --output-on-failure -R test_symmetry
```
Expected: `1/1 Test #4: test_symmetry ... Passed` and `100% tests passed out of 1` (`./build/test_symmetry` prints `test_symmetry: all tests passed`).

- [ ] **Step 6: Write the failing encode test**

Create `tests/test_encode.cpp`. Output buffers are pre-filled with −1 so a point the encoder forgets to write shows up as a failure.

```cpp
#include <cassert>
#include <cstdio>
#include <vector>
#include "azgo/encode.hpp"
#include "azgo/game.hpp"
#include "azgo/symmetry.hpp"

using namespace azgo;

static int pt(int row, int col, int n = 5) { return row * n + col; }

static std::vector<float> encodeB(const std::vector<const Board*>& history, int H, int sym) {
    const int n = history[0]->size();
    std::vector<float> out(static_cast<size_t>(numPlanes(H)) * n * n, -1.0f);  // -1: catch unwritten
    encodeBoards(history, H, sym, out.data());
    return out;
}

static std::vector<float> encodeG(const Game& g, int H, int sym) {
    const int n = g.size();
    std::vector<float> out(static_cast<size_t>(numPlanes(H)) * n * n, -1.0f);
    encodeGame(g, H, sym, out.data());
    return out;
}

static float at(const std::vector<float>& planes, int plane, int point, int n = 5) {
    return planes[static_cast<size_t>(plane) * n * n + point];
}

// The plane holds exactly `ones` set to 1 and every other point 0.
static bool planeIs(const std::vector<float>& planes, int plane, const std::vector<int>& ones, int n = 5) {
    for (int p = 0; p < n * n; ++p) {
        bool expectOne = false;
        for (int q : ones) expectOne = expectOne || (q == p);
        if (at(planes, plane, p, n) != (expectOne ? 1.0f : 0.0f)) return false;
    }
    return true;
}

static std::vector<int> all(int n = 5) {
    std::vector<int> v;
    for (int p = 0; p < n * n; ++p) v.push_back(p);
    return v;
}

static void test_plane_counts() {
    assert(numPlanes(1) == 4);
    assert(numPlanes(8) == 18);
    Board b(9);
    std::vector<float> out = encodeB({&b}, 8, 0);
    assert(out.size() == 18u * 81u);
}

static void test_own_and_opponent_swap_with_side_to_move() {
    Board b(5);
    b.setStone(pt(0, 0), Color::Black);
    b.setStone(pt(4, 4), Color::White);

    b.setToMove(Color::Black);
    std::vector<float> black = encodeB({&b}, 1, 0);
    assert(planeIs(black, 0, {pt(0, 0)}));          // own = Black
    assert(planeIs(black, 1, {pt(4, 4)}));          // opponent = White
    assert(planeIs(black, 2, all()));               // Black to move
    assert(planeIs(black, 3, {}));

    b.setToMove(Color::White);
    std::vector<float> white = encodeB({&b}, 1, 0);
    assert(planeIs(white, 0, {pt(4, 4)}));          // own = White
    assert(planeIs(white, 1, {pt(0, 0)}));
    assert(planeIs(white, 2, {}));
    assert(planeIs(white, 3, all()));               // White to move
}

static void test_missing_history_is_zero() {
    Board b(5);
    b.setStone(pt(2, 2), Color::Black);
    std::vector<float> out = encodeB({&b}, 4, 0);
    assert(planeIs(out, 0, {pt(2, 2)}));
    for (int plane = 1; plane < 8; ++plane) assert(planeIs(out, plane, {}));
    assert(planeIs(out, 8, all()));
    assert(planeIs(out, 9, {}));

    // A fresh game: no stones anywhere, Black to move.
    Game g(5, 7.5f, defaultMaxMoves(5));
    std::vector<float> start = encodeG(g, 8, 0);
    for (int plane = 0; plane < 16; ++plane) assert(planeIs(start, plane, {}));
    assert(planeIs(start, 16, all()));
    assert(planeIs(start, 17, {}));
}

static void test_encode_game_history() {
    // Black (0,0), White (4,4), Black (2,2): White to move after 3 moves.
    Game g(5, 7.5f, defaultMaxMoves(5));
    g.play(pt(0, 0));
    g.play(pt(4, 4));
    g.play(pt(2, 2));
    std::vector<float> out = encodeG(g, 4, 0);
    // Own = White throughout, because White is to move now.
    assert(planeIs(out, 0, {pt(4, 4)}));            // current position
    assert(planeIs(out, 1, {pt(0, 0), pt(2, 2)}));
    assert(planeIs(out, 2, {pt(4, 4)}));            // one ply ago
    assert(planeIs(out, 3, {pt(0, 0)}));
    assert(planeIs(out, 4, {}));                    // two plies ago
    assert(planeIs(out, 5, {pt(0, 0)}));
    assert(planeIs(out, 6, {}));                    // the empty start position
    assert(planeIs(out, 7, {}));
    assert(planeIs(out, 8, {}));                    // not Black to move
    assert(planeIs(out, 9, all()));                 // White to move

    // H = 2 keeps only the two most recent positions.
    std::vector<float> shortH = encodeG(g, 2, 0);
    assert(shortH.size() == 6u * 25u);
    assert(planeIs(shortH, 2, {pt(4, 4)}));
    assert(planeIs(shortH, 3, {pt(0, 0)}));
    assert(planeIs(shortH, 5, all()));

    // encodeGame is encodeBoards over the game's boards, newest first.
    std::vector<float> manual = encodeB({&g.boardAt(3), &g.boardAt(2), &g.boardAt(1), &g.boardAt(0)}, 4, 0);
    assert(manual == out);
}

static void test_symmetry_consistency() {
    Game g(5, 7.5f, defaultMaxMoves(5));
    for (int m : {pt(0, 1), pt(3, 3), pt(1, 0), pt(4, 2)}) g.play(m);
    const int H = 3, n = 5, P = numPlanes(H);
    const std::vector<float> base = encodeG(g, H, 0);
    for (int s = 0; s < kNumSymmetries; ++s) {
        const std::vector<float> t = encodeG(g, H, s);
        for (int plane = 0; plane < P; ++plane) {
            for (int p = 0; p < n * n; ++p) {
                assert(at(t, plane, transformPoint(s, p, n)) == at(base, plane, p));
            }
        }
    }
    // A non-identity transform really moves the stones.
    assert(encodeG(g, H, 1) != base);
}

int main() {
    test_plane_counts();
    test_own_and_opponent_swap_with_side_to_move();
    test_missing_history_is_zero();
    test_encode_game_history();
    test_symmetry_consistency();
    std::printf("test_encode: all tests passed\n");
    return 0;
}
```

- [ ] **Step 7: Run test to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL compiling `tests/test_encode.cpp` with `fatal error: 'azgo/encode.hpp' file not found`.

- [ ] **Step 8: Write the encode header**

Create `include/azgo/encode.hpp`:

```cpp
// What the network sees.
//
// A position is fed to the network as a stack of n x n binary planes, always
// from the point of view of the player about to move: "my stones" and "their
// stones" for the current position and the H - 1 before it (history reveals
// ko and recent fights), then one plane that is all ones if Black is to move
// and one that is all ones if White is. That constant plane also lets a
// zero-padded 3x3 convolution tell the board edge apart from empty points.
#pragma once
#include <cstdint>
#include <vector>
#include "azgo/board.hpp"

namespace azgo {

class Game;

// 2H + 2: (own, opponent) stones for each of the last H positions, then a
// "black to move" plane and a "white to move" plane (exactly one is all ones).
int numPlanes(int history);

// stones[k] points to n*n Color bytes for the position k plies ago (k = 0 is
// current). stones.size() may be < H; missing positions are zero planes.
// "Own" means toMove (the side to move NOW) for every history plane.
// Writes numPlanes(H) * n * n floats to out ([plane][point]) with symmetry
// `sym` applied: value for point p lands at transformPoint(sym, p, n).
void encodePlanes(const std::vector<const uint8_t*>& stones, int n, Color toMove,
                  int H, int sym, float* out);

// history[0] is the current board, history[1] the previous one, ...
void encodeBoards(const std::vector<const Board*>& history, int H, int sym, float* out);

// Current position of a game, history taken from the game's earlier boards.
void encodeGame(const Game& game, int H, int sym, float* out);

} // namespace azgo
```

- [ ] **Step 9: Write the encode implementation**

Create `src/encode.cpp`:

```cpp
#include "azgo/encode.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include "azgo/game.hpp"
#include "azgo/symmetry.hpp"

namespace azgo {

int numPlanes(int history) {
    return 2 * history + 2;
}

void encodePlanes(const std::vector<const uint8_t*>& stones, int n, Color toMove,
                  int H, int sym, float* out) {
    assert(n >= kMinBoardSize && n <= kMaxBoardSize);
    assert(H >= 1);
    const int area = n * n;
    std::fill(out, out + static_cast<size_t>(numPlanes(H)) * area, 0.0f);

    // Where each point lands under the symmetry, computed once per call.
    std::array<int, kMaxBoardSize * kMaxBoardSize> dest;
    for (int p = 0; p < area; ++p) dest[p] = transformPoint(sym, p, n);

    const uint8_t own = static_cast<uint8_t>(toMove);
    const uint8_t opp = static_cast<uint8_t>(opponent(toMove));
    const int available = std::min(H, static_cast<int>(stones.size()));
    for (int k = 0; k < available; ++k) {
        // "Own" is the player to move now, in every history plane, so the
        // network always sees the game from the perspective it must act in.
        float* ownPlane = out + static_cast<size_t>(2 * k) * area;
        float* oppPlane = ownPlane + area;
        const uint8_t* s = stones[static_cast<size_t>(k)];
        for (int p = 0; p < area; ++p) {
            if (s[p] == own) ownPlane[dest[p]] = 1.0f;
            else if (s[p] == opp) oppPlane[dest[p]] = 1.0f;
        }
    }
    // Planes 2H (Black to move) and 2H + 1 (White to move): exactly one is all ones.
    const int colorPlane = 2 * H + (toMove == Color::Black ? 0 : 1);
    float* plane = out + static_cast<size_t>(colorPlane) * area;
    std::fill(plane, plane + area, 1.0f);
}

void encodeBoards(const std::vector<const Board*>& history, int H, int sym, float* out) {
    assert(!history.empty());
    const int n = history[0]->size();
    const int area = n * n;
    const int count = std::min(H, static_cast<int>(history.size()));
    std::vector<uint8_t> bytes(static_cast<size_t>(count) * area);
    std::vector<const uint8_t*> stones(static_cast<size_t>(count));
    for (int k = 0; k < count; ++k) {
        assert(history[static_cast<size_t>(k)]->size() == n);
        uint8_t* dst = bytes.data() + static_cast<size_t>(k) * area;
        history[static_cast<size_t>(k)]->exportStones(dst);
        stones[static_cast<size_t>(k)] = dst;
    }
    encodePlanes(stones, n, history[0]->toMove(), H, sym, out);
}

void encodeGame(const Game& game, int H, int sym, float* out) {
    std::vector<const Board*> history;
    for (int k = 0; k < H && k <= game.moveCount(); ++k) {
        history.push_back(&game.boardAt(game.moveCount() - k));
    }
    encodeBoards(history, H, sym, out);
}

} // namespace azgo
```

- [ ] **Step 10: Run tests to verify they pass**

Run:
```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```
Expected: `test_encode`, `test_symmetry`, `test_board`, `test_game`, `test_rng` all `Passed`, then `100% tests passed out of 5`.

- [ ] **Step 11: Commit**

```bash
git add include/azgo/symmetry.hpp src/symmetry.cpp include/azgo/encode.hpp src/encode.cpp \
        tests/test_symmetry.cpp tests/test_encode.cpp
git commit -F - <<'EOF'
Add board symmetries and the network input planes

The network sees a position as 2H + 2 planes from the side to move's
point of view: own and opponent stones for the last H positions, then
a Black-to-move and a White-to-move plane, exactly one of them all
ones so zero-padded convolutions can tell the edge from empty points.
encodePlanes works on raw stone bytes so the replay buffer can encode
stored games without rebuilding Boards; encodeBoards and encodeGame are
thin adapters for MCTS and analyze.

The 8 dihedral symmetries are applied while encoding (and to policy
targets later), giving 8x data augmentation for free. Tests check that
each symmetry composed with its inverse is the identity on even and odd
boards and that encoding under a symmetry equals transforming the
identity encoding point by point.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

> **Section notes (Tasks 5–9).** Every code block in this section was compiled and its tests run (clang 21, `-std=c++17 -O2 -Wall -Wextra -UNDEBUG`, plus an ASan/UBSan pass, and a `-DAZ_USE_BLAS` build against Accelerate) against a stand-in `Rng` that follows the Task 1 contract. Four things for the reviewer:
>
> 1. **Deviation from the Task 7 decision "im2col buffers per conv".** `Workspace` keeps **one** shared im2col buffer (`col_`, sized for the largest 3×3 conv) and `Network::backward` rebuilds each 3×3 conv's columns with `Backend::im2col` right before that conv's `backward`. This still satisfies the `Conv2d::backward` contract (it receives exactly the columns `forward` produced). Reason: a per-conv buffer is `9·F·B·n²` floats per conv. For the 19×19 preset (10 blocks × 128 filters) at batch 256 that is about 425 MB per conv and 9 GB for 21 convs. Even inference workspaces (batch 8, one per worker) would need about 280 MB each. Rebuilding costs one im2col per conv per training step, which is small next to the three gemms per conv. Every other buffer listed in the decision is kept per layer and allocated in the constructor.
> 2. **Whole-network gradient check step size.** Task 6's per-layer checks use h = 1e-2 as decided. The Task 7 whole-network check keeps the Task 6 *tolerance* but uses h = 2e-3 with a retry at 5e-4, plus a kink test. The network has about 1,000 ReLUs directly behind batch norms, and moving one weight shifts every unit in a channel, so a 1e-2 probe crosses some ReLU kink in most entries (measured: 1–20% of entries fail with correct gradients, depending on the seed). The rule used here (match at 2e-3 or 5e-4, or else be excused as a kink when the one-sided differences disagree, with at most 25% excused) had zero false failures over 150 seeds × all 2,017 entries. It still catches every seeded backward bug I tried: dropped skip gradient, head gradients overwritten instead of summed, missing tanh derivative, missing ReLU mask, swapped policy un-flatten, wrong BN cache.
> 3. **Assumption about Task 1's CMake.** With `AZ_USE_BLAS=ON`, the `azgo` target must also get `target_compile_definitions(azgo PUBLIC AZ_USE_BLAS)` and link Accelerate (macOS) or OpenBLAS. `src/backend.cpp` uses that macro for `blasAvailable()`/`makeBackend("blas")`. Task 5 Step 5 checks this: the BLAS build must *not* print `blas not built: skipping`.
> 4. **Contract headers are unchanged.** The only additions are the teaching comment blocks and the private members the contract marks implementation-defined: `Workspace` buffers; `Network` layers, `Mode`, `forward`/`loss`/`backward`; the stream references in `BinaryWriter`/`BinaryReader`. `src/network.cpp` includes `azgo/board.hpp` (Task 2) only for `kMinBoardSize`/`kMaxBoardSize`.

### Task 5: Compute backend (gemm, im2col, col2im)

**Files:**
- Create: `include/azgo/backend.hpp`, `src/backend.cpp`, `src/backend_blas.cpp`
- Test: `tests/test_backend.cpp`

**Interfaces:**
- Consumes: `azgo::Rng(uint64_t)`, `Rng::uniform()` (Task 1, tests only). Task 1's CMake compiles `src/backend_blas.cpp` only when `AZ_USE_BLAS` is ON. In that case it defines the `AZ_USE_BLAS` macro on `azgo` and links Accelerate/OpenBLAS.
- Produces: `azgo::Backend` (`name`, `gemm`, virtual `im2col`/`col2im` with default implementations), `azgo::NaiveBackend(int threads = 1)`, `bool blasAvailable()`, `std::unique_ptr<Backend> makeBackend(const std::string& name, int threads = 1)`. These match contract.md exactly. `BlasBackend` stays private to `src/backend_blas.cpp`, and `makeBlasBackend()` is declared `extern` only inside `src/backend.cpp`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_backend.cpp`:

```cpp
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/rng.hpp"

using namespace azgo;

namespace {

struct Shape { int M, N, K; };
const Shape kShapes[] = {{7, 5, 3}, {33, 17, 29}, {1, 1, 1}, {64, 81, 576}};
struct Scale { float alpha, beta; };
const Scale kScales[] = {{1.0f, 0.0f}, {0.5f, 1.0f}, {-2.0f, 0.25f}};

std::vector<float> randomVector(Rng& rng, size_t count) {
    std::vector<float> v(count);
    for (float& x : v) x = static_cast<float>(rng.uniform() * 2.0 - 1.0);
    return v;
}

// Straight from the definition, accumulated in double.
void referenceGemm(bool transA, bool transB, int M, int N, int K, float alpha,
                   const float* A, const float* B, float beta, float* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int p = 0; p < K; ++p) {
                double a = transA ? A[p * M + i] : A[i * K + p];
                double b = transB ? B[j * K + p] : B[p * N + j];
                sum += a * b;
            }
            double old = beta == 0.0f ? 0.0 : static_cast<double>(beta) * C[i * N + j];
            C[i * N + j] = static_cast<float>(alpha * sum + old);
        }
    }
}

bool closeRelative(double actual, double expected, double tol) {
    return std::fabs(actual - expected) <= tol * std::max(1.0, std::fabs(expected));
}

// Runs one gemm on `be` and on the reference with identical inputs (C is
// prefilled with random values, or NaN when beta == 0) and compares.
void checkAgainstReference(const Backend& be, bool tA, bool tB, Shape s, Scale sc, Rng& rng) {
    std::vector<float> A = randomVector(rng, static_cast<size_t>(s.M) * s.K);
    std::vector<float> B = randomVector(rng, static_cast<size_t>(s.K) * s.N);
    std::vector<float> C0 = randomVector(rng, static_cast<size_t>(s.M) * s.N);
    if (sc.beta == 0.0f) std::fill(C0.begin(), C0.end(), std::numeric_limits<float>::quiet_NaN());
    std::vector<float> C = C0, expected = C0;
    be.gemm(tA, tB, s.M, s.N, s.K, sc.alpha, A.data(), B.data(), sc.beta, C.data());
    referenceGemm(tA, tB, s.M, s.N, s.K, sc.alpha, A.data(), B.data(), sc.beta, expected.data());
    for (size_t i = 0; i < C.size(); ++i) {
        assert(std::isfinite(C[i]));
        if (!closeRelative(C[i], expected[i], 1e-4)) {
            std::printf("%s gemm mismatch tA=%d tB=%d M=%d N=%d K=%d at %zu: %g vs %g\n",
                        be.name(), tA, tB, s.M, s.N, s.K, i, C[i], expected[i]);
            assert(false);
        }
    }
}

void checkAllCombinations(const Backend& be) {
    Rng rng(11);
    for (const Shape& s : kShapes)
        for (const Scale& sc : kScales)
            for (int t = 0; t < 4; ++t)
                checkAgainstReference(be, (t & 1) != 0, (t & 2) != 0, s, sc, rng);
}

} // namespace

static void test_naive_gemm_matches_reference() {
    NaiveBackend be;
    checkAllCombinations(be);
}

static void test_beta_zero_ignores_nan_in_c() {
    NaiveBackend be;
    const float A[] = {1, 2, 3, 4, 5, 6};          // 2 x 3
    const float B[] = {1, 0, 0, 1, 1, 1};          // 3 x 2
    float C[4];
    std::fill(C, C + 4, std::numeric_limits<float>::quiet_NaN());
    be.gemm(false, false, 2, 2, 3, 1.0f, A, B, 0.0f, C);
    // Row 0: [1*1 + 2*0 + 3*1, 1*0 + 2*1 + 3*1] = [4, 5]; row 1: [10, 11].
    assert(C[0] == 4.0f && C[1] == 5.0f && C[2] == 10.0f && C[3] == 11.0f);
}

static void test_threads_are_bitwise_identical() {
    // 64 x 300 x 576 is far above the threading threshold (2^18 multiply-adds).
    const int M = 64, N = 300, K = 576;
    Rng rng(5);
    std::vector<float> A = randomVector(rng, static_cast<size_t>(M) * K);
    std::vector<float> B = randomVector(rng, static_cast<size_t>(K) * N);
    std::vector<float> C0 = randomVector(rng, static_cast<size_t>(M) * N);
    NaiveBackend one(1), three(3), four(4);
    for (int t = 0; t < 4; ++t) {
        const bool tA = (t & 1) != 0, tB = (t & 2) != 0;
        std::vector<float> c1 = C0, c3 = C0, c4 = C0;
        one.gemm(tA, tB, M, N, K, 0.75f, A.data(), B.data(), 0.5f, c1.data());
        three.gemm(tA, tB, M, N, K, 0.75f, A.data(), B.data(), 0.5f, c3.data());
        four.gemm(tA, tB, M, N, K, 0.75f, A.data(), B.data(), 0.5f, c4.data());
        assert(std::memcmp(c1.data(), c3.data(), c1.size() * sizeof(float)) == 0);
        assert(std::memcmp(c1.data(), c4.data(), c1.size() * sizeof(float)) == 0);
    }
}

static void test_im2col_hand_example() {
    // One 3x3 image 1..9, 3x3 patches. col is 9 rows (patch offsets) x 9
    // columns (output pixels); column q holds the 3x3 neighborhood of pixel q.
    NaiveBackend be;
    const float in[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float col[81];
    be.im2col(in, 1, 1, 3, 3, col);
    for (int r = 0; r < 9; ++r) {
        assert(col[r * 9 + 4] == in[r]);          // center pixel: its neighborhood is the whole image
        assert(col[4 * 9 + r] == in[r]);          // center offset row: the image itself
    }
    const float topLeft[9] = {0, 0, 0, 0, 1, 2, 0, 4, 5};      // pixel (0,0), zero padded
    const float bottomRight[9] = {5, 6, 0, 8, 9, 0, 0, 0, 0};  // pixel (2,2)
    for (int r = 0; r < 9; ++r) {
        assert(col[r * 9 + 0] == topLeft[r]);
        assert(col[r * 9 + 8] == bottomRight[r]);
    }
}

static void test_im2col_k1_is_copy() {
    NaiveBackend be;
    Rng rng(3);
    const int C = 2, B = 3, n = 5;
    std::vector<float> in = randomVector(rng, static_cast<size_t>(C) * B * n * n);
    std::vector<float> col(in.size(), -1.0f);
    be.im2col(in.data(), C, B, n, 1, col.data());
    assert(col == in);
}

static void test_col2im_is_adjoint_of_im2col() {
    NaiveBackend be;
    Rng rng(4);
    const int C = 2, B = 3, n = 5, k = 3;
    const size_t inSize = static_cast<size_t>(C) * B * n * n;
    const size_t colSize = inSize * k * k;
    std::vector<float> x = randomVector(rng, inSize), y = randomVector(rng, colSize);
    std::vector<float> colX(colSize), backY(inSize);
    be.im2col(x.data(), C, B, n, k, colX.data());
    be.col2im(y.data(), C, B, n, k, backY.data());
    // <im2col(x), y> == <x, col2im(y)> is the definition of "adjoint".
    double lhs = 0.0, rhs = 0.0;
    for (size_t i = 0; i < colSize; ++i) lhs += static_cast<double>(colX[i]) * y[i];
    for (size_t i = 0; i < inSize; ++i) rhs += static_cast<double>(x[i]) * backY[i];
    assert(std::fabs(lhs - rhs) <= 1e-3);
}

static void test_col2im_overwrites_and_counts_overlaps() {
    // All-ones col: each pixel receives one contribution per patch that
    // covers it: 4 at corners, 6 on edges, 9 in the middle. The NaN prefill
    // proves col2im overwrites rather than accumulates into its output.
    NaiveBackend be;
    std::vector<float> col(81, 1.0f);
    float in[9];
    std::fill(in, in + 9, std::numeric_limits<float>::quiet_NaN());
    be.col2im(col.data(), 1, 1, 3, 3, in);
    const float expected[9] = {4, 6, 4, 6, 9, 6, 4, 6, 4};
    for (int p = 0; p < 9; ++p) assert(in[p] == expected[p]);
}

static void test_make_backend() {
    std::unique_ptr<Backend> naive = makeBackend("naive", 2);
    assert(std::string(naive->name()) == "naive");
    bool threw = false;
    try {
        makeBackend("bogus");
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()).find("bogus") != std::string::npos;
    }
    assert(threw);
    if (!blasAvailable()) {
        threw = false;
        try {
            makeBackend("blas");
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
}

static void test_blas_matches_naive() {
    if (!blasAvailable()) {
        std::printf("blas not built: skipping\n");
        return;
    }
    std::unique_ptr<Backend> blas = makeBackend("blas");
    assert(std::string(blas->name()) == "blas");
    checkAllCombinations(*blas);   // same reference, so blas and naive agree within 1e-4

    // Direct naive-vs-blas comparison on the largest shape, all four combinations.
    NaiveBackend naive;
    Rng rng(9);
    const int M = 64, N = 81, K = 576;
    std::vector<float> A = randomVector(rng, static_cast<size_t>(M) * K);
    std::vector<float> B = randomVector(rng, static_cast<size_t>(K) * N);
    for (int t = 0; t < 4; ++t) {
        std::vector<float> c1(static_cast<size_t>(M) * N), c2(c1.size());
        naive.gemm(t & 1, t & 2, M, N, K, 1.0f, A.data(), B.data(), 0.0f, c1.data());
        blas->gemm(t & 1, t & 2, M, N, K, 1.0f, A.data(), B.data(), 0.0f, c2.data());
        for (size_t i = 0; i < c1.size(); ++i) assert(closeRelative(c2[i], c1[i], 1e-4));
    }
}

int main() {
    test_naive_gemm_matches_reference();
    test_beta_zero_ignores_nan_in_c();
    test_threads_are_bitwise_identical();
    test_im2col_hand_example();
    test_im2col_k1_is_copy();
    test_col2im_is_adjoint_of_im2col();
    test_col2im_overwrites_and_counts_overlaps();
    test_make_backend();
    test_blas_matches_naive();
    std::printf("test_backend: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target test_backend`
Expected: FAIL at compile time with `fatal error: 'azgo/backend.hpp' file not found`.

- [ ] **Step 3: Write the header**

Create `include/azgo/backend.hpp`:

```cpp
// The numeric core. Almost all of a ResNet's time goes into one operation:
// general matrix multiply, C = alpha * A * B + beta * C. A convolution
// becomes a gemm once im2col copies every k x k input patch into a column,
// so a layer is "unfold, multiply" forward and "multiply, fold back" (col2im)
// backward. Keeping gemm/im2col/col2im behind this small interface lets the
// same layer code run on plain C++ loops, on BLAS (Apple Accelerate uses the
// CPU's matrix units), or on a GPU backend later.
#pragma once
#include <memory>
#include <string>

namespace azgo {

// Where the arithmetic happens. Layers call only these operations, so a
// different backend (BLAS today, Metal on the GPU later) needs no layer changes.
class Backend {
public:
    virtual ~Backend() = default;
    virtual const char* name() const = 0;

    // Row-major C[M x N] = alpha * op(A) * op(B) + beta * C, tightly packed.
    // op(A) is M x K (A is stored K x M when transA); op(B) is K x N
    // (B is stored N x K when transB). beta == 0 means C is overwritten.
    virtual void gemm(bool transA, bool transB, int M, int N, int K, float alpha,
                      const float* A, const float* B, float beta, float* C) const = 0;

    // 'Same'-padded k x k patch unfolding (k odd). in: [C][B][n*n];
    // col: [C*k*k][B*n*n], row index (c*k + dy)*k + dx.
    virtual void im2col(const float* in, int C, int B, int n, int k, float* col) const;
    // Adjoint of im2col: OVERWRITES in with the sum of contributions.
    virtual void col2im(const float* col, int C, int B, int n, int k, float* in) const;
};

// Blocked loops, packing transposed operands first. threads > 1 splits the
// rows of C across std::threads (each element computed by exactly one thread,
// so results do not depend on the thread count).
class NaiveBackend : public Backend {
public:
    explicit NaiveBackend(int threads = 1);
    const char* name() const override { return "naive"; }
    void gemm(bool transA, bool transB, int M, int N, int K, float alpha,
              const float* A, const float* B, float beta, float* C) const override;
private:
    int threads_;
};

bool blasAvailable();  // true when built with -DAZ_USE_BLAS=ON

// "naive" or "blas". threads applies to naive only; BLAS is kept
// single-threaded (our worker threads supply the parallelism).
// Throws std::invalid_argument for an unknown name or unavailable BLAS.
std::unique_ptr<Backend> makeBackend(const std::string& name, int threads = 1);

} // namespace azgo
```

- [ ] **Step 4: Write the implementation**

Create `src/backend.cpp`:

```cpp
#include "azgo/backend.hpp"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <vector>

namespace azgo {

#ifdef AZ_USE_BLAS
// Defined in src/backend_blas.cpp, which is compiled only with -DAZ_USE_BLAS=ON.
extern std::unique_ptr<Backend> makeBlasBackend();
#endif

// ---------------------------------------------------------------- im2col

void Backend::im2col(const float* in, int C, int B, int n, int k, float* col) const {
    const int r = k / 2;                          // 'same' padding: k = 3 reaches one point out
    const size_t area = static_cast<size_t>(n) * n;
    const size_t M = static_cast<size_t>(B) * area;
    for (int c = 0; c < C; ++c) {
        for (int dy = 0; dy < k; ++dy) {
            for (int dx = 0; dx < k; ++dx) {
                // One row of col holds, for every output pixel, the input value
                // at offset (dy - r, dx - r) from it; off-board reads are 0.
                float* row = col + static_cast<size_t>((c * k + dy) * k + dx) * M;
                const int oy = dy - r, ox = dx - r;
                for (int b = 0; b < B; ++b) {
                    const float* src = in + (static_cast<size_t>(c) * B + b) * area;
                    float* dst = row + static_cast<size_t>(b) * area;
                    for (int y = 0; y < n; ++y) {
                        const int sy = y + oy;
                        float* out = dst + static_cast<size_t>(y) * n;
                        if (sy < 0 || sy >= n) {
                            std::fill(out, out + n, 0.0f);
                            continue;
                        }
                        const float* srcRow = src + static_cast<size_t>(sy) * n;
                        for (int x = 0; x < n; ++x) {
                            const int sx = x + ox;
                            out[x] = (sx >= 0 && sx < n) ? srcRow[sx] : 0.0f;
                        }
                    }
                }
            }
        }
    }
}

void Backend::col2im(const float* col, int C, int B, int n, int k, float* in) const {
    const int r = k / 2;
    const size_t area = static_cast<size_t>(n) * n;
    const size_t M = static_cast<size_t>(B) * area;
    std::fill(in, in + static_cast<size_t>(C) * M, 0.0f);
    // Exactly the loops of im2col with the assignment reversed: every place
    // im2col copied in[p] into col, col2im adds that col entry back into in[p].
    // That makes col2im the transpose (adjoint) of im2col, which is what the
    // backward pass of a convolution needs.
    for (int c = 0; c < C; ++c) {
        for (int dy = 0; dy < k; ++dy) {
            for (int dx = 0; dx < k; ++dx) {
                const float* row = col + static_cast<size_t>((c * k + dy) * k + dx) * M;
                const int oy = dy - r, ox = dx - r;
                for (int b = 0; b < B; ++b) {
                    float* dst = in + (static_cast<size_t>(c) * B + b) * area;
                    const float* src = row + static_cast<size_t>(b) * area;
                    for (int y = 0; y < n; ++y) {
                        const int sy = y + oy;
                        if (sy < 0 || sy >= n) continue;
                        float* dstRow = dst + static_cast<size_t>(sy) * n;
                        const float* srcRow = src + static_cast<size_t>(y) * n;
                        for (int x = 0; x < n; ++x) {
                            const int sx = x + ox;
                            if (sx >= 0 && sx < n) dstRow[sx] += srcRow[x];
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- naive gemm

namespace {

// Blocking keeps a kBlockK x kBlockN tile of B (128 x 512 floats = 256 KB)
// hot in cache while every row of A in this thread's range sweeps over it.
constexpr int kBlockK = 128;
constexpr int kBlockN = 512;

// Below this many multiply-adds, starting threads costs more than it saves.
constexpr size_t kThreadingThreshold = size_t(1) << 18;

// dst (cols x rows) = transpose of src (rows x cols).
void transposeInto(const float* src, int rows, int cols, std::vector<float>& dst) {
    dst.resize(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            dst[static_cast<size_t>(c) * rows + r] = src[static_cast<size_t>(r) * cols + c];
}

// Rows [r0, r1) of C = alpha * A * B + beta * C with A (M x K) and B (K x N)
// both non-transposed. i-k-j order: row i of C accumulates alpha * A[i][p]
// times row p of B, so the innermost loop walks C and B contiguously and the
// compiler vectorizes it.
void gemmRows(int r0, int r1, int N, int K, float alpha, const float* A, const float* B,
              float beta, float* C) {
    for (int i = r0; i < r1; ++i) {
        float* c = C + static_cast<size_t>(i) * N;
        if (beta == 0.0f) {
            std::fill(c, c + N, 0.0f);            // never read C: it may hold NaN
        } else if (beta != 1.0f) {
            for (int j = 0; j < N; ++j) c[j] *= beta;
        }
    }
    for (int j0 = 0; j0 < N; j0 += kBlockN) {
        const int j1 = std::min(N, j0 + kBlockN);
        for (int p0 = 0; p0 < K; p0 += kBlockK) {
            const int p1 = std::min(K, p0 + kBlockK);
            for (int i = r0; i < r1; ++i) {
                float* c = C + static_cast<size_t>(i) * N;
                const float* a = A + static_cast<size_t>(i) * K;
                for (int p = p0; p < p1; ++p) {
                    const float s = alpha * a[p];
                    const float* b = B + static_cast<size_t>(p) * N;
                    for (int j = j0; j < j1; ++j) c[j] += s * b[j];
                }
            }
        }
    }
}

} // namespace

NaiveBackend::NaiveBackend(int threads) : threads_(std::max(1, threads)) {}

void NaiveBackend::gemm(bool transA, bool transB, int M, int N, int K, float alpha,
                        const float* A, const float* B, float beta, float* C) const {
    if (M <= 0 || N <= 0) return;
    // Pack transposed operands once into row-major scratch, so the kernel
    // only ever sees the contiguous layout. thread_local: several worker
    // threads may share one backend, and reusing the vectors avoids an
    // allocation per call.
    thread_local std::vector<float> packA, packB;
    if (transA) {
        transposeInto(A, K, M, packA);            // A is stored K x M
        A = packA.data();
    }
    if (transB) {
        transposeInto(B, N, K, packB);            // B is stored N x K
        B = packB.data();
    }

    const size_t work = static_cast<size_t>(M) * N * K;
    const int workers = (threads_ > 1 && work >= kThreadingThreshold) ? std::min(threads_, M) : 1;
    if (workers == 1) {
        gemmRows(0, M, N, K, alpha, A, B, beta, C);
        return;
    }
    // Split the rows of C into contiguous chunks, one per thread. Each element
    // of C is computed by exactly one thread with exactly the same sequence of
    // floating-point operations (the loop order over p does not depend on the
    // row range), so the result is bitwise identical for any thread count.
    const int chunk = (M + workers - 1) / workers;
    std::vector<std::thread> pool;
    for (int t = 1; t < workers; ++t) {
        const int r0 = t * chunk, r1 = std::min(M, r0 + chunk);
        if (r0 >= r1) break;
        pool.emplace_back(gemmRows, r0, r1, N, K, alpha, A, B, beta, C);
    }
    gemmRows(0, std::min(M, chunk), N, K, alpha, A, B, beta, C);   // chunk 0 on this thread
    for (std::thread& t : pool) t.join();
}

// ---------------------------------------------------------------- factory

bool blasAvailable() {
#ifdef AZ_USE_BLAS
    return true;
#else
    return false;
#endif
}

std::unique_ptr<Backend> makeBackend(const std::string& name, int threads) {
    if (name == "naive") return std::make_unique<NaiveBackend>(threads);
    if (name == "blas") {
#ifdef AZ_USE_BLAS
        return makeBlasBackend();
#else
        throw std::invalid_argument("backend 'blas' not available: rebuild with -DAZ_USE_BLAS=ON");
#endif
    }
    throw std::invalid_argument("unknown backend '" + name + "' (expected naive or blas)");
}

} // namespace azgo
```

Create `src/backend_blas.cpp` (only compiled with `-DAZ_USE_BLAS=ON`):

```cpp
// Compiled only when CMake is configured with -DAZ_USE_BLAS=ON.
#include "azgo/backend.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory>

#ifdef __APPLE__
// Select Accelerate's current CBLAS headers (the legacy ones are deprecated
// since macOS 13.3); the calling convention is the standard CBLAS one.
// CMake already defines it for the azgo target (Task 1); the guard keeps a
// second definition from triggering -Wmacro-redefined.
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
extern "C" void openblas_set_num_threads(int);
#endif

namespace azgo {

namespace {

// Only gemm is delegated; im2col/col2im are memory shuffles that BLAS does
// not provide, so the Backend defaults are used.
class BlasBackend : public Backend {
public:
    const char* name() const override { return "blas"; }

    void gemm(bool transA, bool transB, int M, int N, int K, float alpha,
              const float* A, const float* B, float beta, float* C) const override {
        if (M <= 0 || N <= 0) return;
        if (K <= 0) {
            // BLAS requires leading dimensions >= 1; the product is empty anyway.
            const size_t count = static_cast<size_t>(M) * N;
            if (beta == 0.0f) std::fill(C, C + count, 0.0f);
            else for (size_t i = 0; i < count; ++i) C[i] *= beta;
            return;
        }
        // Row-major leading dimensions are the STORED row lengths:
        // A is M x K (or K x M when transposed), B is K x N (or N x K), C is M x N.
        // Like reference BLAS, cblas_sgemm does not read C when beta == 0.
        cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                    transB ? CblasTrans : CblasNoTrans, M, N, K, alpha,
                    A, transA ? M : K, B, transB ? K : N, beta, C, N);
    }
};

// Called once, before the first BLAS call. Self-play and arena already run
// one worker thread per core; a multithreaded BLAS inside each worker would
// oversubscribe the cores and make timings erratic.
bool configureSingleThreaded() {
#ifdef __APPLE__
    // Accelerate reads this when it first spins up its thread pool. (On Apple
    // Silicon, sgemm runs on the AMX/SME matrix unit from a single thread
    // anyway, so this costs little.) setenv is not thread-safe, which is why
    // it happens once, from the first makeBackend("blas") call.
    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
#else
    openblas_set_num_threads(1);
#endif
    return true;
}

} // namespace

std::unique_ptr<Backend> makeBlasBackend() {
    static const bool configured = configureSingleThreaded();
    (void)configured;
    return std::make_unique<BlasBackend>();
}

} // namespace azgo
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -V -R test_backend`
Expected: the output contains `blas not built: skipping` and `test_backend: all tests passed`, and ctest reports `100% tests passed`.

Then the BLAS variant:

Run: `cmake -S . -B build-blas -DCMAKE_BUILD_TYPE=Release -DAZ_USE_BLAS=ON && cmake --build build-blas -j && ctest --test-dir build-blas --output-on-failure -V -R test_backend`
Expected: `test_backend: all tests passed` with **no** `blas not built: skipping` line. That line would mean the `AZ_USE_BLAS` macro did not reach `src/backend.cpp`, so fix Task 1's CMake. `100% tests passed`.

- [ ] **Step 6: Commit**

```bash
git add include/azgo/backend.hpp src/backend.cpp src/backend_blas.cpp tests/test_backend.cpp
git commit -F - <<'EOF'
Add the compute backend: gemm, im2col and col2im

Every layer of the ResNet reduces to a matrix multiply once convolutions
are unfolded with im2col, so this small interface is the only place the
arithmetic lives. The naive backend packs transposed operands and runs an
i-k-j kernel the compiler vectorizes; with several threads it splits rows
of C, so each element is still computed by one thread and results are
bitwise identical for any thread count, which keeps training
deterministic. An optional BLAS backend (Accelerate or OpenBLAS, kept
single-threaded) sits behind the same interface and is checked against a
double-precision reference.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

### Task 6: Layers (Conv2d, BatchNorm2d, Linear, ReLU)

**Files:**
- Create: `include/azgo/layers.hpp`, `src/layers.cpp`
- Test: `tests/test_layers.cpp`

**Interfaces:**
- Consumes: `Backend::gemm`, `Backend::im2col`, `Backend::col2im`, `NaiveBackend` (Task 5). `Rng(uint64_t)`, `Rng::normal()`, `Rng::uniform()` (Task 1).
- Produces: `azgo::Conv2d` (`cin`, `cout`, `k`, `w`, `gw`, `init`, `colSize`, `forward`, `backward`), `azgo::BatchNorm2d` (`kEps`, `kMomentum`, `c`, `gamma`, `beta`, `ggamma`, `gbeta`, `runMean`, `runVar`, `forwardInfer`, `forwardTrain`, `backward`), `azgo::Linear` (`in`, `out`, `w`, `b`, `gw`, `gb`, `init`, `forward`, `backward`), `reluForward`, `reluBackward`. These match contract.md exactly. Extra guarantees Task 7 relies on: `BatchNorm2d::forwardTrain`/`forwardInfer` allow `y == x`, and `BatchNorm2d::backward` allows `dx == dy` (all element-wise after the per-channel sums).

- [ ] **Step 1: Write the failing test**

Create `tests/test_layers.cpp`:

```cpp
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/layers.hpp"
#include "azgo/rng.hpp"

using namespace azgo;

static bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

namespace {

constexpr float kH = 1e-2f;   // finite-difference step

std::vector<float> randomVector(Rng& rng, size_t count, double lo = -1.0, double hi = 1.0) {
    std::vector<float> v(count);
    for (float& x : v) x = static_cast<float>(lo + (hi - lo) * rng.uniform());
    return v;
}

// The scalar loss L = sum_i r_i * y_i: its gradient with respect to y is
// exactly r, so r is what we feed to backward as dy.
double weightedSum(const std::vector<float>& r, const std::vector<float>& y) {
    double s = 0.0;
    for (size_t i = 0; i < r.size(); ++i) s += static_cast<double>(r[i]) * y[i];
    return s;
}

// Compares analytic[i] with the central difference (L(v+h) - L(v-h)) / 2h
// for every entry of `value`. The step actually taken is measured after
// float rounding of v +- h.
template <typename LossFn>
void checkGradient(const char* what, float* value, const float* analytic, size_t count,
                   LossFn loss) {
    for (size_t i = 0; i < count; ++i) {
        const float saved = value[i];
        const float up = saved + kH, down = saved - kH;
        value[i] = up;
        const double lossUp = loss();
        value[i] = down;
        const double lossDown = loss();
        value[i] = saved;
        const double numeric = (lossUp - lossDown) / (static_cast<double>(up) - down);
        const double a = analytic[i];
        const double tol = 2e-2 * std::max(std::fabs(a), std::fabs(numeric)) + 2e-3;
        if (!(std::fabs(a - numeric) <= tol)) {
            std::printf("%s[%zu]: analytic %.6f numeric %.6f\n", what, i, a, numeric);
            assert(false);
        }
    }
}

void checkConvGradients(int k) {
    const int cin = 2, cout = 3, B = 2, n = 4, M = B * n * n;
    NaiveBackend be;
    Rng rng(100 + k);
    Conv2d conv(cin, cout, k);
    conv.init(rng);
    std::vector<float> x = randomVector(rng, static_cast<size_t>(cin) * M);
    std::vector<float> r = randomVector(rng, static_cast<size_t>(cout) * M);
    std::vector<float> col(conv.colSize(B, n)), dcol(col.size()), y(r.size()), dx(x.size());
    auto loss = [&] {
        conv.forward(be, x.data(), B, n, col.data(), y.data());
        return weightedSum(r, y);
    };
    loss();
    conv.backward(be, x.data(), col.data(), r.data(), B, n, dcol.data(), dx.data());
    checkGradient(k == 1 ? "conv1x1 dx" : "conv3x3 dx", x.data(), dx.data(), x.size(), loss);
    checkGradient(k == 1 ? "conv1x1 gw" : "conv3x3 gw", conv.w.data(), conv.gw.data(),
                  conv.w.size(), loss);
}

} // namespace

static void test_conv1x1_gradients() { checkConvGradients(1); }
static void test_conv3x3_gradients() { checkConvGradients(3); }

static void test_batchnorm_gradients() {
    const int c = 3, M = 10;
    Rng rng(7);
    BatchNorm2d bn(c);
    bn.gamma = randomVector(rng, c, 0.5, 1.5);
    bn.beta = randomVector(rng, c, -0.5, 0.5);
    std::vector<float> x = randomVector(rng, static_cast<size_t>(c) * M);
    std::vector<float> r = randomVector(rng, x.size());
    std::vector<float> y(x.size()), xhat(x.size()), invStd(c), dx(x.size());
    // updateRunning = false: probing the loss must not drift the running stats.
    auto loss = [&] {
        bn.forwardTrain(x.data(), M, y.data(), xhat.data(), invStd.data(), false);
        return weightedSum(r, y);
    };
    loss();
    bn.backward(r.data(), xhat.data(), invStd.data(), M, dx.data());
    checkGradient("bn dx", x.data(), dx.data(), x.size(), loss);
    checkGradient("bn ggamma", bn.gamma.data(), bn.ggamma.data(), c, loss);
    checkGradient("bn gbeta", bn.beta.data(), bn.gbeta.data(), c, loss);
    for (int ch = 0; ch < c; ++ch) {
        assert(bn.runMean[ch] == 0.0f && bn.runVar[ch] == 1.0f);
    }
}

static void test_linear_gradients() {
    const int in = 5, out = 4, B = 3;
    NaiveBackend be;
    Rng rng(8);
    Linear fc(in, out);
    fc.init(rng);
    fc.b = randomVector(rng, out);   // nonzero bias so forward's "+ b" is exercised
    std::vector<float> x = randomVector(rng, static_cast<size_t>(B) * in);
    std::vector<float> r = randomVector(rng, static_cast<size_t>(B) * out);
    std::vector<float> y(r.size()), dx(x.size());
    auto loss = [&] {
        fc.forward(be, x.data(), B, y.data());
        return weightedSum(r, y);
    };
    loss();
    fc.backward(be, x.data(), r.data(), B, dx.data());
    checkGradient("linear dx", x.data(), dx.data(), x.size(), loss);
    checkGradient("linear gw", fc.w.data(), fc.gw.data(), fc.w.size(), loss);
    checkGradient("linear gb", fc.b.data(), fc.gb.data(), fc.b.size(), loss);
}

static void test_relu_gradients() {
    Rng rng(9);
    // |x| >= 0.2 > h, so no finite-difference probe crosses the kink at 0.
    std::vector<float> x(40);
    for (float& v : x) v = static_cast<float>((rng.uniform() < 0.5 ? -1 : 1) * (0.2 + 0.8 * rng.uniform()));
    std::vector<float> r = randomVector(rng, x.size());
    std::vector<float> y(x.size());
    auto loss = [&] {
        y = x;
        reluForward(y.data(), y.size());
        return weightedSum(r, y);
    };
    loss();
    std::vector<float> dx = r;
    reluBackward(y.data(), dx.data(), dx.size());
    checkGradient("relu dx", x.data(), dx.data(), x.size(), loss);
    for (size_t i = 0; i < x.size(); ++i) assert(dx[i] == (x[i] > 0 ? r[i] : 0.0f));
}

static void test_conv_forward_hand_example() {
    // All-ones 3x3 kernel: each output is the sum of the input's 3x3
    // neighborhood, with zeros beyond the edge.
    NaiveBackend be;
    Conv2d conv(1, 1, 3);
    std::fill(conv.w.begin(), conv.w.end(), 1.0f);
    const float x[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> col(conv.colSize(1, 3));
    float y[9];
    conv.forward(be, x, 1, 3, col.data(), y);
    const float expected[9] = {12, 21, 16, 27, 45, 33, 24, 39, 28};
    for (int p = 0; p < 9; ++p) assert(y[p] == expected[p]);
}

static void test_batchnorm_inference_hand_values() {
    BatchNorm2d bn(2);
    bn.gamma = {2.0f, 0.5f};
    bn.beta = {1.0f, -1.0f};
    bn.runMean = {1.0f, -2.0f};
    bn.runVar = {4.0f - BatchNorm2d::kEps, 0.25f - BatchNorm2d::kEps};  // sqrt(var + eps) = 2, 0.5
    const float x[6] = {1, 3, -1, -2, -1.5f, 0};
    float y[6];
    bn.forwardInfer(x, 3, y);
    // Channel 0: 2 * (x - 1) / 2 + 1 = x.  Channel 1: 0.5 * (x + 2) / 0.5 - 1 = x + 1.
    const float expected[6] = {1, 3, -1, -1, -0.5f, 1};
    for (int i = 0; i < 6; ++i) assert(near(y[i], expected[i]));
}

static void test_batchnorm_running_stats_update() {
    BatchNorm2d bn(2);
    const int M = 4;
    const float x[8] = {1, 2, 3, 4, 0, 0, 0, 4};
    float y[8], xhat[8], invStd[2];

    bn.forwardTrain(x, M, y, xhat, invStd, false);
    assert(bn.runMean[0] == 0.0f && bn.runVar[0] == 1.0f);   // untouched without updateRunning

    bn.forwardTrain(x, M, y, xhat, invStd, true);
    // Channel 0: mean 2.5, biased var 1.25, unbiased 5/3.
    // Channel 1: mean 1,   biased var 3,    unbiased 4.
    assert(near(bn.runMean[0], 0.1 * 2.5));
    assert(near(bn.runVar[0], 0.9 * 1.0 + 0.1 * (5.0 / 3.0)));
    assert(near(bn.runMean[1], 0.1 * 1.0));
    assert(near(bn.runVar[1], 0.9 * 1.0 + 0.1 * 4.0));
    // Training output uses the batch statistics (biased variance), gamma = 1, beta = 0.
    assert(near(invStd[0], 1.0 / std::sqrt(1.25 + BatchNorm2d::kEps)));
    assert(near(y[0], (1.0 - 2.5) / std::sqrt(1.25 + BatchNorm2d::kEps)));
    assert(near(y[7], (4.0 - 1.0) / std::sqrt(3.0 + BatchNorm2d::kEps)));
}

static void test_he_init_scale() {
    Rng rng(1);
    Conv2d conv(64, 64, 3);                 // w is 64 x 576, fan-in 576
    conv.init(rng);
    double sum = 0.0, sq = 0.0;
    for (float v : conv.w) {
        sum += v;
        sq += static_cast<double>(v) * v;
    }
    const double count = static_cast<double>(conv.w.size());
    const double mean = sum / count;
    const double stddev = std::sqrt(sq / count - mean * mean);
    const double expected = std::sqrt(2.0 / 576.0);
    assert(std::fabs(stddev - expected) <= 0.1 * expected);
    assert(std::fabs(mean) <= 0.1 * expected);

    Linear fc(256, 32);
    fc.b.assign(32, 5.0f);
    fc.init(rng);
    for (float v : fc.b) assert(v == 0.0f);
    double fsq = 0.0;
    for (float v : fc.w) fsq += static_cast<double>(v) * v;
    const double fcStd = std::sqrt(fsq / fc.w.size());
    assert(std::fabs(fcStd - std::sqrt(2.0 / 256.0)) <= 0.1 * std::sqrt(2.0 / 256.0));
}

static void test_gradients_accumulate() {
    // Parameter gradients add up across backward calls (the network zeroes
    // them once per step); a second identical backward doubles them.
    NaiveBackend be;
    Rng rng(12);
    const int B = 2, n = 3, M = B * n * n;

    Conv2d conv(2, 3, 3);
    conv.init(rng);
    std::vector<float> x = randomVector(rng, 2 * M), dy = randomVector(rng, 3 * M);
    std::vector<float> col(conv.colSize(B, n)), dcol(col.size()), y(dy.size()), dx(x.size());
    conv.forward(be, x.data(), B, n, col.data(), y.data());
    conv.backward(be, x.data(), col.data(), dy.data(), B, n, dcol.data(), dx.data());
    std::vector<float> once = conv.gw;
    conv.backward(be, x.data(), col.data(), dy.data(), B, n, dcol.data(), nullptr);   // dx optional
    for (size_t i = 0; i < once.size(); ++i) assert(near(conv.gw[i], 2.0 * once[i], 1e-5 * (1 + std::fabs(once[i]))));

    BatchNorm2d bn(3);
    std::vector<float> xhat(dy.size()), invStd(3), bny(dy.size()), bndx(dy.size());
    bn.forwardTrain(y.data(), M, bny.data(), xhat.data(), invStd.data(), false);
    bn.backward(dy.data(), xhat.data(), invStd.data(), M, bndx.data());
    std::vector<float> g1 = bn.ggamma, b1 = bn.gbeta;
    bn.backward(dy.data(), xhat.data(), invStd.data(), M, bndx.data());
    for (int ch = 0; ch < 3; ++ch) {
        assert(near(bn.ggamma[ch], 2.0 * g1[ch], 1e-5 * (1 + std::fabs(g1[ch]))));
        assert(near(bn.gbeta[ch], 2.0 * b1[ch], 1e-5 * (1 + std::fabs(b1[ch]))));
    }

    Linear fc(4, 3);
    fc.init(rng);
    std::vector<float> fx = randomVector(rng, 8), fdy = randomVector(rng, 6);
    fc.backward(be, fx.data(), fdy.data(), 2, nullptr);
    std::vector<float> w1 = fc.gw, gb1 = fc.gb;
    fc.backward(be, fx.data(), fdy.data(), 2, nullptr);
    for (size_t i = 0; i < w1.size(); ++i) assert(near(fc.gw[i], 2.0 * w1[i], 1e-5 * (1 + std::fabs(w1[i]))));
    for (size_t i = 0; i < gb1.size(); ++i) assert(near(fc.gb[i], 2.0 * gb1[i], 1e-5 * (1 + std::fabs(gb1[i]))));
}

int main() {
    test_conv1x1_gradients();
    test_conv3x3_gradients();
    test_batchnorm_gradients();
    test_linear_gradients();
    test_relu_gradients();
    test_conv_forward_hand_example();
    test_batchnorm_inference_hand_values();
    test_batchnorm_running_stats_update();
    test_he_init_scale();
    test_gradients_accumulate();
    std::printf("test_layers: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target test_layers`
Expected: FAIL at compile time with `fatal error: 'azgo/layers.hpp' file not found`.

- [ ] **Step 3: Write the header**

Create `include/azgo/layers.hpp`:

```cpp
// The building blocks of the ResNet, each with a hand-written backward pass.
// Backprop is the chain rule applied layer by layer in reverse: given dL/dy
// (how the loss changes with a layer's output), a layer produces dL/dx for
// the layer below and accumulates dL/dparameter for the optimizer. Every
// backward here reuses what its forward saved (im2col columns, normalized
// activations), which is why forward and backward share scratch buffers.
// Tensors are plain float arrays; the caller (Network/Workspace) owns them.
#pragma once
#include <cstddef>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/rng.hpp"

namespace azgo {

// All activations are channel-major: [channels][M] with M = B * n * n.

struct Conv2d {
    int cin = 0, cout = 0, k = 1;        // k is 1 or 3; 'same' padding; no bias (BN follows)
    std::vector<float> w, gw;            // [cout][cin * k * k]
    Conv2d() = default;
    Conv2d(int cin, int cout, int k);
    void init(Rng& rng);                 // He normal: stddev sqrt(2 / (cin * k * k))
    size_t colSize(int B, int n) const;  // floats needed for the im2col buffer (0 when k == 1)
    // x [cin][M] -> y [cout][M]. col is scratch of colSize(B, n) (unused if k == 1) and
    // must be passed unchanged to backward.
    void forward(const Backend& be, const float* x, int B, int n, float* col, float* y) const;
    // Accumulates gw += dW. Writes dx (overwrite) unless dx == nullptr.
    // dcol is scratch of colSize(B, n).
    void backward(const Backend& be, const float* x, const float* col, const float* dy,
                  int B, int n, float* dcol, float* dx);
};

struct BatchNorm2d {
    static constexpr float kEps = 1e-5f;
    static constexpr float kMomentum = 0.1f;   // running = (1 - m) * running + m * batch
    int c = 0;
    std::vector<float> gamma, beta, ggamma, gbeta, runMean, runVar;  // gamma=1, beta=0, runVar=1 initially
    BatchNorm2d() = default;
    explicit BatchNorm2d(int c);
    // y = gamma * (x - runMean) / sqrt(runVar + eps) + beta
    void forwardInfer(const float* x, int M, float* y) const;
    // Batch statistics (biased variance). Saves xhat [c][M] and invStd [c].
    // If updateRunning, updates runMean and runVar (unbiased variance M/(M-1)).
    void forwardTrain(const float* x, int M, float* y, float* xhat, float* invStd,
                      bool updateRunning);
    // Accumulates ggamma, gbeta; writes dx (overwrite).
    void backward(const float* dy, const float* xhat, const float* invStd, int M, float* dx);
};

struct Linear {
    int in = 0, out = 0;
    std::vector<float> w, b, gw, gb;     // w [out][in], b [out]
    Linear() = default;
    Linear(int in, int out);
    void init(Rng& rng);                 // He normal weights, zero bias
    void forward(const Backend& be, const float* x, int B, float* y) const;   // x [B][in], y [B][out]
    // Accumulates gw, gb. Writes dx (overwrite) unless dx == nullptr.
    void backward(const Backend& be, const float* x, const float* dy, int B, float* dx);
};

void reluForward(float* x, size_t count);                  // in place
void reluBackward(const float* y, float* dy, size_t count); // dy *= (y > 0), y = relu output

} // namespace azgo
```

- [ ] **Step 4: Write the implementation**

Create `src/layers.cpp`:

```cpp
#include "azgo/layers.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>

namespace azgo {

namespace {

// He initialization: weights ~ N(0, 2 / fanIn) keep the variance of a ReLU
// layer's output close to its input's, so a deep stack starts neither
// exploding nor vanishing.
void heNormal(std::vector<float>& w, int fanIn, Rng& rng) {
    const double stddev = std::sqrt(2.0 / fanIn);
    for (float& v : w) v = static_cast<float>(rng.normal() * stddev);
}

} // namespace

// ---------------------------------------------------------------- Conv2d

Conv2d::Conv2d(int cin_, int cout_, int k_) : cin(cin_), cout(cout_), k(k_) {
    assert(cin > 0 && cout > 0 && (k == 1 || k == 3));
    w.assign(static_cast<size_t>(cout) * cin * k * k, 0.0f);
    gw.assign(w.size(), 0.0f);
}

void Conv2d::init(Rng& rng) { heNormal(w, cin * k * k, rng); }

size_t Conv2d::colSize(int B, int n) const {
    return k == 1 ? 0 : static_cast<size_t>(cin) * k * k * B * n * n;
}

void Conv2d::forward(const Backend& be, const float* x, int B, int n, float* col, float* y) const {
    const int M = B * n * n;
    // A 1x1 convolution needs no unfolding: x [cin][M] already is the column matrix.
    const float* columns = x;
    if (k != 1) {
        be.im2col(x, cin, B, n, k, col);
        columns = col;
    }
    // y [cout][M] = w [cout][cin*k*k] . columns [cin*k*k][M]
    be.gemm(false, false, cout, M, cin * k * k, 1.0f, w.data(), columns, 0.0f, y);
}

void Conv2d::backward(const Backend& be, const float* x, const float* col, const float* dy,
                      int B, int n, float* dcol, float* dx) {
    const int M = B * n * n;
    const int K = cin * k * k;
    const float* columns = k == 1 ? x : col;
    // y = w . columns, so dL/dw = dy . columns^T (summed over all M pixels),
    // accumulated into gw (beta = 1).
    be.gemm(false, true, cout, K, M, 1.0f, dy, columns, 1.0f, gw.data());
    if (dx == nullptr) return;
    if (k == 1) {
        // dL/dx = w^T . dy directly: the columns ARE x.
        be.gemm(true, false, cin, M, cout, 1.0f, w.data(), dy, 0.0f, dx);
    } else {
        // dL/dcolumns = w^T . dy, then fold the columns back onto the pixels
        // they were copied from; col2im sums overlapping patches.
        be.gemm(true, false, K, M, cout, 1.0f, w.data(), dy, 0.0f, dcol);
        be.col2im(dcol, cin, B, n, k, dx);
    }
}

// ---------------------------------------------------------------- BatchNorm2d

BatchNorm2d::BatchNorm2d(int c_)
    : c(c_), gamma(c_, 1.0f), beta(c_, 0.0f), ggamma(c_, 0.0f), gbeta(c_, 0.0f),
      runMean(c_, 0.0f), runVar(c_, 1.0f) {
    assert(c > 0);
}

void BatchNorm2d::forwardInfer(const float* x, int M, float* y) const {
    for (int ch = 0; ch < c; ++ch) {
        // Fold the fixed statistics into one scale and shift per channel.
        const float scale = gamma[ch] / std::sqrt(runVar[ch] + kEps);
        const float shift = beta[ch] - scale * runMean[ch];
        const float* xc = x + static_cast<size_t>(ch) * M;
        float* yc = y + static_cast<size_t>(ch) * M;
        for (int i = 0; i < M; ++i) yc[i] = scale * xc[i] + shift;
    }
}

void BatchNorm2d::forwardTrain(const float* x, int M, float* y, float* xhat, float* invStd,
                               bool updateRunning) {
    assert(M > 1);
    for (int ch = 0; ch < c; ++ch) {
        const float* xc = x + static_cast<size_t>(ch) * M;
        // Two passes in double: mean first, then the variance around it
        // (the one-pass E[x^2] - E[x]^2 loses precision when the mean is large).
        double sum = 0.0;
        for (int i = 0; i < M; ++i) sum += xc[i];
        const double mean = sum / M;
        double sq = 0.0;
        for (int i = 0; i < M; ++i) sq += (xc[i] - mean) * (xc[i] - mean);
        const double var = sq / M;                                  // biased: used to normalize
        const float inv = static_cast<float>(1.0 / std::sqrt(var + kEps));
        invStd[ch] = inv;

        float* xh = xhat + static_cast<size_t>(ch) * M;
        float* yc = y + static_cast<size_t>(ch) * M;
        const float m = static_cast<float>(mean);
        for (int i = 0; i < M; ++i) {
            const float h = (xc[i] - m) * inv;                      // read x before y (they may alias)
            xh[i] = h;
            yc[i] = gamma[ch] * h + beta[ch];
        }
        if (updateRunning) {
            // Inference sees one position at a time, so it needs population
            // estimates: unbiased variance, exponential moving averages.
            const double unbiased = var * M / (M - 1);
            runMean[ch] = (1.0f - kMomentum) * runMean[ch] + kMomentum * static_cast<float>(mean);
            runVar[ch] = (1.0f - kMomentum) * runVar[ch] + kMomentum * static_cast<float>(unbiased);
        }
    }
}

void BatchNorm2d::backward(const float* dy, const float* xhat, const float* invStd, int M,
                           float* dx) {
    for (int ch = 0; ch < c; ++ch) {
        const float* dyc = dy + static_cast<size_t>(ch) * M;
        const float* xh = xhat + static_cast<size_t>(ch) * M;
        double sumDy = 0.0, sumDyXhat = 0.0;
        for (int i = 0; i < M; ++i) {
            sumDy += dyc[i];
            sumDyXhat += static_cast<double>(dyc[i]) * xh[i];
        }
        ggamma[ch] += static_cast<float>(sumDyXhat);                // y = gamma * xhat + beta
        gbeta[ch] += static_cast<float>(sumDy);
        // Every x in the channel moved the batch mean and variance, so each
        // dx collects a share of every dy:
        //   dx = gamma * invStd / M * (M * dy - sum(dy) - xhat * sum(dy * xhat))
        const float scale = gamma[ch] * invStd[ch] / static_cast<float>(M);
        const float fM = static_cast<float>(M);
        const float sDy = static_cast<float>(sumDy), sDyXhat = static_cast<float>(sumDyXhat);
        float* dxc = dx + static_cast<size_t>(ch) * M;
        for (int i = 0; i < M; ++i) dxc[i] = scale * (fM * dyc[i] - sDy - xh[i] * sDyXhat);
    }
}

// ---------------------------------------------------------------- Linear

Linear::Linear(int in_, int out_)
    : in(in_), out(out_), w(static_cast<size_t>(in_) * out_, 0.0f), b(out_, 0.0f),
      gw(w.size(), 0.0f), gb(out_, 0.0f) {
    assert(in > 0 && out > 0);
}

void Linear::init(Rng& rng) {
    heNormal(w, in, rng);
    std::fill(b.begin(), b.end(), 0.0f);
}

void Linear::forward(const Backend& be, const float* x, int B, float* y) const {
    // y [B][out] = x [B][in] . w^T (w is stored [out][in]), then + b per row.
    be.gemm(false, true, B, out, in, 1.0f, x, w.data(), 0.0f, y);
    for (int s = 0; s < B; ++s) {
        float* row = y + static_cast<size_t>(s) * out;
        for (int o = 0; o < out; ++o) row[o] += b[o];
    }
}

void Linear::backward(const Backend& be, const float* x, const float* dy, int B, float* dx) {
    // gw [out][in] += dy^T [out][B] . x [B][in]
    be.gemm(true, false, out, in, B, 1.0f, dy, x, 1.0f, gw.data());
    for (int s = 0; s < B; ++s) {
        const float* row = dy + static_cast<size_t>(s) * out;
        for (int o = 0; o < out; ++o) gb[o] += row[o];
    }
    // dx [B][in] = dy [B][out] . w [out][in]
    if (dx != nullptr) be.gemm(false, false, B, in, out, 1.0f, dy, w.data(), 0.0f, dx);
}

// ---------------------------------------------------------------- ReLU

void reluForward(float* x, size_t count) {
    for (size_t i = 0; i < count; ++i) x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

void reluBackward(const float* y, float* dy, size_t count) {
    // relu'(x) is 1 where the output is positive, 0 elsewhere; the output
    // tells us which, so the pre-activation need not be kept.
    for (size_t i = 0; i < count; ++i) dy[i] = y[i] > 0.0f ? dy[i] : 0.0f;
}

} // namespace azgo
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_layers`
Expected: `100% tests passed` (the executable prints `test_layers: all tests passed`).

- [ ] **Step 6: Commit**

```bash
git add include/azgo/layers.hpp src/layers.cpp tests/test_layers.cpp
git commit -F - <<'EOF'
Add conv, batch norm, linear and ReLU layers with backprop

These are the pieces the ResNet is assembled from, each with a
hand-written backward pass so the chain rule is visible rather than
hidden in a framework. Convolution is im2col + gemm (1x1 skips the
unfolding), batch norm uses batch statistics in training and running
averages in inference, and every gradient is verified against central
finite differences so later network-level bugs cannot hide in a layer.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

### Task 7: Network (ResNet, Workspace, loss, trainStep)

**Files:**
- Create: `include/azgo/network.hpp`, `src/network.cpp`
- Test: `tests/test_network.cpp`

**Interfaces:**
- Consumes: `Conv2d`, `BatchNorm2d`, `Linear`, `reluForward`, `reluBackward` (Task 6). `Backend`, `Backend::im2col`, `NaiveBackend` (Task 5). `Rng(uint64_t)`, `Rng::uniform()`, `Rng::below(int)` (Task 1). `kMinBoardSize`, `kMaxBoardSize` (Task 2, `azgo/board.hpp`).
- Produces: `azgo::NetConfig` (with `planes()`, `area()`, `actions()`, `operator==`/`!=` over all 5 fields), `ParamView`, `BufferView`, `LossStats`, `Workspace(const NetConfig&, int maxBatch, const Backend&)`, `Network(const NetConfig&, uint64_t seed)` with `config`, `params`, `buffers`, `zeroGrad`, `infer`, `trainStep`, `computeLoss`. These match contract.md exactly. Parameter names in `params()` order: `stem.conv.w`, `stem.bn.gamma`, `stem.bn.beta`; for each block i, `block{i}.conv1.w`, `block{i}.bn1.gamma`, `block{i}.bn1.beta`, `block{i}.conv2.w`, `block{i}.bn2.gamma`, `block{i}.bn2.beta`; then `policy.conv.w`, `policy.bn.gamma`, `policy.bn.beta`, `policy.fc.w`, `policy.fc.b`, `value.conv.w`, `value.bn.gamma`, `value.bn.beta`, `value.fc1.w`, `value.fc1.b`, `value.fc2.w`, `value.fc2.b`. `decay` is true exactly for names ending in `.w`. Buffers are `<bn prefix>.runMean` and `<bn prefix>.runVar` in the same BN order. Network is copyable (Task 8's tests and Task 21's trainer copy it), and `infer` is safe to call concurrently from several threads, each with its own `Workspace`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_network.cpp`:

```cpp
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/network.hpp"
#include "azgo/rng.hpp"

using namespace azgo;

static bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

namespace {

NetConfig makeConfig(int n, int history, int blocks, int filters, int valueHidden) {
    NetConfig cfg;
    cfg.boardSize = n;
    cfg.history = history;
    cfg.blocks = blocks;
    cfg.filters = filters;
    cfg.valueHidden = valueHidden;
    return cfg;
}

// A random batch: input planes in {0, 1}, a normalized random policy target
// (or a one-hot one), and outcomes z in {-1, +1}.
struct Batch {
    std::vector<float> input, pi, z;
};

Batch randomBatch(const NetConfig& cfg, int B, Rng& rng, bool oneHotPi = false) {
    Batch batch;
    batch.input.resize(static_cast<size_t>(B) * cfg.planes() * cfg.area());
    for (float& v : batch.input) v = rng.uniform() < 0.5 ? 0.0f : 1.0f;
    batch.pi.assign(static_cast<size_t>(B) * cfg.actions(), 0.0f);
    for (int b = 0; b < B; ++b) {
        float* row = batch.pi.data() + static_cast<size_t>(b) * cfg.actions();
        if (oneHotPi) {
            row[rng.below(cfg.actions())] = 1.0f;
            continue;
        }
        double sum = 0.0;
        for (int a = 0; a < cfg.actions(); ++a) sum += row[a] = static_cast<float>(rng.uniform());
        for (int a = 0; a < cfg.actions(); ++a) row[a] = static_cast<float>(row[a] / sum);
    }
    batch.z.resize(B);
    for (float& v : batch.z) v = rng.uniform() < 0.5 ? -1.0f : 1.0f;
    return batch;
}

double totalLoss(const LossStats& s) { return static_cast<double>(s.policyLoss) + s.valueLoss; }

enum class Probe { Match, Kink, Mismatch };

// Probes dL/dvalue with steps of +-h. Match: the central difference agrees
// with `analytic` (tolerance 2e-2 relative + 2e-3 absolute). Kink: it does
// not, but the forward and backward one-sided differences disagree with
// each other, the signature of a ReLU switching inside [v - h, v + h] (the
// central difference then averages two different slopes). Mismatch: a
// smooth region where the analytic gradient is simply wrong.
template <typename LossFn>
Probe probeGradient(float* value, double analytic, float h, LossFn loss) {
    auto tolerance = [](double a, double b) { return 2e-2 * std::max(std::fabs(a), std::fabs(b)) + 2e-3; };
    const float saved = *value;
    const float up = saved + h, down = saved - h;
    const double lossMid = loss();
    *value = up;
    const double lossUp = loss();
    *value = down;
    const double lossDown = loss();
    *value = saved;
    const double central = (lossUp - lossDown) / (static_cast<double>(up) - down);
    if (std::fabs(analytic - central) <= tolerance(analytic, central)) return Probe::Match;
    const double forward = (lossUp - lossMid) / (static_cast<double>(up) - saved);
    const double backward = (lossMid - lossDown) / (static_cast<double>(saved) - down);
    if (std::fabs(forward - backward) > tolerance(forward, backward)) return Probe::Kink;
    return Probe::Mismatch;
}

bool allFinite(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

} // namespace

static void test_whole_network_gradient_check() {
    const NetConfig cfg = makeConfig(5, 1, 1, 4, 8);
    const int B = 3;
    NaiveBackend be;
    Network net(cfg, 42);
    Workspace ws(cfg, B, be);
    Rng rng(1);
    // Random BN affine parameters: exercises gamma/beta != 1/0 and moves each
    // ReLU's kink away from its channel mean, where values are densest.
    for (ParamView& p : net.params()) {
        const bool isGamma = p.name.find(".gamma") != std::string::npos;
        const bool isBeta = p.name.find(".beta") != std::string::npos;
        for (size_t i = 0; i < p.size; ++i) {
            if (isGamma) p.value[i] = static_cast<float>(0.5 + rng.uniform());
            if (isBeta) p.value[i] = static_cast<float>(rng.uniform() - 0.5);
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
        const std::vector<float> analytic(p.grad, p.grad + p.size);   // snapshot after trainStep
        std::vector<size_t> indices;
        if (p.size <= 64) {
            for (size_t i = 0; i < p.size; ++i) indices.push_back(i);   // small tensors: every entry
        } else {
            for (int k = 0; k < 24; ++k) indices.push_back(static_cast<size_t>(rng.below(static_cast<int>(p.size))));
        }
        for (size_t i : indices) {
            ++checked;
            // Unlike a single layer, the whole network has ~1000 ReLUs right
            // behind batch norms, and one weight moves every unit of a
            // channel, so a probe of 1e-2 nearly always crosses some kink.
            // Smaller steps (2e-3, then 5e-4) still resolve an O(1) float
            // loss; a probe is only excused when it visibly straddles a kink.
            Probe first = probeGradient(p.value + i, analytic[i], 2e-3f, loss);
            if (first == Probe::Match) continue;
            Probe second = probeGradient(p.value + i, analytic[i], 5e-4f, loss);
            if (second == Probe::Match) continue;
            if (second == Probe::Kink) {
                ++kinks;
                continue;
            }
            std::printf("%s[%zu]: analytic %.6f does not match finite differences\n",
                        p.name.c_str(), i, analytic[i]);
            assert(false);
        }
    }
    assert(checked > 200);
    assert(kinks * 4 <= checked);   // kinks excuse only a minority of probes
}

static void test_param_names_order_and_decay() {
    const NetConfig cfg = makeConfig(5, 1, 1, 4, 8);
    Network a(cfg, 1), b(cfg, 2);
    std::vector<ParamView> pa = a.params(), pb = b.params();
    const std::vector<std::string> expected = {
        "stem.conv.w", "stem.bn.gamma", "stem.bn.beta",
        "block0.conv1.w", "block0.bn1.gamma", "block0.bn1.beta",
        "block0.conv2.w", "block0.bn2.gamma", "block0.bn2.beta",
        "policy.conv.w", "policy.bn.gamma", "policy.bn.beta", "policy.fc.w", "policy.fc.b",
        "value.conv.w", "value.bn.gamma", "value.bn.beta",
        "value.fc1.w", "value.fc1.b", "value.fc2.w", "value.fc2.b"};
    assert(pa.size() == expected.size() && pb.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(pa[i].name == expected[i] && pb[i].name == expected[i]);
        const bool isWeight = pa[i].name.size() > 2 && pa[i].name.compare(pa[i].name.size() - 2, 2, ".w") == 0;
        assert(pa[i].decay == isWeight);
    }
    const int A = cfg.area();
    assert(pa[0].size == static_cast<size_t>(4 * cfg.planes() * 9));      // stem.conv.w
    assert(pa[3].size == static_cast<size_t>(4 * 4 * 9));                // block0.conv1.w
    assert(pa[9].size == static_cast<size_t>(2 * 4));                    // policy.conv.w
    assert(pa[12].size == static_cast<size_t>(cfg.actions() * 2 * A));   // policy.fc.w
    assert(pa[17].size == static_cast<size_t>(8 * A));                   // value.fc1.w
    assert(pa[20].size == 1);                                            // value.fc2.b

    const NetConfig deep = makeConfig(5, 1, 3, 4, 8);
    Network d(deep, 1);
    std::set<std::string> names;
    for (const ParamView& p : d.params()) names.insert(p.name);
    assert(names.size() == d.params().size());                           // unique
    assert(names.count("block2.bn1.gamma") == 1 && names.count("block2.conv2.w") == 1);
}

static void test_buffers() {
    const NetConfig cfg = makeConfig(5, 1, 2, 4, 8);
    Network net(cfg, 1);
    std::vector<BufferView> bufs = net.buffers();
    const int bnCount = 1 + 2 * cfg.blocks + 2;
    assert(bufs.size() == static_cast<size_t>(2 * bnCount));
    assert(bufs[0].name == "stem.bn.runMean" && bufs[1].name == "stem.bn.runVar");
    assert(bufs[2].name == "block0.bn1.runMean" && bufs[5].name == "block0.bn2.runVar");
    assert(bufs[bufs.size() - 2].name == "value.bn.runMean" && bufs.back().name == "value.bn.runVar");
    for (const BufferView& b : bufs) {
        const bool isVar = b.name.find("runVar") != std::string::npos;
        for (size_t i = 0; i < b.size; ++i) assert(b.value[i] == (isVar ? 1.0f : 0.0f));
    }
}

static void test_infer_shapes_every_board_size() {
    NaiveBackend be;
    Rng rng(3);
    for (int n = 5; n <= 19; ++n) {
        const NetConfig cfg = makeConfig(n, 2, 1, 2, 4);
        const int B = 2;
        Network net(cfg, static_cast<uint64_t>(n));
        Workspace ws(cfg, B, be);
        Batch batch = randomBatch(cfg, B, rng);
        std::vector<float> logits(static_cast<size_t>(B) * cfg.actions(), std::numeric_limits<float>::quiet_NaN());
        std::vector<float> values(B, std::numeric_limits<float>::quiet_NaN());
        net.infer(ws, batch.input.data(), B, logits.data(), values.data());
        assert(allFinite(logits) && allFinite(values));
        for (float v : values) assert(v >= -1.0f && v <= 1.0f);
    }
}

static void test_zero_blocks() {
    const NetConfig cfg = makeConfig(5, 1, 0, 4, 8);
    NaiveBackend be;
    Network net(cfg, 5);
    assert(net.params().size() == 15);          // stem 3 + policy 5 + value 7
    Workspace ws(cfg, 4, be);
    Rng rng(4);
    Batch batch = randomBatch(cfg, 4, rng);
    std::vector<float> logits(4 * cfg.actions()), values(4);
    net.infer(ws, batch.input.data(), 4, logits.data(), values.data());
    assert(allFinite(logits) && allFinite(values));
    LossStats s = net.trainStep(ws, batch.input.data(), batch.pi.data(), batch.z.data(), 4);
    assert(std::isfinite(s.policyLoss) && std::isfinite(s.valueLoss));
}

static void test_seed_determinism() {
    const NetConfig cfg = makeConfig(5, 2, 1, 4, 8);
    NaiveBackend be;
    Network a(cfg, 7), b(cfg, 7), c(cfg, 8);
    Workspace ws(cfg, 2, be);
    Rng rng(5);
    Batch batch = randomBatch(cfg, 2, rng);
    std::vector<float> la(2 * cfg.actions()), lb(la.size()), lc(la.size()), va(2), vb(2), vc(2);
    a.infer(ws, batch.input.data(), 2, la.data(), va.data());
    b.infer(ws, batch.input.data(), 2, lb.data(), vb.data());
    c.infer(ws, batch.input.data(), 2, lc.data(), vc.data());
    assert(std::memcmp(la.data(), lb.data(), la.size() * sizeof(float)) == 0);
    assert(std::memcmp(va.data(), vb.data(), va.size() * sizeof(float)) == 0);
    assert(la != lc);
}

static void test_running_stats_only_change_in_train_step() {
    const NetConfig cfg = makeConfig(5, 1, 1, 4, 8);
    NaiveBackend be;
    Network net(cfg, 9);
    Workspace ws(cfg, 4, be);
    Rng rng(6);
    Batch batch = randomBatch(cfg, 4, rng);
    auto snapshot = [&net] {
        std::vector<float> all;
        for (const BufferView& b : net.buffers()) all.insert(all.end(), b.value, b.value + b.size);
        return all;
    };
    const std::vector<float> before = snapshot();
    net.computeLoss(ws, batch.input.data(), batch.pi.data(), batch.z.data(), 4);
    assert(snapshot() == before);
    net.trainStep(ws, batch.input.data(), batch.pi.data(), batch.z.data(), 4);
    const std::vector<float> after = snapshot();
    size_t changed = 0;
    for (size_t i = 0; i < before.size(); ++i) changed += after[i] != before[i];
    assert(changed > before.size() / 2);
}

static void test_sgd_reduces_loss_on_fixed_batch() {
    // Plain SGD through params() on one batch: the network can memorize 8
    // positions, so the loss must fall well below its starting value.
    const NetConfig cfg = makeConfig(5, 1, 1, 8, 16);
    const int B = 8;
    NaiveBackend be;
    Network net(cfg, 11);
    Workspace ws(cfg, B, be);
    Rng rng(7);
    Batch batch = randomBatch(cfg, B, rng, /*oneHotPi=*/true);
    double first = -1.0;
    for (int step = 0; step < 60; ++step) {
        LossStats s = net.trainStep(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B);
        if (step == 0) first = totalLoss(s);
        for (ParamView& p : net.params())
            for (size_t i = 0; i < p.size; ++i) p.value[i] -= 0.05f * p.grad[i];
    }
    const double last = totalLoss(net.computeLoss(ws, batch.input.data(), batch.pi.data(), batch.z.data(), B));
    std::printf("sgd: loss %.4f -> %.4f\n", first, last);
    assert(last <= 0.5 * first);
}

static void test_inference_is_per_sample() {
    // Inference BN uses fixed running statistics, so a sample's output must
    // not depend on the rest of the batch.
    const NetConfig cfg = makeConfig(5, 2, 1, 4, 8);
    NaiveBackend be;
    Network net(cfg, 13);
    Workspace ws(cfg, 4, be);
    Rng rng(8);
    for (int i = 0; i < 3; ++i) {                // non-trivial running statistics
        Batch warm = randomBatch(cfg, 4, rng);
        net.trainStep(ws, warm.input.data(), warm.pi.data(), warm.z.data(), 4);
    }
    Batch batch = randomBatch(cfg, 4, rng);
    const int actions = cfg.actions();
    const size_t sampleInput = static_cast<size_t>(cfg.planes()) * cfg.area();
    std::vector<float> logits4(4 * actions), values4(4);
    net.infer(ws, batch.input.data(), 4, logits4.data(), values4.data());
    for (int b = 0; b < 4; ++b) {
        std::vector<float> logits1(actions), values1(1);
        net.infer(ws, batch.input.data() + b * sampleInput, 1, logits1.data(), values1.data());
        for (int a = 0; a < actions; ++a) {
            const float x = logits4[b * actions + a];
            assert(near(logits1[a], x, 1e-5 * (1 + std::fabs(x))));
        }
        assert(near(values1[0], values4[b], 1e-5));
    }
}

int main() {
    test_whole_network_gradient_check();
    test_param_names_order_and_decay();
    test_buffers();
    test_infer_shapes_every_board_size();
    test_zero_blocks();
    test_seed_determinism();
    test_running_stats_only_change_in_train_step();
    test_sgd_reduces_loss_on_fixed_batch();
    test_inference_is_per_sample();
    std::printf("test_network: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target test_network`
Expected: FAIL at compile time with `fatal error: 'azgo/network.hpp' file not found`.

- [ ] **Step 3: Write the header**

Create `include/azgo/network.hpp`. It is contract.md's header plus the teaching block, with the two "implementation-defined" comments replaced by the private members below:

```cpp
// The two-headed residual network of AlphaGo Zero. A shared convolutional
// trunk (a stem plus residual blocks, each adding its output to its input so
// gradients have a short path back through deep stacks) reads the input
// planes; a policy head turns the trunk's features into one logit per action
// and a value head into one number in [-1, 1]: the expected result for the
// side to move. Parameters (Network) are shared read-only by all threads;
// activations (Workspace) are per thread, so many searches can run at once.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/layers.hpp"

namespace azgo {

struct NetConfig {
    int boardSize = 9;
    int history = 8;       // H
    int blocks = 6;        // residual blocks, >= 0
    int filters = 64;
    int valueHidden = 64;
    int planes() const { return 2 * history + 2; }
    int area() const { return boardSize * boardSize; }
    int actions() const { return area() + 1; }
    bool operator==(const NetConfig& o) const;
    bool operator!=(const NetConfig& o) const { return !(*this == o); }
};

struct ParamView  { std::string name; float* value; float* grad; size_t size; bool decay; };
struct BufferView { std::string name; float* value; size_t size; };

struct LossStats { float policyLoss = 0; float valueLoss = 0; };  // batch means

// Per-thread activation and scratch storage for up to maxBatch samples.
class Workspace {
public:
    Workspace(const NetConfig& cfg, int maxBatch, const Backend& backend);
    int maxBatch() const { return maxBatch_; }
    const Backend& backend() const { return *backend_; }
private:
    friend class Network;
    NetConfig cfg_;
    int maxBatch_;
    const Backend* backend_;
    // Activation buffers (Task 7). All are channel-major [C][B][area] unless
    // noted, sized for maxBatch, allocated once here and never resized.
    struct BlockBuffers {
        std::vector<float> xhat1, invStd1, act1;   // bn1 cache; act1 = relu(bn1) = conv2 input
        std::vector<float> xhat2, invStd2, out;    // bn2 cache; out = relu(bn2 + skip)
    };
    std::vector<float> input_;                    // [planes][B][area] (transposed network input)
    std::vector<float> col_, dcol_;               // im2col scratch shared by every conv
    std::vector<float> convOut_;                  // conv output before its BN
    std::vector<float> stemXhat_, stemInvStd_, stemOut_;
    std::vector<BlockBuffers> blocks_;
    std::vector<float> policyXhat_, policyInvStd_, policyAct_;   // policyAct_ [2][B][area]
    std::vector<float> policyFlat_;               // [B][2 * area]
    std::vector<float> logits_;                   // [B][actions]
    std::vector<float> valueXhat_, valueInvStd_, valueAct_;      // valueAct_ [1][B][area] = [B][area]
    std::vector<float> valueHidden_;              // [B][valueHidden]
    std::vector<float> values_;                   // [B] (after tanh)
    // Backward scratch.
    std::vector<float> dTrunk_, dTmp_, dTmp2_;    // [filters][B][area]
    std::vector<float> dLogits_, dPolicyFlat_, dPolicyAct_;
    std::vector<float> dValues_, dValueHidden_, dValueAct_;
};

class Network {
public:
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
    // Layers (Task 7).
    struct ResidualBlock {
        Conv2d conv1;
        BatchNorm2d bn1;
        Conv2d conv2;
        BatchNorm2d bn2;
    };
    Conv2d stemConv_;
    BatchNorm2d stemBn_;
    std::vector<ResidualBlock> blocks_;
    Conv2d policyConv_;
    BatchNorm2d policyBn_;
    Linear policyFc_;
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

- [ ] **Step 4: Write the implementation**

Create `src/network.cpp`:

```cpp
#include "azgo/network.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include "azgo/board.hpp"
#include "azgo/rng.hpp"

namespace azgo {

bool NetConfig::operator==(const NetConfig& o) const {
    return boardSize == o.boardSize && history == o.history && blocks == o.blocks &&
           filters == o.filters && valueHidden == o.valueHidden;
}

// ---------------------------------------------------------------- Workspace

Workspace::Workspace(const NetConfig& cfg, int maxBatch, const Backend& backend)
    : cfg_(cfg), maxBatch_(maxBatch), backend_(&backend) {
    assert(maxBatch >= 1);
    const size_t M = static_cast<size_t>(maxBatch) * cfg.area();    // pixels per channel
    const size_t F = static_cast<size_t>(cfg.filters);
    const size_t trunk = F * M;
    const size_t actions = static_cast<size_t>(maxBatch) * cfg.actions();
    const size_t hidden = static_cast<size_t>(maxBatch) * cfg.valueHidden;

    input_.resize(static_cast<size_t>(cfg.planes()) * M);
    // One im2col buffer serves every 3x3 conv (stem: planes inputs, blocks:
    // F inputs); backward rebuilds a conv's columns right before using them.
    col_.resize(static_cast<size_t>(std::max(cfg.planes(), cfg.filters)) * 9 * M);
    dcol_.resize(col_.size());
    convOut_.resize(std::max<size_t>(F, 2) * M);   // the policy conv has 2 outputs even if F == 1

    stemXhat_.resize(trunk);
    stemInvStd_.resize(F);
    stemOut_.resize(trunk);
    blocks_.resize(cfg.blocks);
    for (BlockBuffers& b : blocks_) {
        b.xhat1.resize(trunk);
        b.invStd1.resize(F);
        b.act1.resize(trunk);
        b.xhat2.resize(trunk);
        b.invStd2.resize(F);
        b.out.resize(trunk);
    }
    policyXhat_.resize(2 * M);
    policyInvStd_.resize(2);
    policyAct_.resize(2 * M);
    policyFlat_.resize(2 * M);
    logits_.resize(actions);
    valueXhat_.resize(M);
    valueInvStd_.resize(1);
    valueAct_.resize(M);
    valueHidden_.resize(hidden);
    values_.resize(maxBatch);

    dTrunk_.resize(trunk);
    dTmp_.resize(trunk);
    dTmp2_.resize(trunk);
    dLogits_.resize(actions);
    dPolicyFlat_.resize(2 * M);
    dPolicyAct_.resize(2 * M);
    dValues_.resize(maxBatch);
    dValueHidden_.resize(hidden);
    dValueAct_.resize(M);
}

// ---------------------------------------------------------------- Network

Network::Network(const NetConfig& cfg, uint64_t seed)
    : cfg_(cfg),
      stemConv_(cfg.planes(), cfg.filters, 3),
      stemBn_(cfg.filters),
      policyConv_(cfg.filters, 2, 1),
      policyBn_(2),
      policyFc_(2 * cfg.area(), cfg.actions()),
      valueConv_(cfg.filters, 1, 1),
      valueBn_(1),
      valueFc1_(cfg.area(), cfg.valueHidden),
      valueFc2_(cfg.valueHidden, 1) {
    assert(cfg.boardSize >= kMinBoardSize && cfg.boardSize <= kMaxBoardSize);
    assert(cfg.history >= 1 && cfg.blocks >= 0 && cfg.filters >= 1 && cfg.valueHidden >= 1);
    const int F = cfg.filters;
    for (int i = 0; i < cfg.blocks; ++i) {
        blocks_.push_back({Conv2d(F, F, 3), BatchNorm2d(F), Conv2d(F, F, 3), BatchNorm2d(F)});
    }
    // One generator, layers in params() order, so a seed fixes every weight.
    Rng rng(seed);
    stemConv_.init(rng);
    for (ResidualBlock& b : blocks_) {
        b.conv1.init(rng);
        b.conv2.init(rng);
    }
    policyConv_.init(rng);
    policyFc_.init(rng);
    valueConv_.init(rng);
    valueFc1_.init(rng);
    valueFc2_.init(rng);
}

std::vector<ParamView> Network::params() {
    std::vector<ParamView> out;
    auto conv = [&out](const std::string& prefix, Conv2d& c) {
        out.push_back({prefix + ".w", c.w.data(), c.gw.data(), c.w.size(), true});
    };
    auto bn = [&out](const std::string& prefix, BatchNorm2d& b) {
        out.push_back({prefix + ".gamma", b.gamma.data(), b.ggamma.data(), b.gamma.size(), false});
        out.push_back({prefix + ".beta", b.beta.data(), b.gbeta.data(), b.beta.size(), false});
    };
    auto linear = [&out](const std::string& prefix, Linear& l) {
        out.push_back({prefix + ".w", l.w.data(), l.gw.data(), l.w.size(), true});
        out.push_back({prefix + ".b", l.b.data(), l.gb.data(), l.b.size(), false});
    };
    conv("stem.conv", stemConv_);
    bn("stem.bn", stemBn_);
    for (size_t i = 0; i < blocks_.size(); ++i) {
        const std::string prefix = "block" + std::to_string(i);
        conv(prefix + ".conv1", blocks_[i].conv1);
        bn(prefix + ".bn1", blocks_[i].bn1);
        conv(prefix + ".conv2", blocks_[i].conv2);
        bn(prefix + ".bn2", blocks_[i].bn2);
    }
    conv("policy.conv", policyConv_);
    bn("policy.bn", policyBn_);
    linear("policy.fc", policyFc_);
    conv("value.conv", valueConv_);
    bn("value.bn", valueBn_);
    linear("value.fc1", valueFc1_);
    linear("value.fc2", valueFc2_);
    return out;
}

std::vector<BufferView> Network::buffers() {
    std::vector<BufferView> out;
    auto bn = [&out](const std::string& prefix, BatchNorm2d& b) {
        out.push_back({prefix + ".runMean", b.runMean.data(), b.runMean.size()});
        out.push_back({prefix + ".runVar", b.runVar.data(), b.runVar.size()});
    };
    bn("stem.bn", stemBn_);
    for (size_t i = 0; i < blocks_.size(); ++i) {
        const std::string prefix = "block" + std::to_string(i);
        bn(prefix + ".bn1", blocks_[i].bn1);
        bn(prefix + ".bn2", blocks_[i].bn2);
    }
    bn("policy.bn", policyBn_);
    bn("value.bn", valueBn_);
    return out;
}

void Network::zeroGrad() {
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
    assert(B >= 1 && B <= ws.maxBatch_);
    assert(ws.cfg_ == cfg_);
    const Backend& be = *ws.backend_;
    const int n = cfg_.boardSize, A = cfg_.area(), P = cfg_.planes();
    const int M = B * A;
    const size_t trunk = static_cast<size_t>(cfg_.filters) * M;
    float* col = ws.col_.data();
    float* convOut = ws.convOut_.data();

    // [B][planes][A] -> [planes][B][A]. Channel-major lets a conv treat the
    // whole batch as one wide matrix: one gemm per layer, not one per sample.
    for (int b = 0; b < B; ++b)
        for (int p = 0; p < P; ++p)
            std::copy(input + (static_cast<size_t>(b) * P + p) * A,
                      input + (static_cast<size_t>(b) * P + p + 1) * A,
                      ws.input_.data() + (static_cast<size_t>(p) * B + b) * A);

    auto norm = [&](BatchNorm2d& bn, const float* x, float* y, float* xhat, float* invStd) {
        if (mode == Mode::Infer) bn.forwardInfer(x, M, y);
        else bn.forwardTrain(x, M, y, xhat, invStd, mode == Mode::Train);
    };

    // Stem: conv3x3 -> BN -> ReLU.
    stemConv_.forward(be, ws.input_.data(), B, n, col, convOut);
    norm(stemBn_, convOut, ws.stemOut_.data(), ws.stemXhat_.data(), ws.stemInvStd_.data());
    reluForward(ws.stemOut_.data(), trunk);

    // Residual blocks: conv -> BN -> ReLU -> conv -> BN -> + input -> ReLU.
    const float* x = ws.stemOut_.data();
    for (size_t i = 0; i < blocks_.size(); ++i) {
        ResidualBlock& blk = blocks_[i];
        Workspace::BlockBuffers& buf = ws.blocks_[i];
        blk.conv1.forward(be, x, B, n, col, convOut);
        norm(blk.bn1, convOut, buf.act1.data(), buf.xhat1.data(), buf.invStd1.data());
        reluForward(buf.act1.data(), trunk);
        blk.conv2.forward(be, buf.act1.data(), B, n, col, convOut);
        norm(blk.bn2, convOut, buf.out.data(), buf.xhat2.data(), buf.invStd2.data());
        for (size_t j = 0; j < trunk; ++j) buf.out[j] += x[j];   // the skip connection
        reluForward(buf.out.data(), trunk);
        x = buf.out.data();
    }

    // Policy head: conv1x1 (F -> 2) -> BN -> ReLU -> per-sample flatten -> Linear.
    policyConv_.forward(be, x, B, n, col, convOut);
    norm(policyBn_, convOut, ws.policyAct_.data(), ws.policyXhat_.data(), ws.policyInvStd_.data());
    reluForward(ws.policyAct_.data(), 2 * static_cast<size_t>(M));
    // [2][B][A] -> [B][2A], feature index c * A + p.
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < 2; ++c)
            std::copy(ws.policyAct_.data() + (static_cast<size_t>(c) * B + b) * A,
                      ws.policyAct_.data() + (static_cast<size_t>(c) * B + b + 1) * A,
                      ws.policyFlat_.data() + static_cast<size_t>(b) * 2 * A + static_cast<size_t>(c) * A);
    policyFc_.forward(be, ws.policyFlat_.data(), B, ws.logits_.data());

    // Value head: conv1x1 (F -> 1) -> BN -> ReLU -> Linear(A -> V) -> ReLU -> Linear(V -> 1) -> tanh.
    valueConv_.forward(be, x, B, n, col, convOut);
    norm(valueBn_, convOut, ws.valueAct_.data(), ws.valueXhat_.data(), ws.valueInvStd_.data());
    reluForward(ws.valueAct_.data(), M);
    // With a single channel, [1][B][A] already is [B][A]: one row per sample.
    valueFc1_.forward(be, ws.valueAct_.data(), B, ws.valueHidden_.data());
    reluForward(ws.valueHidden_.data(), static_cast<size_t>(B) * cfg_.valueHidden);
    valueFc2_.forward(be, ws.valueHidden_.data(), B, ws.values_.data());
    for (int b = 0; b < B; ++b) ws.values_[b] = std::tanh(ws.values_[b]);
}

LossStats Network::loss(Workspace& ws, const float* pi, const float* z, int B,
                        bool withGradients) const {
    const int actions = cfg_.actions();
    double policySum = 0.0, valueSum = 0.0;
    for (int b = 0; b < B; ++b) {
        const float* logits = ws.logits_.data() + static_cast<size_t>(b) * actions;
        const float* target = pi + static_cast<size_t>(b) * actions;
        // log softmax(l)_a = l_a - logSumExp(l). Subtracting the largest
        // logit before exp keeps every exponent <= 0, so nothing overflows.
        const float maxLogit = *std::max_element(logits, logits + actions);
        double sumExp = 0.0;
        for (int a = 0; a < actions; ++a) sumExp += std::exp(static_cast<double>(logits[a]) - maxLogit);
        const double logZ = maxLogit + std::log(sumExp);
        for (int a = 0; a < actions; ++a) policySum -= target[a] * (logits[a] - logZ);

        const double v = ws.values_[b];
        valueSum += (z[b] - v) * (z[b] - v);

        if (withGradients) {
            // Cross-entropy through softmax: dL/dlogit = softmax - pi (pi sums to 1).
            float* d = ws.dLogits_.data() + static_cast<size_t>(b) * actions;
            for (int a = 0; a < actions; ++a) {
                d[a] = static_cast<float>((std::exp(logits[a] - logZ) - target[a]) / B);
            }
            // (z - v)^2 with v = tanh(u): dL/du = 2 (v - z) * (1 - v^2).
            ws.dValues_[b] = static_cast<float>(2.0 * (v - z[b]) * (1.0 - v * v) / B);
        }
    }
    LossStats stats;
    stats.policyLoss = static_cast<float>(policySum / B);
    stats.valueLoss = static_cast<float>(valueSum / B);
    return stats;
}

void Network::backward(Workspace& ws, int B) {
    const Backend& be = *ws.backend_;
    const int n = cfg_.boardSize, A = cfg_.area(), F = cfg_.filters;
    const int M = B * A;
    const size_t trunk = static_cast<size_t>(F) * M;
    float* col = ws.col_.data();
    float* dcol = ws.dcol_.data();
    float* dTrunk = ws.dTrunk_.data();
    float* dTmp = ws.dTmp_.data();
    float* dTmp2 = ws.dTmp2_.data();
    const float* trunkOut = blocks_.empty() ? ws.stemOut_.data() : ws.blocks_.back().out.data();

    // Value head, from the loss back to the trunk output. (1x1 convs ignore col.)
    valueFc2_.backward(be, ws.valueHidden_.data(), ws.dValues_.data(), B, ws.dValueHidden_.data());
    reluBackward(ws.valueHidden_.data(), ws.dValueHidden_.data(), static_cast<size_t>(B) * cfg_.valueHidden);
    valueFc1_.backward(be, ws.valueAct_.data(), ws.dValueHidden_.data(), B, ws.dValueAct_.data());
    reluBackward(ws.valueAct_.data(), ws.dValueAct_.data(), M);
    valueBn_.backward(ws.dValueAct_.data(), ws.valueXhat_.data(), ws.valueInvStd_.data(), M,
                      ws.dValueAct_.data());
    valueConv_.backward(be, trunkOut, col, ws.dValueAct_.data(), B, n, dcol, dTrunk);

    // Policy head.
    policyFc_.backward(be, ws.policyFlat_.data(), ws.dLogits_.data(), B, ws.dPolicyFlat_.data());
    for (int b = 0; b < B; ++b)                   // [B][2A] -> [2][B][A], undoing the flatten
        for (int c = 0; c < 2; ++c)
            std::copy(ws.dPolicyFlat_.data() + static_cast<size_t>(b) * 2 * A + static_cast<size_t>(c) * A,
                      ws.dPolicyFlat_.data() + static_cast<size_t>(b) * 2 * A + static_cast<size_t>(c + 1) * A,
                      ws.dPolicyAct_.data() + (static_cast<size_t>(c) * B + b) * A);
    reluBackward(ws.policyAct_.data(), ws.dPolicyAct_.data(), 2 * static_cast<size_t>(M));
    policyBn_.backward(ws.dPolicyAct_.data(), ws.policyXhat_.data(), ws.policyInvStd_.data(), M,
                       ws.dPolicyAct_.data());
    policyConv_.backward(be, trunkOut, col, ws.dPolicyAct_.data(), B, n, dcol, dTmp);
    for (size_t j = 0; j < trunk; ++j) dTrunk[j] += dTmp[j];   // the trunk feeds both heads

    // Residual blocks in reverse; dTrunk holds dL/d(block output). The
    // shared col buffer was overwritten by later convs, so each 3x3 conv's
    // columns are rebuilt from its saved input before its backward.
    for (size_t i = blocks_.size(); i-- > 0;) {
        ResidualBlock& blk = blocks_[i];
        Workspace::BlockBuffers& buf = ws.blocks_[i];
        const float* in = i == 0 ? ws.stemOut_.data() : ws.blocks_[i - 1].out.data();
        reluBackward(buf.out.data(), dTrunk, trunk);           // now dL/d(bn2 output + skip)
        blk.bn2.backward(dTrunk, buf.xhat2.data(), buf.invStd2.data(), M, dTmp);
        be.im2col(buf.act1.data(), F, B, n, 3, col);
        blk.conv2.backward(be, buf.act1.data(), col, dTmp, B, n, dcol, dTmp2);
        reluBackward(buf.act1.data(), dTmp2, trunk);
        blk.bn1.backward(dTmp2, buf.xhat1.data(), buf.invStd1.data(), M, dTmp2);
        be.im2col(in, F, B, n, 3, col);
        blk.conv1.backward(be, in, col, dTmp2, B, n, dcol, dTmp);
        // The skip path passes dTrunk to the block input unchanged; the conv
        // path adds its own contribution.
        for (size_t j = 0; j < trunk; ++j) dTrunk[j] += dTmp[j];
    }

    // Stem. No gradient is needed for the input planes themselves.
    reluBackward(ws.stemOut_.data(), dTrunk, trunk);
    stemBn_.backward(dTrunk, ws.stemXhat_.data(), ws.stemInvStd_.data(), M, dTrunk);
    be.im2col(ws.input_.data(), cfg_.planes(), B, n, 3, col);
    stemConv_.backward(be, ws.input_.data(), col, dTrunk, B, n, dcol, nullptr);
}

} // namespace azgo
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -V -R test_network`
Expected: a line `sgd: loss <start> -> <end>` with `<end>` well under half of `<start>` (with the reference Rng it was `5.2080 -> 0.0155`), then `test_network: all tests passed`, and `100% tests passed`.

- [ ] **Step 6: Commit**

```bash
git add include/azgo/network.hpp src/network.cpp tests/test_network.cpp
git commit -F - <<'EOF'
Add the two-headed ResNet with its training step

This is the AlphaGo Zero network: a conv stem, residual blocks, and
policy and value heads, with the loss and a full backward pass built from
the layers. Parameters live in Network, shared read-only by worker
threads, while each thread owns a Workspace allocated once for its
largest batch, so self-play never allocates in the hot path. Parameters
and BN running statistics are exposed as named views in a fixed order,
which the optimizer and the checkpoint format both build on. A
whole-network finite-difference check guards the backward pass.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

### Task 8: Optimizer (SGD with momentum, learning-rate schedule)

**Files:**
- Create: `include/azgo/optimizer.hpp`, `src/optimizer.cpp`
- Test: `tests/test_optimizer.cpp`

**Interfaces:**
- Consumes: `Network(const NetConfig&, uint64_t)`, `Network::params()`, `Network::zeroGrad()`, `ParamView` (`name`, `value`, `grad`, `size`, `decay`), `NetConfig` (Task 7). Also relies on `Network` being copyable.
- Produces: `azgo::SgdMomentum(Network&, float momentum, float weightDecay)` with `step(Network&, float lr)` and `velocity()` (one vector per param, in `params()` order), and `azgo::LrSchedule` with `static parse(const std::string&)`, `at(int)`, `toString()`. These match contract.md exactly. `parse` error messages start with `learning-rate schedule '<text>': `. Task 16 wraps them into `ConfigError`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_optimizer.cpp`:

```cpp
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/network.hpp"
#include "azgo/optimizer.hpp"

using namespace azgo;

static bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

namespace {

NetConfig tinyConfig() {
    NetConfig cfg;
    cfg.boardSize = 5;
    cfg.history = 1;
    cfg.blocks = 0;
    cfg.filters = 2;
    cfg.valueHidden = 2;
    return cfg;
}

size_t indexOf(const std::vector<ParamView>& params, const std::string& name) {
    for (size_t i = 0; i < params.size(); ++i)
        if (params[i].name == name) return i;
    assert(false);
    return 0;
}

bool throwsInvalidArgument(const std::string& text) {
    try {
        LrSchedule::parse(text);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

static void test_momentum_and_weight_decay_hand_values() {
    Network net(tinyConfig(), 1);
    SgdMomentum opt(net, /*momentum=*/0.9f, /*weightDecay=*/0.1f);
    net.zeroGrad();
    std::vector<ParamView> params = net.params();
    const size_t wi = indexOf(params, "stem.conv.w");   // decayed
    const size_t bi = indexOf(params, "value.fc2.b");   // not decayed
    const size_t gi = indexOf(params, "stem.bn.gamma"); // not decayed, zero gradient
    ParamView& w = params[wi];
    ParamView& b = params[bi];
    assert(w.decay && !b.decay && !params[gi].decay);
    w.value[0] = 1.0f;
    w.grad[0] = 0.5f;
    b.value[0] = 1.0f;
    b.grad[0] = 0.5f;
    const float w1 = w.value[1];                        // zero gradient: pure decay
    const float gamma0 = params[gi].value[0];

    opt.step(net, 0.5f);
    // Weight: g' = 0.5 + 0.1 * 1 = 0.6; v = 0.6; w = 1 - 0.5 * 0.6 = 0.7.
    // Bias:   g' = 0.5;                  v = 0.5; b = 1 - 0.5 * 0.5 = 0.75.
    assert(near(w.value[0], 0.7));
    assert(near(b.value[0], 0.75));

    opt.step(net, 0.5f);   // step() leaves the gradients in place
    // Weight: g' = 0.5 + 0.1 * 0.7 = 0.57; v = 0.9 * 0.6 + 0.57 = 1.11; w = 0.7 - 0.555 = 0.145.
    // Bias:   g' = 0.5;                    v = 0.9 * 0.5 + 0.5  = 0.95; b = 0.75 - 0.475 = 0.275.
    assert(near(w.value[0], 0.145));
    assert(near(b.value[0], 0.275));
    assert(near(opt.velocity()[wi][0], 1.11));
    assert(near(opt.velocity()[bi][0], 0.95));
    // Pure decay: w -> 0.95 w (v = 0.1 w), then v = 0.09 w + 0.095 w = 0.185 w, w -> 0.8575 w.
    assert(near(w.value[1], 0.8575 * w1));
    assert(params[gi].value[0] == gamma0);              // no gradient, no decay: untouched
}

static void test_velocity_shapes() {
    Network net(tinyConfig(), 2);
    SgdMomentum opt(net, 0.9f, 1e-4f);
    std::vector<ParamView> params = net.params();
    assert(opt.velocity().size() == params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        assert(opt.velocity()[i].size() == params[i].size);
        for (float v : opt.velocity()[i]) assert(v == 0.0f);
    }
}

static void test_velocity_assignment_round_trip() {
    // Resuming training restores the velocity into a fresh optimizer; the
    // next step must then be exactly what the original optimizer would do.
    Network a(tinyConfig(), 3);
    SgdMomentum optA(a, 0.9f, 0.1f);
    auto setGradients = [](Network& net) {
        for (ParamView& p : net.params())
            for (size_t i = 0; i < p.size; ++i) p.grad[i] = 0.01f * static_cast<float>(i % 7) - 0.03f;
    };
    setGradients(a);
    optA.step(a, 0.1f);
    optA.step(a, 0.1f);

    Network restored = a;
    SgdMomentum optRestored(restored, 0.9f, 0.1f);
    optRestored.velocity() = optA.velocity();
    Network fresh = a;
    SgdMomentum optFresh(fresh, 0.9f, 0.1f);   // no restored velocity

    optA.step(a, 0.1f);
    optRestored.step(restored, 0.1f);
    optFresh.step(fresh, 0.1f);
    std::vector<ParamView> pa = a.params(), pr = restored.params(), pf = fresh.params();
    bool freshDiffers = false;
    for (size_t k = 0; k < pa.size(); ++k) {
        for (size_t i = 0; i < pa[k].size; ++i) {
            assert(pa[k].value[i] == pr[k].value[i]);
            freshDiffers = freshDiffers || pa[k].value[i] != pf[k].value[i];
        }
    }
    assert(freshDiffers);
    assert(optRestored.velocity() == optA.velocity());
}

static void test_lr_schedule_lookup() {
    LrSchedule s = LrSchedule::parse("0:0.02,50:0.002");
    assert(s.at(0) == 0.02f);
    assert(s.at(49) == 0.02f);
    assert(s.at(50) == 0.002f);
    assert(s.at(1000) == 0.002f);
    LrSchedule one = LrSchedule::parse("0:0.1");
    assert(one.at(0) == 0.1f && one.at(123456) == 0.1f);
}

static void test_lr_schedule_whitespace_and_canonical_form() {
    LrSchedule s = LrSchedule::parse("  0 : 0.02 ,\t50:0.002 , 100 :1e-4 ");
    assert(s.at(75) == 0.002f && s.at(100) == 1e-4f);
    const std::string canonical = s.toString();
    assert(canonical == "0:0.02,50:0.002,100:0.0001");
    assert(LrSchedule::parse(canonical).toString() == canonical);
}

static void test_lr_schedule_errors() {
    assert(throwsInvalidArgument(""));
    assert(throwsInvalidArgument("   "));
    assert(throwsInvalidArgument("5:0.1"));            // must start at generation 0
    assert(throwsInvalidArgument("0:0.1,0:0.2"));      // not strictly increasing
    assert(throwsInvalidArgument("0:0.1,20:0.2,10:0.3"));
    assert(throwsInvalidArgument("0:0.1,10:-1"));      // lr <= 0
    assert(throwsInvalidArgument("0:0"));
    assert(throwsInvalidArgument("0:abc"));
    assert(throwsInvalidArgument("0:0.1x"));
    assert(throwsInvalidArgument("0:0.1,5"));          // entry without ':'
    assert(throwsInvalidArgument("0:0.1,"));           // trailing comma
    assert(throwsInvalidArgument("-1:0.1"));
}

int main() {
    test_momentum_and_weight_decay_hand_values();
    test_velocity_shapes();
    test_velocity_assignment_round_trip();
    test_lr_schedule_lookup();
    test_lr_schedule_whitespace_and_canonical_form();
    test_lr_schedule_errors();
    std::printf("test_optimizer: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target test_optimizer`
Expected: FAIL at compile time with `fatal error: 'azgo/optimizer.hpp' file not found`.

- [ ] **Step 3: Write the header**

Create `include/azgo/optimizer.hpp`:

```cpp
// How the network learns from a gradient. Plain SGD steps straight down the
// current minibatch's gradient, which is noisy; momentum keeps a running
// "velocity" so consistent directions build up speed while noise cancels.
// Weight decay adds a pull toward zero on the weights (L2 regularization),
// which keeps them small and the network smoother. The learning rate follows
// a schedule: large early for fast progress, smaller later for fine-tuning.
#pragma once
#include <string>
#include <utility>
#include <vector>
#include "azgo/network.hpp"

namespace azgo {

// SGD with momentum and L2 weight decay:
//   g' = grad + weightDecay * w   (only for params with decay == true)
//   v  = momentum * v + g'
//   w -= lr * v
class SgdMomentum {
public:
    SgdMomentum(Network& net, float momentum, float weightDecay);
    void step(Network& net, float lr);
    std::vector<std::vector<float>>& velocity() { return velocity_; }   // one per param, params() order
    const std::vector<std::vector<float>>& velocity() const { return velocity_; }
private:
    float momentum_, weightDecay_;
    std::vector<std::vector<float>> velocity_;
};

// Stepped learning-rate schedule keyed on generation, e.g. "0:0.02,50:0.002":
// lr = value of the last entry whose generation <= g.
class LrSchedule {
public:
    // Throws std::invalid_argument: empty, malformed, first entry not at 0,
    // generations not strictly increasing, or lr <= 0.
    static LrSchedule parse(const std::string& text);
    float at(int generation) const;
    std::string toString() const;   // canonical "g:lr,g:lr" form
private:
    std::vector<std::pair<int, float>> steps_;
};

} // namespace azgo
```

- [ ] **Step 4: Write the implementation**

Create `src/optimizer.cpp`:

```cpp
#include "azgo/optimizer.hpp"
#include <cassert>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace azgo {

// ---------------------------------------------------------------- SgdMomentum

SgdMomentum::SgdMomentum(Network& net, float momentum, float weightDecay)
    : momentum_(momentum), weightDecay_(weightDecay) {
    for (const ParamView& p : net.params()) velocity_.emplace_back(p.size, 0.0f);
}

void SgdMomentum::step(Network& net, float lr) {
    std::vector<ParamView> params = net.params();
    assert(params.size() == velocity_.size());
    for (size_t k = 0; k < params.size(); ++k) {
        ParamView& p = params[k];
        std::vector<float>& v = velocity_[k];
        assert(v.size() == p.size);
        // Decay only weights: shrinking BN's gamma/beta or a bias toward zero
        // does not make the function smoother, it just fights the fit.
        const float decay = p.decay ? weightDecay_ : 0.0f;
        for (size_t i = 0; i < p.size; ++i) {
            const float g = p.grad[i] + decay * p.value[i];
            v[i] = momentum_ * v[i] + g;
            p.value[i] -= lr * v[i];
        }
    }
}

// ---------------------------------------------------------------- LrSchedule

namespace {

std::string trim(const std::string& s) {
    const size_t first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const size_t last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

[[noreturn]] void fail(const std::string& text, const std::string& why) {
    throw std::invalid_argument("learning-rate schedule '" + text + "': " + why);
}

// The shortest decimal that reads back as exactly v ("0.02", not
// "0.0199999996"), so toString() output is canonical and round-trips.
std::string formatFloat(float v) {
    char buf[32];
    for (int precision = 1; precision <= 9; ++precision) {
        std::snprintf(buf, sizeof buf, "%.*g", precision, static_cast<double>(v));
        if (std::strtof(buf, nullptr) == v) break;
    }
    return buf;
}

} // namespace

LrSchedule LrSchedule::parse(const std::string& text) {
    if (trim(text).empty()) fail(text, "empty");
    LrSchedule schedule;
    size_t start = 0;
    while (true) {
        const size_t comma = text.find(',', start);
        const std::string entry =
            trim(text.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        const size_t colon = entry.find(':');
        if (colon == std::string::npos) fail(text, "entry '" + entry + "' is not generation:lr");
        const std::string genText = trim(entry.substr(0, colon));
        const std::string lrText = trim(entry.substr(colon + 1));

        // strtol/strtof must consume the whole field: "5x" or "" is an error.
        char* end = nullptr;
        errno = 0;
        const long generation = std::strtol(genText.c_str(), &end, 10);
        if (genText.empty() || *end != '\0' || errno == ERANGE || generation < 0 || generation > INT_MAX)
            fail(text, "bad generation '" + genText + "'");
        errno = 0;
        const float lr = std::strtof(lrText.c_str(), &end);
        if (lrText.empty() || *end != '\0' || errno == ERANGE || !std::isfinite(lr))
            fail(text, "bad learning rate '" + lrText + "'");
        if (!(lr > 0.0f)) fail(text, "learning rate " + lrText + " is not > 0");
        if (schedule.steps_.empty() && generation != 0) fail(text, "the first entry must be at generation 0");
        if (!schedule.steps_.empty() && generation <= schedule.steps_.back().first)
            fail(text, "generations must be strictly increasing");
        schedule.steps_.emplace_back(static_cast<int>(generation), lr);

        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return schedule;
}

float LrSchedule::at(int generation) const {
    assert(!steps_.empty());
    float lr = steps_.front().second;
    for (const auto& step : steps_) {
        if (step.first > generation) break;
        lr = step.second;
    }
    return lr;
}

std::string LrSchedule::toString() const {
    std::string out;
    for (const auto& step : steps_) {
        if (!out.empty()) out += ',';
        out += std::to_string(step.first) + ':' + formatFloat(step.second);
    }
    return out;
}

} // namespace azgo
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_optimizer`
Expected: `100% tests passed` (the executable prints `test_optimizer: all tests passed`).

- [ ] **Step 6: Commit**

```bash
git add include/azgo/optimizer.hpp src/optimizer.cpp tests/test_optimizer.cpp
git commit -F - <<'EOF'
Add SGD with momentum and a stepped learning-rate schedule

The optimizer is kept separate from the network, as in tic-tac-toe, and
works through the named parameter views, so weight decay can apply to
conv and linear weights only. Its velocity is exposed so a resumed run
can restore it and continue with exactly the steps it would have taken.
The schedule parser is strict and prints a canonical form, so a config
file and a saved run agree on the text they mean.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

### Task 9: Serialization and checkpoints

**Files:**
- Create: `include/azgo/serialize.hpp`, `src/serialize.cpp`, `include/azgo/checkpoint.hpp`, `src/checkpoint.cpp`
- Test: `tests/test_serialize.cpp`, `tests/test_checkpoint.cpp`

**Interfaces:**
- Consumes: `Network(const NetConfig&, uint64_t)`, `Network::config()`, `params()`, `buffers()`, `infer`, `trainStep`, `Workspace`, `NetConfig::operator==` (Task 7). `NaiveBackend` (Task 5). `Rng` (Task 1, tests only).
- Produces: `azgo::BinaryWriter` / `azgo::BinaryReader` (all methods in contract.md; reads throw `std::runtime_error("unexpected end of data")`), `writeFileAtomically(path, contents)`, `saveNetwork(Network&, path)`, `loadNetwork(path)`, `readCheckpointConfig(path)`. These match contract.md exactly. Error messages: `cannot open checkpoint <path>`, or `<path>: <problem>` where `<problem>` is `not an AZGO checkpoint`, `unsupported version N`, `invalid network shape in header`, `has N tensors but the network needs M`, `tensor i is 'x' but expected 'y'`, `tensor 'x' has N values but expected M`, `unexpected data after the last tensor`, or `truncated`. Task 20 (`run_state`) reuses `BinaryWriter`/`BinaryReader`/`writeFileAtomically`.

- [ ] **Step 1: Write the failing serialization test**

Create `tests/test_serialize.cpp`:

```cpp
#include <unistd.h>   // getpid
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/serialize.hpp"

using namespace azgo;
namespace fs = std::filesystem;

namespace {

std::vector<unsigned char> bytesOf(const std::string& s) { return {s.begin(), s.end()}; }

std::string readFile(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

template <typename Read>
bool throwsEndOfData(const std::string& data, Read read) {
    std::istringstream in(data, std::ios::binary);
    BinaryReader r(in);
    try {
        read(r);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

} // namespace

static void test_round_trip_every_type() {
    std::ostringstream out(std::ios::binary);
    BinaryWriter w(out);
    const float tensor[5] = {0.0f, -1.5f, 3.25e-8f, 1e30f, std::numeric_limits<float>::infinity()};
    const unsigned char raw[3] = {0x00, 0xFF, 0x7F};
    const std::string withNul("a\0b", 3);
    w.u8(0xAB);
    w.u32(0xDEADBEEFu);
    w.u64(0x0123456789ABCDEFull);
    w.i32(-123456);
    w.f32(-0.1f);
    w.f64(3.141592653589793);
    w.str("hello");
    w.str("");
    w.str(withNul);
    w.floats(tensor, 5);
    w.bytes(raw, 3);

    std::istringstream in(out.str(), std::ios::binary);
    BinaryReader r(in);
    assert(r.u8() == 0xAB);
    assert(r.u32() == 0xDEADBEEFu);
    assert(r.u64() == 0x0123456789ABCDEFull);
    assert(r.i32() == -123456);
    assert(r.f32() == -0.1f);
    assert(r.f64() == 3.141592653589793);
    assert(r.str() == "hello");
    assert(r.str().empty());
    assert(r.str() == withNul);
    float back[5];
    r.floats(back, 5);
    for (int i = 0; i < 5; ++i) assert(back[i] == tensor[i]);
    unsigned char rawBack[3];
    r.bytes(rawBack, 3);
    assert(rawBack[0] == 0x00 && rawBack[1] == 0xFF && rawBack[2] == 0x7F);
    assert(in.peek() == std::char_traits<char>::eof());   // nothing extra was written
}

static void test_exact_byte_layout() {
    std::ostringstream out(std::ios::binary);
    BinaryWriter w(out);
    w.u32(0x01020304u);
    w.f32(1.0f);            // IEEE-754 0x3F800000
    w.i32(-1);
    w.str("ab");
    w.u64(1);
    const std::vector<unsigned char> expected = {
        0x04, 0x03, 0x02, 0x01,
        0x00, 0x00, 0x80, 0x3F,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x02, 0x00, 0x00, 0x00, 'a', 'b',
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    assert(bytesOf(out.str()) == expected);
}

static void test_reading_past_end_throws() {
    assert(throwsEndOfData("", [](BinaryReader& r) { r.u8(); }));
    assert(throwsEndOfData("abc", [](BinaryReader& r) { r.u32(); }));
    assert(throwsEndOfData("1234567", [](BinaryReader& r) { r.u64(); }));
    assert(throwsEndOfData(std::string("\x0A\0\0\0xy", 6), [](BinaryReader& r) { r.str(); }));   // claims 10 bytes
    assert(throwsEndOfData(std::string("\xFF\xFF\xFF\xFF", 4), [](BinaryReader& r) { r.str(); }));  // 4 GB claim
    assert(throwsEndOfData(std::string(8, '\0'), [](BinaryReader& r) {
        float f[3];
        r.floats(f, 3);
    }));
    // Exactly enough data is not an error.
    assert(!throwsEndOfData("abcd", [](BinaryReader& r) { r.u32(); }));
}

static void test_write_file_atomically() {
    const fs::path dir = fs::temp_directory_path() / ("azgo_test_serialize_" + std::to_string(getpid()));
    fs::remove_all(dir);
    assert(fs::create_directory(dir));
    const fs::path file = dir / "data.bin";
    const fs::path tmp = dir / "data.bin.tmp";

    writeFileAtomically(file.string(), "first version, longer");
    assert(readFile(file) == "first version, longer");
    writeFileAtomically(file.string(), std::string("2nd\0", 4));
    assert(readFile(file) == std::string("2nd\0", 4));   // replaced, not merged
    assert(!fs::exists(tmp));

    // A leftover temp file from a crash is simply overwritten.
    { std::ofstream(tmp, std::ios::binary) << "garbage garbage garbage"; }
    writeFileAtomically(file.string(), "third");
    assert(readFile(file) == "third");
    assert(!fs::exists(tmp));

    // Unwritable destination: a directory that does not exist.
    bool threw = false;
    try {
        writeFileAtomically((dir / "missing" / "x.bin").string(), "x");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    fs::remove_all(dir);
}

int main() {
    test_round_trip_every_type();
    test_exact_byte_layout();
    test_reading_past_end_throws();
    test_write_file_atomically();
    std::printf("test_serialize: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target test_serialize`
Expected: FAIL at compile time with `fatal error: 'azgo/serialize.hpp' file not found`.

- [ ] **Step 3: Write the serialization header**

Create `include/azgo/serialize.hpp`:

```cpp
// Binary files that read back identically on any machine. Multi-byte values
// are written least significant byte first (little-endian) by explicit
// shifts, never by dumping memory, so the format does not depend on the
// host's byte order. Files are replaced atomically: write a temporary file,
// then rename it over the old one, so a crash mid-write leaves either the
// old file or the new one, never half of each.
#pragma once
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>

namespace azgo {

// Little-endian binary I/O, independent of host byte order.
class BinaryWriter {
public:
    explicit BinaryWriter(std::ostream& os);
    void u8(uint8_t v);
    void u32(uint32_t v);
    void u64(uint64_t v);
    void i32(int32_t v);
    void f32(float v);
    void f64(double v);
    void str(const std::string& s);           // u32 length + bytes
    void floats(const float* p, size_t n);
    void bytes(const void* p, size_t n);
private:
    std::ostream& os_;
};

// Every read throws std::runtime_error on premature end of stream.
class BinaryReader {
public:
    explicit BinaryReader(std::istream& is);
    uint8_t u8();
    uint32_t u32();
    uint64_t u64();
    int32_t i32();
    float f32();
    double f64();
    std::string str();
    void floats(float* p, size_t n);
    void bytes(void* p, size_t n);
private:
    std::istream& is_;
};

// Writes to path + ".tmp", flushes, then renames over path (atomic replace).
// Throws std::runtime_error on I/O failure.
void writeFileAtomically(const std::string& path, const std::string& contents);

} // namespace azgo
```

- [ ] **Step 4: Write the serialization implementation**

Create `src/serialize.cpp`:

```cpp
#include "azgo/serialize.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace azgo {

namespace {

// Bulk transfers go through a fixed-size buffer: fast for million-float
// tensors, and a corrupt length field fails at end of data instead of
// allocating gigabytes first.
constexpr size_t kChunkFloats = 4096;

void storeU32(unsigned char* out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out[i] = static_cast<unsigned char>(v >> (8 * i));
}

uint32_t loadU32(const unsigned char* in) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(in[i]) << (8 * i);
    return v;
}

} // namespace

// ---------------------------------------------------------------- BinaryWriter

BinaryWriter::BinaryWriter(std::ostream& os) : os_(os) {}

void BinaryWriter::bytes(const void* p, size_t n) {
    os_.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
}

void BinaryWriter::u8(uint8_t v) { bytes(&v, 1); }

void BinaryWriter::u32(uint32_t v) {
    unsigned char b[4];
    storeU32(b, v);
    bytes(b, 4);
}

void BinaryWriter::u64(uint64_t v) {
    unsigned char b[8];
    for (int i = 0; i < 8; ++i) b[i] = static_cast<unsigned char>(v >> (8 * i));
    bytes(b, 8);
}

void BinaryWriter::i32(int32_t v) { u32(static_cast<uint32_t>(v)); }   // two's complement bits

void BinaryWriter::f32(float v) {
    // memcpy reinterprets the IEEE-754 bits without undefined behavior.
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    u32(bits);
}

void BinaryWriter::f64(double v) {
    uint64_t bits;
    std::memcpy(&bits, &v, 8);
    u64(bits);
}

void BinaryWriter::str(const std::string& s) {
    u32(static_cast<uint32_t>(s.size()));
    bytes(s.data(), s.size());
}

void BinaryWriter::floats(const float* p, size_t n) {
    unsigned char buf[kChunkFloats * 4];
    while (n > 0) {
        const size_t count = std::min(n, kChunkFloats);
        for (size_t i = 0; i < count; ++i) {
            uint32_t bits;
            std::memcpy(&bits, p + i, 4);
            storeU32(buf + 4 * i, bits);
        }
        bytes(buf, 4 * count);
        p += count;
        n -= count;
    }
}

// ---------------------------------------------------------------- BinaryReader

BinaryReader::BinaryReader(std::istream& is) : is_(is) {}

void BinaryReader::bytes(void* p, size_t n) {
    is_.read(static_cast<char*>(p), static_cast<std::streamsize>(n));
    if (static_cast<size_t>(is_.gcount()) != n) throw std::runtime_error("unexpected end of data");
}

uint8_t BinaryReader::u8() {
    uint8_t v;
    bytes(&v, 1);
    return v;
}

uint32_t BinaryReader::u32() {
    unsigned char b[4];
    bytes(b, 4);
    return loadU32(b);
}

uint64_t BinaryReader::u64() {
    unsigned char b[8];
    bytes(b, 8);
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(b[i]) << (8 * i);
    return v;
}

int32_t BinaryReader::i32() { return static_cast<int32_t>(u32()); }

float BinaryReader::f32() {
    const uint32_t bits = u32();
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

double BinaryReader::f64() {
    const uint64_t bits = u64();
    double v;
    std::memcpy(&v, &bits, 8);
    return v;
}

std::string BinaryReader::str() {
    const uint32_t length = u32();
    std::string s;
    char buf[4096];
    for (uint32_t done = 0; done < length;) {
        const uint32_t count = std::min<uint32_t>(length - done, sizeof buf);
        bytes(buf, count);
        s.append(buf, count);
        done += count;
    }
    return s;
}

void BinaryReader::floats(float* p, size_t n) {
    unsigned char buf[kChunkFloats * 4];
    while (n > 0) {
        const size_t count = std::min(n, kChunkFloats);
        bytes(buf, 4 * count);
        for (size_t i = 0; i < count; ++i) {
            const uint32_t bits = loadU32(buf + 4 * i);
            std::memcpy(p + i, &bits, 4);
        }
        p += count;
        n -= count;
    }
}

// ---------------------------------------------------------------- atomic write

void writeFileAtomically(const std::string& path, const std::string& contents) {
    const std::string tmp = path + ".tmp";
    {
        // Truncates any leftover .tmp from an earlier crash.
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open " + tmp + " for writing");
        out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        out.flush();
        if (!out) throw std::runtime_error("error writing " + tmp);
        out.close();
        if (out.fail()) throw std::runtime_error("error closing " + tmp);
    }
    // rename() replaces the destination in one step on POSIX file systems:
    // readers see the old file or the new one, never a partial write.
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) throw std::runtime_error("cannot rename " + tmp + " to " + path + ": " + ec.message());
}

} // namespace azgo
```

- [ ] **Step 5: Run the serialization test to verify it passes**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_serialize`
Expected: `100% tests passed` (the executable prints `test_serialize: all tests passed`).

- [ ] **Step 6: Write the failing checkpoint test**

Create `tests/test_checkpoint.cpp`:

```cpp
#include <unistd.h>   // getpid
#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/network.hpp"
#include "azgo/rng.hpp"

using namespace azgo;
namespace fs = std::filesystem;

namespace {

fs::path gDir;

std::string pathFor(const std::string& name) { return (gDir / name).string(); }

std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void writeFile(const std::string& path, const std::string& data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

NetConfig smallConfig() {
    NetConfig cfg;
    cfg.boardSize = 5;
    cfg.history = 2;
    cfg.blocks = 1;
    cfg.filters = 4;
    cfg.valueHidden = 8;
    return cfg;
}

// A network whose weights AND BN running statistics differ from a fresh one.
Network trainedNetwork(uint64_t seed) {
    const NetConfig cfg = smallConfig();
    Network net(cfg, seed);
    NaiveBackend be;
    Workspace ws(cfg, 4, be);
    Rng rng(seed);
    std::vector<float> input(4 * cfg.planes() * cfg.area()), pi(4 * cfg.actions(), 0.0f), z(4);
    for (float& v : input) v = rng.uniform() < 0.5 ? 0.0f : 1.0f;
    for (int b = 0; b < 4; ++b) pi[b * cfg.actions() + rng.below(cfg.actions())] = 1.0f;
    for (float& v : z) v = rng.uniform() < 0.5 ? -1.0f : 1.0f;
    net.trainStep(ws, input.data(), pi.data(), z.data(), 4);
    for (ParamView& p : net.params())
        for (size_t i = 0; i < p.size; ++i) p.value[i] -= 0.01f * p.grad[i];
    return net;
}

bool sameTensors(Network& a, Network& b) {
    std::vector<ParamView> pa = a.params(), pb = b.params();
    std::vector<BufferView> ba = a.buffers(), bb = b.buffers();
    if (pa.size() != pb.size() || ba.size() != bb.size()) return false;
    for (size_t i = 0; i < pa.size(); ++i)
        if (pa[i].name != pb[i].name || pa[i].size != pb[i].size ||
            std::memcmp(pa[i].value, pb[i].value, pa[i].size * sizeof(float)) != 0)
            return false;
    for (size_t i = 0; i < ba.size(); ++i)
        if (ba[i].name != bb[i].name || ba[i].size != bb[i].size ||
            std::memcmp(ba[i].value, bb[i].value, ba[i].size * sizeof(float)) != 0)
            return false;
    return true;
}

// Loads `path` expecting a runtime_error whose message names the file and
// contains `fragment`.
void expectLoadError(const std::string& path, const std::string& fragment) {
    try {
        loadNetwork(path);
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        if (what.find(path) == std::string::npos || what.find(fragment) == std::string::npos) {
            std::printf("unexpected message: %s\n", what.c_str());
            assert(false);
        }
        return;
    }
    assert(false);   // loading must fail
}

} // namespace

static void test_round_trip() {
    Network net = trainedNetwork(3);
    const std::string path = pathFor("round_trip.azn");
    saveNetwork(net, path);
    assert(!fs::exists(path + ".tmp"));

    Network loaded = loadNetwork(path);
    assert(loaded.config() == net.config());
    assert(sameTensors(net, loaded));
    assert(readCheckpointConfig(path) == smallConfig());

    // BN running statistics were restored too, so inference is identical.
    NaiveBackend be;
    Workspace ws(smallConfig(), 2, be);
    Rng rng(9);
    std::vector<float> input(2 * smallConfig().planes() * smallConfig().area());
    for (float& v : input) v = rng.uniform() < 0.5 ? 0.0f : 1.0f;
    std::vector<float> la(2 * smallConfig().actions()), lb(la.size()), va(2), vb(2);
    net.infer(ws, input.data(), 2, la.data(), va.data());
    loaded.infer(ws, input.data(), 2, lb.data(), vb.data());
    assert(std::memcmp(la.data(), lb.data(), la.size() * sizeof(float)) == 0);
    assert(std::memcmp(va.data(), vb.data(), va.size() * sizeof(float)) == 0);
}

static void test_missing_file() {
    const std::string path = pathFor("does_not_exist.azn");
    bool threw = false;
    try {
        readCheckpointConfig(path);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()) == "cannot open checkpoint " + path;
    }
    assert(threw);
    expectLoadError(path, "cannot open checkpoint");
}

static void test_bad_magic_and_version() {
    const std::string path = pathFor("bad_magic.azn");
    writeFile(path, "NOT A CHECKPOINT AT ALL, JUST TEXT");
    expectLoadError(path, "not an AZGO checkpoint");

    Network net(smallConfig(), 1);
    const std::string good = pathFor("good.azn");
    saveNetwork(net, good);
    std::string data = readFile(good);
    data[4] = 7;                                       // version field, little-endian u32 at offset 4
    const std::string versioned = pathFor("version7.azn");
    writeFile(versioned, data);
    expectLoadError(versioned, "unsupported version 7");
    bool threw = false;
    try {
        readCheckpointConfig(versioned);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()) == versioned + ": unsupported version 7";
    }
    assert(threw);
}

static void test_header_does_not_match_tensors() {
    // A header claiming 5 filters in front of a 4-filter network's tensors:
    // the first tensor's size gives it away.
    Network net(smallConfig(), 1);
    const std::string good = pathFor("four_filters.azn");
    saveNetwork(net, good);
    std::string data = readFile(good);
    data[8 + 3 * 4] = 5;                               // filters: 4th i32 of the NetConfig
    const std::string path = pathFor("five_filters.azn");
    writeFile(path, data);
    NetConfig claimed = readCheckpointConfig(path);
    assert(claimed.filters == 5 && claimed != smallConfig());
    expectLoadError(path, "tensor 'stem.conv.w' has");

    // A checkpoint of a different architecture loads as that architecture;
    // callers compare readCheckpointConfig() with what they expected.
    NetConfig other = smallConfig();
    other.blocks = 2;
    Network deeper(other, 1);
    const std::string deeperPath = pathFor("deeper.azn");
    saveNetwork(deeper, deeperPath);
    assert(readCheckpointConfig(deeperPath) != smallConfig());
    assert(loadNetwork(deeperPath).config() == other);

    // Trailing garbage is rejected too.
    writeFile(path, readFile(good) + "x");
    expectLoadError(path, "unexpected data after the last tensor");
}

static void test_truncated_checkpoint_names_file() {
    // Review Focus 3: a crash can leave a cut-off file; loading must fail
    // with a message naming it, never crash or return a half-loaded network.
    Network net = trainedNetwork(4);
    const std::string good = pathFor("full.azn");
    saveNetwork(net, good);
    const std::string data = readFile(good);
    const std::string path = pathFor("truncated.azn");
    writeFile(path, data.substr(0, data.size() / 2));
    expectLoadError(path, "truncated");
    writeFile(path, data.substr(0, 10));                // inside the header
    bool threw = false;
    try {
        readCheckpointConfig(path);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()) == path + ": truncated";
    }
    assert(threw);
    writeFile(path, "");
    expectLoadError(path, "truncated");
}

static void test_leftover_tmp_is_ignored_and_overwritten() {
    // Review Focus 3: garbage "<path>.tmp" from an interrupted save must not
    // affect loading, and the next save must replace it.
    Network first = trainedNetwork(5);
    const std::string path = pathFor("leftover.azn");
    saveNetwork(first, path);
    writeFile(path + ".tmp", "half-written garbage from a crash");
    Network loaded = loadNetwork(path);
    assert(sameTensors(first, loaded));

    Network second = trainedNetwork(6);
    saveNetwork(second, path);
    assert(!fs::exists(path + ".tmp"));
    Network reloaded = loadNetwork(path);
    assert(sameTensors(second, reloaded));
    assert(!sameTensors(first, reloaded));
}

int main() {
    gDir = fs::temp_directory_path() / ("azgo_test_checkpoint_" + std::to_string(getpid()));
    fs::remove_all(gDir);
    assert(fs::create_directory(gDir));
    test_round_trip();
    test_missing_file();
    test_bad_magic_and_version();
    test_header_does_not_match_tensors();
    test_truncated_checkpoint_names_file();
    test_leftover_tmp_is_ignored_and_overwritten();
    fs::remove_all(gDir);
    std::printf("test_checkpoint: all tests passed\n");
    return 0;
}
```

- [ ] **Step 7: Run test to verify it fails**

Run: `cmake --build build --target test_checkpoint`
Expected: FAIL at compile time with `fatal error: 'azgo/checkpoint.hpp' file not found`.

- [ ] **Step 8: Write the checkpoint header**

Create `include/azgo/checkpoint.hpp`:

```cpp
// A checkpoint is everything needed to rebuild a network exactly: its shape
// (NetConfig) and every tensor, including batch norm's running statistics,
// which inference depends on but the optimizer never touches. Each tensor is
// stored with its name and size, so loading can check it is putting each
// array where it came from and report precisely what does not match.
#pragma once
#include <string>
#include "azgo/network.hpp"

namespace azgo {

// Format: magic "AZGO" (u32 0x4F475A41), version u32 = 1, NetConfig as 5 x i32
// (boardSize, history, blocks, filters, valueHidden), u32 tensor count, then
// per tensor: name (str), u64 size, floats. Params in params() order, then
// buffers in buffers() order. Saved atomically.
void saveNetwork(Network& net, const std::string& path);

// Throws std::runtime_error naming the problem: missing file, bad magic,
// unsupported version, tensor name/size mismatch, truncation.
Network loadNetwork(const std::string& path);
NetConfig readCheckpointConfig(const std::string& path);

} // namespace azgo
```

- [ ] **Step 9: Write the checkpoint implementation**

Create `src/checkpoint.cpp`:

```cpp
#include "azgo/checkpoint.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "azgo/serialize.hpp"

namespace azgo {

namespace {

constexpr uint32_t kMagic = 0x4F475A41;   // bytes 'A' 'Z' 'G' 'O' when written little-endian
constexpr uint32_t kVersion = 1;

// A problem found in the file's contents (as opposed to running out of data).
struct FormatError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// One tensor as the checkpoint stores it.
struct Tensor {
    std::string name;
    float* data;
    size_t size;
};

std::vector<Tensor> tensorsOf(Network& net) {
    std::vector<Tensor> out;
    for (const ParamView& p : net.params()) out.push_back({p.name, p.value, p.size});
    for (const BufferView& b : net.buffers()) out.push_back({b.name, b.value, b.size});
    return out;
}

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
        cfg.blocks < 0 || cfg.blocks > 40 || cfg.filters < 1 || cfg.filters > 512 ||
        cfg.valueHidden < 1 || cfg.valueHidden > 1024) {
        throw FormatError("invalid network shape in header");
    }
    return cfg;
}

// Opens path and runs body(reader, stream). Every failure becomes a
// runtime_error that names the file: the reader's own "unexpected end of
// data" means the file was cut short.
template <typename Body>
auto withCheckpoint(const std::string& path, Body body) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open checkpoint " + path);
    BinaryReader reader(in);
    try {
        return body(reader, in);
    } catch (const FormatError& e) {
        throw std::runtime_error(path + ": " + e.what());
    } catch (const std::runtime_error&) {
        throw std::runtime_error(path + ": truncated");
    }
}

} // namespace

void saveNetwork(Network& net, const std::string& path) {
    std::ostringstream buffer(std::ios::binary);
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
        w.str(t.name);
        w.u64(t.size);
        w.floats(t.data, t.size);
    }
    writeFileAtomically(path, buffer.str());
}

Network loadNetwork(const std::string& path) {
    return withCheckpoint(path, [](BinaryReader& r, std::istream& in) {
        Network net(readHeader(r), 0);   // the seed is irrelevant: every tensor is overwritten
        const std::vector<Tensor> tensors = tensorsOf(net);
        const uint32_t count = r.u32();
        if (count != tensors.size()) {
            throw FormatError("has " + std::to_string(count) + " tensors but the network needs " +
                              std::to_string(tensors.size()));
        }
        for (size_t i = 0; i < tensors.size(); ++i) {
            const std::string name = r.str();
            if (name != tensors[i].name) {
                throw FormatError("tensor " + std::to_string(i) + " is '" + name + "' but expected '" +
                                  tensors[i].name + "'");
            }
            const uint64_t size = r.u64();
            if (size != tensors[i].size) {
                throw FormatError("tensor '" + name + "' has " + std::to_string(size) +
                                  " values but expected " + std::to_string(tensors[i].size));
            }
            r.floats(tensors[i].data, tensors[i].size);
        }
        if (in.peek() != std::char_traits<char>::eof()) throw FormatError("unexpected data after the last tensor");
        return net;
    });
}

NetConfig readCheckpointConfig(const std::string& path) {
    return withCheckpoint(path, [](BinaryReader& r, std::istream&) { return readHeader(r); });
}

} // namespace azgo
```

- [ ] **Step 10: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R "test_serialize|test_checkpoint"`
Expected: `100% tests passed, 0 tests failed out of 2`. Afterwards, no `azgo_test_checkpoint_*` or `azgo_test_serialize_*` directory remains under the system temp directory.

Then run the full suite in both builds, since this is the last task of the numeric core:

Run: `ctest --test-dir build --output-on-failure && cmake --build build-blas -j && ctest --test-dir build-blas --output-on-failure`
Expected: `100% tests passed` in both.

- [ ] **Step 11: Commit**

```bash
git add include/azgo/serialize.hpp src/serialize.cpp include/azgo/checkpoint.hpp src/checkpoint.cpp tests/test_serialize.cpp tests/test_checkpoint.cpp
git commit -F - <<'EOF'
Add little-endian binary I/O and network checkpoints

Checkpoints must load identically on any machine and survive crashes, so
values are written byte by byte in little-endian order and every file is
replaced atomically through a temporary file and rename. Loading checks
the magic, version, architecture and every tensor's name and size, and
reports failures with the file's path, including a checkpoint cut short
by a crash. A leftover .tmp from an interrupted save is ignored and
overwritten by the next save.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

> **Section notes (Tasks 10–15).**
>
> These do not change any public signature; they record choices the contract leaves open or slightly under-specifies.
>
> 1. **`ReplayBuffer` save format, `totalAdded`.** `BinaryWriter` has no `i64`, so `totalAdded` is written with `u64(static_cast<uint64_t>(totalAdded_))` and read back with the inverse cast. The value is never negative, so nothing is lost. If a later task wants a true `i64`, add `i64()` to `BinaryWriter`/`BinaryReader` in Task 9.
> 2. **`GameRecord::visits` counts are `uint16_t`.** Config allows `simulations` up to 100000, and tree reuse can push a root child past 65535 visits. `playSelfPlayGame` clamps counts at 65535. π is a ratio, so the effect is negligible, but lowering the `simulations` range to ≤ 30000 in Task 16 would make the clamp unreachable.
> 3. **Dirichlet with `dirichlet_scale = 0`.** Config allows 0, but `dirichlet()` needs α > 0. `Mcts` skips the noise when `dirichletScale <= 0` or `noiseFraction <= 0`, since the noise would be a no-op in those cases anyway.
> 4. **Evaluator batch capacity.** `Mcts`'s constructor asserts `cfg.batchSize <= evaluator.maxBatch()`. Whoever builds a `LocalEvaluator` for self-play or the arena (Tasks 18 and 21) must pass `maxBatch >= mcts_batch`.
> 5. **Tree match test.** The `search()` comment says the root is matched by "hash + ply". The implementation compares `Board::operator==` (stones and side to move) plus ply, which is slightly stricter than that and never weaker.

### Task 10: Evaluator

**Files:**
- Create: `include/azgo/evaluator.hpp`, `src/evaluator.cpp`
- Test: `tests/test_evaluator.cpp`

**Interfaces:**
- Consumes: `NetConfig` (`boardSize`, `history`, `planes()`, `area()`, `actions()`), `Workspace(const NetConfig&, int maxBatch, const Backend&)`, `Workspace::maxBatch()`, `Network(const NetConfig&, uint64_t)`, `Network::config()`, `Network::infer(Workspace&, const float*, int, float*, float*) const`, `NaiveBackend`, `kMinBoardSize`/`kMaxBoardSize`, `Game(int, float, int)`, `Game::play`, `defaultMaxMoves`, `encodeGame(const Game&, int H, int sym, float*)`, `numPlanes(int)`.
- Produces: `azgo::Evaluator` (virtual `boardSize()`, `history()`, `maxBatch()`, `evaluate(const float* input, int B, float* logits, float* values)`), `azgo::LocalEvaluator(const Network&, const Backend&, int maxBatch)` with `evaluations()`, and `azgo::UniformEvaluator(int n, int history, float value = 0, int maxBatch = 64)` with `calls()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluator.cpp`:

```cpp
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "azgo/encode.hpp"
#include "azgo/evaluator.hpp"
#include "azgo/game.hpp"

using namespace azgo;

static NetConfig tinyConfig() {
    NetConfig c;
    c.boardSize = 5;
    c.history = 2;
    c.blocks = 1;
    c.filters = 4;
    c.valueHidden = 8;
    return c;
}

// Three distinct positions of one short game, encoded back to back.
static std::vector<float> threePositions(const NetConfig& c) {
    const size_t per = static_cast<size_t>(c.planes()) * c.area();
    std::vector<float> input(3 * per);
    Game g(c.boardSize, 7.5f, defaultMaxMoves(c.boardSize));
    encodeGame(g, c.history, 0, input.data());
    g.play(12);                                   // C3, the centre
    encodeGame(g, c.history, 0, input.data() + per);
    g.play(6);                                    // B4
    encodeGame(g, c.history, 0, input.data() + 2 * per);
    return input;
}

static void test_local_evaluator_matches_network_infer_bitwise() {
    NetConfig c = tinyConfig();
    Network net(c, 123);
    NaiveBackend be;
    std::vector<float> input = threePositions(c);

    Workspace ws(c, 4, be);
    std::vector<float> refLogits(3 * c.actions()), refValues(3);
    net.infer(ws, input.data(), 3, refLogits.data(), refValues.data());

    LocalEvaluator ev(net, be, 4);
    assert(ev.boardSize() == 5);
    assert(ev.history() == 2);
    assert(ev.maxBatch() == 4);
    std::vector<float> logits(3 * c.actions(), 99.0f), values(3, 99.0f);
    ev.evaluate(input.data(), 3, logits.data(), values.data());

    // Same weights, same backend, same batch: the wrapper must add nothing.
    assert(std::memcmp(logits.data(), refLogits.data(), logits.size() * sizeof(float)) == 0);
    assert(std::memcmp(values.data(), refValues.data(), values.size() * sizeof(float)) == 0);
    for (float v : values) assert(v >= -1.0f && v <= 1.0f);   // tanh head
}

static void test_local_evaluator_counts_positions() {
    NetConfig c = tinyConfig();
    Network net(c, 5);
    NaiveBackend be;
    std::vector<float> input = threePositions(c);
    std::vector<float> logits(3 * c.actions()), values(3);

    LocalEvaluator ev(net, be, 3);
    assert(ev.evaluations() == 0);
    ev.evaluate(input.data(), 3, logits.data(), values.data());
    assert(ev.evaluations() == 3);
    ev.evaluate(input.data(), 1, logits.data(), values.data());
    assert(ev.evaluations() == 4);   // positions, not calls
}

static void test_uniform_evaluator() {
    UniformEvaluator ev(5, 2, 0.25f, 4);
    assert(ev.boardSize() == 5);
    assert(ev.history() == 2);
    assert(ev.maxBatch() == 4);
    assert(ev.calls() == 0);

    std::vector<float> input(4 * numPlanes(2) * 25, 1.0f);
    std::vector<float> logits(4 * 26, 7.0f), values(4, 7.0f);
    ev.evaluate(input.data(), 4, logits.data(), values.data());
    for (float l : logits) assert(l == 0.0f);
    for (float v : values) assert(v == 0.25f);
    assert(ev.calls() == 1);

    // A smaller batch writes only its own rows.
    std::fill(logits.begin(), logits.end(), 7.0f);
    std::fill(values.begin(), values.end(), 7.0f);
    ev.evaluate(input.data(), 2, logits.data(), values.data());
    for (int i = 0; i < 2 * 26; ++i) assert(logits[i] == 0.0f);
    for (int i = 2 * 26; i < 4 * 26; ++i) assert(logits[i] == 7.0f);
    assert(values[1] == 0.25f && values[2] == 7.0f);
    assert(ev.calls() == 2);   // calls, not positions

    UniformEvaluator neutral(9, 8);   // defaults: value 0, maxBatch 64
    assert(neutral.maxBatch() == 64);
    std::vector<float> in9(numPlanes(8) * 81), lg9(82, 1.0f), v9(1, 1.0f);
    neutral.evaluate(in9.data(), 1, lg9.data(), v9.data());
    assert(v9[0] == 0.0f && lg9[81] == 0.0f);
}

int main() {
    test_local_evaluator_matches_network_infer_bitwise();
    test_local_evaluator_counts_positions();
    test_uniform_evaluator();
    std::printf("test_evaluator: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL — compile error `fatal error: 'azgo/evaluator.hpp' file not found` in `tests/test_evaluator.cpp`.

- [ ] **Step 3: Write the header**

Create `include/azgo/evaluator.hpp`:

```cpp
// The evaluator is the seam between search and the neural network.
// MCTS never touches Network directly: it hands a batch of encoded
// positions to an Evaluator and gets back one row of policy logits and
// one value per position. LocalEvaluator runs the shared network on the
// calling thread with its own scratch memory; UniformEvaluator is a
// "know-nothing" stand-in (equal priors, constant value) that lets tests
// check the search logic without any network at all.
#pragma once
#include <memory>
#include "azgo/network.hpp"

namespace azgo {

// What MCTS asks of "the network": a batch of encoded positions in, policy
// logits and values out. A future GPU evaluator batching across threads
// would implement this same interface.
class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual int boardSize() const = 0;
    virtual int history() const = 0;
    virtual int maxBatch() const = 0;
    // input [B][numPlanes(history())][n*n]; logits [B][n*n+1]; values [B].
    virtual void evaluate(const float* input, int B, float* logits, float* values) = 0;
};

// One per worker thread: shares a read-only Network, owns a Workspace.
class LocalEvaluator : public Evaluator {
public:
    LocalEvaluator(const Network& net, const Backend& backend, int maxBatch);
    int boardSize() const override;
    int history() const override;
    int maxBatch() const override;
    void evaluate(const float* input, int B, float* logits, float* values) override;
    long long evaluations() const { return evaluations_; }   // positions evaluated so far
private:
    const Network& net_;
    Workspace ws_;
    long long evaluations_ = 0;
};

// Uniform priors (all logits 0) and a constant value. For tests and bench.
class UniformEvaluator : public Evaluator {
public:
    UniformEvaluator(int n, int history, float value = 0.0f, int maxBatch = 64);
    int boardSize() const override { return n_; }
    int history() const override { return history_; }
    int maxBatch() const override { return maxBatch_; }
    void evaluate(const float* input, int B, float* logits, float* values) override;
    long long calls() const { return calls_; }               // evaluate() invocations
private:
    int n_, history_, maxBatch_;
    float value_;
    long long calls_ = 0;
};

} // namespace azgo
```

- [ ] **Step 4: Write the implementation**

Create `src/evaluator.cpp`:

```cpp
#include "azgo/evaluator.hpp"
#include <algorithm>
#include <cassert>
#include "azgo/board.hpp"

namespace azgo {

// net_ is declared before ws_, so net_ is already bound when the Workspace
// is sized from its config.
LocalEvaluator::LocalEvaluator(const Network& net, const Backend& backend, int maxBatch)
    : net_(net), ws_(net.config(), maxBatch, backend) {
    assert(maxBatch >= 1);
}

int LocalEvaluator::boardSize() const { return net_.config().boardSize; }
int LocalEvaluator::history() const { return net_.config().history; }
int LocalEvaluator::maxBatch() const { return ws_.maxBatch(); }

void LocalEvaluator::evaluate(const float* input, int B, float* logits, float* values) {
    assert(B >= 1 && B <= ws_.maxBatch());
    net_.infer(ws_, input, B, logits, values);
    evaluations_ += B;
}

UniformEvaluator::UniformEvaluator(int n, int history, float value, int maxBatch)
    : n_(n), history_(history), maxBatch_(maxBatch), value_(value) {
    assert(n >= kMinBoardSize && n <= kMaxBoardSize);
    assert(history >= 1 && maxBatch >= 1);
    assert(value >= -1.0f && value <= 1.0f);
}

void UniformEvaluator::evaluate(const float* input, int B, float* logits, float* values) {
    (void)input;  // a uniform guess does not look at the position
    assert(B >= 1 && B <= maxBatch_);
    // All logits equal -> the softmax over the legal moves is uniform.
    std::fill(logits, logits + static_cast<size_t>(B) * (n_ * n_ + 1), 0.0f);
    std::fill(values, values + B, value_);
    ++calls_;
}

} // namespace azgo
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_evaluator`
Expected: `100% tests passed, 0 tests failed out of 1`. The executable itself prints `test_evaluator: all tests passed`.
Then run `ctest --test-dir build --output-on-failure`. Expected: every earlier test still passes.

- [ ] **Step 6: Commit**

```bash
git add include/azgo/evaluator.hpp src/evaluator.cpp tests/test_evaluator.cpp
git commit -F - <<'EOF'
Add the Evaluator seam between search and the network

MCTS will only ever see this interface: a batch of encoded positions in,
logits and values out. LocalEvaluator gives each worker thread its own
Workspace around the shared read-only Network. UniformEvaluator lets the
search tests run without any network. A cross-thread GPU batcher can
later implement the same interface without touching MCTS.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 11: MCTS — serial PUCT core

**Files:**
- Create: `include/azgo/mcts.hpp` (complete, final form), `src/mcts.cpp`
- Test: `tests/test_mcts.cpp`

**Interfaces:**
- Consumes: `Evaluator` (Task 10), `UniformEvaluator` (tests), `Game` (`board()`, `boardAt()`, `moveCount()`, `consecutivePasses()`, `komi()`, `maxMoves()`, `seen()`, `isLegal()`, `isTerminal()`, `play()`), `Board` (`isLegal`, `play`, `hashAfter`, `hash`, `areaScore`, `toMove`, `passAction`, `at`, `operator==`), `boardFromRows`, `encodeBoards(const std::vector<const Board*>&, int H, int sym, float*)`, `numPlanes`, `Rng` (`below`, `uniform`), `defaultMaxMoves`.
- Produces: `MctsConfig`, `MctsEdge`, `MctsNode`, and `Mcts` exactly as in the contract. In this task `search` runs one descent per evaluator call. `rootVisits`, `rootPriors`, `rootValue`, `rootTotalVisits`, `selectAction`, `reset`, `nodeCount`, `totalVirtualLoss`, `evaluatorBatches`, `node` and `edge` are final. `advance` and the private `setRootPriors` are declared here but defined in Task 13; nothing calls them before then, so the build links.

The header's private section lists every helper that Tasks 11–13 use, so the header never changes again. `src/mcts.cpp` grows: Task 12 replaces `Mcts::search`, and Task 13 replaces it once more and adds two definitions.

Binding algorithm details, as implemented below:
- A **terminal node** is one with `passes >= 2 || ply >= game.maxMoves()`. Its value is `sign(areaScore − komi)` for Black (0 on an exact tie), negated when White is to move.
- **Expansion** creates edges only for actions `a` where `board.isLegal(a)`, and where either `a` is pass or `h = hashAfter(a)` is neither seen by the game nor on the path from the node to the root. Priors are a softmax over those legal logits only.
- **PUCT:** `n = visits + virtualLoss`, `q = n ? (valueSum − virtualLoss) / n : 0`, `u = cPuct · prior · sqrt(max(1, Σn)) / (1 + n)`. The edge with the largest `q + u` wins, and on a tie the first index wins.
- **Backup:** the leaf's value is negated once per edge on the way up. Every node on the path, the leaf and the root included, gets `visits++`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mcts.cpp`:

```cpp
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>
#include "azgo/mcts.hpp"

using namespace azgo;

static bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

static MctsConfig serialConfig(int simulations) {
    MctsConfig c;
    c.simulations = simulations;
    c.batchSize = 1;
    return c;
}

static int sum(const std::vector<int>& v) { return std::accumulate(v.begin(), v.end(), 0); }

// Index of the root edge for `action`, or -1.
static int rootEdge(const Mcts& m, int action) {
    const MctsNode& root = m.node(0);
    for (int i = 0; i < root.numEdges; ++i)
        if (m.edge(root.firstEdge + i).action == action) return root.firstEdge + i;
    return -1;
}

// Black to move; Black's only winning move is D1 (row 4, col 3 = point 23),
// which captures the three white stones and leaves Black owning the whole
// board. With komi 20.5: capture -> 25 - 20.5 > 0; any other move or pass
// leaves Black 5 stones + 16 territory vs White 3 -> 18 - 20.5 < 0.
static Game winningCaptureGame(int maxMoves) {
    Board start = boardFromRows({".....",
                                 ".....",
                                 ".....",
                                 "XXXX.",
                                 "OOO.X"}, Color::Black);
    return Game(start, 20.5f, maxMoves);
}

static void test_root_visits_sum_to_simulations_and_tree_is_kept() {
    Game game(5, 7.5f, defaultMaxMoves(5));
    UniformEvaluator ev(5, 2);
    Mcts mcts(serialConfig(40), ev);
    Rng rng(1);
    mcts.search(game, rng);
    assert(sum(mcts.rootVisits()) == 40);
    assert(mcts.rootTotalVisits() == 40);
    assert(mcts.node(0).visits == 40);   // expanding the root is not a simulation
    const size_t nodesAfterFirst = mcts.nodeCount();

    mcts.search(game, rng);   // same position: the tree is kept and grows
    assert(sum(mcts.rootVisits()) == 80);
    assert(mcts.nodeCount() > nodesAfterFirst);
}

static void test_only_legal_actions_get_edges() {
    // White to move. A5 (point 0) is empty but surrounded by Black: suicide.
    Board start = boardFromRows({".X...",
                                 "X....",
                                 ".....",
                                 ".....",
                                 "....."}, Color::White);
    Game game(start, 7.5f, 50);
    UniformEvaluator ev(5, 2);
    Mcts mcts(serialConfig(10), ev);
    Rng rng(2);
    mcts.search(game, rng);

    const MctsNode& root = mcts.node(0);
    assert(root.numEdges == 25 - 2 - 1 + 1);   // minus stones, minus suicide, plus pass
    std::vector<int> visits = mcts.rootVisits();
    std::vector<float> priors = mcts.rootPriors();
    assert(visits.size() == 26 && priors.size() == 26);
    for (int a : {0, 1, 5}) {
        assert(rootEdge(mcts, a) == -1);
        assert(visits[a] == 0 && priors[a] == 0.0f);
    }
    float priorSum = 0.0f;
    for (int i = 0; i < root.numEdges; ++i) {
        const MctsEdge& e = mcts.edge(root.firstEdge + i);
        assert(game.isLegal(e.action));
        assert(near(e.rawPrior, 1.0 / 23));   // uniform over the legal actions only
        if (i > 0) assert(e.action > mcts.edge(root.firstEdge + i - 1).action);
        priorSum += e.prior;
    }
    assert(near(priorSum, 1.0));
}

static void test_superko_excludes_immediate_ko_retake_inside_tree() {
    // Black captures the ko at C4 (point 7), taking the white stone at B4
    // (point 6). White retaking at B4 would recreate the root position, so
    // the child node must have no edge for it -- even though Board alone
    // would allow it.
    Board start = boardFromRows({".XO..",
                                 "XO.O.",
                                 ".XO..",
                                 ".....",
                                 "....."}, Color::Black);
    Game game(start, 7.5f, 50);
    assert(game.isLegal(7));
    UniformEvaluator ev(5, 2);
    Mcts mcts(serialConfig(60), ev);   // > root edge count, so every root child gets expanded
    Rng rng(3);
    mcts.search(game, rng);

    const int e = rootEdge(mcts, 7);
    assert(e >= 0 && mcts.edge(e).child >= 0);
    const MctsNode& child = mcts.node(mcts.edge(e).child);
    assert(child.expanded);
    assert(child.board.at(6) == Color::Empty && child.board.toMove() == Color::White);
    assert(child.board.isLegal(6));   // locally legal ...
    for (int i = 0; i < child.numEdges; ++i)
        assert(mcts.edge(child.firstEdge + i).action != 6);   // ... but superko forbids it
    // A normal move for White is still there.
    bool hasE1 = false;
    for (int i = 0; i < child.numEdges; ++i)
        if (mcts.edge(child.firstEdge + i).action == 24) hasE1 = true;
    assert(hasE1);
}

// Same logits for every position: strongly prefers E1 (24), then C4 (7).
class PreferE1ThenC4 : public Evaluator {
public:
    int boardSize() const override { return 5; }
    int history() const override { return 2; }
    int maxBatch() const override { return 8; }
    void evaluate(const float*, int B, float* logits, float* values) override {
        for (int b = 0; b < B; ++b) {
            for (int a = 0; a < 26; ++a) logits[b * 26 + a] = 0.0f;
            logits[b * 26 + 24] = 10.0f;
            logits[b * 26 + 7] = 5.0f;
            values[b] = 0.0f;
        }
    }
};

static void test_superko_checks_positions_on_the_tree_path() {
    // Now it is White who can take the ko at C4 (7), capturing B4 (6).
    // Path: root -(Black E1)-> N1 -(White C4 takes)-> N2. Black retaking at
    // B4 from N2 would recreate N1, a position that exists only inside the
    // tree (the game has never seen it), so only the path check can catch it.
    Board start = boardFromRows({".OX..",
                                 "OX.X.",
                                 ".OX..",
                                 ".....",
                                 "....."}, Color::Black);
    Game game(start, 7.5f, 50);
    PreferE1ThenC4 ev;
    Mcts mcts(serialConfig(5), ev);
    Rng rng(8);
    mcts.search(game, rng);

    const int n1 = mcts.edge(rootEdge(mcts, 24)).child;
    assert(n1 >= 0 && mcts.node(n1).expanded);
    int n2 = -1;
    const MctsNode& a = mcts.node(n1);
    for (int i = 0; i < a.numEdges; ++i)
        if (mcts.edge(a.firstEdge + i).action == 7) n2 = mcts.edge(a.firstEdge + i).child;
    assert(n2 >= 0 && mcts.node(n2).expanded);
    const MctsNode& b = mcts.node(n2);
    assert(b.board.isLegal(6));
    assert(b.board.hashAfter(6) == a.board.hash());   // the retake recreates N1 ...
    assert(!game.seen(a.board.hash()));                // ... which the game never saw
    for (int i = 0; i < b.numEdges; ++i) assert(mcts.edge(b.firstEdge + i).action != 6);
}

static void test_finds_the_single_winning_capture() {
    Game game = winningCaptureGame(1);   // every child position is terminal
    UniformEvaluator ev(5, 2);
    Mcts mcts(serialConfig(50), ev);
    Rng rng(4);
    mcts.search(game, rng);

    assert(ev.calls() == 1);   // only the root: terminal leaves are scored by the rules
    assert(mcts.selectAction(0.0f, rng) == 23);
    assert(mcts.rootValue() > 0.0f);   // Black, to move, expects to win
    const MctsEdge& win = mcts.edge(rootEdge(mcts, 23));
    assert(win.visits > 25);
    assert(win.valueSum == static_cast<float>(win.visits));   // every visit was a +1 for Black
    const MctsEdge& pass = mcts.edge(rootEdge(mcts, 25));
    if (pass.visits > 0) assert(pass.valueSum == -static_cast<float>(pass.visits));
}

static void test_select_action_temperature_zero_picks_most_visited() {
    Game game(5, 7.5f, defaultMaxMoves(5));
    UniformEvaluator ev(5, 2);
    Mcts mcts(serialConfig(60), ev);
    Rng rng(5);
    mcts.search(game, rng);
    std::vector<int> visits = mcts.rootVisits();
    const int best = *std::max_element(visits.begin(), visits.end());
    for (int i = 0; i < 20; ++i) assert(visits[mcts.selectAction(0.0f, rng)] == best);
}

static void test_select_action_temperature_one_follows_visit_counts() {
    Game game(5, 7.5f, defaultMaxMoves(5));
    UniformEvaluator ev(5, 2);
    Mcts mcts(serialConfig(60), ev);
    Rng rng(6);
    mcts.search(game, rng);
    std::vector<int> visits = mcts.rootVisits();
    const int total = sum(visits);
    std::vector<int> counts(visits.size(), 0);
    const int draws = 2000;
    for (int i = 0; i < draws; ++i) counts[mcts.selectAction(1.0f, rng)]++;
    for (size_t a = 0; a < visits.size(); ++a) {
        if (visits[a] == 0) assert(counts[a] == 0);   // never an unvisited action
        assert(near(static_cast<double>(counts[a]) / draws, static_cast<double>(visits[a]) / total, 0.05));
    }
}

static void test_backup_flips_sign_one_ply() {
    // Every leaf is worth +0.5 to ITS side to move (White, one ply down), so
    // each root edge -- chosen by Black -- must average -0.5. With 20
    // simulations and 26 root edges every simulation stops one ply down:
    // an unvisited edge (Q = 0, full bonus) always beats a visited one.
    Game game(5, 7.5f, defaultMaxMoves(5));
    UniformEvaluator ev(5, 2, 0.5f);
    Mcts mcts(serialConfig(20), ev);
    Rng rng(7);
    mcts.search(game, rng);
    assert(mcts.node(0).netValue == 0.5f);
    const MctsNode& root = mcts.node(0);
    int visited = 0;
    for (int i = 0; i < root.numEdges; ++i) {
        const MctsEdge& e = mcts.edge(root.firstEdge + i);
        if (e.visits == 0) continue;
        ++visited;
        assert(e.valueSum / e.visits == -0.5f);
    }
    assert(visited == 20);
    assert(mcts.rootValue() == -0.5f);
}

int main() {
    test_root_visits_sum_to_simulations_and_tree_is_kept();
    test_only_legal_actions_get_edges();
    test_superko_excludes_immediate_ko_retake_inside_tree();
    test_superko_checks_positions_on_the_tree_path();
    test_finds_the_single_winning_capture();
    test_select_action_temperature_zero_picks_most_visited();
    test_select_action_temperature_one_follows_visit_counts();
    test_backup_flips_sign_one_ply();
    std::printf("test_mcts: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL — compile error `fatal error: 'azgo/mcts.hpp' file not found` in `tests/test_mcts.cpp`.

- [ ] **Step 3: Write the header**

Create `include/azgo/mcts.hpp`. The public part is verbatim from the contract, and the private helpers are the implementation-defined part:

```cpp
// Monte Carlo tree search with PUCT, as in AlphaZero. Each simulation walks
// down the tree choosing the edge with the best Q + U (value so far plus an
// exploration bonus proportional to the network's prior), asks the network
// about the new leaf, and backs the leaf's value up the path, flipping its
// sign at every ply because the players alternate. Nodes and edges live in
// two flat vectors (an arena) and refer to each other by index. Several
// descents are batched into one network call using "virtual loss": a path
// in flight temporarily counts as a loss so the next descent avoids it.
#pragma once
#include <cstdint>
#include <vector>
#include "azgo/evaluator.hpp"
#include "azgo/game.hpp"
#include "azgo/rng.hpp"

namespace azgo {

struct MctsConfig {
    int simulations = 200;       // new simulations per search() call
    int batchSize = 8;           // K descents per network call (virtual loss)
    float cPuct = 1.5f;
    bool addNoise = false;       // root Dirichlet noise (self-play only)
    float dirichletScale = 10.0f;// alpha = dirichletScale / (n * n)
    float noiseFraction = 0.25f; // epsilon
};

// Edge statistics are from the perspective of the player who chooses the
// edge (the parent's side to move).
struct MctsEdge {
    int action;
    float rawPrior;       // network prior (softmax over legal logits)
    float prior;          // rawPrior, or noise-mixed at the root
    int visits = 0;
    float valueSum = 0;
    int virtualLoss = 0;
    int child = -1;       // node index, -1 until created
};

struct MctsNode {
    Board board;
    int parent;           // -1 for root
    int ply;              // game move count at this node
    int passes;           // consecutive passes leading here
    int firstEdge = 0;
    int numEdges = 0;
    bool expanded = false;
    bool pending = false; // awaiting evaluation in the current round
    int visits = 0;       // completed simulations through this node
    int virtualLoss = 0;
    float netValue = 0;   // evaluator value at expansion (side to move)
};

class Mcts {
public:
    Mcts(const MctsConfig& cfg, Evaluator& evaluator);

    // Runs cfg.simulations simulations from game's current position. If the
    // tree root does not match game.board() (hash + ply), the tree is reset.
    // Asserts !game.isTerminal().
    void search(const Game& game, Rng& rng);

    std::vector<int> rootVisits() const;      // [numActions], 0 for unexpanded/illegal
    std::vector<float> rootPriors() const;    // [numActions], priors used (after noise)
    float rootValue() const;                  // mean backed-up value, root side-to-move view
    int rootTotalVisits() const;

    // temperature > 0: sample with probability ∝ visits^(1/temperature).
    // temperature == 0: most visited, ties broken uniformly with rng.
    int selectAction(float temperature, Rng& rng) const;

    // Re-roots at the child reached by action, keeping its subtree
    // (compacted into a fresh arena); resets if that child does not exist.
    void advance(int action);
    void reset();

    // Introspection for tests.
    size_t nodeCount() const { return nodes_.size(); }
    long long totalVirtualLoss() const;       // sum over all edges and nodes; 0 between searches
    long long evaluatorBatches() const { return evaluatorBatches_; }
    const MctsNode& node(int i) const { return nodes_[i]; }
    const MctsEdge& edge(int i) const { return edges_[i]; }

private:
    MctsConfig cfg_;
    Evaluator& eval_;
    std::vector<MctsNode> nodes_;
    std::vector<MctsEdge> edges_;
    long long evaluatorBatches_ = 0;
    // Further private members and helpers: implementation-defined (Tasks 11-13).
    const Game* game_ = nullptr;          // the game being searched; valid only inside search()
    std::vector<float> input_;            // one evaluator batch: [batchSize][planes][area]
    std::vector<float> logits_;           // [batchSize][actions]
    std::vector<float> values_;           // [batchSize]

    void prepareRoot(const Game& game);   // keep the tree if it matches, else reset + new root
    bool isTerminal(const MctsNode& node) const;
    float terminalValue(const MctsNode& node) const;   // exact result, node's side to move
    int selectEdge(int node) const;       // PUCT argmax over the node's edges
    int descend(std::vector<int>& pathEdges);          // root -> first unexpanded or terminal node
    void addVirtualLoss(int leaf, const std::vector<int>& pathEdges, int delta);
    void evaluateLeaves(const std::vector<int>& leaves);   // one evaluator call, then expand each
    void expand(int node, const float* logits, float value);
    void backup(int leaf, const std::vector<int>& pathEdges, float value);
    void setRootPriors(Rng& rng);         // Dirichlet noise at the root (Task 13)
};

} // namespace azgo
```

- [ ] **Step 4: Write the implementation**

Create `src/mcts.cpp`:

```cpp
#include "azgo/mcts.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include "azgo/encode.hpp"

namespace azgo {

Mcts::Mcts(const MctsConfig& cfg, Evaluator& evaluator) : cfg_(cfg), eval_(evaluator) {
    assert(cfg_.simulations >= 1);
    assert(cfg_.batchSize >= 1 && cfg_.batchSize <= eval_.maxBatch());
    const int area = eval_.boardSize() * eval_.boardSize();
    // Allocated once: every evaluator call reuses these buffers.
    input_.resize(static_cast<size_t>(cfg_.batchSize) * numPlanes(eval_.history()) * area);
    logits_.resize(static_cast<size_t>(cfg_.batchSize) * (area + 1));
    values_.resize(cfg_.batchSize);
}

void Mcts::reset() {
    nodes_.clear();
    edges_.clear();
}

void Mcts::prepareRoot(const Game& game) {
    assert(game.size() == eval_.boardSize());
    game_ = &game;
    // Keep the tree only if its root is exactly the game's current position
    // (same stones, same side to move, same move number).
    if (!nodes_.empty() && nodes_[0].board == game.board() && nodes_[0].ply == game.moveCount())
        return;
    reset();
    nodes_.push_back(MctsNode{game.board(), -1, game.moveCount(), game.consecutivePasses()});
}

bool Mcts::isTerminal(const MctsNode& node) const {
    return node.passes >= 2 || node.ply >= game_->maxMoves();
}

float Mcts::terminalValue(const MctsNode& node) const {
    // The game is over, so no guess is needed: score it with the rules.
    float margin = static_cast<float>(node.board.areaScore()) - game_->komi();
    float blackResult = margin > 0 ? 1.0f : (margin < 0 ? -1.0f : 0.0f);
    return node.board.toMove() == Color::Black ? blackResult : -blackResult;
}

int Mcts::selectEdge(int nodeIndex) const {
    const MctsNode& nd = nodes_[nodeIndex];
    assert(nd.expanded && nd.numEdges > 0);
    // In-flight descents count as visits (and as losses, below) so that the
    // next descent in the same batch is pushed towards other edges.
    int parentN = 0;
    for (int i = 0; i < nd.numEdges; ++i) {
        const MctsEdge& e = edges_[nd.firstEdge + i];
        parentN += e.visits + e.virtualLoss;
    }
    // max(1, .) lets the very first descent from a fresh node follow the
    // prior instead of seeing an all-zero exploration term.
    const float sqrtN = std::sqrt(static_cast<float>(std::max(1, parentN)));
    int best = nd.firstEdge;
    float bestScore = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < nd.numEdges; ++i) {
        const MctsEdge& e = edges_[nd.firstEdge + i];
        const int n = e.visits + e.virtualLoss;
        // Each virtual loss adds a -1 result for the chooser; unvisited Q = 0.
        const float q = n > 0 ? (e.valueSum - static_cast<float>(e.virtualLoss)) / n : 0.0f;
        const float u = cfg_.cPuct * e.prior * sqrtN / (1 + n);
        if (q + u > bestScore) {   // strict '>': the first index wins ties
            bestScore = q + u;
            best = nd.firstEdge + i;
        }
    }
    return best;
}

int Mcts::descend(std::vector<int>& pathEdges) {
    pathEdges.clear();
    int cur = 0;
    while (!isTerminal(nodes_[cur]) && nodes_[cur].expanded) {
        const int e = selectEdge(cur);
        pathEdges.push_back(e);
        if (edges_[e].child < 0) {
            // Children are created lazily, the first time a descent takes the edge.
            const MctsNode& parent = nodes_[cur];
            const int action = edges_[e].action;
            const bool isPass = action == parent.board.passAction();
            MctsNode child{parent.board, cur, parent.ply + 1, isPass ? parent.passes + 1 : 0};
            child.board.play(action);
            edges_[e].child = static_cast<int>(nodes_.size());
            nodes_.push_back(child);   // may reallocate: `parent` is not used after this
        }
        cur = edges_[e].child;
    }
    return cur;
}

void Mcts::addVirtualLoss(int leaf, const std::vector<int>& pathEdges, int delta) {
    for (int e : pathEdges) edges_[e].virtualLoss += delta;
    for (int x = leaf; x != -1; x = nodes_[x].parent) nodes_[x].virtualLoss += delta;
}

void Mcts::evaluateLeaves(const std::vector<int>& leaves) {
    const int B = static_cast<int>(leaves.size());
    assert(B >= 1 && B <= cfg_.batchSize);
    const int H = eval_.history();
    const int area = eval_.boardSize() * eval_.boardSize();
    const size_t inputStride = static_cast<size_t>(numPlanes(H)) * area;
    std::vector<const Board*> history;
    for (int i = 0; i < B; ++i) {
        // History planes: the leaf, its ancestors up to the root, then the
        // game's earlier positions (the root is the game's current board).
        history.clear();
        for (int x = leaves[i]; x != -1 && static_cast<int>(history.size()) < H; x = nodes_[x].parent)
            history.push_back(&nodes_[x].board);
        for (int ply = game_->moveCount() - 1; ply >= 0 && static_cast<int>(history.size()) < H; --ply)
            history.push_back(&game_->boardAt(ply));
        encodeBoards(history, H, 0, input_.data() + i * inputStride);
    }
    eval_.evaluate(input_.data(), B, logits_.data(), values_.data());
    ++evaluatorBatches_;
    for (int i = 0; i < B; ++i)
        expand(leaves[i], logits_.data() + static_cast<size_t>(i) * (area + 1), values_[i]);
}

void Mcts::expand(int nodeIndex, const float* logits, float value) {
    const Board& board = nodes_[nodeIndex].board;
    const int pass = board.passAction();
    // Positions on the path from this node back to the root: together with
    // the game's own history they are what positional superko forbids.
    std::vector<uint64_t> pathHashes;
    for (int x = nodeIndex; x != -1; x = nodes_[x].parent) pathHashes.push_back(nodes_[x].board.hash());

    std::vector<int> legal;
    for (int a = 0; a <= pass; ++a) {
        if (!board.isLegal(a)) continue;
        if (a != pass) {
            const uint64_t h = board.hashAfter(a);
            if (game_->seen(h) || std::find(pathHashes.begin(), pathHashes.end(), h) != pathHashes.end())
                continue;
        }
        legal.push_back(a);
    }
    assert(!legal.empty());   // pass is always legal

    // Softmax over the legal actions only; subtracting the max avoids overflow.
    float maxLogit = -std::numeric_limits<float>::infinity();
    for (int a : legal) maxLogit = std::max(maxLogit, logits[a]);
    std::vector<float> p(legal.size());
    float sum = 0.0f;
    for (size_t i = 0; i < legal.size(); ++i) {
        p[i] = std::exp(logits[legal[i]] - maxLogit);
        sum += p[i];
    }

    MctsNode& nd = nodes_[nodeIndex];
    nd.firstEdge = static_cast<int>(edges_.size());
    nd.numEdges = static_cast<int>(legal.size());
    for (size_t i = 0; i < legal.size(); ++i) {
        MctsEdge e;
        e.action = legal[i];
        e.rawPrior = p[i] / sum;
        e.prior = e.rawPrior;
        edges_.push_back(e);
    }
    nd.expanded = true;
    nd.netValue = value;
}

void Mcts::backup(int leaf, const std::vector<int>& pathEdges, float value) {
    // value is from the leaf's side to move. Each edge belongs to the player
    // choosing it, who is the opponent of the node it leads to: flip first.
    float v = value;
    for (auto it = pathEdges.rbegin(); it != pathEdges.rend(); ++it) {
        v = -v;
        edges_[*it].valueSum += v;
        edges_[*it].visits += 1;
    }
    for (int x = leaf; x != -1; x = nodes_[x].parent) nodes_[x].visits += 1;
}

void Mcts::search(const Game& game, Rng& rng) {
    assert(!game.isTerminal());
    (void)rng;   // used for root noise from Task 13 on
    prepareRoot(game);
    // The root needs priors before the first descent; evaluating it is not a simulation.
    if (!nodes_[0].expanded) evaluateLeaves({0});

    std::vector<int> path;
    for (int done = 0; done < cfg_.simulations; ++done) {
        const int leaf = descend(path);
        float value;
        if (isTerminal(nodes_[leaf])) {
            value = terminalValue(nodes_[leaf]);   // exact, no network call
        } else {
            evaluateLeaves({leaf});
            value = nodes_[leaf].netValue;
        }
        backup(leaf, path, value);
    }
    game_ = nullptr;
}

std::vector<int> Mcts::rootVisits() const {
    std::vector<int> v(eval_.boardSize() * eval_.boardSize() + 1, 0);
    if (nodes_.empty()) return v;
    const MctsNode& root = nodes_[0];
    for (int i = 0; i < root.numEdges; ++i) v[edges_[root.firstEdge + i].action] = edges_[root.firstEdge + i].visits;
    return v;
}

std::vector<float> Mcts::rootPriors() const {
    std::vector<float> p(eval_.boardSize() * eval_.boardSize() + 1, 0.0f);
    if (nodes_.empty()) return p;
    const MctsNode& root = nodes_[0];
    for (int i = 0; i < root.numEdges; ++i) p[edges_[root.firstEdge + i].action] = edges_[root.firstEdge + i].prior;
    return p;
}

int Mcts::rootTotalVisits() const {
    if (nodes_.empty()) return 0;
    int total = 0;
    for (int i = 0; i < nodes_[0].numEdges; ++i) total += edges_[nodes_[0].firstEdge + i].visits;
    return total;
}

float Mcts::rootValue() const {
    if (nodes_.empty()) return 0.0f;
    const MctsNode& root = nodes_[0];
    // Root edges hold values from the root player's view, so their mean is
    // the search's estimate of the root position.
    double sum = 0.0;
    int visits = 0;
    for (int i = 0; i < root.numEdges; ++i) {
        sum += edges_[root.firstEdge + i].valueSum;
        visits += edges_[root.firstEdge + i].visits;
    }
    return visits > 0 ? static_cast<float>(sum / visits) : root.netValue;
}

int Mcts::selectAction(float temperature, Rng& rng) const {
    const std::vector<int> visits = rootVisits();
    assert(rootTotalVisits() > 0);
    if (temperature <= 0.0f) {
        const int best = *std::max_element(visits.begin(), visits.end());
        std::vector<int> ties;
        for (int a = 0; a < static_cast<int>(visits.size()); ++a)
            if (visits[a] == best) ties.push_back(a);
        return ties[rng.below(static_cast<int>(ties.size()))];
    }
    // visits^(1/T) in double: T = 1 samples in proportion to visits, smaller
    // T sharpens towards the most visited action.
    std::vector<double> w(visits.size(), 0.0);
    double total = 0.0;
    for (size_t a = 0; a < visits.size(); ++a) {
        if (visits[a] > 0) w[a] = std::pow(static_cast<double>(visits[a]), 1.0 / temperature);
        total += w[a];
    }
    const double r = rng.uniform() * total;
    double cumulative = 0.0;
    int last = -1;
    for (size_t a = 0; a < w.size(); ++a) {
        if (w[a] <= 0.0) continue;
        cumulative += w[a];
        last = static_cast<int>(a);
        if (r < cumulative) return last;
    }
    return last;   // only reached through floating-point rounding at the very top
}

long long Mcts::totalVirtualLoss() const {
    long long total = 0;
    for (const MctsEdge& e : edges_) total += e.virtualLoss;
    for (const MctsNode& nd : nodes_) total += nd.virtualLoss;
    return total;
}

} // namespace azgo
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_mcts`
Expected: `100% tests passed, 0 tests failed out of 1`. The executable prints `test_mcts: all tests passed`.

- [ ] **Step 6: Commit**

```bash
git add include/azgo/mcts.hpp src/mcts.cpp tests/test_mcts.cpp
git commit -F - <<'EOF'
Add serial PUCT tree search over a flat node arena

Nodes and edges live in two vectors and point at each other by index,
so the tree is cheap to build, walk and later compact. Terminal
positions are scored exactly by the rules instead of by the network.
Superko is enforced inside the tree using both the game's history and
the positions on the current path. Without that, the search would
happily explore ko retakes that the real game forbids.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 12: MCTS — virtual-loss batching

**Files:**
- Modify: `src/mcts.cpp` (replace the whole `Mcts::search` definition)
- Modify: `tests/test_mcts.cpp` (add test functions before `main()`, replace `main()`)

**Interfaces:**
- Consumes: everything in Task 11's `src/mcts.cpp`: `prepareRoot`, `isTerminal`, `terminalValue`, `descend`, `addVirtualLoss`, `evaluateLeaves`, `backup`.
- Produces: the batched `Mcts::search` semantics that later tasks rely on. There are up to `cfg.batchSize` leaves per evaluator call, `evaluatorBatches()` counts those calls (the root's own call included), and `totalVirtualLoss() == 0` whenever `search` returns.

Each round runs up to `min(batchSize, simulations − done − pending)` descents, and every descent puts a virtual loss on its path. A descent can end in one of three ways:
- **Terminal node:** it is backed up at once, and its virtual loss is removed.
- **Node already pending in this round (a collision):** its virtual loss is removed and the round ends early.
- **Unexpanded node:** the node becomes pending.

After that, every pending leaf is encoded into one input buffer and the evaluator is called once. Each leaf is then expanded, its virtual loss removed, and its value backed up.

- [ ] **Step 1: Write the failing test**

In `tests/test_mcts.cpp`, add these functions directly above `int main()`:

```cpp
// Records the batch size of every evaluate() call.
class RecordingEvaluator : public Evaluator {
public:
    explicit RecordingEvaluator(int n) : n_(n) {}
    int boardSize() const override { return n_; }
    int history() const override { return 2; }
    int maxBatch() const override { return 64; }
    void evaluate(const float*, int B, float* logits, float* values) override {
        batches.push_back(B);
        std::fill(logits, logits + static_cast<size_t>(B) * (n_ * n_ + 1), 0.0f);
        std::fill(values, values + B, 0.0f);
    }
    std::vector<int> batches;
private:
    int n_;
};

static Game earlyNineByNine() {
    Game g(9, 7.5f, defaultMaxMoves(9));
    g.play(40);   // E5
    g.play(30);   // D6
    return g;
}

static void test_batched_visits_exact_and_virtual_loss_removed() {
    for (int batch : {1, 3, 8, 16}) {
        for (int sims : {10, 50}) {
            Game game = earlyNineByNine();
            UniformEvaluator ev(9, 2);
            MctsConfig cfg;
            cfg.simulations = sims;
            cfg.batchSize = batch;   // 16 > 10 simulations: the round is cut short
            Mcts mcts(cfg, ev);
            Rng rng(11);
            mcts.search(game, rng);
            assert(mcts.rootTotalVisits() == sims);
            assert(mcts.node(0).visits == sims);
            assert(mcts.totalVirtualLoss() == 0);
            for (size_t i = 0; i < mcts.nodeCount(); ++i) assert(!mcts.node(static_cast<int>(i)).pending);
            mcts.search(game, rng);
            assert(mcts.rootTotalVisits() == 2 * sims);
            assert(mcts.totalVirtualLoss() == 0);
        }
    }
}

static void test_batch_one_calls_evaluator_once_per_leaf() {
    // 30 simulations over 80 root edges: every simulation stops one ply down,
    // so no leaf is terminal and each simulation evaluates exactly one leaf.
    Game game = earlyNineByNine();
    UniformEvaluator ev(9, 2);
    MctsConfig cfg;
    cfg.simulations = 30;
    cfg.batchSize = 1;
    Mcts mcts(cfg, ev);
    Rng rng(12);
    mcts.search(game, rng);
    assert(mcts.evaluatorBatches() == 30 + 1);   // + the root
    assert(ev.calls() == 31);
}

static void test_batching_reduces_evaluator_calls() {
    Game game = earlyNineByNine();
    UniformEvaluator ev1(9, 2), ev8(9, 2);
    MctsConfig cfg;
    cfg.simulations = 64;
    cfg.batchSize = 1;
    Mcts serial(cfg, ev1);
    cfg.batchSize = 8;
    Mcts batched(cfg, ev8);
    Rng r1(13), r8(13);
    serial.search(game, r1);
    batched.search(game, r8);
    assert(batched.rootTotalVisits() == 64);
    assert(batched.evaluatorBatches() < serial.evaluatorBatches());
    // Virtual loss spreads the 8 descents of a round over 8 different root
    // edges, so no round collides here: 1 root call + 64 / 8 rounds.
    assert(batched.evaluatorBatches() == 1 + 8);
}

static void test_evaluator_never_sees_more_than_batch_size() {
    // White owns the board with two one-point eyes (A5 = 0, E1 = 24). Black's
    // only legal action is pass, so every descent of a round funnels into the
    // same child: collisions are guaranteed. Pass-pass endings also give
    // terminal leaves inside the rounds.
    Board start = boardFromRows({".OOOO",
                                 "OOOOO",
                                 "OOOOO",
                                 "OOOOO",
                                 "OOOO."}, Color::Black);
    Game game(start, 7.5f, 50);
    RecordingEvaluator ev(5);
    MctsConfig cfg;
    cfg.simulations = 40;
    cfg.batchSize = 5;
    Mcts mcts(cfg, ev);
    Rng rng(14);
    mcts.search(game, rng);

    assert(mcts.node(0).numEdges == 1);                  // pass only
    assert(ev.batches.size() >= 2);
    assert(ev.batches[0] == 1);                          // the root, alone
    assert(ev.batches[1] == 1);                          // 2nd descent collided with the 1st
    int evaluated = 0;
    for (int b : ev.batches) {
        assert(b >= 1 && b <= 5);
        evaluated += b;
    }
    int expanded = 0, terminal = 0;
    for (size_t i = 0; i < mcts.nodeCount(); ++i) {
        const MctsNode& nd = mcts.node(static_cast<int>(i));
        expanded += nd.expanded ? 1 : 0;
        terminal += nd.passes >= 2 ? 1 : 0;
    }
    assert(terminal > 0);
    assert(evaluated == expanded);   // every evaluated leaf is expanded exactly once
    assert(static_cast<long long>(ev.batches.size()) == mcts.evaluatorBatches());
    assert(mcts.rootTotalVisits() == 40);
    assert(mcts.totalVirtualLoss() == 0);
}

static void test_batched_search_is_deterministic() {
    Game game = earlyNineByNine();
    MctsConfig cfg;
    cfg.simulations = 100;
    cfg.batchSize = 4;
    UniformEvaluator ev1(9, 2), ev2(9, 2);
    Mcts a(cfg, ev1), b(cfg, ev2);
    Rng ra(99), rb(99);
    a.search(game, ra);
    b.search(game, rb);
    assert(a.rootVisits() == b.rootVisits());
    assert(a.nodeCount() == b.nodeCount());
}

static void test_batched_search_still_finds_winning_capture() {
    Game game = winningCaptureGame(1);
    UniformEvaluator ev(5, 2);
    MctsConfig cfg;
    cfg.simulations = 50;
    cfg.batchSize = 8;
    Mcts mcts(cfg, ev);
    Rng rng(15);
    mcts.search(game, rng);
    assert(ev.calls() == 1);   // terminal leaves are backed up inside the round
    assert(mcts.rootTotalVisits() == 50);
    assert(mcts.totalVirtualLoss() == 0);
    assert(mcts.selectAction(0.0f, rng) == 23);
}
```

Replace `main()` with:

```cpp
int main() {
    test_root_visits_sum_to_simulations_and_tree_is_kept();
    test_only_legal_actions_get_edges();
    test_superko_excludes_immediate_ko_retake_inside_tree();
    test_superko_checks_positions_on_the_tree_path();
    test_finds_the_single_winning_capture();
    test_select_action_temperature_zero_picks_most_visited();
    test_select_action_temperature_one_follows_visit_counts();
    test_backup_flips_sign_one_ply();
    test_batched_visits_exact_and_virtual_loss_removed();
    test_batch_one_calls_evaluator_once_per_leaf();
    test_batching_reduces_evaluator_calls();
    test_evaluator_never_sees_more_than_batch_size();
    test_batched_search_is_deterministic();
    test_batched_search_still_finds_winning_capture();
    std::printf("test_mcts: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_mcts`
Expected: FAIL — `Assertion failed: (batched.evaluatorBatches() < serial.evaluatorBatches())` in `test_batching_reduces_evaluator_calls`. The serial search makes one evaluator call per simulation whatever `batchSize` is.

- [ ] **Step 3: Write the implementation**

In `src/mcts.cpp`, replace the entire `void Mcts::search(const Game& game, Rng& rng) { ... }` definition with:

```cpp
void Mcts::search(const Game& game, Rng& rng) {
    assert(!game.isTerminal());
    (void)rng;   // used for root noise from Task 13 on
    prepareRoot(game);
    // The root needs priors before the first descent; evaluating it is not a simulation.
    if (!nodes_[0].expanded) evaluateLeaves({0});

    std::vector<int> leaves;                  // pending leaves of this round
    std::vector<std::vector<int>> paths;      // their edge paths
    std::vector<int> path;
    int done = 0;
    while (done < cfg_.simulations) {
        leaves.clear();
        paths.clear();
        // One round: up to batchSize descents, never more than the
        // simulations still owed. Each descent's virtual loss steers the
        // next one elsewhere.
        for (int d = 0; d < cfg_.batchSize; ++d) {
            if (done + static_cast<int>(leaves.size()) >= cfg_.simulations) break;
            const int leaf = descend(path);
            addVirtualLoss(leaf, path, +1);
            if (isTerminal(nodes_[leaf])) {
                // Exact value, no network needed: finish this simulation now.
                addVirtualLoss(leaf, path, -1);
                backup(leaf, path, terminalValue(nodes_[leaf]));
                ++done;
            } else if (nodes_[leaf].pending) {
                // Collision: another descent this round already waits on this
                // leaf. Undo and evaluate what we have rather than spin.
                addVirtualLoss(leaf, path, -1);
                break;
            } else {
                nodes_[leaf].pending = true;
                leaves.push_back(leaf);
                paths.push_back(path);
            }
        }
        if (leaves.empty()) continue;
        evaluateLeaves(leaves);   // one evaluator call for the whole round
        for (size_t i = 0; i < leaves.size(); ++i) {
            nodes_[leaves[i]].pending = false;
            addVirtualLoss(leaves[i], paths[i], -1);
            backup(leaves[i], paths[i], nodes_[leaves[i]].netValue);
            ++done;
        }
    }
    game_ = nullptr;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_mcts`
Expected: `100% tests passed, 0 tests failed out of 1`. The executable prints `test_mcts: all tests passed`.

- [ ] **Step 5: Commit**

```bash
git add src/mcts.cpp tests/test_mcts.cpp
git commit -F - <<'EOF'
Batch MCTS descents with virtual loss

A network call on one position wastes most of a gemm, so each round now
collects up to batchSize leaves and evaluates them together. A path in
flight counts as a temporary loss, which steers the next descent
elsewhere. Terminal leaves are finished inside the round. A collision
on a pending leaf ends the round early instead of spinning. With
batchSize 1 this is exactly the serial search.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 13: MCTS — root noise and tree reuse

**Files:**
- Modify: `src/mcts.cpp` (replace `Mcts::search` again, and add `Mcts::setRootPriors` and `Mcts::advance`)
- Modify: `tests/test_mcts.cpp` (add test functions before `main()`, replace `main()`)

**Interfaces:**
- Consumes: `dirichlet(Rng&, double alpha, int n)` (Task 1), plus Task 11/12 internals.
- Produces: `Mcts::advance(int action)` and root Dirichlet noise when `MctsConfig::addNoise` is set. Every `search()` sets the root priors afresh: `prior = (1 − ε)·rawPrior + ε·η` with `η ~ Dir(dirichletScale / n²)`, or `prior = rawPrior` without noise. Non-root priors always equal `rawPrior`. Self-play (Task 14) and the arena (Task 18) rely on `advance`.

- [ ] **Step 1: Write the failing test**

In `tests/test_mcts.cpp`, add these functions directly above `int main()`:

```cpp
// Nodes reachable from `from` (inclusive) through created children.
static int reachable(const Mcts& m, int from) {
    int count = 0;
    std::vector<int> stack{from};
    while (!stack.empty()) {
        const int x = stack.back();
        stack.pop_back();
        ++count;
        const MctsNode& nd = m.node(x);
        for (int i = 0; i < nd.numEdges; ++i)
            if (m.edge(nd.firstEdge + i).child >= 0) stack.push_back(m.edge(nd.firstEdge + i).child);
    }
    return count;
}

static void test_noise_changes_root_priors_only() {
    Game game(5, 7.5f, defaultMaxMoves(5));
    UniformEvaluator ev(5, 2);
    MctsConfig cfg;
    cfg.simulations = 100;
    cfg.batchSize = 4;
    cfg.addNoise = true;
    Mcts mcts(cfg, ev);
    Rng rng(21);
    mcts.search(game, rng);

    const MctsNode& root = mcts.node(0);
    double sum = 0.0;
    int changed = 0;
    for (int i = 0; i < root.numEdges; ++i) {
        const MctsEdge& e = mcts.edge(root.firstEdge + i);
        assert(near(e.rawPrior, 1.0 / 26));
        assert(e.prior > 0.0f);
        sum += e.prior;
        if (!near(e.prior, e.rawPrior, 1e-4)) ++changed;
    }
    assert(near(sum, 1.0));
    assert(changed > root.numEdges / 2);
    std::vector<float> priors = mcts.rootPriors();
    for (int i = 0; i < root.numEdges; ++i)
        assert(priors[mcts.edge(root.firstEdge + i).action] == mcts.edge(root.firstEdge + i).prior);

    int innerExpanded = 0;
    for (size_t k = 1; k < mcts.nodeCount(); ++k) {
        const MctsNode& nd = mcts.node(static_cast<int>(k));
        innerExpanded += nd.expanded ? 1 : 0;
        for (int i = 0; i < nd.numEdges; ++i)
            assert(mcts.edge(nd.firstEdge + i).prior == mcts.edge(nd.firstEdge + i).rawPrior);
    }
    assert(innerExpanded > 0);
}

static void test_no_noise_means_prior_equals_raw_prior() {
    Game game(5, 7.5f, defaultMaxMoves(5));
    UniformEvaluator ev(5, 2);
    MctsConfig cfg;
    cfg.simulations = 60;
    cfg.addNoise = false;
    Mcts mcts(cfg, ev);
    Rng rng(22);
    mcts.search(game, rng);
    for (size_t k = 0; k < mcts.nodeCount(); ++k) {
        const MctsNode& nd = mcts.node(static_cast<int>(k));
        for (int i = 0; i < nd.numEdges; ++i)
            assert(mcts.edge(nd.firstEdge + i).prior == mcts.edge(nd.firstEdge + i).rawPrior);
    }
}

static void test_noise_follows_the_rng_stream() {
    Game game(5, 7.5f, defaultMaxMoves(5));
    UniformEvaluator ev(5, 2);
    MctsConfig cfg;
    cfg.simulations = 30;
    cfg.addNoise = true;
    Mcts a(cfg, ev), b(cfg, ev), c(cfg, ev);
    Rng ra(5), rb(5), rc(6);
    a.search(game, ra);
    b.search(game, rb);
    c.search(game, rc);
    assert(a.rootPriors() == b.rootPriors() && a.rootVisits() == b.rootVisits());
    assert(a.rootPriors() != c.rootPriors());
}

static void test_advance_keeps_the_chosen_subtree() {
    Game game(5, 7.5f, defaultMaxMoves(5));
    UniformEvaluator ev(5, 2);
    MctsConfig cfg;
    cfg.simulations = 64;
    cfg.batchSize = 4;
    Mcts mcts(cfg, ev);
    Rng rng(23);
    mcts.search(game, rng);

    const int a = mcts.selectAction(0.0f, rng);
    const int e = rootEdge(mcts, a);
    const int oldChild = mcts.edge(e).child;
    assert(oldChild >= 0);
    const MctsNode before = mcts.node(oldChild);
    std::vector<int> oldVisits;
    std::vector<float> oldValues;
    for (int i = 0; i < before.numEdges; ++i) {
        oldVisits.push_back(mcts.edge(before.firstEdge + i).visits);
        oldValues.push_back(mcts.edge(before.firstEdge + i).valueSum);
    }
    const int oldReachable = reachable(mcts, oldChild);
    const int reused = sum(oldVisits);
    assert(reused > 0);   // the most visited child has been searched below

    mcts.advance(a);
    assert(mcts.node(0).board == before.board);
    assert(mcts.node(0).parent == -1);
    assert(mcts.node(0).visits == before.visits);
    assert(mcts.node(0).numEdges == before.numEdges);
    assert(static_cast<int>(mcts.nodeCount()) == oldReachable);
    assert(reachable(mcts, 0) == oldReachable);   // compacted: every node is reachable
    for (int i = 0; i < before.numEdges; ++i) {
        const MctsEdge& ne = mcts.edge(mcts.node(0).firstEdge + i);
        assert(ne.visits == oldVisits[i]);
        assert(ne.valueSum == oldValues[i]);
        if (ne.child >= 0) assert(mcts.node(ne.child).parent == 0);
    }
    assert(mcts.totalVirtualLoss() == 0);
    assert(mcts.rootTotalVisits() == reused);

    // The next search recognises the position and builds on the kept tree.
    game.play(a);
    const long long batchesBefore = mcts.evaluatorBatches();
    mcts.search(game, rng);
    assert(mcts.rootTotalVisits() == reused + 64);
    assert(mcts.node(0).board == game.board());
    // The root was already expanded, so no extra batch was spent on it:
    // 64 simulations in rounds of 4 without collisions = 16 batches.
    assert(mcts.evaluatorBatches() - batchesBefore <= 16);
}

static void test_advance_to_unexplored_child_resets() {
    Game game(5, 7.5f, defaultMaxMoves(5));
    UniformEvaluator ev(5, 2);
    MctsConfig cfg;
    cfg.simulations = 5;   // 5 of 26 root edges get a child
    cfg.batchSize = 1;
    Mcts mcts(cfg, ev);
    Rng rng(24);
    mcts.advance(12);   // empty tree: no-op
    assert(mcts.nodeCount() == 0);
    mcts.search(game, rng);
    int unexplored = -1;
    const MctsNode& root = mcts.node(0);
    for (int i = 0; i < root.numEdges; ++i)
        if (mcts.edge(root.firstEdge + i).child < 0) unexplored = mcts.edge(root.firstEdge + i).action;
    assert(unexplored >= 0);
    mcts.advance(unexplored);
    assert(mcts.nodeCount() == 0);
    assert(mcts.rootTotalVisits() == 0);
}

static void test_search_on_a_different_position_resets() {
    Game game(5, 7.5f, defaultMaxMoves(5));
    UniformEvaluator ev(5, 2);
    MctsConfig cfg;
    cfg.simulations = 30;
    Mcts mcts(cfg, ev);
    Rng rng(25);
    mcts.search(game, rng);
    game.play(12);   // moved on without advance(): the tree is stale
    mcts.search(game, rng);
    assert(mcts.node(0).board == game.board());
    assert(mcts.node(0).ply == 1);
    assert(mcts.rootTotalVisits() == 30);
    assert(rootEdge(mcts, 12) == -1);   // C3 is occupied now
}
```

Replace `main()` with:

```cpp
int main() {
    test_root_visits_sum_to_simulations_and_tree_is_kept();
    test_only_legal_actions_get_edges();
    test_superko_excludes_immediate_ko_retake_inside_tree();
    test_superko_checks_positions_on_the_tree_path();
    test_finds_the_single_winning_capture();
    test_select_action_temperature_zero_picks_most_visited();
    test_select_action_temperature_one_follows_visit_counts();
    test_backup_flips_sign_one_ply();
    test_batched_visits_exact_and_virtual_loss_removed();
    test_batch_one_calls_evaluator_once_per_leaf();
    test_batching_reduces_evaluator_calls();
    test_evaluator_never_sees_more_than_batch_size();
    test_batched_search_is_deterministic();
    test_batched_search_still_finds_winning_capture();
    test_noise_changes_root_priors_only();
    test_no_noise_means_prior_equals_raw_prior();
    test_noise_follows_the_rng_stream();
    test_advance_keeps_the_chosen_subtree();
    test_advance_to_unexplored_child_resets();
    test_search_on_a_different_position_resets();
    std::printf("test_mcts: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL — link error `undefined symbol: azgo::Mcts::advance(int)` (on macOS: `"azgo::Mcts::advance(int)", referenced from: ...`) for `test_mcts`.

- [ ] **Step 3: Write the implementation**

In `src/mcts.cpp`, replace the entire `Mcts::search` definition with this version. It differs from Task 12 only at the top, where `(void)rng;` goes and `setRootPriors(rng);` is called after root expansion:

```cpp
void Mcts::search(const Game& game, Rng& rng) {
    assert(!game.isTerminal());
    prepareRoot(game);
    // The root needs priors before the first descent; evaluating it is not a simulation.
    if (!nodes_[0].expanded) evaluateLeaves({0});
    setRootPriors(rng);   // fresh noise (or none) for this search

    std::vector<int> leaves;                  // pending leaves of this round
    std::vector<std::vector<int>> paths;      // their edge paths
    std::vector<int> path;
    int done = 0;
    while (done < cfg_.simulations) {
        leaves.clear();
        paths.clear();
        // One round: up to batchSize descents, never more than the
        // simulations still owed. Each descent's virtual loss steers the
        // next one elsewhere.
        for (int d = 0; d < cfg_.batchSize; ++d) {
            if (done + static_cast<int>(leaves.size()) >= cfg_.simulations) break;
            const int leaf = descend(path);
            addVirtualLoss(leaf, path, +1);
            if (isTerminal(nodes_[leaf])) {
                // Exact value, no network needed: finish this simulation now.
                addVirtualLoss(leaf, path, -1);
                backup(leaf, path, terminalValue(nodes_[leaf]));
                ++done;
            } else if (nodes_[leaf].pending) {
                // Collision: another descent this round already waits on this
                // leaf. Undo and evaluate what we have rather than spin.
                addVirtualLoss(leaf, path, -1);
                break;
            } else {
                nodes_[leaf].pending = true;
                leaves.push_back(leaf);
                paths.push_back(path);
            }
        }
        if (leaves.empty()) continue;
        evaluateLeaves(leaves);   // one evaluator call for the whole round
        for (size_t i = 0; i < leaves.size(); ++i) {
            nodes_[leaves[i]].pending = false;
            addVirtualLoss(leaves[i], paths[i], -1);
            backup(leaves[i], paths[i], nodes_[leaves[i]].netValue);
            ++done;
        }
    }
    game_ = nullptr;
}
```

Then add these two definitions at the end of `src/mcts.cpp`, directly above the closing `} // namespace azgo`:

```cpp
void Mcts::setRootPriors(Rng& rng) {
    MctsNode& root = nodes_[0];
    MctsEdge* e = edges_.data() + root.firstEdge;
    // Skip the noise when it would be a no-op (and Dirichlet needs alpha > 0).
    if (!cfg_.addNoise || cfg_.noiseFraction <= 0.0f || cfg_.dirichletScale <= 0.0f) {
        for (int i = 0; i < root.numEdges; ++i) e[i].prior = e[i].rawPrior;
        return;
    }
    // alpha shrinks with the board so the total noise "mass" stays about
    // dirichletScale spread over the legal moves, whatever the board size.
    const int area = eval_.boardSize() * eval_.boardSize();
    const std::vector<double> eta = dirichlet(rng, cfg_.dirichletScale / area, root.numEdges);
    const float eps = cfg_.noiseFraction;
    for (int i = 0; i < root.numEdges; ++i)
        e[i].prior = (1.0f - eps) * e[i].rawPrior + eps * static_cast<float>(eta[i]);
}

void Mcts::advance(int action) {
    if (nodes_.empty()) return;
    const MctsNode& root = nodes_[0];
    int newRoot = -1;
    for (int i = 0; i < root.numEdges; ++i)
        if (edges_[root.firstEdge + i].action == action) newRoot = edges_[root.firstEdge + i].child;
    if (newRoot < 0) {   // never explored (or illegal): nothing worth keeping
        reset();
        return;
    }
    // Copy only the subtree under newRoot into fresh vectors, so memory stays
    // bounded by what one move's search built. Explicit-stack DFS; each stack
    // entry is (old node, its new parent, the new edge that must point to it).
    std::vector<MctsNode> nodes;
    std::vector<MctsEdge> edges;
    struct Item { int oldNode, newParent, newEdge; };
    std::vector<Item> stack{{newRoot, -1, -1}};
    while (!stack.empty()) {
        const Item it = stack.back();
        stack.pop_back();
        const int idx = static_cast<int>(nodes.size());
        MctsNode nd = nodes_[it.oldNode];
        nd.parent = it.newParent;
        nd.firstEdge = static_cast<int>(edges.size());
        if (it.newEdge >= 0) edges[it.newEdge].child = idx;
        for (int i = 0; i < nd.numEdges; ++i) {
            MctsEdge e = edges_[nodes_[it.oldNode].firstEdge + i];
            const int oldChild = e.child;
            e.child = -1;   // filled in when that child is copied
            edges.push_back(e);
            if (oldChild >= 0) stack.push_back({oldChild, idx, static_cast<int>(edges.size()) - 1});
        }
        nodes.push_back(nd);
    }
    nodes_.swap(nodes);
    edges_.swap(edges);
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_mcts`
Expected: `100% tests passed, 0 tests failed out of 1`. The executable prints `test_mcts: all tests passed`.
Then run `ctest --test-dir build --output-on-failure`. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/mcts.cpp tests/test_mcts.cpp
git commit -F - <<'EOF'
Add root Dirichlet noise and subtree reuse to MCTS

Self-play needs noise at the root so that moves the network dislikes
still get tried now and then. Alpha is scaled by board area so 5x5 and
19x19 get comparable exploration. advance() keeps the searched subtree
under the move actually played, copying only the reachable nodes into
fresh vectors. The next search starts from those statistics and memory
stays bounded by one move's work.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 14: Parallel runner and self-play

**Files:**
- Create: `include/azgo/parallel.hpp`, `src/parallel.cpp`, `include/azgo/selfplay.hpp`, `src/selfplay.cpp`
- Test: `tests/test_parallel.cpp`, `tests/test_selfplay.cpp`

**Interfaces:**
- Consumes: `Mcts` (`search`, `rootVisits`, `rootTotalVisits`, `selectAction`, `advance`), `MctsConfig`, `Evaluator`, `UniformEvaluator`, `LocalEvaluator`, `Network`, `NetConfig`, `NaiveBackend`, `Game` (`Game(int, float, int)`, `board`, `play`, `isTerminal`, `moveCount`, `resultForBlack`), `defaultMaxMoves`, `Board::exportStones`, `Rng`, `deriveSeed`.
- Produces: `parallelFor(int count, int workers, const std::function<void(int, int)>&)`, `resolveWorkers(int)`, `SelfPlayConfig`, `GameRecord` (with `zAt`), and `playSelfPlayGame(Evaluator&, const SelfPlayConfig&, Rng&)`. The trainer (Task 21) and the arena (Task 18) use them.

The rng is consumed in a fixed order for each ply: first the root noise inside `search`, then `selectAction`. Two runs with the same seed therefore play the same game, whichever thread plays them.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_parallel.cpp`:

```cpp
#include <atomic>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/parallel.hpp"

using namespace azgo;

static void test_every_index_exactly_once() {
    const int cases[][2] = {{0, 4}, {1, 4}, {100, 1}, {100, 7}};
    for (const auto& c : cases) {
        const int count = c[0], workers = c[1];
        std::vector<std::atomic<int>> hits(count);
        for (auto& h : hits) h.store(0);
        std::atomic<int> badWorker{0};
        parallelFor(count, workers, [&](int index, int worker) {
            if (worker < 0 || worker >= workers) badWorker.fetch_add(1);
            hits[index].fetch_add(1);
        });
        assert(badWorker.load() == 0);
        for (int i = 0; i < count; ++i) assert(hits[i].load() == 1);
    }
}

static void test_results_stored_by_index() {
    std::vector<int> squares(50, -1);
    parallelFor(50, 5, [&](int index, int) { squares[index] = index * index; });
    for (int i = 0; i < 50; ++i) assert(squares[i] == i * i);
}

static void test_exception_propagates_after_join() {
    for (int workers : {1, 4}) {
        std::atomic<int> started{0};
        bool caught = false;
        try {
            parallelFor(100, workers, [&](int index, int) {
                started.fetch_add(1);
                if (index == 13) throw std::runtime_error("boom at 13");
            });
        } catch (const std::runtime_error& e) {
            caught = std::string(e.what()) == "boom at 13";
        }
        assert(caught);
        // Workers stop taking new indices after the failure, so not all 100
        // run; with one worker exactly indices 0..13 do.
        assert(started.load() >= 14 && started.load() <= 100);
        if (workers == 1) assert(started.load() == 14);
    }
}

static void test_resolve_workers() {
    assert(resolveWorkers(0) >= 1);
    assert(resolveWorkers(-2) >= 1);
    assert(resolveWorkers(3) == 3);
    assert(resolveWorkers(1) == 1);
}

int main() {
    test_every_index_exactly_once();
    test_results_stored_by_index();
    test_exception_propagates_after_join();
    test_resolve_workers();
    std::printf("test_parallel: all tests passed\n");
    return 0;
}
```

Create `tests/test_selfplay.cpp`:

```cpp
#include <cassert>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#include "azgo/game.hpp"
#include "azgo/parallel.hpp"
#include "azgo/selfplay.hpp"

using namespace azgo;

// An "untrained network" that is certain it should pass.
class AlwaysPassEvaluator : public Evaluator {
public:
    int boardSize() const override { return 5; }
    int history() const override { return 2; }
    int maxBatch() const override { return 8; }
    void evaluate(const float*, int B, float* logits, float* values) override {
        for (int b = 0; b < B; ++b) {
            for (int a = 0; a < 25; ++a) logits[b * 26 + a] = -20.0f;
            logits[b * 26 + 25] = 20.0f;   // pass
            values[b] = 0.0f;
        }
    }
};

static void test_immediate_pass_pass_game() {
    // Review Focus 1. Why both sides pass even with root noise: with one
    // simulation the only descent from the root compares Q + U with Q = 0
    // for every edge and U proportional to the prior (sqrt(max(1, 0)) = 1).
    // Pass has raw prior ~1 (e^40 odds), so after mixing in 25% noise it
    // keeps >= 0.75 while every other move gets <= 0.25 * eta <= 0.25. The
    // single visit goes to pass, and temperature-1 sampling over the visit
    // counts {pass: 1} can only return pass. The same holds at ply 1.
    AlwaysPassEvaluator ev;
    SelfPlayConfig cfg;
    cfg.mcts.simulations = 1;
    cfg.mcts.batchSize = 1;
    cfg.komi = 7.5f;
    Rng rng(1);
    GameRecord rec = playSelfPlayGame(ev, cfg, rng);

    assert(rec.boardSize == 5 && rec.komi == 7.5f);
    assert(rec.positions() == 2);
    assert(rec.moves == std::vector<int>({25, 25}));
    assert(rec.stones.size() == 2u * 25);
    for (uint8_t s : rec.stones) assert(s == static_cast<uint8_t>(Color::Empty));
    for (int ply = 0; ply < 2; ++ply) {
        assert(rec.visits[ply].size() == 1);
        assert(rec.visits[ply][0].first == 25 && rec.visits[ply][0].second == 1);
    }
    // Empty board: area score 0, so komi decides it for White.
    assert(rec.resultForBlack == -1.0f);
    assert(rec.toMoveAt(0) == Color::Black && rec.toMoveAt(1) == Color::White);
    assert(rec.zAt(0) == -1.0f);   // Black's position: Black lost
    assert(rec.zAt(1) == +1.0f);   // White's position: White won
}

static void test_z_for_a_black_win_and_a_tie() {
    GameRecord rec;
    rec.boardSize = 5;
    rec.moves = {12, 25, 25};
    rec.resultForBlack = 1.0f;
    assert(rec.zAt(0) == 1.0f && rec.zAt(1) == -1.0f && rec.zAt(2) == 1.0f);
    rec.resultForBlack = 0.0f;
    for (int ply = 0; ply < 3; ++ply) assert(rec.zAt(ply) == 0.0f);
}

static void test_record_is_consistent_with_a_replay() {
    UniformEvaluator ev(5, 2);
    SelfPlayConfig cfg;
    cfg.mcts.simulations = 16;
    cfg.mcts.batchSize = 4;
    cfg.temperatureMoves = 6;
    cfg.komi = 7.5f;
    Rng rng(2024);
    GameRecord rec = playSelfPlayGame(ev, cfg, rng);

    const int n = 5, area = 25;
    assert(rec.positions() >= 2);
    assert(rec.stones.size() == static_cast<size_t>(rec.positions()) * area);
    assert(rec.visits.size() == static_cast<size_t>(rec.positions()));
    for (int p = 0; p < area; ++p) assert(rec.stonesAt(0)[p] == static_cast<uint8_t>(Color::Empty));

    // Replay the same game with the same seed and a second search tree: every
    // recorded field must match what the search produced at that ply.
    Game game(n, cfg.komi, defaultMaxMoves(n));
    MctsConfig m = cfg.mcts;
    m.addNoise = true;
    UniformEvaluator ev2(5, 2);
    Mcts mcts(m, ev2);
    Rng replayRng(2024);
    std::vector<uint8_t> stones(area);
    bool reusedSomething = false;
    for (int ply = 0; ply < rec.positions(); ++ply) {
        assert(!game.isTerminal());
        assert(game.board().toMove() == rec.toMoveAt(ply));   // alternates, Black first
        game.board().exportStones(stones.data());
        assert(std::memcmp(stones.data(), rec.stonesAt(ply), area) == 0);

        mcts.search(game, replayRng);
        std::vector<int> visits = mcts.rootVisits();
        int recorded = 0;
        int prevAction = -1;
        for (const auto& av : rec.visits[ply]) {
            assert(av.first > prevAction);   // ascending action order
            prevAction = av.first;
            assert(av.second > 0 && visits[av.first] == av.second);
            recorded += av.second;
        }
        // Sum = this search's 16 simulations + whatever the reused subtree held.
        assert(recorded == mcts.rootTotalVisits());
        assert(recorded >= 16);
        if (recorded > 16) reusedSomething = true;

        const float T = ply < cfg.temperatureMoves ? 1.0f : 0.0f;
        const int a = mcts.selectAction(T, replayRng);
        assert(a == rec.moves[ply]);
        game.play(a);
        mcts.advance(a);
    }
    assert(game.isTerminal());
    assert(reusedSomething);   // tree reuse carried visits into later plies
    assert(rec.resultForBlack == game.resultForBlack());
    for (int ply = 0; ply < rec.positions(); ++ply)
        assert(rec.zAt(ply) == (ply % 2 == 0 ? rec.resultForBlack : -rec.resultForBlack));
}

static bool sameRecord(const GameRecord& a, const GameRecord& b) {
    return a.boardSize == b.boardSize && a.komi == b.komi && a.moves == b.moves &&
           a.stones == b.stones && a.visits == b.visits && a.resultForBlack == b.resultForBlack;
}

static std::vector<GameRecord> playGames(const Network& net, const Backend& be, int workers) {
    SelfPlayConfig cfg;
    cfg.mcts.simulations = 8;
    cfg.mcts.batchSize = 4;
    cfg.temperatureMoves = 4;
    std::vector<std::unique_ptr<LocalEvaluator>> evaluators;
    for (int w = 0; w < workers; ++w)
        evaluators.push_back(std::make_unique<LocalEvaluator>(net, be, cfg.mcts.batchSize));
    std::vector<GameRecord> records(6);
    parallelFor(6, workers, [&](int index, int worker) {
        Rng rng(deriveSeed(42, static_cast<uint64_t>(index)));
        records[index] = playSelfPlayGame(*evaluators[worker], cfg, rng);
    });
    return records;
}

static void test_self_play_is_independent_of_worker_count() {
    NetConfig c;
    c.boardSize = 5;
    c.history = 2;
    c.blocks = 1;
    c.filters = 4;
    c.valueHidden = 8;
    Network net(c, 7);
    NaiveBackend be;
    std::vector<GameRecord> one = playGames(net, be, 1);
    std::vector<GameRecord> three = playGames(net, be, 3);
    bool anyDifferent = false;
    for (int i = 0; i < 6; ++i) {
        assert(sameRecord(one[i], three[i]));
        assert(one[i].positions() >= 2);
        if (i > 0 && one[i].moves != one[0].moves) anyDifferent = true;
    }
    assert(anyDifferent);   // different seeds give different games
}

int main() {
    test_immediate_pass_pass_game();
    test_z_for_a_black_win_and_a_tie();
    test_record_is_consistent_with_a_replay();
    test_self_play_is_independent_of_worker_count();
    std::printf("test_selfplay: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake --build build -j`
Expected: FAIL — `fatal error: 'azgo/parallel.hpp' file not found` (and `'azgo/selfplay.hpp' file not found`).

- [ ] **Step 3: Write the headers**

Create `include/azgo/parallel.hpp`:

```cpp
// A tiny work-sharing loop. Self-play and arena games are independent, so
// W threads simply pull the next game index from a shared atomic counter
// until none are left. Which thread plays which game varies from run to
// run, but every game's randomness is seeded from its index and results
// are stored by index, so the output does not depend on the thread count.
#pragma once
#include <functional>

namespace azgo {

// Calls fn(index, worker) for every index in [0, count) using `workers`
// std::threads (worker in [0, workers)). Indices are handed out by an atomic
// counter, so callers must store results by index, never by completion order.
// The first exception thrown by fn is rethrown after all threads join.
void parallelFor(int count, int workers, const std::function<void(int index, int worker)>& fn);

// workers <= 0 means std::thread::hardware_concurrency() (at least 1).
int resolveWorkers(int workers);

} // namespace azgo
```

Create `include/azgo/selfplay.hpp`:

```cpp
// Self-play: the network plays both sides against itself, choosing every
// move by MCTS. Each position is stored with the search's visit counts
// (the improved policy target pi); once the game ends, every position also
// gets the final result z seen from its own side to move. Root Dirichlet
// noise and temperature-1 sampling in the opening keep the games varied.
#pragma once
#include <cstdint>
#include <utility>
#include <vector>
#include "azgo/board.hpp"
#include "azgo/evaluator.hpp"
#include "azgo/mcts.hpp"
#include "azgo/rng.hpp"

namespace azgo {

struct SelfPlayConfig {
    MctsConfig mcts;             // addNoise is forced on by playSelfPlayGame
    int temperatureMoves = 20;   // plies sampled with temperature 1, then argmax
    float komi = 7.5f;
    int maxMoves = 0;            // 0 means defaultMaxMoves(n)
};

// A finished game in compact form. Position `ply` is the board BEFORE
// moves[ply] was played; Black moves at even plies (no handicap).
struct GameRecord {
    int boardSize = 0;
    float komi = 0;
    std::vector<int> moves;
    std::vector<uint8_t> stones;   // positions() * n*n Color bytes
    std::vector<std::vector<std::pair<int16_t, uint16_t>>> visits;  // per ply: (action, count), count > 0
    float resultForBlack = 0;      // +1, -1, or 0

    int positions() const { return static_cast<int>(moves.size()); }
    const uint8_t* stonesAt(int ply) const { return stones.data() + static_cast<size_t>(ply) * boardSize * boardSize; }
    Color toMoveAt(int ply) const { return ply % 2 == 0 ? Color::Black : Color::White; }
    float zAt(int ply) const;      // resultForBlack from the side to move at ply
};

// Plays one self-play game from the empty board with root noise and tree reuse.
GameRecord playSelfPlayGame(Evaluator& evaluator, const SelfPlayConfig& cfg, Rng& rng);

} // namespace azgo
```

- [ ] **Step 4: Write the implementations**

Create `src/parallel.cpp`:

```cpp
#include "azgo/parallel.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace azgo {

void parallelFor(int count, int workers, const std::function<void(int index, int worker)>& fn) {
    assert(workers >= 1);
    if (count <= 0) return;
    workers = std::min(workers, count);   // no idle threads

    std::atomic<int> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr firstError;
    std::mutex errorMutex;

    auto run = [&](int worker) {
        // Stop taking new indices once any worker has failed; indices already
        // running on other threads finish normally.
        while (!failed.load()) {
            const int index = next.fetch_add(1);
            if (index >= count) return;
            try {
                fn(index, worker);
            } catch (...) {
                std::lock_guard<std::mutex> lock(errorMutex);
                if (!firstError) firstError = std::current_exception();
                failed.store(true);
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (int w = 0; w < workers; ++w) threads.emplace_back(run, w);
    for (std::thread& t : threads) t.join();
    if (firstError) std::rethrow_exception(firstError);
}

int resolveWorkers(int workers) {
    if (workers > 0) return workers;
    const unsigned hw = std::thread::hardware_concurrency();   // 0 if unknown
    return hw > 0 ? static_cast<int>(hw) : 1;
}

} // namespace azgo
```

Create `src/selfplay.cpp`:

```cpp
#include "azgo/selfplay.hpp"
#include <algorithm>
#include <cassert>
#include <limits>
#include "azgo/game.hpp"

namespace azgo {

float GameRecord::zAt(int ply) const {
    // The value target is always from the side to move's point of view.
    return toMoveAt(ply) == Color::Black ? resultForBlack : -resultForBlack;
}

GameRecord playSelfPlayGame(Evaluator& evaluator, const SelfPlayConfig& cfg, Rng& rng) {
    const int n = evaluator.boardSize();
    const int area = n * n;
    Game game(n, cfg.komi, cfg.maxMoves > 0 ? cfg.maxMoves : defaultMaxMoves(n));
    MctsConfig mcfg = cfg.mcts;
    mcfg.addNoise = true;   // self-play always explores at the root
    Mcts mcts(mcfg, evaluator);

    GameRecord rec;
    rec.boardSize = n;
    rec.komi = cfg.komi;
    while (!game.isTerminal()) {
        mcts.search(game, rng);

        // Position before the move ...
        const size_t offset = rec.stones.size();
        rec.stones.resize(offset + area);
        game.board().exportStones(rec.stones.data() + offset);
        // ... and the search's visit counts, the policy target pi.
        const std::vector<int> visits = mcts.rootVisits();
        std::vector<std::pair<int16_t, uint16_t>> sparse;
        for (int a = 0; a < static_cast<int>(visits.size()); ++a) {
            if (visits[a] == 0) continue;
            // Counts beyond 65535 (only with huge simulation budgets) are
            // clamped; pi is a ratio, so this barely changes it.
            const int count = std::min(visits[a], static_cast<int>(std::numeric_limits<uint16_t>::max()));
            sparse.emplace_back(static_cast<int16_t>(a), static_cast<uint16_t>(count));
        }
        rec.visits.push_back(std::move(sparse));

        // Sample in the opening for variety, then play the best move.
        const float temperature = game.moveCount() < cfg.temperatureMoves ? 1.0f : 0.0f;
        const int action = mcts.selectAction(temperature, rng);
        rec.moves.push_back(action);
        game.play(action);
        mcts.advance(action);   // reuse the subtree under the move just played
    }
    rec.resultForBlack = game.resultForBlack();
    return rec;
}

} // namespace azgo
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R "test_parallel|test_selfplay"`
Expected: `100% tests passed, 0 tests failed out of 2`. The executables print `test_parallel: all tests passed` and `test_selfplay: all tests passed`.
Then run `ctest --test-dir build --output-on-failure`. Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add include/azgo/parallel.hpp src/parallel.cpp include/azgo/selfplay.hpp src/selfplay.cpp tests/test_parallel.cpp tests/test_selfplay.cpp
git commit -F - <<'EOF'
Add the parallel game runner and self-play games

parallelFor hands out game indices from an atomic counter. Each game is
seeded from its index and stored by index, so the output does not
depend on the worker count. The test checks this with 1 and 3 workers.
A self-play game records each position, the root visit counts
(the policy target) and the final result, and z is derived per side to
move. An untrained network that passes at once gives a legal two-ply
game won by White on komi. That case is pinned down by a test.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 15: Replay buffer

**Files:**
- Create: `include/azgo/replay_buffer.hpp`, `src/replay_buffer.cpp`
- Test: `tests/test_replay_buffer.cpp`

**Interfaces:**
- Consumes: `GameRecord` (`positions`, `stonesAt`, `toMoveAt`, `zAt`, `visits`, `boardSize`) (Task 14), `encodePlanes(const std::vector<const uint8_t*>&, int n, Color, int H, int sym, float*)`, `numPlanes`, `transformAction`, `transformPoint`, `kNumSymmetries`, `Rng::below`, `BinaryWriter`/`BinaryReader` (`u32`, `u64`, `i32`, `f32`, `bytes`), `kMinBoardSize`/`kMaxBoardSize`, `Game` (tests).
- Produces: `TrainingBatch` and `ReplayBuffer` as in the contract. The training state (Task 20) and the trainer (Task 21) use them. The save layout is: u64 capacity, u64 totalAdded, u32 game count. Then, for each game: i32 boardSize, f32 komi, u32 move count, then i32 for each move, the raw stone bytes (positions × n²), for each ply a u32 entry count followed by one u32 `(action << 16) | count` per entry, and f32 resultForBlack.

A sample draws a global position index uniformly from `[0, positions())`, then a symmetry with `rng.below(8)`, in that order. The index is mapped to (game, ply) by binary search over the prefix sums of the game lengths, which are rebuilt after every add, evict or load.

- [ ] **Step 1: Write the failing test**

Create `tests/test_replay_buffer.cpp`:

```cpp
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "azgo/encode.hpp"
#include "azgo/game.hpp"
#include "azgo/replay_buffer.hpp"
#include "azgo/symmetry.hpp"

using namespace azgo;

static bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

// Plays `moves` on a 5x5 Game and records it as self-play would: the board
// before each move, and visit counts {move: 3, pass: 1} (just {pass: 4} when
// the move is a pass). The result comes from the game unless it is still
// running, in which case `result` is used.
static GameRecord makeRecord(const std::vector<int>& moves, float komi = 7.5f, float result = 1.0f) {
    const int n = 5, area = 25;
    Game game(n, komi, defaultMaxMoves(n));
    GameRecord rec;
    rec.boardSize = n;
    rec.komi = komi;
    for (int m : moves) {
        const size_t off = rec.stones.size();
        rec.stones.resize(off + area);
        game.board().exportStones(rec.stones.data() + off);
        if (m == area) rec.visits.push_back({{static_cast<int16_t>(area), 4}});
        else rec.visits.push_back({{static_cast<int16_t>(m), 3}, {static_cast<int16_t>(area), 1}});
        rec.moves.push_back(m);
        game.play(m);
    }
    rec.resultForBlack = game.isTerminal() ? game.resultForBlack() : result;
    return rec;
}

static GameRecord lengthGame(int positions, int tag) {
    // `tag` as the first move tells games apart; the rest are distinct points.
    std::vector<int> moves;
    for (int k = 0; k < positions; ++k) moves.push_back((tag + 5 * k) % 25);
    return makeRecord(moves);
}

static void test_add_evicts_oldest_games() {
    ReplayBuffer buf(10);
    buf.add(lengthGame(4, 0));
    buf.add(lengthGame(4, 1));
    assert(buf.games() == 2 && buf.positions() == 8);
    buf.add(lengthGame(4, 2));   // 12 > 10: the oldest game goes
    assert(buf.games() == 2);
    assert(buf.positions() == 8);
    assert(buf.game(0).moves[0] == 1 && buf.game(1).moves[0] == 2);
    assert(buf.totalPositionsAdded() == 12);   // evicted positions still count
    assert(buf.capacity() == 10);
}

static void test_oversized_game_is_kept_alone() {
    ReplayBuffer buf(3);
    buf.add(lengthGame(4, 0));
    assert(buf.games() == 1 && buf.positions() == 4);   // never evict the newest
    buf.add(lengthGame(2, 1));
    assert(buf.games() == 1 && buf.positions() == 2);
    assert(buf.game(0).moves[0] == 1);
    assert(buf.totalPositionsAdded() == 6);
}

static void test_set_capacity_shrinks_by_evicting() {
    ReplayBuffer buf(100);
    for (int t = 0; t < 3; ++t) buf.add(lengthGame(3, t));
    assert(buf.positions() == 9);
    buf.setCapacity(5);
    assert(buf.capacity() == 5);
    assert(buf.games() == 1 && buf.positions() == 3);
    assert(buf.game(0).moves[0] == 2);
    buf.setCapacity(50);   // growing evicts nothing
    assert(buf.games() == 1);
    assert(buf.totalPositionsAdded() == 9);
}

static void test_encode_sample_at_game_start() {
    ReplayBuffer buf(100);
    buf.add(makeRecord({6, 18, 12}));
    const int H = 2, planes = numPlanes(H), area = 25;
    std::vector<float> input(planes * area, 9.0f), pi(26, 9.0f);
    float z = 9.0f;
    buf.encodeSample(0, 0, H, 0, input.data(), pi.data(), &z);
    // Empty board, nothing before it: every plane is zero except "black to move".
    for (int p = 0; p < planes; ++p)
        for (int i = 0; i < area; ++i)
            assert(input[p * area + i] == (p == 2 * H ? 1.0f : 0.0f));
    assert(pi[6] == 0.75f && pi[25] == 0.25f);
    assert(z == buf.game(0).zAt(0));
}

static void test_encode_sample_history_planes() {
    // Black C4 (6), White E2 (18), Black C3 (12); positions at plies 0..2.
    ReplayBuffer buf(100);
    buf.add(makeRecord({6, 18, 12}));
    const int H = 2, area = 25;
    std::vector<float> input(numPlanes(H) * area), pi(26);
    float z;

    // Ply 2, Black to move: current stones B6, W18; one ply ago only B6.
    buf.encodeSample(0, 2, H, 0, input.data(), pi.data(), &z);
    auto plane = [&](int p, int point) { return input[p * area + point]; };
    assert(plane(0, 6) == 1.0f && plane(0, 18) == 0.0f);   // own (Black) now
    assert(plane(1, 18) == 1.0f && plane(1, 6) == 0.0f);   // opponent (White) now
    assert(plane(2, 6) == 1.0f && plane(2, 18) == 0.0f);   // own, ply 1
    for (int i = 0; i < area; ++i) assert(plane(3, i) == 0.0f);   // White had no stones at ply 1
    for (int i = 0; i < area; ++i) assert(plane(4, i) == 1.0f && plane(5, i) == 0.0f);
    // The second plane pair is exactly ply 1's stones.
    std::vector<float> ref(numPlanes(1) * area);
    encodePlanes({buf.game(0).stonesAt(1)}, 5, Color::Black, 1, 0, ref.data());
    for (int i = 0; i < 2 * area; ++i) assert(input[2 * area + i] == ref[i]);

    // Ply 1, White to move: "own" is White (no stones), opponent is Black.
    buf.encodeSample(0, 1, H, 0, input.data(), pi.data(), &z);
    assert(plane(0, 6) == 0.0f && plane(1, 6) == 1.0f);
    for (int i = 0; i < area; ++i) assert(plane(4, i) == 0.0f && plane(5, i) == 1.0f);
    assert(pi[18] == 0.75f && pi[25] == 0.25f);
}

static void test_symmetry_moves_planes_and_pi_together() {
    ReplayBuffer buf(100);
    buf.add(makeRecord({6, 18, 12}));
    const GameRecord& g = buf.game(0);
    const int H = 2, n = 5, area = 25;
    std::vector<float> input(numPlanes(H) * area), ref(numPlanes(H) * area), pi(26);
    float z;
    for (int sym = 0; sym < kNumSymmetries; ++sym) {
        buf.encodeSample(0, 2, H, sym, input.data(), pi.data(), &z);
        encodePlanes({g.stonesAt(2), g.stonesAt(1)}, n, Color::Black, H, sym, ref.data());
        assert(input == ref);
        assert(input[transformPoint(sym, 6, n)] == 1.0f);   // B6 lands where the symmetry sends it
        // Ply 2's move was C3 (12) with 3 visits, pass with 1.
        float total = 0.0f;
        for (int a = 0; a < 26; ++a) {
            float expected = 0.0f;
            if (a == transformAction(sym, 12, n)) expected = 0.75f;
            if (a == transformAction(sym, 25, n)) expected = 0.25f;
            assert(pi[a] == expected);
            total += pi[a];
        }
        assert(near(total, 1.0));
        assert(pi[25] == 0.25f);   // pass is fixed by every symmetry
    }
}

static void test_z_sign_follows_side_to_move() {
    ReplayBuffer buf(100);
    buf.add(makeRecord({6, 18, 12}, 7.5f, 1.0f));    // unfinished game, Black declared winner
    buf.add(makeRecord({6, 18, 12}, 7.5f, -1.0f));   // same moves, White declared winner
    std::vector<float> input(numPlanes(1) * 25), pi(26);
    float z;
    const float expected[2][3] = {{1, -1, 1}, {-1, 1, -1}};
    for (int g = 0; g < 2; ++g)
        for (int ply = 0; ply < 3; ++ply) {
            buf.encodeSample(g, ply, 1, 0, input.data(), pi.data(), &z);
            assert(z == expected[g][ply]);
        }
}

static void test_exact_tie_gives_zero_z_everywhere() {
    // Review Focus 2: integer komi 0 on an empty board, pass-pass: area score
    // 0 - komi 0 = 0, an exact tie. Neither side learns "win" or "loss".
    GameRecord tie = makeRecord({25, 25}, 0.0f);
    assert(tie.resultForBlack == 0.0f);
    ReplayBuffer buf(100);
    buf.add(tie);
    std::vector<float> input(numPlanes(2) * 25), pi(26);
    for (int ply = 0; ply < 2; ++ply) {
        float z = 9.0f;
        buf.encodeSample(0, ply, 2, 0, input.data(), pi.data(), &z);
        assert(z == 0.0f);
        assert(pi[25] == 1.0f);
    }
    TrainingBatch batch;
    Rng rng(3);
    buf.sample(32, 2, rng, batch);
    for (float z : batch.z) assert(z == 0.0f);
}

static void test_sample_shapes_and_pi_sums() {
    ReplayBuffer buf(100);
    buf.add(makeRecord({6, 18, 12, 25}));
    buf.add(makeRecord({25, 25}));
    const int H = 3, B = 40;
    TrainingBatch batch;
    Rng rng(4);
    buf.sample(B, H, rng, batch);
    assert(batch.size == B);
    assert(batch.input.size() == static_cast<size_t>(B) * numPlanes(H) * 25);
    assert(batch.pi.size() == static_cast<size_t>(B) * 26);
    assert(batch.z.size() == static_cast<size_t>(B));
    for (int b = 0; b < B; ++b) {
        double total = 0.0;
        for (int a = 0; a < 26; ++a) total += batch.pi[b * 26 + a];
        assert(near(total, 1.0));
        // Exactly one colour plane is all ones.
        const float* in = batch.input.data() + static_cast<size_t>(b) * numPlanes(H) * 25;
        assert(in[2 * H * 25] + in[(2 * H + 1) * 25] == 1.0f);
    }
}

static void test_sampling_is_uniform_over_positions() {
    // Game 0: one position with a tied result (z = 0). Game 1: three
    // positions of a decided game (z = +-1). Uniform over the 4 positions
    // means game 1 is drawn 3/4 of the time.
    ReplayBuffer buf(100);
    buf.add(makeRecord({25}, 0.0f, 0.0f));   // one Black pass, recorded as a tie
    buf.add(makeRecord({6, 18, 12}, 7.5f, -1.0f));
    assert(buf.positions() == 4);
    TrainingBatch batch;
    Rng rng(5);
    const int draws = 20000;
    buf.sample(draws, 1, rng, batch);
    int fromGame1 = 0;
    for (float z : batch.z) fromGame1 += (z != 0.0f) ? 1 : 0;
    assert(near(static_cast<double>(fromGame1) / draws, 0.75, 0.03));
}

static void test_save_load_round_trip() {
    ReplayBuffer buf(9);
    buf.add(lengthGame(4, 0));
    buf.add(makeRecord({6, 18, 12, 25}));
    buf.add(makeRecord({25, 25}, 0.0f));   // evicts the first game
    std::ostringstream os;
    BinaryWriter w(os);
    buf.save(w);
    const std::string bytes = os.str();

    std::istringstream is(bytes);
    BinaryReader r(is);
    ReplayBuffer back = ReplayBuffer::load(r);
    assert(back.capacity() == buf.capacity());
    assert(back.positions() == buf.positions());
    assert(back.games() == buf.games());
    assert(back.totalPositionsAdded() == buf.totalPositionsAdded());
    for (size_t g = 0; g < buf.games(); ++g) {
        const GameRecord& a = buf.game(g);
        const GameRecord& b = back.game(g);
        assert(a.boardSize == b.boardSize && a.komi == b.komi && a.moves == b.moves);
        assert(a.stones == b.stones && a.visits == b.visits && a.resultForBlack == b.resultForBlack);
    }
    TrainingBatch x, y;
    Rng r1(77), r2(77);
    buf.sample(64, 3, r1, x);
    back.sample(64, 3, r2, y);
    assert(std::memcmp(x.input.data(), y.input.data(), x.input.size() * sizeof(float)) == 0);
    assert(std::memcmp(x.pi.data(), y.pi.data(), x.pi.size() * sizeof(float)) == 0);
    assert(std::memcmp(x.z.data(), y.z.data(), x.z.size() * sizeof(float)) == 0);

    // A truncated stream is an error, not a crash.
    std::istringstream cut(bytes.substr(0, bytes.size() / 2));
    BinaryReader rc(cut);
    bool threw = false;
    try {
        ReplayBuffer::load(rc);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

int main() {
    test_add_evicts_oldest_games();
    test_oversized_game_is_kept_alone();
    test_set_capacity_shrinks_by_evicting();
    test_encode_sample_at_game_start();
    test_encode_sample_history_planes();
    test_symmetry_moves_planes_and_pi_together();
    test_z_sign_follows_side_to_move();
    test_exact_tie_gives_zero_z_everywhere();
    test_sample_shapes_and_pi_sums();
    test_sampling_is_uniform_over_positions();
    test_save_load_round_trip();
    std::printf("test_replay_buffer: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL — `fatal error: 'azgo/replay_buffer.hpp' file not found` in `tests/test_replay_buffer.cpp`.

- [ ] **Step 3: Write the header**

Create `include/azgo/replay_buffer.hpp`:

```cpp
// The replay buffer is the training set: the most recent self-play games,
// kept whole in a sliding window of about `capacity` positions. A training
// sample is one (game, ply); its history planes are rebuilt from the same
// game's earlier positions, so storing history costs nothing extra. Each
// sample is shown to the network under a random one of the board's 8
// symmetries, which multiplies the data for free (Go has no preferred side).
#pragma once
#include <cstddef>
#include <deque>
#include <vector>
#include "azgo/rng.hpp"
#include "azgo/selfplay.hpp"
#include "azgo/serialize.hpp"

namespace azgo {

struct TrainingBatch {
    int size = 0;
    std::vector<float> input;   // [size][planes][area]
    std::vector<float> pi;      // [size][area + 1]
    std::vector<float> z;       // [size]
};

// Sliding window of whole games. A sample is (game, ply); its history planes
// come from the same game's earlier positions, so history costs no storage.
class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t capacityPositions);
    void add(GameRecord game);            // then evicts oldest games while positions() > capacity,
                                          // never evicting the newest game
    void setCapacity(size_t capacityPositions);   // evicts as add() does
    size_t capacity() const { return capacity_; }
    size_t positions() const { return positions_; }
    size_t games() const { return games_.size(); }
    long long totalPositionsAdded() const { return totalAdded_; }
    const GameRecord& game(size_t i) const { return games_[i]; }   // 0 = oldest

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
    size_t positions_ = 0;
    long long totalAdded_ = 0;
    std::deque<GameRecord> games_;
    // Sampling index helpers: implementation-defined (Task 15).
    std::vector<size_t> firstPosition_;   // firstPosition_[g] = positions in games 0..g-1; back() = positions_
    void evictOldest();                   // drop oldest games while over capacity (keeps the newest)
    void rebuildIndex();                  // recompute firstPosition_ after add/evict/load
};

} // namespace azgo
```

- [ ] **Step 4: Write the implementation**

Create `src/replay_buffer.cpp`:

```cpp
#include "azgo/replay_buffer.hpp"
#include <algorithm>
#include <cassert>
#include <climits>
#include <stdexcept>
#include <string>
#include <utility>
#include "azgo/encode.hpp"
#include "azgo/symmetry.hpp"

namespace azgo {

ReplayBuffer::ReplayBuffer(size_t capacityPositions) : capacity_(capacityPositions) {
    assert(capacity_ >= 1);
    rebuildIndex();
}

void ReplayBuffer::add(GameRecord game) {
    const size_t area = static_cast<size_t>(game.boardSize) * game.boardSize;
    assert(game.positions() > 0);
    assert(game.stones.size() == static_cast<size_t>(game.positions()) * area);
    assert(game.visits.size() == static_cast<size_t>(game.positions()));
    assert(games_.empty() || games_.front().boardSize == game.boardSize);
    (void)area;
    positions_ += game.positions();
    totalAdded_ += game.positions();
    games_.push_back(std::move(game));
    evictOldest();
    rebuildIndex();
}

void ReplayBuffer::setCapacity(size_t capacityPositions) {
    assert(capacityPositions >= 1);
    capacity_ = capacityPositions;
    evictOldest();
    rebuildIndex();
}

void ReplayBuffer::evictOldest() {
    // Whole games leave the window, oldest first. The newest game always
    // stays, even if it alone is longer than the capacity.
    while (positions_ > capacity_ && games_.size() > 1) {
        positions_ -= games_.front().positions();
        games_.pop_front();
    }
}

void ReplayBuffer::rebuildIndex() {
    firstPosition_.assign(1, 0);
    for (const GameRecord& g : games_) firstPosition_.push_back(firstPosition_.back() + g.positions());
}

void ReplayBuffer::encodeSample(size_t gameIndex, int ply, int history, int sym,
                                float* input, float* pi, float* z) const {
    const GameRecord& g = games_[gameIndex];
    assert(ply >= 0 && ply < g.positions());
    const int n = g.boardSize;

    // History comes straight from the same game's earlier positions.
    std::vector<const uint8_t*> stones;
    for (int k = 0; k < history && ply - k >= 0; ++k) stones.push_back(g.stonesAt(ply - k));
    encodePlanes(stones, n, g.toMoveAt(ply), history, sym, input);

    // pi = visit counts normalised, moved to where the symmetry puts each point.
    std::fill(pi, pi + n * n + 1, 0.0f);
    int total = 0;
    for (const auto& av : g.visits[ply]) total += av.second;
    assert(total > 0);
    for (const auto& av : g.visits[ply])
        pi[transformAction(sym, av.first, n)] = static_cast<float>(av.second) / static_cast<float>(total);

    *z = g.zAt(ply);
}

void ReplayBuffer::sample(int B, int history, Rng& rng, TrainingBatch& out) const {
    assert(positions_ > 0 && B >= 1);
    assert(positions_ <= static_cast<size_t>(INT_MAX));
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
        const int sym = rng.below(kNumSymmetries);
        // Last game whose first position is <= global.
        const size_t g = static_cast<size_t>(
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
    for (const GameRecord& g : games_) {
        w.i32(g.boardSize);
        w.f32(g.komi);
        w.u32(static_cast<uint32_t>(g.moves.size()));
        for (int m : g.moves) w.i32(m);
        w.bytes(g.stones.data(), g.stones.size());
        for (const auto& ply : g.visits) {
            w.u32(static_cast<uint32_t>(ply.size()));
            for (const auto& av : ply)
                w.u32((static_cast<uint32_t>(static_cast<uint16_t>(av.first)) << 16) | av.second);
        }
        w.f32(g.resultForBlack);
    }
}

ReplayBuffer ReplayBuffer::load(BinaryReader& r) {
    const uint64_t capacity = r.u64();
    if (capacity == 0) throw std::runtime_error("replay buffer: capacity is 0");
    ReplayBuffer buf(static_cast<size_t>(capacity));
    const long long totalAdded = static_cast<long long>(r.u64());
    const uint32_t games = r.u32();
    for (uint32_t i = 0; i < games; ++i) {
        GameRecord g;
        g.boardSize = r.i32();
        if (g.boardSize < kMinBoardSize || g.boardSize > kMaxBoardSize)
            throw std::runtime_error("replay buffer: bad board size " + std::to_string(g.boardSize));
        const int area = g.boardSize * g.boardSize;
        g.komi = r.f32();
        // Sanity bounds keep a corrupt file from asking for gigabytes.
        const uint32_t moves = r.u32();
        if (moves == 0 || moves > 100000) throw std::runtime_error("replay buffer: bad move count");
        g.moves.resize(moves);
        for (int& m : g.moves) {
            m = r.i32();
            if (m < 0 || m > area) throw std::runtime_error("replay buffer: bad move");
        }
        g.stones.resize(static_cast<size_t>(moves) * area);
        r.bytes(g.stones.data(), g.stones.size());
        g.visits.resize(moves);
        for (auto& ply : g.visits) {
            const uint32_t k = r.u32();
            if (k == 0 || k > static_cast<uint32_t>(area + 1))
                throw std::runtime_error("replay buffer: bad visit list");
            ply.resize(k);
            for (auto& av : ply) {
                const uint32_t packed = r.u32();
                av.first = static_cast<int16_t>(packed >> 16);
                av.second = static_cast<uint16_t>(packed & 0xFFFFu);
                if (av.first < 0 || av.first > area || av.second == 0)
                    throw std::runtime_error("replay buffer: bad visit entry");
            }
        }
        g.resultForBlack = r.f32();
        // Straight into the deque: the saved window already respected the
        // capacity, and totalAdded is restored below rather than recounted.
        buf.positions_ += g.positions();
        buf.games_.push_back(std::move(g));
    }
    buf.totalAdded_ = totalAdded;
    buf.rebuildIndex();
    return buf;
}

} // namespace azgo
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_replay_buffer`
Expected: `100% tests passed, 0 tests failed out of 1`. The executable prints `test_replay_buffer: all tests passed`.
Then run `ctest --test-dir build --output-on-failure`. Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add include/azgo/replay_buffer.hpp src/replay_buffer.cpp tests/test_replay_buffer.cpp
git commit -F - <<'EOF'
Add the game-window replay buffer with symmetry sampling

Whole games are kept in a sliding window measured in positions, and
the oldest are evicted first. A sample is a (game, ply) pair whose
history planes come from the same game, so history costs no storage.
Sampling is uniform over positions and applies a random dihedral
symmetry to the planes and to pi together. An exact komi tie trains
towards z = 0 for both sides. The buffer serializes into the training
state, and loading rejects implausible counts instead of allocating
blindly.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 16: Config + preset files

**Files:**
- Create: `include/azgo/config.hpp`, `src/config.cpp`
- Create: `configs/5x5.cfg`, `configs/9x9.cfg`, `configs/19x19.cfg`
- Test: `tests/test_config.cpp`

**Interfaces:**
- Consumes: `NetConfig` (Task 7); `LrSchedule::parse(const std::string&)` throwing `std::invalid_argument` (Task 8); `int resolveWorkers(int)` (Task 14); `int defaultMaxMoves(int n)` (Task 3); `kMinBoardSize`, `kMaxBoardSize` (Task 2). The test uses the `AZGO_SOURCE_DIR` compile definition (repo root string, Task 1 CMake).
- Produces: `ConfigError`, `Config` (all fields and defaults as in contract), `Config::netConfig()`, `Config::effectiveMaxMoves()`, `Config::effectiveWorkers()`, `loadConfigFile`, `parseConfigText`, `applyOverride`, `configToString`, `architectureDiff`; the three preset files. Error format: `"<source>:<line>: <key>: <reason>"`, `"<source>:<line>: expected key = value"` for a line without `=`, and `"--set: <key>: <reason>"` from `applyOverride`.

The implementation is one static table `kKeys` of `{key, architecture, parse, print}` entries in contract key order. `parseConfigText`, `applyOverride`, `configToString` and `architectureDiff` all iterate it, so adding a setting is one table entry. Numbers are parsed strictly (the whole string must convert; `"12abc"`, `""`, `"1e999"`, `"nan"` and `"-1"` for the seed are rejected). Floats print with the shortest `%.{6..9}g` that parses back to exactly the same float, so `momentum = 0.9` rather than `0.899999976`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.cpp`:

```cpp
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "azgo/config.hpp"

using namespace azgo;

static bool contains(const std::string& s, const std::string& part) {
    return s.find(part) != std::string::npos;
}

// Parses text as file "t.cfg" and returns the ConfigError message.
static std::string errorOf(const std::string& text) {
    try {
        parseConfigText(text, "t.cfg");
    } catch (const ConfigError& e) {
        return e.what();
    }
    assert(false && "expected ConfigError");
    return "";
}

static std::string overrideErrorOf(const std::string& setting) {
    Config cfg;
    try {
        applyOverride(cfg, setting);
    } catch (const ConfigError& e) {
        return e.what();
    }
    assert(false && "expected ConfigError");
    return "";
}

static std::string preset(const std::string& name) {
    return std::string(AZGO_SOURCE_DIR) + "/configs/" + name;
}

static void test_defaults() {
    Config c;
    assert(c.seed == 1 && c.generations == 100 && c.workers == 0 && c.backend == "naive");
    assert(c.boardSize == 9 && c.komi == 7.5f && c.maxMoves == 0);
    assert(c.history == 8 && c.blocks == 6 && c.filters == 64 && c.valueHidden == 64);
    assert(c.gamesPerGeneration == 100 && c.simulations == 200 && c.mctsBatch == 8);
    assert(c.cPuct == 1.5f && c.dirichletScale == 10.0f && c.noiseFraction == 0.25f);
    assert(c.temperatureMoves == 20);
    assert(c.bufferPositions == 200000 && c.trainSteps == 200 && c.batchSize == 256);
    assert(c.lrSchedule == "0:0.02" && c.momentum == 0.9f && c.weightDecay == 1e-4f);
    assert(c.arenaEvery == 1 && c.arenaGames == 20 && c.arenaOpponents == 4);
    assert(c.arenaSimulations == 100 && c.arenaTemperatureMoves == 4);
    assert(!c.gating && c.gatingThreshold == 0.55f && c.sgfPerGeneration == 2);

    // Empty text changes nothing.
    assert(configToString(parseConfigText("", "empty")) == configToString(c));

    NetConfig net = c.netConfig();
    assert(net.boardSize == 9 && net.history == 8 && net.blocks == 6);
    assert(net.filters == 64 && net.valueHidden == 64);
    assert(c.effectiveMaxMoves() == 2 * 9 * 9);
    c.maxMoves = 50;
    assert(c.effectiveMaxMoves() == 50);
    assert(c.effectiveWorkers() >= 1);
    c.workers = 3;
    assert(c.effectiveWorkers() == 3);
}

static void test_presets_load() {
    Config c5 = loadConfigFile(preset("5x5.cfg"));
    assert(c5.boardSize == 5 && c5.blocks == 3 && c5.filters == 32 && c5.valueHidden == 32);
    assert(c5.lrSchedule == "0:0.02,40:0.002" && c5.arenaTemperatureMoves == 2);

    Config c9 = loadConfigFile(preset("9x9.cfg"));
    assert(c9.boardSize == 9 && c9.blocks == 6 && c9.filters == 64 && c9.valueHidden == 64);
    assert(c9.lrSchedule == "0:0.02,60:0.002" && c9.temperatureMoves == 20);

    Config c19 = loadConfigFile(preset("19x19.cfg"));
    assert(c19.boardSize == 19 && c19.blocks == 10 && c19.filters == 128);
    assert(c19.valueHidden == 256 && c19.simulations == 400 && c19.arenaGames == 10);
}

static void test_missing_file() {
    bool threw = false;
    try {
        loadConfigFile("no/such/file.cfg");
    } catch (const ConfigError& e) {
        threw = contains(e.what(), "no/such/file.cfg");
    }
    assert(threw);
}

static void test_round_trip() {
    std::vector<Config> configs = {Config{}, loadConfigFile(preset("5x5.cfg")),
                                   loadConfigFile(preset("9x9.cfg")),
                                   loadConfigFile(preset("19x19.cfg"))};
    // Unusual values too: large seed, negative komi, gating on.
    Config odd;
    odd.seed = 18446744073709551615ull;
    odd.komi = -0.5f;
    odd.gating = true;
    odd.weightDecay = 3e-5f;
    configs.push_back(odd);

    for (const Config& c : configs) {
        std::string text = configToString(c);
        Config back = parseConfigText(text, "roundtrip");
        assert(configToString(back) == text);
    }
    Config back = parseConfigText(configToString(odd), "roundtrip");
    assert(back.seed == odd.seed && back.komi == odd.komi && back.gating);
    assert(back.weightDecay == odd.weightDecay);   // bit-identical float
    assert(contains(configToString(odd), "gating = true\n"));
    assert(contains(configToString(Config{}), "board_size = 9\n"));
    // Floats print in their shortest exact form, not "0.899999976".
    assert(contains(configToString(Config{}), "momentum = 0.9\n"));
    assert(contains(configToString(Config{}), "weight_decay = 0.0001\n"));
}

static void test_overrides() {
    Config c;
    applyOverride(c, "simulations=50");
    assert(c.simulations == 50);
    applyOverride(c, " komi = 6.5 ");
    assert(c.komi == 6.5f);
    applyOverride(c, "gating=on");
    assert(c.gating);

    // Everything else is untouched.
    Config expected;
    expected.simulations = 50;
    expected.komi = 6.5f;
    expected.gating = true;
    assert(configToString(c) == configToString(expected));

    // Base config: text only changes the keys it mentions.
    Config base;
    base.filters = 16;
    Config merged = parseConfigText("blocks = 2\n", "t.cfg", base);
    assert(merged.filters == 16 && merged.blocks == 2);

    std::string msg = overrideErrorOf("simulations=0");
    assert(contains(msg, "--set: simulations: "));
    msg = overrideErrorOf("nonsense=1");
    assert(contains(msg, "--set: nonsense: unknown key"));
    msg = overrideErrorOf("simulations");
    assert(contains(msg, "--set: expected key = value"));
}

static void test_architecture_diff() {
    Config a, b;
    assert(architectureDiff(a, b).empty());
    b.filters = 32;
    b.history = 4;
    b.simulations = 50;        // tunable: not reported
    b.lrSchedule = "0:0.1";    // tunable: not reported
    std::vector<std::string> diff = architectureDiff(a, b);
    assert((diff == std::vector<std::string>{"history", "filters"}));   // table order
    b = a;
    b.boardSize = 5;
    b.valueHidden = 8;
    b.blocks = 0;
    diff = architectureDiff(a, b);
    assert((diff == std::vector<std::string>{"board_size", "blocks", "value_hidden"}));
}

static void test_errors_name_key_and_line() {
    std::string msg = errorOf("komi = 7.5\nfoo = 1\n");
    assert(contains(msg, "t.cfg:2") && contains(msg, "foo") && contains(msg, "unknown key"));

    msg = errorOf("board_size = 4");
    assert(contains(msg, "t.cfg:1: board_size: ") && contains(msg, "[5, 19]"));
    msg = errorOf("board_size = 20");
    assert(contains(msg, "t.cfg:1: board_size: "));

    msg = errorOf("\n\nsimulations = 12abc");
    assert(contains(msg, "t.cfg:3: simulations: ") && contains(msg, "12abc"));
    msg = errorOf("simulations =");
    assert(contains(msg, "t.cfg:1: simulations: "));
    msg = errorOf("komi = 1e999");
    assert(contains(msg, "t.cfg:1: komi: "));
    msg = errorOf("komi = nan");
    assert(contains(msg, "t.cfg:1: komi: "));
    msg = errorOf("seed = -1");
    assert(contains(msg, "t.cfg:1: seed: "));
    msg = errorOf("c_puct = 0");   // lower bound excluded
    assert(contains(msg, "t.cfg:1: c_puct: "));
    msg = errorOf("momentum = 1"); // upper bound excluded
    assert(contains(msg, "t.cfg:1: momentum: "));

    msg = errorOf("backend = gpu");
    assert(contains(msg, "t.cfg:1: backend: ") && contains(msg, "naive|blas"));
    msg = errorOf("lr_schedule = 5:0.1");   // first entry must be at generation 0
    assert(contains(msg, "t.cfg:1: lr_schedule: "));
    msg = errorOf("gating = maybe");
    assert(contains(msg, "t.cfg:1: gating: "));

    msg = errorOf("komi = 7.5\nsimulations 50\n");
    assert(contains(msg, "t.cfg:2: expected key = value"));
}

// Review Focus 5: hand-edited files.
static void test_hand_edited_files() {
    std::string text =
        "# my experiment\r\n"
        "\r\n"
        "board_size = 5   # small board\r\n"
        "   komi=6.5\r\n"
        "\t\r\n"
        "lr_schedule =  0:0.05,10:0.005  \r\n"
        "gating\t=\ton # tabs too\r\n";
    Config c = parseConfigText(text, "edited.cfg");
    assert(c.boardSize == 5);
    assert(c.komi == 6.5f);
    assert(c.lrSchedule == "0:0.05,10:0.005");
    assert(c.gating);

    // A key repeated in one file is an error that names the second line.
    std::string msg = errorOf("simulations = 10\r\nkomi = 7\r\nsimulations = 20\r\n");
    assert(contains(msg, "t.cfg:3: simulations: ") && contains(msg, "duplicate"));

    // Colors alternate in the arena, so an odd game count is rejected.
    msg = errorOf("arena_games = 7");
    assert(contains(msg, "t.cfg:1: arena_games: ") && contains(msg, "even"));
    assert(parseConfigText("arena_games = 8", "t.cfg").arenaGames == 8);
}

int main() {
    test_defaults();
    test_presets_load();
    test_missing_file();
    test_round_trip();
    test_overrides();
    test_architecture_diff();
    test_errors_name_key_and_line();
    test_hand_edited_files();
    std::printf("test_config: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL, compile error in `tests/test_config.cpp`: `'azgo/config.hpp' file not found`.

- [ ] **Step 3: Write the header**

Create `include/azgo/config.hpp`:

```cpp
// Run configuration. Every knob of a training run lives in one plain-text
// "key = value" file so a run can be reproduced from its config.cfg alone.
// One table in config.cpp lists each key with its parser and printer, so the
// parser, the printer and the architecture check can never disagree.
// Architecture keys fix the network's tensor shapes and cannot change on
// resume; every other key is tunable between generations.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/network.hpp"

namespace azgo {

struct ConfigError : std::runtime_error { using std::runtime_error::runtime_error; };

struct Config {
    // run
    uint64_t seed = 1;
    int generations = 100;
    int workers = 0;                   // 0 = hardware concurrency
    std::string backend = "naive";     // naive | blas
    // game
    int boardSize = 9;
    float komi = 7.5f;
    int maxMoves = 0;                  // 0 = 2 * n * n
    // network (architecture keys)
    int history = 8;
    int blocks = 6;
    int filters = 64;
    int valueHidden = 64;
    // self-play and search
    int gamesPerGeneration = 100;
    int simulations = 200;
    int mctsBatch = 8;
    float cPuct = 1.5f;
    float dirichletScale = 10.0f;
    float noiseFraction = 0.25f;
    int temperatureMoves = 20;
    // training
    int bufferPositions = 200000;
    int trainSteps = 200;
    int batchSize = 256;
    std::string lrSchedule = "0:0.02";
    float momentum = 0.9f;
    float weightDecay = 1e-4f;
    // arena
    int arenaEvery = 1;                // 0 = never
    int arenaGames = 20;               // per opponent, even
    int arenaOpponents = 4;            // most recent ladder entries (generation 0 always added)
    int arenaSimulations = 100;
    int arenaTemperatureMoves = 4;
    bool gating = false;
    float gatingThreshold = 0.55f;
    // output
    int sgfPerGeneration = 2;

    NetConfig netConfig() const;
    int effectiveMaxMoves() const;     // maxMoves or 2 * n * n
    int effectiveWorkers() const;      // resolveWorkers(workers)
};

// key = value per line; '#' starts a comment (anywhere); blank lines ignored;
// surrounding whitespace and a trailing '\r' are stripped. Unknown key,
// malformed or out-of-range value, or a key repeated in one file ->
// ConfigError("<file>:<line>: <key>: <reason>").
Config loadConfigFile(const std::string& path, const Config& base = Config{});
Config parseConfigText(const std::string& text, const std::string& sourceName,
                       const Config& base = Config{});
// "key=value" (whitespace around '=' allowed). Same validation.
void applyOverride(Config& cfg, const std::string& keyEqualsValue);

// Every key, one "key = value" line each, in table order. Round-trips
// through parseConfigText exactly.
std::string configToString(const Config& cfg);

// Keys that fix the network's shape: board_size, history, blocks, filters, value_hidden.
std::vector<std::string> architectureDiff(const Config& a, const Config& b);

} // namespace azgo
```

- [ ] **Step 4: Write the implementation**

Create `src/config.cpp`:

```cpp
#include "azgo/config.hpp"
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include "azgo/game.hpp"
#include "azgo/optimizer.hpp"
#include "azgo/parallel.hpp"

namespace azgo {

namespace {

std::string trim(const std::string& s) {
    const char* ws = " \t\r\n\f\v";
    size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

std::string quoted(const std::string& v) { return "'" + v + "'"; }

// ---- strict value parsers: return "" on success, otherwise the reason ----

// The WHOLE string must be a base-10 integer in [lo, hi]. strtoll alone would
// accept "12abc" (stopping at 'a') and silently clamp on overflow.
std::string parseInt(const std::string& v, long long lo, long long hi, int& out) {
    std::string reason = "expected an integer in [" + std::to_string(lo) + ", " +
                         std::to_string(hi) + "], got " + quoted(v);
    if (v.empty()) return reason;
    errno = 0;
    char* end = nullptr;
    long long x = std::strtoll(v.c_str(), &end, 10);
    if (errno == ERANGE || *end != '\0' || x < lo || x > hi) return reason;
    out = static_cast<int>(x);
    return "";
}

std::string parseU64(const std::string& v, uint64_t& out) {
    std::string reason = "expected an unsigned 64-bit integer, got " + quoted(v);
    // strtoull accepts a leading '-' and wraps it around; require a digit first.
    if (v.empty() || v[0] < '0' || v[0] > '9') return reason;
    errno = 0;
    char* end = nullptr;
    unsigned long long x = std::strtoull(v.c_str(), &end, 10);
    if (errno == ERANGE || *end != '\0') return reason;
    out = static_cast<uint64_t>(x);
    return "";
}

// A finite number in the interval lo..hi; loOpen / hiOpen exclude the bound.
std::string parseFloat(const std::string& v, double lo, double hi, float& out,
                       bool loOpen = false, bool hiOpen = false) {
    char range[96];
    std::snprintf(range, sizeof range, "%c%g, %g%c", loOpen ? '(' : '[', lo, hi,
                  hiOpen ? ')' : ']');
    std::string reason = std::string("expected a number in ") + range + ", got " + quoted(v);
    if (v.empty()) return reason;
    errno = 0;
    char* end = nullptr;
    double x = std::strtod(v.c_str(), &end);
    // ERANGE catches "1e999"; isfinite catches "nan" and "inf".
    if (errno == ERANGE || *end != '\0' || !std::isfinite(x)) return reason;
    if (x < lo || x > hi || (loOpen && x == lo) || (hiOpen && x == hi)) return reason;
    out = static_cast<float>(x);
    return "";
}

std::string parseBool(const std::string& v, bool& out) {
    if (v == "true" || v == "on" || v == "1") { out = true; return ""; }
    if (v == "false" || v == "off" || v == "0") { out = false; return ""; }
    return "expected true|false|on|off|1|0, got " + quoted(v);
}

// The shortest "%.Ng" (N = 6..9) that reads back as exactly the same float:
// 0.9f prints as "0.9", not "0.899999976". Nine digits always round-trip.
std::string formatFloat(float x) {
    char buf[32];
    for (int digits = 6; digits <= 9; ++digits) {
        std::snprintf(buf, sizeof buf, "%.*g", digits, static_cast<double>(x));
        if (std::strtof(buf, nullptr) == x) break;
    }
    return buf;
}

// Table helpers, instantiated per field so each table line stays short.
template <int Config::*Field, long long Lo, long long Hi>
std::string intKey(Config& c, const std::string& v) { return parseInt(v, Lo, Hi, c.*Field); }
template <int Config::*Field>
std::string printInt(const Config& c) { return std::to_string(c.*Field); }
template <float Config::*Field>
std::string printFloat(const Config& c) { return formatFloat(c.*Field); }

struct KeyEntry {
    const char* key;
    bool architecture;   // fixes the network's shape: may not change on resume
    std::string (*parse)(Config& cfg, const std::string& value);   // "" or reason
    std::string (*print)(const Config& cfg);
};

// THE list of settings. Order here is the order of configToString.
const KeyEntry kKeys[] = {
    // run
    {"seed", false, [](Config& c, const std::string& v) { return parseU64(v, c.seed); },
     [](const Config& c) { return std::to_string(c.seed); }},
    {"generations", false, intKey<&Config::generations, 1, 1000000>, printInt<&Config::generations>},
    {"workers", false, intKey<&Config::workers, 0, 256>, printInt<&Config::workers>},
    {"backend", false,
     [](Config& c, const std::string& v) -> std::string {
         if (v != "naive" && v != "blas") return "expected naive|blas, got " + quoted(v);
         c.backend = v;
         return "";
     },
     [](const Config& c) { return c.backend; }},
    // game
    {"board_size", true, intKey<&Config::boardSize, kMinBoardSize, kMaxBoardSize>, printInt<&Config::boardSize>},
    {"komi", false, [](Config& c, const std::string& v) { return parseFloat(v, -500, 500, c.komi); },
     printFloat<&Config::komi>},
    {"max_moves", false, intKey<&Config::maxMoves, 0, 100000>, printInt<&Config::maxMoves>},
    // network
    {"history", true, intKey<&Config::history, 1, 16>, printInt<&Config::history>},
    {"blocks", true, intKey<&Config::blocks, 0, 40>, printInt<&Config::blocks>},
    {"filters", true, intKey<&Config::filters, 1, 512>, printInt<&Config::filters>},
    {"value_hidden", true, intKey<&Config::valueHidden, 1, 1024>, printInt<&Config::valueHidden>},
    // self-play and search
    {"games_per_generation", false, intKey<&Config::gamesPerGeneration, 1, 1000000>,
     printInt<&Config::gamesPerGeneration>},
    {"simulations", false, intKey<&Config::simulations, 1, 100000>, printInt<&Config::simulations>},
    {"mcts_batch", false, intKey<&Config::mctsBatch, 1, 256>, printInt<&Config::mctsBatch>},
    {"c_puct", false, [](Config& c, const std::string& v) { return parseFloat(v, 0, 100, c.cPuct, true); },
     printFloat<&Config::cPuct>},
    {"dirichlet_scale", false,
     [](Config& c, const std::string& v) { return parseFloat(v, 0, 1000, c.dirichletScale); },
     printFloat<&Config::dirichletScale>},
    {"noise_fraction", false,
     [](Config& c, const std::string& v) { return parseFloat(v, 0, 1, c.noiseFraction); },
     printFloat<&Config::noiseFraction>},
    {"temperature_moves", false, intKey<&Config::temperatureMoves, 0, 100000>,
     printInt<&Config::temperatureMoves>},
    // training
    {"buffer_positions", false, intKey<&Config::bufferPositions, 1, 100000000>,
     printInt<&Config::bufferPositions>},
    {"train_steps", false, intKey<&Config::trainSteps, 0, 1000000>, printInt<&Config::trainSteps>},
    {"batch_size", false, intKey<&Config::batchSize, 1, 4096>, printInt<&Config::batchSize>},
    {"lr_schedule", false,
     [](Config& c, const std::string& v) -> std::string {
         try {
             LrSchedule::parse(v);   // validate only; the text is stored as written
         } catch (const std::invalid_argument& e) {
             return std::string("invalid schedule ") + quoted(v) + ": " + e.what();
         }
         c.lrSchedule = v;
         return "";
     },
     [](const Config& c) { return c.lrSchedule; }},
    {"momentum", false,
     [](Config& c, const std::string& v) { return parseFloat(v, 0, 1, c.momentum, false, true); },
     printFloat<&Config::momentum>},
    {"weight_decay", false,
     [](Config& c, const std::string& v) { return parseFloat(v, 0, 1, c.weightDecay); },
     printFloat<&Config::weightDecay>},
    // arena
    {"arena_every", false, intKey<&Config::arenaEvery, 0, 1000000>, printInt<&Config::arenaEvery>},
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
    {"arena_simulations", false, intKey<&Config::arenaSimulations, 1, 100000>,
     printInt<&Config::arenaSimulations>},
    {"arena_temperature_moves", false, intKey<&Config::arenaTemperatureMoves, 0, 100000>,
     printInt<&Config::arenaTemperatureMoves>},
    {"gating", false, [](Config& c, const std::string& v) { return parseBool(v, c.gating); },
     [](const Config& c) { return std::string(c.gating ? "true" : "false"); }},
    {"gating_threshold", false,
     [](Config& c, const std::string& v) { return parseFloat(v, 0.5, 1, c.gatingThreshold); },
     printFloat<&Config::gatingThreshold>},
    // output
    {"sgf_per_generation", false, intKey<&Config::sgfPerGeneration, 0, 1000>,
     printInt<&Config::sgfPerGeneration>},
};

const KeyEntry* findKey(const std::string& key) {
    for (const KeyEntry& e : kKeys)
        if (key == e.key) return &e;
    return nullptr;
}

// "file:12" for file text, "--set" for command-line overrides (line 0).
std::string location(const std::string& source, int line) {
    return line > 0 ? source + ":" + std::to_string(line) : source;
}

// Applies one "key = value" line (comment already stripped, non-blank).
// firstLine (may be null) remembers where each key was set, to catch repeats.
void applyLine(Config& cfg, const std::string& text, const std::string& source, int line,
               std::map<std::string, int>* firstLine) {
    size_t eq = text.find('=');
    std::string key = eq == std::string::npos ? "" : trim(text.substr(0, eq));
    if (key.empty()) throw ConfigError(location(source, line) + ": expected key = value");
    std::string value = trim(text.substr(eq + 1));
    std::string where = location(source, line) + ": " + key + ": ";

    const KeyEntry* entry = findKey(key);
    if (!entry) throw ConfigError(where + "unknown key");
    if (firstLine) {
        auto it = firstLine->find(key);
        if (it != firstLine->end())
            throw ConfigError(where + "duplicate key (first set on line " +
                              std::to_string(it->second) + ")");
        (*firstLine)[key] = line;
    }
    std::string reason = entry->parse(cfg, value);
    if (!reason.empty()) throw ConfigError(where + reason);
}

} // namespace

NetConfig Config::netConfig() const {
    NetConfig net;
    net.boardSize = boardSize;
    net.history = history;
    net.blocks = blocks;
    net.filters = filters;
    net.valueHidden = valueHidden;
    return net;
}

int Config::effectiveMaxMoves() const {
    return maxMoves > 0 ? maxMoves : defaultMaxMoves(boardSize);
}

int Config::effectiveWorkers() const { return resolveWorkers(workers); }

Config parseConfigText(const std::string& text, const std::string& sourceName, const Config& base) {
    Config cfg = base;
    std::map<std::string, int> firstLine;
    std::istringstream in(text);
    std::string raw;
    int lineNo = 0;
    while (std::getline(in, raw)) {
        ++lineNo;
        size_t hash = raw.find('#');
        if (hash != std::string::npos) raw.erase(hash);
        std::string line = trim(raw);   // also strips the '\r' of CRLF files
        if (line.empty()) continue;
        applyLine(cfg, line, sourceName, lineNo, &firstLine);
    }
    return cfg;
}

Config loadConfigFile(const std::string& path, const Config& base) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw ConfigError(path + ": cannot open config file");
    std::ostringstream text;
    text << in.rdbuf();
    return parseConfigText(text.str(), path, base);
}

void applyOverride(Config& cfg, const std::string& keyEqualsValue) {
    applyLine(cfg, trim(keyEqualsValue), "--set", 0, nullptr);
}

std::string configToString(const Config& cfg) {
    std::string out;
    for (const KeyEntry& e : kKeys) out += std::string(e.key) + " = " + e.print(cfg) + "\n";
    return out;
}

std::vector<std::string> architectureDiff(const Config& a, const Config& b) {
    std::vector<std::string> keys;
    for (const KeyEntry& e : kKeys)
        if (e.architecture && e.print(a) != e.print(b)) keys.push_back(e.key);
    return keys;
}

} // namespace azgo
```

- [ ] **Step 5: Write the preset files**

Create `configs/5x5.cfg`:

```ini
# 5x5 preset: a small board where a full run on a laptop CPU shows a
# clearly rising Elo curve within hours.
#
# This file is a preset, not a fixed contract: any key can be overridden on
# the command line with --set key=value (e.g. --set simulations=100). The
# values are tuned in Task 23 from bench measurements and measured runs.

# run: reproducibility seed, how long to train, threads (0 = all cores), math backend
seed = 1
generations = 60
workers = 0
backend = naive

# game: board, komi (Tromp-Taylor area scoring), move cap (0 = 2 * n * n)
board_size = 5
komi = 7.5
max_moves = 0

# network (architecture keys; cannot change when resuming a run)
history = 8
blocks = 3
filters = 32
value_hidden = 32

# self-play and search: games per generation, MCTS budget, root noise, opening temperature
games_per_generation = 200
simulations = 200
mcts_batch = 8
c_puct = 1.5
dirichlet_scale = 10
noise_fraction = 0.25
temperature_moves = 8

# training: replay window, SGD steps per generation, learning rate by generation
buffer_positions = 100000
train_steps = 200
batch_size = 128
lr_schedule = 0:0.02,40:0.002
momentum = 0.9
weight_decay = 0.0001

# arena: how often, how many games per opponent (even), ladder size, search budget, gating
arena_every = 1
arena_games = 20
arena_opponents = 4
arena_simulations = 100
arena_temperature_moves = 2
gating = false
gating_threshold = 0.55

# output: self-play games saved as SGF per generation
sgf_per_generation = 2
```

Create `configs/9x9.cfg`:

```ini
# 9x9 preset: the main target. Big enough for real Go shape (life and death,
# ko fights), small enough to train on a CPU over a day or two.
#
# This file is a preset, not a fixed contract: any key can be overridden on
# the command line with --set key=value (e.g. --set simulations=100). The
# values are tuned in Task 23 from bench measurements and measured runs.

# run: reproducibility seed, how long to train, threads (0 = all cores), math backend
seed = 1
generations = 100
workers = 0
backend = naive

# game: board, komi (Tromp-Taylor area scoring), move cap (0 = 2 * n * n)
board_size = 9
komi = 7.5
max_moves = 0

# network (architecture keys; cannot change when resuming a run)
history = 8
blocks = 6
filters = 64
value_hidden = 64

# self-play and search: games per generation, MCTS budget, root noise, opening temperature
games_per_generation = 100
simulations = 200
mcts_batch = 8
c_puct = 1.5
dirichlet_scale = 10
noise_fraction = 0.25
temperature_moves = 20

# training: replay window, SGD steps per generation, learning rate by generation
buffer_positions = 200000
train_steps = 200
batch_size = 256
lr_schedule = 0:0.02,60:0.002
momentum = 0.9
weight_decay = 0.0001

# arena: how often, how many games per opponent (even), ladder size, search budget, gating
arena_every = 1
arena_games = 20
arena_opponents = 4
arena_simulations = 100
arena_temperature_moves = 4
gating = false
gating_threshold = 0.55

# output: self-play games saved as SGF per generation
sgf_per_generation = 2
```

Create `configs/19x19.cfg`:

```ini
# 19x19 preset: the full board. The code supports it end to end, but a CPU
# can afford only a short demonstration run; expect slow, modest progress.
#
# This file is a preset, not a fixed contract: any key can be overridden on
# the command line with --set key=value (e.g. --set simulations=100). The
# values are tuned in Task 23 from bench measurements and measured runs.

# run: reproducibility seed, how long to train, threads (0 = all cores), math backend
seed = 1
generations = 20
workers = 0
backend = naive

# game: board, komi (Tromp-Taylor area scoring), move cap (0 = 2 * n * n)
board_size = 19
komi = 7.5
max_moves = 0

# network (architecture keys; cannot change when resuming a run)
history = 8
blocks = 10
filters = 128
value_hidden = 256

# self-play and search: games per generation, MCTS budget, root noise, opening temperature
games_per_generation = 20
simulations = 400
mcts_batch = 8
c_puct = 1.5
dirichlet_scale = 10
noise_fraction = 0.25
temperature_moves = 30

# training: replay window, SGD steps per generation, learning rate by generation
buffer_positions = 500000
train_steps = 100
batch_size = 256
lr_schedule = 0:0.02
momentum = 0.9
weight_decay = 0.0001

# arena: how often, how many games per opponent (even), ladder size, search budget, gating
arena_every = 1
arena_games = 10
arena_opponents = 2
arena_simulations = 100
arena_temperature_moves = 4
gating = false
gating_threshold = 0.55

# output: self-play games saved as SGF per generation
sgf_per_generation = 2
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_config`
Expected: `100% tests passed, 0 tests failed out of 1` (the test prints `test_config: all tests passed`).

- [ ] **Step 7: Commit**

```bash
git add include/azgo/config.hpp src/config.cpp configs/5x5.cfg configs/9x9.cfg configs/19x19.cfg tests/test_config.cpp
git commit -F - <<'EOF'
Add table-driven config parser and board-size presets

A run must be reproducible from its config.cfg alone, and hand-edited
files are the main input users touch. One table of key, parser and
printer keeps parsing, printing and the resume architecture check in
agreement. Parsing is strict: CRLF, comments and spacing are accepted,
but unknown or repeated keys, trailing garbage in numbers, out-of-range
values and odd arena game counts fail with file:line and the key.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 17: Elo

**Files:**
- Create: `include/azgo/elo.hpp`, `src/elo.cpp`
- Test: `tests/test_elo.cpp`

**Interfaces:**
- Consumes: nothing from earlier tasks (standard library only).
- Produces: `ResultsTable` (`Pair`, `addGame`, `addMatch`, `pairs`, `players`, `empty`, `toCsv`, `fromCsv`), `fitBradleyTerry(const ResultsTable&, int anchor = 0)`, `wilsonInterval(double score, int games, double z = 1.96)`. CSV format: header `a,b,games,score_a`, rows sorted by `(a, b)`, score printed with `%.9g`. `fromCsv` throws `std::runtime_error("results csv line N: <reason>")`.

Fitting method: BFS from the anchor finds the players that can be rated. Every played pair gets one virtual drawn game (`n_ij + 1`, `s_ij + 0.5`). MM sweeps `γ_i ← W_i / Σ_j n_ij / (γ_i + γ_j)` run until the largest `|log γ|` change is below 1e-10 (at most 100000 sweeps), with a rescale to `γ_anchor = 1` after each sweep. Elo = `400 · log10 γ`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_elo.cpp`:

```cpp
#include <cassert>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include "azgo/elo.hpp"

using namespace azgo;

static bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

static double expectedScore(double eloA, double eloB) {
    return 1.0 / (1.0 + std::pow(10.0, (eloB - eloA) / 400.0));
}

static void test_add_normalizes_order() {
    ResultsTable x, y;
    x.addGame(5, 2, 1.0);   // 5 beat 2
    y.addGame(2, 5, 0.0);   // 2 lost to 5
    auto px = x.pairs(), py = y.pairs();
    assert(px.size() == 1 && py.size() == 1);
    assert(px[0].a == 2 && px[0].b == 5 && px[0].games == 1 && px[0].scoreA == 0.0);
    assert(py[0].a == 2 && py[0].b == 5 && py[0].games == 1 && py[0].scoreA == 0.0);

    x.addGame(2, 5, 0.5);
    x.addMatch(5, 2, 10, 7.0);   // 5 scored 7 of 10, so 2 scored 3
    auto p = x.pairs();
    assert(p.size() == 1 && p[0].games == 12 && near(p[0].scoreA, 3.5, 1e-12));
    x.addGame(0, 9, 1.0);
    assert((x.players() == std::vector<int>{0, 2, 5, 9}));
    p = x.pairs();
    assert(p[0].a == 0 && p[0].b == 9 && p[1].a == 2 && p[1].b == 5);   // sorted
    assert(!x.empty() && ResultsTable().empty());
}

static void test_equal_results_equal_ratings() {
    ResultsTable t;
    t.addMatch(0, 1, 10, 5.0);
    auto elo = fitBradleyTerry(t);
    assert(elo.size() == 2);
    assert(near(elo.at(0), 0.0, 1e-9) && near(elo.at(1), 0.0, 1e-9));
}

static void test_recovers_known_ladder() {
    // True ratings 0, 200, 400; every pair plays 100000 games scored exactly
    // at the Bradley-Terry expectation.
    std::map<int, double> truth{{0, 0.0}, {1, 200.0}, {2, 400.0}};
    ResultsTable t;
    const int games = 100000;
    for (int a = 0; a < 3; ++a)
        for (int b = a + 1; b < 3; ++b)
            t.addMatch(a, b, games, games * expectedScore(truth[a], truth[b]));
    auto elo = fitBradleyTerry(t, 0);
    assert(elo.size() == 3);
    assert(elo.at(0) == 0.0);   // anchor pinned
    for (int p = 0; p < 3; ++p) assert(near(elo.at(p), truth[p], 2.0));

    // Order of recording does not matter.
    ResultsTable r;
    for (int a = 2; a >= 0; --a)
        for (int b = 2; b > a; --b)
            r.addMatch(b, a, games, games * expectedScore(truth[b], truth[a]));
    auto elo2 = fitBradleyTerry(r, 0);
    for (int p = 0; p < 3; ++p) assert(near(elo.at(p), elo2.at(p), 1e-6));
}

static void test_perfect_record_is_finite() {
    ResultsTable t;
    t.addMatch(0, 1, 20, 0.0);   // 1 won all 20
    auto elo = fitBradleyTerry(t);
    // With the virtual draw: 1 scored 20.5 of 21 -> gamma = 41 -> ~645 Elo.
    assert(std::isfinite(elo.at(1)) && elo.at(1) > 400.0);
    assert(near(elo.at(1), 400.0 * std::log10(41.0), 1e-6));
}

static void test_disconnected_players_omitted() {
    ResultsTable t;
    t.addMatch(0, 1, 10, 4.0);
    t.addMatch(1, 2, 10, 6.0);
    t.addMatch(5, 6, 10, 3.0);   // no path to generation 0
    auto elo = fitBradleyTerry(t);
    assert(elo.size() == 3 && elo.count(0) && elo.count(1) && elo.count(2));
    assert(!elo.count(5) && !elo.count(6));

    ResultsTable only;
    only.addMatch(3, 4, 2, 1.0);   // anchor never played
    auto lone = fitBradleyTerry(only, 0);
    assert(lone.size() == 1 && lone.at(0) == 0.0);
    assert(fitBradleyTerry(ResultsTable{}).size() == 1);
}

static void test_csv_round_trip() {
    ResultsTable t;
    t.addMatch(0, 1, 20, 12.5);
    t.addMatch(3, 1, 40, 21.0);
    t.addGame(0, 3, 0.5);
    std::string csv = t.toCsv();
    assert(csv == "a,b,games,score_a\n0,1,20,12.5\n0,3,1,0.5\n1,3,40,19\n");
    ResultsTable back = ResultsTable::fromCsv(csv);
    assert(back.toCsv() == csv);
    auto p = back.pairs();
    assert(p.size() == 3 && p[2].a == 1 && p[2].b == 3 && p[2].games == 40 && p[2].scoreA == 19.0);
    // CRLF line endings and an empty table are fine.
    assert(ResultsTable::fromCsv("a,b,games,score_a\r\n0,1,2,1\r\n").pairs().size() == 1);
    assert(ResultsTable::fromCsv("a,b,games,score_a\n").empty());
}

static bool csvThrows(const std::string& csv, const std::string& expectInMessage) {
    try {
        ResultsTable::fromCsv(csv);
    } catch (const std::runtime_error& e) {
        return std::string(e.what()).find(expectInMessage) != std::string::npos;
    }
    return false;
}

static void test_csv_rejects_bad_input() {
    assert(csvThrows("", "line 1"));
    assert(csvThrows("a,b,games,score\n0,1,2,1\n", "line 1"));
    assert(csvThrows("a,b,games,score_a\n0,1,2\n", "line 2"));
    assert(csvThrows("a,b,games,score_a\n0,1,2,1\n0,x,2,1\n", "line 3"));
    assert(csvThrows("a,b,games,score_a\n0,1,2,3\n", "line 2"));    // score > games
    assert(csvThrows("a,b,games,score_a\n1,1,2,1\n", "line 2"));    // a == b
    assert(csvThrows("a,b,games,score_a\n0,1,0,0\n", "line 2"));    // no games
}

static void test_wilson_interval() {
    auto ci = wilsonInterval(20, 40);
    assert(near(ci.first, 0.352, 1e-3) && near(ci.second, 0.648, 1e-3));
    ci = wilsonInterval(40, 40);
    assert(near(ci.second, 1.0, 1e-12) && ci.second <= 1.0);
    assert(near(ci.first, 0.912, 1e-3));
    ci = wilsonInterval(0, 40);
    assert(ci.first == 0.0 && near(ci.second, 0.088, 1e-3));
    ci = wilsonInterval(0, 0);
    assert(ci.first == 0.0 && ci.second == 1.0);
}

int main() {
    test_add_normalizes_order();
    test_equal_results_equal_ratings();
    test_recovers_known_ladder();
    test_perfect_record_is_finite();
    test_disconnected_players_omitted();
    test_csv_round_trip();
    test_csv_rejects_bad_input();
    test_wilson_interval();
    std::printf("test_elo: all tests passed\n");
    return 0;
}
```

The expected values were computed by hand. For the 20-0 case, the virtual draw gives player 1 a score of 20.5 out of 21, so `γ1/(1+γ1) = 20.5/21`, `γ1 = 41` and the Elo is `400·log10 41 ≈ 645`. For the Wilson interval at 20/40: `center = 0.5`, `half = 1.96·sqrt(0.00625 + 0.0006)/1.096 ≈ 0.148`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL, compile error in `tests/test_elo.cpp`: `'azgo/elo.hpp' file not found`.

- [ ] **Step 3: Write the header**

Create `include/azgo/elo.hpp`:

```cpp
// Measuring strength without an oracle. Networks play each other; every
// pairwise result is kept, and Bradley-Terry fits one strength per player
// such that P(i beats j) = g_i / (g_i + g_j). On the Elo scale a 200-point
// gap means the stronger player is expected to score about 76%.
// Refitting from the whole table is order-independent, so a resumed run
// reproduces the same ratings.
#pragma once
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace azgo {

// All arena results so far. Players are generation numbers.
class ResultsTable {
public:
    struct Pair { int a; int b; int games; double scoreA; };  // a < b; scoreA in [0, games]
    // scoreA: 1 win, 0.5 draw, 0 loss for player a. a != b; either order is
    // accepted and stored normalized as (min, max) with the score flipped as needed.
    void addGame(int a, int b, double scoreA);
    void addMatch(int a, int b, int games, double scoreA);    // many games at once
    std::vector<Pair> pairs() const;                          // sorted by (a, b)
    std::vector<int> players() const;                         // sorted
    bool empty() const { return table_.empty(); }
    std::string toCsv() const;                                // "a,b,games,score_a" header + rows
    static ResultsTable fromCsv(const std::string& csv);      // throws std::runtime_error
private:
    std::map<std::pair<int, int>, std::pair<int, double>> table_;  // (a,b) -> (games, scoreA)
};

// Maximum-likelihood Bradley-Terry ratings by the MM algorithm (Hunter 2004),
// on the Elo scale (400 * log10), with `anchor` pinned at 0. Each played pair
// gets one extra virtual game scored as a draw, so a player who has only won
// (or only lost) still gets a finite rating. Players unreachable from the
// anchor through played pairs are omitted.
std::map<int, double> fitBradleyTerry(const ResultsTable& results, int anchor = 0);

// 95% (z = 1.96) Wilson score interval for a win rate of score/games.
std::pair<double, double> wilsonInterval(double score, int games, double z = 1.96);

} // namespace azgo
```

- [ ] **Step 4: Write the implementation**

Create `src/elo.cpp`:

```cpp
#include "azgo/elo.hpp"
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <set>
#include <sstream>
#include <stdexcept>

namespace azgo {

void ResultsTable::addGame(int a, int b, double scoreA) { addMatch(a, b, 1, scoreA); }

void ResultsTable::addMatch(int a, int b, int games, double scoreA) {
    assert(a != b);
    assert(games > 0 && scoreA >= 0 && scoreA <= games);
    if (a > b) {
        // Store every pair once, as (smaller, larger): b's score is what a didn't score.
        std::swap(a, b);
        scoreA = games - scoreA;
    }
    auto& entry = table_[{a, b}];
    entry.first += games;
    entry.second += scoreA;
}

std::vector<ResultsTable::Pair> ResultsTable::pairs() const {
    std::vector<Pair> out;
    for (const auto& kv : table_)   // std::map iterates in (a, b) order
        out.push_back({kv.first.first, kv.first.second, kv.second.first, kv.second.second});
    return out;
}

std::vector<int> ResultsTable::players() const {
    std::set<int> ids;
    for (const auto& kv : table_) {
        ids.insert(kv.first.first);
        ids.insert(kv.first.second);
    }
    return std::vector<int>(ids.begin(), ids.end());
}

std::string ResultsTable::toCsv() const {
    std::string out = "a,b,games,score_a\n";
    char line[96];
    for (const Pair& p : pairs()) {
        std::snprintf(line, sizeof line, "%d,%d,%d,%.9g\n", p.a, p.b, p.games, p.scoreA);
        out += line;
    }
    return out;
}

namespace {

bool parseIntField(const std::string& s, int& out) {
    if (s.empty()) return false;
    errno = 0;
    char* end = nullptr;
    long v = std::strtol(s.c_str(), &end, 10);
    if (errno == ERANGE || *end != '\0' || v < -2147483647L - 1 || v > 2147483647L) return false;
    out = static_cast<int>(v);
    return true;
}

bool parseDoubleField(const std::string& s, double& out) {
    if (s.empty()) return false;
    errno = 0;
    char* end = nullptr;
    out = std::strtod(s.c_str(), &end);
    return errno != ERANGE && *end == '\0' && std::isfinite(out);
}

} // namespace

ResultsTable ResultsTable::fromCsv(const std::string& csv) {
    ResultsTable table;
    std::istringstream in(csv);
    std::string line;
    int lineNo = 0;
    auto fail = [&](const std::string& why) {
        throw std::runtime_error("results csv line " + std::to_string(lineNo) + ": " + why);
    };
    while (std::getline(in, line)) {
        ++lineNo;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (lineNo == 1) {
            if (line != "a,b,games,score_a") fail("expected header 'a,b,games,score_a'");
            continue;
        }
        if (line.empty()) continue;
        std::vector<std::string> f;
        std::istringstream fields(line);
        std::string field;
        while (std::getline(fields, field, ',')) f.push_back(field);
        int a = 0, b = 0, games = 0;
        double score = 0;
        if (f.size() != 4) fail("expected 4 fields");
        if (!parseIntField(f[0], a) || !parseIntField(f[1], b) || !parseIntField(f[2], games) ||
            !parseDoubleField(f[3], score))
            fail("malformed number");
        if (a == b || a < 0 || b < 0) fail("players must be distinct generations >= 0");
        if (games <= 0 || score < 0 || score > games) fail("score_a must be in [0, games]");
        table.addMatch(a, b, games, score);
    }
    if (lineNo == 0) throw std::runtime_error("results csv line 1: missing header");
    return table;
}

std::map<int, double> fitBradleyTerry(const ResultsTable& results, int anchor) {
    std::vector<ResultsTable::Pair> pairs = results.pairs();

    // 1. Players connected to the anchor (BFS over played pairs). Anyone else
    //    has no rating relative to the anchor.
    std::map<int, std::vector<int>> neighbors;
    for (const auto& p : pairs) {
        neighbors[p.a].push_back(p.b);
        neighbors[p.b].push_back(p.a);
    }
    if (!neighbors.count(anchor)) return {{anchor, 0.0}};
    std::map<int, int> index;   // generation -> dense index
    std::vector<int> ids;
    std::deque<int> queue{anchor};
    index[anchor] = 0;
    ids.push_back(anchor);
    while (!queue.empty()) {
        int p = queue.front();
        queue.pop_front();
        for (int q : neighbors[p]) {
            if (index.count(q)) continue;
            index[q] = static_cast<int>(ids.size());
            ids.push_back(q);
            queue.push_back(q);
        }
    }

    // 2. Pair counts plus one virtual drawn game per played pair: n_ij + 1
    //    games, s_ij + 0.5 score. Without it a player who never lost would have
    //    an infinite maximum-likelihood rating.
    struct Edge { int i, j; double games; };
    std::vector<Edge> edges;
    std::vector<double> wins(ids.size(), 0.0);   // W_i, including virtual halves
    for (const auto& p : pairs) {
        auto it = index.find(p.a);
        if (it == index.end()) continue;         // other component (then b is too)
        int i = it->second, j = index.at(p.b);
        double n = p.games + 1.0;
        double si = p.scoreA + 0.5;
        wins[i] += si;
        wins[j] += n - si;
        edges.push_back({i, j, n});
    }

    // 3. MM iteration (Hunter 2004): gamma_i <- W_i / sum_j n_ij / (gamma_i + gamma_j).
    //    Each sweep increases the likelihood; rescaling keeps gamma_anchor = 1.
    const size_t m = ids.size();
    std::vector<double> gamma(m, 1.0), denom(m), next(m);
    for (int sweep = 0; sweep < 100000; ++sweep) {
        std::fill(denom.begin(), denom.end(), 0.0);
        for (const Edge& e : edges) {
            double d = e.games / (gamma[e.i] + gamma[e.j]);
            denom[e.i] += d;
            denom[e.j] += d;
        }
        for (size_t i = 0; i < m; ++i) next[i] = wins[i] / denom[i];
        double scale = next[0];   // index 0 is the anchor
        double maxChange = 0;
        for (size_t i = 0; i < m; ++i) {
            next[i] /= scale;
            maxChange = std::max(maxChange, std::fabs(std::log(next[i] / gamma[i])));
        }
        gamma.swap(next);
        if (maxChange < 1e-10) break;
    }

    std::map<int, double> elo;
    for (size_t i = 0; i < m; ++i) elo[ids[i]] = 400.0 * std::log10(gamma[i]);
    return elo;
}

std::pair<double, double> wilsonInterval(double score, int games, double z) {
    if (games <= 0) return {0.0, 1.0};
    double n = games;
    double p = score / n;
    double z2 = z * z;
    // Center pulled toward 1/2 and a half-width that stays positive at p = 0
    // or 1: unlike p +- z*sqrt(p(1-p)/n), it behaves for small samples.
    double denom = 1.0 + z2 / n;
    double center = (p + z2 / (2.0 * n)) / denom;
    double half = z * std::sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom;
    return {std::max(0.0, center - half), std::min(1.0, center + half)};
}

} // namespace azgo
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_elo`
Expected: `100% tests passed, 0 tests failed out of 1`.

- [ ] **Step 6: Commit**

```bash
git add include/azgo/elo.hpp src/elo.cpp tests/test_elo.cpp
git commit -F - <<'EOF'
Add results table, Bradley-Terry Elo fit and Wilson interval

Without an external oracle, progress is measured by checkpoints playing
each other. Refitting all ratings from the full results table by
maximum likelihood (MM algorithm, generation 0 pinned at 0) is
order-independent, so a resumed run reproduces identical ratings. One
virtual draw per played pair keeps unbeaten players finite. The table
round-trips through CSV so it can be persisted and plotted.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 18: Arena

**Files:**
- Create: `include/azgo/arena.hpp`, `src/arena.cpp`
- Test: `tests/test_arena.cpp`

**Interfaces:**
- Consumes: `Network`, `Network::config()`, `NetConfig::operator!=` (Task 7); `std::unique_ptr<Backend> makeBackend(const std::string&, int)` (Task 5); `LocalEvaluator(const Network&, const Backend&, int maxBatch)` (Task 10); `MctsConfig`, `Mcts(const MctsConfig&, Evaluator&)`, `Mcts::search(const Game&, Rng&)`, `Mcts::selectAction(float, Rng&) const`, `Mcts::advance(int)`, `Mcts::nodeCount()` (Tasks 11–13); `Game(int n, float komi, int maxMoves)`, `Game::isTerminal`, `play`, `moves`, `moveCount`, `board`, `resultForBlack`, `defaultMaxMoves` (Task 3); `Color` (Task 2); `Rng`, `deriveSeed` (Task 1); `parallelFor`, `resolveWorkers` (Task 14).
- Produces: `ArenaConfig`, `MatchResult`, `playMatch(const Network&, const Network&, const ArenaConfig&, uint64_t seed, int workers, const std::string& backendName)`, `ladderOpponents(int, int)`, `passesGate(double, int, double)`, `matchScore(float, bool)`.

Per game `i`: `Rng rng(deriveSeed(seed, i))`, A is Black when `i` is even, and there is one `Mcts` per network (no noise). The side to move searches, then samples at T = 1 for the first `temperatureMoves` plies and T = 0 after that. Both trees `advance` after every move. Results are stored by game index, so the match is identical for any worker count. `workers <= 0` means hardware concurrency, and no more workers than games are started.

- [ ] **Step 1: Write the failing test**

Create `tests/test_arena.cpp`:

```cpp
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include "azgo/arena.hpp"

using namespace azgo;

static NetConfig tinyNet() {
    NetConfig c;
    c.boardSize = 5;
    c.history = 2;
    c.blocks = 1;
    c.filters = 4;
    c.valueHidden = 8;
    return c;
}

static ArenaConfig tinyArena() {
    ArenaConfig c;
    c.games = 8;
    c.simulations = 8;
    c.mctsBatch = 4;
    c.temperatureMoves = 0;
    c.komi = 7.5f;
    return c;
}

static bool sameResult(const MatchResult& x, const MatchResult& y) {
    return x.games == y.games && x.scoreA == y.scoreA && x.winsA == y.winsA &&
           x.winsB == y.winsB && x.draws == y.draws && x.distinctGames == y.distinctGames;
}

// Every game is counted once and scoreA agrees with the win/draw counts.
static void checkBookkeeping(const MatchResult& r, int games) {
    assert(r.games == games);
    assert(r.winsA + r.winsB + r.draws == games);
    assert(r.scoreA == r.winsA + 0.5 * r.draws);
    double scoreB = r.winsB + 0.5 * r.draws;
    assert(r.scoreA + scoreB == games);
    assert(r.distinctGames >= 1 && r.distinctGames <= games);
}

static void test_ladder() {
    assert(ladderOpponents(0, 4).empty());
    assert((ladderOpponents(1, 4) == std::vector<int>{0}));
    assert((ladderOpponents(2, 4) == std::vector<int>{1, 0}));
    assert((ladderOpponents(3, 4) == std::vector<int>{2, 1, 0}));
    assert((ladderOpponents(10, 4) == std::vector<int>{9, 8, 6, 2, 0}));
    assert((ladderOpponents(10, 2) == std::vector<int>{9, 8, 0}));
    assert((ladderOpponents(9, 8) == std::vector<int>{8, 7, 5, 1, 0}));
}

static void test_gate() {
    assert(passesGate(11, 20, 0.55));
    assert(!passesGate(10.9, 20, 0.55));
    assert(passesGate(20, 20, 0.55));
    assert(!passesGate(0, 0, 0.55));
}

// Review Focus 2: an exact tie scores half a point for each side.
static void test_match_score() {
    assert(matchScore(0.0f, true) == 0.5 && matchScore(0.0f, false) == 0.5);
    assert(matchScore(+1.0f, true) == 1.0 && matchScore(-1.0f, true) == 0.0);
    assert(matchScore(+1.0f, false) == 0.0 && matchScore(-1.0f, false) == 1.0);
}

static void test_deterministic_across_workers_and_runs() {
    Network a(tinyNet(), 11), b(tinyNet(), 22);
    ArenaConfig cfg = tinyArena();
    MatchResult one = playMatch(a, b, cfg, 42, 1, "naive");
    MatchResult three = playMatch(a, b, cfg, 42, 3, "naive");
    MatchResult again = playMatch(a, b, cfg, 42, 1, "naive");
    checkBookkeeping(one, 8);
    assert(sameResult(one, three));
    assert(sameResult(one, again));
}

static void test_colors_balanced() {
    // With komi 100 on 5x5 (area score at most 25) White wins every game, so
    // A's score counts exactly the games in which A played White.
    Network a(tinyNet(), 11), b(tinyNet(), 22);
    ArenaConfig cfg = tinyArena();
    cfg.komi = 100.0f;
    MatchResult r = playMatch(a, b, cfg, 7, 2, "naive");
    checkBookkeeping(r, 8);
    assert(r.scoreA == 4.0 && r.winsA == 4 && r.winsB == 4 && r.draws == 0);
    cfg.komi = -100.0f;   // Black always wins: again 4 of 8 for A
    r = playMatch(a, b, cfg, 7, 2, "naive");
    assert(r.scoreA == 4.0 && r.winsA == 4 && r.winsB == 4);

    // A network against itself: the bookkeeping still adds up.
    cfg.komi = 7.5f;
    r = playMatch(a, a, cfg, 3, 2, "naive");
    checkBookkeeping(r, 8);
}

static void test_opening_temperature_diversifies() {
    Network a(tinyNet(), 11), b(tinyNet(), 22);
    ArenaConfig cfg = tinyArena();
    cfg.temperatureMoves = 4;
    MatchResult r = playMatch(a, b, cfg, 5, 2, "naive");
    checkBookkeeping(r, 8);
    assert(r.distinctGames > 1);
}

static bool throwsInvalid(const Network& a, const Network& b, const ArenaConfig& cfg) {
    try {
        playMatch(a, b, cfg, 1, 1, "naive");
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

static void test_rejects_bad_setup() {
    NetConfig other = tinyNet();
    other.filters = 8;
    Network a(tinyNet(), 1), wide(other, 2);
    ArenaConfig cfg = tinyArena();
    assert(throwsInvalid(a, wide, cfg));
    cfg.games = 7;
    assert(throwsInvalid(a, a, cfg));
    cfg.games = 0;
    assert(throwsInvalid(a, a, cfg));
}

int main() {
    test_ladder();
    test_gate();
    test_match_score();
    test_deterministic_across_workers_and_runs();
    test_colors_balanced();
    test_opening_temperature_diversifies();
    test_rejects_bad_setup();
    std::printf("test_arena: all tests passed\n");
    return 0;
}
```

The color-balance test pins every outcome with komi. A 5×5 area score is at most 25, so with komi +100 White wins every game and with −100 Black wins every game. In both cases A must score exactly 4 of 8, and that only happens if A played each color in exactly 4 games.

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL, compile error in `tests/test_arena.cpp`: `'azgo/arena.hpp' file not found`.

- [ ] **Step 3: Write the header**

Create `include/azgo/arena.hpp`:

```cpp
// The arena: checkpoints play each other with search but without the
// exploration noise of self-play, so the result measures playing strength.
// Colors alternate game by game (Black has the first-move advantage), a few
// opening moves are sampled at temperature 1 so the games are not all the
// same, and each new generation meets a sparse ladder of older ones.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "azgo/network.hpp"

namespace azgo {

struct ArenaConfig {
    int games = 20;              // even; A plays Black in even-indexed games
    int simulations = 100;
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

// Plays cfg.games games between two networks of the same NetConfig (throws
// std::invalid_argument otherwise) on `workers` threads, no root noise, tree
// reuse on. Game i uses Rng(deriveSeed(seed, i)). Deterministic for a seed.
MatchResult playMatch(const Network& a, const Network& b, const ArenaConfig& cfg,
                      uint64_t seed, int workers, const std::string& backendName);

// Opponents for generation g: g-1, g-2, g-4, g-8, ... (>= 1), the first
// `maxOpponents` of those, then 0 if not already present. Empty for g == 0.
std::vector<int> ladderOpponents(int generation, int maxOpponents);

// score / games >= threshold.
bool passesGate(double score, int games, double threshold);

// Score of one finished game for player A: 1, 0.5 (resultForBlack == 0) or 0.
double matchScore(float resultForBlack, bool aIsBlack);

} // namespace azgo
```

- [ ] **Step 4: Write the implementation**

Create `src/arena.cpp`:

```cpp
#include "azgo/arena.hpp"
#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>
#include "azgo/backend.hpp"
#include "azgo/evaluator.hpp"
#include "azgo/game.hpp"
#include "azgo/mcts.hpp"
#include "azgo/parallel.hpp"
#include "azgo/rng.hpp"

namespace azgo {

namespace {

// One network's per-thread resources: its own backend and workspace.
struct Seat {
    std::unique_ptr<Backend> backend;
    std::unique_ptr<LocalEvaluator> evaluator;
    Seat(const Network& net, const std::string& backendName, int maxBatch)
        : backend(makeBackend(backendName, 1)),
          evaluator(new LocalEvaluator(net, *backend, maxBatch)) {}
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

    MctsConfig mc;
    mc.simulations = cfg.simulations;
    mc.batchSize = cfg.mctsBatch;
    mc.cPuct = cfg.cPuct;
    mc.addNoise = false;   // measuring strength, not exploring
    Mcts searchA(mc, *ctx.a.evaluator), searchB(mc, *ctx.b.evaluator);

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

} // namespace

MatchResult playMatch(const Network& a, const Network& b, const ArenaConfig& cfg, uint64_t seed,
                      int workers, const std::string& backendName) {
    if (a.config() != b.config())
        throw std::invalid_argument("playMatch: networks have different architectures");
    if (cfg.games < 2 || cfg.games % 2 != 0)
        throw std::invalid_argument("playMatch: games must be even and >= 2, got " +
                                    std::to_string(cfg.games));

    int threads = std::min(resolveWorkers(workers), cfg.games);
    std::vector<std::unique_ptr<WorkerContext>> contexts;
    for (int w = 0; w < threads; ++w)
        contexts.emplace_back(new WorkerContext{Seat(a, backendName, cfg.mctsBatch),
                                                Seat(b, backendName, cfg.mctsBatch)});

    // Results are stored by game index, never by completion order, so the
    // match is identical for any number of workers.
    std::vector<GameOutcome> outcomes(cfg.games);
    int n = a.config().boardSize;
    parallelFor(cfg.games, threads, [&](int index, int worker) {
        outcomes[index] = playArenaGame(*contexts[worker], cfg, n, seed, index);
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

std::vector<int> ladderOpponents(int generation, int maxOpponents) {
    std::vector<int> out;
    if (generation <= 0) return out;
    // g-1, g-2, g-4, g-8, ...: dense near the present, sparse into the past.
    for (int d = 1; generation - d >= 1 && static_cast<int>(out.size()) < maxOpponents; d *= 2)
        out.push_back(generation - d);
    out.push_back(0);   // the anchor: every rating is relative to generation 0
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

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_arena`
Expected: `100% tests passed, 0 tests failed out of 1`.

- [ ] **Step 6: Commit**

```bash
git add include/azgo/arena.hpp src/arena.cpp tests/test_arena.cpp
git commit -F - <<'EOF'
Add checkpoint arena with color balance, ladder and gating helpers

Elo needs games between checkpoints played for strength rather than
exploration: no root noise, tree reuse, alternating colors, and a few
temperature-1 opening moves so the games are not all identical (the
tic-tac-toe arena once produced 2 distinct games out of 40). Games are
seeded per index and stored by index, so a match is reproducible for
any number of worker threads. Exact ties score half a point.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 19: SGF

**Files:**
- Create: `include/azgo/sgf.hpp`, `src/sgf.cpp`
- Test: `tests/test_sgf.cpp`

**Interfaces:**
- Consumes: the test only uses `Game`, `Game::legalActions`, `play`, `moves`, `komi`, `moveCount`, `scoreForBlack`, `board`, `defaultMaxMoves` (Task 3); `Board::passAction`, `parseAction` (Task 2); `Rng::below` (Task 1). `src/sgf.cpp` itself uses only the standard library.
- Produces: `gameToSgf(int n, float komi, const std::vector<int>& moves, float scoreForBlack, const std::string& comment = "")`, `SgfGame`, `parseSgf(const std::string&)`.

Writer: `(;GM[1]FF[4]SZ[n]KM[%g]RE[..]` + optional `C[escaped]` + `;B[xy]`/`;W[xy]` (Black on even plies, pass = `[]`) + `)`. Here x is the column letter and y the row letter, and `aa` is the top-left point. Parser: a recursive-descent parser over `(`, `;`, `Ident[value]...`. It handles escapes and whitespace, collects only the main line (the first child at every branch), and checks the rest of the collection for well-formedness. Validation goes beyond the contract minimum: the moves must alternate B, W, B, … starting with Black, because our `Game` cannot replay anything else (handicap or setup stones), so other input is reported as malformed rather than silently misread.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sgf.cpp`:

```cpp
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/board.hpp"
#include "azgo/game.hpp"
#include "azgo/rng.hpp"
#include "azgo/sgf.hpp"

using namespace azgo;

static void test_write_exact() {
    // C3 on 5x5 is column 2, row 5 - 3 = 2 from the top: point 12, "cc".
    assert(parseAction("C3", 5) == 12);
    std::string sgf = gameToSgf(5, 7.5f, {12, 25}, 3.5f);
    assert(sgf == "(;GM[1]FF[4]SZ[5]KM[7.5]RE[B+3.5];B[cc];W[])");

    // Top-right corner of 9x9 is column i (index 8), row a: "ia".
    assert(gameToSgf(9, 6.5f, {8}, 1.0f) == "(;GM[1]FF[4]SZ[9]KM[6.5]RE[B+1];B[ia])");
    assert(gameToSgf(5, 7.5f, {}, -7.5f) == "(;GM[1]FF[4]SZ[5]KM[7.5]RE[W+7.5])");
}

// Review Focus 2: integer komi can produce an exact tie.
static void test_tie_result() {
    std::string sgf = gameToSgf(5, 7.0f, {25, 25}, 0.0f);
    assert(sgf == "(;GM[1]FF[4]SZ[5]KM[7]RE[0];B[];W[])");
}

static void test_comment_escaping() {
    std::string sgf = gameToSgf(5, 7.5f, {0}, -7.5f, "gen 3] \\ note");
    assert(sgf == "(;GM[1]FF[4]SZ[5]KM[7.5]RE[W+7.5]C[gen 3\\] \\\\ note];B[aa])");
    SgfGame g = parseSgf(sgf);   // the escaped ']' does not end the value early
    assert(g.boardSize == 5 && g.komi == 7.5f && (g.moves == std::vector<int>{0}));
}

static void test_round_trip_random_game() {
    Game game(9, 7.5f, defaultMaxMoves(9));
    Rng rng(2024);
    for (int ply = 0; ply < 40; ++ply) {
        std::vector<int> legal = game.legalActions();
        std::vector<int> stones;   // avoid passing, except once, so the game continues
        for (int a : legal)
            if (a != game.board().passAction()) stones.push_back(a);
        int action = (ply == 17 || stones.empty()) ? game.board().passAction()
                                                   : stones[rng.below(static_cast<int>(stones.size()))];
        game.play(action);
    }
    assert(game.moveCount() == 40);
    std::string sgf = gameToSgf(9, game.komi(), game.moves(), game.scoreForBlack(), "round trip");
    SgfGame back = parseSgf(sgf);
    assert(back.boardSize == 9);
    assert(back.komi == 7.5f);
    assert(back.moves == game.moves());
}

static void test_parse_variants() {
    // Whitespace and newlines between tokens, "tt" as pass, unknown properties ignored.
    SgfGame g = parseSgf("  (;GM[1]FF[4]\n SZ[5] KM[6.5]PB[Black player]\r\n"
                         "  ;B[aa]\n  ;W[tt] ;B [ee]\n)\n");
    assert(g.boardSize == 5 && g.komi == 6.5f);
    assert((g.moves == std::vector<int>{0, 25, 24}));

    // Only the main line (first child at each branch) is read.
    g = parseSgf("(;SZ[5];B[aa](;W[bb];B[cc](;W[dd])(;W[ee]))(;W[ed];B[de]))");
    assert((g.moves == std::vector<int>{0, 6, 12, 18}));

    // Defaults: SZ 19, KM 0.
    g = parseSgf("(;B[pd];W[dp])");
    assert(g.boardSize == 19 && g.komi == 0.0f);
    assert((g.moves == std::vector<int>{3 * 19 + 15, 15 * 19 + 3}));

    // Setup and comment properties with several values or escapes are skipped.
    g = parseSgf("(;SZ[7]AB[aa][bb]C[a \\] tricky ( comment ;];B[];W[gg])");
    assert(g.boardSize == 7 && (g.moves == std::vector<int>{49, 48}));

    // A collection of several games: the first one is read.
    g = parseSgf("(;SZ[5];B[bb])\n(;SZ[9];B[aa])\n");
    assert(g.boardSize == 5 && (g.moves == std::vector<int>{6}));
}

static bool rejects(const std::string& text) {
    try {
        parseSgf(text);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

static void test_malformed_inputs_throw() {
    assert(rejects(""));
    assert(rejects("garbage"));
    assert(rejects("(;SZ[5];B[aa]"));         // missing ')'
    assert(rejects("(;SZ[5];B[aa]))"));        // extra ')'
    assert(rejects("(;SZ[5]C[unterminated"));
    assert(rejects("(SZ[5])"));                // no node
    assert(rejects("(;SZ[5];B)"));             // property without value
    assert(rejects("(;SZ[5];B[ff])"));         // off a 5x5 board
    assert(rejects("(;SZ[5];B[a])"));          // one letter
    assert(rejects("(;SZ[5];B[AA])"));         // uppercase
    assert(rejects("(;SZ[4])"));
    assert(rejects("(;SZ[20])"));
    assert(rejects("(;SZ[9x9])"));
    assert(rejects("(;SZ[9]KM[seven])"));
    assert(rejects("(;SZ[5];W[aa])"));         // White cannot move first
    assert(rejects("(;SZ[5];B[aa];B[bb])"));   // colors must alternate
}

int main() {
    test_write_exact();
    test_tie_result();
    test_comment_escaping();
    test_round_trip_random_game();
    test_parse_variants();
    test_malformed_inputs_throw();
    std::printf("test_sgf: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL, compile error in `tests/test_sgf.cpp`: `'azgo/sgf.hpp' file not found`.

- [ ] **Step 3: Write the header**

Create `include/azgo/sgf.hpp`:

```cpp
// SGF (Smart Game Format) is the text format every Go program and viewer
// understands. A game is a tree of nodes "(;...;...)"; each node holds
// properties like B[cd] (Black plays column c, row d, counted from the top
// left) or SZ[9]. We write self-play games for viewing in e.g. Sabaki and
// read the main line of simple SGF files for analyze.
#pragma once
#include <string>
#include <vector>

namespace azgo {

// "(;GM[1]FF[4]SZ[n]KM[komi]RE[B+R|W+R|0]C[comment];B[dd];W[tt]...)".
// Point coordinates: column letter then row letter, 'a' = left/top.
// Pass is written as an empty value "B[]". RE uses the sign of resultForBlack
// with the Tromp-Taylor margin, e.g. "B+3.5", "W+7.5", "0".
std::string gameToSgf(int n, float komi, const std::vector<int>& moves,
                      float scoreForBlack, const std::string& comment = "");

struct SgfGame { int boardSize = 19; float komi = 0; std::vector<int> moves; };
// Reads SZ, KM and the main line of B[]/W[] moves (pass: "" or "tt" when n <= 19).
// Ignores other properties and variations. Throws std::runtime_error on
// malformed input or off-board coordinates.
SgfGame parseSgf(const std::string& text);

} // namespace azgo
```

- [ ] **Step 4: Write the implementation**

Create `src/sgf.cpp`:

```cpp
#include "azgo/sgf.hpp"
#include <cassert>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace azgo {

namespace {

std::string formatG(double x) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%g", x);
    return buf;
}

// Inside a property value only ']' and '\' are special; a backslash escapes them.
std::string escapeValue(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == ']' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

std::string pointToSgf(int action, int n) {
    assert(action >= 0 && action <= n * n);
    if (action == n * n) return "";   // pass
    char xy[3] = {static_cast<char>('a' + action % n), static_cast<char>('a' + action / n), 0};
    return xy;
}

// ---- parser -------------------------------------------------------------

using Property = std::pair<std::string, std::vector<std::string>>;
using Node = std::vector<Property>;

// Recursive descent over  GameTree = "(" Node+ GameTree* ")",
// Node = ";" Property*,  Property = Ident ("[" value "]")+.
// Nodes on the main line (first child at every branch) are collected;
// variations are parsed for well-formedness and then dropped.
class Parser {
public:
    explicit Parser(const std::string& text) : s_(text) {}

    std::vector<Node> parse() {
        skipSpace();
        if (pos_ >= s_.size() || s_[pos_] != '(') fail("expected '(' at the start");
        std::vector<Node> mainLine;
        gameTree(true, mainLine);
        // A collection may hold further games; check them, but read only the first.
        while (peek() == '(') gameTree(false, mainLine);
        if (pos_ < s_.size()) fail("unexpected text after the game");
        return mainLine;
    }

private:
    const std::string& s_;
    size_t pos_ = 0;

    [[noreturn]] void fail(const std::string& what) const {
        throw std::runtime_error("sgf: " + what + " (at offset " + std::to_string(pos_) + ")");
    }
    void skipSpace() {
        while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
    }
    char peek() {
        skipSpace();
        return pos_ < s_.size() ? s_[pos_] : '\0';
    }

    void gameTree(bool onMainLine, std::vector<Node>& mainLine) {
        ++pos_;   // '('
        if (peek() != ';') fail("expected ';' to start a node");
        while (peek() == ';') {
            Node n = node();
            if (onMainLine) mainLine.push_back(std::move(n));
        }
        bool first = true;
        while (peek() == '(') {
            gameTree(onMainLine && first, mainLine);   // only the first child continues the main line
            first = false;
        }
        if (peek() != ')') fail("expected ')' (unbalanced parentheses)");
        ++pos_;
    }

    Node node() {
        ++pos_;   // ';'
        Node n;
        while (std::isalpha(static_cast<unsigned char>(peek()))) {
            std::string ident;
            while (pos_ < s_.size() && std::isalpha(static_cast<unsigned char>(s_[pos_]))) {
                // Old FF[3] files may spell identifiers with lowercase letters
                // ("AddBlack"); only the uppercase ones are significant.
                if (std::isupper(static_cast<unsigned char>(s_[pos_]))) ident += s_[pos_];
                ++pos_;
            }
            if (ident.empty()) fail("property identifier without uppercase letters");
            std::vector<std::string> values;
            while (peek() == '[') values.push_back(value());
            if (values.empty()) fail("property " + ident + " has no value");
            n.emplace_back(ident, std::move(values));
        }
        return n;
    }

    std::string value() {
        ++pos_;   // '['
        std::string v;
        while (pos_ < s_.size() && s_[pos_] != ']') {
            if (s_[pos_] == '\\') {
                ++pos_;   // the escaped character is taken literally
                if (pos_ >= s_.size()) break;
            }
            v += s_[pos_++];
        }
        if (pos_ >= s_.size()) fail("unterminated property value");
        ++pos_;   // ']'
        return v;
    }
};

int parseSgfInt(const std::string& v, const char* what) {
    errno = 0;
    char* end = nullptr;
    long x = std::strtol(v.c_str(), &end, 10);
    if (v.empty() || errno == ERANGE || *end != '\0')
        throw std::runtime_error(std::string("sgf: malformed ") + what + " '" + v + "'");
    return static_cast<int>(x);
}

int sgfToAction(const std::string& v, int n) {
    // "" is the FF[4] pass; "tt" is the traditional pass on boards up to 19x19.
    if (v.empty() || (v == "tt" && n <= 19)) return n * n;
    if (v.size() != 2 || v[0] < 'a' || v[0] >= 'a' + n || v[1] < 'a' || v[1] >= 'a' + n)
        throw std::runtime_error("sgf: bad coordinate '" + v + "' for a " + std::to_string(n) +
                                 "x" + std::to_string(n) + " board");
    int col = v[0] - 'a', row = v[1] - 'a';
    return row * n + col;
}

} // namespace

std::string gameToSgf(int n, float komi, const std::vector<int>& moves, float scoreForBlack,
                      const std::string& comment) {
    std::string result;
    if (scoreForBlack > 0) result = "B+" + formatG(scoreForBlack);
    else if (scoreForBlack < 0) result = "W+" + formatG(-scoreForBlack);
    else result = "0";   // exact tie, possible with integer komi

    std::string out = "(;GM[1]FF[4]SZ[" + std::to_string(n) + "]KM[" + formatG(komi) + "]RE[" +
                      result + "]";
    if (!comment.empty()) out += "C[" + escapeValue(comment) + "]";
    for (size_t ply = 0; ply < moves.size(); ++ply) {
        out += ply % 2 == 0 ? ";B[" : ";W[";   // Black moves first, no handicap
        out += pointToSgf(moves[ply], n) + "]";
    }
    out += ")";
    return out;
}

SgfGame parseSgf(const std::string& text) {
    std::vector<Node> nodes = Parser(text).parse();

    SgfGame game;
    bool haveSize = false, haveKomi = false;
    for (const Node& node : nodes) {
        for (const Property& p : node) {
            if (p.first == "SZ" && !haveSize) {
                game.boardSize = parseSgfInt(p.second[0], "SZ");
                haveSize = true;
            } else if (p.first == "KM" && !haveKomi) {
                const std::string& v = p.second[0];
                errno = 0;
                char* end = nullptr;
                double k = std::strtod(v.c_str(), &end);
                if (v.empty() || errno == ERANGE || *end != '\0' || !std::isfinite(k))
                    throw std::runtime_error("sgf: malformed KM '" + v + "'");
                game.komi = static_cast<float>(k);
                haveKomi = true;
            }
        }
    }
    if (game.boardSize < 5 || game.boardSize > 19)
        throw std::runtime_error("sgf: board size " + std::to_string(game.boardSize) +
                                 " out of range 5..19");

    for (const Node& node : nodes) {
        for (const Property& p : node) {
            if (p.first != "B" && p.first != "W") continue;
            // Our games alternate strictly, Black first; anything else
            // (handicap stones, two moves by one color) cannot be replayed.
            const char* expected = game.moves.size() % 2 == 0 ? "B" : "W";
            if (p.first != expected)
                throw std::runtime_error("sgf: move " + std::to_string(game.moves.size() + 1) +
                                         " should be played by " + expected);
            game.moves.push_back(sgfToAction(p.second[0], game.boardSize));
        }
    }
    return game;
}

} // namespace azgo
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_sgf`
Expected: `100% tests passed, 0 tests failed out of 1`.

- [ ] **Step 6: Commit**

```bash
git add include/azgo/sgf.hpp src/sgf.cpp tests/test_sgf.cpp
git commit -F - <<'EOF'
Add SGF writer and main-line SGF reader

Self-play games become viewable in any Go program (e.g. Sabaki) as
SGF, and analyze can load positions from SGF files. The writer records
size, komi, the Tromp-Taylor result (RE[0] for an exact tie) and the
moves; the reader follows the main line, ignores other properties and
variations, and reports malformed input as an error instead of
asserting, since SGF files come from users.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 20: Run state

**Files:**
- Create: `include/azgo/run_state.hpp`, `src/run_state.cpp`
- Test: `tests/test_run_state.cpp`

**Interfaces:**
- Consumes: `BinaryWriter`, `BinaryReader` (reads throw `std::runtime_error` at end of stream), `writeFileAtomically(path, contents)` (writes `path + ".tmp"` then renames) (Task 9); `ReplayBuffer`, `ReplayBuffer::save(BinaryWriter&)`, `ReplayBuffer::load(BinaryReader&)` (which restores capacity, positions, `totalPositionsAdded` and every `GameRecord`), `ReplayBuffer::sample`, `TrainingBatch` (Task 15); `ResultsTable::toCsv`, `ResultsTable::fromCsv` (Task 17). The test uses `UniformEvaluator` (Task 10), `SelfPlayConfig` and `playSelfPlayGame` (Task 14), and `Rng` (Task 1).
- Produces: `TrainingState`, `saveTrainingState(const std::string& path, const TrainingState&)`, `loadTrainingState(const std::string& path)`, `checkpointPath(runDir, generation)`, `statePath(runDir)`, `configPath(runDir)`. File layout: u32 magic `0x54535A41`, u32 version 1, i32 generation, i32 bestGeneration, u64 samplesTrained, replay buffer, u32 velocity count followed by (u64 size + floats) for each vector, and results CSV as `str`. Every load failure throws `std::runtime_error("<path>: <reason>")`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_state.cpp`:

```cpp
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/evaluator.hpp"
#include "azgo/rng.hpp"
#include "azgo/run_state.hpp"
#include "azgo/selfplay.hpp"

using namespace azgo;
namespace fs = std::filesystem;

static fs::path freshDir(const std::string& name) {
    fs::path dir = fs::temp_directory_path() / ("azgo_test_run_state_" + name);
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

static std::string readBytes(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static void writeBytes(const fs::path& p, const std::string& bytes) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// A state with real content: two short self-play games, momentum buffers,
// and arena results for two pairs.
static TrainingState makeState() {
    UniformEvaluator eval(5, 2, 0.0f, 8);
    SelfPlayConfig sp;
    sp.mcts.simulations = 8;
    sp.mcts.batchSize = 4;
    sp.temperatureMoves = 4;
    sp.maxMoves = 12;
    Rng rng(99);
    TrainingState s;
    s.generation = 7;
    s.bestGeneration = 5;
    s.samplesTrained = (1LL << 40) + 3;   // needs all 64 bits
    s.buffer = ReplayBuffer(1000);
    s.buffer.add(playSelfPlayGame(eval, sp, rng));
    s.buffer.add(playSelfPlayGame(eval, sp, rng));
    s.velocity = {{0.5f, -1.25f, 3.0f}, {}, {1e-7f}};
    s.results.addMatch(0, 7, 20, 12.5);
    s.results.addMatch(6, 7, 20, 9.0);
    return s;
}

static bool sameGame(const GameRecord& x, const GameRecord& y) {
    return x.boardSize == y.boardSize && x.komi == y.komi && x.moves == y.moves &&
           x.stones == y.stones && x.visits == y.visits && x.resultForBlack == y.resultForBlack;
}

static void assertSameState(const TrainingState& x, const TrainingState& y) {
    assert(x.generation == y.generation);
    assert(x.bestGeneration == y.bestGeneration);
    assert(x.samplesTrained == y.samplesTrained);
    assert(x.buffer.capacity() == y.buffer.capacity());
    assert(x.buffer.positions() == y.buffer.positions());
    assert(x.buffer.games() == y.buffer.games());
    assert(x.buffer.totalPositionsAdded() == y.buffer.totalPositionsAdded());
    for (size_t i = 0; i < x.buffer.games(); ++i) assert(sameGame(x.buffer.game(i), y.buffer.game(i)));
    assert(x.velocity == y.velocity);
    assert(x.results.toCsv() == y.results.toCsv());
    assert(x.results.pairs().size() == y.results.pairs().size());
}

static void test_round_trip() {
    fs::path dir = freshDir("roundtrip");
    std::string path = (dir / "state.bin").string();
    TrainingState original = makeState();
    assert(original.buffer.games() == 2 && original.buffer.positions() > 0);
    saveTrainingState(path, original);
    TrainingState loaded = loadTrainingState(path);
    assertSameState(original, loaded);

    // The restored buffer samples exactly like the original one.
    TrainingBatch a, b;
    Rng ra(5), rb(5);
    original.buffer.sample(16, 2, ra, a);
    loaded.buffer.sample(16, 2, rb, b);
    assert(a.size == 16 && b.size == 16);
    assert(a.input == b.input && a.pi == b.pi && a.z == b.z);

    // Saving the loaded state reproduces the same bytes.
    std::string path2 = (dir / "again.bin").string();
    saveTrainingState(path2, loaded);
    assert(readBytes(path) == readBytes(path2));
    fs::remove_all(dir);
}

static std::string loadError(const std::string& path) {
    try {
        loadTrainingState(path);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    assert(false && "expected std::runtime_error");
    return "";
}

// Review Focus 3: leftovers from a crash.
static void test_crash_leftovers() {
    fs::path dir = freshDir("crash");
    fs::path state = dir / "state.bin";
    fs::path tmp = dir / "state.bin.tmp";
    TrainingState original = makeState();
    saveTrainingState(state.string(), original);

    // A half-written temporary file from an interrupted save is ignored...
    Rng rng(1);
    std::string garbage;
    for (int i = 0; i < 1000; ++i) garbage += static_cast<char>(rng.below(256));
    writeBytes(tmp, garbage);
    assertSameState(original, loadTrainingState(state.string()));

    // ...and the next save replaces it and leaves no .tmp behind.
    original.generation = 8;
    saveTrainingState(state.string(), original);
    assert(!fs::exists(tmp));
    assert(loadTrainingState(state.string()).generation == 8);

    // A truncated state file fails with a message naming the file.
    std::string bytes = readBytes(state);
    fs::path truncated = dir / "truncated.bin";
    writeBytes(truncated, bytes.substr(0, bytes.size() / 2));
    std::string msg = loadError(truncated.string());
    assert(msg.find(truncated.string()) != std::string::npos);

    // Missing file.
    fs::path missing = dir / "missing.bin";
    msg = loadError(missing.string());
    assert(msg.find(missing.string()) != std::string::npos);

    // Wrong magic (e.g. a checkpoint passed by mistake).
    std::string badMagic = bytes;
    badMagic[0] = 'X';
    fs::path wrong = dir / "wrong.bin";
    writeBytes(wrong, badMagic);
    msg = loadError(wrong.string());
    assert(msg.find(wrong.string()) != std::string::npos && msg.find("magic") != std::string::npos);

    // A newer format version.
    std::string newer = bytes;
    newer[4] = 2;   // version is the u32 after the magic, little-endian
    writeBytes(wrong, newer);
    msg = loadError(wrong.string());
    assert(msg.find("version") != std::string::npos);

    // Trailing bytes mean the file is not what we wrote.
    writeBytes(wrong, bytes + "x");
    msg = loadError(wrong.string());
    assert(msg.find(wrong.string()) != std::string::npos);
    fs::remove_all(dir);
}

static void test_paths() {
    assert(checkpointPath("run", 7) == "run/checkpoints/gen_0007.azn");
    assert(checkpointPath("run", 0) == "run/checkpoints/gen_0000.azn");
    assert(checkpointPath("run", 12345) == "run/checkpoints/gen_12345.azn");
    assert(checkpointPath("runs/a/", 42) == "runs/a/checkpoints/gen_0042.azn");
    assert(statePath("run") == "run/state.bin");
    assert(configPath("run") == "run/config.cfg");
}

int main() {
    test_round_trip();
    test_crash_leftovers();
    test_paths();
    std::printf("test_run_state: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build -j`
Expected: FAIL, compile error in `tests/test_run_state.cpp`: `'azgo/run_state.hpp' file not found`.

- [ ] **Step 3: Write the header**

Create `include/azgo/run_state.hpp`:

```cpp
// Resuming a run. A checkpoint holds only the network; continuing training
// exactly also needs the optimizer's momentum, the replay buffer, the
// generation counters and the arena results. They go into one binary file
// that is replaced atomically, so a crash mid-save leaves the old state.
#pragma once
#include <string>
#include <vector>
#include "azgo/elo.hpp"
#include "azgo/replay_buffer.hpp"

namespace azgo {

// Everything needed to continue a run exactly, besides the checkpoints and
// config.cfg. The network for `generation` is checkpoints/gen_NNNN.azn.
struct TrainingState {
    int generation = 0;           // last completed generation
    int bestGeneration = 0;       // gating: network used for self-play
    long long samplesTrained = 0; // total training samples drawn
    ReplayBuffer buffer{1};
    std::vector<std::vector<float>> velocity;  // optimizer momentum, params() order
    ResultsTable results;
};

// Binary, magic "AZST" (u32 0x54535A41), version u32 = 1. Atomic replace.
void saveTrainingState(const std::string& path, const TrainingState& state);
TrainingState loadTrainingState(const std::string& path);   // throws std::runtime_error

// Run-directory paths.
std::string checkpointPath(const std::string& runDir, int generation);  // runDir/checkpoints/gen_0007.azn
std::string statePath(const std::string& runDir);                        // runDir/state.bin
std::string configPath(const std::string& runDir);                       // runDir/config.cfg

} // namespace azgo
```

- [ ] **Step 4: Write the implementation**

Create `src/run_state.cpp`:

```cpp
#include "azgo/run_state.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include "azgo/serialize.hpp"

namespace azgo {

namespace {
constexpr uint32_t kStateMagic = 0x54535A41;   // "AZST" read as little-endian bytes
constexpr uint32_t kStateVersion = 1;
} // namespace

void saveTrainingState(const std::string& path, const TrainingState& state) {
    // Build the whole file in memory first: writeFileAtomically then either
    // replaces the old state completely or (on a crash) not at all.
    std::ostringstream os(std::ios::binary);
    BinaryWriter w(os);
    w.u32(kStateMagic);
    w.u32(kStateVersion);
    w.i32(state.generation);
    w.i32(state.bestGeneration);
    w.u64(static_cast<uint64_t>(state.samplesTrained));
    state.buffer.save(w);
    w.u32(static_cast<uint32_t>(state.velocity.size()));
    for (const std::vector<float>& v : state.velocity) {
        w.u64(v.size());
        w.floats(v.data(), v.size());
    }
    w.str(state.results.toCsv());
    writeFileAtomically(path, os.str());
}

TrainingState loadTrainingState(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(path + ": cannot open training state");
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::istringstream is(bytes, std::ios::binary);

    // Every failure below (including BinaryReader's "unexpected end of
    // stream") is reported with the file name, so the user knows which file
    // is damaged.
    try {
        BinaryReader r(is);
        if (r.u32() != kStateMagic) throw std::runtime_error("bad magic (not a training state file)");
        uint32_t version = r.u32();
        if (version != kStateVersion)
            throw std::runtime_error("unsupported version " + std::to_string(version));
        TrainingState state;
        state.generation = r.i32();
        state.bestGeneration = r.i32();
        state.samplesTrained = static_cast<long long>(r.u64());
        state.buffer = ReplayBuffer::load(r);
        uint32_t tensors = r.u32();
        state.velocity.resize(tensors);
        for (std::vector<float>& v : state.velocity) {
            uint64_t size = r.u64();
            // A corrupt size must not trigger a huge allocation.
            if (size > bytes.size() / sizeof(float)) throw std::runtime_error("truncated velocity tensor");
            v.resize(static_cast<size_t>(size));
            r.floats(v.data(), v.size());
        }
        state.results = ResultsTable::fromCsv(r.str());
        if (is.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("unexpected trailing bytes");
        return state;
    } catch (const std::exception& e) {
        throw std::runtime_error(path + ": " + e.what());
    }
}

std::string checkpointPath(const std::string& runDir, int generation) {
    char name[32];
    std::snprintf(name, sizeof name, "gen_%04d.azn", generation);   // 7 -> gen_0007.azn
    return (std::filesystem::path(runDir) / "checkpoints" / name).string();
}

std::string statePath(const std::string& runDir) {
    return (std::filesystem::path(runDir) / "state.bin").string();
}

std::string configPath(const std::string& runDir) {
    return (std::filesystem::path(runDir) / "config.cfg").string();
}

} // namespace azgo
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_run_state`
Expected: `100% tests passed, 0 tests failed out of 1`.

- [ ] **Step 6: Commit**

```bash
git add include/azgo/run_state.hpp src/run_state.cpp tests/test_run_state.cpp
git commit -F - <<'EOF'
Add atomic training-state file and run-directory paths

Resuming a run exactly needs more than the latest checkpoint: the
optimizer momentum, the replay buffer, the generation counters and the
arena results. They are serialized into one buffer in memory and
written with an atomic rename, so a crash mid-save leaves the previous
state.bin intact and a stale state.bin.tmp is simply overwritten.
Truncated or foreign files fail with an error naming the file.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

---

> **Section notes (Tasks 21–22).**
>
> Section: Tasks 21-22. Contract notes: none (no contract defects found).
> Integration assumptions for the assembler:
> 1. Task 1's test glob names each test target after its file stem (tests/test_cli.cpp -> target test_cli);
> Task 21 adds apps/ to that target's include path with target_include_directories.
> 2. Executables land in CMAKE_BINARY_DIR (single-config generators such as Makefiles/Ninja), which the
> integration test relies on via $BUILD/<app>.
> 3. configs/5x5.cfg exists (Task 16) and is the base of the smoke test's tiny run.

### Task 21: Trainer, ArgParser and the `train` app

**Files:**
- Create: `include/azgo/trainer.hpp`, `src/trainer.cpp`, `apps/cli.hpp`, `apps/train.cpp`
- Modify: `CMakeLists.txt` (append the `train` executable, and add `apps/` to `test_cli`'s include path; exact lines in Steps 6 and 10)
- Test: `tests/test_trainer.cpp`, `tests/test_cli.cpp`

**Interfaces:**
- Consumes:
  - `config.hpp` (Task 16): `Config` (all fields), `ConfigError`, `Config::netConfig()`, `Config::effectiveMaxMoves()`, `Config::effectiveWorkers()`, `loadConfigFile(const std::string&, const Config& = Config{})`, `applyOverride(Config&, const std::string&)`, `configToString(const Config&)`, `architectureDiff(const Config&, const Config&)`
  - `run_state.hpp` (Task 20): `TrainingState`, `saveTrainingState(const std::string&, const TrainingState&)`, `loadTrainingState(const std::string&)`, `checkpointPath(const std::string&, int)`, `statePath(const std::string&)`, `configPath(const std::string&)`
  - `network.hpp` (Task 7): `Network(const NetConfig&, uint64_t)`, `Network::config()`, `Network::params()`, `Network::trainStep(Workspace&, const float*, const float*, const float*, int)`, `Workspace(const NetConfig&, int, const Backend&)`, `ParamView`, `LossStats`, `NetConfig::operator!=`
  - `checkpoint.hpp` (Task 9): `saveNetwork(Network&, const std::string&)`, `loadNetwork(const std::string&)`, `readCheckpointConfig(const std::string&)` (test)
  - `serialize.hpp` (Task 9): `writeFileAtomically(const std::string&, const std::string&)`
  - `optimizer.hpp` (Task 8): `SgdMomentum(Network&, float, float)`, `SgdMomentum::step(Network&, float)`, `SgdMomentum::velocity()`, `LrSchedule::parse(const std::string&)`, `LrSchedule::at(int)`
  - `backend.hpp` (Task 5): `makeBackend(const std::string&, int)`
  - `evaluator.hpp` (Task 10): `LocalEvaluator(const Network&, const Backend&, int)`
  - `selfplay.hpp`, `parallel.hpp` (Task 14): `SelfPlayConfig`, `GameRecord`, `playSelfPlayGame(Evaluator&, const SelfPlayConfig&, Rng&)`, `parallelFor(int, int, const std::function<void(int, int)>&)`
  - `replay_buffer.hpp` (Task 15): `ReplayBuffer(size_t)`, `add`, `setCapacity`, `sample(int, int, Rng&, TrainingBatch&)`, `capacity()`, `positions()`, `games()`, `totalPositionsAdded()`, `TrainingBatch`
  - `game.hpp` (Task 3): `Game(int, float, int)`, `Game::play`, `Game::scoreForBlack()`
  - `sgf.hpp` (Task 19): `gameToSgf(int, float, const std::vector<int>&, float, const std::string&)`
  - `arena.hpp` (Task 18): `ArenaConfig`, `MatchResult`, `playMatch(const Network&, const Network&, const ArenaConfig&, uint64_t, int, const std::string&)`, `ladderOpponents(int, int)`, `passesGate(double, int, double)`
  - `elo.hpp` (Task 17): `ResultsTable::{addMatch, pairs, toCsv, fromCsv, empty}`, `ResultsTable::Pair`, `fitBradleyTerry(const ResultsTable&, int)`, `wilsonInterval(double, int, double)`
  - `rng.hpp` (Task 1): `Rng(uint64_t)`, `deriveSeed(uint64_t, uint64_t, uint64_t, uint64_t)`
- Produces (exactly as in the Interface Contract):
  - `include/azgo/trainer.hpp`: `GenerationStats`, `Trainer::{startNew, resume, generation, config, runGeneration, runUntil, setLog}`, `kPhaseInit`, `kPhaseSelfPlay`, `kPhaseTrain`, `kPhaseArena`
  - `apps/cli.hpp`: `ArgParser` (header-only, global namespace)
  - executable `train`
  - run-directory files: `config.cfg`, `checkpoints/gen_NNNN.azn`, `state.bin`, `log.csv`, `elo.csv`, `results.csv`, `sgf/gen_NNNN_game_KK.sgf` (headers exactly as listed under the trainer in the contract)

This task has two TDD cycles (trainer, then ArgParser) followed by the app.

- [ ] **Step 1: Write the failing trainer test**

`tests/test_trainer.cpp`. Every test uses its own temporary run directory under `std::filesystem::temp_directory_path()`, removed on entry (stale leftovers) and on exit. The byte-identity tests are the heart of it: they are the only place that checks the determinism promise of the whole pipeline end to end.

```cpp
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>
#include "azgo/checkpoint.hpp"
#include "azgo/config.hpp"
#include "azgo/elo.hpp"
#include "azgo/network.hpp"
#include "azgo/run_state.hpp"
#include "azgo/trainer.hpp"

using namespace azgo;
namespace fs = std::filesystem;

// A 5x5 run small enough that a generation takes a fraction of a second.
static Config tinyConfig() {
    Config c;
    c.boardSize = 5;
    c.history = 2;
    c.blocks = 1;
    c.filters = 8;
    c.valueHidden = 8;
    c.gamesPerGeneration = 4;
    c.simulations = 8;
    c.mctsBatch = 4;
    c.temperatureMoves = 4;
    c.bufferPositions = 1000;
    c.trainSteps = 4;
    c.batchSize = 16;
    c.arenaGames = 2;
    c.arenaSimulations = 4;
    c.arenaOpponents = 2;
    c.workers = 2;
    c.sgfPerGeneration = 1;
    c.lrSchedule = "0:0.02";
    return c;
}

// A run directory under the system temp dir; removed on construction (stale
// leftovers from a crashed test) and on destruction.
struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& name)
        : path(fs::temp_directory_path() /
               ("azgo_test_trainer_" + std::to_string(getpid()) + "_" + name)) {
        fs::remove_all(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string str() const { return path.string(); }
    std::string file(const std::string& rel) const { return (path / rel).string(); }
};

static std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    assert(in && "file should exist");
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static int countLines(const std::string& s) {
    return static_cast<int>(std::count(s.begin(), s.end(), '\n'));
}

static std::vector<std::string> splitLines(const std::string& s) {
    std::vector<std::string> lines;
    std::istringstream in(s);
    for (std::string line; std::getline(in, line);) lines.push_back(line);
    return lines;
}

static std::unique_ptr<Trainer> startQuiet(const Config& c, const TempDir& dir) {
    std::unique_ptr<Trainer> t = Trainer::startNew(c, dir.str());
    t->setLog(nullptr);
    return t;
}

static std::unique_ptr<Trainer> resumeQuiet(const TempDir& dir, const std::vector<std::string>& overrides) {
    std::unique_ptr<Trainer> t = Trainer::resume(dir.str(), overrides);
    t->setLog(nullptr);
    return t;
}

// (a, b) -> Pair from results.csv.
static std::map<std::pair<int, int>, ResultsTable::Pair> readResults(const TempDir& dir) {
    std::map<std::pair<int, int>, ResultsTable::Pair> out;
    for (const ResultsTable::Pair& p : ResultsTable::fromCsv(readFile(dir.file("results.csv"))).pairs())
        out[{p.a, p.b}] = p;
    return out;
}

static void test_start_new_writes_run_files() {
    TempDir dir("start");
    const Config c = tinyConfig();
    std::unique_ptr<Trainer> t = startQuiet(c, dir);
    assert(t->generation() == 0);

    assert(fs::is_directory(dir.path / "sgf"));
    assert(configToString(loadConfigFile(configPath(dir.str()))) == configToString(c));
    assert(readCheckpointConfig(checkpointPath(dir.str(), 0)) == c.netConfig());

    TrainingState s = loadTrainingState(statePath(dir.str()));
    assert(s.generation == 0 && s.bestGeneration == 0 && s.samplesTrained == 0);
    assert(s.buffer.positions() == 0 && s.buffer.capacity() == 1000);
    // Momentum starts at zero, one buffer per parameter tensor.
    Network net = loadNetwork(checkpointPath(dir.str(), 0));
    std::vector<ParamView> params = net.params();
    assert(s.velocity.size() == params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        assert(s.velocity[i].size() == params[i].size);
        for (float v : s.velocity[i]) assert(v == 0.0f);
    }

    assert(readFile(dir.file("log.csv")) ==
           "generation,games,avg_length,pass_rate,black_win_rate,policy_loss,value_loss,"
           "lr,replay_ratio,selfplay_s,train_s,arena_s,elo,winrate_vs_prev\n");
    assert(readFile(dir.file("elo.csv")) ==
           "generation,elo,winrate_vs_prev,wr_low,wr_high\n0,0,,,\n");
}

static void test_two_generations() {
    TempDir dir("two");
    std::unique_ptr<Trainer> t = startQuiet(tinyConfig(), dir);

    // Generation 1 by hand, to check the returned statistics.
    GenerationStats st = t->runGeneration();
    assert(st.generation == 1 && st.games == 4);
    assert(st.avgLength >= 1 && st.avgLength <= 2 * 5 * 5);   // move cap 50
    assert(st.passRate >= 0 && st.passRate <= 1);
    assert(st.blackWinRate >= 0 && st.blackWinRate <= 1);
    assert(std::isfinite(st.policyLoss) && st.policyLoss > 0);
    assert(std::isfinite(st.valueLoss) && st.valueLoss >= 0);
    assert(st.lr == 0.02f);
    // 4 steps x 16 samples drawn from 4 games of avgLength plies each.
    assert(std::fabs(st.replayRatio - 64.0 / (4 * st.avgLength)) < 1e-9);
    assert(st.winRateVsPrev >= 0 && st.winRateVsPrev <= 1);   // played generation 0
    assert(std::isfinite(st.elo));

    t->runUntil(2);
    assert(t->generation() == 2);
    assert(fs::exists(checkpointPath(dir.str(), 1)));
    assert(fs::exists(checkpointPath(dir.str(), 2)));
    assert(!fs::exists(checkpointPath(dir.str(), 3)));

    // sgf_per_generation = 1: exactly one game per generation.
    const std::string sgf = readFile(dir.file("sgf/gen_0001_game_00.sgf"));
    assert(sgf.rfind("(;GM[1]", 0) == 0);
    assert(sgf.find("SZ[5]") != std::string::npos);
    assert(fs::exists(dir.path / "sgf" / "gen_0002_game_00.sgf"));
    assert(!fs::exists(dir.path / "sgf" / "gen_0001_game_01.sgf"));

    assert(countLines(readFile(dir.file("log.csv"))) == 3);   // header + 2 rows

    // Generation 1 plays 0; generation 2 plays its ladder {1} plus the anchor 0.
    std::map<std::pair<int, int>, ResultsTable::Pair> pairs = readResults(dir);
    assert(pairs.size() == 3);
    for (std::pair<int, int> key : {std::make_pair(0, 1), std::make_pair(0, 2), std::make_pair(1, 2)}) {
        assert(pairs.count(key));
        assert(pairs[key].games == 2);
    }

    TrainingState s = loadTrainingState(statePath(dir.str()));
    assert(s.generation == 2);
    assert(s.bestGeneration == 2);                 // no gating: always the newest
    assert(s.samplesTrained == 2 * 4 * 16);
    assert(s.buffer.games() == 8);                 // at most 400 positions < 1000: nothing evicted
    assert(static_cast<long long>(s.buffer.positions()) == s.buffer.totalPositionsAdded());

    // elo.csv: header plus generations 0, 1, 2. The anchor is exactly 0 with no
    // predecessor; generations 1 and 2 each played g-1, so all five fields are filled.
    const std::vector<std::string> rows = splitLines(readFile(dir.file("elo.csv")));
    assert(rows.size() == 4);
    assert(rows[1] == "0,0.0,,,");
    for (int g = 1; g <= 2; ++g) {
        const std::string& row = rows[static_cast<size_t>(g) + 1];
        assert(row.rfind(std::to_string(g) + ",", 0) == 0);
        assert(std::count(row.begin(), row.end(), ',') == 4);
        assert(row.find(",,") == std::string::npos && row.back() != ',');
    }
}

// Review: resume must continue exactly where an uninterrupted run would be.
static void test_resume_is_byte_identical() {
    TempDir a("resume_a"), b("resume_b");
    const Config c = tinyConfig();
    {
        std::unique_ptr<Trainer> t = startQuiet(c, a);
        t->runUntil(4);
    }
    {
        std::unique_ptr<Trainer> t = startQuiet(c, b);
        t->runUntil(2);
    }   // destroyed: nothing survives in memory
    {
        std::unique_ptr<Trainer> t = resumeQuiet(b, {});
        assert(t->generation() == 2);
        t->runUntil(4);
        assert(t->generation() == 4);
    }
    for (const char* f : {"checkpoints/gen_0004.azn", "state.bin", "elo.csv", "results.csv",
                          "sgf/gen_0004_game_00.sgf"})
        assert(readFile(a.file(f)) == readFile(b.file(f)));
}

// Workers change who plays which game and how gemm rows are split, never the result.
static void test_thread_count_independence() {
    TempDir one("workers1"), three("workers3");
    Config c = tinyConfig();
    c.workers = 1;
    startQuiet(c, one)->runUntil(2);
    c.workers = 3;
    startQuiet(c, three)->runUntil(2);
    for (const char* f : {"checkpoints/gen_0002.azn", "state.bin", "results.csv"})
        assert(readFile(one.file(f)) == readFile(three.file(f)));
}

static void test_resume_overrides() {
    TempDir dir("overrides");
    startQuiet(tinyConfig(), dir);   // generation 0 on disk; trainer discarded
    const std::string before = readFile(configPath(dir.str()));

    // Architecture keys are refused, naming the key, and nothing is rewritten.
    bool threw = false;
    try {
        Trainer::resume(dir.str(), {"filters=16"});
    } catch (const ConfigError& e) {
        threw = std::string(e.what()).find("filters") != std::string::npos;
    }
    assert(threw);
    assert(readFile(configPath(dir.str())) == before);

    // Tunable keys are accepted and persisted.
    std::unique_ptr<Trainer> t = resumeQuiet(dir, {"simulations=12"});
    assert(t->config().simulations == 12);
    assert(loadConfigFile(configPath(dir.str())).simulations == 12);
    assert(readFile(configPath(dir.str())).find("simulations = 12\n") != std::string::npos);
    t->runUntil(1);
    assert(t->generation() == 1);

    // Resuming a directory that holds no run is a ConfigError too.
    TempDir empty("no_run");
    threw = false;
    try {
        Trainer::resume(empty.str(), {});
    } catch (const ConfigError&) {
        threw = true;
    }
    assert(threw);
}

static void test_start_new_refuses_existing_run() {
    TempDir dir("existing");
    startQuiet(tinyConfig(), dir);
    bool threw = false;
    try {
        Trainer::startNew(tinyConfig(), dir.str());
    } catch (const ConfigError& e) {
        threw = std::string(e.what()).find("--resume") != std::string::npos;
    }
    assert(threw);
}

// With threshold 1.0 a new network becomes "best" only by winning every game
// against the current best. Replaying that rule over results.csv must land on
// the bestGeneration stored in state.bin.
static void test_gating_state_is_consistent() {
    TempDir dir("gating");
    Config c = tinyConfig();
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
    assert(loadTrainingState(statePath(dir.str())).bestGeneration == best);
}

int main() {
    test_start_new_writes_run_files();
    test_two_generations();
    test_resume_is_byte_identical();
    test_thread_count_independence();
    test_resume_overrides();
    test_start_new_refuses_existing_run();
    test_gating_state_is_consistent();
    std::printf("test_trainer: all tests passed\n");
    return 0;
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`
Expected: the build of `test_trainer` fails with `fatal error: 'azgo/trainer.hpp' file not found`.

- [ ] **Step 3: Write the header**

`include/azgo/trainer.hpp` — verbatim from the contract plus the teaching comment block.

```cpp
#pragma once
// The trainer is the outer AlphaZero loop, run as synchronous generations:
// self-play with the current (or, with gating, the best) network fills the
// replay buffer; SGD on samples from that buffer produces the next network;
// an arena measures it against earlier generations for Elo; then the
// checkpoint and the full training state are written. Every random stream is
// derived from (seed, generation, phase, index), so a run is reproducible for a
// seed, independent of the worker count, and a resumed run continues
// byte-identically to one that was never interrupted.
#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include "azgo/config.hpp"
#include "azgo/network.hpp"
#include "azgo/run_state.hpp"

namespace azgo {

struct GenerationStats {
    int generation = 0;
    int games = 0;
    double avgLength = 0;       // plies per game
    double passRate = 0;        // passes / plies
    double blackWinRate = 0;
    float policyLoss = 0, valueLoss = 0;   // mean over the generation's steps
    float lr = 0;
    double replayRatio = 0;     // samples drawn this generation / new positions
    double selfPlaySeconds = 0, trainSeconds = 0, arenaSeconds = 0;
    double elo = 0;             // after refit; previous value if no arena this generation
    double winRateVsPrev = -1;  // -1 when no match vs g-1
};

// The generation loop: self-play -> train -> arena/Elo -> checkpoint + state.
class Trainer {
public:
    // New run. Creates runDir (and checkpoints/, sgf/); throws ConfigError if
    // runDir already contains state.bin. Writes config.cfg, gen_0000.azn and
    // an initial state.bin (generation 0).
    static std::unique_ptr<Trainer> startNew(const Config& cfg, const std::string& runDir);
    // Loads config.cfg, applies overrides, refuses (ConfigError listing keys)
    // if an architecture key changed, loads state.bin and the latest checkpoint.
    // Rewrites config.cfg with the overridden values.
    static std::unique_ptr<Trainer> resume(const std::string& runDir,
                                           const std::vector<std::string>& overrides);

    int generation() const { return state_.generation; }
    const Config& config() const { return cfg_; }
    GenerationStats runGeneration();
    void runUntil(int lastGeneration);   // while generation() < lastGeneration
    void setLog(std::ostream* os) { log_ = os; }   // progress lines; nullptr = silent

private:
    Trainer(const Config& cfg, const std::string& runDir, Network net, TrainingState state);
    Config cfg_;
    std::string runDir_;
    Network net_;
    TrainingState state_;
    std::ostream* log_;
    // Further members: implementation-defined (Task 21).
};

// Seed phases for deriveSeed(cfg.seed, generation, phase, index).
constexpr uint64_t kPhaseInit = 0, kPhaseSelfPlay = 1, kPhaseTrain = 2, kPhaseArena = 3;

} // namespace azgo
```

- [ ] **Step 4: Write the implementation**

`src/trainer.cpp`. Things to notice while reading it:

- **Seeds.** Self-play game `i` of generation `g` uses `deriveSeed(seed, g, kPhaseSelfPlay, i)`; training uses `deriveSeed(seed, g, kPhaseTrain)`; the match against opponent `o` uses `deriveSeed(seed, g, kPhaseArena, o)`. No RNG state carries from one generation to the next, so `state.bin` needs no RNG state and a resume cannot drift.
- **Write order.** Checkpoint `g` → SGF/`results.csv`/`elo.csv` → `state.bin` → the `log.csv` row. `state.bin` is the commit point: until it says `g`, generation `g` has not happened, and redoing it rewrites the earlier files with identical bytes. Note that the `elo` column of `log.csv` is the rating as fitted at that generation, while `elo.csv` is rewritten from a refit of the whole results table, so older generations' ratings in `elo.csv` move as evidence accumulates; the explainer (Task 24, chapter 07) should say so.
- **`resume` rewrites `config.cfg` only after every check has passed**, so a refused resume (such as `filters=16`) leaves the run directory untouched. The test checks this.
- **elo.csv** starts as `0,0,,,` (written by `startNew`, before any arena). After each arena round it is regenerated from the whole results table, so the anchor row then reads `0,0.0,,,`.

```cpp
#include "azgo/trainer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>

#include "azgo/arena.hpp"
#include "azgo/backend.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/elo.hpp"
#include "azgo/evaluator.hpp"
#include "azgo/game.hpp"
#include "azgo/optimizer.hpp"
#include "azgo/parallel.hpp"
#include "azgo/selfplay.hpp"
#include "azgo/serialize.hpp"
#include "azgo/sgf.hpp"

namespace azgo {

namespace fs = std::filesystem;

namespace {

const char* const kLogHeader =
    "generation,games,avg_length,pass_rate,black_win_rate,policy_loss,value_loss,"
    "lr,replay_ratio,selfplay_s,train_s,arena_s,elo,winrate_vs_prev\n";
const char* const kEloHeader = "generation,elo,winrate_vs_prev,wr_low,wr_high\n";

using Clock = std::chrono::steady_clock;

double secondsSince(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string logPath(const std::string& runDir) { return (fs::path(runDir) / "log.csv").string(); }
std::string eloPath(const std::string& runDir) { return (fs::path(runDir) / "elo.csv").string(); }
std::string resultsPath(const std::string& runDir) { return (fs::path(runDir) / "results.csv").string(); }

std::string sgfPath(const std::string& runDir, int generation, int game) {
    char name[64];
    std::snprintf(name, sizeof name, "gen_%04d_game_%02d.sgf", generation, game);
    return (fs::path(runDir) / "sgf" / name).string();
}

// elo.csv: one row per rated generation, ascending. The win rate columns come
// from the (g-1, g) pair of the results table: how often g beat its
// predecessor, with a 95% Wilson interval. Empty when the pair was never played.
std::string eloCsv(const ResultsTable& results) {
    std::map<std::pair<int, int>, ResultsTable::Pair> byPair;
    for (const ResultsTable::Pair& p : results.pairs()) byPair[{p.a, p.b}] = p;

    std::string out = kEloHeader;
    for (const auto& [generation, rating] : fitBradleyTerry(results, 0)) {
        const double elo = rating + 0.0;   // turns a -0.0 into 0.0, so the anchor prints "0.0"
        char line[160];
        auto it = byPair.find({generation - 1, generation});
        if (it != byPair.end() && it->second.games > 0) {
            const int games = it->second.games;
            const double score = games - it->second.scoreA;   // scoreA is g-1's score
            const std::pair<double, double> ci = wilsonInterval(score, games);
            std::snprintf(line, sizeof line, "%d,%.1f,%.4f,%.4f,%.4f\n", generation, elo,
                          score / games, ci.first, ci.second);
        } else {
            std::snprintf(line, sizeof line, "%d,%.1f,,,\n", generation, elo);
        }
        out += line;
    }
    return out;
}

// Rating of `generation`, or of the latest rated generation before it when it
// has not played yet (no arena this generation), or 0 before any arena round.
double ratingAtOrBefore(const ResultsTable& results, int generation) {
    if (results.empty()) return 0.0;
    const std::map<int, double> ratings = fitBradleyTerry(results, 0);
    auto it = ratings.upper_bound(generation);
    if (it == ratings.begin()) return 0.0;
    return std::prev(it)->second;
}

void appendLogRow(const std::string& runDir, const GenerationStats& s) {
    char winRate[32] = "";
    if (s.winRateVsPrev >= 0) std::snprintf(winRate, sizeof winRate, "%.4f", s.winRateVsPrev);
    char line[512];
    std::snprintf(line, sizeof line,
                  "%d,%d,%.2f,%.4f,%.4f,%.5f,%.5f,%g,%.3f,%.3f,%.3f,%.3f,%.1f,%s\n",
                  s.generation, s.games, s.avgLength, s.passRate, s.blackWinRate,
                  s.policyLoss, s.valueLoss, s.lr, s.replayRatio, s.selfPlaySeconds,
                  s.trainSeconds, s.arenaSeconds, s.elo, winRate);
    std::ofstream out(logPath(runDir), std::ios::app | std::ios::binary);
    out << line;
    if (!out) throw std::runtime_error("cannot append to " + logPath(runDir));
}

} // namespace

Trainer::Trainer(const Config& cfg, const std::string& runDir, Network net, TrainingState state)
    : cfg_(cfg), runDir_(runDir), net_(std::move(net)), state_(std::move(state)),
      log_(&std::cout) {}

std::unique_ptr<Trainer> Trainer::startNew(const Config& cfg, const std::string& runDir) {
    try {
        LrSchedule::parse(cfg.lrSchedule);
    } catch (const std::invalid_argument& e) {
        throw ConfigError(std::string("lr_schedule: ") + e.what());
    }
    if (fs::exists(statePath(runDir)))
        throw ConfigError(runDir + " already contains a run; use --resume");

    fs::create_directories(fs::path(runDir) / "checkpoints");
    fs::create_directories(fs::path(runDir) / "sgf");
    writeFileAtomically(configPath(runDir), configToString(cfg));

    Network net(cfg.netConfig(), deriveSeed(cfg.seed, 0, kPhaseInit));
    saveNetwork(net, checkpointPath(runDir, 0));

    TrainingState state;
    state.buffer = ReplayBuffer(static_cast<size_t>(cfg.bufferPositions));
    state.velocity = SgdMomentum(net, cfg.momentum, cfg.weightDecay).velocity();   // all zeros

    writeFileAtomically(logPath(runDir), kLogHeader);
    writeFileAtomically(eloPath(runDir), std::string(kEloHeader) + "0,0,,,\n");
    // state.bin last: until it exists the directory is not a run, so a crash
    // anywhere above leaves something startNew can simply redo.
    saveTrainingState(statePath(runDir), state);

    return std::unique_ptr<Trainer>(new Trainer(cfg, runDir, std::move(net), std::move(state)));
}

std::unique_ptr<Trainer> Trainer::resume(const std::string& runDir,
                                         const std::vector<std::string>& overrides) {
    if (!fs::exists(statePath(runDir)))
        throw ConfigError(runDir + " does not contain a run (no state.bin); start one with --config");

    const Config saved = loadConfigFile(configPath(runDir));
    Config cfg = saved;
    for (const std::string& o : overrides) applyOverride(cfg, o);

    const std::vector<std::string> diff = architectureDiff(saved, cfg);
    if (!diff.empty()) {
        std::string keys;
        for (const std::string& k : diff) keys += (keys.empty() ? "" : ", ") + k;
        throw ConfigError("cannot change architecture keys on resume: " + keys);
    }

    TrainingState state = loadTrainingState(statePath(runDir));
    const std::string ckpt = checkpointPath(runDir, state.generation);
    Network net = loadNetwork(ckpt);
    if (net.config() != cfg.netConfig())
        throw ConfigError(ckpt + " does not match the architecture in " + configPath(runDir));

    // The momentum buffers must line up with the parameters one-to-one, or the
    // first optimizer step would read out of bounds.
    const std::vector<ParamView> params = net.params();
    bool velocityOk = state.velocity.size() == params.size();
    for (size_t i = 0; velocityOk && i < params.size(); ++i)
        velocityOk = state.velocity[i].size() == params[i].size;
    if (!velocityOk)
        throw std::runtime_error(statePath(runDir) +
                                 ": optimizer state does not match the network's parameters");

    state.buffer.setCapacity(static_cast<size_t>(cfg.bufferPositions));
    // Rewritten only after every check passed, so a refused resume leaves the
    // run directory untouched.
    writeFileAtomically(configPath(runDir), configToString(cfg));

    return std::unique_ptr<Trainer>(new Trainer(cfg, runDir, std::move(net), std::move(state)));
}

GenerationStats Trainer::runGeneration() {
    const int g = state_.generation + 1;
    const int workers = cfg_.effectiveWorkers();
    const int n = cfg_.boardSize;
    GenerationStats stats;
    stats.generation = g;

    // ---- 1. Self-play -----------------------------------------------------
    const auto selfPlayStart = Clock::now();
    const int numGames = cfg_.gamesPerGeneration;
    std::vector<GameRecord> records(static_cast<size_t>(numGames));
    {
        // With gating, games are generated by the best network so far, which
        // may be older than the one being trained.
        std::unique_ptr<Network> bestNet;
        if (cfg_.gating && state_.bestGeneration != state_.generation)
            bestNet = std::make_unique<Network>(
                loadNetwork(checkpointPath(runDir_, state_.bestGeneration)));
        const Network& selfPlayNet = bestNet ? *bestNet : net_;

        SelfPlayConfig sp;
        sp.mcts.simulations = cfg_.simulations;
        sp.mcts.batchSize = cfg_.mctsBatch;
        sp.mcts.cPuct = cfg_.cPuct;
        sp.mcts.dirichletScale = cfg_.dirichletScale;
        sp.mcts.noiseFraction = cfg_.noiseFraction;
        sp.temperatureMoves = cfg_.temperatureMoves;
        sp.komi = cfg_.komi;
        sp.maxMoves = cfg_.effectiveMaxMoves();

        // One single-threaded backend and evaluator per worker; the network
        // itself is shared read-only.
        std::vector<std::unique_ptr<Backend>> backends;
        std::vector<std::unique_ptr<LocalEvaluator>> evaluators;
        for (int w = 0; w < workers; ++w) {
            backends.push_back(makeBackend(cfg_.backend, 1));
            evaluators.push_back(
                std::make_unique<LocalEvaluator>(selfPlayNet, *backends.back(), cfg_.mctsBatch));
        }
        // Game i's randomness depends only on i, and its record is stored at
        // index i, so the result does not depend on which thread played it.
        parallelFor(numGames, workers, [&](int i, int w) {
            Rng rng(deriveSeed(cfg_.seed, static_cast<uint64_t>(g), kPhaseSelfPlay,
                               static_cast<uint64_t>(i)));
            records[static_cast<size_t>(i)] = playSelfPlayGame(*evaluators[static_cast<size_t>(w)], sp, rng);
        });
    }

    long long plies = 0, passes = 0;
    int blackWins = 0;
    for (const GameRecord& r : records) {
        plies += r.positions();
        for (int a : r.moves)
            if (a == n * n) ++passes;
        if (r.resultForBlack > 0) ++blackWins;
    }
    stats.games = numGames;
    stats.avgLength = static_cast<double>(plies) / numGames;
    stats.passRate = plies > 0 ? static_cast<double>(passes) / plies : 0.0;
    stats.blackWinRate = static_cast<double>(blackWins) / numGames;

    // A few games per generation as SGF. The record stores only the result's
    // sign, so the Tromp-Taylor margin for RE[] comes from replaying the moves.
    const int sgfGames = std::min(cfg_.sgfPerGeneration, numGames);
    for (int i = 0; i < sgfGames; ++i) {
        const GameRecord& r = records[static_cast<size_t>(i)];
        Game replay(n, r.komi, cfg_.effectiveMaxMoves());
        for (int a : r.moves) replay.play(a);
        const std::string comment = "azgo self-play, generation " + std::to_string(g) +
                                    ", game " + std::to_string(i);
        writeFileAtomically(sgfPath(runDir_, g, i),
                            gameToSgf(n, r.komi, r.moves, replay.scoreForBlack(), comment));
    }

    for (GameRecord& r : records) state_.buffer.add(std::move(r));   // index order
    stats.selfPlaySeconds = secondsSince(selfPlayStart);

    // ---- 2. Train -----------------------------------------------------------
    const auto trainStart = Clock::now();
    const float lr = LrSchedule::parse(cfg_.lrSchedule).at(g);
    {
        // The naive backend splits gemm rows over `workers` threads here; each
        // output element is still computed by one thread, so results match.
        std::unique_ptr<Backend> backend = makeBackend(cfg_.backend, workers);
        Workspace ws(net_.config(), cfg_.batchSize, *backend);
        SgdMomentum opt(net_, cfg_.momentum, cfg_.weightDecay);
        opt.velocity() = state_.velocity;
        Rng rng(deriveSeed(cfg_.seed, static_cast<uint64_t>(g), kPhaseTrain));
        TrainingBatch batch;
        double policySum = 0, valueSum = 0;
        for (int s = 0; s < cfg_.trainSteps; ++s) {
            state_.buffer.sample(cfg_.batchSize, cfg_.history, rng, batch);
            const LossStats loss = net_.trainStep(ws, batch.input.data(), batch.pi.data(),
                                                  batch.z.data(), batch.size);
            // Nothing of generation g has been written yet, so the files on
            // disk still hold the last good state (generation g-1).
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
        state_.velocity = opt.velocity();
    }
    const long long drawn = static_cast<long long>(cfg_.trainSteps) * cfg_.batchSize;
    state_.samplesTrained += drawn;
    stats.lr = lr;
    stats.replayRatio = plies > 0 ? static_cast<double>(drawn) / plies : 0.0;
    stats.trainSeconds = secondsSince(trainStart);

    // ---- 3. Checkpoint ----------------------------------------------------
    // Written before state.bin (step 5). A crash in between leaves state.bin at
    // generation g-1; rerunning then reproduces generation g exactly (every
    // random stream is derived from the seed) and overwrites this file with
    // identical bytes. The same holds for the SGF, results.csv and elo.csv.
    saveNetwork(net_, checkpointPath(runDir_, g));

    // ---- 4. Arena and Elo ---------------------------------------------------
    const auto arenaStart = Clock::now();
    int newBest = cfg_.gating ? state_.bestGeneration : g;
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
            if (o == g - 1) stats.winRateVsPrev = r.scoreA / r.games;
            if (cfg_.gating && o == state_.bestGeneration &&
                passesGate(r.scoreA, r.games, cfg_.gatingThreshold))
                newBest = g;
        }
        // Refit every rating from all results so far: Bradley-Terry does not
        // depend on the order results arrived in, so a resumed run agrees.
        writeFileAtomically(resultsPath(runDir_), state_.results.toCsv());
        writeFileAtomically(eloPath(runDir_), eloCsv(state_.results));
    }
    stats.elo = ratingAtOrBefore(state_.results, g);
    stats.arenaSeconds = secondsSince(arenaStart);

    // ---- 5. Training state --------------------------------------------------
    state_.bestGeneration = newBest;
    state_.generation = g;
    saveTrainingState(statePath(runDir_), state_);
    // log.csv is a human-facing record (it contains timings) and is appended
    // after state.bin: a crash between the two loses one row, never corrupts
    // the run.
    appendLogRow(runDir_, stats);

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
            std::snprintf(line + len, sizeof line - static_cast<size_t>(len),
                          "arena elo %.1f vs-prev %s %.1fs", stats.elo, vsPrev,
                          stats.arenaSeconds);
        } else {
            std::snprintf(line + len, sizeof line - static_cast<size_t>(len),
                          "no arena, elo %.1f", stats.elo);
        }
        *log_ << line;
        if (cfg_.gating) *log_ << " | best " << state_.bestGeneration;
        *log_ << std::endl;
    }
    return stats;
}

void Trainer::runUntil(int lastGeneration) {
    while (generation() < lastGeneration) runGeneration();
}

} // namespace azgo
```

- [ ] **Step 5: Run the trainer test to verify it passes**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_trainer`
Expected: `100% tests passed, 0 tests failed out of 1`. The test binary prints `test_trainer: all tests passed`. It runs about 18 tiny generations and should take well under 30 s in Release (typically a few seconds).

If the resume or thread-count comparison fails, find the first file that differs before changing anything. `cmp` the two `state.bin` files: if the buffer differs, self-play is nondeterministic (look for a shared RNG or results stored by completion order). If only the velocity or weights differ, training is (look for a gemm thread split that sums one output element across threads).

- [ ] **Step 6: Write the failing ArgParser test and give it the apps/ include path**

`tests/test_cli.cpp`:

```cpp
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "cli.hpp"

static bool near(double a, double b, double tol = 1e-12) { return std::fabs(a - b) <= tol; }

static const std::vector<std::string> kValues = {"games", "komi", "run", "set", "seconds"};
static const std::vector<std::string> kFlags = {"resume", "help"};

// Builds a real argv (program name first) and parses it.
static ArgParser parse(std::vector<std::string> args) {
    args.insert(args.begin(), "prog");
    std::vector<char*> argv;
    for (std::string& a : args) argv.push_back(&a[0]);
    return ArgParser(static_cast<int>(argv.size()), argv.data(), kValues, kFlags);
}

// True when parsing (or the callable) throws std::invalid_argument whose
// message mentions `needle`.
template <typename F>
static bool throwsMentioning(F f, const std::string& needle) {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

static void test_values_and_defaults() {
    ArgParser p = parse({"--games", "40", "--komi", "-6.5", "--run", "runs/a"});
    assert(p.has("games") && p.has("komi") && p.has("run"));
    assert(!p.has("seconds"));
    assert(p.getInt("games", 0) == 40);
    assert(near(p.getDouble("komi", 7.5), -6.5));     // negative numbers are values
    assert(p.get("run") == "runs/a");
    assert(p.get("seconds", "2") == "2");               // absent: default
    assert(p.getInt("seconds", 3) == 3);
    assert(near(p.getDouble("seconds", 2.0), 2.0));
    assert(throwsMentioning([&] { p.getInt("komi", 0); }, "--komi"));   // "-6.5" is not an int
}

static void test_flags() {
    ArgParser p = parse({"--run", "r", "--resume"});
    assert(p.flag("resume"));
    assert(!p.flag("help"));
    assert(!p.has("resume"));                           // flags are not values
    ArgParser q = parse({});
    assert(!q.flag("resume") && !q.has("run") && q.sets().empty());
}

static void test_repeated_set_keeps_order() {
    ArgParser p = parse({"--set", "blocks=1", "--run", "r", "--set", "filters=8",
                         "--set", "komi=-3"});
    assert(p.sets().size() == 3);
    assert(p.sets()[0] == "blocks=1");
    assert(p.sets()[1] == "filters=8");
    assert(p.sets()[2] == "komi=-3");                   // '-' inside the value is fine
    assert(p.has("set"));
}

static void test_unknown_option_and_positional_throw() {
    assert(throwsMentioning([] { parse({"--simulation", "800"}); }, "--simulation"));
    assert(throwsMentioning([] { parse({"runs/a"}); }, "runs/a"));
    assert(throwsMentioning([] { parse({"-g", "4"}); }, "-g"));
}

static void test_missing_value_throws() {
    // At the end of the command line.
    assert(throwsMentioning([] { parse({"--games"}); }, "--games"));
    // Followed by another option: "--resume" is not a value.
    assert(throwsMentioning([] { parse({"--run", "--resume"}); }, "--run"));
    // A value option given twice is a mistake (only --set repeats).
    assert(throwsMentioning([] { parse({"--games", "4", "--games", "6"}); }, "--games"));
}

static void test_malformed_numbers_throw() {
    ArgParser p = parse({"--games", "12x", "--komi", "7.5.1", "--seconds", ""});
    assert(throwsMentioning([&] { p.getInt("games", 0); }, "12x"));
    assert(throwsMentioning([&] { p.getDouble("komi", 0); }, "--komi"));
    assert(throwsMentioning([&] { p.getInt("seconds", 0); }, "--seconds"));
    ArgParser q = parse({"--games", "99999999999"});
    assert(throwsMentioning([&] { q.getInt("games", 0); }, "--games"));   // out of int range
}

int main() {
    test_values_and_defaults();
    test_flags();
    test_repeated_set_keeps_order();
    test_unknown_option_and_positional_throw();
    test_missing_value_throws();
    test_malformed_numbers_throw();
    std::printf("test_cli: all tests passed\n");
    return 0;
}
```

The glob from Task 1 creates the `test_cli` target (linked to `azgo`, compiled with `-UNDEBUG`). `cli.hpp` lives in `apps/`, not `include/azgo/`, because only the executables use it. Append this at the **end** of `CMakeLists.txt`, after the test glob loop, since `test_cli` must already exist as a target:

```cmake
# --- Apps (Tasks 21-22) -------------------------------------------------------
# apps/cli.hpp is header-only and lives next to the apps that use it rather
# than in include/azgo, so test_cli (created by the test glob above) needs
# apps/ on its include path.
target_include_directories(test_cli PRIVATE ${CMAKE_SOURCE_DIR}/apps)
```

Run: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`
Expected: `test_cli` fails to compile with `fatal error: 'cli.hpp' file not found`.

- [ ] **Step 7: Write `apps/cli.hpp`**

The class declaration is verbatim from the contract. The member definitions follow it `inline` (the header is included by several apps and by `test_cli`). A value may start with `-` only if it parses as a number, so `--komi -6.5` works while `--run --resume` is reported as a missing value instead of silently making `--resume` the run directory.

```cpp
#pragma once
// Command-line parsing shared by the four apps. It is deliberately strict: a
// typo such as "--simulation 800" or "--games 4O" must stop the program with a
// message naming the option, not silently run with a default -- a mistake
// like that can waste an overnight training run. Header-only because only the
// apps (and test_cli) use it; it is not part of the azgo library.
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// Strict "--name value" / "--flag" parser. Unknown options, missing values
// and malformed numbers throw std::invalid_argument with the option name.
class ArgParser {
public:
    // valueOptions take a value; flagOptions do not. "--set" may repeat.
    ArgParser(int argc, char** argv, const std::vector<std::string>& valueOptions,
              const std::vector<std::string>& flagOptions);
    bool has(const std::string& name) const;
    std::string get(const std::string& name, const std::string& def = "") const;
    int getInt(const std::string& name, int def) const;       // strtol, whole string
    double getDouble(const std::string& name, double def) const;
    bool flag(const std::string& name) const;
    const std::vector<std::string>& sets() const { return sets_; }   // every --set value
private:
    std::map<std::string, std::string> values_;
    std::vector<std::string> flags_;
    std::vector<std::string> sets_;
};

inline ArgParser::ArgParser(int argc, char** argv, const std::vector<std::string>& valueOptions,
                            const std::vector<std::string>& flagOptions) {
    auto contains = [](const std::vector<std::string>& v, const std::string& s) {
        return std::find(v.begin(), v.end(), s) != v.end();
    };
    // A value may begin with '-' only when it is a number ("--komi -6.5");
    // otherwise "--run --resume" would swallow the next option as a value.
    auto isNumber = [](const std::string& s) {
        if (s.empty()) return false;
        char* end = nullptr;
        std::strtod(s.c_str(), &end);
        return end != s.c_str() && *end == '\0';
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.size() < 3 || arg.compare(0, 2, "--") != 0)
            throw std::invalid_argument("unexpected argument '" + arg + "'");
        const std::string name = arg.substr(2);
        if (contains(flagOptions, name)) {
            if (!contains(flags_, name)) flags_.push_back(name);
            continue;
        }
        if (!contains(valueOptions, name))
            throw std::invalid_argument("unknown option " + arg);
        if (i + 1 >= argc)
            throw std::invalid_argument("option " + arg + " requires a value");
        const std::string value = argv[++i];
        if (!value.empty() && value[0] == '-' && !isNumber(value))
            throw std::invalid_argument("option " + arg + " requires a value");
        if (name == "set") {
            sets_.push_back(value);
        } else {
            if (values_.count(name))
                throw std::invalid_argument("option " + arg + " given more than once");
            values_[name] = value;
        }
    }
}

inline bool ArgParser::has(const std::string& name) const {
    return values_.count(name) > 0 || (name == "set" && !sets_.empty());
}

inline std::string ArgParser::get(const std::string& name, const std::string& def) const {
    auto it = values_.find(name);
    return it == values_.end() ? def : it->second;
}

inline int ArgParser::getInt(const std::string& name, int def) const {
    auto it = values_.find(name);
    if (it == values_.end()) return def;
    const std::string& s = it->second;
    errno = 0;
    char* end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (s.empty() || end != s.c_str() + s.size() || errno == ERANGE || v < INT_MIN || v > INT_MAX)
        throw std::invalid_argument("option --" + name + " expects an integer, got '" + s + "'");
    return static_cast<int>(v);
}

inline double ArgParser::getDouble(const std::string& name, double def) const {
    auto it = values_.find(name);
    if (it == values_.end()) return def;
    const std::string& s = it->second;
    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || end != s.c_str() + s.size() || errno == ERANGE || !std::isfinite(v))
        throw std::invalid_argument("option --" + name + " expects a number, got '" + s + "'");
    return v;
}

inline bool ArgParser::flag(const std::string& name) const {
    return std::find(flags_.begin(), flags_.end(), name) != flags_.end();
}
```

- [ ] **Step 8: Run the ArgParser test to verify it passes**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R test_cli`
Expected: `100% tests passed, 0 tests failed out of 1` (the binary prints `test_cli: all tests passed`).

- [ ] **Step 9: Write `apps/train.cpp`**

```cpp
// train: runs the AlphaZero generation loop for a new or resumed run.
//
//   train --config configs/9x9.cfg --run runs/nine [--set key=value]... [--generations N]
//   train --run runs/nine --resume [--set key=value]... [--generations N]
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
    "\n"
    "A new run copies FILE (plus overrides) to DIR/config.cfg. --resume continues\n"
    "from DIR/state.bin; tunable keys may be overridden, architecture keys may not.\n"
    "--generations N is shorthand for --set generations=N (train until generation N).\n";

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
                                           std::vector<std::string>{"resume", "help"});
    } catch (const std::invalid_argument& e) {
        return usageError(e.what());
    }
    if (args->flag("help")) {
        std::fputs(kUsage, stdout);
        return 0;
    }
    if (!args->has("run")) return usageError("--run DIR is required");
    const bool resume = args->flag("resume");
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

- [ ] **Step 10: Add the `train` executable to CMake**

Append to the end of `CMakeLists.txt`, below the block from Step 6:

```cmake
add_executable(train apps/train.cpp)
target_link_libraries(train PRIVATE azgo)
target_include_directories(train PRIVATE ${CMAKE_SOURCE_DIR}/apps)
```

- [ ] **Step 11: Build, run all tests, and try the app by hand**

Run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure
TINY="--set blocks=1 --set filters=8 --set value_hidden=8 --set history=2 --set games_per_generation=4 --set simulations=8 --set train_steps=4 --set batch_size=16 --set arena_games=2 --set arena_simulations=4"
./build/train --config configs/5x5.cfg --run /tmp/azgo_try $TINY --generations 2
./build/train --run /tmp/azgo_try --resume --set filters=16; echo "exit $?"
./build/train --run /tmp/azgo_try --bogus; echo "exit $?"
rm -rf /tmp/azgo_try
```

Expected:
- ctest: `100% tests passed`.
- The first `train` prints `starting /tmp/azgo_try at generation 0; effective config ...`, the config lines, then two lines shaped like `gen 1 | selfplay 4 games avg 14.0 plies 0.1s | train loss p 3.25 v 1.02 lr 0.02 0.0s | arena elo -279.6 vs-prev 0.00 0.0s` (numbers vary; only the shape matters), and finally `done: generation 2 in /tmp/azgo_try`.
- The second prints `error: cannot change architecture keys on resume: filters` and `exit 1`.
- The third prints `error: unknown option --bogus`, the usage text, and `exit 1`.

BLAS variant: `cmake -S . -B build-blas -DCMAKE_BUILD_TYPE=Release -DAZ_USE_BLAS=ON && cmake --build build-blas -j && ctest --test-dir build-blas --output-on-failure -R 'test_trainer|test_cli'`. Expected: both pass. The test config uses `backend = naive`, so this checks that the BLAS build does not disturb the trainer.

- [ ] **Step 12: Commit**

```bash
git add include/azgo/trainer.hpp src/trainer.cpp apps/cli.hpp apps/train.cpp \
        tests/test_trainer.cpp tests/test_cli.cpp CMakeLists.txt
git commit -F - <<'EOF'
Add the generation-loop trainer and the train app

The trainer connects every component. Each generation, self-play fills
the replay buffer, SGD trains the next network, the arena refits
Bradley-Terry Elo, and the checkpoint and training state are written.
Every random stream is derived from the run seed and results are stored
by index. That lets the tests require byte-identical output across
worker counts and across a resume, which is what makes it safe to stop
an overnight run and continue it.

The ArgParser is strict and lives in apps/ because only the executables
use it. A mistyped option should stop a run rather than fall back to a
default without saying so.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 22: The arena, analyze and bench apps, and the integration smoke test

**Files:**
- Create: `apps/arena.cpp`, `apps/analyze.cpp`, `apps/bench.cpp`, `tests/integration_smoke.sh`
- Modify: `CMakeLists.txt` (append the `integration_smoke` test in Step 2 and the three executables in Step 6; exact lines below)
- Test: `tests/integration_smoke.sh` (registered with CTest as `integration_smoke`)

**Interfaces:**
- Consumes:
  - `apps/cli.hpp` (Task 21): `ArgParser`
  - `checkpoint.hpp` (Task 9): `loadNetwork(const std::string&)`
  - `arena.hpp` (Task 18): `ArenaConfig`, `MatchResult`, `playMatch(...)`; `elo.hpp` (Task 17): `wilsonInterval(double, int, double)`; `parallel.hpp` (Task 14): `resolveWorkers(int)`
  - `board.hpp` (Task 2): `Board::{size, at, isLegal, toMove, toString}`, `Color`, `parseAction(const std::string&, int)`, `actionToString(int, int)`, `kMinBoardSize`, `kMaxBoardSize`
  - `game.hpp` (Task 3): `Game(int, float, int)`, `Game::{board, isLegal, isTerminal, legalActions, play, moveCount}`, `defaultMaxMoves(int)`
  - `encode.hpp` (Task 4): `numPlanes(int)`, `encodeGame(const Game&, int, int, float*)`
  - `evaluator.hpp` (Task 10): `LocalEvaluator(const Network&, const Backend&, int)`, `LocalEvaluator::evaluate`
  - `mcts.hpp` (Tasks 11–13): `MctsConfig`, `Mcts(const MctsConfig&, Evaluator&)`, `Mcts::{search, rootValue, nodeCount, node, edge}`, `MctsNode::{parent, firstEdge, numEdges}`, `MctsEdge::{action, visits, valueSum}`
  - `sgf.hpp` (Task 19): `parseSgf(const std::string&)`, `SgfGame`
  - `backend.hpp` (Task 5): `makeBackend(const std::string&, int)`, `blasAvailable()`
  - `config.hpp` (Task 16): `loadConfigFile`, `Config::netConfig()`
  - `network.hpp` (Task 7): `Network(const NetConfig&, uint64_t)`, `Network::{infer, trainStep}`, `Workspace`, `NetConfig::{planes, area, actions}`
  - `rng.hpp` (Task 1): `Rng::below(int)`
  - executable `train` (Task 21)
- Produces: executables `arena`, `analyze`, `bench` with the command lines and output formats in the contract's "Executables" section, and the CTest `integration_smoke`, which covers Review Focus 4.

- [ ] **Step 1: Write the failing integration test**

`tests/integration_smoke.sh`. Section (5) is Review Focus 4. Each bad move must give `error: move K (<text>): <reason>` and exit status 1, never an assert or a crash. The suicide and superko move lists were worked out by hand (diagrams in the comments) and replayed through a separate rules simulator. They are also the only place outside `test_board`/`test_game` where the reason classification in `analyze` (occupied → suicide → superko) is exercised.

```bash
#!/usr/bin/env bash
# End-to-end smoke test of the four executables on a tiny 5x5 run:
# train -> resume -> arena -> analyze (good and bad input) -> bench, plus the
# refusal to change the architecture on resume.
#
# usage: tests/integration_smoke.sh BUILD_DIR SOURCE_DIR
set -euo pipefail

BUILD="$(cd "${1:?usage: integration_smoke.sh BUILD_DIR SOURCE_DIR}" && pwd)"
SRC="$(cd "${2:?usage: integration_smoke.sh BUILD_DIR SOURCE_DIR}" && pwd)"
W="$(mktemp -d "${TMPDIR:-/tmp}/azgo_smoke.XXXXXX")"
trap 'rm -rf "$W"' EXIT

fail() { echo "integration_smoke: FAILED: $*" >&2; exit 1; }

# The 5x5 preset shrunk to a network and budget that train in seconds.
TINY=(--set blocks=1 --set filters=8 --set value_hidden=8 --set history=2
      --set games_per_generation=4 --set simulations=8 --set train_steps=4
      --set batch_size=16 --set arena_games=2 --set arena_simulations=4 --set workers=2
      --set sgf_per_generation=1)
RUN="$W/run"
CK="$RUN/checkpoints"

echo "== train: new run, 2 generations =="
"$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$RUN" "${TINY[@]}" --generations 2
test -s "$CK/gen_0002.azn" || fail "gen_0002.azn missing"
test -s "$RUN/sgf/gen_0001_game_00.sgf" || fail "no SGF written for generation 1"

echo "== train: resume to generation 3 =="
"$BUILD/train" --run "$RUN" --resume --generations 3
test -s "$CK/gen_0003.azn" || fail "gen_0003.azn missing after resume"
[ "$(wc -l < "$RUN/log.csv" | tr -d ' ')" = "4" ] || fail "log.csv should have a header and 3 rows"

echo "== arena =="
out="$("$BUILD/arena" --a "$CK/gen_0000.azn" --b "$CK/gen_0003.azn" --games 4 --simulations 4 --workers 2)"
echo "$out"
echo "$out" | grep -Eq '^A .* scored' || fail "arena did not print a score line"

echo "== analyze: a legal position =="
out="$("$BUILD/analyze" --checkpoint "$CK/gen_0003.azn" --moves "C3 D4" --simulations 16)"
echo "$out"
echo "$out" | grep -q "MCTS value" || fail "analyze printed no MCTS value"

echo "== analyze: SGF input =="
printf '(;GM[1]FF[4]SZ[5]KM[7.5];B[cc];W[dd])\n' > "$W/ok.sgf"
"$BUILD/analyze" --checkpoint "$CK/gen_0003.azn" --sgf "$W/ok.sgf" --simulations 8 > /dev/null \
    || fail "analyze rejected a valid 5x5 SGF"

# Runs analyze with bad input; it must exit 1 (not crash, not assert) and
# print the expected message ("error: move K (<text>): <reason>" for a bad move).
expect_error() {
    local want="$1"; shift
    local status=0 out
    out="$("$BUILD/analyze" --checkpoint "$CK/gen_0003.azn" --simulations 4 "$@" 2>&1)" || status=$?
    [ "$status" -eq 1 ] || fail "analyze $* exited $status, expected 1; output: $out"
    echo "$out" | grep -qF -- "$want" || fail "analyze $* printed '$out', expected '$want'"
    echo "  ok: $want"
}

echo "== analyze: bad moves are user errors =="
expect_error "error: move 2 (C3): illegal (occupied)" --moves "C3 C3"
expect_error "error: move 1 (Z99): not a move on a 5x5 board" --moves "Z99"
# GTP column letters skip I, so I5 names no point.
expect_error "error: move 1 (I5): not a move on a 5x5 board" --moves "I5"

# Suicide. After C3 A2 C4 B1 (Black X, White O), Black to move:
#
#     A B C D E
#   5 . . . . .
#   4 . . X . .
#   3 . . X . .
#   2 O . . . .
#   1 . O . . .
#
# A1 is a one-point eye of White. A black stone there has no liberty and
# captures nothing (A2 keeps A3 and B2, B1 keeps B2 and C1): suicide.
expect_error "error: move 5 (A1): illegal (suicide)" --moves "C3 A2 C4 B1 A1"

# Superko (a simple ko). After B3 C3 C4 D4 C2 D2 A5 E3 (A5 is a spare Black
# move so that White gets E3 in before Black captures), Black to move:
#
#     A B C D E
#   5 X . . . .
#   4 . . X O .
#   3 . X O . O
#   2 . . X O .
#   1 . . . . .
#
# White C3 has one liberty, D3. Black D3 (move 9) captures it; the new D3
# stone is legal because it then has the liberty C3:
#
#   4 . . X O .
#   3 . X . X O
#   2 . . X O .
#
# White C3 (move 10) would capture D3 in turn and recreate exactly the
# position after move 8. Board::isLegal accepts it (it captures, so it is not
# suicide); only the positional-superko check in Game refuses it.
expect_error "error: move 10 (C3): illegal (superko)" --moves "B3 C3 C4 D4 C2 D2 A5 E3 D3 C3"
# The same sequence without the retake is fine: move 9 really is legal.
"$BUILD/analyze" --checkpoint "$CK/gen_0003.azn" --moves "B3 C3 C4 D4 C2 D2 A5 E3 D3" \
    --simulations 4 > /dev/null || fail "the ko capture D3 was rejected"

echo "== analyze: other bad input =="
printf '(;GM[1]FF[4]SZ[9]KM[7.5];B[cc])\n' > "$W/nine.sgf"
expect_error "error: the SGF is for 9x9" --sgf "$W/nine.sgf"
expect_error "error: move 3 (C3): game is over" --moves "pass pass C3"
expect_error "error: the game is over" --moves "pass pass"

echo "== bench =="
"$BUILD/bench" --sizes 5 --seconds 0.2 --configs "$SRC/configs"

echo "== train: resume must refuse an architecture change =="
status=0
out="$("$BUILD/train" --run "$RUN" --resume --set filters=16 --generations 4 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "resume with filters=16 exited $status, expected 1"
echo "$out" | grep -q "architecture" || fail "resume refusal does not mention architecture: $out"
test ! -e "$CK/gen_0004.azn" || fail "a refused resume still trained"

echo "integration_smoke: all checks passed"
```

Make it executable: `chmod +x tests/integration_smoke.sh`.

- [ ] **Step 2: Register the test and run it to verify it fails**

Append to the end of `CMakeLists.txt`:

```cmake
# End-to-end check of all four executables. ctest does not build targets:
# run `cmake --build build` first, as the usual workflow does.
add_test(NAME integration_smoke
         COMMAND bash ${CMAKE_SOURCE_DIR}/tests/integration_smoke.sh ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR})
```

Run: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure -R integration_smoke`
Expected: FAIL. Both `train` stages pass, then the script stops at `== arena ==` with `.../build/arena: No such file or directory`.

- [ ] **Step 3: Write `apps/arena.cpp`**

```cpp
// arena: plays two checkpoints against each other and prints A's score with a
// 95% Wilson confidence interval. For manual experiments; the trainer runs its
// own arena every generation.
//
//   arena --a runs/nine/checkpoints/gen_0040.azn --b runs/nine/checkpoints/gen_0020.azn
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
    "usage: arena --a CKPT --b CKPT [--games 40] [--simulations 200] [--komi 7.5]\n"
    "             [--temperature-moves 4] [--workers 0] [--backend naive] [--seed 1]\n"
    "\n"
    "Colors alternate: A plays Black in even-numbered games. --workers 0 uses every core.\n";

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
            std::vector<std::string>{"a", "b", "games", "simulations", "komi", "temperature-moves",
                                     "workers", "backend", "seed"},
            std::vector<std::string>{"help"});
    } catch (const std::invalid_argument& e) {
        return usageError(e.what());
    }
    if (args->flag("help")) {
        std::fputs(kUsage, stdout);
        return 0;
    }
    if (!args->has("a") || !args->has("b")) return usageError("--a and --b are required");

    try {
        azgo::ArenaConfig cfg;
        cfg.games = args->getInt("games", 40);
        cfg.simulations = args->getInt("simulations", 200);
        cfg.komi = static_cast<float>(args->getDouble("komi", 7.5));
        cfg.temperatureMoves = args->getInt("temperature-moves", 4);
        const int workers = args->getInt("workers", 0);
        const int seed = args->getInt("seed", 1);
        const std::string backend = args->get("backend", "naive");
        if (cfg.games < 2 || cfg.games % 2 != 0)
            throw std::invalid_argument("--games must be even and at least 2 (colors are balanced)");
        if (cfg.simulations < 1) throw std::invalid_argument("--simulations must be at least 1");
        if (cfg.temperatureMoves < 0) throw std::invalid_argument("--temperature-moves must be >= 0");
        if (workers < 0) throw std::invalid_argument("--workers must be >= 0");
        if (seed < 0) throw std::invalid_argument("--seed must be >= 0");

        const std::string pathA = args->get("a"), pathB = args->get("b");
        const azgo::Network a = azgo::loadNetwork(pathA);   // errors name the file
        const azgo::Network b = azgo::loadNetwork(pathB);

        const azgo::MatchResult r = azgo::playMatch(a, b, cfg, static_cast<uint64_t>(seed),
                                                    azgo::resolveWorkers(workers), backend);
        const std::pair<double, double> ci = azgo::wilsonInterval(r.scoreA, r.games);
        std::printf("A %s vs B %s: A scored %g/%d = %.3f (95%% CI %.3f-%.3f), distinct games %d\n",
                    pathA.c_str(), pathB.c_str(), r.scoreA, r.games, r.scoreA / r.games,
                    ci.first, ci.second, r.distinctGames);
        std::printf("A wins %d, B wins %d, draws %d\n", r.winsA, r.winsB, r.draws);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
```

- [ ] **Step 4: Write `apps/analyze.cpp`**

Moves are checked one at a time before they reach `Game::play` (which asserts legality). The reason is decided in this order:
1. `parseAction == -1`: not a move on this board.
2. The game is already over.
3. The point is occupied.
4. `Board::isLegal` rejects it: suicide.
5. Otherwise only `Game`'s history check refused it: superko.

The root is found as the one node with `parent == -1`, so the code does not depend on the arena index the MCTS implementation gives the root.

```cpp
// analyze: shows what a checkpoint thinks of one position -- the raw network
// (value and policy heat map) next to what search makes of it (MCTS visits).
// The gap between the two is the whole point of AlphaZero: search corrects
// the network's guess, and training teaches the network to guess the result.
//
//   analyze --checkpoint runs/nine/checkpoints/gen_0040.azn --moves "E5 C3 pass"
//   analyze --checkpoint gen_0040.azn --sgf game.sgf --simulations 800
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "azgo/backend.hpp"
#include "azgo/board.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/encode.hpp"
#include "azgo/evaluator.hpp"
#include "azgo/game.hpp"
#include "azgo/mcts.hpp"
#include "azgo/network.hpp"
#include "azgo/rng.hpp"
#include "azgo/sgf.hpp"
#include "cli.hpp"

namespace {

const char* const kUsage =
    "usage: analyze --checkpoint CKPT [--moves \"D4 C3 pass\"] [--sgf FILE]\n"
    "               [--simulations 400] [--komi 7.5] [--backend naive]\n"
    "\n"
    "Plays the moves (GTP coordinates, or the main line of an SGF file) from the\n"
    "empty board, then prints the network's value and policy for the position\n"
    "and the result of an MCTS search. With --sgf the file's komi is used unless\n"
    "--komi is given.\n";

constexpr char kColumns[] = "ABCDEFGHJKLMNOPQRST";   // GTP skips I
constexpr int kMctsBatch = 8;
constexpr int kTopMoves = 10;

int usageError(const std::string& message) {
    std::fprintf(stderr, "error: %s\n\n%s", message.c_str(), kUsage);
    return 1;
}

std::string readTextFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<std::string> splitWhitespace(const std::string& text) {
    std::istringstream in(text);
    std::vector<std::string> out;
    for (std::string tok; in >> tok;) out.push_back(tok);
    return out;
}

// Same frame as a Go diagram: columns A.. left to right, row n at the top.
// Each point shows its prior as a whole percent, "." below 1%, or the stone.
void printHeatMap(const azgo::Board& board, const std::vector<double>& prior) {
    const int n = board.size();
    std::printf("   ");
    for (int c = 0; c < n; ++c) std::printf("  %c", kColumns[c]);
    std::printf("\n");
    for (int r = 0; r < n; ++r) {
        std::printf("%2d ", n - r);
        for (int c = 0; c < n; ++c) {
            const int p = r * n + c;
            const azgo::Color stone = board.at(p);
            if (stone == azgo::Color::Black) std::printf("  X");
            else if (stone == azgo::Color::White) std::printf("  O");
            else if (prior[static_cast<size_t>(p)] < 0.01) std::printf("  .");
            else std::printf("%3ld", std::lround(100.0 * prior[static_cast<size_t>(p)]));
        }
        std::printf("\n");
    }
}

} // namespace

int main(int argc, char** argv) {
    std::unique_ptr<ArgParser> args;
    try {
        args = std::make_unique<ArgParser>(
            argc, argv,
            std::vector<std::string>{"checkpoint", "moves", "sgf", "simulations", "komi", "backend"},
            std::vector<std::string>{"help"});
    } catch (const std::invalid_argument& e) {
        return usageError(e.what());
    }
    if (args->flag("help")) {
        std::fputs(kUsage, stdout);
        return 0;
    }
    if (!args->has("checkpoint")) return usageError("--checkpoint CKPT is required");
    if (args->has("moves") && args->has("sgf")) return usageError("give --moves or --sgf, not both");

    try {
        const int simulations = args->getInt("simulations", 400);
        if (simulations < 1) throw std::invalid_argument("--simulations must be at least 1");
        float komi = static_cast<float>(args->getDouble("komi", 7.5));
        const std::string ckpt = args->get("checkpoint");

        const azgo::Network net = azgo::loadNetwork(ckpt);
        const azgo::NetConfig& nc = net.config();
        const int n = nc.boardSize;

        // (text as the user wrote it, action or -1). Moves are validated one
        // at a time below so the error can name the first bad one.
        std::vector<std::pair<std::string, int>> moves;
        if (args->has("sgf")) {
            const azgo::SgfGame sgf = azgo::parseSgf(readTextFile(args->get("sgf")));
            if (sgf.boardSize != n)
                throw std::runtime_error("the SGF is for " + std::to_string(sgf.boardSize) + "x" +
                                         std::to_string(sgf.boardSize) + " but the checkpoint is " +
                                         std::to_string(n) + "x" + std::to_string(n));
            if (!args->has("komi")) komi = sgf.komi;
            for (int a : sgf.moves) moves.emplace_back(azgo::actionToString(a, n), a);
        } else {
            for (const std::string& tok : splitWhitespace(args->get("moves")))
                moves.emplace_back(tok, azgo::parseAction(tok, n));
        }

        // User input, so every problem is a reported error, never an assert
        // inside Game::play.
        azgo::Game game(n, komi, azgo::defaultMaxMoves(n));
        for (size_t k = 0; k < moves.size(); ++k) {
            const std::string& text = moves[k].first;
            const int a = moves[k].second;
            std::string reason;
            if (a < 0) {
                reason = "not a move on a " + std::to_string(n) + "x" + std::to_string(n) + " board";
            } else if (game.isTerminal()) {
                reason = "game is over";
            } else if (!game.isLegal(a)) {
                // Pass is always legal before the end, so `a` is a point here.
                // Board::isLegal knows occupancy and suicide; what it allows
                // but Game refuses must be a repeated position.
                if (game.board().at(a) != azgo::Color::Empty) reason = "illegal (occupied)";
                else if (!game.board().isLegal(a)) reason = "illegal (suicide)";
                else reason = "illegal (superko)";
            }
            if (!reason.empty()) {
                std::fprintf(stderr, "error: move %zu (%s): %s\n", k + 1, text.c_str(), reason.c_str());
                return 1;
            }
            game.play(a);
        }
        if (game.isTerminal())
            throw std::runtime_error("the game is over after move " + std::to_string(game.moveCount()) +
                                     " (two passes or the move cap); there is nothing to analyze");

        const azgo::Board& board = game.board();
        std::printf("%s: %d moves played, %s to move, komi %g\n\n", ckpt.c_str(), game.moveCount(),
                    board.toMove() == azgo::Color::Black ? "Black" : "White", komi);
        std::printf("%s\n", board.toString().c_str());

        // ---- The raw network --------------------------------------------------
        std::unique_ptr<azgo::Backend> backend = azgo::makeBackend(args->get("backend", "naive"), 1);
        azgo::LocalEvaluator evaluator(net, *backend, kMctsBatch);
        const int actions = n * n + 1;
        std::vector<float> input(static_cast<size_t>(azgo::numPlanes(nc.history)) * n * n);
        azgo::encodeGame(game, nc.history, 0, input.data());
        std::vector<float> logits(static_cast<size_t>(actions));
        float value = 0;
        evaluator.evaluate(input.data(), 1, logits.data(), &value);
        std::printf("Value (side to move): %+.2f\n\n", value);

        // Priors as MCTS sees them: softmax over the legal actions only.
        const std::vector<int> legal = game.legalActions();
        std::vector<double> prior(static_cast<size_t>(actions), 0.0);
        float maxLogit = -std::numeric_limits<float>::infinity();
        for (int a : legal) maxLogit = std::max(maxLogit, logits[static_cast<size_t>(a)]);
        double sum = 0;
        for (int a : legal) {
            const double p = std::exp(static_cast<double>(logits[static_cast<size_t>(a)] - maxLogit));
            prior[static_cast<size_t>(a)] = p;
            sum += p;
        }
        for (int a : legal) prior[static_cast<size_t>(a)] /= sum;

        std::printf("Policy prior, %% of legal moves (\".\" below 1%%):\n");
        printHeatMap(board, prior);
        std::printf("Pass prior: %.1f%%\n\n", 100.0 * prior[static_cast<size_t>(n * n)]);

        std::vector<int> byPrior = legal;
        std::stable_sort(byPrior.begin(), byPrior.end(), [&](int x, int y) {
            return prior[static_cast<size_t>(x)] > prior[static_cast<size_t>(y)];
        });
        std::printf("Top %d moves by prior:\n", kTopMoves);
        for (size_t i = 0; i < byPrior.size() && i < static_cast<size_t>(kTopMoves); ++i)
            std::printf("  %-5s %5.1f%%\n", azgo::actionToString(byPrior[i], n).c_str(),
                        100.0 * prior[static_cast<size_t>(byPrior[i])]);

        // ---- Search -------------------------------------------------------------
        azgo::MctsConfig mc;
        mc.simulations = simulations;
        mc.batchSize = kMctsBatch;
        mc.addNoise = false;
        azgo::Mcts mcts(mc, evaluator);
        azgo::Rng rng(1);
        mcts.search(game, rng);
        std::printf("\nMCTS value (side to move): %+.2f after %d simulations\n", mcts.rootValue(),
                    simulations);

        // The root is the one node without a parent. Its edges carry Q from
        // the root player's point of view: valueSum / visits.
        int root = 0;
        for (size_t i = 0; i < mcts.nodeCount(); ++i)
            if (mcts.node(static_cast<int>(i)).parent == -1) root = static_cast<int>(i);
        const azgo::MctsNode& rootNode = mcts.node(root);
        std::vector<const azgo::MctsEdge*> visited;
        long long totalVisits = 0;
        for (int e = rootNode.firstEdge; e < rootNode.firstEdge + rootNode.numEdges; ++e) {
            const azgo::MctsEdge& edge = mcts.edge(e);
            if (edge.visits > 0) {
                visited.push_back(&edge);
                totalVisits += edge.visits;
            }
        }
        std::stable_sort(visited.begin(), visited.end(), [](const azgo::MctsEdge* x, const azgo::MctsEdge* y) {
            return x->visits > y->visits;
        });
        std::printf("Top %d moves by visits:\n", kTopMoves);
        for (size_t i = 0; i < visited.size() && i < static_cast<size_t>(kTopMoves); ++i) {
            const azgo::MctsEdge& edge = *visited[i];
            std::printf("  %-5s %6d visits (%5.1f%%)  Q %+.2f\n", azgo::actionToString(edge.action, n).c_str(),
                        edge.visits, 100.0 * edge.visits / static_cast<double>(totalVisits),
                        edge.valueSum / edge.visits);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
```

For `--moves "C3 D4"` the output has this shape (numbers depend on the checkpoint):

```
.../gen_0003.azn: 2 moves played, Black to move, komi 7.5

<board diagram from Board::toString>
Value (side to move): -0.12

Policy prior, % of legal moves ("." below 1%):
     A  B  C  D  E
 5   4  4  5  4  4
 4   4  4  4  O  4
 3   4  4  X  4  4
 2   4  4  4  4  4
 1   4  4  4  4  4
Pass prior: 3.9%

Top 10 moves by prior:
  C4      4.6%
  ...

MCTS value (side to move): -0.08 after 16 simulations
Top 10 moves by visits:
  B3         3 visits ( 18.8%)  Q +0.05
  ...
```

- [ ] **Step 5: Write `apps/bench.cpp`**

```cpp
// bench: measures what the presets cost, so their sizes are chosen from
// numbers rather than guesses. For each board size's preset network and each
// backend: inference throughput at the MCTS batch size on one thread (what a
// self-play worker sees) and the time of one training step at batch 128 on
// all cores (what the training phase sees). Then raw rules-engine speed.
//
//   bench [--sizes 5,9,19] [--backend naive|blas|all] [--seconds 2] [--configs configs]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "azgo/backend.hpp"
#include "azgo/config.hpp"
#include "azgo/game.hpp"
#include "azgo/network.hpp"
#include "azgo/parallel.hpp"
#include "azgo/rng.hpp"
#include "cli.hpp"

namespace {

const char* const kUsage =
    "usage: bench [--sizes 5,9,19] [--backend naive|blas|all] [--seconds 2] [--configs DIR]\n"
    "\n"
    "Network shapes come from DIR/<n>x<n>.cfg (default DIR: configs). --seconds is\n"
    "the minimum duration of each measurement.\n";

constexpr int kInferBatch = 8;     // the default MCTS batch (mcts_batch)
constexpr int kTrainBatch = 128;

using Clock = std::chrono::steady_clock;

int usageError(const std::string& message) {
    std::fprintf(stderr, "error: %s\n\n%s", message.c_str(), kUsage);
    return 1;
}

double secondsSince(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::vector<int> parseSizes(const std::string& text) {
    std::vector<int> sizes;
    std::stringstream in(text);
    for (std::string item; std::getline(in, item, ',');) {
        char* end = nullptr;
        const long n = std::strtol(item.c_str(), &end, 10);
        if (item.empty() || *end != '\0' || n < azgo::kMinBoardSize || n > azgo::kMaxBoardSize)
            throw std::invalid_argument("--sizes: '" + item + "' is not a board size in 5..19");
        sizes.push_back(static_cast<int>(n));
    }
    if (sizes.empty()) throw std::invalid_argument("--sizes is empty");
    return sizes;
}

// Random {0,1} planes: the network's cost does not depend on the values.
std::vector<float> randomInput(const azgo::NetConfig& nc, int batch, azgo::Rng& rng) {
    std::vector<float> input(static_cast<size_t>(batch) * nc.planes() * nc.area());
    for (float& x : input) x = static_cast<float>(rng.below(2));
    return input;
}

// Positions evaluated per second, batch kInferBatch, one thread.
double measureInference(const azgo::NetConfig& nc, const std::string& backendName, double seconds) {
    std::unique_ptr<azgo::Backend> backend = azgo::makeBackend(backendName, 1);
    const azgo::Network net(nc, 1);
    azgo::Workspace ws(nc, kInferBatch, *backend);
    azgo::Rng rng(7);
    const std::vector<float> input = randomInput(nc, kInferBatch, rng);
    std::vector<float> logits(static_cast<size_t>(kInferBatch) * nc.actions());
    std::vector<float> values(kInferBatch);

    net.infer(ws, input.data(), kInferBatch, logits.data(), values.data());   // warm-up
    long long calls = 0;
    const auto start = Clock::now();
    double elapsed = 0;
    do {
        net.infer(ws, input.data(), kInferBatch, logits.data(), values.data());
        ++calls;
        elapsed = secondsSince(start);
    } while (elapsed < seconds);
    return static_cast<double>(calls) * kInferBatch / elapsed;
}

// Milliseconds per training step (forward + backward), batch kTrainBatch,
// backend given every hardware thread as in the trainer.
double measureTrainStep(const azgo::NetConfig& nc, const std::string& backendName, double seconds) {
    std::unique_ptr<azgo::Backend> backend = azgo::makeBackend(backendName, azgo::resolveWorkers(0));
    azgo::Network net(nc, 1);
    azgo::Workspace ws(nc, kTrainBatch, *backend);
    azgo::Rng rng(11);
    const std::vector<float> input = randomInput(nc, kTrainBatch, rng);
    std::vector<float> pi(static_cast<size_t>(kTrainBatch) * nc.actions(), 0.0f);
    std::vector<float> z(kTrainBatch);
    for (int b = 0; b < kTrainBatch; ++b) {
        pi[static_cast<size_t>(b) * nc.actions() + rng.below(nc.actions())] = 1.0f;   // one-hot target
        z[static_cast<size_t>(b)] = rng.below(2) ? 1.0f : -1.0f;
    }

    net.trainStep(ws, input.data(), pi.data(), z.data(), kTrainBatch);   // warm-up
    long long steps = 0;
    const auto start = Clock::now();
    double elapsed = 0;
    do {
        net.trainStep(ws, input.data(), pi.data(), z.data(), kTrainBatch);
        ++steps;
        elapsed = secondsSince(start);
    } while (elapsed < seconds);
    return 1000.0 * elapsed / static_cast<double>(steps);
}

// Moves per second of uniformly random legal play through Game (legality
// including superko, captures, hashing), games restarted when they end.
double measurePlayouts(int n, double seconds) {
    azgo::Rng rng(static_cast<uint64_t>(n));
    long long moves = 0;
    const auto start = Clock::now();
    double elapsed = 0;
    do {
        azgo::Game game(n, 7.5f, azgo::defaultMaxMoves(n));
        while (!game.isTerminal()) {
            const std::vector<int> legal = game.legalActions();
            game.play(legal[static_cast<size_t>(rng.below(static_cast<int>(legal.size())))]);
            ++moves;
        }
        elapsed = secondsSince(start);
    } while (elapsed < seconds);
    return static_cast<double>(moves) / elapsed;
}

} // namespace

int main(int argc, char** argv) {
    std::unique_ptr<ArgParser> args;
    try {
        args = std::make_unique<ArgParser>(argc, argv,
                                           std::vector<std::string>{"sizes", "backend", "seconds", "configs"},
                                           std::vector<std::string>{"help"});
    } catch (const std::invalid_argument& e) {
        return usageError(e.what());
    }
    if (args->flag("help")) {
        std::fputs(kUsage, stdout);
        return 0;
    }

    try {
        const std::vector<int> sizes = parseSizes(args->get("sizes", "5,9,19"));
        const double seconds = args->getDouble("seconds", 2.0);
        if (seconds <= 0) throw std::invalid_argument("--seconds must be positive");
        const std::string configDir = args->get("configs", "configs");
        const std::string which = args->get("backend", azgo::blasAvailable() ? "all" : "naive");
        std::vector<std::string> backends;
        if (which == "all") {
            backends.push_back("naive");
            if (azgo::blasAvailable()) backends.push_back("blas");
        } else {
            azgo::makeBackend(which, 1);   // throws for an unknown or unavailable backend
            backends.push_back(which);
        }

        std::printf("%-6s %-17s %-8s %12s %14s\n", "size", "blocks x filters", "backend", "evals/s",
                    "train-step ms");
        for (int n : sizes) {
            const std::string preset =
                (std::filesystem::path(configDir) / (std::to_string(n) + "x" + std::to_string(n) + ".cfg")).string();
            if (!std::filesystem::exists(preset))
                throw std::runtime_error("no preset " + preset + " (use --configs DIR)");
            const azgo::NetConfig nc = azgo::loadConfigFile(preset).netConfig();
            if (nc.boardSize != n)
                throw std::runtime_error(preset + " has board_size " + std::to_string(nc.boardSize));
            for (const std::string& b : backends) {
                const double evals = measureInference(nc, b, seconds);
                const double stepMs = measureTrainStep(nc, b, seconds);
                char shape[32], size[16];
                std::snprintf(shape, sizeof shape, "%d x %d", nc.blocks, nc.filters);
                std::snprintf(size, sizeof size, "%dx%d", n, n);
                std::printf("%-6s %-17s %-8s %12.0f %14.1f\n", size, shape, b.c_str(), evals, stepMs);
                std::fflush(stdout);
            }
        }

        std::printf("\n%-6s %22s\n", "size", "random-playout moves/s");
        for (int n : sizes) {
            char size[16];
            std::snprintf(size, sizeof size, "%dx%d", n, n);
            std::printf("%-6s %22.0f\n", size, measurePlayouts(n, seconds));
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
```

- [ ] **Step 6: Add the three executables to CMake**

Append to the end of `CMakeLists.txt`:

```cmake
foreach(app arena analyze bench)
  add_executable(${app} apps/${app}.cpp)
  target_link_libraries(${app} PRIVATE azgo)
  target_include_directories(${app} PRIVATE ${CMAKE_SOURCE_DIR}/apps)
endforeach()
```

After Tasks 21 and 22, the block appended to `CMakeLists.txt` reads in full:

```cmake
# --- Apps (Tasks 21-22) -------------------------------------------------------
# apps/cli.hpp is header-only and lives next to the apps that use it rather
# than in include/azgo, so test_cli (created by the test glob above) needs
# apps/ on its include path.
target_include_directories(test_cli PRIVATE ${CMAKE_SOURCE_DIR}/apps)

add_executable(train apps/train.cpp)
target_link_libraries(train PRIVATE azgo)
target_include_directories(train PRIVATE ${CMAKE_SOURCE_DIR}/apps)

# End-to-end check of all four executables. ctest does not build targets:
# run `cmake --build build` first, as the usual workflow does.
add_test(NAME integration_smoke
         COMMAND bash ${CMAKE_SOURCE_DIR}/tests/integration_smoke.sh ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR})

foreach(app arena analyze bench)
  add_executable(${app} apps/${app}.cpp)
  target_link_libraries(${app} PRIVATE azgo)
  target_include_directories(${app} PRIVATE ${CMAKE_SOURCE_DIR}/apps)
endforeach()
```

Also replace the Task 1 placeholder comment (the line(s) under the `# ---- Apps ----` header, which stays) that says apps and the integration test are added in Tasks 21–22 with: `# Apps and the integration smoke test: see the block at the end of this file.`

- [ ] **Step 7: Build and run everything**

Run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: `100% tests passed`, including `integration_smoke`. Its log ends with:

```
== analyze: bad moves are user errors ==
  ok: error: move 2 (C3): illegal (occupied)
  ok: error: move 1 (Z99): not a move on a 5x5 board
  ok: error: move 1 (I5): not a move on a 5x5 board
  ok: error: move 5 (A1): illegal (suicide)
  ok: error: move 10 (C3): illegal (superko)
== analyze: other bad input ==
  ok: error: the SGF is for 9x9
  ok: error: move 3 (C3): game is over
  ok: error: the game is over
== bench ==
size   blocks x filters  backend       evals/s  train-step ms
5x5    ...
== train: resume must refuse an architecture change ==
integration_smoke: all checks passed
```

Then the BLAS variant: `cmake -S . -B build-blas -DCMAKE_BUILD_TYPE=Release -DAZ_USE_BLAS=ON && cmake --build build-blas -j && ctest --test-dir build-blas --output-on-failure`. Expected: all pass. In this build `bench` defaults to `--backend all` and prints one `naive` row and one `blas` row.

If a bad-move check fails with a signal (exit 134) instead of status 1, an assert fired. Most likely `analyze` called `Game::play` or `Board::at` on input it had not validated.

- [ ] **Step 8: Commit**

```bash
git add apps/arena.cpp apps/analyze.cpp apps/bench.cpp tests/integration_smoke.sh CMakeLists.txt
git commit -F - <<'EOF'
Add the arena, analyze and bench apps and the integration smoke test

arena plays two checkpoints for manual experiments. analyze puts the raw
network's value and policy heat map next to what MCTS makes of the same
position, which is the comparison the explainer builds on. bench backs
the preset sizes and the README's performance numbers with measurements.

The smoke test runs a tiny 5x5 training run through all four executables.
It also feeds analyze each kind of bad move (occupied, off-board, the
skipped column I, suicide, a superko retake) and requires a one-line
error with exit status 1. User input must never reach an assert.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 23: Tune presets from measurements and run the reference trainings

This task produces the measured results that the README and explainer report. It is operational rather than TDD: the "tests" are the spec's success criteria, checked with the project's own tools. **Report results honestly. If a criterion is not met, stop and report it with the data. Do not change the criterion, and do not retune until the numbers look good without saying so.** If a run fails or misbehaves (Elo flat or falling, loss NaN, games stuck at 2 plies), use superpowers:systematic-debugging before changing anything.

**Files:**
- Modify: `configs/5x5.cfg`, `configs/9x9.cfg`, `configs/19x19.cfg` (only the values that measurements justify)
- Create: `docs/results/README.md`, `docs/results/bench.txt`, `docs/results/5x5/` (`log.csv`, `elo.csv`, `results.csv`, `final-arena.txt`, `sample.sgf`), `docs/results/9x9/` (same plus `analyze-*.txt`), `docs/results/19x19/` (`log.csv`, `timing.txt`)

**Interfaces:**
- Consumes: the `train`, `arena`, `analyze` and `bench` executables (Tasks 21–22), and the preset files (Task 16).
- Produces: tuned presets, plus the result files under `docs/results/` that Tasks 24–25 quote.

- [ ] **Step 1: Build both variants and run the full test suite**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure
cmake -S . -B build-blas -DCMAKE_BUILD_TYPE=Release -DAZ_USE_BLAS=ON && cmake --build build-blas -j && ctest --test-dir build-blas --output-on-failure
```

Expected: every test passes in both builds, including `integration_smoke`.

- [ ] **Step 2: Measure throughput**

```bash
mkdir -p docs/results
./build-blas/bench --sizes 5,9,19 --backend all --seconds 3 --configs configs | tee docs/results/bench.txt
sysctl -n machdep.cpu.brand_string hw.ncpu hw.memsize >> docs/results/bench.txt
```

Expected: a table with evaluations per second and training-step time for each size and backend, plus board moves per second.

- [ ] **Step 3: Estimate how long a generation takes and retune the presets**

For each preset, estimate the self-play time per generation:

`games_per_generation × avg_plies × simulations / (evals_per_sec_batch8 × workers)`

- Use `avg_plies ≈ 1.2·n²` as a first guess: 30 for 5×5, 100 for 9×9, 430 for 19×19.
- `evals_per_sec_batch8` is from Step 2 for the faster backend.
- `workers` is the core count.

Aim for these self-play times per generation:
- **5×5:** at most about 4 minutes.
- **9×9:** at most about 12 minutes, so a 10-hour night gives 40 or more generations.

Change settings in this order until the estimate fits:
1. Set `backend = blas` if BLAS is faster.
2. Lower `simulations`, but not below 100 on 5×5 or 9×9.
3. Lower `games_per_generation`, but not below 50.
4. Shrink `filters`/`blocks`, with 9×9 no smaller than 4 × 48.

Keep `train_steps × batch_size` such that the replay ratio (`train_steps × batch_size / (games × avg_plies)`) stays between 2 and 8. Record every change and its reason in a comment in the `.cfg` file, for example `# simulations 200 -> 160: bench 2026-10-02, 9x9 selfplay 15.1 -> 12.0 min/gen`.

Run `ctest --test-dir build --output-on-failure -R test_config` (the presets must still load), then commit:

```bash
git add configs docs/results/bench.txt
git commit -F - <<'EOF'
Tune presets from bench measurements

bench on this machine (docs/results/bench.txt) puts <numbers>; the
presets are sized so a 5x5 generation takes ~<m> min and a 9x9
generation ~<m> min of self-play. <List each changed key and why.>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

- [ ] **Step 4: Run the 5×5 reference training (two seeds)**

```bash
./build-blas/train --config configs/5x5.cfg --run runs/5x5-seed1 > runs/5x5-seed1.out 2>&1 &
./build-blas/train --config configs/5x5.cfg --run runs/5x5-seed2 --set seed=2 --set workers=5 > runs/5x5-seed2.out 2>&1 &
```

(If running both at once would slow them too much, run them one after the other with `workers=0`.) Watch `runs/*/log.csv` about every 30 minutes and look for:
- average game length growing beyond 2 plies after the first few generations
- policy loss falling
- `elo` rising

Use `runs/<name>.out` for the progress lines. When both runs finish:

```bash
./build-blas/arena --a runs/5x5-seed1/checkpoints/gen_0060.azn --b runs/5x5-seed1/checkpoints/gen_0000.azn --games 100 --simulations 200 | tee docs/results/5x5/final-arena.txt
./build-blas/arena --a runs/5x5-seed2/checkpoints/gen_0060.azn --b runs/5x5-seed2/checkpoints/gen_0000.azn --games 100 --simulations 200 | tee -a docs/results/5x5/final-arena.txt
cp runs/5x5-seed1/{log.csv,elo.csv,results.csv} docs/results/5x5/
cp "$(ls runs/5x5-seed1/sgf/*.sgf | tail -1)" docs/results/5x5/sample.sgf
```

(Use the final generation number from `log.csv` if the preset's `generations` changed.) **Success criterion:** each run's final network scores at least 90% against generation 0, and seed 1's Elo curve rises clearly, with the last 10 generations' mean Elo above the first 10 generations' mean by at least 200. If only one of the two seeds meets it, report that as "1 of 2 runs".

- [ ] **Step 5: Run the 9×9 overnight training**

```bash
./build-blas/train --config configs/9x9.cfg --run runs/9x9-main > runs/9x9-main.out 2>&1 &
```

Let it run for 8–12 hours. `train --run runs/9x9-main --resume` continues it after an interruption. Then:

```bash
G=$(tail -1 runs/9x9-main/log.csv | cut -d, -f1)
cp runs/9x9-main/{log.csv,elo.csv,results.csv} docs/results/9x9/
cp "$(ls runs/9x9-main/sgf/*.sgf | tail -1)" docs/results/9x9/sample.sgf
```

Then run `analyze` on the final checkpoint for three positions, saving each to `docs/results/9x9/analyze-<name>.txt`. Build each move list by hand on a 9×9 board, check it with `analyze` on generation 0 first (it must be accepted), and describe it in the results README:
- **`atari`:** a White stone or group in atari that Black can capture on the move. It should be the network's top-visited move.
- **`eye`:** a Black group that needs one move to make its second eye. The eye-making point should be among the top 3 by visits.
- **`opening`:** the empty board. Just record the policy; there is no pass/fail.

**Success criterion:** Elo is still rising over the last third of the run (a linear fit over those generations has a positive slope), and the atari and eye checks pass. Report the slope and the ranks of the checked moves.

- [ ] **Step 6: Check 19×19 training**

```bash
./build-blas/train --config configs/19x19.cfg --run runs/19x19-check --generations 2 > runs/19x19-check.out 2>&1
cp runs/19x19-check/log.csv docs/results/19x19/
grep -E "gen [12] " runs/19x19-check.out > docs/results/19x19/timing.txt
```

**Success criterion:** both generations complete, the losses are finite, and the checkpoints load in `analyze`. Record the time per generation. Make no claims about strength.

- [ ] **Step 7: Write `docs/results/README.md` and commit**

`docs/results/README.md` contains:
- the machine (from bench.txt)
- the preset values used
- for each board size, a table of generations, wall time, final Elo, final win rate vs generation 0 (with the 95% interval printed by `arena`), and average game length at the start and at the end
- the 9×9 analyze checks, with the ranks of the checked moves
- the 19×19 timings
- a "What did not work" list of any criterion missed, with the data

Commit:

```bash
git add docs/results
git commit -F - <<'EOF'
Record measured 5x5, 9x9 and 19x19 training results

<One paragraph per board size with the headline numbers, and any
criterion that was missed.>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 24: Explainer (Markdown) and citation checker

**Files:**
- Create: `tools/check_citations.py`, `docs/algorithm-explained.md`

**Interfaces:**
- Consumes: the source tree (every `src/*.cpp`, `include/azgo/*.hpp`, `apps/*.cpp`), `docs/results/` (Task 23), and the tic-tac-toe explainer and checker as models: `~/projects/alphazero-tictactoe/docs/algorithm-explained.md` and `~/projects/alphazero-tictactoe/tools/check_citations.py`.
- Produces: the Markdown explainer whose chapter IDs and captions Task 25's HTML mirrors, and `tools/check_citations.py` with the caption formats below.

- [ ] **Step 1: Port the citation checker**

Copy `~/projects/alphazero-tictactoe/tools/check_citations.py` to `tools/check_citations.py` and change:
- the path regexes, so they accept `src/…`, `include/azgo/…` and `apps/…` paths ending in `.cpp` or `.hpp` (`[\w./]+` already allows these; check that it does)
- `MD`/`HTML` constants, which stay `docs/algorithm-explained.{md,html}`
- the docstring, updated for this project

If `docs/algorithm-explained.html` doesn't exist yet, the HTML checks are skipped with the note `html: not present, skipped`. Add that guard: `if HTML.exists()` around the HTML checks.

Keep the three checks and `--fix`:
- a Markdown caption `*src/foo.cpp:NN — `Sym`*` must name the line where `Sym` is defined
- an HTML caption must agree with the Markdown caption for the same symbol
- the HTML concept map `<span class="loc">` line numbers must be definition lines

- [ ] **Step 2: Check that the checker catches a stale citation**

Create a throwaway `docs/algorithm-explained.md` containing one correct caption for `Board::play` (look up its real definition line with `grep -n "int Board::play" src/board.cpp`) and one caption with a wrong line number. Then:

```bash
python3 tools/check_citations.py; echo "exit $?"
```

Expected: the wrong caption is reported and the exit code is non-zero. Then run `python3 tools/check_citations.py --fix && python3 tools/check_citations.py; echo "exit $?"`. Expected: the numbers are rewritten and the exit code is 0.

- [ ] **Step 3: Write `docs/algorithm-explained.md`**

Follow the tic-tac-toe explainer's voice and structure:
- title, a one-paragraph promise, a contents list, numbered `## NN — Title` chapters
- each chapter opens with the idea in plain words, then quotes the real code that implements it (short excerpts, 5–30 lines, each captioned `*path:line — `Symbol`*` on its own line after the block), then explains what the code does and why
- where useful, a "Try it" block with a real command (`analyze`, `arena`, `train --set …`) and its actual output, copied from a real run, never invented
- about 900–1400 lines in total

Chapters and what each must cover:
- **00 — The state:** the Go rules a programmer needs, with small ASCII boards:
  - liberties, capture, suicide (and why AlphaGo Zero made it illegal)
  - ko and positional superko, and Zobrist hashing (`Board::hashAfter`, `Game::isLegal`)
  - Tromp-Taylor area scoring and komi (`Board::areaScore`)
  - the padded 21×21 array and the "Off" sentinel
  - input planes (`encodePlanes`): own/opponent history and the two color planes, and why an all-ones plane lets zero-padded convolutions see the edge
- **01 — The two-headed guess:**
  - from tic-tac-toe's 18→64→64 MLP to a ResNet
  - convolution as im2col + gemm (`Conv2d::forward`, `Backend::im2col`) with a worked 3×3 example
  - the channel-major layout, and why it makes the conv one matrix multiply and batch norm contiguous
  - batch norm: training vs. inference statistics (`BatchNorm2d::forwardTrain`/`forwardInfer`), and the bug you get if you mix them up
  - residual blocks and why depth needs them
  - the policy and value heads, and what `blocks` and `filters` buy (cite the bench numbers for cost)
  - backprop through conv, batch norm and linear layers; the finite-difference gradient check (`tests/test_layers.cpp`)
  - the loss (`Network::trainStep`)
- **02 — Search corrects the guess:**
  - PUCT (quote the selection code)
  - legal masking and softmax over legal moves only
  - superko inside the tree
  - backup and the sign flip at each ply, including after a pass
  - terminal nodes scored by the rules
- **03 — Batching the search:**
  - virtual loss, and collisions ending a round
  - why batch size matters for gemm efficiency: bench numbers at batch 1 vs. 8, measured with `bench` or a one-off timing, stated as measured
  - the `Evaluator` interface and where a GPU batching evaluator and `MetalBackend` would plug in (the spec's Extension Path)
- **04 — Self-play and exploration:**
  - Dirichlet noise and why α scales as 10/n² (0.4 on 5×5, 0.03 on 19×19)
  - temperature and the tic-tac-toe lesson (sampling too briefly made learning plateau)
  - tree reuse and compaction, and why it bounds memory
- **05 — Assigning credit:**
  - z from the final score, and komi
  - passing, and how the two-pass ending works
  - why Tromp-Taylor means dead stones must actually be captured, and what that does to game length (quote avg plies from `log.csv`)
  - the integer-komi draw case
- **06 — Learning from itself:**
  - the replay buffer storing whole games; history reconstruction (`ReplayBuffer::encodeSample`)
  - symmetry augmentation, and why it's valid in Go but wasn't in AlphaZero's chess
  - replay ratio
  - SGD with momentum and weight decay (`SgdMomentum::step`), and the LR schedule
- **07 — Measuring progress without an oracle:**
  - arena, opening diversity and distinct games
  - Bradley-Terry vs. incremental Elo, anchoring, and the virtual draw
  - drift and the ladder of opponents; non-transitive results
  - Wilson intervals: how noisy 20 games are, with actual numbers
  - gating (AlphaGo Zero) vs. no gating (AlphaZero)
- **08 — The full loop at scale:**
  - one generation (`Trainer::runGeneration`)
  - deterministic threading (seed derivation, results stored by index, row-split gemm)
  - crash safety and resume
  - compute budgets: the measured 5×5, 9×9 and 19×19 generation times from `docs/results`, and an honest comparison with AlphaGo Zero's 19×19 run (about 29 million self-play games on thousands of TPUs), including an estimate of how many years this machine would take
- **09 — What changed from tic-tac-toe:** a table of component → tic-tac-toe → Go, and why.
- **10 — Concept map:** a list of concepts, each with the symbols and `file:line` that implement it, like the tic-tac-toe concept map.

Results quoted anywhere (Elo, win rates, timings, game lengths) come from `docs/results/`, with the file named.

- [ ] **Step 4: Check citations and commands**

```bash
python3 tools/check_citations.py; echo "exit $?"
```

Expected: `exit 0`. Then run every "Try it" command in the document against `runs/` or `docs/results/` checkpoints, and confirm the output shown matches in shape (numbers may differ slightly only where the text says they were measured on a named run).

- [ ] **Step 5: Commit**

```bash
git add tools/check_citations.py docs/algorithm-explained.md
git commit -F - <<'EOF'
Add the AlphaZero-for-Go explainer and its citation checker

The explainer walks through rules, network, search, self-play,
training and evaluation, quoting the real code; check_citations.py
(ported from alphazero-tictactoe) keeps every file:line caption honest
as the code moves.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 25: Illustrated HTML explainer, Elo plots and the finished README

**Files:**
- Create: `tools/plot_elo.py`, `docs/algorithm-explained.html`
- Modify: `README.md`

**Interfaces:**
- Consumes: `docs/algorithm-explained.md` and its captions (Task 24), `docs/results/*/elo.csv` (Task 23), `tools/check_citations.py` (Task 24), and the tic-tac-toe HTML as the visual model: `~/projects/alphazero-tictactoe/docs/algorithm-explained.html`.
- Produces: the final documentation set.

- [ ] **Step 1: Write `tools/plot_elo.py`**

This is standard-library Python. `python3 tools/plot_elo.py docs/results/5x5/elo.csv [more.csv ...] > out.svg` writes a self-contained SVG line chart:
- x axis: generation; y axis: Elo; one polyline per CSV file, labelled with its parent directory name
- "nice" ticks (1, 2 or 5 × 10^k) on both axes
- colors as CSS variables with fallbacks, `stroke="var(--x, #2C5FA8)"` and `var(--o, #BD5A26)`, so the chart follows the HTML page's light and dark themes
- `viewBox` 640×360, no fixed width, so it scales on phones

It skips rows whose `elo` field is empty, and exits 1 with a message if a file has no rows. Check it:

```bash
python3 tools/plot_elo.py docs/results/5x5/elo.csv docs/results/9x9/elo.csv > /tmp/elo.svg && head -c 300 /tmp/elo.svg
```

Expected: output starting with `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 360"`.

- [ ] **Step 2: Write `docs/algorithm-explained.html`**

Follow the tic-tac-toe HTML's design system:
- the same CSS custom properties, with the light/dark handling (`prefers-color-scheme` guarded by `:root:not([data-theme="light"])`, plus `:root[data-theme="dark"]`)
- the same fonts (IBM Plex via Google Fonts) and page structure
- for the two-player colors, `--x` renamed `--black` and `--o` renamed `--white`, keeping the same values

Content mirrors the Markdown chapter for chapter. Code excerpts use `<span class="src">path:line — <code>Symbol</code></span>` captions identical to the Markdown. The concept map uses `<code>Sym</code> <span class="loc">— path:NN</span>` entries.

Add inline SVG diagrams, each theme-aware through CSS variables and readable at phone width:
- a board with liberties marked, and a capture before and after (00)
- a ko shape (00)
- im2col unfolding a 3×3 patch into a column (01)
- a residual block and the two heads (01)
- one PUCT descent with virtual loss marking the path (03)
- the generation loop (08)

Embed the Elo chart(s) from `plot_elo.py` in chapter 07 or 08, generated from `docs/results`. The page must work at 360px width with no horizontal page scroll; code blocks scroll inside themselves.

- [ ] **Step 3: Check citations across both documents**

```bash
python3 tools/check_citations.py; echo "exit $?"
```

Expected: `exit 0` with the HTML checks running (not skipped). Open the HTML in a browser (`open docs/algorithm-explained.html`) and check light mode, dark mode (by system setting) and a narrow window.

- [ ] **Step 4: Finish `README.md`**

Keep the existing title, purpose and Authorship sections. Add:
- **Build:** both builds, test commands, and the number of tests from `ctest -N | tail -1`
- **Usage:** `train` (new run, resume, `--set`, `--generations`), `arena`, `analyze`, `bench`, each with one real example command and a short real output excerpt
- **Presets:** a table of board size → blocks × filters, simulations, games per generation, measured time per generation (from `docs/results`)
- **Results:** headline numbers from `docs/results/README.md` (5×5 win rate vs. generation 0 with its interval and "N of 2 runs", 9×9 Elo trend and analyze checks, 19×19 time per generation), plus the embedded Elo chart (`docs/results/elo.svg`, written by `plot_elo.py` and committed)
- **Learn how it works:** links to the Markdown and HTML explainers, and to the spec and plan
- **Extending to the GPU:** three sentences from the spec's Extension Path
- **Authorship:** as before

Every number in the README comes from `docs/results/`.

- [ ] **Step 5: Final verification**

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure
cmake --build build-blas -j && ctest --test-dir build-blas --output-on-failure
python3 tools/check_citations.py
git status --short
```

Expected: all tests pass in both builds, citations pass, and only the intended files are changed.

- [ ] **Step 6: Commit**

```bash
python3 tools/plot_elo.py docs/results/5x5/elo.csv docs/results/9x9/elo.csv > docs/results/elo.svg
git add tools/plot_elo.py docs/algorithm-explained.html docs/results/elo.svg README.md
git commit -F - <<'EOF'
Add the illustrated HTML explainer, Elo charts and complete README

The HTML mirrors the Markdown explainer with theme-aware diagrams and
the measured Elo curves; the README now documents build, usage,
presets and the measured results.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

## Appendix: Interface Contract

> Note (2026-10-01): `MctsEdge` gained `float unvisitedQ` (the exact value of a
> game-ending pass edge, see `src/mcts.cpp`) and `GenerationStats` gained
> `int distinctVsPrev` (log.csv column `distinct_vs_prev`). The headers below
> are otherwise as originally planned.

Every public header in the project, in final form. Plan tasks copy these
verbatim into their "create header" steps. Private members marked
"implementation-defined" are chosen by the owning task. Naming: types
PascalCase, functions/methods camelCase, members trailing underscore,
constants kPascalCase, namespace `azgo`. Each header opens with a
teaching comment block (a few lines explaining the concept, not the API).

Conventions that cross modules:
- A **point** is `row * n + col`, row 0 is the TOP row. An **action** is a
  point in `[0, n*n)` or the pass action `n*n`.
- GTP coordinates: columns `A..T` skipping `I`; row label `n - row`
  (so `A1` is bottom-left). `"pass"` is the pass action.
- Values are always from the **side to move's** perspective, in [-1, 1].
- Activation layout inside the network is **channel-major** `[C][B][n*n]`;
  the network's external input is sample-major `[B][planes][n*n]`, exactly
  what `encodePlanes` writes for one sample.
- All randomness goes through `azgo::Rng`; never `std::random_device`,
  `std::mt19937` or `std::*_distribution` (implementation-defined output).
- Seeds for independent streams: `deriveSeed(runSeed, generation, phase, index)`.

---

### include/azgo/rng.hpp  (Task 1)

```cpp
#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace azgo {

// Advances x and returns the next splitmix64 output.
uint64_t splitmix64(uint64_t& x);

// Mixes up to four integers into one well-distributed 64-bit seed.
uint64_t deriveSeed(uint64_t a, uint64_t b = 0, uint64_t c = 0, uint64_t d = 0);

// xoshiro256** generator, seeded by running splitmix64 on `seed`.
class Rng {
public:
    explicit Rng(uint64_t seed = 0);
    uint64_t next();
    double uniform();              // [0, 1), 53 random bits
    int below(int n);              // uniform in [0, n); asserts n > 0; unbiased (rejection)
    double normal();               // standard normal via Box-Muller (no cached second value)
    double gamma(double shape);    // Gamma(shape, 1); Marsaglia-Tsang, shape < 1 via boost
    std::array<uint64_t, 4> state() const;
    void setState(const std::array<uint64_t, 4>& s);
private:
    std::array<uint64_t, 4> s_;
};

// Symmetric Dirichlet sample of length n with concentration alpha > 0.
std::vector<double> dirichlet(Rng& rng, double alpha, int n);

} // namespace azgo
```

### include/azgo/board.hpp  (Task 2)

```cpp
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace azgo {

constexpr int kMinBoardSize = 5;
constexpr int kMaxBoardSize = 19;
constexpr int kStride = kMaxBoardSize + 2;   // padded row length: one sentinel column each side
constexpr int kCells = kStride * kStride;    // padded array size (441)

enum class Color : uint8_t { Empty = 0, Black = 1, White = 2, Off = 3 };

Color opponent(Color c);   // Black <-> White; asserts c is Black or White

class Board {
public:
    explicit Board(int n);  // empty board, Black to move; asserts kMinBoardSize <= n <= kMaxBoardSize

    int size() const { return n_; }
    int area() const { return n_ * n_; }
    int passAction() const { return n_ * n_; }
    int numActions() const { return n_ * n_ + 1; }
    Color toMove() const { return toMove_; }

    Color at(int point) const;          // asserts 0 <= point < area()
    int stoneCount(Color c) const;
    int libertiesAt(int point) const;   // liberties of the group containing point; 0 if empty

    // Local legality: pass is always legal; a point must be empty and the move
    // must not be suicide (after removing captured opponent groups, the new
    // stone's group must have a liberty). Superko is NOT checked here — it
    // needs history, see Game::isLegal and Mcts.
    bool isLegal(int action) const;
    // Plays a locally legal action (asserted), removes captured groups, flips
    // toMove. Pass only flips toMove. Returns the number of stones captured.
    int play(int action);
    // Hash of the position play(action) would produce. O(1) when the move
    // captures nothing; otherwise copies the board and plays.
    uint64_t hashAfter(int action) const;

    uint64_t hash() const { return hash_; }  // Zobrist over stones only (positional superko)
    uint64_t computeHashFromScratch() const; // test oracle for the incremental hash

    // Tromp-Taylor area score: (Black stones + empty regions reaching only Black)
    // minus (White stones + empty regions reaching only White). No komi.
    int areaScore() const;

    void exportStones(uint8_t* out) const;   // area() bytes of Color values, row-major
    std::string toString() const;            // ASCII diagram with GTP coordinates

    // Setup helpers for tests and tools: no capture logic; hash stays valid.
    void setStone(int point, Color c);       // c is Empty, Black or White
    void setToMove(Color c);

    bool operator==(const Board& o) const;   // size, stones and side to move

private:
    int pad(int point) const;                // point -> padded index
    int n_;
    Color toMove_;
    std::array<Color, kCells> cells_;
    uint64_t hash_;
    // Additional private helpers are implementation-defined (Task 2).
};

// Rows like "X.O.." ('X' black, 'O' white, '.' empty), row 0 first (top).
// Asserts rows form an n x n square with n in range.
Board boardFromRows(const std::vector<std::string>& rows, Color toMove);

// GTP coordinates. parseAction is case-insensitive and returns -1 for
// malformed or off-board input.
std::string actionToString(int action, int n);
int parseAction(const std::string& text, int n);

} // namespace azgo
```

### include/azgo/game.hpp  (Task 3)

```cpp
#pragma once
#include <unordered_set>
#include <vector>
#include "azgo/board.hpp"

namespace azgo {

// A game in progress: the sequence of positions, for history planes and
// positional superko, plus the pass count and the move cap.
class Game {
public:
    Game(int n, float komi, int maxMoves);
    Game(const Board& start, float komi, int maxMoves);

    int size() const { return boards_.front().size(); }
    float komi() const { return komi_; }
    int maxMoves() const { return maxMoves_; }

    const Board& board() const { return boards_.back(); }
    const Board& boardAt(int ply) const;          // 0 = start; asserts ply <= moveCount()
    int moveCount() const { return static_cast<int>(moves_.size()); }
    const std::vector<int>& moves() const { return moves_; }
    int consecutivePasses() const { return passes_; }

    // Board::isLegal plus positional superko: a non-pass move may not
    // recreate any earlier position of this game. Always false once terminal.
    bool isLegal(int action) const;
    std::vector<int> legalActions() const;        // ascending; always contains pass unless terminal
    void play(int action);                        // asserts isLegal(action)
    bool seen(uint64_t positionHash) const;       // position occurred in this game

    // Two consecutive passes, or moveCount() >= maxMoves().
    bool isTerminal() const;
    float scoreForBlack() const;                  // board().areaScore() - komi
    float resultForBlack() const;                 // +1 Black wins, -1 White wins, 0 exact tie

private:
    std::vector<Board> boards_;                   // boards_[k] = position after k moves
    std::vector<int> moves_;
    std::unordered_set<uint64_t> seen_;
    int passes_ = 0;
    float komi_;
    int maxMoves_;
};

// 2 * n * n, the default move cap.
int defaultMaxMoves(int n);

} // namespace azgo
```

### include/azgo/symmetry.hpp  (Task 4)

```cpp
#pragma once

namespace azgo {

constexpr int kNumSymmetries = 8;

// Symmetry s in [0, 8): s & 3 = number of clockwise quarter turns, s & 4 = mirror
// left-right first. 0 is the identity.
int transformPoint(int sym, int point, int n);
int transformAction(int sym, int action, int n);  // pass (n*n) maps to itself
int inverseSymmetry(int sym);                      // transformPoint(inverse(s), transformPoint(s, p)) == p

} // namespace azgo
```

### include/azgo/encode.hpp  (Task 4)

```cpp
#pragma once
#include <cstdint>
#include <vector>
#include "azgo/board.hpp"

namespace azgo {

class Game;

// 2H + 2: (own, opponent) stones for each of the last H positions, then a
// "black to move" plane and a "white to move" plane (exactly one is all ones).
int numPlanes(int history);

// stones[k] points to n*n Color bytes for the position k plies ago (k = 0 is
// current). stones.size() may be < H; missing positions are zero planes.
// "Own" means toMove (the side to move NOW) for every history plane.
// Writes numPlanes(H) * n * n floats to out ([plane][point]) with symmetry
// `sym` applied: value for point p lands at transformPoint(sym, p, n).
void encodePlanes(const std::vector<const uint8_t*>& stones, int n, Color toMove,
                  int H, int sym, float* out);

// history[0] is the current board, history[1] the previous one, ...
void encodeBoards(const std::vector<const Board*>& history, int H, int sym, float* out);

// Current position of a game, history taken from the game's earlier boards.
void encodeGame(const Game& game, int H, int sym, float* out);

} // namespace azgo
```

### include/azgo/backend.hpp  (Task 5)

```cpp
#pragma once
#include <memory>
#include <string>

namespace azgo {

// Where the arithmetic happens. Layers call only these operations, so a
// different backend (BLAS today, Metal on the GPU later) needs no layer changes.
class Backend {
public:
    virtual ~Backend() = default;
    virtual const char* name() const = 0;

    // Row-major C[M x N] = alpha * op(A) * op(B) + beta * C, tightly packed.
    // op(A) is M x K (A is stored K x M when transA); op(B) is K x N
    // (B is stored N x K when transB). beta == 0 means C is overwritten.
    virtual void gemm(bool transA, bool transB, int M, int N, int K, float alpha,
                      const float* A, const float* B, float beta, float* C) const = 0;

    // 'Same'-padded k x k patch unfolding (k odd). in: [C][B][n*n];
    // col: [C*k*k][B*n*n], row index (c*k + dy)*k + dx.
    virtual void im2col(const float* in, int C, int B, int n, int k, float* col) const;
    // Adjoint of im2col: OVERWRITES in with the sum of contributions.
    virtual void col2im(const float* col, int C, int B, int n, int k, float* in) const;
};

// Blocked loops, packing transposed operands first. threads > 1 splits the
// rows of C across std::threads (each element computed by exactly one thread,
// so results do not depend on the thread count).
class NaiveBackend : public Backend {
public:
    explicit NaiveBackend(int threads = 1);
    const char* name() const override { return "naive"; }
    void gemm(bool transA, bool transB, int M, int N, int K, float alpha,
              const float* A, const float* B, float beta, float* C) const override;
private:
    int threads_;
};

bool blasAvailable();  // true when built with -DAZ_USE_BLAS=ON

// "naive" or "blas". threads applies to naive only; BLAS is kept
// single-threaded (our worker threads supply the parallelism).
// Throws std::invalid_argument for an unknown name or unavailable BLAS.
std::unique_ptr<Backend> makeBackend(const std::string& name, int threads = 1);

} // namespace azgo
```

(`BlasBackend` is declared only in `src/backend_blas.cpp`, compiled when
`AZ_USE_BLAS` is defined; `makeBackend` is the only way to get one.)

### include/azgo/layers.hpp  (Task 6)

```cpp
#pragma once
#include <cstddef>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/rng.hpp"

namespace azgo {

// All activations are channel-major: [channels][M] with M = B * n * n.

struct Conv2d {
    int cin = 0, cout = 0, k = 1;        // k is 1 or 3; 'same' padding; no bias (BN follows)
    std::vector<float> w, gw;            // [cout][cin * k * k]
    Conv2d() = default;
    Conv2d(int cin, int cout, int k);
    void init(Rng& rng);                 // He normal: stddev sqrt(2 / (cin * k * k))
    size_t colSize(int B, int n) const;  // floats needed for the im2col buffer (0 when k == 1)
    // x [cin][M] -> y [cout][M]. col is scratch of colSize(B, n) (unused if k == 1) and
    // must be passed unchanged to backward.
    void forward(const Backend& be, const float* x, int B, int n, float* col, float* y) const;
    // Accumulates gw += dW. Writes dx (overwrite) unless dx == nullptr.
    // dcol is scratch of colSize(B, n).
    void backward(const Backend& be, const float* x, const float* col, const float* dy,
                  int B, int n, float* dcol, float* dx);
};

struct BatchNorm2d {
    static constexpr float kEps = 1e-5f;
    static constexpr float kMomentum = 0.1f;   // running = (1 - m) * running + m * batch
    int c = 0;
    std::vector<float> gamma, beta, ggamma, gbeta, runMean, runVar;  // gamma=1, beta=0, runVar=1 initially
    BatchNorm2d() = default;
    explicit BatchNorm2d(int c);
    // y = gamma * (x - runMean) / sqrt(runVar + eps) + beta
    void forwardInfer(const float* x, int M, float* y) const;
    // Batch statistics (biased variance). Saves xhat [c][M] and invStd [c].
    // If updateRunning, updates runMean and runVar (unbiased variance M/(M-1)).
    void forwardTrain(const float* x, int M, float* y, float* xhat, float* invStd,
                      bool updateRunning);
    // Accumulates ggamma, gbeta; writes dx (overwrite).
    void backward(const float* dy, const float* xhat, const float* invStd, int M, float* dx);
};

struct Linear {
    int in = 0, out = 0;
    std::vector<float> w, b, gw, gb;     // w [out][in], b [out]
    Linear() = default;
    Linear(int in, int out);
    void init(Rng& rng);                 // He normal weights, zero bias
    void forward(const Backend& be, const float* x, int B, float* y) const;   // x [B][in], y [B][out]
    // Accumulates gw, gb. Writes dx (overwrite) unless dx == nullptr.
    void backward(const Backend& be, const float* x, const float* dy, int B, float* dx);
};

void reluForward(float* x, size_t count);                  // in place
void reluBackward(const float* y, float* dy, size_t count); // dy *= (y > 0), y = relu output

} // namespace azgo
```

### include/azgo/network.hpp  (Task 7)

```cpp
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/layers.hpp"

namespace azgo {

struct NetConfig {
    int boardSize = 9;
    int history = 8;       // H
    int blocks = 6;        // residual blocks, >= 0
    int filters = 64;
    int valueHidden = 64;
    int planes() const { return 2 * history + 2; }
    int area() const { return boardSize * boardSize; }
    int actions() const { return area() + 1; }
    bool operator==(const NetConfig& o) const;
    bool operator!=(const NetConfig& o) const { return !(*this == o); }
};

struct ParamView  { std::string name; float* value; float* grad; size_t size; bool decay; };
struct BufferView { std::string name; float* value; size_t size; };

struct LossStats { float policyLoss = 0; float valueLoss = 0; };  // batch means

// Per-thread activation and scratch storage for up to maxBatch samples.
class Workspace {
public:
    Workspace(const NetConfig& cfg, int maxBatch, const Backend& backend);
    int maxBatch() const { return maxBatch_; }
    const Backend& backend() const { return *backend_; }
private:
    friend class Network;
    NetConfig cfg_;
    int maxBatch_;
    const Backend* backend_;
    // Activation buffers: implementation-defined (Task 7).
};

class Network {
public:
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
    // Layers (stem, residual blocks, policy and value heads): implementation-defined (Task 7).
};

} // namespace azgo
```

### include/azgo/optimizer.hpp  (Task 8)

```cpp
#pragma once
#include <string>
#include <utility>
#include <vector>
#include "azgo/network.hpp"

namespace azgo {

// SGD with momentum and L2 weight decay:
//   g' = grad + weightDecay * w   (only for params with decay == true)
//   v  = momentum * v + g'
//   w -= lr * v
class SgdMomentum {
public:
    SgdMomentum(Network& net, float momentum, float weightDecay);
    void step(Network& net, float lr);
    std::vector<std::vector<float>>& velocity() { return velocity_; }   // one per param, params() order
    const std::vector<std::vector<float>>& velocity() const { return velocity_; }
private:
    float momentum_, weightDecay_;
    std::vector<std::vector<float>> velocity_;
};

// Stepped learning-rate schedule keyed on generation, e.g. "0:0.02,50:0.002":
// lr = value of the last entry whose generation <= g.
class LrSchedule {
public:
    // Throws std::invalid_argument: empty, malformed, first entry not at 0,
    // generations not strictly increasing, or lr <= 0.
    static LrSchedule parse(const std::string& text);
    float at(int generation) const;
    std::string toString() const;   // canonical "g:lr,g:lr" form
private:
    std::vector<std::pair<int, float>> steps_;
};

} // namespace azgo
```

### include/azgo/serialize.hpp  (Task 9)

```cpp
#pragma once
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>

namespace azgo {

// Little-endian binary I/O, independent of host byte order.
class BinaryWriter {
public:
    explicit BinaryWriter(std::ostream& os);
    void u8(uint8_t v);
    void u32(uint32_t v);
    void u64(uint64_t v);
    void i32(int32_t v);
    void f32(float v);
    void f64(double v);
    void str(const std::string& s);           // u32 length + bytes
    void floats(const float* p, size_t n);
    void bytes(const void* p, size_t n);
};

// Every read throws std::runtime_error on premature end of stream.
class BinaryReader {
public:
    explicit BinaryReader(std::istream& is);
    uint8_t u8();
    uint32_t u32();
    uint64_t u64();
    int32_t i32();
    float f32();
    double f64();
    std::string str();
    void floats(float* p, size_t n);
    void bytes(void* p, size_t n);
};

// Writes to path + ".tmp", flushes, then renames over path (atomic replace).
// Throws std::runtime_error on I/O failure.
void writeFileAtomically(const std::string& path, const std::string& contents);

} // namespace azgo
```

(Private stream members of `BinaryWriter`/`BinaryReader` are implementation-defined.)

### include/azgo/checkpoint.hpp  (Task 9)

```cpp
#pragma once
#include <string>
#include "azgo/network.hpp"

namespace azgo {

// Format: magic "AZGO" (u32 0x4F475A41), version u32 = 1, NetConfig as 5 x i32
// (boardSize, history, blocks, filters, valueHidden), u32 tensor count, then
// per tensor: name (str), u64 size, floats. Params in params() order, then
// buffers in buffers() order. Saved atomically.
void saveNetwork(Network& net, const std::string& path);

// Throws std::runtime_error naming the problem: missing file, bad magic,
// unsupported version, tensor name/size mismatch, truncation.
Network loadNetwork(const std::string& path);
NetConfig readCheckpointConfig(const std::string& path);

} // namespace azgo
```

### include/azgo/evaluator.hpp  (Task 10)

```cpp
#pragma once
#include <memory>
#include "azgo/network.hpp"

namespace azgo {

// What MCTS asks of "the network": a batch of encoded positions in, policy
// logits and values out. A future GPU evaluator batching across threads
// would implement this same interface.
class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual int boardSize() const = 0;
    virtual int history() const = 0;
    virtual int maxBatch() const = 0;
    // input [B][numPlanes(history())][n*n]; logits [B][n*n+1]; values [B].
    virtual void evaluate(const float* input, int B, float* logits, float* values) = 0;
};

// One per worker thread: shares a read-only Network, owns a Workspace.
class LocalEvaluator : public Evaluator {
public:
    LocalEvaluator(const Network& net, const Backend& backend, int maxBatch);
    int boardSize() const override;
    int history() const override;
    int maxBatch() const override;
    void evaluate(const float* input, int B, float* logits, float* values) override;
    long long evaluations() const { return evaluations_; }   // positions evaluated so far
private:
    const Network& net_;
    Workspace ws_;
    long long evaluations_ = 0;
};

// Uniform priors (all logits 0) and a constant value. For tests and bench.
class UniformEvaluator : public Evaluator {
public:
    UniformEvaluator(int n, int history, float value = 0.0f, int maxBatch = 64);
    int boardSize() const override { return n_; }
    int history() const override { return history_; }
    int maxBatch() const override { return maxBatch_; }
    void evaluate(const float* input, int B, float* logits, float* values) override;
    long long calls() const { return calls_; }               // evaluate() invocations
private:
    int n_, history_, maxBatch_;
    float value_;
    long long calls_ = 0;
};

} // namespace azgo
```

### include/azgo/mcts.hpp  (Tasks 11–13)

```cpp
#pragma once
#include <cstdint>
#include <vector>
#include "azgo/evaluator.hpp"
#include "azgo/game.hpp"
#include "azgo/rng.hpp"

namespace azgo {

struct MctsConfig {
    int simulations = 200;       // new simulations per search() call
    int batchSize = 8;           // K descents per network call (virtual loss)
    float cPuct = 1.5f;
    bool addNoise = false;       // root Dirichlet noise (self-play only)
    float dirichletScale = 10.0f;// alpha = dirichletScale / (n * n)
    float noiseFraction = 0.25f; // epsilon
};

// Edge statistics are from the perspective of the player who chooses the
// edge (the parent's side to move).
struct MctsEdge {
    int action;
    float rawPrior;       // network prior (softmax over legal logits)
    float prior;          // rawPrior, or noise-mixed at the root
    int visits = 0;
    float valueSum = 0;
    int virtualLoss = 0;
    int child = -1;       // node index, -1 until created
};

struct MctsNode {
    Board board;
    int parent;           // -1 for root
    int ply;              // game move count at this node
    int passes;           // consecutive passes leading here
    int firstEdge = 0;
    int numEdges = 0;
    bool expanded = false;
    bool pending = false; // awaiting evaluation in the current round
    int visits = 0;       // completed simulations through this node
    int virtualLoss = 0;
    float netValue = 0;   // evaluator value at expansion (side to move)
};

class Mcts {
public:
    Mcts(const MctsConfig& cfg, Evaluator& evaluator);

    // Runs cfg.simulations simulations from game's current position. If the
    // tree root does not match game.board() (hash + ply), the tree is reset.
    // Asserts !game.isTerminal().
    void search(const Game& game, Rng& rng);

    std::vector<int> rootVisits() const;      // [numActions], 0 for unexpanded/illegal
    std::vector<float> rootPriors() const;    // [numActions], priors used (after noise)
    float rootValue() const;                  // mean backed-up value, root side-to-move view
    int rootTotalVisits() const;

    // temperature > 0: sample with probability ∝ visits^(1/temperature).
    // temperature == 0: most visited, ties broken uniformly with rng.
    int selectAction(float temperature, Rng& rng) const;

    // Re-roots at the child reached by action, keeping its subtree
    // (compacted into a fresh arena); resets if that child does not exist.
    void advance(int action);
    void reset();

    // Introspection for tests.
    size_t nodeCount() const { return nodes_.size(); }
    long long totalVirtualLoss() const;       // sum over all edges and nodes; 0 between searches
    long long evaluatorBatches() const { return evaluatorBatches_; }
    const MctsNode& node(int i) const { return nodes_[i]; }
    const MctsEdge& edge(int i) const { return edges_[i]; }

private:
    MctsConfig cfg_;
    Evaluator& eval_;
    std::vector<MctsNode> nodes_;
    std::vector<MctsEdge> edges_;
    long long evaluatorBatches_ = 0;
    // Further private members and helpers: implementation-defined (Tasks 11-13).
};

} // namespace azgo
```

### include/azgo/parallel.hpp  (Task 14)

```cpp
#pragma once
#include <functional>

namespace azgo {

// Calls fn(index, worker) for every index in [0, count) using `workers`
// std::threads (worker in [0, workers)). Indices are handed out by an atomic
// counter, so callers must store results by index, never by completion order.
// The first exception thrown by fn is rethrown after all threads join.
void parallelFor(int count, int workers, const std::function<void(int index, int worker)>& fn);

// workers <= 0 means std::thread::hardware_concurrency() (at least 1).
int resolveWorkers(int workers);

} // namespace azgo
```

### include/azgo/selfplay.hpp  (Task 14)

```cpp
#pragma once
#include <cstdint>
#include <utility>
#include <vector>
#include "azgo/board.hpp"
#include "azgo/evaluator.hpp"
#include "azgo/mcts.hpp"
#include "azgo/rng.hpp"

namespace azgo {

struct SelfPlayConfig {
    MctsConfig mcts;             // addNoise is forced on by playSelfPlayGame
    int temperatureMoves = 20;   // plies sampled with temperature 1, then argmax
    float komi = 7.5f;
    int maxMoves = 0;            // 0 means defaultMaxMoves(n)
};

// A finished game in compact form. Position `ply` is the board BEFORE
// moves[ply] was played; Black moves at even plies (no handicap).
struct GameRecord {
    int boardSize = 0;
    float komi = 0;
    std::vector<int> moves;
    std::vector<uint8_t> stones;   // positions() * n*n Color bytes
    std::vector<std::vector<std::pair<int16_t, uint16_t>>> visits;  // per ply: (action, count), count > 0
    float resultForBlack = 0;      // +1, -1, or 0

    int positions() const { return static_cast<int>(moves.size()); }
    const uint8_t* stonesAt(int ply) const { return stones.data() + static_cast<size_t>(ply) * boardSize * boardSize; }
    Color toMoveAt(int ply) const { return ply % 2 == 0 ? Color::Black : Color::White; }
    float zAt(int ply) const;      // resultForBlack from the side to move at ply
};

// Plays one self-play game from the empty board with root noise and tree reuse.
GameRecord playSelfPlayGame(Evaluator& evaluator, const SelfPlayConfig& cfg, Rng& rng);

} // namespace azgo
```

### include/azgo/replay_buffer.hpp  (Task 15)

```cpp
#pragma once
#include <cstddef>
#include <deque>
#include <vector>
#include "azgo/rng.hpp"
#include "azgo/selfplay.hpp"
#include "azgo/serialize.hpp"

namespace azgo {

struct TrainingBatch {
    int size = 0;
    std::vector<float> input;   // [size][planes][area]
    std::vector<float> pi;      // [size][area + 1]
    std::vector<float> z;       // [size]
};

// Sliding window of whole games. A sample is (game, ply); its history planes
// come from the same game's earlier positions, so history costs no storage.
class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t capacityPositions);
    void add(GameRecord game);            // then evicts oldest games while positions() > capacity,
                                          // never evicting the newest game
    void setCapacity(size_t capacityPositions);   // evicts as add() does
    size_t capacity() const { return capacity_; }
    size_t positions() const { return positions_; }
    size_t games() const { return games_.size(); }
    long long totalPositionsAdded() const { return totalAdded_; }
    const GameRecord& game(size_t i) const { return games_[i]; }   // 0 = oldest

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
    size_t positions_ = 0;
    long long totalAdded_ = 0;
    std::deque<GameRecord> games_;
    // Sampling index helpers: implementation-defined (Task 15).
};

} // namespace azgo
```

### include/azgo/config.hpp  (Task 16)

```cpp
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/network.hpp"

namespace azgo {

struct ConfigError : std::runtime_error { using std::runtime_error::runtime_error; };

struct Config {
    // run
    uint64_t seed = 1;
    int generations = 100;
    int workers = 0;                   // 0 = hardware concurrency
    std::string backend = "naive";     // naive | blas
    // game
    int boardSize = 9;
    float komi = 7.5f;
    int maxMoves = 0;                  // 0 = 2 * n * n
    // network (architecture keys)
    int history = 8;
    int blocks = 6;
    int filters = 64;
    int valueHidden = 64;
    // self-play and search
    int gamesPerGeneration = 100;
    int simulations = 200;
    int mctsBatch = 8;
    float cPuct = 1.5f;
    float dirichletScale = 10.0f;
    float noiseFraction = 0.25f;
    int temperatureMoves = 20;
    // training
    int bufferPositions = 200000;
    int trainSteps = 200;
    int batchSize = 256;
    std::string lrSchedule = "0:0.02";
    float momentum = 0.9f;
    float weightDecay = 1e-4f;
    // arena
    int arenaEvery = 1;                // 0 = never
    int arenaGames = 20;               // per opponent, even
    int arenaOpponents = 4;            // most recent ladder entries (generation 0 always added)
    int arenaSimulations = 100;
    int arenaTemperatureMoves = 4;
    bool gating = false;
    float gatingThreshold = 0.55f;
    // output
    int sgfPerGeneration = 2;

    NetConfig netConfig() const;
    int effectiveMaxMoves() const;     // maxMoves or 2 * n * n
    int effectiveWorkers() const;      // resolveWorkers(workers)
};

// key = value per line; '#' starts a comment (anywhere); blank lines ignored;
// surrounding whitespace and a trailing '\r' are stripped. Unknown key,
// malformed or out-of-range value, or a key repeated in one file ->
// ConfigError("<file>:<line>: <key>: <reason>").
Config loadConfigFile(const std::string& path, const Config& base = Config{});
Config parseConfigText(const std::string& text, const std::string& sourceName,
                       const Config& base = Config{});
// "key=value" (whitespace around '=' allowed). Same validation.
void applyOverride(Config& cfg, const std::string& keyEqualsValue);

// Every key, one "key = value" line each, in table order. Round-trips
// through parseConfigText exactly.
std::string configToString(const Config& cfg);

// Keys that fix the network's shape: board_size, history, blocks, filters, value_hidden.
std::vector<std::string> architectureDiff(const Config& a, const Config& b);

} // namespace azgo
```

Config keys (file spelling → field, range):
`seed` (uint64), `generations` 1..1000000, `workers` 0..256,
`backend` naive|blas, `board_size` 5..19, `komi` -500..500,
`max_moves` 0..100000, `history` 1..16, `blocks` 0..40, `filters` 1..512,
`value_hidden` 1..1024, `games_per_generation` 1..1000000,
`simulations` 1..100000, `mcts_batch` 1..256, `c_puct` (0, 100],
`dirichlet_scale` 0..1000, `noise_fraction` 0..1, `temperature_moves`
0..100000, `buffer_positions` 1..100000000, `train_steps` 0..1000000,
`batch_size` 1..4096, `lr_schedule` (LrSchedule::parse), `momentum` [0, 1),
`weight_decay` 0..1, `arena_every` 0..1000000, `arena_games` 2..10000 and
even, `arena_opponents` 1..32, `arena_simulations` 1..100000,
`arena_temperature_moves` 0..100000, `gating` true|false|on|off|1|0,
`gating_threshold` 0.5..1, `sgf_per_generation` 0..1000.

### include/azgo/elo.hpp  (Task 17)

```cpp
#pragma once
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace azgo {

// All arena results so far. Players are generation numbers.
class ResultsTable {
public:
    struct Pair { int a; int b; int games; double scoreA; };  // a < b; scoreA in [0, games]
    // scoreA: 1 win, 0.5 draw, 0 loss for player a. a != b; either order is
    // accepted and stored normalized as (min, max) with the score flipped as needed.
    void addGame(int a, int b, double scoreA);
    void addMatch(int a, int b, int games, double scoreA);    // many games at once
    std::vector<Pair> pairs() const;                          // sorted by (a, b)
    std::vector<int> players() const;                         // sorted
    bool empty() const { return table_.empty(); }
    std::string toCsv() const;                                // "a,b,games,score_a" header + rows
    static ResultsTable fromCsv(const std::string& csv);      // throws std::runtime_error
private:
    std::map<std::pair<int, int>, std::pair<int, double>> table_;  // (a,b) -> (games, scoreA)
};

// Maximum-likelihood Bradley-Terry ratings by the MM algorithm (Hunter 2004),
// on the Elo scale (400 * log10), with `anchor` pinned at 0. Each played pair
// gets one extra virtual game scored as a draw, so a player who has only won
// (or only lost) still gets a finite rating. Players unreachable from the
// anchor through played pairs are omitted.
std::map<int, double> fitBradleyTerry(const ResultsTable& results, int anchor = 0);

// 95% (z = 1.96) Wilson score interval for a win rate of score/games.
std::pair<double, double> wilsonInterval(double score, int games, double z = 1.96);

} // namespace azgo
```

### include/azgo/arena.hpp  (Task 18)

```cpp
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "azgo/network.hpp"

namespace azgo {

struct ArenaConfig {
    int games = 20;              // even; A plays Black in even-indexed games
    int simulations = 100;
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

// Plays cfg.games games between two networks of the same NetConfig (throws
// std::invalid_argument otherwise) on `workers` threads, no root noise, tree
// reuse on. Game i uses Rng(deriveSeed(seed, i)). Deterministic for a seed.
MatchResult playMatch(const Network& a, const Network& b, const ArenaConfig& cfg,
                      uint64_t seed, int workers, const std::string& backendName);

// Opponents for generation g: g-1, g-2, g-4, g-8, ... (>= 1), the first
// `maxOpponents` of those, then 0 if not already present. Empty for g == 0.
std::vector<int> ladderOpponents(int generation, int maxOpponents);

// score / games >= threshold.
bool passesGate(double score, int games, double threshold);

// Score of one finished game for player A: 1, 0.5 (resultForBlack == 0) or 0.
double matchScore(float resultForBlack, bool aIsBlack);

} // namespace azgo
```

### include/azgo/sgf.hpp  (Task 19)

```cpp
#pragma once
#include <string>
#include <vector>

namespace azgo {

// "(;GM[1]FF[4]SZ[n]KM[komi]RE[B+R|W+R|0]C[comment];B[dd];W[tt]...)".
// Point coordinates: column letter then row letter, 'a' = left/top.
// Pass is written as an empty value "B[]". RE uses the sign of resultForBlack
// with the Tromp-Taylor margin, e.g. "B+3.5", "W+7.5", "0".
std::string gameToSgf(int n, float komi, const std::vector<int>& moves,
                      float scoreForBlack, const std::string& comment = "");

struct SgfGame { int boardSize = 19; float komi = 0; std::vector<int> moves; };
// Reads SZ, KM and the main line of B[]/W[] moves (pass: "" or "tt" when n <= 19).
// Ignores other properties and variations. Throws std::runtime_error on
// malformed input or off-board coordinates.
SgfGame parseSgf(const std::string& text);

} // namespace azgo
```

### include/azgo/run_state.hpp  (Task 20)

```cpp
#pragma once
#include <string>
#include <vector>
#include "azgo/elo.hpp"
#include "azgo/replay_buffer.hpp"

namespace azgo {

// Everything needed to continue a run exactly, besides the checkpoints and
// config.cfg. The network for `generation` is checkpoints/gen_NNNN.azn.
struct TrainingState {
    int generation = 0;           // last completed generation
    int bestGeneration = 0;       // gating: network used for self-play
    long long samplesTrained = 0; // total training samples drawn
    ReplayBuffer buffer{1};
    std::vector<std::vector<float>> velocity;  // optimizer momentum, params() order
    ResultsTable results;
};

// Binary, magic "AZST" (u32 0x54535A41), version u32 = 1. Atomic replace.
void saveTrainingState(const std::string& path, const TrainingState& state);
TrainingState loadTrainingState(const std::string& path);   // throws std::runtime_error

// Run-directory paths.
std::string checkpointPath(const std::string& runDir, int generation);  // runDir/checkpoints/gen_0007.azn
std::string statePath(const std::string& runDir);                        // runDir/state.bin
std::string configPath(const std::string& runDir);                       // runDir/config.cfg

} // namespace azgo
```

### include/azgo/trainer.hpp  (Task 21)

```cpp
#pragma once
#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include "azgo/config.hpp"
#include "azgo/network.hpp"
#include "azgo/run_state.hpp"

namespace azgo {

struct GenerationStats {
    int generation = 0;
    int games = 0;
    double avgLength = 0;       // plies per game
    double passRate = 0;        // passes / plies
    double blackWinRate = 0;
    float policyLoss = 0, valueLoss = 0;   // mean over the generation's steps
    float lr = 0;
    double replayRatio = 0;     // samples drawn this generation / new positions
    double selfPlaySeconds = 0, trainSeconds = 0, arenaSeconds = 0;
    double elo = 0;             // after refit; previous value if no arena this generation
    double winRateVsPrev = -1;  // -1 when no match vs g-1
};

// The generation loop: self-play -> train -> arena/Elo -> checkpoint + state.
class Trainer {
public:
    // New run. Creates runDir (and checkpoints/, sgf/); throws ConfigError if
    // runDir already contains state.bin. Writes config.cfg, gen_0000.azn and
    // an initial state.bin (generation 0).
    static std::unique_ptr<Trainer> startNew(const Config& cfg, const std::string& runDir);
    // Loads config.cfg, applies overrides, refuses (ConfigError listing keys)
    // if an architecture key changed, loads state.bin and the latest checkpoint.
    // Rewrites config.cfg with the overridden values.
    static std::unique_ptr<Trainer> resume(const std::string& runDir,
                                           const std::vector<std::string>& overrides);

    int generation() const { return state_.generation; }
    const Config& config() const { return cfg_; }
    GenerationStats runGeneration();
    void runUntil(int lastGeneration);   // while generation() < lastGeneration
    void setLog(std::ostream* os) { log_ = os; }   // progress lines; nullptr = silent

private:
    Trainer(const Config& cfg, const std::string& runDir, Network net, TrainingState state);
    Config cfg_;
    std::string runDir_;
    Network net_;
    TrainingState state_;
    std::ostream* log_;
    // Further members: implementation-defined (Task 21).
};

// Seed phases for deriveSeed(cfg.seed, generation, phase, index).
constexpr uint64_t kPhaseInit = 0, kPhaseSelfPlay = 1, kPhaseTrain = 2, kPhaseArena = 3;

} // namespace azgo
```

Run directory files written by the trainer:
`config.cfg`, `checkpoints/gen_NNNN.azn`, `state.bin`, `log.csv`
(header `generation,games,avg_length,pass_rate,black_win_rate,policy_loss,value_loss,lr,replay_ratio,selfplay_s,train_s,arena_s,elo,winrate_vs_prev`),
`elo.csv` (header `generation,elo,winrate_vs_prev,wr_low,wr_high`, rewritten
in full after each refit), `results.csv` (ResultsTable::toCsv, rewritten
each arena round), `sgf/gen_NNNN_game_KK.sgf`.

### apps/cli.hpp  (Task 21; header-only, in apps/)

```cpp
#pragma once
#include <map>
#include <string>
#include <vector>

// Strict "--name value" / "--flag" parser. Unknown options, missing values
// and malformed numbers throw std::invalid_argument with the option name.
class ArgParser {
public:
    // valueOptions take a value; flagOptions do not. "--set" may repeat.
    ArgParser(int argc, char** argv, const std::vector<std::string>& valueOptions,
              const std::vector<std::string>& flagOptions);
    bool has(const std::string& name) const;
    std::string get(const std::string& name, const std::string& def = "") const;
    int getInt(const std::string& name, int def) const;       // strtol, whole string
    double getDouble(const std::string& name, double def) const;
    bool flag(const std::string& name) const;
    const std::vector<std::string>& sets() const { return sets_; }   // every --set value
private:
    std::map<std::string, std::string> values_;
    std::vector<std::string> flags_;
    std::vector<std::string> sets_;
};
```

### Executables (Tasks 21–22)

- `train --config FILE --run DIR [--set k=v]... [--generations N]` — new run;
  `train --run DIR --resume [--set k=v]... [--generations N]` — continue.
  `--generations N` is shorthand for `--set generations=N`. Exit 0 on success,
  1 with `error: <message>` on stderr for any ConfigError/runtime_error.
- `arena --a CKPT --b CKPT [--games 40] [--simulations 200] [--komi 7.5]
  [--temperature-moves 4] [--workers 0] [--backend naive] [--seed 1]` —
  prints `A <path> vs B <path>: A scored 27.5/40 = 0.688 (95% CI 0.531-0.811), distinct games 38`.
- `analyze --checkpoint CKPT [--moves "D4 C3 pass"] [--sgf FILE]
  [--simulations 400] [--komi 7.5] [--backend naive]` — prints the board,
  the network value, a policy heat map (per point: percent, rounded, `.`
  below 1%), the top 10 moves by prior and by MCTS visits. Illegal or
  malformed moves: `error: move 3 (C3): illegal` and exit 1.
- `bench [--sizes 5,9,19] [--backend naive|blas|all] [--seconds 2]` — for
  each size's preset network: evaluations/sec at batch 8 (1 thread), train
  step time at batch 128; board random-playout moves/sec.
