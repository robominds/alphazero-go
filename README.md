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
