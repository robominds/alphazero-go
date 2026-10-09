# Multithreaded Training Step — Design

Author: Mark Castelluccio, with design assistance from Claude (Anthropic).

Amends the training section of
`docs/superpowers/specs/2026-09-29-alphazero-go-design.md`. Implements
option 1 of the plan that followed the Metal benchmark
(`docs/results/metal-bench.txt`): use every CPU core for the training
phase before considering a GPU. Option 2, a whole-network Metal backend,
and option 3, a cross-worker batching evaluator, are not part of it. This
design does not make either harder.

## Problem

Self-play and the arena run one game per worker thread on every core.
The training phase of a generation (400 SGD steps at batch 256 in the 9×9
preset, plus the validation loss) ran on one core. BLAS is kept
single-threaded because the self-play workers already fill the cores,
and only the naive backend split its gemms over threads. On an M5 Max a
6×96 step took ~425 ms, so the phase took ~170 s of a ~430 s generation.
Gemm was only ~30% of the step. The rest was im2col, col2im, batch norm
and elementwise loops, all single-threaded.

## Goal

Run the training step on many threads, chosen at run time, so that the
result is **bitwise identical for every thread count, and identical to
the single-threaded code before this change**. This covers the network's
forward, loss, backward and SGD update, assembling the batch from the
replay buffer, and the validation-loss pass. The run's determinism
guarantee (a run depends only on its seed, never on its thread counts,
and resumes byte-identically) must keep holding.

## Non-Goals

- Changing any arithmetic. No reassociated sums, no new kernels, no
  mixed precision. The algorithm stays AlphaGo Zero's.
- Multithreaded inference in search. Each search worker keeps its own
  single-threaded workspace.
- A GPU backend or a batching evaluator (options 2 and 3).

## Profile (before)

One step at batch 256 on 9×9, BLAS backend, one thread, with the
competing run paused, in ms per step. The backend calls were timed inside
the real step by a wrapping backend. Batch norm and ReLU were timed alone
at the trunk's size (×13 layers). The step's other work (batch norm,
ReLU, copies, loss, the heads' Linear layers) is the step minus the
backend calls.

| phase | 6×64 | 6×96 |
|---|---|---|
| whole step (trainStep) | 266 | 427 |
| im2col (26 calls) | 90 | 135 |
| col2im (12 calls) | 41 | 62 |
| gemm, conv forward (15 calls) | 20 | 39 |
| gemm, conv dW (15 calls) | 18 | 34 |
| gemm, conv dX (14 calls) | 27 | 50 |
| everything outside the backend | 71 | 107 |
| of it, batch norm forward + backward (alone, ×13) | 30 | 46 |
| of it, ReLU backward (alone on random data, ×13) | ≤ 50 | ≤ 75 |
| SGD update | 0.07 | 0.15 |
| batch assembly (sampling + encoding) | 0.4 | 0.4 |

ReLU's backward was slow because the loop loaded `dy` conditionally,
which left one branch per element. It was not vectorized, and on random
signs the branch mispredicts half the time.

## Design

### A persistent pool with fixed chunks

`ThreadPool` (`include/azgo/parallel.hpp`) owns `threads − 1` helper
threads for its lifetime, and the caller of `run(chunks, fn)` is the
last thread. Helpers sleep between runs, after a short spin, because a
step makes ~100 parallel loops microseconds apart. `run` hands out
chunk indices through an atomic counter and returns when every chunk is
done, rethrowing the first exception. `parallelChunks(pool, …)` and
`parallelRanges(pool, count, chunk, …)` fall back to a plain loop when
`pool` is null.

**Determinism rule.** The chunks depend only on the data's shape, never
on the thread count. Each chunk writes only its own outputs and computes
each value with exactly the operations, in exactly the order, of the
single-threaded code. Which thread runs a chunk then cannot change a
bit. Reductions keep their order:

- Batch norm: a channel is a chunk, and its sums (in double) run over
  the channel exactly as before.
- Loss: the per-sample work (log-softmax normalizer, gradients) runs on
  the pool in chunks of 16 samples. The batch sums then run serially in
  sample order, with the same expressions, so the same contractions.
- Weight gradients: one gemm call over the whole batch, as before (see
  below).
- Optimizer: elementwise, in chunks of 32k elements per tensor.
- Sampling: all random draws first, serially, in the old order
  (position, then symmetry, per sample). Then the encoding runs in
  chunks of 16 samples.

### Gemm partitions

A convolution's three gemms are cut so that every element of C is still
one gemm call's sum over the whole of K:

- **Forward** (`y = w · columns`) and **input gradient**
  (`dcolumns = wᵀ · dy`): blocks of 512 columns of C (pixels), with the
  last block taking the remainder, through a gemm with leading
  dimensions. A product with fewer than 8 rows (the 1×1 heads' forward)
  or fewer than 1024 columns stays one call.
- **Weight gradient** (`dW = dy · columnsᵀ`, a sum over every pixel):
  one call, run on the pool beside the input gradient's blocks. A
  backend may offer `Backend::gemmRowsOnPool`; the naive backend does,
  and cuts it into bands of 16 rows after packing the transposed operand
  once.
- The heads' Linear layers are tiny and stay single calls.

With a pool, these cuts are made even at one thread. Without a pool
(search), every gemm is one call as before.

**Why these cuts reproduce the old bits.** For the naive backend this
follows from the loops: every element sums over K in one fixed order,
whatever range of rows or columns a call covers. Accelerate is a black
box, so it was measured. In a sweep of ~11,000 convolution shapes
(boards 5–19, 8–256 filters, batches 1–256, forward, dW and dX, blocks
of 128–1024 columns or 8–32 rows, remainder merged), every column block
of a product with at least 8 rows, and every row band of at least 8
rows, matched the unsplit call bit for bit. Single-row bands (Accelerate's
gemv path) did not, and neither did column splits of the 1- and 2-row
head products or of some small-batch weight gradients. Those are exactly
the cuts the design avoids. `tests/test_backend.cpp` keeps a
representative set of the shapes. The thread-count independence is
structural and holds for any backend. Equality with the old binary is
proven for the naive backend and measured for Accelerate.

**Why columns, not rows, for BLAS.** The first version cut every gemm
into 16-row bands. Each band re-reads the whole column matrix (72 MB for
6×96 at batch 256), and on Accelerate's matrix units the forward and dW
bands ran slower than the single call. A 6×96 step reached only 142 ms on
18 threads. Column blocks read disjoint slices of that matrix: 93 ms.

### Smaller single-threaded fixes (same bits)

- ReLU backward loads `dy` unconditionally and selects, so it
  vectorizes. The values are the same: `dy` or +0.
- col2im's inner loop runs over the on-board range only, so it
  vectorizes. It does the same additions in the same order per point.
  im2col uses the same bounds.

A bitwise test against the original loops guards both.

### Configuration

`train_threads` (int, 0 = `std::thread::hardware_concurrency()`, else
1–256; default 0) is the pool size for the validation loss and the
training steps. Results do not depend on it, so it is tunable on
resume. The presets leave it at 0. `bench --train-threads 1,2,4,8
--train-batch 256` times the step at each thread count.

## Testing

- `test_parallel`: every chunk runs exactly once (1–7 threads, many
  runs), inline execution with one thread, helpers are used, exceptions
  propagate, and the pool survives them.
- `test_backend`: strided gemm against packed; column blocks and naive
  row bands against the unsplit call, bit for bit, on pools of 1, 2, 3
  and 7 threads (Accelerate too when built); im2col and col2im against
  the original loops, bit for bit.
- `test_network`: a training step's every output (losses, gradients,
  running statistics, updated weights and momentum, inference outputs
  and losses) on pools of 1, 2, 3 and 7 threads equals the no-pool
  result byte for byte (5×5 and 9×9, with and without the ownership
  head, naive and BLAS).
- `test_replay_buffer`: sampling on a pool gives the same batch and
  leaves the generator in the same state.
- `test_trainer`: `train_threads` 1 and 4, and a run resumed with a third
  value, write byte-identical checkpoints, `state.bin`, results and
  ratings.
- Old against new binary: short runs of main's `train` and this
  branch's write byte-identical checkpoints, `state.bin` and
  `results.csv`. The runs were 5×5 (naive and BLAS), 9×9 at batch 256
  (BLAS 6×64 and 6×96, naive 6×64, each with `train_threads` 0 and 1)
  and the 19×19 preset's network. So did 30 generations of the full 5×5
  preset with BLAS, run with the earlier row-band version.

## Results

M5 Max (18 cores), 9×9, batch 256, one training step (forward, loss,
backward, SGD update), the competing run paused:

| threads | 6×64 BLAS | 6×96 BLAS | 6×64 naive | 6×96 naive |
|---|---|---|---|---|
| before (1) | 266 ms | 427 ms | | |
| 1 | 201 ms | 323 ms | 2018 ms | 4362 ms |
| 2 | 123 ms | 201 ms | 1189 ms | 2433 ms |
| 4 | 79 ms | 133 ms | 765 ms | 1519 ms |
| 8 | 61 ms | 106 ms | 664 ms | 1050 ms |
| 18 | 52 ms | 93 ms | 571 ms | 835 ms |

That is 5.1× (6×64) and 4.6× (6×96) against the old step with BLAS. The
weight-gradient gemms, one call each, and the matrix units shared by
cores bound the rest: Accelerate's forward gemm gains only 2.4× on 18
threads. A 400-step 6×96 training phase drops from ~170 s to ~37 s, and
the validation pass speeds up the same way.
