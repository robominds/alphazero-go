# Distributed Self-Play — Design

Author: Mark Castelluccio, with design assistance from Claude (Anthropic).

Amends the self-play step of the generation loop
(`docs/superpowers/specs/2026-09-29-alphazero-go-design.md`) and the
"Who uses it" section of
`docs/superpowers/specs/2026-10-09-metal-gpu-backend-design.md`. It builds
on the batching hub and `WorkerEvaluators` of that design and on the
determinism argument of `2026-10-10-arena-gpu-design.md`.

The algorithm does not change. Every game is the same AlphaGo Zero
self-play game, seeded the same way, from the same network, with the same
v_resign. What changes is *where* each game is played: on the Mac Studio's
GPU or CPU, or on one of two Mac minis' GPUs or CPUs over a Thunderbolt
bridge.

## Problem

A 19×19 10×128 generation spends about 90% of its time in self-play, and
self-play spends almost all of that evaluating the network. Today one
generation uses one evaluator type on one machine. With `backend = metal`
that is the Studio's GPU alone (`docs/results/bench.txt`):

| engine | 19×19 10×128 evals/s | source |
|---|---|---|
| Studio GPU (M5 Max), hub, 36 threads | 2,743 | measured |
| Studio CPU, 18 BLAS workers | 863 | measured (876 on 2026-10-18; 801 with 4 workers) |
| Studio GPU and 4 CPU workers at once | 2,795 + 443 = 3,238 (default QoS); 2,809 + 339 = 3,148 (utility) | measured: `bench --mixed`, hub alone 2,906–2,931 |
| each mini GPU (M4), hub, 20 threads | 506 | measured (estimate was ~550) |
| each mini CPU (M4, 10 cores), BLAS alone | 390 (2 workers) – 412 (10) | measured (estimate was ~450–500) |
| each mini, GPU and 2 CPU workers at once | 480 + 196 = 676 | measured: `bench --mixed` |
| **all six** | **~5,600** (estimate) | ≈ 2.05 × the Studio GPU alone; measured: the Studio's 3,148 (4 CPU workers, utility) plus 676 per mini ≈ 4,500 ≈ 1.54× the Studio GPU alone (2,931), before tails and transfers |

The measured rows come from `bench --mixed` on an idle M4 mini and on the
Studio with the 19×19 job paused (`docs/results/bench.txt`, "Measuring
engines side by side"). A CPU engine beside a busy hub is not the CPU
engine alone: on the M4, two BLAS workers lower the hub's rate by 5% and
add 196 evals/s, and more workers lower the hub further without raising
the total (8 workers: hub 72%, total 633). On the Studio the hub keeps
≥ 90% with up to 12 workers, but the CPU engine adds only 340–580
evals/s, not its 863 alone: the CPU engine is worth ~11–14% there, the
minis most of the rest. The estimate of ~2× falls to ~1.5× before tails,
below the 1.6× acceptance target; see "Performance".

While the GPU runs, the Studio's CPU is nearly idle: a worker blocked on
the hub uses no CPU, and the hub is saturated at 18 threads (2,742) and
36 (2,743). The minis sit idle. Using everything roughly doubles
self-play throughput, so a 19×19 generation could take about 0.1 + 0.9 /
2.05 ≈ 0.54 of its time today, before tails and transfer overheads.

Two facts shape the design:

- **Different engines give different bits.** A position evaluated by
  the GPU and by BLAS differs in the last bits (within 1e-5). An M4 and
  an M5 Max can differ too: macOS compiles the shaders for the device,
  and Accelerate picks kernels per chip. A game's moves can then differ.
  So which engine plays a game is part of what determines the run.
- **The project's core promise is reproducibility.** A run depends only
  on its config and seed, never on worker counts or timing, and resumes
  byte for byte. Spreading games over machines must keep a precise
  version of that promise, not quietly drop it.

## Goals

- **Use all the compute for self-play**: the coordinator's GPU and CPU
  at once, and the GPU and CPU of any number of remote Macs reached
  over passwordless ssh.
- **Fixed engine assignment.** Each game of a generation is assigned up
  front, by game index, to one *engine* (a machine and a backend). The
  assignment is a pure function of the config and the game count.
- **A precise determinism contract.** Given the assignment and the same
  hardware and OS per engine, a run is reproducible and resumes byte for
  byte; worker counts never matter (see "Determinism contract").
- **Default unchanged.** Without the new keys, a run is byte-identical
  to `main`. Old runs resume unchanged.
- **Switchable at a generation boundary** of a running run with
  `--resume --set selfplay_engines=...`, so the 19×19 run in progress can
  use it.
- **Robust to a remote failing.** A remote that fails or times out has
  its games replayed by the coordinator, recorded exactly, and the run
  goes on, unless the user asks it to stop instead.
- **No host names, user names or addresses in the repo** or in a
  committed config. They live in a machine-local hosts file.
- **Desktop stays responsive** on the coordinator: CPU self-play there
  leaves cores free and runs at lower priority.
- Teaching quality: one small entry point (`selfplay`) that plays a
  slice of a generation, readable file formats, and a determinism
  argument that extends the existing one.

## Non-Goals

- **Bit-identity with a single-machine run.** A game played by BLAS on
  an M4 is not the game the Studio's GPU would have played. The contract
  says exactly what *is* guaranteed.
- **Dynamic load balancing.** Handing the next game to whichever engine
  is free would make the assignment, and so the bits, depend on timing
  (see "Alternatives").
- **Distributed training or arena.** Training stays on the coordinator.
  The arena is a later phase (see "Later: the arena").
- **A general cluster scheduler.** Two or three trusted Macs on a
  private bridge. No authentication beyond ssh's, no encryption beyond
  ssh's, no NAT traversal, no job queue that outlives a generation.
- **Changing the search, the network or the game.** No algorithmic
  change, AlphaGo Zero's self-play as before.

---

## Terms

- **Machine**: the coordinator (`local`) or a remote Mac, named by an
  *alias* such as `mini1`. Aliases are the only machine names that
  appear in configs, logs and the manifest.
- **Engine**: (machine, backend), backend ∈ {`metal`, `blas`, `naive`},
  written `mini1/metal`. A `metal` engine is one `InferenceHub` on that
  machine's GPU with many workers. A `blas` or `naive` engine is one
  `LocalEvaluator` per worker thread on that machine's CPU.
- **Slice**: the contiguous range of game indices [first, first + count)
  that one engine plays in one generation.
- **Coordinator**: the `train` process. It owns the run directory, plans
  the slices, plays the local ones in-process, launches the remote ones,
  verifies and merges everything, then continues the generation as
  today.
- **Fingerprint**: what determines an engine's arithmetic, as a string:
  backend, chip, OS version and build, and for `metal` the shader hash
  (`metalDescription()`). Two engines with equal fingerprints and equal
  build IDs compute the same bits.
- **Build ID**: the git commit the binary was built from, plus a hash of
  the uncommitted diff if the tree was dirty.

## Overview

```
                            coordinator: train (Mac Studio)
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ runGeneration(g)                                                            │
 │   assignGames(selfplay_engines, G) ── slices: [0,52) local/metal            │
 │                                               [52,64) local/blas            │
 │                                               [64,74) mini1/metal  ...      │
 │   ┌── local/metal: playSelfPlayGames(hub, 52 workers)  ─┐ in-process threads│
 │   ├── local/blas:  playSelfPlayGames(6 CPU workers)    ─┤                   │
 │   ├── mini1/metal: job file ─scp─► ssh selfplay ─scp─► result ──┐ verify    │
 │   ├── mini1/blas:  ...                                          ├─► merge by│
 │   └── mini2/...                                                 ┘  index    │
 │   records[0..G) ─► resignation, SGF, validation, buffer, train, arena, commit│
 └─────────────────────────────────────────────────────────────────────────────┘
```

Self-play becomes "produce `records[0..G)`", with as many producers as
engines. Everything after the merge (resignation statistics and
calibration, SGFs, validation loss, buffer add in index order, training,
checkpoint, arena, `state.bin`) is today's code, run on the merged
records.

## Engines and the assignment

### The `selfplay_engines` key

A tunable config key. Empty (the default) means today's behaviour: one
engine, `local/<backend>`, with the `workers` key's rule, through today's
code path.

```
selfplay_engines = local/metal:50, local/blas=12, mini1/metal:10, mini1/blas=8, mini2/metal:10, mini2/blas=8
```

Each item is `machine/backend` followed by either

- `:weight`: a share of the games, an integer in [1, 1,000,000]; or
- `=games`: an exact number of games, in [0, 1,000,000].

Machines are aliases (`[a-z][a-z0-9_-]{0,31}`). `local` is the
coordinator. Every other alias must be defined in the hosts file (below);
the config never says how to reach it. An engine may appear more than
once: each item is its own pool of workers with its own slice, and items
are identified by their position in the list. Two items with one engine
split its games without changing a bit (see "Determinism contract"); the
tests use that. `checkConsistency` refuses unknown backends and fixed
counts summing to more than `games_per_generation`. With no weighted item, the
fixed counts must sum to exactly `games_per_generation`.

Fixed counts exist for CPU engines. A CPU engine plays one game per
worker, so its count should be a multiple of its workers, or the last
wave runs with idle cores. A GPU engine runs all its games at once (see
"Sizing"), so it is the natural home for a weight that absorbs the rest.

A second tunable key, `selfplay_fallback = local | stop` (default
`local`), picks the failure policy (see "Failure handling").

Both keys are printed by `configToString`, so a run's `config.cfg` gains
two lines (`selfplay_engines =` and `selfplay_fallback = local`). An old
run's `config.cfg` lacks them and loads with the defaults, as for every
key added before.

### Apportionment: `assignGames`

`assignGames(engines, G)` turns the list into slices, in integers only,
so no float rounding can differ between machines:

1. Fixed items get their counts. R = G − Σ fixed.
2. Weighted items share R by the largest-remainder method: with W = Σ
   weights, item e gets ⌊R·w_e / W⌋. The R − Σ⌊…⌋ leftover games go one
   each to the items with the largest remainders (R·w_e mod W); ties go to
   the earlier item in the list.
3. Slices are contiguous, in list order: the first engine gets [0, n_0),
   the next [n_0, n_0 + n_1), and so on. Engines with 0 games get no slice.

So the assignment depends only on the key's value and G. It does not
depend on the machine, the time, or which remotes are up. Contiguous
ranges keep a slice describable by two integers. Game index carries no
meaning (every game's randomness is an independent stream), so
contiguity biases nothing. The first `sgf_per_generation` games come
from the first engine, which is the reason to list `local/...` first.

### Why static

Static assignment is what makes the run reproducible: game i is always
played by the same engine, so its bits are always the same. The cost is
a tail. An engine that finishes early waits for the slowest one. The
cost is kept small by sizing (below) rather than by moving games at run
time, which would trade reproducibility for a few minutes per generation
(see "Alternatives").

## The hosts file (machine-local)

How to reach a machine, and how hard to drive it, is not part of a run's
result. It lives in a file outside the repository:

1. `train --hosts FILE`, else
2. `$AZGO_HOSTS`, else
3. `~/.config/azgo/hosts`.

The file is only required when `selfplay_engines` names a machine other
than `local`. It is never copied into the run directory, and `train`
never prints its `ssh` targets except in an error about that host.

```
# ~/.config/azgo/hosts -- machine-local; never commit this file.
# alias  key=value ...   (values: no spaces or shell metacharacters)
local  cpu_workers=6 cpu_qos=utility
mini1  ssh=<user>@<address> exe=<repo>/build-metal/selfplay workdir=<repo>/runs/remote
mini2  ssh=<user>@<address> exe=<repo>/build-metal/selfplay workdir=<repo>/runs/remote
```

| key | meaning | default |
|---|---|---|
| `ssh` | ssh target. Absent for a non-`local` alias: the *subprocess* transport, which runs `exe` on this machine as a separate process (tests, or a niced CPU engine on the coordinator) | — |
| `exe` | path of the `selfplay` binary on that machine | required for remotes |
| `workdir` | a directory on that machine for networks, jobs and results | required for remotes |
| `metal_workers` | worker threads of that machine's `metal` engine; 0 = every game of its slice at once (≤ 256) | 0 |
| `cpu_workers` | worker threads of its `blas`/`naive` engine; 0 = the default rule (`min(4, cores / 3)` locally, 2 on a remote) | 0 |
| `inference_batch` | overrides the run's `inference_batch` there (GPU memory on a 16 GB mini) | the run's |
| `cpu_qos` | `default` or `utility`: the QoS class of CPU engine threads | `utility` for `local`, `default` otherwise |
| `nice` | `nice` value of the remote `selfplay` process | 0 |
| `timeout` | seconds after which a slice counts as failed; 0 = none (dead connections are still caught by ssh keep-alives) | 0 |

The default CPU worker rule is `max(1, min(4, cores / 3))` on `local` (4
on the 18-core Studio, leaving most of the machine to the desktop, the
GPU engine's searches and its dispatcher) and `max(1, min(2, cores − 2))`
on a remote (2 on a mini). BLAS throughput saturates early because cores
share their matrix units (18 workers give only 2.2× one worker on 19×19
on the Studio). `bench --mixed` on an M4 mini set the remote rule: two
workers give 390 evals/s alone, 95% of ten workers' 412, and beside the
hub they keep it at 95% of its rate alone; four workers drop it to 85%
and eight to 72% without raising the total. The first rule, `cores − 2`
(8 on a mini), is replaced. On the Studio four workers are the knee (801
evals/s alone, 91% of 18 workers' 876) and keep the hub at 96%; six or
eight add at most ~140 evals/s (~4% of the total) and take more cores
from the desktop, so the first local rule, `cores / 3` (6), became
`min(4, cores / 3)`.

Values are checked when the file is read. Unknown keys, a repeated
alias, `ssh` on `local`, and characters outside `[A-Za-z0-9_./~@:+-]`
are refused. The last rule means a value can be pasted into an ssh
command line without quoting bugs.

## Several engines on one machine

`WorkerEvaluators` gives every worker of *one* pool the same kind of
evaluator, and `parallelFor` hands the next index to whichever worker
is free. A mixed pool would let a GPU-assigned game land on a CPU
worker, so the assignment would depend on timing. So each engine gets its
own pool:

```
playSelfPlayGames(net, sp, seed, g, first, count, EngineRun{backend, workers, inferenceBatch, qos})
```

is today's self-play loop, lifted out of `runGeneration` unchanged. It
creates one `WorkerEvaluators` and runs `parallelFor` over [first,
first + count), seeding game i with `deriveSeed(seed, g, kPhaseSelfPlay,
i)` and storing it at `records[i − first]`. The default path calls it
once with [0, G) and the run's `backend` and `workers`, which is exactly
today's code. With several local engines, the coordinator runs one call
per engine, each on its own thread, at the same time. The GPU engine's
hub and the CPU engine's per-worker BLAS backends share nothing but the
read-only `Network`.

**Leaving the desktop alone.** The Studio froze once under full load.
Two mechanisms, both result-neutral:

- *Fewer CPU workers than cores* on `local` (the default rule above).
  Because BLAS throughput saturates early, few workers lose little.
- *Lower QoS.* With `cpu_qos = utility`, each CPU engine worker calls
  `pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0)` before its first
  game. macOS then serves user-interactive work (the WindowServer, the
  terminal) first. The GPU engine's workers and dispatcher keep the
  default class, since the GPU, which sets the pace, waits on them. On
  other systems `cpu_qos` is accepted and does nothing.

The training phase still uses all cores (`train_threads`), as now.

A remote can also be the coordinator itself, through the subprocess
transport. That runs a CPU engine as a separate, niced process, an
alternative to QoS when stronger isolation is wanted.

## Self-play slices

### The `selfplay` app

A new, small executable. It is the whole remote interface:

```
selfplay --job JOB --out OUT --backend metal|blas|naive [--workers N] [--inference-batch N]
selfplay --version                    # build ID, e.g. 09e008b (metal)
selfplay --fingerprint --backend B    # checks B is usable here, prints the fingerprint
```

It reads the job, checks the network file's hash, plays the slice with
`playSelfPlayGames`, the same function the coordinator calls in-process,
and writes the result atomically (`writeFileAtomically`). Progress and
errors go to stderr. The exit status is 0 on success, 1 on any error, and
2 on a usage error.

A separate app, rather than `train --selfplay-slice`, keeps the remote
surface tiny and independent of a run directory. `train`'s command line
is already the most complex of the tools.

### The job file

Binary, little-endian through `BinaryWriter`, magic `AZSJ`, version 1.
It carries everything that determines the slice's games, as exact bits
(floats are stored as `f32`, never printed and re-parsed):

- seed, generation, first, count;
- the `SelfPlayConfig`: every `MctsConfig` field, `temperatureMoves`,
  `komi`, `maxMoves` (already resolved with `effectiveMaxMoves`),
  `resignation`, `resignThreshold` (this generation's v_resign) and
  `resignDisabledFraction`;
- the network: its file name in the workdir, and `fileHash` of its bytes;
- the coordinator's build ID;
- `inference_batch` (a speed hint, overridable).

The *job hash* is FNV-1a 64 over the job file's bytes. It names the job
in the result.

### The result file

Binary, magic `AZSR`, version 1:

- the job hash, the producer's build ID and engine fingerprint;
- count records, each game index followed by the whole `GameRecord`:
  the replay buffer's per-game layout (factored out of
  `ReplayBuffer::save` into `writeGameRecord`, with byte-identical
  `state.bin`), followed by `resignDisabled` and both sides' `resignLows`,
  which the buffer does not store but resignation calibration needs;
- statistics: wall seconds, positions, network evaluations;
- a trailing FNV-1a 64 checksum of every preceding byte.

A 19×19 game of 300 plies is about 110 kB of stones plus up to ~0.5 MB of
visit counts, so a slice of 15 games is a few megabytes, well under a
second over the bridge. FNV-1a detects corruption. It is not a defence
against an attacker. The machines are trusted.

### Verification

`readSliceResult(path, job, cfg)` refuses, naming the problem, a file
that:

- has the wrong magic, version or checksum, or is truncated;
- answers another job (job hash) or was built from another commit (build ID);
- does not hold exactly the indices [first, first + count) in order;
- holds a record that does not replay: `replayFinalOwnership` must
  reproduce every stored position and the final ownership, every move
  must be legal, and a played-out game's result must equal the replayed
  board's Tromp–Taylor result;
- has a resigned game whose result is not a loss for the side to move,
  a `resigned` or `resignDisabled` flag with resignation off, or resign
  lows that are not record lows (the checks `loadTrainingState` applies
  to the window).

Replay costs milliseconds per game. It catches a buggy or mismatched
producer, which a checksum cannot.

## The coordinator

### At start or resume

When `selfplay_engines` names remote machines, `Trainer::startNew` and
`Trainer::resume` run a *preflight* before writing anything:

1. read the hosts file; every remote alias must be defined;
2. for each remote machine: `ssh <target> <exe> --version` must print
   this binary's build ID; then `<exe> --fingerprint --backend B` for
   each of its engines must succeed. That checks Metal is usable there,
   the shaders compile, and BLAS is built;
3. for local engines: `checkBackendAvailable`, and for `metal`
   `checkMetalInferenceMemory`;
4. print one line per engine:
   `engine mini1/metal: Apple M4; macOS 26.1 (25B78); Apple M4, macOS 26.1, shaders b7b31a09 [09e008b]`.

A version mismatch is a `ConfigError` naming the alias and both IDs, and
suggesting `tools/cluster_sync.sh`. An unreachable machine is a
`ConfigError` with `selfplay_fallback = stop`. With `local`, it is a
warning, and that machine's slices fall back from the first generation
on. A fingerprint that differs from the previous generation's manifest
row for that engine (a macOS update on a mini) prints a note: from here
on, that engine's games differ in their last bits from before.

### Each generation

1. `assignGames` gives the slices.
2. The self-play network is the checkpoint already on disk
   (`gen_NNNN.azn` of `generation`, or of `bestGeneration` with gating),
   so nothing new is written to ship it.
3. For each remote slice, a launcher thread:
   - uploads the network to `<workdir>/<run-id>/` unless a file with
     that hash is there, and the job file;
   - runs `ssh -o BatchMode=yes -o ConnectTimeout=10
     -o ServerAliveInterval=15 -o ServerAliveCountMax=4 <target>
     'exec nice -n N <exe> --job ... --out ... --backend B --workers W'`,
     with stderr captured to `runDir/selfplay-work/gen_NNNN/<engine>.log`;
   - downloads the result and verifies it.

   `<run-id>` is the run directory's name and a hash of its absolute
   path. File names carry the generation, the slice's first index and an
   attempt number, so a stray process from
   an earlier attempt can never be mistaken for the current one.
4. Meanwhile, each local engine runs `playSelfPlayGames` on its own thread.
5. Join. Failed slices go through "Failure handling".
6. Merge: `records[first + k] = slice.records[k]`, with a check that
   every index in [0, G) was filled exactly once. The order slices
   finished in plays no part.
7. Write the manifest rows for g (below), delete the remote job and
   result files and the local `selfplay-work/gen_NNNN` (logs of failed
   slices are kept), and continue with today's code.

Transports sit behind one small interface: `upload(local, remote)`,
`run(argv, log, timeout)`, `download(remote, local)`, `remove(paths)`.
- `SshTransport` builds `scp` and `ssh` argument vectors and runs them
  with `posix_spawn`.
- `SubprocessTransport` copies files with `std::filesystem` and runs
  `exe` directly. It is used for the fake remote in tests and for a
  separate local process.
- `InProcessTransport` (tests only) calls the app's `runSliceJob`
  function on a thread.

No shell runs on the coordinator. The remote command line is built only
from validated hosts-file values and paths the coordinator generated.

### The manifest: `selfplay-engines.csv`

One row per slice per generation, in the run directory:

```
generation,first,count,planned,engine,fingerprint,build,seconds,positions,evaluations,note
12,0,52,local/metal,local/metal,"metal; Apple M5 Max; macOS 27.0.1 (26A434); Apple M5 Max, macOS 27.0.1, shaders b7b31a09",09e008b,1512.4,14822,5304113,
12,64,10,mini1/metal,local/metal,"metal; Apple M5 Max; ...",09e008b,402.0,3011,1102330,fallback: mini1: ssh exit 255
```

`planned` is the assignment; `engine` and `fingerprint` say who
actually played the slice. Together with the seed, this file is the
exact recipe for every game of the run. It is rewritten atomically
before `state.bin`. Rows of generations > g − 1 are dropped first, so
a crash between the two and the redo leave exactly one set of rows per
committed generation. The timing columns are, like `log.csv`'s, not part
of byte-identity. The default path writes no manifest.

The progress line gains one line per generation with engines on:

```
selfplay engines: local/metal 52 1512s 3.5k ev/s | local/blas 12 1490s 0.9k | mini1/metal 10 1580s 0.7k | ...
```

## Failure handling

A remote slice fails when ssh or scp exits non-zero, the slice exceeds
the host's `timeout`, or its result does not verify. Then:

1. **Retry once on the same engine.** First a best-effort `pkill -f`
   on the remote for this run-id's `selfplay` processes, so an orphan of
   a dropped connection does not compete for the GPU. A transient
   hiccup costs time, not reproducibility.
2. If the retry fails, act on **`selfplay_fallback`**:
   - `local` (default): the coordinator plays the slice itself, on the
     *fallback engine*: the first `local/...` engine in the list, or
     `local/<backend>` if there is none. It starts at once, beside
     whatever local engines are still running; a second hub shares the
     GPU as in an arena wave. The manifest records the actual engine and
     fingerprint and the reason. The log prints `note: generation 12:
     games 64-73 (mini1/metal) played by local/metal: <reason>; from here
     on results differ in the last bits from a run without the failure`.
     Later generations try the remote again.
   - `stop`: the generation throws. Nothing of it is committed
     (`state.bin` is the commit point), so after fixing the machine,
     `--resume` replays the generation as planned, byte-identical to a
     run that never failed.

### The determinism trade-off, and the choice

Three options were weighed for a failed slice:

| option | reproducible? | the run continues? |
|---|---|---|
| rerun on the *same engine type* (another machine with an equal fingerprint) | yes, exactly | only if such a machine exists and is free; the two minis are not guaranteed equal (16 vs 24 GB, possibly different macOS builds), and the peer is busy with its own slice |
| stop and let the user resume | yes, exactly | no, until someone intervenes; an overnight 19×19 run would lose the night |
| play it elsewhere and **record the engine that actually played each game** | yes, *given the manifest*: every game is a pure function of its seed, the network and its engine's fingerprint, so the run can still be replayed exactly | yes |

**Chosen:** retry, then fall back locally and record (`local`), with
`stop` for experiments that must be bit-reproducible from the config
alone. The manifest turns "the run is no longer reproducible from its
config" into "the run is reproducible from its config and its
manifest", which is what the default path already offers for backend
switches (the resume note). Same-fingerprint rerouting is left for
later. It needs a scheduler for busy peers and buys exactness only on
identical hardware.

## Build identity and keeping machines in sync

Every binary embeds a build ID. CMake runs `git rev-parse HEAD` and, if a
tracked file changed (`git status --porcelain --untracked-files=no`) or an
untracked file lies where the build globs its sources (`CMakeLists.txt`,
`cmake`, `src`, `include`, `apps`, `tests`), appends `+` and the first 8
hex digits of a SHA-256 over the status, `git diff HEAD` and those
untracked files' names and contents. Untracked files elsewhere (notes,
run directories) leave the ID clean, so every clean checkout of a commit
gets the same ID. It does this at build time, through a small script
(`cmake/azgo_build_info.cmake`, tested by `tests/build_id.sh`) that
rewrites the generated header only when the ID changes. A cache variable `AZGO_BUILD_ID` overrides it, for trees without
`.git`, such as the minis' copies made with `git archive`. A tree with
neither gets `unknown`, which the coordinator refuses for remote engines.
`azgo::buildId()` returns it; `train --version` and `selfplay --version`
print it with the build options (`metal`, `blas`).

`tools/cluster_sync.sh [--hosts FILE] [--build build-metal]` brings
every remote in the hosts file to the coordinator's commit:

1. refuses a dirty tree, since a remote cannot rebuild uncommitted changes;
2. per remote: `git archive HEAD | ssh <target> 'tar -x -C <repo>.new'`,
   then swaps directories, keeping the old build directory's CMake cache
   and objects so rebuilds are incremental;
3. configures with `-DAZ_USE_METAL=ON -DAZGO_BUILD_ID=<sha>` and builds
   with `-j <cores>`;
4. checks `selfplay --version` there against `train --version` here.

The repo path on each remote comes from the hosts file's `exe` value, as
`exe`'s grandparent directory. GNU Parallel stays a convenience for ad
hoc commands. The trainer does not use it, because it needs per-slice
exit status, timeouts, retries and verification under its own control.

## Switching an existing run on

The 19×19 run in progress was started by a binary without these keys.
To switch it on:

1. Build the new binaries (`-j 4` while the run trains) and sync the
   minis.
2. Stop the run *just after* a commit:
   `tools/stop_after_generation.sh RUN_DIR PID` polls `log.csv`, which
   is appended right after `state.bin`, and sends SIGINT to the trainer
   when a new row appears. At most the first seconds of the next
   generation's self-play are lost.
3. `train --run RUN_DIR --resume --set selfplay_engines=... [--hosts FILE]`.

The resume loads the old `config.cfg` (no engine keys: the defaults),
applies the override, preflights, rewrites `config.cfg` and prints:

```
note: self-play engines changed from local/metal to local/metal:50, local/blas=12, ...; from here on results differ in the
last bits from a run that kept local/metal
```

Generations before the switch are untouched. Switching back
(`--set selfplay_engines=`) is equally allowed and noted.

## Sizing the engines

**Fixed weights, chosen by a person, are the only input.** Any
automatic rebalancing from timings would make the assignment depend on
timing and break reproducibility. Measuring and suggesting is still
useful, so it lives outside the run:

- **`tools/engine_weights.py RUN_DIR [--last N] [--games G]`** reads
  the manifest, computes each planned engine's evaluations per second
  over the last N generations (Σ evaluations / Σ seconds; fallback rows
  excluded), and prints a suggested `selfplay_engines` value. GPU engines
  get weights proportional to their rate. CPU engines get fixed counts
  rounded to a multiple of their workers. It also prints each engine's
  predicted finish time. The user applies it with `--resume --set`.
- **`bench --mixed`** measures one machine's engines running at the same
  time: the hub with N workers, plus k BLAS workers, for k in a list, and
  reports both rates. That is the number weights need, since a CPU engine
  beside a busy hub is not the CPU engine alone. `bench` runs on each
  machine; the minis are idle and can be measured any time.

**GPU engines run every game of their slice at once** (`metal_workers =
0`). A blocked worker costs only a stack, and the hub's throughput is
flat beyond saturation (2,743 evals/s at 36 threads, 2,782 at 72). With
fewer workers than games, a GPU engine plays in waves, and its last wave
under-fills the GPU. With all games at once, its tail comes only from
game-length variance. This is the arena wave's rule
(`2026-10-10-arena-gpu-design.md`) applied to self-play engines. The
default single-engine path keeps its `workers` rule unchanged, for byte
and timing continuity.

**CPU engines play in waves of `cpu_workers` games.** Their fixed count
should be a multiple of their workers.

**A worked example** for G = 100, from the estimates above (the bench
task replaces them with measurements): Studio GPU 2,700, Studio CPU (6
workers) ~700, each mini GPU ~550, each mini CPU (8 workers) ~475; total
~5,450 evals/s. (Measured since, on the Studio: GPU 2,931 alone, 2,809
beside four utility-QoS BLAS workers that add 339, so `local/blas` is worth
about a quarter of the estimate here; on the minis: GPU 506 alone, 480 beside two
BLAS workers that add 196; their eight-worker CPU engine would give 270
and slow the GPU to 363. With the measured rates the minis' items become,
for example, `mini1/metal:9, mini1/blas=2`; `tools/engine_weights.py`
derives them from a run's manifest.)

```
selfplay_engines = local/metal:50, local/blas=12, mini1/metal:10, mini1/blas=8, mini2/metal:10, mini2/blas=8
```

The fixed counts take 28 games. The weights 50:10:10 share the other
R = 72: quotas 51.4, 10.3 and 10.3, so 51, 10 and 10, and the one
leftover game goes to the largest remainder, `local/metal`. The slices
are 52, 12, 10, 8, 10 and 8 games, each engine's count roughly
proportional to its rate. Expected self-play time ≈ the evaluations of
100 games / 5,450 per second, about 2× faster than the Studio GPU alone
before tails.

**When G is small** (the 19×19 preset's 20), six engines cannot be
balanced to better than a game or two each. Use fewer engines (for
example GPUs only), or more games per generation. The tool warns when an
engine gets fewer games than its workers.

## Determinism contract

Let a game be identified by (seed, generation, index). Its record is a
pure function of:

1. its seed stream, `deriveSeed(seed, generation, kPhaseSelfPlay, index)`;
2. the self-play network's bytes;
3. the `SelfPlayConfig`, this generation's v_resign included;
4. the **fingerprint and build ID of the engine that played it**.

It does not depend on worker counts, `inference_batch`, the other games
sharing the engine, the number of engines, the transport, retries,
which slice finished first, or timing. Items 1–3 are the existing
argument. Item 4 is where the CPU and GPU, and an M4 and an M5 Max, may
differ. Within one engine, the existing per-worker (CPU) and
batch-independence (GPU) arguments hold unchanged, because a slice runs
exactly today's code.

Hence, with `selfplay_engines` set:

- **Reproducible.** Two runs with the same config, in which every slice
  was played by its planned engine and every alias resolved to a machine
  with the same fingerprint and build ID, write byte-identical
  checkpoints, `state.bin`, `results.csv`, `elo.csv`, `references.csv`,
  SGFs, `log.csv` (timings aside) and manifest (timings aside).
- **Resume is byte-identical** under the same conditions: a generation
  is committed whole, and a redone generation uses the same assignment.
- **Splitting is free within a fingerprint.** `local/blas:2, local/blas:1`
  equals `local/blas:1`, and two identical minis can split a share any
  way. Changing the counts between engines with *different* fingerprints
  changes which arithmetic plays which games, and so the results.
- **The default is today's run.** With `selfplay_engines` empty, a run
  is byte-identical to `main` for every backend. `selfplay_engines =
  local/<backend>:1` with the same `workers` is byte-identical to the
  default.
- **After a fallback**, the run is reproducible from its config and its
  manifest, not from its config alone.

**Not guaranteed:**

- Equality with a single-machine run, or between `metal` and `blas`.
- Equality across an OS update, a different chip or a different build
  on any engine. The preflight line and the manifest show each engine's
  fingerprint and build ID, and a change prints a note.
- That the engines named by aliases are the same machines next month.
  The manifest records fingerprints, not aliases alone.

## Configuration and CLI summary

| setting | where | default | affects results? |
|---|---|---|---|
| `selfplay_engines` | config key, tunable on resume | empty (today's path) | yes: which engine plays which game |
| `selfplay_fallback` | config key, tunable | `local` | only when a slice fails |
| hosts file (`ssh`, `exe`, `workdir`, workers, `inference_batch`, `cpu_qos`, `nice`, `timeout`) | `--hosts`, `$AZGO_HOSTS`, `~/.config/azgo/hosts` | — | no |
| `train --version`, `selfplay --version` | CLI | — | — |
| `train --hosts FILE` | CLI | — | no |
| `selfplay --job --out --backend [--workers] [--inference-batch]` | CLI | — | no (workers, batch) |
| `bench --mixed --hub-workers N --cpu-workers 0,2,4,8` | CLI | — | — |

## Error handling

| situation | behaviour |
|---|---|
| bad `selfplay_engines` value, fixed counts > G (or ≠ G with no weighted item) | `ConfigError` at start or resume, nothing written |
| a remote alias not in the hosts file; a hosts file with bad keys or values | `ConfigError` naming the alias or line |
| remote build ID ≠ coordinator's | `ConfigError`, suggesting `tools/cluster_sync.sh` |
| remote unreachable or backend unusable at preflight | `stop`: `ConfigError`; `local`: warning, slices fall back |
| slice fails (exit status, timeout, verification) | retry once on the same engine, then fallback or stop (above) |
| a local engine throws (e.g. GPU failure) | the generation fails as today; `state.bin` stays at g − 1 |
| the coordinator dies mid-generation | as today: `--resume` redoes the generation. Remote orphans are killed by the next attempt's `pkill`, and their files are named by attempt |
| disk full on a remote | the slice fails (scp or the atomic write), handled as above |

## Alternatives considered

| option | verdict |
|---|---|
| dynamic work-stealing across all engines, recording who played what | rejected: the fastest, but the assignment depends on timing. A rerun or resume gives other bits, and the run is reproducible only from its manifest, always rather than only after a failure |
| work-stealing only among engines with equal fingerprints | later: exact, but needs equal machines and a shared queue across processes; the gain is a smaller tail |
| automatic weights from the previous generation's timings | rejected: timing-dependent assignment. The tool suggests, a person applies |
| one mixed `WorkerEvaluators` pool | rejected: `parallelFor` hands indices to any free worker, so engine choice would depend on timing; one pool per engine instead |
| `train --selfplay-slice` instead of a new app | rejected: the remote needs no run directory, and `train`'s CLI is already the largest |
| GNU Parallel or a shell script as the launcher | rejected for the trainer: per-slice status, timeouts, retries and verification belong in tested code; kept for ad hoc operations |
| a text job file (the run's `config.cfg` plus extra lines) | rejected: floats re-parsed from text must round-trip exactly, and a binary job states the exact bits once |
| SGFs written by the remotes | rejected: the coordinator writes them from the merged records, as today |
| stream records per game so a dead slice's finished games are kept | later: salvage is exact (those games were played by the planned engine), but whole-slice retry is simpler, and failures are rare |
| distributing training | out of scope: ~10% of a 19×19 generation, and gradients would need exact reductions across machines |

## Testing

Ctest-level tests run on one machine. A "remote" there is the
subprocess or in-process transport.

- **`test_distributed`** (new, every build):
  - `selfplay_engines` parse, print and round-trip; refusals;
    `checkConsistency` rules.
  - `assignGames`: counts sum to G; slices are contiguous and in order;
    the largest-remainder rule and its ties; fixed counts; zero-game
    engines; the result never depends on anything but its inputs.
  - Hosts file: parse, defaults, search order, refusals (unknown key,
    repeated alias, `ssh` on `local`, metacharacters).
  - `playSelfPlayGames`: a slice [a, b) equals indices a..b−1 of the
    whole [0, G), for workers 1 and 3, naive (and blas when built).
  - Job and result round-trip, every field bitwise. Each refusal: magic,
    version, checksum (one flipped byte anywhere), truncation, job hash,
    build ID, indices, a record whose stones, ownership, result, legality,
    flags or lows are wrong.
  - Merge: slices delivered in every order give the same records; a
    missing or doubled index is refused.
  - Subprocess transport end to end: the real `selfplay` binary's result
    equals the in-process slice, bitwise.
  - `runProcess`: exit status, log capture, timeout (SIGTERM then
    SIGKILL), a missing executable.
  - Ssh transport: the argument vectors it builds, without running them.
- **`test_trainer`** (naive in every build, blas when built):
  - `selfplay_engines = local/naive:1` is byte-identical to the default.
  - `local/naive:2, local/naive=1` is byte-identical to the default.
  - A fake remote (`fake/naive` through the subprocess transport) plus
    `local/naive` is byte-identical to the default, since both have the
    same fingerprint. It writes a manifest naming both engines.
  - Resume with engines on is byte-identical to uninterrupted.
  - Switching on at a boundary: generations before it equal the default
    run's, the note is printed, and the switch is recorded in
    `config.cfg`.
  - Failure injection through `distributed_probe::failSlice(engine,
    times, mode)` with modes exit, timeout and corrupt result: one
    failure is retried, with the same bits; two fall back, the manifest
    records it, and the run continues; with `stop`, the trainer throws,
    `state.bin` stays at g − 1, and a resume equals the failure-free run.
  - Preflight refusals: build-ID mismatch (`distributed_probe::
    setRemoteBuildId`), an unknown alias, an unusable backend. Nothing is
    written.
- **`test_metal`**: `local/metal:1` with `metal_workers` 0, 8 and 64 is
  byte-identical (all games at once changes nothing).
- **Old against new binaries** (default path): `main`'s `train` and
  the branch's write byte-identical checkpoints, `state.bin`, rating
  files, SGFs and `log.csv` without timings, for 5×5 and 9×9, naive,
  blas and metal. `config.cfg` differs only by the two new lines.
- **`integration_smoke.sh`**: `selfplay --version` and `--fingerprint`;
  a tiny run with a fake remote through a temporary hosts file, resumed;
  the manifest; a refused version mismatch.
- **Real three-machine smoke** (manual, recorded in
  `docs/results/bench.txt`): a tiny 5×5 run with engines on all three
  machines and both backends. Two such runs are byte-identical, a resume
  is byte-identical, and killing a remote `selfplay` mid-slice falls
  back with the note. Under `stop` it stops, and the resume equals the
  failure-free run.

## Performance

**Acceptance**, on the 19×19 10×128 network with the measured weights:

- self-play phase ≥ 1.6× faster than the Studio GPU alone (ideal ~2.0×;
  the margin covers tails, transfers and CPU engines beside the hub);
- the Studio GPU engine's rate beside the local CPU engine ≥ 90% of its
  rate alone (`bench --mixed`; measured: 95.8% with the default 4 workers
  at utility QoS, 96.2% at default QoS);
- the desktop stays usable during self-play (Mark's judgement; no
  freezes, no input lag in the terminal and browser);
- transfers and launch ≤ 2% of the self-play phase.

Measurements on the Studio need the 19×19 job paused (SIGSTOP/SIGCONT)
and Mark's go-ahead, as before. The minis can be measured at any time.

## Later: the arena

The same machinery can carry the arena's waves: a match (or a group of
games) is a slice with two networks and an arena seed instead of
self-play's. It needs a second job type, several networks per job and
reference networks on every remote, for ~10% of a generation. It is not
part of this design. The coordinator's interfaces (transport, verify,
merge by index, manifest) are built so that adding it does not change
them.

## Teaching material

- **README**: a "Distributed self-play" section after **Backends**. It
  covers engines, the key, the hosts file with placeholders (never real
  addresses), `cluster_sync.sh`, the determinism contract in five lines,
  the failure policy and the switching recipe. `selfplay` joins the list
  of tools. "Learn how it works" links this spec and its plan.
- **Explainer**: a short passage in the self-play chapter, "More than
  one machine". It explains why game-by-index seeding made distribution
  almost free, and why the engine must still be fixed by index. It quotes
  `assignGames` and the merge.
- `tools/check_citations.py` keeps passing.

## Risks

| risk | consequence | mitigation |
|---|---|---|
| a CPU engine slows the GPU engine on the same machine (dispatcher or search threads starved, shared memory bandwidth) | less gain than the sum | `bench --mixed` measures it; default local CPU workers are few; QoS keeps the GPU feeders at default class |
| `utility` QoS moves CPU workers to efficiency cores on chips that have them | lower CPU-engine rate | measured; the weights absorb it; `cpu_qos=default` per machine |
| the GPU at 100% makes the desktop stutter (the compositor shares the GPU) | the freeze Mark saw may not be CPU-bound | passes are short (tens of ms); if it recurs, a smaller `inference_batch` on `local` shortens them; measured in the acceptance run |
| tails: one slow engine holds the generation | less gain | GPU engines play all games at once; CPU counts are multiples of workers; the weights tool; enough games per generation |
| a mini's macOS updates silently | its games' bits change | fingerprint in preflight and manifest; a note on change |
| binaries from different commits | records from different code | build-ID check at preflight and in every result; `cluster_sync.sh` |
| a stray remote process from a dropped connection | competes for the GPU, or writes a stale file | `pkill` before retry; per-attempt file names; job hash in the result |
| little free disk on the 16 GB mini | a slice cannot write | results of a few MB, deleted after each generation; the network kept only for the current hash |
| hosts or user names leak into the repo | privacy (redacted once already, commit 03bece2) | the hosts file lives outside the repo; README uses placeholders; the manifest and logs use aliases |
| verification misses a subtle corruption | a bad game enters the buffer | checksum plus a full replay of every record; resign lows checked like `state.bin`'s |
