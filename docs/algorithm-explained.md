# AlphaZero for Go, from Scratch

How a residual network, a batched tree search and a few thousand games
against itself learn Go on boards from 5×5 to 19×19 — walked through
against the actual C++ in this repository, chapter by chapter, in the
order the code runs. This is the sequel to `alphazero-tictactoe`: where an
idea was already explained there (PUCT, the sign flip, the Monte Carlo
return), this document recaps it in a few lines and spends its words on
what Go changes.

*(An illustrated HTML rendering of this document, with diagrams and the
Elo plots, is `docs/algorithm-explained.html`.)*

**Language:** C++17, no dependencies (BLAS optional) · **Network:** a
ResNet, 3 blocks × 32 filters on 5×5 up to 10 × 128 on 19×19 ·
**Result:** on 5×5, in both of two runs, the final network wins 100 of
100 games against the untrained one and puts 97% of its prior on the
center point, the first move of 5×5's known perfect play; on 9×9, after
one night, generation 150 beats generation 50 in 98 of 100 games, though
it still misreads a basic two-eye shape

**About the numbers.** Everything measured was measured on one Apple M4
with 10 cores (`docs/results/bench.txt`), and every number names the file
it came from. The 5×5 numbers come from two complete 60-generation runs,
seeds 1 and 2, whose `log.csv`, `elo.csv` and `results.csv` are kept in
`docs/results/5x5/` and `docs/results/5x5-seed2/`. The 9×9 and 19×19
numbers come from `docs/results/9x9/` (an overnight run) and
`docs/results/19x19/` (a two-generation check). All of these runs were
made with the final code. A first set, made before the search fix
described in chapters 02 and 07, is kept in `docs/results/passbug/`;
the short pilots in `docs/results/pilots/`, used to size the presets,
also predate that fix. `docs/results/README.md` summarizes everything.

## Contents

- [00 — The state](#00--the-state)
- [01 — The two-headed guess](#01--the-two-headed-guess)
- [02 — Search corrects the guess](#02--search-corrects-the-guess)
- [03 — Batching the search](#03--batching-the-search)
- [04 — Self-play and exploration](#04--self-play-and-exploration)
- [05 — Assigning credit](#05--assigning-credit)
- [06 — Learning from itself](#06--learning-from-itself)
- [07 — Measuring progress without an oracle](#07--measuring-progress-without-an-oracle)
- [08 — The full loop at scale](#08--the-full-loop-at-scale)
- [09 — What changed from tic-tac-toe](#09--what-changed-from-tic-tac-toe)
- [10 — Concept map](#10--concept-map)

---

## 00 — The state

**Concept: the rules as a state machine, and what the network sees**

Tic-tac-toe has 5,478 legal positions; you can print them all. Go on a
5×5 board already has about 4×10¹¹, and on 19×19 about 2×10¹⁷⁰. Nothing
in this project enumerates positions. What it needs from the rules is
much smaller: given a position, which moves are legal, what does a move
do, and who won when the game ends. This chapter is those rules, stated
the way a programmer needs them, and then the encoding that turns a
position into numbers.

**Groups, liberties, capture.** Stones of one color that touch
horizontally or vertically form a *group*. The empty points next to a
group are its *liberties*. A move places a stone on an empty point (or
passes). Then every opponent group left with no liberties is removed —
*captured*. In the diagram below (the same frame `analyze` prints, GTP
coordinates, `X` Black, `O` White), White's stone at C4 has one liberty
left, C5. It is "in atari", and Black C5 would capture it.

```
   A B C D E
 5 . . . . .  5
 4 . X O X .  4
 3 . . X . .  3
 2 . . . . .  2
 1 O . . . O  1
   A B C D E
```

**Suicide.** If, after removing captured opponent stones, the moved
stone's own group has no liberties, the move is *suicide*. Strict
Tromp-Taylor rules allow it (the suicidal group simply disappears);
most human rulesets forbid it, AlphaGo Zero did, and so does this code.
A suicide can only ever lose stones, so forbidding it
removes a whole family of pointless moves from every search without
costing any strength. Note the order of the check: captures first, then
the liberty test. A move into a point with no empty neighbors is legal
if it captures something, because the capture frees a liberty.

`Board::isLegal` answers "is this move locally legal" without playing
it, by looking at the four neighbors:

```cpp
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
```
*src/board.cpp:131 — `Board::isLegal`*

`countLiberties(q, 2)` flood-fills the group at `q` and stops as soon as
it has found two liberties: legality only ever needs to know "none",
"exactly one" or "at least two". Groups are not maintained incrementally
(no union-find, no chain lists). A flood fill over at most 361 points
takes microseconds, while one network evaluation takes tens of
microseconds on 5×5 and milliseconds on 19×19 (chapter 01), so the
simpler code costs nothing that matters. `bench` measures the rules
engine alone at 1.3 million random moves per second on 5×5 and 178,000
on 19×19, including legality, superko and hashing
(`docs/results/bench.txt`).

Playing the move is the same logic with side effects — place the stone,
then remove each neighboring opponent group left without a liberty:

```cpp
int Board::play(int action) {
    assert(isLegal(action));
    /* ... a pass only flips toMove_; otherwise p = pad(action), me, them ... */
    cells_[p] = me;
    hash_ ^= zobristKey(p, me);
    int captured = 0;
    for (int d : kDirs) {
        const int q = p + d;
        // A group already removed via another neighbor reads Empty here.
        if (cells_[q] == them && countLiberties(q, 1) == 0) captured += removeGroup(q);
    }
    /* ... assert the new stone has a liberty, flip toMove_, return captured ... */
}
```
*src/board.cpp:152 — `Board::play` (abridged)*

An illegal move reaching `play` is a programming error, hence the
`assert`; moves typed by a person are checked first (`analyze` reports
`illegal (occupied)`, `illegal (suicide)` or `illegal (superko)`).

**The padded array and the `Off` sentinel.** Board size is a runtime
value from 5 to 19, but the storage is always a 21×21 array: the largest
board plus a one-point border. Every cell outside the playing area holds
the color `Off`.

```cpp
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
```
*src/board.cpp:47 — `Board::Board`*

The four neighbors of a padded index `p` are `p − 21`, `p − 1`, `p + 1`
and `p + 21` (`kDirs`). For any on-board point all four are valid array
indices, and an off-board neighbor reads `Off`, which is neither empty
nor a stone — so the flood fills above never need a bounds check, and a
stone on the edge simply has fewer liberties, which is exactly Go's
rule. One `Board` type serves every size, and it is a plain value (441
bytes of cells plus a few ints) that search copies freely.

**Ko and positional superko.** Captures make repetition possible.
Here Black has just captured a White stone at B3 by playing C3 (left:
before, right: after):

```
   A B C D E                A B C D E
 5 . . . . X  5          5 . . . . X  5
 4 . X O . .  4          4 . X O . .  4
 3 X O . O .  3   --->   3 X . X O .  3
 2 . X O . .  2          2 . X O . .  2
 1 . . . . .  1          1 . . . . .  1
```

Black's new stone at C3 has one liberty, B3. If White could immediately
play B3 again, it would capture C3 and recreate the left-hand position
exactly, and the two players could trade captures forever. This shape
is a *ko*. The simple ko rule forbids the immediate recapture; this code
uses the stronger *positional superko* rule, the one Tromp-Taylor
specifies: **a move may not recreate any whole-board arrangement of
stones that has occurred earlier in the game.** It also covers the rare
longer cycles (triple ko and friends) that the simple rule misses.

Checking "has this position ever occurred" naively means comparing
boards. Instead, every position carries a **Zobrist hash**: one random
64-bit key per (point, color), generated once from a fixed seed, and the
hash of a position is the XOR of the keys of all its stones. Placing a
stone XORs its key in, removing one XORs it out, so the hash updates in
O(1) per stone. `hashAfter` computes the hash a move *would* produce
without copying the board, unless the move captures:

```cpp
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
```
*src/board.cpp:174 — `Board::hashAfter`*

A single `Board` has no memory of the past, so superko lives one level
up, in `Game`, which keeps every position of the game and a hash set of
the positions seen so far:

```cpp
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
```
*src/game.cpp:19 — `Game::isLegal`*

The hash covers the stones only, not the side to move: "positional"
superko compares arrangements of stones. A hash collision (two different
positions with the same 64-bit hash) would wrongly forbid one legal
move; with 2⁶⁴ possible values and at most a few thousand positions per
game, that is not a practical concern.

**Try it** — the retake in the diagram above, typed as a move list
(`analyze` checks every move before it plays it):

```
$ ./build-blas/analyze --checkpoint runs/5x5-seed1/checkpoints/gen_0060.azn \
      --komi 24.5 --moves "B4 C4 A3 C2 B2 D3 E5 B3 C3 B3"
error: move 10 (B3): illegal (superko)
$ ./build-blas/analyze --checkpoint runs/5x5-seed1/checkpoints/gen_0060.azn \
      --komi 24.5 --moves "B1 C3 A2 D3 E5 A1"
error: move 6 (A1): illegal (suicide)
```

**Scoring: Tromp-Taylor area and komi.** The game ends after two
consecutive passes (or at a move cap of 2·n² moves, so that self-play
always finishes). The final position is then scored by **Tromp-Taylor
area scoring**: each player gets one point per stone on the board plus
one point per empty point in regions that touch only that player's
stones. Empty regions that touch both colors count for nobody. There is
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

Black moves first, which is worth several points on 19×19, so White
receives **komi**: `Game::scoreForBlack()` is `areaScore() − komi`, and
the default komi is 7.5. The half point means a draw is impossible; an
integer komi can produce an exact tie, which this code scores as a draw
(chapter 05). How much the first move is worth depends on the board,
and on 5×5 it is worth everything: 5×5 Go is solved, and perfect play
gives Black the entire board, a win by 25 points. That is why the 5×5
preset uses komi 24.5 instead of 7.5 — chapter 05 has the evidence.

**What the network sees: input planes.** The network does not read a
`Board`. It reads a stack of n×n planes of 0s and 1s, always from the
point of view of the player about to move — the Go version of
tic-tac-toe's "my stones / their stones" canonicalization:

- for each of the last H positions (H = `history`, 8 in every preset):
  one plane of the side-to-move's stones and one of the opponent's
  stones — 2H planes;
- a "Black to move" plane and a "White to move" plane, exactly one of
  which is all ones.

That is 2H + 2 = 18 planes. "Own" means the player to move *now*, in
every history plane: the network sees the recent past through the eyes
of the player who has to act.

```cpp
void encodePlanes(const std::vector<const uint8_t*>& stones, int n, Color toMove,
                  int H, int sym, float* out) {
    /* ... asserts; zero the whole output; dest[p] = transformPoint(sym, p, n) ... */
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
```
*src/encode.cpp:14 — `encodePlanes` (abridged)*

Why history at all? The current stones alone do not say which moves
superko forbids, or which stones were just captured; the last few
positions do. Positions before the first move are all-zero planes. The
`sym` argument rotates or mirrors the output (chapter 06); search always
passes 0, the identity.

Why *two* color planes, one of them all ones? Because of how the first
layer reads the board. It is a 3×3 convolution with *zero padding*: at
an edge point, the three neighbors that would lie off the board are read
as 0. In the stone planes, an empty point is also 0 — so without help,
an empty point on the edge looks exactly like an empty point in the
middle of the board, and the network cannot tell where the edge is. A
plane that is 1 on every on-board point breaks the tie: the convolution
sees 1s inside the board and padding 0s outside it, so every 3×3 window
that overlaps the edge produces a different response. AlphaGo Zero used
17 planes with a single color plane that is all ones for Black to move
and all zeros for White; with that encoding the edge signal vanishes
whenever White is to move. Two planes, exactly one of them all ones,
keep it on every turn and still say whose turn it is.

---

## 01 — The two-headed guess

**Concept: function approximation with a residual convolutional network**

As in tic-tac-toe, one network answers two questions about a position:
a **policy** (a probability for each move: which moves look good) and a
**value** (a number in [−1, 1]: the expected result for the player to
move). Search uses the first to decide where to look and the second to
stop looking.

Tic-tac-toe's network was a multilayer perceptron, 18 → 64 → 64 →
{9, 1}. That design does not survive contact with Go. On 19×19 the input
has 18 × 361 = 6,498 numbers; a fully connected first layer of any
useful width would need millions of weights, and worse, it would have to
learn "a stone in atari can be captured" separately at every one of 361
locations. Go's patterns mean the same thing wherever they occur (the
edge aside, which the ones plane from chapter 00 makes visible). A
**convolution** exploits that: one small filter, 3×3 points across all
input planes, slides over the whole board, so a pattern learned once is
recognized everywhere.

This is AlphaGo Zero's architecture, at a size a CPU can train:

```
input planes [18][n][n]
   │  stem: 3×3 conv (F filters) → batch norm → ReLU
   ▼
trunk [F][n][n] ── residual block, repeated `blocks` times:
   │      3×3 conv → BN → ReLU → 3×3 conv → BN → (+ block input) → ReLU
   ▼
trunk [F][n][n]
   ├─ policy head: 1×1 conv (2 filters) → BN → ReLU → Linear(2n² → n²+1) → logits
   └─ value head:  1×1 conv (1 filter)  → BN → ReLU → Linear(n² → V) → ReLU
                                                → Linear(V → 1) → tanh → value
```

The presets set the number of residual blocks (`blocks`), the filter
count F (`filters`) and the value-head width V (`value_hidden`) per
board size (in this document B always means the batch size): 3 × 32 with V = 32 on 5×5, 6 × 64 with V = 64 on 9×9, and
10 × 128 with V = 256 on 19×19 (`configs/*.cfg`). AlphaGo Zero used
20 or 40 blocks of 256 filters.

**Convolution as im2col + gemm.** A convolution could be written as six
nested loops, but nearly all of a ResNet's time goes into one operation,
and it pays to make it a single big one: general matrix multiply, *gemm*.
The trick is **im2col**: copy every 3×3 neighborhood of the input into a
column of a matrix, then the whole convolution is one multiplication of
the weight matrix by that column matrix.

```cpp
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
```
*src/layers.cpp:34 — `Conv2d::forward`*

**Worked example** — the one in `tests/test_layers.cpp` and
`tests/test_backend.cpp`: a single 3×3 input plane holding 1…9, and one
3×3 filter of all ones, so each output is the sum of a 3×3 neighborhood.

```
input          im2col: one column per output pixel, one row per filter offset
1 2 3                  pixel:  0  1  2  3  4  5  6  7  8
4 5 6          (-1,-1)         0  0  0  0  1  2  0  4  5
7 8 9          (-1, 0)         0  0  0  1  2  3  4  5  6
               (-1,+1)         0  0  0  2  3  0  5  6  0
               ( 0,-1)         0  1  2  0  4  5  0  7  8
               ( 0, 0)         1  2  3  4  5  6  7  8  9
               ( 0,+1)         2  3  0  5  6  0  8  9  0
               (+1,-1)         0  4  5  0  7  8  0  0  0
               (+1, 0)         4  5  6  7  8  9  0  0  0
               (+1,+1)         5  6  0  8  9  0  0  0  0

weights [1 1 1 1 1 1 1 1 1]  ×  columns  =  12 21 16 27 45 33 24 39 28

output         12 21 16
               27 45 33
               24 39 28
```

Column 0 is pixel (0,0)'s neighborhood with the five off-board entries
filled with zeros — the zero padding. The 1×9 weight row times the 9×9
column matrix gives all nine outputs at once. With C input planes the
column matrix has 9C rows, and with F filters the weight matrix has F
rows, so a whole layer is one [F × 9C] · [9C × n²] product.
`Backend::im2col`, which builds the columns, is plain memory shuffling,
and so is its adjoint `col2im`; both are virtual with a shared default
implementation. `gemm` is the one pure-virtual operation, the one each
backend must supply: blocked loops in the naive backend, a call to
`cblas_sgemm` in the BLAS backend (Apple Accelerate on macOS, which
runs it on the CPU's matrix units).

**Channel-major layout.** Every activation in the network is stored as
`[channels][B][n·n]`: all of channel 0 for every sample in the batch,
then all of channel 1, and so on. That is not the order the caller
supplies (`[B][planes][n·n]`, one sample after another), so `forward`
transposes the input once:

```cpp
void Network::forward(Workspace& ws, const float* input, int B, Mode mode) {
    /* ... be = backend, n, A = n * n, P = planes, M = B * A, col, convOut ... */
    // [B][planes][A] -> [planes][B][A]. Channel-major lets a conv treat the
    // whole batch as one wide matrix: one gemm per layer, not one per sample.
    for (int b = 0; b < B; ++b)
        for (int p = 0; p < P; ++p)
            std::copy(input + (static_cast<size_t>(b) * P + p) * A,
                      input + (static_cast<size_t>(b) * P + p + 1) * A,
                      ws.input_.data() + (static_cast<size_t>(p) * B + b) * A);

    /* ... norm() runs a batch norm in this call's mode (below); the stem is
           conv3x3 -> norm -> ReLU; then, for each residual block: ... */
        blk.conv1.forward(be, x, B, n, col, convOut);
        norm(blk.bn1, convOut, buf.act1.data(), buf.xhat1.data(), buf.invStd1.data());
        reluForward(buf.act1.data(), trunk);
        blk.conv2.forward(be, buf.act1.data(), B, n, col, convOut);
        norm(blk.bn2, convOut, buf.out.data(), buf.xhat2.data(), buf.invStd2.data());
        for (size_t j = 0; j < trunk; ++j) buf.out[j] += x[j];   // the skip connection
        reluForward(buf.out.data(), trunk);
    /* ... policy head and value head ... */
}
```
*src/network.cpp:177 — `Network::forward` (abridged)*

With this layout the column matrix of a batch is just wider: M = B·n²
columns instead of n², and one gemm computes the layer for every sample
at once. It also makes batch norm, which works one channel at a time,
read one contiguous run of B·n² floats per channel.

**Batch norm, and the bug you get if you mix up its two modes.** After
each convolution, batch norm rescales every channel to mean 0 and
variance 1, then applies a learned scale γ and shift β. It is what makes
a deep stack of convolutions trainable at a reasonable learning rate. The
question is: mean and variance *of what*? In training, of the current
minibatch, per channel, over all B·n² values:

```cpp
void BatchNorm2d::forwardTrain(const float* x, int M, float* y, float* xhat, float* invStd,
                               bool updateRunning) {
    assert(M > 1);
    for (int ch = 0; ch < c; ++ch) {
        /* ... mean and biased variance of channel ch over its M values, in double ... */
        const float inv = static_cast<float>(1.0 / std::sqrt(var + kEps));
        invStd[ch] = inv;
        /* ... y = gamma * (x - mean) * inv + beta, saving xhat = (x - mean) * inv ... */
        if (updateRunning) {
            // Inference sees one position at a time, so it needs population
            // estimates: unbiased variance, exponential moving averages.
            const double unbiased = var * M / (M - 1);
            runMean[ch] = (1.0f - kMomentum) * runMean[ch] + kMomentum * static_cast<float>(mean);
            runVar[ch] = (1.0f - kMomentum) * runVar[ch] + kMomentum * static_cast<float>(unbiased);
        }
    }
}
```
*src/layers.cpp:85 — `BatchNorm2d::forwardTrain` (abridged)*

In inference — every evaluation during search — it uses the running
averages collected during training instead, which reduce to one fixed
scale and shift per channel:

```cpp
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
```
*src/layers.cpp:74 — `BatchNorm2d::forwardInfer`*

Mixing the two up fails in two different, quiet ways. Use batch
statistics during search, and a position's evaluation depends on which
other leaves happen to share its batch of 8 — or, at batch 1, every
channel of a lone position gets normalized to mean 0 over its own n²
points, erasing exactly the "is there more of this feature here than
usual" information the network relies on. Forget to update the running
statistics during training, and inference normalizes with the initial
mean 0 and variance 1, so every layer sees activations at a scale it was
never trained on. Neither bug crashes and both still produce legal
moves. That is why `Network` has three explicit modes — `Infer` (running
statistics), `Train` (batch statistics, running statistics updated) and
`Probe` (batch statistics, nothing updated, for gradient checks) — and
why `tests/test_network.cpp` checks that inference is per-sample: the
same position evaluated alone and inside a batch of four gives the same
output.

**Residual blocks.** Each block computes `relu(x + F(x))`, where F is
two convolutions: the block learns a *correction* to its input rather
than a whole new representation. The payoff is in the backward pass.
Gradients flowing back through a plain stack of layers are multiplied by
each layer's Jacobian in turn and tend to shrink or blow up with depth;
through a residual block they also flow along the skip connection
unchanged. You can see that line in the backward code below: `dTrunk`
passes straight through and the convolution path adds to it. AlphaGo
Zero's 40-block network would be very hard to train without it; this
project's 3 to 10 blocks benefit the same way.

**Growing a network without changing it.** A residual block whose
correction F(x) is exactly zero is an exact identity: the block computes
`relu(x + 0)`, and x is already the output of a ReLU, so `relu(x) = x`.
F ends in a batch norm, `y = gamma * xhat + beta`, so setting `gamma = 0`
and `beta = 0` in a block's second norm forces F(x) = 0 whatever its
convolutions hold. `growBlocks` appends blocks built this way, so a
trained 6-block network becomes a 10-block network that computes exactly
the same policy and value, and training continues from there instead of
from scratch. The new blocks are not stuck: the gradient of the loss with
respect to `gamma` is `sum(dy * xhat)`, which does not depend on `gamma`,
so `gamma` and `beta` move on the first step, and once `gamma` is nonzero
the gradient reaches the convolutions too.

```cpp
Network growBlocks(const Network& net, int newBlocks, uint64_t seed) {
    /* ... throw if newBlocks < net's blocks; build Network(cfg, seed) ... */
    for (const ParamView& p : grown.params()) {
        if (name exists in net)         copy it;       // old blocks keep their indices
        else if (p is "...bn2.gamma")   fill with 0;   // new block: F(x) = 0
    }
    /* ... copy the BN running statistics the same way ... */
}
```
*src/grow.cpp:16 — `growBlocks` (abridged)*

**Widening it.** Width grows the same way, by copying instead of zeroing
(Net2WiderNet). A new channel starts as a copy of an old one: the layer
that writes it gets the old channel's convolution weights and batch-norm
parameters, so it carries exactly the same values. Every layer that
reads the old channel now sees it more than once, so `widenFilters`
splits each weight on it among the copies in random positive shares that
sum to the old weight, and the sum the layer computes is unchanged.
Halving would be exact too, but then the copies would get equal
gradients forever and stay one feature; unequal shares send them
different gradients, and they grow apart. The stem and every block's
second convolution all write the residual stream, and the skip
connections add up their writes, so they must all copy the same
channels.

```cpp
Network widenFilters(const Network& net, int newFilters, uint64_t seed) {
    /* ... throw if narrower or above kMaxFilters; build Network(cfg, seed) ... */
    g = channelMap(F, newFilters, rng);       // stream channel j copies old g(j)
    stem conv rows and stem.bn: from g;
    for (each block) {
        h = channelMap(F, newFilters, rng);   // the block's own inner channels
        conv1: rows from h, inputs split over g;   bn1 from h;
        conv2: rows from g, inputs split over h;   bn2 from g;
    }
    policy and value 1x1 convs: inputs split over g;   the rest copied as is
}
```
*src/grow.cpp:158 — `widenFilters` (abridged)*

**The heads.** The policy head squeezes the trunk to 2 channels with a
1×1 convolution and then a fully connected layer maps those 2n² numbers
to n² + 1 logits: one per point plus one for **pass**. It knows nothing
about legality — occupied points get logits too — so something
downstream must mask them (chapter 02). The value head squeezes to one
channel, then a hidden layer of V units and a single `tanh` output, so
the value lives in the same [−1, 1] units as the game result it is
trained against.

**What depth and width buy, and what they cost.** Each 3×3 convolution
widens the window a unit can see by 2 points, so the trunk's receptive
field is 1 + 2 × (number of 3×3 convs) = 1 + 2(2·blocks + 1) points wide: 15 on
5×5, 27 on 9×9 and 43 on 19×19. In each preset that is at least
2n − 1, enough for every point to be influenced by every other, even
corner to corner. Width (F) is how many distinct features each point can
carry. A 3×3 layer costs 9 · F² · n² multiply-adds per position, so the
trunk costs roughly blocks · F² · n²: 10 · 128² · 361 ≈ 59.1 million for
19×19's preset against 3 · 32² · 25 = 76,800 for 5×5's, about 770 times
the arithmetic per evaluation. Measured on the M4
(`docs/results/bench.txt`; inference at batch 8 on one thread; a
training step at batch 128, which the naive backend splits across all
10 cores while BLAS runs on one thread):

| board | blocks × filters | backend | evaluations/s | training step |
|---|---|---|---|---|
| 5×5 | 3 × 32 | naive | 10,544 | 28.0 ms |
| 5×5 | 3 × 32 | BLAS | 30,488 | 12.1 ms |
| 9×9 | 6 × 64 | naive | 402 | 490.3 ms |
| 9×9 | 6 × 64 | BLAS | 3,696 | 149.0 ms |
| 19×19 | 10 × 128 | naive | 13 | 10,649.8 ms |
| 19×19 | 10 × 128 | BLAS | 229 | 2,393.5 ms |

Two things stand out. First, the 5×5 → 19×19 slowdown is about 130×
with BLAS, not 770×: at 5×5 the matrices are so small that fixed costs
per call dominate. Second, the BLAS advantage grows with the matrices,
from 2.9× on 5×5 to 18× on 19×19 — the matrix unit needs big products to
pay off. (Even the training step, where the naive gemm gets ten cores
and BLAS one, is 2.3–4.4× faster with BLAS.) Chapter 03 makes the same argument about batch
size.

**Backprop through convolution, batch norm and linear layers.** The
forward pass was hand-written, so the backward pass is too — the chain
rule, layer by layer, in reverse. For a convolution `y = w · columns`,
the weight gradient is `dy · columnsᵀ`, and the input gradient is
`wᵀ · dy` folded back onto the pixels it came from by `col2im` — the
adjoint of `im2col`, which *adds* every column entry back into the input
position it was copied from, so overlapping patches sum:

```cpp
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
```
*src/layers.cpp:46 — `Conv2d::backward`*

Three gemms per convolution per step, all through the same `Backend`
call as the forward pass, so the BLAS speedup applies to training too.
Batch norm's backward (`BatchNorm2d::backward`) is the least obvious
one: every input in a channel moved that channel's batch mean and
variance, so every input's gradient collects a share of every output's
gradient, `dx = γ·invStd/M · (M·dy − Σdy − x̂·Σ(dy·x̂))`, while
`dγ = Σ(dy·x̂)` and `dβ = Σdy`. That is why the training forward pass
saves the normalized values `x̂` and `invStd`.

The linear layers are tic-tac-toe's backprop in matrix form (`gw += dyᵀ ·
x`, `dx = dy · w`), and ReLU passes a gradient only where its output was
positive. `Network::backward` strings them together: both heads first,
their trunk gradients added (the trunk feeds both), then the residual
blocks in reverse:

```cpp
void Network::backward(Workspace& ws, int B) {
    /* ... value head and policy head backward; dTrunk = dL/d(trunk output) ... */
    for (size_t i = blocks_.size(); i-- > 0;) {
        ResidualBlock& blk = blocks_[i];
        Workspace::BlockBuffers& buf = ws.blocks_[i];
        const float* in = i == 0 ? ws.stemOut_.data() : ws.blocks_[i - 1].out.data();
        reluBackward(buf.out.data(), dTrunk, trunk);           // now dL/d(bn2 output + skip)
        /* ... bn2, conv2, ReLU and bn1 backward into dTmp2 ... */
        be.im2col(in, F, B, n, 3, col);
        blk.conv1.backward(be, in, col, dTmp2, B, n, dcol, dTmp);
        // The skip path passes dTrunk to the block input unchanged; the conv
        // path adds its own contribution.
        for (size_t j = 0; j < trunk; ++j) dTrunk[j] += dTmp[j];
    }
    /* ... stem backward; no gradient needed for the input planes ... */
}
```
*src/network.cpp:277 — `Network::backward` (abridged)*

One memory detail: all 3×3 convolutions share a single `im2col` buffer,
so by the time the backward pass reaches a convolution its columns have
been overwritten. The code rebuilds them from the saved input just
before use (`be.im2col(...)` above) — one extra cheap copy per layer
instead of keeping 2·blocks + 1 column matrices alive.

**How do you know it is right? Finite differences.** A backprop bug
compiles, runs, and often still lowers the loss, just towards a worse
network. The defense, as in tic-tac-toe, is to check every analytic
gradient against the definition of a derivative: nudge one number by
±h, measure the loss both ways, and compare the slope with what backward
reported. `tests/test_layers.cpp` does this for every entry of every
layer's input and parameters, using the loss `L = Σ rᵢ yᵢ` with random
weights `r`, whose gradient with respect to the layer output is exactly
`r`:

```cpp
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
```
*tests/test_layers.cpp:36 — `checkGradient`*

The step actually taken is measured after float rounding (`up − down`,
not `2h`). `tests/test_network.cpp` then checks the whole network —
two residual blocks on 5×5 — through the real loss. There a nudge often
crosses one of the ReLU kinks behind a batch norm, where the slope
genuinely changes; the test detects that case (the one-sided slopes
disagree with each other) and excuses only a minority of probes, so a
real mismatch in a smooth region still fails.

**The loss.** Training minimizes, averaged over a batch,

```
L = (z − v)²  −  Σₐ π(a) · log softmax(logits)ₐ
```

the squared error between the value `v` and the game result `z`
(chapter 05), plus the cross-entropy between the search's visit
distribution `π` (chapter 02) and the policy. L2 weight decay, the third
term of AlphaGo Zero's loss, is applied by the optimizer instead
(chapter 06). `trainStep` is forward in training mode, the loss with its
gradients, then backward:

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

`Network::loss` computes the log-softmax with the usual guard
(subtract the largest logit before `exp`, so nothing overflows) and the
two closed-form output gradients familiar from tic-tac-toe: `softmax −
π` for the policy logits, and `2(v − z)(1 − v²)` through the `tanh`,
both divided by the batch size.

> **Parameters versus activations.** `Network` holds only weights and
> batch-norm running statistics; every activation buffer lives in a
> separate `Workspace`. Ten self-play threads can then share one
> read-only network, each with its own workspace, with no locks and no
> copies of the weights. The same split is what a GPU backend would
> need: weights uploaded once, scratch per stream.

---

## 02 — Search corrects the guess

**Concept: planning with PUCT tree search, and search as policy improvement**

A network's first guesses are close to random, and even a trained one
has blind spots. **Monte Carlo tree search** checks the guess against
the rules: from the current position it runs many *simulations*, each
walking down the tree of moves to a position it has not looked at yet,
asking the network about that one position, and backing the answer up
the path it came down. After a few hundred simulations the visit counts
at the root are a better policy than the network's raw prior, and that
improved policy is both the move played and the training target.
Tic-tac-toe's chapter 02 walks through this on a 9-square board; the
algorithm here is the same, rebuilt around a flat node arena, batched
evaluation (chapter 03) and Go's extra rules.

The tree lives in two flat vectors. A node (`MctsNode`) holds a `Board`,
its parent's index, the ply and consecutive-pass count, and the range of
its edges; an edge (`MctsEdge`) holds the action, the prior P, the visit
count N, the value sum W and a virtual-loss counter. Everything refers
to everything else by index, so there is no pointer chasing and no
per-node allocation.

**Selection: PUCT.** At each node the descent takes the edge with the
largest

```
Q(s,a) + c_puct · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a))
```

where `Q = W / N` is the average result this move has produced so far
(exploitation) and the second term is an exploration bonus that starts
proportional to the network's prior and shrinks as the move gets
visited. `c_puct` is 1.5 in every preset.

```cpp
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
        // Each virtual loss adds a -1 result for the chooser. An unvisited
        // edge has Q = e.unvisitedQ: 0, except a known game-ending pass.
        const float q = n > 0 ? (e.valueSum - static_cast<float>(e.virtualLoss)) / n : e.unvisitedQ;
        const float u = cfg_.cPuct * e.prior * sqrtN / (1 + n);
        if (q + u > bestScore) {   // strict '>': the first index wins ties
            bestScore = q + u;
            best = nd.firstEdge + i;
        }
    }
    return best;
}
```
*src/mcts.cpp:54 — `Mcts::selectEdge`*

Ignore `virtualLoss` for now (it is 0 when batching is off; chapter 03
explains it). An unvisited edge gets Q = 0, a neutral guess; KataGo-era
engines tune this "first-play urgency", which this project deliberately
leaves out.

There is exactly one exception, and it is not first-play-urgency tuning.
When the previous move was a pass, this node's own pass edge ends the game,
so its value is known exactly: the game result, +1, −1 or 0 from the
chooser's side, given by the sign of the Tromp-Taylor score minus komi.
`expand()` stores that value in `unvisitedQ`, and it is used
while the edge is unvisited. Without it, a game-ending pass with a tiny prior
is never examined at low simulation counts, and a net that likes passing can
walk into a lost game. Chapters 07 and 08 tell how the arena exposed this.

**Legal masking and a softmax over legal moves only.** The policy head
produces a logit for every point, including occupied ones and points
where a stone would be suicide or break superko. When a leaf is
*expanded*, it gets edges for legal actions only, and the priors are a
softmax over just those logits, so the illegal moves' probability mass
is not wasted and the remaining priors still sum to 1:

```cpp
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
    /* ... p[i] = exp(logits[legal[i]] - maxLogit), sum = sum of p ... */
    /* ... one edge per legal action with rawPrior = prior = p[i] / sum ... */
    nd.expanded = true;
    nd.netValue = value;
}
```
*src/mcts.cpp:133 — `Mcts::expand` (abridged)*

**Superko inside the tree.** A node deep in the tree is a position that
has not happened yet. Whether a move from it would repeat a position
depends on two histories: the real game's (`game_->seen`, the hash set
from chapter 00) and the hypothetical moves on the path from the root
down to this node (`pathHashes`). Both are checked. Because each node
has exactly one parent, the path — and so the set of forbidden moves —
is well defined for every node. The analyze output in the "Try it" block
at the end of this chapter shows the result on the ko position from
chapter 00: B3, White's forbidden retake, gets no edge, so no prior and
no visits.

**Backup and the sign flip.** The value the network returns is from the
point of view of the player to move *at the leaf*. Each edge on the path
belongs to the player who chose it, who is the opponent of the player
to move at the node it leads to. So the value is negated once before it
is added to the last edge, and once more at every edge above:

```cpp
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
```
*src/mcts.cpp:184 — `Mcts::backup`*

This includes **pass**. A pass changes no stones but does hand the move
to the opponent (`Board::play` flips `toMove_` for a pass too), so a
pass edge flips the sign exactly like any other move. What a pass *does*
change is the pass count, and that decides whether the next node is the
end of the game. `Mcts::descend` creates children lazily, the first
time a descent takes an edge, and a child made by a pass inherits the
parent's count plus one (any other move resets it to 0). The root's
count comes from the real game, and a kept tree is thrown away if its
root's count no longer matches the game's.

**Terminal nodes are scored by the rules.** When a descent reaches a
node after two consecutive passes, or at the move cap, there is nothing
to guess: the game is over and Tromp-Taylor says who won.

```cpp
bool Mcts::isTerminal(const MctsNode& node) const {
    return node.passes >= 2 || node.ply >= game_->maxMoves();
}

float Mcts::scoreValue(const Board& board) const {
    // Tromp-Taylor score minus komi as a result for board's side to move.
    float margin = static_cast<float>(board.areaScore()) - game_->komi();
    float blackResult = margin > 0 ? 1.0f : (margin < 0 ? -1.0f : 0.0f);
    return board.toMove() == Color::Black ? blackResult : -blackResult;
}

float Mcts::terminalValue(const MctsNode& node) const {
    // The game is over, so no guess is needed: score it with the rules.
    return scoreValue(node.board);
}
```
*src/mcts.cpp:49 — `Mcts::terminalValue`*

`tests/test_mcts.cpp` pins the pass case down: after Black passes on an
empty board, White's pass edge ends the game with White winning by komi,
so every visit to that edge must add exactly +1 for White — `valueSum ==
visits`. Exact terminal values matter more in Go than in tic-tac-toe:
every game ends with passes, and the search must know that passing while
behind loses.

**Try it** — search correcting the prior. The ko position from chapter
00 after Black's capture, White to move, analyzed with the final 5×5
network of seed 1:

```
$ ./build-blas/analyze --checkpoint runs/5x5-seed1/checkpoints/gen_0060.azn \
      --komi 24.5 --simulations 200 --moves "B4 C4 A3 C2 B2 D3 E5 B3 C3"
runs/5x5-seed1/checkpoints/gen_0060.azn: 9 moves played, White to move, komi 24.5

   A B C D E
 5 . . . . X  5
 4 . X O . .  4
 3 X . X O .  3
 2 . X O . .  2
 1 . . . . .  1
   A B C D E
White to move

Value (side to move): +0.77

Policy prior, % of legal moves ("." below 1%):
     A  B  C  D  E
 5   .  3  2  8  X
 4   .  X  O 30 17
 3   X  .  X  O  .
 2   .  X  O 11  8
 1   .  3  2 10  .
Pass prior: 1.2%

Top 10 moves by prior:
  D4     30.3%
  E4     17.3%
  D2     11.3%
  D1      9.9%
...

MCTS value (side to move): +0.64 after 200 simulations
Top 10 moves by visits:
  D1       104 visits ( 52.0%)  Q +0.79
  E4        39 visits ( 19.5%)  Q +0.67
  E2        32 visits ( 16.0%)  Q +0.60
  D4         8 visits (  4.0%)  Q -0.15
  D2         8 visits (  4.0%)  Q +0.15
  D5         6 visits (  3.0%)  Q +0.24
...
```

The prior liked D4 best (30.3%); after 200 simulations search has found
that D4 is bad (8 visits, Q −0.15) and puts more than half its visits on
D1, only the prior's fourth choice at 9.9%, whose Q is highest. B3 prints
as "." and is never visited — the superko mask at work (`analyze`
applies the same legal-move mask to the raw prior). With komi 24.5 White
wins unless Black ends up owning all 25 points, which is why the value
favors White here; chapter 05 explains that komi. The visit distribution, not
the prior, is what the game plays and what the network is trained on:
**search is a policy-improvement operator**, and training distills the
improvement back into the network.

On 9×9, the final network's top-visited move in the atari test position
of `docs/results/9x9/analyze-atari.txt` (a White stone with one liberty
left, Black to move) is the capture, E4, with 63 of 200 visits (31.5%),
so the check as defined passes. But the visits come from the prior, not
from search: E4 is the prior's first choice at 51.4%, while its search Q
of −0.02 is the lowest of the five most visited moves. Search cannot fix what it
never looks at, though: in the eye test position
(`docs/results/9x9/analyze-eye.txt`) the one move that gives a Black group
two eyes has a prior below 1% and is not among the ten most visited
moves, and the network's values show it does not know the group dies
without it (`docs/results/README.md`).

---

## 03 — Batching the search

**Concept: virtual loss, and feeding the matrix unit big matrices**

Serial MCTS evaluates one leaf per network call. On a CPU with a matrix
unit, or on a GPU, that wastes most of the hardware: a call on one
position is a skinny matrix product, and fixed per-call costs dominate.
The fix is to collect several leaves and evaluate them in one call. The
difficulty is that the descents are deterministic: with no new
information in between, eight descents from the root would all follow
the same path to the same leaf.

**Virtual loss.** While a descent is waiting for its leaf to be
evaluated, every edge on its path counts one extra visit *and* one extra
loss for the player choosing it (the `virtualLoss` terms in
`selectEdge`, chapter 02). That lowers Q along the path and lowers the
exploration bonus too, so the next descent is pushed towards other
moves. When the batch comes back, the virtual losses are removed and the
real values backed up. With a batch size of 1 there is never a descent
in flight, and the algorithm is exactly serial MCTS; `tests/test_mcts.cpp`
checks that batch size 1 calls the evaluator once per leaf and that batched
searches leave no virtual loss behind.

```cpp
void Mcts::search(const Game& game, Rng& rng) {
    assert(!game.isTerminal());
    prepareRoot(game);
    // The root needs priors before the first descent; evaluating it is not a simulation.
    if (!nodes_[0].expanded) evaluateLeaves({0});
    setRootPriors(rng);   // fresh noise (or none) for this search
    /* ... leaves, paths, path; done = 0 ... */
    while (done < cfg_.simulations) {
        /* ... clear this round's leaves and paths ... */
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
        /* ... for each leaf: pending = false, remove its virtual loss,
               back up its network value, ++done ... */
    }
    game_ = nullptr;
}
```
*src/mcts.cpp:196 — `Mcts::search` (abridged)*

**Collisions end the round.** Virtual loss discourages repeats but does
not forbid them: when one move is much better than the rest, a second
descent can still land on a leaf that is already waiting. Evaluating it
twice would waste a slot and double-count it, so the round stops early
and evaluates the leaves it has. A terminal leaf never waits at all; its
exact value is backed up on the spot. The cost of batching is a slightly
different search: descents in a round cannot see each other's results.
A batch size of 8 (`mcts_batch` in every preset) is a common compromise — small
enough that the tree does not "go wide" much, large enough to fill the
matrix unit better.

**Why batch size matters to gemm.** Take the 5×5 preset's residual
convolution: its weight matrix is 32 × 288 (32 filters, 32 channels × 9
offsets). At batch 1 it multiplies a column matrix 25 wide; at batch 8,
200 wide. Both products cost the same per column, but the batch-8 call
pays the fixed costs — the call itself, packing the operands, starting
the matrix unit, and all the per-layer bookkeeping around it — once for
eight positions instead of once for each. Measured on the M4, one
thread, BLAS backend (`docs/results/bench.txt`, last table): the 5×5
network evaluates about 23,000 positions per second at batch 1 and 30,400
at batch 8; the 9×9 network about 2,750 and 3,700. Batching buys about
1.3× on this CPU — worth having, but modest: on one CPU thread the
fixed costs are a small part of each call. Hardware with much higher
fixed costs per call, such as a GPU, gains more from the same batching;
this project measured only the CPU.
`bench` measures inference at the MCTS batch of 8
(`docs/results/bench.txt`): 30,488 positions per second on 5×5 is
3,811 network calls per second, each covering a whole batch.

**The `Evaluator` seam.** MCTS never touches `Network`. It encodes its
leaves into input planes (chapter 00) and hands them to an `Evaluator`:

```cpp
class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual int boardSize() const = 0;
    virtual int history() const = 0;
    virtual int maxBatch() const = 0;
    // input [B][numPlanes(history())][n*n]; logits [B][n*n+1]; values [B].
    virtual void evaluate(const float* input, int B, float* logits, float* values) = 0;
};
```
*include/azgo/evaluator.hpp:17 — `Evaluator`*

The shipped implementation, `LocalEvaluator`, is one per worker thread:
it wraps the shared read-only network and a private `Workspace` and just
calls `Network::infer`. `UniformEvaluator` returns equal logits and a
constant value, which lets the search tests run with no network at all
("with a uniform prior, does search still find the one winning
capture?").

This is where a GPU would plug in, without touching MCTS. The design
spec's extension path has two steps. First, a `MetalBackend`
implementing the `Backend` interface (`gemm`, `im2col`, `col2im`) with
Metal compute kernels; on Apple Silicon the CPU and GPU share memory, so
buffers need not be copied. Second, a *batching evaluator*: an
`Evaluator` whose `evaluate` puts a worker's leaves in a shared queue and
blocks until a dispatcher thread has run one large batch — 64 to 256
positions gathered from all ten workers — through the GPU. MCTS would
still see the same `evaluate` call. The spec estimates the M4's GPU at
only about 2× the Accelerate/matrix-unit CPU path, and only at large
batch sizes; that estimate, not a measurement, is why v1 stops at the
CPU.

---

## 04 — Self-play and exploration

**Concept: generating diverse experience without wrecking its quality**

Self-play is the data generator: the current network, guided by search,
plays both sides, and every position becomes a training example. Two
mechanisms keep those games from collapsing into one repeated line.

**Dirichlet noise at the root.** Tic-tac-toe's chapter 03 showed why:
the PUCT exploration bonus is proportional to the prior, so a move the
network has learned to dismiss is never searched, never played, and
never corrected. Before each self-play search, the root's priors are
mixed with a random Dirichlet sample, 25% noise to 75% network:

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
```
*src/mcts.cpp:319 — `Mcts::setRootPriors`*

**Why α scales as 10/n².** A symmetric Dirichlet sample over L moves with
concentration α is a random probability vector; the smaller α, the more
of its mass lands on a few moves. What matters for exploration is
roughly α·L, the total concentration: a sample with α·L ≈ 10 looks like
"a handful of moves get a real boost". The number of legal moves grows
with the board, about n² early in a game, so keeping α·L constant means
α = 10/n²: **0.4 on 5×5, 0.12 on 9×9, 0.028 on 19×19**. AlphaZero
reported the same rule of thumb — α in inverse proportion to the typical
number of legal moves, 0.3 for chess, 0.15 for shogi and 0.03 for Go.
Using 19×19's 0.03 on 5×5 would put nearly all of each sample on one
random move; using 5×5's 0.4 on 19×19 would spread it almost evenly over
362 moves, adding the same sliver to every prior instead of singling out
a few moves to explore.
Tic-tac-toe used a fixed 0.3 over at most 9 moves.

Noise is added only at the root and only in self-play. The arena and
`analyze` switch it off (`addNoise = false`): measuring strength and
inspecting a position want the network's honest judgment. With tree
reuse (below) the root was an interior node of the previous search, so
it starts with its raw priors, and `setRootPriors` mixes fresh noise into
them on every search.

**Temperature.** The second mechanism acts on which move is *played*,
not which moves are searched. For the first `temperature_moves` plies of
a self-play game the move is sampled in proportion to its visit count
(temperature 1); after that, the most visited move is played
(temperature 0, ties broken at random):

```cpp
GameRecord playSelfPlayGame(Evaluator& evaluator, const SelfPlayConfig& cfg, Rng& rng) {
    /* ... n, area, game, mcts with addNoise forced on, rec ... */
    while (!game.isTerminal()) {
        mcts.search(game, rng);

        // Position before the move ...
        const size_t offset = rec.stones.size();
        rec.stones.resize(offset + area);
        game.board().exportStones(rec.stones.data() + offset);
        /* ... and the search's visit counts, the policy target pi, as sparse
               (action, count) pairs in rec.visits ... */

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
```
*src/selfplay.cpp:14 — `playSelfPlayGame` (abridged)*

The presets sample 8 plies on 5×5, 20 on 9×9 and 30 on 19×19 (AlphaGo
Zero's value), roughly the first fifth of a typical game on each board
(5×5 games average 30 to 44 plies, `docs/results/5x5/log.csv`; 9×9
games 87 to 116, `docs/results/9x9/log.csv`; the two 19×19 generations
145 and 365, `docs/results/19x19/log.csv`).
Tic-tac-toe taught the lesson this number encodes: its default used to
be 2, and training plateaued because self-play kept replaying the same
few games — the network can only learn positions self-play reaches. In
a 9-move game the answer was to sample the whole game. In Go that would
be the other mistake: a sampled blunder in the endgame flips the result,
and with it the value label of every position in the game. Sample the
opening, where variety is cheap, then play well.

**Tree reuse and compaction.** After a move, the subtree under the move
just played is exactly the search a player would like to continue, so
it is kept: the chosen child becomes the new root, with all its visits.
The rest of the tree can never be reached again. Rather than leaving it
in the vectors, `advance` copies only the reachable subtree into fresh
vectors and swaps them in:

```cpp
void Mcts::advance(int action) {
    if (nodes_.empty()) return;
    /* ... newRoot = child of the root edge whose action matches ... */
    if (newRoot < 0) {   // never explored (or illegal): nothing worth keeping
        reset();
        return;
    }
    // Copy only the subtree under newRoot into fresh vectors, so memory stays
    // bounded by what one move's search built. Explicit-stack DFS; each stack
    // entry is (old node, its new parent, the new edge that must point to it).
    /* ... DFS from newRoot: copy each node and its edges into fresh vectors
           `nodes` and `edges`, renumbering parent, firstEdge and child ... */
    nodes_.swap(nodes);
    edges_.swap(edges);
}
```
*src/mcts.cpp:336 — `Mcts::advance` (abridged)*

Without compaction the node vector would only ever grow: a 19×19
self-play game at 400 simulations per move and several hundred moves
would keep every node of every search in memory. With it, the arena
holds only positions still reachable from the current one, and
everything built for moves that were not played is released at once.
Each node holds a whole `Board` (the 441-byte padded array), so this is
the difference between megabytes per game and a growing pile per
worker. A reused root already has visits, and `search` adds
`simulations` *new* ones on top, so reuse also buys a slightly deeper
search for free. `tests/test_mcts.cpp` checks that `advance` keeps the
chosen subtree's statistics intact.

---

## 05 — Assigning credit

**Concept: the Monte Carlo return, and what the rules make it mean**

A self-play game produces a hundred positions and one result. As in
tic-tac-toe, credit is assigned the most direct way: once the game is
over, every position is labeled with the final result `z`, seen from the
player to move *at that position*. The result comes from the score:

```cpp
float Game::resultForBlack() const {
    const float s = scoreForBlack();
    // An integer komi can produce an exact tie; it is scored as a draw.
    if (s > 0.0f) return 1.0f;
    if (s < 0.0f) return -1.0f;
    return 0.0f;
}
```
*src/game.cpp:61 — `Game::resultForBlack`*

and `GameRecord::zAt(ply)` turns it around for White's positions (Black
moves at even plies, so `z` is `resultForBlack` at even plies and its
negation at odd ones). Only the sign of the score is kept: winning by
half a point and by fifty are the same `z = +1`. That is AlphaGo Zero's
choice — the value head predicts who wins, not by how much — and it is
why komi matters so much: it decides where "who wins" flips.

**Passing, and how a game ends.** A pass is a legal action at every
turn. `Game::play` counts consecutive passes, and any stone resets the
count:

```cpp
void Game::play(int action) {
    assert(isLegal(action));
    Board next = board();
    next.play(action);
    passes_ = (action == next.passAction()) ? passes_ + 1 : 0;
    moves_.push_back(action);
    seen_.insert(next.hash());  // a pass re-inserts an existing hash: no-op
    boards_.push_back(next);
}
```
*src/game.cpp:39 — `Game::play`*

Two consecutive passes end the game (so does the move cap, 2n² moves:
50 on 5×5, 162 on 9×9, 722 on 19×19). An untrained network can pass on
move 1 and its opponent can pass right back: a two-ply game, scored on
an empty board, which White wins by komi. `tests/test_selfplay.cpp`
checks exactly that case: two positions recorded, `z = −1` at Black's
and `+1` at White's. It is a useful lesson for the network, and learned
fast: passing early loses.

**Tromp-Taylor means dead stones must actually be captured.** Human
rules end with the players agreeing which stones are dead and removing
them before counting. Tromp-Taylor has no such step — a stone on the
board at the end counts, however hopeless it is. So the winning side
must spend moves capturing stones that a human would simply have
removed. Here is a real self-play game, generation 60 of the 5×5 seed-1
run (`runs/5x5-seed1/sgf/gen_0060_game_00.sgf`), after 29 moves:

```
$ ./build-blas/analyze --checkpoint runs/5x5-seed1/checkpoints/gen_0060.azn --komi 24.5 \
      --simulations 200 \
      --moves "C3 D3 C2 C4 D4 B4 D2 E4 D5 C5 E3 B3 B2 A3 A5 A2 B5 A4 B5 A5 E5 B1 C1 E1 A1 E2 B5 C4 C5"
runs/5x5-seed1/checkpoints/gen_0060.azn: 29 moves played, White to move, komi 24.5

   A B C D E
 5 . X X X X  5
 4 . . O X .  4
 3 . . X . X  3
 2 . X X X O  2
 1 X . X . O  1
   A B C D E
White to move

Value (side to move): -1.00
...
Pass prior: 27.1%
...
MCTS value (side to move): -1.00 after 200 simulations
Top 10 moves by visits:
  pass      55 visits ( 27.5%)  Q -1.00
  A3        41 visits ( 20.5%)  Q -1.00
  A4        31 visits ( 15.5%)  Q -1.00
  B3        23 visits ( 11.5%)  Q -1.00
  A5        18 visits (  9.0%)  Q -1.00
  B4        18 visits (  9.0%)  Q -1.00
  A2        14 visits (  7.0%)  Q -1.00
```

Every White stone on the board is dead — C4 and the pair E2–E1 each have
one liberty left — and the network knows it: the value is −1.00 for
White, and every move searches to −1.00. But if both players passed now,
those three White stones would count for White, the empty points next to
them would count for nobody, and Black — who needs all 25 points at komi
24.5 — would lose. So the game goes on. White passes; Black captures E2
and E1 with D1, C4 with B4, and White's two throw-ins at B3 and A2 with
A3; after ply 35 there is no White stone left. From there Black would
win by passing, but every move searches to +1.00 and nothing makes Black
prefer ending the game (`analyze` there spreads the visits, pass
getting the most at 22%; self-play, with root noise, played B1): it
keeps filling its own
points while White
passes or throws in a stone, and the game stops at the 50-move cap,
`B+0.5`. This is why Tromp-Taylor games run long and contain many
passes: the average 5×5 game was 43.1 plies in generation 1 and 43.3 in
generation 60 (as short as 29.6 in between), and between 13.6% and 31.9%
of all plies in a generation were passes (`docs/results/5x5/log.csv`).
The 9×9 run went from 88.6 plies in generation 1 to 93.0 in generation
150, with 10.3% of the plies passes (`docs/results/9x9/log.csv`).
The upside is that the rules need no life-and-death oracle, only a flood
fill, which is exactly what a program learning from nothing needs.
(Engines like KataGo add tricks to shorten this phase; this project
leaves them out.)

**Why 5×5 uses komi 24.5.** 5×5 Go is solved: with perfect play Black
opens on the center point and ends up owning the whole board, a win by
25 points. At the standard komi of 7.5, the network discovers a large
part of that early, and then almost every game is a Black win. The
pilot at komi 7.5 (`docs/results/pilots/5x5-komi7.5.csv`) shows it:
self-play Black win rate above 0.85 in every generation from 26 on
(0.895–0.96 over the last ten), and from then on the arena cannot tell
generations apart, because a color-balanced match between two players
who both win every game as Black is 50% whatever their strength. The
Elo curve peaked at 1066.3 in generation 27 and sagged to 733.1 by
generation 60; the mean rating of the last ten generations was only
+124 above the first ten's (all three figures are `log.csv` ratings,
chapter 07).

At komi 24.5, Black wins only by owning all 25 points, so games stay
undecided far longer and the arena keeps finding differences. In seed 1
(`docs/results/5x5/log.csv`) the self-play Black win rate wanders between
0.16 and 0.845 over generations 3–33 while Elo climbs, then stays above
0.85 once the network plays the perfect line, from generation 34 on
(seed 2: from generation 31 on). The mean `log.csv` rating of the last
ten generations is +470 above the first ten's (+530 in the komi-24.5
pilot, `docs/results/pilots/5x5-komi24.5.csv`). Komi 20.5 behaved
similarly in its pilot (+489, `docs/results/pilots/5x5-komi20.5.csv`)
but saturated sooner.
And the final network agrees with the known solution — its prior on
the empty board:

```
$ ./build-blas/analyze --checkpoint runs/5x5-seed1/checkpoints/gen_0060.azn --komi 24.5 \
      --simulations 200
...
Value (side to move): +0.85

Policy prior, % of legal moves ("." below 1%):
     A  B  C  D  E
 5   .  .  .  .  .
 4   .  .  .  .  .
 3   .  . 97  .  .
 2   .  .  .  .  .
 1   .  .  .  .  .
...
MCTS value (side to move): +0.90 after 200 simulations
Top 10 moves by visits:
  C3       196 visits ( 98.0%)  Q +0.92
  C4         2 visits (  1.0%)  Q +0.22
  D3         1 visits (  0.5%)  Q +0.10
  C2         1 visits (  0.5%)  Q +0.17
```

The same command on generation 0 prints a value of +0.08 and spreads its
prior over the board — A2 highest at 17.3%, the center below 1% — and
search's top ten moves each get 3.5–13% of the 200 simulations, all with Q
between −0.03 and +0.05: the untrained network knows nothing yet.

**The integer-komi draw.** With a half-point komi a tie is impossible.
With an integer komi, `scoreForBlack() == 0` is possible; it is scored
as a draw everywhere: `resultForBlack() == 0`, `z = 0` at every position
of the game, the SGF records `RE[0]`, and the arena scores it as half a
point for each side (`matchScore` returns 0.5). On 5×5 at komi 25, a
perfect game — Black owning all 25 points — is exactly a draw:

```
$ ./build-blas/arena --a runs/5x5-seed1/checkpoints/gen_0060.azn \
      --b runs/5x5-seed1/checkpoints/gen_0033.azn --games 4 --workers 1 \
      --komi 25 --simulations 100 --temperature-moves 2 --backend blas
A runs/5x5-seed1/checkpoints/gen_0060.azn vs B runs/5x5-seed1/checkpoints/gen_0033.azn: A scored 2.5/4 = 0.625 (95% CI 0.219-0.908), distinct games 4
A wins 1, B wins 0, draws 3
```

---

## 06 — Learning from itself

**Concept: supervised learning on self-generated labels, with a replay buffer**

Once positions carry `(π, z)` labels, training is ordinary supervised
learning on the loss from chapter 01. The Go-specific questions are
where the examples live, how they are drawn, and how hard to train on
them.

**A replay buffer of whole games.** Tic-tac-toe's buffer held encoded
examples. Here an encoded example is 18 planes of n² floats — 26 KB on
19×19 — while the position itself is n² bytes, and the 16 history
planes are just the previous 8 positions of the same game. So the buffer
stores **whole games** (`GameRecord`: the moves, one byte per point per
position, the sparse visit counts and the result) in a sliding window of
at most `buffer_positions` positions, evicting the oldest games first.
A sample is a (game, ply) pair, and its planes are rebuilt when it is
drawn:

```cpp
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
    /* ... zero pi; total = sum of counts at this ply ... */
    for (const auto& av : g.visits[ply])
        pi[transformAction(sym, av.first, n)] = static_cast<float>(av.second) / static_cast<float>(total);

    *z = g.zAt(ply);
}
```
*src/replay_buffer.cpp:53 — `ReplayBuffer::encodeSample` (abridged)*

`sample` draws each of a batch's positions uniformly over all positions
in the window (so long games contribute more samples, in proportion to
their length) and gives each its own random symmetry. The windows are
sized in generations of play: 100,000 positions on 5×5 is about twelve
generations of 200 games of ~40 plies; 200,000 on 9×9 about 25
generations of 100 games of ~80 plies.

**Symmetry augmentation.** Go's rules, and komi, are unchanged if the
board is rotated or mirrored, so each position has up to 8 equivalent
versions with the same value and correspondingly transformed best moves.
Showing the network a random one of the 8 multiplies the data for free:

```cpp
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
```
*src/symmetry.cpp:6 — `transformPoint`*

The same permutation is applied to every plane and to π (pass maps to
itself), so input and target stay consistent. AlphaGo Zero used this,
and also evaluated a random symmetry of each leaf during search;
AlphaZero dropped both, because chess and shogi are not symmetric (pawns
move one way, castling is one-sided) and it wanted one algorithm for all
three games. This project augments in training only; search always
evaluates the untransformed board.

**Replay ratio.** Each generation takes `train_steps` SGD steps of
`batch_size` samples. The *replay ratio* — samples drawn per new
position generated — is, once the window is full and game lengths are
steady, the expected number of times a position is drawn over its whole
stay in the window. Too low and self-play time is wasted;
too high and the network overfits the latest games. `log.csv` records
it: 2.9 to 4.3 on 5×5 (200 × 128 samples over 200 games of 30–44 plies,
`docs/results/5x5/log.csv`), 4.4 to 5.9 on 9×9 (`docs/results/9x9/log.csv`)
and 8.49 in the 19×19 pilot (`docs/results/pilots/`). The 19×19 preset's `train_steps`
went from 100 to 80 because of that last number (see the note in
`configs/19x19.cfg`).

**SGD with momentum and weight decay.** The optimizer is AlphaGo Zero's:
momentum 0.9, L2 weight decay 1e-4.

```cpp
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
```
*src/optimizer.cpp:19 — `SgdMomentum::step`*

Adding `λ·w` to the gradient is the same as adding `(λ/2)·‖w‖²` to the
loss. Tic-tac-toe put the penalty in the loss and decayed every
parameter; here it lives in the optimizer and applies only to
convolution and linear weights (`ParamView::decay`), the usual choice for
networks with batch norm. The velocity buffers are part of the saved
training state (chapter 08), so a resumed run continues with the same
momentum.

The learning rate follows a stepped schedule keyed on generation,
`lr_schedule = 0:0.02,40:0.002` on 5×5 and `0:0.02,90:0.002` on 9×9: fast
progress first, finer steps later. On 5×5 seed 1 the value loss kept
falling after the drop at generation 40, from 0.147 to 0.092 at
generation 60, while the policy loss stayed flat at 1.73–1.75
(`docs/results/5x5/log.csv`).

---

## 07 — Measuring progress without an oracle

**Concept: relative evaluation — arenas, Bradley-Terry Elo and confidence intervals**

Tic-tac-toe could be scored against a perfect minimax player. Go cannot
(5×5 is solved, but no solver ships with this project, and 9×9 and 19×19
are not). The only yardstick left is the network's own past: each new
generation plays earlier ones, and ratings are fitted to the results.
This chapter is about how much that can and cannot tell you.

**The arena.** Every generation (`arena_every = 1`), the new network
plays `arena_games` games (20 in the 5×5 and 9×9 presets, colors
alternating) against a few earlier generations and the bots, with search but
without Dirichlet noise:

```cpp
GameOutcome playArenaGame(WorkerContext& ctx, const ArenaConfig& cfg, int n, uint64_t seed,
                          int index) {
    Rng rng(deriveSeed(seed, static_cast<uint64_t>(index)));
    bool aIsBlack = index % 2 == 0;   // colors alternate, so A is Black in half the games
    /* ... game; searchA, searchB from each Seat's MctsConfig, addNoise = false ... */
    while (!game.isTerminal()) {
        bool blackToMove = game.board().toMove() == Color::Black;
        Mcts& mover = blackToMove == aIsBlack ? searchA : searchB;
        mover.search(game, rng);
        // A few sampled opening moves make the games differ from each other.
        float temperature = game.moveCount() < cfg.temperatureMoves ? 1.0f : 0.0f;
        int action = mover.selectAction(temperature, rng);
        game.play(action);
        /* ... both trees advance to the move played ... */
    }
    return {matchScore(game.resultForBlack(), aIsBlack), game.moves()};
}
```
*src/arena.cpp:50 — `playArenaGame` (abridged)*

**Opening diversity.** Without noise, search is deterministic up to
random tie-breaks, and two fixed networks would replay nearly the same
game twenty times — twenty games of evidence that are really one.
Tic-tac-toe's arena once did exactly that: 2 distinct games out of 40.
So the first `arena_temperature_moves` plies (2 on 5×5, 4 on the larger
boards) are sampled from the visit counts, and the arena counts the
distinct move sequences it produced; `log.csv`'s last column,
`distinct_vs_prev`, records that count for the match against g − 1. On
5×5 seed 1 it ranged from 9 to 20 out of 20 (`docs/results/5x5/log.csv`)
— fewer once the network found the perfect line, which is evidence in
its own right: the policy was collapsing onto one opening. On 9×9 it
ranged from 19 to 20 out of 20 (`docs/results/9x9/log.csv`): with 81
points and 4 sampled plies, the openings almost never repeat.

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
generation 0. On 9×9, generation 0 scored at most 2 of 20 in every one
of its 137 matches from generation 14 on, and 0 of 20 in 110 of all 150
matches (`docs/results/9x9/results.csv`). The
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
  E = 1 / (1 + 10^((R_c − R̂)/400)), is closest to 0.7. R_c is the candidate's rating and R̂ the new
  network's estimate; the new network has no rating yet, so g − 1's
  rating stands in.

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

**Bradley-Terry instead of incremental Elo.** Chess-style Elo updates
each player's rating after every game by an amount proportional to the
surprise. The result depends on the order the games arrive in, and it
never revisits old ratings in the light of new evidence. This project
instead keeps every pairwise result in a table (`results.csv`) and, after
each arena round, refits all ratings at once by maximum likelihood under
the **Bradley-Terry** model: P(i beats j) = γᵢ / (γᵢ + γⱼ), reported on
the Elo scale as 400·log₁₀ γ, where a 200-point gap means about 76%.
The fit is Hunter's MM algorithm:

```cpp
std::map<int, double> fitBradleyTerry(const ResultsTable& results, int anchor) {
    /* ... 1. keep only players connected to the anchor through played pairs ... */
    // 2. Pair counts plus one virtual drawn game per played pair: n_ij + 1
    //    games, s_ij + 0.5 score. Without it a player who never lost would have
    //    an infinite maximum-likelihood rating.
    /* ... wins[i] (with the virtual halves) and one edge {i, j, games} per pair ... */
    // 3. MM iteration (Hunter 2004): gamma_i <- W_i / sum_j n_ij / (gamma_i + gamma_j).
    //    Each sweep increases the likelihood; rescaling keeps gamma_anchor = 1.
    for (int sweep = 0; sweep < 100000; ++sweep) {
        std::fill(denom.begin(), denom.end(), 0.0);
        for (const Edge& e : edges) {
            double d = e.games / (gamma[e.i] + gamma[e.j]);
            denom[e.i] += d;
            denom[e.j] += d;
        }
        for (size_t i = 0; i < m; ++i) next[i] = wins[i] / denom[i];
        /* ... divide by next[anchor]; stop when no gamma changes by 1e-10 in log ... */
    }
    /* ... elo[id] = 400 * log10(gamma) ... */
}
```
*src/elo.cpp:113 — `fitBradleyTerry` (abridged)*

Three properties matter. **Order independence**: the fit depends only
on the table, so a resumed run reproduces the same ratings. **The
anchor**: generation 0 is pinned at 0; every rating is "Elo above the
untrained network". **The virtual draw**: each played pair gets one
extra half-won game, so a 20–0 sweep gives a finite gap of
400·log₁₀(20.5/0.5) = 645 Elo rather than infinity. The 5×5 komi-7.5
pilot shows that cap directly: generation 1 swept generation 0, its only
opponent, and was rated 645.1 (`docs/results/pilots/5x5-komi7.5.csv`).

**Why old ratings move: `log.csv` versus `elo.csv`.** The two files
answer different questions. `log.csv`'s `elo` column is each
generation's rating *at the time*, from the table as it stood right
after its own arena round. `elo.csv` is rewritten after every round
with the refit of the whole table, so older generations' ratings move
as later generations play them. On 5×5 seed 1 (`docs/results/5x5/`),
generation 34 was rated 1422.2 on the day; after generations 35, 36, 38
and 42 had played it (scoring 10, 14, 2 and 13 out of 20), the refit
puts it at 1138.7. In that run, a new generation was rated from just five
matches, and a strong-looking 17–3 against an older ladder rung (generation 34 beat 26 by
that score) counts in full until later evidence tempers it.

**The 5×5 curve, stated exactly.** Seed 1's contemporaneous rating
(`log.csv`) rises from 445.6 at generation 1 to a peak of **1422.2 at
generation 34** and ends at **1117.5 at generation 60**, 305 below the
peak. The refit (`elo.csv`) tells a flatter story. Averaged over ten
generations at a time, it rises from 910.0 (generations 21–30) to 1092.2
(31–40), then stays level: 1086.8 (41–50) and 1089.1 (51–60); every
refit rating from generation 38 on lies between 1016 and 1127 (the
highest, 1126.7, at generation 46; the last, 1117.5). Seed 2 has the
same shape: `log.csv` peaks at 1225.2 in generation 33 and ends at
852.5; its refit peaks at 956.9 in generation 33, averages 927.1, 894.5
and 875.1 over the last three decades, and also ends at 852.5
(`docs/results/5x5-seed2/`). Both final networks beat generation 0 20–0
in the arena, and in a 100-game match (komi 24.5, 200 simulations,
`docs/results/5x5/final-arena.txt`) both score 100/100, 95% interval
0.963–1.000.

Why the plateau, and the apparent decline? Several things are going
on, and the data cannot fully separate them:

- **Revision, not regression, explains most of the drop in `log.csv`.**
  Peaks in the contemporaneous column are where a generation was
  briefly overrated; the refit, which uses all later evidence, has no
  such peak. Generation 60 against generation 34 directly, in a short
  check, scores 3 of 4 (95% interval 0.30–0.95) — no sign that 60 is
  weaker, and far too few games to say it is stronger:

  ```
  $ ./build-blas/arena --a runs/5x5-seed1/checkpoints/gen_0060.azn \
        --b runs/5x5-seed1/checkpoints/gen_0034.azn --games 4 --workers 1 \
        --komi 24.5 --simulations 100 --temperature-moves 2 --backend blas
  A runs/5x5-seed1/checkpoints/gen_0060.azn vs B runs/5x5-seed1/checkpoints/gen_0034.azn: A scored 3/4 = 0.750 (95% CI 0.301-0.954), distinct games 4
  A wins 3, B wins 1, draws 0
  ```
- **The game became decided by color.** From generation 34 on, Black
  takes the whole board in 87–96.5% of self-play games. Between two
  players who both win as Black, a color-balanced match is 50%
  whatever their strength, and seed 1's late matches against g − 1 are:
  a mean score of 0.49 over the last 20 generations. The arena has run
  out of resolution, not the network out of skill.
- **Loss of diversity.** The policy has collapsed onto the known perfect
  line (97% on the center point, chapter 05), distinct arena games fell
  from 20 to as few as 9, and self-play, driven by the same policy,
  likely visits fewer distinct positions.
  Positions off the main line then get less training data, which could
  let late generations drift on exactly the positions the arena's
  sampled openings produce.
- **The learning-rate drop at generation 40** (0.02 → 0.002) slows
  changes of every kind. It coincides with the plateau but did not
  obviously cause it: the refit had already reached its plateau level by
  generation 34, six generations before the drop.

**Drift and non-transitivity.** In the ladder-era runs of
`docs/results/`, every rating above the anchor's 645-point cap is held
up by a chain of ladder matches, each only 20 games.
Generation 0 stops being informative almost at once: from generation 1
on, seed 1's generations beat it 20–0 or 19–1. What remains is a chain
of noisy links, whose errors can accumulate like a random walk; the
ladder's long rungs (g − 4, g − 8) limit that drift but cannot remove it.
And game strength need not be transitive at all. Seed 1 has a cycle:
generation 30 beat 29 11–9, 31 beat 30 14–6, and 29 beat 31 12–8
(`docs/results/5x5/results.csv`). With 20 games each, every one of
those could be noise — which is precisely the point of the next
section. A deeper caveat applies to all self-play Elo: a network can
improve against its own ancestors in ways that do not transfer to other
opponents. The uniform bot is the only outside yardstick here, and a
weak one; the reference runs were made before it existed, so their Elo
claims are claims about their own lineage only. External engines stay
out of scope.

**How noisy are 20 games? Wilson intervals.** The log reports each
match against g − 1 with a 95% **Wilson score interval**, which behaves
sensibly for small samples and for scores of 0 or 1, unlike the
textbook p ± 1.96·√(p(1−p)/n):

```cpp
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
```
*src/elo.cpp:184 — `wilsonInterval`*

The numbers are sobering (`docs/results/5x5/elo.csv`): 10 of 20 gives
0.30–0.70; 15 of 20, a clear-looking 75%, gives 0.53–0.89; 3 of 20 gives
0.05–0.36. On the Elo scale, 0.30–0.70 is ±147 points: **one 20-game
match cannot resolve a difference smaller than about 150 Elo**. On its
own, only a lopsided result means much — 15–5 or better (seed 1's
generation 6 beat 5 16–4), 5–15 or worse (generation 24 lost to 23
3–17), or a sweep of generation 0. The fitted curve is meaningful
because it pools hundreds of such matches, not because any one of them
is. And with fewer games even a sweep proves little:

```
$ ./build-blas/arena --a runs/5x5-seed1/checkpoints/gen_0060.azn \
      --b runs/5x5-seed1/checkpoints/gen_0000.azn --games 4 --workers 1 \
      --komi 24.5 --simulations 100 --temperature-moves 2 --backend blas
A runs/5x5-seed1/checkpoints/gen_0060.azn vs B runs/5x5-seed1/checkpoints/gen_0000.azn: A scored 4/4 = 1.000 (95% CI 0.510-1.000), distinct games 4
A wins 4, B wins 0, draws 0
```

Four wins out of four only establish a win rate above 51%.

**Gating (AlphaGo Zero) versus none (AlphaZero).** AlphaGo Zero kept a
"best" network that generated all self-play; a newly trained network
replaced it only after winning at least 55% of a 400-game match.
AlphaZero dropped the gate and always self-played with the latest
network, a simpler loop. This project supports
both. With `gating = on`, self-play uses `bestGeneration`, the best
generation is always in the arena line-up, and it is replaced when
`passesGate(score, games, gating_threshold)` — `score / games >= 0.55` —
holds for the new network against it. The presets leave gating off, and
the numbers above show why it would be a poor fit at this scale: at 20
games, a score of 11/20 clears the 55% bar with a 95% interval of
0.34–0.74, so the gate would mostly be flipping coins. AlphaGo Zero's
400 games shrink that interval to roughly ±5 percentage points.

**When the yardstick misleads: the 9×9 runs.** An Elo curve anchored on
generation 0 measures one thing precisely: how well each generation
beats generation 0. On 9×9 that went wrong twice, in two different ways,
and both times 100-game head-to-heads between checkpoints told the real
story.

The first overnight run was made before the exact-pass rule of chapter
02 existed (its files are in `docs/results/passbug/`). Its curve said
the network learned almost nothing and then got worse: the refit peaked
at 288.4 at generation 35 and ended at 91.8
(`docs/results/passbug/9x9-main/elo.csv`), and the untrained generation
0 won 938 of the 3,000 arena games it played against trained
generations (`docs/results/passbug/9x9-main/results.csv`) — about 7 of
every 20 after generation 50. A trained network should not lose a third
of its games to random weights. Two cheap checks located the problem:

- **Head-to-head instead of the anchor.** 100 games of generation 150
  against generation 30 ended 92–8, and against generation 60 89–11: the
  lineage *was* improving. Whatever was wrong lived in the games against
  generation 0, which the anchor-based fit spreads over every rating.
- **Reading the games.** Generation 150 against generation 0 scored only
  67/100, and an instrumented replay that wrote each move's search value
  and Tromp-Taylor margin into the SGF showed that every loss had the
  same shape. The trained network, clearly ahead on the board, passed
  while generation 0's hopeless stones were still standing in its area —
  Tromp-Taylor counts them, so the margin was against it — and generation
  0 passed back and won on the count.
  `docs/results/passbug/9x9-gen150-vs-gen0-loss.sgf` is one such game:
  at ply 25 Black's search value is +0.93, the margin is −4.5, and Black
  passes.

A probe of that position found the blind spot. After its own pass, the
network gave the game-ending reply (the opponent's pass) a prior of
0.0074 among 59 legal moves; with unvisited edges at Q = 0 and tried in
prior order, 100 simulations never visited it, so the pass kept the
network's optimistic Q of +0.88 and was played. Generation 0, with
nearly flat priors, tried pass at its own root, found an exact win, and
took it. The fix is the `unvisitedQ` rule of chapter 02; with it, the
very same pre-fix checkpoints beat generation 0 100/100. The weights
were fine; the search had a hole that only a strange opponent exposed,
and the anchor turned that one opponent's quirk into the headline curve.
`docs/results/passbug/README.md` has the full numbers.

The second time is subtler, because nothing is broken. In the final
run, every generation from 14 on beats generation 0 by 18–2 or better
(`docs/results/9x9/results.csv`). The refit (`docs/results/9x9/elo.csv`)
rises from 0 to 784.8 at generation 54, then goes flat: between 601 and
735 from generation 80 on, **674.5 at generation 150**. A least-squares
line through generations 101–150, the last third, has a slope of
**+0.41 Elo per generation** — positive, but +20 Elo over 50
generations, far inside the noise. `log.csv`'s rating at the time even
falls, from 1050.6 at generation 85 to 674.5. Yet in 100-game matches
(200 simulations, `docs/results/9x9/h2h-*.txt`) generation 150 beats
generation 50 98–2, generation 90 86–14, generation 100 76–24 and
generation 120 67–33, and generation 90 beats generation 50 89–11 —
about 9 Elo per generation from 50 to 90 and 4–5 per generation after
the learning-rate drop at 90; every later checkpoint tested beats every
earlier one. Two
mechanisms flatten the curve:

- **The anchor's cap.** With the virtual draw, a 20–0 sweep of
  generation 0 says "645 Elo above generation 0". 137 such near-sweeps
  pull every late rating toward 645, and the late refit values sit right
  around it.
- **Links too short to see the slope.** At 4 Elo per generation, a
  generation should score 50.6% against g − 1 and 54.6% against g − 8;
  in generations 101–150 the measured means are 9.6 and 10.7 out of 20.
  Each of those 20-game links is ±150 Elo, so their combined pull is no
  match for the anchor's.

Refitting the same table without the matches against generation 0
(`docs/results/9x9/elo-without-gen0.txt`) gives a steadily rising curve,
generation 150 449 Elo above generation 50, with a last-third slope of
+2.8 Elo per generation. The lesson of both episodes is the same: a
self-play rating is a model of a results table, and when it says
something surprising, play the checkpoints against each other with
enough games, and read some of the games.

**Re-rated with informative opponents.** The rating-aware arena
described earlier in this chapter can be replayed over this run's saved
checkpoints:
`train --run runs/9x9-main --rerate --set arena_opponents=2` writes
`runs/9x9-main/rerate/` and leaves the run itself untouched. In the re-rated fit
(`docs/results/9x9/rerate/elo.csv`), generation 150 is rated **1766.6**
and stands 634 Elo above generation 50, against +676 from the 100-game
head-to-head (`docs/results/9x9/h2h-gen0150-vs-gen0050.txt`). Its
last-third slope is **+2.83 Elo per generation**
(`docs/results/9x9/rerate/elo.csv`), against +0.41 for the ladder
(`docs/results/9x9/elo.csv`). The 100-simulation uniform bot is rated
328.9 (`docs/results/9x9/rerate/references.csv`). A Bradley-Terry scale
depends on which pairs were played, so these numbers hold for
`arena_opponents = 2`. The curve now tells the head-to-heads' story:
steady progress that slows but does not stop, about 9.6 Elo per
generation from 50 to 90 and 4.2 from 90 to 150
(`docs/results/9x9/rerate/elo.csv`).

---

## 08 — The full loop at scale

**Concept: a reproducible, crash-safe training loop, and what it costs**

One **generation** is five steps, in `Trainer::runGeneration`:

```
1. self-play   games_per_generation games with the current network, on W threads
2. train       train_steps SGD steps on minibatches from the replay buffer
3. checkpoint  checkpoints/gen_NNNN.azn
4. arena       new network vs g-1 and chosen references; rating files refit
5. commit      state.bin (the commit point), then one row of log.csv
```

It is AlphaGo Zero's loop run synchronously: no step overlaps another.
An asynchronous actor/learner design would keep every core busy all the
time, but this one reads as a list of plain steps, and it makes two
guarantees that are hard to get otherwise: determinism and a clean
resume.

**Deterministic threading.** Self-play and arena games run on W worker
threads (all cores by default), which finish in an unpredictable order.
The output is identical for any W anyway, because of three rules:

- every random stream is seeded from its *identity*, never from its
  thread: game i of generation g's self-play uses
  `deriveSeed(seed, g, kPhaseSelfPlay, i)`, training uses
  `deriveSeed(seed, g, kPhaseTrain)`, the arena
  `deriveSeed(seed, g, kPhaseArena, opponent)`;
- results are stored by index, not by completion order;
- in the training phase the naive backend splits a gemm's output rows
  across threads, so every output element is computed by exactly one
  thread with the same sequence of floating-point operations.

```cpp
GenerationStats Trainer::runGeneration() {
    /* ... g = state_.generation + 1, workers, per-worker backends and evaluators ... */
        // Game i's randomness depends only on i, and its record is stored at
        // index i, so the result does not depend on which thread played it.
        parallelFor(numGames, workers, [&](int i, int w) {
            Rng rng(deriveSeed(cfg_.seed, static_cast<uint64_t>(g), kPhaseSelfPlay,
                               static_cast<uint64_t>(i)));
            records[static_cast<size_t>(i)] = playSelfPlayGame(*evaluators[static_cast<size_t>(w)], sp, rng);
        });
    /* ... statistics, SGF, then every record into the buffer in index order ... */
}
```
*src/trainer.cpp:313 — `Trainer::runGeneration` (abridged)*

`parallelFor` hands out indices from an atomic counter; which worker
plays game 17 varies from run to run, but game 17 itself does not. The
row split in the naive gemm keeps the arithmetic identical too:

```cpp
void NaiveBackend::gemm(bool transA, bool transB, int M, int N, int K, float alpha,
                        const float* A, const float* B, float beta, float* C) const {
    /* ... pack transposed operands; small products run on this thread ... */
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
```
*src/backend.cpp:138 — `NaiveBackend::gemm` (abridged)*

(A split over the *inner* dimension K would also be parallel, but
partial sums added in a different order give different floating-point
results.) The BLAS backend is single-threaded per call, so the same
holds for it. The payoff is visible in the results. The komi-24.5 pilot
ran with 5 worker threads while another run shared the machine; the
first seed-1 reference run, made with the same code (before the pass
fix explained in chapter 02, whose story is in chapter 07), used all 10. Apart from the three timing columns,
their logs are identical, to the last digit:

```
$ diff <(cut -d, -f1-9,13-15 docs/results/pilots/5x5-komi24.5.csv) \
       <(cut -d, -f1-9,13-15 docs/results/passbug/5x5-seed1/log.csv) && echo identical
identical
```

The final seed-1 run (`docs/results/5x5/log.csv`) differs from both from
generation 1 on — the pass fix changes what search plays — which is the
other half of the same point: the output is a function of the seed, the
configuration and the code, and of nothing else.

**Crash safety and resume.** A network checkpoint alone is not enough
to continue a run exactly; that also needs the optimizer's momentum, the
replay buffer, the generation counter, the gating state and the arena
table. All of it goes into one file, `state.bin`, and **`state.bin` is
the commit point**: a generation exists once its `state.bin` is on disk.
Everything written before it (the checkpoint, SGF files, `results.csv`,
`elo.csv`) may be left half-updated by a crash, and is then simply
rewritten by the rerun of the generation with identical bytes — every
random stream is derived from the seed, so the rerun is bit-identical. `log.csv` is appended after the commit; a crash between
the two loses one human-facing row, never the run. Every file is
replaced with the same routine:

```cpp
void writeFileAtomically(const std::string& path, const std::string& contents) {
    const std::string tmp = path + ".tmp";
    // Truncates any leftover .tmp from an earlier crash.
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
    /* ... write all of contents, retrying on EINTR ... */
    // The data must be on disk before the rename makes it visible under path.
    if (!syncFd(fd)) fail("cannot sync " + tmp);
    /* ... close ... */
    // rename() replaces the destination in one step on POSIX file systems:
    // readers see the old file or the new one, never a partial write.
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    /* ... then fsync the directory, so the rename itself survives a power loss ... */
}
```
*src/serialize.cpp:180 — `writeFileAtomically` (abridged)*

`syncFd` asks macOS for `F_FULLFSYNC`, because plain `fsync` there only
reaches the drive's cache; elsewhere it is `fsync`. Before anything is
committed, the trainer also rejects a bad generation: a non-finite loss
at any step, or any non-finite weight or momentum value after the last
step (which the per-step loss check cannot see), aborts with the
generation number and leaves generation g − 1's state untouched.
`train --run DIR --resume` then continues from `state.bin`. Tunable keys
may be overridden on resume; architecture keys may not, because they fix
the tensor shapes in every checkpoint:

```
$ ./build-blas/train --run runs/5x5-seed1 --resume --set blocks=4
error: cannot change architecture keys on resume: blocks
```

`tests/test_trainer.cpp` checks the strongest form of all this: two
generations, then a resume, then two more is byte-identical to four
uninterrupted generations.

**Try it** — the 5×5 reference run (with a `-DAZ_USE_BLAS=ON` build; the
presets say `backend = blas`, so on the default build add
`--set backend=naive`, about 3× slower on 5×5):

```
$ ./build-blas/train --config configs/5x5.cfg --run runs/5x5-seed1
starting runs/5x5-seed1 at generation 0; effective config (saved to runs/5x5-seed1/config.cfg):
...
gen 1 | selfplay 200 games avg 43.1 plies 6.5s | train loss p 2.85 v 0.49 lr 0.02 2.3s | arena elo 445.6 vs-prev 0.95 (distinct 19/20) 0.4s
gen 2 | selfplay 200 games avg 35.4 plies 6.7s | train loss p 2.89 v 0.42 lr 0.02 2.4s | arena elo 495.3 vs-prev 0.65 (distinct 20/20) 0.8s
...
gen 34 | selfplay 200 games avg 40.8 plies 7.3s | train loss p 1.76 v 0.24 lr 0.02 2.5s | arena elo 1422.2 vs-prev 0.70 (distinct 16/20) 2.0s
...
gen 60 | selfplay 200 games avg 43.3 plies 7.7s | train loss p 1.73 v 0.09 lr 0.002 2.5s | arena elo 1117.5 vs-prev 0.50 (distinct 13/20) 2.0s
done: generation 60 in runs/5x5-seed1
```

These lines come from the reference run, made with the old ladder. A run
today prints the `vs 0:… bot100:…` form instead, with different arena Elo
and arena times; the self-play and training numbers are unchanged, because
gating is off.

**Compute budgets.** What one generation costs on the M4, from the
`selfplay_s`, `train_s` and `arena_s` columns:

| board | preset | per generation | source |
|---|---|---|---|
| 5×5 | 3×32, 200 games × 200 sims | 11.3 s (self-play 7.1, train 2.5, arena 1.8); 60 generations in 11.3 min | `docs/results/5x5/log.csv` |
| 9×9 | 6×64, 100 games × 200 sims | 2.6 min in the 2-generation pilot (152–155 s); in the overnight run 4.1 min on average (self-play 2.0, train 1.0, arena 1.0; 2.7 to 4.8 min), 150 generations in 10.2 hours | `docs/results/pilots/9x9.csv`, `docs/results/9x9/log.csv` |
| 19×19 | 10×128, 20 games × 400 sims | 62.4 min in the 1-generation pilot (self-play 49.5, train 7.9, arena 5.0); in the check run 58.2 min for generation 1 and 129.7 min for generation 2 (self-play 110.5 min: the games grew from 145 to 365 plies) | `docs/results/pilots/19x19.csv`, `docs/results/19x19/` |

Self-play dominates everywhere — search needs hundreds of network
evaluations per move, training needs one forward and backward pass per
sample.

**The reference runs were made twice.** The first set of four runs
(about 12.7 hours of machine time) was made before the exact-pass rule
of chapter 02; chapter 07 tells how the 9×9 arena exposed the blind spot.
After the fix all four were repeated from scratch with the same presets
and seeds, so the before/after comparison differs in one thing only, the
code: generation 0's share of arena points against trained 9×9
generations fell from 31% (938 of 3,000) to 2.4% (72 of 3,000), the 9×9
refit ended at 674.5 instead of 91.8, and the 5×5 runs, where the blind
spot rarely matters, barely changed (`docs/results/passbug/README.md`) —
though the fix changes 5×5 play too, and their logs diverge from
generation 1 on. A
bug found by the evaluation, not by the tests, costs a full night of
compute; deterministic runs at least make the rerun a clean experiment.

**An honest comparison with AlphaGo Zero.** AlphaGo Zero's strongest
version trained a 40-block, 256-filter network for 40 days on 29 million
self-play games, searching 1,600 simulations per move (its 20-block
version: 4.9 million games in 3 days). DeepMind did not itemize its
self-play hardware in the AlphaGo Zero paper; for AlphaZero it reports
5,000 first-generation TPUs generating games. Now the M4. The 19×19
pilot played 20 games in 2,970.8 s of self-play
(`docs/results/pilots/19x19.csv`) — 148.5 s of the whole machine per
game, with a network a sixteenth of the size (a quarter of the blocks,
a quarter of F²) and a quarter of the simulations. At that rate, 29
million games would take **about 137 years**. Scaled up to AlphaGo
Zero's network and search budget — roughly 16 × 4 = 64 times the work
per game — it is **about 8,700 years**, before counting a single training
step, and assuming games no longer than the pilot's 151 plies (stronger
play, with its dead stones captured, likely runs longer; in the
two-generation check, generation 2's games already averaged 365 plies
and 332 s each, `docs/results/19x19/log.csv`). AlphaGo Zero's result
is a statement about compute as much as about algorithms. What this
machine *can* do is run the same algorithm, unchanged, on boards where a
night is enough to watch it learn.

---

## 09 — What changed from tic-tac-toe

| Component | alphazero-tictactoe | alphazero-go | Why |
|---|---|---|---|
| State | 9 cells, win check | padded 21×21 array, captures, suicide, positional superko (Zobrist), Tromp-Taylor score, komi | Go's rules; one type for sizes 5–19 |
| Encoding | 18 floats: mine, theirs | 2H + 2 planes: 8 positions of history plus two color planes | history reveals ko; an all-ones plane shows the edge |
| Network | MLP 18 → 64 → 64 → {9, 1} | ResNet: 3×3 conv stem, `blocks` residual blocks, conv policy and value heads, batch norm | translation-invariant patterns; depth |
| Arithmetic | loops over arrays | im2col + gemm behind a `Backend` (naive or BLAS) | nearly all time is gemm; BLAS uses the matrix unit |
| Search | recursive, one leaf per call, fresh tree per move | flat arena, batched descents with virtual loss, tree reuse with compaction | fill the matrix unit; keep work between moves |
| Legality in search | occupied cells | occupancy, suicide, superko against game and tree path | ko fights |
| Dirichlet α | fixed 0.3 | 10/n² (0.4 to 0.03) | legal-move count grows with the board |
| Temperature | whole game (9 plies) | first 8 / 20 / 30 plies | variety in the opening, clean endgame labels |
| Replay buffer | ring of encoded examples | window of whole games, planes rebuilt on sampling | history for free, 1 byte per point |
| Augmentation | none | 8 board symmetries | Go is symmetric |
| Optimizer | momentum, decay on every parameter, fixed LR | momentum, decay on weights only, stepped LR schedule | batch norm; longer runs |
| Evaluation | vs. exhaustive minimax | arena vs. own references and a uniform bot, Bradley-Terry Elo, Wilson intervals, optional gating | no oracle for Go |
| Randomness | `std::mt19937`, `random_device` | one specified generator, streams from `deriveSeed` | reproducible for any thread count |
| Parallelism | single thread | W self-play/arena workers, row-split gemm | a run takes hours, not minutes |
| Persistence | one weights file | checkpoints plus an atomic, fsync'd `state.bin`; resume | runs outlive a laptop lid |

What did *not* change is the part worth remembering: one network with
two heads, search as a policy-improvement operator, the Monte Carlo
result as the value target, and a loss that asks the network to predict
what search and the game already found.

---

## 10 — Concept map

A quick index back into the repository, for whichever idea you want to
see again in place.

| Concept | Where it lives |
|---|---|
| Groups, liberties, capture, suicide | `Board::isLegal`, `Board::play` — `src/board.cpp:131, 152` |
| Padded board and the `Off` sentinel | `Board::Board`, `Board::pad` — `src/board.cpp:47, 56` |
| Zobrist hashing and positional superko | `Board::hashAfter` — `src/board.cpp:174`; `Game::isLegal` — `src/game.cpp:19` |
| Tromp-Taylor area scoring, komi, the draw | `Board::areaScore` — `src/board.cpp:201`; `Game::resultForBlack` — `src/game.cpp:61` |
| Input planes and the all-ones color plane | `encodePlanes` — `src/encode.cpp:14` |
| Convolution as im2col + gemm | `Conv2d::forward`, `Conv2d::backward` — `src/layers.cpp:34, 46`; `Backend::im2col`, `Backend::col2im` — `src/backend.cpp:17, 50` |
| The compute backend seam | `Backend` — `include/azgo/backend.hpp:16`; `NaiveBackend::gemm` — `src/backend.cpp:138` |
| Batch norm, training vs. inference | `BatchNorm2d::forwardTrain`, `BatchNorm2d::forwardInfer`, `BatchNorm2d::backward` — `src/layers.cpp:85, 74, 119` |
| Residual ResNet, policy and value heads | `Network::forward`, `Network::backward` — `src/network.cpp:177, 277` |
| Growing a network (deeper or wider) without changing its function | `growBlocks`, `widenFilters` — `src/grow.cpp:16, 158` |
| The loss | `Network::loss`, `Network::trainStep` — `src/network.cpp:243, 162` |
| Gradient checking | `checkGradient` — `tests/test_layers.cpp:36` |
| PUCT selection | `Mcts::selectEdge` — `src/mcts.cpp:54` |
| Legal masking, superko in the tree | `Mcts::expand` — `src/mcts.cpp:133` |
| Backup and the sign flip | `Mcts::backup` — `src/mcts.cpp:184` |
| Terminal positions scored by the rules | `Mcts::isTerminal`, `Mcts::terminalValue` — `src/mcts.cpp:38, 49` |
| Virtual loss and batched search | `Mcts::search`, `Mcts::addVirtualLoss` — `src/mcts.cpp:196, 105` |
| The evaluator seam | `Evaluator` — `include/azgo/evaluator.hpp:17`; `LocalEvaluator::evaluate` — `src/evaluator.cpp:19` |
| Dirichlet root noise | `Mcts::setRootPriors` — `src/mcts.cpp:319`; `dirichlet` — `src/rng.cpp:122` |
| Temperature | `Mcts::selectAction` — `src/mcts.cpp:282`; `playSelfPlayGame` — `src/selfplay.cpp:14` |
| Tree reuse and compaction | `Mcts::advance` — `src/mcts.cpp:336` |
| Passes and the end of the game | `Game::play`, `Game::isTerminal` — `src/game.cpp:39, 53` |
| The value target z | `GameRecord::zAt` — `src/selfplay.cpp:9` |
| Replay buffer of whole games | `ReplayBuffer::encodeSample`, `ReplayBuffer::sample` — `src/replay_buffer.cpp:53, 75` |
| Symmetry augmentation | `transformPoint`, `transformAction` — `src/symmetry.cpp:6, 21` |
| SGD with momentum, weight decay | `SgdMomentum::step` — `src/optimizer.cpp:19` |
| Learning-rate schedule | `LrSchedule::at` — `src/optimizer.cpp:100` |
| Arena and opening diversity | `playArenaGame`, `playMatch` — `src/arena.cpp:50, 86` |
| Choosing opponents, gating | `selectOpponents`, `passesGate` — `src/arena.cpp:152, 180` |
| Arena rounds: calibration, promotion | `calibrateBots`, `runArenaRound` — `src/arena_round.cpp:22, 39` |
| Bradley-Terry Elo with a virtual draw | `fitBradleyTerry` — `src/elo.cpp:113` |
| Wilson intervals | `wilsonInterval` — `src/elo.cpp:184` |
| `log.csv` rating vs. the `elo.csv` refit | `ratingAtOrBefore`, `eloCsv` — `src/trainer.cpp:91, 64` |
| One generation | `Trainer::runGeneration` — `src/trainer.cpp:313` |
| Seed derivation | `deriveSeed` — `src/rng.cpp:27` |
| Work sharing by index | `parallelFor` — `src/parallel.cpp:12` |
| Atomic, durable writes | `writeFileAtomically`, `syncFd` — `src/serialize.cpp:180, 169` |
| Training state and resume | `saveTrainingState` — `src/run_state.cpp:27`; `Trainer::resume` — `src/trainer.cpp:261` |
| Inspecting a position | `printHeatMap` and `main` — `apps/analyze.cpp:70, 91` |

---

Build it and run the 5×5 preset yourself:

```sh
cmake -S . -B build-blas -DCMAKE_BUILD_TYPE=Release -DAZ_USE_BLAS=ON
cmake --build build-blas -j
./build-blas/train --config configs/5x5.cfg --run runs/my-5x5
./build-blas/analyze --checkpoint runs/my-5x5/checkpoints/gen_0060.azn --komi 24.5
```

Written by Mark Castelluccio. Drafted with the help of Claude
(Anthropic) through [Claude Code](https://claude.com/claude-code); every
quoted line of code is checked against the source by
`tools/check_citations.py`, and every number against the file it names.
