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
