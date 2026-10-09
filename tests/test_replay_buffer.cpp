#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "azgo/encode.hpp"
#include "azgo/game.hpp"
#include "azgo/parallel.hpp"
#include "azgo/replay_buffer.hpp"
#include "azgo/symmetry.hpp"

using namespace azgo;

static bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

// Plays `moves` on a 5x5 Game and records it as self-play would: the board
// before each move, and visit counts {move: 3, pass: 1} (just {pass: 4} when
// the move is a pass). The result comes from the game unless it is still
// running, in which case `result` is used. finalOwnership is the board after
// the last move, as self-play records it.
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
    rec.finalOwnership.resize(area);
    game.board().ownership(rec.finalOwnership.data());
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
    assert(batch.ownership.size() == static_cast<size_t>(B) * 25);
    for (float o : batch.ownership) assert(o == -1.0f || o == 0.0f || o == 1.0f);
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
        assert(a.finalOwnership == b.finalOwnership && a.resigned == b.resigned);
    }
    TrainingBatch x, y;
    Rng r1(77), r2(77);
    buf.sample(64, 3, r1, x);
    back.sample(64, 3, r2, y);
    assert(std::memcmp(x.input.data(), y.input.data(), x.input.size() * sizeof(float)) == 0);
    assert(std::memcmp(x.pi.data(), y.pi.data(), x.pi.size() * sizeof(float)) == 0);
    assert(std::memcmp(x.z.data(), y.z.data(), x.z.size() * sizeof(float)) == 0);
    assert(x.ownership == y.ownership);

    // The older layout (state.bin versions 1 and 2) has no ownership and no
    // resigned flag; loading it recomputes each game's ownership by replaying
    // the moves, and gets the same.
    std::ostringstream oldOs;
    BinaryWriter oldW(oldOs);
    buf.save(oldW, false, false);
    assert(oldOs.str().size() == bytes.size() - (25 + 1) * buf.games());
    std::istringstream oldIs(oldOs.str());
    BinaryReader oldR(oldIs);
    ReplayBuffer recomputed = ReplayBuffer::load(oldR, false, false);
    for (size_t g = 0; g < buf.games(); ++g)
        assert(recomputed.game(g).finalOwnership == buf.game(g).finalOwnership);

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
    buf.save(w, false, false);
    std::istringstream is(os.str());
    BinaryReader r(is);
    std::string msg;
    try {
        ReplayBuffer::load(r, false, false);
    } catch (const std::runtime_error& e) {
        msg = e.what();
    }
    assert(msg == "replay buffer: game 1 does not replay: move 2 (B4) is illegal");
}

// State.bin version 5 keeps whether each game ended by resignation (a
// resigned game's final board is no game end, so ownership cannot use it).
// Version 3 and 4 files have no flag and load as not resigned.
static void test_resigned_flag_round_trip() {
    ReplayBuffer buf(100);
    GameRecord resigned = makeRecord({6, 18, 12});
    resigned.resigned = true;
    buf.add(resigned);
    buf.add(makeRecord({6, 18}));
    std::ostringstream os;
    BinaryWriter w(os);
    buf.save(w);
    std::istringstream is(os.str());
    BinaryReader r(is);
    const ReplayBuffer back = ReplayBuffer::load(r);
    assert(back.game(0).resigned && !back.game(1).resigned);

    std::ostringstream v4;
    BinaryWriter w4(v4);
    buf.save(w4, true, false);
    assert(v4.str().size() == os.str().size() - buf.games());
    std::istringstream is4(v4.str());
    BinaryReader r4(is4);
    const ReplayBuffer old = ReplayBuffer::load(r4, true, false);
    assert(!old.game(0).resigned && !old.game(1).resigned);

    // The flag is one byte, 0 or 1; anything else is a damaged file.
    std::string bad = os.str();
    bad[bad.size() - 1] = 2;   // the second game's flag is the last byte
    std::istringstream isBad(bad);
    BinaryReader rBad(isBad);
    bool threw = false;
    try {
        ReplayBuffer::load(rBad);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("resigned") != std::string::npos;
    }
    assert(threw);
}

// locate numbers positions from the oldest game's first, as sample() draws
// them; encodeTrainingSample encodes a position of any record, buffered or
// not, exactly as the buffer does.
static void test_locate_and_encode_training_sample() {
    ReplayBuffer buf(100);
    buf.add(makeRecord({6, 18, 12, 25}));
    buf.add(makeRecord({25, 25}));
    buf.add(makeRecord({7}, 7.5f, -1.0f));
    assert(buf.positions() == 7);
    const std::pair<size_t, int> expected[] = {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 0}, {1, 1}, {2, 0}};
    for (size_t i = 0; i < 7; ++i) assert(buf.locate(i) == expected[i]);

    const int H = 2, planes = numPlanes(H) * 25;
    std::vector<float> in1(planes), in2(planes), pi1(26), pi2(26), own1(25), own2(25);
    float z1 = 0, z2 = 0;
    for (size_t g = 0; g < buf.games(); ++g)
        for (int ply = 0; ply < buf.game(g).positions(); ++ply)
            for (int sym : {0, 5}) {
                buf.encodeSample(g, ply, H, sym, in1.data(), pi1.data(), &z1, own1.data());
                encodeTrainingSample(buf.game(g), ply, H, sym, in2.data(), pi2.data(), &z2, own2.data());
                assert(in1 == in2 && pi1 == pi2 && z1 == z2 && own1 == own2);
            }
}

// Sampling on a pool draws the same positions and symmetries, in the same
// order, and leaves the generator in the same state: the batch is the same
// bytes for any thread count.
static void test_sample_on_a_pool_is_identical() {
    ReplayBuffer buf(1000);
    for (int t = 0; t < 7; ++t) buf.add(lengthGame(3 + t, t));
    ThreadPool p2(2), p5(5);
    for (int B : {1, 16, 37}) {
        Rng r0(77), r2(77), r5(77);
        TrainingBatch a, b, c;
        buf.sample(B, 3, r0, a);
        buf.sample(B, 3, r2, b, &p2);
        buf.sample(B, 3, r5, c, &p5);
        for (const TrainingBatch* x : {&b, &c}) {
            assert(x->size == a.size && x->input == a.input && x->pi == a.pi && x->z == a.z &&
                   x->ownership == a.ownership);
        }
        const uint64_t next = r0.next();
        assert(r2.next() == next && r5.next() == next);
    }
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
    test_ownership_target_perspective_and_symmetry();
    test_ownership_target_with_territory();
    test_save_load_round_trip();
    test_old_layout_that_does_not_replay();
    test_resigned_flag_round_trip();
    test_locate_and_encode_training_sample();
    test_sample_on_a_pool_is_identical();
    std::printf("test_replay_buffer: all tests passed\n");
    return 0;
}
