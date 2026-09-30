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
