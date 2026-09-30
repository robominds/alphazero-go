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
