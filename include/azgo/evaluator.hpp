// The evaluator is the seam between search and the neural network.
// MCTS never touches Network directly: it hands a batch of encoded
// positions to an Evaluator and gets back one row of policy logits and
// one value per position. LocalEvaluator runs the shared network on the
// calling thread with its own scratch memory; UniformEvaluator is a
// "know-nothing" stand-in (equal priors, constant value) that lets tests
// check the search logic without any network at all. With backend = metal,
// workers get BatchingEvaluators instead (azgo/inference_hub.hpp).
#pragma once
#include <memory>
#include "azgo/network.hpp"

namespace azgo {

// What MCTS asks of "the network": a batch of encoded positions in, policy
// logits and values out. BatchingEvaluator (azgo/inference_hub.hpp), the
// GPU's, implements it too, gathering leaves from every worker.
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
    // The same, but evaluating in `shared`, a Workspace of net's shape that
    // the caller owns and keeps alive: e.g. one per arena worker for every
    // network the worker plays with, since it evaluates for one at a time.
    // A Workspace is scratch memory, so the outputs are the same as with one's own.
    LocalEvaluator(const Network& net, Workspace& shared);
    int boardSize() const override;
    int history() const override;
    int maxBatch() const override;
    void evaluate(const float* input, int B, float* logits, float* values) override;
    long long evaluations() const { return evaluations_; }   // positions evaluated so far
private:
    const Network& net_;
    std::unique_ptr<Workspace> own_;   // null when the workspace is shared
    Workspace& ws_;
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
