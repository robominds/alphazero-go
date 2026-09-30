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

// Why unvisitedQ exists (and why it is not first-play-urgency tuning): with
// unvisited Q = 0, an edge with a tiny prior may stay unvisited for the whole
// search. That is harmless for unknown moves, but when that edge is the
// game-ending pass reply to a pass, the search never sees the exact, already
// known result, and a net that likes passing (its value head is right about
// the board, but dead stones still count under Tromp-Taylor) passes into a
// lost game. Only this one terminal edge gets a non-zero value; every
// other unvisited edge keeps Q = 0 and the exploration rule is unchanged.
//
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
    // Q used while the edge has no visits. Normally 0 ("first-play urgency
    // 0"). Set exactly once, in expand(), for the one edge whose result is
    // known without search: a pass after a pass ends the game, so its value
    // is the Tromp-Taylor score minus komi, from the chooser's view.
    float unvisitedQ = 0;
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
