#include "azgo/mcts.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include "azgo/encode.hpp"

namespace azgo {

Mcts::Mcts(const MctsConfig& cfg, Evaluator& evaluator) : cfg_(cfg), eval_(evaluator) {
    assert(cfg_.simulations >= 1);
    assert(cfg_.batchSize >= 1 && cfg_.batchSize <= eval_.maxBatch());
    const int area = eval_.boardSize() * eval_.boardSize();
    // Allocated once: every evaluator call reuses these buffers.
    input_.resize(static_cast<size_t>(cfg_.batchSize) * numPlanes(eval_.history()) * area);
    logits_.resize(static_cast<size_t>(cfg_.batchSize) * (area + 1));
    values_.resize(cfg_.batchSize);
}

void Mcts::reset() {
    nodes_.clear();
    edges_.clear();
}

void Mcts::prepareRoot(const Game& game) {
    assert(game.size() == eval_.boardSize());
    game_ = &game;
    // Keep the tree only if its root is exactly the game's current position
    // (same stones, same side to move, same move number, same pass count:
    // the pass count decides whether a pass child is terminal).
    if (!nodes_.empty() && nodes_[0].board == game.board() && nodes_[0].ply == game.moveCount() &&
        nodes_[0].passes == game.consecutivePasses())
        return;
    reset();
    nodes_.push_back(MctsNode{game.board(), -1, game.moveCount(), game.consecutivePasses()});
}

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

int Mcts::descend(std::vector<int>& pathEdges) {
    pathEdges.clear();
    int cur = 0;
    while (!isTerminal(nodes_[cur]) && nodes_[cur].expanded) {
        const int e = selectEdge(cur);
        pathEdges.push_back(e);
        if (edges_[e].child < 0) {
            // Children are created lazily, the first time a descent takes the edge.
            const MctsNode& parent = nodes_[cur];
            const int action = edges_[e].action;
            const bool isPass = action == parent.board.passAction();
            MctsNode child{parent.board, cur, parent.ply + 1, isPass ? parent.passes + 1 : 0};
            child.board.play(action);
            edges_[e].child = static_cast<int>(nodes_.size());
            nodes_.push_back(child);   // may reallocate: `parent` is not used after this
        }
        cur = edges_[e].child;
    }
    return cur;
}

void Mcts::addVirtualLoss(int leaf, const std::vector<int>& pathEdges, int delta) {
    for (int e : pathEdges) edges_[e].virtualLoss += delta;
    for (int x = leaf; x != -1; x = nodes_[x].parent) nodes_[x].virtualLoss += delta;
}

void Mcts::evaluateLeaves(const std::vector<int>& leaves) {
    const int B = static_cast<int>(leaves.size());
    assert(B >= 1 && B <= cfg_.batchSize);
    const int H = eval_.history();
    const int area = eval_.boardSize() * eval_.boardSize();
    const size_t inputStride = static_cast<size_t>(numPlanes(H)) * area;
    std::vector<const Board*> history;
    for (int i = 0; i < B; ++i) {
        // History planes: the leaf, its ancestors up to the root, then the
        // game's earlier positions (the root is the game's current board).
        history.clear();
        for (int x = leaves[i]; x != -1 && static_cast<int>(history.size()) < H; x = nodes_[x].parent)
            history.push_back(&nodes_[x].board);
        for (int ply = game_->moveCount() - 1; ply >= 0 && static_cast<int>(history.size()) < H; --ply)
            history.push_back(&game_->boardAt(ply));
        encodeBoards(history, H, 0, input_.data() + i * inputStride);
    }
    eval_.evaluate(input_.data(), B, logits_.data(), values_.data());
    ++evaluatorBatches_;
    for (int i = 0; i < B; ++i)
        expand(leaves[i], logits_.data() + static_cast<size_t>(i) * (area + 1), values_[i]);
}

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
    float maxLogit = -std::numeric_limits<float>::infinity();
    for (int a : legal) maxLogit = std::max(maxLogit, logits[a]);
    std::vector<float> p(legal.size());
    float sum = 0.0f;
    for (size_t i = 0; i < legal.size(); ++i) {
        p[i] = std::exp(logits[legal[i]] - maxLogit);
        sum += p[i];
    }

    MctsNode& nd = nodes_[nodeIndex];
    nd.firstEdge = static_cast<int>(edges_.size());
    nd.numEdges = static_cast<int>(legal.size());
    for (size_t i = 0; i < legal.size(); ++i) {
        MctsEdge e;
        e.action = legal[i];
        e.rawPrior = p[i] / sum;
        e.prior = e.rawPrior;
        if (legal[i] == pass && nd.passes >= 1) {
            // Passing now ends the game, so its value is exact: score it
            // here instead of waiting for a visit the prior may never earn.
            // A pass changes no stones, so the chooser's result is the
            // score of this very board.
            e.unvisitedQ = scoreValue(board);
        }
        edges_.push_back(e);
    }
    nd.expanded = true;
    nd.netValue = value;
}

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

void Mcts::search(const Game& game, Rng& rng) {
    assert(!game.isTerminal());
    prepareRoot(game);
    // The root needs priors before the first descent; evaluating it is not a simulation.
    if (!nodes_[0].expanded) evaluateLeaves({0});
    setRootPriors(rng);   // fresh noise (or none) for this search

    std::vector<int> leaves;                  // pending leaves of this round
    std::vector<std::vector<int>> paths;      // their edge paths
    std::vector<int> path;
    int done = 0;
    while (done < cfg_.simulations) {
        leaves.clear();
        paths.clear();
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
        for (size_t i = 0; i < leaves.size(); ++i) {
            nodes_[leaves[i]].pending = false;
            addVirtualLoss(leaves[i], paths[i], -1);
            backup(leaves[i], paths[i], nodes_[leaves[i]].netValue);
            ++done;
        }
    }
    game_ = nullptr;
}

std::vector<int> Mcts::rootVisits() const {
    std::vector<int> v(eval_.boardSize() * eval_.boardSize() + 1, 0);
    if (nodes_.empty()) return v;
    const MctsNode& root = nodes_[0];
    for (int i = 0; i < root.numEdges; ++i) v[edges_[root.firstEdge + i].action] = edges_[root.firstEdge + i].visits;
    return v;
}

std::vector<float> Mcts::rootPriors() const {
    std::vector<float> p(eval_.boardSize() * eval_.boardSize() + 1, 0.0f);
    if (nodes_.empty()) return p;
    const MctsNode& root = nodes_[0];
    for (int i = 0; i < root.numEdges; ++i) p[edges_[root.firstEdge + i].action] = edges_[root.firstEdge + i].prior;
    return p;
}

int Mcts::rootTotalVisits() const {
    if (nodes_.empty()) return 0;
    int total = 0;
    for (int i = 0; i < nodes_[0].numEdges; ++i) total += edges_[nodes_[0].firstEdge + i].visits;
    return total;
}

float Mcts::rootValue() const {
    if (nodes_.empty()) return 0.0f;
    const MctsNode& root = nodes_[0];
    // Root edges hold values from the root player's view, so their mean is
    // the search's estimate of the root position.
    double sum = 0.0;
    int visits = 0;
    for (int i = 0; i < root.numEdges; ++i) {
        sum += edges_[root.firstEdge + i].valueSum;
        visits += edges_[root.firstEdge + i].visits;
    }
    return visits > 0 ? static_cast<float>(sum / visits) : root.netValue;
}

int Mcts::selectAction(float temperature, Rng& rng) const {
    const std::vector<int> visits = rootVisits();
    assert(rootTotalVisits() > 0);
    if (temperature <= 0.0f) {
        const int best = *std::max_element(visits.begin(), visits.end());
        std::vector<int> ties;
        for (int a = 0; a < static_cast<int>(visits.size()); ++a)
            if (visits[a] == best) ties.push_back(a);
        return ties[rng.below(static_cast<int>(ties.size()))];
    }
    // visits^(1/T) in double: T = 1 samples in proportion to visits, smaller
    // T sharpens towards the most visited action.
    std::vector<double> w(visits.size(), 0.0);
    double total = 0.0;
    for (size_t a = 0; a < visits.size(); ++a) {
        if (visits[a] > 0) w[a] = std::pow(static_cast<double>(visits[a]), 1.0 / temperature);
        total += w[a];
    }
    const double r = rng.uniform() * total;
    double cumulative = 0.0;
    int last = -1;
    for (size_t a = 0; a < w.size(); ++a) {
        if (w[a] <= 0.0) continue;
        cumulative += w[a];
        last = static_cast<int>(a);
        if (r < cumulative) return last;
    }
    return last;   // only reached through floating-point rounding at the very top
}

long long Mcts::totalVirtualLoss() const {
    long long total = 0;
    for (const MctsEdge& e : edges_) total += e.virtualLoss;
    for (const MctsNode& nd : nodes_) total += nd.virtualLoss;
    return total;
}

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

void Mcts::advance(int action) {
    if (nodes_.empty()) return;
    const MctsNode& root = nodes_[0];
    int newRoot = -1;
    for (int i = 0; i < root.numEdges; ++i)
        if (edges_[root.firstEdge + i].action == action) newRoot = edges_[root.firstEdge + i].child;
    if (newRoot < 0) {   // never explored (or illegal): nothing worth keeping
        reset();
        return;
    }
    // Copy only the subtree under newRoot into fresh vectors, so memory stays
    // bounded by what one move's search built. Explicit-stack DFS; each stack
    // entry is (old node, its new parent, the new edge that must point to it).
    std::vector<MctsNode> nodes;
    std::vector<MctsEdge> edges;
    struct Item { int oldNode, newParent, newEdge; };
    std::vector<Item> stack{{newRoot, -1, -1}};
    while (!stack.empty()) {
        const Item it = stack.back();
        stack.pop_back();
        const int idx = static_cast<int>(nodes.size());
        MctsNode nd = nodes_[it.oldNode];
        nd.parent = it.newParent;
        nd.firstEdge = static_cast<int>(edges.size());
        if (it.newEdge >= 0) edges[it.newEdge].child = idx;
        for (int i = 0; i < nd.numEdges; ++i) {
            MctsEdge e = edges_[nodes_[it.oldNode].firstEdge + i];
            const int oldChild = e.child;
            e.child = -1;   // filled in when that child is copied
            edges.push_back(e);
            if (oldChild >= 0) stack.push_back({oldChild, idx, static_cast<int>(edges.size()) - 1});
        }
        nodes.push_back(nd);
    }
    nodes_.swap(nodes);
    edges_.swap(edges);
}

} // namespace azgo
