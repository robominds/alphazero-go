// analyze: shows what a checkpoint thinks of one position -- the raw network
// (value and policy heat map) next to what search makes of it (MCTS visits).
// The gap between the two is the whole point of AlphaZero: search corrects
// the network's guess, and training teaches the network to guess the result.
// A checkpoint with an ownership head also shows who it thinks will own each
// point at the end, which shows how it judges every group on the board.
//
//   analyze --checkpoint runs/nine/checkpoints/gen_0040.azn --moves "E5 C3 pass"
//   analyze --checkpoint gen_0040.azn --sgf game.sgf --simulations 800
//   analyze --checkpoint gen_0040.azn --moves "E5 C3" --random-symmetry
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "azgo/backend.hpp"
#include "azgo/board.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/encode.hpp"
#include "azgo/evaluator.hpp"
#include "azgo/game.hpp"
#include "azgo/mcts.hpp"
#include "azgo/network.hpp"
#include "azgo/rng.hpp"
#include "azgo/sgf.hpp"
#include "cli.hpp"

namespace {

const char* const kUsage =
    "usage: analyze --checkpoint CKPT [--moves \"D4 C3 pass\"] [--sgf FILE]\n"
    "               [--simulations 400] [--komi 7.5] [--backend naive] [--random-symmetry]\n"
    "\n"
    "Plays the moves (GTP coordinates, or the main line of an SGF file) from the\n"
    "empty board, then prints the network's value, its ownership map (only for a\n"
    "checkpoint with an ownership head) and policy for the position, and the\n"
    "result of an MCTS search. With --sgf the file's komi is used unless --komi\n"
    "is given. --random-symmetry makes the search evaluate each position under a\n"
    "random one of the 8 board symmetries, as AlphaGo Zero did, so each node's\n"
    "priors come from one random transform; the raw network output printed above\n"
    "the search always sees the board as it is.\n";

constexpr char kColumns[] = "ABCDEFGHJKLMNOPQRST";   // GTP skips I
constexpr int kMctsBatch = 8;
constexpr int kTopMoves = 10;

int usageError(const std::string& message) {
    std::fprintf(stderr, "error: %s\n\n%s", message.c_str(), kUsage);
    return 1;
}

std::string readTextFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<std::string> splitWhitespace(const std::string& text) {
    std::istringstream in(text);
    std::vector<std::string> out;
    for (std::string tok; in >> tok;) out.push_back(tok);
    return out;
}

// Same frame as a Go diagram: columns A.. left to right, row n at the top.
// Each point shows its prior as a whole percent, "." below 1%, or the stone.
void printHeatMap(const azgo::Board& board, const std::vector<double>& prior) {
    const int n = board.size();
    std::printf("   ");
    for (int c = 0; c < n; ++c) std::printf("  %c", kColumns[c]);
    std::printf("\n");
    for (int r = 0; r < n; ++r) {
        std::printf("%2d ", n - r);
        for (int c = 0; c < n; ++c) {
            const int p = r * n + c;
            const azgo::Color stone = board.at(p);
            if (stone == azgo::Color::Black) std::printf("  X");
            else if (stone == azgo::Color::White) std::printf("  O");
            else if (prior[static_cast<size_t>(p)] < 0.01) std::printf("  .");
            else std::printf("%3ld", std::lround(100.0 * prior[static_cast<size_t>(p)]));
        }
        std::printf("\n");
    }
}

// Who the network expects to own each point when the game ends, in Black's
// terms: 9 * ownership rounded, from +9 (surely Black's) to -9 (surely
// White's), "." when |ownership| < 0.1. Stones are not drawn: whether a stone
// survives is exactly what the map answers. Same frame as printHeatMap.
void printOwnershipMap(int n, const std::vector<float>& owner) {
    std::printf("   ");
    for (int c = 0; c < n; ++c) std::printf("  %c", kColumns[c]);
    std::printf("\n");
    for (int r = 0; r < n; ++r) {
        std::printf("%2d ", n - r);
        for (int c = 0; c < n; ++c) {
            const float o = owner[static_cast<size_t>(r * n + c)];
            if (std::fabs(o) < 0.1f) std::printf("  .");
            else std::printf("%+3ld", std::lround(9.0 * o));
        }
        std::printf("\n");
    }
}

} // namespace

int main(int argc, char** argv) {
    std::unique_ptr<ArgParser> args;
    try {
        args = std::make_unique<ArgParser>(
            argc, argv,
            std::vector<std::string>{"checkpoint", "moves", "sgf", "simulations", "komi", "backend"},
            std::vector<std::string>{"help", "random-symmetry"});
    } catch (const std::invalid_argument& e) {
        return usageError(e.what());
    }
    if (args->flag("help")) {
        std::fputs(kUsage, stdout);
        return 0;
    }
    if (!args->has("checkpoint")) return usageError("--checkpoint CKPT is required");
    if (args->has("moves") && args->has("sgf")) return usageError("give --moves or --sgf, not both");

    try {
        const int simulations = args->getInt("simulations", 400);
        if (simulations < 1) throw std::invalid_argument("--simulations must be at least 1");
        float komi = static_cast<float>(args->getDouble("komi", 7.5));
        const std::string ckpt = args->get("checkpoint");

        const azgo::Network net = azgo::loadNetwork(ckpt);
        const azgo::NetConfig& nc = net.config();
        const int n = nc.boardSize;

        // (text as the user wrote it, action or -1). Moves are validated one
        // at a time below so the error can name the first bad one.
        std::vector<std::pair<std::string, int>> moves;
        if (args->has("sgf")) {
            const azgo::SgfGame sgf = azgo::parseSgf(readTextFile(args->get("sgf")));
            if (sgf.boardSize != n)
                throw std::runtime_error("the SGF is for " + std::to_string(sgf.boardSize) + "x" +
                                         std::to_string(sgf.boardSize) + " but the checkpoint is " +
                                         std::to_string(n) + "x" + std::to_string(n));
            if (!args->has("komi")) komi = sgf.komi;
            for (int a : sgf.moves) moves.emplace_back(azgo::actionToString(a, n), a);
        } else {
            for (const std::string& tok : splitWhitespace(args->get("moves")))
                moves.emplace_back(tok, azgo::parseAction(tok, n));
        }

        // User input, so every problem is a reported error, never an assert
        // inside Game::play.
        azgo::Game game(n, komi, azgo::defaultMaxMoves(n));
        for (size_t k = 0; k < moves.size(); ++k) {
            const std::string& text = moves[k].first;
            const int a = moves[k].second;
            std::string reason;
            if (a < 0) {
                reason = "not a move on a " + std::to_string(n) + "x" + std::to_string(n) + " board";
            } else if (game.isTerminal()) {
                reason = "game is over";
            } else if (!game.isLegal(a)) {
                // Pass is always legal before the end, so `a` is a point here.
                // Board::isLegal knows occupancy and suicide; what it allows
                // but Game refuses must be a repeated position.
                if (game.board().at(a) != azgo::Color::Empty) reason = "illegal (occupied)";
                else if (!game.board().isLegal(a)) reason = "illegal (suicide)";
                else reason = "illegal (superko)";
            }
            if (!reason.empty()) {
                std::fprintf(stderr, "error: move %zu (%s): %s\n", k + 1, text.c_str(), reason.c_str());
                return 1;
            }
            game.play(a);
        }
        if (game.isTerminal())
            throw std::runtime_error("the game is over after move " + std::to_string(game.moveCount()) +
                                     " (two passes or the move cap); there is nothing to analyze");

        const azgo::Board& board = game.board();
        std::printf("%s: %d moves played, %s to move, komi %g\n\n", ckpt.c_str(), game.moveCount(),
                    board.toMove() == azgo::Color::Black ? "Black" : "White", komi);
        std::printf("%s\n", board.toString().c_str());

        // ---- The raw network --------------------------------------------------
        std::unique_ptr<azgo::Backend> backend = azgo::makeBackend(args->get("backend", "naive"), 1);
        azgo::LocalEvaluator evaluator(net, *backend, kMctsBatch);
        const int actions = n * n + 1;
        std::vector<float> input(static_cast<size_t>(azgo::numPlanes(nc.history)) * n * n);
        azgo::encodeGame(game, nc.history, 0, input.data());
        std::vector<float> logits(static_cast<size_t>(actions));
        float value = 0;
        evaluator.evaluate(input.data(), 1, logits.data(), &value);
        std::printf("Value (side to move): %+.2f\n\n", value);

        if (nc.ownership) {
            // The evaluator interface carries no ownership, so ask the network
            // directly; it answers for the side to move, the map is Black's.
            azgo::Workspace ws(nc, 1, *backend);
            std::vector<float> headLogits(static_cast<size_t>(actions)), owner(static_cast<size_t>(n) * n);
            float headValue = 0;
            net.infer(ws, input.data(), 1, headLogits.data(), &headValue, owner.data());
            if (board.toMove() == azgo::Color::White)
                for (float& o : owner) o = -o;
            std::printf("Ownership, Black + / White - (9 = certain, \".\" below 0.1):\n");
            printOwnershipMap(n, owner);
            std::printf("\n");
        }

        // Priors as MCTS sees them: softmax over the legal actions only.
        const std::vector<int> legal = game.legalActions();
        std::vector<double> prior(static_cast<size_t>(actions), 0.0);
        float maxLogit = -std::numeric_limits<float>::infinity();
        for (int a : legal) maxLogit = std::max(maxLogit, logits[static_cast<size_t>(a)]);
        double sum = 0;
        for (int a : legal) {
            const double p = std::exp(static_cast<double>(logits[static_cast<size_t>(a)] - maxLogit));
            prior[static_cast<size_t>(a)] = p;
            sum += p;
        }
        for (int a : legal) prior[static_cast<size_t>(a)] /= sum;

        std::printf("Policy prior, %% of legal moves (\".\" below 1%%):\n");
        printHeatMap(board, prior);
        std::printf("Pass prior: %.1f%%\n\n", 100.0 * prior[static_cast<size_t>(n * n)]);

        std::vector<int> byPrior = legal;
        std::stable_sort(byPrior.begin(), byPrior.end(), [&](int x, int y) {
            return prior[static_cast<size_t>(x)] > prior[static_cast<size_t>(y)];
        });
        std::printf("Top %d moves by prior:\n", kTopMoves);
        for (size_t i = 0; i < byPrior.size() && i < static_cast<size_t>(kTopMoves); ++i)
            std::printf("  %-5s %5.1f%%\n", azgo::actionToString(byPrior[i], n).c_str(),
                        100.0 * prior[static_cast<size_t>(byPrior[i])]);

        // ---- Search -------------------------------------------------------------
        azgo::MctsConfig mc;
        mc.simulations = simulations;
        mc.batchSize = kMctsBatch;
        mc.addNoise = false;
        mc.randomSymmetry = args->flag("random-symmetry");
        azgo::Mcts mcts(mc, evaluator);
        azgo::Rng rng(1);
        mcts.search(game, rng);
        std::printf("\nMCTS value (side to move): %+.2f after %d simulations%s\n", mcts.rootValue(),
                    simulations, mc.randomSymmetry ? " with random symmetry" : "");

        // The root is the one node without a parent. Its edges carry Q from
        // the root player's point of view: valueSum / visits.
        int root = 0;
        for (size_t i = 0; i < mcts.nodeCount(); ++i)
            if (mcts.node(static_cast<int>(i)).parent == -1) root = static_cast<int>(i);
        const azgo::MctsNode& rootNode = mcts.node(root);
        std::vector<const azgo::MctsEdge*> visited;
        long long totalVisits = 0;
        for (int e = rootNode.firstEdge; e < rootNode.firstEdge + rootNode.numEdges; ++e) {
            const azgo::MctsEdge& edge = mcts.edge(e);
            if (edge.visits > 0) {
                visited.push_back(&edge);
                totalVisits += edge.visits;
            }
        }
        std::stable_sort(visited.begin(), visited.end(), [](const azgo::MctsEdge* x, const azgo::MctsEdge* y) {
            return x->visits > y->visits;
        });
        std::printf("Top %d moves by visits:\n", kTopMoves);
        for (size_t i = 0; i < visited.size() && i < static_cast<size_t>(kTopMoves); ++i) {
            const azgo::MctsEdge& edge = *visited[i];
            std::printf("  %-5s %6d visits (%5.1f%%)  Q %+.2f\n", azgo::actionToString(edge.action, n).c_str(),
                        edge.visits, 100.0 * edge.visits / static_cast<double>(totalVisits),
                        edge.valueSum / edge.visits);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
