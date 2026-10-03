// grow: deepens a checkpoint by appending residual blocks that start as exact
// identities, and/or widens it by copying channels and splitting the weights
// that read them, and/or adds an ownership head (see include/azgo/grow.hpp),
// so the new network plays exactly as the old one did and training can
// continue from there.
//
//   grow --from runs/nine/checkpoints/gen_0100.azn --blocks 10 --out runs/ten/start.azn
//   grow --from runs/nine/checkpoints/gen_0100.azn --filters 96 --out runs/wide/start.azn
//   grow --from runs/nine/checkpoints/gen_0100.azn --ownership --out runs/own/start.azn
//
// With both options it widens first, then deepens, so the new blocks are
// built at the new width; the other order gives an equally exact network.
// --ownership adds the head last; it changes neither policy nor value.
// Before saving, the tool checks the claim on 8 random positions: the grown
// network's policy and value must match the original's within kTolerance.
// Deepening reproduces them bit for bit. Widening is exact in real arithmetic,
// but float32 is not: on the 9x9 run's 6x64 network the original's own float32
// outputs are already about 6.5e-6 away from exact (float64) arithmetic, and
// widening changes how many terms each convolution adds up, so the float32
// results move by about 1e-5. The limit sits ten times above that noise and
// well below what a real mistake produces: a single mis-split weight gives
// about 1e-3, a wrong BN or channel 0.08 to 0.4.
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "azgo/backend.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/config.hpp"
#include "azgo/grow.hpp"
#include "azgo/network.hpp"
#include "azgo/rng.hpp"
#include "cli.hpp"

namespace {

const char* const kUsage =
    "usage: grow --from CKPT [--blocks N] [--filters N] [--ownership] --out PATH [--seed 1]\n"
    "\n"
    "Writes PATH: CKPT widened to --filters filters, then with residual blocks\n"
    "appended up to --blocks. New channels are copies of old ones and new blocks\n"
    "are exact identities, so PATH computes what CKPT computed. Each defaults to\n"
    "CKPT's own value and may not be below it. --ownership adds a randomly\n"
    "initialized ownership head, which changes neither policy nor value (CKPT\n"
    "must not have one). --seed seeds the new blocks' convolution weights, the\n"
    "widening's channel copies and weight shares, and the ownership head.\n";

constexpr double kTolerance = 1e-4;

int usageError(const std::string& message) {
    std::fprintf(stderr, "error: %s\n\n%s", message.c_str(), kUsage);
    return 1;
}

size_t parameterCount(azgo::Network& net) {
    size_t total = 0;
    for (const azgo::ParamView& p : net.params()) total += p.size;
    return total;
}

std::string shape(const azgo::NetConfig& c) {
    return std::to_string(c.boardSize) + "x" + std::to_string(c.boardSize) +
           ", history " + std::to_string(c.history) + ", " + std::to_string(c.blocks) +
           " blocks, " + std::to_string(c.filters) + " filters, value_hidden " +
           std::to_string(c.valueHidden) + (c.ownership ? ", ownership head" : "");
}

// Largest difference between the two networks' logits and values over `count`
// random {0, 1} input positions, and between their ownership maps when both
// have the head.
double maxOutputDifference(const azgo::Network& a, const azgo::Network& b, int count, azgo::Rng& rng) {
    const azgo::NetConfig& cfg = a.config();
    std::unique_ptr<azgo::Backend> backend = azgo::makeBackend("naive", 1);
    azgo::Workspace wa(a.config(), count, *backend), wb(b.config(), count, *backend);
    std::vector<float> input(static_cast<size_t>(count) * cfg.planes() * cfg.area());
    for (float& v : input) v = rng.uniform() < 0.5 ? 0.0f : 1.0f;
    std::vector<float> la(static_cast<size_t>(count) * cfg.actions()), lb(la.size());
    std::vector<float> va(count), vb(count);
    const bool ownership = a.config().ownership && b.config().ownership;
    std::vector<float> oa(ownership ? static_cast<size_t>(count) * cfg.area() : 0), ob(oa.size());
    a.infer(wa, input.data(), count, la.data(), va.data(), ownership ? oa.data() : nullptr);
    b.infer(wb, input.data(), count, lb.data(), vb.data(), ownership ? ob.data() : nullptr);
    double worst = 0;
    for (size_t i = 0; i < la.size(); ++i) worst = std::fmax(worst, std::fabs(la[i] - lb[i]));
    for (int i = 0; i < count; ++i) worst = std::fmax(worst, std::fabs(va[i] - vb[i]));
    for (size_t i = 0; i < oa.size(); ++i) worst = std::fmax(worst, std::fabs(oa[i] - ob[i]));
    return worst;
}

} // namespace

int main(int argc, char** argv) {
    std::unique_ptr<ArgParser> args;
    try {
        args = std::make_unique<ArgParser>(argc, argv,
                                           std::vector<std::string>{"from", "blocks", "filters", "out", "seed"},
                                           std::vector<std::string>{"help", "ownership"});
    } catch (const std::invalid_argument& e) {
        return usageError(e.what());
    }
    if (args->flag("help")) {
        std::fputs(kUsage, stdout);
        return 0;
    }
    try {
        if (!args->has("from")) return usageError("--from CKPT is required");
        if (!args->has("out")) return usageError("--out PATH is required");
        const int seed = args->getInt("seed", 1);
        if (seed < 0) return usageError("--seed must be >= 0");

        azgo::Network net = azgo::loadNetwork(args->get("from"));
        const int blocks = args->getInt("blocks", net.config().blocks);
        const int filters = args->getInt("filters", net.config().filters);
        if (blocks > azgo::kMaxBlocks)
            throw std::runtime_error("--blocks " + std::to_string(blocks) + " exceeds the maximum of " +
                                     std::to_string(azgo::kMaxBlocks));
        if (filters > azgo::kMaxFilters)
            throw std::runtime_error("--filters " + std::to_string(filters) + " exceeds the maximum of " +
                                     std::to_string(azgo::kMaxFilters));
        if (blocks < net.config().blocks)
            throw std::runtime_error("--blocks " + std::to_string(blocks) + " is fewer than the checkpoint's " +
                                     std::to_string(net.config().blocks) + "; growing only adds blocks");
        if (filters < net.config().filters)
            throw std::runtime_error("--filters " + std::to_string(filters) + " is fewer than the checkpoint's " +
                                     std::to_string(net.config().filters) + "; growing only adds filters");
        if (args->flag("ownership") && net.config().ownership)
            throw std::runtime_error("--ownership: " + args->get("from") + " already has an ownership head");
        azgo::Network grown = azgo::growBlocks(azgo::widenFilters(net, filters, static_cast<uint64_t>(seed)),
                                               blocks, static_cast<uint64_t>(seed));
        if (args->flag("ownership")) grown = azgo::addOwnershipHead(grown, static_cast<uint64_t>(seed));

        azgo::Rng rng(static_cast<uint64_t>(seed));
        const double diff = maxOutputDifference(net, grown, 8, rng);
        char limit[32];
        std::snprintf(limit, sizeof limit, "%.2g", kTolerance);
        if (!(diff <= kTolerance))
            throw std::runtime_error("the grown network's outputs differ from the original's by " +
                                     std::to_string(diff) + " (limit " + limit + "); not saving");

        const std::filesystem::path out = args->get("out");
        if (out.has_parent_path()) std::filesystem::create_directories(out.parent_path());
        azgo::saveNetwork(grown, out.string());

        std::printf("old: %s, %zu parameters\n", shape(net.config()).c_str(), parameterCount(net));
        std::printf("new: %s, %zu parameters\n", shape(grown.config()).c_str(), parameterCount(grown));
        if (grown.config() == net.config())
            std::printf("note: neither --blocks, --filters nor --ownership changes the shape; "
                        "PATH is a plain copy\n");
        std::printf("verified on 8 random positions (max difference %.2g); wrote %s\n", diff,
                    out.string().c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
