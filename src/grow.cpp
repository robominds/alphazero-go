#include "azgo/grow.hpp"

#include <algorithm>
#include <cassert>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "azgo/config.hpp"
#include "azgo/rng.hpp"

namespace azgo {

Network growBlocks(const Network& net, int newBlocks, uint64_t seed) {
    const NetConfig& old = net.config();
    if (newBlocks < old.blocks)
        throw std::invalid_argument("cannot shrink a network from " + std::to_string(old.blocks) +
                                    " to " + std::to_string(newBlocks) + " blocks");
    NetConfig cfg = old;
    cfg.blocks = newBlocks;
    Network grown(cfg, seed);   // He-initialized; the copy below overwrites what exists in `net`

    // params() and buffers() are non-const accessors; read the source through
    // a copy so growBlocks can take a const reference.
    Network source = net;
    std::map<std::string, ParamView> sourceParams;
    for (const ParamView& p : source.params()) sourceParams[p.name] = p;
    std::map<std::string, BufferView> sourceBuffers;
    for (const BufferView& b : source.buffers()) sourceBuffers[b.name] = b;

    auto endsWith = [](const std::string& s, const char* suffix) {
        const std::string t(suffix);
        return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
    };
    size_t copied = 0;
    for (const ParamView& p : grown.params()) {
        auto it = sourceParams.find(p.name);
        if (it != sourceParams.end()) {
            std::copy(it->second.value, it->second.value + it->second.size, p.value);
            ++copied;
        } else if (endsWith(p.name, ".bn2.gamma") || endsWith(p.name, ".bn2.beta")) {
            std::fill(p.value, p.value + p.size, 0.0f);   // new block: F(x) = 0, an identity
        }
    }
    for (const BufferView& b : grown.buffers()) {
        auto it = sourceBuffers.find(b.name);
        if (it != sourceBuffers.end()) {
            std::copy(it->second.value, it->second.value + it->second.size, b.value);
            ++copied;
        }
    }
    if (copied < sourceParams.size() + sourceBuffers.size())
        throw std::logic_error("growBlocks: some tensors of the source network have no counterpart");
    return grown;
}

namespace {

// Name -> tensor lookups over a network's parameters and running statistics.
// `written` records the names handed out for writing, so widenFilters can
// check that it filled every tensor of the new network.
struct Tensors {
    std::map<std::string, ParamView> params;
    std::map<std::string, BufferView> buffers;
    mutable std::set<std::string> written;
    explicit Tensors(Network& net) {
        for (const ParamView& p : net.params()) params[p.name] = p;
        for (const BufferView& b : net.buffers()) buffers[b.name] = b;
    }
    const float* param(const std::string& name) const { return params.at(name).value; }
    const float* buffer(const std::string& name) const { return buffers.at(name).value; }
    float* writeParam(const std::string& name) const { written.insert(name); return params.at(name).value; }
    float* writeBuffer(const std::string& name) const { written.insert(name); return buffers.at(name).value; }
};

// copies[c] lists the new channels that are copies of old channel c.
std::vector<std::vector<int>> copiesOf(const std::vector<int>& map, int oldChannels) {
    std::vector<std::vector<int>> copies(oldChannels);
    for (int j = 0; j < static_cast<int>(map.size()); ++j) copies[map[j]].push_back(j);
    return copies;
}

// Widens a conv weight [oldOut][oldIn * taps] into [outMap.size()][inMap.size() * taps].
// New output row o is old row outMap[o]. New input channel i reads old
// channel inMap[i]: each old weight w is split among the copies of its input
// channel in random positive shares that sum to w, drawn afresh for every
// (output row, old input channel, tap). The copies carry identical values,
// so the row's sum is unchanged; the last copy takes the remainder so the
// float shares add up to w as closely as rounding allows.
void widenConv(const float* src, int oldIn, int taps, const std::vector<int>& outMap,
               const std::vector<int>& inMap, Rng& rng, float* dst) {
    const int newIn = static_cast<int>(inMap.size());
    const std::vector<std::vector<int>> copies = copiesOf(inMap, oldIn);
    std::vector<double> share;
    for (size_t o = 0; o < outMap.size(); ++o) {
        const float* srcRow = src + static_cast<size_t>(outMap[o]) * oldIn * taps;
        float* dstRow = dst + o * static_cast<size_t>(newIn) * taps;
        for (int c = 0; c < oldIn; ++c) {
            const std::vector<int>& js = copies[c];
            for (int t = 0; t < taps; ++t) {
                const float w = srcRow[c * taps + t];
                if (js.size() == 1) {
                    dstRow[js[0] * taps + t] = w;
                    continue;
                }
                // Shares from [0.5, 1.5) before normalizing: unequal, but
                // none close to zero, so every copy keeps a real part of w.
                share.resize(js.size());
                double total = 0;
                for (double& r : share) total += (r = 0.5 + rng.uniform());
                float given = 0.0f;
                for (size_t m = 0; m + 1 < js.size(); ++m) {
                    const float part = static_cast<float>(w * share[m] / total);
                    dstRow[js[m] * taps + t] = part;
                    given += part;
                }
                dstRow[js.back() * taps + t] = w - given;
            }
        }
    }
}

// Widens a batch norm over channels: new channel j copies old channel map[j]
// (gamma, beta, running mean and variance).
void widenNorm(const Tensors& from, const Tensors& to, const std::string& prefix,
               const std::vector<int>& map) {
    for (const char* suffix : {".gamma", ".beta"}) {
        const float* src = from.param(prefix + suffix);
        float* dst = to.writeParam(prefix + suffix);
        for (size_t j = 0; j < map.size(); ++j) dst[j] = src[map[j]];
    }
    for (const char* suffix : {".runMean", ".runVar"}) {
        const float* src = from.buffer(prefix + suffix);
        float* dst = to.writeBuffer(prefix + suffix);
        for (size_t j = 0; j < map.size(); ++j) dst[j] = src[map[j]];
    }
}

void copyTensor(const Tensors& from, const Tensors& to, const std::string& name, bool isParam) {
    const float* src = isParam ? from.param(name) : from.buffer(name);
    float* dst = isParam ? to.writeParam(name) : to.writeBuffer(name);
    const size_t size = isParam ? from.params.at(name).size : from.buffers.at(name).size;
    assert(size == (isParam ? to.params.at(name).size : to.buffers.at(name).size));
    std::copy(src, src + size, dst);
}

// j -> j for the first `oldChannels` channels, a random old channel for the rest.
std::vector<int> channelMap(int oldChannels, int newChannels, Rng& rng) {
    std::vector<int> map(newChannels);
    for (int j = 0; j < newChannels; ++j) map[j] = j < oldChannels ? j : rng.below(oldChannels);
    return map;
}

} // namespace

Network widenFilters(const Network& net, int newFilters, uint64_t seed) {
    const NetConfig& old = net.config();
    if (newFilters < old.filters)
        throw std::invalid_argument("cannot narrow a network from " + std::to_string(old.filters) +
                                    " to " + std::to_string(newFilters) + " filters");
    if (newFilters > kMaxFilters)
        throw std::invalid_argument(std::to_string(newFilters) + " filters exceeds the maximum of " +
                                    std::to_string(kMaxFilters));
    if (newFilters == old.filters) return net;
    NetConfig cfg = old;
    cfg.filters = newFilters;
    Network wide(cfg, seed);   // every tensor is overwritten below
    Network source = net;      // params() is non-const; see growBlocks
    const Tensors from(source), to(wide);

    Rng rng(seed);
    const int F = old.filters, P = old.planes();
    // g: the residual stream. Every layer that writes the stream (the stem
    // and each block's conv2 + bn2) uses the same g, so the skip additions
    // keep a channel's copies equal all the way up the trunk.
    const std::vector<int> g = channelMap(F, newFilters, rng);
    std::vector<int> identityPlanes(P);
    for (int p = 0; p < P; ++p) identityPlanes[p] = p;
    widenConv(from.param("stem.conv.w"), P, 9, g, identityPlanes, rng, to.writeParam("stem.conv.w"));
    widenNorm(from, to, "stem.bn", g);

    for (int i = 0; i < old.blocks; ++i) {
        const std::string prefix = "block" + std::to_string(i);
        // h: this block's internal channels (conv1's outputs, conv2's inputs).
        const std::vector<int> h = channelMap(F, newFilters, rng);
        widenConv(from.param(prefix + ".conv1.w"), F, 9, h, g, rng, to.writeParam(prefix + ".conv1.w"));
        widenNorm(from, to, prefix + ".bn1", h);
        widenConv(from.param(prefix + ".conv2.w"), F, 9, g, h, rng, to.writeParam(prefix + ".conv2.w"));
        widenNorm(from, to, prefix + ".bn2", g);
    }

    // The heads read the stream (their 1x1 convs' inputs widen); everything
    // after those convs keeps its shape and is copied as is.
    widenConv(from.param("policy.conv.w"), F, 1, {0, 1}, g, rng, to.writeParam("policy.conv.w"));
    widenConv(from.param("value.conv.w"), F, 1, {0}, g, rng, to.writeParam("value.conv.w"));
    for (const char* name : {"policy.bn.gamma", "policy.bn.beta", "policy.fc.w", "policy.fc.b",
                             "value.bn.gamma", "value.bn.beta", "value.fc1.w", "value.fc1.b",
                             "value.fc2.w", "value.fc2.b"})
        copyTensor(from, to, name, true);
    for (const char* name : {"policy.bn.runMean", "policy.bn.runVar", "value.bn.runMean", "value.bn.runVar"})
        copyTensor(from, to, name, false);
    if (to.written.size() != to.params.size() + to.buffers.size())
        throw std::logic_error("widenFilters: some tensors of the widened network were not filled");
    return wide;
}

} // namespace azgo
