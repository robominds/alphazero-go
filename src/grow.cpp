#include "azgo/grow.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>

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

} // namespace azgo
