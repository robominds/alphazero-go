#include "azgo/run_state.hpp"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "azgo/serialize.hpp"

namespace azgo {

namespace {
constexpr uint32_t kStateMagic = 0x54535A41;   // "AZST" read as little-endian bytes
// 2 added the reference set and bot list, 3 the buffer's ownership targets.
constexpr uint32_t kStateVersion = 3;
} // namespace

std::vector<int> initialReferences(int numBots) {
    if (numBots < 0) throw std::invalid_argument("initialReferences: numBots must be >= 0");
    std::vector<int> refs;
    for (int k = numBots; k >= 1; --k) refs.push_back(-k);   // ascending: -numBots first
    refs.push_back(0);
    return refs;
}

void saveTrainingState(const std::string& path, const TrainingState& state) {
    // Build the whole file in memory first: writeFileAtomically then either
    // replaces the old state completely or (on a crash) not at all.
    std::ostringstream os(std::ios::binary);
    BinaryWriter w(os);
    w.u32(kStateMagic);
    w.u32(kStateVersion);
    w.i32(state.generation);
    w.i32(state.bestGeneration);
    w.u64(static_cast<uint64_t>(state.samplesTrained));
    state.buffer.save(w);
    w.u32(static_cast<uint32_t>(state.velocity.size()));
    for (const std::vector<float>& v : state.velocity) {
        w.u64(v.size());
        w.floats(v.data(), v.size());
    }
    w.str(state.results.toCsv());
    w.u32(static_cast<uint32_t>(state.references.size()));
    for (int r : state.references) w.i32(r);
    w.u32(static_cast<uint32_t>(state.botSimulations.size()));
    for (int b : state.botSimulations) w.i32(b);
    writeFileAtomically(path, os.str());
}

TrainingState loadTrainingState(const std::string& path, const std::vector<int>& bots) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(path + ": cannot open training state");
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::istringstream is(bytes, std::ios::binary);

    // Every failure below (including BinaryReader's "unexpected end of
    // stream") is reported with the file name, so the user knows which file
    // is damaged.
    try {
        BinaryReader r(is);
        if (r.u32() != kStateMagic) throw std::runtime_error("bad magic (not a training state file)");
        uint32_t version = r.u32();
        if (version < 1 || version > kStateVersion)
            throw std::runtime_error("unsupported version " + std::to_string(version));
        TrainingState state;
        state.generation = r.i32();
        state.bestGeneration = r.i32();
        state.samplesTrained = static_cast<long long>(r.u64());
        state.buffer = ReplayBuffer::load(r, version >= 3);
        uint32_t tensors = r.u32();
        state.velocity.resize(tensors);
        for (std::vector<float>& v : state.velocity) {
            uint64_t size = r.u64();
            // A corrupt size must not trigger a huge allocation.
            if (size > bytes.size() / sizeof(float)) throw std::runtime_error("truncated velocity tensor");
            v.resize(static_cast<size_t>(size));
            r.floats(v.data(), v.size());
        }
        state.results = ResultsTable::fromCsv(r.str());
        if (version == 1) {
            state.botSimulations = bots;
            state.references = initialReferences(static_cast<int>(bots.size()));
        } else {
            uint32_t count = r.u32();
            if (count > bytes.size() / sizeof(int32_t)) throw std::runtime_error("truncated reference set");
            state.references.resize(count);
            for (int& ref : state.references) ref = r.i32();
            // Sorted, unique and anchored, or the file is not what we wrote.
            bool valid = std::find(state.references.begin(), state.references.end(), 0) !=
                         state.references.end();
            for (size_t i = 1; valid && i < state.references.size(); ++i)
                valid = state.references[i - 1] < state.references[i];
            if (!valid) throw std::runtime_error("invalid reference set (must be ascending and contain 0)");
            uint32_t numBots = r.u32();
            if (numBots > bytes.size() / sizeof(int32_t)) throw std::runtime_error("truncated bot list");
            state.botSimulations.resize(numBots);
            for (int& b : state.botSimulations) b = r.i32();
            // Bot k is player -k, so no reference may name a bot that is not stored.
            if (state.references.front() < -static_cast<long long>(numBots))
                throw std::runtime_error("invalid reference set (names a bot missing from the bot list)");
        }
        if (is.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("unexpected trailing bytes");
        return state;
    } catch (const std::exception& e) {
        throw std::runtime_error(path + ": " + e.what());
    }
}

std::string checkpointPath(const std::string& runDir, int generation) {
    char name[32];
    std::snprintf(name, sizeof name, "gen_%04d.azn", generation);   // 7 -> gen_0007.azn
    return (std::filesystem::path(runDir) / "checkpoints" / name).string();
}

std::string statePath(const std::string& runDir) {
    return (std::filesystem::path(runDir) / "state.bin").string();
}

std::string configPath(const std::string& runDir) {
    return (std::filesystem::path(runDir) / "config.cfg").string();
}

} // namespace azgo
