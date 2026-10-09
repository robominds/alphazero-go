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
// 2 added the reference set and bot list, 3 the buffer's ownership targets,
// 4 the reference networks, 5 resignation (the buffer's resigned flags, the
// threshold and the window of disabled games).
constexpr uint32_t kStateVersion = 5;
} // namespace

std::vector<int> initialReferences(int numFixed) {
    if (numFixed < 0) throw std::invalid_argument("initialReferences: numFixed must be >= 0");
    std::vector<int> refs;
    for (int k = numFixed; k >= 1; --k) refs.push_back(-k);   // ascending: -numFixed first
    refs.push_back(0);
    return refs;
}

int trainingStateGeneration(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(path + ": cannot open training state");
    try {
        BinaryReader r(in);
        if (r.u32() != kStateMagic) throw std::runtime_error("bad magic (not a training state file)");
        const uint32_t version = r.u32();
        if (version < 1 || version > kStateVersion)
            throw std::runtime_error("unsupported version " + std::to_string(version));
        return r.i32();
    } catch (const std::exception& e) {
        throw std::runtime_error(path + ": " + e.what());
    }
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
    w.u32(static_cast<uint32_t>(state.referenceNetworks.size()));
    for (const ReferenceNetwork& n : state.referenceNetworks) {
        w.str(n.path);
        w.u64(n.hash);
    }
    w.f32(state.resignThreshold);
    w.u32(static_cast<uint32_t>(state.resignWindow.size()));
    for (const ResignTrace& t : state.resignWindow)
        for (const ResignLows* lows : {&t.winnerLows, &t.loserLows}) {
            w.u32(static_cast<uint32_t>(lows->size()));
            for (const auto& [ply, value] : *lows) {
                w.u32(static_cast<uint32_t>(ply));
                w.f32(value);
            }
        }
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
        state.buffer = ReplayBuffer::load(r, version >= 3, version >= 5);
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
            if (version >= 4) {
                uint32_t numNetworks = r.u32();
                // Each entry is at least a u32 length and a u64 hash.
                if (numNetworks > bytes.size() / 12) throw std::runtime_error("truncated reference network list");
                state.referenceNetworks.resize(numNetworks);
                for (ReferenceNetwork& n : state.referenceNetworks) {
                    n.path = r.str();
                    n.hash = r.u64();
                }
            }
            // The fixed players are -1 ... -(bots + reference networks), so
            // no reference may name one that is not stored.
            if (state.references.front() <
                -static_cast<long long>(numBots + state.referenceNetworks.size()))
                throw std::runtime_error("invalid reference set (names a fixed player missing from "
                                         "the bot and reference network lists)");
            if (version >= 5) {
                state.resignThreshold = r.f32();
                // A threshold the config could not have produced means a damaged file.
                if (!(state.resignThreshold >= -1.0f && state.resignThreshold <= 0.0f))
                    throw std::runtime_error("invalid resignation threshold");
                uint32_t games = r.u32();
                // Each game is at least two u32 counts.
                if (games > bytes.size() / 8) throw std::runtime_error("truncated resignation window");
                state.resignWindow.resize(games);
                for (ResignTrace& t : state.resignWindow)
                    for (ResignLows* lows : {&t.winnerLows, &t.loserLows}) {
                        uint32_t count = r.u32();
                        if (count > bytes.size() / 8) throw std::runtime_error("truncated resignation window");
                        lows->resize(count);
                        for (size_t i = 0; i < lows->size(); ++i) {
                            auto& [ply, value] = (*lows)[i];
                            ply = static_cast<int>(r.u32());
                            value = r.f32();
                            // Record lows, as self-play writes them: from ply
                            // 1 on, plies rising, values in [-1, 1] and falling.
                            const bool ok = ply >= 1 && ply <= 1000000 && value >= -1.0f && value <= 1.0f &&
                                            (i == 0 || (ply > (*lows)[i - 1].first && value < (*lows)[i - 1].second));
                            if (!ok) throw std::runtime_error("invalid resignation window entry");
                        }
                    }
            }
        }
        if (is.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("unexpected trailing bytes");
        return state;
    } catch (const std::exception& e) {
        throw std::runtime_error(path + ": " + e.what());
    }
}

uint64_t fileHash(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(path + ": cannot open");
    uint64_t hash = 0xcbf29ce484222325ull;   // FNV-1a: offset basis, then xor and multiply per byte
    char buf[1 << 16];
    while (in.read(buf, sizeof buf) || in.gcount() > 0) {
        for (std::streamsize i = 0; i < in.gcount(); ++i) {
            hash ^= static_cast<unsigned char>(buf[i]);
            hash *= 0x100000001b3ull;
        }
    }
    if (in.bad()) throw std::runtime_error(path + ": read error");
    return hash;
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
