// The pieces both kinds of slice file share (self-play's in
// selfplay_slice.cpp, the arena's in arena_slice.cpp): the FNV-1a trailer
// every job and result ends with, little-endian reads of a file's header,
// and what a network's file name in a job may be. Internal to the library.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include "azgo/serialize.hpp"
#include "azgo/selfplay_slice.hpp"

namespace azgo::slice_files {

// Appends the FNV-1a checksum of `bytes` to them, little-endian.
inline void putChecksum(std::string& bytes) {
    const uint64_t h = fnv1a64(bytes);
    for (int i = 0; i < 8; ++i) bytes += static_cast<char>((h >> (8 * i)) & 0xff);
}

// A little-endian unsigned integer of `size` bytes at `at`.
inline uint64_t storedUint(const std::string& bytes, size_t at, int size) {
    uint64_t v = 0;
    for (int i = 0; i < size; ++i)
        v |= static_cast<uint64_t>(static_cast<unsigned char>(bytes[at + i])) << (8 * i);
    return v;
}

// The checksum stored in a file's last 8 bytes.
inline uint64_t storedChecksum(const std::string& bytes) { return storedUint(bytes, bytes.size() - 8, 8); }

// A network file name in the work directory: no path, nothing to quote.
inline bool plainFileName(const std::string& name) {
    if (name.empty() || name.size() > 128 || name[0] == '.') return false;
    for (char c : name)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
              c == '_' || c == '.' || c == '-'))
            return false;
    return true;
}

// An engine fingerprint a result may carry: not empty, at most 1024
// characters, none of them a control character.
inline bool printableFingerprint(const std::string& f) {
    bool ok = !f.empty() && f.size() <= 1024;
    for (char c : f) ok = ok && static_cast<unsigned char>(c) >= 0x20;
    return ok;
}

// One game as a self-play result file and a stream carry it:
// writeGameRecord's layout, u8 resignDisabled, and per side u32 n and n x
// (i32 ply, f32 value) of resign lows. readSliceRecord throws SliceError
// (or std::runtime_error, truncation) naming `game`; it does not verify.
void writeSliceRecord(BinaryWriter& w, const GameRecord& g);
GameRecord readSliceRecord(BinaryReader& r, const std::string& game);

// A self-play job file and its network, checked as the selfplay app checks
// them: the job parses, was written by this build, and names a network in
// its directory whose hash and board size match (SliceError otherwise).
struct LoadedSliceJob {
    std::string bytes;
    SelfPlayJob job;
    std::unique_ptr<Network> net;
};
LoadedSliceJob loadSliceJob(const std::string& jobPath);

} // namespace azgo::slice_files
