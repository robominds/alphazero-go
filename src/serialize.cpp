#include "azgo/serialize.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace azgo {

namespace {

// Bulk transfers go through a fixed-size buffer: fast for million-float
// tensors, and a corrupt length field fails at end of data instead of
// allocating gigabytes first.
constexpr size_t kChunkFloats = 4096;

void storeU32(unsigned char* out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out[i] = static_cast<unsigned char>(v >> (8 * i));
}

uint32_t loadU32(const unsigned char* in) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(in[i]) << (8 * i);
    return v;
}

} // namespace

// ---------------------------------------------------------------- BinaryWriter

BinaryWriter::BinaryWriter(std::ostream& os) : os_(os) {}

void BinaryWriter::bytes(const void* p, size_t n) {
    os_.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
}

void BinaryWriter::u8(uint8_t v) { bytes(&v, 1); }

void BinaryWriter::u32(uint32_t v) {
    unsigned char b[4];
    storeU32(b, v);
    bytes(b, 4);
}

void BinaryWriter::u64(uint64_t v) {
    unsigned char b[8];
    for (int i = 0; i < 8; ++i) b[i] = static_cast<unsigned char>(v >> (8 * i));
    bytes(b, 8);
}

void BinaryWriter::i32(int32_t v) { u32(static_cast<uint32_t>(v)); }   // two's complement bits

void BinaryWriter::f32(float v) {
    // memcpy reinterprets the IEEE-754 bits without undefined behavior.
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    u32(bits);
}

void BinaryWriter::f64(double v) {
    uint64_t bits;
    std::memcpy(&bits, &v, 8);
    u64(bits);
}

void BinaryWriter::str(const std::string& s) {
    u32(static_cast<uint32_t>(s.size()));
    bytes(s.data(), s.size());
}

void BinaryWriter::floats(const float* p, size_t n) {
    unsigned char buf[kChunkFloats * 4];
    while (n > 0) {
        const size_t count = std::min(n, kChunkFloats);
        for (size_t i = 0; i < count; ++i) {
            uint32_t bits;
            std::memcpy(&bits, p + i, 4);
            storeU32(buf + 4 * i, bits);
        }
        bytes(buf, 4 * count);
        p += count;
        n -= count;
    }
}

// ---------------------------------------------------------------- BinaryReader

BinaryReader::BinaryReader(std::istream& is) : is_(is) {}

void BinaryReader::bytes(void* p, size_t n) {
    is_.read(static_cast<char*>(p), static_cast<std::streamsize>(n));
    if (static_cast<size_t>(is_.gcount()) != n) throw std::runtime_error("unexpected end of data");
}

uint8_t BinaryReader::u8() {
    uint8_t v;
    bytes(&v, 1);
    return v;
}

uint32_t BinaryReader::u32() {
    unsigned char b[4];
    bytes(b, 4);
    return loadU32(b);
}

uint64_t BinaryReader::u64() {
    unsigned char b[8];
    bytes(b, 8);
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(b[i]) << (8 * i);
    return v;
}

int32_t BinaryReader::i32() { return static_cast<int32_t>(u32()); }

float BinaryReader::f32() {
    const uint32_t bits = u32();
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

double BinaryReader::f64() {
    const uint64_t bits = u64();
    double v;
    std::memcpy(&v, &bits, 8);
    return v;
}

std::string BinaryReader::str() {
    const uint32_t length = u32();
    std::string s;
    char buf[4096];
    for (uint32_t done = 0; done < length;) {
        const uint32_t count = std::min<uint32_t>(length - done, sizeof buf);
        bytes(buf, count);
        s.append(buf, count);
        done += count;
    }
    return s;
}

void BinaryReader::floats(float* p, size_t n) {
    unsigned char buf[kChunkFloats * 4];
    while (n > 0) {
        const size_t count = std::min(n, kChunkFloats);
        bytes(buf, 4 * count);
        for (size_t i = 0; i < count; ++i) {
            const uint32_t bits = loadU32(buf + 4 * i);
            std::memcpy(p + i, &bits, 4);
        }
        p += count;
        n -= count;
    }
}

// ---------------------------------------------------------------- atomic write

void writeFileAtomically(const std::string& path, const std::string& contents) {
    const std::string tmp = path + ".tmp";
    {
        // Truncates any leftover .tmp from an earlier crash.
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open " + tmp + " for writing");
        out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        out.flush();
        if (!out) throw std::runtime_error("error writing " + tmp);
        out.close();
        if (out.fail()) throw std::runtime_error("error closing " + tmp);
    }
    // rename() replaces the destination in one step on POSIX file systems:
    // readers see the old file or the new one, never a partial write.
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) throw std::runtime_error("cannot rename " + tmp + " to " + path + ": " + ec.message());
}

} // namespace azgo
