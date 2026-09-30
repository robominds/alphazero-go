// Binary files that read back identically on any machine. Multi-byte values
// are written least significant byte first (little-endian) by explicit
// shifts, never by dumping memory, so the format does not depend on the
// host's byte order. Files are replaced atomically: write a temporary file,
// then rename it over the old one, so a crash mid-write leaves either the
// old file or the new one, never half of each.
#pragma once
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>

namespace azgo {

// Little-endian binary I/O, independent of host byte order.
class BinaryWriter {
public:
    explicit BinaryWriter(std::ostream& os);
    void u8(uint8_t v);
    void u32(uint32_t v);
    void u64(uint64_t v);
    void i32(int32_t v);
    void f32(float v);
    void f64(double v);
    void str(const std::string& s);           // u32 length + bytes
    void floats(const float* p, size_t n);
    void bytes(const void* p, size_t n);
private:
    std::ostream& os_;
};

// Every read throws std::runtime_error on premature end of stream.
class BinaryReader {
public:
    explicit BinaryReader(std::istream& is);
    uint8_t u8();
    uint32_t u32();
    uint64_t u64();
    int32_t i32();
    float f32();
    double f64();
    std::string str();
    void floats(float* p, size_t n);
    void bytes(void* p, size_t n);
private:
    std::istream& is_;
};

// Writes to path + ".tmp", fsyncs it, renames over path (atomic replace), then
// fsyncs the parent directory, so the result is durable across power loss
// (the file is never empty or partial afterwards). On macOS the file sync uses
// F_FULLFSYNC. Throws std::runtime_error naming the path on I/O failure; the
// .tmp is removed best-effort.
void writeFileAtomically(const std::string& path, const std::string& contents);

} // namespace azgo
