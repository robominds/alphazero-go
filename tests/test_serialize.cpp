#include <unistd.h>   // getpid
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/serialize.hpp"

using namespace azgo;
namespace fs = std::filesystem;

namespace {

std::vector<unsigned char> bytesOf(const std::string& s) { return {s.begin(), s.end()}; }

std::string readFile(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

template <typename Read>
bool throwsEndOfData(const std::string& data, Read read) {
    std::istringstream in(data, std::ios::binary);
    BinaryReader r(in);
    try {
        read(r);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

} // namespace

static void test_round_trip_every_type() {
    std::ostringstream out(std::ios::binary);
    BinaryWriter w(out);
    const float tensor[5] = {0.0f, -1.5f, 3.25e-8f, 1e30f, std::numeric_limits<float>::infinity()};
    const unsigned char raw[3] = {0x00, 0xFF, 0x7F};
    const std::string withNul("a\0b", 3);
    w.u8(0xAB);
    w.u32(0xDEADBEEFu);
    w.u64(0x0123456789ABCDEFull);
    w.i32(-123456);
    w.f32(-0.1f);
    w.f64(3.141592653589793);
    w.str("hello");
    w.str("");
    w.str(withNul);
    w.floats(tensor, 5);
    w.bytes(raw, 3);

    std::istringstream in(out.str(), std::ios::binary);
    BinaryReader r(in);
    assert(r.u8() == 0xAB);
    assert(r.u32() == 0xDEADBEEFu);
    assert(r.u64() == 0x0123456789ABCDEFull);
    assert(r.i32() == -123456);
    assert(r.f32() == -0.1f);
    assert(r.f64() == 3.141592653589793);
    assert(r.str() == "hello");
    assert(r.str().empty());
    assert(r.str() == withNul);
    float back[5];
    r.floats(back, 5);
    for (int i = 0; i < 5; ++i) assert(back[i] == tensor[i]);
    unsigned char rawBack[3];
    r.bytes(rawBack, 3);
    assert(rawBack[0] == 0x00 && rawBack[1] == 0xFF && rawBack[2] == 0x7F);
    assert(in.peek() == std::char_traits<char>::eof());   // nothing extra was written
}

static void test_exact_byte_layout() {
    std::ostringstream out(std::ios::binary);
    BinaryWriter w(out);
    w.u32(0x01020304u);
    w.f32(1.0f);            // IEEE-754 0x3F800000
    w.i32(-1);
    w.str("ab");
    w.u64(1);
    const std::vector<unsigned char> expected = {
        0x04, 0x03, 0x02, 0x01,
        0x00, 0x00, 0x80, 0x3F,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x02, 0x00, 0x00, 0x00, 'a', 'b',
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    assert(bytesOf(out.str()) == expected);
}

static void test_reading_past_end_throws() {
    assert(throwsEndOfData("", [](BinaryReader& r) { r.u8(); }));
    assert(throwsEndOfData("abc", [](BinaryReader& r) { r.u32(); }));
    assert(throwsEndOfData("1234567", [](BinaryReader& r) { r.u64(); }));
    assert(throwsEndOfData(std::string("\x0A\0\0\0xy", 6), [](BinaryReader& r) { r.str(); }));   // claims 10 bytes
    assert(throwsEndOfData(std::string("\xFF\xFF\xFF\xFF", 4), [](BinaryReader& r) { r.str(); }));  // 4 GB claim
    assert(throwsEndOfData(std::string(8, '\0'), [](BinaryReader& r) {
        float f[3];
        r.floats(f, 3);
    }));
    // Exactly enough data is not an error.
    assert(!throwsEndOfData("abcd", [](BinaryReader& r) { r.u32(); }));
}

static void test_write_file_atomically() {
    const fs::path dir = fs::temp_directory_path() / ("azgo_test_serialize_" + std::to_string(getpid()));
    fs::remove_all(dir);
    assert(fs::create_directory(dir));
    const fs::path file = dir / "data.bin";
    const fs::path tmp = dir / "data.bin.tmp";

    writeFileAtomically(file.string(), "first version, longer");
    assert(readFile(file) == "first version, longer");
    writeFileAtomically(file.string(), std::string("2nd\0", 4));
    assert(readFile(file) == std::string("2nd\0", 4));   // replaced, not merged
    assert(!fs::exists(tmp));

    // A leftover temp file from a crash is simply overwritten.
    { std::ofstream(tmp, std::ios::binary) << "garbage garbage garbage"; }
    writeFileAtomically(file.string(), "third");
    assert(readFile(file) == "third");
    assert(!fs::exists(tmp));

    // Unwritable destination: a directory that does not exist.
    bool threw = false;
    const std::string bad = (dir / "missing" / "x.bin").string();
    try {
        writeFileAtomically(bad, "x");
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find(bad) != std::string::npos;   // names the path
    }
    assert(threw);
    assert(!fs::exists(dir / "missing"));
    fs::remove_all(dir);
}

int main() {
    test_round_trip_every_type();
    test_exact_byte_layout();
    test_reading_past_end_throws();
    test_write_file_atomically();
    std::printf("test_serialize: all tests passed\n");
    return 0;
}
