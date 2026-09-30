#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "cli.hpp"

static bool near(double a, double b, double tol = 1e-12) { return std::fabs(a - b) <= tol; }

static const std::vector<std::string> kValues = {"games", "komi", "run", "set", "seconds"};
static const std::vector<std::string> kFlags = {"resume", "help"};

// Builds a real argv (program name first) and parses it.
static ArgParser parse(std::vector<std::string> args) {
    args.insert(args.begin(), "prog");
    std::vector<char*> argv;
    for (std::string& a : args) argv.push_back(&a[0]);
    return ArgParser(static_cast<int>(argv.size()), argv.data(), kValues, kFlags);
}

// True when parsing (or the callable) throws std::invalid_argument whose
// message mentions `needle`.
template <typename F>
static bool throwsMentioning(F f, const std::string& needle) {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

static void test_values_and_defaults() {
    ArgParser p = parse({"--games", "40", "--komi", "-6.5", "--run", "runs/a"});
    assert(p.has("games") && p.has("komi") && p.has("run"));
    assert(!p.has("seconds"));
    assert(p.getInt("games", 0) == 40);
    assert(near(p.getDouble("komi", 7.5), -6.5));     // negative numbers are values
    assert(p.get("run") == "runs/a");
    assert(p.get("seconds", "2") == "2");               // absent: default
    assert(p.getInt("seconds", 3) == 3);
    assert(near(p.getDouble("seconds", 2.0), 2.0));
    assert(throwsMentioning([&] { p.getInt("komi", 0); }, "--komi"));   // "-6.5" is not an int
}

static void test_flags() {
    ArgParser p = parse({"--run", "r", "--resume"});
    assert(p.flag("resume"));
    assert(!p.flag("help"));
    assert(!p.has("resume"));                           // flags are not values
    ArgParser q = parse({});
    assert(!q.flag("resume") && !q.has("run") && q.sets().empty());
}

static void test_repeated_set_keeps_order() {
    ArgParser p = parse({"--set", "blocks=1", "--run", "r", "--set", "filters=8",
                         "--set", "komi=-3"});
    assert(p.sets().size() == 3);
    assert(p.sets()[0] == "blocks=1");
    assert(p.sets()[1] == "filters=8");
    assert(p.sets()[2] == "komi=-3");                   // '-' inside the value is fine
    assert(p.has("set"));
}

static void test_unknown_option_and_positional_throw() {
    assert(throwsMentioning([] { parse({"--simulation", "800"}); }, "--simulation"));
    assert(throwsMentioning([] { parse({"runs/a"}); }, "runs/a"));
    assert(throwsMentioning([] { parse({"-g", "4"}); }, "-g"));
}

static void test_missing_value_throws() {
    // At the end of the command line.
    assert(throwsMentioning([] { parse({"--games"}); }, "--games"));
    // Followed by another option: "--resume" is not a value.
    assert(throwsMentioning([] { parse({"--run", "--resume"}); }, "--run"));
    // A value option given twice is a mistake (only --set repeats).
    assert(throwsMentioning([] { parse({"--games", "4", "--games", "6"}); }, "--games"));
}

static void test_malformed_numbers_throw() {
    ArgParser p = parse({"--games", "12x", "--komi", "7.5.1", "--seconds", ""});
    assert(throwsMentioning([&] { p.getInt("games", 0); }, "12x"));
    assert(throwsMentioning([&] { p.getDouble("komi", 0); }, "--komi"));
    assert(throwsMentioning([&] { p.getInt("seconds", 0); }, "--seconds"));
    ArgParser q = parse({"--games", "99999999999"});
    assert(throwsMentioning([&] { q.getInt("games", 0); }, "--games"));   // out of int range
}

int main() {
    test_values_and_defaults();
    test_flags();
    test_repeated_set_keeps_order();
    test_unknown_option_and_positional_throw();
    test_missing_value_throws();
    test_malformed_numbers_throw();
    std::printf("test_cli: all tests passed\n");
    return 0;
}
