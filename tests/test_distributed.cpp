// Distributed self-play, tested on one machine: the build identity every
// binary carries, so the coordinator can tell whether two binaries run the
// same code; the selfplay_engines list and how a generation's games are
// assigned to its engines; the machine-local hosts file and the worker rules.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/build_info.hpp"
#include "azgo/config.hpp"
#include "azgo/engines.hpp"
#include "azgo/metal.hpp"
#include "azgo/rng.hpp"

using namespace azgo;

static bool isLowerHex(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}

// buildId() is "unknown", or the 40-hex commit, optionally followed by "+"
// and 8 hex of the uncommitted changes; shortBuildId() keeps 7 hex and the
// suffix. Both are compiled in, so they never change while a program runs.
static void test_build_id_shape() {
    const std::string id = buildId(), shortId = shortBuildId();
    assert(id == buildId() && shortId == shortBuildId());
    if (id == "unknown") {
        assert(shortId == "unknown");
        return;
    }
    const std::string commit = id.substr(0, 40);
    assert(commit.size() == 40 && isLowerHex(commit));
    const std::string suffix = id.substr(40);
    assert(suffix.empty() ||
           (suffix.size() == 9 && suffix[0] == '+' && isLowerHex(suffix.substr(1))));
    assert(shortId == id.substr(0, 7) + suffix);
}

// The strongest backend the binary was built with.
static void test_build_options() {
    const std::string options = buildOptions();
    if (metalBuilt()) assert(options == "metal");
    else if (blasAvailable()) assert(options == "blas");
    else assert(options == "naive");
}


// ---- the selfplay_engines list ------------------------------------------------

// The spec's example (docs/superpowers/specs/2026-10-18-distributed-selfplay-design.md).
static const char* const kSpecExample =
    "local/metal:50, local/blas=12, mini1/metal:10, mini1/blas=8, mini2/metal:10, mini2/blas=8";

static std::string parseError(const std::string& text) {
    try {
        parseEngineList(text);
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "accepted";
}

static bool contains(const std::string& s, const std::string& part) {
    return s.find(part) != std::string::npos;
}

static void test_engine_list_parse_print() {
    const std::string example = kSpecExample;
    const std::vector<EngineSpec> list = parseEngineList(example);
    assert(list.size() == 6);
    assert(list[0].machine == "local" && list[0].backend == "metal" && list[0].weight == 50 &&
           list[0].games == -1 && list[0].name() == "local/metal");
    assert(list[1].weight == 0 && list[1].games == 12);
    assert(list[4].name() == "mini2/metal");
    assert(printEngineList(list) == example);
    // Whitespace around any separator; the canonical form drops it.
    const std::string spaced = " local / metal : 50 ,local/blas= 12,\tmini1/metal:10 ";
    assert(printEngineList(parseEngineList(spaced)) ==
           "local/metal:50, local/blas=12, mini1/metal:10");
    assert(parseEngineList("").empty() && parseEngineList("   ").empty());
    assert(printEngineList({}).empty());
    // A repeated engine is two items, each with its own slice.
    const std::vector<EngineSpec> twice = parseEngineList("local/naive:1,local/naive:1");
    assert(twice.size() == 2 && twice[0].name() == twice[1].name());
    // Aliases: [a-z][a-z0-9_-]{0,31}.
    assert(parseEngineList("a_1-b/naive=0").size() == 1);
    assert(parseEngineList(std::string(32, 'm') + "/naive:1").size() == 1);
    assert(contains(parseError(std::string(33, 'm') + "/naive:1"), "alias"));
    assert(contains(parseError("Mini1/metal:1"), "alias"));
    assert(contains(parseError("1mini/metal:1"), "alias"));
    assert(contains(parseError("/metal:1"), "alias"));
    // Refusals name the item and the reason.
    assert(contains(parseError("local/metal:1, local/gpu:1"), "engine 2"));
    assert(contains(parseError("local/gpu:1"), "naive|blas|metal"));
    assert(contains(parseError("local/metal:0"), "weight"));
    assert(contains(parseError("local/metal:1000001"), "weight"));
    assert(contains(parseError("local/blas=-1"), "games"));
    assert(contains(parseError("local/blas=1000001"), "games"));
    assert(contains(parseError("local/blas=99999999999999999999"), "games"));
    assert(contains(parseError("local/blas=1x"), "games"));
    assert(contains(parseError("local/metal"), "expected"));
    assert(contains(parseError("local:3"), "expected"));
    assert(contains(parseError("local/metal:1,"), "engine 2"));
    assert(contains(parseError(",local/metal:1"), "engine 1"));
    assert(contains(parseError("local/metal:1:2"), "weight"));
    assert(parseError("local/blas=0") == "accepted");
}

// ---- assignGames ------------------------------------------------------------

static std::vector<int> counts(const std::vector<EngineSpec>& engines, int games) {
    std::vector<int> out(engines.size(), 0);
    for (const Slice& s : assignGames(engines, games)) out[static_cast<size_t>(s.engine)] = s.count;
    return out;
}

// The spec's worked example: the fixed counts take 28 games, the weights
// 50:10:10 share the other 72 as 51.4, 10.3 and 10.3, and the leftover game
// goes to the largest remainder.
static void test_assign_largest_remainder() {
    const std::vector<EngineSpec> example = parseEngineList(kSpecExample);
    assert((counts(example, 100) == std::vector<int>{52, 12, 10, 8, 10, 8}));
    const std::vector<Slice> slices = assignGames(example, 100);
    const int firsts[] = {0, 52, 64, 74, 82, 92};
    assert(slices.size() == 6);
    for (int e = 0; e < 6; ++e) assert(slices[e].engine == e && slices[e].first == firsts[e]);
    // Equal remainders: the earlier item wins.
    assert((counts(parseEngineList("a/naive:1, b/naive:1, c/naive:1"), 10) ==
            std::vector<int>{4, 3, 3}));
    assert((counts(parseEngineList("a/naive:50, b/naive:10, c/naive:10"), 72) ==
            std::vector<int>{52, 10, 10}));
    // An engine left with no game gets no slice.
    const std::vector<Slice> few =
        assignGames(parseEngineList("a/naive:10, b/naive:1, c/blas=2"), 3);
    assert(few.size() == 2 && few[0].engine == 0 && few[0].count == 1 && few[1].engine == 2 &&
           few[1].first == 1 && few[1].count == 2);
}

static void test_assign_fixed_only() {
    const std::vector<EngineSpec> fixed = parseEngineList("a/blas=6, b/naive=0, c/metal=4");
    const std::vector<Slice> slices = assignGames(fixed, 10);
    assert(slices.size() == 2);
    assert(slices[0].engine == 0 && slices[0].first == 0 && slices[0].count == 6);
    assert(slices[1].engine == 2 && slices[1].first == 6 && slices[1].count == 4);
    // Lists checkConsistency refuses are refused here too.
    for (int games : {9, 11}) {
        bool threw = false;
        try {
            assignGames(fixed, games);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
}

// Random valid lists: the slices cover [0, G) in list order without a gap
// or an overlap, engines with no game are absent, a weighted count is
// within one game of its exact quota R * w / W, and a second call agrees.
static void test_assign_properties() {
    Rng rng(20261018);
    int checked = 0;
    while (checked < 2000) {
        const int engines = 1 + rng.below(8);
        const int games = 1 + rng.below(500);
        std::vector<EngineSpec> list;
        long long fixed = 0, weights = 0;
        for (int e = 0; e < engines; ++e) {
            EngineSpec spec;
            spec.machine = "m" + std::to_string(e);
            spec.backend = "naive";
            if (rng.below(2) == 0) {
                spec.games = rng.below(21);
                fixed += spec.games;
            } else {
                spec.weight = 1 + rng.below(1000);
                weights += spec.weight;
            }
            list.push_back(spec);
        }
        // A list the config would refuse.
        if (fixed > games || (weights == 0 && fixed != games)) continue;
        ++checked;
        const std::vector<Slice> slices = assignGames(list, games);
        assert(slices.size() == assignGames(list, games).size());
        int next = 0, lastEngine = -1;
        std::vector<int> got(list.size(), 0);
        for (const Slice& s : slices) {
            assert(s.count > 0 && s.first == next && s.engine > lastEngine);
            next += s.count;
            lastEngine = s.engine;
            got[static_cast<size_t>(s.engine)] = s.count;
        }
        assert(next == games);
        const std::vector<Slice> again = assignGames(list, games);
        for (size_t k = 0; k < slices.size(); ++k)
            assert(again[k].engine == slices[k].engine && again[k].first == slices[k].first &&
                   again[k].count == slices[k].count);
        const long long rest = games - fixed;
        for (size_t e = 0; e < list.size(); ++e) {
            if (list[e].games >= 0) {
                assert(got[e] == list[e].games);
            } else {
                // |count - R w / W| < 1, in integers: |count W - R w| < W.
                const long long diff = got[e] * weights - rest * list[e].weight;
                assert(diff > -weights && diff < weights);
            }
        }
    }
}

// ---- the hosts file -----------------------------------------------------------

static const char* const kHostsExample =
    "# hosts -- machine-local; never commit this file.\n"
    "# alias  key=value ...   (values: no spaces or shell metacharacters)\n"
    "local  cpu_workers=6 cpu_qos=utility\n"
    "\n"
    "mini1  ssh=u@h1 exe=~/azgo/build-metal/selfplay workdir=~/azgo/runs/remote cpu_workers=8\n"
    "mini2  ssh=u@h2 exe=/opt/azgo/selfplay workdir=/tmp/w  # a comment after the values\n"
    "fake   exe=build/selfplay workdir=/tmp/fake metal_workers=64 inference_batch=128 nice=5 "
    "timeout=0.5 cpu_qos=default\n";

static void test_hosts_parse() {
    const HostsFile hosts = parseHostsText(kHostsExample, "hosts");
    assert(hosts.machines.size() == 4);
    const MachineSettings local = hosts.settings("local");
    assert(local.local() && local.cpuWorkers == 6 && local.cpuQos == CpuQos::Utility);
    assert(local.ssh.empty());
    const MachineSettings mini1 = hosts.settings("mini1");
    assert(!mini1.local() && mini1.alias == "mini1" && mini1.ssh == "u@h1");
    assert(mini1.exe == "~/azgo/build-metal/selfplay" && mini1.workdir == "~/azgo/runs/remote");
    assert(mini1.cpuWorkers == 8 && mini1.metalWorkers == 0 && mini1.inferenceBatch == 0);
    assert(mini1.nice == 0 && mini1.timeout == 0 && mini1.cpuQos == CpuQos::Default);
    const MachineSettings mini2 = hosts.settings("mini2");
    assert(mini2.exe == "/opt/azgo/selfplay" && mini2.workdir == "/tmp/w" && mini2.cpuWorkers == 0);
    const MachineSettings fake = hosts.settings("fake");
    assert(fake.ssh.empty() && !fake.local() && fake.metalWorkers == 64);
    assert(fake.inferenceBatch == 128);
    assert(fake.nice == 5 && fake.timeout == 0.5 && fake.cpuQos == CpuQos::Default);
    // local needs no line: its defaults (utility QoS).
    const HostsFile empty = parseHostsText("", "hosts");
    const MachineSettings defaults = empty.settings("local");
    assert(defaults.local() && defaults.cpuQos == CpuQos::Utility && defaults.cpuWorkers == 0);
    const HostsFile defaultQos = parseHostsText("local cpu_qos=default\n", "h");
    assert(defaultQos.settings("local").cpuQos == CpuQos::Default);
    // A remote that is not defined is an error naming it.
    bool threw = false;
    try {
        empty.settings("mini1");
    } catch (const ConfigError& e) {
        threw = contains(e.what(), "mini1");
    }
    assert(threw);
}

static std::string hostsError(const std::string& text) {
    try {
        parseHostsText(text, "h");
    } catch (const ConfigError& e) {
        return e.what();
    }
    return "accepted";
}

// Each refusal names the file and line: "h:2: <reason>".
static void test_hosts_refusals() {
    const std::string ok = "mini1 ssh=u@h exe=/a/selfplay workdir=/w\n";
    assert(hostsError(ok) == "accepted");
    assert(contains(hostsError("local cpu_workers=2\nmini1 colour=red exe=/a workdir=/w\n"),
                    "h:2: unknown key 'colour'"));
    assert(contains(hostsError(ok + "\n" + ok), "h:3: "));
    assert(contains(hostsError(ok + ok), "repeated"));
    assert(contains(hostsError("local ssh=u@h\n"), "h:1: "));
    assert(contains(hostsError("local ssh=u@h\n"), "ssh"));
    assert(contains(hostsError("mini1 ssh=u@h workdir=/w\n"), "h:1: "));
    assert(contains(hostsError("mini1 ssh=u@h workdir=/w\n"), "exe"));
    assert(contains(hostsError("mini1 ssh=u@h exe=/a\n"), "workdir"));
    assert(contains(hostsError("mini1 exe=/a workdir=/w cpu_workers=x\n"), "h:1: cpu_workers: "));
    assert(contains(hostsError("mini1 exe=/a workdir=/w cpu_workers=-1\n"), "cpu_workers"));
    assert(contains(hostsError("mini1 exe=/a workdir=/w nice=4.5\n"), "nice"));
    assert(contains(hostsError("mini1 exe=/a workdir=/w timeout=-1\n"), "timeout"));
    assert(contains(hostsError("mini1 exe=/a workdir=/w cpu_qos=low\n"), "cpu_qos"));
    assert(contains(hostsError("mini1 exe=/a workdir=/w cpu_workers=2 cpu_workers=3\n"),
                    "repeated"));
    assert(contains(hostsError("mini1 exe=/a b workdir=/w\n"), "h:1: "));   // a value with a space
    for (const char* bad : {";", "$", "`", "'", "|", "\"", "&", "(", "*", "\\"}) {
        const std::string msg =
            hostsError("mini1 ssh=u@h exe=/a" + std::string(bad) + "b workdir=/w\n");
        assert(contains(msg, "h:1: exe: ") && contains(msg, "character"));
    }
    assert(contains(hostsError("mini1 ssh= exe=/a workdir=/w\n"), "ssh"));
    assert(contains(hostsError("Mini1 exe=/a workdir=/w\n"), "h:1: "));
    assert(contains(hostsError("Mini1 exe=/a workdir=/w\n"), "alias"));
    assert(contains(hostsError("mini1 exe\n"), "key=value"));
}

// The hosts file's path: --hosts, else $AZGO_HOSTS, else ~/.config/azgo/hosts.
static void test_hosts_search_order() {
    const char* home = std::getenv("HOME");
    const std::string savedHome = home ? home : "";
    setenv("HOME", "/home/someone", 1);
    unsetenv("AZGO_HOSTS");
    assert(resolveHostsPath("") == "/home/someone/.config/azgo/hosts");
    setenv("AZGO_HOSTS", "/etc/azgo-hosts", 1);
    assert(resolveHostsPath("") == "/etc/azgo-hosts");
    assert(resolveHostsPath("my-hosts") == "my-hosts");
    setenv("AZGO_HOSTS", "", 1);
    assert(resolveHostsPath("") == "/home/someone/.config/azgo/hosts");
    unsetenv("AZGO_HOSTS");
    if (home) setenv("HOME", savedHome.c_str(), 1);
    else unsetenv("HOME");
    // A missing file is an empty one: only remotes need it.
    const HostsFile none = loadHostsFile("/nonexistent/azgo/hosts");
    assert(none.machines.empty() && none.settings("local").local());
}

// Worker threads per engine: a GPU engine plays every game of its slice at
// once (at most 256); a CPU engine keeps cores free for the desktop (a
// third of them locally) or for the machine's GPU engine (two, remotely).
// Never more workers than games.
static void test_engine_workers() {
    const HostsFile hosts = parseHostsText(kHostsExample, "hosts");
    MachineSettings local = parseHostsText("", "h").settings("local");
    MachineSettings remote = hosts.settings("mini2");
    assert(engineWorkers(local, "metal", 52, 18) == 52);
    assert(engineWorkers(remote, "metal", 300, 10) == 256);
    assert(engineWorkers(hosts.settings("fake"), "metal", 100, 10) == 64);
    assert(engineWorkers(hosts.settings("fake"), "metal", 10, 10) == 10);
    assert(engineWorkers(local, "blas", 12, 18) == 6);
    assert(engineWorkers(local, "naive", 12, 2) == 1);
    assert(engineWorkers(local, "blas", 4, 18) == 4);
    assert(engineWorkers(remote, "blas", 12, 10) == 8);
    assert(engineWorkers(remote, "blas", 12, 2) == 1);
    assert(engineWorkers(hosts.settings("local"), "blas", 12, 18) == 6);
    assert(engineWorkers(hosts.settings("mini1"), "naive", 20, 10) == 8);
    assert(engineWorkers(hosts.settings("mini1"), "naive", 3, 10) == 3);
}

int main() {
    test_build_id_shape();
    test_build_options();
    test_engine_list_parse_print();
    test_assign_largest_remainder();
    test_assign_fixed_only();
    test_assign_properties();
    test_hosts_parse();
    test_hosts_refusals();
    test_hosts_search_order();
    test_engine_workers();
    std::printf("test_distributed: OK\n");
    return 0;
}
