// Distributed self-play, tested on one machine: the build identity every
// binary carries, so the coordinator can tell whether two binaries run the
// same code; the selfplay_engines list and how a generation's games are
// assigned to its engines; the machine-local hosts file and the worker rules;
// engine fingerprints; the manifest of who played which slice.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/build_info.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/config.hpp"
#include "azgo/coordinator.hpp"
#include "azgo/engines.hpp"
#include "azgo/metal.hpp"
#include "azgo/network.hpp"
#include "azgo/rng.hpp"
#include "azgo/run_state.hpp"
#include "azgo/subprocess.hpp"
#include "azgo/selfplay_slice.hpp"

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
    // An item with neither a weight nor a count (or both) is a programming error.
    for (const auto& [weight, games] : {std::pair<int, int>{0, -1}, std::pair<int, int>{2, 3}}) {
        EngineSpec bad;
        bad.machine = "a";
        bad.backend = "naive";
        bad.weight = weight;
        bad.games = games;
        std::string msg;
        try {
            assignGames({bad}, 3);
        } catch (const std::invalid_argument& e) {
            msg = e.what();
        }
        assert(contains(msg, "a/naive needs exactly one of a weight (> 0) and a count (>= 0)"));
    }
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
    assert(contains(hostsError("mini1 exe=/a workdir=/w cpu_workers=+3\n"), "cpu_workers"));
    assert(contains(hostsError("mini1 exe=/a workdir=/w timeout=+1\n"), "timeout"));
    assert(contains(hostsError("mini1 exe=/a workdir=/w timeout=.5\n"), "timeout"));
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
    // A value that could be taken for an option.
    assert(contains(hostsError("mini1 ssh=-oProxyCommand=x exe=/a workdir=/w\n"), "h:1: ssh: "));
    assert(contains(hostsError("mini1 ssh=u@h exe=-x workdir=/w\n"), "h:1: exe: "));
    assert(contains(hostsError("mini1 ssh=u@h exe=/a workdir=-w\n"), "h:1: workdir: "));
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
// once (at most 256); a CPU engine keeps cores free for the desktop and
// for the machine's GPU engine: at most four workers (a third of the cores)
// locally, two on a remote, where BLAS saturates early.
// Never more workers than games.
static void test_engine_workers() {
    const HostsFile hosts = parseHostsText(kHostsExample, "hosts");
    MachineSettings local = parseHostsText("", "h").settings("local");
    MachineSettings remote = hosts.settings("mini2");
    assert(engineWorkers(local, "metal", 52, 18) == 52);
    assert(engineWorkers(remote, "metal", 300, 10) == 256);
    assert(engineWorkers(hosts.settings("fake"), "metal", 100, 10) == 64);
    assert(engineWorkers(hosts.settings("fake"), "metal", 10, 10) == 10);
    assert(engineWorkers(local, "blas", 12, 18) == 4);   // min(4, 18 / 3)
    assert(engineWorkers(local, "blas", 12, 10) == 3);   // min(4, 10 / 3)
    assert(engineWorkers(local, "naive", 12, 2) == 1);
    assert(engineWorkers(local, "blas", 4, 18) == 4);
    assert(engineWorkers(remote, "blas", 12, 10) == 2);
    assert(engineWorkers(remote, "blas", 12, 3) == 1);
    assert(engineWorkers(remote, "blas", 1, 10) == 1);
    assert(engineWorkers(hosts.settings("local"), "blas", 12, 18) == 6);
    assert(engineWorkers(hosts.settings("mini1"), "naive", 20, 10) == 8);
    assert(engineWorkers(hosts.settings("mini1"), "naive", 3, 10) == 3);
    // What selfplay uses on a remote: --workers, else the remote rule.
    assert(sliceWorkers("blas", 0, 12, 10) == 2);
    assert(sliceWorkers("naive", 0, 12, 3) == 1);
    assert(sliceWorkers("metal", 0, 12, 10) == 12);
    assert(sliceWorkers("metal", 0, 300, 10) == 256);
    assert(sliceWorkers("naive", 5, 3, 10) == 3);
    assert(sliceWorkers("blas", 7, 12, 10) == 7);
}

// ---- engine fingerprints ------------------------------------------------------

// What determines an engine's arithmetic: "<backend>; <cpu>; <os>", and for
// metal "; <metalDescription()>". No host name: fingerprints go into the
// manifest, which a run directory keeps.
static void test_fingerprint() {
    char host[256] = "";
    gethostname(host, sizeof host - 1);
    const std::string hostName = host;
    for (const std::string backend : {"naive", "blas", "metal"}) {
        bool available = true;
        std::string why;
        try {
            checkBackendAvailable(backend);
        } catch (const std::invalid_argument& e) {
            available = false;
            why = e.what();
        }
        if (!available) {
            bool threw = false;
            try {
                engineFingerprint(backend);
            } catch (const std::invalid_argument& e) {
                threw = std::string(e.what()) == why;
            }
            assert(threw);
            continue;
        }
        const std::string f = engineFingerprint(backend);
        std::printf("fingerprint %s\n", f.c_str());
        assert(f == engineFingerprint(backend));
        assert(f.rfind(backend + "; ", 0) == 0);
        assert(f.find('\n') == std::string::npos);
        // backend; cpu; os[; metal description]
        const size_t cpu = backend.size() + 2, os = f.find("; ", cpu);
        assert(os != std::string::npos && os > cpu);
        const size_t osEnd = f.find("; ", os + 2);
        const std::string osPart = f.substr(os + 2, osEnd == std::string::npos ? std::string::npos
                                                                                : osEnd - os - 2);
        assert(!osPart.empty());
#ifdef __APPLE__
        assert(osPart.rfind("macOS ", 0) == 0);
#endif
        const std::string head = backend == "metal" ? f.substr(0, osEnd) : f;
        assert(head.find(',') == std::string::npos);
        if (backend == "metal") {
            assert(osEnd != std::string::npos && f.substr(osEnd + 2) == metalDescription());
        } else {
            assert(osEnd == std::string::npos);
        }
        if (hostName.size() >= 4) assert(f.find(hostName) == std::string::npos);
    }
    bool threw = false;
    try {
        engineFingerprint("gpu");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

// ---- the manifest ---------------------------------------------------------------

static std::string readText(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static SliceReport report(int first, int count, const std::string& engine,
                          const std::string& note = "") {
    SliceReport r;
    r.slice = {0, first, count};
    r.planned = "fake/naive";
    r.engine = engine;
    r.fingerprint = "metal; Apple M4; macOS 26.1 (25B78); Apple M4, macOS 26.1, shaders b7b31a09";
    r.build = "09e008b";
    r.note = note;
    r.stats.seconds = 12.3;
    r.stats.positions = 300;
    r.stats.evaluations = 4567;
    return r;
}

// One row per slice per generation; rewriting generation g first drops
// every row of g and later, so a redone generation (after a crash between
// the manifest and state.bin) leaves exactly one set of rows.
static void test_manifest_rows() {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("azgo_test_manifest_" + std::to_string(getpid()));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string path = manifestPath(dir.string());
    assert(path == (dir / "selfplay-engines.csv").string());
    const std::string header =
        "generation,first,count,planned,engine,fingerprint,build,seconds,positions,evaluations,note\n";
    const std::string row1 = "1,0,3,fake/naive,fake/naive,\"metal; Apple M4; macOS 26.1 (25B78); "
                             "Apple M4, macOS 26.1, shaders b7b31a09\",09e008b,12.3,300,4567,\n";
    writeManifest(dir.string(), 1, {report(0, 3, "fake/naive")});
    assert(readText(path) == header + row1);
    writeManifest(dir.string(), 2,
                  {report(0, 3, "fake/naive"), report(3, 1, "local/naive", "fallback: a, b")});
    const std::string text = readText(path);
    assert(text.find(header + row1 + "2,0,3,") == 0);
    assert(text.find("\n2,3,1,fake/naive,local/naive,\"") != std::string::npos);
    const std::string noteEnd = ",\"fallback: a, b\"\n";
    assert(text.size() > noteEnd.size() && text.substr(text.size() - noteEnd.size()) == noteEnd);
    writeManifest(dir.string(), 3, {report(0, 4, "fake/naive")});
    // Generation 2 redone: rows of 2 and 3 go, the new ones come.
    writeManifest(dir.string(), 2, {report(0, 4, "fake/naive")});
    const std::string redone = readText(path);
    assert(redone.find(header + row1 + "2,0,4,") == 0);
    assert(std::count(redone.begin(), redone.end(), '\n') == 3);
    std::filesystem::remove_all(dir);
}

// One engine failing stops the others at their next evaluation, and play()
// throws that failure, not the others' cancellations. The metal engine
// fails as soon as its thread starts (a hosts-file inference_batch below
// mcts_batch, which preflight would refuse); the naive engine listed before
// it has a slice that would take minutes. Needs a usable Metal GPU.
static void test_failing_engine_stops_the_others() {
    if (!metalBuilt() || !metalAvailable()) return;
    Config cfg;
    cfg.boardSize = 5;
    cfg.history = 2;
    cfg.blocks = 1;
    cfg.filters = 4;
    cfg.valueHidden = 8;
    cfg.gamesPerGeneration = 2000;
    applyOverride(cfg, "selfplay_engines=local/naive=1999, local/metal:1");
    SelfPlayCoordinator coordinator(cfg, "unused",
                                    parseHostsText("local cpu_workers=1 inference_batch=4\n", "h"), nullptr);
    const Network net(cfg.netConfig(), 3);
    SelfPlayConfig sp;
    sp.mcts.simulations = 64;
    sp.mcts.batchSize = 8;
    const auto start = std::chrono::steady_clock::now();
    std::string error;
    try {
        coordinator.play(1, "unused", net, sp, 2000);
    } catch (const SliceCancelled&) {
        error = "cancelled";
    } catch (const std::invalid_argument& e) {
        error = e.what();
    }
    assert(contains(error, "do not fit batches of 4"));
    assert(std::chrono::steady_clock::now() - start < std::chrono::seconds(5));
}

// ---- job and result files --------------------------------------------------------

static std::string tempPath(const std::string& name) {
    return (std::filesystem::temp_directory_path() /
            ("azgo_test_slice_" + std::to_string(getpid()) + "_" + name))
        .string();
}

static void writeText(const std::string& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << bytes;
    assert(out);
}

static NetConfig sliceNet() {
    NetConfig c;
    c.boardSize = 5;
    c.history = 2;
    c.blocks = 1;
    c.filters = 4;
    c.valueHidden = 8;
    return c;
}

// A job for games [3, 9) of generation 2, resignation on with a threshold
// the tiny random network crosses, so the slice holds resigned, disabled
// and played-out games.
static SelfPlayJob sliceJob() {
    SelfPlayJob job;
    job.seed = 11;
    job.generation = 2;
    job.first = 3;
    job.count = 6;
    job.boardSize = 5;
    job.sp.mcts.simulations = 8;
    job.sp.mcts.batchSize = 4;
    job.sp.mcts.cPuct = 1.25f;
    job.sp.mcts.dirichletScale = 9.5f;
    job.sp.mcts.noiseFraction = 0.1f;
    job.sp.mcts.randomSymmetry = true;
    job.sp.temperatureMoves = 4;
    job.sp.komi = 7.5f;
    job.sp.maxMoves = 50;
    job.sp.resignation = true;
    job.sp.resignThreshold = -0.0f;
    job.sp.resignDisabledFraction = 0.5f;
    job.networkFile = "net_0123456789abcdef.azn";
    job.networkHash = 0x0123456789abcdefull;
    job.buildId = "0123456789abcdef0123456789abcdef01234567";
    job.inferenceBatch = 64;
    return job;
}

static SliceResult sliceResult(const SelfPlayJob& job) {
    const Network net(sliceNet(), 21);
    SliceResult r;
    r.jobHash = fnv1a64(serializeJob(job));
    r.buildId = job.buildId;
    r.fingerprint = "naive; test cpu; test os";
    r.records = playSelfPlayGames(net, job.sp, job.seed, job.generation, job.first, job.count,
                                  EngineRun{"naive", 2}, &r.stats);
    return r;
}

static void test_job_round_trip() {
    SelfPlayJob job = sliceJob();
    job.sp.mcts.addNoise = true;
    job.sp.resignThreshold = -0.87f;
    job.sp.mcts.noiseFraction = 0.1f;
    const std::string bytes = serializeJob(job);
    assert(bytes.compare(0, 4, "AZSJ") == 0);
    const SelfPlayJob back = parseJob(bytes);
    assert(back.seed == job.seed && back.generation == job.generation && back.first == job.first &&
           back.count == job.count && back.boardSize == job.boardSize);
    const MctsConfig& a = back.sp.mcts;
    const MctsConfig& b = job.sp.mcts;
    assert(a.simulations == b.simulations && a.batchSize == b.batchSize && a.addNoise == b.addNoise &&
           a.randomSymmetry == b.randomSymmetry);
    for (auto [x, y] : {std::pair<float, float>{a.cPuct, b.cPuct}, {a.dirichletScale, b.dirichletScale},
                        {a.noiseFraction, b.noiseFraction}, {back.sp.komi, job.sp.komi},
                        {back.sp.resignThreshold, job.sp.resignThreshold},
                        {back.sp.resignDisabledFraction, job.sp.resignDisabledFraction}})
        assert(std::memcmp(&x, &y, sizeof x) == 0);
    assert(back.sp.temperatureMoves == job.sp.temperatureMoves && back.sp.maxMoves == job.sp.maxMoves &&
           back.sp.resignation == job.sp.resignation);
    assert(back.networkFile == job.networkFile && back.networkHash == job.networkHash &&
           back.buildId == job.buildId && back.inferenceBatch == job.inferenceBatch);
    assert(serializeJob(back) == bytes);
    // A damaged job is refused, naming the defect.
    auto jobError = [](const std::string& b) {
        try {
            parseJob(b);
        } catch (const SliceError& e) {
            return std::string(e.what());
        }
        return std::string("accepted");
    };
    assert(contains(jobError("AZSX" + bytes.substr(4)), "magic"));
    assert(contains(jobError(bytes.substr(0, bytes.size() - 3)), "job"));
    std::string flipped = bytes;
    flipped[20] ^= 1;
    assert(contains(jobError(flipped), "checksum"));
    SelfPlayJob unresolved = job;
    unresolved.sp.maxMoves = 0;
    assert(contains(jobError(serializeJob(unresolved)), "max_moves"));
    SelfPlayJob overflow = job;
    overflow.first = 2147483647 - 2;
    assert(contains(jobError(serializeJob(overflow)), "game range"));
}

static bool sameRecords(const std::vector<GameRecord>& a, const std::vector<GameRecord>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const GameRecord& x = a[i];
        const GameRecord& y = b[i];
        if (!(x.boardSize == y.boardSize && std::memcmp(&x.komi, &y.komi, 4) == 0 && x.moves == y.moves &&
              x.stones == y.stones && x.visits == y.visits &&
              std::memcmp(&x.resultForBlack, &y.resultForBlack, 4) == 0 &&
              x.finalOwnership == y.finalOwnership && x.resigned == y.resigned &&
              x.resignDisabled == y.resignDisabled && x.resignLows[0] == y.resignLows[0] &&
              x.resignLows[1] == y.resignLows[1]))
            return false;
    }
    return true;
}

static void test_result_round_trip() {
    const SelfPlayJob job = sliceJob();
    const SliceResult result = sliceResult(job);
    int resigned = 0, disabled = 0, playedOut = 0;
    for (const GameRecord& r : result.records) {
        resigned += r.resigned;
        disabled += r.resignDisabled;
        playedOut += !r.resigned && !r.resignDisabled;
    }
    assert(resigned > 0 && disabled > 0);   // the slice exercises resignation
    (void)playedOut;
    const std::string path = tempPath("result");
    const std::string bytes = serializeSliceResult(result, job.first);
    assert(bytes.compare(0, 4, "AZSR") == 0);
    writeText(path, bytes);
    const SliceResult back = readSliceResult(path, serializeJob(job), job.buildId);
    assert(back.jobHash == result.jobHash && back.buildId == result.buildId &&
           back.fingerprint == result.fingerprint);
    assert(sameRecords(back.records, result.records));
    assert(back.stats.positions == result.stats.positions &&
           back.stats.evaluations == result.stats.evaluations &&
           std::memcmp(&back.stats.seconds, &result.stats.seconds, 8) == 0);
    std::filesystem::remove(path);
}

// FNV-1a 64 of the bytes in memory is fileHash of the same bytes on disk.
static void test_fnv1a64_matches_fileHash() {
    const std::string path = tempPath("fnv");
    for (const std::string& bytes : {std::string(), std::string("a"), std::string(70000, '\x5a'),
                                     serializeJob(sliceJob())}) {
        writeText(path, bytes);
        assert(fnv1a64(bytes) == fileHash(path));
    }
    assert(fnv1a64("") == 0xcbf29ce484222325ull);
    std::filesystem::remove(path);
}

// Every way a result can be wrong, stale, truncated or foreign is refused
// with a SliceError naming the defect.
static void test_result_refusals() {
    const SelfPlayJob job = sliceJob();
    const std::string jobBytes = serializeJob(job);
    const SliceResult good = sliceResult(job);
    const std::string goodBytes = serializeSliceResult(good, job.first);
    const std::string path = tempPath("refusal");
    int refusals = 0;
    auto errorOf = [&](const std::string& bytes, const std::string& jb = "",
                       const std::string& build = "") {
        writeText(path, bytes);
        try {
            readSliceResult(path, jb.empty() ? jobBytes : jb, build.empty() ? job.buildId : build);
        } catch (const SliceError& e) {
            ++refusals;
            return std::string(e.what());
        }
        return std::string("accepted");
    };
    // The trailing checksum, recomputed after an edit, so a later check sees it.
    auto rechecksum = [](std::string bytes) {
        const std::string body = bytes.substr(0, bytes.size() - 8);
        uint64_t h = fnv1a64(body);
        for (int i = 0; i < 8; ++i) bytes[bytes.size() - 8 + i] = static_cast<char>((h >> (8 * i)) & 0xff);
        return bytes;
    };
    auto edited = [&](const std::function<void(SliceResult&)>& edit) {
        SliceResult r = good;
        edit(r);
        return serializeSliceResult(r, job.first);
    };
    assert(errorOf(goodBytes) == "accepted");

    assert(contains(errorOf(rechecksum("AZSQ" + goodBytes.substr(4))), "magic"));
    std::string v2 = goodBytes;
    v2[4] = 2;
    assert(contains(errorOf(rechecksum(v2)), "version 2"));
    for (int k = 0; k < 20; ++k) {
        std::string flipped = goodBytes;
        const size_t at = static_cast<size_t>(k) * (goodBytes.size() - 1) / 19;
        flipped[at] ^= 0x10;
        const std::string msg = errorOf(flipped);
        assert(msg != "accepted");
        if (at >= 16) assert(contains(msg, "checksum"));
    }
    for (int k = 0; k < 10; ++k) {
        const size_t keep = static_cast<size_t>(k) * (goodBytes.size() - 1) / 10;
        assert(contains(errorOf(goodBytes.substr(0, keep)), "truncated"));
    }
    assert(contains(errorOf(rechecksum(goodBytes.substr(0, goodBytes.size() - 8) + "x" +
                                       goodBytes.substr(goodBytes.size() - 8))),
                    "length"));
    SelfPlayJob other = job;
    other.seed = 12;
    assert(contains(errorOf(goodBytes, serializeJob(other)), "another job"));
    assert(contains(errorOf(goodBytes, "", "fedcba9876543210fedcba9876543210fedcba98"), "build"));
    assert(contains(errorOf(edited([](SliceResult& r) { r.fingerprint = "naive; x\ny"; })), "fingerprint"));
    assert(contains(errorOf(edited([](SliceResult& r) { r.fingerprint.clear(); })), "fingerprint"));
    assert(contains(errorOf(edited([](SliceResult& r) { r.records.pop_back(); })), "6 games"));
    {
        // The first record's index (just after the header) made first + 1.
        std::string swapped = goodBytes;
        const size_t at = 4 + 4 + 8 + 8 + (4 + good.buildId.size()) + (4 + good.fingerprint.size()) + 4;
        assert(static_cast<int>(static_cast<unsigned char>(swapped[at])) == job.first);
        swapped[at] = static_cast<char>(job.first + 1);
        assert(contains(errorOf(rechecksum(swapped)), "index"));
    }
    // Records that do not replay or contradict the job.
    auto firstWhere = [&](const std::function<bool(const GameRecord&)>& pred) {
        for (size_t i = 0; i < good.records.size(); ++i)
            if (pred(good.records[i])) return i;
        assert(false && "no such record in the slice");
        return size_t(0);
    };
    const size_t resignedGame = firstWhere([](const GameRecord& r) { return r.resigned; });
    const size_t playedOut = firstWhere([](const GameRecord& r) { return !r.resigned && r.positions() > 2; });
    const size_t disabledGame = firstWhere([](const GameRecord& r) { return r.resignDisabled; });
    const size_t longGame = firstWhere([](const GameRecord& r) { return r.positions() > 2; });
    assert(contains(errorOf(edited([&](SliceResult& r) {
                        GameRecord& g = r.records[longGame];
                        uint8_t& s = g.stones[static_cast<size_t>(2) * 25 + 12];
                        s = s == 0 ? 1 : 0;
                    })),
                    "differs from the replayed one"));
    const size_t stoneFirst =
        firstWhere([](const GameRecord& r) { return r.positions() > 2 && r.moves[0] != 25; });
    assert(contains(errorOf(edited([&](SliceResult& r) {
                        GameRecord& g = r.records[stoneFirst];
                        g.moves[1] = g.moves[0];   // White onto Black's first stone
                    })),
                    "is not among the visited moves"));
    assert(contains(errorOf(edited([&](SliceResult& r) {
                        GameRecord& g = r.records[stoneFirst];
                        g.moves[1] = g.moves[0];
                        g.visits[1] = {{static_cast<int16_t>(g.moves[0]), 8}};
                    })),
                    "illegal move"));
    assert(contains(errorOf(edited([&](SliceResult& r) {
                        r.records[playedOut].resultForBlack = -r.records[playedOut].resultForBlack;
                        if (r.records[playedOut].resultForBlack == 0) r.records[playedOut].resultForBlack = 1;
                    })),
                    "result"));
    assert(contains(errorOf(edited([&](SliceResult& r) {
                        r.records[resignedGame].resultForBlack = -r.records[resignedGame].resultForBlack;
                    })),
                    "resign"));
    assert(contains(errorOf(edited([&](SliceResult& r) { r.records[playedOut].komi = 6.5f; })), "komi"));
    assert(contains(errorOf(edited([&](SliceResult& r) {
                        GameRecord& g = r.records[longGame];
                        g.visits[0] = {{static_cast<int16_t>(g.moves[0] == 3 ? 4 : 3), 8}};
                    })),
                    "visit"));
    assert(contains(errorOf(edited([&](SliceResult& r) { r.stats.positions += 1; })), "positions"));
    assert(contains(errorOf(edited([&](SliceResult& r) {
                        GameRecord& g = r.records[disabledGame];
                        auto& lows = g.resignLows[0].empty() ? g.resignLows[1] : g.resignLows[0];
                        lows.push_back({lows.back().first + 2, lows.back().second + 0.5f});
                    })),
                    "lows"));
    assert(contains(errorOf(edited([&](SliceResult& r) {
                        GameRecord& g = r.records[disabledGame];
                        auto& lows = g.resignLows[0].empty() ? g.resignLows[1] : g.resignLows[0];
                        lows.insert(lows.begin(), {0, 1.0f});
                    })),
                    "lows"));
    // Resignation off: no resigned or disabled game and no lows may appear.
    SelfPlayJob off = job;
    off.sp.resignation = false;
    const std::string offBytes = serializeJob(off);
    auto offResult = [&](const std::function<void(SliceResult&)>& edit) {
        SliceResult r = good;
        r.jobHash = fnv1a64(offBytes);
        edit(r);
        return serializeSliceResult(r, job.first);
    };
    assert(contains(errorOf(offResult([&](SliceResult& r) {
                                for (GameRecord& g : r.records) {
                                    g.resignDisabled = false;
                                    g.resignLows[0].clear();
                                    g.resignLows[1].clear();
                                }
                            }),
                            offBytes),
                    "resign"));
    // Resignation as v_resign dictates. A resigned game must end with a
    // low below it, at the position it resigned in.
    assert(contains(errorOf(edited([&](SliceResult& r) {
                        GameRecord& g = r.records[resignedGame];
                        g.resignLows[g.positions() % 2].pop_back();
                    })),
                    "resigned without a value below v_resign"));
    // Whether resignation was disabled is the game's first draw from its seed.
    assert(contains(errorOf(edited([&](SliceResult& r) { r.records[disabledGame].resignDisabled = false; })),
                    "resign-disabled flag does not match the game's seed"));
    assert(contains(errorOf(edited([&](SliceResult& r) { r.records[resignedGame].resignDisabled = true; })),
                    "resigned with resignation disabled"));
    // A game that played on has no low at its last position (no search ran there).
    assert(contains(errorOf(edited([&](SliceResult& r) {
                        GameRecord& g = r.records[disabledGame];
                        const int side = g.positions() % 2;
                        const float below = g.resignLows[side].empty() ? 0.0f : g.resignLows[side].back().second;
                        g.resignLows[side].push_back({g.positions(), std::max(-1.0f, below - 0.01f)});
                    })),
                    "a resign low after the last move"));
    // Every visited move was legal where it was searched.
    assert(contains(errorOf(edited([&](SliceResult& r) {
                        GameRecord& g = r.records[stoneFirst];
                        g.visits[1].push_back({static_cast<int16_t>(g.moves[0]), 1});   // Black's stone
                    })),
                    "visit to an illegal move"));
    {
        // With v_resign at -0.9 nobody resigns; a game whose lows go below it
        // without resigning (and without resignation disabled) is refused.
        SelfPlayJob low = job;
        low.sp.resignThreshold = -0.9f;
        const std::string lowBytes = serializeJob(low);
        SliceResult lowResult = sliceResult(low);
        size_t k = 0;
        while (k < lowResult.records.size() &&
               (lowResult.records[k].resignDisabled || lowResult.records[k].resigned ||
                lowResult.records[k].positions() < 4))
            ++k;
        assert(k < lowResult.records.size());
        assert(errorOf(serializeSliceResult(lowResult, low.first), lowBytes) == "accepted");
        GameRecord& g = lowResult.records[k];
        // Black's next record low: a later even ply before the last position.
        ResignLows& lows = g.resignLows[0];
        const int after = lows.empty() ? 0 : lows.back().first;
        const int ply = ((g.positions() - 1) / 2) * 2;
        assert(ply > after && ply >= 2);
        lows.push_back({ply, -0.95f});
        assert(contains(errorOf(serializeSliceResult(lowResult, low.first), lowBytes),
                        "Black went below v_resign at ply " + std::to_string(ply) + " without resigning"));
    }
    // Another board size than the job's.
    SelfPlayJob nine = job;
    nine.boardSize = 9;
    const std::string nineBytes = serializeJob(nine);
    SliceResult nineResult = good;
    nineResult.jobHash = fnv1a64(nineBytes);
    assert(contains(errorOf(serializeSliceResult(nineResult, job.first), nineBytes), "board size"));
    std::filesystem::remove(path);
    std::printf("result refusals: %d\n", refusals);
    assert(refusals >= 30);
}

// ---- runSliceJob and the selfplay app ----------------------------------------------

// A work directory holding a tiny network and a job for it, as a remote's
// would after the uploads.
struct SliceWork {
    std::filesystem::path dir;
    std::string jobPath, outPath;
    SelfPlayJob job;
    std::string jobBytes;
    explicit SliceWork(const std::string& name)
        : dir(std::filesystem::temp_directory_path() /
              ("azgo_test_work_" + std::to_string(getpid()) + "_" + name)) {
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        Network net(sliceNet(), 33);
        const std::string netPath = (dir / "net_test.azn").string();
        saveNetwork(net, netPath);
        job = sliceJob();
        job.count = 3;
        job.networkFile = "net_test.azn";
        job.networkHash = fileHash(netPath);
        job.buildId = buildId();
        writeJob();
        outPath = (dir / "g0002_3_a1.result").string();
    }
    void writeJob() {
        jobBytes = serializeJob(job);
        jobPath = (dir / "g0002_3_a1.job").string();
        writeText(jobPath, jobBytes);
    }
    ~SliceWork() {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
};

static void test_run_slice_job_in_process() {
    SliceWork work("inproc");
    std::ostringstream err;
    assert(runSliceJob(work.jobPath, work.outPath, "naive", 2, 0, err) == 0);
    const SliceResult result = readSliceResult(work.outPath, work.jobBytes, buildId());
    assert(result.fingerprint == engineFingerprint("naive"));
    const Network net = loadNetwork((work.dir / "net_test.azn").string());
    const std::vector<GameRecord> direct = playSelfPlayGames(
        net, work.job.sp, work.job.seed, work.job.generation, work.job.first, work.job.count,
        EngineRun{"naive", 1});
    assert(sameRecords(result.records, direct));
    assert(!std::filesystem::exists(work.outPath + ".tmp"));
}

static void test_run_slice_job_refusals() {
    auto refusal = [](SliceWork& work, const std::string& backend) {
        std::ostringstream err;
        const int status = runSliceJob(work.jobPath, work.outPath, backend, 1, 0, err);
        assert(status == 1 && !std::filesystem::exists(work.outPath));
        return err.str();
    };
    {
        SliceWork work("hash");
        work.job.networkHash ^= 1;
        work.writeJob();
        assert(contains(refusal(work, "naive"), "hash"));
    }
    {
        SliceWork work("build");
        work.job.buildId = "0000000000000000000000000000000000000000";
        work.writeJob();
        const std::string msg = refusal(work, "naive");
        assert(contains(msg, "build") && contains(msg, buildId()));
    }
    {
        SliceWork work("missing");
        std::filesystem::remove(work.dir / "net_test.azn");
        assert(contains(refusal(work, "naive"), "net_test.azn"));
    }
    {
        SliceWork work("badjob");
        writeText(work.jobPath, work.jobBytes.substr(0, 20));
        assert(contains(refusal(work, "naive"), "job"));
    }
    for (const char* backend : {"blas", "metal"}) {
        try {
            checkBackendAvailable(backend);
            continue;   // available here: nothing to refuse
        } catch (const std::invalid_argument&) {
        }
        SliceWork work(std::string("unavailable_") + backend);
        assert(contains(refusal(work, backend), "not available"));
    }
}

// Waits up to `seconds` for process `pid` to be gone.
static bool goneWithin(pid_t pid, double seconds) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() < end) {
        if (kill(pid, 0) != 0 && errno == ESRCH) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return false;
}

static pid_t readPid(const std::string& path, double seconds) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() < end) {
        const std::string text = readText(path);
        if (!text.empty() && text.back() == '\n') return static_cast<pid_t>(std::atoi(text.c_str()));
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return 0;
}

// A timeout or a cancel stops the child and everything it started (its
// process group): a grandchild must be gone too.
static void test_run_process_stops_the_whole_group() {
    const std::string pidFile = tempPath("grandchild.pid");
    const std::vector<std::string> parentOfSleep = {"sh", "-c", "sleep 30 & echo $! > " + pidFile + "; wait"};
    std::filesystem::remove(pidFile);
    ProcessResult r = runProcess(parentOfSleep, "", 0.5);
    assert(r.timedOut);
    pid_t grandchild = readPid(pidFile, 1);
    assert(grandchild > 0 && goneWithin(grandchild, 2));

    std::filesystem::remove(pidFile);
    std::atomic<bool> cancel{false};
    std::thread canceller([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        cancel = true;
    });
    ProcessOptions options;
    options.cancel = &cancel;
    const auto start = std::chrono::steady_clock::now();
    r = runProcess(parentOfSleep, options);
    canceller.join();
    assert(r.cancelled && !r.timedOut && r.exitCode == -1);
    assert(std::chrono::steady_clock::now() - start < std::chrono::seconds(6));
    grandchild = readPid(pidFile, 1);
    assert(grandchild > 0 && goneWithin(grandchild, 2));
    std::filesystem::remove(pidFile);

    // holdStdin: the child's stdin stays open until the child is reaped,
    // so cat waits; without it, cat reads /dev/null and ends at once.
    ProcessOptions hold;
    hold.holdStdin = true;
    hold.timeout = 0.4;
    assert(runProcess({"cat"}, hold).timedOut);
    assert(runProcess({"cat"}, "", 5).exitCode == 0);
}

// A coordinator killed by SIGINT or SIGTERM (forwardTerminationSignals)
// takes its children's process groups with it.
static void test_termination_signals_reach_children() {
    const std::string pidFile = tempPath("forwarded.pid");
    std::filesystem::remove(pidFile);
    const pid_t child = fork();
    if (child == 0) {
        forwardTerminationSignals();
        runProcess({"sh", "-c", "sleep 30 & echo $! > " + pidFile + "; wait"}, "", 60);
        _exit(0);
    }
    const pid_t grandchild = readPid(pidFile, 5);
    assert(grandchild > 0);
    kill(child, SIGTERM);
    int status = 0;
    waitpid(child, &status, 0);
    assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGTERM);
    assert(goneWithin(grandchild, 2));
    std::filesystem::remove(pidFile);
}

// Runs argv with stdout and stderr to `log` (replaced); returns the exit
// status, or -1.
static int spawnAndWait(const std::vector<std::string>& args, const std::string& log) {
    std::filesystem::remove(log);
    return runProcess(args, log, 300).exitCode;
}

// ---- runProcess ----------------------------------------------------------------------

static int openDescriptors() {
    int count = 0;
    if (DIR* d = opendir("/dev/fd")) {
        while (readdir(d)) ++count;
        closedir(d);
    }
    return count;
}

static void test_run_process() {
    const std::string log = tempPath("process.log");
    std::filesystem::remove(log);
    assert(runProcess({"true"}, "", 10).exitCode == 0);
    const ProcessResult f = runProcess({"false"}, "", 10);
    assert(f.exitCode == 1 && !f.timedOut && f.signal == 0 && f.error.empty());
    const ProcessResult three = runProcess({"sh", "-c", "echo hi; echo err >&2; exit 3"}, log, 10);
    assert(three.exitCode == 3);
    // Appended: a second run adds its lines after the first's.
    runProcess({"sh", "-c", "echo again"}, log, 10);
    assert(readText(log) == "hi\nerr\nagain\n");

    // A timeout terminates the child (and its children: its process group).
    auto start = std::chrono::steady_clock::now();
    const ProcessResult slow = runProcess({"sleep", "10"}, "", 0.3);
    assert(slow.timedOut && slow.exitCode == -1 && slow.signal == SIGTERM);
    assert(std::chrono::steady_clock::now() - start < std::chrono::seconds(6));
    // One that ignores SIGTERM is killed 5 s later.
    start = std::chrono::steady_clock::now();
    const ProcessResult stubborn = runProcess({"sh", "-c", "trap '' TERM; sleep 30; exit 0"}, "", 0.3);
    assert(stubborn.timedOut && stubborn.signal == SIGKILL);
    const auto took = std::chrono::steady_clock::now() - start;
    assert(took >= std::chrono::milliseconds(5000) && took < std::chrono::seconds(9));

    const ProcessResult missing = runProcess({"/nonexistent/azgo-no-such-program"}, log, 10);
    assert(missing.exitCode == -1 && !missing.error.empty() && !missing.timedOut);
    const ProcessResult unnamed = runProcess({"azgo-no-such-program-on-path"}, log, 10);
    assert(unnamed.exitCode == -1 && !unnamed.error.empty());
    bool threw = false;
    try {
        runProcess({}, "", 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    // stdin from a file.
    const std::string input = tempPath("process.in");
    writeText(input, "from a file\n");
    std::filesystem::remove(log);
    assert(runProcess({"cat"}, log, 10, input).exitCode == 0);
    assert(readText(log) == "from a file\n");
    // Without one, stdin is /dev/null: cat ends at once.
    assert(runProcess({"cat"}, "", 10).exitCode == 0);

    // No zombie, no descriptor left open.
    const int before = openDescriptors();
    for (int i = 0; i < 200; ++i) assert(runProcess({"true"}, log, 10).exitCode == 0);
    errno = 0;
    assert(waitpid(-1, nullptr, WNOHANG) == -1 && errno == ECHILD);
    assert(openDescriptors() == before);
    std::filesystem::remove(log);
    std::filesystem::remove(input);
}

// The real binary gives the in-process result, bit for bit (but the
// wall time).
static void test_selfplay_exe_end_to_end() {
    const std::string exe = AZGO_SELFPLAY_EXE;
    SliceWork work("exe");
    const std::string log = (work.dir / "selfplay.log").string();
    assert(spawnAndWait({exe, "--version"}, log) == 0);
    const std::string version = readText(log);
    assert(version == "selfplay " + shortBuildId() + " (" + buildOptions() + ")\nbuild " + buildId() + "\n");
    assert(spawnAndWait({exe, "--fingerprint", "--backend", "naive"}, log) == 0);
    assert(readText(log) == engineFingerprint("naive") + "\n");
    assert(spawnAndWait({exe, "--job", work.jobPath, "--out", work.outPath, "--backend", "naive",
                         "--workers", "3", "--inference-batch", "16"},
                        log) == 0);
    const SliceResult viaExe = readSliceResult(work.outPath, work.jobBytes, buildId());
    const std::string inProcessOut = (work.dir / "inproc.result").string();
    std::ostringstream err;
    assert(runSliceJob(work.jobPath, inProcessOut, "naive", 1, 0, err) == 0);
    const SliceResult inProcess = readSliceResult(inProcessOut, work.jobBytes, buildId());
    assert(sameRecords(viaExe.records, inProcess.records));
    assert(viaExe.fingerprint == inProcess.fingerprint && viaExe.jobHash == inProcess.jobHash);
    assert(viaExe.stats.positions == inProcess.stats.positions &&
           viaExe.stats.evaluations == inProcess.stats.evaluations);
    // Usage errors exit 2, failures 1.
    assert(spawnAndWait({exe, "--job", work.jobPath}, log) == 2);
    assert(spawnAndWait({exe, "--bogus"}, log) == 2);
    assert(spawnAndWait({exe, "--job", work.jobPath + ".missing", "--out", work.outPath, "--backend",
                         "naive"},
                        log) == 1);
    assert(spawnAndWait({exe, "--fingerprint", "--backend", "gpu"}, log) == 1);
}

// With --exit-on-stdin-eof, selfplay stops as soon as its stdin closes
// (the coordinator, or the ssh connection to it, is gone).
static void test_selfplay_exits_when_stdin_closes() {
    SliceWork work("stdin_eof");
    work.job.count = 400;
    work.writeJob();
    const std::string log = (work.dir / "eof.log").string();
    const auto start = std::chrono::steady_clock::now();
    // runProcess gives it /dev/null: end of file at once.
    const int status = spawnAndWait({AZGO_SELFPLAY_EXE, "--job", work.jobPath, "--out", work.outPath,
                                     "--backend", "naive", "--workers", "1", "--exit-on-stdin-eof"},
                                    log);
    assert(status == 1 && contains(readText(log), "stdin closed"));
    assert(std::chrono::steady_clock::now() - start < std::chrono::seconds(5));
    assert(!std::filesystem::exists(work.outPath));
}

// A remote slice honours the generation's stop: when a local engine fails,
// the fake remote's selfplay (a child process) is stopped at once and its
// process is gone.
static void test_failing_engine_stops_remote_slices() {
    if (!metalBuilt() || !metalAvailable()) return;
    SliceWork work("stop_remote");
    Config cfg;
    cfg.boardSize = 5;
    cfg.history = 2;
    cfg.blocks = 1;
    cfg.filters = 4;
    cfg.valueHidden = 8;
    cfg.gamesPerGeneration = 2000;
    applyOverride(cfg, "selfplay_engines=fake/naive=1999, local/metal:1");
    const HostsFile hosts = parseHostsText("local inference_batch=4\nfake exe=" + std::string(AZGO_SELFPLAY_EXE) +
                                               " workdir=" + (work.dir / "remote").string() + " cpu_workers=1\n",
                                           "h");
    const std::string runDir = (work.dir / "run").string();
    std::filesystem::create_directories(runDir);
    SelfPlayCoordinator coordinator(cfg, runDir, hosts, nullptr);
    const Network net = loadNetwork((work.dir / "net_test.azn").string());
    SelfPlayConfig sp;
    sp.mcts.simulations = 64;
    sp.mcts.batchSize = 8;
    sp.maxMoves = 50;
    const auto start = std::chrono::steady_clock::now();
    std::string error;
    try {
        coordinator.play(1, (work.dir / "net_test.azn").string(), net, sp, 2000);
    } catch (const std::exception& e) {
        error = e.what();
    }
    assert(contains(error, "do not fit batches of 4"));
    assert(std::chrono::steady_clock::now() - start < std::chrono::seconds(10));
    assert(runProcess({"pgrep", "-f", runIdOf(runDir)}, "", 10).exitCode == 1);   // none left
}

// ---- transports ---------------------------------------------------------------------

static MachineSettings sshMachine() {
    MachineSettings m;
    m.alias = "mini1";
    m.ssh = "u@h";
    m.exe = "~/azgo/build-metal/selfplay";
    m.workdir = "~/azgo/runs/remote";
    m.nice = 5;
    m.timeout = 600;
    return m;
}

// The exact command lines, and that only validated pieces reach the remote
// shell. No test opens a connection.
static void test_ssh_argv() {
    const MachineSettings m = sshMachine();
    const std::string dir = "~/azgo/runs/remote/run-1a2b3c4d";
    const std::vector<std::string> args = {"--job", dir + "/g0001_0_a1.job", "--out", dir + "/g0001_0_a1.result",
                                           "--backend", "metal", "--workers", "8"};
    const std::vector<std::string> expected = {
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o",
        "ServerAliveCountMax=4", "u@h",
        "exec nice -n 5 ~/azgo/build-metal/selfplay --job " + dir + "/g0001_0_a1.job --out " + dir +
            "/g0001_0_a1.result --backend metal --workers 8"};
    assert(sshArgv(m, args) == expected);
    assert((scpArgv(m, "/tmp/a/net_1.azn", dir + "/net_1.azn", true) ==
            std::vector<std::string>{"scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                     "/tmp/a/net_1.azn", "u@h:" + dir + "/net_1.azn"}));
    assert((scpArgv(m, "/tmp/a/r.result", dir + "/r.result", false) ==
            std::vector<std::string>{"scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                     "u@h:" + dir + "/r.result", "/tmp/a/r.result"}));
    // Settings smuggled past the hosts file's checks, or unsafe arguments.
    auto refused = [&](const MachineSettings& bad, const std::vector<std::string>& a) {
        try {
            sshArgv(bad, a);
        } catch (const std::logic_error&) {
            return true;
        }
        return false;
    };
    MachineSettings bad = m;
    bad.exe = "/a;rm -rf /";
    assert(refused(bad, args));
    bad = m;
    bad.ssh = "u@h -oProxyCommand=x";
    assert(refused(bad, args));
    bad = m;
    bad.ssh = "-oProxyCommand=x";
    assert(refused(bad, args));
    assert(refused(m, {"--job", "a b"}));
    assert(refused(m, {"--job", "$(reboot)"}));
    assert(refused(m, {"--job", "x'y"}));
    // Local paths never reach a shell: any characters, made absolute so a
    // ':' cannot be read as a host.
    assert(scpArgv(m, "/tmp/run dir/x.job", dir + "/x.job", true)[6] == "/tmp/run dir/x.job");
    const std::string relative = scpArgv(m, "a:b/x.result", dir + "/x.result", false)[7];
    assert(relative[0] == '/' && relative.size() > 13 &&
           relative.substr(relative.size() - 13) == "/a:b/x.result");
    bool threw = false;
    try {
        scpArgv(m, "/tmp/x", dir + "/a b", true);   // the remote path still must be safe
    } catch (const std::logic_error&) {
        threw = true;
    }
    assert(threw);
    // The run id of any run directory is safe on a remote.
    for (const std::string& run : {std::string("/tmp/my run"), std::string("a:b"), std::string("x;y/"),
                                   std::string("-run")}) {
        const std::string id = runIdOf(run);
        assert(sshArgv(m, {"--job", "~/w/" + id + "/g.job"}).size() == 11);
        assert(id.size() > 17 && id[id.size() - 17] == '-');   // 16 hex digits of the path's hash
    }
}

static void test_subprocess_transport_round_trip() {
    SliceWork work("transport");
    MachineSettings m;
    m.alias = "fake";
    m.exe = AZGO_SELFPLAY_EXE;
    m.workdir = (work.dir / "remote").string();
    const std::unique_ptr<Transport> t = makeTransport(m);
    const std::string remoteNet = m.workdir + "/run-x/net_test.azn";
    assert(!t->exists(remoteNet));
    t->upload((work.dir / "net_test.azn").string(), remoteNet);
    assert(t->exists(remoteNet));
    const std::string log = (work.dir / "fake.log").string();
    const ProcessResult version = t->run({"--version"}, log, 30);
    assert(version.exitCode == 0 && contains(readText(log), "selfplay " + shortBuildId()));
    const std::string back = (work.dir / "back.azn").string();
    t->download(remoteNet, back);
    assert(readText(back) == readText((work.dir / "net_test.azn").string()));
    t->remove({remoteNet, m.workdir + "/run-x/not-there"});
    assert(!t->exists(remoteNet));
    bool threw = false;
    try {
        t->download(remoteNet, back);
    } catch (const SliceError&) {
        threw = true;
    }
    assert(threw);
    t->killJob("run-x/g0001_0_a1.job");   // best effort: nothing to kill
}

// The in-process transport (tests) plays the job on the calling thread and
// writes the same result as the real binary.
static void test_in_process_transport() {
    SliceWork work("inprocess");
    MachineSettings m;
    m.alias = "fake";
    m.exe = AZGO_SELFPLAY_EXE;
    m.workdir = (work.dir / "remote").string();
    std::vector<std::unique_ptr<Transport>> transports;
    transports.push_back(makeTransport(m));
    transports.push_back(std::make_unique<InProcessTransport>(m));
    std::vector<SliceResult> results;
    for (const std::unique_ptr<Transport>& t : transports) {
        const std::string dir = m.workdir + "/run-y";
        t->upload((work.dir / "net_test.azn").string(), dir + "/net_test.azn");
        t->upload(work.jobPath, dir + "/g0002_3_a1.job");
        const ProcessResult r = t->run({"--job", dir + "/g0002_3_a1.job", "--out", dir + "/g0002_3_a1.result",
                                        "--backend", "naive", "--workers", "2"},
                                       (work.dir / "run.log").string(), 120);
        assert(r.exitCode == 0);
        const std::string local = (work.dir / "fetched.result").string();
        t->download(dir + "/g0002_3_a1.result", local);
        results.push_back(readSliceResult(local, work.jobBytes, buildId()));
        t->remove({dir + "/g0002_3_a1.job", dir + "/g0002_3_a1.result"});
    }
    assert(sameRecords(results[0].records, results[1].records));
    assert(results[0].fingerprint == results[1].fingerprint);
    // A failing job is a non-zero exit, as from the binary.
    InProcessTransport inProcess(m);
    assert(inProcess.run({"--job", m.workdir + "/missing.job", "--out", m.workdir + "/x.result", "--backend",
                          "naive"},
                         (work.dir / "run.log").string(), 10)
               .exitCode == 1);
}

// Three remote slices (in-process transport) finishing in each of the six
// orders, by delays: the merged records are the same, and are the games of
// one engine playing the whole generation.
static void test_merge_order_independent() {
    SliceWork work("merge");
    Config cfg;
    cfg.boardSize = 5;
    cfg.history = 2;
    cfg.blocks = 1;
    cfg.filters = 4;
    cfg.valueHidden = 8;
    cfg.gamesPerGeneration = 6;
    cfg.seed = 11;
    applyOverride(cfg, "selfplay_engines=fa/naive:1, fb/naive:1, fc/naive:1");
    std::string hostsText;
    for (const char* alias : {"fa", "fb", "fc"})
        hostsText += std::string(alias) + " exe=unused workdir=" + (work.dir / "remote").string() + "\n";
    const HostsFile hosts = parseHostsText(hostsText, "h");
    const std::string netPath = (work.dir / "net_test.azn").string();
    const Network net = loadNetwork(netPath);
    SelfPlayConfig sp = work.job.sp;
    const std::vector<GameRecord> whole =
        playSelfPlayGames(net, sp, cfg.seed, 2, 0, 6, EngineRun{"naive", 1});
    const double delays[3] = {0.0, 0.15, 0.3};
    int order[3] = {0, 1, 2};
    int orders = 0;
    distributed_probe::useInProcessTransport(true);
    do {
        distributed_probe::delaySlice("fa/naive", delays[order[0]]);
        distributed_probe::delaySlice("fb/naive", delays[order[1]]);
        distributed_probe::delaySlice("fc/naive", delays[order[2]]);
        SelfPlayCoordinator coordinator(cfg, (work.dir / "run").string(), hosts, nullptr);
        coordinator.preflight();
        const std::vector<GameRecord> merged = coordinator.play(2, netPath, net, sp, 6);
        assert(sameRecords(merged, whole));
        assert(coordinator.reports().size() == 3 && coordinator.reports()[2].engine == "fc/naive");
        ++orders;
    } while (std::next_permutation(order, order + 3));
    distributed_probe::reset();
    assert(orders == 6);
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
    test_fingerprint();
    test_manifest_rows();
    test_failing_engine_stops_the_others();
    test_job_round_trip();
    test_result_round_trip();
    test_fnv1a64_matches_fileHash();
    test_result_refusals();
    test_run_slice_job_in_process();
    test_run_slice_job_refusals();
    test_run_process();
    test_run_process_stops_the_whole_group();
    test_termination_signals_reach_children();
    test_selfplay_exe_end_to_end();
    test_selfplay_exits_when_stdin_closes();
    test_failing_engine_stops_remote_slices();
    test_ssh_argv();
    test_subprocess_transport_round_trip();
    test_in_process_transport();
    test_merge_order_independent();
    std::printf("test_distributed: OK\n");
    return 0;
}
