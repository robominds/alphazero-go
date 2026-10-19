#include "azgo/engines.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/utsname.h>
#include "azgo/arena.hpp"
#include "azgo/backend.hpp"
#include "azgo/metal.hpp"
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace azgo {

namespace {

constexpr long long kMaxShare = 1000000;   // the largest weight or count

std::string trim(const std::string& s) {
    const char* ws = " \t\r\n\f\v";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}

// The whole of v is a decimal integer in [lo, hi] (digits only: no sign,
// no spaces); otherwise the reason.
std::string parseShare(const std::string& v, long long lo, long long hi, int& out) {
    const std::string reason = "expected an integer in [" + std::to_string(lo) + ", " +
                               std::to_string(hi) + "], got '" + v + "'";
    if (v.empty() || v[0] < '0' || v[0] > '9') return reason;
    errno = 0;
    char* end = nullptr;
    const long long x = std::strtoll(v.c_str(), &end, 10);
    if (errno == ERANGE || *end != '\0' || x < lo || x > hi) return reason;
    out = static_cast<int>(x);
    return "";
}

EngineSpec parseItem(const std::string& item) {
    const std::string shape =
        "expected machine/backend:weight or machine/backend=games, got '" + item + "'";
    const size_t slash = item.find('/');
    if (slash == std::string::npos) throw std::invalid_argument(shape);
    const size_t sep = item.find_first_of(":=", slash);
    if (sep == std::string::npos) throw std::invalid_argument(shape);
    EngineSpec e;
    e.machine = trim(item.substr(0, slash));
    e.backend = trim(item.substr(slash + 1, sep - slash - 1));
    const std::string number = trim(item.substr(sep + 1));
    if (!validAlias(e.machine))
        throw std::invalid_argument("bad machine alias '" + e.machine +
                                    "' (expected [a-z][a-z0-9_-]{0,31})");
    if (e.backend != "naive" && e.backend != "blas" && e.backend != "metal")
        throw std::invalid_argument("unknown backend '" + e.backend +
                                    "' (expected naive|blas|metal)");
    if (item[sep] == ':') {
        const std::string reason = parseShare(number, 1, kMaxShare, e.weight);
        if (!reason.empty()) throw std::invalid_argument("weight: " + reason);
    } else {
        const std::string reason = parseShare(number, 0, kMaxShare, e.games);
        if (!reason.empty()) throw std::invalid_argument("games: " + reason);
    }
    return e;
}

} // namespace

std::string EngineSpec::name() const { return machine + "/" + backend; }

bool validAlias(const std::string& alias) {
    if (alias.empty() || alias.size() > 32 || alias[0] < 'a' || alias[0] > 'z') return false;
    for (char c : alias)
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-'))
            return false;
    return true;
}

std::vector<EngineSpec> parseEngineList(const std::string& text) {
    std::vector<EngineSpec> list;
    if (trim(text).empty()) return list;
    // Split by hand, keeping empty items, so that "a/naive:1," is refused.
    for (size_t start = 0;;) {
        const size_t comma = text.find(',', start);
        const size_t length = comma == std::string::npos ? std::string::npos : comma - start;
        const std::string item = trim(text.substr(start, length));
        const std::string where = "engine " + std::to_string(list.size() + 1) + ": ";
        if (item.empty()) throw std::invalid_argument(where + "empty item");
        try {
            list.push_back(parseItem(item));
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument(where + e.what());
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return list;
}

std::string printEngineList(const std::vector<EngineSpec>& engines) {
    std::string out;
    for (const EngineSpec& e : engines) {
        if (!out.empty()) out += ", ";
        out += e.name();
        out += e.games >= 0 ? "=" + std::to_string(e.games) : ":" + std::to_string(e.weight);
    }
    return out;
}

std::vector<Slice> assignGames(const std::vector<EngineSpec>& engines, int games) {
    if (games < 0) throw std::invalid_argument("assignGames: negative game count");
    const size_t n = engines.size();
    long long fixed = 0, weights = 0;
    for (const EngineSpec& e : engines) {
        if ((e.weight > 0) == (e.games >= 0))
            throw std::invalid_argument("assignGames: " + e.name() +
                                        " needs exactly one of a weight (> 0) and a count (>= 0)");
        if (e.games >= 0) fixed += e.games;
        else weights += e.weight;
    }
    const long long rest = games - fixed;
    if (rest < 0)
        throw std::invalid_argument("assignGames: fixed counts add up to " + std::to_string(fixed) +
                                    ", more than " + std::to_string(games) + " games");
    if (weights == 0 && rest != 0)
        throw std::invalid_argument("assignGames: no weighted engine takes the remaining " +
                                    std::to_string(rest) + " games");

    // 1. and 2.: the counts, in 64-bit integers (rest * w <= 10^12).
    std::vector<long long> count(n, 0), remainder(n, -1);
    long long given = 0;
    for (size_t e = 0; e < n; ++e) {
        if (engines[e].games >= 0) {
            count[e] = engines[e].games;
        } else {
            count[e] = rest * engines[e].weight / weights;
            remainder[e] = rest * engines[e].weight % weights;
            given += count[e];
        }
    }
    // Fewer leftover games than weighted items: each gets at most one, the
    // largest remainder first, the earlier item on a tie.
    for (long long left = weights > 0 ? rest - given : 0; left > 0; --left) {
        size_t best = n;
        for (size_t e = 0; e < n; ++e)
            if (remainder[e] >= 0 && (best == n || remainder[e] > remainder[best])) best = e;
        ++count[best];
        remainder[best] = -1;
    }

    // 3. Contiguous slices in list order.
    std::vector<Slice> slices;
    int first = 0;
    for (size_t e = 0; e < n; ++e) {
        if (count[e] == 0) continue;
        slices.push_back({static_cast<int>(e), first, static_cast<int>(count[e])});
        first += static_cast<int>(count[e]);
    }
    return slices;
}

// ---- the hosts file ------------------------------------------------------------

namespace {

// [A-Za-z0-9_./~@:+-]: safe in an ssh command line without quoting.
bool safeChar(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
           c == '_' || c == '.' || c == '/' || c == '~' || c == '@' || c == ':' || c == '+' ||
           c == '-';
}

// An integer in [lo, hi], the whole value, digits only (no sign); "" or the reason.
std::string hostsInt(const std::string& v, int lo, int hi, int& out) {
    const std::string reason = "expected an integer in [" + std::to_string(lo) + ", " +
                               std::to_string(hi) + "], got '" + v + "'";
    if (v.empty() || v[0] < '0' || v[0] > '9') return reason;
    errno = 0;
    char* end = nullptr;
    const long long x = std::strtoll(v.c_str(), &end, 10);
    if (v.empty() || errno == ERANGE || *end != '\0' || x < lo || x > hi) return reason;
    out = static_cast<int>(x);
    return "";
}

// Applies key=value to m; "" or the reason.
std::string applyHostsKey(MachineSettings& m, const std::string& key, const std::string& value) {
    if (key == "ssh") {
        m.ssh = value;
        return "";
    }
    if (key == "exe") {
        m.exe = value;
        return "";
    }
    if (key == "workdir") {
        m.workdir = value;
        return "";
    }
    if (key == "metal_workers") return hostsInt(value, 0, kMaxArenaWorkers, m.metalWorkers);
    if (key == "cpu_workers") return hostsInt(value, 0, 256, m.cpuWorkers);
    if (key == "inference_batch") return hostsInt(value, 0, 1024, m.inferenceBatch);
    if (key == "nice") return hostsInt(value, 0, 19, m.nice);
    if (key == "cpu_qos") {
        if (value == "default") m.cpuQos = CpuQos::Default;
        else if (value == "utility") m.cpuQos = CpuQos::Utility;
        else return "expected default|utility, got '" + value + "'";
        return "";
    }
    if (key == "timeout") {
        if (value[0] < '0' || value[0] > '9')
            return "expected seconds in [0, 1e7], got '" + value + "'";
        errno = 0;
        char* end = nullptr;
        const double t = std::strtod(value.c_str(), &end);
        if (errno == ERANGE || *end != '\0' || !std::isfinite(t) || t < 0 || t > 1e7)
            return "expected seconds in [0, 1e7], got '" + value + "'";
        m.timeout = t;
        return "";
    }
    return "?";   // unknown keys are refused before this is called
}

const char* const kHostsKeys[] = {"ssh",     "exe",  "workdir", "metal_workers", "cpu_workers",
                                  "inference_batch", "cpu_qos", "nice", "timeout"};

} // namespace

MachineSettings HostsFile::settings(const std::string& alias) const {
    auto it = machines.find(alias);
    if (it != machines.end()) return it->second;
    if (alias == "local") {
        MachineSettings m;
        m.alias = "local";
        m.cpuQos = CpuQos::Utility;
        return m;
    }
    throw ConfigError("machine '" + alias + "' is not defined in the hosts file" +
                      (source.empty() ? std::string() : " " + source) +
                      " (add a line: " + alias +
                      " ssh=<user>@<address> exe=<path> workdir=<path>)");
}

HostsFile parseHostsText(const std::string& text, const std::string& source) {
    HostsFile hosts;
    hosts.source = source;
    std::map<std::string, int> firstLine;
    std::istringstream in(text);
    int lineNo = 0;
    for (std::string raw; std::getline(in, raw);) {
        ++lineNo;
        const std::string where = source + ":" + std::to_string(lineNo) + ": ";
        const size_t hash = raw.find('#');
        if (hash != std::string::npos) raw.erase(hash);
        std::istringstream words(raw);
        std::string alias;
        if (!(words >> alias)) continue;   // blank or comment
        if (!validAlias(alias))
            throw ConfigError(where + "bad machine alias '" + alias +
                              "' (expected [a-z][a-z0-9_-]{0,31})");
        if (firstLine.count(alias))
            throw ConfigError(where + "machine '" + alias + "' repeated (first on line " +
                              std::to_string(firstLine[alias]) + ")");
        firstLine[alias] = lineNo;
        MachineSettings m;
        m.alias = alias;
        m.cpuQos = alias == "local" ? CpuQos::Utility : CpuQos::Default;
        std::vector<std::string> seen;
        for (std::string pair; words >> pair;) {
            const size_t eq = pair.find('=');
            if (eq == std::string::npos || eq == 0)
                throw ConfigError(where + "expected key=value, got '" + pair + "'");
            const std::string key = pair.substr(0, eq), value = pair.substr(eq + 1);
            const auto known = std::find(std::begin(kHostsKeys), std::end(kHostsKeys), key);
            if (known == std::end(kHostsKeys))
                throw ConfigError(where + "unknown key '" + key + "'");
            if (std::find(seen.begin(), seen.end(), key) != seen.end())
                throw ConfigError(where + key + ": repeated");
            seen.push_back(key);
            if (value.empty()) throw ConfigError(where + key + ": empty value");
            for (char c : value)
                if (!safeChar(c))
                    throw ConfigError(where + key + ": character '" + std::string(1, c) +
                                      "' not allowed (values may hold only [A-Za-z0-9_./~@:+-])");
            const std::string reason = applyHostsKey(m, key, value);
            if (!reason.empty()) throw ConfigError(where + key + ": " + reason);
        }
        if (m.local() && !m.ssh.empty())
            throw ConfigError(where + "local is this machine: it takes no ssh");
        if (!m.local() && m.exe.empty())
            throw ConfigError(where + alias + " needs exe=<path of its selfplay binary>");
        if (!m.local() && m.workdir.empty())
            throw ConfigError(where + alias +
                              " needs workdir=<directory for networks, jobs and results>");
        hosts.machines[alias] = m;
    }
    return hosts;
}

HostsFile loadHostsFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        HostsFile none;
        none.source = path;
        return none;
    }
    std::ostringstream text;
    text << in.rdbuf();
    return parseHostsText(text.str(), path);
}

std::string resolveHostsPath(const std::string& cli) {
    if (!cli.empty()) return cli;
    const char* env = std::getenv("AZGO_HOSTS");
    if (env && *env) return env;
    const char* home = std::getenv("HOME");
    if (home && *home) return std::string(home) + "/.config/azgo/hosts";
    return "";
}

int engineWorkers(const MachineSettings& machine, const std::string& backend, int sliceGames,
                  int cores) {
    int workers;
    if (backend == "metal") {
        workers = machine.metalWorkers > 0 ? machine.metalWorkers : kMaxArenaWorkers;
    } else if (machine.cpuWorkers > 0) {
        workers = machine.cpuWorkers;
    } else {
        // From bench --mixed (docs/results/bench.txt): beside the hub, BLAS
        // saturates early. On the M5 Max Studio four workers give most of
        // the CPU's rate (801 of 876 evals/s alone) and keep the hub at 96%;
        // more add little and take cores from the desktop. On an M4 mini
        // two give 95% of ten and keep its hub at 95%.
        workers = machine.local() ? std::min(4, cores / 3) : std::min(2, cores - 2);
    }
    return std::max(1, std::min(workers, sliceGames));
}

// ---- fingerprints ---------------------------------------------------------------

namespace {

#ifdef __APPLE__
std::string sysctlString(const char* name) {
    size_t size = 0;
    if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) return "";
    std::string value(size, '\0');
    if (sysctlbyname(name, value.data(), &size, nullptr, 0) != 0) return "";
    value.resize(std::strlen(value.c_str()));
    return value;
}
#endif

// Commas and semicolons separate a fingerprint's parts and the manifest's
// columns: keep them out of the parts this file builds.
std::string tidy(std::string s) {
    for (char& c : s)
        if (c == ',' || c == ';' || c == '\n' || c == '\r' || c == '\t') c = ' ';
    return trim(s);
}

std::string cpuModel() {
#ifdef __APPLE__
    const std::string brand = sysctlString("machdep.cpu.brand_string");
    if (!brand.empty()) return tidy(brand);
#else
    std::ifstream in("/proc/cpuinfo");
    for (std::string line; std::getline(in, line);) {
        if (line.rfind("model name", 0) != 0) continue;
        const size_t colon = line.find(':');
        if (colon != std::string::npos && !tidy(line.substr(colon + 1)).empty())
            return tidy(line.substr(colon + 1));
    }
#endif
    struct utsname u;
    return uname(&u) == 0 ? tidy(u.machine) : "unknown cpu";
}

std::string osVersion() {
#ifdef __APPLE__
    const std::string product = sysctlString("kern.osproductversion");
    const std::string build = sysctlString("kern.osversion");
    if (!product.empty())
        return tidy("macOS " + product + (build.empty() ? "" : " (" + build + ")"));
#endif
    struct utsname u;
    return uname(&u) == 0 ? tidy(std::string(u.sysname) + " " + u.release) : "unknown os";
}

} // namespace

std::string engineFingerprint(const std::string& backend) {
    checkBackendAvailable(backend);
    std::string f = backend + "; " + cpuModel() + "; " + osVersion();
    if (backend == "metal") f += "; " + metalDescription();
    return f;
}

} // namespace azgo
