#include "azgo/config.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include "azgo/game.hpp"
#include "azgo/optimizer.hpp"
#include "azgo/parallel.hpp"

namespace azgo {

namespace {

std::string trim(const std::string& s) {
    const char* ws = " \t\r\n\f\v";
    size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

std::string quoted(const std::string& v) { return "'" + v + "'"; }

// ---- strict value parsers: return "" on success, otherwise the reason ----

// The WHOLE string must be a base-10 integer in [lo, hi]. strtoll alone would
// accept "12abc" (stopping at 'a') and silently clamp on overflow.
std::string parseInt(const std::string& v, long long lo, long long hi, int& out) {
    std::string reason = "expected an integer in [" + std::to_string(lo) + ", " +
                         std::to_string(hi) + "], got " + quoted(v);
    if (v.empty()) return reason;
    errno = 0;
    char* end = nullptr;
    long long x = std::strtoll(v.c_str(), &end, 10);
    if (errno == ERANGE || *end != '\0' || x < lo || x > hi) return reason;
    out = static_cast<int>(x);
    return "";
}

std::string parseU64(const std::string& v, uint64_t& out) {
    std::string reason = "expected an unsigned 64-bit integer, got " + quoted(v);
    // strtoull accepts a leading '-' and wraps it around; require a digit first.
    if (v.empty() || v[0] < '0' || v[0] > '9') return reason;
    errno = 0;
    char* end = nullptr;
    unsigned long long x = std::strtoull(v.c_str(), &end, 10);
    if (errno == ERANGE || *end != '\0') return reason;
    out = static_cast<uint64_t>(x);
    return "";
}

// A finite number in the interval lo..hi; loOpen / hiOpen exclude the bound.
std::string parseFloat(const std::string& v, double lo, double hi, float& out,
                       bool loOpen = false, bool hiOpen = false) {
    char range[96];
    std::snprintf(range, sizeof range, "%c%g, %g%c", loOpen ? '(' : '[', lo, hi,
                  hiOpen ? ')' : ']');
    std::string reason = std::string("expected a number in ") + range + ", got " + quoted(v);
    if (v.empty()) return reason;
    errno = 0;
    char* end = nullptr;
    double x = std::strtod(v.c_str(), &end);
    // ERANGE catches "1e999"; isfinite catches "nan" and "inf".
    if (errno == ERANGE || *end != '\0' || !std::isfinite(x)) return reason;
    if (x < lo || x > hi || (loOpen && x == lo) || (hiOpen && x == hi)) return reason;
    out = static_cast<float>(x);
    return "";
}

std::string parseBool(const std::string& v, bool& out) {
    if (v == "true" || v == "on" || v == "1") { out = true; return ""; }
    if (v == "false" || v == "off" || v == "0") { out = false; return ""; }
    return "expected true|false|on|off|1|0, got " + quoted(v);
}

// Colors alternate game by game, so an odd count would favor one side.
std::string parseEvenGames(const std::string& v, int& out) {
    int games = 0;
    std::string reason = parseInt(v, 2, 10000, games);
    if (!reason.empty()) return reason;
    if (games % 2 != 0) return "expected an even number (colors alternate), got " + quoted(v);
    out = games;
    return "";
}

// "100" or "50, 200": the bots' simulation counts, each in [1, 100000] and
// all different (bots are named by their count, e.g. "bot100"). An empty
// value means no bots. Splitting by hand, rather than with getline, keeps the
// empty item of "100," so that it is rejected.
std::string parseBotList(const std::string& v, std::vector<int>& out) {
    std::vector<int> list;
    for (size_t start = 0; !v.empty();) {
        size_t comma = v.find(',', start);
        // const: a non-const std::string would make quoted() below pick std::quoted by ADL.
        const std::string item =
            trim(v.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        int sims = 0;
        std::string reason = parseInt(item, 1, 100000, sims);
        if (!reason.empty()) return "bot " + std::to_string(list.size() + 1) + ": " + reason;
        if (std::find(list.begin(), list.end(), sims) != list.end())
            return "duplicate entry " + quoted(item) + " (each bot needs its own simulation count)";
        list.push_back(sims);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    out = list;
    return "";
}

std::string printBotList(const std::vector<int>& list) {
    std::string out;
    for (size_t i = 0; i < list.size(); ++i) out += (i ? "," : "") + std::to_string(list[i]);
    return out;
}

// The shortest "%.Ng" (N = 6..9) that reads back as exactly the same float:
// 0.9f prints as "0.9", not "0.899999976". Nine digits always round-trip.
std::string formatFloat(float x) {
    char buf[32];
    for (int digits = 6; digits <= 9; ++digits) {
        std::snprintf(buf, sizeof buf, "%.*g", digits, static_cast<double>(x));
        if (std::strtof(buf, nullptr) == x) break;
    }
    return buf;
}

// Table helpers, instantiated per field so each table line stays short.
template <int Config::*Field, long long Lo, long long Hi>
std::string intKey(Config& c, const std::string& v) { return parseInt(v, Lo, Hi, c.*Field); }
template <int Config::*Field>
std::string printInt(const Config& c) { return std::to_string(c.*Field); }
template <float Config::*Field>
std::string printFloat(const Config& c) { return formatFloat(c.*Field); }

struct KeyEntry {
    const char* key;
    bool architecture;   // fixes the network's shape: may not change on resume
    std::string (*parse)(Config& cfg, const std::string& value);   // "" or reason
    std::string (*print)(const Config& cfg);
};

// THE list of settings. Order here is the order of configToString.
const KeyEntry kKeys[] = {
    // run
    {"seed", false, [](Config& c, const std::string& v) { return parseU64(v, c.seed); },
     [](const Config& c) { return std::to_string(c.seed); }},
    {"generations", false, intKey<&Config::generations, 1, 1000000>, printInt<&Config::generations>},
    {"workers", false, intKey<&Config::workers, 0, 256>, printInt<&Config::workers>},
    {"backend", false,
     [](Config& c, const std::string& v) -> std::string {
         if (v != "naive" && v != "blas") return "expected naive|blas, got " + quoted(v);
         c.backend = v;
         return "";
     },
     [](const Config& c) { return c.backend; }},
    // game
    {"board_size", true, intKey<&Config::boardSize, kMinBoardSize, kMaxBoardSize>, printInt<&Config::boardSize>},
    {"komi", false, [](Config& c, const std::string& v) { return parseFloat(v, -500, 500, c.komi); },
     printFloat<&Config::komi>},
    {"max_moves", false, intKey<&Config::maxMoves, 0, 100000>, printInt<&Config::maxMoves>},
    // network
    {"history", true, intKey<&Config::history, 1, 16>, printInt<&Config::history>},
    {"blocks", true, intKey<&Config::blocks, 0, 40>, printInt<&Config::blocks>},
    {"filters", true, intKey<&Config::filters, 1, 512>, printInt<&Config::filters>},
    {"value_hidden", true, intKey<&Config::valueHidden, 1, 1024>, printInt<&Config::valueHidden>},
    // self-play and search
    {"games_per_generation", false, intKey<&Config::gamesPerGeneration, 1, 1000000>,
     printInt<&Config::gamesPerGeneration>},
    {"simulations", false, intKey<&Config::simulations, 1, 100000>, printInt<&Config::simulations>},
    {"mcts_batch", false, intKey<&Config::mctsBatch, 1, 256>, printInt<&Config::mctsBatch>},
    {"c_puct", false, [](Config& c, const std::string& v) { return parseFloat(v, 0, 100, c.cPuct, true); },
     printFloat<&Config::cPuct>},
    {"dirichlet_scale", false,
     [](Config& c, const std::string& v) { return parseFloat(v, 0, 1000, c.dirichletScale); },
     printFloat<&Config::dirichletScale>},
    {"noise_fraction", false,
     [](Config& c, const std::string& v) { return parseFloat(v, 0, 1, c.noiseFraction); },
     printFloat<&Config::noiseFraction>},
    {"temperature_moves", false, intKey<&Config::temperatureMoves, 0, 100000>,
     printInt<&Config::temperatureMoves>},
    // training
    {"buffer_positions", false, intKey<&Config::bufferPositions, 1, 100000000>,
     printInt<&Config::bufferPositions>},
    {"train_steps", false, intKey<&Config::trainSteps, 0, 1000000>, printInt<&Config::trainSteps>},
    {"batch_size", false, intKey<&Config::batchSize, 1, 4096>, printInt<&Config::batchSize>},
    {"lr_schedule", false,
     [](Config& c, const std::string& v) -> std::string {
         try {
             LrSchedule::parse(v);   // validate only; the text is stored as written
         } catch (const std::invalid_argument& e) {
             return std::string("invalid schedule ") + quoted(v) + ": " + e.what();
         }
         c.lrSchedule = v;
         return "";
     },
     [](const Config& c) { return c.lrSchedule; }},
    {"momentum", false,
     [](Config& c, const std::string& v) { return parseFloat(v, 0, 1, c.momentum, false, true); },
     printFloat<&Config::momentum>},
    {"weight_decay", false,
     [](Config& c, const std::string& v) { return parseFloat(v, 0, 1, c.weightDecay); },
     printFloat<&Config::weightDecay>},
    // arena
    {"arena_every", false, intKey<&Config::arenaEvery, 0, 1000000>, printInt<&Config::arenaEvery>},
    {"arena_games", false, [](Config& c, const std::string& v) { return parseEvenGames(v, c.arenaGames); },
     printInt<&Config::arenaGames>},
    {"arena_opponents", false, intKey<&Config::arenaOpponents, 0, 32>, printInt<&Config::arenaOpponents>},
    {"bot_simulations", false,
     [](Config& c, const std::string& v) { return parseBotList(v, c.botSimulations); },
     [](const Config& c) { return printBotList(c.botSimulations); }},
    {"calibration_games", false,
     [](Config& c, const std::string& v) { return parseEvenGames(v, c.calibrationGames); },
     printInt<&Config::calibrationGames>},
    {"promote_threshold", false,
     [](Config& c, const std::string& v) { return parseFloat(v, 0.5, 1, c.promoteThreshold); },
     printFloat<&Config::promoteThreshold>},
    {"arena_simulations", false, intKey<&Config::arenaSimulations, 1, 100000>,
     printInt<&Config::arenaSimulations>},
    {"arena_temperature_moves", false, intKey<&Config::arenaTemperatureMoves, 0, 100000>,
     printInt<&Config::arenaTemperatureMoves>},
    {"gating", false, [](Config& c, const std::string& v) { return parseBool(v, c.gating); },
     [](const Config& c) { return std::string(c.gating ? "true" : "false"); }},
    {"gating_threshold", false,
     [](Config& c, const std::string& v) { return parseFloat(v, 0.5, 1, c.gatingThreshold); },
     printFloat<&Config::gatingThreshold>},
    // output
    {"sgf_per_generation", false, intKey<&Config::sgfPerGeneration, 0, 1000>,
     printInt<&Config::sgfPerGeneration>},
};

const KeyEntry* findKey(const std::string& key) {
    for (const KeyEntry& e : kKeys)
        if (key == e.key) return &e;
    return nullptr;
}

// "file:12" for file text, "--set" for command-line overrides (line 0).
std::string location(const std::string& source, int line) {
    return line > 0 ? source + ":" + std::to_string(line) : source;
}

// Applies one "key = value" line (comment already stripped, non-blank).
// firstLine (may be null) remembers where each key was set, to catch repeats.
void applyLine(Config& cfg, const std::string& text, const std::string& source, int line,
               std::map<std::string, int>* firstLine) {
    size_t eq = text.find('=');
    std::string key = eq == std::string::npos ? "" : trim(text.substr(0, eq));
    if (key.empty()) throw ConfigError(location(source, line) + ": expected key = value");
    std::string value = trim(text.substr(eq + 1));
    std::string where = location(source, line) + ": " + key + ": ";

    const KeyEntry* entry = findKey(key);
    if (!entry) throw ConfigError(where + "unknown key");
    if (firstLine) {
        auto it = firstLine->find(key);
        if (it != firstLine->end())
            throw ConfigError(where + "duplicate key (first set on line " +
                              std::to_string(it->second) + ")");
        (*firstLine)[key] = line;
    }
    std::string reason = entry->parse(cfg, value);
    if (!reason.empty()) throw ConfigError(where + reason);
}

} // namespace

NetConfig Config::netConfig() const {
    NetConfig net;
    net.boardSize = boardSize;
    net.history = history;
    net.blocks = blocks;
    net.filters = filters;
    net.valueHidden = valueHidden;
    return net;
}

int Config::effectiveMaxMoves() const {
    return maxMoves > 0 ? maxMoves : defaultMaxMoves(boardSize);
}

int Config::effectiveWorkers() const { return resolveWorkers(workers); }

Config parseConfigText(const std::string& text, const std::string& sourceName, const Config& base) {
    Config cfg = base;
    std::map<std::string, int> firstLine;
    std::istringstream in(text);
    std::string raw;
    int lineNo = 0;
    while (std::getline(in, raw)) {
        ++lineNo;
        size_t hash = raw.find('#');
        if (hash != std::string::npos) raw.erase(hash);
        std::string line = trim(raw);   // also strips the '\r' of CRLF files
        if (line.empty()) continue;
        applyLine(cfg, line, sourceName, lineNo, &firstLine);
    }
    return cfg;
}

Config loadConfigFile(const std::string& path, const Config& base) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw ConfigError(path + ": cannot open config file");
    std::ostringstream text;
    text << in.rdbuf();
    return parseConfigText(text.str(), path, base);
}

void applyOverride(Config& cfg, const std::string& keyEqualsValue) {
    applyLine(cfg, trim(keyEqualsValue), "--set", 0, nullptr);
}

std::string configToString(const Config& cfg) {
    std::string out;
    for (const KeyEntry& e : kKeys) out += std::string(e.key) + " = " + e.print(cfg) + "\n";
    return out;
}

std::vector<std::string> architectureDiff(const Config& a, const Config& b) {
    std::vector<std::string> keys;
    for (const KeyEntry& e : kKeys)
        if (e.architecture && e.print(a) != e.print(b)) keys.push_back(e.key);
    return keys;
}

std::vector<std::string> resumeLockedDiff(const Config& a, const Config& b) {
    std::vector<std::string> keys = architectureDiff(a, b);
    if (a.botSimulations != b.botSimulations) keys.push_back("bot_simulations");
    return keys;
}

} // namespace azgo
