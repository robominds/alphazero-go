#include "azgo/engines.hpp"
#include <cerrno>
#include <cstdlib>
#include <stdexcept>

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
                                        " needs a weight or a count, not both");
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

} // namespace azgo
