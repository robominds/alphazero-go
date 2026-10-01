#include "azgo/elo.hpp"
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <set>
#include <sstream>
#include <stdexcept>

namespace azgo {

void ResultsTable::addGame(int a, int b, double scoreA) { addMatch(a, b, 1, scoreA); }

void ResultsTable::addMatch(int a, int b, int games, double scoreA) {
    assert(a != b);
    assert(games > 0 && scoreA >= 0 && scoreA <= games);
    if (a > b) {
        // Store every pair once, as (smaller, larger): b's score is what a didn't score.
        std::swap(a, b);
        scoreA = games - scoreA;
    }
    auto& entry = table_[{a, b}];
    entry.first += games;
    entry.second += scoreA;
}

std::vector<ResultsTable::Pair> ResultsTable::pairs() const {
    std::vector<Pair> out;
    for (const auto& kv : table_)   // std::map iterates in (a, b) order
        out.push_back({kv.first.first, kv.first.second, kv.second.first, kv.second.second});
    return out;
}

std::vector<int> ResultsTable::players() const {
    std::set<int> ids;
    for (const auto& kv : table_) {
        ids.insert(kv.first.first);
        ids.insert(kv.first.second);
    }
    return std::vector<int>(ids.begin(), ids.end());
}

std::string ResultsTable::toCsv() const {
    std::string out = "a,b,games,score_a\n";
    char line[96];
    for (const Pair& p : pairs()) {
        std::snprintf(line, sizeof line, "%d,%d,%d,%.9g\n", p.a, p.b, p.games, p.scoreA);
        out += line;
    }
    return out;
}

namespace {

bool parseIntField(const std::string& s, int& out) {
    if (s.empty()) return false;
    errno = 0;
    char* end = nullptr;
    long v = std::strtol(s.c_str(), &end, 10);
    if (errno == ERANGE || *end != '\0' || v < -2147483647L - 1 || v > 2147483647L) return false;
    out = static_cast<int>(v);
    return true;
}

bool parseDoubleField(const std::string& s, double& out) {
    if (s.empty()) return false;
    errno = 0;
    char* end = nullptr;
    out = std::strtod(s.c_str(), &end);
    return errno != ERANGE && *end == '\0' && std::isfinite(out);
}

} // namespace

ResultsTable ResultsTable::fromCsv(const std::string& csv) {
    ResultsTable table;
    std::istringstream in(csv);
    std::string line;
    int lineNo = 0;
    auto fail = [&](const std::string& why) {
        throw std::runtime_error("results csv line " + std::to_string(lineNo) + ": " + why);
    };
    while (std::getline(in, line)) {
        ++lineNo;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (lineNo == 1) {
            if (line != "a,b,games,score_a") fail("expected header 'a,b,games,score_a'");
            continue;
        }
        if (line.empty()) continue;
        std::vector<std::string> f;
        std::istringstream fields(line);
        std::string field;
        while (std::getline(fields, field, ',')) f.push_back(field);
        int a = 0, b = 0, games = 0;
        double score = 0;
        if (f.size() != 4) fail("expected 4 fields");
        if (!parseIntField(f[0], a) || !parseIntField(f[1], b) || !parseIntField(f[2], games) ||
            !parseDoubleField(f[3], score))
            fail("malformed number");
        // Generations are >= 0 and bots < 0, so any int is a player.
        if (a == b) fail("players must be distinct");
        if (games <= 0 || score < 0 || score > games) fail("score_a must be in [0, games]");
        table.addMatch(a, b, games, score);
    }
    if (lineNo == 0) throw std::runtime_error("results csv line 1: missing header");
    return table;
}

std::map<int, double> fitBradleyTerry(const ResultsTable& results, int anchor) {
    std::vector<ResultsTable::Pair> pairs = results.pairs();

    // 1. Players connected to the anchor (BFS over played pairs). Anyone else
    //    has no rating relative to the anchor.
    std::map<int, std::vector<int>> neighbors;
    for (const auto& p : pairs) {
        neighbors[p.a].push_back(p.b);
        neighbors[p.b].push_back(p.a);
    }
    if (!neighbors.count(anchor)) return {{anchor, 0.0}};
    std::map<int, int> index;   // generation -> dense index
    std::vector<int> ids;
    std::deque<int> queue{anchor};
    index[anchor] = 0;
    ids.push_back(anchor);
    while (!queue.empty()) {
        int p = queue.front();
        queue.pop_front();
        for (int q : neighbors[p]) {
            if (index.count(q)) continue;
            index[q] = static_cast<int>(ids.size());
            ids.push_back(q);
            queue.push_back(q);
        }
    }

    // 2. Pair counts plus one virtual drawn game per played pair: n_ij + 1
    //    games, s_ij + 0.5 score. Without it a player who never lost would have
    //    an infinite maximum-likelihood rating.
    struct Edge { int i, j; double games; };
    std::vector<Edge> edges;
    std::vector<double> wins(ids.size(), 0.0);   // W_i, including virtual halves
    for (const auto& p : pairs) {
        auto it = index.find(p.a);
        if (it == index.end()) continue;         // other component (then b is too)
        int i = it->second, j = index.at(p.b);
        double n = p.games + 1.0;
        double si = p.scoreA + 0.5;
        wins[i] += si;
        wins[j] += n - si;
        edges.push_back({i, j, n});
    }

    // 3. MM iteration (Hunter 2004): gamma_i <- W_i / sum_j n_ij / (gamma_i + gamma_j).
    //    Each sweep increases the likelihood; rescaling keeps gamma_anchor = 1.
    const size_t m = ids.size();
    std::vector<double> gamma(m, 1.0), denom(m), next(m);
    for (int sweep = 0; sweep < 100000; ++sweep) {
        std::fill(denom.begin(), denom.end(), 0.0);
        for (const Edge& e : edges) {
            double d = e.games / (gamma[e.i] + gamma[e.j]);
            denom[e.i] += d;
            denom[e.j] += d;
        }
        for (size_t i = 0; i < m; ++i) next[i] = wins[i] / denom[i];
        double scale = next[0];   // index 0 is the anchor
        double maxChange = 0;
        for (size_t i = 0; i < m; ++i) {
            next[i] /= scale;
            maxChange = std::max(maxChange, std::fabs(std::log(next[i] / gamma[i])));
        }
        gamma.swap(next);
        if (maxChange < 1e-10) break;
    }

    std::map<int, double> elo;
    for (size_t i = 0; i < m; ++i) elo[ids[i]] = 400.0 * std::log10(gamma[i]);
    return elo;
}

std::pair<double, double> wilsonInterval(double score, int games, double z) {
    if (games <= 0) return {0.0, 1.0};
    double n = games;
    double p = score / n;
    double z2 = z * z;
    // Center pulled toward 1/2 and a half-width that stays positive at p = 0
    // or 1: unlike p +- z*sqrt(p(1-p)/n), it behaves for small samples.
    double denom = 1.0 + z2 / n;
    double center = (p + z2 / (2.0 * n)) / denom;
    double half = z * std::sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom;
    return {std::max(0.0, center - half), std::min(1.0, center + half)};
}

} // namespace azgo
