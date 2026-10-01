// Measuring strength without an oracle. Networks play each other; every
// pairwise result is kept, and Bradley-Terry fits one strength per player
// such that P(i beats j) = g_i / (g_i + g_j). On the Elo scale a 200-point
// gap means the stronger player is expected to score about 76%.
// Refitting from the whole table is order-independent, so a resumed run
// reproduces the same ratings.
#pragma once
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace azgo {

// All arena results so far. Players are generation numbers (>= 0) and
// fixed-strength bots (-1, -2, ...: bot k is player -k).
class ResultsTable {
public:
    struct Pair { int a; int b; int games; double scoreA; };  // a < b; scoreA in [0, games]
    // scoreA: 1 win, 0.5 draw, 0 loss for player a. a != b; either order is
    // accepted and stored normalized as (min, max) with the score flipped as needed.
    void addGame(int a, int b, double scoreA);
    void addMatch(int a, int b, int games, double scoreA);    // many games at once
    std::vector<Pair> pairs() const;                          // sorted by (a, b)
    std::vector<int> players() const;                         // sorted
    bool empty() const { return table_.empty(); }
    std::string toCsv() const;                                // "a,b,games,score_a" header + rows
    static ResultsTable fromCsv(const std::string& csv);      // throws std::runtime_error
private:
    std::map<std::pair<int, int>, std::pair<int, double>> table_;  // (a,b) -> (games, scoreA)
};

// Maximum-likelihood Bradley-Terry ratings by the MM algorithm (Hunter 2004),
// on the Elo scale (400 * log10), with `anchor` pinned at 0. Each played pair
// gets one extra virtual game scored as a draw, so a player who has only won
// (or only lost) still gets a finite rating. Players unreachable from the
// anchor through played pairs are omitted.
std::map<int, double> fitBradleyTerry(const ResultsTable& results, int anchor = 0);

// 95% (z = 1.96) Wilson score interval for a win rate of score/games.
std::pair<double, double> wilsonInterval(double score, int games, double z = 1.96);

} // namespace azgo
