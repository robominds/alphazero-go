#include <cassert>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include "azgo/elo.hpp"

using namespace azgo;

static bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

static double expectedScore(double eloA, double eloB) {
    return 1.0 / (1.0 + std::pow(10.0, (eloB - eloA) / 400.0));
}

static void test_add_normalizes_order() {
    ResultsTable x, y;
    x.addGame(5, 2, 1.0);   // 5 beat 2
    y.addGame(2, 5, 0.0);   // 2 lost to 5
    auto px = x.pairs(), py = y.pairs();
    assert(px.size() == 1 && py.size() == 1);
    assert(px[0].a == 2 && px[0].b == 5 && px[0].games == 1 && px[0].scoreA == 0.0);
    assert(py[0].a == 2 && py[0].b == 5 && py[0].games == 1 && py[0].scoreA == 0.0);

    x.addGame(2, 5, 0.5);
    x.addMatch(5, 2, 10, 7.0);   // 5 scored 7 of 10, so 2 scored 3
    auto p = x.pairs();
    assert(p.size() == 1 && p[0].games == 12 && near(p[0].scoreA, 3.5, 1e-12));
    x.addGame(0, 9, 1.0);
    assert((x.players() == std::vector<int>{0, 2, 5, 9}));
    p = x.pairs();
    assert(p[0].a == 0 && p[0].b == 9 && p[1].a == 2 && p[1].b == 5);   // sorted
    assert(!x.empty() && ResultsTable().empty());
}

static void test_equal_results_equal_ratings() {
    ResultsTable t;
    t.addMatch(0, 1, 10, 5.0);
    auto elo = fitBradleyTerry(t);
    assert(elo.size() == 2);
    assert(near(elo.at(0), 0.0, 1e-9) && near(elo.at(1), 0.0, 1e-9));
}

static void test_recovers_known_ladder() {
    // True ratings 0, 200, 400; every pair plays 100000 games scored exactly
    // at the Bradley-Terry expectation.
    std::map<int, double> truth{{0, 0.0}, {1, 200.0}, {2, 400.0}};
    ResultsTable t;
    const int games = 100000;
    for (int a = 0; a < 3; ++a)
        for (int b = a + 1; b < 3; ++b)
            t.addMatch(a, b, games, games * expectedScore(truth[a], truth[b]));
    auto elo = fitBradleyTerry(t, 0);
    assert(elo.size() == 3);
    assert(elo.at(0) == 0.0);   // anchor pinned
    for (int p = 0; p < 3; ++p) assert(near(elo.at(p), truth[p], 2.0));

    // Order of recording does not matter.
    ResultsTable r;
    for (int a = 2; a >= 0; --a)
        for (int b = 2; b > a; --b)
            r.addMatch(b, a, games, games * expectedScore(truth[b], truth[a]));
    auto elo2 = fitBradleyTerry(r, 0);
    for (int p = 0; p < 3; ++p) assert(near(elo.at(p), elo2.at(p), 1e-6));
}

static void test_perfect_record_is_finite() {
    ResultsTable t;
    t.addMatch(0, 1, 20, 0.0);   // 1 won all 20
    auto elo = fitBradleyTerry(t);
    // With the virtual draw: 1 scored 20.5 of 21 -> gamma = 41 -> ~645 Elo.
    assert(std::isfinite(elo.at(1)) && elo.at(1) > 400.0);
    assert(near(elo.at(1), 400.0 * std::log10(41.0), 1e-6));
}

static void test_disconnected_players_omitted() {
    ResultsTable t;
    t.addMatch(0, 1, 10, 4.0);
    t.addMatch(1, 2, 10, 6.0);
    t.addMatch(5, 6, 10, 3.0);   // no path to generation 0
    auto elo = fitBradleyTerry(t);
    assert(elo.size() == 3 && elo.count(0) && elo.count(1) && elo.count(2));
    assert(!elo.count(5) && !elo.count(6));

    ResultsTable only;
    only.addMatch(3, 4, 2, 1.0);   // anchor never played
    auto lone = fitBradleyTerry(only, 0);
    assert(lone.size() == 1 && lone.at(0) == 0.0);
    assert(fitBradleyTerry(ResultsTable{}).size() == 1);
}

static void test_csv_round_trip() {
    ResultsTable t;
    t.addMatch(0, 1, 20, 12.5);
    t.addMatch(3, 1, 40, 21.0);
    t.addGame(0, 3, 0.5);
    std::string csv = t.toCsv();
    assert(csv == "a,b,games,score_a\n0,1,20,12.5\n0,3,1,0.5\n1,3,40,19\n");
    ResultsTable back = ResultsTable::fromCsv(csv);
    assert(back.toCsv() == csv);
    auto p = back.pairs();
    assert(p.size() == 3 && p[2].a == 1 && p[2].b == 3 && p[2].games == 40 && p[2].scoreA == 19.0);
    // CRLF line endings and an empty table are fine.
    assert(ResultsTable::fromCsv("a,b,games,score_a\r\n0,1,2,1\r\n").pairs().size() == 1);
    assert(ResultsTable::fromCsv("a,b,games,score_a\n").empty());
}

static bool csvThrows(const std::string& csv, const std::string& expectInMessage) {
    try {
        ResultsTable::fromCsv(csv);
    } catch (const std::runtime_error& e) {
        return std::string(e.what()).find(expectInMessage) != std::string::npos;
    }
    return false;
}

static void test_csv_rejects_bad_input() {
    assert(csvThrows("", "line 1"));
    assert(csvThrows("a,b,games,score\n0,1,2,1\n", "line 1"));
    assert(csvThrows("a,b,games,score_a\n0,1,2\n", "line 2"));
    assert(csvThrows("a,b,games,score_a\n0,1,2,1\n0,x,2,1\n", "line 3"));
    assert(csvThrows("a,b,games,score_a\n0,1,2,3\n", "line 2"));    // score > games
    assert(csvThrows("a,b,games,score_a\n1,1,2,1\n", "line 2"));    // a == b
    assert(csvThrows("a,b,games,score_a\n0,1,0,0\n", "line 2"));    // no games
}

static void test_wilson_interval() {
    auto ci = wilsonInterval(20, 40);
    assert(near(ci.first, 0.352, 1e-3) && near(ci.second, 0.648, 1e-3));
    ci = wilsonInterval(40, 40);
    assert(near(ci.second, 1.0, 1e-12) && ci.second <= 1.0);
    assert(near(ci.first, 0.912, 1e-3));
    ci = wilsonInterval(0, 40);
    assert(ci.first == 0.0 && near(ci.second, 0.088, 1e-3));
    ci = wilsonInterval(0, 0);
    assert(ci.first == 0.0 && ci.second == 1.0);
}

int main() {
    test_add_normalizes_order();
    test_equal_results_equal_ratings();
    test_recovers_known_ladder();
    test_perfect_record_is_finite();
    test_disconnected_players_omitted();
    test_csv_round_trip();
    test_csv_rejects_bad_input();
    test_wilson_interval();
    std::printf("test_elo: all tests passed\n");
    return 0;
}
