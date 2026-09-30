#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include "azgo/rng.hpp"

using namespace azgo;

static bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

static void test_splitmix64_known_outputs() {
    // Reference values of the published splitmix64 algorithm from x = 0.
    uint64_t x = 0;
    assert(splitmix64(x) == 0xE220A8397B1DCDAFULL);
    assert(splitmix64(x) == 0x6E789E6AA1B965F4ULL);
    assert(splitmix64(x) == 0x06C45D188009454FULL);
    assert(x == 3 * 0x9E3779B97F4A7C15ULL);  // advanced by the Weyl increment each call
}

static void test_same_seed_same_sequence() {
    Rng a(42), b(42);
    for (int i = 0; i < 1000; ++i) assert(a.next() == b.next());
}

static void test_different_seeds_differ() {
    Rng a(1), b(2);
    int equal = 0;
    for (int i = 0; i < 100; ++i) equal += (a.next() == b.next());
    assert(equal == 0);
}

static void test_state_round_trip_resumes_sequence() {
    Rng a(7);
    for (int i = 0; i < 10; ++i) a.next();
    std::array<uint64_t, 4> saved = a.state();
    std::vector<uint64_t> expected;
    for (int i = 0; i < 50; ++i) expected.push_back(a.next());

    Rng b(12345);  // different seed, then overwritten
    b.setState(saved);
    for (int i = 0; i < 50; ++i) assert(b.next() == expected[i]);
    assert(b.state() == a.state());
}

static void test_uniform_range_and_mean() {
    Rng rng(3);
    const int draws = 100000;
    double sum = 0.0;
    for (int i = 0; i < draws; ++i) {
        double u = rng.uniform();
        assert(u >= 0.0 && u < 1.0);
        sum += u;
    }
    // Standard error of the mean is sqrt(1/12 / 1e5) ~ 0.0009.
    assert(near(sum / draws, 0.5, 0.005));
}

static void test_below_is_uniform() {
    Rng rng(4);
    const int n = 7, draws = 70000;
    std::vector<int> counts(n, 0);
    for (int i = 0; i < draws; ++i) {
        int k = rng.below(n);
        assert(k >= 0 && k < n);
        ++counts[k];
    }
    const double expected = static_cast<double>(draws) / n;  // 10000
    for (int c : counts) assert(std::fabs(c - expected) <= 0.05 * expected);

    Rng one(5);
    for (int i = 0; i < 100; ++i) assert(one.below(1) == 0);
}

static void test_normal_moments() {
    Rng rng(5);
    const int draws = 200000;
    double sum = 0.0, sumSq = 0.0;
    for (int i = 0; i < draws; ++i) {
        double x = rng.normal();
        assert(std::isfinite(x));
        sum += x;
        sumSq += x * x;
    }
    double mean = sum / draws;
    double var = sumSq / draws - mean * mean;
    assert(near(mean, 0.0, 0.01));
    assert(near(var, 1.0, 0.02));
}

static void test_gamma_moments() {
    // Gamma(k, 1) has mean k and variance k. Small shapes (the Dirichlet
    // alpha on 19x19 is ~0.03) go through the shape < 1 boost path.
    const double shapes[] = {0.03, 0.5, 2.0, 7.5};
    Rng rng(6);
    for (double k : shapes) {
        const int draws = 200000;
        double sum = 0.0, sumSq = 0.0;
        for (int i = 0; i < draws; ++i) {
            double x = rng.gamma(k);
            assert(x >= 0.0 && std::isfinite(x));
            sum += x;
            sumSq += x * x;
        }
        double mean = sum / draws;
        double var = sumSq / draws - mean * mean;
        // Allow 5 standard errors. SE(mean) = sqrt(k / draws); the sample
        // variance of Gamma(k) has SE sqrt((mu4 - k^2) / draws) with fourth
        // central moment mu4 = 3k^2 + 6k. Tiny shapes are very skewed, so
        // their variance estimate is noisy (~3% SE at k = 0.03).
        double seMean = std::sqrt(k / draws);
        double seVar = std::sqrt((2.0 * k * k + 6.0 * k) / draws);
        assert(std::fabs(mean - k) <= 5.0 * seMean);
        assert(std::fabs(var - k) <= 5.0 * seVar);
    }
}

static void test_dirichlet() {
    Rng rng(8);
    const int n = 5, samples = 20000;
    std::vector<double> mean(n, 0.0);
    for (int s = 0; s < samples; ++s) {
        std::vector<double> d = dirichlet(rng, 0.3, n);
        assert(static_cast<int>(d.size()) == n);
        double total = 0.0;
        for (int i = 0; i < n; ++i) {
            assert(d[i] >= 0.0);
            total += d[i];
            mean[i] += d[i] / samples;
        }
        assert(near(total, 1.0, 1e-9));
    }
    for (double m : mean) assert(near(m, 1.0 / n, 0.01));

    // 19x19-sized noise with the small alpha used there.
    std::vector<double> big = dirichlet(rng, 0.03, 362);
    double total = 0.0;
    for (double v : big) { assert(v >= 0.0); total += v; }
    assert(near(total, 1.0, 1e-9));

    // With alpha = 1e-4 every Gamma draw usually underflows to 0, which
    // exercises the one-hot fallback; the result must still be a distribution.
    for (int s = 0; s < 100; ++s) {
        std::vector<double> d = dirichlet(rng, 1e-4, 5);
        double t = 0.0;
        for (double v : d) { assert(v >= 0.0); t += v; }
        assert(near(t, 1.0, 1e-9));
    }
}

static void test_derive_seed() {
    assert(deriveSeed(1, 2, 3, 4) == deriveSeed(1, 2, 3, 4));
    assert(deriveSeed(1, 2, 3, 4) != deriveSeed(4, 3, 2, 1));
    assert(deriveSeed(1, 2, 3, 4) != deriveSeed(2, 1, 3, 4));
    assert(deriveSeed(1, 2) != deriveSeed(2, 1));
    assert(deriveSeed(0) != deriveSeed(1));
    assert(deriveSeed(7) == deriveSeed(7, 0, 0, 0));  // defaults are zeros
    // Neighbouring game indices must give unrelated streams.
    Rng a(deriveSeed(99, 1, 0, 0)), b(deriveSeed(99, 1, 0, 1));
    assert(a.next() != b.next());
}

int main() {
    test_splitmix64_known_outputs();
    test_same_seed_same_sequence();
    test_different_seeds_differ();
    test_state_round_trip_resumes_sequence();
    test_uniform_range_and_mean();
    test_below_is_uniform();
    test_normal_moments();
    test_gamma_moments();
    test_dirichlet();
    test_derive_seed();
    std::printf("test_rng: all tests passed\n");
    return 0;
}
