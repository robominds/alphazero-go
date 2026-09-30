#include "azgo/rng.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <initializer_list>

namespace azgo {

namespace {

uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

constexpr double kTwoPi = 6.283185307179586476925286766559;

} // namespace

uint64_t splitmix64(uint64_t& x) {
    // Weyl sequence step followed by an invertible bit mixer (Steele et al.).
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

uint64_t deriveSeed(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    // Fold the inputs in one at a time: h = mix(h ^ v). Because each step
    // mixes before the next input arrives, the result depends on the order
    // of the inputs, so (1, 2, 3, 4) and (4, 3, 2, 1) name different streams.
    uint64_t h = 0x243F6A8885A308D3ULL;  // arbitrary nonzero start (digits of pi)
    for (uint64_t v : {a, b, c, d}) {
        uint64_t x = h ^ v;
        h = splitmix64(x);
    }
    return h;
}

Rng::Rng(uint64_t seed) {
    // xoshiro's state must not be all zero; four consecutive splitmix64
    // outputs never are, and they spread even small seeds (0, 1, 2) apart.
    uint64_t x = seed;
    for (auto& w : s_) w = splitmix64(x);
}

uint64_t Rng::next() {
    // xoshiro256** (Blackman & Vigna, 2018).
    const uint64_t result = rotl(s_[1] * 5, 7) * 9;
    const uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
}

double Rng::uniform() {
    // The top 53 bits fill a double's mantissa exactly: k / 2^53, k < 2^53.
    return static_cast<double>(next() >> 11) * 0x1.0p-53;
}

int Rng::below(int n) {
    assert(n > 0);
    // Lemire's multiply-shift: the high 32 bits of x * n are uniform in
    // [0, n) except for a small bias, which we remove by rejecting the
    // (2^32 mod n) lowest products.
    const uint32_t range = static_cast<uint32_t>(n);
    uint64_t m = (next() >> 32) * range;
    uint32_t low = static_cast<uint32_t>(m);
    if (low < range) {
        const uint32_t threshold = (0u - range) % range;  // 2^32 mod n
        while (low < threshold) {
            m = (next() >> 32) * range;
            low = static_cast<uint32_t>(m);
        }
    }
    return static_cast<int>(m >> 32);
}

double Rng::normal() {
    const double u1 = 1.0 - uniform();  // (0, 1], so log(u1) is finite
    const double u2 = uniform();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(kTwoPi * u2);
}

double Rng::gamma(double shape) {
    assert(shape > 0.0);
    if (shape < 1.0) {
        // Boost: if X ~ Gamma(shape + 1) and U ~ Uniform(0, 1], then
        // X * U^(1/shape) ~ Gamma(shape). Marsaglia-Tsang needs shape >= 1.
        const double u = 1.0 - uniform();
        return gamma(shape + 1.0) * std::pow(u, 1.0 / shape);
    }
    // Marsaglia & Tsang (2000): transform a normal draw x into d * (1 + c x)^3
    // and accept with a squeeze test (cheap) or the exact log test.
    const double d = shape - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
        double x, v;
        do {
            x = normal();
            v = 1.0 + c * x;
        } while (v <= 0.0);
        v = v * v * v;
        const double u = uniform();
        const double x2 = x * x;
        if (u < 1.0 - 0.0331 * x2 * x2) return d * v;
        if (u > 0.0 && std::log(u) < 0.5 * x2 + d * (1.0 - v + std::log(v))) return d * v;
    }
}

std::array<uint64_t, 4> Rng::state() const {
    return s_;
}

void Rng::setState(const std::array<uint64_t, 4>& s) {
    s_ = s;
}

std::vector<double> dirichlet(Rng& rng, double alpha, int n) {
    assert(alpha > 0.0 && n > 0);
    // A Dirichlet sample is a vector of independent Gamma(alpha) draws
    // divided by their sum.
    std::vector<double> x(static_cast<size_t>(n));
    double sum = 0.0;
    for (double& v : x) {
        v = rng.gamma(alpha);
        sum += v;
    }
    if (!(sum > 0.0)) {
        // With a tiny alpha every Gamma draw can underflow to 0. The limit of
        // Dirichlet(alpha) as alpha -> 0 puts all mass on one component, so
        // return a one-hot vector at a uniformly random index.
        std::fill(x.begin(), x.end(), 0.0);
        x[static_cast<size_t>(rng.below(n))] = 1.0;
        return x;
    }
    for (double& v : x) v /= sum;
    return x;
}

} // namespace azgo
