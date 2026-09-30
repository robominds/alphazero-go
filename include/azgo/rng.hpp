// Deterministic randomness.
//
// AlphaZero is full of randomness: weight initialization, Dirichlet noise at
// the search root, sampling moves by visit count, replay-buffer sampling and
// symmetry augmentation. To make a training run reproducible from one seed,
// every random number in this project comes from this small, fully specified
// generator (xoshiro256**) rather than std::mt19937 + std::*_distribution,
// whose outputs differ between standard libraries. Independent streams (one
// per game, per phase, per generation) get their own seed via deriveSeed.
#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace azgo {

// Advances x and returns the next splitmix64 output.
uint64_t splitmix64(uint64_t& x);

// Mixes up to four integers into one well-distributed 64-bit seed.
uint64_t deriveSeed(uint64_t a, uint64_t b = 0, uint64_t c = 0, uint64_t d = 0);

// xoshiro256** generator, seeded by running splitmix64 on `seed`.
class Rng {
public:
    explicit Rng(uint64_t seed = 0);
    uint64_t next();
    double uniform();              // [0, 1), 53 random bits
    int below(int n);              // uniform in [0, n); asserts n > 0; unbiased (rejection)
    double normal();               // standard normal via Box-Muller (no cached second value)
    double gamma(double shape);    // Gamma(shape, 1); Marsaglia-Tsang, shape < 1 via boost
    std::array<uint64_t, 4> state() const;
    void setState(const std::array<uint64_t, 4>& s);
private:
    std::array<uint64_t, 4> s_;
};

// Symmetric Dirichlet sample of length n with concentration alpha > 0.
std::vector<double> dirichlet(Rng& rng, double alpha, int n);

} // namespace azgo
