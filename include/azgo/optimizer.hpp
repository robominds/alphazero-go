// How the network learns from a gradient. Plain SGD steps straight down the
// current minibatch's gradient, which is noisy; momentum keeps a running
// "velocity" so consistent directions build up speed while noise cancels.
// Weight decay adds a pull toward zero on the weights (L2 regularization),
// which keeps them small and the network smoother. The learning rate follows
// a schedule: large early for fast progress, smaller later for fine-tuning.
#pragma once
#include <string>
#include <utility>
#include <vector>
#include "azgo/network.hpp"

namespace azgo {

// SGD with momentum and L2 weight decay:
//   g' = grad + weightDecay * w   (only for params with decay == true)
//   v  = momentum * v + g'
//   w -= lr * v
class SgdMomentum {
public:
    SgdMomentum(Network& net, float momentum, float weightDecay);
    void step(Network& net, float lr);
    std::vector<std::vector<float>>& velocity() { return velocity_; }   // one per param, params() order
    const std::vector<std::vector<float>>& velocity() const { return velocity_; }
private:
    float momentum_, weightDecay_;
    std::vector<std::vector<float>> velocity_;
};

// Stepped learning-rate schedule keyed on generation, e.g. "0:0.02,50:0.002":
// lr = value of the last entry whose generation <= g.
class LrSchedule {
public:
    // Throws std::invalid_argument: empty, malformed, first entry not at 0,
    // generations not strictly increasing, or lr <= 0.
    static LrSchedule parse(const std::string& text);
    float at(int generation) const;
    std::string toString() const;   // canonical "g:lr,g:lr" form
private:
    std::vector<std::pair<int, float>> steps_;
};

} // namespace azgo
