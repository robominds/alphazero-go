// The building blocks of the ResNet, each with a hand-written backward pass.
// Backprop is the chain rule applied layer by layer in reverse: given dL/dy
// (how the loss changes with a layer's output), a layer produces dL/dx for
// the layer below and accumulates dL/dparameter for the optimizer. Every
// backward here reuses what its forward saved (im2col columns, normalized
// activations), which is why forward and backward share scratch buffers.
// Tensors are plain float arrays; the caller (Network/Workspace) owns them.
#pragma once
#include <cstddef>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/rng.hpp"

namespace azgo {

// All activations are channel-major: [channels][M] with M = B * n * n.
//
// Every operation takes an optional ThreadPool (null: one thread, as
// inference in search uses them). With a pool the work is cut into chunks
// fixed by the shapes alone (a channel, a block of gemm columns, a run of
// kElementChunk elements) and every output value is computed by exactly one
// chunk with the same operations in the same order as without a pool, so
// the bits do not depend on the pool or its size.

class ThreadPool;   // azgo/parallel.hpp

// Elements per chunk of an elementwise loop (ReLU, sums) on a pool.
constexpr size_t kElementChunk = size_t(1) << 15;

struct Conv2d {
    int cin = 0, cout = 0, k = 1;        // k is 1 or 3; 'same' padding; no bias (BN follows)
    std::vector<float> w, gw;            // [cout][cin * k * k]
    Conv2d() = default;
    Conv2d(int cin, int cout, int k);
    void init(Rng& rng);                 // He normal: stddev sqrt(2 / (cin * k * k))
    size_t colSize(int B, int n) const;  // floats needed for the im2col buffer (0 when k == 1)
    // x [cin][M] -> y [cout][M]. col is scratch of colSize(B, n) (unused if k == 1) and
    // must be passed unchanged to backward.
    void forward(const Backend& be, const float* x, int B, int n, float* col, float* y,
                 ThreadPool* pool = nullptr) const;
    // Accumulates gw += dW. Writes dx (overwrite) unless dx == nullptr.
    // dcol is scratch of colSize(B, n).
    void backward(const Backend& be, const float* x, const float* col, const float* dy,
                  int B, int n, float* dcol, float* dx, ThreadPool* pool = nullptr);
};

struct BatchNorm2d {
    static constexpr float kEps = 1e-5f;
    static constexpr float kMomentum = 0.1f;   // running = (1 - m) * running + m * batch
    int c = 0;
    std::vector<float> gamma, beta, ggamma, gbeta, runMean, runVar;  // gamma=1, beta=0, runVar=1 initially
    BatchNorm2d() = default;
    explicit BatchNorm2d(int c);
    // y = gamma * (x - runMean) / sqrt(runVar + eps) + beta
    void forwardInfer(const float* x, int M, float* y, ThreadPool* pool = nullptr) const;
    // Batch statistics (biased variance). Saves xhat [c][M] and invStd [c].
    // If updateRunning, updates runMean and runVar (unbiased variance M/(M-1)).
    void forwardTrain(const float* x, int M, float* y, float* xhat, float* invStd,
                      bool updateRunning, ThreadPool* pool = nullptr);
    // Accumulates ggamma, gbeta; writes dx (overwrite); dx may alias dy (Network relies on it).
    void backward(const float* dy, const float* xhat, const float* invStd, int M, float* dx,
                  ThreadPool* pool = nullptr);
};

struct Linear {
    int in = 0, out = 0;
    std::vector<float> w, b, gw, gb;     // w [out][in], b [out]
    Linear() = default;
    Linear(int in, int out);
    void init(Rng& rng);                 // He normal weights, zero bias
    void forward(const Backend& be, const float* x, int B, float* y) const;   // x [B][in], y [B][out]
    // Accumulates gw, gb. Writes dx (overwrite) unless dx == nullptr.
    void backward(const Backend& be, const float* x, const float* dy, int B, float* dx);
};

void reluForward(float* x, size_t count, ThreadPool* pool = nullptr);                   // in place
void reluBackward(const float* y, float* dy, size_t count, ThreadPool* pool = nullptr);  // dy *= (y > 0), y = relu output

} // namespace azgo
