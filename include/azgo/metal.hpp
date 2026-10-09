// The Metal GPU backend, as the rest of the program sees it. backend = metal
// runs the network's forward pass on the Apple GPU with hand-written compute
// kernels (src/metal/kernels.metal), compiled by the OS when the program
// first asks for them. It exists only in a build configured with
// -DAZ_USE_METAL=ON (macOS); every other build still compiles this header,
// answers "not built" and so can refuse backend = metal with a clear message.
//
// Everything Metal-specific stays in src/metal/: no Objective-C type appears
// in this header, which is plain C++17.
#pragma once
#include <cstddef>
#include <memory>
#include <string>

namespace azgo {

struct BatchNorm2d;   // azgo/layers.hpp
struct Conv2d;
struct Linear;
class Network;        // azgo/network.hpp
class BatchRunner;    // azgo/inference_hub.hpp

// True when this binary was built with -DAZ_USE_METAL=ON.
bool metalBuilt();

// True when the GPU can run the kernels: built with Metal, a Metal device
// of Apple GPU family 7 or later (simdgroup matrices; M1 or later), and every
// kernel compiled. The first call creates the process-wide Metal context,
// which compiles the shaders. When false and why is non-null, *why says what
// is missing, e.g. "not built: rebuild with -DAZ_USE_METAL=ON (macOS only)".
bool metalAvailable(std::string* why = nullptr);

// "Apple M5 Max, macOS 27.0.1, shaders 3f1c9a07": the device, the OS (which
// compiles the shaders) and a hash of the shader source. Two runs whose
// descriptions match ran the same GPU arithmetic. Without a usable GPU it is
// metalAvailable's reason.
std::string metalDescription();

#ifdef AZ_USE_METAL
// The GPU behind an InferenceHub: net's weights uploaded once, two slots
// (kInferenceSlots) of up to maxBatch positions each, a forward pass per
// batch in one command buffer, completed asynchronously. Throws
// std::runtime_error ("metal: inference needs N MB ...; lower
// inference_batch") when the slots would not fit the GPU's recommended
// working set, before allocating them.
std::unique_ptr<BatchRunner> makeMetalRunner(const Network& net, int maxBatch);

// Test probes: each copies host arrays in, runs one command buffer and copies
// the results out. They exist so tests (and bench) can check a kernel against
// its CPU counterpart; the program itself never calls them. Each throws
// std::runtime_error when the GPU is not usable or a command buffer fails.
namespace metal_probe {

// The GPU's name as Metal reports it, e.g. "Apple M5 Max".
std::string deviceName();

// The first kernel: data[0..n) = value. data holds n + tail floats; all of
// them are copied to the GPU and back, so a test can check the tail is untouched.
void fill(float* data, size_t n, float value, size_t tail = 0);

// Backend::gemm on the GPU (gemm_simd): row-major
// C[M x N] = alpha * op(A) * op(B) + beta * C, tightly packed; C is read only
// when beta != 0.
void gemm(bool transA, bool transB, int M, int N, int K, float alpha, const float* A, const float* B,
          float beta, float* C);

// The forward layers, each against its CPU counterpart (Network::forward's
// loops, Backend::im2col, the layer objects of azgo/layers.hpp). Activations
// are channel-major [C][B * n * n] as on the CPU.
void transposeInput(const float* input, int B, int planes, int area, float* out);  // [B][P][A] -> [P][B][A]
void im2col3(const float* in, int C, int B, int n, float* col);              // Backend::im2col, k = 3
void convForward(const Conv2d& conv, const float* x, int B, int n, float* y);  // Conv2d::forward
void bnInfer(const BatchNorm2d& bn, const float* x, int M, float* y);  // BatchNorm2d::forwardInfer
void relu(float* x, size_t count);                                     // reluForward
void addInto(float* dst, const float* src, size_t count);              // dst += src
void policyFlatten(const float* act, int B, int A, float* flat);       // [2][B][A] -> [B][2A]
void linearForward(const Linear& fc, const float* x, int B, float* y);  // Linear::forward
void tanhInPlace(float* x, size_t count);                               // x = tanh(x)

// The whole network's inference forward on the GPU (GpuNetwork): weights
// uploaded once, one workspace for up to maxBatch positions. infer has
// Network::infer's arguments; ownership needs a network with the head.
class InferNet {
public:
    InferNet(const Network& net, int maxBatch);
    ~InferNet();
    InferNet(const InferNet&) = delete;
    InferNet& operator=(const InferNet&) = delete;
    void infer(const float* input, int B, float* logits, float* values, float* ownership = nullptr);
    size_t bytes() const;   // the GPU memory of the activations
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Uploads net into weights laid out for layoutNet (ParamLayout::of): throws
// std::logic_error naming the first tensor the layout does not know.
void checkUpload(const Network& layoutNet, const Network& net);

} // namespace metal_probe
#endif

} // namespace azgo
