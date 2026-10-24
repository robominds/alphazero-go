// The CUDA GPU backend, as the rest of the program sees it. backend = cuda
// runs the network's inference pass on an NVIDIA GPU with hand-written fp32
// kernels (src/cuda/kernels.cu), compiled ahead of time to machine code for
// named architectures only, so no driver ever recompiles them. It is a
// self-play engine (selfplay_engines = pc/cuda:60): a BatchRunner behind the
// inference hub, exactly where the Metal runner sits on a Mac.
//
// It exists only in a build configured with -DAZ_USE_CUDA=ON (Linux); every
// other build still compiles this header, answers "not built" and so can
// refuse backend = cuda with a message that says how to get it. No CUDA type
// appears here: the header is plain C++17.
//
// Design: docs/superpowers/specs/2026-10-20-cuda-inference-backend-design.md.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "azgo/gpu_memory.hpp"
#include "azgo/network.hpp"

namespace azgo {

class BatchRunner;    // azgo/inference_hub.hpp

// True when this binary was built with -DAZ_USE_CUDA=ON.
bool cudaBuilt();

// True when the GPU can run the kernels: built with CUDA, a device present,
// the kernels' machine code built for exactly this GPU's architecture (no
// PTX, no JIT), and a first kernel ran. The first call sets up the
// process-wide CUDA context. When false and why is non-null, *why says what
// is missing, e.g. "not built: rebuild with -DAZ_USE_CUDA=ON (Linux, NVIDIA
// GPU)" or "kernels built for sm_89; this GPU is sm_86: reconfigure with
// -DAZ_CUDA_ARCHITECTURES=86-real".
bool cudaAvailable(std::string* why = nullptr);

// "NVIDIA GeForce RTX 4070 Ti SUPER sm_89, driver API 13.3, runtime 12.8,
// kernels 5d0c2e91": the GPU and the architecture its machine code was built
// for; the CUDA API version the installed driver supports
// (cudaDriverGetVersion: a driver update that raises it shows; the NVIDIA
// driver's own build number, e.g. 581.57, would need NVML, which this
// binary does not link); the CUDA runtime it was built with (static); and a
// hash of the kernel sources, compiler and flags. Two runs whose descriptions match ran the
// same GPU arithmetic. Without a usable GPU it is cudaAvailable's reason.
std::string cudaDescription();

#ifdef AZ_USE_CUDA

// The GPU behind an InferenceHub: net's weights uploaded once, `slots`
// (1..kInferenceSlots) passes of up to maxBatch positions each that may be in
// flight at once, each on its own stream. Throws std::runtime_error ("cuda:
// inference needs N MB ...; lower inference_batch") when the slots would not
// fit the GPU's free memory, before allocating them.
std::unique_ptr<BatchRunner> makeCudaRunner(const Network& net, int maxBatch, int slots = 2);

// The same memory check up front, for a run before it writes anything:
// throws that error unless `hubs` hubs of `slotsPerHub` slots each, for a
// network shaped cfg at inferenceBatch positions per pass, fit.
void checkCudaInferenceMemory(const NetConfig& cfg, int inferenceBatch, int hubs, int slotsPerHub = 2);
// The same for several groups of hubs held at once, as
// checkMetalInferenceMemory's overload.
void checkCudaInferenceMemory(const NetConfig& cfg, const std::vector<GpuHubs>& groups);

// The whole network's inference forward on the GPU, synchronously: weights
// uploaded once, one workspace for up to maxBatch positions. For tests and
// bench; search goes through the hub (makeCudaRunner). Errors throw
// std::runtime_error("GPU inference failed: ...").
class CudaInference {
public:
    CudaInference(const Network& net, int maxBatch);
    ~CudaInference();
    CudaInference(const CudaInference&) = delete;
    CudaInference& operator=(const CudaInference&) = delete;
    // Network::infer's arguments; ownership needs a network with the head.
    void infer(const float* input, int B, float* logits, float* values, float* ownership = nullptr);
    size_t bytes() const;   // the GPU memory of the activations
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Test probes: each copies host arrays in, runs its kernels on one stream,
// waits and copies the results out. They exist so tests (and bench) can
// check a kernel against an exact CPU reference; the program itself never
// calls them. Each throws std::runtime_error when the GPU is not usable or a
// kernel fails.
namespace cuda_probe {

// The GPU's name as CUDA reports it, e.g. "NVIDIA GeForce RTX 4070 Ti SUPER".
std::string deviceName();
// The device's compute capability as 10 * major + minor (89 for sm_89).
int deviceArchitecture();
// The architecture of the machine code that runs the kernels, as
// cudaFuncGetAttributes reports it; equal to deviceArchitecture() whenever
// cudaAvailable() is true.
int binaryVersion();
// The first kernel: data[0..n) = value. data holds n + tail floats; all of
// them are copied to the GPU and back, so a test can check the tail is untouched.
void fill(float* data, size_t n, float value, size_t tail = 0);

// The gemm: row-major C[M][N] = A[M][K] . op(B), op(B) = B [K][N], or B
// [N][K] transposed with transB; C's old contents go to the GPU too, and are
// never read there (only written). Each element is an fmaf
// chain over k = 0 .. K-1. tile: -1 for the rule the network uses for a
// convolution (by M), or 0, 1, 2 to force 32x256, 64x128, 128x128.
void gemm(bool transB, int M, int N, int K, const float* A, const float* B, float* C, int tile = -1);

// The forward layers' other kernels, each against its CPU counterpart
// (Network::forward's loops, Backend::im2col, the layer objects of
// azgo/layers.hpp). Activations are channel-major [C][B * n * n] as on the CPU.
void transposeInput(const float* input, int B, int planes, int area, float* out);  // [B][P][A] -> [P][B][A]
void im2col3(const float* in, int C, int B, int n, float* col);                    // Backend::im2col, k = 3
void bnInfer(const BatchNorm2d& bn, const float* x, int M, float* y);              // BatchNorm2d::forwardInfer
void relu(float* x, size_t count);                                                 // reluForward
void addInto(float* dst, const float* src, size_t count);                          // dst += src
void policyFlatten(const float* act, int B, int A, float* flat);                   // [2][B][A] -> [B][2A]
void biasAddRows(float* y, const float* b, int rows, int cols);                    // y [rows][cols] += b
void tanhInPlace(float* x, size_t count);                                          // x = tanh(x)
// The fused 3x3 convolution with its batch norm, an optional skip (null for
// none) and ReLU: x [cin][B][n][n] -> y [cout][B * n * n]. tile as gemm's.
void convBnRelu(const Conv2d& conv, const BatchNorm2d& bn, const float* x, const float* skip, int B, int n,
                float* y, int tile = -1);

// CudaInference under its test name.
using InferNet = CudaInference;
// Uploads net into weights laid out for layoutNet (ParamLayout::of): throws
// std::logic_error naming the first tensor the layout does not know.
void checkUpload(const Network& layoutNet, const Network& net);
// Uploads net, then enqueues one forward pass of a GPU network built for
// encodeAs (a stand-in for a GPU port that lags behind the CPU network):
// throws std::logic_error naming every uploaded tensor the pass never reads.
void checkForwardReadsEveryTensor(const Network& net, const NetConfig& encodeAs);

// Chaos mode for the hubs that WorkerEvaluators creates from now on (0: off;
// see InferenceHubOptions::chaosSeed). A test hook: a run's own settings
// never turn it on, and it touches none of the run's random streams.
void setHubChaosSeed(uint64_t seed);
uint64_t hubChaosSeed();
// Failure injection: after n more passes of any runner complete normally,
// the next one reports a failure through the normal error path (as a GPU
// fault would); later ones run normally again. n = 0 disables it.
void failPassesAfter(int n);
// The GPU memory the memory checks allow, in place of the device's free
// memory less a margin (0: the device's again). A test hook.
void setMemoryLimit(size_t bytes);
// Passes the runners of this process have reported so far. A test hook.
long long passesCompleted();
// Makes every runner's pass hold the GPU this many microseconds at the start
// of its copy-out (0: never), so a test can tell a pass reported before its
// outputs reached the host. A test hook.
void setCopyOutDelay(int microseconds);

} // namespace cuda_probe

#endif // AZ_USE_CUDA

} // namespace azgo
