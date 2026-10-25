// The 16-bit CUDA backend, as the rest of the program sees it. backend =
// cuda16 runs the network's inference pass on an NVIDIA GPU like backend =
// cuda (azgo/cuda.hpp), except that the residual trunk's 3x3 convolutions run
// on the tensor cores: 16-bit operands (weights and activations, IEEE half),
// fp32 accumulation, batch norm, skip and ReLU in fp32. The heads are the
// cuda backend's fp32 kernels. It is a self-play and arena engine
// (selfplay_engines = pc/cuda16:60) and needs an sm_80 or newer GPU.
//
// Its results are reproducible and independent of batching, as cuda's are,
// but further from the CPU's: within the 16-bit tolerance tier of the design
// (docs/superpowers/specs/2026-10-24-cuda-fp16-design.md), not cuda's 5e-5.
//
// It exists in a build configured with -DAZ_USE_CUDA=ON (with cuda); every
// other build compiles this header, answers "not built" and refuses it. No
// CUDA type appears here.
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

// True when the GPU can run the 16-bit kernels: cudaAvailable(), and the
// device is sm_80 or newer (mma.sync m16n8k16 with fp32 accumulation). When
// false and why is non-null, *why says what is missing.
bool cuda16Available(std::string* why = nullptr);

// "NVIDIA GeForce RTX 4070 Ti SUPER sm_89, driver API 13.3, runtime 12.8,
// fp16 kernels 1a2b3c4d": as cudaDescription(), with the hash of every CUDA
// source a cuda16 pass runs (src/cuda and src/cuda/fp16), the compiler and
// the flags. Without a usable GPU it is cuda16Available's reason.
std::string cuda16Description();

#ifdef AZ_USE_CUDA

// The 16-bit operand type. The backend uses kCuda16Type; bf16 exists for
// measurements (tools/cuda_bench/accuracy).
enum class Cuda16Type { Fp16 = 0, Bf16 = 1 };
constexpr Cuda16Type kCuda16Type = Cuda16Type::Fp16;

// The GPU behind an InferenceHub, as makeCudaRunner: net's weights uploaded
// (and converted to 16 bits) once, `slots` passes of up to maxBatch positions
// in flight. Throws std::runtime_error ("cuda16: inference needs N MB ...;
// lower inference_batch") when the slots would not fit, before allocating.
std::unique_ptr<BatchRunner> makeCuda16Runner(const Network& net, int maxBatch, int slots = 2);

// The same memory check up front (checkCudaInferenceMemory's contract).
void checkCuda16InferenceMemory(const NetConfig& cfg, int inferenceBatch, int hubs, int slotsPerHub = 2);
void checkCuda16InferenceMemory(const NetConfig& cfg, const std::vector<GpuHubs>& groups);

// The whole network's inference forward on the GPU, synchronously, for tests,
// bench and the accuracy tool; search goes through the hub. Errors throw
// std::runtime_error("GPU inference failed: ...").
class Cuda16Inference {
public:
    Cuda16Inference(const Network& net, int maxBatch, Cuda16Type type = kCuda16Type);
    ~Cuda16Inference();
    Cuda16Inference(const Cuda16Inference&) = delete;
    Cuda16Inference& operator=(const Cuda16Inference&) = delete;
    // Network::infer's arguments; ownership needs a network with the head.
    void infer(const float* input, int B, float* logits, float* values, float* ownership = nullptr);
    size_t bytes() const;   // the GPU memory of the activations
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Test probes and hooks, as cuda_probe's; the program never calls them.
// The hub chaos seed of WorkerEvaluators("cuda16") is cuda_probe's.
namespace cuda16_probe {

// The 16-bit convolution alone: x [cin][B][n][n] fp32 (rounded to 16 bits
// on the GPU) -> y [cout][B * n * n] fp32, through conv's weights (rounded),
// bn's folded scale and shift, an optional skip [cout][B][n][n] (rounded;
// null for none) and ReLU. y16, when non-null, receives the 16-bit output
// widened to fp32, [cout][B * n * n] as y. tile: -1 for the network's rule,
// or 2, 4, 6, 8 to force 32, 64, 96, 128 channels per block.
void conv16(const Conv2d& conv, const BatchNorm2d& bn, const float* x, const float* skip, int B, int n, float* y,
            float* y16 = nullptr, int tile = -1, Cuda16Type type = kCuda16Type);
// The values a 16-bit type represents: each of x[0 .. count) rounded to it
// (to nearest even, as the GPU converts) and back.
void roundTo16(float* x, size_t count, Cuda16Type type = kCuda16Type);
// Upload refusals, as cuda_probe's.
void checkUpload(const Network& layoutNet, const Network& net);
void checkForwardReadsEveryTensor(const Network& net, const NetConfig& encodeAs);
// Failure injection, the memory limit, finished passes and the copy-out
// delay of the cuda16 runners, as cuda_probe's for cuda's.
void failPassesAfter(int n);
void setMemoryLimit(size_t bytes);
long long passesCompleted();
void setCopyOutDelay(int microseconds);

} // namespace cuda16_probe

#endif // AZ_USE_CUDA

} // namespace azgo
