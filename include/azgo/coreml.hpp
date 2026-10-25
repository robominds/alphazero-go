// The Core ML engines, as the rest of the program sees them. Two self-play
// and arena engines run the network's inference pass in 16-bit floating
// point through Apple's Core ML on a Mac:
//
//   ane         the Apple Neural Engine (compute units cpuAndNeuralEngine;
//               every operation of the model runs on the Neural Engine)
//   coreml_gpu  the GPU, through Core ML's float16 GPU path (cpuAndGPU)
//
// They are engines only (selfplay_engines = local/ane:30, mini1/ane:10;
// arena_engines likewise), BatchRunners behind the inference hub exactly
// where the Metal and CUDA runners sit. Training, the validation loss and
// the outside ratings stay 32-bit on the CPU or the Metal GPU.
//
// How a network becomes a Core ML model. The model is not converted by a
// Python tool: this program writes it. buildCoreMLModel turns a Network
// into an ML Program (Core ML's typed operation graph) for one fixed batch
// size, with every batch norm folded into the convolution before it and
// the weights rounded to float16, and returns the three files of an
// .mlpackage: the protobuf specification, the weight blob and the
// manifest. The runtime (src/coreml/coreml_runner.mm) writes them to a
// temporary directory, has Core ML compile and load them, and keeps the
// loaded models of a network for as long as a hub uses them. A remote Mac
// therefore converts the network it receives itself; nothing but the .azn
// travels.
//
// Why fixed batch sizes. A model with flexible shapes runs on the CPU, so
// each engine loads one model per batch size it uses (coremlBatchSizes) and
// splits a pass into chunks of those sizes, padding the last one
// (coremlChunks).
//
// The contract (docs/superpowers/specs/2026-10-24-coreml-engines-design.md):
// a position's outputs are a function of its own input and the engine
// fingerprint (backend, chip, OS build, Core ML version, compute units,
// batch sizes and the model format hash), never of the positions it shares
// a pass with; test_coreml checks that per device. Against the 32-bit CPU
// they agree within the float16 tier of tolerance (logits within 5e-3 of
// the largest logit's magnitude, values within 3e-2), not bitwise.
//
// The writer and the "not built" answers are portable C++17, compiled in
// every build; the runtime exists only with -DAZ_USE_COREML=ON (macOS 15
// or later). No Objective-C type appears in this header.
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

// True for "ane" and "coreml_gpu", the Core ML engines' backend names.
bool coremlBackend(const std::string& name);

// True when this binary was built with -DAZ_USE_COREML=ON.
bool coremlBuilt();

// True when `backend` (ane or coreml_gpu) can run here: built with Core
// ML, macOS 15 or later, the device present (a Neural Engine for ane, a
// GPU for coreml_gpu), and a small probe network compiled, loaded and run
// with every operation on that device. The first call per backend does
// the probe (a fraction of a second). When false and why is non-null, *why
// says what is missing, e.g. "not built: rebuild with -DAZ_USE_COREML=ON
// (macOS only)".
bool coremlAvailable(const std::string& backend, std::string* why = nullptr);

// What determines the engine's arithmetic beside the CPU and OS of the
// engine fingerprint (engineFingerprint), e.g.
//   Core ML 3600.25.2, Neural Engine 16 cores, cpuAndNeuralEngine, float16, batches 8/32/128, model ecc217a1
// Without a usable device it is coremlAvailable's reason.
std::string coremlDescription(const std::string& backend);

// The batch sizes the backend's models are compiled for, ascending: the
// Neural Engine's arithmetic is the same at every size (measured, and
// checked per device by test_coreml), the GPU's is not, so coreml_gpu uses
// one size only. std::invalid_argument for another backend.
std::vector<int> coremlBatchSizes(const std::string& backend);

// How a pass of B >= 1 positions runs on models of the given sizes
// (ascending, distinct, each >= 1): chunks of the largest size that fits
// what is left, and once less than the smallest size is left, one chunk of
// the smallest size for the rest (padded). Their sizes in order; their sum is at least B and less than B +
// sizes[0]. std::invalid_argument for B < 1 or bad sizes.
std::vector<int> coremlChunks(const std::vector<int>& sizes, int B);

// ---- the model writer (portable) -------------------------------------------

// The files of an .mlpackage for `net` at a fixed batch size:
//   spec      Data/com.apple.CoreML/model.mlmodel, a CoreML.Specification.Model
//             (specification version 9, an ML Program in opset CoreML8)
//   weights   Data/com.apple.CoreML/weights/weight.bin, the MIL blob file
//   manifest  Manifest.json
// Input "planes" float32 [batch][planes][n][n] (Evaluator's layout);
// outputs "logits" float32 [batch][actions] and "value" float32 [batch].
// Inside, everything is float16: the input is cast down, each convolution
// carries its batch norm folded in (w * gamma / sqrt(var + eps) and beta -
// mean * gamma / sqrt(var + eps), computed in double, rounded to float,
// then to float16 to nearest even), and the outputs are cast up. The
// ownership head, which search never reads, is left out. The bytes depend
// only on the network and the batch: the same network always gives the
// same package.
struct CoreMLModelFiles {
    std::string spec, weights, manifest;
};
CoreMLModelFiles buildCoreMLModel(const Network& net, int batch);

// Writes buildCoreMLModel(net, batch) as the package directory `dir`
// (which must end in ".mlpackage" and not exist yet). std::runtime_error
// when it cannot be written.
void writeCoreMLPackage(const Network& net, int batch, const std::string& dir);

// 64-bit FNV-1a over the network's shape and every tensor's bytes, the
// key under which a process keeps a network's loaded models.
uint64_t coremlNetworkHash(const Network& net);

// The first 8 hex digits of the hash of the writer's output for a fixed
// small network: it changes exactly when the writer's output changes, so
// it stands for the model format in the fingerprint.
std::string coremlModelFormatHash();

// float32 -> float16 bits, round to nearest even, as numpy's astype and
// the hardware conversion do (overflow to infinity, NaN kept a NaN).
uint16_t floatToHalf(float x);
float halfToFloat(uint16_t h);

#ifdef AZ_USE_COREML

// The engine behind an InferenceHub: the network's models loaded once per
// process (compiled on first use, shared by every hub of the network and
// backend), `slots` (1..kInferenceSlots) passes of up to maxBatch
// positions in flight at once, each on its own thread. Throws
// std::runtime_error ("ane: inference needs N MB ...; lower
// inference_batch") when the models would not fit, before loading them,
// and std::runtime_error ("ane: ...") when Core ML refuses the model or
// places an operation on another device.
std::unique_ptr<BatchRunner> makeCoreMLRunner(const Network& net, const std::string& backend, int maxBatch,
                                              int slots = 2);

// The same memory check up front, for a run before it writes anything:
// the models and staging of `groups` of hubs of a network shaped cfg must
// fit the device's budget (the GPU's recommended working set for
// coreml_gpu, a quarter of the memory for ane).
void checkCoreMLInferenceMemory(const NetConfig& cfg, const std::string& backend,
                                const std::vector<GpuHubs>& groups);

// The network's inference forward on the engine, synchronously, in chunks
// as the runner does: for tests and bench; search goes through the hub.
// Errors throw std::runtime_error("GPU inference failed: ...").
class CoreMLInference {
public:
    CoreMLInference(const Network& net, const std::string& backend, int maxBatch);
    ~CoreMLInference();
    CoreMLInference(const CoreMLInference&) = delete;
    CoreMLInference& operator=(const CoreMLInference&) = delete;
    void infer(const float* input, int B, float* logits, float* values);
    // The same through the model of one compiled size only (size must be
    // one of coremlBatchSizes): a test of batch independence across sizes.
    void inferAtSize(int size, const float* input, int B, float* logits, float* values);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

namespace coreml_probe {

// Where Core ML's compute plan puts the model of net at `batch` with the
// backend's compute units: the number of non-const operations per
// preferred device ("ANE", "GPU", "CPU", "none"), and their names per device.
struct Placement {
    std::vector<std::pair<std::string, int>> devices;   // device, operations
    int operations = 0;
    std::string summary;                                 // "conv 23 ANE, relu 24 ANE, ..."
};
Placement placement(const Network& net, const std::string& backend, int batch);

// Chaos mode for the hubs that WorkerEvaluators creates from now on (0:
// off; see InferenceHubOptions::chaosSeed). A test hook.
void setHubChaosSeed(uint64_t seed);
uint64_t hubChaosSeed();
// Failure injection: after n more chunk predictions succeed, the next one
// fails through the normal error path; later ones run normally again.
// n = 0 disables it.
void failPassesAfter(int n);
// The memory budget of the checks, in place of the device's (0: the
// device's again). A test hook.
void setMemoryLimit(size_t bytes);
// Models compiled and loaded by this process so far (the cache's misses).
long long modelsLoaded();
// Predictions (chunks) this process has run so far.
long long predictions();
// Core ML's compiler cache for this program
// (~/Library/Caches/<process name>/com.apple.e5rt.e5bundlecache), which
// gains an entry for every model a process compiles and never finds it
// again: a process's first Core ML use removes its entries older than a
// day. pruneCompilerCache does that to `dir` (<dir>/<OS build>/<entry>)
// for entries older than maxAgeSeconds and returns how many it removed.
std::string compilerCacheDirectory();
int pruneCompilerCache(const std::string& dir, double maxAgeSeconds);

} // namespace coreml_probe

#endif // AZ_USE_COREML

} // namespace azgo
