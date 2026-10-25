// The Core ML runtime: compiling and loading a network's models, the
// BatchRunner of the ane and coreml_gpu engines, and what the fingerprint
// says about them.
//
// A network's models. The first hub for a network on a backend writes the
// network as one .mlpackage per compiled batch size (model_writer.cpp) into
// a private temporary directory, has Core ML compile each package into an
// .mlmodelc and loads it with the backend's compute units:
// cpuAndNeuralEngine for ane, cpuAndGPU for coreml_gpu. The first model of
// each shape is checked against Core ML's compute plan: every operation
// must be placed on the backend's device, or the load fails (an operation
// that silently fell back to the CPU would change both speed and bits).
// The loaded models are kept per (network hash, backend) as long as a hub
// uses them, and the last four stay loaded after that, so an arena that
// builds a hub for the same network wave after wave compiles it once.
//
// The runner. Each slot has a thread of its own: begin hands it the pass,
// it splits the pass into chunks of the compiled sizes (coremlChunks), runs
// one synchronous prediction per chunk with the slot's staging rows as the
// input (no copy: the multi-array points at them), copies the outputs out
// row by row (Core ML may pad a row) and calls done. Two slots put two
// predictions in flight, which keeps the Neural Engine busy while one
// pass's outputs are copied and the next pass is gathered.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "azgo/coreml.hpp"
#include "azgo/inference_hub.hpp"

namespace azgo {

namespace {

std::atomic<uint64_t> gChaosSeed{0};
std::atomic<int> gFailAfter{0};          // 0: off; else predictions left before one fails
std::atomic<size_t> gMemoryLimit{0};
std::atomic<long long> gModelsLoaded{0}, gPredictions{0};

// failPassesAfter's countdown: true for the one prediction that fails.
bool injectFailure() {
    int left = gFailAfter.load();
    while (left > 0)
        if (gFailAfter.compare_exchange_weak(left, left - 1)) return left == 1;
    return false;
}

std::string str(NSString* s) { return s ? std::string([s UTF8String]) : std::string(); }

std::string errorText(NSError* e) {
    if (!e) return "unknown error";
    return str(e.localizedDescription);
}

// The backends: compute units, the device the operations must be on.
struct BackendSpec {
    MLComputeUnits units;
    const char* units_name;
    const char* device;   // "ANE" or "GPU", as placement() names them
};

BackendSpec specFor(const std::string& backend) {
    if (backend == "ane") return {MLComputeUnitsCPUAndNeuralEngine, "cpuAndNeuralEngine", "ANE"};
    if (backend == "coreml_gpu") return {MLComputeUnitsCPUAndGPU, "cpuAndGPU", "GPU"};
    throw std::invalid_argument("not a Core ML backend: '" + backend + "'");
}

MLModelConfiguration* configurationFor(const std::string& backend) API_AVAILABLE(macos(15.0)) {
    MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
    cfg.computeUnits = specFor(backend).units;
    cfg.allowLowPrecisionAccumulationOnGPU = NO;   // the default, stated: float32 accumulation where offered
    return cfg;
}

std::string deviceName(id<MLComputeDeviceProtocol> d) API_AVAILABLE(macos(15.0)) {
    if (!d) return "none";
    if ([(NSObject*)d isKindOfClass:[MLNeuralEngineComputeDevice class]]) return "ANE";
    if ([(NSObject*)d isKindOfClass:[MLGPUComputeDevice class]]) return "GPU";
    if ([(NSObject*)d isKindOfClass:[MLCPUComputeDevice class]]) return "CPU";
    return "other";
}

void removeTree(const std::string& path) {
    if (path.empty()) return;
    @autoreleasepool {
        [[NSFileManager defaultManager] removeItemAtPath:[NSString stringWithUTF8String:path.c_str()] error:nil];
    }
}

// A process-private scratch directory for packages and compiled models.
std::string scratchDirectory() {
    static std::mutex m;
    static long counter = 0;
    std::lock_guard<std::mutex> lock(m);
    @autoreleasepool {
        NSString* base = NSTemporaryDirectory();
        const std::string dir = str(base) + "azgo-coreml-" + std::to_string(::getpid()) + "-" + std::to_string(++counter);
        NSError* err = nil;
        if (![[NSFileManager defaultManager] createDirectoryAtPath:[NSString stringWithUTF8String:dir.c_str()]
                                       withIntermediateDirectories:YES
                                                        attributes:nil
                                                             error:&err])
            throw std::runtime_error("coreml: cannot create " + dir + ": " + errorText(err));
        return dir;
    }
}

// The compute plan's placement of a compiled model's operations.
coreml_probe::Placement planOf(NSURL* compiled, const std::string& backend) API_AVAILABLE(macos(15.0)) {
    __block MLComputePlan* plan = nil;
    __block NSError* error = nil;
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [MLComputePlan loadContentsOfURL:compiled
                       configuration:configurationFor(backend)
                   completionHandler:^(MLComputePlan* p, NSError* e) {
                     plan = p;
                     error = e;
                     dispatch_semaphore_signal(done);
                   }];
    dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
    if (!plan) throw std::runtime_error(backend + ": no compute plan: " + errorText(error));
    MLModelStructureProgram* program = plan.modelStructure.program;
    MLModelStructureProgramFunction* main = program ? program.functions[@"main"] : nil;
    if (!main) throw std::runtime_error(backend + ": the compute plan has no main function");
    coreml_probe::Placement out;
    std::map<std::string, int> perDevice;
    std::map<std::pair<std::string, std::string>, int> perOp;   // (operator, device)
    for (MLModelStructureProgramOperation* op in main.block.operations) {
        const std::string name = str(op.operatorName);
        if (name == "const") continue;
        MLComputePlanDeviceUsage* usage = [plan computeDeviceUsageForMLProgramOperation:op];
        const std::string device = usage ? deviceName(usage.preferredComputeDevice) : "none";
        ++perDevice[device];
        ++perOp[{name, device}];
        ++out.operations;
    }
    for (const auto& [device, count] : perDevice) out.devices.push_back({device, count});
    for (const auto& [key, count] : perOp) {
        if (!out.summary.empty()) out.summary += ", ";
        out.summary += key.first + " " + std::to_string(count) + " " + key.second;
    }
    return out;
}

// One compiled, loaded model.
struct LoadedModel {
    int size = 0;
    MLModel* model = nil;   // strong (ARC)
};

// A network's models on one backend, one per compiled size.
struct LoadedNetwork {
    NetConfig cfg;
    std::string backend;
    std::vector<int> sizes;
    std::vector<LoadedModel> models;   // by size, ascending
    std::string dir;                   // the compiled models live here until the last user goes
    ~LoadedNetwork() {
        @autoreleasepool {
            models.clear();
        }
        removeTree(dir);
    }
    MLModel* modelFor(int size) const {
        for (const LoadedModel& m : models)
            if (m.size == size) return m.model;
        throw std::logic_error("coreml: no model of size " + std::to_string(size));
    }
};

// Has Core ML compile the package at `package` (the synchronous call is
// deprecated) and moves the compiled model to `target`, out of the
// temporary place Core ML puts it. Throws on failure.
NSURL* compilePackage(const std::string& package, const std::string& target, const std::string& backend)
    API_AVAILABLE(macos(15.0)) {
    NSURL* pkgURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:package.c_str()]];
    NSURL* targetURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:target.c_str()]];
    __block NSError* error = nil;
    __block bool moved = false;
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [MLModel compileModelAtURL:pkgURL
             completionHandler:^(NSURL* compiled, NSError* e) {
               error = e;
               if (compiled) {
                   NSError* me = nil;
                   moved = [[NSFileManager defaultManager] moveItemAtURL:compiled toURL:targetURL error:&me];
                   if (!moved) error = me;
               }
               dispatch_semaphore_signal(done);
             }];
    dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
    removeTree(package);
    if (!moved) throw std::runtime_error(backend + ": Core ML could not compile the model: " + errorText(error));
    return targetURL;
}

// Compiles and loads net at `size` into dir; checks placement once per shape.
MLModel* compileAndLoad(const Network& net, const std::string& backend, int size, const std::string& dir,
                        bool checkPlacement) API_AVAILABLE(macos(15.0)) {
    const std::string tag = std::to_string(size);
    const std::string package = dir + "/b" + tag + ".mlpackage";
    writeCoreMLPackage(net, size, package);
    NSURL* target = compilePackage(package, dir + "/b" + tag + ".mlmodelc", backend);
    if (checkPlacement) {
        const coreml_probe::Placement p = planOf(target, backend);
        const std::string want = specFor(backend).device;
        int elsewhere = 0;
        for (const auto& [device, count] : p.devices)
            if (device != want) elsewhere += count;
        if (elsewhere > 0)
            throw std::runtime_error(backend + ": Core ML places " + std::to_string(elsewhere) + " of " +
                                     std::to_string(p.operations) + " operations off the " + want +
                                     " at batch " + tag + " (" + p.summary + ")");
    }
    NSError* loadError = nil;
    MLModel* model = [MLModel modelWithContentsOfURL:target configuration:configurationFor(backend) error:&loadError];
    if (!model) throw std::runtime_error(backend + ": Core ML could not load the model: " + errorText(loadError));
    ++gModelsLoaded;
    return model;
}

// The models of the networks in use, and of the last few used.
class ModelCache {
public:
    static ModelCache& instance() {
        static ModelCache* cache = new ModelCache();   // never destroyed: hubs may outlive static destruction
        return *cache;
    }

    // checkPlacement: the first load of each shape checks it (the probe's
    // own small network does not: where Core ML puts a small network is
    // its cost model's choice, and the CPU may win).
    std::shared_ptr<LoadedNetwork> get(const Network& net, const std::string& backend, bool checkPlacement = true)
        API_AVAILABLE(macos(15.0)) {
        const auto key = std::make_pair(coremlNetworkHash(net), backend);
        std::lock_guard<std::mutex> lock(mutex_);
        if (auto it = live_.find(key); it != live_.end())
            if (std::shared_ptr<LoadedNetwork> hit = it->second.lock()) {
                remember(hit);
                return hit;
            }
        auto loaded = std::make_shared<LoadedNetwork>();
        loaded->cfg = net.config();
        loaded->backend = backend;
        loaded->sizes = coremlBatchSizes(backend);
        loaded->dir = scratchDirectory();
        @autoreleasepool {
            for (int size : loaded->sizes) {
                const auto shape = std::make_tuple(backend, configKey(net.config()), size);
                const bool check = checkPlacement && !placed_.count(shape);
                loaded->models.push_back({size, compileAndLoad(net, backend, size, loaded->dir, check)});
                if (check) placed_.insert(shape);
            }
        }
        live_[key] = loaded;
        remember(loaded);
        return loaded;
    }

private:
    static std::string configKey(const NetConfig& c) {
        return std::to_string(c.boardSize) + "x" + std::to_string(c.history) + "x" + std::to_string(c.blocks) + "x" +
               std::to_string(c.filters) + "x" + std::to_string(c.valueHidden);
    }
    void remember(const std::shared_ptr<LoadedNetwork>& n) {
        recent_.erase(std::remove(recent_.begin(), recent_.end(), n), recent_.end());
        recent_.push_front(n);
        while (recent_.size() > 4) recent_.pop_back();
        for (auto it = live_.begin(); it != live_.end();)
            it = it->second.expired() ? live_.erase(it) : std::next(it);
    }
    std::mutex mutex_;
    std::map<std::pair<uint64_t, std::string>, std::weak_ptr<LoadedNetwork>> live_;
    std::deque<std::shared_ptr<LoadedNetwork>> recent_;
    std::set<std::tuple<std::string, std::string, int>> placed_;
};

// One pass: B positions of `input` (which has room for the padded chunks;
// the padding rows are zeroed here) through the network's models, outputs
// to logits [B][actions] and values [B]. "" or the error.
std::string runPass(const LoadedNetwork& ln, float* input, int B, float* logits, float* values)
    API_AVAILABLE(macos(15.0)) {
    const NetConfig& cfg = ln.cfg;
    const int P = cfg.planes(), n = cfg.boardSize, A = cfg.area(), ACT = cfg.actions();
    const size_t per = static_cast<size_t>(P) * A;
    const std::vector<int> chunks = coremlChunks(ln.sizes, B);
    int total = 0;
    for (int c : chunks) total += c;
    std::fill(input + static_cast<size_t>(B) * per, input + static_cast<size_t>(total) * per, 0.0f);
    int at = 0;
    for (int size : chunks) {
        @autoreleasepool {
            if (injectFailure()) return "coreml: injected failure";
            NSError* err = nil;
            MLMultiArray* in = [[MLMultiArray alloc] initWithDataPointer:input + static_cast<size_t>(at) * per
                                                                   shape:@[ @(size), @(P), @(n), @(n) ]
                                                                dataType:MLMultiArrayDataTypeFloat32
                                                                 strides:@[ @(P * A), @(A), @(n), @1 ]
                                                             deallocator:nil
                                                                   error:&err];
            if (!in) return "coreml: input array: " + errorText(err);
            MLDictionaryFeatureProvider* features =
                [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"planes" : in} error:&err];
            if (!features) return "coreml: input features: " + errorText(err);
            id<MLFeatureProvider> out = [ln.modelFor(size) predictionFromFeatures:features error:&err];
            if (!out) return "coreml: prediction failed: " + errorText(err);
            ++gPredictions;
            MLMultiArray* L = [out featureValueForName:@"logits"].multiArrayValue;
            MLMultiArray* V = [out featureValueForName:@"value"].multiArrayValue;
            if (!L || !V || L.shape.count != 2 || V.shape.count != 1 || L.shape[0].intValue != size ||
                L.shape[1].intValue != ACT || V.shape[0].intValue != size)
                return "coreml: unexpected output shapes";
            const int rows = std::min(size, B - at);
            const long ls0 = L.strides[0].longValue, ls1 = L.strides[1].longValue, vs = V.strides[0].longValue;
            if (L.dataType != MLMultiArrayDataTypeFloat32 || V.dataType != MLMultiArrayDataTypeFloat32)
                return "coreml: unexpected output type";
            float* lout = logits + static_cast<size_t>(at) * ACT;
            float* vout = values + at;
            [L getBytesWithHandler:^(const void* bytes, NSInteger) {
              const float* src = static_cast<const float*>(bytes);
              for (int r = 0; r < rows; ++r)
                  for (int c = 0; c < ACT; ++c) lout[static_cast<size_t>(r) * ACT + c] = src[r * ls0 + c * ls1];
            }];
            [V getBytesWithHandler:^(const void* bytes, NSInteger) {
              const float* src = static_cast<const float*>(bytes);
              for (int r = 0; r < rows; ++r) vout[r] = src[r * vs];
            }];
        }
        at += size;
    }
    return "";
}

// ---- the runner --------------------------------------------------------------

class CoreMLBatchRunner : public BatchRunner {
public:
    CoreMLBatchRunner(std::shared_ptr<LoadedNetwork> net, int maxBatch, int slots)
        : net_(std::move(net)), maxBatch_(maxBatch) {
        const NetConfig& cfg = net_->cfg;
        const size_t rows = static_cast<size_t>(maxBatch) + static_cast<size_t>(net_->sizes.back());
        for (int s = 0; s < slots; ++s) {
            auto slot = std::make_unique<Slot>();
            slot->input.assign(rows * cfg.planes() * cfg.area(), 0.0f);
            slot->logits.assign(static_cast<size_t>(maxBatch) * cfg.actions(), 0.0f);
            slot->values.assign(static_cast<size_t>(maxBatch), 0.0f);
            slots_.push_back(std::move(slot));
        }
        for (int s = 0; s < slots; ++s) slots_[s]->thread = std::thread([this, s] { loop(s); });
    }

    ~CoreMLBatchRunner() override {
        for (auto& slot : slots_) {
            {
                std::lock_guard<std::mutex> lock(slot->mutex);
                slot->stop = true;
            }
            slot->wake.notify_one();
        }
        for (auto& slot : slots_) slot->thread.join();
    }

    int slots() const override { return static_cast<int>(slots_.size()); }
    int maxBatch() const override { return maxBatch_; }
    float* input(int slot) override { return slots_.at(slot)->input.data(); }
    const float* logits(int slot) const override { return slots_.at(slot)->logits.data(); }
    const float* values(int slot) const override { return slots_.at(slot)->values.data(); }

    void begin(int slot, int B, std::function<void(int slot, std::string error)> done) override {
        if (B < 1 || B > maxBatch_) throw std::invalid_argument("coreml runner: bad batch " + std::to_string(B));
        Slot& s = *slots_.at(slot);
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            if (s.pending) throw std::logic_error("coreml runner: slot already busy");
            s.B = B;
            s.done = std::move(done);
            s.pending = true;
        }
        s.wake.notify_one();
    }

private:
    struct Slot {
        std::vector<float> input, logits, values;
        std::mutex mutex;
        std::condition_variable wake;
        bool pending = false, stop = false;
        int B = 0;
        std::function<void(int, std::string)> done;
        std::thread thread;
    };

    void loop(int index) {
        Slot& s = *slots_[index];
        for (;;) {
            std::function<void(int, std::string)> done;
            int B = 0;
            {
                std::unique_lock<std::mutex> lock(s.mutex);
                s.wake.wait(lock, [&] { return s.pending || s.stop; });
                if (!s.pending) return;   // stopped with nothing to do
                done = std::move(s.done);
                B = s.B;
            }
            std::string error;
            try {
                if (@available(macOS 15.0, *)) error = runPass(*net_, s.input.data(), B, s.logits.data(), s.values.data());
                else error = "coreml: needs macOS 15";
            } catch (const std::exception& e) {
                error = std::string("coreml: ") + e.what();
            }
            {
                std::lock_guard<std::mutex> lock(s.mutex);
                s.pending = false;
            }
            done(index, error);
        }
    }

    std::shared_ptr<LoadedNetwork> net_;
    int maxBatch_;
    std::vector<std::unique_ptr<Slot>> slots_;
};

// ---- availability and description ---------------------------------------------

struct Probe {
    bool ok = false;
    std::string why, description;
};

std::string coremlVersion() {
    @autoreleasepool {
        NSBundle* bundle = [NSBundle bundleForClass:[MLModel class]];
        // The framework's build ("3600.25.2"); its marketing version is "1.0".
        NSString* v = bundle.infoDictionary[@"CFBundleVersion"];
        if (!v) v = bundle.infoDictionary[@"CFBundleShortVersionString"];
        return v ? str(v) : "unknown";
    }
}

// Commas and semicolons separate a fingerprint's parts.
std::string tidy(std::string s) {
    for (char& c : s)
        if (c == ',' || c == ';' || c == '\n') c = ' ';
    return s;
}

Probe probe(const std::string& backend) {
    Probe p;
    if (@available(macOS 15.0, *)) {
        @autoreleasepool {
            std::string device;
            for (id<MLComputeDeviceProtocol> d in MLAllComputeDevices()) {
                if (backend == "ane" && [(NSObject*)d isKindOfClass:[MLNeuralEngineComputeDevice class]])
                    device = "Neural Engine " +
                             std::to_string(static_cast<long>(((MLNeuralEngineComputeDevice*)d).totalCoreCount)) +
                             " cores";
                if (backend == "coreml_gpu" && [(NSObject*)d isKindOfClass:[MLGPUComputeDevice class]])
                    device = "GPU " + tidy(str(((MLGPUComputeDevice*)d).metalDevice.name));
            }
            if (device.empty()) {
                p.why = backend == "ane" ? "no Neural Engine on this Mac" : "no GPU for Core ML on this Mac";
                return p;
            }
            // A small network through the whole path: compile, load and one
            // prediction. Its placement is not checked (a real network's is,
            // when it loads).
            try {
                NetConfig cfg;
                cfg.boardSize = 9;
                cfg.history = 8;
                cfg.blocks = 2;
                cfg.filters = 64;
                cfg.valueHidden = 64;
                const Network net(cfg, 7);
                std::shared_ptr<LoadedNetwork> ln = ModelCache::instance().get(net, backend, false);
                const int B = coremlBatchSizes(backend).front();
                std::vector<float> in(static_cast<size_t>(B) * cfg.planes() * cfg.area(), 0.0f);
                std::vector<float> logits(static_cast<size_t>(B) * cfg.actions()), values(B);
                const std::string err = runPass(*ln, in.data(), B, logits.data(), values.data());
                if (!err.empty()) {
                    p.why = err;
                    return p;
                }
            } catch (const std::exception& e) {
                p.why = e.what();
                return p;
            }
            std::string sizes;
            for (int s : coremlBatchSizes(backend)) sizes += (sizes.empty() ? "" : "/") + std::to_string(s);
            const std::vector<int> all = coremlBatchSizes(backend);
            p.description = "Core ML " + tidy(coremlVersion()) + ", " + device + ", " + specFor(backend).units_name +
                            ", float16, " + (all.size() > 1 ? "batches " : "batch ") + sizes + ", model " +
                            coremlModelFormatHash();
            p.ok = true;
        }
    } else {
        p.why = "Core ML engines need macOS 15 or later";
    }
    return p;
}

const Probe& probeOnce(const std::string& backend) {
    static std::mutex m;
    static std::map<std::string, Probe> done;
    std::lock_guard<std::mutex> lock(m);
    auto it = done.find(backend);
    if (it == done.end()) it = done.emplace(backend, probe(backend)).first;
    return it->second;
}

// The memory budget of the checks: the GPU's recommended working set for
// coreml_gpu, a quarter of the machine's memory for the Neural Engine
// (unified memory either way; this is a sanity bound, not an allocator).
double budget(const std::string& backend) {
    if (gMemoryLimit.load() > 0) return static_cast<double>(gMemoryLimit.load());
    if (@available(macOS 15.0, *)) {
        if (backend == "coreml_gpu")
            for (id<MLComputeDeviceProtocol> d in MLAllComputeDevices())
                if ([(NSObject*)d isKindOfClass:[MLGPUComputeDevice class]])
                    return static_cast<double>(((MLGPUComputeDevice*)d).metalDevice.recommendedMaxWorkingSetSize);
    }
    return static_cast<double>([NSProcessInfo processInfo].physicalMemory) / 4;
}

// What one hub holds: every compiled model (float16 weights and three
// activation tensors of its size) and the slots' float32 staging.
double hubBytes(const NetConfig& cfg, const std::string& backend, int maxBatch, int slots) {
    Network shape(cfg, 0);
    double weights = 0;
    for (const ParamView& p : shape.params()) weights += static_cast<double>(p.size);
    const std::vector<int> sizes = coremlBatchSizes(backend);
    double models = 0;
    for (int s : sizes)
        models += 2.0 * (weights + 3.0 * s * cfg.filters * cfg.area() + static_cast<double>(s) * cfg.planes() * cfg.area());
    const double staging = 4.0 * slots *
                           ((static_cast<double>(maxBatch) + sizes.back()) * cfg.planes() * cfg.area() +
                            static_cast<double>(maxBatch) * (cfg.actions() + 1));
    return models + staging;
}

void checkFits(const std::string& backend, double need, int batch, int hubs) {
    const double have = budget(backend);
    if (need <= have) return;
    throw std::runtime_error(backend + ": inference needs " + std::to_string(static_cast<long long>(need / 1e6)) +
                             " MB at inference_batch " + std::to_string(batch) +
                             (hubs > 1 ? " (" + std::to_string(hubs) + " hubs)" : std::string()) +
                             ", more than the budget of " + std::to_string(static_cast<long long>(have / 1e6)) +
                             " MB; lower inference_batch");
}

} // namespace

bool coremlBuilt() { return true; }

bool coremlAvailable(const std::string& backend, std::string* why) {
    if (!coremlBackend(backend)) {
        if (why) *why = "not a Core ML backend: '" + backend + "'";
        return false;
    }
    const Probe& p = probeOnce(backend);
    if (!p.ok && why) *why = p.why;
    return p.ok;
}

std::string coremlDescription(const std::string& backend) {
    if (!coremlBackend(backend)) return "not a Core ML backend: '" + backend + "'";
    const Probe& p = probeOnce(backend);
    return p.ok ? p.description : p.why;
}

std::unique_ptr<BatchRunner> makeCoreMLRunner(const Network& net, const std::string& backend, int maxBatch,
                                              int slots) {
    if (!coremlBackend(backend)) throw std::invalid_argument("makeCoreMLRunner: not a Core ML backend: '" + backend + "'");
    if (maxBatch < 1) throw std::invalid_argument("makeCoreMLRunner: maxBatch must be >= 1");
    if (slots < 1 || slots > kInferenceSlots)
        throw std::invalid_argument("makeCoreMLRunner: slots must be in [1, " + std::to_string(kInferenceSlots) + "]");
    std::string why;
    if (!coremlAvailable(backend, &why)) throw std::runtime_error(backend + ": not usable (" + why + ")");
    checkFits(backend, hubBytes(net.config(), backend, maxBatch, slots), maxBatch, 1);
    if (@available(macOS 15.0, *))
        return std::make_unique<CoreMLBatchRunner>(ModelCache::instance().get(net, backend), maxBatch, slots);
    throw std::runtime_error(backend + ": needs macOS 15");
}

void checkCoreMLInferenceMemory(const NetConfig& cfg, const std::string& backend, const std::vector<GpuHubs>& groups) {
    if (groups.empty()) throw std::invalid_argument("checkCoreMLInferenceMemory: no hubs");
    if (!coremlBackend(backend)) throw std::invalid_argument("checkCoreMLInferenceMemory: not a Core ML backend");
    double need = 0;
    int hubs = 0, batch = 0;
    for (const GpuHubs& g : groups) {
        if (g.inferenceBatch < 1 || g.hubs < 1 || g.slotsPerHub < 1)
            throw std::invalid_argument("checkCoreMLInferenceMemory: bad arguments");
        need += g.hubs * hubBytes(cfg, backend, g.inferenceBatch, g.slotsPerHub);
        hubs += g.hubs;
        batch = std::max(batch, g.inferenceBatch);
    }
    checkFits(backend, need, batch, hubs);
}

// ---- synchronous inference ------------------------------------------------------

struct CoreMLInference::Impl {
    std::shared_ptr<LoadedNetwork> net;
    int maxBatch = 0;
    std::vector<float> staging;
};

CoreMLInference::CoreMLInference(const Network& net, const std::string& backend, int maxBatch)
    : impl_(std::make_unique<Impl>()) {
    if (maxBatch < 1) throw std::invalid_argument("CoreMLInference: maxBatch must be >= 1");
    std::string why;
    if (!coremlAvailable(backend, &why)) throw std::runtime_error(backend + ": not usable (" + why + ")");
    if (@available(macOS 15.0, *)) impl_->net = ModelCache::instance().get(net, backend);
    impl_->maxBatch = maxBatch;
    const NetConfig& cfg = net.config();
    impl_->staging.assign((static_cast<size_t>(maxBatch) + impl_->net->sizes.back()) * cfg.planes() * cfg.area(), 0.0f);
}

CoreMLInference::~CoreMLInference() = default;

void CoreMLInference::infer(const float* input, int B, float* logits, float* values) {
    if (B < 1 || B > impl_->maxBatch) throw std::invalid_argument("CoreMLInference: bad batch");
    const NetConfig& cfg = impl_->net->cfg;
    std::copy(input, input + static_cast<size_t>(B) * cfg.planes() * cfg.area(), impl_->staging.begin());
    std::string err = "coreml: needs macOS 15";
    if (@available(macOS 15.0, *)) err = runPass(*impl_->net, impl_->staging.data(), B, logits, values);
    if (!err.empty()) throw std::runtime_error("GPU inference failed: " + err);
}

void CoreMLInference::inferAtSize(int size, const float* input, int B, float* logits, float* values) {
    const std::vector<int>& sizes = impl_->net->sizes;
    if (std::find(sizes.begin(), sizes.end(), size) == sizes.end())
        throw std::invalid_argument("CoreMLInference: no model of size " + std::to_string(size));
    LoadedNetwork one;   // a view with that size only; it owns nothing on disk
    one.cfg = impl_->net->cfg;
    one.backend = impl_->net->backend;
    one.sizes = {size};
    one.models = {{size, impl_->net->modelFor(size)}};
    if (B < 1 || B > impl_->maxBatch) throw std::invalid_argument("CoreMLInference: bad batch");
    const NetConfig& cfg = one.cfg;
    std::copy(input, input + static_cast<size_t>(B) * cfg.planes() * cfg.area(), impl_->staging.begin());
    std::string err = "coreml: needs macOS 15";
    if (@available(macOS 15.0, *)) err = runPass(one, impl_->staging.data(), B, logits, values);
    if (!err.empty()) throw std::runtime_error("GPU inference failed: " + err);
}

namespace coreml_probe {

Placement placement(const Network& net, const std::string& backend, int batch) {
    if (@available(macOS 15.0, *)) {
        const std::string dir = scratchDirectory();
        try {
            @autoreleasepool {
                const std::string package = dir + "/p.mlpackage";
                writeCoreMLPackage(net, batch, package);
                Placement p = planOf(compilePackage(package, dir + "/p.mlmodelc", backend), backend);
                removeTree(dir);
                return p;
            }
        } catch (...) {
            removeTree(dir);
            throw;
        }
    }
    throw std::runtime_error("placement: needs macOS 15");
}

void setHubChaosSeed(uint64_t seed) { gChaosSeed = seed; }
uint64_t hubChaosSeed() { return gChaosSeed.load(); }
void failPassesAfter(int n) { gFailAfter = n > 0 ? n + 1 : 0; }
void setMemoryLimit(size_t bytes) { gMemoryLimit = bytes; }
long long modelsLoaded() { return gModelsLoaded.load(); }
long long predictions() { return gPredictions.load(); }

} // namespace coreml_probe

} // namespace azgo
