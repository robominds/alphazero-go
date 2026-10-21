// The test probes of azgo/cuda.hpp: each copies its inputs to the GPU, runs
// its kernels on a stream of its own, waits and copies the results back, so a
// test can hold one kernel against an exact CPU reference. The program
// itself never calls them.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <vector>
#include "azgo/cuda.hpp"
#include "cuda_context.cuh"
#include "kernels.cuh"

namespace azgo {
namespace cuda_probe {

using cuda::check;
using cuda::Context;
using cuda::DeviceBuffer;
using cuda::Stream;

namespace {

void upload(const DeviceBuffer& to, const float* from, size_t count, const Stream& s) {
    if (count) check(cudaMemcpyAsync(to.data(), from, count * sizeof(float), cudaMemcpyHostToDevice, s.get()), "upload");
}

void download(float* to, const DeviceBuffer& from, size_t count, const Stream& s) {
    if (count) check(cudaMemcpyAsync(to, from.data(), count * sizeof(float), cudaMemcpyDeviceToHost, s.get()), "download");
}

// Waits for the stream after checking the launches.
void finish(const Stream& s) {
    check(cudaGetLastError(), "kernel launch");
    s.synchronize();
}

} // namespace

std::string deviceName() { return Context::instance().deviceName(); }

int deviceArchitecture() { return Context::instance().architecture(); }

int binaryVersion() { return Context::instance().binaryVersion(); }

void fill(float* data, size_t n, float value, size_t tail) {
    Context::instance();
    Stream s;
    DeviceBuffer d(n + tail);
    upload(d, data, n + tail, s);
    cuda::launchFill(d.data(), n, value, s.get());
    download(data, d, n + tail, s);
    finish(s);
}

} // namespace cuda_probe
} // namespace azgo
