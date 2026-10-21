// The test probes of azgo/cuda.hpp: each copies its inputs to the GPU, runs
// its kernels on a stream of its own, waits and copies the results back, so a
// test can hold one kernel against an exact CPU reference. The program
// itself never calls them.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <stdexcept>
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

void gemm(bool transB, int M, int N, int K, const float* A, const float* B, float* C, int tile) {
    Context::instance();
    if (tile < -1 || tile > 2) throw std::invalid_argument("cuda_probe::gemm: tile must be -1, 0, 1 or 2");
    const size_t a = static_cast<size_t>(M) * K, b = static_cast<size_t>(K) * N, c = static_cast<size_t>(M) * N;
    Stream s;
    DeviceBuffer dA(a), dB(b), dC(c);
    upload(dA, A, a, s);
    upload(dB, B, b, s);
    upload(dC, C, c, s);   // what C held: a test can check it is never read
    const cuda::GemmTile t = tile < 0 ? cuda::gemmTileFor(M) : static_cast<cuda::GemmTile>(tile);
    cuda::launchGemm(transB, M, N, K, dA.data(), dB.data(), dC.data(), t, s.get());
    download(C, dC, c, s);
    finish(s);
}

void transposeInput(const float* input, int B, int planes, int area, float* out) {
    Context::instance();
    const size_t count = static_cast<size_t>(B) * planes * area;
    Stream s;
    DeviceBuffer in(count), d(count);
    upload(in, input, count, s);
    cuda::launchTransposeInput(in.data(), B, planes, area, d.data(), s.get());
    download(out, d, count, s);
    finish(s);
}

void im2col3(const float* in, int C, int B, int n, float* col) {
    Context::instance();
    const size_t count = static_cast<size_t>(C) * B * n * n;
    Stream s;
    DeviceBuffer x(count), d(count * 9);
    upload(x, in, count, s);
    cuda::launchIm2col3(x.data(), C, B, n, d.data(), s.get());
    download(col, d, count * 9, s);
    finish(s);
}

void bnInfer(const BatchNorm2d& bn, const float* x, int M, float* y) {
    Context::instance();
    const size_t count = static_cast<size_t>(bn.c) * M, C = static_cast<size_t>(bn.c);
    Stream s;
    DeviceBuffer dx(count), gamma(C), beta(C), mean(C), var(C);
    upload(dx, x, count, s);
    upload(gamma, bn.gamma.data(), C, s);
    upload(beta, bn.beta.data(), C, s);
    upload(mean, bn.runMean.data(), C, s);
    upload(var, bn.runVar.data(), C, s);
    cuda::launchBnInfer(dx.data(), dx.data(), bn.c, static_cast<size_t>(M), gamma.data(), beta.data(), mean.data(),
                        var.data(), s.get());
    download(y, dx, count, s);
    finish(s);
}

void relu(float* x, size_t count) {
    Context::instance();
    Stream s;
    DeviceBuffer d(count);
    upload(d, x, count, s);
    cuda::launchRelu(d.data(), count, s.get());
    download(x, d, count, s);
    finish(s);
}

void addInto(float* dst, const float* src, size_t count) {
    Context::instance();
    Stream s;
    DeviceBuffer a(count), b(count);
    upload(a, dst, count, s);
    upload(b, src, count, s);
    cuda::launchAddInto(a.data(), b.data(), count, s.get());
    download(dst, a, count, s);
    finish(s);
}

void policyFlatten(const float* act, int B, int A, float* flat) {
    Context::instance();
    const size_t count = static_cast<size_t>(2) * B * A;
    Stream s;
    DeviceBuffer in(count), out(count);
    upload(in, act, count, s);
    cuda::launchPolicyFlatten(in.data(), B, A, out.data(), s.get());
    download(flat, out, count, s);
    finish(s);
}

void biasAddRows(float* y, const float* b, int rows, int cols) {
    Context::instance();
    const size_t count = static_cast<size_t>(rows) * cols;
    Stream s;
    DeviceBuffer dy(count), db(static_cast<size_t>(cols));
    upload(dy, y, count, s);
    upload(db, b, static_cast<size_t>(cols), s);
    cuda::launchBiasAddRows(dy.data(), db.data(), rows, cols, s.get());
    download(y, dy, count, s);
    finish(s);
}

void tanhInPlace(float* x, size_t count) {
    Context::instance();
    Stream s;
    DeviceBuffer d(count);
    upload(d, x, count, s);
    cuda::launchTanh(d.data(), count, s.get());
    download(x, d, count, s);
    finish(s);
}

} // namespace cuda_probe
} // namespace azgo
