// cuda16_probe (azgo/cuda16.hpp): the 16-bit kernels on host arrays, for the
// tests and bench. Each probe copies its inputs in, runs on its own stream,
// waits and copies the results out.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/cuda.hpp"
#include "azgo/cuda16.hpp"
#include "../cuda_context.cuh"
#include "cuda16_network.cuh"

namespace azgo {
namespace cuda16_probe {

namespace {

void requireGpu() {
    std::string why;
    if (!cuda16Available(&why)) throw std::runtime_error("cuda16: " + why);
}

cuda16::Half halfOf(Cuda16Type t) { return t == Cuda16Type::Bf16 ? cuda16::Half::Bf16 : cuda16::Half::Fp16; }

// [C][B][A] -> [B][C][A]: the stage layout the input conversion reads.
std::vector<float> sampleMajor(const float* x, int C, int B, int A) {
    std::vector<float> out(static_cast<size_t>(C) * B * A);
    for (int c = 0; c < C; ++c)
        for (int b = 0; b < B; ++b)
            for (int a = 0; a < A; ++a)
                out[(static_cast<size_t>(b) * C + c) * A + a] = x[(static_cast<size_t>(c) * B + b) * A + a];
    return out;
}

void upload(cuda::DeviceBuffer& d, const float* host, size_t count, const cuda::Stream& s) {
    cuda::check(cudaMemcpyAsync(d.data(), host, count * sizeof(float), cudaMemcpyHostToDevice, s.get()), "upload");
}

} // namespace

void conv16(const Conv2d& conv, const BatchNorm2d& bn, const float* x, const float* skip, int B, int n, float* y,
            float* y16, int tile, Cuda16Type type) {
    requireGpu();
    if (conv.k != 3 || bn.c != conv.cout) throw std::invalid_argument("cuda16_probe::conv16: a 3x3 conv and its BN");
    if (tile != -1 && tile != 2 && tile != 4 && tile != 6 && tile != 8)
        throw std::invalid_argument("cuda16_probe::conv16: tile must be -1, 2, 4, 6 or 8");
    const cuda16::Half h = halfOf(type);
    const int A = n * n, cin = conv.cin, cout = conv.cout;
    const int cinP = cuda16::paddedChannels(cin), coutP = cuda16::paddedChannels(cout);
    if (tile > 0 && coutP % (16 * tile) != 0)
        throw std::invalid_argument("cuda16_probe::conv16: the tile does not divide the padded channels");
    const size_t pixels = static_cast<size_t>(B) * A;
    cuda::Stream s;
    std::vector<cuda::DeviceBuffer> staging;
    cuda16::Conv16Weights w;
    w.upload(h, conv.w.data(), bn.gamma.data(), bn.beta.data(), bn.runMean.data(), bn.runVar.data(), cin, cout, staging,
             s.get());
    const std::vector<float> xs = sampleMajor(x, cin, B, A);
    cuda::DeviceBuffer dx(xs.size()), dskip(skip ? static_cast<size_t>(cout) * pixels : 0),
        dy(static_cast<size_t>(cout) * pixels), dwide(pixels * coutP);
    cuda16::HalfBuffer x16(pixels * cinP), skip16(skip ? pixels * coutP : 0), out16(pixels * coutP);
    upload(dx, xs.data(), xs.size(), s);
    cuda16::launchInputToNhwc(h, dx.data(), B, cin, A, cinP, x16.data(), s.get());
    std::vector<float> ss;
    if (skip) {
        ss = sampleMajor(skip, cout, B, A);
        upload(dskip, ss.data(), ss.size(), s);
        cuda16::launchInputToNhwc(h, dskip.data(), B, cout, A, coutP, skip16.data(), s.get());
    }
    cuda16::Conv16Args a;
    a.in = x16.data();
    a.w = w.w16.data();
    a.batch = B;
    a.n = n;
    a.cinPadded = cinP;
    a.coutPadded = coutP;
    a.cout = cout;
    a.scale = w.scale();
    a.shift = w.shift();
    a.skip = skip ? skip16.data() : nullptr;
    a.out16 = out16.data();
    a.out32 = dy.data();
    cuda16::launchConv16(h, a, tile < 0 ? cuda16::channelTileFor(coutP) : tile, s.get());
    cuda16::launchWiden(h, out16.data(), pixels * coutP, dwide.data(), s.get());
    cuda::check(cudaGetLastError(), "conv16");
    std::vector<float> wide(pixels * coutP);
    cuda::check(cudaMemcpyAsync(y, dy.data(), static_cast<size_t>(cout) * pixels * sizeof(float),
                                cudaMemcpyDeviceToHost, s.get()),
                "download");
    cuda::check(cudaMemcpyAsync(wide.data(), dwide.data(), wide.size() * sizeof(float), cudaMemcpyDeviceToHost, s.get()),
                "download");
    s.synchronize();
    if (y16)
        for (int c = 0; c < cout; ++c)
            for (size_t p = 0; p < pixels; ++p) y16[c * pixels + p] = wide[p * coutP + c];
}

void roundTo16(float* x, size_t count, Cuda16Type type) {
    requireGpu();
    if (count == 0) return;
    const cuda16::Half h = halfOf(type);
    cuda::Stream s;
    cuda::DeviceBuffer d(count), back(count);
    cuda16::HalfBuffer v(count);
    upload(d, x, count, s);
    // One "plane" of count values: the input conversion rounds each one.
    cuda16::launchInputToNhwc(h, d.data(), 1, 1, static_cast<int>(count), 1, v.data(), s.get());
    cuda16::launchWiden(h, v.data(), count, back.data(), s.get());
    cuda::check(cudaGetLastError(), "roundTo16");
    cuda::check(cudaMemcpyAsync(x, back.data(), count * sizeof(float), cudaMemcpyDeviceToHost, s.get()), "download");
    s.synchronize();
}

void checkUpload(const Network& layoutNet, const Network& net) {
    requireGpu();
    cuda16::Cuda16Weights w(ParamLayout::of(const_cast<Network&>(layoutNet)), layoutNet.config(), cuda16::Half::Fp16);
    w.upload(net);
}

void checkForwardReadsEveryTensor(const Network& net, const NetConfig& encodeAs) {
    requireGpu();
    cuda16::Cuda16Weights w(ParamLayout::of(const_cast<Network&>(net)), net.config(), cuda16::Half::Fp16);
    w.upload(net);
    cuda16::Cuda16Workspace ws(encodeAs, 1);
    cuda::Stream s;
    try {
        cuda16::Cuda16Network(encodeAs).encodeForward(s.get(), ws, w, 1);
    } catch (...) {
        s.synchronize();
        throw;
    }
    s.synchronize();
}

} // namespace cuda16_probe
} // namespace azgo
