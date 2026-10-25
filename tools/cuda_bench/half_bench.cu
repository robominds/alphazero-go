// half_bench: the 16-bit tensor-core convolution of backend = cuda16
// (src/cuda/fp16/kernels16.cu, compiled into this program as it is) on its
// own: speed per trunk convolution for fp16 and bf16 at batches 8 to 256,
// its error against a double-precision reference on the same 16-bit
// operands, and the batch-independence and tile-shape checks, bitwise.
// It is the 16-bit counterpart of cuda_bench's gemm and det sections; the
// whole pass, the runner and the hub are measured by bench --backend cuda16.
//
//   tools/cuda_bench/build.sh          # also builds build/half_bench
//   tools/cuda_bench/build/half_bench [--seconds S] [--net TEXT]
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include "../../src/cuda/fp16/kernels16.cu"
// The cuda backend's fp32 fused convolution, for the speed comparison under
// the same conditions (build with -I include).
#include "../../src/cuda/kernels.cu"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace azgo::cuda16;

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "half_bench: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Dev {
    void* p = nullptr;
    size_t bytes = 0;
    explicit Dev(size_t b) : bytes(b) { check(cudaMalloc(&p, b ? b : 1), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    float* f() const { return static_cast<float*>(p); }
};

std::vector<float> randomVector(std::mt19937& rng, size_t count, float scale) {
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(count);
    for (float& x : v) x = d(rng);
    return v;
}

void upload(Dev& d, const std::vector<float>& v) {
    check(cudaMemcpy(d.p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice), "upload");
}

std::vector<float> download(const Dev& d, size_t count) {
    std::vector<float> v(count);
    check(cudaMemcpy(v.data(), d.p, count * sizeof(float), cudaMemcpyDeviceToHost), "download");
    return v;
}

std::vector<uint16_t> download16(const Dev& d, size_t count) {
    std::vector<uint16_t> v(count);
    check(cudaMemcpy(v.data(), d.p, count * 2, cudaMemcpyDeviceToHost), "download");
    return v;
}

// One layer's operands on the GPU, for B positions: input NHWC 16-bit, the
// weights converted, scale and shift, a skip, and both outputs.
struct Layer {
    Half type;
    int B, n, cin, cout, cinP, coutP;
    Dev x32, x16, w32, w16, scale, shift, skip32, skip16, out16, out32;
    Layer(Half t, int B_, int n_, int cin_, int cout_)
        : type(t), B(B_), n(n_), cin(cin_), cout(cout_), cinP(paddedChannels(cin_)), coutP(paddedChannels(cout_)),
          x32(size_t(B_) * cin_ * n_ * n_ * 4), x16(size_t(B_) * n_ * n_ * paddedChannels(cin_) * 2),
          w32(size_t(cout_) * cin_ * 9 * 4), w16(size_t(paddedChannels(cout_)) * 9 * paddedChannels(cin_) * 2),
          scale(paddedChannels(cout_) * 4), shift(paddedChannels(cout_) * 4),
          skip32(size_t(B_) * cout_ * n_ * n_ * 4), skip16(size_t(B_) * n_ * n_ * paddedChannels(cout_) * 2),
          out16(size_t(B_) * n_ * n_ * paddedChannels(cout_) * 2), out32(size_t(B_) * cout_ * n_ * n_ * 4) {}
    int pixels() const { return B * n * n; }
    // x [B][cin][area] and skip [B][cout][area] fp32, as the stage layout.
    void set(const std::vector<float>& x, const std::vector<float>& w, const std::vector<float>& sc,
             const std::vector<float>& sh, const std::vector<float>& skip) {
        upload(x32, x);
        upload(w32, w);
        std::vector<float> s(coutP, 0.0f), h(coutP, 0.0f);
        std::copy(sc.begin(), sc.end(), s.begin());
        std::copy(sh.begin(), sh.end(), h.begin());
        upload(scale, s);
        upload(shift, h);
        upload(skip32, skip);
        launchInputToNhwc(type, x32.f(), B, cin, n * n, cinP, x16.p, 0);
        launchWeightsTo16(type, w32.f(), cout, cin, coutP, cinP, w16.p, 0);
        launchInputToNhwc(type, skip32.f(), B, cout, n * n, coutP, skip16.p, 0);
        check(cudaDeviceSynchronize(), "set");
    }
    Conv16Args args(bool withSkip, bool to16, bool to32) const {
        Conv16Args a;
        a.in = x16.p;
        a.w = w16.p;
        a.batch = B;
        a.n = n;
        a.cinPadded = cinP;
        a.coutPadded = coutP;
        a.cout = cout;
        a.scale = scale.f();
        a.shift = shift.f();
        a.skip = withSkip ? skip16.p : nullptr;
        a.out16 = to16 ? out16.p : nullptr;
        a.out32 = to32 ? out32.f() : nullptr;
        return a;
    }
    void run(bool withSkip, int ni = -1) {
        launchConv16(type, args(withSkip, true, true), ni < 0 ? channelTileFor(coutP) : ni, 0);
        check(cudaGetLastError(), "conv16");
        check(cudaDeviceSynchronize(), "conv16 sync");
    }
};

double secondsSince(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

const char* typeName(Half t) { return t == Half::Fp16 ? "fp16" : "bf16"; }

struct Net {
    const char* name;
    int n, filters, blocks;
};
const Net kNets[] = {{"9x9 6x96", 9, 96, 6}, {"13x13 6x96", 13, 96, 6}, {"19x19 10x128", 19, 128, 10}};

// Speed of one trunk convolution (with skip) and the trunk it implies.
void speed(double seconds, const std::string& filter) {
    std::printf("== conv16 speed: one trunk convolution (K = 9 x filters, + skip), 16-bit out ==\n");
    std::printf("%-14s %-5s %6s %10s %10s %14s %16s\n", "net", "type", "batch", "ms/conv", "TFLOP/s", "trunk ms/pass",
                "trunk evals/s");
    std::mt19937 rng(11);
    for (const Net& net : kNets) {
        if (!filter.empty() && std::string(net.name).find(filter) == std::string::npos) continue;
        for (Half t : {Half::Fp16, Half::Bf16})
            for (int B : {8, 32, 64, 128, 256}) {
                Layer L(t, B, net.n, net.filters, net.filters);
                const size_t A = size_t(net.n) * net.n;
                L.set(randomVector(rng, B * net.filters * A, 1.0f), randomVector(rng, net.filters * net.filters * 9, 0.05f),
                      randomVector(rng, net.filters, 0.5f), randomVector(rng, net.filters, 0.1f),
                      randomVector(rng, B * net.filters * A, 1.0f));
                const Conv16Args a = L.args(true, true, false);
                const int ni = channelTileFor(L.coutP);
                // The fp32 fused kernel on the same shape (channel-major
                // operands; values do not matter for its speed).
                azgo::cuda::ConvArgs cv;
                cv.in = L.x32.f();
                cv.batch = B;
                cv.n = net.n;
                cv.scale = L.scale.f();
                cv.shift = L.shift.f();
                cv.skip = L.skip32.f();
                Dev w32(size_t(net.filters) * net.filters * 9 * 4), x32(size_t(B) * net.filters * A * 4),
                    out32(size_t(B) * net.filters * A * 4);
                check(cudaMemset(w32.p, 0, w32.bytes), "memset");
                check(cudaMemset(x32.p, 0, x32.bytes), "memset");
                cv.in = x32.f();
                cv.skip = nullptr;
                const double flop = 2.0 * L.pixels() * net.filters * 9.0 * net.filters;
                for (int variant = 0; variant < 2; ++variant) {
                    if (variant == 0 && t == Half::Bf16) continue;   // fp32 once per batch
                    auto launch = [&] {
                        if (variant == 1) launchConv16(t, a, ni, 0);
                        else azgo::cuda::launchConvBnRelu(w32.f(), net.filters, net.filters, cv, out32.f(),
                                                          azgo::cuda::gemmTileFor(net.filters), 0);
                    };
                    for (int i = 0; i < 3; ++i) launch();
                    check(cudaDeviceSynchronize(), "warm");
                    int calls = 0;
                    const auto start = std::chrono::steady_clock::now();
                    double s = 0;
                    do {
                        for (int i = 0; i < 10; ++i) launch();
                        check(cudaDeviceSynchronize(), "time");
                        calls += 10;
                        s = secondsSince(start);
                    } while (s < seconds);
                    const double ms = 1000.0 * s / calls;
                    const double trunkMs = ms * (2 * net.blocks + 1);   // the stem's K is smaller; counted as a block's
                    std::printf("%-14s %-5s %6d %10.4f %10.1f %14.3f %16.0f\n", net.name,
                                variant == 0 ? "fp32" : typeName(t), B, ms, flop / (ms * 1e-3) / 1e12, trunkMs,
                                B / (trunkMs * 1e-3));
                    std::fflush(stdout);
                }
            }
    }
}

// Rounds through the type and back, as the GPU stores it.
float roundTo(Half t, float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    if (t == Half::Bf16) {
        const uint32_t lsb = (u >> 16) & 1;
        u = (u + 0x7FFF + lsb) & 0xFFFF0000u;
        float r;
        std::memcpy(&r, &u, 4);
        return r;
    }
    return __half2float(__float2half_rn(v));   // host versions (cuda_fp16.h)
}

// The error against double precision on the same 16-bit operands, per
// output: |gpu - ref| / sum |products| (the scale of a dot product's rounding).
void accuracy(const std::string& filter) {
    std::printf("\n== conv16 accuracy: out32 vs double on the 16-bit operands, |err| / (sum|products| * |scale|) ==\n");
    std::mt19937 rng(12);
    for (const Net& net : kNets) {
        if (!filter.empty() && std::string(net.name).find(filter) == std::string::npos) continue;
        for (Half t : {Half::Fp16, Half::Bf16})
            for (int cin : {18, net.filters}) {
                const int B = 3, n = net.n, cout = net.filters, A = n * n;
                Layer L(t, B, n, cin, cout);
                const std::vector<float> x = randomVector(rng, size_t(B) * cin * A, 1.0f),
                                         w = randomVector(rng, size_t(cout) * cin * 9, 0.05f),
                                         sc = randomVector(rng, cout, 0.5f), sh = randomVector(rng, cout, 0.1f),
                                         sk = randomVector(rng, size_t(B) * cout * A, 1.0f);
                L.set(x, w, sc, sh, sk);
                L.run(true);
                const std::vector<float> got = download(L.out32, size_t(cout) * B * A);
                double worst = 0, worstAbs = 0;
                for (int b = 0; b < B; ++b)
                    for (int m = 0; m < cout; ++m)
                        for (int p = 0; p < A; ++p) {
                            double sum = 0, mag = 0;
                            for (int c = 0; c < cin; ++c)
                                for (int r = 0; r < 9; ++r) {
                                    const int y = p / n + r / 3 - 1, xx = p % n + r % 3 - 1;
                                    if (y < 0 || y >= n || xx < 0 || xx >= n) continue;
                                    const double prod = double(roundTo(t, w[(size_t(m) * cin + c) * 9 + r])) *
                                                        roundTo(t, x[(size_t(b) * cin + c) * A + y * n + xx]);
                                    sum += prod;
                                    mag += std::fabs(prod);
                                }
                            double v = double(sc[m]) * sum + sh[m] + roundTo(t, sk[(size_t(b) * cout + m) * A + p]);
                            v = v > 0 ? v : 0;
                            const double g = got[(size_t(m) * B + b) * A + p];
                            // got is [cout][B * A] channel-major: index m * (B * A) + b * A + p.
                            const double e = std::fabs(g - v);
                            worstAbs = std::max(worstAbs, e);
                            const double scale = mag * std::fabs(sc[m]) + std::fabs(sh[m]) + 1e-30;
                            worst = std::max(worst, e / scale);
                        }
                std::printf("%-14s %-5s cin %3d: max |err| %.3g, max |err| / scale %.3g (2^-23 = %.3g)\n", net.name,
                            typeName(t), cin, worstAbs, worst, std::ldexp(1.0, -23));
            }
    }
}

// A sample's outputs alone and inside batches of 8, 144 and 256 at three
// places, among random companions; a repeat; and every channel tile.
void determinism(const std::string& filter) {
    std::printf("\n== conv16 determinism: a sample alone vs inside a batch (bitwise) ==\n");
    std::mt19937 rng(13);
    for (const Net& net : kNets) {
        if (!filter.empty() && std::string(net.name).find(filter) == std::string::npos) continue;
        for (Half t : {Half::Fp16, Half::Bf16})
            for (int cin : {18, net.filters}) {
                const int n = net.n, cout = net.filters, A = n * n, coutP = paddedChannels(cout);
                const std::vector<float> w = randomVector(rng, size_t(cout) * cin * 9, 0.05f),
                                         sc = randomVector(rng, cout, 0.5f), sh = randomVector(rng, cout, 0.1f);
                const std::vector<float> one = randomVector(rng, size_t(cin) * A, 1.0f),
                                         oneSkip = randomVector(rng, size_t(cout) * A, 1.0f);
                Layer alone(t, 1, n, cin, cout);
                alone.set(one, w, sc, sh, oneSkip);
                alone.run(true);
                const std::vector<uint16_t> ref16 = download16(alone.out16, size_t(A) * coutP);
                const std::vector<float> ref32 = download(alone.out32, size_t(cout) * A);
                std::printf("%-14s %-5s cin %3d:", net.name, typeName(t), cin);
                bool all = true;
                for (int B : {8, 144, 256})
                    for (int at : {0, B / 2 + 1, B - 1}) {
                        std::vector<float> x = randomVector(rng, size_t(B) * cin * A, 1.0f),
                                           sk = randomVector(rng, size_t(B) * cout * A, 1.0f);
                        std::copy(one.begin(), one.end(), x.begin() + size_t(at) * cin * A);
                        std::copy(oneSkip.begin(), oneSkip.end(), sk.begin() + size_t(at) * cout * A);
                        Layer L(t, B, n, cin, cout);
                        L.set(x, w, sc, sh, sk);
                        std::vector<int> tiles = {channelTileFor(coutP)};
                        if (B == 144) tiles = {2, 4, 6, 8};
                        for (int ni : tiles) {
                            if (coutP % (16 * ni)) continue;
                            L.run(true, ni);
                            const std::vector<uint16_t> g16 = download16(L.out16, size_t(B) * A * coutP);
                            const std::vector<float> g32 = download(L.out32, size_t(cout) * B * A);
                            bool same = std::memcmp(g16.data() + size_t(at) * A * coutP, ref16.data(), ref16.size() * 2) == 0;
                            for (int m = 0; m < cout && same; ++m)
                                same = std::memcmp(&g32[(size_t(m) * B + at) * A], &ref32[size_t(m) * A], A * 4) == 0;
                            if (!same) {
                                all = false;
                                std::printf(" B=%d@%d/ni%d DIFFER", B, at, ni);
                            }
                        }
                    }
                // A repeat of the single sample.
                alone.run(true);
                if (download16(alone.out16, size_t(A) * coutP) != ref16) {
                    all = false;
                    std::printf(" repeat DIFFERS");
                }
                std::printf(all ? " all equal (B=8/144/256 at 3 places, tiles, repeat) => batch-independent\n" : "\n");
                std::fflush(stdout);
            }
    }
}

} // namespace

int main(int argc, char** argv) {
    double seconds = 0.3;
    std::string filter, only;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--seconds" && i + 1 < argc) seconds = std::atof(argv[++i]);
        else if (a == "--net" && i + 1 < argc) filter = argv[++i];
        else only = a;
    }
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0), "properties");
    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    std::printf("gpu: %s, compute %d.%d, %d SMs; CUDA runtime %d, driver %d\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount, runtime, driver);
    if (only.empty() || only == "det") determinism(filter);
    if (only.empty() || only == "accuracy") accuracy(filter);
    if (only.empty() || only == "speed") speed(seconds, filter);
    return 0;
}
