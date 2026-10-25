// cuda16_accuracy: how far backend = cuda16 is from the network's exact
// outputs on real positions of a real run, beside the 32-bit engines. It
// loads a checkpoint, replays the run's SGF games, encodes every k-th
// position (as self-play encodes it, identity symmetry), and evaluates them
// with the network in double precision (the reference), the CPU in fp32
// (blas), the cuda backend (fp32) and cuda16 with fp16 and with bf16
// operands. It prints, per engine against the reference:
//
//   logits  max |d|, mean |d|, and the worst position's max |d| / max |ref|
//   policy  (softmax over the legal moves) argmax agreement, mean and max KL(ref || x)
//   value   max |d|, mean |d|, sign agreement where |ref| > 0.05
//
// and the largest activation in the trunk (fp16's range is 65,504). The
// design's tolerance tier comes from these numbers
// (docs/superpowers/specs/2026-10-24-cuda-fp16-design.md, "Accuracy").
//
// Built in a CUDA build tree (not by default):
//   cmake --build build-cuda --target cuda16_accuracy
//   build-cuda/cuda16_accuracy --checkpoint gen_0100.azn --sgf-dir sgf [--every 4] [--max 512]
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "azgo/backend.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/cuda.hpp"
#include "azgo/cuda16.hpp"
#include "azgo/encode.hpp"
#include "azgo/game.hpp"
#include "azgo/layers.hpp"
#include "azgo/network.hpp"
#include "azgo/sgf.hpp"

using namespace azgo;

namespace {

struct Position {
    std::vector<float> planes;   // [planes][area]
    std::vector<int> legal;      // legal actions
};

// One position through the network in double precision: logits, value
// (after tanh), and the largest trunk activation seen.
struct Exact {
    std::vector<double> logits;
    double value = 0, maxActivation = 0;
};

class DoubleNet {
public:
    explicit DoubleNet(Network& net) : cfg_(net.config()) {
        for (const ParamView& p : net.params()) t_[p.name].assign(p.value, p.value + p.size);
        for (const BufferView& v : net.buffers()) t_[v.name].assign(v.value, v.value + v.size);
    }

    Exact run(const std::vector<float>& input) const {
        const int n = cfg_.boardSize, A = cfg_.area(), P = cfg_.planes(), F = cfg_.filters;
        Exact e;
        std::vector<double> x(input.begin(), input.end());   // [P][A]
        auto conv = [&](const std::vector<double>& in, int cin, int cout, int k, const std::string& name) {
            const std::vector<double>& w = t_.at(name + ".w");
            std::vector<double> y(static_cast<size_t>(cout) * A, 0.0);
            for (int o = 0; o < cout; ++o)
                for (int c = 0; c < cin; ++c)
                    for (int dy = 0; dy < k; ++dy)
                        for (int dx = 0; dx < k; ++dx) {
                            const double wv = w[((static_cast<size_t>(o) * cin + c) * k + dy) * k + dx];
                            for (int p = 0; p < A; ++p) {
                                const int yy = p / n + dy - k / 2, xx = p % n + dx - k / 2;
                                if (yy < 0 || yy >= n || xx < 0 || xx >= n) continue;
                                y[static_cast<size_t>(o) * A + p] += wv * in[static_cast<size_t>(c) * A + yy * n + xx];
                            }
                        }
            return y;
        };
        auto norm = [&](std::vector<double>& v, int C, const std::string& name) {
            for (int c = 0; c < C; ++c) {
                const double scale =
                    t_.at(name + ".gamma")[c] / std::sqrt(t_.at(name + ".runVar")[c] + BatchNorm2d::kEps);
                for (int i = 0; i < A; ++i) {
                    double& u = v[static_cast<size_t>(c) * A + i];
                    u = (u - t_.at(name + ".runMean")[c]) * scale + t_.at(name + ".beta")[c];
                }
            }
        };
        auto relu = [&](std::vector<double>& v, bool trunk) {
            for (double& u : v) {
                u = u > 0 ? u : 0;
                if (trunk) e.maxActivation = std::max(e.maxActivation, u);
            }
        };
        auto linear = [&](const std::vector<double>& in, int nin, int nout, const std::string& name) {
            const std::vector<double>&w = t_.at(name + ".w"), &b = t_.at(name + ".b");
            std::vector<double> y(nout);
            for (int o = 0; o < nout; ++o) {
                double s = b[o];
                for (int i = 0; i < nin; ++i) s += in[i] * w[static_cast<size_t>(o) * nin + i];
                y[o] = s;
            }
            return y;
        };
        std::vector<double> h = conv(x, P, F, 3, "stem.conv");
        norm(h, F, "stem.bn");
        relu(h, true);
        for (int i = 0; i < cfg_.blocks; ++i) {
            const std::string pre = "block" + std::to_string(i);
            std::vector<double> mid = conv(h, F, F, 3, pre + ".conv1");
            norm(mid, F, pre + ".bn1");
            relu(mid, true);
            std::vector<double> out = conv(mid, F, F, 3, pre + ".conv2");
            norm(out, F, pre + ".bn2");
            for (size_t j = 0; j < out.size(); ++j) out[j] += h[j];
            relu(out, true);
            h = out;
        }
        std::vector<double> pol = conv(h, F, 2, 1, "policy.conv");
        norm(pol, 2, "policy.bn");
        relu(pol, false);
        e.logits = linear(pol, 2 * A, cfg_.actions(), "policy.fc");
        std::vector<double> val = conv(h, F, 1, 1, "value.conv");
        norm(val, 1, "value.bn");
        relu(val, false);
        std::vector<double> hidden = linear(val, A, cfg_.valueHidden, "value.fc1");
        relu(hidden, false);
        e.value = std::tanh(linear(hidden, cfg_.valueHidden, 1, "value.fc2")[0]);
        return e;
    }

private:
    NetConfig cfg_;
    std::map<std::string, std::vector<double>> t_;
};

std::vector<Position> loadPositions(const NetConfig& cfg, const std::vector<std::string>& files, int every, int max) {
    std::vector<Position> out;
    for (const std::string& f : files) {
        std::ifstream in(f);
        std::stringstream ss;
        ss << in.rdbuf();
        const SgfGame g = parseSgf(ss.str());
        if (g.boardSize != cfg.boardSize) continue;
        Game game(g.boardSize, g.komi, defaultMaxMoves(g.boardSize));
        for (size_t ply = 0; ply <= g.moves.size(); ++ply) {
            if (game.isTerminal()) break;
            if (static_cast<int>(ply) % every == 0) {
                Position p;
                p.planes.resize(static_cast<size_t>(cfg.planes()) * cfg.area());
                encodeGame(game, cfg.history, 0, p.planes.data());
                p.legal = game.legalActions();
                out.push_back(std::move(p));
                if (static_cast<int>(out.size()) >= max) return out;
            }
            if (ply == g.moves.size()) break;
            game.play(g.moves[ply]);
        }
    }
    return out;
}

struct Outputs {
    std::vector<float> logits, values;
};

// Every position through an engine, in batches of 64.
template <class Infer> Outputs evaluate(const NetConfig& cfg, const std::vector<Position>& ps, Infer infer) {
    Outputs o;
    o.logits.resize(ps.size() * cfg.actions());
    o.values.resize(ps.size());
    const size_t in = static_cast<size_t>(cfg.planes()) * cfg.area();
    for (size_t first = 0; first < ps.size(); first += 64) {
        const int B = static_cast<int>(std::min<size_t>(64, ps.size() - first));
        std::vector<float> x(B * in);
        for (int b = 0; b < B; ++b) std::copy(ps[first + b].planes.begin(), ps[first + b].planes.end(), x.begin() + b * in);
        infer(x.data(), B, o.logits.data() + first * cfg.actions(), o.values.data() + first);
    }
    return o;
}

std::vector<double> softmaxLegal(const double* logits, const std::vector<int>& legal) {
    double top = -1e300;
    for (int a : legal) top = std::max(top, logits[a]);
    std::vector<double> p(legal.size());
    double sum = 0;
    for (size_t i = 0; i < legal.size(); ++i) sum += p[i] = std::exp(logits[legal[i]] - top);
    for (double& v : p) v /= sum;
    return p;
}

void report(const char* name, const NetConfig& cfg, const std::vector<Position>& ps, const std::vector<Exact>& ref,
            const Outputs& o) {
    const size_t L = cfg.actions();
    double lMax = 0, lSum = 0, lRel = 0, vMax = 0, vSum = 0, klSum = 0, klMax = 0;
    long long lCount = 0;
    int argmaxSame = 0, signCount = 0, signSame = 0;
    for (size_t i = 0; i < ps.size(); ++i) {
        double posMax = 0, refMax = 0;
        std::vector<double> got(L);
        for (size_t a = 0; a < L; ++a) {
            got[a] = o.logits[i * L + a];
            const double d = std::fabs(got[a] - ref[i].logits[a]);
            lMax = std::max(lMax, d);
            lSum += d;
            ++lCount;
            posMax = std::max(posMax, d);
            refMax = std::max(refMax, std::fabs(ref[i].logits[a]));
        }
        lRel = std::max(lRel, refMax > 0 ? posMax / refMax : posMax);
        const std::vector<double> p = softmaxLegal(ref[i].logits.data(), ps[i].legal), q = softmaxLegal(got.data(), ps[i].legal);
        double kl = 0;
        size_t bestP = 0, bestQ = 0;
        for (size_t k = 0; k < p.size(); ++k) {
            if (p[k] > 0) kl += p[k] * std::log(p[k] / std::max(q[k], 1e-300));
            if (p[k] > p[bestP]) bestP = k;
            if (q[k] > q[bestQ]) bestQ = k;
        }
        klSum += kl;
        klMax = std::max(klMax, kl);
        argmaxSame += bestP == bestQ ? 1 : 0;
        const double dv = std::fabs(o.values[i] - ref[i].value);
        vMax = std::max(vMax, dv);
        vSum += dv;
        if (std::fabs(ref[i].value) > 0.05) {
            ++signCount;
            signSame += (o.values[i] > 0) == (ref[i].value > 0) ? 1 : 0;
        }
    }
    const double N = static_cast<double>(ps.size());
    std::printf("  %-12s logits max %.2e mean %.2e rel %.2e | argmax %6.2f%% KL mean %.2e max %.2e | "
                "value max %.2e mean %.2e sign %6.2f%% (%d)\n",
                name, lMax, lSum / lCount, lRel, 100.0 * argmaxSame / N, klSum / N, klMax, vMax, vSum / N,
                signCount ? 100.0 * signSame / signCount : 100.0, signCount);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    std::string checkpoint, sgfDir;
    int every = 4, max = 512, threads = static_cast<int>(std::thread::hardware_concurrency());
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--checkpoint") checkpoint = v;
        else if (a == "--sgf-dir") sgfDir = v;
        else if (a == "--every") every = std::atoi(v.c_str());
        else if (a == "--max") max = std::atoi(v.c_str());
        else if (a == "--threads") threads = std::atoi(v.c_str());
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (checkpoint.empty() || sgfDir.empty() || every < 1 || max < 1 || threads < 1) {
        std::fprintf(stderr, "usage: cuda16_accuracy --checkpoint CKPT --sgf-dir DIR [--every 4] [--max 512] "
                             "[--threads N]\n");
        return 2;
    }
    try {
        Network net = loadNetwork(checkpoint);
        const NetConfig cfg = net.config();
        std::vector<std::string> files;
        for (const auto& e : std::filesystem::directory_iterator(sgfDir))
            if (e.path().extension() == ".sgf") files.push_back(e.path().string());
        std::sort(files.rbegin(), files.rend());   // the latest games first
        const std::vector<Position> ps = loadPositions(cfg, files, every, max);
        std::printf("%s: %dx%d %dx%d, %zu positions (every %d-th ply, latest games first, of %zu)\n",
                    std::filesystem::path(checkpoint).filename().c_str(), cfg.boardSize, cfg.boardSize, cfg.blocks,
                    cfg.filters, ps.size(), every, files.size());
        std::printf("gpu: %s\n      %s\n", cudaDescription().c_str(), cuda16Description().c_str());
        std::fflush(stdout);

        // The reference, on `threads` threads.
        std::vector<Exact> ref(ps.size());
        const DoubleNet exact(net);
        std::atomic<size_t> next{0};
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t)
            pool.emplace_back([&] {
                for (size_t i = next++; i < ps.size(); i = next++) ref[i] = exact.run(ps[i].planes);
            });
        for (std::thread& th : pool) th.join();
        double maxAct = 0;
        for (const Exact& e : ref) maxAct = std::max(maxAct, e.maxActivation);
        std::printf("largest trunk activation (double): %.1f (fp16 max 65504)\n", maxAct);

        auto blas = makeBackend("blas", 1);
        Workspace ws(cfg, 64, *blas);
        const Outputs cpu = evaluate(cfg, ps, [&](const float* x, int B, float* l, float* v) {
            net.infer(ws, x, B, l, v);
        });
        CudaInference gpu32(net, 64);
        const Outputs c32 = evaluate(cfg, ps, [&](const float* x, int B, float* l, float* v) {
            gpu32.infer(x, B, l, v);
        });
        Cuda16Inference fp16(net, 64, Cuda16Type::Fp16), bf16(net, 64, Cuda16Type::Bf16);
        const Outputs h16 = evaluate(cfg, ps, [&](const float* x, int B, float* l, float* v) { fp16.infer(x, B, l, v); });
        const Outputs b16 = evaluate(cfg, ps, [&](const float* x, int B, float* l, float* v) { bf16.infer(x, B, l, v); });

        std::printf("against the network in double precision:\n");
        report("blas (fp32)", cfg, ps, ref, cpu);
        report("cuda (fp32)", cfg, ps, ref, c32);
        report("cuda16 fp16", cfg, ps, ref, h16);
        report("cuda16 bf16", cfg, ps, ref, b16);
        // Against the 32-bit CPU, as a run sees it: the CPU's fp32 outputs as the reference.
        std::vector<Exact> cpuRef(ps.size());
        for (size_t i = 0; i < ps.size(); ++i) {
            cpuRef[i].logits.assign(cpu.logits.begin() + i * cfg.actions(), cpu.logits.begin() + (i + 1) * cfg.actions());
            cpuRef[i].value = cpu.values[i];
        }
        std::printf("against the CPU in fp32 (blas):\n");
        report("cuda (fp32)", cfg, ps, cpuRef, c32);
        report("cuda16 fp16", cfg, ps, cpuRef, h16);
        report("cuda16 bf16", cfg, ps, cpuRef, b16);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
