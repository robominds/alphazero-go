#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "azgo/network.hpp"
#include "azgo/optimizer.hpp"

using namespace azgo;

static bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

namespace {

NetConfig tinyConfig() {
    NetConfig cfg;
    cfg.boardSize = 5;
    cfg.history = 1;
    cfg.blocks = 0;
    cfg.filters = 2;
    cfg.valueHidden = 2;
    return cfg;
}

size_t indexOf(const std::vector<ParamView>& params, const std::string& name) {
    for (size_t i = 0; i < params.size(); ++i)
        if (params[i].name == name) return i;
    assert(false);
    return 0;
}

bool throwsInvalidArgument(const std::string& text) {
    try {
        LrSchedule::parse(text);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

static void test_momentum_and_weight_decay_hand_values() {
    Network net(tinyConfig(), 1);
    SgdMomentum opt(net, /*momentum=*/0.9f, /*weightDecay=*/0.1f);
    net.zeroGrad();
    std::vector<ParamView> params = net.params();
    const size_t wi = indexOf(params, "stem.conv.w");   // decayed
    const size_t bi = indexOf(params, "value.fc2.b");   // not decayed
    const size_t gi = indexOf(params, "stem.bn.gamma"); // not decayed, zero gradient
    ParamView& w = params[wi];
    ParamView& b = params[bi];
    assert(w.decay && !b.decay && !params[gi].decay);
    w.value[0] = 1.0f;
    w.grad[0] = 0.5f;
    b.value[0] = 1.0f;
    b.grad[0] = 0.5f;
    const float w1 = w.value[1];                        // zero gradient: pure decay
    const float gamma0 = params[gi].value[0];

    opt.step(net, 0.5f);
    // Weight: g' = 0.5 + 0.1 * 1 = 0.6; v = 0.6; w = 1 - 0.5 * 0.6 = 0.7.
    // Bias:   g' = 0.5;                  v = 0.5; b = 1 - 0.5 * 0.5 = 0.75.
    assert(near(w.value[0], 0.7));
    assert(near(b.value[0], 0.75));

    opt.step(net, 0.5f);   // step() leaves the gradients in place
    // Weight: g' = 0.5 + 0.1 * 0.7 = 0.57; v = 0.9 * 0.6 + 0.57 = 1.11; w = 0.7 - 0.555 = 0.145.
    // Bias:   g' = 0.5;                    v = 0.9 * 0.5 + 0.5  = 0.95; b = 0.75 - 0.475 = 0.275.
    assert(near(w.value[0], 0.145));
    assert(near(b.value[0], 0.275));
    assert(near(opt.velocity()[wi][0], 1.11));
    assert(near(opt.velocity()[bi][0], 0.95));
    // Pure decay: w -> 0.95 w (v = 0.1 w), then v = 0.09 w + 0.095 w = 0.185 w, w -> 0.8575 w.
    assert(near(w.value[1], 0.8575 * w1));
    assert(params[gi].value[0] == gamma0);              // no gradient, no decay: untouched
}

static void test_velocity_shapes() {
    Network net(tinyConfig(), 2);
    SgdMomentum opt(net, 0.9f, 1e-4f);
    std::vector<ParamView> params = net.params();
    assert(opt.velocity().size() == params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        assert(opt.velocity()[i].size() == params[i].size);
        for (float v : opt.velocity()[i]) assert(v == 0.0f);
    }
}

static void test_velocity_assignment_round_trip() {
    // Resuming training restores the velocity into a fresh optimizer; the
    // next step must then be exactly what the original optimizer would do.
    Network a(tinyConfig(), 3);
    SgdMomentum optA(a, 0.9f, 0.1f);
    auto setGradients = [](Network& net) {
        for (ParamView& p : net.params())
            for (size_t i = 0; i < p.size; ++i) p.grad[i] = 0.01f * static_cast<float>(i % 7) - 0.03f;
    };
    setGradients(a);
    optA.step(a, 0.1f);
    optA.step(a, 0.1f);

    Network restored = a;
    SgdMomentum optRestored(restored, 0.9f, 0.1f);
    optRestored.velocity() = optA.velocity();
    Network fresh = a;
    SgdMomentum optFresh(fresh, 0.9f, 0.1f);   // no restored velocity

    optA.step(a, 0.1f);
    optRestored.step(restored, 0.1f);
    optFresh.step(fresh, 0.1f);
    std::vector<ParamView> pa = a.params(), pr = restored.params(), pf = fresh.params();
    bool freshDiffers = false;
    for (size_t k = 0; k < pa.size(); ++k) {
        for (size_t i = 0; i < pa[k].size; ++i) {
            assert(pa[k].value[i] == pr[k].value[i]);
            freshDiffers = freshDiffers || pa[k].value[i] != pf[k].value[i];
        }
    }
    assert(freshDiffers);
    assert(optRestored.velocity() == optA.velocity());
}

static void test_lr_schedule_lookup() {
    LrSchedule s = LrSchedule::parse("0:0.02,50:0.002");
    assert(s.at(0) == 0.02f);
    assert(s.at(49) == 0.02f);
    assert(s.at(50) == 0.002f);
    assert(s.at(1000) == 0.002f);
    LrSchedule one = LrSchedule::parse("0:0.1");
    assert(one.at(0) == 0.1f && one.at(123456) == 0.1f);
}

static void test_lr_schedule_whitespace_and_canonical_form() {
    LrSchedule s = LrSchedule::parse("  0 : 0.02 ,\t50:0.002 , 100 :1e-4 ");
    assert(s.at(75) == 0.002f && s.at(100) == 1e-4f);
    const std::string canonical = s.toString();
    assert(canonical == "0:0.02,50:0.002,100:0.0001");
    assert(LrSchedule::parse(canonical).toString() == canonical);
}

static void test_lr_schedule_errors() {
    assert(throwsInvalidArgument(""));
    assert(throwsInvalidArgument("   "));
    assert(throwsInvalidArgument("5:0.1"));            // must start at generation 0
    assert(throwsInvalidArgument("0:0.1,0:0.2"));      // not strictly increasing
    assert(throwsInvalidArgument("0:0.1,20:0.2,10:0.3"));
    assert(throwsInvalidArgument("0:0.1,10:-1"));      // lr <= 0
    assert(throwsInvalidArgument("0:0"));
    assert(throwsInvalidArgument("0:abc"));
    assert(throwsInvalidArgument("0:0.1x"));
    assert(throwsInvalidArgument("0:0.1,5"));          // entry without ':'
    assert(throwsInvalidArgument("0:0.1,"));           // trailing comma
    assert(throwsInvalidArgument("-1:0.1"));
}

int main() {
    test_momentum_and_weight_decay_hand_values();
    test_velocity_shapes();
    test_velocity_assignment_round_trip();
    test_lr_schedule_lookup();
    test_lr_schedule_whitespace_and_canonical_form();
    test_lr_schedule_errors();
    std::printf("test_optimizer: all tests passed\n");
    return 0;
}
