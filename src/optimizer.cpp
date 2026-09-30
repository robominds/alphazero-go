#include "azgo/optimizer.hpp"
#include <cassert>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace azgo {

// ---------------------------------------------------------------- SgdMomentum

SgdMomentum::SgdMomentum(Network& net, float momentum, float weightDecay)
    : momentum_(momentum), weightDecay_(weightDecay) {
    for (const ParamView& p : net.params()) velocity_.emplace_back(p.size, 0.0f);
}

void SgdMomentum::step(Network& net, float lr) {
    std::vector<ParamView> params = net.params();
    assert(params.size() == velocity_.size());
    for (size_t k = 0; k < params.size(); ++k) {
        ParamView& p = params[k];
        std::vector<float>& v = velocity_[k];
        assert(v.size() == p.size);
        // Decay only weights: shrinking BN's gamma/beta or a bias toward zero
        // does not make the function smoother, it just fights the fit.
        const float decay = p.decay ? weightDecay_ : 0.0f;
        for (size_t i = 0; i < p.size; ++i) {
            const float g = p.grad[i] + decay * p.value[i];
            v[i] = momentum_ * v[i] + g;
            p.value[i] -= lr * v[i];
        }
    }
}

// ---------------------------------------------------------------- LrSchedule

namespace {

std::string trim(const std::string& s) {
    const size_t first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const size_t last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

[[noreturn]] void fail(const std::string& text, const std::string& why) {
    throw std::invalid_argument("learning-rate schedule '" + text + "': " + why);
}

// The shortest decimal that reads back as exactly v ("0.02", not
// "0.0199999996"), so toString() output is canonical and round-trips.
std::string formatFloat(float v) {
    char buf[32];
    for (int precision = 1; precision <= 9; ++precision) {
        std::snprintf(buf, sizeof buf, "%.*g", precision, static_cast<double>(v));
        if (std::strtof(buf, nullptr) == v) break;
    }
    return buf;
}

} // namespace

LrSchedule LrSchedule::parse(const std::string& text) {
    if (trim(text).empty()) fail(text, "empty");
    LrSchedule schedule;
    size_t start = 0;
    while (true) {
        const size_t comma = text.find(',', start);
        const std::string entry =
            trim(text.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        const size_t colon = entry.find(':');
        if (colon == std::string::npos) fail(text, "entry '" + entry + "' is not generation:lr");
        const std::string genText = trim(entry.substr(0, colon));
        const std::string lrText = trim(entry.substr(colon + 1));

        // strtol/strtof must consume the whole field: "5x" or "" is an error.
        char* end = nullptr;
        errno = 0;
        const long generation = std::strtol(genText.c_str(), &end, 10);
        if (genText.empty() || *end != '\0' || errno == ERANGE || generation < 0 || generation > INT_MAX)
            fail(text, "bad generation '" + genText + "'");
        errno = 0;
        const float lr = std::strtof(lrText.c_str(), &end);
        if (lrText.empty() || *end != '\0' || errno == ERANGE || !std::isfinite(lr))
            fail(text, "bad learning rate '" + lrText + "'");
        if (!(lr > 0.0f)) fail(text, "learning rate " + lrText + " is not > 0");
        if (schedule.steps_.empty() && generation != 0) fail(text, "the first entry must be at generation 0");
        if (!schedule.steps_.empty() && generation <= schedule.steps_.back().first)
            fail(text, "generations must be strictly increasing");
        schedule.steps_.emplace_back(static_cast<int>(generation), lr);

        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return schedule;
}

float LrSchedule::at(int generation) const {
    assert(!steps_.empty());
    float lr = steps_.front().second;
    for (const auto& step : steps_) {
        if (step.first > generation) break;
        lr = step.second;
    }
    return lr;
}

std::string LrSchedule::toString() const {
    std::string out;
    for (const auto& step : steps_) {
        if (!out.empty()) out += ',';
        out += std::to_string(step.first) + ':' + formatFloat(step.second);
    }
    return out;
}

} // namespace azgo
