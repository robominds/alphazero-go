#include "azgo/game_queue.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace azgo {

namespace {
constexpr int kQueued = -1, kFinished = -3;
}

GameQueue::GameQueue(int games, std::vector<int> concurrency) {
    if (games < 0) throw std::logic_error("GameQueue: negative game count");
    if (concurrency.empty()) throw std::logic_error("GameQueue: no engines");
    for (int c : concurrency) addEngine(c);
    owner_.assign(static_cast<size_t>(games), kQueued);
    player_.assign(static_cast<size_t>(games), -1);
    for (int i = 0; i < games; ++i) pending_.insert(pending_.end(), i);
}

GameQueue::GameQueue(const std::vector<int>& engineOfGame, std::vector<int> concurrency)
    : GameQueue(0, std::move(concurrency)) {
    forcedMode_ = true;
    owner_.assign(engineOfGame.size(), kQueued);
    player_.assign(engineOfGame.size(), -1);
    for (size_t i = 0; i < engineOfGame.size(); ++i) {
        const int e = engineOfGame[i];
        if (e < 0 || e >= engines())
            throw std::logic_error("GameQueue: game " + std::to_string(i) + " forced to engine " +
                                   std::to_string(e) + ", which does not exist");
        at(e).forced.push_back(static_cast<int>(i));
        ++forcedQueued_;
    }
}

GameQueue::Engine& GameQueue::at(int engine) {
    if (engine < 0 || engine >= engines())
        throw std::logic_error("GameQueue: no engine " + std::to_string(engine));
    return engines_[static_cast<size_t>(engine)];
}

const GameQueue::Engine& GameQueue::at(int engine) const {
    if (engine < 0 || engine >= engines())
        throw std::logic_error("GameQueue: no engine " + std::to_string(engine));
    return engines_[static_cast<size_t>(engine)];
}

int GameQueue::addEngine(int concurrency) {
    if (concurrency < 1) throw std::logic_error("GameQueue: an engine's concurrency must be at least 1");
    if (forcedMode_) throw std::logic_error("GameQueue: no engine can join a forced assignment");
    Engine e;
    e.concurrency = concurrency;
    engines_.push_back(e);
    return engines() - 1;
}

bool GameQueue::inTime(int engine, double now) const {
    const double mine = latency(engine);
    if (mine < 0) return true;   // nothing known yet: the start of a generation
    // The latest predicted end of a game in flight: the generation lasts at
    // least that long, whatever this engine does.
    double bound = now;
    for (int f = 0; f < engines(); ++f) {
        const double lf = latency(f);
        if (lf < 0) continue;
        for (const auto& [index, start] : at(f).inFlight) bound = std::max(bound, start + lf);
    }
    if (now + mine <= bound) return true;
    // The games the other engines can still start and finish by then: a
    // slot whose game is predicted to end early enough to play one more,
    // and a free slot.
    long long absorb = 0;
    for (int f = 0; f < engines(); ++f) {
        const Engine& other = at(f);
        const double lf = latency(f);
        if (f == engine || other.retired || lf < 0) continue;
        for (const auto& [index, start] : other.inFlight)
            if (std::max(now, start + lf) + lf <= bound) ++absorb;
        const int free = other.concurrency - static_cast<int>(other.inFlight.size());
        if (free > 0 && now + lf <= bound) absorb += free;
    }
    return queued() > absorb;
}

int GameQueue::take(int engine, double now) {
    Engine& e = at(engine);
    if (done() || e.retired) return kNone;
    if (static_cast<int>(e.inFlight.size()) >= e.concurrency) return kWait;
    int index;
    if (forcedMode_) {
        if (e.forced.empty()) return kWait;
        index = e.forced.front();
        e.forced.pop_front();
        --forcedQueued_;
    } else {
        if (pending_.empty()) return kWait;
        if (rule_ && !inTime(engine, now)) return kWait;
        index = *pending_.begin();
        pending_.erase(pending_.begin());
    }
    owner_[static_cast<size_t>(index)] = engine;
    e.inFlight[index] = now;
    return index;
}

void GameQueue::finished(int engine, int index, double now) {
    Engine& e = at(engine);
    auto it = e.inFlight.find(index);
    if (it == e.inFlight.end())
        throw std::logic_error("GameQueue: engine " + std::to_string(engine) + " finished game " +
                               std::to_string(index) + ", which it does not hold");
    e.latencySum += std::max(0.0, now - it->second);
    ++e.latencyCount;
    ++e.finished;
    e.inFlight.erase(it);
    owner_[static_cast<size_t>(index)] = kFinished;
    player_[static_cast<size_t>(index)] = engine;
    ++finished_;
}

void GameQueue::requeue(int engine, int index) {
    Engine& e = at(engine);
    auto it = e.inFlight.find(index);
    if (it == e.inFlight.end())
        throw std::logic_error("GameQueue: engine " + std::to_string(engine) + " requeued game " +
                               std::to_string(index) + ", which it does not hold");
    e.inFlight.erase(it);
    owner_[static_cast<size_t>(index)] = kQueued;
    if (forcedMode_) {
        // Back to its own engine, in index order.
        e.forced.insert(std::lower_bound(e.forced.begin(), e.forced.end(), index), index);
        ++forcedQueued_;
    } else {
        pending_.insert(index);
    }
}

void GameQueue::retire(int engine) {
    if (forcedMode_)
        throw std::logic_error("GameQueue: an engine of a forced assignment cannot retire");
    at(engine).retired = true;
}

bool GameQueue::retired(int engine) const { return at(engine).retired; }

int GameQueue::inFlight(int engine) const { return static_cast<int>(at(engine).inFlight.size()); }

std::vector<int> GameQueue::held(int engine) const {
    std::vector<int> out;
    for (const auto& [index, start] : at(engine).inFlight) out.push_back(index);
    return out;
}

int GameQueue::finishedBy(int engine) const { return at(engine).finished; }

int GameQueue::playedBy(int index) const {
    if (index < 0 || index >= games()) throw std::logic_error("GameQueue: no game " + std::to_string(index));
    return player_[static_cast<size_t>(index)];
}

double GameQueue::latency(int engine) const {
    const Engine& e = at(engine);
    return e.latencyCount > 0 ? e.latencySum / e.latencyCount : -1.0;
}

} // namespace azgo
