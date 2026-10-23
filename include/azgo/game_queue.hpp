// The queue of a generation's self-play games, for dynamic allocation
// (selfplay_allocation = dynamic). Static allocation fixes before a
// generation starts which engine plays which game; the generation then
// lasts as long as its slowest engine. Here the games wait in one queue,
// each engine keeps up to its concurrency of them in flight and asks for
// another whenever one ends, so a fast engine simply takes more.
//
// A plain "whoever asks first" queue balances the middle of a generation
// but not its end: a game handed out late to an engine whose games take
// long ends long after everything else. So take() hands engine e a game
// only when e will finish it in time, or when the other engines cannot:
//
//   λ_f    each engine's mean latency so far (handout to finished game);
//   bound  the latest predicted end of a game in flight anywhere,
//          max(start + λ_f), and at least now: the generation lasts that
//          long whatever e does;
//   give   if λ_e is not known yet, or now + λ_e <= bound;
//   else   count C, the games the other engines can still start and finish
//          by bound (an in-flight game whose slot frees in time, a free
//          slot), and give only if more than C games are queued.
//
// It never stalls: a slot whose game runs late stops counting as time
// passes, and with nothing in flight bound = now and C = 0.
//
// The queue decides only which engine plays a game, never what the game
// is: game i is seeded by i wherever it is played, and the records are
// merged by index. It is pure bookkeeping, with no threads and no clock
// (every call takes `now`, seconds since the generation began), so a
// simulation can drive the very code the coordinator runs, and the
// coordinator guards it with its own mutex.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <deque>
#include <map>
#include <set>
#include <vector>

namespace azgo {

class GameQueue {
public:
    static constexpr int kWait = -1;   // ask again later
    static constexpr int kNone = -2;   // every game is finished: the engine's work is over

    // Games [0, games), every one queued; engine e (0, 1, ...) keeps at
    // most concurrency[e] in flight (each at least 1).
    GameQueue(int games, std::vector<int> concurrency);
    // Forced assignment, to replay a recorded run: engine engineOfGame[i]
    // plays game i, each engine its games in index order, and the handout
    // rule does not apply. Engines may not retire.
    GameQueue(const std::vector<int>& engineOfGame, std::vector<int> concurrency);

    // A game for `engine` to start at `now`: the lowest queued index (so a
    // requeued game, lower than any never handed out, comes next), kWait,
    // or kNone. A retired engine gets kNone.
    int take(int engine, double now);
    // The engine finished `index` (verified) at `now`. std::logic_error if
    // it does not hold that game.
    void finished(int engine, int index, double now);
    // The engine's game goes back to the queue (its engine failed).
    // std::logic_error if it does not hold that game.
    void requeue(int engine, int index);
    // The engine takes no more games. The games it holds stay with it until
    // the caller requeues them.
    void retire(int engine);
    // A new engine (the fallback engine, when every other engine retired);
    // returns its number. Not in forced mode.
    int addEngine(int concurrency);
    // Off: a plain first-come queue (an engine with a free slot always gets
    // the next game). For comparison only: the simulation and the tests.
    void setHandoutRule(bool on) { rule_ = on; }

    int games() const { return static_cast<int>(owner_.size()); }
    int engines() const { return static_cast<int>(engines_.size()); }
    int queued() const { return static_cast<int>(pending_.size()) + forcedQueued_; }
    int finishedGames() const { return finished_; }
    bool done() const { return finished_ == games(); }
    bool retired(int engine) const;
    int inFlight(int engine) const;
    // The indices the engine holds, ascending.
    std::vector<int> held(int engine) const;
    // Games the engine finished.
    int finishedBy(int engine) const;
    // The engine that finished game i, or -1.
    int playedBy(int index) const;
    // The engine's mean latency so far, or a negative number before its
    // first finished game.
    double latency(int engine) const;

private:
    struct Engine {
        int concurrency = 1;
        bool retired = false;
        std::map<int, double> inFlight;   // index -> handout time
        double latencySum = 0;
        int latencyCount = 0, finished = 0;
        std::deque<int> forced;           // forced mode: its queued games, in order
    };
    Engine& at(int engine);
    const Engine& at(int engine) const;
    bool inTime(int engine, double now) const;   // the handout rule

    std::vector<Engine> engines_;
    std::set<int> pending_;     // queued, lowest first
    std::vector<int> owner_;    // -1 queued, engine holding it, or kFinished
    std::vector<int> player_;   // engine that finished it, or -1
    int finished_ = 0;
    bool forcedMode_ = false;
    bool rule_ = true;
    int forcedQueued_ = 0;
};

} // namespace azgo
