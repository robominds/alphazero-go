#include "azgo/trainer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>

#include "azgo/arena.hpp"
#include "azgo/backend.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/elo.hpp"
#include "azgo/evaluator.hpp"
#include "azgo/game.hpp"
#include "azgo/optimizer.hpp"
#include "azgo/parallel.hpp"
#include "azgo/selfplay.hpp"
#include "azgo/serialize.hpp"
#include "azgo/sgf.hpp"

namespace azgo {

namespace fs = std::filesystem;

namespace {

const char* const kLogHeader =
    "generation,games,avg_length,pass_rate,black_win_rate,policy_loss,value_loss,"
    "lr,replay_ratio,selfplay_s,train_s,arena_s,elo,winrate_vs_prev,distinct_vs_prev\n";
const char* const kEloHeader = "generation,elo,winrate_vs_prev,wr_low,wr_high\n";

using Clock = std::chrono::steady_clock;

double secondsSince(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string logPath(const std::string& runDir) { return (fs::path(runDir) / "log.csv").string(); }
std::string eloPath(const std::string& runDir) { return (fs::path(runDir) / "elo.csv").string(); }
std::string resultsPath(const std::string& runDir) { return (fs::path(runDir) / "results.csv").string(); }

std::string sgfPath(const std::string& runDir, int generation, int game) {
    char name[64];
    std::snprintf(name, sizeof name, "gen_%04d_game_%02d.sgf", generation, game);
    return (fs::path(runDir) / "sgf" / name).string();
}

// elo.csv: one row per rated generation, ascending. The win rate columns come
// from the (g-1, g) pair of the results table: how often g beat its
// predecessor, with a 95% Wilson interval. Empty when the pair was never played.
std::string eloCsv(const ResultsTable& results) {
    std::map<std::pair<int, int>, ResultsTable::Pair> byPair;
    for (const ResultsTable::Pair& p : results.pairs()) byPair[{p.a, p.b}] = p;

    std::string out = kEloHeader;
    for (const auto& [generation, rating] : fitBradleyTerry(results, 0)) {
        const double elo = rating + 0.0;   // turns a -0.0 into 0.0, so the anchor prints "0.0"
        char line[160];
        auto it = byPair.find({generation - 1, generation});
        if (it != byPair.end() && it->second.games > 0) {
            const int games = it->second.games;
            const double score = games - it->second.scoreA;   // scoreA is g-1's score
            const std::pair<double, double> ci = wilsonInterval(score, games);
            std::snprintf(line, sizeof line, "%d,%.1f,%.4f,%.4f,%.4f\n", generation, elo,
                          score / games, ci.first, ci.second);
        } else {
            std::snprintf(line, sizeof line, "%d,%.1f,,,\n", generation, elo);
        }
        out += line;
    }
    return out;
}

// Rating of `generation`, or of the latest rated generation before it when it
// has not played yet (no arena this generation), or 0 before any arena round.
double ratingAtOrBefore(const ResultsTable& results, int generation) {
    if (results.empty()) return 0.0;
    const std::map<int, double> ratings = fitBradleyTerry(results, 0);
    auto it = ratings.upper_bound(generation);
    if (it == ratings.begin()) return 0.0;
    return std::prev(it)->second;
}

void appendLogRow(const std::string& runDir, const GenerationStats& s) {
    char winRate[32] = "";
    if (s.winRateVsPrev >= 0) std::snprintf(winRate, sizeof winRate, "%.4f", s.winRateVsPrev);
    char distinct[32] = "";
    if (s.distinctVsPrev >= 0) std::snprintf(distinct, sizeof distinct, "%d", s.distinctVsPrev);
    char line[512];
    std::snprintf(line, sizeof line,
                  "%d,%d,%.2f,%.4f,%.4f,%.5f,%.5f,%g,%.3f,%.3f,%.3f,%.3f,%.1f,%s,%s\n",
                  s.generation, s.games, s.avgLength, s.passRate, s.blackWinRate,
                  s.policyLoss, s.valueLoss, s.lr, s.replayRatio, s.selfPlaySeconds,
                  s.trainSeconds, s.arenaSeconds, s.elo, winRate, distinct);
    std::ofstream out(logPath(runDir), std::ios::app | std::ios::binary);
    out << line;
    if (!out) throw std::runtime_error("cannot append to " + logPath(runDir));
}

} // namespace

Trainer::Trainer(const Config& cfg, const std::string& runDir, Network net, TrainingState state)
    : cfg_(cfg), runDir_(runDir), net_(std::move(net)), state_(std::move(state)),
      log_(&std::cout) {}

std::unique_ptr<Trainer> Trainer::startNew(const Config& cfg, const std::string& runDir) {
    try {
        LrSchedule::parse(cfg.lrSchedule);
    } catch (const std::invalid_argument& e) {
        throw ConfigError(std::string("lr_schedule: ") + e.what());
    }
    if (fs::exists(statePath(runDir)))
        throw ConfigError(runDir + " already contains a run; use --resume");

    fs::create_directories(fs::path(runDir) / "checkpoints");
    fs::create_directories(fs::path(runDir) / "sgf");
    writeFileAtomically(configPath(runDir), configToString(cfg));

    Network net(cfg.netConfig(), deriveSeed(cfg.seed, 0, kPhaseInit));
    saveNetwork(net, checkpointPath(runDir, 0));

    TrainingState state;
    state.buffer = ReplayBuffer(static_cast<size_t>(cfg.bufferPositions));
    state.velocity = SgdMomentum(net, cfg.momentum, cfg.weightDecay).velocity();   // all zeros
    state.references = initialReferences(static_cast<int>(cfg.botSimulations.size()));

    writeFileAtomically(logPath(runDir), kLogHeader);
    writeFileAtomically(eloPath(runDir), std::string(kEloHeader) + "0,0,,,\n");
    // state.bin last: until it exists the directory is not a run, so a crash
    // anywhere above leaves something startNew can simply redo.
    saveTrainingState(statePath(runDir), state);

    return std::unique_ptr<Trainer>(new Trainer(cfg, runDir, std::move(net), std::move(state)));
}

std::unique_ptr<Trainer> Trainer::resume(const std::string& runDir,
                                         const std::vector<std::string>& overrides) {
    if (!fs::exists(statePath(runDir)))
        throw ConfigError(runDir + " does not contain a run (no state.bin); start one with --config");

    const Config saved = loadConfigFile(configPath(runDir));
    Config cfg = saved;
    for (const std::string& o : overrides) applyOverride(cfg, o);

    const std::vector<std::string> diff = resumeLockedDiff(saved, cfg);
    if (!diff.empty()) {
        std::string keys;
        for (const std::string& k : diff) keys += (keys.empty() ? "" : ", ") + k;
        // resumeLockedDiff lists bot_simulations last, after any architecture keys.
        if (diff.back() == "bot_simulations")
            throw ConfigError("cannot change keys on resume: " + keys +
                              " (bot k is arena player -k, so the bot list is fixed for a run)");
        throw ConfigError("cannot change architecture keys on resume: " + keys);
    }

    // A version-1 state.bin (written before bots existed) gets every
    // configured bot as a reference.
    TrainingState state =
        loadTrainingState(statePath(runDir), static_cast<int>(cfg.botSimulations.size()));
    const std::string ckpt = checkpointPath(runDir, state.generation);
    Network net = loadNetwork(ckpt);
    if (net.config() != cfg.netConfig())
        throw ConfigError(ckpt + " does not match the architecture in " + configPath(runDir));

    // The momentum buffers must line up with the parameters one-to-one, or the
    // first optimizer step would read out of bounds.
    const std::vector<ParamView> params = net.params();
    bool velocityOk = state.velocity.size() == params.size();
    for (size_t i = 0; velocityOk && i < params.size(); ++i)
        velocityOk = state.velocity[i].size() == params[i].size;
    if (!velocityOk)
        throw std::runtime_error(statePath(runDir) +
                                 ": optimizer state does not match the network's parameters");

    state.buffer.setCapacity(static_cast<size_t>(cfg.bufferPositions));
    // Rewritten only after every check passed, so a refused resume leaves the
    // run directory untouched.
    writeFileAtomically(configPath(runDir), configToString(cfg));

    return std::unique_ptr<Trainer>(new Trainer(cfg, runDir, std::move(net), std::move(state)));
}

GenerationStats Trainer::runGeneration() {
    const int g = state_.generation + 1;
    const int workers = cfg_.effectiveWorkers();
    const int n = cfg_.boardSize;
    GenerationStats stats;
    stats.generation = g;

    // ---- 1. Self-play -----------------------------------------------------
    const auto selfPlayStart = Clock::now();
    const int numGames = cfg_.gamesPerGeneration;
    std::vector<GameRecord> records(static_cast<size_t>(numGames));
    {
        // With gating, games are generated by the best network so far, which
        // may be older than the one being trained.
        std::unique_ptr<Network> bestNet;
        if (cfg_.gating && state_.bestGeneration != state_.generation)
            bestNet = std::make_unique<Network>(
                loadNetwork(checkpointPath(runDir_, state_.bestGeneration)));
        const Network& selfPlayNet = bestNet ? *bestNet : net_;

        SelfPlayConfig sp;
        sp.mcts.simulations = cfg_.simulations;
        sp.mcts.batchSize = cfg_.mctsBatch;
        sp.mcts.cPuct = cfg_.cPuct;
        sp.mcts.dirichletScale = cfg_.dirichletScale;
        sp.mcts.noiseFraction = cfg_.noiseFraction;
        sp.temperatureMoves = cfg_.temperatureMoves;
        sp.komi = cfg_.komi;
        sp.maxMoves = cfg_.effectiveMaxMoves();

        // One single-threaded backend and evaluator per worker; the network
        // itself is shared read-only.
        std::vector<std::unique_ptr<Backend>> backends;
        std::vector<std::unique_ptr<LocalEvaluator>> evaluators;
        for (int w = 0; w < workers; ++w) {
            backends.push_back(makeBackend(cfg_.backend, 1));
            evaluators.push_back(
                std::make_unique<LocalEvaluator>(selfPlayNet, *backends.back(), cfg_.mctsBatch));
        }
        // Game i's randomness depends only on i, and its record is stored at
        // index i, so the result does not depend on which thread played it.
        parallelFor(numGames, workers, [&](int i, int w) {
            Rng rng(deriveSeed(cfg_.seed, static_cast<uint64_t>(g), kPhaseSelfPlay,
                               static_cast<uint64_t>(i)));
            records[static_cast<size_t>(i)] = playSelfPlayGame(*evaluators[static_cast<size_t>(w)], sp, rng);
        });
    }

    long long plies = 0, passes = 0;
    int blackWins = 0;
    for (const GameRecord& r : records) {
        plies += r.positions();
        for (int a : r.moves)
            if (a == n * n) ++passes;
        if (r.resultForBlack > 0) ++blackWins;
    }
    stats.games = numGames;
    stats.avgLength = static_cast<double>(plies) / numGames;
    stats.passRate = plies > 0 ? static_cast<double>(passes) / plies : 0.0;
    stats.blackWinRate = static_cast<double>(blackWins) / numGames;

    // A few games per generation as SGF. The record stores only the result's
    // sign, so the Tromp-Taylor margin for RE[] comes from replaying the moves.
    const int sgfGames = std::min(cfg_.sgfPerGeneration, numGames);
    for (int i = 0; i < sgfGames; ++i) {
        const GameRecord& r = records[static_cast<size_t>(i)];
        Game replay(n, r.komi, cfg_.effectiveMaxMoves());
        for (int a : r.moves) replay.play(a);
        const std::string comment = "azgo self-play, generation " + std::to_string(g) +
                                    ", game " + std::to_string(i);
        writeFileAtomically(sgfPath(runDir_, g, i),
                            gameToSgf(n, r.komi, r.moves, replay.scoreForBlack(), comment));
    }

    for (GameRecord& r : records) state_.buffer.add(std::move(r));   // index order
    stats.selfPlaySeconds = secondsSince(selfPlayStart);

    // ---- 2. Train -----------------------------------------------------------
    const auto trainStart = Clock::now();
    const float lr = LrSchedule::parse(cfg_.lrSchedule).at(g);
    {
        // The naive backend splits gemm rows over `workers` threads here; each
        // output element is still computed by one thread, so results match.
        std::unique_ptr<Backend> backend = makeBackend(cfg_.backend, workers);
        Workspace ws(net_.config(), cfg_.batchSize, *backend);
        SgdMomentum opt(net_, cfg_.momentum, cfg_.weightDecay);
        opt.velocity() = state_.velocity;
        Rng rng(deriveSeed(cfg_.seed, static_cast<uint64_t>(g), kPhaseTrain));
        TrainingBatch batch;
        double policySum = 0, valueSum = 0;
        for (int s = 0; s < cfg_.trainSteps; ++s) {
            state_.buffer.sample(cfg_.batchSize, cfg_.history, rng, batch);
            const LossStats loss = net_.trainStep(ws, batch.input.data(), batch.pi.data(),
                                                  batch.z.data(), batch.size);
            // Nothing of generation g is committed yet (state.bin is the
            // commit point), so the last good state (generation g-1) stays.
            if (!std::isfinite(loss.policyLoss) || !std::isfinite(loss.valueLoss))
                throw std::runtime_error("non-finite loss at generation " + std::to_string(g) +
                                         " step " + std::to_string(s) +
                                         "; last good state kept");
            opt.step(net_, lr);
            policySum += loss.policyLoss;
            valueSum += loss.valueLoss;
        }
        if (cfg_.trainSteps > 0) {
            stats.policyLoss = static_cast<float>(policySum / cfg_.trainSteps);
            stats.valueLoss = static_cast<float>(valueSum / cfg_.trainSteps);
        }
        // The per-step loss check cannot see the weights written by the last
        // optimizer step, so scan the parameters and momentum before anything
        // is saved.
        auto allFinite = [](const float* v, size_t count) {
            for (size_t k = 0; k < count; ++k)
                if (!std::isfinite(v[k])) return false;
            return true;
        };
        bool finite = true;
        for (const ParamView& p : net_.params()) finite = finite && allFinite(p.value, p.size);
        for (const std::vector<float>& v : opt.velocity()) finite = finite && allFinite(v.data(), v.size());
        if (!finite)
            throw std::runtime_error("non-finite weights after training generation " +
                                     std::to_string(g) + "; last good state kept");
        state_.velocity = opt.velocity();
    }
    const long long drawn = static_cast<long long>(cfg_.trainSteps) * cfg_.batchSize;
    state_.samplesTrained += drawn;
    stats.lr = lr;
    stats.replayRatio = plies > 0 ? static_cast<double>(drawn) / plies : 0.0;
    stats.trainSeconds = secondsSince(trainStart);

    // ---- 3. Checkpoint ----------------------------------------------------
    // Written before state.bin (step 5). A crash in between leaves state.bin at
    // generation g-1; rerunning then reproduces generation g exactly (every
    // random stream is derived from the seed) and overwrites this file with
    // identical bytes. The same holds for the SGF, results.csv and elo.csv.
    saveNetwork(net_, checkpointPath(runDir_, g));

    // ---- 4. Arena and Elo ---------------------------------------------------
    const auto arenaStart = Clock::now();
    int newBest = cfg_.gating ? state_.bestGeneration : g;
    const bool arenaThisGeneration = cfg_.arenaEvery > 0 && g % cfg_.arenaEvery == 0;
    if (arenaThisGeneration) {
        std::vector<int> opponents = ladderOpponents(g, cfg_.arenaOpponents);
        if (cfg_.gating &&
            std::find(opponents.begin(), opponents.end(), state_.bestGeneration) == opponents.end())
            opponents.push_back(state_.bestGeneration);

        ArenaConfig ac;
        ac.games = cfg_.arenaGames;
        ac.simulations = cfg_.arenaSimulations;
        ac.mctsBatch = cfg_.mctsBatch;
        ac.cPuct = cfg_.cPuct;
        ac.temperatureMoves = cfg_.arenaTemperatureMoves;
        ac.komi = cfg_.komi;
        ac.maxMoves = cfg_.effectiveMaxMoves();

        for (int o : opponents) {
            const Network opponent = loadNetwork(checkpointPath(runDir_, o));
            const MatchResult r = playMatch(net_, opponent, ac,
                                            deriveSeed(cfg_.seed, static_cast<uint64_t>(g),
                                                       kPhaseArena, static_cast<uint64_t>(o)),
                                            workers, cfg_.backend);
            state_.results.addMatch(g, o, r.games, r.scoreA);
            if (o == g - 1) {
                stats.winRateVsPrev = r.scoreA / r.games;
                stats.distinctVsPrev = r.distinctGames;
            }
            if (cfg_.gating && o == state_.bestGeneration &&
                passesGate(r.scoreA, r.games, cfg_.gatingThreshold))
                newBest = g;
        }
        // Refit every rating from all results so far: Bradley-Terry does not
        // depend on the order results arrived in, so a resumed run agrees.
        writeFileAtomically(resultsPath(runDir_), state_.results.toCsv());
        writeFileAtomically(eloPath(runDir_), eloCsv(state_.results));
    }
    stats.elo = ratingAtOrBefore(state_.results, g);
    stats.arenaSeconds = secondsSince(arenaStart);

    // ---- 5. Training state --------------------------------------------------
    state_.bestGeneration = newBest;
    state_.generation = g;
    saveTrainingState(statePath(runDir_), state_);
    // log.csv is a human-facing record (it contains timings) and is appended
    // after state.bin: a crash between the two loses one row, never corrupts
    // the run.
    appendLogRow(runDir_, stats);

    if (log_) {
        char line[512];
        int len = std::snprintf(line, sizeof line,
                                "gen %d | selfplay %d games avg %.1f plies %.1fs | "
                                "train loss p %.2f v %.2f lr %g %.1fs | ",
                                g, stats.games, stats.avgLength, stats.selfPlaySeconds,
                                stats.policyLoss, stats.valueLoss, stats.lr, stats.trainSeconds);
        if (arenaThisGeneration) {
            char vsPrev[32] = "n/a";
            if (stats.winRateVsPrev >= 0)
                std::snprintf(vsPrev, sizeof vsPrev, "%.2f", stats.winRateVsPrev);
            char distinct[48] = "";
            if (stats.distinctVsPrev >= 0)
                std::snprintf(distinct, sizeof distinct, " (distinct %d/%d)",
                              stats.distinctVsPrev, cfg_.arenaGames);
            std::snprintf(line + len, sizeof line - static_cast<size_t>(len),
                          "arena elo %.1f vs-prev %s%s %.1fs", stats.elo, vsPrev, distinct,
                          stats.arenaSeconds);
        } else {
            std::snprintf(line + len, sizeof line - static_cast<size_t>(len),
                          "no arena, elo %.1f", stats.elo);
        }
        *log_ << line;
        if (cfg_.gating) *log_ << " | best " << state_.bestGeneration;
        *log_ << std::endl;
    }
    return stats;
}

void Trainer::runUntil(int lastGeneration) {
    while (generation() < lastGeneration) runGeneration();
}

} // namespace azgo
