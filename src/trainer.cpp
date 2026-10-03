#include "azgo/trainer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "azgo/arena.hpp"
#include "azgo/arena_round.hpp"
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
    "lr,replay_ratio,selfplay_s,train_s,arena_s,elo,winrate_vs_prev,distinct_vs_prev,"
    "ownership_loss\n";
// The header before ownership_loss was added; Trainer::resume upgrades it.
const char* const kLogHeaderBeforeOwnership =
    "generation,games,avg_length,pass_rate,black_win_rate,policy_loss,value_loss,"
    "lr,replay_ratio,selfplay_s,train_s,arena_s,elo,winrate_vs_prev,distinct_vs_prev\n";
const char* const kEloHeader = "generation,elo,winrate_vs_prev,wr_low,wr_high\n";
const char* const kRerateLogHeader =
    "generation,opponents,score_vs_prev,score_vs_strongest,promoted,elo\n";

using Clock = std::chrono::steady_clock;

double secondsSince(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string logPath(const std::string& runDir) { return (fs::path(runDir) / "log.csv").string(); }
std::string eloPath(const std::string& runDir) { return (fs::path(runDir) / "elo.csv").string(); }
std::string resultsPath(const std::string& runDir) { return (fs::path(runDir) / "results.csv").string(); }
std::string referencesPath(const std::string& runDir) {
    return (fs::path(runDir) / "references.csv").string();
}

std::string sgfPath(const std::string& runDir, int generation, int game) {
    char name[64];
    std::snprintf(name, sizeof name, "gen_%04d_game_%02d.sgf", generation, game);
    return (fs::path(runDir) / "sgf" / name).string();
}

// elo.csv: one row per rated generation, ascending; bots (negative IDs) are
// listed in references.csv instead. The win rate columns come
// from the (g-1, g) pair of the results table: how often g beat its
// predecessor, with a 95% Wilson interval. Empty when the pair was never played.
std::string eloCsv(const ResultsTable& results) {
    std::map<std::pair<int, int>, ResultsTable::Pair> byPair;
    for (const ResultsTable::Pair& p : results.pairs()) byPair[{p.a, p.b}] = p;

    std::string out = kEloHeader;
    for (const auto& [generation, rating] : fitBradleyTerry(results, 0)) {
        if (generation < 0) continue;
        const double elo = rating + 0.0;   // turns a -0.0 into 0.0, so the anchor prints "0.0"
        char line[160];
        // Generation 0 has no predecessor; (-1, 0) is bot 1's calibration pair.
        auto it = generation > 0 ? byPair.find({generation - 1, generation}) : byPair.end();
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

// The rating files, rewritten after every refit: results.csv, elo.csv and
// references.csv in `dir`.
void writeRatingFiles(const std::string& dir, const ResultsTable& results,
                      const std::vector<int>& references, const std::vector<int>& botSimulations) {
    writeFileAtomically(resultsPath(dir), results.toCsv());
    writeFileAtomically(eloPath(dir), eloCsv(results));
    writeFileAtomically(referencesPath(dir), referencesCsv(results, references, botSimulations));
}

ArenaConfig arenaConfig(const Config& cfg) {
    ArenaConfig ac;
    ac.games = cfg.arenaGames;
    ac.simulations = cfg.arenaSimulations;
    ac.mctsBatch = cfg.mctsBatch;
    ac.cPuct = cfg.cPuct;
    ac.temperatureMoves = cfg.arenaTemperatureMoves;
    ac.komi = cfg.komi;
    ac.maxMoves = cfg.effectiveMaxMoves();
    return ac;
}

ArenaRoundSettings roundSettings(const Config& cfg, std::optional<int> bestNetwork) {
    ArenaRoundSettings s;
    s.seed = cfg.seed;
    s.arenaGames = cfg.arenaGames;
    s.maxExtra = cfg.arenaOpponents;
    s.calibrationGames = cfg.calibrationGames;
    s.promoteThreshold = cfg.promoteThreshold;
    s.bestNetwork = bestNetwork;
    return s;
}

// Real arena matches between the players of the run in runDir: bots come from
// bot_simulations, generations from their checkpoints, except `liveNet`
// (generation `liveGeneration`), which is already in memory.
MatchFn runMatches(const Config& cfg, const std::string& runDir, int liveGeneration,
                   const Network* liveNet) {
    return [cfg, runDir, liveGeneration, liveNet](int a, int b, int games, uint64_t seed) {
        ArenaConfig ac = arenaConfig(cfg);
        ac.games = games;
        std::unique_ptr<Network> loaded[2];
        ArenaPlayer players[2];
        const int ids[2] = {a, b};
        for (int i = 0; i < 2; ++i) {
            if (ids[i] < 0) {
                players[i] = ArenaPlayer::bot(cfg.botSimulations.at(static_cast<size_t>(-ids[i] - 1)));
            } else if (liveNet && ids[i] == liveGeneration) {
                players[i] = ArenaPlayer::network(*liveNet);
            } else {
                loaded[i] = std::make_unique<Network>(loadNetwork(checkpointPath(runDir, ids[i])));
                players[i] = ArenaPlayer::network(*loaded[i]);
            }
        }
        return playMatch(players[0], players[1], ac, seed, cfg.effectiveWorkers(), cfg.backend);
    };
}

void appendLogRow(const std::string& runDir, const GenerationStats& s) {
    char winRate[32] = "";
    if (s.winRateVsPrev >= 0) std::snprintf(winRate, sizeof winRate, "%.4f", s.winRateVsPrev);
    char distinct[32] = "";
    if (s.distinctVsPrev >= 0) std::snprintf(distinct, sizeof distinct, "%d", s.distinctVsPrev);
    char ownership[32] = "";
    if (s.ownershipLoss >= 0) std::snprintf(ownership, sizeof ownership, "%.5f", s.ownershipLoss);
    char line[512];
    std::snprintf(line, sizeof line,
                  "%d,%d,%.2f,%.4f,%.4f,%.5f,%.5f,%g,%.3f,%.3f,%.3f,%.3f,%.1f,%s,%s,%s\n",
                  s.generation, s.games, s.avgLength, s.passRate, s.blackWinRate,
                  s.policyLoss, s.valueLoss, s.lr, s.replayRatio, s.selfPlaySeconds,
                  s.trainSeconds, s.arenaSeconds, s.elo, winRate, distinct, ownership);
    std::ofstream out(logPath(runDir), std::ios::app | std::ios::binary);
    out << line;
    if (!out) throw std::runtime_error("cannot append to " + logPath(runDir));
}

// "9x9, history 8, 6 blocks, 64 filters, value_hidden 64[, ownership head]":
// the keys a checkpoint's shape depends on, for messages about a mismatch.
std::string shapeString(const NetConfig& c) {
    return std::to_string(c.boardSize) + "x" + std::to_string(c.boardSize) + ", history " +
           std::to_string(c.history) + ", " + std::to_string(c.blocks) + " blocks, " +
           std::to_string(c.filters) + " filters, value_hidden " + std::to_string(c.valueHidden) +
           (c.ownership ? ", ownership head" : "");
}

// A run started before log.csv had the ownership_loss column gets it, empty
// in every old row (those generations had no ownership head), so every row
// has as many fields as the header. Anything else is left alone.
void addOwnershipColumn(const std::string& runDir) {
    std::ifstream in(logPath(runDir), std::ios::binary);
    if (!in) return;
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (text.compare(0, std::strlen(kLogHeaderBeforeOwnership), kLogHeaderBeforeOwnership) != 0) return;
    std::string out = kLogHeader;
    std::istringstream rows(text.substr(std::strlen(kLogHeaderBeforeOwnership)));
    for (std::string row; std::getline(rows, row);)
        if (!row.empty()) out += row + ",\n";
    writeFileAtomically(logPath(runDir), out);
}

std::string joinInts(const std::vector<int>& v) {
    std::string out;
    for (int x : v) out += (out.empty() ? "" : ",") + std::to_string(x);
    return out.empty() ? "(none)" : out;
}

} // namespace

Trainer::Trainer(const Config& cfg, const std::string& runDir, Network net, TrainingState state)
    : cfg_(cfg), runDir_(runDir), net_(std::move(net)), state_(std::move(state)),
      log_(&std::cout) {}

std::unique_ptr<Trainer> Trainer::startNew(const Config& cfg, const std::string& runDir,
                                           const RunInit& init) {
    try {
        LrSchedule::parse(cfg.lrSchedule);
    } catch (const std::invalid_argument& e) {
        throw ConfigError(std::string("lr_schedule: ") + e.what());
    }
    if (fs::exists(statePath(runDir)))
        throw ConfigError(runDir + " already contains a run; use --resume");

    // Everything that can refuse is loaded and checked before the first file
    // is written, so a refused start leaves nothing behind.
    std::optional<Network> given;
    if (!init.initFrom.empty()) {
        given.emplace(loadNetwork(init.initFrom));
        if (given->config() != cfg.netConfig()) {
            NetConfig sameHead = given->config();
            sameHead.ownership = cfg.ownership;
            if (sameHead == cfg.netConfig() && cfg.ownership)   // only the head is missing
                throw ConfigError(init.initFrom + " has no ownership head but the config sets "
                                  "ownership = true; add one with: grow --from " + init.initFrom +
                                  " --ownership --out NEW.azn");
            if (sameHead == cfg.netConfig())                    // only the head is extra
                throw ConfigError(init.initFrom + " has an ownership head but the config sets "
                                  "ownership = false; train it with --set ownership=true");
            throw ConfigError(init.initFrom + " has shape " + shapeString(given->config()) +
                              " but the config asks for " + shapeString(cfg.netConfig()));
        }
    }
    std::optional<ReplayBuffer> donorBuffer;
    if (!init.bufferFrom.empty()) {
        if (!fs::exists(statePath(init.bufferFrom)))
            throw ConfigError(init.bufferFrom + " does not contain a run (no state.bin)");
        const Config donor = loadConfigFile(configPath(init.bufferFrom));
        if (donor.boardSize != cfg.boardSize || donor.history != cfg.history)
            throw ConfigError("--buffer-from run " + init.bufferFrom + " has board_size " +
                              std::to_string(donor.boardSize) + ", history " +
                              std::to_string(donor.history) + " but the config has board_size " +
                              std::to_string(cfg.boardSize) + ", history " +
                              std::to_string(cfg.history));
        TrainingState donorState = loadTrainingState(statePath(init.bufferFrom), donor.botSimulations);
        donorBuffer.emplace(std::move(donorState.buffer));
        donorBuffer->setCapacity(static_cast<size_t>(cfg.bufferPositions));
    }

    fs::create_directories(fs::path(runDir) / "checkpoints");
    fs::create_directories(fs::path(runDir) / "sgf");
    writeFileAtomically(configPath(runDir), configToString(cfg));

    Network net = given ? std::move(*given) : Network(cfg.netConfig(), deriveSeed(cfg.seed, 0, kPhaseInit));
    saveNetwork(net, checkpointPath(runDir, 0));

    TrainingState state;
    state.buffer = donorBuffer ? std::move(*donorBuffer)
                               : ReplayBuffer(static_cast<size_t>(cfg.bufferPositions));
    state.velocity = SgdMomentum(net, cfg.momentum, cfg.weightDecay).velocity();   // all zeros
    state.botSimulations = cfg.botSimulations;
    state.references = initialReferences(static_cast<int>(cfg.botSimulations.size()));
    // Each bot plays generation 0 now, so it has a rating before it can be chosen.
    calibrateBots(state.results, state.references, roundSettings(cfg, std::nullopt),
                  runMatches(cfg, runDir, 0, &net));

    if (given || donorBuffer)
        writeFileAtomically(
            (fs::path(runDir) / "init.txt").string(),
            "init_from = " + (init.initFrom.empty() ? std::string("none") : fs::absolute(init.initFrom).string()) + "\n" +
                "buffer_from = " + (init.bufferFrom.empty() ? std::string("none") : fs::absolute(init.bufferFrom).string()) + "\n" +
                "buffer_positions = " + std::to_string(state.buffer.positions()) + "\n" +
                "buffer_games = " + std::to_string(state.buffer.games()) + "\n");

    writeFileAtomically(logPath(runDir), kLogHeader);
    writeRatingFiles(runDir, state.results, state.references, cfg.botSimulations);
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

    // A version-1 state.bin (written before bots existed) adopts the
    // configured bots and gets each as a reference.
    TrainingState state = loadTrainingState(statePath(runDir), cfg.botSimulations);
    // config.cfg can be edited by hand without --set, which resumeLockedDiff
    // cannot see; state.bin records the list the run actually rated.
    if (state.botSimulations != cfg.botSimulations)
        throw ConfigError("cannot change bot_simulations on resume: run has " +
                          joinInts(state.botSimulations) + " but config has " +
                          joinInts(cfg.botSimulations));
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
    addOwnershipColumn(runDir);

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
        double policySum = 0, valueSum = 0, ownershipSum = 0;
        for (int s = 0; s < cfg_.trainSteps; ++s) {
            state_.buffer.sample(cfg_.batchSize, cfg_.history, rng, batch);
            // The batch always carries ownership targets; only a network
            // with the head learns from them, with weight ownership_weight.
            const LossStats loss = net_.trainStep(
                ws, batch.input.data(), batch.pi.data(), batch.z.data(), batch.size,
                cfg_.ownership ? batch.ownership.data() : nullptr, cfg_.ownershipWeight);
            // Nothing of generation g is committed yet (state.bin is the
            // commit point), so the last good state (generation g-1) stays.
            if (!std::isfinite(loss.policyLoss) || !std::isfinite(loss.valueLoss) ||
                !std::isfinite(loss.ownershipLoss))
                throw std::runtime_error("non-finite loss at generation " + std::to_string(g) +
                                         " step " + std::to_string(s) +
                                         "; last good state kept");
            opt.step(net_, lr);
            policySum += loss.policyLoss;
            valueSum += loss.valueLoss;
            ownershipSum += loss.ownershipLoss;
        }
        if (cfg_.ownership) stats.ownershipLoss = 0;
        if (cfg_.trainSteps > 0) {
            stats.policyLoss = static_cast<float>(policySum / cfg_.trainSteps);
            stats.valueLoss = static_cast<float>(valueSum / cfg_.trainSteps);
            if (cfg_.ownership) stats.ownershipLoss = static_cast<float>(ownershipSum / cfg_.trainSteps);
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
    ArenaRoundResult round;
    if (arenaThisGeneration) {
        const std::optional<int> best =
            cfg_.gating ? std::optional<int>(state_.bestGeneration) : std::nullopt;
        round = runArenaRound(g, state_.results, state_.references, roundSettings(cfg_, best),
                              runMatches(cfg_, runDir_, g, &net_));
        for (const ArenaMatch& m : round.matches) {
            if (m.opponent == g - 1) {
                stats.winRateVsPrev = m.result.scoreA / m.result.games;
                stats.distinctVsPrev = m.result.distinctGames;
            }
            if (best && m.opponent == *best &&
                passesGate(m.result.scoreA, m.result.games, cfg_.gatingThreshold))
                newBest = g;
        }
        // Refit every rating from all results so far: Bradley-Terry does not
        // depend on the order results arrived in, so a resumed run agrees.
        writeRatingFiles(runDir_, state_.results, state_.references, cfg_.botSimulations);
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
        // e.g. "train loss p 1.73 v 0.09 lr 0.002 2.5s", with "o 0.412" after
        // the value loss when the network has an ownership head.
        char ownership[32] = "";
        if (stats.ownershipLoss >= 0)
            std::snprintf(ownership, sizeof ownership, " o %.3f", stats.ownershipLoss);
        char line[512];
        std::snprintf(line, sizeof line,
                      "gen %d | selfplay %d games avg %.1f plies %.1fs | "
                      "train loss p %.2f v %.2f%s lr %g %.1fs | ",
                      g, stats.games, stats.avgLength, stats.selfPlaySeconds, stats.policyLoss,
                      stats.valueLoss, ownership, stats.lr, stats.trainSeconds);
        *log_ << line;
        if (arenaThisGeneration) {
            // e.g. "arena elo 512.3 vs 149:0.55 120:0.70 bot100:1.00 (distinct 19/20) promoted 3.2s"
            char elo[48], tail[64];
            std::snprintf(elo, sizeof elo, "arena elo %.1f ", stats.elo);
            std::snprintf(tail, sizeof tail, " (distinct %d/%d)%s %.1fs", stats.distinctVsPrev,
                          cfg_.arenaGames, round.promoted ? " promoted" : "", stats.arenaSeconds);
            *log_ << elo << opponentSummary(round, cfg_.botSimulations) << tail;
        } else {
            char noArena[48];
            std::snprintf(noArena, sizeof noArena, "no arena, elo %.1f", stats.elo);
            *log_ << noArena;
        }
        if (cfg_.gating) *log_ << " | best " << state_.bestGeneration;
        *log_ << std::endl;
    }
    return stats;
}

void Trainer::runUntil(int lastGeneration) {
    while (generation() < lastGeneration) runGeneration();
}

int rerateRun(const std::string& runDir, const std::vector<std::string>& overrides,
              std::ostream* log) {
    if (!fs::exists(configPath(runDir)))
        throw ConfigError(runDir + " does not contain a run (no config.cfg)");
    const Config saved = loadConfigFile(configPath(runDir));
    Config cfg = saved;
    for (const std::string& o : overrides) applyOverride(cfg, o);
    const std::vector<std::string> diff = architectureDiff(saved, cfg);
    if (!diff.empty()) {
        std::string keys;
        for (const std::string& k : diff) keys += (keys.empty() ? "" : ", ") + k;
        throw ConfigError("cannot change architecture keys when re-rating: " + keys);
    }
    if (!fs::exists(checkpointPath(runDir, 0)))
        throw std::runtime_error(checkpointPath(runDir, 0) + ": missing");

    // Every generation is measured against g-1, so the replay ends at the
    // first missing checkpoint.
    int last = 0;
    while (fs::exists(checkpointPath(runDir, last + 1))) ++last;

    // Fail on an unavailable backend before anything is written.
    makeBackend(cfg.backend, 1);

    const std::string outDir = (fs::path(runDir) / "rerate").string();
    fs::create_directories(outDir);
    ResultsTable results;
    std::vector<int> references = initialReferences(static_cast<int>(cfg.botSimulations.size()));
    const ArenaRoundSettings settings = roundSettings(cfg, std::nullopt);   // gating ignored
    const MatchFn play = runMatches(cfg, runDir, -1, nullptr);   // every network from disk
    calibrateBots(results, references, settings, play);
    writeRatingFiles(outDir, results, references, cfg.botSimulations);
    std::string logCsv = kRerateLogHeader;
    writeFileAtomically(logPath(outDir), logCsv);

    for (int g = 1; g <= last; ++g) {
        if (cfg.arenaEvery == 0 || g % cfg.arenaEvery != 0) continue;
        const auto start = Clock::now();
        const ArenaRoundResult round = runArenaRound(g, results, references, settings, play);
        writeRatingFiles(outDir, results, references, cfg.botSimulations);

        // generation,opponents,score_vs_prev,score_vs_strongest,promoted,elo
        // opponents are player IDs separated by spaces; matches[0] is g-1.
        std::string ids;
        double vsStrongest = 0;
        for (const ArenaMatch& m : round.matches) {
            ids += (ids.empty() ? "" : " ") + std::to_string(m.opponent);
            if (m.opponent == round.strongest) vsStrongest = m.result.scoreA / m.result.games;
        }
        const MatchResult& prev = round.matches.front().result;
        const double elo = fitBradleyTerry(results, 0).at(g) + 0.0;
        char row[128];
        std::snprintf(row, sizeof row, ",%.4f,%.4f,%d,%.1f\n", prev.scoreA / prev.games, vsStrongest,
                      round.promoted ? 1 : 0, elo);
        logCsv += std::to_string(g) + "," + ids + row;
        writeFileAtomically(logPath(outDir), logCsv);

        if (log) {
            char head[64], tail[48];
            std::snprintf(head, sizeof head, "rerate gen %d | elo %.1f ", g, elo);
            std::snprintf(tail, sizeof tail, "%s %.1fs", round.promoted ? " promoted" : "",
                          secondsSince(start));
            *log << head << opponentSummary(round, cfg.botSimulations) << tail << std::endl;
        }
    }
    return last;
}

} // namespace azgo
