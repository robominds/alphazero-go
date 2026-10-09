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
#include "azgo/grow.hpp"
#include "azgo/inference_hub.hpp"
#include "azgo/metal.hpp"
#include "azgo/optimizer.hpp"
#include "azgo/parallel.hpp"
#include "azgo/selfplay.hpp"
#include "azgo/serialize.hpp"
#include "azgo/sgf.hpp"
#include "azgo/validation.hpp"

namespace azgo {

namespace fs = std::filesystem;

namespace {

// With backend = metal: which GPU, OS and shaders a run uses, so two runs'
// logs show whether they ran the same arithmetic.
void printDeviceLine(const Config& cfg, std::ostream* log) {
    if (log && cfg.backend == "metal")
        *log << "gpu: " << metalDescription() << " (inference: self-play, arena, validation)\n";
}

const char* const kLogHeader =
    "generation,games,avg_length,pass_rate,black_win_rate,policy_loss,value_loss,"
    "lr,replay_ratio,selfplay_s,train_s,arena_s,elo,winrate_vs_prev,distinct_vs_prev,"
    "ownership_loss,resign_threshold,resign_rate,resign_false_positive,"
    "val_policy_loss,val_value_loss,val_ownership_loss,"
    "seen_policy_loss,seen_value_loss,seen_ownership_loss\n";
// Earlier headers, which Trainer::resume upgrades: before ownership_loss,
// before the resignation columns, and before the validation columns.
const char* const kLogHeaderBeforeOwnership =
    "generation,games,avg_length,pass_rate,black_win_rate,policy_loss,value_loss,"
    "lr,replay_ratio,selfplay_s,train_s,arena_s,elo,winrate_vs_prev,distinct_vs_prev\n";
const char* const kLogHeaderBeforeResignation =
    "generation,games,avg_length,pass_rate,black_win_rate,policy_loss,value_loss,"
    "lr,replay_ratio,selfplay_s,train_s,arena_s,elo,winrate_vs_prev,distinct_vs_prev,"
    "ownership_loss\n";
const char* const kLogHeaderBeforeValidation =
    "generation,games,avg_length,pass_rate,black_win_rate,policy_loss,value_loss,"
    "lr,replay_ratio,selfplay_s,train_s,arena_s,elo,winrate_vs_prev,distinct_vs_prev,"
    "ownership_loss,resign_threshold,resign_rate,resign_false_positive\n";
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
std::string referenceNetworksPath(const std::string& runDir) {
    return (fs::path(runDir) / "reference-networks.csv").string();
}

std::string sgfPath(const std::string& runDir, int generation, int game) {
    char name[64];
    std::snprintf(name, sizeof name, "gen_%04d_game_%02d.sgf", generation, game);
    return (fs::path(runDir) / "sgf" / name).string();
}

// elo.csv: one row per rated generation, ascending; fixed players (negative
// IDs: bots and reference networks) are listed in references.csv instead.
// The win rate columns come from the (g-1, g) pair of the results table: how
// often g beat its predecessor, with a 95% Wilson interval. Empty when the
// pair was never played.
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
// references.csv in `dir`, and reference-networks.csv when the run has
// reference networks.
void writeRatingFiles(const std::string& dir, const ResultsTable& results,
                      const std::vector<int>& references, const Config& cfg) {
    const int networks = static_cast<int>(cfg.referenceNetworks.size());
    writeFileAtomically(resultsPath(dir), results.toCsv());
    writeFileAtomically(eloPath(dir), eloCsv(results));
    writeFileAtomically(referencesPath(dir),
                        referencesCsv(results, references, cfg.botSimulations, networks));
    if (networks > 0)
        writeFileAtomically(referenceNetworksPath(dir),
                            referenceNetworksCsv(cfg.referenceNetworks,
                                                 static_cast<int>(cfg.botSimulations.size())));
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
    ac.randomSymmetry = cfg.arenaRandomSymmetry;
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

// Real arena matches between the players of the run in runDir: fixed player
// k (player -k) is bot k from bot_simulations, or past the bots one of
// `referenceNets` (the run's reference networks, loaded once and fitted to
// its shape); generations come from their checkpoints, except `liveNet`
// (generation `liveGeneration`), which is already in memory.
MatchFn runMatches(const Config& cfg, const std::string& runDir, int liveGeneration,
                   const Network* liveNet, const std::vector<Network>* referenceNets) {
    return [cfg, runDir, liveGeneration, liveNet, referenceNets](int a, int b, int games, uint64_t seed) {
        ArenaConfig ac = arenaConfig(cfg);
        ac.games = games;
        std::unique_ptr<Network> loaded[2];
        ArenaPlayer players[2];
        const int ids[2] = {a, b};
        const size_t bots = cfg.botSimulations.size();
        for (int i = 0; i < 2; ++i) {
            if (ids[i] < 0) {
                const size_t k = static_cast<size_t>(-ids[i]);
                players[i] = k <= bots ? ArenaPlayer::bot(cfg.botSimulations[k - 1])
                                       : ArenaPlayer::network(referenceNets->at(k - bots - 1));
            } else if (liveNet && ids[i] == liveGeneration) {
                players[i] = ArenaPlayer::network(*liveNet);
            } else {
                loaded[i] = std::make_unique<Network>(loadNetwork(checkpointPath(runDir, ids[i])));
                players[i] = ArenaPlayer::network(*loaded[i]);
            }
        }
        return playMatch(players[0], players[1], ac, seed, cfg.effectiveWorkers(), cfg.backend,
                         cfg.inferenceBatch);
    };
}

// A validation loss as log.csv or the progress line prints it: "" when not
// measured (the -1 of GenerationStats), "nan" when the measurement is not a
// number, so a broken network shows up instead of looking unmeasured.
std::string formatLoss(float v, int decimals) {
    if (v == -1.0f) return "";
    if (std::isnan(v)) return "nan";
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.*f", decimals, static_cast<double>(v));
    return buf;
}

void appendLogRow(const std::string& runDir, const GenerationStats& s) {
    char winRate[32] = "";
    if (s.winRateVsPrev >= 0) std::snprintf(winRate, sizeof winRate, "%.4f", s.winRateVsPrev);
    char distinct[32] = "";
    if (s.distinctVsPrev >= 0) std::snprintf(distinct, sizeof distinct, "%d", s.distinctVsPrev);
    char ownership[32] = "";
    if (s.ownershipLoss >= 0) std::snprintf(ownership, sizeof ownership, "%.5f", s.ownershipLoss);
    // resign_threshold,resign_rate,resign_false_positive: all empty without
    // resignation; a rate or rate of false positives that does not exist
    // yet (no enabled games, an empty window) is empty too.
    char threshold[32] = "", rate[32] = "", falsePositive[32] = "";
    if (s.resignation) {
        std::snprintf(threshold, sizeof threshold, "%.2f", s.resignThreshold);
        if (s.resignRate >= 0) std::snprintf(rate, sizeof rate, "%.4f", s.resignRate);
        if (s.resignFalsePositive >= 0)
            std::snprintf(falsePositive, sizeof falsePositive, "%.4f", s.resignFalsePositive);
    }
    // The validation columns: empty where not measured.
    const float validation[6] = {s.valPolicyLoss,  s.valValueLoss,  s.valOwnershipLoss,
                                 s.seenPolicyLoss, s.seenValueLoss, s.seenOwnershipLoss};
    std::string validationFields;
    for (float v : validation) validationFields += "," + formatLoss(v, 5);
    char line[512];
    std::snprintf(line, sizeof line,
                  "%d,%d,%.2f,%.4f,%.4f,%.5f,%.5f,%g,%.3f,%.3f,%.3f,%.3f,%.1f,%s,%s,%s,%s,%s,%s%s\n",
                  s.generation, s.games, s.avgLength, s.passRate, s.blackWinRate,
                  s.policyLoss, s.valueLoss, s.lr, s.replayRatio, s.selfPlaySeconds,
                  s.trainSeconds, s.arenaSeconds, s.elo, winRate, distinct, ownership,
                  threshold, rate, falsePositive, validationFields.c_str());
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

// A run started before log.csv had the ownership_loss, the resignation or
// the validation columns gets them, empty in every old row (those
// generations had no ownership head, no resignation and no validation
// loss), so every row has as many fields as the header. Anything else is
// left alone.
void addMissingLogColumns(const std::string& runDir) {
    std::ifstream in(logPath(runDir), std::ios::binary);
    if (!in) return;
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::pair<const char*, const char*> upgrades[] = {{kLogHeaderBeforeOwnership, ",,,,,,,,,,"},
                                                            {kLogHeaderBeforeResignation, ",,,,,,,,,"},
                                                            {kLogHeaderBeforeValidation, ",,,,,,"}};
    for (const auto& [header, emptyFields] : upgrades) {
        if (text.compare(0, std::strlen(header), header) != 0) continue;
        std::string out = kLogHeader;
        std::istringstream rows(text.substr(std::strlen(header)));
        for (std::string row; std::getline(rows, row);)
            if (!row.empty()) out += row + emptyFields + "\n";
        writeFileAtomically(logPath(runDir), out);
        return;
    }
}

ResignCalibration resignCalibration(const Config& cfg) {
    ResignCalibration c;
    c.window = cfg.resignWindow;
    c.minGames = cfg.resignMinGames;
    c.maxFalsePositive = cfg.resignFalsePositive;
    c.initialThreshold = cfg.resignThresholdInitial;
    return c;
}

// Keeps only the most recent `window` disabled games.
void trimResignWindow(std::vector<ResignTrace>& games, int window) {
    if (games.size() > static_cast<size_t>(window))
        games.erase(games.begin(), games.end() - window);
}

std::string joinInts(const std::vector<int>& v) {
    std::string out;
    for (int x : v) out += (out.empty() ? "" : ",") + std::to_string(x);
    return out.empty() ? "(none)" : out;
}

std::string joinStrings(const std::vector<std::string>& v) {
    std::string out;
    for (const std::string& x : v) out += (out.empty() ? "" : ",") + x;
    return out.empty() ? "(none)" : out;
}

// The run's reference networks, each read once: `networks` fitted to the
// run's shape (fitToShape, seeded by deriveSeed(seed, 0, kPhaseReference, j)
// for reference network j, 0-based), `recorded` the path and fileHash that
// state.bin keeps. Fitting keeps its policy and value (widening up to float
// rounding), so a reference plays as it did in its own run. A missing or unreadable file, one that cannot be fitted, or a
// file with the same bytes as an earlier one is a ConfigError naming it.
struct ReferenceNetworks {
    std::vector<Network> networks;
    std::vector<ReferenceNetwork> recorded;
};

ReferenceNetworks loadReferenceNetworks(const Config& cfg) {
    ReferenceNetworks out;
    for (size_t j = 0; j < cfg.referenceNetworks.size(); ++j) {
        const std::string& path = cfg.referenceNetworks[j];
        std::optional<Network> net;
        uint64_t hash = 0;
        try {
            hash = fileHash(path);
            net.emplace(loadNetwork(path));   // its messages name the file
        } catch (const std::runtime_error& e) {
            throw ConfigError(std::string("reference_networks: ") + e.what());
        }
        // Two names for one network would be two players with one strength.
        for (const ReferenceNetwork& earlier : out.recorded)
            if (earlier.hash == hash)
                throw ConfigError("reference_networks: " + earlier.path + " and " + path +
                                  " are the same network (identical bytes)");
        try {
            out.networks.push_back(fitToShape(*net, cfg.netConfig(),
                                              deriveSeed(cfg.seed, 0, kPhaseReference, j)));
        } catch (const std::invalid_argument& e) {
            throw ConfigError("reference_networks: " + path + " has shape " + shapeString(net->config()) +
                              ", which cannot play in a run of " + shapeString(cfg.netConfig()) +
                              ": " + e.what());
        }
        out.recorded.push_back({path, hash});
    }
    return out;
}

// state.bin records each reference network's hash; a file that no longer
// matches would put another player under an old ID.
void checkReferenceHashes(const ReferenceNetworks& loaded, const std::vector<ReferenceNetwork>& recorded) {
    for (size_t j = 0; j < recorded.size(); ++j)
        if (loaded.recorded[j].hash != recorded[j].hash)
            throw ConfigError("reference_networks: " + recorded[j].path +
                              " has changed since the run started (its bytes no longer match the "
                              "hash in state.bin); restore the original file");
}

} // namespace

Trainer::Trainer(const Config& cfg, const std::string& runDir, Network net, TrainingState state)
    : cfg_(cfg), runDir_(runDir), net_(std::move(net)), state_(std::move(state)),
      log_(&std::cout) {}

std::unique_ptr<Trainer> Trainer::startNew(const Config& cfg, const std::string& runDir,
                                           const RunInit& init, std::ostream* log) {
    checkConsistency(cfg);
    checkBackendAvailable(cfg.backend);
    printDeviceLine(cfg, log);
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
    ReferenceNetworks referenceNets = loadReferenceNetworks(cfg);
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
        // The ownership target is the final board's owners, and a resigned
        // game's last board is not one: an ownership run must not train on it.
        if (cfg.ownership) {
            size_t resigned = 0;
            for (size_t i = 0; i < donorBuffer->games(); ++i) resigned += donorBuffer->game(i).resigned ? 1 : 0;
            if (resigned > 0)
                throw ConfigError("--buffer-from run " + init.bufferFrom + " holds " + std::to_string(resigned) +
                                  " games that ended by resignation; their last boards are not final "
                                  "positions, so a run with ownership = true cannot train on them");
        }
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
    state.referenceNetworks = referenceNets.recorded;
    state.resignThreshold = calibrateResignThreshold(state.resignWindow, resignCalibration(cfg));   // the initial one
    state.references = initialReferences(
        static_cast<int>(cfg.botSimulations.size() + cfg.referenceNetworks.size()));
    // Each fixed player plays generation 0 now, so it has a rating before it can be chosen.
    calibrateFixedPlayers(state.results, state.references, roundSettings(cfg, std::nullopt),
                          runMatches(cfg, runDir, 0, &net, &referenceNets.networks));

    if (given || donorBuffer)
        writeFileAtomically(
            (fs::path(runDir) / "init.txt").string(),
            "init_from = " + (init.initFrom.empty() ? std::string("none") : fs::absolute(init.initFrom).string()) + "\n" +
                "buffer_from = " + (init.bufferFrom.empty() ? std::string("none") : fs::absolute(init.bufferFrom).string()) + "\n" +
                "buffer_positions = " + std::to_string(state.buffer.positions()) + "\n" +
                "buffer_games = " + std::to_string(state.buffer.games()) + "\n");

    writeFileAtomically(logPath(runDir), kLogHeader);
    writeRatingFiles(runDir, state.results, state.references, cfg);
    // state.bin last: until it exists the directory is not a run, so a crash
    // anywhere above leaves something startNew can simply redo.
    saveTrainingState(statePath(runDir), state);

    std::unique_ptr<Trainer> trainer(new Trainer(cfg, runDir, std::move(net), std::move(state)));
    trainer->referenceNets_ = std::move(referenceNets.networks);
    return trainer;
}

std::unique_ptr<Trainer> Trainer::resume(const std::string& runDir,
                                         const std::vector<std::string>& overrides, std::ostream* log) {
    if (!fs::exists(statePath(runDir)))
        throw ConfigError(runDir + " does not contain a run (no state.bin); start one with --config");

    const Config saved = loadConfigFile(configPath(runDir));
    Config cfg = saved;
    for (const std::string& o : overrides) applyOverride(cfg, o);

    const std::vector<std::string> diff = resumeLockedDiff(saved, cfg);
    if (!diff.empty()) {
        std::string keys;
        for (const std::string& k : diff) keys += (keys.empty() ? "" : ", ") + k;
        // resumeLockedDiff lists the fixed players' keys last, after any architecture keys.
        if (diff.back() == "bot_simulations" || diff.back() == "reference_networks")
            throw ConfigError("cannot change keys on resume: " + keys +
                              " (bot k is arena player -k and the reference networks come after "
                              "the bots, so both lists are fixed for a run)");
        throw ConfigError("cannot change architecture keys on resume: " + keys);
    }
    checkConsistency(cfg);
    checkBackendAvailable(cfg.backend);

    // A version-1 state.bin (written before bots existed) adopts the
    // configured bots and gets each as a reference.
    TrainingState state = loadTrainingState(statePath(runDir), cfg.botSimulations);
    // config.cfg can be edited by hand without --set, which resumeLockedDiff
    // cannot see; state.bin records the list the run actually rated.
    if (state.botSimulations != cfg.botSimulations)
        throw ConfigError("cannot change bot_simulations on resume: run has " +
                          joinInts(state.botSimulations) + " but config has " +
                          joinInts(cfg.botSimulations));
    std::vector<std::string> recordedPaths;
    for (const ReferenceNetwork& n : state.referenceNetworks) recordedPaths.push_back(n.path);
    if (recordedPaths != cfg.referenceNetworks)
        throw ConfigError("cannot change reference_networks on resume: run has " +
                          joinStrings(recordedPaths) + " but config has " +
                          joinStrings(cfg.referenceNetworks));
    ReferenceNetworks referenceNets = loadReferenceNetworks(cfg);
    checkReferenceHashes(referenceNets, state.referenceNetworks);
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
    // The resignation keys are tunable, so v_resign is recalibrated from the
    // stored window with today's values (unchanged keys give the stored
    // threshold back; calibration reads only the last resign_window games);
    // a state.bin from before resignation has no window and gets
    // resign_threshold_initial.
    state.resignThreshold = calibrateResignThreshold(state.resignWindow, resignCalibration(cfg));
    printDeviceLine(cfg, log);
    if (log && saved.backend != cfg.backend)
        *log << "note: backend changed from " << saved.backend << " to " << cfg.backend
             << "; from here on results differ in the\nlast bits from a run that kept " << saved.backend
             << "\n";
    // Rewritten only after every check passed, so a refused resume leaves the
    // run directory untouched.
    writeFileAtomically(configPath(runDir), configToString(cfg));
    addMissingLogColumns(runDir);

    std::unique_ptr<Trainer> trainer(new Trainer(cfg, runDir, std::move(net), std::move(state)));
    trainer->referenceNets_ = std::move(referenceNets.networks);
    return trainer;
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
        sp.mcts.randomSymmetry = cfg_.mctsRandomSymmetry;
        sp.temperatureMoves = cfg_.temperatureMoves;
        sp.komi = cfg_.komi;
        sp.maxMoves = cfg_.effectiveMaxMoves();
        // v_resign was calibrated after the previous generation, from earlier
        // generations' games only, so every game of this one shares it
        // whichever worker plays it.
        sp.resignation = cfg_.resignation;
        sp.resignThreshold = state_.resignThreshold;
        sp.resignDisabledFraction = cfg_.resignDisabledFraction;

        // One evaluator per worker; the network itself is shared read-only.
        // naive, blas: a single-threaded backend and workspace per worker.
        // metal: one batching hub that gathers every worker's leaves into
        // GPU passes; a position's outputs do not depend on its batch, so
        // the games are the same either way (azgo/inference_hub.hpp).
        const int selfPlayWorkers = std::min(workers, numGames);
        WorkerEvaluators evaluators(selfPlayNet, cfg_.backend, selfPlayWorkers, cfg_.mctsBatch,
                                    cfg_.inferenceBatch);
        // Game i's randomness depends only on i, and its record is stored at
        // index i, so the result does not depend on which thread played it.
        parallelFor(numGames, selfPlayWorkers, [&](int i, int w) {
            Rng rng(deriveSeed(cfg_.seed, static_cast<uint64_t>(g), kPhaseSelfPlay,
                               static_cast<uint64_t>(i)));
            records[static_cast<size_t>(i)] = playSelfPlayGame(evaluators.forWorker(w), sp, rng);
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

    // Resignation: how often it ended a game, and the disabled games'
    // traces, in game order, for the next calibration (AlphaGo Zero keeps
    // false positives, games the winner would have resigned, below 5%).
    if (cfg_.resignation) {
        int enabled = 0, resigned = 0;
        for (const GameRecord& r : records) {
            if (r.resignDisabled) {
                state_.resignWindow.push_back(resignTrace(r));
            } else {
                ++enabled;
                if (r.resigned) ++resigned;
            }
        }
        trimResignWindow(state_.resignWindow, cfg_.resignWindow);
        stats.resignation = true;
        stats.resignThreshold = state_.resignThreshold;
        if (enabled > 0) stats.resignRate = static_cast<double>(resigned) / enabled;
        if (!state_.resignWindow.empty())
            stats.resignFalsePositive =
                resignFalsePositiveRate(state_.resignWindow, cfg_.resignWindow, state_.resignThreshold);
    }
    // The threshold generation g + 1 will use.
    state_.resignThreshold = calibrateResignThreshold(state_.resignWindow, resignCalibration(cfg_));

    // A few games per generation as SGF. The record stores only the result's
    // sign, so the Tromp-Taylor margin for RE[] comes from replaying the
    // moves; a resigned game has none ("B+R" or "W+R").
    const int sgfGames = std::min(cfg_.sgfPerGeneration, numGames);
    for (int i = 0; i < sgfGames; ++i) {
        const GameRecord& r = records[static_cast<size_t>(i)];
        const std::string comment = "azgo self-play, generation " + std::to_string(g) +
                                    ", game " + std::to_string(i) +
                                    (r.resignDisabled ? ", resignation disabled" : "");
        std::string sgf;
        if (r.resigned) {
            sgf = gameToSgf(n, r.komi, r.moves, r.resultForBlack > 0 ? "B+R" : "W+R", comment);
        } else {
            Game replay(n, r.komi, cfg_.effectiveMaxMoves());
            for (int a : r.moves) replay.play(a);
            sgf = gameToSgf(n, r.komi, r.moves, replay.scoreForBlack(), comment);
        }
        writeFileAtomically(sgfPath(runDir_, g, i), sgf);
    }

    // ---- Validation loss (a measurement; changes nothing) -------------------
    // net_ is generation g-1's network, about to train on these games, and
    // (without gating) the one that played them. Measured before they join
    // the buffer, they are data it has never trained on. The buffer, not yet
    // changed, is its training distribution: the window it trained on at
    // generation g-1 (a sample of it, at random symmetries, so some of
    // these positions it drew rarely or never). At generation 1 the buffer
    // is empty, or with --buffer-from a donor's games that generation 0 (a
    // fresh network, or --init-from's) never trained on. Inference mode and the identity symmetry, so nothing is drawn from any
    // generator of the run and nothing is written to the network; the
    // buffer sample has its own generator (kPhaseValidation). Its time counts
    // as training time.
    //
    // The network passes from here to the end of training run on a pool of
    // train_threads threads (layers.hpp): batches of 256 are wide enough to
    // share among cores, and every value is computed in a fixed chunk, so
    // the result does not depend on the thread count.
    // The pool lives from here to the end of training (reset below), so its
    // threads exist only while the training phase runs.
    std::unique_ptr<ThreadPool> trainPoolOwner = std::make_unique<ThreadPool>(cfg_.effectiveTrainThreads());
    ThreadPool& trainPool = *trainPoolOwner;
    double validationSeconds = 0;
    if (cfg_.validationLoss) {
        const auto validationStart = Clock::now();
        // The loss of one batch: on the CPU pool, or with metal the forward
        // pass on the GPU and the loss terms on the CPU (lossFromOutputs).
        std::unique_ptr<Backend> backend;
        std::unique_ptr<Workspace> ws;
#ifdef AZ_USE_METAL
        std::unique_ptr<MetalInference> gpu;
#endif
        BatchLossFn lossOf;
        if (cfg_.backend == "metal") {
#ifdef AZ_USE_METAL
            gpu = std::make_unique<MetalInference>(net_, cfg_.batchSize);
            lossOf = [&](const float* input, const float* pi, const float* z, int B, const float* own) {
                return gpu->inferLoss(input, pi, z, B, own);
            };
#else
            throw std::logic_error("backend 'metal' in a build without Metal");   // refused at start
#endif
        } else {
            backend = makeBackend(cfg_.backend, 1);
            ws = std::make_unique<Workspace>(net_.config(), cfg_.batchSize, *backend, &trainPool);
            lossOf = [&](const float* input, const float* pi, const float* z, int B, const float* own) {
                return net_.inferLoss(*ws, input, pi, z, B, own);
            };
        }
        auto measure = [&](const std::vector<PositionRef>& positions) {
            return heldOutLoss(net_.config(), cfg_.batchSize, &trainPool, lossOf, positions, cfg_.history);
        };
        const HeldOutLoss fresh = measure(allPositions(records));
        stats.valPolicyLoss = fresh.policyLoss;
        stats.valValueLoss = fresh.valueLoss;
        stats.valOwnershipLoss = fresh.ownershipLoss;
        if (state_.buffer.positions() > 0) {
            // As many positions as the fresh games have, so both means are
            // equally noisy.
            Rng rng(deriveSeed(cfg_.seed, static_cast<uint64_t>(g), kPhaseValidation));
            const HeldOutLoss seen =
                measure(samplePositions(state_.buffer, static_cast<int>(fresh.positions), rng));
            stats.seenPolicyLoss = seen.policyLoss;
            stats.seenValueLoss = seen.valueLoss;
            stats.seenOwnershipLoss = seen.ownershipLoss;
        }
        validationSeconds = secondsSince(validationStart);
    }

    for (GameRecord& r : records) state_.buffer.add(std::move(r));   // index order
    stats.selfPlaySeconds = secondsSince(selfPlayStart) - validationSeconds;

    // ---- 2. Train -----------------------------------------------------------
    const auto trainStart = Clock::now();
    const float lr = LrSchedule::parse(cfg_.lrSchedule).at(g);
    {
        // A single-threaded backend: the pool splits the gemms (gemmColumns).
        // Training stays on the CPU with metal too (BLAS).
        std::unique_ptr<Backend> backend = makeBackend(cpuBackendName(cfg_.backend), 1);
        Workspace ws(net_.config(), cfg_.batchSize, *backend, &trainPool);
        SgdMomentum opt(net_, cfg_.momentum, cfg_.weightDecay);
        opt.velocity() = state_.velocity;
        Rng rng(deriveSeed(cfg_.seed, static_cast<uint64_t>(g), kPhaseTrain));
        TrainingBatch batch;
        double policySum = 0, valueSum = 0, ownershipSum = 0;
        for (int s = 0; s < cfg_.trainSteps; ++s) {
            state_.buffer.sample(cfg_.batchSize, cfg_.history, rng, batch, &trainPool);
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
            opt.step(net_, lr, &trainPool);
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
    trainPoolOwner.reset();   // training done: stop and join the pool's threads
    const long long drawn = static_cast<long long>(cfg_.trainSteps) * cfg_.batchSize;
    state_.samplesTrained += drawn;
    stats.lr = lr;
    stats.replayRatio = plies > 0 ? static_cast<double>(drawn) / plies : 0.0;
    stats.trainSeconds = secondsSince(trainStart) + validationSeconds;

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
                              runMatches(cfg_, runDir_, g, &net_, &referenceNets_));
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
        writeRatingFiles(runDir_, state_.results, state_.references, cfg_);
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
        // With resignation, e.g. "avg 41.2 plies resign -0.87 rate 0.62 fp 0.031 7.1s":
        // the threshold used, the share of enabled games resigned, the
        // false-positive rate ("-" where there is none yet).
        char resign[96] = "";
        if (stats.resignation) {
            char rate[16] = "-", falsePositive[16] = "-";
            if (stats.resignRate >= 0) std::snprintf(rate, sizeof rate, "%.2f", stats.resignRate);
            if (stats.resignFalsePositive >= 0)
                std::snprintf(falsePositive, sizeof falsePositive, "%.3f", stats.resignFalsePositive);
            std::snprintf(resign, sizeof resign, " resign %.2f rate %s fp %s", stats.resignThreshold, rate,
                          falsePositive);
        }
        // With the validation loss, e.g. " val p 1.80 v 0.12 seen p 1.75 v 0.10":
        // on the fresh games, then on the buffer sample (once there is one).
        std::string validation;
        if (stats.valPolicyLoss != -1.0f) {
            validation = " val p " + formatLoss(stats.valPolicyLoss, 2) + " v " + formatLoss(stats.valValueLoss, 2);
            if (stats.seenPolicyLoss != -1.0f)
                validation += " seen p " + formatLoss(stats.seenPolicyLoss, 2) + " v " +
                              formatLoss(stats.seenValueLoss, 2);
        }
        char line[512];
        std::snprintf(line, sizeof line,
                      "gen %d | selfplay %d games avg %.1f plies%s %.1fs | "
                      "train loss p %.2f v %.2f%s lr %g%s %.1fs | ",
                      g, stats.games, stats.avgLength, resign, stats.selfPlaySeconds, stats.policyLoss,
                      stats.valueLoss, ownership, stats.lr, validation.c_str(), stats.trainSeconds);
        *log_ << line;
        if (arenaThisGeneration) {
            // e.g. "arena elo 512.3 vs 149:0.55 120:0.70 ref1:0.40 (distinct 19/20) promoted 3.2s"
            char elo[48], tail[64];
            std::snprintf(elo, sizeof elo, "arena elo %.1f ", stats.elo);
            std::snprintf(tail, sizeof tail, " (distinct %d/%d)%s %.1fs", stats.distinctVsPrev,
                          cfg_.arenaGames, round.promoted ? " promoted" : "", stats.arenaSeconds);
            *log_ << elo
                  << opponentSummary(round, cfg_.botSimulations,
                                     static_cast<int>(cfg_.referenceNetworks.size()))
                  << tail;
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
    checkBackendAvailable(cfg.backend);
    printDeviceLine(cfg, log);

    // Each reference network is read once. When the replay uses the run's
    // own list (config.cfg's), its files must still be the ones the live run
    // rated, which state.bin records; any other replay does not read state.bin.
    const ReferenceNetworks referenceNets = loadReferenceNetworks(cfg);
    if (!cfg.referenceNetworks.empty() && cfg.referenceNetworks == saved.referenceNetworks &&
        fs::exists(statePath(runDir))) {
        const TrainingState live = loadTrainingState(statePath(runDir), saved.botSimulations);
        std::vector<std::string> livePaths;
        for (const ReferenceNetwork& n : live.referenceNetworks) livePaths.push_back(n.path);
        if (livePaths == cfg.referenceNetworks) checkReferenceHashes(referenceNets, live.referenceNetworks);
    }

    const std::string outDir = (fs::path(runDir) / "rerate").string();
    fs::create_directories(outDir);
    ResultsTable results;
    const int numBots = static_cast<int>(cfg.botSimulations.size());
    const int numNetworks = static_cast<int>(cfg.referenceNetworks.size());
    std::vector<int> references = initialReferences(numBots + numNetworks);
    const ArenaRoundSettings settings = roundSettings(cfg, std::nullopt);   // gating ignored
    // Every generation from disk.
    const MatchFn play = runMatches(cfg, runDir, -1, nullptr, &referenceNets.networks);
    calibrateFixedPlayers(results, references, settings, play);
    writeRatingFiles(outDir, results, references, cfg);
    if (log) {
        // e.g. "rerate calibration | ref1 vs 0: 0.91 of 80 games, elo 402.1"
        const std::map<int, double> ratings = fitBradleyTerry(results, 0);
        for (const ResultsTable::Pair& p : results.pairs()) {
            if (p.a >= 0 || p.b != 0) continue;
            char line[128];
            std::snprintf(line, sizeof line, " vs 0: %.2f of %d games, elo %.1f", p.scoreA / p.games,
                          p.games, ratings.at(p.a) + 0.0);
            *log << "rerate calibration | " << fixedPlayerName(p.a, cfg.botSimulations, numNetworks)
                 << line << std::endl;
        }
    }
    std::string logCsv = kRerateLogHeader;
    writeFileAtomically(logPath(outDir), logCsv);

    for (int g = 1; g <= last; ++g) {
        if (cfg.arenaEvery == 0 || g % cfg.arenaEvery != 0) continue;
        const auto start = Clock::now();
        const ArenaRoundResult round = runArenaRound(g, results, references, settings, play);
        writeRatingFiles(outDir, results, references, cfg);

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
            *log << head << opponentSummary(round, cfg.botSimulations, numNetworks) << tail << std::endl;
        }
    }
    return last;
}

} // namespace azgo
