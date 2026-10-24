#include "azgo/trainer.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

#include "azgo/arena.hpp"
#include "azgo/arena_round.hpp"
#include "azgo/backend.hpp"
#include "azgo/bn_calibration.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/coordinator.hpp"
#include "azgo/elo.hpp"
#include "azgo/engines.hpp"
#include "azgo/evaluator.hpp"
#include "azgo/game.hpp"
#include "azgo/grow.hpp"
#include "azgo/inference_hub.hpp"
#include "azgo/metal.hpp"
#include "azgo/optimizer.hpp"
#include "azgo/parallel.hpp"
#include "azgo/selfplay.hpp"
#include "azgo/selfplay_slice.hpp"
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

// The backend can run here, and with metal its hubs fit in GPU memory at
// inference_batch: self-play's hub (two slots), or an arena wave's hubs (one
// slot each when there are several; playMatches). Checked before a run
// writes anything, so a too-large inference_batch fails up front rather than
// in the first arena. With pipeline = on self-play runs beside the
// validation loss and then beside the arena, so each of those is checked
// together with self-play's hub (the trainer's own: with selfplay_engines
// or arena_engines the coordinator's preflight checks the engines' hubs).
void checkBackend(const Config& cfg) {
    checkBackendAvailable(cfg.backend);
#ifdef AZ_USE_METAL
    if (cfg.backend == "metal" && cfg.pipeline) {
        std::vector<GpuHubs> selfplay, others[2];
        if (cfg.selfplayEngines.empty()) selfplay.push_back({1, kInferenceSlots, cfg.inferenceBatch});
        if (cfg.arenaEngines.empty()) others[0].push_back({arenaWaveNetworks(cfg), 1, cfg.inferenceBatch});
        if (cfg.validationLoss) others[1].push_back({1, 1, cfg.batchSize});
        for (const std::vector<GpuHubs>& other : others) {
            std::vector<GpuHubs> groups = selfplay;
            groups.insert(groups.end(), other.begin(), other.end());
            if (!groups.empty()) checkMetalInferenceMemory(cfg.netConfig(), groups);
        }
    } else if (cfg.backend == "metal") {
        checkMetalInferenceMemory(cfg.netConfig(), cfg.inferenceBatch, 1, kInferenceSlots);
        checkMetalInferenceMemory(cfg.netConfig(), cfg.inferenceBatch, arenaWaveNetworks(cfg), 1);
    }
#endif
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
// (generation `liveGeneration`), which is already in memory. A wave's
// matches are played at once (playMatches): each checkpoint is read once
// and each network gets one set of evaluators (one hub with metal) for all
// its matches, so the GPU sees every search of the wave together. With
// arena_engines and a coordinator, the wave's games are split across its
// engines instead (Coordinator::playArena), merged back by game index.
MatchesFn runMatches(const Config& cfg, const std::string& runDir, int liveGeneration,
                     const Network* liveNet, const std::vector<Network>* referenceNets,
                     Coordinator* coordinator = nullptr) {
    return [cfg, runDir, liveGeneration, liveNet, referenceNets,
            coordinator](const std::vector<MatchRequest>& wave) {
        std::map<int, std::unique_ptr<Network>> loaded;   // generation -> its checkpoint
        const size_t bots = cfg.botSimulations.size();
        auto player = [&](int id) {
            if (id < 0) {
                const size_t k = static_cast<size_t>(-id);
                return k <= bots ? ArenaPlayer::bot(cfg.botSimulations[k - 1])
                                 : ArenaPlayer::network(referenceNets->at(k - bots - 1));
            }
            if (liveNet && id == liveGeneration) return ArenaPlayer::network(*liveNet);
            std::unique_ptr<Network>& net = loaded[id];
            if (!net) net = std::make_unique<Network>(loadNetwork(checkpointPath(runDir, id)));
            return ArenaPlayer::network(*net);
        };
        std::vector<ArenaMatchSpec> specs;
        for (const MatchRequest& m : wave) specs.push_back({player(m.a), player(m.b), m.games, m.seed});
        if (coordinator && !cfg.arenaEngines.empty()) return coordinator->playArena(specs, arenaConfig(cfg));
        return playMatches(specs, arenaConfig(cfg), cfg.workers, cfg.backend, cfg.inferenceBatch);
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
    // No rating with arena_ladder = off: an empty field, not a number.
    char elo[32] = "";
    if (!std::isnan(s.elo)) std::snprintf(elo, sizeof elo, "%.1f", s.elo);
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
                  "%d,%d,%.2f,%.4f,%.4f,%.5f,%.5f,%g,%.3f,%.3f,%.3f,%.3f,%s,%s,%s,%s,%s,%s,%s%s\n",
                  s.generation, s.games, s.avgLength, s.passRate, s.blackWinRate,
                  s.policyLoss, s.valueLoss, s.lr, s.replayRatio, s.selfPlaySeconds,
                  s.trainSeconds, s.arenaSeconds, elo, winRate, distinct, ownership,
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

// A run's config.cfg. bn_calibrate defaults to true for a new config, but
// a config.cfg written before the key existed lacks it, and that run's
// generation 0 was not calibrated: there a missing key reads as false, so a
// resume records (in the rewritten config.cfg) what the run actually did.
Config loadRunConfig(const std::string& runDir) {
    Config before;
    before.bnCalibrate = false;
    return loadConfigFile(configPath(runDir), before);
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

// With selfplay_engines or arena_engines: the coordinator, its preflight
// done (it refuses what cannot run and prints the engine lines). Null on the
// default path, which reads no hosts file.
std::unique_ptr<Coordinator> makeCoordinator(const Config& cfg, const std::string& runDir,
                                             const TrainerOptions& opts, std::ostream* log) {
    if (cfg.selfplayEngines.empty() && cfg.arenaEngines.empty()) return nullptr;
    auto coordinator = std::make_unique<Coordinator>(
        cfg, runDir, loadHostsFile(resolveHostsPath(opts.hostsFile)), log);
    coordinator->preflight();
    return coordinator;
}

// The keys that change a generation's self-play games or who plays them,
// whose values differ between a and b (in config.cfg's order).
std::vector<std::string> selfPlayKeysChanged(const Config& a, const Config& b) {
    static const char* const kKeys[] = {
        "backend", "komi", "max_moves", "games_per_generation", "simulations", "mcts_batch",
        "selfplay_engines", "selfplay_allocation", "c_puct", "dirichlet_scale", "noise_fraction",
        "temperature_moves", "mcts_random_symmetry", "resignation", "resign_threshold_initial",
        "resign_disabled_fraction", "resign_false_positive", "resign_window", "resign_min_games"};
    std::map<std::string, std::string> va, vb;
    auto read = [](const Config& c, std::map<std::string, std::string>& out) {
        std::istringstream in(configToString(c));
        for (std::string line; std::getline(in, line);) {
            const size_t eq = line.find(" = ");
            if (eq != std::string::npos) out[line.substr(0, eq)] = line.substr(eq + 3);
        }
    };
    read(a, va);
    read(b, vb);
    std::vector<std::string> changed;
    for (const auto& [key, value] : va)
        if (std::find(std::begin(kKeys), std::end(kKeys), key) != std::end(kKeys) && vb[key] != value)
            changed.push_back(key);
    // config.cfg's order, not the map's.
    std::istringstream order(configToString(b));
    std::vector<std::string> ordered;
    for (std::string line; std::getline(order, line);) {
        const std::string key = line.substr(0, line.find(" = "));
        if (std::find(changed.begin(), changed.end(), key) != changed.end()) ordered.push_back(key);
    }
    return ordered;
}

// The engines a list plays with, for the note on a change: the list, or
// the default path's one engine.
std::string describeEngines(const std::vector<EngineSpec>& list, const Config& cfg) {
    if (list.empty()) return "local/" + cfg.backend;
    return printEngineList(list);
}

// One "arena engines: ..." line per wave of the arena's reports.
void printArenaEngines(std::ostream& log, const std::vector<SliceReport>& reports) {
    for (size_t k = 0; k < reports.size();) {
        std::vector<SliceReport> wave;
        for (const int w = reports[k].wave; k < reports.size() && reports[k].wave == w; ++k)
            wave.push_back(reports[k]);
        log << enginesLine(wave, "arena engines") << "\n";
    }
}

} // namespace

Trainer::Trainer(const Config& cfg, const std::string& runDir, Network net, TrainingState state)
    : cfg_(cfg), runDir_(runDir), net_(std::move(net)), state_(std::move(state)),
      log_(&std::cout) {}

Trainer::~Trainer() = default;

void Trainer::setLog(std::ostream* os) {
    log_ = os;
    if (coordinator_) coordinator_->setLog(os);
}

std::unique_ptr<Trainer> Trainer::startNew(const Config& cfg, const std::string& runDir,
                                           const RunInit& init, std::ostream* log,
                                           const TrainerOptions& opts) {
    checkConsistency(cfg);
    checkBackend(cfg);
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
        const Config donor = loadRunConfig(init.bufferFrom);
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

    std::unique_ptr<Coordinator> coordinator = makeCoordinator(cfg, runDir, opts, log);

    // Every check passed: the run starts here.
    printDeviceLine(cfg, log);
    fs::create_directories(fs::path(runDir) / "checkpoints");
    fs::create_directories(fs::path(runDir) / "sgf");
    writeFileAtomically(configPath(runDir), configToString(cfg));

    Network net = given ? std::move(*given) : Network(cfg.netConfig(), deriveSeed(cfg.seed, 0, kPhaseInit));
    // A random network's running statistics (mean 0, variance 1) do not
    // normalize, and generation 0 plays in inference mode: calibrate them
    // (bn_calibration.hpp). A given network keeps the statistics it has.
    if (!given && cfg.bnCalibrate)
        calibrateBatchNorm(net, deriveSeed(cfg.seed, 0, kPhaseBatchNorm), cfg.effectiveTrainThreads());
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
    // Each fixed player plays generation 0 now, so it has a rating before it
    // can be chosen. Only the ladder rates: with arena_ladder = off they
    // wait for the first round of a ladder switched on (runArenaRound
    // calibrates first, with these seeds).
    const bool arenaEngines = coordinator && !cfg.arenaEngines.empty();
    if (cfg.arenaLadder) {
        if (arenaEngines) coordinator->beginArena(0);
        calibrateFixedPlayers(state.results, state.references, roundSettings(cfg, std::nullopt),
                              runMatches(cfg, runDir, 0, &net, &referenceNets.networks, coordinator.get()));
    }

    if (given || donorBuffer)
        writeFileAtomically(
            (fs::path(runDir) / "init.txt").string(),
            "init_from = " + (init.initFrom.empty() ? std::string("none") : fs::absolute(init.initFrom).string()) + "\n" +
                "buffer_from = " + (init.bufferFrom.empty() ? std::string("none") : fs::absolute(init.bufferFrom).string()) + "\n" +
                "buffer_positions = " + std::to_string(state.buffer.positions()) + "\n" +
                "buffer_games = " + std::to_string(state.buffer.games()) + "\n");

    writeFileAtomically(logPath(runDir), kLogHeader);
    writeRatingFiles(runDir, state.results, state.references, cfg);
    // Who played the calibration games, as generation 0's rows.
    if (arenaEngines && !coordinator->arenaReports().empty()) {
        writeArenaManifest(runDir, 0, coordinator->arenaReports());
        if (log) printArenaEngines(*log, coordinator->arenaReports());
    }
    // state.bin last: until it exists the directory is not a run, so a crash
    // anywhere above leaves something startNew can simply redo.
    saveTrainingState(statePath(runDir), state);

    std::unique_ptr<Trainer> trainer(new Trainer(cfg, runDir, std::move(net), std::move(state)));
    trainer->referenceNets_ = std::move(referenceNets.networks);
    trainer->coordinator_ = std::move(coordinator);
    return trainer;
}

std::unique_ptr<Trainer> Trainer::resume(const std::string& runDir,
                                         const std::vector<std::string>& overrides,
                                         std::ostream* log, const TrainerOptions& opts) {
    if (!fs::exists(statePath(runDir)))
        throw ConfigError(runDir + " does not contain a run (no state.bin); start one with --config");

    const Config saved = loadRunConfig(runDir);
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
    checkBackend(cfg);

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
    if (state.pending && state.pending->records.front().boardSize != cfg.boardSize)
        throw std::runtime_error(statePath(runDir) + ": its pending self-play games are not " +
                                 std::to_string(cfg.boardSize) + "x" + std::to_string(cfg.boardSize));
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
    std::unique_ptr<Coordinator> coordinator = makeCoordinator(cfg, runDir, opts, log);
    printDeviceLine(cfg, log);
    if (log && saved.backend != cfg.backend)
        *log << "note: backend changed from " << saved.backend << " to " << cfg.backend
             << "; from here on results differ in the\nlast bits from a run that kept " << saved.backend
             << "\n";
    // Which engine plays which game is part of the result, like the backend.
    if (log && printEngineList(saved.selfplayEngines) != printEngineList(cfg.selfplayEngines))
        *log << "note: self-play engines changed from " << describeEngines(saved.selfplayEngines, saved)
             << " to " << describeEngines(cfg.selfplayEngines, cfg) << "; from here on results differ in the\n"
             << "last bits from a run that kept " << describeEngines(saved.selfplayEngines, saved) << "\n";
    // Which engine plays which game: fixed by the list (static), or by
    // timing and recorded (dynamic). Only meaningful with engines.
    if (log && saved.selfplayAllocation != cfg.selfplayAllocation && !cfg.selfplayEngines.empty())
        *log << "note: self-play allocation changed from " << saved.selfplayAllocation << " to "
             << cfg.selfplayAllocation
             << (cfg.selfplayAllocation == "dynamic"
                     ? "; from here on which engine plays which game depends on timing\n(recorded in "
                       "selfplay-games.csv)\n"
                     : "; from here on each engine plays a fixed slice of every generation\n");
    if (log && printEngineList(saved.arenaEngines) != printEngineList(cfg.arenaEngines))
        *log << "note: arena engines changed from " << describeEngines(saved.arenaEngines, saved) << " to "
             << describeEngines(cfg.arenaEngines, cfg) << "; from here on arena results differ in the\n"
             << "last bits from a run that kept " << describeEngines(saved.arenaEngines, saved) << "\n";
    if (log && saved.arenaLadder != cfg.arenaLadder)
        *log << "note: arena_ladder changed from " << (saved.arenaLadder ? "on" : "off") << " to "
             << (cfg.arenaLadder ? "on" : "off")
             << (cfg.arenaLadder
                     ? "; from here on the arena plays the full round again (the Elo ladder), its refit\n"
                       "including the gating games played meanwhile\n"
                     : "; from here on the arena plays only the gating match (the new network against the\n"
                       "best), log.csv's elo is empty, and elo.csv and references.csv keep the last round's "
                       "ratings\n(train --rerate rebuilds the ladder)\n");
    if (log && saved.pipeline != cfg.pipeline)
        *log << "note: pipeline changed from " << (saved.pipeline ? "on" : "off") << " to "
             << (cfg.pipeline ? "on" : "off")
             << (cfg.pipeline ? "; from here on self-play plays the best network of the last commit, "
                                "one generation\nbehind training\n"
                              : "; from here on self-play plays the best network after each arena\n");
    // A pipelined commit holds generation g+1's games: what they were played
    // with is settled, so a changed self-play key applies from g+2.
    if (log && state.pending) {
        const std::vector<std::string> changed = selfPlayKeysChanged(saved, cfg);
        if (!changed.empty()) {
            std::string keys;
            for (const std::string& k : changed) keys += (keys.empty() ? "" : ", ") + k;
            *log << "note: generation " << state.generation + 1
                 << "'s self-play games were committed before this resume (pipeline), played with the earlier "
                    "values of the changed\nself-play keys ("
                 << keys << "); their new values apply to self-play from generation " << state.generation + 2
                 << "\n";
        }
    }
    // Rewritten only after every check passed, so a refused resume leaves the
    // run directory untouched.
    writeFileAtomically(configPath(runDir), configToString(cfg));
    addMissingLogColumns(runDir);
    // The manifests describe committed generations only (engines on or off),
    // and the self-play manifests a pipelined commit's pending games too.
    const int selfPlayAfter = state.generation + (state.pending ? 2 : 1);
    pruneManifest(runDir, selfPlayAfter);
    pruneGamesManifest(runDir, selfPlayAfter);
    pruneArenaManifest(runDir, state.generation + 1);

    std::unique_ptr<Trainer> trainer(new Trainer(cfg, runDir, std::move(net), std::move(state)));
    trainer->referenceNets_ = std::move(referenceNets.networks);
    trainer->coordinator_ = std::move(coordinator);
    return trainer;
}

// What B of a generation decided: the best network after its arena, and
// the arena round for the progress line.
struct Trainer::RoundOutcome {
    int newBest = 0;
    bool arenaThisGeneration = false;
    ArenaRoundResult round;
};

// ---- one generation, in parts ------------------------------------------------
//
// A generation is self-play (SP), then B: the validation loss, the buffer,
// training, the checkpoint and the arena, then the commit. The sequential
// loop (pipeline = off) runs them in that order; the pipelined loop
// (pipeline = on) runs B of generation g beside SP of g + 1 and commits
// both together. Both loops use the same parts, so a pipelined generation
// differs from a sequential one only in which network played its games.

SelfPlayConfig Trainer::selfPlayConfig(float resignThreshold) const {
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
    sp.resignThreshold = resignThreshold;
    sp.resignDisabledFraction = cfg_.resignDisabledFraction;
    return sp;
}

PendingSelfPlay Trainer::playSelfPlay(int g, const Network& net, int network, float resignThreshold,
                                      const std::atomic<bool>* cancel) {
    const SelfPlayConfig sp = selfPlayConfig(resignThreshold);
    PendingSelfPlay played;
    played.network = network;
    played.resignation = sp.resignation;
    played.resignThreshold = sp.resignThreshold;
    played.maxMoves = sp.maxMoves;
    // The whole generation, games [0, G), as one slice
    // (azgo/selfplay_slice.hpp). One evaluator per worker; the network
    // itself is shared read-only. naive, blas: a single-threaded backend
    // and workspace per worker. metal: one batching hub that gathers
    // every worker's leaves into GPU passes; a position's outputs do not
    // depend on its batch, so the games are the same either way
    // (azgo/inference_hub.hpp). Game i's randomness depends only on i,
    // and its record is stored at index i, so the result does not
    // depend on which thread played it.
    const int numGames = cfg_.gamesPerGeneration;
    if (coordinator_ && !cfg_.selfplayEngines.empty()) {
        // selfplay_engines: a slice per engine, all at once (static), or
        // one queue all engines take from (dynamic); merged by index.
        // Remote engines load the network's checkpoint.
        played.records = coordinator_->play(g, checkpointPath(runDir_, network), net, sp, numGames);
    } else {
        EngineRun run{cfg_.backend, std::min(cfg_.effectiveWorkers(), numGames), cfg_.inferenceBatch};
        run.cancel = cancel;
        played.records = playSelfPlayGames(net, sp, cfg_.seed, g, 0, numGames, run);
    }
    return played;
}

void Trainer::processSelfPlay(int g, const PendingSelfPlay& played, GenerationStats& stats) {
    const int n = cfg_.boardSize;
    const std::vector<GameRecord>& records = played.records;
    const int numGames = static_cast<int>(records.size());
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
    // The flag and v_resign are the ones the games were played with.
    if (played.resignation) {
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
        stats.resignThreshold = played.resignThreshold;
        if (enabled > 0) stats.resignRate = static_cast<double>(resigned) / enabled;
        if (!state_.resignWindow.empty())
            stats.resignFalsePositive =
                resignFalsePositiveRate(state_.resignWindow, cfg_.resignWindow, played.resignThreshold);
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
            Game replay(n, r.komi, played.maxMoves);
            for (int a : r.moves) replay.play(a);
            sgf = gameToSgf(n, r.komi, r.moves, replay.scoreForBlack(), comment);
        }
        writeFileAtomically(sgfPath(runDir_, g, i), sgf);
    }
}

Trainer::RoundOutcome Trainer::trainAndRate(int g, std::vector<GameRecord>& records, GenerationStats& stats,
                                            const std::atomic<bool>* cancel) {
    auto checkCancel = [&] {
        if (cancel && *cancel) throw SliceCancelled();
    };
    long long plies = 0;
    for (const GameRecord& r : records) plies += r.positions();

    // ---- Validation loss (a measurement; changes nothing) -------------------
    // net_ is generation g-1's network, about to train on these games, and
    // (sequentially, without gating) the one that played them. Measured
    // before they join the buffer, they are data it has never trained on.
    // The buffer, not yet changed, is its training distribution: the window
    // it trained on at generation g-1 (a sample of it, at random symmetries,
    // so some of these positions it drew rarely or never). At generation 1
    // the buffer is empty, or with --buffer-from a donor's games that
    // generation 0 (a fresh network, or --init-from's) never trained on.
    // Inference mode and the identity symmetry, so nothing is drawn from any
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
    const auto validationStart = Clock::now();
    if (cfg_.validationLoss) {
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
    }
    const double validationSeconds = secondsSince(validationStart);
    pipeline_probe::crashPoint(pipeline_probe::Point::Validation, g);

    for (GameRecord& r : records) state_.buffer.add(std::move(r));   // index order

    // ---- Train ----------------------------------------------------------------
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
            checkCancel();   // pipelined: self-play failed, nothing will be committed
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
    pipeline_probe::crashPoint(pipeline_probe::Point::Training, g);

    // ---- Checkpoint -----------------------------------------------------------
    // Written before state.bin (the commit). A crash in between leaves state.bin
    // at generation g-1; rerunning then reproduces generation g exactly (every
    // random stream is derived from the seed) and overwrites this file with
    // identical bytes. The same holds for the SGF, results.csv and elo.csv.
    saveNetwork(net_, checkpointPath(runDir_, g));
    pipeline_probe::crashPoint(pipeline_probe::Point::Checkpoint, g);
    checkCancel();

    // ---- Arena and Elo ----------------------------------------------------------
    const auto arenaStart = Clock::now();
    RoundOutcome outcome;
    outcome.newBest = cfg_.gating ? state_.bestGeneration : g;
    outcome.arenaThisGeneration = cfg_.arenaEvery > 0 && g % cfg_.arenaEvery == 0;
    // With arena_engines, each wave's games are split across the engines.
    const bool arenaEngines = coordinator_ && !cfg_.arenaEngines.empty();
    if (arenaEngines) coordinator_->beginArena(g);
    if (outcome.arenaThisGeneration) {
        const std::optional<int> best =
            cfg_.gating ? std::optional<int>(state_.bestGeneration) : std::nullopt;
        const MatchesFn play = runMatches(cfg_, runDir_, g, &net_, &referenceNets_, coordinator_.get());
        // arena_ladder = off (gating is on: checkConsistency): the gating
        // match alone, the games the full round would play against the best.
        outcome.round = cfg_.arenaLadder
                            ? runArenaRound(g, state_.results, state_.references, roundSettings(cfg_, best), play)
                            : runGatingMatch(g, state_.results, roundSettings(cfg_, best), play);
        for (const ArenaMatch& m : outcome.round.matches) {
            if (m.opponent == g - 1) {
                stats.winRateVsPrev = m.result.scoreA / m.result.games;
                stats.distinctVsPrev = m.result.distinctGames;
            }
            if (best && m.opponent == *best &&
                passesGate(m.result.scoreA, m.result.games, cfg_.gatingThreshold))
                outcome.newBest = g;
        }
        // Refit every rating from all results so far: Bradley-Terry does not
        // depend on the order results arrived in, so a resumed run agrees.
        // Without the ladder only results.csv: a fit of gating games alone
        // would chain each generation to the best network, so elo.csv and
        // references.csv keep the last ladder round's.
        if (cfg_.arenaLadder) writeRatingFiles(runDir_, state_.results, state_.references, cfg_);
        else writeFileAtomically(resultsPath(runDir_), state_.results.toCsv());
    }
    stats.elo = cfg_.arenaLadder ? ratingAtOrBefore(state_.results, g) : std::nan("");
    stats.arenaSeconds = secondsSince(arenaStart);
    pipeline_probe::crashPoint(pipeline_probe::Point::Arena, g);
    return outcome;
}

void Trainer::printGenerationLine(const GenerationStats& stats, const RoundOutcome& outcome) const {
    // e.g. "train loss p 1.73 v 0.09 lr 0.002 2.5s", with "o 0.412" after
    // the value loss when the network has an ownership head.
    char ownership[32] = "";
    if (stats.ownershipLoss >= 0) std::snprintf(ownership, sizeof ownership, " o %.3f", stats.ownershipLoss);
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
                  stats.generation, stats.games, stats.avgLength, resign, stats.selfPlaySeconds, stats.policyLoss,
                  stats.valueLoss, ownership, stats.lr, validation.c_str(), stats.trainSeconds);
    *log_ << line;
    if (outcome.arenaThisGeneration && !cfg_.arenaLadder) {
        // e.g. "arena gating vs 3:0.55 (distinct 40/40) 812.3s": the one match, no rating.
        char tail[64];
        std::snprintf(tail, sizeof tail, " (distinct %d/%d) %.1fs", outcome.round.matches[0].result.distinctGames,
                      cfg_.arenaGames, stats.arenaSeconds);
        *log_ << "arena gating "
              << opponentSummary(outcome.round, cfg_.botSimulations, static_cast<int>(cfg_.referenceNetworks.size()))
              << tail;
    } else if (!cfg_.arenaLadder) {
        *log_ << "no arena";
    } else if (outcome.arenaThisGeneration) {
        // e.g. "arena elo 512.3 vs 149:0.55 120:0.70 ref1:0.40 (distinct 19/20) promoted 3.2s"
        char elo[48], tail[64];
        std::snprintf(elo, sizeof elo, "arena elo %.1f ", stats.elo);
        std::snprintf(tail, sizeof tail, " (distinct %d/%d)%s %.1fs", stats.distinctVsPrev, cfg_.arenaGames,
                      outcome.round.promoted ? " promoted" : "", stats.arenaSeconds);
        *log_ << elo
              << opponentSummary(outcome.round, cfg_.botSimulations, static_cast<int>(cfg_.referenceNetworks.size()))
              << tail;
    } else {
        char noArena[48];
        std::snprintf(noArena, sizeof noArena, "no arena, elo %.1f", stats.elo);
        *log_ << noArena;
    }
    if (cfg_.gating) *log_ << " | best " << state_.bestGeneration;
    *log_ << std::endl;
}

// ---- the sequential loop (pipeline = off) ---------------------------------------

GenerationStats Trainer::runGeneration() {
    // A generation that failed may have changed the state in memory (the
    // buffer, the resignation window, a pipelined state's pending games)
    // without committing it: only a resume continues correctly from there.
    if (failed_)
        throw std::logic_error("a generation of this run failed in this process; resume it (train --resume)");
    try {
        return cfg_.pipeline ? runPipelined() : runSequential();
    } catch (...) {
        failed_ = true;
        throw;
    }
}

GenerationStats Trainer::runSequential() {
    const int g = state_.generation + 1;
    GenerationStats stats;
    stats.generation = g;

    // ---- 1. Self-play -----------------------------------------------------
    // Games are generated by the best network so far with gating, which
    // may be older than the one being trained, else by the latest. A run
    // switched from pipeline = on has generation g's games already: they
    // are committed data, used as they are.
    const auto selfPlayStart = Clock::now();
    PendingSelfPlay played;
    const bool playedNow = !state_.pending;
    double pendingSeconds = 0;
    if (!playedNow) {
        played = std::move(*state_.pending);
        state_.pending.reset();
        pendingSeconds = pendingSelfPlaySeconds(g);
    } else if (cfg_.gating && state_.bestGeneration != state_.generation) {
        const Network best = loadNetwork(checkpointPath(runDir_, state_.bestGeneration));
        played = playSelfPlay(g, best, state_.bestGeneration, state_.resignThreshold, nullptr);
    } else {
        played = playSelfPlay(g, net_, state_.generation, state_.resignThreshold, nullptr);
    }
    processSelfPlay(g, played, stats);
    stats.selfPlaySeconds = pendingSeconds + secondsSince(selfPlayStart);

    // ---- 2-4. Validation, training, checkpoint, arena -----------------------------
    const RoundOutcome outcome = trainAndRate(g, played.records, stats, nullptr);

    // ---- 5. Training state --------------------------------------------------
    state_.bestGeneration = outcome.newBest;
    state_.generation = g;
    // Who played which slice, written before the commit point: a crash in
    // between leaves rows of an uncommitted generation, which its redo
    // replaces. Games committed before (pipeline switched off) have theirs.
    const bool selfplayEngines = playedNow && coordinator_ && !cfg_.selfplayEngines.empty();
    if (selfplayEngines) {
        writeManifest(runDir_, g, coordinator_->reports());
        // Dynamic allocation: which engine played each game.
        if (!coordinator_->assignment().empty()) writeGamesManifest(runDir_, g, coordinator_->assignment());
    }
    const bool arenaEngines = coordinator_ && !cfg_.arenaEngines.empty();
    if (arenaEngines) writeArenaManifest(runDir_, g, coordinator_->arenaReports());
    saveTrainingState(statePath(runDir_), state_);
    // log.csv is a human-facing record (it contains timings) and is appended
    // after state.bin: a crash between the two loses one row, never corrupts
    // the run.
    appendLogRow(runDir_, stats);

    if (log_) {
        if (selfplayEngines) *log_ << enginesLine(coordinator_->reports()) << "\n";
        if (arenaEngines) printArenaEngines(*log_, coordinator_->arenaReports());
        printGenerationLine(stats, outcome);
    }
    return stats;
}

// ---- the pipelined loop (pipeline = on) -------------------------------------------

namespace {

std::string pendingSecondsPath(const std::string& runDir) {
    return (fs::path(runDir) / "pending-selfplay.txt").string();
}

// The failure itself, rather than the cancellation it caused in the other
// half of an iteration.
[[noreturn]] void rethrowFailure(std::exception_ptr first, std::exception_ptr second) {
    for (std::exception_ptr e : {first, second}) {
        if (!e) continue;
        try {
            std::rethrow_exception(e);
        } catch (const SliceCancelled&) {
            continue;
        } catch (...) {
        }
        std::rethrow_exception(e);
    }
    std::rethrow_exception(first ? first : second);
}

} // namespace

namespace {

// pending-selfplay.txt: "<generation> <seconds>" per line, the generations
// whose self-play a commit holds or is about to hold.
std::map<int, double> readPendingSeconds(const std::string& runDir) {
    std::map<int, double> out;
    std::ifstream in(pendingSecondsPath(runDir));
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream fields(line);
        int g = -1;
        double seconds = -1;
        if (fields >> g >> seconds && std::isfinite(seconds) && seconds >= 0) out[g] = seconds;
    }
    return out;
}

} // namespace

double Trainer::pendingSelfPlaySeconds(int g) const {
    if (pendingSeconds_.first == g) return pendingSeconds_.second;
    // Played by an earlier process: its time, if it left it.
    const std::map<int, double> recorded = readPendingSeconds(runDir_);
    auto it = recorded.find(g);
    return it == recorded.end() ? 0.0 : it->second;
}

void Trainer::keepPendingSeconds(int g, double seconds) {
    pendingSeconds_ = {g, seconds};
    // Written before state.bin, and keeping generation g-1's line: a crash
    // between the two leaves state.bin's pending generation g-1, whose time
    // is still here.
    std::map<int, double> recorded = readPendingSeconds(runDir_);
    std::string text = "# generation seconds: the wall time of each pending generation's self-play\n";
    recorded[g] = seconds;
    for (const auto& [generation, s] : recorded) {
        if (generation < g - 1) continue;
        char line[64];
        std::snprintf(line, sizeof line, "%d %.3f\n", generation, s);
        text += line;
    }
    writeFileAtomically(pendingSecondsPath(runDir_), text);
}

void Trainer::prepareForThreads() {
    // The BLAS backend sets its environment once, with setenv, which is not
    // thread-safe: that first call happens here, before the two halves of an
    // iteration run, whenever training or an engine may use BLAS.
    bool blas = cpuBackendName(cfg_.backend) == "blas";
    for (const std::vector<EngineSpec>* list : {&cfg_.selfplayEngines, &cfg_.arenaEngines})
        for (const EngineSpec& e : *list) blas = blas || e.backend == "blas";
    if (blas) makeBackend("blas", 1);
}

void Trainer::playAhead() {
    // The prologue: generation g's games, alone, with the network of the
    // last commit (as the sequential loop would), committed with
    // generation g-1 so that an iteration can start from them.
    const int g = state_.generation + 1;
    const int network = cfg_.gating ? state_.bestGeneration : state_.generation;
    const auto start = Clock::now();
    if (coordinator_) coordinator_->resetCancel();
    PendingSelfPlay played;
    if (network != state_.generation) {
        const Network best = loadNetwork(checkpointPath(runDir_, network));
        played = playSelfPlay(g, best, network, state_.resignThreshold, nullptr);
    } else {
        played = playSelfPlay(g, net_, network, state_.resignThreshold, nullptr);
    }
    const double seconds = secondsSince(start);
    pipeline_probe::crashPoint(pipeline_probe::Point::PrologueSelfPlay, g);
    const int games = static_cast<int>(played.records.size());
    state_.pending = std::move(played);
    const bool selfplayEngines = coordinator_ && !cfg_.selfplayEngines.empty();
    if (selfplayEngines) {
        writeManifest(runDir_, g, coordinator_->reports());
        if (!coordinator_->assignment().empty()) writeGamesManifest(runDir_, g, coordinator_->assignment());
    }
    keepPendingSeconds(g, seconds);
    pipeline_probe::crashPoint(pipeline_probe::Point::PrologueManifests, g);
    saveTrainingState(statePath(runDir_), state_);
    if (log_) {
        if (selfplayEngines)
            *log_ << enginesLine(coordinator_->reports(), "selfplay engines (gen " + std::to_string(g) + ")") << "\n";
        char line[160];
        std::snprintf(line, sizeof line,
                      "pipeline: gen %d self-play by network %d, %d games %.1fs, committed with generation %d\n", g,
                      network, games, seconds, g - 1);
        *log_ << line << std::flush;
    }
}

GenerationStats Trainer::runPipelined() {
    prepareForThreads();
    if (!state_.pending) playAhead();
    const auto iterationStart = Clock::now();
    const int g = state_.generation + 1;
    GenerationStats stats;
    stats.generation = g;

    // ---- 1. Generation g's games, committed with g-1 -----------------------------
    PendingSelfPlay played = std::move(*state_.pending);
    state_.pending.reset();
    stats.selfPlaySeconds = pendingSelfPlaySeconds(g);
    processSelfPlay(g, played, stats);   // also v_resign for g + 1
    pipeline_probe::crashPoint(pipeline_probe::Point::Processed, g);

    // ---- 2. Generation g+1's self-play, beside 3 ----------------------------------
    // By the best network of the last commit (g-1's), with its own copy:
    // the trainer's network trains in place below.
    const int network = cfg_.gating ? state_.bestGeneration : state_.generation;
    const std::unique_ptr<const Network> aheadNet =
        network == state_.generation ? std::make_unique<const Network>(net_)
                                     : std::make_unique<const Network>(loadNetwork(checkpointPath(runDir_, network)));
    const float aheadThreshold = state_.resignThreshold;
    if (coordinator_) coordinator_->resetCancel();
    std::atomic<bool> stopAhead{false}, aheadFailed{false};
    PendingSelfPlay ahead;
    double aheadSeconds = 0;
    std::exception_ptr aheadError, roundError;
    std::thread aheadThread([&] {
        const auto start = Clock::now();
        try {
            ahead = playSelfPlay(g + 1, *aheadNet, network, aheadThreshold, &stopAhead);
            pipeline_probe::crashPoint(pipeline_probe::Point::SelfPlay, g + 1);
        } catch (...) {
            aheadError = std::current_exception();
            aheadFailed = true;
            if (coordinator_) coordinator_->cancel();   // the arena, if it is playing
        }
        aheadSeconds = secondsSince(start);
    });

    // ---- 3. Validation, training, checkpoint and arena of generation g -----------
    RoundOutcome outcome;
    try {
        outcome = trainAndRate(g, played.records, stats, &aheadFailed);
    } catch (...) {
        roundError = std::current_exception();
        stopAhead = true;
        if (coordinator_) coordinator_->cancel();
    }
    // ---- 4. The barrier ---------------------------------------------------------
    aheadThread.join();
    if (roundError || aheadError) rethrowFailure(roundError, aheadError);

    // ---- 5. Commit generation g with generation g+1's games ----------------------
    state_.bestGeneration = outcome.newBest;
    state_.generation = g;
    const int aheadGames = static_cast<int>(ahead.records.size());
    state_.pending = std::move(ahead);
    const bool selfplayEngines = coordinator_ && !cfg_.selfplayEngines.empty();
    if (selfplayEngines) {
        writeManifest(runDir_, g + 1, coordinator_->reports());
        if (!coordinator_->assignment().empty()) writeGamesManifest(runDir_, g + 1, coordinator_->assignment());
    }
    const bool arenaEngines = coordinator_ && !cfg_.arenaEngines.empty();
    if (arenaEngines) writeArenaManifest(runDir_, g, coordinator_->arenaReports());
    keepPendingSeconds(g + 1, aheadSeconds);
    pipeline_probe::crashPoint(pipeline_probe::Point::Manifests, g);
    saveTrainingState(statePath(runDir_), state_);
    pipeline_probe::crashPoint(pipeline_probe::Point::State, g);
    appendLogRow(runDir_, stats);

    if (log_) {
        if (selfplayEngines)
            *log_ << enginesLine(coordinator_->reports(), "selfplay engines (gen " + std::to_string(g + 1) + ")")
                  << "\n";
        if (arenaEngines) printArenaEngines(*log_, coordinator_->arenaReports());
        printGenerationLine(stats, outcome);
        // e.g. "pipeline: gen 13 self-play by network 11, 400 games 15402.6s, beside gen 12's training and
        // arena 8045.7s; generation 12 took 15403.1s"
        char line[256];
        std::snprintf(line, sizeof line,
                      "pipeline: gen %d self-play by network %d, %d games %.1fs, beside gen %d's training and "
                      "arena %.1fs; generation %d took %.1fs\n",
                      g + 1, network, aheadGames, aheadSeconds, g, stats.trainSeconds + stats.arenaSeconds, g,
                      secondsSince(iterationStart));
        *log_ << line << std::flush;
    }
    return stats;
}

void Trainer::runUntil(int lastGeneration) {
    while (generation() < lastGeneration) runGeneration();
}

namespace pipeline_probe {
namespace {
std::mutex gCrashMutex;
std::map<Point, int> gCrashes;   // point -> the generation it fires at
const char* pointName(Point p) {
    switch (p) {
    case Point::PrologueSelfPlay: return "the prologue's self-play";
    case Point::PrologueManifests: return "the prologue's manifests";
    case Point::Processed: return "processing self-play";
    case Point::SelfPlay: return "self-play";
    case Point::Validation: return "the validation loss";
    case Point::Training: return "training";
    case Point::Checkpoint: return "the checkpoint";
    case Point::Arena: return "the arena";
    case Point::Manifests: return "the manifests";
    case Point::State: return "state.bin";
    }
    return "?";
}
} // namespace
void crashAt(Point point, int generation) {
    std::lock_guard<std::mutex> lock(gCrashMutex);
    gCrashes[point] = generation;
}
void reset() {
    std::lock_guard<std::mutex> lock(gCrashMutex);
    gCrashes.clear();
}
void crashPoint(Point point, int generation) {
    std::lock_guard<std::mutex> lock(gCrashMutex);
    auto it = gCrashes.find(point);
    if (it == gCrashes.end() || it->second != generation) return;
    gCrashes.erase(it);
    throw std::runtime_error(std::string("pipeline_probe: crash after ") + pointName(point) + " of generation " +
                             std::to_string(generation));
}
} // namespace pipeline_probe

int rerateRun(const std::string& runDir, const std::vector<std::string>& overrides,
              std::ostream* log) {
    if (!fs::exists(configPath(runDir)))
        throw ConfigError(runDir + " does not contain a run (no config.cfg)");
    const Config saved = loadRunConfig(runDir);
    Config cfg = saved;
    for (const std::string& o : overrides) applyOverride(cfg, o);
    const std::vector<std::string> diff = architectureDiff(saved, cfg);
    if (!diff.empty()) {
        std::string keys;
        for (const std::string& k : diff) keys += (keys.empty() ? "" : ", ") + k;
        throw ConfigError("cannot change architecture keys when re-rating: " + keys);
    }
    // Re-rating rebuilds the ladder, whatever arena_ladder says (it ignores
    // gating too), and checks the GPU memory of the ladder's waves.
    cfg.arenaLadder = true;
    if (!fs::exists(checkpointPath(runDir, 0)))
        throw std::runtime_error(checkpointPath(runDir, 0) + ": missing");

    // Every generation is measured against g-1, so the replay ends at the
    // first missing checkpoint, and at the generation state.bin committed:
    // a generation that failed after writing its checkpoint (in the arena,
    // say) left that file behind, but the run has not got that far.
    int last = 0;
    while (fs::exists(checkpointPath(runDir, last + 1))) ++last;
    // A damaged state.bin does not block a replay that needs nothing else
    // from it: then every checkpoint on disk counts, as before.
    if (fs::exists(statePath(runDir))) {
        try {
            last = std::min(last, trainingStateGeneration(statePath(runDir)));
        } catch (const std::runtime_error&) {
        }
    }

    // Fail on an unavailable backend before anything is written. Re-rating
    // plays no self-play, so a pipelined run's combined check does not apply.
    Config sequential = cfg;
    sequential.pipeline = false;
    checkBackend(sequential);
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
    const MatchesFn play = runMatches(cfg, runDir, -1, nullptr, &referenceNets.networks);
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
