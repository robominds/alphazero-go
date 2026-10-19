#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>
#include "azgo/arena.hpp"
#include "azgo/backend.hpp"
#include "azgo/bn_calibration.hpp"
#include "azgo/checkpoint.hpp"
#include "azgo/build_info.hpp"
#include "azgo/config.hpp"
#include "azgo/coordinator.hpp"
#include "azgo/elo.hpp"
#include "azgo/engines.hpp"
#include "azgo/metal.hpp"
#include "azgo/network.hpp"
#include "azgo/rng.hpp"
#include "azgo/run_state.hpp"
#include "azgo/selfplay.hpp"
#include "azgo/serialize.hpp"
#include "azgo/trainer.hpp"
#include "azgo/validation.hpp"

using namespace azgo;
namespace fs = std::filesystem;

// A 5x5 run small enough that a generation takes a fraction of a second.
static Config tinyConfig() {
    Config c;
    c.boardSize = 5;
    c.history = 2;
    c.blocks = 1;
    c.filters = 8;
    c.valueHidden = 8;
    c.gamesPerGeneration = 4;
    c.simulations = 8;
    c.mctsBatch = 4;
    c.temperatureMoves = 4;
    c.bufferPositions = 1000;
    c.trainSteps = 4;
    c.batchSize = 16;
    c.arenaGames = 2;
    c.arenaSimulations = 4;
    c.arenaOpponents = 2;
    c.workers = 2;
    c.sgfPerGeneration = 1;
    c.lrSchedule = "0:0.02";
    c.botSimulations = {8};   // one cheap bot: player -1
    c.calibrationGames = 2;
    return c;
}

// A run directory under the system temp dir; removed on construction (stale
// leftovers from a crashed test) and on destruction.
struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& name)
        : path(fs::temp_directory_path() /
               ("azgo_test_trainer_" + std::to_string(getpid()) + "_" + name)) {
        fs::remove_all(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string str() const { return path.string(); }
    std::string file(const std::string& rel) const { return (path / rel).string(); }
};

static std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    assert(in && "file should exist");
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void writeFile(const std::string& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << bytes;
    assert(out && "file should be writable");
}

static int countLines(const std::string& s) {
    return static_cast<int>(std::count(s.begin(), s.end(), '\n'));
}

static std::vector<std::string> splitLines(const std::string& s) {
    std::vector<std::string> lines;
    std::istringstream in(s);
    for (std::string line; std::getline(in, line);) lines.push_back(line);
    return lines;
}

// One CSV row's fields; a trailing comma gives a trailing empty field.
static std::vector<std::string> splitFields(const std::string& row) {
    std::vector<std::string> fields(1);
    for (char ch : row) {
        if (ch == ',') fields.emplace_back();
        else fields.back() += ch;
    }
    return fields;
}

// log.csv's data row `index` (0 = generation 1's; -1 = the last), as
// column name -> field. Asserts the row has as many fields as the header.
static std::map<std::string, std::string> logRow(const std::string& runDir, int index = -1) {
    const std::vector<std::string> rows = splitLines(readFile((fs::path(runDir) / "log.csv").string()));
    assert(rows.size() >= 2);
    const std::vector<std::string> header = splitFields(rows[0]);
    const std::vector<std::string> fields =
        splitFields(rows[index < 0 ? rows.size() - 1 : static_cast<size_t>(index) + 1]);
    assert(fields.size() == header.size());
    std::map<std::string, std::string> out;
    for (size_t i = 0; i < header.size(); ++i) out[header[i]] = fields[i];
    return out;
}

// The first `count` fields of every line of log.csv: the file as code that
// had only those columns wrote it.
static std::string firstLogColumns(const std::string& runDir, size_t count) {
    std::string out;
    for (const std::string& row : splitLines(readFile((fs::path(runDir) / "log.csv").string()))) {
        const std::vector<std::string> fields = splitFields(row);
        assert(fields.size() > count);
        for (size_t i = 0; i < count; ++i) out += (i ? "," : "") + fields[i];
        out += "\n";
    }
    return out;
}

constexpr int kLogCommas = 24;   // 25 columns

// "%.5f" of x: how log.csv prints a loss.
static std::string loss5(float x) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.5f", x);
    return buf;
}

static std::unique_ptr<Trainer> startQuiet(const Config& c, const TempDir& dir) {
    std::unique_ptr<Trainer> t = Trainer::startNew(c, dir.str());
    t->setLog(nullptr);
    return t;
}

static std::unique_ptr<Trainer> resumeQuiet(const TempDir& dir, const std::vector<std::string>& overrides) {
    std::unique_ptr<Trainer> t = Trainer::resume(dir.str(), overrides);
    t->setLog(nullptr);
    return t;
}

// (a, b) -> Pair from results.csv.
static std::map<std::pair<int, int>, ResultsTable::Pair> readResults(const TempDir& dir) {
    std::map<std::pair<int, int>, ResultsTable::Pair> out;
    for (const ResultsTable::Pair& p : ResultsTable::fromCsv(readFile(dir.file("results.csv"))).pairs())
        out[{p.a, p.b}] = p;
    return out;
}

static void test_start_new_writes_run_files() {
    TempDir dir("start");
    const Config c = tinyConfig();
    std::unique_ptr<Trainer> t = startQuiet(c, dir);
    assert(t->generation() == 0);

    assert(fs::is_directory(dir.path / "sgf"));
    assert(configToString(loadConfigFile(configPath(dir.str()))) == configToString(c));
    assert(readCheckpointConfig(checkpointPath(dir.str(), 0)) == c.netConfig());

    TrainingState s = loadTrainingState(statePath(dir.str()));
    assert(s.generation == 0 && s.bestGeneration == 0 && s.samplesTrained == 0);
    assert(s.buffer.positions() == 0 && s.buffer.capacity() == 1000);
    assert((s.references == std::vector<int>{-1, 0}));   // the bot and generation 0
    // The bot played calibration_games = 2 games against generation 0.
    const std::vector<ResultsTable::Pair> calibration = s.results.pairs();
    assert(calibration.size() == 1);
    assert(calibration[0].a == -1 && calibration[0].b == 0 && calibration[0].games == 2);
    assert(readFile(dir.file("results.csv")) == s.results.toCsv());
    const std::vector<std::string> refs = splitLines(readFile(dir.file("references.csv")));
    assert(refs.size() == 3 && refs[0] == "player,kind,simulations,promoted_at,elo");
    assert(refs[1].rfind("-1,bot,8,0,", 0) == 0 && refs[1].size() > 11);   // rated
    assert(refs[2] == "0,checkpoint,,0,0.0");
    // Momentum starts at zero, one buffer per parameter tensor.
    Network net = loadNetwork(checkpointPath(dir.str(), 0));
    std::vector<ParamView> params = net.params();
    assert(s.velocity.size() == params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        assert(s.velocity[i].size() == params[i].size);
        for (float v : s.velocity[i]) assert(v == 0.0f);
    }

    assert(readFile(dir.file("log.csv")) ==
           "generation,games,avg_length,pass_rate,black_win_rate,policy_loss,value_loss,"
           "lr,replay_ratio,selfplay_s,train_s,arena_s,elo,winrate_vs_prev,distinct_vs_prev,"
           "ownership_loss,resign_threshold,resign_rate,resign_false_positive,"
           "val_policy_loss,val_value_loss,val_ownership_loss,"
           "seen_policy_loss,seen_value_loss,seen_ownership_loss\n");
    assert(readFile(dir.file("elo.csv")) ==
           "generation,elo,winrate_vs_prev,wr_low,wr_high\n0,0.0,,,\n");
}

static void test_two_generations() {
    TempDir dir("two");
    std::unique_ptr<Trainer> t = startQuiet(tinyConfig(), dir);

    // Generation 1 by hand, to check the returned statistics.
    GenerationStats st = t->runGeneration();
    assert(st.generation == 1 && st.games == 4);
    assert(st.avgLength >= 1 && st.avgLength <= 2 * 5 * 5);   // move cap 50
    assert(st.passRate >= 0 && st.passRate <= 1);
    assert(st.blackWinRate >= 0 && st.blackWinRate <= 1);
    assert(std::isfinite(st.policyLoss) && st.policyLoss > 0);
    assert(std::isfinite(st.valueLoss) && st.valueLoss >= 0);
    assert(st.lr == 0.02f);
    // 4 steps x 16 samples drawn from 4 games of avgLength plies each.
    assert(std::fabs(st.replayRatio - 64.0 / (4 * st.avgLength)) < 1e-9);
    assert(st.winRateVsPrev >= 0 && st.winRateVsPrev <= 1);   // played generation 0
    assert(std::isfinite(st.elo));
    assert(st.distinctVsPrev >= 1 && st.distinctVsPrev <= 2);   // arena_games = 2

    assert(st.ownershipLoss == -1.0f);   // no ownership head
    assert(!st.resignation);
    {
        // The log row has the same diversity count, and empty ownership_loss
        // and resignation fields (no head, no resignation).
        const std::map<std::string, std::string> row = logRow(dir.str());
        assert(!row.at("distinct_vs_prev").empty() && std::stoi(row.at("distinct_vs_prev")) == st.distinctVsPrev);
        for (const char* empty : {"ownership_loss", "resign_threshold", "resign_rate", "resign_false_positive"})
            assert(row.at(empty).empty());
    }

    t->runUntil(2);
    assert(t->generation() == 2);
    assert(fs::exists(checkpointPath(dir.str(), 1)));
    assert(fs::exists(checkpointPath(dir.str(), 2)));
    assert(!fs::exists(checkpointPath(dir.str(), 3)));

    // sgf_per_generation = 1: exactly one game per generation.
    const std::string sgf = readFile(dir.file("sgf/gen_0001_game_00.sgf"));
    assert(sgf.rfind("(;GM[1]", 0) == 0);
    assert(sgf.find("SZ[5]") != std::string::npos);
    assert(fs::exists(dir.path / "sgf" / "gen_0002_game_00.sgf"));
    assert(!fs::exists(dir.path / "sgf" / "gen_0001_game_01.sgf"));

    assert(countLines(readFile(dir.file("log.csv"))) == 3);   // header + 2 rows

    // References start as {bot -1, 0}, and arena_opponents = 2 leaves room for
    // all of them: generation 1 plays 0 and the bot, generation 2 plays 1, 0
    // and the bot. A promoted generation played calibration_games = 2 more
    // games against the strongest reference, so exactly one of its pairs has 4.
    TrainingState s = loadTrainingState(statePath(dir.str()));
    std::map<std::pair<int, int>, ResultsTable::Pair> pairs = readResults(dir);
    assert(pairs.size() == 6);
    assert(pairs.at({-1, 0}).games == 2);   // calibration
    for (std::pair<int, int> key : {std::make_pair(0, 1), std::make_pair(-1, 1), std::make_pair(0, 2),
                                    std::make_pair(1, 2), std::make_pair(-1, 2)}) {
        assert(pairs.count(key));
        assert(pairs[key].games == 2 || pairs[key].games == 4);
    }
    for (int g = 1; g <= 2; ++g) {
        int extended = 0;
        for (const auto& [key, p] : pairs)
            if (key.second == g && p.games == 4) ++extended;
        const bool promoted = std::find(s.references.begin(), s.references.end(), g) != s.references.end();
        assert(extended == (promoted ? 1 : 0));
    }
    assert(s.generation == 2);
    assert(s.bestGeneration == 2);                 // no gating: always the newest
    assert(s.samplesTrained == 2 * 4 * 16);
    assert(s.buffer.games() == 8);                 // at most 400 positions < 1000: nothing evicted
    assert(static_cast<long long>(s.buffer.positions()) == s.buffer.totalPositionsAdded());

    // elo.csv: header plus generations 0, 1, 2. The anchor is exactly 0 with no
    // predecessor; generations 1 and 2 each played g-1, so all five fields are filled.
    const std::vector<std::string> rows = splitLines(readFile(dir.file("elo.csv")));
    assert(rows.size() == 4);
    assert(rows[1] == "0,0.0,,,");
    for (int g = 1; g <= 2; ++g) {
        const std::string& row = rows[static_cast<size_t>(g) + 1];
        assert(row.rfind(std::to_string(g) + ",", 0) == 0);
        assert(std::count(row.begin(), row.end(), ',') == 4);
        assert(row.find(",,") == std::string::npos && row.back() != ',');
    }
}

// Review: resume must continue exactly where an uninterrupted run would be.
static void test_resume_is_byte_identical() {
    TempDir a("resume_a"), b("resume_b");
    const Config c = tinyConfig();
    {
        std::unique_ptr<Trainer> t = startQuiet(c, a);
        t->runUntil(4);
    }
    {
        std::unique_ptr<Trainer> t = startQuiet(c, b);
        t->runUntil(2);
    }   // destroyed: nothing survives in memory
    {
        std::unique_ptr<Trainer> t = resumeQuiet(b, {});
        assert(t->generation() == 2);
        t->runUntil(4);
        assert(t->generation() == 4);
    }
    for (const char* f : {"checkpoints/gen_0004.azn", "state.bin", "elo.csv", "results.csv",
                          "references.csv", "sgf/gen_0004_game_00.sgf"})
        assert(readFile(a.file(f)) == readFile(b.file(f)));
    // The validation loss resumes exactly too, buffer sample included.
    for (int i = 0; i < 4; ++i)
        for (const char* column : {"val_policy_loss", "val_value_loss", "val_ownership_loss",
                                   "seen_policy_loss", "seen_value_loss", "seen_ownership_loss"})
            assert(logRow(a.str(), i).at(column) == logRow(b.str(), i).at(column));
    assert(!logRow(b.str(), 3).at("seen_value_loss").empty());
}

// Workers change who plays which game and how gemm rows are split, never the result.
static void test_thread_count_independence() {
    TempDir one("workers1"), three("workers3");
    Config c = tinyConfig();
    c.workers = 1;
    startQuiet(c, one)->runUntil(2);
    c.workers = 3;
    startQuiet(c, three)->runUntil(2);
    for (const char* f : {"checkpoints/gen_0002.azn", "state.bin", "results.csv", "references.csv"})
        assert(readFile(one.file(f)) == readFile(three.file(f)));
}

static void test_resume_overrides() {
    TempDir dir("overrides");
    startQuiet(tinyConfig(), dir);   // generation 0 on disk; trainer discarded
    const std::string before = readFile(configPath(dir.str()));

    // Architecture keys are refused, naming the key, and nothing is rewritten.
    bool threw = false;
    try {
        Trainer::resume(dir.str(), {"filters=16"});
    } catch (const ConfigError& e) {
        threw = std::string(e.what()).find("filters") != std::string::npos;
    }
    assert(threw);
    assert(readFile(configPath(dir.str())) == before);

    // So is bot_simulations: bots are numbered by their position in it.
    threw = false;
    try {
        Trainer::resume(dir.str(), {"bot_simulations=50"});
    } catch (const ConfigError& e) {
        const std::string msg = e.what();
        threw = msg.find("cannot change") != std::string::npos &&
                msg.find("bot_simulations") != std::string::npos;
    }
    assert(threw);
    assert(readFile(configPath(dir.str())) == before);

    // And reference_networks: reference networks are numbered after the bots.
    std::string msg;
    try {
        Trainer::resume(dir.str(), {"reference_networks=" + checkpointPath(dir.str(), 0)});
    } catch (const ConfigError& e) {
        msg = e.what();
    }
    assert(msg.rfind("cannot change keys on resume: reference_networks (", 0) == 0);
    assert(readFile(configPath(dir.str())) == before);

    // Tunable keys are accepted and persisted.
    std::unique_ptr<Trainer> t = resumeQuiet(dir, {"simulations=12"});
    assert(t->config().simulations == 12);
    assert(loadConfigFile(configPath(dir.str())).simulations == 12);
    assert(readFile(configPath(dir.str())).find("simulations = 12\n") != std::string::npos);
    t->runUntil(1);
    assert(t->generation() == 1);

    // Resuming a directory that holds no run is a ConfigError too.
    TempDir empty("no_run");
    threw = false;
    try {
        Trainer::resume(empty.str(), {});
    } catch (const ConfigError&) {
        threw = true;
    }
    assert(threw);
}

// state.bin records the bot list, so a hand-edited config.cfg cannot relabel
// player -1 or empty the list under a run that already rated its bots.
static void test_resume_refuses_edited_bot_list() {
    for (const std::string edited : {"4", ""}) {
        TempDir dir(edited.empty() ? "editbots_empty" : "editbots_4");
        startQuiet(tinyConfig(), dir);   // bot_simulations = 8
        std::string cfgText = readFile(configPath(dir.str()));
        const size_t at = cfgText.find("bot_simulations = 8");
        assert(at != std::string::npos);
        cfgText.replace(at, std::string("bot_simulations = 8").size(),
                        "bot_simulations = " + edited);
        writeFile(configPath(dir.str()), cfgText);
        const std::string state = readFile(statePath(dir.str()));

        bool threw = false;
        try {
            Trainer::resume(dir.str(), {});
        } catch (const ConfigError& e) {
            const std::string msg = e.what();
            threw = msg.find("cannot change bot_simulations on resume") != std::string::npos &&
                    msg.find("run has 8") != std::string::npos;
        }
        assert(threw);
        assert(readFile(statePath(dir.str())) == state);
        assert(readFile(configPath(dir.str())) == cfgText);
    }
}

static void test_start_new_refuses_existing_run() {
    TempDir dir("existing");
    startQuiet(tinyConfig(), dir);
    bool threw = false;
    try {
        Trainer::startNew(tinyConfig(), dir.str());
    } catch (const ConfigError& e) {
        threw = std::string(e.what()).find("--resume") != std::string::npos;
    }
    assert(threw);
}

// With threshold 1.0 a new network becomes "best" only by winning every game
// against the current best. Replaying that rule over results.csv must land on
// the bestGeneration stored in state.bin.
static void test_gating_state_is_consistent() {
    TempDir dir("gating");
    Config c = tinyConfig();
    c.gating = true;
    c.gatingThreshold = 1.0f;
    c.promoteThreshold = 1.0f;
    startQuiet(c, dir)->runUntil(3);

    std::map<std::pair<int, int>, ResultsTable::Pair> pairs = readResults(dir);
    int best = 0;
    for (int g = 1; g <= 3; ++g) {
        // The current best is always among g's opponents.
        auto it = pairs.find({best, g});
        assert(it != pairs.end());
        // scoreA is the older network's score: 0 means g won every game. More
        // than arena_games games means promotion games were added, which with
        // promote_threshold 1 needs a clean sweep of the arena games, so g
        // passed the gate too.
        if (it->second.scoreA == 0.0 || it->second.games > c.arenaGames) best = g;
    }
    assert(loadTrainingState(statePath(dir.str())).bestGeneration == best);
}

// The progress line names every opponent with the new network's score.
static void test_progress_line_lists_opponents() {
    TempDir dir("progress");
    std::unique_ptr<Trainer> t = startQuiet(tinyConfig(), dir);
    std::ostringstream out;
    t->setLog(&out);
    t->runGeneration();
    const std::string line = out.str();
    assert(line.rfind("gen 1 | ", 0) == 0);
    assert(line.find("| arena elo ") != std::string::npos);
    assert(line.find(" vs 0:") != std::string::npos);      // g-1 first
    assert(line.find(" bot8:") != std::string::npos);      // bots by simulation count
    assert(line.find(" (distinct ") != std::string::npos);
}

// Review Focus 1: a run started before rating-aware opponents. Its state.bin
// is version 1 (no reference set) and its config.cfg has none of the new
// keys, so the defaults apply on resume: one bot at 100 simulations, which
// is calibrated with 80 games at the first arena and then played like any
// reference.
static void test_resume_from_version1_state() {
    TempDir dir("version1");
    Config c = tinyConfig();
    c.botSimulations = {};   // the old run had no bots
    startQuiet(c, dir)->runUntil(1);

    // state.bin as version 1: the buffer without ownership, and without the
    // reference set {0} (u32 count + one i32), the empty bot list (u32 count),
    // the empty reference network list (u32 count) and the resignation
    // threshold and empty window (f32 + u32 count).
    const std::string bytes = readFile(statePath(dir.str()));
    const TrainingState current = loadTrainingState(statePath(dir.str()));
    std::ostringstream with, without;
    BinaryWriter ww(with), wo(without);
    current.buffer.save(ww, true);
    current.buffer.save(wo, false, false);
    const size_t header = 24;   // magic, version, generation, best generation, samples trained
    assert(bytes.compare(header, with.str().size(), with.str()) == 0);
    std::string v1 = bytes.substr(0, header) + without.str() +
                     bytes.substr(header + with.str().size(), bytes.size() - header - with.str().size() - 24);
    v1[4] = 1;
    writeFile(statePath(dir.str()), v1);
    // config.cfg without the keys the old code did not know.
    std::string oldConfig;
    for (const std::string& line : splitLines(readFile(configPath(dir.str()))))
        if (line.rfind("bot_simulations", 0) != 0 && line.rfind("calibration_games", 0) != 0 &&
            line.rfind("promote_threshold", 0) != 0)
            oldConfig += line + "\n";
    writeFile(configPath(dir.str()), oldConfig);

    std::unique_ptr<Trainer> t = resumeQuiet(dir, {});
    assert((t->config().botSimulations == std::vector<int>{100}));
    t->runUntil(2);
    std::map<std::pair<int, int>, ResultsTable::Pair> pairs = readResults(dir);
    assert(pairs.at({-1, 0}).games == 80);   // calibrated once, at generation 2
    assert(pairs.count({-1, 2}));            // and then chosen
    assert(!pairs.count({-1, 1}));           // generation 1 ran before the bot existed
    TrainingState s = loadTrainingState(statePath(dir.str()));
    assert(s.references.front() == -1);
    assert(readFile(statePath(dir.str()))[4] == 5);   // saved as version 5
    assert(readFile(dir.file("references.csv")).find("\n-1,bot,100,0,") != std::string::npos);
}

// Review Focus 4: no bots and no extra opponents. Each generation plays only
// g-1 and the strongest reference (the same player while generation 0 is the
// whole reference set), and nothing negative appears anywhere.
static void test_no_bots_no_extras() {
    TempDir dir("nobots");
    Config c = tinyConfig();
    c.botSimulations = {};
    c.arenaOpponents = 0;
    startQuiet(c, dir)->runUntil(3);
    std::map<std::pair<int, int>, ResultsTable::Pair> pairs = readResults(dir);
    for (const auto& [key, p] : pairs) assert(key.first >= 0);
    for (int g = 1; g <= 3; ++g) {
        assert(pairs.count({g - 1, g}));
        int opponents = 0;
        for (const auto& [key, p] : pairs)
            if (key.second == g) ++opponents;
        assert(opponents >= 1 && opponents <= 2);
    }
    assert(pairs.size() >= 3 && pairs.at({0, 1}).games >= 2);
    const std::vector<std::string> refs = splitLines(readFile(dir.file("references.csv")));
    assert(refs[1] == "0,checkpoint,,0,0.0");
}

// --rerate replays the arena over the checkpoints. With gating off the arena
// never feeds back into training, so a re-rate of a finished run must play
// exactly the games the live run played: the same results.csv, byte for byte.
static void test_rerate_reproduces_live_results() {
    TempDir dir("rerate");
    startQuiet(tinyConfig(), dir)->runUntil(3);
    const std::string state = readFile(statePath(dir.str()));
    const std::string config = readFile(configPath(dir.str()));

    assert(rerateRun(dir.str(), {}, nullptr) == 3);
    for (const char* f : {"results.csv", "elo.csv", "references.csv"})
        assert(readFile(dir.file(std::string("rerate/") + f)) == readFile(dir.file(f)));
    // It trains nothing and changes nothing outside rerate/.
    assert(readFile(statePath(dir.str())) == state);
    assert(readFile(configPath(dir.str())) == config);
    assert(!fs::exists(checkpointPath(dir.str(), 4)));

    // log.csv: one row per generation; the opponents start with g-1.
    const std::vector<std::string> rows = splitLines(readFile(dir.file("rerate/log.csv")));
    assert(rows.size() == 4);
    assert(rows[0] == "generation,opponents,score_vs_prev,score_vs_strongest,promoted,elo");
    for (int g = 1; g <= 3; ++g) {
        const std::string& row = rows[static_cast<size_t>(g)];
        assert(row.rfind(std::to_string(g) + "," + std::to_string(g - 1), 0) == 0);
        assert(std::count(row.begin(), row.end(), ',') == 5);
    }

    // Independent of the worker count, log.csv included (it has no timings).
    std::map<std::string, std::string> one;
    rerateRun(dir.str(), {"workers=1"}, nullptr);
    for (const char* f : {"results.csv", "elo.csv", "references.csv", "log.csv"})
        one[f] = readFile(dir.file(std::string("rerate/") + f));
    rerateRun(dir.str(), {"workers=3"}, nullptr);
    for (const char* f : {"results.csv", "elo.csv", "references.csv", "log.csv"})
        assert(readFile(dir.file(std::string("rerate/") + f)) == one[f]);
}

// Review Focus 5: re-rating something that is not a complete run.
static void test_rerate_refusals_and_gaps() {
    TempDir empty("rerate_empty");
    fs::create_directories(empty.path);
    bool threw = false;
    try {
        rerateRun(empty.str(), {}, nullptr);
    } catch (const ConfigError& e) {
        threw = std::string(e.what()).find(empty.str()) != std::string::npos;
    }
    assert(threw);

    TempDir dir("rerate_gap");
    startQuiet(tinyConfig(), dir)->runUntil(3);
    threw = false;
    try {
        rerateRun(dir.str(), {"filters=16"}, nullptr);   // would not match the checkpoints
    } catch (const ConfigError& e) {
        threw = std::string(e.what()).find("filters") != std::string::npos;
    }
    assert(threw);
    assert(!fs::exists(dir.path / "rerate"));

    // An unavailable backend fails before rerate/ is created. Only the
    // default (naive-only) build can show it.
    if (!blasAvailable()) {
        threw = false;
        try {
            rerateRun(dir.str(), {"backend=blas"}, nullptr);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
        assert(!fs::exists(dir.path / "rerate"));
    }

    // Every generation is measured against g-1, so a missing checkpoint ends
    // the replay: with gen_0002 gone only generation 1 is re-rated.
    fs::remove(checkpointPath(dir.str(), 2));
    assert(rerateRun(dir.str(), {}, nullptr) == 1);
    assert(countLines(readFile(dir.file("rerate/log.csv"))) == 2);
}

// A blown-up learning rate must abort the generation before anything of it is
// committed: state.bin stays at generation 0 and no gen_0001 checkpoint exists.
static void test_non_finite_weights_keep_last_good_state() {
    TempDir dir("nonfinite");
    Config c = tinyConfig();
    c.lrSchedule = "0:3e38";
    c.trainSteps = 1;   // only the post-loop check can see the last step's weights
    std::unique_ptr<Trainer> t = startQuiet(c, dir);
    bool threw = false;
    try {
        t->runGeneration();
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("last good state kept") != std::string::npos;
    }
    assert(threw);
    assert(loadTrainingState(statePath(dir.str())).generation == 0);
    assert(!fs::exists(checkpointPath(dir.str(), 1)));
}

// ---- starting from a given network and a donor buffer ----------------------

static bool sameWeights(Network& a, Network& b) {
    std::vector<ParamView> pa = a.params(), pb = b.params();
    if (pa.size() != pb.size()) return false;
    for (size_t i = 0; i < pa.size(); ++i)
        if (pa[i].size != pb[i].size ||
            std::memcmp(pa[i].value, pb[i].value, pa[i].size * sizeof(float)) != 0)
            return false;
    return true;
}

static void test_start_from_network_and_buffer() {
    TempDir donor("donor"), fresh("init_from"), small("init_small");
    {
        std::unique_ptr<Trainer> t = startQuiet(tinyConfig(), donor);
        t->runUntil(1);
    }
    const std::string donorCkpt = checkpointPath(donor.str(), 1);
    const TrainingState donorState = loadTrainingState(statePath(donor.str()), {8});
    assert(donorState.buffer.positions() > 0);

    // Network only: gen_0000 is the donor's weights, the buffer is empty.
    Config c = tinyConfig();
    c.seed = 5;   // a different seed would otherwise give a different random gen 0
    Trainer::RunInit init;
    init.initFrom = donorCkpt;
    {
        std::unique_ptr<Trainer> t = Trainer::startNew(c, fresh.str(), init);
        t->setLog(nullptr);
        assert(t->generation() == 0);
    }
    Network src = loadNetwork(donorCkpt), gen0 = loadNetwork(checkpointPath(fresh.str(), 0));
    assert(sameWeights(src, gen0));
    TrainingState s = loadTrainingState(statePath(fresh.str()), {8});
    assert(s.buffer.positions() == 0);
    const std::string initTxt = readFile(fresh.file("init.txt"));
    assert(initTxt.find(donorCkpt) != std::string::npos);
    assert(initTxt.find("none") != std::string::npos);

    // Network and buffer: the donor's games, up to the new capacity.
    TempDir both("init_both");
    init.bufferFrom = donor.str();
    {
        std::unique_ptr<Trainer> t = Trainer::startNew(c, both.str(), init);
        t->setLog(nullptr);
    }
    TrainingState b = loadTrainingState(statePath(both.str()), {8});
    assert(b.generation == 0);
    assert(b.buffer.positions() == donorState.buffer.positions());
    assert(b.buffer.games() == donorState.buffer.games());
    assert(b.buffer.game(0).moves == donorState.buffer.game(0).moves);
    for (const std::vector<float>& v : b.velocity)
        for (float x : v) assert(x == 0.0f);
    const std::string bothTxt = readFile(both.file("init.txt"));
    assert(bothTxt.find(donor.str()) != std::string::npos);
    assert(bothTxt.find("positions") != std::string::npos);

    // A smaller capacity evicts the oldest games, as setCapacity does.
    Config smallCfg = c;
    smallCfg.bufferPositions = static_cast<int>(donorState.buffer.positions() / 2 + 1);
    {
        std::unique_ptr<Trainer> t = Trainer::startNew(smallCfg, small.str(), init);
        t->setLog(nullptr);
    }
    TrainingState sm = loadTrainingState(statePath(small.str()), {8});
    assert(sm.buffer.capacity() == static_cast<size_t>(smallCfg.bufferPositions));
    assert(sm.buffer.positions() < donorState.buffer.positions());

    // The run is an ordinary one: it trains, and resumes from where it stopped.
    {
        std::unique_ptr<Trainer> t = resumeQuiet(both, {});
        assert(t->generation() == 0);
        t->runUntil(1);
    }
    std::unique_ptr<Trainer> t = resumeQuiet(both, {});
    assert(t->generation() == 1);
    t->runUntil(2);
    assert(t->generation() == 2);
    assert(fs::exists(checkpointPath(both.str(), 2)));
}

static void test_start_from_refusals() {
    TempDir donor("donor_refuse"), target("target_refuse");
    startQuiet(tinyConfig(), donor);

    // A network of another shape names the shape and leaves no run behind.
    Config c = tinyConfig();
    c.blocks = 2;
    Trainer::RunInit init;
    init.initFrom = checkpointPath(donor.str(), 0);
    bool threw = false;
    try {
        Trainer::startNew(c, target.str(), init);
    } catch (const ConfigError& e) {
        const std::string msg = e.what();
        threw = msg.find("shape") != std::string::npos && msg.find("blocks") != std::string::npos;
    }
    assert(threw);
    assert(!fs::exists(statePath(target.str())));

    // A donor buffer from another board or history is refused.
    c = tinyConfig();
    c.history = 3;
    init.initFrom.clear();
    init.bufferFrom = donor.str();
    threw = false;
    try {
        Trainer::startNew(c, target.str(), init);
    } catch (const ConfigError& e) {
        threw = std::string(e.what()).find("history") != std::string::npos;
    }
    assert(threw);
    assert(!fs::exists(statePath(target.str())));

    // A missing checkpoint is a plain error, again with no run left behind.
    init = {};
    init.initFrom = target.file("nope.azn");
    threw = false;
    try {
        Trainer::startNew(tinyConfig(), target.str(), init);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    assert(!fs::exists(statePath(target.str())));
}

// ---- the ownership head ------------------------------------------------------

static Config ownershipConfig() {
    Config c = tinyConfig();
    c.ownership = true;
    return c;
}

// A run with the head trains it: the loss is logged, the progress line shows
// it, and the head's weights move.
static void test_ownership_run_trains_the_head() {
    TempDir dir("ownership");
    std::unique_ptr<Trainer> t = startQuiet(ownershipConfig(), dir);
    assert(readCheckpointConfig(checkpointPath(dir.str(), 0)).ownership);
    std::ostringstream out;
    t->setLog(&out);
    const GenerationStats st = t->runGeneration();
    // A mean squared error of values in [-1, 1].
    assert(std::isfinite(st.ownershipLoss) && st.ownershipLoss > 0 && st.ownershipLoss < 4);
    // ownership_loss, and the three resignation columns empty.
    const std::map<std::string, std::string> row = logRow(dir.str());
    assert(row.at("ownership_loss") == loss5(st.ownershipLoss));
    for (const char* empty : {"resign_threshold", "resign_rate", "resign_false_positive"})
        assert(row.at(empty).empty());
    assert(out.str().find(" v ") != std::string::npos && out.str().find(" o ") != std::string::npos);

    Network gen0 = loadNetwork(checkpointPath(dir.str(), 0));
    Network gen1 = loadNetwork(checkpointPath(dir.str(), 1));
    std::vector<ParamView> p0 = gen0.params(), p1 = gen1.params();
    assert(p1.back().name == "ownership.bn.beta" && p1[p1.size() - 3].name == "ownership.conv.w");
    const ParamView& w0 = p0[p0.size() - 3];
    const ParamView& w1 = p1[p1.size() - 3];
    assert(std::memcmp(w0.value, w1.value, w0.size * sizeof(float)) != 0);
}

// Determinism holds with the head: resume continues byte-identically and the
// worker count changes nothing.
static void test_ownership_determinism() {
    TempDir a("own_resume_a"), b("own_resume_b");
    const Config c = ownershipConfig();
    startQuiet(c, a)->runUntil(3);
    startQuiet(c, b)->runUntil(1);
    resumeQuiet(b, {})->runUntil(3);
    for (const char* f : {"checkpoints/gen_0003.azn", "state.bin", "elo.csv", "results.csv",
                          "references.csv", "sgf/gen_0003_game_00.sgf"})
        assert(readFile(a.file(f)) == readFile(b.file(f)));

    TempDir one("own_workers1"), three("own_workers3");
    Config w = c;
    w.workers = 1;
    startQuiet(w, one)->runUntil(2);
    w.workers = 3;
    startQuiet(w, three)->runUntil(2);
    for (const char* f : {"checkpoints/gen_0002.azn", "state.bin", "results.csv", "references.csv"})
        assert(readFile(one.file(f)) == readFile(three.file(f)));
}

// --init-from with a checkpoint whose head does not match the config says how
// to fix it; resume refuses to switch the head on or off.
static void test_ownership_mismatches_are_refused() {
    TempDir plain("own_plain"), owner("own_owner"), target("own_target");
    startQuiet(tinyConfig(), plain);
    startQuiet(ownershipConfig(), owner);

    Trainer::RunInit init;
    init.initFrom = checkpointPath(plain.str(), 0);
    std::string msg;
    try {
        Trainer::startNew(ownershipConfig(), target.str(), init);
    } catch (const ConfigError& e) {
        msg = e.what();
    }
    assert(msg.find("has no ownership head") != std::string::npos);
    assert(msg.find("grow --from " + init.initFrom + " --ownership") != std::string::npos);
    assert(!fs::exists(statePath(target.str())));

    init.initFrom = checkpointPath(owner.str(), 0);
    msg.clear();
    try {
        Trainer::startNew(tinyConfig(), target.str(), init);
    } catch (const ConfigError& e) {
        msg = e.what();
    }
    assert(msg.find("has an ownership head") != std::string::npos);
    assert(msg.find("--set ownership=true") != std::string::npos);
    assert(!fs::exists(statePath(target.str())));

    // Another shape difference as well: the shape message, naming the head.
    Config wider = ownershipConfig();
    wider.filters = 16;
    msg.clear();
    try {
        Trainer::startNew(wider, target.str(), init);
    } catch (const ConfigError& e) {
        msg = e.what();
    }
    assert(msg.find("8 filters, value_hidden 8, ownership head but the config asks for") !=
           std::string::npos);

    // A matching head is accepted: generation 0 is that checkpoint.
    {
        std::unique_ptr<Trainer> t = Trainer::startNew(ownershipConfig(), target.str(), init);
        t->setLog(nullptr);
    }
    assert(readFile(checkpointPath(target.str(), 0)) == readFile(init.initFrom));

    for (const char* change : {"ownership=true", "ownership=false"}) {
        const TempDir& dir = std::string(change) == "ownership=true" ? plain : owner;
        msg.clear();
        try {
            Trainer::resume(dir.str(), {change});
        } catch (const ConfigError& e) {
            msg = e.what();
        }
        assert(msg == "cannot change architecture keys on resume: ownership");
    }
    // ownership_weight is tunable.
    assert(resumeQuiet(owner, {"ownership_weight=0.5"})->config().ownershipWeight == 0.5f);
}

// A run made before this change: state.bin version 2 (no ownership in the
// buffer), a config.cfg without the ownership keys and a log.csv without the
// ownership_loss column. Resuming it recomputes the buffer's ownership, adds
// the column to every old row, and continues as a run without the head.
static void test_resume_run_from_before_ownership() {
    TempDir dir("pre_ownership");
    startQuiet(tinyConfig(), dir)->runUntil(1);

    // state.bin version 2: the buffer without ownership, no reference
    // network list (an empty one is a u32 count) and no resignation state
    // at the end (an f32 threshold and an empty window's u32 count).
    const std::string bytes = readFile(statePath(dir.str()));
    const TrainingState current = loadTrainingState(statePath(dir.str()));
    std::ostringstream with, without;
    BinaryWriter ww(with), wo(without);
    current.buffer.save(ww, true);
    current.buffer.save(wo, false, false);
    const size_t header = 24;
    std::string v2 = bytes.substr(0, header) + without.str() +
                     bytes.substr(header + with.str().size(), bytes.size() - header - with.str().size() - 12);
    v2[4] = 2;
    writeFile(statePath(dir.str()), v2);
    // config.cfg and log.csv as the old code wrote them.
    std::string oldConfig;
    for (const std::string& line : splitLines(readFile(configPath(dir.str()))))
        if (line.rfind("ownership", 0) != 0 && line.rfind("validation_loss", 0) != 0)
            oldConfig += line + "\n";
    writeFile(configPath(dir.str()), oldConfig);
    // Without the ownership_loss column and every later one.
    const std::string oldLog = firstLogColumns(dir.str(), 15);
    assert(oldLog.rfind("generation,", 0) == 0 && oldLog.find(",distinct_vs_prev\n") != std::string::npos);
    assert(std::count(oldLog.begin(), oldLog.end(), ',') == 2 * 14);
    writeFile(dir.file("log.csv"), oldLog);

    std::unique_ptr<Trainer> t = resumeQuiet(dir, {});
    assert(!t->config().ownership);
    const TrainingState resumed = loadTrainingState(statePath(dir.str()));
    for (size_t i = 0; i < current.buffer.games(); ++i)
        assert(resumed.buffer.game(i).finalOwnership == current.buffer.game(i).finalOwnership);
    t->runUntil(2);
    const std::vector<std::string> rows = splitLines(readFile(dir.file("log.csv")));
    assert(rows.size() == 3 && rows[0].find(",distinct_vs_prev,ownership_loss,") != std::string::npos);
    for (const std::string& row : rows) assert(std::count(row.begin(), row.end(), ',') == kLogCommas);
    // The old row's later columns are empty; the new row measures the
    // validation loss (validation_loss defaults to on).
    const std::map<std::string, std::string> old = logRow(dir.str(), 0), fresh = logRow(dir.str(), 1);
    for (const char* column : {"ownership_loss", "resign_threshold", "val_policy_loss", "seen_value_loss"})
        assert(old.at(column).empty());
    assert(!fresh.at("val_policy_loss").empty() && !fresh.at("seen_value_loss").empty());
    assert(readFile(statePath(dir.str()))[4] == 5);
}

// ---- reference networks ------------------------------------------------------

// tinyConfig's generation 1, copied out of a donor run into `dir` as ref.azn.
// The arena never feeds back into training (no gating), so it is also
// generation 1 of any run with tinyConfig's seed and shape: a run's own
// early checkpoint, frozen as its reference.
static std::string frozenGeneration1(const TempDir& dir, const Config& c = tinyConfig()) {
    TempDir donor(dir.path.filename().string() + "_donor");
    startQuiet(c, donor)->runUntil(1);
    fs::create_directories(dir.path);
    const std::string ref = dir.file("ref.azn");
    fs::copy_file(checkpointPath(donor.str(), 1), ref);
    return ref;
}

static std::string configErrorOf(const std::function<void()>& f) {
    try {
        f();
    } catch (const ConfigError& e) {
        return e.what();
    }
    return "";
}

// Reference networks come after the bots: with one bot the reference network
// is player -2 (ref1); without bots it is -1. Each is calibrated against
// generation 0 at the start, then chosen like any reference.
static void test_reference_network_ids_and_files() {
    TempDir refs("refnet_files"), dir("refnet_run"), nobots("refnet_nobots");
    Config c = tinyConfig();
    c.referenceNetworks = {frozenGeneration1(refs)};
    {
        std::unique_ptr<Trainer> t = startQuiet(c, dir);
        std::ostringstream out;
        t->setLog(&out);
        t->runGeneration();
        // arena_opponents = 2 leaves room for every reference.
        assert(out.str().find(" ref1:") != std::string::npos && out.str().find(" bot8:") != std::string::npos);
    }
    TrainingState s = loadTrainingState(statePath(dir.str()));
    assert(s.referenceNetworks.size() == 1);
    assert(s.referenceNetworks[0].path == c.referenceNetworks[0]);
    assert(s.referenceNetworks[0].hash == fileHash(c.referenceNetworks[0]));
    std::map<std::pair<int, int>, ResultsTable::Pair> pairs = readResults(dir);
    assert(pairs.at({-2, 0}).games == 2 && pairs.at({-1, 0}).games == 2);   // calibration
    assert(pairs.count({-2, 1}));                                              // and chosen
    const std::vector<std::string> refRows = splitLines(readFile(dir.file("references.csv")));
    assert(refRows[0] == "player,kind,simulations,promoted_at,elo");
    assert(refRows[1].rfind("-2,network,,0,", 0) == 0 && refRows[1].size() > 14);   // rated
    assert(refRows[2].rfind("-1,bot,8,0,", 0) == 0);
    assert(readFile(dir.file("reference-networks.csv")) ==
           "player,name,path\n-2,ref1," + c.referenceNetworks[0] + "\n");

    // Without bots the reference network is player -1.
    c.botSimulations = {};
    startQuiet(c, nobots)->runUntil(1);
    assert((loadTrainingState(statePath(nobots.str())).references.front() == -1));
    assert(readFile(nobots.file("reference-networks.csv")) ==
           "player,name,path\n-1,ref1," + c.referenceNetworks[0] + "\n");
    assert(readResults(nobots).at({-1, 0}).games == 2);
    assert(readFile(nobots.file("references.csv")).find("\n-1,network,,0,") != std::string::npos);
    // A run without reference networks writes no reference-networks.csv.
    TempDir plain("refnet_plain");
    startQuiet(tinyConfig(), plain);
    assert(!fs::exists(plain.file("reference-networks.csv")));
}

// A smaller network is fitted to the run's shape (wider, deeper, with the
// head) and plays; a bigger one, a missing file or a corrupt one is refused
// naming the file, before anything is written.
static void test_reference_network_shapes_and_refusals() {
    TempDir refs("refnet_shapes"), big("refnet_big"), target("refnet_target");
    Config c = ownershipConfig();
    c.blocks = 2;
    c.filters = 12;
    c.referenceNetworks = {frozenGeneration1(refs)};   // 1 block, 8 filters, no head
    {
        std::unique_ptr<Trainer> t = Trainer::startNew(c, target.str());
        t->setLog(nullptr);
        t->runUntil(1);
    }
    assert(readResults(target).at({-2, 0}).games == 2);
    fs::remove_all(target.path);

    Config wide = tinyConfig();
    wide.filters = 16;
    startQuiet(wide, big)->runUntil(1);
    Config small = tinyConfig();
    small.referenceNetworks = {checkpointPath(big.str(), 1)};
    std::string msg = configErrorOf([&] { Trainer::startNew(small, target.str()); });
    assert(msg.find("reference_networks: " + small.referenceNetworks[0]) != std::string::npos);
    assert(msg.find("16 filters is more than 8") != std::string::npos);
    assert(!fs::exists(target.path / "config.cfg") && !fs::exists(statePath(target.str())));

    small.referenceNetworks = {refs.file("missing.azn")};
    msg = configErrorOf([&] { Trainer::startNew(small, target.str()); });
    assert(msg.find(refs.file("missing.azn")) != std::string::npos);
    assert(!fs::exists(statePath(target.str())));

    // The same network twice (identical bytes under two names) would be two
    // players with one strength: refused, naming both files.
    fs::copy_file(refs.file("ref.azn"), refs.file("copy.azn"));
    small.referenceNetworks = {refs.file("ref.azn"), refs.file("copy.azn")};
    msg = configErrorOf([&] { Trainer::startNew(small, target.str()); });
    assert(msg.find(refs.file("ref.azn")) != std::string::npos &&
           msg.find(refs.file("copy.azn")) != std::string::npos && msg.find("identical") != std::string::npos);
    assert(!fs::exists(statePath(target.str())));

    writeFile(refs.file("corrupt.azn"), "AZGO not really a checkpoint");
    small.referenceNetworks = {refs.file("corrupt.azn")};
    msg = configErrorOf([&] { Trainer::startNew(small, target.str()); });
    assert(msg.find(refs.file("corrupt.azn")) != std::string::npos);
    assert(!fs::exists(statePath(target.str())));
}

// The list is fixed for a run (state.bin records it, so a hand edit of
// config.cfg is caught too), and so is each file's content.
static void test_resume_refuses_changed_reference_networks() {
    TempDir refs("refnet_resume_files"), dir("refnet_resume");
    Config c = tinyConfig();
    c.referenceNetworks = {frozenGeneration1(refs)};
    startQuiet(c, dir)->runUntil(1);
    const std::string state = readFile(statePath(dir.str()));

    std::string cfgText = readFile(configPath(dir.str()));
    const std::string line = "reference_networks = " + c.referenceNetworks[0];
    const size_t at = cfgText.find(line);
    assert(at != std::string::npos);
    std::string edited = cfgText;
    edited.replace(at, line.size(), "reference_networks =");
    writeFile(configPath(dir.str()), edited);
    std::string msg = configErrorOf([&] { Trainer::resume(dir.str(), {}); });
    assert(msg.find("cannot change reference_networks on resume: run has " + c.referenceNetworks[0] +
                    " but config has (none)") != std::string::npos);
    assert(readFile(configPath(dir.str())) == edited);
    writeFile(configPath(dir.str()), cfgText);

    // Another network under the same name: refused, naming the file.
    const std::string original = readFile(c.referenceNetworks[0]);
    fs::copy_file(checkpointPath(dir.str(), 0), c.referenceNetworks[0], fs::copy_options::overwrite_existing);
    msg = configErrorOf([&] { Trainer::resume(dir.str(), {}); });
    assert(msg.find(c.referenceNetworks[0]) != std::string::npos && msg.find("changed") != std::string::npos);
    assert(readFile(statePath(dir.str())) == state);
    // A missing file is refused too.
    fs::remove(c.referenceNetworks[0]);
    msg = configErrorOf([&] { Trainer::resume(dir.str(), {}); });
    assert(msg.find(c.referenceNetworks[0]) != std::string::npos);
    // The original file resumes.
    writeFile(c.referenceNetworks[0], original);
    resumeQuiet(dir, {})->runUntil(2);
}

// Determinism holds with a reference network: the worker count changes
// nothing, and resume continues byte-identically. The reference is the run's
// own generation 1.
static void test_reference_network_determinism() {
    TempDir refs("refnet_det_files"), a("refnet_det_a"), b("refnet_det_b");
    Config c = tinyConfig();
    c.referenceNetworks = {frozenGeneration1(refs)};
    startQuiet(c, a)->runUntil(3);
    assert(readFile(checkpointPath(a.str(), 1)) == readFile(c.referenceNetworks[0]));
    startQuiet(c, b)->runUntil(1);
    resumeQuiet(b, {})->runUntil(3);
    for (const char* f : {"checkpoints/gen_0003.azn", "state.bin", "elo.csv", "results.csv",
                          "references.csv", "reference-networks.csv", "sgf/gen_0003_game_00.sgf"})
        assert(readFile(a.file(f)) == readFile(b.file(f)));

    TempDir one("refnet_workers1"), three("refnet_workers3");
    Config w = c;
    w.workers = 1;
    startQuiet(w, one)->runUntil(2);
    w.workers = 3;
    startQuiet(w, three)->runUntil(2);
    for (const char* f : {"checkpoints/gen_0002.azn", "state.bin", "results.csv", "references.csv"})
        assert(readFile(one.file(f)) == readFile(three.file(f)));
}

// --rerate honours reference_networks: a re-rate of a run with a reference
// network plays exactly the live games, and a re-rate that adds one (and
// drops the bot) plays exactly what a live run with those settings played.
static void test_rerate_with_reference_networks() {
    TempDir refs("refnet_rerate_files"), live("refnet_rerate_live"), plain("refnet_rerate_plain");
    Config c = tinyConfig();
    c.botSimulations = {};
    c.referenceNetworks = {frozenGeneration1(refs)};
    startQuiet(c, live)->runUntil(3);
    std::ostringstream out;
    assert(rerateRun(live.str(), {}, &out) == 3);
    for (const char* f : {"results.csv", "elo.csv", "references.csv", "reference-networks.csv"})
        assert(readFile(live.file(std::string("rerate/") + f)) == readFile(live.file(f)));
    // The calibration is reported, then each generation names ref1 when it plays it.
    assert(out.str().rfind("rerate calibration | ref1 vs 0: ", 0) == 0);

    // A run with the bot and no reference network, re-rated with the live
    // run's fixed players. Its list differs from the replay's, so state.bin
    // is not needed, and a damaged one does not block the replay.
    startQuiet(tinyConfig(), plain)->runUntil(3);
    writeFile(statePath(plain.str()), "not a training state");
    assert(rerateRun(plain.str(), {"bot_simulations=", "reference_networks=" + c.referenceNetworks[0]},
                     nullptr) == 3);
    for (const char* f : {"results.csv", "elo.csv", "references.csv", "reference-networks.csv"})
        assert(readFile(plain.file(std::string("rerate/") + f)) == readFile(live.file(f)));
    assert(loadConfigFile(configPath(plain.str())).referenceNetworks.empty());   // config.cfg untouched

    // A replaced reference file would not reproduce the live run: refused.
    fs::copy_file(checkpointPath(live.str(), 0), c.referenceNetworks[0], fs::copy_options::overwrite_existing);
    const std::string msg = configErrorOf([&] { rerateRun(live.str(), {}, nullptr); });
    assert(msg.find(c.referenceNetworks[0]) != std::string::npos && msg.find("changed") != std::string::npos);
}

// mcts_random_symmetry reaches self-play and arena_random_symmetry reaches
// the arena (live rounds and rerate share one match function), each draw
// from a per-game Rng, so a run with both on is still independent of the
// worker count. Each comparison with the plain run is fixed by the seed.
static void test_random_symmetry_run() {
    TempDir plain("sym_off"), selfPlayOnly("sym_selfplay"), one("sym_workers1"), three("sym_workers3");
    Config c = tinyConfig();
    c.arenaGames = 8;
    startQuiet(c, plain)->runUntil(2);
    // mcts_random_symmetry alone (arena key off): other self-play games, so
    // another generation 1. A self-play wired to the arena key would not differ.
    c.mctsRandomSymmetry = true;
    startQuiet(c, selfPlayOnly)->runUntil(1);
    assert(readFile(selfPlayOnly.file("checkpoints/gen_0001.azn")) !=
           readFile(plain.file("checkpoints/gen_0001.azn")));
    c.arenaRandomSymmetry = true;
    c.workers = 1;
    startQuiet(c, one)->runUntil(2);
    c.workers = 3;
    startQuiet(c, three)->runUntil(2);
    for (const char* f : {"checkpoints/gen_0002.azn", "state.bin", "results.csv", "references.csv"})
        assert(readFile(one.file(f)) == readFile(three.file(f)));

    // arena_random_symmetry alone: the same checkpoints play other games.
    rerateRun(plain.str(), {"arena_random_symmetry=on"}, nullptr);
    assert(readFile(plain.file("rerate/results.csv")) != readFile(plain.file("results.csv")));

    // Both are tunable: a resume may turn them on, and config.cfg keeps them.
    std::unique_ptr<Trainer> t = resumeQuiet(plain, {"mcts_random_symmetry=on", "arena_random_symmetry=on"});
    assert(t->config().mctsRandomSymmetry && t->config().arenaRandomSymmetry);
    const Config saved = loadConfigFile(configPath(plain.str()));
    assert(saved.mctsRandomSymmetry && saved.arenaRandomSymmetry);
    t->runUntil(3);
    assert(t->generation() == 3);
}

// ---- resignation --------------------------------------------------------------

// Aggressive enough that an untrained 5x5 network resigns: v_resign starts
// at 0 (resign as soon as the search thinks it is behind), half the games
// play on to measure false positives, and calibration starts after 2 of them.
static Config resignConfig() {
    Config c = tinyConfig();
    c.resignation = true;
    c.resignThresholdInitial = 0.0f;
    c.resignDisabledFraction = 0.5f;
    c.resignWindow = 6;
    c.resignMinGames = 2;
    c.gamesPerGeneration = 8;
    c.sgfPerGeneration = 8;
    return c;
}

static ResignCalibration calibrationOf(const Config& c) {
    ResignCalibration cal;
    cal.window = c.resignWindow;
    cal.minGames = c.resignMinGames;
    cal.maxFalsePositive = c.resignFalsePositive;
    cal.initialThreshold = c.resignThresholdInitial;
    return cal;
}

// Generation g resigns at the threshold calibrated from generations before
// it; its disabled games join the window afterwards. Resigned games say so
// in their SGF, and log.csv and the progress line report the threshold,
// the resignation rate and the measured false-positive rate.
static void test_resignation_run() {
    TempDir dir("resign");
    const Config c = resignConfig();
    std::unique_ptr<Trainer> t = startQuiet(c, dir);
    TrainingState s0 = loadTrainingState(statePath(dir.str()));
    assert(s0.resignWindow.empty() && s0.resignThreshold == 0.0f);   // the initial threshold

    std::ostringstream out;
    t->setLog(&out);
    int resignedSgf = 0;
    bool calibrated = false;   // some generation resigned at a threshold the data chose
    float expectedThreshold = 0.0f;
    std::vector<ResignTrace> window;
    for (int g = 1; g <= 3; ++g) {
        const GenerationStats st = t->runGeneration();
        assert(st.resignation && st.resignThreshold == expectedThreshold);
        if (st.resignThreshold != c.resignThresholdInitial) calibrated = true;
        const TrainingState s = loadTrainingState(statePath(dir.str()));
        // The window gained this generation's disabled games, at most 6 kept.
        assert(s.resignWindow.size() >= window.size() && s.resignWindow.size() <= 6u);
        window = s.resignWindow;
        assert(window.empty() ? st.resignFalsePositive == -1
                              : st.resignFalsePositive ==
                                    resignFalsePositiveRate(window, c.resignWindow, st.resignThreshold));
        assert(st.resignRate >= 0 && st.resignRate <= 1);
        // The next generation's threshold is calibrated now, from this window.
        expectedThreshold = calibrateResignThreshold(window, calibrationOf(c));
        assert(s.resignThreshold == expectedThreshold);

        const std::map<std::string, std::string> row = logRow(dir.str());
        char threshold[16], rate[16], falsePositive[16] = "";
        std::snprintf(threshold, sizeof threshold, "%.2f", st.resignThreshold);
        std::snprintf(rate, sizeof rate, "%.4f", st.resignRate);
        if (st.resignFalsePositive >= 0)
            std::snprintf(falsePositive, sizeof falsePositive, "%.4f", st.resignFalsePositive);
        assert(row.at("resign_threshold") == threshold && row.at("resign_rate") == rate &&
               row.at("resign_false_positive") == falsePositive);

        int resigned = 0;
        for (int i = 0; i < 8; ++i) {
            char name[64];
            std::snprintf(name, sizeof name, "sgf/gen_%04d_game_%02d.sgf", g, i);
            const std::string sgf = readFile(dir.file(name));
            if (sgf.find("RE[B+R]") != std::string::npos || sgf.find("RE[W+R]") != std::string::npos) ++resigned;
        }
        resignedSgf += resigned;
        // rate = resigned / resign-enabled games, and only enabled games resign.
        assert(resigned == 0 || st.resignRate > 0);
    }
    assert(resignedSgf > 0);       // the untrained network gives up somewhere
    // The buffer (every game: nothing evicted yet) knows which games resigned.
    const TrainingState last = loadTrainingState(statePath(dir.str()));
    assert(last.buffer.games() == 24u);
    int resignedInBuffer = 0;
    for (size_t i = 0; i < last.buffer.games(); ++i) resignedInBuffer += last.buffer.game(i).resigned ? 1 : 0;
    assert(resignedInBuffer == resignedSgf);
    assert(!window.empty());       // and some games were played out to measure
    assert(calibrated);            // which moved v_resign off its initial value
    assert(splitLines(readFile(dir.file("log.csv")))[0] ==
           "generation,games,avg_length,pass_rate,black_win_rate,policy_loss,value_loss,"
           "lr,replay_ratio,selfplay_s,train_s,arena_s,elo,winrate_vs_prev,distinct_vs_prev,"
           "ownership_loss,resign_threshold,resign_rate,resign_false_positive,"
           "val_policy_loss,val_value_loss,val_ownership_loss,"
           "seen_policy_loss,seen_value_loss,seen_ownership_loss");
    assert(out.str().find("gen 1 | selfplay 8 games avg ") != std::string::npos);
    assert(out.str().find(" resign 0.00 rate ") != std::string::npos);
}

// Resignation keeps a run reproducible: a resumed run continues byte for
// byte, and the worker count changes nothing, because each generation's
// threshold depends only on earlier generations' games.
static void test_resignation_determinism() {
    TempDir a("resign_a"), b("resign_b");
    const Config c = resignConfig();
    startQuiet(c, a)->runUntil(3);
    startQuiet(c, b)->runUntil(1);
    resumeQuiet(b, {})->runUntil(3);
    for (const char* f : {"checkpoints/gen_0003.azn", "state.bin", "elo.csv", "results.csv",
                          "references.csv", "sgf/gen_0003_game_00.sgf", "sgf/gen_0003_game_07.sgf"})
        assert(readFile(a.file(f)) == readFile(b.file(f)));

    TempDir one("resign_workers1"), three("resign_workers3");
    Config w = c;
    w.workers = 1;
    startQuiet(w, one)->runUntil(2);
    w.workers = 3;
    startQuiet(w, three)->runUntil(2);
    for (const char* f : {"checkpoints/gen_0002.azn", "state.bin", "results.csv", "references.csv"})
        assert(readFile(one.file(f)) == readFile(three.file(f)));

    // Resignation changes the games: the same run without it trains another network.
    TempDir off("resign_off");
    Config plain = c;
    plain.resignation = false;
    startQuiet(plain, off)->runUntil(1);
    assert(readFile(off.file("checkpoints/gen_0001.azn")) != readFile(a.file("checkpoints/gen_0001.azn")));
}

// ownership = true and resignation = true together are refused, by a new
// run and by a resume, before anything is written; on a run without the
// head, resignation is an ordinary tunable key.
static void test_resignation_refusals_and_resume() {
    TempDir target("resign_own");
    Config both = ownershipConfig();
    both.resignation = true;
    std::string msg;
    try {
        Trainer::startNew(both, target.str());
    } catch (const ConfigError& e) {
        msg = e.what();
    }
    assert(msg.find("resignation") != std::string::npos && msg.find("ownership") != std::string::npos);
    assert(!fs::exists(target.path));

    TempDir owner("resign_owner");
    startQuiet(ownershipConfig(), owner);
    const std::string before = readFile(configPath(owner.str()));
    msg.clear();
    try {
        Trainer::resume(owner.str(), {"resignation=true"});
    } catch (const ConfigError& e) {
        msg = e.what();
    }
    assert(msg.find("resignation") != std::string::npos && msg.find("ownership") != std::string::npos);
    assert(readFile(configPath(owner.str())) == before);

    // A plain run turns resignation on at resume; its window starts empty,
    // so the first generation uses resign_threshold_initial.
    TempDir plain("resign_later");
    startQuiet(tinyConfig(), plain)->runUntil(1);
    std::unique_ptr<Trainer> t = resumeQuiet(plain, {"resignation=on", "resign_threshold_initial=-0.5"});
    assert(t->config().resignation && loadConfigFile(configPath(plain.str())).resignation);
    const GenerationStats st = t->runGeneration();
    assert(st.resignation && st.resignThreshold == -0.5f);
}

// A log.csv written after ownership_loss but before the resignation columns
// gets them, empty in every old row, on resume.
static void test_resume_adds_resignation_columns() {
    TempDir dir("pre_resign_log");
    startQuiet(tinyConfig(), dir)->runUntil(1);
    // Without the resignation columns and every later one.
    const std::string oldLog = firstLogColumns(dir.str(), 16);
    assert(oldLog.find(",ownership_loss\n") != std::string::npos);
    assert(std::count(oldLog.begin(), oldLog.end(), ',') == 2 * 15);
    writeFile(dir.file("log.csv"), oldLog);
    resumeQuiet(dir, {})->runUntil(2);
    const std::vector<std::string> rows = splitLines(readFile(dir.file("log.csv")));
    assert(rows.size() == 3 && rows[0].find(",resign_threshold,resign_rate,resign_false_positive,") != std::string::npos);
    for (const std::string& row : rows) assert(std::count(row.begin(), row.end(), ',') == kLogCommas);
    for (const char* column : {"resign_threshold", "resign_false_positive", "val_value_loss", "seen_ownership_loss"})
        assert(logRow(dir.str(), 0).at(column).empty());
}

// A resigned game's last board is no final position, so a run with the
// ownership head must not train on one: --buffer-from refuses a donor whose
// buffer holds resigned games, naming it. Without the head it is accepted.
static void test_buffer_from_refuses_resigned_games_for_ownership() {
    TempDir donor("resign_donor"), target("resign_donor_own"), plain("resign_donor_plain");
    startQuiet(resignConfig(), donor)->runUntil(1);
    const TrainingState s = loadTrainingState(statePath(donor.str()));
    bool anyResigned = false;
    for (size_t i = 0; i < s.buffer.games(); ++i) anyResigned = anyResigned || s.buffer.game(i).resigned;
    assert(anyResigned);

    Trainer::RunInit init;
    init.bufferFrom = donor.str();
    const std::string msg = configErrorOf([&] { Trainer::startNew(ownershipConfig(), target.str(), init); });
    assert(msg.find(donor.str()) != std::string::npos && msg.find("resign") != std::string::npos &&
           msg.find("ownership") != std::string::npos);
    assert(!fs::exists(statePath(target.str())));

    std::unique_ptr<Trainer> t = Trainer::startNew(tinyConfig(), plain.str(), init);
    t->setLog(nullptr);
    assert(loadTrainingState(statePath(plain.str())).buffer.games() == s.buffer.games());
}

// ---- the validation loss -------------------------------------------------------

// Generation g measures generation g-1's network, in inference mode at the
// identity symmetry, on every position of its own self-play games before
// they join the buffer (val_*), and on as many positions drawn from the
// buffer that network was trained on (seen_*), with the generator
// deriveSeed(seed, g, kPhaseValidation). Recomputed here from the run's
// files, the numbers agree exactly. Generation 1 has an empty buffer, so no
// seen_* loss; without the ownership head, no ownership columns.
static void test_validation_loss_columns() {
    TempDir dir("validation");
    const Config c = tinyConfig();
    assert(c.validationLoss);   // on by default
    std::unique_ptr<Trainer> t = startQuiet(c, dir);
    std::ostringstream out;
    t->setLog(&out);
    const GenerationStats st1 = t->runGeneration();
    const TrainingState after1 = loadTrainingState(statePath(dir.str()));
    const GenerationStats st2 = t->runGeneration();
    const TrainingState after2 = loadTrainingState(statePath(dir.str()));

    std::unique_ptr<Backend> backend = makeBackend(c.backend, 1);
    Workspace ws(c.netConfig(), c.batchSize, *backend);
    // Generation g's games are the buffer's last games_per_generation (nothing is evicted).
    auto gamesOf = [&c](const TrainingState& s) {
        std::vector<GameRecord> games;
        for (size_t i = s.buffer.games() - c.gamesPerGeneration; i < s.buffer.games(); ++i)
            games.push_back(s.buffer.game(i));
        return games;
    };
    const Network gen0 = loadNetwork(checkpointPath(dir.str(), 0));
    const Network gen1 = loadNetwork(checkpointPath(dir.str(), 1));
    const std::vector<GameRecord> games1 = gamesOf(after1), games2 = gamesOf(after2);
    const HeldOutLoss val1 = heldOutLoss(gen0, ws, allPositions(games1), c.history);
    const HeldOutLoss val2 = heldOutLoss(gen1, ws, allPositions(games2), c.history);
    Rng rng(deriveSeed(c.seed, 2, kPhaseValidation));
    const HeldOutLoss seen2 =
        heldOutLoss(gen1, ws, samplePositions(after1.buffer, static_cast<int>(val2.positions), rng), c.history);

    assert(st1.valPolicyLoss == val1.policyLoss && st1.valValueLoss == val1.valueLoss);
    assert(st2.valPolicyLoss == val2.policyLoss && st2.valValueLoss == val2.valueLoss);
    assert(st2.seenPolicyLoss == seen2.policyLoss && st2.seenValueLoss == seen2.valueLoss);
    assert(st1.seenPolicyLoss == -1 && st1.seenValueLoss == -1);
    assert(st1.valOwnershipLoss == -1 && st2.valOwnershipLoss == -1 && st2.seenOwnershipLoss == -1);
    assert(st2.valPolicyLoss > 0 && st2.valValueLoss >= 0 && st2.seenPolicyLoss > 0);

    const std::map<std::string, std::string> row1 = logRow(dir.str(), 0), row2 = logRow(dir.str(), 1);
    assert(row1.at("val_policy_loss") == loss5(val1.policyLoss) && row1.at("val_value_loss") == loss5(val1.valueLoss));
    assert(row2.at("val_policy_loss") == loss5(val2.policyLoss) && row2.at("val_value_loss") == loss5(val2.valueLoss));
    assert(row2.at("seen_policy_loss") == loss5(seen2.policyLoss) &&
           row2.at("seen_value_loss") == loss5(seen2.valueLoss));
    for (const char* empty : {"seen_policy_loss", "seen_value_loss", "seen_ownership_loss", "val_ownership_loss"})
        assert(row1.at(empty).empty());
    assert(row2.at("val_ownership_loss").empty() && row2.at("seen_ownership_loss").empty());

    // The progress line: "val p X v Y" after the training loss, and "seen p X v Y" once the buffer has games.
    const std::vector<std::string> lines = splitLines(out.str());
    assert(lines.size() == 2);
    char val[64], seen[64];
    std::snprintf(val, sizeof val, " lr 0.02 val p %.2f v %.2f ", st1.valPolicyLoss, st1.valValueLoss);
    assert(lines[0].find(val) != std::string::npos && lines[0].find(" seen ") == std::string::npos);
    std::snprintf(val, sizeof val, " val p %.2f v %.2f seen p ", st2.valPolicyLoss, st2.valValueLoss);
    std::snprintf(seen, sizeof seen, " seen p %.2f v %.2f ", st2.seenPolicyLoss, st2.seenValueLoss);
    assert(lines[1].find(val) != std::string::npos && lines[1].find(seen) != std::string::npos);
}

// With the ownership head, the head's loss is measured too: the unweighted
// per-point squared error, like ownership_loss.
static void test_validation_loss_with_ownership_head() {
    TempDir dir("validation_own");
    std::unique_ptr<Trainer> t = startQuiet(ownershipConfig(), dir);
    t->runGeneration();
    const GenerationStats st = t->runGeneration();
    assert(st.valOwnershipLoss > 0 && st.valOwnershipLoss < 4);
    assert(st.seenOwnershipLoss > 0 && st.seenOwnershipLoss < 4);
    const std::map<std::string, std::string> row = logRow(dir.str());
    assert(row.at("val_ownership_loss") == loss5(st.valOwnershipLoss));
    assert(row.at("seen_ownership_loss") == loss5(st.seenOwnershipLoss));
}

// The measurement changes nothing a run produces: with validation_loss off,
// every checkpoint, state.bin, the ratings and the games are byte-identical,
// with and without the ownership head, and so are log.csv's other columns
// except the timings. Off, the six columns are empty and the progress line
// does not mention them.
static void test_validation_loss_changes_nothing() {
    for (bool ownership : {false, true}) {
        TempDir on(ownership ? "val_own_on" : "val_on"), off(ownership ? "val_own_off" : "val_off");
        Config c = ownership ? ownershipConfig() : tinyConfig();
        startQuiet(c, on)->runUntil(3);
        c.validationLoss = false;
        std::unique_ptr<Trainer> t = startQuiet(c, off);
        std::ostringstream out;
        t->setLog(&out);
        t->runUntil(3);
        for (const char* f : {"checkpoints/gen_0001.azn", "checkpoints/gen_0002.azn", "checkpoints/gen_0003.azn",
                              "state.bin", "elo.csv", "results.csv", "references.csv",
                              "sgf/gen_0001_game_00.sgf", "sgf/gen_0003_game_00.sgf"})
            assert(readFile(on.file(f)) == readFile(off.file(f)));
        for (int i = 0; i < 3; ++i) {
            std::map<std::string, std::string> a = logRow(on.str(), i), b = logRow(off.str(), i);
            assert(!a.at("val_policy_loss").empty());
            for (const char* column : {"val_policy_loss", "val_value_loss", "val_ownership_loss",
                                       "seen_policy_loss", "seen_value_loss", "seen_ownership_loss"}) {
                assert(b.at(column).empty());
                a.erase(column);
                b.erase(column);
            }
            for (const char* timing : {"selfplay_s", "train_s", "arena_s"}) {
                a.erase(timing);
                b.erase(timing);
            }
            assert(a == b);
        }
        assert(out.str().find(" val ") == std::string::npos && out.str().find(" seen ") == std::string::npos);
    }
}

// Like everything else in a run, the measurement does not depend on the worker count.
static void test_validation_loss_worker_independence() {
    TempDir one("val_workers1"), three("val_workers3");
    Config c = tinyConfig();
    c.workers = 1;
    startQuiet(c, one)->runUntil(2);
    c.workers = 3;
    startQuiet(c, three)->runUntil(2);
    for (int i = 0; i < 2; ++i) {
        const std::map<std::string, std::string> a = logRow(one.str(), i), b = logRow(three.str(), i);
        for (const char* column : {"val_policy_loss", "val_value_loss", "seen_policy_loss", "seen_value_loss"})
            assert(a.at(column) == b.at(column));
    }
    assert(!logRow(one.str(), 1).at("seen_value_loss").empty());
}

// A log.csv written before the validation columns gets them, empty in every
// old row, on resume; validation_loss is tunable, so a resume can turn it
// off (or on) and the next row follows.
static void test_resume_adds_validation_columns() {
    TempDir dir("pre_validation_log");
    startQuiet(tinyConfig(), dir)->runUntil(1);
    const std::string oldLog = firstLogColumns(dir.str(), 19);
    assert(oldLog.find(",resign_false_positive\n") != std::string::npos);
    writeFile(dir.file("log.csv"), oldLog);
    resumeQuiet(dir, {})->runUntil(2);
    std::vector<std::string> rows = splitLines(readFile(dir.file("log.csv")));
    assert(rows.size() == 3 && rows[0].find(",resign_false_positive,val_policy_loss,") != std::string::npos);
    for (const std::string& row : rows) assert(std::count(row.begin(), row.end(), ',') == kLogCommas);
    assert(rows[1].size() > 6 && rows[1].compare(rows[1].size() - 6, 6, ",,,,,,") == 0);
    assert(!logRow(dir.str(), 1).at("val_policy_loss").empty());

    std::unique_ptr<Trainer> t = resumeQuiet(dir, {"validation_loss=false"});
    assert(!t->config().validationLoss && !loadConfigFile(configPath(dir.str())).validationLoss);
    t->runUntil(3);
    assert(logRow(dir.str()).at("val_policy_loss").empty());
    resumeQuiet(dir, {"validation_loss=true"})->runUntil(4);
    assert(!logRow(dir.str()).at("val_policy_loss").empty());
}

// A measurement that is not a number is printed "nan", not hidden as an
// empty (unmeasured) field. Here generation 0's ownership head has NaN
// running means: search and training never read them (training uses batch
// statistics), so the run goes on, but the inference-mode ownership loss
// is NaN at every generation.
static void test_validation_loss_prints_nan() {
    TempDir dir("val_nan");
    fs::create_directories(dir.path);
    Network net(ownershipConfig().netConfig(), 5);
    for (const BufferView& b : net.buffers())
        if (b.name == "ownership.bn.runMean") std::fill(b.value, b.value + b.size, std::nanf(""));
    Trainer::RunInit init;
    init.initFrom = dir.file("nan.azn");
    saveNetwork(net, init.initFrom);
    TempDir run("val_nan_run");
    std::unique_ptr<Trainer> t = Trainer::startNew(ownershipConfig(), run.str(), init);
    std::ostringstream out;
    t->setLog(&out);
    t->runUntil(2);
    const std::map<std::string, std::string> row = logRow(run.str());
    assert(row.at("val_ownership_loss") == "nan" && row.at("seen_ownership_loss") == "nan");
    assert(!row.at("val_value_loss").empty() && row.at("val_value_loss") != "nan");
    assert(out.str().find(" val p ") != std::string::npos);
}

// train_threads changes how the training phase's network passes are spread
// over cores, never the result: train_threads 1 and 4 write the same bytes,
// with batches wide enough (48 x 25 pixels) for the convolutions' gemms to
// be cut into column blocks, with BLAS when it is built. It is tunable: a run resumed with a
// different value continues exactly as the uninterrupted one.
static void test_train_threads_change_nothing() {
    TempDir one("train_threads1"), four("train_threads4"), resumed("train_threads_resumed");
    Config c = tinyConfig();
    c.filters = 32;
    c.batchSize = 48;
    if (blasAvailable()) c.backend = "blas";
    c.trainThreads = 1;
    startQuiet(c, one)->runUntil(2);
    c.trainThreads = 4;
    startQuiet(c, four)->runUntil(2);
    startQuiet(c, resumed)->runUntil(1);
    {
        std::unique_ptr<Trainer> t = resumeQuiet(resumed, {"train_threads=2"});
        assert(t->config().trainThreads == 2);
        t->runUntil(2);
    }
    assert(readFile(resumed.file("config.cfg")).find("train_threads = 2\n") != std::string::npos);
    for (const TempDir* other : {&four, &resumed}) {
        for (const char* f : {"checkpoints/gen_0001.azn", "checkpoints/gen_0002.azn", "state.bin", "results.csv",
                              "elo.csv", "references.csv"})
            assert(readFile(one.file(f)) == readFile(other->file(f)));
        for (int i = 0; i < 2; ++i)
            for (const char* column : {"policy_loss", "value_loss", "val_policy_loss", "val_value_loss",
                                       "seen_policy_loss", "seen_value_loss"})
                assert(logRow(one.str(), i).at(column) == logRow(other->str(), i).at(column));
    }
}

// bn_calibrate: a fresh generation 0 is calibrated (bn_calibration.hpp),
// to the same bytes for any workers, train_threads and backend; with the
// key off its running statistics are the initial ones; a network given with
// --init-from is used as it is.
static void test_generation_0_batch_norm_calibration() {
    TempDir reference("bn_reference"), off("bn_off"), given("bn_given");
    Config c = tinyConfig();
    c.workers = 1;
    c.trainThreads = 1;
    c.backend = "naive";
    startQuiet(c, reference);
    const std::string gen0 = readFile(checkpointPath(reference.str(), 0));
    assert(readFile(reference.file("config.cfg")).find("\nbn_calibrate = true\n") != std::string::npos);
    // Exactly the documented calibration of the documented network.
    Network expected(c.netConfig(), deriveSeed(c.seed, 0, kPhaseInit));
    calibrateBatchNorm(expected, deriveSeed(c.seed, 0, kPhaseBatchNorm), 1);
    saveNetwork(expected, reference.file("expected.azn"));
    assert(readFile(reference.file("expected.azn")) == gen0);

    // Other worker and training thread counts, and every available backend.
    std::vector<std::string> backends = {"naive"};
    if (blasAvailable()) backends.push_back("blas");
    if (metalAvailable()) backends.push_back("metal");
    for (const std::string& backend : backends) {
        TempDir other("bn_" + backend);
        c.workers = 3;
        c.trainThreads = 4;
        c.backend = backend;
        startQuiet(c, other);
        assert(readFile(checkpointPath(other.str(), 0)) == gen0);
    }

    // Off: the same weights with the initial running statistics.
    c = tinyConfig();
    c.bnCalibrate = false;
    startQuiet(c, off);
    assert(readFile(off.file("config.cfg")).find("\nbn_calibrate = false\n") != std::string::npos);
    Network fresh = loadNetwork(checkpointPath(off.str(), 0));
    const std::vector<ParamView> freshParams = fresh.params(), expectedParams = expected.params();
    for (size_t i = 0; i < freshParams.size(); ++i)
        assert(std::equal(freshParams[i].value, freshParams[i].value + freshParams[i].size, expectedParams[i].value));
    for (const BufferView& b : fresh.buffers()) {
        const bool isVar = b.name.find("runVar") != std::string::npos;
        for (size_t i = 0; i < b.size; ++i) assert(b.value[i] == (isVar ? 1.0f : 0.0f));
    }
    assert(readFile(checkpointPath(off.str(), 0)) != gen0);

    // --init-from with bn_calibrate on: generation 0 is the given file.
    Trainer::RunInit init;
    init.initFrom = checkpointPath(off.str(), 0);
    c.bnCalibrate = true;
    Trainer::startNew(c, given.str(), init)->setLog(nullptr);
    assert(readFile(checkpointPath(given.str(), 0)) == readFile(init.initFrom));
}

// Resume never calibrates: a run's checkpoints are untouched, and a
// config.cfg written before bn_calibrate existed (whose generation 0 was not
// calibrated) is rewritten with bn_calibrate = false. A run that has the key
// keeps it.
static void test_resume_and_bn_calibrate() {
    TempDir old("bn_resume_old"), current("bn_resume_current");
    Config c = tinyConfig();
    c.bnCalibrate = false;
    startQuiet(c, old)->runUntil(1);
    std::string text = readFile(old.file("config.cfg"));
    const size_t line = text.find("bn_calibrate = false\n");
    assert(line != std::string::npos);
    text.erase(line, std::string("bn_calibrate = false\n").size());   // as code before the key wrote it
    writeFile(old.file("config.cfg"), text);
    const std::string gen0 = readFile(checkpointPath(old.str(), 0)), gen1 = readFile(checkpointPath(old.str(), 1));
    resumeQuiet(old, {})->runUntil(2);
    assert(readFile(old.file("config.cfg")).find("\nbn_calibrate = false\n") != std::string::npos);
    assert(readFile(checkpointPath(old.str(), 0)) == gen0 && readFile(checkpointPath(old.str(), 1)) == gen1);

    startQuiet(tinyConfig(), current)->runUntil(1);
    const std::string calibrated0 = readFile(checkpointPath(current.str(), 0));
    resumeQuiet(current, {})->runUntil(2);
    assert(readFile(current.file("config.cfg")).find("\nbn_calibrate = true\n") != std::string::npos);
    assert(readFile(checkpointPath(current.str(), 0)) == calibrated0);
}

// backend = metal in a build without Metal: start, resume and re-rate refuse
// before writing anything, with the message that says how to get it.
static void test_metal_unavailable_writes_nothing() {
    if (metalBuilt()) return;   // tests/test_metal.cpp covers a Metal build
    const std::string notBuilt = "backend 'metal' not available: rebuild with -DAZ_USE_METAL=ON";
    TempDir dir("metal_unbuilt");
    Config c = tinyConfig();
    c.backend = "metal";
    bool threw = false;
    try {
        Trainer::startNew(c, dir.str());
    } catch (const std::exception& e) {
        threw = std::string(e.what()).find(notBuilt) != std::string::npos;
    }
    assert(threw);
    assert(!fs::exists(statePath(dir.str())) && !fs::exists(configPath(dir.str())));
    assert(!fs::exists(checkpointPath(dir.str(), 0)));

    // An existing run: resume and re-rate with metal refuse and change nothing.
    startQuiet(tinyConfig(), dir)->runUntil(1);
    const std::string config = readFile(configPath(dir.str())), state = readFile(statePath(dir.str()));
    for (int which = 0; which < 2; ++which) {
        threw = false;
        try {
            if (which == 0) Trainer::resume(dir.str(), {"backend=metal"});
            else rerateRun(dir.str(), {"backend=metal"}, nullptr);
        } catch (const std::exception& e) {
            threw = std::string(e.what()).find(notBuilt) != std::string::npos;
        }
        assert(threw);
    }
    assert(readFile(configPath(dir.str())) == config && readFile(statePath(dir.str())) == state);
    assert(!fs::exists(dir.file("rerate")));
}

// Switching backends on resume is allowed (the algorithm is the same) but
// changes the last bits from then on, and the trainer says so.
static void test_backend_switch_prints_note() {
    if (!blasAvailable()) return;
    TempDir dir("switch");
    startQuiet(tinyConfig(), dir)->runUntil(1);
    std::ostringstream quiet;
    Trainer::resume(dir.str(), {}, &quiet);
    assert(quiet.str().find("note:") == std::string::npos);   // nothing changed, nothing said
    std::ostringstream out;
    std::unique_ptr<Trainer> t = Trainer::resume(dir.str(), {"backend=blas"}, &out);
    assert(out.str() == "note: backend changed from naive to blas; from here on results differ in the\n"
                        "last bits from a run that kept naive\n");
    assert(t->config().backend == "blas");
}

// A generation that fails after writing its checkpoint (in the arena,
// say) leaves a file the run has not committed. Re-rating stops at the
// generation state.bin committed, and a resume overwrites the file with the
// same bytes.
static void test_rerate_ignores_uncommitted_checkpoint() {
    TempDir dir("orphan");
    startQuiet(tinyConfig(), dir)->runUntil(2);
    const std::string results = readFile(dir.file("results.csv"));
    // An orphan gen 3, as a failed arena would leave it.
    writeFile(checkpointPath(dir.str(), 3), readFile(checkpointPath(dir.str(), 2)));
    assert(rerateRun(dir.str(), {}, nullptr) == 2);
    assert(readFile(dir.file("rerate/results.csv")) == results);
}


// ---- arena rounds played at once ---------------------------------------------

// Everything a run wrote that must not depend on how its arena matches were
// scheduled: every checkpoint, state.bin, the rating files, the SGFs, and
// log.csv without its three timing columns (selfplay_s, train_s, arena_s).
static std::string runFingerprint(const TempDir& dir, int generations) {
    std::string out;
    for (int g = 0; g <= generations; ++g) out += readFile(checkpointPath(dir.str(), g));
    for (const char* f : {"state.bin", "results.csv", "references.csv", "elo.csv"}) out += readFile(dir.file(f));
    if (fs::exists(dir.file("reference-networks.csv"))) out += readFile(dir.file("reference-networks.csv"));
    std::vector<std::string> sgf;
    for (const auto& e : fs::directory_iterator(dir.file("sgf"))) sgf.push_back(e.path().string());
    std::sort(sgf.begin(), sgf.end());
    assert(!sgf.empty());
    for (const std::string& f : sgf) out += readFile(f);
    for (const std::string& row : splitLines(readFile(dir.file("log.csv")))) {
        const std::vector<std::string> fields = splitFields(row);
        for (size_t i = 0; i < fields.size(); ++i)
            if (i < 9 || i > 11) out += fields[i] + ",";
        out += "\n";
    }
    return out;
}

// A run whose arena rounds have every kind of match: two bots and two
// reference networks (all calibrated against generation 0 in one wave),
// gating (the best network joins the opponents), and promote_threshold 0.5
// so generations are promoted and play calibration games in a second wave.
static Config roundsConfig(const TempDir& refs, const std::string& backend) {
    Config c = tinyConfig();
    c.backend = backend;
    c.botSimulations = {8, 16};
    c.arenaGames = 4;
    c.calibrationGames = 4;
    c.promoteThreshold = 0.5f;
    c.gating = true;
    c.gatingThreshold = 0.5f;
    if (!fs::exists(refs.file("ref.azn"))) frozenGeneration1(refs);
    if (!fs::exists(refs.file("ref0.azn"))) {
        TempDir donor(refs.path.filename().string() + "_donor0");
        Config d = tinyConfig();
        d.seed = 5;
        startQuiet(d, donor)->runUntil(0);
        fs::copy_file(checkpointPath(donor.str(), 0), refs.file("ref0.azn"));
    }
    c.referenceNetworks = {refs.file("ref.azn"), refs.file("ref0.azn")};
    return c;
}

// The trainer plays each wave of an arena round (the calibrations at the
// start, a generation's opponents, its promotion games) at once. Every file
// is byte for byte what playing the matches one after another (the arena
// test hook) gives, for 1, 3, 8 and all-core workers and both inference
// batches, on resume and on re-rating.
static void test_rounds_at_once_equal_one_at_a_time() {
    std::vector<std::string> backends = {"naive"};
    if (blasAvailable()) backends.push_back("blas");
    for (const std::string& be : backends) {
        TempDir refs("rounds_refs_" + be), sequential("rounds_seq_" + be);
        const Config base = roundsConfig(refs, be);
        Config c = base;
        c.workers = 1;
        arena_probe::setOneMatchAtATime(true);
        startQuiet(c, sequential)->runUntil(4);
        arena_probe::setOneMatchAtATime(false);
        const std::string reference = runFingerprint(sequential, 4);

        // The run exercises what it should: both bots and both reference
        // networks were rated, a generation was promoted, gating is on.
        const TrainingState s = loadTrainingState(statePath(sequential.str()));
        assert(s.references.back() > 0);   // a promoted generation
        const std::map<std::pair<int, int>, ResultsTable::Pair> pairs = readResults(sequential);
        for (int p : {-4, -3, -2, -1}) assert(pairs.count({p, 0}) && pairs.at({p, 0}).games == 4);
        bool fixedOpponent = false;
        for (const auto& [key, pair] : pairs) fixedOpponent = fixedOpponent || (key.first < 0 && key.second > 0);
        assert(fixedOpponent);

        const int workerCounts[] = {1, 3, 8, 0};
        const int batches[] = {8, 256, 256, 8};
        for (int v = 0; v < 4; ++v) {
            TempDir dir("rounds_" + be + "_" + std::to_string(v));
            c = base;
            c.workers = workerCounts[v];
            c.inferenceBatch = batches[v];
            startQuiet(c, dir)->runUntil(4);
            assert(runFingerprint(dir, 4) == reference);
        }

        // Resume: two generations, then two more.
        TempDir resumed("rounds_resume_" + be);
        c = base;
        c.workers = 3;
        startQuiet(c, resumed)->runUntil(2);
        resumeQuiet(resumed, {})->runUntil(4);
        assert(runFingerprint(resumed, 4) == reference);

        // Re-rating (which ignores gating) gives the same files whether its
        // waves are played at once or one match after another.
        arena_probe::setOneMatchAtATime(true);
        assert(rerateRun(sequential.str(), {}, nullptr) == 4);
        arena_probe::setOneMatchAtATime(false);
        assert(rerateRun(resumed.str(), {"workers=5"}, nullptr) == 4);
        for (const char* f : {"results.csv", "elo.csv", "references.csv", "reference-networks.csv", "log.csv"})
            assert(readFile(resumed.file(std::string("rerate/") + f)) ==
                   readFile(sequential.file(std::string("rerate/") + f)));
    }
}

// ---- self-play engines (distributed self-play, one machine) -------------------

// The fields of one CSV row; a quoted field may hold commas.
static std::vector<std::string> csvFields(const std::string& row) {
    std::vector<std::string> fields(1);
    bool quoted = false;
    for (char ch : row) {
        if (ch == '"') quoted = !quoted;
        else if (ch == ',' && !quoted) fields.emplace_back();
        else fields.back() += ch;
    }
    return fields;
}

// selfplay-engines.csv without its timing column (seconds, the 8th).
static std::string manifestWithoutTimings(const TempDir& dir) {
    std::string out;
    for (const std::string& row : splitLines(readFile(dir.file("selfplay-engines.csv")))) {
        const std::vector<std::string> fields = csvFields(row);
        assert(fields.size() == 11);
        for (size_t i = 0; i < fields.size(); ++i)
            if (i != 7) out += fields[i] + "|";
        out += "\n";
    }
    return out;
}

static int manifestRows(const TempDir& dir) {
    return static_cast<int>(splitLines(readFile(dir.file("selfplay-engines.csv"))).size()) - 1;
}

// A run with the given hosts file (a path that does not exist: none), its
// startup lines discarded.
static std::unique_ptr<Trainer> startEngines(const Config& c, const TempDir& dir,
                                             const std::string& hosts) {
    TrainerOptions opts;
    opts.hostsFile = hosts;
    std::ostringstream startup;
    std::unique_ptr<Trainer> t = Trainer::startNew(c, dir.str(), {}, &startup, opts);
    t->setLog(nullptr);
    return t;
}

static std::unique_ptr<Trainer> resumeEngines(const TempDir& dir,
                                              const std::vector<std::string>& overrides,
                                              const std::string& hosts,
                                              std::ostream* out = nullptr) {
    TrainerOptions opts;
    opts.hostsFile = hosts;
    std::ostringstream quiet;
    std::unique_ptr<Trainer> t = Trainer::resume(dir.str(), overrides, out ? out : &quiet, opts);
    t->setLog(nullptr);
    return t;
}

static Config withEngines(Config c, const std::string& engines) {
    applyOverride(c, "selfplay_engines=" + engines);
    return c;
}

// selfplay_engines = local/naive:1 is the default run's engine with the
// default worker rule's results: every file is the same, and only the
// engine run writes a manifest, one row per generation.
static void test_single_local_engine_equals_default() {
    TempDir plain("engines_default"), engines("engines_single");
    const std::string noHosts = engines.file("no-hosts");
    startQuiet(tinyConfig(), plain)->runUntil(2);
    std::ostringstream startup;
    TrainerOptions opts;
    opts.hostsFile = noHosts;
    const Config single = withEngines(tinyConfig(), "local/naive:1");
    std::unique_ptr<Trainer> t = Trainer::startNew(single, engines.str(), {}, &startup, opts);
    t->setLog(nullptr);
    t->runUntil(2);
    assert(startup.str() == "engine local/naive: " + engineFingerprint("naive").substr(7) + " [" +
                                shortBuildId() + "]\n");
    assert(runFingerprint(plain, 2) == runFingerprint(engines, 2));
    assert(!fs::exists(plain.file("selfplay-engines.csv")));
    assert(manifestRows(engines) == 2);
    const std::vector<std::string> rows = splitLines(readFile(engines.file("selfplay-engines.csv")));
    assert(rows[0] == "generation,first,count,planned,engine,fingerprint,build,seconds,positions,"
                      "evaluations,note");
    const std::vector<std::string> row = csvFields(rows[1]);
    assert(row[0] == "1" && row[1] == "0" && row[2] == "4" && row[3] == "local/naive");
    assert(row[4] == "local/naive" && row[5] == engineFingerprint("naive"));
    assert(row[6] == shortBuildId() && row[10].empty());
    assert(std::stoll(row[8]) > 0 && std::stoll(row[9]) > 0);
}

// Two items of one engine split its games without changing a bit: one
// weighted, one fixed, with resignation on.
static void test_split_local_engines_equal_default() {
    TempDir plain("split_default"), engines("split_engines");
    startQuiet(resignConfig(), plain)->runUntil(3);
    const Config split = withEngines(resignConfig(), "local/naive:2, local/naive=1");
    startEngines(split, engines, engines.file("none"))->runUntil(3);
    assert(runFingerprint(plain, 3) == runFingerprint(engines, 3));
    assert(manifestRows(engines) == 6);
    const std::string manifest = readFile(engines.file("selfplay-engines.csv"));
    const std::vector<std::string> row = csvFields(splitLines(manifest)[2]);
    assert(row[0] == "1" && row[1] == "7" && row[2] == "1");   // 8 games: 7 weighted, 1 fixed
}

static void test_engines_resume_is_byte_identical() {
    TempDir straight("engines_straight"), resumed("engines_resumed");
    const Config c = withEngines(resignConfig(), "local/naive:1, local/naive:3");
    startEngines(c, straight, straight.file("none"))->runUntil(4);
    startEngines(c, resumed, resumed.file("none"))->runUntil(2);
    resumeEngines(resumed, {}, resumed.file("none"))->runUntil(4);
    assert(runFingerprint(straight, 4) == runFingerprint(resumed, 4));
    assert(manifestWithoutTimings(straight) == manifestWithoutTimings(resumed));
    assert(manifestRows(resumed) == 8);
}

// Engines can be switched on (and off) at a generation boundary. The
// earlier generations stay what they were, the switch is noted once, and
// config.cfg records it.
static void test_switch_on_at_resume() {
    TempDir plain("switch_default"), switched("switch_engines");
    const std::string none = switched.file("none");
    startQuiet(tinyConfig(), plain)->runUntil(4);
    startQuiet(tinyConfig(), switched)->runUntil(2);
    std::ostringstream out;
    resumeEngines(switched, {"selfplay_engines=local/naive:1,local/naive:1"}, none, &out)->runUntil(4);
    const std::string note = "note: self-play engines changed from local/naive to "
                             "local/naive:1, local/naive:1; from here on results differ in the\n"
                             "last bits from a run that kept local/naive\n";
    assert(out.str().find(note) != std::string::npos);
    assert(out.str().find("note:") == out.str().rfind("note:"));   // once
    // Naive on one machine: the same arithmetic, so even generations 3-4 agree.
    assert(runFingerprint(plain, 4) == runFingerprint(switched, 4));
    for (int g = 1; g <= 2; ++g)
        assert(readFile(checkpointPath(plain.str(), g)) == readFile(checkpointPath(switched.str(), g)));
    const std::string config = readFile(configPath(switched.str()));
    assert(config.find("\nselfplay_engines = local/naive:1, local/naive:1\n") != std::string::npos);
    const std::vector<std::string> rows = splitLines(readFile(switched.file("selfplay-engines.csv")));
    assert(rows.size() == 5 && rows[1].rfind("3,0,2,", 0) == 0 && rows[4].rfind("4,2,2,", 0) == 0);
    // Resuming without a change says nothing; switching back is noted again.
    std::ostringstream same, back;
    resumeEngines(switched, {}, none, &same);
    assert(same.str().find("note:") == std::string::npos);
    resumeEngines(switched, {"selfplay_engines="}, none, &back)->runUntil(5);
    assert(back.str() == "note: self-play engines changed from local/naive:1, local/naive:1 to "
                         "local/naive; from here on results differ in the\nlast bits from a run "
                         "that kept local/naive:1, local/naive:1\n");
    assert(manifestRows(switched) == 4);   // generation 5 played on the default path
}

// A crash between the manifest and state.bin leaves rows of a generation
// that was never committed. Every resume drops them, with engines on or
// off, so the manifest only describes committed generations.
static void test_resume_prunes_uncommitted_manifest_rows() {
    for (bool enginesOn : {true, false}) {
        TempDir dir(enginesOn ? "prune_on" : "prune_off");
        const std::string none = dir.file("none");
        startEngines(withEngines(tinyConfig(), "local/naive:1, local/naive:1"), dir, none)->runUntil(2);
        const std::string committed = readFile(dir.file("selfplay-engines.csv"));
        // What a crash after the manifest of generation 3 would leave.
        writeFile(dir.file("selfplay-engines.csv"),
                  committed + "3,0,2,local/naive,local/naive,\"x\",b,1.0,10,20,\n");
        const std::vector<std::string> overrides =
            enginesOn ? std::vector<std::string>{} : std::vector<std::string>{"selfplay_engines="};
        resumeEngines(dir, overrides, none);
        assert(readFile(dir.file("selfplay-engines.csv")) == committed);
    }
}

// ---- remote slices through a fake remote ------------------------------------------

// A hosts file defining `fake`: no ssh, so the subprocess transport runs
// the real selfplay binary here, in its own work directory.
static std::string fakeHosts(const TempDir& dir, const std::string& extra = "") {
    fs::create_directories(dir.path);
    writeFile(dir.file("hosts"), "fake exe=" + std::string(AZGO_SELFPLAY_EXE) + " workdir=" +
                                     dir.file("remote") + (extra.empty() ? "" : " " + extra) + "\n");
    return dir.file("hosts");
}

// Both engines are naive on this machine: the same arithmetic, so a run
// with half its games played by another process is the default run.
static void test_fake_remote_equals_local() {
    TempDir plain("fake_default"), engines("fake_engines");
    startQuiet(resignConfig(), plain)->runUntil(3);
    const Config c = withEngines(resignConfig(), "local/naive:1, fake/naive:1");
    TrainerOptions opts;
    opts.hostsFile = fakeHosts(engines);
    std::ostringstream startup;
    std::unique_ptr<Trainer> t = Trainer::startNew(c, engines.str(), {}, &startup, opts);
    t->setLog(nullptr);
    t->runUntil(3);
    const std::string fp = engineFingerprint("naive").substr(7);
    assert(startup.str().find("engine local/naive: " + fp + " [" + shortBuildId() + "]\n") !=
           std::string::npos);
    assert(startup.str().find("engine fake/naive: " + fp + " [" + shortBuildId() + "]\n") !=
           std::string::npos);
    assert(runFingerprint(plain, 3) == runFingerprint(engines, 3));
    const std::vector<std::string> rows = splitLines(readFile(engines.file("selfplay-engines.csv")));
    assert(rows.size() == 7);
    const std::vector<std::string> local = csvFields(rows[1]), fake = csvFields(rows[2]);
    assert(local[4] == "local/naive" && fake[3] == "fake/naive" && fake[4] == "fake/naive");
    assert(fake[1] == "4" && fake[2] == "4" && fake[5] == local[5] && fake[6] == shortBuildId());
    assert(std::stoll(fake[8]) > 0 && std::stoll(fake[9]) > 0 && fake[10].empty());
}

// A run directory whose name no remote shell could take (a space, ':')
// still has a safe run id there.
static void test_fake_remote_unusual_run_dir() {
    TempDir dir("fake run:dir"), remote("fake_unusual_remote");
    startEngines(withEngines(tinyConfig(), "local/naive:1, fake/naive:1"), dir, fakeHosts(remote))
        ->runUntil(1);
    assert(manifestRows(dir) == 2);
}

static void test_fake_remote_resume() {
    TempDir straight("fake_straight"), resumed("fake_resumed");
    const Config c = withEngines(resignConfig(), "fake/naive:2, local/naive=3");
    startEngines(c, straight, fakeHosts(straight))->runUntil(4);
    startEngines(c, resumed, fakeHosts(resumed))->runUntil(2);
    resumeEngines(resumed, {}, resumed.file("hosts"))->runUntil(4);
    assert(runFingerprint(straight, 4) == runFingerprint(resumed, 4));
    assert(manifestWithoutTimings(straight) == manifestWithoutTimings(resumed));
}

// Preflight refuses, before anything is written: a remote built from
// another commit, an alias the hosts file lacks, a backend the remote
// cannot run.
static void test_preflight_refusals() {
    auto startError = [](const TempDir& dir, const std::string& engines, const std::string& hosts) {
        try {
            startEngines(withEngines(tinyConfig(), engines), dir, hosts);
        } catch (const ConfigError& e) {
            return std::string(e.what());
        }
        return std::string("accepted");
    };
    {
        TempDir dir("preflight_build");
        const std::string hosts = fakeHosts(dir);
        distributed_probe::setRemoteBuildId("0000000000000000000000000000000000000000");
        const std::string msg = startError(dir, "local/naive:1, fake/naive:1", hosts);
        distributed_probe::reset();
        assert(msg.find("fake") != std::string::npos && msg.find(buildId()) != std::string::npos &&
               msg.find("0000000000000000000000000000000000000000") != std::string::npos);
        assert(!fs::exists(statePath(dir.str())) && !fs::exists(configPath(dir.str())));
        // The same on resume: config.cfg is left as it was.
        startQuiet(tinyConfig(), dir)->runUntil(1);
        const std::string config = readFile(configPath(dir.str()));
        distributed_probe::setRemoteBuildId("0000000000000000000000000000000000000000");
        bool threw = false;
        try {
            resumeEngines(dir, {"selfplay_engines=local/naive:1, fake/naive:1"}, hosts);
        } catch (const ConfigError&) {
            threw = true;
        }
        distributed_probe::reset();
        assert(threw && readFile(configPath(dir.str())) == config);
    }
    {
        TempDir dir("preflight_alias");
        const std::string msg = startError(dir, "local/naive:1, mini9/naive:1", fakeHosts(dir));
        assert(msg.find("mini9") != std::string::npos && !fs::exists(configPath(dir.str())));
    }
    if (!metalBuilt()) {
        TempDir dir("preflight_metal");
        const std::string msg = startError(dir, "local/naive:1, fake/metal:1", fakeHosts(dir));
        assert(msg.find("fake/metal") != std::string::npos && msg.find("metal") != std::string::npos);
        assert(!fs::exists(configPath(dir.str())));
    }
}

// After a generation the remote work directory holds only the network,
// and the local work directory is empty.
// A truncated network left on a remote (an upload cut short before this
// coordinator started) is never trusted: each coordinator uploads the
// network itself, atomically, and the run is the default run.
static void test_partial_remote_network_is_replaced() {
    TempDir plain("partial_default"), engines("partial_engines");
    startQuiet(tinyConfig(), plain)->runUntil(2);
    const Config c = withEngines(tinyConfig(), "local/naive:1, fake/naive:1");
    std::unique_ptr<Trainer> t = startEngines(c, engines, fakeHosts(engines));
    char name[48];
    std::snprintf(name, sizeof name, "net_%016llx.azn",
                  static_cast<unsigned long long>(fileHash(checkpointPath(engines.str(), 0))));
    const fs::path remoteDir = fs::path(engines.file("remote")) / runIdOf(engines.str());
    fs::create_directories(remoteDir);
    const std::string full = readFile(checkpointPath(engines.str(), 0));
    writeFile((remoteDir / name).string(), full.substr(0, full.size() / 2));
    t->runUntil(2);
    assert(runFingerprint(plain, 2) == runFingerprint(engines, 2));
    // No partial upload is left behind either.
    for (const auto& e : fs::recursive_directory_iterator(engines.file("remote")))
        assert(e.path().filename().string().find(".part") == std::string::npos);
}

static std::vector<std::string> remoteFiles(const TempDir& dir) {
    std::vector<std::string> files;
    if (!fs::exists(dir.file("remote"))) return files;
    for (const auto& e : fs::recursive_directory_iterator(dir.file("remote")))
        if (e.is_regular_file()) files.push_back(e.path().filename().string());
    return files;
}

// While the run goes on, the remote work directory holds only the network,
// and the local work directory is empty; once the run's coordinator is
// gone, the network is gone too. Stale files of an earlier coordinator
// (a crash) are cleared at the first contact.
static void test_remote_cleanup() {
    TempDir dir("fake_cleanup");
    {
        std::unique_ptr<Trainer> t =
            startEngines(withEngines(tinyConfig(), "local/naive:1, fake/naive:1"), dir, fakeHosts(dir));
        t->runUntil(2);
        const std::vector<std::string> remote = remoteFiles(dir);
        assert(remote.size() == 1 && remote[0].rfind("net_", 0) == 0);
        assert(fs::is_directory(dir.file("selfplay-work")) && fs::is_empty(dir.file("selfplay-work")));
    }
    assert(remoteFiles(dir).empty());
    assert(!fs::exists(fs::path(dir.file("remote")) / runIdOf(dir.str())));   // the run's directory too
    // Leftovers of a crashed coordinator: an old network, a job, a result,
    // a part of an upload.
    const fs::path runRemote = fs::path(dir.file("remote")) / runIdOf(dir.str());
    fs::create_directories(runRemote);
    for (const char* name : {"net_0000000000000001.azn", "g0003_2_a1.job", "g0003_2_a1.result",
                             "net_0000000000000002.azn.part-1-1"})
        writeFile((runRemote / name).string(), "stale");
    resumeEngines(dir, {}, dir.file("hosts"))->runUntil(3);
    assert(remoteFiles(dir).empty());
}

// ---- failures: retry, fallback, stop ---------------------------------------------

static std::string manifestNote(const TempDir& dir, int row) {
    return csvFields(splitLines(readFile(dir.file("selfplay-engines.csv")))[static_cast<size_t>(row)])[10];
}

// One failure of any kind is retried on the same engine, with new file
// names: nothing of it shows in the result.
static void test_one_failure_is_retried() {
    TempDir plain("retry_default");
    startQuiet(tinyConfig(), plain)->runUntil(2);
    using Fail = distributed_probe::Fail;
    for (Fail mode : {Fail::Exit, Fail::Timeout, Fail::Corrupt}) {
        TempDir dir("retry_" + std::to_string(static_cast<int>(mode)));
        const Config c = withEngines(tinyConfig(), "local/naive:1, fake/naive:1");
        std::unique_ptr<Trainer> t =
            startEngines(c, dir, fakeHosts(dir, mode == Fail::Timeout ? "timeout=5" : ""));
        distributed_probe::failSlice("fake/naive", 1, mode);
        t->runUntil(2);
        distributed_probe::reset();
        assert(runFingerprint(plain, 2) == runFingerprint(dir, 2));
        for (int row = 1; row <= 4; ++row) assert(manifestNote(dir, row).empty());
        // The failed attempt's log is kept; its files are gone.
        assert(fs::exists(dir.file("selfplay-work/gen_0001/fake_naive.log")));
        for (const std::string& f : remoteFiles(dir)) assert(f.rfind("net_", 0) == 0);
    }
}

// Two failures under selfplay_fallback = local: the coordinator plays the
// slice itself, and the manifest and the log say so. local/naive has the
// fake remote's fingerprint, so here the files are still the default run's.
static void test_two_failures_fall_back() {
    TempDir plain("fallback_default"), dir("fallback");
    startQuiet(tinyConfig(), plain)->runUntil(2);
    std::unique_ptr<Trainer> t =
        startEngines(withEngines(tinyConfig(), "local/naive:1, fake/naive:1"), dir, fakeHosts(dir));
    std::ostringstream log;
    t->setLog(&log);
    distributed_probe::failSlice("fake/naive", 2, distributed_probe::Fail::Exit);
    t->runUntil(2);
    distributed_probe::reset();
    assert(runFingerprint(plain, 2) == runFingerprint(dir, 2));
    const std::vector<std::string> row = csvFields(splitLines(readFile(dir.file("selfplay-engines.csv")))[2]);
    assert(row[0] == "1" && row[3] == "fake/naive" && row[4] == "local/naive");
    // The note says why without a path or a host: details are in the slice's log.
    assert(row[5] == engineFingerprint("naive") && row[10] == "fallback: fake: selfplay exit 255");
    assert(manifestNote(dir, 4).empty());   // generation 2 tried the remote again, and it worked
    assert(log.str().find("note: generation 1: games 2-3 (fake/naive) played by local/naive: ") !=
           std::string::npos);
    assert(log.str().find("from here on results differ in the last bits from a run without the failure") !=
           std::string::npos);
}

// A fallback on another backend changes the games; the same failures
// reproduce them exactly (the manifest records who played what).
static void test_fallback_to_different_backend() {
    if (!blasAvailable()) return;
    TempDir plain("fallback_blas_default"), a("fallback_blas_a"), b("fallback_blas_b");
    // Resignation on: state.bin keeps the disabled games' search values,
    // whose last bits show which arithmetic played them even when no move
    // changed.
    const Config c = withEngines(resignConfig(), "local/blas:1, fake/naive:1");
    startEngines(c, plain, fakeHosts(plain))->runUntil(1);
    for (const TempDir* dir : {&a, &b}) {
        std::unique_ptr<Trainer> t = startEngines(c, *dir, fakeHosts(*dir));
        distributed_probe::failSlice("fake/naive", 2, distributed_probe::Fail::Corrupt);
        t->runUntil(1);
        distributed_probe::reset();
    }
    assert(readFile(statePath(a.str())) != readFile(statePath(plain.str())));
    assert(runFingerprint(a, 1) == runFingerprint(b, 1));
    assert(manifestWithoutTimings(a) == manifestWithoutTimings(b));
    assert(csvFields(splitLines(readFile(a.file("selfplay-engines.csv")))[2])[4] == "local/blas");
}

// Under stop, two failures end the generation before anything of it is
// committed; a resume replays it as planned, as if nothing had failed.
static void test_stop_keeps_last_good_state() {
    TempDir plain("stop_default"), dir("stop");
    const Config c = withEngines(tinyConfig(), "local/naive:1, fake/naive:1");
    Config stop = c;
    stop.selfplayFallback = "stop";
    startEngines(stop, plain, fakeHosts(plain))->runUntil(3);
    std::unique_ptr<Trainer> t = startEngines(stop, dir, fakeHosts(dir));
    t->runUntil(1);
    distributed_probe::failSlice("fake/naive", 2, distributed_probe::Fail::Exit);
    bool threw = false;
    try {
        t->runUntil(2);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("fake/naive") != std::string::npos;
    }
    distributed_probe::reset();
    assert(threw && t->generation() == 1);
    t.reset();
    assert(loadTrainingState(statePath(dir.str())).generation == 1);
    assert(manifestRows(dir) == 2);   // generation 1's only
    resumeEngines(dir, {}, dir.file("hosts"))->runUntil(3);
    assert(runFingerprint(plain, 3) == runFingerprint(dir, 3));
    assert(manifestWithoutTimings(plain) == manifestWithoutTimings(dir));
}

// A remote that cannot be reached at the start: stop refuses the run;
// local warns, and its slices fall back every generation.
static void test_unreachable_at_preflight() {
    TempDir stopDir("unreachable_stop"), localDir("unreachable_local");
    auto hostsWithoutExe = [](const TempDir& dir) {
        fs::create_directories(dir.path);
        writeFile(dir.file("hosts"), "fake exe=" + dir.file("no-such-selfplay") + " workdir=" + dir.file("remote") + "\n");
        return dir.file("hosts");
    };
    Config c = withEngines(tinyConfig(), "local/naive:1, fake/naive:1");
    c.selfplayFallback = "stop";
    bool threw = false;
    try {
        startEngines(c, stopDir, hostsWithoutExe(stopDir));
    } catch (const ConfigError& e) {
        threw = std::string(e.what()).find("fake") != std::string::npos;
    }
    assert(threw && !fs::exists(configPath(stopDir.str())));
    c.selfplayFallback = "local";
    TrainerOptions opts;
    opts.hostsFile = hostsWithoutExe(localDir);
    std::ostringstream startup;
    std::unique_ptr<Trainer> t = Trainer::startNew(c, localDir.str(), {}, &startup, opts);
    assert(startup.str().find("warning: fake unreachable (") != std::string::npos);
    assert(startup.str().find("its slices fall back to local/naive") != std::string::npos);
    t->setLog(nullptr);
    t->runUntil(2);
    // A machine that does not answer is not tried twice per generation: a
    // quick probe, then the fallback.
    assert(manifestNote(localDir, 2) == "fallback: fake unreachable");
    assert(manifestNote(localDir, 4) == "fallback: fake unreachable");
    // Two aliases for one machine's work directory would clean each
    // other's files away: refused.
    TempDir twin("twin_aliases");
    fs::create_directories(twin.path);
    writeFile(twin.file("hosts"), "fa exe=" + std::string(AZGO_SELFPLAY_EXE) + " workdir=" + twin.file("w") +
                                      "\nfb exe=" + std::string(AZGO_SELFPLAY_EXE) + " workdir=" + twin.file("w") +
                                      "\n");
    threw = false;
    try {
        startEngines(withEngines(tinyConfig(), "fa/naive:1, fb/naive:1"), twin, twin.file("hosts"));
    } catch (const ConfigError& e) {
        threw = std::string(e.what()).find("fa and fb") != std::string::npos;
    }
    assert(threw && !fs::exists(configPath(twin.str())));
}

// Different backends on one machine give different bits, but each game is
// still a function of its engine: two runs agree byte for byte.
// The hosts file's local line, written to dir/hosts; returns its path.
static std::string localHosts(const TempDir& dir, const std::string& settings) {
    fs::create_directories(dir.path);
    writeFile(dir.file("hosts"), "local " + settings + "\n");
    return dir.file("hosts");
}

static void test_mixed_local_engines_deterministic() {
    if (!blasAvailable()) return;
    TempDir a("mixed_a"), b("mixed_b");
    const Config c = withEngines(tinyConfig(), "local/naive:1, local/blas=3");
    // The engines' worker counts and QoS come from the hosts file (the
    // workers key is the default path's); neither changes a bit.
    startEngines(c, a, localHosts(a, "cpu_workers=3 cpu_qos=default"))->runUntil(2);
    startEngines(c, b, localHosts(b, "cpu_workers=1 cpu_qos=utility"))->runUntil(2);
    assert(runFingerprint(a, 2) == runFingerprint(b, 2));
    assert(manifestWithoutTimings(a) == manifestWithoutTimings(b));
    const std::string manifest = readFile(a.file("selfplay-engines.csv"));
    const std::vector<std::string> row = csvFields(splitLines(manifest)[2]);
    assert(row[3] == "local/blas" && row[5] == engineFingerprint("blas"));
}

int main() {
    test_single_local_engine_equals_default();
    test_split_local_engines_equal_default();
    test_engines_resume_is_byte_identical();
    test_switch_on_at_resume();
    test_fake_remote_equals_local();
    test_fake_remote_resume();
    test_fake_remote_unusual_run_dir();
    test_preflight_refusals();
    test_remote_cleanup();
    test_one_failure_is_retried();
    test_two_failures_fall_back();
    test_fallback_to_different_backend();
    test_stop_keeps_last_good_state();
    test_unreachable_at_preflight();
    test_partial_remote_network_is_replaced();
    test_resume_prunes_uncommitted_manifest_rows();
    test_mixed_local_engines_deterministic();
    test_rerate_ignores_uncommitted_checkpoint();
    test_metal_unavailable_writes_nothing();
    test_backend_switch_prints_note();
    test_train_threads_change_nothing();
    test_start_new_writes_run_files();
    test_two_generations();
    test_resume_is_byte_identical();
    test_thread_count_independence();
    test_resume_overrides();
    test_start_new_refuses_existing_run();
    test_gating_state_is_consistent();
    test_non_finite_weights_keep_last_good_state();
    test_progress_line_lists_opponents();
    test_resume_from_version1_state();
    test_resume_refuses_edited_bot_list();
    test_no_bots_no_extras();
    test_rerate_reproduces_live_results();
    test_rerate_refusals_and_gaps();
    test_start_from_network_and_buffer();
    test_start_from_refusals();
    test_ownership_run_trains_the_head();
    test_ownership_determinism();
    test_ownership_mismatches_are_refused();
    test_resume_run_from_before_ownership();
    test_reference_network_ids_and_files();
    test_reference_network_shapes_and_refusals();
    test_resume_refuses_changed_reference_networks();
    test_reference_network_determinism();
    test_rerate_with_reference_networks();
    test_random_symmetry_run();
    test_resignation_run();
    test_resignation_determinism();
    test_resignation_refusals_and_resume();
    test_resume_adds_resignation_columns();
    test_buffer_from_refuses_resigned_games_for_ownership();
    test_validation_loss_columns();
    test_validation_loss_with_ownership_head();
    test_validation_loss_changes_nothing();
    test_validation_loss_worker_independence();
    test_resume_adds_validation_columns();
    test_validation_loss_prints_nan();
    test_rounds_at_once_equal_one_at_a_time();
    test_generation_0_batch_norm_calibration();
    test_resume_and_bn_calibrate();
    std::printf("test_trainer: all tests passed\n");
    return 0;
}
