#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>
#include "azgo/checkpoint.hpp"
#include "azgo/config.hpp"
#include "azgo/elo.hpp"
#include "azgo/network.hpp"
#include "azgo/run_state.hpp"
#include "azgo/serialize.hpp"
#include "azgo/trainer.hpp"

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
           "lr,replay_ratio,selfplay_s,train_s,arena_s,elo,winrate_vs_prev,distinct_vs_prev\n");
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

    {
        // The log row's last field is the same diversity count.
        const std::string log = readFile(dir.file("log.csv"));
        const size_t end = log.find_last_not_of('\n');
        const std::string row = log.substr(log.find_last_of('\n', end) + 1, end);
        const std::string last = row.substr(row.find_last_of(',') + 1);
        assert(!last.empty() && std::stoi(last) == st.distinctVsPrev);
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
    // reference set {0} (u32 count + one i32) and the empty bot list (u32 count).
    const std::string bytes = readFile(statePath(dir.str()));
    const TrainingState current = loadTrainingState(statePath(dir.str()));
    std::ostringstream with, without;
    BinaryWriter ww(with), wo(without);
    current.buffer.save(ww, true);
    current.buffer.save(wo, false);
    const size_t header = 24;   // magic, version, generation, best generation, samples trained
    assert(bytes.compare(header, with.str().size(), with.str()) == 0);
    std::string v1 = bytes.substr(0, header) + without.str() +
                     bytes.substr(header + with.str().size(), bytes.size() - header - with.str().size() - 12);
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
    assert(readFile(statePath(dir.str()))[4] == 3);   // saved as version 3
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

int main() {
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
    std::printf("test_trainer: all tests passed\n");
    return 0;
}
