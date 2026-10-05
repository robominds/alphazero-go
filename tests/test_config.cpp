#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include "azgo/config.hpp"

using namespace azgo;

static bool contains(const std::string& s, const std::string& part) {
    return s.find(part) != std::string::npos;
}

// Parses text as file "t.cfg" and returns the ConfigError message.
static std::string errorOf(const std::string& text) {
    try {
        parseConfigText(text, "t.cfg");
    } catch (const ConfigError& e) {
        return e.what();
    }
    assert(false && "expected ConfigError");
    return "";
}

static std::string overrideErrorOf(const std::string& setting) {
    Config cfg;
    try {
        applyOverride(cfg, setting);
    } catch (const ConfigError& e) {
        return e.what();
    }
    assert(false && "expected ConfigError");
    return "";
}

static std::string preset(const std::string& name) {
    return std::string(AZGO_SOURCE_DIR) + "/configs/" + name;
}

static void test_defaults() {
    Config c;
    assert(c.seed == 1 && c.generations == 100 && c.workers == 0 && c.backend == "naive");
    assert(c.boardSize == 9 && c.komi == 7.5f && c.maxMoves == 0);
    assert(c.history == 8 && c.blocks == 6 && c.filters == 64 && c.valueHidden == 64);
    assert(c.gamesPerGeneration == 100 && c.simulations == 200 && c.mctsBatch == 8);
    assert(c.cPuct == 1.5f && c.dirichletScale == 10.0f && c.noiseFraction == 0.25f);
    assert(c.temperatureMoves == 20);
    assert(c.bufferPositions == 200000 && c.trainSteps == 200 && c.batchSize == 256);
    assert(c.lrSchedule == "0:0.02" && c.momentum == 0.9f && c.weightDecay == 1e-4f);
    assert(c.arenaEvery == 1 && c.arenaGames == 20 && c.arenaOpponents == 2);
    assert((c.botSimulations == std::vector<int>{100}));
    assert(c.referenceNetworks.empty());
    assert(c.calibrationGames == 80 && c.promoteThreshold == 0.85f);
    assert(c.arenaSimulations == 100 && c.arenaTemperatureMoves == 4);
    assert(!c.gating && c.gatingThreshold == 0.55f && c.sgfPerGeneration == 2);

    // Empty text changes nothing.
    assert(configToString(parseConfigText("", "empty")) == configToString(c));

    NetConfig net = c.netConfig();
    assert(net.boardSize == 9 && net.history == 8 && net.blocks == 6);
    assert(net.filters == 64 && net.valueHidden == 64);
    assert(c.effectiveMaxMoves() == 2 * 9 * 9);
    c.maxMoves = 50;
    assert(c.effectiveMaxMoves() == 50);
    assert(c.effectiveWorkers() >= 1);
    c.workers = 3;
    assert(c.effectiveWorkers() == 3);
}

static void test_presets_load() {
    Config c5 = loadConfigFile(preset("5x5.cfg"));
    assert(c5.boardSize == 5 && c5.blocks == 3 && c5.filters == 32 && c5.valueHidden == 32);
    assert(c5.lrSchedule == "0:0.02,40:0.002" && c5.arenaTemperatureMoves == 2);

    Config c9 = loadConfigFile(preset("9x9.cfg"));
    assert(c9.boardSize == 9 && c9.blocks == 6 && c9.filters == 64 && c9.valueHidden == 64);
    assert(c9.lrSchedule == "0:0.02,90:0.002" && c9.temperatureMoves == 20);

    Config c19 = loadConfigFile(preset("19x19.cfg"));
    assert(c19.boardSize == 19 && c19.blocks == 10 && c19.filters == 128);
    assert(c19.valueHidden == 256 && c19.simulations == 400 && c19.arenaGames == 10);

    // Every preset uses the same rating-aware arena settings: no bots (a
    // 100-simulation bot saturates early) and no reference networks, which
    // name files of a particular machine's runs.
    for (const Config& c : {c5, c9, c19}) {
        assert(c.arenaOpponents == 2);
        assert(c.botSimulations.empty() && c.referenceNetworks.empty());
        assert(c.calibrationGames == 80 && c.promoteThreshold == 0.85f);
    }
    // The commented-out reference_networks line is there to be filled in.
    for (const char* name : {"5x5.cfg", "9x9.cfg", "19x19.cfg"}) {
        std::ifstream in(preset(name));
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        assert(contains(text, "\nbot_simulations =\n") && contains(text, "\n# reference_networks =\n"));
    }
}

static void test_missing_file() {
    bool threw = false;
    try {
        loadConfigFile("no/such/file.cfg");
    } catch (const ConfigError& e) {
        threw = contains(e.what(), "no/such/file.cfg");
    }
    assert(threw);
}

static void test_round_trip() {
    std::vector<Config> configs = {Config{}, loadConfigFile(preset("5x5.cfg")),
                                   loadConfigFile(preset("9x9.cfg")),
                                   loadConfigFile(preset("19x19.cfg"))};
    // Unusual values too: large seed, negative komi, gating on.
    Config odd;
    odd.seed = 18446744073709551615ull;
    odd.komi = -0.5f;
    odd.gating = true;
    odd.weightDecay = 3e-5f;
    configs.push_back(odd);

    for (const Config& c : configs) {
        std::string text = configToString(c);
        Config back = parseConfigText(text, "roundtrip");
        assert(configToString(back) == text);
    }
    Config back = parseConfigText(configToString(odd), "roundtrip");
    assert(back.seed == odd.seed && back.komi == odd.komi && back.gating);
    assert(back.weightDecay == odd.weightDecay);   // bit-identical float
    assert(contains(configToString(odd), "gating = true\n"));
    assert(contains(configToString(Config{}), "board_size = 9\n"));
    // Floats print in their shortest exact form, not "0.899999976".
    assert(contains(configToString(Config{}), "momentum = 0.9\n"));
    assert(contains(configToString(Config{}), "weight_decay = 0.0001\n"));
}

static void test_overrides() {
    Config c;
    applyOverride(c, "simulations=50");
    assert(c.simulations == 50);
    applyOverride(c, " komi = 6.5 ");
    assert(c.komi == 6.5f);
    applyOverride(c, "gating=on");
    assert(c.gating);

    // Everything else is untouched.
    Config expected;
    expected.simulations = 50;
    expected.komi = 6.5f;
    expected.gating = true;
    assert(configToString(c) == configToString(expected));

    // Base config: text only changes the keys it mentions.
    Config base;
    base.filters = 16;
    Config merged = parseConfigText("blocks = 2\n", "t.cfg", base);
    assert(merged.filters == 16 && merged.blocks == 2);

    std::string msg = overrideErrorOf("simulations=0");
    assert(contains(msg, "--set: simulations: "));
    msg = overrideErrorOf("nonsense=1");
    assert(contains(msg, "--set: nonsense: unknown key"));
    msg = overrideErrorOf("simulations");
    assert(contains(msg, "--set: expected key = value"));
}

static void test_architecture_diff() {
    Config a, b;
    assert(architectureDiff(a, b).empty());
    b.filters = 32;
    b.history = 4;
    b.simulations = 50;        // tunable: not reported
    b.lrSchedule = "0:0.1";    // tunable: not reported
    std::vector<std::string> diff = architectureDiff(a, b);
    assert((diff == std::vector<std::string>{"history", "filters"}));   // table order
    b = a;
    b.boardSize = 5;
    b.valueHidden = 8;
    b.blocks = 0;
    diff = architectureDiff(a, b);
    assert((diff == std::vector<std::string>{"board_size", "blocks", "value_hidden"}));
}

static void test_arena_keys() {
    // A list, printed without spaces whatever the input spacing.
    Config c = parseConfigText("bot_simulations = 50, 200\n", "t.cfg");
    assert((c.botSimulations == std::vector<int>{50, 200}));
    assert(contains(configToString(c), "bot_simulations = 50,200\n"));
    assert(configToString(parseConfigText(configToString(c), "rt")) == configToString(c));

    // Empty means no bots, in a file and as an override, and round-trips.
    Config none = parseConfigText("bot_simulations =\n", "t.cfg");
    assert(none.botSimulations.empty());
    assert(configToString(parseConfigText(configToString(none), "rt")) == configToString(none));
    Config o;
    applyOverride(o, "bot_simulations=");
    assert(o.botSimulations.empty());
    applyOverride(o, "bot_simulations=100000");
    assert((o.botSimulations == std::vector<int>{100000}));

    c = parseConfigText("calibration_games = 2\npromote_threshold = 1\narena_opponents = 0\n", "t.cfg");
    assert(c.calibrationGames == 2 && c.promoteThreshold == 1.0f && c.arenaOpponents == 0);
    c = parseConfigText("calibration_games = 10000\npromote_threshold = 0.5\narena_opponents = 32\n", "t.cfg");
    assert(c.calibrationGames == 10000 && c.promoteThreshold == 0.5f && c.arenaOpponents == 32);

    std::string msg = errorOf("calibration_games = 7");
    assert(contains(msg, "t.cfg:1: calibration_games: ") && contains(msg, "even"));
    assert(contains(errorOf("calibration_games = 0"), "t.cfg:1: calibration_games: "));
    assert(contains(errorOf("calibration_games = 10002"), "t.cfg:1: calibration_games: "));
    assert(contains(errorOf("promote_threshold = 0.4"), "t.cfg:1: promote_threshold: "));
    assert(contains(errorOf("promote_threshold = 1.01"), "t.cfg:1: promote_threshold: "));
    assert(contains(errorOf("arena_opponents = 33"), "t.cfg:1: arena_opponents: "));
    assert(contains(errorOf("arena_opponents = -1"), "t.cfg:1: arena_opponents: "));
}

// Frozen checkpoints as fixed references: a comma-separated list of paths,
// printed without spaces, empty by default; it sits right after bot_simulations.
static void test_reference_networks_key() {
    Config c = parseConfigText("reference_networks = runs/a/gen_0200.azn , /abs/b.azn\n", "t.cfg");
    assert((c.referenceNetworks == std::vector<std::string>{"runs/a/gen_0200.azn", "/abs/b.azn"}));
    const std::string text = configToString(c);
    assert(contains(text, "bot_simulations = 100\nreference_networks = runs/a/gen_0200.azn,/abs/b.azn\n"));
    assert(configToString(parseConfigText(text, "rt")) == text);
    assert(parseConfigText(text, "rt").referenceNetworks == c.referenceNetworks);

    // Empty means none, in a file and as an override, and round-trips.
    assert(contains(configToString(Config{}), "reference_networks = \n"));
    Config none = parseConfigText("reference_networks =\n", "t.cfg");
    assert(none.referenceNetworks.empty());
    assert(configToString(parseConfigText(configToString(none), "rt")) == configToString(none));
    Config o;
    applyOverride(o, "reference_networks=x.azn");
    assert((o.referenceNetworks == std::vector<std::string>{"x.azn"}));
    applyOverride(o, "reference_networks=");
    assert(o.referenceNetworks.empty());

    // An empty item (a stray comma) and a repeated path are rejected, naming the entry.
    for (const char* bad : {"a.azn,", ",a.azn", "a.azn,,b.azn", "a.azn, a.azn"})
        assert(contains(errorOf(std::string("reference_networks = ") + bad), "t.cfg:1: reference_networks: "));
    assert(contains(errorOf("reference_networks = a.azn,"), "reference network 2"));
    assert(contains(errorOf("reference_networks = a.azn, a.azn"), "duplicate"));

    // Reference network j is player -(bots + j), so the list is locked on
    // resume like bot_simulations, and listed after it.
    Config a, b;
    b.referenceNetworks = {"x.azn"};
    assert((resumeLockedDiff(a, b) == std::vector<std::string>{"reference_networks"}));
    assert(architectureDiff(a, b).empty());
    b.botSimulations = {};
    b.blocks = 2;
    assert((resumeLockedDiff(a, b) ==
            std::vector<std::string>{"blocks", "bot_simulations", "reference_networks"}));
}

// Review Focus 2: hand-edited bot lists.
static void test_bot_simulations_rejects_bad_lists() {
    for (const char* bad : {"0", "-5", "100001", "1e3", "abc", "100,", ",100", "100,,200",
                            "100 200", "100,100"}) {
        std::string msg = errorOf(std::string("bot_simulations = ") + bad);
        assert(contains(msg, "t.cfg:1: bot_simulations: "));
    }
    assert(contains(errorOf("bot_simulations = 100,100"), "duplicate"));
    assert(contains(errorOf("bot_simulations = 50,0"), "bot 2"));
}

static void test_resume_locked_diff() {
    Config a, b;
    assert(resumeLockedDiff(a, b).empty());
    // Tunable arena keys are not locked.
    b.arenaOpponents = 5;
    b.calibrationGames = 20;
    b.promoteThreshold = 0.9f;
    assert(resumeLockedDiff(a, b).empty());
    // bot_simulations is locked but is not an architecture key.
    b.botSimulations = {50};
    assert((resumeLockedDiff(a, b) == std::vector<std::string>{"bot_simulations"}));
    assert(architectureDiff(a, b).empty());
    b.filters = 32;
    assert((resumeLockedDiff(a, b) == std::vector<std::string>{"filters", "bot_simulations"}));
    b = a;
    b.botSimulations = {};
    assert((resumeLockedDiff(a, b) == std::vector<std::string>{"bot_simulations"}));
}

static void test_ownership_keys() {
    Config c;
    assert(!c.ownership && c.ownershipWeight == 1.5f && !c.netConfig().ownership);
    // ownership sits with the network keys, ownership_weight with the training keys.
    assert(contains(configToString(c), "value_hidden = 64\nownership = false\n"));
    assert(contains(configToString(c), "weight_decay = 0.0001\nownership_weight = 1.5\n"));

    c = parseConfigText("ownership = on\nownership_weight = 0\n", "t.cfg");
    assert(c.ownership && c.ownershipWeight == 0.0f && c.netConfig().ownership);
    assert(contains(configToString(c), "ownership = true\n"));
    assert(contains(configToString(c), "ownership_weight = 0\n"));
    assert(configToString(parseConfigText(configToString(c), "rt")) == configToString(c));
    assert(parseConfigText("ownership_weight = 10", "t.cfg").ownershipWeight == 10.0f);
    assert(contains(errorOf("ownership = maybe"), "t.cfg:1: ownership: "));
    assert(contains(errorOf("ownership_weight = -0.5"), "t.cfg:1: ownership_weight: "));
    assert(contains(errorOf("ownership_weight = 10.5"), "t.cfg:1: ownership_weight: "));

    // The head changes the parameter set, so ownership is an architecture key
    // (and NetConfig tells the two shapes apart); its weight is tunable.
    Config a, b;
    b.ownership = true;
    assert((architectureDiff(a, b) == std::vector<std::string>{"ownership"}));
    assert((resumeLockedDiff(a, b) == std::vector<std::string>{"ownership"}));
    assert(a.netConfig() != b.netConfig());
    b = a;
    b.ownershipWeight = 3.0f;
    assert(architectureDiff(a, b).empty() && resumeLockedDiff(a, b).empty());

    // A config.cfg written before these keys existed leaves ownership off,
    // and the presets do not turn it on.
    assert(!parseConfigText("filters = 96\n", "old.cfg").ownership);
    for (const char* name : {"5x5.cfg", "9x9.cfg", "19x19.cfg"}) {
        const Config p = loadConfigFile(preset(name));
        assert(!p.ownership && p.ownershipWeight == 1.5f);
    }
}

static void test_random_symmetry_keys() {
    Config c;
    assert(!c.mctsRandomSymmetry && !c.arenaRandomSymmetry);
    // Each sits with the search settings it changes: self-play's after
    // temperature_moves, the arena's after arena_temperature_moves.
    assert(contains(configToString(c), "temperature_moves = 20\nmcts_random_symmetry = false\n"));
    assert(contains(configToString(c), "arena_temperature_moves = 4\narena_random_symmetry = false\n"));

    c = parseConfigText("mcts_random_symmetry = on\narena_random_symmetry = 1\n", "t.cfg");
    assert(c.mctsRandomSymmetry && c.arenaRandomSymmetry);
    assert(contains(configToString(c), "mcts_random_symmetry = true\n"));
    assert(contains(configToString(c), "arena_random_symmetry = true\n"));
    assert(configToString(parseConfigText(configToString(c), "rt")) == configToString(c));
    Config o;
    applyOverride(o, "arena_random_symmetry=true");
    assert(o.arenaRandomSymmetry && !o.mctsRandomSymmetry);
    assert(contains(errorOf("mcts_random_symmetry = maybe"), "t.cfg:1: mcts_random_symmetry: "));
    assert(contains(errorOf("arena_random_symmetry = 2"), "t.cfg:1: arena_random_symmetry: "));

    // They change how search reads the network, not its shape or the
    // players' IDs: both are tunable on resume.
    Config a;
    assert(architectureDiff(a, c).empty() && resumeLockedDiff(a, c).empty());

    // A config.cfg written before these keys existed leaves both off, and
    // the presets do not turn them on.
    const Config old = parseConfigText("filters = 96\n", "old.cfg");
    assert(!old.mctsRandomSymmetry && !old.arenaRandomSymmetry);
    for (const char* name : {"5x5.cfg", "9x9.cfg", "19x19.cfg"}) {
        const Config p = loadConfigFile(preset(name));
        assert(!p.mctsRandomSymmetry && !p.arenaRandomSymmetry);
    }
    // The 9x9 preset, the main target, names both keys in comments.
    std::ifstream in(preset("9x9.cfg"));
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    assert(contains(text, "\n# mcts_random_symmetry = false\n"));
    assert(contains(text, "\n# arena_random_symmetry = false\n"));
}

static void test_errors_name_key_and_line() {
    std::string msg = errorOf("komi = 7.5\nfoo = 1\n");
    assert(contains(msg, "t.cfg:2") && contains(msg, "foo") && contains(msg, "unknown key"));

    msg = errorOf("board_size = 4");
    assert(contains(msg, "t.cfg:1: board_size: ") && contains(msg, "[5, 19]"));
    msg = errorOf("board_size = 20");
    assert(contains(msg, "t.cfg:1: board_size: "));

    msg = errorOf("\n\nsimulations = 12abc");
    assert(contains(msg, "t.cfg:3: simulations: ") && contains(msg, "12abc"));
    msg = errorOf("simulations =");
    assert(contains(msg, "t.cfg:1: simulations: "));
    msg = errorOf("komi = 1e999");
    assert(contains(msg, "t.cfg:1: komi: "));
    msg = errorOf("komi = nan");
    assert(contains(msg, "t.cfg:1: komi: "));
    msg = errorOf("seed = -1");
    assert(contains(msg, "t.cfg:1: seed: "));
    msg = errorOf("c_puct = 0");   // lower bound excluded
    assert(contains(msg, "t.cfg:1: c_puct: "));
    msg = errorOf("momentum = 1"); // upper bound excluded
    assert(contains(msg, "t.cfg:1: momentum: "));

    msg = errorOf("backend = gpu");
    assert(contains(msg, "t.cfg:1: backend: ") && contains(msg, "naive|blas"));
    msg = errorOf("lr_schedule = 5:0.1");   // first entry must be at generation 0
    assert(contains(msg, "t.cfg:1: lr_schedule: "));
    msg = errorOf("gating = maybe");
    assert(contains(msg, "t.cfg:1: gating: "));

    msg = errorOf("komi = 7.5\nsimulations 50\n");
    assert(contains(msg, "t.cfg:2: expected key = value"));
}

// Review Focus 5: hand-edited files.
static void test_hand_edited_files() {
    std::string text =
        "# my experiment\r\n"
        "\r\n"
        "board_size = 5   # small board\r\n"
        "   komi=6.5\r\n"
        "\t\r\n"
        "lr_schedule =  0:0.05,10:0.005  \r\n"
        "gating\t=\ton # tabs too\r\n";
    Config c = parseConfigText(text, "edited.cfg");
    assert(c.boardSize == 5);
    assert(c.komi == 6.5f);
    assert(c.lrSchedule == "0:0.05,10:0.005");
    assert(c.gating);

    // A key repeated in one file is an error that names the second line.
    std::string msg = errorOf("simulations = 10\r\nkomi = 7\r\nsimulations = 20\r\n");
    assert(contains(msg, "t.cfg:3: simulations: ") && contains(msg, "duplicate"));

    // Colors alternate in the arena, so an odd game count is rejected.
    msg = errorOf("arena_games = 7");
    assert(contains(msg, "t.cfg:1: arena_games: ") && contains(msg, "even"));
    assert(parseConfigText("arena_games = 8", "t.cfg").arenaGames == 8);
}

int main() {
    test_defaults();
    test_presets_load();
    test_missing_file();
    test_round_trip();
    test_overrides();
    test_architecture_diff();
    test_arena_keys();
    test_bot_simulations_rejects_bad_lists();
    test_resume_locked_diff();
    test_reference_networks_key();
    test_ownership_keys();
    test_random_symmetry_keys();
    test_errors_name_key_and_line();
    test_hand_edited_files();
    std::printf("test_config: all tests passed\n");
    return 0;
}
