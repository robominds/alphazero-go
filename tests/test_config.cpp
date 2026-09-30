#include <cassert>
#include <cmath>
#include <cstdio>
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
    assert(c.arenaEvery == 1 && c.arenaGames == 20 && c.arenaOpponents == 4);
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
    test_errors_name_key_and_line();
    test_hand_edited_files();
    std::printf("test_config: all tests passed\n");
    return 0;
}
