// train: runs the AlphaZero generation loop for a new or resumed run.
//
//   train --config configs/9x9.cfg --run runs/nine [--set key=value]... [--generations N]
//   train --config configs/9x9.cfg --run runs/ten --init-from CKPT [--buffer-from RUNDIR]
//   train --run runs/nine --resume [--set key=value]... [--generations N]
//   train --run runs/nine --rerate [--set key=value]...
//   train --version
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "azgo/build_info.hpp"
#include "azgo/config.hpp"
#include "azgo/run_state.hpp"
#include "azgo/trainer.hpp"
#include "cli.hpp"

namespace {

const char* const kUsage =
    "usage:\n"
    "  train --config FILE --run DIR [--set key=value]... [--generations N]\n"
    "  train --config FILE --run DIR --init-from CKPT [--buffer-from RUNDIR] [--set key=value]...\n"
    "  train --run DIR --resume [--set key=value]... [--generations N]\n"
    "  train --run DIR --rerate [--set key=value]...\n"
    "  train --version\n"
    "\n"
    "A new run copies FILE (plus overrides) to DIR/config.cfg. --resume continues\n"
    "from DIR/state.bin; tunable keys may be overridden, architecture keys may not.\n"
    "--init-from CKPT makes CKPT (whose shape must match the config) generation 0 of a\n"
    "new run, e.g. a network made by grow; --buffer-from RUNDIR seeds the replay\n"
    "buffer with that run's games. Both are for new runs only.\n"
    "--generations N is shorthand for --set generations=N (train until generation N).\n"
    "--rerate trains nothing: it replays the arena over DIR's checkpoints with the\n"
    "current opponent selection and writes DIR/rerate/ (results.csv, elo.csv,\n"
    "references.csv, log.csv). state.bin and config.cfg are left as they are;\n"
    "--set applies to the replay only and may change any key but the architecture\n"
    "keys, e.g. --set bot_simulations= --set reference_networks=CKPT to measure the\n"
    "run against other fixed players.\n"
    "--version prints the build ID (the git commit, with +HASH when the tree had\n"
    "uncommitted changes) and the strongest backend built, e.g. 'train 09e008b (metal)',\n"
    "then the full ID on a line 'build <40 hex>[+HASH]'.\n";

int usageError(const std::string& message) {
    std::fprintf(stderr, "error: %s\n\n%s", message.c_str(), kUsage);
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    std::unique_ptr<ArgParser> args;
    try {
        args = std::make_unique<ArgParser>(argc, argv,
                                           std::vector<std::string>{"config", "run", "set", "generations", "init-from", "buffer-from"},
                                           std::vector<std::string>{"resume", "rerate", "help", "version"});
    } catch (const std::invalid_argument& e) {
        return usageError(e.what());
    }
    if (args->flag("help")) {
        std::fputs(kUsage, stdout);
        return 0;
    }
    if (args->flag("version")) {
        std::printf("train %s (%s)\nbuild %s\n", azgo::shortBuildId().c_str(),
                    azgo::buildOptions().c_str(), azgo::buildId().c_str());
        return 0;
    }
    if (!args->has("run")) return usageError("--run DIR is required");
    const bool resume = args->flag("resume");
    if (args->flag("rerate")) {
        if (resume || args->has("config") || args->has("generations") || args->has("init-from") ||
            args->has("buffer-from"))
            return usageError("--rerate cannot be combined with --resume, --config, --generations, "
                              "--init-from or --buffer-from");
        try {
            const std::string runDir = args->get("run");
            std::cout << "re-rating " << runDir << " (results in " << runDir << "/rerate)" << std::endl;
            const int last = azgo::rerateRun(runDir, args->sets(), &std::cout);
            std::cout << "done: re-rated generations 1-" << last << " of " << runDir << std::endl;
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "error: %s\n", e.what());
            return 1;
        }
    }
    if (resume && args->has("config"))
        return usageError("--config cannot be combined with --resume (the run's config.cfg is used)");
    if (resume && (args->has("init-from") || args->has("buffer-from")))
        return usageError("--init-from and --buffer-from start a new run; they cannot be combined with "
                          "--resume (a resumed run keeps its network and buffer)");
    if (!resume && !args->has("config"))
        return usageError("a new run needs --config FILE (or pass --resume to continue one)");

    try {
        const std::string runDir = args->get("run");
        std::vector<std::string> overrides = args->sets();
        if (args->has("generations"))
            overrides.push_back("generations=" + std::to_string(args->getInt("generations", 0)));

        std::unique_ptr<azgo::Trainer> trainer;
        if (resume) {
            trainer = azgo::Trainer::resume(runDir, overrides);
        } else {
            azgo::Config cfg = azgo::loadConfigFile(args->get("config"));
            for (const std::string& o : overrides) azgo::applyOverride(cfg, o);
            azgo::Trainer::RunInit init;
            init.initFrom = args->get("init-from");
            init.bufferFrom = args->get("buffer-from");
            trainer = azgo::Trainer::startNew(cfg, runDir, init);
        }

        std::cout << (resume ? "resuming " : "starting ") << runDir << " at generation "
                  << trainer->generation() << "; effective config (saved to "
                  << azgo::configPath(runDir) << "):\n"
                  << azgo::configToString(trainer->config()) << std::endl;

        trainer->setLog(&std::cout);
        trainer->runUntil(trainer->config().generations);
        std::cout << "done: generation " << trainer->generation() << " in " << runDir << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
