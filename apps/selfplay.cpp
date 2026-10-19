// selfplay: plays one slice of a generation's self-play games from a job
// file and writes the verified-format result file. It is the whole remote
// interface of distributed self-play: the coordinator (train) uploads the
// network and the job to a machine, runs this there, and downloads and
// verifies the result. It needs no run directory.
//
//   selfplay --job JOB --out OUT --backend naive|blas|metal [--workers N] [--inference-batch N]
//   selfplay --version
//   selfplay --fingerprint --backend B
//
// Exit status: 0 on success, 1 on any error (the reason on stderr), 2 on a
// usage error.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "azgo/build_info.hpp"
#include "azgo/engines.hpp"
#include "azgo/selfplay_slice.hpp"
#include "cli.hpp"

namespace {

const char* const kUsage =
    "usage:\n"
    "  selfplay --job JOB --out OUT --backend naive|blas|metal [--workers N] [--inference-batch N]\n"
    "  selfplay --version\n"
    "  selfplay --fingerprint --backend B\n"
    "\n"
    "--job JOB plays the games of a self-play slice that JOB (a job file written by\n"
    "train) describes, with the network named in it, which must lie in JOB's directory\n"
    "and match the job's hash, and writes the result to OUT (replaced atomically).\n"
    "The job's build ID must be this binary's. --workers: threads (default: every game\n"
    "at once with metal, every core otherwise); --inference-batch: most positions per\n"
    "GPU pass (default: the job's). Neither changes a result.\n"
    "--version prints the build ID, short and full, and the strongest backend built.\n"
    "--fingerprint checks that backend B can run here and prints what determines its\n"
    "arithmetic (backend, CPU, OS, and with metal the GPU and shaders).\n"
    "Exit status: 0 success, 1 error, 2 usage error.\n";

int usageError(const std::string& message) {
    std::fprintf(stderr, "error: %s\n\n%s", message.c_str(), kUsage);
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    std::unique_ptr<ArgParser> args;
    try {
        const std::vector<std::string> values = {"job", "out", "backend", "workers", "inference-batch"};
        const std::vector<std::string> flags = {"help", "version", "fingerprint"};
        args = std::make_unique<ArgParser>(argc, argv, values, flags);
    } catch (const std::invalid_argument& e) {
        return usageError(e.what());
    }
    if (args->flag("help")) {
        std::fputs(kUsage, stdout);
        return 0;
    }
    if (args->flag("version")) {
        std::printf("selfplay %s (%s)\nbuild %s\n", azgo::shortBuildId().c_str(),
                    azgo::buildOptions().c_str(), azgo::buildId().c_str());
        return 0;
    }
    if (!args->has("backend")) return usageError("--backend is required");
    const std::string backend = args->get("backend");
    if (args->flag("fingerprint")) {
        try {
            std::printf("%s\n", azgo::engineFingerprint(backend).c_str());
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "selfplay: %s\n", e.what());
            return 1;
        }
    }
    if (!args->has("job") || !args->has("out")) return usageError("--job and --out are required");
    int workers = 0, inferenceBatch = 0;
    try {
        workers = args->getInt("workers", 0);
        inferenceBatch = args->getInt("inference-batch", 0);
    } catch (const std::invalid_argument& e) {
        return usageError(e.what());
    }
    if (workers < 0 || workers > 256) return usageError("--workers must be in 0..256");
    if (inferenceBatch < 0 || inferenceBatch > 1024)
        return usageError("--inference-batch must be in 0..1024");
    return azgo::runSliceJob(args->get("job"), args->get("out"), backend, workers, inferenceBatch,
                             std::cerr);
}
