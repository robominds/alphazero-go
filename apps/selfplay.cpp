// selfplay: plays one slice of a generation's self-play games, or of an
// arena wave's games, from a job file and writes the verified-format result
// file. It is the whole remote interface of distributed self-play and the
// distributed arena: the coordinator (train) uploads the networks and the
// job to a machine, runs this there, and downloads and verifies the result.
// The job's first bytes say which kind it is. It needs no run directory.
//
//   selfplay --job JOB --out OUT --backend naive|blas|metal|cuda [--workers N] [--inference-batch N]
//            [--exit-on-stdin-eof]
//   selfplay --stream --job JOB --backend B [--workers N] [--inference-batch N]
//   selfplay --version
//   selfplay --fingerprint --backend B
//
// --stream plays the games of a self-play job that the coordinator names
// one at a time on stdin and writes each back on stdout as it ends
// (dynamic allocation, azgo/selfplay_stream.hpp).
//
// Exit status: 0 on success, 1 on any error (the reason on stderr), 2 on a
// usage error.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "azgo/arena_slice.hpp"
#include "azgo/build_info.hpp"
#include "azgo/engines.hpp"
#include "azgo/selfplay_slice.hpp"
#include "azgo/selfplay_stream.hpp"
#include "cli.hpp"

namespace {

const char* const kUsage =
    "usage:\n"
    "  selfplay --job JOB --out OUT --backend naive|blas|metal|cuda [--workers N] [--inference-batch N]\n"
    "           [--exit-on-stdin-eof]\n"
    "  selfplay --stream --job JOB --backend B [--workers N] [--inference-batch N]\n"
    "  selfplay --version\n"
    "  selfplay --fingerprint --backend B\n"
    "\n"
    "--job JOB plays the games of a self-play slice or of an arena slice that JOB (a\n"
    "job file written by train) describes, with the networks named in it, which must\n"
    "lie in JOB's directory and match the job's hashes, and writes the result to OUT\n"
    "(replaced atomically).\n"
    "The job's build ID must be this binary's. --workers: threads (default: every game\n"
    "at once with metal and cuda, max(1, min(2, cores - 2)) with naive and blas, the remote\n"
    "rule of the hosts file); --inference-batch: most positions per GPU pass (default:\n"
    "the job's). Neither changes a result. --exit-on-stdin-eof stops at once when\n"
    "stdin reaches its end: the coordinator holds it open while it waits.\n"
    "--stream plays the games of the self-play job JOB that stdin names, one command\n"
    "per line (`play <index>`, then `end`), on --workers threads, and writes each\n"
    "game to stdout as a checksummed frame as soon as it ends (train's dynamic\n"
    "allocation; the coordinator verifies every game). End of file before `end`\n"
    "stops it at once.\n"
    "--version prints the build ID, short and full, and the strongest backend built.\n"
    "--fingerprint checks that backend B can run here and prints what determines its\n"
    "arithmetic (backend, CPU, OS, and with metal or cuda the GPU and kernels).\n"
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
        const std::vector<std::string> flags = {"help", "version", "fingerprint",
                                                "exit-on-stdin-eof", "stream"};
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
    const bool stream = args->flag("stream");
    if (stream && (!args->has("job") || args->has("out") || args->flag("exit-on-stdin-eof")))
        return usageError("--stream takes --job and no --out (stdout carries the games; stdin's end "
                          "stops it)");
    if (!stream && (!args->has("job") || !args->has("out")))
        return usageError("--job and --out are required");
    if (args->flag("exit-on-stdin-eof")) {
        // The coordinator holds our stdin open while it waits for us (over
        // ssh too): end of file means it is gone, and so is any use for
        // the slice. A detached thread watches; the slice's normal end
        // exits the process with it.
        std::thread([] {
            char buf[256];
            for (;;) {
                const ssize_t got = read(0, buf, sizeof buf);
                if (got == 0 || (got < 0 && errno != EINTR)) {
                    std::fputs("selfplay: stdin closed (the coordinator is gone); stopping\n", stderr);
                    std::_Exit(1);
                }
            }
        }).detach();
    }
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
    if (stream) {
        // stdout is the coordinator's pipe: a write after it is gone must
        // fail, not kill the process before it can say why.
        std::signal(SIGPIPE, SIG_IGN);
        return azgo::runStreamJob(args->get("job"), backend, workers, inferenceBatch, 0, 1, std::cerr);
    }
    return azgo::runJob(args->get("job"), args->get("out"), backend, workers, inferenceBatch, std::cerr);
}
