// Running a child process, for distributed self-play's transports: ssh and
// scp to reach a remote, or the selfplay binary itself on this machine.
// No shell is involved: the program and its arguments go to posix_spawnp as
// a vector, so nothing in them is ever interpreted. The child gets a clean
// signal mask, stdin from a file or /dev/null, stdout and stderr appended
// to a log file, no other descriptor of ours, and its own process group, so
// a timeout can stop it and everything it started.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <string>
#include <vector>

namespace azgo {

// exitCode: the child's exit status, or -1 when it did not exit normally
// (killed by a signal, timed out, or never started). signal: the signal
// that ended it, else 0. error: why it could not be started, else "".
struct ProcessResult {
    int exitCode = -1;
    int signal = 0;
    bool timedOut = false;
    std::string error;
};

// Runs argv (argv[0] looked up in PATH unless it holds a '/') and waits for
// it, polling every 50 ms (sooner at first). logPath: where stdout and stderr are appended
// ("" = /dev/null); stdinPath: the child's stdin ("" = /dev/null).
// timeoutSeconds > 0: after that long the child's process group gets
// SIGTERM, and SIGKILL 5 s later if it is still there; timedOut is set. The
// child is always reaped. Throws std::invalid_argument for an empty argv.
ProcessResult runProcess(const std::vector<std::string>& argv, const std::string& logPath,
                         double timeoutSeconds, const std::string& stdinPath = "");

} // namespace azgo
