#include "azgo/subprocess.hpp"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <spawn.h>
#include <stdexcept>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char** environ;

namespace azgo {

namespace {

constexpr double kKillGraceSeconds = 5.0;
constexpr int kPollMillis = 50;

// Owns the spawn attributes and file actions, so every path frees them.
struct SpawnSetup {
    posix_spawnattr_t attr;
    posix_spawn_file_actions_t actions;
    SpawnSetup() {
        posix_spawnattr_init(&attr);
        posix_spawn_file_actions_init(&actions);
    }
    ~SpawnSetup() {
        posix_spawn_file_actions_destroy(&actions);
        posix_spawnattr_destroy(&attr);
    }
};

} // namespace

ProcessResult runProcess(const std::vector<std::string>& argv, const std::string& logPath,
                         double timeoutSeconds, const std::string& stdinPath) {
    if (argv.empty()) throw std::invalid_argument("runProcess: no program");
    ProcessResult result;
    SpawnSetup s;
    // A clean signal mask and default dispositions, in a new process group.
    sigset_t none, all;
    sigemptyset(&none);
    sigfillset(&all);
    posix_spawnattr_setsigmask(&s.attr, &none);
    posix_spawnattr_setsigdefault(&s.attr, &all);
    posix_spawnattr_setpgroup(&s.attr, 0);
    short flags = POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETPGROUP;
#ifdef __APPLE__
    // Only the descriptors the file actions below set up reach the child;
    // nothing else of ours does. (Elsewhere the coordinator's own files
    // would have to be close-on-exec.)
    flags |= POSIX_SPAWN_CLOEXEC_DEFAULT;
#endif
    posix_spawnattr_setflags(&s.attr, flags);
    const std::string in = stdinPath.empty() ? "/dev/null" : stdinPath;
    const std::string out = logPath.empty() ? "/dev/null" : logPath;
    posix_spawn_file_actions_addopen(&s.actions, 0, in.c_str(), O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&s.actions, 1, out.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    posix_spawn_file_actions_adddup2(&s.actions, 1, 2);
    std::vector<char*> args;
    for (const std::string& a : argv) args.push_back(const_cast<char*>(a.c_str()));
    args.push_back(nullptr);

    pid_t pid = 0;
    const int rc = posix_spawnp(&pid, args[0], &s.actions, &s.attr, args.data(), environ);
    if (rc != 0) {
        result.error = "cannot run " + argv[0] + ": " + std::strerror(rc);
        return result;
    }

    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - start).count(); };
    bool termSent = false, killSent = false;
    double termAt = 0;
    int status = 0;
    int pollMillis = 1;
    for (;;) {
        const pid_t done = waitpid(pid, &status, WNOHANG);
        if (done == pid) break;
        if (done < 0 && errno != EINTR) {
            result.error = std::string("waitpid: ") + std::strerror(errno);
            return result;
        }
        const double t = elapsed();
        if (timeoutSeconds > 0 && !termSent && t >= timeoutSeconds) {
            kill(-pid, SIGTERM);   // the whole group: ssh, or a shell, and what it started
            termSent = true;
            termAt = t;
            result.timedOut = true;
        } else if (termSent && !killSent && t >= termAt + kKillGraceSeconds) {
            kill(-pid, SIGKILL);
            killSent = true;
        }
        // Short processes are reaped at once; long ones are polled every 50 ms.
        std::this_thread::sleep_for(std::chrono::milliseconds(pollMillis));
        pollMillis = std::min(kPollMillis, pollMillis * 2);
    }
    if (WIFEXITED(status)) result.exitCode = result.timedOut ? -1 : WEXITSTATUS(status);
    if (WIFSIGNALED(status)) result.signal = WTERMSIG(status);
    // The group may outlive its leader (a child ignoring SIGTERM): finish it.
    if (killSent || termSent) kill(-pid, SIGKILL);
    return result;
}

} // namespace azgo
