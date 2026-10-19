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

// The process groups of the children being waited for, for the signal
// handlers: a fixed table of atomics, which a handler may read.
constexpr int kMaxChildren = 256;
std::atomic<pid_t> gChildren[kMaxChildren];

int registerChild(pid_t pid) {
    for (int i = 0; i < kMaxChildren; ++i) {
        pid_t empty = 0;
        if (gChildren[i].compare_exchange_strong(empty, pid)) return i;
    }
    return -1;   // full: the child is still waited for, just not forwarded to
}

void unregisterChild(int slot) {
    if (slot >= 0) gChildren[slot] = 0;
}

extern "C" void forwardAndDie(int sig) {
    for (int i = 0; i < kMaxChildren; ++i) {
        const pid_t pid = gChildren[i].load();
        if (pid > 0) kill(-pid, SIGTERM);
    }
    signal(sig, SIG_DFL);
    raise(sig);
}

// Owns the spawn attributes and file actions, so every path frees them.
struct SpawnSetup {
    posix_spawnattr_t attr;
    posix_spawn_file_actions_t actions;
    int initialized = 0;
    SpawnSetup() {
        if (posix_spawnattr_init(&attr) == 0) ++initialized;
        if (posix_spawn_file_actions_init(&actions) == 0) ++initialized;
    }
    ~SpawnSetup() {
        if (initialized == 2) {
            posix_spawn_file_actions_destroy(&actions);
            posix_spawnattr_destroy(&attr);
        }
    }
};

} // namespace

ProcessResult runProcess(const std::vector<std::string>& argv, const std::string& logPath,
                         double timeoutSeconds, const std::string& stdinPath) {
    ProcessOptions options;
    options.log = logPath;
    options.timeout = timeoutSeconds;
    options.stdinPath = stdinPath;
    return runProcess(argv, options);
}

ProcessResult runProcess(const std::vector<std::string>& argv, const ProcessOptions& options) {
    if (argv.empty()) throw std::invalid_argument("runProcess: no program");
    ProcessResult result;
    SpawnSetup s;
    if (s.initialized != 2) {
        result.error = "posix_spawn setup failed";
        return result;
    }
    // A clean signal mask and default dispositions, in a new process group.
    sigset_t none, all;
    sigemptyset(&none);
    sigfillset(&all);
    short flags = POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETPGROUP;
#ifdef __APPLE__
    // Only the descriptors the file actions below set up reach the child;
    // nothing else of ours does (no other child's held stdin, above all).
    // Elsewhere the coordinator's own files would have to be close-on-exec.
    flags |= POSIX_SPAWN_CLOEXEC_DEFAULT;
#endif
    int pipeFds[2] = {-1, -1};
    if (options.holdStdin) {
        if (pipe(pipeFds) != 0) {
            result.error = std::string("pipe: ") + std::strerror(errno);
            return result;
        }
        fcntl(pipeFds[0], F_SETFD, FD_CLOEXEC);
        fcntl(pipeFds[1], F_SETFD, FD_CLOEXEC);
    }
    auto closePipe = [&](int which) {
        if (pipeFds[which] >= 0) close(pipeFds[which]);
        pipeFds[which] = -1;
    };
    const std::string in = options.stdinPath.empty() ? "/dev/null" : options.stdinPath;
    const std::string out = options.log.empty() ? "/dev/null" : options.log;
    int rc = posix_spawnattr_setsigmask(&s.attr, &none);
    if (rc == 0) rc = posix_spawnattr_setsigdefault(&s.attr, &all);
    if (rc == 0) rc = posix_spawnattr_setpgroup(&s.attr, 0);
    if (rc == 0) rc = posix_spawnattr_setflags(&s.attr, flags);
    if (rc == 0)
        rc = options.holdStdin ? posix_spawn_file_actions_adddup2(&s.actions, pipeFds[0], 0)
                               : posix_spawn_file_actions_addopen(&s.actions, 0, in.c_str(), O_RDONLY, 0);
    if (rc == 0)
        rc = posix_spawn_file_actions_addopen(&s.actions, 1, out.c_str(), O_WRONLY | O_CREAT | O_APPEND,
                                              0644);
    if (rc == 0) rc = posix_spawn_file_actions_adddup2(&s.actions, 1, 2);
    if (rc != 0) {
        closePipe(0);
        closePipe(1);
        result.error = std::string("posix_spawn setup: ") + std::strerror(rc);
        return result;
    }
    std::vector<char*> args;
    for (const std::string& a : argv) args.push_back(const_cast<char*>(a.c_str()));
    args.push_back(nullptr);

    pid_t pid = 0;
    rc = posix_spawnp(&pid, args[0], &s.actions, &s.attr, args.data(), environ);
    closePipe(0);   // the child has its copy; ours would hide the end of file
    if (rc != 0) {
        closePipe(1);
        result.error = "cannot run " + argv[0] + ": " + std::strerror(rc);
        return result;
    }
    const int slot = registerChild(pid);

    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - start).count(); };
    bool termSent = false, killSent = false;
    double termAt = 0;
    int pollMillis = 1;
    for (;;) {
        // Has it ended? Without reaping it yet, so its process group still
        // names it and cannot be reused before the group is finished below.
        siginfo_t info;
        std::memset(&info, 0, sizeof info);
        const int w = waitid(P_PID, static_cast<id_t>(pid), &info, WEXITED | WNOHANG | WNOWAIT);
        if (w == 0 && info.si_pid == pid) break;
        if (w < 0 && errno != EINTR) {
            result.error = std::string("waitid: ") + std::strerror(errno);
            break;
        }
        const double t = elapsed();
        const bool cancelled = options.cancel && options.cancel->load();
        if (!termSent && ((options.timeout > 0 && t >= options.timeout) || cancelled)) {
            kill(-pid, SIGTERM);   // the whole group: ssh, or a shell, and what it started
            termSent = true;
            termAt = t;
            if (cancelled) result.cancelled = true;
            else result.timedOut = true;
        } else if (termSent && !killSent && t >= termAt + kKillGraceSeconds) {
            kill(-pid, SIGKILL);
            killSent = true;
        }
        // Short processes are reaped at once; long ones are polled every 50 ms.
        std::this_thread::sleep_for(std::chrono::milliseconds(pollMillis));
        pollMillis = std::min(kPollMillis, pollMillis * 2);
    }
    // What is left of a group that was told to stop (a grandchild that
    // ignored SIGTERM) goes before the leader is reaped.
    if (termSent) kill(-pid, SIGKILL);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    unregisterChild(slot);
    closePipe(1);   // only now may a held child see the end of its stdin
    if (WIFEXITED(status) && !termSent) result.exitCode = WEXITSTATUS(status);
    if (WIFSIGNALED(status)) result.signal = WTERMSIG(status);
    return result;
}

void forwardTerminationSignals() {
    struct sigaction sa;
    std::memset(&sa, 0, sizeof sa);
    sa.sa_handler = forwardAndDie;
    sigemptyset(&sa.sa_mask);
    for (int sig : {SIGINT, SIGTERM, SIGHUP}) sigaction(sig, &sa, nullptr);
}

} // namespace azgo
