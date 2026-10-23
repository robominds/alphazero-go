// Running a child process, for distributed self-play's transports: ssh and
// scp to reach a remote, or the selfplay binary itself on this machine.
// No shell is involved: the program and its arguments go to posix_spawnp as
// a vector, so nothing in them is ever interpreted. The child gets a clean
// signal mask, stdin from a file, /dev/null or a pipe this process holds,
// stdout and stderr appended to a log file, no other descriptor of ours,
// and its own process group, so a timeout or a cancel can stop it and
// everything it started.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <atomic>
#include <cstddef>   // on glibc, also its version macros
#include <memory>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <vector>

// Whether a child really gets no other descriptor of ours. macOS has
// POSIX_SPAWN_CLOEXEC_DEFAULT; glibc 2.34 and later have
// posix_spawn_file_actions_addclosefrom_np (close_range on Linux 5.9 and
// later, else a walk of /proc/self/fd). Elsewhere a descriptor this process
// opened without close-on-exec reaches the child (CMake warns at configure).
#if defined(__APPLE__) || \
    (defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 34)))
#define AZGO_SPAWN_CLOSES_INHERITED_FDS 1
#else
#define AZGO_SPAWN_CLOSES_INHERITED_FDS 0
#endif

namespace azgo {

inline constexpr bool kSpawnClosesInheritedFds = AZGO_SPAWN_CLOSES_INHERITED_FDS != 0;

// exitCode: the child's exit status, or -1 when it did not exit normally
// (killed by a signal, timed out, cancelled, or never started). signal:
// the signal that ended it, else 0. error: why it could not be started (or
// waited for), else "".
struct ProcessResult {
    int exitCode = -1;
    int signal = 0;
    bool timedOut = false;
    bool cancelled = false;
    std::string error;
};

struct ProcessOptions {
    std::string log;         // stdout and stderr appended here ("" = /dev/null)
    std::string stdinPath;   // the child's stdin ("" = /dev/null)
    double timeout = 0;      // seconds; 0 = none
    // When set (from another thread), the child is stopped as on a timeout.
    const std::atomic<bool>* cancel = nullptr;
    // stdin is a pipe whose writing end this process holds until the child
    // has ended: the child sees end of file only once this process is gone
    // (however it died), so `selfplay --exit-on-stdin-eof` never outlives its
    // coordinator, even across ssh. Overrides stdinPath.
    bool holdStdin = false;
};

// Runs argv (argv[0] looked up in PATH unless it holds a '/') and waits for
// it, polling every 50 ms (sooner at first). After the timeout, or once
// cancel is set, the child's process group gets SIGTERM, and SIGKILL 5 s
// later if it is still there; timedOut or cancelled is set. When the child
// ends after a SIGTERM, what is left of its group gets SIGKILL before the
// child is reaped. The child is always reaped. Throws std::invalid_argument
// for an empty argv.
ProcessResult runProcess(const std::vector<std::string>& argv, const ProcessOptions& options);
// The same with only a log, a timeout and stdin.
ProcessResult runProcess(const std::vector<std::string>& argv, const std::string& logPath,
                         double timeoutSeconds, const std::string& stdinPath = "");

// Both ends of a conversation with a running program: what a self-play
// stream (azgo/selfplay_stream.hpp) runs over, whether the program is a
// child process here, ssh to another machine, or (tests) a thread.
class StreamChannel {
public:
    virtual ~StreamChannel() = default;
    // Writes to the program's stdin; false once it cannot (it is gone).
    virtual bool send(const std::string& text) = 0;
    // What the program wrote to stdout since: up to `size` bytes, waiting
    // at most timeoutSeconds. > 0 bytes read, 0 at end of file, -1 when
    // nothing came in time.
    virtual long receive(char* buf, size_t size, double timeoutSeconds) = 0;
    // Stops the program now (its stdin closed, then SIGTERM to its process
    // group and SIGKILL 5 s later, as runProcess's timeout).
    virtual void stop() = 0;
    // Waits for the program to end and reaps it; once.
    virtual ProcessResult wait() = 0;
};

// A child process with a pipe to its stdin and one from its stdout, its
// stderr appended to `log` ("" = /dev/null), started as runProcess starts
// one: no shell, a clean signal mask, its own process group, no other
// descriptor of ours, and its group registered for
// forwardTerminationSignals. A child that cannot be started gives a
// channel whose receive() is at end of file and whose wait() carries the
// error. The destructor stops and reaps a child still running. send() and
// stop() may be called from another thread than receive() and wait().
class ChildProcess : public StreamChannel {
public:
    ChildProcess(const std::vector<std::string>& argv, const std::string& log);
    ~ChildProcess() override;
    ChildProcess(const ChildProcess&) = delete;
    ChildProcess& operator=(const ChildProcess&) = delete;
    bool send(const std::string& text) override;
    long receive(char* buf, size_t size, double timeoutSeconds) override;
    void stop() override;
    ProcessResult wait() override;
    pid_t pid() const { return pid_; }
private:
    pid_t pid_ = -1;
    std::mutex inMutex_;   // send and stop may come from different threads
    int in_ = -1, out_ = -1, slot_ = -1;
    std::atomic<bool> termSent_{false};
    std::atomic<double> termAt_{0};
    bool waited_ = false;
    ProcessResult result_;
};

// Writes all of `data` to fd, retrying after EINTR and short writes.
// Returns false on any error, a reader that is gone (EPIPE) included,
// without raising SIGPIPE: the signal is blocked on the calling thread for
// the write and a SIGPIPE it caused is taken back.
bool writeAll(int fd, const std::string& data);

// Installs handlers for SIGINT, SIGTERM and SIGHUP that send SIGTERM to
// the process group of every child runProcess is waiting for, then let the
// signal end this process as it would have. Children run in their own
// groups, so a Ctrl-C at the terminal would not reach them otherwise. A
// signal already ignored (nohup's SIGHUP, a background job's SIGINT) is left
// ignored. Call once, early, from a program's main (train does).
void forwardTerminationSignals();

} // namespace azgo
