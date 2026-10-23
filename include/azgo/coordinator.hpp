// The coordinator: plays one generation's self-play games on the engines
// that selfplay_engines lists, and its arena waves on the engines that
// arena_engines lists, and records who played what.
//
// assignGames cuts the games (a generation's self-play games, or one arena
// wave's games, numbered match-major) into one contiguous slice per engine.
// Each local engine (local/metal, local/blas, ...) plays its slice on its
// own thread, with its own pool of workers and evaluators, at the same time
// as the others: a GPU engine's hubs and a CPU engine's per-worker BLAS
// backends share nothing but the read-only networks. The slices are then
// merged by game index, never by the order they finished in, so the results
// depend only on the assignment and on what each engine computes. A remote
// engine's slice (a machine of the hosts file) is a job file sent through a
// Transport to that machine's selfplay binary, whose verified result file
// comes back.
//
// Every slice is written to the run's manifests, selfplay-engines.csv and
// arena-engines.csv: the engine planned for it, the engine that played it,
// that engine's fingerprint and build, and what it cost. With the seed, the
// manifests are the exact recipe for every game of the run.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <ostream>
#include <string>
#include <vector>
#include "azgo/arena.hpp"
#include "azgo/config.hpp"
#include "azgo/engines.hpp"
#include "azgo/network.hpp"
#include "azgo/selfplay.hpp"
#include "azgo/selfplay_slice.hpp"
#include "azgo/subprocess.hpp"

namespace azgo {

// One slice of one generation: the games, the engine the assignment
// planned and the one that played it (equal unless a slice fell back),
// that engine's fingerprint and build (shortBuildId), the statistics, and
// a note ("" or why another engine played it). An arena slice also has its
// wave (1, 2, ... in the generation); its games are wave indices.
struct SliceReport {
    Slice slice;
    std::string planned, engine, fingerprint, build, note;
    SliceStats stats;
    int wave = 0;   // arena slices only
};

// ---- transports ----------------------------------------------------------------
//
// How the coordinator reaches a machine's work directory and its selfplay
// binary. Remote paths are the coordinator's own (workdir/run-id/...);
// errors are SliceErrors, which the failure policy handles like a failed
// slice.
class Transport {
public:
    virtual ~Transport() = default;
    // Copies a local file to `remote`, creating its directory, atomically
    // (to a unique remote + ".part-..." name, then renamed).
    virtual void upload(const std::string& local, const std::string& remote) = 0;
    virtual bool exists(const std::string& remote) = 0;
    // The names of the files in a remote directory (none if it is missing).
    virtual std::vector<std::string> list(const std::string& remoteDir) = 0;
    // Runs the machine's selfplay binary with `args` (niced), its stdout
    // and stderr appended to the local file `log`; timeout and cancel as
    // runProcess. The process gets a held stdin (ProcessOptions::holdStdin):
    // with --exit-on-stdin-eof it stops when the coordinator is gone or the
    // connection drops.
    virtual ProcessResult run(const std::vector<std::string>& args, const std::string& log,
                              double timeout, const std::atomic<bool>* cancel) = 0;
    ProcessResult run(const std::vector<std::string>& args, const std::string& log,
                      double timeout) {
        return run(args, log, timeout, nullptr);
    }
    virtual void download(const std::string& remote, const std::string& local) = 0;
    // Starts the machine's selfplay with `args` (niced, as run()) as a
    // stream (azgo/selfplay_stream.hpp): the channel is its stdin and
    // stdout; its stderr is appended to the local file `log`.
    virtual std::unique_ptr<StreamChannel> openStream(const std::vector<std::string>& args,
                                                      const std::string& log) = 0;
    // Removes the files that exist; missing ones are no error.
    virtual void remove(const std::vector<std::string>& remote) = 0;
    // Best effort: stops the processes whose command line holds `pattern`,
    // a job file's run-id/name (unique to one slice attempt), after a
    // timeout or a cancel and before a retry, in case the remote outlived
    // its connection.
    virtual void killJob(const std::string& pattern) = 0;
    // Whether the machine answers at all, quickly (ssh with a 3 s connect
    // timeout running `true`; for this machine, whether exe exists).
    virtual bool reachable() = 0;
    // Removes a remote directory if it is empty (best effort).
    virtual void removeDirIfEmpty(const std::string& remoteDir) = 0;
    // The timeout of the small commands (mkdir, test, ls, rm, pkill); short
    // for the cleanup after a cancel, so a dead machine cannot hold it up.
    void setCommandTimeout(double seconds) { commandTimeout_ = seconds; }
protected:
    double commandTimeout_ = 60;
};

// A machine with ssh: scp for files and ssh for commands, each a child
// process (runProcess), never a local shell. The one string the remote
// shell parses is built from hosts-file values and coordinator paths, each
// checked against [A-Za-z0-9_./~@:+-] again here.
class SshTransport : public Transport {
public:
    explicit SshTransport(MachineSettings machine, std::string log = "");
    void upload(const std::string& local, const std::string& remote) override;
    bool exists(const std::string& remote) override;
    std::vector<std::string> list(const std::string& remoteDir) override;
    using Transport::run;
    ProcessResult run(const std::vector<std::string>& args, const std::string& log, double timeout,
                      const std::atomic<bool>* cancel) override;
    void download(const std::string& remote, const std::string& local) override;
    std::unique_ptr<StreamChannel> openStream(const std::vector<std::string>& args,
                                              const std::string& log) override;
    void remove(const std::vector<std::string>& remote) override;
    void killJob(const std::string& pattern) override;
    bool reachable() override;
    void removeDirIfEmpty(const std::string& remoteDir) override;
private:
    ProcessResult command(const std::vector<std::string>& words, double timeout);
    std::string why(const ProcessResult& r) const;
    MachineSettings m_;
    std::string log_;   // where scp's and ssh's own messages go ("" = /dev/null)
};

// A machine without ssh: this machine, the selfplay binary as a separate
// process, files copied with std::filesystem ("~/" is $HOME). Tests use it
// as a fake remote; it can also run a niced CPU engine here.
class SubprocessTransport : public Transport {
public:
    explicit SubprocessTransport(MachineSettings machine);
    void upload(const std::string& local, const std::string& remote) override;
    bool exists(const std::string& remote) override;
    std::vector<std::string> list(const std::string& remoteDir) override;
    using Transport::run;
    ProcessResult run(const std::vector<std::string>& args, const std::string& log, double timeout,
                      const std::atomic<bool>* cancel) override;
    void download(const std::string& remote, const std::string& local) override;
    std::unique_ptr<StreamChannel> openStream(const std::vector<std::string>& args,
                                              const std::string& log) override;
    void remove(const std::vector<std::string>& remote) override;
    void killJob(const std::string& pattern) override;
    bool reachable() override;
    void removeDirIfEmpty(const std::string& remoteDir) override;
protected:
    MachineSettings m_;
};

// For tests: SubprocessTransport's files, but run() calls runJob on
// the calling thread (the selfplay app's arguments, parsed here), and
// answers --version and --fingerprint as the app does; openStream runs
// runStreamJob on a thread of its own, over two pipes.
class InProcessTransport : public SubprocessTransport {
public:
    using SubprocessTransport::SubprocessTransport;
    using Transport::run;
    ProcessResult run(const std::vector<std::string>& args, const std::string& log, double timeout,
                      const std::atomic<bool>* cancel) override;
    std::unique_ptr<StreamChannel> openStream(const std::vector<std::string>& args,
                                              const std::string& log) override;
};

// SshTransport when the machine has ssh, else SubprocessTransport. `log`:
// where the ssh transport's own messages go.
std::unique_ptr<Transport> makeTransport(const MachineSettings& machine,
                                         const std::string& log = "");

// The argument vectors SshTransport runs, for tests. sshArgv runs the
// machine's selfplay with `args`:
//   ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15
//       -o ServerAliveCountMax=4 <ssh> 'exec nice -n <nice> <exe> <args...>'
// scpArgv copies local to remote (upload) or back. Both throw
// std::logic_error if any piece holds a character outside
// [A-Za-z0-9_./~@:+-] or the target or a path starts with '-': the
// hosts file refuses such values, so it means a bug, not bad input. The
// local path of scpArgv is not checked (it reaches no shell) but made
// absolute.
std::vector<std::string> sshArgv(const MachineSettings& machine,
                                 const std::vector<std::string>& args);
std::vector<std::string> scpArgv(const MachineSettings& machine, const std::string& local,
                                 const std::string& remote, bool upload);

// ---- the coordinator ---------------------------------------------------------------

class Coordinator {
public:
    // cfg.selfplayEngines and cfg.arenaEngines must not both be empty.
    // `hosts` gives each machine's settings (local's defaults when it has no
    // line). Progress and notes go to `log` (null: silent). runDir names the
    // run on remotes (runIdOf).
    Coordinator(const Config& cfg, const std::string& runDir, HostsFile hosts, std::ostream* log);
    // Removes the networks it uploaded (best effort): a remote keeps no
    // file of a run between two coordinators.
    ~Coordinator();

    // Before a run writes anything (and writing nothing), for the engines of
    // both lists:
    //  - local engines: checkBackendAvailable; for metal (cuda)
    //    checkMetalInferenceMemory (checkCudaInferenceMemory) for one
    //    two-slot hub per self-play item, and for arenaWaveNetworks(cfg)
    //    one-slot hubs per arena item;
    //  - remote machines: every alias must be in the hosts file (else
    //    ConfigError); `selfplay --version` there must report this binary's
    //    build ID (else ConfigError naming the alias and both IDs), and
    //    `selfplay --fingerprint --backend B` must succeed for each of its
    //    engines (a backend it cannot run: ConfigError). A machine that
    //    cannot be reached at all is a ConfigError with selfplay_fallback =
    //    stop, and a warning with local. Messages name the key (selfplay_engines
    //    or arena_engines) that lists the machine;
    //  - one line per engine, its fingerprint and build:
    //      engine mini1/metal: Apple M4; macOS 26.1 (25B78); Apple M4, ... [09e008b]
    //    and a note for an engine whose fingerprint differs from either
    //    manifest's last row for it (its games' last bits change from here).
    void preflight();

    // Generation g's `games` self-play games from `net` (the checkpoint at
    // networkPath), merged in index order. cfg.selfplayEngines must not be
    // empty. Local slices run in this process, each on its own thread; each
    // remote slice on a thread that uploads the network (unless that
    // machine has it) and the job, runs selfplay there, downloads the
    // result and verifies it (readSliceResult). Work files go to
    // runDir/selfplay-work/gen_NNNN and workdir/<run-id>/ on the machine,
    // and are removed after the merge (the network stays until a later
    // phase no longer needs it). Throws the first failure of any slice, by
    // list order, after every slice has stopped.
    std::vector<GameRecord> play(int g, const std::string& networkPath, const Network& net,
                                 const SelfPlayConfig& sp, int games);
    // The last play()'s slices, in list order.
    const std::vector<SliceReport>& reports() const { return reports_; }

    // Starts generation g's arena: its waves are numbered from 1 and its
    // reports start empty.
    void beginArena(int g);
    // One wave of the arena begun last (cfg.arenaEngines must not be empty):
    // assignGames(arena_engines, the wave's games) over its games numbered
    // match-major (waveSlice), every slice at once as play()'s are, local
    // ones with playArenaGames, remote ones as arena jobs (the wave's
    // networks are saved from memory, each slice's uploaded, readArenaResult
    // verifies the answer). Work files go to runDir/arena-work/gen_NNNN_wW.
    // Returns each match's result (matchResult of its games in index
    // order), which is exactly what playMatches gives when every engine
    // computes the same bits. Failures as play()'s; the fallback engine is
    // the first local/... item of arena_engines, else local/<backend>.
    std::vector<MatchResult> playArena(const std::vector<ArenaMatchSpec>& wave, const ArenaConfig& cfg);
    // The slices of every wave since beginArena, wave by wave, in list order.
    const std::vector<SliceReport>& arenaReports() const { return arenaReports_; }

    void setLog(std::ostream* log) { log_ = log; }

private:
    // What a remote attempt at a slice needs, whatever its kind: its file
    // base name (base + "_a<attempt>.job"), the job file's bytes, the local
    // files of the networks it names (each uploaded as net_<hash>.azn), and
    // how to read and keep a downloaded result (it fills the report's
    // fingerprint, build and stats, and throws SliceError if the result does
    // not verify).
    struct RemoteWork {
        std::string base, jobBytes;
        std::vector<std::pair<std::string, uint64_t>> networks;   // local path, fileHash
        std::function<void(const std::string& localResult, SliceReport& report)> accept;
    };
    // One slice of a phase (self-play, or an arena wave): its engine, its
    // games, the remote work when the engine is remote, and how to play it
    // here (a local engine, or the fallback), keeping the results.
    struct PhaseSlice {
        EngineSpec engine;
        Slice slice;
        RemoteWork remote;
        std::function<void(const EngineRun& run, SliceStats* stats)> playLocal;
    };
    std::unique_ptr<Transport> transportFor(const MachineSettings& m, const std::string& log) const;
    std::string remoteDir(const MachineSettings& m) const;
    // The key that lists a remote machine, for messages.
    std::string keyFor(const std::string& alias) const;
    void unreachable(const std::string& alias, const std::string& why);
    // Runs a phase's slices at once, one thread each, and returns their
    // reports in list order. `list` is the engine list they come from (its
    // first local item is the fallback engine); workDir must exist when a
    // slice is remote; `what` names the games in notes ("games", "arena wave
    // 1 games"). Throws the first failure by list order after every slice
    // stopped.
    std::vector<SliceReport> runSlices(int g, std::vector<PhaseSlice>& slices,
                                       const std::vector<EngineSpec>& list, const std::string& workDir,
                                       const std::string& what);
    // One attempt at a remote slice, start to verified result; fills
    // report. Throws SliceError on a failure, after removing the attempt's
    // files and process, and SliceCancelled once `stop` is set.
    void attemptRemote(int attempt, const EngineSpec& e, const RemoteWork& work,
                       const std::string& workDir, const std::atomic<bool>& stop, SliceReport& report);
    // A remote slice with the failure policy: two attempts, then with
    // selfplay_fallback = local the fallback engine plays it (report.engine,
    // fingerprint and note say so, and a note goes to the log), with stop a
    // SliceError.
    void playRemote(int g, PhaseSlice& slice, const std::string& fallback, const std::string& workDir,
                    const std::atomic<bool>& stop, const std::string& what, SliceReport& report);
    Config cfg_;
    std::string runDir_, runId_;
    HostsFile hosts_;
    std::ostream* log_;
    std::map<std::string, std::string> fingerprints_;   // engine name -> fingerprint
    std::map<std::string, std::string> remoteBuilds_;   // alias -> its full build ID
    std::set<std::string> unreachable_;                 // aliases preflight could not reach
    // What the coordinator knows of a remote machine. The map is filled by
    // runSlices before any slice thread starts and never changes while they
    // run; each entry's fields are touched only under its own mutex.
    struct MachineState {
        std::mutex mutex;              // its uploads and cleanup, one slice at a time
        bool contacted = false;        // stale files of an earlier coordinator cleared
        std::set<std::string> uploaded;   // network paths uploaded there
        std::set<std::string> keep;       // those the current phase's slices need
        bool suspect = false;          // unreachable at preflight or in an earlier phase
        bool down = false;             // found unreachable this phase: its slices fall back
    };
    MachineState& machine(const std::string& alias);   // must exist (runSlices made it)
    std::map<std::string, std::unique_ptr<MachineState>> machines_;   // by alias
    std::mutex logMutex_;                               // slices' notes, from their threads
    std::atomic<bool> failedSlices_{false};             // this phase: keep the logs
    std::vector<SliceReport> reports_;
    int arenaGeneration_ = 0, arenaWave_ = 0;
    std::vector<SliceReport> arenaReports_;
};

// The most networks an arena wave can hold, so the most hubs it runs at once
// on a GPU engine (bots need none): the calibrations when a run starts
// (generation 0 and every reference network), or a round (the new network,
// g-1, the strongest reference, arena_opponents more and, with gating, the
// best network).
int arenaWaveNetworks(const Config& cfg);

// How a run is named on remotes: its directory's name (characters outside
// [A-Za-z0-9_.-] become '_'), '-', and the 16 hex digits of the FNV-1a hash
// of its absolute path, so two runs of one name never share files.
std::string runIdOf(const std::string& runDir);

// Test hooks; a run's configuration can never reach them.
namespace distributed_probe {
// The next `times` attempts at the engine's remote slices fail: Exit as a
// failed run (ssh's exit 255), Timeout as a run that hangs past the
// machine's timeout, Corrupt as a result damaged after the download.
enum class Fail { Exit, Timeout, Corrupt };
void failSlice(const std::string& engine, int times, Fail mode);
// Preflight sees this as every remote's build ID ("" = the real one).
void setRemoteBuildId(const std::string& id);
// The engine's remote slices wait this long before running selfplay.
void delaySlice(const std::string& engine, double seconds);
// Machines without ssh run slices with InProcessTransport, not as processes.
void useInProcessTransport(bool on);
void reset();
} // namespace distributed_probe

// runDir/selfplay-engines.csv
std::string manifestPath(const std::string& runDir);

// Rewrites the manifest atomically: its rows of generations before g, then
// one row per report for generation g:
//   generation,first,count,planned,engine,fingerprint,build,seconds,positions,evaluations,note
// The fingerprint is always quoted (a metal one holds commas); a note is
// quoted when it holds a comma or a quote. Rows of g and later are dropped
// first, so a redone generation leaves one set of rows. seconds is, like
// log.csv's timings, not part of a run's byte-identity.
void writeManifest(const std::string& runDir, int g, const std::vector<SliceReport>& reports);

// Drops the manifest's rows of generations g and later, if it has any (a
// crash between the manifest and state.bin leaves rows of an uncommitted
// generation). No manifest: nothing. A resume calls it with
// g = the committed generation + 1, with engines on or off.
void pruneManifest(const std::string& runDir, int g);

// runDir/arena-engines.csv
std::string arenaManifestPath(const std::string& runDir);

// Rewrites the arena manifest atomically, as writeManifest the self-play
// one: its rows of generations before g, then one row per report:
//   generation,wave,first,count,planned,engine,fingerprint,build,seconds,positions,evaluations,note
// first and count are wave indices (a wave's games numbered match-major).
void writeArenaManifest(const std::string& runDir, int g, const std::vector<SliceReport>& reports);

// Drops the arena manifest's rows of generations g and later, as pruneManifest.
void pruneArenaManifest(const std::string& runDir, int g);

// "selfplay engines: local/metal 52 1512s 3.5k ev/s | local/blas 12 1490s 0.9k": each
// slice's engine, games, wall time and network evaluations per second, after
// `label` ("arena engines" for an arena wave's slices).
std::string enginesLine(const std::vector<SliceReport>& reports,
                        const std::string& label = "selfplay engines");

} // namespace azgo
