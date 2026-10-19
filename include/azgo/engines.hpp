// Engines for distributed self-play. An engine is a machine and a backend,
// written "mini1/metal": the coordinator is the machine "local", every other
// alias names a Mac defined in a machine-local hosts file. The
// selfplay_engines key lists engines with their share of a generation's
// games; assignGames turns that list into contiguous slices of game
// indices, using integers only, so the assignment depends on nothing but
// the key and the number of games: never on timing, load or which machine
// answered first. That is what keeps a run with several engines
// reproducible: game i is always played by the same engine.
//
// How to reach a machine and how hard to drive it is not part of a run's
// result, so it lives outside the repository and the run, in a hosts file
// (~/.config/azgo/hosts by default), one line per machine:
//
//   local  cpu_workers=6 cpu_qos=utility
//   mini1  ssh=<user>@<address> exe=<repo>/build-metal/selfplay workdir=<repo>/runs/remote
//
// Values may hold only [A-Za-z0-9_./~@:+-], so they can be pasted into an
// ssh command line without any quoting.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <map>
#include <string>
#include <vector>
#include "azgo/config.hpp"
#include "azgo/selfplay_slice.hpp"

namespace azgo {

// True for a machine alias: [a-z][a-z0-9_-]{0,31}.
bool validAlias(const std::string& alias);

// "local/metal:50, local/blas=12, mini1/metal:10": items separated by ',',
// each machine/backend followed by ":weight" (an integer in [1, 1000000]) or
// "=games" (an integer in [0, 1000000]); whitespace around any separator is
// ignored. An empty or blank text is the empty list. Throws
// std::invalid_argument("engine 2: ...") naming the item and the reason: an
// empty item (so a trailing comma), a bad alias, a backend other than
// naive, blas or metal, or a bad number.
std::vector<EngineSpec> parseEngineList(const std::string& text);
// The canonical form: items joined by ", ", each machine/backend:weight or
// machine/backend=games. parseEngineList(printEngineList(x)) == x.
std::string printEngineList(const std::vector<EngineSpec>& engines);

// The games [first, first + count) of a generation that engine number
// `engine` (its position in the list) plays.
struct Slice {
    int engine = 0, first = 0, count = 0;
};

// Largest-remainder apportionment of `games` among the list:
//   1. fixed items get their counts; R = games - (their sum);
//   2. weighted items share R: item e gets floor(R * w_e / W), W the sum of
//      the weights, and the R - (sum of those) games left over go one each to
//      the items with the largest remainders (R * w_e mod W), ties to the
//      earlier item;
//   3. slices are contiguous in list order from 0; an item with no game
//      gets no slice.
// 64-bit integers throughout, so no rounding can differ between machines.
// Throws std::invalid_argument for a list checkConsistency refuses (fixed
// counts above `games`, or not equal to it with no weighted item).
std::vector<Slice> assignGames(const std::vector<EngineSpec>& engines, int games);

// One machine's line of the hosts file. Remotes need exe and workdir; one
// without ssh is run on this machine as a separate process (the subprocess
// transport: tests, or a niced CPU engine). The numbers are 0 for "the
// default": every game of the slice at once for metal_workers, the rule of
// engineWorkers for cpu_workers, the run's inference_batch, nice 0, no
// timeout. cpu_qos defaults to utility on local and default elsewhere.
struct MachineSettings {
    std::string alias, ssh, exe, workdir;
    int metalWorkers = 0, cpuWorkers = 0, inferenceBatch = 0, nice = 0;
    CpuQos cpuQos = CpuQos::Default;
    double timeout = 0;   // seconds; 0 = none
    bool local() const { return alias == "local"; }
};

struct HostsFile {
    std::string source;                                 // its path, for messages
    std::map<std::string, MachineSettings> machines;    // by alias
    // The machine's settings. "local" may be absent (its defaults); any
    // other alias must be defined, else ConfigError naming it and the file.
    MachineSettings settings(const std::string& alias) const;
};

// Parses the hosts file's text: per line an alias, then key=value pairs
// separated by whitespace; '#' starts a comment; blank lines are ignored.
// Keys: ssh, exe, workdir, metal_workers, cpu_workers, inference_batch,
// cpu_qos (default|utility), nice, timeout. Refuses, with ConfigError
// "<source>:<line>: <reason>", a bad alias, an alias or key repeated, an
// unknown key, a pair without '=', an empty value or one with a character
// outside [A-Za-z0-9_./~@:+-], a bad number, ssh on local, and a remote
// without exe or workdir.
HostsFile parseHostsText(const std::string& text, const std::string& source);
// The file at `path`, or an empty HostsFile when there is none (only
// remote engines need one).
HostsFile loadHostsFile(const std::string& path);
// Where the hosts file is: `cli` (train --hosts) if given, else
// $AZGO_HOSTS if set and not empty, else $HOME/.config/azgo/hosts ("" when
// HOME is not set either).
std::string resolveHostsPath(const std::string& cli);

// Worker threads for an engine on `machine` with a slice of sliceGames
// games, on a machine with `cores` cores. metal: metal_workers, or with 0
// every game of the slice at once (at most kMaxArenaWorkers, 256): a waiting
// worker costs only a stack, and a GPU engine with fewer workers than games
// plays in waves whose last under-fills the GPU. naive, blas: cpu_workers,
// or with 0 max(1, cores / 3) on local (most of the machine stays with the
// desktop and the GPU engine's searches) and max(1, cores - 2) on a remote
// (two cores for its GPU engine). Always at most sliceGames, at least 1.
int engineWorkers(const MachineSettings& machine, const std::string& backend, int sliceGames,
                  int cores);

// What determines the arithmetic of `backend` on this machine, as one line:
// the backend, the CPU model, the OS version and build, and with metal the
// GPU, OS and shader hash of metalDescription(), e.g.
//   metal; Apple M5 Max; macOS 27.0.1 (27A123); Apple M5 Max, macOS 27.0.1, shaders b7b31a09
// Two engines with equal fingerprints and equal build IDs compute the same
// bits. It names no host. Throws std::invalid_argument (checkBackendAvailable's
// message) for a backend that cannot run here.
std::string engineFingerprint(const std::string& backend);

} // namespace azgo
