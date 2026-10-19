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
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <string>
#include <vector>
#include "azgo/config.hpp"

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

} // namespace azgo
