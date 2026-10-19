// Which source a binary was built from. Distributed self-play plays the
// games of one generation in several processes, possibly on several
// machines; a game is a function of the code that played it, so every
// binary carries a build ID, and the coordinator refuses a remote binary
// whose ID differs from its own. CMake computes the ID at build time
// (cmake/azgo_build_info.cmake): the git commit, plus a hash of the
// uncommitted changes when the tree is dirty, or the AZGO_BUILD_ID cache
// variable for a copy without .git.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once
#include <string>

namespace azgo {

// The full ID: 40 lowercase hex digits of the commit, followed by "+" and 8
// hex digits when the tree had uncommitted changes, or "unknown" when the
// build had neither git nor AZGO_BUILD_ID.
std::string buildId();
// For printing: the first 7 hex digits and the same suffix (e.g.
// "09e008b+3fa1c2d0"), or "unknown".
std::string shortBuildId();
// The strongest backend built in: "metal", "blas" or "naive".
std::string buildOptions();

} // namespace azgo
