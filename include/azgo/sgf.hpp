// SGF (Smart Game Format) is the text format every Go program and viewer
// understands. A game is a tree of nodes "(;...;...)"; each node holds
// properties like B[cd] (Black plays column c, row d, counted from the top
// left) or SZ[9]. We write self-play games for viewing in e.g. Sabaki and
// read the main line of simple SGF files for analyze.
#pragma once
#include <string>
#include <vector>

namespace azgo {

// "(;GM[1]FF[4]SZ[n]KM[komi]RE[B+R|W+R|0]C[comment];B[dd];W[tt]...)".
// Point coordinates: column letter then row letter, 'a' = left/top.
// Pass is written as an empty value "B[]". RE uses the sign of resultForBlack
// with the Tromp-Taylor margin, e.g. "B+3.5", "W+7.5", "0".
std::string gameToSgf(int n, float komi, const std::vector<int>& moves,
                      float scoreForBlack, const std::string& comment = "");
// The same with RE given as written, e.g. "B+R" for a game White resigned.
std::string gameToSgf(int n, float komi, const std::vector<int>& moves,
                      const std::string& result, const std::string& comment = "");

struct SgfGame { int boardSize = 19; float komi = 0; std::vector<int> moves; };
// Reads SZ, KM and the main line of B[]/W[] moves (pass: "" or "tt" when n <= 19).
// Ignores other properties and variations. Throws std::runtime_error on
// malformed input or off-board coordinates.
SgfGame parseSgf(const std::string& text);

} // namespace azgo
