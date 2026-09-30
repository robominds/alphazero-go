// What the network sees.
//
// A position is fed to the network as a stack of n x n binary planes, always
// from the point of view of the player about to move: "my stones" and "their
// stones" for the current position and the H - 1 before it (history reveals
// ko and recent fights), then one plane that is all ones if Black is to move
// and one that is all ones if White is. That constant plane also lets a
// zero-padded 3x3 convolution tell the board edge apart from empty points.
#pragma once
#include <cstdint>
#include <vector>
#include "azgo/board.hpp"

namespace azgo {

class Game;

// 2H + 2: (own, opponent) stones for each of the last H positions, then a
// "black to move" plane and a "white to move" plane (exactly one is all ones).
int numPlanes(int history);

// stones[k] points to n*n Color bytes for the position k plies ago (k = 0 is
// current). stones.size() may be < H; missing positions are zero planes.
// "Own" means toMove (the side to move NOW) for every history plane.
// Writes numPlanes(H) * n * n floats to out ([plane][point]) with symmetry
// `sym` applied: value for point p lands at transformPoint(sym, p, n).
void encodePlanes(const std::vector<const uint8_t*>& stones, int n, Color toMove,
                  int H, int sym, float* out);

// history[0] is the current board, history[1] the previous one, ...
void encodeBoards(const std::vector<const Board*>& history, int H, int sym, float* out);

// Current position of a game, history taken from the game's earlier boards.
void encodeGame(const Game& game, int H, int sym, float* out);

} // namespace azgo
