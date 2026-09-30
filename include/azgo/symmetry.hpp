// The eight symmetries of the square board.
//
// Go's rules do not change if the board is rotated or mirrored, so a
// position and its 7 transformed copies have the same value, and their best
// moves correspond. Training exploits this by showing the network a random
// transform of each sample (8x more data for free). A transform is a
// permutation of points; pass is not a point and maps to itself.
#pragma once

namespace azgo {

constexpr int kNumSymmetries = 8;

// Symmetry s in [0, 8): s & 3 = number of clockwise quarter turns, s & 4 = mirror
// left-right first. 0 is the identity.
int transformPoint(int sym, int point, int n);
int transformAction(int sym, int action, int n);  // pass (n*n) maps to itself
int inverseSymmetry(int sym);                      // transformPoint(inverse(s), transformPoint(s, p)) == p

} // namespace azgo
