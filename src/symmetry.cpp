#include "azgo/symmetry.hpp"
#include <cassert>

namespace azgo {

int transformPoint(int sym, int point, int n) {
    assert(sym >= 0 && sym < kNumSymmetries);
    assert(point >= 0 && point < n * n);
    int row = point / n, col = point % n;
    if (sym & 4) col = n - 1 - col;               // mirror left-right first
    for (int turn = 0; turn < (sym & 3); ++turn) {
        // One clockwise quarter turn: the top row becomes the right column.
        const int r = col;
        const int c = n - 1 - row;
        row = r;
        col = c;
    }
    return row * n + col;
}

int transformAction(int sym, int action, int n) {
    if (action == n * n) return action;           // pass has no location
    return transformPoint(sym, action, n);
}

int inverseSymmetry(int sym) {
    assert(sym >= 0 && sym < kNumSymmetries);
    // Pure rotations: undo k clockwise turns with 4 - k more.
    // Mirrored ones: T = R^k M, and mirroring reverses the direction of a
    // rotation (M R^k = R^-k M), so T T = R^k (M R^k) M = R^k R^-k M M = I.
    // Every mirrored transform is its own inverse.
    if (sym & 4) return sym;
    return (4 - sym) % 4;
}

} // namespace azgo
