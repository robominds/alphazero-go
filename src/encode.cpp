#include "azgo/encode.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include "azgo/game.hpp"
#include "azgo/symmetry.hpp"

namespace azgo {

int numPlanes(int history) {
    return 2 * history + 2;
}

void encodePlanes(const std::vector<const uint8_t*>& stones, int n, Color toMove,
                  int H, int sym, float* out) {
    assert(n >= kMinBoardSize && n <= kMaxBoardSize);
    assert(H >= 1);
    const int area = n * n;
    std::fill(out, out + static_cast<size_t>(numPlanes(H)) * area, 0.0f);

    // Where each point lands under the symmetry, computed once per call.
    std::array<int, kMaxBoardSize * kMaxBoardSize> dest;
    for (int p = 0; p < area; ++p) dest[p] = transformPoint(sym, p, n);

    const uint8_t own = static_cast<uint8_t>(toMove);
    const uint8_t opp = static_cast<uint8_t>(opponent(toMove));
    const int available = std::min(H, static_cast<int>(stones.size()));
    for (int k = 0; k < available; ++k) {
        // "Own" is the player to move now, in every history plane, so the
        // network always sees the game from the perspective it must act in.
        float* ownPlane = out + static_cast<size_t>(2 * k) * area;
        float* oppPlane = ownPlane + area;
        const uint8_t* s = stones[static_cast<size_t>(k)];
        for (int p = 0; p < area; ++p) {
            if (s[p] == own) ownPlane[dest[p]] = 1.0f;
            else if (s[p] == opp) oppPlane[dest[p]] = 1.0f;
        }
    }
    // Planes 2H (Black to move) and 2H + 1 (White to move): exactly one is all ones.
    const int colorPlane = 2 * H + (toMove == Color::Black ? 0 : 1);
    float* plane = out + static_cast<size_t>(colorPlane) * area;
    std::fill(plane, plane + area, 1.0f);
}

void encodeBoards(const std::vector<const Board*>& history, int H, int sym, float* out) {
    assert(!history.empty());
    const int n = history[0]->size();
    const int area = n * n;
    const int count = std::min(H, static_cast<int>(history.size()));
    std::vector<uint8_t> bytes(static_cast<size_t>(count) * area);
    std::vector<const uint8_t*> stones(static_cast<size_t>(count));
    for (int k = 0; k < count; ++k) {
        assert(history[static_cast<size_t>(k)]->size() == n);
        uint8_t* dst = bytes.data() + static_cast<size_t>(k) * area;
        history[static_cast<size_t>(k)]->exportStones(dst);
        stones[static_cast<size_t>(k)] = dst;
    }
    encodePlanes(stones, n, history[0]->toMove(), H, sym, out);
}

void encodeGame(const Game& game, int H, int sym, float* out) {
    std::vector<const Board*> history;
    for (int k = 0; k < H && k <= game.moveCount(); ++k) {
        history.push_back(&game.boardAt(game.moveCount() - k));
    }
    encodeBoards(history, H, sym, out);
}

} // namespace azgo
