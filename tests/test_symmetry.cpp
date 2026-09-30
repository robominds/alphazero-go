#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>
#include "azgo/symmetry.hpp"

using namespace azgo;

static int pt(int row, int col, int n) { return row * n + col; }

static void test_identity() {
    for (int n : {5, 9}) {
        for (int p = 0; p < n * n; ++p) assert(transformPoint(0, p, n) == p);
    }
    assert(inverseSymmetry(0) == 0);
}

static void test_hand_computed_images() {
    const int n = 5;
    // s = 1: one clockwise quarter turn. Top-left goes to top-right,
    // top-right to bottom-right.
    assert(transformPoint(1, pt(0, 0, n), n) == pt(0, 4, n));
    assert(transformPoint(1, pt(0, 4, n), n) == pt(4, 4, n));
    assert(transformPoint(1, pt(0, 1, n), n) == pt(1, 4, n));
    // s = 2: half turn. s = 3: counter-clockwise quarter turn.
    assert(transformPoint(2, pt(0, 1, n), n) == pt(4, 3, n));
    assert(transformPoint(3, pt(0, 1, n), n) == pt(3, 0, n));
    // s = 4: mirror left-right.
    assert(transformPoint(4, pt(0, 1, n), n) == pt(0, 3, n));
    // s = 5: mirror, then a clockwise turn = reflection in the anti-diagonal:
    // (r, c) -> (n-1-c, n-1-r).
    assert(transformPoint(5, pt(0, 1, n), n) == pt(3, 4, n));
    // s = 6: mirror, then a half turn = flip top-bottom.
    assert(transformPoint(6, pt(0, 1, n), n) == pt(4, 1, n));
    // s = 7: mirror, then three turns = transpose (r, c) -> (c, r).
    assert(transformPoint(7, pt(0, 1, n), n) == pt(1, 0, n));
    // The center of an odd board is fixed by everything.
    for (int s = 0; s < kNumSymmetries; ++s) assert(transformPoint(s, pt(2, 2, n), n) == pt(2, 2, n));
}

static void test_inverse_composes_to_identity() {
    for (int n : {5, 6, 9, 19}) {
        for (int s = 0; s < kNumSymmetries; ++s) {
            const int inv = inverseSymmetry(s);
            assert(inv >= 0 && inv < kNumSymmetries);
            for (int p = 0; p < n * n; ++p) {
                assert(transformPoint(inv, transformPoint(s, p, n), n) == p);
                assert(transformPoint(s, transformPoint(inv, p, n), n) == p);
            }
        }
    }
    assert(inverseSymmetry(1) == 3 && inverseSymmetry(3) == 1 && inverseSymmetry(2) == 2);
}

static void test_transform_is_bijection() {
    for (int n : {5, 6, 19}) {
        for (int s = 0; s < kNumSymmetries; ++s) {
            std::vector<int> hits(static_cast<size_t>(n * n), 0);
            for (int p = 0; p < n * n; ++p) {
                const int q = transformPoint(s, p, n);
                assert(q >= 0 && q < n * n);
                ++hits[static_cast<size_t>(q)];
            }
            for (int h : hits) assert(h == 1);
        }
    }
}

static void test_eight_distinct_images() {
    // An L of three and two stones in the corner has no symmetry of its own,
    // so its 8 images are pairwise different.
    const int n = 5;
    const std::vector<int> shape = {pt(0, 0, n), pt(0, 1, n), pt(0, 2, n), pt(1, 0, n)};
    std::vector<std::vector<int>> images;
    for (int s = 0; s < kNumSymmetries; ++s) {
        std::vector<int> img;
        for (int p : shape) img.push_back(transformPoint(s, p, n));
        std::sort(img.begin(), img.end());
        images.push_back(img);
    }
    for (int a = 0; a < kNumSymmetries; ++a) {
        for (int b = a + 1; b < kNumSymmetries; ++b) assert(images[a] != images[b]);
    }
}

static void test_transform_action_keeps_pass() {
    for (int n : {5, 9, 19}) {
        for (int s = 0; s < kNumSymmetries; ++s) {
            assert(transformAction(s, n * n, n) == n * n);
            assert(transformAction(s, 0, n) == transformPoint(s, 0, n));
        }
    }
}

int main() {
    test_identity();
    test_hand_computed_images();
    test_inverse_composes_to_identity();
    test_transform_is_bijection();
    test_eight_distinct_images();
    test_transform_action_keeps_pass();
    std::printf("test_symmetry: all tests passed\n");
    return 0;
}
