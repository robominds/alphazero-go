#!/usr/bin/env bash
# End-to-end smoke test of the four executables on a tiny 5x5 run:
# train -> resume -> arena -> analyze (good and bad input) -> bench, plus the
# refusal to change the architecture on resume.
#
# usage: tests/integration_smoke.sh BUILD_DIR SOURCE_DIR
set -euo pipefail

BUILD="$(cd "${1:?usage: integration_smoke.sh BUILD_DIR SOURCE_DIR}" && pwd)"
SRC="$(cd "${2:?usage: integration_smoke.sh BUILD_DIR SOURCE_DIR}" && pwd)"
W="$(mktemp -d "${TMPDIR:-/tmp}/azgo_smoke.XXXXXX")"
trap 'rm -rf "$W"' EXIT

fail() { echo "integration_smoke: FAILED: $*" >&2; exit 1; }

# The 5x5 preset shrunk to a network and budget that train in seconds, on the
# naive backend so the test also runs in builds without BLAS.
TINY=(--set backend=naive --set blocks=1 --set filters=8 --set value_hidden=8 --set history=2
      --set games_per_generation=4 --set simulations=8 --set train_steps=4
      --set batch_size=16 --set arena_games=2 --set arena_simulations=4 --set workers=2
      --set sgf_per_generation=1)
RUN="$W/run"
CK="$RUN/checkpoints"

echo "== train: new run, 2 generations =="
"$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$RUN" "${TINY[@]}" --generations 2
test -s "$CK/gen_0002.azn" || fail "gen_0002.azn missing"
test -s "$RUN/sgf/gen_0001_game_00.sgf" || fail "no SGF written for generation 1"

echo "== train: resume to generation 3 =="
"$BUILD/train" --run "$RUN" --resume --generations 3
test -s "$CK/gen_0003.azn" || fail "gen_0003.azn missing after resume"
[ "$(wc -l < "$RUN/log.csv" | tr -d ' ')" = "4" ] || fail "log.csv should have a header and 3 rows"

echo "== arena =="
out="$("$BUILD/arena" --a "$CK/gen_0000.azn" --b "$CK/gen_0003.azn" --games 4 --simulations 4 --workers 2)"
echo "$out"
echo "$out" | grep -Eq '^A .* scored' || fail "arena did not print a score line"

echo "== analyze: a legal position =="
out="$("$BUILD/analyze" --checkpoint "$CK/gen_0003.azn" --moves "C3 D4" --simulations 16)"
echo "$out"
echo "$out" | grep -q "MCTS value" || fail "analyze printed no MCTS value"

echo "== analyze: SGF input =="
printf '(;GM[1]FF[4]SZ[5]KM[7.5];B[cc];W[dd])\n' > "$W/ok.sgf"
"$BUILD/analyze" --checkpoint "$CK/gen_0003.azn" --sgf "$W/ok.sgf" --simulations 8 > /dev/null \
    || fail "analyze rejected a valid 5x5 SGF"

# Runs analyze with bad input; it must exit 1 (not crash, not assert) and
# print the expected message ("error: move K (<text>): <reason>" for a bad move).
expect_error() {
    local want="$1"; shift
    local status=0 out
    out="$("$BUILD/analyze" --checkpoint "$CK/gen_0003.azn" --simulations 4 "$@" 2>&1)" || status=$?
    [ "$status" -eq 1 ] || fail "analyze $* exited $status, expected 1; output: $out"
    echo "$out" | grep -qF -- "$want" || fail "analyze $* printed '$out', expected '$want'"
    echo "  ok: $want"
}

echo "== analyze: bad moves are user errors =="
expect_error "error: move 2 (C3): illegal (occupied)" --moves "C3 C3"
expect_error "error: move 1 (Z99): not a move on a 5x5 board" --moves "Z99"
# GTP column letters skip I, so I5 names no point.
expect_error "error: move 1 (I5): not a move on a 5x5 board" --moves "I5"

# Suicide. After C3 A2 C4 B1 (Black X, White O), Black to move:
#
#     A B C D E
#   5 . . . . .
#   4 . . X . .
#   3 . . X . .
#   2 O . . . .
#   1 . O . . .
#
# A1 is a one-point eye of White. A black stone there has no liberty and
# captures nothing (A2 keeps A3 and B2, B1 keeps B2 and C1): suicide.
expect_error "error: move 5 (A1): illegal (suicide)" --moves "C3 A2 C4 B1 A1"

# Superko (a simple ko). After B3 C3 C4 D4 C2 D2 A5 E3 (A5 is a spare Black
# move so that White gets E3 in before Black captures), Black to move:
#
#     A B C D E
#   5 X . . . .
#   4 . . X O .
#   3 . X O . O
#   2 . . X O .
#   1 . . . . .
#
# White C3 has one liberty, D3. Black D3 (move 9) captures it; the new D3
# stone is legal because it then has the liberty C3:
#
#   4 . . X O .
#   3 . X . X O
#   2 . . X O .
#
# White C3 (move 10) would capture D3 in turn and recreate exactly the
# position after move 8. Board::isLegal accepts it (it captures, so it is not
# suicide); only the positional-superko check in Game refuses it.
expect_error "error: move 10 (C3): illegal (superko)" --moves "B3 C3 C4 D4 C2 D2 A5 E3 D3 C3"
# The same sequence without the retake is fine: move 9 really is legal.
"$BUILD/analyze" --checkpoint "$CK/gen_0003.azn" --moves "B3 C3 C4 D4 C2 D2 A5 E3 D3" \
    --simulations 4 > /dev/null || fail "the ko capture D3 was rejected"

echo "== analyze: other bad input =="
printf '(;GM[1]FF[4]SZ[9]KM[7.5];B[cc])\n' > "$W/nine.sgf"
expect_error "error: the SGF is for 9x9" --sgf "$W/nine.sgf"
expect_error "error: move 3 (C3): game is over" --moves "pass pass C3"
expect_error "error: the game is over" --moves "pass pass"

echo "== bench =="
"$BUILD/bench" --sizes 5 --seconds 0.2 --configs "$SRC/configs"

echo "== train: resume must refuse an architecture change =="
status=0
out="$("$BUILD/train" --run "$RUN" --resume --set filters=16 --generations 4 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "resume with filters=16 exited $status, expected 1"
echo "$out" | grep -q "architecture" || fail "resume refusal does not mention architecture: $out"
test ! -e "$CK/gen_0004.azn" || fail "a refused resume still trained"

echo "integration_smoke: all checks passed"
