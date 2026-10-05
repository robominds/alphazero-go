#!/usr/bin/env bash
# End-to-end smoke test of the executables on a tiny 5x5 run:
# train -> resume (with the validation loss) -> re-rate (as live, then against a reference network) ->
# arena (checkpoint, bot, random symmetry, c_puct and MCTS batch, on both sides or one) -> analyze (good, random symmetry and
# bad input) -> bench -> grow (deeper, wider, and a run started from the grown network), plus the refusal to change the architecture on resume,
# then the ownership head: a run with it, grow --ownership, and analyze's ownership map,
# and resignation: a run with it, and its refusal alongside the ownership head.
#
# usage: tests/integration_smoke.sh BUILD_DIR SOURCE_DIR
set -euo pipefail

BUILD="$(cd "${1:?usage: integration_smoke.sh BUILD_DIR SOURCE_DIR}" && pwd)"
SRC="$(cd "${2:?usage: integration_smoke.sh BUILD_DIR SOURCE_DIR}" && pwd)"
W="$(mktemp -d "${TMPDIR:-/tmp}/azgo_smoke.XXXXXX")"
trap 'rm -rf "$W"' EXIT

fail() { echo "integration_smoke: FAILED: $*" >&2; exit 1; }
# col FILE NAME: the field under column NAME in the last row of CSV FILE.
col() { awk -F, -v name="$2" 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == name) c = i } END { print $c }' "$1"; }

# The 5x5 preset shrunk to a network and budget that train in seconds, on the
# naive backend so the test also runs in builds without BLAS.
TINY=(--set backend=naive --set blocks=1 --set filters=8 --set value_hidden=8 --set history=2
      --set games_per_generation=4 --set simulations=8 --set train_steps=4
      --set batch_size=16 --set arena_games=2 --set arena_simulations=4 --set workers=2
      --set sgf_per_generation=1)
RUN="$W/run"
CK="$RUN/checkpoints"

echo "== train: new run, 2 generations =="
# The presets have no bot; this run adds one so bots stay covered end to end.
"$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$RUN" "${TINY[@]}" --set bot_simulations=100 --generations 2
test -s "$CK/gen_0002.azn" || fail "gen_0002.azn missing"
test -s "$RUN/sgf/gen_0001_game_00.sgf" || fail "no SGF written for generation 1"
grep -q '^-1,bot,100,0,' "$RUN/references.csv" || fail "references.csv does not list the calibrated bot"
grep -q '^-1,0,80,' "$RUN/results.csv" || fail "results.csv has no bot calibration games"

echo "== train: resume to generation 3 =="
"$BUILD/train" --run "$RUN" --resume --generations 3 | tee "$W/resume.out"
test -s "$CK/gen_0003.azn" || fail "gen_0003.azn missing after resume"
[ "$(wc -l < "$RUN/log.csv" | tr -d ' ')" = "4" ] || fail "log.csv should have a header and 3 rows"
# The validation loss is on by default: on the fresh games and on a buffer sample.
grep -Eq '^gen 3 \|.* val p [0-9.]+ v [0-9.]+ seen p [0-9.]+ v [0-9.]+ ' "$W/resume.out" \
    || fail "the progress line does not report the validation loss"
for c in val_policy_loss val_value_loss seen_policy_loss seen_value_loss; do
    col "$RUN/log.csv" $c | grep -Eq '^[0-9]+\.[0-9]{5}$' || fail "log.csv has no $c"
done

echo "== train: re-rate the finished run =="
state_before="$(cksum < "$RUN/state.bin")"
"$BUILD/train" --run "$RUN" --rerate
# Gating is off, so the replay plays exactly the live run's arena games.
cmp "$RUN/results.csv" "$RUN/rerate/results.csv" || fail "rerate/results.csv differs from the live run's"
cmp "$RUN/references.csv" "$RUN/rerate/references.csv" || fail "rerate/references.csv differs"
[ "$(wc -l < "$RUN/rerate/log.csv" | tr -d ' ')" = "4" ] || fail "rerate/log.csv should have a header and 3 rows"
[ "$(cksum < "$RUN/state.bin")" = "$state_before" ] || fail "--rerate changed state.bin"
status=0
out="$("$BUILD/train" --run "$RUN" --rerate --resume 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "--rerate --resume exited $status, expected 1"
echo "$out" | grep -q -- "--rerate cannot be combined" || fail "unexpected refusal message: $out"

echo "== train: re-rate against a reference network instead of the bot =="
out="$("$BUILD/train" --run "$RUN" --rerate --set bot_simulations= --set "reference_networks=$CK/gen_0001.azn")"
echo "$out"
echo "$out" | grep -q '^rerate calibration | ref1 vs 0: ' || fail "no calibration line for ref1"
grep -q '^-1,network,,0,' "$RUN/rerate/references.csv" || fail "rerate/references.csv does not list ref1"
grep -qx -- "-1,ref1,$CK/gen_0001.azn" "$RUN/rerate/reference-networks.csv" || fail "reference-networks.csv does not map -1 to its file"
[ "$(cksum < "$RUN/state.bin")" = "$state_before" ] || fail "--rerate changed state.bin"

echo "== arena =="
out="$("$BUILD/arena" --a "$CK/gen_0000.azn" --b "$CK/gen_0003.azn" --games 4 --simulations 4 --workers 2)"
echo "$out"
echo "$out" | grep -Eq '^A .* scored' || fail "arena did not print a score line"

echo "== arena: against the uniform bot =="
out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b-bot 8 --games 4 --simulations 4 --workers 2)"
echo "$out"
echo "$out" | grep -q 'vs B bot(8 simulations): A scored' || fail "arena --b-bot printed no score line"

echo "== arena: per-player simulations =="
out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0000.azn" --a-simulations 2 --b-simulations 16 --games 4 --simulations 4 --workers 2)"
echo "$out"
echo "$out" | grep -q '(2 simulations) vs B .*(16 simulations): A scored' || fail "arena --a/--b-simulations did not name both counts"

echo "== arena: random symmetry =="
out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0000.azn" --random-symmetry --games 4 --simulations 4 --workers 2)"
echo "$out"
echo "$out" | grep -Eq '^A .* scored' || fail "arena --random-symmetry did not print a score line"
echo "$out" | grep -q '^Search: random symmetry' || fail "arena --random-symmetry did not say so"

echo "== arena: random symmetry for one side =="
out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0003.azn" --a-random-symmetry --games 4 --simulations 4 --workers 2)"
echo "$out"
echo "$out" | grep -Eq '^A .* scored' || fail "arena --a-random-symmetry did not print a score line"
echo "$out" | grep -qx 'Search: random symmetry per network evaluation, A only' || fail "arena --a-random-symmetry did not say A only"
out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0000.azn" --b-random-symmetry --games 4 --simulations 4 --workers 2)"
echo "$out" | grep -qx 'Search: random symmetry per network evaluation, B only' || fail "arena --b-random-symmetry did not say B only"
# Both one-sided flags are --random-symmetry: the same games, the same lines.
both="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0000.azn" --a-random-symmetry --b-random-symmetry --games 4 --simulations 4 --workers 2)"
all="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0000.azn" --random-symmetry --games 4 --simulations 4 --workers 1)"
[ "$both" = "$all" ] || fail "arena --a- plus --b-random-symmetry differs from --random-symmetry: '$both' vs '$all'"
echo "$both" | grep -qx 'Search: random symmetry per network evaluation' || fail "arena with both one-sided flags did not say both"
# Against a bot only A has a network, so --random-symmetry is A's alone.
bot="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b-bot 8 --a-random-symmetry --games 4 --simulations 4 --workers 2)"
[ "$bot" = "$("$BUILD/arena" --a "$CK/gen_0003.azn" --b-bot 8 --random-symmetry --games 4 --simulations 4 --workers 2)" ] \
    || fail "arena --a-random-symmetry against a bot differs from --random-symmetry"
echo "$bot" | grep -qx 'Search: random symmetry per network evaluation, A only' || fail "arena against a bot did not say A only"

echo "== arena: c_puct and MCTS batch, for both sides or one =="
plain="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0003.azn" --games 4 --simulations 4 --workers 2)"
if echo "$plain" | grep -q '^Search: c_puct'; then fail "arena with the default search settings printed a settings line"; fi
# The defaults spelled out are the default match, line for line.
same="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0003.azn" --c-puct 1.5 --mcts-batch 8 \
    --a-c-puct 1.5 --b-mcts-batch 8 --games 4 --simulations 4 --workers 1)"
[ "$same" = "$plain" ] || fail "arena with the default search settings spelled out differs: '$same' vs '$plain'"
out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0003.azn" --a-c-puct 3 --b-mcts-batch 1 --games 4 --simulations 4 --workers 2)"
echo "$out"
echo "$out" | grep -q '^A .*(c_puct 3) vs B .*(mcts batch 1): A scored' || fail "arena did not name the one-sided settings"
echo "$out" | grep -qx 'Search: c_puct A 3, B 1.5; mcts batch A 8, B 1' || fail "arena did not print the per-side settings"
out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0000.azn" --c-puct 2 --mcts-batch 1 --games 4 --simulations 4 --workers 2)"
echo "$out" | grep -qx 'Search: c_puct 2; mcts batch 1' || fail "arena did not print the shared settings"
echo "$out" | grep -q '^A [^(]* vs B [^(]*: A scored' || fail "arena labelled a side for settings both share"
# A bot searches with c_puct and a batch too, so B's settings reach it.
out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b-bot 8 --b-c-puct 3 --b-mcts-batch 1 --games 4 --simulations 4 --workers 2)"
echo "$out" | grep -q 'vs B bot(8 simulations) (c_puct 3, mcts batch 1): A scored' || fail "arena did not name the bot's settings"
# Random symmetry and the other settings share one Search: line, the first
# line of the output (printed before the match starts).
out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0003.azn" --a-random-symmetry --b-c-puct 2 --games 4 --simulations 4 --workers 2)"
[ "$(echo "$out" | grep -c '^Search:')" -eq 1 ] || fail "arena printed more than one Search: line: '$out'"
[ "$(echo "$out" | head -n 1)" = 'Search: random symmetry per network evaluation, A only; c_puct A 1.5, B 2; mcts batch 8' ] \
    || fail "arena did not start with the combined Search: line: '$out'"

# B is exactly one of a checkpoint and a bot, and a bot must search.
expect_arena_error() {
    local want="$1"; shift
    local status=0 out
    out="$("$BUILD/arena" --a "$CK/gen_0003.azn" "$@" 2>&1)" || status=$?
    [ "$status" -eq 1 ] || fail "arena $* exited $status, expected 1; output: $out"
    echo "$out" | grep -qF -- "$want" || fail "arena $* printed '$out', expected '$want'"
    echo "  ok: $want"
}
expect_arena_error "give exactly one of --b CKPT and --b-bot SIMS" --b "$CK/gen_0000.azn" --b-bot 8
expect_arena_error "give exactly one of --b CKPT and --b-bot SIMS"
expect_arena_error "--b-bot must be in [1, 100000]" --b-bot 0
expect_arena_error "--b-simulations applies to a checkpoint B" --b-bot 8 --b-simulations 16
expect_arena_error "--b-simulations must be in [1, 100000]" --b "$CK/gen_0000.azn" --b-simulations 0
expect_arena_error "--b-random-symmetry applies to a checkpoint B" --b-bot 8 --b-random-symmetry
expect_arena_error "--c-puct must be in (0, 100]" --b-bot 8 --c-puct 0
expect_arena_error "--a-c-puct must be in (0, 100]" --b-bot 8 --a-c-puct 101
expect_arena_error "--b-c-puct must be in (0, 100]" --b-bot 8 --b-c-puct -1
expect_arena_error "option --c-puct expects a number" --b-bot 8 --c-puct nan
expect_arena_error "--a-c-puct must be in (0, 100]" --b-bot 8 --a-c-puct 1e-50   # 0 as a float
expect_arena_error "--c-puct must be in (0, 100]" --b-bot 8 --c-puct 1e-50
expect_arena_error "--mcts-batch must be in [1, 256]" --b-bot 8 --mcts-batch 0
expect_arena_error "--a-mcts-batch must be in [1, 256]" --b-bot 8 --a-mcts-batch 257
expect_arena_error "--b-mcts-batch must be in [1, 256]" --b "$CK/gen_0000.azn" --b-mcts-batch 0

echo "== analyze: a legal position =="
out="$("$BUILD/analyze" --checkpoint "$CK/gen_0003.azn" --moves "C3 D4" --simulations 16)"
echo "$out"
echo "$out" | grep -q "MCTS value" || fail "analyze printed no MCTS value"

echo "== analyze: random symmetry =="
out="$("$BUILD/analyze" --checkpoint "$CK/gen_0003.azn" --moves "C3 D4" --simulations 16 --random-symmetry)"
echo "$out" | grep -q "MCTS value .* with random symmetry" || fail "analyze --random-symmetry did not say so"

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

echo "== grow: deepen the tiny network, then use the grown checkpoint =="
out="$("$BUILD/grow" --from "$CK/gen_0003.azn" --blocks 3 --out "$W/grown.azn")"
echo "$out"
echo "$out" | grep -q "new: 5x5, history 2, 3 blocks" || fail "grow did not report the new shape: $out"
"$BUILD/analyze" --checkpoint "$W/grown.azn" --moves "C3 D4" --simulations 8 | grep -q "MCTS value" \
    || fail "analyze rejected the grown checkpoint"

echo "== grow: widen the tiny network (alone and with deepening), then analyze it =="
out="$("$BUILD/grow" --from "$CK/gen_0003.azn" --filters 12 --out "$W/wide.azn")"
echo "$out"
echo "$out" | grep -q "new: 5x5, history 2, 1 blocks, 12 filters" || fail "grow --filters did not report the new shape: $out"
"$BUILD/analyze" --checkpoint "$W/wide.azn" --moves "C3 D4" --simulations 8 | grep -q "MCTS value" \
    || fail "analyze rejected the widened checkpoint"
out="$("$BUILD/grow" --from "$CK/gen_0003.azn" --filters 12 --blocks 2 --out "$W/wide_deep.azn")"
echo "$out" | grep -q "new: 5x5, history 2, 2 blocks, 12 filters" || fail "grow --filters --blocks: $out"
out="$("$BUILD/grow" --from "$CK/gen_0003.azn" --out "$W/copy.azn")"
echo "$out" | grep -q "plain copy" || fail "a no-op grow did not say so: $out"
status=0
out="$("$BUILD/grow" --from "$CK/gen_0003.azn" --filters 513 --out "$W/too_wide.azn" 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "grow beyond the filter limit exited $status, expected 1"
echo "$out" | grep -q "^error: .*maximum" || fail "grow filter limit refusal unexpected: $out"
test ! -e "$W/too_wide.azn" || fail "a refused widen still wrote a checkpoint"
status=0
out="$("$BUILD/grow" --from "$CK/gen_0003.azn" --filters 4 --out "$W/narrow.azn" 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "grow to fewer filters exited $status, expected 1"
test ! -e "$W/narrow.azn" || fail "a refused narrowing still wrote a checkpoint"

echo "== train: a new run from the grown network and the old run's buffer =="
GROWN_TINY=("${TINY[@]}" --set blocks=3)
"$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/grown_run" "${GROWN_TINY[@]}" \
    --init-from "$W/grown.azn" --buffer-from "$RUN" --generations 1
cmp "$W/grown.azn" "$W/grown_run/checkpoints/gen_0000.azn" || fail "gen_0000 is not the grown checkpoint"
grep -q "^buffer_from = $RUN" "$W/grown_run/init.txt" || fail "init.txt does not name the donor run"
test -s "$W/grown_run/checkpoints/gen_0001.azn" || fail "the grown run did not train"
status=0
out="$("$BUILD/train" --run "$W/grown_run" --resume --init-from "$W/grown.azn" 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "--resume --init-from exited $status, expected 1"
status=0
out="$("$BUILD/grow" --from "$CK/gen_0003.azn" --blocks 0 --out "$W/shrunk.azn" 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "grow to fewer blocks exited $status, expected 1"
echo "$out" | grep -q "^error: " || fail "grow refusal is not an error line: $out"
test ! -e "$W/shrunk.azn" || fail "a refused grow still wrote a checkpoint"
status=0
out="$("$BUILD/grow" --from "$CK/gen_0003.azn" --blocks 41 --out "$W/huge.azn" 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "grow beyond the block limit exited $status, expected 1"
echo "$out" | grep -q "^error: .*maximum" || fail "grow limit refusal unexpected: $out"

echo "== train: resume must refuse an architecture change =="
status=0
out="$("$BUILD/train" --run "$RUN" --resume --set filters=16 --generations 4 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "resume with filters=16 exited $status, expected 1"
echo "$out" | grep -q "architecture" || fail "resume refusal does not mention architecture: $out"
test ! -e "$CK/gen_0004.azn" || fail "a refused resume still trained"

echo "== train: a run with the ownership head =="
"$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/own_run" "${TINY[@]}" --set ownership=true \
    --generations 1
head -1 "$W/own_run/log.csv" | grep -q ',distinct_vs_prev,ownership_loss,' || fail "log.csv has no ownership_loss column"
col "$W/own_run/log.csv" ownership_loss | grep -Eq '^[0-9]+\.[0-9]{5}$' || fail "log.csv row has no ownership loss"
col "$W/own_run/log.csv" val_ownership_loss | grep -Eq '^[0-9]+\.[0-9]{5}$' || fail "log.csv row has no validation ownership loss"
[ -z "$(col "$RUN/log.csv" ownership_loss)$(col "$RUN/log.csv" resign_threshold)$(col "$RUN/log.csv" val_ownership_loss)" ] \
    || fail "a run without the head or resignation should leave those columns empty"
status=0
out="$("$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/own_bad" "${TINY[@]}" --set ownership=true \
    --init-from "$CK/gen_0003.azn" --generations 1 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "--init-from a checkpoint without the head exited $status, expected 1"
echo "$out" | grep -q -- "--ownership --out" || fail "the refusal does not suggest grow --ownership: $out"
test ! -e "$W/own_bad/state.bin" || fail "a refused start left a run behind"

echo "== grow: add an ownership head, then train from it =="
out="$("$BUILD/grow" --from "$CK/gen_0003.azn" --ownership --out "$W/owner.azn")"
echo "$out"
echo "$out" | grep -q "new: 5x5, history 2, 1 blocks, 8 filters, value_hidden 8, ownership head" \
    || fail "grow --ownership did not report the head: $out"
out="$("$BUILD/grow" --from "$W/owner.azn" --filters 12 --blocks 2 --out "$W/owner_big.azn")"
echo "$out" | grep -q "new: 5x5, history 2, 2 blocks, 12 filters, value_hidden 8, ownership head" \
    || fail "widening and deepening did not keep the head: $out"
status=0
out="$("$BUILD/grow" --from "$W/owner.azn" --ownership --out "$W/owner2.azn" 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "a second --ownership exited $status, expected 1"
echo "$out" | grep -q "already has an ownership head" || fail "unexpected refusal: $out"
test ! -e "$W/owner2.azn" || fail "a refused grow still wrote a checkpoint"
"$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/owner_run" "${TINY[@]}" --set ownership=true \
    --init-from "$W/owner.azn" --buffer-from "$RUN" --generations 1
cmp "$W/owner.azn" "$W/owner_run/checkpoints/gen_0000.azn" || fail "gen_0000 is not the checkpoint with the head"
test -s "$W/owner_run/checkpoints/gen_0001.azn" || fail "the run from the grown head did not train"

echo "== analyze: the ownership map, only for a checkpoint with the head =="
out="$("$BUILD/analyze" --checkpoint "$W/owner_run/checkpoints/gen_0001.azn" --moves "C3 D4" --simulations 8)"
echo "$out"
echo "$out" | grep -q '^Ownership, Black + / White - ' || fail "analyze printed no ownership map"
# Five rows of five cells after the column letters, each " +9" .. " -9" or "  .".
rows="$(echo "$out" | sed -n '/^Ownership/,/^$/p' | grep -Ec '^ [1-5] ( [+-][1-9]|  \.){5}$')"
[ "$rows" = "5" ] || fail "the ownership map should have 5 rows of 5 cells, found $rows"
[ "$(echo "$out" | grep -n '^Value' | cut -d: -f1)" -lt "$(echo "$out" | grep -n '^Ownership' | cut -d: -f1)" ] \
    || fail "the ownership map should follow the value line"
"$BUILD/analyze" --checkpoint "$CK/gen_0003.azn" --moves "C3 D4" --simulations 8 | grep -q '^Ownership' \
    && fail "analyze printed an ownership map for a checkpoint without the head"

echo "== train: a run with resignation =="
# v_resign 0: the untrained network resigns whenever its search thinks it is behind.
"$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/resign_run" "${TINY[@]}" --set resignation=true \
    --set resign_threshold_initial=0 --set resign_disabled_fraction=0.5 --set games_per_generation=8 \
    --set sgf_per_generation=8 --generations 2 | tee "$W/resign.out"
grep -Eq '^gen 1 \| selfplay 8 games avg [0-9.]+ plies resign 0\.00 rate ' "$W/resign.out" \
    || fail "the progress line does not report resignation"
head -1 "$W/resign_run/log.csv" | grep -q ',ownership_loss,resign_threshold,resign_rate,resign_false_positive,' \
    || fail "log.csv has no resignation columns"
col "$W/resign_run/log.csv" resign_threshold | grep -Eq '^-?[01]\.[0-9]{2}$' || fail "log.csv row has no resign_threshold"
grep -lq 'RE\[[BW]+R\]' "$W"/resign_run/sgf/*.sgf || fail "no self-play SGF records a resignation"
status=0
out="$("$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/resign_own" "${TINY[@]}" --set ownership=true \
    --set resignation=true --generations 1 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "resignation with the ownership head exited $status, expected 1"
echo "$out" | grep -q "cannot be combined with ownership" || fail "unexpected refusal: $out"
test ! -e "$W/resign_own/state.bin" || fail "a refused start left a run behind"

echo "integration_smoke: all checks passed"
