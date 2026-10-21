#!/usr/bin/env bash
# End-to-end smoke test of the executables on a tiny 5x5 run:
# train -> resume (with the validation loss) -> re-rate (as live, then against a reference network) ->
# arena (checkpoint, bot, random symmetry, c_puct and MCTS batch, on both sides or one) -> analyze (good, random symmetry and
# bad input) -> bench -> grow (deeper, wider, and a run started from the grown network), plus the refusal to change the architecture on resume,
# then the ownership head: a run with it, grow --ownership, and analyze's ownership map,
# and resignation: a run with it, and its refusal alongside the ownership head,
# and the GTP engine: a scripted session and a two-game match through tools/gtp_match.py,
# and distributed self-play: selfplay's version and fingerprint, a run with a fake remote
# (the subprocess transport) resumed and compared with the default run, AZGO_HOSTS, a refused alias.
#
# and backend = metal: with metal=ON a run on the GPU (start, resume, re-rate) and arena
# matches on it; with metal=OFF the refusals, which write nothing.
#
# and backend = cuda as a self-play engine: with cuda=ON a run whose games go to the GPU, in
# this process and as a remote slice (a selfplay process), resumed and compared with a run
# done straight; with cuda=OFF the refusals, which write nothing.
#
# usage: tests/integration_smoke.sh BUILD_DIR SOURCE_DIR [metal=ON|metal=OFF] [cuda=ON|cuda=OFF]
set -euo pipefail

BUILD="$(cd "${1:?usage: integration_smoke.sh BUILD_DIR SOURCE_DIR}" && pwd)"
SRC="$(cd "${2:?usage: integration_smoke.sh BUILD_DIR SOURCE_DIR}" && pwd)"
METAL="${3:-metal=OFF}"
CUDA="${4:-cuda=OFF}"
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
    grep -Eq '^[0-9]+\.[0-9]{5}$' <<<"$(col "$RUN/log.csv" $c)" || fail "log.csv has no $c"
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
grep -q -- "--rerate cannot be combined" <<<"$out" || fail "unexpected refusal message: $out"

echo "== train: re-rate against a reference network instead of the bot =="
out="$("$BUILD/train" --run "$RUN" --rerate --set bot_simulations= --set "reference_networks=$CK/gen_0001.azn")"
echo "$out"
grep -q '^rerate calibration | ref1 vs 0: ' <<<"$out" || fail "no calibration line for ref1"
grep -q '^-1,network,,0,' "$RUN/rerate/references.csv" || fail "rerate/references.csv does not list ref1"
grep -qx -- "-1,ref1,$CK/gen_0001.azn" "$RUN/rerate/reference-networks.csv" || fail "reference-networks.csv does not map -1 to its file"
[ "$(cksum < "$RUN/state.bin")" = "$state_before" ] || fail "--rerate changed state.bin"

echo "== arena =="
out="$("$BUILD/arena" --a "$CK/gen_0000.azn" --b "$CK/gen_0003.azn" --games 4 --simulations 4 --workers 2)"
echo "$out"
grep -Eq '^A .* scored' <<<"$out" || fail "arena did not print a score line"

echo "== arena: against the uniform bot =="
out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b-bot 8 --games 4 --simulations 4 --workers 2)"
echo "$out"
grep -q 'vs B bot(8 simulations): A scored' <<<"$out" || fail "arena --b-bot printed no score line"

echo "== arena: per-player simulations =="
out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0000.azn" --a-simulations 2 --b-simulations 16 --games 4 --simulations 4 --workers 2)"
echo "$out"
grep -q '(2 simulations) vs B .*(16 simulations): A scored' <<<"$out" || fail "arena --a/--b-simulations did not name both counts"

echo "== arena: random symmetry =="
out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0000.azn" --random-symmetry --games 4 --simulations 4 --workers 2)"
echo "$out"
grep -Eq '^A .* scored' <<<"$out" || fail "arena --random-symmetry did not print a score line"
grep -q '^Search: random symmetry' <<<"$out" || fail "arena --random-symmetry did not say so"

echo "== arena: random symmetry for one side =="
out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0003.azn" --a-random-symmetry --games 4 --simulations 4 --workers 2)"
echo "$out"
grep -Eq '^A .* scored' <<<"$out" || fail "arena --a-random-symmetry did not print a score line"
grep -qx 'Search: random symmetry per network evaluation, A only' <<<"$out" || fail "arena --a-random-symmetry did not say A only"
out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0000.azn" --b-random-symmetry --games 4 --simulations 4 --workers 2)"
grep -qx 'Search: random symmetry per network evaluation, B only' <<<"$out" || fail "arena --b-random-symmetry did not say B only"
# Both one-sided flags are --random-symmetry: the same games, the same lines.
both="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0000.azn" --a-random-symmetry --b-random-symmetry --games 4 --simulations 4 --workers 2)"
all="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0000.azn" --random-symmetry --games 4 --simulations 4 --workers 1)"
[ "$both" = "$all" ] || fail "arena --a- plus --b-random-symmetry differs from --random-symmetry: '$both' vs '$all'"
grep -qx 'Search: random symmetry per network evaluation' <<<"$both" || fail "arena with both one-sided flags did not say both"
# Against a bot only A has a network, so --random-symmetry is A's alone.
bot="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b-bot 8 --a-random-symmetry --games 4 --simulations 4 --workers 2)"
[ "$bot" = "$("$BUILD/arena" --a "$CK/gen_0003.azn" --b-bot 8 --random-symmetry --games 4 --simulations 4 --workers 2)" ] \
    || fail "arena --a-random-symmetry against a bot differs from --random-symmetry"
grep -qx 'Search: random symmetry per network evaluation, A only' <<<"$bot" || fail "arena against a bot did not say A only"

echo "== arena: c_puct and MCTS batch, for both sides or one =="
plain="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0003.azn" --games 4 --simulations 4 --workers 2)"
if grep -q '^Search: c_puct' <<<"$plain"; then fail "arena with the default search settings printed a settings line"; fi
# The defaults spelled out are the default match, line for line.
same="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0003.azn" --c-puct 1.5 --mcts-batch 8 \
    --a-c-puct 1.5 --b-mcts-batch 8 --games 4 --simulations 4 --workers 1)"
[ "$same" = "$plain" ] || fail "arena with the default search settings spelled out differs: '$same' vs '$plain'"
out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0003.azn" --a-c-puct 3 --b-mcts-batch 1 --games 4 --simulations 4 --workers 2)"
echo "$out"
grep -q '^A .*(c_puct 3) vs B .*(mcts batch 1): A scored' <<<"$out" || fail "arena did not name the one-sided settings"
grep -qx 'Search: c_puct A 3, B 1.5; mcts batch A 8, B 1' <<<"$out" || fail "arena did not print the per-side settings"
out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0000.azn" --c-puct 2 --mcts-batch 1 --games 4 --simulations 4 --workers 2)"
grep -qx 'Search: c_puct 2; mcts batch 1' <<<"$out" || fail "arena did not print the shared settings"
grep -q '^A [^(]* vs B [^(]*: A scored' <<<"$out" || fail "arena labelled a side for settings both share"
# A bot searches with c_puct and a batch too, so B's settings reach it.
out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b-bot 8 --b-c-puct 3 --b-mcts-batch 1 --games 4 --simulations 4 --workers 2)"
grep -q 'vs B bot(8 simulations) (c_puct 3, mcts batch 1): A scored' <<<"$out" || fail "arena did not name the bot's settings"
# Random symmetry and the other settings share one Search: line, the first
# line of the output (printed before the match starts).
out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b "$CK/gen_0003.azn" --a-random-symmetry --b-c-puct 2 --games 4 --simulations 4 --workers 2)"
[ "$(grep -c '^Search:' <<<"$out")" -eq 1 ] || fail "arena printed more than one Search: line: '$out'"
[ "$(echo "$out" | head -n 1)" = 'Search: random symmetry per network evaluation, A only; c_puct A 1.5, B 2; mcts batch 8' ] \
    || fail "arena did not start with the combined Search: line: '$out'"

# B is exactly one of a checkpoint and a bot, and a bot must search.
expect_arena_error() {
    local want="$1"; shift
    local status=0 out
    out="$("$BUILD/arena" --a "$CK/gen_0003.azn" "$@" 2>&1)" || status=$?
    [ "$status" -eq 1 ] || fail "arena $* exited $status, expected 1; output: $out"
    grep -qF -- "$want" <<<"$out" || fail "arena $* printed '$out', expected '$want'"
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
grep -q "MCTS value" <<<"$out" || fail "analyze printed no MCTS value"

echo "== analyze: random symmetry =="
out="$("$BUILD/analyze" --checkpoint "$CK/gen_0003.azn" --moves "C3 D4" --simulations 16 --random-symmetry)"
grep -q "MCTS value .* with random symmetry" <<<"$out" || fail "analyze --random-symmetry did not say so"

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
    grep -qF -- "$want" <<<"$out" || fail "analyze $* printed '$out', expected '$want'"
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
grep -q "new: 5x5, history 2, 3 blocks" <<<"$out" || fail "grow did not report the new shape: $out"
grep -q "MCTS value" <<<"$("$BUILD/analyze" --checkpoint "$W/grown.azn" --moves "C3 D4" --simulations 8)" \
    || fail "analyze rejected the grown checkpoint"

echo "== grow: widen the tiny network (alone and with deepening), then analyze it =="
out="$("$BUILD/grow" --from "$CK/gen_0003.azn" --filters 12 --out "$W/wide.azn")"
echo "$out"
grep -q "new: 5x5, history 2, 1 blocks, 12 filters" <<<"$out" || fail "grow --filters did not report the new shape: $out"
grep -q "MCTS value" <<<"$("$BUILD/analyze" --checkpoint "$W/wide.azn" --moves "C3 D4" --simulations 8)" \
    || fail "analyze rejected the widened checkpoint"
out="$("$BUILD/grow" --from "$CK/gen_0003.azn" --filters 12 --blocks 2 --out "$W/wide_deep.azn")"
grep -q "new: 5x5, history 2, 2 blocks, 12 filters" <<<"$out" || fail "grow --filters --blocks: $out"
out="$("$BUILD/grow" --from "$CK/gen_0003.azn" --out "$W/copy.azn")"
grep -q "plain copy" <<<"$out" || fail "a no-op grow did not say so: $out"
status=0
out="$("$BUILD/grow" --from "$CK/gen_0003.azn" --filters 513 --out "$W/too_wide.azn" 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "grow beyond the filter limit exited $status, expected 1"
grep -q "^error: .*maximum" <<<"$out" || fail "grow filter limit refusal unexpected: $out"
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
grep -q "^error: " <<<"$out" || fail "grow refusal is not an error line: $out"
test ! -e "$W/shrunk.azn" || fail "a refused grow still wrote a checkpoint"
status=0
out="$("$BUILD/grow" --from "$CK/gen_0003.azn" --blocks 41 --out "$W/huge.azn" 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "grow beyond the block limit exited $status, expected 1"
grep -q "^error: .*maximum" <<<"$out" || fail "grow limit refusal unexpected: $out"

echo "== train: resume must refuse an architecture change =="
status=0
out="$("$BUILD/train" --run "$RUN" --resume --set filters=16 --generations 4 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "resume with filters=16 exited $status, expected 1"
grep -q "architecture" <<<"$out" || fail "resume refusal does not mention architecture: $out"
test ! -e "$CK/gen_0004.azn" || fail "a refused resume still trained"

echo "== train: a run with the ownership head =="
"$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/own_run" "${TINY[@]}" --set ownership=true \
    --generations 1
grep -q ',distinct_vs_prev,ownership_loss,' <<<"$(head -1 "$W/own_run/log.csv")" || fail "log.csv has no ownership_loss column"
grep -Eq '^[0-9]+\.[0-9]{5}$' <<<"$(col "$W/own_run/log.csv" ownership_loss)" || fail "log.csv row has no ownership loss"
grep -Eq '^[0-9]+\.[0-9]{5}$' <<<"$(col "$W/own_run/log.csv" val_ownership_loss)" || fail "log.csv row has no validation ownership loss"
[ -z "$(col "$RUN/log.csv" ownership_loss)$(col "$RUN/log.csv" resign_threshold)$(col "$RUN/log.csv" val_ownership_loss)" ] \
    || fail "a run without the head or resignation should leave those columns empty"
status=0
out="$("$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/own_bad" "${TINY[@]}" --set ownership=true \
    --init-from "$CK/gen_0003.azn" --generations 1 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "--init-from a checkpoint without the head exited $status, expected 1"
grep -q -- "--ownership --out" <<<"$out" || fail "the refusal does not suggest grow --ownership: $out"
test ! -e "$W/own_bad/state.bin" || fail "a refused start left a run behind"

echo "== grow: add an ownership head, then train from it =="
out="$("$BUILD/grow" --from "$CK/gen_0003.azn" --ownership --out "$W/owner.azn")"
echo "$out"
grep -q "new: 5x5, history 2, 1 blocks, 8 filters, value_hidden 8, ownership head" <<<"$out" \
    || fail "grow --ownership did not report the head: $out"
out="$("$BUILD/grow" --from "$W/owner.azn" --filters 12 --blocks 2 --out "$W/owner_big.azn")"
grep -q "new: 5x5, history 2, 2 blocks, 12 filters, value_hidden 8, ownership head" <<<"$out" \
    || fail "widening and deepening did not keep the head: $out"
status=0
out="$("$BUILD/grow" --from "$W/owner.azn" --ownership --out "$W/owner2.azn" 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "a second --ownership exited $status, expected 1"
grep -q "already has an ownership head" <<<"$out" || fail "unexpected refusal: $out"
test ! -e "$W/owner2.azn" || fail "a refused grow still wrote a checkpoint"
"$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/owner_run" "${TINY[@]}" --set ownership=true \
    --init-from "$W/owner.azn" --buffer-from "$RUN" --generations 1
cmp "$W/owner.azn" "$W/owner_run/checkpoints/gen_0000.azn" || fail "gen_0000 is not the checkpoint with the head"
test -s "$W/owner_run/checkpoints/gen_0001.azn" || fail "the run from the grown head did not train"

echo "== analyze: the ownership map, only for a checkpoint with the head =="
out="$("$BUILD/analyze" --checkpoint "$W/owner_run/checkpoints/gen_0001.azn" --moves "C3 D4" --simulations 8)"
echo "$out"
grep -q '^Ownership, Black + / White - ' <<<"$out" || fail "analyze printed no ownership map"
# Five rows of five cells after the column letters, each " +9" .. " -9" or "  .".
rows="$(echo "$out" | sed -n '/^Ownership/,/^$/p' | grep -Ec '^ [1-5] ( [+-][1-9]|  \.){5}$')"
[ "$rows" = "5" ] || fail "the ownership map should have 5 rows of 5 cells, found $rows"
[ "$(grep -n '^Value' <<<"$out" | cut -d: -f1)" -lt "$(grep -n '^Ownership' <<<"$out" | cut -d: -f1)" ] \
    || fail "the ownership map should follow the value line"
grep -q '^Ownership' <<<"$("$BUILD/analyze" --checkpoint "$CK/gen_0003.azn" --moves "C3 D4" --simulations 8)" \
    && fail "analyze printed an ownership map for a checkpoint without the head"

echo "== train: a run with resignation =="
# v_resign 0: the untrained network resigns whenever its search thinks it is behind.
"$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/resign_run" "${TINY[@]}" --set resignation=true \
    --set resign_threshold_initial=0 --set resign_disabled_fraction=0.5 --set games_per_generation=8 \
    --set sgf_per_generation=8 --generations 2 | tee "$W/resign.out"
grep -Eq '^gen 1 \| selfplay 8 games avg [0-9.]+ plies resign 0\.00 rate ' "$W/resign.out" \
    || fail "the progress line does not report resignation"
grep -q ',ownership_loss,resign_threshold,resign_rate,resign_false_positive,' <<<"$(head -1 "$W/resign_run/log.csv")" \
    || fail "log.csv has no resignation columns"
grep -Eq '^-?[01]\.[0-9]{2}$' <<<"$(col "$W/resign_run/log.csv" resign_threshold)" || fail "log.csv row has no resign_threshold"
grep -lq 'RE\[[BW]+R\]' "$W"/resign_run/sgf/*.sgf || fail "no self-play SGF records a resignation"
status=0
out="$("$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/resign_own" "${TINY[@]}" --set ownership=true \
    --set resignation=true --generations 1 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "resignation with the ownership head exited $status, expected 1"
grep -q "cannot be combined with ownership" <<<"$out" || fail "unexpected refusal: $out"
test ! -e "$W/resign_own/state.bin" || fail "a refused start left a run behind"

echo "== gtp: a scripted session =="
# Nine commands, so nine responses, each a line starting with = or ? and
# ending in a blank line; nothing else may reach stdout.
printf '%s\n' '1 protocol_version' 'boardsize 9' 'boardsize 5' '# a comment' 'komi 6.5' 'play b C3' \
    'play w c3' '2 genmove w' 'final_score' 'quit' 'name' \
    | "$BUILD/gtp" --checkpoint "$CK/gen_0003.azn" --simulations 8 > "$W/gtp.out" 2> "$W/gtp.err"
out="$(cat "$W/gtp.out")"
echo "$out"
[ "$(grep -c '^[=?]' "$W/gtp.out")" = "9" ] || fail "gtp did not answer exactly the nine commands before quit"
[ "$(grep -c '^$' "$W/gtp.out")" = "9" ] && [ "$(wc -l < "$W/gtp.out" | tr -d ' ')" = "18" ] \
    || fail "gtp responses are not each followed by one blank line"
grep -qx '=1 2' <<<"$out" || fail "gtp protocol_version"
grep -qx '? unacceptable size' <<<"$out" || fail "gtp accepted a board size the checkpoint cannot play"
grep -qx '? illegal move' <<<"$out" || fail "gtp accepted a move on an occupied point"
grep -Eqx '=2 ([A-E][1-5]|pass)' <<<"$out" || fail "gtp genmove did not answer a vertex"
grep -Eqx '= (B\+|W\+)[0-9.]+|= 0' <<<"$out" || fail "gtp final_score is not in B+3.5 form"
grep -q '^genmove W: ' "$W/gtp.err" || fail "gtp logged no genmove line on stderr"
status=0
out="$("$BUILD/gtp" --checkpoint "$W/missing.azn" < /dev/null 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "gtp with a missing checkpoint exited $status, expected 1"
grep -q '^error: ' <<<"$out" || fail "gtp did not report the missing checkpoint: $out"

echo "== gtp: a two-game match between two instances, with tools/gtp_match.py =="
ENGINE="$BUILD/gtp --checkpoint $CK/gen_0003.azn --simulations 4"
out="$(python3 "$SRC/tools/gtp_match.py" --a "$ENGINE" --b "$ENGINE --random-symmetry --seed 2" \
    --games 2 --size 5 --komi 7.5 --out "$W/match")"
echo "$out"
grep -Eq '^A alphazero-go gen_0003\.azn, 4 simulations vs B .*: A scored [0-9.]+/2 = ' <<<"$out" \
    || fail "gtp_match printed no score line"
grep -q '^Elo A - B: ' <<<"$out" || fail "gtp_match printed no Elo line"
[ "$(wc -l < "$W/match/results.csv" | tr -d ' ')" = "3" ] || fail "results.csv should have a header and 2 rows"
for g in 1 2; do
    grep -Eq 'RE\[([BW]\+[0-9.]+|0)\]' "$W/match/game_000$g.sgf" || fail "game_000$g.sgf has no scored result"
done
# The SGF is readable by the project's own parser, and ends where the match did.
out="$("$BUILD/analyze" --checkpoint "$CK/gen_0003.azn" --sgf "$W/match/game_0001.sgf" --simulations 4 2>&1)" || true
grep -q 'nothing to analyze' <<<"$out" || fail "analyze did not read the match SGF as a finished game: $out"
status=0
out="$(python3 "$SRC/tools/gtp_match.py" --a "$ENGINE" --b "$ENGINE" --size 9 --out "$W/match9" 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "gtp_match on a size the engine refuses exited $status, expected 1"
grep -q "refused 'boardsize 9': unacceptable size" <<<"$out" || fail "unexpected gtp_match error: $out"

echo "== distributed self-play: selfplay, a fake remote, resume, refusals =="
# The build ID: train and selfplay are built from the same tree.
train_id="$("$BUILD/train" --version | sed -n 's/^build //p')"
selfplay_id="$("$BUILD/selfplay" --version | sed -n 's/^build //p')"
[ -n "$train_id" ] && [ "$train_id" = "$selfplay_id" ] || fail "selfplay --version '$selfplay_id' differs from train's '$train_id'"
"$BUILD/selfplay" --fingerprint --backend naive | grep -q '^naive; ' || fail "selfplay --fingerprint --backend naive failed"
if [ "$METAL" != "metal=ON" ]; then
    status=0
    out="$("$BUILD/selfplay" --fingerprint --backend metal 2>&1)" || status=$?
    [ "$status" -eq 1 ] || fail "selfplay --fingerprint --backend metal without Metal exited $status, expected 1"
    grep -q "backend 'metal' not available" <<<"$out" || fail "selfplay did not say metal is not built: $out"
fi
# A machine-local hosts file (never in the repo): "fake" has no ssh, so its
# slices run the selfplay binary here, as a separate process.
HOSTS="$W/hosts"
echo "fake exe=$BUILD/selfplay workdir=$W/remote" > "$HOSTS"
DRUN="$W/distributed"
ENGINES="selfplay_engines=local/naive:1,fake/naive=2"
"$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$DRUN" "${TINY[@]}" --hosts "$HOSTS" --set "$ENGINES" \
    --generations 2 > "$W/distributed.out"
grep -q '^engine fake/naive: ' "$W/distributed.out" || fail "no engine line for fake/naive"
grep -q '^selfplay engines: local/naive 2 .* | fake/naive 2 ' "$W/distributed.out" || fail "no selfplay engines line"
"$BUILD/train" --run "$DRUN" --resume --hosts "$HOSTS" --generations 3 > /dev/null
MANIFEST="$DRUN/selfplay-engines.csv"
for g in 1 2 3; do
    [ "$(grep -c "^$g," "$MANIFEST")" = 2 ] || fail "selfplay-engines.csv should have 2 rows for generation $g"
done
# The same arithmetic on both engines: equal fingerprint columns.
[ "$(awk -F, 'NR > 1 { print $6 }' "$MANIFEST" | sort -u | wc -l | tr -d ' ')" = 1 ] \
    || fail "local/naive and fake/naive have different fingerprints"
grep -q '^3,2,2,fake/naive,fake/naive,"naive; ' "$MANIFEST" || fail "no fake/naive row for generation 3"
# Without engines, the same run: naive on one machine computes the same bits.
"$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/plain" "${TINY[@]}" --generations 3 > /dev/null
cmp -s "$W/plain/state.bin" "$DRUN/state.bin" || fail "the run with a fake remote differs from the default run"
# AZGO_HOSTS in place of --hosts.
AZGO_HOSTS="$HOSTS" "$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/byenv" "${TINY[@]}" --set "$ENGINES" \
    --generations 1 > /dev/null || fail "AZGO_HOSTS was not used"
cmp -s "$W/plain/checkpoints/gen_0001.azn" "$W/byenv/checkpoints/gen_0001.azn" || fail "AZGO_HOSTS run differs"
# An alias the hosts file does not define: refused, nothing written.
status=0
out="$("$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/unknown" "${TINY[@]}" --hosts "$HOSTS" \
       --set "selfplay_engines=local/naive:1,nowhere/naive:1" --generations 1 2>&1)" || status=$?
[ "$status" -eq 1 ] || fail "an unknown alias exited $status, expected 1"
grep -q "machine 'nowhere' is not defined in the hosts file" <<<"$out" || fail "unexpected refusal: $out"
test ! -e "$W/unknown/config.cfg" || fail "a refused run wrote config.cfg"

# nohup: a train started with SIGHUP ignored keeps ignoring it.
( trap '' HUP; exec "$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/nohup" "${TINY[@]}" --generations 3 \
      > "$W/nohup.out" 2>&1 ) &
nohup_pid=$!
sleep 0.3
kill -HUP "$nohup_pid" 2>/dev/null || true
wait "$nohup_pid" || fail "train with SIGHUP ignored died of SIGHUP"
test -s "$W/nohup/checkpoints/gen_0003.azn" || fail "the nohup run did not finish"

echo "== distributed self-play: the operations tools =="
for script in cluster_sync.sh stop_after_generation.sh; do
    bash -n "$SRC/tools/$script" || fail "tools/$script does not parse"
    "$SRC/tools/$script" --help | grep -q '^usage: ' || fail "tools/$script has no usage text"
done
python3 -I -m py_compile "$SRC/tools/engine_weights.py" || fail "tools/engine_weights.py does not compile"
# No machine with ssh in the hosts file: nothing to sync.
out="$("$SRC/tools/cluster_sync.sh" --hosts "$HOSTS" --dry-run)"
grep -q '^no remote machines' <<<"$out" || fail "cluster_sync.sh --dry-run: $out"
# Remotes in a scratch repository (the script syncs the HEAD of the one it
# runs in): each is built where its exe lies, with the options the build
# directory's name asks for; a directory not named build* is refused.
if command -v git > /dev/null; then
    SYNC="$W/syncrepo"
    mkdir -p "$SYNC"
    echo x > "$SYNC/file"
    git -C "$SYNC" init -q
    git -C "$SYNC" -c user.name=t -c user.email=t@t add -A
    git -C "$SYNC" -c user.name=t -c user.email=t@t commit -q -m one
    printf '%s\n' "gpu  ssh=u@h1 exe=~/c/build-metal/selfplay workdir=~/w" \
                  "cpu  ssh=u@h2 exe=~/c/build-blas/selfplay workdir=~/w" \
                  "pure ssh=u@h3 exe=~/c/build/selfplay workdir=~/w" \
                  "word ssh=u@h5 exe=~/c/build_metal_2/selfplay workdir=~/w" \
                  "sub  ssh=u@h6 exe=~/c/build-nometal/selfplay workdir=~/w" > "$W/sync-hosts"
    out="$(cd "$SYNC" && "$SRC/tools/cluster_sync.sh" --hosts "$W/sync-hosts" --dry-run)"
    cmake_line() { grep "u@h$1 cmake" <<<"$out"; }
    cmake_line 1 | grep -q 'c/build-metal.*-DAZ_USE_METAL=ON' || fail "cluster_sync.sh: metal build: $out"
    ! cmake_line 1 | grep -q 'AZ_USE_BLAS' || fail "cluster_sync.sh: metal build names BLAS: $out"
    cmake_line 2 | grep -q 'c/build-blas.*-DAZ_USE_METAL=OFF.*-DAZ_USE_BLAS=ON' || fail "cluster_sync.sh: blas build: $out"
    cmake_line 3 | grep -Eq 'c/build[^-].*-DAZ_USE_METAL=OFF.*-DAZ_USE_BLAS=OFF' || fail "cluster_sync.sh: plain build: $out"
    # Words of the name, not substrings: build_metal_2 is Metal, build-nometal is not.
    cmake_line 5 | grep -q 'c/build_metal_2.*-DAZ_USE_METAL=ON' || fail "cluster_sync.sh: build_metal_2: $out"
    cmake_line 6 | grep -Eq 'c/build-nometal[^-].*-DAZ_USE_METAL=OFF.*-DAZ_USE_BLAS=OFF' \
        || fail "cluster_sync.sh: build-nometal took a flavour: $out"
    [ "$(grep -c '_NPROCESSORS_ONLN' <<<"$out")" = 5 ] || fail "cluster_sync.sh: no portable core count: $out"
    printf '%s\n' "odd ssh=u@h4 exe=~/c/src/selfplay workdir=~/w" > "$W/sync-hosts"
    status=0
    out="$(cd "$SYNC" && "$SRC/tools/cluster_sync.sh" --hosts "$W/sync-hosts" --dry-run 2>&1)" || status=$?
    [ "$status" = 1 ] && grep -q 'not in a build directory' <<<"$out" || fail "cluster_sync.sh took exe in src/: $out"
fi
# A suggested selfplay_engines value that train accepts (on a copy of the run).
out="$(python3 -I "$SRC/tools/engine_weights.py" "$DRUN" --last 3 --hosts "$HOSTS")"
echo "$out"
suggested="$(sed -n 's/^selfplay_engines = //p' <<<"$out")"
[ -n "$suggested" ] || fail "engine_weights.py printed no selfplay_engines line"
cp -R "$DRUN" "$W/weights_copy"
"$BUILD/train" --run "$W/weights_copy" --resume --hosts "$HOSTS" --set "selfplay_engines=$suggested" \
    --generations 4 > /dev/null || fail "train refused the suggested selfplay_engines '$suggested'"
# stop_after_generation.sh: SIGINT once log.csv gains a row.
mkdir -p "$W/stoprun"
printf 'generation,games\n1,4\n' > "$W/stoprun/log.csv"
# (A background job of a script ignores SIGINT; perl restores the default.)
perl -e '$SIG{INT} = "DEFAULT"; sleep 60' &
victim=$!
"$SRC/tools/stop_after_generation.sh" "$W/stoprun" "$victim" > "$W/stop.out" 2> "$W/stop.err" &
stopper=$!
sleep 0.5
kill -0 "$victim" || fail "stop_after_generation.sh stopped the process before a new row"
printf '2,4\n' >> "$W/stoprun/log.csv"
wait "$stopper" || fail "stop_after_generation.sh failed"
status=0
wait "$victim" || status=$?
[ "$status" -ne 0 ] || fail "the process was not interrupted"
grep -q 'generation 2' "$W/stop.out" || fail "stop_after_generation.sh did not name generation 2: $(cat "$W/stop.out")"
# The process's parent is this script: the tool warns about a wrapper.
grep -q "was started by process $$ " "$W/stop.err" || fail "no warning about the parent script: $(cat "$W/stop.err")"

if [ "$METAL" = "metal=ON" ]; then
    echo "== metal: a run on the GPU, resumed and re-rated =="
    MRUN="$W/metal"
    GPU_LINE='^gpu: .+, macOS [0-9.]+, shaders [0-9a-f]{8} \(inference: self-play, arena, validation\)$'
    out="$("$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$MRUN" "${TINY[@]}" --set backend=metal --generations 2)"
    grep -Eq "$GPU_LINE" <<<"$out" || fail "train --set backend=metal printed no device line"
    test -s "$MRUN/checkpoints/gen_0002.azn" || fail "metal run wrote no gen_0002.azn"
    grep -q '^backend = metal$' "$MRUN/config.cfg" || fail "config.cfg does not record backend = metal"
    out="$("$BUILD/train" --run "$MRUN" --resume --generations 3)"
    grep -Eq "$GPU_LINE" <<<"$out" || fail "the resumed metal run printed no device line"
    test -s "$MRUN/checkpoints/gen_0003.azn" || fail "resumed metal run wrote no gen_0003.azn"
    out="$("$BUILD/train" --run "$MRUN" --rerate)"
    grep -Eq "$GPU_LINE" <<<"$out" || fail "the metal re-rate printed no device line"
    # Gating is off, so the replay plays exactly the live arena games, on the GPU too.
    cmp -s "$MRUN/rerate/results.csv" "$MRUN/results.csv" || fail "metal re-rate differs from the live results"

    echo "== metal: arena on the GPU =="
    MCK="$MRUN/checkpoints"
    out="$("$BUILD/arena" --a "$MCK/gen_0000.azn" --b "$MCK/gen_0003.azn" --games 4 --simulations 4 --workers 3 --backend metal)"
    grep -Eq '^A .* scored' <<<"$out" || fail "arena --backend metal printed no score line"
    one="$("$BUILD/arena" --a "$MCK/gen_0000.azn" --b "$MCK/gen_0003.azn" --games 4 --simulations 4 --workers 1 --backend metal --inference-batch 8)"
    [ "$out" = "$one" ] || fail "arena --backend metal depends on --workers or --inference-batch: '$out' vs '$one'"
    out="$("$BUILD/arena" --a "$MCK/gen_0003.azn" --b-bot 8 --games 4 --simulations 4 --workers 2 --backend metal)"
    grep -q 'vs B bot(8 simulations): A scored' <<<"$out" || fail "arena --backend metal against a bot printed no score line"
    status=0
    out="$("$BUILD/arena" --a "$MCK/gen_0003.azn" --b-bot 8 --mcts-batch 8 --inference-batch 4 --backend metal 2>&1)" || status=$?
    [ "$status" -eq 1 ] || fail "arena with --inference-batch below the mcts batch exited $status, expected 1"
    grep -q 'inference-batch 4 is below the mcts batch 8' <<<"$out" || fail "unexpected arena error: $out"

    echo "== metal: analyze and gtp on the GPU =="
    out="$("$BUILD/analyze" --checkpoint "$MCK/gen_0003.azn" --moves "C3 D4 B2 C4 D3 B4" --simulations 16 --backend metal)"
    grep -q '^Value (side to move): ' <<<"$out" || fail "analyze --backend metal printed no value"
    grep -q '^Policy prior' <<<"$out" || fail "analyze --backend metal printed no policy"
    grep -q '^MCTS value' <<<"$out" || fail "analyze --backend metal printed no search"
    # The training run has an ownership head here only if TINY gave it one;
    # the ownership run from above does.
    out="$("$BUILD/analyze" --checkpoint "$W/owner_run/checkpoints/gen_0001.azn" --moves "C3 D4" --simulations 8 --backend metal)"
    grep -q '^Ownership, Black + / White - ' <<<"$out" || fail "analyze --backend metal printed no ownership map"
    printf '%s\n' 'boardsize 5' 'play b C3' 'genmove w' 'quit' \
        | "$BUILD/gtp" --checkpoint "$MCK/gen_0003.azn" --simulations 8 --backend metal > "$W/gtp-metal.out" 2> /dev/null \
        || fail "gtp --backend metal failed"
    grep -Eqx '= ([A-E][1-5]|pass)' "$W/gtp-metal.out" || fail "gtp --backend metal played no move"

    echo "== metal: bench =="
    out="$("$BUILD/bench" --sizes 5 --seconds 0.1 --backend metal --configs "$SRC/configs" --hub-workers 2)"
    grep -Eq '^5x5 +3 x 32 +metal +[0-9]+ +-$' <<<"$out" || fail "bench --backend metal printed no metal row"
    grep -q 'metal hub' <<<"$out" || fail "bench --backend metal printed no hub table"

    echo "== metal: bench --mixed (the hub and CPU workers at once) =="
    out="$("$BUILD/bench" --mixed --sizes 5 --seconds 0.2 --hub-workers 2 --cpu-workers 0,1 --configs "$SRC/configs")"
    echo "$out"
    grep -Eq '^cpu workers +hub evals/s +cpu evals/s +total evals/s +hub vs alone +mean pass +slots busy$' <<<"$out" \
        || fail "bench --mixed printed no mixed table"
    grep -Eq '^ +0 +[0-9]+ +0 +[0-9]+ +100\.0% +[0-9.]+ +[0-9]+%$' <<<"$out" || fail "bench --mixed printed no row for 0 CPU workers"
    grep -Eq '^ +1 +[0-9]+ +[1-9][0-9]* +[0-9]+ +[0-9.]+% +[0-9.]+ +[0-9]+%$' <<<"$out" \
        || fail "bench --mixed printed no row for 1 CPU worker"
    grep -Eq '^cpu workers alone +cpu evals/s +per worker$' <<<"$out" || fail "bench --mixed printed no CPU-alone table"
    grep -Eq '^ +1 +[1-9][0-9]* +[1-9][0-9]*$' <<<"$out" || fail "bench --mixed printed no CPU-alone row"
    out="$("$BUILD/bench" --mixed --sizes 5 --seconds 0.1 --hub-workers 2 --cpu-workers 1 --qos utility --configs "$SRC/configs")"
    grep -q 'qos utility' <<<"$out" || fail "bench --mixed --qos utility did not say so"
else
    echo "== metal: refused in a build without it =="
    NOT_BUILT="backend 'metal' not available: rebuild with -DAZ_USE_METAL=ON (macOS only)"
    status=0
    out="$("$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/nometal" "${TINY[@]}" --set backend=metal --generations 1 2>&1)" || status=$?
    [ "$status" -eq 1 ] || fail "train --set backend=metal without Metal exited $status, expected 1"
    grep -qF "$NOT_BUILT" <<<"$out" || fail "train did not say metal is not built: $out"
    test ! -e "$W/nometal/state.bin" || fail "a refused metal run wrote state.bin"
    test ! -e "$W/nometal/config.cfg" || fail "a refused metal run wrote config.cfg"
    status=0
    out="$("$BUILD/train" --run "$RUN" --resume --set backend=metal 2>&1)" || status=$?
    [ "$status" -eq 1 ] || fail "resume --set backend=metal without Metal exited $status, expected 1"
    grep -q '^backend = naive$' "$RUN/config.cfg" || fail "a refused metal resume rewrote config.cfg"
    status=0
    out="$("$BUILD/arena" --a "$CK/gen_0003.azn" --b-bot 8 --games 2 --backend metal 2>&1)" || status=$?
    [ "$status" -eq 1 ] || fail "arena --backend metal without Metal exited $status, expected 1"
    grep -qF "$NOT_BUILT" <<<"$out" || fail "arena did not say metal is not built: $out"
    status=0
    out="$("$BUILD/analyze" --checkpoint "$CK/gen_0003.azn" --moves "C3 D4" --backend metal 2>&1)" || status=$?
    [ "$status" -eq 1 ] || fail "analyze --backend metal without Metal exited $status, expected 1"
    grep -qF "$NOT_BUILT" <<<"$out" || fail "analyze did not say metal is not built: $out"
    status=0
    out="$(echo quit | "$BUILD/gtp" --checkpoint "$CK/gen_0003.azn" --backend metal 2>&1)" || status=$?
    [ "$status" -eq 1 ] || fail "gtp --backend metal without Metal exited $status, expected 1"
    grep -qF "$NOT_BUILT" <<<"$out" || fail "gtp did not say metal is not built: $out"
    status=0
    out="$("$BUILD/bench" --sizes 5 --seconds 0.1 --backend metal --configs "$SRC/configs" 2>&1)" || status=$?
    [ "$status" -eq 1 ] || fail "bench --backend metal without Metal exited $status, expected 1"
    grep -qF "$NOT_BUILT" <<<"$out" || fail "bench did not say metal is not built: $out"
    status=0
    out="$("$BUILD/bench" --mixed --sizes 5 --seconds 0.1 --hub-backend metal --configs "$SRC/configs" 2>&1)" || status=$?
    [ "$status" -eq 1 ] || fail "bench --mixed without Metal exited $status, expected 1"
    grep -qF "$NOT_BUILT" <<<"$out" || fail "bench --mixed did not say metal is not built: $out"
fi

if [ "$CUDA" = "cuda=ON" ]; then
    echo "== cuda: self-play on the GPU, in this process and as a remote slice =="
    "$BUILD/selfplay" --fingerprint --backend cuda | grep -Eq '^cuda; .+ sm_[0-9]+, driver [0-9.]+, runtime [0-9.]+, kernels [0-9a-f]{8}$' \
        || fail "selfplay --fingerprint --backend cuda printed no cuda fingerprint"
    CHOSTS="$W/cuda-hosts"
    echo "fake exe=$BUILD/selfplay workdir=$W/cuda-remote" > "$CHOSTS"
    CENGINES="selfplay_engines=local/naive:1,local/cuda=2,fake/cuda=2"
    CRUN="$W/cuda"
    "$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$CRUN" "${TINY[@]}" --set games_per_generation=6 \
        --hosts "$CHOSTS" --set "$CENGINES" --generations 2 > "$W/cuda.out"
    grep -q '^engine local/cuda: ' "$W/cuda.out" || fail "no engine line for local/cuda"
    grep -q '^engine fake/cuda: ' "$W/cuda.out" || fail "no engine line for fake/cuda"
    "$BUILD/train" --run "$CRUN" --resume --hosts "$CHOSTS" --generations 3 > /dev/null
    "$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/cuda-straight" "${TINY[@]}" --set games_per_generation=6 \
        --hosts "$CHOSTS" --set "$CENGINES" --generations 3 > /dev/null
    for f in state.bin results.csv checkpoints/gen_0003.azn; do
        cmp -s "$CRUN/$f" "$W/cuda-straight/$f" || fail "the resumed cuda run's $f differs from the straight run's"
    done
    grep -q '^3,2,2,local/cuda,local/cuda,"cuda; ' "$CRUN/selfplay-engines.csv" || fail "no local/cuda row for generation 3"
    grep -q '^3,4,2,fake/cuda,fake/cuda,"cuda; ' "$CRUN/selfplay-engines.csv" || fail "no fake/cuda row for generation 3"

    echo "== cuda: bench =="
    out="$("$BUILD/bench" --sizes 5 --backend cuda --seconds 0.1 --hub-workers 2,4 --configs "$SRC/configs")"
    grep -Eq '^gpu: .+ sm_[0-9]+, driver' <<<"$out" || fail "bench --backend cuda printed no gpu line"
    grep -Eq '^5x5 +3 x 32 +cuda +[1-9][0-9]* +-$' <<<"$out" || fail "bench --backend cuda printed no cuda row"
    grep -Eq 'ms/pass +\(cuda\)$' <<<"$out" || fail "bench --backend cuda printed no raw-pass table"
    grep -Eq '^5x5 +3 x 32 +256 +[1-9]' <<<"$out" || fail "bench --backend cuda printed no batch-256 pass row"
    grep -Eq '^5x5 +3 x 32 +cuda hub +4 +[1-9]' <<<"$out" || fail "bench --backend cuda printed no hub row"
    out="$("$BUILD/bench" --mixed --sizes 5 --seconds 0.1 --hub-backend cuda --hub-workers 2 --cpu-workers 0,1 \
           --configs "$SRC/configs")"
    grep -q 'cuda hub 2 threads' <<<"$out" || fail "bench --mixed --hub-backend cuda printed no mixed table"
    grep -Eq '^ +1 +[1-9][0-9]* +[1-9][0-9]*$' <<<"$out" || fail "bench --mixed --hub-backend cuda printed no CPU-alone row"
else
    echo "== cuda: refused in a build without it =="
    CUDA_NOT_BUILT="backend 'cuda' not available: rebuild with -DAZ_USE_CUDA=ON (Linux, NVIDIA GPU)"
    status=0
    out="$("$BUILD/selfplay" --fingerprint --backend cuda 2>&1)" || status=$?
    [ "$status" -eq 1 ] || fail "selfplay --fingerprint --backend cuda without CUDA exited $status, expected 1"
    grep -qF "$CUDA_NOT_BUILT" <<<"$out" || fail "selfplay did not say cuda is not built: $out"
    status=0
    out="$("$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/nocuda" "${TINY[@]}" \
           --set "selfplay_engines=local/naive:1,local/cuda:1" --generations 1 2>&1)" || status=$?
    [ "$status" -eq 1 ] || fail "a run with a local cuda engine without CUDA exited $status, expected 1"
    grep -qF "$CUDA_NOT_BUILT" <<<"$out" || fail "train did not say cuda is not built: $out"
    test ! -e "$W/nocuda/state.bin" && test ! -e "$W/nocuda/config.cfg" || fail "a refused cuda run wrote files"
    for args in "--sizes 5 --seconds 0.1 --backend cuda" "--mixed --sizes 5 --seconds 0.1 --hub-backend cuda"; do
        status=0
        # shellcheck disable=SC2086
        out="$("$BUILD/bench" $args --configs "$SRC/configs" 2>&1)" || status=$?
        [ "$status" -eq 1 ] || fail "bench $args without CUDA exited $status, expected 1"
        grep -qF "$CUDA_NOT_BUILT" <<<"$out" || fail "bench $args did not say cuda is not built: $out"
    done
fi
# Never a run's own backend, in any build.
status=0
out="$("$BUILD/train" --config "$SRC/configs/5x5.cfg" --run "$W/cudabackend" "${TINY[@]}" --set backend=cuda \
       --generations 1 2>&1)" || status=$?
[ "$status" -eq 1 ] && grep -q 'selfplay_engines' <<<"$out" || fail "train --set backend=cuda was not refused: $out"

echo "integration_smoke: all checks passed"
