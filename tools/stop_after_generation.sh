#!/usr/bin/env bash
# stop_after_generation.sh: stops a training run just after it commits a
# generation, so it can be resumed with other settings (for instance
# --set selfplay_engines=...) losing at most the first seconds of the next
# generation.
#
# train appends a row to RUN_DIR/log.csv right after it writes state.bin,
# the commit point. This script waits for log.csv to gain a row, then sends
# SIGINT to PID and prints the generation of that row.
#
# Mark Castelluccio, with assistance from Claude (Anthropic).
set -euo pipefail

usage() {
    cat <<'EOF'
usage: tools/stop_after_generation.sh RUN_DIR PID

Waits until RUN_DIR/log.csv gains a row (train appends it right after
committing a generation in state.bin), sends SIGINT to PID (the train
process) and prints the generation it stopped after. Resume with
train --run RUN_DIR --resume [--set ...].
EOF
}

case "${1:-}" in --help|-h) usage; exit 0 ;; esac
[ $# -eq 2 ] || { usage >&2; exit 2; }
RUN="$1" PID="$2"
LOG="$RUN/log.csv"
[ -f "$LOG" ] || { echo "error: no $LOG" >&2; exit 1; }
kill -0 "$PID" 2>/dev/null || { echo "error: no process $PID" >&2; exit 1; }

rows() { wc -l < "$LOG" | tr -d ' '; }
start="$(rows)"
echo "waiting for $LOG to gain a row (now $start lines; process $PID)" >&2
while [ "$(rows)" -le "$start" ]; do
    kill -0 "$PID" 2>/dev/null || { echo "error: process $PID ended first" >&2; exit 1; }
    sleep 1
done
kill -INT "$PID"
generation="$(tail -n 1 "$LOG" | cut -d, -f1)"
echo "stopped process $PID after generation $generation (committed)"
