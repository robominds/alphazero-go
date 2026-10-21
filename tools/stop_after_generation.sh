#!/usr/bin/env bash
# stop_after_generation.sh: stops a training run just after it commits a
# generation, so it can be resumed with other settings (for instance
# --set selfplay_engines=...) losing at most the first seconds of the next
# generation.
#
# train appends a row to RUN_DIR/log.csv right after it writes state.bin,
# the commit point. This script waits for log.csv to gain a row, then sends
# SIGTERM to PID, waits for it to end and prints the generation of that row.
# Not SIGINT: a command a script starts in the background (nohup ... &)
# ignores SIGINT, and train keeps an ignored signal ignored, so it would go
# on into the next generation. train handles SIGTERM as it does SIGINT.
#
# Mark Castelluccio, with assistance from Claude (Anthropic).
set -euo pipefail

usage() {
    cat <<'EOF'
usage: tools/stop_after_generation.sh RUN_DIR PID

Waits until RUN_DIR/log.csv gains a row (train appends it right after
committing a generation in state.bin), sends SIGTERM to PID (the train
process), waits up to a minute for it to end and prints the generation it
stopped after. Resume with
train --run RUN_DIR --resume [--set ...].

If train was started by a wrapper script that goes on after it ends (a
loop that resumes it, or a job that runs more steps), stop that script
first, or it will carry on with its own settings as soon as train stops.
The script warns when PID's parent looks like such a script, and names it.
EOF
}

case "${1:-}" in --help|-h) usage; exit 0 ;; esac
[ $# -eq 2 ] || { usage >&2; exit 2; }
RUN="$1" PID="$2"
LOG="$RUN/log.csv"
[ -f "$LOG" ] || { echo "error: no $LOG" >&2; exit 1; }
kill -0 "$PID" 2>/dev/null || { echo "error: no process $PID" >&2; exit 1; }

# A wrapper script around train would go on when train stops: say so.
parent="$(ps -o ppid= -p "$PID" | tr -d ' ')"
if [ -n "$parent" ] && [ "$parent" -gt 1 ]; then
    parent_command="$(ps -o command= -p "$parent" 2>/dev/null || true)"
    case "$(basename "$(ps -o comm= -p "$parent" 2>/dev/null | sed 's/^-//')")" in
        bash|sh|zsh|dash|ksh)
            words=($parent_command)
            if [ ${#words[@]} -gt 1 ]; then
                echo "warning: process $PID was started by process $parent ($parent_command)." >&2
                echo "  If that script restarts train or runs more steps when train ends, stop it" >&2
                echo "  first (kill $parent), or it will go on with its own settings." >&2
            fi ;;
    esac
fi

rows() { wc -l < "$LOG" | tr -d ' '; }
start="$(rows)"
echo "waiting for $LOG to gain a row (now $start lines; process $PID)" >&2
while [ "$(rows)" -le "$start" ]; do
    kill -0 "$PID" 2>/dev/null || { echo "error: process $PID ended first" >&2; exit 1; }
    sleep 1
done
kill -TERM "$PID"
generation="$(tail -n 1 "$LOG" | cut -d, -f1)"
# Running: neither gone nor a zombie its parent has yet to reap.
running() { case "$(ps -o stat= -p "$PID" 2>/dev/null)" in ''|Z*) return 1 ;; esac; }
for _ in $(seq 60); do
    running || break
    sleep 1
done
if running; then
    echo "error: process $PID is still running a minute after SIGTERM (generation $generation is committed)" >&2
    exit 1
fi
echo "stopped process $PID after generation $generation (committed)"
