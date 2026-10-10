#!/bin/sh
# Run the matches listed in job files (lines as for pachi-jobs.sh: LABEL
# PLAYER PLAYOUTS SEED [SIMULATIONS [MCTS_BATCH]]), starting each one when
# fewer than MAX (default 8) matches (gtp_match.py processes) are running
# on this host, so that several lists can share the machine. Each match's
# start and end times (Unix seconds) and exit status are appended to
# queue.log.
#
#   queue.sh JOBFILE...
#
# PACHI_ARGS, if set, is passed on to pachi-match.sh. Lines are started
# 10 s apart, and two queues running at once can briefly exceed MAX by
# one. Killing a queue stops it starting matches, but the subshells that
# would log the running ones go with it while the matches run on.
#
# Written by Mark Castelluccio, with assistance from Claude (Anthropic).
cd "$(dirname "$0")"
MAX=${MAX:-8}
cat "$@" | while read -r label rest; do
    [ -n "$label" ] || continue
    while [ "$(pgrep -f gtp_match.py | wc -l)" -ge "$MAX" ]; do sleep 15; done
    ( s=$(date +%s); ./pachi-match.sh $label $rest; rc=$?; echo "$label start $s end $(date +%s) exit $rc" >> queue.log ) &
    sleep 10
done
wait
