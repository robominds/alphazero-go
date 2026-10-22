#!/bin/sh
# Run the matches listed in a job file (one per line: LABEL PLAYER
# OPPONENT SEED GAMES [SIMULATIONS [MCTS_BATCH]]) with GNU Parallel, on this
# machine or spread over several; each match is two single-threaded
# engines taking turns.
#
#   pachi-jobs.sh JOBFILE
#
# HOSTS is GNU Parallel's --sshlogin list, with the number of matches to
# run at once on each host (for example "16/:,8/HOST1,8/HOST2"; ":" is
# this machine; default "8/:"). WORKDIR (default ".") is the directory
# the matches run in: relative to the current directory here and to the
# home directory on the other hosts, each holding this script,
# pachi-match.sh, bin/ and nets/ (see pachi-match.sh). The matches'
# output stays on the host that played them, in WORKDIR/matches/.
#
# Written by Mark Castelluccio, with assistance from Claude (Anthropic).
set -eu
parallel -S "${HOSTS:-8/:}" --colsep ' ' --joblog "$1.joblog" \
    "cd ${WORKDIR:-.} && ./pachi-match.sh {1} {2} {3} {4} {5} {6} {7}" :::: "$1"
