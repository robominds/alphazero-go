#!/bin/sh
# Run the matches listed in a job file (one per line: LABEL CHECKPOINT
# SIMULATIONS SEED [MCTS_BATCH]), JOBS (default 8) at a time, with GNU
# Parallel; each match is two single-threaded processes (our engine with
# the BLAS backend, GNU Go).
#
#   gnugo-jobs.sh JOBFILE
#
# Written by Mark Castelluccio, with assistance from Claude (Anthropic).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
parallel -j "${JOBS:-8}" --colsep ' ' --joblog "$1.joblog" \
    "$here/gnugo-match.sh" {1} {2} {3} {4} {5} :::: "$1"
