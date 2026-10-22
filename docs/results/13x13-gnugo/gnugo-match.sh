#!/bin/sh
# Play one 13x13 match of GAMES (default 100) games between an
# alphazero-go checkpoint and GNU Go 3.8 at level LEVEL (default 10),
# komi 7.5, colors alternating, scored by our engine (Tromp-Taylor).
#
#   gnugo-match.sh LABEL CHECKPOINT SIMULATIONS SEED \
#                  [MCTS_BATCH [GAMES [LEVEL]]]
#
# Our engine is gtp with the backend BACKEND (blas, one thread, or
# metal, the GPU; default the contents of the file ./backend, else
# blas), SIMULATIONS simulations, leaf batch MCTS_BATCH (default 8, the
# run's mcts_batch), --random-symmetry and --seed SEED (GNU Go is
# deterministic, so the seed is what makes the games differ). GNU Go
# plays at --level LEVEL with --chinese-rules --capture-all-dead
# --positional-superko, as on 9x9. A LABEL listed in the file ./skip is
# not played (the script exits at once; used to trim a running batch).
# A long match is played as several chunks of an even number of games,
# each with its own label (LABEL-cNN) and seed; summarize.py pools them.
# Labels of matches against a level other than 10 end in -lLEVEL before
# the chunk number.
#
# Writes matches/LABEL/ (results.csv, one SGF per game, both engines'
# stderr) and matches/LABEL/summary.txt (gtp_match.py's output). Run from
# the directory holding nets/ and matches/; the backend used is written
# to matches/LABEL/backend.txt. GTP is our gtp binary (default bin/gtp,
# built with -DAZ_USE_BLAS=ON, or bin/gtp-metal, built with
# -DAZ_USE_METAL=ON, for the metal backend), GTP_MATCH the
# repository's tools/gtp_match.py (default bin/gtp_match.py), GNUGO the
# GNU Go binary (default bin/gnugo). gnugo-jobs.sh runs a list of these
# with GNU Parallel.
#
# Written by Mark Castelluccio, with assistance from Claude (Anthropic).
set -eu
grep -qx "$1" skip 2>/dev/null && exit 0
label=$1 ckpt=$2 sims=$3 seed=$4 batch=${5:-8} games=${6:-100} level=${7:-10}
BACKEND=${BACKEND:-$(cat backend 2>/dev/null || echo blas)}
if [ "$BACKEND" = metal ]; then GTP=${GTP:-bin/gtp-metal}; else GTP=${GTP:-bin/gtp}; fi
GTP_MATCH=${GTP_MATCH:-bin/gtp_match.py}
GNUGO=${GNUGO:-bin/gnugo}
mkdir -p matches
python3 "$GTP_MATCH" \
    --a "$GTP --checkpoint $ckpt --backend $BACKEND --simulations $sims --mcts-batch $batch --random-symmetry --seed $seed" \
    --b "$GNUGO --mode gtp --level $level --chinese-rules --capture-all-dead --positional-superko" \
    --games "$games" --size 13 --komi 7.5 --out "matches/$label" > "matches/$label.summary.tmp" 2>&1
echo "$BACKEND" > "matches/$label/backend.txt"
mv "matches/$label.summary.tmp" "matches/$label/summary.txt"
