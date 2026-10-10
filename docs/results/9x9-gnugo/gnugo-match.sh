#!/bin/sh
# Play one 100-game match between an alphazero-go checkpoint and GNU Go 3.8
# at level 10 on 9x9, komi 7.5, scored by our engine (Tromp-Taylor).
#
#   gnugo-match.sh LABEL CHECKPOINT SIMULATIONS SEED [MCTS_BATCH]
#
# MCTS_BATCH (default 8, the presets' mcts_batch) is the search's leaf
# batch. Below about 64 simulations use 1: with 8 leaves in flight, virtual
# loss spreads a batch over 8 different root moves, so an 8-simulation
# search visits 8 moves once each and picks one of them at random.
#
# Writes matches/LABEL/ (results.csv, one SGF per game, both engines'
# stderr) and matches/LABEL/summary.txt (gtp_match.py's output). Run from
# the directory holding nets/ and matches/; AZ is the alphazero-go source
# tree with build-blas built (-DAZ_USE_BLAS=ON), GNUGO the GNU Go binary.
# gnugo-jobs.sh runs a list of these with GNU Parallel.
#
# Written by Mark Castelluccio, with assistance from Claude (Anthropic).
set -eu
label=$1 ckpt=$2 sims=$3 seed=$4 batch=${5:-8}
AZ=${AZ:-$HOME/alphazero-go}
GNUGO=${GNUGO:-$HOME/bin/gnugo}
mkdir -p matches
python3 "$AZ/tools/gtp_match.py" \
    --a "$AZ/build-blas/gtp --checkpoint $ckpt --backend blas --simulations $sims --mcts-batch $batch --random-symmetry --seed $seed" \
    --b "$GNUGO --mode gtp --level 10 --chinese-rules --capture-all-dead --positional-superko" \
    --games 100 --size 9 --komi 7.5 --out "matches/$label" > "matches/$label.summary.tmp" 2>&1
mv "matches/$label.summary.tmp" "matches/$label/summary.txt"
