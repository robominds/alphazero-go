#!/bin/sh
# Play one 9x9 match of GAMES (default 100) games (komi 7.5, colors
# alternating) against Pachi 12.90 at a fixed number of playouts per move.
#
#   pachi-match.sh LABEL PLAYER PLAYOUTS SEED [SIMULATIONS [MCTS_BATCH]]
#
# PLAYER, engine A, is one of
#   nets/NAME.azn  an alphazero-go checkpoint: our gtp with the BLAS
#                  backend, SIMULATIONS (default 200) simulations, leaf
#                  batch MCTS_BATCH (default 8; use 1 below about 64
#                  simulations, see ../9x9-gnugo/), --random-symmetry and
#                  --seed SEED; it scores the games (Tromp-Taylor).
#   gnugo          GNU Go 3.8 at level 10 with the flags of ../9x9-gnugo/.
#   pachi=N        Pachi itself (PACHI) at N playouts per move, with the
#                  seed SEED + 1000, to link the Pachi levels.
#   PATH           another Pachi binary, with the same settings as B and
#                  the seed SEED + 1000 (to compare two builds).
# Engine B is Pachi: no DCNN, no joseki module (19x19 only), Chinese rules,
# random seed SEED, -t =PLAYOUTS, one thread, no pondering (its default),
# resigning at its default threshold, pass_all_alive (no pass until the
# opponent's dead stones are captured, like GNU Go's --capture-all-dead),
# and no board diagrams in its log (-D). -t =PLAYOUTS stops the search at
# the first check, every 50 ms (the very first after 100 ms), after the
# root holds PLAYOUTS playouts, counting those kept from the previous
# move's tree, so Pachi searches somewhat more than PLAYOUTS, and how
# much more depends on its speed; summarize.py reports the counts it
# reached.
# PACHI_ARGS, if set, is added to the arguments of every Pachi in the
# match (the matches with Pachi's resignation off used
# PACHI_ARGS="resign_threshold=0 allow_losing_pass" and a label ending in
# -nr). Engine A is the scorer; both engines'
# final_score are recorded and summarize.py recounts every finished game
# Tromp-Taylor.
#
# Writes matches/LABEL/ (results.csv, one SGF per game, both engines'
# stderr) and matches/LABEL/summary.txt (gtp_match.py's output). Run from
# the directory holding nets/ and matches/; AZ is the alphazero-go source
# tree with build-blas built (-DAZ_USE_BLAS=ON), GNUGO the GNU Go binary,
# PACHI the Pachi binary (patterns_mm.gamma and patterns_mm.spat next to
# it). pachi-jobs.sh runs a list of these with GNU Parallel.
#
# Written by Mark Castelluccio, with assistance from Claude (Anthropic).
set -eu
label=$1 player=$2 playouts=$3 seed=$4 sims=${5:-200} batch=${6:-8}
AZ=${AZ:-$HOME/alphazero-go}
GNUGO=${GNUGO:-$HOME/bin/gnugo}
PACHI=${PACHI:-$HOME/bin/pachi-12.90/pachi}
settings="--nojoseki -D -r chinese -t =$playouts threads=1 pass_all_alive"
case $player in
    gnugo)
        a="$GNUGO --mode gtp --level 10 --chinese-rules --capture-all-dead --positional-superko" ;;
    *.azn)
        a="$AZ/build-blas/gtp --checkpoint $player --backend blas --simulations $sims --mcts-batch $batch --random-symmetry --seed $seed" ;;
    pachi=*)
        a="$PACHI -s $((seed + 1000)) --nojoseki -D -r chinese -t =${player#pachi=} threads=1 pass_all_alive ${PACHI_ARGS:-}" ;;
    *)
        a="$player -s $((seed + 1000)) $settings ${PACHI_ARGS:-}" ;;
esac
mkdir -p matches
python3 "$AZ/tools/gtp_match.py" \
    --a "$a" \
    --b "$PACHI -s $seed $settings ${PACHI_ARGS:-}" \
    --games "${GAMES:-100}" --size 9 --komi 7.5 --out "matches/$label" > "matches/$label.summary.tmp" 2>&1
mv "matches/$label.summary.tmp" "matches/$label/summary.txt"
