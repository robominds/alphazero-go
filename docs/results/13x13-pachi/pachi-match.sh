#!/bin/sh
# Play one 13x13 match of GAMES games (komi 7.5, colors
# alternating) between a player A and a player B from the ladder below.
#
#   pachi-match.sh LABEL PLAYER OPPONENT SEED GAMES [SIMULATIONS [MCTS_BATCH]]
#
# PLAYER, engine A, is one of
#   nets/NAME.azn  an alphazero-go checkpoint: our gtp with the backend
#                  BACKEND (blas, one thread, or metal, the GPU; default
#                  the contents of the file ./backend, else blas; it is
#                  written to matches/LABEL/backend.txt), SIMULATIONS
#                  (default 800, the
#                  run's) simulations, leaf batch MCTS_BATCH (default 8,
#                  the run's), --random-symmetry and --seed SEED; it scores
#                  the games (Tromp-Taylor).
#   gnugo10        GNU Go 3.8 at level 10 with the flags of ../13x13-gnugo/.
#   gnugo0         the same at level 0.
#   pachi:SETTING  Pachi, as for B below, with the seed SEED + 1000.
# OPPONENT, engine B, is Pachi 12.90 (no DCNN, no joseki module, Chinese
# rules, no board diagrams in its log, random seed SEED) with one of
#   uct=N          its tree search (the default engine) at -t =N (N >= 500,
#                  Pachi's minimum), one thread, no pondering (its
#                  default), pass_all_alive (no pass until the opponent's
#                  dead stones are captured, like GNU Go's
#                  --capture-all-dead) and resignation off
#                  (resign_threshold=0 allow_losing_pass): the settings of
#                  the 9x9 work's main results.
#   uctnp=N        the same with --nopatterns: no MM patterns to give the
#                  tree its priors.
#   uctlight=N     the same as uctnp=N with playout=light: uniformly
#                  random playouts instead of Pachi's playout policy.
#   replay         -e replay: no search; each move is the move Pachi's
#                  playout policy plays most often in 1,000 samples from
#                  the position (its default).
#   pattern        -e pattern: no search; the move its MM patterns rate
#                  highest.
# -t =N stops the search at the first check, every 50 ms (the very first
# after 100 ms), after the root holds N playouts, counting those kept
# from the previous move's tree, so Pachi searches more than N, and how
# much more depends on its speed; summarize.py reports the counts it
# reached. The engines without a search never resign and pass only when
# no move is left to them (pattern, in practice, never; its games end at
# the move cap).
#
# A LABEL listed in the file ./skip is not played (the script exits at
# once; used to trim a running batch).
#
# Writes matches/LABEL/ (results.csv, one SGF per game, both engines'
# stderr) and matches/LABEL/summary.txt (gtp_match.py's output). A is
# the scorer; both engines' final_score are recorded and summarize.py
# recounts every finished game Tromp-Taylor. Run from the directory
# holding nets/ and matches/. GTP is our gtp binary (default bin/gtp,
# built with -DAZ_USE_BLAS=ON, or bin/gtp-metal, built with
# -DAZ_USE_METAL=ON, for the metal backend), GTP_MATCH the repository's
# tools/gtp_match.py (default bin/gtp_match.py), GNUGO the GNU Go binary
# (default bin/gnugo), PACHI the Pachi binary (default
# bin/pachi-12.90/pachi, with patterns_mm.gamma and patterns_mm.spat
# next to it). pachi-jobs.sh runs a list of these with GNU Parallel.
#
# Written by Mark Castelluccio, with assistance from Claude (Anthropic).
set -eu
grep -qx "$1" skip 2>/dev/null && exit 0
label=$1 player=$2 opponent=$3 seed=$4 games=$5 sims=${6:-800} batch=${7:-8}
BACKEND=${BACKEND:-$(cat backend 2>/dev/null || echo blas)}
if [ "$BACKEND" = metal ]; then GTP=${GTP:-bin/gtp-metal}; else GTP=${GTP:-bin/gtp}; fi
GTP_MATCH=${GTP_MATCH:-bin/gtp_match.py}
GNUGO=${GNUGO:-bin/gnugo}
PACHI=${PACHI:-bin/pachi-12.90/pachi}
search="threads=1 pass_all_alive resign_threshold=0 allow_losing_pass"
pachi() {   # pachi SEED SETTING: one Pachi command line
    p="$PACHI -s $1 --nojoseki -D -r chinese"
    case $2 in
        uct=*) echo "$p -t =${2#uct=} $search" ;;
        uctnp=*) echo "$p --nopatterns -t =${2#uctnp=} $search" ;;
        uctlight=*) echo "$p --nopatterns -t =${2#uctlight=} $search playout=light" ;;
        replay|pattern) echo "$p -e $2" ;;
        *) echo "unknown Pachi setting $2" >&2; return 2 ;;
    esac
}
case $player in
    gnugo10|gnugo0)
        a="$GNUGO --mode gtp --level ${player#gnugo} --chinese-rules --capture-all-dead --positional-superko" ;;
    *.azn)
        a="$GTP --checkpoint $player --backend $BACKEND --simulations $sims --mcts-batch $batch --random-symmetry --seed $seed" ;;
    pachi:*)
        a=$(pachi $((seed + 1000)) "${player#pachi:}") ;;
    *)
        echo "unknown player $player" >&2; exit 2 ;;
esac
b=$(pachi "$seed" "$opponent")
mkdir -p matches
python3 "$GTP_MATCH" --a "$a" --b "$b" \
    --games "$games" --size 13 --komi 7.5 --out "matches/$label" > "matches/$label.summary.tmp" 2>&1
echo "$BACKEND" > "matches/$label/backend.txt"
mv "matches/$label.summary.tmp" "matches/$label/summary.txt"
