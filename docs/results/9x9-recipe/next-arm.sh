#!/bin/bash
# Next-round AGZ arm, runnable on either Mac (repo at ~/projects/alphazero-go or ~/alphazero-go).
# usage: next-arm.sh NAME GENERATIONS [--set k=v ...]
# Same start, seed and settings as the sims200 control plus the overrides; then 400-game matches and a gen-200 rerate.
cd "$(dirname "$0")/.." || exit 1
NAME=$1; GENS=$2; shift 2
B=runs/bin-next; E=runs/$NAME-eval; mkdir -p $E
START=runs/9x9-wide96/checkpoints/gen_0050.azn
echo "$NAME start $(date) on $(hostname)"
$B/train --config configs/9x9.cfg --run runs/9x9-$NAME --init-from $START --buffer-from runs/9x9-wide96 \
  --set filters=96 --set lr_schedule=0:0.002 --set simulations=200 --set bot_simulations=100 "$@" \
  --generations $GENS > runs/9x9-$NAME.out 2>&1
echo "$NAME train exit $? $(date)"
run(){ [ -f "$1" ] && [ -f "$2" ] || { echo "$NAME $3: skipped (missing checkpoint)"; return; }
  $B/arena --a "$1" --b "$2" --games 400 --simulations 200 --komi 7.5 --backend blas > $E/$3.txt 2>&1; echo "$NAME $3: $(head -1 $E/$3.txt)"; }
G=runs/9x9-$NAME/checkpoints; C=runs/9x9-sims200/checkpoints; D=runs/9x9-budget-g200-s400/checkpoints
run $G/gen_0025.azn $C/gen_0025.azn g25-vs-control-g25
run $G/gen_0050.azn $C/gen_0050.azn g50-vs-control-g50
run $G/gen_0050.azn $D/gen_0050.azn g50-vs-budget-g50
run $G/gen_0050.azn $START g50-vs-start
if [ "$GENS" -ge 100 ]; then
  run $G/gen_0100.azn $G/gen_0050.azn g100-vs-g50
  run $G/gen_0100.azn $START g100-vs-start
fi
$B/train --run runs/9x9-$NAME --rerate --set bot_simulations= --set reference_networks=runs/9x9-main/checkpoints/gen_0200.azn \
  --set arena_opponents=2 > runs/9x9-$NAME-rerate-ref200.out 2>&1
echo "$NAME rerate exit $? $(date)"
echo "$NAME all done $(date)"
