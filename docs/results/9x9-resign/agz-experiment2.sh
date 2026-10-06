#!/bin/bash
# AGZ-only experiment arm: same start, seed and settings as runs/9x9-sims200 (the control) plus extra --set overrides.
# usage: agz-experiment.sh NAME [--set k=v ...]
set -uo pipefail
cd .
NAME=$1; shift
B=${B:-runs/bin-agz}; E=runs/$NAME-eval; mkdir -p $E
START=runs/9x9-wide96/checkpoints/gen_0050.azn
echo "$NAME start $(date)"
$B/train --config configs/9x9.cfg --run runs/9x9-$NAME --init-from $START --buffer-from runs/9x9-wide96 \
  --set filters=96 --set lr_schedule=0:0.002 --set simulations=200 --set bot_simulations=100 "$@" \
  --generations 50 > runs/9x9-$NAME.out 2>&1
echo "$NAME train exit $? $(date)"
run(){ $B/arena --a "$1" --b "$2" --games 400 --simulations 200 --komi 7.5 --backend blas > $E/$3.txt 2>&1; echo "$NAME $3: $(head -1 $E/$3.txt)"; }
G=runs/9x9-$NAME/checkpoints; L=runs/9x9-sims200/checkpoints
run $G/gen_0025.azn $L/gen_0025.azn g25-vs-control-g25
run $G/gen_0050.azn $L/gen_0050.azn g50-vs-control-g50
run $G/gen_0050.azn $START g50-vs-start
echo "$NAME all done $(date)"
