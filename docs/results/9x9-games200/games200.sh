#!/bin/bash
# Data-limit test: same start and settings as runs/9x9-sims200 but 200 self-play games per generation.
set -uo pipefail
cd .
B=runs/bin-games; E=runs/games200-eval; mkdir -p $E
START=runs/9x9-wide96/checkpoints/gen_0050.azn
echo "start $(date)"
$B/train --config configs/9x9.cfg --run runs/9x9-games200 --init-from $START --buffer-from runs/9x9-wide96 \
  --set filters=96 --set lr_schedule=0:0.002 --set simulations=200 --set bot_simulations=100 \
  --set games_per_generation=200 --generations 50 > runs/9x9-games200.out 2>&1
echo "train exit $? $(date)"
run(){ $B/arena --a "$1" --b "$2" --games 100 --simulations 200 --komi 7.5 --backend blas > $E/$3.txt 2>&1; echo "$3: $(head -2 $E/$3.txt | tr '\n' ' ')"; }
G=runs/9x9-games200/checkpoints; L=runs/9x9-sims200/checkpoints
run $G/gen_0025.azn $L/gen_0025.azn g25-vs-s200g25
run $G/gen_0050.azn $L/gen_0050.azn g50-vs-s200g50
run $G/gen_0050.azn $START g50-vs-start
$B/train --run runs/9x9-games200 --rerate --set bot_simulations= --set reference_networks=runs/9x9-main/checkpoints/gen_0200.azn --set arena_opponents=2 > runs/9x9-games200-rerate-ref200.out 2>&1
echo "rerate exit $? $(date)"
echo "all done $(date)"
