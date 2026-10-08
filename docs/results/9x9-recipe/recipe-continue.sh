#!/bin/bash
# Continue the recipe arm from generation 100 to 200, then 400-game matches and a gen-200 rerate.
cd "$(dirname "$0")/.." || exit 1
B=runs/bin-next; E=runs/recipe-eval; NAME=recipe
START=runs/9x9-wide96/checkpoints/gen_0050.azn; G=runs/9x9-$NAME/checkpoints
echo "continue start $(date) on $(hostname)"
$B/train --run runs/9x9-$NAME --resume --generations 200 >> runs/9x9-$NAME.out 2>&1
echo "train exit $? $(date)"
run(){ $B/arena --a "$1" --b "$2" --games 400 --simulations 200 --komi 7.5 --backend blas > $E/$3.txt 2>&1; echo "$3: $(head -1 $E/$3.txt)"; }
run $G/gen_0150.azn $G/gen_0100.azn g150-vs-g100
run $G/gen_0200.azn $G/gen_0100.azn g200-vs-g100
run $G/gen_0200.azn $START g200-vs-start
$B/train --run runs/9x9-$NAME --rerate --set bot_simulations= --set reference_networks=runs/9x9-main/checkpoints/gen_0200.azn \
  --set arena_opponents=2 > runs/9x9-$NAME-rerate-ref200-to200.out 2>&1
echo "rerate exit $? $(date)"
echo "all done $(date)"
