#!/bin/bash
# New 9x9 preset from scratch vs the old-preset 9x9-main (same 6x64 net, seed 1): equal generations, equal SGD steps, and final.
cd "$(dirname "$0")/.." || exit 1
until grep -q "train exit" runs/scratch-recipe.log; do sleep 60; done
B=runs/bin-scratch; E=runs/scratch-eval; mkdir -p $E; S=runs/9x9-scratch/checkpoints; M=runs/9x9-main/checkpoints
run(){ $B/arena --a "$1" --b "$2" --games 400 --simulations 200 --komi 7.5 --backend blas > $E/$3.txt 2>&1; echo "$3: $(head -1 $E/$3.txt)"; }
echo "start $(date)"
run $S/gen_0050.azn $M/gen_0050.azn g50-vs-main-g50
run $S/gen_0100.azn $M/gen_0100.azn g100-vs-main-g100
run $S/gen_0150.azn $M/gen_0150.azn g150-vs-main-g150
run $S/gen_0075.azn $M/gen_0150.azn g75-vs-main-g150-equal-steps
run $S/gen_0125.azn $M/gen_0250.azn g125-vs-main-g250-equal-steps
run $S/gen_0150.azn $M/gen_0250.azn g150-vs-main-g250
echo "all done $(date)"
