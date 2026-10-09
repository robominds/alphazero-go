#!/bin/bash
# Continue the new-preset from-scratch run from generation 150 to 250, then 400-game matches.
cd "$(dirname "$0")/.." || exit 1
B=runs/bin-scratch; E=runs/scratch-eval; S=runs/9x9-scratch/checkpoints; M=runs/9x9-main/checkpoints
echo "continue start $(date)"
$B/train --run runs/9x9-scratch --resume --generations 250 >> runs/9x9-scratch.out 2>&1
echo "train exit $? $(date)"
run(){ $B/arena --a "$1" --b "$2" --games 400 --simulations 200 --komi 7.5 --backend blas > $E/$3.txt 2>&1; echo "$3: $(head -1 $E/$3.txt)"; }
run $S/gen_0200.azn $S/gen_0150.azn g200-vs-g150
run $S/gen_0250.azn $S/gen_0150.azn g250-vs-g150
run $S/gen_0250.azn $M/gen_0250.azn g250-vs-main-g250
echo "all done $(date)"
