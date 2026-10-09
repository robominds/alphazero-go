#!/bin/bash
# Continue the 6x96 recipe arm from generation 200 to 250, then 400-game matches.
cd "$(dirname "$0")/.." || exit 1
B=runs/bin-next; E=runs/recipe-eval; G=runs/9x9-recipe/checkpoints
START=runs/9x9-wide96/checkpoints/gen_0050.azn
echo "continue start $(date)"
$B/train --run runs/9x9-recipe --resume --generations 250 >> runs/9x9-recipe.out 2>&1
echo "train exit $? $(date)"
run(){ $B/arena --a "$1" --b "$2" --games 400 --simulations 200 --komi 7.5 --backend blas > $E/$3.txt 2>&1; echo "$3: $(head -1 $E/$3.txt)"; }
run $G/gen_0250.azn $G/gen_0200.azn g250-vs-g200
run $G/gen_0250.azn $START g250-vs-start
echo "all done $(date)"
