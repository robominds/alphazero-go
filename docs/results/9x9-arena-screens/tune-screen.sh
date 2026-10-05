#!/bin/bash
# Arena-only AGZ search tuning: gen 250 vs itself, one setting changed on side A, temperature 0 with symmetry on both sides.
cd .
until grep -q "all done" runs/sym-screen.log; do sleep 60; done
B=runs/bin-tune; E=runs/tune-screen; mkdir -p $E
G=runs/9x9-main/checkpoints/gen_0250.azn
echo "start $(date)"
run(){ name=$1; shift; $B/arena --a $G --b $G "$@" --temperature-moves 0 --random-symmetry --games 400 --simulations 200 --komi 7.5 --backend blas > $E/$name.txt 2>&1; echo "$name: $(grep -v '^Search' $E/$name.txt | head -1)"; }
run cpuct1 --a-c-puct 1
run cpuct2 --a-c-puct 2
run cpuct3 --a-c-puct 3
run batch1 --a-mcts-batch 1
echo "all done $(date)"
