#!/bin/bash
# Follow-up c_puct points around the default 1.5, same protocol as tune-screen.sh.
cd .
B=runs/bin-tune; E=runs/tune-screen; mkdir -p $E
G=runs/9x9-main/checkpoints/gen_0250.azn
echo "start $(date)"
run(){ name=$1; shift; $B/arena --a $G --b $G "$@" --temperature-moves 0 --random-symmetry --games 400 --simulations 200 --komi 7.5 --backend blas > $E/$name.txt 2>&1; echo "$name: $(grep -v '^Search' $E/$name.txt | head -1)"; }
run cpuct0.75 --a-c-puct 0.75
run cpuct1.25 --a-c-puct 1.25
echo "all done $(date)"
