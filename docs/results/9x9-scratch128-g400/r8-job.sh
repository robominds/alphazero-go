#!/bin/bash
# Round 8 (Mac Studio): continue 9x9-scratch128-g400 (10x128, 400 games and 400 steps per generation) from 250 to 300,
# then 400-game matches against its own gen 250 and against the 6x64 seed-1 network's gen 400.
cd "$(dirname "$0")/.." || exit 1
B=runs/bin-r6; JOB=scratch128-g400-to300; E=runs/r8-eval/$JOB; mkdir -p $E
run(){ [ -f "$1" ] && [ -f "$2" ] || { echo "$3: skipped (missing $1 or $2)"; return; }
  $B/arena --a "$1" --b "$2" --games 400 --simulations 200 --komi 7.5 --backend metal > $E/$3.txt 2>&1; echo "$3: $(head -1 $E/$3.txt)"; }
g(){ printf "%s/gen_%04d.azn" "$1" "$2"; }
N=runs/9x9-scratch128-g400/checkpoints
echo "$JOB start $(date) on $(hostname -s)"
echo "--- continued to 300 $(date)" >> runs/9x9-scratch128-g400.out
$B/train --run runs/9x9-scratch128-g400 --resume --generations 300 >> runs/9x9-scratch128-g400.out 2>&1
echo "train exit $? $(date)"
run $(g $N 300) $(g $N 250) g300-vs-g250
run $(g $N 300) runs/r7-eval/scratch128-g400/scratch64-g400-as-10x128.azn g300-vs-scratch64-g400
echo "$JOB all done $(date)"
