#!/bin/bash
# Round 11 (Mac Studio): continue 9x9-scratch128-g400-lr2 (10x128, 400 games per generation, lr 0.00002 since gen 350)
# from 400 to 450 at the same settings, then 400-game matches against its gen 400, the stepped gen 350 and 6x64 gen 400.
cd "$(dirname "$0")/.." || exit 1
B=runs/bin-r6; JOB=scratch128-g400-lr2-to450; E=runs/r11-eval/$JOB; mkdir -p $E
run(){ [ -f "$1" ] && [ -f "$2" ] || { echo "$3: skipped (missing $1 or $2)"; return; }
  $B/arena --a "$1" --b "$2" --games 400 --simulations 200 --komi 7.5 --backend metal > $E/$3.txt 2>&1; echo "$3: $(head -1 $E/$3.txt)"; }
g(){ printf "%s/gen_%04d.azn" "$1" "$2"; }
N=runs/9x9-scratch128-g400-lr2/checkpoints; P=runs/9x9-scratch128-g400-lr/checkpoints
echo "$JOB start $(date) on $(hostname -s)"
echo "--- continued to 450 $(date)" >> runs/9x9-scratch128-g400-lr2.out
$B/train --run runs/9x9-scratch128-g400-lr2 --resume --generations 450 >> runs/9x9-scratch128-g400-lr2.out 2>&1
echo "train exit $? $(date)"
run $(g $N 450) $(g $N 400) g450-vs-g400
run $(g $N 450) $(g $P 350) g450-vs-g350
run $(g $N 450) runs/r7-eval/scratch128-g400/scratch64-g400-as-10x128.azn g450-vs-scratch64-g400
echo "$JOB all done $(date)"
