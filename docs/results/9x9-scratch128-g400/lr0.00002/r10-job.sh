#!/bin/bash
# Round 10 (Mac Studio): the learning-rate step at generation 300 (0.002 -> 0.0002) gave +115 Elo by generation 350.
# A copy of that run (APFS clone of runs/9x9-scratch128-g400-lr at generation 350) continues to 400 with a second step,
# to 0.00002 at generation 350, as AlphaGo Zero lowered its learning rate in several steps.
cd "$(dirname "$0")/.." || exit 1
B=runs/bin-r6; JOB=scratch128-g400-lr2; E=runs/r10-eval/$JOB; mkdir -p $E
run(){ [ -f "$1" ] && [ -f "$2" ] || { echo "$3: skipped (missing $1 or $2)"; return; }
  $B/arena --a "$1" --b "$2" --games 400 --simulations 200 --komi 7.5 --backend metal > $E/$3.txt 2>&1; echo "$3: $(head -1 $E/$3.txt)"; }
g(){ printf "%s/gen_%04d.azn" "$1" "$2"; }
N=runs/9x9-scratch128-g400-lr2/checkpoints; P=runs/9x9-scratch128-g400-lr/checkpoints
echo "$JOB start $(date) on $(hostname -s)"
echo "--- copy continued to 400 with lr 0.00002 from generation 350 $(date)" >> runs/9x9-scratch128-g400-lr2.out
$B/train --run runs/9x9-scratch128-g400-lr2 --resume --set "lr_schedule=0:0.02,90:0.002,300:0.0002,350:0.00002" \
  --generations 400 >> runs/9x9-scratch128-g400-lr2.out 2>&1
echo "train exit $? $(date)"
run $(g $N 375) $(g $P 350) g375-vs-g350
run $(g $N 400) $(g $P 350) g400-vs-g350
run $(g $N 400) runs/r7-eval/scratch128-g400/scratch64-g400-as-10x128.azn g400-vs-scratch64-g400
echo "$JOB all done $(date)"
