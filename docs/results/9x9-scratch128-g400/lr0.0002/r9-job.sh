#!/bin/bash
# Round 9 (Mac Studio): the 10x128 400-games run plateaued between generations 250 and 300 (gen 300 vs 250: 0.472).
# A copy of the run (APFS clone of runs/9x9-scratch128-g400 at generation 300) continues to 350 with the learning rate
# stepped down from 0.002 to 0.0002 at generation 300, as AlphaGo Zero lowered its learning rate in steps.
cd "$(dirname "$0")/.." || exit 1
B=runs/bin-r6; JOB=scratch128-g400-lr; E=runs/r9-eval/$JOB; mkdir -p $E
run(){ [ -f "$1" ] && [ -f "$2" ] || { echo "$3: skipped (missing $1 or $2)"; return; }
  $B/arena --a "$1" --b "$2" --games 400 --simulations 200 --komi 7.5 --backend metal > $E/$3.txt 2>&1; echo "$3: $(head -1 $E/$3.txt)"; }
g(){ printf "%s/gen_%04d.azn" "$1" "$2"; }
N=runs/9x9-scratch128-g400-lr/checkpoints; O=runs/9x9-scratch128-g400/checkpoints
echo "$JOB start $(date) on $(hostname -s)"
echo "--- copy continued to 350 with lr 0.0002 from generation 300 $(date)" >> runs/9x9-scratch128-g400-lr.out
$B/train --run runs/9x9-scratch128-g400-lr --resume --set "lr_schedule=0:0.02,90:0.002,300:0.0002" --generations 350 \
  >> runs/9x9-scratch128-g400-lr.out 2>&1
echo "train exit $? $(date)"
run $(g $N 325) $(g $O 300) g325-vs-g300
run $(g $N 350) $(g $O 300) g350-vs-g300
run $(g $N 350) $(g $O 250) g350-vs-g250
run $(g $N 350) runs/r7-eval/scratch128-g400/scratch64-g400-as-10x128.azn g350-vs-scratch64-g400
echo "$JOB all done $(date)"
