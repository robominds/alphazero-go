#!/bin/bash
# Round 5: a third seed of the 9x9 preset from scratch, 6x64, to replicate 9x9-scratch (seed 1) and 9x9-scratch-s2.
# bn_calibrate=false keeps generation 0 made the same way as seeds 1 and 2; self-play and arena on the GPU (backend=metal),
# which changes only the last bits of evaluations. bot_simulations=100 as in the other runs.
cd "$(dirname "$0")/.." || exit 1
B=runs/bin-r5; JOB=scratch-s3; E=runs/r5-eval/$JOB; mkdir -p $E
run(){ [ -f "$1" ] && [ -f "$2" ] || { echo "$3: skipped (missing $1 or $2)"; return; }
  $B/arena --a "$1" --b "$2" --games 400 --simulations 200 --komi 7.5 --backend metal > $E/$3.txt 2>&1; echo "$3: $(head -1 $E/$3.txt)"; }
g(){ printf "%s/gen_%04d.azn" "$1" "$2"; }
S1=runs/9x9-scratch/checkpoints; N=runs/9x9-scratch-s3/checkpoints
echo "$JOB start $(date) on $(hostname -s)"
if [ -f runs/9x9-scratch-s3/state.bin ]; then $B/train --run runs/9x9-scratch-s3 --resume >> runs/9x9-scratch-s3.out 2>&1
else $B/train --config configs/9x9.cfg --run runs/9x9-scratch-s3 --set seed=3 --set generations=250 --set bot_simulations=100 \
  --set bn_calibrate=false --set backend=metal > runs/9x9-scratch-s3.out 2>&1; fi
echo "train exit $? $(date)"
for k in 50 100 150 250; do run $(g $N $k) $(g $S1 $k) g$k-vs-seed1-g$k; done
echo "$JOB all done $(date)"
