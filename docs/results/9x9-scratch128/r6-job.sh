#!/bin/bash
# Round 6 (Mac Studio): the 9x9 preset from scratch with a 10x128 network, generation 0 batch-norm calibrated (the default),
# self-play and arena on the GPU. Started after the Studio was rebooted.
# Earlier tries on a Mac mini without calibration collapsed by generation 2 (runs/9x9-scratch128-collapsed*, on that mini).
cd "$(dirname "$0")/.." || exit 1
B=runs/bin-r6; JOB=scratch128; E=runs/r6-eval/$JOB; mkdir -p $E
run(){ [ -f "$1" ] && [ -f "$2" ] || { echo "$3: skipped (missing $1 or $2)"; return; }
  $B/arena --a "$1" --b "$2" --games 400 --simulations 200 --komi 7.5 --backend metal > $E/$3.txt 2>&1; echo "$3: $(head -1 $E/$3.txt)"; }
g(){ printf "%s/gen_%04d.azn" "$1" "$2"; }
S1=runs/mini3/9x9-scratch/checkpoints; N=runs/9x9-scratch128/checkpoints
echo "$JOB start $(date) on $(hostname -s)"
if [ -f runs/9x9-scratch128/state.bin ]; then $B/train --run runs/9x9-scratch128 --resume >> runs/9x9-scratch128.out 2>&1
else $B/train --config configs/9x9.cfg --run runs/9x9-scratch128 --set blocks=10 --set filters=128 --set backend=metal \
  --set generations=250 --set bot_simulations=100 > runs/9x9-scratch128.out 2>&1; fi
echo "train exit $? $(date)"
for k in 50 100 150 250 400; do
  $B/grow --from $(g $S1 $k) --blocks 10 --filters 128 --out $E/scratch64-g$k-as-10x128.azn > $E/grow-g$k.txt 2>&1
done
for k in 50 100 150 250; do run $(g $N $k) $E/scratch64-g$k-as-10x128.azn g$k-vs-scratch64-g$k; done
run $(g $N 250) $E/scratch64-g400-as-10x128.azn g250-vs-scratch64-g400
echo "$JOB all done $(date)"
