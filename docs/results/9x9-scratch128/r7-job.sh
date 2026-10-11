#!/bin/bash
# Round 7 (Mac Studio): the 10x128 run again with 400 games per generation (400 steps), halving the replay ratio, because
# 9x9-scratch128 (200 games) overfit its value head (gen 250: fresh-game value loss 0.81 vs 0.42 on its buffer) and lost to 6x64.
# Generation 0 batch-norm calibrated (the default); self-play and arena on the GPU.
cd "$(dirname "$0")/.." || exit 1
B=runs/bin-r6; P=runs/9x9-scratch128/checkpoints; JOB=scratch128-g400; E=runs/r7-eval/$JOB; mkdir -p $E
run(){ [ -f "$1" ] && [ -f "$2" ] || { echo "$3: skipped (missing $1 or $2)"; return; }
  $B/arena --a "$1" --b "$2" --games 400 --simulations 200 --komi 7.5 --backend metal > $E/$3.txt 2>&1; echo "$3: $(head -1 $E/$3.txt)"; }
g(){ printf "%s/gen_%04d.azn" "$1" "$2"; }
S1=runs/mini3/9x9-scratch/checkpoints; N=runs/9x9-scratch128-g400/checkpoints
echo "$JOB start $(date) on $(hostname -s)"
if [ -f runs/9x9-scratch128-g400/state.bin ]; then $B/train --run runs/9x9-scratch128-g400 --resume >> runs/9x9-scratch128-g400.out 2>&1
else $B/train --config configs/9x9.cfg --run runs/9x9-scratch128-g400 --set blocks=10 --set filters=128 --set backend=metal \
  --set games_per_generation=400 --set generations=250 --set bot_simulations=100 > runs/9x9-scratch128-g400.out 2>&1; fi
echo "train exit $? $(date)"
for k in 50 100 150 250 400; do
  $B/grow --from $(g $S1 $k) --blocks 10 --filters 128 --out $E/scratch64-g$k-as-10x128.azn > $E/grow-g$k.txt 2>&1
done
for k in 50 100 150 250; do run $(g $N $k) $E/scratch64-g$k-as-10x128.azn g$k-vs-scratch64-g$k; done
run $(g $N 250) $E/scratch64-g400-as-10x128.azn g250-vs-scratch64-g400
for k in 50 100 150 250; do run $(g $N $k) $(g $P $k) g$k-vs-scratch128-g$k; done
echo "$JOB all done $(date)"
