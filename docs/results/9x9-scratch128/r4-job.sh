#!/bin/bash
# Round 4: the 9x9 preset from scratch with a 10x128 network, self-play and arena on the GPU (backend=metal).
# Does a bigger network keep improving where 6x64 levels off (9x9-scratch: +99, +115, +135 Elo at gens 300/350/400 vs 250)?
# bot_simulations=100 is kept only so the run is rated like 9x9-main and 9x9-scratch.
cd "$(dirname "$0")/.." || exit 1
B=runs/bin-r4; JOB=scratch128; E=runs/r4-eval/$JOB; mkdir -p $E
run(){ [ -f "$1" ] && [ -f "$2" ] || { echo "$3: skipped (missing $1 or $2)"; return; }
  $B/arena --a "$1" --b "$2" --games 400 --simulations 200 --komi 7.5 --backend metal > $E/$3.txt 2>&1; echo "$3: $(head -1 $E/$3.txt)"; }
g(){ printf "%s/gen_%04d.azn" "$1" "$2"; }
S1=runs/9x9-scratch/checkpoints; N=runs/9x9-scratch128/checkpoints
echo "$JOB start $(date) on $(hostname -s)"
# Warm-up: the random 10x128 network's value head saturates (+1.00 at the empty board, CPU and GPU agree),
# so with resignation on from generation 1 the first try collapsed to 7-ply games by generation 2
# (kept as runs/9x9-scratch128-collapsed). Generations 1-10 therefore play every game out; resignation
# (tunable on resume) is switched on from generation 11.
if [ ! -f runs/9x9-scratch128/state.bin ]; then
  $B/train --config configs/9x9.cfg --run runs/9x9-scratch128 --set blocks=10 --set filters=128 --set backend=metal \
    --set generations=10 --set bot_simulations=100 --set resignation=false > runs/9x9-scratch128.out 2>&1
  echo "warm-up exit $? $(date)"
fi
$B/train --run runs/9x9-scratch128 --resume --set resignation=true --generations 250 >> runs/9x9-scratch128.out 2>&1
echo "train exit $? $(date)"
for k in 50 100 150 250; do
  $B/grow --from $(g $S1 $k) --blocks 10 --filters 128 --out $E/scratch64-g$k-as-10x128.azn > $E/grow-g$k.txt 2>&1
  run $(g $N $k) $E/scratch64-g$k-as-10x128.azn g$k-vs-scratch64-g$k
done
$B/grow --from $(g $S1 400) --blocks 10 --filters 128 --out $E/scratch64-g400-as-10x128.azn > $E/grow-g400.txt 2>&1
run $(g $N 250) $E/scratch64-g400-as-10x128.azn g250-vs-scratch64-g400
echo "$JOB all done $(date)"
