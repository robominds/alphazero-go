#!/bin/bash
# Round 3 of 9x9 experiments with the new preset (configs/9x9.cfg), one job per machine, launched with GNU Parallel.
# usage: r3-job.sh scratch96 | scratch-s2 | scratch-to400
#  scratch96     : the preset from scratch with a 6x96 network, 250 generations (capacity from scratch)
#  scratch-s2    : the preset from scratch, 6x64, seed 2, 250 generations (replicates 9x9-scratch, seed 1)
#  scratch-to400 : continue 9x9-scratch (seed 1, 6x64) from generation 250 to 400 (where does it plateau?)
# bot_simulations=100 is kept only so the runs are rated like 9x9-main and 9x9-scratch.
cd "$(dirname "$0")/.." || exit 1
B=runs/bin-r3; JOB=$1; E=runs/r3-eval/$JOB; mkdir -p $E
run(){ [ -f "$1" ] && [ -f "$2" ] || { echo "$3: skipped (missing $1 or $2)"; return; }
  $B/arena --a "$1" --b "$2" --games 400 --simulations 200 --komi 7.5 --backend blas > $E/$3.txt 2>&1; echo "$3: $(head -1 $E/$3.txt)"; }
g(){ printf "%s/gen_%04d.azn" "$1" "$2"; }
S1=runs/9x9-scratch/checkpoints; M=runs/9x9-main/checkpoints
echo "$JOB start $(date) on $(hostname -s)"
case $JOB in
scratch96)
  $B/train --config configs/9x9.cfg --run runs/9x9-scratch96 --set filters=96 --set generations=250 --set bot_simulations=100 > runs/9x9-scratch96.out 2>&1
  echo "train exit $? $(date)"
  N=runs/9x9-scratch96/checkpoints
  for k in 50 100 150 250; do
    $B/grow --from $(g $S1 $k) --filters 96 --out $E/scratch-g$k-as-6x96.azn > $E/grow-g$k.txt 2>&1
    run $(g $N $k) $E/scratch-g$k-as-6x96.azn g$k-vs-scratch64-g$k
  done
  run $(g $N 250) $(g runs/9x9-recipe/checkpoints 250) g250-vs-recipe-g250 ;;
scratch-s2)
  $B/train --config configs/9x9.cfg --run runs/9x9-scratch-s2 --set seed=2 --set generations=250 --set bot_simulations=100 > runs/9x9-scratch-s2.out 2>&1
  echo "train exit $? $(date)"
  N=runs/9x9-scratch-s2/checkpoints
  for k in 50 100 150 250; do run $(g $N $k) $(g $S1 $k) g$k-vs-seed1-g$k; done
  run $(g $N 250) $(g $M 250) g250-vs-main-g250 ;;
scratch-to400)
  $B/train --run runs/9x9-scratch --resume --generations 400 >> runs/9x9-scratch.out 2>&1
  echo "train exit $? $(date)"
  run $(g $S1 300) $(g $S1 250) g300-vs-g250
  run $(g $S1 350) $(g $S1 250) g350-vs-g250
  run $(g $S1 400) $(g $S1 250) g400-vs-g250
  run $(g $S1 400) $(g $M 250) g400-vs-main-g250 ;;
*) echo "unknown job $JOB"; exit 2 ;;
esac
echo "$JOB all done $(date)"
