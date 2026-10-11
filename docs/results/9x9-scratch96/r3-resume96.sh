#!/bin/bash
# Resume the Studio's scratch96 job on binaries with the multi-threaded training step (main bea6ada).
# Results are byte-identical at any train_threads and to the old single-threaded step (measured for
# Accelerate on this M5 Max), so switching binaries mid-run does not change the run. Same evals as r3-job.sh.
cd "$(dirname "$0")/.." || exit 1
B=runs/bin-r3b; JOB=scratch96; E=runs/r3-eval/$JOB; mkdir -p $E
run(){ [ -f "$1" ] && [ -f "$2" ] || { echo "$3: skipped (missing $1 or $2)"; return; }
  $B/arena --a "$1" --b "$2" --games 400 --simulations 200 --komi 7.5 --backend blas > $E/$3.txt 2>&1; echo "$3: $(head -1 $E/$3.txt)"; }
g(){ printf "%s/gen_%04d.azn" "$1" "$2"; }
S1=runs/9x9-scratch/checkpoints; N=runs/9x9-scratch96/checkpoints
echo "$JOB resume $(date) on $(hostname -s)"
echo "--- resumed on runs/bin-r3b (multi-threaded training) $(date)" >> runs/9x9-scratch96.out
$B/train --run runs/9x9-scratch96 --resume >> runs/9x9-scratch96.out 2>&1
echo "train exit $? $(date)"
for k in 50 100 150 250; do
  $B/grow --from $(g $S1 $k) --filters 96 --out $E/scratch-g$k-as-6x96.azn > $E/grow-g$k.txt 2>&1
  run $(g $N $k) $E/scratch-g$k-as-6x96.azn g$k-vs-scratch64-g$k
done
run $(g $N 250) $(g runs/9x9-recipe/checkpoints 250) g250-vs-recipe-g250
echo "$JOB all done $(date)"
