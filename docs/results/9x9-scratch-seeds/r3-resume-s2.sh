#!/bin/bash
# Resume the seed-2 job on its own Mac mini, detached from the Studio (the Studio is about to reboot, and a GNU Parallel
# controller there would take the job down with it). Same binaries and evals as r3-job.sh scratch-s2.
cd "$(dirname "$0")/.." || exit 1
B=runs/bin-r3; JOB=scratch-s2; E=runs/r3-eval/$JOB; mkdir -p $E
run(){ [ -f "$1" ] && [ -f "$2" ] || { echo "$3: skipped (missing $1 or $2)"; return; }
  $B/arena --a "$1" --b "$2" --games 400 --simulations 200 --komi 7.5 --backend blas > $E/$3.txt 2>&1; echo "$3: $(head -1 $E/$3.txt)"; }
g(){ printf "%s/gen_%04d.azn" "$1" "$2"; }
S1=runs/9x9-scratch/checkpoints; M=runs/9x9-main/checkpoints; N=runs/9x9-scratch-s2/checkpoints
echo "$JOB resume $(date) on $(hostname -s)"
echo "--- resumed detached on $(hostname -s) $(date)" >> runs/9x9-scratch-s2.out
$B/train --run runs/9x9-scratch-s2 --resume >> runs/9x9-scratch-s2.out 2>&1
echo "train exit $? $(date)"
for k in 50 100 150 250; do run $(g $N $k) $(g $S1 $k) g$k-vs-seed1-g$k; done
run $(g $N 250) $(g $M 250) g250-vs-main-g250
echo "$JOB all done $(date)"
