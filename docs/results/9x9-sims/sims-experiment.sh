#!/bin/bash
# Two arms from the same start (6x96 gen 50 + its replay buffer): 400 vs 200 simulations, 50 generations each.
set -uo pipefail
cd .
B=runs/bin-sims; E=runs/sims-eval; mkdir -p $E
START=runs/9x9-wide96/checkpoints/gen_0050.azn
COMMON="--config configs/9x9.cfg --init-from $START --buffer-from runs/9x9-wide96 --set filters=96 --set lr_schedule=0:0.002 --generations 50"
echo "start $(date)"
$B/train $COMMON --run runs/9x9-sims400 --set simulations=400 > runs/9x9-sims400.out 2>&1; echo "sims400 exit $? $(date)"
$B/train $COMMON --run runs/9x9-sims200 --set simulations=200 > runs/9x9-sims200.out 2>&1; echo "sims200 exit $? $(date)"
run(){ $B/arena --a "$1" --b "$2" --games 100 --simulations 200 --komi 7.5 --backend blas > $E/$3.txt 2>&1; echo "$3: $(head -2 $E/$3.txt | tr '\n' ' ')"; }
H=runs/9x9-sims400/checkpoints; L=runs/9x9-sims200/checkpoints
run $H/gen_0050.azn $L/gen_0050.azn s400g50-vs-s200g50
run $H/gen_0025.azn $L/gen_0025.azn s400g25-vs-s200g25
run $H/gen_0050.azn $START s400g50-vs-start
run $L/gen_0050.azn $START s200g50-vs-start
for arm in sims400 sims200; do for s in 200 800; do
  $B/analyze --checkpoint runs/9x9-$arm/checkpoints/gen_0050.azn --moves "A2 A3 B2 B3 C2 C3 D2 D3 D1 E3 F7 E2 G5 E1" --simulations $s --komi 7.5 --backend blas > $E/eye-$arm-s$s.txt 2>&1
done; done
echo "all done $(date)"
