#!/bin/bash
# Fair eye comparison: each side gets one move. Run from the repository root.
set -u
D=docs/results/9x9-ownership
EYE="A2 A3 B2 B3 C2 C3 D2 D3 D1 E3 F7 E2 G5 E1"
A=runs/bin-ownership/analyze
an(){ $A --checkpoint "$1" --moves "$2" --simulations 800 --komi 7.5 --backend blas > $D/$3.txt 2>&1; echo "$3 exit $?"; }
for pair in "own50 runs/9x9-ownership/checkpoints/gen_0050.azn" "own0 runs/9x9-ownership/checkpoints/gen_0000.azn" "sims200g50 runs/9x9-sims200/checkpoints/gen_0050.azn" "main250 runs/9x9-main/checkpoints/gen_0250.azn"; do
  set -- $pair
  an $2 "$EYE C6 B1" fair-$1-tenuki
  an $2 "$EYE B1 C6" fair-$1-save
done
