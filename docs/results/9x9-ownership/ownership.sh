#!/bin/bash
# Ownership head: give the 6x96 generation 50 an ownership head, train 50 generations
# with the settings of runs/9x9-sims200 plus ownership = true, then evaluate.
set -uo pipefail
cd "$(dirname "$0")/.."
B=runs/bin-ownership; E=runs/ownership-eval; mkdir -p $E runs/grown
START=runs/grown/wide96-g50-own.azn
EYE="A2 A3 B2 B3 C2 C3 D2 D3 D1 E3 F7 E2 G5 E1"
echo "start $(date)"
$B/grow --from runs/9x9-wide96/checkpoints/gen_0050.azn --ownership --out $START || exit 1
$B/train --config configs/9x9.cfg --run runs/9x9-ownership --init-from $START \
  --buffer-from runs/9x9-wide96 --set filters=96 --set lr_schedule=0:0.002 \
  --set simulations=200 --set ownership=true --generations 50 > runs/9x9-ownership.out 2>&1
echo "train exit $? $(date)"
O=runs/9x9-ownership/checkpoints; L=runs/9x9-sims200/checkpoints
# The 200 arm's checkpoints with a head added (it changes no move), so that arena accepts the pair.
for g in 25 50; do
  $B/grow --from $L/gen_00$g.azn --ownership --out runs/grown/sims200-g$g-own.azn | tail -1
done
run(){ $B/arena --a "$1" --b "$2" --games 100 --simulations 200 --komi 7.5 --backend blas > $E/$3.txt 2>&1; echo "$3: $(head -2 $E/$3.txt | tr '\n' ' ')"; }
run $O/gen_0025.azn $O/gen_0000.azn own25-vs-start
run $O/gen_0050.azn $O/gen_0000.azn own50-vs-start
run $O/gen_0025.azn runs/grown/sims200-g25-own.azn own25-vs-s200g25
run $O/gen_0050.azn runs/grown/sims200-g50-own.azn own50-vs-s200g50
for s in 200 800; do
  $B/analyze --checkpoint $O/gen_0050.azn --moves "$EYE" --simulations $s --komi 7.5 --backend blas > $E/eye-own50-s$s.txt 2>&1
done
# The same group after White's killing B1 (Black first plays elsewhere, C6), and after Black's saving B1.
$B/analyze --checkpoint $O/gen_0050.azn --moves "$EYE C6 B1" --simulations 200 --komi 7.5 --backend blas > $E/eye-own50-killed.txt 2>&1
$B/analyze --checkpoint $O/gen_0050.azn --moves "$EYE B1" --simulations 200 --komi 7.5 --backend blas > $E/eye-own50-saved.txt 2>&1
echo "all done $(date)"
