#!/bin/bash
# Final-strength comparison: each post-250 arm's gens 46-50 against the shared start (wide96 gen 50), 200 games each,
# so every arm gets 1000 games averaged over five checkpoints (checkpoint-to-checkpoint variance was ~50 Elo).
cd "$(dirname "$0")/.." || exit 1
B=runs/bin-next; E=runs/final-eval; mkdir -p $E
START=runs/9x9-wide96/checkpoints/gen_0050.azn
echo "start $(date) on $(hostname)"
for a in "$@"; do
  for g in 46 47 48 49 50; do
    f=$E/$a-g$g.txt
    [ -s $f ] && continue
    $B/arena --a runs/9x9-$a/checkpoints/gen_00$g.azn --b $START --games 200 --simulations 200 --komi 7.5 --backend blas > $f 2>&1
    echo "$a g$g: $(head -1 $f | sed 's/.*: A scored/A scored/')"
  done
done
echo "all done $(date)"
