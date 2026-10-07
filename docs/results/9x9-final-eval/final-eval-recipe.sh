#!/bin/bash
# Pooled evaluation of the recipe arm: gens 46-50 and 96-100 vs the shared start, 200 games each (same protocol as final-eval.sh).
cd "$(dirname "$0")/.." || exit 1
B=runs/bin-next; E=runs/final-eval; START=runs/9x9-wide96/checkpoints/gen_0050.azn
for g in 46 47 48 49 50 96 97 98 99 100; do
  p=$(printf %04d $g); f=$E/recipe-g$g.txt; [ -s $f ] && continue
  $B/arena --a runs/9x9-recipe/checkpoints/gen_$p.azn --b $START --games 200 --simulations 200 --komi 7.5 --backend blas > $f 2>&1
  echo "recipe g$g: $(head -1 $f | sed "s/.*: A scored/A scored/")"
done
echo "all done $(date)"
