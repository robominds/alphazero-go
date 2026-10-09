#!/bin/bash
# Head-to-head: the 6x96 recipe (continued from wide96) vs the 6x64 new preset from scratch, both at generation 250.
# Arena needs equal shapes, so the 6x64 network is widened to 96 filters with grow, which preserves its function exactly.
cd "$(dirname "$0")/.." || exit 1
B=runs/bin-next; E=runs/versus; mkdir -p $E
until grep -q "all done" runs/recipe-continue2.log; do sleep 60; done
until ssh -o BatchMode=yes [the second Mac mini] 'grep -q "all done" ~/alphazero-go/runs/scratch-continue.log'; do sleep 120; done
echo "start $(date)"
rsync -a [the second Mac mini]:alphazero-go/runs/9x9-scratch [the second Mac mini]:alphazero-go/runs/9x9-scratch.out \
  [the second Mac mini]:alphazero-go/runs/scratch-eval [the second Mac mini]:alphazero-go/runs/scratch-continue.log runs/
for g in 150 250; do
  $B/grow --from runs/9x9-scratch/checkpoints/gen_0$g.azn --filters 96 --out $E/scratch-g$g-as-6x96.azn > $E/grow-g$g.txt 2>&1
  echo "grow g$g exit $?"
done
run(){ $B/arena --a "$1" --b "$2" --games 400 --simulations 200 --komi 7.5 --backend blas > $E/$3.txt 2>&1; echo "$3: $(head -1 $E/$3.txt)"; }
R=runs/9x9-recipe/checkpoints
run $R/gen_0250.azn $E/scratch-g250-as-6x96.azn recipe-g250-vs-scratch-g250
run $R/gen_0200.azn $E/scratch-g250-as-6x96.azn recipe-g200-vs-scratch-g250
run $R/gen_0250.azn $E/scratch-g150-as-6x96.azn recipe-g250-vs-scratch-g150
echo "all done $(date)"
