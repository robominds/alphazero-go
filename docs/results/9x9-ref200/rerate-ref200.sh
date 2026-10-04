#!/bin/bash
# Re-rate the post-250 experiment runs against one shared fixed reference (9x9-main gen 200), no bots.
cd "$(dirname "$0")/../../.."
REF=runs/9x9-main/checkpoints/gen_0200.azn
for r in 9x9-sims200 9x9-sims400 9x9-ownership 9x9-wide96 9x9-grow10; do
  rm -rf runs/$r/rerate
  runs/bin-refnets/train --run runs/$r --rerate --set bot_simulations= --set reference_networks=$REF --set arena_opponents=2 > runs/$r-rerate-ref200.out 2>&1
  echo "$r exit $? $(date)"
done
echo "all done $(date)"
