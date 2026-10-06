#!/bin/bash
# Re-rate finished AGZ arms against gen 200 of 9x9-main (same method as the other post-250 runs).
cd .
rr(){ $1/train --run runs/9x9-$2 --rerate --set bot_simulations= --set reference_networks=runs/9x9-main/checkpoints/gen_0200.azn --set arena_opponents=2 > runs/9x9-$2-rerate-ref200.out 2>&1; echo "$2 rerate exit $? $(date)"; }
rr runs/bin-resign resign
until grep -q "temp7 all done" runs/agz-queue.log; do sleep 60; done
rr runs/bin-agz temp7
echo "all done $(date)"
