#!/bin/bash
# From-scratch 9x9 run with the new preset (temperature 7, resignation, 200 games, 400 steps), after the pooled eval.
# bot100 kept only so its elo.csv is rated like 9x9-main (the bot never affects training).
cd "$(dirname "$0")/.." || exit 1
until grep -q "all done" runs/final-eval-recipe.log 2>/dev/null; do sleep 60; done
echo "start $(date) on $(hostname)"
runs/bin-scratch/train --config configs/9x9.cfg --run runs/9x9-scratch --set bot_simulations=100 > runs/9x9-scratch.out 2>&1
echo "train exit $? $(date)"
