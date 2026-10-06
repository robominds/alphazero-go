#!/bin/bash
# Overnight queue of config-only AGZ experiments (run one after another).
cd .
runs/agz-experiment.sh temp7 --set temperature_moves=7
runs/agz-experiment.sh budget-g200-s400 --set games_per_generation=200 --set train_steps=400
echo "queue done $(date)"
