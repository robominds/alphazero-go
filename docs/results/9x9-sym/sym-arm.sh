#!/bin/bash
# AGZ experiment 2 (self-play arm): random symmetry in self-play search, after the arena tuning screen.
cd .
until grep -q "all done" runs/tune-screen.log; do sleep 60; done
B=runs/bin-tune runs/agz-experiment2.sh sym --set mcts_random_symmetry=true
