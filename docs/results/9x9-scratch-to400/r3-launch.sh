#!/bin/bash
# Launch round 3: one job per machine via GNU Parallel over the Thunderbolt bridge (each invocation pins its job to one host).
cd "$(dirname "$0")/.." || exit 1
P='cd ~/projects/alphazero-go 2>/dev/null || cd ~/alphazero-go; caffeinate -i runs/r3-job.sh'
parallel -S 1/: --tag --line-buffer --joblog runs/r3-studio.joblog "$P {}" ::: scratch96 &
parallel -S 1/[the second Mac mini] --tag --line-buffer --joblog runs/r3-mini2.joblog "$P {}" ::: scratch-s2 &
parallel -S 1/[the first Mac mini] --tag --line-buffer --joblog runs/r3-mini3.joblog "$P {}" ::: scratch-to400 &
wait
echo "round 3 all done $(date)"
