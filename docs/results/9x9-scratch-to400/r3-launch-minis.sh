#!/bin/bash
# Relaunch of round 3's two Mac mini jobs (the first launch could not find the job script on them).
cd "$(dirname "$0")/.." || exit 1
P='cd ~/projects/alphazero-go 2>/dev/null || cd ~/alphazero-go; caffeinate -i runs/r3-job.sh'
parallel -S 1/[the second Mac mini] --tag --line-buffer --joblog runs/r3-mini2.joblog "$P {}" ::: scratch-s2 &
parallel -S 1/[the first Mac mini] --tag --line-buffer --joblog runs/r3-mini3.joblog "$P {}" ::: scratch-to400 &
wait
echo "minis done $(date)"
