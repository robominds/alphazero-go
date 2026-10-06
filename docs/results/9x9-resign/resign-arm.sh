#!/bin/bash
# AGZ experiment 1: resignation, same start/settings as the sims200 control, run alongside the agz queue.
cd .
B=runs/bin-resign runs/agz-experiment2.sh resign --set resignation=true
