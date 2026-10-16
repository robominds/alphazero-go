#!/bin/bash
# Round 13 (Mac Studio): does resignation starve the 10x128 network of endgame data? Its losses to Pachi are long, close games
# (median ~92 moves, nearly half by <= 3.5 points) while ~73% of buffer positions come from games resigned at ~30 plies.
# An APFS clone of 9x9-scratch128-g400-lr at generation 350 (400 games, lr 0.0002) continues to 400 with
# resign_false_positive = 0.02 instead of 0.05, so v_resign drops and games are resigned later.
cd "$(dirname "$0")/.." || exit 1
B=runs/bin-r6; JOB=scratch128-fp2; E=runs/r13-eval/$JOB; mkdir -p $E
run(){ [ -f "$1" ] && [ -f "$2" ] || { echo "$3: skipped (missing $1 or $2)"; return; }
  $B/arena --a "$1" --b "$2" --games 400 --simulations 200 --komi 7.5 --backend metal > $E/$3.txt 2>&1; echo "$3: $(head -1 $E/$3.txt)"; }
g(){ printf "%s/gen_%04d.azn" "$1" "$2"; }
N=runs/9x9-scratch128-fp2/checkpoints; P=runs/9x9-scratch128-g400-lr/checkpoints
echo "$JOB start $(date) on $(hostname -s)"
echo "--- copy continued to 400 with resign_false_positive 0.02 from generation 350 $(date)" >> runs/9x9-scratch128-fp2.out
$B/train --run runs/9x9-scratch128-fp2 --resume --set resign_false_positive=0.02 --generations 400 >> runs/9x9-scratch128-fp2.out 2>&1
echo "train exit $? $(date)"
run $(g $N 375) $(g $P 350) g375-vs-g350
run $(g $N 400) $(g $P 350) g400-vs-g350
run $(g $N 400) runs/9x9-scratch128-g800/checkpoints/gen_0400.azn g400-vs-g800-g400
run $(g $N 400) runs/9x9-scratch128-g400-lr2/checkpoints/gen_0450.azn g400-vs-lr2-g450
run $(g $N 400) runs/r7-eval/scratch128-g400/scratch64-g400-as-10x128.azn g400-vs-scratch64-g400
echo "$JOB all done $(date)"
