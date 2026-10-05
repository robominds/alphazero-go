#!/bin/bash
# Screen random symmetry in search: gen 250 against itself, symmetry on side A only, 400 games.
cd .
B=runs/bin-sym; E=runs/sym-screen; mkdir -p $E
G=runs/9x9-main/checkpoints/gen_0250.azn
echo "start $(date)"
$B/arena --a $G --b $G --a-random-symmetry --games 400 --simulations 200 --komi 7.5 --backend blas > $E/g250-symA-vs-g250.txt 2>&1
echo "symA: $(head -1 $E/g250-symA-vs-g250.txt)"
$B/arena --a $G --b $G --a-random-symmetry --games 400 --simulations 800 --komi 7.5 --backend blas > $E/g250-symA-vs-g250-s800.txt 2>&1
echo "symA s800: $(head -1 $E/g250-symA-vs-g250-s800.txt)"
echo "all done $(date)"
