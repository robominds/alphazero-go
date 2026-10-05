#!/usr/bin/env python3
"""Search-target statistics by side to move and eventual result.

    python3 tools/side.py RUN_DIR/state.bin

For every buffered position from ply 20 up to the game's first pass (the
middle game), groups the stored MCTS visit counts by whether the side to
move went on to win or lose the game and by its color, and prints for each
group: the number of positions, the mean entropy of the visit
distribution (the policy target) in nats, the mean number of moves with at
least one visit, and the mean share of the most-visited move. Used for
finding 1 of "Next experiments" in docs/results/README.md: the losing
side's search spreads over many more moves, because unvisited moves keep
Q = 0, which for a side that is behind beats every move it has tried.
A resigned game (version 5) has no plies after the resignation, so its
middle game ends there if it had no pass before; the resigning side is the
loser. For version 5 the number of resigned games is printed first.

Reads only the replay buffer at the start of a run's state.bin, versions 2
to 5 (the versions `train` has written since the reference set was added;
3 added ownership targets, 4 reference networks, 5 resignation: one
resigned byte per buffered game, then the threshold and its window after
the buffer, which is not read). It refuses any other magic or version,
and it never writes: it is safe to run on the state of a finished run. A
run that is still training rewrites state.bin at the end of every
generation, so read a copy of it or wait. Standard library only.

Written by Mark Castelluccio, with assistance from Claude (Anthropic).
"""
import struct,sys,math,collections
d=open(sys.argv[1],'rb').read();o=0
def rd(fmt):
    global o;v=struct.unpack_from('<'+fmt,d,o);o+=struct.calcsize('<'+fmt);return v
magic,ver,gen,best,st,cap,tot,ng=rd('IIiiQQQI')
if magic != 0x54535A41 or not 2 <= ver <= 5:
    sys.exit("%s: not a training state of version 2-5 (magic %#x, version %d)" % (sys.argv[1], magic, ver))
acc=collections.defaultdict(lambda:[0,0.0,0.0,0.0]);resigned=0
for gi in range(ng):
    n,komi,m=rd('ifI');moves=rd('%di'%m);o+=m*n*n
    vis=[]
    for p in range(m):
        k,=rd('I');vs=rd('%dI'%k);vis.append([v&0xffff for v in vs])
    r,=rd('f')
    if ver>=3:o+=n*n
    if ver>=5:
        rs,=rd('B')
        if rs>1: sys.exit("%s: bad resigned flag %d in game %d" % (sys.argv[1], rs, gi))
        resigned+=rs
    fp=next((i for i,a in enumerate(moves) if a==n*n),m)
    for p in range(20,fp):
        black=p%2==0;win=(r>0)==black
        c=vis[p];T=sum(c);h=-sum(x/T*math.log(x/T) for x in c)
        a=acc[("winner" if win else "loser")+("/B" if black else "/W")]
        a[0]+=1;a[1]+=h;a[2]+=len(c);a[3]+=max(c)/T
if ver>=5: print("resigned games %d of %d"%(resigned,ng))
for k in sorted(acc):
    a=acc[k];print("%-10s n=%6d entropy %.3f  moves-visited %.1f  top-share %.3f"%(k,a[0],a[1]/a[0],a[2]/a[0],a[3]/a[0]))
