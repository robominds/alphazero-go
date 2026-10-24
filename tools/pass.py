#!/usr/bin/env python3
"""Who passes first, and how long games go on afterwards.

    python3 tools/pass.py RUN_DIR/state.bin

Over every buffered game prints: Black's win rate; how often the player
who passed first went on to win, and how many games had no pass; the mean
number of plies from the first pass to the end; and the passes made before
the final two moves, split by the eventual winner and loser. Used for
finding 2 of "Next experiments" in docs/results/README.md: the winner
passes first, and the loser plays on rather than pass into a counted loss.

A resigned game (version 5) has no plies after the resignation and no
final passes, so the pass statistics above are over the played-out games
only. Resigned games get their own lines: how many, which color resigned,
how many passed before the resignation (and how often the first passer
won), and all their passes split by the eventual winner and loser.

Reads only the replay buffer at the start of a run's state.bin, versions 2
to 6 (the versions `train` has written since the reference set was added;
3 added ownership targets, 4 reference networks, 5 resignation: one
resigned byte per buffered game, then the threshold and its window after
the buffer, which is not read; 6, a pipelined run's, is 5 with the next
generation's games after it, also not read: they are not in the buffer yet). It refuses any other magic or version,
and it never writes: it is safe to run on the state of a finished run. A
run that is still training rewrites state.bin at the end of every
generation, so read a copy of it or wait. Standard library only.

Written by Mark Castelluccio, with assistance from Claude (Anthropic).
"""
import struct,sys,collections
d=open(sys.argv[1],'rb').read();o=0
def rd(fmt):
    global o;v=struct.unpack_from('<'+fmt,d,o);o+=struct.calcsize('<'+fmt);return v
magic,ver,gen,best,st,cap,tot,ng=rd('IIiiQQQI')
if magic != 0x54535A41 or not 2 <= ver <= 6:
    sys.exit("%s: not a training state of version 2-6 (magic %#x, version %d)" % (sys.argv[1], magic, ver))
def frac(a,b): return "%.3f"%(a/b) if b else "n/a"
firstwin=0;cnt=0;loserpasses=0;winnerpasses=0;tail=[];nopass=0;bw=0
res=0;resblack=0;rescnt=0;resfirstwin=0;reswinner=0;resloser=0
for gi in range(ng):
    n,komi,m=rd('ifI');moves=rd('%di'%m);o+=m*n*n
    for p in range(m):
        k,=rd('I');o+=4*k
    r,=rd('f')
    if ver>=3:o+=n*n
    rs=0
    if ver>=5:
        rs,=rd('B')
        if rs>1: sys.exit("%s: bad resigned flag %d in game %d" % (sys.argv[1], rs, gi))
    bw+= r>0
    fp=next((i for i,a in enumerate(moves) if a==n*n),None)
    if rs:
        # The side to move after the last stored ply resigned: a loss for it.
        res+=1;resblack+= r<0
        if fp is None: continue
        rescnt+=1;resfirstwin+= (r>0)==(fp%2==0)
        for i,a in enumerate(moves):
            if a==n*n:
                if (r>0)==(i%2==0): reswinner+=1
                else: resloser+=1
        continue
    if fp is None: nopass+=1;continue
    passer_black = fp%2==0
    won = (r>0)==passer_black
    firstwin+=won;cnt+=1
    tail.append(m-fp)
    for i,a in enumerate(moves):
        if a==n*n and i<m-2:
            isb=i%2==0
            if (r>0)==isb: winnerpasses+=1
            else: loserpasses+=1
print(sys.argv[1],"games",ng,"black win %s"%frac(bw,ng))
if ver>=5: print("played-out games %d (pass statistics below); resigned games %d"%(ng-res,res))
print("first passer wins %s of %d games; games w/o pass %d"%(frac(firstwin,cnt),cnt,nopass))
print("mean plies after first pass %s"%("%.1f"%(sum(tail)/len(tail)) if tail else "n/a"))
print("non-final passes by eventual winner %d, by loser %d"%(winnerpasses,loserpasses))
if ver>=5:
    print("resigned: by Black %d, by White %d; with a pass before resigning %d, first passer wins %s"%(resblack,res-resblack,rescnt,frac(resfirstwin,rescnt)))
    print("resigned: passes by eventual winner %d, by loser %d"%(reswinner,resloser))
