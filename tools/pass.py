#!/usr/bin/env python3
"""Who passes first, and how long games go on afterwards.

    python3 tools/pass.py RUN_DIR/state.bin

Over every buffered game prints: Black's win rate; how often the player
who passed first went on to win, and how many games had no pass; the mean
number of plies from the first pass to the end; and the passes made before
the final two moves, split by the eventual winner and loser. Used for
finding 2 of "Next experiments" in docs/results/README.md: the winner
passes first, and the loser plays on rather than pass into a counted loss.

Reads only the replay buffer at the start of a run's state.bin, versions 2
to 4 (the versions `train` has written since the reference set was added;
3 added ownership targets, 4 reference networks). It refuses any other
magic or version, and it never writes: it is safe to run on the state of a
finished run. A run that is still training rewrites state.bin at the end of
every generation, so read a copy of it or wait. Standard library only.

Written by Mark Castelluccio, with assistance from Claude (Anthropic).
"""
import struct,sys,collections
d=open(sys.argv[1],'rb').read();o=0
def rd(fmt):
    global o;v=struct.unpack_from('<'+fmt,d,o);o+=struct.calcsize('<'+fmt);return v
magic,ver,gen,best,st,cap,tot,ng=rd('IIiiQQQI')
if magic != 0x54535A41 or not 2 <= ver <= 4:
    sys.exit("%s: not a training state of version 2-4 (magic %#x, version %d)" % (sys.argv[1], magic, ver))
firstwin=0;cnt=0;loserpasses=0;winnerpasses=0;tail=[];nopass=0;bw=0
for gi in range(ng):
    n,komi,m=rd('ifI');moves=rd('%di'%m);o+=m*n*n
    for p in range(m):
        k,=rd('I');o+=4*k
    r,=rd('f')
    if ver>=3:o+=n*n
    bw+= r>0
    fp=next((i for i,a in enumerate(moves) if a==n*n),None)
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
print(sys.argv[1],"games",ng,"black win %.3f"%(bw/ng))
print("first passer wins %.3f of %d games; games w/o pass %d"%(firstwin/cnt,cnt,nopass))
print("mean plies after first pass %.1f"%(sum(tail)/len(tail)))
print("non-final passes by eventual winner %d, by loser %d"%(winnerpasses,loserpasses))
