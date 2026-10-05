#!/usr/bin/env python3
"""Whole-buffer statistics of the policy targets and of passing.

    python3 tools/buf.py RUN_DIR/state.bin

Prints the state version, generation and number of games, then over every
buffered position: the mean entropy of the visit distribution (the policy
target) in nats, the mean and median root visit totals, the mean number of
moves with visits, the mean share of the most-visited move, the fraction
of positions at or after the game's first pass, the fraction whose
most-visited move is a pass, and the count, fraction and mean target
entropy of three phases (plies 0-19, ply 20 up to the first pass, and from
the first pass on). For version 5 it also prints the number of resigned
games and their positions: a resigned game has no plies after the
resignation, so it adds to the middle game but rarely to the after-pass
phase, and the share of positions at or after the first pass is printed
again over the played-out games alone. Used for finding 2 of "Next experiments" in
docs/results/README.md.

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
path=sys.argv[1]
d=open(path,'rb').read();o=0
def u32():
    global o;v=struct.unpack_from('<I',d,o)[0];o+=4;return v
def i32():
    global o;v=struct.unpack_from('<i',d,o)[0];o+=4;return v
def u64():
    global o;v=struct.unpack_from('<Q',d,o)[0];o+=8;return v
def f32():
    global o;v=struct.unpack_from('<f',d,o)[0];o+=4;return v
magic=u32();ver=u32();gen=i32();best=i32();st=u64()
if magic != 0x54535A41 or not 2 <= ver <= 5:
    sys.exit("%s: not a training state of version 2-5 (magic %#x, version %d)" % (sys.argv[1], magic, ver))
cap=u64();tot=u64();ng=u32()
print("version",ver,"gen",gen,"games",ng)
H=[];post=[];npos=0;resigned=0;rpos=0;played=[0,0];topk=[];nvis=[];passtop=0;totv=[];maxfrac=[]
phase=collections.defaultdict(list)
for gi in range(ng):
    n=i32();komi=f32();m=u32();moves=struct.unpack_from('<%di'%m,d,o);o+=4*m
    o+=m*n*n
    fp=next((i for i,a in enumerate(moves) if a==n*n),m)
    for p in range(m):
        k=u32();vs=struct.unpack_from('<%dI'%k,d,o);o+=4*k
        cnt=[v&0xffff for v in vs];acts=[v>>16 for v in vs];T=sum(cnt)
        h=-sum(c/T*math.log(c/T) for c in cnt)
        H.append(h);nvis.append(k);totv.append(T);maxfrac.append(max(cnt)/T)
        if acts[cnt.index(max(cnt))]==n*n: passtop+=1
        post.append(p>=fp)
        ph='open<20' if p<20 else ('mid' if p<fp else 'after1stpass')
        phase[ph].append(h)
    r=f32()
    if ver>=3: o+=n*n
    rs=0
    if ver>=5:
        rs=d[o];o+=1
        if rs>1: sys.exit("%s: bad resigned flag %d in game %d" % (path, rs, gi))
    if rs: resigned+=1;rpos+=m
    else: played[0]+=m;played[1]+=m-fp
N=len(H)
print("positions",N)
print("mean pi entropy %.3f nats"%(sum(H)/N))
print("mean root visits %.1f, median %d"%(sum(totv)/N,sorted(totv)[N//2]))
print("mean #moves with visits %.1f"%(sum(nvis)/N))
print("mean top-move share %.3f"%(sum(maxfrac)/N))
print("frac positions at/after first pass %.3f"%(sum(post)/N))
print("frac positions where pass is top visit %.3f"%(passtop/N))
if ver>=5:
    print("resigned games %d of %d, %d positions (%.3f)"%(resigned,ng,rpos,rpos/N))
    if played[0]: print("played-out games only: frac positions at/after first pass %.3f"%(played[1]/played[0]))
for k,v in phase.items(): print(" phase",k,"n=%d frac=%.2f H=%.3f"%(len(v),len(v)/N,sum(v)/len(v)))
