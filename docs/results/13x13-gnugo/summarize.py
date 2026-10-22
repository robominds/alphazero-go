#!/usr/bin/env python3
"""Summarize the 13x13 GNU Go matches into one table.

    python3 summarize.py MATCHES_DIR > summary.txt   # also writes results.csv

MATCHES_DIR holds one directory per chunk of a match, named
LABEL-cNN (LABEL is agz-gGEN-sSIMS[-bBATCH]: the 13x13-agz network of
that generation, the simulations, and the search's leaf batch when it
is not the default 8), each written by tools/gtp_match.py
(gnugo-match.sh) with our engine as A and GNU Go as B. The chunks of a
label are pooled. For each match it reports A's score with a 95% Wilson
interval, the Elo difference A - GNU Go it implies (400 * log10(p /
(1 - p)), the interval's ends converted the same way, as gtp_match.py
does; infinite at a score of 0 or 1), the number of distinct games
(identical move sequences counted once, over all the chunks' SGFs), the
score as Black and as White, how the games ended, the games where GNU
Go's own final_score named a different winner and those where it gave a
different margin, the mean game length, and how many games our engine
passed in while GNU Go's count gave GNU Go the win by more than half the
board (a sign of a group given up). results.csv (in the current
directory) gets one row per match.

Standard library only. Written by Mark Castelluccio, with assistance from
Claude (Anthropic).
"""
import csv
import glob
import math
import os
import re
import sys

NETWORK = ("13x13-agz (gated)", "6x96")
SCORE = re.compile(r"^([BW])\+(\d+(?:\.\d+)?)$|^0$", re.I)
SIZE = 13


def wilson(score, games, z=1.96):
    p = score / games
    denom = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def elo(p):
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    return 400 * math.log10(p / (1 - p))


def fmt(x):
    return ("+inf" if x > 0 else "-inf") if math.isinf(x) else "%+.0f" % x


def signed(text):
    m = SCORE.match((text or "").strip())
    if not m:
        return None
    if m.group(1) is None:
        return 0.0
    return float(m.group(2)) * (1 if m.group(1).upper() == "B" else -1)


def moves_of(path):
    text = open(path).read()
    return tuple(re.findall(r";[BW]\[([a-s]{0,2})\]", text))


def backends(dirs):
    """How many chunks each backend of our engine played, from backend.txt (absent: blas)."""
    count = {}
    for d in dirs:
        path = os.path.join(d, "backend.txt")
        b = open(path).read().strip() if os.path.exists(path) else "blas"
        count[b] = count.get(b, 0) + 1
    return ", ".join("%s %d" % kv for kv in sorted(count.items()))


def main():
    root = sys.argv[1]
    chunks = {}
    for d in sorted(glob.glob(os.path.join(root, "*", "results.csv"))):
        d = os.path.dirname(d)
        m = re.match(r"^(agz-g(\d+)-s(\d+)(?:-b(\d+))?(?:-l\d+)?)-c(\d+)$", os.path.basename(d))
        if not m:
            sys.exit("%s: not a match label" % d)
        chunks.setdefault(m.group(1), []).append(d)
    rows = []
    for label, dirs in chunks.items():
        m = re.match(r"^agz-g(\d+)-s(\d+)(?:-b(\d+))?(?:-l(\d+))?$", label)
        gen, sims = int(m.group(1)), int(m.group(2))
        batch = min(int(m.group(3) or 8), sims)
        level = int(m.group(4) or 10)
        games, sequences = [], set()
        for d in dirs:
            rs = list(csv.DictReader(open(os.path.join(d, "results.csv"))))
            summary = open(os.path.join(d, "summary.txt")).read()
            if not re.search(r"distinct games \d+", summary):
                sys.exit("%s: unfinished (no summary)" % d)
            for g in rs:
                g["_dir"] = d
                sequences.add(moves_of(os.path.join(d, "game_%04d.sgf" % int(g["game"]))))
            games += rs
        n = len(games)
        score = sum(float(g["score_a"]) for g in games)
        black = sum(float(g["score_a"]) for g in games if g["black"] == "A")
        ends = {"two passes": 0, "resigned": 0, "forfeit": 0, "max moves": 0}
        winner_dis = margin_dis = wipeouts = 0
        for g in games:
            e = g["end"]
            key = next(k for k in ends if e.startswith(k) or k in e)
            ends[key] += 1
            a, b = signed(g["a_final_score"]), signed(g["b_final_score"])
            if a is not None and b is not None:
                if (a > 0) - (a < 0) != (b > 0) - (b < 0):
                    winner_dis += 1
                if a != b:
                    margin_dis += 1
            if a is not None and float(g["score_a"]) == 0 and abs(a) > SIZE * SIZE / 2:
                wipeouts += 1
        resigned_by = {"A": 0, "B": 0}
        for g in games:
            if "resigned" in g["end"]:
                color = g["end"][0]
                resigned_by["A" if g["black" if color == "B" else "white"] == "A" else "B"] += 1
        lo, hi = wilson(score, n)
        p = score / n
        rows.append({
            "match": label, "network": NETWORK[0], "shape": NETWORK[1], "chunks": len(dirs),
            "backend_chunks": backends(dirs), "gnugo_level": level,
            "generation": gen, "simulations": sims, "mcts_batch": batch, "games": n, "score": "%g" % score,
            "fraction": "%.3f" % p, "ci_low": "%.3f" % lo, "ci_high": "%.3f" % hi,
            "elo": fmt(elo(p)), "elo_low": fmt(elo(lo)), "elo_high": fmt(elo(hi)),
            "distinct": len(sequences), "score_as_black": "%g" % black, "score_as_white": "%g" % (score - black),
            "two_passes": ends["two passes"], "gnugo_resigned": resigned_by["B"],
            "ours_resigned": resigned_by["A"], "forfeits": ends["forfeit"],
            "max_moves": ends["max moves"], "winner_disagreements": winner_dis,
            "margin_disagreements": margin_dis, "losses_by_over_half_board": wipeouts,
            "mean_moves": "%.1f" % (sum(int(g["moves"]) for g in games) / n),
        })
    rows.sort(key=lambda r: (-r["gnugo_level"], r["generation"], -r["simulations"], r["mcts_batch"]))
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("| network | gen | sims | batch | GNU Go level | score vs GNU Go | 95% interval | Elo vs GNU Go (interval) "
          "| distinct | as B / as W | ends: passes / resign / cap / forfeit | winner disagreements "
          "| losses by > half the board | mean moves | backend (chunks) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %s %s | %d | %d | %d | %d | %s/%d = %s | %s-%s | %s (%s to %s) | %d | %s / %s | %d / %d / %d / %d "
              "| %d | %d | %s | %s |" % (
                  r["network"], r["shape"], r["generation"], r["simulations"], r["mcts_batch"], r["gnugo_level"],
                  r["score"],
                  r["games"], r["fraction"], r["ci_low"], r["ci_high"], r["elo"], r["elo_low"], r["elo_high"],
                  r["distinct"], r["score_as_black"], r["score_as_white"], r["two_passes"],
                  r["gnugo_resigned"] + r["ours_resigned"], r["max_moves"], r["forfeits"],
                  r["winner_disagreements"], r["losses_by_over_half_board"], r["mean_moves"], r["backend_chunks"]))


if __name__ == "__main__":
    main()
