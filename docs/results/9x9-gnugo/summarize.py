#!/usr/bin/env python3
"""Summarize the GNU Go matches into one table.

    python3 summarize.py MATCHES_DIR > summary.txt   # also writes results.csv

MATCHES_DIR holds one directory per match, named LABEL (for example
recipe-g250-s200: network, generation, simulations, and a -b1 suffix when
the search's leaf batch was 1 instead of the default 8), each written by
tools/gtp_match.py with our engine as A and GNU Go as B. For each match
it reads results.csv and the SGFs and reports A's score with a 95%
Wilson interval, the Elo difference A - GNU Go it implies (the
interval's ends converted the same way, as gtp_match.py does), the
number of distinct games (identical move sequences counted once; read from
summary.txt, gtp_match.py's output, and checked against the SGFs when all
of them are there), the
score as Black and as White, how the games ended, the games where GNU
Go's own final_score named a different winner and those where it gave
a different margin, and the mean game length. results.csv (in the
current directory) gets one row per match.

Standard library only. Written by Mark Castelluccio, with assistance from
Claude (Anthropic).
"""
import csv
import glob
import math
import os
import re
import sys

NETWORKS = {   # label prefix -> (description, shape)
    "main": ("old preset (9x9-main)", "6x64"),
    "scratch": ("new preset from scratch, seed 1", "6x64"),
    "scratchs2": ("new preset from scratch, seed 2", "6x64"),
    "scratch96": ("new preset from scratch, 6x96", "6x96"),
    "recipe": ("recipe (continued 6x96)", "6x96"),
}
SCORE = re.compile(r"^([BW])\+(\d+(?:\.\d+)?)$|^0$", re.I)


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


def main():
    root = sys.argv[1]
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*", "results.csv"))):
        d = os.path.dirname(d)
        label = os.path.basename(d)
        m = re.match(r"^([a-z0-9]+)-g(\d+)-s(\d+)(?:-b(\d+))?$", label)
        net, gen, sims = m.group(1), int(m.group(2)), int(m.group(3))
        # A search never has more leaves in flight than simulations left,
        # so a 1-simulation search has a batch of 1 whatever its setting.
        batch = min(int(m.group(4) or 8), sims)
        games = list(csv.DictReader(open(os.path.join(d, "results.csv"))))
        n = len(games)
        if n == 0:
            continue
        score = sum(float(g["score_a"]) for g in games)
        black = sum(float(g["score_a"]) for g in games if g["black"] == "A")
        white = score - black
        ends = {"two passes": 0, "resigned": 0, "forfeit": 0, "max moves": 0}
        winner_dis = margin_dis = 0
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
        resigned_by = {"A": 0, "B": 0}
        for g in games:
            if "resigned" in g["end"]:
                color = g["end"][0]
                resigned_by["A" if g["black" if color == "B" else "white"] == "A" else "B"] += 1
        # gtp_match.py's own count, from summary.txt; the SGFs, where they
        # are kept, must agree with it.
        summary = open(os.path.join(d, "summary.txt")).read()
        distinct = int(re.search(r"distinct games (\d+)", summary).group(1))
        sgfs = glob.glob(os.path.join(d, "game_*.sgf"))
        if len(sgfs) == n and len({moves_of(p) for p in sgfs}) != distinct:
            sys.exit("%s: the SGFs give a different distinct-game count" % label)
        lo, hi = wilson(score, n)
        p = score / n
        rows.append({
            "match": label, "network": NETWORKS[net][0], "shape": NETWORKS[net][1],
            "generation": gen, "simulations": sims, "mcts_batch": batch, "games": n, "score": "%g" % score,
            "fraction": "%.3f" % p, "ci_low": "%.3f" % lo, "ci_high": "%.3f" % hi,
            "elo": fmt(elo(p)), "elo_low": fmt(elo(lo)), "elo_high": fmt(elo(hi)),
            "distinct": distinct, "score_as_black": "%g" % black, "score_as_white": "%g" % white,
            "two_passes": ends["two passes"], "gnugo_resigned": resigned_by["B"],
            "ours_resigned": resigned_by["A"], "forfeits": ends["forfeit"],
            "max_moves": ends["max moves"], "winner_disagreements": winner_dis,
            "margin_disagreements": margin_dis,
            "mean_moves": "%.1f" % (sum(int(g["moves"]) for g in games) / n),
        })
    order = list(NETWORKS)
    rows.sort(key=lambda r: (order.index(next(k for k, v in NETWORKS.items() if v[0] == r["network"])),
                             r["generation"], -r["simulations"], r["mcts_batch"]))
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("| network | gen | sims | batch | score vs GNU Go | 95% interval | Elo vs GNU Go (interval) "
          "| distinct | as B / as W | GNU Go resigned | forfeits | winner disagreements |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %s %s | %d | %d | %d | %s/%d = %s | %s-%s | %s (%s to %s) | %d | %s / %s | %d | %d | %d |" % (
            r["network"], r["shape"], r["generation"], r["simulations"], r["mcts_batch"], r["score"], r["games"],
            r["fraction"], r["ci_low"], r["ci_high"], r["elo"], r["elo_low"], r["elo_high"],
            r["distinct"], r["score_as_black"], r["score_as_white"], r["gnugo_resigned"],
            r["forfeits"], r["winner_disagreements"]))


if __name__ == "__main__":
    main()
