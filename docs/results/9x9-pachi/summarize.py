#!/usr/bin/env python3
"""Summarize the matches against Pachi into one table.

    python3 summarize.py MATCHES_DIR > summary.txt   # also writes results.csv

MATCHES_DIR holds one directory per match, written by tools/gtp_match.py
(pachi-match.sh) with the player being rated as A and Pachi as B, and
named PLAYER-pNk: N thousand is Pachi's playouts per move; PLAYER is
gnugo (GNU Go 3.8, level 10), pachiMk (Pachi itself at M thousand
playouts), pachigcc (the GCC build of Pachi against the clang build, at
the same playouts) or NET-gGEN-sSIMS[-bBATCH] (an alphazero-go network,
generation, simulations, and the search's leaf batch when it is not the
default 8). A label ending in -nr is a control with Pachi's resignation
off.

For each match it reads results.csv, summary.txt and the SGFs, and
reports A's score with a 95% Wilson interval, the Elo difference
A - Pachi it implies (400 * log10(p / (1 - p)), the interval's ends
converted the same way), the number of distinct games, the score as
Black and as White, how the games ended, and the mean game length.

Every game that ended in two passes or at the move cap is recounted here
Tromp-Taylor (the SGF's moves replayed with captures; every stone on the
board and every empty region reaching only one color count; komi 7.5),
which is how our engine scores. That recount is the score reported for
every match: for our networks it is the scorer's own count (any game
where the two differ stops the script), for GNU Go and Pachi as A it
replaces the scorer's count, and the games where they name a different
winner are counted (scorer_tt_disagreements). Games where the two
engines' own final_score name different winners are counted too
(a_b_disagreements).

When Pachi's stderr (b.stderr.log) is there, the median and the 10th and
90th percentiles of the playouts its root held when it chose each move
(its "*** WINNER is ... (x/N games)" lines) are reported, a check on
-t =N.

It needs the full outputs (the SGFs and, for the playout counts,
Pachi's stderr); the repository keeps only each match's results.csv
and summary.txt. results.csv (in the current directory) gets one row
per match. Standard library only. Written by Mark Castelluccio, with assistance from Claude
(Anthropic).
"""
import csv
import glob
import math
import os
import re
import statistics
import sys

NETWORKS = {   # label prefix -> (description, shape)
    "gnugo": ("GNU Go 3.8 level 10", ""),
    "pachi": ("Pachi 12.90 (clang build)", ""),
    "pachigcc": ("Pachi 12.90, GCC 16 build", ""),
    "scratch": ("new preset from scratch, seed 1", "6x64"),
    "scratchs2": ("new preset from scratch, seed 2", "6x64"),
    "scratch96": ("new preset from scratch, 6x96", "6x96"),
    "scratch128": ("new preset from scratch, 10x128", "10x128"),
    "recipe": ("recipe (continued 6x96)", "6x96"),
}
LABEL = re.compile(r"^(?P<player>gnugo|pachigcc|pachi(?P<apo>\d+)k|(?P<net>[a-z0-9]+)-g(?P<gen>\d+)-s(?P<sims>\d+)(?:-b(?P<batch>\d+))?)"
                   r"-p(?P<po>\d+)k(?P<nr>-nr)?$")
SCORE = re.compile(r"^([BW])\+(\d+(?:\.\d+)?)$|^0$", re.I)
WINNER_LINE = re.compile(r"^\*\*\* WINNER is \S+ with score \S+ \(\d+/(\d+) games\)")
SIZE, KOMI = 9, 7.5


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
    m = SCORE.match((text or "").strip().split()[0] if (text or "").strip() else "")
    if not m:
        return None
    if m.group(1) is None:
        return 0.0
    return float(m.group(2)) * (1 if m.group(1).upper() == "B" else -1)


def sgf_moves(path):
    """[(color, (col, row) or None for a pass)] from one of gtp_match.py's SGFs."""
    text = open(path).read()
    out = []
    for color, point in re.findall(r";([BW])\[([a-s]{0,2})\]", text):
        out.append((color, None if point == "" else (ord(point[0]) - 97, ord(point[1]) - 97)))
    return out


def neighbors(p):
    c, r = p
    for q in ((c - 1, r), (c + 1, r), (c, r - 1), (c, r + 1)):
        if 0 <= q[0] < SIZE and 0 <= q[1] < SIZE:
            yield q


def group(board, p):
    color, seen, stack, libs = board[p], {p}, [p], set()
    while stack:
        for q in neighbors(stack.pop()):
            if board.get(q) == color and q not in seen:
                seen.add(q)
                stack.append(q)
            elif q not in board:
                libs.add(q)
    return seen, libs


def tromp_taylor(moves):
    """Black's area minus White's minus komi after replaying the moves."""
    board = {}
    for color, p in moves:
        if p is None:
            continue
        board[p] = color
        for q in neighbors(p):
            if board.get(q, color) != color:
                stones, libs = group(board, q)
                if not libs:
                    for s in stones:
                        del board[s]
        stones, libs = group(board, p)
        if not libs:   # suicide: no engine here plays one, but count it right
            for s in stones:
                del board[s]
    score = sum(1 if c == "B" else -1 for c in board.values())
    seen = set()
    for c in range(SIZE):
        for r in range(SIZE):
            p = (c, r)
            if p in board or p in seen:
                continue
            region, stack, borders = {p}, [p], set()
            while stack:
                for q in neighbors(stack.pop()):
                    if q in board:
                        borders.add(board[q])
                    elif q not in region:
                        region.add(q)
                        stack.append(q)
            seen |= region
            if borders == {"B"}:
                score += len(region)
            elif borders == {"W"}:
                score -= len(region)
    return score - KOMI


def pachi_playouts(path):
    if not os.path.exists(path):
        return None
    n = [int(m.group(1)) for m in map(WINNER_LINE.match, open(path, errors="replace")) if m]
    if not n:
        return None
    q = statistics.quantiles(n, n=10)
    return {"moves": len(n), "median": statistics.median(n), "p10": q[0], "p90": q[-1]}


def main():
    root = sys.argv[1]
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*", "results.csv"))):
        d = os.path.dirname(d)
        label = os.path.basename(d)
        m = LABEL.match(label)
        if not m:
            sys.exit("%s: not a match label" % label)
        key = m.group("net") or ("pachi" if m.group("apo") else m.group("player"))
        if m.group("net"):
            gen, sims = int(m.group("gen")), int(m.group("sims"))
            batch = min(int(m.group("batch") or 8), sims)
        elif m.group("apo"):   # Pachi against Pachi: A's playouts in the simulations column
            gen, sims, batch = "", int(m.group("apo")) * 1000, ""
        else:
            gen = sims = batch = ""
        ours = bool(m.group("net"))
        playouts = int(m.group("po")) * 1000
        games = list(csv.DictReader(open(os.path.join(d, "results.csv"))))
        n = len(games)
        if n == 0:
            continue
        score = black = 0.0
        ends = {"two passes": 0, "resigned": 0, "forfeit": 0, "max moves": 0}
        resigned_by = {"A": 0, "B": 0}
        scorer_tt = ab_dis = 0
        for g in games:
            e = g["end"]
            key_end = next(k for k in ends if e.startswith(k) or k in e)
            ends[key_end] += 1
            pts = float(g["score_a"])
            a_black = g["black"] == "A"
            if key_end in ("two passes", "max moves"):
                tt = tromp_taylor(sgf_moves(os.path.join(d, "game_%04d.sgf" % int(g["game"]))))
                tt_pts = 0.5 if tt == 0 else float((tt > 0) == a_black)
                if tt_pts != pts:
                    if ours:
                        sys.exit("%s game %s: our engine scored %s, the recount %+g" % (
                            label, g["game"], g["a_final_score"], tt))
                    scorer_tt += 1
                pts = tt_pts
                a, b = signed(g["a_final_score"]), signed(g["b_final_score"])
                if a is not None and b is not None and (a > 0) - (a < 0) != (b > 0) - (b < 0):
                    ab_dis += 1
            if key_end == "resigned":
                color = e[0]
                resigned_by["A" if (color == "B") == a_black else "B"] += 1
            score += pts
            black += pts if a_black else 0
        summary = open(os.path.join(d, "summary.txt")).read()
        distinct = int(re.search(r"distinct games (\d+)", summary).group(1))
        lo, hi = wilson(score, n)
        p = score / n
        po = pachi_playouts(os.path.join(d, "b.stderr.log"))
        rows.append({
            "match": label, "player": NETWORKS[key][0], "shape": NETWORKS[key][1],
            "pachi_resigns": "no" if m.group("nr") else "yes",
            "generation": gen, "simulations": sims, "mcts_batch": batch, "pachi_playouts": playouts,
            "games": n, "score": "%g" % score, "fraction": "%.3f" % p,
            "ci_low": "%.3f" % lo, "ci_high": "%.3f" % hi,
            "elo": fmt(elo(p)), "elo_low": fmt(elo(lo)), "elo_high": fmt(elo(hi)),
            "distinct": distinct, "score_as_black": "%g" % black, "score_as_white": "%g" % (score - black),
            "two_passes": ends["two passes"], "pachi_resigned": resigned_by["B"],
            "a_resigned": resigned_by["A"], "forfeits": ends["forfeit"], "max_moves": ends["max moves"],
            "scorer_tt_disagreements": scorer_tt, "a_b_disagreements": ab_dis,
            "mean_moves": "%.1f" % (sum(int(g["moves"]) for g in games) / n),
            "pachi_moves_logged": po["moves"] if po else "",
            "pachi_root_playouts_median": "%.0f" % po["median"] if po else "",
            "pachi_root_playouts_p10": "%.0f" % po["p10"] if po else "",
            "pachi_root_playouts_p90": "%.0f" % po["p90"] if po else "",
        })
    order = list(NETWORKS)
    rows.sort(key=lambda r: (order.index(next(k for k, v in NETWORKS.items() if v[0] == r["player"])),
                             r["generation"] or 0, -(r["simulations"] or 0), r["mcts_batch"] or 0,
                             r["pachi_playouts"], r["pachi_resigns"]))
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("| player | gen | sims | batch | Pachi playouts | Pachi resigns | score vs Pachi | 95% interval "
          "| Elo vs Pachi (interval) | distinct | as B / as W | Pachi resigned | A resigned | forfeits "
          "| max moves | scorer vs TT | A vs B count | Pachi root playouts p10 / median / p90 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %s %s | %s | %s | %s | %d | %s | %s/%d = %s | %s-%s | %s (%s to %s) | %d | %s / %s | %d | %d | %d "
              "| %d | %d | %d | %s / %s / %s |" % (
                  r["player"], r["shape"], r["generation"], r["simulations"], r["mcts_batch"],
                  r["pachi_playouts"], r["pachi_resigns"], r["score"], r["games"], r["fraction"],
                  r["ci_low"], r["ci_high"],
                  r["elo"], r["elo_low"], r["elo_high"], r["distinct"], r["score_as_black"],
                  r["score_as_white"], r["pachi_resigned"], r["a_resigned"], r["forfeits"], r["max_moves"],
                  r["scorer_tt_disagreements"], r["a_b_disagreements"], r["pachi_root_playouts_p10"],
                  r["pachi_root_playouts_median"], r["pachi_root_playouts_p90"]))


if __name__ == "__main__":
    main()
