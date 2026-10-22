#!/usr/bin/env python3
"""How often each player puts a stone on the first or second line early on.

    python3 edges.py MOVES MATCHES_DIR... > edges.txt

For every match chunk directory under the MATCHES_DIRs (written by
tools/gtp_match.py, as for summarize.py; chunks of one match are pooled
by stripping the -cNN suffix), it reads the SGFs and results.csv and
counts, over each player's first MOVES moves of every game (passes
excluded), the share played on the first line (the edge) and on the
second line, for A and B separately. A 13x13 board has 48 first-line
and 40 second-line points out of 169 (28% and 24%).

Standard library only. Written by Mark Castelluccio, with assistance
from Claude (Anthropic).
"""
import csv
import glob
import os
import re
import sys

SIZE = 13


def line(p):
    c, r = ord(p[0]) - 97, ord(p[1]) - 97
    return min(c, r, SIZE - 1 - c, SIZE - 1 - r) + 1


def main():
    limit = int(sys.argv[1])
    counts = {}
    for root in sys.argv[2:]:
        for d in sorted(glob.glob(os.path.join(root, "*", "results.csv"))):
            d = os.path.dirname(d)
            match = re.sub(r"-c\d+$", "", os.path.basename(d))
            c = counts.setdefault(match, {"A": [0, 0, 0], "B": [0, 0, 0]})
            for g in csv.DictReader(open(os.path.join(d, "results.csv"))):
                text = open(os.path.join(d, "game_%04d.sgf" % int(g["game"]))).read()
                seen = {"B": 0, "W": 0}
                for color, p in re.findall(r";([BW])\[([a-s]{0,2})\]", text):
                    if not p or seen[color] >= limit:
                        continue
                    seen[color] += 1
                    who = "A" if g["black" if color == "B" else "white"] == "A" else "B"
                    k = line(p)
                    c[who][0] += 1
                    c[who][1] += k == 1
                    c[who][2] += k == 2
    print("Each player's first %d moves of every game (passes excluded).\n" % limit)
    print("| match | A: moves | A: line 1 | A: line 2 | B: moves | B: line 1 | B: line 2 |")
    print("|---|---|---|---|---|---|---|")
    for match, c in counts.items():
        a, b = c["A"], c["B"]
        print("| %s | %d | %.1f%% | %.1f%% | %d | %.1f%% | %.1f%% |" % (
            match, a[0], 100.0 * a[1] / max(a[0], 1), 100.0 * a[2] / max(a[0], 1),
            b[0], 100.0 * b[1] / max(b[0], 1), 100.0 * b[2] / max(b[0], 1)))


if __name__ == "__main__":
    main()
