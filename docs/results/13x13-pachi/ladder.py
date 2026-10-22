#!/usr/bin/env python3
"""Put the 13x13 networks, GNU Go 3.8 and the Pachi settings on one scale.

    python3 ladder.py results.csv ../13x13-gnugo/results.csv > ladder.txt
    python3 ladder.py --without pattern results.csv ../13x13-gnugo/results.csv > ladder-nopattern.txt

The first file is summarize.py's (every match with Pachi as B: our
networks, GNU Go at levels 10 and 0 and Pachi's own settings as A), the
second the GNU Go write-up's (our networks against GNU Go). The scale is
Elo with GNU Go 3.8 level 10 at 0. --without TOKEN (repeatable; a
label's player token, such as pattern) leaves out every match of that
player. Prints:

1. The anchor: the matches among GNU Go and the Pachi settings, with
   each one's Elo difference, and a Bradley-Terry (Elo) fit of those
   players to those matches only, with no network's games in it.
2. A joint fit of every player to every game, GNU Go level 10 fixed at
   0, and each network's rating against each opponent it played (the
   opponent's joint rating plus the network's Elo against it).
3. The joint fit's residuals per pairing (observed against expected
   score) and Pearson's chi-square over all pairings, with pairings minus
   free ratings as its degrees of freedom.

A maximum-likelihood rating exists only when every player both scored
and conceded against the players it is linked to through results that
are not all wins: formally, when the graph with an edge from A to B
whenever A scored at least half a point against B is strongly connected.
Where it is not, each fit is made within the strongly connected group
holding GNU Go level 10, and every other player is reported with the
matches that bound it (a player that won or lost every game against the
group has no finite rating, only a one-sided bound).

The fits are maximum likelihood (Newton's method), with standard errors
from the inverse of the information matrix. Standard library only.
Written by Mark Castelluccio, with assistance from Claude (Anthropic).
"""
import csv
import math
import re
import sys

C = math.log(10) / 400   # Elo to natural log-odds
GNUGO = "GNU Go L10"


def player(token):
    m = re.match(r"^agz-g(\d+)-s(\d+)$", token)
    if m:
        return "gen %s" % m.group(1)
    m = re.match(r"^(uct|uctnp|uctlight)(\d+)$", token)
    if m:
        return {"uct": "Pachi", "uctnp": "Pachi nopat", "uctlight": "Pachi light"}[m.group(1)] + " =" + m.group(2)
    return {"gnugo10": GNUGO, "gnugo0": "GNU Go L0", "replay": "Pachi replay", "pattern": "Pachi pattern"}[token]


def is_net(p):
    return p.startswith("gen ")


def elo(p):
    return 400 * math.log10(p / (1 - p))


def se(p, n):
    return math.sqrt(1 / (n * p * (1 - p))) / C


def expected(d):
    return 1 / (1 + 10 ** (-d / 400))


def fmt(x):
    return "%+.0f" % x


def solve(a, b):
    """x with a x = b, by Gauss-Jordan elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for k in range(n):
        piv = max(range(k, n), key=lambda i: abs(m[i][k]))
        m[k], m[piv] = m[piv], m[k]
        for i in range(n):
            if i != k:
                f = m[i][k] / m[k][k]
                for j in range(k, n + 1):
                    m[i][j] -= f * m[k][j]
    return [m[i][n] / m[i][i] for i in range(n)]


def inverse(a):
    n = len(a)
    cols = [solve(a, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def group_of(pairings, root):
    """The players in root's strongly connected group of the 'scored against' graph."""
    players = {x for q in pairings for x in q[:2]}
    edges = {p: set() for p in players}
    for a, b, s, n in pairings:
        if s > 0:
            edges[a].add(b)
        if s < n:
            edges[b].add(a)

    def reach(start, e):
        seen, stack = {start}, [start]
        while stack:
            for y in e[stack.pop()]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return seen
    back = {p: set() for p in players}
    for x, ys in edges.items():
        for y in ys:
            back[y].add(x)
    return reach(root, edges) & reach(root, back)


def fit(pairings, players):
    """Maximum-likelihood ratings (GNU Go L10 = 0), standard errors and covariance."""
    free = [p for p in players if p != GNUGO]
    idx = {p: i for i, p in enumerate(free)}
    r = {p: 0.0 for p in players}
    for _ in range(500):
        g = [0.0] * len(free)
        h = [[0.0] * len(free) for _ in free]
        for a, b, s, n in pairings:
            e = expected(r[a] - r[b])
            w = n * e * (1 - e) * C * C
            for x, sign in ((a, 1), (b, -1)):
                if x in idx:
                    g[idx[x]] += sign * (s - n * e) * C
            for x in (a, b):
                for y in (a, b):
                    if x in idx and y in idx:
                        h[idx[x]][idx[y]] += w if x == y else -w
        step = solve(h, g)
        biggest = max(abs(v) for v in step)
        scale = min(1.0, 200 / biggest) if biggest else 1.0
        for p, i in idx.items():
            r[p] += scale * step[i]
        if biggest < 1e-6:
            break
    else:
        sys.exit("the fit did not converge")
    cov = inverse(h)
    return r, {p: math.sqrt(cov[i][i]) for p, i in idx.items()}


def rated(pairings, title):
    """Fit within GNU Go's strongly connected group; print the fit and the players left out."""
    players = sorted({x for q in pairings for x in q[:2]})
    group = group_of(pairings, GNUGO)
    inside = [q for q in pairings if q[0] in group and q[1] in group]
    r, rse = fit(inside, [p for p in players if p in group])
    print(title + "\n")
    print("| player | Elo | 95% interval |")
    print("|---|---|---|")
    for p in sorted(group, key=lambda x: r[x]):
        if p == GNUGO:
            print("| %s | 0 | (fixed) |" % p)
        else:
            print("| %s | %s | %s to %s |" % (p, fmt(r[p]), fmt(r[p] - 1.96 * rse[p]), fmt(r[p] + 1.96 * rse[p])))
    out = [p for p in players if p not in group]
    if out:
        print("\nNot rated (outside GNU Go level 10's strongly connected group), with their matches:\n")
        for p in out:
            parts = []
            for a, b, s, n in pairings:
                if p in (a, b):
                    mine, other = (s, b) if a == p else (n - s, a)
                    parts.append("%g/%d vs %s%s" % (mine, n, other,
                                                    " (%s)" % fmt(r[other]) if other in r else ""))
            print("- %s: %s" % (p, "; ".join(parts)))
    return r, rse, inside


def main():
    args, without = sys.argv[1:], set()
    while args and args[0] == "--without":
        without.add(args[1])
        args = args[2:]
    pachi_rows = [r for r in csv.DictReader(open(args[0])) if r["a"] not in without and r["b"] not in without]
    gnugo_rows = list(csv.DictReader(open(args[1])))
    if without:
        print("Without the matches of: %s.\n" % ", ".join(sorted(without)))
    pairings = []   # (A, B, A's score, games)
    for row in pachi_rows:
        pairings.append((player(row["a"]), player(row["b"]), float(row["score"]), int(row["games"])))
    for row in gnugo_rows:
        m = re.match(r"^agz-g(\d+)-s(\d+)(?:-l(\d+))?$", row["match"])
        level = m.group(3) or "10"
        pairings.append(("gen %s" % m.group(1), GNUGO if level == "10" else "GNU Go L%s" % level,
                         float(row["score"]), int(row["games"])))
    nets = sorted({x for q in pairings for x in q[:2] if is_net(x)}, key=lambda x: int(x[4:]))

    anchor = [q for q in pairings if not is_net(q[0]) and not is_net(q[1])]
    print("1. The anchor: GNU Go 3.8 and the Pachi settings\n")
    print("| A | B | A's score | Elo A - B (95% interval) |")
    print("|---|---|---|---|")
    for a, b, s, n in anchor:
        p = s / n
        if 0 < p < 1:
            print("| %s | %s | %g/%d | %s (%s to %s) |" % (a, b, s, n, fmt(elo(p)), fmt(elo(p) - 1.96 * se(p, n)),
                                                         fmt(elo(p) + 1.96 * se(p, n))))
        else:
            print("| %s | %s | %g/%d | %s |" % (a, b, s, n, "infinite"))
    print()
    rated(anchor, "Fitted to these matches only:")

    print()
    r, rse, inside = rated(pairings, "2. Joint fit (every game above and below; GNU Go 3.8 level 10 = 0)")
    print("\nEach network against each opponent (opponent's joint rating plus the network's Elo against it;"
          " ± is the match's own 95% half-width):\n")
    print("| network | opponent | score | Elo vs opponent | opponent's rating | network via opponent |")
    print("|---|---|---|---|---|---|")
    for net in nets:
        for a, b, s, n in pairings:
            if a != net:
                continue
            p = s / n
            opp = ("%s" % fmt(r[b])) if b in r else "not rated"
            if 0 < p < 1:
                via = (fmt(r[b] + elo(p)) + " ± %.0f" % (1.96 * se(p, n))) if b in r else ""
                print("| %s | %s | %g/%d | %s | %s | %s |" % (net, b, s, n, fmt(elo(p)), opp, via))
            else:
                bound = ""
                if b in r:
                    # the one-sided bound from the score's 95% Wilson interval
                    z = 1.96
                    lo = (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
                    hi = (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
                    bound = ("below %s" % fmt(r[b] + elo(hi))) if p == 0 else ("above %s" % fmt(r[b] + elo(lo)))
                print("| %s | %s | %g/%d | %s | %s | %s |" % (net, b, s, n, "-inf" if p == 0 else "+inf", opp,
                                                            bound))

    print("\n3. Residuals of the joint fit (rated players only)\n")
    print("| A | B | games | A's score | expected | Pearson residual |")
    print("|---|---|---|---|---|---|")
    chi2 = 0.0
    for a, b, s, n in inside:
        e = expected(r[a] - r[b])
        z = (s - n * e) / math.sqrt(n * e * (1 - e))
        chi2 += z * z
        print("| %s | %s | %d | %g | %.1f | %+.2f |" % (a, b, n, s, n * e, z))
    free = len({x for q in inside for x in q[:2]}) - 1
    print("\nPearson chi-square %.1f on %d degrees of freedom (%d pairings, %d free ratings)" % (
        chi2, len(inside) - free, len(inside), free))


if __name__ == "__main__":
    main()
