#!/usr/bin/env python3
"""Put GNU Go 3.8 level 10, the Pachi levels and our networks on one scale.

    python3 ladder.py results.csv ../9x9-gnugo/results.csv > ladder.txt
    python3 ladder.py --resign results.csv ../9x9-gnugo/results.csv > ladder-resign.txt

The first file is summarize.py's (matches against Pachi: GNU Go, Pachi
at other playouts, our networks), the second the GNU Go write-up's (our
networks against GNU Go). The scale is Elo with GNU Go 3.8 level 10 at 0.
Only the networks that played Pachi enter, at the simulations and leaf
batch they played it with. By default only the matches with Pachi's
resignation off (labels ending in -nr) enter; with --resign only those
with it on (Pachi's default). The build and self-play checks (the GCC
build of Pachi against the clang build, Pachi against itself at the same
playouts) are left out. Prints:

1. The anchor: a Bradley-Terry (Elo) fit of GNU Go and the Pachi levels
   to their own matches only (GNU Go against each level, each level
   against the next), with no network's games in it.
2. For each network and Pachi level, the chain "level's anchor Elo plus
   the network's Elo against that level", set against the network's Elo
   against GNU Go directly; the difference and its z-score. Each match's
   Elo has the standard error 400 / ln 10 * sqrt(1 / (n p (1 - p))), and
   independent errors add in quadrature. A match with a score of 0 or 1
   has no finite Elo and is skipped here.
3. A joint fit of every player to every game, GNU Go fixed at 0.
4. The joint fit's residuals per pairing (observed against expected
   score) and Pearson's chi-square over all pairings, with pairings minus
   free ratings as its degrees of freedom: a check that one Elo per
   player describes all the results.

The fits are maximum likelihood (Newton's method), with standard errors
from the inverse of the information matrix.

Standard library only. Written by Mark Castelluccio, with assistance
from Claude (Anthropic).
"""
import csv
import math
import re
import sys

C = math.log(10) / 400   # Elo to natural log-odds
GNUGO = "GNU Go L10"


def pachi(n):
    return "Pachi %dk" % (n // 1000)


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


def fit(pairings, players):
    """Maximum-likelihood ratings (GNU Go = 0) and their standard errors."""
    free = [p for p in players if p != GNUGO]
    idx = {p: i for i, p in enumerate(free)}
    r = {p: 0.0 for p in players}
    for _ in range(200):
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
        sys.exit("the fit did not converge (a player won or lost every game?)")
    cov = inverse(h)
    return r, {p: math.sqrt(cov[i][i]) for p, i in idx.items()}


def player(label):
    m = re.match(r"^pachi(\d+)k$", label)
    if m:
        return pachi(int(m.group(1)) * 1000)
    return GNUGO if label == "gnugo" else label


def print_fit(title, r, rse, players):
    print(title + "\n")
    print("| player | Elo | 95% interval |")
    print("|---|---|---|")
    for p in sorted(players, key=lambda x: r[x]):
        if p == GNUGO:
            print("| %s | 0 | (fixed) |" % p)
        else:
            print("| %s | %s | %s to %s |" % (p, fmt(r[p]), fmt(r[p] - 1.96 * rse[p]), fmt(r[p] + 1.96 * rse[p])))


def main():
    args = sys.argv[1:]
    resign = "--resign" in args
    args = [a for a in args if a != "--resign"]
    pachi_rows = [r for r in csv.DictReader(open(args[0])) if r["match"].endswith("-nr") != resign]
    gnugo_rows = list(csv.DictReader(open(args[1])))
    print("Pachi's resignation %s.\n" % ("on (its default)" if resign else "off"))
    pairings = []        # (A, B, A's score, games)
    nets = []
    for row in pachi_rows:
        label = re.sub(r"-p\d+k(-nr)?$", "", row["match"])
        a = player(label)
        if label == "pachigcc" or a == pachi(int(row["pachi_playouts"])):
            continue   # build and self-play checks
        if a != GNUGO and not a.startswith("Pachi") and a not in nets:
            nets.append(a)
        pairings.append((a, pachi(int(row["pachi_playouts"])), float(row["score"]), int(row["games"])))
    direct = {}
    for row in gnugo_rows:
        if row["match"] in nets:
            pairings.append((row["match"], GNUGO, float(row["score"]), int(row["games"])))
            direct[row["match"]] = (float(row["score"]), int(row["games"]))
    levels = sorted({x for a, b, s, n in pairings for x in (a, b) if x.startswith("Pachi")},
                    key=lambda x: int(x[6:-1]))
    players = [GNUGO] + levels + nets

    anchor = [q for q in pairings if q[0] not in nets and q[1] not in nets]
    print("1. The anchor: GNU Go 3.8 level 10 and the Pachi levels\n")
    print("| A | B | A's score | Elo A - B (95% interval) |")
    print("|---|---|---|---|")
    for a, b, s, n in anchor:
        p = s / n
        if 0 < p < 1:
            print("| %s | %s | %g/%d | %s (%s to %s) |" % (a, b, s, n, fmt(elo(p)), fmt(elo(p) - 1.96 * se(p, n)),
                                                         fmt(elo(p) + 1.96 * se(p, n))))
        else:
            print("| %s | %s | %g/%d | %s |" % (a, b, s, n, "infinite"))
    anchored = [x for x in levels if any(x in q[:2] for q in anchor)]
    ar, arse = fit(anchor, [GNUGO] + anchored)
    print()
    print_fit("Fitted to these matches only:", ar, arse, [GNUGO] + anchored)

    print("\n2. Each network through each Pachi level against GNU Go directly\n")
    print("| network | via | score vs level | Elo vs level | rating via level | rating vs GNU Go directly "
          "| difference (z) |")
    print("|---|---|---|---|---|---|---|")
    for net in nets:
        ds, dn = direct.get(net, (None, None))
        dp = ds / dn if ds is not None else None
        for a, b, s, n in pairings:
            if a != net or b == GNUGO:
                continue
            p = s / n
            if not 0 < p < 1 or b not in ar:
                print("| %s | %s | %g/%d | (no finite chain) | | | |" % (net, b, s, n))
                continue
            via, via_se = ar[b] + elo(p), math.hypot(arse[b], se(p, n))
            if dp is not None and 0 < dp < 1:
                d, dse = elo(dp), se(dp, dn)
                diff = via - d
                print("| %s | %s | %g/%d | %s | %s ± %.0f | %s ± %.0f | %s (%+.1f) |" % (
                    net, b, s, n, fmt(elo(p)), fmt(via), 1.96 * via_se, fmt(d), 1.96 * dse, fmt(diff),
                    diff / math.hypot(via_se, dse)))
            else:
                print("| %s | %s | %g/%d | %s | %s ± %.0f | | |" % (net, b, s, n, fmt(elo(p)), fmt(via),
                                                                   1.96 * via_se))

    r, rse = fit(pairings, players)
    print()
    print_fit("3. Joint fit (all the games above; GNU Go 3.8 level 10 = 0)", r, rse, players)

    print("\n4. Residuals of the joint fit\n")
    print("| A | B | games | A's score | expected | Pearson residual |")
    print("|---|---|---|---|---|---|")
    chi2 = 0.0
    for a, b, s, n in pairings:
        e = expected(r[a] - r[b])
        z = (s - n * e) / math.sqrt(n * e * (1 - e))
        chi2 += z * z
        print("| %s | %s | %d | %g | %.1f | %+.2f |" % (a, b, n, s, n * e, z))
    df = len(pairings) - (len(players) - 1)
    print("\nPearson chi-square %.1f on %d degrees of freedom (%d pairings, %d free ratings)" % (
        chi2, df, len(pairings), len(players) - 1))


if __name__ == "__main__":
    main()
