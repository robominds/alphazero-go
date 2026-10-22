#!/usr/bin/env python3
"""Summarize the 13x13 matches against Pachi into one table.

    python3 summarize.py MATCHES_DIR > summary.txt   # also writes results.csv

MATCHES_DIR holds one directory per chunk of a match, written by
tools/gtp_match.py (pachi-match.sh) and named A-vs-B-cNN. A is
agz-gGEN-sSIMS[-bBATCH] (the 13x13-agz network of that generation, the
simulations, and the search's leaf batch when it is not the default 8),
gnugo10 or gnugo0 (GNU Go 3.8 at level 10 or 0), or a Pachi setting; B
is a Pachi setting: uctN (its tree search at -t =N), uctnpN (the same
without its MM patterns), uctlightN (without patterns and with light
playouts), replay or pattern (see pachi-match.sh). The chunks of a
match are pooled.

For each match it reports A's score with a 95% Wilson interval, the Elo
difference A - B it implies (400 * log10(p / (1 - p)), the interval's
ends converted the same way), the number of distinct games (over all
the chunks), the score as Black and as White, how the games ended, and
the mean game length. Every game that ended in two passes or at the move
cap is recounted here Tromp-Taylor (the SGF's moves replayed with
captures; every stone on the board and every empty region reaching only
one color count; komi 7.5), which is how our engine scores. That recount
is the score reported for every match: for our networks it is the
scorer's own count (any game where the two differ stops the script), for
GNU Go and Pachi as A it replaces the scorer's count, and the games
where they name a different winner are counted (scorer_tt_disagreements).
Games where the two engines' own final_score name different winners are
counted too (a_b_disagreements).

For our networks it also counts, from the SGFs, the games where our
engine passed while the game went on (a pass not answered by a pass),
separately for the games it won and lost, with the median of its root
value at the first such pass (from its stderr, a.stderr.log, checked
move by move against the SGFs), and the games it lost by more than half
the board (169 / 2 points: most of its stones captured by the end).

When Pachi's stderr (b.stderr.log) is there and B searches, the median
and the 10th and 90th percentiles of the playouts its root held when it
chose each move (its "*** WINNER is ... (x/N games)" lines) are
reported, a check on -t =N.

It needs the full outputs (the SGFs and the engines' stderr); the
repository keeps only each chunk's results.csv and summary.txt.
results.csv (in the current directory) gets one row per match. Standard
library only. Written by Mark Castelluccio, with assistance from Claude
(Anthropic).
"""
import csv
import glob
import math
import os
import re
import statistics
import sys

SIZE, KOMI = 13, 7.5
LABEL = re.compile(r"^(?P<a>.+)-vs-(?P<b>uct\d+|uctnp\d+|uctlight\d+|replay|pattern)-c(?P<chunk>\d+)$")
NET = re.compile(r"^agz-g(?P<gen>\d+)-s(?P<sims>\d+)(?:-b(?P<batch>\d+))?$")
SCORE = re.compile(r"^([BW])\+(\d+(?:\.\d+)?)$|^0$", re.I)
WINNER_LINE = re.compile(r"^\*\*\* WINNER is \S+ with score \S+ \(\d+/(\d+) games\)")
VALUE_LINE = re.compile(r"^genmove [BW]: (\S+), .* value ([+-]\d+\.\d+)")


def name(token):
    """A display name for a label's player token."""
    m = NET.match(token)
    if m:
        return "13x13-agz gen %s, %s sims" % (m.group("gen"), m.group("sims"))
    m = re.match(r"^(uct|uctnp|uctlight)(\d+)$", token)
    if m:
        kind = {"uct": "Pachi", "uctnp": "Pachi no patterns", "uctlight": "Pachi light"}[m.group(1)]
        return "%s =%s" % (kind, m.group(2))
    return {"gnugo10": "GNU Go 3.8 level 10", "gnugo0": "GNU Go 3.8 level 0",
            "replay": "Pachi replay", "pattern": "Pachi pattern"}[token]


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


def playouts(paths):
    n = []
    for path in paths:
        if os.path.exists(path):
            n += [int(m.group(1)) for m in map(WINNER_LINE.match, open(path, errors="replace")) if m]
    if not n:
        return None
    q = statistics.quantiles(n, n=10)
    return {"moves": len(n), "median": statistics.median(n), "p10": q[0], "p90": q[-1]}


def a_values(path):
    """[(move, value)] of our engine's genmoves, in order, from its stderr."""
    return [(m.group(1), float(m.group(2))) for m in map(VALUE_LINE.match, open(path, errors="replace")) if m]


def midgame_cell(r):
    if r["ours_passed_midgame_won"] == "":
        return ""
    def one(k):
        v = r["ours_value_at_midgame_pass_%s_median" % k]
        return "%s%s" % (r["ours_passed_midgame_" + k], (" (%s)" % v) if v else "")
    return "%s / %s" % (one("won"), one("lost"))


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
        m = LABEL.match(os.path.basename(d))
        if not m:
            sys.exit("%s: not a match label" % d)
        chunks.setdefault((m.group("a"), m.group("b")), []).append(d)
    rows = []
    for (a_tok, b_tok), dirs in chunks.items():
        net = NET.match(a_tok)
        ours = bool(net)
        score = black = 0.0
        n = 0
        ends = {"two passes": 0, "resigned": 0, "forfeit": 0, "max moves": 0}
        resigned_by = {"A": 0, "B": 0}
        scorer_tt = ab_dis = moves_total = 0
        sequences = set()
        wipeouts = 0
        midgame = {"won": [], "lost": []}   # our root value at a mid-game pass, by the game's result
        for d in dirs:
            summary = open(os.path.join(d, "summary.txt")).read()
            if not re.search(r"distinct games \d+", summary):
                sys.exit("%s: unfinished (no summary)" % d)
            games = list(csv.DictReader(open(os.path.join(d, "results.csv"))))
            values = a_values(os.path.join(d, "a.stderr.log")) if ours else []
            vi = 0
            for g in games:
                n += 1
                e = g["end"]
                key_end = next(k for k in ends if e.startswith(k) or k in e)
                ends[key_end] += 1
                moves_total += int(g["moves"])
                a_black = g["black"] == "A"
                moves = sgf_moves(os.path.join(d, "game_%04d.sgf" % int(g["game"])))
                sequences.add(tuple(moves))
                pts = float(g["score_a"])
                if key_end in ("two passes", "max moves"):
                    tt = tromp_taylor(moves)
                    tt_pts = 0.5 if tt == 0 else float((tt > 0) == a_black)
                    if tt_pts != pts:
                        if ours:
                            sys.exit("%s game %s: our engine scored %s, the recount %+g" % (
                                d, g["game"], g["a_final_score"], tt))
                        scorer_tt += 1
                    pts = tt_pts
                    a, b = signed(g["a_final_score"]), signed(g["b_final_score"])
                    if a is not None and b is not None and (a > 0) - (a < 0) != (b > 0) - (b < 0):
                        ab_dis += 1
                    if ours and pts == 0 and abs(tt) > SIZE * SIZE / 2:
                        wipeouts += 1
                if key_end == "resigned":
                    resigned_by["A" if (e[0] == "B") == a_black else "B"] += 1
                if ours:
                    a_color = "B" if a_black else "W"
                    a_moves = [p for c, p in moves if c == a_color]
                    # our genmoves of this game, from its stderr, in order
                    mine = values[vi:vi + len(a_moves)]
                    vi += len(a_moves)
                    if len(mine) != len(a_moves) or any((v[0] == "pass") != (p is None) for v, p in zip(mine, a_moves)):
                        sys.exit("%s game %s: our stderr does not match the SGF" % (d, g["game"]))
                    for i, (c, p) in enumerate(moves):
                        if c == a_color and p is None and i + 1 < len(moves) and moves[i + 1][1] is not None:
                            k = sum(1 for c2, _ in moves[:i] if c2 == a_color)
                            midgame["won" if pts == 1 else "lost"].append(mine[k][1])
                            break
                score += pts
                black += pts if a_black else 0
        lo, hi = wilson(score, n)
        p = score / n
        po = None if b_tok in ("replay", "pattern") else playouts([os.path.join(d, "b.stderr.log") for d in dirs])
        gen = net.group("gen") if net else ""
        sims = net.group("sims") if net else ""
        batch = (min(int(net.group("batch") or 8), int(sims)) if net else "")
        rows.append({
            "match": "%s-vs-%s" % (a_tok, b_tok), "a": a_tok, "b": b_tok, "player": name(a_tok),
            "opponent": name(b_tok), "chunks": len(dirs), "backend_chunks": backends(dirs) if ours else "",
            "generation": gen, "simulations": sims,
            "mcts_batch": batch, "games": n, "score": "%g" % score, "fraction": "%.3f" % p,
            "ci_low": "%.3f" % lo, "ci_high": "%.3f" % hi,
            "elo": fmt(elo(p)), "elo_low": fmt(elo(lo)), "elo_high": fmt(elo(hi)),
            "distinct": len(sequences), "score_as_black": "%g" % black, "score_as_white": "%g" % (score - black),
            "two_passes": ends["two passes"], "b_resigned": resigned_by["B"], "a_resigned": resigned_by["A"],
            "forfeits": ends["forfeit"], "max_moves": ends["max moves"],
            "scorer_tt_disagreements": scorer_tt, "a_b_disagreements": ab_dis,
            "mean_moves": "%.1f" % (moves_total / n),
            "ours_passed_midgame_won": len(midgame["won"]) if ours else "",
            "ours_value_at_midgame_pass_won_median": ("%+.2f" % statistics.median(midgame["won"])) if midgame["won"] else "",
            "ours_passed_midgame_lost": len(midgame["lost"]) if ours else "",
            "ours_value_at_midgame_pass_lost_median": ("%+.2f" % statistics.median(midgame["lost"])) if midgame["lost"] else "",
            "ours_lost_by_over_half_board": wipeouts if ours else "",
            "pachi_moves_logged": po["moves"] if po else "",
            "pachi_root_playouts_median": "%.0f" % po["median"] if po else "",
            "pachi_root_playouts_p10": "%.0f" % po["p10"] if po else "",
            "pachi_root_playouts_p90": "%.0f" % po["p90"] if po else "",
        })

    def order(r):
        net = NET.match(r["a"])
        return (0 if net else 1, -int(net.group("gen")) if net else 0, r["a"], r["b"])
    rows.sort(key=order)
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("| A | B | score | 95% interval | Elo A - B (interval) | distinct | as B / as W "
          "| ends: passes / resign / cap / forfeit | scorer vs TT | A vs B count | mean moves "
          "| ours passed mid-game, won / lost (median value) | ours lost by > half the board "
          "| Pachi root playouts p10 / median / p90 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %s | %s | %s/%d = %s | %s-%s | %s (%s to %s) | %d | %s / %s | %d / %d / %d / %d | %d | %d | %s "
              "| %s | %s | %s / %s / %s |" % (
                  r["player"], r["opponent"], r["score"], r["games"], r["fraction"], r["ci_low"], r["ci_high"],
                  r["elo"], r["elo_low"], r["elo_high"], r["distinct"], r["score_as_black"],
                  r["score_as_white"], r["two_passes"], r["b_resigned"] + r["a_resigned"], r["max_moves"],
                  r["forfeits"], r["scorer_tt_disagreements"], r["a_b_disagreements"], r["mean_moves"],
                  midgame_cell(r),
                  r["ours_lost_by_over_half_board"], r["pachi_root_playouts_p10"],
                  r["pachi_root_playouts_median"], r["pachi_root_playouts_p90"]))


if __name__ == "__main__":
    main()
