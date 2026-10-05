#!/usr/bin/env python3
"""Play a match between two GTP engines and rate the result.

    python3 tools/gtp_match.py --a "build-blas/gtp --checkpoint gen_0200.azn --backend blas" \\
        --b "gnugo --mode gtp --chinese-rules --capture-all-dead" \\
        --games 40 --size 9 --komi 7.5 --out runs/gnugo-match

Starts both engines once (each command line is split like a shell would,
without running a shell), then plays --games games between them with
colors alternating: A is Black in odd-numbered games (game 1, 3, ...), so
an even count gives each side Black equally often. Before each game both
engines get boardsize, clear_board and komi; then the side to move is
asked for genmove and its answer is relayed to the other engine with play.

A game ends when both sides pass in a row, when --max-moves moves have been
played (default 2 * size * size, the project's own cap), or when a side
answers resign. A finished game is scored by asking the --scorer engine
(A by default; give the alphazero-go engine that role) for final_score;
the other engine is asked too and its answer is recorded, and the summary
counts the games where it names a different winner. Our engine scores Tromp-Taylor:
every stone on the board counts, so a program that passes with dead
stones still on the board (GNU Go does unless told --capture-all-dead) is
scored as if they were alive. A move that the other engine refuses (for
example a ko retake that is legal under one program's ko rule but not
under the other's) ends the game as a loss for the side that played it,
and is recorded as a forfeit.

Results: --out DIR receives one SGF per game (game_0001.sgf, ...; PB and
PW are the engines' name and version, RE the scorer's result, C how the
game ended and both engines' scores), results.csv with one row per game,
and each engine's stderr (a.stderr.log, b.stderr.log). The summary
printed at the end gives A's score with a 95% Wilson interval, the Elo
difference A - B implied by it (400 * log10(p / (1 - p)), with the
interval's ends converted the same way; infinite at a score of 0 or 1),
and the number of distinct games: two engines that both play
deterministically repeat the same two games, and only distinct games carry
information. `gtp --random-symmetry` makes our engine vary its play.

Standard library only. Written by Mark Castelluccio, with assistance from
Claude (Anthropic).
"""
import argparse
import csv
import math
import os
import re
import shlex
import subprocess
import sys

COLUMNS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"   # GTP skips I
SCORE = re.compile(r"^(?:([BW])\+(\d+(?:\.\d+)?)|0)$", re.IGNORECASE)


class EngineError(RuntimeError):
    pass


class Engine:
    """One GTP engine process: send a command, read its response."""

    def __init__(self, label, command, stderr_path):
        self.label = label
        self.command = command
        self.stderr = open(stderr_path, "w")
        try:
            self.proc = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=self.stderr,
                                         text=True, bufsize=1)
        except OSError as e:
            self.stderr.close()
            raise EngineError("%s: cannot start %r: %s" % (label, command, e))
        self.name = self.label

    def send(self, command):
        """(succeeded, text) for one command; raises EngineError if the engine dies."""
        try:
            self.proc.stdin.write(command + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise EngineError("%s (%s) stopped reading before %r" % (self.label, self.command, command))
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise EngineError("%s (%s) exited during %r; see its stderr log"
                                  % (self.label, self.command, command))
            line = line.rstrip("\r\n")
            if line.strip() == "":
                if lines:
                    break        # the blank line that ends a response
                continue         # stray blank lines before one
            lines.append(line)
        head = lines[0]
        if head[0] not in "=?":
            raise EngineError("%s answered %r with %r, not a GTP response" % (self.label, command, head))
        text = "\n".join([re.sub(r"^[=?]\d*\s?", "", head)] + lines[1:]).strip()
        return head[0] == "=", text

    def must(self, command):
        ok, text = self.send(command)
        if not ok:
            raise EngineError("%s refused %r: %s" % (self.label, command, text))
        return text

    def close(self):
        if self.proc.poll() is None:
            try:
                self.send("quit")
                self.proc.wait(timeout=10)
            except (EngineError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait()
        self.stderr.close()


def parse_vertex(text, n):
    """(column, row from the top) for a vertex, None for pass; ValueError otherwise."""
    t = text.strip().upper()
    if t == "PASS":
        return None
    m = re.fullmatch(r"([A-HJ-Z])(\d{1,2})", t)
    if not m or COLUMNS.index(m.group(1)) >= n or not 1 <= int(m.group(2)) <= n:
        raise ValueError("not a vertex on a %dx%d board: %r" % (n, n, text))
    return COLUMNS.index(m.group(1)), n - int(m.group(2))


def parse_score(text):
    """Black-view score from 'B+3.5', 'W+7', '0', or None if unreadable."""
    m = SCORE.match(text.strip().split()[0] if text.strip() else "")
    if not m:
        return None
    if m.group(1) is None:
        return 0.0
    return float(m.group(2)) * (1 if m.group(1).upper() == "B" else -1)


def sgf_escape(s):
    return s.replace("\\", "\\\\").replace("]", "\\]")


def write_sgf(path, n, komi, black, white, result, comment, moves):
    out = ["(;GM[1]FF[4]CA[UTF-8]SZ[%d]KM[%g]RU[Tromp-Taylor, positional superko]" % (n, komi),
           "PB[%s]PW[%s]RE[%s]C[%s]" % (sgf_escape(black), sgf_escape(white), result, sgf_escape(comment))]
    for k, v in enumerate(moves):
        point = "" if v is None else chr(ord("a") + v[0]) + chr(ord("a") + v[1])
        out.append(";%s[%s]" % ("B" if k % 2 == 0 else "W", point))
    out.append(")\n")
    with open(path, "w") as f:
        f.write("".join(out))


def wilson(score, games, z=1.96):
    if games <= 0:
        return 0.0, 1.0
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


def fmt_elo(x):
    return ("+inf" if x > 0 else "-inf") if math.isinf(x) else "%+.0f" % x


def play_game(black, white, n, komi, max_moves):
    """Plays one game: (moves, how it ended, the winner 'B'/'W' if decided by
    resignation or forfeit else None, {label: final_score text or None})."""
    for e in (black, white):
        e.must("boardsize %d" % n)
        e.must("clear_board")
        e.must("komi %g" % komi)
    moves, passes = [], 0
    players = {"B": black, "W": white}
    while True:
        color = "B" if len(moves) % 2 == 0 else "W"
        other = "W" if color == "B" else "B"
        mover = players[color]
        answer = mover.must("genmove %s" % color)
        if answer.strip().lower() == "resign":
            return moves, "%s resigned" % color, other, {}
        try:
            vertex = parse_vertex(answer, n)
        except ValueError as e:
            return moves, "forfeit: %s (%s) answered %s" % (color, mover.label, e), other, {}
        spelled = "pass" if vertex is None else COLUMNS[vertex[0]] + str(n - vertex[1])
        ok, text = players[other].send("play %s %s" % (color, spelled))
        if not ok:
            return (moves, "forfeit: %s (%s) played %s, refused by %s: %s"
                    % (color, mover.label, spelled, players[other].label, text), other, {})
        moves.append(vertex)
        passes = passes + 1 if vertex is None else 0
        if passes >= 2 or len(moves) >= max_moves:
            end = "two passes" if passes >= 2 else "max moves (%d)" % max_moves
            scores = {}
            for e in (black, white):
                ok, text = e.send("final_score")
                scores[e.label] = text if ok else None
            return moves, end, None, scores


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", required=True, help="engine A's command line")
    ap.add_argument("--b", required=True, help="engine B's command line")
    ap.add_argument("--games", type=int, default=2, help="games to play (even: colors balanced)")
    ap.add_argument("--size", type=int, default=9, help="board size")
    ap.add_argument("--komi", type=float, default=7.5)
    ap.add_argument("--max-moves", type=int, default=0, help="adjudicate after this many moves (0: 2*size*size)")
    ap.add_argument("--scorer", choices=["a", "b"], default="a", help="whose final_score decides a game")
    ap.add_argument("--out", required=True, help="directory for SGFs, results.csv and engine logs")
    args = ap.parse_args()
    if args.games < 2 or args.games % 2:
        ap.error("--games must be even and at least 2 (colors are balanced)")
    if not 2 <= args.size <= 25:
        ap.error("--size must be in [2, 25]")
    max_moves = args.max_moves or 2 * args.size * args.size
    if max_moves < 1:
        ap.error("--max-moves must be positive")
    os.makedirs(args.out, exist_ok=True)

    engines = []
    try:
        a = Engine("A", args.a, os.path.join(args.out, "a.stderr.log"))
        engines.append(a)
        b = Engine("B", args.b, os.path.join(args.out, "b.stderr.log"))
        engines.append(b)
        for e in engines:
            name = e.send("name")
            version = e.send("version")
            e.name = " ".join(t for ok, t in (name, version) if ok and t) or e.command
        scorer = a if args.scorer == "a" else b

        score_a, wins_a, wins_b, draws, disagreements = 0.0, 0, 0, 0, 0
        distinct = set()
        with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
            rows = csv.writer(f)
            rows.writerow(["game", "black", "white", "moves", "end", "result", "score_a", "a_final_score",
                           "b_final_score"])
            for g in range(1, args.games + 1):
                black, white = (a, b) if g % 2 == 1 else (b, a)
                moves, end, winner, scores = play_game(black, white, args.size, args.komi, max_moves)
                distinct.add(tuple(moves))
                comment = "Game %d: %s. " % (g, end)
                if winner is not None:
                    result = winner + "+R" if "resigned" in end else winner + "+F"
                else:
                    text = scores.get(scorer.label)
                    s = parse_score(text) if text is not None else None
                    if s is None:
                        raise EngineError("%s (the scorer) gave no readable final_score: %r" % (scorer.label, text))
                    winner = "B" if s > 0 else "W" if s < 0 else None
                    result = text.split()[0].upper()
                    other_text = scores.get((b if scorer is a else a).label)
                    other = parse_score(other_text) if other_text is not None else None
                    if other is not None and (other > 0) - (other < 0) != (s > 0) - (s < 0):
                        disagreements += 1   # a different winner, not just a different margin
                    comment += "final_score: A %s, B %s; scored by %s." % (
                        scores.get("A"), scores.get("B"), scorer.label)
                if winner is None:
                    pts = 0.5
                    draws += 1
                elif (winner == "B") == (black is a):
                    pts = 1.0
                    wins_a += 1
                else:
                    pts = 0.0
                    wins_b += 1
                score_a += pts
                write_sgf(os.path.join(args.out, "game_%04d.sgf" % g), args.size, args.komi, black.name,
                          white.name, result, comment, moves)
                rows.writerow([g, black.label, white.label, len(moves), end, result, pts,
                               scores.get("A", ""), scores.get("B", "")])
                f.flush()
                print("game %d: A %s, %s after %d moves (%s); A scores %g" % (
                    g, "Black" if black is a else "White", result, len(moves), end, pts), flush=True)

        lo, hi = wilson(score_a, args.games)
        p = score_a / args.games
        print("A %s vs B %s: A scored %g/%d = %.3f (95%% CI %.3f-%.3f), distinct games %d"
              % (a.name, b.name, score_a, args.games, p, lo, hi, len(distinct)))
        print("A wins %d, B wins %d, draws %d" % (wins_a, wins_b, draws))
        print("Elo A - B: %s (95%% CI %s to %s)" % (fmt_elo(elo(p)), fmt_elo(elo(lo)), fmt_elo(elo(hi))))
        print("final_score: scored by %s; the other engine named a different winner in %d game(s)"
              % (scorer.label, disagreements))
        return 0
    except EngineError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    finally:
        for e in engines:
            e.close()


if __name__ == "__main__":
    sys.exit(main())
