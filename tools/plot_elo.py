#!/usr/bin/env python3
"""Draw Elo curves from elo.csv files as one self-contained SVG line chart.

    python3 tools/plot_elo.py docs/results/5x5/elo.csv [more.csv ...] > out.svg
    python3 tools/plot_elo.py "seed 1=docs/results/5x5/elo.csv" ... > out.svg

Each input is a CSV with at least the columns `generation` and `elo` (the
`elo.csv` a training run writes, or any two-column generation,elo file).
Lines starting with `#` are comments, and rows whose `elo` is empty are
skipped. Each file becomes one polyline, labelled with its parent
directory's name, or with LABEL when the argument is written LABEL=PATH.

The SVG has a 640x360 viewBox and no fixed width, so it scales to its
container. Colors are CSS variables with fallbacks -- the first series is
`var(--x, #2C5FA8)`, the second `var(--o, #BD5A26)`, later ones repeat the
two dashed -- so inside the explainer page the chart follows its light and
dark themes, and anywhere else it renders with the light colors on a white
background. Standard library only; exits 1 if a file has no usable rows.
"""
import csv
import html
import math
import pathlib
import sys

WIDTH, HEIGHT = 640, 360
LEFT, RIGHT, TOP, BOTTOM = 70, 18, 18, 50
COLORS = ["var(--x, #2C5FA8)", "var(--o, #BD5A26)"]
INK = "var(--ink, #1A1D23)"
INK_DIM = "var(--ink-dim, #565E6B)"
RULE = "var(--rule, #DADDE4)"
RULE_STRONG = "var(--rule-strong, #C3C8D1)"
SURFACE = "var(--surface, #FFFFFF)"
FONT = "IBM Plex Mono, ui-monospace, Menlo, monospace"


def fail(message):
    print(f"plot_elo: {message}", file=sys.stderr)
    sys.exit(1)


def read_series(arg):
    """(label, [(generation, elo), ...]) from `PATH` or `LABEL=PATH`."""
    label, _, path = arg.rpartition("=")
    file = pathlib.Path(path)
    if not label:
        label = file.resolve().parent.name
    try:
        lines = [line for line in file.read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    except OSError as e:
        fail(f"cannot read {path}: {e.strerror}")
    reader = csv.DictReader(lines)
    if reader.fieldnames is None or not {"generation", "elo"} <= set(reader.fieldnames):
        fail(f"{path}: needs a header with the columns generation and elo")
    points = []
    for row in reader:
        if not (row.get("elo") or "").strip():
            continue
        try:
            points.append((int(row["generation"]), float(row["elo"])))
        except ValueError:
            fail(f"{path}: bad row {row}")
    if not points:
        fail(f"{path}: no rows with an elo value")
    return label, sorted(points)


def nice_step(span, target_ticks):
    """A step of 1, 2 or 5 x 10^k giving about target_ticks intervals."""
    raw = max(span, 1e-9) / target_ticks
    power = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * power:
            return m * power
    return 10 * power


def ticks(lo, hi, target):
    step = nice_step(hi - lo, target)
    start = math.floor(lo / step) * step
    end = math.ceil(hi / step) * step
    count = int(round((end - start) / step))
    return [start + i * step for i in range(count + 1)]


def fmt(v):
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:g}"


def render(series):
    xs = [g for _, pts in series for g, _ in pts]
    ys = [e for _, pts in series for _, e in pts]
    xt = ticks(min(xs), max(xs), 6)
    yt = ticks(min(ys), max(ys), 6)
    x0, x1, y0, y1 = xt[0], xt[-1], yt[0], yt[-1]
    # The legend sits above the plot area, one row per series, so it never
    # covers a curve.
    top = TOP + 22 * len(series)
    pw, ph = WIDTH - LEFT - RIGHT, HEIGHT - top - BOTTOM

    def px(g):
        return LEFT + (g - x0) / (x1 - x0 or 1) * pw

    def py(e):
        return top + ph - (e - y0) / (y1 - y0 or 1) * ph

    names = ", ".join(label for label, _ in series)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'role="img" aria-label="Elo by generation: {html.escape(names)}" '
        f'font-family="{FONT}" font-size="14">',
        f"<title>Elo by generation: {html.escape(names)}</title>",
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="{SURFACE}"/>',
    ]
    for v in yt:
        y = py(v)
        out.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH - RIGHT}" y2="{y:.1f}" '
                   f'stroke="{RULE}" stroke-width="1"/>')
        out.append(f'<text x="{LEFT - 8}" y="{y + 5:.1f}" text-anchor="end" fill="{INK_DIM}">{fmt(v)}</text>')
    for v in xt:
        x = px(v)
        out.append(f'<line x1="{x:.1f}" y1="{top + ph}" x2="{x:.1f}" y2="{top + ph + 5}" '
                   f'stroke="{RULE_STRONG}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{top + ph + 22}" text-anchor="middle" fill="{INK_DIM}">{fmt(v)}</text>')
    out.append(f'<line x1="{LEFT}" y1="{top + ph}" x2="{WIDTH - RIGHT}" y2="{top + ph}" '
               f'stroke="{RULE_STRONG}" stroke-width="1.2"/>')
    out.append(f'<text x="{LEFT + pw / 2:.1f}" y="{HEIGHT - 6}" text-anchor="middle" fill="{INK_DIM}">generation</text>')
    out.append(f'<text x="16" y="{top + ph / 2:.1f}" text-anchor="middle" fill="{INK_DIM}" '
               f'transform="rotate(-90 16 {top + ph / 2:.1f})">Elo</text>')

    for i, (label, pts) in enumerate(series):
        color = COLORS[i % 2]
        dash = ' stroke-dasharray="7 5"' if i >= 2 else ""
        coords = " ".join(f"{px(g):.1f},{py(e):.1f}" for g, e in pts)
        out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.2" '
                   f'stroke-linejoin="round" stroke-linecap="round"{dash}/>')

    for i, (label, _) in enumerate(series):
        y = TOP + 22 * i
        dash = ' stroke-dasharray="7 5"' if i >= 2 else ""
        out.append(f'<line x1="{LEFT}" y1="{y}" x2="{LEFT + 22}" y2="{y}" stroke="{COLORS[i % 2]}" '
                   f'stroke-width="2.6"{dash}/>')
        out.append(f'<text x="{LEFT + 30}" y="{y + 5}" fill="{INK}">{html.escape(label)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)
        return 1 if not argv else 0
    sys.stdout.write(render([read_series(arg) for arg in argv]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
