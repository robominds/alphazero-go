#!/usr/bin/env python3
"""Suggest a selfplay_engines value from a run's measured engine rates.

    python3 -I tools/engine_weights.py RUN_DIR [--last N] [--games G] [--hosts FILE]

Reads RUN_DIR/selfplay-engines.csv (the manifest distributed self-play
writes) and, over its last N generations (default 3), each planned engine's
rate: network evaluations per second, summed evaluations over summed
seconds, from the rows where the planned engine played (fallback rows are
left out). Games cost about the same everywhere, so an engine's fair share
of G games (default: the run's games_per_generation) is proportional to
its rate.

The suggestion follows the spec's sizing rules. GPU engines (metal) play
every game of their slice at once and get weights proportional to their
rates; CPU engines (naive, blas) play in waves of their worker count and
get fixed counts rounded to a multiple of it (the workers come from the
hosts file's cpu_workers, else the default rule: min(4, cores / 3) on
local, 2 on a remote). Without a GPU engine the fastest engine takes the
weight. It prints each engine's predicted finish time (its games times
the run's evaluations per game, over its rate), warns when an engine gets
fewer games than its workers, and ends with the line to apply with
`train --run RUN_DIR --resume --set selfplay_engines=...`.

Choosing weights is a person's decision: the run never rebalances itself,
because an assignment that depends on timings would make its games depend
on them. Standard library only.

Mark Castelluccio, with assistance from Claude (Anthropic).
"""
import argparse
import csv
import math
import os
import pathlib
import sys

MIN_SECONDS = 0.05   # a slice shorter than this (tests) still gives a finite rate


def read_config(run_dir):
    values = {}
    path = run_dir / "config.cfg"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0]
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    return values


def read_cpu_workers(hosts_path):
    """alias -> cpu_workers (0 = the default rule) from a hosts file."""
    workers = {}
    if not hosts_path or not pathlib.Path(hosts_path).exists():
        return workers
    for line in pathlib.Path(hosts_path).read_text().splitlines():
        words = line.split("#", 1)[0].split()
        if not words:
            continue
        workers[words[0]] = 0
        for pair in words[1:]:
            if pair.startswith("cpu_workers="):
                workers[words[0]] = int(pair.split("=", 1)[1])
    return workers


def default_workers(machine, cpu_workers):
    if cpu_workers.get(machine, 0) > 0:
        return cpu_workers[machine]
    if machine == "local":
        return max(1, min(4, (os.cpu_count() or 3) // 3))
    return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir")
    parser.add_argument("--last", type=int, default=3, help="generations to measure (default 3)")
    parser.add_argument("--games", type=int, default=0, help="games per generation (default: the run's)")
    parser.add_argument("--hosts", default="", help="the hosts file (for cpu_workers)")
    args = parser.parse_args()
    run_dir = pathlib.Path(args.run_dir)
    manifest = run_dir / "selfplay-engines.csv"
    if not manifest.exists():
        sys.exit(f"error: {manifest} does not exist (a run with selfplay_engines writes it)")
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"error: {manifest} has no rows")
    last = max(int(r["generation"]) for r in rows)
    window = [r for r in rows if int(r["generation"]) > last - args.last]
    first = min(int(r["generation"]) for r in window)
    games = args.games or int(read_config(run_dir).get("games_per_generation", "0") or 0)
    if games <= 0:
        sys.exit("error: give --games (the run's config.cfg has no games_per_generation)")
    hosts = args.hosts or os.environ.get("AZGO_HOSTS") or str(pathlib.Path.home() / ".config/azgo/hosts")
    cpu_workers = read_cpu_workers(hosts)

    # Rates of the engines that played as planned, in the manifest's order.
    order, evaluations, seconds = [], {}, {}
    fallbacks = {}   # (planned, engine) -> games played as a fallback
    planned_engines = []
    total_evals = total_games = 0
    for r in window:
        total_evals += int(r["evaluations"])
        total_games += int(r["count"])
        if r["planned"] not in planned_engines:
            planned_engines.append(r["planned"])
        if r["planned"] != r["engine"] or r["note"]:
            key = (r["planned"], r["engine"])
            fallbacks[key] = fallbacks.get(key, 0) + int(r["count"])
            continue
        name = r["planned"]
        if name not in evaluations:
            order.append(name)
            evaluations[name] = seconds[name] = 0
        evaluations[name] += int(r["evaluations"])
        seconds[name] += max(float(r["seconds"]), MIN_SECONDS)
    for (planned, engine), n in fallbacks.items():
        print(f"note: {n} games planned for {planned} were played by {engine} as a fallback "
              "(not counted in any rate)")
    missing = [e for e in planned_engines if e not in order]
    for e in missing:
        print(f"warning: {e} only ran as a fallback in these generations: it has no rate and is "
              "left out of the suggestion; check the machine, or measure over more generations",
              file=sys.stderr)
    if not order:
        sys.exit(f"error: no engine played as planned in generations {first}-{last}")
    rate = {e: evaluations[e] / seconds[e] for e in order}
    per_game = total_evals / max(1, total_games)
    total_rate = sum(rate.values())

    gpu = [e for e in order if e.endswith("/metal")]
    weighted = gpu if gpu else [max(order, key=lambda e: rate[e])]
    count = {}
    for e in order:
        if e in weighted:
            continue
        workers = default_workers(e.split("/")[0], cpu_workers)
        ideal = games * rate[e] / total_rate
        count[e] = max(0, int(round(ideal / workers)) * workers)
    while sum(count.values()) > games:   # leave the weighted engines their share
        biggest = max(count, key=lambda e: count[e])
        count[biggest] -= default_workers(biggest.split("/")[0], cpu_workers)
        count[biggest] = max(0, count[biggest])
    rest = games - sum(count.values())
    weight_total = sum(rate[e] for e in weighted)
    for e in weighted:
        count[e] = rest * rate[e] / weight_total   # an estimate: assignGames apportions it

    print(f"generations {first}-{last}: {per_game:.0f} evaluations per game, {games} games")
    print(f"{'engine':<20} {'evals/s':>10} {'games':>8} {'predicted s':>12}")
    items = []
    for e in order:
        predicted = count[e] * per_game / rate[e]
        print(f"{e:<20} {rate[e]:>10.0f} {count[e]:>8.1f} {predicted:>12.0f}")
        machine = e.split("/")[0]
        if e not in weighted:
            workers = default_workers(machine, cpu_workers)
            if count[e] < workers:
                print(f"warning: {e} gets {count[e]:.0f} games, fewer than its {workers} workers",
                      file=sys.stderr)
            items.append(f"{e}={count[e]:.0f}")
        else:
            items.append(f"{e}:{max(1, min(1000000, int(round(rate[e]))))}")
    print("selfplay_engines = " + ", ".join(items))


if __name__ == "__main__":
    main()
