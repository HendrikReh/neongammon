# SPDX-License-Identifier: MPL-2.0
"""Compare public scalar and batch search on the five-point game from NG #76.

Run with PYTHONPATH=build:python python scripts/bench_search_batches.py.
Timings exclude model loading. Each sample starts with a fresh evaluator identity
so the shared pool cannot reuse results from another method or repetition.
"""
import argparse
from statistics import median
from time import perf_counter

import neongammon as ng


def measure(run, threads):
    models = ng.BatchModels()
    analyzer = ng.BgBotAnalyzer(eval_level="3ply", parallel_threads=threads, models=models)
    models.multipy(3, 5, 0.08)  # Preload the native search before starting the clock.
    options = dict(eval_level="3ply", models=models, n_threads=threads)
    start = perf_counter()
    result = run(analyzer, options)
    return result, (perf_counter() - start) * 1000


def compare(label, methods, repeats, threads):
    expected, _ = measure(methods[0][1], threads)
    samples = [[] for _ in methods]
    for repeat in range(repeats):
        # Rotate the order to distribute clock and thermal effects.
        for offset in range(len(methods)):
            i = (offset + repeat) % len(methods)
            result, elapsed = measure(methods[i][1], threads)
            if result != expected:
                raise AssertionError(f"{label}: {methods[i][0]} differs from scalar")
            samples[i].append(elapsed)
    print(label, *(f"{name}={median(times):.2f} ms" for (name, _), times in zip(methods, samples)),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.threads, args.repeats) < 1:
        parser.error("threads and repeats must be positive")
    game = ng.simulate_game(seed=42, config=ng.GameConfig(match_length=5))
    checkers, cubes = [], []
    for t in game["turns"]:
        context = {k: t[k] for k in ("board", "away1", "away2", "is_crawford", "cube_value", "cube_owner")}
        if t["turn"]:
            cubes.append(context)
        if t["dice"]:
            checkers.append(dict(context, die1=t["dice"][0], die2=t["dice"][1],
                                 cube_value=t["checker_cube_value"], cube_owner=t["checker_cube_owner"]))
    heavy = max(checkers, key=lambda p: len(ng.possible_moves(p["board"], p["die1"], p["die2"])))
    endpoints = len(ng.possible_moves(heavy["board"], heavy["die1"], heavy["die2"]))
    print(f"{len(checkers)} checker and {len(cubes)} pre-roll rows; largest roll has {endpoints} moves")
    def checker_chunks(n, options):
        return [r for i in range(0, len(checkers), n)
                for r in ng.batch_checker_play(checkers[i:i + n], **options)]

    compare("checker", [("scalar", lambda a, _: [a.checker_play(**p) for p in checkers]),
                        ("chunk4", lambda _, opts: checker_chunks(4, opts)),
                        ("chunk64", lambda _, opts: checker_chunks(64, opts))],
            args.repeats, args.threads)
    bearoff = [0, 3, 3, 2, 3, 2, 2] + [0] * 12 + [-2, -3, -2, -3, -2, -2] + [0]
    for size in range(1, 5):
        positions = [dict(cubes[0], board=bearoff)] * (size - 1) + [cubes[0]]
        compare(f"cube 1 contact + {size - 1} bearoff",
                [("scalar", lambda a, _: [a.cube_action(**p) for p in positions]),
                 ("batch", lambda _, opts: ng.batch_cube_action_nply(positions, **opts))],
                args.repeats, args.threads)


if __name__ == "__main__":
    main()
