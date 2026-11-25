# SPDX-License-Identifier: MPL-2.0
"""Time the installed analyzer's first and repeated 3-ply cube calls on fixed contact positions."""
from dataclasses import asdict
from importlib.metadata import version
from multiprocessing import get_context
from statistics import median
from time import perf_counter_ns

from neongammon import BgBotAnalyzer

# Refpos 5 and 6 from benchmark_4ply_cube.py, and SNAKE from test_cube_decision.py.
CASES = [
    ("5-away/4-away, centered",
     [1, 2, -1, 2, 0, 0, 3, 0, 1, 0, 0, 0, -4, 5, 0, 0, 0, -4, 0, -5, 0, 0, 0, 0, 2, 0],
     dict(cube_value=1, cube_owner="centered", away1=5, away2=4, jacoby=False, beaver=False)),
    ("Money, player owns 2-cube",
     [0, 0, 0, 1, 2, -2, 3, 2, 3, 0, 0, 0, -3, 4, 0, 0, 0, -3, 0, -3, -2, -2, 0, 0, 0, 0],
     dict(cube_value=2, cube_owner="player", jacoby=True, beaver=False)),
    ("SNAKE, centered money",
     [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 3, 3, 2, 2, 1, -7, -7, 0],
     dict(cube_value=1, cube_owner="centered", jacoby=True, beaver=False)),
]
PAIRS = 16
WARM_CALLS = 5


def sample(method_name, board, options):
    """One fresh process and analyzer per sample. Time just the public call."""
    analyzer = BgBotAnalyzer(eval_level="3ply", parallel_threads=4)
    method = getattr(analyzer, method_name)
    timings = []
    for _ in range(1 + 3 + WARM_CALLS):
        start = perf_counter_ns()
        result = method(board, **options)
        timings.append((perf_counter_ns() - start) / 1_000_000)
        if len(timings) == 1:
            first = asdict(result)
        elif asdict(result) != first:
            raise ValueError(f"{method_name} returned a different result on repeat")
    return timings[0], timings[-WARM_CALLS:], first


def main():
    methods = ("cube_action", "cube_decision")
    print(f"NeonGammon {version('neongammon')}, 3-ply, 4 threads, median ms")
    print("Position | first action/decision | warm action/decision | action")
    # A sequential spawn pool avoids inheriting native pools or allowing one
    # method to warm the other method's evaluation caches.
    with get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        for label, board, options in CASES:
            first_calls = {name: [] for name in methods}
            warm_calls = {name: [] for name in methods}
            for pair in range(PAIRS):
                results = {}
                for name in methods if pair % 2 == 0 else reversed(methods):
                    first_ms, warm_ms, results[name] = pool.apply(sample, (name, board, options))
                    first_calls[name].append(first_ms)
                    warm_calls[name].extend(warm_ms)
                decision = results["cube_decision"]
                if any(results["cube_action"][k] != v for k, v in decision.items()):
                    raise ValueError(f"Decision or equity mismatch on {label}, pair {pair}")
            first = [median(first_calls[name]) for name in methods]
            warm = [median(warm_calls[name]) for name in methods]
            print(f"{label} | {first[0]:.2f}/{first[1]:.2f} | "
                  f"{warm[0]:.2f}/{warm[1]:.2f} | {decision['optimal_action']}", flush=True)


if __name__ == "__main__":
    main()
