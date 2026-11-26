# SPDX-License-Identifier: MPL-2.0
"""Compare two native extensions with the same installed 3.2.1 Python and models."""
import argparse
from dataclasses import asdict
from functools import partial
from importlib.metadata import version
import json
import os
from pathlib import Path
import runpy
from statistics import median
import subprocess
import sys
from time import perf_counter_ns


def load_analyzer(extension, depth):
    if version("neongammon") != "3.2.1":
        raise ValueError("Use the locked 3.2.1 wheel interpreter")
    sys.path.insert(0, str(Path(extension).resolve()))
    import bgbot_cpp
    from neongammon import BgBotAnalyzer

    if Path(bgbot_cpp.__file__).resolve().parent != Path(extension).resolve():
        raise ValueError("Unexpected extension")
    return BgBotAnalyzer(eval_level=f"{depth}ply", parallel_threads=4)


def fixture_rows(analyzer):
    checkers, cubes = json.load(sys.stdin)
    for row in checkers:
        yield asdict(analyzer.checker_play(row["board"], *row["dice"], force_boards=[row["force_board"]],
                                          beaver=False, **row["context"]))
    for row in cubes:
        yield asdict(analyzer.cube_action(row["board"], beaver=False, **row["context"]))


def resident_kib():
    # macOS ps reports current resident memory in KiB.
    return int(subprocess.check_output(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())]))


def sample(analyzer, case):
    from benchmark_cube_action_decision import CASES

    if case == "bearoff":
        board = [0] * 26
        board[4], board[6], board[19], board[22] = 3, 5, -6, -1
        call = partial(analyzer.post_move_analytics, board)
    else:
        _, board, options = CASES[1]
        call = partial(analyzer.cube_action, board, **options)

    before = resident_kib()
    times = []
    for i in range(9):
        start = perf_counter_ns()
        result = call()
        times.append((perf_counter_ns() - start) / 1e6)
        if i == 0:
            after = resident_kib()
            first = asdict(result)
        elif asdict(result) != first:
            raise ValueError("Repeat returned a different result")
    return dict(first=times[0], warm=times[-5:], rss_mib=(after - before) / 1024, result=first)


def launch(build, case, depth, rows=None):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-I", __file__, "worker", build, case, str(depth)],
                          input=json.dumps(rows) if rows else None, text=True, capture_output=True, env=env)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    return json.loads(proc.stdout)


def parity(builds, fixture_script):
    rows = runpy.run_path(str(fixture_script))["fixture"]()
    if tuple(map(len, rows)) != (44, 43):
        raise ValueError("Expected the seed-42 match fixture")
    for depth in (3, 4):
        baseline, candidate = [launch(build, "fixture", depth, rows) for build in builds]
        if len(baseline) != 87:
            raise ValueError("Incomplete fixture")
        for index, (left, right) in enumerate(zip(baseline, candidate, strict=True)):
            if left != right:
                raise ValueError(f"{depth}-ply differs at row {index}")
        print(f"{depth}-ply: 44 checker and 43 cube results identical across builds", flush=True)


def timing(builds, pairs):
    for case in ("bearoff", "contact"):
        samples = {build: [] for build in builds}
        reference = None
        for pair in range(pairs):
            for build in builds if pair % 2 == 0 else reversed(builds):
                row = launch(build, case, 3)
                if reference is None:
                    reference = row["result"]
                if reference != row["result"]:
                    raise ValueError("Result differs between builds or samples")
                samples[build].append(row)
        for build, rows in samples.items():
            first = median(r["first"] for r in rows)
            warm = median(t for r in rows for t in r["warm"])
            rss = median(r["rss_mib"] for r in rows)
            print(f"{case} {build}: {first:.4f} ms first, {warm:.4f} ms warm, {rss:.3f} MiB RSS growth", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["worker"]:
        _, _, extension, case, depth = sys.argv
        analyzer = load_analyzer(extension, int(depth))
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        result = list(fixture_rows(analyzer)) if case == "fixture" else sample(analyzer, case)
        print(json.dumps(result, allow_nan=False))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("builds", nargs=2, metavar="EXTENSION_DIR", type=Path)
        modes = parser.add_subparsers(required=True, dest="mode")
        modes.add_parser("parity").add_argument("fixture_script", type=Path)
        modes.add_parser("timing").add_argument("--pairs", type=int, default=16)
        args = parser.parse_args()
        if args.mode == "parity":
            parity(args.builds, args.fixture_script)
        else:
            if args.pairs < 2 or args.pairs % 2:
                parser.error("use an even number of pairs")
            timing(args.builds, args.pairs)
