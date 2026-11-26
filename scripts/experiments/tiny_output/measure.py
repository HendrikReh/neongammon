# SPDX-License-Identifier: MPL-2.0
"""Count tiny NN groups, compare all public fields, and time a seed-42 full game."""
import argparse
import ctypes
from dataclasses import asdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import runpy
from statistics import median
import subprocess
import sys
from time import perf_counter


def worker(build, depth, mode):
    # Each process loads one local extension, with the locked wheel's Python and assets.
    sys.path.insert(0, str(build))
    import bgbot_cpp
    import neongammon as ng

    if Path(bgbot_cpp.__file__).resolve().parent != build:
        raise RuntimeError("Unexpected native extension")
    previous = Path(__file__).resolve().parents[1] / "pcore_pool" / "measure.py"
    checkers, cubes = runpy.run_path(str(previous))["native_inputs"](json.load(sys.stdin))
    start = perf_counter()
    models = ng.BatchModels()
    level = f"{depth}ply"
    models.multipy(depth, 5, 0.08)
    analyzer = ng.BgBotAnalyzer(eval_level=level, models=models, parallel_threads=4)
    setup_s = perf_counter() - start
    options = dict(models=models, eval_level=level, jacoby=False, beaver=False, n_threads=4)
    if mode == "trace":
        counter = ctypes.CDLL(bgbot_cpp.__file__).take_group_calls
        counter.argtypes = [ctypes.c_int]
        counter.restype = ctypes.c_uint64
        for bin_index in range(3):
            counter(bin_index)
    groups = []
    results = []
    times = []
    for inputs, batch, scalar in [(checkers, ng.batch_checker_play, analyzer.checker_play),
                                  (cubes, ng.batch_cube_action_nply, analyzer.cube_action)]:
        start = perf_counter()
        for i in range(0, len(inputs), 4):
            chunk = inputs[i:i + 4]
            if len(chunk) == 1:
                results.append(scalar(**chunk[0], beaver=False))
            else:
                results.extend(batch(chunk, **options))
        times.append(perf_counter() - start)
        if mode == "trace":
            groups.append([counter(i) for i in range(3)])
    if len(results) != 87:
        raise ValueError("Incomplete game")
    print(json.dumps(dict(setup_s=setup_s, times=times, groups=groups,
                          results=[asdict(row) for row in results]), allow_nan=False))


def launch(build, depth, mode, rows):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-IB", __file__, "worker", str(build.resolve()), str(depth), mode],
                          env=env, input=rows, text=True, capture_output=True, timeout=900)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    return json.loads(proc.stdout)


def parity(baseline, candidate):
    pairs = list(zip(baseline, candidate, strict=True))
    def boards(row):
        return [tuple(move["board"]) for move in row["moves"]]

    checker_equity_deltas = []
    for a, b in pairs[:44]:
        candidate_moves = dict(zip(boards(b), b["moves"], strict=True))
        for move in a["moves"]:
            other = candidate_moves.get(tuple(move["board"]))
            if other is not None:
                checker_equity_deltas.append(abs(move["equity"] - other["equity"]))
    return dict(
        checker_rows=sum(a != b for a, b in pairs[:44]),
        cube_rows=sum(a != b for a, b in pairs[44:]),
        best_boards=sum(boards(a)[:1] != boards(b)[:1] for a, b in pairs[:44]),
        rankings=sum(boards(a) != boards(b) for a, b in pairs[:44]),
        cube_actions=sum(a["optimal_action"] != b["optimal_action"] for a, b in pairs[44:]),
        max_checker_equity=max(checker_equity_deltas, default=0),
        max_cube_equity=max(abs(a[key] - b[key]) for a, b in pairs[44:]
                            for key in ("equity_nd", "equity_dt", "equity_dp")),
    )


def run(args):
    if version("neongammon") != "3.2.1":
        raise ValueError("Use the locked 3.2.1 interpreter")
    import neongammon as ng

    rows = runpy.run_path(str(args.fixture_script))["fixture"]()
    double_two_moves = [len(ng.possible_moves(row["board"], 2, 2))
                        for row in rows[0] if row["dice"] == [2, 2]]
    if tuple(map(len, rows)) != (44, 43) or 133 not in double_two_moves:
        raise ValueError("Unexpected seed-42 game")
    rows = json.dumps(rows)
    for depth in args.depths:
        if args.mode == "trace":
            trace = launch(args.build_dir / "trace", depth, "trace", rows)
            print(json.dumps(dict(depth=depth, checker=trace["groups"][0], cube=trace["groups"][1])), flush=True)
            continue
        builds = ["baseline", args.candidate]
        samples = {name: [] for name in builds}
        reference = {}
        for pair in range(args.pairs):
            order = builds if (pair + args.reverse) % 2 == 0 else reversed(builds)
            for name in order:
                row = launch(args.build_dir / name, depth, "game", rows)
                results = row.pop("results")
                if name not in reference:
                    reference[name] = results
                elif results != reference[name]:
                    raise ValueError(f"{name} {depth}-ply differed on repeat")
                samples[name].append(sum(row["times"]))
                print(json.dumps(dict(depth=depth, pair=pair + 1, build=name,
                                      setup_s=row["setup_s"], checker_s=row["times"][0],
                                      cube_s=row["times"][1], game_s=sum(row["times"]))), flush=True)
        for name, times in samples.items():
            print(json.dumps(dict(depth=depth, build=name, median_game_s=median(times),
                                  decisions_per_s=87 / median(times))), flush=True)
        print(json.dumps(dict(depth=depth, different=parity(*[reference[name] for name in builds]))), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["worker"]:
        _, _, path, depth, mode = sys.argv
        worker(Path(path), int(depth), mode)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("build_dir", type=Path)
        parser.add_argument("fixture_script", type=Path)
        parser.add_argument("mode", choices=["trace", "compare"])
        parser.add_argument("--candidate", choices=["candidate", "dot_only"], default="candidate")
        parser.add_argument("--depths", type=int, nargs="+", choices=[3, 4], default=[3, 4])
        parser.add_argument("--pairs", type=int, default=2)
        parser.add_argument("--reverse", action="store_true")
        args = parser.parse_args()
        if args.pairs < 1:
            parser.error("--pairs must be positive")
        run(args)
