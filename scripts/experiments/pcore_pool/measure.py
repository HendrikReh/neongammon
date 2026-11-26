# SPDX-License-Identifier: MPL-2.0
"""A/B the automatic multi-ply pool on the verified seed-42 five-point game."""
import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import runpy
from statistics import median
import subprocess
import sys
from time import perf_counter_ns
from unittest.mock import patch


def native_inputs(rows):
    checkers, cubes = rows
    return (
        [dict(board=r["board"], die1=r["dice"][0], die2=r["dice"][1],
              force_boards=[r["force_board"]], **r["context"]) for r in checkers],
        [dict(board=r["board"], **r["context"]) for r in cubes],
    )


def controls(ng, native, models, level, inputs, checker):
    """Observe the real native calls and grow a previously used pool in-place."""
    batch = ng.batch_checker_play if checker else ng.batch_cube_action_nply
    binding = "batch_checker_play" if checker else "batch_cube_action_nply"
    width_arg = 5 if checker else 4
    reference = None
    pool_threads = []
    for budget in (0, 1, 4, os.cpu_count()):
        with patch.object(native, binding, wraps=getattr(native, binding)) as observed:
            results = batch(inputs, eval_level=level, models=models, n_threads=budget,
                            jacoby=False, beaver=False)
        observed.assert_called_once()
        if observed.call_args.args[width_arg] != budget:
            raise ValueError(f"{binding} received the wrong thread budget")
        ps = subprocess.check_output(["/bin/ps", "-M", "-p", str(os.getpid())], text=True)
        pool_threads.append(len(ps.strip().splitlines()) - 1)
        values = [asdict(result) for result in results]
        if reference is None:
            reference = values
        elif values != reference:
            raise ValueError(f"{binding} changed a field with explicit budget {budget}")
    # On Darwin ps -M reports all Mach threads. Compare deltas because Python
    # and Accelerate can have threads of their own in this process.
    if hasattr(native, "default_search_threads"):
        expected_growth = os.cpu_count() - native.default_search_threads()
        if pool_threads[-1] - pool_threads[0] != expected_growth:
            raise ValueError(f"{binding} pool did not grow as requested: {pool_threads}")
    for budget in (1, 4, os.cpu_count()):
        analyzer = ng.BgBotAnalyzer(eval_level=level, models=models, parallel_threads=budget)
        if analyzer._analyzer._inner._parallel_threads != budget:
            raise ValueError(f"Scalar analyzer dropped explicit budget {budget}")
        scalar = analyzer.checker_play if checker else analyzer.cube_action
        if [asdict(scalar(**row, beaver=False)) for row in inputs] != reference:
            raise ValueError(f"Scalar and batch differ at explicit budget {budget}")
    return dict(result=reference, threads=pool_threads)


def worker(build, depth, workload):
    sys.path.insert(0, str(build))
    import bgbot_cpp
    import neongammon as ng

    if (Path(bgbot_cpp.__file__).resolve().parent != build or
            Path(ng.__file__).resolve().parents[1] != build):
        raise ValueError("The requested native extension and Python sources must be loaded together")
    checkers, cubes = native_inputs(json.load(sys.stdin))
    models = ng.BatchModels()
    level = f"{depth}ply"
    if workload.startswith("control-"):
        checker = workload == "control-checker"
        inputs = checkers if checker else cubes
        print(json.dumps(controls(ng, bgbot_cpp, models, level, inputs[:2], checker), allow_nan=False))
        return
    analyzer = ng.BgBotAnalyzer(eval_level=level, models=models)
    models.multipy(depth, 5, 0.08)
    budget = analyzer._analyzer._inner._parallel_threads
    options = dict(models=models, eval_level=level, jacoby=False, beaver=False)

    def game():
        results = []
        for rows, batch, scalar in [(checkers, ng.batch_checker_play, analyzer.checker_play),
                                    (cubes, ng.batch_cube_action_nply, analyzer.cube_action)]:
            for start in range(0, len(rows), 4):
                chunk = rows[start:start + 4]
                if len(chunk) == 1:
                    results.append(scalar(**chunk[0], beaver=False))
                else:
                    results.extend(batch(chunk, **options))
        return results

    if workload == "game":
        call = game
    elif workload == "checker":
        def call():
            return [analyzer.checker_play(**checkers[0], beaver=False)]
    else:
        def call():
            return [analyzer.cube_action(**cubes[0], beaver=False)]

    times = []
    reference = None
    for _ in range(1 if workload == "game" else 7):
        start = perf_counter_ns()
        results = call()
        times.append((perf_counter_ns() - start) / 1e6)
        values = [asdict(result) for result in results]
        if reference is None:
            reference = values
        elif reference != values:
            raise ValueError("Repeated call returned different fields")
    print(json.dumps(dict(times=times, result=reference, budget=budget), allow_nan=False))


def launch(build, depth, workload, fixture):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    env.pop("BGBOT_MULTIPLY_THREADS", None)
    proc = subprocess.run([sys.executable, "-IB", __file__, "worker", str(build.resolve()),
                           str(depth), workload], input=fixture,
                          text=True, capture_output=True, env=env, timeout=900)
    if proc.returncode:
        raise RuntimeError(f"{build.name} {depth}-ply {workload} failed: {proc.stderr}")
    return json.loads(proc.stdout)


def run(args):
    if version("neongammon") != "3.2.1":
        raise ValueError("Use the verified 3.2.1 interpreter")
    import neongammon as ng

    rows = runpy.run_path(str(args.fixture_script))["fixture"]()
    if tuple(map(len, rows)) != (44, 43) or len(ng.possible_moves(rows[0][3]["board"], 2, 2)) != 133:
        raise ValueError("Unexpected fixture")
    fixture = json.dumps(rows)
    widths = [os.cpu_count(), int(subprocess.check_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"]))]
    builds = [args.build_dir / name for name in ("baseline", "candidate")]
    for depth in args.depths:
        if args.controls_only:
            for workload in ("control-checker", "control-cube"):
                baseline, candidate = [launch(build, depth, workload, fixture) for build in builds]
                if baseline["result"] != candidate["result"]:
                    raise ValueError(f"{depth}-ply {workload} differs across builds")
                print(f"{depth}-ply {workload}: native arguments 0/1/4/{widths[0]}, "
                      f"process thread counts {baseline['threads']} / {candidate['threads']}", flush=True)
            continue
        for workload in args.workloads:
            samples = {build.name: [] for build in builds}
            reference = None
            for pair in range(args.pairs):
                order = builds if (pair + args.reverse) % 2 == 0 else reversed(builds)
                for build in order:
                    result = launch(build, depth, workload, fixture)
                    expected = widths[builds.index(build)]
                    if result["budget"] != expected:
                        raise ValueError(f"{build.name} automatic analyzer budget should be {expected}")
                    if len(result["result"]) != (87 if workload == "game" else 1):
                        raise ValueError("Incomplete result")
                    if reference is None:
                        reference = result["result"]
                    elif reference != result["result"]:
                        raise ValueError(f"{build.name} {depth}-ply {workload} result differs")
                    times = result["times"]
                    samples[build.name].append(times)
                    print(f"{depth}-ply {workload} pair {pair + 1} {build.name} "
                          f"first {times[0]:.3f} ms", flush=True)
            for name, times in samples.items():
                first = median(t[0] for t in times)
                print(f"{depth}-ply {workload} {name}: median {first:.3f} ms first", end="")
                if workload != "game":
                    print(f", {median(x for t in times for x in t[-5:]):.3f} ms warm", end="")
                else:
                    print(f", {87_000 / first:.2f} decisions/s", end="")
                print(flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["worker"]:
        _, _, path, depth, workload = sys.argv
        worker(Path(path), int(depth), workload)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("build_dir", type=Path)
        parser.add_argument("fixture_script", type=Path)
        parser.add_argument("--pairs", type=int, default=4)
        parser.add_argument("--depths", nargs="+", choices=(3, 4), type=int, default=(3, 4))
        parser.add_argument("--workloads", nargs="+", choices=("checker", "cube", "game"),
                            default=("checker", "cube", "game"))
        parser.add_argument("--reverse", action="store_true", help="Start with candidate in the first pair")
        parser.add_argument("--controls-only", action="store_true", help="Run untimed scalar and batch width checks")
        args = parser.parse_args()
        if args.pairs < 2 or args.pairs % 2:
            parser.error("use an even number of pairs, at least two")
        run(args)
