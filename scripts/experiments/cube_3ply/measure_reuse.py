# SPDX-License-Identifier: MPL-2.0
"""Measure roll-batch reuse on the 3.2.1 owned-money cube. See docs/benchmarking.md."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import runpy
from statistics import median
import subprocess
import sys


def worker(action, extension):
    if version("neongammon") != "3.2.1":
        raise RuntimeError("Use the official 3.2.1 wheel")
    if extension:
        sys.path.insert(0, str(Path(extension).resolve()))
    import bgbot_cpp
    from neongammon import BgBotAnalyzer

    if extension and Path(bgbot_cpp.__file__).resolve().parent != Path(extension).resolve():
        raise RuntimeError("Diagnostic extension did not load")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmark_cube_action_decision import CASES, sample

    _, board, options = CASES[1]
    if action in ("cube_action", "cube_decision"):
        print(json.dumps(sample(action, board, options)))
        return
    analyzer = BgBotAnalyzer(eval_level="3ply", parallel_threads=4)
    if action == "independent":
        result = analyzer.cube_action(board, incl_2ply_details=True, **options)
        print(json.dumps([asdict(result.probs), result.cubeless_equity]))
        return
    checkers, cubes = json.load(sys.stdin)
    results = []
    for row in checkers:
        result = analyzer.checker_play(row["board"], *row["dice"], force_boards=[row["force_board"]],
                                       beaver=False, **row["context"])
        results.append(asdict(result))
    for row in cubes:
        result = analyzer.cube_action(row["board"], beaver=False, **row["context"])
        results.append(asdict(result))
    print(json.dumps(results, allow_nan=False))


def launch(action, extension=None, disable=False, fixture=None):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    env.pop("NG63_DISABLE_SHARING", None)
    if disable:
        env["NG63_DISABLE_SHARING"] = "1"
    command = [sys.executable, "-I", __file__, "_worker", action]
    if extension:
        command.append(extension)
    proc = subprocess.run(command, env=env, text=True, capture_output=True,
                          input=json.dumps(fixture) if fixture is not None else None)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-4000:])
    return json.loads(proc.stdout), proc.stderr


def timing(extension, pairs):
    variants = {"wheel action": ("cube_action", None, False), "wheel decision": ("cube_decision", None, False),
                "control": ("cube_action", extension, False), "disabled": ("cube_action", extension, True)}
    first, warm = defaultdict(list), defaultdict(list)
    reference = None
    names = list(variants)
    for i in range(pairs):
        # Each variant occupies each position in a four-process block equally often.
        for name in names[i % len(names):] + names[:i % len(names)]:
            (ms, repeats, result), stderr = launch(*variants[name])
            if "NG63 " in stderr:
                raise ValueError("Use the untraced A/B build for timing")
            if reference is None:
                reference = result
            expected = {key: reference[key] for key in result} if name == "wheel decision" else reference
            if result != expected:
                raise ValueError(f"{name} differs from official cube_action")
            first[name].append(ms)
            warm[name].extend(repeats)
    independent, _ = launch("independent")
    if independent != [reference["probs"], reference["cubeless_equity"]]:
        raise ValueError("Independent cubeless traversal differs")
    print(reference["optimal_action"])
    for name in names:
        print(f"{name:15s} first {median(first[name]):7.3f} ms   warm {median(warm[name]):7.3f} ms")


def summarize(lines):
    # The two traversals share a strategy instance within one public call.
    # A node's dice and pre-move board determine the generated candidates.
    # Ordered post-filter rows capture how each traversal actually evaluates them.
    live = defaultdict(list)
    counts = Counter()
    for line in lines:
        _, _, reading, path, roll, model, pin, hit, encoded = line.split()
        boards = tuple(encoded[i:i + 52] for i in range(0, len(encoded), 52))
        key, rows = (boards[0], roll, model, pin), boards[1:]
        if reading == "0":
            live[key].append((path, rows))
            continue
        exact = any(batch == rows for _, batch in live[key])
        shared_rows = set(rows).intersection(row for _, batch in live[key] for row in batch)
        counts["requested_batches"] += 1
        counts["requested_rows"] += len(rows)
        if hit == "1":
            if (path, rows) not in live[key]:
                raise ValueError("Cache hit without identical ordered batch at the same path")
            kind = "hit"
        elif exact:
            kind = "additional_exact"
        elif shared_rows:
            kind = "partial_only"
            counts["partial_intersection_rows"] += len(shared_rows)
        else:
            kind = "no_overlap"
        counts[kind + "_batches"] += 1
        counts[kind + "_rows"] += len(rows)
    return dict(counts)


def trace(extension, processes):
    reference, _ = launch("cube_action")
    totals = {"first": Counter(), "warm": Counter()}
    for _ in range(processes):
        (_, warm, result), stderr = launch("cube_action", extension)
        if result != reference[2]:
            raise ValueError("Trace build differs from official wheel")
        groups = []
        for line in stderr.splitlines():
            if line == "NG63 start":
                groups.append([])
            elif line.startswith("NG63 roll "):
                groups[-1].append(line)
        if len(groups) != 1 + 3 + len(warm) or not any(groups):
            raise ValueError("Unexpected trace count")
        totals["first"].update(summarize(groups[0]))
        for group in groups[-len(warm):]:
            totals["warm"].update(summarize(group))
    for phase, counts in totals.items():
        print(phase, json.dumps(dict(counts), sort_keys=True))


def float_deltas(left, right):
    """Compare the entire result tree, requiring exact discrete fields and shape."""
    if type(left) is not type(right):
        raise ValueError(f"Type mismatch: {left!r}, {right!r}")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("Result fields differ")
        for key in left:
            yield from float_deltas(left[key], right[key])
    elif isinstance(left, list):
        for a, b in zip(left, right, strict=True):
            yield from float_deltas(a, b)
    elif isinstance(left, float):
        yield abs(left - right)
    elif left != right:
        raise ValueError(f"Discrete field mismatch: {left!r}, {right!r}")


def compare_builds(label, reference, actual):
    """Summarize a cross-build comparison after checking shape and discrete values."""
    maximum = 0.0
    for kind, subset in (("checker", slice(0, 44)), ("cube", slice(44, None))):
        a, b = reference[subset], actual[subset]
        delta = max(float_deltas(a, b))
        same = sum(x == y for x, y in zip(a, b, strict=True))
        print(f"{label}: {same}/{len(a)} {kind} rows exactly equal, maximum absolute difference {delta:.9g}")
        maximum = max(maximum, delta)
    return maximum


def parity(extension, pristine, fixture_script, subset_extension):
    fixture = runpy.run_path(str(fixture_script))["fixture"]()
    if tuple(map(len, fixture)) != (44, 43):
        raise ValueError("Unexpected seed-42 fixture")
    official, _ = launch("fixture", fixture=fixture)
    reference, _ = launch("fixture", pristine, fixture=fixture)
    if compare_builds("wheel vs pristine", official, reference) > 1e-6:
        raise ValueError("Official versus pristine source drift exceeds 1e-6")
    for label, disabled in (("sharing", False), ("disabled", True)):
        actual, _ = launch("fixture", extension, disabled, fixture)
        for index, (expected, result) in enumerate(zip(reference, actual, strict=True)):
            if result != expected:
                raise ValueError(f"{label}: differs from pristine source build at row {index}")
        print(f"{label}: all fields exactly match pristine source on 44 checker and 43 cube rows (3-ply)")
    if subset_extension:
        actual, _ = launch("fixture", subset_extension, fixture=fixture)
        compare_builds("pristine vs forced subset reuse", reference, actual)
        owned, _ = launch("cube_action", pristine)
        changed, _ = launch("cube_action", subset_extension)
        fields = [key for key in owned[2] if owned[2][key] != changed[2][key]]
        print("owned-money forced subset fields differing from pristine:", fields)


if __name__ == "__main__":
    if sys.argv[1:2] == ["_worker"]:
        worker(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        subs = parser.add_subparsers(dest="command", required=True)
        bench = subs.add_parser("timing")
        bench.add_argument("extension", type=Path)
        bench.add_argument("--pairs", type=int, default=16, choices=range(4, 65, 4))
        count = subs.add_parser("trace")
        count.add_argument("extension", type=Path)
        count.add_argument("--processes", type=int, default=16)
        verify = subs.add_parser("parity")
        verify.add_argument("extension", type=Path)
        verify.add_argument("pristine", type=Path)
        verify.add_argument("fixture_script", type=Path)
        verify.add_argument("--subset", type=Path)
        args = parser.parse_args()
        if args.command == "timing":
            timing(args.extension, args.pairs)
        elif args.command == "trace":
            trace(args.extension, args.processes)
        else:
            parity(args.extension, args.pristine, args.fixture_script, args.subset)
