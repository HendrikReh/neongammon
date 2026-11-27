# SPDX-License-Identifier: MPL-2.0
"""Compare sequential and overlapping 3-ply cube_action traversals."""

import argparse
import json
import os
from pathlib import Path
import runpy
from statistics import median
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def sample():
    fixture = runpy.run_path(str(ROOT / "scripts/benchmark_cube_action_decision.py"))
    results = [(label, *fixture["sample"]("cube_action", board, options))
               for label, board, options in fixture["CASES"]]
    print(json.dumps(results))


def compare(baseline, overlap, pairs):
    builds = {"baseline": baseline.resolve(), "overlap": overlap.resolve()}
    timings = {build: {} for build in builds}
    reference = {}
    for pair in range(pairs):
        order = builds if pair % 2 == 0 else reversed(builds)
        for name in order:
            env = dict(os.environ, PYTHONPATH=f"{builds[name]}:{ROOT / 'python'}",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
            worker = subprocess.run(
                [sys.executable, __file__, "--sample"], env=env, check=True,
                capture_output=True, text=True, timeout=60,
            )
            for label, first, warm, result in json.loads(worker.stdout):
                if reference.setdefault(label, result) != result:
                    raise ValueError(f"{name} changed a cube result on {label}")
                timings[name].setdefault(label, []).append((first, warm))

    print("All three cube fixtures matched in every field.")
    for label in reference:
        rows = {name: timings[name][label] for name in builds}
        changes = [(median(b[1]) / median(a[1]) - 1) * 100
                   for a, b in zip(rows["baseline"], rows["overlap"], strict=True)]
        print(json.dumps(dict(
            fixture=label,
            first_ms={name: median(first for first, _ in row) for name, row in rows.items()},
            warm_ms={name: median(ms for _, warm in row for ms in warm)
                     for name, row in rows.items()},
            paired_change_percent=changes,
        )))


if __name__ == "__main__":
    if sys.argv[1:] == ["--sample"]:
        sample()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("baseline", type=Path, help="Directory containing the baseline extension")
        parser.add_argument("overlap", type=Path, help="Directory containing the patched extension")
        parser.add_argument("--pairs", type=int, default=12)
        options = parser.parse_args()
        if options.pairs < 1:
            parser.error("--pairs must be positive")
        compare(options.baseline, options.overlap, options.pairs)
