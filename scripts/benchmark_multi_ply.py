# SPDX-License-Identifier: MPL-2.0
"""Time NeonGammon public analysis calls on the fixed Open Sage corpus."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
from statistics import mean, median
import subprocess
from time import perf_counter_ns

import bgbot_cpp
from neongammon import BgBotAnalyzer, WeightConfig
from neongammon.weights import PRODUCTION_MODEL, bearoff_db_path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "multi_ply_benchmark"
LEVELS = ("1ply", "2ply", "3ply", "4ply", "truncated1", "truncated2", "truncated3")
APIS = (
    ("checker_play", "checker_play", "inputs"),
    ("cube_action", "cube_action", "inputs"),
    ("post_move_analytics", "cube_action", "post_move_inputs"),
)
EQUITY_FIELDS = {
    "checker_play": ("equity", "cubeless_equity"),
    "cube_action": ("equity_nd", "equity_dt", "equity_dp", "optimal_equity", "cubeless_equity"),
    "post_move_analytics": ("cubeful_equity", "cubeless_equity"),
}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_corpus():
    reference_dir = DATA / "rollout_positions"
    manifest_path = reference_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    for directory, files in ((DATA, manifest["fixed_sets"]), (reference_dir, manifest["outputs"])):
        for name, expected in files.items():
            if digest(directory / name) != expected:
                raise ValueError(f"Corpus hash mismatch: {name}")

    with (reference_dir / "results.jsonl").open(encoding="utf-8") as source:
        refs = {(row["api"], row["position_id"]): row for row in map(json.loads, source)}
    cases = {}
    for api, file, input_field in APIS:
        positions = json.loads((DATA / f"{file}.json").read_bytes())["positions"]
        if len(positions) != 100 or len({row["id"] for row in positions}) != 100:
            raise ValueError(f"Expected 100 distinct {api} cases")
        cases[api] = []
        for row in positions:
            ref = refs[api, row["id"]]
            inputs = row[input_field]
            if ref["inputs"] != inputs or ref["source"] != row["source"]:
                raise ValueError(f"Rollout reference context mismatch: {row['id']}")
            cases[api].append((row, inputs, ref["analytics"]))
    return cases, digest(manifest_path)


def record_errors(errors, api, prediction, reference):
    for field in EQUITY_FIELDS[api]:
        errors[field].append(abs(prediction[field] - reference[field]))
    for field, value in reference["probs"].items():
        errors[f"p.{field}"].append(abs(prediction["probs"][field] - value))


def summarize(api, results, references):
    timings = sorted(row["elapsed_ns"] / 1e6 for row in results)
    errors = defaultdict(list)
    decisions = Counter()
    predicted_depths = Counter()
    for row, reference in zip(results, references, strict=True):
        result = row["analytics"]
        if api == "checker_play":
            candidates = {tuple(move["board"]): move for move in result["moves"]}
            targets = {tuple(move["board"]): move for move in reference["moves"]}
            if len(candidates) != len(result["moves"]) or candidates.keys() != targets.keys():
                raise ValueError(f"Checker endpoint mismatch: {row['position_id']}")
            best = tuple(result["moves"][0]["board"])
            rolled = {board: move for board, move in targets.items() if move["eval_level"] == "Rollout"}
            decisions["matches_reference_top_board"] += best == tuple(reference["moves"][0]["board"])
            decisions["best_board_has_rollout"] += best in rolled
            for board, target in rolled.items():
                candidate = candidates[board]
                predicted_depths[candidate["eval_level"]] += 1
                record_errors(errors, api, candidate, target)
        else:
            record_errors(errors, api, result, reference)
            if api == "cube_action":
                decisions["matches_reference_cube_action"] += result["optimal_action"] == reference["optimal_action"]
    return {
        "calls": len(timings),
        "total_ms": sum(timings),
        "median_ms": median(timings),
        "p95_ms": timings[math.ceil(0.95 * len(timings)) - 1],
        "rollout_error": {field: {"n": len(values), "mae": mean(values)} for field, values in errors.items()},
        "decisions": dict(decisions),
        "checker_depths_on_rollout_endpoints": dict(predicted_depths),
    }


def run_api(analyzer, level, api, cases):
    call = getattr(analyzer, api)
    for index, (row, inputs, _) in enumerate(cases, 1):
        seed = None
        if level.startswith("truncated"):
            key = "truncated-benchmark-v1:" + row["source"]["key"]
            seed = int.from_bytes(sha256(key.encode()).digest()[:4], "big")
            analyzer.set_seed(seed)
        start = perf_counter_ns()
        result = call(**inputs)
        elapsed = perf_counter_ns() - start
        yield {
            "level": level,
            "api": api,
            "position_id": row["id"],
            "seed": seed,
            "elapsed_ns": elapsed,
            "analytics": asdict(result),
        }
        if index % 25 == 0:
            print(f"{level} {api}: {index}/{len(cases)}", flush=True)


def sysctl(name):
    result = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=LEVELS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--model", default=PRODUCTION_MODEL)
    parser.add_argument("--output", type=Path, required=True, help="New directory for the report and compressed results")
    args = parser.parse_args()
    if args.threads < 1 or len(args.levels) != len(set(args.levels)):
        parser.error("Specify at least one thread and distinct evaluation levels")

    corpus, manifest_sha = load_corpus()
    weights = WeightConfig.from_model(args.model)
    bearoff = bearoff_db_path()
    if bearoff is None:
        raise RuntimeError("The one-sided bearoff database is required")
    started = datetime.now(timezone.utc)
    native = Path(bgbot_cpp.__file__).resolve()
    report = {
        "schema_version": 1,
        "started_utc": started.isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "engine_tracked_diff": bool(subprocess.check_output(
            ["git", "diff", "HEAD", "--name-only", "--", "cpp", "python", "models", "CMakeLists.txt"],
            cwd=ROOT, text=True,
        ).strip()),
        "native": {"name": native.name, "sha256": digest(native)},
        "platform": platform.platform(),
        "python": platform.python_version(),
        "chip": sysctl("machdep.cpu.brand_string"),
        "mac_model": sysctl("hw.model"),
        "mac_build": sysctl("kern.osversion"),
        "logical_cpus": os.cpu_count(),
        "load_avg_start": os.getloadavg(),
        "environment": {"VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS")},
        "threads_requested": args.threads,
        "model": args.model,
        "weight_slots": [
            {"name": Path(path).name, "hidden_size": size, "sha256": digest(path)}
            for path, size in zip(weights.weight_paths_list, weights.hidden_sizes_list, strict=True)
        ],
        "bearoff_sha256": digest(bearoff),
        "reference_manifest_sha256": manifest_sha,
        "levels": args.levels,
        "results": {},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "results.jsonl.gz"
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=6) as stream:
        for level in args.levels:
            analyzer = BgBotAnalyzer(
                weights=weights, bearoff_db=bearoff, eval_level=level,
                cubeful=True, parallel_threads=args.threads,
            )
            report["results"][level] = {}
            for api, cases in corpus.items():
                results = list(run_api(analyzer, level, api, cases))
                references = [ref for _, _, ref in cases]
                report["results"][level][api] = summarize(api, results, references)
                for row in results:
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
    report["load_avg_end"] = os.getloadavg()
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["results_sha256"] = digest(output)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved baseline in {args.output}")


if __name__ == "__main__":
    main()
