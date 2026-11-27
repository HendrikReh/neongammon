#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Opt-in native 4/5-ply pilot over fixed money and five-point rollout rows."""

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = {
    "money": (
        "data/money_benchmark/benchmark.json.gz",
        {
            "checker": {
                "centered_contact": "00cdf38419da56d483aff079ec83f0aa",
                "owned_contact": "011c94d705bfa099d8aa818cb752fa80",
                "race": "01cf40902323205de166128b68cb0229",
                "bearoff": "00832bf370f848f8330bf4e0a7161cad",
            },
            "cube": {
                "centered_contact": "00fd1f62930bb92d82816bb348ca244c",
                "owned_contact": "03001c7f3b06a43777327f480f400dc1",
                "race": "06db196507f9ed994ea3e126ef1052fa",
                "bearoff": "06e8e94877ed834f1692cb9e211760eb",
            },
        },
    ),
    "five_point": (
        "data/match_benchmark/5pt/benchmark.json.gz",
        {
            "checker": {
                "centered_contact_5_away": "00adf9a495c651494db575b75f23d36a",
                "owned_contact_asymmetric": "01ccdc4e4b5e8cf5c9ced01ad687216a",
                "crawford_contact": "0016236c3af0e570651d2840217c1c00",
                "crawford_bearoff": "070dc22f0def61518b0a9c23824106eb",
            },
            "cube": {
                "centered_contact_5_away": "1ab2eafa0a2caee7a9c32029ce19f7d6",
                "owned_contact_asymmetric": "178336b63210ac9b90c69606c33b99cf",
                "centered_contact_2_away_1_away": "05df732a5eedecba996ceb449b550c36",
                "race": "1c6cae33a12d0a4dfc2a46644d86d2ec",
            },
        },
    ),
}


def sample_rows():
    for context, (path, kinds) in SAMPLES.items():
        sample_keys = {key for cases in kinds.values() for key in cases.values()}
        with gzip.open(ROOT / path, "rt", encoding="utf-8") as source:
            rows = {row["key"]: row for row in json.load(source)["decisions"]
                    if row["key"] in sample_keys}
        for kind, cases in kinds.items():
            for name, key in cases.items():
                row = rows[key]
                if row["kind"] != kind or row["tier"] != "rollout":
                    raise ValueError(f"Unexpected reference tier or decision kind for {key}")
                yield context, kind, name, row


def timed(function, *args, **kwargs):
    start = perf_counter()
    result = function(*args, **kwargs)
    return result, perf_counter() - start


def checker_case(native, analyzer, row, context, label):
    board, dice = row["board"], row["dice"]
    legal = {tuple(move) for move in native.possible_moves(board, *dice)}
    reference = {tuple(move["board"]): move for move in row["moves"]}
    ranked = sorted((key for key in reference if key in legal),
                    key=lambda key: (reference[key]["equity"], key))
    forced_boards = ranked[:2]
    if len(forced_boards) < 2:
        raise ValueError(f"Need two legal reference boards to force for {row['key']}")

    moves, elapsed = timed(analyzer.checker_play_analytics, board, *dice,
                           force_boards=[list(move) for move in forced_boards], **context)
    returned = {tuple(move["board"]): (rank, move)
                for rank, move in enumerate(moves, 1)}
    depths = Counter(move["eval_level"] for move in moves)
    forced = [
        {"board": move, "rank": returned[move][0], "depth": returned[move][1]["eval_level"]}
        if move in returned else {"board": move, "rank": None, "depth": None}
        for move in forced_boards
    ]
    best = moves[0] if moves else None
    best_reference = reference.get(tuple(best["board"])) if best else None
    top_reference = row["moves"][0]
    return {
        "seconds": elapsed,
        "legal_endpoints": len(legal),
        "returned_endpoints": len(moves),
        "exact_legal_coverage": legal == returned.keys() and len(moves) == len(legal),
        "depth_counts": dict(depths),
        "forced": forced,
        "best_board": best["board"] if best else None,
        "best_equity": best["equity"] if best else None,
        "best_reference_depth": best_reference["eval_level"] if best_reference else None,
        "rollout_reference_loss": (
            max(0.0, top_reference["equity"] - best_reference["equity"])
            if best_reference and top_reference["eval_level"] == best_reference["eval_level"] == "Rollout"
            else None
        ),
        "ok": (bool(legal) and legal == returned.keys() and len(moves) == len(legal)
               and depths == {label: len(legal)}
               and all(move["depth"] == label for move in forced)),
    }


def cube_case(inner, row, context, label, decision_first):
    order = (True, False) if decision_first else (False, True)
    results = {
        decision_only: timed(inner.cube_action_analytics, row["board"],
                             decision_only=decision_only, **context)
        for decision_only in order
    }
    rich, rich_seconds = results[False]
    decision, decision_seconds = results[True]
    differences = {
        key: {"rich": rich.get(key), "decision_only": value}
        for key, value in decision.items() if key not in rich or rich[key] != value
    }
    return {
        "rich_seconds": rich_seconds,
        "decision_seconds": decision_seconds,
        "first_call": "decision_only" if decision_first else "rich",
        "rich_depth": rich["eval_level"],
        "decision_depth": decision["eval_level"],
        "shared_fields": decision,
        "differences": differences,
        "reference_action": {"should_double": row["should_double_ref"],
                             "should_take": row["should_take_ref"]},
        "ok": not differences and rich["eval_level"] == decision["eval_level"] == label,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4, help="native threads per analysis")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")

    import bgbot_cpp as native
    from neongammon import BatchModels
    from neongammon.analyzer import _CubefulAnalyzer, _MultiPlyAnalyzer

    models = BatchModels(bearoff_db=True)
    if models.bearoff_db is None:
        raise RuntimeError("The default bearoff database did not load")
    inners = {ply: _MultiPlyAnalyzer(models, n_plies=ply, parallel_threads=args.threads)
              for ply in (4, 5)}
    for inner in inners.values():
        native.multipy_set_bearoff_db(inner._strategy_nply, models.bearoff_db)
    checkers = {ply: _CubefulAnalyzer(inner) for ply, inner in inners.items()}
    report = {"python": platform.python_version(), "system": platform.platform(),
              "native_module": native.__file__, "threads": args.threads,
              "blas_threads": os.environ.get("VECLIB_MAXIMUM_THREADS"),
              "load_average_before": os.getloadavg(), "rows": []}

    for index, (context, kind, name, row) in enumerate(sample_rows()):
        money = context == "money"
        cube_context = {
            "cube_value": row["cube_value"], "cube_owner": row["cube_owner"],
            "away1": row.get("away1", 0), "away2": row.get("away2", 0),
            "is_crawford": row.get("is_crawford", False), "jacoby": money, "beaver": money,
        }
        order = (4, 5) if index % 2 == 0 else (5, 4)
        results = {}
        for ply in order:
            label = f"{ply}-ply"
            if kind == "checker":
                results[ply] = checker_case(native, checkers[ply], row, cube_context, label)
            else:
                results[ply] = cube_case(inners[ply], row, cube_context, label,
                                         decision_first=(index + ply) % 2 == 0)
        report["rows"].append({"context": context, "kind": kind, "case": name,
                               "key": row["key"], "cube": cube_context, "depths": results})
        print(f"{context} {kind} {name}", file=sys.stderr, flush=True)

    report["median_seconds"] = {
        ply: {
            field: median(row["depths"][ply][field] for row in report["rows"]
                          if row["kind"] == kind)
            for kind, field in (("checker", "seconds"), ("cube", "rich_seconds"),
                                ("cube", "decision_seconds"))
        }
        for ply in (4, 5)
    }
    report["load_average_after"] = os.getloadavg()
    ok = all(result["ok"] for row in report["rows"] for result in row["depths"].values())
    report["ok"] = ok
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
