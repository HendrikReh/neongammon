#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Fit and validate difficulty profiles in five-point matches.

Run with PYTHONPATH=build:python from the repository root. `fit` uses
the checked-in match benchmark. `validate` plays new seeded matches and
scores a uniform sample of turns against Stage 11 at 3-ply.
"""

from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from gzip import open as open_gzip
import json
from pathlib import Path
from random import Random
from statistics import quantiles
import sys

from neongammon import (
    BgBotAnalyzer, Difficulty, DifficultyPolicy, WeightConfig,
    can_double_match, possible_moves, simulate_match,
)
from neongammon.difficulty import _PROFILES, _choice_weights, _cube_error_probability, _cube_gaps
from neongammon.xg_compare import _is_trivial_cube, _is_trivial_take_pass


_TARGETS = {Difficulty.BEGINNER: 25, Difficulty.CASUAL: 15, Difficulty.INTERMEDIATE: 8}
_TEMPERATURES = (0, .004, .006, .008, .010, .012, .015, .018, .022, .027, .034, .044,
                 .056, .072, .095, .125, .165, .220, .300, .400, .500)
_KINDS = ("checker", "double", "take")
_DATASET = Path(__file__).resolve().parents[1] / "data/match_benchmark/5pt/benchmark.json.gz"


def cube_loss(kind, chosen, equities):
    nd, dt, dp = equities
    double = min(dt, dp)
    if kind == "double":
        return max(nd, double) - (double if chosen else nd)
    return (dt if chosen else dp) - double


def _context(turn):
    return {key: turn[key] for key in ("cube_value", "cube_owner", "away1", "away2", "is_crawford")}


def _analyzer(level):
    return BgBotAnalyzer(weights=WeightConfig.from_model("stage11"), eval_level=level,
                         parallel_threads=1)


def _fit():
    with open_gzip(_DATASET, "rt") as source:
        entries = json.load(source)["decisions"]
    analyzer = _analyzer("1ply")
    checker = []
    cube = []
    for entry in entries:
        board = entry["board"]
        ctx = _context(entry)
        if entry["kind"] == "checker":
            moves = analyzer.checker_play(board, *entry["dice"], **ctx).moves
            ref = {tuple(move["board"]): move["equity"] for move in entry["moves"]}
            best = entry["moves"][0]["equity"]
            checker.append((
                [moves[0].equity - move.equity for move in moves],
                [max(0, best - ref[tuple(move.board)]) for move in moves],
            ))
        else:
            if not can_double_match(ctx["away1"], ctx["away2"], ctx["cube_value"],
                                    ctx["cube_owner"], ctx["is_crawford"]):
                continue
            assessment = analyzer.cube_decision(board, **ctx)
            ref = tuple(entry[f"equity_{key}"] for key in ("nd", "dt", "dp"))
            double_gap, take_gap = _cube_gaps(assessment)
            if entry["has_double"]:
                choose = assessment.should_double
                cube.append((double_gap, cube_loss("double", choose, ref),
                             cube_loss("double", not choose, ref)))
            if entry["has_take"]:
                choose = assessment.should_take
                cube.append((take_gap, cube_loss("take", choose, ref),
                             cube_loss("take", not choose, ref)))
    results = {}
    for level, target in _TARGETS.items():
        profile = _PROFILES[level]
        cube_errors = 0
        for gap, best, other in cube:
            probability = _cube_error_probability(gap, profile)
            cube_errors += (1 - probability) * best + probability * other
        temperatures = sorted(set((*_TEMPERATURES, profile.checker_temperature, float("inf"))))
        curve = {}
        for temperature in temperatures:
            errors = cube_errors
            for losses, ref in checker:
                weights = _choice_weights(losses, profile.checker_limit, temperature)
                errors += sum(weight * loss for weight, loss in zip(weights, ref)) / sum(weights)
            curve[temperature] = 500 * errors / (len(checker) + len(cube))
        selected = min(temperatures[:-1], key=lambda temperature: abs(curve[temperature] - target))
        results[level.value] = {
            "target": target,
            "candidate_temperature": selected,
            "candidate_pr": curve[selected],
            "shipped_temperature": profile.checker_temperature,
            "shipped_pr": curve[profile.checker_temperature],
            "uniform_pr": curve[float("inf")],
        }
    return {"calibration_decisions": len(checker) + len(cube), "levels": results}


def _score_turn(turn, side, reference, errors):
    ctx = _context(turn)
    on_roll = turn["player"] == side
    action = turn["cube_action"]
    offered = action in ("double/take", "double/pass")
    if action and ((on_roll and can_double_match(
        ctx["away1"], ctx["away2"], ctx["cube_value"], ctx["cube_owner"], ctx["is_crawford"],
    )) or (offered and not on_roll)):
        cube = reference.cube_decision(turn["board"], **ctx)
        equities = cube.equity_nd, cube.equity_dt, cube.equity_dp
        analysis = dict(zip(("equity_nd", "equity_dt", "equity_dp"), equities))
        if on_roll:
            loss = cube_loss("double", offered, equities)
            # An actual offer counts even in a position with an obvious cube action.
            if offered or loss >= 0.001 or not _is_trivial_cube(analysis):
                errors["double"].append(loss)
        elif not _is_trivial_take_pass(analysis):
            errors["take"].append(cube_loss("take", action == "double/take", equities))

    if not on_roll or turn["dice"] is None:
        return
    board, dice, played = turn["board"], turn["dice"], turn["post_board"]
    if len(possible_moves(board, *dice)) <= 1:
        return
    ctx.update(cube_value=turn["checker_cube_value"], cube_owner=turn["checker_cube_owner"])
    moves = reference.checker_play(board, *dice, force_boards=[played], **ctx).moves
    if moves[0].equity - moves[-1].equity >= 0.001:
        played_equity = next(move.equity for move in moves if move.board == played)
        errors["checker"].append(moves[0].equity - played_equity)


_REFERENCE = None


def _init_worker(reference_level):
    global _REFERENCE
    _REFERENCE = _analyzer(reference_level)


def _score_match(job):
    level, opponent, seed, sample = job
    side = seed % 2 + 1
    policy = DifficultyPolicy(level, seed=2 * seed)
    rival = DifficultyPolicy(opponent, seed=2 * seed + 1)
    players = (policy, rival) if side == 1 else (rival, policy)
    match = simulate_match(match_length=5, seed=seed, players=players)
    sampler = Random(seed + 10_000_000)
    errors = {kind: [] for kind in _KINDS}
    for game in match["games"]:
        for turn in game["turns"]:
            if sampler.random() < sample:
                _score_turn(turn, side, _REFERENCE, errors)
    return {"level": level, "seed": seed, "won": match["winner"] == side, **errors}


def summarize(rows):
    totals = [
        (sum(sum(row[kind]) for kind in _KINDS), sum(len(row[kind]) for kind in _KINDS))
        for row in rows
    ]
    errors, decisions = map(sum, zip(*totals))
    rng = Random(329)
    draws = []
    for _ in range(4000):
        loss, count = map(sum, zip(*rng.choices(totals, k=len(totals))))
        if count:
            draws.append(500 * loss / count)
    limits = quantiles(draws, n=40, method="inclusive") if len(draws) > 1 else []
    by_kind = {}
    for kind in _KINDS:
        losses = [loss for row in rows for loss in row[kind]]
        by_kind[kind] = {
            "decisions": len(losses),
            "pr": 500 * sum(losses) / len(losses) if losses else None,
            "blunders": sum(loss > 0.08 for loss in losses),
            "max_loss": max(losses, default=None),
        }
    return {
        "matches": len(rows),
        "won": sum(row["won"] for row in rows),
        "decisions": decisions,
        "pr": 500 * errors / decisions if decisions else None,
        "ci95": [limits[0], limits[-1]] if limits else None,
        "blunders": sum(result["blunders"] for result in by_kind.values()),
        "by_kind": by_kind,
    }


def _validate(args):
    jobs = [
        (level, args.opponent, args.seed + index, args.sample)
        for index in range(args.matches)
        for level in args.levels
    ]
    rows = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.reference,)) as pool:
        for index, row in enumerate(pool.map(_score_match, jobs), start=1):
            rows[row["level"]].append(row)
            if index % 32 == 0:
                print(f"{index}/{len(jobs)} matches scored", file=sys.stderr, flush=True)
    return {
        "match_length": 5, "first_seed": args.seed, "opponent": args.opponent,
        "sample": args.sample, "reference": args.reference,
        "levels": {level: summarize(results) for level, results in rows.items()},
    }


def main():
    levels = [level.value for level in Difficulty]
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fit")
    validate = commands.add_parser("validate")
    validate.add_argument("--matches", type=int, default=120, help="matches per level")
    validate.add_argument("--seed", type=int, default=700_000, help="first dice seed")
    validate.add_argument("--workers", type=int, default=4)
    validate.add_argument("--sample", type=float, default=0.2, help="fraction of turns to score")
    validate.add_argument("--levels", nargs="+", choices=levels, default=levels)
    validate.add_argument("--opponent", choices=levels, default=Difficulty.FULL_STRENGTH.value)
    validate.add_argument("--reference", choices=("2ply", "3ply"), default="3ply")
    args = parser.parse_args()
    if args.command == "fit":
        result = _fit()
    else:
        if args.matches < 1 or args.workers < 1 or not 0 < args.sample <= 1:
            parser.error("matches and workers must be positive and sample must be in (0, 1]")
        result = _validate(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
