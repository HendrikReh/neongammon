# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Paskogammon "PR against a rollout" benchmark.

The Paskogammon analog of :mod:`benchmark_money`. Every game starts from
``PASKO_STARTING_BOARD`` (Player 1 on roll first, uniform dice including doubles),
and reuses the shared adaptive-precision pipeline and money-game scoring.

Usage::

    python scripts/archive/pasko/benchmark_pasko.py build --n-games 10 --n-threads 16 --write-txt
    python scripts/archive/pasko/benchmark_pasko.py score --level 3ply --n-threads 16
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import benchmark_common as bc  # noqa: E402
import benchmark_money as bm  # noqa: E402
from benchmark_common import (  # noqa: E402, F401
    BLUNDER_THRESHOLD,
    GAME_PLANS,
    PR_MULTIPLIER,
    ROLLOUT_GAP,
    ROLLOUT_MAX_BATCHES,
    ROLLOUT_N_TRIALS,
    ROLLOUT_TARGET_SE,
    THREE_T_GAP,
    TIER_3P,
    TIER_3T,
    TIER_ROLLOUT,
    TRIVIAL_SPREAD,
    CheckerCandidate,
    CubeAssessment,
    _BUILD_DIR,
    _NEONGAMMON_PYTHON,
    _PROJECT_ROOT,
    _fmt_dur,
    _log,
)

BenchmarkBot = bm.BenchmarkBot
NeongammonBot = bm.NeongammonBot
make_decision_key = bm.make_decision_key

# ---------------------------------------------------------------------------
# Paskogammon opening position & output locations
# ---------------------------------------------------------------------------

PASKO_STARTING_BOARD: list[int] = [
    0, -2, -2, 0, 0, -1, 5, -1, 3, 0, 0, 0, -2,   # [0]=opp bar, points 1-12
    5, 0, 0, 0, -2, -1, -3, -1, 0, 0, 0, 2,       # points 13-24
    0,                                            # [25]=player bar
]

_DATA_DIR = _PROJECT_ROOT / "data" / "pasko_money_benchmark"
_BUILD_SUBDIR = _DATA_DIR / "build"
_STAGE1_DIR = _BUILD_SUBDIR / "stage1"
_STAGE2_FILE = _BUILD_SUBDIR / "stage2_3t.jsonl"
_STAGE3_FILE = _BUILD_SUBDIR / "stage3_rollout.jsonl"
_XG_DIR = _DATA_DIR / "xg"
_SCORES_DIR = _DATA_DIR / "scores"
DEFAULT_DATASET = _DATA_DIR / "benchmark.json"
_THREET_JOBS_FILE = _DATA_DIR / "threet_jobs.jsonl"
_ROLLOUT_JOBS_FILE = _DATA_DIR / "rollout_jobs.jsonl"


# ===========================================================================
# Pass 1 - Paskogammon self-play simulation
# ===========================================================================


def _simulate_and_capture(seed: int, parallel_threads: int) -> tuple[list[dict], dict]:
    """Play one Neongammon-3P vs Neongammon-3P Paskogammon game and capture 3P analytics."""
    analyzer = bc._get_analyzer("3ply", parallel_threads)
    rng = random.Random(seed)
    decisions, move_history, winner, win_type, cube_at_end = bc.simulate_game_and_capture(
        analyzer, rng, starting_board=PASKO_STARTING_BOARD, seed=seed,
    )
    xg_record = bc.build_unlimited_xg_record(move_history, winner, win_type, cube_at_end)
    return decisions, xg_record


def _build_stage1_game(seed: int, write_txt: bool, parallel_threads: int) -> dict:
    """Worker: play one game, persist its decisions (and optional XG .txt), atomically."""
    decisions, xg_record = _simulate_and_capture(seed, parallel_threads)
    bc.atomic_write_text(
        _STAGE1_DIR / f"seed_{seed}.json",
        json.dumps({"seed": seed, "variant": "paskogammon", "decisions": decisions},
                   separators=(",", ":")),
    )
    if write_txt:
        from neongammon.text_export import export_history_to_txt

        (_XG_DIR / f"seed_{seed}.txt").write_bytes(export_history_to_txt(xg_record))

    n_checker = sum(1 for d in decisions if d["kind"] == "checker")
    return {"seed": seed, "n_checker": n_checker, "n_cube": len(decisions) - n_checker}


def _run_pass1(n_games: int, seed: int, write_txt: bool, workers: int) -> None:
    _STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    if write_txt:
        _XG_DIR.mkdir(parents=True, exist_ok=True)
    bc.run_pass1_pool(
        [seed + i for i in range(n_games)],
        is_done_fn=lambda s: (_STAGE1_DIR / f"seed_{s}.json").exists(),
        worker_fn=_build_stage1_game,
        worker_args_fn=lambda s, threads: (s, write_txt, threads),
        label="Pass 1/3 (Paskogammon 3P self-play)",
        unit_name="games",
        workers=workers,
        format_result=lambda r: (
            f"seed={r['seed']}: {r['n_checker']} checker + {r['n_cube']} cube positions"
        ),
    )


def _load_stage1_decisions() -> list[dict]:
    return bc.load_stage1_decisions(_STAGE1_DIR, "seed_*.json")


# ===========================================================================
# Job export & assembly
# ===========================================================================


def _export_threet_jobs(
    unique: dict[str, dict], min_seed: int | None = None, max_seed: int | None = None,
) -> Path:
    return bc.export_threet_jobs(
        unique, _STAGE2_FILE, _THREET_JOBS_FILE, min_seed=min_seed, max_seed=max_seed,
    )


def _export_rollout_jobs(
    unique: dict[str, dict], min_seed: int | None = None, max_seed: int | None = None,
) -> Path:
    return bc.export_rollout_jobs(
        unique, _STAGE3_FILE, _ROLLOUT_JOBS_FILE, min_seed=min_seed, max_seed=max_seed,
    )


def _assemble_dataset(unique: dict[str, dict], dataset_path: Path, n_games: int, seed: int) -> dict:
    return bc.assemble_dataset(
        unique,
        dataset_path,
        meta_prefix={
            "variant": "paskogammon",
            "starting_board": list(PASKO_STARTING_BOARD),
            "n_games": n_games,
            "seed": seed,
        },
    )


def build_benchmark_data(
    n_games: int,
    seed: int = 1,
    n_threads: int = 0,
    write_txt: bool = True,
    workers: int = 6,
    dataset_path: Path | str = DEFAULT_DATASET,
    rollout_mode: str = "compute",
    threet_mode: str = "compute",
    export_min_seed: int | None = None,
    export_max_seed: int | None = None,
    stages: tuple = ("pass1", "pass2", "pass3"),
) -> dict:
    """Build the Paskogammon benchmark data set (three adaptive-precision passes)."""
    dataset_path = Path(dataset_path)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    stage_set = set(stages)
    _log(f"=== build_benchmark_data (paskogammon): n_games={n_games} seed={seed} "
         f"n_threads={n_threads} workers={workers} write_txt={write_txt} "
         f"rollout_mode={rollout_mode} threet_mode={threet_mode} stages={sorted(stage_set)} ===")

    if "pass1" in stage_set:
        _run_pass1(n_games, seed, write_txt, workers)
    all_decisions = _load_stage1_decisions()
    if not all_decisions:
        _log("No stage-1 decisions on disk -- run the 'pass1' stage first.")
        return {}
    unique = bc._unique_by_key(all_decisions)
    _log(f"Captured {len(all_decisions)} decisions ({len(unique)} unique positions).")

    unique = bc.run_passes_2_and_3(
        unique,
        stage2_file=_STAGE2_FILE,
        stage3_file=_STAGE3_FILE,
        threet_jobs_file=_THREET_JOBS_FILE,
        rollout_jobs_file=_ROLLOUT_JOBS_FILE,
        stages=stage_set,
        threet_mode=threet_mode,
        rollout_mode=rollout_mode,
        n_threads=n_threads,
        export_min_seed=export_min_seed,
        export_max_seed=export_max_seed,
        reeval_3t_fn=bm._reeval_decision,
        reeval_rollout_fn=bm._reeval_decision_rollout,
        match_mode=False,
    )
    return _assemble_dataset(unique, dataset_path, n_games, seed)


# ===========================================================================
# Scoring
# ===========================================================================


def benchmark_pr(
    bot: BenchmarkBot,
    dataset_path: Path | str | None = None,
    n_threads: int = 1,
    label: str = "bot",
    cache_path: Path | str | None = None,
    progress: bool = True,
    max_seed: int | None = None,
) -> dict:
    """Score ``bot`` against the Paskogammon benchmark; return the PR breakdown."""
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET
    if cache_path is None:
        _SCORES_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _SCORES_DIR / f"{label}.jsonl"
    return bm.benchmark_pr(
        bot, dataset_path=dataset_path, n_threads=n_threads, label=label,
        cache_path=cache_path, progress=progress, max_seed=max_seed,
    )


# ===========================================================================
# CLI
# ===========================================================================


def _main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build the Paskogammon benchmark data set")
    p_build.add_argument("--n-games", type=int, default=100, help="Games to simulate")
    bc.add_common_build_args(p_build, unit_name="game", default_dataset=DEFAULT_DATASET)

    p_score = sub.add_parser("score", help="Score the Neongammon engine against the data set")
    bc.add_common_score_args(p_score, unit_name="games", default_dataset=DEFAULT_DATASET)

    args = parser.parse_args(argv)
    if args.command == "build":
        build_benchmark_data(
            n_games=args.n_games, seed=args.seed, n_threads=args.n_threads,
            write_txt=args.write_txt, workers=args.workers, dataset_path=args.dataset,
            rollout_mode=args.rollout_mode, threet_mode=args.threet_mode,
            export_min_seed=args.min_seed, export_max_seed=args.max_seed,
            stages=tuple(s.strip() for s in args.stages.split(",") if s.strip()),
        )
    elif args.command == "score":
        bot = NeongammonBot(eval_level=args.level, model=args.model, parallel_threads=1)
        label = args.label or f"sage_{args.level}" + (f"_{args.model}" if args.model else "")
        result = benchmark_pr(bot, dataset_path=args.dataset, n_threads=args.n_threads,
                              label=label, max_seed=args.max_seed)
        bm._print_report(result)


if __name__ == "__main__":
    _main()
