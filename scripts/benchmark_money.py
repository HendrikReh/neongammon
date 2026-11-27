# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Money-game "PR against a rollout" benchmark.

Builds a production-quality reference data set of backgammon money-game
decisions (checker plays and cube actions) with adaptive rollout-grade analytics
(3-ply -> 3T -> full rollout) and scores any :class:`BenchmarkBot` against it.

Two public entry points:

  * :func:`build_benchmark_data` - simulate Neongammon-3P vs Neongammon-3P games
    and emit the reference data set (three resumable adaptive-precision passes).
  * :func:`benchmark_pr` - score a :class:`BenchmarkBot` against the data set and
    return total / checker / cube PR plus a per-game-plan breakdown.

Usage::

    python scripts/benchmark_money.py build --n-games 200 --n-threads 16 --write-txt
    python scripts/benchmark_money.py score --level 3ply --n-threads 16
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import benchmark_common as bc  # noqa: E402
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
    _MAX_TURNS,
    _NEONGAMMON_PYTHON,
    _PROGRESS_EVERY_S,
    _PROJECT_ROOT,
    _ROLLOUT_BASE_SEED,
    _SEED_STEP,
    _SLOW_ITEM_S,
    _TIER_RANK,
    _SimState,
    _aggregate,
    _apply_refined,
    _checker_gap,
    _cube_fields_from_result,
    _cube_gap,
    _flip_sim,
    _fmt_dur,
    _gap_for,
    _get_analyzer,
    _is_trivial_cube,
    _load_jsonl_by_key,
    _log,
    _make_3t_analyzer,
    _make_rollout_analyzer,
    _missing_tier,
    _move_list_from_result,
    _print_report,
    _read_dataset,
    _run_refinement_pass,
    _scored,
    _seed_in_range,
    _take_gap,
    _unique_by_key,
)

# ---------------------------------------------------------------------------
# Output locations (all under neongammon/data/money_benchmark/)
# ---------------------------------------------------------------------------

_DATA_DIR = _PROJECT_ROOT / "data" / "money_benchmark"
_BUILD_SUBDIR = _DATA_DIR / "build"
_STAGE1_DIR = _BUILD_SUBDIR / "stage1"
_STAGE2_FILE = _BUILD_SUBDIR / "stage2_3t.jsonl"
_STAGE3_FILE = _BUILD_SUBDIR / "stage3_rollout.jsonl"
_XG_DIR = _DATA_DIR / "xg"
_SCORES_DIR = _DATA_DIR / "scores"
DEFAULT_DATASET = _DATA_DIR / "benchmark.json"
_ROLLOUT_JOBS_FILE = _DATA_DIR / "rollout_jobs.jsonl"
_THREET_JOBS_FILE = _DATA_DIR / "threet_jobs.jsonl"


# ===========================================================================
# Bot interface
# ===========================================================================


class BenchmarkBot(ABC):
    """Interface a bot must implement to be scored by :func:`benchmark_pr`."""

    @abstractmethod
    def checker_play(
        self, board: list[int], die1: int, die2: int, cube_value: int, cube_owner: str,
    ) -> list[CheckerCandidate]:
        """Return candidate plays ranked best-first (the bot's choice is index 0)."""

    @abstractmethod
    def cube_action(self, board: list[int], cube_value: int, cube_owner: str) -> CubeAssessment:
        """Return the bot's cube assessment for the player on roll."""


class NeongammonBot(BenchmarkBot):
    """The Neongammon production engine as a :class:`BenchmarkBot`."""

    def __init__(
        self,
        eval_level: str = "3ply",
        weights: Any = None,
        model: Optional[str] = None,
        parallel_threads: int = 0,
    ):
        from neongammon import BgBotAnalyzer
        from neongammon.weights import WeightConfig

        if weights is None and model is not None:
            weights = WeightConfig.from_model(model)
        self.eval_level = eval_level
        self._analyzer = BgBotAnalyzer(
            weights=weights, eval_level=eval_level, cubeful=True,
            parallel_threads=parallel_threads,
        )

    def checker_play(
        self, board: list[int], die1: int, die2: int, cube_value: int, cube_owner: str,
    ) -> list[CheckerCandidate]:
        res = self._analyzer.checker_play(
            board, die1, die2, cube_value=cube_value, cube_owner=cube_owner,
            jacoby=True, beaver=True,
        )
        return [CheckerCandidate(board=list(m.board), equity=m.equity) for m in res.moves]

    def cube_action(self, board: list[int], cube_value: int, cube_owner: str) -> CubeAssessment:
        c = self._analyzer.cube_action(
            board, cube_value=cube_value, cube_owner=cube_owner, jacoby=True, beaver=True,
        )
        return CubeAssessment(should_double=c.should_double, should_take=c.should_take)


# ===========================================================================
# Decision keys & Pass 1 simulation
# ===========================================================================


def make_decision_key(
    kind: str, board: list[int], dice: Optional[tuple[int, int] | list[int]],
    cube_value: int, cube_owner: str,
) -> str:
    return bc.make_decision_key(kind, board, dice, cube_value, cube_owner)


def _simulate_and_capture(seed: int, parallel_threads: int) -> tuple[list[dict], dict]:
    """Play one Neongammon-3P vs Neongammon-3P money game and capture 3P analytics."""
    from neongammon import STARTING_BOARD

    analyzer = _get_analyzer("3ply", parallel_threads)
    rng = random.Random(seed)
    decisions, move_history, winner, win_type, cube_at_end = bc.simulate_game_and_capture(
        analyzer, rng, starting_board=STARTING_BOARD, seed=seed,
    )
    xg_record = bc.build_unlimited_xg_record(move_history, winner, win_type, cube_at_end)
    return decisions, xg_record


def _build_stage1_game(seed: int, write_txt: bool, parallel_threads: int) -> dict:
    """Worker: play one game, persist its decisions (and optional XG .txt), atomically."""
    decisions, xg_record = _simulate_and_capture(seed, parallel_threads)
    bc.atomic_write_text(
        _STAGE1_DIR / f"seed_{seed}.json",
        json.dumps({"seed": seed, "decisions": decisions}, separators=(",", ":")),
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
        label="Pass 1/3 (3P self-play)",
        unit_name="games",
        workers=workers,
        format_result=lambda r: (
            f"seed={r['seed']}: {r['n_checker']} checker + {r['n_cube']} cube positions"
        ),
    )


def _load_stage1_decisions() -> list[dict]:
    return bc.load_stage1_decisions(_STAGE1_DIR, "seed_*.json")


# ===========================================================================
# Passes 2 & 3, job export & assembly
# ===========================================================================


def _reeval_decision(analyzer, decision: dict, progress_callback=None) -> dict:
    return bc.reeval_decision(analyzer, decision, progress_callback=progress_callback)


def _reeval_decision_rollout(analyzer, decision: dict, progress_callback=None) -> dict:
    return bc.reeval_decision_rollout(analyzer, decision, progress_callback=progress_callback)


def _export_rollout_jobs(
    unique: dict[str, dict], min_seed: int | None = None, max_seed: int | None = None,
) -> Path:
    return bc.export_rollout_jobs(
        unique, _STAGE3_FILE, _ROLLOUT_JOBS_FILE, min_seed=min_seed, max_seed=max_seed,
    )


def _export_threet_jobs(
    unique: dict[str, dict], min_seed: int | None = None, max_seed: int | None = None,
) -> Path:
    return bc.export_threet_jobs(
        unique, _STAGE2_FILE, _THREET_JOBS_FILE, min_seed=min_seed, max_seed=max_seed,
    )


def _assemble_dataset(unique: dict[str, dict], dataset_path: Path, n_games: int, seed: int) -> dict:
    return bc.assemble_dataset(unique, dataset_path, meta_prefix={"n_games": n_games, "seed": seed})


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
    """Build the money-game benchmark data set (three adaptive-precision passes)."""
    dataset_path = Path(dataset_path)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    stage_set = set(stages)
    _log(f"=== build_benchmark_data: n_games={n_games} seed={seed} "
         f"n_threads={n_threads} workers={workers} write_txt={write_txt} "
         f"rollout_mode={rollout_mode} stages={sorted(stage_set)} ===")

    if "pass1" in stage_set:
        _run_pass1(n_games, seed, write_txt, workers)
    all_decisions = _load_stage1_decisions()
    if not all_decisions:
        _log("No stage-1 decisions on disk -- run the 'pass1' stage first.")
        return {}
    unique = _unique_by_key(all_decisions)
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
        reeval_3t_fn=_reeval_decision,
        reeval_rollout_fn=_reeval_decision_rollout,
        match_mode=False,
    )
    return _assemble_dataset(unique, dataset_path, n_games, seed)


# ===========================================================================
# Scoring
# ===========================================================================


def _score_checker(bot: BenchmarkBot, entry: dict) -> Optional[dict]:
    return bc.score_checker(bot, entry, match_mode=False)


def _score_cube(bot: BenchmarkBot, entry: dict) -> list[dict]:
    return bc.score_cube(bot, entry, match_mode=False)


def _score_entry(bot: BenchmarkBot, entry: dict) -> Optional[dict]:
    return bc.score_entry(bot, entry, match_mode=False)


def benchmark_pr(
    bot: BenchmarkBot,
    dataset_path: Path | str = DEFAULT_DATASET,
    n_threads: int = 1,
    label: str = "bot",
    cache_path: Path | str | None = None,
    progress: bool = True,
    max_seed: int | None = None,
) -> dict:
    """Score ``bot`` against the money benchmark data set; return the PR breakdown."""
    return bc.run_benchmark_pr(
        bot,
        dataset_path=dataset_path,
        scores_dir=_SCORES_DIR,
        n_threads=n_threads,
        label=label,
        cache_path=cache_path,
        progress=progress,
        max_seed=max_seed,
        match_mode=False,
        unit_name="games",
        score_entry_fn=_score_entry,
    )


# ===========================================================================
# CLI
# ===========================================================================


def _main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build the benchmark data set")
    p_build.add_argument("--n-games", type=int, default=200, help="Games to simulate")
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
        _print_report(result)


if __name__ == "__main__":
    _main()
