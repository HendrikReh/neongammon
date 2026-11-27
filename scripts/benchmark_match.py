# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Match-play "PR against a rollout" benchmark.

The match-play analog of :mod:`benchmark_money`. Builds a reference data set of
backgammon **match** decisions (checker plays and cube actions) with adaptive
rollout-grade analytics and match state threaded through every decision.

Usage::

    python scripts/benchmark_match.py build --match-length 5 --n-matches 130 --stages pass1 --write-txt
    python scripts/benchmark_match.py build --match-length 5 --n-matches 130 --stages pass2
    python scripts/benchmark_match.py build --match-length 5 --n-matches 130 --stages pass3
    python scripts/benchmark_match.py score --match-length 5 --level 3ply --n-threads 16
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
    _away_counts_for_mover,
    _fmt_dur,
    _log,
)

_MAX_GAMES = 200


# ===========================================================================
# Output locations (parameterized by match length)
# ===========================================================================


class _Paths:
    """All on-disk locations for an ``L``-point match benchmark."""

    def __init__(self, match_length: int):
        self.match_length = int(match_length)
        self.data_dir = _PROJECT_ROOT / "data" / "match_benchmark" / f"{self.match_length}pt"
        self.build = self.data_dir / "build"
        self.stage1 = self.build / "stage1"
        self.stage2 = self.build / "stage2_3t.jsonl"
        self.stage3 = self.build / "stage3_rollout.jsonl"
        self.xg = self.data_dir / "xg"
        self.scores = self.data_dir / "scores"
        self.dataset = self.data_dir / "benchmark.json"
        self.rollout_jobs = self.data_dir / "rollout_jobs.jsonl"
        self.threet_jobs = self.data_dir / "threet_jobs.jsonl"


# ===========================================================================
# Decision keys & bot interface (match-aware)
# ===========================================================================


def make_decision_key(
    kind: str, board: list[int], dice: Optional[tuple[int, int] | list[int]],
    cube_value: int, cube_owner: str, away1: int, away2: int, is_crawford: bool,
) -> str:
    return bc.make_decision_key(
        kind, board, dice, cube_value, cube_owner,
        match_context=(away1, away2, is_crawford),
    )


class BenchmarkBot(ABC):
    """Match-aware bot interface for :func:`benchmark_pr`."""

    @abstractmethod
    def checker_play(
        self, board: list[int], die1: int, die2: int, cube_value: int, cube_owner: str,
        away1: int, away2: int, is_crawford: bool,
    ) -> list[CheckerCandidate]:
        """Return candidate plays ranked best-first (the bot's choice is index 0)."""

    @abstractmethod
    def cube_action(
        self, board: list[int], cube_value: int, cube_owner: str,
        away1: int, away2: int, is_crawford: bool,
    ) -> CubeAssessment:
        """Return the bot's cube assessment for the player on roll."""


class NeongammonBot(BenchmarkBot):
    """The Neongammon production engine as a match-aware :class:`BenchmarkBot`."""

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
        away1: int, away2: int, is_crawford: bool,
    ) -> list[CheckerCandidate]:
        res = self._analyzer.checker_play(
            board, die1, die2, cube_value=cube_value, cube_owner=cube_owner,
            away1=away1, away2=away2, is_crawford=is_crawford,
        )
        return [CheckerCandidate(board=list(m.board), equity=m.equity) for m in res.moves]

    def cube_action(
        self, board: list[int], cube_value: int, cube_owner: str,
        away1: int, away2: int, is_crawford: bool,
    ) -> CubeAssessment:
        c = self._analyzer.cube_action(
            board, cube_value=cube_value, cube_owner=cube_owner,
            away1=away1, away2=away2, is_crawford=is_crawford,
        )
        return CubeAssessment(should_double=c.should_double, should_take=c.should_take)


# ===========================================================================
# Pass 1 - match self-play with analytics capture
# ===========================================================================


def _is_crawford_game(score1: int, score2: int, match_length: int, crawford_done: bool) -> bool:
    if crawford_done:
        return False
    return score1 == match_length - 1 or score2 == match_length - 1


def _capture_one_game(
    analyzer, rng: random.Random, match_length: int,
    score1: int, score2: int, is_crawford: bool, seed: int, game_number: int,
) -> tuple[list[dict], list[dict], Optional[int], Optional[str], int]:
    from neongammon import STARTING_BOARD

    return bc.simulate_game_and_capture(
        analyzer, rng,
        starting_board=STARTING_BOARD,
        seed=seed,
        match_length=match_length,
        score1=score1,
        score2=score2,
        is_crawford=is_crawford,
        game_number=game_number,
    )


def _simulate_match_and_capture(
    seed: int, match_length: int, parallel_threads: int,
) -> tuple[list[dict], dict]:
    """Play one Neongammon-3P vs Neongammon-3P match; capture every decision's 3P analytics."""
    analyzer = bc._get_analyzer("3ply", parallel_threads)
    rng = random.Random(seed)

    score1 = score2 = 0
    crawford_done = False
    all_decisions: list[dict] = []
    game_histories: list[dict] = []
    game_number = 0

    while score1 < match_length and score2 < match_length:
        game_number += 1
        if game_number > _MAX_GAMES:
            raise RuntimeError(f"match seed={seed} exceeded MAX_GAMES={_MAX_GAMES} at {match_length}pt")

        is_crawford = _is_crawford_game(score1, score2, match_length, crawford_done)
        start1, start2 = score1, score2

        decisions, move_history, winner, win_type, cube_at_end = _capture_one_game(
            analyzer, rng, match_length, start1, start2, is_crawford, seed, game_number,
        )
        all_decisions.extend(decisions)

        mult = {"single": 1, "gammon": 2, "backgammon": 3}.get(win_type or "single", 1)
        raw_points = (cube_at_end or 1) * mult
        if winner == 1:
            new_score = min(start1 + raw_points, match_length)
            points_awarded = new_score - start1
            score1 = new_score
        elif winner == 2:
            new_score = min(start2 + raw_points, match_length)
            points_awarded = new_score - start2
            score2 = new_score
        else:
            points_awarded = 0

        if winner is None:
            result_str, result_points = "", 0
        else:
            side = "player1" if winner == 1 else "player2"
            result_str, result_points = f"{side}-win-{win_type}", int(points_awarded)

        game_histories.append({
            "game_number": game_number,
            "player_score": start1,
            "opponent_score": start2,
            "result": result_str,
            "result_points": result_points,
            "move_history": move_history,
        })
        if is_crawford:
            crawford_done = True

    match_record = {
        "player1_name": "Neongammon",
        "player2_name": "Neongammon",
        "mode": "match",
        "match_length": match_length,
        "match_game_histories": game_histories,
    }
    return all_decisions, match_record


def _build_stage1_match(seed: int, match_length: int, write_txt: bool, parallel_threads: int) -> dict:
    """Worker: play one match, persist its decisions (and optional XG .txt), atomically."""
    paths = _Paths(match_length)
    decisions, match_record = _simulate_match_and_capture(seed, match_length, parallel_threads)
    bc.atomic_write_text(
        paths.stage1 / f"match_seed_{seed}.json",
        json.dumps({"seed": seed, "match_length": match_length, "decisions": decisions},
                   separators=(",", ":")),
    )
    if write_txt:
        from neongammon.text_export import export_history_to_txt

        (paths.xg / f"match_seed_{seed}.txt").write_bytes(export_history_to_txt(match_record))

    n_games = len(match_record["match_game_histories"])
    n_checker = sum(1 for d in decisions if d["kind"] == "checker")
    return {"seed": seed, "n_games": n_games, "n_checker": n_checker, "n_cube": len(decisions) - n_checker}


def _run_pass1(
    paths: _Paths, n_matches: int, match_length: int, seed: int, write_txt: bool, workers: int,
) -> None:
    paths.stage1.mkdir(parents=True, exist_ok=True)
    if write_txt:
        paths.xg.mkdir(parents=True, exist_ok=True)
    bc.run_pass1_pool(
        [seed + i for i in range(n_matches)],
        is_done_fn=lambda s: (paths.stage1 / f"match_seed_{s}.json").exists(),
        worker_fn=_build_stage1_match,
        worker_args_fn=lambda s, threads: (s, match_length, write_txt, threads),
        label=f"Pass 1/3 ({match_length}pt 3P self-play)",
        unit_name="matches",
        workers=workers,
        format_result=lambda r: (
            f"seed={r['seed']}: {r['n_games']} games, "
            f"{r['n_checker']} checker + {r['n_cube']} cube positions"
        ),
    )


def _load_stage1_decisions(paths: _Paths) -> list[dict]:
    return bc.load_stage1_decisions(paths.stage1, "match_seed_*.json")


# ===========================================================================
# Passes 2 & 3, job export & assembly
# ===========================================================================


def _reeval_decision(analyzer, decision: dict, progress_callback=None) -> dict:
    return bc.reeval_decision(analyzer, decision, progress_callback=progress_callback, match_mode=True)


def _reeval_decision_rollout(analyzer, decision: dict, progress_callback=None) -> dict:
    return bc.reeval_decision_rollout(analyzer, decision, progress_callback=progress_callback, match_mode=True)


def _export_threet_jobs(
    paths: _Paths, unique: dict[str, dict],
    min_seed: int | None = None, max_seed: int | None = None,
) -> Path:
    return bc.export_threet_jobs(
        unique, paths.stage2, paths.threet_jobs,
        match_mode=True, min_seed=min_seed, max_seed=max_seed,
    )


def _export_rollout_jobs(
    paths: _Paths, unique: dict[str, dict],
    min_seed: int | None = None, max_seed: int | None = None,
) -> Path:
    return bc.export_rollout_jobs(
        unique, paths.stage3, paths.rollout_jobs,
        match_mode=True, min_seed=min_seed, max_seed=max_seed,
    )


def _assemble_dataset(
    paths: _Paths, unique: dict[str, dict], n_matches: int, match_length: int, seed: int,
) -> dict:
    return bc.assemble_dataset(
        unique,
        paths.dataset,
        meta_prefix={
            "mode": "match",
            "match_length": match_length,
            "n_matches": n_matches,
            "seed": seed,
        },
        sort_key=lambda d: (d.get("seed", 0), d.get("game_number", 0), d.get("turn", 0), d["kind"]),
    )


def build_benchmark_data(
    match_length: int,
    n_matches: int,
    seed: int = 1,
    n_threads: int = 0,
    write_txt: bool = True,
    workers: int = 6,
    rollout_mode: str = "compute",
    threet_mode: str = "compute",
    export_min_seed: int | None = None,
    export_max_seed: int | None = None,
    stages: tuple = ("pass1", "pass2", "pass3"),
) -> dict:
    """Build the match-game benchmark data set (three adaptive-precision passes)."""
    if match_length <= 0:
        raise ValueError(f"match_length must be positive (got {match_length})")
    if n_matches <= 0:
        raise ValueError(f"n_matches must be positive (got {n_matches})")

    paths = _Paths(match_length)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    stage_set = set(stages)
    _log(f"=== build_benchmark_data (match): match_length={match_length} n_matches={n_matches} "
         f"seed={seed} n_threads={n_threads} workers={workers} write_txt={write_txt} "
         f"rollout_mode={rollout_mode} threet_mode={threet_mode} stages={sorted(stage_set)} ===")

    if "pass1" in stage_set:
        _run_pass1(paths, n_matches, match_length, seed, write_txt, workers)
    all_decisions = _load_stage1_decisions(paths)
    if not all_decisions:
        _log("No stage-1 decisions on disk -- run the 'pass1' stage first.")
        return {}
    unique = bc._unique_by_key(all_decisions)
    _log(f"Captured {len(all_decisions)} decisions ({len(unique)} unique positions).")

    unique = bc.run_passes_2_and_3(
        unique,
        stage2_file=paths.stage2,
        stage3_file=paths.stage3,
        threet_jobs_file=paths.threet_jobs,
        rollout_jobs_file=paths.rollout_jobs,
        stages=stage_set,
        threet_mode=threet_mode,
        rollout_mode=rollout_mode,
        n_threads=n_threads,
        export_min_seed=export_min_seed,
        export_max_seed=export_max_seed,
        reeval_3t_fn=_reeval_decision,
        reeval_rollout_fn=_reeval_decision_rollout,
        match_mode=True,
    )
    return _assemble_dataset(paths, unique, n_matches, match_length, seed)


# ===========================================================================
# Scoring
# ===========================================================================


def _score_checker(bot: BenchmarkBot, entry: dict) -> Optional[dict]:
    return bc.score_checker(bot, entry, match_mode=True)


def _score_cube(bot: BenchmarkBot, entry: dict) -> list[dict]:
    return bc.score_cube(bot, entry, match_mode=True)


def _score_entry(bot: BenchmarkBot, entry: dict) -> Optional[dict]:
    return bc.score_entry(bot, entry, match_mode=True)


def benchmark_pr(
    bot: BenchmarkBot,
    match_length: int,
    dataset_path: Path | str | None = None,
    n_threads: int = 1,
    label: str = "bot",
    cache_path: Path | str | None = None,
    progress: bool = True,
    max_seed: int | None = None,
) -> dict:
    """Score ``bot`` against the match benchmark data set; return the PR breakdown."""
    paths = _Paths(match_length)
    ds = Path(dataset_path) if dataset_path is not None else paths.dataset
    return bc.run_benchmark_pr(
        bot,
        dataset_path=ds,
        scores_dir=paths.scores,
        n_threads=n_threads,
        label=label,
        cache_path=cache_path,
        progress=progress,
        max_seed=max_seed,
        match_mode=True,
        unit_name="matches",
        score_entry_fn=_score_entry,
    )


# ===========================================================================
# CLI
# ===========================================================================


def _main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build the match benchmark data set")
    p_build.add_argument("--match-length", type=int, required=True, help="Match length in points")
    p_build.add_argument("--n-matches", type=int, default=130, help="Matches to simulate")
    bc.add_common_build_args(p_build, unit_name="match")

    p_score = sub.add_parser("score", help="Score the Neongammon engine against the data set")
    p_score.add_argument("--match-length", type=int, required=True, help="Match length in points")
    bc.add_common_score_args(p_score, unit_name="matches")

    args = parser.parse_args(argv)
    if args.command == "build":
        build_benchmark_data(
            match_length=args.match_length, n_matches=args.n_matches, seed=args.seed,
            n_threads=args.n_threads, write_txt=args.write_txt, workers=args.workers,
            rollout_mode=args.rollout_mode, threet_mode=args.threet_mode,
            export_min_seed=args.min_seed, export_max_seed=args.max_seed,
            stages=tuple(s.strip() for s in args.stages.split(",") if s.strip()),
        )
    elif args.command == "score":
        bot = NeongammonBot(eval_level=args.level, model=args.model, parallel_threads=1)
        label = args.label or f"sage_{args.level}" + (f"_{args.model}" if args.model else "")
        result = benchmark_pr(bot, match_length=args.match_length, n_threads=args.n_threads,
                              label=label, max_seed=args.max_seed)
        bm._print_report(result)


if __name__ == "__main__":
    _main()
