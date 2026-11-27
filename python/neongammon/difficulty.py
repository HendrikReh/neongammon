# SPDX-License-Identifier: MPL-2.0
"""Seeded opponent policies with bounded deviations from Stage 11."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import exp
from random import Random
from typing import Protocol

from ._primitives import resolve_owner as _resolve_owner_name
from .facade import BgBotAnalyzer
from .board import can_double_match, possible_moves
from .types import CubeDecisionResult
from .batch import BatchModels


class Policy(Protocol):
    """An opponent that supplies a legal checker move and cube response."""

    def select_checker_move(self, board: list[int], die1: int, die2: int, **context) -> list[int]: ...

    def cube_decision(self, board: list[int], cube_value: int = 1,
                      cube_owner: str = "centered", **context) -> CubeDecisionResult: ...


class Difficulty(str, Enum):
    """Opponent presets, in order of increasing strength."""

    BEGINNER = "beginner"
    CASUAL = "casual"
    INTERMEDIATE = "intermediate"
    FULL_STRENGTH = "full_strength"


@dataclass(frozen=True)
class _Profile:
    checker_limit: float
    checker_temperature: float
    cube_limit: float
    cube_temperature: float


_PROFILES = {
    Difficulty.BEGINNER: _Profile(0.160, 0.500, 0.060, 0.025),
    Difficulty.CASUAL: _Profile(0.120, 0.500, 0.040, 0.016),
    Difficulty.INTERMEDIATE: _Profile(0.080, 0.072, 0.025, 0.009),
}


def _choice_weights(losses: list[float], limit: float, temperature: float) -> list[float]:
    """Weights for a best-first list of nonnegative equity losses."""
    if temperature == 0:
        return [1.0] + [0.0] * (len(losses) - 1)
    return [exp(-loss / temperature) if loss <= limit else 0.0 for loss in losses]


def _cube_gaps(decision: CubeDecisionResult) -> tuple[float, float]:
    nd, dt, dp = decision.equity_nd, decision.equity_dt, decision.equity_dp
    return abs(nd - min(dt, dp)), abs(dt - dp)


def _cube_error_probability(gap: float, profile: _Profile) -> float:
    best, other = _choice_weights([0.0, gap], profile.cube_limit, profile.cube_temperature)
    return other / (best + other)


class DifficultyPolicy:
    """Choose checker moves and cube actions for one opponent.

    Create a fresh policy with a separate seed for each player and match.
    The simulation's dice are independent of this policy's random stream.
    Use BgBotAnalyzer separately to rank moves or review the game.
    """

    def __init__(
        self,
        difficulty: Difficulty | str,
        *,
        seed: int = 42,
        parallel_threads: int = 1,
        analyzer: BgBotAnalyzer | None = None,
        checker_analyzer: BgBotAnalyzer | None = None,
        cube_analyzer: BgBotAnalyzer | None = None,
        models: BatchModels | None = None,
    ):
        if (
            analyzer is not None
            and checker_analyzer is not None
            and analyzer is not checker_analyzer
        ):
            raise ValueError("pass analyzer or checker_analyzer, not both")
        self.difficulty = Difficulty(difficulty)
        self._profile = _PROFILES.get(self.difficulty)
        resolved_checker = checker_analyzer if checker_analyzer is not None else analyzer
        if resolved_checker is None:
            checker_models = models if models is not None else (
                BatchModels.from_analyzer(cube_analyzer) if cube_analyzer is not None else None
            )
            resolved_checker = BgBotAnalyzer(
                eval_level="1ply" if self._profile else "3ply",
                models=checker_models,
                parallel_threads=parallel_threads,
            )
        if cube_analyzer is None:
            if self._profile is None and resolved_checker._eval_level != "2ply":
                cube_models = (
                    models if models is not None
                    else BatchModels.from_analyzer(resolved_checker)
                )
                cube_analyzer = BgBotAnalyzer(
                    eval_level="2ply",
                    parallel_threads=parallel_threads,
                    models=cube_models,
                )
            else:
                cube_analyzer = resolved_checker
        self._analyzer = resolved_checker
        self._cube_analyzer = cube_analyzer
        self._rng = Random(seed)

    @property
    def checker_analyzer(self) -> BgBotAnalyzer:
        """Analyzer used for checker play."""
        return self._analyzer

    @property
    def cube_analyzer(self) -> BgBotAnalyzer:
        """Analyzer used for cube decisions."""
        return self._cube_analyzer

    def reseed(self, seed: int) -> None:
        """Restart this policy's random stream while retaining its models."""
        self._rng.seed(seed)

    def select_checker_move(self, board: list[int], die1: int, die2: int, **context) -> list[int]:
        """Return a legal post-move board from the mover's perspective."""
        legal = possible_moves(board, die1, die2)
        if len(legal) == 1:
            return legal[0]
        moves = self._analyzer.checker_play(board, die1, die2, **context).moves
        profile = self._profile
        if profile is None:
            return moves[0].board
        losses = [moves[0].equity - move.equity for move in moves]
        weights = _choice_weights(losses, profile.checker_limit, profile.checker_temperature)
        return self._rng.choices(moves, weights=weights)[0].board

    def cube_decision(self, board: list[int], cube_value: int = 1, cube_owner: str = "centered",
                      **context) -> CubeDecisionResult:
        """Return policy choices with the underlying analyzer's equities and recommendation.

        The board and equities are from the doubler's perspective even when
        the caller reads the responder's ``should_take`` flag.
        """
        cube_owner = _resolve_owner_name(cube_owner)
        decision = self._cube_analyzer.cube_decision(board, cube_value, cube_owner, **context)
        profile = self._profile
        if profile is None or decision.is_beaver:
            return decision
        cap = context.get("max_cube_value", 0)
        away1, away2 = context.get("away1", 0), context.get("away2", 0)
        crawford = context.get("is_crawford", False)
        if not can_double_match(away1, away2, cube_value, cube_owner, crawford, max_cube_value=cap):
            return decision
        double_gap, take_gap = _cube_gaps(decision)
        flip_double = self._rng.random() < _cube_error_probability(double_gap, profile)
        flip_take = self._rng.random() < _cube_error_probability(take_gap, profile)
        return replace(decision, should_double=decision.should_double != flip_double,
                       should_take=decision.should_take != flip_take)
