# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""neongammon — the Neongammon bot engine, a neural-network backgammon engine.

Provides position analysis (checker play and cube decisions) at multiple
evaluation levels: raw NN (1-ply), N-ply search, and Monte Carlo rollout.

Quick start::

    from neongammon import BgBotAnalyzer, STARTING_BOARD

    analyzer = BgBotAnalyzer()
    result = analyzer.checker_play(STARTING_BOARD, 3, 1)
    for m in result.moves[:3]:
        print(f"{m.equity:+.3f}  {m.probs.win:.1%}")
"""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING

from . import settings as settings
from .types import (
    CheckerPlayResult as CheckerPlayResult,
    CubeActionResult as CubeActionResult,
    CubeDecisionResult as CubeDecisionResult,
    GamePlanResult as GamePlanResult,
    GameStats as GameStats,
    MatchInfo as MatchInfo,
    MoveAnalysis as MoveAnalysis,
    PostMoveAnalysis as PostMoveAnalysis,
    Probabilities as Probabilities,
    LuckResult as LuckResult,
    RollEquity as RollEquity,
)
from .weights import (
    MODELS as MODELS,
    PRODUCTION_MODEL as PRODUCTION_MODEL,
    WeightConfig as WeightConfig,
    WeightConfigPair as WeightConfigPair,
    default_weights as default_weights,
)

if TYPE_CHECKING:
    from bgbot_cpp import TrialEvalConfig as TrialEvalConfig

    from .analyzer import (
        BgBotAnalyzer as BgBotAnalyzer,
        RolloutCancelled as RolloutCancelled,
        contact_cube_life as contact_cube_life,
        contact_cube_slope as contact_cube_slope,
        create_analyzer as create_analyzer,
        flip_owner as flip_owner,
    )
    from .batch import (
        BatchModels as BatchModels,
        PositionEval as PositionEval,
        batch_checker_play as batch_checker_play,
        batch_cube_action as batch_cube_action,
        batch_cube_action_nply as batch_cube_action_nply,
        batch_evaluate as batch_evaluate,
        batch_post_move_evaluate as batch_post_move_evaluate,
    )
    from .board import (
        STARTING_BOARD as STARTING_BOARD,
        can_double_match as can_double_match,
        check_game_over as check_game_over,
        classify_game_plan as classify_game_plan,
        classify_game_plans as classify_game_plans,
        flip_board as flip_board,
        invert_probs as invert_probs,
        is_crashed as is_crashed,
        is_race as is_race,
        possible_moves as possible_moves,
        possible_moves_all as possible_moves_all,
        possible_single_die_moves as possible_single_die_moves,
    )
    from .gnubg import GnuBgAnalyzer as GnuBgAnalyzer
    from .difficulty import (
        Difficulty as Difficulty,
        DifficultyPolicy as DifficultyPolicy,
        Policy as Policy,
    )
    from .puretd import (
        PureTDPolicy as PureTDPolicy,
        default_money_policy as default_money_policy,
    )
    from .luck import (
        luck_from_equities as luck_from_equities,
        roll_luck as roll_luck,
    )
    from .simulation import (
        GameConfig as GameConfig,
        SimulationCancelled as SimulationCancelled,
        SimulationLimitExceeded as SimulationLimitExceeded,
        simulate_game as simulate_game,
        simulate_match as simulate_match,
    )

# Resolve engine exports on first access. The reporting, weights and settings
# modules can be imported even when the native extension is unavailable.
_ENGINE_MODULES = {
    "analyzer": (
        "BgBotAnalyzer",
        "RolloutCancelled",
        "contact_cube_life",
        "contact_cube_slope",
        "create_analyzer",
        "flip_owner",
    ),
    "batch": (
        "BatchModels",
        "PositionEval",
        "batch_checker_play",
        "batch_cube_action",
        "batch_cube_action_nply",
        "batch_evaluate",
        "batch_post_move_evaluate",
    ),
    "board": (
        "STARTING_BOARD",
        "can_double_match",
        "check_game_over",
        "classify_game_plan",
        "classify_game_plans",
        "flip_board",
        "invert_probs",
        "is_crashed",
        "is_race",
        "possible_moves",
        "possible_moves_all",
        "possible_single_die_moves",
    ),
    "gnubg": ("GnuBgAnalyzer",),
    "difficulty": (
        "Difficulty",
        "DifficultyPolicy",
        "Policy",
    ),
    "puretd": (
        "PureTDPolicy",
        "default_money_policy",
    ),
    "luck": (
        "luck_from_equities",
        "roll_luck",
    ),
    "simulation": (
        "GameConfig",
        "SimulationCancelled",
        "SimulationLimitExceeded",
        "simulate_game",
        "simulate_match",
    ),
}
_ENGINE_EXPORTS = {
    name: module for module, names in _ENGINE_MODULES.items() for name in names
}

def __getattr__(name: str):
    if name == "TrialEvalConfig":
        # Older extensions do not expose the trial configuration.
        try:
            value = _import_module("bgbot_cpp").TrialEvalConfig
        except (ImportError, AttributeError):
            value = None
    elif name in _ENGINE_EXPORTS:
        module = _import_module(f".{_ENGINE_EXPORTS[name]}", __name__)
        value = getattr(module, name)
    elif name in _ENGINE_MODULES:
        value = _import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _ENGINE_MODULES.keys())


__all__ = [
    "GameConfig", "SimulationCancelled", "SimulationLimitExceeded", "simulate_game", "simulate_match",
    "BgBotAnalyzer",
    "RolloutCancelled",
    "GnuBgAnalyzer",
    "Difficulty",
    "DifficultyPolicy",
    "Policy",
    "default_money_policy",
    "flip_owner",
    "contact_cube_life",
    "contact_cube_slope",
    "PureTDPolicy",
    "BatchModels",
    "batch_checker_play",
    "batch_cube_action",
    "batch_cube_action_nply",
    "batch_evaluate",
    "batch_post_move_evaluate",
    "create_analyzer",
    "WeightConfig",
    "WeightConfigPair",
    "default_weights",
    "PRODUCTION_MODEL",
    "MODELS",
    "Probabilities",
    "MoveAnalysis",
    "PostMoveAnalysis",
    "CheckerPlayResult",
    "CubeActionResult",
    "CubeDecisionResult",
    "GamePlanResult",
    "GameStats",
    "MatchInfo",
    "LuckResult",
    "RollEquity",
    "luck_from_equities",
    "roll_luck",
    "PositionEval",
    "STARTING_BOARD",
    "flip_board",
    "possible_moves",
    "possible_moves_all",
    "possible_single_die_moves",
    "check_game_over",
    "classify_game_plan",
    "classify_game_plans",
    "is_race",
    "is_crashed",
    "invert_probs",
    "can_double_match",
    "settings",
    "TrialEvalConfig",
]
