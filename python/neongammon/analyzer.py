# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Compatibility imports for the NeonGammon analysis API."""

from .cubeful import (
    CUBEFUL_FILTERED_PLIES as CUBEFUL_FILTERED_PLIES,
    SERIAL_CUBEFUL_1PLY as SERIAL_CUBEFUL_1PLY,
)
from .facade import (
    BgBotAnalyzer as BgBotAnalyzer,
    _optimal_action as _optimal_action,
    create_analyzer as create_analyzer,
)
from .rollout import RolloutCancelled as RolloutCancelled
from .strategies import (
    MONEY_CONTACT_CUBE_LIFE as MONEY_CONTACT_CUBE_LIFE,
    MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE as MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE,
    OWNER_MAP as OWNER_MAP,
    _contact_cube_life as _contact_cube_life,
    _contact_cube_slope as _contact_cube_slope,
    _validate_cube_limit as _validate_cube_limit,
    contact_cube_life as contact_cube_life,
    contact_cube_slope as contact_cube_slope,
    flip_owner as flip_owner,
    load_bearoff_db as load_bearoff_db,
    resolve_owner as resolve_owner,
)
from .types import (
    CheckerPlayResult as CheckerPlayResult,
    CubeActionResult as CubeActionResult,
    CubeDecisionResult as CubeDecisionResult,
    LuckResult as LuckResult,
    MoveAnalysis as MoveAnalysis,
    PostMoveAnalysis as PostMoveAnalysis,
    Probabilities as Probabilities,
)
from .weights import (
    WeightConfig as WeightConfig,
    WeightConfigPair as WeightConfigPair,
    default_weights as default_weights,
)
