# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Board utilities — thin wrappers around bgbot_cpp functions.

These are importable without creating an analyzer, useful for game logic,
move generation, and position classification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._native import load_native
from ._primitives import (
    STARTING_BOARD as STARTING_BOARD,
    _validate_away_scores,
    _validate_cube_limit,
    invert_probs as invert_probs,
    resolve_owner as _resolve_owner_name,
    validate_board as validate_board,
)

bgbot_cpp = load_native()

if TYPE_CHECKING:
    from .types import GamePlanResult

_OWNER_ENUM_MAP = {
    "centered": bgbot_cpp.CubeOwner.CENTERED,
    "player": bgbot_cpp.CubeOwner.PLAYER,
    "opponent": bgbot_cpp.CubeOwner.OPPONENT,
}
_VALID_NATIVE_OWNERS = frozenset(_OWNER_ENUM_MAP.values())


def flip_board(board: list[int]) -> list[int]:
    """Flip a board to the other player's perspective."""
    return bgbot_cpp.flip_board(board)


def possible_moves(board: list[int], die1: int, die2: int) -> list[list[int]]:
    """Return all legal resulting board positions for the given dice.

    Each element is a 26-element board representing one distinct final position.
    The list is never empty: when the roll cannot be played at all (e.g. dancing
    on the bar), the unchanged board is returned as the single result. Test for
    that with ``result == [board]``; a single result alone may be a forced move.
    """
    return [list(b) for b in bgbot_cpp.possible_moves(board, die1, die2)]


def possible_moves_all(board):
    """All legal endpoints and roll masks as contiguous NumPy arrays.

    Returns int8 boards of shape (N, 26) and uint32 masks of shape (N,).
    Bit k corresponds to roll (a, b) in a=1..6, b=a..6 order. Uses the
    native batch binding when available and an exact older-engine fallback.
    """
    from .move_arrays import possible_moves_all as all_moves
    return all_moves(board)


def possible_single_die_moves(
    board: list[int], die: int
) -> list[dict[str, object]]:
    """Return legal single-die moves for click-to-move UI.

    Each dict has keys: ``from`` (int), ``to`` (int), ``board`` (list[int]).
    """
    return bgbot_cpp.possible_single_die_moves(board, die)


def check_game_over(board: list[int]) -> int:
    """Check if the game is over.

    Returns:
        0 if not over; +1/-1 single win/loss; +2/-2 gammon; +3/-3 backgammon.
        Positive means the player whose perspective the board is in has won.
    """
    return bgbot_cpp.check_game_over(board)


def classify_game_plan(board: list[int]) -> str:
    """Classify the game plan for the player on roll.

    Returns one of: ``"purerace"``, ``"racing"``, ``"attacking"``,
    ``"priming"``, ``"anchoring"``.
    """
    gp = bgbot_cpp.classify_game_plan(board)
    return gp.name.lower() if hasattr(gp, "name") else str(gp).lower()


def classify_game_plans(board: list[int]) -> GamePlanResult:
    """Classify game plans for both the player on roll and the opponent.

    Returns a :class:`~neongammon.types.GamePlanResult` with ``player`` and
    ``opponent`` fields, each one of: ``"purerace"``, ``"racing"``,
    ``"attacking"``, ``"priming"``, ``"anchoring"``.
    """
    from .types import GamePlanResult

    player_gp = classify_game_plan(board)
    flipped = bgbot_cpp.flip_board(board)
    opponent_gp = classify_game_plan(flipped)
    return GamePlanResult(player=player_gp, opponent=opponent_gp)


def is_race(board: list[int]) -> bool:
    """Return True if contact is broken (pure race position)."""
    return bgbot_cpp.is_race(board)


def is_crashed(board: list[int]) -> bool:
    """Return True if the position is classified as crashed."""
    return bgbot_cpp.is_crashed(board)


def can_double_match(
    away1: int, away2: int, cube_value: int, owner: str,
    is_crawford: bool = False, *, max_cube_value: int = 0,
) -> bool:
    """Check whether the player on roll may offer the cube in match or money play.

    Args:
        away1: Points the roller needs to win.
        away2: Points the opponent needs to win.
        cube_value: Current cube value.
        owner: Cube owner — ``"centered"``, ``"player"``, or ``"opponent"``.
        is_crawford: Whether this is the Crawford game.
        max_cube_value: Maximum cube value (0 for unlimited).

    Returns ``False`` for Crawford game, dead cube (cube already covers
    remaining points), or when the opponent owns the cube.  For unlimited
    games (both away values zero), eligibility depends on ownership. The
    maximum cube value applies to both kinds of game.
    """
    _validate_away_scores(away1, away2)
    _validate_cube_limit(cube_value, max_cube_value)
    if isinstance(owner, bgbot_cpp.CubeOwner) and owner in _VALID_NATIVE_OWNERS:
        cpp_owner = owner
    else:
        cpp_owner = _OWNER_ENUM_MAP[_resolve_owner_name(owner)]
    return bgbot_cpp.can_double_match(
        away1, away2, cube_value, cpp_owner, is_crawford, max_cube_value)
