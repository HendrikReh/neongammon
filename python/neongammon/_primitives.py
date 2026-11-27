# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Pure-Python backgammon primitives shared across modules without ``bgbot_cpp``.

Keeping board validation, perspective flips, probability inversion, cube-owner
normalization, cubeless equity, and cube-context normalization here lets
pure-Python modules (``xg_file``, ``xg_compare``, ``data``, ``gnubg``,
``types``, ``cube_points``) remain importable even when the native
``bgbot_cpp`` extension is absent.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Integral

VALID_CUBE_OWNERS: frozenset[str] = frozenset(("centered", "player", "opponent"))

MAX_MATCH_LENGTH: int = 25

#: Standard backgammon starting position.
STARTING_BOARD: list[int] = [
    0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5,
    5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0,
]

_FLIP_OWNER_NAME: dict[str, str] = {
    "centered": "centered",
    "player": "opponent",
    "opponent": "player",
}


def validate_board(board: Iterable[int], *, nonterminal: bool = False) -> tuple[int, ...]:
    """Validate a 26-element backgammon board and return it as a tuple of ints.

    Board layout (mover's perspective):
      - ``board[0]``: opponent checkers on bar (``>= 0``)
      - ``board[1..24]``: points (``> 0`` mover, ``< 0`` opponent)
      - ``board[25]``: mover checkers on bar (``>= 0``)

    Raises ``ValueError`` if ``board`` does not have 26 integer elements
    (booleans rejected; NumPy integer scalars accepted), has negative bar
    counts, has more than 15 checkers for either side, or (when
    ``nonterminal=True``) has 0 checkers remaining for either side.
    """
    try:
        values = tuple(board)
    except TypeError as exc:
        raise ValueError("board must contain 26 integers") from exc
    if len(values) != 26 or any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in values
    ):
        raise ValueError("board must contain 26 integers")
    ints = tuple(int(value) for value in values)
    mover_on_board = ints[25] + sum(value for value in ints[1:25] if value > 0)
    opp_on_board = ints[0] + sum(-value for value in ints[1:25] if value < 0)
    if (
        ints[0] < 0
        or ints[25] < 0
        or mover_on_board > 15
        or opp_on_board > 15
    ):
        raise ValueError("invalid checker counts")
    if nonterminal and (mover_on_board == 0 or opp_on_board == 0):
        raise ValueError("board must be nonterminal")
    return ints


def flip_board(board: Sequence[int]) -> list[int]:
    """Flip a 26-element board to the other player's perspective."""
    return [board[25], *(-board[i] for i in range(24, 0, -1)), board[0]]


def invert_probs(probs: Sequence[float]) -> list[float]:
    """Invert probabilities from one player's perspective to the other.

    ``[P(win), P(gw), P(bw), P(gl), P(bl)]`` becomes
    ``[1-P(win), P(gl), P(bl), P(gw), P(bw)]``.
    """
    return [
        1.0 - probs[0],
        probs[3],
        probs[4],
        probs[1],
        probs[2],
    ]


def resolve_owner(cube_owner: object) -> str:
    """Resolve a case-insensitive cube owner name to a canonical string.

    Always raises ``ValueError`` for invalid inputs.
    """
    if isinstance(cube_owner, str):
        normalized = cube_owner.lower()
        if normalized in VALID_CUBE_OWNERS:
            return normalized
    raise ValueError(
        f"Invalid cube owner {cube_owner!r}. Expected centered, player, or opponent."
    )


def flip_owner(cube_owner: object) -> str:
    """Return the canonical cube owner name from the other player's perspective."""
    return _FLIP_OWNER_NAME[resolve_owner(cube_owner)]


def cubeless_equity(probs: Sequence[float]) -> float:
    """Compute cubeless equity from ``[P(win), P(gw), P(bw), P(gl), P(bl)]``."""
    return (
        2.0 * probs[0] - 1.0
        + probs[1] - probs[3]
        + probs[2] - probs[4]
    )


def _validate_away_scores(away1: int, away2: int) -> None:
    """Validate away scores: (0, 0) for money play or both in [1, 25]."""
    if (
        isinstance(away1, bool)
        or isinstance(away2, bool)
        or not isinstance(away1, Integral)
        or not isinstance(away2, Integral)
    ):
        raise ValueError("Match away scores must be integers")
    a1, a2 = int(away1), int(away2)
    if a1 == 0 and a2 == 0:
        return
    if a1 < 1 or a2 < 1:
        raise ValueError(
            f"Match away scores must both be 0 (money game) or both in [1, {MAX_MATCH_LENGTH}]"
        )
    if a1 > MAX_MATCH_LENGTH or a2 > MAX_MATCH_LENGTH:
        raise ValueError(
            f"Match equity requires each away score to be at most {MAX_MATCH_LENGTH}"
        )


def _validate_cube_limit(cube_value: int, max_cube_value: int) -> None:
    if (
        isinstance(cube_value, bool)
        or not isinstance(cube_value, Integral)
        or cube_value < 1
        or (int(cube_value) & (int(cube_value) - 1))
    ):
        raise ValueError("cube_value must be a positive power of two")
    if (
        isinstance(max_cube_value, bool)
        or not isinstance(max_cube_value, Integral)
        or max_cube_value < 0
        or max_cube_value == 1
        or (max_cube_value and int(max_cube_value) & (int(max_cube_value) - 1))
    ):
        raise ValueError("max_cube_value must be 0 or a power of two >= 2")
    if max_cube_value and cube_value > max_cube_value:
        raise ValueError("cube_value exceeds max_cube_value")


def _normalize_cube_context(
    cube_value: int,
    away1: int = 0,
    away2: int = 0,
    jacoby: bool = True,
    beaver: bool = True,
    max_cube_value: int = 0,
) -> tuple[bool, bool]:
    """Validate away scores and cube limits, and auto-disable Jacoby/beaver in match play."""
    _validate_away_scores(away1, away2)
    _validate_cube_limit(cube_value, max_cube_value)
    if away1 > 0:
        return False, False
    return bool(jacoby), bool(beaver)
