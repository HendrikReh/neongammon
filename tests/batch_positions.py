# SPDX-License-Identifier: MPL-2.0
"""Reproducible legal boards and dice for the public batch and parity tests."""
from random import Random

from neongammon import STARTING_BOARD, check_game_over, flip_board, possible_moves


def checker_positions(count, seed):
    rng = Random(seed)
    board = STARTING_BOARD
    contexts = (
        dict(cube_value=1, cube_owner="centered"),
        dict(cube_value=2, cube_owner="player", away1=5, away2=3),
        dict(cube_value=2, cube_owner="opponent"),
        dict(cube_value=1, cube_owner="centered", away1=1, away2=4, is_crawford=True),
    )
    positions = []
    for index in range(count):
        die1, die2 = rng.randint(1, 6), rng.randint(1, 6)
        positions.append(dict(board=board, die1=die1, die2=die2, **contexts[index % len(contexts)]))
        endpoint = rng.choice(possible_moves(board, die1, die2))
        board = STARTING_BOARD if check_game_over(endpoint) else flip_board(endpoint)
    return positions


def random_positions(
    count,
    seed,
    *,
    min_steps=4,
    max_steps=40,
    with_dice=True,
    min_moves=2,
    contexts=None,
):
    """Sample reproducible mid-game positions by random legal play from STARTING_BOARD.

    - ``with_dice=False`` returns ``list[list[int]]`` (pre-roll boards).
    - ``with_dice=True`` and ``contexts=None`` returns ``list[tuple[list[int], int, int]]``
      (``(board, die1, die2)`` with at least ``min_moves`` legal moves).
    - ``with_dice=True`` and ``contexts`` given returns ``list[dict]`` with
      ``board``, ``die1``, ``die2``, and ``**contexts[i % len(contexts)]``.
    """
    rng = Random(seed)
    out = []
    while len(out) < count:
        board = list(STARTING_BOARD)
        for _ in range(rng.randint(min_steps, max_steps)):
            moves = possible_moves(board, rng.randint(1, 6), rng.randint(1, 6))
            if moves:
                board = list(rng.choice(moves))
            if check_game_over(board):
                break
            board = list(flip_board(board))
        else:
            if not with_dice:
                out.append(board)
            else:
                d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
                if len(possible_moves(board, d1, d2)) >= min_moves:
                    if contexts is not None:
                        out.append(dict(board=board, die1=d1, die2=d2,
                                        **contexts[len(out) % len(contexts)]))
                    else:
                        out.append((board, d1, d2))
    return out
