# SPDX-License-Identifier: MPL-2.0
"""Reproducible legal boards and dice for the public batch APIs."""
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
