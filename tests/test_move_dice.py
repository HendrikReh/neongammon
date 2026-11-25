# SPDX-License-Identifier: MPL-2.0
import time

import bgbot_cpp
import pytest

from neongammon import (
    STARTING_BOARD,
    BgBotAnalyzer,
    batch_checker_play,
    possible_moves,
    possible_single_die_moves,
)

# Verified legal 15-checker boards whose 1-1 roll exceeds MoveDedup's initial
# 2,048-slot table (zero-bar: 2,125 legal endpoints; bar: 2,223 legal endpoints).
HIGH_BRANCHING_ZERO_BAR = [
    0, -7, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1,
    -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 0,
]
HIGH_BRANCHING_BAR = [
    6, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1,
    -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 0,
]


def _brute_force_moves(board: list[int], die1: int, die2: int) -> list[list[int]]:
    """Reference move generator built strictly from possible_single_die_moves."""
    if die1 == die2:
        frontier = {tuple(board)}
        best: set[tuple[int, ...]] = set()
        for _ in range(4):
            nxt: set[tuple[int, ...]] = set()
            for b in frontier:
                for m in possible_single_die_moves(list(b), die1):
                    nxt.add(tuple(m["board"]))
            if not nxt:
                break
            frontier = nxt
            best = nxt
        return [list(b) for b in sorted(best)] if best else [list(board)]

    two_die: set[tuple[int, ...]] = set()
    for first, second in ((die1, die2), (die2, die1)):
        for m1 in possible_single_die_moves(board, first):
            b1 = list(m1["board"])
            if not any(x > 0 for x in b1[1:26]):
                two_die.add(tuple(b1))
                continue
            for m2 in possible_single_die_moves(b1, second):
                two_die.add(tuple(m2["board"]))
    if two_die:
        return [list(b) for b in sorted(two_die)]
    for d in (max(die1, die2), min(die1, die2)):
        one_die = {tuple(m["board"]) for m in possible_single_die_moves(board, d)}
        if one_die:
            return [list(b) for b in sorted(one_die)]
    return [list(board)]


@pytest.mark.parametrize("die", [0, -1, 7, 32])
def test_invalid_die_rejected_by_move_generators(die):
    for first, second in ((die, 1), (1, die), (die, die)):
        with pytest.raises(ValueError, match="between 1 and 6"):
            possible_moves(STARTING_BOARD, first, second)
    with pytest.raises(ValueError, match="between 1 and 6"):
        possible_single_die_moves(STARTING_BOARD, die)


@pytest.mark.parametrize("level", ["1ply", "2ply"])
def test_invalid_dice_rejected_before_worker_dispatch(level):
    good = dict(board=STARTING_BOARD, die1=1, die2=2)
    with pytest.raises(ValueError, match="between 1 and 6"):
        batch_checker_play([good, {**good, "die2": 32}], eval_level=level, n_threads=2)


@pytest.mark.parametrize(
    "board,expected_count",
    [
        (HIGH_BRANCHING_ZERO_BAR, 2125),
        (HIGH_BRANCHING_BAR, 2223),
    ],
)
def test_high_branching_1_1_moves_above_2048_exact_parity(board, expected_count):
    # Confirm both test boards are legal 15-checker positions.
    assert sum(x for x in board[1:26] if x > 0) == 15
    assert board[0] + sum(-x for x in board[1:25] if x < 0) == 15

    moves_sorted = [list(b) for b in possible_moves(board, 1, 1)]
    moves_unsorted = [list(b) for b in bgbot_cpp.possible_moves_unsorted(board, 1, 1)]
    reference = _brute_force_moves(board, 1, 1)

    assert len(moves_sorted) == expected_count
    assert len(moves_unsorted) == expected_count
    assert len({tuple(b) for b in moves_unsorted}) == expected_count
    assert moves_sorted == sorted(moves_unsorted)
    assert moves_sorted == reference

    # Also spot-check mixed roll MoveDedup parity on the same high-branching boards.
    for d1, d2 in ((1, 2), (2, 1)):
        mixed = [list(b) for b in possible_moves(board, d1, d2)]
        assert mixed == _brute_force_moves(board, d1, d2)


def test_high_branching_2ply_cube_action_and_checker_play_no_hang():
    analyzer = BgBotAnalyzer(eval_level="2ply")
    # Warm up weights and thread pool before timing.
    analyzer.cube_action(STARTING_BOARD)

    for board, expected_count in (
        (HIGH_BRANCHING_ZERO_BAR, 2125),
        (HIGH_BRANCHING_BAR, 2223),
    ):
        t0 = time.perf_counter()
        cube = analyzer.cube_action(board)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 50.0, f"2-ply cube_action took {elapsed_ms:.1f} ms (expected < 50 ms)"
        assert 0.0 <= cube.probs.win <= 1.0

        checker = analyzer.checker_play(board, 1, 1)
        expected_set = {tuple(b) for b in possible_moves(board, 1, 1)}
        got_set = {tuple(m.board) for m in checker.moves}
        assert len(checker.moves) == expected_count
        assert got_set == expected_set
