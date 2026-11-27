# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""The cube policy and the cubeless probability policy may share NN work."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pytest

from neongammon import BgBotAnalyzer


# The policies choose different continuations on this 1-away/4-away position.
# Substituting the cube search's probabilities shifts gammon wins by 1.52 pp.
MATCH = [1, 2, -1, 3, 3, 2, 3, -1, 2, 0, 0, 0, -3,
         0, 0, 0, 0, -3, 0, -4, 0, -2, 0, 0, 0, 0]
SCORE = dict(cube_value=2, cube_owner="player", away1=1, away2=4)
# This position exercises the specialist root routing and large doubles pools.
SNAKE = [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 1, 0, 2, 3, 3, 2, 2, 1, -7, -7, 0]
# The #63 latency fixture has many candidates with a live owned cube.
OWNED_MONEY = [0, 0, 0, 1, 2, -2, 3, 2, 3, 0, 0, 0, -3,
               4, 0, 0, 0, -3, 0, -3, -2, -2, 0, 0, 0, 0]


@pytest.mark.parametrize("eval_level", ["2ply", "3ply"])
@pytest.mark.parametrize(("board", "options"), [(MATCH, SCORE), (SNAKE, {})])
def test_cube_probabilities_match_independent_cubeless_tree(eval_level, board, options):
    analyzer = BgBotAnalyzer(eval_level=eval_level, parallel_threads=2)
    plain = analyzer.cube_action(board, **options)
    # Per-roll details use their own traversal and still run the standalone
    # cubeless search to populate the five headline probabilities.
    independent = analyzer.cube_action(board, incl_2ply_details=True, **options)
    assert plain.probs == independent.probs
    assert plain.cubeless_equity == independent.cubeless_equity
    assert independent.details is not None


@pytest.mark.parametrize("eval_level", ["3ply", "4ply"])
def test_owned_money_cube_matches_both_independent_searches(eval_level):
    analyzer = BgBotAnalyzer(eval_level=eval_level, parallel_threads=4)
    options = dict(cube_value=2, cube_owner="player", jacoby=True, beaver=False)
    rich = analyzer.cube_action(OWNED_MONEY, **options)
    decision = analyzer.cube_decision(OWNED_MONEY, **options)
    independent = analyzer.cube_action(OWNED_MONEY, incl_2ply_details=True, **options)

    rich_fields = asdict(rich)
    decision_fields = asdict(decision)
    assert {key: rich_fields[key] for key in decision_fields} == decision_fields
    assert rich.probs == independent.probs
    assert rich.cubeless_equity == independent.cubeless_equity


def test_concurrent_cube_calls_keep_ordered_batches_local():
    analyzer = BgBotAnalyzer(eval_level="3ply", parallel_threads=2)
    requests = [(MATCH, SCORE), (SNAKE, {}), (MATCH, {}), (SNAKE, SCORE)]

    def evaluate(request):
        board, options = request
        return analyzer.cube_action(board, **options)

    expected = [evaluate(request) for request in requests]
    assert expected[0].probs == expected[2].probs
    assert expected[0].equity_nd != expected[2].equity_nd
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(evaluate, requests * 2)) == expected * 2


def test_4ply_checker_play_cooperative_and_subtrees_match_serial():
    """4-ply parallel cooperative roll splitting + CheckerSubtrees matches 1-thread."""
    serial = BgBotAnalyzer(eval_level="4ply", parallel_threads=1)
    parallel = BgBotAnalyzer(eval_level="4ply", parallel_threads=8)
    s_res = serial.checker_play(MATCH, 3, 1, **SCORE)
    p_res = parallel.checker_play(MATCH, 3, 1, **SCORE)
    assert len(s_res.moves) == len(p_res.moves)
    for sm, pm in zip(s_res.moves, p_res.moves):
        assert sm.board == pm.board
        assert sm.equity == pytest.approx(pm.equity, abs=1e-6)
        assert sm.cubeless_equity == pytest.approx(pm.cubeless_equity, abs=1e-6)
