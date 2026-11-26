# SPDX-License-Identifier: MPL-2.0
from dataclasses import asdict, fields

import pytest
import neongammon as ng


BEAROFF = [0, 3, 3, 2, 3, 2, 2] + [0] * 12 + [-2, -3, -2, -3, -2, -2] + [0]
POSITIONS = [
    dict(board=ng.STARTING_BOARD, beaver=False),
    dict(board=ng.STARTING_BOARD, beaver=True),
    dict(board=BEAROFF, cube_value=2, cube_owner="player", max_cube_value=2),
    dict(board=ng.STARTING_BOARD, away1=5, away2=3),
    dict(board=BEAROFF, away1=1, away2=3, is_crawford=True),
]


@pytest.mark.parametrize("decision_only", [False, True])
@pytest.mark.parametrize("level", ["1ply", "2ply", "3ply"])
def test_recursive_batch_matches_independent_scalar(level, decision_only):
    analyzer = ng.BgBotAnalyzer(eval_level=level, parallel_threads=1)
    scalar = analyzer.cube_decision if decision_only else analyzer.cube_action
    expected = [scalar(**position) for position in POSITIONS]
    results = ng.batch_cube_action_nply(POSITIONS, eval_level=level, decision_only=decision_only,
                                        n_threads=3)
    assert len(results) == len(expected)
    for got, want in zip(results, expected):
        assert type(got) is type(want)
        for field in fields(want):
            value = getattr(want, field.name)
            actual = getattr(got, field.name)
            if isinstance(value, ng.Probabilities):
                assert asdict(actual) == pytest.approx(asdict(value), abs=1e-7, rel=0)
            elif isinstance(value, float):
                assert actual == pytest.approx(value, abs=1e-7, rel=0)
            else:
                assert actual == value


def test_recursive_batch_validates_cube_limit_and_level_before_empty_return():
    with pytest.raises(ValueError):
        ng.batch_cube_action_nply([dict(board=BEAROFF, max_cube_value=3)])
    with pytest.raises(ValueError):
        ng.batch_cube_action_nply([], eval_level="rollout")
    assert ng.batch_cube_action_nply([]) == []


def test_public_top_two_matches_full_three_ply_at_same_root():
    shallow = ng.BgBotAnalyzer(eval_level="2ply", parallel_threads=2)
    deep = ng.BgBotAnalyzer(eval_level="3ply", parallel_threads=1)
    context = dict(cube_value=2, cube_owner="player", away1=5, away2=3)
    moves = shallow.checker_play(ng.STARTING_BOARD, 3, 1, **context).moves
    promoted, level = shallow.top_two_promote(ng.STARTING_BOARD, moves, max_gap=10, **context)
    assert level == "3-ply"
    expected = deep.checker_play(ng.STARTING_BOARD, 3, 1, **context).moves
    expected = [m for m in expected if m.board in [old.board for old in moves[:2]]]
    assert [m.board for m in promoted] == [m.board for m in expected]
    for actual, reference in zip(promoted, expected):
        assert actual.equity == pytest.approx(reference.equity, abs=1e-7)
        assert actual.probs == reference.probs
        assert actual.equity_diff == actual.equity - promoted[0].equity
    assert shallow.top_two_promote(BEAROFF, moves) == (moves, "2-ply")

@pytest.mark.parametrize("decision_only", [False, True])
def test_four_ply_batch_matches_scalar(decision_only):
    # Checkers outside the home board keep the root outside the bearoff DB.
    board = [0, 2, 3, 0, 2, 3, 1, 0, 1, 1] + [0] * 9 + [-3, -2, -2, -3, -2, -2] + [0]
    positions = [dict(board=board, cube_value=2, cube_owner="player", max_cube_value=4),
                 dict(board=board, away1=4, away2=2)]
    analyzer = ng.BgBotAnalyzer(eval_level="4ply", parallel_threads=1)
    scalar = analyzer.cube_decision if decision_only else analyzer.cube_action
    batch = ng.batch_cube_action_nply(positions, eval_level="4ply", n_threads=2,
                                      decision_only=decision_only)
    assert batch == [scalar(**position) for position in positions]


def test_short_batches_preserve_order_across_contact_and_bearoff_groups():
    analyzer = ng.BgBotAnalyzer(eval_level="3ply", parallel_threads=4)
    models = ng.BatchModels.from_analyzer(analyzer)
    rows = [POSITIONS[2], POSITIONS[0], POSITIONS[4], POSITIONS[3]]
    expected = [analyzer.cube_action(**row) for row in rows]
    for size in range(1, 5):
        assert ng.batch_cube_action_nply(rows[:size], eval_level="3ply", n_threads=4,
                                         models=models) == expected[:size]
