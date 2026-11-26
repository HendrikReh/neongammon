# SPDX-License-Identifier: MPL-2.0
from dataclasses import asdict, fields, replace

import pytest
import bgbot_cpp
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


@pytest.mark.parametrize("plies", [-1, 0])
@pytest.mark.parametrize("empty", [False, True])
def test_raw_checker_endpoints_reject_nonpositive_depth(plies, empty):
    base = ng.BatchModels().base()
    endpoints = [] if empty else ng.possible_moves(ng.STARTING_BOARD, 3, 1)[:1]
    with pytest.raises(ValueError, match="n_plies"):
        bgbot_cpp.score_checker_endpoints(ng.STARTING_BOARD, endpoints, bgbot_cpp.CubeInfo(),
                                          base, plies)


@pytest.mark.parametrize("plies", [-1, 0])
@pytest.mark.parametrize("empty", [False, True])
def test_raw_cube_batch_rejects_nonpositive_depth(plies, empty):
    base = ng.BatchModels().base()
    rows = [] if empty else [(ng.STARTING_BOARD, bgbot_cpp.CubeInfo())]
    with pytest.raises(ValueError, match="n_plies"):
        bgbot_cpp.batch_cube_action_nply(rows, base, None, plies, n_threads=1)


def test_raw_search_bindings_accept_one_ply():
    base = ng.BatchModels().base()
    cube = bgbot_cpp.CubeInfo(away1=3, away2=5)
    endpoint = ng.possible_moves(ng.STARTING_BOARD, 3, 1)[0]
    move, = bgbot_cpp.score_checker_endpoints(ng.STARTING_BOARD, [endpoint], cube, base, 1)
    assert move["board"] == endpoint
    assert move["eval_level"] == "1-ply"
    result, = bgbot_cpp.batch_cube_action_nply([(ng.STARTING_BOARD, cube)], base, None, 1)
    scalar = ng.BgBotAnalyzer(eval_level="1ply").cube_action(ng.STARTING_BOARD, away1=3, away2=5)
    assert result["equity_nd"] == pytest.approx(scalar.equity_nd)


@pytest.mark.parametrize("board", [ng.STARTING_BOARD, BEAROFF], ids=["contact", "race"])
def test_top_two_rejects_reversed_ranking(board):
    shallow = ng.BgBotAnalyzer(eval_level="2ply", parallel_threads=1)
    first, second = shallow.checker_play(board, 3, 1).moves[:2]
    assert first.equity > second.equity
    with pytest.raises(ValueError, match="equity"):
        shallow.top_two_promote(board, [second, first])


def test_top_two_promotes_tied_leaders_at_zero_gap():
    shallow = ng.BgBotAnalyzer(eval_level="2ply", parallel_threads=1)
    first, second = shallow.checker_play(ng.STARTING_BOARD, 3, 1).moves[:2]
    tied = [first, replace(second, equity=first.equity, equity_diff=0)]
    promoted, level = shallow.top_two_promote(ng.STARTING_BOARD, tied, max_gap=0)
    assert level == "3-ply"
    assert {tuple(move.board) for move in promoted} == {tuple(move.board) for move in tied}


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


@pytest.mark.parametrize("level", ["3ply", "4ply"])
@pytest.mark.parametrize("kwargs", [{"filter_max_moves": 3}, {"filter_threshold": 0.2}])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("batch,mode", [
    (ng.batch_checker_play, {}), (ng.batch_cube_action_nply, {}),
    (ng.batch_cube_action_nply, {"decision_only": True}),
], ids=["checker", "cube", "decision-only"])
def test_deep_batches_reject_ineffective_move_filters(level, kwargs, empty, batch, mode):
    rows = [] if empty else [dict(board=BEAROFF, die1=1, die2=2)]
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        batch(rows, eval_level=level, n_threads=1, **mode, **kwargs)


@pytest.mark.parametrize("plies", [3, 4])
def test_native_deep_batches_reject_ineffective_move_filters(plies):
    models = ng.BatchModels()
    with pytest.raises(ValueError, match="filter_threshold"):
        bgbot_cpp.batch_checker_play([], models.base(), models.multipy(plies),
                                     filter_threshold=0.3, cubeful_filtered_plies=[2])
    with pytest.raises(ValueError, match="filter_max_moves"):
        bgbot_cpp.batch_cube_action_nply([], models.base(), None, plies, filter_max_moves=10)


@pytest.mark.parametrize("plies", [2, 3, 4])
def test_checker_filters_control_initial_survivors_at_filtered_depths(plies):
    row = dict(board=BEAROFF, die1=1, die2=2, cube_value=1)
    models = ng.BatchModels()

    def searched_count(max_moves, threshold):
        if plies == 2:
            result, = ng.batch_checker_play([row], "2ply", models=models, n_threads=2,
                                            filter_max_moves=max_moves, filter_threshold=threshold)
            return sum(move.eval_level == "2-ply" for move in result.moves)
        # Raw callers can opt into a filtered checker root at deeper levels.
        result, = bgbot_cpp.batch_checker_play(
            [dict(row, cube_owner=bgbot_cpp.CubeOwner.CENTERED)], models.base(),
            models.multipy(plies), filter_max_moves=max_moves, filter_threshold=threshold,
            cubeful_filtered_plies=[plies], n_threads=2)
        return sum(move["eval_level"] == f"{plies}-ply" for move in result["moves"])

    legal = len(ng.possible_moves(BEAROFF, 1, 2))
    assert searched_count(legal, 1.0) == legal
    assert 2 <= searched_count(1, 1.0) < legal
    assert 2 <= searched_count(legal, 0.0) < legal


@pytest.mark.parametrize("level", ["3ply", "4ply"])
def test_explicit_default_deep_batch_filters_match_omitted(level):
    rows = [dict(board=BEAROFF, die1=1, die2=2)]
    models = ng.BatchModels()
    for batch in (ng.batch_checker_play, ng.batch_cube_action_nply):
        expected = batch(rows, level, models=models, n_threads=2)
        for threshold in (0.08, bgbot_cpp.MOVE_FILTER_TINY.threshold):
            assert batch(rows, level, models=models, n_threads=2, filter_max_moves=5,
                         filter_threshold=threshold) == expected


@pytest.mark.parametrize("batch,level", [
    (ng.batch_checker_play, "1ply"),
    (ng.batch_cube_action_nply, "1ply"),
    (ng.batch_cube_action_nply, "2ply"),
])
def test_shallow_batches_accept_legacy_filter_settings(batch, level):
    rows = [dict(board=BEAROFF, die1=1, die2=2)]
    models = ng.BatchModels()
    expected = batch(rows, level, models=models, n_threads=1)
    assert batch(rows, level, models=models, n_threads=1, filter_max_moves=1,
                 filter_threshold=0.01) == expected
