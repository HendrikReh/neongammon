# SPDX-License-Identifier: MPL-2.0
"""Match analysis must report scores outside the Kazaross-XG2 table."""

import bgbot_cpp
import pytest

from neongammon import BgBotAnalyzer, STARTING_BOARD, batch_checker_play
from neongammon.cube_points import cube_action_points
from neongammon.matchinfo import take_points_dead_cube


@pytest.mark.parametrize("away1,away2", [(26, 26), (31, 31), (25, 26), (26, 25)])
def test_checker_play_rejects_unsupported_away_scores(away1, away2):
    analyzer = BgBotAnalyzer(eval_level="1ply")
    with pytest.raises(ValueError, match="away.*25"):
        analyzer.checker_play(STARTING_BOARD, 3, 1, away1=away1, away2=away2)


def test_batch_checker_play_rejects_unsupported_away_scores():
    row = dict(board=STARTING_BOARD, die1=3, die2=1)
    positions = [dict(row, away1=25, away2=25), dict(row, away1=31, away2=31)]
    with pytest.raises(ValueError, match="away.*25"):
        batch_checker_play(positions, n_threads=2)


def test_met_rejects_unsupported_away_scores():
    with pytest.raises(ValueError, match="away.*25"):
        bgbot_cpp.get_met(26, 25)


def test_match_equity_validates_the_origin_even_when_every_result_is_terminal():
    with pytest.raises(ValueError, match="away.*25"):
        bgbot_cpp.eq2mwc(0.0, 31, 31, cube_value=32)


def test_match_points_validate_the_origin_even_when_every_result_is_terminal():
    with pytest.raises(ValueError, match="away.*25"):
        take_points_dead_cube(31, 31, cube_value=32)


def test_cube_action_points_reject_unsupported_crawford_scores():
    with pytest.raises(ValueError, match="away.*25"):
        cube_action_points([0.5, 0, 0, 0, 0], "centered", 1, 1, 26, True)


def test_opening_checker_play_still_ranks_moves_at_the_table_boundary():
    analyzer = BgBotAnalyzer(eval_level="1ply")
    result = analyzer.checker_play(STARTING_BOARD, 3, 1, away1=25, away2=25)
    assert len(result.moves) == 16
    assert result.moves[0].equity > result.moves[-1].equity


def test_met_accepts_terminal_outcomes():
    assert bgbot_cpp.get_met(-2, 25) == 1.0
    assert bgbot_cpp.get_met(25, 0) == 0.0
    assert bgbot_cpp.eq2mwc(0.0, 25, 25, cube_value=32) == 0.5


INVALID_AWAY_PAIRS = [(0, 5), (5, 0), (-1, 5), (5, -1), (-1, -2), (0, -1)]


@pytest.mark.parametrize("away1,away2", INVALID_AWAY_PAIRS)
def test_analyzer_and_batch_reject_negative_or_half_zero_away_scores(away1, away2):
    from neongammon import (
        batch_cube_action,
        batch_cube_action_nply,
        batch_evaluate,
        batch_post_move_evaluate,
        can_double_match,
    )

    analyzer = BgBotAnalyzer(eval_level="1ply", parallel_threads=1)
    with pytest.raises(ValueError, match="away"):
        analyzer.cube_action(STARTING_BOARD, away1=away1, away2=away2)
    with pytest.raises(ValueError, match="away"):
        analyzer.cube_decision(STARTING_BOARD, away1=away1, away2=away2)
    with pytest.raises(ValueError, match="away"):
        analyzer.checker_play(STARTING_BOARD, 3, 1, away1=away1, away2=away2)
    with pytest.raises(ValueError, match="away"):
        analyzer.post_move_analytics(STARTING_BOARD, away1=away1, away2=away2)

    row = dict(board=STARTING_BOARD, away1=away1, away2=away2)
    with pytest.raises(ValueError, match="away"):
        batch_evaluate([row], n_threads=1)
    with pytest.raises(ValueError, match="away"):
        batch_cube_action([row], n_threads=1)
    with pytest.raises(ValueError, match="away"):
        batch_post_move_evaluate([row], n_threads=1)
    with pytest.raises(ValueError, match="away"):
        batch_cube_action_nply([row], eval_level="1ply", n_threads=1)
    with pytest.raises(ValueError, match="away"):
        batch_checker_play([dict(row, die1=3, die2=1)], n_threads=1)
    with pytest.raises(ValueError, match="away"):
        can_double_match(away1, away2, 1, "centered")

    strat = analyzer._models.base()
    with pytest.raises(ValueError, match="away"):
        bgbot_cpp.cubeful_equity_nply(
            STARTING_BOARD, bgbot_cpp.CubeOwner.CENTERED, strat, 1,
            away1=away1, away2=away2,
        )
    with pytest.raises(ValueError, match="away"):
        bgbot_cpp.cubeful_probs_nply(
            STARTING_BOARD, bgbot_cpp.CubeOwner.CENTERED, strat, 1,
            away1=away1, away2=away2,
        )
    with pytest.raises(ValueError, match="away"):
        bgbot_cpp.cubeful_probs_and_equity_nply(
            STARTING_BOARD, bgbot_cpp.CubeOwner.CENTERED, strat, 1,
            away1=away1, away2=away2,
        )


@pytest.mark.parametrize("bad_cv", [0, -1, 3, True])
def test_analyzer_and_batch_reject_invalid_cube_value(bad_cv):
    from neongammon import batch_cube_action_nply, batch_evaluate, batch_post_move_evaluate

    analyzer = BgBotAnalyzer(eval_level="1ply", parallel_threads=1)
    with pytest.raises(ValueError, match="cube_value"):
        analyzer.cube_action(STARTING_BOARD, cube_value=bad_cv)
    with pytest.raises(ValueError, match="cube_value"):
        analyzer.checker_play(STARTING_BOARD, 3, 1, cube_value=bad_cv)
    with pytest.raises(ValueError, match="cube_value"):
        analyzer.post_move_analytics(STARTING_BOARD, cube_value=bad_cv)

    row = dict(board=STARTING_BOARD, cube_value=bad_cv)
    with pytest.raises(ValueError, match="cube_value"):
        batch_evaluate([row], n_threads=1)
    with pytest.raises(ValueError, match="cube_value"):
        batch_post_move_evaluate([row], n_threads=1)
    with pytest.raises(ValueError, match="cube_value"):
        batch_cube_action_nply([row], eval_level="1ply", n_threads=1)
    with pytest.raises(ValueError, match="cube_value"):
        batch_checker_play([dict(row, die1=3, die2=1)], n_threads=1)


def test_upfront_cube_owner_validation_on_cubeless_1ply_and_top_two_promote():
    analyzer_1ply = BgBotAnalyzer(eval_level="1ply", cubeful=False)
    with pytest.raises(ValueError, match="Invalid cube owner"):
        analyzer_1ply.checker_play(STARTING_BOARD, 3, 1, cube_owner="bogus")

    analyzer_2ply = BgBotAnalyzer(eval_level="2ply", cubeful=True, parallel_threads=1)
    # Early return (len(moves) < 2) must still validate cube_owner upfront.
    with pytest.raises(ValueError, match="Invalid cube owner"):
        analyzer_2ply.top_two_promote(STARTING_BOARD, [], cube_owner="bogus")
