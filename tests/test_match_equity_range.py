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
