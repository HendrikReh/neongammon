# SPDX-License-Identifier: MPL-2.0

from math import isnan

import pytest

from neongammon.board import flip_board
from neongammon.xg_compare import compute_game_pr_stats


def test_game_pr_attributes_response_and_checker_errors_to_their_players():
    played = [0] * 26
    played[1] = 1
    turns = [
        {
            "player": "user",
            "cube_action": "double/take",
            "cube_analysis": {"equity_nd": 0.9, "equity_dt": 1.2, "equity_dp": 1.0},
        },
        {
            "player": "user",
            "cube_action": "no double",
            "cube_analysis": {"equity_nd": 0.4, "equity_dt": 0.6, "equity_dp": 1.0},
        },
        {
            "player": "bot",
            "board_after": played,
            "checker_analysis": [
                {"board": [0] * 26, "equity": 0.4, "equity_diff": 0},
                {"board": flip_board(played), "equity": 0.3, "equity_diff": -0.1},
            ],
        },
    ]
    stats = compute_game_pr_stats(turns)
    assert stats == pytest.approx({
        "user_err": 0.0, "user_dec": 1, "bot_err": 0.3, "bot_dec": 2,
        "total_err": 0.3, "total_dec": 3, "pr": 50.0,
    })
    assert compute_game_pr_stats(turns) == stats


def test_game_pr_is_nan_when_every_decision_is_forced_or_trivial():
    stats = compute_game_pr_stats([{
        "player": "user",
        "cube_action": "double/take",
        "cube_analysis": {"equity_nd": 1.0, "equity_dt": 1.0, "equity_dp": 1.0},
        "checker_analysis": [{"equity": 0.5}],
    }])
    assert stats["total_dec"] == stats["total_err"] == 0
    assert isnan(stats["pr"])
