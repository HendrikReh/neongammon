# SPDX-License-Identifier: MPL-2.0
"""Tests for the public XG-compatible match text exporter."""

import pytest

from neongammon.text_export import compute_move_notation, export_history_to_txt


def test_money_game_export_contains_utf8_metadata_and_cube_actions():
    record = {
        "player1_name": "Éva",
        "started_at": "2026-11-27T07:15:00+00:00",
        "history_id": "money-1",
        "move_history": [
            {"player": "user", "dice": [6, 4], "move": "bar/19* 4/off(2)"},
            {"player": "bot", "cube_action": "double/take", "dice": [1, 1]},
            {"player": "bot", "cube_action": "double/pass"},
        ],
        "result": "player2-win",
        "result_points": 2,
    }
    text = export_history_to_txt(record).decode("utf-8")
    assert text.splitlines()[:6] == [
        '; [Site "NeonGammon"]',
        '; [Match ID "money-1"]',
        '; [Player 1 "Éva"]',
        '; [Player 2 "NeonGammon"]',
        '; [EventDate "2026.11.27"]',
        '; [EventTime "07.15"]',
    ]
    assert "0 point match" in text
    assert "64: 25/19 4/0 4/0" in text
    assert text.index("Doubles => 2") < text.index("Takes") < text.index("11:")
    assert text.index("Doubles => 4") < text.index("Drops") < text.index("Wins 2 point")
    assert text.endswith("\n")


def test_match_export_orders_games_and_carries_the_score():
    text = export_history_to_txt({
        "mode": "match",
        "match_length": 3,
        "match_game_histories": [
            {"game_number": 2, "result": "player1-win", "result_points": 2},
            {"game_number": 1, "result": "player1-win", "result_points": 1},
        ],
    }).decode("utf-8")
    assert "3 point match" in text
    assert text.index("Game 1") < text.index("You : 0") < text.index("Wins 1 point")
    assert text.index("Game 2") < text.index("You : 1") < text.index("Wins 2 point and the match")


@pytest.mark.parametrize("dice", [[True, 2], [2, False], [6, 7], [4.5, 1]])
def test_invalid_history_dice_do_not_produce_a_move_row(dice):
    text = export_history_to_txt({
        "move_history": [{"dice": dice, "move": "13/8"}],
    }).decode("utf-8")
    assert "13/8" not in text


def test_legacy_history_accepts_dice_as_strings():
    text = export_history_to_txt({
        "move_history": [{"dice": ["5", "1"], "move": "13/8 6/5"}],
    }).decode("utf-8")
    assert "51: 13/8 6/5" in text


@pytest.mark.parametrize("die", [True, False, 0, 7, 1.5])
def test_move_notation_requires_valid_dice(die):
    board = [0] * 26
    with pytest.raises(ValueError, match="dice"):
        compute_move_notation(board, board, die, 1)
