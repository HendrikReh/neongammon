# SPDX-License-Identifier: MPL-2.0

import pytest

from neongammon import CubeActionResult, Probabilities, RollEquity, luck_from_equities, roll_luck


def test_luck_weights_each_roll_and_accepts_reversed_dice():
    rolls = [RollEquity(1, 1, -0.5, 1), RollEquity(1, 2, 0.25, 2)]
    result = luck_from_equities(rolls, 2, 1, ply=3, level_label="3-ply")
    assert result is not None
    assert result.actual_equity == 0.25
    assert result.average_equity == 0
    assert result.luck == 0.25
    assert (result.ply, result.level_label, result.per_roll) == (3, "3-ply", rolls)


def test_opening_luck_excludes_doubles_and_renormalizes():
    rolls = [RollEquity(1, 1, -1, 1), RollEquity(1, 2, 0, 2), RollEquity(1, 3, 0.5, 2)]
    result = luck_from_equities(rolls, 2, 1, ply=1, level_label="1-ply", is_opening_roll=True)
    assert result is not None
    assert result.luck == -0.25
    assert result.per_roll == rolls[1:]
    assert luck_from_equities(
        rolls, 1, 1, ply=1, level_label="1-ply", is_opening_roll=True,
    ) is None


def test_luck_requires_nonzero_weight():
    assert luck_from_equities(
        [RollEquity(1, 2, 0, 0)], 1, 2, ply=1, level_label="1-ply",
    ) is None


@pytest.fixture
def cube_action():
    return CubeActionResult(
        probs=Probabilities(0.5, 0, 0, 0, 0),
        cubeless_equity=0,
        equity_nd=0,
        equity_dt=0,
        equity_dp=1,
        should_double=False,
        should_take=True,
        optimal_equity=0,
        optimal_action="No Double",
        eval_level="3-ply",
        details={
            "nd": [
                {"die1": 1, "die2": 1, "cubeful_equity": 0.7},
                {"die1": 1, "die2": 2, "cubeful_equity": 0.4},
            ],
            "dt": [{"die1": 1, "die2": 2, "cubeful_equity": -0.5}],
        },
    )


@pytest.mark.parametrize(("eval_level", "ply"), [("3-ply", 2), ("Rollout", 1)])
def test_roll_luck_uses_no_double_equities_and_derives_ply(cube_action, eval_level, ply):
    cube_action.eval_level = eval_level
    result = roll_luck(cube_action, 1, 2)
    assert result is not None
    assert result.actual_equity == 0.4
    assert result.average_equity == pytest.approx(0.5)
    assert result.luck == pytest.approx(-0.1)
    assert (result.ply, result.level_label) == (ply, f"{ply}-ply")


def test_roll_luck_handles_missing_details(cube_action):
    assert roll_luck(cube_action, 2, 3) is None
    cube_action.details = None
    assert roll_luck(cube_action, 1, 2) is None
