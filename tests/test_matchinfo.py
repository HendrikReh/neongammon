# SPDX-License-Identifier: MPL-2.0
import pytest

from neongammon.matchinfo import (
    gammon_prices, take_points, take_points_dead_cube, take_points_janowski,
)


def test_gammon_price_at_one_away():
    # At cube 2 the leader wins the match with a single. The trailer
    # advances to three-away on a single and one-away on a gammon, both
    # post-Crawford.
    trailer_single = 0.32264
    trailer_gammon = 0.5
    price = (trailer_gammon - trailer_single) / trailer_single
    assert gammon_prices(1, 5, 1) == pytest.approx((0.0, price))
    assert gammon_prices(5, 1, 1) == pytest.approx((price, 0.0))


@pytest.mark.parametrize("take_points_fn", [take_points, take_points_dead_cube, take_points_janowski])
def test_take_points_at_post_crawford_score(take_points_fn):
    # The leader's MWC is 0.511970 after passing, 0.5 after taking and
    # losing, and 1 after taking and winning. The trailer is always at
    # least as well off taking a hypothetical double from the leader.
    points = (0.02394, 0.0)
    assert take_points_fn(1, 3) == pytest.approx(points, abs=1e-6)
    assert take_points_fn(3, 1) == pytest.approx(points[::-1], abs=1e-6)


def test_dead_cube_take_points_before_the_crawford_game():
    # A doubled loss at 2-away vs 3-away makes the new score 2-away vs
    # 1-away with Crawford still to play. Passing leaves 2-away vs 2-away.
    assert take_points_dead_cube(2, 3) == pytest.approx(
        ((0.5 - 0.32264) / (1.0 - 0.32264), 0.24924 / 0.67736), abs=1e-6,
    )


def test_dead_cube_take_points_from_the_crawford_game_use_post_crawford_outcomes():
    # A take point in the Crawford game is hypothetical. Any scored game
    # outcome takes the match into the post-Crawford phase or ends it.
    assert take_points_dead_cube(1, 3, is_crawford=True) == pytest.approx(
        (0.02394, 0.0), abs=1e-6,
    )


@pytest.mark.parametrize("cube", [1, 2, 4])
def test_live_and_dead_money_take_points(cube):
    assert take_points(0, 0, cube) == (0.22, 0.22)
    assert take_points_dead_cube(0, 0, cube) == (0.25, 0.25)


@pytest.mark.parametrize("x", [0.0, 0.5, 0.68, 1.0])
def test_take_points_janowski_money_honors_cube_life_index(x):
    expected = 0.5 / (2.0 + 0.5 * x)
    assert take_points_janowski(0, 0, 1, x=x) == pytest.approx((expected, expected), abs=1e-9)


def test_take_points_2a_2a_uses_auto_redouble_formula():
    from neongammon import matchinfo

    assert (2, 2) not in matchinfo._TAKE_POINTS_CV1
    assert take_points(2, 2, 1) == pytest.approx((0.32264, 0.32264), abs=1e-5)


@pytest.mark.parametrize("fn", [take_points, take_points_dead_cube, take_points_janowski, gammon_prices])
@pytest.mark.parametrize("away1,away2", [(0, 5), (5, 0), (-1, 5), (-1, -2), (26, 5)])
def test_matchinfo_rejects_invalid_away_scores(fn, away1, away2):
    with pytest.raises(ValueError, match="away"):
        fn(away1, away2)


@pytest.mark.parametrize("fn", [take_points, take_points_dead_cube, take_points_janowski, gammon_prices])
@pytest.mark.parametrize("cube_value", [0, -1, 3])
def test_matchinfo_rejects_invalid_cube_value(fn, cube_value):
    with pytest.raises(ValueError, match="cube_value"):
        fn(5, 3, cube_value)
