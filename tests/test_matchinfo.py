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
