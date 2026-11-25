# SPDX-License-Identifier: MPL-2.0
import pytest
import bgbot_cpp

from neongammon.matchinfo import gammon_prices, take_points, take_points_dead_cube


def test_gammon_price_at_one_away():
    # At cube 2 the leader wins the match with a single. The trailer
    # advances to three-away on a single and one-away on a gammon.
    trailer_single = bgbot_cpp.get_met(3, 1, False)
    trailer_gammon = 0.5
    price = (trailer_gammon - trailer_single) / trailer_single
    assert gammon_prices(1, 5, 1) == pytest.approx((0.0, price))
    assert gammon_prices(5, 1, 1) == pytest.approx((price, 0.0))


@pytest.mark.parametrize("cube", [1, 2, 4])
def test_live_and_dead_money_take_points(cube):
    assert take_points(0, 0, cube) == (0.22, 0.22)
    assert take_points_dead_cube(0, 0, cube) == (0.25, 0.25)
