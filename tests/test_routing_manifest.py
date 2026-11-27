# SPDX-License-Identifier: MPL-2.0
from dataclasses import replace

import bgbot_cpp
import pytest

from neongammon import STARTING_BOARD, BatchModels
from neongammon.weights import WeightConfigPair

SNAKE = [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 3, 3, 2, 2, 1, -7, -7, 0]


@pytest.mark.parametrize("name", ["stage9", "stage11"])
def test_registered_routes_match_legacy_factory(name):
    weights = WeightConfigPair.from_model(name)
    configured = BatchModels(weights, bearoff_db=False).base()
    legacy = bgbot_cpp.create_strategy(weights.strategy_type, *weights.weight_args)

    assert configured.root_pinned() == legacy.root_pinned()
    for board in (STARTING_BOARD, SNAKE):
        assert configured.select_nn_idx(board) == legacy.select_nn_idx(board)
        assert configured.evaluate_board(board, board) == legacy.evaluate_board(board, board)


def test_analyzer_uses_explicit_route_slots_and_pins():
    weights = WeightConfigPair.from_model("stage11")
    weights = replace(weights, routing={
        "contact_pairs": (2,) * 16,
        "snake": 19,
        "root_pins": (19,),
    })
    strategy = BatchModels(weights, bearoff_db=False).base()

    assert strategy.select_nn_idx(STARTING_BOARD) == 2
    assert strategy.select_nn_idx(SNAKE) == 19
    assert strategy.root_pin_for(SNAKE) == 19


def test_invalid_manifest_is_rejected_before_weight_loading():
    routing = bgbot_cpp.RoutingManifest()
    routing.snake = 200
    with pytest.raises(ValueError, match="existing non-race network slot"):
        bgbot_cpp.RoutedStrategy(["unused"] * 17, [1] * 17, routing)
