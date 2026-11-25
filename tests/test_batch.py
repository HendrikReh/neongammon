# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Pre-roll batch evaluation on deterministic money and match positions."""
import pytest

from neongammon import BgBotAnalyzer, batch_evaluate
from .batch_positions import checker_positions


@pytest.fixture(scope="module")
def positions():
    return [{key: value for key, value in row.items() if key not in ("die1", "die2")}
            for row in checker_positions(100, seed=12345)]


def test_one_ply_matches_scalar_cube_analysis(positions):
    analyzer = BgBotAnalyzer(eval_level="1ply", parallel_threads=1)
    results = batch_evaluate(positions, eval_level="1ply", n_threads=2)
    assert len(results) == len(positions)
    for position, result in zip(positions, results):
        expected = analyzer.cube_action(**position)
        assert result.probs.to_list() == pytest.approx(expected.probs.to_list(), abs=1e-6)
        assert result.cubeful_equity == pytest.approx(expected.equity_nd, abs=1e-6)
        for field in ("cubeless_equity", "equity_nd", "equity_dt", "equity_dp"):
            assert getattr(result, field) == pytest.approx(getattr(expected, field), abs=1e-6)
        for field in ("should_double", "should_take", "optimal_action", "is_beaver"):
            assert getattr(result, field) == getattr(expected, field)


@pytest.mark.parametrize("level", ["1ply", "2ply"])
def test_parallel_matches_inline(positions, level):
    assert batch_evaluate(positions, eval_level=level, n_threads=2) == batch_evaluate(
        positions, eval_level=level, n_threads=1)
