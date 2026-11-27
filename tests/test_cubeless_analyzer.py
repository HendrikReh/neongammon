# SPDX-License-Identifier: MPL-2.0
"""Regression tests for cubeful=False checker_play across rollout and N-ply analyzers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from neongammon import BgBotAnalyzer, STARTING_BOARD
from neongammon import strategies


def test_cubeless_rollout_checker_play_runs_cubeless_trials_and_prefilter():
    """BgBotAnalyzer(eval_level='rollout', cubeful=False).checker_play() must run
    cubeless rollouts (evaluate_board, never cubeful_evaluate_board) and
    prefilter candidates with cubeless equity (never _janowski_1ply_equity)."""
    analyzer = BgBotAnalyzer(
        eval_level="rollout",
        cubeful=False,
        n_trials=36,
        truncation_depth=5,
        decision_ply=1,
        parallel_threads=1,
    )

    counts = {"cl": 0, "cf": 0}
    orig_eval = analyzer._analyzer._rollout_strategy.evaluate_board
    orig_cf_eval = analyzer._analyzer._rollout_strategy.cubeful_evaluate_board

    class SpyStrategy:
        def __getattr__(self, name):
            return getattr(analyzer._analyzer._rollout_strategy, name)

        def evaluate_board(self, *args, **kwargs):
            counts["cl"] += 1
            return orig_eval(*args, **kwargs)

        def cubeful_evaluate_board(self, *args, **kwargs):
            counts["cf"] += 1
            return orig_cf_eval(*args, **kwargs)

    # Also check upfront cube_owner validation on cubeless rollout.
    with pytest.raises(ValueError, match="Invalid cube owner"):
        analyzer.checker_play(STARTING_BOARD, 3, 1, cube_owner="bogus")

    orig_strategy = analyzer._analyzer._rollout_strategy
    analyzer._analyzer._rollout_strategy = SpyStrategy()
    try:
        with patch.object(
            strategies, "_janowski_1ply_equity",
            side_effect=AssertionError("cubeless rollout must not call Janowski prefilter"),
        ):
            res = analyzer.checker_play(STARTING_BOARD, 3, 1)
    finally:
        analyzer._analyzer._rollout_strategy = orig_strategy

    assert counts["cl"] > 0
    assert counts["cf"] == 0
    assert res.moves
    for move in res.moves:
        assert move.equity == pytest.approx(move.cubeless_equity, abs=1e-6)


def test_cubeless_multiply_checker_play_prefilters_cubelessly():
    """BgBotAnalyzer(eval_level='2ply', cubeful=False).checker_play() must
    prefilter candidates using cubeless 1-ply equity, never Janowski."""
    analyzer = BgBotAnalyzer(eval_level="2ply", cubeful=False, parallel_threads=1)

    with pytest.raises(ValueError, match="Invalid cube owner"):
        analyzer.checker_play(STARTING_BOARD, 3, 1, cube_owner="bogus")

    with patch.object(
        strategies, "_janowski_1ply_equity",
        side_effect=AssertionError("cubeless 2-ply must not call Janowski prefilter"),
    ):
        res = analyzer.checker_play(STARTING_BOARD, 3, 1)

    assert len(res.moves) == 16
    for move in res.moves:
        assert move.equity == pytest.approx(move.cubeless_equity, abs=1e-6)
