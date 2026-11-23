# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Parallel cubeful rollouts must read complete opening-move cache entries,
and nested trial rollouts must preserve parent move-candidate buffers."""

import pytest

from neongammon import BgBotAnalyzer, TrialEvalConfig, WeightConfig

# Contact positions with live cubes that exercise move-0 / move-1 prefill and VR.
_CONTACT_CUBE = [1, -2, 0, 0, 2, 2, 3, 0, 3, 0, 0, 2, -3, 2, -1, 0, 0, 0,
                 -2, -2, -2, -2, 0, 1, 0, 0]
_CONTACT_CHECKER = [0, 0, 0, 2, 3, 0, 4, -2, 2, 0, 0, 0, -4,
                    2, -3, 0, -1, 0, 0, -3, 2, 0, -2, 0, 0, 0]


@pytest.mark.parametrize(
    "level,api,inputs,seed,repetitions",
    [
        ("truncated1", "cube_action", dict(board=_CONTACT_CUBE), 3972021868, 6),
        ("truncated2", "checker_play", dict(board=_CONTACT_CHECKER, die1=4, die2=2), 2570830711, 2),
    ],
)
def test_parallel_prefill_matches_serial(level, api, inputs, seed, repetitions):
    weights = WeightConfig.from_model("stage11")
    serial = BgBotAnalyzer(weights=weights, eval_level=level, parallel_threads=1)
    serial.set_seed(seed)
    expected = getattr(serial, api)(**inputs)

    parallel = BgBotAnalyzer(weights=weights, eval_level=level, parallel_threads=10)
    for _ in range(repetitions):
        parallel.set_seed(seed)
        assert getattr(parallel, api)(**inputs) == expected


def test_partial_opening_stratum_preserves_shared_reply():
    options = dict(
        weights=WeightConfig.from_model("stage11"),
        eval_level="rollout",
        n_trials=37,
        truncation_depth=3,
        decision_ply=1,
        truncation_ply=2,
        seed=1720888050,
        cubeful=True,
    )
    serial = BgBotAnalyzer(parallel_threads=1, **options).cube_action(_CONTACT_CUBE)
    assert BgBotAnalyzer(parallel_threads=4, **options).cube_action(_CONTACT_CUBE) == serial


@pytest.mark.parametrize("purpose", ["checker", "checker_late", "cube"])
def test_nested_trial_keeps_its_parents_legal_boards(purpose):
    options = dict(
        weights=WeightConfig.from_model("stage11"),
        eval_level="rollout",
        n_trials=4,
        truncation_depth=3,
        decision_ply=1,
        truncation_ply=1,
        late_threshold=2,
        seed=1720888050,
        cubeful=True,
    )
    options[purpose] = TrialEvalConfig(rollout_trials=2, rollout_depth=1, rollout_ply=1)
    serial = BgBotAnalyzer(parallel_threads=1, **options).cube_action(_CONTACT_CUBE)
    assert BgBotAnalyzer(parallel_threads=2, **options).cube_action(_CONTACT_CUBE) == serial
