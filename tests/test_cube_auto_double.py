# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Tests for unified cube decision logic (#102):

1. Inner truncated-rollout cube decisions in match play must apply post-Crawford
   auto_double consistently with 1-ply and N-ply cube decisions.
2. Target-SE batch mode is honored consistently across RolloutStrategy.cube_decision
   and cubeful_evaluate_board (preserving checker-cube symmetry under target_se).
3. Batched/single-batch rollout runner cancellation & parity.
"""

import pytest

import bgbot_cpp
from neongammon import BgBotAnalyzer, TrialEvalConfig, flip_board
from neongammon.weights import WeightConfig


# Short racing position where the player on roll (SP) leads slightly, so the
# opponent (trailing 4-away vs 1-away post-Crawford) is an underdog who would
# have ND >= DT in a 1-ply-truncated inner rollout (because cl2cf_match already
# values ND at the auto-doubled cube) and therefore MUST rely on explicit
# auto_double to turn the cube on move 1 rather than playing out the full game
# at cube=1. At 1-away vs 4-away, losing a single at cube=1 (1a/3a, ~68% MWC)
# vs cube=2 (1a/2a, 50% MWC) separates auto-doubled from frozen cube=1 by >0.30
# equity and >0.05 MWC.
_SHORT_RACE_LEADER_ON_ROLL = [0] * 26
_SHORT_RACE_LEADER_ON_ROLL[5] = 2
_SHORT_RACE_LEADER_ON_ROLL[6] = 3
_SHORT_RACE_LEADER_ON_ROLL[19] = -3
_SHORT_RACE_LEADER_ON_ROLL[20] = -2

# Mirror: 4-away trailer on roll, 1-away leader off roll, cube=1 centered.
_SHORT_RACE_TRAILER_ON_ROLL = flip_board(_SHORT_RACE_LEADER_ON_ROLL)

# Midgame contact position for target_se symmetry checks.
_CONTACT_BOARD = [
    0, -2, 0, 0, 0, 2, 4, 0, 2, 0, 0, 0, -5,
    5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0,
]


def test_inner_truncated_rollout_applies_post_crawford_auto_double_on_move1():
    """Leader (1-away) on roll vs trailer (4-away) post-Crawford, cube=1 centered.

    Move 0: leader plays (cannot double).
    Move 1: trailer is on roll with cube=1 centered (post-Crawford auto-double!).
    In an inner truncated rollout (rollout_depth=1), cl2cf_match values ND at
    the auto-doubled cube so nd_m == dt_m up to variance. Without explicit
    auto_double in run_phase1_cube_check, min(dt_m, dp_m) > nd_m fails and the
    cube freezes at 1 for the whole full-game trial, mispricing the leader's
    MWC/equity by >0.30 equity relative to 1-ply cube decisions and relative to
    the dead-cube-at-2 reference.
    """
    common = dict(
        eval_level="rollout",
        n_trials=72,
        truncation_depth=0,
        decision_ply=1,
        seed=42,
        parallel_threads=1,
        cubeful=True,
    )
    an_1ply = BgBotAnalyzer(**common, cube=TrialEvalConfig(ply=1))
    an_inner_ro = BgBotAnalyzer(
        **common,
        cube=TrialEvalConfig(rollout_trials=18, rollout_depth=1, rollout_ply=1),
    )

    # Post-Crawford: away1=1 (leader on roll), away2=4 (trailer), cube=1 centered.
    res_1ply = an_1ply.cube_action(
        _SHORT_RACE_LEADER_ON_ROLL,
        cube_value=1, cube_owner="centered",
        away1=1, away2=4, is_crawford=False,
    )
    res_inner_ro = an_inner_ro.cube_action(
        _SHORT_RACE_LEADER_ON_ROLL,
        cube_value=1, cube_owner="centered",
        away1=1, away2=4, is_crawford=False,
    )
    # Reference: cube already doubled to 2 (leader owns dead cube). Because the
    # leader takes the trailer's auto-double on move 1 (trailer is behind),
    # ND MWC at cube=1 must match the cube=2 dead-cube rollout closely, NOT
    # a frozen cube=1 game (which differs by >0.30 equity / >0.05 MWC).
    res_cube2 = an_1ply.cube_action(
        _SHORT_RACE_LEADER_ON_ROLL,
        cube_value=2, cube_owner="player",
        away1=1, away2=4, is_crawford=False,
    )
    res_frozen_c1 = an_1ply.cube_action(
        _SHORT_RACE_LEADER_ON_ROLL,
        cube_value=1, cube_owner="centered",
        away1=1, away2=4, is_crawford=True,  # Crawford freezes cube at 1
    )

    mwc_inner_ro = bgbot_cpp.eq2mwc(res_inner_ro.equity_nd, 1, 4, 1, False)
    mwc_cube2 = bgbot_cpp.eq2mwc(res_cube2.equity_nd, 1, 4, 2, False)
    mwc_frozen_c1 = bgbot_cpp.eq2mwc(res_frozen_c1.equity_nd, 1, 4, 1, True)

    assert not res_inner_ro.should_double  # 1-away leader cannot double
    assert abs(res_inner_ro.equity_nd - res_1ply.equity_nd) < 0.03, (
        f"Inner truncated-rollout cube drifted from 1-ply post-Crawford auto_double: "
        f"inner_ro={res_inner_ro.equity_nd:.4f}, 1ply={res_1ply.equity_nd:.4f}, "
        f"frozen_c1={res_frozen_c1.equity_nd:.4f}"
    )
    assert abs(mwc_inner_ro - mwc_cube2) < 0.01, (
        f"Auto-doubled inner rollout MWC ({mwc_inner_ro:.5f}) drifted from "
        f"cube=2 dead-cube MWC ({mwc_cube2:.5f})"
    )
    # Sanity check that the fixture strongly separates auto-doubled vs frozen cube=1!
    assert abs(res_1ply.equity_nd - res_frozen_c1.equity_nd) > 0.25
    assert abs(mwc_inner_ro - mwc_frozen_c1) > 0.04


def test_inner_truncated_rollout_post_crawford_trailer_on_roll():
    """Trailer (4-away) on roll vs leader (1-away) post-Crawford, cube=1 centered.

    Under ND at move 0, the trailer reaches move 2 with cube=1 centered and must
    still auto-double inside the inner truncated-rollout cube check, keeping ND
    and DT closely aligned (and should_double=True at the root).
    """
    common = dict(
        eval_level="rollout",
        n_trials=72,
        truncation_depth=0,
        decision_ply=1,
        seed=42,
        parallel_threads=2,
        cubeful=True,
    )
    an_1ply = BgBotAnalyzer(**common, cube=TrialEvalConfig(ply=1))
    an_inner_ro = BgBotAnalyzer(
        **common,
        cube=TrialEvalConfig(rollout_trials=18, rollout_depth=1, rollout_ply=1),
    )

    res_1ply = an_1ply.cube_action(
        _SHORT_RACE_TRAILER_ON_ROLL,
        cube_value=1, cube_owner="centered",
        away1=4, away2=1, is_crawford=False,
    )
    res_inner_ro = an_inner_ro.cube_action(
        _SHORT_RACE_TRAILER_ON_ROLL,
        cube_value=1, cube_owner="centered",
        away1=4, away2=1, is_crawford=False,
    )

    assert res_inner_ro.should_double is True
    assert abs(res_inner_ro.equity_nd - res_1ply.equity_nd) < 0.04
    assert abs(res_inner_ro.equity_dt - res_1ply.equity_dt) < 0.04


def test_crawford_inner_truncated_rollout_does_not_auto_double():
    """During the Crawford game (is_crawford=True), inner truncated-rollout cube
    checks must NEVER double."""
    an_1ply = BgBotAnalyzer(
        eval_level="rollout", n_trials=36, truncation_depth=0,
        decision_ply=1, seed=42, parallel_threads=1, cubeful=True,
        cube=TrialEvalConfig(ply=1),
    )
    an_inner_ro = BgBotAnalyzer(
        eval_level="rollout", n_trials=36, truncation_depth=0,
        decision_ply=1, seed=42, parallel_threads=1, cubeful=True,
        cube=TrialEvalConfig(rollout_trials=18, rollout_depth=1, rollout_ply=1),
    )

    res_1ply = an_1ply.cube_action(
        _SHORT_RACE_TRAILER_ON_ROLL,
        cube_value=1, cube_owner="centered",
        away1=4, away2=1, is_crawford=True,
    )
    res_inner_ro = an_inner_ro.cube_action(
        _SHORT_RACE_TRAILER_ON_ROLL,
        cube_value=1, cube_owner="centered",
        away1=4, away2=1, is_crawford=True,
    )

    assert res_inner_ro.should_double is False
    assert res_inner_ro.equity_nd == pytest.approx(res_1ply.equity_nd, abs=1e-6)


def test_target_se_honored_consistently_between_checker_and_cube():
    """cubeful_evaluate_board (cubeful_rollout_position) and cube_decision must
    both honor target_se so checker_play and cube_action stay symmetric."""
    w = WeightConfig.from_model("stage11")
    rs = bgbot_cpp.create_rollout(
        w.strategy_type,
        w.weight_paths_list,
        w.hidden_sizes_list,
        n_trials=36,
        truncation_depth=2,
        decision_ply=1,
        n_threads=2,
        seed=42,
        target_se=1e-12,  # forces max_batches=2 batches
        max_batches=2,
    )
    # Single-batch baseline for comparison (must differ from 2-batch result).
    rs_single = bgbot_cpp.create_rollout(
        w.strategy_type,
        w.weight_paths_list,
        w.hidden_sizes_list,
        n_trials=36,
        truncation_depth=2,
        decision_ply=1,
        n_threads=2,
        seed=42,
        target_se=0.0,
        max_batches=1,
    )

    post_move = _CONTACT_BOARD
    opp_board = flip_board(post_move)

    ce = rs.cubeful_evaluate_board(
        post_move, post_move, cube_value=1, owner=bgbot_cpp.CubeOwner.CENTERED
    )
    ca = rs.cube_decision(
        opp_board, cube_value=1, owner=bgbot_cpp.CubeOwner.CENTERED
    )
    ce_single = rs_single.cubeful_evaluate_board(
        post_move, post_move, cube_value=1, owner=bgbot_cpp.CubeOwner.CENTERED
    )

    # Exact symmetry between cubeful_evaluate_board and -cube_decision under target_se.
    assert ce["cubeful_equity"] == pytest.approx(-ca["optimal_equity"], abs=1e-6)
    # Confirm 2 batches actually ran (distinct from 1-batch single run).
    assert ce["cubeful_se"] < ce_single["cubeful_se"]
