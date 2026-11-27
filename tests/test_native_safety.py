# SPDX-License-Identifier: MPL-2.0
"""Regression tests for native C++/pybind11 safety and lifetime edges (#118)."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

import bgbot_cpp
from neongammon import STARTING_BOARD, flip_board
from neongammon.weights import bearoff_db_path, default_weights


def _load_base():
    w = default_weights()
    return bgbot_cpp.create_strategy(w.strategy_type, w.weight_paths_list, w.hidden_sizes_list)


def test_non_contiguous_numpy_arrays_match_contiguous():
    b0 = np.asarray(STARTING_BOARD, dtype=np.int32)
    b1 = np.asarray(flip_board(STARTING_BOARD), dtype=np.int32)
    b2 = b0.copy()
    b2[6] -= 1
    b2[5] += 1
    b2[8] -= 1
    b2[5] += 1
    arr = np.stack([b0, b1, b2], axis=0)
    rev = arr[::-1]  # negative row stride, non-C-contiguous
    assert not rev.flags["C_CONTIGUOUS"]
    rev_contig = np.ascontiguousarray(rev)

    for n_inputs in (196, 244):
        got = bgbot_cpp.encode_boards_batch(rev, n_inputs)
        want = bgbot_cpp.encode_boards_batch(rev_contig, n_inputs)
        np.testing.assert_array_equal(got, want)

    np.testing.assert_array_equal(
        bgbot_cpp.classify_game_plans_batch(rev),
        bgbot_cpp.classify_game_plans_batch(rev_contig),
    )
    got_p, got_o = bgbot_cpp.classify_game_plan_pairs_batch(rev)
    want_p, want_o = bgbot_cpp.classify_game_plan_pairs_batch(rev_contig)
    np.testing.assert_array_equal(got_p, want_p)
    np.testing.assert_array_equal(got_o, want_o)


@pytest.mark.parametrize("bad_n_inputs", [0, -1, 245, 512])
def test_encode_boards_batch_rejects_out_of_range_n_inputs(bad_n_inputs):
    arr = np.asarray([STARTING_BOARD], dtype=np.int32)
    with pytest.raises(ValueError, match="n_inputs"):
        bgbot_cpp.encode_boards_batch(arr, bad_n_inputs)


def test_none_strategy_arguments_raise_value_error_not_segfault():
    base = _load_base()
    owner = bgbot_cpp.CubeOwner.CENTERED
    pos_pre = [(list(STARTING_BOARD), 1, owner)]
    pos_post = [(list(STARTING_BOARD), owner)]
    checker_row = [{
        "board": list(STARTING_BOARD), "die1": 3, "die2": 1,
        "cube_value": 1, "cube_owner": owner,
    }]

    with pytest.raises((ValueError, TypeError)):
        bgbot_cpp.NNStrategy(None)
    with pytest.raises((ValueError, TypeError)):
        bgbot_cpp.BearoffStrategy(None, None)
    with pytest.raises(ValueError, match="None"):
        bgbot_cpp.create_multipy_from_base(None)
    with pytest.raises(ValueError, match="None"):
        bgbot_cpp.create_rollout_from_base(None)
    with pytest.raises(ValueError, match="None"):
        bgbot_cpp.cubeful_equity_nply(list(STARTING_BOARD), owner, None)
    with pytest.raises(ValueError, match="None"):
        bgbot_cpp.cubeful_probs_nply(list(STARTING_BOARD), owner, None)
    with pytest.raises(ValueError, match="None"):
        bgbot_cpp.cubeful_probs_and_equity_nply(list(STARTING_BOARD), owner, None)
    with pytest.raises(ValueError, match="None"):
        bgbot_cpp.checker_play_nply(list(STARTING_BOARD), 3, 1, base, None)
    with pytest.raises(ValueError, match="None"):
        bgbot_cpp.batch_checker_play(checker_row, base, None)
    with pytest.raises(ValueError, match="None"):
        bgbot_cpp.batch_cube_action_nply([(list(STARTING_BOARD), bgbot_cpp.CubeInfo())], None, None, 2)
    with pytest.raises((ValueError, TypeError)):
        bgbot_cpp.batch_evaluate_positions(pos_pre, None)
    with pytest.raises((ValueError, TypeError)):
        bgbot_cpp.batch_evaluate_post_move(pos_post, None)

    ss = bgbot_cpp.ScenarioSet()
    with pytest.raises(ValueError, match="None"):
        bgbot_cpp.score_benchmarks_multipy(ss, None)
    with pytest.raises(ValueError, match="None"):
        bgbot_cpp.score_benchmarks_rollout(ss, None)
    with pytest.raises(ValueError, match="None"):
        bgbot_cpp.play_games_multipy_vs_pubeval(None, 1)
    with pytest.raises(ValueError, match="None"):
        bgbot_cpp.score_benchmark_pr_strategy([], None)


def test_multipy_set_bearoff_db_detach_swap_and_lifetime():
    db_path = bearoff_db_path()
    assert db_path is not None
    base = _load_base()
    mp = bgbot_cpp.create_multipy_from_base(base, n_plies=2, parallel_evaluate=False)
    mp.clear_cache()

    # Bearoff post-move board where DB and NN differ noticeably.
    bearoff = [0] * 26
    bearoff[1], bearoff[2], bearoff[24], bearoff[23] = 2, 1, -2, -1

    nn_probs = list(mp.evaluate_board(bearoff, bearoff)["probs"])

    db1 = bgbot_cpp.BearoffDB()
    assert db1.load(db_path)
    db1_ref = weakref.ref(db1)

    bgbot_cpp.multipy_set_bearoff_db(mp, db1)
    db_probs = list(mp.evaluate_board(bearoff, bearoff)["probs"])
    assert db_probs != nn_probs

    # Keep-alive: deleting local db1 reference must keep db1 alive while mp lives.
    del db1
    gc.collect()
    assert db1_ref() is not None
    assert list(mp.evaluate_board(bearoff, bearoff)["probs"]) == db_probs

    # Detaching with None must restore NN evaluation without explicit clear_cache().
    bgbot_cpp.multipy_set_bearoff_db(mp, None)
    detached_probs = list(mp.evaluate_board(bearoff, bearoff)["probs"])
    assert detached_probs == nn_probs

    # Re-attach loaded DB, then swap to an unloaded DB -> must not keep loaded DB.
    db2 = bgbot_cpp.BearoffDB()
    assert db2.load(db_path)
    bgbot_cpp.multipy_set_bearoff_db(mp, db2)
    assert list(mp.evaluate_board(bearoff, bearoff)["probs"]) == db_probs
    db_unloaded = bgbot_cpp.BearoffDB()
    bgbot_cpp.multipy_set_bearoff_db(mp, db_unloaded)
    assert list(mp.evaluate_board(bearoff, bearoff)["probs"]) == nn_probs


def test_rollout_set_bearoff_db_keeps_db_alive():
    base = _load_base()
    ro = bgbot_cpp.create_rollout_from_base(base, n_trials=4, truncation_depth=2, n_threads=1)
    db = bgbot_cpp.BearoffDB()
    db_ref = weakref.ref(db)
    bgbot_cpp.rollout_set_bearoff_db(ro, db)
    del db
    gc.collect()
    assert db_ref() is not None
    del ro
    gc.collect()
    assert db_ref() is None


def test_routed_strategy_small_manifest_and_root_pin_validation(tmp_path):
    race_w = str(tmp_path / "race.weights")
    contact_w = str(tmp_path / "contact.weights")
    bgbot_cpp.NeuralNetwork(4, 196, 0.1, 1).save_weights(race_w)
    bgbot_cpp.NeuralNetwork(4, 244, 0.1, 2).save_weights(contact_w)

    routing = bgbot_cpp.RoutingManifest()
    routing.contact_pairs = [1] * 16
    routed = bgbot_cpp.RoutedStrategy([race_w, contact_w], [4, 4], routing)

    assert routed.select_nn_idx(list(STARTING_BOARD)) == 1
    res = routed.evaluate_board(list(STARTING_BOARD), list(STARTING_BOARD))
    assert len(res["probs"]) == 5

    for bad_slot in (-1, 0, 2, 10):
        with pytest.raises(ValueError, match="non-race network slot"):
            routed.set_root_pinned([bad_slot])

    routed.set_root_pinned([1])
    assert routed.root_pinned() == [1]
