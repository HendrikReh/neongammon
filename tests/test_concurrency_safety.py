# SPDX-License-Identifier: MPL-2.0
"""Concurrency, re-entrancy, and memory-safety tests for C++ and Python paths."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import astuple
from unittest.mock import patch

import bgbot_cpp
import pytest

from neongammon import (
    STARTING_BOARD,
    BgBotAnalyzer,
    WeightConfig,
    flip_board,
    possible_moves,
)
from neongammon.analyzer import load_bearoff_db
from neongammon.weights import bearoff_db_path

_CONTACT_BOARD = [
    1, -2, 0, 0, 2, 2, 3, 0, 3, 0, 0, 2, -3, 2, -1, 0, 0, 0,
    -2, -2, -2, -2, 0, 1, 0, 0,
]

# Near-bearoff boundary position: some 2-ply replies stay outside the 6-point
# bearoff DB while others enter it, exercising partial compaction in
# eval_each_impl across nested BearoffStrategy wrappers.
_NEAR_BEAROFF_BOARD = [
    0, 2, 2, 2, 3, 3, 2, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, -1, -2, -3, -3, -3, -2, -1, 0,
]

_PURE_BEAROFF_BOARD = [
    0, 3, 3, 2, 3, 2, 2, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, -2, -3, -2, -3, -2, -2, 0,
]


def test_full_depth_opponent_3ply_parallel_matches_serial():
    """full_depth_opponent=True at 3-ply must not SIGSEGV or clobber opp_candidates."""
    weights = WeightConfig.from_model("stage11")
    base = bgbot_cpp.create_strategy(
        weights.strategy_type, weights.weight_paths_list, weights.hidden_sizes_list
    )

    serial = bgbot_cpp.create_multipy_from_base(
        base,
        n_plies=3,
        full_depth_opponent=True,
        parallel_evaluate=False,
        parallel_threads=1,
    )
    parallel = bgbot_cpp.create_multipy_from_base(
        base,
        n_plies=3,
        full_depth_opponent=True,
        parallel_evaluate=True,
        parallel_threads=4,
    )

    # Opening 3-1 making the 5-point (8/5 6/5) is a canonical post-move board.
    moves_31 = [list(b) for b in possible_moves(STARTING_BOARD, 3, 1)]
    make_five = [
        0, -2, 0, 0, 0, 2, 4, 0, 2, 0, 0, 0, -5,
        5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0,
    ]
    assert make_five in moves_31

    serial.clear_cache()
    res_serial = serial.evaluate_board(make_five, list(STARTING_BOARD))

    parallel.clear_cache()
    res_parallel = parallel.evaluate_board(make_five, list(STARTING_BOARD))

    assert res_parallel["probs"] == res_serial["probs"]
    assert res_parallel["equity"] == res_serial["equity"]

    # Second call on another candidate to confirm no lingering state corruption.
    other = moves_31[0] if moves_31[0] != make_five else moves_31[1]
    serial.clear_cache()
    res_serial_2 = serial.evaluate_board(other, list(STARTING_BOARD))
    parallel.clear_cache()
    res_parallel_2 = parallel.evaluate_board(other, list(STARTING_BOARD))
    assert res_parallel_2["probs"] == res_serial_2["probs"]
    assert res_parallel_2["equity"] == res_serial_2["equity"]


def _checker_key(res):
    return [(tuple(m.board), m.equity, m.cubeless_equity, astuple(m.probs)) for m in res.moves]


def _cube_key(res):
    return (
        res.equity_nd,
        res.equity_dt,
        res.equity_dp,
        res.should_double,
        res.should_take,
        astuple(res.probs),
    )


def _post_move_key(res):
    return (res.cubeless_equity, res.cubeful_equity, astuple(res.probs))


@pytest.mark.parametrize("level", ["1ply", "2ply", "truncated1"])
def test_concurrent_shared_analyzer_methods_match_serial(level):
    """Concurrent checker_play, cube_action, and post_move_analytics on one
    shared BgBotAnalyzer must complete cleanly without races or drift."""
    post_board = list(possible_moves(STARTING_BOARD, 3, 1)[0])
    analyzer = BgBotAnalyzer(eval_level=level, parallel_threads=2, seed=42)

    want_checker = _checker_key(analyzer.checker_play(STARTING_BOARD, 3, 1))
    want_cube = _cube_key(analyzer.cube_action(_CONTACT_BOARD))
    want_post = _post_move_key(analyzer.post_move_analytics(post_board))

    tasks = ["checker", "cube", "post"] * 4
    barrier = threading.Barrier(len(tasks))

    def worker(kind: str):
        barrier.wait(timeout=10)
        if kind == "checker":
            return kind, _checker_key(analyzer.checker_play(STARTING_BOARD, 3, 1))
        if kind == "cube":
            return kind, _cube_key(analyzer.cube_action(_CONTACT_BOARD))
        return kind, _post_move_key(analyzer.post_move_analytics(post_board))

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        results = list(pool.map(worker, tasks))

    for kind, got in results:
        if kind == "checker":
            assert got == want_checker
        elif kind == "cube":
            assert got == want_cube
        else:
            assert got == want_post


@pytest.mark.parametrize("level", ["1ply", "2ply"])
def test_concurrent_cube_calls_reuse_checker_model(level):
    analyzer = BgBotAnalyzer(eval_level=level, parallel_threads=2)
    barrier = threading.Barrier(8)
    with patch.object(bgbot_cpp, "create_strategy", side_effect=AssertionError("extra model load")):
        def call_cube(_):
            barrier.wait(timeout=10)
            return analyzer.cube_action(STARTING_BOARD)
        with ThreadPoolExecutor(max_workers=8) as pool:
            outputs = list(pool.map(call_cube, range(8)))
    assert all(res == outputs[0] for res in outputs)


def test_parallel_rollout_progress_callback_monotonic_and_serialized():
    """Progress callbacks during parallel rollouts must never overlap across
    threads and must report strictly increasing trial counts."""
    def check_stream(run_fn, expected_batches: int, expected_total: int):
        in_flight = 0
        overlap_detected = False
        guard = threading.Lock()
        seen: list[tuple[int, int]] = []

        def cb(done, total, _partial):
            nonlocal in_flight, overlap_detected
            with guard:
                in_flight += 1
                if in_flight > 1:
                    overlap_detected = True
                seen.append((done, total))
            # Sleep briefly so any concurrent worker without serialization
            # would enter the callback simultaneously.
            time.sleep(0.001)
            with guard:
                in_flight -= 1

        run_fn(cb)
        assert not overlap_detected, "progress callback invoked concurrently from multiple threads"
        assert seen, "expected at least one progress report"
        assert seen.count((expected_total, expected_total)) == expected_batches

        # Split by batch boundary (done resets after reaching expected_total).
        batches: list[list[int]] = [[]]
        for done, total in seen:
            assert total == expected_total
            assert 1 <= done <= expected_total
            batches[-1].append(done)
            if done == expected_total:
                batches.append([])
        batches = [b for b in batches if b]
        assert len(batches) == expected_batches
        for batch in batches:
            assert batch == sorted(batch)
            assert len(set(batch)) == len(batch), f"duplicate/non-monotonic progress: {batch}"
            assert batch[-1] == expected_total

    # 1. Single-batch parallel cubeful cube rollout (cubeful_cube_decision)
    az_single = BgBotAnalyzer(
        eval_level="rollout",
        n_trials=72,
        truncation_depth=5,
        decision_ply=1,
        parallel_threads=4,
        seed=42,
    )
    check_stream(
        lambda cb: az_single.cube_action(_CONTACT_BOARD, progress_callback=cb),
        expected_batches=1,
        expected_total=72,
    )

    # 2. Multi-batch parallel cubeful cube rollout (cubeful_cube_decision_batched)
    az_batched = BgBotAnalyzer(
        eval_level="rollout",
        n_trials=37,
        truncation_depth=5,
        decision_ply=1,
        parallel_threads=4,
        target_se=1e-12,
        max_batches=2,
        seed=42,
    )
    check_stream(
        lambda cb: az_batched.cube_action(_CONTACT_BOARD, progress_callback=cb),
        expected_batches=2,
        expected_total=37,
    )

    # 3. Parallel cubeless rollout (run_trials_parallel)
    az_cubeless = BgBotAnalyzer(
        eval_level="rollout",
        cubeful=False,
        cubeful_trial_moves=False,
        n_trials=72,
        truncation_depth=5,
        decision_ply=1,
        parallel_threads=4,
        seed=42,
    )
    check_stream(
        lambda cb: az_cubeless.post_move_analytics(
            flip_board(_CONTACT_BOARD),
            progress_callback=lambda done, total: cb(done, total, []),
        ),
        expected_batches=1,
        expected_total=72,
    )


def test_double_wrapped_bearoff_strategy_parity():
    """Nested BearoffStrategy layers must match a single BearoffStrategy layer
    across contact, near-bearoff boundary, and pure-bearoff positions."""
    db_path = bearoff_db_path()
    assert db_path is not None
    db = load_bearoff_db(db_path)
    assert db is not None

    weights = WeightConfig.from_model("stage11")
    base = bgbot_cpp.create_strategy(
        weights.strategy_type, weights.weight_paths_list, weights.hidden_sizes_list
    )
    single = bgbot_cpp.BearoffStrategy(base, db)
    double = bgbot_cpp.BearoffStrategy(single, db)

    owner = bgbot_cpp.CubeOwner.CENTERED
    mp_clear = bgbot_cpp.create_multipy_from_base(
        base, n_plies=2, parallel_evaluate=False, parallel_threads=1
    )
    for board in (_CONTACT_BOARD, _NEAR_BEAROFF_BOARD, _PURE_BEAROFF_BOARD):
        for n_plies in (1, 2):
            mp_clear.clear_cache()
            eq_single = bgbot_cpp.cubeful_equity_nply(
                board, owner, single, n_plies, n_threads=2
            )
            probs_single = bgbot_cpp.cubeful_probs_nply(
                board, owner, single, n_plies, n_threads=2
            )

            mp_clear.clear_cache()
            eq_double = bgbot_cpp.cubeful_equity_nply(
                board, owner, double, n_plies, n_threads=2
            )
            probs_double = bgbot_cpp.cubeful_probs_nply(
                board, owner, double, n_plies, n_threads=2
            )

            assert eq_double == eq_single
            assert list(probs_double) == list(probs_single)

    # Also test BearoffStrategy wrapping a MultiPlyStrategy (hits the fallback
    # loop in eval_each_impl and re-enters eval_each_impl recursively).
    mp2 = bgbot_cpp.create_multipy_from_base(
        base, n_plies=2, parallel_evaluate=False, parallel_threads=1
    )
    bgbot_cpp.multipy_set_bearoff_db(mp2, db)
    single_mp = bgbot_cpp.BearoffStrategy(mp2, db)
    double_mp = bgbot_cpp.BearoffStrategy(single_mp, db)

    mp2.clear_cache()
    eq_single_mp = bgbot_cpp.cubeful_equity_nply(
        _NEAR_BEAROFF_BOARD, owner, single_mp, 2, n_threads=1
    )
    mp2.clear_cache()
    eq_double_mp = bgbot_cpp.cubeful_equity_nply(
        _NEAR_BEAROFF_BOARD, owner, double_mp, 2, n_threads=1
    )
    assert eq_double_mp == eq_single_mp

@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("settings", [((20, 15), (1, 1)), ((1, 1), (20, 15))])
def test_legacy_evaluator_reconfigures_reply_prefilter(threads, settings):
    weights = WeightConfig.from_model("stage11")

    def evaluator():
        return bgbot_cpp.create_multipy(
            weights.strategy_type, weights.weight_paths_list, weights.hidden_sizes_list,
            n_plies=3 if threads > 1 else 2, full_depth_opponent=True, parallel_evaluate=threads > 1,
            parallel_threads=threads,
        )

    reused, cold = evaluator(), evaluator()
    boards = possible_moves(STARTING_BOARD, 3, 1)[:2]
    first, second = settings
    reused.set_prefilter_params(*first)
    before = [reused.evaluate_board(b, STARTING_BOARD) for b in boards]
    reused.set_prefilter_params(*second)
    after = [reused.evaluate_board(b, STARTING_BOARD) for b in boards]
    cold.set_prefilter_params(*second)
    assert after == [cold.evaluate_board(b, STARTING_BOARD) for b in boards]
    assert before != after


@pytest.mark.parametrize("operation", ["weights", "single_board", "single_cube", "recursive_batch"])
def test_python_thread_can_run_during_each_native_operation(operation):
    import sys
    from neongammon import BatchModels, batch_cube_action_nply

    weights = WeightConfig.from_model("stage11")
    args = (weights.strategy_type, weights.weight_paths_list, weights.hidden_sizes_list)
    models = BatchModels(weights, bearoff_db=False)
    base = models.base()
    owner = bgbot_cpp.CubeOwner.CENTERED
    operations = {
        "weights": lambda: bgbot_cpp.create_strategy(*args),
        "single_board": lambda: base.evaluate_board(STARTING_BOARD, STARTING_BOARD),
        "single_cube": lambda: bgbot_cpp.evaluate_cube_decision_unified(
            STARTING_BOARD, 1, owner, *args, strategy=base),
        "recursive_batch": lambda: batch_cube_action_nply(
            [dict(board=STARTING_BOARD)], models=models, decision_only=True),
    }
    evaluate = operations[operation]
    ready, start, ran = threading.Event(), threading.Event(), threading.Event()

    def observer():
        ready.set()
        start.wait()
        ran.set()

    observer_thread = threading.Thread(target=observer)
    observer_thread.start()
    ready.wait()
    old_interval = sys.getswitchinterval()
    try:
        # Disable bytecode-timeslice handoffs. Only the native call can let
        # the waiting observer run before this loop returns to normal scheduling.
        sys.setswitchinterval(10.0)
        start.set()
        for _ in range(500):
            evaluate()
            if ran.is_set():
                break
        assert ran.is_set(), operation
    finally:
        sys.setswitchinterval(old_interval)
        observer_thread.join()
