# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Rollout candidate filtering, execution, progress, and cancellation."""

from __future__ import annotations

import threading

from ._native import load_native
from ._primitives import cubeless_equity
from .runtime import _notify_thread_budget
from .strategies import (
    _CubelessBase,
    _cube_limit_kwargs,
    _default_parallel_threads,
    resolve_owner,
)

bgbot_cpp = load_native()


class RolloutCancelled(Exception):
    """Raised when a rollout is cancelled via cancel()."""
    pass


def _rollout_eval_candidate(
    rollout_analyzer: _RolloutAnalyzer,
    cand_board,
    ref_board,
    *,
    owner=None,
    cube_value: int = 1,
    away1: int = 0,
    away2: int = 0,
    is_crawford: bool = False,
    jacoby: bool = True,
    beaver: bool = True,
    max_cube_value: int = 0,
    progress=None,
) -> dict:
    """Evaluate one post-move candidate board via the rollout strategy.

    Checks cancellation before launching native trials, translates
    ``bgbot_cpp.RolloutCancelled`` to :class:`RolloutCancelled`, and overrides
    Monte Carlo probabilities with exact bearoff DB values when available.
    """
    rollout_analyzer._check_cancel()
    use_cubeful = owner is not None
    _notify_thread_budget(
        "rollout_evaluate", rollout_analyzer._parallel_threads, cubeful=use_cubeful,
    )
    try:
        if use_cubeful:
            r = rollout_analyzer._rollout_strategy.cubeful_evaluate_board(
                cand_board, ref_board,
                cube_value=cube_value, owner=owner,
                away1=away1, away2=away2, is_crawford=is_crawford,
                jacoby=jacoby, beaver=beaver, **_cube_limit_kwargs(max_cube_value),
                progress=progress,
            )
        else:
            r = rollout_analyzer._rollout_strategy.evaluate_board(
                cand_board, ref_board, progress,
            )
    except bgbot_cpp.RolloutCancelled:
        raise RolloutCancelled()

    probs = list(r["probs"])
    if rollout_analyzer._bearoff_db is not None and rollout_analyzer._bearoff_db.is_bearoff(cand_board):
        probs = rollout_analyzer._bearoff_db.lookup_probs(cand_board, post_move=True)

    entry = {
        "board": cand_board,
        "equity": r["equity"],          # cubeless equity
        "probs": probs,
        "std_error": r.get("std_error") or 0,
        "prob_std_errors": list(r.get("prob_std_errors", [0] * 5)),
        "eval_level": "Rollout",
    }
    if use_cubeful:
        # cubeful_evaluate_board returns rollout-level cubeful that already
        # accounts for opp's optimal cube action — store it so _CubefulAnalyzer
        # uses it directly instead of recomputing.
        entry["rollout_cubeful_equity"] = r["cubeful_equity"]
        entry["rollout_cubeful_se"] = r["cubeful_se"]
    return entry


class _RolloutAnalyzer(_CubelessBase):

    FILTER_MAX_MOVES = 5
    FILTER_THRESHOLD = 0.08

    def __init__(
        self, models, n_trials=1296, truncation_depth=0,
        decision_ply=1, truncation_ply=-1, n_threads=0, seed=42,
        late_ply=-1, late_threshold=20,
        parallelize_trials=True,
        checker=None, checker_late=None,
        cube=None, cube_late=None,
        ultra_late_threshold=9999,
        cubeful_trial_moves=True,
        cubeful_late_threshold=0,
        prefilter_threshold=0.0,
        target_se=0.0,
        max_batches=50,
        filter_max_moves=None,
        filter_threshold=None,
    ):
        super().__init__(models)
        # Candidate-selection filter for checker_play_analytics: how many of
        # the top scored moves get rolled out (the rest keep their filter-stage
        # equity). Instance attrs shadow the TINY class defaults when a wider
        # search interval is requested.
        if filter_max_moves is not None:
            self.FILTER_MAX_MOVES = int(filter_max_moves)
        if filter_threshold is not None:
            self.FILTER_THRESHOLD = float(filter_threshold)
        requested_threads = n_threads if n_threads > 0 else _default_parallel_threads()
        self._parallel_threads = max(1, requested_threads)

        # Two-stage candidate filter for checker_play_analytics: when
        # prefilter_threshold > 0, stage 1 is a loose 1-ply cull at that
        # threshold (no max_moves cap), stage 2 is the standard TINY filter
        # at 2-ply on stage 1's survivors. Survivors of stage 2 are rolled
        # out. When 0, the legacy single-stage 1-ply TINY filter is used.
        self._prefilter_threshold = float(prefilter_threshold)
        self._strategy_2ply = None
        if self._prefilter_threshold > 0:
            self._strategy_2ply = bgbot_cpp.create_multipy_from_base(
                self._raw_base,
                n_plies=2,
                parallel_evaluate=True,
                parallel_threads=self._parallel_threads,
            )
        self._parallelize_trials = bool(parallelize_trials)
        self._rollout_config = {
            "n_trials": n_trials,
            "truncation_depth": truncation_depth,
            "decision_ply": decision_ply,
            "truncation_ply": truncation_ply,
            "n_threads": self._parallel_threads,
            "seed": seed,
            "late_ply": late_ply,
            "late_threshold": late_threshold,
        }
        self._cancel_event = threading.Event()

        # Convert None to default TrialEvalConfig (cube_late is accepted for
        # compatibility only; cube decisions use `cube` throughout the trial).
        del cube_late
        _empty = bgbot_cpp.TrialEvalConfig()
        checker_cfg = checker if checker is not None else _empty
        checker_late_cfg = checker_late if checker_late is not None else _empty
        cube_cfg = cube if cube is not None else _empty

        self._rollout_strategy = bgbot_cpp.create_rollout_from_base(
            self._raw_base,
            n_trials=n_trials,
            truncation_depth=truncation_depth,
            decision_ply=decision_ply,
            truncation_ply=truncation_ply,
            n_threads=self._parallel_threads,
            seed=seed,
            late_ply=late_ply,
            late_threshold=late_threshold,
            parallelize_trials=parallelize_trials,
            checker=checker_cfg,
            checker_late=checker_late_cfg,
            cube=cube_cfg,
            ultra_late_threshold=ultra_late_threshold,
            cubeful_trial_moves=cubeful_trial_moves,
            cubeful_late_threshold=cubeful_late_threshold,
            target_se=target_se,
            max_batches=max_batches,
        )

    def set_seed(self, seed):
        """Reseed the rollout RNG (for independent seeded batches; keeps the
        SharedPosCache warm). Clears the strategy's cached stratified dice."""
        self._rollout_strategy.set_seed(int(seed))

    def cancel(self):
        """Request cancellation of in-progress rollout."""
        self._cancel_event.set()
        self._rollout_strategy.cancel()

    def reset_cancel(self):
        """Clear cancellation flag for reuse."""
        self._cancel_event.clear()
        self._rollout_strategy.reset_cancel()

    def _check_cancel(self):
        """Raise RolloutCancelled if cancellation was requested."""
        if self._cancel_event.is_set():
            raise RolloutCancelled()

    def checker_play_analytics(
        self, board, die1, die2, cube_value=1, cube_owner="centered",
        progress_callback=None,
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True, max_cube_value=0,
        force_boards=None, finalize_progress=None,
    ) -> list[dict]:
        # Only the cubeful wrapper runs the finalizing (promotion) phase, so
        # there is nothing here to report against ``finalize_progress``.
        del finalize_progress
        candidates = bgbot_cpp.possible_moves(board, die1, die2)
        if not candidates:
            return []

        scored_1ply = self._score_candidates(
            candidates, board, cube_owner,
            cube_value=cube_value, away1=away1, away2=away2,
            is_crawford=is_crawford, jacoby=jacoby, max_cube_value=max_cube_value,
        )

        # Stage 1: 1-ply filter. If prefilter_threshold > 0, this is a loose
        # cull (no max_moves cap) that just drops obvious garbage; stage 2 at
        # 2-ply then applies the TINY filter. Otherwise it's a single-stage
        # TINY filter at 1-ply.
        if self._prefilter_threshold > 0:
            stage1_survivors, _ = self._filter_candidates(
                scored_1ply, self._prefilter_threshold, max_moves=len(scored_1ply)
            )
        else:
            stage1_survivors = scored_1ply

        # Stage 2: 2-ply rescore + TINY filter (only when prefilter is on and
        # more than one candidate survived stage 1).
        scored_2ply = None
        scored_2ply_set: set = set()
        if self._prefilter_threshold > 0 and len(stage1_survivors) > 1:
            stage1_boards = [item[2] for item in stage1_survivors]
            scored_2ply = self._score_candidates(
                stage1_boards, board, cube_owner,
                cube_value=cube_value, away1=away1, away2=away2,
                is_crawford=is_crawford, jacoby=jacoby, max_cube_value=max_cube_value,
                strategy=self._strategy_2ply,
            )
            scored_2ply_set = {tuple(item[2]) for item in scored_2ply}
            survivors, survivor_set = self._filter_candidates(
                scored_2ply, self.FILTER_THRESHOLD, self.FILTER_MAX_MOVES
            )
            self._strategy_2ply.clear_cache()
        else:
            survivors, survivor_set = self._filter_candidates(
                stage1_survivors, self.FILTER_THRESHOLD, self.FILTER_MAX_MOVES
            )

        # Ensure at least 2 candidates get rolled out so the cubeful sort
        # always has at least two rollout-quality entries to compare.
        fallback_pool = scored_2ply if scored_2ply is not None else scored_1ply
        if len(survivors) < 2 and len(fallback_pool) >= 2:
            for item in fallback_pool:
                if tuple(item[2]) not in survivor_set:
                    survivors.append(item)
                    survivor_set.add(tuple(item[2]))
                    if len(survivors) >= 2:
                        break
        # Force-include caller-specified boards (e.g. the move actually played
        # in an expert re-eval) so they're rolled out even when the filter
        # dropped them. Pulls from the same pool survivors are drawn from so the
        # forced entry is rolled out (not left at its 2-ply / 1-ply fallback).
        self._force_include(fallback_pool, survivors, survivor_set, force_boards)
        # A forced board the stage-1 prefilter culled is not in that pool at
        # all (scored_2ply holds only stage-1 survivors), so it was silently
        # skipped — measured 2026-09-02 on the candidate-completion run: 134 of
        # 912 forced boards never rolled. Score those at the pool's ply and add
        # them, so "forced" means rolled out, whatever the filter thought.
        if force_boards:
            legal = {tuple(b) for b in candidates}
            missing = [list(b) for b in {tuple(fb) for fb in force_boards}
                       if b in legal and b not in survivor_set]
            if missing:
                scored_missing = self._score_candidates(
                    missing, board, cube_owner,
                    cube_value=cube_value, away1=away1, away2=away2,
                    is_crawford=is_crawford, jacoby=jacoby, max_cube_value=max_cube_value,
                    **({"strategy": self._strategy_2ply}
                       if scored_2ply is not None else {}),
                )
                for item in scored_missing:
                    survivors.append(item)
                    survivor_set.add(tuple(item[2]))
        self._check_cancel()

        n_trials = self._rollout_config["n_trials"]
        results = []
        total_moves = len(survivors)
        # When cube info is present, use cubeful_evaluate_board: the C++
        # function returns rollout-level cubeful equity that already
        # incorporates the opponent's optimal cube action at the start of
        # their turn (it delegates to cubeful_cube_decision on the flipped
        # opp position and collapses ND/DT/DP into opp's optimal). Cubeless
        # probs/equity come from the same trials, already inverted to SP's
        # perspective at the post-move board `b`. No perspective fiddling
        # needed at the python layer.
        owner = resolve_owner(cube_owner) if cube_owner is not None else None
        for i, (feq, cleq, b, p0) in enumerate(survivors):
            # Trial-level progress callback: maps trial progress within
            # the current move to overall progress across all moves.
            _trial_progress = None
            if progress_callback:
                def _trial_progress(completed_trials, total_trials, _move_idx=i):
                    overall = _move_idx * n_trials + completed_trials
                    overall_total = total_moves * n_trials
                    progress_callback(overall, overall_total, results)

            results.append(_rollout_eval_candidate(
                self, b, board,
                owner=owner, cube_value=cube_value,
                away1=away1, away2=away2, is_crawford=is_crawford,
                jacoby=jacoby, beaver=beaver, max_cube_value=max_cube_value,
                progress=_trial_progress,
            ))

        # Non-rolled-out moves: those that survived stage 1 (1-ply prefilter)
        # but failed stage 2 (2-ply TINY) get their 2-ply equity; those that
        # failed stage 1 keep their 1-ply equity. When prefilter is off, all
        # non-survivors get their 1-ply equity (legacy single-stage path).
        if scored_2ply is not None:
            for feq, cleq, b, p in scored_2ply:
                if tuple(b) not in survivor_set:
                    results.append({
                        "board": b,
                        "equity": cleq,
                        "probs": p,
                        "eval_level": "2-ply",
                    })
        for feq, cleq, b, p in scored_1ply:
            t = tuple(b)
            if t in survivor_set or t in scored_2ply_set:
                continue
            results.append({
                "board": b,
                "equity": cleq,
                "probs": p,
                "eval_level": "1-ply",
            })

        return self._finalize_results(results)

    def cube_action_analytics(
        self, board, cube_value=1, cube_owner="centered",
        progress_callback=None,
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True, max_cube_value=0,
        incl_2ply_details=False,
    ) -> dict:
        self._check_cancel()
        owner = resolve_owner(cube_owner)

        # Wire trial-level progress for cube rollout
        def _cube_trial_progress(completed_trials, total_trials):
            if progress_callback:
                progress_callback(completed_trials, total_trials, [])

        try:
            _notify_thread_budget(
                "rollout_cube_decision", self._parallel_threads, cubeful=True,
            )
            r = self._rollout_strategy.cube_decision(
                board, cube_value, owner,
                away1=away1, away2=away2, is_crawford=is_crawford,
                jacoby=jacoby, beaver=beaver, **_cube_limit_kwargs(max_cube_value),
                progress=_cube_trial_progress if progress_callback else None,
            )
        except bgbot_cpp.RolloutCancelled:
            raise RolloutCancelled()
        result = self._format_cube_result(r, eval_level="Rollout")

        # Override cubeless probs with exact bearoff DB values when available.
        # The rollout's cubeful equities (ND/DT/DP) are still valuable, but
        # the cubeless probs from Monte Carlo are noisy — the DB is exact.
        if self._bearoff_db is not None and self._bearoff_db.is_bearoff(board):
            pre_roll = self._bearoff_db.lookup_probs(board, post_move=False)
            result["probs"] = pre_roll
            result["cubeless_equity"] = cubeless_equity(pre_roll)

        return result
