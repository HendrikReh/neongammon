# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Cubeful checker-play scoring for fixed-depth searches and rollouts."""

from __future__ import annotations

# Kept for tests/integrations that patch neongammon.cubeful.ThreadPoolExecutor
# to verify that cubeful checker play spawns zero Python thread pools.
from concurrent.futures import ThreadPoolExecutor  # noqa: F401

from ._native import load_native
from .rollout import _RolloutAnalyzer, _rollout_eval_candidate
from .runtime import _notify_thread_budget

bgbot_cpp = load_native()
from .strategies import (
    _FLIP_OWNER,
    _CubelessBase,
    _MultiPlyAnalyzer,
    _OnePlyAnalyzer,
    _contact_cube_life,
    _contact_cube_slope,
    _cube_limit_kwargs,
    _extension_constant,
    _janowski_1ply_equity,
    _move_order_key,
    resolve_owner,
)


#: Cubeful N-ply depths at which checker play searches only the 1-ply move
#: filter's survivors (plus promotions) through the cube-aware tree; at other
#: depths every legal move is scored there. Measured 2026-11-19: at 2-ply the
#: filter takes 2.2-2.9 PR off the snake folder under XG and both rollout
#: references and leaves the money benchmark flat (checker -0.016 +- 0.022);
#: at 3-ply it costs the money benchmark +0.044 +- 0.011 checker PR, so 3-ply
#: keeps the full search.
CUBEFUL_FILTERED_PLIES: frozenset[int] = frozenset(_extension_constant("CUBEFUL_FILTERED_PLIES"))

# Capability observed by integrations that verify Python candidate pools.
SERIAL_CUBEFUL_1PLY = True


class _CubefulAnalyzer:
    """Cubeful wrapper around any cubeless analyzer."""

    def __init__(self, inner: _CubelessBase):
        self._inner = inner
        self._weights = inner._weights
        if isinstance(inner, _MultiPlyAnalyzer):
            self._cubeful_ply = inner._n_plies
        elif isinstance(inner, _RolloutAnalyzer):
            # Use the rollout's decision_ply for cubeful equity per-move.
            # This gives N-ply cubeful evaluation matching the rollout's strength,
            # rather than falling back to crude 1-ply Janowski.
            dp = inner._rollout_config["decision_ply"]
            self._cubeful_ply = max(dp, 1)
        else:
            self._cubeful_ply = 1

    def set_seed(self, seed):
        """Delegate reseeding to the wrapped (rollout) analyzer if supported."""
        if hasattr(self._inner, "set_seed"):
            self._inner.set_seed(seed)

    def _cubeful_equity(
        self, post_move_board, probs, owner,
        cube_value=1, away1=0, away2=0, is_crawford=False, jacoby=True,
        beaver=True, max_cube_value=0, ply=None,
    ) -> float:
        """Cubeful equity of a post-move board at ``ply`` (default: the
        analyzer's cubeful depth); ``ply=1`` is the 1-ply Janowski conversion."""
        ply = self._cubeful_ply if ply is None else ply
        if ply == 1:
            return _janowski_1ply_equity(
                probs, post_move_board, owner,
                cube_value=cube_value, away1=away1, away2=away2,
                is_crawford=is_crawford, jacoby=jacoby,
                max_cube_value=max_cube_value,
            )
        is_match = away1 > 0 or away2 > 0
        opp_pre_roll = bgbot_cpp.flip_board(post_move_board)
        opp_owner = _FLIP_OWNER[owner]
        db = getattr(self._inner, '_bearoff_db', None)
        n_threads = getattr(self._inner, '_parallel_threads', 1)
        _notify_thread_budget(
            "cubeful_equity_nply", n_threads or 1, plies=ply, cubeful=True,
        )
        if is_match:
            opp_eq = bgbot_cpp.cubeful_equity_nply(
                opp_pre_roll, opp_owner,
                self._inner._strategy_1ply, ply,
                n_threads=n_threads,
                cube_value=cube_value,
                away1=away2, away2=away1, is_crawford=is_crawford,
                jacoby=jacoby, beaver=beaver, **_cube_limit_kwargs(max_cube_value),
                contact_x=_contact_cube_life(ply, away1, away2),
                contact_x_pip_slope=_contact_cube_slope(ply, away1, away2),
                bearoff_db=db,
            )
        else:
            opp_eq = bgbot_cpp.cubeful_equity_nply(
                opp_pre_roll, opp_owner,
                self._inner._strategy_1ply, ply,
                n_threads=n_threads,
                cube_value=cube_value,
                jacoby=jacoby, beaver=beaver, **_cube_limit_kwargs(max_cube_value),
                contact_x=_contact_cube_life(ply, away1, away2),
                contact_x_pip_slope=_contact_cube_slope(ply, away1, away2),
                bearoff_db=db,
            )
        return -opp_eq

    def _uses_cube_aware_tree(self, owner) -> bool:
        """Whether checker play scores candidates in the N-ply cube-aware tree."""
        return (
            isinstance(self._inner, _MultiPlyAnalyzer)
            and self._cubeful_ply > 1
            and owner is not None
        )

    def _checker_play_native(self, board, die1, die2, owner, cube_value, away1, away2,
                             is_crawford, jacoby, beaver, max_cube_value, force_boards):
        """N-ply cubeful checker play in the native routine batch_checker_play
        runs per position (bgbot_cpp.checker_play_nply): at the depths in
        CUBEFUL_FILTERED_PLIES the 1-ply filter (plus ``force_boards``), the
        cubeless search of its survivors with promotion, the cube-aware tree
        for the searched moves and 1-ply Janowski for the rest, then the
        cubeful promotion; at other depths every legal move goes through the
        cube-aware tree. The cube-tree calls are spread over the analyzer's
        thread budget.

        Why the filter at 2-ply: scoring every legal move in the cube-aware
        2-ply tree let the pick fall on moves the filter had rejected (snake
        folder, 2026-11-19: 103 of 795 picks, 0.04-0.06 worse under XG and
        both rollout references). At 3-ply searching every move pays.
        """
        inner = self._inner
        ply = self._cubeful_ply
        n_threads = getattr(inner, "_parallel_threads", 1) or 1
        _notify_thread_budget(
            "checker_play_nply", n_threads, plies=ply, cubeful=True,
        )
        moves = bgbot_cpp.checker_play_nply(
            list(board), die1, die2, inner._strategy_1ply, inner._strategy_nply,
            cube_value=cube_value, cube_owner=owner,
            away1=away1, away2=away2, is_crawford=is_crawford,
            max_cube_value=max_cube_value or 0,
            force_boards=[list(b) for b in force_boards or ()],
            filter_max_moves=inner.FILTER_MAX_MOVES,
            filter_threshold=inner.FILTER_THRESHOLD,
            jacoby=jacoby, beaver=beaver,
            contact_x_1ply=_contact_cube_life(1, away1, away2),
            contact_x_pip_slope_1ply=_contact_cube_slope(1, away1, away2),
            contact_x_nply=_contact_cube_life(ply, away1, away2),
            contact_x_pip_slope_nply=_contact_cube_slope(ply, away1, away2),
            cubeful_filtered_plies=sorted(CUBEFUL_FILTERED_PLIES),
            n_threads=n_threads,
        )
        for m in moves:
            m["probs"] = list(m["probs"])
        return moves

    def _checker_play_1ply_native(self, board, die1, die2, owner, cube_value,
                                  away1, away2, is_crawford, jacoby, beaver,
                                  max_cube_value):
        """1-ply cubeful checker play in the single-threaded native routine
        shared with batch_checker_play and 2-ply's 1-ply filter
        (bgbot_cpp.batch_checker_play(..., n_threads=1) -> score_checker_1ply).

        Keeps SERIAL_CUBEFUL_1PLY intact (zero thread pools spawned) while
        batching candidate NN evaluations in BLAS and guaranteeing exact
        bit-for-bit parity with batch_checker_play("1ply") and pruned 1-ply
        moves in 2-ply checker play.
        """
        inner = self._inner
        pos = {
            "board": list(board),
            "die1": die1,
            "die2": die2,
            "cube_value": cube_value,
            "cube_owner": owner,
        }
        if away1 > 0 or away2 > 0:
            pos["away1"] = away1
            pos["away2"] = away2
            pos["is_crawford"] = is_crawford
        if max_cube_value:
            pos["max_cube_value"] = max_cube_value
        _notify_thread_budget("checker_play_1ply", 1, plies=1, cubeful=True)
        moves = bgbot_cpp.batch_checker_play(
            [pos], inner._strategy_1ply,
            n_threads=1,
            jacoby=jacoby, beaver=beaver,
            contact_x_1ply=_contact_cube_life(1, away1, away2),
            contact_x_pip_slope_1ply=_contact_cube_slope(1, away1, away2),
        )[0]["moves"]
        for m in moves:
            m["probs"] = list(m["probs"])
        return moves

    def checker_play_analytics(
        self, board, die1, die2, cube_value=1, cube_owner="centered",
        progress_callback=None,
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True, max_cube_value=0,
        force_boards=None, finalize_progress=None,
    ) -> list[dict]:
        owner = resolve_owner(cube_owner)
        inner = self._inner

        if self._uses_cube_aware_tree(owner):
            return self._checker_play_native(
                board, die1, die2, owner, cube_value, away1, away2, is_crawford,
                jacoby, beaver, max_cube_value, force_boards)

        if isinstance(inner, _OnePlyAnalyzer) and owner is not None:
            return self._checker_play_1ply_native(
                board, die1, die2, owner, cube_value, away1, away2, is_crawford,
                jacoby, beaver, max_cube_value)

        if not isinstance(inner, _RolloutAnalyzer):
            raise TypeError(f"Unsupported inner analyzer: {type(inner).__name__}")

        results = inner.checker_play_analytics(
            board, die1, die2, cube_value, cube_owner, progress_callback,
            away1=away1, away2=away2, is_crawford=is_crawford, jacoby=jacoby, beaver=beaver, max_cube_value=max_cube_value,
            force_boards=force_boards,
        )
        if not results:
            return results

        # Rollout path: survivors carry rollout-level cubeful equity from
        # cube_decision (set in _RolloutAnalyzer.checker_play_analytics).
        # Non-survivors fall back to N-ply cubeful_equity_nply at the
        # rollout's decision_ply — faster than running cube_decision per
        # candidate, but at a lower ply than the rollout itself, so the
        # cubeful values may disagree with the rollout-level cube action
        # (e.g. 3-ply says D/T where 3T says D/P).
        for m in results:
            cubeless_eq = m["equity"]
            if "rollout_cubeful_equity" in m:
                cf_eq = m["rollout_cubeful_equity"]
            else:
                cf_eq = self._cubeful_equity(
                    m["board"], m["probs"], owner,
                    cube_value=cube_value, away1=away1, away2=away2,
                    is_crawford=is_crawford, jacoby=jacoby, beaver=beaver, **_cube_limit_kwargs(max_cube_value),
                )
            m["cubeless_equity"] = cubeless_eq
            m["equity"] = cf_eq

        results.sort(key=_move_order_key)

        # Promote non-rollout entries that rank above any rollout entry,
        # so the top of the displayed list is at rollout-level cubeful and
        # the symmetry checker_play.best.cubeful == -cube_action.optimal
        # holds (required for parity with the multi-ply path that uses
        # cubeful_equity_nply uniformly across survivors and non-survivors).
        # Pathological positions where many candidates flip cube action
        # between decision_ply and rollout-level can trigger many extra
        # rollouts here — that's the cost of getting the displayed best
        # right at the rollout level.
        inner_ro = inner  # _RolloutAnalyzer
        # The trial loop's denominator (survivors x n_trials) was fixed
        # before any promotion, so it cannot cover the extra rollouts below
        # — and promotions are discovered one at a time, so no honest
        # percentage exists here. Keep reporting past that total instead:
        # ``completed > total`` is the caller's signal that the planned
        # trials are done and we are finalizing the best move. Without this
        # the caller sees a legitimate 100% and then silence for however
        # long the promotions take.
        #
        # Each promotion IS a full rollout, though, so its own trials do
        # have an honest percentage: ``finalize_progress`` reports that
        # 0 -> n_trials arc, restarting per promoted move, which is what
        # lets a caller show a live bar through the finalizing phase
        # instead of a motionless full one.
        n_trials_ro = inner_ro._rollout_config["n_trials"]
        trial_total = n_trials_ro * sum(
            1 for r in results if r.get("eval_level") == "Rollout"
        )
        promoted = 0
        while results and "rollout_cubeful_equity" not in results[0]:
            promoted += 1
            if progress_callback:
                progress_callback(trial_total + promoted, trial_total, results)
            if finalize_progress:
                # Restart the sub-arc before the trials begin, so the bar
                # drops to 0 as this move's rollout starts rather than
                # sitting at the previous promotion's 100%.
                finalize_progress(0, n_trials_ro)
            m = results[0]
            pr = _rollout_eval_candidate(
                inner_ro, m["board"], m["board"],
                owner=owner, cube_value=cube_value,
                away1=away1, away2=away2, is_crawford=is_crawford,
                jacoby=jacoby, beaver=beaver, max_cube_value=max_cube_value,
                progress=finalize_progress,
            )
            m["probs"] = pr["probs"]
            m["prob_std_errors"] = pr["prob_std_errors"]
            m["cubeless_equity"] = pr["equity"]
            m["std_error"] = pr["std_error"]
            m["equity"] = pr["rollout_cubeful_equity"]
            m["rollout_cubeful_equity"] = pr["rollout_cubeful_equity"]
            m["rollout_cubeful_se"] = pr["rollout_cubeful_se"]
            m["eval_level"] = "Rollout"
            m.pop("is_1ply_only", None)
            results.sort(key=_move_order_key)

        if results:
            best = results[0]["equity"]
            for r in results:
                r["equity_diff"] = r["equity"] - best
        return results

    def cube_action_analytics(
        self, board, cube_value=1, cube_owner="centered",
        progress_callback=None,
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True, max_cube_value=0,
        incl_2ply_details=False,
    ) -> dict:
        return self._inner.cube_action_analytics(
            board, cube_value, cube_owner,
            progress_callback=progress_callback,
            away1=away1, away2=away2, is_crawford=is_crawford,
            jacoby=jacoby, beaver=beaver, **_cube_limit_kwargs(max_cube_value),
            incl_2ply_details=incl_2ply_details,
        )
