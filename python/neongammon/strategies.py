# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Fixed-depth analyzers and scoring shared by the analysis strategies."""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Any

from ._native import extension_constant as _extension_constant, load_native
from ._primitives import (
    _normalize_cube_context as _normalize_cube_context,
    _validate_cube_limit as _validate_cube_limit,
    cubeless_equity,
    resolve_owner as _resolve_owner_name,
)
from .runtime import _notify_thread_budget

bgbot_cpp = load_native()

if TYPE_CHECKING:
    from .batch import BatchModels


# ---------------------------------------------------------------------------
# Cube owner mapping
# ---------------------------------------------------------------------------

OWNER_MAP: dict[str, bgbot_cpp.CubeOwner] = {
    "centered": bgbot_cpp.CubeOwner.CENTERED,
    "player": bgbot_cpp.CubeOwner.PLAYER,
    "opponent": bgbot_cpp.CubeOwner.OPPONENT,
}

_FLIP_OWNER: dict[bgbot_cpp.CubeOwner, bgbot_cpp.CubeOwner] = {
    bgbot_cpp.CubeOwner.CENTERED: bgbot_cpp.CubeOwner.CENTERED,
    bgbot_cpp.CubeOwner.PLAYER: bgbot_cpp.CubeOwner.OPPONENT,
    bgbot_cpp.CubeOwner.OPPONENT: bgbot_cpp.CubeOwner.PLAYER,
}


def resolve_owner(cube_owner: object) -> bgbot_cpp.CubeOwner:
    """Resolve a case-insensitive owner name or a valid native cube owner."""
    if isinstance(cube_owner, bgbot_cpp.CubeOwner) and cube_owner in _FLIP_OWNER:
        return cube_owner
    return OWNER_MAP[_resolve_owner_name(cube_owner)]


def flip_owner(cube_owner: object) -> bgbot_cpp.CubeOwner:
    """Return the native cube owner from the other player's perspective."""
    return _FLIP_OWNER[resolve_owner(cube_owner)]


def _cube_limit_kwargs(max_cube_value: int) -> dict:
    return {"max_cube_value": max_cube_value} if max_cube_value else {}


def _dead_cube_equity(probs, cube_value, away1, away2, is_crawford):
    if away1 > 0 or away2 > 0:
        mwc = bgbot_cpp.cubeless_mwc(probs, away1, away2, cube_value, is_crawford)
        return bgbot_cpp.mwc2eq(mwc, away1, away2, cube_value, is_crawford)
    return cubeless_equity(probs)


def _default_parallel_threads() -> int:
    env_threads = os.getenv("BGBOT_MULTIPLY_THREADS", "")
    if not env_threads:
        return max(2, os.cpu_count() or 2)
    try:
        parsed = int(env_threads)
    except ValueError:
        return 0
    return max(parsed, 0)


@functools.lru_cache(maxsize=None)
def load_bearoff_db(path: str):
    """The one-sided bearoff database at ``path``, loaded once per process
    and shared (it is immutable once loaded; its data identity keys the
    evaluation caches), or None when it does not load."""
    db = bgbot_cpp.BearoffDB()
    return db if db.load(path) else None


def _move_order_key(m: dict):
    """Deterministic total order for a checker-play result list.

    Cubeful equity alone is not a total order: it saturates at the pass value in
    lost positions, so entire candidate lists tie exactly and the "best" move
    then depends on list insertion order. Ties break on cubeless equity, then on
    the board itself, so the same position always yields the same ranking -- and
    so batch_checker_play can agree with this path exactly.
    """
    return (-m["equity"], -m.get("cubeless_equity", m["equity"]), tuple(m["board"]))


#: Janowski cube life for CONTACT positions in money-game cube analysis, by
#: evaluation depth. The engine's default (cube_efficiency: 0.68) is right
#: where the cube tree is deep enough (3-ply and up). The 1-ply and 2-ply
#: trees end in a 1-ply Janowski leaf before the player's own next cube
#: node, and with 0.68 there they under-value No Double for the side holding
#: the initiative, so they over-double. Tuned on the odd seeds of the money
#: benchmark's 2,842 cube decisions and checked on the even ones (cube PR
#: 1-ply 3.06 -> 2.53, 2-ply 2.78 -> 1.27). Races keep their pip-based
#: value. Not applied to match play: on the 5-point match benchmark the same
#: values do not help (1-ply 2.35 -> 2.38, 2-ply 1.58 -> 1.82). The values
#: are the ones at 120 average pips; the slopes below make them rise with
#: the pip count (they replaced a flat 0.75 and 0.80).
MONEY_CONTACT_CUBE_LIFE: dict[int, float] = dict(_extension_constant("MONEY_CONTACT_CUBE_LIFE"))

#: Per-pip slope of the contact cube life above, by depth: the engine uses
#: ``MONEY_CONTACT_CUBE_LIFE[n] + slope * (average pips - 120)`` (clamped to
#: [0.5, 1.0]) for every position it converts, root and leaves alike, so at
#: 1-ply the cube life runs from about 0.66 at 75 pips to 0.79 at 150, and
#: at 2-ply from 0.71 to 0.80. Fitted to XG World Class's cube decisions on
#: the money benchmark's cube positions, half the seeds at a time; scored on
#: the other half against a flat cube life fitted the same way, cube PR
#: 2.07 -> 1.68 (1-ply) and 1.09 -> 0.90 (2-ply) against XG, 2.49 -> 1.98
#: and 1.44 -> 1.21 against the Neongammon reference, with fewer blunders
#: (scripts/fit_contact_cube_life.py).
MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE: dict[int, float] = dict(
    _extension_constant("MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE")
)


def contact_cube_life(n_plies: int, away1: int = 0, away2: int = 0) -> float:
    """``contact_x`` for a cube tree of depth ``n_plies``; -1 = engine default."""
    if away1 > 0 or away2 > 0:
        return -1.0
    return MONEY_CONTACT_CUBE_LIFE.get(n_plies, -1.0)


def contact_cube_slope(n_plies: int, away1: int = 0, away2: int = 0) -> float:
    """``contact_x_pip_slope`` for a cube tree of depth ``n_plies``; 0 = flat.
    Only takes effect where ``_contact_cube_life`` sets ``contact_x``."""
    if _contact_cube_life(n_plies, away1, away2) < 0.0:
        return 0.0
    return MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE.get(n_plies, 0.0)


# Older integrations imported these internal names directly.
_contact_cube_life = contact_cube_life
_contact_cube_slope = contact_cube_slope


def _janowski_x(probs, board, n_plies: int | None, away1: int = 0, away2: int = 0) -> float:
    """Cube life for a Janowski conversion of ``board`` in a depth-``n_plies``
    analysis (None: not a cube tree, engine default)."""
    race = bgbot_cpp.is_race(board)
    if not race and n_plies is not None:
        cx = _contact_cube_life(n_plies, away1, away2)
        if cx >= 0.0:
            pp, op = bgbot_cpp.pip_counts(board)
            return bgbot_cpp.contact_cube_x(
                cx, _contact_cube_slope(n_plies, away1, away2), pp, op)
    pp, op = bgbot_cpp.pip_counts(board)
    return bgbot_cpp.cube_efficiency(probs, race, pp, op)


def _janowski_1ply_equity(
    probs,
    post_move_board,
    owner: bgbot_cpp.CubeOwner,
    *,
    cube_value: int = 1,
    away1: int = 0,
    away2: int = 0,
    is_crawford: bool = False,
    jacoby: bool = True,
    max_cube_value: int = 0,
) -> float:
    """1-ply Janowski cubeful equity for a post-move board (or dead-cube equity at cap)."""
    if max_cube_value and cube_value >= max_cube_value:
        return _dead_cube_equity(probs, cube_value, away1, away2, is_crawford)
    x = _janowski_x(probs, post_move_board, 1, away1, away2)
    if away1 > 0 or away2 > 0:
        return bgbot_cpp.cl2cf(
            probs, cube_value, owner, x,
            away1, away2, is_crawford, jacoby=jacoby,
        )
    jacoby_active = jacoby and owner == bgbot_cpp.CubeOwner.CENTERED
    return bgbot_cpp.cl2cf_money(probs, owner, x, jacoby_active=jacoby_active)


class _CubelessBase:
    """Shared infrastructure for cubeless analyzers."""

    def __init__(self, models: BatchModels):
        self._weights = models.weights
        self._strategy_1ply = models.base()
        self._raw_base = models._raw_base
        self._bearoff_db = models.bearoff_db

    def _score_candidates(
        self,
        candidates: list,
        board: list[int],
        cube_owner: str | None = None,
        cube_value: int = 1,
        away1: int = 0,
        away2: int = 0,
        is_crawford: bool = False,
        jacoby: bool = True,
        max_cube_value: int = 0,
        strategy=None,
    ) -> list[tuple[float, float, list[int], list[float]]]:
        if strategy is None:
            strategy = self._strategy_1ply
        owner = resolve_owner(cube_owner) if cube_owner is not None else None
        scored = []
        for b in candidates:
            bl = list(b)
            r = strategy.evaluate_board(bl, board)
            cl_eq = r["equity"]
            probs = list(r["probs"])
            if owner is not None:
                cf_eq = _janowski_1ply_equity(
                    probs, bl, owner,
                    cube_value=cube_value, away1=away1, away2=away2,
                    is_crawford=is_crawford, jacoby=jacoby,
                    max_cube_value=max_cube_value,
                )
            else:
                cf_eq = cl_eq
            scored.append((cf_eq, cl_eq, bl, probs))
        scored.sort(key=lambda item: -item[0])
        return scored

    @staticmethod
    def _filter_candidates(
        scored_1ply: list,
        threshold: float,
        max_moves: int,
    ) -> tuple[list, set]:
        best_eq = scored_1ply[0][0]
        survivors = [
            item
            for item in scored_1ply
            if (best_eq - item[0]) < threshold
        ][:max_moves]
        survivor_set = {tuple(item[2]) for item in survivors}
        return survivors, survivor_set

    @staticmethod
    def _force_include(
        scored_pool: list,
        survivors: list,
        survivor_set: set,
        force_boards,
    ) -> None:
        """Force caller-specified post-move boards into the deep-eval survivor
        set so they are always evaluated at the full (N-ply / rollout) level,
        even when the move filter would otherwise drop them.

        Used by the app's expert re-evaluation to guarantee the move a player
        actually made is scored at the expert level — i.e. apples-to-apples with
        the best move, instead of carrying only its cheap 1-ply filter equity.

        ``scored_pool`` is the list of ``(cf_eq, cl_eq, board, probs)`` tuples
        from which survivors are drawn (the 1-ply scored list, or the 2-ply
        rescored list for the rollout prefilter path). Boards not present in the
        pool (already a survivor, or not a legal candidate) are skipped. Mutates
        ``survivors`` / ``survivor_set`` in place.
        """
        if not force_boards:
            return
        force_set = {tuple(b) for b in force_boards}
        for item in scored_pool:
            tb = tuple(item[2])
            if tb in force_set and tb not in survivor_set:
                survivors.append(item)
                survivor_set.add(tb)

    @staticmethod
    def _first_stale_top_two(results: list) -> dict | None:
        """The highest-ranked of the top two entries still carrying a 1-ply value.

        A candidate the filter pruned keeps its 1-ply equity while the survivors
        are re-scored at N-ply. The two scales are not comparable: whenever the
        deeper search values every survivor below a pruned move's 1-ply number,
        that move tops the list on a value the search never checked — and it is
        exactly in the positions where N-ply matters most (the leaves discover
        what 1-ply missed) that the survivors' values fall the furthest. Measured
        on the snake benchmark, 2026-09-04: 312 of 795 cubeless 2-ply picks were
        such unevaluated moves, and they carried 55 of the level's 68 PR points.
        So nothing may sit in the top two on a 1-ply value: the best move must be
        a full-depth value, and the runner-up too, since callers read both.
        """
        for r in results[:2]:
            if r.get("is_1ply_only"):
                return r
        return None

    @staticmethod
    def _promote_second_best(results: list, board: list[int], evaluate_fn) -> None:
        results.sort(key=_move_order_key)
        while (r := _CubelessBase._first_stale_top_two(results)) is not None:
            equity, probs, eval_level, extra = evaluate_fn(r["board"], board)
            r["equity"] = equity
            r["probs"] = probs
            r["eval_level"] = eval_level
            r.pop("is_1ply_only", None)
            r.update(extra)
            results.sort(key=_move_order_key)

    @staticmethod
    def _finalize_results(results: list[dict]) -> list[dict]:
        results.sort(key=_move_order_key)
        if results:
            best = results[0]["equity"]
            for r in results:
                r["equity_diff"] = r["equity"] - best
        return results

    @staticmethod
    def _format_cube_result(r: dict, eval_level: str = "1-ply") -> dict:
        return {
            "probs": list(r["probs"]),
            "cubeless_equity": r.get("cubeless_equity", 0),
            "equity_nd": r["equity_nd"],
            "equity_dt": r["equity_dt"],
            "equity_dp": r["equity_dp"],
            "should_double": bool(r["should_double"]),
            "should_take": bool(r["should_take"]),
            "optimal_equity": r["optimal_equity"],
            "is_beaver": bool(r.get("is_beaver", False)),
            "cubeless_se": r.get("cubeless_se", None),
            "equity_nd_se": r.get("equity_nd_se", None),
            "equity_dt_se": r.get("equity_dt_se", None),
            "eval_level": eval_level,
        }


class _OnePlyAnalyzer(_CubelessBase):

    def __init__(self, models, parallel_threads=0):
        super().__init__(models)
        self._parallel_threads = (
            parallel_threads if parallel_threads > 0 else _default_parallel_threads()
        )

    def checker_play_analytics(
        self, board, die1, die2, cube_value=1, cube_owner="centered",
        progress_callback=None,
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True, max_cube_value=0,
        force_boards=None, finalize_progress=None,
    ) -> list[dict]:
        # ``force_boards`` is a no-op at 1-ply: every candidate is already
        # evaluated at the full level, so there's nothing to force into a
        # filtered deep-eval set. Accepted for call-signature symmetry, as is
        # ``finalize_progress`` — only the cubeful wrapper has a finalizing
        # phase to report.
        del force_boards, finalize_progress
        if cube_owner is not None:
            resolve_owner(cube_owner)
        candidates = bgbot_cpp.possible_moves(board, die1, die2)
        if not candidates:
            return []
        results = []
        for b in candidates:
            bl = list(b)
            r = self._strategy_1ply.evaluate_board(bl, board)
            results.append({
                "board": bl,
                "equity": r["equity"],
                "probs": list(r["probs"]),
                "eval_level": "1-ply",
            })
        return self._finalize_results(results)

    def cube_action_analytics(
        self, board, cube_value=1, cube_owner="centered",
        progress_callback=None,
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True, max_cube_value=0,
        incl_2ply_details=False,
    ) -> dict:
        if incl_2ply_details:
            raise ValueError("incl_2ply_details requires at least 2-ply evaluation")
        owner = resolve_owner(cube_owner)
        r = bgbot_cpp.evaluate_cube_decision_unified(
            board, cube_value, owner,
            self._weights.strategy_type,
            self._weights.weight_paths_list,
            self._weights.hidden_sizes_list,
            away1=away1, away2=away2, is_crawford=is_crawford,
            jacoby=jacoby, beaver=beaver, **_cube_limit_kwargs(max_cube_value),
            contact_x=_contact_cube_life(1, away1, away2),
            contact_x_pip_slope=_contact_cube_slope(1, away1, away2),
            bearoff_db=self._bearoff_db, strategy=self._raw_base,
        )
        return self._format_cube_result(r, eval_level="1-ply")


class _MultiPlyAnalyzer(_CubelessBase):

    FILTER_MAX_MOVES = 5
    FILTER_THRESHOLD = 0.08

    def __init__(self, models, n_plies, parallel_evaluate=True, parallel_threads=0):
        super().__init__(models)
        self._n_plies = n_plies
        self._parallel_evaluate = parallel_evaluate
        requested_threads = parallel_threads
        if parallel_threads <= 0 and n_plies > 1:
            requested_threads = _default_parallel_threads()
        # An explicit budget is kept as given, one included: a caller that
        # parallelises across positions (the benchmarks) asks for one thread
        # per analysis and gets inline evaluation, no worker pool.
        self._parallel_threads = requested_threads
        self._strategy_nply = bgbot_cpp.create_multipy_from_base(
            self._raw_base,
            n_plies=n_plies,
            parallel_evaluate=parallel_evaluate,
            parallel_threads=self._parallel_threads,
        )

    def checker_play_analytics(
        self, board, die1, die2, cube_value=1, cube_owner="centered",
        progress_callback=None,
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True, max_cube_value=0,
        force_boards=None, finalize_progress=None,
    ) -> list[dict]:
        # Only the cubeful wrapper runs the finalizing (promotion) phase, so
        # there is nothing here to report against ``finalize_progress``.
        del finalize_progress
        if cube_owner is not None:
            resolve_owner(cube_owner)
        candidates = bgbot_cpp.possible_moves(board, die1, die2)
        if not candidates:
            return []

        scored_1ply = self._score_candidates(
            candidates, board, cube_owner,
            cube_value=cube_value, away1=away1, away2=away2,
            is_crawford=is_crawford, jacoby=jacoby, max_cube_value=max_cube_value,
        )
        survivors, survivor_set = self._filter_candidates(
            scored_1ply, self.FILTER_THRESHOLD, self.FILTER_MAX_MOVES
        )
        # Force-include caller-specified boards (e.g. the move actually played
        # in an expert re-eval) so they're evaluated at N-ply even when the
        # filter dropped them.
        self._force_include(scored_1ply, survivors, survivor_set, force_boards)

        results = []
        effective_threads = self._strategy_nply.effective_parallel_threads()
        for feq, cleq, b, p0 in survivors:
            _notify_thread_budget(
                "multipy_evaluate", effective_threads,
                plies=self._n_plies, cubeful=False,
            )
            r = self._strategy_nply.evaluate_board(b, board)
            results.append({
                "board": b,
                "equity": r["equity"],
                "probs": list(r["probs"]),
                "eval_level": f"{self._n_plies}-ply",
            })

        for feq, cleq, b, p in scored_1ply:
            if tuple(b) not in survivor_set:
                results.append({
                    "board": b,
                    "equity": cleq,
                    "probs": p,
                    "is_1ply_only": True,
                    "eval_level": "1-ply",
                })

        n_plies = self._n_plies
        strategy = self._strategy_nply

        def _nply_eval(b, board_ref):
            _notify_thread_budget(
                "multipy_evaluate", effective_threads,
                plies=n_plies, cubeful=False,
            )
            r = strategy.evaluate_board(b, board_ref)
            return r["equity"], list(r["probs"]), f"{n_plies}-ply", {}

        self._promote_second_best(results, board, _nply_eval)
        self._strategy_nply.clear_cache()
        return self._finalize_results(results)

    def cube_action_analytics(
        self, board, cube_value=1, cube_owner="centered",
        progress_callback=None,
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True, max_cube_value=0,
        incl_2ply_details=False, decision_only=False,
    ) -> dict:
        owner = resolve_owner(cube_owner)
        _notify_thread_budget(
            "cube_decision_nply", self._parallel_threads or 1,
            plies=self._n_plies, cubeful=True,
        )
        r = bgbot_cpp.cube_decision_nply_unified(
            board, cube_value, owner, self._n_plies,
            self._weights.strategy_type,
            self._weights.weight_paths_list,
            self._weights.hidden_sizes_list,
            n_threads=self._parallel_threads,
            away1=away1, away2=away2, is_crawford=is_crawford,
            jacoby=jacoby, beaver=beaver, **_cube_limit_kwargs(max_cube_value),
            contact_x=_contact_cube_life(self._n_plies, away1, away2),
            contact_x_pip_slope=_contact_cube_slope(self._n_plies, away1, away2),
            bearoff_db=self._bearoff_db,
            incl_2ply_details=incl_2ply_details, strategy=self._raw_base,
            decision_only=decision_only,
        )
        if decision_only:
            # No cubeless probability search: the decision fields only.
            self._strategy_nply.clear_cache()
            return {k: r[k] for k in ("equity_nd", "equity_dt", "equity_dp", "should_double",
                                      "should_take", "optimal_equity", "is_beaver")} | {
                "eval_level": f"{self._n_plies}-ply"}
        result = self._format_cube_result(r, eval_level=f"{self._n_plies}-ply")

        # Pass through 2-ply details if present
        if "details" in r:
            result["details"] = r["details"]

        # The C++ binding computes N-ply cubeless probs with parallelism and
        # bearoff DB. No need to recompute in Python.

        # Always clear the shared position cache after cube analysis.
        # cube_decision_nply() fills the cache internally; without clearing,
        # subsequent analyses (especially on bearoff positions where the
        # non-bearoff branch above is skipped) can return stale cached values.
        self._strategy_nply.clear_cache()

        return result
