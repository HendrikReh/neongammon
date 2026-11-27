# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""High-level analysis interface for the Neongammon bot engine.

This module provides :class:`BgBotAnalyzer`, the main entry point for
checker play and cube action analysis at any evaluation level (1-ply
through N-ply and Monte Carlo rollout).

Typical usage::

    from neongammon import BgBotAnalyzer

    analyzer = BgBotAnalyzer()                       # 1-ply, cubeful
    result = analyzer.checker_play(STARTING_BOARD, 3, 1)
    for m in result.moves:
        print(f"{m.equity:+.3f}  {m.probs.win:.1%}")
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import bgbot_cpp

from .cubeful import _CubefulAnalyzer
from .luck import roll_luck as _roll_luck
from .rollout import RolloutCancelled, _RolloutAnalyzer
from .strategies import (
    _CubelessBase,
    _MultiPlyAnalyzer,
    _OnePlyAnalyzer,
    _cube_limit_kwargs,
    _dead_cube_equity,
    _janowski_x,
    _validate_cube_limit,
    contact_cube_life,
    contact_cube_slope,
    flip_owner,
    resolve_owner,
)
from .types import (
    CheckerPlayResult,
    CubeActionResult,
    CubeDecisionResult,
    LuckResult,
    MoveAnalysis,
    PostMoveAnalysis,
    Probabilities,
)
from .weights import WeightConfig, WeightConfigPair

if TYPE_CHECKING:
    from .batch import BatchModels


def _dict_to_move_analysis(d: dict, include_game_plans: bool = False) -> MoveAnalysis:
    """Convert an internal result dict to a :class:`MoveAnalysis`."""
    player_gp = None
    opponent_gp = None
    if include_game_plans:
        b = d["board"]
        gp = bgbot_cpp.classify_game_plan(b)
        player_gp = gp.name.lower() if hasattr(gp, "name") else str(gp).lower()
        flipped = bgbot_cpp.flip_board(b)
        gp2 = bgbot_cpp.classify_game_plan(flipped)
        opponent_gp = gp2.name.lower() if hasattr(gp2, "name") else str(gp2).lower()

    return MoveAnalysis(
        board=d["board"],
        equity=d["equity"],
        cubeless_equity=d.get("cubeless_equity", d["equity"]),
        probs=Probabilities.from_list(d["probs"]),
        equity_diff=d.get("equity_diff", 0.0),
        eval_level=d["eval_level"],
        player_game_plan=player_gp,
        opponent_game_plan=opponent_gp,
        std_error=d.get("std_error"),
        prob_std_errors=d.get("prob_std_errors"),
    )


def _optimal_action(should_double: bool, should_take: bool,
                    is_beaver: bool = False) -> str:
    if not should_double:
        return "No Double"
    if is_beaver:
        return "Double/Beaver"
    if should_take:
        return "Double/Take"
    return "Double/Pass"


class BgBotAnalyzer:
    """High-level interface to the Neongammon bot engine.

    Thread-safe: multiple threads can call analysis methods concurrently.
    For rollout evaluation levels, concurrent rollout calls on the same
    instance serialize internally while ``cancel()``, ``reset_cancel()``, and
    ``set_seed()`` affect the shared instance.

    Args:
        weights: Weight file configuration (defaults to production model).
        eval_level: ``'1ply'``, ``'2ply'``, ``'3ply'``, ``'4ply'``,
            ``'truncated1'``, ``'truncated2'``, ``'truncated3'``, or ``'rollout'``.
        cubeful: If True, compute cubeful equities via Janowski.
        filter_max_moves: Candidate-selection filter width for
            ``eval_level='rollout'`` checker play — how many of the top
            scored moves are rolled out (default 5; the rest keep their
            filter-stage equity). Ignored by other eval levels.
        filter_threshold: Equity window for the same filter — candidates
            more than this far below the best are dropped even within the
            ``filter_max_moves`` cap (default 0.08).
        parallel_threads: Thread count (0 = auto-detect).
        n_trials: Rollout trial count.
        truncation_depth: Rollout truncation (0 = play to completion).
        decision_ply: Ply depth for move selection during rollout trials.
        truncation_ply: Ply depth for evaluation at the truncation point
            (-1 = same as ``decision_ply``). Only used by ``eval_level='rollout'``;
            the named ``truncated1/2/3`` levels fix their own truncation ply.
        late_ply: Ply for move selection after ``late_threshold`` half-moves
            (-1 = same as ``decision_ply``).
        late_threshold: Half-move index where decision ply switches to ``late_ply``.
        seed: RNG seed for rollout.
        checker: TrialEvalConfig for checker play during rollout trials.
        checker_late: TrialEvalConfig for late-game checker play.
        cube: TrialEvalConfig for cube decisions during rollout trials.
        cube_late: TrialEvalConfig for late-game cube decisions.
        prefilter_threshold: Two-stage checker_play filter for rollout levels.
            When > 0, stage 1 culls candidates with > prefilter_threshold
            1-ply equity error from best (no count cap), then stage 2 applies
            the TINY filter at 2-ply on the survivors before rollout. Defaults
            to 0.15 for truncated2, truncated3, and any non-truncated rollout
            (eval_level="rollout" with truncation_depth=0); 0.0 (legacy
            single-stage 1-ply TINY) for truncated1 and user-configured
            truncated rollouts. Pass an explicit value to override.
    """

    def __init__(
        self,
        weights: WeightConfig | WeightConfigPair | None = None,
        eval_level: str = "1ply",
        cubeful: bool = True,
        *,
        filter_max_moves: int = 5,
        filter_threshold: float = 0.08,
        parallel_threads: int = 0,
        n_trials: int = 1296,
        truncation_depth: int = 0,
        decision_ply: int = 1,
        truncation_ply: int = -1,
        late_ply: int = -1,
        late_threshold: int = 20,
        seed: int = 42,
        bearoff_db: bool | str | None = None,
        models: BatchModels | None = None,
        checker=None,
        checker_late=None,
        cube=None,
        cube_late=None,
        ultra_late_threshold: int = 9999,
        cubeful_trial_moves: bool = True,
        cubeful_late_threshold: int = 0,
        prefilter_threshold: float | None = None,
        target_se: float = 0.0,
        max_batches: int = 50,
    ):
        from .batch import _resolve_models

        self._models = _resolve_models(weights, models, bearoff_db)
        self._weights = self._models.weights
        self._eval_level = eval_level
        self._cubeful = bool(cubeful)
        # Two-stage checker_play filter: 1-ply loose cull (prefilter_threshold)
        # then 2-ply TINY. Defaults to 0.15 for truncated2, truncated3, and
        # any non-truncated rollout (eval_level="rollout" with
        # truncation_depth=0); 0.0 (disabled, legacy single-stage 1-ply
        # TINY) elsewhere. Rationale: the rollout is expensive enough at all
        # those levels that pre-rollout filtering is worth the small 2-ply
        # cost; 1T and user-configured truncated rollouts are cheap enough
        # that the extra 2-ply scoring isn't worth it. User-supplied value
        # overrides the per-level default.
        if prefilter_threshold is None:
            is_full_rollout = (
                eval_level == "rollout" and truncation_depth == 0
            )
            if is_full_rollout:
                # Keep the loose stage-1 cull comfortably wider than the
                # final candidate filter so a widened filter_threshold
                # (wide/gigantic search interval) is never strangled by the
                # prefilter. The default filter (0.08) yields the historical
                # 0.15.
                prefilter_threshold = max(0.15, filter_threshold + 0.05)
            elif eval_level in ("truncated2", "truncated3"):
                prefilter_threshold = 0.15
            else:
                prefilter_threshold = 0.0
        self._prefilter_threshold = float(prefilter_threshold)
        # Flag-gated cube-aware trial moves (see ROLLOUT.md). Default
        # False keeps existing rollout behavior. When True, all rollout-based
        # eval levels select trial moves by cubeful equity against the trial's
        # cube state instead of cubeless equity.
        self._cubeful_trial_moves = bool(cubeful_trial_moves)
        # Per-move drop-to-cubeless threshold for full-game rollouts where
        # ultra_late_threshold=9999 keeps cubeful active for ~50 half-moves.
        # 0 = inherit from ultra_late_threshold (no separate drop).
        self._cubeful_late_threshold = int(cubeful_late_threshold)
        self._bearoff_db = self._models.bearoff_db

        if eval_level == "1ply":
            inner: _CubelessBase = _OnePlyAnalyzer(
                self._models, parallel_threads=parallel_threads,
            )
        elif eval_level in ("2ply", "3ply", "4ply"):
            n_plies = int(eval_level[0])
            inner = _MultiPlyAnalyzer(
                self._models, n_plies=n_plies,
                parallel_threads=parallel_threads,
            )
        elif eval_level == "truncated1":
            # 1T: 72 trials (2x36) -- a multiple of 36, so the first roll is
            # exactly stratified. XG Roller's 42 is not, which 2x over-weights
            # 6 ordered first rolls and biases the result (benchmark PR 2.23 ->
            # 0.50 going 42 -> 72). Otherwise XG-Roller-style: trunc-5, 1-ply.
            inner = _RolloutAnalyzer(
                self._models,
                n_trials=72,
                truncation_depth=5,
                decision_ply=1,
                n_threads=parallel_threads,
                seed=seed,
                ultra_late_threshold=2,
                cubeful_trial_moves=self._cubeful_trial_moves,
                cubeful_late_threshold=self._cubeful_late_threshold,
            )
        elif eval_level == "truncated2":
            # 2T: trunc-7, 360 trials. Checker 2-ply on the first ply then
            # 1-ply (late_threshold=1); cube 2-ply throughout; 2-ply truncation
            # eval; no ultra-late 1-ply drop (ultra_late=9999). Beats XG Roller+
            # (benchmark PR 0.89 -> 0.36). Was: late_threshold=2, ultra_late=2,
            # 1-ply late cube + 1-ply truncation eval (the ply-drop economies
            # that cost ~0.5 PR).
            inner = _RolloutAnalyzer(
                self._models,
                n_trials=360,
                truncation_depth=7,
                decision_ply=2,
                truncation_ply=2,
                n_threads=parallel_threads,
                seed=seed,
                late_ply=1,
                late_threshold=1,
                cube=bgbot_cpp.TrialEvalConfig(ply=2),
                cube_late=bgbot_cpp.TrialEvalConfig(ply=2),
                ultra_late_threshold=9999,
                cubeful_trial_moves=self._cubeful_trial_moves,
                cubeful_late_threshold=self._cubeful_late_threshold,
                prefilter_threshold=self._prefilter_threshold,
            )
        elif eval_level == "truncated3":
            # 3T: trunc-7, 3-ply early then 2-ply late (ultra_late=9999 disables
            # the 1-ply drop). Closer to XG Roller++. (Was trunc-5 / 1-ply late.)
            inner = _RolloutAnalyzer(
                self._models,
                n_trials=360,
                truncation_depth=7,
                decision_ply=3,
                n_threads=parallel_threads,
                seed=seed,
                late_ply=2,
                late_threshold=2,
                ultra_late_threshold=9999,
                cubeful_trial_moves=self._cubeful_trial_moves,
                cubeful_late_threshold=self._cubeful_late_threshold,
                prefilter_threshold=self._prefilter_threshold,
            )
        elif eval_level == "rollout":
            inner = _RolloutAnalyzer(
                self._models,
                n_trials=n_trials,
                truncation_depth=truncation_depth,
                decision_ply=decision_ply,
                truncation_ply=truncation_ply,
                n_threads=parallel_threads,
                seed=seed,
                late_ply=late_ply,
                late_threshold=late_threshold,
                checker=checker,
                checker_late=checker_late,
                cube=cube,
                cube_late=cube_late,
                ultra_late_threshold=ultra_late_threshold,
                cubeful_trial_moves=self._cubeful_trial_moves,
                cubeful_late_threshold=self._cubeful_late_threshold,
                prefilter_threshold=self._prefilter_threshold,
                target_se=target_se,
                max_batches=max_batches,
                filter_max_moves=filter_max_moves,
                filter_threshold=filter_threshold,
            )
        else:
            raise ValueError(f"Unknown eval_level: {eval_level!r}")

        # Set bearoff DB on inner analyzer and its C++ strategies
        if self._bearoff_db is not None:
            if isinstance(inner, _MultiPlyAnalyzer):
                bgbot_cpp.multipy_set_bearoff_db(inner._strategy_nply, self._bearoff_db)
            elif isinstance(inner, _RolloutAnalyzer):
                bgbot_cpp.rollout_set_bearoff_db(inner._rollout_strategy, self._bearoff_db)
                if inner._strategy_2ply is not None:
                    bgbot_cpp.multipy_set_bearoff_db(inner._strategy_2ply, self._bearoff_db)

        if cubeful:
            self._analyzer = _CubefulAnalyzer(inner)
        else:
            self._analyzer = inner

    def cancel(self):
        """Request cancellation of an in-progress rollout.

        Thread-safe. Only effective when the analyzer uses rollout evaluation.
        After calling cancel(), the next or in-progress checker_play() or
        cube_action() call will raise :class:`RolloutCancelled`.
        Call :meth:`reset_cancel` before reusing the analyzer.
        """
        inner = self._analyzer
        if isinstance(inner, _CubefulAnalyzer):
            inner = inner._inner
        if isinstance(inner, _RolloutAnalyzer):
            inner.cancel()

    def reset_cancel(self):
        """Clear cancellation flag so the analyzer can be reused."""
        inner = self._analyzer
        if isinstance(inner, _CubefulAnalyzer):
            inner = inner._inner
        if isinstance(inner, _RolloutAnalyzer):
            inner.reset_cancel()

    def epc(self, board: list[int], player: int = 0) -> float | None:
        """Return Effective Pip Count for a player in a bearoff position.

        EPC = mean_rolls × (49/6), where mean_rolls is the expected number
        of rolls to bear off all checkers (including the upcoming roll).

        Args:
            board: 26-element board array.
            player: 0 = player on roll, 1 = opponent.

        Returns:
            EPC as a float, or None if the position is not in the bearoff DB.
        """
        if self._bearoff_db is None or not self._bearoff_db.is_bearoff(board):
            return None
        return self._bearoff_db.lookup_epc(board, player)

    def set_seed(self, seed: int) -> None:
        """Reseed the underlying rollout RNG, if this analyzer is rollout-based.

        Used to run independent seeded batches of the same position while
        keeping the rollout SharedPosCache warm (e.g. lockstep target-SE checker
        rollouts). No-op for non-rollout eval levels.
        """
        inner = getattr(self, "_analyzer", None)
        if inner is not None and hasattr(inner, "set_seed"):
            inner.set_seed(seed)

    def checker_play(
        self,
        board: list[int],
        die1: int,
        die2: int,
        cube_value: int = 1,
        cube_owner: str = "centered",
        include_game_plans: bool = False,
        progress_callback: Any | None = None,
        *,
        away1: int = 0,
        away2: int = 0,
        is_crawford: bool = False,
        jacoby: bool = True,
        beaver: bool = True,
        max_cube_value: int = 0,
        force_boards: list[list[int]] | None = None,
        finalize_progress: Any | None = None,
    ) -> CheckerPlayResult:
        """Analyze all legal moves for a checker play decision.

        Returns moves sorted by equity (best first). Each move includes
        post-move probabilities and equity difference from the best move.

        Args:
            board: 26-element board array.
            die1, die2: Dice values (1-6).
            cube_value: Current cube value.
            cube_owner: ``'centered'``, ``'player'``, or ``'opponent'``.
            include_game_plans: If True, populate ``player_game_plan`` and
                ``opponent_game_plan`` on each :class:`MoveAnalysis`.
            progress_callback: Optional ``callback(completed, total, partial)``
                for rollout progress.
            finalize_progress: Optional ``callback(completed, total)`` for the
                rollout's finalizing phase, where moves the filter dropped are
                promoted and rolled out one at a time (see
                ``_CubefulAnalyzer.checker_play_analytics``). Reports each
                promotion's own 0 -> n_trials trial arc, restarting per move,
                so a UI can keep a live progress bar through a phase that
                ``progress_callback`` can only report as "past 100%".
            away1: Points player needs to win (0 = money game).
            away2: Points opponent needs to win (0 = money game).
            is_crawford: True if this is the Crawford game.
            jacoby: If True, gammons/backgammons don't count when cube is
                centered (money games only). Auto-disabled for match play.
            max_cube_value: Maximum cube (0 = unlimited), or a power of two >= 2.
            beaver: If True, opponent can beaver after being doubled
                (money games only). Auto-disabled for match play.
            force_boards: Optional list of post-move boards (mover's
                perspective) to always evaluate at the full level, bypassing
                the move filter. Boards that aren't legal candidates are
                ignored. Used by the app's expert re-evaluation so the move a
                player actually made is always scored at the expert level.
        """
        _validate_cube_limit(cube_value, max_cube_value)
        if away1 > 0 or away2 > 0:
            jacoby = False
            beaver = False
        raw = self._analyzer.checker_play_analytics(
            board, die1, die2, cube_value, cube_owner, progress_callback,
            away1=away1, away2=away2, is_crawford=is_crawford, jacoby=jacoby,
            beaver=beaver, **_cube_limit_kwargs(max_cube_value), force_boards=force_boards,
            finalize_progress=finalize_progress,
        )
        moves = [_dict_to_move_analysis(d, include_game_plans) for d in raw]
        eval_level = moves[0].eval_level if moves else self._eval_level
        return CheckerPlayResult(
            moves=moves, board=board, die1=die1, die2=die2,
            eval_level=eval_level,
        )

    def top_two_promote(
        self, board: list[int], moves: list[MoveAnalysis], *, max_gap: float = 0.010,
        cube_value: int = 1, cube_owner: str = "centered", away1: int = 0,
        away2: int = 0, is_crawford: bool = False, jacoby: bool = True,
        beaver: bool = True, max_cube_value: int = 0,
    ) -> tuple[list[MoveAnalysis], str]:
        """Re-rank two close 2-ply contact moves at 3-ply, using the original root.

        The first two moves must be ordered by descending equity. Ties are allowed.
        Returns the input ranking and "2-ply" for races, forced moves and gaps
        above max_gap. A promotion returns just the two searched candidates.
        """
        if self._eval_level != "2ply" or not self._cubeful:
            raise ValueError("top_two_promote requires a cubeful 2-ply analyzer")
        _validate_cube_limit(cube_value, max_cube_value)
        if len(moves) < 2:
            return moves, "2-ply"
        first, second = moves[:2]
        if first.equity < second.equity:
            raise ValueError("the two leading moves must be ranked by descending equity")
        if bgbot_cpp.is_race(board):
            return moves, "2-ply"
        if any(move.eval_level != "2-ply" for move in (first, second)):
            raise ValueError("both leading moves must have 2-ply evaluations")
        if first.equity - second.equity > max_gap:
            return moves, "2-ply"
        inner = self._analyzer._inner
        owner = flip_owner(cube_owner)
        money = away1 == away2 == 0

        opponent_cube = bgbot_cpp.CubeInfo(
            cube_value, owner, away2, away1, is_crawford, money and jacoby,
            money and beaver, max_cube_value, contact_cube_life(3, away1, away2),
            contact_cube_slope(3, away1, away2),
        )
        endpoints = [first.board, second.board]
        scores = bgbot_cpp.score_checker_endpoints(
            board, endpoints, opponent_cube, inner._strategy_1ply, 3, inner._parallel_threads,
        )
        promoted = [
            replace(first if r["board"] == first.board else second,
                    equity=r["equity"], cubeless_equity=r["cubeless_equity"],
                    probs=Probabilities.from_list(r["probs"]), equity_diff=r["equity_diff"],
                    eval_level=r["eval_level"])
            for r in scores
        ]
        return promoted, "3-ply"

    def post_move_analytics(
        self,
        board: list[int],
        cube_owner: str = "centered",
        cube_value: int = 1,
        *,
        away1: int = 0,
        away2: int = 0,
        is_crawford: bool = False,
        jacoby: bool = True,
        beaver: bool = True,
        max_cube_value: int = 0,
        progress_callback: Any | None = None,
    ) -> PostMoveAnalysis:
        """Evaluate a post-move position (right before the opponent's turn).

        Returns cubeless probabilities, cubeless equity, and cubeful equity
        from the perspective of the player who just moved. When this analyzer
        was created with ``cubeful=False``, ``cubeful_equity`` equals
        ``cubeless_equity``.

        Args:
            board: 26-element post-move board array (player who moved's perspective).
            cube_owner: ``'centered'``, ``'player'``, or ``'opponent'``.
            cube_value: Current cube value.
            away1: Points player needs to win (0 = money game).
            away2: Points opponent needs to win (0 = money game).
            is_crawford: True if this is the Crawford game.
            jacoby: If True, gammons/backgammons don't count when cube is
                centered (money games only). Auto-disabled for match play.
            beaver: If True, opponent can beaver after being doubled (money
                games only). Auto-disabled for match play.
            max_cube_value: Maximum cube (0 = unlimited), or a power of two >= 2.
            progress_callback: Optional ``callback(completed, total)`` called
                from inside the rollout's worker threads.  Only meaningful for
                rollout eval levels — N-ply and 1-ply paths return synchronously
                and don't emit progress.
        """
        _validate_cube_limit(cube_value, max_cube_value)
        if away1 > 0 or away2 > 0:
            jacoby = False
            beaver = False

        inner = self._analyzer
        if isinstance(inner, _CubefulAnalyzer):
            inner = inner._inner

        # Evaluate the post-move board (NN outputs from mover's perspective).
        # When cubeful=True and the inner analyzer is a rollout with cube-aware
        # trial moves enabled, route through cubeful_evaluate_board so the trial
        # loop receives the cube state and selects moves by cubeful equity.
        owner = resolve_owner(cube_owner)
        use_cubeful_rollout = (
            isinstance(inner, _RolloutAnalyzer)
            and self._cubeful
            and self._cubeful_trial_moves
        )

        if isinstance(inner, _MultiPlyAnalyzer):
            n_plies = inner._n_plies
            inner._strategy_nply.clear_cache()
            r = inner._strategy_nply.evaluate_board(board, board)
            eval_level = f"{n_plies}-ply"
        elif isinstance(inner, _RolloutAnalyzer):
            inner._check_cancel()
            try:
                if use_cubeful_rollout:
                    r = inner._rollout_strategy.cubeful_evaluate_board(
                        board, board,
                        cube_value=cube_value, owner=owner,
                        away1=away1, away2=away2, is_crawford=is_crawford,
                        jacoby=jacoby, beaver=beaver,
                        **_cube_limit_kwargs(max_cube_value),
                        progress=progress_callback,
                    )
                else:
                    r = inner._rollout_strategy.evaluate_board(
                        board, board,
                        progress=progress_callback,
                    )
            except bgbot_cpp.RolloutCancelled:
                raise RolloutCancelled()
            eval_level = "Rollout"
        else:
            r = inner._strategy_1ply.evaluate_board(board, board)
            eval_level = "1-ply"

        probs_list = list(r["probs"])
        cl_eq = r["equity"]

        if not self._cubeful:
            cf_eq = cl_eq
        elif use_cubeful_rollout:
            # cubeful_evaluate_board returns cubeful_equity in basis-cube
            # units already (computed from the rollout's own trial paths,
            # including the opponent's optimal cube action at the start of
            # their turn). No post-hoc Janowski needed.
            cf_eq = float(r["cubeful_equity"])
        elif max_cube_value and cube_value >= max_cube_value:
            cf_eq = _dead_cube_equity(probs_list, cube_value, away1, away2, is_crawford)
        else:
            # Cubeful equity via post-hoc root Janowski on cubeless probs.
            # Root-Janowski conversion uses the 1-ply money contact cube life
            # across all depths (matching batch_post_move_evaluate).
            x = _janowski_x(probs_list, board, 1, away1, away2)
            is_match = away1 > 0 or away2 > 0
            if is_match:
                cf_eq = bgbot_cpp.cl2cf(probs_list, cube_value, owner, x,
                                        away1, away2, is_crawford,
                                        jacoby=jacoby)
            else:
                jacoby_active = (
                    jacoby and owner == bgbot_cpp.CubeOwner.CENTERED
                )
                cf_eq = bgbot_cpp.cl2cf_money(probs_list, owner, x,
                                              jacoby_active=jacoby_active)

        return PostMoveAnalysis(
            probs=Probabilities.from_list(probs_list),
            cubeless_equity=cl_eq,
            cubeful_equity=cf_eq,
            eval_level=eval_level,
            cubeless_se=(float(r["std_error"])
                         if isinstance(r, dict) and r.get("std_error") is not None
                         else None),
            cubeful_se=(float(r["cubeful_se"])
                        if isinstance(r, dict) and r.get("cubeful_se") is not None
                        else None),
        )

    def cube_action(
        self,
        board: list[int],
        cube_value: int = 1,
        cube_owner: str = "centered",
        *,
        away1: int = 0,
        away2: int = 0,
        is_crawford: bool = False,
        jacoby: bool = True,
        beaver: bool = True,
        max_cube_value: int = 0,
        incl_2ply_details: bool = False,
        progress_callback=None,
    ) -> CubeActionResult:
        """Analyze the cube decision for a pre-roll position.

        Returns cubeful equities for No Double, Double/Take, Double/Pass,
        with the optimal action and pre-roll cubeless probabilities.

        Args:
            board: 26-element board array.
            cube_value: Current cube value.
            cube_owner: ``'centered'``, ``'player'``, or ``'opponent'``.
            away1: Points player needs to win (0 = money game).
            away2: Points opponent needs to win (0 = money game).
            is_crawford: True if this is the Crawford game.
            jacoby: If True, gammons/backgammons don't count when cube is
                centered (money games only). Auto-disabled for match play.
            max_cube_value: Maximum cube (0 = unlimited), or a power of two >= 2.
            beaver: If True, opponent can beaver (redouble while retaining
                ownership) after being doubled. Money games only.
                Auto-disabled for match play.
            incl_2ply_details: If True, include per-roll details for the
                first two turns under both ND and DT scenarios. Requires
                2-ply or higher evaluation. At 2-ply only the player-roll
                level is captured (per-roll equities at 1-ply, no
                opponent_rolls); the headline equities match the plain
                2-ply call.
        """
        _validate_cube_limit(cube_value, max_cube_value)
        if away1 > 0 or away2 > 0:
            jacoby = False
            beaver = False
        raw = self._analyzer.cube_action_analytics(
            board, cube_value, cube_owner,
            progress_callback=progress_callback,
            away1=away1, away2=away2, is_crawford=is_crawford,
            jacoby=jacoby, beaver=beaver, **_cube_limit_kwargs(max_cube_value),
            incl_2ply_details=incl_2ply_details,
        )
        probs = Probabilities.from_list(raw["probs"])
        is_beaver = raw.get("is_beaver", False)
        return CubeActionResult(
            probs=probs,
            cubeless_equity=raw["cubeless_equity"],
            equity_nd=raw["equity_nd"],
            equity_dt=raw["equity_dt"],
            equity_dp=raw["equity_dp"],
            should_double=raw["should_double"],
            should_take=raw["should_take"],
            optimal_equity=raw["optimal_equity"],
            optimal_action=_optimal_action(
                raw["should_double"], raw["should_take"], is_beaver),
            eval_level=raw["eval_level"],
            is_beaver=is_beaver,
            cubeless_se=raw.get("cubeless_se"),
            equity_nd_se=raw.get("equity_nd_se"),
            equity_dt_se=raw.get("equity_dt_se"),
            details=raw.get("details"),
        )

    def cube_decision(
        self,
        board: list[int],
        cube_value: int = 1,
        cube_owner: str = "centered",
        *,
        away1: int = 0,
        away2: int = 0,
        is_crawford: bool = False,
        jacoby: bool = True,
        beaver: bool = True,
        max_cube_value: int = 0,
    ) -> CubeDecisionResult:
        """The cube decision alone: what :meth:`cube_action` returns minus the
        cubeless probabilities and their equity.

        At 2-ply and deeper ``cube_action`` runs a second, cubeless N-ply
        search for those fields after the cube tree has decided; this skips
        it, which saves about half of a 2-ply call and two thirds of a 3-ply
        one. Equities and flags equal ``cube_action``'s. At 1-ply and the
        rollout levels it is ``cube_action`` with those fields dropped.
        Arguments as in :meth:`cube_action`.
        """
        _validate_cube_limit(cube_value, max_cube_value)
        if away1 > 0 or away2 > 0:
            jacoby = False
            beaver = False
        kw = dict(away1=away1, away2=away2, is_crawford=is_crawford, jacoby=jacoby,
                  beaver=beaver, **_cube_limit_kwargs(max_cube_value))
        inner = getattr(self._analyzer, "_inner", self._analyzer)
        if isinstance(inner, _MultiPlyAnalyzer):
            raw = inner.cube_action_analytics(board, cube_value, cube_owner,
                                              decision_only=True, **kw)
        else:
            raw = self._analyzer.cube_action_analytics(board, cube_value, cube_owner, **kw)
        is_beaver = raw.get("is_beaver", False)
        return CubeDecisionResult(
            equity_nd=raw["equity_nd"],
            equity_dt=raw["equity_dt"],
            equity_dp=raw["equity_dp"],
            should_double=raw["should_double"],
            should_take=raw["should_take"],
            optimal_equity=raw["optimal_equity"],
            optimal_action=_optimal_action(raw["should_double"], raw["should_take"], is_beaver),
            eval_level=raw["eval_level"],
            is_beaver=is_beaver,
        )

    def roll_luck(
        self,
        board: list[int],
        die1: int,
        die2: int,
        *,
        cube_value: int = 1,
        cube_owner: str = "centered",
        away1: int = 0,
        away2: int = 0,
        is_crawford: bool = False,
        jacoby: bool = True,
        beaver: bool = True,
        is_opening_roll: bool = False,
    ) -> LuckResult | None:
        """Compute how lucky ``(die1, die2)`` was from ``board``.

        Convenience wrapper: runs the cube analysis this analyzer's eval level
        implies (with per-roll details) and computes luck from it. Callers that
        already hold a ``cube_action(incl_2ply_details=True)`` result should call
        :func:`neongammon.roll_luck` on it directly to avoid a second evaluation.

        ``is_opening_roll=True`` excludes doubles (an opening roll has 15 rolls,
        not 21). Returns ``None`` on a degenerate position. See
        :func:`neongammon.luck.roll_luck` for the meaning of the result.
        """
        cube = self.cube_action(
            board, cube_value, cube_owner,
            away1=away1, away2=away2, is_crawford=is_crawford,
            jacoby=jacoby, beaver=beaver, incl_2ply_details=True,
        )
        return _roll_luck(cube, die1, die2, is_opening_roll=is_opening_roll)


def create_analyzer(
    level: str = "1ply",
    weights: WeightConfig | WeightConfigPair | None = None,
    cubeful: bool = True,
    **kwargs: Any,
) -> BgBotAnalyzer:
    """Convenience factory for creating analyzers."""
    return BgBotAnalyzer(weights=weights, eval_level=level, cubeful=cubeful, **kwargs)
