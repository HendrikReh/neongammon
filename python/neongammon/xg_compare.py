# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Parse eXtreme Gammon (.xg) batch-analyzed game files and compute PR stats.

Used to benchmark Neongammon's evaluations against XG's: feed Neongammon-vs-Neongammon transcripts
(produced by ``scripts/run_sage_vs_sage_games.py``) into XG's Batch Analyze with
"Save Games after analyze" enabled, then run this module on the resulting .xg
files to extract per-game error totals and Performance Rating (PR).

PR = sum(equity errors) / decision count * 500.

Public API:
- ``parse_xg_game(xg_bytes)`` — parse a single-game .xg file into turn dicts.
- ``parse_xg_match(xg_bytes)`` — parse a multi-game .xg match file.
- ``compute_game_pr_stats(turns)`` — sum errors + decisions across both players
  for one game and report PR.
- ``apply_decision_flags(turns)`` — annotate turns with ``is_cube_decision`` /
  ``is_checker_decision`` / ``cube_error`` / ``opp_cube_error`` / ``checker_error``
  using the same filters XG uses to count "real" decisions.

The module reads .xg binary files only via :mod:`neongammon.xg_file`; it does
not call XG and does not require XG or ``bgbot_cpp`` to be loaded.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ._primitives import cubeless_equity, flip_board
from .xg_file import (
    MAGIC_DMLI as MAGIC_DMLI,
    MAGIC_RGMH as MAGIC_RGMH,
    RICH_HEADER_SIZE as RICH_HEADER_SIZE,
    TSAVEREC_SIZE as TSAVEREC_SIZE,
    TS_CUBE as TS_CUBE,
    TS_FOOTER_GAME as TS_FOOTER_GAME,
    TS_FOOTER_MATCH as TS_FOOTER_MATCH,
    TS_HEADER_GAME as TS_HEADER_GAME,
    TS_HEADER_MATCH as TS_HEADER_MATCH,
    TS_MOVE as TS_MOVE,
    XgArchive,
    iter_records,
    parse_cube_record,
    parse_game_footer,
    parse_game_header,
    parse_header,
    parse_match_footer,
    parse_move_record,
    xg_eval_to_probs,
)


# XG analyze_m / analyze_c (per-record) -> our label
_XG_ANALYZE_LEVEL_MAP = {
    1: "1ply", 2: "2ply", 3: "3ply", 4: "4ply", 5: "5ply",
    10: "rollout", 11: "rollout", 12: "rollout",
}


# ---------------------------------------------------------------------------
# Perspective helpers
# ---------------------------------------------------------------------------


def _mover_to_user_board(board: list[int], actif: int) -> list[int]:
    """Convert a mover's-perspective board to user (player1) perspective.

    actif: 1 = player1 (user); 2 or any negative value = player2 (bot).
    """
    if actif < 0 or actif == 2:
        return flip_board(board)
    return list(board)


def _active_player_from_actif(actif: int) -> str:
    """Normalise XG ActiveP into ``"user"`` / ``"bot"``.

    Supports both legacy exports (±1) and XG-native (1/2) encodings.
    """
    if actif == 2 or actif < 0:
        return "bot"
    return "user"


# ---------------------------------------------------------------------------
# Archive & record decoding (delegated to neongammon.xg_file)
# ---------------------------------------------------------------------------


_RECORD_PARSERS = {
    TS_HEADER_MATCH: parse_header,
    TS_HEADER_GAME: parse_game_header,
    TS_CUBE: parse_cube_record,
    TS_MOVE: parse_move_record,
    TS_FOOTER_GAME: parse_game_footer,
    TS_FOOTER_MATCH: parse_match_footer,
}


def _decode_records(xg_bytes: bytes) -> list[dict[str, Any]]:
    """Extract ``temp.xg`` with CRC-32 verification and decode all TSaveRecs."""
    archive = XgArchive.from_bytes(xg_bytes)
    game_data = archive.get("temp.xg")
    if game_data is None:
        raise ValueError("temp.xg not found in XG archive")

    records: list[dict[str, Any]] = []
    for off, entry_type in iter_records(game_data):
        parser = _RECORD_PARSERS.get(entry_type)
        if parser is None:
            continue
        parsed = parser(game_data, off)
        parsed["type"] = entry_type
        records.append(parsed)
    return records


def _infer_analysis_level(records: list[dict[str, Any]]) -> str:
    """Infer the analysis level from per-record analyze_m / analyze_c values."""
    levels: Counter[int] = Counter()
    for r in records:
        if r.get("type") == TS_MOVE:
            am = r.get("analyze_m", -1)
            if am > 0:
                levels[am] += 1
        elif r.get("type") == TS_CUBE:
            ac = r.get("analyze_c", -1)
            if ac > 0:
                levels[ac] += 1
    if not levels:
        return "xg-2ply"
    most_common = levels.most_common(1)[0][0]
    base = _XG_ANALYZE_LEVEL_MAP.get(most_common, f"{most_common}ply")
    return f"xg-{base}"


def _group_records_by_game(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Partition records into per-game lists (HEADER_GAME .. FOOTER_GAME)."""
    games: list[list[dict[str, Any]]] = []
    current_game: list[dict[str, Any]] = []
    in_game = False
    for rec in records:
        if rec["type"] == TS_HEADER_GAME:
            current_game = [rec]
            in_game = True
        elif rec["type"] == TS_FOOTER_GAME:
            if in_game:
                current_game.append(rec)
                games.append(current_game)
            current_game = []
            in_game = False
        elif in_game:
            current_game.append(rec)
    return games


# ---------------------------------------------------------------------------
# Turn assembly
# ---------------------------------------------------------------------------


def _build_cube_analysis(cube_rec: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert a parsed tsCube record to the analyzer-style cube dict."""
    if not cube_rec:
        return None
    eval_nd = cube_rec["eval_nd"]
    if all(abs(v) < 1e-10 for v in eval_nd[:6]):
        return None

    # flag_double is XG's cube-action recommendation:
    #   0 = no double
    #   1 = double (take recommended)
    #   2 = double, pass (probably)
    #   -1 = too good to double (still NOT recommending double)
    #   -100 = no analysis (record skipped)
    # Treat only positive values as a recommendation to double.
    should_double = cube_rec["flag_double"] > 0
    nd = cube_rec["equity_nd"]
    dt = cube_rec["equity_dt"]
    dp = cube_rec["equity_dp"]
    probs = xg_eval_to_probs(eval_nd)
    should_take = dt <= dp
    if should_double:
        optimal_action = "double/take" if should_take else "double/pass"
    else:
        optimal_action = "no double"
    optimal = max(nd, min(dt, dp)) if should_double else nd
    return {
        "probs": probs,
        "cubeless_equity": cubeless_equity(probs),
        "cubeful_equity": nd,
        "equity_nd": nd,
        "equity_dt": dt,
        "equity_dp": dp,
        "should_double": should_double,
        "should_take": should_take,
        "optimal_action": optimal_action,
        "optimal_equity": optimal,
    }


def _build_checker_analysis(
    move_rec: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """Convert a parsed tsMove record to a list of move dicts (mover's frame).

    XG stores moves in its own recommendation order (sorted by eval level
    first, then by cubeful equity within the same level), so ``moves[0]`` is
    XG's #1 recommendation. We trust that order — do NOT re-sort by the
    stored ``equity`` field, since the cubeful equity at offset 6 doesn't
    capture eval-level priority and naive sorting can promote a shallowly-
    evaluated higher-cubeful move above XG's actual recommendation.
    """
    raw_moves = move_rec.get("moves")
    if not raw_moves:
        return None
    moves: list[dict[str, Any]] = []
    for m in raw_moves:
        xg_eval = m["eval"]
        moves.append({
            "board": list(m["board"]),
            "equity": float(xg_eval[6]),
            "probs": xg_eval_to_probs(xg_eval),
        })
    best_eq = moves[0]["equity"]
    for m in moves:
        m["equity_diff"] = m["equity"] - best_eq
    return moves


def _make_double_pass_turn(
    turn_number: int,
    pending_cube: dict[str, Any],
    analysis_level: str,
) -> dict[str, Any]:
    """Build a turn dict for a double/pass (no checker play occurred)."""
    return {
        "turn_number": turn_number,
        "player": _active_player_from_actif(pending_cube["actif"]),
        "board_before": None,
        "board_after": None,
        "dice": None,
        "move": None,
        "cube_action": "double/pass",
        "cube_analysis": _build_cube_analysis(pending_cube),
        "checker_analysis": None,
        "analysis_level": analysis_level if pending_cube.get("analyze_c", 0) > 0 else None,
        "checker_error": None,
        "cube_error": pending_cube.get("err_cube"),
        "opp_cube_error": None,
    }


def _assemble_turns(
    game_records: list[dict[str, Any]],
    analysis_level: str,
) -> list[dict[str, Any]]:
    """Assemble cube + move records into per-turn dicts.

    XG emits separate CUBE and MOVE records; this function pairs them so each
    turn dict carries the cube action (if any) alongside the checker move.
    """
    turns: list[dict[str, Any]] = []
    turn_number = 0
    pending_cube: dict[str, Any] | None = None

    for rec in game_records:
        if rec["type"] == TS_CUBE:
            # Trailing cube + new cube => the prior was a double/pass.
            if pending_cube is not None and pending_cube["double"] == 1:
                turn_number += 1
                turns.append(
                    _make_double_pass_turn(turn_number, pending_cube, analysis_level)
                )
            pending_cube = rec

        elif rec["type"] == TS_MOVE:
            turn_number += 1
            actif = rec["actif"]
            player = _active_player_from_actif(actif)

            board_before = list(rec["position_i"])
            board_after = _mover_to_user_board(rec["position_end"], actif)

            cube_action: str | None = None
            cube_analysis: dict[str, Any] | None = None
            cube_error = None
            if pending_cube is not None:
                if pending_cube["double"] == 1:
                    cube_action = "double/take" if pending_cube["take"] == 1 else "double/pass"
                elif pending_cube["double"] == 0:
                    cube_action = "no double"
                else:
                    cube_action = None
                if cube_action is not None:
                    cube_analysis = _build_cube_analysis(pending_cube)
                    cube_error = pending_cube.get("err_cube")
                pending_cube = None

            checker_analysis = _build_checker_analysis(rec)
            checker_error = rec.get("err_move")

            die1, die2 = rec["dice"]
            dice = None if (die1 == 0 and die2 == 0) else [die1, die2]
            has_checker_analysis = rec.get("analyze_m", 0) > 0
            has_cube_analysis = cube_analysis is not None

            turns.append({
                "turn_number": turn_number,
                "player": player,
                "board_before": board_before,
                "board_after": board_after,
                "dice": dice,
                "move": None,
                "cube_action": cube_action,
                "cube_analysis": cube_analysis,
                "checker_analysis": checker_analysis,
                "analysis_level": analysis_level if (has_checker_analysis or has_cube_analysis) else None,
                "checker_error": checker_error,
                "cube_error": cube_error,
                "opp_cube_error": None,
            })

    # Trailing pending cube => game-ending double/pass.
    if pending_cube is not None and pending_cube["double"] == 1:
        turn_number += 1
        turns.append(
            _make_double_pass_turn(turn_number, pending_cube, analysis_level)
        )

    # Fill in board_before for double/pass turns from the previous turn.
    for i, turn in enumerate(turns):
        if turn["board_before"] is None and i > 0:
            prev = turns[i - 1]
            if prev.get("board_after"):
                turn["board_before"] = list(prev["board_after"])

    return turns


# ---------------------------------------------------------------------------
# Decision filters (XG-style: skip "trivial" or forced decisions)
# ---------------------------------------------------------------------------


def _is_trivial_cube(analysis: dict[str, Any]) -> bool:
    """A cube position is trivial when the action is so obvious it shouldn't
    count as a real decision (XG's filter)."""
    nd = analysis.get("equity_nd")
    dt = analysis.get("equity_dt")
    dp = analysis.get("equity_dp")
    if nd is None or dt is None or dp is None:
        return False
    if abs(nd - dt) < 0.001:
        return True
    if nd - dt > 0.200:
        return True
    if nd - dp > 0.200:
        return True
    if nd < -0.900 and dt < -0.900:
        return True
    return False


def _is_trivial_take_pass(analysis: dict[str, Any]) -> bool:
    """A take/pass decision is trivial when DT ≈ DP (no real choice)."""
    dt = analysis.get("equity_dt")
    dp = analysis.get("equity_dp")
    if dt is None or dp is None:
        return False
    return abs(dt - dp) < 0.001


def _is_cube_decision(entry: dict[str, Any], had_cube_access: bool) -> bool:
    """Whether the cube_action on this turn counts as a PR decision."""
    if not had_cube_access:
        return False
    cube_action = entry.get("cube_action")
    if cube_action is None or cube_action == "resign":
        return False
    analysis = entry.get("cube_analysis")
    if analysis is None:
        return False
    if _is_trivial_cube(analysis):
        # Trivial position but a wrong choice still counts.
        if compute_cube_error(entry) < 0.001:
            return False
    return True


def _is_checker_decision(entry: dict[str, Any]) -> bool:
    """Whether the checker play on this turn counts as a PR decision."""
    checker_analysis = entry.get("checker_analysis")
    if not checker_analysis or len(checker_analysis) < 2:
        return False
    first = checker_analysis[0]
    last = checker_analysis[-1]
    best_eq = first.get("equity") if isinstance(first, dict) else getattr(first, "equity", None)
    worst_eq = last.get("equity") if isinstance(last, dict) else getattr(last, "equity", None)
    if best_eq is not None and worst_eq is not None:
        if abs(best_eq - worst_eq) < 0.001:
            return False
    return True


# ---------------------------------------------------------------------------
# Error computation
# ---------------------------------------------------------------------------


def compute_cube_error(entry: dict[str, Any]) -> float:
    """Equity lost on the doubler's no-double / double choice (>= 0)."""
    analysis = entry.get("cube_analysis")
    if not analysis:
        return 0.0
    cube_action = entry.get("cube_action")
    if cube_action is None or cube_action == "resign":
        return 0.0
    nd = analysis["equity_nd"]
    dt = analysis["equity_dt"]
    dp = analysis["equity_dp"]
    optimal = analysis.get("optimal_equity", max(nd, min(dt, dp)))
    doubled = cube_action in ("double/take", "double/pass")
    actual = min(dt, dp) if doubled else nd
    return max(0.0, optimal - actual)


def compute_opp_cube_error(entry: dict[str, Any]) -> float:
    """Equity lost on the responder's take/pass choice (>= 0)."""
    cube_action = entry.get("cube_action")
    if cube_action not in ("double/take", "double/pass"):
        return 0.0
    analysis = entry.get("cube_analysis")
    if not analysis:
        return 0.0
    dt = analysis["equity_dt"]
    dp = analysis["equity_dp"]
    optimal = min(dt, dp)
    actual = dt if cube_action == "double/take" else dp
    return max(0.0, actual - optimal)


def compute_checker_error(entry: dict[str, Any]) -> float:
    """Equity lost on the checker move (>= 0).

    Looks up ``board_after`` in ``checker_analysis`` and returns the equity gap
    vs the best move. Bot turns have ``board_after`` in user perspective; the
    analysis boards in turn dicts are in mover's perspective, so we flip
    ``board_after`` for bot turns to match.
    """
    analysis = entry.get("checker_analysis")
    if not analysis or len(analysis) < 2:
        return 0.0
    played_board = entry.get("board_after")
    if played_board is None:
        return 0.0
    if entry.get("player") == "bot":
        played_board = flip_board(played_board)
    played_tuple = tuple(played_board)
    for move in analysis:
        if tuple(move.get("board", [])) == played_tuple:
            return abs(move.get("equity_diff", 0.0))
    return 0.0


# ---------------------------------------------------------------------------
# Decision-flag application + per-game aggregation
# ---------------------------------------------------------------------------


def apply_decision_flags(turns: list[dict[str, Any]]) -> None:
    """Annotate each turn with ``is_cube_decision`` / ``is_checker_decision``
    and fill in ``cube_error`` / ``opp_cube_error`` / ``checker_error``."""
    cube_owner = "centered"  # relative to the active player at each turn

    for entry in turns:
        player = entry["player"]
        had_cube_access = (
            cube_owner == "centered"
            or (cube_owner == "player" and player == "user")
            or (cube_owner == "opponent" and player == "bot")
        )

        entry["is_cube_decision"] = _is_cube_decision(entry, had_cube_access)
        if entry.get("cube_analysis") is not None and entry.get("cube_action") is not None:
            entry["cube_error"] = compute_cube_error(entry)
            entry["opp_cube_error"] = compute_opp_cube_error(entry)
        elif not had_cube_access or entry.get("cube_action") in (None, "resign"):
            entry["cube_error"] = 0.0
            entry["opp_cube_error"] = 0.0

        entry["is_checker_decision"] = _is_checker_decision(entry)
        entry["checker_error"] = (
            compute_checker_error(entry)
            if entry.get("checker_analysis") is not None
            else 0.0
        )

        # Update cube ownership for the next turn.
        if entry.get("cube_action") == "double/take":
            cube_owner = "player" if player == "bot" else "opponent"


def compute_game_pr_stats(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-player error totals + decision counts over one game.

    Calls ``apply_decision_flags`` first (idempotent in effect; rewrites flag
    fields). Returns:

    - ``user_err`` / ``user_dec``: P1 error total (cube + checker) and decision count
    - ``bot_err``  / ``bot_dec``:  P2 error total and decision count
    - ``total_err`` / ``total_dec``: sums across both players
    - ``pr``: ``(total_err / total_dec) * 500``, or ``float('nan')`` if no
      decisions were recorded
    """
    apply_decision_flags(turns)

    user_err = 0.0
    user_dec = 0
    bot_err = 0.0
    bot_dec = 0

    for entry in turns:
        player = entry.get("player")
        is_user = player == "user"

        if entry.get("is_cube_decision"):
            err = entry.get("cube_error", 0.0) or 0.0
            if is_user:
                user_dec += 1
                user_err += err
            else:
                bot_dec += 1
                bot_err += err

        # opp_cube_error belongs to the OTHER player (the responder).
        cube_action = entry.get("cube_action")
        analysis = entry.get("cube_analysis")
        if (
            cube_action in ("double/take", "double/pass")
            and analysis is not None
            and not _is_trivial_take_pass(analysis)
        ):
            opp_err = entry.get("opp_cube_error", 0.0) or 0.0
            if is_user:
                bot_dec += 1
                bot_err += opp_err
            else:
                user_dec += 1
                user_err += opp_err

        if entry.get("is_checker_decision"):
            err = entry.get("checker_error", 0.0) or 0.0
            if is_user:
                user_dec += 1
                user_err += err
            else:
                bot_dec += 1
                bot_err += err

    total_err = user_err + bot_err
    total_dec = user_dec + bot_dec
    pr = (total_err / total_dec * 500.0) if total_dec > 0 else float("nan")

    return {
        "user_err": user_err,
        "user_dec": user_dec,
        "bot_err": bot_err,
        "bot_dec": bot_dec,
        "total_err": total_err,
        "total_dec": total_dec,
        "pr": pr,
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def parse_xg_game(xg_bytes: bytes) -> list[dict[str, Any]]:
    """Parse a single-game .xg file and return a list of turn dicts.

    Raises ``ValueError`` if the file is not a valid XG archive or contains
    more than one game.
    """
    records = _decode_records(xg_bytes)
    games = _group_records_by_game(records)
    if len(games) != 1:
        raise ValueError(
            f"Expected exactly 1 game in .xg file, found {len(games)}"
        )
    return _assemble_turns(games[0], _infer_analysis_level(records))


def parse_xg_match(xg_bytes: bytes) -> dict[str, Any]:
    """Parse a multi-game (match) .xg file.

    Returns a dict with match-level metadata and a list of per-game records
    in play order::

        {
            "match_length": int,         # 0 if money game
            "player1": str,
            "player2": str,
            "jacoby": bool,
            "beaver": bool,
            "analysis_level": str,       # e.g. "xg-2ply"
            "games": [
                {
                    "game_number": int,
                    "score1_start": int,     # P1 score at game start
                    "score2_start": int,     # P2 score at game start
                    "crawford_apply": bool,  # True for the Crawford game
                    "turns": [...],          # output of _assemble_turns
                },
                ...
            ],
        }

    Single-game .xg files parse fine too (``games`` will be a list of one).
    Raises ``ValueError`` if the file isn't a valid XG archive.
    """
    records = _decode_records(xg_bytes)
    analysis_level = _infer_analysis_level(records)

    match_header: dict[str, Any] | None = None
    for r in records:
        if r.get("type") == TS_HEADER_MATCH:
            match_header = r
            break

    games_records = _group_records_by_game(records)
    if not games_records:
        raise ValueError("No games found in .xg file")

    games_out: list[dict[str, Any]] = []
    for game_recs in games_records:
        header = game_recs[0]
        turns = _assemble_turns(game_recs, analysis_level)
        games_out.append({
            "game_number": header.get("game_number", len(games_out) + 1),
            "score1_start": header.get("score1", 0),
            "score2_start": header.get("score2", 0),
            "crawford_apply": header.get("crawford", False),
            "turns": turns,
        })

    # Normalize XG's sentinel for "money / unlimited" (99999) to 0 so
    # downstream code can treat ``match_length == 0`` as the universal
    # "no match state" signal.
    raw_match_length = (
        match_header.get("match_length", 0) if match_header else 0
    )
    match_length = 0 if raw_match_length >= 99999 else int(raw_match_length)

    return {
        "match_length": match_length,
        "player1": match_header.get("player1", "Player 1") if match_header else "Player 1",
        "player2": match_header.get("player2", "Player 2") if match_header else "Player 2",
        "jacoby": match_header.get("jacoby", True) if match_header else True,
        "beaver": match_header.get("beaver", True) if match_header else True,
        "analysis_level": analysis_level,
        "games": games_out,
    }
