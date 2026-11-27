# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Lean CLI for ``python -m neongammon`` / ``neongammon`` (info, probe, eval)."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import json
import math
import sys
from typing import Sequence

from ._primitives import STARTING_BOARD, validate_board
from .data import board_from_gnubg_position_string
from .runtime import observe_thread_budgets, runtime_info


def _decode_gnubg_position_id(pos_id: str) -> list[int]:
    """Decode either a 14-char base64 GNUbg Position ID or 20-char ``nn`` key."""
    text = pos_id.strip()
    if len(text) == 20 and all("A" <= ch <= "P" for ch in text):
        return board_from_gnubg_position_string(text)
    if len(text) == 14:
        try:
            raw = base64.b64decode(text + "==", validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid 14-character GNUbg position ID: {pos_id!r}") from exc
        if len(raw) != 10:
            raise ValueError(f"Invalid 14-character GNUbg position ID: {pos_id!r}")
        # Reuse the 20-char nibble parser by encoding the 10 key bytes into A..P pairs.
        nn_str = "".join(
            chr(ord("A") + ((b >> 4) & 0xF)) + chr(ord("A") + (b & 0xF))
            for b in raw
        )
        return board_from_gnubg_position_string(nn_str)
    raise ValueError(
        "GNUbg position ID must be either 14 base64 characters or 20 A-P key characters"
    )


def _parse_board(board_arg: str | None, gnubg_id: str | None) -> list[int]:
    if board_arg is not None and gnubg_id is not None:
        raise ValueError("Specify at most one of --board and --gnubg-id")
    if gnubg_id is not None:
        return list(validate_board(_decode_gnubg_position_id(gnubg_id)))
    if board_arg is None:
        return list(STARTING_BOARD)
    cleaned = board_arg.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    tokens = [tok for tok in cleaned.replace(",", " ").split() if tok]
    try:
        values = [int(tok) for tok in tokens]
    except ValueError as exc:
        raise ValueError("Board entries must be integers") from exc
    return list(validate_board(values))


def _cmd_info(args: argparse.Namespace) -> int:
    info = runtime_info()
    if args.json:
        print(json.dumps(info, indent=2, sort_keys=True))
        return 0
    print(f"NeonGammon {info['version']}")
    print(f"  Platform:      {info['platform']} {info['machine']} (Python {info['python_version']})")
    print(f"  Native ext:    {'available' if info['native_available'] else 'UNAVAILABLE'}")
    if info["native_available"]:
        print(f"  Native path:   {info['native_path']}")
        print(f"  ABI version:   {info['abi_version']} (expected {info['expected_abi_version']})")
        print(f"  Accelerate:    {info['accelerate_enabled']}")
    elif info.get("native_error"):
        print(f"  Native error:  {info['native_error']}")
    print(f"  Model:         {info['production_model']} ({len(info['models'])} registered)")
    print(f"  Bearoff DB:    {'available' if info['bearoff_db_available'] else 'missing'}")
    print(f"  MLX package:   {'installed' if info['mlx_installed'] else 'not installed'}")
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    from .facade import BgBotAnalyzer
    from .weights import default_weights

    info = runtime_info()
    if not info["native_available"]:
        payload = {"ok": False, "error": info.get("native_error") or "native extension unavailable", "runtime": info}
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"PROBE FAILED: {payload['error']}", file=sys.stderr)
        return 1

    try:
        weights = default_weights()
        weights.validate()
        analyzer = BgBotAnalyzer(
            weights=weights,
            eval_level=args.level,
            parallel_threads=args.threads,
        )
        with observe_thread_budgets() as events:
            checker = analyzer.checker_play(list(STARTING_BOARD), 3, 1, beaver=False)
            cube = analyzer.cube_action(list(STARTING_BOARD), beaver=False)
        if not checker.moves:
            raise RuntimeError("checker_play returned no moves")
        best = checker.moves[0]
        if not math.isfinite(best.equity) or not math.isfinite(cube.optimal_equity):
            raise RuntimeError("non-finite equity returned by probe")
        if args.threads > 0 and any(e.n_threads > args.threads for e in events):
            raise RuntimeError(
                f"observed thread budget exceeded requested {args.threads}: {events}"
            )
    except Exception as exc:
        payload = {"ok": False, "error": str(exc), "runtime": info}
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"PROBE FAILED: {exc}", file=sys.stderr)
        return 1

    payload = {
        "ok": True,
        "level": args.level,
        "threads": args.threads,
        "runtime": info,
        "observed_thread_budgets": [asdict(e) for e in events],
        "checker_best_equity": best.equity,
        "checker_eval_level": best.eval_level,
        "cube_optimal_action": cube.optimal_action,
        "cube_optimal_equity": cube.optimal_equity,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"PROBE OK: NeonGammon {info['version']} (ABI {info['abi_version']}, "
            f"model={info['production_model']}, level={args.level}, "
            f"best_31={best.equity:+.4f}, cube={cube.optimal_action})"
        )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .facade import BgBotAnalyzer

    board = _parse_board(args.board, args.gnubg_id)
    analyzer = BgBotAnalyzer(
        eval_level=args.level,
        cubeful=not args.cubeless,
        parallel_threads=args.threads,
    )
    context = dict(
        cube_value=args.cube_value,
        cube_owner=args.cube_owner,
        away1=args.away1,
        away2=args.away2,
        is_crawford=args.crawford,
        jacoby=not args.no_jacoby,
        beaver=not args.no_beaver,
        max_cube_value=args.max_cube,
    )

    if args.dice is not None:
        d1, d2 = args.dice
        result = analyzer.checker_play(board, d1, d2, **context)
        if args.json:
            data = asdict(result)
            if args.top > 0:
                data["moves"] = data["moves"][: args.top]
            print(json.dumps(data, indent=2))
            return 0
        print(f"Checker play ({d1}-{d2}, {result.eval_level}):")
        limit = args.top if args.top > 0 else len(result.moves)
        for idx, move in enumerate(result.moves[:limit], start=1):
            p = move.probs
            print(
                f"  {idx:2d}. eq={move.equity:+.4f} (diff={move.equity_diff:+.4f}) "
                f"W={p.win:.3f} G={p.gammon_win:.3f}/{p.gammon_loss:.3f} [{move.eval_level}]"
            )
        return 0

    cube = analyzer.cube_action(board, **context)
    if args.json:
        print(json.dumps(asdict(cube), indent=2))
        return 0
    p = cube.probs
    print(f"Cube action ({cube.eval_level}): {cube.optimal_action} (opt={cube.optimal_equity:+.4f})")
    print(
        f"  ND={cube.equity_nd:+.4f}  DT={cube.equity_dt:+.4f}  DP={cube.equity_dp:+.4f}  "
        f"CL={cube.cubeless_equity:+.4f}"
    )
    print(
        f"  W={p.win:.3f}  GW={p.gammon_win:.3f}  BGW={p.backgammon_win:.3f}  "
        f"GL={p.gammon_loss:.3f}  BGL={p.backgammon_loss:.3f}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neongammon",
        description="NeonGammon neural-network backgammon engine CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Show version and runtime capability information.")
    p_info.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_info.set_defaults(func=_cmd_info)

    p_probe = sub.add_parser("probe", help="Verify native extension, weights, and live evaluation.")
    p_probe.add_argument("--level", default="1ply", help="Evaluation level to probe (default: 1ply).")
    p_probe.add_argument("--threads", type=int, default=1, help="Thread budget (default: 1).")
    p_probe.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_probe.set_defaults(func=_cmd_probe)

    p_eval = sub.add_parser("eval", help="Evaluate a position (cube decision, or checker play with --dice).")
    p_eval.add_argument("--board", default=None, help="26 comma/space-separated ints (default: starting position).")
    p_eval.add_argument("--gnubg-id", default=None, help="14-char GNUbg position ID or 20-char nn key.")
    p_eval.add_argument("--dice", type=int, nargs=2, metavar=("D1", "D2"), default=None, help="Dice roll (1-6 1-6).")
    p_eval.add_argument("--level", default="2ply", help="Evaluation level (default: 2ply).")
    p_eval.add_argument("--cube-value", type=int, default=1, help="Current cube value (default: 1).")
    p_eval.add_argument("--cube-owner", default="centered", choices=("centered", "player", "opponent"))
    p_eval.add_argument("--away1", type=int, default=0, help="Player away score (0 = money).")
    p_eval.add_argument("--away2", type=int, default=0, help="Opponent away score (0 = money).")
    p_eval.add_argument("--crawford", action="store_true", help="Mark game as Crawford.")
    p_eval.add_argument("--no-jacoby", action="store_true", help="Disable Jacoby rule in money play.")
    p_eval.add_argument("--no-beaver", action="store_true", help="Disable beavers in money play.")
    p_eval.add_argument("--max-cube", type=int, default=0, help="Maximum cube value (0 = unlimited).")
    p_eval.add_argument("--cubeless", action="store_true", help="Run cubeless analysis.")
    p_eval.add_argument("--threads", type=int, default=1, help="Thread budget (default: 1).")
    p_eval.add_argument("--top", type=int, default=5, help="Max checker moves to display (0 = all).")
    p_eval.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_eval.set_defaults(func=_cmd_eval)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
