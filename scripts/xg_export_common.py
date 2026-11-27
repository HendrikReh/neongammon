# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Shared native .xg archive export helpers for XG Batch Analyze scripts."""

from __future__ import annotations

import struct
from pathlib import Path


def _cube_code(cube_value: int, owner: int) -> int:
    """XG cube code: 0 = centered 1-cube; +/-n = 2^n owned by P1 / P2."""
    exp = cube_value.bit_length() - 1
    return 0 if owner == 0 else owner * exp


def derive_half_moves(pre, post, d1: int, d2: int):
    """Exact per-die (from, to) half-move sequence (neongammon points) for pre -> post."""
    from neongammon import possible_single_die_moves

    target = tuple(post)
    orders = [[d1] * k for k in (4, 3, 2, 1)] if d1 == d2 else [[d1, d2], [d2, d1]]

    def dfs(board, dice_left):
        if tuple(board) == target:
            return []
        if not dice_left:
            return None
        for mv in possible_single_die_moves(board, dice_left[0]):
            sub = dfs(mv["board"], dice_left[1:])
            if sub is not None:
                return [(mv["from"], mv["to"])] + sub
        return dfs(board, dice_left[1:])

    best = None
    for order in orders:
        res = dfs(pre, order)
        if res is not None and (best is None or len(res) > len(best)):
            best = res
    return best or []


def _templates(template_path: Path) -> dict:
    """One record of each type from a real .xg archive, to clone."""
    from neongammon import xg_file as xf

    tempxg = xf.XgArchive.load(template_path).get("temp.xg")
    tpl = {}
    for off, rt in xf.iter_records(tempxg):
        tpl.setdefault(rt, tempxg[off:off + xf.TSAVEREC_SIZE])
    return tpl


def _build_cube_rec(tpl: dict, board_mover, actif: int,
                    double: int, take: int, cube_b: int) -> bytes:
    """A tsCube record with the double/take/cube fields set explicitly."""
    from neongammon import xg_file as xf

    return xf.build_cube_record(tpl[xf.TS_CUBE], board_mover, actif=actif,
                                double=double, take=take, cube_b=cube_b)


def _build_move_rec(tpl: dict, pre_mover, post_mover, d1: int, d2: int,
                    half_moves, actif: int, cube_a: int) -> bytes:
    """A tsMove record with the cube-state field set (build_move_record leaves it)."""
    from neongammon import xg_file as xf

    rec = bytearray(xf.build_move_record(
        tpl[xf.TS_MOVE], pre_mover, post_mover, d1, d2, half_moves, actif=actif))
    struct.pack_into("<i", rec, xf._MOVE_CUBE_A, cube_a)
    return bytes(rec)


def write_xg(template_path: Path, stream: bytes, out_path: Path) -> None:
    """Write a rebuilt temp.xg stream and temp.xgi index into a cloned .xg archive."""
    from neongammon import xg_file as xf

    arch = xf.XgArchive.load(template_path)
    arch.set("temp.xg", stream)
    arch.set("temp.xgi", xf.rebuild_xgi(stream))
    arch.save(out_path)
