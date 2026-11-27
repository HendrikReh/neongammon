# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Compact legal move arrays, with compatibility for older engine builds."""
import importlib.util
from pathlib import Path

import bgbot_cpp
import numpy as np

if __package__:
    from ._primitives import validate_board
else:
    # Support isolated spec_from_file_location loading (tests/test_move_arrays.py
    # and scripts/benchmark_move_arrays.py) without importing the package root.
    _prim_spec = importlib.util.spec_from_file_location(
        "_neongammon_primitives", Path(__file__).with_name("_primitives.py")
    )
    _prim_mod = importlib.util.module_from_spec(_prim_spec)
    _prim_spec.loader.exec_module(_prim_mod)
    validate_board = _prim_mod.validate_board


ROLLS = tuple((first, second) for first in range(1, 7) for second in range(first, 7))


def possible_moves_all(board):
    """Return ``(boards, masks)`` for all 21 rolls in the mover's perspective.

    ``boards`` is a contiguous ``int8`` array of shape ``(N, 26)``; ``masks``
    is contiguous ``uint32`` of shape ``(N,)``. Bit k indicates legality under
    ``ROLLS[k]``. Rows preserve first occurrence while iterating those rolls
    and each roll's existing sorted native move list. A blocked roll includes
    the unchanged board. Each result owns its storage and is independently
    writable. This function performs no evaluation and uses no model weights.

    Older extensions transparently use the equivalent 21-call construction.
    """
    values = validate_board(board)
    native = getattr(bgbot_cpp, "possible_moves_all", None)
    if native is not None:
        return native(values)
    found = {}
    for k, (first, second) in enumerate(ROLLS):
        for after in bgbot_cpp.possible_moves(values, first, second):
            key = tuple(after)
            found[key] = found.get(key, 0) | (1 << k)
    return (np.asarray(list(found), dtype=np.int8).reshape((-1, 26)),
            np.fromiter(found.values(), dtype=np.uint32, count=len(found)))
