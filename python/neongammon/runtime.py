# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Runtime capability introspection and native thread-budget observation.

This module is pure Python at import time and does not eagerly import
``bgbot_cpp``. Call :func:`runtime_info` to inspect installed capabilities and
:func:`observe_thread_budgets` to verify native thread budgets without
monkey-patching private submodules.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib.util import find_spec
import platform
from typing import Any

from ._native import EXPECTED_ABI_VERSION, load_native
from ._primitives import MAX_MATCH_LENGTH
from .weights import MODELS, PRODUCTION_MODEL, bearoff_db_path

_FIXED_DEPTH_LEVELS: tuple[str, ...] = ("1ply", "2ply", "3ply", "4ply")
_ROLLOUT_PRESET_LEVELS: tuple[str, ...] = ("truncated1", "truncated2", "truncated3")
_EVAL_LEVELS: tuple[str, ...] = (*_FIXED_DEPTH_LEVELS, *_ROLLOUT_PRESET_LEVELS, "rollout")


@dataclass(frozen=True)
class ThreadBudgetEvent:
    """A single native thread-budget observation emitted before native execution."""

    operation: str
    n_threads: int
    plies: int | None = None
    cubeful: bool | None = None


_THREAD_BUDGET_OBSERVERS: contextvars.ContextVar[
    tuple[Callable[[ThreadBudgetEvent], None], ...]
] = contextvars.ContextVar("neongammon_thread_budget_observers", default=())


def _notify_thread_budget(
    operation: str,
    n_threads: int,
    *,
    plies: int | None = None,
    cubeful: bool | None = None,
) -> None:
    """Emit a :class:`ThreadBudgetEvent` to all active observers in this context."""
    observers = _THREAD_BUDGET_OBSERVERS.get()
    if not observers:
        return
    event = ThreadBudgetEvent(
        operation=operation,
        n_threads=int(n_threads),
        plies=plies,
        cubeful=cubeful,
    )
    for callback in observers:
        callback(event)


@contextlib.contextmanager
def observe_thread_budgets(
    callback: Callable[[ThreadBudgetEvent], None] | None = None,
) -> Iterator[list[ThreadBudgetEvent]]:
    """Observe native thread budgets handed out during the ``with`` block.

    Covers both cubeful native execution (``checker_play_nply``,
    ``checker_play_1ply``, ``cube_decision_nply``, ``cubeful_equity_nply``,
    ``score_checker_endpoints``, batch and rollout calls) and cubeless N-ply
    native execution (``MultiPlyStrategy`` via ``multipy_evaluate``).

    Yields a list populated with :class:`ThreadBudgetEvent` instances as calls
    occur. When ``callback`` is provided, it is also invoked synchronously for
    each event. Isolated per thread / ``contextvars`` context and safe to nest.
    """
    events: list[ThreadBudgetEvent] = []

    def _record(event: ThreadBudgetEvent) -> None:
        events.append(event)
        if callback is not None:
            callback(event)

    current = _THREAD_BUDGET_OBSERVERS.get()
    token = _THREAD_BUDGET_OBSERVERS.set((*current, _record))
    try:
        yield events
    finally:
        _THREAD_BUDGET_OBSERVERS.reset(token)


def runtime_info() -> dict[str, Any]:
    """Return a JSON-serializable summary of NeonGammon runtime capabilities."""
    from . import __version__

    db_path = bearoff_db_path()
    info: dict[str, Any] = {
        "version": __version__,
        "platform": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "production_model": PRODUCTION_MODEL,
        "models": list(MODELS),
        "bearoff_db_available": db_path is not None,
        "bearoff_db_path": db_path,
        "eval_levels": list(_EVAL_LEVELS),
        "fixed_depth_levels": list(_FIXED_DEPTH_LEVELS),
        "rollout_preset_levels": list(_ROLLOUT_PRESET_LEVELS),
        "max_away": MAX_MATCH_LENGTH,
        "serial_cubeful_1ply": True,
        "expected_abi_version": EXPECTED_ABI_VERSION,
        "mlx_installed": find_spec("mlx") is not None,
        "supports_difficulty_policy": True,
        "supports_shared_models": True,
        "supports_cube_decision": True,
        "supports_thread_budget_observer": True,
    }

    try:
        native = load_native()
    except ImportError as exc:
        info.update({
            "native_available": False,
            "native_path": None,
            "abi_version": None,
            "accelerate_enabled": None,
            "cubeful_filtered_plies": None,
            "money_contact_cube_life": None,
            "money_contact_cube_life_pip_slope": None,
            "native_error": str(exc),
        })
        return info

    info.update({
        "native_available": True,
        "native_path": getattr(native, "__file__", None),
        "abi_version": int(native.ABI_VERSION),
        "accelerate_enabled": bool(native.ACCELERATE_ENABLED),
        "cubeful_filtered_plies": sorted(int(p) for p in native.CUBEFUL_FILTERED_PLIES),
        "money_contact_cube_life": {
            int(k): float(v) for k, v in dict(native.MONEY_CONTACT_CUBE_LIFE).items()
        },
        "money_contact_cube_life_pip_slope": {
            int(k): float(v)
            for k, v in dict(native.MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE).items()
        },
        "native_error": None,
    })
    return info
