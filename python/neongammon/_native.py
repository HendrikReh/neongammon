# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Centralized native extension loader and ABI compatibility check."""

from __future__ import annotations

import importlib
from typing import Any

#: Expected ``bgbot_cpp.ABI_VERSION`` matching this Python package.
EXPECTED_ABI_VERSION: int = 1

_REQUIRED_SYMBOLS: tuple[str, ...] = (
    "ABI_VERSION",
    "ACCELERATE_ENABLED",
    "CUBEFUL_FILTERED_PLIES",
    "MONEY_CONTACT_CUBE_LIFE",
    "MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE",
    "RoutingManifest",
    "_get_met_after",
    "_validate_met_away_scores",
    "_validate_batch_move_filter",
)

_validated_module: Any = None


def _stale_error(module: Any, detail: str) -> ImportError:
    location = getattr(module, "__file__", "<unknown location>")
    return ImportError(
        "NeonGammon requires a matching bgbot_cpp extension. "
        f"The extension at {location} {detail}. "
        "Rebuild the local extension or reinstall the NeonGammon wheel."
    )


def load_native() -> Any:
    """Import ``bgbot_cpp`` and validate its ABI once per loaded module object."""
    global _validated_module
    module = importlib.import_module("bgbot_cpp")
    if _validated_module is module:
        return module

    abi = getattr(module, "ABI_VERSION", None)
    if abi != EXPECTED_ABI_VERSION:
        if abi is None:
            raise _stale_error(
                module,
                f"is missing ABI_VERSION (expected {EXPECTED_ABI_VERSION})",
            )
        raise _stale_error(
            module,
            f"has ABI_VERSION={abi!r}, expected {EXPECTED_ABI_VERSION}",
        )

    for name in _REQUIRED_SYMBOLS:
        if not hasattr(module, name):
            raise _stale_error(module, f"is missing {name}")

    _validated_module = module
    return module


def extension_constant(name: str) -> Any:
    """Fetch a required symbol from the validated ``bgbot_cpp`` extension."""
    module = load_native()
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise _stale_error(module, f"is missing {name}") from exc
