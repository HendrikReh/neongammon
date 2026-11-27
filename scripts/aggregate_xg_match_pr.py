# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Match-play wrapper for ``aggregate_xg_pr.py --mode match``.

Preserves CLI and module-level compatibility with the original standalone
``aggregate_xg_match_pr.py`` script while sharing all aggregation, re-analysis,
and rollout logic with ``aggregate_xg_pr.py``.

Usage:

    python scripts/aggregate_xg_match_pr.py
    python scripts/aggregate_xg_match_pr.py --re-eval-level truncated2
    python scripts/aggregate_xg_match_pr.py --skip-rollouts
    python scripts/aggregate_xg_match_pr.py --limit 5
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from aggregate_xg_pr import (  # noqa: E402, F401
    Dispute,
    _DEFAULT_MATCH_DIR as _DEFAULT_DIR,
    _DEFAULT_RE_EVAL_LEVEL,
    _DEFAULT_ROLLOUT_FILENAME,
    _DEFAULT_THRESHOLD_BREAKDOWN,
    _RE_EVAL_LEVEL_CHOICES,
    _ROLLOUT_DECISION_PLY,
    _ROLLOUT_N_TRIALS,
    _ROLLOUT_TRUNCATION_DEPTH,
    _away_counts_for_active,
    _below_threshold_stub,
    _board_for_analysis,
    _build_rollout_analyzer,
    _classify_case,
    _dispute_key,
    _dispute_summary_row,
    _ensure_boards_at_re_eval,
    _evaluate_single_board_at_re_eval,
    _find_disputes_in_game,
    _flip_owner,
    _get_or_compute_re_eval_checker,
    _get_or_compute_re_eval_cube,
    _get_re_eval_analyzer,
    _load_done_rollouts,
    _load_re_eval_cache,
    _normalize_bar,
    _owner_relative_to,
    _print_threshold_breakdown,
    _print_top_disputes,
    _re_eval_cache_filename,
    _re_eval_cache_key,
    _record_key,
    _rollout_checker,
    _rollout_cube_doubler,
    _rollout_cube_responder,
    _rollout_quality_equity,
    _run_rollout_for_dispute,
    _sage_played_board,
    _seed_sort_key,
    _shared_cube_result,
    _write_re_eval_cache_entry,
    _write_rollout_record,
    main as _unified_main,
)


def main(argv: list[str] | None = None) -> None:
    _unified_main(argv, default_mode="match")


if __name__ == "__main__":
    main()
