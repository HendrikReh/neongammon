# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Shared benchmark runner, refinement, scoring, and I/O helpers.

Used by ``benchmark_money.py``, ``benchmark_match.py``, ``benchmark_pasko.py``,
and ``backgame_benchmark.py`` so decision filtering, adaptive-precision passes,
rollout convergence, PR scoring, and atomic persistence stay single-sourced.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

# ---------------------------------------------------------------------------
# Self-contained neongammon repo path setup
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent          # = neongammon repo root
_NEONGAMMON_PYTHON = _PROJECT_ROOT / "python"
_BUILD_DIR = _PROJECT_ROOT / "build"


def ensure_repo_paths(*, include_scripts: bool = False) -> Path:
    """Put repo-local ``python/``, ``build/``, and optionally ``scripts/`` on ``sys.path``."""
    paths = [_NEONGAMMON_PYTHON, _BUILD_DIR]
    if include_scripts:
        paths.insert(0, _SCRIPT_DIR)
    for p in paths:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    return _PROJECT_ROOT


ensure_repo_paths(include_scripts=True)


# ---------------------------------------------------------------------------
# Shared benchmark parameters
# ---------------------------------------------------------------------------

TRIVIAL_SPREAD = 0.001
THREE_T_GAP = 0.05
ROLLOUT_GAP = 0.02
BLUNDER_THRESHOLD = 0.08
PR_MULTIPLIER = 500

ROLLOUT_N_TRIALS = 1296
ROLLOUT_TARGET_SE = 0.00255
ROLLOUT_MAX_BATCHES = 16
_ROLLOUT_BASE_SEED = 42
_SEED_STEP = 0x9E3779B9

GAME_PLANS = ["purerace", "racing", "attacking", "priming", "anchoring"]

TIER_3P = "3P"
TIER_3T = "3T"
TIER_ROLLOUT = "rollout"
_TIER_RANK = {TIER_3P: 0, TIER_3T: 1, TIER_ROLLOUT: 2}

_MAX_TURNS = 600
_PROGRESS_EVERY_S = 15.0
_SLOW_ITEM_S = 2.0


# ---------------------------------------------------------------------------
# Dataclasses & small helpers
# ---------------------------------------------------------------------------


@dataclass
class CheckerCandidate:
    """One candidate checker play returned by a bot (mover's perspective)."""

    board: list[int]
    equity: float


@dataclass
class CubeAssessment:
    """A bot's cube assessment of a position (player-on-roll perspective)."""

    should_double: bool
    should_take: bool


@dataclass
class _SimState:
    board: list[int]
    cube_value: int
    cube_owner: str   # "centered" | "player" | "opponent" (relative to active)
    active: int       # 1 or 2


def flip_owner(owner: str) -> str:
    if owner == "player":
        return "opponent"
    if owner == "opponent":
        return "player"
    return "centered"


def _flip_sim(state: _SimState) -> None:
    from neongammon import flip_board

    state.board = list(flip_board(state.board))
    state.cube_owner = flip_owner(state.cube_owner)
    state.active = 3 - state.active


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _fmt_dur(seconds: float) -> str:
    """Human-readable duration like ``2h05m``, ``7m30s``, ``42s``."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically via a temp sibling + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _read_dataset(dataset_path: Path | str) -> dict:
    """Load the assembled benchmark, transparently reading a gzip ``.gz`` sibling."""
    p = Path(dataset_path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    gz = Path(f"{p}.gz")
    if gz.exists():
        return json.loads(gzip.decompress(gz.read_bytes()).decode("utf-8"))
    raise FileNotFoundError(f"Benchmark dataset not found: {p} (or {gz})")


def _load_jsonl_by_key(path: Path) -> dict[str, dict]:
    """Load an incremental ``{key: record}`` JSONL, if present."""
    out: dict[str, dict] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out[rec["key"]] = rec
    return out


def _unique_by_key(decisions: list[dict]) -> dict[str, dict]:
    """Collapse repeated positions to one decision each (first occurrence wins)."""
    by_key: dict[str, dict] = {}
    for d in decisions:
        by_key.setdefault(d["key"], d)
    return by_key


def _seed_in_range(dec: dict, min_seed: int | None, max_seed: int | None) -> bool:
    s = dec.get("seed", 0)
    return (min_seed is None or s >= min_seed) and (max_seed is None or s <= max_seed)


def make_decision_key(
    kind: str,
    board: list[int],
    dice: Optional[tuple[int, int] | list[int]],
    cube_value: int,
    cube_owner: str,
    *,
    match_context: Optional[tuple[int, int, bool]] = None,
) -> str:
    """Stable hash identifying a decision position (optional match context folded in)."""
    dice_part = None if dice is None else sorted(int(d) for d in dice)
    fields: list[Any] = [kind, [int(x) for x in board], dice_part, int(cube_value), str(cube_owner)]
    if match_context is not None:
        away1, away2, is_crawford = match_context
        fields.extend([int(away1), int(away2), bool(is_crawford)])
    payload = json.dumps(fields, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _is_trivial_cube(nd: float, dt: float, dp: float) -> bool:
    """Whether a cube position is too obvious to count as a decision."""
    if abs(nd - dt) < TRIVIAL_SPREAD:
        return True
    if nd - dt > 0.200:
        return True
    if nd - dp > 0.200:
        return True
    if nd < -0.900 and dt < -0.900:
        return True
    return False


def cube_decision_flags(cube: Any, *, allow_beaver: bool = True) -> tuple[bool, bool]:
    """Return ``(has_double, has_take)`` eligibility flags for a cube evaluation."""
    nd, dt, dp = cube.equity_nd, cube.equity_dt, cube.equity_dp
    is_beaver = allow_beaver and bool(getattr(cube, "is_beaver", False))
    has_double = not _is_trivial_cube(nd, dt, dp)
    has_take = bool(cube.should_double) and (abs(dt - dp) >= TRIVIAL_SPREAD or is_beaver)
    return has_double, has_take


def is_checker_decision(equities: Sequence[float]) -> bool:
    """Whether a ranked list of candidate equities counts as a non-trivial checker decision."""
    return len(equities) >= 2 and (equities[0] - equities[-1]) >= TRIVIAL_SPREAD


def _cube_gap(nd: float, dt: float, dp: float) -> float:
    return abs(nd - min(dt, dp))


def _checker_gap(moves: list[dict]) -> float:
    if len(moves) < 2:
        return float("inf")
    return moves[0]["equity"] - moves[1]["equity"]


def _take_gap(dt: float, dp: float) -> float:
    return abs(dt - dp)


def _gap_for(entry: dict) -> float:
    if entry["kind"] == "checker":
        return _checker_gap(entry["moves"])
    nd, dt, dp = entry["equity_nd"], entry["equity_dt"], entry["equity_dp"]
    gaps = []
    if entry.get("has_double"):
        gaps.append(_cube_gap(nd, dt, dp))
    if entry.get("has_take"):
        gaps.append(_take_gap(dt, dp))
    return min(gaps) if gaps else float("inf")


def _missing_tier(entry: dict) -> Optional[str]:
    gap = _gap_for(entry)
    tier = entry.get("tier")
    if gap < ROLLOUT_GAP and tier != TIER_ROLLOUT:
        return "rollout"
    if gap < THREE_T_GAP and tier == TIER_3P:
        return "3t"
    return None


def _move_list_from_result(result: Any) -> list[dict]:
    return [
        {
            "board": list(m.board),
            "equity": m.equity,
            "cubeless_equity": m.cubeless_equity,
            "probs": list(m.probs.to_list()),
            "eval_level": m.eval_level,
        }
        for m in result.moves
    ]


def _cube_fields_from_result(cube: Any) -> dict:
    return {
        "equity_nd": cube.equity_nd,
        "equity_dt": cube.equity_dt,
        "equity_dp": cube.equity_dp,
        "probs": list(cube.probs.to_list()),
        "should_double_ref": cube.should_double,
        "should_take_ref": cube.should_take,
        "is_beaver": getattr(cube, "is_beaver", False),
        "optimal_action": cube.optimal_action,
        "eval_level": cube.eval_level,
    }


def _apply_refined(entry: dict, refined: dict, tier: str) -> dict:
    merged = dict(entry)
    if entry["kind"] == "checker":
        merged["moves"] = refined["moves"]
    else:
        for k in ("equity_nd", "equity_dt", "equity_dp", "probs",
                  "should_double_ref", "should_take_ref", "is_beaver",
                  "optimal_action", "eval_level"):
            if k in refined:
                merged[k] = refined[k]
    merged["tier"] = tier
    return merged


# ---------------------------------------------------------------------------
# Analyzer factories
# ---------------------------------------------------------------------------

_analyzer_cache: dict[str, Any] = {}


def _get_analyzer(eval_level: str, parallel_threads: int):
    """Cache a BgBotAnalyzer per (level, threads) within a process."""
    key = f"{eval_level}:{parallel_threads}"
    cached = _analyzer_cache.get(key)
    if cached is not None:
        return cached
    from neongammon import BgBotAnalyzer

    a = BgBotAnalyzer(eval_level=eval_level, cubeful=True, parallel_threads=parallel_threads)
    _analyzer_cache[key] = a
    return a


def _make_3t_analyzer(n_threads: int):
    from neongammon import BgBotAnalyzer

    return BgBotAnalyzer(eval_level="truncated3", cubeful=True, parallel_threads=n_threads)


def _make_rollout_analyzer(n_threads: int):
    from neongammon import BgBotAnalyzer, TrialEvalConfig

    return BgBotAnalyzer(
        eval_level="rollout",
        cubeful=True,
        parallel_threads=n_threads,
        n_trials=ROLLOUT_N_TRIALS,
        truncation_depth=0,
        decision_ply=3,
        late_ply=3,
        late_threshold=20,
        ultra_late_threshold=9999,
        checker=TrialEvalConfig(ply=3),
        cube=TrialEvalConfig(ply=3),
        seed=_ROLLOUT_BASE_SEED,
        target_se=ROLLOUT_TARGET_SE,
        max_batches=ROLLOUT_MAX_BATCHES,
    )


def _away_counts_for_mover(
    active: int, score1: int, score2: int, match_length: int,
) -> tuple[int, int]:
    if active == 1:
        mover_score, opp_score = score1, score2
    else:
        mover_score, opp_score = score2, score1
    return (max(1, match_length - mover_score), max(1, match_length - opp_score))


# ---------------------------------------------------------------------------
# Pass 1: game simulation & capture + process-pool runner
# ---------------------------------------------------------------------------


def simulate_game_and_capture(
    analyzer: Any,
    rng: random.Random,
    *,
    starting_board: Sequence[int],
    seed: int,
    match_length: int = 0,
    score1: int = 0,
    score2: int = 0,
    is_crawford: bool = False,
    game_number: Optional[int] = None,
) -> tuple[list[dict], list[dict], Optional[int], Optional[str], int]:
    """Play one 3P vs 3P game (money, Pasko, or match) and capture every decision."""
    from neongammon import check_game_over, classify_game_plan, possible_moves
    from neongammon.text_export import compute_move_notation

    is_match = match_length > 0
    state = _SimState(board=list(starting_board), cube_value=1, cube_owner="centered", active=1)
    decisions: list[dict] = []
    move_history: list[dict] = []

    winner: Optional[int] = None
    win_type: Optional[str] = None
    cube_at_end = 1

    for turn in range(_MAX_TURNS):
        away_mover = away_opp = 0
        if is_match:
            away_mover, away_opp = _away_counts_for_mover(state.active, score1, score2, match_length)
        cube_action_str: Optional[str] = None

        # --- Cube decision ---
        can_offer = (not is_crawford) if is_match else True
        if can_offer and state.cube_owner in ("centered", "player"):
            if is_match:
                cube = analyzer.cube_action(
                    state.board, cube_value=state.cube_value, cube_owner=state.cube_owner,
                    away1=away_mover, away2=away_opp, is_crawford=is_crawford,
                )
            else:
                cube = analyzer.cube_action(
                    state.board, cube_value=state.cube_value, cube_owner=state.cube_owner,
                    jacoby=True, beaver=True,
                )
            has_double, has_take = cube_decision_flags(cube, allow_beaver=not is_match)
            if has_double or has_take:
                match_ctx = (away_mover, away_opp, is_crawford) if is_match else None
                entry: dict[str, Any] = {
                    "kind": "cube",
                    "has_double": has_double,
                    "has_take": has_take,
                    "board": list(state.board),
                    "dice": None,
                    "cube_value": state.cube_value,
                    "cube_owner": state.cube_owner,
                }
                if is_match:
                    entry.update({"away1": away_mover, "away2": away_opp, "is_crawford": is_crawford})
                entry.update({
                    "game_plan": classify_game_plan(state.board),
                    "tier": TIER_3P,
                    "key": make_decision_key(
                        "cube", state.board, None, state.cube_value, state.cube_owner,
                        match_context=match_ctx,
                    ),
                    "seed": seed,
                })
                if game_number is not None:
                    entry["game_number"] = game_number
                entry.update({
                    "turn": turn,
                    "player": state.active,
                    **_cube_fields_from_result(cube),
                })
                decisions.append(entry)

            if cube.should_double:
                if cube.should_take:
                    cube_action_str = "double/take"
                    state.cube_value *= 2
                    state.cube_owner = "opponent"
                else:
                    cube_action_str = "double/pass"
                    move_history.append({
                        "player": "user" if state.active == 1 else "bot",
                        "cube_action": cube_action_str, "dice": None, "move": None,
                    })
                    winner, win_type, cube_at_end = state.active, "single", state.cube_value
                    break

        # --- Dice roll + checker play ---
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        cands = possible_moves(state.board, d1, d2)
        if not cands:
            post = list(state.board)
        else:
            if is_match:
                result = analyzer.checker_play(
                    state.board, d1, d2, cube_value=state.cube_value, cube_owner=state.cube_owner,
                    away1=away_mover, away2=away_opp, is_crawford=is_crawford,
                )
            else:
                result = analyzer.checker_play(
                    state.board, d1, d2, cube_value=state.cube_value, cube_owner=state.cube_owner,
                )
            post = list(result.moves[0].board)
            moves = _move_list_from_result(result)
            if len(moves) >= 2 and (moves[0]["equity"] - moves[-1]["equity"]) >= TRIVIAL_SPREAD:
                match_ctx = (away_mover, away_opp, is_crawford) if is_match else None
                entry = {
                    "kind": "checker",
                    "board": list(state.board),
                    "dice": [d1, d2],
                    "cube_value": state.cube_value,
                    "cube_owner": state.cube_owner,
                }
                if is_match:
                    entry.update({"away1": away_mover, "away2": away_opp, "is_crawford": is_crawford})
                entry.update({
                    "game_plan": classify_game_plan(state.board),
                    "tier": TIER_3P,
                    "moves": moves,
                    "key": make_decision_key(
                        "checker", state.board, (d1, d2), state.cube_value, state.cube_owner,
                        match_context=match_ctx,
                    ),
                    "seed": seed,
                })
                if game_number is not None:
                    entry["game_number"] = game_number
                entry.update({"turn": turn, "player": state.active})
                decisions.append(entry)

        move_history.append({
            "player": "user" if state.active == 1 else "bot",
            "cube_action": cube_action_str,
            "dice": [d1, d2],
            "move": compute_move_notation(state.board, post, d1, d2),
        })

        state.board = post
        code = check_game_over(state.board)
        if code > 0:
            winner = state.active
            win_type = {1: "single", 2: "gammon", 3: "backgammon"}.get(code, "single")
            cube_at_end = state.cube_value
            break
        _flip_sim(state)
    else:
        g_tag = f" g{game_number}" if game_number is not None else ""
        raise RuntimeError(f"game seed={seed}{g_tag} exceeded {_MAX_TURNS} turns")

    return decisions, move_history, winner, win_type, cube_at_end


def build_unlimited_xg_record(
    move_history: list[dict], winner: Optional[int], win_type: Optional[str], cube_at_end: int,
) -> dict:
    if winner is None:
        result_str, points = "", 0
    else:
        side = "player1" if winner == 1 else "player2"
        mult = {"single": 1, "gammon": 2, "backgammon": 3}[win_type or "single"]
        result_str, points = f"{side}-win-{win_type}", cube_at_end * mult
    return {
        "player1_name": "Neongammon",
        "player2_name": "Neongammon",
        "mode": "unlimited",
        "result": result_str,
        "result_points": points,
        "move_history": move_history,
    }


def run_pass1_pool(
    seeds: list[int],
    *,
    is_done_fn: Callable[[int], bool],
    worker_fn: Callable[..., dict],
    worker_args_fn: Callable[[int, int], tuple],
    label: str,
    unit_name: str,
    workers: int,
    format_result: Callable[[dict], str],
) -> None:
    """Run Pass 1 simulation across ``seeds`` with optional ProcessPoolExecutor."""
    todo = [s for s in seeds if not is_done_fn(s)]
    done = len(seeds) - len(todo)
    _log(f"{label}: {len(seeds)} {unit_name}, {done} already done, "
         f"{len(todo)} to play (workers={workers}).")
    if not todo:
        return

    total = len(todo)
    start = time.perf_counter()
    completed = 0

    def _report(res: dict) -> None:
        nonlocal completed
        completed += 1
        rate = completed / max(1e-9, time.perf_counter() - start)
        eta = (total - completed) / rate if rate > 0 else 0
        _log(f"  [{completed}/{total}] {format_result(res)}  "
             f"({rate:.2f} {unit_name}/s, ETA {_fmt_dur(eta)})")

    if workers == 1:
        for s in todo:
            _report(worker_fn(*worker_args_fn(s, 0)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(worker_fn, *worker_args_fn(s, 1)): s for s in todo}
            for fut in as_completed(futures):
                _report(fut.result())
    _log(f"{label}: complete -- {total} {unit_name} in {_fmt_dur(time.perf_counter() - start)}.")


def load_stage1_decisions(stage1_dir: Path, pattern: str = "seed_*.json") -> list[dict]:
    out: list[dict] = []
    for path in sorted(stage1_dir.glob(pattern)):
        data = json.loads(path.read_text(encoding="utf-8"))
        out.extend(data["decisions"])
    return out


# ---------------------------------------------------------------------------
# Passes 2 & 3: adaptive refinement & distributed job export
# ---------------------------------------------------------------------------


def _match_kwargs(decision: dict, match_mode: bool) -> dict:
    if not match_mode:
        return {}
    return dict(
        away1=decision["away1"],
        away2=decision["away2"],
        is_crawford=decision["is_crawford"],
    )


def reeval_decision(
    analyzer: Any, decision: dict, progress_callback=None, *, match_mode: bool = False,
) -> dict:
    """Re-evaluate one decision with ``analyzer``; return refined analytics (+ key)."""
    mk = _match_kwargs(decision, match_mode)
    if decision["kind"] == "checker":
        d1, d2 = decision["dice"]
        result = analyzer.checker_play(
            decision["board"], d1, d2,
            cube_value=decision["cube_value"], cube_owner=decision["cube_owner"],
            progress_callback=progress_callback, **mk,
        )
        return {"key": decision["key"], "kind": "checker", "moves": _move_list_from_result(result)}
    cube = analyzer.cube_action(
        decision["board"], cube_value=decision["cube_value"], cube_owner=decision["cube_owner"],
        progress_callback=progress_callback, **mk,
    )
    return {"key": decision["key"], "kind": "cube", **_cube_fields_from_result(cube)}


def reeval_decision_rollout(
    analyzer: Any, decision: dict, progress_callback=None, *, match_mode: bool = False,
) -> dict:
    """Roll out one decision to ROLLOUT_TARGET_SE, sharing caches across batches."""
    mk = _match_kwargs(decision, match_mode)
    if decision["kind"] == "cube":
        analyzer.set_seed(_ROLLOUT_BASE_SEED)
        cube = analyzer.cube_action(
            decision["board"], cube_value=decision["cube_value"],
            cube_owner=decision["cube_owner"], progress_callback=progress_callback, **mk,
        )
        rec = {"key": decision["key"], "kind": "cube", **_cube_fields_from_result(cube)}
        rec["rollout_se"] = cube.equity_nd_se
        return rec

    d1, d2 = decision["dice"]
    acc: dict = {}
    n_batches = 0
    best_se = float("inf")
    agg: list[dict] = []
    for b in range(ROLLOUT_MAX_BATCHES):
        analyzer.set_seed((_ROLLOUT_BASE_SEED + b * _SEED_STEP) & 0xFFFFFFFF)
        res = analyzer.checker_play(
            decision["board"], d1, d2,
            cube_value=decision["cube_value"], cube_owner=decision["cube_owner"],
            progress_callback=progress_callback, **mk,
        )
        for m in res.moves:
            key = tuple(m.board)
            a = acc.get(key)
            if a is None:
                a = {"eq": 0.0, "cl": 0.0, "p": [0.0] * 5, "se2": 0.0, "n": 0, "lvl": m.eval_level}
                acc[key] = a
            a["eq"] += m.equity
            a["cl"] += m.cubeless_equity
            pl = m.probs.to_list()
            for k in range(5):
                a["p"][k] += pl[k]
            se = m.std_error or 0.0
            a["se2"] += se * se
            a["n"] += 1
        n_batches = b + 1
        agg = []
        for bkey, a in acc.items():
            n = a["n"]
            agg.append({
                "board": list(bkey),
                "equity": a["eq"] / n,
                "cubeless_equity": a["cl"] / n,
                "probs": [v / n for v in a["p"]],
                "eval_level": a["lvl"],
                "std_error": (a["se2"] ** 0.5) / n,
            })
        agg.sort(key=lambda x: -x["equity"])
        best_se = agg[0]["std_error"]
        if ROLLOUT_TARGET_SE <= 0 or best_se <= ROLLOUT_TARGET_SE:
            break
    return {
        "key": decision["key"], "kind": "checker", "moves": agg,
        "rollout_batches": n_batches, "rollout_se": best_se,
    }


def _run_refinement_pass(
    unique: dict[str, dict], analyzer: Any, out_file: Path, gap_threshold: float, tier: str,
    pass_label: str, reeval_fn=reeval_decision,
) -> dict[str, dict]:
    """Re-evaluate every decision whose current gap < ``gap_threshold`` at ``tier``."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    done = _load_jsonl_by_key(out_file)

    for key, refined in done.items():
        if key in unique:
            unique[key] = _apply_refined(unique[key], refined, tier)

    todo = [
        key for key, dec in unique.items()
        if key not in done and _gap_for(dec) < gap_threshold
    ]
    _log(f"{pass_label}: {len(done)} already refined, {len(todo)} to compute "
         f"(gap < {gap_threshold}).")
    if not todo:
        return unique

    total = len(todo)
    start = time.perf_counter()
    last_print = [start]
    cur_label = [""]

    def _heartbeat(completed, total_trials, *_):
        now = time.perf_counter()
        if total_trials and now - last_print[0] >= _PROGRESS_EVERY_S:
            last_print[0] = now
            pct = 100 * completed // total_trials
            print(f"      {cur_label[0]}: {completed}/{total_trials} trials ({pct}%)", flush=True)

    with out_file.open("a", encoding="utf-8") as f:
        for i, key in enumerate(todo, 1):
            dec = unique[key]
            cur_label[0] = f"[{i}/{total}] {dec['kind']} {key[:8]}"
            t0 = time.perf_counter()
            refined = reeval_fn(analyzer, dec, progress_callback=_heartbeat)
            item_dt = time.perf_counter() - t0
            f.write(json.dumps(refined, separators=(",", ":")) + "\n")
            f.flush()
            unique[key] = _apply_refined(dec, refined, tier)
            now = time.perf_counter()
            if item_dt >= _SLOW_ITEM_S or now - last_print[0] >= _PROGRESS_EVERY_S or i == total:
                last_print[0] = now
                rate = i / max(1e-9, now - start)
                eta = (total - i) / rate if rate > 0 else 0
                _log(f"  [{i}/{total}] {tier} {dec['kind']} {key[:8]} done in "
                     f"{item_dt:.1f}s  ({rate:.2f}/s, ETA {_fmt_dur(eta)})")
    _log(f"{pass_label}: complete -- {total} refined in {_fmt_dur(time.perf_counter() - start)}.")
    return unique


def export_threet_jobs(
    unique: dict[str, dict],
    stage2_file: Path,
    out_file: Path,
    *,
    match_mode: bool = False,
    min_seed: int | None = None,
    max_seed: int | None = None,
) -> Path:
    done = _load_jsonl_by_key(stage2_file)
    jobs: list[dict] = []
    for key, dec in unique.items():
        if key in done or _gap_for(dec) >= THREE_T_GAP or not _seed_in_range(dec, min_seed, max_seed):
            continue
        job: dict[str, Any] = {
            "key": key,
            "kind": dec["kind"],
            "board": dec["board"],
            "dice": dec.get("dice"),
            "cube_value": dec["cube_value"],
            "cube_owner": dec["cube_owner"],
        }
        if match_mode:
            job.update({"away1": dec["away1"], "away2": dec["away2"], "is_crawford": dec["is_crawford"]})
        job.update({"game_plan": dec.get("game_plan"), "level": "truncated3"})
        jobs.append(job)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, separators=(",", ":")) + "\n")
    tmp.replace(out_file)
    n_ck = sum(1 for j in jobs if j["kind"] == "checker")
    _log(f"Export: wrote {len(jobs)} 3T jobs ({n_ck} checker, {len(jobs) - n_ck} cube) "
         f"to {out_file} -- no 3T evals computed.")
    return out_file


def export_rollout_jobs(
    unique: dict[str, dict],
    stage3_file: Path,
    out_file: Path,
    *,
    match_mode: bool = False,
    min_seed: int | None = None,
    max_seed: int | None = None,
) -> Path:
    done = _load_jsonl_by_key(stage3_file)
    jobs: list[dict] = []
    for key, dec in unique.items():
        if key in done or _gap_for(dec) >= ROLLOUT_GAP or not _seed_in_range(dec, min_seed, max_seed):
            continue
        away1, away2, is_crawford = (
            (dec["away1"], dec["away2"], dec["is_crawford"]) if match_mode else (0, 0, False)
        )
        jobs.append({
            "key": key,
            "kind": dec["kind"],
            "board": dec["board"],
            "dice": dec.get("dice"),
            "cube_value": dec["cube_value"],
            "cube_owner": dec["cube_owner"],
            "away1": away1, "away2": away2, "is_crawford": is_crawford,
            "jacoby": not match_mode, "beaver": not match_mode,
            "game_plan": dec.get("game_plan"),
            "rollout": {
                "n_trials": ROLLOUT_N_TRIALS,
                "truncation_depth": 0,
                "decision_ply": 3,
                "target_se": ROLLOUT_TARGET_SE,
                "max_batches": ROLLOUT_MAX_BATCHES,
                "gate": "equity_nd_se" if dec["kind"] == "cube" else "best_move_equity_se",
            },
        })
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, separators=(",", ":")) + "\n")
    tmp.replace(out_file)
    n_ck = sum(1 for j in jobs if j["kind"] == "checker")
    _log(f"Export: wrote {len(jobs)} rollout jobs ({n_ck} checker, {len(jobs) - n_ck} cube) "
         f"to {out_file} -- no rollouts computed.")
    return out_file


def run_passes_2_and_3(
    unique: dict[str, dict],
    *,
    stage2_file: Path,
    stage3_file: Path,
    threet_jobs_file: Path,
    rollout_jobs_file: Path,
    stages: set[str],
    threet_mode: str,
    rollout_mode: str,
    n_threads: int,
    export_min_seed: int | None = None,
    export_max_seed: int | None = None,
    reeval_3t_fn=reeval_decision,
    reeval_rollout_fn=reeval_decision_rollout,
    match_mode: bool = False,
) -> dict[str, dict]:
    """Run (or ingest existing results for) Pass 2 (3T) and Pass 3 (rollout)."""
    if "pass2" in stages:
        if threet_mode == "export":
            for key, refined in _load_jsonl_by_key(stage2_file).items():
                if key in unique:
                    unique[key] = _apply_refined(unique[key], refined, TIER_3T)
            export_threet_jobs(
                unique, stage2_file, threet_jobs_file,
                match_mode=match_mode, min_seed=export_min_seed, max_seed=export_max_seed,
            )
        else:
            analyzer_3t = _make_3t_analyzer(n_threads)
            unique = _run_refinement_pass(
                unique, analyzer_3t, stage2_file, THREE_T_GAP, TIER_3T,
                "Pass 2/3 (3T refine)", reeval_fn=reeval_3t_fn,
            )
    else:
        done2 = _load_jsonl_by_key(stage2_file)
        for key, refined in done2.items():
            if key in unique:
                unique[key] = _apply_refined(unique[key], refined, TIER_3T)
        pending2 = sum(1 for key, dec in unique.items()
                       if key not in done2 and _gap_for(dec) < THREE_T_GAP)
        _log(f"Pass 2 (3T) not run: {pending2} positions want 3T "
             f"(3P gap < {THREE_T_GAP}); {len(done2)} already done. Run with --stages pass2.")

    if "pass3" in stages:
        if rollout_mode == "export":
            for key, refined in _load_jsonl_by_key(stage3_file).items():
                if key in unique:
                    unique[key] = _apply_refined(unique[key], refined, TIER_ROLLOUT)
            export_rollout_jobs(
                unique, stage3_file, rollout_jobs_file,
                match_mode=match_mode, min_seed=export_min_seed, max_seed=export_max_seed,
            )
        else:
            analyzer_ro = _make_rollout_analyzer(n_threads)
            unique = _run_refinement_pass(
                unique, analyzer_ro, stage3_file, ROLLOUT_GAP, TIER_ROLLOUT,
                "Pass 3/3 (full rollout to SE)", reeval_fn=reeval_rollout_fn,
            )
    else:
        done3 = _load_jsonl_by_key(stage3_file)
        for key, refined in done3.items():
            if key in unique:
                unique[key] = _apply_refined(unique[key], refined, TIER_ROLLOUT)
        pending3 = sum(1 for key, dec in unique.items()
                       if key not in done3 and _gap_for(dec) < ROLLOUT_GAP)
        _log(f"Pass 3 (rollout) not run: {pending3} positions want rollout "
             f"(current-tier gap < {ROLLOUT_GAP}); {len(done3)} already done. "
             f"Run with --stages pass3.")

    return unique


def assemble_dataset(
    unique: dict[str, dict],
    dataset_path: Path,
    *,
    meta_prefix: dict[str, Any],
    sort_key: Optional[Callable[[dict], Any]] = None,
) -> dict:
    """Write the final benchmark JSON (+ .gz sibling) and return its metadata dict."""
    if sort_key is None:
        sort_key = lambda d: (d.get("seed", 0), d.get("turn", 0), d["kind"])
    decisions = sorted(unique.values(), key=sort_key)

    tier_counts = {TIER_3P: 0, TIER_3T: 0, TIER_ROLLOUT: 0}
    n_checker = n_cube_entries = n_double = n_take = 0
    for d in decisions:
        tier_counts[d["tier"]] += 1
        if d["kind"] == "checker":
            n_checker += 1
        else:
            n_cube_entries += 1
            n_double += int(d.get("has_double", False))
            n_take += int(d.get("has_take", False))
    n_cube_decisions = n_double + n_take
    n_decisions = n_checker + n_cube_decisions

    meta = {
        **meta_prefix,
        "n_decisions": n_decisions,
        "n_positions": len(decisions),
        "n_checker": n_checker,
        "n_cube_decisions": n_cube_decisions,
        "n_cube_double": n_double,
        "n_cube_take": n_take,
        "n_cube_entries": n_cube_entries,
        "tier_counts": tier_counts,
        "three_t_gap": THREE_T_GAP,
        "rollout_gap": ROLLOUT_GAP,
        "rollout_n_trials": ROLLOUT_N_TRIALS,
        "trivial_spread": TRIVIAL_SPREAD,
        "blunder_threshold": BLUNDER_THRESHOLD,
        "pr_multiplier": PR_MULTIPLIER,
    }
    atomic_write_text(dataset_path, json.dumps({"meta": meta, "decisions": decisions}))
    Path(f"{dataset_path}.gz").write_bytes(gzip.compress(dataset_path.read_bytes(), compresslevel=9))
    _log(f"Wrote {len(decisions)} positions ({n_decisions} decisions) to {dataset_path} (+ .gz)")
    _log(f"  checker={n_checker}  cube: {n_cube_decisions} decisions "
         f"(double={n_double}, take={n_take}) over {n_cube_entries} positions  tiers={tier_counts}")
    return meta


# ---------------------------------------------------------------------------
# Scoring & aggregation
# ---------------------------------------------------------------------------


def _scored(bucket: str, sub: str, plan: Optional[str], err: float) -> dict:
    return {"bucket": bucket, "sub": sub, "game_plan": plan, "error": err,
            "is_blunder": err > BLUNDER_THRESHOLD}


def score_checker(bot: Any, entry: dict, *, match_mode: bool = False) -> Optional[dict]:
    """Scored checker decision (bot's chosen play vs the reference best). None on mismatch."""
    if match_mode:
        cands = bot.checker_play(
            entry["board"], entry["dice"][0], entry["dice"][1],
            entry["cube_value"], entry["cube_owner"],
            entry["away1"], entry["away2"], entry["is_crawford"],
        )
    else:
        cands = bot.checker_play(
            entry["board"], entry["dice"][0], entry["dice"][1],
            entry["cube_value"], entry["cube_owner"],
        )
    if not cands:
        return None
    chosen = tuple(cands[0].board)
    ref_by_board = {tuple(m["board"]): m["equity"] for m in entry["moves"]}
    chosen_eq = ref_by_board.get(chosen)
    if chosen_eq is None:
        return None
    best_eq = entry["moves"][0]["equity"]
    s = _scored("checker", "checker", entry.get("game_plan"), max(0.0, best_eq - chosen_eq))
    s["pick"] = list(chosen)
    return s


def score_cube(bot: Any, entry: dict, *, match_mode: bool = False) -> list[dict]:
    """Scored cube sub-decisions (doubler and/or receiver), standard PR formulas."""
    nd, dt, dp = entry["equity_nd"], entry["equity_dt"], entry["equity_dp"]
    plan = entry.get("game_plan")
    if match_mode:
        a = bot.cube_action(
            entry["board"], entry["cube_value"], entry["cube_owner"],
            entry["away1"], entry["away2"], entry["is_crawford"],
        )
    else:
        a = bot.cube_action(entry["board"], entry["cube_value"], entry["cube_owner"])
    out: list[dict] = []
    if entry.get("has_double"):
        optimal = max(nd, min(dt, dp))
        actual = min(dt, dp) if a.should_double else nd
        out.append(_scored("cube", "double", plan, max(0.0, optimal - actual)))
    if entry.get("has_take"):
        optimal = min(dt, dp)
        actual = dt if a.should_take else dp
        out.append(_scored("cube", "take", plan, max(0.0, actual - optimal)))
    for o in out:
        o["pick"] = {"should_double": bool(a.should_double), "should_take": bool(a.should_take)}
    return out


def score_entry(bot: Any, entry: dict, *, match_mode: bool = False) -> Optional[dict]:
    if entry["kind"] == "checker":
        s = score_checker(bot, entry, match_mode=match_mode)
        if s is None:
            return None
        scored = [s]
    else:
        scored = score_cube(bot, entry, match_mode=match_mode)
    pick = None
    for s in scored:
        pick = s.pop("pick", pick)
    return {"key": entry["key"], "scored": scored, "kind": entry["kind"], "pick": pick}


def _aggregate(entry_results: Iterable[dict]) -> dict:
    """Aggregate scored positions into the PR breakdown (each sub-decision counts once)."""
    sums = {"checker": 0.0, "cube": 0.0}
    counts = {"checker": 0, "cube": 0}
    blunders = {"checker": 0, "cube": 0}
    by_plan = {p: {"error": 0.0, "n": 0, "checker_n": 0, "cube_n": 0, "blunders": 0}
               for p in GAME_PLANS}

    for er in entry_results:
        for s in er.get("scored", []):
            bucket = s["bucket"]
            err = s["error"]
            sums[bucket] += err
            counts[bucket] += 1
            if s["is_blunder"]:
                blunders[bucket] += 1
            plan = s.get("game_plan")
            if plan in by_plan:
                b = by_plan[plan]
                b["error"] += err
                b["n"] += 1
                b[f"{bucket}_n"] += 1
                if s["is_blunder"]:
                    b["blunders"] += 1

    def pr(total_err: float, n: int) -> float:
        return (total_err / n * PR_MULTIPLIER) if n > 0 else 0.0

    n_checker, n_cube = counts["checker"], counts["cube"]
    n_total = n_checker + n_cube
    err_total = sums["checker"] + sums["cube"]

    plan_out = {}
    for p, b in by_plan.items():
        plan_out[p] = {
            "pr": pr(b["error"], b["n"]),
            "n": b["n"],
            "checker_n": b["checker_n"],
            "cube_n": b["cube_n"],
            "error": b["error"],
            "blunders": b["blunders"],
        }

    return {
        "total_pr": pr(err_total, n_total),
        "checker_pr": pr(sums["checker"], n_checker),
        "cube_pr": pr(sums["cube"], n_cube),
        "n_decisions": n_total,
        "n_checker": n_checker,
        "n_cube": n_cube,
        "total_error": err_total,
        "checker_error": sums["checker"],
        "cube_error": sums["cube"],
        "blunders": {
            "total": blunders["checker"] + blunders["cube"],
            "checker": blunders["checker"],
            "cube": blunders["cube"],
        },
        "by_game_plan": plan_out,
    }


def run_benchmark_pr(
    bot: Any,
    *,
    dataset_path: Path | str,
    scores_dir: Path,
    n_threads: int = 1,
    label: str = "bot",
    cache_path: Path | str | None = None,
    progress: bool = True,
    max_seed: int | None = None,
    match_mode: bool = False,
    unit_name: str = "games",
    score_entry_fn: Optional[Callable[[Any, dict], Optional[dict]]] = None,
) -> dict:
    """Score ``bot`` against a benchmark dataset with resumable JSONL caching."""
    dataset_path = Path(dataset_path)
    data = _read_dataset(dataset_path)
    all_entries = data["decisions"]
    if max_seed is not None:
        n_all = len(all_entries)
        all_entries = [e for e in all_entries if e.get("seed", 0) <= max_seed]
        _log(f"Limiting to the first {max_seed} {unit_name} (seed <= {max_seed}): "
             f"{len(all_entries)} of {n_all} decisions.")

    entries = []
    skipped_missing = {"rollout": 0, "3t": 0}
    for e in all_entries:
        miss = _missing_tier(e)
        if miss is None:
            entries.append(e)
        else:
            skipped_missing[miss] += 1
    n_skipped = skipped_missing["rollout"] + skipped_missing["3t"]

    if cache_path is None:
        scores_dir.mkdir(parents=True, exist_ok=True)
        cache_path = scores_dir / f"{label}.jsonl"
    else:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached = _load_jsonl_by_key(cache_path)
    todo = [e for e in entries if e["key"] not in cached]
    if progress:
        _log(f"Scoring {label}: {len(entries)} positions, {len(cached)} cached, "
             f"{len(todo)} to score (n_threads={n_threads}).")

    scorer = score_entry_fn or (lambda b, e: score_entry(b, e, match_mode=match_mode))
    lock = threading.Lock()
    mismatches = 0
    f = cache_path.open("a", encoding="utf-8")
    picks_path = cache_path.with_name(cache_path.stem + ".picks.jsonl")
    pf = picks_path.open("a", encoding="utf-8")
    try:
        done = 0
        total = len(todo)
        start = time.perf_counter()
        last_print = [start]

        def _work(entry: dict):
            nonlocal mismatches, done
            result = scorer(bot, entry)
            with lock:
                if result is None:
                    mismatches += 1
                else:
                    pf.write(json.dumps(
                        {"key": result["key"], "kind": result["kind"], "pick": result["pick"]},
                        separators=(",", ":"),
                    ) + "\n")
                    pf.flush()
                    row = {"key": result["key"], "scored": result["scored"]}
                    cached[row["key"]] = row
                    f.write(json.dumps(row, separators=(",", ":")) + "\n")
                    f.flush()
                done += 1
                now = time.perf_counter()
                if progress and (done == total or now - last_print[0] >= _PROGRESS_EVERY_S):
                    last_print[0] = now
                    rate = done / max(1e-9, now - start)
                    eta = (total - done) / rate if rate > 0 else 0
                    _log(f"  [{done}/{total}] scored ({rate:.1f}/s, ETA {_fmt_dur(eta)})")

        if total:
            if n_threads <= 1:
                for e in todo:
                    _work(e)
            else:
                with ThreadPoolExecutor(max_workers=n_threads) as ex:
                    list(ex.map(_work, todo))
    finally:
        f.close()
        pf.close()

    if mismatches:
        _log(f"WARNING: {mismatches} decisions skipped - the bot's chosen checker play "
             f"was not a legal reference move (not counted toward PR).")
    if n_skipped:
        _log(f"NOTE: skipped {n_skipped} positions with a missing higher-precision "
             f"reference ({skipped_missing['rollout']} missing rollout, "
             f"{skipped_missing['3t']} missing 3T) -- not scored.")

    scoreable_keys = {e["key"] for e in entries}
    result = _aggregate(r for k, r in cached.items() if k in scoreable_keys)
    result["mismatches"] = mismatches
    result["skipped"] = n_skipped
    result["skipped_missing_rollout"] = skipped_missing["rollout"]
    result["skipped_missing_3t"] = skipped_missing["3t"]
    if progress:
        _log(f"{label}: total PR={result['total_pr']:.2f} "
             f"(checker={result['checker_pr']:.2f}, cube={result['cube_pr']:.2f}) "
             f"over {result['n_decisions']} decisions")
    return result


def _print_report(result: dict) -> None:
    print()
    print(f"{'='*60}")
    print(f"Total PR:   {result['total_pr']:7.2f}   ({result['n_decisions']} decisions)")
    print(f"Checker PR: {result['checker_pr']:7.2f}   ({result['n_checker']} decisions)")
    print(f"Cube PR:    {result['cube_pr']:7.2f}   ({result['n_cube']} decisions)")
    print(f"Blunders:   {result['blunders']['total']} "
          f"(checker {result['blunders']['checker']}, cube {result['blunders']['cube']})")
    if result.get("skipped"):
        print(f"Skipped:    {result['skipped']} positions (missing reference: "
              f"{result.get('skipped_missing_rollout', 0)} rollout, "
              f"{result.get('skipped_missing_3t', 0)} 3T)")
    if result.get("mismatches"):
        print(f"Mismatches: {result['mismatches']} (bot move not a legal reference move)")
    print(f"{'-'*60}")
    print(f"{'Game plan':>10}  {'PR':>7}  {'N':>6}  {'chk':>6}  {'cube':>5}  {'blnd':>5}")
    for plan in GAME_PLANS:
        b = result["by_game_plan"][plan]
        print(f"{plan:>10}  {b['pr']:7.2f}  {b['n']:6d}  {b['checker_n']:6d}  "
              f"{b['cube_n']:5d}  {b['blunders']:5d}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Shared CLI parser helpers
# ---------------------------------------------------------------------------


def add_common_build_args(
    p_build: argparse.ArgumentParser,
    *,
    unit_name: str = "game",
    default_dataset: Optional[Path] = None,
) -> None:
    p_build.add_argument("--seed", type=int, default=1, help=f"First {unit_name} seed")
    p_build.add_argument("--n-threads", type=int, default=0,
                         help="Threads for the 3T and rollout passes (0=auto)")
    p_build.add_argument("--workers", type=int, default=6,
                         help="Parallel processes for the pass-1 self-play")
    p_build.add_argument("--write-txt", action=argparse.BooleanOptionalAction, default=True,
                         help=f"Write an XG-import .txt per {unit_name} to xg/ (default: on)")
    p_build.add_argument("--rollout-mode", choices=["compute", "export"], default="compute",
                         help="compute: run Pass-3 rollouts in-process; "
                              "export: write rollout_jobs.jsonl instead (no rollouts)")
    p_build.add_argument("--threet-mode", choices=["compute", "export"], default="compute",
                         help="compute: run the Pass-2 3T re-evals in-process; "
                              "export: write threet_jobs.jsonl instead (no 3T evals)")
    p_build.add_argument("--stages", default="pass1,pass2,pass3",
                         help="comma-separated passes to run this invocation "
                              "(pass1=self-play+3P, pass2=3T, pass3=rollout)")
    p_build.add_argument("--min-seed", type=int, default=None,
                         help="In export modes, only emit jobs for positions with seed >= this")
    p_build.add_argument("--max-seed", type=int, default=None,
                         help="In export modes, only emit jobs for positions with seed <= this")
    if default_dataset is not None:
        p_build.add_argument("--dataset", type=Path, default=default_dataset,
                             help="Output dataset path")


def add_common_score_args(
    p_score: argparse.ArgumentParser,
    *,
    unit_name: str = "games",
    default_dataset: Optional[Path] = None,
) -> None:
    p_score.add_argument("--level", default="3ply",
                         help="NeongammonBot eval level (1ply..4ply, truncated1/2/3, rollout)")
    p_score.add_argument("--model", default=None, help="Model name (default: production)")
    p_score.add_argument("--n-threads", type=int, default=1,
                         help="Threads scoring decisions concurrently")
    if default_dataset is not None:
        p_score.add_argument("--dataset", type=Path, default=default_dataset)
    p_score.add_argument("--label", default=None, help="Resume-cache label")
    p_score.add_argument("--max-seed", type=int, default=None,
                         help=f"Only score the first N {unit_name} (decisions with seed <= MAX_SEED)")
