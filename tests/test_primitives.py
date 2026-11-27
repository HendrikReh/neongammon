# SPDX-License-Identifier: MPL-2.0
"""Unit tests for shared pure-Python primitives, lazy imports, rollout cancel, and DifficultyPolicy."""

from __future__ import annotations

import ast
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest

from neongammon import _primitives as prim
from neongammon import cube_points, gnubg, validate_board
from neongammon.types import CubeDecisionResult, Probabilities

START = [
    0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5,
    5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0,
]


# ---------------------------------------------------------------------------
# validate_board
# ---------------------------------------------------------------------------


def test_validate_board_accepts_valid_inputs_and_returns_int_tuple():
    assert validate_board(START) == tuple(START)
    assert validate_board(tuple(START)) == tuple(START)
    assert validate_board(x for x in START) == tuple(START)

    # Terminal boards allowed when nonterminal=False
    empty = [0] * 26
    assert validate_board(empty) == tuple(empty)
    one_side_off = [0] * 26
    one_side_off[24] = -2
    assert validate_board(one_side_off) == tuple(one_side_off)

    # Nonterminal boards pass when both sides have >= 1 checker
    minimal = [0] * 26
    minimal[1], minimal[24] = 1, -1
    assert validate_board(minimal, nonterminal=True) == tuple(minimal)


def test_validate_board_accepts_numpy_integers_if_installed():
    np = pytest.importorskip("numpy")
    arr = np.asarray(START, dtype=np.int64)
    res = validate_board(arr, nonterminal=True)
    assert res == tuple(START)
    assert all(type(v) is int for v in res)


@pytest.mark.parametrize(
    "bad_board,match",
    [
        (START[:-1], "26 integers"),
        (START + [0], "26 integers"),
        ([False] + START[1:], "26 integers"),
        ([True] + START[1:], "26 integers"),
        ([0.0] + START[1:], "26 integers"),
        (["0"] + START[1:], "26 integers"),
        (42, "26 integers"),
        ([-1] + START[1:], "checker counts"),
        (START[:-1] + [-1], "checker counts"),
        (START[:-1] + [1], "checker counts"),   # 16 mover checkers
        ([1] + START[1:], "checker counts"),    # 16 opponent checkers
        ([16] + [0] * 25, "checker counts"),
        ([0, -16] + [0] * 24, "checker counts"),
    ],
)
def test_validate_board_rejects_invalid_boards(bad_board, match):
    with pytest.raises(ValueError, match=match):
        validate_board(bad_board)


@pytest.mark.parametrize(
    "terminal_board",
    [
        [0] * 26,
        [0, 1] + [0] * 24,   # opponent borne off
        [0, -1] + [0] * 24,  # mover borne off
    ],
)
def test_validate_board_nonterminal_rejects_terminal_boards(terminal_board):
    with pytest.raises(ValueError, match="nonterminal"):
        validate_board(terminal_board, nonterminal=True)


# ---------------------------------------------------------------------------
# flip_board, invert_probs, cubeless_equity
# ---------------------------------------------------------------------------


def test_flip_board_swaps_bars_and_reverses_negates_points():
    board = list(START)
    board[0] = 2
    board[25] = 1
    flipped = prim.flip_board(board)
    assert flipped[0] == 1 and flipped[25] == 2
    assert flipped[1] == -board[24] and flipped[24] == -board[1]
    assert prim.flip_board(flipped) == board


def test_invert_probs_and_cubeless_equity_invariants():
    probs = [0.62, 0.21, 0.03, 0.11, 0.01]
    inv = prim.invert_probs(probs)
    assert inv == pytest.approx([0.38, 0.11, 0.01, 0.21, 0.03])
    assert prim.invert_probs(inv) == pytest.approx(probs)

    eq = prim.cubeless_equity(probs)
    expected = 2.0 * 0.62 - 1.0 + 0.21 - 0.11 + 0.03 - 0.01
    assert eq == pytest.approx(expected)
    assert Probabilities.from_list(probs).equity == pytest.approx(eq)
    assert prim.cubeless_equity(inv) == pytest.approx(-eq)


# ---------------------------------------------------------------------------
# Cube owner normalization across pure-Python & native modules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected,flipped",
    [
        ("centered", "centered", "centered"),
        ("CeNtErEd", "centered", "centered"),
        ("player", "player", "opponent"),
        ("PLAYER", "player", "opponent"),
        ("opponent", "opponent", "player"),
        ("Opponent", "opponent", "player"),
    ],
)
def test_primitives_resolve_and_flip_owner(raw, expected, flipped):
    assert prim.resolve_owner(raw) == expected
    assert prim.flip_owner(raw) == flipped
    assert cube_points._flip_owner_str(raw) == flipped
    assert gnubg._flip_cube_owner(raw) == flipped


@pytest.mark.parametrize("bad", ["white", "center", "", None, 0, []])
def test_all_pure_python_cube_owner_helpers_raise_value_error(bad):
    for fn in (
        prim.resolve_owner,
        prim.flip_owner,
        cube_points._flip_owner_str,
        gnubg._flip_cube_owner,
        gnubg.GnuBgAnalyzer._cube_owner_cmd,
    ):
        with pytest.raises(ValueError, match="Invalid cube owner.*centered, player, or opponent"):
            fn(bad)

    with pytest.raises(ValueError, match="Invalid cube owner"):
        cube_points.cube_action_points((0.5, 0.0, 0.0, 0.0, 0.0), bad)


def test_cube_points_money_accepts_mixed_case_owner():
    res = cube_points.cube_action_points((0.5, 0.0, 0.0, 0.0, 0.0), "PLAYER")
    assert res.cannot_take is True
    assert res.double_label == "Redouble"


def test_normalize_cube_context_validates_limit_and_disables_money_rules_in_match():
    assert prim._normalize_cube_context(2, 0, 0, True, True, 4) == (True, True)
    assert prim._normalize_cube_context(2, 0, 0, False, True, 0) == (False, True)
    assert prim._normalize_cube_context(2, 5, 3, True, True, 4) == (False, False)
    assert prim._normalize_cube_context(1, 0, 1, True, True, 0) == (False, False)

    for bad_max in (-2, 1, 3, True, "4"):
        with pytest.raises(ValueError, match="max_cube_value"):
            prim._normalize_cube_context(1, 0, 0, True, True, bad_max)
    with pytest.raises(ValueError, match="exceeds max_cube_value"):
        prim._normalize_cube_context(4, 0, 0, True, True, 2)


# ---------------------------------------------------------------------------
# PEP 562 lazy imports & dead-code removal
# ---------------------------------------------------------------------------


def test_pure_python_submodules_import_without_bgbot_cpp():
    package = find_spec("neongammon")
    assert package is not None and package.origin is not None
    import_root = Path(package.origin).resolve().parent.parent
    script = """
import sys

class BlockNative:
    attempted = False
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "bgbot_cpp":
            self.attempted = True
            raise ModuleNotFoundError("blocked")

sys.path.insert(0, sys.argv[1])
blocker = BlockNative()
sys.meta_path.insert(0, blocker)

import neongammon
import neongammon._primitives
import neongammon.xg_file
import neongammon.xg_compare
import neongammon.text_export
import neongammon.settings
import neongammon.gnubg
import neongammon.data
import neongammon.cube_points

assert neongammon.validate_board([0] * 26) == (0,) * 26
assert not blocker.attempted and "bgbot_cpp" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(import_root)],
        capture_output=True, text=True, timeout=20,
    )
    assert completed.returncode == 0, completed.stderr


def test_cubeful_dead_legacy_symbols_removed():
    cubeful_path = Path(__file__).resolve().parent.parent / "python" / "neongammon" / "cubeful.py"
    tree = ast.parse(cubeful_path.read_text(encoding="utf-8"))
    func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "_pool_map" not in func_names
    assert "_convert_move_legacy" not in func_names


# ---------------------------------------------------------------------------
# Fake-extension unit verification (runs even without a rebuilt bgbot_cpp .so)
# ---------------------------------------------------------------------------


def test_native_consumers_with_stub_extension_verify_cancel_janowski_and_difficulty():
    """Exercise board, strategies, rollout, cubeful, and difficulty against a stub bgbot_cpp."""
    package = find_spec("neongammon")
    assert package is not None and package.origin is not None
    import_root = Path(package.origin).resolve().parent.parent
    script = """
import enum, sys, types

class CubeOwner(enum.IntEnum):
    CENTERED = 0
    PLAYER = 1
    OPPONENT = 2

class NativeRolloutCancelled(Exception):
    pass

stub = types.ModuleType("bgbot_cpp")
stub.CubeOwner = CubeOwner
stub.RolloutCancelled = NativeRolloutCancelled
stub.MONEY_CONTACT_CUBE_LIFE = {1: 0.72, 2: 0.75}
stub.MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE = {1: 0.001, 2: 0.001}
stub.CUBEFUL_FILTERED_PLIES = [2]
stub.can_double_match = lambda a1, a2, cv, owner, crawford, cap=0: (
    not crawford and owner != CubeOwner.OPPONENT and (cap == 0 or cv < cap)
)
stub.is_race = lambda b: True
stub.pip_counts = lambda b: (100, 100)
stub.cube_efficiency = lambda probs, race, pp, op: 0.68
stub.cl2cf_money = lambda probs, owner, x, jacoby_active=False: 0.42 + (0.01 if jacoby_active else 0.0)
stub.cl2cf = lambda probs, cv, owner, x, a1, a2, crawford, jacoby=False: 0.33
stub.cubeless_mwc = lambda probs, a1, a2, cv, crawford: 0.55
stub.mwc2eq = lambda mwc, a1, a2, cv, crawford: 0.11
stub.flip_board = lambda b: [b[25], *(-b[i] for i in range(24, 0, -1)), b[0]]
stub.possible_moves = lambda b, d1, d2: [list(b), [0, -2, 0, 0, 0, 1, 4, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0]]

sys.modules["bgbot_cpp"] = stub
sys.path.insert(0, sys.argv[1])

from types import SimpleNamespace
from neongammon import board, cubeful, difficulty, rollout, strategies
from neongammon.types import CubeDecisionResult

# 1. board.validate_board + board.can_double_match ValueError (not KeyError)
assert board.validate_board is board.validate_board
assert board.can_double_match(5, 5, 1, "PLAYER") is True
assert board.can_double_match(5, 5, 1, CubeOwner.OPPONENT) is False
try:
    board.can_double_match(5, 5, 1, "bogus")
    raise AssertionError("expected ValueError")
except ValueError as exc:
    assert "Invalid cube owner" in str(exc)

# 2. _janowski_1ply_equity: money, match, and dead-cube cap
probs = [0.6, 0.2, 0.0, 0.1, 0.0]
assert abs(strategies._janowski_1ply_equity(probs, board.STARTING_BOARD, CubeOwner.CENTERED, jacoby=True) - 0.43) < 1e-9
assert abs(strategies._janowski_1ply_equity(probs, board.STARTING_BOARD, CubeOwner.PLAYER, jacoby=True) - 0.42) < 1e-9
assert abs(strategies._janowski_1ply_equity(probs, board.STARTING_BOARD, CubeOwner.CENTERED, away1=5, away2=3) - 0.33) < 1e-9
# Dead-cube cap in money -> cubeless equity = 2*0.6 - 1 + 0.2 - 0.1 = 0.3
assert abs(strategies._janowski_1ply_equity(probs, board.STARTING_BOARD, CubeOwner.PLAYER, cube_value=2, max_cube_value=2) - 0.3) < 1e-9
# Dead-cube cap in match -> mwc2eq = 0.11
assert abs(strategies._janowski_1ply_equity(probs, board.STARTING_BOARD, CubeOwner.PLAYER, cube_value=2, away1=5, away2=3, max_cube_value=2) - 0.11) < 1e-9

# 3. _rollout_eval_candidate enforces _check_cancel() before native call
#    AND cubeful rollout promotion path hits _check_cancel()!
called = False
class FakeRo(rollout._RolloutAnalyzer):
    def __init__(self):
        self._weights = None
        self._rollout_config = {"decision_ply": 1, "n_trials": 36}
        self._bearoff_db = None
        self._cancel = False
        class Strat:
            def cubeful_evaluate_board(self, *a, **kw):
                global called
                called = True
                return {}
        self._rollout_strategy = Strat()
    def _check_cancel(self):
        if self._cancel:
            raise rollout.RolloutCancelled("cancelled")
    def checker_play_analytics(self, *a, **kw):
        # Return one non-rollout candidate ranking above a rollout candidate to force promotion,
        # and cancel right before promotion runs.
        self._cancel = True
        return [
            {"board": list(board.STARTING_BOARD), "equity": 0.9, "probs": probs, "eval_level": "1-ply"},
            {"board": list(board.STARTING_BOARD), "equity": 0.1, "probs": probs, "eval_level": "Rollout", "rollout_cubeful_equity": 0.1, "rollout_cubeful_se": 0.01},
        ]

ro = FakeRo()
cf = cubeful._CubefulAnalyzer(ro)
try:
    cf.checker_play_analytics(board.STARTING_BOARD, 3, 1)
    raise AssertionError("expected RolloutCancelled on promotion path")
except rollout.RolloutCancelled:
    pass
assert not called, "native cubeful_evaluate_board must not run after cancel"

# 4. DifficultyPolicy cached checker_analyzer / cube_analyzer + backward-compat + reseed
checker_calls = []
cube_calls = []
class StubAnalyzer:
    def __init__(self, level):
        self._eval_level = level
    def checker_play(self, b, d1, d2, **ctx):
        checker_calls.append((self._eval_level, d1, d2, ctx))
        return SimpleNamespace(moves=[
            SimpleNamespace(board=list(b), equity=0.25),
            SimpleNamespace(board=[0]*26, equity=0.24),
        ])
    def cube_decision(self, b, cube_value=1, cube_owner="centered", **ctx):
        cube_calls.append((self._eval_level, cube_value, cube_owner, ctx))
        return CubeDecisionResult(
            equity_nd=0.1, equity_dt=0.2, equity_dp=1.0,
            should_double=True, should_take=True, optimal_equity=0.2,
            optimal_action="Double/Take", eval_level=f"{self._eval_level[0]}-ply",
        )

a3 = StubAnalyzer("3ply")
a2 = StubAnalyzer("2ply")
pol = difficulty.DifficultyPolicy("full_strength", checker_analyzer=a3, cube_analyzer=a2, seed=7)
assert pol.checker_analyzer is a3 and pol.cube_analyzer is a2
assert pol.select_checker_move(board.STARTING_BOARD, 3, 1, away1=5, away2=5) == board.STARTING_BOARD
assert checker_calls == [("3ply", 3, 1, {"away1": 5, "away2": 5})]
dec = pol.cube_decision(board.STARTING_BOARD, cube_value=1, cube_owner="CENTERED", away1=5, away2=5)
assert dec.should_double is True and cube_calls == [("2ply", 1, "centered", {"away1": 5, "away2": 5})]
pol.reseed(7)

# Backward-compatible analyzer= + cube_analyzer=
pol2 = difficulty.DifficultyPolicy("full_strength", analyzer=a3, cube_analyzer=a2)
assert pol2.checker_analyzer is a3 and pol2.cube_analyzer is a2

# Conflict check
try:
    difficulty.DifficultyPolicy("full_strength", analyzer=a3, checker_analyzer=a2, cube_analyzer=a2)
    raise AssertionError("expected ValueError")
except ValueError:
    pass

# Invalid cube_owner in DifficultyPolicy.cube_decision raises ValueError
try:
    pol.cube_decision(board.STARTING_BOARD, cube_owner="bogus")
    raise AssertionError("expected ValueError")
except ValueError as exc:
    assert "Invalid cube owner" in str(exc)

print("STUB_NATIVE_TESTS_OK")
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(import_root)],
        capture_output=True, text=True, timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert "STUB_NATIVE_TESTS_OK" in completed.stdout


# ---------------------------------------------------------------------------
# Native-backed primitive & policy checks (skipped cleanly if bgbot_cpp absent or stale)
# ---------------------------------------------------------------------------


def _require_matching_bgbot_cpp():
    bgbot_cpp = pytest.importorskip("bgbot_cpp")
    if not hasattr(bgbot_cpp, "MONEY_CONTACT_CUBE_LIFE"):
        pytest.skip("stale bgbot_cpp extension missing current constants; rebuild required")
    return bgbot_cpp


def test_board_validate_board_export_and_can_double_match_value_error():
    bgbot_cpp = _require_matching_bgbot_cpp()
    from neongammon import board

    assert board.validate_board is validate_board
    assert board.can_double_match(5, 5, 1, "PLAYER") is True
    assert board.can_double_match(5, 5, 1, bgbot_cpp.CubeOwner.OPPONENT) is False
    with pytest.raises(ValueError, match="Invalid cube owner"):
        board.can_double_match(5, 5, 1, "bogus")


def test_rollout_eval_candidate_checks_cancel_before_native_call():
    _require_matching_bgbot_cpp()
    from neongammon.rollout import RolloutCancelled, _rollout_eval_candidate

    called = False

    class FakeStrategy:
        def cubeful_evaluate_board(self, *args, **kwargs):
            nonlocal called
            called = True
            return {}

    class FakeRolloutAnalyzer:
        _rollout_strategy = FakeStrategy()
        _bearoff_db = None

        def _check_cancel(self):
            raise RolloutCancelled("cancelled before launch")

    with pytest.raises(RolloutCancelled):
        _rollout_eval_candidate(
            FakeRolloutAnalyzer(),
            START,
            START,
            owner="centered",
        )
    assert not called


def test_difficulty_policy_accepts_cached_checker_and_cube_analyzers():
    _require_matching_bgbot_cpp()
    from neongammon import Difficulty, DifficultyPolicy

    checker_calls = []
    cube_calls = []

    class StubAnalyzer:
        def __init__(self, level: str):
            self._eval_level = level

        def checker_play(self, board, die1, die2, **context):
            checker_calls.append((self._eval_level, die1, die2, context))
            return SimpleNamespace(moves=[SimpleNamespace(board=list(board), equity=0.25)])

        def cube_decision(self, board, cube_value=1, cube_owner="centered", **context):
            cube_calls.append((self._eval_level, cube_value, cube_owner, context))
            return CubeDecisionResult(
                equity_nd=0.1, equity_dt=0.2, equity_dp=1.0,
                should_double=True, should_take=True, optimal_equity=0.2,
                optimal_action="Double/Take", eval_level=f"{self._eval_level[0]}-ply",
            )

    a3 = StubAnalyzer("3ply")
    a2 = StubAnalyzer("2ply")

    # Explicit checker_analyzer + cube_analyzer
    policy = DifficultyPolicy(
        Difficulty.FULL_STRENGTH,
        checker_analyzer=a3,
        cube_analyzer=a2,
        seed=123,
    )
    assert policy.checker_analyzer is a3
    assert policy.cube_analyzer is a2

    selected = policy.select_checker_move(START, 3, 1, away1=5, away2=5)
    assert selected == START
    assert checker_calls == [("3ply", 3, 1, {"away1": 5, "away2": 5})]

    dec = policy.cube_decision(START, cube_value=1, cube_owner="CENTERED", away1=5, away2=5)
    assert dec.should_double is True and dec.eval_level == "2-ply"
    assert cube_calls == [("2ply", 1, "centered", {"away1": 5, "away2": 5})]

    # Backward-compatible analyzer= + cube_analyzer=
    policy_compat = DifficultyPolicy("full_strength", analyzer=a3, cube_analyzer=a2, seed=99)
    assert policy_compat.checker_analyzer is a3 and policy_compat.cube_analyzer is a2

    # Same object passed to both is allowed; distinct objects rejected
    policy_same = DifficultyPolicy("full_strength", analyzer=a3, checker_analyzer=a3, cube_analyzer=a2)
    assert policy_same.checker_analyzer is a3
    with pytest.raises(ValueError, match="not both"):
        DifficultyPolicy("full_strength", analyzer=a3, checker_analyzer=a2, cube_analyzer=a2)

    # Invalid cube_owner rejected with ValueError
    with pytest.raises(ValueError, match="Invalid cube owner"):
        policy.cube_decision(START, cube_owner="bogus")
