# SPDX-License-Identifier: MPL-2.0
"""N-ply batch_checker_play returns BgBotAnalyzer.checker_play's ranked list.

Both N-ply batch overloads used to run the cube-aware tree for every legal
move, so at 2-ply the pick could land on a move the analyzer's 1-ply filter
had rejected. Beyond that, they took the probs from the cubeless tree, routed
the cube tree per node instead of from the decision's root, and searched match
positions at cube value 1. They now go through the analyzer's steps one for
one (filter, cubeless search and promotion, cube tree for the searched moves,
1-ply Janowski for the rest, cubeful promotion), so the two lists are equal
field for field: with the bearoff database (both default to it) and without.
"""

import os
import random
import subprocess
import sys
from dataclasses import asdict
from unittest.mock import patch

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "build"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bgbot_cpp  # noqa: E402

from neongammon import (  # noqa: E402
    BgBotAnalyzer, STARTING_BOARD, batch_checker_play, batch_evaluate,
    batch_post_move_evaluate, flip_board, possible_moves,
)
from neongammon.analyzer import SERIAL_CUBEFUL_1PLY  # noqa: E402
from neongammon.board import check_game_over  # noqa: E402

# A snake: the 24-NN layout pins every contact node of its tree to NN 22.
SNAKE = [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 3, 3, 2, 2, 1, -7, -7, 0]
# Both sides home: candidates and leaves come from the bearoff database.
BEAROFF = [0, 3, 3, 2, 3, 2, 2] + [0] * 12 + [-2, -3, -2, -3, -2, -2] + [0]
# Two checkers still outside: some candidates are bearoff positions, some not.
NEAR_BEAROFF = [0, 2, 3, 0, 2, 3, 1, 0, 1, 1] + [0] * 9 + [-3, -2, -2, -3, -3, -2] + [0]

CUBES = [
    dict(cube_value=1, cube_owner="centered"),
    dict(cube_value=2, cube_owner="opponent"),
    dict(cube_value=2, cube_owner="player"),
    dict(cube_value=1, cube_owner="centered", away1=5, away2=3),
    dict(cube_value=2, cube_owner="player", away1=7, away2=4),
    dict(cube_value=1, cube_owner="centered", away1=1, away2=4, is_crawford=True),
]


def _random_positions(n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = list(STARTING_BOARD)
        for _ in range(rng.randint(4, 40)):
            moves = possible_moves(board, rng.randint(1, 6), rng.randint(1, 6))
            if moves:
                board = list(rng.choice(moves))
            if check_game_over(board):
                break
            board = list(flip_board(board))
        else:
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
            if len(possible_moves(board, d1, d2)) >= 2:
                out.append(dict(board=board, die1=d1, die2=d2,
                                **CUBES[len(out) % len(CUBES)]))
    return out


FIXED = [dict(board=SNAKE, die1=6, die2=5, **CUBES[0]),
         dict(board=SNAKE, die1=2, die2=1, **CUBES[4]),
         dict(board=BEAROFF, die1=4, die2=2, **CUBES[3]),
         dict(board=BEAROFF, die1=6, die2=1, **CUBES[0]),
         dict(board=NEAR_BEAROFF, die1=5, die2=3, **CUBES[0]),
         dict(board=NEAR_BEAROFF, die1=6, die2=6, **CUBES[1])]
POSITIONS = {"1ply": _random_positions(36, seed=7) + FIXED,
             "2ply": _random_positions(36, seed=7) + FIXED,
             "3ply": _random_positions(12, seed=7) + FIXED}


def _analyzer_moves(level, positions, bearoff_db=True):
    analyzer = BgBotAnalyzer(eval_level=level, cubeful=True, parallel_threads=2,
                             bearoff_db=bearoff_db)
    out = []
    for p in positions:
        kw = {k: v for k, v in p.items() if k not in ("board", "die1", "die2")}
        out.append([asdict(m) for m in
                    analyzer.checker_play(p["board"], p["die1"], p["die2"], **kw).moves])
    return out


def _batch_moves(level, positions, n_threads=4, bearoff_db=None):
    return [[asdict(m) for m in r.moves]
            for r in batch_checker_play(positions, eval_level=level, n_threads=n_threads,
                                        bearoff_db=bearoff_db)]


@pytest.fixture(scope="module", params=["1ply", "2ply", "3ply"])
def lists(request):
    level = request.param
    positions = POSITIONS[level]
    return level, _batch_moves(level, positions), _analyzer_moves(level, positions)


def _assert_moves_match(level, batch_moves, analyzer_moves, *, tol=1e-5):
    if level != "1ply":
        assert batch_moves == analyzer_moves
        return
    # At 1-ply, batch_checker_play evaluates candidates via batched BLAS SGEMM
    # (score_checker_1ply -> batch_evaluate_candidates_equity_probs), whereas
    # BgBotAnalyzer("1ply") calls scalar evaluate_board per candidate; float32
    # accumulation order can differ by ~1e-6 while move ordering stays exact.
    assert len(batch_moves) == len(analyzer_moves)
    for bm, am in zip(batch_moves, analyzer_moves):
        assert bm["board"] == am["board"]
        assert bm["eval_level"] == am["eval_level"]
        assert bm["player_game_plan"] == am["player_game_plan"]
        assert bm["opponent_game_plan"] == am["opponent_game_plan"]
        assert bm["std_error"] == am["std_error"]
        assert bm["prob_std_errors"] == am["prob_std_errors"]
        assert bm["equity"] == pytest.approx(am["equity"], abs=tol)
        assert bm["cubeless_equity"] == pytest.approx(am["cubeless_equity"], abs=tol)
        assert bm["equity_diff"] == pytest.approx(am["equity_diff"], abs=tol)
        for k in ("win", "gammon_win", "backgammon_win", "gammon_loss", "backgammon_loss"):
            assert bm["probs"][k] == pytest.approx(am["probs"][k], abs=tol)


def test_batch_returns_the_analyzers_list(lists):
    level, batch, analyzer = lists
    for i, (b, a) in enumerate(zip(batch, analyzer)):
        _assert_moves_match(level, b, a)


def test_two_ply_searches_only_the_filtered_moves(lists):
    # At 2-ply the moves the filter pruned stay on 1-ply values unless they
    # were promoted; at 3-ply every move is searched.
    level, batch, _ = lists
    labels = {m["eval_level"] for moves in batch for m in moves}
    assert labels == {"1ply": {"1-ply"}, "2ply": {"2-ply", "1-ply"}, "3ply": {"3-ply"}}[level]
    for moves in batch:
        assert all(m["eval_level"] == f"{level[0]}-ply" for m in moves[:2])


def test_the_fixture_exercises_promotion(lists):
    # The parity above covers the promotion loops only if moves outside the
    # 1-ply filter get searched: here several do, and some end up best.
    level, batch, _ = lists
    if level != "2ply":
        pytest.skip("only 2-ply filters and promotes")
    inner = BgBotAnalyzer(eval_level="2ply", cubeful=True, parallel_threads=2)._analyzer._inner
    promoted_best = promoted_any = 0
    for p, moves in zip(POSITIONS["2ply"], batch):
        kw = {k: p[k] for k in ("cube_value", "away1", "away2", "is_crawford") if k in p}
        scored = inner._score_candidates(
            possible_moves(p["board"], p["die1"], p["die2"]), p["board"],
            p["cube_owner"], **kw)
        _, survivors = inner._filter_candidates(
            scored, inner.FILTER_THRESHOLD, inner.FILTER_MAX_MOVES)
        promoted = [k for k, m in enumerate(moves)
                    if m["eval_level"] == "2-ply" and tuple(m["board"]) not in survivors]
        promoted_any += bool(promoted)
        promoted_best += 0 in promoted
    assert promoted_any >= 10
    assert promoted_best >= 1


def test_the_filtered_depths_come_from_python():
    # With no filtered depth, 2-ply searches every move, in the analyzer too.
    positions = POSITIONS["2ply"][:6]
    with patch("neongammon.batch.CUBEFUL_FILTERED_PLIES", frozenset()), \
            patch("neongammon.analyzer.CUBEFUL_FILTERED_PLIES", frozenset()):
        batch = _batch_moves("2ply", positions)
        analyzer = _analyzer_moves("2ply", positions)
    assert batch == analyzer
    assert all(m["eval_level"] == "2-ply" for moves in batch for m in moves)


def test_without_the_bearoff_database():
    positions = FIXED[2:]
    assert _batch_moves("2ply", positions, bearoff_db=False) == \
        _analyzer_moves("2ply", positions, bearoff_db=False)
    for db in (False, True):
        for bm, am in zip(_batch_moves("1ply", positions, bearoff_db=db),
                          _analyzer_moves("1ply", positions, bearoff_db=db)):
            _assert_moves_match("1ply", bm, am)
    # And the database does change these positions.
    assert _batch_moves("2ply", positions, bearoff_db=False) != \
        _batch_moves("2ply", positions)


def test_thread_count_does_not_change_the_result():
    positions = POSITIONS["2ply"][:12]
    assert _batch_moves("2ply", positions, n_threads=1) == \
        _batch_moves("2ply", positions, n_threads=6)


# ---------------------------------------------------------------------------
# Dead-cube, beaver, and force_boards parity in batch_checker_play
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", ["1ply", "2ply"])
@pytest.mark.parametrize("extra", [
    {},
    {"away1": 5, "away2": 3, "is_crawford": False},
], ids=["money", "match"])
def test_dead_cube_parity_in_batch_checker_play(level, extra):
    """max_cube_value=2 at cube_value=2 makes the cube dead; batch_checker_play
    must match BgBotAnalyzer.checker_play within 1e-5 at both 1ply and 2ply
    (eliminating the +0.226 live-cube drift)."""
    pos = dict(board=list(STARTING_BOARD), die1=3, die2=1,
               cube_value=2, cube_owner="player", max_cube_value=2, **extra)
    analyzer = BgBotAnalyzer(eval_level=level, cubeful=True, parallel_threads=2)
    expected = [asdict(m) for m in analyzer.checker_play(
        pos["board"], pos["die1"], pos["die2"],
        cube_value=pos["cube_value"], cube_owner=pos["cube_owner"],
        max_cube_value=pos["max_cube_value"], **extra,
    ).moves]
    got = [asdict(m) for m in batch_checker_play(
        [pos], eval_level=level, n_threads=2,
    )[0].moves]
    _assert_moves_match(level, got, expected, tol=1e-5)

    # Confirm the dead-cube cap actually changes the money equity by ~0.226
    # relative to an uncapped cube, so this test cannot pass vacuously.
    if not extra:
        live = analyzer.checker_play(
            pos["board"], pos["die1"], pos["die2"],
            cube_value=2, cube_owner="player",
        ).moves[0]
        assert abs(live.equity - got[0]["equity"]) > 0.15


def test_batch_checker_play_validates_max_cube_value():
    base = dict(board=list(STARTING_BOARD), die1=3, die2=1,
                cube_value=2, cube_owner="player")
    for bad in (1, 3, -2, True):
        with pytest.raises(ValueError):
            batch_checker_play([dict(base, max_cube_value=bad)], eval_level="1ply")
    with pytest.raises(ValueError):
        batch_checker_play([dict(base, cube_value=4, max_cube_value=2)], eval_level="2ply")


@pytest.mark.parametrize("level", ["1ply", "2ply"])
@pytest.mark.parametrize("beaver", [True, False])
def test_beaver_forwarding_and_parity(level, beaver):
    positions = [
        dict(board=list(STARTING_BOARD), die1=3, die2=1,
             cube_value=1, cube_owner="centered"),
        dict(board=SNAKE, die1=6, die2=5, cube_value=2, cube_owner="opponent"),
        dict(board=list(STARTING_BOARD), die1=6, die2=5,
             cube_value=1, cube_owner="centered", away1=5, away2=3),
    ]
    analyzer = BgBotAnalyzer(eval_level=level, cubeful=True, parallel_threads=2)
    with patch.object(bgbot_cpp, "batch_checker_play",
                      wraps=bgbot_cpp.batch_checker_play) as spy:
        batch_res = batch_checker_play(
            positions, eval_level=level, n_threads=2, beaver=beaver,
        )
    assert spy.call_args.kwargs["beaver"] is beaver
    for p, br in zip(positions, batch_res):
        kw = {k: v for k, v in p.items() if k not in ("board", "die1", "die2")}
        ar = analyzer.checker_play(p["board"], p["die1"], p["die2"],
                                   beaver=beaver, **kw)
        _assert_moves_match(level, [asdict(m) for m in br.moves],
                            [asdict(m) for m in ar.moves], tol=1e-5)


@pytest.mark.parametrize("level", ["1ply", "2ply"])
def test_force_boards_forwarding_and_parity(level):
    """Per-position force_boards in batch_checker_play must force pruned moves
    to be scored at 2-ply and match BgBotAnalyzer.checker_play."""
    analyzer_2 = BgBotAnalyzer(eval_level="2ply", cubeful=True, parallel_threads=2)
    baseline = analyzer_2.checker_play(list(STARTING_BOARD), 3, 1,
                                       cube_value=1, cube_owner="centered").moves
    pruned = [m.board for m in baseline if m.eval_level == "1-ply"]
    assert pruned, "fixture: 3-1 from STARTING_BOARD should have 1-ply pruned moves at 2-ply"
    target = list(pruned[-1])

    pos = dict(board=list(STARTING_BOARD), die1=3, die2=1,
               cube_value=1, cube_owner="centered", force_boards=[target])
    analyzer = BgBotAnalyzer(eval_level=level, cubeful=True, parallel_threads=2)
    expected = [asdict(m) for m in analyzer.checker_play(
        pos["board"], pos["die1"], pos["die2"],
        cube_value=pos["cube_value"], cube_owner=pos["cube_owner"],
        force_boards=[target],
    ).moves]
    got = [asdict(m) for m in batch_checker_play(
        [pos], eval_level=level, n_threads=2,
    )[0].moves]
    _assert_moves_match(level, got, expected, tol=1e-5)
    if level == "2ply":
        by_board = {tuple(m["board"]): m for m in got}
        assert by_board[tuple(target)]["eval_level"] == "2-ply"


# ---------------------------------------------------------------------------
# Impossible 1-ply gammon/backgammon probability clamping across ALL entry points
# ---------------------------------------------------------------------------

# Legal 14-checker hit-back contact board: player already has 1 checker borne
# off (11 in home on points 1-5, 3 hit back on point 24), opponent has 15
# checkers on points 18-23 (contact!). Because player has >= 1 checker borne
# off, P(gammon_loss) == 0.0 and P(backgammon_loss) == 0.0 is an absolute
# backgammon invariant (pre-fix, raw 1-ply NN emitted P(gl) ~ 0.013-0.024 and
# P(bl) ~ 0.003-0.004 across all 1-ply entry points).
HITBACK_14_CONTACT = [
    0, 3, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, -1, -2, -3, -3, -3, -3, 3, 0,
]


def _cl_eq_from_probs(p):
    return 2.0 * p.win - 1.0 + p.gammon_win - p.gammon_loss + p.backgammon_win - p.backgammon_loss


@pytest.mark.parametrize("bearoff_db", [True, False])
def test_one_ply_clamps_impossible_gammons_across_all_entry_points(bearoff_db):
    board = list(HITBACK_14_CONTACT)
    assert sum(c for c in board[1:25] if c > 0) + board[25] == 14
    assert sum(-c for c in board[1:25] if c < 0) + board[0] == 15
    assert not bgbot_cpp.is_race(board)

    analyzer = BgBotAnalyzer(eval_level="1ply", cubeful=True, bearoff_db=bearoff_db)

    # 1. BgBotAnalyzer("1ply").checker_play
    cp = analyzer.checker_play(board, 3, 1, cube_value=1, cube_owner="centered")
    assert cp.moves
    for m in cp.moves:
        assert m.probs.gammon_loss == 0.0
        assert m.probs.backgammon_loss == 0.0
        assert m.cubeless_equity == pytest.approx(_cl_eq_from_probs(m.probs), abs=1e-6)

    # 2. batch_checker_play("1ply")
    bcp = batch_checker_play(
        [{"board": board, "die1": 3, "die2": 1, "cube_value": 1, "cube_owner": "centered"}],
        eval_level="1ply", bearoff_db=bearoff_db,
    )[0]
    assert len(bcp.moves) == len(cp.moves)
    for m in bcp.moves:
        assert m.probs.gammon_loss == 0.0
        assert m.probs.backgammon_loss == 0.0
        assert m.cubeless_equity == pytest.approx(_cl_eq_from_probs(m.probs), abs=1e-6)
    _assert_moves_match("1ply", [asdict(m) for m in bcp.moves],
                        [asdict(m) for m in cp.moves], tol=1e-5)

    # 3. BgBotAnalyzer("1ply").post_move_analytics
    pm = analyzer.post_move_analytics(board, cube_owner="centered")
    assert pm.probs.gammon_loss == 0.0
    assert pm.probs.backgammon_loss == 0.0
    assert pm.cubeless_equity == pytest.approx(_cl_eq_from_probs(pm.probs), abs=1e-6)

    # 4. batch_post_move_evaluate("1ply")
    bpm = batch_post_move_evaluate(
        [{"board": board, "cube_owner": "centered"}],
        eval_level="1ply", bearoff_db=bearoff_db,
    )[0]
    assert bpm.probs.gammon_loss == 0.0
    assert bpm.probs.backgammon_loss == 0.0
    assert bpm.cubeless_equity == pytest.approx(_cl_eq_from_probs(bpm.probs), abs=1e-6)
    assert bpm.cubeful_equity == pytest.approx(pm.cubeful_equity, abs=1e-6)

    # 5. BgBotAnalyzer("1ply").cube_action
    ca = analyzer.cube_action(board, cube_value=1, cube_owner="centered")
    assert ca.probs.gammon_loss == 0.0
    assert ca.probs.backgammon_loss == 0.0
    assert ca.cubeless_equity == pytest.approx(_cl_eq_from_probs(ca.probs), abs=1e-6)

    # 6. batch_evaluate("1ply")
    be = batch_evaluate(
        [{"board": board, "cube_value": 1, "cube_owner": "centered"}],
        eval_level="1ply", bearoff_db=bearoff_db,
    )[0]
    assert be.probs.gammon_loss == 0.0
    assert be.probs.backgammon_loss == 0.0
    assert be.cubeless_equity == pytest.approx(_cl_eq_from_probs(be.probs), abs=1e-6)
    assert be.equity_nd == pytest.approx(ca.equity_nd, abs=1e-6)

    # 7. Flipped board (opponent has 1 checker borne off -> P(gw)==0, P(bw)==0)
    flipped = list(flip_board(board))
    pm_f = analyzer.post_move_analytics(flipped, cube_owner="centered")
    assert pm_f.probs.gammon_win == 0.0 and pm_f.probs.backgammon_win == 0.0
    bpm_f = batch_post_move_evaluate(
        [{"board": flipped, "cube_owner": "centered"}],
        eval_level="1ply", bearoff_db=bearoff_db,
    )[0]
    assert bpm_f.probs.gammon_win == 0.0 and bpm_f.probs.backgammon_win == 0.0
    be_f = batch_evaluate(
        [{"board": flipped, "cube_value": 1, "cube_owner": "centered"}],
        eval_level="1ply", bearoff_db=bearoff_db,
    )[0]
    assert be_f.probs.gammon_win == 0.0 and be_f.probs.backgammon_win == 0.0

    # 8. 2-ply score_checker_position pruned 1-ply moves must also be clamped
    bcp2 = batch_checker_play(
        [{"board": board, "die1": 3, "die2": 1, "cube_value": 1, "cube_owner": "centered"}],
        eval_level="2ply", bearoff_db=bearoff_db,
    )[0]
    for m in bcp2.moves:
        assert m.probs.gammon_loss == 0.0
        assert m.probs.backgammon_loss == 0.0


def test_one_ply_serial_guard_spawns_no_pool_threads():
    """Preserve SERIAL_CUBEFUL_1PLY and ensure single-position 1-ply checker_play
    (and n_threads=1 batch_checker_play) never construct MultiPlyThreadPool."""
    assert SERIAL_CUBEFUL_1PLY is True
    if sys.platform != "darwin":
        pytest.skip("OS thread count check via proc_pidinfo is macOS-specific")
    probe = f"""
import ctypes, os, sys
sys.path.insert(0, {os.path.join(_ROOT, "build")!r})
sys.path.insert(0, {os.path.join(_ROOT, "python")!r})
libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
class ProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("pti_virtual_size", ctypes.c_uint64), ("pti_resident_size", ctypes.c_uint64),
        ("pti_total_user", ctypes.c_uint64), ("pti_total_system", ctypes.c_uint64),
        ("pti_threads_user", ctypes.c_uint64), ("pti_threads_system", ctypes.c_uint64),
        ("pti_policy", ctypes.c_int32), ("pti_faults", ctypes.c_int32),
        ("pti_pageins", ctypes.c_int32), ("pti_cow_faults", ctypes.c_int32),
        ("pti_messages_sent", ctypes.c_int32), ("pti_messages_received", ctypes.c_int32),
        ("pti_syscalls_mach", ctypes.c_int32), ("pti_syscalls_unix", ctypes.c_int32),
        ("pti_csw", ctypes.c_int32), ("pti_threadnum", ctypes.c_int32),
        ("pti_numrunning", ctypes.c_int32), ("pti_priority", ctypes.c_int32),
    ]
def threads():
    info = ProcTaskInfo()
    assert libc.proc_pidinfo(os.getpid(), 4, 0, ctypes.byref(info), ctypes.sizeof(info)) == ctypes.sizeof(info)
    return info.pti_threadnum
from neongammon import BgBotAnalyzer, STARTING_BOARD, batch_checker_play
t0 = threads()
BgBotAnalyzer(eval_level="1ply", cubeful=True).checker_play(list(STARTING_BOARD), 3, 1)
assert threads() == t0, f"1-ply checker_play spawned threads: {{t0}} -> {{threads()}}"
batch_checker_play(
    [{{"board": list(STARTING_BOARD), "die1": 3, "die2": 1, "cube_value": 1, "cube_owner": "centered"}}],
    eval_level="1ply", n_threads=1,
)
assert threads() == t0, f"1-ply n_threads=1 batch spawned threads: {{t0}} -> {{threads()}}"
"""
    subprocess.run([sys.executable, "-c", probe], check=True)


@pytest.mark.parametrize("level", ["1ply", "2ply"])
def test_post_move_analytics_cubeful_false_and_batch_parity(level):
    """Issue #59: post_move_analytics honors cubeful=False, matches
    batch_post_move_evaluate at 1-ply and 2-ply across money, match, and
    dead-cube rules, and preserves money cube_value."""
    board = list(possible_moves(list(STARTING_BOARD), 3, 1)[0])

    # cubeful=False must return cubeful_equity == cubeless_equity
    cl_analyzer = BgBotAnalyzer(eval_level=level, cubeful=False, parallel_threads=1)
    cl_res = cl_analyzer.post_move_analytics(board, cube_owner="centered", cube_value=1)
    assert cl_res.cubeful_equity == cl_res.cubeless_equity

    # cubeful=True must match batch_post_move_evaluate on money, match, and dead-cube
    cf_analyzer = BgBotAnalyzer(eval_level=level, cubeful=True, parallel_threads=1)
    cases = [
        dict(cube_owner="centered", cube_value=1),
        dict(cube_owner="player", cube_value=2),
        dict(cube_owner="player", cube_value=2, max_cube_value=2),
        dict(cube_owner="player", cube_value=2, away1=5, away2=3),
        dict(cube_owner="player", cube_value=2, away1=5, away2=3, max_cube_value=2),
    ]
    batch_rows = [dict(board=board, **kw) for kw in cases]
    batch_res = batch_post_move_evaluate(batch_rows, eval_level=level, n_threads=1)
    for kw, br in zip(cases, batch_res):
        sr = cf_analyzer.post_move_analytics(board, **kw)
        assert sr.cubeless_equity == pytest.approx(br.cubeless_equity, abs=1e-6)
        assert sr.cubeful_equity == pytest.approx(br.cubeful_equity, abs=1e-6)
        assert sr.probs.to_list() == pytest.approx(br.probs.to_list(), abs=1e-6)


def test_post_move_analytics_translates_rollout_cancelled():
    """Issue #59: post_move_analytics must translate bgbot_cpp.RolloutCancelled
    into neongammon.RolloutCancelled for pre-cancelled and mid-rollout cancels."""
    from neongammon import RolloutCancelled

    board = list(possible_moves(list(STARTING_BOARD), 3, 1)[0])
    for cubeful in (True, False):
        for n_threads in (1, 2):
            an = BgBotAnalyzer(
                eval_level="rollout", cubeful=cubeful,
                n_trials=36, truncation_depth=3, decision_ply=1,
                parallel_threads=n_threads,
            )
            an.cancel()
            with pytest.raises(RolloutCancelled):
                an.post_move_analytics(board, cube_owner="centered")

            # Also verify C++ mid-rollout cancellation translates cleanly
            an.reset_cancel()
            def cancel_midway(done, total, _an=an):
                _an.cancel()
            with pytest.raises(RolloutCancelled):
                an.post_move_analytics(
                    board, cube_owner="centered", progress_callback=cancel_midway,
                )

