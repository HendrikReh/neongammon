# SPDX-License-Identifier: MPL-2.0
"""Regression tests for [NG-CPP-03]:
- Null-safe batch_evaluate_candidates_* across all NN strategies
- BearoffStrategy batch overrides (non-bearoff, all-bearoff, mixed bearoff/non-bearoff, nested)
- Eliminated ~1.52x non-bearoff hybrid 2-ply slowdown when BearoffDB is attached
- Unconditional batched Strategy::best_move_index_cubeful / best_move_index_cubeful_multi
- Batched 1-ply prefilter in RolloutStrategy::best_move_index_cubeful_multi
"""

import time

import pytest

import bgbot_cpp
from neongammon import BgBotAnalyzer, STARTING_BOARD, possible_moves
from neongammon.weights import WeightConfig, bearoff_db_path


# Standard contact opening (non-bearoff)
CONTACT_BOARD = list(STARTING_BOARD)

# Midgame contact position (non-bearoff)
MIDGAME_CONTACT = [
    0, -2, 0, 0, 0, 2, 4, 0, 2, 0, 0, 0, -5,
    5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0,
]

# Deep 3-anchor backgame (hits BLENDED_PLAYER_BG_IDX on stage10)
BLENDED_BACKGAME = [0] * 26
BLENDED_BACKGAME[24] = 2
BLENDED_BACKGAME[23] = 2
BLENDED_BACKGAME[22] = 2
BLENDED_BACKGAME[13] = 3
BLENDED_BACKGAME[8] = 3
BLENDED_BACKGAME[6] = 3
BLENDED_BACKGAME[19] = -5
BLENDED_BACKGAME[20] = -5
BLENDED_BACKGAME[21] = -5

# Pure bearoff pre-move board (on roll 2-1, includes terminal win candidate)
BEAROFF_PRE = [0] * 26
BEAROFF_PRE[1] = 1
BEAROFF_PRE[2] = 1
BEAROFF_PRE[22] = -2
BEAROFF_PRE[23] = -2

# Mixed bearoff / non-bearoff boundary position:
# Player has 1 checker on point 7 and 9 checkers in home board; opponent is all home.
# On roll 2-1, some legal moves bring the 7-point checker into home (bearoff DB hit)
# while others leave it on point 7 (non-bearoff NN evaluation).
MIXED_BEAROFF_PRE = [0] * 26
MIXED_BEAROFF_PRE[7] = 1
MIXED_BEAROFF_PRE[6] = 3
MIXED_BEAROFF_PRE[5] = 3
MIXED_BEAROFF_PRE[4] = 3
MIXED_BEAROFF_PRE[19] = -4
MIXED_BEAROFF_PRE[20] = -4
MIXED_BEAROFF_PRE[21] = -4


def _load_db():
    path = bearoff_db_path()
    if path is None:
        pytest.skip("bearoff database not present")
    db = bgbot_cpp.BearoffDB()
    assert db.load(path)
    return db


def _load_model_strategy(model_name: str):
    w = WeightConfig.from_model(model_name)
    try:
        w.validate()
    except FileNotFoundError:
        pytest.skip(f"weights for {model_name} not available")
    return w, bgbot_cpp.create_strategy(
        w.strategy_type, w.weight_paths_list, w.hidden_sizes_list
    )


@pytest.fixture(scope="module")
def bearoff_db():
    return _load_db()


@pytest.fixture(scope="module")
def model_strategies():
    out = {}
    for name in ("stage5", "stage8", "stage10", "stage11"):
        w = WeightConfig.from_model(name)
        try:
            w.validate()
        except FileNotFoundError:
            continue
        out[name] = (
            w,
            bgbot_cpp.create_strategy(
                w.strategy_type, w.weight_paths_list, w.hidden_sizes_list
            ),
        )
    assert "stage11" in out, "production stage11 weights must be present"
    return out


# ---------------------------------------------------------------------------
# 1. Non-bearoff parity between BearoffStrategy-wrapped and unwrapped
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ["stage5", "stage8", "stage10", "stage11"])
def test_bearoff_wrapped_matches_unwrapped_on_non_bearoff(
    model_strategies, bearoff_db, model_name
):
    if model_name not in model_strategies:
        pytest.skip(f"{model_name} weights not present")
    _, raw = model_strategies[model_name]
    wrapped = bgbot_cpp.BearoffStrategy(raw, bearoff_db)
    nested_same = bgbot_cpp.BearoffStrategy(wrapped, bearoff_db)
    nested_unloaded = bgbot_cpp.BearoffStrategy(wrapped, bgbot_cpp.BearoffDB())

    positions = [
        (CONTACT_BOARD, 3, 1),
        (MIDGAME_CONTACT, 6, 4),
        (BLENDED_BACKGAME, 6, 4),
    ]

    for board, d1, d2 in positions:
        cands = [list(c) for c in possible_moves(board, d1, d2)]
        assert cands and all(not bearoff_db.is_bearoff(c) for c in cands)

        # 1-ply evaluate_board parity
        for c in cands[:5]:
            r_raw = raw.evaluate_board(c, board)
            for s in (wrapped, nested_same, nested_unloaded):
                r_wrap = s.evaluate_board(c, board)
                assert r_wrap["probs"] == pytest.approx(r_raw["probs"], abs=1e-7)
                assert r_wrap["equity"] == pytest.approx(r_raw["equity"], abs=1e-7)

        # 2-ply MultiPlyStrategy parity (exercises batch_evaluate_candidates_best_prob
        # and batch_evaluate_candidates_equity through BearoffStrategy!)
        mp_raw = bgbot_cpp.create_multipy_from_base(raw, n_plies=2)
        mp_wrap = bgbot_cpp.create_multipy_from_base(raw, n_plies=2)
        bgbot_cpp.multipy_set_bearoff_db(mp_wrap, bearoff_db)
        mp_raw.clear_cache()
        ref = mp_raw.evaluate_board(cands[0], board)
        mp_wrap.clear_cache()
        got = mp_wrap.evaluate_board(cands[0], board)
        assert got["probs"] == pytest.approx(ref["probs"], abs=1e-6)
        assert got["equity"] == pytest.approx(ref["equity"], abs=1e-6)

        # 2-ply best_move_index parity
        mp_raw.clear_cache()
        idx_raw = mp_raw.best_move_index(cands, board)
        mp_wrap.clear_cache()
        idx_wrap = mp_wrap.best_move_index(cands, board)
        assert idx_wrap == idx_raw


# ---------------------------------------------------------------------------
# 2. Exact DB probs on all-bearoff and mixed bearoff/non-bearoff candidate sets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ["stage5", "stage8", "stage11"])
def test_bearoff_and_mixed_candidate_sets(model_strategies, bearoff_db, model_name):
    if model_name not in model_strategies:
        pytest.skip(f"{model_name} weights not present")
    _, raw = model_strategies[model_name]
    wrapped = bgbot_cpp.BearoffStrategy(raw, bearoff_db)

    # Confirm MIXED_BEAROFF_PRE on 2-1 produces both bearoff and non-bearoff candidates
    mixed_cands = [list(c) for c in possible_moves(MIXED_BEAROFF_PRE, 2, 1)]
    bo_cands = [c for c in mixed_cands if bearoff_db.is_bearoff(c)]
    non_bo_cands = [c for c in mixed_cands if not bearoff_db.is_bearoff(c)]
    assert bo_cands and non_bo_cands

    # Bearoff candidates must match exact DB probs; non-bearoff candidates match raw NN
    for c in bo_cands:
        got = wrapped.evaluate_board(c, MIXED_BEAROFF_PRE)["probs"]
        expected = bearoff_db.lookup_probs(c, post_move=True)
        assert list(got) == pytest.approx(list(expected), abs=1e-7)
    for c in non_bo_cands:
        got = wrapped.evaluate_board(c, MIXED_BEAROFF_PRE)["probs"]
        expected = raw.evaluate_board(c, MIXED_BEAROFF_PRE)["probs"]
        assert list(got) == pytest.approx(list(expected), abs=1e-7)

    # 2-ply MultiPlyStrategy on mixed bearoff/non-bearoff boundary:
    # exercises BearoffStrategy::batch_evaluate_candidates_equity (mixed set)
    # and BearoffStrategy::batch_evaluate_candidates_best_prob on opponent replies
    mp_bo = bgbot_cpp.create_multipy_from_base(raw, n_plies=2)
    bgbot_cpp.multipy_set_bearoff_db(mp_bo, bearoff_db)
    mp_bo.clear_cache()
    idx_bo = mp_bo.best_move_index(mixed_cands, MIXED_BEAROFF_PRE)
    assert 0 <= idx_bo < len(mixed_cands)

    # Pure bearoff candidate set (includes terminal win)
    pure_bo_cands = [list(c) for c in possible_moves(BEAROFF_PRE, 2, 1)]
    assert pure_bo_cands and all(bearoff_db.is_bearoff(c) for c in pure_bo_cands)
    mp_bo.clear_cache()
    idx_pure = mp_bo.best_move_index(pure_bo_cands, BEAROFF_PRE)
    # Winning terminal (all player checkers off) must be chosen
    assert sum(max(0, x) for x in pure_bo_cands[idx_pure][1:26]) == 0

    # Mixed bearoff/non-bearoff cubeful 1-ply analyzer check
    if model_name == "stage11":
        an = BgBotAnalyzer(eval_level="1ply", cubeful=True, bearoff_db=True)
        res = an.checker_play(MIXED_BEAROFF_PRE, 2, 1, cube_value=1, cube_owner="centered")
        assert len(res.moves) == len(mixed_cands)
        for m in res.moves:
            b = list(m.board)
            if bearoff_db.is_bearoff(b):
                exp = bearoff_db.lookup_probs(b, post_move=True)
                assert [
                    m.probs.win, m.probs.gammon_win, m.probs.backgammon_win,
                    m.probs.gammon_loss, m.probs.backgammon_loss,
                ] == pytest.approx(list(exp), abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Non-bearoff 2-ply hybrid / BearoffStrategy timing regression guard
# ---------------------------------------------------------------------------

def test_non_bearoff_2ply_hybrid_bearoff_no_slowdown(model_strategies, bearoff_db):
    """Attaching BearoffDB to a 2-ply hybrid MultiPlyStrategy must NOT regress
    non-bearoff evaluation speed (~1.52x before BearoffStrategy batch overrides)."""
    if "stage5" not in model_strategies:
        pytest.skip("stage5 weights not present")
    w5, s5 = model_strategies["stage5"]
    w11, s11 = model_strategies["stage11"]

    # Build both standard 2-ply (stage11) and hybrid 2-ply (5-NN filter + 5-NN base)
    mp_plain = bgbot_cpp.create_multipy_from_base(s11, n_plies=2)
    mp_with_db = bgbot_cpp.create_multipy_from_base(s11, n_plies=2)
    bgbot_cpp.multipy_set_bearoff_db(mp_with_db, bearoff_db)

    hyb_plain = bgbot_cpp.create_multipy_hybrid_5nn(
        *w5.weight_args, *w5.weight_args, n_plies=2
    )
    hyb_with_db = bgbot_cpp.create_multipy_hybrid_5nn(
        *w5.weight_args, *w5.weight_args, n_plies=2
    )
    bgbot_cpp.multipy_set_bearoff_db(hyb_with_db, bearoff_db)

    cands = [list(c) for c in possible_moves(CONTACT_BOARD, 3, 1)][:6]

    def _bench(plain, with_db, iters):
        # Warmup + parity check
        plain.clear_cache()
        ref = [plain.evaluate_board(c, CONTACT_BOARD)["probs"] for c in cands]
        with_db.clear_cache()
        got = [with_db.evaluate_board(c, CONTACT_BOARD)["probs"] for c in cands]
        for r, g in zip(ref, got):
            assert list(g) == pytest.approx(list(r), abs=1e-6)

        # Interleave runs to avoid thermal/frequency skew
        t_plain = 0.0
        t_db = 0.0
        for _ in range(iters):
            plain.clear_cache()
            t0 = time.perf_counter()
            for c in cands:
                plain.evaluate_board(c, CONTACT_BOARD)
            t_plain += time.perf_counter() - t0

            with_db.clear_cache()
            t0 = time.perf_counter()
            for c in cands:
                with_db.evaluate_board(c, CONTACT_BOARD)
            t_db += time.perf_counter() - t0
        return t_plain, t_db

    t_p, t_d = _bench(mp_plain, mp_with_db, iters=12)
    ratio_std = t_d / t_p
    assert ratio_std < 1.25, (
        f"2-ply stage11 with BearoffDB regressed: plain={t_p*1000:.1f}ms, "
        f"with_db={t_d*1000:.1f}ms, ratio={ratio_std:.2f}x"
    )

    t_hp, t_hd = _bench(hyb_plain, hyb_with_db, iters=12)
    ratio_hyb = t_hd / t_hp
    assert ratio_hyb < 1.25, (
        f"2-ply hybrid with BearoffDB regressed: plain={t_hp*1000:.1f}ms, "
        f"with_db={t_hd*1000:.1f}ms, ratio={ratio_hyb:.2f}x"
    )


# ---------------------------------------------------------------------------
# 4. Cubeful rollout & best_move_index_cubeful_multi across all strategy types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ["stage5", "stage8", "stage11"])
@pytest.mark.parametrize("use_bearoff", [False, True])
def test_cubeful_rollout_across_models_and_bearoff(
    model_strategies, bearoff_db, model_name, use_bearoff
):
    """Cubeful rollouts exercise Strategy::best_move_index_cubeful_multi and
    RolloutStrategy::best_move_probs_for_candidates across 5-NN, 17-NN, 24-NN,
    and BearoffStrategy-wrapped bases (including VR-on vs VR-off parity)."""
    if model_name not in model_strategies:
        pytest.skip(f"{model_name} weights not present")
    w, _ = model_strategies[model_name]

    def _run(enable_vr, n_threads):
        rs = bgbot_cpp.create_rollout(
            w.strategy_type,
            w.weight_paths_list,
            w.hidden_sizes_list,
            n_trials=72,
            truncation_depth=4,
            decision_ply=1,
            n_threads=n_threads,
            seed=42,
            enable_vr=enable_vr,
            cubeful_trial_moves=True,
            parallelize_trials=(n_threads > 1),
        )
        if use_bearoff:
            bgbot_cpp.rollout_set_bearoff_db(rs, bearoff_db)
        return rs.cube_decision(MIDGAME_CONTACT, 1, bgbot_cpp.CubeOwner.CENTERED)

    r_vr_1 = _run(enable_vr=True, n_threads=1)
    r_vr_4 = _run(enable_vr=True, n_threads=4)
    r_novr = _run(enable_vr=False, n_threads=1)

    # Serial vs 4-thread parallel determinism & thread safety of batched cubeful BMI
    assert r_vr_4["equity_nd"] == pytest.approx(r_vr_1["equity_nd"], abs=1e-4)
    assert r_vr_4["equity_dt"] == pytest.approx(r_vr_1["equity_dt"], abs=1e-4)
    assert list(r_vr_4["probs"]) == pytest.approx(list(r_vr_1["probs"]), abs=1e-4)

    # VR must remain unbiased and produce valid probabilities
    for p in r_vr_1["probs"]:
        assert -0.02 <= p <= 1.02
    assert abs(r_vr_1["cubeless_equity"] - r_novr["cubeless_equity"]) < 0.12
    assert abs(r_vr_1["equity_nd"] - r_novr["equity_nd"]) < 0.15


def test_nested_rollout_exercises_rollout_best_move_index_cubeful_multi(
    model_strategies, bearoff_db
):
    """Using a truncated rollout as the trial-level checker strategy directly
    invokes RolloutStrategy::best_move_index_cubeful_multi (whose 1-ply prefilter
    now uses batched batch_evaluate_candidates_equity_probs with nullptr equities)."""
    w, _ = model_strategies["stage11"]
    nested_checker = bgbot_cpp.TrialEvalConfig(
        ply=0, rollout_trials=18, rollout_depth=2, rollout_ply=1
    )
    rs = bgbot_cpp.create_rollout(
        w.strategy_type,
        w.weight_paths_list,
        w.hidden_sizes_list,
        n_trials=18,
        truncation_depth=2,
        decision_ply=1,
        filter_max_moves=3,
        filter_threshold=0.08,
        n_threads=1,
        seed=123,
        enable_vr=True,
        checker=nested_checker,
        cubeful_trial_moves=True,
    )
    bgbot_cpp.rollout_set_bearoff_db(rs, bearoff_db)
    res = rs.cube_decision(MIDGAME_CONTACT, 1, bgbot_cpp.CubeOwner.CENTERED)
    assert -3.0 < res["equity_nd"] < 3.0
    assert -3.0 < res["equity_dt"] < 3.0
    assert 0.0 <= res["probs"][0] <= 1.0
