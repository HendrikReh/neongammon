# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""
Tests for the gnubg interface.

Run with:
    python python/tests/test_gnubg.py

Tests:
1. Endgame: both players have 2 on their 1-point -> player on roll wins (equity +1)
2. Gammon certain: player has 2 on 1-point, opponent has not borne off -> P(win)=1, P(gw)=1
3. Compare gnubg 2-ply vs Stage 4 2-ply on top-100 worst 0-ply positions
"""

import math
import os
import shutil
import time
import unittest

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)  # the neongammon repo root

from neongammon.gnubg import (
    GnuBgAnalyzer,
    _board_simple_str,
    _flip_board,
    _flip_cube_owner,
    _parse_cube_analytics,
    _parse_cube_analytics_full,
    _parse_eval_batch,
    _parse_eval_section,
    _parse_eval_single,
    post_move_analytics,
    post_move_analytics_many,
)
from neongammon.settings import gnubg_path

_GNUBG = gnubg_path()
_GNUBG_FOUND = bool(shutil.which(_GNUBG)) or os.path.isfile(_GNUBG)
_NO_GNUBG = f"GNUbg binary not found ({_GNUBG}); see neongammon.settings"


def flip_board(checkers):
    """Flip a board to the other player's perspective."""
    flipped = [0] * 26
    flipped[0] = checkers[25]
    flipped[25] = checkers[0]
    for i in range(1, 25):
        flipped[i] = -checkers[25 - i]
    return flipped


@unittest.skipUnless(_GNUBG_FOUND, _NO_GNUBG)
class TestGnubgEndgame(unittest.TestCase):
    """Test simple endgame positions where the answer is known exactly.

    post_move_analytics takes a post-move board from the MOVER's perspective.
    """

    def test_both_on_1point_player_wins(self):
        """Both players have 2 checkers on their own 1-point.
        Player just moved and it's their post-move position. Player is the mover
        and will bear off first on their next turn -> P(win) should be very high.

        But this is a post-move board: the opponent will roll next. Since both
        have 2 on their 1-point, the opponent will also bear off in one roll.
        Who wins depends on who rolls first — and that's the opponent.

        So: opponent bears off next, then player bears off. The opponent wins!
        P(win) for the mover should be 0 here. Let's use a game-over case instead.
        """
        # Mover just bore off their last checker. Post-move: mover has 0 checkers.
        # Opponent still has 2 on point 24 (their 1-point).
        checkers = [0] * 26
        checkers[24] = -2  # opponent: 2 on point 24

        result = post_move_analytics(checkers, n_plies=0)

        self.assertAlmostEqual(result['probs'][0], 1.0, places=3,
                               msg="P(win) should be 1.0 (game over, mover won)")
        self.assertAlmostEqual(result['probs'][1], 0.0, places=3,
                               msg="P(gw) should be 0.0 (not a gammon)")
        self.assertAlmostEqual(result['equity'], 1.0, places=3)

    def test_gammon_certain(self):
        """Mover just bore off their last checker. Opponent has all 15 still on board.
        P(win)=1, P(gw)=1, P(bw)=0 (opponent not in mover's home or bar).
        """
        checkers = [0] * 26
        checkers[24] = -15  # opponent: 15 on point 24 (their 1-point)

        result = post_move_analytics(checkers, n_plies=0)

        self.assertAlmostEqual(result['probs'][0], 1.0, places=3)
        self.assertAlmostEqual(result['probs'][1], 1.0, places=3,
                               msg="P(gw) should be 1.0 (gammon)")
        self.assertAlmostEqual(result['probs'][2], 0.0, places=3,
                               msg="P(bw) should be 0.0 (not backgammon)")
        self.assertAlmostEqual(result['equity'], 2.0, places=3)

    def test_backgammon_certain(self):
        """Mover bore off all. Opponent has 15 on bar -> backgammon."""
        checkers = [0] * 26
        checkers[0] = 15    # opponent: 15 on bar

        result = post_move_analytics(checkers, n_plies=0)

        self.assertAlmostEqual(result['probs'][0], 1.0, places=3)
        self.assertAlmostEqual(result['probs'][1], 1.0, places=3)
        self.assertAlmostEqual(result['probs'][2], 1.0, places=3,
                               msg="P(bw) should be 1.0 (backgammon)")
        self.assertAlmostEqual(result['equity'], 3.0, places=3)

    def test_non_terminal_race(self):
        """Non-terminal race position: mover has 2 on 1-pt, opponent has 2 on their
        1-pt (our pt 24). Post-move, opponent rolls next. Opponent will bear off in
        1 roll, then mover bears off in 1 roll. Both bear off -> opponent wins.
        P(win) for mover should be 0.0.
        """
        checkers = [0] * 26
        checkers[1] = 2     # mover: 2 on point 1
        checkers[24] = -2   # opponent: 2 on their 1-point

        result = post_move_analytics(checkers, n_plies=0)

        # Opponent rolls next and bears off. Then mover rolls and bears off.
        # Opponent wins -> mover P(win) = 0
        self.assertAlmostEqual(result['probs'][0], 0.0, places=3,
                               msg="P(win) should be 0.0 (opponent bears off first)")
        self.assertAlmostEqual(result['equity'], -1.0, places=3)

    def test_near_race(self):
        """Mover has 2 on 1-pt. Opponent has 2 on their 2-pt (our pt 23).
        Post-move, opponent rolls next. Opponent needs specific rolls to bear off
        both in one turn. Should be close to P(win)=0 but not exactly 0.
        """
        checkers = [0] * 26
        checkers[1] = 2     # mover: 2 on point 1
        checkers[23] = -2   # opponent: 2 on their 2-point (our pt 23)

        result = post_move_analytics(checkers, n_plies=0)

        # Opponent needs doubles or a roll including a 2 to bear off both.
        # Most rolls bear off both. P(win) for mover should be small.
        self.assertLess(result['probs'][0], 0.5,
                        msg="P(win) should be < 0.5 (opponent almost certain to bear off)")
        # Verify equity is internally consistent
        eq = (2.0 * result['probs'][0] - 1.0 + result['probs'][1] - result['probs'][3]
              + result['probs'][2] - result['probs'][4])
        self.assertAlmostEqual(result['equity'], eq, places=3)


@unittest.skipUnless(_GNUBG_FOUND, _NO_GNUBG)
class TestGnubgVsStage4(unittest.TestCase):
    """Compare gnubg 2-ply with Stage 4 2-ply on top-100 worst 0-ply positions."""

    @classmethod
    def setUpClass(cls):
        """Load models and find top-100 positions."""
        import bgbot_cpp
        from neongammon.data import load_benchmark_file

        cls.bgbot_cpp = bgbot_cpp

        DATA_DIR = os.path.join(project_dir, 'data')
        MODELS_DIR = os.path.join(project_dir, 'models')

        # The GNUbg benchmark files are gitignored and not shipped.
        contact_file = os.path.join(DATA_DIR, 'contact.bm')
        crashed_file = os.path.join(DATA_DIR, 'crashed.bm')
        missing = [f for f in (contact_file, crashed_file) if not os.path.isfile(f)]
        if missing:
            raise unittest.SkipTest(
                "GNUbg benchmark files not present (gitignored): "
                + ", ".join(os.path.relpath(f, project_dir) for f in missing))

        cls.NH = {'purerace': 120, 'racing': 250, 'attacking': 250,
                  'priming': 250, 'anchoring': 250}

        # Load Stage 4 weights (fall back to Stage 3 .best files)
        cls.weights = {}
        for t in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
            for prefix in ['sl_s4_', 'sl_']:
                path = os.path.join(MODELS_DIR, f'{prefix}{t}.weights.best')
                if os.path.exists(path):
                    cls.weights[t] = path
                    break
            else:
                raise FileNotFoundError(f"No weights found for {t}")

        # Load contact + crashed benchmarks
        print("Loading benchmarks...")
        scenarios_contact = load_benchmark_file(contact_file)
        scenarios_crashed = load_benchmark_file(crashed_file)

        # Score at 0-ply to find worst positions
        print("Scoring at 0-ply to find top-100 worst...")
        w = cls.weights
        nh = cls.NH
        errors_contact = bgbot_cpp.score_benchmarks_per_scenario_5nn(
            scenarios_contact,
            w['purerace'], w['racing'], w['attacking'], w['priming'], w['anchoring'],
            nh['purerace'], nh['racing'], nh['attacking'], nh['priming'], nh['anchoring'])
        errors_crashed = bgbot_cpp.score_benchmarks_per_scenario_5nn(
            scenarios_crashed,
            w['purerace'], w['racing'], w['attacking'], w['priming'], w['anchoring'],
            nh['purerace'], nh['racing'], nh['attacking'], nh['priming'], nh['anchoring'])

        # Combine and sort to find top 100
        all_errors = []
        for i, err in enumerate(errors_contact):
            all_errors.append((err, 'contact', i))
        for i, err in enumerate(errors_crashed):
            all_errors.append((err, 'crashed', i))
        all_errors.sort(key=lambda x: -x[0])

        top_100 = all_errors[:100]

        # Extract the actual checker positions for these scenarios
        cls.test_positions = []
        cls.position_sources = []

        contact_indices = {e[2] for e in top_100 if e[1] == 'contact'}
        crashed_indices = {e[2] for e in top_100 if e[1] == 'crashed'}

        from neongammon.data import board_from_gnubg_position_string

        def read_bm_positions(filepath, indices):
            """Read specific positions from a .bm file."""
            positions = {}
            with open(filepath, 'r') as f:
                move_idx = 0
                for line in f:
                    if not line.startswith('m '):
                        continue
                    if move_idx in indices:
                        bits = line.split()
                        start_board = board_from_gnubg_position_string(bits[1])
                        die1 = int(bits[2])
                        die2 = int(bits[3])
                        # The first ranked board is the best move result (flipped post-move)
                        best_board_flipped = board_from_gnubg_position_string(bits[4])
                        positions[move_idx] = {
                            'start_board': start_board,
                            'die1': die1,
                            'die2': die2,
                            'best_board_flipped': best_board_flipped,
                        }
                    move_idx += 1
            return positions

        contact_positions = read_bm_positions(contact_file, contact_indices)
        crashed_positions = read_bm_positions(crashed_file, crashed_indices)

        for err, source, idx in top_100:
            if source == 'contact':
                pos = contact_positions[idx]
            else:
                pos = crashed_positions[idx]
            cls.test_positions.append(pos)
            cls.position_sources.append((source, idx, err))

        print(f"  Found {len(cls.test_positions)} test positions")

        # Create Stage 4 2-ply strategy
        cls.multipy_2 = bgbot_cpp.create_multipy_5nn(
            w['purerace'], w['racing'], w['attacking'], w['priming'], w['anchoring'],
            nh['purerace'], nh['racing'], nh['attacking'], nh['priming'], nh['anchoring'],
            n_plies=2)

    def test_top100_comparison(self):
        """Compare gnubg 2-ply vs Stage 4 2-ply on top-100 positions.

        The .bm best_board_flipped is the flipped post-move board (opponent's
        perspective). We un-flip it to get the post-move board from the mover's
        perspective, then pass that to both gnubg and our NN.

        Both gnubg (via post_move_analytics) and our NN (via evaluate_board)
        return probabilities from the mover's perspective.
        """
        bgbot_cpp = self.bgbot_cpp

        gnubg_results = []
        s4_results = []
        game_plans = []

        # Un-flip all boards to get post-move positions from mover's perspective
        post_move_boards = [flip_board(pos['best_board_flipped'])
                            for pos in self.test_positions]

        # gnubg: evaluate post-move boards at 2-ply
        print("\nEvaluating 100 positions with gnubg 2-ply...")
        t0 = time.perf_counter()
        gnubg_results = post_move_analytics_many(post_move_boards, n_plies=2,
                                                  max_workers=8, timeout=120)
        t_gnubg = time.perf_counter() - t0
        print(f"  gnubg 2-ply: {t_gnubg:.1f}s ({t_gnubg/len(post_move_boards)*1000:.0f}ms per position)")

        # Stage 4: evaluate post-move boards with start_board for context
        print("Evaluating 100 positions with Stage 4 2-ply...")
        t0 = time.perf_counter()

        for i, pos in enumerate(self.test_positions):
            self.multipy_2.clear_cache()
            r = self.multipy_2.evaluate_board(post_move_boards[i], pos['start_board'])
            s4_results.append({'probs': list(r['probs']), 'equity': r['equity']})

            gp = bgbot_cpp.classify_game_plan(pos['start_board'])
            game_plans.append(gp)

        t_s4 = time.perf_counter() - t0
        print(f"  Stage 4 2-ply: {t_s4:.1f}s ({t_s4/len(post_move_boards)*1000:.0f}ms per position)")

        # Compare probabilities
        prob_names = ['P(win)', 'P(gw)', 'P(bw)', 'P(gl)', 'P(bl)']

        diffs = {i: [] for i in range(5)}
        eq_diffs = []
        gp_diffs = {}
        gp_eq_diffs = {}
        big_diff_count = {i: 0 for i in range(5)}
        big_diff_positions = []

        for idx in range(len(self.test_positions)):
            gnubg_probs = gnubg_results[idx]['probs']
            s4_probs = s4_results[idx]['probs']
            gnubg_eq = gnubg_results[idx]['equity']
            s4_eq = s4_results[idx]['equity']

            gp_name = game_plans[idx]

            if gp_name not in gp_diffs:
                gp_diffs[gp_name] = {i: [] for i in range(5)}
                gp_eq_diffs[gp_name] = []

            eq_diff = s4_eq - gnubg_eq
            eq_diffs.append(eq_diff)
            gp_eq_diffs[gp_name].append(eq_diff)

            has_big = False
            for i in range(5):
                d = s4_probs[i] - gnubg_probs[i]
                diffs[i].append(d)
                gp_diffs[gp_name][i].append(d)
                if abs(d) > 0.02:
                    big_diff_count[i] += 1
                    has_big = True

            if has_big:
                big_diff_positions.append((idx, gnubg_probs, s4_probs, gp_name,
                                           gnubg_eq, s4_eq))

        # Print report
        print("\n" + "=" * 80)
        print("COMPARISON: gnubg 2-ply vs Stage 4 2-ply (top 100 worst 0-ply positions)")
        print("=" * 80)

        print(f"\n{'Metric':<12} {'RMS':>8} {'Mean':>8} {'MaxAbs':>8} {'|d|>0.02':>10}")
        print("-" * 50)
        for i in range(5):
            rms = math.sqrt(sum(d**2 for d in diffs[i]) / len(diffs[i]))
            mean = sum(diffs[i]) / len(diffs[i])
            maxabs = max(abs(d) for d in diffs[i])
            print(f"{prob_names[i]:<12} {rms:>8.4f} {mean:>+8.4f} {maxabs:>8.4f} {big_diff_count[i]:>8}/{len(diffs[i])}")

        rms_eq = math.sqrt(sum(d**2 for d in eq_diffs) / len(eq_diffs))
        mean_eq = sum(eq_diffs) / len(eq_diffs)
        maxabs_eq = max(abs(d) for d in eq_diffs)
        big_eq_count = sum(1 for d in eq_diffs if abs(d) > 0.02)
        print(f"{'Equity':<12} {rms_eq:>8.4f} {mean_eq:>+8.4f} {maxabs_eq:>8.4f} {big_eq_count:>8}/{len(eq_diffs)}")

        # Per-gameplan breakdown
        print(f"\nPer-Game-Plan RMS Differences:")
        gp_header = f"{'GamePlan':<12} {'N':>4} {'P(win)':>8} {'P(gw)':>8} {'P(bw)':>8} {'P(gl)':>8} {'P(bl)':>8} {'Equity':>8}"
        print(gp_header)
        print("-" * len(gp_header))

        for gp_name in sorted(gp_diffs.keys()):
            n = len(gp_diffs[gp_name][0])
            if n == 0:
                continue
            rms_vals = []
            for i in range(5):
                rms = math.sqrt(sum(d**2 for d in gp_diffs[gp_name][i]) / n)
                rms_vals.append(rms)
            rms_eq_gp = math.sqrt(sum(d**2 for d in gp_eq_diffs[gp_name]) / n)
            print(f"{gp_name:<12} {n:>4} {rms_vals[0]:>8.4f} {rms_vals[1]:>8.4f} "
                  f"{rms_vals[2]:>8.4f} {rms_vals[3]:>8.4f} {rms_vals[4]:>8.4f} {rms_eq_gp:>8.4f}")

        # Show positions with biggest differences
        if big_diff_positions:
            print(f"\nPositions with |prob diff| > 0.02 ({len(big_diff_positions)} positions):")
            print(f"{'#':>4} {'GP':<12} {'Prob':>8} {'gnubg':>8} {'S4':>8} {'Diff':>8}")
            print("-" * 55)
            big_diff_positions.sort(
                key=lambda x: max(abs(x[1][i] - x[2][i]) for i in range(5)),
                reverse=True)
            for idx, gp, sp, gp_name, geq, seq in big_diff_positions[:20]:
                worst_i = max(range(5), key=lambda i: abs(gp[i] - sp[i]))
                d = sp[worst_i] - gp[worst_i]
                print(f"{idx:>4} {gp_name:<12} {prob_names[worst_i]:>8} "
                      f"{gp[worst_i]:>8.4f} {sp[worst_i]:>8.4f} {d:>+8.4f}")

        # Investigate really large differences (> 0.10)
        very_big = [(idx, gp, sp, gpn, geq, seq) for idx, gp, sp, gpn, geq, seq
                    in big_diff_positions if max(abs(gp[i] - sp[i]) for i in range(5)) > 0.10]
        if very_big:
            print(f"\n*** {len(very_big)} positions with |diff| > 0.10 ***")
            for idx, gp, sp, gp_name, geq, seq in very_big[:5]:
                pos = self.test_positions[idx]
                source, bm_idx, err_0ply = self.position_sources[idx]
                print(f"\n  Position #{idx} (from {source}[{bm_idx}], 0-ply err={err_0ply*1000:.1f}):")
                print(f"    Start board:      {pos['start_board']}")
                print(f"    Post-move board:  {flip_board(pos['best_board_flipped'])}")
                print(f"    Dice: {pos['die1']},{pos['die2']}")
                print(f"    Game plan: {gp_name}")
                print(f"    gnubg probs: {[f'{p:.4f}' for p in gp]}")
                print(f"    S4    probs: {[f'{p:.4f}' for p in sp]}")
                print(f"    gnubg eq: {geq:+.4f}, S4 eq: {seq:+.4f}, diff: {seq-geq:+.4f}")

        # Summary stats
        print(f"\nSummary:")
        print(f"  Total positions compared: {len(self.test_positions)}")
        print(f"  gnubg time: {t_gnubg:.1f}s, Stage 4 time: {t_s4:.1f}s")
        print(f"  Equity RMS diff: {rms_eq:.4f}")
        print(f"  P(win) RMS diff: {math.sqrt(sum(d**2 for d in diffs[0]) / len(diffs[0])):.4f}")
        total_big = sum(big_diff_count.values())
        print(f"  Total prob diffs > 0.02: {total_big} / {len(self.test_positions) * 5}")

        if very_big:
            print(f"\n  NOTE: {len(very_big)} positions have probability differences > 0.10")
            print(f"        These are the hardest positions (top-100 worst 0-ply errors)")
            print(f"        and large differences are expected on these challenging positions.")


# ---------------------------------------------------------------------------
# Shared test fixtures (boards and real GNUbg 1.08.003 captured outputs)
# ---------------------------------------------------------------------------

_STARTING_BOARD = [
    0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5,
    5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0,
]

# Race position (player trailing; flipped = player leading strongly).
_RACE_BOARD = [
    0, 0, 0, 0, 4, 3, 4, 2, 2, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, -5, -4, -3, -2, -1, 0, 0,
]

# Contact / blitz position with substantial gammon chances.
_BLITZ_BOARD = [
    2, 2, 2, 2, 3, 2, 2, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, -3, -3, -3, -2, -2, 0, 0,
]

# Player on bar with only point 24 open, points 18-23 made by opponent,
# remaining 14 checkers on point 1. Rolling 1-6 yields exactly 1 legal move
# (bar/24; the 6 cannot be played).
_SINGLE_MOVE_BOARD = [
    0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, -5, -2, -2, -2, -2, -2, 0, 1,
]

# Player on bar with all 6 points 19-24 closed -> 0 legal moves on any roll.
_FORCED_PASS_BOARD = [
    0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, -3, -2, -2, -2, -2, -2, -2, 1,
]

# Real captured GNUbg 1.08.003 outputs for binary-free parser unit tests.
_EVAL_MONEY_CENTERED_0PLY = """\
`user' is now on roll.
Position ID:\t2r0PAAB49zYAAA
Match ID:\tcAkAAAAAAAAA

Evaluator: \tRace


        Win     W(g)    W(bg)   L(g)    L(bg)   Equity    Cubeful
static: 0.253   0.000   0.000   0.000   0.000   -0.494    -0.757



Cube analysis
0-ply cubeless equity -0.494
  0.253 0.000 0.000 - 0.747 0.000 0.000
Cubeful equities:
1. No double           -0.757
2. Double, pass        +1.000  (+1.757)
3. Double, take        -1.585  (-0.828)
Proper cube action: No double, beaver (32.0%)
"""

_EVAL_MONEY_OPP_1PLY = """\
`user' is now on roll.
Position ID:\t2r0PAAB49zYAAA
Match ID:\tQgkAAAAAAAAE

Evaluator: \tRace


        Win     W(g)    W(bg)   L(g)    L(bg)   Equity    Cubeful
static: 0.253   0.000   0.000   0.000   0.000   -0.494    -0.793
 1 ply: 0.255   0.000   0.000   0.000   0.000   -0.491    -0.816
"""

_EVAL_MATCH_2A5A_BLITZ_0PLY = """\
`user' is now on roll.
Position ID:\ttrsDAGDbbgMAAA
Match ID:\tcAmgAAAAGAAE

Evaluator: \tContact


        Win     W(g)    W(bg)   L(g)    L(bg)   Equity    Cubeful
static: 0.944   0.486   0.011   0.000   0.000   +1.777    +1.731



Cube analysis
0-ply cubeless equity +1.777 (Money: +1.386)
  0.944 0.486 0.011 - 0.056 0.000 0.000
Cubeful equities:
1. No double           +1.731
2. Double, take        +2.436  (+0.705)
3. Double, pass        +1.000  (-0.731)
Proper cube action: Too good to double, pass (50.9%)
"""

_EVAL_MATCH_5A2A_PLAYER_REDOUBLE_1PLY = """\
`user' is now on roll.
Position ID:\t2r0PAAB49zYAAA
Match ID:\tUQmgADAAAAAE

Evaluator: \tRace


        Win     W(g)    W(bg)   L(g)    L(bg)   Equity    Cubeful
static: 0.253   0.000   0.000   0.000   0.000   -0.494    -0.214
 1 ply: 0.254   0.000   0.000   0.000   0.000   -0.492    -0.223



Cube analysis
1-ply cubeless equity -0.492 (Money: -0.492)
  0.254 0.000 0.000 - 0.746 0.000 0.000
Cubeful equities:
1. Double, take        -0.140
2. Double, pass        +1.000  (+1.140)
3. No double           -0.223  (-0.083)
Proper cube action: Redouble, take
"""

_EVAL_MATCH_CRAWFORD_1A3A = """\
`user' is now on roll.
Position ID:\ttrsDAGDbbgMAAA
Match ID:\t8AlgAAAAEAAE

Evaluator: \tContact


        Win     W(g)    W(bg)   L(g)    L(bg)   Equity    Cubeful
static: 0.944   0.486   0.011   0.000   0.000   +0.889    +0.889
"""

_EVAL_BEAROFF2_2PLY = """\
`user' is now on roll.
Position ID:\tWwAAwBYAAAAAAA
Match ID:\tcAkAAAAAAAAA

Evaluator: \tBearoff2

             Player       Opponent
Position          232           232

Effective pip count:
\tPlayer\tOpponent
EPC\t 22.137\t 22.137

        Win     W(g)    W(bg)   L(g)    L(bg)   Equity    Cubeful
static: 0.794   0.000   0.000   0.000   0.000   +0.589    +0.769
 1 ply: 0.794   0.000   0.000   0.000   0.000   +0.589    +0.769
 2 ply: 0.794   0.000   0.000   0.000   0.000   +0.589    +0.769



Cube analysis
2-ply cubeless equity +0.589
  0.794 0.000 0.000 - 0.206 0.000 0.000
Cubeful equities:
1. Double, pass        +1.000
2. Double, take        +1.162  (+0.162)
3. No double           +0.769  (-0.231)
Proper cube action: Double, pass
"""

_EVAL_BATCH_TWO_BOARDS = """\
 GNU Backgammon  Position ID: 2r0PAAB49zYAAA
                 Match ID   : QQmgADAAAAAE
`user' is now on roll.
Position ID:\t2r0PAAB49zYAAA
Match ID:\tQQmgADAAAAAE

Evaluator: \tRace


        Win     W(g)    W(bg)   L(g)    L(bg)   Equity    Cubeful
static: 0.253   0.000   0.000   0.000   0.000   -0.494    -0.494

 GNU Backgammon  Position ID: 4HPwATDgc/ABMA
                 Match ID   : QQmgADAAAAAE
`user' is now on roll.
Position ID:\t4HPwATDgc/ABMA
Match ID:\tQQmgADAAAAAE

Evaluator: \tContact


        Win     W(g)    W(bg)   L(g)    L(bg)   Equity    Cubeful
static: 0.521   0.154   0.008   0.130   0.007   +0.268    +0.268
"""


class TestGnubgCommandGeneration(unittest.TestCase):
    """Binary-free unit tests for GNUbg command construction & perspective flips."""

    def test_flip_cube_owner(self):
        self.assertEqual(_flip_cube_owner("player"), "opponent")
        self.assertEqual(_flip_cube_owner("opponent"), "player")
        self.assertEqual(_flip_cube_owner("centered"), "centered")
        with self.assertRaises(ValueError):
            _flip_cube_owner("invalid")

    def test_flip_board_involution(self):
        self.assertEqual(_flip_board(_flip_board(_RACE_BOARD)), _RACE_BOARD)
        self.assertEqual(_flip_board(_flip_board(_BLITZ_BOARD)), _BLITZ_BOARD)
        self.assertEqual(_flip_board(_flip_board(_SINGLE_MOVE_BOARD)), _SINGLE_MOVE_BOARD)

    def test_match_prefix_variants(self):
        # Money game
        self.assertEqual(GnuBgAnalyzer._match_prefix(0, 0, False), "new session\n")

        # Asymmetric match: player 2-away (score 3), opp 5-away (score 0)
        self.assertEqual(
            GnuBgAnalyzer._match_prefix(2, 5, False),
            "new match 5\nset score 0 3\n",
        )
        # Reversed: player 5-away (score 0), opp 2-away (score 3)
        self.assertEqual(
            GnuBgAnalyzer._match_prefix(5, 2, False),
            "new match 5\nset score 3 0\n",
        )

        # Crawford: 1-away vs 3-away -> NO set postcrawford on
        craw = GnuBgAnalyzer._match_prefix(1, 3, True)
        self.assertEqual(craw, "new match 3\nset score 0 2\n")
        self.assertNotIn("postcrawford", craw)

        # Post-Crawford: 1-away vs 3-away and 3-away vs 1-away -> set postcrawford on
        self.assertEqual(
            GnuBgAnalyzer._match_prefix(1, 3, False),
            "new match 3\nset score 0 2\nset postcrawford on\n",
        )
        self.assertEqual(
            GnuBgAnalyzer._match_prefix(3, 1, False),
            "new match 3\nset score 2 0\nset postcrawford on\n",
        )

        # DMP: 1-away vs 1-away
        self.assertIn("new match 1\nset score 0 0\n", GnuBgAnalyzer._match_prefix(1, 1, False))

    def test_preroll_command_fields(self):
        analyzer = GnuBgAnalyzer(eval_level="2ply")
        cmd = analyzer._build_preroll_command(
            _RACE_BOARD, cube_value=2, cube_owner="opponent",
            away1=2, away2=5, is_crawford=False, jacoby=False,
        )
        self.assertIn("new match 5\n", cmd)
        self.assertIn("set score 0 3\n", cmd)
        self.assertIn("set jacoby off\n", cmd)
        self.assertIn("set evaluation chequer eval plies 1\n", cmd)
        self.assertIn("set evaluation cubedecision eval plies 1\n", cmd)
        self.assertIn("set cube value 2\n", cmd)
        self.assertIn("set cube owner 0\n", cmd)
        self.assertIn(f"set board simple {_board_simple_str(_RACE_BOARD)}\n", cmd)
        self.assertTrue(cmd.endswith("set turn 1\neval\n"))
        self.assertNotIn("\nhint\n", cmd)

    def test_preroll_command_centered_and_player_cube(self):
        analyzer = GnuBgAnalyzer(eval_level="1ply")
        cmd_c = analyzer._build_preroll_command(_RACE_BOARD, cube_value=1, cube_owner="centered")
        self.assertIn("set cube centre\n", cmd_c)
        self.assertIn("set jacoby on\n", cmd_c)

        cmd_p = analyzer._build_preroll_command(_RACE_BOARD, cube_value=4, cube_owner="player", jacoby=False)
        self.assertIn("set cube value 4\n", cmd_p)
        self.assertIn("set cube owner 1\n", cmd_p)
        self.assertIn("set jacoby off\n", cmd_p)

        with self.assertRaises(ValueError):
            analyzer._build_preroll_command(_RACE_BOARD, cube_owner="bogus")

    def test_batch_postmove_flips_board_owner_and_scores(self):
        analyzer = GnuBgAnalyzer(eval_level="1ply")
        # Mover is 2-away vs opp 5-away, mover owns 2-cube ("player").
        # After mover's turn, opponent is on roll:
        #   - opponent is 5-away vs mover 2-away -> set score 3 0
        #   - cube belongs to mover (opponent's opponent = player 0) -> set cube owner 0
        #   - board is flipped to opponent's POV
        cmd = analyzer._build_batch_postmove_command(
            [_RACE_BOARD, _BLITZ_BOARD],
            cube_value=2, cube_owner="player",
            away1=2, away2=5, is_crawford=False, jacoby=False,
        )
        self.assertIn("new match 5\nset score 3 0\n", cmd)
        self.assertIn("set cube value 2\nset cube owner 0\n", cmd)
        self.assertIn(f"set board simple {_board_simple_str(_flip_board(_RACE_BOARD))}\n", cmd)
        self.assertIn(f"set board simple {_board_simple_str(_flip_board(_BLITZ_BOARD))}\n", cmd)
        self.assertEqual(cmd.count("\neval\n"), 2)
        self.assertNotIn("\nhint\n", cmd)

        # Reverse: mover faces opponent-owned cube at 5-away vs 2-away ->
        # opponent on roll is 2-away vs 5-away (set score 0 3) and owns cube (set cube owner 1).
        cmd_opp = analyzer._build_batch_postmove_command(
            [_RACE_BOARD],
            cube_value=2, cube_owner="opponent",
            away1=5, away2=2, is_crawford=False, jacoby=False,
        )
        self.assertIn("new match 5\nset score 0 3\n", cmd_opp)
        self.assertIn("set cube value 2\nset cube owner 1\n", cmd_opp)

    def test_post_move_analytics_emits_exact_flipped_preroll_command(self):
        """post_move_analytics(board, owner, a1, a2) must send the exact same
        GNUbg command as cube_action(flip(board), flip(owner), a2, a1)."""
        from unittest.mock import patch
        import neongammon.gnubg as gnubg_mod

        analyzer = GnuBgAnalyzer(eval_level="1ply")
        captured = []

        def fake_run(cmd, timeout=60, gnubg_path=None):
            captured.append(cmd)
            return _EVAL_MATCH_CRAWFORD_1A3A

        with patch.object(gnubg_mod, "_run_gnubg", side_effect=fake_run):
            for owner in ("centered", "player", "opponent"):
                for a1, a2, craw in [(0, 0, False), (2, 5, False), (1, 3, True), (1, 3, False)]:
                    captured.clear()
                    analyzer.post_move_analytics(
                        _RACE_BOARD, cube_owner=owner, cube_value=2,
                        away1=a1, away2=a2, is_crawford=craw,
                    )
                    analyzer.cube_action(
                        _flip_board(_RACE_BOARD),
                        cube_owner=_flip_cube_owner(owner), cube_value=2,
                        away1=a2, away2=a1, is_crawford=craw,
                    )
                    self.assertEqual(
                        captured[0], captured[1],
                        f"Command mismatch for owner={owner}, away=({a1},{a2}), craw={craw}",
                    )

    def test_checker_play_raises_on_truncated_eval_output(self):
        """checker_play must raise instead of silently dropping moves when
        GNUbg output contains fewer eval sections than candidates."""
        from unittest.mock import patch
        import neongammon.gnubg as gnubg_mod

        analyzer = GnuBgAnalyzer(eval_level="1ply")
        with patch.object(gnubg_mod, "_run_gnubg", return_value=_EVAL_MONEY_OPP_1PLY):
            with self.assertRaises(ValueError):
                analyzer.checker_play(_STARTING_BOARD, 3, 1, cube_value=2, cube_owner="opponent")


class TestGnubgParsers(unittest.TestCase):
    """Binary-free parser unit tests on real captured GNUbg eval/hint outputs."""

    def test_parse_eval_money_centered_with_cube_analysis(self):
        r = _parse_eval_single(_EVAL_MONEY_CENTERED_0PLY, n_plies=0)
        self.assertAlmostEqual(r["p_win"], 0.253, places=4)
        self.assertAlmostEqual(r["equity_cubeless"], -0.494, places=4)
        self.assertAlmostEqual(r["equity_nd"], -0.757, places=4)
        self.assertAlmostEqual(r["equity_dt"], -1.585, places=4)
        self.assertAlmostEqual(r["equity_dp"], 1.000, places=4)
        self.assertTrue(r["can_double"])
        self.assertAlmostEqual(r["optimal_equity"], -0.757, places=4)
        self.assertIn("No double", r["optimal_action"])

    def test_parse_eval_money_opponent_cube_1ply_no_cube_analysis(self):
        r0 = _parse_eval_single(_EVAL_MONEY_OPP_1PLY, n_plies=0)
        self.assertAlmostEqual(r0["p_win"], 0.253, places=4)
        self.assertAlmostEqual(r0["equity_cubeless"], -0.494, places=4)
        self.assertAlmostEqual(r0["equity_nd"], -0.793, places=4)
        self.assertFalse(r0["can_double"])
        self.assertIsNone(r0["equity_dt"])
        self.assertAlmostEqual(r0["optimal_equity"], -0.793, places=4)

        r1 = _parse_eval_single(_EVAL_MONEY_OPP_1PLY, n_plies=1)
        self.assertAlmostEqual(r1["p_win"], 0.255, places=4)
        self.assertAlmostEqual(r1["equity_cubeless"], -0.491, places=4)
        self.assertAlmostEqual(r1["equity_nd"], -0.816, places=4)
        self.assertFalse(r1["can_double"])
        self.assertAlmostEqual(r1["optimal_equity"], -0.816, places=4)

    def test_parse_eval_match_blitz_too_good(self):
        r = _parse_eval_single(_EVAL_MATCH_2A5A_BLITZ_0PLY, n_plies=0)
        self.assertAlmostEqual(r["p_win"], 0.944, places=4)
        self.assertAlmostEqual(r["p_gw"], 0.486, places=4)
        self.assertAlmostEqual(r["p_bw"], 0.011, places=4)
        self.assertAlmostEqual(r["equity_cubeless"], 1.777, places=4)
        self.assertAlmostEqual(r["equity_nd"], 1.731, places=4)
        self.assertAlmostEqual(r["equity_dt"], 2.436, places=4)
        self.assertAlmostEqual(r["equity_dp"], 1.000, places=4)
        self.assertTrue(r["can_double"])
        # Too good to double: optimal is ND (+1.731), NOT DP (+1.000)
        self.assertAlmostEqual(r["optimal_equity"], 1.731, places=4)

    def test_parse_eval_match_redouble_take_1ply(self):
        r = _parse_eval_single(_EVAL_MATCH_5A2A_PLAYER_REDOUBLE_1PLY, n_plies=1)
        self.assertAlmostEqual(r["p_win"], 0.254, places=4)
        self.assertAlmostEqual(r["equity_cubeless"], -0.492, places=4)
        self.assertAlmostEqual(r["equity_nd"], -0.223, places=4)
        self.assertAlmostEqual(r["equity_dt"], -0.140, places=4)
        self.assertAlmostEqual(r["equity_dp"], 1.000, places=4)
        self.assertTrue(r["can_double"])
        # Redouble/take is better than ND -> optimal_equity is DT (-0.140)
        self.assertAlmostEqual(r["optimal_equity"], -0.140, places=4)

    def test_parse_eval_match_crawford_no_cube_analysis(self):
        r = _parse_eval_single(_EVAL_MATCH_CRAWFORD_1A3A, n_plies=0)
        self.assertAlmostEqual(r["p_win"], 0.944, places=4)
        self.assertAlmostEqual(r["equity_cubeless"], 0.889, places=4)
        self.assertAlmostEqual(r["equity_nd"], 0.889, places=4)
        self.assertAlmostEqual(r["optimal_equity"], 0.889, places=4)
        self.assertFalse(r["can_double"])

    def test_parse_eval_bearoff2_double_pass(self):
        r = _parse_eval_single(_EVAL_BEAROFF2_2PLY, n_plies=2)
        self.assertAlmostEqual(r["p_win"], 0.794, places=4)
        self.assertAlmostEqual(r["equity_cubeless"], 0.589, places=4)
        self.assertAlmostEqual(r["equity_nd"], 0.769, places=4)
        self.assertAlmostEqual(r["equity_dt"], 1.162, places=4)
        self.assertAlmostEqual(r["equity_dp"], 1.000, places=4)
        self.assertAlmostEqual(r["optimal_equity"], 1.000, places=4)

    def test_parse_eval_batch_ignores_ascii_board_headers(self):
        results = _parse_eval_batch(_EVAL_BATCH_TWO_BOARDS, n_plies=0)
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0]["p_win"], 0.253, places=4)
        self.assertAlmostEqual(results[0]["equity_nd"], -0.494, places=4)
        self.assertAlmostEqual(results[1]["p_win"], 0.521, places=4)
        self.assertAlmostEqual(results[1]["p_gw"], 0.154, places=4)
        self.assertAlmostEqual(results[1]["p_gl"], 0.130, places=4)
        self.assertAlmostEqual(results[1]["equity_nd"], 0.268, places=4)

    def test_parse_legacy_hint_parsers_still_work(self):
        r = _parse_cube_analytics(_EVAL_MONEY_CENTERED_0PLY, n_plies=0)
        self.assertAlmostEqual(r["p_win"], 0.253, places=4)
        self.assertAlmostEqual(r["equity_cubeless"], -0.494, places=4)

        full, _ = _parse_cube_analytics_full(_EVAL_MATCH_2A5A_BLITZ_0PLY)
        self.assertAlmostEqual(full["equity_cubeless"], 1.777, places=4)
        self.assertAlmostEqual(full["equity_nd"], 1.731, places=4)

    def test_parse_eval_errors_on_empty_or_malformed(self):
        with self.assertRaises(ValueError):
            _parse_eval_single("no evaluation here", n_plies=0)
        with self.assertRaises(ValueError):
            _parse_eval_section("Evaluator:\tRace\n(no rows)", n_plies=0)

    def test_cube_action_no_double_returns_clean_result(self):
        """Mocked cube_action when opponent owns cube returns should_double=False
        without crashing."""
        from unittest.mock import patch
        import neongammon.gnubg as gnubg_mod

        analyzer = GnuBgAnalyzer(eval_level="2ply")
        with patch.object(gnubg_mod, "_run_gnubg", return_value=_EVAL_MONEY_OPP_1PLY):
            res = analyzer.cube_action(_RACE_BOARD, cube_value=4, cube_owner="opponent")
        self.assertFalse(res.should_double)
        self.assertEqual(res.optimal_action, "No Double")
        self.assertAlmostEqual(res.equity_nd, -0.816, places=4)
        self.assertAlmostEqual(res.optimal_equity, -0.816, places=4)
        self.assertAlmostEqual(res.cubeless_equity, -0.491, places=4)
        self.assertAlmostEqual(res.probs.win, 0.255, places=4)


@unittest.skipUnless(_GNUBG_FOUND, _NO_GNUBG)
class TestGnuBgAnalyzerLive(unittest.TestCase):
    """Live GNUbg integration tests covering score/owner symmetry, opponent
    cube ownership, Crawford/post-Crawford/DMP, and BgBotAnalyzer agreement."""

    def test_post_move_analytics_score_and_owner_symmetry(self):
        """post_move_analytics(board, owner, a1, a2) must match the negated
        pre-roll cube_action of (flip(board), flip(owner), a2, a1)."""
        analyzer = GnuBgAnalyzer(eval_level="1ply")
        cases = [
            # (board, cube_owner, cube_value, away1, away2, is_crawford)
            (_RACE_BOARD, "player", 2, 2, 5, False),
            (_RACE_BOARD, "opponent", 2, 2, 5, False),
            (_RACE_BOARD, "player", 2, 5, 2, False),
            (_RACE_BOARD, "opponent", 2, 5, 2, False),
            (_flip_board(_RACE_BOARD), "centered", 1, 2, 5, False),
            (_flip_board(_RACE_BOARD), "centered", 1, 5, 2, False),
            (_BLITZ_BOARD, "centered", 1, 2, 5, False),
            (_BLITZ_BOARD, "player", 2, 2, 5, False),
            (_BLITZ_BOARD, "opponent", 2, 2, 5, False),
            (_BLITZ_BOARD, "centered", 1, 1, 3, True),
            (_BLITZ_BOARD, "centered", 1, 3, 1, True),
            (_BLITZ_BOARD, "centered", 1, 1, 3, False),
            (_BLITZ_BOARD, "centered", 1, 3, 1, False),
            (_BLITZ_BOARD, "centered", 1, 1, 1, False),
            (_RACE_BOARD, "player", 2, 0, 0, False),
            (_RACE_BOARD, "opponent", 2, 0, 0, False),
        ]
        for board, owner, cv, a1, a2, craw in cases:
            with self.subTest(owner=owner, cv=cv, away=(a1, a2), craw=craw):
                pm = analyzer.post_move_analytics(
                    board, cube_owner=owner, cube_value=cv,
                    away1=a1, away2=a2, is_crawford=craw,
                )
                opp_ca = analyzer.cube_action(
                    _flip_board(board),
                    cube_value=cv,
                    cube_owner=_flip_cube_owner(owner),
                    away1=a2, away2=a1, is_crawford=craw,
                )
                self.assertAlmostEqual(pm.probs.win, 1.0 - opp_ca.probs.win, places=3)
                self.assertAlmostEqual(pm.probs.gammon_win, opp_ca.probs.gammon_loss, places=3)
                self.assertAlmostEqual(pm.probs.backgammon_win, opp_ca.probs.backgammon_loss, places=3)
                self.assertAlmostEqual(pm.probs.gammon_loss, opp_ca.probs.gammon_win, places=3)
                self.assertAlmostEqual(pm.probs.backgammon_loss, opp_ca.probs.backgammon_win, places=3)
                self.assertAlmostEqual(pm.cubeless_equity, -opp_ca.cubeless_equity, places=3)
                self.assertAlmostEqual(pm.cubeful_equity, -opp_ca.equity_nd, places=3)

        # Explicitly verify asymmetric match scores are NOT swapped:
        # On flip(_RACE_BOARD) (mover leads strongly), centered 1-cube:
        # at 2-away vs 5-away, opponent trails 5a-2a and has big cube leverage
        # -> mover's ND cubeful equity (+0.846) is substantially higher than
        # at 5-away vs 2-away (+0.682) — a 0.164 swing!
        pm_2a5a = analyzer.post_move_analytics(
            _flip_board(_RACE_BOARD), cube_owner="centered", cube_value=1,
            away1=2, away2=5,
        )
        pm_5a2a = analyzer.post_move_analytics(
            _flip_board(_RACE_BOARD), cube_owner="centered", cube_value=1,
            away1=5, away2=2,
        )
        self.assertGreater(pm_2a5a.cubeful_equity - pm_5a2a.cubeful_equity, 0.10)

    def test_checker_play_opponent_cube_money_returns_all_16_moves(self):
        analyzer = GnuBgAnalyzer(eval_level="1ply")
        res = analyzer.checker_play(
            _STARTING_BOARD, 3, 1, cube_value=2, cube_owner="opponent",
        )
        self.assertEqual(len(res.moves), 16)
        self.assertAlmostEqual(res.moves[0].equity_diff, 0.0, places=6)
        # Verify sorted descending by cubeful equity
        for i in range(len(res.moves) - 1):
            self.assertGreaterEqual(res.moves[i].equity, res.moves[i + 1].equity)
            self.assertLessEqual(res.moves[i + 1].equity_diff, 0.0)
        # Top move should be 8/5 6/5: positive cubeless (~+0.16), slightly
        # negative cubeful (~-0.01) because opponent owns the 2-cube.
        best = res.moves[0]
        expected_8565 = list(_STARTING_BOARD)
        expected_8565[8] -= 1
        expected_8565[6] -= 1
        expected_8565[5] += 2
        self.assertEqual(best.board, expected_8565)
        self.assertGreater(best.cubeless_equity, 0.10)
        self.assertLess(best.equity, 0.05)
        self.assertGreater(best.equity, -0.10)
        for m in res.moves:
            self.assertTrue(0.0 <= m.probs.win <= 1.0)
            self.assertTrue(math.isfinite(m.equity))
            self.assertTrue(math.isfinite(m.cubeless_equity))

    def test_checker_play_asymmetric_match_and_crawford_and_dmp(self):
        analyzer = GnuBgAnalyzer(eval_level="1ply")
        expected_8565 = list(_STARTING_BOARD)
        expected_8565[8] -= 1
        expected_8565[6] -= 1
        expected_8565[5] += 2

        # 2-away vs 5-away, opponent owns 2-cube: opponent redoubles aggressively
        # -> cubeful equity strongly negative (~-0.72).
        res_opp = analyzer.checker_play(
            _STARTING_BOARD, 3, 1, cube_value=2, cube_owner="opponent",
            away1=2, away2=5,
        )
        self.assertEqual(len(res_opp.moves), 16)
        self.assertEqual(res_opp.moves[0].board, expected_8565)
        self.assertLess(res_opp.moves[0].equity, -0.5)

        # 5-away vs 2-away, player owns 2-cube: player has huge match leverage
        # -> cubeful equity strongly positive (~+0.88).
        res_pl = analyzer.checker_play(
            _STARTING_BOARD, 3, 1, cube_value=2, cube_owner="player",
            away1=5, away2=2,
        )
        self.assertEqual(len(res_pl.moves), 16)
        self.assertEqual(res_pl.moves[0].board, expected_8565)
        self.assertGreater(res_pl.moves[0].equity, 0.5)

        # Crawford 1a-3a and DMP 1a-1a both return all 16 moves cleanly
        for a1, a2, craw in [(1, 3, True), (3, 1, True), (1, 1, False)]:
            res = analyzer.checker_play(
                _STARTING_BOARD, 3, 1, cube_value=1, cube_owner="centered",
                away1=a1, away2=a2, is_crawford=craw,
            )
            self.assertEqual(len(res.moves), 16)
            self.assertEqual(res.moves[0].board, expected_8565)

    def test_checker_play_single_candidate_and_forced_pass(self):
        from neongammon.board import possible_moves

        single_cands = possible_moves(_SINGLE_MOVE_BOARD, 1, 6)
        self.assertEqual(len(single_cands), 1)
        self.assertNotEqual(list(single_cands[0]), _SINGLE_MOVE_BOARD)

        # Dancing on a closed board produces a single forced-pass candidate
        # (the unchanged board).
        pass_cands = possible_moves(_FORCED_PASS_BOARD, 1, 6)
        self.assertEqual(len(pass_cands), 1)
        self.assertEqual(list(pass_cands[0]), _FORCED_PASS_BOARD)

        analyzer = GnuBgAnalyzer(eval_level="1ply")
        for owner, a1, a2 in [
            ("opponent", 0, 0),
            ("opponent", 2, 5),
            ("opponent", 5, 2),
            ("player", 2, 5),
            ("centered", 1, 3),
        ]:
            with self.subTest(owner=owner, away=(a1, a2)):
                res = analyzer.checker_play(
                    _SINGLE_MOVE_BOARD, 1, 6,
                    cube_value=2 if owner != "centered" else 1,
                    cube_owner=owner, away1=a1, away2=a2,
                    is_crawford=(a1 == 1 and a2 == 3),
                )
                self.assertEqual(len(res.moves), 1)
                m = res.moves[0]
                self.assertEqual(m.equity_diff, 0.0)
                self.assertTrue(0.0 <= m.probs.win <= 1.0)
                self.assertTrue(math.isfinite(m.equity))
                self.assertTrue(math.isfinite(m.cubeless_equity))
                # Matches negated opponent optimal cube_action on flipped board
                opp_ca = analyzer.cube_action(
                    _flip_board(m.board),
                    cube_value=2 if owner != "centered" else 1,
                    cube_owner=_flip_cube_owner(owner),
                    away1=a2, away2=a1,
                    is_crawford=(a1 == 1 and a2 == 3),
                )
                self.assertAlmostEqual(m.equity, -opp_ca.optimal_equity, places=3)
                self.assertAlmostEqual(m.cubeless_equity, -opp_ca.cubeless_equity, places=3)

        # Forced pass (dance) with opponent-owned cube and asymmetric score
        res_pass = analyzer.checker_play(
            _FORCED_PASS_BOARD, 1, 6, cube_value=2, cube_owner="opponent",
            away1=2, away2=5,
        )
        self.assertEqual(len(res_pass.moves), 1)
        self.assertEqual(res_pass.moves[0].board, _FORCED_PASS_BOARD)
        self.assertTrue(math.isfinite(res_pass.moves[0].equity))

    def test_cube_action_all_cube_owners_money_match_crawford_dmp(self):
        analyzer1 = GnuBgAnalyzer(eval_level="1ply")
        analyzer2 = GnuBgAnalyzer(eval_level="2ply")

        # Money: centered, player, opponent (jacoby True & False)
        for owner, cv, jac in [
            ("centered", 1, True),
            ("centered", 1, False),
            ("player", 2, True),
            ("opponent", 2, True),
            ("opponent", 4, False),
        ]:
            with self.subTest(mode="money", owner=owner, cv=cv, jac=jac):
                res = analyzer1.cube_action(
                    _RACE_BOARD, cube_value=cv, cube_owner=owner, jacoby=jac,
                )
                self.assertTrue(0.0 <= res.probs.win <= 1.0)
                self.assertTrue(math.isfinite(res.equity_nd))
                self.assertTrue(math.isfinite(res.optimal_equity))
                if owner == "opponent":
                    self.assertFalse(res.should_double)
                    self.assertEqual(res.optimal_action, "No Double")
                    self.assertAlmostEqual(res.optimal_equity, res.equity_nd, places=4)

        # Match: asymmetric scores, Crawford, post-Crawford, DMP, dead cube
        match_cases = [
            # (board, owner, cv, a1, a2, craw, must_not_double)
            (_RACE_BOARD, "centered", 1, 2, 5, False, False),
            (_RACE_BOARD, "centered", 1, 5, 2, False, False),
            (_RACE_BOARD, "player", 2, 2, 5, False, True),   # dead cube (2-away on 2-cube)
            (_RACE_BOARD, "player", 2, 5, 2, False, False),  # live redouble
            (_RACE_BOARD, "opponent", 2, 2, 5, False, True),
            (_RACE_BOARD, "opponent", 2, 5, 2, False, True),
            (_BLITZ_BOARD, "centered", 1, 1, 3, True, True),  # Crawford
            (_BLITZ_BOARD, "centered", 1, 3, 1, True, True),  # Crawford
            (_BLITZ_BOARD, "centered", 1, 1, 3, False, True), # post-Crawford leader
            (_BLITZ_BOARD, "centered", 1, 3, 1, False, False),# post-Crawford trailer
            (_BLITZ_BOARD, "centered", 1, 1, 1, False, True), # DMP
        ]
        for board, owner, cv, a1, a2, craw, must_not_double in match_cases:
            with self.subTest(owner=owner, cv=cv, away=(a1, a2), craw=craw):
                res = analyzer1.cube_action(
                    board, cube_value=cv, cube_owner=owner,
                    away1=a1, away2=a2, is_crawford=craw,
                )
                self.assertTrue(0.0 <= res.probs.win <= 1.0)
                self.assertTrue(math.isfinite(res.equity_nd))
                self.assertTrue(math.isfinite(res.equity_dt))
                self.assertTrue(math.isfinite(res.equity_dp))
                self.assertTrue(math.isfinite(res.optimal_equity))
                if must_not_double:
                    self.assertFalse(res.should_double)
                    self.assertEqual(res.optimal_action, "No Double")
                    self.assertAlmostEqual(res.optimal_equity, res.equity_nd, places=4)

        # Verify 2-ply (GNUbg 1-ply) also works with opponent-owned cube
        res2 = analyzer2.cube_action(
            _RACE_BOARD, cube_value=2, cube_owner="opponent", away1=2, away2=5,
        )
        self.assertEqual(res2.eval_level, "2-ply")
        self.assertFalse(res2.should_double)
        self.assertLess(res2.equity_nd, -0.5)

    def test_cross_check_gnubg_vs_bgbot_analyzer(self):
        """Cross-check GnuBgAnalyzer vs BgBotAnalyzer agreement in sign and
        ballpark equity on asymmetric match scores and opponent-owned cubes."""
        from neongammon import BgBotAnalyzer

        gnu = GnuBgAnalyzer(eval_level="1ply")
        bot = BgBotAnalyzer(eval_level="1ply")

        # 1. cube_action comparisons (money opp cube, asymmetric match, Crawford)
        cube_cases = [
            (_RACE_BOARD, 2, "opponent", 0, 0, False),
            (_RACE_BOARD, 2, "player", 0, 0, False),
            (_RACE_BOARD, 2, "opponent", 2, 5, False),
            (_RACE_BOARD, 2, "player", 2, 5, False),
            (_RACE_BOARD, 1, "centered", 2, 5, False),
            (_RACE_BOARD, 1, "centered", 5, 2, False),
            (_BLITZ_BOARD, 1, "centered", 1, 3, True),
            (_BLITZ_BOARD, 2, "opponent", 2, 5, False),
        ]
        for board, cv, owner, a1, a2, craw in cube_cases:
            with self.subTest(check="cube_action", owner=owner, cv=cv, away=(a1, a2), craw=craw):
                g = gnu.cube_action(
                    board, cube_value=cv, cube_owner=owner,
                    away1=a1, away2=a2, is_crawford=craw,
                )
                b = bot.cube_action(
                    board, cube_value=cv, cube_owner=owner,
                    away1=a1, away2=a2, is_crawford=craw,
                )
                # Same sign on ND cubeful equity
                self.assertEqual(math.copysign(1.0, g.equity_nd), math.copysign(1.0, b.equity_nd))
                self.assertLess(
                    abs(g.equity_nd - b.equity_nd), 0.25,
                    f"ND mismatch: gnubg={g.equity_nd:+.4f}, bgbot={b.equity_nd:+.4f}",
                )

        # 2. post_move_analytics comparisons (asymmetric match & owned cubes)
        pm_cases = [
            (_RACE_BOARD, "player", 2, 2, 5),
            (_RACE_BOARD, "opponent", 2, 2, 5),
            (_RACE_BOARD, "player", 2, 5, 2),
            (_flip_board(_RACE_BOARD), "centered", 1, 2, 5),
            (_flip_board(_RACE_BOARD), "centered", 1, 5, 2),
        ]
        for board, owner, cv, a1, a2 in pm_cases:
            with self.subTest(check="post_move", owner=owner, cv=cv, away=(a1, a2)):
                g = gnu.post_move_analytics(
                    board, cube_owner=owner, cube_value=cv, away1=a1, away2=a2,
                )
                b = bot.post_move_analytics(
                    board, cube_owner=owner, cube_value=cv, away1=a1, away2=a2,
                )
                self.assertEqual(
                    math.copysign(1.0, g.cubeful_equity),
                    math.copysign(1.0, b.cubeful_equity),
                )
                self.assertLess(
                    abs(g.cubeful_equity - b.cubeful_equity), 0.25,
                    f"PM cubeful mismatch: gnubg={g.cubeful_equity:+.4f}, bgbot={b.cubeful_equity:+.4f}",
                )

        # 3. checker_play comparisons (STARTING 3-1 with opp cube & match scores)
        bot2 = BgBotAnalyzer(eval_level="2ply")
        cp_cases = [
            (2, "opponent", 0, 0, bot),
            (2, "opponent", 2, 5, bot2),  # 2-ply bot models opp redouble like GNUbg eval
            (2, "player", 5, 2, bot),
        ]
        for cv, owner, a1, a2, ref_bot in cp_cases:
            with self.subTest(check="checker_play", owner=owner, away=(a1, a2)):
                g_cp = gnu.checker_play(
                    _STARTING_BOARD, 3, 1, cube_value=cv, cube_owner=owner,
                    away1=a1, away2=a2,
                )
                b_cp = ref_bot.checker_play(
                    _STARTING_BOARD, 3, 1, cube_value=cv, cube_owner=owner,
                    away1=a1, away2=a2,
                )
                self.assertEqual(len(g_cp.moves), len(b_cp.moves))
                # Same top move (8/5 6/5)
                self.assertEqual(g_cp.moves[0].board, b_cp.moves[0].board)
                self.assertEqual(
                    math.copysign(1.0, g_cp.moves[0].equity),
                    math.copysign(1.0, b_cp.moves[0].equity),
                )
                self.assertLess(
                    abs(g_cp.moves[0].equity - b_cp.moves[0].equity), 0.15,
                    f"Best move eq mismatch: gnubg={g_cp.moves[0].equity:+.4f}, "
                    f"bgbot={b_cp.moves[0].equity:+.4f}",
                )


if __name__ == '__main__':
    unittest.main(verbosity=2)
