# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""
Interface to GNUbg CLI for position evaluation.

Provides:
  - Module-level functions (backward-compatible):
      post_move_analytics()     — cubeless post-move evaluation
      post_move_analytics_many() — parallel batch version

  - GnuBgAnalyzer class (same interface as BgBotAnalyzer):
      cube_action()        — cubeful ND/DT/DP equities + cubeless probs
      post_move_analytics() — cubeless probs + cubeful equity
      checker_play()       — ranked legal moves with equities and probs
"""

import os
import re
import subprocess
import tempfile

from . import settings
from ._primitives import (
    _normalize_cube_context,
    cubeless_equity,
    flip_board as _flip_board,
    flip_owner as _flip_cube_owner,
    invert_probs,
    resolve_owner as _resolve_owner_name,
    validate_board,
)
from .types import (
    CheckerPlayResult,
    CubeActionResult,
    MoveAnalysis,
    PostMoveAnalysis,
    Probabilities,
)


def __getattr__(name):
    # GNUBG_CLI reports the binary resolved by neongammon.settings (configure it
    # there, not by assignment).
    if name == 'GNUBG_CLI':
        return settings.gnubg_path()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

_EVAL_LEVELS = {'1ply': 0, '2ply': 1, '3ply': 2, '4ply': 3}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _run_gnubg(cmd, timeout=60, gnubg_path=None):
    """Run the GNUbg CLI with the given command and return stdout.

    ``gnubg_path`` overrides the binary; by default it is resolved by
    :func:`neongammon.settings.gnubg_path` on every call.
    """
    exe = gnubg_path or settings.gnubg_path()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(cmd)
        cmd_file = f.name

    try:
        result = subprocess.run(
            [exe, '-q', '-t', '--no-rc', '-c', cmd_file],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"gnubg failed (rc={result.returncode}):\n"
                f"stderr: {result.stderr}\nstdout: {result.stdout}\n"
                f"command:\n{cmd}")
        return result.stdout
    finally:
        os.remove(cmd_file)


def _board_simple_str(checkers):
    """Format a board as 'set board simple ...' argument string."""
    parts = [str(checkers[25])]
    for n in checkers[1:25]:
        parts.append(str(n))
    parts.append(str(checkers[0]))
    return ' '.join(parts)


def _build_cube_analytics_command(checkers, n_plies=0, jacoby=True):
    """Build a gnubg command file to get pre-roll cube analytics for a position.

    The checkers represent the board from the perspective of the player on roll
    (pre-roll). gnubg's 'hint' with 'set turn 1' will give cube analytics
    including cubeless probabilities.

    Args:
        checkers: 26-element list (player on roll's perspective)
        n_plies: evaluation depth (0, 1, 2, or 3)
        jacoby: if True, enable Jacoby rule (gammons don't count when cube
            is centered)

    Returns:
        Command string for gnubg.
    """
    cmd = 'new session\n'
    cmd += f'set jacoby {"on" if jacoby else "off"}\n'
    cmd += f'set evaluation chequer eval plies {n_plies}\n'
    cmd += f'set evaluation cubedecision eval plies {n_plies}\n'
    # Lock cube at 1 for cubeless evaluation
    cmd += 'set cube value 1\n'

    cmd += f'set board simple {_board_simple_str(checkers)}\n'

    # Ensure it's the player's turn (player = "mghig" = X in gnubg)
    cmd += 'set turn 1\n'
    cmd += 'hint\n'
    return cmd


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_cube_analytics(output, n_plies=0):
    """Parse gnubg cube analytics output and return the 5 probabilities.

    The cube analytics section looks like:
        Cube analysis
        N-ply cubeless equity +0.077
          0.527 0.148 0.008 - 0.473 0.128 0.005
        Cubeful equities:
        ...

    Returns:
        dict with keys: p_win, p_gw, p_bw, p_gl, p_bl, equity_cubeless
    """
    if 'Cube analysis' not in output:
        raise ValueError(
            f"No cube analysis found in gnubg output:\n{output}")

    section = output[output.index('Cube analysis'):]
    lines = section.split('\n')

    # Line 1: "N-ply cubeless equity +X.XXX" or "0-ply cubeless equity +X.XXX"
    eq_line = lines[1].strip()
    eq_cubeless = float(eq_line.split()[-1])

    # Line 2: "  0.527 0.148 0.008 - 0.473 0.128 0.005"
    prob_line = lines[2].strip()
    bits = prob_line.split()
    # Format: p_win p_gw p_bw "-" p_loss p_gl p_bl
    p_win = float(bits[0])
    p_gw = float(bits[1])
    p_bw = float(bits[2])
    # bits[3] is "-"
    p_loss = float(bits[4])
    p_gl = float(bits[5])
    p_bl = float(bits[6])

    # Verify cubeless equity matches
    eq_computed = cubeless_equity((p_win, p_gw, p_bw, p_gl, p_bl))

    if abs(eq_computed - eq_cubeless) > 0.005:
        raise ValueError(
            f"Cubeless equity mismatch: computed {eq_computed:.4f} vs "
            f"reported {eq_cubeless:.4f}. "
            f"Probs: Win={p_win}, W(g)={p_gw}, W(bg)={p_bw}, "
            f"L(g)={p_gl}, L(bg)={p_bl}. "
            f"This may indicate incorrect gnubg setup.")

    return {
        'p_win': p_win,
        'p_gw': p_gw,
        'p_bw': p_bw,
        'p_gl': p_gl,
        'p_bl': p_bl,
        'equity_cubeless': eq_cubeless,
    }


def _parse_cube_section_full(lines):
    """Parse a single 'Cube analysis' section including cubeful equities.

    Args:
        lines: list of strings starting from the 'Cube analysis' header line.

    Returns:
        dict with cubeless probs/equity + cubeful ND/DT/DP + action info.
    """
    # Line 0: "Cube analysis"
    # Line 1: "N-ply cubeless equity +X.XXX" or "+X.XXX (Money: +Y.YYY)" in match play
    eq_line = lines[1].strip()
    eq_match = re.search(r'equity\s+([+-]?\d+\.\d+)', eq_line)
    if eq_match is None:
        raise ValueError(f"Could not parse cubeless equity line: {eq_line!r}")
    eq_cubeless = float(eq_match.group(1))

    # Line 2: "  0.527 0.148 0.008 - 0.473 0.128 0.005"
    prob_line = lines[2].strip()
    bits = prob_line.split()
    p_win = float(bits[0])
    p_gw = float(bits[1])
    p_bw = float(bits[2])
    p_gl = float(bits[5])
    p_bl = float(bits[6])

    # Parse cubeful equities from numbered lines:
    # "1. No double           +0.465"
    # "2. Double, pass        +1.000  (+0.535)"
    # "3. Double, take        +0.394  (-0.071)"
    nd = dt = dp = None
    optimal_action = ""
    for line in lines[3:]:
        line = line.strip()
        m = re.match(r'^\d+\.\s+(.+?)\s+([+-]?\d+\.\d+)', line)
        if m:
            act_name = m.group(1).strip()
            val = float(m.group(2))
            if ('No double' in act_name or 'No dbl' in act_name
                    or 'No redouble' in act_name):
                nd = val
            elif ('Double, take' in act_name or 'Redouble, take' in act_name
                    or 'Double, beaver' in act_name or 'Redouble, beaver' in act_name):
                dt = val
            elif 'Double, pass' in act_name or 'Redouble, pass' in act_name:
                dp = val
        elif 'Proper cube action' in line:
            if ':' in line:
                optimal_action = line.split(':', 1)[1].strip()
            break

    return {
        'p_win': p_win,
        'p_gw': p_gw,
        'p_bw': p_bw,
        'p_gl': p_gl,
        'p_bl': p_bl,
        'equity_cubeless': eq_cubeless,
        'equity_nd': nd,
        'equity_dt': dt,
        'equity_dp': dp,
        'optimal_action': optimal_action,
    }


def _parse_cube_analytics_full(output, search_start=0):
    """Parse a full cube analytics section including cubeful equities.

    Args:
        output: full gnubg stdout string.
        search_start: character offset to start searching from.

    Returns:
        (result_dict, next_offset) where result_dict has cubeless probs/equity
        plus cubeful ND/DT/DP, and next_offset is the position after this section.
    """
    idx = output.find('Cube analysis', search_start)
    if idx == -1:
        raise ValueError(
            f"No cube analysis found in gnubg output (from offset {search_start})")

    section = output[idx:]
    lines = section.split('\n')
    result = _parse_cube_section_full(lines)

    # Find next offset past this section
    next_offset = idx + len('Cube analysis') + 1
    return result, next_offset


_EVAL_ROW_RE = re.compile(
    r'^\s*(static|(\d+)\s*-?\s*ply):\s+'
    r'(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+'
    r'([+-]?\d+\.\d+)(?:\s+([+-]?\d+\.\d+))?'
)

_EVAL_BLOCK_SPLIT_RE = re.compile(r'^Evaluator:\s*', re.MULTILINE)


def _parse_eval_section(section, n_plies=0):
    """Parse a single GNUbg ``eval`` output section.

    GNUbg's ``eval`` command always emits a ply evaluation table::

                Win     W(g)    W(bg)   L(g)    L(bg)   Equity    Cubeful
        static: 0.253   0.000   0.000   0.000   0.000   -0.494    -0.793
         1 ply: 0.255   0.000   0.000   0.000   0.000   -0.491    -0.816

    regardless of cube ownership, Crawford, DMP, or dead-cube status.  When
    the player on roll is allowed to double, ``eval`` also appends a
    ``Cube analysis`` block with ND/DT/DP equities.

    Returns:
        dict with keys: p_win, p_gw, p_bw, p_gl, p_bl, equity_cubeless,
        equity_nd, equity_dt, equity_dp, optimal_equity, optimal_action,
        can_double.
    """
    cube_idx = section.find('Cube analysis')
    table_part = section[:cube_idx] if cube_idx != -1 else section

    rows_by_ply = {}
    last_row = None
    for line in table_part.split('\n'):
        m = _EVAL_ROW_RE.match(line)
        if not m:
            continue
        ply = 0 if m.group(1) == 'static' else int(m.group(2))
        eq_cl = float(m.group(8))
        eq_cf = float(m.group(9)) if m.group(9) is not None else eq_cl
        row = {
            'p_win': float(m.group(3)),
            'p_gw': float(m.group(4)),
            'p_bw': float(m.group(5)),
            'p_gl': float(m.group(6)),
            'p_bl': float(m.group(7)),
            'equity_cubeless': eq_cl,
            'equity_cubeful': eq_cf,
        }
        rows_by_ply[ply] = row
        last_row = row

    if last_row is None:
        raise ValueError(f"No eval table row found in gnubg output:\n{section}")

    chosen = rows_by_ply.get(n_plies, last_row)
    nd = chosen['equity_cubeful']
    dt = None
    dp = None
    optimal_action = ""
    can_double = False
    optimal_equity = nd

    if cube_idx != -1:
        cube = _parse_cube_section_full(section[cube_idx:].split('\n'))
        if cube['equity_nd'] is not None:
            nd = cube['equity_nd']
        dt = cube['equity_dt']
        dp = cube['equity_dp']
        optimal_action = cube['optimal_action']
        if dt is not None and dp is not None:
            can_double = True
            should_double = min(dt, dp) > nd
            should_take = dt <= dp
            optimal_equity = (dt if should_take else dp) if should_double else nd
        else:
            optimal_equity = nd

    return {
        'p_win': chosen['p_win'],
        'p_gw': chosen['p_gw'],
        'p_bw': chosen['p_bw'],
        'p_gl': chosen['p_gl'],
        'p_bl': chosen['p_bl'],
        'equity_cubeless': chosen['equity_cubeless'],
        'equity_nd': nd,
        'equity_dt': dt,
        'equity_dp': dp,
        'optimal_equity': optimal_equity,
        'optimal_action': optimal_action,
        'can_double': can_double,
    }


def _parse_eval_batch(output, n_plies=0):
    """Parse all ``eval`` sections from a GNUbg stdout string."""
    parts = _EVAL_BLOCK_SPLIT_RE.split(output)
    if len(parts) > 1:
        return [_parse_eval_section(blk, n_plies=n_plies) for blk in parts[1:]]
    if _EVAL_ROW_RE.search(output):
        return [_parse_eval_section(output, n_plies=n_plies)]
    return []


def _parse_eval_single(output, n_plies=0):
    """Parse a single ``eval`` section from a GNUbg stdout string."""
    results = _parse_eval_batch(output, n_plies=n_plies)
    if not results:
        raise ValueError(f"No evaluation found in gnubg output:\n{output}")
    return results[-1]


# ---------------------------------------------------------------------------
# Module-level functions (backward-compatible)
# ---------------------------------------------------------------------------

def post_move_analytics(checkers, n_plies=0, timeout=60, jacoby=True, gnubg_path=None):
    """Evaluate a post-move board position using gnubg.

    Returns cubeless probabilities from the perspective of the player who just
    moved. This matches our NN's evaluation semantics: the NN outputs are
    probabilities assuming the player just moved to this position and it's
    about to switch to the other player's turn.

    Approach:
    1. Flip the board to the opponent's perspective (opponent is about to roll)
    2. Ask gnubg for pre-roll cube analytics from opponent's perspective
    3. Invert the probabilities back to the mover's perspective

    Args:
        checkers: 26-element list in standard board format (mover's perspective):
            [0] = opponent checkers on bar (>= 0)
            [1-24] = board points (positive = mover, negative = opponent)
            [25] = mover checkers on bar (>= 0)
        n_plies: evaluation depth (0, 1, 2, or 3)
        timeout: seconds before killing gnubg process
        jacoby: if True, enable Jacoby rule
        gnubg_path: GNUbg binary to run (default: neongammon.settings.gnubg_path())

    Returns:
        dict with:
            probs: [P(win), P(gw), P(bw), P(gl), P(bl)] from mover's perspective
            equity: cubeless equity from mover's perspective
    """
    checkers = list(validate_board(checkers))

    # Check for game-over: if the mover has no checkers left, they've won
    n_mover = sum(c for c in checkers[1:25] if c > 0) + checkers[25]
    n_opponent = sum(-c for c in checkers[1:25] if c < 0) + checkers[0]

    if n_mover == 0:
        # Mover has borne off all checkers — they win
        # Check if opponent has any borne off
        opp_borne_off = 15 - n_opponent
        if opp_borne_off > 0:
            # Opponent has borne off at least one: single win
            return {'probs': [1.0, 0.0, 0.0, 0.0, 0.0], 'equity': 1.0}
        else:
            # Opponent has borne off none: gammon at least
            # Check for backgammon: opponent has checker in mover's home or on bar
            has_in_home_or_bar = checkers[0] > 0  # opponent on bar
            if not has_in_home_or_bar:
                for i in range(1, 7):  # mover's home board = points 1-6
                    if checkers[i] < 0:
                        has_in_home_or_bar = True
                        break
            if has_in_home_or_bar:
                return {'probs': [1.0, 1.0, 1.0, 0.0, 0.0], 'equity': 3.0}
            else:
                return {'probs': [1.0, 1.0, 0.0, 0.0, 0.0], 'equity': 2.0}

    if n_opponent == 0:
        # Opponent has borne off all checkers — mover loses
        mover_borne_off = 15 - n_mover
        if mover_borne_off > 0:
            return {'probs': [0.0, 0.0, 0.0, 0.0, 0.0], 'equity': -1.0}
        else:
            has_in_home_or_bar = checkers[25] > 0
            if not has_in_home_or_bar:
                for i in range(19, 25):  # opponent's home = points 19-24
                    if checkers[i] > 0:
                        has_in_home_or_bar = True
                        break
            if has_in_home_or_bar:
                return {'probs': [0.0, 0.0, 0.0, 1.0, 1.0], 'equity': -3.0}
            else:
                return {'probs': [0.0, 0.0, 0.0, 1.0, 0.0], 'equity': -2.0}

    # Flip to opponent's perspective (opponent is about to roll)
    opp_board = _flip_board(checkers)

    # Get pre-roll cube analytics from opponent's perspective
    cmd = _build_cube_analytics_command(opp_board, n_plies, jacoby=jacoby)
    output = _run_gnubg(cmd, timeout=timeout, gnubg_path=gnubg_path)
    opp_result = _parse_cube_analytics(output, n_plies)

    # Invert probabilities: opponent's perspective -> mover's perspective
    mover_probs = invert_probs((
        opp_result['p_win'],
        opp_result['p_gw'],
        opp_result['p_bw'],
        opp_result['p_gl'],
        opp_result['p_bl'],
    ))
    mover_eq = -opp_result['equity_cubeless']

    probs = Probabilities.clamp_list(mover_probs)
    return {
        'probs': probs,
        'equity': mover_eq,
    }


def post_move_analytics_many(checkers_list, n_plies=0, max_workers=None, timeout=60,
                             jacoby=True, gnubg_path=None):
    """Evaluate multiple post-move positions in parallel.

    Args:
        checkers_list: list of 26-element checker lists (mover's perspective)
        n_plies: evaluation depth
        max_workers: thread pool size (default: CPU count)
        timeout: per-position timeout in seconds
        jacoby: if True, enable Jacoby rule

    Returns:
        List of result dicts (same format as post_move_analytics)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing

    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(checkers_list))

    results = [None] * len(checkers_list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, checkers in enumerate(checkers_list):
            future = executor.submit(post_move_analytics, checkers, n_plies, timeout,
                                     jacoby=jacoby, gnubg_path=gnubg_path)
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return results


# ---------------------------------------------------------------------------
# GnuBgAnalyzer class
# ---------------------------------------------------------------------------

class GnuBgAnalyzer:
    """GNUbg-backed analyzer with the same interface as BgBotAnalyzer.

    Usage::

        from neongammon import GnuBgAnalyzer
        analyzer = GnuBgAnalyzer(eval_level="3ply")
        cube = analyzer.cube_action(board, cube_value=1, cube_owner="centered")
        print(cube.equity_nd, cube.equity_dt, cube.optimal_action)
    """

    def __init__(self, eval_level="1ply", timeout=120, jacoby=True, gnubg_path=None):
        """``gnubg_path``: the GNUbg binary for this analyzer; ``None`` uses
        :func:`neongammon.settings.gnubg_path` (code setting, ``NEONGAMMON_GNUBG``,
        the user settings file, then auto-detection)."""
        self.gnubg_path = os.fspath(gnubg_path) if gnubg_path is not None else None
        if eval_level not in _EVAL_LEVELS:
            raise ValueError(
                f"eval_level must be one of {list(_EVAL_LEVELS)}, got {eval_level!r}")
        self.n_plies = _EVAL_LEVELS[eval_level]
        # Display label uses our XG convention (1-ply = raw NN)
        self.eval_level_str = f"{self.n_plies + 1}-ply"
        self.timeout = timeout
        self.jacoby = jacoby

    # -- cube_action --------------------------------------------------------

    def cube_action(self, board, cube_value=1, cube_owner="centered",
                    *, away1=0, away2=0, is_crawford=False, jacoby=None):
        """Evaluate cube decision for a pre-roll position.

        Returns CubeActionResult with cubeful ND/DT/DP equities and cubeless
        probabilities, all from the current player's perspective.  When the
        player on roll cannot double (opponent owns the cube, Crawford game,
        DMP, or dead cube), returns ``should_double=False``,
        ``optimal_action="No Double"``, and ``equity_nd == optimal_equity``
        from GNUbg's ``eval`` output without crashing.

        Args:
            jacoby: Override the instance default. None = use self.jacoby.
                Auto-disabled for match play.
        """
        board = list(validate_board(board))

        j, _ = _normalize_cube_context(
            cube_value, away1, away2, self.jacoby if jacoby is None else jacoby, False
        )
        cmd = self._build_preroll_command(board, cube_value, cube_owner,
                                          away1=away1, away2=away2,
                                          is_crawford=is_crawford, jacoby=j)
        output = _run_gnubg(cmd, timeout=self.timeout, gnubg_path=self.gnubg_path)
        r = _parse_eval_single(output, n_plies=self.n_plies)

        probs = Probabilities(r['p_win'], r['p_gw'], r['p_bw'],
                              r['p_gl'], r['p_bl'])

        nd = r['equity_nd']
        if r['can_double']:
            dt = r['equity_dt']
            dp = r['equity_dp']
            should_double = min(dt, dp) > nd
            should_take = dt <= dp
            if should_double:
                optimal_action = "Double/Take" if should_take else "Double/Pass"
                optimal_equity = dt if should_take else dp
            else:
                optimal_action = "No Double"
                optimal_equity = nd
        else:
            dt = nd
            dp = 1.0
            should_double = False
            should_take = dt <= dp
            optimal_action = "No Double"
            optimal_equity = nd

        return CubeActionResult(
            probs=probs,
            cubeless_equity=r['equity_cubeless'],
            equity_nd=nd,
            equity_dt=dt,
            equity_dp=dp,
            should_double=should_double,
            should_take=should_take,
            optimal_equity=optimal_equity,
            optimal_action=optimal_action,
            eval_level=self.eval_level_str,
        )

    # -- post_move_analytics ------------------------------------------------

    def post_move_analytics(self, board, cube_owner="centered", cube_value=1,
                            *, away1=0, away2=0, is_crawford=False, jacoby=None):
        """Evaluate a post-move position (after move, before opponent rolls).

        Returns PostMoveAnalysis with cubeless probs/equity and cubeful equity.
        The cubeful equity is the ND cubeful equity from GNUbg (the value of
        the position without any doubling action).

        Args:
            jacoby: Override the instance default. None = use self.jacoby.
                Auto-disabled for match play.
        """
        board = list(validate_board(board))

        j, _ = _normalize_cube_context(
            cube_value, away1, away2, self.jacoby if jacoby is None else jacoby, False
        )

        # Flip board, cube ownership, and match scores to opponent's perspective
        # (opponent is about to roll).
        opp_board = _flip_board(board)
        opp_owner = _flip_cube_owner(cube_owner)
        opp_away1, opp_away2 = away2, away1

        cmd = self._build_preroll_command(opp_board, cube_value, opp_owner,
                                          away1=opp_away1, away2=opp_away2,
                                          is_crawford=is_crawford, jacoby=j)
        output = _run_gnubg(cmd, timeout=self.timeout, gnubg_path=self.gnubg_path)
        r = _parse_eval_single(output, n_plies=self.n_plies)

        # Invert to mover's perspective
        probs = Probabilities.from_list(invert_probs((
            r['p_win'], r['p_gw'], r['p_bw'], r['p_gl'], r['p_bl'],
        )))
        cubeless_eq = -r['equity_cubeless']
        cubeful_eq = -r['equity_nd']

        return PostMoveAnalysis(
            probs=probs,
            cubeless_equity=cubeless_eq,
            cubeful_equity=cubeful_eq,
            eval_level=self.eval_level_str,
        )

    # -- checker_play -------------------------------------------------------

    def checker_play(self, board, die1, die2, cube_value=1, cube_owner="centered",
                     *, away1=0, away2=0, is_crawford=False, jacoby=None):
        """Evaluate all legal moves for a position + dice.

        Returns CheckerPlayResult with moves sorted best-first by cubeful equity
        (tie-broken by cubeless equity, then board).  Uses batch evaluation:
        generates all legal boards, flips each board along with cube ownership
        and match scores to the opponent's perspective, and evaluates them with
        GNUbg ``eval`` in a single subprocess.

        Args:
            jacoby: Override the instance default. None = use self.jacoby.
                Auto-disabled for match play.
        """
        from .board import possible_moves

        board = list(validate_board(board))

        j, _ = _normalize_cube_context(
            cube_value, away1, away2, self.jacoby if jacoby is None else jacoby, False
        )

        candidates = possible_moves(board, die1, die2)

        if len(candidates) == 0:
            return CheckerPlayResult(
                moves=[], board=list(board), die1=die1, die2=die2,
                eval_level=self.eval_level_str)

        # Evaluate all candidate boards in a single GNUbg subprocess.
        cmd = self._build_batch_postmove_command(
            candidates, cube_value, cube_owner,
            away1=away1, away2=away2, is_crawford=is_crawford, jacoby=j)
        output = _run_gnubg(cmd, timeout=self.timeout, gnubg_path=self.gnubg_path)
        eval_results = _parse_eval_batch(output, n_plies=self.n_plies)

        if len(eval_results) != len(candidates):
            raise ValueError(
                f"Expected {len(candidates)} GNUbg eval results, "
                f"got {len(eval_results)}"
            )

        move_analyses = []
        for cand_board, r in zip(candidates, eval_results):
            # Invert to mover's perspective
            probs = Probabilities.from_list(invert_probs((
                r['p_win'], r['p_gw'], r['p_bw'], r['p_gl'], r['p_bl'],
            )))
            cubeless_eq = -r['equity_cubeless']
            cubeful_eq = -r['optimal_equity']

            move_analyses.append(MoveAnalysis(
                board=list(cand_board), equity=cubeful_eq,
                cubeless_equity=cubeless_eq,
                probs=probs, equity_diff=0.0,
                eval_level=self.eval_level_str))

        # Sort best-first by cubeful equity, breaking ties by cubeless equity
        # (saturated -1.000 pass positions) and board representation.
        move_analyses.sort(
            key=lambda m: (-m.equity, -m.cubeless_equity, tuple(m.board))
        )

        # Compute equity_diff relative to best
        best_eq = move_analyses[0].equity
        for m in move_analyses:
            m.equity_diff = m.equity - best_eq

        return CheckerPlayResult(
            moves=move_analyses, board=list(board), die1=die1, die2=die2,
            eval_level=self.eval_level_str)

    # -- command builders ---------------------------------------------------

    @staticmethod
    def _match_prefix(away1, away2, is_crawford):
        """Build GNUbg commands for match context.

        Returns 'new session\\n' for money games (away1==0 and away2==0),
        or 'new match N\\nset score ...\\n[set postcrawford on\\n]' for match play.
        """
        if not away1 and not away2:
            return 'new session\n'
        match_length = max(away1, away2)
        player_score = match_length - away1
        opp_score = match_length - away2
        cmd = f'new match {match_length}\n'
        cmd += f'set score {opp_score} {player_score}\n'
        # Post-Crawford: one player is 1-away and Crawford has already occurred.
        # GNUbg defaults to Crawford when a player first reaches 1-away, so we
        # must explicitly enable post-Crawford play.  During Crawford
        # (is_crawford=True) we don't set this — GNUbg handles it automatically.
        if not is_crawford and (away1 == 1 or away2 == 1):
            cmd += 'set postcrawford on\n'
        return cmd

    @staticmethod
    def _cube_owner_cmd(cube_owner):
        """Return the GNUbg command setting cube ownership from the on-roll POV.

        In GNUbg, player 1 is on roll (``set turn 1``) and player 0 is the
        opponent.
        """
        owner = _resolve_owner_name(cube_owner)
        if owner == "player":
            return 'set cube owner 1\n'
        if owner == "opponent":
            return 'set cube owner 0\n'
        return 'set cube centre\n'

    def _build_preroll_command(self, checkers, cube_value=1, cube_owner="centered",
                               *, away1=0, away2=0, is_crawford=False, jacoby=True):
        """Build a GNUbg command for a pre-roll position evaluation."""
        cmd = self._match_prefix(away1, away2, is_crawford)
        cmd += f'set jacoby {"on" if jacoby else "off"}\n'
        cmd += f'set evaluation chequer eval plies {self.n_plies}\n'
        cmd += f'set evaluation cubedecision eval plies {self.n_plies}\n'
        cmd += f'set cube value {cube_value}\n'
        cmd += self._cube_owner_cmd(cube_owner)

        cmd += f'set board simple {_board_simple_str(checkers)}\n'
        cmd += 'set turn 1\n'
        cmd += 'eval\n'
        return cmd

    def _build_batch_postmove_command(self, boards, cube_value=1,
                                      cube_owner="centered",
                                      *, away1=0, away2=0, is_crawford=False,
                                      jacoby=True):
        """Build a GNUbg command that evaluates multiple post-move positions.

        ``boards``, ``cube_owner``, and ``(away1, away2)`` are given from the
        mover's perspective.  Each board is flipped to the opponent's
        perspective together with flipped cube ownership and swapped match
        scores, and evaluated via GNUbg ``eval`` in a single subprocess.
        """
        opp_owner = _flip_cube_owner(cube_owner)
        opp_away1, opp_away2 = away2, away1
        cmd = self._match_prefix(opp_away1, opp_away2, is_crawford)
        cmd += f'set jacoby {"on" if jacoby else "off"}\n'
        cmd += f'set evaluation chequer eval plies {self.n_plies}\n'
        cmd += f'set evaluation cubedecision eval plies {self.n_plies}\n'
        cmd += f'set cube value {cube_value}\n'
        cmd += self._cube_owner_cmd(opp_owner)

        for board in boards:
            opp_board = _flip_board(board)
            cmd += f'set board simple {_board_simple_str(opp_board)}\n'
            cmd += 'set turn 1\n'
            cmd += 'eval\n'

        return cmd
