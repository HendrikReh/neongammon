# SPDX-License-Identifier: MPL-2.0
"""Unit tests for unified XG binary parsing (xg_file.py) and PR stats (xg_compare.py).

Pure-Python suite: runs without importing ``bgbot_cpp``.
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

import pytest

from neongammon import xg_compare, xg_file


# ---------------------------------------------------------------------------
# Synthetic XG record & archive builders (using xg_file's Delphi offsets)
# ---------------------------------------------------------------------------


def _xg_eval(win: float, gw: float, bw: float, gl: float, bl: float, equity: float) -> tuple[float, ...]:
    """Assemble XG's 7-float eval: [bg_loss, g_loss, loss, win, g_win, bg_win, equity]."""
    return (bl, gl, 1.0 - win, win, gw, bw, equity)


def _make_match_header(
    *,
    player1: str = "Alice",
    player2: str = "Bob",
    unicode_names: bool = True,
    match_length: int = 5,
    jacoby: bool = False,
    beaver: bool = False,
    version: int = 30,
    magic: int = xg_file.MAGIC_DMLI,
) -> bytes:
    rec = bytearray(xg_file.TSAVEREC_SIZE)
    rec[8] = xg_file.TS_HEADER_MATCH
    p1_ascii = player1.encode("ascii", errors="ignore")[:40]
    p2_ascii = player2.encode("ascii", errors="ignore")[:40]
    rec[xg_file._HDR_SPLAYER1] = len(p1_ascii)
    rec[xg_file._HDR_SPLAYER1 + 1:xg_file._HDR_SPLAYER1 + 1 + len(p1_ascii)] = p1_ascii
    rec[xg_file._HDR_SPLAYER2] = len(p2_ascii)
    rec[xg_file._HDR_SPLAYER2 + 1:xg_file._HDR_SPLAYER2 + 1 + len(p2_ascii)] = p2_ascii
    struct.pack_into("<i", rec, xg_file._HDR_MATCH_LENGTH, match_length)
    rec[xg_file._HDR_JACOBY] = 1 if jacoby else 0
    rec[xg_file._HDR_BEAVER] = 1 if beaver else 0
    struct.pack_into("<d", rec, xg_file._HDR_DATE, 45000.5)
    struct.pack_into("<iI", rec, xg_file._HDR_VERSION, version, magic)
    if unicode_names:
        u1 = (player1 + "\x00").encode("utf-16-le")[:256]
        u2 = (player2 + "\x00").encode("utf-16-le")[:256]
        rec[xg_file._HDR_PLAYER1:xg_file._HDR_PLAYER1 + len(u1)] = u1
        rec[xg_file._HDR_PLAYER2:xg_file._HDR_PLAYER2 + len(u2)] = u2
    return bytes(rec)


def _make_game_header(
    *,
    game_number: int = 1,
    score1: int = 0,
    score2: int = 0,
    crawford: bool = False,
) -> bytes:
    rec = bytearray(xg_file.TSAVEREC_SIZE)
    rec[8] = xg_file.TS_HEADER_GAME
    struct.pack_into("<i", rec, xg_file._GAME_SCORE1, score1)
    struct.pack_into("<i", rec, xg_file._GAME_SCORE2, score2)
    rec[xg_file._GAME_CRAWFORD] = 1 if crawford else 0
    struct.pack_into("<i", rec, xg_file._GAME_NUMBER, game_number)
    return bytes(rec)


def _make_game_footer(
    *,
    score1: int = 1,
    score2: int = 0,
    winner: int = 1,
    points: int = 1,
    termination: int = 1,
    crawford_next: bool = False,
) -> bytes:
    rec = bytearray(xg_file.TSAVEREC_SIZE)
    rec[8] = xg_file.TS_FOOTER_GAME
    struct.pack_into("<i", rec, xg_file._FG_SCORE1, score1)
    struct.pack_into("<i", rec, xg_file._FG_SCORE2, score2)
    rec[xg_file._FG_CRAWFORD_NEXT] = 1 if crawford_next else 0
    struct.pack_into("<i", rec, xg_file._FG_WINNER, winner)
    struct.pack_into("<i", rec, xg_file._FG_POINTS, points)
    struct.pack_into("<i", rec, xg_file._FG_TERMINATION, termination)
    return bytes(rec)


def _make_match_footer(*, score1: int = 5, score2: int = 2, winner: int = 1) -> bytes:
    rec = bytearray(xg_file.TSAVEREC_SIZE)
    rec[8] = xg_file.TS_FOOTER_MATCH
    struct.pack_into("<i", rec, xg_file._FM_SCORE1, score1)
    struct.pack_into("<i", rec, xg_file._FM_SCORE2, score2)
    struct.pack_into("<i", rec, xg_file._FM_WINNER, winner)
    return bytes(rec)


def _make_cube_record(
    *,
    actif: int = 1,
    double: int = 0,
    take: int = -1,
    cube_b: int = 0,
    flag_double: int = 0,
    eval_nd: tuple[float, ...] | None = None,
    equity_nd: float = 0.20,
    equity_dt: float = 0.10,
    equity_dp: float = 1.00,
    err_cube: float = 0.0,
    analyze_c: int = 3,
    position_raw: list[int] | None = None,
) -> bytes:
    rec = bytearray(xg_file.TSAVEREC_SIZE)
    rec[8] = xg_file.TS_CUBE
    struct.pack_into("<i", rec, xg_file._CUBE_ACTIF, actif)
    struct.pack_into("<i", rec, xg_file._CUBE_DOUBLE, double)
    struct.pack_into("<i", rec, xg_file._CUBE_TAKE, take)
    struct.pack_into("<i", rec, xg_file._CUBE_B, cube_b)
    pos = position_raw if position_raw is not None else [0] * 26
    struct.pack_into("<26b", rec, xg_file._CUBE_POSITION, *pos)
    da = xg_file._CUBE_DOUBLE_ACTION
    struct.pack_into("<h", rec, da + xg_file._DA_FLAG_DOUBLE, flag_double)
    ev = eval_nd if eval_nd is not None else _xg_eval(0.55, 0.15, 0.01, 0.10, 0.005, equity_nd)
    struct.pack_into("<7f", rec, da + xg_file._DA_EVAL, *ev)
    struct.pack_into("<3f", rec, da + xg_file._DA_EQU_ND, equity_nd, equity_dt, equity_dp)
    struct.pack_into("<d", rec, xg_file._CUBE_ERR_CUBE, err_cube)
    struct.pack_into("<i", rec, xg_file._CUBE_ANALYZE_C, analyze_c)
    struct.pack_into("<i", rec, xg_file._CUBE_ROLLOUT_INDEX, -1)
    return bytes(rec)


def _make_move_record(
    *,
    actif: int = 1,
    position_i_raw: list[int],
    position_end_raw: list[int],
    dice: tuple[int, int] = (3, 1),
    moves: list[tuple[list[int], tuple[float, ...], int]] | None = None,
    err_move: float = 0.0,
    analyze_m: int = 3,
    is_forced: bool = False,
) -> bytes:
    rec = bytearray(xg_file.TSAVEREC_SIZE)
    rec[8] = xg_file.TS_MOVE
    struct.pack_into("<26b", rec, xg_file._MOVE_POSITION_I, *position_i_raw)
    struct.pack_into("<26b", rec, xg_file._MOVE_POSITION_END, *position_end_raw)
    struct.pack_into("<i", rec, xg_file._MOVE_ACTIF, actif)
    struct.pack_into("<2i", rec, xg_file._MOVE_DICE, dice[0], dice[1])
    dm = xg_file._MOVE_DATAMOVES
    move_list = moves or []
    struct.pack_into("<i", rec, dm + xg_file._BM_NMOVES, len(move_list))
    for i, (pos_raw, ev, level) in enumerate(move_list):
        struct.pack_into("<26b", rec, dm + xg_file._BM_POS_PLAYED + 26 * i, *pos_raw)
        struct.pack_into("<h", rec, dm + xg_file._BM_EVAL_LEVEL + 4 * i, level)
        struct.pack_into("<7f", rec, dm + xg_file._BM_EVAL + 28 * i, *ev)
    rec[dm + xg_file._BM_IS_FORCED] = 1 if is_forced else 0
    struct.pack_into("<d", rec, xg_file._MOVE_ERR_MOVE, err_move)
    struct.pack_into("<32i", rec, xg_file._MOVE_ROLLOUT_INDEX, *([-1] * 32))
    struct.pack_into("<i", rec, xg_file._MOVE_ANALYZE_M, analyze_m)
    return bytes(rec)


def _pack_archive(
    tempxg: bytes,
    *,
    compress_registry: bool = True,
    comp_reg_upper_garbage: int = 0,
    entry_name: str = "temp.xg",
) -> bytes:
    prefix = bytearray(xg_file.RICH_HEADER_SIZE)
    struct.pack_into("<I", prefix, 0, xg_file.MAGIC_RGMH)

    blob = zlib.compress(tempxg)
    fcrc = xg_file._crc32_signed(tempxg)
    reg_rec = bytearray(xg_file.REGISTRY_ENTRY_SIZE)
    name_b = entry_name.encode("ascii")
    reg_rec[0] = len(name_b)
    reg_rec[1:1 + len(name_b)] = name_b
    struct.pack_into("<iiii", reg_rec, 512, len(tempxg), len(blob), 0, fcrc)
    reg_rec[528] = 0  # fsCompressed
    reg_rec[529] = 2

    reg_bytes = zlib.compress(bytes(reg_rec)) if compress_registry else bytes(reg_rec)
    stream = blob + reg_bytes
    crc = xg_file._crc32_signed(stream)
    comp_reg_flag = (1 if compress_registry else 0) | (comp_reg_upper_garbage & ~0xFF)
    trailer = struct.pack(
        "<iiiiiI",
        crc,
        1,
        1,
        len(reg_bytes),
        len(blob),
        comp_reg_flag & 0xFFFFFFFF,
    )
    trailer += bytes(xg_file.ARCHIVE_TRAILER_SIZE - len(trailer))
    return bytes(prefix) + stream + trailer


# ---------------------------------------------------------------------------
# Archive integrity: CRC-32, comp_reg & 0xFF, malformed containers
# ---------------------------------------------------------------------------


def test_archive_crc32_validation_rejects_corrupted_bytes():
    tempxg = (
        _make_match_header()
        + _make_game_header()
        + _make_game_footer()
        + _make_match_footer()
    )
    raw = bytearray(_pack_archive(tempxg))
    # Flip a byte inside the compressed stream covered by the trailer CRC-32.
    corrupt_idx = xg_file.RICH_HEADER_SIZE + 2
    raw[corrupt_idx] ^= 0x01
    corrupted = bytes(raw)

    with pytest.raises(ValueError, match="CRC mismatch"):
        xg_file.XgArchive.from_bytes(corrupted)
    with pytest.raises(ValueError, match="CRC mismatch"):
        xg_compare.parse_xg_game(corrupted)
    with pytest.raises(ValueError, match="CRC mismatch"):
        xg_compare.parse_xg_match(corrupted)


@pytest.mark.parametrize("compress_registry", [False, True])
def test_archive_comp_reg_masks_delphi_1byte_bool(compress_registry: bool):
    """Upper 3 garbage bytes in Delphi's 1-byte CompressedRegistry bool must be masked."""
    tempxg = (
        _make_match_header(player1="Müller", player2="Østergaard", match_length=7)
        + _make_game_header(game_number=1, score1=2, score2=1, crawford=False)
        + _make_game_footer(score1=3, score2=1)
        + _make_match_footer(score1=7, score2=1)
    )
    raw = _pack_archive(
        tempxg,
        compress_registry=compress_registry,
        comp_reg_upper_garbage=0xAB12CD00,
    )
    arch = xg_file.XgArchive.from_bytes(raw)
    assert arch.get("temp.xg") == tempxg

    match = xg_compare.parse_xg_match(raw)
    assert match["match_length"] == 7
    assert match["player1"] == "Müller"
    assert match["player2"] == "Østergaard"
    assert len(match["games"]) == 1


def test_archive_validation_errors():
    with pytest.raises(ValueError, match="too small"):
        xg_file.XgArchive.from_bytes(b"short")

    bad_magic = bytearray(xg_file.RICH_HEADER_SIZE + xg_file.ARCHIVE_TRAILER_SIZE)
    with pytest.raises(ValueError, match="bad magic"):
        xg_file.XgArchive.from_bytes(bytes(bad_magic))

    # Missing temp.xg entry
    no_tempxg = _pack_archive(b"dummy", entry_name="other.bin")
    with pytest.raises(ValueError, match="temp.xg not found"):
        xg_compare.parse_xg_match(no_tempxg)

    # Bad DMLI magic in tsHeaderMatch
    bad_dmli = _pack_archive(_make_match_header(magic=0xDEADBEEF))
    with pytest.raises(ValueError, match="DMLI magic"):
        xg_compare.parse_xg_match(bad_dmli)


# ---------------------------------------------------------------------------
# norm_bars & perspective consistency
# ---------------------------------------------------------------------------


def test_norm_bars_normalizes_signed_bars_for_both_players():
    """XG signs opponent bar <= 0; parsed turns must have nonnegative bars and match errors."""
    # User turn (actif=1): opponent has 2 on bar (-2 at idx 0), user has 1 on bar (+1 at idx 25).
    p1_pre_raw = [-2, 0, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -3, 0, 0, 0, 0, 1, 1]
    p1_best_raw = [-2, 0, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -3, 0, 0, 1, 0, 1, 0]
    p1_played_raw = [-2, 0, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -3, 0, 1, 0, 0, 1, 0]

    # Bot turn (actif=-1): mover (bot) has 2 on bar (+2 at idx 25 in mover frame),
    # opponent (user) has 1 on bar (-1 at idx 0 in mover frame).
    # In P1 frame (PositionI), user bar is +1 at idx 25, bot bar is -2 at idx 0.
    bot_pre_p1_raw = [-2, 1, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -3, 0, 0, 0, 0, 0, 1]
    bot_best_mover_raw = [-1, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 5, -5, 0, 0, 0, -3, 0, -5, 0, 0, 1, 0, -1, 1]
    bot_played_mover_raw = [-1, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 5, -5, 0, 0, 0, -3, 0, -5, 0, 1, 0, 0, -1, 1]

    tempxg = (
        _make_match_header(match_length=99999, jacoby=True, beaver=True)
        + _make_game_header(game_number=1)
        + _make_move_record(
            actif=1,
            position_i_raw=p1_pre_raw,
            position_end_raw=p1_played_raw,
            dice=(3, 1),
            moves=[
                (p1_best_raw, _xg_eval(0.55, 0.14, 0.01, 0.11, 0.005, 0.150), 2),
                (p1_played_raw, _xg_eval(0.53, 0.13, 0.01, 0.12, 0.005, 0.100), 2),
            ],
            err_move=0.050,
            analyze_m=3,
        )
        + _make_move_record(
            actif=-1,
            position_i_raw=bot_pre_p1_raw,
            position_end_raw=bot_played_mover_raw,
            dice=(4, 2),
            moves=[
                (bot_best_mover_raw, _xg_eval(0.60, 0.18, 0.01, 0.09, 0.004, 0.250), 2),
                (bot_played_mover_raw, _xg_eval(0.56, 0.16, 0.01, 0.10, 0.004, 0.170), 2),
            ],
            err_move=0.080,
            analyze_m=3,
        )
        + _make_game_footer()
        + _make_match_footer()
    )

    match = xg_compare.parse_xg_match(_pack_archive(tempxg))
    assert match["match_length"] == 0  # 99999 sentinel normalized to money (0)
    assert match["analysis_level"] == "xg-3ply"
    turns = match["games"][0]["turns"]
    assert len(turns) == 2

    # Every board_before, board_after, and checker_analysis board must have nonnegative bars.
    for turn in turns:
        assert turn["board_before"][0] >= 0 and turn["board_before"][25] >= 0
        assert turn["board_after"][0] >= 0 and turn["board_after"][25] >= 0
        for m in turn["checker_analysis"]:
            assert m["board"][0] >= 0 and m["board"][25] >= 0

    assert turns[0]["board_before"][0] == 2 and turns[0]["board_before"][25] == 1
    assert turns[1]["board_before"][0] == 2 and turns[1]["board_before"][25] == 1
    # Bot turn board_after is in P1 (user) frame: user bar=1 at [25], bot bar=1 at [0].
    assert turns[1]["board_after"][0] == 1 and turns[1]["board_after"][25] == 1

    # Checker errors match the played board despite signed bar cells in raw XG bytes.
    assert xg_compare.compute_checker_error(turns[0]) == pytest.approx(0.050, abs=1e-5)
    assert xg_compare.compute_checker_error(turns[1]) == pytest.approx(0.080, abs=1e-5)

    stats = xg_compare.compute_game_pr_stats(turns)
    assert stats["user_dec"] == 1
    assert stats["bot_dec"] == 1
    assert stats["user_err"] == pytest.approx(0.050, abs=1e-5)
    assert stats["bot_err"] == pytest.approx(0.080, abs=1e-5)
    assert stats["total_err"] == pytest.approx(0.130, abs=1e-5)
    assert stats["pr"] == pytest.approx((0.130 / 2) * 500.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Mid-game & end-of-game double/pass bug fix (:559 checker_error is None)
# ---------------------------------------------------------------------------


def test_midgame_and_endgame_double_pass_have_checker_error_none():
    """Both mid-game (cube->cube) and trailing double/pass turns must set checker_error=None."""
    board_a = [0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0]
    board_b = [0, -2, 0, 0, 0, 1, 4, 0, 2, 1, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0]

    tempxg = (
        _make_match_header(match_length=5, unicode_names=False, player1="AsciiP1", player2="AsciiP2")
        + _make_game_header(game_number=1, score1=0, score2=0, crawford=False)
        # Turn 1: regular user move (establishes board_after for next turn's board_before)
        + _make_move_record(
            actif=1,
            position_i_raw=board_a,
            position_end_raw=board_b,
            dice=(3, 1),
            moves=[(board_b, _xg_eval(0.54, 0.14, 0.01, 0.12, 0.005, 0.12), 2)],
            err_move=0.0,
            analyze_m=2,
        )
        # Turn 2: mid-game double/pass (followed immediately by another TS_CUBE record)
        + _make_cube_record(
            actif=-1,
            double=1,
            take=0,
            flag_double=0,  # should NOT have doubled -> non-zero err_cube
            equity_nd=0.35,
            equity_dt=0.25,
            equity_dp=1.00,
            err_cube=0.10,
            analyze_c=2,
        )
        # Turn 3: trailing end-of-game double/pass
        + _make_cube_record(
            actif=1,
            double=1,
            take=0,
            flag_double=2,
            equity_nd=0.90,
            equity_dt=1.15,
            equity_dp=1.00,
            err_cube=0.04,
            analyze_c=2,
        )
        + _make_game_footer(score1=1, score2=0, termination=0)
        + _make_match_footer(score1=1, score2=0)
    )

    match = xg_compare.parse_xg_match(_pack_archive(tempxg))
    assert match["player1"] == "AsciiP1"
    assert match["player2"] == "AsciiP2"
    assert match["analysis_level"] == "xg-2ply"

    turns = match["games"][0]["turns"]
    assert len(turns) == 3

    mid_dp = turns[1]
    end_dp = turns[2]
    assert mid_dp["cube_action"] == "double/pass"
    assert mid_dp["player"] == "bot"
    assert mid_dp["checker_error"] is None
    assert mid_dp["cube_error"] == pytest.approx(0.10)
    assert mid_dp["checker_analysis"] is None
    assert mid_dp["board_before"] == board_b

    assert end_dp["cube_action"] == "double/pass"
    assert end_dp["player"] == "user"
    assert end_dp["checker_error"] is None
    assert end_dp["cube_error"] == pytest.approx(0.04)
    assert end_dp["checker_analysis"] is None


# ---------------------------------------------------------------------------
# Multi-game parse_xg_match, flag_double=-1 (too good), and PR stats
# ---------------------------------------------------------------------------


def test_parse_xg_match_multi_game_and_too_good_flag():
    board = [0] * 26
    board[6] = 15
    board[19] = -15

    tempxg = (
        _make_match_header(match_length=5, player1="P1", player2="P2", jacoby=False, beaver=False)
        + _make_game_header(game_number=1, score1=0, score2=0, crawford=False)
        + _make_cube_record(
            actif=1,
            double=0,
            take=-1,
            flag_double=-1,  # too good to double -> should_double must be False
            equity_nd=1.25,
            equity_dt=1.40,
            equity_dp=1.00,
            err_cube=0.0,
            analyze_c=3,
        )
        + _make_move_record(
            actif=1,
            position_i_raw=board,
            position_end_raw=board,
            dice=(6, 5),
            moves=[(board, _xg_eval(0.95, 0.60, 0.02, 0.0, 0.0, 1.25), 2)],
            analyze_m=3,
        )
        + _make_game_footer(score1=4, score2=0)
        + _make_game_header(game_number=2, score1=4, score2=0, crawford=True)
        + _make_move_record(
            actif=2,  # XG-native actif=2 -> bot
            position_i_raw=board,
            position_end_raw=board,
            dice=(2, 1),
            moves=[(board, _xg_eval(0.50, 0.10, 0.0, 0.10, 0.0, 0.0), 2)],
            analyze_m=3,
        )
        + _make_game_footer(score1=5, score2=0)
        + _make_match_footer(score1=5, score2=0)
    )
    raw = _pack_archive(tempxg)

    with pytest.raises(ValueError, match="Expected exactly 1 game"):
        xg_compare.parse_xg_game(raw)

    match = xg_compare.parse_xg_match(raw)
    assert match["match_length"] == 5
    assert match["jacoby"] is False and match["beaver"] is False
    assert len(match["games"]) == 2

    g1, g2 = match["games"]
    assert (g1["game_number"], g1["score1_start"], g1["score2_start"], g1["crawford_apply"]) == (1, 0, 0, False)
    assert (g2["game_number"], g2["score1_start"], g2["score2_start"], g2["crawford_apply"]) == (2, 4, 0, True)

    ca = g1["turns"][0]["cube_analysis"]
    assert ca["should_double"] is False
    assert ca["optimal_action"] == "no double"
    assert g2["turns"][0]["player"] == "bot"


def test_compute_game_pr_stats_cube_ownership_and_idempotency():
    played_a = [0] * 26
    played_a[1] = 2
    played_b = [0] * 26
    played_b[2] = 2

    turns = [
        # Turn 1: user doubles and bot takes (bot blunder of 0.15 on take; user correct).
        # Afterwards bot owns the cube.
        {
            "player": "user",
            "cube_action": "double/take",
            "cube_analysis": {"equity_nd": 0.85, "equity_dt": 1.15, "equity_dp": 1.00},
            "board_after": played_a,
            "checker_analysis": [
                {"board": played_b, "equity": 0.40, "equity_diff": 0.0},
                {"board": played_a, "equity": 0.35, "equity_diff": -0.05},
            ],
        },
        # Turn 2: user has a cube_action recorded, but bot owns the cube -> NOT a user cube decision!
        {
            "player": "user",
            "cube_action": "no double",
            "cube_analysis": {"equity_nd": 0.40, "equity_dt": 0.70, "equity_dp": 1.00},
            "checker_analysis": None,
        },
        # Turn 3: bot redoubles and user drops (trivial take/pass because DT == DP -> opp_dec skipped).
        {
            "player": "bot",
            "cube_action": "double/pass",
            "cube_analysis": {"equity_nd": 0.90, "equity_dt": 1.0005, "equity_dp": 1.00},
            "checker_analysis": None,
        },
    ]

    stats1 = xg_compare.compute_game_pr_stats(turns)
    # Decisions:
    # - Turn 1 user cube (0.0 err), bot take (0.15 err), user checker (0.05 err)
    # - Turn 2 user has no cube access -> 0 decisions
    # - Turn 3 bot cube (0.0 err), user take/pass is trivial (|1.0005-1.0| < 0.001) -> skipped
    assert stats1["user_dec"] == 2
    assert stats1["user_err"] == pytest.approx(0.05)
    assert stats1["bot_dec"] == 2
    assert stats1["bot_err"] == pytest.approx(0.15)
    assert stats1["total_dec"] == 4
    assert stats1["total_err"] == pytest.approx(0.20)
    assert stats1["pr"] == pytest.approx((0.20 / 4) * 500.0)

    # Idempotent when called again on the same annotated turns.
    stats2 = xg_compare.compute_game_pr_stats(turns)
    assert stats2 == stats1

    # Empty game produces NaN PR.
    empty = xg_compare.compute_game_pr_stats([])
    assert empty["total_dec"] == 0 and math.isnan(empty["pr"])


@pytest.mark.external_data
def test_real_pilot_xg_file_if_present():
    sample = Path(__file__).resolve().parent.parent / "logs" / "xg_pilot" / "seed_1.xg"
    if not sample.exists():
        pytest.skip("local xg_pilot fixture not present")
    turns = xg_compare.parse_xg_game(sample.read_bytes())
    assert len(turns) > 0
    for t in turns:
        if t["board_before"] is not None:
            assert t["board_before"][0] >= 0 and t["board_before"][25] >= 0
        if t["board_after"] is not None:
            assert t["board_after"][0] >= 0 and t["board_after"][25] >= 0
        if t["cube_action"] == "double/pass":
            assert t["checker_error"] is None
    stats = xg_compare.compute_game_pr_stats(turns)
    assert stats["total_dec"] > 0
    assert math.isfinite(stats["pr"])
