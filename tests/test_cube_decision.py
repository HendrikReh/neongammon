# SPDX-License-Identifier: MPL-2.0
"""BgBotAnalyzer.cube_decision equals cube_action without the probabilities.

At 2-ply and deeper cube_action runs a second, cubeless N-ply search for its
probs and cubeless equity after the cube tree has decided. cube_decision
skips that search (``decision_only`` in cube_decision_nply_unified); its
equities and flags must equal cube_action's, and the simulation, which reads
only the flags, uses it and records the same games.
"""

from dataclasses import asdict

import pytest

from neongammon import BgBotAnalyzer, STARTING_BOARD, batch_cube_action_nply
from neongammon.simulation import GameConfig, simulate_game

from .batch_positions import random_positions

SNAKE = [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 3, 3, 2, 2, 1, -7, -7, 0]
BEAROFF = [0, 3, 3, 2, 3, 2, 2] + [0] * 12 + [-2, -3, -2, -3, -2, -2] + [0]
CONTEXTS = [
    dict(),
    dict(jacoby=False, beaver=False),
    dict(cube_value=2, cube_owner="player"),
    dict(cube_value=2, cube_owner="player", max_cube_value=4),
    dict(away1=5, away2=3),
    dict(away1=4, away2=1),
    dict(away1=1, away2=4, is_crawford=True),
]
FIELDS = ("equity_nd", "equity_dt", "equity_dp", "should_double", "should_take",
          "optimal_equity", "optimal_action", "eval_level", "is_beaver")


BOARDS = random_positions(5, seed=772, with_dice=False) + [SNAKE, BEAROFF]


@pytest.mark.parametrize("level", ["1ply", "2ply", "3ply"])
def test_cube_decision_equals_cube_action(level):
    a = BgBotAnalyzer(eval_level=level, parallel_threads=2)
    for i, board in enumerate(BOARDS):
        for kw in CONTEXTS:
            full = asdict(a.cube_action(board, **kw))
            lean = asdict(a.cube_decision(board, **kw))
            assert {k: lean[k] for k in FIELDS} == {k: full[k] for k in FIELDS}, (level, i, kw)
            assert set(lean) == set(FIELDS)


def test_cube_decision_equals_cube_action_at_four_ply():
    a = BgBotAnalyzer(eval_level="4ply", parallel_threads=4)
    for board, kw in ((BOARDS[0], CONTEXTS[0]), (SNAKE, CONTEXTS[0]), (BOARDS[1], CONTEXTS[4])):
        full, lean = asdict(a.cube_action(board, **kw)), asdict(a.cube_decision(board, **kw))
        assert {k: lean[k] for k in FIELDS} == {k: full[k] for k in FIELDS}


def test_last_legal_take_has_dead_cube_equity():
    analyzer = BgBotAnalyzer(eval_level="1ply", parallel_threads=1)
    for beaver in (False, True):
        cube = analyzer.cube_action(
            STARTING_BOARD, max_cube_value=2, jacoby=False, beaver=beaver,
        )
        assert cube.equity_dt == pytest.approx(2 * cube.cubeless_equity)
        assert not cube.is_beaver


@pytest.mark.parametrize("level", ["1ply", "2ply", "3ply"])
def test_cube_cap_disallows_beavers_in_scalar_and_batch(level):
    analyzer = BgBotAnalyzer(eval_level=level, parallel_threads=1)
    capped = dict(board=STARTING_BOARD, max_cube_value=2, jacoby=False)
    # Include a clearly losing race so deeper searches also exercise the
    # beaver branch. The opponent is already within one roll of bearing off.
    losing_race = dict(capped, board=[0, 15] + [0] * 22 + [-1, 0])
    assert analyzer.cube_decision(losing_race["board"], jacoby=False, beaver=True).is_beaver
    expected = [analyzer.cube_decision(**position, beaver=False)
                for position in (capped, losing_race)]
    assert [analyzer.cube_decision(**position, beaver=True)
            for position in (capped, losing_race)] == expected
    assert batch_cube_action_nply(
        [capped, losing_race], eval_level=level, n_threads=2,
        beaver=True, decision_only=True,
    ) == expected


class _CubeActionOnly:
    """A policy without cube_decision: the simulation falls back to cube_action."""

    def __init__(self, analyzer):
        self._a = analyzer

    def cube_action(self, *args, **kw):
        return self._a.cube_action(*args, **kw)

    def checker_play(self, *args, **kw):
        return self._a.checker_play(*args, **kw)


@pytest.mark.parametrize("config", [GameConfig(), GameConfig(match_length=5, score1=2, score2=3)])
def test_seeded_simulation_records_do_not_change(config):
    a = BgBotAnalyzer(eval_level="2ply", parallel_threads=2)
    for seed in (3, 11):
        lean = simulate_game(seed=seed, config=config, players=(a, a))
        full = simulate_game(seed=seed, config=config,
                             players=(_CubeActionOnly(a), _CubeActionOnly(a)))
        assert lean == full, seed
