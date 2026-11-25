import math

import pytest

import neongammon as ng


def test_weights_favor_better_moves_and_exclude_large_errors():
    from neongammon.difficulty import _choice_weights

    assert _choice_weights([0, 0.05, 0.10, 0.11], 0.10, 0.05) == pytest.approx(
        [1, math.exp(-1), math.exp(-2), 0]
    )
    assert _choice_weights([0, 0, 0.05], 0.10, 0) == [1, 0, 0]


def test_full_strength_depths():
    policy = ng.DifficultyPolicy(ng.Difficulty.FULL_STRENGTH)
    analyzer = ng.BgBotAnalyzer(eval_level="2ply", parallel_threads=1)
    context = {"away1": 5, "away2": 5}
    best = ng.BgBotAnalyzer(eval_level="3ply", parallel_threads=1).checker_play(ng.STARTING_BOARD, 4, 3, **context).moves[0]

    assert policy.select_checker_move(ng.STARTING_BOARD, 4, 3, **context) == best.board
    assert policy.cube_decision(ng.STARTING_BOARD, **context) == analyzer.cube_decision(
        ng.STARTING_BOARD, **context
    )


def test_seeded_beginner_replays_through_match_simulator():
    def play():
        return ng.simulate_match(seed=741, players=(
            ng.DifficultyPolicy("beginner", seed=519),
            ng.DifficultyPolicy("intermediate", seed=520),
        ))

    match = play()
    assert match == play()
    assert match["match_length"] == 5
    assert match["final_scores"][match["winner"] - 1] == 5


def test_beginner_keeps_forced_checker_play_and_crawford_cube_rules():
    policy = ng.DifficultyPolicy("beginner", seed=1)
    board = [0] * 26
    board[1], board[24] = 1, -1

    assert policy.select_checker_move(board, 6, 5, away1=1, away2=3, is_crawford=True)[1] == 0
    assert not policy.cube_decision(board, away1=1, away2=3, is_crawford=True).should_double


@pytest.mark.parametrize("away", [0, 5])
def test_beginner_cannot_double_opponents_cube_with_mixed_case_owner(away):
    policy = ng.DifficultyPolicy("beginner", seed=1)
    board = [0, 3, 3, 3, 3, 2, 0, 1, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, -3, -3, -3, -3, -3, 0]

    assert not policy.cube_decision(
        board, cube_owner="Opponent", away1=away, away2=away, beaver=False,
    ).should_double

def test_public_reseed_replays_policy_stream():
    policy = ng.DifficultyPolicy("beginner", seed=35)
    def sequence():
        return [policy.select_checker_move(ng.STARTING_BOARD, 4, 3) for _ in range(10)]
    expected = sequence()
    policy.reseed(35)
    assert sequence() == expected
