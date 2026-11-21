import pytest


def test_cube_loss_charges_the_decision_maker():
    from scripts.calibrate_difficulty import cube_loss

    # A double gains 0.10 with correct response (pass), while a take gives
    # away 0.20 from the responder's perspective.
    equities = (0.90, 1.20, 1.00)
    assert cube_loss("double", False, equities) == pytest.approx(0.10)
    assert cube_loss("double", True, equities) == 0
    assert cube_loss("take", True, equities) == pytest.approx(0.20)
    assert cube_loss("take", False, equities) == 0


def test_summary_pools_decisions_and_resamples_complete_matches():
    from scripts.calibrate_difficulty import summarize

    rows = [
        {"won": False, "checker": [0.12], "double": [], "take": []},
        {"won": True, "checker": [0.00, 0.00], "double": [0.04], "take": []},
    ]
    result = summarize(rows)
    assert result["pr"] == 20
    assert result["decisions"] == 4
    assert result["blunders"] == 1
    assert result["ci95"] == [pytest.approx(20 / 3), 60]
