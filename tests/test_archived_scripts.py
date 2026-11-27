# SPDX-License-Identifier: MPL-2.0
"""Smoke-test archived CLIs from outside the source checkout."""

from pathlib import Path
import subprocess
import sys

import pytest


ARCHIVE = Path(__file__).resolve().parents[1] / "scripts" / "archive"


@pytest.mark.skipif(not ARCHIVE.is_dir(), reason="archived scripts require a source checkout")
@pytest.mark.parametrize("script", [
    "diagnostics/test_cube_decision.py",
    "diagnostics/test_evaluate_probs.py",
    "historical/run_benchmark.py",
    "pasko/benchmark_pasko.py",
    "stage11/run_s11_backgame_td.py",
])
def test_archived_cli_finds_its_checkout(script, tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    result = subprocess.run(
        [sys.executable, "-B", str(ARCHIVE / script), "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
