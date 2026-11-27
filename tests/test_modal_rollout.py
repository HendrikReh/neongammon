# SPDX-License-Identifier: MPL-2.0
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("modal")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "modal_rollout.py"


@pytest.fixture
def rollout(tmp_path):
    positions = tmp_path / "positions"
    boards = [[0, 0, 1] + [0] * 21 + [-1, 0], [0, 0, 0, 1] + [0] * 20 + [-1, 0]]
    positions.write_text("".join(" ".join(map(str, board)) + "\n" for board in boards))
    output = tmp_path / "rollout"
    metadata = tmp_path / "rollout.meta.jsonl"

    def run(*args):
        return subprocess.run(
            [sys.executable, "-B", str(SCRIPT), str(positions), "--out", str(output),
             "--local", "--threads", "1", "--n-trials", "36", "--checker-ply", "1",
             *args],
            check=False, capture_output=True, text=True, timeout=30,
        )

    started = run("--limit", "1")
    assert started.returncode == 0, started.stderr
    assert len(output.read_text().splitlines()) == 1
    return run, output, metadata


def test_local_rollout_resumes_with_the_same_provenance(rollout):
    run, output, metadata = rollout
    completed = run()
    assert completed.returncode == 0, completed.stderr
    assert len(output.read_text().splitlines()) == 2
    assert len(metadata.read_text().splitlines()) == 2


def test_local_rollout_refuses_mixed_settings(rollout):
    run, output, metadata = rollout
    before = output.read_bytes(), metadata.read_bytes()
    changed = run("--n-trials", "72")
    assert changed.returncode != 0
    assert "provenance" in changed.stderr
    assert (output.read_bytes(), metadata.read_bytes()) == before


def test_local_rollout_refuses_a_row_without_committed_metadata(rollout):
    run, output, metadata = rollout
    metadata.write_bytes(b"")
    before = output.read_bytes()
    interrupted = run()
    assert interrupted.returncode != 0
    assert "incomplete" in interrupted.stderr
    assert output.read_bytes() == before
    assert metadata.read_bytes() == b""


def test_local_rollout_detects_a_changed_label_for_the_same_board(rollout):
    run, output, metadata = rollout
    old_row = output.read_text().split()
    changed = " ".join(old_row[:26] + ["0.5000", "0.0000", "0.0000", "0.0000", "0.0000"]) + "\n"
    output.write_text(changed)
    before = metadata.read_bytes()
    resumed = run()
    assert resumed.returncode != 0
    assert "label/metadata mismatch" in resumed.stderr
    assert output.read_text() == changed
    assert metadata.read_bytes() == before
