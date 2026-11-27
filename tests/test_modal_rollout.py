# SPDX-License-Identifier: MPL-2.0
import contextlib
import importlib.util
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "modal_rollout.py"
ROOT = Path(__file__).resolve().parents[1]


def _load_modal_rollout_module():
    spec = importlib.util.spec_from_file_location("modal_rollout_test_mod", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def test_local_rollout_runs_without_modal_installed(tmp_path):
    positions = tmp_path / "positions"
    board = [0, 0, 1] + [0] * 21 + [-1, 0]
    positions.write_text(" ".join(map(str, board)) + "\n")
    output = tmp_path / "rollout"

    # Block `import modal` inside a fresh subprocess to simulate an environment
    # without modal installed.
    runner = (
        "import runpy, sys\n"
        "sys.modules['modal'] = None\n"
        f"sys.argv = [{str(SCRIPT)!r}, {str(positions)!r}, '--out', {str(output)!r}, "
        "'--local', '--threads', '1', '--n-trials', '36', '--checker-ply', '1']\n"
        f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-c", runner],
        check=False, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert len(output.read_text().splitlines()) == 1

    # Without --local, missing modal must fail with a clear message.
    remote_runner = (
        "import runpy, sys\n"
        "sys.modules['modal'] = None\n"
        f"sys.argv = [{str(SCRIPT)!r}, {str(positions)!r}, '--out', {str(tmp_path / 'r2')!r}, "
        "'--threads', '1', '--n-trials', '36', '--checker-ply', '1']\n"
        f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')\n"
    )
    remote = subprocess.run(
        [sys.executable, "-B", "-c", remote_runner],
        check=False, capture_output=True, text=True, timeout=30,
    )
    assert remote.returncode != 0
    assert "modal is required" in remote.stderr


def test_provenance_ignores_untracked_cpp_files():
    mod = _load_modal_rollout_module()
    cfg = dict(model="stage11", n_trials=36, checker_ply=1, truncation_depth=0, seed=42)
    before = mod._provenance(cfg, threads=1, local=False)

    untracked = ROOT / "cpp" / "_tmp_untracked_provenance_probe.txt"
    try:
        untracked.write_text("untracked scratch file\n")
        after = mod._provenance(cfg, threads=1, local=False)
    finally:
        untracked.unlink(missing_ok=True)

    assert after == before


def test_provenance_and_strategy_builder_honor_routing(monkeypatch):
    import bgbot_cpp
    from neongammon import STARTING_BOARD
    from neongammon.weights import WeightConfig, WeightConfigPair

    mod = _load_modal_rollout_module()
    cfg = dict(model="stage11", n_trials=36, checker_ply=1, truncation_depth=0, seed=42)
    base_weights = WeightConfigPair.from_model("stage11")
    alt_weights = replace(
        base_weights,
        routing={
            "contact_pairs": (2,) * 16,
            "snake": 19,
            "root_pins": (19,),
        },
    )

    h1 = mod._provenance(cfg, threads=1, local=True)
    monkeypatch.setattr(WeightConfig, "from_model", classmethod(lambda cls, name: alt_weights))
    h2 = mod._provenance(cfg, threads=1, local=True)
    assert h1 != h2

    strat = mod._build_strategy(bgbot_cpp, alt_weights)
    assert strat.select_nn_idx(list(STARTING_BOARD)) == 2
    assert strat.root_pinned() == [19]


def test_consume_remote_stream_counts_partial_tail_failure_accurately():
    mod = _load_modal_rollout_module()

    def run_stream(items, pending_count, chunk_size):
        ok = 0

        def record(_res):
            nonlocal ok
            ok += 1

        return mod._consume_remote_stream(
            items,
            pending_count,
            chunk_size,
            record,
            lambda: "",
            max_usd=0.0,
            usd_per_sec=0.0,
            get_counts=lambda: (ok, 0.0),
        )

    # 5 pending boards with chunk=4 -> chunks of size 4 and 1.
    # Tail chunk (1 board) fails: exact n_fail must be 1, not 4.
    n_fail, failed_chunks, stopped = run_stream(
        [[{"secs": 0.1}] * 4, RuntimeError("tail failed")],
        pending_count=5,
        chunk_size=4,
    )
    assert (n_fail, failed_chunks, stopped) == (1, 1, False)

    # First chunk (4 boards) fails, tail chunk (1 board) succeeds -> exact n_fail=4.
    n_fail, failed_chunks, stopped = run_stream(
        [RuntimeError("head failed"), [{"secs": 0.1}]],
        pending_count=5,
        chunk_size=4,
    )
    assert (n_fail, failed_chunks, stopped) == (4, 1, False)

    # Both chunks fail -> exact n_fail=5, not 8.
    n_fail, failed_chunks, stopped = run_stream(
        [RuntimeError("c0"), RuntimeError("c1")],
        pending_count=5,
        chunk_size=4,
    )
    assert (n_fail, failed_chunks, stopped) == (5, 2, False)


def test_remote_failures_exit_nonzero(tmp_path, monkeypatch):
    mod = _load_modal_rollout_module()
    positions = tmp_path / "positions"
    boards = [
        [0, 0, 1] + [0] * 21 + [-1, 0],
        [0, 0, 0, 1] + [0] * 20 + [-1, 0],
        [0, 0, 0, 0, 1] + [0] * 19 + [-1, 0],
    ]
    positions.write_text("".join(" ".join(map(str, b)) + "\n" for b in boards))
    output = tmp_path / "rollout"

    class FakeMethod:
        def map(self, chunks, **_kwargs):
            # First chunk (2 boards) succeeds, partial tail chunk (1 board) fails.
            yield [
                {
                    "board": list(b),
                    "probs": [0.5, 0.0, 0.0, 0.0, 0.0],
                    "equity": 0.0,
                    "se": 0.01,
                    "prob_se": [0.01] * 5,
                    "secs": 0.05,
                }
                for b in chunks[0]
            ]
            yield RuntimeError("tail chunk failed")

    class FakeRoller:
        def __init__(self, **_kwargs):
            self.rollout_many = FakeMethod()
            self.rollout = FakeMethod()

        @classmethod
        def with_options(cls, **_kwargs):
            return cls

    fake_modal = SimpleNamespace(enable_output=contextlib.nullcontext)
    fake_app = SimpleNamespace(run=contextlib.nullcontext)

    monkeypatch.setattr(mod, "modal", fake_modal)
    monkeypatch.setattr(mod, "app", fake_app)
    monkeypatch.setattr(mod, "Roller", FakeRoller)
    monkeypatch.setattr(
        sys, "argv",
        [str(SCRIPT), str(positions), "--out", str(output),
         "--chunk", "2", "--threads", "1", "--n-trials", "36", "--checker-ply", "1"],
    )

    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1
    assert len(output.read_text().splitlines()) == 2
