# SPDX-License-Identifier: MPL-2.0
"""Tests for runtime_info, thread-budget observer, __version__, lazy routing, ABI check, and CLI (#119, #120)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
import sys

import pytest

import neongammon as ng
from neongammon import (
    BgBotAnalyzer,
    STARTING_BOARD,
    ThreadBudgetEvent,
    observe_thread_budgets,
    runtime_info,
)


def _import_root() -> Path:
    package = find_spec("neongammon")
    assert package is not None and package.origin is not None
    return Path(package.origin).resolve().parent.parent


def _run_isolated(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-c", script, *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_pure_python_lazy_exports_and_sl_train_do_not_load_native(tmp_path):
    root = _import_root()
    weights_file = Path(__file__).resolve().parent.parent / "models" / "sl_s9_race_race.weights.best"
    out_file = tmp_path / "copy.weights"
    script = """
import sys

class BlockNativeImport:
    attempted = False
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "bgbot_cpp":
            self.attempted = True
            raise ModuleNotFoundError("native extension disabled")

sys.path.insert(0, sys.argv[1])
blocker = BlockNativeImport()
sys.meta_path.insert(0, blocker)

import neongammon
from neongammon import (
    STARTING_BOARD,
    invert_probs,
    __version__,
    runtime_info,
    observe_thread_budgets,
    ThreadBudgetEvent,
)
from neongammon import sl_train

assert isinstance(__version__, str) and len(__version__) >= 5
assert len(STARTING_BOARD) == 26
assert invert_probs([0.6, 0.2, 0.05, 0.1, 0.01]) == [0.4, 0.1, 0.01, 0.2, 0.05]

# NumPy weight I/O in sl_train must work without bgbot_cpp.
layers = sl_train.read_weights(sys.argv[2])
sl_train.write_weights(sys.argv[3], *layers)

assert not blocker.attempted and "bgbot_cpp" not in sys.modules
"""
    completed = _run_isolated(script, str(root), str(weights_file), str(out_file))
    assert completed.returncode == 0, completed.stderr
    assert out_file.read_bytes() == weights_file.read_bytes()


def test_abi_version_mismatch_produces_actionable_error_everywhere():
    import bgbot_cpp

    paths = [
        str(_import_root()),
        str(Path(bgbot_cpp.__file__).resolve().parent),
    ]
    for target_call in (
        "from neongammon import BgBotAnalyzer",
        "from neongammon.matchinfo import take_points_dead_cube; take_points_dead_cube(5, 3)",
        "from neongammon.matchinfo import gammon_prices; gammon_prices(5, 3)",
        "from neongammon.cube_points import cube_action_points; cube_action_points((0.5, 0.1, 0.01, 0.1, 0.01), 'centered', 1, 5, 3)",
    ):
        script = f"""
import sys
sys.path[:0] = sys.argv[1:]
import bgbot_cpp
bgbot_cpp.ABI_VERSION = 999
{target_call}
"""
        completed = _run_isolated(script, *paths)
        assert completed.returncode != 0, target_call
        assert "ImportError: NeonGammon requires a matching bgbot_cpp extension" in completed.stderr
        assert "ABI_VERSION=999" in completed.stderr
        assert "Rebuild" in completed.stderr and "reinstall" in completed.stderr


def test_runtime_info_contract_and_json_serializability():
    info = runtime_info()
    # Must be JSON-serializable cleanly.
    encoded = json.dumps(info)
    decoded = json.loads(encoded)
    assert decoded["version"] == ng.__version__
    assert decoded["native_available"] is True
    assert decoded["abi_version"] == decoded["expected_abi_version"] >= 1
    assert isinstance(decoded["accelerate_enabled"], bool)
    assert decoded["production_model"] == ng.PRODUCTION_MODEL
    assert decoded["max_away"] == 25
    assert decoded["serial_cubeful_1ply"] is True
    assert decoded["supports_thread_budget_observer"] is True
    assert set(decoded["fixed_depth_levels"]) == {"1ply", "2ply", "3ply", "4ply"}


def test_observe_thread_budgets_covers_cubeful_and_cubeless_multipy():
    # 1. Cubeful 2-ply checker play -> checker_play_nply
    cf2 = BgBotAnalyzer(eval_level="2ply", cubeful=True, parallel_threads=2)
    with observe_thread_budgets() as cf_events:
        res_cf = cf2.checker_play(list(STARTING_BOARD), 3, 1, beaver=False)
    assert res_cf.moves
    assert any(
        e.operation == "checker_play_nply" and e.n_threads == 2 and e.plies == 2 and e.cubeful is True
        for e in cf_events
    ), cf_events

    # 2. Cubeless 2-ply checker play -> multipy_evaluate (MultiPlyStrategy native budget!)
    cl2 = BgBotAnalyzer(eval_level="2ply", cubeful=False, parallel_threads=2)
    with observe_thread_budgets() as cl_events:
        res_cl = cl2.checker_play(list(STARTING_BOARD), 3, 1, beaver=False)
    assert res_cl.moves
    assert cl_events and all(isinstance(e, ThreadBudgetEvent) for e in cl_events)
    assert any(
        e.operation == "multipy_evaluate" and 1 <= e.n_threads <= 2 and e.plies == 2 and e.cubeful is False
        for e in cl_events
    ), cl_events

    # 3. Cubeless 2-ply with parallel_threads=1 reports budget 1.
    cl2_serial = BgBotAnalyzer(eval_level="2ply", cubeful=False, parallel_threads=1)
    with observe_thread_budgets() as serial_events:
        cl2_serial.checker_play(list(STARTING_BOARD), 3, 1, beaver=False)
    assert serial_events and all(e.n_threads == 1 for e in serial_events)

    # 4. Cubeful 1-ply checker play -> checker_play_1ply with n_threads=1
    cf1 = BgBotAnalyzer(eval_level="1ply", cubeful=True, parallel_threads=4)
    with observe_thread_budgets() as cf1_events:
        cf1.checker_play(list(STARTING_BOARD), 3, 1, beaver=False)
    assert any(e.operation == "checker_play_1ply" and e.n_threads == 1 for e in cf1_events)

    # 5. Cube action at 2-ply -> cube_decision_nply
    with observe_thread_budgets() as cube_events:
        cf2.cube_action(list(STARTING_BOARD), beaver=False)
    assert any(e.operation == "cube_decision_nply" and e.n_threads == 2 for e in cube_events)

    # 6. Nested observers + real-time callback + per-thread isolation.
    streamed = []
    with observe_thread_budgets(streamed.append) as outer:
        with observe_thread_budgets() as inner:
            cf2.cube_action(list(STARTING_BOARD), beaver=False)
    assert len(outer) == len(inner) == len(streamed) >= 1

    # Thread isolation: concurrent observers do not leak into each other.
    def _worker(threads: int):
        an = BgBotAnalyzer(eval_level="2ply", cubeful=True, parallel_threads=threads)
        with observe_thread_budgets() as evs:
            an.checker_play(list(STARTING_BOARD), 3, 1, beaver=False)
        return [e.n_threads for e in evs if e.operation == "checker_play_nply"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        r1, r2 = list(pool.map(_worker, [1, 3]))
    assert r1 == [1] and r2 == [3]


@pytest.mark.parametrize(
    "level,kwargs",
    [
        ("truncated1", {"n_trials": 36}),
        ("truncated1", {"prefilter_threshold": 0.1}),
        ("truncated2", {"n_trials": 1296}),
        ("truncated2", {"truncation_depth": 5}),
        ("truncated2", {"decision_ply": 1}),
        ("truncated2", {"filter_max_moves": 3}),
        ("truncated3", {"late_ply": 1}),
        ("truncated3", {"target_se": 0.01}),
        ("1ply", {"n_trials": 72}),
        ("2ply", {"truncation_depth": 5}),
    ],
)
def test_fixed_presets_reject_ignored_rollout_kwargs(level, kwargs):
    with pytest.raises(ValueError, match="eval_level='rollout'"):
        BgBotAnalyzer(eval_level=level, **kwargs)


def test_fixed_presets_still_accept_honored_kwargs():
    # truncated2 honors seed, parallel_threads, prefilter_threshold, cubeful_trial_moves.
    an = BgBotAnalyzer(
        eval_level="truncated2",
        parallel_threads=1,
        seed=7,
        prefilter_threshold=0.2,
        cubeful_trial_moves=False,
    )
    assert an._prefilter_threshold == 0.2
    # Fixed depth levels keep backward-compatible filter/seed kwargs used by probes.
    an2 = BgBotAnalyzer(
        eval_level="2ply",
        parallel_threads=1,
        filter_max_moves=1,
        filter_threshold=0.001,
        seed=42,
    )
    assert an2._eval_level == "2ply"


def test_cli_info_probe_and_eval():
    import bgbot_cpp

    env_paths = f"{Path(bgbot_cpp.__file__).resolve().parent}:{_import_root()}"
    env = {"PYTHONPATH": env_paths}

    def run_cli(*cli_args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "neongammon", *cli_args],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

    info_proc = run_cli("info", "--json")
    assert info_proc.returncode == 0, info_proc.stderr
    info_json = json.loads(info_proc.stdout)
    assert info_json["version"] == ng.__version__
    assert info_json["native_available"] is True

    probe_proc = run_cli("probe", "--level", "2ply", "--threads", "2", "--json")
    assert probe_proc.returncode == 0, probe_proc.stderr
    probe_json = json.loads(probe_proc.stdout)
    assert probe_json["ok"] is True
    assert probe_json["observed_thread_budgets"]

    eval_cube = run_cli("eval", "--level", "1ply", "--json")
    assert eval_cube.returncode == 0, eval_cube.stderr
    cube_json = json.loads(eval_cube.stdout)
    assert cube_json["optimal_action"] == "No Double"

    eval_checker = run_cli(
        "eval", "--gnubg-id", "4HPwATDgc/ABMA", "--dice", "3", "1", "--level", "1ply", "--top", "3", "--json"
    )
    assert eval_checker.returncode == 0, eval_checker.stderr
    checker_json = json.loads(eval_checker.stdout)
    assert checker_json["board"] == STARTING_BOARD
    assert len(checker_json["moves"]) == 3


def test_sdist_contents_verifier_enforces_allowlist_and_bounds(tmp_path):
    import io
    import tarfile

    from tests import sdist_contents

    root = Path(__file__).resolve().parent.parent
    pyproject_path = root / "pyproject.toml"
    pyproject = sdist_contents.tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    expected_includes = list(pyproject["tool"]["scikit-build"]["sdist"]["include"])

    def write_tar(name: str, extra_files: list[str] = (), drop_files: set[str] = frozenset()) -> Path:
        archive = tmp_path / name
        rel_files = (list(sdist_contents.REQUIRED_BUILD_FILES) + expected_includes + list(extra_files))
        with tarfile.open(archive, "w:gz") as tf:
            for rel in rel_files:
                if rel in drop_files:
                    continue
                payload = b"x"
                info = tarfile.TarInfo(name=f"neongammon-0.0.0/{rel}")
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
        return archive

    ok_tar = write_tar("ok.tar.gz")
    summary = sdist_contents.verify_sdist(ok_tar, pyproject_path)
    assert summary["included_assets"] == len(expected_includes)

    missing_tar = write_tar("missing.tar.gz", drop_files={"CMakeLists.txt"})
    with pytest.raises(AssertionError, match="missing required build files"):
        sdist_contents.verify_sdist(missing_tar, pyproject_path)

    leaked_model = write_tar("leaked_model.tar.gz", extra_files=["models/extra.weights"])
    with pytest.raises(AssertionError, match="leaked unexpected models/ or data/"):
        sdist_contents.verify_sdist(leaked_model, pyproject_path)

    leaked_tests = write_tar("leaked_tests.tar.gz", extra_files=["tests/test_x.py"])
    with pytest.raises(AssertionError, match="leaked excluded directories"):
        sdist_contents.verify_sdist(leaked_tests, pyproject_path)

    leaked_tdinit = write_tar("leaked_tdinit.tar.gz", extra_files=["foo_tdinit.bin"])
    with pytest.raises(AssertionError, match="leaked tdinit"):
        sdist_contents.verify_sdist(leaked_tdinit, pyproject_path)
