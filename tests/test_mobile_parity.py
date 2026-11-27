# SPDX-License-Identifier: MPL-2.0
"""Build the standalone C ABI and its matching reference before comparing them."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _find_build_tool(tool: str) -> str | None:
    venv_tool = Path(sys.executable).parent / tool
    if venv_tool.is_file() and os.access(venv_tool, os.X_OK):
        return str(venv_tool)
    return shutil.which(tool)


def test_mobile_scoring_matches_desktop(tmp_path):
    pybind11 = pytest.importorskip("pybind11")
    root = Path(__file__).resolve().parents[1]
    if not (root / "mobile").is_dir():
        pytest.skip("mobile parity requires the mobile source tree")
    tools = {}
    for tool in ("cmake", "ninja"):
        found = _find_build_tool(tool)
        if found is None:
            pytest.skip(f"mobile parity requires {tool}")
        tools[tool] = found

    env = os.environ.copy()
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"

    build = tmp_path / "mobile"
    subprocess.run([
        tools["cmake"], "-S", str(root / "mobile"), "-B", str(build), "-G", "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={tools['ninja']}",
        "-DCMAKE_BUILD_TYPE=Release", "-DNEONGAMMON_MOBILE_REFERENCE_PYTHON=ON",
        "-DPYBIND11_FINDPYTHON=ON", f"-DPython_EXECUTABLE={sys.executable}",
        f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
    ], check=True, env=env)
    subprocess.run([tools["cmake"], "--build", str(build), "--parallel", "4"], check=True, env=env)
    subprocess.run([str(build / "test_cube_budget")], check=True, env=env)
    # Keep previously imported extensions from other tests out of the reference.
    subprocess.run([sys.executable, str(root / "mobile/test_parity.py"), "--build", str(build)],
                   check=True, env=env)
