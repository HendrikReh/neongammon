# SPDX-License-Identifier: MPL-2.0
"""Build the standalone C ABI and its matching reference before comparing them."""
import subprocess
import sys
from pathlib import Path

import pybind11


def test_mobile_scoring_matches_desktop():
    root = Path(__file__).resolve().parents[1]
    build = root / "build" / f"mobile-parity-{sys.implementation.cache_tag}"
    subprocess.run([
        "cmake", "-S", str(root / "mobile"), "-B", str(build), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DNEONGAMMON_MOBILE_REFERENCE_PYTHON=ON",
        "-DPYBIND11_FINDPYTHON=ON", f"-DPython_EXECUTABLE={sys.executable}",
        f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
    ], check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", "4"], check=True)
    # Keep previously imported extensions from other tests out of the reference.
    subprocess.run([sys.executable, str(root / "mobile/test_parity.py"), "--build", str(build)],
                   check=True)
