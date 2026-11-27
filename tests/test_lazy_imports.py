# SPDX-License-Identifier: MPL-2.0
"""Verify the pure-Python API in a fresh interpreter with no extension."""
from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys


def test_pure_python_modules_do_not_import_the_extension():
    package = find_spec("neongammon")
    assert package is not None and package.origin is not None
    import_root = Path(package.origin).resolve().parent.parent
    script = """
import sys

class BlockNativeImport:
    attempted = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "bgbot_cpp":
            self.attempted = True
            raise ModuleNotFoundError("native extension disabled for this test")

sys.path.insert(0, sys.argv[1])
blocker = BlockNativeImport()
sys.meta_path.insert(0, blocker)

import neongammon
path_before = sys.path.copy()
from neongammon import data, settings, weights, text_export
from neongammon.settings import gnubg_path
from neongammon.weights import WeightConfig
from neongammon.text_export import export_history_to_txt

assert neongammon.WeightConfig is WeightConfig and callable(gnubg_path)
assert callable(export_history_to_txt)
assert settings is neongammon.settings
assert weights is neongammon.weights
assert text_export is neongammon.text_export
assert data.board_from_gnubg_position_string('A' * 20) == [0] * 26
assert sys.path == path_before
assert not blocker.attempted and "bgbot_cpp" not in sys.modules
assert "BgBotAnalyzer" in dir(neongammon)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(import_root)],
        capture_output=True, text=True, timeout=20,
    )
    assert completed.returncode == 0, completed.stderr


def test_stale_extension_explains_how_to_load_matching_version():
    import bgbot_cpp
    import neongammon

    paths = [
        str(Path(neongammon.__file__).resolve().parent.parent),
        str(Path(bgbot_cpp.__file__).resolve().parent),
    ]
    script = """
import sys
sys.path[:0] = sys.argv[1:]
import bgbot_cpp
del bgbot_cpp.MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE
from neongammon import BgBotAnalyzer
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, *paths],
        capture_output=True, text=True, timeout=20,
    )
    assert completed.returncode != 0
    assert "ImportError: NeonGammon requires a matching bgbot_cpp extension" in completed.stderr
    assert "MONEY_CONTACT_CUBE_LIFE_PIP_SLOPE" in completed.stderr
    assert "Rebuild" in completed.stderr and "reinstall" in completed.stderr
    assert bgbot_cpp.__file__ in completed.stderr
