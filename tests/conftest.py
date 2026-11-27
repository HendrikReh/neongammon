# SPDX-License-Identifier: MPL-2.0
"""Pytest configuration: add local checkout paths when neongammon is not installed."""
from importlib.util import find_spec
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent

# In installed-wheel environments (such as cibuildwheel's `python -I -m pytest`),
# `neongammon` and `bgbot_cpp` are already resolvable from site-packages. Only
# prepend checkout directories when the module is not yet resolvable, and never
# import either module here (preserving `test_lazy_imports.py`).
for _mod, _subdir in (("neongammon", "python"), ("bgbot_cpp", "build")):
    if find_spec(_mod) is None:
        _candidate = _ROOT / _subdir
        if _candidate.is_dir():
            _path_str = str(_candidate)
            if _path_str not in sys.path:
                sys.path.insert(0, _path_str)
