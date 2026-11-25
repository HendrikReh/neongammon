# SPDX-License-Identifier: MPL-2.0
"""Run explicitly in cibuildwheel's installed-wheel test environment."""
from importlib.metadata import distribution
from pathlib import Path
import sysconfig

import bgbot_cpp
import neongammon as ng
from neongammon.weights import bearoff_db_path, default_weights


def test_modules_and_model_files_belong_to_the_installed_wheel():
    dist = distribution("neongammon")
    installed = {dist.locate_file(entry).resolve() for entry in dist.files}
    site_packages = Path(sysconfig.get_path("platlib")).resolve()
    for module in (ng, bgbot_cpp):
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(site_packages) and path in installed

    weights = default_weights()
    weights.validate()
    db = bearoff_db_path()
    assert db is not None
    for path in [*weights.weight_paths_list, db]:
        assert Path(path).resolve() in installed

    # A positive EPC requires the bundled bearoff database to load successfully.
    one_checker_each = [0, 1] + [0] * 22 + [-1, 0]
    assert ng.BgBotAnalyzer().epc(one_checker_each) > 0
