# SPDX-License-Identifier: MPL-2.0
"""Run explicitly in cibuildwheel's installed-wheel test environment."""
from importlib.metadata import distribution
import math
from pathlib import Path
import sysconfig

import bgbot_cpp
import neongammon as ng
from neongammon.weights import bearoff_db_path


def test_modules_and_model_files_belong_to_the_installed_wheel():
    dist = distribution("neongammon")
    installed = {dist.locate_file(entry).resolve() for entry in dist.files}
    site_packages = Path(sysconfig.get_path("platlib")).resolve()
    for module in (ng, bgbot_cpp):
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(site_packages) and path in installed

    assert Path(ng.__file__).with_name("py.typed").resolve() in installed
    assert ng.TrialEvalConfig is bgbot_cpp.TrialEvalConfig
    assert set(ng.__all__) <= set(dir(ng))
    assert all(getattr(ng, name) is not None for name in ng.__all__)

    db = bearoff_db_path()
    assert db is not None
    assert Path(db).resolve() in installed

    # A positive EPC requires the bundled bearoff database to load successfully.
    one_checker_each = [0, 1] + [0] * 22 + [-1, 0]
    assert ng.BgBotAnalyzer().epc(one_checker_each) > 0


def test_wheel_bundles_exact_weights_for_supported_models():
    required = set()
    for model in (ng.PRODUCTION_MODEL, "stage9"):
        weights = ng.WeightConfig.from_model(model)
        weights.validate()
        required.update(Path(path).resolve() for path in weights.weight_paths_list)
        analyzer = ng.BgBotAnalyzer(weights=weights, eval_level="1ply")
        result = analyzer.checker_play(ng.STARTING_BOARD, 3, 1)
        assert result.moves and all(math.isfinite(move.equity) for move in result.moves)

    dist = distribution("neongammon")
    model_dir = Path("neongammon/_assets/models")
    bundled = {
        dist.locate_file(entry).resolve()
        for entry in dist.files if model_dir in entry.parents
    }
    assert bundled == required
