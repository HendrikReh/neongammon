# SPDX-License-Identifier: MPL-2.0
"""Print the sdist's exact model paths as a CMake list for wheel installation."""
from pathlib import Path
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

with Path("pyproject.toml").open("rb") as config:
    manifest = tomllib.load(config)["tool"]["scikit-build"]["sdist"]["include"]

paths = [path for path in manifest if path.startswith("models/")]
if not paths:
    sys.exit("No model weights in the sdist manifest")
for path in paths:
    if not Path(path).is_file():
        sys.exit(f"Missing model weight listed in the sdist manifest: {path}")
print(";".join(paths))
