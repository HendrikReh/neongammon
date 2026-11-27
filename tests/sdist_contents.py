# SPDX-License-Identifier: MPL-2.0
"""Verify source distribution (sdist) manifest, allowlist boundaries, and size.

Run directly against ``dist/*.tar.gz`` in CI (``python tests/sdist_contents.py``)
or via pytest when ``NEONGAMMON_SDIST_PATH`` is set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tarfile

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

MAX_SDIST_BYTES = 100 * 1024 * 1024  # PyPI single-file limit (100 MiB)

REQUIRED_BUILD_FILES: tuple[str, ...] = (
    "pyproject.toml",
    "CMakeLists.txt",
    "cpp/CMakeLists.txt",
    "cmake/model_weights.py",
    "LICENSE",
    "README.md",
    "python/neongammon/__init__.py",
    "python/neongammon/__main__.py",
    "python/neongammon/py.typed",
    "cpp/pybind/bindings.cpp",
)

FORBIDDEN_TOP_DIRS: frozenset[str] = frozenset((
    "backgame_ref_positions",
    "assets",
    "scripts",
    "tests",
    "logs",
))


def verify_sdist(sdist_path: Path, pyproject_path: Path) -> dict[str, object]:
    if not sdist_path.is_file():
        raise AssertionError(f"sdist archive not found: {sdist_path}")
    size_bytes = sdist_path.stat().st_size
    if size_bytes <= 0 or size_bytes >= MAX_SDIST_BYTES:
        raise AssertionError(
            f"sdist size {size_bytes / (1024 * 1024):.2f} MiB exceeds bound "
            f"({MAX_SDIST_BYTES / (1024 * 1024):.0f} MiB)"
        )

    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    expected_includes = set(pyproject["tool"]["scikit-build"]["sdist"]["include"])

    with tarfile.open(sdist_path, "r:gz") as tf:
        file_members = [m.name for m in tf.getmembers() if m.isfile()]

    if not file_members:
        raise AssertionError("sdist archive contains no files")

    prefixes = {Path(name).parts[0] for name in file_members}
    if len(prefixes) != 1:
        raise AssertionError(f"sdist expected single top-level directory, found {sorted(prefixes)}")
    rel_paths = {"/".join(Path(name).parts[1:]) for name in file_members}

    missing_build = sorted(set(REQUIRED_BUILD_FILES) - rel_paths)
    if missing_build:
        raise AssertionError(f"sdist missing required build files: {missing_build}")

    missing_includes = sorted(expected_includes - rel_paths)
    if missing_includes:
        raise AssertionError(f"sdist missing sdist.include entries: {missing_includes}")

    # Enforce allowlist boundary on models/ and data/.
    data_and_models = {p for p in rel_paths if p.startswith(("models/", "data/"))}
    unexpected_assets = sorted(data_and_models - expected_includes)
    if unexpected_assets:
        raise AssertionError(
            f"sdist leaked unexpected models/ or data/ files: {unexpected_assets[:10]}"
        )

    leaked_dirs = sorted(
        p for p in rel_paths if Path(p).parts and Path(p).parts[0] in FORBIDDEN_TOP_DIRS
    )
    if leaked_dirs:
        raise AssertionError(f"sdist leaked excluded directories: {leaked_dirs[:10]}")

    leaked_tdinit = sorted(p for p in rel_paths if "_tdinit" in p)
    if leaked_tdinit:
        raise AssertionError(f"sdist leaked tdinit files: {leaked_tdinit[:10]}")

    return {
        "sdist": str(sdist_path),
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "file_count": len(rel_paths),
        "included_assets": len(expected_includes),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sdist",
        nargs="?",
        default=None,
        help="Path to sdist .tar.gz (default: unique dist/*.tar.gz)",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    if args.sdist:
        sdist_path = Path(args.sdist)
    else:
        candidates = sorted((root / "dist").glob("*.tar.gz"))
        if len(candidates) != 1:
            raise SystemExit(f"Expected exactly 1 dist/*.tar.gz, found {len(candidates)}: {candidates}")
        sdist_path = candidates[0]
    summary = verify_sdist(sdist_path, root / "pyproject.toml")
    print(f"sdist OK: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
