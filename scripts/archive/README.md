# Archived Scripts

Historical recipes, one-off benchmarks, variant experiments, and manual
diagnostic scripts moved out of the top-level `scripts/` directory.

- **Top-level (`scripts/archive/*.py`)**: Stage 4–10 training and benchmark
  recipes (with compatibility symlinks in `scripts/`).
- **`pasko/`**: Legacy Paskogammon variant benchmark, data-generation, and XG
  export scripts. (Shared `.xg` export helpers live in `scripts/xg_export_common.py`.)
- **`diagnostics/`**: Manual ad-hoc verification and diagnostic scripts
  (formerly `scripts/test_*.py`, VR bias probes, race-gap probes, and
  `rank_distill.py`). Automated tests live under `tests/` and `cpp/tests/`.
- **`stage11/`**: One-off Stage 11 position-harvesting, data-segregation, and
  specialist training scripts.
- **`historical/`**: Retired stubs (`play_sage_vs_xg.py`,
  `run_sage_vs_xg_games.py`), legacy AWS ECS rollout worker, and superseded
  one-off micro-benchmarks.

For current training and benchmarking workflows, see
[docs/training.md](../../docs/training.md) and
[docs/benchmarking.md](../../docs/benchmarking.md).
