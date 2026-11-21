# NeonGammon (`neongammon`)

Neural-network backgammon engine with a C++17 core (`cpp/`, bound via pybind11 as `bgbot_cpp`) and a Python package (`python/neongammon/`). Licensed under MPL-2.0.

## Platform Scope & Repo Boundaries

- **Apple Silicon macOS (arm64) only** (fork `HendrikReh/neongammon` of `markbgsage/bgsage`; dev machine M4 Pro, 10P+4E cores, 64 GB). Optimize freely with NEON, Accelerate (SME/AMX), Metal, or MLX; portable fallbacks and Windows/Linux/x86/CUDA code or docs are out of scope.
- **Single x86-64 exception (Modal):** `scripts/modal_rollout.py` builds the engine on Linux x86-64 cloud containers, so keep the `"Modal only"` AVX2 kernels in `cpp/src/neural_net.cpp` and the x86-64 branch in `cpp/CMakeLists.txt` compiling.
- **Upstream merges:** resolve conflicts in removed Windows/Linux/CUDA code by keeping this fork's side. Re-measure any upstream timings on Apple Silicon before citing them.
- **Standalone repo boundary:** keep all edits, commits, and runtime paths inside this repo (note: older checkouts may live in a folder named `bgsage/`). Put Python modules in `python/neongammon/`, scripts in `scripts/`, C++ in `cpp/`, compiled `bgbot_cpp*.so` in `build/`, and default outputs in `logs/`. Scripts resolve `_PROJECT_ROOT = _SCRIPT_DIR.parent` (fix legacy `.parent.parent` host-project paths when touching a script).
- **Git worktrees:** when working in a worktree, use relative paths or the worktree path for every read, edit, build, and command so files and `build/bgbot_cpp*.so` stay on the worktree branch rather than the main checkout. Commit and push from the worktree; merge to `main` via PR or local merge.

## Build, Test & Packaging

### Build the C++ Extension

Create a local venv in `build/venv` (gitignored) with `numpy pybind11 cmake ninja pytest mlx`, then build from the repo root:

```bash
export PATH="$PWD/build/venv/bin:$PATH"
cmake -S . -B build/cmake_accel -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')"
ninja -C build/cmake_accel bgbot_cpp
rm -f build/bgbot_cpp*.so && cp build/cmake_accel/bgbot_cpp*.so build/
```

- **Always `rm` before `cp`:** overwriting an existing `.so` in place leaves macOS with a stale cached code signature for that inode, causing every later `import bgbot_cpp` to die with silent `SIGKILL` (exit 137).
- `build/cmake_accel` (Accelerate on, default) is the canonical build dir. Keep `build/cmake_portable` (`-DBGBOT_USE_ACCELERATE=OFF`, NEON delta kernels) and `build/cmake_base` for A/B checks only.
- Use `cmake -S cpp` only when building standalone C++ benchmark/profiler binaries.

### Run Tests

```bash
PYTHONPATH="$PWD/build:$PWD/python" python -m pytest -q tests
```

Key regression tests to run when touching sensitive areas:
- `tests/test_encoding_golden.py` — pins 244-feature encoder outputs on 1,000 positions (must stay bit-identical when editing `cpp/src/encoding.cpp`).
- `tests/test_sl_train.py` — holds the MLX GPU trainer to a float64 NumPy reference (tolerance `2e-5`).
- `tests/test_batch_checker_play_parity.py` & `tests/test_2ply_details.py` — parity between `BgBotAnalyzer`, batch checker play, and 2-ply details.
- `tests/test_contact_cube_life.py` — money contact cube-life overrides.
- `tests/test_simulation.py`, `tests/test_puretd.py`, `tests/test_difficulty.py` — simulation, PureTD hybrid, and opponent presets.

### Promoting a New Production Model

1. Register the model in `MODELS` and set `PRODUCTION_MODEL` in [python/neongammon/weights.py](python/neongammon/weights.py) (currently `"stage11"`, a 24-NN backgame-aware phased pair strategy).
2. Add any new required weight files to the `sdist.include` **allowlist** in [pyproject.toml](pyproject.toml) (the sdist excludes `models/` and `data/` by default; omitting a weight file breaks wheel/sdist builds).
3. Record results in [MODEL_BENCHMARKS.md](MODEL_BENCHMARKS.md) and architecture/training details in [docs/model-history.md](docs/model-history.md).

## Core Domain Invariants

- **Use the public entry points:** use `BgBotAnalyzer` (and `neongammon.batch` / `BatchModels`) for analysis and `simulate_game()` / `simulate_match()` for play. Call raw `bgbot_cpp` bindings only when targeting a specific low-level model type or parameter not exposed publicly. See [python/neongammon/__init__.py](python/neongammon/__init__.py) for exported symbols.
- **Board representation (`std::array<int, 26>`):** always from the **player on roll's** perspective. Points `1..24` (`>0` player, `<0` opponent); index `0` = opponent bar (`>=0`), index `25` = player bar (`>=0`). `flip()` reverses, negates, and swaps bars after every move.
- **NN output semantics & tempo:** networks output `[P(win), P(gw), P(bw), P(gl), P(bl)]` for the player who **just moved** (post-move, before opponent rolls). Equity = `2*P(win) - 1 + P(gw) - P(gl) + P(bw) - P(bl)`. To get pre-roll probabilities: `flip(board)` → evaluate → `invert_probs()`. `evaluate(board) != invert(evaluate(flip(board)))` because of tempo.
- **Ply convention (XG numbering everywhere):** `1-ply` = raw NN evaluation (= GNUbg `0-ply`), `2-ply` = one opponent reply (= GNUbg `1-ply` = PureTD `checker_plies=1`), etc.
- **Experiment guardrails:** put each experiment in its own directory, compare against current `PRODUCTION_MODEL` (`stage11`), and get explicit approval before modifying `cpp/` evaluation code during an experiment. Note that XG 4-ply saturates as a referee at 3-ply; evaluating 3-ply+ gains requires a rollout-grade reference.

## Technical Gotchas

- **Shared thread-local cache:** `MultiPlyStrategy::get_cache()` uses a `thread_local static PosCache` shared across all instances — call `clear_cache()` between strategy comparisons.
- **C++ stream formatting:** `std::fixed` is sticky; reset with `std::defaultfloat` when default formatting is needed.
- **Extended encoder init:** call `init_escape_tables()` once before using the 244-input contact encoder.
- **Full rollouts & `ultra_late_threshold`:** the default `ultra_late_threshold=2` drops checker/cube eval to 1-ply at move 2+ (and thins VR to even moves). Pass `ultra_late_threshold=9999` for accurate full rollouts (`truncation_depth=0`).
- **Apple Silicon numeric behavior:** keep `-ffast-math` off on Apple Silicon. Accelerate's `forward_group` (`cblas_sgemm`) uses exact Cephes sigmoid rather than the NEON LUT (up to `~0.003` equity shift vs `-DBGBOT_USE_ACCELERATE=OFF`).
- **Training prerequisites:** TD self-play pre-training is required before SL (SL from random init finds poor minima). Train contact NNs on **all** contact+crashed data weighted by `--gameplan-weight`, never narrow game-plan subsets.
- **Python vs C++ rule defaults:** Python public API defaults `jacoby=True, beaver=True` for money play (auto-disabled in match play); C++ bindings default both to `false`.

## Documentation Index (Read Before Touching a Subsystem)

| When working on… | Read |
|---|---|
| Engine behavior, Stage 11 (24-NN) routing, PureTD hybrid, `DifficultyPolicy`, filters & rollout presets | [ENGINE.md](ENGINE.md) |
| Game/match simulation (`simulate_game`, `simulate_match`, `GameConfig`, `PureTDPolicy`, `DifficultyPolicy`) | [SIMULATION.md](SIMULATION.md) |
| Public Python/C++ APIs (`BgBotAnalyzer`, `BatchModels`, 2-ply details, `roll_luck`, `matchinfo`, GNUbg path, rollout Python snippets, glossary) | [docs/api.md](docs/api.md) |
| Multi-ply search, cubeful tree (`cube_eval.cpp`), Janowski, money contact cube-life override, match play & MET, 2-ply detail extraction | [MULTI-PLY.md](MULTI-PLY.md) |
| Rollouts, truncated rollouts (`1T`/`2T`/`3T`), variance reduction, `Move0Cache`/`Move1Cache`/`SharedPosCache`, `TrialEvalConfig` | [ROLLOUT.md](ROLLOUT.md) |
| One-sided bearoff DB (`data/bearoff_1sided.db`), combinatorial indexing, `BearoffStrategy`, EPC | [docs/bearoff.md](docs/bearoff.md) |
| Training pipeline (TD, MLX GPU SL `neongammon.sl_train`, Modal cloud rollouts, 5-NN/17-NN/backgame recipes, `.bm`/train data formats) | [docs/training.md](docs/training.md) |
| Benchmark scripts, 1-in-20 Benchmark PR rollout sample, XG referee, `bench_3t*.py` regression baselines | [docs/benchmarking.md](docs/benchmarking.md) |
| XG comparison studies, `.xg` Batch Analyze workflow, `scripts/xg_h2h.py`, `neongammon.xg_compare` | [XG_COMPARISON.md](XG_COMPARISON.md) |
| Historical benchmark tables (Stages 3–11, five-point difficulty profiles) | [MODEL_BENCHMARKS.md](MODEL_BENCHMARKS.md) |
| Historical stage design & investigations (Stages 5–11m, cube-life tuning, `rank_distill`, Accelerate/PR #3 timings, 2026-09-06 cube postmortem) | [docs/model-history.md](docs/model-history.md) |
