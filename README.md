<img src="docs/assets/neondojo-logo.svg" alt="NeonDojo logo: a torii gate whose pillars are backgammon points, in front of a checker as the rising sun" width="128" height="128">

# NeonGammon

[![Release](https://img.shields.io/github/v/release/HendrikReh/neongammon)](https://github.com/HendrikReh/neongammon/releases)
[![Build wheels](https://github.com/HendrikReh/neongammon/actions/workflows/wheels.yml/badge.svg)](https://github.com/HendrikReh/neongammon/actions/workflows/wheels.yml)
[![NeonGammon manual](https://img.shields.io/badge/NeonGammon%20manual-0969DA)](https://hendrikreh.github.io/neongammon/)
[![NeonDojo manual](https://img.shields.io/badge/NeonDojo%20manual-0969DA)](https://hendrikreh.github.io/neondojo/)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon%20%28arm64%29-000000?logo=apple&logoColor=white)](#installation--building-on-apple-silicon)
[![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.14-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License: MPL-2.0](https://img.shields.io/github/license/HendrikReh/neongammon)](LICENSE)

NeonGammon is a neural-network backgammon engine for Apple Silicon Macs, forked from [Open Sage](https://github.com/markbgsage/bgsage). Its Python interface exposes a C++17 core for comparing checker plays, evaluating cube decisions, measuring roll luck, and simulating games and matches with configurable policies and calibrated opponent presets.

You can use NeonGammon directly from Python or through [NeonDojo](https://github.com/HendrikReh/neondojo), its companion study app. In NeonDojo, you can play against NeonGammon, review its analysis of your imported matches, and practice the positions where you made mistakes.

> **Apple Silicon fork.** NeonGammon ([HendrikReh/neongammon](https://github.com/HendrikReh/neongammon), formerly `HendrikReh/bgsage`) targets **macOS on Apple Silicon (arm64) only**. Batched neural-network evaluation runs through Apple's Accelerate framework on the CPU's matrix unit (SME on M4, AMX on M1–M3), and supervised training runs on the Apple GPU via [MLX](https://github.com/ml-explore/mlx). Upstream's Windows, Linux, and NVIDIA/CUDA code has been removed; the sole non-macOS target is the Linux x86-64 image used by `scripts/modal_rollout.py` for cloud rollouts.

## Quick Start

```python
from neongammon import BgBotAnalyzer, STARTING_BOARD, simulate_game, Difficulty, DifficultyPolicy

# Analyze checker play and cube decisions (defaults to Stage 11, 24-NN production model)
analyzer = BgBotAnalyzer(eval_level="2ply")
play = analyzer.checker_play(STARTING_BOARD, 3, 1)
for m in play.moves[:3]:
    print(f"{m.equity:+.3f}  win={m.probs.win:.1%}  diff={m.equity_diff:+.4f}")

cube = analyzer.cube_action(STARTING_BOARD, cube_value=1, cube_owner="centered")
print(cube.optimal_action, f"ND={cube.equity_nd:+.3f} DT={cube.equity_dt:+.3f}")

# Simulate a complete seeded game (or play against a calibrated difficulty preset)
game = simulate_game(seed=1)
preset_game = simulate_game(players=(DifficultyPolicy(Difficulty.INTERMEDIATE), analyzer), seed=1)
```

## What It Contains

- **24-NN production evaluator (`stage11`):** 17 game-plan pair networks (with Stage 5's 200h pure-race network) plus 7 specialized backgame, containment, snake, and massive-backgame networks, combined with an exact one-sided bearoff database (`data/bearoff_1sided.db`). See [docs/engine.md](docs/engine.md).
- **Multi-ply search & rollouts:** Parallelized CPU N-ply cubeful search ([docs/multi-ply.md](docs/multi-ply.md)) and full/truncated Monte Carlo rollouts with stratified dice and variance reduction ([docs/rollout.md](docs/rollout.md)).
- **Game & match simulation:** Seeded `simulate_game()` and `simulate_match()` with full turn/dice/cube recording, optional PureTD hybrid money play (`PureTDPolicy`), and calibrated opponent presets (`DifficultyPolicy`). See [docs/simulation.md](docs/simulation.md).
- **Training & benchmarking framework:** CPU TD(0) self-play, Apple GPU supervised learning via MLX (`neongammon.sl_train`), Modal cloud rollouts (`scripts/modal_rollout.py`), and benchmark/XG comparison tooling.
- **Test suite:** Automated `pytest` coverage across move generation, encoding golden fixtures, cubeful/match rules, simulation, PureTD, difficulty presets, and MLX training.

## Installation & Building on Apple Silicon

Prebuilt macOS arm64 wheels (Python 3.10–3.14) are published on [GitHub Releases](https://github.com/HendrikReh/neongammon/releases).

To build from source, you need Xcode Command Line Tools (`xcode-select --install`) and Python 3.10+. From the repo root:

```bash
python3 -m venv build/venv
build/venv/bin/pip install numpy pybind11 cmake ninja pytest mlx   # mlx: GPU training only
export PATH="$PWD/build/venv/bin:$PATH"

cmake -S . -B build/cmake_accel -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')"
ninja -C build/cmake_accel bgbot_cpp
rm -f build/bgbot_cpp*.so && cp build/cmake_accel/bgbot_cpp*.so build/

PYTHONPATH="$PWD/build:$PWD/python" python -m pytest -q tests
```

After a C++ change, rerun the `ninja` and `rm`/`cp` lines. **Remove the old module before copying:** on macOS, overwriting an installed `.so` in place leaves a stale cached code signature for that inode, causing subsequent `import bgbot_cpp` calls to die with `SIGKILL` (exit 137, no output).

Accelerate is enabled by default; `-DBGBOT_USE_ACCELERATE=OFF` builds the NEON delta kernels instead. For measured speedups see [docs/model-history.md](docs/model-history.md); for GNUbg CLI configuration (`NEONGAMMON_GNUBG` or `neongammon.settings.set_gnubg_path`), see [docs/api.md](docs/api.md#gnubg-path).

## Evaluation Levels

NeonGammon uses the **XG ply convention** (`1ply` = raw neural-network evaluation = GNUbg `0-ply`):

| `eval_level` | Method | Notes |
|---|---|---|
| `1ply` | Direct NN evaluation | Instant; Janowski cubeful conversion |
| `2ply`, `3ply`, `4ply` | N-ply cubeful search | Lookahead over opponent replies; each ply ~20× slower |
| `truncated1` (1T) | 72-trial truncated rollout | 5 half-moves, 1-ply play, variance reduction |
| `truncated2` (2T) | 360-trial truncated rollout | 7 half-moves, 2-ply first move/cube/truncation |
| `truncated3` (3T) | 360-trial truncated rollout | 7 half-moves, 3-ply early then 2-ply |
| `rollout` | Full rollout (1,296 trials default) | Configurable checker/cube trial strengths & `target_se` |

See [docs/engine.md](docs/engine.md) for filter behavior and default simulation policies.

## Interfaces

Both Python (`neongammon`) and C++ (`bgbot` / `bgbot_cpp`) interfaces are provided — prefer the Python public API ([docs/api.md](docs/api.md)) unless targeting low-level C++ internals:

- **Checker play analytics** (`BgBotAnalyzer.checker_play`, `batch_checker_play`): ranks all legal moves by cubeful or cubeless equity with post-move probabilities.
- **Post-move analytics** (`BgBotAnalyzer.post_move_analytics`, `batch_post_move_evaluate`): cubeful equity, cubeless equity, and probabilities right before the opponent's turn.
- **Cube action analytics** (`BgBotAnalyzer.cube_action`, `cube_decision`, `batch_cube_action`, `batch_evaluate`): No Double / Double-Take / Double-Pass equities, optimal action, and optional 2-ply per-roll details (`incl_2ply_details=True`).
- **Per-roll luck & match info** (`roll_luck`, `luck_from_equities`, `neongammon.matchinfo`): signed dice-fortune equity from cube details, plus MET take points and gammon prices.
- **Bearoff database & EPC** (`BgBotAnalyzer.epc`, `BearoffDB`): exact home-board bearoff probabilities and Effective Pip Count.
- **Game plan classification & board utilities** (`classify_game_plans`, `possible_moves`, `flip_board`, `is_race`, `is_crashed`, `check_game_over`).
- **Simulation & difficulty presets** (`simulate_game`, `simulate_match`, `PureTDPolicy`, `Difficulty`, `DifficultyPolicy`).

## How Does It Compare to XG?

Money games, 5-point matches, backgames, Paskogammon, and real tournament matches were benchmarked against eXtreme Gammon (XG). At matched truncated-rollout strength (`3T` vs XG Roller++), NeonGammon and XG are very close — with NeonGammon slightly ahead in the rollout PR studies and matching XG's tournament match PR ratings to within `+0.002` PR (`r = 0.98`).

Full results and replication instructions — including running XG under CrossOver on macOS with Batch Analyze and `scripts/xg_h2h.py` — are in [docs/xg-comparison.md](docs/xg-comparison.md) and [docs/benchmarking.md](docs/benchmarking.md).

## Documentation

The guides below are also published as a searchable site at **[hendrikreh.github.io/neongammon](https://hendrikreh.github.io/neongammon/)**, rebuilt from `docs/` on every change to `main`.

| Document | Contents |
|---|---|
| [docs/engine.md](docs/engine.md) | Engine behavior by game situation, Stage 11 routing, PureTD hybrid, difficulty presets, filters, rollouts |
| [docs/simulation.md](docs/simulation.md) | `simulate_game` / `simulate_match`, `GameConfig`, `PureTDPolicy`, `DifficultyPolicy` |
| [docs/api.md](docs/api.md) | Full Python & C++ API reference, batch APIs, 2-ply detail schema, luck, matchinfo, GNUbg path, glossary |
| [docs/bearoff.md](docs/bearoff.md) | One-sided bearoff database format, indexing, API, and EPC |
| [docs/multi-ply.md](docs/multi-ply.md) | Technical specification of N-ply cubeful/cubeless search, Janowski, money cube life, match play, and caching |
| [docs/rollout.md](docs/rollout.md) | Technical specification of full/truncated rollouts, stratified dice, variance reduction, and configuration |
| [docs/training.md](docs/training.md) | Input encodings, TD+SL pipeline, Apple MLX GPU trainer, Modal cloud rollouts, and data formats |
| [docs/benchmarking.md](docs/benchmarking.md) | Benchmark scripts, 1-in-20 Benchmark PR rollout sample, XG referee, and regression baselines |
| [docs/xg-comparison.md](docs/xg-comparison.md) | Head-to-head and Batch Analyze PR studies against eXtreme Gammon + `.xg` workflows |
| [docs/model-benchmarks.md](docs/model-benchmarks.md) | Historical benchmark tables across model stages and five-point difficulty profiles |
| [docs/strength-references.md](docs/strength-references.md) | Primary sources on playing strength, PureTD preprint, GNUbg/XG specs & usage rights, and rollout policy dependence |
| [docs/model-history.md](docs/model-history.md) | Model evolution (Stages 5–11m), tuning experiments, Apple Silicon speedups, and resolved investigations |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [docs/vibing.md](docs/vibing.md) | Notes on AI-assisted development workflow in this codebase |

## License

NeonGammon is released under the **Mozilla Public License 2.0** — see [LICENSE](LICENSE).
Files that came from upstream keep their original copyright notices (Mark Higgins).

Copyright © 2026 Hendrik Reh for his original contributions to NeonGammon, a fork of Open Sage.

The MPL-2.0 license covers everything in this repository, including the **trained model weights** in
`models/` and the bearoff database in `data/`. Those aren't source code, so to be
unambiguous: they are licensed on exactly the same terms as the code, and you are free to
use them in a commercial product.

In plain English:

* **You can use NeonGammon in a commercial, closed-source product.** Import it, link it,
  bundle the weights, ship it. Your own code stays yours, under whatever license you choose.
* **If you modify a file that came from NeonGammon and distribute it, publish that file's
  source** under the MPL, so the improvement comes back to everyone. Files *you* write remain
  entirely yours, even when they sit right alongside ours.
* **Private use carries no obligations at all.** Experiment, retrain, tinker — if you aren't
  distributing it, nothing is asked of you.

The MPL is file-level copyleft, so the obligation is narrow and predictable: it attaches only
to the files you changed, and never to the code you build around them. If you do improve the
engine or train better weights, a pull request is very welcome.
