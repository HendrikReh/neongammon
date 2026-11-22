# NeonGammon Documentation

Reference guides, technical specifications, training runbooks, and benchmark archives for NeonGammon. For installation, Quick Start, and license details, see the repository [README.md](../README.md).

## Usage & API

| Document | Description |
|---|---|
| [engine.md](engine.md) | Engine behavior by game situation, Stage 11 (24-NN) routing, PureTD hybrid, difficulty presets, move filters, and rollout presets |
| [simulation.md](simulation.md) | Seeded `simulate_game()` and `simulate_match()`, `GameConfig`, `PureTDPolicy`, and `DifficultyPolicy` |
| [api.md](api.md) | Full Python and C++ API reference (`BgBotAnalyzer`, `BatchModels`, 2-ply details, `roll_luck`, `matchinfo`, GNUbg path, rollout snippets, glossary) |
| [bearoff.md](bearoff.md) | One-sided bearoff database format (`data/bearoff_1sided.db`), combinatorial indexing, `BearoffStrategy`, and EPC |

## Algorithms & Architecture

| Document | Description |
|---|---|
| [multi-ply.md](multi-ply.md) | Technical specification for N-ply cubeful/cubeless search, Janowski interpolation, money contact cube life, match play & MET, caching, and 2-ply detail extraction |
| [rollout.md](rollout.md) | Technical specification for full and truncated rollouts (`1T`/`2T`/`3T`), stratified dice, variance reduction, move caches, and `RolloutConfig` / `TrialEvalConfig` |

## Training & Benchmarking

| Document | Description |
|---|---|
| [training.md](training.md) | Input encodings, TD(0) + Apple MLX GPU supervised learning (`neongammon.sl_train`), Modal cloud rollouts, pair/backgame recipes, and data formats |
| [benchmarking.md](benchmarking.md) | Benchmark scripts, 1-in-20 Benchmark PR rollout sample, XG referee runs, and `bench_3t*.py` regression baselines |
| [xg-comparison.md](xg-comparison.md) | Rollout PR, disputed positions, backgames, Paskogammon, and tournament match studies against eXtreme Gammon + `.xg` Batch Analyze & `xg_h2h.py` workflows |
| [model-benchmarks.md](model-benchmarks.md) | Historical benchmark tables across model stages and five-point difficulty profiles |
| [strength-references.md](strength-references.md) | Primary sources on playing strength and evaluation methodology: PureTD preprint, GNUbg/XG specs & usage rights, Modal rollouts, and rollout policy dependence |

## History, Development & Archives

| Document | Description |
|---|---|
| [model-history.md](model-history.md) | Model evolution (Stages 5–11m), cube-life and distillation experiments, Apple Silicon / Accelerate speedups, and resolved investigations |
| [vibing.md](vibing.md) | Notes on AI-assisted development workflow in this codebase |
| [performance-handoff.md](performance-handoff.md) | Archived performance review handoff 1 (PRs #27–#36, completed) |
| [performance-handoff-2.md](performance-handoff-2.md) | Archived performance review handoff 2 (PRs #41–#45, completed) |

## Repository Root Docs

- [README.md](../README.md) — Overview, Quick Start, Apple Silicon build instructions, and MPL-2.0 license summary
- [CHANGELOG.md](../CHANGELOG.md) — Release history
- [CLAUDE.md](../CLAUDE.md) (mirrored by [AGENTS.md](../AGENTS.md)) — Coding agent rules, build/test commands, invariants, and gotchas
