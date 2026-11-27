# Multi-ply benchmark corpus

The two input sets and the rollout reference files were copied byte for byte from Mark Higgins's Open Sage benchmark at [`82e6397`](https://github.com/markbgsage/bgsage/tree/82e63972379b77ab52ebafc716620e813ff7baa1/data/multi_ply_benchmark).

| Inputs | Public API | Cases |
| --- | --- | ---: |
| [Checker positions](checker_play.json) | `checker_play` | 100 |
| [Cube positions](cube_action.json) | `cube_action` | 100 |
| [Cube positions](cube_action.json), flipped to the previous mover's perspective | `post_move_analytics` | 100 |

Each input set has 50 money and 50 five-point match positions, with ten per game plan in each. The files preserve dice, cube owner and value, Crawford status, away scores, and the applicable money rules. The post-move cases carry their own board, cube owner and away scores.

[The reference manifest](rollout_positions/manifest.json) pins the inputs, source archives, engine and rollout policy. [The reference outputs](rollout_positions/results.jsonl) include 37 checker decisions recovered from the original money and match archives and 263 public calls made with the frozen Open Sage Stage 11 engine. These public calls use 1,296 stratified games per rollout with variance reduction. The checker results contain 1,709 legal endpoints, of which 378 reached rollout depth. The [coverage records](rollout_positions/position_coverage.jsonl), [checker cubeful standard errors](rollout_positions/checker_cubeful_se.jsonl) and [supplemental archival cube rows](rollout_positions/archived_cube_support.jsonl) retain the available provenance and uncertainty.

## Run the benchmark

Build the NeonGammon Release extension with Accelerate as described in the [repository README](../../README.md#installation--building-on-apple-silicon), then run from the repository root with the same Python environment.

```sh
VECLIB_MAXIMUM_THREADS=1 PYTHONPATH="$PWD/build/cmake_accel:$PWD/python" \
  python scripts/benchmark_multi_ply.py --model stage11 --threads 4 \
  --output logs/multi-ply-m4-pro
```

The runner verifies the fixture hashes and input contexts before timing. It runs 1–4 ply and the 1T, 2T and 3T presets by default. Use `--levels 2ply 3ply` for a shorter run. Select a new output directory for each run.

Each level uses a fresh analyzer and one pass through each API in fixture order. Wall times include all work inside the public call. Construction, per-position seed resets, serialization and disk writes are outside the timer. The truncated presets use the same seeded dice schedule across levels, derived from the source position key. The analyzer keeps its ordinary caches for the duration of each level.

The summary reports latency and mean absolute error against the frozen rollout values. Checker comparisons pair by resulting board and use the 378 endpoints labeled `Rollout` in the reference. The summary records the prediction depths for those endpoints and how often the top checker pick has a rollout value. Cube and post-move errors compare each API to its own reference output. The measurements describe this fixed corpus and rollout policy. The compressed results retain every returned field and exact per-call timing for future comparisons.

## Apple Silicon baseline

See the [M4 Pro baseline](baselines/m4-pro-20261127/README.md) for timings, rollout comparisons and the complete saved results.
