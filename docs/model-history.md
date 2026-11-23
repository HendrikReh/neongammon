# Model Evolution, Experiments, and Performance Measurements

Historical model stages, tuning experiments, Apple Silicon performance measurements, and resolved investigations. For current engine behavior see [docs/engine.md](engine.md); for summary benchmark tables see [docs/model-benchmarks.md](model-benchmarks.md).

## Contents

- [Apple Silicon & Analyzer Performance Measurements](#apple-silicon--analyzer-performance-measurements)
- [Money Contact Cube Life Tuning](#money-contact-cube-life-tuning)
- [Interior Move-Pick Validation: Cubeful vs Cubeless (2026-11-19)](#interior-move-pick-validation-cubeful-vs-cubeless-2026-11-19)
- [Search Distillation Experiment](#search-distillation-experiment)
- [Stage 5 Scores](#stage-5-scores-historical-baseline--not-the-production-model)
- [Stage 6 (S6)](#stage-6-s6--mid-size-model)
- [Stage 7 (S7)](#stage-7-s7--17-nn-pair-strategy-model)
- [Stage 8 (S8)](#stage-8-s8--17-nn-pair-strategy-400h-contact-s5-fallback)
- [Stage 5 Small (S5S)](#stage-5-small-s5s--fast-filter-model)
- [Backgame Stages (S10, S11, S11p, S11s, S11m) and Cube Scoring Investigation](#backgame-stages-s10-s11-s11p-s11s-s11m-and-cube-scoring-investigation)

## Apple Silicon & Analyzer Performance Measurements

**Accelerate (Apple Silicon only, on by default).** The batched candidate
kernels — every `batch_evaluate_candidates_*` and the cubeful engine's
`eval_groups_pair` — go through `NeuralNetwork::forward_group`, which on
Apple arm64 runs each layer as one dense `cblas_sgemm` through Accelerate.
Accelerate executes these on the CPU's matrix unit (SME on M4, AMX on M1-M3),
which beats the sparse delta kernels even though ~77% of the inputs are zero.
The hidden-layer sigmoid on this path is the exact Cephes one, not the lookup
table the other NEON kernels originally used: the table clamped to exactly 0/1
beyond ±8 (up to 3e-4 error per hidden unit), which moved equities by up to
~0.003. Since v3.1.1 (PR #58), all arm64 forward paths share the vectorized
polynomial `sigmoid_inplace_neon` (matching x86-64 AVX2 `sigmoid256_ps`),
removing the single-board vs batched LUT discrepancy; `-DBGBOT_USE_ACCELERATE=OFF`
still selects the portable NEON delta kernels instead of `cblas_sgemm`.

Measured 2026-11-17 on an M4 Pro (10P+4E cores), together with the encoder
optimisation below (which is bit-identical on every platform). Wall time of
`checker_play` on a fixed money-benchmark sample, against the engine before
both changes:

| Level, threads | Encoder only (`BGBOT_USE_ACCELERATE=OFF`) | Encoder + Accelerate |
|---|---|---|
| 2-ply, 1 thread  | 1.17x | 1.38x |
| 3-ply, 1 thread  | 1.17x | 1.37x |
| 3-ply, 10 threads | 1.13x | 1.37-1.41x |

Money benchmark (17,535 decisions) with Accelerate: 2-ply PR 1.67 -> 1.68
(cube 2.90 -> 2.92), 3-ply PR 0.60 -> 0.60, blunder counts unchanged. Max
equity shift on the sample 0.004; 2 of 264 2-ply best moves changed (near-ties).

**Encoder hot spots.** `hitting_shots`, `double_hitting_shots`,
`compute_piploss`, `moment2` and `backbone` in `encoding.cpp` use bitmask
formulations (distance masks per blot, a mirrored 8x8 roll matrix for the
double-shot count, per-combination block masks for piploss). On 655k
money-benchmark boards and candidates, `compute_extended_contact_inputs` went
1.80 -> 1.17 us per board with every output bit-identical.
`tests/test_encoding_golden.py` pins those features on 1,000 positions computed
by the original implementations; keep it passing when touching them.

**Analyzer call overhead (PR #3).** Two Python-side costs used to dominate the
cheap levels. `cube_action` built a fresh base strategy on every call,
reloading every weight file (24 for `stage11`, ~20 ms); `_CubelessBase._cube_strategy`
now loads it once per analyzer and passes it to `cube_decision_*_unified` via
`strategy=`. And 1-ply cubeful `checker_play` ran its conversion through a
`ThreadPoolExecutor` (~0.9 ms of pool start-up for microseconds of work); it now
runs serially (`SERIAL_CUBEFUL_1PLY`), N-ply keeps the pool. The shared cube
strategy is safe to call from several threads: networks are immutable after
construction, and the per-thread cubeful cache key covers every `CubeInfo`
field, ply, deep prefilter and root pin, so cache hits across calls return
fresh-evaluation values (8 Python threads x 1,440 calls at 1-3 ply,
bit-identical to a fresh analyzer per call).

Measured 2026-11-18 on the M4 Pro, `stage11`, one persistent analyzer, median
ms per call over fixed random positions, `main` before (`c8ff5a6`) vs after
(`623c458`), three interleaved rounds. Another job was loading the CPU (load
average ~45), so absolute times run high; the ratios hold.

| Call | before | after | speed-up |
|---|---|---|---|
| `cube_action` 1-ply | 19-25 | 0.015 | ~1,500x |
| `cube_action` 2-ply | 22-29 | 1.9 | ~13x |
| `cube_action` 3-ply | 36-40 | 16 | ~2.3x |
| `checker_play` 1-ply | 1.07-1.15 | 0.13-0.14 | ~8x |
| `checker_play` 1-ply, 1 thread | 0.96-1.06 | 0.13-0.14 | ~7.5x |
| `checker_play` 2-ply | 11.0-14.3 | 13.9-15.3 | unchanged (noise) |

The weight-load saving needs a long-lived analyzer: code that builds a new
`BgBotAnalyzer` per call still pays it (the cube strategy loads lazily on the
analyzer's first `cube_action`).

Since 2026-11-19 an N-ply analyzer's checker side loads the model once: the
N-ply strategy is built around the 1-ply strategy's base
(`create_multipy_from_base`, also for the rollout analyzer's 2-ply prefilter
strategy), so a 3-ply analyzer loads Stage 11 twice (checker side, then the
cube strategy on first use) instead of three times. The cube strategy keeps
its own base, so its evaluator configuration (root routing) stays separate.
The bearoff database is loaded once per process (`analyzer.load_bearoff_db`)
and shared by analyzers and `BatchModels`, which loads it only when a batch
has positions. Constructing a 3-ply analyzer takes 17-20 ms instead of
35-41 ms; analysis times are unchanged (`tests/test_analyzer_model_reuse.py`).

**Engine & Rollout Optimizations (v3.1.1, PR #58, 2026-11-23).** Ported and
adapted 27 engine optimization commits for Apple Silicon (`BGBOT_USE_ACCELERATE=ON`),
spanning:

- **Cross-roll VR batching** ([`rollout_vr.h`](../cpp/include/bgbot/rollout_vr.h)):
  evaluates all 21 rolls' candidates in one grouped `cblas_sgemm` pass on Apple
  Accelerate with `VrEndpointDedup` endpoint deduplication (and a shared
  pre-roll hidden base on the portable path), eliminating redundant `Move1Cache`
  passes and adding a 2,048-slot lock-free `VarianceReductionCache` for moves 2+.
- **Multi-ply batch & cache reuse**: `CandidateBatchCache` (call-local 1-ply
  candidate batch reuse across the live and cubeless passes of
  `cube_decision_with_cubeless_probs`), `DiceNode` per-depth pre-roll base
  reuse, `CheckerSubtrees` (64 mutex-sharded shards × ≤128 entries call-local
  4-ply transposition table), generation-stamped O(1) `CubefulEvalCache` clear,
  and lazy `PosCache` / `SharedPosCache` with `STATE_CLAIMED` spin and
  cache-line padding.
- **Work-stealing & cooperative scheduling**: `allow_parallel` enabled at
  `n_plies >= 2` (with dynamic `next_roll` work-stealing for `4 <= n_threads <= 21`),
  two-level `multipy_cooperative_for` for 4-ply checker play, and adaptive
  `trial_chunk_size` / `trial_report_interval` for rollouts.
- **Hot-path kernels**: `possible_boards_unsorted`, faster contact encoding,
  `TrialMoveScratch` buffer reuse, batched `BackgameAwarePairStrategy` cubeful
  multi-evaluation, and unified arm64 `sigmoid_inplace_neon` across single-board
  and grouped forward passes.

Measured 2026-11-23 on an Apple Silicon M4 Pro (10 threads, `stage11`,
`BGBOT_USE_ACCELERATE=ON`), comparing `v3.1.0` (`cd9c004`) vs `v3.1.1`
(`abaedf9`):

| Workload | v3.1.0 | v3.1.1 | Speed-up |
|---|---:|---:|---:|
| `move_gen_unsorted` (3,600 rolls) | 0.0186 s | 0.0094 s | 1.98x |
| `move_gen_sorted` (3,600 rolls) | 0.0424 s | 0.0125 s | 3.39x |
| `encode_contact` (20,000 boards) | 0.0538 s | 0.0368 s | 1.46x |
| `2ply_cube` (10 positions) | 0.0194 s | 0.0035 s | 5.54x |
| `2ply_checker` (6 positions) | 0.0277 s | 0.0120 s | 2.31x |
| `3ply_cube` (10 positions) | 0.0473 s | 0.0249 s | 1.90x |
| `3ply_checker` (6 positions) | 0.6126 s | 0.3171 s | 1.93x |
| `4ply_checker` (2 positions) | 1.6733 s | 0.8336 s | 2.01x |
| `truncated1` cube rollout (1 position) | 0.0449 s | 0.0148 s | 3.03x |
| `truncated2` cube rollout (1 position) | 0.6784 s | 0.1761 s | 3.85x |
| `truncated3` cube rollout (1 position) | 1.6851 s | 0.8484 s | 1.99x |
| `full_rollout` cube (144 trials, 1-ply) | 1.0424 s | 0.2190 s | 4.76x |
| `truncated2` checker rollout (1 position) | 0.9134 s | 0.2044 s | 4.47x |

**2-ply cube thread scaling on M4 Pro** (100 calls, `v3.1.1`, Accelerate ON):
because macOS passes `threads` through at 2-ply (unlike Linux, where
`root_roll_threads` forces 1 thread on 2-ply races), both contact and race
positions scale across threads:

| Position class (100 calls) | 1 thread | 4 threads | 10 threads | 1T → 10T |
|---|---:|---:|---:|---:|
| Contact (2-ply cube) | 0.0615 s | 0.0245 s | 0.0214 s | 2.87x |
| Race (2-ply cube) | 0.0277 s | 0.0125 s | 0.0126 s | 2.20x |

**Per-call analyzer latency on an idle M4 Pro** (`scripts/bench_analyzer_overhead.py`,
`v3.1.1`, `stage11`, default threads, 3 rounds, median ms per call):

| Call | Median ms/call (3 rounds) |
|---|---:|
| `cube_action` 1-ply | 0.0107–0.0110 |
| `cube_action` 2-ply | 0.206–0.218 |
| `cube_action` 3-ply | 2.358–2.376 |
| `checker_play` 1-ply | 0.081–0.084 |
| `checker_play` 1-ply, 1 thread | 0.082–0.085 |
| `checker_play` 2-ply | 0.794–0.842 |


## Money Contact Cube Life Tuning

Operational specification lives in [docs/multi-ply.md](multi-ply.md#money-contact-cube-life-by-depth).

**Tuning.** Tuned 2026-11-18 on the odd seeds of the money benchmark's 2,842
cube decisions and checked on the even seeds. Cube PR, old -> new:

| Level | odd seeds (tuning) | even seeds (held out) |
|---|---|---|
| 1-ply | 3.37 -> 2.19 | 3.06 -> 2.52 |
| 2-ply | 3.06 -> 1.35 | 2.78 -> 1.26 |

On the tuning half the optimum is flat: 0.74-0.76 at 1-ply and 0.79-0.81 at
2-ply. At 3-ply, 0.68 is already best (0.52; 0.57 at 0.72).

**Full money benchmark** (17,535 decisions), PR old -> new:

| Level | total | checker | cube | blunders |
|---|---|---|---|---|
| 1-ply | 2.55 -> 2.43 | 2.42 -> 2.44 | 3.21 -> 2.36 | 153 -> 166 |
| 2-ply | 1.65 -> 1.38 | 1.41 -> 1.40 | 2.92 -> 1.31 | 55 -> 58 |

Checker play is unaffected within noise (paired: 1-ply +0.017 +- 0.011, 2-ply
-0.010 +- 0.013).

**Backgame folders**, cube PR at 1P / 2P, old -> new:

| Folder | S9-play reference | S11-play twin |
|---|---|---|
| containment | 6.32 / 6.80 -> 4.09 / 2.74 | 9.03 / 11.10 -> 7.56 / 6.18 |
| massive | 6.97 / 6.46 -> 3.42 / 2.32 | 8.49 / 7.97 -> 4.23 / 3.09 |

Caveats:
- At 1-ply the flat 0.75 cut the cube PR but raised cube blunders (29 -> 39
  on the full set); the pip slope below brings them back down.
- Match play is excluded because the same values do not help on the 5-point
  match benchmark: 1-ply 2.35 -> 2.38, 2-ply 1.58 -> 1.82. 0.75 at 2-ply gave
  1.32 there, but without a held-out check.
- Every reference above is a NeonGammon rollout, which makes its own in-trial
  cube decisions at 3-ply with 0.68. The XG check below is the independent
  one.

**XG as referee** (2026-11-18). `scripts/export_money_cube_xg.py` writes
the money benchmark's cube positions as one-decision `.xg` games:
`generate` writes 11 shards of 200 to `data/money_cube_xg/`. Positions where
the opponent owns the cube are skipped, because the mover has no decision
there. XG Batch-Analyzes them ("World Class": 4-ply, XG Roller+ for close
decisions). `score --xg-dir` reads XG's ND / DT / DP back into
`xg_reference.jsonl`. `sage --xg-dir` scores NeonGammon old and new against both
references, with paired differences.

All 2,196 positions (2,842 decisions) were analysed: 1,503 at 4-ply, 685 by
XG Roller+ and 8 at 2-ply. XG's equities come out normalised (DP = 1.0,
player-owned cubes included). XG's own cube PR against the NeonGammon reference is
0.44, so the two references are close. Cube PR old -> new, blunders in
brackets:

| Level | vs NeonGammon reference | vs XG | paired new - old vs XG |
|---|---|---|---|
| 1-ply | 3.21 (29) -> 2.36 (39) | 2.74 (33) -> 2.00 (42) | -0.73 +- 0.19 |
| 2-ply | 2.92 (18) -> 1.30 (19) | 2.30 (15) -> 1.02 (18) | -1.28 +- 0.15 |
| 3-ply | 0.61 (4), unchanged | 0.39 (3), unchanged | 0 |

On the held-out even seeds against XG (1,107 positions):

| Level | Cube PR, old -> new (blunders) | Paired new - old |
|---|---|---|
| 1-ply | 2.50 (11) -> 2.01 (19) | -0.48 +- 0.30 |
| 2-ply | 2.14 (4) -> 0.99 (8) | -1.15 +- 0.20 |

So the 2-ply gain is confirmed independently. The 1-ply value helps on
average, but it trades many small over-doubles for a few missed clear
doubles: blunders rise under every reference.

**Pip slope** (1-ply 2026-11-18, 2-ply 2026-11-19).
`scripts/fit_contact_cube_life.py --xg-dir data/money_cube_xg` fits the
contact cube life to XG's verdicts on the same 2,196 positions: it plays
every cube decision through the engine at each point of a (value, slope)
grid, so every position the cube tree converts, root and leaves, takes the
cube life of its own pip count. The features were chosen in an earlier
exploratory sweep of a flat cube life per position (0.55 to 1.00). Of the
features tried there, only the
pip count held up on both seed halves at both levels: the best cube life is
lower late in the game (about 0.65 under 100 average pips, 0.80 over 130)
and lower when the doubler trails in the race, but the race standing adds
nothing once the average pip count is in. Gammon rate, plan pair, P(win) and
cube ownership gave no optimum that held across halves.

Fitted on one half of the seeds against XG and scored on the other, both ways
round (cube PR, blunders in brackets):

| Out of fold | vs XG | vs NeonGammon reference |
|---|---|---|
| 1-ply, flat cube life (refitted the same way) | 2.07 (40) | 2.49 (37) |
| 1-ply, pip slope | 1.68 (33) | 1.98 (35) |
| 1-ply, paired, slope - flat | -0.40 +- 0.10 | -0.51 +- 0.11 |
| 2-ply, flat cube life (refitted the same way) | 1.09 (16) | 1.44 (18) |
| 2-ply, pip slope | 0.90 (10) | 1.21 (16) |
| 2-ply, paired, slope - flat | -0.19 +- 0.08 | -0.22 +- 0.09 |

Fits on the even / odd / all seeds: 1-ply (0.7125, 0.00250) / (0.7375,
0.00150) / (0.7375, 0.00150); 2-ply (0.7375, 0.00175) / (0.7625, 0.00150) /
(0.7625, 0.00125). The 2-ply table is the all-seed fit. The 1-ply table
keeps the 0.00175 of the first, grid-snapped fit; the optimum is flat along
the diagonal of the (value, slope) grid at both levels, and 0.00150 vs
0.00175 is within it.

Against the previous flat values (same positions as the fit, so in
sample):
- 1-ply, flat 0.75: XG cube set 2.00 (42) -> 1.62 (33) vs XG and
  2.36 (39) -> 1.89 (34) vs the NeonGammon reference. Full money benchmark: total
  PR 2.43 -> 2.35, cube 2.36 -> 1.89, checker 2.441 -> 2.435 (paired
  -0.006 +- 0.010; 30 of 14,693 picks change).
- 2-ply, flat 0.80: full money benchmark total PR 1.383 -> 1.338, cube
  1.305 -> 1.038 (paired -0.27 +- 0.09; cube blunders 19 -> 12), checker
  1.398 -> 1.396 (paired -0.002 +- 0.008; 57 of 14,693 picks change).

Not re-measured with the slope: the backgame folders in the table above and
the 5-point match benchmark (the slope is money only, like the rest of the
table).

Tests: `tests/test_contact_cube_life.py`. `tests/test_2ply_details.py` pins
the table empty, because it compares an N-ply tree's interior with
standalone shallower analyses.

## Interior Move-Pick Validation: Cubeful vs Cubeless (2026-11-19)

  **Measured against a cubeless pick (2026-11-19)**: an experiment switch made
  every interior pick by money cubeless equity instead (same prefilter and
  tie-break), and the benchmarks were rescored with paired per-decision
  differences (cubeless - cubeful, PR):

  | Benchmark | 2-ply | 3-ply |
  |---|---|---|
  | money benchmark (17,535 decisions) | -0.000 +- 0.005 | -0.011 +- 0.007 |
  | money cube decisions vs XG (2,196 positions) | -0.007 +- 0.015 | -0.027 +- 0.014 |
  | 5-point match benchmark (18,292 decisions) | **+0.028 +- 0.009** | **+0.029 +- 0.012** |

  The cubeful pick is clearly better in match play: there the cubeless pick
  plays worse checker moves (3-ply checker +0.034 +- 0.013, checker blunders
  5 -> 9), because it ignores what gammons are worth at the score. In money
  play the two are level within noise (no gap reaches two standard errors,
  blunder counts unchanged), so a money-only cubeless pick would gain at most
  ~0.01 PR, unproven. The rule stays cubeful everywhere. (The 3-ply
  "deeper value" comparison in the #36 notes compared two different trees and
  is not evidence either way.)

## Search Distillation Experiment

### Search distillation with a ranking loss (`scripts/rank_distill.py`, 2026-11-20)

A go/no-go test for training on candidate sets instead of single positions,
with NeonGammon's own 3-ply as a free teacher. `mine` played 8,057 stage11
1-ply self-play games (15% exploration among the top 3) and kept 200,000
contact decisions whose 1-ply top moves were close (up to 6 within 0.12),
843,831 candidates labelled with their 3-ply cubeless post-move probs (13
minutes on the M4 Pro). Benchmark PR sample boards were excluded. `train`
fine-tuned the 13 contact pair nets (Adam, lr 1e-4, up to 20 epochs, best
epoch by held-out pick regret against the teacher). 1-ply checker play routes
every candidate to the pre-move board's net, so each net trains on its own
decisions. Two losses:
- `abs`: MSE of the five probs, the loss of every earlier run;
- `rank`: MSE of the candidates' equities centred on their decision's mean,
  plus `--anchor` x the abs loss.

Scored on the Benchmark PR sample's 4,138 contact decisions (`score`), PR
and paired difference to stage11 at the same depth:

| Variant | vs XG 4-ply 1P / 2P / 3P | diff vs XG, 1P | 2P | 3P |
|---|---|---|---|---|
| stage11 | 2.466 / 1.406 / 0.639 | | | |
| abs | 2.311 / 1.427 / 0.699 | -0.156 +- 0.063 | +0.021 +- 0.049 | +0.060 +- 0.032 |
| rank, anchor 0.1 | 2.238 / 1.477 / 0.667 | -0.229 +- 0.084 | +0.071 +- 0.059 | +0.028 +- 0.039 |
| rank, anchor 1.0 | 2.319 / 1.415 / 0.660 | -0.147 +- 0.083 | +0.008 +- 0.053 | +0.022 +- 0.033 |

The Stage 5 rollout reference agrees in sign (1P: abs -0.18 +- 0.08,
rank 0.1 -0.34 +- 0.10). Paired rank 0.1 - abs at 1P: -0.07 +- 0.08 vs XG,
-0.15 +- 0.09 vs the rollout.

What it shows:
- Distilling 3-ply into the nets improves 1-ply picks, and the ranking loss
  probably more than plain MSE (not significant on its own). So a
  decision-level loss does carry over to decisions where the earlier
  absolute-target runs did not.
- 2-ply and 3-ply gain nothing; the abs loss costs 3-ply ~2 SE. The 2-ply
  and 3-ply trees evaluate opponent positions one or two moves on, as
  absolute values averaged over rolls, which the decision sets do not
  constrain. A lower anchor (better 1-ply ranking) is the worse 2-ply net.
- **The XG referee saturates at 3-ply.** stage11 3-ply scores 0.535 against
  XG 4-ply on all 5,117 decisions, below XG's own #1 (0.64, ranked cubeful)
  and far below the Stage 5 rollout's best moves (1.92). It cannot tell
  whether a change helps at 3-ply; that needs a rollout-grade reference.

Not shipped. The candidates are in `models/rank_diag/` (untracked).


## Stage 5 Scores (historical baseline — NOT the production model)

These are Stage 5's numbers, kept as the baseline the later stages are measured
against. **The production model is `stage11`** (`PRODUCTION_MODEL` in
`python/neongammon/weights.py`); its results live in the Stage 11 sections below and
in `docs/model-benchmarks.md`. The 1-ply contact/race ER and PubEval figures here have
not been re-run for Stage 11 — re-run `scripts/run_full_benchmark.py --model
stage11` before quoting a headline 1-ply ER for the current engine.

| Metric (Stage 5) | 1-ply | Target |
|--------|-------|--------|
| Contact ER | 9.87 | < 10.5 |
| Race ER | 0.95 | < 0.643 |
| vs PubEval | +0.633 | > +0.63 |

Benchmark PR (103k decisions), Stage 5: 1-ply=2.47, 2-ply=1.85, 3-ply=1.53.

## Stage 6 (S6) — Mid-Size Model

**Purpose:** Mid-size model (100h PureRace, 300h contact NNs) between Stage 5 Small
(100h/200h) and Stage 5 (200h/400h). Tests whether 300h contact NNs close the gap
to 400h.

**Weights:** Registered as `"stage6"` in `python/neongammon/weights.py`. Weight files
are `sl_s6_{plan}.weights.best` in `models/`.

**Training:** Same TD + SL pipeline as Stage 5. TD: 200k games @ α=0.1 + 1M @ α=0.02.
SL: same schedule except Racing and Priming use gpw=2.0 (not 5.0 — gpw=5.0 causes
divergence at 300h for Priming, similar to Racing at 200h in S5S).

**Per-plan ER (1-ply):**

| Plan | Stage 5 (400h) | S6 (300h) |
|------|---------------|-----------|
| PureRace | 0.95 | 1.00 |
| Racing | 5.74 | 5.90 |
| Attacking | 8.74 | 8.73 |
| Priming | 8.59 | 9.58 |
| Anchoring | 11.06 | 11.34 |
| **Contact** | **9.87** | **10.09** |

**Summary:** Contact ER=10.09 meets the <10.5 target. Attacking is essentially
identical to Stage 5 (8.73 vs 8.74). Other plans show small regressions from the
reduced hidden size. Priming shows the largest gap (9.58 vs 8.59), likely due to
the lower gpw=2.0 needed to prevent divergence.

## Stage 7 (S7) — 17-NN Pair Strategy Model

**Purpose:** Uses 17 NNs (1 PureRace + 16 contact) selected by the (player, opponent)
game plan pair, instead of just the player's game plan. This allows each NN to
specialize for specific matchups (e.g., attacking vs anchoring). 14 distinct weight
files — 3 rare pairs (prim_prim, prim_anch, anch_prim, anch_anch) share one NN.

**Architecture:** `GamePlanPairStrategy` in `neural_net.h`. Array of 17 NNs indexed by
`pair_nn_index(player_gp, opponent_gp)`. Classification: `classify_game_plan(board)`
for player, `classify_game_plan(flip(board))` for opponent. PureRace selected when
`is_race(board)` is true.

**Hidden sizes:** 100h PureRace, 300h for all 16 contact pair NNs.

**Weights:** Registered as `"stage7"` in `python/neongammon/weights.py`. Weight files are
`sl_s7_{pair}.weights.best` in `models/`. Shared pairs use `sl_s7_prim_anch.weights.best`.

**Training:**
- TD: 300k games @ α=0.1 + 1.5M @ α=0.02 (50% more than S5/S6 due to 16 contact NNs)
- GPW scan: For each NN, test gpw values (1.5, 3, 5, 7, 10) with pair-filtered
  benchmarks. Pick the gpw that minimizes ER on the NN's specific (player, opponent)
  pair subset of the benchmark.
- SL: 4-phase schedule (100ep@α=20 → 200ep@α=10 → 200ep@α=3.1 → 500ep@α=1.0) using
  optimal gpw per NN. Pair-filtered benchmarks used for best-weight selection.

**Pair frequencies in training data and optimal GPW values:**

| NN | Freq | GPW | Pair ER (S7) | Pair ER (S6) | Δ |
|----|------|-----|-------------|-------------|---|
| race_race | 10.0% | 7.0 | 7.26 | 7.97 | -0.71 |
| race_att | 5.7% | 10.0 | 4.72 | 5.00 | -0.28 |
| race_prim | 9.5% | 5.0 | 4.38 | 4.51 | -0.13 |
| race_anch | 11.3% | 7.0 | 5.26 | 5.82 | -0.56 |
| att_race | 5.7% | 5.0 | 5.39 | 6.99 | -1.60 |
| att_att | 3.9% | 7.0 | 9.90 | 9.96 | -0.06 |
| att_prim | 6.3% | 5.0 | 8.31 | 8.33 | -0.02 |
| att_anch | 6.8% | 10.0 | 9.98 | 10.57 | -0.59 |
| prim_race | 9.5% | 7.0 | 8.50 | 9.40 | -0.90 |
| prim_att | 6.3% | 7.0 | 9.71 | 10.04 | -0.33 |
| prim_anch (shared) | 6.8% | 5.0 | 9.15 | 9.60 | -0.45 |
| anch_race | 11.3% | 7.0 | 11.79 | 11.84 | -0.05 |
| anch_att | 6.8% | 7.0 | 10.36 | 10.72 | -0.36 |

**Benchmark results (1-ply):**

| Metric | S7 (17-NN, 300h) | S6 (5-NN, 300h) | S5 (5-NN, 400h) |
|--------|-----------------|-----------------|-----------------|
| Contact ER | 9.76 | 10.09 | **9.66** |
| Race ER | 1.00 | 1.00 | **0.95** |

**Summary:** S7 reduces Contact ER from 10.09 (S6) to 9.76 — recovering 77% of
the gap between S6 and S5 (400h) through pair specialization alone, using the same
300h hidden size. All 13 canonical NNs beat their S6 baseline on pair-filtered
benchmarks. Biggest wins on cross-plan pairs (att_race: -1.60, prim_race: -0.90).

**C++ bindings:**
```python
# 1-ply scoring
bgbot_cpp.score_benchmarks_pair(scenarios, weight_paths, hidden_sizes)

# Multi-ply
multipy = bgbot_cpp.create_multipy_pair(weight_paths, hidden_sizes, n_plies=3)

# Rollout
rollout = bgbot_cpp.create_rollout_pair(weight_paths, hidden_sizes,
    n_trials=360, truncation_depth=7, decision_ply=2, n_threads=16)
```

**Training scripts:**
- `scripts/run_stage7_training.py` — TD training
- `scripts/run_s7_sl_training.py` — GPW scan
- `scripts/run_s7_sl_phase34.py` — Full SL training (phases 3-4)

## Stage 8 (S8) — 17-NN Pair Strategy, 400h Contact, S5 Fallback

**Purpose:** Same 17-NN pair strategy architecture as Stage 7, but with 400h contact
NNs (matching Stage 5 hidden size) instead of S7's 300h. Tests whether pair
specialization + larger hidden layers can beat S5's single-plan 400h NNs. PureRace
weights copied from S7 (same 100h architecture).

**S5 fallback:** After training, each pair NN is scored on its pair-filtered benchmark
subset. Any NN with worse ER than S5's corresponding plan NN is replaced with the S5
weights, guaranteeing no regressions. 10 of 13 pair NNs beat S5; 3 were replaced
(race_prim, att_att, prim_att).

**Weights:** Registered as `"stage8"` in `python/neongammon/weights.py`. Weight files
are `sl_s8_{pair}.weights.best` in `models/`. 3 of the 14 canonical weight files
are copies of S5 plan weights (due to fallback).

**Hidden sizes:** 100h PureRace, 400h for all 16 contact pair NNs.

**Training:** Same pipeline as S7 with these differences:
- Contact NNs: 400h hidden (vs S7's 300h)
- GPW candidates: [2, 5, 7, 10, 12] (vs S7's [1.5, 3, 5, 7, 10])
- PureRace: copied from S7 (not retrained)
- S5 fallback applied after SL training

**Training schedule:**
- TD: 300k games @ α=0.1 + 1.5M @ α=0.02 (same as S7)
- GPW scan: For each of 13 canonical contact NNs, try gpw in [2, 5, 7, 10, 12]
  with phases 1-2 (100ep@α=20 + 200ep@α=10). Pick gpw minimizing pair-filtered ER.
- SL phases 3-4: 200ep@α=3.1 + 500ep@α=1.0 with optimal gpw, starting from
  scan's best weights.
- S5 fallback: replace any pair NN with worse pair-filtered ER than S5.

**Benchmark results (1-ply):**

| Metric | S8 (17-NN, 400h) | S7 (17-NN, 300h) | S5 (5-NN, 400h) |
|--------|-----------------|-----------------|-----------------|
| Contact ER | **9.49** | 9.76 | 9.87 |
| Race ER | 1.00 | 1.00 | 0.95 |
| vs PubEval | +0.633 | — | +0.633 |

**Multi-ply Contact ER:** 1-ply=9.49, 2-ply=8.44, 3-ply=7.76, 4-ply=7.66.

**C++ bindings:** Same as S7 — `score_benchmarks_pair`, `create_multipy_pair`,
`create_rollout_pair`.

**Training script:** `scripts/run_s8_training.py` — unified pipeline (TD + GPW scan +
SL phases 3-4 + S5 fallback + scoring + benchmarks).


## Stage 5 Small (S5S) — Fast Filter Model

**Purpose:** Half-size model (100h PureRace, 200h contact NNs) trained as a potential
fast filter for multi-ply search and truncated rollouts. The hypothesis was that using
a smaller model for 1-ply filtering and a full-size model for leaf evaluations could
speed up 4-ply and Roller++ calculations.

**Weights:** Registered as `"stage5small"` in `python/neongammon/weights.py`. Weight files
are `sl_s5s_{plan}.weights.best` in `models/`.

**Training:** Same TD + SL pipeline as Stage 5. TD: 200k games @ α=0.1 + 1M @ α=0.02.
SL: same schedule except Racing uses gpw=2.0 (not 5.0 — gpw=5.0 diverges at 200h due
to Racing's 37% share of training data dominating the gradient at 74%).

**Per-plan ER (1-ply):**

| Plan | Stage 5 (400h) | S5S (200h) |
|------|---------------|------------|
| PureRace | 0.82 | 1.23 |
| Racing | 5.74 | 6.40 |
| Attacking | 8.74 | 8.75 |
| Priming | 8.59 | 9.07 |
| Anchoring | 11.06 | 12.05 |
| **Contact** | **9.87** | **10.58** |

**Timing results — S5S is NOT significantly faster than S5 at high ply:**

The original hypothesis (2x faster NN → ~1.4-1.7x faster 4-ply) did not hold.
Profiling revealed that the NN forward pass (matrix multiply) is a minority of
total per-node cost at 4-ply depth. The dominant costs are:

1. **Move generation** (`possible_boards`): O(candidates) per dice roll, fixed cost
2. **Input encoding**: Computing 244 extended features (escape counts, containment, etc.)
   is a fixed cost that doesn't shrink with fewer hidden nodes
3. **Position cache divergence**: S5S's noisier 1-ply evaluations produce less consistent
   move ordering, creating more unique positions in the search tree (fewer cache hits)

| Level | S5 Time | S5S Time | S5S Speedup |
|-------|---------|----------|-------------|
| 1-ply | 10.8ms | 7.7ms | 1.40x |
| 2-ply | 88ms | 79ms | 1.11x |
| 3-ply (1T) | 146ms/pos | 96ms/pos | 1.52x |
| 4-ply (1T, cache on) | 1,598ms/pos | 1,533ms/pos | 1.04x |
| 4-ply (1T, cache off) | 4,401ms | 4,634ms | 0.95x (slower) |

With cache disabled, S5S does 3x fewer leaf evaluations but each takes 3x longer
in amortized terms — the fixed overhead (move gen + encoding + filtering) dominates.

**Hybrid evaluator** (S5S filter + S5 leaf) was also tested. It's slower than pure S5
at 4-ply due to overhead from managing two strategy sets. At Roller++ it showed a
modest 1.19x speedup but crashed on large benchmark runs due to memory accumulation
in thread-local caches.

**Conclusion:** Halving hidden nodes does not meaningfully speed up multi-ply search
because the NN matrix multiply is not the bottleneck. Future speedup efforts should
target move generation, encoding, or cache efficiency rather than smaller networks.


## Backgame Stages (S10, S11, S11p, S11s, S11m) and Cube Scoring Investigation

### Stage 10 (S10) — Gated Paskogammon Backgame Hybrid

Stage 10 is Stage 9 plus a **precision-gated committee of a second backgame NN
pair**. It is the full Stage 9 19-NN model **carried unchanged**, plus two extra
backgame NNs trained on Paskogammon-game backgame rollouts, used ONLY inside a
tight structural gate. Outside the gate — including every non-backgame position
and all ordinary (shallow) backgames — S10 is **bit-identical to Stage 9**.

**Architecture (21 NNs).** Indices 0-18 are exactly Stage 9's NNs (including
`sl_s9_player_bg`/`sl_s9_opponent_bg` at 17/18). Indices 19/20 are the extra
Paskogammon-trained backgame NNs (`sl_s10_player_bg`/`sl_s10_opponent_bg`, 400h).

**Gate + blended eval.** `BackgameAwarePairStrategy` accepts 19 **or** 21 NNs.
With 21, a *detected* backgame (same detection as S9) is routed to the committee
only when the backgame side has

```
anchors >= 3   OR   (anchors == 2  AND  back checkers >= 7  AND  pips >= 200)
```

Inside the gate both backgame NNs evaluate the same inputs and the probs are
mixed, `probs = (1-w)·base + w·pasko`, with an anchor-zone weight (**0.80** for
4+ anchors, **0.70** for 3, **0.60** for the massive-2-anchor arm) scaled by a
pip ramp (×0.5 at ≤170 pips → ×1.0 at ≥230). Constants: `BACKGAME_GATE_*` /
`BACKGAME_BLEND_*` in `neural_net.h`. `select_nn_idx` reports gated positions
via sentinels 21/22; the batch delta-eval kernels evaluate those per-candidate
(no shared delta base). Non-gated backgames return 17/18 and keep the fast
paths. With 19 NNs everything is off (pure S9).

**Why a gate.** The pasko nets are decisively better on *deep/massive*
backgames but indistinguishable-to-worse on ordinary ones, and no board feature
separates standard-game from pasko-game shallow backgames — an ungated blend
improved the neutral backgame benchmarks but leaked small divergences from S9
into full-game rollout-PR scores everywhere. The gate confines the pasko nets
to the region where they provably win (verified on 60.7k rolled-out backgame
positions): **1-ply ER, Paskogammon backgame benchmark: S9 43.24 → S10 35.5**
(gated slices: 3-anchor 46.98→30.65, 4+-anchor 92.38→48.71, massive-2-anchor
50.58→31.62); **standard backgame benchmark: S9 23.10 → S10 23.05** (the gated
regular slices improve slightly too; both even/odd halves ≤ 23.06). Gate
footprint: ~43% of pasko-game backgame evals, ~9% of standard-game ones — so
standard-game play is byte-equal to S9 outside rare deep backgames.

**Where it lives.** `backgame_gate` + `backgame_blend_weight` +
`blended_backgame_probs` + `select_nn_idx` in `cpp/src/neural_net.cpp` (sentinel
handling in the batch methods and in `cube_eval.cpp`'s `eval_groups_pair`);
constants in `neural_net.h`; the `stage10` registry entry (`extra_backgame`
list) and 21-length `plan_names` in `python/neongammon/weights.py`. Weight files
(**v3** — the pasko nets retrained on the S10-gated benchmark/train sets and
warm-restarted twice): `sl_s10_player_bg_v3.weights.best`,
`sl_s10_opponent_bg_v3.weights.best`. The original shipped nets
(`sl_s10_*_bg.weights.best`) are kept for A/B.

**Retrained pasko nets (v3).** The gate footprint (~9%/43% of standard/pasko
backgames) barely intersects the full-game benchmarks, so the pasko nets are
better re-benchmarked and re-trained on a **gated-only** backgame set — the
positions where the blend weight > 0 (`generate_pasko_data.py --gate`). On a new
20k/side gated benchmark, 1-ply cubeless ER: **S9 49.51 → S10 v1 31.62 → S10 v3
28.97** (−42% vs S9). v3 = SL on the gated 100k/side train set warm-started from
the shipped v1 nets, then one warm-restart. (Warm-starting from the TD net lands
~1.4 ER worse; a further restart gains <0.1 — converged.)

### Stage 11 (S11) — Backgame-Aware Phased Model (Promoted to Production)

**The registry has ONE Stage 11 entry, `stage11`: the 24-NN phased layout**
(Stage 9's 17 standard nets + the category trio + P3 + containment + snake +
massive), i.e. what the sections below arrived at. The intermediate names
those sections use — the 20-NN trio, `stage11p` (22), `stage11s` (23),
`stage11m` (24) — are the history of how it was built and no longer
resolve; the C++ strategy still accepts 22 or 23 paths for A/B work.
**`stage11` is the production model** (`PRODUCTION_MODEL` in `weights.py`).

#### The categorized backgame trio (the first step)

Stage 11 replaces Stage 9's two backgame NNs (player/opponent) with **three,
selected by the backgame's CATEGORY — the same NN whichever side holds it**:

| Index | NN | Categories | Anchor rule |
|-------|----|-----------|-------------|
| 17 | `bg_deep` | 21, 31, 32 | both anchors on the 1/2/3 points |
| 18 | `bg_middle` | 41, 42, 51, 52 | one anchor on the 1/2 point, one higher |
| 19 | `bg_double` | 43, 53, 54 | exactly two anchors, none deeper than the 3-pt |

With 3+ anchors: deep when at least two sit on the 1/2/3 points, else middle
(never double). The 6-point counts as an anchor (Stage 9 detection convention);
pairs with a 6-anchor map by the same min/max rules ({1,6}/{2,6} middle, the
rest double). **Detection itself is Stage 9's, unchanged**: plan pair
(anchoring, racing), backgame side behind on pips, 2+ anchors in the opponent's
home board. The category is perspective-invariant.

**Where it lives.** `BackgameCategory` + free `backgame_category(board)` +
`category_from_anchor_mask` / `backgame_category_given_plans` in
`neural_net.h/.cpp`; `BackgameAwarePairStrategy` accepts **20** NNs
(`NUM_BACKGAME_PAIR_NNS_CATEGORIZED`) and routes detected backgames to
17/18/19 by category — with 19 or 21 NNs behaviour is exactly S9/S10. The
`stage11` registry entry in `weights.py` carries S9's 17 standard NNs
unchanged plus `sl_s11_bg_{deep,middle,double}.weights.best` via
`extra_backgame` (plans type `backgame_pair_categorized`; dispatches to the
same C++ class). Python: `bgbot_cpp.backgame_category(board)` returns
`"deep"/"middle"/"double"/"none"`. Tests: `tests/test_backgame_category.py`.

**Training step 1 — truncated TD** (`td_train_backgame_truncated` in
`training.h/.cpp`, driver `scripts/run_s11_backgame_td.py`): each backgame NN
starts from random small weights and trains by TD(0) self-play with games that

* start from the category's reference positions
  (`backgame_ref_positions/benchmark/<folder> starting.txt`, cycled, a coin
  picking the first mover);
* are played with 1-ply decisions by the training NN;
* END when a post-move position is no longer in ANY backgame category — the
  terminal TD target is then **Stage 9's 3-ply cubeless post-move eval** of
  that position, standing in for the game outcome exactly as the 0/1 outcome
  vector does at a real terminal (which still ends the game the ordinary way —
  a backgame can lose by bear-off with the anchors still held).

Progress is scored pasko-style (mean |equity − target| × 1000) against the
`data/*-backgame-benchmark-rollout` rows falling in the training category.
Outputs `models/td_s11_bg_<cat>.weights(.best)`; promote by copying the three
`.best` files to the `sl_s11_bg_*` names the registry points at.

```bash
py -3.14 scripts/run_s11_backgame_td.py --category deep --n-games 200000
py -3.14 scripts/run_s11_backgame_td.py --category all
```

**Training step 2 — SL on rollout targets** (the quality stage, mirroring how
Stage 9's backgame nets were made). ``scripts/segregate_s11_backgame_data.py``
splits every backgame-labelled rollout pile from the S9/S10 eras (the
player/opponent S9 sets plus the three pasko families; ``.s8`` targets and the
position-only ``*-data`` files skipped) into
``data/s11-bg-<cat>-{train,benchmark}-rollout`` by ``backgame_category`` —
about 245k/167k/94k train rows for deep/middle/double, benchmarks held out
and deduplicated against train. Every row in those piles passes S11 detection
(0 dropped), and ``pasko-gated-remainder`` is entirely duplicates of the gated
sets. ``scripts/run_s11_backgame_sl.py`` then trains each category NN with the
S9 backgame recipe (GPU ``sl_train.supervised_train_preencoded``, 2,500-epoch
chunks at batch 4096, 100k ep @ a=3.1 then 250k @ a=1.0, best-ER
checkpointing), warm-started from the TD bootstrap by default, writing
``models/sl_s11_bg_<cat>.weights.best`` — the exact filenames the ``stage11``
registry loads. ~2 h per category on upstream's RTX 4070S.

**Measured path lengths** (why truncated games are short): under S9 1-ply play
from the deep seeds, the region exits after a median of ~9 half-moves — mostly
because the PLAN PAIR flips (the racer starts `attacking`, or the backgame side
reads as `priming`) while the anchors are still physically held, which is
inherited Stage 9 detection behaviour. Under a random-init NN it is ~4
half-moves, lengthening as the NN learns. The S9 3-ply reference eval is ~10 ms
cold and cached across games (16 recurring seeds), so TD throughput stays high.

**The flee equilibrium — why `--anchor-boundary` exists** (measured on a 200k
deep run, 2026-08-29). Plain truncated TD converges CORRECTLY to the value of
the truncated game under its own play — and that game's optimal policy is
degenerate. Interior states are worth what the net's own (weak) continuation
makes them (net −1.02 vs an independent on-policy Monte-Carlo estimate −1.01,
corr 0.93), while crossing the boundary swaps the appraiser to Stage 9, whose
grades assume EXPERT continuation (backgame-side exits −0.79 vs the true
good-play interior value −0.60; measured play-quality wedge S9 − V_pi = +0.39).
So handing the position to the expert appraiser beats keeping it, the backgame
side (which holds exit moves on 77% of its rolls; the racer holds 0%) exits in
median 1 half-move, paths collapse (3.8 → 2.8 over training while the
stay-vs-exit value gap grows −0.03 → +0.24 and exit-choice rises 42% → 94%),
and the benchmark ER plateaus (~262 by game 15k, flat thereafter, vs S9's 22).
It is one-sided because the subsidy is proportional to each side's own
incompetence gap — the racer's decisions here cost only ~0.03/move vs S9, so it
has no wedge to chase and every true-equity reason to welcome the opponent's
exits. `anchor_boundary` adds one supervised update per game — the exit
position itself toward its S9 eval — which re-grounds the slice to expert
scale: at 15k games, exit-choice 72% → 21%, backgame-side move cost
0.216 → 0.046 equity, path median ~10 half-moves (S9-play territory), ER 250
and still falling where the plain run had plateaued. `boundary_extra_plies` (the grace period) attacks
the same disease from the other side: the path plays N more half-moves past
the first out-of-region position (updates continuing, the net still moving)
before the Stage 9 grade is taken, so the boundary zone becomes trained input
and fleeing means living with your own play N moves longer. Measured on
30k-game deep runs (play cost = mean equity lost per backgame-side seed
choice vs S9-3P; N=0 is the plain trainer, byte-verified):

    N=0: cost 0.202, exits 71%, ER 301 (the flee equilibrium)
    N=2/4/6/10: cost 0.045/0.043/0.042/0.041, exits ~19-26%, ER 260/252/225/222
    anchor only: cost 0.033, ER 226
    N=6 + anchor: cost 0.034, ER 206 (best) and still falling

Even N=2 removes the equilibrium; larger N mainly lowers target noise (the
grade lands where positions are settled and Stage 9 is reliable). The
recommended mode for any real S11 training run is anchor_boundary +
boundary_extra_plies ~6 — the driver's defaults. A 30k-game A/B/C
(`ref_move_games`: S9 picks the moves for the first N games) showed the
S9-move warm-up is not it: pure S9-move training gives the best VALUE fit
(ER 223, and in-region values within 0.04 of S9's) but 4x worse PLAY at the
seeds (0.122 equity lost per backgame-side choice vs 0.033 for anchored
self-play) — value accuracy on S9's trajectories does not calibrate the
off-distribution candidates greedy selection must compare, and the exit
ordering inverts again. A 15k warm-up then pivot lands exactly where pure
self-play lands (0.034). The net should make its own moves from game 0.

### Stage 11p — Historical: phased layout (P3 + containment NNs)

`stage11p` is the trio model plus two more NNs, **22 in all**
(`NUM_BACKGAME_PAIR_NNS_PHASED`, strategy type `backgame_pair_phased`, the
`BackgameAwarePairStrategy` constructor flag `phase_containment`):

| Index | NN | Selected when |
|-------|----|---------------|
| 17-19 | `bg_deep` / `bg_middle` / `bg_double` | a detected backgame, by category (the trio above) |
| 20 | `bg_p3` | **early containment** phase: the backgame side still holds >= 2 anchors and is behind, has hit (>= 1 straggler), and the racer has <= 2 off (`backgame_phase()` == `EARLY_CONTAINMENT`) |
| 21 | `bg_containment` | the **containment rule** below |

`select_nn_idx` tests them in the order containment rule -> category trio ->
P3 phase -> standard pair NN, so the containment NN wins whenever its rule
fires. Everything else is Stage 9. Python: `bgbot_cpp.backgame_phase(board)`
(`"waiting"/"bear_in"/"early_containment"/"late_containment"/"none"`) and
`bgbot_cpp.containment_category(board)`.

**The containment rule is escaper-centric** (`containment_category` in
`neural_net.cpp`, mirrored by `scripts/containment_rule.py`, which the tests
hold the C++ to): the position is a containment game when one side, the
ESCAPER, has borne off >= 3 checkers (`CONTAINMENT_E_OFF_MIN`) and has 1-3
stragglers (`CONTAINMENT_STRAGGLERS_MAX`) — checkers on the bar or outside its
home board with a container checker still ahead of them — and there is
contact. Nothing is required of the container: no anchors, no particular
structure, blots allowed. The rule is flip-invariant. Measured footprint:
99% of the `containment` family folder, 39% of `Positions XG gets wrong`,
1.3% of the money benchmark, 2.6% of pasko, <= 1.2% of the ten backgame
folders (their windows sit in the waiting phase; a hit shot there routes to
P3, 8-21% of their positions, because the racer has <= 2 off). The P3
phase's single-anchor fallback was tried and rejected (money PR +0.75).

**Blend sentinels.** `BLENDED_PLAYER_BG_IDX` / `BLENDED_OPPONENT_BG_IDX`
were 21/22, compared with `>= NUM_BACKGAME_PAIR_NNS_HYBRID`; the phased
layout's slot 21 collided with them, so every batch evaluator treated a
containment decision as a gated S10 blend and slot 21 was never read (two
different containment nets scored byte-identically). They are now 1000/1001
(`BLENDED_SENTINEL_BASE`) and `tests/test_backgame_category.py` asserts a
containment board evaluated through the phased strategy equals the
containment NN evaluated directly.

**Data.** `scripts/harvest_containment_positions.py` plays 3,000 two-ply
stage11 self-play games from the containment seeds (every benchmark decision
and candidate board, both orientations, excluded) and keeps the rule's
positions; `scripts/split_containment_targets.py` moves the ones the S9-era
piles had already rolled into `data/s11-bg-containment-pile-rollout` (9,734)
and leaves `data/s11-bg-containment-data` (35,266 fresh boards), rolled out
under S9 play at 1,296 paths / 3-ply into `data/s11-bg-containment-rollout`
(Fargate Spot, $176 realized). `scripts/extract_general_containment_rows.py`
adds the rule's rows from the main corpus — 16k of `contact-train-data`
(1.3%) and 275k of `crashed-train-data` (46%: closeouts and late-hit
bear-offs), GNUbg targets — as `data/s11-bg-containment-general-rollout`.

**Training** is `scripts/run_s11_containment_sl.py` (GPU SL, 20k epochs @
alpha 3.1 then 60k @ 1.0, best-holdout-ER checkpointing). The installed net
is the S9 `prim_race` warm start trained on family rows x4 plus the general
rows (`--extra-data s11-bg-containment-general-rollout --family-weight 4`).
The family-only version was the narrow-subset failure [docs/training.md](training.md) warns about
elsewhere: on the money benchmark's 237 containment-routed decisions it
scored PR 4.62 with 6 blunders against 2.91 / 2 for Stage 9's ordinary nets,
while ordinary-game containment simply never appeared in its data. The
widened set restores that slice to 3.03 and improves the family folder too
(on the then-uncompleted reference, 1P PR 11.05 -> 10.12 against S9's 11.82;
the completed-reference numbers are in the table below). Warm-starting from
the S9-initialised deep net (14.59) or fine-tuning the family-only net on the
widened set (tied on the folder, worse on the money slice and on `Positions
XG gets wrong`) both lost. The general rows carry GNUbg's targets.

Re-rolling them under S9 did NOT help (2026-11-18). All 291,242 general rows
were rolled out on Modal under the S9 convention
(`data/s11-bg-containment-general-s9-rollout`, ~$223, mean SE 0.006; the S9
targets differ from GNUbg's by 0.029 equity on average). The installed recipe
with those targets in place of GNUbg's (prim_race warm start, family x4)
produced `models/s11_diag/sl_containment_s9gen.weights.best`. Its holdout ER
on the same split fell sharply, 36.8 -> 24.4 on the general rows and
37.8 -> 34.5 / 31.1 -> 28.6 on the two family files, but its decisions did
not improve.

On the money benchmark's 237 containment-routed decisions (installed / new,
PR at 1P / 2P / 3P):

| Net | 1P | 2P | 3P |
|---|---|---|---|
| installed | 3.06 | 1.94 | 1.18 |
| new | 3.50 | 2.05 | 1.28 |
| paired difference | +0.45 +- 0.35 | +0.11 +- 0.42 | +0.11 +- 0.15 |

Most of the 1P loss is cube (2.69 -> 5.11).

On the containment folder it is a tie. Bracketed by PR(RO), because the new
net leaves the completed candidate set on 1-2% of its picks:

| Reference | installed (1P / 2P / 3P) | new (1P / 2P / 3P) |
|---|---|---|
| S9-play | 6.77 / 4.74 / 2.52 | 6.11-6.80 / 4.58-4.85 / 2.38-2.60 |
| S11-play twin | 6.17 / 5.67 / 2.22 | 5.67-5.96 / 5.79-6.51 / 2.50-2.68 |

So the installed net stays. This is the same result as the pure-race
retraining: holdout ER against better targets is not a lever for decision
quality here.

**Candidate completion.** A folder reference rolled out only the candidates
the reference player's 2-ply filter kept; the phased model's picks fall
outside that set far more often than Stage 9's (56% of its containment-folder
error mass sat on filter-graded picks at 1P, 63% at 3P), which over-charges
it. The parent repo's `scripts/build_rollout_jobs_containment.py` (both
references of a family folder, `--category`, `--models`, `--key-suffix`) and
`scripts/build_rollout_jobs_folders_stage11p.py` force those picks into
rollout jobs; `scripts/merge_candidate_rollouts.py` splices the results in.
Read the `PR(RO)` and `filt mass` columns of `score_backgame_pr.py` before
trusting a headline PR on any of these folders.

**Results (2026-09-03).** Every pick of every model rollout-graded (one 3-3
decision in `Positions XG gets wrong` excepted — its 3-ply trials run into
hours-long containment endgames and the batch was abandoned). PR, with
blunder counts where they matter:

| Benchmark (1P unless noted) | Stage 9 | trio (`stage11`) | `stage11p` |
|---|---|---|---|
| containment folder, 3,000 decisions, 1P / 2P / 3P | 10.23 / 10.79 / 6.21 | 9.40 / 10.20 / 5.10 | **6.83 / 4.94 / 2.82** |
| ... blunders at 3P | 140 | 99 | **40** |
| containment, S11-play sample (300), 1P / 2P / 3P | 8.76 / 8.58 / 4.21 | 8.50 / 8.72 / 3.04 | **6.16 / 5.47 / 2.13** |
| Positions XG gets wrong (421), S9-play / S11-play ref | 16.22 / 16.63 | 14.20 / 16.32 | **10.01 / 12.26** |
| ten backgame folders pooled (10,021), blunders | 4.75 (278) | 3.28 (127) | **2.98 (89)** |
| money benchmark (17,535) | 2.59 | 2.56 | 2.56 |
| ... its 237 containment-routed decisions | 2.91 | 2.91 | 3.03 |
| Paskogammon benchmark (2,556) | 7.70 | 6.85 | **6.52** |

Two things to keep in mind when reading it. Completing the candidates moved
Stage 9's own containment number from 11.82 to 10.23 — its filter-graded
picks had been over-charged too — so never compare a completed reference
with an uncompleted one. And the 2-ply CUBE numbers in it (Stage 9's cube
2.77 -> 7.23 -> 2.77 across 3P/2P/1P) carried the opponent-owned-cube
scoring artifact described under "Cube scoring: the opponent-owned-cube
artifact" below; the checker-play 2P-worse-than-1P shape for Stage 9 and
the trio is the data-coverage effect the Stage 11s section diagnoses.

### Stage 11s — Historical: the snake NN (index 22)

`stage11s` is `stage11p` plus a 23rd NN for the **snake**: a far-side prime
trapping a straggler against a crunched board. `snake_category(board)` —
mirrored by `scripts/snake_rule.py`, and the same rule as the benchmark's
`snake` family filter in `scripts/backgame_benchmark.py`, which the tests
hold all three to — fires when either side holds a run of >=
`SNAKE_PRIME_MIN_POINTS` (4) consecutive points, each with >= 2 checkers,
entirely on the opponent's half of the board, the opponent has >= 1
straggler on the bar or in the holder's home board, and >= `SNAKE_MIN_HOME`
(10) checkers already in its own home board. It is a priming / containment
structure, not a backgame; it reads as one to Stage 9's plan-pair gate only
because the holder's points sit in the opponent's home board and the holder
trails in the race, which is why the trio's double and middle nets serve it
today at 1-ply PR 200 and 109. The phased strategy accepts 22 or 23 NNs;
with 23 a snake routes to NN 22 ahead of everything else.

**The region has no data and never occurs in play.** ~100 rule rows in the
1.8M-row GNUbg corpus, ~36 in the Stage 11 piles, none in the pasko or S9
backgame piles, and 0 positions in the money and pasko benchmarks and in
every other folder — so a snake NN carries no regression risk anywhere, and
everything it learns from has to be generated. Self-play from the benchmark
seeds cannot generate it: measured 2026-09-03, stage11p (snake PR ~54)
breaks the prime or runs within a move or two, so 900 games yielded ~2.4
snake decisions each. `scripts/harvest_snake_positions.py` therefore
samples the region directly — 5,000 random synthetic snakes (prime length
and location, spares spread from the far side to home, a crunched opponent
with 0-5 off and 1-3 stragglers) seed short 2-ply games in which the holder
prefers structure-keeping moves — and keeps, per decision, the trajectory
position plus its top-5 candidates and up to 3 region-leaving candidates
(routing is by the pre-move board, so the snake NN is the one asked to value
the release moves): 18,557 decisions -> 98,515 distinct boards,
`data/s11-bg-snake-data` holds 44,000 of them. Note the rule's own edge: a
straggler that reaches the holder's OUTER board is still trapped behind the
prime but no longer counts, so trajectories leave the region after ~3-4
decisions; the benchmark is defined the same way, so the routing matches it.

**Snake rollouts cost ~6x containment ones.** Measured on ten random
harvested boards at the reference convention (1,296 paths, 3-ply in-trial
play): 78.6 s each on the 32-thread dev box against ~12 s for containment
positions, because the holder must bring 15 checkers round before the
trapped side bears off. 648 paths / 3-ply takes 35.8 s, 1,296 / 2-ply
22.3 s. The first rollout (run `0d6c74fc8753`) therefore used 648 paths at
3-ply on the Batch backend — 3-ply in-trial play was kept because Stage 9's
2-ply play in these regions is markedly worse than its 3-ply (containment
folder 10.79 vs 6.21), which biases every target, whereas halving the paths
only widens the target SE from ~0.009 to ~0.013 — for the first 22,000
boards of the file. `rollout_backgame_positions.py --n-trials / --checker-ply`
(parent repo) bakes the configuration into the cloud-pickled task function.
The run realized ~$159 on Batch (64 workers, 9.4 h); the folder's
candidate-completion pass (2,062 boards at the reference convention) a
further ~$75 on Fargate.

**Training and results (2026-09-04).** `run_s11_containment_sl.py
--family-data s11-bg-snake-rollout --out-prefix sl_snake` from the S9
`prim_race` net: holdout ER 299.6 -> 53.95 in 2.5 minutes (a schedule is
passes over the data; a 4x schedule plateaus at 53.6 and scores slightly
worse on the folder, and the trio's deep net as warm start is worse again).
Installed as `models/sl_s11_bg_snake.weights.best` (replaced on 2026-09-05 by
the round-2 `sl_snake2_warm` net, below). Snake folder, every pick
rollout-graded, PR at 1P / 2P / 3P: Stage 9 63.56 / 48.06 / 32.18 (364 /
286 / 205 blunders); trio 52.45 / 34.72 / 28.49; `stage11s` **15.71 / 35.65 /
24.35** (141 / 251 / 165). The 1-ply figure is the first real competence any
NeonGammon net has shown in the region. **Why 2-ply scores worse than 1-ply
— measured 2026-09-04** on the 795 checker decisions (cubeless 1-ply PR
16.0): two separate effects, one a bug and one not.

1. *A ranking bug in the multi-ply candidate list.* The filter keeps the top
   five 1-ply candidates within 0.08 and re-scores them at N-ply, and the
   list was then sorted with the PRUNED candidates still on their 1-ply
   values. Whenever the deeper search values every survivor below a pruned
   move's 1-ply number — exactly what happens where N-ply matters most —
   that move topped the list without ever being evaluated: 312 of the 795
   cubeless 2-ply picks were such moves, carrying 55 of that level's 68 PR
   points, against 31 for a faithful 2-ply of the survivors alone. The
   cubeful wrapper was immune (it then re-scored every candidate through the
   cube-aware N-ply tree; at 2-ply it no longer does, see "The cause of the
   snake 2-ply gap"), so the benchmark tables never saw it, but the C++
   `batch_checker_play` overloads — the app's analytics path — promoted only
   the runner-up in the same way. Both now promote whichever of the top two
   still carries a 1-ply value until both are full-depth
   (`_first_stale_top_two` in `analyzer.py`, the promotion loops of
   `score_checker_position` in `bindings.cpp`).
   The rollout path always had the right rule (promote #1 until it is
   rollout-grade); it is now the rule everywhere.

2. *The search walks out of the training distribution.* Against the rollout
   references the snake net's 2-ply values are BETTER than its 1-ply ones on
   candidates that keep the snake (RMSE 0.106 vs 0.136, holder on roll) and
   much WORSE on candidates that release it (0.204 vs 0.151): the leaves
   below a release are one half-move beyond every position the harvest
   rolled out, and there no net is any good — the router hands them to the
   standard nets, whose error on the harvest's own exit rows is RMSE 0.32
   (bias −0.10), while the snake net fits those same rows at 0.047. A
   hold-versus-release comparison therefore pits a good estimate against a
   bad one and the max picks the noise: pick-PR among rollout-graded
   candidates 16.3 at 1-ply, 27.7 at 2-ply. Routing the whole tree by the
   ROOT position's net helps only a little (27.8 vs 31.2 at 2-ply) because
   the snake net has never seen an exit+1 position either. The fix is data,
   not routing: roll out the opponent's replies to the exit boards (and
   their replies, for 3-ply) and train the snake net on them, under root
   routing. It is the containment net's lesson extended to depth — a
   specialist must cover the neighbourhood the search visits, not only its
   own region — and the containment folder's 2P-worse-than-1P shape for
   Stage 9 and the trio is the same effect.

**Round 2 (2026-09-04): root routing, the exit-reply data, and the widened
containment rule.** Both remedies for effect 2 are implemented; choosing
between them waits on the candidate-completion rollout of their picks (see
the table at the end of this section).

*Root routing.* A decision's whole tree is evaluated with the net chosen for
its ROOT position. `Strategy::root_pin_for(root)` (virtual, -1 = route per
node) is answered by `BackgameAwarePairStrategy` with `select_nn_idx(root)`
whenever that index is in `root_pinned_`, which defaults to `{22}` on the
23-NN layout and to nothing on every other layout, so Stage 9 through
`stage11p` are untouched; `BearoffStrategy` forwards it. The cubeful engine
carries the pin in `EvalCtx.pinned_nn`: `eval_groups_pair` (the batched
interior picks) and `pinned_post_probs` (leaves, dances, forced moves and the
1-ply entries) evaluate every contact position with the pinned net, while
races and exact bearoff positions keep their own values. It enters through
`root_board` on `cubeful_equity_nply_multi` / `cubeful_probs_and_equity_nply`
(bound as `root_board=`; the analyzer passes the decision's pre-move board)
and through `cubeless_tree_probs`, which passes its pre-move board;
`cube_decision_nply_multi` pins from the position itself. The cubeful eval
cache folds the pin into its fingerprint. Switch it off with
`set_root_pinned([])` on the strategy (bound on both classes — the analyzer's
`_strategy_1ply` is the bearoff wrapper) or process-wide with
`NEONGAMMON_NO_ROOT_ROUTING=1` in the environment at construction.

Two leaks made the first measurements wrong, and are why
`tests/test_backgame_category.py::test_root_routing_pins_leaves_and_cache_through_nested_wrappers`
exists: `pinned_post_probs` unwrapped ONE bearoff layer, but the analyzer
wraps its strategy in `BearoffStrategy` and the binding wraps that again, so
every dance and forced-move leaf silently fell back to the board's own
routing (the pinned 2-ply snake number moved 29.73 -> 37.27 when the widened
rule below changed where those leaves routed, with the pin verified on every
root); and the cache fingerprint ignored the pin, so a pinned node could be
served by an unpinned evaluation of the same board from an earlier call on
the same thread.

*Exit-reply data.* `scripts/harvest_snake_exit_replies.py` builds
`data/s11-bg-snake-exit1-data`: for every exit board in the snake harvest the
opponent's best reply under root routing, plus 4,000 replies picked with
per-node routing — 25,892 boards, rolled out at 648 paths / 3-ply in-trial
play (Fargate, 1,000 workers, 62 min, ~$218) into
`data/s11-bg-snake-exit1-rollout`. `models/s11_diag/sl_snake2_{primrace,warm}.weights.best`
are the snake recipe on the snake rows plus these (warm = from the installed
snake net, primrace = from S9 `prim_race`).

*The widened containment rule* — the proposal that a broken snake is a kind
of containment game, so the containment net should cover the hand-off.
`containment_category` gains a "crunched" arm: an escaper with fewer than
`CONTAINMENT_E_OFF_MIN` off still qualifies when it has >=
`CONTAINMENT_CRUNCHED_HOME_MIN` (10) checkers in its home board and the
container has >= `CONTAINMENT_CONTAINER_FAR_MIN` (8) checkers with >=
`CONTAINMENT_CONTAINER_FAR_POINTS_MIN` (3) made points on the escaper's half;
the straggler and contact conditions are unchanged, and
`scripts/containment_rule.py` mirrors it. Measured footprint: 92% of the
snake harvest's exit boards and 92% of the exit replies; 98.2% of the
containment folder's decisions; money 1.3% -> 1.3%; pasko 2.6% -> 2.9%; the
ten backgame folders 0.7%; massive 4.6%. (A first draft with the home
threshold alone claimed 23.5% of the money benchmark; the container clauses
are what confine it.) Its data: `data/s11-bg-containment-snakeexit-rollout`
(the 10,078 harvest exit rows and 21,522 exit replies the arm claims, with
their rollout targets) and `data/s11-bg-containment-general-wide-rollout`
(301,976 GNUbg rows under the widened rule;
`extract_general_containment_rows.py <output name>`).
`models/s11_diag/sl_cont2_{warm,primrace}.weights.best` train the containment
recipe (family x4) on those plus the original family rows.

*Results on completed references (2026-09-04 evening).* Two completion
passes rolled out every pick of every configuration below — the three snake
nets' root-routed picks (709 boards) and the two containment candidates'
per-node picks on the containment and snake folders (569 boards); ~$30 and
~$50 realized on Fargate against $169 estimates each, because single-board
completion tasks finish in minutes rather than the estimator's twenty — so
every number here is rollout-graded. PR at 1P / 2P / 3P, S9-play reference
first, S11-play twin second:

| Configuration (`stage11s`) | snake, S9-play | snake, S11-play twin |
|---|---|---|
| root routing, round-1 snake net | 16.75 / 17.34 / 25.22 | 18.79 / 18.66 / 15.15 |
| root routing, `sl_snake2_primrace` | 16.37 / 17.67 / 25.11 | 19.60 / 19.87 / 15.35 |
| root routing, `sl_snake2_warm` (**installed 2026-09-05**) | 15.13 / 16.80 / 24.20 | 17.58 / 19.61 / 15.05 |
| per-node, widened rule, old containment net | 16.75 / 46.35 / 39.18 | 18.79 / 45.00 / 29.69 |
| per-node, widened rule, `sl_cont2_warm` | 16.75 / 15.68 / 23.99 | 18.79 / 18.64 / 14.17 |
| per-node, widened rule, `sl_cont2_primrace` | 16.75 / 15.10 / 23.57 | 18.79 / 18.18 / 14.23 |
| per-node, widened rule, `sl_cont2_primrace` + `sl_snake2_warm` | 15.13 / 15.95 / 23.71 | 17.58 / 18.32 / 15.77 |

What the table says. (1) The 2-ply anomaly is gone under either remedy:
1-ply and 2-ply are level, and per-node routing collapses to ~45 only while
the exits reach a containment net that has never seen one. (2) The two
references disagree about 3-ply in opposite directions — the worst level
under S9-play rollouts, the best under S11-play ones — and the split sits
in the values a weak trial player distorts most, the release-versus-hold
checker picks and the cube (cube PR 13-15 against 6-9 at 3-ply). Stage 9
plays these positions at PR 63, so its rollouts are the shakier reference,
but neither is authoritative; a reference rolled out under `stage11s` play,
or an XG pass, would arbitrate. (3) `sl_snake2_warm` is the best snake net
at 1-ply on both references. (4) The widened rule's retrained containment
nets are the stronger snake mechanism at 2-ply and 3-ply, by about 1 PR.

The table is stale. It predates both the rebuild of the snake references on
2026-09-07 (`99e6701`) and the money cube-life change of 2026-11-18. With
today's engine and references, the installed `stage11` scores
20.58 / 20.38 / 21.47 on the S9-play reference and 18.21 / 21.69 / 16.87 on
the twin.

*XG as referee (2026-11-19).* XG "World Class" settles point (2): 3-ply is
the best level and 2-ply the worst.

How the run worked:
- `scripts/export_folder_benchmark_xg.py` wrote both references' decisions
  as one-decision `.xg` games, which XG Batch-Analyzed.
- `scripts/score_folder_vs_xg.py` scores the picks (`picks` computes them,
  `score` grades them). It grades a pick only against values at XG's top
  level for that decision. The top level was Roller+ for 447 decisions,
  4-ply for 380, 3-ply for 47 and 2-ply for 77 (S9-play reference; the
  twin is within 4 of each).
- XG takes its leading candidates to full depth and leaves the rest
  shallower. In snakes those shallow values are off the leaders' scale:
  graded against them, XG's own #1 cost PR 61 against XG.
- XG always analyses the played move at full depth. So
  `score_folder_vs_xg.py complete` writes a second set of games that play
  each of NeonGammon's picks XG had left shallower (1,244 games). With that pass,
  every decision is graded.

Results, PR (paired differences +- 1 SE). The twin row covers 959 decisions,
the S9-play row 951:

| Reference | 1P / 2P / 3P vs XG | 3P - 2P | 3P - 1P | 2P - 1P | same, vs the rollout |
|---|---|---|---|---|---|
| S9-play | 35.83 / 38.85 / 32.52 | -6.33 +- 1.65 | -3.31 +- 1.88 | +3.02 +- 1.94 | +1.08 / +0.89 / -0.20, all within 1 SE |
| S11-play twin | 33.98 / 37.58 / 31.30 | -6.28 +- 1.73 | -2.68 +- 1.80 | +3.59 +- 1.79 | -4.82 +- 1.31 / -1.34 +- 0.95 / +3.48 +- 1.26 |

What it shows:
- XG agrees with the twin on all three level differences.
- On the rebuilt S9-play reference, 3-ply is no longer worst: the three
  levels are level within noise. The split of the table above is gone.
- Cube PR against XG falls with depth on both references: 25.2 / 19.9 /
  17.5 and 28.7 / 24.1 / 21.7.
- 2-ply is still worse than 1-ply in snakes, by ~3 PR under XG and the
  twin. So the "2-ply anomaly is gone" of point (1) holds only on the
  S9-play reference. Picks from before the cube-life change show the same
  gap (+3.3 / +3.8 vs XG), so it is not caused by that change.
  The cause and the fix are in the next paragraph.

*The cause of the snake 2-ply gap (2026-11-19).* Cubeful checker play
scored every legal move in the cube-aware 2-ply tree. It skipped the 1-ply
move filter that cubeless N-ply applies. In 103 of
the 795 snake checker decisions the pick was a move the filter had pruned:
- At 1-ply that move ranked 7th (median), 0.11 behind the filtered pick.
- 2-ply valued it 0.044 above the filtered pick.
- XG and both rollouts put it 0.04-0.06 below.
- 92 of the 103 are holder decisions, mostly releases. The decisions have
  many legal moves (median 45), and the more candidates compete, the more
  surely the maximum lands on an over-valued one.

Not the cause: the cube life (0.68 to 1.0 changes ~25 picks, the gap
stays), the Jacoby rule, and cubeful versus cubeless conversion at 1-ply
(the two pick alike).

Root routing is not the cause either. Whether switching it off is worse
depends on the reference: the rollouts say clearly yes at 2-ply, XG says
no. Measured 2026-11-20 with the filtered 2-ply, every pick of both
configurations graded by XG (`score_folder_vs_xg.py complete` for both
sets of picks, `data/snake_xg_completion_rr`):
- Switching root routing off changes 384 of the 795 snake checker picks at
  2-ply and 330 at 3-ply.
- XG takes only its leading moves to full depth. In 179 / 176 of the 384
  at 2-ply (S9-play / twin) and 110 / 111 of the 330 at 3-ply, it had left
  one of the two picks shallower, almost always the no-pin one.

No-pin minus pin, contribution to snake checker PR over all 795 decisions
(S9-play / S11-play twin):

| Decisions | ply | n | vs XG | vs rollout |
|---|---|---|---|---|
| graded by XG's first pass | 2 | 205 / 208 | -5.5 / -5.6 (+- 1.7) | -0.3 / +0.7 (+- 1.3 / 0.9) |
| graded only by the completion pass | 2 | 179 / 176 | +1.0 / +1.9 (+- 1.7) | +15.5 / +13.0 (+- 1.9 / 1.8) |
| all changed picks | 2 | 384 | **-4.5 / -3.7 (+- 2.4)** | **+15.1 / +13.7 (+- 2.4 / 2.1)** |
| graded by XG's first pass | 3 | 220 / 219 | -2.3 / -1.7 (+- 1.7) | -6.1 / -1.5 (+- 1.5 / 0.7) |
| graded only by the completion pass | 3 | 110 / 111 | +3.0 / +3.3 (+- 1.3) | +5.5 / +4.3 (+- 1.4 / 1.0) |
| all changed picks | 3 | 330 | +0.7 / +1.6 (+- 2.1) | -0.6 / +2.8 (+- 2.2 / 1.3) |

What it shows:
- At 3-ply root routing is neutral under both references.
- At 2-ply the references disagree by ~19 PR, on the decisions XG had
  first left shallow. There the rollouts charge the no-pin pick 0.14
  equity per decision on average; XG, once it searches the move, rates it
  level with the pinned one. An earlier reading (2026-11-19) took the gap
  for a selection effect of XG grading only where it had both values. The
  completion pass refutes that: it is a real disagreement about these
  positions.
- XG's edge for no-pin on its first-pass decisions comes mostly from a
  handful of 0.5-1.1 swings: trimming the ten largest on each side takes
  the all-changed figure from -4.5 to -3.8 and -3.7 to -3.1.

Root routing stays on: it rests on the rollouts, which are the better
reference in snakes (see "Why XG and the rollouts disagree in snakes"
below), and XG's case against it is under two standard errors. It is not
independently confirmed. To compare two configurations against XG, run a
completion pass for both sets of picks first. Grading only where XG has
both values leaves out exactly the decisions where the two configurations
differ most.

At 2-ply, cubeful checker play now searches only the filter's survivors.
The pruned moves keep 1-ply Janowski values and enter the search only
through the stale-top-two promotion (`CUBEFUL_FILTERED_PLIES` in
`analyzer.py`, `tests/test_cubeful_nply_filter.py`). `batch_checker_play`
had the same flaw (it ran the cube tree for every move) and takes the same
steps since 2026-11-19. Paired against the old 2-ply, checker PR on the
snake folder:

| Reference | vs XG | vs rollout |
|---|---|---|
| S9-play | -2.73 +- 0.66 | -2.85 +- 0.70 |
| S11-play twin | -2.51 +- 0.66 | -2.15 +- 0.65 |

That brings 2-ply level with 1-ply in snakes, not ahead of it. Elsewhere at
2-ply the change is neutral, and it runs 2.2-2.4x faster:
- money benchmark: checker 1.40 -> 1.38, paired -0.016 +- 0.022, blunders
  40 -> 37, 45 s -> 19 s;
- 5-point match benchmark: checker 1.22 -> 1.22, paired +0.001 +- 0.012,
  43 s -> 20 s;
- ten backgame folders pooled: -0.04 +- 0.02;
- containment, massive and `Positions XG gets wrong`: within noise.

**What is left of the snake 2-ply gap (2026-11-20).** With every pick graded,
2-ply minus 1-ply over all 795 snake checker decisions is +1.1 +- 1.9 /
+1.5 +- 1.8 against XG and -1.6 +- 1.2 / +1.7 +- 1.3 against the rollouts
(S9-play / twin): level. Split by what the two picks do to the snake, one
part loses under all four references, the other gains back:

| 2-ply pick vs 1-ply pick | n | vs XG | vs rollout |
|---|---|---|---|
| 2-ply releases, 1-ply keeps the snake | 64 | +2.4 / +1.8 (+- 1.2 / 1.1) | +1.3 / +2.6 (+- 0.6 / 0.8) |
| 2-ply keeps, 1-ply releases | 38 | -0.8 / -0.7 (+- 0.3) | -1.5 / -0.4 (+- 0.6 / 0.5) |

(Contributions to checker PR over the 795 decisions. "Releases" = the
post-move board is no longer a snake, `scripts/snake_rule.py`.) The
release over-valuation described under Round 2 is still there, smaller,
and offset by the opposite case. Its cause under root routing (where the
pinned snake net values the exit replies it was trained on) is not
measured. 3-ply minus 2-ply
is -4.2 +- 1.7 / -4.0 +- 1.8 against XG and +4.1 +- 2.0 / -1.6 +- 1.3
against the rollouts.

3-ply keeps the full search. With the filter, the money benchmark's 3-ply
checker PR rose 0.59 -> 0.64 (paired +0.044 +- 0.011; 124 decisions worse,
62 better), although it helped the snake folder against the rollouts
(-5.3 +- 1.7 and -2.2 +- 0.7) and was neutral against XG.

Read only the ranking of the levels. In absolute terms XG and the rollouts
disagree widely here:
- XG's #1 costs 25.9 / 38.8 against the two rollouts (checker play 26.9 /
  46.3).
- NeonGammon costs 31-39 against XG.
- XG's own figures are a mix of 4-ply and Roller+.

*Why XG and the rollouts disagree in snakes (2026-11-19).* XG misjudges
the positions themselves. The cube and the move ranking are not the cause:
- **Values, same move, cubeless.** XG's offset from either NeonGammon rollout
  has sd 0.39 equity per decision (P(win) sd 0.12). The two NeonGammon rollouts
  (S9 play, S11 play) agree to 0.07 (P(win) 0.022). In ordinary contact
  positions XG and NeonGammon agree to 0.009 in P(win) (Benchmark PR sample, XG
  4-ply #1 against NeonGammon 2-ply).
- **Not the cube.** Graded on shared moves, XG's #1 costs PR 24.2 against
  the rollout cubeful, and its cubeless pick costs 17.7 graded cubeless.
  Within each engine, switching between cubeful and cubeless ranking
  changes little (PR 1.3 for the rollout, 5.4 for XG).
- **One-sided.** XG rates the trapped side higher. When the mover holds
  the snake, XG's cubeless equity is 0.27 (4-ply) / 0.34 (Roller+) below
  the rollouts, with 0.11 more gammon losses. When the mover is trapped,
  it is 0.17 / 0.40 above. The worst cases have one straggler in front of
  a full far-side prime, the trapped side with 14 checkers home and none
  off, and the holder ~240 pips behind. There XG gives the trapped side
  +1.25 and the rollouts -0.34.
- **XG's net is outside its range.** 14% of XG's full-depth 4-ply
  evaluations are impossible probability vectors (gammon loss above total
  loss, or gammon win above win). They occur in 130 of 696 decisions,
  against none in 600 contact decisions.
- **XG's levels disagree with each other.** XG analysed 123 positions at
  4-ply in one copy and Roller+ in the other. Its values for the same
  moves differ by sd 0.34 between the two. On the 79 of them the rollouts
  cover, Roller+ is further from the rollouts than 4-ply (offset sd 0.42
  vs 0.22, pick PR 4.7 vs 2.5), so `XG Roller++` would not settle it.
- **The rollouts are robust to trial play.** Re-rolling 40 XG #1 boards
  with 1-ply play (Stage 9 and stage11 alike) moves their values by
  0.13-0.15. On the 25 largest disagreements every one moves toward XG,
  but by only 0.15 of a 1.24 gap. GNUbg 2-ply, an independent net, lands
  nearer the rollouts than XG on 23 of those 25.
- **Where XG's error comes from.** Where XG's level is within 0.2 of the
  rollouts (422 / 399 decisions), XG's #1 costs PR 10.1 / 7.0 against
  them. The decisions off by more than 0.5 (200 / 97) cost 51 / 40 and
  carry 57% / 44% of its error.

So in snakes the rollouts are the better reference. The XG verdicts above
count where they agree with the twin, not as an independent arbiter.

Where the widened rule costs something. A control run (the arm disabled,
same completed references) reproduces the widened-rule containment-folder
numbers to the decimal — 6.72 / 5.67 / 3.62 on the S9-play reference and
6.16 / 5.47 / 2.13 on the twin for the installed net — so the rule itself
changes nothing there. (Those differ from the Stage 11p table above because
this folder's reference was completed further today: every model's headline
moves whenever a reference is completed, so the earlier table is stale on
the reference now on disk and needs rescoring before it is quoted again.)
The cost is on the money benchmark's containment slice, split at 1-ply by
whether the OLD rule already claimed the decision: on the 237 old-rule
decisions the installed net scores 3.03 (4 blunders) but `sl_cont2_warm`
4.27 (7) and `sl_cont2_primrace` 3.94 (7) — retraining on family x4 + the
snake exits + the wide general rows diluted ordinary-game containment —
while on the 8 decisions the arm adds, the old net scores 22.77 (3
blunders) and the retrained ones 8.5-9.4 (1). Both candidates therefore
regress a slice of ordinary games by about 1 PR. The widened rule is the
better snake mechanism only once a containment net exists that also keeps
the old slice at 3.03, which is a training question (weight the
crunched-arm rows as their own family, or give them their own slot) rather
than a routing one.

*Shipped (2026-09-05).* Root routing, with `sl_snake2_warm` installed as
`models/sl_s11_bg_snake.weights.best` (the round-1 net it replaced is the
first row of the table). The widened containment rule is NOT in the code:
its crunched arm was removed again, because with the old containment net it
costs the money slice 3.03 -> 3.67 and with either retrained net 3.03 -> ~4.
The definition, footprint and numbers above are the record of that
experiment; its data (`data/s11-bg-containment-general-wide-rollout`,
`data/s11-bg-containment-snakeexit-rollout`) and the `sl_cont2_*` candidates
stay on disk, untracked. The two mechanisms are compatible — root routing
acts only on decisions whose ROOT is a snake, the arm on everything else it
claims (a broken snake's later decisions, which no folder measures) — so the
arm is worth re-adding once a containment net exists that keeps the old
money slice at 3.03.

### Stage 11m — Historical: the massive-backgame NN (index 23)

`stage11m` is `stage11s` plus a 24th NN for the benchmark's **massive
backgame** family: a holder behind in the race with >= `MASSIVE_ALT_ANCHORS`
(3) anchors in the opponent's home board, or >= `MASSIVE_MIN_ANCHORS` (2)
anchors and >= `MASSIVE_BACK_MIN` (7) checkers back (opponent's home board
plus the bar), that is neither a containment game nor a snake.
`massive_category(board)` in `neural_net.cpp` mirrors the family filter in
`scripts/backgame_benchmark.py` (`_massive`), which `tests/test_backgame_category.py`
holds it to on the folder references; the phased strategy accepts 22, 23 or
24 NNs and routes snake (22) -> containment (21) -> massive (23) -> trio ->
P3 -> standard. One known difference: the benchmark's own `_containment`
filter is older than the engine's containment rule and disagrees with it on
21 of 7,400 folder boards, which the engine therefore routes to the
containment net while the folder lists them as massive; the test tolerates
that, the benchmark filter is what needs aligning.

**The region was never short of data** — unlike the snake, massive backgames
are 62% of the deep category's rollout pile, 38% of the middle's and 17% of
the double's — so the question was whether a net trained on "massiveness"
across the categories beats the category trio, which serves those decisions
today (a massive-folder root goes to the deep net 51% of the time, the
middle net 22%, P3 20%, the double net 7%). `data/s11-bg-massive-rollout`
(240,173 region rows from the three train piles) and
`data/s11-bg-massive-nbhd-rollout` (their 385,501 non-massive rows plus the
three exit piles) — both built by `scripts/build_massive_data.py` and, like
the category piles they come from, not tracked — feed the containment recipe (`run_s11_containment_sl.py`,
family x4); warm starts from the deep net and from the P3 net converge to
the same net (1-ply ER on the benchmark piles' massive rows 26.59 / 26.55
against the routed trio's 27.38 — deep-category rows 25.6 vs 26.1, middle
27.6 vs 28.7, double 31.6 vs 35.0 — and 30.0 vs 25.9 on the non-massive
neighbourhood), so the data decides and the warm start does not. The
deep-warm net is installed as `models/sl_s11_bg_massive.weights.best`.

**Root routing is OFF for this net** (`root_pinned_` stays `{22}` on the 24-NN
layout): it values the non-massive positions below a massive root worse than
per-node routing does, and pinning it bought nothing on the folder (2-ply
pinned 6.25 / 4.94 vs per-node 6.14 / 4.84; 3-ply 4.34 / 2.94 vs 4.48 / 3.07,
S9-play reference / S11-play twin). The lesson is the snake's in reverse: a
specialist whose neighbourhood is already well served by the nets it would
displace should route per node.

**Results (2026-09-05).** The massive folder's references were completed
twice: first for Stage 9's and the previous layout's picks (run
f364980e6a9b, 508 of 512 positions pooled), then for the massive net's own
(run 2b0459a83c9e, 106 of 110; both tails abandoned once they were retrying
three-hour task timeouts on a handful of positions). That moved the previous
layout from the 7.28 the page still quotes to 6.30 at 1-ply and Stage 9 from
9.43 to 9.78. PR at 1P / 2P / 3P on the final references, S9-play reference
| S11-play twin, every pick rollout-graded, blunders in the second row:

| Model | massive folder |
|---|---|
| Stage 9 | 9.78 / 9.38 / 6.18 \| 9.94 / 8.11 / 4.89 |
| previous layout (23 NNs, `stage11s`) | 6.30 / 5.89 / 4.06 \| 6.66 / 4.78 / 2.83 |
| ... blunders | 85 / 76 / 33 \| 86 / 63 / 29 |
| with the massive net (24 NNs), per-node | 5.39 / 5.77 / 4.14 \| 5.96 / 4.71 / 2.95 |
| ... blunders | 63 / 75 / 41 \| 83 / 64 / 29 |

(The cube part of every 2-ply and 3-ply figure here was inflated by the
opponent-owned-cube artifact below until 2026-09-06; the rescored table is
in that section.) So: clearly better at 1-ply, slightly better at 2-ply,
level at 3-ply, and on the ordinary-game slice a gain rather than a regression — the money
benchmark's 52 massive-rule decisions score 3.73 with no blunders against
the previous layout's 5.89 and Stage 9's 5.47 — while the ten backgame
folders, where the rule claims up to 18% of a folder, pool to 2.98 at 1-ply
with it, the same as without. A small, clean win at the leaves that the
search does not amplify.

### Cube scoring: the opponent-owned-cube artifact (found and fixed 2026-09-06)

On the massive and containment folders the CUBE part of the PR was two to
three times higher at 2-ply than at 1-ply or 3-ply, for Stage 9 and Stage
11 alike. A nine-agent investigation (four independent angles, three
adversarial verifications) found no 2-ply defect; it was a scoring artifact
with three links:

* `backgame_benchmark.py` started recording every SEED as a cube decision on
  2026-09-04 without the ownership filter it applies to in-play decisions,
  so 20 massive and 14 containment records (none in the S11-play twins,
  rolled out two days earlier, and none on the snake folder) had the
  OPPONENT owning the cube — no decision for the player at all.
* the parent repo's rollout aggregator re-derived `should_double` from the
  pooled ND/DT with no ownership test, and for an opponent-owned cube the
  "DT branch" is the same game at doubled stakes, so DT = 2 x ND and the
  record graded as a Double whenever ND > 0;
* `score_backgame_pr.py` then charged any engine answering No Double the
  full ND. The 1-ply money cube path (`cube_decision_1ply_money`) had no
  `can_double` gate, "doubled", and was charged nothing; every N-ply path
  gates, answered No Double, and was charged exactly ND per record — 6.05
  equity on massive (8.45 cube-PR points, ~1.4 points of folder PR) and
  5.40 on containment (3.15 and ~0.8), identically at 2-ply and 3-ply. So
  the 1-ply cube numbers were flattered and the 3-ply ones as distorted as
  the 2-ply ones.

Fixed in four places: the scorer skips opponent-owned cube records (and
reports the count), the aggregator writes No Double flags for them (the
existing references' flags were rewritten in place — 31 records, ND/DT/DP
untouched), the generator applies the in-play ownership filter to seeds,
and `cube_decision_1ply_money` gates on `can_double` like the match path
(every rollout call to it already sat behind a `can_double` guard, so no
rollout changes; `cube_points.py` only consults a side's `should_double`
when that side can double).

**What remains after the fix is real, general and not a tree defect**: 2-ply
is no better than 1-ply on cube decisions and 3-ply is about three times
better — on the money benchmark cube PR runs 3.24 / 2.91 / 0.61 / 0.50 at
1 to 4 ply. The 1-ply Janowski leaf, with cube life fixed at
`cube_efficiency`'s 0.68 for contact, under-values the centred-cube No
Double for the side holding the initiative, so 1-ply and 2-ply both
over-double; a 2-ply cube tree adds only the opponent's cube node, and the
player's own next cube node first appears at 3-ply, which is the repair. A
Python replica reproduces the 2-ply tree bit-exactly, Jacoby is not the
cause, and a uniform cube life of 0.85 fixes 2-ply (massive 6.50 -> 1.68,
money 2.91 -> 1.87) while regressing 1-ply and 3-ply on the money
benchmark — so the real fix is a family- and ply-dependent cube life, the
ML implied-cube-life index `cube_efficiency` already reserves a slot for.
The ply-dependent half of that fix landed on 2026-11-18: money analysis at
1-ply and 2-ply now uses a higher contact cube life (see "Cube life by
depth (money)" under "Doubling Cube"). Side findings, not fixed: the
reference pipeline double-applies the beaver on opponent-owned records
(harmless to PR); the S9-play snake reference's cube values are
trial-player distorted (its 3-ply cube PR is worst, the twin's best); and
checker play routes a candidate's net from the pre-move board while the
cube tree routes from the candidate board.

**Rescored with the artifact removed (2026-09-06)** — the completed
references as they stand, opponent-owned cube records skipped (20 on massive, 14 on containment). PR at 1P / 2P / 3P,
with the cube part alone in the second column:

| Model, folder | PR | cube PR | blunders |
|---|---|---|---|
| Stage 9, massive backgame | 9.93 / 8.08 / 4.86 | 8.84 / 7.15 / 2.54 | 158 / 133 / 57 |
| previous layout (23 NNs), massive backgame | 6.32 / 4.51 / 2.66 | 6.74 / 6.67 / 1.74 | 84 / 60 / 17 |
| Stage 11 (24 NNs), massive backgame | 5.44 / 4.38 / 2.74 | 6.74 / 6.50 / 2.04 | 63 / 59 / 25 |
| Stage 9, containment | 10.10 / 10.53 / 6.16 | 5.93 / 6.50 / 2.85 | 257 / 266 / 144 |
| Stage 11 (24 NNs), containment | 6.77 / 4.88 / 2.82 | 5.95 / 6.39 / 2.02 | 150 / 83 / 40 |

Cube PR is now monotone in ply on both folders, as it always was on the
S11-play twins.
