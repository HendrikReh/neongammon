# Benchmarking and Evaluation Scripts

Historical model results live in [docs/model-benchmarks.md](model-benchmarks.md). Head-to-head and `.xg` Batch Analyze workflows live in [docs/xg-comparison.md](xg-comparison.md#operational-workflows-xg-files-and-h2h). Primary literature and external engine/rollout references live in [docs/strength-references.md](strength-references.md).

## Benchmark Scripts

All benchmark scripts default to the production model and accept `--model <name>`
to override. Scripts live in `scripts/`.

| Script | What it measures | Key args |
|--------|-----------------|----------|
| `run_full_benchmark.py` | Full suite: per-plan ER + contact/crashed/race ER + vs PubEval ppg + self-play distribution. Supports 1-ply through N-ply. | `--model`, `--ply N`, `--scenarios N`, `--threads N`, `--games N` |
| `run_rollout_benchmark.py` | Top-N worst 1-ply errors compared at 2-ply, 3-ply, 4-ply, rollout | `--model`, `--top N`, `--threads N` |
| `score_benchmark_pr.py` | Benchmark PR (equity error vs rollout reference, 103k decisions). Any registered model type (5-NN, pair, backgame) via `bgbot_cpp.score_benchmark_pr_strategy`. | `--model`, `--plies N`, `--all-models`, `--all-plies`, `--decision-step N`, `--no-bearoff` |
| `score_benchmark_pr_gnubg.py` | GNUbg's Benchmark PR (parallel GNUbg CLI subprocesses) | `--plies N`, `--workers N`, `--decision-step N` |
| `test_evaluate_probs.py` | Single position eval at 1-4 ply + GNUbg + rollouts | `--model`, `--checkers`, `--ply N` |
| `test_cube_decision.py` | Cube decisions vs 3 reference positions at 1-4 ply + rollout | `--model` |
| `test_unified_rollout.py` | Verify cubeful(max_cube=1) == cubeless at N-ply + rollout | `--model` |
| `eval_position.py` | Side-by-side NeonGammon (production model) vs GNUbg evaluation (cube action or checker play, 1-4 ply + rollout, money or match play) | `cube`/`checker` subcommand, `--checkers`, `--dice`, `--match`, `--score`, `--cube-value`, `--cube-owner` |
| `profile_cube_benchmark.py` | Profiling benchmark: 8 fixed cube positions (mix of money/match, bearoff/contact) evaluated serially with timing. For verifying optimisations don't change values. | level positional arg: `1ply`–`4ply`, `1T`, `2T`, `3T` |

```bash
# Full benchmark with production model (1-ply)
python scripts/run_full_benchmark.py

# Compare two models
python scripts/run_full_benchmark.py --model stage5
python scripts/run_full_benchmark.py --model stage3

# Multi-ply benchmark
python scripts/run_full_benchmark.py --ply 2
python scripts/run_full_benchmark.py --ply 3 --scenarios 500

# Score all registered models on Benchmark PR
python scripts/score_benchmark_pr.py --all-models

# Side-by-side NeonGammon vs GNUbg cube analysis (money game)
python scripts/eval_position.py cube --checkers "0,-2,0,0,0,0,5,0,3,0,0,0,-5,5,0,0,0,-3,0,-5,0,0,0,0,2,0"

# Side-by-side cube analysis (match play: 5-point match, player 3pts, opp 0pts)
python scripts/eval_position.py cube --checkers "..." --match 5 --score 3 0

# Side-by-side checker play analysis
python scripts/eval_position.py checker --checkers "..." --dice 3 1

# Side-by-side checker play analysis (match play)
python scripts/eval_position.py checker --checkers "..." --dice 3 1 --match 5 --score 3 0
```

### Native batch scheduling

Run `PYTHONPATH=build:python python scripts/bench_search_batches.py --threads 4`
to compare scalar 3-ply checker play with batches of four and 64 on the same
seeded five-point game. The cube cases combine one contact row with zero to
three bearoff rows. The script checks complete results against scalar and
reports median wall time with models loaded. Each sample uses a fresh
evaluator identity so earlier searches cannot warm the pool's per-thread
position caches.

On an M4 Pro with four threads (five samples, Accelerate, seed 42), the
44 checker positions took 1.794 s scalar, 1.722 s in chunks of four, and
1.722 s in a single chunk of up to 64. One contact cube took 7.12 ms scalar
and 7.17 ms batched. With three bearoff rows added the same comparison was
8.14 ms and 8.06 ms. All returned fields matched.

### Benchmark PR rollout sample (this fork)

The full reference `data/benchmark_pr/rollouts.jsonl` is gitignored and was
never shipped, so this fork rolled out a 1-in-20 sample of the 138,872 saved
decisions instead (`generate_benchmark_pr.py --resume --decision-step 20`:
6,944 decisions, 20,302 positions, Stage 5 full rollouts at 1,296 trials,
~4.5 h on the M4 Pro). Score it with the same step, so that only the sampled
decisions count:

```bash
python scripts/score_benchmark_pr_gnubg.py --decision-step 20 --plies 1 --plies 2
python scripts/score_benchmark_pr.py --decision-step 20 --all-models --all-plies
```

Without `--decision-step` the scorers also take every other decision whose
candidates all happen to be in the file, and those lean towards positions
shared between many games: Stage 5 2-ply scores 3.37 on the sample but 5.27
on those 4,233 extras.

Measured 2026-11-18 (5,140 scored decisions; the 1,804 with a single legal
move are skipped). `score_benchmark_pr.py` plays bearoff positions from the
bearoff database, as the engine and GNUbg do; `--no-bearoff` scores the raw
networks, which puts every model's pure race at 0.66-0.82:

| Strategy | Overall | PureRace | Racing | Attacking | Priming | Anchoring |
|---|---|---|---|---|---|---|
| GNUbg 1-ply | 4.31 | 0.39 | 4.85 | 5.35 | 5.76 | 5.10 |
| GNUbg 2-ply | 3.48 | 0.37 | 3.89 | 3.85 | 4.59 | 4.62 |
| stage5 1-ply | 4.17 | 0.54 | 4.57 | 4.93 | 5.19 | 5.39 |
| stage5 2-ply | 3.34 | 0.44 | 3.06 | 3.82 | 4.72 | 4.58 |
| stage9 2-ply | 3.24 | 0.57 | 3.16 | 3.80 | 4.50 | 4.12 |
| stage11 1-ply | 4.11 | 0.55 | 4.30 | 5.02 | 5.51 | 5.03 |
| stage11 2-ply | 3.17 | 0.57 | 3.03 | 3.78 | 4.42 | 4.00 |
| stage11 + S5 200h race net, 1-ply | 4.10 | 0.54 | 4.30 | 5.02 | 5.51 | 5.03 |
| stage11 + S5 200h race net, 2-ply | 3.14 | 0.44 | 3.00 | 3.79 | 4.42 | 3.99 |

(`stage11` has used Stage 5's 200h pure-race net since 2026-11-18, the last
two rows; the two above them are the S6 100h net it replaced.)

Read it with three caveats. The reference favours Stage 5: it is Stage 5's
rollout, and only the moves Stage 5's 2-ply filter kept were rolled out, so a
pick outside that set is charged the worst rolled-out move (GNUbg 1-ply lands
outside it 159 times, stage11 about 80). No significance test was run, so gaps
of a few hundredths (Stages 8-11 span 3.17-3.24 at 2-ply) may be noise. And
the numbers are not comparable with the full-set PRs quoted in [docs/model-benchmarks.md](model-benchmarks.md).

**The pure-race gap.** The one clear gap to GNUbg on this reference is the
pure race (979 decisions); against XG it disappears (end of this paragraph). Split by whether the pre-move position is in the bearoff database
(both sides home), stage11 PR at 1-ply / 2-ply:

| Race decisions | n | raw net | with bearoff DB | GNUbg |
|---|---|---|---|---|
| bearoff (both sides home) | 371 | 0.75 / 0.76 | 0.09 / 0.09 | 0.11 / 0.11 |
| outside the database | 608 | 0.84 / 0.86 | 0.83 / 0.86 | 0.57 / 0.52 |

So the database settles the home-board bearoffs, and what remains is the
pure-race net in races with checkers still outside home. Every model from
Stage 6 to Stage 11 uses the same file there (`sl_s9_purerace` is
byte-identical to S6's 100h net), and with it 2-ply is no better than 1-ply
(0.98 vs 0.89 on the 520 races with no candidate in the database). Swapping
Stage 5's 200h pure-race net into stage11 gives 0.81 / 0.64 outside the
database and 0.54 / 0.44 over all races, but this reference is Stage 5's own
rollout, played with that very net, so part of the gain may be bias.
GNUbg's race benchmark (`race.bm`), the independent check, is not on this
machine; XG is (next section). It agrees with the swap, which `stage11` now
carries, in direction but not significantly: the 2-ply race gain is one
decision (paired comparison in the next section).

**Against XG there is no race gap (2026-11-20).** Scored against XG 4-ply
(`export_benchmark_pr_xg.py sage --gnubg`), `stage11` with the S5 200h race
net and GNUbg are level outside the database. PR, and the paired difference
on the same 608 decisions:

| Reference | stage11 1P / 2P | GNUbg 1P / 2P | stage11 - GNUbg, 1P | 2P |
|---|---|---|---|---|
| XG 4-ply | 0.51 / 0.51 | 0.57 / 0.50 | -0.06 +- 0.19 | +0.00 +- 0.18 |
| Stage 5 rollout | 0.81 / 0.64 | 0.57 / 0.52 | +0.25 +- 0.09 | +0.12 +- 0.08 |

So the gap in the first table comes from the reference, not from the race
net: the Stage 5 rollout's own best moves cost 0.34 against XG there, and
XG's #1 costs 0.37 against the rollout. Against XG, 2-ply is no better than
1-ply (+0.00 +- 0.09). stage11 makes more small errors than GNUbg (vs XG at
1-ply it is worse on 96 decisions and better on 60), and GNUbg makes fewer
but larger ones. A larger one-sided database would help little: only 21% of
these races have every checker within the 10-point.

Retraining the pure-race net on our own targets did NOT help (2026-11-18).
`scripts/harvest_race_positions.py` + `modal_rollout.py` produced 330k race
positions rolled out under Stage 9 play (~$22), and `scripts/run_purerace_sl.py`
fine-tuned both nets on them: holdout ER 16.9 -> 11.1 (100h) and 15.4 -> 11.1
(200h). On the Benchmark PR races the gain does not carry over: absolute
error against XG 4-ply moves 9.95 -> 9.33, the within-decision error that
decides the pick barely moves (5.52 -> 5.30; the 200h fine-tune 4.90 -> 5.29),
and the picks are no better (outside the database, against XG at 1-ply /
2-ply: 100h 0.66 / 0.69, 200h 0.67 / 0.58, Stage 5's untouched net 0.51 /
0.51). Holdout ER on our own positions is the wrong selection metric for a
race net; the candidates are in `models/race_diag/`.

### XG as referee for the Benchmark PR sample

`scripts/export_benchmark_pr_xg.py` writes the sample's decisions as
one-decision `.xg` games (`generate --subset race|contact|all`; the record
templates are cloned from any XG-analysed game, `data/benchmark_pr_xg/template.xg`),
XG Batch-Analyzes them, `score` reads XG's ranked moves back into
`xg_reference.jsonl`, and `sage` scores NeonGammon (1-ply / 2-ply, `--race-net` for
other race nets) and optionally GNUbg (`--gnubg`) against BOTH the Stage 5
rollout and XG. XG's reference value for a move is the cubeless equity of its
own evaluation, at whatever depth XG's filter left it (4-ply for the leaders),
the convention of XG's own error figures. 23 sample decisions carry an illegal
saved candidate (a move that leaves a playable die unused; 17 as the
reference's best) and are left out.

Measured 2026-11-18, XG "World Class" (4-ply), 5,117 decisions (979 race +
4,138 contact), before the race-net swap:

| Player | vs Stage 5 rollout | vs XG 4-ply |
|---|---|---|
| XG's #1 move | 2.80 | 0.64 |
| Stage 5 rollout's best move | 0 | 1.92 |
| stage11 1-ply / 2-ply | 4.05 / 3.11 | 2.07 / 1.24 |
| GNUbg 1-ply / 2-ply | 4.26 / 3.42 | 2.35 / 1.40 |

Paired on the same decisions, stage11 beats GNUbg under both references
(2-ply: -0.17 +- 0.07 against XG, -0.30 +- 0.10 against the rollout), and XG
rates stage11 2-ply above the Stage 5 rollout's own best moves (-0.68 +-
0.09): beyond today's level the old reference mostly measures distance to
Stage 5, so check improvements against XG. XG's own #1 does not score 0
because XG ranks moves cubeful (centred cube, Jacoby off) while the reference
is cubeless; read the XG column from ~0.6 up. In the bearoff races XG agrees
with the exact database (NeonGammon with the DB scores 0.00 there), while the Stage 5
rollout's best moves cost 0.13: that is the rollout reference's noise.

**The race-net swap, paired (2026-11-18).** `sage --race-net "s6
100h=sl_s9_purerace.weights.best:100"` on both subsets scores the old race
net next to the one `stage11` carries, on the same 5,117 decisions (the
S6 columns reproduce the pre-swap run bit-for-bit). PR against XG 4-ply,
diff = S5 200h - S6 100h (negative = the swap helps), with the number of
decisions whose pick changed (better with S5 / with S6):

| Subset | Ply | S6 100h | S5 200h | diff +- SE | picks changed |
|---|---|---|---|---|---|
| all | 1 | 2.065 | 2.055 | -0.010 +- 0.008 | 145 (84 / 61) |
| all | 2 | 1.235 | 1.198 | -0.038 +- 0.028 | 157 (86 / 71) |
| contact | 2 | 1.421 | 1.406 | -0.015 +- 0.007 | 37 (19 / 18) |
| race outside DB | 1 | 0.595 | 0.507 | -0.088 +- 0.070 | 145 (84 / 61) |
| race outside DB | 2 | 0.727 | 0.508 | -0.219 +- 0.230 | 120 (67 / 53) |

Bearoff races, and contact at 1-ply, are identical under both nets. The swap
is never worse, but no gap reaches two standard errors. The 2-ply race gain
is a single decision (key 112280, 4-1, where the S6 net's 2-ply pick costs
0.26 against XG and 0.23 against the rollout); the other 607 races net to
-0.003. The 1-ply race gain is spread over many decisions (-0.083 with the
six most extreme removed). At 2-ply the race net reaches contact decisions
through race leaves (37 picks change), for a gain within noise. The Stage 5
rollout reference agrees in sign everywhere but is weaker (overall 1-ply
-0.002, 2-ply -0.035 +- 0.025). Read the swap as safe and probably slightly
better, not as a measured strength gain.

`score_benchmark_pr.py` builds its N-ply strategy with `create_multipy`, which
adds the PubEval opponent prefilter BgBotAnalyzer uses; the older 5-NN-only
`score_benchmark_pr_nply` binding has none (Stage 5 2-ply 4.104 vs 4.097 on
an 8k-decision set; 1-ply identical).


## 3-ply cube analysis on the published 3.2.0 wheel

Measured on 2026-11-25 on an Apple M4 Pro, macOS 27.0.1 arm64, CPython 3.12.13 and NumPy 2.5.4. The benchmark used the official CPython 3.12 wheel (SHA256 `9362a0084be2fc5e609dcb9455962b0807d19f6de7dd8f3c89f483aec3a8e7d8`), the default Stage 11 models and NeonDojo's 3-ply analysis setting with four engine threads. [Release](https://github.com/HendrikReh/neongammon/releases/tag/v3.2.0) · [Benchmark scope in #63](https://github.com/HendrikReh/neongammon/issues/63)

Median wall time per public Python call, in milliseconds. “First” is the first evaluation in a newly spawned process after analyzer construction. “Warm” repeats the same board and cube context in that process. Calls ran sequentially. Each method had 16 fresh processes per position, with the two methods alternating first in each pair. After each first call the benchmark discarded three warmups and timed five repeats, giving 80 warm samples per cell. Timing excludes process startup, model loading, HTTP, sidecar transport and persistence.

| Position | First `cube_action` | First `cube_decision` | Warm `cube_action` | Warm `cube_decision` | Decision |
|---|---:|---:|---:|---:|---|
| 5-away/4-away, centered cube | 3.63 | 2.70 | 1.65 | 1.05 | No double |
| Money, player owns 2-cube | 9.05 | 4.66 | 5.97 | 2.83 | Double/take |
| SNAKE, centered money cube | 3.51 | 2.05 | 1.56 | 0.60 | No double |

All shared result fields, including ND/DT/DP equities, matched exactly for every pair. A second complete run returned median times within 0.06 ms of the first. The owned money cube had the largest absolute warm gap (3.14 ms). Profile the additional cubeless probability search on that fixture first. Any change to candidate reuse also needs comparison against the independent cubeless traversal, since the cube and cubeless policies can choose different continuations. [Parity cases](../tests/test_cube_candidate_batch_reuse.py)

To reproduce from the repository root, set `ENGINE_PYTHON` to the interpreter of the verified 3.2.0 installation and run the benchmark without concurrent builds or test suites.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  "$ENGINE_PYTHON" -I scripts/benchmark_cube_action_decision.py
```

### Where the owned money cube spends its time

On the same M4 Pro and four-thread fixture, a diagnostic build from `v3.2.0` timed the two native traversals inside `cube_action`. The first interval includes candidate-cache construction and the cubeful tree. The second includes the cubeless probability tree and conversion back to the player on roll. Both use elapsed wall time on the calling thread, including time waiting for the search workers. The installed 3.2.0 wheel supplied the Python code and models for both measurements.

| Median wall time | Cubeful tree | Cubeless tree | Cubeless share of both trees | Full call on official wheel |
|---|---:|---:|---:|---:|
| First call, 16 processes | 4.67 ms | 4.30 ms | 47.7% | 9.06 ms |
| Warm call, 80 samples | 2.47 ms | 3.07 ms | 55.9% | 5.87 ms |

A second run measured **3.11 ms** for the warm cubeless traversal (55.6%). This accounts for most of the earlier 3.14 ms warm gap between the public methods on this board. The full call in the last column uses the uninstrumented wheel in a separate process. Its timing includes the Python boundary and cache cleanup. The share is the median of the per-call ratios.

Both builds produced identical rich results on every repeat. The decision-only method agreed on all nine shared fields, and the independent traversal used for per-roll details agreed exactly on all five probabilities and cubeless equity. For this fixture, measure how many candidate batches the cubeless traversal can reuse before changing the Accelerate batch policy.

To reproduce, use a fresh build directory, the CPython 3.12 interpreter containing the official wheel (`ENGINE_PYTHON`), and CMake, Ninja and pybind11's CMake package on the build path. The patch adds timing probes to a disposable copy of the release source. The driver runs a fresh interpreter per build per pair, alternating the order. [Timer locations](../scripts/experiments/cube_3ply/phase-timers.patch) · [Parity checks and sample aggregation](../scripts/experiments/cube_3ply/measure_phases.py)

```sh
profile_dir="$PWD/build/cube-phase-profile"
mkdir -p "$profile_dir/src"
git archive v3.2.0 cpp | tar -x -C "$profile_dir/src"
patch -d "$profile_dir/src" -p1 < scripts/experiments/cube_3ply/phase-timers.patch
cmake -S "$profile_dir/src/cpp" -B "$profile_dir/native" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
  -DBGBOT_BUILD_APPS=OFF -DPYBIND11_FINDPYTHON=ON \
  -DPython_EXECUTABLE="$ENGINE_PYTHON" -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build "$profile_dir/native" --target bgbot_cpp
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  "$ENGINE_PYTHON" -I scripts/experiments/cube_3ply/measure_phases.py "$profile_dir/native"
```

### Candidate reuse on the published 3.2.1 wheel

On 2026-11-26, the owned-money fixture returned **Double/Take** on the official CPython 3.12 arm64 wheel (SHA256 `42686dd524b052640e56074a99d063d332ff0c6da2372e4675bab3d72cb14b7e`). The M4 Pro ran macOS 27.0.1, Stage 11, four engine threads and single-thread BLAS. Each cell below is a median in milliseconds. Each of the two runs used 16 fresh processes per variant and 80 warm repeats per variant, with the four variants rotated through the process order. The host had background VM and intermittent desktop/compiler activity. [Release](https://github.com/HendrikReh/neongammon/releases/tag/v3.2.1)

| Public call | First, run 1 | First, run 2 | Warm, run 1 | Warm, run 2 |
|---|---:|---:|---:|---:|
| Official wheel, `cube_action` | 9.353 | 9.351 | 5.929 | 5.854 |
| Official wheel, `cube_decision` | 5.043 | 5.025 | 2.883 | 2.819 |
| Local release source, normal sharing | 9.520 | 9.335 | 5.909 | 5.832 |
| Same extension, sharing disabled | 9.859 | 9.775 | 6.178 | 6.164 |

Whole-batch reuse saved **0.27–0.33 ms per warm rich call** (4.4–5.4% of the time with sharing disabled). All rich results matched exactly across timing variants and repeats. The decision-only fields and independent cubeless probabilities/equity also matched. Separate phase timers measured 4.737 ms cubeful and 4.348 ms cubeless on the first call, then 2.508 ms and 3.088 ms warm. The cubeless traversal accounted for 55.0% of warm native elapsed time.

The trace captured every roll with two or more evaluated candidates in both trees, including the ordered list *after* filtering. The matcher compared the pre-move board, dice, evaluator identity, root pin and ordered candidate boards within the same public call. Each reported hit also passed the production path-slot check. Exact matches in other path slots count as additional opportunities. Rolls served by the tree cache, dances and forced moves never enter the candidate-batch denominator. The trace counts candidate probability rows before the engine groups them by NN for BLAS. Variation in worker-local tree-cache hits changes how many batches each repeat requests. The counts below pool 16 first calls and 80 warm calls in a separate tracing run.

| Cubeless demand and overlap with the cubeful tree | First calls | Warm calls |
|---|---:|---:|
| Requested batches / rows | 7,392 / 236,864 | 28,119 / 897,741 |
| Whole batches actually reused | 1,194 (16.15%) | 4,061 (14.44%) |
| Rows in those hits | 16,808 (7.10%) | 57,051 (6.35%) |
| Additional identical whole batches / rows | 6 / 88 | 156 / 2,288 |
| Batches with only partial overlap / common rows | 6,192 / 92,880 | 20,500 / 307,500 |
| Batches with zero overlap | 0 | 3,402 |

Keep the current Accelerate policy on this workload. A broader lookup for identical roll batches could reach another **0.25% of warm candidate rows**. The larger partial overlap (34.25% of warm rows) needs different batch shapes for shortlist survivors and the remainder. Forcing the existing survivor-reuse path on Accelerate changed the owned-money gammon-win and gammon-loss probabilities. It also changed 9 of the 43 seed-42 cube results, with a maximum absolute field difference of `5.96e-8`. This fails the engine's exact probability parity requirement.

On the verified seed-42 match fixture, normal and disabled sharing both matched the pristine **same-toolchain** build exactly on every field for 44 checker and 43 cube rows at 3-ply. The pristine build and official wheel agreed on all discrete fields. They matched exactly on 35/44 checker and 42/43 cube rows, with maximum absolute float differences of `7.75e-7` and `6.82e-13`. The local compiler was Apple Clang 21 (SDK 27.2) and the wheel records SDK 15.5. The official NeonDojo sidecar verifier independently matched scalar and batched results exactly on all 87 rows at **both 3- and 4-ply**.

To reproduce, set `ENGINE_PYTHON` to the locked 3.2.1 interpreter and `FIXTURE_SCRIPT` to NeonDojo's `tools/open-sage/verify_game_batches.py`. Put CMake, Ninja and a Python interpreter with pybind11 on `PATH`. All diagnostic builds use a disposable export of tag `v3.2.1` under the supplied directory. The installed wheel supplies Python and model assets. [Build script](../scripts/experiments/cube_3ply/build_probes.sh) · [Native probes](../scripts/experiments/cube_3ply/candidate-probe.patch) · [Counting, timing and parity driver](../scripts/experiments/cube_3ply/measure_reuse.py)

```sh
export ENGINE_PYTHON
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
exp=scripts/experiments/cube_3ply
out=build/ng63
"$exp/build_probes.sh" "$out"
"$ENGINE_PYTHON" -I "$exp/measure_reuse.py" timing "$out/ab"
"$ENGINE_PYTHON" -I "$exp/measure_phases.py" "$out/phases"
"$ENGINE_PYTHON" -I "$exp/measure_reuse.py" trace "$out/trace"
"$ENGINE_PYTHON" -I "$exp/measure_reuse.py" parity "$out/ab" "$out/pristine" "$FIXTURE_SCRIPT" --subset "$out/subset"
"$ENGINE_PYTHON" -I "$FIXTURE_SCRIPT"
```

### Lazy allocation of the per-thread cube evaluation cache

On 2026-11-26, deferring the 32,768-entry table until a thread first stores a result
saved **6.77 MiB of resident memory** on the first 3-ply bearoff post-move evaluation.
That public call clears the cache and obtains probabilities from the bearoff database.
A contact cube search stores results and still needs the table. Its warm timings
varied in both directions across two runs.

| 3-ply public call | Build | First call (ms) | Warm call (ms) | RSS growth on first call (MiB) |
|---|---|---:|---:|---:|
| Bearoff `post_move_analytics` | Baseline | 0.403 / 0.419 | 0.0049 / 0.0048 | 6.859 / 6.859 |
| | Lazy | 0.075 / 0.075 | 0.0048 / 0.0050 | 0.094 / 0.094 |
| Owned-money contact `cube_action` | Baseline | 9.119 / 9.125 | 5.917 / 5.985 | 51.172 / 51.148 |
| | Lazy | 9.094 / 9.078 | 5.836 / 6.033 | 51.211 / 51.336 |

Values are medians from two independent runs on an Apple M4 Pro, macOS 27.0.1,
Apple Clang 21.0.0, SDK 27.2, Accelerate, four engine threads and single-thread
BLAS. Each run used 16 fresh processes per build per position, alternating
baseline and candidate order. Timing starts after analyzer construction and ends
when the public call returns. Warm values pool five calls per process after three
discarded warmups. RSS growth is the difference between two macOS `ps` samples
immediately before and after the first call. It measures the entire process,
including worker threads and other allocations. Background desktop and VM
activity remained present.

The local baseline (`39db41d`) and candidate used the same toolchain and locked
3.2.1 Python package, Stage 11 assets and bearoff database. All public result
fields matched exactly at **both 3 and 4 ply** across 44 checker and 43 cube
decisions from the seed-42 five-point game, including forced played boards.
Separately, the official 3.2.1 sidecar verified its scalar and batched results on
all 87 decisions at both depths. The sidecar comparison uses the shipped
extension. The source-build comparison isolates the allocation change.

To reproduce from this revision, use a fresh output directory, CMake, Ninja and
a Python interpreter with pybind11 on `PATH`. Set `ENGINE_PYTHON` to the locked
CPython 3.12 NeonGammon 3.2.1 interpreter and `FIXTURE_SCRIPT` to NeonDojo's
`tools/open-sage/verify_game_batches.py`. The driver loads one local extension per
process and uses the installed wheel for Python code and model assets.
[Timing and parity driver](../scripts/experiments/lazy_cube_cache/measure.py)

```sh
out=build/lazy-cache-ab
mkdir -p "$out/src"
git archive 39db41d cpp | tar -x -C "$out/src"
cmake -S "$out/src/cpp" -B "$out/native" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
  -DBGBOT_BUILD_APPS=OFF -DPYBIND11_FINDPYTHON=ON \
  -DPython_EXECUTABLE="$ENGINE_PYTHON" -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build "$out/native" --target bgbot_cpp
mkdir "$out/pristine"
cp "$out/native"/bgbot_cpp*.so "$out/pristine/"
cp cpp/src/cube_eval.cpp "$out/src/cpp/src/cube_eval.cpp"
cmake --build "$out/native" --target bgbot_cpp
driver=scripts/experiments/lazy_cube_cache/measure.py
"$ENGINE_PYTHON" -I "$driver" "$out/pristine" "$out/native" timing
"$ENGINE_PYTHON" -I "$driver" "$out/pristine" "$out/native" parity "$FIXTURE_SCRIPT"
"$ENGINE_PYTHON" -I "$FIXTURE_SCRIPT"
```

### M4 Pro automatic search thread count

The 14-thread default at `1b6e682` delivered higher full-game throughput than
a 10-thread performance-core default at both depths in both runs on 2026-11-26.
The experimental change sized both the shared pool and the automatic scalar and
batch search budgets from `hw.perflevel0.physicalcpu`. macOS scheduled the
threads normally. The engine retains its existing default.

| Full game | 14 threads, seconds | 10 threads, seconds | 14 threads, decisions/s | 10 threads, decisions/s |
|---|---:|---:|---:|---:|
| 3-ply, run 1 | 0.841 | 0.932 | 103.50 | 93.30 |
| 3-ply, run 2 | 0.814 | 0.907 | 106.90 | 95.92 |
| 4-ply, run 1 | 15.656 | 17.548 | 5.56 | 4.96 |
| 4-ply, run 2 | 15.503 | 17.452 | 5.61 | 4.99 |

The single-position cases used the game's opening 6–1 checker decision (10
legal moves) and the first pre-roll cube decision, both in contact at 5-away/
5-away. Each cell gives the two runs' median public-call latencies in milliseconds.

| Position | 14 threads, first | 10 threads, first | 14 threads, warm | 10 threads, warm |
|---|---:|---:|---:|---:|
| 3-ply checker | 21.89 / 21.92 | 21.69 / 22.03 | 18.97 / 19.47 | 19.26 / 19.44 |
| 3-ply cube | 6.15 / 6.22 | 5.96 / 5.87 | 3.32 / 3.51 | 3.47 / 3.47 |
| 4-ply checker | 353.84 / 361.72 | 357.62 / 369.08 | 231.96 / 289.00 | 274.74 / 266.18 |
| 4-ply cube | 53.39 / 54.14 | 60.69 / 60.72 | 43.57 / 44.66 | 46.21 / 46.26 |

Each run alternated baseline/candidate order within pairs, with the starting
order reversed for run 2. Each sample used a fresh process with Stage 11 models
already loaded. Each run had four paired processes per scalar case (one first
call, then the last five of six repeats for warm latency) and two pairs per
full game. The verified seed-42 game comprised 44 checker decisions with forced
played boards and 43 rich cube decisions in chunks of four. Both builds matched
exactly in every returned field at both depths on every sample. The timer covers
Python package calls and native search. Sidecar startup, JSON transport,
integrity checks and persistence are outside it. These automatic budgets differ
from NeonDojo's explicitly configured four threads.

Two-row checker and cube controls observed the native functions receive explicit
budgets of 1, 4 and 14 at both depths. Each batch matched scalar results across
budgets and builds. After an automatic two-row call, the candidate process had
11 threads, then grew to 15 on an explicit budget of 14. The baseline process
had 15 throughout. Measurements used an M4 Pro (10 performance and 4 efficiency
cores, 64 GB), macOS 27.0.1, Apple Clang 21.0.0, SDK 27.2, Accelerate, CPython
3.12.13 and single-thread BLAS. Desktop services continued running, including
an App Store process using about one core.

Set `ENGINE_PYTHON` to the verified 3.2.1 interpreter and `FIXTURE_SCRIPT` to
NeonDojo's `tools/open-sage/verify_game_batches.py`. Put CMake, Ninja and Python
with pybind11 on the build path. Both builds use the same compiler, SDK, source
revision and verified model assets. [Build script](../scripts/experiments/pcore_pool/build.sh) · [Experimental patch](../scripts/experiments/pcore_pool/candidate.patch) · [Timing and parity driver](../scripts/experiments/pcore_pool/measure.py)

```sh
exp=scripts/experiments/pcore_pool
out=build/pcore-repro
"$exp/build.sh" "$out"
for order in '' '--reverse'; do
    "$ENGINE_PYTHON" -IB "$exp/measure.py" "$out" "$FIXTURE_SCRIPT" --depths 4 3 --workloads game --pairs 2 $order
    "$ENGINE_PYTHON" -IB "$exp/measure.py" "$out" "$FIXTURE_SCRIPT" --workloads checker cube --pairs 4 $order
done
"$ENGINE_PYTHON" -IB "$exp/measure.py" "$out" "$FIXTURE_SCRIPT" --controls-only
```

### NEON output layer for groups of one or two

On 2026-11-26, a four-thread Stage 11 analysis of the seed-42 five-point game
used groups of one or two boards in approximately 19% of calls to
`NeuralNetwork::forward_group` at each depth. The proposed NEON output kernel
changed public analysis results. Production keeps the Accelerate output matrix
multiply and the sigmoid lookup table.

| Search | Decisions | One board | Two boards | Three or more | Share with one or two |
|---|---:|---:|---:|---:|---:|
| 3-ply checker | 44 | 40,737 | 22,273 | 259,796 | 19.52% |
| 3-ply rich cube | 43 | 2,083 | 1,364 | 18,870 | 15.45% |
| 4-ply checker | 44 | 612,700 | 472,609 | 4,509,701 | 19.40% |
| 4-ply rich cube | 43 | 37,291 | 22,842 | 315,307 | 16.02% |

The table gives one instrumented pass. A second 3-ply pass on a clean rebuild
counted 40,727 single-board and 22,272 two-board checker calls, again 19.52%
combined. Its cube counts matched the table. The probe counts positive-sized
groups across search workers and resets between checker and cube workloads.
The timed builds omit instrumentation. For each affected row the candidate
computes five output dot products with the engine's NEON `simd_dot`, adds the
output biases, then runs `sigmoid_inplace_neon` on the five logits (four
Cephes lanes and one scalar tail). The dot-only control keeps the baseline
lookup sigmoid, including its saturation at ±8. All three extensions share
the Accelerate hidden layer and the baseline output path for larger groups.

Every sample compares the complete returned dataclasses. With the proposed
kernel, 33 of 44 checker rows and 29 of 43 cube rows differed at 3-ply. At
4-ply, 36 checker rows and 31 cube rows differed. Every best checker board and
optimal cube action matched. The ordered list of remaining moves differed in
one 3-ply position and two 4-ply positions. The maximum absolute checker equity
difference for the same move was 0.0000801 at 3-ply and 0.000194 at 4-ply.
Across no-double, double/take and double/pass equities the maxima were
0.0000134 and 0.000411. Even with the original output sigmoid, the NEON dot
products alone changed 24 checker and 7 cube rows at 3-ply, and 25 checker and
10 cube rows at 4-ply.

| Full game | Accelerate, seconds | NEON, seconds | Accelerate, decisions/s | NEON, decisions/s |
|---|---:|---:|---:|---:|
| 3-ply, run 1 | 1.9601 | 1.9605 | 44.386 | 44.377 |
| 3-ply, run 2 | 1.9648 | 1.9631 | 44.279 | 44.318 |
| 4-ply, run 1 | 36.8554 | 36.8133 | 2.361 | 2.363 |
| 4-ply, run 2 | 36.8860 | 36.7207 | 2.359 | 2.369 |

Each cell uses the median of two fresh processes per build. Run 1 used
baseline, candidate, candidate, baseline order. Run 2 reversed that order and
the depth order. NEON throughput changed by −0.02% and +0.09% at 3-ply,
and by +0.11% and +0.45% at 4-ply. These sub-percent differences describe
this single game. Model and analyzer construction took a separate median of
18.2 ms across the 16 timed processes.

The baseline and candidate were built from `34558d2` in the same release build
tree with Apple Clang 21.0.0, SDK 27.2 and Accelerate. The M4 Pro (64 GB) ran
macOS 27.0.1 and the locked CPython 3.12.13 wheel environment for NeonGammon
3.2.1, with Stage 11 weights, the bearoff database and single-thread BLAS.
Each process received the same 44 checker rows with forced played boards and
43 rich cube rows, in chunks of four at four search threads. The timer starts
after model and analyzer construction with fresh search caches, and covers
one full pass through the public Python batch calls. Process launch, model
loading, sidecar transport and database writes are outside the timed region.
Desktop services continued to run.

Set `ENGINE_PYTHON` to the locked NeonGammon 3.2.1 interpreter and
`FIXTURE_SCRIPT` to NeonDojo's `tools/open-sage/verify_game_batches.py`.
Put CMake, Ninja and a Python interpreter with pybind11 on the build path.
[Build script](../scripts/experiments/tiny_output/build.sh) ·
[NEON candidate](../scripts/experiments/tiny_output/candidate.patch) ·
[Dot-only control](../scripts/experiments/tiny_output/dot_only.patch) ·
[Counter patch](../scripts/experiments/tiny_output/trace.patch) ·
[Measurement driver](../scripts/experiments/tiny_output/measure.py)

```sh
exp=scripts/experiments/tiny_output
out=build/tiny-output-ab
"$exp/build.sh" "$out"
"$ENGINE_PYTHON" -IB "$exp/measure.py" "$out" "$FIXTURE_SCRIPT" trace
"$ENGINE_PYTHON" -IB "$exp/measure.py" "$out" "$FIXTURE_SCRIPT" compare --candidate dot_only --pairs 1
"$ENGINE_PYTHON" -IB "$exp/measure.py" "$out" "$FIXTURE_SCRIPT" compare --depths 4 3 --pairs 2
"$ENGINE_PYTHON" -IB "$exp/measure.py" "$out" "$FIXTURE_SCRIPT" compare --depths 3 4 --pairs 2 --reverse
```

## Rollout and N-Ply Regression Benchmarks

Saved-baseline regression scripts (each supports `--save`/`--compare` with a material band of `max(SE, 0.01)` for equities and `0.005` for probabilities; the cubeless rollout bench widens the prob band to each prob's own SE):

- `scripts/bench_3t.py` — cube action
- `scripts/bench_3t_checker.py` — checker play (fixed dice in `scripts/bench_checker_dice.json`)
- `scripts/bench_3t_postmove.py` — post-move eval
- `scripts/bench_3t_cubeless.py` — cubeless rollout of post-move positions
- `scripts/bench_nply_cubeless.py` — standalone N-ply cubeless (post-move probs at 2–4 ply + contact/race benchmark ER)

### Apple Silicon Timing Baselines (v3.1.1, M4 Pro)

For fast local latency checks without external position files, run
[`scripts/bench_analyzer_overhead.py`](../scripts/bench_analyzer_overhead.py)
against `stage11` with `BGBOT_USE_ACCELERATE=ON`. On an idle Apple Silicon M4 Pro
at `v3.1.1` (3 rounds, median ms per call):

| Call | Median ms/call |
|---|---:|
| `cube_action` 1-ply | 0.011 |
| `cube_action` 2-ply | 0.21–0.22 |
| `cube_action` 3-ply | 2.36–2.38 |
| `checker_play` 1-ply | 0.081–0.084 |
| `checker_play` 2-ply | 0.79–0.84 |

For the full 13-workload `v3.1.0` → `v3.1.1` speedup table (move generation,
encoding, 2–4 ply cube/checker, 1T/2T/3T/full rollouts) and 2-ply cube thread
scaling on M4 Pro, see
[Apple Silicon & Analyzer Performance Measurements](model-history.md#apple-silicon--analyzer-performance-measurements).
