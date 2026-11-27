# Training Pipeline and Data Formats

## Input Encodings

**Tesauro (196 inputs)** — PureRace only:
- 4 thermometer inputs per point per player (24 points x 2 players x 4 = 192)
- Plus bar and borne-off per player (4 more)

**Extended (244 inputs)** — Racing/Attacking/Priming/Anchoring:
- 122 features per player
- [0-95]: Point encoding (same as Tesauro)
- [96]: Bar / 2.0
- [97-99]: Borne-off (3 buckets)
- [100-121]: 22 GNUbg-style features (escape, containment, timing, etc.)
- Requires `init_escape_tables()` once before use

## Training Pipeline

Stage-specific training and benchmark recipes for Stages 4–10 live in
[`scripts/archive/`](../scripts/archive/README.md). Their existing script paths
also work. Archived recipes write their default experiment reports under `logs/`.

**CRITICAL: TD pre-training is required before SL.** SL from random init finds
unnatural minima. TD gives realistic probability distributions that SL refines.

1. **TD Self-Play (CPU)**: 25k-200k games @ alpha=0.1. Serial only (parallel TD deprecated).
2. **Supervised Learning (GPU)**: Backprop against GNUbg rolled-out probabilities,
   batch size 128 (4096 in the S11 scripts). Scripts call
   `neongammon.sl_train.supervised_train` / `supervised_train_preencoded`, which
   run on the Mac GPU through **Apple MLX** (`pip install mlx`). See "GPU SL
   trainer (`neongammon.sl_train`)" below.

**Key training rules:**
- All contact NNs train on ALL contact+crashed data, NOT game-plan subsets
- Narrow subsets cause catastrophic regression
- Game plan weight (`--gameplan-weight`) specializes each NN during SL

### Weight files

Both the C++ engine and the Python trainer read legacy and versioned files.
For a versioned file, pass `versioned=True` to `NeuralNetwork.save_weights`
or `sl_train.write_weights`. Existing training paths still write legacy files.

| Field | Encoding |
| --- | --- |
| Magic | Four bytes `NGWT` |
| Schema | Little-endian uint32, currently 1 |
| Encoder | Little-endian uint32. 1 = 196-input Tesauro, 2 = 244-input extended contact, 3 = 214-input legacy contact |
| Topology | Three little-endian uint32 values in hidden, input, output order |
| Payload | Hidden and output matrices in row-major float32, little-endian, with each bias in its row's final column |

The legacy format starts at the topology field. A versioned file must
match the encoder and dimensions, including all five outputs and the exact
payload size.

### GPU SL trainer (`neongammon.sl_train`)

`supervised_train(boards, targets, ...)` and
`supervised_train_preencoded(inputs, targets, ...)` return a dict with
`best_score`, `best_epoch`, `epochs_completed`, `total_seconds`,
`weights_path` and `best_weights_path`. All training scripts call them, and
gate on `sl_train.gpu_available()`. `sl_train.backend()` is `"mlx"` when MLX
has Metal, else `None`.

The update rules are upstream's CUDA trainer's: sigmoid-MSE delta
`(t-p)·p·(1-p)` (times the per-sample weight), backprop through the
pre-update output weights, SGD at `alpha / batch_rows` (the partial last batch
uses its own row count), one epoch = one shuffled pass (NumPy draws it), CPU
benchmark scoring directly from in-memory weights after every epoch, with
best-score weights saved as `.best`,
and random init from `bgbot_cpp.NeuralNetwork(nh, ni, 0.1, seed)`.
`tests/test_sl_train.py` holds the MLX weights to a float64 NumPy
transcription of the update rules (2e-5).

Measured 2026-11-17 on the M4 Pro (CPU busy with a rollout job): the snake-net
recipe (`run_s11_containment_sl.py --family-data s11-bg-snake-rollout`, from
`sl_s9_prim_race`, 20k ep @ α=3.1 + 60k ep @ α=1.0, batch 4096) reaches
holdout ER **53.96** (CUDA on the RTX 4070S: 53.95). A 4,096-row step takes
about 0.9 ms, close to the GPU's compute limit for a 244-400-5 net, which puts
that schedule at ~7.5 minutes against 2.5 minutes on the RTX. (That figure is
computed from the step time; the validation run itself took 11 minutes with an
earlier loop that synced the GPU after every epoch, 1.55x slower, same weights.)


> For the unshipped search-distillation experiment (`scripts/rank_distill.py`), see [docs/model-history.md](model-history.md#search-distillation-with-a-ranking-loss-scriptsrank_distillpy-2026-11-20).

### Cloud rollouts on Modal (`scripts/modal_rollout.py`)

Rollout targets, not GPU time, are what training data costs (SL takes minutes
on the Mac; TD is serial). `scripts/modal_rollout.py` rolls out a position
file (26 ints per line, side that just moved positive — what the harvest
scripts write) on Modal CPU containers and appends to the matching
`<name>-rollout` file in the SL trainers' format, plus a `.meta.jsonl`
sidecar (equity, SEs, seconds). It replaces the parent repo's
`rollout_backgame_positions.py` for this fork. Defaults are the S11 piles'
convention: `--model stage9`, 1,296 trials, no truncation, 3-ply checker play
throughout (`ultra_late_threshold=9999`), VR, PubEval 20/15 prefilter,
and the bearoff DB. Resume checks every label against its sidecar entry and
requires the same rollout settings, model weights, bearoff data, and engine
identity. Local runs identify the loaded native extension. Modal runs identify
the uploaded C++ source. If a file predates these checks or is missing a paired
row, use a new `--out` path or restore a verified data and metadata pair.

`--limit N` caps a run, `--max-usd` stops it at an estimated compute cost,
and `--local` runs it in-process on the Mac.

```bash
build/venv/bin/pip install modal && build/venv/bin/modal setup   # once
python scripts/modal_rollout.py data/<name>-data --limit 20        # small test
python scripts/modal_rollout.py data/<name>-data --containers 100 --max-usd 300
```

The image builds the engine from `cpp/` for Linux x86-64 (AVX2/FMA,
`-ffast-math`) — the one place this fork builds for anything but Apple
Silicon, and the reason the AVX2 kernels are still in the tree. It
picks marginally different moves from the Accelerate build, so values are not
bit-identical, only statistically equal. Measured 2026-11-18 on
`s11-bg-containment-rollout` rows: 40 boards rolled out locally reproduce the
stored targets (equity z mean 0.16, std 0.77, max 2.4), and 20 of them on
Modal match the local results within 1.7 SE. An 8-core container (16
threads, $0.41/h at list price) runs at ~0.8x the M4 Pro: ~20 s and
**~$0.002 per containment-type board** at 1,296 trials / 3-ply (Fargate was
~$0.005), so snake-type boards (~6x longer) cost ~$0.013. The cost estimate
counts rollout time only; Modal's dashboard has the real bill. Pure-race
boards take ~0.6 s (~$0.0001 each); run them with `--chunk 50` so one Modal
input carries 50 boards, or the per-input overhead rivals the rollout.

Operating notes. The account's container limit (100 on the Starter plan) is
shared by every run: a second run started while the first holds all 100 gets
no containers and waits silently, so split them (`--containers 85` and
`--containers 15`). Stop a run with `kill -TERM <pid>` of its Python process
(the script turns SIGTERM into a clean app shutdown) and check
`modal app list` — an ephemeral app whose client died keeps its containers
running and billing until stopped with `modal app stop -y <app id>`. Modal
pre-empts containers now and then; the affected boards are restarted
automatically. Re-running the same command resumes.

For primary Modal documentation links and rollout policy-dependence references (Tesauro & Galperin), see [docs/strength-references.md](strength-references.md#modal-and-rollout-measurements).

### Training a New 5-NN Model from Scratch

To train a new 5-NN model (e.g., with different hidden sizes), follow these steps.
The process is long-running (~25-30 hours total for the historical Stage 5 5-NN schedule;
the current `stage11` production model builds on a 17-NN pair model plus specialized
backgame/containment nets — see the pair-strategy and backgame sections below).

**Step 1: Create a training script** based on `scripts/run_stage5_training.py`.
Key parameters to customize:
- `N_HIDDEN` / `N_HIDDEN_PURERACE`: hidden layer sizes for contact / purerace NNs
- `MODEL_PREFIX` / `TD_MODEL_NAME`: file naming prefix (e.g., `sl_s5s` / `td_s5s`)
- `CONFIGS`: per-NN SL schedule (epochs, learning rates, game plan weights)

**Step 2: Register the model** in `python/neongammon/weights.py`:
```python
MODELS["stage5small"] = {
    "hidden": (100, 200, 200, 200, 200),
    "pattern": "sl_s5s_{plan}.weights.best",
}
```

**Step 3: Launch training as a detached process.**
Training is long-running and should be detached from your terminal or agent session.
Run from the neongammon repo root:
```bash
# IMPORTANT: Use python -u for unbuffered output.
nohup build/venv/bin/python -u scripts/run_stage5small_training.py \
  > logs/training.log 2> logs/training_err.log &
```

**Output buffering note:** Even with `-u`, C++ stdout from `bgbot_cpp` functions
(TD training, SL training) is internally buffered until the C++ function returns.
The TD benchmark_interval (default 10k games) triggers a benchmark + CSV write.
To monitor progress during TD training, check:
- `models/<td_model>_<plan>.weights` file timestamps (updated every benchmark_interval)
- `models/<td_model>.history.csv` (updated every benchmark_interval with game count +
  elapsed time + benchmark score). Note: CSV writes may also be delayed by OS buffering.

TD training prints benchmark scores every `benchmark_interval` games (default 10k).
SL training prints loss and benchmark scores every `print_interval` epochs (auto-set
to ~20 prints per phase, e.g., every 10 epochs for a 200-epoch phase).

**Step 4: Monitor training progress:**
```bash
# Check if process is alive
pgrep -fl run_stage5small_training

# Check weight file timestamps (updated every benchmark_interval)
stat -f '%m' models/td_s5s_racing.weights && date +%s

# Check history CSV
cat models/td_s5s.history.csv

# Check log output (may be delayed due to C++ internal buffering)
tail -20 logs/training.log
```

**Step 5: After training completes**, run benchmarks:
```bash
python scripts/run_stage5small_benchmarks.py
```

**Estimated timing** (Stage 5 Small — 100h/200h hidden, measured upstream on Windows + RTX 4070S; not re-measured on the Mac):
- TD Phase 1 (200k games @ α=0.1): ~3.9 hours
- TD Phase 2 (1M games @ α=0.02): ~19 hours
- SL (5 NNs, ~2500 epochs each): ~2-4 hours (GPU)
- Total: ~25-27 hours

For reference, Stage 5 (200h/400h) TD training at the larger hidden size is ~2x
slower per game due to the larger matrix multiplies.

### Key Training Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_stage5_training.py` | Stage 5 (5-NN, 200h/400h) full training |
| `scripts/run_stage5small_training.py` | Stage 5 Small (5-NN, 100h/200h) full training |
| `scripts/run_s8_training.py` | Stage 8 (17-NN pair strategy, 400h contact) end-to-end training |
| `scripts/run_s11_backgame_td.py` | Stage 11 backgame TD pre-training |
| `scripts/run_s11_backgame_sl.py` | Stage 11 backgame SL training |
| `scripts/run_s11_containment_sl.py` | Stage 11 containment / snake / massive SL training |
| `scripts/modal_rollout.py` | Cloud rollout target generation on Modal CPU containers |
| `scripts/run_td_gameplan_training.py` | TD self-play (standalone, low-level) |
| `scripts/run_gpu_sl_training.py` | GPU SL training via MLX (standalone, per-NN) |

### Standard 5-NN TD + SL Schedule (Stage 5 Baseline)

The standard 5-NN training schedule (used for Stage 5):

**TD (CPU, serial):**
- Phase 1: 200k games @ α=0.1 (high learning rate for initial learning)
- Phase 2: 1M games @ α=0.02 (low learning rate for refinement)

**SL (GPU, per-NN):**

All contact NNs train on ALL contact+crashed data (not game-plan subsets). The
**game plan weight (gpw)** controls specialization: positions matching the NN's
game plan get `gpw` × the gradient weight, while all other positions get weight 1.0.
Higher gpw = stronger specialization toward that plan's positions. PureRace trains
on separate purerace-only data (gpw not applicable).

| NN | Schedule | gpw | Effective gradient % |
|----|----------|-----|---------------------|
| Racing | `100ep@α=20 → 200ep@α=10 → 200ep@α=3.1 → 500ep@α=1.0` | 2.0 | ~48% |
| Attacking | same | 5.0 | ~59% |
| Priming | same | 5.0 | ~56% |
| Anchoring | same | 1.5 | ~27% |
| PureRace | `200ep@α=20 → 500ep@α=6.3 → 500ep@α=2.0` | — | 100% (separate data) |

**gpw tuning:** Racing uses gpw=2.0 instead of 5.0 because Racing positions are
the most common contact plan (~37% of training data). At gpw=5.0, Racing positions
dominate 74% of the gradient, which destabilizes smaller networks (200h). The
threshold is between 63% (gpw=3.0, stable) and 74% (gpw=5.0, diverges). Larger
networks (400h, Stage 5) can handle gpw=5.0 for Racing.

Each SL phase resumes from the `.best` weights of the previous phase. Benchmark
scoring runs after each epoch; the best-scoring weights are saved as `.best`.

See [docs/benchmarking.md](benchmarking.md) for all benchmarking commands.

### Hybrid Evaluator (Multi-Ply with Separate Filter Model)

`MultiPlyStrategy` supports an optional separate filter strategy for 1-ply filtering
and opponent move selection while keeping the main strategy at leaves. Full details live
in [docs/multi-ply.md](multi-ply.md#9-hybrid-evaluator-mode); see
[docs/model-history.md](model-history.md#stage-5-small-s5s--fast-filter-model) for why
halving hidden nodes did not meaningfully speed up 4-ply search.


## Training a New Pair Strategy Model (End-to-End Process)

This section documents the full algorithm for training a new 17-NN pair strategy
model (like S7 or S8), so future training runs can follow the same process.

### Prerequisites

- MLX installed (`pip install mlx`) for the SL phases
- C++ build with pair strategy support (`td_train_gameplan_pair`,
  `score_benchmarks_pair`, `create_multipy_pair`, `create_rollout_pair`)
- Training data in `data/`: `contact-train-data`, `crashed-train-data`, `purerace-train-data`
- Benchmark files in `data/`: `purerace.bm`, `racing.bm`, `attacking.bm`, `priming.bm`,
  `anchoring.bm`, `contact.bm`, `crashed.bm`, `race.bm`

### Step 1: Register the Model

Add an entry to `MODELS` in `python/neongammon/weights.py`:
```python
MODELS["stage9"] = {
    "hidden": (100,) + (400,) * 16,        # (purerace_h,) + (contact_h,) * 16
    "pattern": "sl_s9_{plan}.weights.best",
    "plans": "pair",
    "canonical_map": [0,1,2,3,4,5,6,7,8,9,10,12,12,13,14,12,12],
}
```

### Step 2: Create the Training Script

Copy `scripts/run_s8_training.py` and update these constants:
- `N_HIDDEN`: contact NN hidden size (e.g., 400)
- `MODEL_PREFIX`: e.g., `'sl_s9'`
- `TD_MODEL_NAME`: e.g., `'td_s9'`
- `GPW_CANDIDATES`: list of gpw values to scan (e.g., `[2, 5, 7, 10, 12]`)
- PureRace source: which model to copy PureRace weights from

### Step 3: Launch Training

Training is long-running (~30-50 hours depending on hidden size, measured
upstream on Windows + RTX 4070S). Launch it as a detached process from the
neongammon repo root; its SL phases run through MLX:

```bash
nohup build/venv/bin/python -u scripts/run_s9_training.py \
  > logs/s9_training.log 2> logs/s9_training_err.log &
```

### Step 4: Monitor Progress

Check training progress periodically (e.g. every 10 minutes via a scheduled task or manual check) with:

1. **Process alive?** `ps -o pid,%cpu,rss,lstart,command -p $(pgrep -f run_s9_training)`
2. **Training log tail:** `tail -30 logs/s9_training.log`
3. **Errors:** `tail -10 logs/s9_training_err.log`
4. **Weight files:** `ls -lt models/td_s9* models/sl_s9* models/s9_gpw_scan/ 2>/dev/null | head -20`

Summarize the current phase (TD Phase 1, TD Phase 2, GPW scan, SL phases 3-4, benchmarks), progress, and any errors.

The training script also prints its own progress updates every 10 minutes to the log
(via the `ProgressTracker` class).

You can also check manually:
```bash
tail -30 logs/s9_training.log
```

Check weight file timestamps to verify training is progressing:
```bash
ls -la models/td_s9_*.weights
ls -la models/s9_gpw_scan/
ls -la models/sl_s9_*.weights.best
```

### Step 5: Training Pipeline Phases

The script runs these phases automatically:

1. **Copy PureRace** from previous model (no retraining needed if same 100h arch)
2. **TD Phase 1**: 300k self-play games @ α=0.1 (~5-11h depending on hidden size)
3. **TD Phase 2**: 1.5M self-play games @ α=0.02 (~25-55h)
4. **GPW Scan**: For each of 13 canonical contact NNs:
   - Try each GPW candidate with SL phases 1-2 (100ep@α=20 + 200ep@α=10)
   - Score using pair-filtered benchmarks (only positions matching the NN's
     player×opponent game plan pair)
   - Select GPW minimizing pair-filtered ER
   - Save results to `models/s*_gpw_scan/optimal_gpw.json`
5. **SL Phases 3-4**: For each canonical contact NN:
   - Resume from GPW scan's optimal `.best` weights
   - Train phases 3 (200ep@α=3.1) and 4 (500ep@α=1.0)
   - Use pair-filtered benchmarks for best-weight selection
   - Copy canonical weights to alias NNs (shared group)
6. **Scoring**: Pair-filtered comparison with reference model (e.g., S5)
7. **Benchmarks**: 1-ply contact/race, 2-4 ply contact, top-100 worst positions

### Step 6: Record Results

After training completes:
1. Update `docs/model-benchmarks.md` with benchmark results
2. Update `docs/model-history.md` with the new model details (hidden sizes, GPW values,
   pair-filtered ER results, training scripts)
3. If promoting to `PRODUCTION_MODEL`, also add the new weight files to `sdist.include` in `pyproject.toml`
4. Commit weight files and updated docs

### Recovery / Partial Reruns

The script supports resuming from any point:
- `--sl-only`: Skip TD, use existing TD weights
- `--phase34-only`: Skip TD and GPW scan, resume from saved optimal GPW
- `--score-only`: Skip all training, just score existing weights
- `--benchmark-only`: Run multi-ply and top-100 benchmarks only
- `--nn race_race att_att`: Train only specific NNs


## Benchmark Data Format

GNUbg `.bm` files. Each "move" line:
```
m <position_string> <die1> <die2> <best_pos> <2nd_pos> <2nd_err> ...
```
Score = mean error * 1000 (millipips). Lower is better.

## GNUbg Training Data Format

Each line: `<20-char position string> <P_win> <P_gw> <P_bw> <P_gl> <P_bl>`


## Backgame Data Files

Backgame positions and rollout probabilities for training specialized backgame NNs.
All files in `data/`. Generated by `scripts/generate_backgame_data.py` (positions)
and `scripts/modal_rollout.py` (rollout probabilities on Modal; replaced upstream's `rollout_backgame_positions.py` via Parallelizor in this fork).

### Position files (input)

One position per line: 26 space-separated integers (board representation).

| File | Positions | Description |
|------|-----------|-------------|
| `player-backgame-train-data` | 93,770 | Player BG training positions |
| `player-backgame-benchmark-data` | 10,419 | Player BG benchmark positions |
| `opponent-backgame-train-data` | 93,770 | Opponent BG training positions |
| `opponent-backgame-benchmark-data` | 10,419 | Opponent BG benchmark positions |

### Rollout files (output)

One position per line: 26 space-separated integers (board) followed by 5 floats
to 4 decimal places (cubeless rolled-out probabilities: W, Gw, Bw, Gl, Bl).
Probabilities are post-move, from the perspective of the player whose checkers
are positive. Rolled out with Stage 8 pair model, 1296 trials, 3-ply cubeless
decisions throughout, PubEval 20/15 prefilter, VR enabled.

| File | Description |
|------|-------------|
| `player-backgame-train-rollout` | Rolled-out probs for player BG training |
| `player-backgame-benchmark-rollout` | Rolled-out probs for player BG benchmark |
| `opponent-backgame-train-rollout` | Rolled-out probs for opponent BG training |
| `opponent-backgame-benchmark-rollout` | Rolled-out probs for opponent BG benchmark |

### Backgame criteria

A position qualifies as a **player backgame** if: game plan pair is (anchoring,
racing), player is behind in the race (higher pip count), and player holds ≥2
anchors in the opponent's home board (points 19-24). **Opponent backgame** is the
mirror: (racing, anchoring) pair with opponent behind and holding ≥2 anchors in
player's home board (points 1-6).

### Benchmark ER (1-ply Stage 8)

Player backgame: **45.98** (vs overall contact ER of 9.49 — 4.8× worse).

### Stage 9 Backgame NN Training Process

The Stage 9 backgame NNs (player_bg, opponent_bg) are trained iteratively,
re-rolling out training and benchmark data between rounds to improve target
quality as the model improves.

**Round 1 — Bootstrap from S8 rollouts:**
1. Roll out all player/opponent backgame training and benchmark positions using
   Stage 8 pair model (1,296 trials, 3-ply cubeless decisions throughout, PubEval
   20/15 prefilter, VR enabled). Output: `*-backgame-*-rollout` files.
2. SL training for both player_bg and opponent_bg NNs:
   - 100k steps @ α=3.1
   - 250k steps @ α=1.0
   - Starting from S8 fallback weights (anchoring pair NN)

**Round 2 — Re-rollout with improved model:**
1. Re-roll out all training and benchmark positions using the Stage 9 model
   (same rollout settings: 1,296 trials, 3-ply, VR, PubEval prefilter). The
   improved backgame NNs produce more accurate rollout targets.
2. Redo SL training with the same schedule:
   - 100k steps @ α=3.1
   - 250k steps @ α=1.0
   - Starting from round 1's best weights

The re-rollout step is necessary because the S8 rollout targets have ~22 ER
noise on backgame positions (measured by comparing S8 vs S9 rollouts on 10
random positions). Once the model's backgame ER drops below ~20, the training
data noise becomes the limiting factor for further improvement.
