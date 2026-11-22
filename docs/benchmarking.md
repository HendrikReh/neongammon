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


## Rollout and N-Ply Regression Benchmarks

Saved-baseline regression scripts (each supports `--save`/`--compare` with a material band of `max(SE, 0.01)` for equities and `0.005` for probabilities; the cubeless rollout bench widens the prob band to each prob's own SE):

- `scripts/bench_3t.py` — cube action
- `scripts/bench_3t_checker.py` — checker play (fixed dice in `scripts/bench_checker_dice.json`)
- `scripts/bench_3t_postmove.py` — post-move eval
- `scripts/bench_3t_cubeless.py` — cubeless rollout of post-move positions
- `scripts/bench_nply_cubeless.py` — standalone N-ply cubeless (post-move probs at 2–4 ply + contact/race benchmark ER)
