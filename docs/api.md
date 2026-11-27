# Python and C++ API Reference

The pure-Python `neongammon.weights`, `neongammon.text_export` and
`neongammon.settings` modules can be imported without the native extension.
Engine exports such as `BgBotAnalyzer` load the extension on first use.

For game/match simulation (`simulate_game`, `simulate_match`, `PureTDPolicy`, `DifficultyPolicy`), see [docs/simulation.md](simulation.md) and [docs/engine.md](engine.md). For algorithmic details of multi-ply search and rollouts, see [docs/multi-ply.md](multi-ply.md) and [docs/rollout.md](rollout.md).

## Interfaces

**IMPORTANT: When writing scripts or code that needs checker play or cube action
analysis without targeting a specific model, always use `BgBotAnalyzer` from
`neongammon`.** This is the standard model-independent Python interface — it
abstracts away whether the production model is a 5-NN, 17-NN pair, or 24-NN
backgame-aware phased strategy (`stage11`).
Do NOT call low-level `bgbot_cpp` functions directly (e.g.
`cube_decision_nply`, `cube_decision_nply_pair`) unless you need to target a
specific model type or need parameters not exposed by the public API.

The library provides both **Python** and **C++** interfaces across seven categories
of functionality. All Python functions default to the production model (`stage11`); pass
`weights=WeightConfig.from_model("stage9")` to use the other bundled model.

NeonGammon match analysis supports away scores of up to 25 for each player.
Python analysis raises `ValueError` when either player needs more points. A game
in a longer match is supported once both players need 25 points or fewer.

### 1. Checker Play Analytics

Given a board position, two dice, and cube information, return all legal moves
ranked by equity, with cubeless post-move probabilities for each.

**Python** — `BgBotAnalyzer.checker_play()` (`python/neongammon/facade.py`):
```python
from neongammon import BgBotAnalyzer, STARTING_BOARD

analyzer = BgBotAnalyzer(eval_level="3ply", cubeful=True)
result = analyzer.checker_play(STARTING_BOARD, 3, 1, cube_value=1, cube_owner="centered")
# result: CheckerPlayResult with .moves (list[MoveAnalysis], best first)
for m in result.moves[:3]:
    print(f"{m.equity:+.3f}  {m.probs.win:.1%}  diff={m.equity_diff:+.4f}")

# Match play: add away1, away2, is_crawford keyword args
result = analyzer.checker_play(STARTING_BOARD, 3, 1, cube_value=1, cube_owner="centered",
                                away1=5, away2=3, is_crawford=False)
```

**C++** — Compose `possible_boards()` + `GamePlanStrategy::evaluate_probs()` + sort:
```cpp
#include "bgbot/moves.h"
#include "bgbot/strategy.h"
std::vector<Board> candidates;
possible_boards(board, die1, die2, candidates);
GamePlanStrategy strat(pr_w, rc_w, at_w, pm_w, an_w, ...);
for (auto& c : candidates) {
    auto probs = strat.evaluate_probs(c, board);
    double eq = NeuralNetwork::compute_equity(probs);
}
// Sort by equity descending.
```

### 2. Post-Move Position Analytics

Given a post-move board (right before the opponent's turn) and cube information,
return cubeful equity, cubeless equity, and cubeless probabilities.

**Python (single)** — `BgBotAnalyzer.post_move_analytics()` (`python/neongammon/facade.py`):
```python
from neongammon import BgBotAnalyzer

analyzer = BgBotAnalyzer(eval_level="2ply")
result = analyzer.post_move_analytics(post_move_board, cube_owner="centered")
# result: PostMoveAnalysis with .probs, .cubeless_equity, .cubeful_equity, .eval_level

# Match play:
result = analyzer.post_move_analytics(board, cube_owner="centered",
                                       away1=5, away2=3, is_crawford=False)
```

**Python (batch, parallelized)** — `batch_post_move_evaluate()` (`python/neongammon/batch.py`):
```python
from neongammon import batch_post_move_evaluate

positions = [
    {"board": board1, "cube_owner": "centered"},
    {"board": board2, "cube_owner": "player"},
]
results = batch_post_move_evaluate(positions, eval_level="1ply", n_threads=0)
# results: list[PostMoveAnalysis]
for r in results:
    print(f"CL={r.cubeless_equity:+.3f}  CF={r.cubeful_equity:+.3f}")
```

**C++** — `GamePlanStrategy::evaluate_probs()` + `cl2cf_money()`:
```cpp
GamePlanStrategy strat(...);
bool race = is_race(board);
auto probs = strat.evaluate_probs(board, race);
float cl_eq = NeuralNetwork::compute_equity(probs);
float x = cube_efficiency(board, race);
float cf_eq = cl2cf_money(probs, owner, x);
```

C++ batch: `bgbot_cpp.batch_evaluate_post_move(positions, strategy, n_threads)` via
pybind11; takes `list[(board, CubeOwner)]`, returns `list[dict]` with `probs`,
`cubeless_equity`, `cubeful_equity`.

### 3. Cube Action Analytics

Given a pre-roll board position and cube information, return cubeful equity
information for the three cube states (No Double, Double/Take, Double/Pass),
cubeless equity, and cubeless probabilities.

**Python** — `BgBotAnalyzer.cube_action()` (`python/neongammon/facade.py`):
```python
from neongammon import BgBotAnalyzer

analyzer = BgBotAnalyzer(eval_level="3ply", cubeful=True)
cube = analyzer.cube_action(board, cube_value=1, cube_owner="centered")
# cube: CubeActionResult with .equity_nd, .equity_dt, .equity_dp,
#   .should_double, .should_take, .optimal_action, .probs, .cubeless_equity
# jacoby=True by default for unlimited games; pass jacoby=False to disable

# Match play (Jacoby auto-disabled):
cube = analyzer.cube_action(board, cube_value=1, cube_owner="centered",
                             away1=5, away2=3, is_crawford=False)

# 2-ply details: per-roll breakdown for ND and DT scenarios (requires >= 2-ply)
cube = analyzer.cube_action(board, cube_value=1, cube_owner="centered",
                             incl_2ply_details=True)
# cube.details: dict with "nd" and "dt" keys, each a list of 21 player roll dicts
# See "2-Ply Detail Fields" below for structure
nd_rolls = cube.details["nd"]   # No Double scenario
dt_rolls = cube.details["dt"]   # Double/Take scenario

# The decision alone (no probs / cubeless equity): skips the second, cubeless
# N-ply search cube_action runs for those fields, about 60% of the call at
# 2-4 ply. Same equities and flags; simulate_game uses it when a policy has it.
d = analyzer.cube_decision(board, cube_value=1, cube_owner="centered")
# d: CubeDecisionResult with .equity_nd/dt/dp, .should_double, .should_take,
#   .optimal_equity, .optimal_action, .is_beaver, .eval_level
```

Cube-owner names are case-insensitive. Use `"centered"`, `"player"`,
`"opponent"`, or the corresponding native `CubeOwner` value. Invalid owners
raise `ValueError` when supplied to the cube-owner resolver.

**2-Ply Detail Fields** (`incl_2ply_details=True`, requires >= 2-ply):

Returns per-roll details for the first two turns under both the **ND** (No Double)
and **DT** (Double/Take) scenarios. The ND section shows equities assuming the
player does not double; the DT section shows equities assuming the player doubles
and the opponent takes — even when those are not the optimal decisions. All boards
are from the original player's perspective. All equities are per-initial-cube, from
the player's perspective. No extra computational cost: the details are recorded
from the very tree the plain call walks (`cube_decision_nply_with_details` in
`cube_eval.cpp` runs the evaluation engine with a per-roll capture at its top two
levels), so `equity_nd` / `equity_dt` / the decision equal the plain call's exactly.

`cube.details` — dict with two keys:
- `"nd"` — list of 21 player roll dicts (No Double scenario)
- `"dt"` — list of 21 player roll dicts (Double/Take scenario)

Boards (`checkers`) are identical in both sections: the tree makes one move pick
per roll for both scenarios. Only the equities differ.

Each player roll dict (in both `"nd"` and `"dt"`):

| Field | Type | Description |
|-------|------|-------------|
| `die1` | int | First die (1-6) |
| `die2` | int | Second die (1-6) |
| `checkers` | list[int] | 26-element post-move board after optimal checker play |
| `cubeful_equity` | float | Cubeful equity incorporating opponent's optimal cube decision within this scenario. |
| `opponent_dp` | bool | Present and `True` when opponent has D/P in this scenario (game over) |
| `opponent_rolls` | list[dict] | 21 opponent roll details. **Absent** if the player's move is terminal, opponent has D/P, or the analysis is 2-ply. |

Each element of `opponent_rolls`:

| Field | Type | Description |
|-------|------|-------------|
| `die1` | int | First die (1-6) |
| `die2` | int | Second die (1-6) |
| `checkers` | list[int] | 26-element post-move board (player's perspective) |
| `cubeful_equity` | float | Cubeful equity incorporating player's optimal cube decision, scaled to initial-cube units. |

**ND section**: Cube remains at its current value. The opponent may double (if
they own or it's centered); per-roll equities reflect the opponent's optimal cube
action. The weighted average of `cubeful_equity` across all 21 ND player rolls
(doubles weight 1, non-doubles weight 2, divided by 36) equals `equity_nd`.

**DT section**: Cube is doubled (2× initial), opponent owns. The opponent may
redouble; per-roll equities reflect the opponent's optimal cube action at the
doubled cube level, scaled back to per-initial-cube units. The weighted average
of `cubeful_equity` across all 21 DT player rolls equals `equity_dt`.

Move selection is the engine's at every level, the captured ones included:
1-ply cubeful equity against the ND scenario's cube state of the side to move
(see [docs/multi-ply.md](multi-ply.md#4-n-ply-cubeful-evaluation-evaluate-all-and-decide)), after the PubEval keep-15 prefilter when a roll
has >16 candidates, with root routing (a snake root pins the whole tree, as in
the plain call). Equities are evaluated at (N-1)-ply for player rolls and
(N-2)-ply for opponent rolls — so at 3-ply, player-roll equities are 2-ply
accurate and opponent-roll equities use 1-ply Janowski. An ND player-roll equity
is the opponent's standalone (N-1)-ply cube analysis of that board (exactly, from
3-ply on; at 2-ply within ~0.003, because the 1-ply leaf reuses the pick batch's
grouped NN output). A DT per-roll equity can differ more from a standalone
cube-2 analysis, since its subtree's picks are made against the ND cube state
(`tests/test_2ply_details.py`).

**2-ply behavior**: at 2-ply only the player-roll level is captured — the
opponent-roll level would sit below the 1-ply leaf. Each post-move position is
evaluated directly at the 1-ply Janowski leaf (the same subtree the plain 2-ply
call evaluates), so per-roll equities are 1-ply accurate, `opponent_rolls` is
absent from every entry, and the cost is that of a plain 2-ply cube evaluation.

**Python (batch, pre-roll)** — `batch_evaluate()` and `batch_cube_action()` (`python/neongammon/batch.py`):
```python
from neongammon import batch_evaluate, batch_cube_action

positions = [{"board": b, "cube_value": 1, "cube_owner": "centered"} for b in boards]
results = batch_evaluate(positions, eval_level="3ply", n_threads=0)
# results: list[PositionEval] — includes probs, cubeless/cubeful equity, cube decision

# Same root-Janowski batch evaluation returned as list[CubeActionResult]:
cube_results = batch_cube_action(positions, eval_level="3ply", n_threads=0)

# Match play: add optional away1, away2, is_crawford to position dicts
positions = [{"board": b, "cube_value": 1, "cube_owner": "centered",
              "away1": 5, "away2": 3} for b in boards]
```
Note: `batch_evaluate` and `batch_cube_action` apply Janowski once at the root on N-ply
cubeless probabilities (using the 1-ply money contact cube life at every depth) rather
than running the recursive N-ply cubeful cube tree.

**C++ / pybind11 (`bgbot_cpp`)** — `evaluate_cube_decision()` (1-ply), `cube_decision_nply()` (N-ply),
`cube_decision_rollout()`:
```python
# pybind11 defaults: jacoby=True, beaver=True for money play (auto-disabled in match play)
# C++ struct defaults: CubeInfo has jacoby=false, beaver=false; RolloutConfig has ultra_late_threshold=2
# Python simulation / cube_points default beaver=False; Python full/2T/3T rollouts default ultra_late_threshold=9999
bgbot_cpp.evaluate_cube_decision(checkers, cube_value, owner, ..., jacoby=True, beaver=True)
bgbot_cpp.cube_decision_nply(checkers, cube_value, owner, n_plies, ..., jacoby=True, beaver=True)
bgbot_cpp.cube_decision_rollout(checkers, cube_value, owner, ..., jacoby=True, beaver=True)
```

C++ batch pre-roll: `bgbot_cpp.batch_evaluate_positions(positions, strategy, n_threads)`
via pybind11; takes `list[(board, cube_value, CubeOwner[, away1, away2, is_crawford])]`,
returns `list[dict]`.

C++ batch checker play: `bgbot_cpp.batch_checker_play(inputs, strategy_1ply, [strategy_nply,]
filter_max_moves, filter_threshold, n_threads)` via pybind11; takes `list[dict]` with
`{board, die1, die2, cube_value, cube_owner}`, returns `list[dict]` each with `moves` list
sorted by cubeful equity desc. Overloads for 1-ply and for N-ply (1-ply strategy +
MultiPlyStrategy). The N-ply overloads use the checker search in `AnalyzerSession`.
`BgBotAnalyzer.checker_play` calls the single-position routine
through `bgbot_cpp.checker_play_nply`, passing its forced boards, cube cap and thread
budget. At 2-ply the batch distributes threads across rows first. At 3-ply it
distributes legal moves from all rows over the same pool. At 4-ply it keeps
each roll together to reuse subtrees across its moves. Both APIs return the
same ranked moves with the bearoff database enabled or disabled. Rows may
carry `max_cube_value` and `force_boards`. At the depths in
`cubeful_filtered_plies` (the wrapper passes `CUBEFUL_FILTERED_PLIES`, i.e. 2-ply) only the
filter's survivors and promotions go through the cube-aware tree, the rest keep 1-ply
values; at other depths every move does. `tests/test_batch_checker_play_parity.py`.

For `batch_checker_play`, `filter_max_moves` and `filter_threshold` are
unused at 1-ply and control the initial 2-ply shortlist. Its defaults
keep up to five moves less than 0.08 cubeful equity behind the 1-ply
leader. Forced boards and promotions can expand that shortlist. At 3- and
4-ply the public API evaluates every
legal root move and raises `ValueError` for a non-default width or window.
Raw `bgbot_cpp.batch_checker_play` callers can enable a shortlist at those
depths by including the requested depth in `cubeful_filtered_plies`. The
width and window then apply to that root alone. Inside each candidate's
recursive search the engine selects one checker reply per roll.

The standalone mobile C ABI uses the same checker scoring and cube evaluation
at 1-ply and 2-ply. A 2-ply checker response labels moves retained at 1-ply
with `"eval_level": "1-ply"`. Run `python -m pytest -q tests/test_mobile_parity.py`
in the development build environment to compile both native targets and compare
the public desktop results with the C ABI JSON.

**Python batch wrapper** — `batch_checker_play()` (`python/neongammon/batch.py`):
```python
from neongammon import batch_checker_play
positions = [
    {"board": b, "die1": 3, "die2": 1, "cube_value": 1, "cube_owner": "centered"}
    for b in boards
]
results = batch_checker_play(positions, eval_level="3ply", n_threads=0)
# results: list[CheckerPlayResult], each with .moves sorted best-first
```

**Keeping models loaded across batch calls** — `BatchModels` (`python/neongammon/batch.py`).
Each batch call loads the model's weights (~15 ms for Stage 11). Callers that submit many
small batches pass one `BatchModels` instead, which loads the base strategy once and builds
every N-ply strategy around it (`bgbot_cpp.create_multipy_from_base`). Results are identical.
An empty batch loads nothing. Like `BgBotAnalyzer`, it attaches the one-sided bearoff
database by default (`bearoff_db=True`, a path, or `False`; loaded once per process): the
1-ply strategy is wrapped in `BearoffStrategy` and the N-ply strategies stop their search at
bearoff positions, so batch and analyzer agree there too. The batch functions take
`bearoff_db=` when they are given `weights` rather than `models`.
```python
from neongammon import BatchModels, batch_checker_play, batch_evaluate
models = BatchModels()          # or BatchModels(WeightConfig.from_model("stage9"))
for chunk in chunks:
    batch_checker_play(chunk, eval_level="2ply", models=models)
    batch_evaluate(chunk, eval_level="1ply", models=models)
```

**Shared analysis and recursive cube batches**

```python
from neongammon import BatchModels, BgBotAnalyzer, batch_cube_action_nply

checker = BgBotAnalyzer(eval_level="2ply")
models = BatchModels.from_analyzer(checker)
rollout = BgBotAnalyzer(eval_level="truncated1", models=models)
cube_decisions = batch_cube_action_nply(
    positions, eval_level="2ply", models=models, n_threads=4, decision_only=True,
)
```

The recursive batch returns `CubeActionResult` at 1–4 ply, or
`CubeDecisionResult` with `decision_only=True`. Positions accept the same
cube owner, match score, Crawford, Jacoby, beaver and maximum cube value
as a scalar analysis. Short batches spend spare threads inside each cube
search. At 3-ply and deeper, bearoff database rows run separately so a lone
contact row gets the full thread budget.
`batch_cube_action_nply` has no root checker shortlist. The recursive cube
search and the accompanying cubeless probability calculation each choose
one checker reply per roll. At 3- and 4-ply both the Python and raw batch
APIs require the default `filter_max_moves=5` and `filter_threshold=0.08`
in full and decision-only mode. Omitting them or passing them explicitly
gives the same results. A non-default value raises `ValueError` even for
an empty batch. At 1- and 2-ply the recursive cube batch still accepts
these settings for compatibility and ignores their values.
`batch_cube_action` continues to apply a Janowski conversion at the root.

`checker.top_two_promote(board, moves, max_gap=0.010, **cube_context)`
returns the ranking and its depth label. Supply the two leading moves in
descending equity order. Ties are valid, and a reversed pair raises
`ValueError`. For a close 2-ply contact decision it searches both leaders
at 3-ply and returns those two moves. Races, forced moves and wider gaps
retain the original list. The public `flip_owner`,
`contact_cube_life` and `contact_cube_slope` functions expose the engine's
perspective and money cube-life conventions.

The raw `bgbot_cpp.score_checker_endpoints` and `bgbot_cpp.batch_cube_action_nply`
bindings require `n_plies >= 1`, including for empty inputs.

### 4. Game Plan Classification

Given a board position, return the game plan for the player on roll and the
opponent. Plans: `"purerace"`, `"racing"`, `"attacking"`, `"priming"`, `"anchoring"`.

**Python (both plans)** — `classify_game_plans()` (`python/neongammon/board.py`):
```python
from neongammon import classify_game_plans

result = classify_game_plans(board)
# result: GamePlanResult with .player and .opponent
print(f"Player: {result.player}, Opponent: {result.opponent}")
```

**Python (player only)** — `classify_game_plan()` (`python/neongammon/board.py`):
```python
from neongammon import classify_game_plan
plan = classify_game_plan(board)  # -> str: "purerace", "racing", etc.
```

**C++** — `classify_game_plan()` (`encoding.h`):
```cpp
GamePlan gp = classify_game_plan(board);          // player on roll
GamePlan opp_gp = classify_game_plan(flip(board)); // opponent
```

C++ batch: `bgbot_cpp.classify_game_plans_batch(boards_np)` via pybind11;
takes `numpy array [N, 26]`, returns `int32 array` (0=purerace, 1=racing, etc.).

### 5. Game Utilities

Board manipulation, move generation, and game state queries.

**Python** — `python/neongammon/board.py` (all importable from `neongammon`):

| Function | Purpose |
|----------|---------|
| `flip_board(board)` | Flip to opponent's perspective |
| `possible_moves(board, die1, die2)` | All legal resulting positions |
| `possible_moves_all(board)` | Compact legal resulting positions across all 21 rolls |
| `possible_single_die_moves(board, die)` | Single-die moves with from/to (for UI) |
| `check_game_over(board)` | 0=not over, ±1/±2/±3 = single/gammon/backgammon |
| `is_race(board)` | True if contact is broken |
| `is_crashed(board)` | True if position is crashed |
| `can_double_match(away1, away2, cube_value, owner, is_crawford=False, *, max_cube_value=0)` | Whether the player on roll can double at this score and cube cap. Use (0, 0) for money play and 0 for an unlimited cube. |
| `invert_probs(probs)` | Swap player/opponent probabilities |
| `STARTING_BOARD` | Standard 26-element starting position |

**C++** — `board.h`, `moves.h`:

| Function | Purpose |
|----------|---------|
| `flip(board)` | Flip perspective |
| `possible_boards(board, d1, d2, results)` | Legal move generation |
| `check_game_over(board)` | Terminal detection |
| `is_race(board)` | Contact check |
| `is_crashed(board)` | Crashed position check |
| `invert_probs(probs)` | Invert 5-probability array |

### 6. Match Info

Match play gammonless take points, dead cube take points, and gammon prices.

**Python** — `python/neongammon/matchinfo.py`:

| Function | Description |
|----------|-------------|
| `take_points(away1, away2, cube_value)` | Gammonless live cube take points. Uses empirically-derived lookup tables (away-scores 2-5 for cv=1; select scores for cv=2) with Janowski approximation (x=0.68) as fallback. Handles automatic redouble cases (doubler wins match on single win at new cube level) with a special dead cube formula at 4C. Returns `(player1, player2)` tuple. |
| `take_points_dead_cube(away1, away2, cube_value)` | Gammonless dead cube take points, calculated exactly from the MET. Returns `(player1, player2)` tuple. |
| `take_points_janowski(away1, away2, cube_value, x)` | Gammonless Janowski live cube take points with explicit cube life index x. Returns `(player1, player2)` tuple. |
| `gammon_prices(away1, away2, cube_value)` | Gammon prices calculated exactly from the MET. Returns `(player1, player2)` tuple. |

All take points are gammonless (assume zero gammon/backgammon probability). For money
games: take point = 0.22, dead cube take point = 0.25, gammon price = 0.5 (both players).

### 7. Luck (Per-Roll Dice Fortune)

Given a position's cube analytics and the roll that actually happened, return how
lucky that roll was — a signed equity number.

**What the luck value means.** Luck is measured in **equity units, from the
perspective of the player who rolled**:

```
luck = actual_equity - average_equity
```

- `actual_equity` — cubeful equity after the *best play* with the roll that
  actually happened.
- `average_equity` — the weight-averaged equity over *all* possible rolls from
  the same position (doubles weight 1, non-doubles weight 2; ÷36).

So luck answers "how much did this roll help versus a typical roll here?"
**Positive = lucky** (the roll's best play beats an average roll), **negative =
unlucky**. Because it is a deviation from the mean, luck **averages to zero over
many rolls** — summing per-roll luck across a game isolates dice fortune from
decision quality (a player with high total luck won the dice, not necessarily the
game). This is the app-facing per-roll Luck shown in PR breakdowns; it is
distinct from **VR** (variance-reduction luck used internally to denoise
rollouts — see [docs/rollout.md](rollout.md#4-variance-reduction-vr)).

**This is a pure function over analytics, not a fresh evaluation.** The per-roll
equities are exactly the **ND (No Double) per-roll cubeful equities** that
`cube_action(incl_2ply_details=True)` already returns in `details["nd"]`. Luck
runs **no additional neural-net evaluation** — pass in analytics you already
have. Cube legality is irrelevant: the ND per-roll details exist whether or not
the player owns the cube.

**Python** — `python/neongammon/luck.py` (both importable from `neongammon`):

```python
from neongammon import BgBotAnalyzer, roll_luck

analyzer = BgBotAnalyzer(eval_level="3ply")

# Preferred: reuse analytics you already computed (no extra evaluation).
cube = analyzer.cube_action(board, cube_value=1, cube_owner="centered",
                            incl_2ply_details=True)          # the "bot analytics"
luck = roll_luck(cube, die1=3, die2=1)
# luck: LuckResult | None
#   .luck, .actual_equity, .average_equity, .ply, .level_label, .per_roll
print(f"{luck.luck:+.3f}  ({luck.level_label})")

# One-shot convenience (runs the cube analysis for you, then computes luck):
luck = analyzer.roll_luck(board, 3, 1)

# Opening roll excludes doubles (15 rolls, not 21):
luck = analyzer.roll_luck(STARTING_BOARD, 3, 1, is_opening_roll=True)
```

| Function | Description |
|----------|-------------|
| `roll_luck(cube, die1, die2, *, ply=None, level_label=None, is_opening_roll=False)` | Luck from a `CubeActionResult` produced with `incl_2ply_details=True`. `ply`/`level_label` default to being derived from `cube.eval_level` (an N-ply cube analysis gives (N-1)-ply luck). Returns `LuckResult` or `None`. |
| `luck_from_equities(per_roll, die1, die2, *, ply, level_label, is_opening_roll=False)` | Pure kernel over a sequence of `RollEquity`. Use when the per-roll equities come from somewhere other than `cube_action` (e.g. `batch_checker_play` best-move equities). |
| `BgBotAnalyzer.roll_luck(board, die1, die2, *, cube_value, cube_owner, away1, away2, is_crawford, jacoby, beaver, is_opening_roll=False)` | Convenience: runs `cube_action(incl_2ply_details=True)` at the analyzer's eval level, then computes luck. Prefer the free `roll_luck` when you already hold the cube analytics. |

**Match play** flows through the same `away1`/`away2`/`is_crawford` cube analysis,
so luck is expressed in match-equity-derived cubeful units automatically.

**Degenerate positions** (the actual roll missing from the details, or no details
present) return `None` — the caller decides how to record "no luck computed".

### Model Selection

Wheels and source distributions include weights for the production model (`stage11`)
and `stage9`. The other registered models (`stage3` through `stage8`, including
`stage5small`, and `stage10`) require weights from the Git checkout.

```python
from neongammon import BgBotAnalyzer, batch_evaluate, batch_post_move_evaluate
from neongammon.weights import WeightConfig

weights = WeightConfig.from_model("stage9")

# Single-position
analyzer = BgBotAnalyzer(weights=weights, eval_level="2ply")

# Batch pre-roll
results = batch_evaluate(positions, eval_level="1ply", weights=weights)

# Batch post-move
results = batch_post_move_evaluate(positions, eval_level="1ply", weights=weights)
```

Backgame-aware models can supply a `routing` table on `WeightConfigPair`.
The table assigns network slots to contact pairs and optional specialist
routes. The search checks snake, containment, massive backgames, backgame
category, early containment, and the side-specific backgame gate in that
order before choosing the normal contact-pair slot. Slot zero uses the
pure-race encoder. The other slots use the extended-contact encoder. A
specialist can pin contact evaluation below the root via `root_pins`.
The Stage 9–11 registry entries provide concrete examples in
[`weights.py`](../python/neongammon/weights.py).

### Return Types

| Type | Used by | Key fields |
|------|---------|------------|
| `CheckerPlayResult` | `checker_play()` | `.moves` (list[MoveAnalysis]), `.board`, `.die1`, `.die2` |
| `MoveAnalysis` | In CheckerPlayResult | `.board`, `.equity`, `.cubeless_equity`, `.probs`, `.equity_diff` |
| `PostMoveAnalysis` | `post_move_analytics()`, `batch_post_move_evaluate()` | `.probs`, `.cubeless_equity`, `.cubeful_equity` |
| `CubeDecisionResult` | `cube_decision()` | `.equity_nd/dt/dp`, `.should_double`, `.should_take`, `.optimal_equity`, `.optimal_action`, `.is_beaver` |
| `CubeActionResult` | `cube_action()` | `.equity_nd/dt/dp`, `.should_double`, `.should_take`, `.optimal_action`, `.probs`, `.details` (optional dict with `"nd"`/`"dt"` keys, with `incl_2ply_details`) |
| `PositionEval` | `batch_evaluate()` | `.probs`, `.cubeless_equity`, `.cubeful_equity`, `.equity_nd/dt/dp`, `.optimal_action` |
| `GamePlanResult` | `classify_game_plans()` | `.player`, `.opponent` |
| `LuckResult` | `roll_luck()`, `luck_from_equities()`, `BgBotAnalyzer.roll_luck()` | `.luck`, `.actual_equity`, `.average_equity`, `.ply`, `.level_label`, `.per_roll` (list[RollEquity]) |
| `RollEquity` | In LuckResult; input to `luck_from_equities()` | `.die1`, `.die2`, `.equity`, `.weight` (1 for doubles, 2 otherwise) |
| `Probabilities` | In all analysis types | `.win`, `.gammon_win`, `.backgammon_win`, `.gammon_loss`, `.backgammon_loss`, `.equity` |


## GNUbg Path

`neongammon.gnubg` (`GnuBgAnalyzer`, `post_move_analytics*`) runs the GNUbg CLI.
`neongammon.settings.gnubg_path()` picks the binary, first hit wins:

1. `settings.set_gnubg_path(path)` in code (this process), or per instance
   `GnuBgAnalyzer(gnubg_path=...)` / `post_move_analytics(..., gnubg_path=...)`
2. the `NEONGAMMON_GNUBG` environment variable
3. `"gnubg_path"` in the user settings file: `~/.config/neongammon/settings.json`
   (`$XDG_CONFIG_HOME` respected), or wherever `NEONGAMMON_SETTINGS` points
   (the pre-rename `BGSAGE_GNUBG` / `BGSAGE_SETTINGS` and
   `~/.config/bgsage/settings.json` are still read when the new ones are absent)
4. auto-detection: `gnubg` on PATH, then `/opt/homebrew/bin/gnubg` and
   `/usr/local/bin/gnubg`

```python
from neongammon import settings
settings.set_gnubg_path("/opt/homebrew/bin/gnubg", persist=True)  # saved for later sessions
settings.set_gnubg_path(None, persist=True)                       # back to auto-detection
```

`scripts/score_benchmark_pr_gnubg.py` and `scripts/test_match_gnubg_comparison.py`
resolve the binary the same way.


## Rollout Python Examples

See [docs/rollout.md](rollout.md) for the full rollout specification, `RolloutConfig` / `TrialEvalConfig` reference, and `truncated1`/`truncated2`/`truncated3` presets.

```python
from neongammon import BgBotAnalyzer

# XGRoller equivalent
analyzer = BgBotAnalyzer(eval_level="rollout",
    n_trials=42, truncation_depth=5, decision_ply=1)

# XGRoller+ equivalent
analyzer = BgBotAnalyzer(eval_level="rollout",
    n_trials=360, truncation_depth=7, decision_ply=2,
    late_ply=1, late_threshold=2)

# XGRoller++ Checker equivalent
analyzer = BgBotAnalyzer(eval_level="rollout",
    n_trials=360, truncation_depth=5, decision_ply=3,
    late_ply=2, late_threshold=2)

# XGRoller++ Cube equivalent
analyzer = BgBotAnalyzer(eval_level="rollout",
    n_trials=360, truncation_depth=7, decision_ply=3,
    late_ply=2, late_threshold=2)
```


### Low-Level `bgbot_cpp` Rollout API (`TrialEvalConfig` & Progress Callback)

> Prefer `BgBotAnalyzer(eval_level="rollout", checker=..., cube=..., target_se=...)`
> for model-independent rollouts (including the 24-NN `stage11` production model).
> The low-level `bgbot_cpp.cube_decision_rollout` and `create_rollout_5nn` helpers
> below accept a 5-NN `WeightConfig` (10 unpacked args) — use `WeightConfig.from_model("stage5")` from the Git checkout,
> not `WeightConfig.default()` (which returns a 24-NN `WeightConfigPair`).

```python
import bgbot_cpp
from neongammon.weights import WeightConfig

w = WeightConfig.from_model("stage5")

# Full rollout, 3-ply for both checker and cube, no ply reductions
# IMPORTANT: C++ defaults ultra_late_threshold=2 (drops to 1-ply at move 2+);
# pass ultra_late_threshold=9999 for accurate full rollouts (BgBotAnalyzer defaults to 9999).
result = bgbot_cpp.cube_decision_rollout(
    checkers=board,
    *w.weight_args[:5],   # 5 weight paths
    **dict(zip(['n_hidden_purerace','n_hidden_racing','n_hidden_attacking',
                'n_hidden_priming','n_hidden_anchoring'], w.hidden_sizes)),
    n_trials=1296, truncation_depth=0, decision_ply=1,
    n_threads=16, enable_vr=True,
    checker=bgbot_cpp.TrialEvalConfig(ply=3),
    cube=bgbot_cpp.TrialEvalConfig(ply=3),
    ultra_late_threshold=9999,
    progress=lambda done, total: print(f"{done}/{total}"),
)
# result: dict with equity_nd, equity_nd_se, equity_dt, equity_dt_se,
#   cubeless_equity, cubeless_se, probs, prob_std_errors, ...

# Full rollout, 3-ply checker, 1T cube (XG Roller-style cube decisions)
result = bgbot_cpp.cube_decision_rollout(
    checkers=board, *w.weight_args[:5],
    **dict(zip(['n_hidden_purerace','n_hidden_racing','n_hidden_attacking',
                'n_hidden_priming','n_hidden_anchoring'], w.hidden_sizes)),
    n_trials=1296, truncation_depth=0, decision_ply=1,
    n_threads=16, enable_vr=True,
    checker=bgbot_cpp.TrialEvalConfig(ply=3),
    cube=bgbot_cpp.TrialEvalConfig(rollout_trials=42, rollout_depth=5, rollout_ply=1),
    ultra_late_threshold=9999,
)

# Strategy object approach (reusable across positions)
rollout = bgbot_cpp.create_rollout_5nn(
    *w.weight_args[:5],
    **dict(zip(['n_hidden_purerace','n_hidden_racing','n_hidden_attacking',
                'n_hidden_priming','n_hidden_anchoring'], w.hidden_sizes)),
    n_trials=1296, truncation_depth=0, decision_ply=1,
    n_threads=16, enable_vr=True,
    checker=bgbot_cpp.TrialEvalConfig(ply=3),
    cube=bgbot_cpp.TrialEvalConfig(ply=3),
    ultra_late_threshold=9999,
)
result = rollout.cube_decision(checkers=board, cube_value=1,
    owner=bgbot_cpp.CubeOwner.CENTERED)
```

**`ultra_late_threshold` guidance:**
- `BgBotAnalyzer` defaults to `ultra_late_threshold=9999` (no late 1-ply drop); C++ `RolloutConfig` defaults to `2` (fast for short truncated rollouts).
- Set `ultra_late_threshold=9999` whenever running full rollouts (`truncation_depth=0`) directly through C++ with N-ply or rollout evaluators.

**Adaptive stopping (`target_se` / `max_batches`):** `BgBotAnalyzer(eval_level="rollout", target_se=0.01, max_batches=50)` runs batches until standard error reaches `target_se` (or `max_batches` is hit); `target_se=0.0` (default) runs a fixed `n_trials`.

**Progress callback**: Both `cube_decision_rollout()` and `rollout.cube_decision()`
accept `progress=callable` — called with `(completed, total)` periodically during
execution. Useful for progress bars in UI. The callback is called from worker threads
with the GIL automatically acquired.

## Glossary

- **ER**: Error Rate — mean equity loss per decision vs GNUbg best, millipips (x1000)
- **PR**: Performance Rating - equal to ER / 2 (total error / # of decisions * 500)
- **ppg**: Points per game
- **PubEval**: Tesauro's linear evaluator, standard weak opponent
- **TD(0)**: Temporal Difference learning (no eligibility trace)
- **gpw**: Game plan weight — gradient multiplier for matching positions in SL
- **TINY filter**: Default move filter (5 moves, 0.08 threshold)
- **VR**: Variance Reduction — luck-tracking for rollout noise reduction (always 1-ply, decoupled from decision ply). Internal to rollouts; distinct from the app-facing per-roll **Luck** metric.
- **Luck** (per-roll): How lucky an actual roll was, in equity units — `actual_equity - average_equity` from the roller's perspective (see [§7 Luck](#7-luck-per-roll-dice-fortune)). Positive = lucky, averages to zero over many rolls. Not the same as VR.
- **BMI**: Best Move Index — the core function that selects the best move from legal candidates (1-ply score + filter + N-ply rescore)
- **Move0Cache/Move1Cache**: Pre-computed move decisions for the first/second half-moves of a rollout trial, shared across all trials
- **SharedPosCache**: Lock-free cross-thread N-ply position evaluation cache for rollout trials
- **Jacoby rule**: Optional money game rule — gammons/backgammons count as single while cube is centered. Default on in Python API, auto-disabled for match play. `CubeInfo::jacoby_active()` checks `jacoby && is_money() && owner == CENTERED`.
- **Beaver rule**: Optional money game rule — opponent can redouble while retaining cube after being doubled. Punishes bad doubles (DT < 0). DB = 2*DT. Default on in Python API, auto-disabled for match play.
- **Janowski**: Cubeless-to-cubeful equity interpolation
- **ND/DT/DP/DB**: No Double / Double-Take / Double-Pass / Double-Beaver
- **MET**: Match Equity Table — lookup table of match-winning probabilities at each score
- **MWC**: Match Winning Chance — probability of winning the match from a given score
- **Crawford**: First game after a player reaches 1-away; no doubling allowed
- **MatchInfo**: `{away1, away2, is_crawford}` — match state; `{0,0,false}` = money game
