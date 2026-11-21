# Bearoff Database

One-sided bearoff database for exact evaluation of endgame positions where all
checkers are in the home board (or borne off). Covers 54,264 positions
(C(21,6) = 15 checkers on 6 points). File: `data/bearoff_1sided.db` (~4.7 MB).

## What It Stores (Per Position)

- **Bearoff distribution** (32 × uint16): P(all checkers borne off in exactly k rolls)
- **Mean rolls** (float32): expected rolls to bear off all (for EPC)
- **Gammon distribution** (32 × uint16, only for all-15-on-board positions):
  P(0 checkers borne off after k rolls under optimal play)

## Position Indexing

Combinatorial number system (stars-and-bars). A 6-element checker count array
`[c1..c6]` maps to a unique index in `[0, 54263]`. O(1) arithmetic, no hash tables.

## Two-Sided Probability Computation

Combines two one-sided distributions for exact cubeless probabilities:
- `P(win) = Σ P_player[i] × (1 - CDF_opponent[i-1])` (player-on-roll advantage)
- `P(gammon_win) = Σ P_player[i] × ZeroOff_opponent[i-1]`
- `P(backgammon) = 0` (impossible in home-board bearoff)

## `is_bearoff` Check

Returns true when both bars are empty, the player's checkers are only on points 1–6,
and the opponent's checkers are only on points 19–24 (~10 comparisons, negligible cost).

## Integration with Multi-Ply and Rollout

Both `MultiPlyStrategy` and `RolloutStrategy` accept an optional `BearoffDB*`
via `set_bearoff_db()`. When set:
- **Multi-ply:** bearoff positions short-circuit recursion and return exact probabilities.
- **Rollout:** bearoff input positions skip all trials (SE=0); truncation uses the DB.

The `BearoffStrategy` wrapper intercepts 1-ply leaf evaluations (variance reduction and
move selection).

## EPC (Effective Pip Count)

`BearoffDB::lookup_epc(board, player)` returns `mean_rolls × (49/6)`.
`mean_rolls` includes the upcoming roll (1 checker on point 1 → `mean=1.0` → `EPC=8.167`).

## Python API

The database is loaded once per process via `neongammon.analyzer.load_bearoff_db()` and
shared across `BgBotAnalyzer` instances and `BatchModels`.

```python
from neongammon import BgBotAnalyzer, BatchModels

# BearoffDB auto-loaded from data/ directory (bearoff_db=True by default)
analyzer = BgBotAnalyzer(eval_level="3ply")

# EPC for one side (0 = player on roll, 1 = opponent); returns float or None
epc = analyzer.epc(board, player=0)

# Disable bearoff DB (also supported on BatchModels and batch_* functions)
analyzer = BgBotAnalyzer(eval_level="1ply", bearoff_db=False)
models = BatchModels(bearoff_db=False)
```

## C++ API

```cpp
#include "bgbot/bearoff.h"
BearoffDB db;
db.load("data/bearoff_1sided.db");

if (db.is_bearoff(board)) {
    auto probs = db.lookup_probs(board);      // exact cubeless probs
    float epc = db.lookup_epc(board, 0);      // EPC for player on roll
}

// Wrap base strategy for automatic bearoff interception
auto base = std::make_shared<GamePlanStrategy>(...);
auto bearoff_strat = std::make_shared<BearoffStrategy>(base, &db);

// Set on multi-ply/rollout for deeper integration
multi_ply.set_bearoff_db(&db);
rollout.set_bearoff_db(&db);
```

## Generation

```bash
python scripts/generate_bearoff_db.py  # ~4 seconds, outputs data/bearoff_1sided.db
```
