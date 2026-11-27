# SPDX-License-Identifier: MPL-2.0
"""Model sharing and bearoff database lifetime across public analysis interfaces."""

import os
import subprocess
import sys
import textwrap
from dataclasses import asdict
from unittest.mock import patch

import pytest

import bgbot_cpp

import neongammon.analyzer as A
from neongammon import (
    BatchModels, BgBotAnalyzer, STARTING_BOARD, batch_checker_play, batch_evaluate,
    batch_post_move_evaluate,
)
from neongammon.weights import WeightConfig, bearoff_db_path

FACTORIES = ("create_strategy", "create_multipy", "create_multipy_from_base",
             "create_multipy_5nn", "GamePlanStrategy", "BearoffDB")
SNAKE = [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 3, 3, 2, 2, 1, -7, -7, 0]
BEAROFF = [0, 3, 3, 2, 3, 2, 2] + [0] * 12 + [-2, -3, -2, -3, -2, -2] + [0]
CONTACT = [0, 0, 0, 0, 2, 2, 4, -2, 2, 0, 2, 0, -2, 3, -1, 0, -1, -2, 0, -2, -3, -2, 0, 0, 0, 0]


class _Count:
    def __enter__(self):
        self._patches = [patch.object(bgbot_cpp, n, wraps=getattr(bgbot_cpp, n)) for n in FACTORIES]
        self.mocks = {n: p.start() for n, p in zip(FACTORIES, self._patches)}
        A.load_bearoff_db.cache_clear()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()

    def take(self):
        out = {n: m.call_count for n, m in self.mocks.items() if m.call_count}
        for m in self.mocks.values():
            m.reset_mock()
        return out


def test_loads_at_each_stage_of_a_three_ply_analyzer():
    with _Count() as c:
        a = BgBotAnalyzer(eval_level="3ply", parallel_threads=2)
        assert c.take() == {"create_strategy": 1, "create_multipy_from_base": 1, "BearoffDB": 1}
        a.checker_play(STARTING_BOARD, 3, 1)
        assert c.take() == {}
        a.cube_action(STARTING_BOARD)
        assert c.take() == {}
        a.cube_action(CONTACT)
        a.checker_play(CONTACT, 6, 2)
        assert c.take() == {}
        BgBotAnalyzer(eval_level="2ply", parallel_threads=2)
        assert c.take() == {"create_strategy": 1, "create_multipy_from_base": 1}  # no DB reload


@pytest.mark.parametrize("fn", [batch_evaluate, batch_post_move_evaluate, batch_checker_play])
def test_an_empty_batch_loads_no_model_and_no_database(fn):
    with _Count() as c:
        assert fn([], eval_level="2ply") == []
        assert c.take() == {}
    with pytest.raises(ValueError):
        fn([], eval_level="5ply")
    with pytest.raises(ValueError):
        fn([], models=BatchModels(), bearoff_db=False)


def _old_nply(analyzer):
    """Rebuild the analyzer's N-ply strategy the old way: its own base."""
    inner = analyzer._analyzer._inner
    w = inner._weights
    s = bgbot_cpp.create_multipy(w.strategy_type, w.weight_paths_list, w.hidden_sizes_list,
                                 n_plies=inner._n_plies, parallel_evaluate=True,
                                 parallel_threads=inner._parallel_threads)
    if inner._bearoff_db is not None:
        bgbot_cpp.multipy_set_bearoff_db(s, inner._bearoff_db)
    inner._strategy_nply = s


def _results(a):
    out = []
    for board, d in ((CONTACT, (6, 2)), (SNAKE, (6, 5)), (BEAROFF, (4, 2))):
        out.append([asdict(m) for m in a.checker_play(board, *d).moves])
        out.append(asdict(a.cube_action(board)))
        out.append(asdict(a.post_move_analytics(board, cube_owner="centered")))
    return out


@pytest.mark.parametrize("model,bearoff_db", [
    ("stage11", True), ("stage11", False), ("stage5", True), ("stage11", "path"),
])
@pytest.mark.parametrize("level", ["2ply", "3ply"])
def test_results_equal_the_old_construction(model, bearoff_db, level):
    w = WeightConfig.from_model(model) if model != "stage11" else None
    db = bearoff_db_path() if bearoff_db == "path" else bearoff_db
    new = BgBotAnalyzer(weights=w, eval_level=level, parallel_threads=2, bearoff_db=db)
    old = BgBotAnalyzer(weights=w, eval_level=level, parallel_threads=2, bearoff_db=db)
    _old_nply(old)
    assert _results(new) == _results(old)

def test_all_public_consumers_share_one_model():
    from neongammon import Difficulty, DifficultyPolicy, batch_cube_action_nply

    with (patch.object(bgbot_cpp, "create_strategy", wraps=bgbot_cpp.create_strategy) as load,
          patch.object(bgbot_cpp, "create_rollout", side_effect=AssertionError("duplicate weights"))):
        checker = BgBotAnalyzer(eval_level="3ply", parallel_threads=1)
        models = BatchModels.from_analyzer(checker)
        cube = BgBotAnalyzer(eval_level="2ply", models=models, parallel_threads=1)
        rollout = BgBotAnalyzer(eval_level="truncated1", models=models, parallel_threads=1)
        assert BatchModels.from_analyzer(rollout) is models
        cube.cube_decision(CONTACT)
        batch_checker_play([dict(board=CONTACT, die1=6, die2=2)], models=models)
        batch_cube_action_nply([dict(board=CONTACT)], models=models)
        weak = BgBotAnalyzer(models=models)
        policies = [DifficultyPolicy(d, analyzer=checker if d == Difficulty.FULL_STRENGTH else weak)
                    for d in Difficulty]
        assert len(policies) == 4
        assert load.call_count == 1

@pytest.mark.skipif(sys.platform != "darwin", reason="macOS process RSS")
def test_four_shared_presets_add_less_than_five_mib():
    # Measure in a fresh process, before any search populates worker caches.
    probe = textwrap.dedent("""
        import gc, os, subprocess
        from neongammon import BatchModels, BgBotAnalyzer, Difficulty, DifficultyPolicy

        def resident_kib():
            gc.collect()
            return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]))

        one = BgBotAnalyzer(parallel_threads=1)
        models = BatchModels.from_analyzer(one)
        before = resident_kib()
        three = BgBotAnalyzer(models=models, eval_level="3ply", parallel_threads=1)
        policies = [DifficultyPolicy(d, analyzer=three if d == Difficulty.FULL_STRENGTH else one)
                    for d in Difficulty]
        print(resident_kib() - before)
    """)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    kib = int(subprocess.check_output([sys.executable, "-c", probe], env=env))
    assert kib < 5 * 1024
