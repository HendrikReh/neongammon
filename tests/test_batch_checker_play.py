# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Parallel checker-play parity on legal positions without an external corpus."""
import unittest

from neongammon import BgBotAnalyzer, batch_checker_play
from neongammon.weights import default_weights
from .batch_positions import checker_positions


class TestBatchCheckerPlay(unittest.TestCase):
    """Verify batch_checker_play matches serial BgBotAnalyzer.checker_play."""

    @classmethod
    def setUpClass(cls):
        cls.positions = checker_positions(50, seed=54321)
        cls.weights = default_weights()
        cls.weights.validate()

    def test_invalid_eval_level_rejected(self):
        with self.assertRaises(ValueError):
            batch_checker_play(self.positions[:1], eval_level="0ply",
                               weights=self.weights)
        with self.assertRaises(ValueError):
            batch_checker_play(self.positions[:1], eval_level="5ply",
                               weights=self.weights)

    def test_search_matches_the_entire_scalar_ranking(self):
        for level in ("1ply", "2ply"):
            analyzer = BgBotAnalyzer(eval_level=level, parallel_threads=1)
            positions = self.positions[:10]
            results = batch_checker_play(positions, eval_level=level, n_threads=2)
            self.assertEqual(len(results), len(positions))
            for position, result in zip(positions, results):
                with self.subTest(level=level, position=position):
                    expected = analyzer.checker_play(**position)
                    self.assertEqual(result, expected)

    def test_1ply_parallel_matches_serial(self):
        """batch_checker_play at 1-ply: parallel vs n_threads=1."""
        positions = self.positions[:10]
        serial = batch_checker_play(
            positions, eval_level="1ply",
            weights=self.weights, n_threads=1,
        )
        parallel = batch_checker_play(
            positions, eval_level="1ply",
            weights=self.weights, n_threads=0,
        )

        self.assertEqual(len(serial), len(parallel))
        for i, (s, p) in enumerate(zip(serial, parallel)):
            with self.subTest(position=i):
                if s.moves and p.moves:
                    self.assertAlmostEqual(
                        s.moves[0].equity, p.moves[0].equity, places=4,
                    )
                    self.assertEqual(s.moves[0].board, p.moves[0].board)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_input(self):
        """batch_checker_play with empty list returns empty list."""
        results = batch_checker_play([], eval_level="1ply", weights=self.weights)
        self.assertEqual(results, [])

    def test_result_structure(self):
        """Verify CheckerPlayResult fields are populated correctly."""
        pos = self.positions[0]
        results = batch_checker_play(
            [pos], eval_level="1ply", weights=self.weights,
        )
        self.assertEqual(len(results), 1)
        r = results[0]

        # CheckerPlayResult fields
        self.assertEqual(r.board, list(pos["board"]))
        self.assertEqual(r.die1, pos["die1"])
        self.assertEqual(r.die2, pos["die2"])
        self.assertEqual(r.eval_level, "1-ply")

        # MoveAnalysis fields
        self.assertGreater(len(r.moves), 0)
        m = r.moves[0]
        self.assertEqual(len(m.board), 26)
        self.assertIsInstance(m.equity, float)
        self.assertIsInstance(m.cubeless_equity, float)
        self.assertAlmostEqual(m.equity_diff, 0.0, places=6)
        self.assertEqual(m.eval_level, "1-ply")
        self.assertIsNotNone(m.probs)
        self.assertGreaterEqual(m.probs.win, 0.0)
        self.assertLessEqual(m.probs.win, 1.0)

    def test_moves_sorted_descending(self):
        """Moves are sorted by equity descending (best first)."""
        results = batch_checker_play(
            self.positions[:20], eval_level="1ply", weights=self.weights,
        )
        for i, r in enumerate(results):
            with self.subTest(position=i):
                for j in range(len(r.moves) - 1):
                    self.assertGreaterEqual(
                        r.moves[j].equity, r.moves[j + 1].equity,
                        msg=f"pos {i}: moves not sorted at index {j}",
                    )


if __name__ == "__main__":
    unittest.main()
