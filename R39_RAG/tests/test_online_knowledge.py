"""在线前缀检索和知识加成"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "algorithms" / "stmc" / "src" / "tracker"))

from r39_rag.knowledge.online import LaneIndex, link_decision, match_prefix
from r39_rag.knowledge.trajectory import resample_path
from knowledge_prior import KnowledgePrior


def _line(start: float, end: float) -> np.ndarray:
    xs = np.linspace(start, end, 16)
    return np.stack([xs, np.zeros(16)], axis=1)


class PrefixTest(unittest.TestCase):
    def test_middle_prefix_matches_progress_not_endpoint(self):
        prototype = _line(0.0, 40.0)
        prefix = np.stack([np.linspace(12.0, 20.0, 6), np.zeros(6)], axis=1)
        matches = match_prefix(prefix, prototype.reshape(1, 16, 2), sigma_m=8.0, top_k=1)
        self.assertEqual(len(matches), 1)
        self.assertGreater(matches[0]["score"], 0.45)
        self.assertGreater(matches[0]["progress"], 0.2)
        self.assertLess(matches[0]["progress"], 0.8)
        continuation = np.asarray(matches[0]["continuation"])
        self.assertGreater(float(continuation[:, 0].mean()), 20.0)

    def test_short_history_has_no_reliable_knowledge(self):
        decision = link_decision(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.array([2.0, 0.0]),
            _line(0.0, 40.0).reshape(1, 16, 2),
            None,
            None,
            sigma_m=8.0,
            top_k=3,
            min_score=0.45,
            continuation_m=8.0,
            min_points=3,
        )
        self.assertFalse(decision["reliable"])
        self.assertEqual(decision["action"], "none")

    def test_continuation_cannot_flip_a_negative_edge(self):
        prototype = resample_path(np.stack([np.linspace(0, 40, 8), np.zeros(8)], axis=1), 16)
        history = np.stack([np.linspace(10, 18, 5), np.zeros(5)], axis=1)
        decision = link_decision(
            history,
            np.array([22.0, 0.5]),
            prototype.reshape(1, 16, 2),
            None,
            None,
            sigma_m=8.0,
            top_k=3,
            min_score=0.35,
            continuation_m=8.0,
            min_points=3,
            frame_gap=4,
            prototype_supports=np.array([8.0]),
            prototype_speeds=np.array([6.0]),
        )
        self.assertEqual(decision["action"], "continuation")
        dummy = type("Cfg", (), {"weight": 0.2})()
        delta = KnowledgePrior._delta(dummy, decision, base_weight=-0.8)
        self.assertGreater(delta, 0.0)
        self.assertLessEqual(-0.8 + delta, 0.0)
        self.assertLessEqual(delta, 0.12)

    def test_offroad_and_static_jump_get_no_bonus(self):
        prototype = _line(0.0, 40.0)
        history = np.stack([np.linspace(0, 8, 5), np.zeros(5)], axis=1)
        offroad = link_decision(
            history,
            np.array([280.0, 40.0]),
            prototype.reshape(1, 16, 2),
            LaneIndex.load(),
            None,
            sigma_m=8.0,
            top_k=1,
            min_score=0.2,
            continuation_m=8.0,
            min_points=3,
            frame_gap=4,
        )
        self.assertFalse(offroad["reliable"])
        static = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.05, 0.0]])
        jumped = link_decision(
            static,
            np.array([8.0, 0.0]),
            np.zeros((0, 16, 2)),
            None,
            None,
            sigma_m=8.0,
            top_k=1,
            min_score=0.45,
            continuation_m=8.0,
            min_points=3,
            frame_gap=10,
        )
        self.assertNotEqual(jumped["action"], "hold")
        self.assertFalse(jumped["reliable"])

    def test_nearby_reappearance_holds_without_forcing_strong_rejection(self):
        history = np.stack([np.linspace(0, 2, 5), np.zeros(5)], axis=1)
        decision = link_decision(
            history,
            np.array([2.4, 0.2]),
            np.zeros((0, 16, 2)),
            None,
            None,
            sigma_m=8.0,
            top_k=3,
            min_score=0.45,
            continuation_m=8.0,
            min_points=3,
            frame_gap=9,
        )
        self.assertEqual(decision["action"], "hold")
        dummy = type("Cfg", (), {"weight": 0.2})()
        soft = KnowledgePrior._delta(dummy, decision, base_weight=-0.03)
        self.assertGreater(soft, 0.0)
        self.assertLessEqual(-0.03 + soft, 0.0)
        self.assertEqual(KnowledgePrior._delta(dummy, decision, base_weight=-0.8), 0.0)

    def test_low_trust_does_not_emit_penalty(self):
        prototype = _line(0.0, 40.0)
        history = np.stack([np.linspace(20, 10, 5), np.zeros(5)], axis=1)
        decision = link_decision(
            history,
            np.array([0.0, 30.0]),
            prototype.reshape(1, 16, 2),
            None,
            None,
            sigma_m=8.0,
            top_k=1,
            min_score=0.2,
            continuation_m=8.0,
            min_points=3,
        )
        self.assertNotEqual(decision["action"], "contradict")


if __name__ == "__main__":
    unittest.main()
