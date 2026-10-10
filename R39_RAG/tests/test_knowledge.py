"""投影约定与网格规则"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r39_rag.config import load_knowledge
from r39_rag.knowledge.spatial import cell_index, confidence_label
from r39_rag.knowledge.trajectory import path_kind, resample_path
from r39_rag.projection.homography import GroundProjector, bottom_center, load_projectors
from r39_rag.projection.quality import project_observation


class ProjectionTest(unittest.TestCase):
    def test_bottom_center_matches_tlwh_bottom(self):
        self.assertEqual(bottom_center([10, 20, 30, 50]), (20.0, 50.0))

    def test_image_to_world_uses_inverse(self):
        config = load_knowledge()
        projector = load_projectors(config["calibration_dir"], ["R39_Aw_CamS"], float(config["min_denominator"]))["R39_Aw_CamS"]
        manual = np.linalg.inv(projector.image_from_world) @ np.array([960.0, 540.0, 1.0])
        expected = manual[:2] / manual[2]
        world, singular = projector.image_to_world(np.array([[960.0, 540.0]]))
        self.assertFalse(bool(singular[0]))
        self.assertTrue(np.allclose(world[0], expected))

    def test_border_and_far_points_are_excluded(self):
        config = load_knowledge()
        border = project_observation(
            {"bbox_xyxy": [100, 100, 140, 1078], "quality_flags": []},
            GroundProjector(np.eye(3), 1e-6),
            config,
        )
        self.assertEqual(border["status"], "border_contact")
        far = project_observation(
            {"bbox_xyxy": [1700, 40, 1900, 120], "quality_flags": []},
            load_projectors(config["calibration_dir"], ["R39_Aw_CamS"], 1e-6)["R39_Aw_CamS"],
            config,
        )
        self.assertEqual(far["status"], "out_of_bounds")

    def test_straight_path_and_cell(self):
        config = load_knowledge()
        points = np.stack([np.linspace(0, 20, 12), np.zeros(12)], axis=1)
        self.assertEqual(path_kind(resample_path(points, 16), config), "straight")
        self.assertEqual(cell_index(0, 0, config), (30, 30))
        self.assertIsNone(cell_index(80, 0, config))
        self.assertEqual(confidence_label(10), "high")
        self.assertEqual(confidence_label(0), "unknown")

    def test_holdout_sequences_stay_out_of_the_library(self):
        config = load_knowledge()
        self.assertTrue(set(config["build_sequences"]).isdisjoint(config["holdout_sequences"]))
        self.assertIn("r39_20251021_070428", config["holdout_sequences"])


if __name__ == "__main__":
    unittest.main()
