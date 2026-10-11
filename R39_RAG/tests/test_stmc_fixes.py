"""STMC 位置槽位和短时重关联 不启动 Multicut"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "algorithms" / "stmc"))

from src.tracker.supertrack import SuperTrack  # noqa: E402
from src.tracker.tracker import Tracker  # noqa: E402


class Bag:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        return getattr(self, key, default)


def make_tracker(active: bool, recovery: bool) -> Tracker:
    tracker_cfg = Bag(
        fdim=4,
        knowledge=Bag(enabled=False),
        position_state=Bag(active_cameras_only=active),
        short_recovery=Bag(
            enabled=recovery,
            max_gap=3,
            iou_bonus=0.60,
            match_iou=0.50,
            reject_iou=0.30,
        ),
        matching=Bag(reid_decay=0.7, rescale_threshold=0.7, distance_threshold=8.0, rescale_weight=0.5, distance_weight=0.5),
        prematching=Bag(enabled=False, iou_bias=0.6),
    )
    return Tracker(solver_opts=None, cfg=Bag(tracker=tracker_cfg), n_cams=4, device="cpu")


def track(frame, cam, xy, box, label):
    features = torch.full((4, 4), float("nan"))
    boxes = torch.full((4, 4), float("nan"))
    pos2 = torch.full((4, 2), float("nan"))
    pos3 = torch.full((4, 2), float("nan"))
    features[cam] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    boxes[cam] = torch.tensor(box, dtype=torch.float32)
    pos3[cam] = torch.tensor(xy, dtype=torch.float32)
    item = SuperTrack(frame=frame, features=features, boxes=boxes, positions_2d=pos2, positions_3d=pos3, class_id=0)
    item.activate()
    item.set_label(label)
    return item


class PositionAndRecoveryTest(unittest.TestCase):
    def test_inactive_cameras_do_not_inflate_distance(self):
        tracker = make_tracker(True, False)
        older = track(10, 0, [-15.0, -65.0], [570, 200, 60, 50], 2)
        older.positions_3d[1] = torch.tensor([-20.0, -37.0])
        older.track_where[1] = False
        newer = track(11, 0, [-15.2, -64.9], [575, 202, 60, 50], 9)
        newer.label = None
        features, positions, _boxes, where = tracker._association_inputs([older, newer])
        appearance, position, _iou = tracker._compute_similarities(features, positions, _boxes)
        appearance, position = tracker._fill_unshared_cameras(appearance, position, features, positions, where)
        self.assertLess(float(position[0, 1]), 1.0)

    def test_default_still_averages_inactive_slots(self):
        tracker = make_tracker(False, False)
        self.assertFalse(SuperTrack.use_active_positions)
        older = track(10, 0, [-15.0, -65.0], [570, 200, 60, 50], 2)
        older.positions_3d[1] = torch.tensor([-20.0, -37.0])
        older.features[1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        older.track_where[1] = False
        mean = older.mean_positions_3d
        self.assertAlmostEqual(float(mean[1]), (-65.0 + -37.0) / 2, places=3)

    def test_short_gap_iou_lifts_and_rejects_neighbor(self):
        tracker = make_tracker(True, True)
        tracker.frame = 12
        older = track(10, 0, [-43.0, 18.0], [228.0, 201.0, 88.0, 52.0], 18)
        good = track(12, 0, [-42.7, 17.3], [226.0, 201.0, 88.0, 54.0], 99)
        good.label = None
        bad = track(12, 0, [-39.8, 15.9], [290.0, 207.0, 82.0, 52.0], 98)
        bad.label = None
        adj = torch.tensor([[0.0, 0.2, 0.2], [0.2, 0.0, -100.0], [0.2, -100.0, 0.0]])
        updated = tracker._apply_short_recovery(adj, [older, good, bad])
        self.assertGreater(float(updated[0, 1]), 0.7)
        self.assertEqual(float(updated[0, 2]), 0.0)

    def test_consecutive_frame_does_not_stack_recovery_on_prematch(self):
        tracker = make_tracker(True, True)
        tracker.frame = 11
        older = track(10, 0, [-15.0, -65.0], [570.0, 200.0, 60.0, 50.0], 2)
        newer = track(11, 0, [-15.2, -64.9], [572.0, 201.0, 60.0, 50.0], 9)
        newer.label = None
        adj = torch.tensor([[0.0, 0.4], [0.4, 0.0]])
        updated = tracker._apply_short_recovery(adj, [older, newer])
        self.assertAlmostEqual(float(updated[0, 1]), 0.4, places=5)

    def test_uncomputable_edges_do_not_keep_nan(self):
        tracker = make_tracker(True, False)
        older = track(10, 0, [-15.0, -65.0], [570, 200, 60, 50], 2)
        newer = track(11, 0, [-15.2, -64.9], [575, 202, 60, 50], 9)
        adj = torch.full((2, 2), float("nan"))
        finalized = tracker._finalize_adjacency_matrix(adj, -100.0, [older, newer])
        self.assertFalse(bool(torch.isnan(finalized).any()))
        self.assertEqual(float(finalized[1, 0]), 0.0)
        self.assertEqual(float(finalized[0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
