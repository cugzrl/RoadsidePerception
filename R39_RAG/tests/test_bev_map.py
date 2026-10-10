"""车道级 BEV 底图能覆盖路口中心并被流水线脚本调用"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.bev_map import BEVBasemap, LaneMap, build_road_mask, make_r39_view


class BevMapTest(unittest.TestCase):
    def test_lane_map_covers_intersection(self):
        graph = LaneMap.load()
        view, directions = make_r39_view(graph, 400, 400, 90.0)
        pixel = view.to_px(graph.center)[0]
        self.assertAlmostEqual(float(pixel[0]), 200.0, delta=1.5)
        self.assertAlmostEqual(float(pixel[1]), 200.0, delta=1.5)
        self.assertEqual(set(directions), {"N", "E", "S", "W"})
        image = BEVBasemap(graph, view).render()
        self.assertEqual(image.shape, (400, 400, 3))
        self.assertGreater(int(image.mean()), 22)
        mask = build_road_mask(graph, dilate_m=2.0)
        entry = next(node for node in graph.nodes.values() if node.get("role") == "entry")
        centerline = np.asarray(entry["centerline"], dtype=np.float64)
        self.assertTrue(bool(mask.contains(centerline[len(centerline) // 2])[0]))

    def test_visualize_bev_writes_r39_still(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = {
                "scene_id": "R39_smoke",
                "fps": 20.0,
                "cameras": [{"camera_id": "R39_Aw_CamS"}],
                "frames": [
                    {
                        "camera_id": "R39_Aw_CamS",
                        "frame_index": index,
                        "detections": [
                            {
                                "global_track_id": 1,
                                "world_position": {"x": -20.0 + index, "y": -26.0},
                            }
                        ],
                    }
                    for index in range(4)
                ],
            }
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "visualize_bev.py"),
                    "--result",
                    str(result_path),
                    "--output",
                    str(root / "bev.mp4"),
                    "--still",
                    str(root / "bev_map.png"),
                    "--width",
                    "320",
                    "--height",
                    "320",
                    "--still-size",
                    "320",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertGreater((root / "bev_map.png").stat().st_size, 1000)
            self.assertGreater((root / "bev.mp4").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
