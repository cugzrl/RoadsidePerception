"""小样本转换、时间基准和软链接测试"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r39_rag.cleaning.categories import apply_category
from r39_rag.cleaning.convert import (
    build_timebase,
    convert_aligned,
    convert_pipeline_result,
    convert_tracks,
    stamp_frame,
)
from r39_rag.config import load_categories
from r39_rag.links import ensure_link
from r39_rag.utils.json_stream import dump_json, iter_json_array


def _timebase() -> dict:
    return {
        "usable": True,
        "source_fps": 60.0,
        "sample_fps": 20.0,
        "frame_stride": 3,
        "frame_count": 1200,
        "image_width": 1920,
        "image_height": 1080,
        "timebase_source": "测试时间基准",
        "sync_status": "sampling_pattern_consistent_clock_not_verified",
    }


def _box(x0, y0, x1, y1, local_id, frame, gid=1, class_name="car"):
    return {
        "camera_key": "R39_Aw_CamS",
        "local_obj_id": local_id,
        "global_obj_id": gid,
        "frame_index": frame,
        "class_name": class_name,
        "label": f"{class_name}:{local_id}",
        "bbox_xyxy": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "pixel_centroid": {"x": (x0 + x1) / 2, "y": (y0 + y1) / 2},
    }


def _camera(record, length, width):
    item = dict(record)
    item.pop("bbox_xyxy")
    item.pop("pixel_centroid")
    item["camera_box_3d"] = {
        "bbox_3d": {
            "center": {"x": 10, "y": 4, "z": 0.75},
            "size": {"length": length, "width": width, "height": 1.5},
        }
    }
    return item


class CleaningTest(unittest.TestCase):
    def test_category_mapping_keeps_unknown_classes(self):
        mapping = load_categories()["mappings"]
        name, class_id, status, _flags = apply_category("car", mapping)
        self.assertEqual((name, class_id, status), ("Car", 0, "mapped"))
        name, class_id, status, flags = apply_category("motorcycle", mapping)
        self.assertEqual(name, "motorcycle")
        self.assertIsNone(class_id)
        self.assertIn("class_unmapped", flags)
        for raw in ("bicycle", "truck", "person"):
            mapped, _class_id, _status, _flags = apply_category(raw, mapping)
            if raw == "person":
                self.assertEqual(mapped, "Pedestrian")
            else:
                self.assertEqual(mapped, raw)

    def test_timebase_uses_stride_and_stays_empty_without_evidence(self):
        source_frame, timestamp, estimated, _source = stamp_frame(1, _timebase())
        self.assertEqual(source_frame, 3)
        self.assertEqual(timestamp, 50.0)
        self.assertTrue(estimated)
        source_frame, timestamp, estimated, source = stamp_frame(2, {"usable": False})
        self.assertIsNone(source_frame)
        self.assertIsNone(timestamp)
        self.assertFalse(estimated)
        self.assertEqual(source, "unavailable")

    def test_stream_array(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.json"
            dump_json(path, {"num_records": 2, "pixel_boxes": [{"frame_index": 0}, {"frame_index": 1}]})
            rows = list(iter_json_array(path, "pixel_boxes"))
            self.assertEqual([row["frame_index"] for row in rows], [0, 1])

    def test_aligned_dedup_collision_and_geometry(self):
        categories = load_categories()
        good = _box(10, 20, 40, 60, 1, 0)
        duplicate = json.loads(json.dumps(good))
        collision_a = _box(10, 20, 30, 40, 8, 1)
        collision_b = _box(100, 20, 140, 80, 8, 1)
        invalid = _box(5, 5, 5, 9, 3, 2)
        outside = _box(10, 10, 3000, 40, 4, 3)
        truck = _box(10, 10, 40, 40, 5, 4, class_name="truck")
        pixels = [good, duplicate, collision_a, collision_b, invalid, outside, truck]
        cameras = [
            _camera(good, 4.2, 1.8),
            _camera(duplicate, 4.2, 1.8),
            _camera(collision_a, 4.0, 1.7),
            _camera(collision_b, 80, 20),
            _camera(invalid, 4, 1.8),
            _camera(outside, 4, 1.8),
            _camera(truck, 6, 2),
        ]
        projected = {
            "num_output_records": len(pixels),
            "records": pixels,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bbox_info = root / "aligned_bbox_info.json"
            pixel_file = root / "aligned_pixel_bbox.json"
            camera_file = root / "aligned_camera_bbox.json"
            projected_file = root / "projected_pixel_bbox.json"
            dump_json(bbox_info, {"sequence_dir": "demo", "num_records": len(pixels), "pixel_boxes": pixels, "camera_boxes": cameras})
            dump_json(pixel_file, pixels)
            dump_json(camera_file, cameras)
            dump_json(projected_file, projected)
            output = root / "out"
            quarantine = root / "bad"
            stats = convert_aligned(
                site_id="R39",
                sequence_id="r39_demo_a",
                source_version="sam3_aligned",
                files={
                    "bbox_info": bbox_info,
                    "pixel_file": pixel_file,
                    "camera_file": camera_file,
                    "projected_file": projected_file,
                },
                categories=categories,
                timebase=_timebase(),
                output_dir=output,
                quarantine_dir=quarantine,
            )
            rows = [json.loads(line) for line in (output / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(stats["exact_duplicates_removed"], 1)
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row["score"] is None for row in rows))
            self.assertTrue(all(row["world_position"] is None for row in rows))
            self.assertTrue(all(row["sequence_id"] == "r39_demo_a" for row in rows))
            collision = [row for row in rows if row["local_track_id"] == 8]
            self.assertEqual(len(collision), 2)
            self.assertTrue(all("local_id_frame_collision" in row["quality_flags"] for row in collision))
            invalid_rows = [row for row in rows if row["local_track_id"] == 3]
            self.assertIn("invalid_bbox", invalid_rows[0]["quality_flags"])
            self.assertEqual(invalid_rows[0]["bbox_xyxy"], [5.0, 5.0, 5.0, 9.0])
            truck_row = next(row for row in rows if row["class_name_raw"] == "truck")
            self.assertEqual(truck_row["class_name"], "truck")
            self.assertIsNone(truck_row["class_id"])
            huge = next(row for row in collision if "geometry_size_implausible" in row["quality_flags"])
            self.assertEqual(huge["coordinate_frame"], "unknown")
            self.assertTrue((quarantine / "conflicts.jsonl").exists())
            second = convert_aligned(
                site_id="R39",
                sequence_id="r39_demo_a",
                source_version="sam3_aligned",
                files={
                    "bbox_info": bbox_info,
                    "pixel_file": pixel_file,
                    "camera_file": camera_file,
                    "projected_file": projected_file,
                },
                categories=categories,
                timebase=_timebase(),
                output_dir=output,
                quarantine_dir=quarantine,
            )
            self.assertEqual(second["written_observations"], stats["written_observations"])

    def test_manual_gid_does_not_drop_original(self):
        categories = load_categories()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            track = root / "tracks.json"
            auto = root / "auto.json"
            manual = root / "manual.json"
            dump_json(
                track,
                {
                    "camera_key": "R39_Aw_CamS",
                    "ground_z": 0.0,
                    "tracks": {
                        "1": {
                            "obj_id": 1,
                            "label": "car:1",
                            "class_name": "car",
                            "assumed_height_m": 1.5,
                            "frames": [
                                {
                                    "frame_index": 0,
                                    "bbox_xyxy": {"x0": 1, "y0": 2, "x1": 11, "y1": 22},
                                    "pixel_centroid": {"x": 6, "y": 10},
                                    "ground_point_3d": {"x": 3, "y": -20, "z": 0},
                                    "center_point_3d": {"x": 3, "y": -20, "z": 0.75},
                                }
                            ],
                        }
                    },
                },
            )
            assignment = {
                "camera_key": "R39_Aw_CamS",
                "local_obj_id": 1,
                "label": "car:1",
                "class_name": "car",
                "start_frame": 0,
                "end_frame": 0,
                "num_frames": 1,
            }
            dump_json(auto, {"tracks": [{**assignment, "global_obj_id": 7}]})
            dump_json(
                manual,
                {
                    "tracks": [
                        {
                            **assignment,
                            "global_obj_id": 9,
                            "original_global_obj_id": 7,
                            "is_deleted": False,
                        }
                    ]
                },
            )
            (root / "auto.txt").write_text("[R39_Aw_CamS]\nLID 1 -> GID 7 (car)\n", encoding="utf-8")
            (root / "manual.txt").write_text("[R39_Aw_CamS]\nLID 1 -> GID 9 (car)\n", encoding="utf-8")
            (root / "corrections.txt").write_text("R39_Aw_CamS,1,9,car,car:1\n", encoding="utf-8")
            output = root / "out"
            stats = convert_tracks(
                site_id="R39",
                sequence_id="r39_demo_b",
                source_version="sam3_manual_gid",
                files={
                    "tracks": {"R39_Aw_CamS": track},
                    "auto_global": auto,
                    "manual_global": manual,
                    "manual_corrections": root / "corrections.txt",
                    "lid_map": root / "auto.txt",
                    "lid_map_manual": root / "manual.txt",
                },
                categories=categories,
                timebase=_timebase(),
                output_dir=output,
                quarantine_dir=root / "bad",
            )
            row = json.loads((output / "observations.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["global_track_id"], 9)
            self.assertEqual(row["original_global_track_id"], 7)
            self.assertTrue(row["manual_fix"])
            self.assertIsNone(row["world_position"])
            self.assertEqual(row["coordinate_frame"], "unverified_assumed_ground_plane")
            self.assertEqual(stats["gid_changed"], 1)
            other = json.loads((output / "global_tracks.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(other["sequence_id"], "r39_demo_b")
            self.assertNotEqual(other["sequence_id"], "r39_demo_a")

    def test_pipeline_result_keeps_timestamp_and_score(self):
        example = Path(__file__).resolve().parents[2] / "examples" / "final_result.example.json"
        categories = load_categories()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = convert_pipeline_result(
                example,
                output_dir=root / "out",
                quarantine_dir=root / "bad",
                categories=categories,
            )
            row = json.loads((root / "out" / "observations.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["score"], 0.9468)
            self.assertEqual(row["class_name"], "Car")
            self.assertEqual(row["class_id"], 0)
            self.assertEqual(row["timestamp_ms"], 0.0)
            self.assertFalse(row["timestamp_estimated"])
            self.assertIsNone(row["source_frame_index"])
            self.assertEqual(row["world_position"], {"x": 12.3, "y": 8.4})
            self.assertEqual(row["coordinate_frame"], "road_homography_xy")
            self.assertEqual(row["bbox_xyxy"], [100.0, 200.0, 300.0, 400.0])
            self.assertTrue(row["sequence_id"].startswith("pipeline_"))
            self.assertEqual(stats["written_observations"], 1)

    def test_link_is_repeatable_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "origin"
            source.mkdir()
            link = root / "raw"
            self.assertEqual(ensure_link(source, link), "created")
            self.assertEqual(ensure_link(source, link), "reuse")
            other = root / "other"
            other.mkdir()
            wrong = root / "wrong"
            wrong.symlink_to(other)
            with self.assertRaises(FileExistsError):
                ensure_link(source, wrong)
            occupied = root / "file"
            occupied.write_text("x", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_link(source, occupied)

    def test_inconsistent_cameras_do_not_invent_fps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.json"
            second = root / "b.json"
            dump_json(first, {"frame_stride": 3, "saved_frames": 10, "video_frames": 30, "video_size": 1})
            dump_json(second, {"frame_stride": 5, "saved_frames": 10, "video_frames": 30, "video_size": 1})
            video = root / "video.mp4"
            video.write_bytes(b"x")
            probes = {
                "cam_a": {"path": str(video), "fps": 60, "width": 1920, "height": 1080},
                "cam_b": {"path": str(video), "fps": 60, "width": 1920, "height": 1080},
            }
            # 文件大小与 meta 不一致，但步长已经不一致
            video.write_bytes(b"x")
            result = build_timebase({"cam_a": first, "cam_b": second}, probes)
            self.assertFalse(result["usable"])
            self.assertIsNone(result["sample_fps"])


if __name__ == "__main__":
    unittest.main()
