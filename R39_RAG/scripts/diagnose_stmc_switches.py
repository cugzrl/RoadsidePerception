#!/usr/bin/env python3
"""从已有 STMC 结果里找远端单相机路段的轨迹碎片 不把碎片数当成 IDSW"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CENTER = np.array([-20.66, -25.71])


def arm_of(point: np.ndarray) -> str:
    offset = point - CENTER
    if float(np.linalg.norm(offset)) < 25:
        return "核心"
    names = ["E", "N", "W", "S"]
    axes = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    return names[int(np.argmax(offset @ axes.T))]


def main() -> int:
    result_path = ROOT / "outputs" / "r39_four_camera_test" / "result.json"
    detection_path = ROOT / "outputs" / "r39_four_camera_test" / "detections.json"
    result = json.loads(result_path.read_text())
    tracks = defaultdict(list)
    for frame in result["frames"]:
        for detection in frame["detections"]:
            world = detection.get("world_position")
            gid = detection.get("global_track_id")
            if gid is None or not isinstance(world, dict):
                continue
            tracks[int(gid)].append(
                {
                    "frame": int(frame["frame_index"]),
                    "camera": frame["camera_id"],
                    "score": float(detection["score"]),
                    "class_name": detection["class_name"],
                    "xy": np.array([float(world["x"]), float(world["y"])]),
                    "box": detection["bbox_xyxy"],
                }
            )
    summaries = []
    for gid, rows in tracks.items():
        rows.sort(key=lambda item: (item["frame"], item["camera"]))
        by_frame = defaultdict(list)
        for row in rows:
            by_frame[row["frame"]].append(row)
        frames = sorted(by_frame)
        if len(frames) < 2:
            continue
        end = by_frame[frames[-1]]
        start = by_frame[frames[0]]
        end_xy = np.mean([item["xy"] for item in end], axis=0)
        start_xy = np.mean([item["xy"] for item in start], axis=0)
        summaries.append(
            {
                "gid": gid,
                "class_name": rows[0]["class_name"],
                "start": frames[0],
                "end": frames[-1],
                "start_xy": start_xy,
                "end_xy": end_xy,
                "end_arm": arm_of(end_xy),
                "end_dist": float(np.linalg.norm(end_xy - CENTER)),
                "end_cameras": sorted({item["camera"] for item in end}),
                "n": len(frames),
                "score": float(np.mean([item["score"] for item in end])),
            }
        )
    cases = []
    for older in summaries:
        if older["end_arm"] not in ("N", "S") or older["end_dist"] < 35 or len(older["end_cameras"]) != 1:
            continue
        for newer in summaries:
            if newer["gid"] == older["gid"] or newer["class_name"] != older["class_name"]:
                continue
            gap = newer["start"] - older["end"]
            if gap <= 0 or gap > 15:
                continue
            jump = float(np.linalg.norm(newer["start_xy"] - older["end_xy"]))
            if jump > 18:
                continue
            cases.append(
                {
                    "old_id": older["gid"],
                    "new_id": newer["gid"],
                    "class_name": older["class_name"],
                    "end_frame": older["end"],
                    "start_frame": newer["start"],
                    "gap_frames": gap,
                    "camera": older["end_cameras"][0],
                    "new_cameras": newer["end_cameras"],
                    "arm": older["end_arm"],
                    "end_dist_m": round(older["end_dist"], 1),
                    "jump_m": round(jump, 1),
                    "old_score": round(older["score"], 3),
                    "old_len": older["n"],
                    "new_len": newer["n"],
                }
            )
    cases.sort(key=lambda item: (item["jump_m"], item["gap_frames"]))
    short_far = [
        item for item in summaries if item["end_arm"] in ("N", "S") and item["end_dist"] >= 35 and item["n"] <= 8
    ]
    lengths = [item["n"] for item in summaries]
    report = {
        "source": str(result_path),
        "sequence_note": "这段视频是 R39_new/0 即建库序列 r39_20251021_070125 只用于机制诊断 不作为留出测试",
        "track_count": len(summaries),
        "median_length_frames": float(np.median(lengths)) if lengths else 0,
        "shorter_than_10": int(sum(item <= 10 for item in lengths)),
        "far_ns_single_camera_tracks": int(
            sum(item["end_arm"] in ("N", "S") and item["end_dist"] >= 35 and len(item["end_cameras"]) == 1 for item in summaries)
        ),
        "candidate_handoffs": cases[:12],
        "short_far_fragments": len(short_far),
    }
    if detection_path.is_file():
        detections = json.loads(detection_path.read_text())
        far_small = 0
        far_kept = 0
        for frame in detections["frames"]:
            for detection in frame["detections"]:
                box = detection["bbox_xyxy"]
                area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                if area < 200:
                    far_small += 1
                else:
                    far_kept += 1
        report["detections_area_below_200"] = far_small
        report["detections_area_at_least_200"] = far_kept
    out = ROOT / "R39_RAG" / "reports" / "stmc_switch_cases.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("track_count", "median_length_frames", "shorter_than_10", "far_ns_single_camera_tracks", "short_far_fragments")}, ensure_ascii=False))
    for case in cases[:8]:
        print(case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
