"""统计轨迹长度"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def iou(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def lengths(result: dict) -> list[int]:
    return [int(track["observation_count"]) for track in result["tracks"]]


def follow(result: dict, camera_id: str, frame_index: int, box: list[float]) -> dict:
    by_frame = {}
    for frame in result["frames"]:
        if frame["camera_id"] != camera_id:
            continue
        by_frame[int(frame["frame_index"])] = frame["detections"]
    current = None
    for detection in by_frame.get(frame_index, []):
        if iou(detection["bbox_xyxy"], box) > 0.3:
            if current is None or iou(detection["bbox_xyxy"], box) > iou(current["bbox_xyxy"], box):
                current = detection
    if current is None:
        return {"found": False}
    switches = []
    identity = current.get("global_track_id")
    last_box = current["bbox_xyxy"]
    last_frame = frame_index
    gap = 0
    for index in range(frame_index + 1, frame_index + 40):
        best = None
        best_iou = 0.0
        for detection in by_frame.get(index, []):
            value = iou(detection["bbox_xyxy"], last_box)
            if value > best_iou:
                best = detection
                best_iou = value
        if best is None or best_iou < 0.7:
            gap += 1
            if gap > 3:
                break
            continue
        gap = 0
        new_id = best.get("global_track_id")
        if new_id != identity:
            switches.append(
                {
                    "frame_index": index,
                    "from_id": identity,
                    "to_id": new_id,
                    "missed_frames": index - last_frame - 1,
                    "iou": best_iou,
                }
            )
            identity = new_id
        last_box = best["bbox_xyxy"]
        last_frame = index
    return {"found": True, "start_id": current.get("global_track_id"), "switches": switches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    values = np.asarray(lengths(result), dtype=np.int32)
    seeds = {
        "south_aw_101": follow(result, "R39_Aw_CamS", 101, [574.8, 203.3, 637.5, 255.1]),
        "north_ce_115": follow(result, "R39_Ce_CamN", 115, [228.752, 201.559, 316.713, 253.713]),
    }
    summary = {
        "name": args.name,
        "tracks": int(values.size),
        "median_observations": float(np.median(values)) if values.size else 0,
        "shorter_than_10": int((values < 10).sum()) if values.size else 0,
        "cases": seeds,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
