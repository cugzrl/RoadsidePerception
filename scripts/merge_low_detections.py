#!/usr/bin/env python3
"""保留原始高分框 只追加同帧没有重叠的低分候选"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.common import CLASS_SCORE_THRESHOLDS, box_iou, load_json, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--rescored", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duplicate-iou", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    original = load_json(args.original)
    rescored = load_json(args.rescored)
    extra = {}
    for frame in rescored["frames"]:
        key = (frame["camera_id"], int(frame["frame_index"]))
        extra[key] = frame["detections"]
    added = 0
    kept = 0
    frames = []
    for frame in original["frames"]:
        key = (frame["camera_id"], int(frame["frame_index"]))
        detections = [dict(item) for item in frame["detections"]]
        kept += len(detections)
        for candidate in extra.get(key, []):
            if candidate["score"] >= CLASS_SCORE_THRESHOLDS[candidate["class_id"]]:
                continue
            duplicated = False
            for existing in detections:
                if existing["class_id"] != candidate["class_id"]:
                    continue
                if box_iou(existing["bbox_xyxy"], candidate["bbox_xyxy"]) >= args.duplicate_iou:
                    duplicated = True
                    break
            if duplicated:
                continue
            item = dict(candidate)
            item["detection_id"] = f"{frame['camera_id']}:{frame['frame_index']}:low:{added}"
            item["candidate_pool"] = "low"
            detections.append(item)
            added += 1
        for item in detections:
            item.setdefault("candidate_pool", "high")
        frames.append({**frame, "detections": detections})
    payload = {**original, "frames": frames, "candidate_pool": "original_high_plus_low"}
    save_json(args.output, payload)
    print(f"保留高分 {kept} 追加低分 {added} 写出 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
