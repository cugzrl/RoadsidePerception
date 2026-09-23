#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.common import PLATE_CLASS_IDS, box_area, load_json, padded_box, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在全局跟踪后识别车辆车牌")
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models/license_plate/hyperlpr3",
    )
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-box-area", type=float, default=2500)
    return parser.parse_args()


def valid_plate(text: str) -> bool:
    normalized = re.sub(r"[·.\s-]", "", text.upper())
    return bool(re.fullmatch(r"[\u4e00-\u9fff][A-Z][A-Z0-9]{5,6}", normalized))


def normalize_plate(text: str) -> str:
    return re.sub(r"[·.\s-]", "", text.upper())


def load_catcher(model_dir: Path):
    os.environ["HYPERLPR_HOME"] = str(model_dir)
    sys.path.insert(0, str(ROOT / "third_party/HyperLPR/Prj-Python"))
    import hyperlpr3 as lpr3

    return lpr3.LicensePlateCatcher(
        folder=str(model_dir), detect_level=lpr3.DETECT_LEVEL_LOW
    )


def main() -> int:
    args = parse_args()
    data = load_json(args.tracking.resolve())
    catcher = load_catcher(args.model_dir.resolve())
    candidates = defaultdict(list)
    for frame in data["frames"]:
        for detection in frame["detections"]:
            track_id = detection.get("global_track_id")
            if track_id is None or detection["class_id"] not in PLATE_CLASS_IDS:
                continue
            area = box_area(detection["bbox_xyxy"])
            if area < args.min_box_area:
                continue
            quality = area * float(detection["score"])
            candidates[int(track_id)].append((quality, frame, detection))

    plate_results = {}
    for track_id, items in candidates.items():
        votes = defaultdict(float)
        best_confidence = defaultdict(float)
        selected = sorted(items, key=lambda item: item[0], reverse=True)[: args.max_candidates]
        for _, frame, detection in selected:
            image = cv2.imread(frame["image_path"])
            if image is None:
                data["errors"].append(
                    {"stage": "license_plate", "message": f"图片读取失败{frame['image_path']}"}
                )
                continue
            height, width = image.shape[:2]
            left, top, right, bottom = padded_box(
                detection["bbox_xyxy"], width, height, 0.03
            )
            vehicle_crop = image[top:bottom, left:right]
            for code, confidence, _, _ in catcher(vehicle_crop):
                confidence = float(confidence)
                code = normalize_plate(str(code))
                if confidence < args.min_confidence or not valid_plate(code):
                    continue
                votes[code] += confidence
                best_confidence[code] = max(best_confidence[code], confidence)
        if votes:
            text = max(votes, key=votes.get)
            plate_results[track_id] = (text, round(best_confidence[text], 6))
        else:
            plate_results[track_id] = ("", None)

    for track in data["tracks"]:
        if track["class_id"] in PLATE_CLASS_IDS:
            text, confidence = plate_results.get(track["global_track_id"], ("", None))
        else:
            text, confidence = "", None
        track["plate"] = text
        track["plate_confidence"] = confidence
    track_plates = {
        track["global_track_id"]: track["plate"] for track in data["tracks"]
    }
    for frame in data["frames"]:
        for detection in frame["detections"]:
            track_id = detection.get("global_track_id")
            detection["plate"] = track_plates.get(track_id, "")

    save_json(args.output.resolve(), data)
    print(f"最终结果已写入{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
