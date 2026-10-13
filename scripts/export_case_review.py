#!/usr/bin/env python3
"""把人工指出的身份对齐到各组全局 ID 并输出放大对比"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_reid_identity import link_tracklets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", required=True, help="名称=tracking.json")
    return parser.parse_args()


def load_tracking(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping = {}
    frames = defaultdict(list)
    for frame in data["frames"]:
        for det in frame["detections"]:
            gid = det.get("global_track_id")
            if gid is None:
                continue
            mapping[det["detection_id"]] = int(gid)
            frames[det["detection_id"]] = det
    return data, mapping, frames


def reference_ids(tracking_path: Path, global_id: int) -> set[str]:
    data = json.loads(tracking_path.read_text(encoding="utf-8"))
    found = set()
    for frame in data["frames"]:
        for det in frame["detections"]:
            if det.get("global_track_id") == global_id:
                found.add(det["detection_id"])
    return found


def extend_chain(records: list[dict], seeds: set[str]) -> list[dict]:
    chains = link_tracklets(records)
    chosen = []
    for chain in chains:
        ids = {records[index]["detection_id"] for index in chain}
        if ids & seeds:
            chosen.extend(records[index] for index in chain)
    chosen.sort(key=lambda item: (item["camera_id"], item["frame_index"]))
    return chosen


def gid_story(chain: list[dict], mapping: dict[str, int]) -> dict:
    sequence = []
    switches = []
    previous = None
    covered = 0
    for record in chain:
        gid = mapping.get(record["detection_id"])
        if gid is None:
            continue
        covered += 1
        if not sequence or sequence[-1]["global_id"] != gid:
            sequence.append(
                {
                    "global_id": gid,
                    "frame": int(record["frame_index"]),
                    "camera_id": record["camera_id"],
                }
            )
            if previous is not None and previous != gid:
                switches.append(
                    {
                        "frame": int(record["frame_index"]),
                        "camera_id": record["camera_id"],
                        "from_id": previous,
                        "to_id": gid,
                    }
                )
        previous = gid
    unique = list(dict.fromkeys(item["global_id"] for item in sequence))
    return {"covered": covered, "unique_ids": unique, "switches": switches, "segments": sequence}


def crop_box(box, width, height, scale=3.2):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1, 80) * scale
    left = int(max(0, cx - side))
    top = int(max(0, cy - side * 0.75))
    right = int(min(width, cx + side))
    bottom = int(min(height, cy + side * 0.75))
    if right - left < 40 or bottom - top < 40:
        return 0, 0, width, height
    return left, top, right, bottom


def render_zoom(chain, group_maps, image_root: Path, output: Path, title: str) -> None:
    by_frame = defaultdict(list)
    for record in chain:
        by_frame[(record["camera_id"], int(record["frame_index"]))].append(record)
    keys = sorted(by_frame)
    if not keys:
        return
    names = list(group_maps)
    sample = cv2.imread(str(image_root / keys[0][0] / f"{keys[0][1] + 1:06d}.jpg"))
    if sample is None:
        print("缺少画面", keys[0])
        return
    tile_w, tile_h = 360, 220
    canvas_w = tile_w * len(names)
    canvas_h = tile_h + 28
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (canvas_w, canvas_h))
    last_gid = {name: None for name in names}
    for camera_id, frame_index in keys:
        image = cv2.imread(str(image_root / camera_id / f"{frame_index + 1:06d}.jpg"))
        if image is None:
            continue
        height, width = image.shape[:2]
        boxes = [item["bbox_xyxy"] for item in by_frame[(camera_id, frame_index)]]
        union = [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]
        left, top, right, bottom = crop_box(union, width, height)
        canvas = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)
        cv2.putText(canvas, f"{title} {camera_id} frame {frame_index}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        for column, name in enumerate(names):
            crop = image[top:bottom, left:right].copy()
            mapping = group_maps[name]
            gids = []
            for record in by_frame[(camera_id, frame_index)]:
                gid = mapping.get(record["detection_id"])
                if gid is None:
                    continue
                gids.append(gid)
                x1, y1, x2, y2 = record["bbox_xyxy"]
                color = (0, 220, 0) if last_gid[name] in (None, gid) else (0, 0, 255)
                cv2.rectangle(crop, (int(x1 - left), int(y1 - top)), (int(x2 - left), int(y2 - top)), color, 2)
                cv2.putText(crop, str(gid), (int(x1 - left), max(16, int(y1 - top) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            if gids:
                last_gid[name] = gids[-1]
            tile = cv2.resize(crop, (tile_w, tile_h - 28))
            x0 = column * tile_w
            canvas[28:tile_h, x0 : x0 + tile_w] = tile
            cv2.putText(canvas, name, (x0 + 6, canvas_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()
    print("放大视频", output)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    records = meta["records"]
    groups = {}
    maps = {}
    for item in args.groups:
        name, path = item.split("=", 1)
        _data, mapping, _frames = load_tracking(Path(path))
        groups[name] = Path(path)
        maps[name] = mapping
    cases = [
        ("id46_scooter", ROOT / "outputs/r39_full_070327/variants/D/tracking.json", 46),
        ("id26_scooter", ROOT / "outputs/r39_full_070327/variants/Aplus/tracking.json", 26),
    ]
    report = {}
    image_root = ROOT / "outputs/r39_full_070327/frames"
    for name, tracking_path, global_id in cases:
        seeds = reference_ids(tracking_path, global_id)
        chain = extend_chain(records, seeds)
        report[name] = {
            "seed_detections": len(seeds),
            "chain_detections": len(chain),
            "groups": {group: gid_story(chain, maps[group]) for group in maps},
        }
        render_zoom(chain, maps, image_root, args.output_dir / f"{name}.mp4", name)
    (args.output_dir / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: {group: value["unique_ids"] for group, value in item["groups"].items()} for key, item in report.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
