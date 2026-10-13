#!/usr/bin/env python3
"""用相机内高 IoU 短链检查外观特征 不用 STMC 预测 ID 当真值"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


FOCUS = ("ScooterRider", "Cyclist", "Pedestrian", "Car")


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
    return float(inter / max(area_a + area_b - inter, 1e-6))


def center(box):
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float64)


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "p10": round(float(np.quantile(arr, 0.10)), 4),
        "p50": round(float(np.quantile(arr, 0.50)), 4),
        "p90": round(float(np.quantile(arr, 0.90)), 4),
    }


def link_tracklets(records: list[dict], min_iou: float = 0.45, max_gap: int = 2) -> list[list[int]]:
    """同相机同类别 高重叠短间隙连成核查链"""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[(record["camera_id"], record["class_name"])].append(index)
    chains: list[list[int]] = []
    for indexes in groups.values():
        indexes.sort(key=lambda item: (records[item]["frame_index"], records[item]["detection_id"]))
        active: list[list[int]] = []
        for index in indexes:
            record = records[index]
            best_chain = None
            best_score = min_iou
            for chain in active:
                last = records[chain[-1]]
                gap = int(record["frame_index"]) - int(last["frame_index"])
                if gap <= 0 or gap > max_gap:
                    continue
                score = iou(record["bbox_xyxy"], last["bbox_xyxy"])
                if score > best_score:
                    best_score = score
                    best_chain = chain
            if best_chain is None:
                active.append([index])
            else:
                best_chain.append(index)
            frame = int(record["frame_index"])
            kept = []
            for chain in active:
                if frame - int(records[chain[-1]]["frame_index"]) <= max_gap:
                    kept.append(chain)
                else:
                    chains.append(chain)
            active = kept
        chains.extend(active)
    return [chain for chain in chains if len(chain) >= 8]


def pair_similarity(features: np.ndarray, left: int, right: int) -> float:
    return float(np.dot(features[left], features[right]))


def sample_same(features, records, chains, gap_frames: int, limit: int = 400) -> list[float]:
    values = []
    if not chains:
        return values
    quota = max(1, limit // max(len(chains), 1))
    for chain in chains:
        taken = 0
        for i, left in enumerate(chain):
            for right in chain[i + 1 :]:
                delta = int(records[right]["frame_index"]) - int(records[left]["frame_index"])
                if delta < gap_frames:
                    continue
                if delta > gap_frames + 2:
                    break
                values.append(pair_similarity(features, left, right))
                taken += 1
                if taken >= quota or len(values) >= limit:
                    break
            if taken >= quota or len(values) >= limit:
                break
        if len(values) >= limit:
            break
    return values


def sample_same_frame_negative(features, records, class_name: str, limit: int = 400) -> list[float]:
    """同一帧同一相机的两个框不可能是同一个目标"""
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record["class_name"] != class_name:
            continue
        buckets[(record["camera_id"], record["frame_index"])].append(index)
    values = []
    for indexes in buckets.values():
        if len(indexes) < 2:
            continue
        for i, left in enumerate(indexes):
            for right in indexes[i + 1 :]:
                dist = float(np.linalg.norm(center(records[left]["bbox_xyxy"]) - center(records[right]["bbox_xyxy"])))
                if dist > 220:
                    continue
                values.append(pair_similarity(features, left, right))
                if len(values) >= limit:
                    return values
    return values


def sample_different(features, records, chains, limit: int = 400) -> list[float]:
    """时间重叠且中心较近的不同核查链 当作外观相近的负样本"""
    by_key: dict[tuple, list[list[int]]] = defaultdict(list)
    for chain in chains:
        record = records[chain[0]]
        by_key[(record["camera_id"], record["class_name"])].append(chain)
    values = []
    for group in by_key.values():
        spans = []
        for chain in group:
            frames = [int(records[index]["frame_index"]) for index in chain]
            spans.append((min(frames), max(frames), chain))
        for i, (a0, a1, left_chain) in enumerate(spans):
            for b0, b1, right_chain in spans[i + 1 :]:
                if a1 < b0 or b1 < a0:
                    continue
                left = records[left_chain[len(left_chain) // 2]]
                right = records[right_chain[len(right_chain) // 2]]
                if abs(int(left["frame_index"]) - int(right["frame_index"])) > 15:
                    continue
                dist = float(np.linalg.norm(center(left["bbox_xyxy"]) - center(right["bbox_xyxy"])))
                if dist > 180:
                    continue
                values.append(pair_similarity(features, left_chain[len(left_chain) // 2], right_chain[len(right_chain) // 2]))
                if len(values) >= limit:
                    return values
    return values


def crop_stats(records: list[dict]) -> dict:
    ratios = []
    areas = []
    for record in records:
        box = record["bbox_xyxy"]
        crop = record["crop_xyxy"]
        box_w = max(1.0, box[2] - box[0])
        box_h = max(1.0, box[3] - box[1])
        crop_w = max(1.0, crop[2] - crop[0])
        crop_h = max(1.0, crop[3] - crop[1])
        ratios.append((crop_w * crop_h) / (box_w * box_h))
        areas.append(box_w * box_h)
    return {"crop_over_box": quantiles(ratios), "box_area": quantiles(areas)}


def track_case(tracking_path: Path, features, records, global_id: int) -> dict:
    data = json.loads(tracking_path.read_text(encoding="utf-8"))
    by_det = {record["detection_id"]: index for index, record in enumerate(records)}
    hits = []
    for frame in data["frames"]:
        for det in frame["detections"]:
            if det.get("global_track_id") != global_id:
                continue
            index = by_det.get(det["detection_id"])
            if index is None:
                continue
            hits.append((frame["camera_id"], frame["frame_index"], index, det["class_name"]))
    hits.sort(key=lambda item: (item[1], item[0]))
    if len(hits) < 2:
        return {"global_id": global_id, "n": len(hits)}
    consecutive = []
    early_late = []
    jumps = []
    split = max(1, len(hits) // 3)
    early = hits[:split]
    late = hits[-split:]
    for left in early[:: max(1, len(early) // 8)]:
        for right in late[:: max(1, len(late) // 8)]:
            early_late.append(pair_similarity(features, left[2], right[2]))
    for left, right in zip(hits, hits[1:]):
        sim = pair_similarity(features, left[2], right[2])
        gap = right[1] - left[1]
        if left[0] == right[0] and 0 < gap <= 3:
            consecutive.append(sim)
        if left[0] != right[0] or gap > 5:
            jumps.append(
                {
                    "from": [left[0], left[1]],
                    "to": [right[0], right[1]],
                    "gap": gap,
                    "sim": round(sim, 4),
                    "class": right[3],
                }
            )
    return {
        "global_id": global_id,
        "n": len(hits),
        "class": hits[0][3],
        "frames": [hits[0][1], hits[-1][1]],
        "consecutive": quantiles(consecutive),
        "early_vs_late": quantiles(early_late),
        "jumps": jumps[:12],
    }


def continuation(tracking_path: Path, global_id: int) -> dict:
    """看某条短轨迹结束时 同相机同类附近是否立刻出现新 ID"""
    data = json.loads(tracking_path.read_text(encoding="utf-8"))
    points = []
    for frame in data["frames"]:
        for det in frame["detections"]:
            if det.get("global_track_id") is None:
                continue
            points.append(
                {
                    "id": det["global_track_id"],
                    "camera": frame["camera_id"],
                    "frame": frame["frame_index"],
                    "cls": det["class_name"],
                    "box": det["bbox_xyxy"],
                }
            )
    mine = [item for item in points if item["id"] == global_id]
    if not mine:
        return {"global_id": global_id, "n": 0}
    last = max(mine, key=lambda item: item["frame"])
    next_ids = []
    for item in points:
        if item["camera"] != last["camera"] or item["cls"] != last["cls"] or item["id"] == global_id:
            continue
        if not 0 < item["frame"] - last["frame"] <= 8:
            continue
        if iou(item["box"], last["box"]) >= 0.2:
            next_ids.append({"id": item["id"], "frame": item["frame"], "iou": round(iou(item["box"], last["box"]), 3)})
    return {
        "global_id": global_id,
        "class": last["cls"],
        "end": [last["camera"], last["frame"]],
        "nearby_new_ids": next_ids[:8],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--tracking", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focus-ids", nargs="*", type=int, default=[26, 46])
    args = parser.parse_args()
    meta = json.loads((args.features_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    features = np.load(args.features_dir / "features.npy", mmap_mode="r")
    records = meta["records"]
    assert features.shape[0] == len(records)
    chains = link_tracklets(records)
    by_class: dict[str, list] = defaultdict(list)
    for chain in chains:
        by_class[records[chain[0]]["class_name"]].append(chain)
    report = {
        "provider": meta.get("provider"),
        "feature_dim": meta.get("feature_dim"),
        "normalized": meta.get("normalized"),
        "preprocess_note": "ResNet50 ImageNet 权重 输入经过 Resize 256 再 CenterCrop 224",
        "audit_chains": {name: len(items) for name, items in sorted(by_class.items())},
        "crop": {},
        "similarity": {},
    }
    for name in FOCUS:
        selected = [record for record in records if record["class_name"] == name]
        report["crop"][name] = crop_stats(selected[:2000])
        class_chains = by_class.get(name, [])
        same = sample_same(features, records, class_chains, 8)
        same_far = sample_same(features, records, class_chains, 20)
        near = sample_different(features, records, class_chains)
        negatives = sample_same_frame_negative(features, records, name)
        margin_src = negatives or near
        gap = None
        if same and margin_src:
            gap = round(float(np.median(same) - np.median(margin_src)), 4)
        report["similarity"][name] = {
            "same_gap8": quantiles(same),
            "same_gap20": quantiles(same_far),
            "different_nearby_chains": quantiles(near),
            "same_frame_different": quantiles(negatives),
            "median_margin": gap,
        }
    cases = []
    follows = []
    for path in args.tracking:
        for gid in args.focus_ids:
            item = track_case(path, features, records, gid)
            item["tracking"] = path.parent.name
            cases.append(item)
            follow = continuation(path, gid)
            follow["tracking"] = path.parent.name
            follows.append(follow)
    report["focus_tracks"] = cases
    report["continuations"] = follows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("provider", "audit_chains", "similarity")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
