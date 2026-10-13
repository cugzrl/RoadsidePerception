#!/usr/bin/env python3
"""在相机内高 IoU 核查链上比较身份特征 不把 STMC 的 ID 当真值"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.reid_backbones import (  # noqa: E402
    deepsort_batch,
    load_deepsort,
    load_resnet50,
    resnet_batch,
)
from scripts.diagnose_reid_identity import center, iou, link_tracklets, quantiles  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--per-class", type=int, default=80)
    parser.add_argument(
        "--mot-weights",
        type=Path,
        nargs="*",
        default=[
            Path("/home/kemove/devdata1/xy/track/FastTracker/pretrained/tbdq_reid_mot17.pth"),
            Path("/home/kemove/devdata1/xy/track/FastTracker/pretrained/tbdq_reid_mot20.pth"),
        ],
    )
    return parser.parse_args()


def collect_pairs(records, chains, class_name: str, limit: int):
    class_chains = [chain for chain in chains if records[chain[0]]["class_name"] == class_name]
    positive = []
    for chain in class_chains:
        taken = 0
        for i, left in enumerate(chain):
            for right in chain[i + 1 :]:
                delta = int(records[right]["frame_index"]) - int(records[left]["frame_index"])
                if delta < 8:
                    continue
                if delta > 12:
                    break
                positive.append((left, right, "same"))
                taken += 1
                break
            if taken:
                break
        if len(positive) >= limit:
            break
    buckets = defaultdict(list)
    for index, record in enumerate(records):
        if record["class_name"] == class_name:
            buckets[(record["camera_id"], record["frame_index"])].append(index)
    negative = []
    for indexes in buckets.values():
        if len(indexes) < 2:
            continue
        for i, left in enumerate(indexes):
            for right in indexes[i + 1 :]:
                dist = float(np.linalg.norm(center(records[left]["bbox_xyxy"]) - center(records[right]["bbox_xyxy"])))
                overlap = iou(records[left]["bbox_xyxy"], records[right]["bbox_xyxy"])
                if dist > 220 and overlap < 0.05:
                    continue
                negative.append((left, right, "diff"))
                if len(negative) >= limit:
                    return positive, negative
    return positive, negative


def load_crop(record) -> np.ndarray:
    image = cv2.imread(record["image_path"])
    if image is None:
        raise FileNotFoundError(record["image_path"])
    left, top, right, bottom = record["crop_xyxy"]
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        x1, y1, x2, y2 = [int(v) for v in record["bbox_xyxy"]]
        crop = image[max(0, y1) : max(y1 + 1, y2), max(0, x1) : max(x1 + 1, x2)]
    return crop


def pack(features: np.ndarray, pairs) -> dict:
    same = [float(np.dot(features[a], features[b])) for a, b, kind in pairs if kind == "same"]
    diff = [float(np.dot(features[a], features[b])) for a, b, kind in pairs if kind == "diff"]
    margin = None
    if same and diff:
        margin = round(float(np.median(same) - np.median(diff)), 4)
    return {"same": quantiles(same), "diff": quantiles(diff), "median_margin": margin}


def encode_named(name, crops, device, models_ready):
    if name == "resnet50_centercrop":
        return resnet_batch(models_ready["resnet"], crops, device, full=False)
    if name == "resnet50_fullcrop":
        return resnet_batch(models_ready["resnet"], crops, device, full=True)
    if name.startswith("mot_stretch_"):
        return deepsort_batch(models_ready[name], crops, device, letterbox=False)
    if name.startswith("mot_letter_"):
        return deepsort_batch(models_ready[name], crops, device, letterbox=True)
    raise KeyError(name)


def main() -> int:
    args = parse_args()
    meta = json.loads((args.features_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    base = np.load(args.features_dir / "features.npy", mmap_mode="r")
    records = meta["records"]
    chains = link_tracklets(records)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models_ready = {
        "resnet": load_resnet50(ROOT / "models/reid/resnet50-0676ba61.pth", device),
    }
    names = ["resnet50_centercrop", "resnet50_fullcrop"]
    for path in args.mot_weights:
        if not path.is_file():
            continue
        tag = path.stem.replace("tbdq_reid_", "")
        models_ready[f"mot_stretch_{tag}"] = load_deepsort(path, device)
        models_ready[f"mot_letter_{tag}"] = models_ready[f"mot_stretch_{tag}"]
        names.append(f"mot_stretch_{tag}")
        names.append(f"mot_letter_{tag}")
    report = {
        "note": "正样本来自同相机 IoU>=0.45 且间隔约 8 帧的核查链 负样本是同帧近距离不同框",
        "weights": {
            "resnet50": "torchvision ImageNet IMAGENET1K_V1 resnet50-0676ba61.pth 2048 维 L2",
            "mot": "DeepSORT 结构 512 维 L2 权重来自 FastTracker tbdq_reid_mot17/mot20 行人身份分类",
        },
        "classes": {},
    }
    focus = ["ScooterRider", "Pedestrian", "Car"]
    for class_name in focus:
        positive, negative = collect_pairs(records, chains, class_name, args.per_class)
        pairs = positive + negative
        indexes = sorted({index for pair in pairs for index in pair[:2]})
        crops = {index: load_crop(records[index]) for index in indexes}
        ordered = [crops[index] for index in indexes]
        remap = {index: pos for pos, index in enumerate(indexes)}
        local_pairs = [(remap[a], remap[b], kind) for a, b, kind in pairs]
        class_report = {"pairs": {"same": len(positive), "diff": len(negative)}}
        stored = np.stack([base[index] for index in indexes])
        class_report["stored_resnet50"] = pack(stored, local_pairs)
        for name in names:
            encoded = encode_named(name, ordered, device, models_ready)
            class_report[name] = pack(encoded, local_pairs)
            print(class_name, name, class_report[name]["median_margin"], flush=True)
        report["classes"][class_name] = class_report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
