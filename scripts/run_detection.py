#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RDDETR_ROOT = ROOT / "algorithms" / "rddetr"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RDDETR_ROOT))

from pipeline.common import (
    CATEGORIES,
    CATEGORY_NAMES,
    CLASS_MIN_BOX_AREAS,
    CLASS_SCORE_THRESHOLDS,
    box_area,
    clip_box,
    load_json,
    save_json,
)
from src.core import YAMLConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行RD-DETR目标检测")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "algorithms/rddetr/configs/rddetr/rddetr_r50vd_6x_coco.yml",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "models/detection/detection_R39.pth",
    )
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--min-box-area", type=float, default=200.0)
    parser.add_argument("--class-thresholds", nargs="*", default=[])
    parser.add_argument("--class-min-box-areas", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def parse_class_values(values: list[str], defaults: dict[int, float], name: str) -> dict[int, float]:
    result = dict(defaults)
    class_ids = {value.lower(): key for key, value in CATEGORY_NAMES.items()}
    for item in values:
        if "=" not in item:
            raise ValueError(f"{name}格式应为类别=数值")
        key, raw_value = item.split("=", 1)
        class_id = int(key) if key.isdigit() else class_ids.get(key.lower())
        if class_id not in CATEGORY_NAMES:
            raise ValueError(f"未知类别{key}")
        result[class_id] = float(raw_value)
    return result


class DeployModel(nn.Module):
    def __init__(self, cfg: YAMLConfig) -> None:
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images: torch.Tensor, sizes: torch.Tensor):
        return self.postprocessor(self.model(images), sizes)


def load_model(config: Path, weights: Path, device: torch.device) -> nn.Module:
    cfg = YAMLConfig(str(config), resume=str(weights))
    checkpoint = torch.load(weights, map_location="cpu")
    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        state_dict = checkpoint["ema"]["module"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    if next(iter(state_dict)).startswith("module."):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    cfg.model.load_state_dict(state_dict, strict=True)
    return DeployModel(cfg).to(device).eval()


def main() -> int:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("检测阈值必须位于0到1之间")
    if args.batch_size <= 0 or args.min_box_area < 0:
        raise ValueError("批大小或最小框面积无效")
    class_thresholds = parse_class_values(
        args.class_thresholds, CLASS_SCORE_THRESHOLDS, "分类别阈值"
    )
    class_min_areas = parse_class_values(
        args.class_min_box_areas, CLASS_MIN_BOX_AREAS, "分类别最小框面积"
    )
    if any(not 0 <= value <= 1 for value in class_thresholds.values()):
        raise ValueError("分类别阈值必须位于0到1之间")
    if any(value < 0 for value in class_min_areas.values()):
        raise ValueError("分类别最小框面积不能小于零")
    manifest = load_json(args.manifest.resolve())
    frames = manifest["frames"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.config.resolve(), args.weights.resolve(), device)
    transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    output_frames = []

    with torch.inference_mode():
        for start in range(0, len(frames), args.batch_size):
            batch_records = frames[start : start + args.batch_size]
            images = []
            tensors = []
            sizes = []
            for record in batch_records:
                image = Image.open(record["image_path"]).convert("RGB")
                images.append(image)
                tensors.append(transform(image))
                sizes.append([image.width, image.height])
            batch = torch.stack(tensors).to(device)
            size_tensor = torch.tensor(sizes, dtype=torch.float32, device=device)
            amp_enabled = device.type == "cuda"
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                labels, boxes, scores = model(batch, size_tensor)
            labels = labels.detach().cpu()
            boxes = boxes.detach().cpu()
            scores = scores.detach().cpu()

            for index, (record, image) in enumerate(zip(batch_records, images)):
                detections = []
                ordinal = 0
                for label, box, score in zip(labels[index], boxes[index], scores[index]):
                    score_value = float(score)
                    class_id = int(label)
                    if class_id not in CATEGORY_NAMES:
                        continue
                    if score_value < max(args.threshold, class_thresholds[class_id]):
                        continue
                    clipped = clip_box(box.tolist(), image.width, image.height)
                    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                        continue
                    if box_area(clipped) < max(args.min_box_area, class_min_areas[class_id]):
                        continue
                    detections.append(
                        {
                            "detection_id": f"{record['camera_id']}:{record['frame_index']}:{ordinal}",
                            "class_id": class_id,
                            "class_name": CATEGORY_NAMES[class_id],
                            "score": round(score_value, 6),
                            "bbox_xyxy": [round(value, 3) for value in clipped],
                        }
                    )
                    ordinal += 1
                output_frames.append({**record, "detections": detections})
                image.close()
            done = min(start + len(batch_records), len(frames))
            print(f"检测进度{done}/{len(frames)}")

    payload = {
        "scene_id": manifest["scene_id"],
        "fps": manifest["fps"],
        "cameras": manifest["cameras"],
        "categories": CATEGORIES,
        "frames": output_frames,
    }
    save_json(args.output.resolve(), payload)
    print(f"检测结果已写入{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
