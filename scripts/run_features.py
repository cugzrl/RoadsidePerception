#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.common import load_json, padded_box, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取目标外观特征")
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--resnet-weights",
        type=Path,
        default=ROOT / "models/reid/resnet50-0676ba61.pth",
    )
    parser.add_argument("--provider", choices=("resnet50", "mot20"), default="resnet50")
    parser.add_argument(
        "--mot-weights",
        type=Path,
        default=ROOT / "models/reid/tbdq_reid_mot20.pth",
    )
    return parser.parse_args()


def load_resnet(weights_path: Path, device: torch.device):
    model = models.resnet50(weights=None)
    state_dict = torch.load(weights_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.fc = torch.nn.Identity()
    model.to(device).eval()
    preprocess = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    def encode(crops: list[np.ndarray]) -> torch.Tensor:
        tensors = [preprocess(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))) for crop in crops]
        return model(torch.stack(tensors).to(device))

    return encode, 2048


def build_records(detection_data: dict, padding: float) -> list[dict]:
    records = []
    camera_sizes = {
        camera["camera_id"]: (int(camera["width"]), int(camera["height"]))
        for camera in detection_data["cameras"]
    }
    for frame in detection_data["frames"]:
        image_path = frame["image_path"]
        width, height = camera_sizes[frame["camera_id"]]
        for detection in frame["detections"]:
            crop_box = padded_box(detection["bbox_xyxy"], width, height, padding)
            records.append(
                {
                    "feature_index": len(records),
                    "detection_id": detection["detection_id"],
                    "camera_id": frame["camera_id"],
                    "frame_index": frame["frame_index"],
                    "file_name": f"{frame['camera_id']}/{frame['frame_index'] + 1:06d}.jpg",
                    "image_path": image_path,
                    "class_id": detection["class_id"],
                    "class_name": detection["class_name"],
                    "score": detection["score"],
                    "bbox_xyxy": detection["bbox_xyxy"],
                    "crop_xyxy": list(crop_box),
                }
            )
    return records


def load_crops(records: list[dict]) -> list[np.ndarray]:
    crops = []
    current_path = None
    current_image = None
    for record in records:
        if record["image_path"] != current_path:
            current_image = cv2.imread(record["image_path"])
            if current_image is None:
                raise FileNotFoundError(f"图片读取失败{record['image_path']}")
            current_path = record["image_path"]
        left, top, right, bottom = record["crop_xyxy"]
        crops.append(current_image[top:bottom, left:right].copy())
    return crops


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.padding < 0:
        raise ValueError("批大小和裁剪扩边参数无效")
    detection_data = load_json(args.detections.resolve())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.provider == "resnet50":
        encode, feature_dim = load_resnet(args.resnet_weights.resolve(), device)
        provider_meta = {
            "provider": "resnet50",
            "weights": str(args.resnet_weights.resolve()),
            "source": "torchvision ImageNet IMAGENET1K_V1 分类预训练",
            "input": "Resize 256 再 CenterCrop 224",
            "normalized": True,
        }
    else:
        from pipeline.reid_backbones import deepsort_batch, load_deepsort

        mot_model = load_deepsort(args.mot_weights.resolve(), device)
        feature_dim = 512

        def encode(crops: list[np.ndarray]) -> torch.Tensor:
            array = deepsort_batch(mot_model, crops, device, letterbox=False)
            return torch.from_numpy(array).to(device)

        provider_meta = {
            "provider": "mot20",
            "weights": str(args.mot_weights.resolve()),
            "source": "DeepSORT 骨干 在 MOT20 行人轨迹上做身份分类 512 维",
            "input": "检测框直接缩放到 64x128 不做 CenterCrop",
            "normalized": True,
            "selection": "r39_070327 核查链上 median margin 高于 ImageNet ResNet-50 和 MOT17",
        }
    records = build_records(detection_data, args.padding)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.npy"
    if records:
        matrix = np.lib.format.open_memmap(
            feature_path, mode="w+", dtype=np.float32, shape=(len(records), feature_dim)
        )
    else:
        matrix = np.empty((0, feature_dim), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            output = encode(load_crops(batch_records))
            output = F.normalize(output, p=2, dim=1).detach().cpu().float().numpy()
            matrix[start : start + len(batch_records)] = output
            done = min(start + args.batch_size, len(records))
            print(f"特征进度{done}/{len(records)}")
    if records:
        matrix.flush()
    else:
        np.save(feature_path, matrix)
    save_json(
        output_dir / "feature_metadata.json",
        {
            "scene_id": detection_data["scene_id"],
            **provider_meta,
            "feature_dim": feature_dim,
            "feature_count": len(records),
            "feature_matrix": "features.npy",
            "records": records,
        },
    )
    print(f"特征已写入{output_dir}形状{matrix.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
