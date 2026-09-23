#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.common import COLORS, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成全局跟踪四宫格视频")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-width", type=int, default=960)
    parser.add_argument("--cell-height", type=int, default=540)
    parser.add_argument("--label-font-size", type=int, default=20)
    parser.add_argument("--camera-font-size", type=int, default=34)
    parser.add_argument("--box-width", type=int, default=3)
    parser.add_argument(
        "--font",
        type=Path,
        default=ROOT / "third_party/HyperLPR/resource/font/platech.ttf",
    )
    return parser.parse_args()


def draw_frame(
    frame: dict,
    label_font: ImageFont.FreeTypeFont,
    camera_font: ImageFont.FreeTypeFont,
    width: int,
    height: int,
    box_width: int,
) -> np.ndarray:
    image = cv2.imread(frame["image_path"])
    if image is None:
        raise FileNotFoundError(f"图片读取失败{frame['image_path']}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    source_height, source_width = rgb.shape[:2]
    canvas = Image.fromarray(rgb).resize((width, height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas)
    scale_x = width / source_width
    scale_y = height / source_height
    for detection in frame["detections"]:
        source_box = detection["bbox_xyxy"]
        x1 = int(round(source_box[0] * scale_x))
        y1 = int(round(source_box[1] * scale_y))
        x2 = int(round(source_box[2] * scale_x))
        y2 = int(round(source_box[3] * scale_y))
        color = COLORS[detection["class_id"] % len(COLORS)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=box_width)
        track_id = detection.get("global_track_id")
        identity = "-" if track_id is None else str(track_id)
        text = f"{detection['class_name']} ID:{identity}"
        if detection.get("plate"):
            text += f" {detection['plate']}"
        text_box = draw.textbbox((0, 0), text, font=label_font, stroke_width=1)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_left = max(0, min(x1, width - text_width - 10))
        label_top = y1 - text_height - 10
        if label_top < 0:
            label_top = min(height - text_height - 10, y1 + box_width)
        label_box = (
            label_left,
            label_top,
            label_left + text_width + 10,
            label_top + text_height + 8,
        )
        draw.rectangle(label_box, fill=color)
        draw.text(
            (label_left + 5, label_top + 3),
            text,
            font=label_font,
            fill=(255, 255, 255),
            stroke_width=1,
            stroke_fill=(70, 70, 70),
        )
    camera_text = frame["camera_id"]
    camera_box = draw.textbbox((0, 0), camera_text, font=camera_font)
    camera_width = camera_box[2] - camera_box[0]
    camera_height = camera_box[3] - camera_box[1]
    draw.rectangle((12, 10, 28 + camera_width, 24 + camera_height), fill=(0, 0, 0))
    draw.text((20, 14), camera_text, font=camera_font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def main() -> int:
    args = parse_args()
    data = load_json(args.result.resolve())
    cameras = [camera["camera_id"] for camera in data["cameras"]]
    frames = {(item["camera_id"], item["frame_index"]): item for item in data["frames"]}
    frame_count = max(camera["frame_count"] for camera in data["cameras"])
    columns = 1 if len(cameras) == 1 else 2
    rows = math.ceil(len(cameras) / columns)
    output_size = (columns * args.cell_width, rows * args.cell_height)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output.resolve()),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(data["fps"]),
        output_size,
    )
    if not writer.isOpened():
        raise RuntimeError("可视化视频创建失败")
    label_font = ImageFont.truetype(str(args.font.resolve()), args.label_font_size)
    camera_font = ImageFont.truetype(str(args.font.resolve()), args.camera_font_size)
    empty = np.zeros((args.cell_height, args.cell_width, 3), dtype=np.uint8)
    for frame_index in range(frame_count):
        cells = []
        for camera_id in cameras:
            record = frames.get((camera_id, frame_index))
            if record is None:
                cells.append(empty.copy())
                continue
            cells.append(
                draw_frame(
                    record,
                    label_font,
                    camera_font,
                    args.cell_width,
                    args.cell_height,
                    args.box_width,
                )
            )
        while len(cells) < rows * columns:
            cells.append(empty.copy())
        row_images = [
            np.hstack(cells[row * columns : (row + 1) * columns])
            for row in range(rows)
        ]
        writer.write(np.vstack(row_images))
        if (frame_index + 1) % 100 == 0 or frame_index + 1 == frame_count:
            print(f"可视化进度{frame_index + 1}/{frame_count}")
    writer.release()
    print(f"可视化视频已写入{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
