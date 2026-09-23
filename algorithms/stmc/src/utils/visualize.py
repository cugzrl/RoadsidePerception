from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.utils.export_json import lookup_class


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    """为同一个全局 ID 生成固定颜色。"""
    hue = (track_id * 0.61803398875) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return tuple(int(value * 255) for value in rgb)


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """将 DataLoader 输出的图像 Tensor 转为 RGB PIL 图像。"""
    image = image.detach().cpu()

    # DataLoader 的 batch_size=1 会多出一个 batch 维度
    if image.ndim == 4:
        image = image.squeeze(0)

    # CHW -> HWC
    if image.ndim != 3:
        raise ValueError(f"不支持的图像形状：{image.shape}")

    if image.shape[0] in (1, 3, 4):
        image = image.permute(1, 2, 0)

    image = image.numpy()

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = image * 255.0

        image = np.clip(image, 0, 255).astype(np.uint8)

    # 灰度图转 RGB
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    # RGBA 转 RGB
    if image.shape[2] == 4:
        image = image[:, :, :3]

    return Image.fromarray(image, mode="RGB")


def _get_font(size: int = 16) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def save_tracking_visualization(
    images,
    results: torch.Tensor,
    camera_names,
    output_dir,
    frame_id: int,
    class_lookup: Optional[Dict[Tuple[str, int], List[dict]]] = None,
    class_match_iou: float = 0.3,
) -> None:
    """
    为每个摄像头保存一张带检测框、全局 ID 和类别的图片。

    results 每行格式：
    [camera_id, global_id, frame, x, y, w, h, bev_x, bev_y]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = results.detach().cpu()
    font = _get_font(18)

    for camera_id, image_tensor in enumerate(images):
        image = _tensor_to_pil(image_tensor)
        draw = ImageDraw.Draw(image)

        if results.numel() > 0:
            camera_results = results[results[:, 0].long() == camera_id]
            camera_name = str(camera_names[camera_id])

            for row in camera_results:
                global_id = int(row[1].item())
                frame = int(row[2].item())

                x = float(row[3].item())
                y = float(row[4].item())
                width = float(row[5].item())
                height = float(row[6].item())

                x2 = x + width
                y2 = y + height

                color = _color_for_id(global_id)

                draw.rectangle(
                    [x, y, x2, y2],
                    outline=color,
                    width=3,
                )

                class_name = None
                if class_lookup:
                    class_name, _, _ = lookup_class(
                        class_lookup,
                        camera_name,
                        frame,
                        {"x0": x, "y0": y, "x1": x2, "y1": y2},
                        class_match_iou,
                    )

                if class_name:
                    label = f"ID {global_id}|{class_name}"
                else:
                    label = f"ID {global_id}"

                text_y = max(0, y - 20)
                # 深色底，保证白字/彩字可读
                try:
                    left, top, right, bottom = draw.textbbox((x, text_y), label, font=font)
                    draw.rectangle([left - 2, top - 1, right + 2, bottom + 1], fill=(0, 0, 0))
                except Exception:
                    pass

                draw.text(
                    (x, text_y),
                    label,
                    fill=color,
                    font=font,
                    stroke_width=1,
                    stroke_fill=(0, 0, 0),
                )

        camera_name = str(camera_names[camera_id])
        camera_output_dir = output_dir / camera_name
        camera_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = camera_output_dir / f"{frame_id:06d}.jpg"
        image.save(output_path, quality=95)
