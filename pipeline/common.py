from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


CATEGORIES = [
    {"id": 0, "name": "Car"},
    {"id": 1, "name": "Van"},
    {"id": 2, "name": "Bus"},
    {"id": 3, "name": "Tricycle"},
    {"id": 4, "name": "ScooterRider"},
    {"id": 5, "name": "Cyclist"},
    {"id": 6, "name": "Pedestrian"},
]
CATEGORY_NAMES = {item["id"]: item["name"] for item in CATEGORIES}
PLATE_CLASS_IDS = {0, 1, 2}
CLASS_SCORE_THRESHOLDS = {
    0: 0.65,
    1: 0.65,
    2: 0.65,
    3: 0.70,
    4: 0.72,
    5: 0.72,
    6: 0.75,
}
CLASS_MIN_BOX_AREAS = {
    0: 200.0,
    1: 300.0,
    2: 500.0,
    3: 300.0,
    4: 300.0,
    5: 300.0,
    6: 350.0,
}
COLORS = [
    (137, 207, 240),
    (146, 224, 174),
    (250, 218, 134),
    (244, 168, 150),
    (190, 178, 239),
    (137, 221, 223),
    (244, 167, 185),
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )


def clip_box(box: Iterable[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [
        max(0.0, min(x1, width)),
        max(0.0, min(y1, height)),
        max(0.0, min(x2, width)),
        max(0.0, min(y2, height)),
    ]


def padded_box(
    box: Iterable[float], width: int, height: int, padding: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = clip_box(box, width, height)
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    left = max(0, int(math.floor(x1 - pad_x)))
    top = max(0, int(math.floor(y1 - pad_y)))
    right = min(width, int(math.ceil(x2 + pad_x)))
    bottom = min(height, int(math.ceil(y2 + pad_y)))
    if right <= left or bottom <= top:
        raise ValueError(f"无效检测框{list(box)}")
    return left, top, right, bottom


def box_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / max(area_a + area_b - intersection, 1e-6)


def box_area(box: Iterable[float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
