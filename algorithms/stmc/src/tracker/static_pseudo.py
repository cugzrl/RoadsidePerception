"""固定画面位置上的矮框诊断

交通锥常被检成行人 长时间几乎不动 且高宽比明显小于站立行人
只在累计足够历史后拒绝后续框 不追溯删除已经建立的轨迹
静止的汽车不在这里处理 避免误删停车和等灯车辆
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


class _Prototype:
    def __init__(self, class_id: int, camera: int, box: np.ndarray) -> None:
        self.class_id = class_id
        self.camera = camera
        self.box = box.astype(np.float64).copy()
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5
        self.hits = 1
        self.mean_x = float(cx)
        self.mean_y = float(cy)
        self.m2_x = 0.0
        self.m2_y = 0.0
        self.height = float(box[3] - box[1])
        self.width = float(box[2] - box[0])

    def center_std(self) -> float:
        if self.hits < 2:
            return 0.0
        var = (self.m2_x + self.m2_y) / (self.hits - 1)
        return float(np.sqrt(max(var, 0.0)))

    def aspect(self) -> float:
        if self.width <= 1:
            return 99.0
        return self.height / self.width

    def update(self, box: np.ndarray) -> None:
        self.hits += 1
        cx = (float(box[0]) + float(box[2])) * 0.5
        cy = (float(box[1]) + float(box[3])) * 0.5
        dx = cx - self.mean_x
        self.mean_x += dx / self.hits
        self.m2_x += dx * (cx - self.mean_x)
        dy = cy - self.mean_y
        self.mean_y += dy / self.hits
        self.m2_y += dy * (cy - self.mean_y)
        self.box = 0.9 * self.box + 0.1 * box
        self.height = float(self.box[3] - self.box[1])
        self.width = float(self.box[2] - self.box[0])


class StaticPseudo:
    def __init__(self, cfg) -> None:
        self.min_hits = int(cfg.get("min_hits", 40))
        self.max_center_std = float(cfg.get("max_center_std", 2.0))
        self.low_attach_limit = int(cfg.get("low_attach_limit", 3))
        self.max_aspect = float(cfg.get("max_aspect", 1.75))
        self.max_height = float(cfg.get("max_height", 130.0))
        raw = cfg.get("classes", [6])
        self.classes = {int(item) for item in raw}
        self.match_px = float(cfg.get("match_px", 12.0))
        self.log_path = cfg.get("log_path")
        self.prototypes: list[_Prototype] = []
        self._lines: list[str] = []
        self.dropped = 0

    def consider(self, class_id: int, camera: int, box: np.ndarray, frame: int | None = None) -> str | None:
        """观察一个框 已经确认的静态矮目标返回拒绝原因"""
        class_id = int(class_id)
        if class_id not in self.classes:
            return None
        box = np.asarray(box, dtype=np.float64)
        found = self._match(class_id, camera, box)
        if found is None:
            self.prototypes.append(_Prototype(class_id, camera, box))
            return None
        already = self._confirmed(found)
        found.update(box)
        if not already and not self._confirmed(found):
            return None
        if not self._confirmed(found):
            return None
        reason = "static_squat"
        self.dropped += 1
        self._log(class_id, camera, found, reason, frame)
        return reason

    def _confirmed(self, item: _Prototype) -> bool:
        return (
            item.hits >= self.min_hits
            and item.center_std() <= self.max_center_std
            and item.aspect() <= self.max_aspect
            and item.height <= self.max_height
        )

    def _match(self, class_id: int, camera: int, box: np.ndarray) -> _Prototype | None:
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5
        best = None
        best_iou = 0.55
        for item in self.prototypes:
            if item.class_id != class_id or item.camera != camera:
                continue
            dist = ((cx - item.mean_x) ** 2 + (cy - item.mean_y) ** 2) ** 0.5
            if dist > self.match_px:
                continue
            overlap = box_iou(box, item.box)
            if overlap > best_iou:
                best_iou = overlap
                best = item
        return best

    def _log(self, class_id: int, camera: int, item: _Prototype, reason: str, frame: int | None) -> None:
        if not self.log_path:
            return
        record = {
            "frame": None if frame is None else int(frame),
            "reason": reason,
            "class_id": class_id,
            "camera": int(camera),
            "hits": int(item.hits),
            "center_std": round(item.center_std(), 3),
            "aspect": round(item.aspect(), 3),
            "height": round(item.height, 1),
        }
        self._lines.append(json.dumps(record, ensure_ascii=False) + "\n")
        if len(self._lines) >= 40:
            self.flush()

    def flush(self) -> None:
        if not self._lines or not self.log_path:
            return
        target = Path(str(self.log_path))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as file:
            file.writelines(self._lines)
        self._lines.clear()
