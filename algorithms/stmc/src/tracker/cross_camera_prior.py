"""同一时刻不同相机的共视边权修正

几何接近只提供候选支持 不强制同 ID
旁边还有另一辆车时不抬边 避免把相邻车辆并到一起
样本或投影不够时保持原边权
"""

from __future__ import annotations

import json
from pathlib import Path


def coview_delta(
    distance: float,
    appearance: float,
    second_distance: float,
    base_weight: float,
    near_m: float = 4.0,
    appearance_min: float = 0.72,
    neighbor_gap_m: float = 3.0,
    max_delta: float = 0.06,
) -> tuple[float, str]:
    """返回修正量和原因 零修正表示回退原 STMC"""
    if base_weight <= 0.0:
        return 0.0, "no_edge"
    if distance > near_m:
        return 0.0, "not_near"
    if appearance < appearance_min:
        return 0.0, "weak_reid"
    if second_distance - distance < neighbor_gap_m:
        return 0.0, "neighbor"
    return float(max_delta), "boost"


class CrossCameraPrior:
    def __init__(self, cfg) -> None:
        self.near_m = float(cfg.get("near_m", 4.0))
        self.appearance_min = float(cfg.get("appearance_min", 0.72))
        self.neighbor_gap_m = float(cfg.get("neighbor_gap_m", 3.0))
        self.max_delta = float(cfg.get("max_delta", 0.06))
        self.log_path = cfg.get("log_path")
        self._lines: list[str] = []
        self.counts = {"boost": 0, "not_near": 0, "weak_reid": 0, "neighbor": 0, "no_edge": 0}

    def note(self, frame: int, left: int, right: int, distance: float, appearance: float, base: float, delta: float, reason: str) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1
        if not self.log_path or reason == "no_edge":
            return
        record = {
            "frame": int(frame),
            "left": int(left),
            "right": int(right),
            "distance_m": round(float(distance), 3),
            "appearance": round(float(appearance), 4),
            "base_weight": round(float(base), 4),
            "delta": round(float(delta), 4),
            "final_weight": round(float(base + delta), 4),
            "reason": reason,
        }
        self._lines.append(json.dumps(record, ensure_ascii=False) + "\n")
        if len(self._lines) >= 80:
            self.flush()

    def flush(self) -> None:
        if not self._lines or not self.log_path:
            return
        target = Path(str(self.log_path))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as file:
            file.writelines(self._lines)
        self._lines.clear()
