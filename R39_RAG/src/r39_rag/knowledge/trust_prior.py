"""只用四段建库序列统计跟踪用的 BEV 可信度 不覆盖原知识库"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from r39_rag.config import PROJECT_ROOT, load_knowledge, load_sources
from r39_rag.projection.evaluate import iter_observations
from r39_rag.projection.homography import load_projectors
from r39_rag.projection.quality import project_observation

CENTER = np.array([-20.66, -25.71], dtype=np.float64)
CELL_M = 10.0
TAIL_M = 15.0


def _level(count: int, median: float, tail: float) -> str:
    if count >= 40 and median <= 5 and tail <= 0.20:
        return "高"
    if count >= 20 and median <= 8 and tail <= 0.35:
        return "中"
    if count >= 10:
        return "低"
    return "样本不足"


def build_tracking_trust(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_knowledge()
    sources = load_sources()
    sequences = list(config["build_sequences"])
    projectors = load_projectors(config["calibration_dir"], sources["cameras"], float(config["min_denominator"]))
    groups: dict[tuple, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for row in iter_observations(sequences):
        projector = projectors.get(row["camera_id"])
        if projector is None:
            continue
        projected = project_observation(row, projector, config)
        if projected["status"] != "ok" or projected["reassigned"] or row.get("global_track_id") is None:
            continue
        point = np.array([projected["x"], projected["y"]], dtype=np.float64)
        groups[(row["sequence_id"], row["source_version"], int(row["frame_index"]), int(row["global_track_id"]))].append(
            (row["camera_id"], point)
        )
    cells: dict[str, list[float]] = defaultdict(list)
    for items in groups.values():
        by_camera: dict[str, np.ndarray | None] = {}
        for camera_id, point in items:
            if camera_id in by_camera:
                by_camera[camera_id] = None
            else:
                by_camera[camera_id] = point
        present = [(camera_id, point) for camera_id, point in by_camera.items() if point is not None]
        for left in range(len(present)):
            for right in range(left + 1, len(present)):
                midpoint = 0.5 * (present[left][1] + present[right][1])
                distance = float(np.linalg.norm(present[left][1] - present[right][1]))
                index = np.floor((midpoint - CENTER) / CELL_M).astype(int)
                cells[f"{int(index[0])},{int(index[1])}"].append(distance)
    packed = {}
    for key, values in cells.items():
        array = np.asarray(values, dtype=np.float64)
        median = float(np.median(array))
        tail = float(np.mean(array > TAIL_M))
        packed[key] = {
            "count": int(len(array)),
            "median_m": median,
            "tail_15": tail,
            "level": _level(len(array), median, tail),
        }
    return {
        "cell_m": CELL_M,
        "center": CENTER.tolist(),
        "tail_m": TAIL_M,
        "source_sequences": sequences,
        "holdout_excluded": list(config["holdout_sequences"]),
        "cells": packed,
        "note": "等级来自建库序列的跨相机距离 不是关联准确率 样本不足的格子在跟踪里保持中性",
    }


def write_tracking_trust(path: Path | None = None) -> Path:
    target = path or (PROJECT_ROOT / "knowledge_base" / "tracking_trust_build.json")
    payload = build_tracking_trust()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return target
