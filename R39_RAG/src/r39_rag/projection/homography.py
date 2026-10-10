"""与 STMC Projector.image_to_world 相同的地面投影"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class GroundProjector:
    """标定矩阵按 world 到 image 存放，投影到地面时用逆矩阵"""

    def __init__(self, matrix: np.ndarray, min_denominator: float) -> None:
        homography = np.asarray(matrix, dtype=np.float64)
        if homography.shape != (3, 3) or not np.isfinite(homography).all():
            raise ValueError("单应矩阵必须是 3x3 有限数值")
        self.image_from_world = homography
        self.world_from_image = np.linalg.inv(homography)
        self.min_denominator = min_denominator

    def image_to_world(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 Nx2 地面坐标，以及分母过小的布尔掩码"""
        array = np.asarray(points, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, 2)
        ones = np.ones((array.shape[0], 1), dtype=np.float64)
        homogeneous = np.concatenate([array, ones], axis=1)
        projected = homogeneous @ self.world_from_image.T
        denominator = projected[:, 2]
        valid = np.isfinite(projected).all(axis=1) & (np.abs(denominator) >= self.min_denominator)
        world = np.full((array.shape[0], 2), np.nan, dtype=np.float64)
        world[valid] = projected[valid, :2] / denominator[valid, None]
        return world, ~valid


def load_projectors(calibration_dir: Path, camera_ids: list[str], min_denominator: float) -> dict[str, GroundProjector]:
    projectors: dict[str, GroundProjector] = {}
    for camera_id in camera_ids:
        path = calibration_dir / camera_id / "calibration.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        projectors[camera_id] = GroundProjector(np.asarray(payload["homography"], dtype=np.float64), min_denominator)
    return projectors


def bottom_center(bbox: list[float] | tuple[float, ...]) -> tuple[float, float]:
    """xyxy 框底边中点，对应 STMC 中 tlwh 的 bottom center"""
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return (x1 + x2) / 2.0, y2
