"""知识库图"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


CAMERA_COLORS = {
    "R39_Aw_CamS": "#e41a1c",
    "R39_Bn_CamW": "#377eb8",
    "R39_Ce_CamN": "#4daf4a",
    "R39_Ds_CamE": "#984ea3",
}


def use_chinese_font(font_path: str) -> None:
    font_manager.fontManager.addfont(font_path)
    name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = name
    plt.rcParams["axes.unicode_minus"] = False


def _save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_error_hist(distances: np.ndarray, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    clipped = distances[np.isfinite(distances)]
    axis.hist(np.clip(clipped, 0, 40), bins=40, color="#4c78a8")
    axis.set_xlabel("跨相机地面距离 (m)")
    axis.set_ylabel("配对数量")
    axis.set_title("同一帧同一全局 ID 的投影误差，超过 40 m 并入末箱")
    _save(figure, path)


def plot_camera_scatter(samples: dict[str, np.ndarray], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 8))
    for camera_id, points in samples.items():
        if len(points) == 0:
            continue
        axis.scatter(points[:, 0], points[:, 1], s=2, alpha=0.35, c=CAMERA_COLORS.get(camera_id, "#333333"), label=camera_id)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_title("四路相机可用地面投影")
    axis.legend(markerscale=4, fontsize=8)
    _save(figure, path)


def plot_heatmap(grid: np.ndarray, config: dict[str, Any], path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    extent = [float(config["grid_min_x"]), float(config["grid_max_x"]), float(config["grid_min_y"]), float(config["grid_max_y"])]
    image = axis.imshow(np.log1p(grid), origin="lower", extent=extent, cmap="YlOrRd", aspect="equal")
    figure.colorbar(image, ax=axis, label="log(1 + 数量)")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_title(title)
    _save(figure, path)


def plot_prototypes(points: np.ndarray, kinds: list[str], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 8))
    kind_color = {"straight": "#1f77b4", "left": "#2ca02c", "right": "#d62728", "other": "#7f7f7f"}
    shown: set[str] = set()
    for index, kind in enumerate(kinds):
        label = kind if kind not in shown else None
        shown.add(kind)
        axis.plot(points[index, :, 0], points[index, :, 1], color=kind_color.get(kind, "#333333"), lw=1.6, alpha=0.85, label=label)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_title("典型行驶路径")
    axis.legend()
    _save(figure, path)


def plot_retrieval(
    grid: np.ndarray,
    config: dict[str, Any],
    query_xy: tuple[float, float],
    prototype: np.ndarray | None,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    extent = [float(config["grid_min_x"]), float(config["grid_max_x"]), float(config["grid_min_y"]), float(config["grid_max_y"])]
    axis.imshow(np.log1p(grid), origin="lower", extent=extent, cmap="Greys", aspect="equal")
    if prototype is not None and len(prototype):
        axis.plot(prototype[:, 0], prototype[:, 1], color="#d62728", lw=2.2, label="匹配到的典型路径")
    axis.scatter([query_xy[0]], [query_xy[1]], c="#1f77b4", s=40, label="查询位置")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_title("检索示例")
    axis.legend()
    _save(figure, path)
