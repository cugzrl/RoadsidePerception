"""BEV 网格与可信度"""

from __future__ import annotations

from typing import Any

import numpy as np


def grid_shape(config: dict[str, Any]) -> tuple[int, int, float, float, float]:
    size = float(config["grid_size_m"])
    min_x = float(config["grid_min_x"])
    min_y = float(config["grid_min_y"])
    nx = int(round((float(config["grid_max_x"]) - min_x) / size))
    ny = int(round((float(config["grid_max_y"]) - min_y) / size))
    return nx, ny, size, min_x, min_y


def cell_index(x_value: float, y_value: float, config: dict[str, Any]) -> tuple[int, int] | None:
    nx, ny, size, min_x, min_y = grid_shape(config)
    ix = int(np.floor((x_value - min_x) / size))
    iy = int(np.floor((y_value - min_y) / size))
    if ix < 0 or iy < 0 or ix >= nx or iy >= ny:
        return None
    return ix, iy


def cell_bounds(ix: int, iy: int, config: dict[str, Any]) -> tuple[float, float, float, float]:
    _nx, _ny, size, min_x, min_y = grid_shape(config)
    x0 = min_x + ix * size
    y0 = min_y + iy * size
    return x0, y0, x0 + size, y0 + size


def cell_id(ix: int, iy: int, config: dict[str, Any]) -> int:
    nx, _ny, _size, _min_x, _min_y = grid_shape(config)
    return iy * nx + ix


def confidence_label(clean_tracks: int) -> str:
    if clean_tracks >= 10:
        return "high"
    if clean_tracks >= 3:
        return "medium"
    if clean_tracks >= 1:
        return "low"
    return "unknown"
