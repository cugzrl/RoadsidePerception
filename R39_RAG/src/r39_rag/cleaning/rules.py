"""数值、框和三维几何的质量规则"""

from __future__ import annotations

import math
from typing import Any


def is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def as_float(value: Any) -> float | None:
    if not is_finite_number(value):
        return None
    return float(value)


def bbox_from_raw(raw: Any) -> list[float] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        values = [raw.get("x0"), raw.get("y0"), raw.get("x1"), raw.get("y1")]
    elif isinstance(raw, (list, tuple)) and len(raw) == 4:
        values = list(raw)
    else:
        return None
    if not all(is_finite_number(value) for value in values):
        return None
    return [float(value) for value in values]


def centroid_from_raw(raw: Any) -> list[float] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        values = [raw.get("x"), raw.get("y")]
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        values = list(raw)
    else:
        return None
    if not all(is_finite_number(value) for value in values):
        return None
    return [float(value) for value in values]


def point_from_raw(raw: Any) -> list[float] | None:
    if not isinstance(raw, dict):
        return None
    values = [raw.get("x"), raw.get("y"), raw.get("z")]
    if not all(is_finite_number(value) for value in values):
        return None
    return [float(value) for value in values]


def analyze_bbox(bbox: list[float] | None, width: int | None, height: int | None) -> list[str]:
    if bbox is None:
        return []
    x1, y1, x2, y2 = bbox
    flags: list[str] = []
    if x2 <= x1 or y2 <= y1:
        flags.append("invalid_bbox")
    if width is not None and height is not None:
        if x1 < -1 or y1 < -1 or x2 > width + 1 or y2 > height + 1:
            flags.append("bbox_out_of_frame")
    return flags


def geometry_from_camera_box(camera_box: Any) -> dict[str, Any] | None:
    if not isinstance(camera_box, dict):
        return None
    bbox_3d = camera_box.get("bbox_3d") if isinstance(camera_box.get("bbox_3d"), dict) else {}
    center = point_from_raw(bbox_3d.get("center")) or point_from_raw(camera_box.get("center_point_3d"))
    size = bbox_3d.get("size") if isinstance(bbox_3d.get("size"), dict) else {}
    length = as_float(size.get("length"))
    width = as_float(size.get("width"))
    height = as_float(size.get("height"))
    size_lwh = [length, width, height] if any(value is not None for value in (length, width, height)) else None
    return {
        "type": "camera_box_3d",
        "center": center,
        "size_lwh": size_lwh,
        "ground_point": point_from_raw(camera_box.get("ground_point_3d")),
        "coordinate_frame": "unknown",
        "unit": "unknown",
        "height_source": "not_verified",
        "corners_stored": False,
    }


def geometry_from_track_frame(frame: dict[str, Any], assumed_height: Any) -> dict[str, Any] | None:
    ground = point_from_raw(frame.get("ground_point_3d"))
    center = point_from_raw(frame.get("center_point_3d"))
    if ground is None and center is None:
        return None
    return {
        "type": "ground_point_3d",
        "center": center,
        "size_lwh": None,
        "ground_point": ground,
        "coordinate_frame": "unverified_assumed_ground_plane",
        "unit": "unknown",
        "height_source": "assumed_by_class",
        "assumed_height_m": as_float(assumed_height),
        "corners_stored": False,
    }


def geometry_flags(geometry: dict[str, Any] | None) -> list[str]:
    if not geometry:
        return []
    flags: list[str] = []
    size = geometry.get("size_lwh")
    if isinstance(size, list) and len(size) == 3 and size[0] is not None and size[1] is not None:
        length, width = size[0], size[1]
        if length == 0 or width == 0 or length > 12 or width > 3.5:
            flags.append("geometry_size_implausible")
    points = [geometry.get("center"), geometry.get("ground_point")]
    for point in points:
        if isinstance(point, list) and len(point) >= 2:
            if abs(point[0]) > 150 or abs(point[1]) > 150:
                flags.append("geometry_range_suspect")
                break
    return flags


def content_key(
    camera_id: str,
    frame_index: int,
    local_track_id: int | None,
    global_track_id: int | None,
    class_name_raw: str | None,
    bbox: list[float] | None,
    score: float | None,
) -> tuple[Any, ...]:
    bbox_key = None if bbox is None else tuple(round(value, 3) for value in bbox)
    score_key = None if score is None else round(float(score), 6)
    return (camera_id, frame_index, local_track_id, global_track_id, class_name_raw, bbox_key, score_key)
