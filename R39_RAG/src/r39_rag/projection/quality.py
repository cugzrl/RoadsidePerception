"""单点投影是否能进入知识库"""

from __future__ import annotations

from typing import Any

import numpy as np

from r39_rag.projection.homography import GroundProjector, bottom_center


def project_observation(row: dict[str, Any], projector: GroundProjector, config: dict[str, Any]) -> dict[str, Any]:
    flags = row.get("quality_flags") or []
    bbox = row.get("bbox_xyxy")
    result = {
        "status": "ok",
        "pixel_x": None,
        "pixel_y": None,
        "x": None,
        "y": None,
        "reassigned": "id_reassigned" in flags or row.get("raw_local_track_id") is not None or row.get("raw_global_track_id") is not None,
    }
    if "invalid_bbox" in flags or not isinstance(bbox, list) or len(bbox) != 4:
        result["status"] = "invalid_bbox"
        return result
    if "bbox_out_of_frame" in flags:
        result["status"] = "bbox_out_of_frame"
        return result
    pixel_x, pixel_y = bottom_center(bbox)
    result["pixel_x"] = pixel_x
    result["pixel_y"] = pixel_y
    width = int(config["image_width"])
    height = int(config["image_height"])
    border = float(config["border_px"])
    if pixel_x < border or pixel_y < border or pixel_x > width - border or pixel_y > height - border:
        result["status"] = "border_contact"
        return result
    world, singular = projector.image_to_world(np.asarray([[pixel_x, pixel_y]], dtype=np.float64))
    if singular[0] or not np.isfinite(world[0]).all():
        result["status"] = "non_finite"
        return result
    x_value = float(world[0, 0])
    y_value = float(world[0, 1])
    result["x"] = x_value
    result["y"] = y_value
    if abs(x_value) > float(config["max_abs_x_m"]) or abs(y_value) > float(config["max_abs_y_m"]):
        result["status"] = "out_of_bounds"
        return result
    return result
