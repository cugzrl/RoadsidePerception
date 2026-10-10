"""不同 JSON 的字段读取"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

from r39_rag.cleaning.rules import (
    bbox_from_raw,
    centroid_from_raw,
    geometry_from_camera_box,
    geometry_from_track_frame,
)
from r39_rag.utils.json_stream import iter_json_array, load_json


def identity_of(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("camera_key"),
        record.get("local_obj_id"),
        record.get("global_obj_id"),
        record.get("frame_index"),
        record.get("class_name"),
    )


def iter_named_or_top_array(path: Path, array_key: str | None) -> Iterator[dict[str, Any]]:
    yield from iter_json_array(path, array_key)


def compact_aligned_record(record: dict[str, Any], index: int, source_file: str) -> dict[str, Any]:
    bbox_raw = record.get("bbox_xyxy")
    if bbox_raw is None:
        bbox_raw = record.get("projected_pixel_bbox_xyxy")
    return {
        "camera_id": record.get("camera_key"),
        "local_track_id": record.get("local_obj_id"),
        "global_track_id": record.get("global_obj_id"),
        "frame_index": record.get("frame_index"),
        "class_name_raw": record.get("class_name"),
        "source_label": record.get("label"),
        "bbox_xyxy": bbox_from_raw(bbox_raw),
        "pixel_centroid": centroid_from_raw(record.get("pixel_centroid")),
        "geometry_3d": geometry_from_camera_box(record.get("camera_box_3d")),
        "source_file": source_file,
        "source_record_index": index,
        "identity": identity_of(record),
    }


def load_track_file(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict) or "tracks" not in data:
        raise ValueError(f"不是 SAM3 轨迹文件: {path}")
    return data


def iter_track_observations(path: Path, source_file: str) -> Iterator[dict[str, Any]]:
    payload = load_track_file(path)
    camera_id = payload.get("camera_key")
    tracks = payload.get("tracks") or {}
    for track in tracks.values():
        local_id = track.get("obj_id")
        assumed_height = track.get("assumed_height_m")
        for index, frame in enumerate(track.get("frames") or []):
            yield {
                "camera_id": camera_id,
                "local_track_id": local_id,
                "frame_index": frame.get("frame_index"),
                "class_name_raw": track.get("class_name"),
                "source_label": track.get("label"),
                "bbox_xyxy": bbox_from_raw(frame.get("bbox_xyxy")),
                "pixel_centroid": centroid_from_raw(frame.get("pixel_centroid")),
                "geometry_3d": geometry_from_track_frame(frame, assumed_height),
                "source_file": source_file,
                "source_record_index": index,
            }


_LID_LINE = re.compile(r"LID\s+(\d+)\s+->\s+GID\s+(\d+)\s+\(([^)]+)\)")


def parse_lid_map(path: Path) -> dict[tuple[str, int], tuple[int, str]]:
    current = ""
    mapping: dict[tuple[str, int], tuple[int, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text.startswith("[") and text.endswith("]"):
            current = text[1:-1]
            continue
        matched = _LID_LINE.match(text)
        if matched and current:
            mapping[(current, int(matched.group(1)))] = (int(matched.group(2)), matched.group(3))
    return mapping


def load_global_assignment(path: Path) -> dict[str, Any]:
    data = load_json(path)
    tracks = data.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError(f"全局 ID 文件缺少 tracks: {path}")
    by_local: dict[tuple[str, int], dict[str, Any]] = {}
    for track in tracks:
        by_local[(track["camera_key"], int(track["local_obj_id"]))] = track
    return {"payload": data, "by_local": by_local}


def parse_detection_name(file_name: str) -> tuple[str, int]:
    path = Path(file_name)
    camera_id = path.parts[0]
    frame_index = int(path.stem)
    return camera_id, frame_index
