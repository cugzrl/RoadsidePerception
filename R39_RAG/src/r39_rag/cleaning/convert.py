"""把各版本原始记录整理成统一观测、局部轨迹和全局轨迹"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from r39_rag.adapters.readers import (
    compact_aligned_record,
    iter_track_observations,
    load_global_assignment,
    parse_detection_name,
    parse_lid_map,
)
from r39_rag.cleaning.categories import apply_category, apply_pipeline_category
from r39_rag.cleaning.rules import (
    analyze_bbox,
    bbox_from_raw,
    centroid_from_raw,
    content_key,
    geometry_flags,
    point_from_raw,
)
from r39_rag.config import display_path, load_schema
from r39_rag.utils.json_stream import dump_json, dump_jsonl, iter_json_array, load_json

SCHEMA = load_schema()
OBSERVATION_KEYS = [item["name"] for item in SCHEMA["observation_fields"]]
TRACKLET_KEYS = [item["name"] for item in SCHEMA["tracklet_fields"]]
GLOBAL_KEYS = [item["name"] for item in SCHEMA["global_track_fields"]]


class ExampleBag:
    def __init__(self, limit: int = 5) -> None:
        self.limit = limit
        self.rows: dict[str, list[Any]] = defaultdict(list)

    def add(self, kind: str, item: Any) -> None:
        bucket = self.rows[kind]
        if len(bucket) < self.limit:
            bucket.append(json.loads(json.dumps(item, default=list)))

    def as_dict(self) -> dict[str, list[Any]]:
        return dict(self.rows)


def _ordered(keys: list[str], row: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in keys if key not in row]
    if missing:
        raise KeyError(f"缺少字段 {missing}")
    return {key: row[key] for key in keys}


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def _rel(path: Path | None) -> str | None:
    if path is None:
        return None
    return display_path(path)


def build_timebase(frame_caches: dict[str, Path], probes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    strides: list[int] = []
    saved: list[int] = []
    video_frames: list[int] = []
    fps_values: list[float | None] = []
    widths: list[int | None] = []
    heights: list[int | None] = []
    size_matches: dict[str, bool] = {}
    for camera_id, cache_path in frame_caches.items():
        meta = load_json(cache_path)
        probe = probes[camera_id]
        strides.append(int(meta["frame_stride"]))
        saved.append(int(meta["saved_frames"]))
        video_frames.append(int(meta["video_frames"]))
        fps_values.append(probe.get("fps"))
        widths.append(probe.get("width"))
        heights.append(probe.get("height"))
        video_path = probes[camera_id]["path"]
        actual_size = Path(video_path).stat().st_size if Path(video_path).exists() else None
        size_matches[camera_id] = actual_size == meta.get("video_size")
    consistent = (
        len(set(strides)) == 1
        and len(set(saved)) == 1
        and len(set(video_frames)) == 1
        and len(set(fps_values)) == 1
        and fps_values[0] not in (None, 0)
        and len(set(widths)) == 1
        and len(set(heights)) == 1
        and widths[0] is not None
    )
    result: dict[str, Any] = {
        "usable": False,
        "source_fps": None,
        "sample_fps": None,
        "frame_stride": None,
        "frame_count": None,
        "video_frames": None,
        "image_width": widths[0] if widths else None,
        "image_height": heights[0] if heights else None,
        "video_size_matches_frame_cache": size_matches,
        "timebase_source": "unavailable",
        "sync_status": "not_verified",
    }
    if not consistent:
        result["sync_status"] = "camera_metadata_inconsistent"
        return result
    source_fps = float(fps_values[0])
    stride = strides[0]
    result.update(
        {
            "usable": True,
            "source_fps": source_fps,
            "sample_fps": source_fps / stride,
            "frame_stride": stride,
            "frame_count": saved[0],
            "video_frames": video_frames[0],
            "timebase_source": "frame_cache_meta.frame_stride 与 ffprobe 帧率，抽帧起点按 0 估计",
            "sync_status": "sampling_pattern_consistent_clock_not_verified",
        }
    )
    return result


def stamp_frame(frame_index: int | None, timebase: dict[str, Any]) -> tuple[int | None, float | None, bool, str]:
    if frame_index is None or not timebase.get("usable"):
        return None, None, False, "unavailable"
    source_frame_index = int(frame_index) * int(timebase["frame_stride"])
    timestamp_ms = source_frame_index * 1000.0 / float(timebase["source_fps"])
    return source_frame_index, timestamp_ms, True, timebase["timebase_source"]


def _apply_class(raw_name: str | None, categories: dict[str, Any]) -> tuple[str | None, int | None, str, list[str]]:
    return apply_category(raw_name, categories["mappings"])


def assemble_observations(
    raw_rows: list[dict[str, Any]],
    *,
    site_id: str,
    sequence_id: str,
    source_version: str,
    categories: dict[str, Any],
    timebase: dict[str, Any],
    width: int | None,
    height: int | None,
    examples: ExampleBag,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    seen: set[tuple[Any, ...]] = set()
    kept: list[dict[str, Any]] = []
    duplicate_count = 0
    for row in raw_rows:
        key = content_key(
            row["camera_id"],
            row["frame_index"],
            row["local_track_id"],
            row.get("global_track_id"),
            row.get("class_name_raw"),
            row.get("bbox_xyxy"),
            row.get("score"),
        )
        if key in seen:
            duplicate_count += 1
            examples.add(
                "exact_duplicate",
                {
                    "camera_id": row["camera_id"],
                    "frame_index": row["frame_index"],
                    "local_track_id": row["local_track_id"],
                    "bbox_xyxy": row.get("bbox_xyxy"),
                },
            )
            continue
        seen.add(key)
        kept.append(row)

    grouped: dict[tuple[Any, Any, Any], list[int]] = defaultdict(list)
    for index, row in enumerate(kept):
        grouped[(row["camera_id"], row["frame_index"], row["local_track_id"])].append(index)

    observations: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    invalid_bbox = 0
    out_of_frame = 0
    unmapped = 0
    implausible = 0
    range_suspect = 0
    missing_global = 0
    manual_override = 0
    collision_groups = 0
    missing_timestamp = 0

    for (camera_id, frame_index, local_track_id), indexes in grouped.items():
        collision = local_track_id is not None and len(indexes) > 1
        if collision:
            collision_groups += 1
            conflicts.append(
                {
                    "type": "local_id_frame_collision",
                    "camera_id": camera_id,
                    "frame_index": frame_index,
                    "local_track_id": local_track_id,
                    "bboxes": [kept[index].get("bbox_xyxy") for index in indexes],
                    "global_track_ids": [kept[index].get("global_track_id") for index in indexes],
                }
            )
            examples.add(
                "local_id_frame_collision",
                {
                    "camera_id": camera_id,
                    "frame_index": frame_index,
                    "local_track_id": local_track_id,
                    "bboxes": [kept[index].get("bbox_xyxy") for index in indexes],
                },
            )
        for occurrence, index in enumerate(indexes):
            row = kept[index]
            if row.get("_pipeline_class"):
                class_name, class_id, mapping_status, class_flags = row["_pipeline_class"]
            else:
                class_name, class_id, mapping_status, class_flags = _apply_class(row.get("class_name_raw"), categories)
            flags = list(class_flags)
            flags.extend(analyze_bbox(row.get("bbox_xyxy"), width, height))
            flags.extend(geometry_flags(row.get("geometry_3d")))
            if collision:
                flags.append("local_id_frame_collision")
            if row.get("manual_override"):
                flags.append("manual_gid_override")
                manual_override += 1
            if row.get("manual_deleted"):
                flags.append("manual_deleted")
            if row.get("global_track_id") is None and row.get("expect_global_id"):
                flags.append("missing_global_id")
                missing_global += 1
            if "class_unmapped" in flags:
                unmapped += 1
            if "invalid_bbox" in flags:
                invalid_bbox += 1
            if "bbox_out_of_frame" in flags:
                out_of_frame += 1
            if "geometry_size_implausible" in flags:
                implausible += 1
            if "geometry_range_suspect" in flags:
                range_suspect += 1
            source_frame_index, timestamp_ms, estimated, timebase_source = stamp_frame(frame_index, timebase)
            if row.get("timestamp_ms_override") is not None:
                timestamp_ms = row["timestamp_ms_override"]
                estimated = False
                timebase_source = row.get("timebase_source_override") or timebase_source
                source_frame_index = row.get("source_frame_index_override")
            if timestamp_ms is None:
                missing_timestamp += 1
            geometry = row.get("geometry_3d")
            coordinate_frame = None
            if row.get("coordinate_frame"):
                coordinate_frame = row["coordinate_frame"]
            elif geometry:
                coordinate_frame = geometry.get("coordinate_frame")
            local_part = "none" if local_track_id is None else str(local_track_id)
            record_uid = f"{sequence_id}|{source_version}|{camera_id}|{frame_index}|{local_part}|{occurrence}"
            if row.get("record_uid_suffix"):
                record_uid = f"{record_uid}|{row['record_uid_suffix']}"
            observation = _ordered(
                OBSERVATION_KEYS,
                {
                    "site_id": site_id,
                    "sequence_id": sequence_id,
                    "source_version": source_version,
                    "camera_id": camera_id,
                    "frame_index": frame_index,
                    "source_frame_index": source_frame_index,
                    "timestamp_ms": timestamp_ms,
                    "timestamp_estimated": estimated,
                    "timebase_source": timebase_source,
                    "global_timestamp_ms": None,
                    "local_track_id": local_track_id,
                    "global_track_id": row.get("global_track_id"),
                    "original_global_track_id": row.get("original_global_track_id"),
                    "class_name": class_name,
                    "class_name_raw": row.get("class_name_raw"),
                    "class_id": class_id,
                    "class_mapping_status": mapping_status,
                    "score": row.get("score"),
                    "bbox_xyxy": row.get("bbox_xyxy"),
                    "pixel_centroid": row.get("pixel_centroid"),
                    "world_position": row.get("world_position"),
                    "coordinate_frame": coordinate_frame,
                    "geometry_3d": geometry,
                    "source_file": row["source_file"],
                    "source_record_index": row["source_record_index"],
                    "record_uid": record_uid,
                    "quality_flags": flags,
                    "manual_fix": bool(row.get("manual_fix")),
                    "source_label": row.get("source_label"),
                    "appearance_ref": row.get("appearance_ref"),
                    "detection_id": row.get("detection_id"),
                    "plate": row.get("plate"),
                },
            )
            observations.append(observation)
            if "invalid_bbox" in flags or "bbox_out_of_frame" in flags:
                anomalies.append(
                    {
                        "type": "bbox_anomaly",
                        "record_uid": record_uid,
                        "camera_id": camera_id,
                        "frame_index": frame_index,
                        "bbox_xyxy": row.get("bbox_xyxy"),
                        "quality_flags": flags,
                    }
                )
                examples.add("invalid_bbox" if "invalid_bbox" in flags else "bbox_out_of_frame", anomalies[-1])

    if implausible:
        sample = next((row for row in observations if "geometry_size_implausible" in row["quality_flags"]), None)
        if sample:
            examples.add(
                "geometry_size_implausible",
                {
                    "record_uid": sample["record_uid"],
                    "class_name_raw": sample["class_name_raw"],
                    "geometry_3d": sample["geometry_3d"],
                },
            )
    if range_suspect:
        sample = next((row for row in observations if "geometry_range_suspect" in row["quality_flags"]), None)
        if sample:
            examples.add(
                "geometry_range_suspect",
                {"record_uid": sample["record_uid"], "geometry_3d": sample["geometry_3d"]},
            )
    stats = {
        "input_records": len(raw_rows),
        "exact_duplicates_removed": duplicate_count,
        "written_observations": len(observations),
        "collision_groups": collision_groups,
        "invalid_bbox": invalid_bbox,
        "bbox_out_of_frame": out_of_frame,
        "class_unmapped": unmapped,
        "geometry_size_implausible": implausible,
        "geometry_range_suspect": range_suspect,
        "missing_global_id": missing_global,
        "manual_gid_override": manual_override,
        "missing_timestamp": missing_timestamp,
    }
    return observations, conflicts, anomalies, stats


def build_tracklets(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        if observation["local_track_id"] is None:
            continue
        grouped[(observation["camera_id"], observation["local_track_id"])].append(observation)
    tracklets: list[dict[str, Any]] = []
    for (camera_id, local_track_id), rows in grouped.items():
        ordered = sorted(rows, key=lambda item: (item["source_file"], item["source_record_index"]))
        frames = [row["frame_index"] for row in ordered]
        unique_frames = sorted(set(frames))
        flags: list[str] = []
        if any(frames[index] > frames[index + 1] for index in range(len(frames) - 1)):
            flags.append("source_frame_order_reversed")
        if len(unique_frames) < unique_frames[-1] - unique_frames[0] + 1:
            flags.append("frame_gap")
        if any("local_id_frame_collision" in row["quality_flags"] for row in rows):
            flags.append("local_id_frame_collision")
        raw_classes = sorted({row["class_name_raw"] for row in rows})
        if len(raw_classes) > 1:
            flags.append("class_inconsistent")
        global_ids = {row["global_track_id"] for row in rows}
        original_ids = {row["original_global_track_id"] for row in rows if row["original_global_track_id"] is not None}
        if len(global_ids) > 1:
            flags.append("global_id_inconsistent")
        first = rows[0]
        timestamps = [row["timestamp_ms"] for row in rows if row["timestamp_ms"] is not None]
        tracklets.append(
            _ordered(
                TRACKLET_KEYS,
                {
                    "sequence_id": first["sequence_id"],
                    "source_version": first["source_version"],
                    "camera_id": camera_id,
                    "local_track_id": local_track_id,
                    "global_track_id": next(iter(global_ids)) if len(global_ids) == 1 else None,
                    "original_global_track_id": next(iter(original_ids)) if len(original_ids) == 1 else None,
                    "class_name": first["class_name"],
                    "class_name_raw": first["class_name_raw"],
                    "class_id": first["class_id"],
                    "start_frame": unique_frames[0],
                    "end_frame": unique_frames[-1],
                    "start_timestamp_ms": min(timestamps) if timestamps else None,
                    "end_timestamp_ms": max(timestamps) if timestamps else None,
                    "observation_count": len(rows),
                    "unique_frame_count": len(unique_frames),
                    "source_files": sorted({row["source_file"] for row in rows}),
                    "quality_flags": flags,
                    "manual_fix": any(row["manual_fix"] for row in rows),
                },
            )
        )
    tracklets.sort(key=lambda item: (item["camera_id"], item["local_track_id"]))
    return tracklets


def build_global_tracks(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        if observation["global_track_id"] is None:
            continue
        grouped[observation["global_track_id"]].append(observation)
    tracks: list[dict[str, Any]] = []
    for global_track_id, rows in grouped.items():
        frames = sorted({row["frame_index"] for row in rows})
        raw_classes = sorted({row["class_name_raw"] for row in rows if row["class_name_raw"] is not None})
        flags: list[str] = []
        if len(raw_classes) > 1:
            flags.append("class_conflict")
        if any("manual_deleted" in row["quality_flags"] for row in rows):
            flags.append("manual_deleted")
        camera_counts: dict[str, int] = defaultdict(int)
        local_ids: set[str] = set()
        original_ids: set[int] = set()
        for row in rows:
            camera_counts[row["camera_id"]] += 1
            if row["local_track_id"] is not None:
                local_ids.add(f"{row['camera_id']}:{row['local_track_id']}")
            if row["original_global_track_id"] is not None:
                original_ids.add(row["original_global_track_id"])
        timestamps = [row["timestamp_ms"] for row in rows if row["timestamp_ms"] is not None]
        plates = {row["plate"] for row in rows if row["plate"]}
        first = rows[0]
        tracks.append(
            _ordered(
                GLOBAL_KEYS,
                {
                    "sequence_id": first["sequence_id"],
                    "source_version": first["source_version"],
                    "global_track_id": global_track_id,
                    "class_name": first["class_name"],
                    "class_names_raw": raw_classes,
                    "camera_ids": sorted(camera_counts),
                    "local_track_ids": sorted(local_ids),
                    "start_frame": frames[0],
                    "end_frame": frames[-1],
                    "start_timestamp_ms": min(timestamps) if timestamps else None,
                    "end_timestamp_ms": max(timestamps) if timestamps else None,
                    "observation_count": len(rows),
                    "unique_frame_count": len(frames),
                    "camera_observation_counts": dict(sorted(camera_counts.items())),
                    "original_global_track_ids": sorted(original_ids),
                    "source_files": sorted({row["source_file"] for row in rows}),
                    "quality_flags": flags,
                    "manual_fix": any(row["manual_fix"] for row in rows),
                    "plate": next(iter(plates)) if len(plates) == 1 else None,
                },
            )
        )
    tracks.sort(key=lambda item: item["global_track_id"])
    return tracks


def compare_assignment_to_frames(
    observations: list[dict[str, Any]], by_local: dict[tuple[str, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        if observation["local_track_id"] is None:
            continue
        grouped[(observation["camera_id"], int(observation["local_track_id"]))].append(observation)
    mismatches: list[dict[str, Any]] = []
    for key, summary in by_local.items():
        rows = grouped.get(key, [])
        if not rows:
            mismatches.append(
                {
                    "type": "summary_without_frames",
                    "camera_id": key[0],
                    "local_track_id": key[1],
                    "global_track_id": summary.get("global_obj_id"),
                    "start_frame": summary.get("start_frame"),
                    "end_frame": summary.get("end_frame"),
                    "num_frames": summary.get("num_frames"),
                }
            )
            continue
        frames = sorted({row["frame_index"] for row in rows})
        if frames[0] != summary.get("start_frame") or frames[-1] != summary.get("end_frame") or len(frames) != summary.get("num_frames"):
            mismatches.append(
                {
                    "type": "summary_frame_mismatch",
                    "camera_id": key[0],
                    "local_track_id": key[1],
                    "summary_start": summary.get("start_frame"),
                    "summary_end": summary.get("end_frame"),
                    "summary_num_frames": summary.get("num_frames"),
                    "observed_start": frames[0],
                    "observed_end": frames[-1],
                    "observed_unique_frames": len(frames),
                }
            )
    for key in grouped:
        if key not in by_local:
            mismatches.append({"type": "frames_without_global_assignment", "camera_id": key[0], "local_track_id": key[1]})
    return mismatches


def compare_global_versions(
    auto_map: dict[tuple[str, int], dict[str, Any]], manual_map: dict[tuple[str, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for key in sorted(set(auto_map) | set(manual_map)):
        auto = auto_map.get(key)
        manual = manual_map.get(key)
        base = {"camera_id": key[0], "local_track_id": key[1]}
        if auto and manual is None:
            conflicts.append({**base, "type": "only_in_auto", "auto_global_track_id": auto.get("global_obj_id")})
            continue
        if manual and auto is None:
            conflicts.append({**base, "type": "only_in_manual", "manual_global_track_id": manual.get("global_obj_id")})
            continue
        assert auto is not None and manual is not None
        if auto.get("global_obj_id") != manual.get("global_obj_id"):
            conflicts.append(
                {
                    **base,
                    "type": "gid_changed",
                    "auto_global_track_id": auto.get("global_obj_id"),
                    "manual_global_track_id": manual.get("global_obj_id"),
                    "original_global_track_id": manual.get("original_global_obj_id"),
                    "class_name": manual.get("class_name"),
                }
            )
        if auto.get("class_name") != manual.get("class_name"):
            conflicts.append(
                {
                    **base,
                    "type": "class_changed_between_versions",
                    "auto_class": auto.get("class_name"),
                    "manual_class": manual.get("class_name"),
                }
            )
        if manual.get("is_deleted"):
            conflicts.append({**base, "type": "manual_deleted", "manual_global_track_id": manual.get("global_obj_id")})
    return conflicts


def compare_text_map(
    text_map: dict[tuple[str, int], tuple[int, str]], json_map: dict[tuple[str, int], dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for key, (gid, class_name) in text_map.items():
        track = json_map.get(key)
        if track is None or track.get("global_obj_id") != gid or track.get("class_name") != class_name:
            conflicts.append(
                {
                    "type": "text_map_mismatch",
                    "map_name": label,
                    "camera_id": key[0],
                    "local_track_id": key[1],
                    "text_global_track_id": gid,
                    "text_class": class_name,
                    "json_global_track_id": None if track is None else track.get("global_obj_id"),
                    "json_class": None if track is None else track.get("class_name"),
                }
            )
    return conflicts


def _class_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        name = row.get(field)
        counts[str(name)] += 1
    return dict(sorted(counts.items()))


def _compare_identities(left: list[dict[str, Any]], right_path: Path, array_key: str | None, examples: ExampleBag, label: str) -> dict[str, Any]:
    """按身份和二维框的多重集合比较，避免只因顺序不同被记成冲突"""
    from collections import Counter

    def signature(item: dict[str, Any]) -> tuple[Any, ...]:
        bbox = item.get("bbox_xyxy")
        bbox_key = None if bbox is None else tuple(bbox)
        return (item["identity"], bbox_key)

    left_counts: Counter[tuple[Any, ...]] = Counter(signature(item) for item in left)
    right_counts: Counter[tuple[Any, ...]] = Counter()
    compared = 0
    for record in iter_json_array(right_path, array_key):
        other = compact_aligned_record(record, compared, label)
        right_counts[signature(other)] += 1
        compared += 1
    only_left = left_counts - right_counts
    only_right = right_counts - left_counts
    mismatches = sum(only_left.values()) + sum(only_right.values())
    for key, _count in list(only_left.items())[:2]:
        examples.add(label, {"side": "only_in_reference", "identity": key[0], "bbox_xyxy": key[1]})
    for key, _count in list(only_right.items())[:2]:
        examples.add(label, {"side": "only_in_other", "identity": key[0], "bbox_xyxy": key[1]})
    return {
        "compared": compared,
        "reference_count": len(left),
        "mismatches": mismatches,
        "duplicate_of_reference": mismatches == 0 and compared == len(left),
    }


def write_version(
    output_dir: Path,
    quarantine_dir: Path,
    observations: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    if quarantine_dir.exists():
        shutil.rmtree(quarantine_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(output_dir / "observations.jsonl", observations)
    dump_jsonl(output_dir / "tracklets.jsonl", build_tracklets(observations))
    dump_jsonl(output_dir / "global_tracks.jsonl", build_global_tracks(observations))
    dump_jsonl(quarantine_dir / "conflicts.jsonl", conflicts)
    dump_jsonl(quarantine_dir / "anomalies.jsonl", anomalies)


def convert_aligned(
    *,
    site_id: str,
    sequence_id: str,
    source_version: str,
    files: dict[str, Any],
    categories: dict[str, Any],
    timebase: dict[str, Any],
    output_dir: Path,
    quarantine_dir: Path,
) -> dict[str, Any]:
    examples = ExampleBag()
    bbox_info = files["bbox_info"]
    source_name = _rel(bbox_info) or str(bbox_info)
    pixels = [
        compact_aligned_record(record, index, source_name)
        for index, record in enumerate(iter_json_array(bbox_info, "pixel_boxes"))
    ]
    cameras = [
        compact_aligned_record(record, index, source_name)
        for index, record in enumerate(iter_json_array(bbox_info, "camera_boxes"))
    ]
    geometry_queues: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for item in cameras:
        geometry_queues[item["identity"]].append(item["geometry_3d"])
    geometry_mismatches = 0
    attached = 0
    for item in pixels:
        queue = geometry_queues.get(item["identity"]) or []
        if not queue:
            geometry_mismatches += 1
            examples.add("pixel_camera_identity_mismatch", {"pixel": item["identity"]})
            continue
        item["geometry_3d"] = queue.pop(0)
        attached += 1
    geometry_mismatches += sum(len(queue) for queue in geometry_queues.values())

    pixel_check = _compare_identities(pixels, files["pixel_file"], None, examples, "aligned_pixel_bbox")
    camera_identity_mismatches = 0
    camera_compared = 0
    for index, record in enumerate(iter_json_array(files["camera_file"], None)):
        camera_compared += 1
        if index >= len(cameras) or cameras[index]["identity"] != (
            record.get("camera_key"),
            record.get("local_obj_id"),
            record.get("global_obj_id"),
            record.get("frame_index"),
            record.get("class_name"),
        ):
            camera_identity_mismatches += 1
    camera_check = {
        "compared": camera_compared,
        "reference_count": len(cameras),
        "mismatches": camera_identity_mismatches + abs(camera_compared - len(cameras)),
        "duplicate_of_reference": camera_identity_mismatches == 0 and camera_compared == len(cameras),
    }
    projected_check = _compare_identities(pixels, files["projected_file"], "records", examples, "projected_pixel_bbox")

    raw_rows = []
    for item in pixels:
        raw_rows.append(
            {
                "camera_id": item["camera_id"],
                "local_track_id": item["local_track_id"],
                "global_track_id": item["global_track_id"],
                "original_global_track_id": None,
                "frame_index": item["frame_index"],
                "class_name_raw": item["class_name_raw"],
                "source_label": item["source_label"],
                "bbox_xyxy": item["bbox_xyxy"],
                "pixel_centroid": item["pixel_centroid"],
                "geometry_3d": item["geometry_3d"],
                "score": None,
                "source_file": item["source_file"],
                "source_record_index": item["source_record_index"],
                "manual_fix": False,
                "expect_global_id": True,
                "world_position": None,
            }
        )
    observations, conflicts, anomalies, stats = assemble_observations(
        raw_rows,
        site_id=site_id,
        sequence_id=sequence_id,
        source_version=source_version,
        categories=categories,
        timebase=timebase,
        width=timebase.get("image_width"),
        height=timebase.get("image_height"),
        examples=examples,
    )
    if geometry_mismatches:
        conflicts.append({"type": "pixel_camera_alignment", "mismatches": geometry_mismatches, "attached": attached})
    write_version(output_dir, quarantine_dir, observations, conflicts, anomalies)
    stats.update(
        {
            "source_version": source_version,
            "class_counts_raw": _class_counts(observations, "class_name_raw"),
            "class_counts_mapped": _class_counts(observations, "class_name"),
            "geometry_attached": attached,
            "geometry_alignment_mismatches": geometry_mismatches,
            "coordinate_frame": "unknown",
            "coordinate_status": "三维框尺寸明显偏离车辆物理尺寸，坐标系和单位未核实，未写入 world_position",
            "sibling_files": {
                "aligned_pixel_bbox": pixel_check,
                "aligned_camera_bbox": camera_check,
                "projected_pixel_bbox": projected_check,
            },
            "tracklets": _count_lines(output_dir / "tracklets.jsonl"),
            "global_tracks": _count_lines(output_dir / "global_tracks.jsonl"),
            "conflict_records": len(conflicts),
            "anomaly_records": len(anomalies),
            "examples": examples.as_dict(),
            "source_files": [source_name, _rel(files["pixel_file"]), _rel(files["camera_file"]), _rel(files["projected_file"])],
        }
    )
    return stats


def convert_tracks(
    *,
    site_id: str,
    sequence_id: str,
    source_version: str,
    files: dict[str, Any],
    categories: dict[str, Any],
    timebase: dict[str, Any],
    output_dir: Path,
    quarantine_dir: Path,
) -> dict[str, Any]:
    examples = ExampleBag()
    auto = load_global_assignment(files["auto_global"])
    manual = load_global_assignment(files["manual_global"])
    version_conflicts = compare_global_versions(auto["by_local"], manual["by_local"])
    text_conflicts = compare_text_map(parse_lid_map(files["lid_map"]), auto["by_local"], "lid_to_gid_mapping")
    text_conflicts.extend(parse_note for parse_note in compare_text_map(parse_lid_map(files["lid_map_manual"]), manual["by_local"], "lid_to_gid_mapping_manual"))
    for item in version_conflicts[:5]:
        examples.add(item["type"], item)
    for item in text_conflicts[:5]:
        examples.add("text_map_mismatch", item)

    raw_rows: list[dict[str, Any]] = []
    for camera_id, track_path in files["tracks"].items():
        source_name = _rel(track_path) or str(track_path)
        for item in iter_track_observations(track_path, source_name):
            if item["camera_id"] != camera_id:
                raise ValueError(f"轨迹文件相机与目录不一致: {track_path}")
            key = (item["camera_id"], int(item["local_track_id"]))
            manual_track = manual["by_local"].get(key)
            auto_track = auto["by_local"].get(key)
            global_id = None if manual_track is None else manual_track.get("global_obj_id")
            original = None if auto_track is None else auto_track.get("global_obj_id")
            deleted = bool(manual_track and manual_track.get("is_deleted"))
            changed = manual_track is not None and auto_track is not None and global_id != original
            raw_rows.append(
                {
                    **item,
                    "global_track_id": global_id,
                    "original_global_track_id": original,
                    "score": None,
                    "manual_fix": changed or deleted,
                    "manual_override": changed,
                    "manual_deleted": deleted,
                    "expect_global_id": True,
                    "world_position": None,
                }
            )
    observations, conflicts, anomalies, stats = assemble_observations(
        raw_rows,
        site_id=site_id,
        sequence_id=sequence_id,
        source_version=source_version,
        categories=categories,
        timebase=timebase,
        width=timebase.get("image_width"),
        height=timebase.get("image_height"),
        examples=examples,
    )
    summary_conflicts = compare_assignment_to_frames(observations, manual["by_local"])
    conflicts.extend(version_conflicts)
    conflicts.extend(text_conflicts)
    conflicts.extend(summary_conflicts)
    for item in summary_conflicts[:5]:
        examples.add(item["type"], item)
    write_version(output_dir, quarantine_dir, observations, conflicts, anomalies)
    stats.update(
        {
            "source_version": source_version,
            "class_counts_raw": _class_counts(observations, "class_name_raw"),
            "class_counts_mapped": _class_counts(observations, "class_name"),
            "coordinate_frame": "unverified_assumed_ground_plane",
            "coordinate_status": "ground_point_3d 使用假定地面 z=0，坐标原点和单位未核实，未写入 world_position",
            "auto_tracks": len(auto["by_local"]),
            "manual_tracks": len(manual["by_local"]),
            "gid_changed": sum(1 for item in version_conflicts if item["type"] == "gid_changed"),
            "only_in_auto": sum(1 for item in version_conflicts if item["type"] == "only_in_auto"),
            "only_in_manual": sum(1 for item in version_conflicts if item["type"] == "only_in_manual"),
            "manual_deleted_tracks": sum(1 for item in version_conflicts if item["type"] == "manual_deleted"),
            "text_map_mismatches": len(text_conflicts),
            "summary_mismatches": len(summary_conflicts),
            "tracklets": _count_lines(output_dir / "tracklets.jsonl"),
            "global_tracks": _count_lines(output_dir / "global_tracks.jsonl"),
            "conflict_records": len(conflicts),
            "anomaly_records": len(anomalies),
            "examples": examples.as_dict(),
            "manual_fix_sources": {
                "manual_json": _rel(files["manual_global"]),
                "auto_json": _rel(files["auto_global"]),
                "correction_txt": _rel(files["manual_corrections"]),
            },
            "source_files": [_rel(path) for path in files["tracks"].values()] + [_rel(files["manual_global"]), _rel(files["auto_global"])],
        }
    )
    return stats


def _feature_index(path: Path) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        header_text = file.read(1200)
    header = {
        "feature_matrix": None,
        "feature_dim": None,
    }
    for line in header_text.splitlines():
        if '"feature_matrix"' in line:
            header["feature_matrix"] = line.split(":", 1)[1].strip().strip('",')
        if '"feature_dim"' in line:
            header["feature_dim"] = int(line.split(":", 1)[1].strip().rstrip(","))
    index: dict[tuple[int, int], int] = {}
    for record in iter_json_array(path, "records"):
        index[(int(record["image_id"]), int(record["detection_index"]))] = int(record["feature_index"])
    return index, header


def _coco_count(path: Path) -> int:
    return sum(1 for _ in iter_json_array(path, None))


def convert_detr(
    *,
    site_id: str,
    sequence_id: str,
    source_version: str,
    files: dict[str, Any],
    categories: dict[str, Any],
    timebase: dict[str, Any],
    output_dir: Path,
    quarantine_dir: Path,
) -> dict[str, Any]:
    examples = ExampleBag()
    detections_path = files["detections"]
    source_name = _rel(detections_path) or str(detections_path)
    feature_map, feature_header = _feature_index(files["feature_metadata"])
    raw_rows: list[dict[str, Any]] = []
    image_count = 0
    for image in iter_json_array(detections_path, "images"):
        image_count += 1
        camera_id, frame_index = parse_detection_name(image["file_name"])
        for detection_index, detection in enumerate(image.get("detections") or []):
            feature_index = feature_map.get((int(image["image_id"]), detection_index))
            appearance = None
            if feature_index is not None:
                appearance = {
                    "feature_index": feature_index,
                    "feature_matrix": feature_header.get("feature_matrix"),
                    "feature_dim": feature_header.get("feature_dim"),
                    "metadata_file": _rel(files["feature_metadata"]),
                }
            raw_rows.append(
                {
                    "camera_id": camera_id,
                    "local_track_id": None,
                    "global_track_id": None,
                    "original_global_track_id": None,
                    "frame_index": frame_index,
                    "class_name_raw": detection.get("label"),
                    "source_label": detection.get("label"),
                    "bbox_xyxy": bbox_from_raw(detection.get("bbox_xyxy")),
                    "pixel_centroid": centroid_from_raw(None),
                    "geometry_3d": None,
                    "score": detection.get("score"),
                    "source_file": source_name,
                    "source_record_index": detection_index,
                    "manual_fix": False,
                    "world_position": None,
                    "appearance_ref": appearance,
                    "detection_id": f"{camera_id}:{frame_index}:{detection_index}",
                    "record_uid_suffix": f"img{image['image_id']}",
                }
            )
    observations, conflicts, anomalies, stats = assemble_observations(
        raw_rows,
        site_id=site_id,
        sequence_id=sequence_id,
        source_version=source_version,
        categories=categories,
        timebase=timebase,
        width=timebase.get("image_width"),
        height=timebase.get("image_height"),
        examples=examples,
    )
    coco_count = _coco_count(files["coco"])
    header = load_json_header_counts(detections_path)
    coco_status = {
        "coco_records": coco_count,
        "detections_header_count": header.get("total_detection_count"),
        "normalized_separately": False,
        "reason": "detections_coco.json 是同一批检测的 COCO xywh 导出，二维框以 detections.json 的 bbox_xyxy 为准",
    }
    if coco_count != header.get("total_detection_count"):
        conflicts.append({"type": "coco_count_mismatch", **coco_status})
        examples.add("coco_count_mismatch", coco_status)
    write_version(output_dir, quarantine_dir, observations, conflicts, anomalies)
    linked = sum(1 for row in observations if row["appearance_ref"] is not None)
    stats.update(
        {
            "source_version": source_version,
            "class_counts_raw": _class_counts(observations, "class_name_raw"),
            "class_counts_mapped": _class_counts(observations, "class_name"),
            "image_count": image_count,
            "appearance_links": linked,
            "coco": coco_status,
            "track_layer": "检测结果没有局部或全局轨迹 ID，不生成伪造轨迹",
            "coordinate_frame": None,
            "coordinate_status": "该版本只有二维框",
            "tracklets": 0,
            "global_tracks": 0,
            "conflict_records": len(conflicts),
            "anomaly_records": len(anomalies),
            "examples": examples.as_dict(),
            "source_files": [source_name, _rel(files["feature_metadata"]), _rel(files["coco"])],
        }
    )
    return stats


def load_json_header_counts(path: Path) -> dict[str, Any]:
    """只解析检测文件头部的计数字段"""
    with path.open("r", encoding="utf-8") as file:
        text = file.read(2500)
    result: dict[str, Any] = {}
    for key in ("total_detection_count", "processed_image_count", "discovered_image_count"):
        token = f'"{key}"'
        position = text.find(token)
        if position < 0:
            continue
        tail = text[position + len(token):]
        colon = tail.find(":")
        number = ""
        for char in tail[colon + 1:]:
            if char.isdigit():
                number += char
            elif number:
                break
        if number:
            result[key] = int(number)
    return result


def convert_pipeline_result(
    path: Path,
    *,
    output_dir: Path,
    quarantine_dir: Path,
    categories: dict[str, Any],
    site_id: str = "R39",
    sequence_id: str | None = None,
) -> dict[str, Any]:
    payload = load_json(path)
    sequence_id = sequence_id or f"pipeline_{payload['scene_id']}"
    source_version = "roadside_pipeline"
    source_name = _rel(path) or str(path)
    examples = ExampleBag()
    raw_rows: list[dict[str, Any]] = []
    for frame in payload.get("frames") or []:
        for detection in frame.get("detections") or []:
            class_name, class_id, status, flags = apply_pipeline_category(
                detection.get("class_name"), detection.get("class_id"), categories["pipeline_names"]
            )
            world = detection.get("world_position")
            world_point = None
            coordinate = None
            if isinstance(world, dict) and point_from_raw({"x": world.get("x"), "y": world.get("y"), "z": 0}) is not None:
                world_point = {"x": float(world["x"]), "y": float(world["y"])}
                coordinate = "road_homography_xy"
            raw_rows.append(
                {
                    "camera_id": frame["camera_id"],
                    "local_track_id": None,
                    "global_track_id": detection.get("global_track_id"),
                    "original_global_track_id": None,
                    "frame_index": frame["frame_index"],
                    "class_name_raw": detection.get("class_name"),
                    "_pipeline_class": (class_name, class_id, status, flags),
                    "source_label": detection.get("class_name"),
                    "bbox_xyxy": bbox_from_raw(detection.get("bbox_xyxy")),
                    "pixel_centroid": None,
                    "geometry_3d": None,
                    "score": detection.get("score"),
                    "source_file": source_name,
                    "source_record_index": 0,
                    "manual_fix": False,
                    "world_position": world_point,
                    "coordinate_frame": coordinate,
                    "detection_id": detection.get("detection_id"),
                    "plate": detection.get("plate") or None,
                    "timestamp_ms_override": frame.get("timestamp_ms"),
                    "timebase_source_override": "result.json timestamp_ms",
                    "source_frame_index_override": None,
                    "record_uid_suffix": detection.get("detection_id"),
                }
            )
    # 流水线类别已经是标准名，跳过历史映射表
    observations, conflicts, anomalies, stats = assemble_observations(
        raw_rows,
        site_id=site_id,
        sequence_id=sequence_id,
        source_version=source_version,
        categories=categories,
        timebase={"usable": False},
        width=_single_camera_size(payload, "width"),
        height=_single_camera_size(payload, "height"),
        examples=examples,
    )
    write_version(output_dir, quarantine_dir, observations, conflicts, anomalies)
    stats.update(
        {
            "source_version": source_version,
            "sequence_id": sequence_id,
            "scene_id": payload.get("scene_id"),
            "sample_fps": payload.get("fps"),
            "track_layer": "由 result.json 的检测框汇总，保留文件中的 timestamp_ms",
            "examples": examples.as_dict(),
        }
    )
    return stats


def _single_camera_size(payload: dict[str, Any], key: str) -> int | None:
    values = {camera.get(key) for camera in payload.get("cameras") or []}
    if len(values) == 1:
        value = next(iter(values))
        return int(value) if value is not None else None
    return None


