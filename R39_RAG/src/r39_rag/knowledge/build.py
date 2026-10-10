"""从统一观测提取空间与轨迹知识"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from r39_rag.config import PROJECT_ROOT, load_sources
from r39_rag.knowledge.spatial import cell_bounds, cell_id, cell_index, confidence_label, grid_shape
from r39_rag.knowledge.store import write_arrays, write_database
from r39_rag.knowledge.trajectory import build_prototypes
from r39_rag.projection.evaluate import iter_observations
from r39_rag.projection.homography import load_projectors
from r39_rag.projection.quality import project_observation


def _direction_bin(dx: float, dy: float) -> int:
    angle = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    return int(angle // 45.0) % 8


def _mode(values: list[str]) -> str | None:
    if not values:
        return None
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def build_knowledge(config: dict[str, Any], status_path: Path) -> dict[str, Any]:
    if not status_path.exists():
        raise RuntimeError("还没有投影验证结果，请先运行 verify_bev.py")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if not status.get("allow_fusion"):
        raise RuntimeError("投影验证未通过，不融合四路坐标")
    sources = load_sources()
    projectors = load_projectors(config["calibration_dir"], sources["cameras"], float(config["min_denominator"]))
    sequence_ids = list(config["build_sequences"])
    tracks: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in iter_observations(sequence_ids):
        local_id = row.get("local_track_id")
        projector = projectors.get(row["camera_id"])
        if local_id is None or projector is None:
            continue
        projected = project_observation(row, projector, config)
        if projected["status"] != "ok":
            continue
        index = cell_index(projected["x"], projected["y"], config)
        if index is None:
            continue
        tracks[(row["sequence_id"], row["source_version"], row["camera_id"], int(local_id))].append(
            {
                "frame": int(row["frame_index"]),
                "time": row.get("timestamp_ms"),
                "x": projected["x"],
                "y": projected["y"],
                "ix": index[0],
                "iy": index[1],
                "class_name": row.get("class_name"),
                "global_id": None if row.get("global_track_id") is None else int(row["global_track_id"]),
                "reassigned": projected["reassigned"],
            }
        )

    nx, ny, _size, _min_x, _min_y = grid_shape(config)
    observation_count = np.zeros((ny, nx), dtype=np.int32)
    clean_track_grid = np.zeros((ny, nx), dtype=np.int32)
    cell_obs: dict[tuple[int, int], int] = defaultdict(int)
    cell_local: dict[tuple[int, int], set[tuple[Any, ...]]] = defaultdict(set)
    cell_clean: dict[tuple[int, int], set[tuple[Any, ...]]] = defaultdict(set)
    cell_reassigned: dict[tuple[int, int], set[tuple[Any, ...]]] = defaultdict(set)
    cell_global: dict[tuple[int, int], set[tuple[Any, ...]]] = defaultdict(set)
    cell_class: dict[tuple[int, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    cell_direction: dict[tuple[int, int], dict[int, int]] = defaultdict(lambda: defaultdict(int))
    cell_speed: dict[tuple[int, int], list[float]] = defaultdict(list)
    cell_camera_obs: dict[tuple[int, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    cell_camera_tracks: dict[tuple[int, int], dict[str, set[tuple[Any, ...]]]] = defaultdict(lambda: defaultdict(set))
    cell_sequences: dict[tuple[int, int], set[str]] = defaultdict(set)
    transitions: dict[tuple[tuple[int, int], tuple[int, int]], list[float]] = defaultdict(list)
    co_distances: dict[tuple[str, str], list[float]] = defaultdict(list)
    transfer_dt: dict[tuple[str, str], list[float]] = defaultdict(list)
    prototype_tracks: list[dict[str, Any]] = []
    used_observations = 0
    clean_tracks = 0
    reassigned_tracks = 0
    jumped_tracks = 0
    global_points: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)

    for key, points in tracks.items():
        sequence_id, source_version, camera_id, _local_id = key
        ordered = sorted(points, key=lambda item: (item["frame"], item["time"] or 0))
        reassigned = any(item["reassigned"] for item in ordered)
        if reassigned:
            reassigned_tracks += 1
        else:
            clean_tracks += 1
        jumped = False
        speeds: list[float] = []
        cell_steps: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
        cell_classes: dict[tuple[int, int], list[str]] = defaultdict(list)
        for index, item in enumerate(ordered):
            cell = (item["ix"], item["iy"])
            used_observations += 1
            cell_obs[cell] += 1
            observation_count[item["iy"], item["ix"]] += 1
            cell_local[cell].add(key)
            cell_sequences[cell].add(sequence_id)
            cell_camera_obs[cell][camera_id] += 1
            cell_camera_tracks[cell][camera_id].add(key)
            if item["class_name"]:
                cell_classes[cell].append(item["class_name"])
            if item["global_id"] is not None and not reassigned:
                cell_global[cell].add((sequence_id, source_version, item["global_id"]))
                global_points[(sequence_id, source_version, item["global_id"])].append({**item, "camera_id": camera_id})
            if reassigned:
                cell_reassigned[cell].add(key)
            else:
                cell_clean[cell].add(key)
            if index == 0:
                continue
            before = ordered[index - 1]
            frame_gap = item["frame"] - before["frame"]
            dx = item["x"] - before["x"]
            dy = item["y"] - before["y"]
            distance = float(np.hypot(dx, dy))
            limit = float(config["jump_step_m"]) * max(frame_gap, 1)
            step_jump = frame_gap <= 0 or distance > limit
            if step_jump:
                jumped = True
            dt_ms = None
            if item["time"] is not None and before["time"] is not None:
                dt_ms = float(item["time"]) - float(before["time"])
            before_cell = (before["ix"], before["iy"])
            if not reassigned and not step_jump and frame_gap == 1 and dt_ms and dt_ms > 0:
                speeds.append(distance / (dt_ms / 1000.0))
                if distance >= float(config["direction_min_step_m"]):
                    cell_steps[before_cell].append((dx, dy))
            if (
                not reassigned
                and not step_jump
                and cell != before_cell
                and 0 < frame_gap <= int(config["max_transition_gap_frames"])
                and dt_ms is not None
            ):
                transitions[(before_cell, cell)].append(dt_ms)
        if jumped:
            jumped_tracks += 1
        if not reassigned:
            for cell, names in cell_classes.items():
                chosen = _mode(names)
                if chosen:
                    cell_class[cell][chosen] += 1
            for cell, vectors in cell_steps.items():
                median_vector = np.median(np.asarray(vectors), axis=0)
                cell_direction[cell][_direction_bin(float(median_vector[0]), float(median_vector[1]))] += 1
        median_speed = None if not speeds else float(np.median(speeds))
        if median_speed is not None and not reassigned:
            for cell in {(item["ix"], item["iy"]) for item in ordered}:
                cell_speed[cell].append(median_speed)
        prototype_tracks.append(
            {
                "sequence_id": sequence_id,
                "camera_id": camera_id,
                "local_track_id": key[3],
                "reassigned": reassigned,
                "jumped": jumped,
                "median_speed": median_speed,
                "points": [[item["x"], item["y"]] for item in ordered],
            }
        )

    _collect_cross_camera(global_points, config, co_distances, transfer_dt)
    cell_rows = []
    occupied = 0
    for iy in range(ny):
        for ix in range(nx):
            cell = (ix, iy)
            obs = cell_obs.get(cell, 0)
            if obs == 0:
                continue
            occupied += 1
            clean = len(cell_clean.get(cell, ()))
            clean_track_grid[iy, ix] = clean
            speed_array = np.asarray(cell_speed.get(cell, []), dtype=np.float64)
            x0, y0, x1, y1 = cell_bounds(ix, iy, config)
            camera_counts = {
                camera_id: {
                    "observations": cell_camera_obs[cell][camera_id],
                    "tracks": len(cell_camera_tracks[cell][camera_id]),
                }
                for camera_id in cell_camera_obs[cell]
            }
            cell_rows.append(
                (
                    cell_id(ix, iy, config),
                    ix,
                    iy,
                    x0,
                    y0,
                    x1,
                    y1,
                    obs,
                    len(cell_local[cell]),
                    clean,
                    len(cell_global.get(cell, ())),
                    len(cell_reassigned.get(cell, ())),
                    json.dumps(dict(cell_class.get(cell, {})), ensure_ascii=False),
                    json.dumps({str(bin_id): count for bin_id, count in sorted(cell_direction.get(cell, {}).items())}),
                    None if speed_array.size == 0 else float(np.percentile(speed_array, 50)),
                    None if speed_array.size == 0 else float(np.percentile(speed_array, 90)),
                    json.dumps(camera_counts, ensure_ascii=False),
                    "unknown",
                    confidence_label(clean),
                    json.dumps(sorted(cell_sequences[cell]), ensure_ascii=False),
                )
            )
    transition_rows = []
    for (start, end), times in transitions.items():
        array = np.asarray(times, dtype=np.float64)
        transition_rows.append((cell_id(*start, config), cell_id(*end, config), int(array.size), float(np.median(array)), confidence_label(int(array.size))))
    prototypes = build_prototypes(prototype_tracks, config)
    prototype_rows = []
    prototype_points = np.zeros((len(prototypes), int(config["prototype_samples"]), 2), dtype=np.float64)
    for index, item in enumerate(prototypes):
        prototype_points[index] = item["points"]
        prototype_rows.append(
            (
                index,
                item["kind"],
                item["support_tracks"],
                json.dumps(item["support_sequences"], ensure_ascii=False),
                json.dumps(item["camera_counts"], ensure_ascii=False),
                item["median_speed"],
                item["confidence"],
                item["exemplar_sequence"],
                item["exemplar_camera"],
                item["exemplar_local_track_id"],
            )
        )
    cross_rows = _cross_rows(co_distances, transfer_dt)
    knowledge_dir = PROJECT_ROOT / "knowledge_base"
    meta = {
        "coordinate_frame": config["coordinate_frame"],
        "homography_convention": config["homography_convention"],
        "grid_size_m": config["grid_size_m"],
        "build_sequences": sequence_ids,
        "holdout_sequences": list(config["holdout_sequences"]),
        "projection_median_m": status.get("median_m"),
        "observation_reliability": "unknown",
        "note": "观测可靠性没有独立证据，相机字段只表示历史覆盖。不同序列的全局 ID 不合并。自动拆分轨迹不进入典型路径和高可信跨相机统计。",
    }
    write_database(
        knowledge_dir / "r39_knowledge.sqlite",
        {
            "meta": meta,
            "cells": cell_rows,
            "transitions": transition_rows,
            "prototypes": prototype_rows,
            "cross_camera": cross_rows,
        },
    )
    write_arrays(knowledge_dir / "knowledge_arrays.npz", observation_count, clean_track_grid, prototype_points, config)
    summary = {
        "observations": used_observations,
        "local_tracks": len(tracks),
        "clean_tracks": clean_tracks,
        "reassigned_tracks": reassigned_tracks,
        "jumped_tracks": jumped_tracks,
        "occupied_cells": occupied,
        "transitions": len(transition_rows),
        "prototypes": len(prototype_rows),
        "co_observation_pairs": len(co_distances),
        "temporal_transfers": len(transfer_dt),
        "prototype_kinds": {kind: sum(1 for row in prototype_rows if row[1] == kind) for kind in ("straight", "left", "right", "other")},
    }
    _write_report(PROJECT_ROOT / "reports" / "knowledge_report.md", summary, config, prototype_rows, cross_rows)
    return summary


def _collect_cross_camera(
    global_points: dict[tuple[str, str, int], list[dict[str, Any]]],
    config: dict[str, Any],
    co_distances: dict[tuple[str, str], list[float]],
    transfer_dt: dict[tuple[str, str], list[float]],
) -> None:
    for items in global_points.values():
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_frame[item["frame"]].append(item)
        frames = sorted(by_frame)
        for frame in frames:
            seen: dict[str, dict[str, Any] | None] = {}
            for item in by_frame[frame]:
                if item["camera_id"] in seen:
                    seen[item["camera_id"]] = None
                else:
                    seen[item["camera_id"]] = item
            present = [(camera_id, item) for camera_id, item in seen.items() if item is not None]
            for left in range(len(present)):
                for right in range(left + 1, len(present)):
                    distance = float(np.hypot(present[left][1]["x"] - present[right][1]["x"], present[left][1]["y"] - present[right][1]["y"]))
                    if distance <= float(config["co_observation_max_gap_m"]):
                        pair = tuple(sorted((present[left][0], present[right][0])))
                        co_distances[pair].append(distance)
        singles = []
        for frame in frames:
            cameras = {item["camera_id"] for item in by_frame[frame]}
            if len(cameras) == 1:
                singles.append(by_frame[frame][0])
        for index in range(1, len(singles)):
            before = singles[index - 1]
            current = singles[index]
            if before["camera_id"] == current["camera_id"] or before["time"] is None or current["time"] is None:
                continue
            dt_ms = float(current["time"]) - float(before["time"])
            distance = float(np.hypot(current["x"] - before["x"], current["y"] - before["y"]))
            if 0 < dt_ms <= float(config["transfer_max_gap_s"]) * 1000 and distance <= float(config["transfer_max_step_m"]):
                transfer_dt[(before["camera_id"], current["camera_id"])].append(dt_ms)


def _cross_rows(
    co_distances: dict[tuple[str, str], list[float]],
    transfer_dt: dict[tuple[str, str], list[float]],
) -> list[tuple[Any, ...]]:
    rows = []
    for pair, values in sorted(co_distances.items()):
        array = np.asarray(values, dtype=np.float64)
        rows.append((pair[0], pair[1], "co_observation", int(array.size), float(np.median(array)), None, confidence_label(int(array.size))))
    for pair, values in sorted(transfer_dt.items()):
        array = np.asarray(values, dtype=np.float64)
        rows.append((pair[0], pair[1], "temporal_transfer", int(array.size), None, float(np.median(array)), confidence_label(int(array.size))))
    return rows


def _write_report(
    path: Path,
    summary: dict[str, Any],
    config: dict[str, Any],
    prototypes: list[tuple[Any, ...]],
    cross_rows: list[tuple[Any, ...]],
) -> None:
    lines = [
        "# 时空知识库",
        "",
        f"建库序列：{', '.join(config['build_sequences'])}。",
        f"留出序列：{', '.join(config['holdout_sequences'])}。留出序列不进入知识库，避免后续感知实验直接看到这段历史。",
        "",
        f"网格 {config['grid_size_m']} m，范围 X [{config['grid_min_x']}, {config['grid_max_x']}]、Y [{config['grid_min_y']}, {config['grid_max_y']}]。坐标系 `{config['coordinate_frame']}`。",
        "",
        "## 规模",
        "",
        f"- 进入网格的观测 {summary['observations']}",
        f"- 局部轨迹 {summary['local_tracks']}，未拆分 {summary['clean_tracks']}，自动拆分 {summary['reassigned_tracks']}，含跳变 {summary['jumped_tracks']}",
        f"- 有观测的网格 {summary['occupied_cells']}",
        f"- 区域转移 {summary['transitions']}",
        f"- 典型路径 {summary['prototypes']}，类别计数 {summary['prototype_kinds']}",
        f"- 重叠视野共同观测的相机对 {summary['co_observation_pairs']}，非重叠时序转移的有向相机对 {summary['temporal_transfers']}",
        "",
        "观测次数和轨迹数分开保存。停留多帧不会单独抬高轨迹数。没有用检测框数量估计漏检或误检。相机覆盖不等于观测可靠性，可靠性字段为 unknown。",
        "",
        "## 典型路径支持数最高的几条",
        "",
        "| ID | 类型 | 轨迹数 | 可信度 | 中位速度 m/s | 示例相机 |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ]
    for row in prototypes[:8]:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[6]} | {row[5]} | {row[8]} |")
    lines.extend(
        [
            "",
            "## 跨相机关系",
            "",
            "| 相机 A | 相机 B | 关系 | 支持数 | 中位距离 m | 中位时间 ms | 可信度 |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in cross_rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
