"""从四段建库序列切分Key-Value运动片段"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from r39_rag.config import load_knowledge, load_sources  # noqa: E402
from r39_rag.knowledge.kv_motion import (  # noqa: E402
    assert_build_only,
    default_kv_config,
    iter_windows,
    split_segments,
)
from r39_rag.knowledge.online import LaneIndex  # noqa: E402
from r39_rag.projection.evaluate import iter_observations  # noqa: E402
from r39_rag.projection.homography import load_projectors  # noqa: E402
from r39_rag.projection.quality import project_observation  # noqa: E402


def _projection_reject_reason(projected: dict, kv: dict, lane_cloud: np.ndarray | None) -> str | None:
    """返回拒绝原因 返回空表示这一点可以进入轨迹"""
    if projected.get("reassigned"):
        return "reassigned_id"
    status = projected["status"]
    x_value, y_value = projected.get("x"), projected.get("y")
    if x_value is None or y_value is None:
        return "missing_xy"
    if status == "ok":
        return None
    if status != "out_of_bounds":
        return f"projection_{status}"
    if max(abs(float(x_value)), abs(float(y_value))) > float(kv["max_abs_m"]):
        return "beyond_max_abs"
    if lane_cloud is None or len(lane_cloud) == 0:
        return "no_lane_cloud"
    distance = float(np.linalg.norm(lane_cloud - np.array([x_value, y_value]), axis=1).min())
    if distance > float(kv["lane_distance_m"]):
        return "far_from_lane"
    return None


def _accept_point(projected: dict, kv: dict, lane_cloud: np.ndarray | None) -> dict | None:
    if _projection_reject_reason(projected, kv, lane_cloud) is not None:
        return None
    if projected["status"] == "ok":
        return projected
    accepted = dict(projected)
    accepted["status"] = "ok"
    return accepted


def _lane_id(lane_cloud: np.ndarray, owners: np.ndarray, xy: np.ndarray, limit: float) -> str:
    if len(lane_cloud) == 0:
        return ""
    index = int(np.argmin(np.linalg.norm(lane_cloud - xy, axis=1)))
    if float(np.linalg.norm(lane_cloud[index] - xy)) > limit:
        return ""
    return str(owners[index])


def build(config: dict, sequence_ids: list[str] | None = None, allow_holdout: bool = False) -> dict:
    kv = default_kv_config(config)
    sequence_ids = list(config["build_sequences"] if sequence_ids is None else sequence_ids)
    if not allow_holdout:
        assert_build_only(config, sequence_ids)
    sources = load_sources()
    projectors = load_projectors(config["calibration_dir"], sources["cameras"], float(config["min_denominator"]))
    graph = LaneIndex.load()
    lane_cloud = graph._cloud
    owners = graph._owner
    grouped: dict[tuple, list] = defaultdict(list)
    filter_counts: dict[str, int] = defaultdict(int)
    for row in iter_observations(sequence_ids):
        filter_counts["observations"] += 1
        local_id = row.get("local_track_id")
        projector = projectors.get(row["camera_id"])
        if local_id is None or projector is None or row.get("timestamp_ms") is None:
            filter_counts["no_local_or_time"] += 1
            continue
        projected_raw = project_observation(row, projector, config)
        reason = _projection_reject_reason(projected_raw, kv, lane_cloud)
        if reason is not None:
            filter_counts[reason] += 1
            continue
        projected = _accept_point(projected_raw, kv, lane_cloud)
        if projected is None:
            filter_counts["unknown_reject"] += 1
            continue
        grouped[(row["sequence_id"], row["camera_id"], int(local_id))].append(
            {
                "time": float(row["timestamp_ms"]),
                "frame": int(row["frame_index"]),
                "x": float(projected["x"]),
                "y": float(projected["y"]),
                "class_name": str(row.get("class_name") or ""),
            }
        )

    rows = []
    track_of: dict[tuple, int] = {}
    for key, points in grouped.items():
        ordered = sorted(points, key=lambda item: (item["time"], item["frame"]))
        for segment in split_segments(ordered, kv):
            filter_counts["segments"] += 1
            times = np.asarray([item["time"] for item in segment], dtype=np.float64)
            xy = np.asarray([[item["x"], item["y"]] for item in segment], dtype=np.float64)
            names = [str(item.get("class_name") or "") for item in segment]
            counted = {}
            for name in names:
                if name:
                    counted[name] = counted.get(name, 0) + 1
            class_name = ""
            if counted:
                class_name = max(counted, key=counted.get)
                if counted[class_name] < 0.8 * len(names):
                    class_name = ""
            windows = iter_windows(times, xy, kv, config)
            if not windows:
                filter_counts["segments_without_full_window"] += 1
            for window in windows:
                if key not in track_of:
                    track_of[key] = len(track_of)
                lane = _lane_id(lane_cloud, owners, window["key_xy"][-1], float(kv["lane_distance_m"]))
                rows.append(
                    {
                        **window,
                        "sequence_id": key[0],
                        "camera_id": key[1],
                        "local_track_id": key[2],
                        "track_index": track_of[key],
                        "lane_id": lane,
                        "class_name": class_name,
                    }
                )

    kind_count: dict[str, int] = defaultdict(int)
    lengths = []
    for row in rows:
        kind_count[row["kind"]] += 1
        lengths.append(row["t_end_s"] - row["t_start_s"])
    meta = {
        "build_sequences": sequence_ids if not allow_holdout else list(config["build_sequences"]),
        "query_sequences": sequence_ids,
        "holdout_sequences": list(config["holdout_sequences"]),
        "history_seconds": kv["history_seconds"],
        "future_seconds": kv["future_seconds"],
        "stride_seconds": kv["stride_seconds"],
        "min_valid_ratio": kv["min_valid_ratio"],
        "top_k": kv["top_k"],
        "min_score": kv["min_score"],
        "spatial_radius_m": kv["spatial_radius_m"],
        "uses_topology_filter": False,
        "lane_id_role": "场景描述 检索不做道路硬筛选",
        "window_count": len(rows),
        "class_count": {
            name: sum(1 for row in rows if row.get("class_name") == name)
            for name in sorted({row.get("class_name") or "" for row in rows})
        },
        "independent_tracks": len(track_of),
        "kind_count": dict(kind_count),
        "source_tracks_seen": len(grouped),
    }
    filter_counts["windows_kept"] = len(rows)
    filter_counts["tracks_grouped"] = len(grouped)
    return {"rows": rows, "meta": meta, "lengths": lengths, "filter_counts": dict(filter_counts)}


def save(payload: dict, npz_path: Path, sqlite_path: Path) -> None:
    rows = payload["rows"]
    meta = payload["meta"]
    count = len(rows)
    feature = np.zeros((count, 9), dtype=np.float64)
    end_xy = np.zeros((count, 2), dtype=np.float64)
    heading = np.zeros(count, dtype=np.float64)
    speed = np.zeros(count, dtype=np.float64)
    value_rel = np.zeros((count, 4, 2), dtype=np.float64)
    key_xy = np.zeros((count, 5, 2), dtype=np.float64)
    value_xy = np.zeros((count, 4, 2), dtype=np.float64)
    track_index = np.zeros(count, dtype=np.int32)
    local_track_id = np.zeros(count, dtype=np.int32)
    sequence_id = np.empty(count, dtype=object)
    camera_id = np.empty(count, dtype=object)
    kind = np.empty(count, dtype=object)
    lane_id = np.empty(count, dtype=object)
    class_name = np.empty(count, dtype=object)
    for index, row in enumerate(rows):
        feature[index] = row["feature"]
        end_xy[index] = row["key_xy"][-1]
        heading[index] = row["heading"]
        speed[index] = row["speed"]
        value_rel[index] = row["value_rel"]
        key_xy[index] = row["key_xy"]
        value_xy[index] = row["value_xy"]
        track_index[index] = row["track_index"]
        local_track_id[index] = row["local_track_id"]
        sequence_id[index] = row["sequence_id"]
        camera_id[index] = row["camera_id"]
        kind[index] = row["kind"]
        lane_id[index] = row["lane_id"]
        class_name[index] = row.get("class_name") or ""
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        feature=feature,
        end_xy=end_xy,
        heading=heading,
        speed=speed,
        value_rel=value_rel,
        key_xy=key_xy,
        value_xy=value_xy,
        track_index=track_index,
        local_track_id=local_track_id,
        sequence_id=sequence_id,
        camera_id=camera_id,
        kind=kind,
        lane_id=lane_id,
        class_name=class_name,
        meta_json=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    if sqlite_path.exists():
        sqlite_path.unlink()
    connection = sqlite3.connect(sqlite_path)
    try:
        connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute(
            """
            CREATE TABLE windows (
                id INTEGER PRIMARY KEY,
                sequence_id TEXT,
                camera_id TEXT,
                local_track_id INTEGER,
                track_index INTEGER,
                t_start_ms REAL,
                key_end_ms REAL,
                t_end_ms REAL,
                kind TEXT,
                speed REAL,
                heading REAL,
                lane_id TEXT,
                end_x REAL,
                end_y REAL
            )
            """
        )
        connection.execute("INSERT INTO meta (key, value) VALUES ('build', ?)", (json.dumps(meta, ensure_ascii=False),))
        connection.executemany(
            """
            INSERT INTO windows (
                id, sequence_id, camera_id, local_track_id, track_index,
                t_start_ms, key_end_ms, t_end_ms, kind, speed, heading, lane_id, end_x, end_y
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    index,
                    row["sequence_id"],
                    row["camera_id"],
                    row["local_track_id"],
                    row["track_index"],
                    row["t_start_s"] * 1000.0,
                    row["key_end_s"] * 1000.0,
                    row["t_end_s"] * 1000.0,
                    row["kind"],
                    row["speed"],
                    row["heading"],
                    row["lane_id"],
                    float(row["key_xy"][-1, 0]),
                    float(row["key_xy"][-1, 1]),
                )
                for index, row in enumerate(rows)
            ],
        )
        connection.commit()
    finally:
        connection.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="kv_motion_class", help="以免覆盖上一轮知识库")
    args = parser.parse_args()
    if args.name == "kv_motion":
        raise RuntimeError("原始kv_motion不能覆盖")
    config = load_knowledge()
    payload = build(config)
    base = ROOT / "knowledge_base"
    npz_path = base / f"{args.name}.npz"
    sqlite_path = base / f"{args.name}.sqlite"
    save(payload, npz_path, sqlite_path)
    report = {
        "meta": payload["meta"],
        "length_mean_s": None if not payload["lengths"] else float(np.mean(payload["lengths"])),
        "npz": str(npz_path),
        "sqlite": str(sqlite_path),
    }
    report_path = ROOT / "reports" / "kv_motion_quality.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["meta"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
