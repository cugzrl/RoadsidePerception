"""历史运动片段到真实后续行为的 Key-Value 知识

Key 是已经发生的运动 Value 是同一辆车之后真实走过的位置
不把不同转向平均成一条路径 也不用留出序列建库
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from r39_rag.knowledge.trajectory import path_kind, turn_degrees


KEY_SAMPLES = 5
VALUE_OFFSETS = np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
FEATURE_DIM = 9


def default_kv_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = dict(config.get("kv_knowledge") or {})
    raw.setdefault("history_seconds", 1.0)
    raw.setdefault("future_seconds", 2.0)
    raw.setdefault("stride_seconds", 0.5)
    raw.setdefault("min_valid_ratio", 0.9)
    raw.setdefault("top_k", 5)
    raw.setdefault("min_score", 0.30)
    raw.setdefault("spatial_radius_m", 25.0)
    raw.setdefault("max_abs_m", 110.0)
    raw.setdefault("lane_distance_m", 8.0)
    raw.setdefault("max_speed_mps", 22.0)
    raw.setdefault("max_gap_seconds", 0.4)
    raw.setdefault("nominal_fps", 20.0)
    raw["uses_topology_filter"] = False
    return raw


def assert_build_only(config: dict[str, Any], sequence_ids: list[str]) -> None:
    holdout = set(config.get("holdout_sequences") or [])
    leaked = sorted(set(sequence_ids) & holdout)
    if leaked:
        raise RuntimeError(f"留出序列不能进入 Key-Value 知识库 {leaked}")


def _wrap_angle(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2 * np.pi) - np.pi


def to_heading_frame(delta: np.ndarray, heading: float) -> np.ndarray:
    cosine, sine = np.cos(-heading), np.sin(-heading)
    x_value = delta[:, 0]
    y_value = delta[:, 1]
    return np.stack([cosine * x_value - sine * y_value, sine * x_value + cosine * y_value], axis=1)


def from_heading_frame(delta: np.ndarray, heading: float) -> np.ndarray:
    cosine, sine = np.cos(heading), np.sin(heading)
    x_value = delta[:, 0]
    y_value = delta[:, 1]
    return np.stack([cosine * x_value - sine * y_value, sine * x_value + cosine * y_value], axis=1)


def interp_bracketed(times: np.ndarray, xy: np.ndarray, query: np.ndarray) -> np.ndarray | None:
    """只在真实观测之间插值 查询落到区间外就放弃这条窗口"""
    if len(times) < 2:
        return None
    if float(query[0]) < float(times[0]) - 1e-6 or float(query[-1]) > float(times[-1]) + 1e-6:
        return None
    order = np.argsort(times)
    ordered_t = times[order]
    ordered_xy = xy[order]
    unique_t, unique_index = np.unique(ordered_t, return_index=True)
    ordered_xy = ordered_xy[unique_index]
    if len(unique_t) < 2:
        return None
    sampled = np.stack(
        [
            np.interp(query, unique_t, ordered_xy[:, 0]),
            np.interp(query, unique_t, ordered_xy[:, 1]),
        ],
        axis=1,
    )
    return sampled


def split_segments(points: list[dict[str, Any]], kv: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """时间断开或速度不合理时切开 后面的窗口不能跨过这条缝"""
    if not points:
        return []
    max_gap = float(kv["max_gap_seconds"])
    max_speed = float(kv["max_speed_mps"])
    segments: list[list[dict[str, Any]]] = []
    current = [points[0]]
    for item in points[1:]:
        dt = (float(item["time"]) - float(current[-1]["time"])) / 1000.0
        dist = float(np.hypot(item["x"] - current[-1]["x"], item["y"] - current[-1]["y"]))
        broken = dt <= 0 or dt > max_gap or dist > max(6.0, max_speed * max(dt, 1e-3))
        if broken:
            if len(current) >= 2:
                segments.append(current)
            current = [item]
        else:
            current.append(item)
    if len(current) >= 2:
        segments.append(current)
    return segments


def _heading_and_speed(key_xy: np.ndarray, history_s: float) -> tuple[float, float]:
    step = key_xy[-1] - key_xy[max(0, len(key_xy) - 3)]
    if float(np.linalg.norm(step)) < 0.4:
        step = key_xy[-1] - key_xy[0]
    heading = float(np.arctan2(step[1], step[0])) if float(np.linalg.norm(step)) >= 0.2 else 0.0
    length = float(np.linalg.norm(np.diff(key_xy, axis=0), axis=1).sum())
    speed = length / max(history_s, 1e-3)
    return heading, speed


def encode_window(key_xy: np.ndarray, value_xy: np.ndarray, history_s: float, kind_config: dict[str, Any]) -> dict[str, Any]:
    heading, speed = _heading_and_speed(key_xy, history_s)
    relative = to_heading_frame(key_xy[:-1] - key_xy[-1], heading)
    value_rel = to_heading_frame(value_xy - key_xy[-1], heading)
    feature = np.concatenate([relative.reshape(-1), [speed / 5.0]]).astype(np.float64)
    kind = path_kind(np.vstack([key_xy[-1], value_xy]), kind_config)
    return {
        "heading": heading,
        "speed": speed,
        "feature": feature,
        "value_rel": value_rel.astype(np.float64),
        "kind": kind,
        "turn_deg": turn_degrees(np.vstack([key_xy[-1], value_xy])),
    }


def iter_windows(
    times_ms: np.ndarray,
    xy: np.ndarray,
    kv: dict[str, Any],
    kind_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """按时间戳滑动 点数不够完整后续窗口时不补未来位置"""
    history = float(kv["history_seconds"])
    future = float(kv["future_seconds"])
    stride = float(kv["stride_seconds"])
    ratio = float(kv["min_valid_ratio"])
    times = times_ms.astype(np.float64) / 1000.0
    if len(times) < 4 or times[-1] - times[0] < history + future:
        return []
    dt = np.diff(times)
    dt = dt[dt > 1e-6]
    if len(dt) == 0:
        return []
    fps = 1.0 / float(np.median(dt))
    key_offsets = np.linspace(-history, 0.0, KEY_SAMPLES)
    value_offsets = VALUE_OFFSETS[VALUE_OFFSETS <= future + 1e-6]
    if len(value_offsets) < 4 or abs(float(value_offsets[-1]) - future) > 1e-6:
        return []
    found = []
    cursor = float(times[0]) + history
    last = float(times[-1])
    while cursor + future <= last + 1e-6:
        key_xy = interp_bracketed(times, xy, cursor + key_offsets)
        value_xy = interp_bracketed(times, xy, cursor + value_offsets)
        n_key = int(np.sum((times >= cursor - history - 1e-6) & (times <= cursor + 1e-6)))
        n_value = int(np.sum((times > cursor + 1e-6) & (times <= cursor + future + 1e-6)))
        expect_key = ratio * history * fps
        expect_value = ratio * future * fps
        if key_xy is not None and value_xy is not None and n_key >= expect_key and n_value >= expect_value:
            encoded = encode_window(key_xy, value_xy, history, kind_config)
            encoded.update(
                {
                    "key_xy": key_xy,
                    "value_xy": value_xy,
                    "key_end_s": cursor,
                    "t_start_s": cursor - history,
                    "t_end_s": cursor + future,
                }
            )
            found.append(encoded)
        cursor += stride
    return found


def map_futures(end_xy: np.ndarray, heading: float, value_rel: np.ndarray) -> np.ndarray:
    """把历史后续位移转到当前 Key 末端的位置和朝向"""
    return end_xy.reshape(1, 2) + from_heading_frame(np.asarray(value_rel, dtype=np.float64), heading)


class KVIndex:
    """内存最近邻 条目在几千到几万时直接批量算距离"""

    def __init__(self, arrays: dict[str, Any], meta: dict[str, Any]):
        self.meta = meta
        self.feature = np.asarray(arrays["feature"], dtype=np.float64)
        self.end_xy = np.asarray(arrays["end_xy"], dtype=np.float64)
        self.heading = np.asarray(arrays["heading"], dtype=np.float64)
        self.value_rel = np.asarray(arrays["value_rel"], dtype=np.float64)
        self.speed = np.asarray(arrays["speed"], dtype=np.float64)
        self.track_index = np.asarray(arrays["track_index"], dtype=np.int32)
        self.sequence_id = [str(item) for item in arrays["sequence_id"].tolist()]
        self.camera_id = [str(item) for item in arrays["camera_id"].tolist()]
        self.local_track_id = np.asarray(arrays["local_track_id"], dtype=np.int32)
        self.kind = [str(item) for item in arrays["kind"].tolist()]
        self.lane_id = [str(item) for item in arrays["lane_id"].tolist()]
        self.class_name = None
        if "class_name" in arrays:
            self.class_name = [str(item) for item in arrays["class_name"].tolist()]
        self.uses_topology_filter = bool(meta.get("uses_topology_filter", False))

    @classmethod
    def load(cls, npz_path: Path, sqlite_path: Path | None = None) -> "KVIndex":
        loaded = np.load(npz_path, allow_pickle=True)
        arrays = {key: loaded[key] for key in loaded.files}
        meta = {}
        if sqlite_path is not None and sqlite_path.is_file():
            connection = sqlite3.connect(sqlite_path)
            try:
                row = connection.execute("SELECT value FROM meta WHERE key = 'build'").fetchone()
            finally:
                connection.close()
            if row is not None:
                meta = json.loads(row[0])
        if not meta and "meta_json" in arrays:
            meta = json.loads(str(arrays["meta_json"]))
        return cls(arrays, meta)

    def query(
        self,
        key_xy: np.ndarray,
        top_k: int | None = None,
        min_score: float | None = None,
        class_name: str | None = None,
        diverse: bool = False,
    ) -> dict[str, Any]:
        key_xy = np.asarray(key_xy, dtype=np.float64)
        if key_xy.shape != (KEY_SAMPLES, 2) or len(self.feature) == 0:
            return {"reliable": False, "reason": "查询片段不完整", "matches": []}
        history = float(self.meta.get("history_seconds", 1.0))
        kind_config = {"straight_turn_deg": 25, "turn_deg": 50}
        encoded = encode_window(key_xy, np.repeat(key_xy[-1:], 4, axis=0), history, kind_config)
        radius = float(self.meta.get("spatial_radius_m", 25.0))
        limit = int(top_k or self.meta.get("top_k", 5))
        score_floor = float(self.meta.get("min_score", 0.30) if min_score is None else min_score)
        offset = self.end_xy - key_xy[-1]
        position = np.linalg.norm(offset, axis=1)
        chosen = np.flatnonzero(position <= radius)
        if class_name is not None:
            if self.class_name is None:
                return {"reliable": False, "reason": "知识库没有类别", "matches": [], "support_tracks": 0}
            chosen = np.asarray([row for row in chosen if self.class_name[int(row)] == class_name], dtype=np.int64)
            if len(chosen) < 3:
                return {
                    "reliable": False,
                    "reason": "同类历史样本不足",
                    "matches": [],
                    "candidate_count": int(len(chosen)),
                    "support_tracks": int(len({int(self.track_index[int(row)]) for row in chosen})),
                    "class_name": class_name,
                }
        if len(chosen) < 3:
            return {"reliable": False, "reason": "附近历史片段不足", "matches": [], "candidate_count": int(len(chosen))}
        shape = np.linalg.norm(self.feature[chosen] - encoded["feature"], axis=1)
        heading_delta = _wrap_angle(self.heading[chosen] - encoded["heading"])
        direction = 0.5 + 0.5 * np.cos(heading_delta)
        score = np.exp(-position[chosen] / 12.0) * np.exp(-shape / 3.0) * direction
        ranked = np.argsort(-score)
        if diverse:
            picked = []
            seen = set()
            for index in ranked:
                row = int(chosen[int(index)])
                track = int(self.track_index[row])
                if track in seen:
                    continue
                seen.add(track)
                picked.append(int(index))
                if len(picked) >= limit:
                    break
            order = np.asarray(picked, dtype=np.int64)
            if len(order) == 0:
                return {"reliable": False, "reason": "没有独立历史轨迹", "matches": [], "support_tracks": 0}
        else:
            order = ranked[:limit]
        best = float(score[order[0]])
        if best < score_floor:
            return {
                "reliable": False,
                "reason": "最相似片段仍低于阈值",
                "matches": [],
                "best_score": best,
                "candidate_count": int(len(chosen)),
            }
        matches = []
        for index in order:
            row = int(chosen[index])
            future_xy = map_futures(key_xy[-1], encoded["heading"], self.value_rel[row])
            matches.append(
                {
                    "index": row,
                    "score": float(score[index]),
                    "sequence_id": self.sequence_id[row],
                    "camera_id": self.camera_id[row],
                    "local_track_id": int(self.local_track_id[row]),
                    "track_index": int(self.track_index[row]),
                    "kind": self.kind[row],
                    "lane_id": self.lane_id[row],
                    "speed": float(self.speed[row]),
                    "future_xy": future_xy,
                    "uses_topology_filter": False,
                }
            )
        support_tracks = len({item["track_index"] for item in matches})
        return {
            "reliable": True,
            "reason": "命中历史运动片段",
            "matches": matches,
            "best_score": best,
            "candidate_count": int(len(chosen)),
            "support_tracks": support_tracks,
            "class_name": class_name,
            "diverse": bool(diverse),
            "uses_topology_filter": False,
            "query_heading": encoded["heading"],
            "query_speed": encoded["speed"],
        }


def future_at_time(future_xy: np.ndarray, seconds: float) -> np.ndarray:
    """Value 存在 0.5 1.0 1.5 2.0 秒 中间时刻在相邻真实点之间插值"""
    grid = np.concatenate([[0.0], VALUE_OFFSETS])
    origin = np.zeros((1, 2), dtype=np.float64)
    points = np.vstack([origin, np.asarray(future_xy, dtype=np.float64)])
    if seconds <= 0:
        return points[0]
    if seconds >= float(grid[-1]):
        return points[-1]
    return np.array(
        [np.interp(seconds, grid, points[:, 0]), np.interp(seconds, grid, points[:, 1])],
        dtype=np.float64,
    )


def constant_velocity(key_xy: np.ndarray, seconds: np.ndarray, history_s: float = 1.0) -> np.ndarray:
    heading, speed = _heading_and_speed(key_xy, history_s)
    direction = np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
    return key_xy[-1] + seconds.reshape(-1, 1) * speed * direction
