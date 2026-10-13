"""在线轨迹前缀、道路连通和可信度查询

只使用调用方已经观测到的点 不读取未来帧
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from r39_rag.knowledge.trajectory import resample_path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_GRAPH = ROOT / "configs" / "maps" / "r39_lane_graph.json"
DEFAULT_TRUST = Path(__file__).resolve().parents[3] / "knowledge_base" / "tracking_trust_build.json"


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm < 0.5 or right_norm < 0.5:
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


def match_prefix(
    points: np.ndarray,
    prototypes: np.ndarray,
    sigma_m: float = 8.0,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """把已观测折线对齐到典型路径的某一段 不要求和整条路径的起终点重合"""
    array = np.asarray(points, dtype=np.float64)
    stored = np.ascontiguousarray(prototypes, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 2 or stored.size == 0:
        return []
    window = int(np.clip(len(array), 4, min(8, stored.shape[1])))
    if stored.shape[1] < window:
        return []
    query = resample_path(array, window)
    query_step = query[-1] - query[0]
    # (路径数, 起点数, 窗口, 2) 一次算完全部对齐 数值与逐段循环一致
    windows = np.moveaxis(np.lib.stride_tricks.sliding_window_view(stored, window, axis=1), -1, -2)
    distance = np.linalg.norm(windows - query, axis=-1).mean(axis=-1)
    proto_step = windows[:, :, -1, :] - windows[:, :, 0, :]
    query_norm = float(np.linalg.norm(query_step))
    proto_norm = np.linalg.norm(proto_step, axis=-1)
    valid = (query_norm >= 0.5) & (proto_norm >= 0.5)
    cosine = np.zeros(distance.shape, dtype=np.float64)
    np.divide(
        np.sum(proto_step * query_step, axis=-1),
        proto_norm * max(query_norm, 1e-9),
        out=cosine,
        where=valid,
    )
    direction_term = np.where(valid, np.maximum(cosine, 0.0), 0.5)
    score = np.exp(-distance / sigma_m) * (0.5 + 0.5 * direction_term)
    best_start = np.argmax(score, axis=1)
    found = []
    n_samples = stored.shape[1]
    for index, start in enumerate(best_start.tolist()):
        end = int(start) + window
        direction = None if not valid[index, start] else float(cosine[index, start])
        found.append(
            {
                "prototype_id": int(index),
                "score": float(score[index, start]),
                "mean_point_distance_m": float(distance[index, start]),
                "direction_cos": direction,
                "progress": float((start + window * 0.5) / n_samples),
                "window_start": int(start),
                "window_end": int(end),
                "continuation": stored[index, end : end + 4].tolist(),
            }
        )
    found.sort(key=lambda item: item["score"], reverse=True)
    return found[:top_k]


class LaneIndex:
    """车道中心线与后继 用于判断两帧位置是否沿道路连通"""

    def __init__(self, payload: dict[str, Any]):
        self.center = np.asarray(payload["center_world"], dtype=np.float64)
        self.nodes = {}
        self.successors: dict[str, list[str]] = defaultdict(list)
        for node in payload["lane_nodes"]:
            if node["type"] not in ("vehicle_lane", "connector", "bike_lane"):
                continue
            centerline = np.asarray(node["centerline"], dtype=np.float64)
            self.nodes[node["lane_id"]] = {
                "centerline": centerline,
                "arm": node.get("arm"),
                "role": node.get("role"),
                "type": node["type"],
            }
        for edge in payload.get("lane_lane_topology", []):
            if edge.get("relation") == "successor":
                self.successors[edge["source"]].append(edge["target"])
        clouds = []
        owners = []
        locals_ = []
        headings = []
        for lane_id, node in self.nodes.items():
            centerline = node["centerline"]
            clouds.append(centerline)
            owners.append(np.full(len(centerline), lane_id, dtype=object))
            locals_.append(np.arange(len(centerline)))
            step = np.diff(centerline, axis=0, append=centerline[-1:])
            if len(centerline) > 1:
                step[-1] = centerline[-1] - centerline[-2]
            else:
                step[-1] = 0.0
            norm = np.linalg.norm(step, axis=1, keepdims=True)
            headings.append(step / np.maximum(norm, 1e-9))
        self._cloud = np.concatenate(clouds, axis=0) if clouds else np.zeros((0, 2))
        self._owner = np.concatenate(owners) if owners else np.array([], dtype=object)
        self._local = np.concatenate(locals_) if locals_ else np.array([], dtype=int)
        self._heading = np.concatenate(headings, axis=0) if headings else np.zeros((0, 2))
        self._locate_cache: dict[tuple, dict | None] = {}
        self._reach_cache: dict[tuple, bool] = {}

    @classmethod
    def load(cls, path: Path | str = DEFAULT_GRAPH) -> "LaneIndex":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def locate(self, xy: np.ndarray, max_distance_m: float = 6.0) -> dict[str, Any] | None:
        point = np.asarray(xy, dtype=np.float64).reshape(2)
        key = (float(point[0]), float(point[1]), float(max_distance_m))
        if key in self._locate_cache:
            return self._locate_cache[key]
        if len(self._cloud) == 0:
            self._locate_cache[key] = None
            return None
        nearest_index = int(np.argmin(np.linalg.norm(self._cloud - point, axis=1)))
        nearest = float(np.linalg.norm(self._cloud[nearest_index] - point))
        if nearest > max_distance_m:
            self._locate_cache[key] = None
            return None
        lane_id = str(self._owner[nearest_index])
        node = self.nodes[lane_id]
        best = {
            "lane_id": lane_id,
            "distance_m": nearest,
            "heading": self._heading[nearest_index],
            "arm": node["arm"],
            "role": node["role"],
            "type": node["type"],
        }
        self._locate_cache[key] = best
        return best

    def locate_typed(self, xy: np.ndarray, max_distance_m: float, allowed_types: set[str]) -> dict[str, Any] | None:
        """只在指定道路类型里找最近点 找不到就当投影不可靠"""
        point = np.asarray(xy, dtype=np.float64).reshape(2)
        if len(self._cloud) == 0:
            return None
        distance = np.linalg.norm(self._cloud - point, axis=1)
        order = np.argsort(distance)
        for index in order:
            nearest = float(distance[index])
            if nearest > max_distance_m:
                return None
            lane_id = str(self._owner[index])
            node = self.nodes[lane_id]
            if node["type"] not in allowed_types:
                continue
            return {
                "lane_id": lane_id,
                "distance_m": nearest,
                "heading": self._heading[index],
                "arm": node["arm"],
                "role": node["role"],
                "type": node["type"],
            }
        return None

    def reachable(self, source: str, target: str, hops: int = 4) -> bool:
        key = (source, target, hops)
        if key in self._reach_cache:
            return self._reach_cache[key]
        answer = self._reachable(source, target, hops)
        self._reach_cache[key] = answer
        return answer

    def _reachable(self, source: str, target: str, hops: int) -> bool:
        if source == target:
            return True
        frontier = [source]
        seen = {source}
        for _ in range(hops):
            nxt = []
            for lane_id in frontier:
                for child in self.successors.get(lane_id, []):
                    if child == target:
                        return True
                    if child not in seen:
                        seen.add(child)
                        nxt.append(child)
            frontier = nxt
        return False


class TrustGrid:
    """四段建库序列上的跨相机误差格子 样本不足时保持中性"""

    def __init__(self, payload: dict[str, Any]):
        self.center = np.asarray(payload["center"], dtype=np.float64)
        self.cell_m = float(payload["cell_m"])
        self.cells = payload["cells"]
        self.source_sequences = payload.get("source_sequences", [])

    @classmethod
    def load(cls, path: Path | str = DEFAULT_TRUST) -> "TrustGrid | None":
        file_path = Path(path)
        if not file_path.is_file():
            return None
        return cls(json.loads(file_path.read_text(encoding="utf-8")))

    def query(self, xy: np.ndarray) -> dict[str, Any]:
        point = np.asarray(xy, dtype=np.float64).reshape(2)
        offset = np.floor((point - self.center) / self.cell_m).astype(int)
        key = f"{int(offset[0])},{int(offset[1])}"
        row = self.cells.get(key)
        if row is None:
            return {
                "found": False,
                "level": "样本不足",
                "neutral": True,
                "reason": "该位置没有建库序列的跨相机样本",
            }
        level = row["level"]
        neutral = level in ("低", "样本不足", "范围外")
        return {
            "found": True,
            "level": level,
            "neutral": neutral,
            "count": row["count"],
            "median_m": row["median_m"],
            "tail_15": row["tail_15"],
            "source_sequences": self.source_sequences,
        }


def _motion_gate(points: np.ndarray, target: np.ndarray, frame_gap: int) -> dict[str, float | bool | None]:
    """用时间间隔检查位移 静止前缀不能跳到远处 超速点不加分"""
    step = target - points[-1]
    step_m = float(np.linalg.norm(step))
    dt = max(int(frame_gap), 1) / 20.0
    speed = step_m / dt
    span = max(len(points) - 1, 1) / 20.0
    history_speed = float(np.linalg.norm(points[-1] - points[0])) / span
    finite = bool(np.isfinite(target).all() and np.isfinite(points).all())
    exploded = float(np.max(np.abs(np.concatenate([points[-1:], target.reshape(1, 2)])))) > 140.0
    static_jump = history_speed < 1.5 and step_m > 6.0
    too_fast = (not finite) or exploded or speed > 25.0
    return {
        "step_m": step_m,
        "speed": speed,
        "history_speed": history_speed,
        "too_fast": too_fast,
        "static_jump": static_jump,
        "exploded": exploded,
    }


def continuation_distance(candidate: np.ndarray, continuation: list[list[float]]) -> float | None:
    if not continuation:
        return None
    array = np.asarray(continuation, dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.linalg.norm(array - np.asarray(candidate, dtype=np.float64).reshape(1, 2), axis=1).min())


def link_decision(
    history: np.ndarray,
    candidate: np.ndarray,
    prototypes: np.ndarray,
    lanes: LaneIndex | None,
    trust: TrustGrid | None,
    sigma_m: float,
    top_k: int,
    min_score: float,
    continuation_m: float,
    min_points: int,
    frame_gap: int = 1,
    cached_matches: list[dict[str, Any]] | None = None,
    prototype_speeds: np.ndarray | None = None,
    prototype_supports: np.ndarray | None = None,
) -> dict[str, Any]:
    """判断新观测是否沿历史路径和道路继续 证据不足时 reliable 为假"""
    points = np.asarray(history, dtype=np.float64)
    target = np.asarray(candidate, dtype=np.float64).reshape(2)
    result: dict[str, Any] = {
        "reliable": False,
        "action": "none",
        "score": 0.0,
        "matches": [],
        "trust": None,
        "lanes": None,
        "reasons": [],
    }
    if len(points) < min_points:
        result["reasons"].append("轨迹前缀太短")
        return result
    matches = cached_matches if cached_matches is not None else match_prefix(points, prototypes, sigma_m=sigma_m, top_k=top_k)
    result["matches"] = matches
    best = matches[0] if matches else None
    trust_row = trust.query(target) if trust is not None else {"found": False, "level": "样本不足", "neutral": True}
    result["trust"] = trust_row
    source = lanes.locate(points[-1]) if lanes is not None else None
    destination = lanes.locate(target) if lanes is not None else None
    result["lanes"] = {
        "from": None if source is None else source["lane_id"],
        "to": None if destination is None else destination["lane_id"],
        "connected": False,
    }
    connected = False
    if source is not None and destination is not None and lanes is not None:
        connected = lanes.reachable(source["lane_id"], destination["lane_id"])
        result["lanes"]["connected"] = connected
        motion = target - points[-1]
        heading = destination["heading"]
        alignment = _cosine(motion, heading)
        result["lanes"]["heading_cos"] = alignment
    motion = _motion_gate(points, target, frame_gap)
    step_m = float(motion["step_m"])
    recent = points[-1] - points[0]
    recent_cos = _cosine(recent, target - points[-1])
    dt = max(int(frame_gap), 1) / 20.0
    hold_limit = min(12.0, 3.0 + 18.0 * dt)
    near_own_track = step_m <= hold_limit and (recent_cos is None or recent_cos >= -0.2)
    on_lane = lanes is None or (source is not None and destination is not None)
    if motion["too_fast"] or motion["exploded"]:
        result["reasons"].append("位移速度不合理或 BEV 超出道路可用范围")
        return result
    if motion["static_jump"]:
        result["reasons"].append("静止轨迹不能因为距离接近接到运动位置")
        near_own_track = False
    if best is None or best["score"] < min_score:
        result["reasons"].append("没有足够相似的历史路径")
        if connected and step_m <= continuation_m * 2:
            result["reliable"] = True
            result["action"] = "topology"
            result["score"] = 0.4
            result["reasons"].append("两点落在连通车道上")
            return result
        if near_own_track:
            result["reliable"] = True
            result["action"] = "hold"
            result["score"] = 0.5
            result["reasons"].append("新位置仍靠近这条轨迹自己的最近 BEV 点")
        return result
    gap = continuation_distance(target, best["continuation"])
    result["continuation_gap_m"] = gap
    direction = best["direction_cos"]
    support = None
    proto_speed = None
    if best is not None and prototype_supports is not None and 0 <= int(best["prototype_id"]) < len(prototype_supports):
        support = float(prototype_supports[int(best["prototype_id"])])
    if best is not None and prototype_speeds is not None and 0 <= int(best["prototype_id"]) < len(prototype_speeds):
        proto_speed = float(prototype_speeds[int(best["prototype_id"])])
        if not np.isfinite(proto_speed):
            proto_speed = None
    on_path = gap is not None and gap <= continuation_m and (direction is None or direction >= 0.0)
    if on_path and not on_lane:
        result["reasons"].append("路径形状接近 但端点不在车道附近")
        on_path = False
    if on_path and support is not None and support < 3:
        result["reasons"].append("历史路径的独立轨迹支持不足")
        on_path = False
    if on_path and proto_speed is not None and proto_speed > 4.0 and float(motion["history_speed"]) < 1.5:
        result["reasons"].append("静止前缀不能沿快速历史路径加分")
        on_path = False
    contradicts = direction is not None and direction < -0.2 and (gap is None or gap > continuation_m * 2)
    if on_path:
        result["reliable"] = True
        result["action"] = "continuation"
        result["score"] = float(best["score"])
        result["reasons"].append("新位置落在典型路径的后续片段上")
        return result
    if contradicts and not trust_row.get("neutral", True):
        result["reliable"] = True
        result["action"] = "contradict"
        result["score"] = float(best["score"])
        result["reasons"].append("运动方向与高可信历史路径相反")
        return result
    result["reasons"].append("路径匹配存在 但新位置对不上下一段")
    if near_own_track and not contradicts:
        result["reliable"] = True
        result["action"] = "hold"
        result["score"] = 0.5
        result["reasons"].append("新位置仍靠近这条轨迹自己的最近 BEV 点")
    return result
