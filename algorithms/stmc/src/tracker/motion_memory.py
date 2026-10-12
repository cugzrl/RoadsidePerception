"""在线真实观测记忆 以及可选的道路筛选和 Key-Value 重关联

默认关闭 只在目标已经中断之后给唯一候选加边
预测位置不会写回真实轨迹
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _cfg(cfg: Any, key: str, default: Any) -> Any:
    if hasattr(cfg, "get"):
        value = cfg.get(key, default)
    else:
        value = getattr(cfg, key, default)
    return default if value is None else value


def _load_kv(path: Path):
    root = Path(__file__).resolve().parents[4]
    rag_src = root / "R39_RAG" / "src"
    if str(rag_src) not in sys.path:
        sys.path.insert(0, str(rag_src))
    from r39_rag.knowledge.kv_motion import KVIndex, future_at_time

    sqlite_path = path.with_suffix(".sqlite")
    return KVIndex.load(path, sqlite_path if sqlite_path.is_file() else None), future_at_time


def _load_lanes(path: Path):
    root = Path(__file__).resolve().parents[4]
    rag_src = root / "R39_RAG" / "src"
    if str(rag_src) not in sys.path:
        sys.path.insert(0, str(rag_src))
    from r39_rag.knowledge.online import LaneIndex

    return LaneIndex.load(path)


class MotionMemory:
    def __init__(self, cfg: Any):
        self.enabled = bool(_cfg(cfg, "enabled", False))
        self.retain_frames = int(_cfg(cfg, "retain_frames", 30))
        self.min_missed = int(_cfg(cfg, "min_missed", 4))
        self.max_missed = int(_cfg(cfg, "max_missed", 30))
        self.fps = float(_cfg(cfg, "fps", 20.0))
        self.reid_min = float(_cfg(cfg, "reid_min", 0.78))
        self.static_reid_min = float(_cfg(cfg, "static_reid_min", 0.88))
        self.ambiguity_margin = float(_cfg(cfg, "ambiguity_margin", 0.08))
        self.max_speed = float(_cfg(cfg, "max_speed_mps", 22.0))
        self.cv_base = float(_cfg(cfg, "cv_radius_base", 3.0))
        self.cv_gain = float(_cfg(cfg, "cv_radius_gain", 4.0))
        self.cv_cap = float(_cfg(cfg, "cv_radius_cap", 8.0))
        self.kv_base = float(_cfg(cfg, "kv_radius_base", 3.0))
        self.kv_gain = float(_cfg(cfg, "kv_radius_gain", 3.0))
        self.kv_cap = float(_cfg(cfg, "kv_radius_cap", 8.0))
        self.protect_edge = float(_cfg(cfg, "protect_edge", 0.40))
        self.weight = float(_cfg(cfg, "weight", 0.85))
        self.use_topology = bool(_cfg(cfg, "use_topology", False))
        self.use_kv = bool(_cfg(cfg, "use_kv", False))
        self.log_path = _cfg(cfg, "log_path", None)
        self.real_history: dict[int, list[tuple[int, np.ndarray]]] = {}
        self.retrieve_s = 0.0
        self.apply_s = 0.0
        self.accepts = 0
        self.kv_hits = 0
        self.fallbacks = 0
        self.ambiguous = 0
        self._log_lines: list[str] = []
        self.index = None
        self.future_at_time = None
        self.lanes = None
        if self.use_kv:
            kv_path = _cfg(cfg, "kv_path", None)
            if kv_path:
                loaded, future_at_time = _load_kv(Path(str(kv_path)))
                self.index = loaded
                self.future_at_time = future_at_time
        if self.use_topology:
            graph = _cfg(cfg, "lane_graph", None)
            if graph is None:
                graph = Path(__file__).resolve().parents[4] / "configs" / "maps" / "r39_lane_graph.json"
            self.lanes = _load_lanes(Path(str(graph)))

    def remember(self, label: int, frame: int, xy: np.ndarray) -> None:
        """只记录刚刚关联上的真实位置"""
        history = self.real_history.setdefault(int(label), [])
        point = np.asarray(xy, dtype=np.float64).reshape(2).copy()
        if history and history[-1][0] == int(frame):
            history[-1] = (int(frame), point)
        else:
            history.append((int(frame), point))
        if len(history) > 40:
            del history[:-40]

    def flush(self, frame: int) -> None:
        if not self.log_path:
            return
        path = Path(str(self.log_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._log_lines:
            with path.open("a", encoding="utf-8") as file:
                file.write("".join(self._log_lines))
            self._log_lines.clear()
        if frame % 100 == 0:
            summary = path.with_name(path.stem + "_summary.json")
            summary.write_text(
                json.dumps(
                    {
                        "frame": int(frame),
                        "accepts": self.accepts,
                        "kv_hits": self.kv_hits,
                        "fallbacks": self.fallbacks,
                        "ambiguous": self.ambiguous,
                        "retrieve_s": self.retrieve_s,
                        "apply_s": self.apply_s,
                        "use_topology": self.use_topology,
                        "use_kv": self.use_kv,
                        "uses_topology_filter": self.use_topology,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def _feature(self, track) -> np.ndarray | None:
        features = track.features
        finite = ~torch.isnan(features).any(dim=-1)
        active = finite & track.track_where
        mask = active if bool(active.any()) else finite
        if not bool(mask.any()):
            return None
        vector = features[mask].mean(dim=0)
        norm = torch.linalg.vector_norm(vector).clamp(min=1e-6)
        return (vector / norm).detach().cpu().numpy()

    def _position(self, track) -> np.ndarray | None:
        positions = track.positions_3d
        finite = ~torch.isnan(positions).any(dim=-1)
        if positions.shape[-1] > 2:
            positions = positions[..., :2]
            finite = ~torch.isnan(positions).any(dim=-1)
        if not bool(finite.any()):
            return None
        return positions[finite].mean(dim=0).detach().cpu().numpy()

    def _key_xy(self, history: list[tuple[int, np.ndarray]]) -> np.ndarray | None:
        frames = np.asarray([item[0] for item in history], dtype=np.float64)
        xy = np.stack([item[1] for item in history], axis=0)
        last = frames[-1]
        times = (frames - last) / self.fps
        if float(times[0]) > -0.95:
            return None
        offsets = np.linspace(-1.0, 0.0, 5)
        if float(offsets[0]) < float(times[0]) - 1e-6:
            return None
        return np.stack(
            [np.interp(offsets, times, xy[:, 0]), np.interp(offsets, times, xy[:, 1])],
            axis=1,
        )

    def _topology_ok(self, origin: np.ndarray, target: np.ndarray) -> bool:
        if self.lanes is None:
            return False
        left = self.lanes.locate(origin, 8.0)
        right = self.lanes.locate(target, 8.0)
        if left is None or right is None:
            return False
        if not self.lanes.reachable(left["lane_id"], right["lane_id"], hops=6):
            if not self.lanes.reachable(right["lane_id"], left["lane_id"], hops=6):
                return False
        step = target - origin
        if float(np.linalg.norm(step)) > 1.0:
            heading = left["heading"]
            cosine = float(np.dot(step, heading) / (np.linalg.norm(step) * max(np.linalg.norm(heading), 1e-6)))
            if cosine < -0.2:
                return False
        return True

    def apply(self, adj: torch.Tensor, tracks: list, frame: int, labels_alive: set[int] | None = None) -> torch.Tensor:
        if not self.enabled:
            return adj
        started = time.perf_counter()
        current = [index for index, track in enumerate(tracks) if track.frame == frame and track.label is None]
        older = []
        for index, track in enumerate(tracks):
            if track.label is None:
                continue
            missed = int(frame - track.frame) - 1
            if self.min_missed <= missed <= self.max_missed:
                older.append(index)
        if not current or not older:
            self.apply_s += time.perf_counter() - started
            return adj
        weight_cpu = adj.detach().cpu()
        protected = []
        for left in current:
            best = 0.0
            for right, track in enumerate(tracks):
                if track.label is None:
                    continue
                missed = int(frame - track.frame) - 1
                if missed < self.min_missed:
                    best = max(best, float(weight_cpu[left, right]))
            protected.append(best >= self.protect_edge)
        det_feat = [self._feature(tracks[index]) for index in current]
        det_xy = [self._position(tracks[index]) for index in current]
        old_feat = [self._feature(tracks[index]) for index in older]
        kv_cache: dict[tuple[int, int], dict | None] = {}
        pairs = []
        for col, left in enumerate(current):
            if protected[col] or det_feat[col] is None or det_xy[col] is None:
                continue
            if float(np.max(np.abs(det_xy[col]))) > 140.0:
                continue
            for row, right in enumerate(older):
                track = tracks[right]
                if old_feat[row] is None or int(track.class_id) != int(tracks[left].class_id):
                    continue
                history = self.real_history.get(int(track.label), [])
                if len(history) < 2:
                    continue
                origin = history[-1][1]
                gap_frames = frame - history[-1][0]
                if gap_frames <= 0 or gap_frames > self.retain_frames:
                    continue
                dt = gap_frames / self.fps
                step = det_xy[col] - origin
                step_m = float(np.linalg.norm(step))
                if not np.isfinite(step_m) or float(np.max(np.abs(origin))) > 140.0:
                    continue
                speed = step_m / dt
                if speed > self.max_speed:
                    continue
                span_frames = max(history[-1][0] - history[0][0], 1)
                history_speed = float(np.linalg.norm(history[-1][1] - history[0][1])) / (span_frames / self.fps)
                reid_need = self.static_reid_min if history_speed < 1.5 else self.reid_min
                if history_speed < 1.5 and step_m > 4.0:
                    continue
                reid = float(np.dot(old_feat[row], det_feat[col]))
                if reid < reid_need:
                    continue
                mode = "memory"
                dist = None
                radius = min(self.cv_cap, self.cv_base + self.cv_gain * dt)
                kv_score = None
                kv_source = None
                uses_topology = False
                if self.use_kv and self.index is not None:
                    cache_key = (int(track.label), int(history[-1][0]))
                    if cache_key not in kv_cache:
                        query_key = self._key_xy(history)
                        retrieve_started = time.perf_counter()
                        kv_cache[cache_key] = None if query_key is None else self.index.query(query_key)
                        self.retrieve_s += time.perf_counter() - retrieve_started
                        cached = kv_cache[cache_key]
                        if cached is not None and cached.get("reliable"):
                            self.kv_hits += 1
                        elif query_key is not None:
                            self.fallbacks += 1
                    found = kv_cache[cache_key]
                    if found is not None and found.get("reliable"):
                        points = [
                            self.future_at_time(item["future_xy"], dt) for item in found["matches"]
                        ]
                        dist = min(float(np.linalg.norm(det_xy[col] - point)) for point in points)
                        radius = min(self.kv_cap, self.kv_base + self.kv_gain * dt)
                        mode = "kv"
                        kv_score = found["best_score"]
                        kv_source = {
                            "sequence_id": found["matches"][0]["sequence_id"],
                            "camera_id": found["matches"][0]["camera_id"],
                            "local_track_id": found["matches"][0]["local_track_id"],
                            "kind": found["matches"][0]["kind"],
                            "support_tracks": len({item["track_index"] for item in found["matches"]}),
                        }
                    else:
                        mode = "fallback_memory"
                if mode != "kv":
                    frames = [item[0] for item in history]
                    delta_f = frames[-1] - frames[-2]
                    if delta_f <= 0:
                        continue
                    velocity = (history[-1][1] - history[-2][1]) / (delta_f / self.fps)
                    predicted = history[-1][1] + velocity * dt
                    dist = float(np.linalg.norm(det_xy[col] - predicted))
                if history_speed < 1.5:
                    radius = min(radius, 3.0)
                if dist is None or dist > radius:
                    continue
                if self.use_topology:
                    if not self._topology_ok(origin, det_xy[col]):
                        continue
                    uses_topology = True
                score = reid * float(np.exp(-dist / 4.0))
                pairs.append(
                    {
                        "row": row,
                        "col": col,
                        "left": left,
                        "right": right,
                        "score": score,
                        "reid": reid,
                        "dist": dist,
                        "mode": mode,
                        "dt": dt,
                        "label": int(track.label),
                        "kv_score": kv_score,
                        "kv_source": kv_source,
                        "uses_topology": uses_topology,
                        "pred_xy": origin.tolist(),
                    }
                )
        if not pairs:
            self.apply_s += time.perf_counter() - started
            return adj
        n_old = len(older)
        n_new = len(current)
        score_matrix = np.full((n_old, n_new), -1.0, dtype=np.float64)
        payload = {}
        for item in pairs:
            previous = score_matrix[item["row"], item["col"]]
            if item["score"] > previous:
                score_matrix[item["row"], item["col"]] = item["score"]
                payload[(item["row"], item["col"])] = item
        assigned = self._unique(score_matrix)
        for row, col in assigned:
            item = payload[(row, col)]
            base = float(weight_cpu[item["left"], item["right"]])
            if base != base:
                base = 0.0
            updated = max(base, self.weight)
            adj[item["left"], item["right"]] = updated
            adj[item["right"], item["left"]] = updated
            self.accepts += 1
            self._log(frame, item, "accept")
        self.apply_s += time.perf_counter() - started
        return adj

    def _unique(self, score_matrix: np.ndarray) -> list[tuple[int, int]]:
        valid = score_matrix >= 0
        blocked_row = np.zeros(score_matrix.shape[0], dtype=bool)
        blocked_col = np.zeros(score_matrix.shape[1], dtype=bool)
        for col in range(score_matrix.shape[1]):
            values = score_matrix[valid[:, col], col] if valid[:, col].any() else np.array([])
            if len(values) >= 2 and float(np.sort(values)[-1] - np.sort(values)[-2]) < self.ambiguity_margin:
                blocked_col[col] = True
                self.ambiguous += 1
        for row in range(score_matrix.shape[0]):
            values = score_matrix[row, valid[row]] if valid[row].any() else np.array([])
            if len(values) >= 2 and float(np.sort(values)[-1] - np.sort(values)[-2]) < self.ambiguity_margin:
                blocked_row[row] = True
                self.ambiguous += 1
        cost = np.where(valid, 1.0 - score_matrix, 1e3)
        rows, cols = linear_sum_assignment(cost)
        chosen = []
        for row, col in zip(rows.tolist(), cols.tolist()):
            if not valid[row, col] or blocked_row[row] or blocked_col[col]:
                continue
            chosen.append((row, col))
        return chosen

    def _log(self, frame: int, item: dict, action: str) -> None:
        if not self.log_path:
            return
        record = {
            "frame": int(frame),
            "action": action,
            "label": item["label"],
            "mode": item["mode"],
            "reid": round(float(item["reid"]), 4),
            "dist_m": round(float(item["dist"]), 3),
            "score": round(float(item["score"]), 4),
            "dt_s": round(float(item["dt"]), 3),
            "kv_score": None if item["kv_score"] is None else round(float(item["kv_score"]), 4),
            "kv_source": item["kv_source"],
            "uses_topology_filter": bool(item["uses_topology"]),
            "pred_xy": item["pred_xy"],
        }
        self._log_lines.append(json.dumps(record, ensure_ascii=False) + "\n")
