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
        self.strict_protect = bool(_cfg(cfg, "strict_protect", False))
        self.weight = float(_cfg(cfg, "weight", 0.85))
        self.use_topology = bool(_cfg(cfg, "use_topology", False))
        self.use_kv = bool(_cfg(cfg, "use_kv", False))
        self.kv_class_aware = bool(_cfg(cfg, "kv_class_aware", False))
        self.kv_diverse = bool(_cfg(cfg, "kv_diverse", False))
        self.kv_blend = bool(_cfg(cfg, "kv_blend", False))
        self.class_reid_min = _cfg(cfg, "class_reid_min", None)
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
        active = finite & track.track_where
        mask = active if bool(active.any()) else None
        if mask is None:
            return None
        return positions[mask].mean(dim=0).detach().cpu().numpy()

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

    def _reid_need(self, class_id: int, history_speed: float) -> float:
        if self.class_reid_min is not None:
            values = list(self.class_reid_min)
            if 0 <= int(class_id) < len(values) and values[int(class_id)] is not None:
                return float(values[int(class_id)])
        return self.static_reid_min if history_speed < 1.5 else self.reid_min

    def _topology_decision(self, class_id: int, origin: np.ndarray, target: np.ndarray) -> tuple[bool, str, dict]:
        """行人不套机动车车道 投影不可靠时不拒绝"""
        info = {"class_id": int(class_id), "from": None, "to": None}
        if self.lanes is None:
            return True, "没有道路图", info
        if int(class_id) == 6:
            return True, "行人不用机动车拓扑", info
        allowed = {"bike_lane", "connector"} if int(class_id) in (3, 4, 5) else {"vehicle_lane", "connector"}
        info["allowed"] = sorted(allowed)
        left = self.lanes.locate_typed(origin, 8.0, allowed)
        right = self.lanes.locate_typed(target, 8.0, allowed)
        info["from"] = None if left is None else left["lane_id"]
        info["to"] = None if right is None else right["lane_id"]
        if left is None or right is None:
            return True, "不在该类道路附近 不用拓扑硬拒绝", info
        connected = self.lanes.reachable(left["lane_id"], right["lane_id"], hops=6) or self.lanes.reachable(
            right["lane_id"], left["lane_id"], hops=6
        )
        info["connected"] = connected
        if not connected:
            return False, "道路区域不连通", info
        step = target - origin
        if float(np.linalg.norm(step)) > 1.0:
            heading = left["heading"]
            cosine = float(np.dot(step, heading) / (np.linalg.norm(step) * max(np.linalg.norm(heading), 1e-6)))
            info["heading_cos"] = round(cosine, 3)
            if cosine < -0.2:
                return False, "位移与车道方向相反", info
        return True, "道路连通", info

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
            recent = []
            for right, track in enumerate(tracks):
                if track.label is None:
                    continue
                missed = int(frame - track.frame) - 1
                if missed < self.min_missed:
                    recent.append(float(weight_cpu[left, right]))
            recent.sort(reverse=True)
            best = recent[0] if recent else 0.0
            second = recent[1] if len(recent) > 1 else -1.0
            stable = best >= self.protect_edge and (not self.strict_protect or best - second >= self.ambiguity_margin)
            protected.append(stable)
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
                reid_need = self._reid_need(int(track.class_id), history_speed)
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
                        query_class = None
                        if self.kv_class_aware:
                            names = {
                                0: "Car",
                                1: "Van",
                                2: "Bus",
                                3: "Tricycle",
                                4: "ScooterRider",
                                5: "Cyclist",
                                6: "Pedestrian",
                            }
                            query_class = names.get(int(track.class_id))
                        kv_cache[cache_key] = (
                            None
                            if query_key is None
                            else self.index.query(query_key, class_name=query_class, diverse=self.kv_diverse)
                        )
                        self.retrieve_s += time.perf_counter() - retrieve_started
                        cached = kv_cache[cache_key]
                        if cached is not None and cached.get("reliable"):
                            self.kv_hits += 1
                        elif query_key is not None:
                            self.fallbacks += 1
                    found = kv_cache[cache_key]
                    if found is not None and found.get("reliable"):
                        points = [self.future_at_time(item["future_xy"], dt) for item in found["matches"]]
                        kv_dist = min(float(np.linalg.norm(det_xy[col] - point)) for point in points)
                        spread = 0.0
                        if len(points) >= 2:
                            stacked = np.stack(points, axis=0)
                            spread = float(np.max(np.linalg.norm(stacked[:, None, :] - stacked[None, :, :], axis=-1)))
                        support = len({item["track_index"] for item in found["matches"]})
                        kv_score = found["best_score"]
                        kv_source = {
                            "sequence_id": found["matches"][0]["sequence_id"],
                            "camera_id": found["matches"][0]["camera_id"],
                            "local_track_id": found["matches"][0]["local_track_id"],
                            "kind": found["matches"][0]["kind"],
                            "support_tracks": support,
                            "spread_m": round(spread, 3),
                            "reason": found.get("reason"),
                            "futures": [np.asarray(item["future_xy"]).round(3).tolist() for item in found["matches"][:5]],
                        }
                        if not self.kv_blend:
                            dist = kv_dist
                            radius = min(self.kv_cap, self.kv_base + self.kv_gain * dt)
                            mode = "kv"
                        else:
                            mode = "kv_pending"
                            kv_source["kv_dist"] = round(kv_dist, 3)
                    else:
                        mode = "fallback_memory"
                        if found is not None:
                            kv_source = {"reason": found.get("reason"), "support_tracks": found.get("support_tracks", 0)}
                frames = [item[0] for item in history]
                delta_f = frames[-1] - frames[-2]
                if delta_f <= 0:
                    continue
                velocity = (history[-1][1] - history[-2][1]) / (delta_f / self.fps)
                predicted = history[-1][1] + velocity * dt
                cv_dist = float(np.linalg.norm(det_xy[col] - predicted))
                if mode == "kv_pending":
                    kv_dist = float(kv_source["kv_dist"])
                    support = int(kv_source["support_tracks"])
                    spread = float(kv_source["spread_m"])
                    reliable_kv = support >= 2 and spread <= 6.0 and kv_score is not None and kv_score >= 0.30
                    if reliable_kv and kv_dist + 1.5 < cv_dist and reid >= reid_need + 0.04:
                        dist = kv_dist
                        radius = min(self.kv_cap, self.kv_base + self.kv_gain * dt)
                        mode = "kv"
                    else:
                        dist = cv_dist
                        mode = "blend_memory"
                elif mode != "kv":
                    dist = cv_dist
                if history_speed < 1.5:
                    radius = min(radius, 3.0)
                if dist is None or dist > radius:
                    continue
                topology_reason = None
                if self.use_topology:
                    allowed, topology_reason, topology_info = self._topology_decision(int(track.class_id), origin, det_xy[col])
                    if not allowed:
                        self._log(
                            frame,
                            {
                                "label": int(track.label),
                                "mode": "topology_reject",
                                "reid": reid,
                                "dist": dist if dist is not None else -1,
                                "score": 0.0,
                                "dt": dt,
                                "kv_score": kv_score,
                                "kv_source": {"reason": topology_reason, "lanes": topology_info},
                                "uses_topology": True,
                                "pred_xy": origin.tolist(),
                            },
                            "reject",
                        )
                        continue
                    uses_topology = topology_reason == "道路连通"
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
            "kv_future": None if not item.get("kv_source") else item["kv_source"].get("futures"),
        }
        self._log_lines.append(json.dumps(record, ensure_ascii=False) + "\n")
