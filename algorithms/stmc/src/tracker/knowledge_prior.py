"""把固定历史知识库转成有界的关联加成 知识关闭时不加载"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def bounded_bonus(base_weight: float, bonus: float) -> float:
    """加分有上限 负边最多补到 0 不能被知识抬成正边"""
    bonus = max(0.0, float(bonus))
    if base_weight >= 0.0:
        return bonus
    return min(bonus, -float(base_weight))


def _load_api():
    root = Path(__file__).resolve().parents[4]
    rag_src = root / "R39_RAG" / "src"
    for path in (str(root), str(rag_src)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from r39_rag.knowledge.retrieval import KnowledgeBase

    return KnowledgeBase


class KnowledgePrior:
    """新关联权重 = 原 STMC 权重 + 这里返回的 delta

    delta 只在历史证据足够时非零 不能突破随后写入的互斥惩罚
    """

    def __init__(self, cfg):
        self.weight = float(getattr(cfg, "weight", 0.2))
        self.top_k = int(getattr(cfg, "top_k", 5))
        self.min_score = float(getattr(cfg, "min_score", 0.45))
        self.continuation_m = float(getattr(cfg, "continuation_m", 8.0))
        self.min_points = int(getattr(cfg, "min_prefix_points", 3))
        self.prefix_frames = int(getattr(cfg, "prefix_frames", 20))
        self.use_spatial = bool(getattr(cfg, "use_spatial", True))
        self.use_trajectory = bool(getattr(cfg, "use_trajectory", True))
        self.use_topology = bool(getattr(cfg, "use_topology", True))
        self.log_path = getattr(cfg, "log_path", None)
        self.logs: list[dict] = []
        self._match_cache: dict[bytes, list] = {}
        knowledge_base = _load_api()
        self.kb = knowledge_base()
        self.prototypes = np.asarray(self.kb.arrays["prototypes"]) if self.use_trajectory else np.zeros((0, 1, 2))
        self.prototype_speeds = None
        self.prototype_supports = None
        if self.use_trajectory and len(self.prototypes):
            count = len(self.prototypes)
            speeds = np.full(count, np.nan, dtype=np.float64)
            supports = np.zeros(count, dtype=np.float64)
            rows = self.kb.connection.execute("SELECT prototype_id, support_tracks, median_speed FROM prototypes")
            for row in rows:
                index = int(row["prototype_id"])
                if 0 <= index < count:
                    supports[index] = float(row["support_tracks"])
                    if row["median_speed"] is not None:
                        speeds[index] = float(row["median_speed"])
            far_path = self.kb.root / "far_road_prototypes.npz"
            if far_path.is_file():
                far = np.load(far_path)
                extra = np.asarray(far["prototypes"], dtype=np.float64)
                if extra.ndim == 3 and extra.shape[1:] == self.prototypes.shape[1:]:
                    self.prototypes = np.concatenate([self.prototypes, extra], axis=0)
                    speeds = np.concatenate([speeds, np.asarray(far["speeds"], dtype=np.float64)])
                    supports = np.concatenate([supports, np.asarray(far["supports"], dtype=np.float64)])
            self.prototype_speeds = speeds
            self.prototype_supports = supports
        self.sigma = float(self.kb.config["retrieval_sigma_m"])

    def adjust(self, history: np.ndarray, candidate: np.ndarray, base_weight: float, frame_gap: int) -> tuple[float, dict]:
        if frame_gap <= 0 or frame_gap > self.prefix_frames:
            return 0.0, {"action": "none", "reason": "时间间隔超出前缀窗口"}
        points = np.asarray(history, dtype=np.float64)
        if len(points) > self.prefix_frames:
            points = points[-self.prefix_frames :]
        prototypes = self.prototypes if self.use_trajectory else np.zeros((0, 1, 2))
        lanes = self.kb.lanes if self.use_topology else None
        trust = self.kb.trust if self.use_spatial else None
        decision = self.kb_decision(points, candidate, prototypes, lanes, trust, frame_gap)
        delta = self._delta(decision, base_weight)
        if delta != 0.0 and self.log_path:
            self.logs.append(
                {
                    "frame_gap": int(frame_gap),
                    "base_weight": float(base_weight),
                    "delta": float(delta),
                    "action": decision["action"],
                    "score": decision["score"],
                    "reasons": decision["reasons"],
                    "candidate": [float(candidate[0]), float(candidate[1])],
                }
            )
            if len(self.logs) % 50 == 0:
                self.flush()
        return delta, decision

    def kb_decision(self, points, candidate, prototypes, lanes, trust, frame_gap: int):
        from r39_rag.knowledge.online import link_decision

        key = np.ascontiguousarray(points, dtype=np.float64).tobytes()
        matches = self._match_cache.get(key)
        if matches is None:
            from r39_rag.knowledge.online import match_prefix

            matches = match_prefix(points, prototypes, sigma_m=self.sigma, top_k=self.top_k)
            if len(self._match_cache) > 4000:
                self._match_cache.clear()
            self._match_cache[key] = matches
        return link_decision(
            points,
            np.asarray(candidate, dtype=np.float64),
            prototypes,
            lanes,
            trust,
            sigma_m=self.sigma,
            top_k=self.top_k,
            min_score=self.min_score,
            continuation_m=self.continuation_m,
            min_points=self.min_points,
            frame_gap=frame_gap,
            cached_matches=matches if len(points) >= self.min_points else None,
            prototype_speeds=self.prototype_speeds,
            prototype_supports=self.prototype_supports,
        )

    def _delta(self, decision: dict, base_weight: float) -> float:
        if not decision.get("reliable"):
            return 0.0
        score = float(decision.get("score") or 0.0)
        action = decision.get("action")
        cap = min(float(self.weight), 0.12) * max(score, 0.0)
        if action == "continuation":
            return bounded_bonus(base_weight, cap)
        if action == "topology":
            if base_weight < -0.05:
                return 0.0
            return bounded_bonus(base_weight, min(cap, 0.10 * score))
        if action == "hold":
            # 自身运动延续 幅度小于历史路径 负边仍然不能变成正边
            if base_weight < -0.3:
                return 0.0
            return bounded_bonus(base_weight, min(cap, 0.05))
        if action == "contradict":
            return float(-min(float(self.weight), 0.12) * score)
        return 0.0

    def flush(self) -> None:
        if not self.log_path:
            return
        path = Path(str(self.log_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for row in self.logs:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
