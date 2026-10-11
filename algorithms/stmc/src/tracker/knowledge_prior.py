"""把固定历史知识库转成有界的关联加成 知识关闭时不加载"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


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
        )

    def _delta(self, decision: dict, base_weight: float) -> float:
        if not decision.get("reliable"):
            return 0.0
        score = float(decision.get("score") or 0.0)
        action = decision.get("action")
        if action == "continuation":
            floor = 0.05 + self.weight * score
            if base_weight < floor:
                return float(floor - base_weight)
            return float(self.weight * score)
        if action == "topology":
            return float(min(self.weight, 0.1) * score)
        if action == "hold":
            # 外观已经强烈反对时不抬权重 避免把近处的另一辆车绑进来
            if base_weight < -0.3:
                return 0.0
            floor = 0.05
            if base_weight < floor:
                return float(floor - base_weight)
            return 0.0
        if action == "contradict":
            return float(-self.weight * score)
        return 0.0

    def flush(self) -> None:
        if not self.log_path:
            return
        path = Path(str(self.log_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for row in self.logs:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
