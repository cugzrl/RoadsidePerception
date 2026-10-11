"""空间与轨迹检索"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from r39_rag.config import PROJECT_ROOT, load_knowledge
from r39_rag.knowledge.online import LaneIndex, TrustGrid, link_decision, match_prefix
from r39_rag.knowledge.spatial import cell_index
from r39_rag.knowledge.store import connect, load_arrays
from r39_rag.knowledge.trajectory import resample_path


def direction_name(bin_id: int) -> str:
    start = bin_id * 45
    return f"{start}° 到 {start + 45}°，相对路面 +X 轴"


class KnowledgeBase:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (PROJECT_ROOT / "knowledge_base")
        self.config = load_knowledge()
        database = self.root / "r39_knowledge.sqlite"
        arrays = self.root / "knowledge_arrays.npz"
        if not database.exists() or not arrays.exists():
            raise FileNotFoundError("知识库文件不存在，请先构建")
        self.connection = connect(database)
        self.arrays = load_arrays(arrays)
        self.lanes = LaneIndex.load()
        self.trust = TrustGrid.load()

    def close(self) -> None:
        self.connection.close()

    def query_spatial(
        self,
        x_value: float,
        y_value: float,
        camera_id: str | None = None,
        class_name: str | None = None,
    ) -> dict[str, Any]:
        index = cell_index(x_value, y_value, self.config)
        if index is None:
            return {"found": False, "reason": "位置超出建库 BEV 范围", "x": x_value, "y": y_value}
        ix, iy = index
        row = self.connection.execute("SELECT * FROM cells WHERE ix = ? AND iy = ?", (ix, iy)).fetchone()
        if row is None:
            return {"found": False, "reason": "该网格没有历史观测", "ix": ix, "iy": iy, "x": x_value, "y": y_value}
        identifier = int(row["cell_id"])
        transitions = self.connection.execute(
            """
            SELECT to_cell, track_count, median_dt_ms, confidence
            FROM transitions WHERE from_cell = ? ORDER BY track_count DESC LIMIT 8
            """,
            (identifier,),
        ).fetchall()
        class_counts = json.loads(row["class_counts"])
        camera_counts = json.loads(row["camera_counts"])
        direction = {direction_name(int(key)): value for key, value in json.loads(row["direction_hist"]).items()}
        result = {
            "found": True,
            "cell_id": identifier,
            "ix": ix,
            "iy": iy,
            "bounds_m": [row["x_min"], row["y_min"], row["x_max"], row["y_max"]],
            "observation_count": row["observation_count"],
            "local_track_count": row["local_track_count"],
            "clean_track_count": row["clean_track_count"],
            "global_track_count": row["global_track_count"],
            "class_counts": class_counts,
            "direction_hist": direction,
            "speed_p50_m_s": row["speed_p50"],
            "speed_p90_m_s": row["speed_p90"],
            "camera_counts": camera_counts,
            "observation_reliability": row["observation_reliability"],
            "confidence": row["confidence"],
            "source_sequences": json.loads(row["source_sequences"]),
            "next_cells": [
                {
                    "cell_id": item["to_cell"],
                    "track_count": item["track_count"],
                    "median_dt_ms": item["median_dt_ms"],
                    "confidence": item["confidence"],
                }
                for item in transitions
            ],
            "coordinate_frame": self.config["coordinate_frame"],
        }
        if camera_id:
            result["query_camera"] = camera_counts.get(camera_id)
        if class_name:
            result["query_class_tracks"] = class_counts.get(class_name, 0)
        return result

    def query_trajectory(self, points: list[list[float]], camera_id: str | None = None) -> dict[str, Any]:
        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2 or len(array) == 0:
            raise ValueError("轨迹需要 Nx2 的地面坐标")
        samples = int(self.config["prototype_samples"])
        feature = resample_path(array, samples).reshape(-1)
        stored = np.asarray(self.arrays["prototypes"])
        matches = []
        if len(stored):
            distance = np.linalg.norm(stored.reshape(len(stored), -1) - feature, axis=1) / np.sqrt(samples)
            order = np.argsort(distance)[:5]
            sigma = float(self.config["retrieval_sigma_m"])
            rows = self.connection.execute("SELECT * FROM prototypes").fetchall()
            by_id = {int(row["prototype_id"]): row for row in rows}
            for index in order:
                row = by_id.get(int(index))
                if row is None:
                    continue
                matches.append(
                    {
                        "prototype_id": int(index),
                        "kind": row["kind"],
                        "score": float(np.exp(-float(distance[index]) / sigma)),
                        "mean_point_distance_m": float(distance[index]),
                        "support_tracks": row["support_tracks"],
                        "support_sequences": json.loads(row["support_sequences"]),
                        "camera_counts": json.loads(row["camera_counts"]),
                        "median_speed_m_s": row["median_speed"],
                        "confidence": row["confidence"],
                        "exemplar_sequence": row["exemplar_sequence"],
                        "exemplar_camera": row["exemplar_camera"],
                        "exemplar_local_track_id": row["exemplar_local_track_id"],
                    }
                )
        end = self.query_spatial(float(array[-1, 0]), float(array[-1, 1]), camera_id=camera_id)
        cross = []
        if camera_id:
            found = self.connection.execute(
                """
                SELECT * FROM cross_camera
                WHERE camera_a = ? OR camera_b = ?
                ORDER BY support_count DESC LIMIT 8
                """,
                (camera_id, camera_id),
            ).fetchall()
            cross = [dict(item) for item in found]
        return {
            "matches": matches,
            "end_cell": end,
            "cross_camera": cross,
            "coordinate_frame": self.config["coordinate_frame"],
            "note": "分数由历史路径形状距离换算，同时给出支持轨迹数和可信度，不是未来位置的确定预测",
        }

    def query_prefix(self, points: list[list[float]], top_k: int = 5) -> dict[str, Any]:
        """用已经看到的轨迹前缀对齐典型路径中的一段"""
        array = np.asarray(points, dtype=np.float64)
        stored = np.asarray(self.arrays["prototypes"])
        sigma = float(self.config["retrieval_sigma_m"])
        matches = match_prefix(array, stored, sigma_m=sigma, top_k=top_k)
        rows = {int(row["prototype_id"]): row for row in self.connection.execute("SELECT * FROM prototypes")}
        detailed = []
        for item in matches:
            row = rows.get(item["prototype_id"])
            if row is None:
                continue
            detailed.append(
                {
                    **item,
                    "kind": row["kind"],
                    "support_tracks": row["support_tracks"],
                    "support_sequences": json.loads(row["support_sequences"]),
                    "confidence": row["confidence"],
                    "median_speed_m_s": row["median_speed"],
                    "exemplar_sequence": row["exemplar_sequence"],
                }
            )
        min_score = 0.45
        reliable = bool(detailed) and detailed[0]["score"] >= min_score
        return {
            "reliable": reliable,
            "matches": detailed if reliable else [],
            "rejected_matches": [] if reliable else detailed[:top_k],
            "reason": None if reliable else "历史路径匹配不足 不给出后续区域",
            "note": "只比较已经观测到的前缀 分数和支持轨迹数都要保留 不是确定的未来轨迹",
        }

    def query_region(self, x_value: float, y_value: float) -> dict[str, Any]:
        point = np.array([x_value, y_value], dtype=np.float64)
        lane = self.lanes.locate(point)
        spatial = self.query_spatial(x_value, y_value)
        trust = self.trust.query(point) if self.trust is not None else {"found": False, "level": "样本不足", "neutral": True}
        next_lanes = []
        if lane is not None:
            for child in self.lanes.successors.get(lane["lane_id"], [])[:8]:
                next_lanes.append(child)
        return {
            "lane": None
            if lane is None
            else {
                "lane_id": lane["lane_id"],
                "arm": lane["arm"],
                "role": lane["role"],
                "distance_m": lane["distance_m"],
            },
            "next_lanes": next_lanes,
            "spatial_cell": spatial,
            "trust": trust,
            "note": "共同观测和时序转移在 spatial_cell 的来源统计里分开存放 可信度不是关联准确率",
        }

    def query_link(self, history: list[list[float]], candidate: list[float], frame_gap: int = 1) -> dict[str, Any]:
        stored = np.asarray(self.arrays["prototypes"])
        return link_decision(
            np.asarray(history, dtype=np.float64),
            np.asarray(candidate, dtype=np.float64),
            stored,
            self.lanes,
            self.trust,
            sigma_m=float(self.config["retrieval_sigma_m"]),
            top_k=5,
            min_score=0.45,
            continuation_m=8.0,
            min_points=3,
            frame_gap=frame_gap,
        )
