"""轨迹平滑、区域转移和典型路径"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from r39_rag.knowledge.spatial import confidence_label


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def smooth_path(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return points.copy()
    smoothed = points.copy()
    smoothed[1:-1] = (points[:-2] + points[1:-1] + points[2:]) / 3.0
    return smoothed


def resample_path(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((count, 2), dtype=np.float64)
    if len(points) == 1 or path_length(points) < 1e-6:
        return np.repeat(points[:1], count, axis=0)
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(step)])
    target = np.linspace(0.0, cumulative[-1], count)
    return np.stack(
        [np.interp(target, cumulative, points[:, 0]), np.interp(target, cumulative, points[:, 1])],
        axis=1,
    )


def turn_degrees(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    start = points[min(3, len(points) - 1)] - points[0]
    end = points[-1] - points[max(0, len(points) - 4)]
    if np.linalg.norm(start) < 1e-6 or np.linalg.norm(end) < 1e-6:
        return 0.0
    start_angle = np.arctan2(start[1], start[0])
    end_angle = np.arctan2(end[1], end[0])
    delta = (end_angle - start_angle + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(delta))


def path_kind(points: np.ndarray, config: dict[str, Any]) -> str:
    delta = turn_degrees(points)
    magnitude = abs(delta)
    if magnitude < float(config["straight_turn_deg"]):
        return "straight"
    if magnitude >= float(config["turn_deg"]):
        return "left" if delta > 0 else "right"
    return "other"


def _cluster_kind(items: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    features = np.stack([item["feature"] for item in items])
    if len(items) == 1:
        labels = np.zeros(1, dtype=int)
    else:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(config["cluster_distance_m"]),
            linkage="average",
            metric="euclidean",
        )
        labels = model.fit_predict(features)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[int(label)].append(index)
    prototypes = []
    for members in grouped.values():
        if len(members) < int(config["min_prototype_support"]):
            continue
        subset = features[members]
        center = np.median(subset, axis=0)
        distances = np.linalg.norm(subset - center, axis=1)
        exemplar = items[members[int(np.argmin(distances))]]
        sequences = sorted({items[index]["sequence_id"] for index in members})
        cameras: dict[str, int] = defaultdict(int)
        speeds = []
        for index in members:
            cameras[items[index]["camera_id"]] += 1
            if items[index]["speed"] is not None:
                speeds.append(items[index]["speed"])
        prototypes.append(
            {
                "kind": items[members[0]]["kind"],
                "support_tracks": len(members),
                "support_sequences": sequences,
                "camera_counts": dict(cameras),
                "median_speed": None if not speeds else float(np.median(speeds)),
                "confidence": confidence_label(len(members)),
                "points": exemplar["points"],
                "feature": exemplar["feature"],
                "exemplar_sequence": exemplar["sequence_id"],
                "exemplar_camera": exemplar["camera_id"],
                "exemplar_local_track_id": exemplar["local_track_id"],
            }
        )
    return prototypes


def build_prototypes(tracks: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    samples = int(config["prototype_samples"])
    prepared = []
    min_points = int(config["min_track_points"])
    min_length = float(config["min_track_length_m"])
    for track in tracks:
        if track["reassigned"] or track["jumped"]:
            continue
        points = np.asarray(track["points"], dtype=np.float64)
        if len(points) < min_points or path_length(points) < min_length:
            continue
        smoothed = smooth_path(points)
        sampled = resample_path(smoothed, samples)
        prepared.append(
            {
                "kind": path_kind(sampled, config),
                "feature": sampled.reshape(-1),
                "points": sampled,
                "sequence_id": track["sequence_id"],
                "camera_id": track["camera_id"],
                "local_track_id": track["local_track_id"],
                "speed": track.get("median_speed"),
            }
        )
    prototypes: list[dict[str, Any]] = []
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in prepared:
        by_kind[item["kind"]].append(item)
    for items in by_kind.values():
        prototypes.extend(_cluster_kind(items, config))
    prototypes.sort(key=lambda item: (-item["support_tracks"], item["kind"]))
    return prototypes[: int(config["max_prototypes"])]
