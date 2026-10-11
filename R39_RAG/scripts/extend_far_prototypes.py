"""从四段建库序列补充车道上的远端典型路径 不改原 sqlite 和 npz"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from r39_rag.config import PROJECT_ROOT, load_knowledge  # noqa: E402
from r39_rag.knowledge.online import LaneIndex  # noqa: E402
from r39_rag.knowledge.trajectory import build_prototypes  # noqa: E402
from r39_rag.projection.evaluate import iter_observations  # noqa: E402
from r39_rag.projection.homography import load_projectors  # noqa: E402
from r39_rag.projection.quality import project_observation  # noqa: E402
from r39_rag.config import load_sources  # noqa: E402


def main() -> None:
    config = load_knowledge()
    wide = dict(config)
    wide["max_abs_x_m"] = 120
    wide["max_abs_y_m"] = 120
    sources = load_sources()
    projectors = load_projectors(config["calibration_dir"], sources["cameras"], float(config["min_denominator"]))
    lanes = LaneIndex.load()
    grouped: dict[tuple, list[dict]] = {}
    for row in iter_observations(list(config["build_sequences"])):
        local_id = row.get("local_track_id")
        projector = projectors.get(row["camera_id"])
        if local_id is None or projector is None:
            continue
        projected = project_observation(row, projector, wide)
        if projected["status"] != "ok" or projected["x"] is None:
            continue
        key = (row["sequence_id"], row["camera_id"], int(local_id))
        grouped.setdefault(key, []).append(
            {
                "frame": int(row["frame_index"]),
                "time": row.get("timestamp_ms"),
                "x": projected["x"],
                "y": projected["y"],
                "reassigned": projected["reassigned"],
            }
        )

    tracks = []
    outside = 0
    for (sequence_id, camera_id, local_id), points in grouped.items():
        ordered = sorted(points, key=lambda item: (item["frame"], item["time"] or 0))
        if any(item["reassigned"] for item in ordered):
            continue
        xy = np.asarray([[item["x"], item["y"]] for item in ordered], dtype=np.float64)
        if float(np.max(np.abs(xy))) > 120.0:
            continue
        nearest = np.linalg.norm(lanes._cloud[None, :, :] - xy[:, None, :], axis=-1).min(axis=1)
        if float((nearest <= 6.0).mean()) < 0.8:
            continue
        far = np.any((np.abs(xy[:, 0]) > 60.0) | (np.abs(xy[:, 1]) > 60.0))
        if not far:
            continue
        outside += 1
        jumped = False
        speeds = []
        for index in range(1, len(ordered)):
            gap = ordered[index]["frame"] - ordered[index - 1]["frame"]
            distance = float(np.hypot(xy[index, 0] - xy[index - 1, 0], xy[index, 1] - xy[index - 1, 1]))
            if gap <= 0 or distance > float(config["jump_step_m"]) * max(gap, 1):
                jumped = True
            dt = ordered[index]["time"]
            before = ordered[index - 1]["time"]
            if gap == 1 and dt is not None and before is not None and float(dt) > float(before):
                speeds.append(distance / ((float(dt) - float(before)) / 1000.0))
        tracks.append(
            {
                "sequence_id": sequence_id,
                "camera_id": camera_id,
                "local_track_id": local_id,
                "reassigned": False,
                "jumped": jumped,
                "median_speed": None if not speeds else float(np.median(speeds)),
                "points": xy.tolist(),
            }
        )
    prototypes = build_prototypes(tracks, config)
    samples = int(config["prototype_samples"])
    points = np.zeros((len(prototypes), samples, 2), dtype=np.float64)
    speeds = np.full(len(prototypes), np.nan, dtype=np.float64)
    supports = np.zeros(len(prototypes), dtype=np.float64)
    for index, item in enumerate(prototypes):
        points[index] = item["points"]
        supports[index] = item["support_tracks"]
        if item["median_speed"] is not None:
            speeds[index] = item["median_speed"]
    target = PROJECT_ROOT / "knowledge_base" / "far_road_prototypes.npz"
    np.savez(
        target,
        prototypes=points,
        speeds=speeds,
        supports=supports,
        source_sequences=np.asarray(list(config["build_sequences"])),
    )
    cloud = lanes._cloud
    print(f"far_tracks {outside} prototypes {len(prototypes)} saved {target}")
    if len(cloud):
        print(
            "lane_bbox",
            float(cloud[:, 0].min()),
            float(cloud[:, 0].max()),
            float(cloud[:, 1].min()),
            float(cloud[:, 1].max()),
        )
    if len(points):
        print(
            "far_bbox",
            float(points[:, :, 0].min()),
            float(points[:, :, 0].max()),
            float(points[:, :, 1].min()),
            float(points[:, :, 1].max()),
        )


if __name__ == "__main__":
    main()
