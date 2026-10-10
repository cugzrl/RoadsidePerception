"""跨摄像头投影误差与是否允许融合"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from r39_rag.config import PROJECT_ROOT, load_sources
from r39_rag.projection.homography import load_projectors
from r39_rag.projection.quality import project_observation


def iter_observations(sequence_ids: list[str]) -> Iterator[dict[str, Any]]:
    root = PROJECT_ROOT / "data" / "normalized" / "sequences"
    for sequence_id in sequence_ids:
        version_root = root / sequence_id / "versions"
        if not version_root.exists():
            continue
        for version_dir in sorted(path for path in version_root.iterdir() if path.is_dir()):
            path = version_dir / "observations.jsonl"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row["sequence_id"] = sequence_id
                    row["source_version"] = version_dir.name
                    yield row


def _empty_counts() -> dict[str, int]:
    return {
        "input": 0,
        "ok": 0,
        "invalid_bbox": 0,
        "bbox_out_of_frame": 0,
        "border_contact": 0,
        "non_finite": 0,
        "out_of_bounds": 0,
        "reassigned_ok": 0,
    }


def evaluate_sequences(sequence_ids: list[str], config: dict[str, Any]) -> dict[str, Any]:
    sources = load_sources()
    projectors = load_projectors(config["calibration_dir"], sources["cameras"], float(config["min_denominator"]))
    counts = _empty_counts()
    per_camera: dict[str, dict[str, int]] = {camera_id: _empty_counts() for camera_id in sources["cameras"]}
    per_sequence: dict[str, dict[str, int]] = {sequence_id: _empty_counts() for sequence_id in sequence_ids}
    groups: dict[tuple[str, str, int, int], list[tuple[str, np.ndarray]]] = defaultdict(list)
    sample_limit = 6000
    samples: dict[str, list[np.ndarray]] = {camera_id: [] for camera_id in sources["cameras"]}
    sample_seen = {camera_id: 0 for camera_id in sources["cameras"]}

    for row in iter_observations(sequence_ids):
        camera_id = row["camera_id"]
        projector = projectors.get(camera_id)
        if projector is None:
            continue
        projected = project_observation(row, projector, config)
        counts["input"] += 1
        per_camera[camera_id]["input"] += 1
        per_sequence[row["sequence_id"]]["input"] += 1
        status = projected["status"]
        if status != "ok":
            counts[status] += 1
            per_camera[camera_id][status] += 1
            per_sequence[row["sequence_id"]][status] += 1
            continue
        counts["ok"] += 1
        per_camera[camera_id]["ok"] += 1
        per_sequence[row["sequence_id"]]["ok"] += 1
        point = np.asarray([projected["x"], projected["y"]], dtype=np.float64)
        if projected["reassigned"]:
            counts["reassigned_ok"] += 1
            per_camera[camera_id]["reassigned_ok"] += 1
            per_sequence[row["sequence_id"]]["reassigned_ok"] += 1
        else:
            global_id = row.get("global_track_id")
            if global_id is not None:
                groups[(row["sequence_id"], row["source_version"], int(row["frame_index"]), int(global_id))].append((camera_id, point))
        sample_seen[camera_id] += 1
        bucket = samples[camera_id]
        if len(bucket) < sample_limit:
            bucket.append(point)
        else:
            index = int(np.random.randint(0, sample_seen[camera_id]))
            if index < sample_limit:
                bucket[index] = point

    distances: list[float] = []
    pair_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for items in groups.values():
        by_camera: dict[str, np.ndarray | None] = {}
        for camera_id, point in items:
            if camera_id in by_camera:
                by_camera[camera_id] = None
            else:
                by_camera[camera_id] = point
        present = [(camera_id, point) for camera_id, point in by_camera.items() if point is not None]
        if len(present) < 2:
            continue
        for left in range(len(present)):
            for right in range(left + 1, len(present)):
                distance = float(np.linalg.norm(present[left][1] - present[right][1]))
                distances.append(distance)
                pair_values[tuple(sorted((present[left][0], present[right][0])))].append(distance)

    distance_array = np.asarray(distances, dtype=np.float64)
    pair_stats = []
    for pair, values in sorted(pair_values.items()):
        array = np.asarray(values, dtype=np.float64)
        pair_stats.append(
            {
                "camera_a": pair[0],
                "camera_b": pair[1],
                "count": int(array.size),
                "median_m": float(np.median(array)),
                "p90_m": float(np.percentile(array, 90)),
                "fraction_within_5m": float(np.mean(array <= 5)),
                "fraction_within_10m": float(np.mean(array <= 10)),
            }
        )
    overall = {
        "count": int(distance_array.size),
        "median_m": None if distance_array.size == 0 else float(np.median(distance_array)),
        "p90_m": None if distance_array.size == 0 else float(np.percentile(distance_array, 90)),
        "fraction_within_5m": None if distance_array.size == 0 else float(np.mean(distance_array <= 5)),
        "fraction_within_10m": None if distance_array.size == 0 else float(np.mean(distance_array <= 10)),
    }
    reliable_pairs = [item for item in pair_stats if item["count"] >= int(config["fusion_min_pair_count"])]
    passing_pairs = [item for item in reliable_pairs if item["median_m"] <= float(config["fusion_median_limit_m"])]
    allow = (
        overall["median_m"] is not None
        and overall["median_m"] <= float(config["fusion_median_limit_m"])
        and len(passing_pairs) >= int(config["fusion_min_pairs"])
    )
    return {
        "sequence_ids": sequence_ids,
        "coordinate_frame": config["coordinate_frame"],
        "homography_convention": config["homography_convention"],
        "counts": counts,
        "per_camera": per_camera,
        "per_sequence": per_sequence,
        "pairs": pair_stats,
        "overall": overall,
        "allow_fusion": allow,
        "passing_pairs": len(passing_pairs),
        "reliable_pairs": len(reliable_pairs),
        "limit_m": float(config["fusion_median_limit_m"]),
        "samples": {camera_id: np.stack(points) if points else np.zeros((0, 2)) for camera_id, points in samples.items()},
        "distances": distance_array,
    }


def write_projection_report(result: dict[str, Any], path: Path) -> None:
    counts = result["counts"]
    valid_ratio = 0 if counts["input"] == 0 else counts["ok"] / counts["input"]
    lines = [
        "# BEV 投影质量",
        "",
        "地面点使用检测框底边中点。像素到路面的变换与 `algorithms/stmc/src/tracker/geometry.py` 的 `image_to_world` 相同，即对标定文件中的 `homography` 取逆。得到的是地面二维坐标，不是车辆三维框。",
        "",
        "甲方 `camera_box_3d` 尺寸和未核实地面点没有参与投影。贴着图像边缘的底边不能代表车轮接地点，超出配置范围的投影视为单应外推，两者都不进入知识库。",
        "",
        f"检查序列：{', '.join(result['sequence_ids'])}。",
        "",
        "## 数量",
        "",
        f"- 输入观测 {counts['input']}",
        f"- 可用于建库 {counts['ok']}，比例 {valid_ratio:.3f}",
        f"- 其中自动拆分 ID 后仍落在范围内的观测 {counts['reassigned_ok']}，不参与高可信跨相机误差",
        f"- 无效框 {counts['invalid_bbox']}，越界框 {counts['bbox_out_of_frame']}，贴边 {counts['border_contact']}，非有限值 {counts['non_finite']}，平面越界 {counts['out_of_bounds']}",
        "",
        "## 各相机可用观测",
        "",
        "| 相机 | 输入 | 可用 | 贴边 | 平面越界 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for camera_id, item in result["per_camera"].items():
        lines.append(f"| {camera_id} | {item['input']} | {item['ok']} | {item['border_contact']} | {item['out_of_bounds']} |")
    overall = result["overall"]
    lines.extend(
        [
            "",
            "## 同一时刻同一全局 ID 的跨相机误差",
            "",
            "只比较未拆分 ID、同一帧每路相机只有一个可用投影的观测。不同序列的全局 ID 没有放在一起。",
            "",
            f"配对 {overall['count']} 条，中位误差 {overall['median_m']:.2f} m，90 分位 {overall['p90_m']:.2f} m，5 m 以内比例 {overall['fraction_within_5m']:.3f}，10 m 以内比例 {overall['fraction_within_10m']:.3f}。",
            "",
            "| 相机对 | 配对数 | 中位 m | 90 分位 m | 5 m 以内 | 10 m 以内 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in result["pairs"]:
        lines.append(
            f"| {item['camera_a']} / {item['camera_b']} | {item['count']} | {item['median_m']:.2f} | {item['p90_m']:.2f} | {item['fraction_within_5m']:.3f} | {item['fraction_within_10m']:.3f} |"
        )
    decision = "允许把四路投影写入同一 BEV 知识库" if result["allow_fusion"] else "不融合四路空间坐标"
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"{decision}。通过的相机对 {result['passing_pairs']} / 有效相机对 {result['reliable_pairs']}，阈值是中位误差不超过 {result['limit_m']} m 且至少 4 对达到样本量。",
            "",
            "直接使用标定矩阵而不取逆时，四路点不能落到同一数值范围，因此保留 STMC 的取逆约定。中位误差数米来自底边接地点近似和标定残差。长尾主要来自错误全局关联，以及底边被图像边界截断的近处目标，这些点已排除在建库和本表之外。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
