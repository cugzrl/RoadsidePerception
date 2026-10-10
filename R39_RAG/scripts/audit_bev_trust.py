#!/usr/bin/env python3
"""在车道级 BEV 上检查投影有效范围和跨相机误差长尾"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
RAG_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RAG_SRC))

from pipeline.bev_map import (  # noqa: E402
    BEVBasemap,
    LaneMap,
    arm_directions,
    build_road_mask,
    draw_camera_markers,
    load_camera_poses,
    make_r39_view,
    short_camera_name,
)
from r39_rag.config import PROJECT_ROOT, load_knowledge, load_sources  # noqa: E402
from r39_rag.projection.evaluate import iter_observations  # noqa: E402
from r39_rag.projection.homography import bottom_center, load_projectors  # noqa: E402

CAMERAS = ["R39_Aw_CamS", "R39_Bn_CamW", "R39_Ce_CamN", "R39_Ds_CamE"]
RINGS = ((25.0, "路口核心 0-25 m"), (55.0, "近臂 25-55 m"), (82.0, "车道外端 55-82 m"), (1e9, "超出车道图"))
TAIL_M = 15.0
CELL_M = 10.0
PAIR_COLORS = {
    ("R39_Aw_CamS", "R39_Bn_CamW"): (80, 180, 255),
    ("R39_Aw_CamS", "R39_Ce_CamN"): (180, 80, 255),
    ("R39_Aw_CamS", "R39_Ds_CamE"): (80, 255, 180),
    ("R39_Bn_CamW", "R39_Ce_CamN"): (255, 180, 80),
    ("R39_Bn_CamW", "R39_Ds_CamE"): (255, 80, 180),
    ("R39_Ce_CamN", "R39_Ds_CamE"): (80, 220, 255),
}


def project_point(row: dict, projector, config: dict) -> tuple[str, np.ndarray | None, bool]:
    flags = row.get("quality_flags") or []
    reassigned = (
        "id_reassigned" in flags
        or row.get("raw_local_track_id") is not None
        or row.get("raw_global_track_id") is not None
    )
    bbox = row.get("bbox_xyxy")
    if "invalid_bbox" in flags or not isinstance(bbox, list) or len(bbox) != 4:
        return "invalid_bbox", None, reassigned
    if "bbox_out_of_frame" in flags:
        return "bbox_out_of_frame", None, reassigned
    pixel_x, pixel_y = bottom_center(bbox)
    border = float(config["border_px"])
    width = int(config["image_width"])
    height = int(config["image_height"])
    if pixel_x < border or pixel_y < border or pixel_x > width - border or pixel_y > height - border:
        return "border_contact", None, reassigned
    world, singular = projector.image_to_world(np.asarray([[pixel_x, pixel_y]], dtype=np.float64))
    if singular[0] or not np.isfinite(world[0]).all():
        return "non_finite", None, reassigned
    return "ok", world[0].astype(np.float64), reassigned


def ring_index(distance: float) -> int:
    for index, (limit, _name) in enumerate(RINGS):
        if distance < limit:
            return index
    return len(RINGS) - 1


def summarize(distance: np.ndarray) -> dict:
    if len(distance) == 0:
        return {"count": 0, "median": None, "p90": None, "within_5": None, "within_8": None, "tail_15": None, "tail_30": None}
    return {
        "count": int(len(distance)),
        "median": float(np.median(distance)),
        "p90": float(np.percentile(distance, 90)),
        "within_5": float(np.mean(distance <= 5.0)),
        "within_8": float(np.mean(distance <= 8.0)),
        "tail_15": float(np.mean(distance > TAIL_M)),
        "tail_30": float(np.mean(distance > 30.0)),
    }


def pair_level(stats: dict) -> str:
    count = stats["count"]
    if count < 50 or stats["median"] is None:
        return "样本不足"
    if count >= 200 and stats["median"] <= 5 and stats["p90"] <= 12 and stats["tail_15"] <= 0.20:
        return "高"
    if count >= 100 and stats["median"] <= 8 and stats["p90"] <= 25 and stats["tail_15"] <= 0.35:
        return "中"
    return "低"


def cell_level(count: int, median: float, tail: float, on_road: bool, distance: float) -> str:
    if not on_road or distance > 82:
        return "范围外"
    if count >= 40 and median <= 5 and tail <= 0.20:
        return "高"
    if count >= 20 and median <= 8 and tail <= 0.35:
        return "中"
    if count >= 10:
        return "低"
    return "样本不足"


def fmt(stats: dict) -> str:
    if stats["count"] == 0:
        return "0 对"
    return (
        f"{stats['count']} 对 中位 {stats['median']:.2f} m "
        f"p90 {stats['p90']:.2f} m ≤5 m {stats['within_5']:.1%} "
        f">15 m {stats['tail_15']:.1%}"
    )


def draw_chinese(image: np.ndarray, lines: list[str], font_path: str) -> np.ndarray:
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(font_path, 22)
    y_value = 16
    for line in lines:
        draw.text((16, y_value), line, font=font, fill=(245, 245, 245), stroke_width=2, stroke_fill=(0, 0, 0))
        y_value += 28
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def main() -> int:
    config = load_knowledge()
    sources = load_sources()
    sequences = list(config["build_sequences"]) + list(config["holdout_sequences"])
    projectors = load_projectors(config["calibration_dir"], sources["cameras"], float(config["min_denominator"]))
    graph = LaneMap.load()
    center = graph.center
    directions = arm_directions(graph)
    arm_names = [name for name in ("N", "E", "S", "W") if name in directions]
    arm_vectors = np.stack([directions[name] for name in arm_names], axis=0)
    strict = build_road_mask(graph, resolution=1.0, dilate_m=0.0)
    road = build_road_mask(graph, resolution=1.0, dilate_m=2.0)

    counts = defaultdict(int)
    point_ring = np.zeros(len(RINGS), dtype=np.int64)
    point_ring_road = np.zeros(len(RINGS), dtype=np.int64)
    groups: dict[tuple, list[tuple[str, np.ndarray]]] = defaultdict(list)
    kept_points: list[np.ndarray] = []

    for row in iter_observations(sequences):
        projector = projectors.get(row["camera_id"])
        if projector is None:
            continue
        counts["input"] += 1
        status, point, reassigned = project_point(row, projector, config)
        if status != "ok":
            counts[status] += 1
            continue
        if float(np.linalg.norm(point - center)) > 200:
            counts["blowup"] += 1
            continue
        counts["projected"] += 1
        if reassigned:
            counts["reassigned"] += 1
            continue
        kept_points.append(point)
        global_id = row.get("global_track_id")
        if global_id is None:
            continue
        groups[(row["sequence_id"], row["source_version"], int(row["frame_index"]), int(global_id))].append(
            (row["camera_id"], point)
        )

    if kept_points:
        kept = np.stack(kept_points, axis=0)
        kept_distance = np.linalg.norm(kept - center, axis=1)
        kept_ring = np.array([ring_index(value) for value in kept_distance], dtype=np.int64)
        kept_road = road.contains(kept)
        kept_polygon = strict.contains(kept)
        in_box = (np.abs(kept[:, 0]) <= 60) & (np.abs(kept[:, 1]) <= 60)
        for index in range(len(RINGS)):
            chosen = kept_ring == index
            point_ring[index] = int(np.sum(chosen))
            point_ring_road[index] = int(np.sum(chosen & kept_road))
        counts["on_road"] = int(np.sum(kept_road))
        counts["on_polygon"] = int(np.sum(kept_polygon))
        counts["in_origin_box"] = int(np.sum(in_box))
        counts["box_and_road"] = int(np.sum(in_box & kept_road))

    pair_distance = []
    pair_mid = []
    pair_left = []
    pair_right = []
    pair_a = []
    pair_b = []
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
        for left_index in range(len(present)):
            for right_index in range(left_index + 1, len(present)):
                a_name, a_point = present[left_index]
                b_name, b_point = present[right_index]
                ordered = sorted(((a_name, a_point), (b_name, b_point)), key=lambda item: item[0])
                pair_left.append(ordered[0][0])
                pair_right.append(ordered[1][0])
                pair_a.append(ordered[0][1])
                pair_b.append(ordered[1][1])
                pair_distance.append(float(np.linalg.norm(a_point - b_point)))
                pair_mid.append(0.5 * (a_point + b_point))
    if pair_a:
        both_road = road.contains(np.stack(pair_a)) & road.contains(np.stack(pair_b))
    else:
        both_road = np.zeros(0, dtype=bool)

    distance = np.asarray(pair_distance, dtype=np.float64)
    midpoint = np.asarray(pair_mid, dtype=np.float64).reshape(-1, 2) if pair_mid else np.zeros((0, 2))
    left = np.asarray(pair_left)
    right = np.asarray(pair_right)
    if len(midpoint):
        offset = midpoint - center
        mid_distance = np.linalg.norm(offset, axis=1)
        mid_ring = np.array([ring_index(value) for value in mid_distance])
        score = offset @ arm_vectors.T
        arm_index = np.argmax(score, axis=1)
        arm_index = np.where(mid_distance < 25.0, -1, arm_index)
    else:
        mid_distance = np.zeros(0)
        mid_ring = np.zeros(0, dtype=int)
        arm_index = np.zeros(0, dtype=int)

    overall = summarize(distance)
    road_stats = summarize(distance[both_road]) if len(distance) else summarize(distance)
    off_stats = summarize(distance[~both_road]) if len(distance) else summarize(distance)

    pair_rows = []
    for a_name, b_name in PAIR_COLORS:
        chosen = (left == a_name) & (right == b_name) if len(distance) else np.zeros(0, dtype=bool)
        near = chosen & both_road & (mid_distance <= 55)
        far = chosen & both_road & (mid_distance > 55) & (mid_distance <= 82)
        pair_rows.append(
            {
                "pair": f"{short_camera_name(a_name)} / {short_camera_name(b_name)}",
                "all": summarize(distance[chosen]),
                "road_near": summarize(distance[near]) if len(distance) else summarize(distance),
                "road_far": summarize(distance[far]) if len(distance) else summarize(distance),
                "level_near": pair_level(summarize(distance[near]) if np.any(near) else summarize(np.zeros(0))),
                "level_all": pair_level(summarize(distance[chosen]) if np.any(chosen) else summarize(np.zeros(0))),
            }
        )

    ring_rows = []
    for index, (_limit, name) in enumerate(RINGS):
        chosen = mid_ring == index
        ring_rows.append(
            {
                "name": name,
                "points": int(point_ring[index]),
                "points_road": int(point_ring_road[index]),
                "pairs": summarize(distance[chosen]) if len(distance) else summarize(distance),
                "pairs_road": summarize(distance[chosen & both_road]) if len(distance) else summarize(distance),
            }
        )

    arm_rows = []
    core = arm_index == -1
    arm_rows.append({"name": "路口核心", "pairs_road": summarize(distance[core & both_road]) if len(distance) else summarize(distance)})
    for index, name in enumerate(arm_names):
        chosen = (arm_index == index) & both_road & (mid_distance <= 82)
        arm_rows.append({"name": f"{name} 臂", "pairs_road": summarize(distance[chosen]) if len(distance) else summarize(distance)})

    cells: dict[tuple[int, int], list[float]] = defaultdict(list)
    cell_road_votes: dict[tuple[int, int], int] = defaultdict(int)
    if len(midpoint):
        cell_xy = np.floor((midpoint - center) / CELL_M).astype(int)
        for index, (ix, iy) in enumerate(cell_xy):
            key = (int(ix), int(iy))
            cells[key].append(float(distance[index]))
            if both_road[index]:
                cell_road_votes[key] += 1

    cell_records = []
    for key, values in cells.items():
        array = np.asarray(values, dtype=np.float64)
        ix, iy = key
        cell_center = center + np.array([(ix + 0.5) * CELL_M, (iy + 0.5) * CELL_M])
        on_road = bool(road.contains(cell_center.reshape(1, 2))[0])
        record = {
            "ix": ix,
            "iy": iy,
            "center": cell_center,
            "distance": float(np.linalg.norm(cell_center - center)),
            "on_road": on_road,
            "count": int(len(array)),
            "road_count": int(cell_road_votes[key]),
            "median": float(np.median(array)),
            "tail": float(np.mean(array > TAIL_M)),
        }
        record["level"] = cell_level(record["count"], record["median"], record["tail"], on_road, record["distance"])
        cell_records.append(record)

    level_counts = defaultdict(int)
    for record in cell_records:
        level_counts[record["level"]] += 1

    knowledge_path = PROJECT_ROOT / "knowledge_base" / "r39_knowledge.sqlite"
    kb_summary = {"cells": 0, "high_track_cells": 0, "by_level": defaultdict(int), "off_road": 0, "single_view": 0}
    if knowledge_path.is_file():
        connection = sqlite3.connect(knowledge_path)
        rows = connection.execute(
            "SELECT x_min, y_min, x_max, y_max, clean_track_count, confidence FROM cells"
        ).fetchall()
        connection.close()
        lookup = {(record["ix"], record["iy"]): record for record in cell_records}
        for x_min, y_min, x_max, y_max, clean_count, confidence in rows:
            kb_summary["cells"] += 1
            point = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5])
            if clean_count < 3:
                continue
            kb_summary["high_track_cells"] += 1
            if not bool(road.contains(point.reshape(1, 2))[0]):
                kb_summary["off_road"] += 1
                kb_summary["by_level"]["路面外"] += 1
                continue
            ix, iy = np.floor((point - center) / CELL_M).astype(int)
            record = lookup.get((int(ix), int(iy)))
            if record is None or record["level"] == "样本不足":
                kb_summary["single_view"] += 1
                kb_summary["by_level"]["缺少跨相机对照"] += 1
            else:
                kb_summary["by_level"][record["level"]] += 1

    prototype_note = {"count": 0, "on_road": 0, "partial": 0}
    array_path = PROJECT_ROOT / "knowledge_base" / "knowledge_arrays.npz"
    if array_path.is_file():
        stored = np.load(array_path)["prototypes"]
        prototype_note["count"] = int(len(stored))
        for path in stored:
            inside = road.contains(path)
            fraction = float(np.mean(inside))
            if fraction >= 0.9:
                prototype_note["on_road"] += 1
            elif fraction >= 0.5:
                prototype_note["partial"] += 1

    report_dir = PROJECT_ROOT / "reports"
    figure_dir = report_dir / "knowledge"
    figure_dir.mkdir(parents=True, exist_ok=True)
    view, _directions = make_r39_view(graph, 1280, 1280, 90.0)
    basemap = BEVBasemap(graph, view)
    poses = {
        short_camera_name(name): value
        for name, value in load_camera_poses().items()
    }
    font_path = config["font_path"]

    trust = basemap.render()
    colors = {"高": (60, 170, 70), "中": (40, 190, 220), "低": (50, 60, 210)}
    for record in cell_records:
        color = colors.get(record["level"])
        if color is None:
            continue
        ix, iy = record["ix"], record["iy"]
        corners = center + np.array(
            [
                [ix * CELL_M, iy * CELL_M],
                [(ix + 1) * CELL_M, iy * CELL_M],
                [(ix + 1) * CELL_M, (iy + 1) * CELL_M],
                [ix * CELL_M, (iy + 1) * CELL_M],
            ]
        )
        overlay = trust.copy()
        cv2.fillPoly(overlay, [view.to_px_int(corners)], color, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.45, trust, 0.55, 0, dst=trust)
    box = np.array([[-60, -60], [60, -60], [60, 60], [-60, 60]], dtype=np.float64)
    cv2.polylines(trust, [view.to_px_int(box)], True, (180, 180, 80), 1, cv2.LINE_AA)
    draw_camera_markers(trust, view, poses)
    basemap.draw_arm_labels(trust, directions)
    basemap.draw_scale_bar(trust)
    trust = draw_chinese(
        trust,
        [
            "可信范围  绿=高  黄=中  红=低",
            "格子 10 m  路面内且距中心不超过 82 m",
            "青框是知识库原先的原点 ±60 m 方框",
        ],
        font_path,
    )
    cv2.imwrite(str(figure_dir / "bev_trust_map.png"), trust)

    tail = basemap.render()
    if len(distance):
        tail_index = np.flatnonzero(distance > TAIL_M)
        if len(tail_index) > 8000:
            tail_index = np.random.default_rng(0).choice(tail_index, 8000, replace=False)
        for index in tail_index:
            color = PAIR_COLORS[(str(left[index]), str(right[index]))]
            point = view.to_px_int(midpoint[index])[0]
            cv2.circle(tail, tuple(point), 2, color, -1, cv2.LINE_AA)
    cv2.polylines(tail, [view.to_px_int(box)], True, (180, 180, 80), 1, cv2.LINE_AA)
    draw_camera_markers(tail, view, poses)
    basemap.draw_arm_labels(tail, directions)
    basemap.draw_scale_bar(tail)
    tail = draw_chinese(
        tail,
        [
            "跨相机误差长尾  中点距离大于 15 m",
            "颜色区分相机对  青框为原点 ±60 m",
        ],
        font_path,
    )
    cv2.imwrite(str(figure_dir / "bev_longtail_map.png"), tail)

    lines = [
        "# R39 车道图上的 BEV 有效范围",
        "",
        "车道图与流水线 `world_position` 使用同一套地面坐标 路口中心 "
        f"[{center[0]:.2f}, {center[1]:.2f}] 四路相机光轴都指向路口",
        "有效路面取车道多边形外扩 2 m 车道图臂长约 82 m 超出后单应会快速放大",
        f"长尾定义为同一全局 ID 的跨相机地面距离大于 {TAIL_M:.0f} m",
        "",
        "## 落点",
        "",
        f"- 输入 {counts['input']} 贴边 {counts['border_contact']} 无效框 {counts['invalid_bbox']} "
        f"非有限 {counts['non_finite']} 距中心超过 200 m {counts['blowup']}",
        f"- 可用投影 {counts['projected']} 其中重分配 ID {counts['reassigned']} 不进入下面的跨相机统计",
        f"- 未重分配且落在外扩路面上 {counts['on_road']} 落在车道多边形内 {counts['on_polygon']} "
        f"落在原点 ±60 m 方框内 {counts['in_origin_box']} 同时在方框和路面上 {counts['box_and_road']}",
        "",
        "| 范围 | 未重分配落点 | 其中在路面上 | 全部配对 | 两端都在路面上的配对 |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in ring_rows:
        lines.append(
            f"| {row['name']} | {row['points']} | {row['points_road']} | {fmt(row['pairs'])} | {fmt(row['pairs_road'])} |"
        )
    lines.extend(["", "## 相机对", "", "近距离指两端都在路面上且配对中点距路口中心不超过 55 m", ""])
    lines.append("| 相机对 | 全部配对 | 等级 | 路面近距离 | 等级 | 路面 55-82 m |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in pair_rows:
        lines.append(
            f"| {row['pair']} | {fmt(row['all'])} | {row['level_all']} | {fmt(row['road_near'])} | {row['level_near']} | {fmt(row['road_far'])} |"
        )
    lines.extend(["", "## 分臂", "", "只统计两端都在路面上且中点距中心不超过 82 m 的配对", ""])
    for row in arm_rows:
        lines.append(f"- {row['name']} {fmt(row['pairs_road'])}")
    lines.extend(
        [
            "",
            "## 10 m 格子",
            "",
            "高 至少 40 对 中位不超过 5 m 长尾不超过 20% 且格子在路面上距中心不超过 82 m",
            "中 至少 20 对 中位不超过 8 m 长尾不超过 35% 同样限制在这段路面内",
            "",
            f"- 有配对的格子 {len(cell_records)} 高 {level_counts['高']} 中 {level_counts['中']} "
            f"低 {level_counts['低']} 范围外 {level_counts['范围外']} 样本不足 {level_counts['样本不足']}",
            "",
            "## 知识库格子",
            "",
            f"- 知识库格子 {kb_summary['cells']} 清洁轨迹不少于 3 条的格子 {kb_summary['high_track_cells']}",
            f"- 这些格子里 高 {kb_summary['by_level']['高']} 中 {kb_summary['by_level']['中']} "
            f"低 {kb_summary['by_level']['低']} 缺少跨相机对照 {kb_summary['by_level']['缺少跨相机对照']} "
            f"路面外 {kb_summary['by_level']['路面外']}",
            f"- 典型路径 {prototype_note['count']} 条 至少 90% 采样点在路面上 {prototype_note['on_road']} 条 "
            f"一半以上在路面上 {prototype_note['partial']} 条",
            "",
            "## 读图",
            "",
            "绿色格子是可以拿来做跨相机空间知识的位置 黄色格子中位误差仍可用 但长尾更明显",
            "红色格子和彩色长尾点集中的路段 同一全局 ID 经常对不上 不应当成可靠共视",
            "青框以外的西臂和南臂外端仍在车道图上 原点方框会把它们切掉 方框以内远离路面的角落则没有车道支撑",
            "",
        ]
    )
    report_path = report_dir / "bev_trust_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    stats_path = figure_dir / "bev_trust_stats.json"
    payload = {
        "counts": dict(counts),
        "overall": overall,
        "both_on_road": road_stats,
        "not_both_on_road": off_stats,
        "pairs": [
            {
                "pair": row["pair"],
                "level_all": row["level_all"],
                "level_near": row["level_near"],
                "all": row["all"],
                "road_near": row["road_near"],
                "road_far": row["road_far"],
            }
            for row in pair_rows
        ],
        "rings": ring_rows,
        "arms": [{"name": row["name"], "pairs_road": row["pairs_road"]} for row in arm_rows],
        "cells": {key: level_counts[key] for key in ("高", "中", "低", "范围外", "样本不足")},
        "knowledge_cells": {key: kb_summary["by_level"][key] for key in kb_summary["by_level"]},
        "prototypes": prototype_note,
    }
    stats_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告 {report_path}")
    print(f"底图 {figure_dir / 'bev_trust_map.png'}")
    print(fmt(overall))
    for row in pair_rows:
        print(row["pair"], "近", row["level_near"], fmt(row["road_near"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
