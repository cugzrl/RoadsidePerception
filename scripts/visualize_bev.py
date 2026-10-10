#!/usr/bin/env python3
"""在流水线结果末尾生成 R39 全局 BEV

读取 result.json 里的 world_position 叠到车道级底图上
写出整段轨迹静图 bev_map.png 和逐帧尾迹视频 bev.mp4
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.bev_map import (  # noqa: E402
    BEVBasemap,
    BEVView,
    LaneMap,
    color_for_id,
    draw_camera_markers,
    draw_trails,
    load_camera_poses,
    make_r39_view,
    short_camera_name,
)
from pipeline.common import load_json  # noqa: E402

R39_CAMERAS = ("R39_Aw_CamS", "R39_Bn_CamW", "R39_Ce_CamN", "R39_Ds_CamE")
# 名字里的方位是相机看向的方向 画面按安装侧摆放
SIDE_OF_CAMERA = {
    "R39_Aw_CamS": "top",
    "R39_Ds_CamE": "left",
    "R39_Bn_CamW": "right",
    "R39_Ce_CamN": "bottom",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把跟踪结果画成 R39 全局 BEV")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="逐帧 BEV 视频")
    parser.add_argument("--still", type=Path, help="整段轨迹静图 默认与视频同目录的 bev_map.png")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--still-size", type=int, default=1280)
    parser.add_argument("--radius-m", type=float, default=90.0)
    parser.add_argument("--camera-width", type=int, default=800)
    parser.add_argument("--camera-height", type=int, default=450)
    parser.add_argument("--trail-seconds", type=float, default=2.0)
    parser.add_argument("--min-path-m", type=float, default=8.0)
    parser.add_argument("--max-step-m", type=float, default=15.0, help="相邻点超过该距离就断开，避免远处单应把轨迹拉成直线")
    return parser.parse_args()


def as_xy(value) -> np.ndarray | None:
    if isinstance(value, dict) and "x" in value and "y" in value:
        point = np.array([float(value["x"]), float(value["y"])], dtype=np.float64)
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        point = np.array([float(value[0]), float(value[1])], dtype=np.float64)
    else:
        return None
    if not np.isfinite(point).all():
        return None
    return point


def collect(result: dict) -> tuple[dict[int, list[tuple[int, np.ndarray]]], dict[int, dict[int, np.ndarray]], list[int]]:
    """按全局 ID 收集轨迹 同一帧多相机取均值作为该帧位置"""
    grouped: dict[int, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for frame in result.get("frames", []):
        frame_index = int(frame["frame_index"])
        for detection in frame.get("detections", []):
            track_id = detection.get("global_track_id")
            point = as_xy(detection.get("world_position"))
            if track_id is None or point is None:
                continue
            grouped[int(track_id)][frame_index].append(point)
    trails: dict[int, list[tuple[int, np.ndarray]]] = {}
    by_frame: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    frames: set[int] = set()
    for track_id, frame_points in grouped.items():
        ordered = []
        for frame_index in sorted(frame_points):
            point = np.mean(frame_points[frame_index], axis=0)
            ordered.append((frame_index, point))
            by_frame[frame_index][track_id] = point
            frames.add(frame_index)
        trails[track_id] = ordered
    return trails, by_frame, sorted(frames)


def limit_range(trails, by_frame, center: np.ndarray, radius_m: float):
    """丢掉视野外的点 那些点多半是地平线附近的单应放大"""
    kept_trails = {}
    kept_frames: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    seen: set[int] = set()
    for track_id, samples in trails.items():
        kept = []
        for frame_index, point in samples:
            if float(np.linalg.norm(point - center)) > radius_m:
                continue
            kept.append((frame_index, point))
            kept_frames[frame_index][track_id] = point
            seen.add(frame_index)
        if kept:
            kept_trails[track_id] = kept
    if not seen:
        return kept_trails, kept_frames, []
    return kept_trails, kept_frames, list(range(min(seen), max(seen) + 1))


def decorate(image, basemap, directions, poses, title: str) -> None:
    markers = {
        short_camera_name(name): (position, look)
        for name, (position, look) in poses.items()
    }
    draw_camera_markers(image, basemap.view, markers)
    basemap.draw_arm_labels(image, directions)
    basemap.draw_scale_bar(image)
    cv2.putText(image, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)


def plain_view(points: list[np.ndarray], width: int, height: int) -> BEVView:
    if points:
        stacked = np.stack(points, axis=0)
        center = np.median(stacked, axis=0)
        radius = max(20.0, float(np.percentile(np.linalg.norm(stacked - center, axis=1), 95)) * 1.3)
    else:
        center = np.zeros(2)
        radius = 40.0
    return BEVView(center=center, width=width, height=height, px_per_m=min(width, height) / (2 * radius))


def render_plain(result, trails, by_frame, frames, view: BEVView, still_view: BEVView, output: Path, still: Path, trail_seconds: float, min_path_m: float) -> None:
    def paint(width_view: BEVView, fade: bool, current_only: dict[int, np.ndarray] | None) -> np.ndarray:
        image = np.full((width_view.height, width_view.width, 3), (22, 22, 24), dtype=np.uint8)
        drawn, colors, current, labels = {}, {}, {}, {}
        if current_only is None:
            for track_id, samples in trails.items():
                xy = np.stack([point for _, point in samples], axis=0)
                if len(xy) >= 2 and float(np.linalg.norm(xy[-1] - xy[0])) < min_path_m:
                    continue
                drawn[track_id] = xy
                colors[track_id] = color_for_id(track_id)
                current[track_id] = xy[-1]
                labels[track_id] = str(track_id)
        else:
            current = current_only
            colors = {track_id: color_for_id(track_id) for track_id in current_only}
        if drawn or current:
            draw_trails(image, width_view, drawn, colors, current, labels if current_only is None else None, fade=fade)
        cv2.putText(image, "BEV tracks", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
        return image

    still.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(still), paint(still_view, False, None))
    fps = float(result.get("fps") or 20.0)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (view.width, view.height))
    if not frames:
        writer.write(paint(view, False, None))
    window = max(1, int(round(trail_seconds * fps)))
    cursors = {track_id: 0 for track_id in trails}
    for frame_index in frames:
        image = np.full((view.height, view.width, 3), (22, 22, 24), dtype=np.uint8)
        active = by_frame.get(frame_index, {})
        trail_xy, colors = {}, {}
        for track_id, samples in trails.items():
            cursor = cursors[track_id]
            while cursor < len(samples) and samples[cursor][0] <= frame_index:
                cursor += 1
            cursors[track_id] = cursor
            window_points = [point for _index, point in samples[max(0, cursor - window) : cursor]]
            if len(window_points) >= 2:
                trail_xy[track_id] = np.stack(window_points, axis=0)
            if track_id in active or len(window_points) >= 2:
                colors[track_id] = color_for_id(track_id)
        if trail_xy or active:
            draw_trails(image, view, trail_xy, colors, active, fade=True)
        cv2.putText(image, "BEV tracks", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
        writer.write(image)
    writer.release()


def render_still(result: dict, trails, basemap, directions, poses, path: Path, min_path_m: float, max_step_m: float) -> int:
    image = basemap.render()
    drawn = {}
    colors = {}
    current = {}
    labels = {}
    for track_id, samples in trails.items():
        xy = np.stack([point for _, point in samples], axis=0)
        if len(xy) >= 2 and float(np.linalg.norm(xy[-1] - xy[0])) < min_path_m:
            continue
        drawn[track_id] = xy
        colors[track_id] = color_for_id(track_id)
        current[track_id] = xy[-1]
        labels[track_id] = str(track_id)
    if drawn:
        draw_trails(image, basemap.view, drawn, colors, current, labels, fade=False, thickness=2, max_step_m=max_step_m)
    scene = result.get("scene_id") or "R39"
    decorate(image, basemap, directions, poses, f"{scene}  lane-level BEV")
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return len(drawn)


def frame_lookup(result: dict) -> tuple[dict[tuple[str, int], str], dict[tuple[str, int], list]]:
    images: dict[tuple[str, int], str] = {}
    detections: dict[tuple[str, int], list] = defaultdict(list)
    for frame in result.get("frames", []):
        key = (frame["camera_id"], int(frame["frame_index"]))
        path = frame.get("image_path")
        if path:
            images[key] = path
        detections[key] = frame.get("detections") or []
    return images, detections


def camera_panel(path: str | None, detections: list, width: int, height: int) -> np.ndarray:
    image = cv2.imread(path) if path else None
    if image is None:
        return np.full((height, width, 3), (22, 22, 24), dtype=np.uint8)
    source_height, source_width = image.shape[:2]
    panel = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    scale_x = width / source_width
    scale_y = height / source_height
    for detection in detections:
        box = detection.get("bbox_xyxy")
        track_id = detection.get("global_track_id")
        if track_id is None or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        color = color_for_id(int(track_id))
        x1 = int(round(box[0] * scale_x))
        y1 = int(round(box[1] * scale_y))
        x2 = int(round(box[2] * scale_x))
        y2 = int(round(box[3] * scale_y))
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = str(int(track_id))
        origin = (x1, max(16, y1 - 4))
        cv2.putText(panel, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 16, 16), 3, cv2.LINE_AA)
        cv2.putText(panel, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return panel


def place_cross(bev: np.ndarray, panels: dict[str, np.ndarray], camera_width: int, camera_height: int) -> np.ndarray:
    bev_height, bev_width = bev.shape[:2]
    canvas = np.full(
        (camera_height + bev_height + camera_height, camera_width + bev_width + camera_width, 3),
        (22, 22, 24),
        dtype=np.uint8,
    )
    canvas[camera_height : camera_height + bev_height, camera_width : camera_width + bev_width] = bev
    top_x = camera_width + max(0, (bev_width - camera_width) // 2)
    side_y = camera_height + max(0, (bev_height - camera_height) // 2)
    slots = {
        "top": (0, top_x),
        "left": (side_y, 0),
        "right": (side_y, camera_width + bev_width),
        "bottom": (camera_height + bev_height, top_x),
    }
    for side, (y_value, x_value) in slots.items():
        panel = panels.get(side)
        if panel is None:
            continue
        canvas[y_value : y_value + camera_height, x_value : x_value + camera_width] = panel
    return canvas


def render_video(
    result,
    trails,
    by_frame,
    frames,
    basemap,
    path: Path,
    trail_seconds: float,
    max_step_m: float,
    camera_width: int,
    camera_height: int,
) -> None:
    fps = float(result.get("fps") or 20.0)
    window = max(1, int(round(trail_seconds * fps)))
    images, detections = frame_lookup(result)
    sample = basemap.render()
    canvas_size = place_cross(
        sample,
        {side: np.zeros((camera_height, camera_width, 3), dtype=np.uint8) for side in ("top", "left", "right", "bottom")},
        camera_width,
        camera_height,
    ).shape[1::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, canvas_size)
    if not writer.isOpened():
        raise RuntimeError(f"无法写出 BEV 视频 {path}")
    cursors = {track_id: 0 for track_id in trails}
    for frame_index in frames:
        image = basemap.render()
        active = by_frame.get(frame_index, {})
        trail_xy = {}
        colors = {}
        for track_id, samples in trails.items():
            cursor = cursors[track_id]
            while cursor < len(samples) and samples[cursor][0] <= frame_index:
                cursor += 1
            cursors[track_id] = cursor
            window_points = [point for index, point in samples[max(0, cursor - window) : cursor]]
            if len(window_points) >= 2:
                trail_xy[track_id] = np.stack(window_points, axis=0)
                colors[track_id] = color_for_id(track_id)
            elif track_id in active:
                colors[track_id] = color_for_id(track_id)
        current = {track_id: point for track_id, point in active.items()}
        labels = {track_id: str(track_id) for track_id in current}
        if trail_xy or current:
            draw_trails(
                image,
                basemap.view,
                trail_xy,
                colors,
                current,
                labels,
                fade=True,
                thickness=2,
                dot_radius=4,
                label_scale=0.55,
                max_step_m=max_step_m,
            )
        panels = {}
        for camera_id, side in SIDE_OF_CAMERA.items():
            key = (camera_id, frame_index)
            panels[side] = camera_panel(images.get(key), detections.get(key, []), camera_width, camera_height)
        writer.write(place_cross(image, panels, camera_width, camera_height))
        if frame_index % 200 == 0:
            print(f"已写 BEV 帧 {frame_index}", flush=True)
    writer.release()


def main() -> int:
    args = parse_args()
    result = load_json(args.result)
    camera_ids = [item.get("camera_id", "") for item in result.get("cameras", [])]
    trails, by_frame, frames = collect(result)
    still_path = args.still or (args.output.parent / "bev_map.png")
    use_lane_map = any(camera_id in R39_CAMERAS or str(camera_id).startswith("R39_") for camera_id in camera_ids)
    if not use_lane_map:
        points = [point for samples in trails.values() for _frame, point in samples]
        view = plain_view(points, args.width, args.height)
        still_view = plain_view(points, args.still_size, args.still_size)
        render_plain(result, trails, by_frame, frames, view, still_view, args.output, still_path, args.trail_seconds, args.min_path_m)
        print(f"场景没有 R39 相机 已写出轨迹 BEV {still_path} 与 {args.output}")
        return 0
    graph = LaneMap.load()
    trails, by_frame, frames = limit_range(trails, by_frame, graph.center, args.radius_m)
    view, directions = make_r39_view(graph, args.width, args.height, args.radius_m)
    basemap = BEVBasemap(graph, view)
    basemap.render()
    poses = load_camera_poses()
    still_view, still_directions = make_r39_view(graph, args.still_size, args.still_size, args.radius_m)
    still_map = BEVBasemap(graph, still_view)
    drawn = render_still(result, trails, still_map, still_directions, poses, still_path, args.min_path_m, args.max_step_m)
    if frames:
        render_video(
            result,
            trails,
            by_frame,
            frames,
            basemap,
            args.output,
            args.trail_seconds,
            args.max_step_m,
            args.camera_width,
            args.camera_height,
        )
    else:
        image = basemap.render()
        decorate(image, basemap, directions, poses, f"{result.get('scene_id') or 'R39'}  BEV")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(result.get("fps") or 20.0),
            (view.width, view.height),
        )
        writer.write(image)
        writer.release()
    print(f"BEV 静图 {still_path} 轨迹 {drawn} 条 视频 {args.output} 帧 {len(frames)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
