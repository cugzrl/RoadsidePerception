"""R39 车道级 BEV 底图

底图来自路口车道图 路面多边形 分隔线 停止线和转向中心线
坐标与 STMC 单应反投影的地面 XY 相同 轨迹可以直接叠上去
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH = ROOT / "configs/maps/r39_lane_graph.json"
DEFAULT_CAMERAS = ROOT / "configs/maps/r39_cameras.json"

TURN_COLORS = {
    "straight": (90, 200, 235),
    "left": (235, 130, 235),
    "right": (120, 235, 160),
    "straight_right": (255, 160, 40),
    "u_turn": (110, 130, 255),
    None: (170, 170, 170),
}
BIKE_COLOR = (40, 170, 255)
ROAD_COLOR = (64, 64, 70)
DIVIDER_COLOR = (168, 168, 175)
STOPLINE_COLOR = (235, 235, 240)
BG_COLOR = (22, 22, 24)


class LaneMap:
    """只保留画底图需要的车道节点 不依赖 networkx 或 shapely"""

    def __init__(self, payload: dict):
        self.road_id = payload.get("road_id", "")
        self.frame = payload.get("frame", "")
        self.center = np.asarray(payload["center_world"], dtype=np.float64)
        self.nodes = {node["lane_id"]: node for node in payload["lane_nodes"]}

    @classmethod
    def load(cls, path: Path | str = DEFAULT_GRAPH) -> "LaneMap":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def heading_at_end(self, lane_id: str) -> float:
        centerline = np.asarray(self.nodes[lane_id]["centerline"], dtype=np.float64)
        step = centerline[-1] - centerline[-2]
        return float(np.degrees(np.arctan2(step[1], step[0])))


def arm_directions(graph: LaneMap) -> dict[str, np.ndarray]:
    """每条臂自路口中心向外的单位方向 进口道朝路口 所以取中心线反向"""
    grouped: dict[str, list[np.ndarray]] = {}
    for node in graph.nodes.values():
        if node["type"] != "vehicle_lane" or node.get("role") != "entry":
            continue
        arm = node.get("arm")
        centerline = np.asarray(node["centerline"], dtype=np.float64)
        direction = centerline[0] - centerline[-1]
        length = float(np.linalg.norm(direction))
        if arm and length > 1e-6:
            grouped.setdefault(arm, []).append(direction / length)
    return {
        arm: np.mean(vectors, axis=0) / np.linalg.norm(np.mean(vectors, axis=0))
        for arm, vectors in grouped.items()
    }


@dataclass
class BEVView:
    """world XY 到画布像素 像素 y 向下 world y 向上

    up_deg 是朝画布正上方的 world 方向角 默认 90 度即 +Y 朝上
    给出 ortho_east 和 ortho_north 时做一次展示用的正交化 不改车道坐标
    """

    center: np.ndarray
    width: int
    height: int
    px_per_m: float
    up_deg: float = 90.0
    ortho_east: Optional[np.ndarray] = None
    ortho_north: Optional[np.ndarray] = None

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64).reshape(2)
        if self.ortho_east is not None and self.ortho_north is not None:
            east = np.asarray(self.ortho_east, dtype=float).reshape(2)
            north = np.asarray(self.ortho_north, dtype=float).reshape(2)
            east = east / max(float(np.linalg.norm(east)), 1e-9)
            north = north / max(float(np.linalg.norm(north)), 1e-9)
            basis = np.column_stack([east, north])
            self._inverse = np.linalg.inv(basis)
            self._rotation = None
            self._mode = "ortho"
        else:
            angle = np.radians(self.up_deg - 90.0)
            cosine, sine = np.cos(angle), np.sin(angle)
            self._rotation = np.array([[cosine, sine], [-sine, cosine]])
            self._inverse = None
            self._mode = "rot"

    def to_px(self, xy) -> np.ndarray:
        points = np.atleast_2d(np.asarray(xy, dtype=np.float64)) - self.center
        if self._mode == "ortho":
            aligned = (self._inverse @ points.T).T
            column = self.width * 0.5 + aligned[:, 0] * self.px_per_m
            row = self.height * 0.5 - aligned[:, 1] * self.px_per_m
        else:
            rotated = points @ self._rotation.T
            column = self.width * 0.5 + rotated[:, 0] * self.px_per_m
            row = self.height * 0.5 - rotated[:, 1] * self.px_per_m
        return np.stack([column, row], axis=-1)

    def to_px_int(self, xy) -> np.ndarray:
        return np.round(self.to_px(xy)).astype(np.int32)


def make_r39_view(
    graph: LaneMap,
    width: int = 1280,
    height: int = 1280,
    radius_m: float = 90.0,
    up_arm: str = "N",
) -> tuple[BEVView, dict[str, np.ndarray]]:
    directions = arm_directions(graph)
    up = directions.get(up_arm)
    up_deg = 90.0 if up is None else float(np.degrees(np.arctan2(up[1], up[0])))
    order = ["N", "E", "S", "W"]
    north = directions.get(up_arm)
    east = None
    if up_arm in order:
        east = directions.get(order[(order.index(up_arm) + 1) % 4])
    view = BEVView(
        center=graph.center,
        width=width,
        height=height,
        px_per_m=min(width, height) / (2 * radius_m),
        up_deg=up_deg,
        ortho_east=east,
        ortho_north=north,
    )
    return view, directions


def _fill(image: np.ndarray, points_px: np.ndarray, color, alpha: float = 1.0) -> None:
    if len(points_px) < 3:
        return
    if alpha >= 1.0:
        cv2.fillPoly(image, [points_px], color, lineType=cv2.LINE_AA)
        return
    overlay = image.copy()
    cv2.fillPoly(overlay, [points_px], color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)


def _dashed(image: np.ndarray, points_px: np.ndarray, color, dash: int = 9, gap: int = 7, thickness: int = 1) -> None:
    for start, end in zip(points_px[:-1], points_px[1:]):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            continue
        direction = delta / length
        cursor = 0.0
        while cursor < length:
            first = start + direction * cursor
            second = start + direction * min(cursor + dash, length)
            cv2.line(
                image,
                tuple(np.round(first).astype(int)),
                tuple(np.round(second).astype(int)),
                color,
                thickness,
                cv2.LINE_AA,
            )
            cursor += dash + gap


class BEVBasemap:
    """车道图底图 渲染后缓存 每帧复制使用"""

    def __init__(self, graph: LaneMap, view: BEVView, show_lane_ids: bool = False):
        self.graph = graph
        self.view = view
        self.show_lane_ids = show_lane_ids
        self._base: Optional[np.ndarray] = None

    def render(self) -> np.ndarray:
        if self._base is not None:
            return self._base.copy()
        view = self.view
        image = np.full((view.height, view.width, 3), BG_COLOR, dtype=np.uint8)
        for node in self.graph.nodes.values():
            if node["type"] not in ("vehicle_lane", "connector", "bike_lane"):
                continue
            _fill(image, view.to_px_int(node["polygon"]), ROAD_COLOR)

        divider = max(1, int(round(view.px_per_m / 8.0)))
        for node in self.graph.nodes.values():
            if node["type"] != "vehicle_lane":
                continue
            polygon = np.asarray(node["polygon"], dtype=float)
            half = len(polygon) // 2
            _dashed(image, view.to_px(polygon[:half]), DIVIDER_COLOR, thickness=divider)

        connector = max(1, int(round(view.px_per_m / 9.0)))
        for node in self.graph.nodes.values():
            if node["type"] != "connector":
                continue
            _dashed(
                image,
                view.to_px(np.asarray(node["centerline"], dtype=float)),
                TURN_COLORS[None],
                dash=6,
                gap=5,
                thickness=connector,
            )

        line = max(2, int(round(view.px_per_m / 3.0)))
        for node in self.graph.nodes.values():
            if node["type"] != "vehicle_lane":
                continue
            if node.get("role") == "entry":
                color = TURN_COLORS.get(node.get("turn_type"), TURN_COLORS[None])
            else:
                color = TURN_COLORS[None]
            centerline = np.asarray(node["centerline"], dtype=float)
            cv2.polylines(image, [np.round(view.to_px(centerline)).astype(np.int32)], False, color, line, cv2.LINE_AA)
            if len(centerline) >= 2:
                steps = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
                distance = np.concatenate([[0.0], np.cumsum(steps)])
                index = int(np.searchsorted(distance, max(distance[-1] - 5.0, 0.0)))
                index = min(index, len(centerline) - 2)
                start, end = view.to_px_int([centerline[index], centerline[-1]])
                if np.linalg.norm(end - start) >= 4:
                    cv2.arrowedLine(image, tuple(start), tuple(end), color, 2, cv2.LINE_AA, tipLength=0.35)

        bike = max(2, int(round(view.px_per_m / 3.5)))
        for node in self.graph.nodes.values():
            if node["type"] != "bike_lane":
                continue
            centerline = np.asarray(node["centerline"], dtype=float)
            cv2.polylines(image, [np.round(view.to_px(centerline)).astype(np.int32)], False, BIKE_COLOR, bike, cv2.LINE_AA)

        for lane_id, node in self.graph.nodes.items():
            if node["type"] != "vehicle_lane" or node.get("role") != "entry":
                continue
            centerline = np.asarray(node["centerline"], dtype=float)
            end = centerline[-1]
            heading = np.radians(self.graph.heading_at_end(lane_id))
            normal = np.array([-np.sin(heading), np.cos(heading)])
            half_width = 0.5 * float(node.get("width", 3.2))
            start_px, end_px = view.to_px_int([end - normal * half_width, end + normal * half_width])
            cv2.line(image, tuple(start_px), tuple(end_px), STOPLINE_COLOR, 2, cv2.LINE_AA)

        if self.show_lane_ids:
            for lane_id, node in self.graph.nodes.items():
                if node["type"] != "vehicle_lane":
                    continue
                centerline = np.asarray(node["centerline"], dtype=float)
                point = view.to_px_int(centerline[len(centerline) // 2])[0]
                cv2.putText(
                    image,
                    lane_id.replace("R39_", ""),
                    tuple(point),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.28,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
        self._base = image
        return image.copy()

    def road_half_width_m(self, arm: str) -> float:
        points = []
        axis = None
        for node in self.graph.nodes.values():
            if node["type"] != "vehicle_lane" or node.get("arm") != arm:
                continue
            centerline = np.asarray(node["centerline"], dtype=float)
            if axis is None:
                direction = centerline[-1] - centerline[0]
                axis = direction / max(np.linalg.norm(direction), 1e-9)
            points.append(np.asarray(node["polygon"], dtype=float))
        if axis is None or not points:
            return 12.0
        normal = np.array([-axis[1], axis[0]])
        stacked = np.vstack(points) - self.view.center
        return float(np.abs(stacked @ normal).max())

    def draw_scale_bar(self, image: np.ndarray, scale: float = 1.0) -> np.ndarray:
        view = self.view
        length = int(round(20.0 * view.px_per_m))
        x0, y0 = int(20 * scale), view.height - int(24 * scale)
        thickness = max(1, int(round(2 * scale)))
        cv2.line(image, (x0, y0), (x0 + length, y0), (230, 230, 230), thickness, cv2.LINE_AA)
        for x_value in (x0, x0 + length):
            cv2.line(
                image,
                (x_value, y0 - int(5 * scale)),
                (x_value, y0 + int(5 * scale)),
                (230, 230, 230),
                thickness,
                cv2.LINE_AA,
            )
        cv2.putText(
            image,
            "20 m",
            (x0 + length + int(10 * scale), y0 + int(6 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * scale,
            (230, 230, 230),
            thickness,
            cv2.LINE_AA,
        )
        return image

    def draw_arm_labels(
        self,
        image: np.ndarray,
        directions: Dict[str, np.ndarray],
        scale: float = 1.0,
        side_gap_m: float = 7.0,
        edge_inset_px: float = 22.0,
    ) -> np.ndarray:
        view = self.view
        font_scale = 1.05 * scale
        thickness = max(1, int(round(2.5 * scale)))
        inset = edge_inset_px * scale
        canvas_center = np.array([view.width * 0.5, view.height * 0.5])
        for name, direction in directions.items():
            direction = np.asarray(direction, dtype=float)
            direction = direction / max(np.linalg.norm(direction), 1e-9)
            far = view.to_px(view.center + direction * 1000.0)[0]
            screen = far - canvas_center
            norm = float(np.linalg.norm(screen))
            if norm < 1e-6:
                continue
            screen /= norm
            offset = (self.road_half_width_m(name) + side_gap_m) * view.px_per_m
            (text_width, text_height), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            if abs(screen[0]) >= abs(screen[1]):
                x_value = view.width - inset - text_width if screen[0] >= 0 else inset
                y_value = canvas_center[1] - offset if name == "W" else canvas_center[1] + offset
            else:
                y_value = view.height - inset if screen[1] >= 0 else inset + text_height
                x_value = canvas_center[0] + offset if name == "N" else canvas_center[0] - offset - text_width
            x_value = float(np.clip(x_value, inset, view.width - text_width - inset))
            y_value = float(np.clip(y_value, text_height + inset * 0.5, view.height - inset * 0.5))
            origin = (int(round(x_value)), int(round(y_value)))
            cv2.putText(image, name, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(image, name, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (130, 230, 150), thickness, cv2.LINE_AA)
        return image


def assign_corners(view: BEVView, positions: Dict[str, np.ndarray]) -> Dict[str, int]:
    import itertools

    corner_angles = {0: 135.0, 1: 45.0, 2: 225.0, 3: 315.0}
    canvas_center = np.array([view.width * 0.5, view.height * 0.5])
    names = list(positions)
    angles = {}
    for name, xy in positions.items():
        delta = view.to_px(xy)[0] - canvas_center
        angles[name] = float(np.degrees(np.arctan2(-delta[1], delta[0]))) % 360.0

    def difference(left: float, right: float) -> float:
        return abs((left - right + 180.0) % 360.0 - 180.0)

    best, best_cost = None, np.inf
    for permutation in itertools.permutations(range(4)):
        cost = sum(difference(angles[name], corner_angles[corner]) for name, corner in zip(names, permutation))
        if cost < best_cost:
            best, best_cost = permutation, cost
    return {name: corner for name, corner in zip(names, best)}


def draw_camera_markers(
    image: np.ndarray,
    view: BEVView,
    cameras: Dict[str, Tuple[np.ndarray, np.ndarray]],
    scale: float = 1.0,
    color: Tuple[int, int, int] = (110, 235, 255),
    top_margin_px: int = 0,
) -> np.ndarray:
    """相机三角朝向光轴 名字放在画面四角

    名字里的 CamS 是看向的方向 安装位置在路口另一侧
    """
    corners = assign_corners(view, {name: value[0] for name, value in cameras.items()})
    font_scale = 0.55 * scale
    thickness = max(1, int(round(2 * scale)))
    pad = int(22 * scale)
    line = max(1, int(round(1.6 * scale)))
    for name, (position, look) in cameras.items():
        point = view.to_px(position)[0]
        ahead = view.to_px(position + look * 8.0)[0]
        direction = ahead - point
        norm = float(np.linalg.norm(direction))
        direction = np.array([0.0, -1.0]) if norm < 1e-6 else direction / norm
        normal = np.array([-direction[1], direction[0]])
        size = 9.0 * scale
        triangle = np.round(
            np.array(
                [
                    point + direction * size * 1.6,
                    point - direction * size * 0.5 + normal * size,
                    point - direction * size * 0.5 - normal * size,
                ]
            )
        ).astype(np.int32)
        cv2.fillPoly(image, [triangle], color, cv2.LINE_AA)
        cv2.circle(image, tuple(np.round(point).astype(int)), int(3 * scale), (20, 20, 20), -1, cv2.LINE_AA)
        (text_width, text_height), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        corner = corners[name]
        x_value = pad if corner in (0, 2) else view.width - text_width - pad
        y_value = (top_margin_px + text_height + pad) if corner in (0, 1) else view.height - pad
        anchor = np.array([x_value + text_width * 0.5, y_value - text_height * 0.5], dtype=float)
        tip_direction = anchor - point
        tip_length = float(np.linalg.norm(tip_direction))
        if tip_length > 1e-3:
            tip = anchor - tip_direction / tip_length * max(12.0 * scale, text_height * 0.4)
            cv2.arrowedLine(
                image,
                tuple(np.round(point).astype(int)),
                tuple(np.round(tip).astype(int)),
                color,
                line,
                cv2.LINE_AA,
                tipLength=0.14,
            )
        cv2.putText(image, name, (int(x_value), int(y_value)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(image, name, (int(x_value), int(y_value)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return image


def draw_trails(
    image: np.ndarray,
    view: BEVView,
    trails: Dict[int, np.ndarray],
    colors: Dict[int, Tuple[int, int, int]],
    current: Dict[int, np.ndarray] | None = None,
    labels: Dict[int, str] | None = None,
    thickness: int = 2,
    dot_radius: int = 4,
    fade: bool = True,
    label_scale: float = 0.4,
    max_step_m: float | None = None,
) -> np.ndarray:
    for track_id, xy in trails.items():
        if len(xy) < 2:
            continue
        points = view.to_px(xy)
        steps = np.linalg.norm(np.diff(np.asarray(xy, dtype=np.float64), axis=0), axis=1)
        color = colors.get(track_id, (200, 200, 200))
        if fade:
            count = len(points)
            for index in range(1, count):
                if max_step_m is not None and steps[index - 1] > max_step_m:
                    continue
                weight = 0.25 + 0.75 * (index / count)
                faded = tuple(int(channel * weight) for channel in color)
                cv2.line(
                    image,
                    tuple(np.round(points[index - 1]).astype(int)),
                    tuple(np.round(points[index]).astype(int)),
                    faded,
                    thickness,
                    cv2.LINE_AA,
                )
        else:
            run = [points[0]]
            for index in range(1, len(points)):
                if max_step_m is not None and steps[index - 1] > max_step_m:
                    if len(run) >= 2:
                        cv2.polylines(image, [np.round(np.stack(run)).astype(np.int32)], False, color, thickness, cv2.LINE_AA)
                    run = [points[index]]
                else:
                    run.append(points[index])
            if len(run) >= 2:
                cv2.polylines(image, [np.round(np.stack(run)).astype(np.int32)], False, color, thickness, cv2.LINE_AA)
    if current:
        for track_id, xy in current.items():
            point = view.to_px_int(np.asarray(xy, dtype=np.float64).reshape(1, 2))[0]
            color = colors.get(track_id, (200, 200, 200))
            cv2.circle(image, tuple(point), dot_radius + 2, (16, 16, 16), -1, cv2.LINE_AA)
            cv2.circle(image, tuple(point), dot_radius, color, -1, cv2.LINE_AA)
            if labels and track_id in labels:
                offset = (int(point[0] + dot_radius + 4), int(point[1] - dot_radius - 2))
                stroke = max(1, int(round(label_scale / 0.4)))
                cv2.putText(image, labels[track_id], offset, cv2.FONT_HERSHEY_SIMPLEX, label_scale, (20, 20, 20), stroke + 2, cv2.LINE_AA)
                cv2.putText(image, labels[track_id], offset, cv2.FONT_HERSHEY_SIMPLEX, label_scale, color, stroke, cv2.LINE_AA)
    return image


def color_for_id(track_id: int) -> tuple[int, int, int]:
    import colorsys

    hue = (int(track_id) * 0.6180339887498949) % 1.0
    if 0.60 < hue < 0.72:
        hue = 0.60 + (hue - 0.60) * 0.4
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.82, 1.0)
    return (int(blue * 255), int(green * 255), int(red * 255))


def load_camera_poses(path: Path | str = DEFAULT_CAMERAS) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    poses = {}
    for name, item in payload.items():
        position = np.asarray(item["position"], dtype=np.float64)
        look = np.asarray(item["look"], dtype=np.float64)
        look = look / max(float(np.linalg.norm(look)), 1e-9)
        poses[name] = (position, look)
    return poses


def short_camera_name(camera_id: str) -> str:
    return camera_id.replace("R39_", "")


@dataclass
class RoadMask:
    """车道多边形栅格 用于判断地面点是否落在图上的路面内"""

    mask: np.ndarray
    origin_x: float
    origin_y: float
    resolution: float

    def contains(self, xy) -> np.ndarray:
        points = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        column = np.floor((points[:, 0] - self.origin_x) / self.resolution).astype(np.int32)
        row = np.floor((points[:, 1] - self.origin_y) / self.resolution).astype(np.int32)
        valid = (row >= 0) & (row < self.mask.shape[0]) & (column >= 0) & (column < self.mask.shape[1])
        inside = np.zeros(len(points), dtype=bool)
        if np.any(valid):
            inside[valid] = self.mask[row[valid], column[valid]] > 0
        return inside


def build_road_mask(graph: LaneMap, resolution: float = 1.0, dilate_m: float = 0.0) -> RoadMask:
    polygons = []
    for node in graph.nodes.values():
        if node["type"] not in ("vehicle_lane", "connector", "bike_lane"):
            continue
        polygons.append(np.asarray(node["polygon"], dtype=np.float64))
    stacked = np.vstack(polygons)
    origin_x = float(np.floor(stacked[:, 0].min() - 4.0))
    origin_y = float(np.floor(stacked[:, 1].min() - 4.0))
    width = int(np.ceil((stacked[:, 0].max() + 4.0 - origin_x) / resolution)) + 1
    height = int(np.ceil((stacked[:, 1].max() + 4.0 - origin_y) / resolution)) + 1
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        pixels = np.stack(
            [
                np.round((polygon[:, 0] - origin_x) / resolution),
                np.round((polygon[:, 1] - origin_y) / resolution),
            ],
            axis=1,
        ).astype(np.int32)
        cv2.fillPoly(mask, [pixels], 1)
    if dilate_m > 0:
        radius = max(1, int(round(dilate_m / resolution)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        mask = cv2.dilate(mask, kernel)
    return RoadMask(mask=mask, origin_x=origin_x, origin_y=origin_y, resolution=resolution)
