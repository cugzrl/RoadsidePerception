"""配置与路径"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"配置不是对象: {path}")
    return data


def load_sources() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "sources.yaml")


def load_categories() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "category_mapping.yaml")


def load_schema() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "schema.yaml")


def dataset_root(sources: dict[str, Any], dataset_name: str) -> Path:
    dataset = sources["datasets"][dataset_name]
    link = PROJECT_ROOT / dataset["root_link"]
    if link.exists():
        return link
    return Path(dataset["source"])


def fill_template(template: str | None, relative_dir: str, camera_id: str, track_subdir: str) -> str | None:
    if not template:
        return None
    return template.format(dir=relative_dir, cam=camera_id, track_subdir=track_subdir)


def sequence_files(sources: dict[str, Any], sequence: dict[str, Any]) -> dict[str, Any]:
    """按配置展开一个采集序列的原始文件路径"""
    dataset_name = sequence["dataset"]
    dataset = sources["datasets"][dataset_name]
    root = dataset_root(sources, dataset_name)
    relative_dir = sequence["relative_dir"]
    track_subdir = sources["track_subdir"]
    cameras = sources["cameras"]

    def path_for(template: str | None, camera_id: str = "") -> Path | None:
        relative = fill_template(template, relative_dir, camera_id, track_subdir)
        if relative is None:
            return None
        return root / relative

    videos = {camera_id: path_for(dataset.get("video"), camera_id) for camera_id in cameras}
    frame_caches = {camera_id: path_for(dataset.get("frame_cache"), camera_id) for camera_id in cameras}
    files: dict[str, Any] = {
        "root": root,
        "videos": videos,
        "frame_caches": frame_caches,
        "calibration": path_for(dataset.get("calibration")),
        "pipeline_calibration_dir": (PROJECT_ROOT / sources["pipeline_calibration_dir"]).resolve(),
    }
    if dataset_name == "result":
        files.update(
            {
                "bbox_info": path_for(dataset["bbox_info"]),
                "pixel_file": path_for(dataset["pixel_file"]),
                "camera_file": path_for(dataset["camera_file"]),
                "projected_file": path_for(dataset["projected_file"]),
                "detections": path_for(dataset["detections"]) if sequence.get("include_detr") else None,
                "coco": path_for(dataset["coco"]) if sequence.get("include_detr") else None,
                "feature_metadata": path_for(dataset["feature_metadata"]) if sequence.get("include_detr") else None,
            }
        )
    else:
        files.update(
            {
                "tracks": {camera_id: path_for(dataset["tracks"], camera_id) for camera_id in cameras},
                "auto_global": path_for(dataset["auto_global"]),
                "manual_global": path_for(dataset["manual_global"]),
                "manual_corrections": path_for(dataset["manual_corrections"]),
                "lid_map": path_for(dataset["lid_map"]),
                "lid_map_manual": path_for(dataset["lid_map_manual"]),
                "split_config": path_for(dataset["split_config"]),
            }
        )
    return files


def display_path(path: Path) -> str:
    """保留软链接相对路径，避免解析后离开项目目录"""
    absolute = path if path.is_absolute() else PROJECT_ROOT / path
    try:
        return str(absolute.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(absolute)
