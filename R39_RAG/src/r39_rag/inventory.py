"""扫描原始目录并生成盘点结果"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from r39_rag.adapters.readers import iter_track_observations
from r39_rag.config import PROJECT_ROOT, dataset_root, load_sources, sequence_files
from r39_rag.utils.json_stream import iter_json_array, load_json
from r39_rag.utils.video import probe_video

JSON_KINDS = {
    "aligned_pixel_bbox.json": "逐帧像素框",
    "aligned_camera_bbox.json": "相机三维框",
    "aligned_bbox_info.json": "像素框与相机框合并文件",
    "projected_pixel_bbox.json": "三维框投影回像素",
    "detections.json": "DETR 逐帧检测",
    "detections_coco.json": "DETR 的 COCO 导出",
    "feature_metadata.json": "ResNet50 特征索引",
    "failures.json": "检测失败记录",
    "tracks.json": "单相机 SAM3 轨迹",
    "object_labels.json": "局部 ID 类别标签",
    "multi_view_global_ids.json": "自动全局关联",
    "multi_view_global_ids_manual.json": "人工修正全局关联",
    "mask_manifest.json": "实例掩码清单",
    "frame_cache_meta.json": "抽帧元数据",
    "calib.json": "历史相机标定",
}


def _read_head(path: Path, size: int = 500) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as file:
        return file.read(size)


def classify_json(path: Path) -> str:
    if path.name in JSON_KINDS:
        return JSON_KINDS[path.name]
    head = _read_head(path)
    if '"scene_id"' in head and '"frames"' in head:
        return "流水线 result.json"
    return "未分类 JSON"


def _count_classes_in_array(path: Path, array_key: str | None) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in iter_json_array(path, array_key):
        counts[str(record.get("class_name"))] += 1
    return counts


def scan_files(root: Path) -> dict[str, Any]:
    rows = []
    extension_counts: Counter[str] = Counter()
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            suffix = path.suffix.lower()
            extension_counts[suffix or "(无后缀)"] += 1
            if suffix not in {".json", ".txt", ".csv", ".yaml", ".yml", ".mp4"}:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                size = -1
            kind = classify_json(path) if suffix == ".json" else suffix
            rows.append(
                {
                    "path": str(path),
                    "relative": str(path.relative_to(root)),
                    "size": size,
                    "kind": kind,
                }
            )
    rows.sort(key=lambda item: item["relative"])
    return {"files": rows, "extension_counts": dict(extension_counts)}


def inspect_sources() -> dict[str, Any]:
    sources = load_sources()
    datasets = {}
    for name in sources["datasets"]:
        root = dataset_root(sources, name)
        datasets[name] = {"root": str(root), **scan_files(root)}

    sequences = []
    for sequence in sources["sequences"]:
        files = sequence_files(sources, sequence)
        probes = {camera_id: probe_video(path) for camera_id, path in files["videos"].items()}
        caches = {}
        for camera_id, path in files["frame_caches"].items():
            caches[camera_id] = load_json(path)
        class_counts: dict[str, int] = {}
        record_count = None
        if sequence["dataset"] == "result":
            counts = _count_classes_in_array(files["pixel_file"], None)
            class_counts = dict(counts)
            record_count = sum(counts.values())
            head = _read_head(files["bbox_info"], 400)
        else:
            counts: Counter[str] = Counter()
            total = 0
            for path in files["tracks"].values():
                for row in iter_track_observations(path, str(path)):
                    counts[str(row["class_name_raw"])] += 1
                    total += 1
            class_counts = dict(counts)
            record_count = total
            head = _read_head(files["manual_global"], 300)
        manual = None
        if sequence["dataset"] == "R39_new":
            payload = load_json(files["manual_global"])
            auto_payload = load_json(files["auto_global"])
            manual = {
                "manual_tracks": payload.get("num_input_tracks"),
                "manual_global_objects": payload.get("num_global_objects"),
                "auto_tracks": auto_payload.get("num_input_tracks"),
                "auto_global_objects": auto_payload.get("num_global_objects"),
                "has_manual_fix": "manual_fix" in payload,
            }
        sequences.append(
            {
                "sequence_id": sequence["sequence_id"],
                "dataset": sequence["dataset"],
                "relative_dir": sequence["relative_dir"],
                "origin_sequence_dir": sequence["origin_sequence_dir"],
                "capture_time_candidate": sequence["capture_time_candidate"],
                "capture_time_status": "unverified_directory_name",
                "record_count": record_count,
                "class_counts": class_counts,
                "probes": probes,
                "frame_caches": caches,
                "manual": manual,
                "include_detr": sequence.get("include_detr", False),
                "has_historical_calibration": bool(files.get("calibration") and files["calibration"] and files["calibration"].exists()),
                "header_excerpt": head,
            }
        )
    return {
        "datasets": datasets,
        "sequences": sequences,
        "unverified_groups": sources["unverified_groups"],
        "project_root": str(PROJECT_ROOT),
    }
