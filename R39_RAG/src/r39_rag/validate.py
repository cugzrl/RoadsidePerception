"""复查统一数据和软链接"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from r39_rag.config import PROJECT_ROOT, load_schema, load_sources
from r39_rag.utils.json_stream import load_json

OBSERVATION_KEYS = [item["name"] for item in load_schema()["observation_fields"]]


def _check_links() -> list[str]:
    errors = []
    for item in load_sources()["links"]:
        link = PROJECT_ROOT / item["link"]
        source = Path(item["source"])
        if not link.is_symlink():
            errors.append(f"不是软链接: {link}")
            continue
        current = Path(os.readlink(link))
        if current != source:
            errors.append(f"链接指向错误: {link} -> {current}")
        if not link.exists():
            errors.append(f"链接失效: {link}")
    return errors


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"无法解析 {path}:{line_number}") from exc


def validate_normalized() -> dict[str, Any]:
    errors: list[str] = []
    checked = 0
    root = PROJECT_ROOT / "data" / "normalized" / "sequences"
    if not root.exists():
        return {"ok": False, "errors": ["还没有统一数据"], "checked": 0, "git": []}
    errors.extend(_check_links())
    for metadata_path in sorted(root.glob("*/metadata.json")):
        metadata = load_json(metadata_path)
        sequence_id = metadata_path.parent.name
        if metadata["sequence_id"] != sequence_id:
            errors.append(f"metadata 序列号不一致: {metadata_path}")
        if metadata["capture_start_time"] is not None:
            errors.append(f"未核实的采集时间被写成了确定值: {sequence_id}")
        for version in metadata["source_versions"]:
            version_dir = metadata_path.parent / "versions" / version
            observations = version_dir / "observations.jsonl"
            count = 0
            seen_uid = set()
            seen_local = set()
            seen_global = set()
            duplicate_reports = 0
            for line_number, row in _iter_jsonl(observations):
                count += 1
                checked += 1
                missing = [key for key in OBSERVATION_KEYS if key not in row]
                if missing:
                    errors.append(f"{observations}:{line_number} 缺少 {missing}")
                    break
                if row["sequence_id"] != sequence_id or row["source_version"] != version:
                    errors.append(f"{observations}:{line_number} 序列或版本不匹配")
                if row["record_uid"] in seen_uid:
                    errors.append(f"重复 record_uid {row['record_uid']}")
                seen_uid.add(row["record_uid"])
                bbox = row["bbox_xyxy"]
                if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 4):
                    errors.append(f"{observations}:{line_number} 检测框格式错误")
                if version.startswith("sam3") and row["score"] is not None:
                    errors.append(f"{observations}:{line_number} 将缺失置信度写成了 {row['score']}")
                if version.startswith("sam3") and row["world_position"] is not None:
                    errors.append(f"{observations}:{line_number} 把未核实坐标写入了 world_position")
                expected_class = {
                    "truck": ("Car", 0),
                    "motorcycle": ("ScooterRider", 4),
                    "bicycle": ("Cyclist", 5),
                }.get(row["class_name_raw"])
                if expected_class and (row["class_name"], row["class_id"]) != expected_class:
                    errors.append(f"{observations}:{line_number} 类别映射不符合约定 {row['class_name_raw']}")
                if row["class_name_raw"] == "id" and row["class_id"] is not None:
                    errors.append(f"{observations}:{line_number} 占位类别 id 被写入了 class_id")
                if row["local_track_id"] is not None:
                    local_key = (row["camera_id"], row["frame_index"], row["local_track_id"])
                    if local_key in seen_local and duplicate_reports < 8:
                        errors.append(f"{observations}:{line_number} 局部 ID 在同一帧重复 {local_key}")
                        duplicate_reports += 1
                    seen_local.add(local_key)
                if row["global_track_id"] is not None:
                    global_key = (row["camera_id"], row["frame_index"], row["global_track_id"])
                    if global_key in seen_global and duplicate_reports < 8:
                        errors.append(f"{observations}:{line_number} 全局 ID 在同一帧重复 {global_key}")
                        duplicate_reports += 1
                    seen_global.add(global_key)
            expected = metadata["versions"][version]["written_observations"]
            if count != expected:
                errors.append(f"{sequence_id}/{version} 行数 {count} 与记录 {expected} 不一致")
            for name in ("tracklets.jsonl", "global_tracks.jsonl"):
                list(_iter_jsonl(version_dir / name))
    git = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    protected_prefixes = ("pipeline/", "scripts/", "algorithms/", "configs/cameras/", "schemas/")
    visualization_allow = {
        "pipeline/bev_map.py",
        "scripts/visualize_bev.py",
        "scripts/run_pipeline.py",
    }
    touched = []
    for line in git:
        changed = line[3:].strip()
        if changed in visualization_allow:
            continue
        if any(changed.startswith(prefix) for prefix in protected_prefixes):
            touched.append(line)
    if touched:
        errors.append("原有流水线路径出现改动: " + "; ".join(touched))
    return {"ok": not errors, "errors": errors, "checked": checked, "git_entries": len(git)}
