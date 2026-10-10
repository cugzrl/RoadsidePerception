"""按配置清洗全部采集序列"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from r39_rag.cleaning.convert import build_timebase, convert_aligned, convert_detr, convert_tracks
from r39_rag.config import PROJECT_ROOT, load_categories, load_sources, sequence_files
from r39_rag.utils.json_stream import dump_json, load_json
from r39_rag.utils.video import probe_video


def _split_config_note(path: Path | None, sample_fps: float | None) -> str | None:
    if path is None or not path.exists():
        return None
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    interval = ((payload or {}).get("capture") or {}).get("interval_ms")
    start = ((payload or {}).get("video") or {}).get("start_time")
    expected = None if not sample_fps else round(1000.0 / sample_fps)
    return (
        f"split_config.yaml 的 start_time={start}、interval_ms={interval}。"
        f"文件中的路径指向其他数据目录，且间隔与抽帧帧率 {sample_fps}（约 {expected} ms）不一致，不作为时间依据"
    )


def _metadata(sequence: dict[str, Any], files: dict[str, Any], probes: dict[str, Any], timebase: dict[str, Any], versions: dict[str, Any], sources: dict[str, Any]) -> dict[str, Any]:
    cameras = list(files["videos"])
    pipeline_dir = files["pipeline_calibration_dir"]
    pipeline_files = {
        camera_id: str(pipeline_dir / camera_id / "calibration.json")
        for camera_id in cameras
    }
    notes = [
        "capture_start_time 为空，因为目录时间没有被视频元数据或其他独立时钟证实",
        "相同 frame_index 只表示抽帧序号一致，不能单独证明四路相机曝光同步",
        timebase["timebase_source"],
    ]
    split_note = _split_config_note(files.get("split_config"), timebase.get("sample_fps"))
    if split_note:
        notes.append(split_note)
    for group in sources["unverified_groups"]:
        if sequence["sequence_id"] in group["sequence_ids"]:
            notes.append(group["note"])
    return {
        "schema_version": sources["schema_version"],
        "site_id": sources["site_id"],
        "sequence_id": sequence["sequence_id"],
        "dataset": sequence["dataset"],
        "relative_dir": sequence["relative_dir"],
        "origin_sequence_dir": sequence["origin_sequence_dir"],
        "source_versions": list(versions),
        "source_files": sorted({item for version in versions.values() for item in version.get("source_files", []) if item}),
        "camera_ids": cameras,
        "video_paths": {camera_id: str(path) for camera_id, path in files["videos"].items()},
        "video_probe": probes,
        "calibration_paths": {
            "historical": None if not files.get("calibration") else str(files["calibration"]),
            "pipeline_homography": pipeline_files,
        },
        "capture_start_time": None,
        "capture_time_candidate": sequence["capture_time_candidate"],
        "capture_time_status": "unverified_directory_name",
        "source_fps": timebase.get("source_fps"),
        "sample_fps": timebase.get("sample_fps"),
        "frame_stride": timebase.get("frame_stride"),
        "frame_count": timebase.get("frame_count"),
        "video_frames": timebase.get("video_frames"),
        "image_width": timebase.get("image_width"),
        "image_height": timebase.get("image_height"),
        "timebase_source": timebase.get("timebase_source"),
        "video_size_matches_frame_cache": timebase.get("video_size_matches_frame_cache"),
        "sync_status": timebase.get("sync_status"),
        "versions": versions,
        "notes": notes,
    }


def normalize_all(sequence_ids: set[str] | None = None) -> dict[str, Any]:
    sources = load_sources()
    categories = load_categories()
    normalized = PROJECT_ROOT / "data" / "normalized"
    quarantine = PROJECT_ROOT / "data" / "quarantine"
    normalized.mkdir(parents=True, exist_ok=True)
    quarantine.mkdir(parents=True, exist_ok=True)
    written = []
    for sequence in sources["sequences"]:
        sequence_id = sequence["sequence_id"]
        if sequence_ids and sequence_id not in sequence_ids:
            continue
        print(f"开始清洗 {sequence_id}")
        files = sequence_files(sources, sequence)
        probes = {camera_id: probe_video(path) for camera_id, path in files["videos"].items()}
        timebase = build_timebase(files["frame_caches"], probes)
        versions: dict[str, Any] = {}
        if sequence["dataset"] == "result":
            versions["sam3_aligned"] = convert_aligned(
                site_id=sources["site_id"],
                sequence_id=sequence_id,
                source_version="sam3_aligned",
                files=files,
                categories=categories,
                timebase=timebase,
                output_dir=normalized / "sequences" / sequence_id / "versions" / "sam3_aligned",
                quarantine_dir=quarantine / sequence_id / "sam3_aligned",
            )
            if sequence.get("include_detr"):
                print(f"开始清洗 {sequence_id} / detr_car")
                versions["detr_car"] = convert_detr(
                    site_id=sources["site_id"],
                    sequence_id=sequence_id,
                    source_version="detr_car",
                    files=files,
                    categories=categories,
                    timebase=timebase,
                    output_dir=normalized / "sequences" / sequence_id / "versions" / "detr_car",
                    quarantine_dir=quarantine / sequence_id / "detr_car",
                )
        else:
            versions["sam3_manual_gid"] = convert_tracks(
                site_id=sources["site_id"],
                sequence_id=sequence_id,
                source_version="sam3_manual_gid",
                files=files,
                categories=categories,
                timebase=timebase,
                output_dir=normalized / "sequences" / sequence_id / "versions" / "sam3_manual_gid",
                quarantine_dir=quarantine / sequence_id / "sam3_manual_gid",
            )
        metadata = _metadata(sequence, files, probes, timebase, versions, sources)
        dump_json(normalized / "sequences" / sequence_id / "metadata.json", metadata)
        written.append(sequence_id)
        print(f"完成 {sequence_id}: " + ", ".join(f"{name}={item['written_observations']}" for name, item in versions.items()))
    manifest = _manifest(sources)
    dump_json(normalized / "source_manifest.json", manifest)
    return manifest


def _manifest(sources: dict[str, Any]) -> dict[str, Any]:
    root = PROJECT_ROOT / "data" / "normalized" / "sequences"
    sequences = []
    if root.exists():
        for path in sorted(root.glob("*/metadata.json")):
            metadata = load_json(path)
            sequences.append(
                {
                    "sequence_id": metadata["sequence_id"],
                    "dataset": metadata["dataset"],
                    "relative_dir": metadata["relative_dir"],
                    "origin_sequence_dir": metadata["origin_sequence_dir"],
                    "source_versions": metadata["source_versions"],
                    "source_fps": metadata["source_fps"],
                    "sample_fps": metadata["sample_fps"],
                    "frame_count": metadata["frame_count"],
                    "sync_status": metadata["sync_status"],
                    "capture_start_time": metadata["capture_start_time"],
                    "capture_time_candidate": metadata["capture_time_candidate"],
                    "versions": {
                        name: {
                            "written_observations": item.get("written_observations"),
                            "exact_duplicates_removed": item.get("exact_duplicates_removed"),
                            "conflict_records": item.get("conflict_records"),
                            "input_records": item.get("input_records"),
                        }
                        for name, item in metadata["versions"].items()
                    },
                }
            )
    return {
        "schema_version": sources["schema_version"],
        "site_id": sources["site_id"],
        "unverified_groups": sources["unverified_groups"],
        "sequences": sequences,
    }
