#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.common import box_iou, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行STMC全局多目标跟踪")
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--stmc-python",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def prepare_dataset(detection_data: dict, dataset_path: Path) -> None:
    scene_path = dataset_path / "scene1"
    frame_dirs = {}
    for frame in detection_data["frames"]:
        frame_dirs.setdefault(frame["camera_id"], Path(frame["image_path"]).parent)
    for camera in detection_data["cameras"]:
        camera_id = camera["camera_id"]
        camera_dir = scene_path / camera_id
        camera_dir.mkdir(parents=True, exist_ok=True)
        image_link = camera_dir / "sam3_multi_frames"
        if image_link.is_symlink() or image_link.exists():
            if image_link.is_symlink():
                image_link.unlink()
            elif image_link.is_dir():
                shutil.rmtree(image_link)
            else:
                image_link.unlink()
        os.symlink(frame_dirs[camera_id], image_link, target_is_directory=True)
        calibration = Path(camera["calibration_path"])
        if not calibration.is_file():
            raise FileNotFoundError(f"标定文件不存在{calibration}")
        shutil.copy2(calibration, camera_dir / "calibration.json")


def write_resources(
    detection_data: dict,
    metadata: dict,
    features: np.ndarray,
    output_dir: Path,
) -> None:
    records_by_camera = defaultdict(list)
    for record in metadata["records"]:
        records_by_camera[record["camera_id"]].append(record)
    dataset_name = "Roadside"
    for camera in detection_data["cameras"]:
        camera_id = camera["camera_id"]
        target = output_dir / f"{dataset_name}_scene1-{camera_id}_RDDETR_resnet50.txt"
        with target.open("w", encoding="utf-8") as file:
            for record in records_by_camera[camera_id]:
                x1, y1, x2, y2 = record["bbox_xyxy"]
                prefix = [
                    record["frame_index"] + 1,
                    -1,
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                    record["score"],
                    -1,
                    -1,
                    record["class_id"],
                ]
                feature = features[int(record["feature_index"])]
                values = [*(f"{value:.6f}" for value in prefix), *(f"{value:.7f}" for value in feature)]
                file.write(",".join(values) + "\n")


def run_stmc(
    args: argparse.Namespace,
    dataset_path: Path,
    resources_path: Path,
    metadata_path: Path,
    feature_dim: int,
    stmc_output: Path,
) -> Path:
    stmc_root = ROOT / "algorithms/stmc"
    command = [
        str(args.stmc_python),
        "-m",
        "tools.track",
        "+experiment=R39",
        f"dataset_path={dataset_path}",
        f"output_path={stmc_output}",
        f"resources.path={resources_path}",
        "resources.detector=RDDETR",
        "resources.reid=resnet50",
        "dataset.name=Roadside",
        "dataset.scene_path=scene1",
        f"tracker.fdim={feature_dim}",
        f"export_json.class_metadata_path={metadata_path}",
        "export_json.enable=true",
        "visuals.plot_results=false",
        "visuals.store_files=false",
        f"device={args.device}",
        f"hydra.run.dir={stmc_output / 'hydra'}",
    ]
    subprocess.run(command, cwd=stmc_root, check=True)
    result_path = stmc_output / "Roadside" / "tracking_results.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"STMC结果不存在{result_path}")
    return result_path


def associate_tracks(detection_data: dict, tracking_data: dict, threshold: float) -> dict:
    track_index = defaultdict(list)
    for record in tracking_data["records"]:
        key = (record["camera_key"], int(record["frame_index"]))
        track_index[key].append(record)

    track_observations = defaultdict(list)
    output_frames = []
    for frame in detection_data["frames"]:
        detections = [
            {
                **detection,
                "global_track_id": None,
                "world_position": None,
                "plate": "",
            }
            for detection in frame["detections"]
        ]
        candidates = track_index.get((frame["camera_id"], frame["frame_index"]), [])
        if detections and candidates:
            costs = np.ones((len(detections), len(candidates)), dtype=np.float64)
            for row, detection in enumerate(detections):
                for column, candidate in enumerate(candidates):
                    if detection["class_id"] != candidate.get("class_id"):
                        continue
                    candidate_box = candidate["bbox_xyxy"]
                    box = [candidate_box[key] for key in ("x0", "y0", "x1", "y1")]
                    costs[row, column] = 1.0 - box_iou(detection["bbox_xyxy"], box)
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                iou = 1.0 - costs[row, column]
                if iou < threshold:
                    continue
                candidate = candidates[column]
                track_key = (int(candidate["track_id"]), detections[row]["class_id"])
                detections[row]["global_track_id"] = track_key
                detections[row]["world_position"] = candidate.get("bev")
                track_observations[track_key].append(
                    {
                        "camera_id": frame["camera_id"],
                        "frame_index": frame["frame_index"],
                        "timestamp_ms": frame["timestamp_ms"],
                        "detection_id": detections[row]["detection_id"],
                        "class_id": detections[row]["class_id"],
                        "score": detections[row]["score"],
                    }
                )
        output_frames.append({**frame, "detections": detections})

    final_ids = {
        track_key: final_id
        for final_id, track_key in enumerate(sorted(track_observations), start=1)
    }
    for frame in output_frames:
        for detection in frame["detections"]:
            track_key = detection["global_track_id"]
            if track_key is not None:
                detection["global_track_id"] = final_ids[track_key]

    tracks = []
    for track_key, observations in sorted(track_observations.items()):
        class_id = track_key[1]
        camera_counts = Counter(item["camera_id"] for item in observations)
        tracks.append(
            {
                "global_track_id": final_ids[track_key],
                "class_id": class_id,
                "class_name": detection_data["categories"][class_id]["name"],
                "plate": "",
                "plate_confidence": None,
                "start_timestamp_ms": min(item["timestamp_ms"] for item in observations),
                "end_timestamp_ms": max(item["timestamp_ms"] for item in observations),
                "observation_count": len(observations),
                "camera_observation_counts": dict(camera_counts),
            }
        )

    return {
        "scene_id": detection_data["scene_id"],
        "fps": detection_data["fps"],
        "feature_provider": None,
        "cameras": detection_data["cameras"],
        "categories": detection_data["categories"],
        "frames": output_frames,
        "tracks": tracks,
        "errors": [],
    }


def main() -> int:
    args = parse_args()
    detections_path = args.detections.resolve()
    features_dir = args.features_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_data = load_json(detections_path)
    metadata_path = features_dir / "feature_metadata.json"
    metadata = load_json(metadata_path)
    features = np.load(features_dir / "features.npy", mmap_mode="r")
    if len(metadata["records"]) != features.shape[0]:
        raise ValueError("特征数量与元数据不一致")
    dataset_path = output_dir / "stmc_dataset"
    resources_path = output_dir / "stmc_input"
    stmc_output = output_dir / "stmc_output"
    resources_path.mkdir(parents=True, exist_ok=True)
    prepare_dataset(detection_data, dataset_path)
    write_resources(detection_data, metadata, features, resources_path)
    raw_path = run_stmc(
        args,
        dataset_path,
        resources_path,
        metadata_path,
        int(metadata["feature_dim"]),
        stmc_output,
    )
    result = associate_tracks(
        detection_data, load_json(raw_path), args.iou_threshold
    )
    result["feature_provider"] = "resnet50"
    save_json(output_dir / "tracking.json", result)
    print(f"跟踪结果已写入{output_dir / 'tracking.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
