#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行路侧端到端感知流水线")
    parser.add_argument("--videos", nargs="+", type=Path, required=True)
    parser.add_argument("--camera-ids", nargs="+")
    parser.add_argument("--calibrations", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-id", default="roadside_scene")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--det-threshold", type=float, default=0.6)
    parser.add_argument("--det-min-box-area", type=float, default=200.0)
    parser.add_argument("--class-thresholds", nargs="*", default=[])
    parser.add_argument("--class-min-box-areas", nargs="*", default=[])
    parser.add_argument("--det-batch-size", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--stmc-python",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument("--no-plate", action="store_true")
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command: list[str], output: Path | None, resume: bool) -> None:
    if resume and output is not None and output.exists():
        print(f"复用已有结果{output}")
        return
    print("执行" + " ".join(command))
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    if not 1 <= len(args.videos) <= 4:
        raise ValueError("只支持一到四路视频")
    if len(args.videos) != len(args.calibrations):
        raise ValueError("视频和标定文件数量必须一致")
    if args.camera_ids and len(args.camera_ids) != len(args.videos):
        raise ValueError("相机ID数量必须与视频数量一致")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    extract_command = [
        sys.executable,
        str(SCRIPTS / "extract_frames.py"),
        "--videos",
        *(str(path) for path in args.videos),
        "--calibrations",
        *(str(path) for path in args.calibrations),
        "--output",
        str(output),
        "--scene-id",
        args.scene_id,
        "--fps",
        str(args.fps),
    ]
    if args.camera_ids:
        extract_command.extend(["--camera-ids", *args.camera_ids])
    if args.max_frames is not None:
        extract_command.extend(["--max-frames", str(args.max_frames)])
    if args.overwrite:
        extract_command.append("--overwrite")
    run(extract_command, manifest, args.resume)

    detections = output / "detections.json"
    detection_command = [
        sys.executable,
        str(SCRIPTS / "run_detection.py"),
        "--manifest",
        str(manifest),
        "--output",
        str(detections),
        "--threshold",
        str(args.det_threshold),
        "--min-box-area",
        str(args.det_min_box_area),
        "--batch-size",
        str(args.det_batch_size),
        "--device",
        args.device,
    ]
    if args.class_thresholds:
        detection_command.extend(["--class-thresholds", *args.class_thresholds])
    if args.class_min_box_areas:
        detection_command.extend(
            ["--class-min-box-areas", *args.class_min_box_areas]
        )
    run(detection_command, detections, args.resume)

    features_dir = output / "features"
    metadata_path = features_dir / "feature_metadata.json"
    run(
        [
            sys.executable,
            str(SCRIPTS / "run_features.py"),
            "--detections",
            str(detections),
            "--output-dir",
            str(features_dir),
            "--batch-size",
            str(args.feature_batch_size),
            "--device",
            args.device,
        ],
        metadata_path,
        args.resume,
    )
    tracking_dir = output / "tracking"
    tracking_path = tracking_dir / "tracking.json"
    run(
        [
            sys.executable,
            str(SCRIPTS / "run_tracking.py"),
            "--detections",
            str(detections),
            "--features-dir",
            str(features_dir),
            "--output-dir",
            str(tracking_dir),
            "--stmc-python",
            str(args.stmc_python),
            "--device",
            args.device,
        ],
        tracking_path,
        args.resume,
    )
    final_result = output / "result.json"
    if args.no_plate:
        final_result.parent.mkdir(parents=True, exist_ok=True)
        if not (args.resume and final_result.exists()):
            final_result.write_bytes(tracking_path.read_bytes())
    else:
        run(
            [
                sys.executable,
                str(SCRIPTS / "run_plate_recognition.py"),
                "--tracking",
                str(tracking_path),
                "--output",
                str(final_result),
            ],
            final_result,
            args.resume,
        )
    if not args.no_visualization:
        visualization = output / "visualization.mp4"
        run(
            [
                sys.executable,
                str(SCRIPTS / "visualize_result.py"),
                "--result",
                str(final_result),
                "--output",
                str(visualization),
            ],
            visualization,
            args.resume,
        )
    print(f"流水线完成{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
