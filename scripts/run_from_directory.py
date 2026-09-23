#!/usr/bin/env python3
"""Run the pipeline on all videos in one directory."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="输入包含1到4路视频的目录，自动匹配标定并运行完整流水线"
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=ROOT / "configs/cameras/r39",
        help="标定根目录，支持 <相机ID>/calibration.json 或 <相机ID>.json",
    )
    parser.add_argument("--camera-ids", nargs="+")
    parser.add_argument("--calibrations", nargs="+", type=Path)
    return parser.parse_known_args()


def main() -> int:
    args, pipeline_args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"视频目录不存在：{input_dir}")

    videos = sorted(
        (
            path.resolve()
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    if not 1 <= len(videos) <= 4:
        raise ValueError(
            f"视频目录需要包含1到4个视频，当前找到{len(videos)}个：{input_dir}"
        )

    camera_ids = args.camera_ids or [video.stem for video in videos]
    if len(camera_ids) != len(videos) or len(set(camera_ids)) != len(camera_ids):
        raise ValueError("相机ID数量必须与视频数量一致且不能重复")

    if args.calibrations:
        calibrations = [path.expanduser().resolve() for path in args.calibrations]
        if len(calibrations) != len(videos):
            raise ValueError("标定文件数量必须与视频数量一致")
    else:
        calibration_dir = args.calibration_dir.expanduser().resolve()
        calibrations = []
        for camera_id in camera_ids:
            candidates = [
                calibration_dir / camera_id / "calibration.json",
                calibration_dir / f"{camera_id}.json",
            ]
            match = next((path for path in candidates if path.is_file()), None)
            if match is None:
                raise FileNotFoundError(
                    f"未找到相机 {camera_id} 的标定文件；请使用 --calibration-dir "
                    f"或 --calibrations 指定。查找位置：{candidates}"
                )
            calibrations.append(match.resolve())

    for calibration in calibrations:
        if not calibration.is_file():
            raise FileNotFoundError(f"标定文件不存在：{calibration}")

    command = [
        sys.executable,
        str(ROOT / "scripts/run_pipeline.py"),
        "--videos",
        *(str(video) for video in videos),
        "--camera-ids",
        *camera_ids,
        "--calibrations",
        *(str(path) for path in calibrations),
        "--output",
        str(args.output.expanduser().resolve()),
        *pipeline_args,
    ]
    print("输入视频：" + ", ".join(str(path) for path in videos), flush=True)
    print("相机ID：" + ", ".join(camera_ids), flush=True)
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
