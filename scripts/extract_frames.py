#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from pipeline.common import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按时间戳把一到四路视频抽成图片")
    parser.add_argument("--videos", nargs="+", type=Path, required=True)
    parser.add_argument("--camera-ids", nargs="+")
    parser.add_argument("--calibrations", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-id", default="roadside_scene")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--quality", type=int, default=2)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def probe_video(video: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(video),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    numerator, denominator = stream.get("avg_frame_rate", "0/1").split("/")
    source_fps = float(numerator) / max(float(denominator), 1.0)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "source_fps": source_fps,
        "duration_ms": round(float(payload["format"].get("duration", 0)) * 1000, 3),
    }


def extract_one(
    video: Path,
    frame_dir: Path,
    fps: float,
    quality: int,
    max_frames: int | None,
    overwrite: bool,
) -> int:
    if frame_dir.exists() and any(frame_dir.glob("*.jpg")):
        if not overwrite:
            raise FileExistsError(f"抽帧目录非空{frame_dir}")
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps=fps={fps}:round=near",
        "-q:v",
        str(quality),
    ]
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])
    command.extend(["-start_number", "1", str(frame_dir / "%06d.jpg")])
    subprocess.run(command, check=True)
    return len(list(frame_dir.glob("*.jpg")))


def main() -> int:
    args = parse_args()
    if not 1 <= len(args.videos) <= 4:
        raise ValueError("只支持一到四路视频")
    if args.fps <= 0:
        raise ValueError("FPS必须大于零")
    camera_ids = args.camera_ids or [path.stem for path in args.videos]
    if len(camera_ids) != len(args.videos) or len(set(camera_ids)) != len(camera_ids):
        raise ValueError("相机ID数量不匹配或存在重复")
    calibrations = args.calibrations or [None] * len(args.videos)
    if len(calibrations) != len(args.videos):
        raise ValueError("标定文件数量必须与视频数量一致")

    output = args.output.resolve()
    camera_records = []
    frame_records = []
    for video, camera_id, calibration in zip(args.videos, camera_ids, calibrations):
        video = video.expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(f"视频不存在{video}")
        if calibration is not None:
            calibration = calibration.expanduser().resolve()
            if not calibration.is_file():
                raise FileNotFoundError(f"标定文件不存在{calibration}")
        metadata = probe_video(video)
        frame_dir = output / "frames" / camera_id
        count = extract_one(
            video,
            frame_dir,
            args.fps,
            args.quality,
            args.max_frames,
            args.overwrite,
        )
        if count == 0:
            raise RuntimeError(f"视频未抽出图片{video}")
        with Image.open(frame_dir / "000001.jpg") as image:
            width, height = image.size
        camera_records.append(
            {
                "camera_id": camera_id,
                "video_path": str(video),
                "calibration_path": str(calibration) if calibration else "",
                "width": width,
                "height": height,
                "source_fps": round(metadata["source_fps"], 6),
                "frame_count": count,
            }
        )
        for frame_index in range(count):
            frame_records.append(
                {
                    "camera_id": camera_id,
                    "frame_index": frame_index,
                    "timestamp_ms": round(frame_index * 1000.0 / args.fps, 3),
                    "image_path": str(frame_dir / f"{frame_index + 1:06d}.jpg"),
                }
            )
        print(f"{camera_id}抽帧完成{count}张")

    manifest = {
        "scene_id": args.scene_id,
        "fps": args.fps,
        "cameras": camera_records,
        "frames": frame_records,
    }
    save_json(output / "manifest.json", manifest, pretty=True)
    print(f"抽帧清单已写入{output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
