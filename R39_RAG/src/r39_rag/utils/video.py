"""视频元数据。只读封装信息，不解码画面"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def parse_ratio(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        if den_f == 0:
            return None
        return float(num) / den_f
    return float(value)


def probe_video(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "error": "文件不存在"}
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip() or "ffprobe 失败"
        return {"path": str(path), "error": message}
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if not streams:
        return {"path": str(path), "error": "没有视频流"}
    stream = streams[0]
    duration = stream.get("duration") or (payload.get("format") or {}).get("duration")
    fps = parse_ratio(stream.get("avg_frame_rate")) or parse_ratio(stream.get("r_frame_rate"))
    nb_frames = stream.get("nb_frames")
    return {
        "path": str(path),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps,
        "nb_frames": int(nb_frames) if nb_frames not in (None, "N/A") else None,
        "duration_s": float(duration) if duration not in (None, "N/A") else None,
        "error": None,
    }
