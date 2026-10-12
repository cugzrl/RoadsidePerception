"""在同一份检测和 ReID 上跑 B C D 不覆盖上一轮 A+"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
KV_NPZ = ROOT / "R39_RAG" / "knowledge_base" / "kv_motion.npz"
GRAPH = ROOT / "configs" / "maps" / "r39_lane_graph.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--names", nargs="+", default=["B", "C", "D"])
    return parser.parse_args()


def overrides(name: str, log_path: Path) -> list[str]:
    fixed = [
        "tracker.position_state.active_cameras_only=true",
        "tracker.short_recovery.enabled=true",
        "tracker.knowledge.enabled=false",
        "tracker.motion_memory.enabled=true",
        "tracker.motion_memory.min_missed=4",
        "tracker.motion_memory.max_missed=30",
        f"tracker.motion_memory.log_path={log_path}",
        f"tracker.motion_memory.kv_path={KV_NPZ}",
        f"tracker.motion_memory.lane_graph={GRAPH}",
    ]
    if name == "B":
        return [*fixed, "tracker.motion_memory.use_topology=false", "tracker.motion_memory.use_kv=false"]
    if name == "C":
        return [*fixed, "tracker.motion_memory.use_topology=true", "tracker.motion_memory.use_kv=false"]
    if name == "D":
        return [*fixed, "tracker.motion_memory.use_topology=false", "tracker.motion_memory.use_kv=true"]
    raise ValueError(name)


def main() -> int:
    args = parse_args()
    base = args.base.resolve()
    detections = base / "detections.json"
    features = base / "features"
    if not detections.is_file() or not (features / "features.npy").is_file():
        raise FileNotFoundError(base)
    python = sys.executable
    for name in args.names:
        output = base / "variants" / name
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "memory_log.jsonl"
        command = [
            python,
            str(ROOT / "scripts" / "run_tracking.py"),
            "--detections",
            str(detections),
            "--features-dir",
            str(features),
            "--output-dir",
            str(output),
            "--device",
            args.device,
            "--stmc-override",
            *overrides(name, log_path),
        ]
        started = time.perf_counter()
        subprocess.run(command, cwd=ROOT, check=True)
        shutil.copyfile(output / "tracking.json", output / "result.json")
        (output / "run_meta.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "elapsed_s": time.perf_counter() - started,
                    "overrides": overrides(name, log_path),
                    "detections": str(detections),
                    "features": str(features),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"{name} done {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
