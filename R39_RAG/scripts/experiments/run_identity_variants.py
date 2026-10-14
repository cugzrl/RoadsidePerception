"""在同一份检测上进行实验对比"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="已经有 detections.json 和 features 的目录")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def overrides(name: str, log_path: Path) -> list[str]:
    fixed = [
        "tracker.position_state.active_cameras_only=true",
        "tracker.short_recovery.enabled=true",
    ]
    if name == "Aplus":
        return [*fixed, "tracker.knowledge.enabled=false"]
    knowledge = [
        "tracker.knowledge.enabled=true",
        "tracker.knowledge.weight=0.2",
        "tracker.knowledge.min_score=0.45",
        f"tracker.knowledge.log_path={log_path}",
    ]
    if name == "K":
        return [
            *fixed,
            *knowledge,
            "tracker.knowledge.use_trajectory=true",
            "tracker.knowledge.use_spatial=false",
            "tracker.knowledge.use_topology=false",
        ]
    if name == "Dplus":
        return [
            *fixed,
            *knowledge,
            "tracker.knowledge.use_trajectory=true",
            "tracker.knowledge.use_spatial=true",
            "tracker.knowledge.use_topology=true",
        ]
    raise ValueError(name)


def main() -> int:
    args = parse_args()
    base = args.base.resolve()
    detections = base / "detections.json"
    features = base / "features"
    if not detections.is_file() or not (features / "features.npy").is_file():
        raise FileNotFoundError(f"检测或特征还不存在 {base}")
    python = sys.executable
    for name in ("Aplus", "K", "Dplus"):
        output = base / "variants" / name
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "knowledge_log.jsonl"
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
        subprocess.run(command, check=True)
        elapsed = time.perf_counter() - started
        tracking = output / "tracking.json"
        shutil.copy2(tracking, output / "result.json")
        (output / "run_meta.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "elapsed_s": elapsed,
                    "overrides": overrides(name, log_path),
                    "detections": str(detections),
                    "features": str(features),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"{name} {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
