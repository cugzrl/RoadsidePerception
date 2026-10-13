#!/usr/bin/env python3
"""按 A0 A1 A2 B C D E 跑留出序列 不覆盖上一轮 variants"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
GRAPH = ROOT / "configs" / "maps" / "r39_lane_graph.json"
KV = ROOT / "R39_RAG" / "knowledge_base" / "kv_motion_class.npz"
HIGH = "[0.65,0.65,0.65,0.70,0.72,0.72,0.75]"
# 一次核查得到的工作点 不再在留出序列上搜索
REID = "[0.80,0.80,0.80,0.75,0.75,0.75,0.78]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True, help="070327 或 070428")
    parser.add_argument("--source", type=Path, required=True, help="已有 detections.json 的目录")
    parser.add_argument("--low-detections", type=Path, required=True)
    parser.add_argument("--mot-features", type=Path, required=True, help="原检测框上的 mot20 特征")
    parser.add_argument("--low-features", type=Path, required=True, help="低分候选上的 mot20 特征")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", default=["A0", "A1", "A2", "A2assoc", "B", "C", "D", "E"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--a0-tracking", type=Path, default=None, help="上一轮 A+ tracking.json")
    return parser.parse_args()


def base_flags(log_path: Path | None = None) -> list[str]:
    flags = [
        "tracker.position_state.active_cameras_only=true",
        "tracker.short_recovery.enabled=true",
        "tracker.knowledge.enabled=false",
        "tracker.motion_memory.enabled=false",
        "tracker.link_guard.enabled=false",
    ]
    return flags


def memory_flags(name: str, log_path: Path) -> list[str]:
    flags = [
        "tracker.position_state.active_cameras_only=true",
        "tracker.short_recovery.enabled=true",
        "tracker.short_recovery.one_to_one=true",
        "tracker.knowledge.enabled=false",
        "tracker.link_guard.enabled=true",
        f"tracker.link_guard.log_path={log_path.with_name('link_guard.jsonl')}",
        "tracker.low_confidence_thresh=0.30",
        f"tracker.class_high_thresh={HIGH}",
        "tracker.motion_memory.enabled=true",
        "tracker.motion_memory.min_missed=4",
        "tracker.motion_memory.max_missed=30",
        "tracker.motion_memory.reid_min=0.75",
        "tracker.motion_memory.static_reid_min=0.85",
        f"tracker.motion_memory.class_reid_min={REID}",
        "tracker.motion_memory.strict_protect=true",
        f"tracker.motion_memory.log_path={log_path}",
        f"tracker.motion_memory.kv_path={KV}",
        f"tracker.motion_memory.lane_graph={GRAPH}",
        "tracker.motion_memory.kv_class_aware=false",
        "tracker.motion_memory.kv_diverse=false",
        "tracker.motion_memory.kv_blend=false",
        "tracker.motion_memory.use_topology=false",
        "tracker.motion_memory.use_kv=false",
    ]
    if name == "C":
        flags += ["tracker.motion_memory.use_topology=true"]
    if name == "D":
        flags += [
            "tracker.motion_memory.use_kv=true",
            "tracker.motion_memory.kv_class_aware=true",
            "tracker.motion_memory.kv_diverse=true",
            "tracker.motion_memory.kv_blend=true",
        ]
    if name == "E":
        flags += [
            "tracker.motion_memory.use_topology=true",
            "tracker.motion_memory.use_kv=true",
            "tracker.motion_memory.kv_class_aware=true",
            "tracker.motion_memory.kv_diverse=true",
            "tracker.motion_memory.kv_blend=true",
        ]
    return flags


def track(detections: Path, features: Path, output: Path, overrides: list[str], device: str) -> float:
    output.mkdir(parents=True, exist_ok=True)
    command = [
        PY,
        str(ROOT / "scripts" / "run_tracking.py"),
        "--detections",
        str(detections),
        "--features-dir",
        str(features),
        "--output-dir",
        str(output),
        "--device",
        device,
        "--stmc-override",
        *overrides,
    ]
    started = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    return time.perf_counter() - started


def write_meta(output: Path, payload: dict) -> None:
    (output / "run_meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if (output / "tracking.json").is_file():
        shutil.copyfile(output / "tracking.json", output / "result.json")


def main() -> int:
    args = parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        code = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        code = "unknown"
    for name in args.groups:
        output = root / name
        if name == "A0":
            if args.a0_tracking is None or not args.a0_tracking.is_file():
                raise FileNotFoundError("A0 需要上一轮 A+ tracking.json")
            output.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.a0_tracking, output / "tracking.json")
            shutil.copyfile(args.a0_tracking, output / "result.json")
            write_meta(
                output,
                {
                    "group": "A0",
                    "sequence": args.sequence,
                    "note": "复制上一轮修复后的 A+ 不重跑 避免覆盖历史视频",
                    "source_tracking": str(args.a0_tracking),
                    "code": code,
                },
            )
            print("A0 copied")
            continue
        if name == "A1":
            detections, features = args.source / "detections.json", args.mot_features
            overrides = base_flags()
        elif name == "A2assoc":
            detections, features = args.source / "detections.json", args.mot_features
            overrides = base_flags() + [
                "tracker.short_recovery.one_to_one=true",
                "tracker.link_guard.enabled=true",
                f"tracker.link_guard.log_path={output / 'link_guard.jsonl'}",
            ]
        elif name == "A2":
            detections, features = args.low_detections, args.low_features
            overrides = base_flags() + [
                "tracker.short_recovery.one_to_one=true",
                "tracker.link_guard.enabled=true",
                f"tracker.link_guard.log_path={output / 'link_guard.jsonl'}",
                "tracker.low_confidence_thresh=0.30",
                f"tracker.class_high_thresh={HIGH}",
            ]
        elif name == "F":
            detections, features = args.low_detections, args.low_features
            overrides = base_flags() + [
                "tracker.short_recovery.one_to_one=true",
                "tracker.link_guard.enabled=true",
                f"tracker.link_guard.log_path={output / 'link_guard.jsonl'}",
                "tracker.low_confidence_thresh=0.30",
                f"tracker.class_high_thresh={HIGH}",
                "tracker.static_pseudo.enabled=true",
                f"tracker.static_pseudo.log_path={output / 'static_pseudo.jsonl'}",
            ]
        elif name == "S":
            detections, features = args.low_detections, args.low_features
            overrides = base_flags() + [
                "tracker.short_recovery.one_to_one=true",
                "tracker.link_guard.enabled=true",
                f"tracker.link_guard.log_path={output / 'link_guard.jsonl'}",
                "tracker.low_confidence_thresh=0.30",
                f"tracker.class_high_thresh={HIGH}",
                "tracker.static_pseudo.enabled=true",
                f"tracker.static_pseudo.log_path={output / 'static_pseudo.jsonl'}",
                "tracker.cross_camera_prior.enabled=true",
                f"tracker.cross_camera_prior.log_path={output / 'cross_camera.jsonl'}",
            ]
        elif name in {"M", "T", "K", "Full"}:
            base_name = {"M": "B", "T": "C", "K": "D", "Full": "E"}[name]
            detections, features = args.low_detections, args.low_features
            overrides = memory_flags(base_name, output / "memory_log.jsonl") + [
                "tracker.static_pseudo.enabled=true",
                f"tracker.static_pseudo.log_path={output / 'static_pseudo.jsonl'}",
                "tracker.cross_camera_prior.enabled=true",
                f"tracker.cross_camera_prior.log_path={output / 'cross_camera.jsonl'}",
            ]
        elif name in {"B", "C", "D", "E"}:
            detections, features = args.low_detections, args.low_features
            overrides = memory_flags(name, output / "memory_log.jsonl")
        else:
            raise ValueError(name)
        if output.exists():
            shutil.rmtree(output)
        elapsed = track(detections, features, output, overrides, args.device)
        write_meta(
            output,
            {
                "group": name,
                "sequence": args.sequence,
                "elapsed_s": elapsed,
                "detections": str(detections),
                "features": str(features),
                "overrides": overrides,
                "code": code,
                "reid": "mot20 DeepSORT 512",
            },
        )
        print(f"{name} {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
