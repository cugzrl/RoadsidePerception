#!/usr/bin/env python3
"""用相机内高 IoU 链对照全局 ID 统计碎片和疑似合并 不当成 HOTA"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_reid_identity import link_tracklets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-meta", type=Path, required=True)
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    meta = json.loads(args.features_meta.read_text(encoding="utf-8"))
    tracking = json.loads(args.tracking.read_text(encoding="utf-8"))
    det_to_gid = {}
    lengths = Counter()
    classes = Counter()
    for frame in tracking["frames"]:
        for det in frame["detections"]:
            gid = det.get("global_track_id")
            det_to_gid[det["detection_id"]] = gid
            if gid is not None:
                lengths[gid] += 1
    for track in tracking["tracks"]:
        classes[track["class_name"]] += 1
    records = meta["records"]
    # 低分检测和原始特征的 detection_id 可能对不上 只统计能对上的
    known = [record for record in records if record["detection_id"] in det_to_gid]
    index_of = {record["detection_id"]: i for i, record in enumerate(known)}
    chains = link_tracklets(known)
    fragments = Counter()
    merges = Counter()
    examples = []
    chain_ids = []
    for chain in chains:
        gids = []
        for index in chain:
            gid = det_to_gid.get(known[index]["detection_id"])
            if gid is not None:
                gids.append(gid)
        unique = list(dict.fromkeys(gids))
        name = known[chain[0]]["class_name"]
        if len(unique) > 1:
            fragments[name] += 1
            if len(examples) < 30:
                examples.append(
                    {
                        "type": "fragment",
                        "class_name": name,
                        "camera_id": known[chain[0]]["camera_id"],
                        "frames": [known[chain[0]]["frame_index"], known[chain[-1]]["frame_index"]],
                        "global_ids": unique[:6],
                    }
                )
        chain_ids.append((name, unique))
    by_gid = defaultdict(list)
    for name, unique in chain_ids:
        for gid in unique:
            by_gid[gid].append(name)
    for gid, names in by_gid.items():
        # 同一全局 ID 吃进多条核查链 视为疑似合并
        if len(names) >= 2:
            merges[names[0]] += 1
    short = Counter()
    for track in tracking["tracks"]:
        if track["observation_count"] < 10:
            short[track["class_name"]] += 1
    report = {
        "tracks": len(tracking["tracks"]),
        "by_class": dict(classes),
        "short_under_10": dict(short),
        "iou_chains": len(chains),
        "fragmented_chains": dict(fragments),
        "ids_covering_multiple_chains": dict(merges),
        "examples": examples,
        "note": "碎片是一条高 IoU 链被拆成多个全局 ID 合并是一个全局 ID 覆盖多条链 需要人工复核",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "examples"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
