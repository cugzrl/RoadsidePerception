#!/usr/bin/env python3
"""检索空间区域或典型轨迹"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r39_rag.config import PROJECT_ROOT
from r39_rag.knowledge.retrieval import KnowledgeBase
from r39_rag.knowledge.store import connect


def _demo(knowledge: KnowledgeBase) -> dict:
    connection = connect(PROJECT_ROOT / "knowledge_base" / "r39_knowledge.sqlite")
    try:
        cell = connection.execute("SELECT * FROM cells ORDER BY clean_track_count DESC LIMIT 1").fetchone()
        prototype = connection.execute("SELECT * FROM prototypes ORDER BY support_tracks DESC LIMIT 1").fetchone()
    finally:
        connection.close()
    x_value = (cell["x_min"] + cell["x_max"]) / 2.0
    y_value = (cell["y_min"] + cell["y_max"]) / 2.0
    spatial = knowledge.query_spatial(x_value, y_value, camera_id="R39_Aw_CamS", class_name="Car")
    points = knowledge.arrays["prototypes"][int(prototype["prototype_id"])].tolist()
    trajectory = knowledge.query_trajectory(points, camera_id=prototype["exemplar_camera"])
    return {"spatial": spatial, "trajectory": trajectory}


def main() -> None:
    parser = argparse.ArgumentParser(description="查询 R39 时空知识库")
    sub = parser.add_subparsers(dest="command", required=True)
    spatial = sub.add_parser("spatial")
    spatial.add_argument("--x", type=float, required=True)
    spatial.add_argument("--y", type=float, required=True)
    spatial.add_argument("--camera", default=None)
    spatial.add_argument("--class-name", default=None)
    trajectory = sub.add_parser("trajectory")
    trajectory.add_argument("--points", nargs="+", required=True, help="若干 x,y")
    trajectory.add_argument("--camera", default=None)
    sub.add_parser("demo")
    args = parser.parse_args()
    knowledge = KnowledgeBase()
    try:
        if args.command == "spatial":
            payload = knowledge.query_spatial(args.x, args.y, camera_id=args.camera, class_name=args.class_name)
        elif args.command == "trajectory":
            points = [[float(part) for part in item.split(",")] for item in args.points]
            payload = knowledge.query_trajectory(points, camera_id=args.camera)
        else:
            payload = _demo(knowledge)
    finally:
        knowledge.close()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
