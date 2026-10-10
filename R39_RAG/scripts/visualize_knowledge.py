#!/usr/bin/env python3
"""画出 BEV 热力图、典型路径和检索示例"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r39_rag.config import PROJECT_ROOT, load_knowledge
from r39_rag.knowledge.retrieval import KnowledgeBase
from r39_rag.knowledge.store import connect, load_arrays
from r39_rag.visualization.plots import plot_heatmap, plot_prototypes, plot_retrieval, use_chinese_font


def main() -> None:
    config = load_knowledge()
    use_chinese_font(config["font_path"])
    arrays = load_arrays(PROJECT_ROOT / "knowledge_base" / "knowledge_arrays.npz")
    figure_dir = PROJECT_ROOT / "reports" / "knowledge"
    plot_heatmap(arrays["grid_count"], config, figure_dir / "spatial_heatmap.png", "历史观测次数")
    plot_heatmap(arrays["grid_tracks"], config, figure_dir / "track_heatmap.png", "未拆分局部轨迹数")
    connection = connect(PROJECT_ROOT / "knowledge_base" / "r39_knowledge.sqlite")
    try:
        rows = connection.execute("SELECT prototype_id, kind FROM prototypes ORDER BY prototype_id").fetchall()
        best = connection.execute("SELECT prototype_id, exemplar_camera FROM prototypes ORDER BY support_tracks DESC LIMIT 1").fetchone()
    finally:
        connection.close()
    kinds = [row["kind"] for row in rows]
    if len(arrays["prototypes"]):
        plot_prototypes(arrays["prototypes"], kinds, figure_dir / "prototypes.png")
    knowledge = KnowledgeBase()
    try:
        prototype = None if best is None else arrays["prototypes"][int(best["prototype_id"])]
        x_value = float(prototype[-1, 0])
        y_value = float(prototype[-1, 1])
        camera_id = None if best is None else best["exemplar_camera"]
        spatial = knowledge.query_spatial(x_value, y_value, camera_id=camera_id, class_name="Car")
        trajectory = None if prototype is None else knowledge.query_trajectory(prototype.tolist(), camera_id=camera_id)
    finally:
        knowledge.close()
    plot_retrieval(arrays["grid_tracks"], config, (x_value, y_value), prototype, figure_dir / "retrieval_example.png")
    payload = {"spatial": spatial, "trajectory": trajectory}
    (figure_dir / "retrieval_example.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"图已写入 {figure_dir}")


if __name__ == "__main__":
    main()
