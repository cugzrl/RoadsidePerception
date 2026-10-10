#!/usr/bin/env python3
"""检查四路 Homography 是否落到同一地面坐标"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r39_rag.config import PROJECT_ROOT, load_knowledge, load_sources
from r39_rag.projection.evaluate import evaluate_sequences, write_projection_report
from r39_rag.visualization.plots import plot_camera_scatter, plot_error_hist, use_chinese_font


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 R39 地面投影")
    parser.add_argument("--sequence", action="append", default=[], help="只检查指定 sequence_id，可重复")
    args = parser.parse_args()
    config = load_knowledge()
    use_chinese_font(config["font_path"])
    sequence_ids = args.sequence or [item["sequence_id"] for item in load_sources()["sequences"]]
    result = evaluate_sequences(sequence_ids, config)
    report = PROJECT_ROOT / "reports" / "bev_projection_report.md"
    write_projection_report(result, report)
    figure_dir = PROJECT_ROOT / "reports" / "knowledge"
    plot_error_hist(result["distances"], figure_dir / "cross_camera_error.png")
    plot_camera_scatter(result["samples"], figure_dir / "bev_cameras.png")
    status = {
        "allow_fusion": result["allow_fusion"],
        "median_m": result["overall"]["median_m"],
        "p90_m": result["overall"]["p90_m"],
        "pair_count": result["overall"]["count"],
        "fraction_within_5m": result["overall"]["fraction_within_5m"],
        "passing_pairs": result["passing_pairs"],
        "reliable_pairs": result["reliable_pairs"],
        "sequence_ids": sequence_ids,
        "ok": result["counts"]["ok"],
        "input": result["counts"]["input"],
    }
    status_path = PROJECT_ROOT / "knowledge_base" / "projection_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))
    print(f"报告 {report}")


if __name__ == "__main__":
    main()
