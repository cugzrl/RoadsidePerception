#!/usr/bin/env python3
"""清洗并写出统一后的序列数据"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r39_rag.config import PROJECT_ROOT
from r39_rag.jobs import normalize_all
from r39_rag.reporting import write_quality


def main() -> None:
    parser = argparse.ArgumentParser(description="将多版本R39历史JSON清洗为统一JSONL")
    parser.add_argument("--sequence", action="append", default=[], help="只处理指定sequence_id，可重复传入")
    args = parser.parse_args()
    selected = set(args.sequence) if args.sequence else None
    normalize_all(selected)
    write_quality(PROJECT_ROOT / "reports" / "data_quality_report.md")
    print(f"已写入 {PROJECT_ROOT / 'reports' / 'data_quality_report.md'}")


if __name__ == "__main__":
    main()
