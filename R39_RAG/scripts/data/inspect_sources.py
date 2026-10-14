#!/usr/bin/env python3
"""只读盘点原始JSON、视频和采集序列"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from r39_rag.config import PROJECT_ROOT
from r39_rag.inventory import inspect_sources
from r39_rag.reporting import write_inventory, write_schema_doc
from r39_rag.utils.json_stream import dump_json


def main() -> None:
    parser = argparse.ArgumentParser(description="盘点R39历史结构化数据和视频元数据")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "data_inventory.md")
    args = parser.parse_args()
    report = inspect_sources()
    dump_json(PROJECT_ROOT / "reports" / "inventory.json", report)
    write_inventory(report, args.output)
    write_schema_doc(PROJECT_ROOT / "reports" / "unified_schema.md")
    print(f"已写入 {args.output}")


if __name__ == "__main__":
    main()
