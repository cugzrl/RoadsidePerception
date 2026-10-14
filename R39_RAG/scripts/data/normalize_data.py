#!/usr/bin/env python3
"""清洗并写出统一后的序列数据"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from r39_rag.jobs import normalize_all


def main() -> None:
    parser = argparse.ArgumentParser(description="将多版本 R39 历史 JSON 清洗为统一 JSONL")
    parser.add_argument("--sequence", action="append", default=[], help="只处理指定 sequence_id 可重复传入")
    args = parser.parse_args()
    selected = set(args.sequence) if args.sequence else None
    manifest = normalize_all(selected)
    print(f"已清洗 {len(manifest['sequences'])} 个序列")


if __name__ == "__main__":
    main()
