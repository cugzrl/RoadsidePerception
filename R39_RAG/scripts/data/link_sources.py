#!/usr/bin/env python3
"""把原始数据目录链接到data/raw，不复制文件"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from r39_rag.links import link_all


def main() -> None:
    parser = argparse.ArgumentParser(description="为R39原始数据创建可重复执行的软链接")
    parser.parse_args()
    for name, status, link in link_all():
        print(f"{name}: {status} -> {link}")


if __name__ == "__main__":
    main()
