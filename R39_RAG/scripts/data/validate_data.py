#!/usr/bin/env python3
"""检查统一结果、软链接和原工程是否被改动"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from r39_rag.config import PROJECT_ROOT
from r39_rag.utils.json_stream import dump_json
from r39_rag.validate import validate_normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="校验统一JSONL、软链接和字段约束")
    parser.parse_args()
    result = validate_normalized()
    dump_json(PROJECT_ROOT / "reports" / "validation.json", result)
    if result["errors"]:
        for error in result["errors"]:
            print(error)
        raise SystemExit(1)
    print("校验通过")


if __name__ == "__main__":
    main()
