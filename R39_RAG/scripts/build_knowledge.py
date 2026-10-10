#!/usr/bin/env python3
"""用通过投影验证的序列构建时空知识库"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r39_rag.config import PROJECT_ROOT, load_knowledge
from r39_rag.knowledge.build import build_knowledge


def main() -> None:
    config = load_knowledge()
    summary = build_knowledge(config, PROJECT_ROOT / "knowledge_base" / "projection_status.json")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
