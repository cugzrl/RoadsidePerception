"""建立只读原始数据软链接"""

from __future__ import annotations

import os
from pathlib import Path

from r39_rag.config import PROJECT_ROOT, load_sources


def ensure_link(source: Path, link: Path) -> str:
    if not source.exists():
        raise FileNotFoundError(f"原始数据不存在: {source}")
    if link.is_symlink():
        current = Path(os.readlink(link))
        if current == source:
            return "reuse"
        raise FileExistsError(f"链接已存在但指向 {current}，拒绝覆盖: {link}")
    if link.exists():
        raise FileExistsError(f"目标已存在且不是软链接，拒绝覆盖: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(source)
    return "created"


def link_all() -> list[tuple[str, str, str]]:
    sources = load_sources()
    results = []
    for item in sources["links"]:
        source = Path(item["source"])
        link = PROJECT_ROOT / item["link"]
        status = ensure_link(source, link)
        results.append((item["name"], status, str(link)))
    return results
