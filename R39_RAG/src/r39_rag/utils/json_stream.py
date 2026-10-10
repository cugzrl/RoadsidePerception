"""流式读取大型 JSON 数组，避免整文件载入"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def iter_json_array(path: Path, array_key: str | None = None) -> Iterator[Any]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as file:
        buffer = ""
        pos = 0
        eof = False

        def fill() -> None:
            nonlocal buffer, eof
            chunk = file.read(1024 * 1024)
            if chunk == "":
                eof = True
            else:
                buffer += chunk

        def skip_ws() -> None:
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer) or eof:
                    return
                fill()

        fill()
        if array_key is None:
            skip_ws()
            if pos >= len(buffer) or buffer[pos] != "[":
                raise ValueError(f"文件不是 JSON 数组: {path}")
            pos += 1
        else:
            needle = f'"{array_key}"'
            while True:
                found = buffer.find(needle, pos)
                if found >= 0:
                    pos = found + len(needle)
                    break
                if eof:
                    raise ValueError(f"未找到数组字段 {array_key}: {path}")
                keep = max(0, len(buffer) - len(needle) - 2)
                buffer = buffer[keep:]
                pos = 0
                fill()
            skip_ws()
            if pos >= len(buffer) or buffer[pos] != ":":
                raise ValueError(f"字段 {array_key} 后面没有冒号: {path}")
            pos += 1
            skip_ws()
            if pos >= len(buffer) or buffer[pos] != "[":
                raise ValueError(f"字段 {array_key} 不是数组: {path}")
            pos += 1

        while True:
            skip_ws()
            if pos >= len(buffer):
                break
            if buffer[pos] == "]":
                break
            if buffer[pos] == ",":
                pos += 1
                continue
            while True:
                try:
                    item, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"JSON 数组解析失败: {path}") from None
                    fill()
                    continue
                pos = end
                break
            if pos > 2_000_000:
                buffer = buffer[pos:]
                pos = 0
            yield item


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
            file.write("\n")
