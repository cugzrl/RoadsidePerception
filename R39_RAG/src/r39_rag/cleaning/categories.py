"""原始类别到路侧七类的映射"""

from __future__ import annotations

from typing import Any


def apply_category(raw_name: str | None, mapping: dict[str, Any]) -> tuple[str | None, int | None, str, list[str]]:
    if raw_name is None:
        return None, None, "unmapped", ["class_unmapped"]
    spec = mapping.get(str(raw_name).strip().lower())
    if spec and spec.get("status") == "mapped":
        return spec["class_name"], int(spec["class_id"]), "mapped", []
    return str(raw_name), None, "unmapped", ["class_unmapped"]


def apply_pipeline_category(class_name: str, class_id: int | None, pipeline_names: dict[str, int]) -> tuple[str, int | None, str, list[str]]:
    expected = pipeline_names.get(class_name)
    if expected is None:
        return class_name, None, "unmapped", ["class_unmapped"]
    if class_id is not None and int(class_id) != int(expected):
        return class_name, int(expected), "mapped", ["class_id_mismatch_corrected"]
    return class_name, int(expected), "mapped", []
