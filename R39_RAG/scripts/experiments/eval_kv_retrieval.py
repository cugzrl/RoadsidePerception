"""在留出序列上比较匀速 典型轨迹和Key-Value后续位置
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "build"))

from build_kv_knowledge import build  # noqa: E402
from r39_rag.config import load_knowledge  # noqa: E402
from r39_rag.knowledge.kv_motion import KVIndex, constant_velocity, future_at_time  # noqa: E402
from r39_rag.knowledge.online import match_prefix  # noqa: E402


HORIZONS = np.array([0.5, 1.0, 2.0])
VALUE_INDEX = {0.5: 0, 1.0: 1, 2.0: 3}


def prototype_points(key_xy: np.ndarray, prototypes: np.ndarray, speed: float) -> np.ndarray | None:
    found = match_prefix(key_xy, prototypes, top_k=1)
    if not found or found[0]["score"] < 0.45 or not found[0]["continuation"]:
        return None
    continuation = np.asarray(found[0]["continuation"], dtype=np.float64)
    chain = np.vstack([key_xy[-1], continuation])
    step = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    cumulative = np.cumsum(step)
    if speed < 0.5 or cumulative[-1] < 0.5:
        return None
    predicted = []
    for seconds in HORIZONS:
        travel = speed * float(seconds)
        if travel > float(cumulative[-1]):
            predicted.append([np.nan, np.nan])
            continue
        predicted.append(
            [
                float(np.interp(travel, cumulative, chain[1:, 0])),
                float(np.interp(travel, cumulative, chain[1:, 1])),
            ]
        )
    return np.asarray(predicted, dtype=np.float64)


def _errors(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.linalg.norm(predicted - truth, axis=1)


def _summarize(name: str, errors: np.ndarray) -> dict:
    summary = {"name": name}
    for index, seconds in enumerate(HORIZONS):
        column = errors[:, index]
        valid = column[np.isfinite(column)]
        summary[f"error_{seconds:g}s_m"] = None if len(valid) == 0 else float(np.mean(valid))
        summary[f"available_{seconds:g}s"] = int(len(valid))
        for radius in (2.0, 4.0, 8.0):
            summary[f"cover_{seconds:g}s_{radius:g}m"] = None if len(column) == 0 else float(np.mean(np.isfinite(column) & (column <= radius)))
    finite = errors[np.isfinite(errors).all(axis=1)]
    summary["ade_m"] = None if len(finite) == 0 else float(finite.mean())
    summary["fde_m"] = None if len(finite) == 0 else float(finite[:, -1].mean())
    summary["count"] = int(len(errors))
    return summary


def _draw(path: Path, key_xy: np.ndarray, truth: np.ndarray, cv_xy: np.ndarray, kv_paths: list[np.ndarray]) -> None:
    canvas = np.full((640, 640, 3), 28, dtype=np.uint8)
    origin = key_xy[-1]
    scale = 8.0

    def px(points: np.ndarray) -> np.ndarray:
        local = np.asarray(points, dtype=np.float64) - origin
        column = 320 + local[:, 0] * scale
        row = 320 - local[:, 1] * scale
        return np.round(np.stack([column, row], axis=1)).astype(np.int32)

    def line(points: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
        pts = px(points)
        for start, end in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, tuple(start), tuple(end), color, thickness, cv2.LINE_AA)

    for future in kv_paths:
        line(np.vstack([origin, future]), (80, 180, 80), 1)
    line(key_xy, (230, 230, 230), 2)
    line(np.vstack([origin, truth]), (60, 180, 255), 2)
    line(np.vstack([origin, cv_xy]), (180, 180, 60), 2)
    cv2.circle(canvas, tuple(px(origin.reshape(1, 2))[0]), 4, (255, 255, 255), -1)
    cv2.putText(canvas, "white observed", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, "orange reference", (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "yellow constant vel", (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, "green kv topk", (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 180, 80), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def main() -> None:
    config = load_knowledge()
    index = KVIndex.load(ROOT / "knowledge_base" / "kv_motion.npz", ROOT / "knowledge_base" / "kv_motion.sqlite")
    prototypes = np.load(ROOT / "knowledge_base" / "knowledge_arrays.npz")["prototypes"]
    reference = build(config, sequence_ids=list(config["holdout_sequences"]), allow_holdout=True)
    rows = reference["rows"]
    cv_error = []
    proto_error = []
    kv_top1 = []
    kv_topk = []
    reliable = 0
    cases = []
    for row in rows:
        truth = np.stack([row["value_xy"][VALUE_INDEX[float(seconds)]] for seconds in HORIZONS])
        cv_xy = constant_velocity(row["key_xy"], HORIZONS, 1.0)
        cv_error.append(_errors(cv_xy, truth))
        proto = prototype_points(row["key_xy"], prototypes, float(row["speed"]))
        if proto is None:
            proto_error.append(np.full(3, np.nan))
        else:
            proto_error.append(_errors(proto, truth))
        found = index.query(row["key_xy"])
        if not found["reliable"]:
            kv_top1.append(np.full(3, np.nan))
            kv_topk.append(np.full(3, np.nan))
            continue
        reliable += 1
        paths = []
        stacked = []
        for match in found["matches"]:
            points = np.stack([future_at_time(match["future_xy"], float(seconds)) for seconds in HORIZONS])
            paths.append(match["future_xy"])
            stacked.append(_errors(points, truth))
        stacked_array = np.stack(stacked)
        kv_top1.append(stacked_array[0])
        kv_topk.append(stacked_array.min(axis=0))
        cases.append(
            {
                "kind": row["kind"],
                "sequence_id": row["sequence_id"],
                "camera_id": row["camera_id"],
                "local_track_id": row["local_track_id"],
                "cv": float(np.mean(cv_error[-1])),
                "kv": float(np.mean(kv_topk[-1])),
                "key_xy": row["key_xy"],
                "truth": truth,
                "cv_xy": cv_xy,
                "paths": paths,
                "score": found["best_score"],
            }
        )
    report = {
        "note": "误差相对于留出序列里通过同一套过滤的参考轨迹 不是人工真值",
        "reference_windows": len(rows),
        "independent_tracks": reference["meta"]["independent_tracks"],
        "kv_reliable": reliable,
        "kv_reliable_ratio": 0.0 if not rows else reliable / len(rows),
        "constant_velocity": _summarize("constant_velocity", np.asarray(cv_error)),
        "prototype": _summarize("prototype", np.asarray(proto_error) if proto_error else np.zeros((0, 3))),
        "kv_top1": _summarize("kv_top1", np.asarray(kv_top1) if kv_top1 else np.zeros((0, 3))),
        "kv_topk_min": _summarize("kv_topk_min", np.asarray(kv_topk) if kv_topk else np.zeros((0, 3))),
        "uses_topology_filter": False,
    }
    out = ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "kv_retrieval_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    picture_dir = out / "kv_cases"
    if cases:
        ordered = sorted(cases, key=lambda item: item["kv"] - item["cv"])
        picked = []
        for item in ordered:
            if item["kind"] != "straight":
                picked.append(item)
                break
        picked.append(ordered[0])
        picked.append(ordered[len(ordered) // 2])
        picked.append(ordered[-1])
        for index, item in enumerate(picked):
            _draw(
                picture_dir / f"case_{index}_{item['kind']}.png",
                item["key_xy"],
                item["truth"],
                item["cv_xy"],
                item["paths"],
            )
    print(json.dumps({key: report[key] for key in ("reference_windows", "kv_reliable_ratio", "constant_velocity", "prototype", "kv_top1", "kv_topk_min")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
