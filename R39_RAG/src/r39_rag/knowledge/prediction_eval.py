"""留出序列上的轨迹预测指标

在线方法只看已经发生的1秒轨迹
Oracle用参考未来挑选一条完整候选 不能把不同车的时刻拼在一起
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from r39_rag.knowledge.kv_motion import VALUE_OFFSETS, constant_velocity
from r39_rag.knowledge.online import match_prefix


HORIZONS = np.asarray(VALUE_OFFSETS, dtype=np.float64)
KIND_NAME = {"straight": "直行", "left": "左转", "right": "右转", "other": "其他"}


def assert_library_split(bank_sequences: list[str], holdout_sequences: list[str], query_sequences: list[str]) -> None:
    bank = set(bank_sequences)
    holdout = set(holdout_sequences)
    leaked = sorted(bank & holdout)
    if leaked:
        raise RuntimeError(f"检索库包含留出序列 {leaked}")
    outside = sorted(set(query_sequences) - holdout)
    if outside:
        raise RuntimeError(f"查询序列超出留出划分 {outside}")
    if not query_sequences or set(query_sequences) != holdout:
        raise RuntimeError("查询序列必须正好是全部留出序列")


def prototype_forecast(
    key_xy: np.ndarray,
    prototypes: np.ndarray,
    speed: float,
    horizons: np.ndarray = HORIZONS,
    score_min: float = 0.45,
) -> np.ndarray | None:
    """沿已匹配的典型轨迹按历史速度往前走 不读取未来位置"""
    found = match_prefix(key_xy, prototypes, top_k=1)
    if not found or found[0]["score"] < score_min or not found[0]["continuation"]:
        return None
    continuation = np.asarray(found[0]["continuation"], dtype=np.float64)
    chain = np.vstack([key_xy[-1], continuation])
    step = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    cumulative = np.cumsum(step)
    if speed < 0.5 or cumulative[-1] < 0.5:
        return None
    predicted = []
    for seconds in horizons:
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


def point_errors(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(predicted, dtype=np.float64) - np.asarray(truth, dtype=np.float64), axis=1)


def _finite_rows(stack: np.ndarray) -> np.ndarray:
    return np.isfinite(stack).all(axis=1)


def select_by_ade(stack: np.ndarray) -> int | None:
    """用整条候选的平均误差选一条 不按时刻分别取最小"""
    stack = np.asarray(stack, dtype=np.float64)
    if stack.ndim != 2 or stack.shape[0] == 0:
        return None
    finite = _finite_rows(stack)
    if not finite.any():
        return None
    score = np.where(finite, stack.mean(axis=1), np.inf)
    return int(np.argmin(score))


def select_by_fde(stack: np.ndarray) -> int | None:
    stack = np.asarray(stack, dtype=np.float64)
    if stack.ndim != 2 or stack.shape[0] == 0:
        return None
    finite = _finite_rows(stack)
    if not finite.any():
        return None
    score = np.where(finite, stack[:, -1], np.inf)
    return int(np.argmin(score))


def assemble_forecasts(
    cv_xy: np.ndarray,
    prototype_xy: np.ndarray | None,
    kv_futures: list[np.ndarray],
    reliable: bool,
    truth: np.ndarray,
) -> dict[str, Any]:
    """CV、典型轨迹和KV-Top1不读取truth Oracle才用truth在已检索候选中选一条"""
    width = len(HORIZONS)
    missing = np.full((width, 2), np.nan, dtype=np.float64)
    cv_xy = np.asarray(cv_xy, dtype=np.float64)
    top1 = np.asarray(kv_futures[0], dtype=np.float64) if reliable and kv_futures else missing.copy()
    fused = top1.copy() if reliable and kv_futures else cv_xy.copy()
    oracle = missing.copy()
    min_fde_xy = missing.copy()
    if reliable and kv_futures:
        stack = np.stack([point_errors(item, truth) for item in kv_futures])
        ade_index = select_by_ade(stack)
        fde_index = select_by_fde(stack)
        if ade_index is not None:
            oracle = np.asarray(kv_futures[ade_index], dtype=np.float64)
        if fde_index is not None:
            min_fde_xy = np.asarray(kv_futures[fde_index], dtype=np.float64)
    prototype = missing.copy() if prototype_xy is None else np.asarray(prototype_xy, dtype=np.float64)
    return {
        "cv": cv_xy,
        "prototype": prototype,
        "kv_top1": top1,
        "kv_oracle": oracle,
        "kv_min_fde": min_fde_xy,
        "kv_cv": fused,
        "kv_hit": bool(reliable and kv_futures),
    }


def cv_from_key(key_xy: np.ndarray, history_s: float = 1.0) -> np.ndarray:
    return constant_velocity(np.asarray(key_xy, dtype=np.float64), HORIZONS, history_s)


def sample_ade(errors: np.ndarray) -> float:
    if not np.isfinite(errors).all():
        return float("nan")
    return float(np.mean(errors))


def sample_fde(errors: np.ndarray) -> float:
    if not np.isfinite(errors).all():
        return float("nan")
    return float(errors[-1])


def paired_reduction(method_errors: np.ndarray, cv_errors: np.ndarray) -> dict[str, Any]:
    """相对CV的降幅只在两种方法都有效的同一批样本上计算"""
    method_errors = np.asarray(method_errors, dtype=np.float64)
    cv_errors = np.asarray(cv_errors, dtype=np.float64)
    finite = np.isfinite(method_errors).all(axis=1) & np.isfinite(cv_errors).all(axis=1)
    method = summarize(method_errors[finite])
    baseline = summarize(cv_errors[finite])
    return {
        "ade_drop_vs_cv_pct": drop_percent(baseline["ade_m"], method["ade_m"]),
        "fde_drop_vs_cv_pct": drop_percent(baseline["fde_m"], method["fde_m"]),
        "paired_cv_ade_m": baseline["ade_m"],
        "paired_cv_fde_m": baseline["fde_m"],
        "paired_count": method["count"],
    }


def drop_percent(baseline: float | None, value: float | None) -> float | None:
    if baseline is None or value is None or not np.isfinite(baseline) or baseline == 0 or not np.isfinite(value):
        return None
    return float((baseline - value) / baseline * 100.0)


def _mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None
    return float(np.mean(finite))


def summarize(errors: np.ndarray) -> dict[str, Any]:
    """errors形状是样本数乘4个时刻 无效样本整行保持nan"""
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0:
        errors = np.zeros((0, len(HORIZONS)), dtype=np.float64)
    complete = np.isfinite(errors).all(axis=1) if len(errors) else np.zeros(0, dtype=bool)
    used = errors[complete]
    summary: dict[str, Any] = {"count": int(complete.sum()), "input_count": int(len(errors))}
    for index, seconds in enumerate(HORIZONS):
        column = errors[:, index] if len(errors) else np.zeros(0)
        summary[f"error_{float(seconds):.1f}s_m"] = _mean(column)
    summary["ade_m"] = None if len(used) == 0 else float(used.mean())
    summary["fde_m"] = None if len(used) == 0 else float(used[:, -1].mean())
    return summary


def track_macro(errors: np.ndarray, track_ids: list[str]) -> dict[str, Any]:
    """先对同一轨迹的窗口取平均 再对各轨迹取平均"""
    errors = np.asarray(errors, dtype=np.float64)
    buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    for index, track_id in enumerate(track_ids):
        if np.isfinite(errors[index]).all():
            buckets[track_id].append(errors[index])
    if not buckets:
        return {"tracks": 0, "ade_m": None, "fde_m": None}
    ades = [float(np.mean(np.stack(items))) for items in buckets.values()]
    fdes = [float(np.mean(np.stack(items)[:, -1])) for items in buckets.values()]
    return {"tracks": len(buckets), "ade_m": float(np.mean(ades)), "fde_m": float(np.mean(fdes))}


def distance_bin(distance_m: float) -> str:
    if distance_m < 25.0:
        return "0-25m"
    if distance_m < 55.0:
        return "25-55m"
    return "55m以上"
