"""同相机短时关联的歧义处理

只决定哪些边不该保留 不改最终 JSON 里的 ID
"""

from __future__ import annotations

import numpy as np


def competition_clear(
    weight: np.ndarray,
    iou: np.ndarray,
    margin: float,
    iou_decide: float,
    weak_iou: float = 0.10,
) -> np.ndarray:
    """一对多且边权接近时留下 IoU 更好的一个 只有两边都不像同一目标才一起清掉"""
    clear = np.zeros(weight.shape, dtype=bool)
    valid = np.isfinite(weight) & (weight > 0) & np.isfinite(iou)

    def _compete(indexes: np.ndarray, values: np.ndarray, overlaps: np.ndarray) -> list[int]:
        if len(indexes) < 2:
            return []
        order = indexes[np.argsort(-values[indexes])]
        best = int(order[0])
        second = int(order[1])
        if float(values[best] - values[second]) >= margin:
            return []
        gap = float(overlaps[best] - overlaps[second])
        if gap >= iou_decide:
            return [int(item) for item in order[1:]]
        if -gap >= iou_decide:
            return [int(item) for item in order if int(item) != second]
        # IoU 也接近时留给 Multicut 主动拆边会把正常轨迹打成碎片
        return []

    for col in range(weight.shape[1]):
        rows = np.flatnonzero(valid[:, col])
        for row in _compete(rows, weight[:, col], iou[:, col]):
            clear[row, col] = True
    for row in range(weight.shape[0]):
        cols = np.flatnonzero(valid[row])
        for col in _compete(cols, weight[row], iou[row]):
            clear[row, col] = True
    return clear


def mutual_keep(score: np.ndarray, margin: float) -> np.ndarray:
    """行是当前检测 列是已有轨迹 只给双方都明显第一的一对加分"""
    keep = np.zeros(score.shape, dtype=bool)
    if score.size == 0:
        return keep
    for row in range(score.shape[0]):
        order = np.argsort(-score[row])
        best = int(order[0])
        if float(score[row, best]) < 0:
            continue
        if score.shape[1] > 1 and float(score[row, best] - score[row, int(order[1])]) < margin:
            continue
        column = np.argsort(-score[:, best])
        if int(column[0]) != row:
            continue
        if score.shape[0] > 1 and float(score[int(column[0]), best] - score[int(column[1]), best]) < margin:
            continue
        keep[row, best] = True
    return keep
