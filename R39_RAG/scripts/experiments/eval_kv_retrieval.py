"""在留出序列上比较匀速、典型轨迹和Key-Value未来位置

参考轨迹来自质量筛选后的历史结构化结果，不是人工标注真值
Oracle只用未来误差挑选已经检索出的整条候选，在线方法不看未来
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "build"))

from build_kv_knowledge import build  # noqa: E402
from r39_rag.config import load_knowledge  # noqa: E402
from r39_rag.knowledge.kv_motion import KVIndex  # noqa: E402
from r39_rag.knowledge.online import LaneIndex  # noqa: E402
from r39_rag.knowledge.prediction_eval import (  # noqa: E402
    HORIZONS,
    KIND_NAME,
    assemble_forecasts,
    assert_library_split,
    cv_from_key,
    distance_bin,
    paired_reduction,
    point_errors,
    prototype_forecast,
    summarize,
    track_macro,
)


BANK_NAME = "kv_motion_class"
METHODS = (
    ("cv", "CV"),
    ("prototype", "Prototype"),
    ("kv_top1", "KV-Top1"),
    ("kv_oracle", "KV-Top5 Oracle"),
    ("kv_min_fde", "minFDE@5"),
    ("kv_cv", "KV+CV"),
)
ONLINE = {"cv", "prototype", "kv_top1", "kv_cv"}


def _round(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _load_bank(root: Path) -> tuple[KVIndex, dict]:
    npz_path = root / "knowledge_base" / f"{BANK_NAME}.npz"
    sqlite_path = root / "knowledge_base" / f"{BANK_NAME}.sqlite"
    if not npz_path.is_file() or not sqlite_path.is_file():
        raise FileNotFoundError(f"缺少知识库 {npz_path}")
    index = KVIndex.load(npz_path, sqlite_path)
    sequences = sorted(set(index.sequence_id))
    meta_sequences = list(index.meta.get("build_sequences") or sequences)
    if sorted(meta_sequences) != sequences:
        raise RuntimeError(f"知识库元数据和样本序列不一致 {meta_sequences} {sequences}")
    return index, {"npz": str(npz_path), "sqlite": str(sqlite_path), "meta": index.meta, "sequences": sequences}


def _collect(rows: list[dict], index: KVIndex, prototypes: np.ndarray, center: np.ndarray) -> dict[str, np.ndarray | list]:
    count = len(rows)
    width = len(HORIZONS)
    packed = {key: np.full((count, width), np.nan, dtype=np.float64) for key, _name in METHODS}
    hit = np.zeros(count, dtype=bool)
    reasons: dict[str, int] = {}
    kinds = []
    classes = []
    sequences = []
    tracks = []
    bins = []
    for index_row, row in enumerate(rows):
        truth = np.asarray(row["value_xy"], dtype=np.float64)
        cv_xy = cv_from_key(row["key_xy"], float(index.meta.get("history_seconds", 1.0)))
        prototype_xy = prototype_forecast(row["key_xy"], prototypes, float(row["speed"]))
        found = index.query(row["key_xy"])
        reliable = bool(found.get("reliable"))
        futures = [np.asarray(item["future_xy"], dtype=np.float64) for item in found.get("matches") or []]
        if not reliable:
            reasons[str(found.get("reason") or "未命中")] = reasons.get(str(found.get("reason") or "未命中"), 0) + 1
        forecasts = assemble_forecasts(cv_xy, prototype_xy, futures, reliable, truth)
        for key, _name in METHODS:
            packed[key][index_row] = point_errors(forecasts[key], truth)
        hit[index_row] = bool(forecasts["kv_hit"])
        kinds.append(str(row.get("kind") or "other"))
        classes.append(str(row.get("class_name") or "未知"))
        sequences.append(str(row["sequence_id"]))
        tracks.append(f"{row['sequence_id']}|{row['camera_id']}|{row['local_track_id']}")
        end = np.asarray(row["key_xy"][-1], dtype=np.float64)
        bins.append(distance_bin(float(np.linalg.norm(end - center))))
    return {
        "errors": packed,
        "hit": hit,
        "reasons": reasons,
        "kind": kinds,
        "class_name": classes,
        "sequence": sequences,
        "track": tracks,
        "distance_bin": bins,
    }


def _mask_finite(errors: np.ndarray) -> np.ndarray:
    if len(errors) == 0:
        return np.zeros(0, dtype=bool)
    return np.isfinite(errors).all(axis=1)


def _method_block(errors: np.ndarray, track_ids: list[str], baseline: np.ndarray | None) -> dict:
    summary = summarize(errors)
    summary["track_macro"] = track_macro(errors, track_ids)
    if baseline is None:
        summary["ade_drop_vs_cv_pct"] = None
        summary["fde_drop_vs_cv_pct"] = None
        summary["paired_cv_ade_m"] = None
        summary["paired_cv_fde_m"] = None
        summary["paired_count"] = summary["count"]
        return summary
    summary.update(paired_reduction(errors, baseline))
    return summary


def _subset(collected: dict, mask: np.ndarray) -> dict[str, dict]:
    tracks = [track for track, keep in zip(collected["track"], mask) if keep]
    baseline = collected["errors"]["cv"][mask]
    return {
        key: _method_block(collected["errors"][key][mask], tracks, None if key == "cv" else baseline)
        for key, _name in METHODS
    }


def _group_table(collected: dict, labels: list[str], mask: np.ndarray, methods: tuple[str, ...]) -> dict[str, dict]:
    table = {}
    for label in sorted(set(labels)):
        chosen = mask & np.asarray([item == label for item in labels])
        if not chosen.any():
            continue
        tracks = [track for track, keep in zip(collected["track"], chosen) if keep]
        baseline = collected["errors"]["cv"][chosen]
        table[label] = {
            key: _method_block(collected["errors"][key][chosen], tracks, None if key == "cv" else baseline)
            for key in methods
        }
        table[label]["windows"] = int(chosen.sum())
        table[label]["tracks"] = len(set(tracks))
    return table


def _coverage(mask: np.ndarray, errors: np.ndarray) -> float | None:
    if len(mask) == 0:
        return None
    finite = _mask_finite(errors)
    return float(np.mean(finite[mask])) if mask.any() else 0.0


def build_report_numbers(config: dict, bank: dict, reference: dict, prototypes_path: Path, center: np.ndarray) -> dict:
    rows = reference["rows"]
    index = bank["index"]
    assert_library_split(bank["sequences"], list(config["holdout_sequences"]), list(reference["meta"]["query_sequences"]))
    collected = _collect(rows, index, np.load(prototypes_path)["prototypes"], center)
    total = len(rows)
    hit = collected["hit"]
    proto_ok = _mask_finite(collected["errors"]["prototype"])
    common = hit & proto_ok
    all_input = np.ones(total, dtype=bool)
    numbers = {
        "note": "误差相对于留出序列中通过同一套过滤的参考轨迹，不是人工标注真值",
        "library": {
            "name": BANK_NAME,
            "npz": bank["npz"],
            "sqlite": bank["sqlite"],
            "build_sequences": bank["sequences"],
            "holdout_sequences": list(config["holdout_sequences"]),
            "history_seconds": bank["meta"].get("history_seconds"),
            "future_seconds": bank["meta"].get("future_seconds"),
            "top_k": bank["meta"].get("top_k"),
            "min_score": bank["meta"].get("min_score"),
            "spatial_radius_m": bank["meta"].get("spatial_radius_m"),
            "window_count": bank["meta"].get("window_count"),
            "independent_tracks": bank["meta"].get("independent_tracks"),
            "class_count": bank["meta"].get("class_count"),
            "not_used": str(Path(bank["npz"]).with_name("kv_motion.npz")),
            "not_used_reason": "与kv_motion_class的运动数组一致，但没有类别字段，本实验不加载",
        },
        "prototypes": str(prototypes_path),
        "horizons_s": [float(item) for item in HORIZONS],
        "intersection_center_m": [float(item) for item in center],
        "filter_counts": reference.get("filter_counts") or {},
        "reference_meta": reference["meta"],
        "samples": {
            "windows": total,
            "independent_tracks": int(reference["meta"]["independent_tracks"]),
            "kv_hit": int(hit.sum()),
            "kv_hit_ratio": 0.0 if total == 0 else float(hit.mean()),
            "kv_fallback": int((~hit).sum()),
            "kv_fallback_ratio": 0.0 if total == 0 else float((~hit).mean()),
            "kv_miss_reasons": collected["reasons"],
            "prototype_valid": int(proto_ok.sum()),
            "prototype_coverage": 0.0 if total == 0 else float(proto_ok.mean()),
            "common_windows": int(common.sum()),
            "common_tracks": len({track for track, keep in zip(collected["track"], common) if keep}),
        },
        "common": _subset(collected, common),
        "all_input": _subset(collected, all_input),
        "by_kind_common": _group_table(collected, collected["kind"], common, tuple(key for key, _name in METHODS)),
        "by_kind_all": _group_table(collected, collected["kind"], all_input, ("cv", "kv_top1", "kv_cv")),
        "by_sequence_all": _group_table(collected, collected["sequence"], all_input, ("cv", "kv_top1", "kv_oracle", "kv_cv")),
        "by_class_all": _group_table(collected, collected["class_name"], all_input, ("cv", "kv_cv")),
        "by_distance_all": _group_table(collected, collected["distance_bin"], all_input, ("cv", "kv_cv")),
    }
    for split_name, mask in (("common", common), ("all_input", all_input)):
        for key, _name in METHODS:
            numbers[split_name][key]["coverage"] = _coverage(mask, collected["errors"][key])
    return numbers


def _cell(summary: dict | None, field: str, digits: int = 3) -> str:
    if not summary:
        return ""
    value = summary.get(field)
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def write_csv(path: Path, numbers: dict) -> None:
    rows = []
    for split in ("common", "all_input"):
        for key, name in METHODS:
            item = numbers[split][key]
            rows.append(
                {
                    "split": split,
                    "method": name,
                    "online": "yes" if key in ONLINE else "no",
                    "samples": item["count"],
                    "coverage": _round(item.get("coverage"), 4),
                    "ade_m": _round(item["ade_m"]),
                    "fde_m": _round(item["fde_m"]),
                    "error_0.5s_m": _round(item["error_0.5s_m"]),
                    "error_1.0s_m": _round(item["error_1.0s_m"]),
                    "error_1.5s_m": _round(item["error_1.5s_m"]),
                    "error_2.0s_m": _round(item["error_2.0s_m"]),
                    "ade_drop_vs_cv_pct": _round(item["ade_drop_vs_cv_pct"], 2),
                    "fde_drop_vs_cv_pct": _round(item["fde_drop_vs_cv_pct"], 2),
                    "track_macro_ade_m": _round(item["track_macro"]["ade_m"]),
                    "track_macro_fde_m": _round(item["track_macro"]["fde_m"]),
                    "tracks": item["track_macro"]["tracks"],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, numbers: dict) -> None:
    library = numbers["library"]
    samples = numbers["samples"]
    filters = numbers["filter_counts"]
    lines = [
        "# R39留出序列轨迹预测评估",
        "",
        "本结果是参考轨迹预测评估，不是标准人工真值评估。留出序列没有独立人工标注，未来位置来自同一套投影和质量筛选后的历史结构化轨迹。",
        "",
        "## 数据划分",
        "",
        f"- 检索库只使用 `{library['name']}`：`{library['npz']}`",
        f"- 建库序列：{', '.join(library['build_sequences'])}",
        f"- 留出序列：{', '.join(library['holdout_sequences'])}",
        f"- 库内窗口 {library.get('window_count')}，独立轨迹 {library.get('independent_tracks')}",
        f"- `{Path(library['not_used']).name}` 的运动数组与本次使用的库一致，但没有类别，因此没有加载",
        f"- 典型轨迹使用已有文件 `{numbers['prototypes']}`，构建代码只读建库序列，本次不重建",
        f"- 历史输入 {library.get('history_seconds')} 秒，预测 0.5、1.0、1.5、2.0 秒",
        f"- KV检索 top-{library.get('top_k')}，最低相似度 {library.get('min_score')}，空间半径 {library.get('spatial_radius_m')} 米",
        "",
        "## 样本",
        "",
        f"- 原始观测 {filters.get('observations', '未统计')}",
        f"- 缺局部编号或时间 {filters.get('no_local_or_time', 0)}，身份被拆开 {filters.get('reassigned_id', 0)}，投影不可用 {sum(filters.get(key, 0) for key in filters if str(key).startswith('projection_'))}",
        f"- 超出范围 {filters.get('beyond_max_abs', 0)}，离车道过远 {filters.get('far_from_lane', 0)}，缺坐标 {filters.get('missing_xy', 0)}",
        f"- 连续片段 {filters.get('segments', 0)}，其中时长不足完整3秒窗口 {filters.get('segments_without_full_window', 0)}",
        f"- 进入评估的窗口 {samples['windows']}，独立轨迹 {samples['independent_tracks']}",
        f"- KV命中 {samples['kv_hit']}（{samples['kv_hit_ratio']:.1%}），回退到CV {samples['kv_fallback']}（{samples['kv_fallback_ratio']:.1%}）",
        f"- 典型轨迹有效 {samples['prototype_valid']}（覆盖率 {samples['prototype_coverage']:.1%}）",
        f"- 各在线方法都有效的共同样本 {samples['common_windows']} 个窗口、{samples['common_tracks']} 条轨迹",
        "",
        "KV未命中原因：",
        "",
    ]
    if samples["kv_miss_reasons"]:
        for reason, count in sorted(samples["kv_miss_reasons"].items(), key=lambda item: -item[1]):
            lines.append(f"- {reason}：{count}")
    else:
        lines.append("- 无")
    lines.extend(
        [
            "",
            "运动类型由参考轨迹的转角得到，只用于事后分组，不参与检索或候选选择。",
            "",
            "## 共同有效样本",
            "",
            "只保留CV、典型轨迹和KV-Top1都给出完整4个时刻的窗口。KV-Top5 Oracle和minFDE@5是理想上界，不能在线使用。Oracle按整条候选的ADE选择，minFDE@5按整条候选的2秒误差选择，两者都可以对应到同一条历史未来轨迹。",
            "",
            "| 方法 | 样本 | ADE(m) | FDE(m) | 0.5s | 1.0s | 1.5s | 2.0s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, name in METHODS:
        item = numbers["common"][key]
        lines.append(
            f"| {name} | {item['count']} | {_cell(item, 'ade_m')} | {_cell(item, 'fde_m')} | "
            f"{_cell(item, 'error_0.5s_m')} | {_cell(item, 'error_1.0s_m')} | {_cell(item, 'error_1.5s_m')} | {_cell(item, 'error_2.0s_m')} |"
        )
    lines.extend(["", "相对CV的误差变化，正数表示误差下降：", ""])
    for key, name in METHODS:
        if key == "cv":
            continue
        item = numbers["common"][key]
        ade = item.get("ade_drop_vs_cv_pct")
        fde = item.get("fde_drop_vs_cv_pct")
        ade_text = "不适用" if ade is None else f"{ade:+.1f}%"
        fde_text = "不适用" if fde is None else f"{fde:+.1f}%"
        lines.append(f"- {name}：ADE {ade_text}，FDE {fde_text}")
    lines.extend(
        [
            "",
            "## 全部有效输入",
            "",
            "KV未命中时，KV+CV使用同一条历史的匀速预测。这一组覆盖全部质量筛选后的窗口，避免只报告容易检索成功的样本。",
            "",
            "| 方法 | 样本 | 覆盖率 | ADE(m) | FDE(m) | 0.5s | 1.0s | 1.5s | 2.0s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, name in METHODS:
        item = numbers["all_input"][key]
        lines.append(
            f"| {name} | {item['count']} | {_cell(item, 'coverage', 3)} | {_cell(item, 'ade_m')} | {_cell(item, 'fde_m')} | "
            f"{_cell(item, 'error_0.5s_m')} | {_cell(item, 'error_1.0s_m')} | {_cell(item, 'error_1.5s_m')} | {_cell(item, 'error_2.0s_m')} |"
        )
    fused = numbers["all_input"]["kv_cv"]
    lines.extend(
        [
            "",
            "Prototype、KV-Top1、Oracle和minFDE@5的平均值只包含该方法有完整预测的样本。相对CV的百分比使用同一批样本上的CV，不和全部窗口的CV直接相减。KV+CV覆盖全部窗口，才与全部CV比较。",
            "",
            f"全部窗口上，KV+CV相对CV的ADE变化 {fused['ade_drop_vs_cv_pct']:+.1f}%，FDE变化 {fused['fde_drop_vs_cv_pct']:+.1f}%。"
            if fused["ade_drop_vs_cv_pct"] is not None and fused["fde_drop_vs_cv_pct"] is not None
            else "",
            "同一轨迹先平均再跨轨迹平均后，CV的ADE为 "
            f"{numbers['all_input']['cv']['track_macro']['ade_m']:.3f} 米，KV+CV为 "
            f"{numbers['all_input']['kv_cv']['track_macro']['ade_m']:.3f} 米。"
            "窗口数远多于轨迹数，窗口平均的降幅不能直接当成独立样本上的提升。",
            "",
            "## 运动类型",
            "",
            "下表是全部有效窗口上的CV和KV+CV。",
            "",
            "| 类型 | 窗口 | 轨迹 | CV ADE | CV FDE | KV+CV ADE | KV+CV FDE | ADE变化 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for kind, block in numbers["by_kind_all"].items():
        title = KIND_NAME.get(kind, kind)
        change = block["kv_cv"].get("ade_drop_vs_cv_pct")
        change_text = "" if change is None else f"{change:+.1f}%"
        lines.append(
            f"| {title} | {block['windows']} | {block['tracks']} | {_cell(block['cv'], 'ade_m')} | {_cell(block['cv'], 'fde_m')} | "
            f"{_cell(block['kv_cv'], 'ade_m')} | {_cell(block['kv_cv'], 'fde_m')} | {change_text} |"
        )
    lines.extend(["", "## 留出序列", "", "| 序列 | 窗口 | CV ADE | KV+CV ADE | KV命中上的Oracle ADE |", "| --- | ---: | ---: | ---: | ---: |"])
    for sequence, block in numbers["by_sequence_all"].items():
        lines.append(
            f"| {sequence} | {block['windows']} | {_cell(block['cv'], 'ade_m')} | {_cell(block['kv_cv'], 'ade_m')} | {_cell(block['kv_oracle'], 'ade_m')} |"
        )
    lines.extend(["", "## 距离与类别", "", "距离是查询时刻到路口中心的平面距离。样本少于30的分组只作参考。", ""])
    lines.append("| 分组 | 窗口 | CV ADE | KV+CV ADE |")
    lines.append("| --- | ---: | ---: | ---: |")
    for title, block in {**{f"距离 {key}": value for key, value in numbers["by_distance_all"].items()}, **numbers["by_class_all"]}.items():
        note = "" if block["windows"] >= 30 else "（样本少）"
        lines.append(f"| {title}{note} | {block['windows']} | {_cell(block['cv'], 'ade_m')} | {_cell(block['kv_cv'], 'ade_m')} |")
    lines.extend(
        [
            "",
            "## 局限与后续工作",
            "",
            "- 参考轨迹仍可能包含历史关联错误，误差不能写成官方跟踪精度。",
            "- 相邻0.5秒窗口高度相关，窗口数会大于独立轨迹数。",
            "- KV-Top5 Oracle和minFDE@5使用了未来误差来选候选，只说明检索池里存在更好的轨迹，不是当前在线算法。",
            "- 典型轨迹在路口转向和短续接上经常给不出完整2秒，共同样本会比全部窗口少。",
            "- 检索没有按类别过滤。行人和车辆共用同一套运动相似度，密集目标仍容易配到邻近轨迹。",
            "- 后续可以在不使用未来信息的前提下，用类别、行驶方向和检索置信度决定何时相信KV、何时保持匀速。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_knowledge()
    bank_index, bank_info = _load_bank(ROOT)
    bank_info["index"] = bank_index
    reference = build(config, sequence_ids=list(config["holdout_sequences"]), allow_holdout=True)
    query_sequences = list(reference["meta"]["query_sequences"])
    if set(query_sequences) != set(config["holdout_sequences"]):
        raise RuntimeError("留出查询序列和配置不一致")
    if set(bank_info["sequences"]) & set(config["holdout_sequences"]):
        raise RuntimeError("检索库混入了留出序列")
    prototypes_path = ROOT / "knowledge_base" / "knowledge_arrays.npz"
    center = LaneIndex.load().center
    numbers = build_report_numbers(config, bank_info, reference, prototypes_path, center)
    out = ROOT / "reports" / "prediction_eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out / "comparison.csv", numbers)
    write_markdown(out / "prediction_report.md", numbers)
    samples = numbers["samples"]
    common = numbers["common"]
    fused = numbers["all_input"]["kv_cv"]
    print(
        json.dumps(
            {
                "windows": samples["windows"],
                "tracks": samples["independent_tracks"],
                "kv_hit_ratio": round(samples["kv_hit_ratio"], 4),
                "common": samples["common_windows"],
                "common_ade": {name: numbers["common"][key]["ade_m"] for key, name in METHODS},
                "all_cv_ade": numbers["all_input"]["cv"]["ade_m"],
                "all_kvcv_ade": fused["ade_m"],
                "all_kvcv_ade_drop_pct": fused["ade_drop_vs_cv_pct"],
                "common_defined": common["cv"]["count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
