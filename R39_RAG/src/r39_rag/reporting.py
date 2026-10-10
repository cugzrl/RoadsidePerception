"""生成盘点、质量和字段说明"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from r39_rag.config import PROJECT_ROOT, load_schema
from r39_rag.utils.json_stream import load_json


def _size_mb(size: int) -> str:
    return f"{size / 1024 / 1024:.2f} MB"


def _join_unique(values) -> str:
    rendered = []
    for value in dict.fromkeys(values):
        if value is None:
            rendered.append("缺失")
        elif isinstance(value, float) and value.is_integer():
            rendered.append(str(int(value)))
        else:
            rendered.append(str(value))
    return "、".join(rendered)


def write_schema_doc(path: Path) -> None:
    schema = load_schema()
    lines = [
        "# 统一字段说明",
        "",
        f"版本 `{schema['schema_version']}`。历史 JSON 和现有流水线 `result.json` 都先转成同一套观测、局部轨迹和全局轨迹。缺失值写 `null`，不补默认置信度，也不编造检测框。",
        "",
        "同一 `global_track_id` 只在同一个 `sequence_id` 和 `source_version` 内有意义。不同采集段即使编号相同，也不是同一辆车。",
        "",
        "## 关键字段",
        "",
        "### sequence_id",
        "",
        "一次独立视频采集的标识，例如 `r39_20250924_070957`、`r39_20251021_070125`。路口编号 `site_id` 固定为 R39，不能代替序列。目录名看起来连续的 10 月 21 日五段视频仍各自保留序列号，合并关系标记为未核实。",
        "",
        "### source_version",
        "",
        "同一序列上的一种处理结果。`sam3_aligned` 来自对齐后的像素框和相机三维框，`sam3_manual_gid` 来自单相机轨迹并采用人工修正后的全局 ID，`detr_car` 来自只检测 car 的 DETR 结果，`roadside_pipeline` 预留给现有 `result.json`。不同版本不会互相覆盖。",
        "",
        "### frame_index",
        "",
        "抽帧后的帧编号，历史数据从 0 到 1199。`source_frame_index` 是估计的原始视频帧号，当前按 `frame_index * frame_stride` 计算，并假设抽帧起点为 0。它不是文件里直接记录的字段。",
        "",
        "### timestamp_ms",
        "",
        "当前序列内的相对毫秒。历史数据没有逐条时间戳。四路原始视频由 ffprobe 确认为 60 FPS，`frame_cache_meta.json` 的 `frame_stride` 为 3，因此采样帧率是 20，相对时间是 `source_frame_index * 1000 / 60`。`timestamp_estimated` 为 true。无法满足该条件时时间戳保持为空，不会退回到固定的 `frame_index * 1000 / 20`。",
        "",
        "`global_timestamp_ms` 和序列级 `capture_start_time` 保持为空。目录中的日期时间只写入 `capture_time_candidate`，状态为未核实。各序列不共用时间轴。",
        "",
        "### global_track_id",
        "",
        "该版本采用的全局目标编号。人工修正版本同时保留 `original_global_track_id`。修正来源是 `multi_view_global_ids_manual.json`，自动结果来自 `multi_view_global_ids.json`。编号变化、仅一侧存在和人工删除都记在隔离目录，不静默覆盖。",
        "",
        "### bbox_xyxy",
        "",
        "图像像素坐标 `[x1, y1, x2, y2]`。原始字典 `{x0, y0, x1, y1}` 由适配器转换。没有二维框时为 `null`。零面积或越界框保留原值并打上 `invalid_bbox` 或 `bbox_out_of_frame`。",
        "",
        "### world_position",
        "",
        "只有坐标系已经明确时才填写。现有流水线 `result.json` 中的路面点记为 `road_homography_xy`。历史三维结果不写入该字段。",
        "",
        "### coordinate_frame",
        "",
        "- `unknown`：`camera_box_3d` 的长宽中位数约十几米到几十米，高度多为类别假定值，不能当成真实车体尺寸或统一世界坐标",
        "- `unverified_assumed_ground_plane`：SAM3 轨迹里的 `ground_point_3d`，z 被假定为 0，原点、轴向和单位都没有在本轮核实",
        "- `road_homography_xy`：现有工程用单应矩阵得到的路面坐标",
        "- `null`：这条观测没有三维或路面坐标",
        "",
        "完整角点仍留在原始 JSON。统一文件里的 `geometry_3d` 只保留中心、尺寸和地面点，并标记 `corners_stored: false`。",
        "",
        "## 观测字段",
        "",
        "| 字段 | 类型 | 可空 | 含义 |",
        "| --- | --- | --- | --- |",
    ]
    for item in schema["observation_fields"]:
        nullable = "是" if item["nullable"] else "否"
        lines.append(f"| `{item['name']}` | {item['type']} | {nullable} | {item['description']} |")
    lines.extend(["", "## 局部轨迹", "", "局部轨迹由观测按相机和 `local_track_id` 汇总。只统计真实出现的帧，不用起止帧补全中间缺失。", "", "| 字段 | 类型 | 可空 | 含义 |", "| --- | --- | --- | --- |"])
    for item in schema["tracklet_fields"]:
        nullable = "是" if item["nullable"] else "否"
        lines.append(f"| `{item['name']}` | {item['type']} | {nullable} | {item['description']} |")
    lines.extend(["", "## 全局轨迹", "", "全局轨迹按本版本的 `global_track_id` 汇总。没有全局 ID 的检测不会被编成轨迹。", "", "| 字段 | 类型 | 可空 | 含义 |", "| --- | --- | --- | --- |"])
    for item in schema["global_track_fields"]:
        nullable = "是" if item["nullable"] else "否"
        lines.append(f"| `{item['name']}` | {item['type']} | {nullable} | {item['description']} |")
    lines.extend(
        [
            "",
            "## 质量标记",
            "",
            "- `class_unmapped`：原始类别不能对应七类标准名",
            "- `invalid_bbox`：框为空、零面积或左右上下颠倒",
            "- `bbox_out_of_frame`：框超出图像宽高",
            "- `local_id_frame_collision`：同一局部 ID 在同一帧有多套不同的框",
            "- `geometry_size_implausible`：三维长大于 12 米、宽大于 3.5 米，或长宽为 0",
            "- `geometry_range_suspect`：平面坐标绝对值大于 150",
            "- `missing_global_id`：没有可用的全局 ID",
            "- `manual_gid_override`：人工全局 ID 与自动结果不同",
            "- `manual_deleted`：人工结果标记删除",
            "- `frame_gap`：局部轨迹起止帧之间有缺失帧",
            "- `class_conflict`：同一个全局 ID 下出现多个原始类别",
            "",
            "这些标记是数据一致性检查，不是 MOTA、IDF1 或 HOTA。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_inventory(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# R39 历史数据盘点",
        "",
        "盘点对象是两处只读原始目录。采集序列按原始 `sequence_dir` 和视频文件区分，没有因为同为 R39 或全局 ID 相同而合并。",
        "",
        "## 目录规模",
        "",
    ]
    for name, dataset in report["datasets"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"根目录 `{dataset['root']}`。")
        lines.append("")
        lines.append("| 后缀 | 数量 |")
        lines.append("| --- | ---: |")
        for suffix, count in sorted(dataset["extension_counts"].items()):
            lines.append(f"| `{suffix}` | {count} |")
        lines.append("")
        lines.append("| 相对路径 | 大小 | 类型 |")
        lines.append("| --- | ---: | --- |")
        for item in dataset["files"]:
            lines.append(f"| `{item['relative']}` | {_size_mb(item['size'])} | {item['kind']} |")
        lines.append("")

    lines.extend(
        [
            "## 采集序列",
            "",
            "下表的记录数来自像素框或单相机轨迹，还没有做跨版本去重。",
            "",
            "| sequence_id | 目录 | 原始序列目录 | 候选时间 | 记录数 | 帧率结论 |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for sequence in report["sequences"]:
        probes = list(sequence["probes"].values())
        fps = _join_unique(item.get("fps") for item in probes)
        duration = _join_unique(item.get("duration_s") for item in probes)
        caches = list(sequence["frame_caches"].values())
        stride = _join_unique(item.get("frame_stride") for item in caches)
        conclusion = f"源视频 {fps} FPS，时长 {duration} 秒，抽帧步长 {stride}"
        lines.append(
            f"| `{sequence['sequence_id']}` | `{sequence['dataset']}/{sequence['relative_dir']}` | `{sequence['origin_sequence_dir']}` | {sequence['capture_time_candidate']}（未核实） | {sequence['record_count']} | {conclusion} |"
        )
    lines.extend(["", "## 各类别数量", ""])
    for sequence in report["sequences"]:
        lines.append(f"### {sequence['sequence_id']}")
        lines.append("")
        lines.append("| 原始类别 | 数量 |")
        lines.append("| --- | ---: |")
        for name, count in sorted(sequence["class_counts"].items(), key=lambda item: -item[1]):
            lines.append(f"| `{name}` | {count} |")
        lines.append("")
        if sequence["manual"]:
            manual = sequence["manual"]
            lines.append(
                f"自动关联 {manual['auto_tracks']} 条局部轨迹、{manual['auto_global_objects']} 个全局目标；"
                f"人工版本 {manual['manual_tracks']} 条局部轨迹、{manual['manual_global_objects']} 个全局目标。"
            )
            lines.append("")
        if sequence["include_detr"]:
            lines.append("该序列另外包含 DETR car 检测和 ResNet50 特征索引，作为 `detr_car` 版本，不与 SAM3 轨迹合并。")
            lines.append("")

    lines.extend(
        [
            "## 格式",
            "",
            "类型 A 是逐帧像素框，字段包括 `camera_key`、`local_obj_id`、`global_obj_id`、`frame_index`、`class_name`、`bbox_xyxy` 和 `pixel_centroid`。`aligned_bbox_info.json` 把 `pixel_boxes` 与 `camera_boxes` 放在同一个文件，拆分后的像素文件和相机文件与它对照。",
            "",
            "类型 B 是轨迹级全局关联。`multi_view_global_ids_manual.json` 含 `original_global_obj_id`、`is_deleted` 和 `manual_fix`。它没有逐帧框，逐帧框在各相机 `tracks.json`。",
            "",
            "类型 C 是 `camera_box_3d`。对齐文件里的长宽远大于正常车辆，投影文件中的长度在抽样中为 0。两者都保留来源，但不写成已核实世界坐标。",
            "",
            "类型 D 是现有流水线 `result.json`。本次原始目录里没有这种结果，适配器已能读取仓库中的示例结构。",
            "",
            "DETR `detections.json` 只有 car，有置信度和 `bbox_xyxy`，没有轨迹 ID。`detections_coco.json` 使用 COCO 的 xywh，不单独再转一份观测。`feature_metadata.json` 只保存特征索引，不复制 `.npy`。",
            "",
            "## 时间与视频",
            "",
            "原始视频均为 1920x1080、60 FPS、3600 帧、60 秒。抽帧元数据为步长 3、保存 1200 帧，对应 20 FPS 采样。可视化 `multi_view_global_ids.mp4` 是 10 FPS、1200 帧、120 秒，它是播放帧率，不是采样帧率。",
            "",
            "R39_new 说明文字要求按 10 Hz 抽帧，`split_config.yaml` 写了 100 ms 间隔和 2025-04-22，但路径指向其他数据集。这两处与 ffprobe 和 `frame_cache_meta.json` 冲突，时间基准不采用它们。",
            "",
            "四路相机的帧率、步长和帧数一致，因此采样模式一致。跨相机时钟同步没有单独验证，`sync_status` 不是已同步。",
            "",
            "## 版本关系",
            "",
            "`result/20250924_070957` 与 `result/20251021_070024` 各有一套对齐框。20250924 还有 DETR 和 ResNet 特征。`aligned_pixel_bbox.json`、`aligned_camera_bbox.json` 是合并文件的拆分，投影文件是派生结果。",
            "",
            "R39_new 的 0、1、2、3 不是 20251021_070024 的重复处理。它们的原始序列目录分别是 `20251021_070125`、`070226`、`070327`、`070428`，视频文件大小也不同。每个目录里的自动全局 ID、人工全局 ID、文本映射和轨迹框属于同一段视频的不同版本。",
            "",
            "## 未核实关系",
            "",
        ]
    )
    for group in report["unverified_groups"]:
        joined = "、".join(f"`{item}`" for item in group["sequence_ids"])
        lines.append(f"- {group['id']}（{group['status']}）：{joined}。{group['note']}")
    lines.extend(
        [
            "",
            "`R39_new/multi_view_global_ids.mp4` 无法被 ffprobe 读取。`result/20251021_070024/raw_video` 下还有一份与子目录同名的视频拷贝，大小相同但不是同一 inode，清洗使用子目录中的文件。",
            "",
            "掩码 `.npz` 和抽帧图片数量很大，本轮只统计数量，不把掩码或图片转成观测。未观测到目标，和原始文件里没有标注，不是同一件事；统一文件只记录原始文件中存在的框。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_quality(path: Path) -> None:
    root = PROJECT_ROOT / "data" / "normalized" / "sequences"
    metas = [load_json(item) for item in sorted(root.glob("*/metadata.json"))]
    lines = [
        "# 数据质量报告",
        "",
        "本报告只描述格式转换和一致性检查。它不评估检测或跟踪准确率，也不计算 MOTA、IDF1、HOTA。",
        "",
        "完全相同的记录会去掉重复并计数。同一主键内容不同的记录全部保留，并写入 `data/quarantine`。不同处理版本分目录保存。",
        "",
        "## 总览",
        "",
        "| 序列 | 版本 | 输入 | 写出 | 完全重复 | 冲突条数 | 无效框 | 越界框 | 未映射类别 | 三维尺寸异常 | 坐标范围异常 | 缺少时间 | 缺少全局 ID |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for meta in metas:
        for name, item in meta["versions"].items():
            lines.append(
                "| {seq} | {ver} | {inp} | {out} | {dup} | {conf} | {bad} | {outb} | {unmap} | {geo} | {rng} | {time} | {gid} |".format(
                    seq=meta["sequence_id"],
                    ver=name,
                    inp=item.get("input_records"),
                    out=item.get("written_observations"),
                    dup=item.get("exact_duplicates_removed"),
                    conf=item.get("conflict_records"),
                    bad=item.get("invalid_bbox"),
                    outb=item.get("bbox_out_of_frame"),
                    unmap=item.get("class_unmapped"),
                    geo=item.get("geometry_size_implausible", 0),
                    rng=item.get("geometry_range_suspect", 0),
                    time=item.get("missing_timestamp"),
                    gid=item.get("missing_global_id"),
                )
            )
    lines.extend(["", "## 分序列说明", ""])
    for meta in metas:
        lines.append(f"### {meta['sequence_id']}")
        lines.append("")
        lines.append(
            f"源帧率 {meta['source_fps']}，采样帧率 {meta['sample_fps']}，抽帧步长 {meta['frame_stride']}，抽帧数 {meta['frame_count']}。"
            f"同步状态 `{meta['sync_status']}`。绝对采集时间为空，候选时间 {meta['capture_time_candidate']} 未核实。"
        )
        lines.append("")
        for note in meta["notes"]:
            lines.append(f"- {note}")
        lines.append("")
        for name, item in meta["versions"].items():
            lines.append(f"#### {name}")
            lines.append("")
            lines.append(item.get("coordinate_status") or "")
            lines.append("")
            if item.get("class_counts_raw"):
                pairs = "，".join(f"{key} {value}" for key, value in item["class_counts_raw"].items())
                lines.append(f"原始类别：{pairs}。")
                lines.append("")
            siblings = item.get("sibling_files") or {}
            for label, check in siblings.items():
                lines.append(
                    f"`{label}` 对照 {check['compared']} 条，差异 {check['mismatches']}，"
                    f"{'内容与主文件一致，不重复写出' if check['duplicate_of_reference'] else '存在差异，详见隔离示例'}。"
                )
            if siblings:
                lines.append("")
            if "gid_changed" in item:
                lines.append(
                    f"人工修正相对自动结果改变了 {item['gid_changed']} 条局部轨迹，"
                    f"只在自动结果中的 {item['only_in_auto']} 条，只在人工结果中的 {item['only_in_manual']} 条，"
                    f"人工删除 {item['manual_deleted_tracks']} 条。文本映射差异 {item['text_map_mismatches']} 条，"
                    f"轨迹摘要与逐帧观测差异 {item['summary_mismatches']} 条。"
                )
                lines.append("")
            if item.get("coco"):
                coco = item["coco"]
                lines.append(f"COCO 导出 {coco['coco_records']} 条，检测头记录 {coco['detections_header_count']} 条。{coco['reason']}")
                lines.append("")
            if item.get("track_layer"):
                lines.append(item["track_layer"])
                lines.append("")
            examples = item.get("examples") or {}
            if examples:
                lines.append("示例：")
                lines.append("")
                for kind, rows in examples.items():
                    lines.append(f"- `{kind}`：`{rows[0]}`")
                lines.append("")
    lines.extend(
        [
            "## 坐标系",
            "",
            "`sam3_aligned` 的三维框来自 `camera_boxes`。尺寸检查使用长度大于 12 米、宽度大于 3.5 米或长宽为 0 作为明显异常，只打标记，不删除，也不回写二维框。",
            "",
            "`sam3_manual_gid` 的地面点来自轨迹文件。部分点的平面坐标绝对值超过 150，标记为 `geometry_range_suspect`。现有工程 `configs/cameras/r39` 的单应矩阵没有拿来重算这些点。",
            "",
            "因此本轮没有可以确认已经处在统一世界坐标系中的历史三维结果。",
            "",
            "## 隔离文件",
            "",
            "`data/quarantine/<sequence_id>/<source_version>/conflicts.jsonl` 保存同键冲突、版本差异和摘要不一致。`anomalies.jsonl` 保存无效框和越界框。原始文件没有被修改。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
