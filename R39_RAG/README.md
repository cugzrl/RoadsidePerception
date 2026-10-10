# R39_RAG

苏州 R39 路口的结构化历史数据工程，用于后续固定场景时空知识增强。当前只完成数据接入、清洗和统一

## 当前范围

已完成：

- 盘点 `/home/kemove/devdata1/R2/result` 与 `/home/kemove/devdata1/R2/R39_new`
- 用软链接接入原始数据，不复制、不修改
- 将多版本JSON转为统一观测、局部轨迹和全局轨迹
- 保留原始路径、原始类别、原始全局ID和未核实坐标

未做：

- 空间、轨迹、外观知识库
- 向量数据库和RAG检索
- RT-DETR、STMC、车牌识别的修改或重跑
- 跟踪指标评估

## 目录

```text
R39_RAG/
├── configs/          数据来源、字段和类别映射
├── data/raw/         原始目录软链接
├── data/normalized/  统一后的 JSONL
├── data/quarantine/  冲突和异常清单
├── knowledge_base/   预留给后续知识库
├── reports/          盘点、质量和字段说明
├── scripts/          可重复执行的入口
├── src/r39_rag/      适配、清洗和校验
└── tests/
```

## 数据来源

六段采集分开保存，不能把 R39 当成唯一场景号：

| sequence_id | 位置 | 说明 |
| --- | --- | --- |
| r39_20250924_070957 | result/20250924_070957 | 对齐框。目录内 DETR 与 ResNet 为本地补充，已忽略 |
| r39_20251021_070024 | result/20251021_070024 | 对齐框 |
| r39_20251021_070125 | R39_new/0 | 轨迹加人工全局ID |
| r39_20251021_070226 | R39_new/1 | 同上 |
| r39_20251021_070327 | R39_new/2 | 同上 |
| r39_20251021_070428 | R39_new/3 | 同上 |

## 执行

在仓库根目录或任意目录都可以运行：

```bash
python R39_RAG/scripts/link_sources.py
python R39_RAG/scripts/inspect_sources.py
python R39_RAG/scripts/normalize_data.py
python R39_RAG/scripts/validate_data.py
python -m unittest discover -s R39_RAG/tests -v
```

`link_sources.py` 可以重复执行。链接正确时直接复用，目标被其他文件占用时拒绝覆盖。

只处理某一段时：

```bash
python R39_RAG/scripts/normalize_data.py --sequence r39_20251021_070125
```

## 输出

```text
data/normalized/source_manifest.json
data/normalized/sequences/<sequence_id>/metadata.json
data/normalized/sequences/<sequence_id>/versions/<source_version>/observations.jsonl
data/normalized/sequences/<sequence_id>/versions/<source_version>/tracklets.jsonl
data/normalized/sequences/<sequence_id>/versions/<source_version>/global_tracks.jsonl
data/normalized/sequences/<sequence_id>/versions/<source_version>/id_reassignments.jsonl
```

字段说明见 `reports/unified_schema.md`。现有流水线的 `result.json` 可由 `roadside_pipeline` 适配器读取，本轮原始目录中没有这种文件。

## 已知问题

- 目录名里的日期时间没有独立时钟佐证，`capture_start_time` 为空
- 采样时间按60 FPS和抽帧步长3估计，并假设起点是第0帧
- 四路相机采样模式一致，曝光同步尚未验证
- `camera_box_3d` 的长宽不符合普通车辆尺寸，例如 Car 的 length=132.73、width=36.77、height=1.50。`geometry_3d.size_lwh` 只保留追溯，不用于后续知识库
- 轨迹地面点使用假定 z=0，不写入 `world_position`
- 同一相机同一帧的重复 ID 会拆成不同目标。观测最多、位置更靠左的一条保留原 ID，其余使用尚未占用的新 ID。其他相机若共用该全局 ID，同帧冲突的目标一起改号，没有冲突的跨相机关联保留原全局 ID。原编号在 `raw_local_track_id` 和 `raw_global_track_id`
- `truck` 映射为 Car，`motorcycle` 映射为ScooterRider，`bicycle` 映射为Cyclist，原始类别仍保留在 `class_name_raw`
- 占位类别 `id` 无法对应七类，保持原名且 `class_id` 为空
- `split_config.yaml` 与视频帧率冲突，不作为时间依据
- `R39_new/multi_view_global_ids.mp4` 无法读取
