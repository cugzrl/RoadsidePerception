# 统一字段说明

版本 `r39_rag.v1`。历史 JSON 和现有流水线 `result.json` 都先转成同一套观测、局部轨迹和全局轨迹。缺失值写 `null`，不补默认置信度，也不编造检测框。

同一 `global_track_id` 只在同一个 `sequence_id` 和 `source_version` 内有意义。不同采集段即使编号相同，也不是同一辆车。

## 关键字段

### sequence_id

一次独立视频采集的标识，例如 `r39_20250924_070957`、`r39_20251021_070125`。路口编号 `site_id` 固定为 R39，不能代替序列。目录名看起来连续的 10 月 21 日五段视频仍各自保留序列号，合并关系标记为未核实。

### source_version

同一序列上的一种处理结果。`sam3_aligned` 来自对齐后的像素框和相机三维框，`sam3_manual_gid` 来自单相机轨迹并采用人工修正后的全局 ID，`detr_car` 来自只检测 car 的 DETR 结果，`roadside_pipeline` 预留给现有 `result.json`。不同版本不会互相覆盖。

### frame_index

抽帧后的帧编号，历史数据从 0 到 1199。`source_frame_index` 是估计的原始视频帧号，当前按 `frame_index * frame_stride` 计算，并假设抽帧起点为 0。它不是文件里直接记录的字段。

### timestamp_ms

当前序列内的相对毫秒。历史数据没有逐条时间戳。四路原始视频由 ffprobe 确认为 60 FPS，`frame_cache_meta.json` 的 `frame_stride` 为 3，因此采样帧率是 20，相对时间是 `source_frame_index * 1000 / 60`。`timestamp_estimated` 为 true。无法满足该条件时时间戳保持为空，不会退回到固定的 `frame_index * 1000 / 20`。

`global_timestamp_ms` 和序列级 `capture_start_time` 保持为空。目录中的日期时间只写入 `capture_time_candidate`，状态为未核实。各序列不共用时间轴。

### global_track_id

该版本采用的全局目标编号。人工修正版本同时保留 `original_global_track_id`。修正来源是 `multi_view_global_ids_manual.json`，自动结果来自 `multi_view_global_ids.json`。编号变化、仅一侧存在和人工删除都记在隔离目录，不静默覆盖。

### bbox_xyxy

图像像素坐标 `[x1, y1, x2, y2]`。原始字典 `{x0, y0, x1, y1}` 由适配器转换。没有二维框时为 `null`。零面积或越界框保留原值并打上 `invalid_bbox` 或 `bbox_out_of_frame`。

### world_position

只有坐标系已经明确时才填写。现有流水线 `result.json` 中的路面点记为 `road_homography_xy`。历史三维结果不写入该字段。

### coordinate_frame

- `unknown`：`camera_box_3d` 的长宽中位数约十几米到几十米，高度多为类别假定值，不能当成真实车体尺寸或统一世界坐标
- `unverified_assumed_ground_plane`：SAM3 轨迹里的 `ground_point_3d`，z 被假定为 0，原点、轴向和单位都没有在本轮核实
- `road_homography_xy`：现有工程用单应矩阵得到的路面坐标
- `null`：这条观测没有三维或路面坐标

完整角点仍留在原始 JSON。统一文件里的 `geometry_3d` 只保留中心、尺寸和地面点，并标记 `corners_stored: false`。

### 类别映射

`class_name_raw` 保留原始类别。映射后的 `class_name` 和 `class_id` 对齐 RD-DETR 七类：`car` 与 `truck` 为 Car（0），`bus` 为 Bus（2），`person` 为 Pedestrian（6），`motorcycle` 为 ScooterRider（4），`bicycle` 为 Cyclist（5）。占位类别 `id` 无法对应，`class_id` 为空。

## 观测字段

| 字段 | 类型 | 可空 | 含义 |
| --- | --- | --- | --- |
| `site_id` | string | 否 | 路口编号 |
| `sequence_id` | string | 否 | 视频序列编号，按照时间编号 |
| `source_version` | string | 否 | 同一序列上的处理版本，例如sam3_aligned、sam3_manual_gid、detr_car |
| `camera_id` | string | 否 | 摄像头编号，如R39_Aw_CamS |
| `frame_index` | int | 否 | 抽帧后的帧编号，从0开始 |
| `source_frame_index` | int | 是 | 估计的原始视频帧号。无法确认抽帧起点时为空 |
| `timestamp_ms` | float | 是 | 序列内相对时间，单位毫秒 |
| `timestamp_estimated` | bool | 否 | timestamp_ms是否由帧率与抽帧步长估计 |
| `timebase_source` | string | 否 | 时间计算依据。估计失败时为unavailable |
| `global_timestamp_ms` | float | 是 | 绝对采集时间。目录名未核实，历史数据中保持为空 |
| `local_track_id` | int | 是 | 单摄像头局部轨迹ID。检测框没有轨迹时为空 |
| `global_track_id` | int | 是 | 当前版本采用的全局ID，只在所在序列本版本内有效 |
| `original_global_track_id` | int | 是 | 人工修正前的全局ID。没有修正版本时为空 |
| `class_name` | string | 是 | 目标类别名 |
| `class_name_raw` | string | 是 | 原始类别字符串 |
| `class_id` | int | 是 | 目标类别编号。未映射时为空 |
| `class_mapping_status` | string | 否 | mapped 或 unmapped |
| `score` | float | 是 | 检测置信度。原始记录没有该字段时为空，不填 1.0 |
| `bbox_xyxy` | float[4] | 是 | 图像像素框 [x1, y1, x2, y2]。没有二维框时为空 |
| `pixel_centroid` | float[2] | 是 | 像素质心 [x, y]。原始记录没有时为空 |
| `world_position` | object | 是 | 已确认含义的路面坐标。坐标系未核实的三维点不写入此字段 |
| `coordinate_frame` | string | 是 | world_position 或几何的坐标系名称。未知时为 unknown 或 unverified_assumed_ground_plane，没有几何时为空 |
| `geometry_3d` | object | 是 | 原始三维信息的紧凑保留，不含八个角点。角点仍在原始文件中 |
| `source_file` | string | 否 | 相对 data/raw 或配置根的原始文件路径 |
| `source_record_index` | int | 否 | 原始文件中的记录序号，用于回溯 |
| `record_uid` | string | 否 | 序列、版本、相机、帧、局部 ID 与同帧序号组成的观测标识 |
| `quality_flags` | string[] | 否 | 质量标记。空列表表示本条没有触发规则 |
| `manual_fix` | bool | 否 | 全局 ID 是否相对自动关联结果被人工修改 |
| `source_label` | string | 是 | 原始 label，例如 car:10 |
| `appearance_ref` | object | 是 | 外观特征文件中的索引。本轮不复制特征向量 |
| `detection_id` | string | 是 | 现有流水线检测框编号。历史 JSON 没有该字段时为空 |
| `plate` | string | 是 | 车牌号。历史 JSON 与未识别时为空 |

## 局部轨迹

局部轨迹由观测按相机和 `local_track_id` 汇总。只统计真实出现的帧，不用起止帧补全中间缺失。

| 字段 | 类型 | 可空 | 含义 |
| --- | --- | --- | --- |
| `sequence_id` | string | 否 | 采集序列 |
| `source_version` | string | 否 | 数据版本 |
| `camera_id` | string | 否 | 单摄像头轨迹所属相机 |
| `local_track_id` | int | 否 | 局部轨迹 ID |
| `global_track_id` | int | 是 | 该局部轨迹对应的全局 ID。同轨迹出现多个全局 ID 时为空并记冲突 |
| `original_global_track_id` | int | 是 | 修正前全局 ID |
| `class_name` | string | 是 | 轨迹类别。多个原始类别并存时取首次出现的映射结果，并打标记 |
| `class_name_raw` | string | 是 | 首次出现的原始类别 |
| `class_id` | int | 是 | 映射后的类别编号 |
| `start_frame` | int | 否 | 已观测到的最小抽帧编号 |
| `end_frame` | int | 否 | 已观测到的最大抽帧编号 |
| `start_timestamp_ms` | float | 是 | 起始帧相对时间 |
| `end_timestamp_ms` | float | 是 | 结束帧相对时间 |
| `observation_count` | int | 否 | 框记录数，不是起止帧差加一 |
| `unique_frame_count` | int | 否 | 去重后的帧数。同一帧多个框时小于 observation_count |
| `source_files` | string[] | 否 | 来源文件 |
| `quality_flags` | string[] | 否 | 轨迹级质量标记 |
| `manual_fix` | bool | 否 | 是否包含人工修正 |

## 全局轨迹

全局轨迹按本版本的 `global_track_id` 汇总。没有全局 ID 的检测不会被编成轨迹。

| 字段 | 类型 | 可空 | 含义 |
| --- | --- | --- | --- |
| `sequence_id` | string | 否 | 采集序列 |
| `source_version` | string | 否 | 数据版本 |
| `global_track_id` | int | 否 | 本序列本版本内的全局 ID |
| `class_name` | string | 是 | 全局轨迹类别。类别冲突时仍保留首次类别并打标记 |
| `class_names_raw` | string[] | 否 | 关联到的原始类别集合 |
| `camera_ids` | string[] | 否 | 出现过的相机 |
| `local_track_ids` | string[] | 否 | 相机与局部 ID 的对应，格式 camera:local_id |
| `start_frame` | int | 否 | 已观测最小帧 |
| `end_frame` | int | 否 | 已观测最大帧 |
| `start_timestamp_ms` | float | 是 | 起始相对时间 |
| `end_timestamp_ms` | float | 是 | 结束相对时间 |
| `observation_count` | int | 否 | 框记录数 |
| `unique_frame_count` | int | 否 | 跨相机按帧编号去重后的数量，不表示已经证明时钟同步 |
| `camera_observation_counts` | object | 否 | 各相机框数量 |
| `original_global_track_ids` | int[] | 否 | 合并进来的修正前全局 ID |
| `source_files` | string[] | 否 | 来源文件 |
| `quality_flags` | string[] | 否 | 全局轨迹质量标记 |
| `manual_fix` | bool | 否 | 组成该全局轨迹的局部轨迹是否存在人工修正 |
| `plate` | string | 是 | 流水线轨迹车牌。历史全局轨迹没有车牌时为空 |

## 质量标记

- `class_unmapped`：原始类别不能对应七类标准名
- `invalid_bbox`：框为空、零面积或左右上下颠倒
- `bbox_out_of_frame`：框超出图像宽高
- `local_id_frame_collision`：同一局部 ID 在同一帧有多套不同的框
- `geometry_size_implausible`：三维长大于 12 米、宽大于 3.5 米，或长宽为 0
- `geometry_range_suspect`：平面坐标绝对值大于 150
- `missing_global_id`：没有可用的全局 ID
- `manual_gid_override`：人工全局 ID 与自动结果不同
- `manual_deleted`：人工结果标记删除
- `frame_gap`：局部轨迹起止帧之间有缺失帧
- `class_conflict`：同一个全局 ID 下出现多个原始类别

这些标记是数据一致性检查，不是 MOTA、IDF1 或 HOTA。
