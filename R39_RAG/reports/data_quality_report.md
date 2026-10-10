# 数据质量报告

本报告只描述格式转换和一致性检查。它不评估检测或跟踪准确率，也不计算 MOTA、IDF1、HOTA。

完全相同的记录会去掉重复并计数。同一主键内容不同的记录全部保留，并写入 `data/quarantine`。不同处理版本分目录保存。

## 总览

| 序列 | 版本 | 输入 | 写出 | 完全重复 | 冲突条数 | 无效框 | 越界框 | 未映射类别 | 三维尺寸异常 | 坐标范围异常 | 缺少时间 | 缺少全局 ID |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r39_20250924_070957 | sam3_aligned | 125990 | 125989 | 1 | 2109 | 6 | 0 | 74 | 111636 | 38283 | 0 | 0 |
| r39_20250924_070957 | detr_car | 161189 | 161189 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| r39_20251021_070024 | sam3_aligned | 64071 | 64071 | 0 | 44 | 22 | 0 | 0 | 61123 | 30396 | 0 | 0 |
| r39_20251021_070125 | sam3_manual_gid | 63028 | 63028 | 0 | 2093 | 6 | 0 | 0 | 0 | 28856 | 0 | 21 |
| r39_20251021_070226 | sam3_manual_gid | 77819 | 77819 | 0 | 2800 | 11 | 0 | 0 | 0 | 26072 | 0 | 0 |
| r39_20251021_070327 | sam3_manual_gid | 69612 | 69612 | 0 | 3697 | 9 | 0 | 0 | 0 | 25779 | 0 | 0 |
| r39_20251021_070428 | sam3_manual_gid | 73924 | 73924 | 0 | 4655 | 9 | 0 | 0 | 0 | 30486 | 0 | 0 |

## 分序列说明

### r39_20250924_070957

源帧率 60.0，采样帧率 20.0，抽帧步长 3，抽帧数 1200。同步状态 `sampling_pattern_consistent_clock_not_verified`。绝对采集时间为空，候选时间 2025-09-24 07:09:57 未核实。

- capture_start_time 为空，因为目录时间没有被视频元数据或其他独立时钟证实
- 相同 frame_index 只表示抽帧序号一致，不能单独证明四路相机曝光同步
- frame_cache_meta.frame_stride 与 ffprobe 帧率，抽帧起点按 0 估计

#### sam3_aligned

三维框尺寸明显偏离车辆物理尺寸，坐标系和单位未核实，未写入 world_position

原始类别：bicycle 2400，bus 1514，car 74374，id 74，motorcycle 13207，person 34375，truck 45。

映射后类别：Bus 1514，Car 74419，Cyclist 2400，Pedestrian 34375，ScooterRider 13207，id 74。

`aligned_pixel_bbox` 对照 125990 条，差异 0，内容与主文件一致，不重复写出。
`aligned_camera_bbox` 对照 125990 条，差异 0，内容与主文件一致，不重复写出。
`projected_pixel_bbox` 对照 125990 条，差异 0，内容与主文件一致，不重复写出。

示例：

- `exact_duplicate`：`{'camera_id': 'R39_Ce_CamN', 'frame_index': 197, 'local_track_id': 42, 'bbox_xyxy': [1279.0, 950.0, 1414.0, 1079.0]}`
- `local_id_frame_collision`：`{'camera_id': 'R39_Aw_CamS', 'frame_index': 0, 'local_track_id': 108, 'bboxes': [[710.0, 118.0, 745.0, 146.0], [721.0, 98.0, 751.0, 122.0], [687.0, 100.0, 712.0, 122.0]]}`
- `invalid_bbox`：`{'type': 'bbox_anomaly', 'record_uid': 'r39_20250924_070957|sam3_aligned|R39_Aw_CamS|699|137|0', 'camera_id': 'R39_Aw_CamS', 'frame_index': 699, 'bbox_xyxy': [696.0, 65.0, 696.0, 65.0], 'quality_flags': ['invalid_bbox', 'geometry_size_implausible', 'geometry_range_suspect']}`
- `geometry_size_implausible`：`{'record_uid': 'r39_20250924_070957|sam3_aligned|R39_Aw_CamS|0|10|0', 'class_name_raw': 'car', 'geometry_3d': {'type': 'camera_box_3d', 'center': [60.91100437423163, -364.5714754819049, 0.75], 'size_lwh': [132.7296530429087, 36.77254658843947, 1.5], 'ground_point': None, 'coordinate_frame': 'unknown', 'unit': 'unknown', 'height_source': 'not_verified', 'corners_stored': False}}`
- `geometry_range_suspect`：`{'record_uid': 'r39_20250924_070957|sam3_aligned|R39_Aw_CamS|0|10|0', 'geometry_3d': {'type': 'camera_box_3d', 'center': [60.91100437423163, -364.5714754819049, 0.75], 'size_lwh': [132.7296530429087, 36.77254658843947, 1.5], 'ground_point': None, 'coordinate_frame': 'unknown', 'unit': 'unknown', 'height_source': 'not_verified', 'corners_stored': False}}`

#### detr_car

该版本只有二维框

原始类别：car 161189。

映射后类别：Car 161189。

COCO 导出 161189 条，检测头记录 161189 条。detections_coco.json 是同一批检测的 COCO xywh 导出，二维框以 detections.json 的 bbox_xyxy 为准

检测结果没有局部或全局轨迹 ID，不生成伪造轨迹

### r39_20251021_070024

源帧率 60.0，采样帧率 20.0，抽帧步长 3，抽帧数 1200。同步状态 `sampling_pattern_consistent_clock_not_verified`。绝对采集时间为空，候选时间 2025-10-21 07:00:24 未核实。

- capture_start_time 为空，因为目录时间没有被视频元数据或其他独立时钟证实
- 相同 frame_index 只表示抽帧序号一致，不能单独证明四路相机曝光同步
- frame_cache_meta.frame_stride 与 ffprobe 帧率，抽帧起点按 0 估计
- 目录时间大约每61秒一段，且每段原始视频时长60秒。没有独立时间戳或画面对齐证据，各段保持独立序列

#### sam3_aligned

三维框尺寸明显偏离车辆物理尺寸，坐标系和单位未核实，未写入 world_position

原始类别：bicycle 1808，bus 1675，car 42176，motorcycle 4384，person 11281，truck 2747。

映射后类别：Bus 1675，Car 44923，Cyclist 1808，Pedestrian 11281，ScooterRider 4384。

`aligned_pixel_bbox` 对照 64071 条，差异 0，内容与主文件一致，不重复写出。
`aligned_camera_bbox` 对照 64071 条，差异 0，内容与主文件一致，不重复写出。
`projected_pixel_bbox` 对照 64071 条，差异 0，内容与主文件一致，不重复写出。

示例：

- `invalid_bbox`：`{'type': 'bbox_anomaly', 'record_uid': 'r39_20251021_070024|sam3_aligned|R39_Aw_CamS|700|2|0', 'camera_id': 'R39_Aw_CamS', 'frame_index': 700, 'bbox_xyxy': [699.0, 76.0, 699.0, 76.0], 'quality_flags': ['invalid_bbox', 'geometry_size_implausible', 'geometry_range_suspect']}`
- `local_id_frame_collision`：`{'camera_id': 'R39_Ce_CamN', 'frame_index': 756, 'local_track_id': 67, 'bboxes': [[8.0, 442.0, 47.0, 513.0], [0.0, 424.0, 27.0, 476.0]]}`
- `geometry_size_implausible`：`{'record_uid': 'r39_20251021_070024|sam3_aligned|R39_Aw_CamS|0|1|0', 'class_name_raw': 'car', 'geometry_3d': {'type': 'camera_box_3d', 'center': [38.29899035478905, -293.40243417925274, 0.75], 'size_lwh': [64.49728506133914, 17.076406197014002, 1.5], 'ground_point': None, 'coordinate_frame': 'unknown', 'unit': 'unknown', 'height_source': 'not_verified', 'corners_stored': False}}`
- `geometry_range_suspect`：`{'record_uid': 'r39_20251021_070024|sam3_aligned|R39_Aw_CamS|0|1|0', 'geometry_3d': {'type': 'camera_box_3d', 'center': [38.29899035478905, -293.40243417925274, 0.75], 'size_lwh': [64.49728506133914, 17.076406197014002, 1.5], 'ground_point': None, 'coordinate_frame': 'unknown', 'unit': 'unknown', 'height_source': 'not_verified', 'corners_stored': False}}`

### r39_20251021_070125

源帧率 60.0，采样帧率 20.0，抽帧步长 3，抽帧数 1200。同步状态 `sampling_pattern_consistent_clock_not_verified`。绝对采集时间为空，候选时间 2025-10-21 07:01:25 未核实。

- capture_start_time 为空，因为目录时间没有被视频元数据或其他独立时钟证实
- 相同 frame_index 只表示抽帧序号一致，不能单独证明四路相机曝光同步
- frame_cache_meta.frame_stride 与 ffprobe 帧率，抽帧起点按 0 估计
- split_config.yaml 的 start_time=2025-04-22 14:58:00、interval_ms=100。文件中的路径指向其他数据目录，且间隔与抽帧帧率 20.0（约 50 ms）不一致，不作为时间依据
- 目录时间大约每61秒一段，且每段原始视频时长60秒。没有独立时间戳或画面对齐证据，各段保持独立序列

#### sam3_manual_gid

ground_point_3d 使用假定地面 z=0，坐标原点和单位未核实，未写入 world_position

原始类别：bicycle 77，bus 998，car 47281，motorcycle 4518，person 8486，truck 1668。

映射后类别：Bus 998，Car 48949，Cyclist 77，Pedestrian 8486，ScooterRider 4518。

人工修正相对自动结果改变了 18 条局部轨迹，只在自动结果中的 1 条，只在人工结果中的 1 条，人工删除 0 条。文本映射差异 0 条，轨迹摘要与逐帧观测差异 6 条。

示例：

- `only_in_manual`：`{'camera_id': 'R39_Bn_CamW', 'local_track_id': 55, 'type': 'only_in_manual', 'manual_global_track_id': 18}`
- `gid_changed`：`{'camera_id': 'R39_Bn_CamW', 'local_track_id': 56, 'type': 'gid_changed', 'auto_global_track_id': 60, 'manual_global_track_id': 19, 'original_global_track_id': 60, 'class_name': 'person'}`
- `local_id_frame_collision`：`{'camera_id': 'R39_Aw_CamS', 'frame_index': 0, 'local_track_id': 9, 'bboxes': [[752.0, 88.0, 762.0, 100.0], [693.0, 81.0, 716.0, 96.0]]}`
- `invalid_bbox`：`{'type': 'bbox_anomaly', 'record_uid': 'r39_20251021_070125|sam3_manual_gid|R39_Ds_CamE|776|50|0', 'camera_id': 'R39_Ds_CamE', 'frame_index': 776, 'bbox_xyxy': [827.0, 72.0, 827.0, 72.0], 'quality_flags': ['invalid_bbox', 'geometry_range_suspect']}`
- `geometry_range_suspect`：`{'record_uid': 'r39_20251021_070125|sam3_manual_gid|R39_Aw_CamS|0|3|0', 'geometry_3d': {'type': 'ground_point_3d', 'center': [28.888845764146524, -174.93817540057177, 0.75], 'size_lwh': None, 'ground_point': [28.888845764146524, -174.93817540057177, 0.0], 'coordinate_frame': 'unverified_assumed_ground_plane', 'unit': 'unknown', 'height_source': 'assumed_by_class', 'assumed_height_m': 1.5, 'corners_stored': False}}`
- `summary_frame_mismatch`：`{'type': 'summary_frame_mismatch', 'camera_id': 'R39_Aw_CamS', 'local_track_id': 9, 'summary_start': 0, 'summary_end': 1099, 'summary_num_frames': 2347, 'observed_start': 0, 'observed_end': 1099, 'observed_unique_frames': 1100}`

### r39_20251021_070226

源帧率 60.0，采样帧率 20.0，抽帧步长 3，抽帧数 1200。同步状态 `sampling_pattern_consistent_clock_not_verified`。绝对采集时间为空，候选时间 2025-10-21 07:02:26 未核实。

- capture_start_time 为空，因为目录时间没有被视频元数据或其他独立时钟证实
- 相同 frame_index 只表示抽帧序号一致，不能单独证明四路相机曝光同步
- frame_cache_meta.frame_stride 与 ffprobe 帧率，抽帧起点按 0 估计
- split_config.yaml 的 start_time=2025-04-22 14:58:00、interval_ms=100。文件中的路径指向其他数据目录，且间隔与抽帧帧率 20.0（约 50 ms）不一致，不作为时间依据
- 目录时间大约每61秒一段，且每段原始视频时长60秒。没有独立时间戳或画面对齐证据，各段保持独立序列

#### sam3_manual_gid

ground_point_3d 使用假定地面 z=0，坐标原点和单位未核实，未写入 world_position

原始类别：bicycle 71，bus 1207，car 53878，motorcycle 8121，person 12791，truck 1751。

映射后类别：Bus 1207，Car 55629，Cyclist 71，Pedestrian 12791，ScooterRider 8121。

人工修正相对自动结果改变了 14 条局部轨迹，只在自动结果中的 0 条，只在人工结果中的 1 条，人工删除 0 条。文本映射差异 0 条，轨迹摘要与逐帧观测差异 10 条。

示例：

- `gid_changed`：`{'camera_id': 'R39_Aw_CamS', 'local_track_id': 10, 'type': 'gid_changed', 'auto_global_track_id': 2, 'manual_global_track_id': 87, 'original_global_track_id': 2, 'class_name': 'car'}`
- `local_id_frame_collision`：`{'camera_id': 'R39_Aw_CamS', 'frame_index': 0, 'local_track_id': 2, 'bboxes': [[687.0, 79.0, 711.0, 92.0], [683.0, 66.0, 700.0, 78.0]]}`
- `invalid_bbox`：`{'type': 'bbox_anomaly', 'record_uid': 'r39_20251021_070226|sam3_manual_gid|R39_Aw_CamS|61|11|0', 'camera_id': 'R39_Aw_CamS', 'frame_index': 61, 'bbox_xyxy': [689.0, 77.0, 694.0, 77.0], 'quality_flags': ['invalid_bbox', 'geometry_range_suspect']}`
- `geometry_range_suspect`：`{'record_uid': 'r39_20251021_070226|sam3_manual_gid|R39_Aw_CamS|49|1|0', 'geometry_3d': {'type': 'ground_point_3d', 'center': [6.388479638815862, -150.20535628457927, 0.75], 'size_lwh': None, 'ground_point': [6.388479638815862, -150.20535628457927, 0.0], 'coordinate_frame': 'unverified_assumed_ground_plane', 'unit': 'unknown', 'height_source': 'assumed_by_class', 'assumed_height_m': 1.5, 'corners_stored': False}}`
- `summary_frame_mismatch`：`{'type': 'summary_frame_mismatch', 'camera_id': 'R39_Aw_CamS', 'local_track_id': 2, 'summary_start': 0, 'summary_end': 399, 'summary_num_frames': 492, 'observed_start': 0, 'observed_end': 399, 'observed_unique_frames': 392}`

### r39_20251021_070327

源帧率 60.0，采样帧率 20.0，抽帧步长 3，抽帧数 1200。同步状态 `sampling_pattern_consistent_clock_not_verified`。绝对采集时间为空，候选时间 2025-10-21 07:03:27 未核实。

- capture_start_time 为空，因为目录时间没有被视频元数据或其他独立时钟证实
- 相同 frame_index 只表示抽帧序号一致，不能单独证明四路相机曝光同步
- frame_cache_meta.frame_stride 与 ffprobe 帧率，抽帧起点按 0 估计
- split_config.yaml 的 start_time=2025-04-22 14:58:00、interval_ms=100。文件中的路径指向其他数据目录，且间隔与抽帧帧率 20.0（约 50 ms）不一致，不作为时间依据
- 目录时间大约每61秒一段，且每段原始视频时长60秒。没有独立时间戳或画面对齐证据，各段保持独立序列

#### sam3_manual_gid

ground_point_3d 使用假定地面 z=0，坐标原点和单位未核实，未写入 world_position

原始类别：bicycle 531，bus 263，car 46894，motorcycle 5026，person 14962，truck 1936。

映射后类别：Bus 263，Car 48830，Cyclist 531，Pedestrian 14962，ScooterRider 5026。

人工修正相对自动结果改变了 12 条局部轨迹，只在自动结果中的 0 条，只在人工结果中的 0 条，人工删除 0 条。文本映射差异 0 条，轨迹摘要与逐帧观测差异 12 条。

示例：

- `gid_changed`：`{'camera_id': 'R39_Aw_CamS', 'local_track_id': 131, 'type': 'gid_changed', 'auto_global_track_id': 12, 'manual_global_track_id': 24, 'original_global_track_id': 12, 'class_name': 'car'}`
- `invalid_bbox`：`{'type': 'bbox_anomaly', 'record_uid': 'r39_20251021_070327|sam3_manual_gid|R39_Aw_CamS|258|2|0', 'camera_id': 'R39_Aw_CamS', 'frame_index': 258, 'bbox_xyxy': [699.0, 76.0, 699.0, 76.0], 'quality_flags': ['invalid_bbox', 'geometry_range_suspect']}`
- `local_id_frame_collision`：`{'camera_id': 'R39_Aw_CamS', 'frame_index': 0, 'local_track_id': 5, 'bboxes': [[683.0, 66.0, 700.0, 78.0], [288.0, 877.0, 671.0, 1079.0]]}`
- `geometry_range_suspect`：`{'record_uid': 'r39_20251021_070327|sam3_manual_gid|R39_Aw_CamS|0|2|0', 'geometry_3d': {'type': 'ground_point_3d', 'center': [12.14722584509858, -184.3508460524692, 0.75], 'size_lwh': None, 'ground_point': [12.14722584509858, -184.3508460524692, 0.0], 'coordinate_frame': 'unverified_assumed_ground_plane', 'unit': 'unknown', 'height_source': 'assumed_by_class', 'assumed_height_m': 1.5, 'corners_stored': False}}`
- `summary_frame_mismatch`：`{'type': 'summary_frame_mismatch', 'camera_id': 'R39_Aw_CamS', 'local_track_id': 5, 'summary_start': 0, 'summary_end': 1199, 'summary_num_frames': 2260, 'observed_start': 0, 'observed_end': 1199, 'observed_unique_frames': 1200}`

### r39_20251021_070428

源帧率 60.0，采样帧率 20.0，抽帧步长 3，抽帧数 1200。同步状态 `sampling_pattern_consistent_clock_not_verified`。绝对采集时间为空，候选时间 2025-10-21 07:04:28 未核实。

- capture_start_time 为空，因为目录时间没有被视频元数据或其他独立时钟证实
- 相同 frame_index 只表示抽帧序号一致，不能单独证明四路相机曝光同步
- frame_cache_meta.frame_stride 与 ffprobe 帧率，抽帧起点按 0 估计
- split_config.yaml 的 start_time=2025-04-22 14:58:00、interval_ms=100。文件中的路径指向其他数据目录，且间隔与抽帧帧率 20.0（约 50 ms）不一致，不作为时间依据
- 目录时间大约每61秒一段，且每段原始视频时长60秒。没有独立时间戳或画面对齐证据，各段保持独立序列

#### sam3_manual_gid

ground_point_3d 使用假定地面 z=0，坐标原点和单位未核实，未写入 world_position

原始类别：bus 45，car 56804，motorcycle 3829，person 11676，truck 1570。

映射后类别：Bus 45，Car 58374，Pedestrian 11676，ScooterRider 3829。

人工修正相对自动结果改变了 18 条局部轨迹，只在自动结果中的 0 条，只在人工结果中的 0 条，人工删除 0 条。文本映射差异 0 条，轨迹摘要与逐帧观测差异 12 条。

示例：

- `gid_changed`：`{'camera_id': 'R39_Aw_CamS', 'local_track_id': 2, 'type': 'gid_changed', 'auto_global_track_id': 29, 'manual_global_track_id': 69, 'original_global_track_id': 29, 'class_name': 'car'}`
- `local_id_frame_collision`：`{'camera_id': 'R39_Aw_CamS', 'frame_index': 0, 'local_track_id': 133, 'bboxes': [[1280.0, 307.0, 1512.0, 381.0], [368.0, 173.0, 419.0, 213.0]]}`
- `invalid_bbox`：`{'type': 'bbox_anomaly', 'record_uid': 'r39_20251021_070428|sam3_manual_gid|R39_Aw_CamS|1017|203|0', 'camera_id': 'R39_Aw_CamS', 'frame_index': 1017, 'bbox_xyxy': [688.0, 66.0, 688.0, 66.0], 'quality_flags': ['invalid_bbox', 'geometry_range_suspect']}`
- `geometry_range_suspect`：`{'record_uid': 'r39_20251021_070428|sam3_manual_gid|R39_Aw_CamS|350|1|0', 'geometry_3d': {'type': 'ground_point_3d', 'center': [3.4138045052240935, -150.29119640110372, 0.75], 'size_lwh': None, 'ground_point': [3.4138045052240935, -150.29119640110372, 0.0], 'coordinate_frame': 'unverified_assumed_ground_plane', 'unit': 'unknown', 'height_source': 'assumed_by_class', 'assumed_height_m': 1.5, 'corners_stored': False}}`
- `summary_frame_mismatch`：`{'type': 'summary_frame_mismatch', 'camera_id': 'R39_Aw_CamS', 'local_track_id': 133, 'summary_start': 0, 'summary_end': 1199, 'summary_num_frames': 2200, 'observed_start': 0, 'observed_end': 1199, 'observed_unique_frames': 1200}`

## 坐标系

`sam3_aligned` 的三维框来自 `camera_boxes`。尺寸检查使用长度大于 12 米、宽度大于 3.5 米或长宽为 0 作为明显异常，只打标记，不删除，也不回写二维框。

`sam3_manual_gid` 的地面点来自轨迹文件。部分点的平面坐标绝对值超过 150，标记为 `geometry_range_suspect`。现有工程 `configs/cameras/r39` 的单应矩阵没有拿来重算这些点。

因此本轮没有可以确认已经处在统一世界坐标系中的历史三维结果。

## 隔离文件

`data/quarantine/<sequence_id>/<source_version>/conflicts.jsonl` 保存同键冲突、版本差异和摘要不一致。`anomalies.jsonl` 保存无效框和越界框。原始文件没有被修改。
