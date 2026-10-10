# R39 历史数据盘点

盘点对象是两处只读原始目录。采集序列按原始 `sequence_dir` 和视频文件区分，没有因为同为 R39 或全局 ID 相同而合并。

## 目录规模

### result

根目录 `/home/kemove/devdata1/RodeTest/roadside_perception/R39_RAG/data/raw/result`。

| 后缀 | 数量 |
| --- | ---: |
| `.jpg` | 175589 |
| `.json` | 22 |
| `.log` | 1 |
| `.md` | 1 |
| `.mp4` | 14 |
| `.npy` | 1 |
| `.png` | 2424 |
| `.pt` | 1 |
| `.py` | 1 |

| 相对路径 | 大小 | 类型 |
| --- | ---: | --- |
| `20250924_070957/box_info/aligned_bbox_info.json` | 237.77 MB | 像素框与相机框合并文件 |
| `20250924_070957/box_info/aligned_camera_bbox.json` | 176.01 MB | 相机三维框 |
| `20250924_070957/box_info/aligned_pixel_bbox.json` | 42.05 MB | 逐帧像素框 |
| `20250924_070957/box_info/projected_pixel_bbox.json` | 211.84 MB | 三维框投影回像素 |
| `20250924_070957/calib.json` | 0.00 MB | 历史相机标定 |
| `20250924_070957/detr_result/detections.json` | 52.13 MB | DETR 逐帧检测，本地补充，不纳入清洗 |
| `20250924_070957/detr_result/detections_coco.json` | 22.80 MB | DETR 的 COCO 导出，本地补充，不纳入清洗 |
| `20250924_070957/detr_result/failures.json` | 0.00 MB | 检测失败记录，本地补充，不纳入清洗 |
| `20250924_070957/raw_video/R39_Aw_CamS/R39_Aw_CamS.mp4` | 50.71 MB | .mp4 |
| `20250924_070957/raw_video/R39_Aw_CamS/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `20250924_070957/raw_video/R39_Bn_CamW/R39_Bn_CamW.mp4` | 58.89 MB | .mp4 |
| `20250924_070957/raw_video/R39_Bn_CamW/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `20250924_070957/raw_video/R39_Ce_CamN/R39_Ce_CamN.mp4` | 78.48 MB | .mp4 |
| `20250924_070957/raw_video/R39_Ce_CamN/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `20250924_070957/raw_video/R39_Ds_CamE/R39_Ds_CamE.mp4` | 51.56 MB | .mp4 |
| `20250924_070957/raw_video/R39_Ds_CamE/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `20250924_070957/resnet_result/feature_metadata.json` | 73.82 MB | ResNet50 特征索引，本地补充，不纳入清洗 |
| `20250924_070957/result_data/multi_view_global_ids.mp4` | 125.24 MB | .mp4 |
| `20251021_070024/box_info/aligned_bbox_info.json` | 120.57 MB | 像素框与相机框合并文件 |
| `20251021_070024/box_info/aligned_camera_bbox.json` | 89.34 MB | 相机三维框 |
| `20251021_070024/box_info/aligned_pixel_bbox.json` | 21.20 MB | 逐帧像素框 |
| `20251021_070024/box_info/projected_pixel_bbox.json` | 107.48 MB | 三维框投影回像素 |
| `20251021_070024/calib.json` | 0.00 MB | 历史相机标定 |
| `20251021_070024/raw_video/R39_Aw_CamS.mp4` | 118.64 MB | .mp4 |
| `20251021_070024/raw_video/R39_Aw_CamS/R39_Aw_CamS.mp4` | 118.64 MB | .mp4 |
| `20251021_070024/raw_video/R39_Aw_CamS/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `20251021_070024/raw_video/R39_Bn_CamW.mp4` | 65.05 MB | .mp4 |
| `20251021_070024/raw_video/R39_Bn_CamW/R39_Bn_CamW.mp4` | 65.05 MB | .mp4 |
| `20251021_070024/raw_video/R39_Bn_CamW/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `20251021_070024/raw_video/R39_Ce_CamN.mp4` | 69.74 MB | .mp4 |
| `20251021_070024/raw_video/R39_Ce_CamN/R39_Ce_CamN.mp4` | 69.74 MB | .mp4 |
| `20251021_070024/raw_video/R39_Ce_CamN/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `20251021_070024/raw_video/R39_Ds_CamE.mp4` | 54.60 MB | .mp4 |
| `20251021_070024/raw_video/R39_Ds_CamE/R39_Ds_CamE.mp4` | 54.60 MB | .mp4 |
| `20251021_070024/raw_video/R39_Ds_CamE/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `20251021_070024/result_data/multi_view_global_ids.mp4` | 112.34 MB | .mp4 |

### R39_new

根目录 `/home/kemove/devdata1/RodeTest/roadside_perception/R39_RAG/data/raw/R39_new`。

| 后缀 | 数量 |
| --- | ---: |
| `.csv` | 4 |
| `.jpg` | 19200 |
| `.json` | 76 |
| `.md` | 1 |
| `.mp4` | 21 |
| `.npz` | 269993 |
| `.py` | 2 |
| `.txt` | 12 |
| `.yaml` | 4 |

| 相对路径 | 大小 | 类型 |
| --- | ---: | --- |
| `0/R39_Aw_CamS/sam3_multi_car__person__truck__bus__motorcycle__bicycle/object_labels.json` | 0.00 MB | 局部 ID 类别标签 |
| `0/R39_Aw_CamS/sam3_multi_car__person__truck__bus__motorcycle__bicycle/tracks.json` | 8.43 MB | 单相机 SAM3 轨迹 |
| `0/R39_Aw_CamS/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `0/R39_Bn_CamW/sam3_multi_car__person__truck__bus__motorcycle__bicycle/object_labels.json` | 0.00 MB | 局部 ID 类别标签 |
| `0/R39_Bn_CamW/sam3_multi_car__person__truck__bus__motorcycle__bicycle/tracks.json` | 8.90 MB | 单相机 SAM3 轨迹 |
| `0/R39_Bn_CamW/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `0/R39_Ce_CamN/sam3_multi_car__person__truck__bus__motorcycle__bicycle/object_labels.json` | 0.00 MB | 局部 ID 类别标签 |
| `0/R39_Ce_CamN/sam3_multi_car__person__truck__bus__motorcycle__bicycle/tracks.json` | 7.66 MB | 单相机 SAM3 轨迹 |
| `0/R39_Ce_CamN/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `0/R39_Ds_CamE/sam3_multi_car__person__truck__bus__motorcycle__bicycle/object_labels.json` | 0.00 MB | 局部 ID 类别标签 |
| `0/R39_Ds_CamE/sam3_multi_car__person__truck__bus__motorcycle__bicycle/tracks.json` | 9.41 MB | 单相机 SAM3 轨迹 |
| `0/R39_Ds_CamE/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `0/exported_instance_masks/mask_manifest.json` | 20.71 MB | 实例掩码清单 |
| `0/lid_to_gid_mapping.txt` | 0.00 MB | .txt |
| `0/lid_to_gid_mapping_manual.txt` | 0.00 MB | .txt |
| `0/manual_id_corrections.txt` | 0.00 MB | .txt |
| `0/manual_id_inventory.csv` | 0.03 MB | .csv |
| `0/multi_view_global_ids.json` | 0.19 MB | 自动全局关联 |
| `0/multi_view_global_ids_manual.json` | 0.13 MB | 人工修正全局关联 |
| `0/multi_view_vis/multi_view_global_ids.mp4` | 111.84 MB | .mp4 |
| `0/result/20260812_173716/raw_video/R39_Aw_CamS/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `0/result/20260812_173716/raw_video/R39_Bn_CamW/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `0/result/20260812_173716/raw_video/R39_Ce_CamN/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `0/result/20260812_173716/raw_video/R39_Ds_CamE/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `0/原始视频/R39_Aw_CamS.mp4` | 123.61 MB | .mp4 |
| `0/原始视频/R39_Bn_CamW.mp4` | 59.56 MB | .mp4 |
| `0/原始视频/R39_Ce_CamN.mp4` | 72.65 MB | .mp4 |
| `0/原始视频/R39_Ds_CamE.mp4` | 54.71 MB | .mp4 |
| `0/原始视频/split_config.yaml` | 0.00 MB | .yaml |
| `1/R39_Aw_CamS/sam3_multi_car__person__truck__bus__motorcycle__bicycle/object_labels.json` | 0.00 MB | 局部 ID 类别标签 |
| `1/R39_Aw_CamS/sam3_multi_car__person__truck__bus__motorcycle__bicycle/tracks.json` | 6.91 MB | 单相机 SAM3 轨迹 |
| `1/R39_Aw_CamS/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `1/R39_Bn_CamW/sam3_multi_car__person__truck__bus__motorcycle__bicycle/object_labels.json` | 0.00 MB | 局部 ID 类别标签 |
| `1/R39_Bn_CamW/sam3_multi_car__person__truck__bus__motorcycle__bicycle/tracks.json` | 13.04 MB | 单相机 SAM3 轨迹 |
| `1/R39_Bn_CamW/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `1/R39_Ce_CamN/sam3_multi_car__person__truck__bus__motorcycle__bicycle/object_labels.json` | 0.00 MB | 局部 ID 类别标签 |
| `1/R39_Ce_CamN/sam3_multi_car__person__truck__bus__motorcycle__bicycle/tracks.json` | 10.20 MB | 单相机 SAM3 轨迹 |
| `1/R39_Ce_CamN/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `1/R39_Ds_CamE/sam3_multi_car__person__truck__bus__motorcycle__bicycle/object_labels.json` | 0.00 MB | 局部 ID 类别标签 |
| `1/R39_Ds_CamE/sam3_multi_car__person__truck__bus__motorcycle__bicycle/tracks.json` | 12.37 MB | 单相机 SAM3 轨迹 |
| `1/R39_Ds_CamE/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `1/exported_instance_masks/mask_manifest.json` | 25.69 MB | 实例掩码清单 |
| `1/lid_to_gid_mapping.txt` | 0.01 MB | .txt |
| `1/lid_to_gid_mapping_manual.txt` | 0.01 MB | .txt |
| `1/manual_id_corrections.txt` | 0.00 MB | .txt |
| `1/manual_id_inventory.csv` | 0.05 MB | .csv |
| `1/multi_view_global_ids.json` | 0.29 MB | 自动全局关联 |
| `1/multi_view_global_ids_manual.json` | 0.18 MB | 人工修正全局关联 |
| `1/multi_view_vis/multi_view_global_ids.mp4` | 112.60 MB | .mp4 |
| `1/result/20260812_173716/raw_video/R39_Aw_CamS/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `1/result/20260812_173716/raw_video/R39_Bn_CamW/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `1/result/20260812_173716/raw_video/R39_Ce_CamN/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `1/result/20260812_173716/raw_video/R39_Ds_CamE/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `1/原始视频/R39_Aw_CamS.mp4` | 80.12 MB | .mp4 |
| `1/原始视频/R39_Bn_CamW.mp4` | 74.45 MB | .mp4 |
| `1/原始视频/R39_Ce_CamN.mp4` | 57.79 MB | .mp4 |
| `1/原始视频/R39_Ds_CamE.mp4` | 54.58 MB | .mp4 |
| `1/原始视频/split_config.yaml` | 0.00 MB | .yaml |
| `2/R39_Aw_CamS/sam3_multi_car__person__truck__bus__motorcycle__bicycle/object_labels.json` | 0.00 MB | 局部 ID 类别标签 |
| `2/R39_Aw_CamS/sam3_multi_car__person__truck__bus__motorcycle__bicycle/tracks.json` | 12.88 MB | 单相机 SAM3 轨迹 |
| `2/R39_Aw_CamS/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `2/R39_Bn_CamW/sam3_multi_car__person__truck__bus__motorcycle__bicycle/object_labels.json` | 0.00 MB | 局部 ID 类别标签 |
| `2/R39_Bn_CamW/sam3_multi_car__person__truck__bus__motorcycle__bicycle/tracks.json` | 9.22 MB | 单相机 SAM3 轨迹 |
| `2/R39_Bn_CamW/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `2/R39_Ce_CamN/sam3_multi_car__person__truck__bus__motorcycle__bicycle/object_labels.json` | 0.00 MB | 局部 ID 类别标签 |
| `2/R39_Ce_CamN/sam3_multi_car__person__truck__bus__motorcycle__bicycle/tracks.json` | 9.57 MB | 单相机 SAM3 轨迹 |
| `2/R39_Ce_CamN/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `2/R39_Ds_CamE/sam3_multi_car__person__truck__bus__motorcycle__bicycle/object_labels.json` | 0.00 MB | 局部 ID 类别标签 |
| `2/R39_Ds_CamE/sam3_multi_car__person__truck__bus__motorcycle__bicycle/tracks.json` | 6.36 MB | 单相机 SAM3 轨迹 |
| `2/R39_Ds_CamE/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `2/exported_instance_masks/mask_manifest.json` | 22.96 MB | 实例掩码清单 |
| `2/lid_to_gid_mapping.txt` | 0.01 MB | .txt |
| `2/lid_to_gid_mapping_manual.txt` | 0.01 MB | .txt |
| `2/manual_id_corrections.txt` | 0.00 MB | .txt |
| `2/manual_id_inventory.csv` | 0.04 MB | .csv |
| `2/multi_view_global_ids.json` | 0.25 MB | 自动全局关联 |
| `2/multi_view_global_ids_manual.json` | 0.16 MB | 人工修正全局关联 |
| `2/multi_view_vis/multi_view_global_ids.mp4` | 111.03 MB | .mp4 |
| `2/result/20260812_173716/raw_video/R39_Aw_CamS/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `2/result/20260812_173716/raw_video/R39_Bn_CamW/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `2/result/20260812_173716/raw_video/R39_Ce_CamN/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `2/result/20260812_173716/raw_video/R39_Ds_CamE/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `2/原始视频/R39_Aw_CamS.mp4` | 74.47 MB | .mp4 |
| `2/原始视频/R39_Bn_CamW.mp4` | 79.83 MB | .mp4 |
| `2/原始视频/R39_Ce_CamN.mp4` | 72.87 MB | .mp4 |
| `2/原始视频/R39_Ds_CamE.mp4` | 57.70 MB | .mp4 |
| `2/原始视频/split_config.yaml` | 0.00 MB | .yaml |
| `3/R39_Aw_CamS/sam3_multi_car__person__truck__bus__motorcycle__bicycle/object_labels.json` | 0.00 MB | 局部 ID 类别标签 |
| `3/R39_Aw_CamS/sam3_multi_car__person__truck__bus__motorcycle__bicycle/tracks.json` | 11.26 MB | 单相机 SAM3 轨迹 |
| `3/R39_Aw_CamS/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `3/R39_Bn_CamW/sam3_multi_car__person__truck__bus__motorcycle__bicycle/object_labels.json` | 0.00 MB | 局部 ID 类别标签 |
| `3/R39_Bn_CamW/sam3_multi_car__person__truck__bus__motorcycle__bicycle/tracks.json` | 11.08 MB | 单相机 SAM3 轨迹 |
| `3/R39_Bn_CamW/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `3/R39_Ce_CamN/sam3_multi_car__person__truck__bus__motorcycle__bicycle/object_labels.json` | 0.00 MB | 局部 ID 类别标签 |
| `3/R39_Ce_CamN/sam3_multi_car__person__truck__bus__motorcycle__bicycle/tracks.json` | 7.73 MB | 单相机 SAM3 轨迹 |
| `3/R39_Ce_CamN/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `3/R39_Ds_CamE/sam3_multi_car__person__truck__bus__motorcycle__bicycle/object_labels.json` | 0.00 MB | 局部 ID 类别标签 |
| `3/R39_Ds_CamE/sam3_multi_car__person__truck__bus__motorcycle__bicycle/tracks.json` | 10.29 MB | 单相机 SAM3 轨迹 |
| `3/R39_Ds_CamE/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `3/exported_instance_masks/mask_manifest.json` | 24.30 MB | 实例掩码清单 |
| `3/lid_to_gid_mapping.txt` | 0.01 MB | .txt |
| `3/lid_to_gid_mapping_manual.txt` | 0.01 MB | .txt |
| `3/manual_id_corrections.txt` | 0.00 MB | .txt |
| `3/manual_id_inventory.csv` | 0.04 MB | .csv |
| `3/multi_view_global_ids.json` | 0.23 MB | 自动全局关联 |
| `3/multi_view_global_ids_manual.json` | 0.14 MB | 人工修正全局关联 |
| `3/multi_view_vis/multi_view_global_ids.mp4` | 111.09 MB | .mp4 |
| `3/result/20260812_173716/raw_video/R39_Aw_CamS/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `3/result/20260812_173716/raw_video/R39_Bn_CamW/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `3/result/20260812_173716/raw_video/R39_Ce_CamN/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `3/result/20260812_173716/raw_video/R39_Ds_CamE/sam3_multi_frames/frame_cache_meta.json` | 0.00 MB | 抽帧元数据 |
| `3/原始视频/R39_Aw_CamS.mp4` | 59.94 MB | .mp4 |
| `3/原始视频/R39_Bn_CamW.mp4` | 71.07 MB | .mp4 |
| `3/原始视频/R39_Ce_CamN.mp4` | 100.59 MB | .mp4 |
| `3/原始视频/R39_Ds_CamE.mp4` | 60.31 MB | .mp4 |
| `3/原始视频/split_config.yaml` | 0.00 MB | .yaml |
| `multi_view_global_ids.mp4` | 39.00 MB | .mp4 |

## 采集序列

下表的记录数来自像素框或单相机轨迹，还没有做跨版本去重。

| sequence_id | 目录 | 原始序列目录 | 候选时间 | 记录数 | 帧率结论 |
| --- | --- | --- | --- | ---: | --- |
| `r39_20250924_070957` | `result/20250924_070957` | `/home/stu/ZH20260617/sam3/output/multi_view/20250924_070957` | 2025-09-24 07:09:57（未核实） | 125990 | 源视频 60 FPS，时长 60 秒，抽帧步长 3 |
| `r39_20251021_070024` | `result/20251021_070024` | `/home/stu/ZH20260617/sam3/output/multi_view/20251021_070024` | 2025-10-21 07:00:24（未核实） | 64071 | 源视频 60 FPS，时长 60 秒，抽帧步长 3 |
| `r39_20251021_070125` | `R39_new/0` | `/home/stu/ZH20260617/sam3/output/multi_view/20251021_070125` | 2025-10-21 07:01:25（未核实） | 63028 | 源视频 60 FPS，时长 60 秒，抽帧步长 3 |
| `r39_20251021_070226` | `R39_new/1` | `/home/stu/ZH20260617/sam3/output/multi_view/20251021_070226` | 2025-10-21 07:02:26（未核实） | 77819 | 源视频 60 FPS，时长 60 秒，抽帧步长 3 |
| `r39_20251021_070327` | `R39_new/2` | `/home/stu/ZH20260617/sam3/output/multi_view/20251021_070327` | 2025-10-21 07:03:27（未核实） | 69612 | 源视频 60 FPS，时长 60 秒，抽帧步长 3 |
| `r39_20251021_070428` | `R39_new/3` | `/home/stu/ZH20260617/sam3/output/multi_view/20251021_070428` | 2025-10-21 07:04:28（未核实） | 73924 | 源视频 60 FPS，时长 60 秒，抽帧步长 3 |

## 各类别数量

### r39_20250924_070957

| 原始类别 | 数量 |
| --- | ---: |
| `car` | 74374 |
| `person` | 34375 |
| `motorcycle` | 13208 |
| `bicycle` | 2400 |
| `bus` | 1514 |
| `id` | 74 |
| `truck` | 45 |

detr_result 与 resnet_result 是本地补充的检测和外观特征，不是甲方结构化数据，不进入统一结果

### r39_20251021_070024

| 原始类别 | 数量 |
| --- | ---: |
| `car` | 42176 |
| `person` | 11281 |
| `motorcycle` | 4384 |
| `truck` | 2747 |
| `bicycle` | 1808 |
| `bus` | 1675 |

### r39_20251021_070125

| 原始类别 | 数量 |
| --- | ---: |
| `car` | 47281 |
| `person` | 8486 |
| `motorcycle` | 4518 |
| `truck` | 1668 |
| `bus` | 998 |
| `bicycle` | 77 |

自动关联 191 条局部轨迹、122 个全局目标；人工版本 191 条局部轨迹、115 个全局目标。

### r39_20251021_070226

| 原始类别 | 数量 |
| --- | ---: |
| `car` | 53878 |
| `person` | 12791 |
| `motorcycle` | 8121 |
| `truck` | 1751 |
| `bus` | 1207 |
| `bicycle` | 71 |

自动关联 267 条局部轨迹、150 个全局目标；人工版本 268 条局部轨迹、146 个全局目标。

### r39_20251021_070327

| 原始类别 | 数量 |
| --- | ---: |
| `car` | 46894 |
| `person` | 14962 |
| `motorcycle` | 5026 |
| `truck` | 1936 |
| `bicycle` | 531 |
| `bus` | 263 |

自动关联 238 条局部轨迹、141 个全局目标；人工版本 238 条局部轨迹、136 个全局目标。

### r39_20251021_070428

| 原始类别 | 数量 |
| --- | ---: |
| `car` | 56804 |
| `person` | 11676 |
| `motorcycle` | 3829 |
| `truck` | 1570 |
| `bus` | 45 |

自动关联 216 条局部轨迹、132 个全局目标；人工版本 216 条局部轨迹、125 个全局目标。

## 格式

类型 A 是逐帧像素框，字段包括 `camera_key`、`local_obj_id`、`global_obj_id`、`frame_index`、`class_name`、`bbox_xyxy` 和 `pixel_centroid`。`aligned_bbox_info.json` 把 `pixel_boxes` 与 `camera_boxes` 放在同一个文件，拆分后的像素文件和相机文件与它对照。

类型 B 是轨迹级全局关联。`multi_view_global_ids_manual.json` 含 `original_global_obj_id`、`is_deleted` 和 `manual_fix`。它没有逐帧框，逐帧框在各相机 `tracks.json`。

类型 C 是 `camera_box_3d`。对齐文件里的长宽远大于正常车辆，投影文件中的长度在抽样中为 0。两者都保留来源，但不写成已核实世界坐标。

类型 D 是现有流水线 `result.json`。本次原始目录里没有这种结果，适配器已能读取仓库中的示例结构。

`20250924_070957/detr_result` 和 `resnet_result` 是本地补充的检测与特征，不是甲方结构化数据。盘点时能看到这些文件，清洗时忽略，不写入 `data/normalized`。

## 时间与视频

原始视频均为 1920x1080、60 FPS、3600 帧、60 秒。抽帧元数据为步长 3、保存 1200 帧，对应 20 FPS 采样。可视化 `multi_view_global_ids.mp4` 是 10 FPS、1200 帧、120 秒，它是播放帧率，不是采样帧率。

R39_new 说明文字要求按 10 Hz 抽帧，`split_config.yaml` 写了 100 ms 间隔和 2025-04-22，但路径指向其他数据集。这两处与 ffprobe 和 `frame_cache_meta.json` 冲突，时间基准不采用它们。

四路相机的帧率、步长和帧数一致，因此采样模式一致。跨相机时钟同步没有单独验证，`sync_status` 不是已同步。

## 版本关系

`result/20250924_070957` 与 `result/20251021_070024` 各有一套对齐框。20250924 目录里的 DETR 和 ResNet 是本地补充，不纳入统一结果。`aligned_pixel_bbox.json`、`aligned_camera_bbox.json` 是合并文件的拆分，投影文件是派生结果。

R39_new 的 0、1、2、3 不是 20251021_070024 的重复处理。它们的原始序列目录分别是 `20251021_070125`、`070226`、`070327`、`070428`，视频文件大小也不同。每个目录里的自动全局 ID、人工全局 ID、文本映射和轨迹框属于同一段视频的不同版本。

## 未核实关系

- possible_consecutive_20251021（unverified）：`r39_20251021_070024`、`r39_20251021_070125`、`r39_20251021_070226`、`r39_20251021_070327`、`r39_20251021_070428`。目录时间大约每61秒一段，且每段原始视频时长60秒。没有独立时间戳或画面对齐证据，各段保持独立序列

`R39_new/multi_view_global_ids.mp4` 无法被 ffprobe 读取。`result/20251021_070024/raw_video` 下还有一份与子目录同名的视频拷贝，大小相同但不是同一 inode，清洗使用子目录中的文件。

掩码 `.npz` 和抽帧图片数量很大，本轮只统计数量，不把掩码或图片转成观测。未观测到目标，和原始文件里没有标注，不是同一件事；统一文件只记录原始文件中存在的框。
