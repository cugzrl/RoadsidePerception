# 时空知识库

建库序列：r39_20250924_070957, r39_20251021_070024, r39_20251021_070125, r39_20251021_070226。
留出序列：r39_20251021_070327, r39_20251021_070428。留出序列不进入知识库，避免后续感知实验直接看到这段历史。

网格 2.0 m，范围 X [-60, 60]、Y [-60, 60]。坐标系 `road_homography_xy`。

## 规模

- 进入网格的观测 138759
- 局部轨迹 661，未拆分 629，自动拆分 32，含跳变 72
- 有观测的网格 1868
- 区域转移 5773
- 典型路径 48，类别计数 {'straight': 23, 'left': 6, 'right': 7, 'other': 12}
- 重叠视野共同观测的相机对 6，非重叠时序转移的有向相机对 7

观测次数和轨迹数分开保存。停留多帧不会单独抬高轨迹数。没有用检测框数量估计漏检或误检。相机覆盖不等于观测可靠性，可靠性字段为 unknown。

## 典型路径支持数最高的几条

| ID | 类型 | 轨迹数 | 可信度 | 中位速度 m/s | 示例相机 |
| --- | --- | ---: | --- | ---: | --- |
| 0 | straight | 14 | high | 6.752502639985703 | R39_Aw_CamS |
| 1 | right | 9 | medium | 6.459802917111512 | R39_Ds_CamE |
| 2 | right | 9 | medium | 8.116847312270615 | R39_Ce_CamN |
| 3 | other | 8 | medium | 4.95979656966726 | R39_Ce_CamN |
| 4 | straight | 8 | medium | 6.927748583017694 | R39_Ds_CamE |
| 5 | straight | 8 | medium | 7.926504480003562 | R39_Aw_CamS |
| 6 | other | 7 | medium | 11.408874753558896 | R39_Aw_CamS |
| 7 | other | 7 | medium | 8.292047465050347 | R39_Ds_CamE |

## 跨相机关系

| 相机 A | 相机 B | 关系 | 支持数 | 中位距离 m | 中位时间 ms | 可信度 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| R39_Aw_CamS | R39_Bn_CamW | co_observation | 8610 | 3.118651992622664 | None | high |
| R39_Aw_CamS | R39_Ce_CamN | co_observation | 3414 | 2.180864161236534 | None | high |
| R39_Aw_CamS | R39_Ds_CamE | co_observation | 5520 | 2.043459087319736 | None | high |
| R39_Bn_CamW | R39_Ce_CamN | co_observation | 8114 | 1.935711168786883 | None | high |
| R39_Bn_CamW | R39_Ds_CamE | co_observation | 13470 | 1.915308348107451 | None | high |
| R39_Ce_CamN | R39_Ds_CamE | co_observation | 12933 | 2.4225584739018236 | None | high |
| R39_Aw_CamS | R39_Bn_CamW | temporal_transfer | 1 | None | 1050.0 | low |
| R39_Aw_CamS | R39_Ce_CamN | temporal_transfer | 8 | None | 950.0 | medium |
| R39_Bn_CamW | R39_Aw_CamS | temporal_transfer | 2 | None | 1000.0 | low |
| R39_Bn_CamW | R39_Ds_CamE | temporal_transfer | 1 | None | 1400.0 | low |
| R39_Ce_CamN | R39_Ds_CamE | temporal_transfer | 3 | None | 1550.0 | medium |
| R39_Ds_CamE | R39_Bn_CamW | temporal_transfer | 1 | None | 1500.0 | low |
| R39_Ds_CamE | R39_Ce_CamN | temporal_transfer | 3 | None | 650.0 | medium |
