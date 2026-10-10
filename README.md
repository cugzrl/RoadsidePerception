# 路侧端到端感知

镜像交付请参阅 [打包说明](docs/docker_deploy.md) 和 [接收方部署指南](packaging/README.md)。

流水线支持一到四路视频，固定按20FPS抽帧，依次完成RD-DETR检测、ResNet特征提取、STMC全局跟踪、HyperLPR3车牌识别、JSON导出和全局 BEV。R39 使用车道图底图，其他场景只画轨迹。

七个类别全部参与跟踪，只对Car、Van、Bus识别车牌

## 输出

指定`--output`后会生成以下内容

```text
输出目录/
├── result.json
├── visualization.mp4
├── bev.mp4
├── bev_map.png
├── manifest.json
├── detections.json
├── frames/
├── features/
└── tracking/
```

- `result.json`是最终保存算法结果的JSON文件，包含逐帧检测、全局轨迹和车牌结果
- `visualization.mp4`是多目拼接可视化视频，检测框标签包含类别、全局ID和车牌号
- `bev.mp4`和`bev_map.png`是 R39 车道级全局 BEV，轨迹使用`world_position`的地面坐标
- `manifest.json`记录输入视频、相机ID、标定文件、抽帧率和图片数量
- `frames`按相机保存20FPS抽出的图片
- `detections.json`保存RD-DETR原始检测结果，没有全局轨迹ID和车牌
- `features`保存每个检测框对应的ResNet特征和索引信息
- `tracking`保存STMC输入、中间文件和带全局轨迹ID的`tracking.json`

对于输出的结果查看`result.json`、`visualization.mp4`和`bev.mp4`，其余中间内容用于调试

### result.json字段

- `scene_id`表示本次任务名称
- `fps`表示结果时间轴的帧率，当前固定为20
- `feature_provider`表示特征提取方法，当前固定为`resnet50`
- `cameras`记录每路相机的视频、标定、分辨率和帧数
- `categories`记录七个类别的ID和名称
- `frames`保存每路相机每一帧的检测结果
- `tracks`保存跨帧、跨相机合并后的全局轨迹摘要
- `errors`保存非致命处理错误，正常运行时为空数组

`frames[].detections[]`中的主要字段如下

- `detection_id`是检测框唯一编号
- `class_id`和`class_name`是类别编号和名称
- `score`是检测置信度
- `bbox_xyxy`是图像坐标中的`[左,上,右,下]`
- `global_track_id`是STMC分配的全局目标ID，未成功关联时为`null`
- `world_position`是标定计算的路面坐标，无法计算时为`null`
- `plate`是该全局轨迹的车牌号，未识别或非车辆类别时为空字符串

`tracks[]`中的主要字段如下

- `global_track_id`是全局目标ID
- `class_id`和`class_name`是轨迹类别
- `plate`和`plate_confidence`是车牌号及置信度，未识别时分别为空字符串和`null`
- `start_timestamp_ms`和`end_timestamp_ms`是轨迹起止时间
- `observation_count`是该轨迹关联到的检测框数量
- `camera_observation_counts`是该轨迹在各相机中的检测框数量

## 单目示例

```bash
python scripts/run_pipeline.py \
  --videos /data/camera_1.mp4 \
  --camera-ids camera_1 \
  --calibrations /data/calibration_1.json \
  --output /output/result \
  --fps 20
```

## 四目示例

```bash
python scripts/run_pipeline.py \
  --videos /data/camera_1.mp4 /data/camera_2.mp4 /data/camera_3.mp4 /data/camera_4.mp4 \
  --camera-ids camera_1 camera_2 camera_3 camera_4 \
  --calibrations /data/calibration_1.json /data/calibration_2.json /data/calibration_3.json /data/calibration_4.json \
  --output /output/result \
  --fps 20
```

R39四目数据可直接运行命令示例：

```bash
cd /home/kemove/devdata1/RodeTest/roadside_perception
source /home/kemove/anaconda3/etc/profile.d/conda.sh
conda activate roadside_perception

python scripts/run_pipeline.py \
  --videos \
    /home/kemove/devdata1/R2/R39_new/0/原始视频/R39_Aw_CamS.mp4 \
    /home/kemove/devdata1/R2/R39_new/0/原始视频/R39_Bn_CamW.mp4 \
    /home/kemove/devdata1/R2/R39_new/0/原始视频/R39_Ce_CamN.mp4 \
    /home/kemove/devdata1/R2/R39_new/0/原始视频/R39_Ds_CamE.mp4 \
  --camera-ids R39_Aw_CamS R39_Bn_CamW R39_Ce_CamN R39_Ds_CamE \
  --calibrations \
    configs/cameras/r39/R39_Aw_CamS/calibration.json \
    configs/cameras/r39/R39_Bn_CamW/calibration.json \
    configs/cameras/r39/R39_Ce_CamN/calibration.json \
    configs/cameras/r39/R39_Ds_CamE/calibration.json \
  --fps 20 \
  --output outputs/r39_four_camera_test \
  --overwrite
```

完整视频运行时间较长，首次测试可以在末尾增加`--max-frames 200`

## 误检参数

默认对Pedestrian、Cyclist、ScooterRider采用更严格阈值

```bash
python scripts/run_pipeline.py \
  --videos /data/camera_1.mp4 \
  --camera-ids camera_1 \
  --calibrations /data/calibration_1.json \
  --output /output/result \
  --class-thresholds Pedestrian=0.82 Cyclist=0.78 ScooterRider=0.78 \
  --class-min-box-areas Pedestrian=500 Cyclist=450 ScooterRider=450
```

使用`--max-frames 200`可进行短序列验证，使用`--resume`可复用已完成阶段
