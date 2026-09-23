# 接口约定

## 统一约定

- 抽帧率固定为20FPS
- `frame_index`从0开始
- 图片文件名从`000001.jpg`开始
- 检测框格式为`[x1,y1,x2,y2]`
- 七个类别全部参与跟踪
- STMC只允许同类别目标关联
- 仅对Car、Van和Bus识别车牌
- 车牌识别在全局跟踪后执行
- 未识别到车牌时填写空字符串

## 抽帧接口

输入为一到四个视频、相机ID和标定文件

输出为`manifest.json`和`frames/<camera_id>/%06d.jpg`

## 检测接口

输入为`manifest.json`

每个检测目标包含以下字段

```text
detection_id
class_id
class_name
score
bbox_xyxy
```

## 特征接口

输入为检测结果和原始图片

输出为`features.npy`和`feature_metadata.json`

使用ResNet50输出2048维L2归一化特征

## 跟踪接口

检测通过`detection_id`与特征关联

STMC输入使用内部MOT文本格式，不作为公开接口

跟踪结果为每个检测补充以下字段

```text
global_track_id
world_position
plate
```

## 最终JSON

顶层仅保留以下字段

```text
scene_id
fps
feature_provider
cameras
categories
frames
tracks
errors
```

不再输出`available`、`schema_version`和`run`
