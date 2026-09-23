# 来源清单

| 内容 | 集成位置 | 用途 |
|---|---|---|
| RD-DETR | `algorithms/rddetr` | 七类目标检测 |
| 检测权重 | `models/detection/detection_R39.pth` | R39检测 |
| ResNet50 | `scripts/run_features.py`、`models/reid/resnet50-0676ba61.pth` | Torchvision模型与本地权重 |
| ResNet50权重 | `models/reid/resnet50-0676ba61.pth` | ImageNet预训练 |
| STMC | `algorithms/stmc` | 全局多目标跟踪 |
| HyperLPR3 | `third_party/HyperLPR` | 车牌识别 |
| R39标定 | `configs/cameras/r39` | 世界坐标投影 |

原始视频和数据集不复制到工程目录
