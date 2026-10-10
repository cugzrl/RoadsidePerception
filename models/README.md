# 模型权重

| 路径 | 用途 |
|---|---|
| `detection/detection_R39.pth` | RD-DETR R39 检测权重 |
| `reid/resnet50-0676ba61.pth` | ResNet50 特征提取权重 |
| `license_plate/hyperlpr3/20230229/onnx/*.onnx` | HyperLPR3 车牌 ONNX 模型（4 个） |

目录结构与 `SOURCE_MANIFEST.md` 一致。Docker 构建与本地流水线均默认从上述路径读取。