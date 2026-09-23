# 运行环境

统一Conda环境名为`roadside_perception`

```bash
conda activate roadside_perception
```

运行时通过当前环境的`python`启动流水线

检测、特征提取、跟踪、车牌识别和可视化均使用同一个Python解释器

RAMA需要在环境创建后从本地源码编译

```bash
pip install --no-build-isolation ./third_party/RAMA
```

系统还需要提供`ffmpeg`、`ffprobe`、`cmake`、`gcc`和`g++`

## 主要版本

| 组件 | 版本 |
|---|---|
| Python | 3.9.25 |
| PyTorch | 2.0.1+cu117 |
| TorchVision | 0.15.2+cu117 |
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |
| OpenCV | 4.11.0.86 |
| ONNX | 1.14.0 |
| ONNXRuntime | 1.15.1 |
| Transformers | 4.30.2 |
| Hydra | 1.3.2 |
| Motmetrics | 1.4.0 |
| HyperLPR3依赖Pillow | 11.3.0 |
| RAMA | 0.0.8 |
| JSONSchema | 4.25.1 |

## 系统组件

| 组件 | 版本 |
|---|---|
| NVIDIA驱动 | 580.178.04 |
| GPU | 2张RTX 4090 24GB |
| FFmpeg | 4.4.2 |
| CMake | 3.22.1 |
| GCC | 9.5.0 |

完整Python直接依赖记录在`requirements.pipeline.txt`
