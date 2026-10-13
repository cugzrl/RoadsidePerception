# STMC全局跟踪

此目录仅保留路侧端到端流水线需要的STMC
正式入口由项目根目录下的`scripts/run_tracking.py`调用

## 配置

```text
conf/
├── config.yaml
├── dataset/R39.yaml
├── encoder/precomputed.yaml
└── experiment/R39.yaml
```

- `config.yaml`保存STMC基础参数
- `dataset/R39.yaml`使用通用相机目录匹配并支持1～4目
- `encoder/precomputed.yaml`读取ResNet50预计算特征
- `experiment/R39.yaml`保存路侧全局跟踪参数

## 输入输出

流水线会自动生成STMC所需的图片、标定、检测框和特征文件

STMC输出的轨迹由`scripts/run_tracking.py`整理为统一的`tracking.json`，随后进入车牌识别和最终JSON导出阶段
