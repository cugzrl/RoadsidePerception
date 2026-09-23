# 镜像打包目录

```text
roadside_perception/
├── algorithms/
│   ├── rddetr/
│   └── stmc/
├── configs/
│   └── cameras/
├── models/
│   ├── detection/
│   │   └── detection_R39.pth
│   ├── reid/
│   │   └── resnet50-0676ba61.pth
│   └── license_plate/
│       └── hyperlpr3/20230229/onnx/
├── pipeline/
├── scripts/
├── third_party/
│   ├── HyperLPR/
│   │   ├── Prj-Python/hyperlpr3/
│   │   └── resource/font/platech.ttf
│   └── RAMA/
├── docs/
├── examples/
├── README.md
├── environment.yml
└── requirements.pipeline.txt
```

`outputs`、`__pycache__`、`.git`、测试图片、其他平台源码和构建缓存不参与镜像打包

RAMA源码只在镜像构建阶段使用，运行阶段只保留编译后的`rama_py`扩展
