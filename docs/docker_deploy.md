# Docker 打包与部署

镜像名：`roadside-perception:20260923`。镜像包含检测、跟踪、车牌识别、可视化所需的源码、权重、标定文件和已编译的 RAMA 扩展。容器内没有 RAMA 源码、原始视频或历史输出。

## 在本机运行

准备一个目录，直接放入 1 到 4 个视频文件。支持 `.mp4`、`.mov`、`.mkv`、`.avi`、`.m4v`。文件名（不含扩展名）默认作为相机 ID；内置 R39 标定文件按相机 ID 自动匹配。

```bash
cd /home/kemove/devdata1/RodeTest/roadside_perception
./run_container.sh /path/to/videos /path/to/output --overwrite
```

短序列验证：

```bash
./run_container.sh /path/to/videos /path/to/output --max-frames 2 --overwrite
```

如需使用其他标定根目录，目录中放置 `<相机ID>/calibration.json` 或 `<相机ID>.json`，然后运行：

```bash
ROAD_CALIBRATION_DIR=/path/to/calibrations \
  ./run_container.sh /path/to/videos /path/to/output --overwrite
```

其余参数（例如 `--scene-id`、`--resume`、`--device`、`--det-threshold`）会传给原有 `scripts/run_pipeline.py`。视频按文件名字母顺序处理；若文件名与相机 ID 不同，可追加 `--camera-ids ID1 ID2`。输出目录中最终使用 `result.json` 和 `visualization.mp4`。

## 从镜像归档载入

```bash
gzip -dc /home/kemove/devdata1/RodeTest/roadside-perception_20260923.tar.gz | docker load
```

服务器需要 NVIDIA 驱动、NVIDIA Container Toolkit 和 Docker GPU 支持。运行脚本会挂载视频目录为只读、挂载输出目录为可写，并以当前用户身份写入结果。

## 重建镜像

在已有 `roadside_perception` Conda 环境的服务器上：

```bash
cd /home/kemove/devdata1/RodeTest
conda pack -n roadside_perception -o roadside_perception-env.tar.gz --ignore-missing-files
ln roadside_perception-env.tar.gz roadside_perception/roadside_perception-env.tar.gz
docker build -t roadside-perception:20260923 roadside_perception
rm roadside_perception/roadside_perception-env.tar.gz
docker save roadside-perception:20260923 | gzip -1 > roadside-perception_20260923.tar.gz
```

`Dockerfile` 通过多阶段构建展开环境，运行镜像里不会保留压缩归档。建议重新构建后用短视频实际运行一次，确认 `result.json` 与 `visualization.mp4` 均生成。
