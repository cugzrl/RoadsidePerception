# Docker 打包与部署

沿用 roadside-perception:20260923。本次仅补充启动脚本、交付工具和文档，不修改算法或重建已验证镜像。

接收方说明见 [交付 README](../packaging/README.md)。整个交付目录一起发送，不需要开发源码或 Conda 环境归档。

## 从已有归档生成交付目录

项目根目录执行（需要 Python 3.9+、Docker、可用 NVIDIA GPU、gzip）：

```bash
python3 packaging/create_release.py \
  --archive ../roadside-perception_20260923.tar.gz \
  --output ../roadside-perception-release-20260923
```

不传 --archive 则 docker save 后 gzip -1 导出；--image 可选择其他已加载标签。输出目录必须不存在。脚本检查平台、执行断网 GPU 自检、验证归档标签和镜像 ID、采集镜像内依赖及源码/模型指纹、生成 SHA256SUMS。全部成功后才生成最终目录，不覆盖原归档或重建算法。

BUILD_INFO.txt 的 Git 信息描述打包工作区，不代表旧镜像源码提交。镜像内容以 IMAGE_ID 和 IMAGE_CONTENTS.json 为准。

## 验收

```bash
cd ../roadside-perception-release-20260923
./load_image.sh
./check_env.sh /data/videos /data/results-check
./run_container.sh /data/videos /data/results-smoke --max-frames 20
```

检查 result.json 和 visualization.mp4；正式验收另需清晰车牌、多目关联和长视频，自检不替代算法准确率验收。

## 确需重建算法镜像时

Dockerfile 仍通过多阶段构建解包现有 Conda 环境，不在最终镜像保留环境归档。以下仅在需要更新镜像时执行，事先准备权重和已编译 RAMA，使用新标签：

```bash
conda pack -n roadside_perception -o roadside_perception-env.tar.gz
docker build -t roadside-perception:NEW_VERSION .
rm roadside_perception-env.tar.gz
python3 packaging/create_release.py \
  --image roadside-perception:NEW_VERSION \
  --output ../roadside-perception-release-NEW_VERSION
```

不要跳过缺失文件检查以代替完整环境验证。新镜像还需真实视频测试。CUDA 基础镜像、依赖版本和 RAMA 编译配置保持原样，瘦身及兼容范围扩展应单独验证。

## 合并为最终交付压缩包

部署指南修改后，同步到交付目录的 README.md，重新生成校验清单，再压缩整个目录（文件名沿用当前交付版本）：

```bash
cd ../roadside-perception-release-20260923
find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\0' | sort -z | xargs -0 sha256sum > SHA256SUMS
cd ..
tar -czf roadside-perception-delivery-20260923.tar.gz roadside-perception-release-20260923
```

外层文件是交付压缩包，内层 image.tar.gz 是 Docker 镜像归档。对外只发送 roadside-perception-delivery-20260923.tar.gz。首次发布前应解压校验，或逐项验证压缩包内文件与 SHA256SUMS 一致。
