#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
command -v docker >/dev/null || { echo "请先安装 Docker" >&2; exit 1; }
docker info >/dev/null
echo "校验交付文件（大文件可能需要数分钟）……"
sha256sum -c SHA256SUMS
IFS= read -r image < IMAGE_NAME
IFS= read -r expected < IMAGE_ID
gzip -dc image.tar.gz | docker load
actual=$(docker image inspect "$image" --format '{{.Id}}')
[[ "$actual" == "$expected" ]] || { echo "镜像 ID 与交付记录不一致" >&2; exit 1; }
echo "镜像加载完成: $image ($actual)"
echo "下一步: ./check_env.sh，或 ./check_env.sh <视频目录> <输出目录>"
