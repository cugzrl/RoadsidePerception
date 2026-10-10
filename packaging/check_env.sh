#!/usr/bin/env bash
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo "用法: $0 [视频目录 输出目录]"
    echo "支持 ROAD_IMAGE、ROAD_GPU、ROAD_SHM_SIZE、ROAD_CALIBRATION_DIR；不会运行完整算法。"
    exit 0
fi
if (( $# != 0 && $# != 2 )); then echo "用法: $0 [视频目录 输出目录]" >&2; exit 2; fi
image=${ROAD_IMAGE:-roadside-perception:20260923}
gpu=${ROAD_GPU:-0}
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "ROAD_GPU 必须为单个 GPU 编号" >&2; exit 2; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo "本交付支持 Linux x86_64" >&2; exit 1; }
command -v docker >/dev/null || { echo "请先安装 Docker" >&2; exit 1; }
docker info >/dev/null
docker image inspect "$image" --format '镜像: {{.Id}} 平台: {{.Os}}/{{.Architecture}}'
args=(run --rm --init --network none --gpus "device=$gpu" --shm-size="${ROAD_SHM_SIZE:-8g}" --user "$(id -u):$(id -g)" -e HOME=/tmp)
args+=(-v "$base/check_runtime.py:/tmp/check_runtime.py:ro")
if (( $# == 2 )); then
    input=$(realpath -e "$1")
    [[ -d "$input" ]] || { echo "视频目录无效" >&2; exit 2; }
    mkdir -p -- "$2"
    output=$(realpath -e "$2")
    [[ "$input" != "$output" && "$input$output" != *:* ]] || { echo "输入输出目录相同或包含冒号" >&2; exit 2; }
    args+=(-v "$input:/data/input:ro" -v "$output:/data/output")
fi
if [[ -n "${ROAD_CALIBRATION_DIR:-}" ]]; then
    calibration=$(realpath -e "$ROAD_CALIBRATION_DIR")
    [[ -d "$calibration" && "$calibration" != *:* ]] || { echo "标定目录无效" >&2; exit 2; }
    args+=(-v "$calibration:/data/calibrations:ro")
fi
docker "${args[@]}" --entrypoint python "$image" /tmp/check_runtime.py
echo "环境自检通过；仍需执行短视频验收确认实际算法输出。"
