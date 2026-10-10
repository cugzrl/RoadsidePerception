#!/usr/bin/env bash
set -euo pipefail

IMAGE="${ROAD_IMAGE:-roadside-perception:20260923}"
GPU="${ROAD_GPU:-0}"
SHM_SIZE="${ROAD_SHM_SIZE:-8g}"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo "用法: $0 <视频目录> <输出目录> [流水线参数]"
    echo "环境变量: ROAD_IMAGE, ROAD_GPU (单个宿主机GPU编号，默认0), ROAD_SHM_SIZE (默认8g), ROAD_CALIBRATION_DIR"
    exit 0
fi
if (( $# < 2 )); then
    echo "用法: $0 <视频目录> <输出目录> [流水线参数]" >&2
    echo "例如: $0 /data/videos /data/result --max-frames 200 --overwrite" >&2
    exit 2
fi
command -v docker >/dev/null || { echo "未安装 Docker" >&2; exit 1; }
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "ROAD_GPU 必须为单个 GPU 编号" >&2; exit 2; }
docker image inspect "$IMAGE" >/dev/null || { echo "请先加载镜像 $IMAGE" >&2; exit 1; }

input_dir=$(realpath -e "$1")
if [[ ! -d "$input_dir" ]]; then
    echo "视频目录不存在: $input_dir" >&2
    exit 2
fi
mkdir -p "$2"
output_dir=$(realpath -e "$2")
[[ -w "$output_dir" ]] || { echo "输出目录不可写: $output_dir" >&2; exit 2; }
if [[ "$input_dir" == "$output_dir" ]]; then
    echo "输入和输出目录不能相同" >&2
    exit 2
fi
if [[ "$input_dir$output_dir" == *:* ]]; then
    echo "挂载目录名不能包含冒号" >&2
    exit 2
fi
shift 2

docker_args=(
    run --rm --init --network none --gpus "device=$GPU" --shm-size="$SHM_SIZE"
    --user "$(id -u):$(id -g)"
    -e HOME=/tmp
    -v "$input_dir:/data/input:ro"
    -v "$output_dir:/data/output"
)
pipeline_args=(--input-dir /data/input --output /data/output)
if [[ -n "${ROAD_CALIBRATION_DIR:-}" ]]; then
    calibration_dir=$(realpath -e "$ROAD_CALIBRATION_DIR")
    [[ -d "$calibration_dir" && "$calibration_dir" != *:* ]] || { echo "标定目录无效" >&2; exit 2; }
    docker_args+=(-v "$calibration_dir:/data/calibrations:ro")
    pipeline_args+=(--calibration-dir /data/calibrations)
fi

exec docker "${docker_args[@]}" "$IMAGE" "${pipeline_args[@]}" "$@"
